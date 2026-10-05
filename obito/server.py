"""Lokaler HTTP-Server von OBITO (API + HUD-Oberfläche).

Nur Standardbibliothek. Der Server bindet standardmäßig an ``127.0.0.1`` und ist gegen
Angriffe aus dem Browser gehärtet:

* ``Host``- und ``Origin``-Prüfung (DNS-Rebinding, fremde Seiten) → 403
* ``POST``/``DELETE`` nur mit ``Content-Type: application/json`` → sonst 415
* Körper größer als ``max_body`` → 413
* keine CORS-Header, ``GET /`` liefert ausschließlich ``static/index.html``
* unbehandelte Fehler → 500 ``{"ok": false, "fehler": …}`` + Traceback in ``logs/server.log``

Alle Antworten sind JSON-Objekte der Form ``{"ok": bool, …}``; die Frage-Schnittstelle kann
alternativ als ``text/event-stream`` (Server-Sent Events, chunked) streamen.
"""

from __future__ import annotations

import datetime as _dt
import ipaddress
import json
import re
import socket
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from .memory import KINDS, Memory

__all__ = ["ObitoServer", "ObitoHandler", "INDEX_PATH", "DEPTHS", "memory_to_dict"]

INDEX_PATH = Path(__file__).resolve().parent / "static" / "index.html"
DEPTHS = ("auto", "schnell", "mittel", "tief")
_LOOPBACK_NAMES = {"localhost", "127.0.0.1", "::1", "[::1]"}
_MAX_LIST = 500

_PLACEHOLDER_HTML = """<!doctype html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>OBITO</title>
<style>
body{background:#0b0f14;color:#d7e1ea;font-family:system-ui,sans-serif;margin:0;
display:flex;align-items:center;justify-content:center;min-height:100vh}
main{max-width:40rem;padding:2rem;border:1px solid #243041;border-radius:8px;background:#111823}
code{background:#1a2433;padding:.1rem .3rem;border-radius:3px}
</style>
</head>
<body>
<main>
<h1>OBITO läuft</h1>
<p>Die Oberfläche <code>obito/static/index.html</code> wurde nicht gefunden.
Die API ist erreichbar, zum Beispiel unter <a href="/api/status" style="color:#7fd1ff">/api/status</a>.</p>
</main>
</body>
</html>
"""


# ------------------------------------------------------------ Serialisierung
def _fallback_memory_to_dict(m: Memory) -> dict:
    """Spiegelbild von ``brain.memory_to_dict`` – ohne Embedding."""
    created = _dt.datetime.fromtimestamp(m.created_at).isoformat(timespec="seconds")
    return {
        "id": m.id, "art": m.kind, "inhalt": m.content, "tags": list(m.tags), "projekt": m.project,
        "quelle": m.source, "wichtigkeit": m.importance, "erstellt": created, "score": m.score,
    }


def memory_to_dict(m: Memory) -> dict:
    """Serialisiert eine Erinnerung. Nutzt ``obito.brain.memory_to_dict``, sobald der Denkkern
    vorhanden ist (lazy importiert), sonst die lokale, formatgleiche Variante."""
    try:
        from .brain import memory_to_dict as _impl  # lazy: brain.py ist optional zur Importzeit
    except Exception:  # noqa: BLE001 – fehlendes/kaputtes Modul darf den Server nicht stoppen
        return _fallback_memory_to_dict(m)
    return _impl(m)


def _dumps(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8", errors="replace")


class _HttpError(Exception):
    """Interner Kurzschluss: HTTP-Fehler mit Statuscode und deutscher Meldung."""

    def __init__(self, status: int, message: str, headers: dict[str, str] | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.headers = headers or {}


def _split_hostport(value: str) -> tuple[str, int | None] | None:
    """Zerlegt ``host[:port]`` (auch ``[::1]:8765``); ``None`` bei unbrauchbarem Wert."""
    v = (value or "").strip().lower()
    if not v:
        return None
    if v.startswith("["):
        end = v.find("]")
        if end == -1:
            return None
        host, rest = v[:end + 1], v[end + 1:]
        if not rest:
            return host, None
        if rest.startswith(":") and rest[1:].isdigit():
            return host, int(rest[1:])
        return None
    if v.count(":") == 1:
        host, port = v.split(":")
        if not host or not port.isdigit():
            return None
        return host, int(port)
    if ":" in v:          # nackte IPv6-Adresse ohne Klammern ist in Host/Origin nicht erlaubt
        return None
    return v, None


def _is_loopback(host: str) -> bool:
    h = host.strip().lower().strip("[]")
    if h in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


# -------------------------------------------------------------------- Server
class ObitoServer(ThreadingHTTPServer):
    """HTTP-Server für die HUD-Oberfläche und die lokale API.

    ``port=0`` wählt einen freien Port (Tests); der tatsächliche Port steht in ``self.port``.
    ``allow_dangerous`` steuert, ob gefährliche Werkzeuge (Dateien schreiben, Code/Befehle
    ausführen) über die API freigegeben sind – im Server gibt es keine Rückfrage, deshalb
    ist der Standard ``False``.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, brain, host: str = "127.0.0.1", port: int = 8765, *,
                 allow_dangerous: bool = False, max_body: int = 1_048_576):
        self.brain = brain
        self.cfg = brain.cfg
        self.host = host
        self.allow_dangerous = bool(allow_dangerous)
        self.max_body = int(max_body)
        self.index_path: Path = INDEX_PATH
        self._log_lock = threading.Lock()
        self.logs_dir = Path(self.cfg.logs_dir)
        self.log_path = self.logs_dir / "server.log"
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"WARNUNG: Log-Verzeichnis {self.logs_dir} nicht anlegbar: {e}", file=sys.stderr)

        if ":" in host:
            self.address_family = socket.AF_INET6
        super().__init__((host, port), ObitoHandler)
        self.port: int = int(self.server_address[1])

        allowed = {"127.0.0.1", "localhost", "[::1]"}
        h = host.strip().lower()
        allowed.add(f"[{h}]" if ":" in h and not h.startswith("[") else h)
        # Wildcard-Bindung: Anfragen aus dem LAN tragen die LAN-IP bzw. den Rechnernamen im Host-Header
        self.wildcard = h in ("", "0.0.0.0", "::", "[::]")
        if self.wildcard:
            try:
                name = socket.gethostname()
                allowed.add(name.lower())
                for info in socket.getaddrinfo(name, None):
                    ip = info[4][0]
                    allowed.add(f"[{ip}]" if ":" in ip else ip)
            except OSError:
                pass
        self.allowed_hosts: frozenset[str] = frozenset(allowed)

        # Werkzeug-Freigabe: ohne Terminal keine Rückfrage möglich -> pauschal ja/nein.
        tools = getattr(brain, "tools", None)
        if tools is not None:
            flag = self.allow_dangerous
            tools.set_policy(confirm=lambda name, args: flag, confirm_dangerous=True)
        set_policy = getattr(brain, "set_dangerous_policy", None)
        if callable(set_policy):
            set_policy(self.allow_dangerous)
            if tools is not None:
                tools.set_policy(confirm=lambda name, args: flag, confirm_dangerous=True)

        if not _is_loopback(host):
            print(
                f"WARNUNG: OBITO-Server lauscht auf {host}:{self.port} und ist damit nicht nur lokal "
                "erreichbar. Es gibt keine Anmeldung – nur in vertrauenswürdigen Netzen verwenden"
                + (" (gefährliche Werkzeuge sind freigegeben!)." if self.allow_dangerous else "."),
                file=sys.stderr,
            )
        self.log(f"Server gestartet auf {host}:{self.port} (gefaehrliche Werkzeuge: "
                 f"{'erlaubt' if self.allow_dangerous else 'gesperrt'})")

    # ------------------------------------------------------ Hintergrund
    def start_background(self) -> bool:
        """Startet den Automations-Zeitplaner des Denkkerns (falls vorhanden). ``True`` wenn gestartet."""
        scheduler = getattr(self.brain, "automation", None)
        start = getattr(scheduler, "start", None)
        if not callable(start):
            return False
        try:
            start()
            self.log("Automations-Zeitplaner gestartet")
            return True
        except Exception as e:  # noqa: BLE001
            self.log_exception("Zeitplaner starten", e)
            return False

    def stop_background(self) -> None:
        scheduler = getattr(self.brain, "automation", None)
        stop = getattr(scheduler, "stop", None)
        if callable(stop):
            try:
                stop()
            except Exception as e:  # noqa: BLE001
                self.log_exception("Zeitplaner stoppen", e)

    def server_close(self) -> None:
        self.stop_background()
        super().server_close()

    # ------------------------------------------------------------- Protokoll
    def log(self, text: str) -> None:
        """Hängt eine Zeile mit Zeitstempel an ``logs/server.log`` an (Fehler werden ignoriert)."""
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{stamp} {text.rstrip()}\n"
        with self._log_lock:
            try:
                with open(self.log_path, "a", encoding="utf-8") as fh:
                    fh.write(line)
            except OSError:
                pass

    def log_exception(self, where: str, exc: BaseException) -> None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        self.log(f"FEHLER in {where}: {exc!r}\n{tb}")

    def host_allowed(self, value: str | None) -> bool:
        parsed = _split_hostport(value or "")
        if parsed is None:
            return False
        host, port = parsed
        if host not in self.allowed_hosts:
            if not getattr(self, "wildcard", False):
                return False
            # IP-Literale können nicht per DNS-Rebinding untergeschoben werden
            try:
                ipaddress.ip_address(host.strip("[]"))
            except ValueError:
                return False
        return port is None or port == self.port

    def origin_allowed(self, origin: str) -> bool:
        parts = urlsplit(origin.strip())
        if parts.scheme not in ("http", "https") or not parts.netloc or "@" in parts.netloc:
            return False
        return self.host_allowed(parts.netloc)

    def handle_error(self, request, client_address) -> None:  # noqa: D401 – Signatur der Basisklasse
        """Fehler außerhalb des Handlers (z. B. abgebrochene Verbindungen) nur protokollieren."""
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, socket.timeout)):
            return
        if exc is not None:
            self.log_exception(f"Verbindung {client_address}", exc)


# ------------------------------------------------------------------- Handler
_Route = tuple[str, "re.Pattern[str]", str]

_ROUTES: tuple[_Route, ...] = (
    ("GET", re.compile(r"^/$"), "route_index"),
    ("GET", re.compile(r"^/api/status$"), "route_status"),
    ("POST", re.compile(r"^/api/frage$"), "route_frage"),
    ("POST", re.compile(r"^/api/feedback$"), "route_feedback"),
    ("GET", re.compile(r"^/api/erinnerungen$"), "route_erinnerungen_get"),
    ("POST", re.compile(r"^/api/erinnerungen$"), "route_erinnerungen_post"),
    ("DELETE", re.compile(r"^/api/erinnerungen/(\d+)$"), "route_erinnerungen_delete"),
    ("GET", re.compile(r"^/api/lektionen$"), "route_lektionen_get"),
    ("DELETE", re.compile(r"^/api/lektionen/(\d+)$"), "route_lektionen_delete"),
    ("GET", re.compile(r"^/api/modelle$"), "route_modelle"),
    # ---- Phase 2
    ("POST", re.compile(r"^/api/modell$"), "route_modell_post"),
    ("POST", re.compile(r"^/api/modelle/pull$"), "route_modelle_pull"),
    ("DELETE", re.compile(r"^/api/modelle/([^/]+)$"), "route_modelle_delete"),
    ("GET", re.compile(r"^/api/dokumente$"), "route_dokumente_get"),
    ("POST", re.compile(r"^/api/dokumente$"), "route_dokumente_post"),
    ("GET", re.compile(r"^/api/dokumente/suche$"), "route_dokumente_suche"),
    ("POST", re.compile(r"^/api/dokumente/sync$"), "route_dokumente_sync"),
    ("DELETE", re.compile(r"^/api/dokumente/(\d+)$"), "route_dokumente_delete"),
    ("GET", re.compile(r"^/api/projekte$"), "route_projekte_get"),
    ("POST", re.compile(r"^/api/projekte$"), "route_projekte_post"),
    ("GET", re.compile(r"^/api/projekte/([^/]+)$"), "route_projekt_get"),
    ("DELETE", re.compile(r"^/api/projekte/([^/]+)$"), "route_projekt_delete"),
    ("POST", re.compile(r"^/api/projekte/([^/]+)/notizen$"), "route_projekt_notiz_post"),
    ("POST", re.compile(r"^/api/notizen/(\d+)/erledigt$"), "route_notiz_erledigt"),
    ("DELETE", re.compile(r"^/api/notizen/(\d+)$"), "route_notiz_delete"),
    ("GET", re.compile(r"^/api/missionen$"), "route_missionen_get"),
    ("POST", re.compile(r"^/api/missionen$"), "route_missionen_post"),
    ("GET", re.compile(r"^/api/missionen/(\d+)$"), "route_mission_get"),
    ("POST", re.compile(r"^/api/missionen/(\d+)/start$"), "route_mission_start"),
    ("POST", re.compile(r"^/api/missionen/(\d+)/stop$"), "route_mission_stop"),
    ("DELETE", re.compile(r"^/api/missionen/(\d+)$"), "route_mission_delete"),
    ("GET", re.compile(r"^/api/automationen$"), "route_automationen_get"),
    ("POST", re.compile(r"^/api/automationen$"), "route_automationen_post"),
    ("GET", re.compile(r"^/api/automationen/vorschlaege$"), "route_automationen_vorschlaege"),
    ("POST", re.compile(r"^/api/automationen/vorschlaege$"), "route_automationen_vorschlaege_post"),
    ("POST", re.compile(r"^/api/automationen/(\d+)/jetzt$"), "route_automation_jetzt"),
    ("POST", re.compile(r"^/api/automationen/(\d+)/aktiv$"), "route_automation_aktiv"),
    ("GET", re.compile(r"^/api/automationen/(\d+)/laeufe$"), "route_automation_laeufe"),
    ("DELETE", re.compile(r"^/api/automationen/(\d+)$"), "route_automation_delete"),
    ("POST", re.compile(r"^/api/pflege/konsolidieren$"), "route_konsolidieren"),
    ("POST", re.compile(r"^/api/pflege/backup$"), "route_backup"),
)


class ObitoHandler(BaseHTTPRequestHandler):
    """Bearbeitet eine Anfrage; ``self.server`` ist der :class:`ObitoServer`."""

    server: ObitoServer
    server_version = "OBITO/1.0"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    timeout = 300.0
    error_content_type = "application/json; charset=utf-8"

    # ------------------------------------------------------------- Protokoll
    def log_message(self, format: str, *args) -> None:  # noqa: A002 – Name aus der Basisklasse
        try:
            text = format % args
        except Exception:  # noqa: BLE001
            text = f"{format} {args}"
        self.server.log(f"{self.address_string()} {text}")

    # ------------------------------------------------------------ Antworten
    def _send_bytes(self, status: int, body: bytes, content_type: str,
                    extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: Any, extra: dict[str, str] | None = None) -> None:
        self._send_bytes(status, _dumps(payload), "application/json; charset=utf-8", extra)

    def _fail(self, status: int, message: str, extra: dict[str, str] | None = None) -> None:
        self._send_json(status, {"ok": False, "fehler": message}, extra)

    def send_error(self, code, message=None, explain=None):  # noqa: D401 – Basisklasse (Parser-Fehler)
        """JSON statt HTML für Fehler, die schon beim Einlesen der Anfrage entstehen."""
        try:
            short, _long = self.responses[code]
        except (KeyError, TypeError):
            short = "Fehler"
        self.close_connection = True
        text = message or short
        body = b""
        if code >= 200 and code not in (HTTPStatus.NO_CONTENT, HTTPStatus.RESET_CONTENT, HTTPStatus.NOT_MODIFIED):
            body = _dumps({"ok": False, "fehler": text})
        try:
            self.log_error("code %d, message %s", code, text)
            self.send_response(code, short)
            self.send_header("Connection", "close")
            self.send_header("Content-Type", self.error_content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body and getattr(self, "command", None) != "HEAD":
                self.wfile.write(body)
        except OSError:
            pass

    # ------------------------------------------------------------- Einstieg
    def do_GET(self) -> None:
        self._dispatch()

    def do_POST(self) -> None:
        self._dispatch()

    def do_DELETE(self) -> None:
        self._dispatch()

    def _dispatch(self) -> None:
        self._streaming = False
        try:
            self._check_headers()
            path = urlsplit(self.path).path or "/"
            self.query = parse_qs(urlsplit(self.path).query, keep_blank_values=True)
            handler_name, match, allowed = None, None, []
            for method, pattern, name in _ROUTES:
                m = pattern.match(path)
                if not m:
                    continue
                allowed.append(method)
                if method == self.command:
                    handler_name, match = name, m
            if not allowed or handler_name is None or match is None:
                # Körper auch bei 404/405 verbrauchen, sonst wird er auf einer Keep-Alive-Verbindung
                # als nächste Anfrage gelesen (Request-Smuggling)
                self._discard_body()
            if not allowed:
                raise _HttpError(404, f"Nicht gefunden: {path}")
            if handler_name is None or match is None:
                raise _HttpError(405, f"Methode {self.command} für {path} nicht erlaubt",
                                 {"Allow": ", ".join(dict.fromkeys(allowed))})
            self.body = self._read_body()
            getattr(self, handler_name)(*match.groups())
        except _HttpError as e:
            if not getattr(self, "_body_consumed", False) and int(self.headers.get("Content-Length") or 0) > 0:
                self.close_connection = True
            self._fail(e.status, e.message, e.headers)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            self.close_connection = True
        except Exception as e:  # noqa: BLE001 – jeder Rest wird als 500 gemeldet
            self.server.log_exception(f"{self.command} {self.path}", e)
            if self._streaming:
                self._sse_emit("fehler", {"fehler": f"Interner Fehler: {e}"})
                self._sse_end()
            else:
                self._fail(500, f"Interner Fehler: {e}")

    # -------------------------------------------------------------- Härtung
    def _check_headers(self) -> None:
        srv = self.server
        if not srv.host_allowed(self.headers.get("Host")):
            self.close_connection = True
            raise _HttpError(403, "Host-Header nicht erlaubt")
        origin = self.headers.get("Origin")
        if origin is not None and not srv.origin_allowed(origin):
            self.close_connection = True
            raise _HttpError(403, "Origin nicht erlaubt")
        length_raw = self.headers.get("Content-Length")
        length = 0
        if length_raw is not None:
            try:
                length = int(length_raw)
            except ValueError:
                self.close_connection = True
                raise _HttpError(400, "Ungültiger Content-Length-Header") from None
            if length < 0:
                self.close_connection = True
                raise _HttpError(400, "Ungültiger Content-Length-Header")
        if length > srv.max_body:
            self.close_connection = True
            raise _HttpError(413, f"Anfrage zu groß (max. {srv.max_body} Bytes)")
        if self.command in ("POST", "DELETE"):
            ctype = (self.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if ctype != "application/json":
                if length:
                    self.close_connection = True
                raise _HttpError(415, "Content-Type muss application/json sein")
        if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
            self.close_connection = True
            raise _HttpError(411, "Content-Length erforderlich")

    def _discard_body(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length:
            self.rfile.read(length)
        self._body_consumed = True

    def _read_body(self) -> dict | None:
        """Liest den JSON-Körper eines POST (``None`` bei GET/DELETE oder leerem Körper)."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        self._body_consumed = True
        if self.command != "POST":
            return None
        if not raw.strip():
            raise _HttpError(400, "JSON-Körper fehlt")
        try:
            data = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError:
            raise _HttpError(400, "Körper ist kein gültiges UTF-8") from None
        except json.JSONDecodeError as e:
            raise _HttpError(400, f"Ungültiges JSON: {e.msg} (Position {e.pos})") from None
        except (ValueError, RecursionError):
            raise _HttpError(400, "Ungültiges JSON (zu tief verschachtelt oder zu große Zahl)") from None
        if not isinstance(data, dict):
            raise _HttpError(400, "JSON-Objekt erwartet")
        try:
            json.dumps(data, ensure_ascii=False).encode("utf-8")
        except UnicodeEncodeError:
            raise _HttpError(400, "Ungültige Zeichen (einsame Surrogate) im JSON") from None
        return data

    # ------------------------------------------------------- Validierung
    @staticmethod
    def _opt_str(data: dict, key: str, *, required: bool = False, allow_empty: bool = False) -> str | None:
        value = data.get(key)
        if value is None:
            if required:
                raise _HttpError(400, f"Feld »{key}« fehlt")
            return None
        if not isinstance(value, str):
            raise _HttpError(400, f"Feld »{key}« muss ein Text sein")
        value = value.strip()
        if not value and not allow_empty:
            if required:
                raise _HttpError(400, f"Feld »{key}« darf nicht leer sein")
            return None
        return value

    @staticmethod
    def _opt_int(data: dict, key: str, *, required: bool = False) -> int | None:
        value = data.get(key)
        if value is None:
            if required:
                raise _HttpError(400, f"Feld »{key}« fehlt")
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, str) and value.strip().lstrip("-").isdigit() and len(value.strip()) <= 18:
                value = int(value)
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            else:
                raise _HttpError(400, f"Feld »{key}« muss eine ganze Zahl sein")
        if abs(value) >= 2 ** 63:
            raise _HttpError(400, f"Feld »{key}« ist zu groß")
        return value

    @staticmethod
    def _opt_bool(data: dict, key: str, default: bool = False) -> bool:
        value = data.get(key)
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("1", "true", "ja", "wahr", "0", "false", "nein", "falsch"):
            return value.strip().lower() in ("1", "true", "ja", "wahr")
        raise _HttpError(400, f"Feld »{key}« muss true oder false sein")

    def _query_int(self, key: str, default: int | None) -> int | None:
        values = self.query.get(key)
        if not values or values[-1].strip() == "":
            return default
        try:
            n = int(values[-1])
        except ValueError:
            raise _HttpError(400, f"Parameter »{key}« muss eine ganze Zahl sein") from None
        if n < 1 or n > _MAX_LIST:
            raise _HttpError(400, f"Parameter »{key}« muss zwischen 1 und {_MAX_LIST} liegen")
        return n

    def _query_str(self, key: str) -> str | None:
        values = self.query.get(key)
        if not values:
            return None
        v = values[-1].strip()
        return v or None

    # ---------------------------------------------------------------- SSE
    def _sse_start(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()
        self.close_connection = True
        self._streaming = True
        self._sse_broken = False

    def _sse_emit(self, event: str, data: Any) -> None:
        if self._sse_broken:
            return
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n".encode("utf-8")
        chunk = f"{len(payload):x}\r\n".encode("ascii") + payload + b"\r\n"
        try:
            self.wfile.write(chunk)
            self.wfile.flush()
        except OSError:
            self._sse_broken = True

    def _sse_end(self) -> None:
        if self._sse_broken:
            return
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except OSError:
            self._sse_broken = True

    # ------------------------------------------------------------ Endpunkte
    def route_index(self) -> None:
        path = self.server.index_path
        try:
            body = Path(path).read_bytes() if Path(path).is_file() else _PLACEHOLDER_HTML.encode("utf-8")
        except OSError as e:
            self.server.log(f"index.html nicht lesbar ({path}): {e}")
            body = _PLACEHOLDER_HTML.encode("utf-8")
        self._send_bytes(200, body, "text/html; charset=utf-8", {
            "X-Frame-Options": "DENY",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self' 'unsafe-inline' data: blob:; "
                                       "connect-src 'self'; frame-ancestors 'none'",
        })

    def route_status(self) -> None:
        self._send_json(200, {"ok": True, **self.server.brain.status()})

    def route_frage(self) -> None:
        data = self.body or {}
        frage = self._opt_str(data, "frage", required=True)
        sitzung = self._opt_str(data, "sitzung") or "standard"
        projekt = self._opt_str(data, "projekt")
        tiefe = self._opt_str(data, "tiefe")
        if tiefe is not None:
            tiefe = tiefe.lower()
            if tiefe not in DEPTHS:
                raise _HttpError(400, f"Feld »tiefe« muss eines von {', '.join(DEPTHS)} sein")
        stream = self._opt_bool(data, "stream", False)
        brain = self.server.brain
        if brain.busy:
            raise _HttpError(409, "OBITO denkt gerade an einer anderen Frage – bitte kurz warten.")

        if not stream:
            answer = brain.ask(frage, session_id=sitzung, project=projekt, depth=tiefe)
            payload = answer.to_dict()
            if answer.depth == "fehler":
                self._send_json(503, {"ok": False, "fehler": answer.text, **payload})
            else:
                self._send_json(200, {"ok": True, **payload})
            return

        self._sse_start()
        cancel = threading.Event()

        def on_token(text: str) -> None:
            if text:
                self._sse_emit("token", {"text": text})
            if self._sse_broken:
                cancel.set()

        def on_step(step) -> None:
            self._sse_emit("schritt", step.to_dict())
            if self._sse_broken:
                cancel.set()

        try:
            answer = brain.ask(frage, session_id=sitzung, project=projekt, depth=tiefe,
                               stream=on_token, progress=on_step, cancel=cancel)
        except Exception as e:  # noqa: BLE001 – Brain fängt LLM-Fehler selbst; Rest hier
            self.server.log_exception("POST /api/frage (stream)", e)
            self._sse_emit("fehler", {"fehler": f"Interner Fehler: {e}"})
        else:
            if answer.depth == "fehler":
                self._sse_emit("fehler", {"fehler": answer.text})
            else:
                self._sse_emit("antwort", answer.to_dict())
        finally:
            self._sse_end()

    def route_feedback(self) -> None:
        data = self.body or {}
        interaction_id = self._opt_int(data, "interaktion_id", required=True)
        rating = self._opt_int(data, "bewertung")
        if rating is not None and rating not in (-1, 0, 1):
            raise _HttpError(400, "Feld »bewertung« muss -1, 0 oder 1 sein")
        comment = self._opt_str(data, "kommentar")
        correction = self._opt_str(data, "korrektur")
        correction_full = self._opt_str(data, "korrektur_voll")
        rewrite = self._opt_bool(data, "umschreiben", False)
        if rewrite and not correction:
            raise _HttpError(400, "»umschreiben« benötigt eine »korrektur«")
        brain = self.server.brain
        suggestion: str | None = None
        try:
            if rewrite and correction:
                suggestion = brain.rewrite_correction(interaction_id, correction)
                if correction_full is None and suggestion:
                    correction_full = suggestion
            fb = brain.feedback(interaction_id, rating=rating, comment=comment,
                                correction=correction, correction_full=correction_full)
        except ValueError as e:
            raise _HttpError(404, str(e) or f"Interaktion {interaction_id} unbekannt") from None
        self._send_json(200, {
            "ok": True,
            "interaktion": fb.interaction.to_dict(),
            "lektionen": [l.to_dict() for l in fb.lessons],
            "korrektur_vorschlag": suggestion,
        })

    def route_erinnerungen_get(self) -> None:
        q = self._query_str("q")
        projekt = self._query_str("projekt")
        brain = self.server.brain
        if q:
            n = self._query_int("n", None)
            memories = brain.recall(q, k=n, project=projekt)
        else:
            n = self._query_int("n", 20)
            memories = brain.memory.recent(n, projekt)
        self._send_json(200, {"ok": True, "erinnerungen": [memory_to_dict(m) for m in memories]})

    def route_erinnerungen_post(self) -> None:
        data = self.body or {}
        inhalt = self._opt_str(data, "inhalt", required=True)
        art = self._opt_str(data, "art") or "notiz"
        if art not in KINDS:
            raise _HttpError(400, f"Feld »art« muss eines von {', '.join(KINDS)} sein")
        projekt = self._opt_str(data, "projekt")
        importance_raw = data.get("wichtigkeit", 0.6)
        if isinstance(importance_raw, bool) or not isinstance(importance_raw, (int, float)):
            raise _HttpError(400, "Feld »wichtigkeit« muss eine Zahl zwischen 0 und 1 sein")
        importance = float(importance_raw)
        if not 0.0 <= importance <= 1.0:
            raise _HttpError(400, "Feld »wichtigkeit« muss eine Zahl zwischen 0 und 1 sein")
        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list) and all(isinstance(t, str) for t in tags_raw):
            tags = [t.strip() for t in tags_raw if t.strip()]
        else:
            raise _HttpError(400, "Feld »tags« muss eine Liste von Texten sein")
        m = self.server.brain.remember(inhalt, kind=art, tags=tags, project=projekt, importance=importance)
        self._send_json(200, {"ok": True, "erinnerung": memory_to_dict(m)})

    def route_erinnerungen_delete(self, memory_id: str) -> None:
        if not self.server.brain.forget(int(memory_id)):
            raise _HttpError(404, f"Erinnerung {memory_id} unbekannt")
        self._send_json(200, {"ok": True, "geloescht": int(memory_id)})

    def route_lektionen_get(self) -> None:
        alle = (self._query_str("alle") or "").lower() in ("1", "true", "ja", "wahr")
        lessons = self.server.brain.learning.list_lessons(include_inactive=alle)
        self._send_json(200, {"ok": True, "lektionen": [l.to_dict() for l in lessons]})

    def route_lektionen_delete(self, lesson_id: str) -> None:
        if not self.server.brain.learning.delete_lesson(int(lesson_id)):
            raise _HttpError(404, f"Lektion {lesson_id} unbekannt")
        self._send_json(200, {"ok": True, "geloescht": int(lesson_id)})

    def route_modelle(self) -> None:
        brain = self.server.brain
        models = brain.models()
        self._send_json(200, {"ok": True, "modelle": [m.to_dict() for m in models], "aktuell": brain.cfg.model,
                              "schnell": brain.cfg.fast_model})

    # ------------------------------------------------------ Phase 2: Modell-Hub
    def route_modell_post(self) -> None:
        data = self.body or {}
        name = self._opt_str(data, "name", required=True, allow_empty=True) or ""
        fast = self._opt_bool(data, "schnell", False)
        brain = self.server.brain
        try:
            brain.set_model(name, fast=fast)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        except Exception as e:  # ModelNotFound und andere Backend-Fehler
            if type(e).__name__ == "ModelNotFound":
                raise _HttpError(404, f"Modell »{name}« ist nicht installiert – `ollama pull {name}`.") from None
            raise
        self._send_json(200, {"ok": True, "modell": brain.cfg.model, "schnell": brain.cfg.fast_model})

    def route_modelle_pull(self) -> None:
        data = self.body or {}
        name = self._opt_str(data, "name", required=True)
        backend = self.server.brain.backend
        self._sse_start()
        try:
            ok = backend.pull(name, progress=lambda chunk: self._sse_emit("fortschritt", chunk))
        except Exception as e:  # noqa: BLE001
            self._sse_emit("fehler", {"fehler": f"Modell »{name}« konnte nicht geladen werden: {e}"})
        else:
            if ok:
                self._sse_emit("fertig", {"name": name})
            else:
                self._sse_emit("fehler", {"fehler": f"Modell »{name}« konnte nicht geladen werden (Backend ohne Pull)."})
        finally:
            self._sse_end()

    def route_modelle_delete(self, name: str) -> None:
        from urllib.parse import unquote
        name = unquote(name)
        if not self.server.brain.backend.delete(name):
            raise _HttpError(404, f"Modell »{name}« unbekannt oder Löschen nicht unterstützt")
        self._send_json(200, {"ok": True, "geloescht": name})

    # ------------------------------------------------------ Phase 2: Dokumente
    def _knowledge(self):
        store = getattr(self.server.brain, "knowledge", None)
        if store is None:
            raise _HttpError(503, "Wissensbasis ist in diesem Denkkern nicht verfügbar")
        return store

    def route_dokumente_get(self) -> None:
        docs = self._knowledge().list(self._query_str("projekt"))
        self._send_json(200, {"ok": True, "dokumente": [d.to_dict() for d in docs]})

    def route_dokumente_post(self) -> None:
        data = self.body or {}
        store = self._knowledge()
        projekt = self._opt_str(data, "projekt")
        pfad = self._opt_str(data, "pfad")
        titel = self._opt_str(data, "titel")
        text = data.get("text")
        try:
            if pfad:
                try:
                    resolved = self.server.brain.tools.resolve(pfad)
                except PermissionError as e:
                    raise _HttpError(403, str(e)) from None
                if not Path(resolved).exists():
                    raise _HttpError(404, f"Datei oder Ordner nicht gefunden: {pfad}")
                if Path(resolved).is_dir():
                    result = store.add_directory(resolved, project=projekt)
                    self._send_json(200, {"ok": True, "verzeichnis": result})
                    return
                doc = store.add_file(resolved, project=projekt, title=titel)
            else:
                if not isinstance(text, str) or not text.strip():
                    raise _HttpError(400, "Feld »pfad« oder »titel« + »text« erforderlich")
                if not titel:
                    raise _HttpError(400, "Feld »titel« fehlt")
                doc = store.add_text(titel, text, project=projekt)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        except FileNotFoundError as e:
            raise _HttpError(404, str(e)) from None
        self._send_json(200, {"ok": True, "dokument": doc.to_dict()})

    def route_dokumente_suche(self) -> None:
        q = self._query_str("q")
        if not q:
            raise _HttpError(400, "Parameter »q« fehlt")
        n = self._query_int("n", 5) or 5
        hits = self._knowledge().search(q, k=n, project=self._query_str("projekt"))
        self._send_json(200, {"ok": True, "treffer": [c.to_dict() for c in hits]})

    def route_dokumente_sync(self) -> None:
        result = self._knowledge().sync()
        self._send_json(200, {"ok": True, **result})

    def route_dokumente_delete(self, doc_id: str) -> None:
        if not self._knowledge().remove(int(doc_id)):
            raise _HttpError(404, f"Dokument {doc_id} unbekannt")
        self._send_json(200, {"ok": True, "geloescht": int(doc_id)})

    # ------------------------------------------------------ Phase 2: Projekte
    def _projects(self):
        store = getattr(self.server.brain, "projects", None)
        if store is None:
            raise _HttpError(503, "Projektsystem ist in diesem Denkkern nicht verfügbar")
        return store

    @staticmethod
    def _unquote(value: str) -> str:
        from urllib.parse import unquote
        return unquote(value).strip()

    def route_projekte_get(self) -> None:
        alle = (self._query_str("alle") or "").lower() in ("1", "true", "ja", "wahr")
        projects = self._projects().list(include_archived=alle)
        self._send_json(200, {"ok": True, "projekte": [p.to_dict() for p in projects]})

    def route_projekte_post(self) -> None:
        data = self.body or {}
        name = self._opt_str(data, "name", required=True)
        beschreibung = self._opt_str(data, "beschreibung", allow_empty=True) or ""
        tags_raw = data.get("tags", [])
        if isinstance(tags_raw, str):
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()]
        elif isinstance(tags_raw, list) and all(isinstance(t, str) for t in tags_raw):
            tags = [t.strip() for t in tags_raw if t.strip()]
        else:
            raise _HttpError(400, "Feld »tags« muss eine Liste von Texten sein")
        try:
            project = self._projects().create(name, beschreibung, tags)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        self._send_json(200, {"ok": True, "projekt": project.to_dict()})

    def route_projekt_get(self, name: str) -> None:
        name = self._unquote(name)
        store = self._projects()
        project = store.get(name)
        if project is None:
            raise _HttpError(404, f"Projekt »{name}« unbekannt")
        self._send_json(200, {
            "ok": True, "projekt": project.to_dict(),
            "notizen": [n.to_dict() for n in store.notes(name, limit=200)],
            "dateien": store.files(name),
            "zusammenfassung": store.summary(name),
        })

    def route_projekt_delete(self, name: str) -> None:
        name = self._unquote(name)
        try:
            project = self._projects().archive(name)
        except ValueError as e:
            raise _HttpError(404, str(e)) from None
        self._send_json(200, {"ok": True, "projekt": project.to_dict()})

    def route_projekt_notiz_post(self, name: str) -> None:
        name = self._unquote(name)
        data = self.body or {}
        art = self._opt_str(data, "art") or "notiz"
        titel = self._opt_str(data, "titel", required=True)
        inhalt = self._opt_str(data, "inhalt", allow_empty=True) or ""
        store = self._projects()
        if store.get(name) is None:
            raise _HttpError(404, f"Projekt »{name}« unbekannt")
        try:
            note = store.add_note(name, art, titel, inhalt)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        self._send_json(200, {"ok": True, "notiz": note.to_dict()})

    def route_notiz_erledigt(self, note_id: str) -> None:
        data = self.body or {}
        done = self._opt_bool(data, "erledigt", True)
        try:
            note = self._projects().complete(int(note_id), done)
        except ValueError as e:
            raise _HttpError(404, str(e)) from None
        self._send_json(200, {"ok": True, "notiz": note.to_dict()})

    def route_notiz_delete(self, note_id: str) -> None:
        if not self._projects().delete_note(int(note_id)):
            raise _HttpError(404, f"Notiz {note_id} unbekannt")
        self._send_json(200, {"ok": True, "geloescht": int(note_id)})

    # ------------------------------------------------------ Phase 2: Missionen
    def _missions(self):
        runner = getattr(self.server.brain, "missions", None)
        if runner is None:
            raise _HttpError(503, "Missionen sind in diesem Denkkern nicht verfügbar")
        return runner

    def route_missionen_get(self) -> None:
        runner = self._missions()
        status = self._query_str("status")
        n = self._query_int("n", 50) or 50
        missions = runner.store.list(status=status, limit=n)
        self._send_json(200, {"ok": True, "missionen": [m.to_dict() for m in missions], "laufend": runner.running()})

    def route_missionen_post(self) -> None:
        data = self.body or {}
        ziel = self._opt_str(data, "ziel", required=True)
        projekt = self._opt_str(data, "projekt")
        start = self._opt_bool(data, "start", False)
        runner = self._missions()
        if self.server.brain.busy:
            raise _HttpError(409, "OBITO denkt gerade – bitte kurz warten.")
        try:
            mission = runner.plan(ziel, project=projekt)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        if start:
            try:
                runner.start(mission.id)
            except RuntimeError as e:
                raise _HttpError(409, str(e)) from None
            mission = runner.store.get(mission.id) or mission
        self._send_json(200, {"ok": True, "mission": mission.to_dict()})

    def route_mission_get(self, mid: str) -> None:
        mission = self._missions().store.get(int(mid))
        if mission is None:
            raise _HttpError(404, f"Mission {mid} unbekannt")
        self._send_json(200, {"ok": True, "mission": mission.to_dict(),
                              "laeuft": int(mid) in self._missions().running()})

    def route_mission_start(self, mid: str) -> None:
        runner = self._missions()
        try:
            runner.start(int(mid))
        except ValueError as e:
            raise _HttpError(404, str(e)) from None
        except RuntimeError as e:
            raise _HttpError(409, str(e)) from None
        mission = runner.store.get(int(mid))
        self._send_json(200, {"ok": True, "mission": mission.to_dict() if mission else None})

    def route_mission_stop(self, mid: str) -> None:
        runner = self._missions()
        if runner.store.get(int(mid)) is None:
            raise _HttpError(404, f"Mission {mid} unbekannt")
        stopped = runner.stop(int(mid))
        mission = runner.store.get(int(mid))
        self._send_json(200, {"ok": True, "gestoppt": bool(stopped), "mission": mission.to_dict() if mission else None})

    def route_mission_delete(self, mid: str) -> None:
        runner = self._missions()
        if runner.store.get(int(mid)) is None:
            raise _HttpError(404, f"Mission {mid} unbekannt")
        if int(mid) in runner.running():
            runner.stop(int(mid))
            join = getattr(runner, "join", None)
            if callable(join):
                join(int(mid), 5.0)
        runner.store.delete(int(mid))
        self._send_json(200, {"ok": True, "geloescht": int(mid)})

    # ---------------------------------------------------- Phase 2: Automationen
    def _automation(self):
        scheduler = getattr(self.server.brain, "automation", None)
        if scheduler is None:
            raise _HttpError(503, "Automationen sind in diesem Denkkern nicht verfügbar")
        return scheduler

    def route_automationen_get(self) -> None:
        scheduler = self._automation()
        self._send_json(200, {"ok": True, "automationen": [a.to_dict() for a in scheduler.store.list()],
                              "zeitplaner_aktiv": bool(scheduler.running)})

    def route_automationen_post(self) -> None:
        data = self.body or {}
        name = self._opt_str(data, "name", required=True)
        art = self._opt_str(data, "art", required=True)
        intervall = self._opt_int(data, "intervall_minuten", required=True)
        params = data.get("parameter", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise _HttpError(400, "Feld »parameter« muss ein Objekt sein")
        aktiv = self._opt_bool(data, "aktiv", True)
        try:
            auto = self._automation().store.create(name, art, int(intervall), params, aktiv)
        except ValueError as e:
            raise _HttpError(400, str(e)) from None
        self._send_json(200, {"ok": True, "automation": auto.to_dict()})

    def route_automationen_vorschlaege(self) -> None:
        from .automation import default_automations
        self._send_json(200, {"ok": True, "vorschlaege": default_automations()})

    def route_automationen_vorschlaege_post(self) -> None:
        from .automation import install_defaults
        data = self.body or {}
        aktiv = self._opt_bool(data, "aktiv", True)
        created = install_defaults(self._automation().store, enabled=aktiv)
        self._send_json(200, {"ok": True, "automationen": [a.to_dict() for a in created]})

    def route_automation_jetzt(self, aid: str) -> None:
        scheduler = self._automation()
        try:
            status, message = scheduler.run_once(int(aid))
        except ValueError as e:
            raise _HttpError(404, str(e)) from None
        auto = scheduler.store.get(int(aid))
        self._send_json(200, {"ok": True, "status": status, "meldung": message,
                              "automation": auto.to_dict() if auto else None})

    def route_automation_aktiv(self, aid: str) -> None:
        data = self.body or {}
        aktiv = self._opt_bool(data, "aktiv", True)
        try:
            auto = self._automation().store.update(int(aid), enabled=aktiv)
        except ValueError as e:
            raise _HttpError(404, str(e)) from None
        self._send_json(200, {"ok": True, "automation": auto.to_dict()})

    def route_automation_laeufe(self, aid: str) -> None:
        scheduler = self._automation()
        if scheduler.store.get(int(aid)) is None:
            raise _HttpError(404, f"Automation {aid} unbekannt")
        n = self._query_int("n", 20) or 20
        self._send_json(200, {"ok": True, "laeufe": scheduler.store.runs(int(aid), n)})

    def route_automation_delete(self, aid: str) -> None:
        if not self._automation().store.delete(int(aid)):
            raise _HttpError(404, f"Automation {aid} unbekannt")
        self._send_json(200, {"ok": True, "geloescht": int(aid)})

    # ---------------------------------------------------------- Phase 2: Pflege
    def route_konsolidieren(self) -> None:
        data = self.body or {}
        tage = self._opt_int(data, "tage")
        projekt = self._opt_str(data, "projekt")
        brain = self.server.brain
        if brain.busy:
            raise _HttpError(409, "OBITO denkt gerade – bitte kurz warten.")
        result = brain.consolidate(days=7 if tage is None else int(tage), project=projekt)
        self._send_json(200, {"ok": True, **result})

    def route_backup(self) -> None:
        data = self.body or {}
        keep = self._opt_int(data, "behalten")
        target = self.server.brain.backup(keep=7 if keep is None else int(keep))
        self._send_json(200, {"ok": True, "pfad": str(target)})


# ----------------------------------------------------------------- Komfort
def serve(brain, host: str | None = None, port: int | None = None, *, allow_dangerous: bool = False,
          on_start: Callable[[ObitoServer], None] | None = None) -> ObitoServer:
    """Startet den Server blockierend (``serve_forever``); Strg+C beendet ihn sauber.
    ``host``/``port`` fallen auf ``cfg.server_host``/``cfg.server_port`` zurück."""
    cfg = brain.cfg
    server = ObitoServer(brain, host or cfg.server_host, cfg.server_port if port is None else port,
                         allow_dangerous=allow_dangerous)
    if on_start is not None:
        on_start(server)
    server.start_background()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
