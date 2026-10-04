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
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


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
        self.allowed_hosts: frozenset[str] = frozenset(allowed)

        # Werkzeug-Freigabe: ohne Terminal keine Rückfrage möglich -> pauschal ja/nein.
        tools = getattr(brain, "tools", None)
        if tools is not None:
            flag = self.allow_dangerous
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
            if not allowed:
                raise _HttpError(404, f"Nicht gefunden: {path}")
            if handler_name is None or match is None:
                raise _HttpError(405, f"Methode {self.command} für {path} nicht erlaubt",
                                 {"Allow": ", ".join(dict.fromkeys(allowed))})
            self.body = self._read_body()
            getattr(self, handler_name)(*match.groups())
        except _HttpError as e:
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

    def _read_body(self) -> dict | None:
        """Liest den JSON-Körper eines POST (``None`` bei GET/DELETE oder leerem Körper)."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
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
        if not isinstance(data, dict):
            raise _HttpError(400, "JSON-Objekt erwartet")
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
            if isinstance(value, str) and value.strip().lstrip("-").isdigit():
                return int(value)
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise _HttpError(400, f"Feld »{key}« muss eine ganze Zahl sein")
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
        self._send_json(200, {"ok": True, "modelle": [m.to_dict() for m in models], "aktuell": brain.cfg.model})


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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return server
