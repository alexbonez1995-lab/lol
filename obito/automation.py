"""Automationen von OBITO: wiederkehrende Aufgaben im Hintergrund.

Eine *Automation* ist eine Aufgabe mit Art, Intervall und Parametern, die der
:class:`Scheduler` fällig werden lässt und nacheinander ausführt:

* ``gedaechtnis_konsolidieren`` – alte, unwichtige KI-Erinnerungen zusammenfassen
  (``brain.consolidate``).
* ``backup`` – Gedächtnis, Datenbanken und Konfiguration sichern (``brain.backup``).
* ``eval`` – das Modell gegen ein Eval-Set messen (``obito.training.evaluate``), Bericht ins Log-Verzeichnis.
* ``wissen_sync`` – indexierte Dokumente mit dem Dateisystem abgleichen (``brain.knowledge.sync``).
* ``mission`` – eine Mission planen und ausführen (``brain.missions``).
* ``werkzeug`` – ein Werkzeug der Registry aufrufen (gefährliche nur mit ``allow_dangerous``).

Zustand und Lauf-Protokoll liegen in SQLite (:class:`AutomationStore`, Datei
``cfg.data_path / "automationen.db"``). Fähigkeiten des Denkkerns, die noch nicht verdrahtet
sind, werden defensiv geprüft: fehlt eine, meldet der Lauf ``("fehler", <klare Meldung>)``
statt abzustürzen. Ein Fehler eines Laufs beendet nie den Hintergrund-Thread.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .agents import clip
from .llm import BackendUnavailable, LLMError, ModelNotFound

KINDS = ("gedaechtnis_konsolidieren", "backup", "eval", "wissen_sync", "mission", "werkzeug")

KIND_LABELS: dict[str, str] = {
    "gedaechtnis_konsolidieren": "Gedächtnis konsolidieren",
    "backup": "Backup",
    "eval": "Eval-Lauf",
    "wissen_sync": "Wissens-Sync",
    "mission": "Mission",
    "werkzeug": "Werkzeug",
}

STATUSES = ("ok", "fehler")
UPDATABLE_FIELDS = ("name", "interval_minutes", "params", "enabled")

MAX_MESSAGE_CHARS = 1000         # letzte_meldung / Lauf-Meldung
MAX_TOOL_OUTPUT_CHARS = 600      # Werkzeugausgabe in der Meldung
MAX_NAME_CHARS = 120
DEFAULT_TICK_SECONDS = 30.0
DEFAULT_CONSOLIDATE_DAYS = 7
DEFAULT_BACKUP_KEEP = 7
DEFAULT_EVAL_FILE = Path(__file__).resolve().parent.parent / "beispiele" / "eval_fragen.jsonl"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS automations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    kind              TEXT    NOT NULL,
    interval_minutes  INTEGER NOT NULL,
    params            TEXT    NOT NULL DEFAULT '{}',
    enabled           INTEGER NOT NULL DEFAULT 1,
    last_run          REAL,
    last_status       TEXT    NOT NULL DEFAULT '',
    last_message      TEXT    NOT NULL DEFAULT '',
    next_run          REAL,
    created_at        REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_automations_due ON automations(enabled, next_run);

CREATE TABLE IF NOT EXISTS runs (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    automation_id  INTEGER NOT NULL,
    started        REAL    NOT NULL,
    finished       REAL    NOT NULL,
    status         TEXT    NOT NULL,
    message        TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_runs_automation ON runs(automation_id, id);
"""


def _iso(ts: float | None) -> str | None:
    """Zeitstempel als ISO-8601 in lokaler Zeit (wie ``brain.memory_to_dict``); ``None`` bleibt ``None``."""
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(float(ts)))


def _squash(text: Any) -> str:
    """Eine Zeile: Zeilenumbrüche und Mehrfach-Leerzeichen zusammenfassen."""
    return " ".join(str(text if text is not None else "").split())


def _message(text: Any, limit: int = MAX_MESSAGE_CHARS) -> str:
    return clip(_squash(text), limit)


def _check_kind(kind: Any) -> str:
    if not isinstance(kind, str) or kind.strip().lower() not in KINDS:
        raise ValueError(f"Unbekannte Automations-Art {kind!r}. Erlaubt: {', '.join(KINDS)}")
    return kind.strip().lower()


def _check_interval(value: Any) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError
        iv = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Intervall muss eine ganze Zahl in Minuten sein, nicht {value!r}.") from None
    if iv < 1:
        raise ValueError("Intervall muss mindestens 1 Minute betragen.")
    return iv


def _check_name(value: Any) -> str:
    name = _squash(value)
    if not name:
        raise ValueError("Automation braucht einen Namen.")
    return name[:MAX_NAME_CHARS]


def _check_params(value: Any) -> dict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Parameter müssen ein JSON-Objekt (dict) sein.")
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Parameter sind nicht als JSON speicherbar: {e}") from None
    return dict(value)


def _int_param(params: dict, key: str, default: int, minimum: int = 1) -> int:
    """Ganzzahl-Parameter mit Standard; unbrauchbare Werte -> ``ValueError`` (deutsch)."""
    raw = params.get(key, default)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Parameter »{key}« muss eine ganze Zahl sein, nicht {raw!r}.") from None
    if value < minimum:
        raise ValueError(f"Parameter »{key}« muss mindestens {minimum} sein.")
    return value


# ------------------------------------------------------------------ Datenklasse
@dataclass
class Automation:
    """Eine geplante Aufgabe samt Zustand des letzten Laufs."""

    id: int
    name: str
    kind: str
    interval_minutes: int
    params: dict
    enabled: bool
    last_run: float | None
    last_status: str            # "ok" | "fehler" | "" (noch nie gelaufen)
    last_message: str
    next_run: float | None
    created_at: float

    @property
    def interval_seconds(self) -> float:
        return float(self.interval_minutes) * 60.0

    def is_due(self, now: float | None = None) -> bool:
        """Fällig = aktiv und ``next_run`` erreicht."""
        if not self.enabled or self.next_run is None:
            return False
        now = time.time() if now is None else float(now)
        return self.next_run <= now

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "art": self.kind,
            "intervall_minuten": self.interval_minutes,
            "parameter": dict(self.params),
            "aktiv": bool(self.enabled),
            "letzter_lauf": _iso(self.last_run),
            "letzter_status": self.last_status,
            "letzte_meldung": self.last_message,
            "naechster_lauf": _iso(self.next_run),
            "erstellt": _iso(self.created_at),
        }


# ------------------------------------------------------------------ Speicher
class AutomationStore:
    """Thread-sicherer SQLite-Speicher für Automationen und ihre Läufe."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._closed = False

    # ------------------------------------------------------------ intern
    @staticmethod
    def _row(row: sqlite3.Row) -> Automation:
        try:
            params = json.loads(row["params"] or "{}")
        except (TypeError, json.JSONDecodeError):
            params = {}
        if not isinstance(params, dict):
            params = {}
        return Automation(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            interval_minutes=int(row["interval_minutes"]),
            params=params,
            enabled=bool(row["enabled"]),
            last_run=row["last_run"],
            last_status=row["last_status"] or "",
            last_message=row["last_message"] or "",
            next_run=row["next_run"],
            created_at=row["created_at"],
        )

    def _require(self, aid: int) -> Automation:
        auto = self.get(aid)
        if auto is None:
            raise ValueError(f"Automation #{aid} nicht gefunden.")
        return auto

    # ------------------------------------------------------------ schreiben
    def create(self, name: str, kind: str, interval_minutes: int, params: dict | None = None,
               enabled: bool = True, *, now: float | None = None) -> Automation:
        """Legt eine Automation an. Der erste Lauf ist sofort fällig (``next_run = now``);
        danach gilt ``next_run = letzter Lauf + Intervall``. ``ValueError`` bei ungültiger Art,
        Intervall < 1 oder unbrauchbaren Parametern."""
        name = _check_name(name)
        kind = _check_kind(kind)
        interval = _check_interval(interval_minutes)
        params = _check_params(params)
        now = time.time() if now is None else float(now)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO automations(name, kind, interval_minutes, params, enabled, next_run, created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (name, kind, interval, json.dumps(params, ensure_ascii=False), 1 if enabled else 0, now, now),
            )
            self._db.commit()
            return self._require(int(cur.lastrowid))

    def update(self, aid: int, **fields: Any) -> Automation:
        """Ändert ``name``, ``interval_minutes``, ``params`` und/oder ``enabled``.

        Ein neues Intervall verschiebt den nächsten Lauf auf ``letzter Lauf + Intervall``
        (ohne bisherigen Lauf bleibt ``next_run`` unverändert). Unbekannte Felder -> ``ValueError``."""
        unknown = [k for k in fields if k not in UPDATABLE_FIELDS]
        if unknown:
            raise ValueError(f"Unbekannte Felder: {', '.join(unknown)}. Änderbar: {', '.join(UPDATABLE_FIELDS)}")
        with self._lock:
            auto = self._require(int(aid))
            sets: list[str] = []
            args: list[Any] = []
            if "name" in fields:
                sets.append("name = ?")
                args.append(_check_name(fields["name"]))
            if "interval_minutes" in fields:
                interval = _check_interval(fields["interval_minutes"])
                sets.append("interval_minutes = ?")
                args.append(interval)
                if auto.last_run is not None:
                    sets.append("next_run = ?")
                    args.append(float(auto.last_run) + interval * 60.0)
            if "params" in fields:
                sets.append("params = ?")
                args.append(json.dumps(_check_params(fields["params"]), ensure_ascii=False))
            if "enabled" in fields:
                sets.append("enabled = ?")
                args.append(1 if fields["enabled"] else 0)
            if sets:
                args.append(auto.id)
                self._db.execute(f"UPDATE automations SET {', '.join(sets)} WHERE id = ?", args)
                self._db.commit()
            return self._require(auto.id)

    def delete(self, aid: int) -> bool:
        """Löscht eine Automation samt Lauf-Protokoll."""
        with self._lock:
            cur = self._db.execute("DELETE FROM automations WHERE id = ?", (int(aid),))
            if cur.rowcount:
                self._db.execute("DELETE FROM runs WHERE automation_id = ?", (int(aid),))
            self._db.commit()
            return cur.rowcount > 0

    def record_run(self, aid: int, status: str, message: str, *, started: float | None = None,
                   now: float | None = None) -> None:
        """Protokolliert einen Lauf und setzt ``last_run``/``last_status``/``last_message`` sowie
        ``next_run = now + Intervall``. Unbekannter Status wird zu ``"fehler"``."""
        status = status if status in STATUSES else "fehler"
        now = time.time() if now is None else float(now)
        started = now if started is None else float(started)
        msg = _message(message)
        with self._lock:
            auto = self._require(int(aid))
            self._db.execute(
                "INSERT INTO runs(automation_id, started, finished, status, message) VALUES (?,?,?,?,?)",
                (auto.id, started, now, status, msg),
            )
            self._db.execute(
                "UPDATE automations SET last_run = ?, last_status = ?, last_message = ?, next_run = ? WHERE id = ?",
                (now, status, msg, now + auto.interval_seconds, auto.id),
            )
            self._db.commit()

    # ------------------------------------------------------------ lesen
    def get(self, aid: int) -> Automation | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM automations WHERE id = ?", (int(aid),)).fetchone()
        return self._row(row) if row else None

    def list(self) -> list[Automation]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM automations ORDER BY id").fetchall()
        return [self._row(r) for r in rows]

    def due(self, now: float | None = None) -> list[Automation]:
        """Aktive Automationen mit ``next_run <= now``, älteste Fälligkeit zuerst."""
        now = time.time() if now is None else float(now)
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM automations WHERE enabled = 1 AND next_run IS NOT NULL AND next_run <= ?"
                " ORDER BY next_run, id",
                (now,),
            ).fetchall()
        return [self._row(r) for r in rows]

    def runs(self, aid: int, limit: int = 20) -> list[dict]:
        """Letzte Läufe (neueste zuerst): ``id, automation_id, gestartet, beendet, dauer, status, meldung``."""
        limit = max(1, int(limit))
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM runs WHERE automation_id = ? ORDER BY id DESC LIMIT ?", (int(aid), limit)
            ).fetchall()
        return [
            {
                "id": r["id"],
                "automation_id": r["automation_id"],
                "gestartet": _iso(r["started"]),
                "beendet": _iso(r["finished"]),
                "dauer": round(max(0.0, float(r["finished"]) - float(r["started"])), 3),
                "status": r["status"],
                "meldung": r["message"],
            }
            for r in rows
        ]

    def stats(self) -> dict:
        """``automationen, aktiv, faellig, laeufe, fehler`` (Automationen mit letztem Status ``fehler``)."""
        now = time.time()
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) FROM automations").fetchone()[0]
            active = self._db.execute("SELECT COUNT(*) FROM automations WHERE enabled = 1").fetchone()[0]
            due = self._db.execute(
                "SELECT COUNT(*) FROM automations WHERE enabled = 1 AND next_run IS NOT NULL AND next_run <= ?",
                (now,),
            ).fetchone()[0]
            runs = self._db.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
            failed = self._db.execute(
                "SELECT COUNT(*) FROM automations WHERE last_status = 'fehler'").fetchone()[0]
        return {"automationen": total, "aktiv": active, "faellig": due, "laeufe": runs, "fehler": failed}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True


# ------------------------------------------------------------------ Vorschläge
def default_automations() -> list[dict]:
    """Empfohlene Automationen (``name, kind, interval_minutes, params, beschreibung``)."""
    return [
        {
            "name": "Konsolidierung täglich",
            "kind": "gedaechtnis_konsolidieren",
            "interval_minutes": 1440,
            "params": {"tage": DEFAULT_CONSOLIDATE_DAYS},
            "beschreibung": "Fasst alte, unwichtige KI-Erinnerungen und lange Sitzungen zusammen.",
        },
        {
            "name": "Backup täglich",
            "kind": "backup",
            "interval_minutes": 1440,
            "params": {"behalten": DEFAULT_BACKUP_KEEP},
            "beschreibung": "Sichert Gedächtnis, Datenbanken und Konfiguration; behält die letzten 7 Sicherungen.",
        },
        {
            "name": "Wissens-Sync",
            "kind": "wissen_sync",
            "interval_minutes": 60,
            "params": {},
            "beschreibung": "Gleicht indexierte Dokumente stündlich mit dem Dateisystem ab.",
        },
    ]


def install_defaults(store: AutomationStore, *, enabled: bool = True) -> list[Automation]:
    """Legt die Vorschläge an, die (nach Name) noch nicht existieren. Rückgabe: neu angelegte."""
    existing = {a.name for a in store.list()}
    created: list[Automation] = []
    for d in default_automations():
        if d["name"] in existing:
            continue
        created.append(store.create(d["name"], d["kind"], d["interval_minutes"], d.get("params"), enabled=enabled))
    return created


# ------------------------------------------------------------------ Zeitplaner
class Scheduler:
    """Führt fällige Automationen nacheinander in einem Hintergrund-Thread aus.

    ``run_once`` ist synchron und testbar; es protokolliert den Lauf im Speicher und im Log.
    Fehler eines Laufs werden zu ``("fehler", meldung)`` – der Thread läuft weiter.
    ``clock`` ist injizierbar (Tests)."""

    def __init__(self, brain: Any, store: AutomationStore, *, log_path: str | Path | None = None,
                 allow_dangerous: bool = False, tick_seconds: float = DEFAULT_TICK_SECONDS,
                 clock: Callable[[], float] = time.time):
        self.brain = brain
        self.store = store
        self.log_path = Path(log_path) if log_path else None
        self.allow_dangerous = bool(allow_dangerous)
        self.tick_seconds = max(0.01, float(tick_seconds))
        self.clock = clock
        self.current: int | None = None          # ID der gerade laufenden Automation
        self.last_error: str | None = None       # letzte Ausnahme der Schleife (nicht eines Laufs)
        self._run_lock = threading.Lock()        # Läufe strikt nacheinander
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handlers: dict[str, Callable[[Automation], tuple[str, str]]] = {
            "gedaechtnis_konsolidieren": self._run_consolidate,
            "backup": self._run_backup,
            "eval": self._run_eval,
            "wissen_sync": self._run_knowledge_sync,
            "mission": self._run_mission,
            "werkzeug": self._run_tool,
        }

    # ------------------------------------------------------------ Thread
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Startet den Hintergrund-Thread (daemon); mehrfacher Aufruf ist harmlos."""
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="obito-automation", daemon=True)
        self._thread.start()
        self._log("Zeitplaner gestartet.")

    def stop(self, timeout: float | None = 10.0) -> None:
        """Beendet den Thread sauber; wartet höchstens ``timeout`` Sekunden auf den laufenden Lauf."""
        thread = self._thread
        self._stop.set()
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
        if thread is not None and not thread.is_alive():
            self._thread = None
            self._log("Zeitplaner gestoppt.")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_due()
            except Exception as e:  # noqa: BLE001 - die Schleife darf nie sterben
                self.last_error = f"{e.__class__.__name__}: {e}"
                self._log(f"Fehler in der Zeitplaner-Schleife: {self.last_error}")
            self._stop.wait(self.tick_seconds)

    def run_due(self, now: float | None = None) -> list[tuple[int, str, str]]:
        """Führt alle fälligen Automationen aus. Rückgabe ``[(id, status, meldung), …]``."""
        now = self.clock() if now is None else float(now)
        results: list[tuple[int, str, str]] = []
        for auto in self.store.due(now):
            if self._stop.is_set():
                break
            status, msg = self.run_once(auto)
            results.append((auto.id, status, msg))
        return results

    # ------------------------------------------------------------ ein Lauf
    def run_once(self, automation: Automation | int) -> tuple[str, str]:
        """Führt eine Automation jetzt aus (synchron), protokolliert Lauf und Log-Zeile.

        Rückgabe ``("ok" | "fehler", deutsche Meldung)``. Wirft nur ``ValueError`` bei unbekannter ID."""
        auto = automation if isinstance(automation, Automation) else self.store.get(int(automation))
        if auto is None:
            raise ValueError(f"Automation #{automation} nicht gefunden.")
        with self._run_lock:
            self.current = auto.id
            started = self.clock()
            try:
                handler = self._handlers.get(auto.kind)
                if handler is None:
                    status, msg = "fehler", f"Unbekannte Automations-Art »{auto.kind}«."
                else:
                    status, msg = handler(auto)
                    if status not in STATUSES:
                        status = "fehler"
                    msg = _message(msg)
            except BackendUnavailable as e:
                status, msg = "fehler", f"Modell-Server nicht erreichbar: {e}"
            except ModelNotFound as e:
                status, msg = "fehler", f"Modell nicht installiert: {e}"
            except LLMError as e:
                status, msg = "fehler", f"Modellfehler: {e}"
            except OSError as e:
                status, msg = "fehler", f"Datei-Fehler: {e}"
            except (ValueError, TypeError, KeyError) as e:
                status, msg = "fehler", f"Ungültige Parameter oder Daten: {e}"
            except Exception as e:  # noqa: BLE001 - jeder Fehler wird zum Laufergebnis
                status, msg = "fehler", f"{e.__class__.__name__}: {e}"
                self._log("Rückverfolgung:\n" + traceback.format_exc().rstrip())
            finally:
                self.current = None
            msg = _message(msg)
            finished = self.clock()
            try:
                self.store.record_run(auto.id, status, msg, started=started, now=finished)
            except Exception as e:  # noqa: BLE001 - Speicherfehler dürfen den Thread nicht stoppen
                self._log(f"Lauf von #{auto.id} konnte nicht gespeichert werden: {e}")
            self._log(f"[{status}] #{auto.id} {auto.name} ({auto.kind}): {msg}")
            return status, msg

    # ------------------------------------------------------------ Hilfen
    def _capability(self, path: str) -> Any | None:
        """``brain.a.b`` defensiv auflösen; ``None`` wenn ein Glied fehlt oder nicht aufrufbar ist."""
        obj: Any = self.brain
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return obj if callable(obj) else None

    @staticmethod
    def _missing(what: str, attr: str) -> tuple[str, str]:
        return "fehler", f"{what} ist in diesem Denkkern nicht verfügbar (»{attr}« fehlt) – Integration nachrüsten."

    def _logs_dir(self) -> Path:
        cfg = getattr(self.brain, "cfg", None)
        logs = getattr(cfg, "logs_dir", None)
        if logs:
            return Path(logs)
        if self.log_path is not None:
            return self.log_path.parent
        return Path(tempfile.gettempdir()) / "obito-logs"

    def _log(self, text: str) -> None:
        if self.log_path is None:
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock()))
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(f"{stamp} {text}\n")
        except OSError:
            pass

    def log_tail(self, n: int = 50) -> list[str]:
        """Letzte ``n`` Zeilen der Log-Datei (leer, wenn es keine gibt)."""
        if self.log_path is None or not self.log_path.is_file():
            return []
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return []
        return lines[-max(1, int(n)):]

    @staticmethod
    def _format_dict(result: Any) -> str:
        if isinstance(result, dict):
            parts = []
            for k, v in result.items():
                if isinstance(v, (list, tuple, set)):
                    v = len(v)
                parts.append(f"{k}: {v}")
            return ", ".join(parts) if parts else "keine Änderungen"
        return _squash(result) if result is not None else "fertig"

    # ------------------------------------------------------------ Arten
    def _run_consolidate(self, auto: Automation) -> tuple[str, str]:
        fn = self._capability("consolidate")
        if fn is None:
            return self._missing("Gedächtnis-Konsolidierung", "Brain.consolidate")
        days = _int_param(auto.params, "tage", DEFAULT_CONSOLIDATE_DAYS)
        kwargs: dict[str, Any] = {"days": days}
        if auto.params.get("projekt"):
            kwargs["project"] = str(auto.params["projekt"])
        result = fn(**kwargs)
        if isinstance(result, dict):
            return "ok", (
                f"Konsolidierung (älter als {days} Tage): {result.get('zusammengefasst', 0)} zusammengefasst, "
                f"{result.get('geloescht', 0)} gelöscht, {result.get('sitzungen', 0)} Sitzungen"
            )
        return "ok", f"Konsolidierung (älter als {days} Tage): {self._format_dict(result)}"

    def _run_backup(self, auto: Automation) -> tuple[str, str]:
        fn = self._capability("backup")
        if fn is None:
            return self._missing("Backup", "Brain.backup")
        keep = _int_param(auto.params, "behalten", DEFAULT_BACKUP_KEEP)
        path = fn(keep=keep)
        return "ok", f"Backup erstellt: {path} (behalten: {keep})"

    def _resolve_eval_file(self, params: dict) -> Path:
        raw = params.get("datei")
        if raw is None or not str(raw).strip():
            return DEFAULT_EVAL_FILE
        p = Path(str(raw)).expanduser()
        if p.is_file():
            return p
        if not p.is_absolute():
            alt = DEFAULT_EVAL_FILE.parent.parent / p
            if alt.is_file():
                return alt
        raise FileNotFoundError(f"Eval-Datei nicht gefunden: {raw}")

    def _run_eval(self, auto: Automation) -> tuple[str, str]:
        from .training import evaluate

        backend = getattr(self.brain, "backend", None)
        if backend is None:
            return self._missing("Eval", "Brain.backend")
        cfg = getattr(self.brain, "cfg", None)
        params = auto.params
        path = self._resolve_eval_file(params)
        items = evaluate.load_items(path)
        max_items = params.get("max_fragen")
        if max_items is not None and str(max_items).strip():
            items = items[: _int_param(params, "max_fragen", len(items))]
        if not items:
            return "fehler", f"Eval-Datei {path} enthält keine Fragen."

        model = str(params.get("modell") or getattr(cfg, "model", "") or getattr(backend, "default_model", "") or "")
        judge = params.get("richter")
        judge = str(judge) if judge else None
        depth = params.get("tiefe")
        through_brain = None
        if depth:
            if self._capability("ask") is None:
                return self._missing("Eval über das Gremium", "Brain.ask")
            through_brain = self.brain
        num_ctx = getattr(cfg, "num_ctx", None)
        report = evaluate.run_eval(
            backend, model, items, judge_model=judge, through_brain=through_brain,
            depth=str(depth) if depth else None, num_ctx=int(num_ctx) if num_ctx else None,
        )
        logs = self._logs_dir()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.clock()))
        target = logs / f"eval-{stamp}.json"
        n = 1
        while target.exists():
            n += 1
            target = logs / f"eval-{stamp}-{n}.json"
        saved = evaluate.save_report(report, target)

        score = report.get("stichwort_score")
        judge_score = report.get("richter_score")
        parts = [f"Eval: {report.get('anzahl', len(items))} Fragen"]
        parts.append(f"Stichwort-Score {score:.2f}" if isinstance(score, (int, float)) else "kein Stichwort-Score")
        if isinstance(judge_score, (int, float)):
            parts.append(f"Richter {judge_score:.1f}/10")
        parts.append(f"Modell {model or '?'}")
        if depth:
            parts.append(f"Tiefe {depth}")
        parts.append(f"Bericht {saved}")
        warnings = report.get("warnungen") or []
        if warnings:
            parts.append(f"{len(warnings)} Warnung(en): {warnings[0]}")
        return "ok", ", ".join(parts)

    def _run_knowledge_sync(self, auto: Automation) -> tuple[str, str]:
        fn = self._capability("knowledge.sync")
        if fn is None:
            return self._missing("Wissens-Sync", "Brain.knowledge.sync")
        result = fn()
        return "ok", f"Wissens-Sync: {self._format_dict(result)}"

    def _run_mission(self, auto: Automation) -> tuple[str, str]:
        plan = self._capability("missions.plan")
        run = self._capability("missions.run")
        if plan is None or run is None:
            return self._missing("Missionen", "Brain.missions.plan/run")
        goal = _squash(auto.params.get("ziel"))
        if not goal:
            return "fehler", "Parameter »ziel« fehlt – eine Mission braucht ein Ziel."
        kwargs: dict[str, Any] = {}
        if auto.params.get("projekt"):
            kwargs["project"] = str(auto.params["projekt"])
        mission = plan(goal, **kwargs)
        mid = self._field(mission, "id")
        if mid is None:
            return "fehler", "Missionsplanung lieferte keine Mission."
        mission = run(mid)
        status = str(self._field(mission, "status") or "?")
        title = _squash(self._field(mission, "title") or self._field(mission, "titel") or goal)
        steps = self._field(mission, "steps") or self._field(mission, "schritte") or []
        done = sum(1 for s in steps if str(self._field(s, "status") or "") == "fertig")
        report = _squash(self._field(mission, "report") or self._field(mission, "bericht") or "")
        error = _squash(self._field(mission, "error") or self._field(mission, "fehler") or "")
        msg = f"Mission #{mid} »{title}«: {status}, {done}/{len(steps)} Schritte"
        if report:
            msg += f" – {clip(report, 300)}"
        if error:
            msg += f" – Fehler: {clip(error, 200)}"
        return ("ok" if status == "fertig" else "fehler"), msg

    @staticmethod
    def _field(obj: Any, name: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    def _run_tool(self, auto: Automation) -> tuple[str, str]:
        tools = getattr(self.brain, "tools", None)
        if tools is None or not callable(getattr(tools, "run", None)):
            return self._missing("Werkzeuge", "Brain.tools")
        name = _squash(auto.params.get("werkzeug"))
        if not name:
            return "fehler", "Parameter »werkzeug« fehlt."
        args = auto.params.get("args", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            return "fehler", "Parameter »args« muss ein JSON-Objekt sein."
        getter = getattr(tools, "get", None)
        tool = getter(name) if callable(getter) else None
        if tool is None:
            available = ", ".join(t.name for t in tools.list()) if callable(getattr(tools, "list", None)) else "?"
            return "fehler", f"Unbekanntes Werkzeug »{name}«. Verfügbar: {available}"
        if getattr(tool, "dangerous", False):
            if not self.allow_dangerous:
                return "fehler", (f"Werkzeug »{name}« ist gefährlich und läuft in Automationen nur mit "
                                  "ausdrücklicher Freigabe (allow_dangerous).")
            result = self._run_dangerous(tools, tool, args)
        else:
            result = tools.run(name, args)
        if getattr(result, "ok", False):
            output = _squash(getattr(result, "output", "")) or "(keine Ausgabe)"
            return "ok", f"{name}: {clip(output, MAX_TOOL_OUTPUT_CHARS)}"
        return "fehler", f"{name}: {getattr(result, 'error', None) or 'unbekannter Fehler'}"

    @staticmethod
    def _run_dangerous(tools: Any, tool: Any, args: dict) -> Any:
        """Führt ein freigegebenes gefährliches Werkzeug über eine Schatten-Registry aus, damit die
        Freigabe-Regel der Haupt-Registry (Nutzer-Rückfrage) nicht verändert werden muss."""
        from .tools import ToolRegistry

        shadow = ToolRegistry(getattr(tools, "workspace", ".") or ".", confirm=lambda n, a: True,
                              confirm_dangerous=False)
        shadow.memory = getattr(tools, "memory", None)
        shadow.register(tool)
        return shadow.run(tool.name, args)


__all__ = [
    "KINDS", "KIND_LABELS", "STATUSES", "UPDATABLE_FIELDS", "DEFAULT_EVAL_FILE",
    "Automation", "AutomationStore", "Scheduler", "default_automations", "install_defaults",
]
