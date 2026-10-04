"""Missionen von OBITO – autonome Mehrschritt-Aufgaben.

Eine Mission ist ein Ziel in natürlicher Sprache („Vergleiche drei Materialien für den
Drohnenrahmen und empfiehl eines“), das der Denkkern in höchstens acht Schritte zerlegt:

* ``frage``    – eine Teilfrage, die über :meth:`Brain.ask` (Stufe ``mittel``) beantwortet wird;
                 die Ergebnisse der vorigen Schritte wandern gekürzt in die Frage.
* ``werkzeug`` – ein konkreter Werkzeugaufruf (``rechnen``, ``verzeichnis`` …) über
                 ``brain.tools.run``; gefährliche Werkzeuge nur mit Freigabe.

Der :class:`MissionRunner` plant (JSON-Stufe ``mission``), führt sequenziell aus, zählt
Fehler, achtet auf Abbruch, lässt OMEGA einen Bericht schreiben (Stufe ``bericht``) und legt
ihn – wenn ein Projektsystem am Brain hängt – als Projektnotiz ``ergebnis`` ab. Jede Änderung
wird sofort im :class:`MissionStore` (SQLite) gespeichert, damit HUD und CLI den Fortschritt
jederzeit lesen können. Alles lokal, nur Standardbibliothek, mit ``FakeBackend`` testbar.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import agents
from .agents import OMEGA, STAGE_TAG, clip
from .llm import LLMError, parse_json

STEP_KINDS = ("frage", "werkzeug")
STEP_STATUS = ("offen", "laeuft", "fertig", "fehler", "uebersprungen")
MISSION_STATUS = ("geplant", "laeuft", "pausiert", "fertig", "fehler", "abgebrochen")
FINAL_STATUS = ("fertig", "fehler", "abgebrochen")

MAX_STEPS = 8                       # Obergrenze je Plan
MAX_FAILED_STEPS = 2                # so viele Schrittfehler beenden die Mission mit „fehler“
MAX_TITLE_CHARS = 60                # Titel-Fallback: Anfang des Ziels
MAX_DESCRIPTION_CHARS = 500         # je Schrittbeschreibung
MAX_PREVIOUS_RESULTS_CHARS = 1500   # bisherige Ergebnisse in einer Teilfrage
MAX_RESULT_PER_STEP_CHARS = 600     # je Einzelergebnis im Ergebnisblock
MAX_RESULT_IN_REPORT_CHARS = 1500   # je Schrittergebnis im Berichts-Prompt
MAX_GOAL_CHARS = 3000
MAX_SUMMARY_CHARS = 1200
MIN_PLAN_TOKENS = 900               # ein Plan mit 8 Schritten passt nicht in 500 Tokens
CANCELLED_TEXT = "Abgebrochen"

ProgressCallback = Callable[["Mission"], None]
ConfirmCallback = Callable[[str, dict], bool]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS missions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    goal        TEXT    NOT NULL,
    project     TEXT,
    status      TEXT    NOT NULL DEFAULT 'geplant',
    steps       TEXT    NOT NULL DEFAULT '[]',
    report      TEXT    NOT NULL DEFAULT '',
    error       TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL,
    updated_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_missions_status ON missions(status, created_at);
CREATE INDEX IF NOT EXISTS idx_missions_project ON missions(project);

CREATE VIRTUAL TABLE IF NOT EXISTS missions_fts USING fts5(
    title, goal, report,
    content='missions', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS missions_ai AFTER INSERT ON missions BEGIN
    INSERT INTO missions_fts(rowid, title, goal, report) VALUES (new.id, new.title, new.goal, new.report);
END;
CREATE TRIGGER IF NOT EXISTS missions_ad AFTER DELETE ON missions BEGIN
    INSERT INTO missions_fts(missions_fts, rowid, title, goal, report)
        VALUES ('delete', old.id, old.title, old.goal, old.report);
END;
CREATE TRIGGER IF NOT EXISTS missions_au AFTER UPDATE OF title, goal, report ON missions BEGIN
    INSERT INTO missions_fts(missions_fts, rowid, title, goal, report)
        VALUES ('delete', old.id, old.title, old.goal, old.report);
    INSERT INTO missions_fts(rowid, title, goal, report) VALUES (new.id, new.title, new.goal, new.report);
END;
"""


def _iso(ts: float) -> str:
    """Zeitstempel als ISO-8601 in lokaler Zeit (wie ``brain.memory_to_dict``)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


# ------------------------------------------------------------------ Datenklassen
@dataclass
class MissionStep:
    """Ein Schritt einer Mission.

    ``kind`` ∈ frage|werkzeug, ``status`` ∈ offen|laeuft|fertig|fehler|uebersprungen."""

    idx: int
    description: str
    kind: str = "frage"
    tool: str | None = None
    args: dict = field(default_factory=dict)
    status: str = "offen"
    result: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict:
        return {
            "nr": int(self.idx),
            "beschreibung": self.description,
            "art": self.kind,
            "werkzeug": self.tool,
            "args": dict(self.args or {}),
            "status": self.status,
            "ergebnis": self.result,
            "dauer": round(float(self.duration), 3),
        }

    @classmethod
    def from_dict(cls, d: dict, idx: int | None = None) -> "MissionStep":
        """Liest einen Schritt aus den Schlüsseln von :meth:`to_dict` (tolerant gegenüber Lücken)."""
        if not isinstance(d, dict):
            d = {}
        nr = d.get("nr", idx if idx is not None else 1)
        try:
            nr = int(nr)
        except (TypeError, ValueError):
            nr = int(idx or 1)
        kind = str(d.get("art") or "frage")
        if kind not in STEP_KINDS:
            kind = "frage"
        status = str(d.get("status") or "offen")
        if status not in STEP_STATUS:
            status = "offen"
        args = d.get("args")
        tool = d.get("werkzeug")
        try:
            duration = float(d.get("dauer") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        return cls(
            idx=nr,
            description=str(d.get("beschreibung") or ""),
            kind=kind,
            tool=str(tool) if tool else None,
            args=dict(args) if isinstance(args, dict) else {},
            status=status,
            result=str(d.get("ergebnis") or ""),
            duration=duration,
        )


@dataclass
class Mission:
    """Eine Mission mit Ziel, Plan, Status, Bericht.

    ``status`` ∈ geplant|laeuft|pausiert|fertig|fehler|abgebrochen."""

    id: int | None
    title: str
    goal: str
    project: str | None
    status: str = "geplant"
    steps: list[MissionStep] = field(default_factory=list)
    report: str = ""
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def progress(self) -> float:
        """Anteil fertiger Schritte (0.0 … 1.0); ohne Schritte 0.0."""
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == "fertig")
        return done / len(self.steps)

    def failed_steps(self) -> int:
        return sum(1 for s in self.steps if s.status == "fehler")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "titel": self.title,
            "ziel": self.goal,
            "projekt": self.project,
            "status": self.status,
            "fortschritt": round(self.progress, 3),
            "schritte": [s.to_dict() for s in self.steps],
            "bericht": self.report,
            "fehler": self.error,
            "erstellt": _iso(self.created_at) if self.created_at else None,
            "geaendert": _iso(self.updated_at) if self.updated_at else None,
        }


# ------------------------------------------------------------------ Speicher
class MissionStore:
    """Thread-sicherer SQLite-Speicher für Missionen (Schritte als JSON-Spalte)."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA_SQL)
        self._db.commit()
        self._closed = False

    # ------------------------------------------------------------ intern
    @staticmethod
    def _row(row: sqlite3.Row) -> Mission:
        try:
            raw_steps = json.loads(row["steps"] or "[]")
        except (TypeError, ValueError):
            raw_steps = []
        if not isinstance(raw_steps, list):
            raw_steps = []
        steps = [MissionStep.from_dict(d, i + 1) for i, d in enumerate(raw_steps) if isinstance(d, dict)]
        status = row["status"] if row["status"] in MISSION_STATUS else "geplant"
        return Mission(
            id=int(row["id"]),
            title=row["title"],
            goal=row["goal"],
            project=row["project"],
            status=status,
            steps=steps,
            report=row["report"] or "",
            error=row["error"] or "",
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MissionStore ist bereits geschlossen.")

    # ------------------------------------------------------------ schreiben
    def save(self, mission: Mission) -> Mission:
        """Legt eine Mission an (``id`` leer) oder aktualisiert sie (Upsert). Setzt ``updated_at``
        und – beim Anlegen – ``created_at`` und ``id``. Rückgabe: dieselbe Mission mit ``id``."""
        if not isinstance(mission, Mission):
            raise TypeError("save erwartet eine Mission.")
        if not (mission.goal or "").strip():
            raise ValueError("Mission ohne Ziel kann nicht gespeichert werden.")
        if mission.status not in MISSION_STATUS:
            raise ValueError(f"Unbekannter Missionsstatus {mission.status!r}.")
        now = time.time()
        mission.updated_at = now
        if not mission.created_at:
            mission.created_at = now
        if not mission.title:
            mission.title = clip(" ".join(mission.goal.split()), MAX_TITLE_CHARS)
        steps_json = json.dumps([s.to_dict() for s in mission.steps], ensure_ascii=False, default=str)
        with self._lock:
            self._ensure_open()
            if mission.id is None or int(mission.id) <= 0:
                cur = self._db.execute(
                    "INSERT INTO missions(title, goal, project, status, steps, report, error, created_at,"
                    " updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                    (mission.title, mission.goal, mission.project, mission.status, steps_json,
                     mission.report or "", mission.error or "", mission.created_at, mission.updated_at),
                )
                mission.id = int(cur.lastrowid)
            else:
                cur = self._db.execute(
                    "UPDATE missions SET title = ?, goal = ?, project = ?, status = ?, steps = ?, report = ?,"
                    " error = ?, updated_at = ? WHERE id = ?",
                    (mission.title, mission.goal, mission.project, mission.status, steps_json,
                     mission.report or "", mission.error or "", mission.updated_at, int(mission.id)),
                )
                if cur.rowcount == 0:
                    # ID unbekannt (z. B. gelöscht): unter dieser ID neu anlegen
                    self._db.execute(
                        "INSERT INTO missions(id, title, goal, project, status, steps, report, error,"
                        " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (int(mission.id), mission.title, mission.goal, mission.project, mission.status,
                         steps_json, mission.report or "", mission.error or "", mission.created_at,
                         mission.updated_at),
                    )
            self._db.commit()
        return mission

    def delete(self, mid: int) -> bool:
        with self._lock:
            self._ensure_open()
            cur = self._db.execute("DELETE FROM missions WHERE id = ?", (int(mid),))
            self._db.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------ lesen
    def get(self, mid: int) -> Mission | None:
        with self._lock:
            self._ensure_open()
            row = self._db.execute("SELECT * FROM missions WHERE id = ?", (int(mid),)).fetchone()
        return self._row(row) if row else None

    def list(self, status: str | None = None, limit: int = 50) -> list[Mission]:
        """Missionen, neueste zuerst; optional nach Status gefiltert."""
        sql = "SELECT * FROM missions"
        args: tuple = ()
        if status:
            sql += " WHERE status = ?"
            args = (str(status),)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self._lock:
            self._ensure_open()
            rows = self._db.execute(sql, args + (max(0, int(limit)),)).fetchall()
        return [self._row(r) for r in rows]

    def search(self, query: str, k: int = 10) -> list[Mission]:
        """Volltextsuche über Titel, Ziel und Bericht."""
        words = [w for w in agents._WORD.findall((query or "").lower()) if len(w) > 1]
        if not words:
            return []
        fts = " OR ".join(f'"{w[:-1]}"*' if len(w) >= 5 else f'"{w}"' for w in dict.fromkeys(words))
        with self._lock:
            self._ensure_open()
            rows = self._db.execute(
                "SELECT m.* FROM missions_fts JOIN missions m ON m.id = missions_fts.rowid"
                " WHERE missions_fts MATCH ? ORDER BY bm25(missions_fts) LIMIT ?",
                (fts, max(0, int(k))),
            ).fetchall()
        return [self._row(r) for r in rows]

    def count(self, status: str | None = None) -> int:
        with self._lock:
            self._ensure_open()
            if status:
                return self._db.execute("SELECT COUNT(*) FROM missions WHERE status = ?", (status,)).fetchone()[0]
            return self._db.execute("SELECT COUNT(*) FROM missions").fetchone()[0]

    def stats(self) -> dict:
        with self._lock:
            self._ensure_open()
            by_status = dict(self._db.execute("SELECT status, COUNT(*) FROM missions GROUP BY status").fetchall())
        return {"missionen": sum(by_status.values()), "nach_status": by_status,
                "aktiv": by_status.get("laeuft", 0)}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True


# ------------------------------------------------------------------ Prompts
MISSION_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "titel": {"type": "string"},
        "schritte": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "beschreibung": {"type": "string"},
                    "art": {"type": "string", "enum": list(STEP_KINDS)},
                    "werkzeug": {"type": ["string", "null"]},
                    "args": {"type": "object"},
                },
                "required": ["beschreibung", "art", "werkzeug", "args"],
            },
        },
    },
    "required": ["titel", "schritte"],
}

_PLAN_EXAMPLE = {
    "titel": "Rahmenmaterial für 600-g-Quadcopter wählen",
    "schritte": [
        {"beschreibung": "Anforderungen an den Rahmen (Masse, Steifigkeit, Budget) aus dem Ziel ableiten.",
         "art": "frage", "werkzeug": None, "args": {}},
        {"beschreibung": "Masse eines 120-cm³-Rahmens aus Aluminium berechnen (Dichte 2,7 g/cm³).",
         "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "120*2.7"}},
        {"beschreibung": "CFK, GFK und Aluminium gegeneinander abwägen und eines empfehlen.",
         "art": "frage", "werkzeug": None, "args": {}},
    ],
}


def _system(stage: str, who: str, body: str) -> dict:
    body = (body or "").rstrip()
    tag = STAGE_TAG.format(stage=stage, who=who)
    return {"role": "system", "content": (body + "\n" + tag) if body else tag}


def _join(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def mission_plan_messages(goal: str, project_summary: str, tools_block: str) -> list[dict]:
    """Stufe ``mission``: ein Ziel in höchstens :data:`MAX_STEPS` Schritte zerlegen (JSON).

    Die System-Nachricht trägt die Regeln und den Werkzeugblock und endet mit dem Marker
    ``[OBITO:mission:system]``; Ziel und Projektzusammenfassung stehen in der Nutzer-Nachricht."""
    tools_block = (tools_block or "").strip()
    body = (
        "Du bist der Missionsplaner von OBITO, einem lokalen KI-Assistenten für Ingenieurprojekte. "
        "Du zerlegst ein Ziel in einen kurzen, ausführbaren Plan. Du beantwortest das Ziel NICHT selbst.\n\n"
        f"Regeln:\n"
        f"- Höchstens {MAX_STEPS} Schritte, in sinnvoller Reihenfolge; jeder Schritt ein konkreter, "
        "eigenständig verständlicher Satz.\n"
        "- „art“ ist „frage“ oder „werkzeug“. Bevorzuge „frage“: Analyse, Vergleich, Empfehlung, "
        "Recherche, Entwurf – das Expertengremium beantwortet jede Teilfrage und kennt die Ergebnisse "
        "der vorigen Schritte.\n"
        "- „werkzeug“ nur für einen konkreten Werkzeugaufruf, dessen Argumente du jetzt schon vollständig "
        "und gültig angeben kannst (z. B. eine Rechnung mit Zahlen). Dann enthält „werkzeug“ exakt den "
        "Werkzeugnamen und „args“ die Argumente laut Signatur. Bei „frage“ ist „werkzeug“ null und "
        "„args“ {}.\n"
        "- Erfinde keine Werkzeuge und keine Fakten. Ist ein Wert unbekannt, formuliere einen "
        "„frage“-Schritt, der ihn klärt.\n"
        "- „titel“: höchstens 60 Zeichen, prägnant.\n\n"
        + (f"{tools_block}" if tools_block else "Es sind keine Werkzeuge verfügbar – nutze nur „frage“.")
    )
    user = _join(
        f"Ziel der Mission:\n{clip((goal or '').strip(), MAX_GOAL_CHARS)}",
        f"Projektkontext:\n{clip((project_summary or '').strip(), MAX_SUMMARY_CHARS)}"
        if (project_summary or "").strip() else "",
        "Antworte nur mit JSON.\nBeispiel:\n" + json.dumps(_PLAN_EXAMPLE, ensure_ascii=False),
    )
    return [_system("mission", "system", body), {"role": "user", "content": user}]


def _status_label(step: MissionStep) -> str:
    return {"fertig": "erledigt", "fehler": "fehlgeschlagen", "uebersprungen": "übersprungen",
            "offen": "nicht ausgeführt", "laeuft": "nicht abgeschlossen"}.get(step.status, step.status)


def mission_report_messages(goal: str, steps: Sequence[MissionStep]) -> list[dict]:
    """Stufe ``bericht``: OMEGA fasst die Mission zusammen (Ziel, Ergebnisse je Schritt, offene
    Punkte, Empfehlung). Marker ``[OBITO:bericht:OMEGA]``."""
    body = _join(
        OMEGA.system_prompt,
        agents.BASE_RULES,
        "Aufgabe: Schreibe den Abschlussbericht einer Mission auf Deutsch, nüchtern und konkret, mit "
        "genau diesen Abschnitten als Markdown-Überschriften:\n"
        "## Ziel – das Ziel in einem Satz.\n"
        "## Ergebnisse je Schritt – je Schritt eine Zeile „n. Beschreibung – Kernergebnis“ mit Zahlen "
        "und Einheiten; fehlgeschlagene oder übersprungene Schritte als solche benennen.\n"
        "## Offene Punkte – was fehlt, unsicher ist oder nicht geklärt werden konnte (ehrlich, keine "
        "erfundenen Ergebnisse).\n"
        "## Empfehlung – die konkrete nächste Handlung für den Nutzer.\n"
        "Übernimm Zahlen wörtlich aus den Ergebnissen; erfinde nichts, was in den Ergebnissen fehlt.",
    )
    lines: list[str] = []
    for s in steps:
        head = f"{s.idx}. [{s.kind}{' ' + s.tool if s.tool else ''}] {s.description} – {_status_label(s)}"
        result = " ".join((s.result or "").split())
        lines.append(head + (f"\nErgebnis: {clip(result, MAX_RESULT_IN_REPORT_CHARS)}" if result else ""))
    user = _join(
        f"Ziel der Mission:\n{clip((goal or '').strip(), MAX_GOAL_CHARS)}",
        "Schritte und Ergebnisse:\n" + ("\n\n".join(lines) if lines else "(keine Schritte ausgeführt)"),
        "Schreibe jetzt den Bericht.",
    )
    return [_system("bericht", OMEGA.id, body), {"role": "user", "content": user}]


# ------------------------------------------------------------------ Normalisierer
_KIND_WORDS_TOOL = ("werkzeug", "tool", "aufruf", "call", "rechn", "funktion", "action", "aktion")


def _norm_kind(value: Any) -> str:
    s = agents._fold(value).strip()
    if any(w in s for w in _KIND_WORDS_TOOL):
        return "werkzeug"
    return "frage"


def _norm_args(value: Any) -> dict:
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    if isinstance(value, str) and value.strip().startswith("{"):
        parsed = parse_json(value)
        if isinstance(parsed, dict):
            return {str(k): v for k, v in parsed.items()}
    return {}


def _norm_tool(value: Any, known: dict[str, str]) -> str | None:
    """Werkzeugname tolerant auflösen (Groß-/Kleinschreibung, ``rechnen(...)``, ``werkzeug: rechnen``)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, dict):
        value = agents._get(agents._dict(value) or {}, "name", "werkzeug", "tool")
    s = agents._fold(value).strip()
    if not s:
        return None
    s = s.split("(")[0].split(":")[-1].strip().strip("`'\"»« ")
    s = s.replace("-", "_").replace(" ", "_")
    return known.get(s)


def parse_mission(text: Any, known_tools: Iterable[str] | None = None, *, goal: str = "",
                  max_steps: int = MAX_STEPS) -> dict | None:
    """Normalisiert eine Planantwort auf ``{"titel": str, "schritte": [{"beschreibung", "art",
    "werkzeug", "args"}]}``.

    Tolerant wie die ``agents.parse_*``-Helfer: Schlüssel case-/umlaut-unempfindlich, Liste auf
    oberster Ebene gilt als Schrittliste, Einzelstring als Beschreibung. Unbekannte Werkzeuge werden
    zu ``art="frage"``; ``args`` muss ein Objekt sein (sonst ``{}``); höchstens ``max_steps``
    Schritte. Fehlt der Titel, dient der Anfang von ``goal`` (60 Zeichen). ``None``, wenn kein
    verwertbarer Schritt enthalten ist."""
    obj = agents._load(text)
    if obj is None:
        return None
    known: dict[str, str] = {}
    for name in known_tools or ():
        s = str(name)
        known[agents._fold(s).strip()] = s
        known[s] = s

    title = ""
    raw_steps: Any
    if isinstance(obj, list):
        raw_steps = obj
    else:
        d = agents._dict(obj)
        if d is None:
            return None
        title = agents._text(agents._get(d, "titel", "title", "name", "ueberschrift", "mission", default=""))
        raw_steps = agents._get(d, "schritte", "steps", "plan", "aufgaben", "tasks", "schritt", default=None)
        if raw_steps is None:
            # Vielleicht ist das Objekt selbst ein einzelner Schritt
            if agents._get(d, "beschreibung", "description", "aufgabe", "text") is not None:
                raw_steps = [obj]
            else:
                return None
    if isinstance(raw_steps, (dict, str)):
        raw_steps = [raw_steps]
    if not isinstance(raw_steps, list):
        return None

    steps: list[dict] = []
    limit = max(1, int(max_steps or MAX_STEPS))
    for it in raw_steps:
        if len(steps) >= limit:
            break
        if isinstance(it, str):
            desc = " ".join(it.split())
            if desc:
                steps.append({"beschreibung": clip(desc, MAX_DESCRIPTION_CHARS), "art": "frage",
                              "werkzeug": None, "args": {}})
            continue
        if not isinstance(it, dict):
            continue
        item = agents._dict(it) or {}
        desc = " ".join(agents._text(agents._get(item, "beschreibung", "description", "aufgabe", "text",
                                                  "schritt", "titel", "name", default="")).split())
        kind = _norm_kind(agents._get(item, "art", "kind", "typ", "type", default="frage"))
        tool_raw = agents._get(item, "werkzeug", "tool", "funktion", "name_werkzeug")
        tool = _norm_tool(tool_raw, known)
        args = _norm_args(agents._get(item, "args", "argumente", "parameter", "arguments", default={}))
        if tool is None and kind == "werkzeug" and isinstance(tool_raw, dict):
            # {"werkzeug": {"name": "rechnen", "args": {...}}}
            args = args or _norm_args(agents._get(agents._dict(tool_raw) or {}, "args", "argumente", default={}))
        if kind == "werkzeug" and tool is None and tool_raw is None and known:
            # Werkzeugname direkt in der Beschreibung („rechnen: 2*3“)?
            tool = _norm_tool(desc.split()[0] if desc else "", known)
        if tool is None:
            kind = "frage"
            args = {}
        else:
            kind = "werkzeug"
        if not desc:
            if tool is None:
                continue
            desc = f"Werkzeug »{tool}« ausführen" + (f" mit {json.dumps(args, ensure_ascii=False)}" if args else "")
        steps.append({"beschreibung": clip(desc, MAX_DESCRIPTION_CHARS), "art": kind, "werkzeug": tool,
                      "args": args})
    if not steps:
        return None
    title = " ".join(title.split())
    if not title:
        title = clip(" ".join((goal or "").split()), MAX_TITLE_CHARS)
    return {"titel": clip(title, 200), "schritte": steps}


def _results_block(steps: Sequence[MissionStep], limit: int = MAX_PREVIOUS_RESULTS_CHARS) -> str:
    """Bisherige Ergebnisse (fertige und fehlgeschlagene Schritte) in chronologischer Reihenfolge;
    die jüngsten bleiben vollständig, ältere fallen weg, wenn das Budget nicht reicht."""
    entries: list[str] = []
    for s in steps:
        if s.status not in ("fertig", "fehler"):
            continue
        result = " ".join((s.result or "").split())
        if not result:
            continue
        label = "Ergebnis" if s.status == "fertig" else "Fehler"
        entries.append(f"Schritt {s.idx} ({s.description}) – {label}: {clip(result, MAX_RESULT_PER_STEP_CHARS)}")
    chosen: list[str] = []
    total = 0
    for entry in reversed(entries):
        if chosen and total + len(entry) + 1 > limit:
            break
        chosen.append(entry)
        total += len(entry) + 1
    chosen.reverse()
    block = "\n".join(chosen)
    return clip(block, limit) if len(block) > limit else block


# ------------------------------------------------------------------ Runner
class MissionRunner:
    """Plant und führt Missionen aus – synchron (:meth:`run`) oder im Hintergrund (:meth:`start`).

    ``confirm(name, args) -> bool`` ist eine zusätzliche Freigabe für gefährliche Werkzeuge
    (``None`` = allein die Regel von ``brain.tools`` entscheidet)."""

    def __init__(self, brain: Any, store: MissionStore, *, confirm: ConfirmCallback | None = None,
                 max_steps: int = MAX_STEPS, max_failures: int = MAX_FAILED_STEPS):
        self.brain = brain
        self.store = store
        self.confirm = confirm
        self.max_steps = max(1, int(max_steps or MAX_STEPS))
        self.max_failures = max(1, int(max_failures or MAX_FAILED_STEPS))
        self._lock = threading.RLock()
        self._threads: dict[int, threading.Thread] = {}
        self._cancels: dict[int, threading.Event] = {}

    # ------------------------------------------------------------ Hilfen
    @property
    def cfg(self) -> Any:
        return getattr(self.brain, "cfg", None)

    def _projects(self) -> Any:
        return getattr(self.brain, "projects", None)

    def _project_summary(self, project: str | None) -> str:
        projects = self._projects()
        if not project or projects is None:
            return ""
        try:
            ensure = getattr(projects, "ensure", None)
            if callable(ensure):
                ensure(project)
            summary = getattr(projects, "summary", None)
            if callable(summary):
                return str(summary(project) or "")
        except Exception:  # noqa: BLE001 – Projektkontext ist optional
            return ""
        return ""

    def _known_tools(self) -> list[str]:
        tools = getattr(self.brain, "tools", None)
        if tools is None:
            return []
        try:
            return [t.name for t in tools.list()]
        except Exception:  # noqa: BLE001
            return []

    def _tools_block(self) -> str:
        cfg = self.cfg
        tools = getattr(self.brain, "tools", None)
        if tools is None or (cfg is not None and not getattr(cfg, "allow_tools", True)):
            return ""
        try:
            return tools.describe(compact=True)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _notify(progress: ProgressCallback | None, mission: Mission) -> None:
        if progress is None:
            return
        try:
            progress(mission)
        except Exception:  # noqa: BLE001 – ein defekter Rückruf stoppt keine Mission
            pass

    def _save(self, mission: Mission) -> Mission:
        return self.store.save(mission)

    # ------------------------------------------------------------ Planung
    def plan(self, goal: str, *, project: str | None = None) -> Mission:
        """Zerlegt ``goal`` per JSON-Stufe ``mission`` in Schritte und speichert die Mission
        (Status ``geplant``). Liefert das Modell nichts Verwertbares (oder ist nicht erreichbar),
        besteht der Plan aus genau einem ``frage``-Schritt mit dem Ziel selbst."""
        goal = " ".join(str(goal or "").split())
        if not goal:
            raise ValueError("Leeres Missionsziel.")
        if project is not None:
            project = str(project).strip() or None
        cfg = self.cfg
        summary = self._project_summary(project)
        known = self._known_tools()
        messages = mission_plan_messages(goal, summary, self._tools_block())
        parsed: dict | None = None
        error = ""
        json_call = getattr(self.brain, "_json_call", None)
        if callable(json_call) and cfg is not None:
            max_tokens = max(int(getattr(cfg, "max_tokens_json", 500) or 500), MIN_PLAN_TOKENS)
            try:
                outcome = json_call(
                    "mission", messages, MISSION_SCHEMA,
                    lambda t: parse_mission(t, known, goal=goal, max_steps=self.max_steps),
                    model=getattr(cfg, "model", None) or "", max_tokens=max_tokens,
                )
                value = getattr(outcome, "value", outcome)
                if isinstance(value, dict) and value.get("schritte"):
                    parsed = value
            except LLMError as e:
                error = f"Planung ohne Modell: {e}"
        if parsed is None:
            parsed = {"titel": clip(goal, MAX_TITLE_CHARS),
                      "schritte": [{"beschreibung": goal, "art": "frage", "werkzeug": None, "args": {}}]}
        steps = [
            MissionStep(idx=i + 1, description=str(s["beschreibung"]), kind=str(s["art"]),
                        tool=s.get("werkzeug"), args=dict(s.get("args") or {}))
            for i, s in enumerate(parsed["schritte"][:self.max_steps])
        ]
        mission = Mission(id=None, title=str(parsed.get("titel") or clip(goal, MAX_TITLE_CHARS)), goal=goal,
                          project=project, status="geplant", steps=steps, error=error)
        return self._save(mission)

    # ------------------------------------------------------------ Ausführung
    def run(self, mission_id: int, *, progress: ProgressCallback | None = None,
            cancel: threading.Event | None = None) -> Mission:
        """Führt die Mission sequenziell aus und liefert den Endzustand.

        Bereits ``fertig``e Schritte werden übersprungen (Fortsetzen nach Abbruch); alle anderen
        laufen (erneut). Schrittfehler stoppen die Mission erst ab ``max_failures`` Fehlern
        (Status ``fehler``). Ein gesetztes ``cancel`` zwischen den Schritten führt zu
        ``abgebrochen``. Nach jedem Schritt wird gespeichert und ``progress(mission)`` gerufen.
        ``ValueError`` bei unbekannter ID, ``RuntimeError``, wenn sie gerade im Hintergrund läuft."""
        mid = int(mission_id)
        mission = self.store.get(mid)
        if mission is None:
            raise ValueError(f"Unbekannte Mission #{mid}.")
        own_cancel = False
        with self._lock:
            thread = self._threads.get(mid)
            in_worker = thread is not None and threading.current_thread() is thread
            if not in_worker and ((thread is not None and thread.is_alive()) or mid in self._cancels):
                raise RuntimeError(f"Mission #{mid} läuft bereits.")
            if cancel is None:
                cancel = self._cancels.get(mid) or threading.Event()
            if mid not in self._cancels:
                self._cancels[mid] = cancel
                own_cancel = True
        try:
            return self._run_locked(mission, progress, cancel)
        finally:
            if own_cancel:
                with self._lock:
                    if self._cancels.get(mid) is cancel:
                        self._cancels.pop(mid, None)

    def _run_locked(self, mission: Mission, progress: ProgressCallback | None, cancel: threading.Event) -> Mission:
        mission.status = "laeuft"
        mission.error = ""
        mission.report = ""
        for s in mission.steps:
            if s.status != "fertig":
                s.status, s.result, s.duration = "offen", "", 0.0
        self._save(mission)
        self._notify(progress, mission)

        try:
            failed = 0
            aborted_on_failures = False
            for step in mission.steps:
                if cancel.is_set():
                    return self._finish_cancelled(mission, progress)
                if step.status == "fertig":
                    continue
                step.status = "laeuft"
                self._save(mission)
                t0 = time.time()
                try:
                    if step.kind == "werkzeug":
                        ok, text = self._run_tool_step(step)
                    else:
                        ok, text = self._run_question_step(mission, step, cancel)
                except Exception as e:  # noqa: BLE001 – ein Schrittfehler ist nie fatal
                    ok, text = False, f"{e.__class__.__name__}: {e}"
                step.duration = time.time() - t0
                step.result = (text or "").strip()
                if cancel.is_set():
                    step.status = "fehler" if not ok else "fertig"
                    if not ok and not step.result:
                        step.result = CANCELLED_TEXT
                    return self._finish_cancelled(mission, progress)
                step.status = "fertig" if ok else "fehler"
                if not ok:
                    failed += 1
                self._save(mission)
                self._notify(progress, mission)
                if failed >= self.max_failures:
                    aborted_on_failures = True
                    break

            if aborted_on_failures:
                for s in mission.steps:
                    if s.status in ("offen", "laeuft"):
                        s.status = "uebersprungen"
                mission.status = "fehler"
                mission.error = (f"{failed} Schritte fehlgeschlagen (Grenze {self.max_failures}) – "
                                 f"Mission abgebrochen.")
            else:
                mission.status = "fertig"
            self._save(mission)

            report, report_error = self._write_report(mission)
            mission.report = report
            if report_error and not mission.error:
                mission.error = report_error
            self._save(mission)
            self._store_project_note(mission)
            self._save(mission)
        except Exception as e:  # noqa: BLE001 – der Hintergrund-Thread darf nie stumm sterben
            mission.status = "fehler"
            mission.error = f"Interner Fehler: {e.__class__.__name__}: {e}"
            for s in mission.steps:
                if s.status == "laeuft":
                    s.status = "fehler"
                    s.result = s.result or mission.error
            try:
                self._save(mission)
            except Exception:  # noqa: BLE001
                pass
        self._notify(progress, mission)
        return mission

    def _finish_cancelled(self, mission: Mission, progress: ProgressCallback | None) -> Mission:
        for s in mission.steps:
            if s.status in ("offen", "laeuft"):
                s.status = "uebersprungen"
        mission.status = "abgebrochen"
        mission.error = CANCELLED_TEXT
        self._save(mission)
        self._notify(progress, mission)
        return mission

    def _run_tool_step(self, step: MissionStep) -> tuple[bool, str]:
        tools = getattr(self.brain, "tools", None)
        name = step.tool or ""
        if tools is None:
            return False, "Keine Werkzeuge verfügbar."
        tool = tools.get(name)
        if tool is None:
            return False, f"Unbekanntes Werkzeug »{name}«."
        args = dict(step.args or {})
        if getattr(tool, "dangerous", False) and bool(getattr(tools, "confirm_dangerous", True)) \
                and self.confirm is not None:
            try:
                allowed = bool(self.confirm(name, args))
            except Exception as e:  # noqa: BLE001
                return False, f"Freigabe fehlgeschlagen: {e}"
            if not allowed:
                return False, "Vom Nutzer abgelehnt"
        result = tools.run(name, args)
        if result.ok:
            return True, result.output or "(keine Ausgabe)"
        return False, result.error or result.output or "unbekannter Fehler"

    def _run_question_step(self, mission: Mission, step: MissionStep, cancel: threading.Event) -> tuple[bool, str]:
        previous = _results_block(mission.steps)
        question = step.description.strip()
        if previous:
            question = f"{question}\n\nBisherige Ergebnisse:\n{previous}"
        answer = self.brain.ask(question, session_id=f"mission-{mission.id}", project=mission.project,
                                depth="mittel", learn=False, cancel=cancel)
        text = (getattr(answer, "text", "") or "").strip()
        if getattr(answer, "depth", "") == "fehler":
            return False, text or "Modellfehler"
        return bool(text), text or "Leere Antwort"

    def _write_report(self, mission: Mission) -> tuple[str, str]:
        """Bericht via OMEGA (Stufe ``bericht``); bei Modellfehlern ein lokaler Kurzbericht."""
        cfg = self.cfg
        messages = mission_report_messages(mission.goal, mission.steps)
        max_tokens = int(getattr(cfg, "max_tokens_synthesis", 1800) or 1800)
        try:
            chat = getattr(self.brain, "_chat", None)
            if callable(chat):
                res = chat(messages, max_tokens=max_tokens, model=getattr(cfg, "model", None), temperature=0.3)
            else:
                res = self.brain.backend.chat(messages, max_tokens=max_tokens, temperature=0.3)
            text = (getattr(res, "text", "") or "").strip()
            if text:
                return text, ""
            return self._fallback_report(mission), "Bericht: leere Modellantwort – lokaler Kurzbericht."
        except LLMError as e:
            return self._fallback_report(mission), f"Bericht ohne Modell: {e}"
        except Exception as e:  # noqa: BLE001
            return self._fallback_report(mission), f"Bericht fehlgeschlagen: {e.__class__.__name__}: {e}"

    @staticmethod
    def _fallback_report(mission: Mission) -> str:
        lines = [f"## Ziel\n{mission.goal}", "## Ergebnisse je Schritt"]
        for s in mission.steps:
            result = " ".join((s.result or "").split())
            lines.append(f"{s.idx}. {s.description} – {_status_label(s)}"
                         + (f": {clip(result, 300)}" if result else ""))
        open_items = [f"- Schritt {s.idx}: {s.description}" for s in mission.steps if s.status != "fertig"]
        lines.append("## Offene Punkte\n" + ("\n".join(open_items) if open_items else "- keine"))
        lines.append("## Empfehlung\nErgebnisse prüfen; dieser Kurzbericht wurde ohne Modell erzeugt.")
        return "\n".join(lines)

    def _store_project_note(self, mission: Mission) -> None:
        projects = self._projects()
        if projects is None or not mission.project or not mission.report:
            return
        add_note = getattr(projects, "add_note", None)
        if not callable(add_note):
            return
        try:
            add_note(mission.project, "ergebnis", clip(mission.title, 80), mission.report)
        except Exception as e:  # noqa: BLE001 – Notiz ist Bonus, nie fatal
            hint = f"Projektnotiz fehlgeschlagen: {e}"
            mission.error = f"{mission.error}; {hint}" if mission.error else hint

    # ------------------------------------------------------------ Hintergrund
    def start(self, mission_id: int, *, progress: ProgressCallback | None = None) -> threading.Thread:
        """Startet :meth:`run` in einem Daemon-Thread (einer je Mission). ``RuntimeError``, wenn
        die Mission bereits läuft; ``ValueError`` bei unbekannter ID."""
        mid = int(mission_id)
        if self.store.get(mid) is None:
            raise ValueError(f"Unbekannte Mission #{mid}.")
        with self._lock:
            existing = self._threads.get(mid)
            if (existing is not None and existing.is_alive()) or mid in self._cancels:
                raise RuntimeError(f"Mission #{mid} läuft bereits.")
            cancel = threading.Event()
            self._cancels[mid] = cancel

            def worker() -> None:
                try:
                    self.run(mid, progress=progress, cancel=cancel)
                except Exception:  # noqa: BLE001 – run() meldet Fehler über den Status
                    pass
                finally:
                    with self._lock:
                        if self._threads.get(mid) is threading.current_thread():
                            self._threads.pop(mid, None)
                        if self._cancels.get(mid) is cancel:
                            self._cancels.pop(mid, None)

            thread = threading.Thread(target=worker, name=f"obito-mission-{mid}", daemon=True)
            self._threads[mid] = thread
            thread.start()
            return thread

    def stop(self, mission_id: int) -> bool:
        """Bricht eine laufende Mission ab (setzt ``cancel``; der Lauf endet nach dem aktuellen
        Schritt mit Status ``abgebrochen``). Eine nicht laufende Mission mit Status geplant/laeuft/
        pausiert wird direkt auf ``abgebrochen`` gesetzt. ``False``, wenn nichts zu tun war."""
        mid = int(mission_id)
        with self._lock:
            cancel = self._cancels.get(mid)
            thread = self._threads.get(mid)
        if cancel is not None:
            cancel.set()
            return True
        if thread is not None and thread.is_alive():
            return True
        mission = self.store.get(mid)
        if mission is None or mission.status in FINAL_STATUS:
            return False
        for s in mission.steps:
            if s.status in ("offen", "laeuft"):
                s.status = "uebersprungen"
        mission.status = "abgebrochen"
        mission.error = CANCELLED_TEXT
        self._save(mission)
        return True

    def running(self) -> list[int]:
        """IDs der Missionen, die gerade im Hintergrund laufen."""
        with self._lock:
            return sorted(mid for mid, t in self._threads.items() if t.is_alive())

    def join(self, mission_id: int, timeout: float | None = None) -> bool:
        """Wartet auf den Hintergrund-Thread einer Mission. ``True``, wenn er (inzwischen) beendet ist."""
        with self._lock:
            thread = self._threads.get(int(mission_id))
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()


__all__ = [
    "Mission", "MissionStep", "MissionStore", "MissionRunner", "MISSION_SCHEMA",
    "mission_plan_messages", "mission_report_messages", "parse_mission",
    "STEP_KINDS", "STEP_STATUS", "MISSION_STATUS", "MAX_STEPS", "MAX_FAILED_STEPS",
]
