"""Lernen aus Feedback.

Protokolliert jede Interaktion (Frage, Antwort, Kontext, Bewertung, Korrektur) in SQLite
und macht daraus in Stufen eine bessere KI:

* **Lektionen** – kurze Regeln aus Feedback, die sofort in den Kontext wandern
  (``allgemein`` für alle Fragen, ``thema`` per Volltext/Vektor passend zur Frage).
  Jede Bewertung einer Antwort, bei der eine Lektion im Kontext war, zählt für
  (``helped``) oder gegen (``hurt``) die Lektion; schädliche Lektionen werden deaktiviert.
* **Beispiele** – positiv bewertete oder korrigierte Antworten als Few-Shot-Paare.
* **Datensatz** – Export als JSONL (``chat``/``alpaca``/``dpo``) mit Dedup und
  deterministischem Train/Eval-Split für ``obito.training``.

Alles lokal, nur Standardbibliothek, ohne Modell testbar.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .memory import cosine, normalize

Embedder = Callable[[Sequence[str]], "list[list[float]] | None"]

SCOPES = ("allgemein", "thema")
RATINGS = (-1, 0, 1)
FORMATS = ("chat", "alpaca", "dpo")

MAX_LESSON_RENDER_CHARS = 300
MAX_GENERAL_LESSONS = 2
MIN_OWN_EXAMPLES = 20
OWN_PER_CORRECTION = 3
MAX_KEYWORDS = 8
MIN_EXAMPLE_ANSWER_CHARS = 40     # kürzere „beste Antworten“ taugen nicht als Few-Shot-Beispiel
MIN_KEYWORD_CHARS = 5

# Fallback, falls ``obito.training.modelfile`` (parallel entwickelt) nicht importierbar ist.
FALLBACK_SYSTEM_PROMPT = (
    "Du bist OBITO, ein lokaler KI-Assistent. Antworte auf Deutsch, konkret und ehrlich. "
    "Nenne Zahlen mit Einheiten, erfinde keine Fakten und markiere Unsicheres mit „Unsicher:“. "
    "Befolge Erinnerungen und Lektionen aus früherem Feedback und frage bei fehlenden "
    "Informationen gezielt nach."
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL,
    question        TEXT    NOT NULL,
    answer          TEXT    NOT NULL,
    project         TEXT,
    experts         TEXT    NOT NULL DEFAULT '[]',
    depth           TEXT    NOT NULL DEFAULT '',
    model           TEXT    NOT NULL DEFAULT '',
    context         TEXT    NOT NULL DEFAULT '',
    history         TEXT    NOT NULL DEFAULT '[]',
    memories_used   TEXT    NOT NULL DEFAULT '[]',
    new_memories    TEXT    NOT NULL DEFAULT '[]',
    lessons_used    TEXT    NOT NULL DEFAULT '[]',
    tools_used      TEXT    NOT NULL DEFAULT '[]',
    tokens          INTEGER NOT NULL DEFAULT 0,
    duration        REAL    NOT NULL DEFAULT 0.0,
    rating          INTEGER NOT NULL DEFAULT 0,
    comment         TEXT,
    correction      TEXT,
    correction_full TEXT,
    trainable       INTEGER NOT NULL DEFAULT 1,
    created_at      REAL    NOT NULL,
    trace           TEXT    NOT NULL DEFAULT '',
    embedding       TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_session ON interactions(session_id, id);
CREATE INDEX IF NOT EXISTS idx_interactions_rating ON interactions(rating);

CREATE VIRTUAL TABLE IF NOT EXISTS interactions_fts USING fts5(
    question, comment, correction,
    content='interactions', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS interactions_ai AFTER INSERT ON interactions BEGIN
    INSERT INTO interactions_fts(rowid, question, comment, correction)
        VALUES (new.id, new.question, new.comment, new.correction);
END;
CREATE TRIGGER IF NOT EXISTS interactions_ad AFTER DELETE ON interactions BEGIN
    INSERT INTO interactions_fts(interactions_fts, rowid, question, comment, correction)
        VALUES ('delete', old.id, old.question, old.comment, old.correction);
END;
CREATE TRIGGER IF NOT EXISTS interactions_au AFTER UPDATE OF question, comment, correction
ON interactions BEGIN
    INSERT INTO interactions_fts(interactions_fts, rowid, question, comment, correction)
        VALUES ('delete', old.id, old.question, old.comment, old.correction);
    INSERT INTO interactions_fts(rowid, question, comment, correction)
        VALUES (new.id, new.question, new.comment, new.correction);
END;

CREATE TABLE IF NOT EXISTS lessons (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    rule                TEXT    NOT NULL,
    norm                TEXT    NOT NULL,
    scope               TEXT    NOT NULL DEFAULT 'thema',
    topics              TEXT    NOT NULL DEFAULT '',
    source_interaction  INTEGER,
    created_at          REAL    NOT NULL,
    helped              INTEGER NOT NULL DEFAULT 0,
    hurt                INTEGER NOT NULL DEFAULT 0,
    active              INTEGER NOT NULL DEFAULT 1,
    embedding           TEXT
);
CREATE INDEX IF NOT EXISTS idx_lessons_norm ON lessons(norm);
CREATE INDEX IF NOT EXISTS idx_lessons_scope ON lessons(scope, active);

CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
    rule, topics,
    content='lessons', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
    INSERT INTO lessons_fts(rowid, rule, topics) VALUES (new.id, new.rule, new.topics);
END;
CREATE TRIGGER IF NOT EXISTS lessons_ad AFTER DELETE ON lessons BEGIN
    INSERT INTO lessons_fts(lessons_fts, rowid, rule, topics)
        VALUES ('delete', old.id, old.rule, old.topics);
END;
CREATE TRIGGER IF NOT EXISTS lessons_au AFTER UPDATE OF rule, topics ON lessons BEGIN
    INSERT INTO lessons_fts(lessons_fts, rowid, rule, topics)
        VALUES ('delete', old.id, old.rule, old.topics);
    INSERT INTO lessons_fts(rowid, rule, topics) VALUES (new.id, new.rule, new.topics);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

_WORD = re.compile(r"\w+", re.UNICODE)

# Werkzeug-Blöcke im Modelltext (lokale Kopie des Protokolls aus ``obito.tools``, damit dieses
# Modul ohne Werkzeug-Modul auskommt): geschlossene Tags, offener Block am Textende und ein
# ```json-Zaun mit genau {"name", "args"}.
_TOOL_TAG_RE = re.compile(r"<werkzeug\s*>.*?</werkzeug\s*>", re.DOTALL | re.IGNORECASE)
_TOOL_OPEN_RE = re.compile(r"<werkzeug\s*>(?!.*?</werkzeug\s*>).*\Z", re.DOTALL | re.IGNORECASE)
_TOOL_FENCE_RE = re.compile(r"```(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)


def _iso(ts: float) -> str:
    """Zeitstempel als ISO-8601 in lokaler Zeit (wie ``brain.memory_to_dict``)."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _squash(text: str) -> str:
    return " ".join(str(text).split())


def strip_tool_blocks(text: str) -> str:
    """Entfernt Werkzeugaufrufe (``<werkzeug>…</werkzeug>``, offener Block am Ende,
    ```` ```json ````-Zaun mit ``{"name","args"}``) aus einer Antwort."""
    if not text or not isinstance(text, str):
        return text or ""
    if "<werkzeug" in text.lower():
        text = _TOOL_TAG_RE.sub("", text)
        text = _TOOL_OPEN_RE.sub("", text)
    if "```" in text:
        def _fence(m: re.Match) -> str:
            inner = m.group(1).strip()
            if inner.startswith("{"):
                try:
                    obj = json.loads(inner)
                except json.JSONDecodeError:
                    return m.group(0)
                if isinstance(obj, dict) and set(obj.keys()) == {"name", "args"}:
                    return ""
            return m.group(0)
        text = _TOOL_FENCE_RE.sub(_fence, text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def keywords(text: str, limit: int = MAX_KEYWORDS, min_chars: int = MIN_KEYWORD_CHARS) -> list[str]:
    """Bis zu ``limit`` Inhaltswörter (≥ ``min_chars`` Zeichen, keine reinen Zahlen) in
    Reihenfolge des ersten Auftretens – Stichworte für ``evaluate.py``."""
    out: list[str] = []
    for w in _WORD.findall(normalize(text or "")):
        if len(w) < min_chars or w.isdigit() or w in out:
            continue
        out.append(w)
        if len(out) >= limit:
            break
    return out


def _fts_query(query: str) -> str:
    """FTS5-Ausdruck wie in ``memory.py``: ODER-verknüpft, Präfixsuche für längere Wörter."""
    words = [w for w in _WORD.findall((query or "").lower()) if len(w) > 1]
    parts = []
    for w in dict.fromkeys(words):
        w = w.replace('"', "")
        parts.append(f'"{w[:-1]}"*' if len(w) >= 5 else f'"{w}"')
    return " OR ".join(parts)


def _str_list(items: Iterable[Any] | None) -> list[str]:
    out: list[str] = []
    for x in items or ():
        s = str(x).strip() if x is not None else ""
        if s:
            out.append(s)
    return out


def _int_list(items: Iterable[Any] | None) -> list[int]:
    out: list[int] = []
    for x in items or ():
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out


def _history_list(history: Iterable[Any] | None) -> list[dict]:
    """Nur ``{"role", "content"}`` mit Rollen user/assistant und nicht-leerem Text – streng
    abwechselnd und mit ``user`` beginnend (Chat-Vorlagen von Mistral/Gemma lehnen anderes ab)."""
    out: list[dict] = []
    for m in history or ():
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        if not out and role == "assistant":
            continue                                   # führende Assistant-Nachricht weglassen
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + content     # gleiche Rollen zusammenführen
            continue
        out.append({"role": role, "content": content})
    if out and out[-1]["role"] == "user":
        out.pop()                                      # Verlauf endet vor der eigentlichen Frage
    return out


def _json_list(raw: str | None, conv: Callable[[Any], list]) -> list:
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    return conv(data) if isinstance(data, list) else []


def _topics_str(topics: Iterable[Any] | None) -> str:
    seen: dict[str, None] = {}
    for t in topics or ():
        s = _squash(t).lower() if t is not None else ""
        if s:
            seen[s] = None
    return ",".join(seen)


# ------------------------------------------------------------------ Daten
@dataclass
class Interaction:
    """Eine protokollierte Frage-Antwort-Interaktion samt Feedback."""

    id: int
    session_id: str
    question: str
    answer: str
    project: str | None
    experts: list[str]
    depth: str
    model: str
    context: str
    history: list[dict]
    memories_used: list[int]
    new_memories: list[int]
    lessons_used: list[int]
    tools_used: list[str]
    tokens: int
    duration: float
    rating: int
    comment: str | None
    correction: str | None
    correction_full: str | None
    trainable: bool
    created_at: float
    trace: str

    def best_answer(self) -> str:
        """Beste bekannte Antwort: vollständige Korrektur, sonst Korrektur, sonst Antwort."""
        return self.correction_full or self.correction or self.answer

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sitzung": self.session_id,
            "frage": self.question,
            "antwort": self.answer,
            "projekt": self.project,
            "experten": list(self.experts),
            "tiefe": self.depth,
            "modell": self.model,
            "bewertung": self.rating,
            "kommentar": self.comment,
            "korrektur": self.correction,
            "korrektur_voll": self.correction_full,
            "trainierbar": self.trainable,
            "erstellt": _iso(self.created_at),
            "werkzeuge": list(self.tools_used),
            "tokens": self.tokens,
            "dauer": self.duration,
        }


@dataclass
class Lesson:
    """Eine aus Feedback gelernte Regel."""

    id: int
    rule: str
    scope: str                      # "allgemein" | "thema"
    topics: list[str]
    source_interaction: int | None
    created_at: float
    helped: int
    hurt: int
    active: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "regel": self.rule,
            "bereich": self.scope,
            "themen": list(self.topics),
            "quelle": self.source_interaction,
            "erstellt": _iso(self.created_at),
            "geholfen": self.helped,
            "geschadet": self.hurt,
            "aktiv": self.active,
        }

    def render(self) -> str:
        """Kurzform für den Prompt-Kontext (≤ 300 Zeichen):
        „Regel (bestätigt 4×): …“, „Regel (umstritten 2/3): …“ oder „Regel: …“."""
        if self.hurt > 0:
            head = f"Regel (umstritten {self.helped}/{self.hurt}): "
        elif self.helped > 0:
            head = f"Regel (bestätigt {self.helped}×): "
        else:
            head = "Regel: "
        rule = _squash(self.rule)
        room = MAX_LESSON_RENDER_CHARS - len(head)
        if len(rule) > room:
            rule = rule[: max(0, room - 2)].rstrip() + " …"
        return head + rule


# ------------------------------------------------------------------ Store
class LearningStore:
    """Thread-sicherer Zugriff auf Interaktionen, Lektionen und den Datensatz-Export."""

    def __init__(self, path: str | Path, embedder: Embedder | None = None):
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
        self.embedder: Embedder | None = embedder
        self.embed_dim = int(self._meta("embed_dim") or 0)

    # ------------------------------------------------------------ intern
    def _meta(self, key: str) -> str | None:
        row = self._db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        self._db.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", (key, value))
        self._db.commit()

    def set_embedder(self, embedder: Embedder | None) -> None:
        """Setzt (oder entfernt) die Embedding-Funktion; Vektoren anderer Dimension werden
        in der Suche ignoriert."""
        with self._lock:
            self.embedder = embedder

    def _embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        """Vektoren oder ``None`` – Fehler des Embedders sind nie fatal."""
        if not self.embedder or not texts:
            return None
        try:
            vecs = self.embedder(list(texts))
        except Exception:
            return None
        if not vecs or len(vecs) != len(texts):
            return None
        try:
            vecs = [[float(x) for x in v] for v in vecs]
        except (TypeError, ValueError):
            return None
        dim = len(vecs[0])
        if not dim or any(len(v) != dim for v in vecs):
            return None
        if dim != self.embed_dim:
            with self._lock:
                self.embed_dim = dim
                self._set_meta("embed_dim", str(dim))
        return vecs

    def _vec_ok(self, raw: str | None) -> list[float] | None:
        if not raw:
            return None
        try:
            vec = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(vec, list) or (self.embed_dim and len(vec) != self.embed_dim):
            return None
        return vec

    @staticmethod
    def _interaction(row: sqlite3.Row) -> Interaction:
        return Interaction(
            id=row["id"],
            session_id=row["session_id"],
            question=row["question"],
            answer=row["answer"],
            project=row["project"],
            experts=_json_list(row["experts"], _str_list),
            depth=row["depth"],
            model=row["model"],
            context=row["context"],
            history=_json_list(row["history"], _history_list),
            memories_used=_json_list(row["memories_used"], _int_list),
            new_memories=_json_list(row["new_memories"], _int_list),
            lessons_used=_json_list(row["lessons_used"], _int_list),
            tools_used=_json_list(row["tools_used"], _str_list),
            tokens=int(row["tokens"] or 0),
            duration=float(row["duration"] or 0.0),
            rating=int(row["rating"] or 0),
            comment=row["comment"],
            correction=row["correction"],
            correction_full=row["correction_full"],
            trainable=bool(row["trainable"]),
            created_at=float(row["created_at"]),
            trace=row["trace"] or "",
        )

    @staticmethod
    def _lesson(row: sqlite3.Row) -> Lesson:
        return Lesson(
            id=row["id"],
            rule=row["rule"],
            scope=row["scope"],
            topics=[t for t in (row["topics"] or "").split(",") if t],
            source_interaction=row["source_interaction"],
            created_at=float(row["created_at"]),
            helped=int(row["helped"] or 0),
            hurt=int(row["hurt"] or 0),
            active=bool(row["active"]),
        )

    def _require(self, interaction_id: int) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,)).fetchone()
        if row is None:
            raise ValueError(f"Unbekannte Interaktion #{interaction_id}.")
        return row

    # ------------------------------------------------------ Interaktionen
    def record(
        self,
        session_id: str,
        question: str,
        answer: str,
        *,
        project: str | None = None,
        experts: Iterable[str] = (),
        depth: str = "",
        trace: str = "",
        model: str = "",
        context: str = "",
        history: Iterable[dict] = (),
        memories_used: Iterable[int] = (),
        new_memories: Iterable[int] = (),
        lessons_used: Iterable[int] = (),
        tools_used: Iterable[str] = (),
        tokens: int = 0,
        duration: float = 0.0,
    ) -> int:
        """Protokolliert eine Interaktion und liefert ihre ID.

        ``trainable = depth != "fehler" and not tools_used``. Die Frage wird eingebettet,
        falls ein Embedder gesetzt ist (Fehler → kein Vektor)."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Interaktion ohne Frage kann nicht protokolliert werden.")
        answer = answer or ""
        tools = _str_list(tools_used)
        depth = str(depth or "")
        trainable = depth != "fehler" and not tools
        emb = self._embed([question])
        emb_json = json.dumps(emb[0]) if emb else None
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO interactions(session_id, question, answer, project, experts, depth, model,"
                " context, history, memories_used, new_memories, lessons_used, tools_used, tokens,"
                " duration, rating, comment, correction, correction_full, trainable, created_at, trace,"
                " embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,NULL,NULL,NULL,?,?,?,?)",
                (
                    str(session_id or "standard"), question, answer, project,
                    json.dumps(_str_list(experts), ensure_ascii=False), depth, str(model or ""),
                    str(context or ""), json.dumps(_history_list(history), ensure_ascii=False),
                    json.dumps(_int_list(memories_used)), json.dumps(_int_list(new_memories)),
                    json.dumps(_int_list(lessons_used)), json.dumps(tools, ensure_ascii=False),
                    int(tokens or 0), float(duration or 0.0), int(trainable), time.time(),
                    str(trace or ""), emb_json,
                ),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def _shift_lessons(self, lesson_ids: Sequence[int], old: int, new: int) -> None:
        """Verbucht einen Bewertungswechsel ``old → new`` auf den genutzten Lektionen und
        deaktiviert Lektionen mit ``hurt >= 3 and hurt > 2*helped``."""
        if old == new or not lesson_ids:
            return
        for lid in dict.fromkeys(lesson_ids):
            if old == 1:
                self._db.execute("UPDATE lessons SET helped = MAX(0, helped - 1) WHERE id = ?", (lid,))
            elif old == -1:
                self._db.execute("UPDATE lessons SET hurt = MAX(0, hurt - 1) WHERE id = ?", (lid,))
            if new == 1:
                self._db.execute("UPDATE lessons SET helped = helped + 1 WHERE id = ?", (lid,))
            elif new == -1:
                self._db.execute("UPDATE lessons SET hurt = hurt + 1 WHERE id = ?", (lid,))
            self._db.execute(
                "UPDATE lessons SET active = 0 WHERE id = ? AND hurt >= 3 AND hurt > 2 * helped", (lid,)
            )

    def rate(self, interaction_id: int, rating: int, comment: str | None = None) -> Interaction:
        """Bewertet eine Interaktion (-1, 0, 1) und justiert ``helped``/``hurt`` aller
        dabei genutzten Lektionen. Eine frühere Bewertung wird vorher zurückgenommen."""
        try:
            rating = int(rating)
        except (TypeError, ValueError):
            raise ValueError(f"Ungültige Bewertung {rating!r} – erlaubt sind -1, 0, 1.") from None
        if rating not in RATINGS:
            raise ValueError(f"Ungültige Bewertung {rating!r} – erlaubt sind -1, 0, 1.")
        if comment is not None:
            comment = str(comment).strip() or None
        with self._lock:
            row = self._require(int(interaction_id))
            old = int(row["rating"] or 0)
            if comment is not None:
                self._db.execute("UPDATE interactions SET rating = ?, comment = ? WHERE id = ?",
                                 (rating, comment, row["id"]))
            else:
                self._db.execute("UPDATE interactions SET rating = ? WHERE id = ?", (rating, row["id"]))
            self._shift_lessons(_json_list(row["lessons_used"], _int_list), old, rating)
            self._db.commit()
            return self._interaction(self._require(row["id"]))

    def correct(self, interaction_id: int, correction: str,
                correction_full: str | None = None) -> Interaction:
        """Hinterlegt eine Korrektur (und optional die vollständige korrigierte Antwort).
        Eine unbewertete Interaktion (0) wird dabei auf -1 gesetzt."""
        correction = (correction or "").strip()
        if not correction:
            raise ValueError("Leere Korrektur kann nicht gespeichert werden.")
        if correction_full is not None:
            correction_full = str(correction_full).strip() or None
        with self._lock:
            row = self._require(int(interaction_id))
            old = int(row["rating"] or 0)
            new = -1 if old == 0 else old
            self._db.execute(
                "UPDATE interactions SET correction = ?, correction_full = ?, rating = ? WHERE id = ?",
                (correction, correction_full, new, row["id"]),
            )
            self._shift_lessons(_json_list(row["lessons_used"], _int_list), old, new)
            self._db.commit()
            return self._interaction(self._require(row["id"]))

    def get(self, interaction_id: int) -> Interaction | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM interactions WHERE id = ?", (int(interaction_id),)).fetchone()
        return self._interaction(row) if row else None

    def last(self, session_id: str | None = None, n: int = 1) -> list[Interaction]:
        """Die ``n`` jüngsten Interaktionen (neueste zuerst), optional je Sitzung."""
        n = max(0, int(n))
        if n == 0:
            return []
        sql = "SELECT * FROM interactions"
        args: tuple = ()
        if session_id is not None:
            sql += " WHERE session_id = ?"
            args = (session_id,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, args + (n,)).fetchall()
        return [self._interaction(r) for r in rows]

    # ------------------------------------------------------------ Lektionen
    def add_lesson(self, rule: str, *, scope: str = "thema", topics: Iterable[str] = (),
                   source_interaction: int | None = None) -> Lesson:
        """Legt eine Lektion an. Gleiche Regel (``normalize``) → vorhandene Lektion zurück."""
        rule = _squash(rule or "")
        if not rule:
            raise ValueError("Leere Lektion kann nicht gespeichert werden.")
        scope = str(scope or "").strip().lower()
        if scope not in SCOPES:
            scope = "thema"
        norm = normalize(rule)
        topic_str = _topics_str(topics)
        if source_interaction is not None:
            try:
                source_interaction = int(source_interaction)
            except (TypeError, ValueError):
                source_interaction = None
        with self._lock:
            dup = self._db.execute("SELECT * FROM lessons WHERE norm = ?", (norm,)).fetchone()
            if dup is not None:
                if not dup["active"]:
                    # Erneut gelernt: deaktivierte Lektion wieder aktivieren, Schaden-Zähler zurücksetzen
                    self._db.execute(
                        "UPDATE lessons SET active = 1, hurt = 0, scope = ?, topics = CASE WHEN ? != '' THEN ? ELSE topics END,"
                        " source_interaction = COALESCE(?, source_interaction) WHERE id = ?",
                        (scope, topic_str, topic_str, source_interaction, dup["id"]),
                    )
                    self._db.commit()
                    dup = self._db.execute("SELECT * FROM lessons WHERE id = ?", (dup["id"],)).fetchone()
                return self._lesson(dup)
        emb = self._embed([rule + (" " + topic_str.replace(",", " ") if topic_str else "")])
        emb_json = json.dumps(emb[0]) if emb else None
        with self._lock:
            dup = self._db.execute("SELECT * FROM lessons WHERE norm = ?", (norm,)).fetchone()
            if dup is not None:
                return self._lesson(dup)
            cur = self._db.execute(
                "INSERT INTO lessons(rule, norm, scope, topics, source_interaction, created_at,"
                " helped, hurt, active, embedding) VALUES (?,?,?,?,?,?,0,0,1,?)",
                (rule, norm, scope, topic_str, source_interaction, time.time(), emb_json),
            )
            self._db.commit()
            row = self._db.execute("SELECT * FROM lessons WHERE id = ?", (cur.lastrowid,)).fetchone()
            return self._lesson(row)

    def lessons(self, query: str, k: int = 3) -> list[Lesson]:
        """Lektionen für den Kontext: bis zu 2 aktive allgemeine (bestes ``helped-hurt``) plus
        thematische per Volltext (Regel + Themen) und Vektor; insgesamt ≤ ``k``."""
        k = int(k)
        if k <= 0:
            return []
        with self._lock:
            general = self._db.execute(
                "SELECT * FROM lessons WHERE active = 1 AND scope = 'allgemein'"
                " ORDER BY (helped - hurt) DESC, id DESC LIMIT ?",
                (min(MAX_GENERAL_LESSONS, k),),
            ).fetchall()
        result = [self._lesson(r) for r in general]
        remaining = k - len(result)
        if remaining <= 0:
            return result

        candidates: dict[int, dict] = {}
        fts = _fts_query(query)
        qvec = self._embed([query]) if (query or "").strip() else None
        with self._lock:
            if fts:
                rows = self._db.execute(
                    "SELECT l.*, bm25(lessons_fts) AS rank FROM lessons_fts"
                    " JOIN lessons l ON l.id = lessons_fts.rowid"
                    " WHERE lessons_fts MATCH ? AND l.active = 1 AND l.scope = 'thema'"
                    " ORDER BY rank LIMIT 50",
                    (fts,),
                ).fetchall()
                if rows:
                    ranks = [-r["rank"] for r in rows]
                    top = max(ranks) or 1.0
                    for r, rk in zip(rows, ranks):
                        candidates[r["id"]] = {"row": r, "text": max(0.0, rk / top)}
            if qvec:
                for r in self._db.execute(
                    "SELECT * FROM lessons WHERE active = 1 AND scope = 'thema' AND embedding IS NOT NULL"
                ):
                    vec = self._vec_ok(r["embedding"])
                    if vec is None:
                        continue
                    sim = cosine(qvec[0], vec)
                    if sim > 0.3:
                        c = candidates.setdefault(r["id"], {"row": r, "text": 0.0})
                        c["vec"] = sim

        scored: list[tuple[float, int, int, sqlite3.Row]] = []
        for c in candidates.values():
            r = c["row"]
            text = c.get("text", 0.0)
            relevance = 0.6 * c["vec"] + 0.4 * text if "vec" in c else text
            if relevance > 0:
                scored.append((relevance, int(r["helped"]) - int(r["hurt"]), int(r["id"]), r))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        result.extend(self._lesson(r) for _, _, _, r in scored[:remaining])
        return result

    def list_lessons(self, include_inactive: bool = False) -> list[Lesson]:
        sql = "SELECT * FROM lessons" + ("" if include_inactive else " WHERE active = 1") + " ORDER BY id"
        with self._lock:
            rows = self._db.execute(sql).fetchall()
        return [self._lesson(r) for r in rows]

    def delete_lesson(self, lesson_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM lessons WHERE id = ?", (int(lesson_id),))
            self._db.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------ Beispiele
    _EXAMPLE_FILTER = "i.trainable = 1 AND (i.rating >= 1 OR i.correction IS NOT NULL)"

    def examples(self, query: str, k: int = 3) -> list[tuple[str, str]]:
        """Few-Shot-Paare ``(frage, beste_antwort)`` aus positiv bewerteten oder korrigierten,
        trainierbaren Interaktionen; Rang ``0.6·Cosinus + 0.4·BM25`` (ohne Vektor nur BM25)."""
        k = int(k)
        if k <= 0:
            return []
        candidates: dict[int, dict] = {}
        fts = _fts_query(query)
        qvec = self._embed([query]) if (query or "").strip() else None
        with self._lock:
            if fts:
                rows = self._db.execute(
                    "SELECT i.*, bm25(interactions_fts) AS rank FROM interactions_fts"
                    " JOIN interactions i ON i.id = interactions_fts.rowid"
                    f" WHERE interactions_fts MATCH ? AND {self._EXAMPLE_FILTER}"
                    " ORDER BY rank LIMIT 50",
                    (f"question : ({fts})",),
                ).fetchall()
                if rows:
                    ranks = [-r["rank"] for r in rows]
                    top = max(ranks) or 1.0
                    for r, rk in zip(rows, ranks):
                        candidates[r["id"]] = {"row": r, "text": max(0.0, rk / top)}
            if qvec:
                for r in self._db.execute(
                    f"SELECT i.* FROM interactions i WHERE {self._EXAMPLE_FILTER} AND i.embedding IS NOT NULL"
                ):
                    vec = self._vec_ok(r["embedding"])
                    if vec is None:
                        continue
                    sim = cosine(qvec[0], vec)
                    if sim > 0.3:
                        c = candidates.setdefault(r["id"], {"row": r, "text": 0.0})
                        c["vec"] = sim

        scored: list[tuple[float, int, int, Interaction]] = []
        for c in candidates.values():
            text = c.get("text", 0.0)
            relevance = 0.6 * c["vec"] + 0.4 * text if "vec" in c else text
            if relevance <= 0:
                continue
            it = self._interaction(c["row"])
            best = strip_tool_blocks(it.best_answer())
            if not best.strip():
                continue
            if it.correction and not it.correction_full and len(best.strip()) < MIN_EXAMPLE_ANSWER_CHARS:
                # Kurze Korrekturnotizen („Nein, 45 g.“) taugen nicht als Few-Shot-Antwort
                continue
            scored.append((relevance, 1 if it.correction else 0, it.id, it))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        return [(it.question, strip_tool_blocks(it.best_answer())) for _, _, _, it in scored[:k]]

    # ------------------------------------------------------------ Export
    @staticmethod
    def _default_system_prompt() -> str:
        try:
            from .training.modelfile import default_system_prompt  # lazy: parallel entwickelt
            text = default_system_prompt()
            if isinstance(text, str) and text.strip():
                return text
        except Exception:
            pass
        return FALLBACK_SYSTEM_PROMPT

    @staticmethod
    def _split_is_eval(question: str, eval_share: float) -> bool:
        if eval_share <= 0:
            return False
        if eval_share >= 1:
            return True
        digest = hashlib.sha1(normalize(question).encode("utf-8")).hexdigest()
        return int(digest[:8], 16) / 2 ** 32 < eval_share

    def export_dataset(
        self,
        path: str | Path,
        *,
        fmt: str = "chat",
        min_rating: int = 1,
        system_prompt: str | None = None,
        eval_share: float = 0.1,
        seed: int = 42,
        dedup: bool = True,
        min_answer_chars: int = 40,
        include_context: bool = True,
        max_own: int | None = None,
    ) -> dict:
        """Schreibt den Trainingsdatensatz als JSONL.

        Auswahl: ``trainable AND (rating >= min_rating OR correction IS NOT NULL)``; Ziel ist
        ``correction_full or correction or answer`` ohne Werkzeugblöcke. Korrigierte Antworten
        kommen immer hinein, eigene positive höchstens ``max_own`` (Standard: 3 × Korrekturen,
        mindestens 20). Dedup über ``normalize(frage) + sha1(normalize(ziel))`` (neueste/korrigierte
        bleiben). Der Split ist deterministisch über ``sha1(normalize(frage))``.

        Dateien: ``<path>`` (Training), ``<path>.eval.jsonl`` (Eval-Anteil) und
        ``<path>.fragen.jsonl`` (Fragen für ``evaluate.py``) – auch bei 0 Zeilen.
        ``seed`` ist für den Split ohne Wirkung (sha1), bleibt aber für Aufrufer stabil.
        Rückgabe: ``{"train", "eval", "verworfen": {...}}``."""
        fmt = str(fmt or "chat").lower()
        if fmt not in FORMATS:
            raise ValueError(f"Unbekanntes Format {fmt!r} – erlaubt: {', '.join(FORMATS)}.")
        try:
            eval_share = float(eval_share)
        except (TypeError, ValueError):
            eval_share = 0.1
        eval_share = max(0.0, min(1.0, eval_share))
        min_answer_chars = max(0, int(min_answer_chars))
        # ``seed`` bleibt für Aufrufer stabil, der Split braucht ihn nicht (sha1 der Frage).

        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM interactions WHERE rating >= ? OR correction IS NOT NULL"
                " ORDER BY (correction IS NOT NULL) DESC, id DESC",
                (int(min_rating),),
            ).fetchall()
        items = [self._interaction(r) for r in rows]

        rejected = {"duplikat": 0, "zu_kurz": 0, "werkzeug": 0, "fehler": 0, "eigene_limit": 0}
        selected: list[tuple[Interaction, str]] = []
        seen: set[str] = set()
        n_corrected = sum(1 for it in items if it.correction and it.trainable)
        if max_own is None:
            own_limit = max(MIN_OWN_EXAMPLES, OWN_PER_CORRECTION * n_corrected)
        else:
            own_limit = max(0, int(max_own))
        own_count = 0

        for it in items:
            if not it.trainable:
                rejected["werkzeug" if it.tools_used else "fehler"] += 1
                continue
            if fmt == "dpo" and not it.correction:
                continue  # DPO braucht ein Paar (bessere vs. ursprüngliche Antwort)
            target = strip_tool_blocks(it.best_answer())
            if len(target) < min_answer_chars or not target:
                rejected["zu_kurz"] += 1
                continue
            if fmt == "dpo" and normalize(target) == normalize(strip_tool_blocks(it.answer)):
                rejected["duplikat"] += 1  # chosen == rejected ist kein Lernsignal
                continue
            if dedup:
                key = normalize(it.question) + hashlib.sha1(normalize(target).encode("utf-8")).hexdigest()
                if key in seen:
                    rejected["duplikat"] += 1
                    continue
                seen.add(key)
            if not it.correction:
                if own_count >= own_limit:
                    rejected["eigene_limit"] += 1
                    continue
                own_count += 1
            selected.append((it, target))

        selected.sort(key=lambda t: t[0].id)
        sys_prompt = system_prompt if system_prompt is not None else self._default_system_prompt()

        def _prompt(it: Interaction) -> str:
            if include_context and it.context:
                return it.context + "\n\n" + it.question
            return it.question

        def _row(it: Interaction, target: str) -> dict:
            if fmt == "chat":
                messages = [{"role": "system", "content": sys_prompt}]
                messages.extend({"role": m["role"], "content": m["content"]} for m in it.history)
                messages.append({"role": "user", "content": _prompt(it)})
                messages.append({"role": "assistant", "content": target})
                return {"messages": messages}
            if fmt == "alpaca":
                return {"instruction": it.question, "input": it.context if include_context else "",
                        "output": target}
            return {"prompt": _prompt(it), "chosen": target, "rejected": strip_tool_blocks(it.answer)}

        train: list[dict] = []
        evals: list[dict] = []
        eval_items: list[tuple[Interaction, str]] = []
        for it, target in selected:
            if self._split_is_eval(it.question, eval_share):
                evals.append(_row(it, target))
                eval_items.append((it, target))
            else:
                train.append(_row(it, target))

        questions = [
            {"frage": it.question, "erwartet": target, "stichworte": keywords(target), "quelle_id": it.id}
            for it, target in (eval_items or selected)
        ]

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_jsonl(path, train)
        self._write_jsonl(Path(str(path) + ".eval.jsonl"), evals)
        self._write_jsonl(Path(str(path) + ".fragen.jsonl"), questions)
        return {"train": len(train), "eval": len(evals), "verworfen": rejected}

    @staticmethod
    def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ Statistik
    def stats(self) -> dict:
        with self._lock:
            r = self._db.execute(
                "SELECT COUNT(*) AS n,"
                " SUM(rating != 0) AS bewertet,"
                " SUM(rating = 1) AS positiv,"
                " SUM(rating = -1) AS negativ,"
                " SUM(correction IS NOT NULL) AS korrigiert,"
                " SUM(trainable = 1) AS trainierbar FROM interactions"
            ).fetchone()
            lessons = self._db.execute(
                "SELECT COUNT(*) AS n, SUM(active = 1) AS aktiv FROM lessons"
            ).fetchone()
        return {
            "interaktionen": int(r["n"] or 0),
            "bewertet": int(r["bewertet"] or 0),
            "positiv": int(r["positiv"] or 0),
            "negativ": int(r["negativ"] or 0),
            "korrigiert": int(r["korrigiert"] or 0),
            "trainierbar": int(r["trainierbar"] or 0),
            "lektionen": int(lessons["n"] or 0),
            "lektionen_aktiv": int(lessons["aktiv"] or 0),
        }

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
