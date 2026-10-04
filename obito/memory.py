"""Gedächtnis-System von OBITO.

Dauerhaftes, durchsuchbares Gedächtnis auf Basis von SQLite (lokal, offline):

* Langzeit-Erinnerungen (Fakten, Präferenzen, Entscheidungen, Lösungen, Fehler)
  mit Wichtigkeit, Zeitstempel, Quelle und optionalem Projekt.
* Gesprächsverlauf pro Sitzung als Kurzzeitgedächtnis.
* Hybride Suche: Volltext (FTS5/BM25) + semantische Vektoren (falls ein
  Embedding-Modell verfügbar ist) + Wichtigkeit/Aktualität.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

KINDS = ("fakt", "praeferenz", "entscheidung", "loesung", "fehler", "zusammenfassung", "notiz")

Embedder = Callable[[Sequence[str]], "list[list[float]] | None"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT    NOT NULL DEFAULT 'notiz',
    content       TEXT    NOT NULL,
    norm          TEXT    NOT NULL,
    tags          TEXT    NOT NULL DEFAULT '',
    project       TEXT,
    source        TEXT    NOT NULL DEFAULT 'nutzer',
    importance    REAL    NOT NULL DEFAULT 0.5,
    created_at    REAL    NOT NULL,
    last_access   REAL    NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    embedding     TEXT
);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project);
CREATE INDEX IF NOT EXISTS idx_memories_norm ON memories(norm);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content, tags,
    content='memories', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
        VALUES ('delete', old.id, old.content, old.tags);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE OF content, tags ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags)
        VALUES ('delete', old.id, old.content, old.tags);
    INSERT INTO memories_fts(rowid, content, tags) VALUES (new.id, new.content, new.tags);
END;

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    project     TEXT,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

_WORD = re.compile(r"\w+", re.UNICODE)


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class Memory:
    id: int
    kind: str
    content: str
    tags: list[str]
    project: str | None
    source: str
    importance: float
    created_at: float
    last_access: float
    access_count: int
    score: float = 0.0
    embedding: list[float] | None = field(default=None, repr=False)

    def short(self) -> str:
        date = time.strftime("%d.%m.%Y", time.localtime(self.created_at))
        proj = f" · {self.project}" if self.project else ""
        return f"#{self.id} [{self.kind}{proj} · {date}] {self.content}"


class MemoryStore:
    """Thread-sicherer Zugriff auf das persistente Gedächtnis."""

    def __init__(self, path: str | Path, embedder: Embedder | None = None):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # ------------------------------------------------------------------ intern
    def _embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        if not self.embedder:
            return None
        try:
            vecs = self.embedder(list(texts))
        except Exception:
            return None
        if not vecs or len(vecs) != len(texts):
            return None
        return vecs

    @staticmethod
    def _row(row: sqlite3.Row, score: float = 0.0) -> Memory:
        emb = json.loads(row["embedding"]) if row["embedding"] else None
        return Memory(
            id=row["id"],
            kind=row["kind"],
            content=row["content"],
            tags=[t for t in row["tags"].split(",") if t],
            project=row["project"],
            source=row["source"],
            importance=row["importance"],
            created_at=row["created_at"],
            last_access=row["last_access"],
            access_count=row["access_count"],
            score=score,
            embedding=emb,
        )

    # -------------------------------------------------------------- schreiben
    def remember(
        self,
        content: str,
        kind: str = "notiz",
        tags: Iterable[str] = (),
        project: str | None = None,
        source: str = "nutzer",
        importance: float = 0.5,
    ) -> Memory:
        """Speichert eine Erinnerung. Duplikate werden zusammengeführt statt doppelt abgelegt."""
        content = content.strip()
        if not content:
            raise ValueError("Leere Erinnerung kann nicht gespeichert werden.")
        if kind not in KINDS:
            kind = "notiz"
        importance = max(0.0, min(1.0, float(importance)))
        norm = normalize(content)
        tag_str = ",".join(sorted({t.strip().lower() for t in tags if t.strip()}))
        now = time.time()
        emb = self._embed([content])
        emb_vec = emb[0] if emb else None

        with self._lock:
            dup = self._find_duplicate(norm, emb_vec, project)
            if dup is not None:
                self._db.execute(
                    "UPDATE memories SET importance = MIN(1.0, MAX(importance, ?) + 0.05),"
                    " last_access = ?, access_count = access_count + 1 WHERE id = ?",
                    (importance, now, dup),
                )
                self._db.commit()
                return self.get(dup)  # type: ignore[return-value]

            cur = self._db.execute(
                "INSERT INTO memories(kind, content, norm, tags, project, source, importance,"
                " created_at, last_access, embedding) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (kind, content, norm, tag_str, project, source, importance, now, now,
                 json.dumps(emb_vec) if emb_vec else None),
            )
            self._db.commit()
            return self.get(cur.lastrowid)  # type: ignore[return-value]

    def _find_duplicate(self, norm: str, emb: list[float] | None, project: str | None) -> int | None:
        row = self._db.execute(
            "SELECT id FROM memories WHERE norm = ? AND project IS ?", (norm, project)
        ).fetchone()
        if row:
            return row["id"]
        if emb is None:
            return None
        for r in self._db.execute(
            "SELECT id, embedding FROM memories WHERE embedding IS NOT NULL AND project IS ?",
            (project,),
        ):
            if cosine(emb, json.loads(r["embedding"])) >= 0.97:
                return r["id"]
        return None

    def forget(self, memory_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            self._db.commit()
            return cur.rowcount > 0

    def update_importance(self, memory_id: int, delta: float) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE memories SET importance = MIN(1.0, MAX(0.0, importance + ?)) WHERE id = ?",
                (delta, memory_id),
            )
            self._db.commit()

    # ----------------------------------------------------------------- lesen
    def get(self, memory_id: int) -> Memory | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row(row) if row else None

    def recent(self, limit: int = 10, project: str | None = None) -> list[Memory]:
        sql = "SELECT * FROM memories"
        args: tuple = ()
        if project:
            sql += " WHERE project = ? OR project IS NULL"
            args = (project,)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(sql, args + (limit,)).fetchall()
        return [self._row(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

    def stats(self) -> dict:
        with self._lock:
            kinds = dict(self._db.execute("SELECT kind, COUNT(*) FROM memories GROUP BY kind").fetchall())
            projects = [r[0] for r in self._db.execute(
                "SELECT DISTINCT project FROM memories WHERE project IS NOT NULL ORDER BY project")]
            msgs = self._db.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            with_vec = self._db.execute(
                "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL").fetchone()[0]
        return {
            "erinnerungen": sum(kinds.values()),
            "nach_art": kinds,
            "projekte": projects,
            "nachrichten": msgs,
            "mit_vektor": with_vec,
        }

    @staticmethod
    def _fts_query(query: str) -> str:
        words = [w for w in _WORD.findall(query.lower()) if len(w) > 1]
        parts = []
        for w in dict.fromkeys(words):
            w = w.replace('"', "")
            # Präfixsuche für längere Wörter: "drohne" findet auch "drohnen"
            parts.append(f'"{w[:-1]}"*' if len(w) >= 5 else f'"{w}"')
        return " OR ".join(parts)

    def search(
        self,
        query: str,
        k: int = 6,
        project: str | None = None,
        min_score: float = 0.05,
        touch: bool = True,
    ) -> list[Memory]:
        """Hybride Suche: Volltext + Vektor + Wichtigkeit + Aktualität."""
        candidates: dict[int, dict] = {}
        fts = self._fts_query(query)
        proj_sql = " AND (m.project = ? OR m.project IS NULL)" if project else ""
        proj_args: tuple = (project,) if project else ()

        with self._lock:
            if fts:
                rows = self._db.execute(
                    "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts"
                    " JOIN memories m ON m.id = memories_fts.rowid"
                    f" WHERE memories_fts MATCH ?{proj_sql} ORDER BY rank LIMIT 50",
                    (fts,) + proj_args,
                ).fetchall()
                if rows:
                    ranks = [-r["rank"] for r in rows]  # bm25: kleiner = besser
                    top = max(ranks) or 1.0
                    for r, rk in zip(rows, ranks):
                        candidates[r["id"]] = {"row": r, "text": max(0.0, rk / top)}

            qvec = self._embed([query])
            if qvec:
                where = " WHERE embedding IS NOT NULL" + (
                    " AND (project = ? OR project IS NULL)" if project else "")
                for r in self._db.execute(f"SELECT * FROM memories{where}", proj_args):
                    sim = cosine(qvec[0], json.loads(r["embedding"]))
                    if sim > 0.3:
                        c = candidates.setdefault(r["id"], {"row": r, "text": 0.0})
                        c["vec"] = sim

        now = time.time()
        scored: list[Memory] = []
        for c in candidates.values():
            r = c["row"]
            age_days = (now - r["last_access"]) / 86400
            recency = math.exp(-age_days / 30)
            text, vec = c.get("text", 0.0), c.get("vec", 0.0)
            relevance = 0.6 * vec + 0.4 * text if qvec else text
            score = 0.75 * relevance + 0.15 * r["importance"] + 0.10 * recency
            if relevance > 0 and score >= min_score:
                scored.append(self._row(r, score))

        scored.sort(key=lambda m: m.score, reverse=True)
        result = scored[:k]
        if touch and result:
            with self._lock:
                self._db.executemany(
                    "UPDATE memories SET last_access = ?, access_count = access_count + 1 WHERE id = ?",
                    [(now, m.id) for m in result],
                )
                self._db.commit()
        return result

    # ------------------------------------------------------- Gesprächsverlauf
    def add_message(self, session_id: str, role: str, content: str, project: str | None = None) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO messages(session_id, role, content, project, created_at) VALUES (?,?,?,?,?)",
                (session_id, role, content, project, time.time()),
            )
            self._db.commit()

    def history(self, session_id: str, limit: int = 12) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    # ------------------------------------------------------------ Export
    def export(self, path: str | Path) -> int:
        with self._lock:
            rows = self._db.execute("SELECT * FROM memories ORDER BY id").fetchall()
        data = [
            {k: r[k] for k in ("id", "kind", "content", "tags", "project", "source",
                               "importance", "created_at", "access_count")}
            for r in rows
        ]
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(data)

    def import_json(self, path: str | Path) -> int:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for d in data:
            self.remember(
                d["content"], kind=d.get("kind", "notiz"),
                tags=(d.get("tags") or "").split(","), project=d.get("project"),
                source=d.get("source", "import"), importance=d.get("importance", 0.5),
            )
        return len(data)

    def close(self) -> None:
        with self._lock:
            self._db.close()
