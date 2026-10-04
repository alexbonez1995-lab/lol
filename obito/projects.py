"""Projektsystem von OBITO.

Ein Projekt bündelt alles, was zu einem Vorhaben gehört – lokal in SQLite, mit derselben
Technik wie :mod:`obito.memory` (``check_same_thread=False``, ``RLock``, WAL, FTS5 mit
Triggern, idempotentes ``close()``):

* **Projekte** mit Beschreibung, Status (``aktiv`` | ``pausiert`` | ``archiviert`` | ``fertig``)
  und Tags. Namen sind eindeutig (Groß-/Kleinschreibung und Leerraum werden ignoriert).
* **Notizen** je Projekt in sechs Arten: ``notiz``, ``entscheidung``, ``aufgabe`` (erledigbar),
  ``version``, ``ergebnis`` (z. B. Missionsberichte) und ``problem``. Titel und Inhalt sind
  per Volltext (FTS5/BM25) durchsuchbar.
* **Dateien**, die zum Projekt gehören (Pfad wie angegeben plus aufgelöster Realpfad).
* :meth:`ProjectStore.summary` liefert einen kompakten Projektkontext mit Zeichenbudget für
  den Prompt des Denkkerns.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: Zulässige Projekt-Status.
STATUSES = ("aktiv", "pausiert", "archiviert", "fertig")
#: Zulässige Notiz-Arten.
NOTE_KINDS = ("notiz", "entscheidung", "aufgabe", "version", "ergebnis", "problem")
KINDS = NOTE_KINDS   # Alias analog zu ``memory.KINDS``

MAX_NAME_CHARS = 120
MAX_TITLE_CHARS = 200
MAX_TAGS = 20
MAX_TAG_CHARS = 40
SUMMARY_OPEN_TASKS = 5
SUMMARY_DECISIONS = 3
SUMMARY_ITEM_CHARS = 100
SUMMARY_DESCRIPTION_CHARS = 300
CLIP_SUFFIX = " … [gekürzt]"
MIN_SECTION_CHARS = 24          # kürzer als das wird kein Abschnitt mehr angerissen

# Abschnitts-Beschriftungen der Zusammenfassung (fester Vertrag für Prompt und Tests)
LABEL_OPEN_TASKS = "Offene Aufgaben: "
LABEL_DECISIONS = "Letzte Entscheidungen: "
LABEL_VERSION = "Aktuelle Version: "
LABEL_PROBLEM = "Letztes Problem: "
SECTION_SEP = " | "

_WORD = re.compile(r"\w+", re.UNICODE)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL,
    name_norm     TEXT    NOT NULL UNIQUE,
    description   TEXT    NOT NULL DEFAULT '',
    status        TEXT    NOT NULL DEFAULT 'aktiv',
    tags          TEXT    NOT NULL DEFAULT '',
    created_at    REAL    NOT NULL,
    updated_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_projects_status ON projects(status);

CREATE TABLE IF NOT EXISTS notes (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL,
    kind          TEXT    NOT NULL DEFAULT 'notiz',
    title         TEXT    NOT NULL,
    content       TEXT    NOT NULL DEFAULT '',
    done          INTEGER NOT NULL DEFAULT 0,
    created_at    REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notes_project ON notes(project_id, kind, done, id);

CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    title, content,
    content='notes', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS notes_ai AFTER INSERT ON notes BEGIN
    INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS notes_ad AFTER DELETE ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
        VALUES ('delete', old.id, old.title, old.content);
END;
CREATE TRIGGER IF NOT EXISTS notes_au AFTER UPDATE OF title, content ON notes BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, title, content)
        VALUES ('delete', old.id, old.title, old.content);
    INSERT INTO notes_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
END;

CREATE TABLE IF NOT EXISTS files (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id    INTEGER NOT NULL,
    path          TEXT    NOT NULL,
    realpath      TEXT    NOT NULL,
    description   TEXT    NOT NULL DEFAULT '',
    added_at      REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id, id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_files_project_real ON files(project_id, realpath);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Spaltenliste eines Projekts samt Zählern (offene Aufgaben, Notizen) in einer Abfrage
_PROJECT_SELECT = """
SELECT p.*,
       (SELECT COUNT(*) FROM notes n WHERE n.project_id = p.id AND n.kind = 'aufgabe' AND n.done = 0)
           AS open_tasks,
       (SELECT COUNT(*) FROM notes n WHERE n.project_id = p.id) AS note_count
FROM projects p
"""


def _iso(ts: float | None) -> str | None:
    """Zeitstempel als ISO-8601 in lokaler Zeit (wie ``brain.memory_to_dict``)."""
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _squash(text: object) -> str:
    """Leerraum zusammenziehen; ``None`` wird zu ``""``."""
    if text is None:
        return ""
    return " ".join(str(text).split())


def _clip(text: str, n: int) -> str:
    """Kürzt ``text`` auf ``n`` Zeichen (inklusive Suffix ``" … [gekürzt]"``)."""
    text = "" if text is None else str(text)
    n = max(0, int(n))
    if len(text) <= n:
        return text
    keep = n - len(CLIP_SUFFIX)
    if keep <= 0:
        return text[:n]
    return text[:keep].rstrip() + CLIP_SUFFIX


def normalize_name(name: object) -> str:
    """Vergleichsschlüssel eines Projektnamens: Leerraum zusammengezogen, ``casefold``."""
    return _squash(name).casefold()


def _fts_query(query: str) -> str:
    """FTS5-Ausdruck wie in ``memory.py``: ODER-verknüpft, Präfixsuche für längere Wörter."""
    words = [w for w in _WORD.findall((query or "").lower()) if len(w) > 1]
    parts = []
    for w in dict.fromkeys(words):
        w = w.replace('"', "")
        parts.append(f'"{w[:-1]}"*' if len(w) >= 5 else f'"{w}"')
    return " OR ".join(parts)


def _tags_str(tags: Iterable[object] | None) -> str:
    """Tags bereinigen: getrimmt, klein, ohne Duplikate, alphabetisch (wie ``memory.py``)."""
    if tags is None:
        return ""
    if isinstance(tags, str):
        tags = tags.split(",")
    seen: set[str] = set()
    for t in tags:
        s = _squash(t).lower()[:MAX_TAG_CHARS].strip()
        if s:
            seen.add(s)
    return ",".join(sorted(seen)[:MAX_TAGS])


def _split_tags(raw: str | None) -> list[str]:
    return [t for t in (raw or "").split(",") if t]


@dataclass
class Project:
    """Ein Projekt mit Zählern für offene Aufgaben und Notizen (nur lesend)."""

    id: int
    name: str
    description: str
    status: str
    tags: list[str]
    created_at: float
    updated_at: float
    open_tasks: int = 0
    note_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "beschreibung": self.description,
            "status": self.status,
            "tags": list(self.tags),
            "erstellt": _iso(self.created_at),
            "geaendert": _iso(self.updated_at),
            "offene_aufgaben": int(self.open_tasks),
            "notizen": int(self.note_count),
        }

    def short(self) -> str:
        """Einzeiler für CLI-Listen."""
        tags = f" [{', '.join(self.tags)}]" if self.tags else ""
        tasks = f", {self.open_tasks} offen" if self.open_tasks else ""
        return f"#{self.id} {self.name} ({self.status}{tasks}){tags}"


@dataclass
class Note:
    """Eine Projektnotiz; ``score`` ist nur bei Suchtreffern gesetzt."""

    id: int
    project_id: int
    kind: str
    title: str
    content: str
    done: bool
    created_at: float
    score: float = field(default=0.0, compare=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "projekt_id": self.project_id,
            "art": self.kind,
            "titel": self.title,
            "inhalt": self.content,
            "erledigt": bool(self.done),
            "erstellt": _iso(self.created_at),
        }

    def short(self) -> str:
        """Einzeiler für CLI-Listen, z. B. ``#3 [aufgabe] ☐ Motor bestellen``."""
        mark = ""
        if self.kind == "aufgabe":
            mark = "☑ " if self.done else "☐ "
        date = time.strftime("%d.%m.%Y", time.localtime(self.created_at))
        return f"#{self.id} [{self.kind} · {date}] {mark}{self.title}"


class ProjectStore:
    """Thread-sicherer Zugriff auf Projekte, Notizen und Projektdateien."""

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

    # ------------------------------------------------------------------ intern
    @staticmethod
    def _project_row(row: sqlite3.Row) -> Project:
        keys = row.keys()
        return Project(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            status=row["status"],
            tags=_split_tags(row["tags"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            open_tasks=int(row["open_tasks"]) if "open_tasks" in keys else 0,
            note_count=int(row["note_count"]) if "note_count" in keys else 0,
        )

    @staticmethod
    def _note_row(row: sqlite3.Row, score: float = 0.0) -> Note:
        return Note(
            id=row["id"],
            project_id=row["project_id"],
            kind=row["kind"],
            title=row["title"],
            content=row["content"],
            done=bool(row["done"]),
            created_at=row["created_at"],
            score=score,
        )

    @staticmethod
    def _file_row(row: sqlite3.Row) -> dict:
        real = row["realpath"]
        return {
            "id": row["id"],
            "projekt_id": row["project_id"],
            "pfad": row["path"],
            "realpfad": real,
            "beschreibung": row["description"],
            "hinzugefuegt": _iso(row["added_at"]),
            "existiert": os.path.exists(real),
        }

    @staticmethod
    def _clean_name(name: object) -> str:
        clean = _squash(name)
        if not clean:
            raise ValueError("Der Projektname darf nicht leer sein.")
        if len(clean) > MAX_NAME_CHARS:
            raise ValueError(f"Der Projektname ist zu lang (max. {MAX_NAME_CHARS} Zeichen).")
        return clean

    @staticmethod
    def _check_kind(kind: object) -> str:
        k = _squash(kind).lower()
        if k not in NOTE_KINDS:
            raise ValueError(f"Unbekannte Notiz-Art »{kind}«. Erlaubt: {', '.join(NOTE_KINDS)}.")
        return k

    @staticmethod
    def _check_status(status: object) -> str:
        s = _squash(status).lower()
        if s not in STATUSES:
            raise ValueError(f"Unbekannter Status »{status}«. Erlaubt: {', '.join(STATUSES)}.")
        return s

    def _fetch_by_norm(self, norm: str) -> sqlite3.Row | None:
        return self._db.execute(_PROJECT_SELECT + " WHERE p.name_norm = ?", (norm,)).fetchone()

    def _fetch_by_id(self, pid: int) -> sqlite3.Row | None:
        return self._db.execute(_PROJECT_SELECT + " WHERE p.id = ?", (pid,)).fetchone()

    def _require(self, name: str) -> Project:
        """Projekt zum Namen oder ``ValueError``."""
        row = self._fetch_by_norm(normalize_name(name))
        if row is None:
            raise ValueError(f"Unbekanntes Projekt »{_squash(name)}«.")
        return self._project_row(row)

    def _touch(self, pid: int, now: float | None = None) -> None:
        """``updated_at`` eines Projekts setzen (ohne Commit)."""
        self._db.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now or time.time(), pid))

    # ------------------------------------------------------------- Projekte
    def create(self, name: str, description: str = "", tags: Iterable[str] = ()) -> Project:
        """Legt ein Projekt an. ``ValueError`` bei leerem oder bereits vergebenem Namen."""
        clean = self._clean_name(name)
        norm = normalize_name(clean)
        now = time.time()
        with self._lock:
            if self._fetch_by_norm(norm) is not None:
                raise ValueError(f"Projekt »{clean}« existiert bereits.")
            cur = self._db.execute(
                "INSERT INTO projects(name, name_norm, description, status, tags, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (clean, norm, (description or "").strip(), "aktiv", _tags_str(tags), now, now),
            )
            self._db.commit()
            return self.get_by_id(cur.lastrowid)  # type: ignore[return-value]

    def ensure(self, name: str) -> Project:
        """Vorhandenes Projekt oder – falls es fehlt – ein neu angelegtes."""
        clean = self._clean_name(name)
        with self._lock:
            existing = self.get(clean)
            if existing is not None:
                return existing
            return self.create(clean)

    def get(self, name: str) -> Project | None:
        norm = normalize_name(name)
        if not norm:
            return None
        with self._lock:
            row = self._fetch_by_norm(norm)
        return self._project_row(row) if row else None

    def get_by_id(self, pid: int) -> Project | None:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return None
        with self._lock:
            row = self._fetch_by_id(pid)
        return self._project_row(row) if row else None

    def list(self, include_archived: bool = False) -> list[Project]:
        """Alle Projekte, zuletzt geänderte zuerst; archivierte nur auf Wunsch."""
        sql = _PROJECT_SELECT
        if not include_archived:
            sql += " WHERE p.status != 'archiviert'"
        sql += " ORDER BY p.updated_at DESC, p.id DESC"
        with self._lock:
            rows = self._db.execute(sql).fetchall()
        return [self._project_row(r) for r in rows]

    def update(
        self,
        name: str,
        *,
        description: str | None = None,
        status: str | None = None,
        tags: Iterable[str] | None = None,
    ) -> Project:
        """Ändert Beschreibung, Status und/oder Tags. Nicht übergebene Felder bleiben."""
        sets: list[str] = []
        args: list[object] = []
        if description is not None:
            sets.append("description = ?")
            args.append(str(description).strip())
        if status is not None:
            sets.append("status = ?")
            args.append(self._check_status(status))
        if tags is not None:
            sets.append("tags = ?")
            args.append(_tags_str(tags))
        with self._lock:
            project = self._require(name)
            if sets:
                sets.append("updated_at = ?")
                args.append(time.time())
                self._db.execute(f"UPDATE projects SET {', '.join(sets)} WHERE id = ?", (*args, project.id))
                self._db.commit()
            return self.get_by_id(project.id)  # type: ignore[return-value]

    def rename(self, name: str, new_name: str) -> Project:
        """Benennt ein Projekt um. ``ValueError``, wenn der neue Name leer oder vergeben ist."""
        clean = self._clean_name(new_name)
        norm = normalize_name(clean)
        with self._lock:
            project = self._require(name)
            other = self._fetch_by_norm(norm)
            if other is not None and other["id"] != project.id:
                raise ValueError(f"Projekt »{clean}« existiert bereits.")
            self._db.execute(
                "UPDATE projects SET name = ?, name_norm = ?, updated_at = ? WHERE id = ?",
                (clean, norm, time.time(), project.id),
            )
            self._db.commit()
            return self.get_by_id(project.id)  # type: ignore[return-value]

    def archive(self, name: str) -> Project:
        """Setzt den Status auf ``archiviert`` (idempotent)."""
        return self.update(name, status="archiviert")

    def delete(self, name: str) -> bool:
        """Löscht ein Projekt samt Notizen und Dateieinträgen. ``False``, wenn unbekannt."""
        norm = normalize_name(name)
        with self._lock:
            row = self._fetch_by_norm(norm) if norm else None
            if row is None:
                return False
            pid = row["id"]
            self._db.execute("DELETE FROM notes WHERE project_id = ?", (pid,))
            self._db.execute("DELETE FROM files WHERE project_id = ?", (pid,))
            self._db.execute("DELETE FROM projects WHERE id = ?", (pid,))
            self._db.commit()
            return True

    # -------------------------------------------------------------- Notizen
    def add_note(self, name: str, kind: str, title: str, content: str = "") -> Note:
        """Legt eine Notiz an. ``kind`` muss aus :data:`NOTE_KINDS` sein, der Titel nicht leer."""
        k = self._check_kind(kind)
        clean_title = _squash(title)
        if not clean_title:
            raise ValueError("Der Titel einer Notiz darf nicht leer sein.")
        if len(clean_title) > MAX_TITLE_CHARS:
            clean_title = _clip(clean_title, MAX_TITLE_CHARS)
        body = ("" if content is None else str(content)).strip()
        now = time.time()
        with self._lock:
            project = self._require(name)
            cur = self._db.execute(
                "INSERT INTO notes(project_id, kind, title, content, done, created_at) VALUES (?,?,?,?,0,?)",
                (project.id, k, clean_title, body, now),
            )
            self._touch(project.id, now)
            self._db.commit()
            return self.get_note(cur.lastrowid)  # type: ignore[return-value]

    def get_note(self, note_id: int) -> Note | None:
        try:
            note_id = int(note_id)
        except (TypeError, ValueError):
            return None
        with self._lock:
            row = self._db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
        return self._note_row(row) if row else None

    def notes(
        self,
        name: str,
        kind: str | None = None,
        limit: int = 50,
        include_done: bool = True,
    ) -> list[Note]:
        """Notizen eines Projekts, neueste zuerst. Unbekanntes Projekt → leere Liste."""
        k = self._check_kind(kind) if kind is not None else None
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            limit = 50
        if limit <= 0:
            return []
        sql = "SELECT * FROM notes WHERE project_id = ?"
        args: list[object] = []
        with self._lock:
            project = self.get(name)
            if project is None:
                return []
            args.append(project.id)
            if k is not None:
                sql += " AND kind = ?"
                args.append(k)
            if not include_done:
                sql += " AND done = 0"
            sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
            args.append(limit)
            rows = self._db.execute(sql, args).fetchall()
        return [self._note_row(r) for r in rows]

    def complete(self, note_id: int, done: bool = True) -> Note:
        """Markiert eine Notiz (meist eine Aufgabe) als erledigt bzw. wieder offen."""
        with self._lock:
            note = self.get_note(note_id)
            if note is None:
                raise ValueError(f"Unbekannte Notiz #{note_id}.")
            now = time.time()
            self._db.execute("UPDATE notes SET done = ? WHERE id = ?", (1 if done else 0, note.id))
            self._touch(note.project_id, now)
            self._db.commit()
            return self.get_note(note.id)  # type: ignore[return-value]

    def delete_note(self, note_id: int) -> bool:
        with self._lock:
            note = self.get_note(note_id)
            if note is None:
                return False
            self._db.execute("DELETE FROM notes WHERE id = ?", (note.id,))
            self._touch(note.project_id)
            self._db.commit()
            return True

    # -------------------------------------------------------------- Dateien
    def add_file(self, name: str, path: str | Path, description: str = "") -> dict:
        """Verknüpft eine Datei mit dem Projekt. Der Pfad wird wie angegeben gespeichert,
        zusätzlich sein Realpfad; dieselbe Datei wird je Projekt nur einmal geführt
        (Beschreibung wird dann aktualisiert)."""
        given = str(path).strip() if path is not None else ""
        if not given:
            raise ValueError("Der Dateipfad darf nicht leer sein.")
        real = os.path.realpath(os.path.expanduser(given))
        desc = _squash(description)
        now = time.time()
        with self._lock:
            project = self._require(name)
            row = self._db.execute(
                "SELECT * FROM files WHERE project_id = ? AND realpath = ?", (project.id, real)
            ).fetchone()
            if row is not None:
                if desc and desc != row["description"]:
                    self._db.execute("UPDATE files SET description = ? WHERE id = ?", (desc, row["id"]))
                    self._touch(project.id, now)
                    self._db.commit()
                fid = row["id"]
            else:
                cur = self._db.execute(
                    "INSERT INTO files(project_id, path, realpath, description, added_at) VALUES (?,?,?,?,?)",
                    (project.id, given, real, desc, now),
                )
                self._touch(project.id, now)
                self._db.commit()
                fid = cur.lastrowid
            row = self._db.execute("SELECT * FROM files WHERE id = ?", (fid,)).fetchone()
        return self._file_row(row)

    def files(self, name: str) -> list[dict]:
        """Dateieinträge eines Projekts in Reihenfolge des Hinzufügens."""
        with self._lock:
            project = self.get(name)
            if project is None:
                return []
            rows = self._db.execute(
                "SELECT * FROM files WHERE project_id = ? ORDER BY added_at, id", (project.id,)
            ).fetchall()
        return [self._file_row(r) for r in rows]

    def remove_file(self, file_id: int) -> bool:
        """Entfernt einen Dateieintrag (die Datei selbst bleibt unberührt)."""
        with self._lock:
            row = self._db.execute("SELECT project_id FROM files WHERE id = ?", (file_id,)).fetchone()
            if row is None:
                return False
            self._db.execute("DELETE FROM files WHERE id = ?", (file_id,))
            self._touch(row["project_id"])
            self._db.commit()
            return True

    # ----------------------------------------------------- Zusammenfassung
    @staticmethod
    def _item(note: Note, with_content: bool = False) -> str:
        text = note.title
        if with_content and note.content:
            body = _squash(note.content)
            if body.startswith(note.title):
                text = body                      # Titel ist nur der Anfang des Inhalts
            elif body:
                text = f"{text} – {body}"
        return _clip(text, SUMMARY_ITEM_CHARS)

    def summary(self, name: str, max_chars: int = 800) -> str:
        """Kompakter Projektkontext für den Prompt, höchstens ``max_chars`` Zeichen.

        Aufbau: ``Projekt »X« (status): Beschreibung | Offene Aufgaben: … | Letzte
        Entscheidungen: … | Aktuelle Version: … | Letztes Problem: …`` – nur Abschnitte mit
        Inhalt; bis zu fünf offene Aufgaben, drei Entscheidungen, jeweils die neueste
        Version und das neueste Problem. Unbekanntes Projekt → ``""``."""
        try:
            max_chars = int(max_chars)
        except (TypeError, ValueError):
            max_chars = 800
        if max_chars <= 0:
            return ""
        with self._lock:
            project = self.get(name)
            if project is None:
                return ""
            pid = project.id
            tasks = self._notes_for_summary(pid, "aufgabe", SUMMARY_OPEN_TASKS + 1, open_only=True)
            decisions = self._notes_for_summary(pid, "entscheidung", SUMMARY_DECISIONS)
            versions = self._notes_for_summary(pid, "version", 1)
            problems = self._notes_for_summary(pid, "problem", 1)
            open_total = project.open_tasks

        head = f"Projekt »{project.name}« ({project.status})"
        if project.description:
            head += ": " + _clip(_squash(project.description), SUMMARY_DESCRIPTION_CHARS)

        sections: list[str] = []
        if tasks:
            shown = tasks[:SUMMARY_OPEN_TASKS]
            text = "; ".join(self._item(t) for t in shown)
            rest = open_total - len(shown)
            if rest > 0:
                text += f" (+{rest} weitere)"
            sections.append(LABEL_OPEN_TASKS + text)
        if decisions:
            sections.append(LABEL_DECISIONS + "; ".join(self._item(d) for d in decisions))
        if versions:
            sections.append(LABEL_VERSION + self._item(versions[0], with_content=True))
        if problems:
            sections.append(LABEL_PROBLEM + self._item(problems[0], with_content=True))

        out = _clip(head, max_chars)
        for section in sections:
            remaining = max_chars - len(out) - len(SECTION_SEP)
            if remaining < MIN_SECTION_CHARS:
                break
            out += SECTION_SEP + _clip(section, remaining)
        return out

    def _notes_for_summary(self, pid: int, kind: str, limit: int, open_only: bool = False) -> list[Note]:
        sql = "SELECT * FROM notes WHERE project_id = ? AND kind = ?"
        if open_only:
            sql += " AND done = 0"
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        rows = self._db.execute(sql, (pid, kind, limit)).fetchall()
        return [self._note_row(r) for r in rows]

    # ------------------------------------------------------------- Suche
    def search(self, query: str, k: int = 10) -> list[Note]:
        """Volltextsuche über Titel und Inhalt aller Notizen (BM25), beste zuerst."""
        fts = _fts_query(query)
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = 10
        if not fts or k <= 0:
            return []
        with self._lock:
            rows = self._db.execute(
                "SELECT n.*, bm25(notes_fts, 2.0, 1.0) AS rank FROM notes_fts"
                " JOIN notes n ON n.id = notes_fts.rowid"
                " WHERE notes_fts MATCH ? ORDER BY rank, n.id DESC LIMIT ?",
                (fts, k),
            ).fetchall()
        if not rows:
            return []
        ranks = [-r["rank"] for r in rows]      # bm25: kleiner = besser
        top = max(ranks) or 1.0
        return [self._note_row(r, max(0.0, rk / top)) for r, rk in zip(rows, ranks)]

    # ------------------------------------------------------------- Statistik
    def stats(self) -> dict:
        with self._lock:
            by_status = dict(self._db.execute(
                "SELECT status, COUNT(*) FROM projects GROUP BY status").fetchall())
            by_kind = dict(self._db.execute(
                "SELECT kind, COUNT(*) FROM notes GROUP BY kind").fetchall())
            open_tasks = self._db.execute(
                "SELECT COUNT(*) FROM notes WHERE kind = 'aufgabe' AND done = 0").fetchone()[0]
            n_files = self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {
            "projekte": sum(by_status.values()),
            "aktiv": int(by_status.get("aktiv", 0)),
            "nach_status": {s: int(by_status.get(s, 0)) for s in STATUSES},
            "notizen": sum(by_kind.values()),
            "nach_art": {k: int(by_kind.get(k, 0)) for k in NOTE_KINDS},
            "offene_aufgaben": int(open_tasks),
            "dateien": int(n_files),
        }

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True
