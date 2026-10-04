"""Datenzentrum von OBITO: Dokumente indexieren und semantisch durchsuchen (RAG).

Lokale Wissensbasis auf SQLite-Basis (gleiche Technik wie :mod:`obito.memory`):

* :func:`extract_text` holt Text aus Text-/Code-Dateien, HTML/XML, Office-Dateien
  (.docx/.xlsx/.pptx über ``zipfile`` + ``xml.etree``) und – falls ``pypdf``
  installiert ist – aus PDFs. Alles ohne Zusatzabhängigkeiten.
* :func:`chunk_text` teilt Text deterministisch an Absatz-/Satzgrenzen in überlappende
  Abschnitte, nie mitten im Wort.
* :class:`KnowledgeStore` speichert Dokumente und Abschnitte, erkennt unveränderte Dateien
  per SHA-256, hält Verzeichnisse per :meth:`KnowledgeStore.sync` aktuell und sucht hybrid
  (BM25-Volltext + Kosinus-Ähnlichkeit, falls ein Embedding-Modell verfügbar ist).
* :func:`register_tools` stellt dem Modell ``dokumente_suchen`` und ``dokument_hinzufuegen``
  zur Verfügung; Dateien werden nur innerhalb des Arbeitsbereichs der ``ToolRegistry``
  hinzugefügt.
"""

from __future__ import annotations

import fnmatch
import hashlib
import html
import json
import os
import re
import sqlite3
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Sequence
from xml.etree import ElementTree as ET

from .memory import cosine

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfer
    from .tools import ToolRegistry

Embedder = Callable[[Sequence[str]], "list[list[float]] | None"]

#: Dateiendungen, die :func:`extract_text` und :meth:`KnowledgeStore.add_directory` kennen.
SUPPORTED = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".log",
             ".py", ".js", ".ts", ".html", ".htm", ".xml", ".c", ".cpp", ".h", ".ino", ".ps1", ".bat", ".sh",
             ".docx", ".xlsx", ".pptx", ".pdf"}

#: Verzeichnisse, die :meth:`KnowledgeStore.add_directory` nie betritt (zusätzlich alle versteckten).
SKIP_DIRS = {".git", "__pycache__", "node_modules"}

MAX_FILE_BYTES = 50 * 1024 * 1024     # größere Dateien werden nicht indexiert
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100
EMBED_BATCH = 32
MAX_SNIPPET_CHARS = 400               # je Treffer in der Werkzeugausgabe
CLIP_SUFFIX = " … [gekürzt]"
CONTEXT_HEADER = "Auszüge aus deinen Dokumenten (Quelle in eckigen Klammern):"

_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE_END = re.compile(r"(?<=[.!?…;:])\s+")
_HTML_DROP = re.compile(r"<(script|style|head|noscript|template)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BLOCK = re.compile(
    r"</?(p|div|ul|ol|table|h[1-6]|section|article|header|footer|nav|blockquote|pre|"
    r"dl|figure|figcaption|title|main|aside|form|fieldset|legend|address|summary|details)\b[^>]*>",
    re.IGNORECASE,
)
# Zeilen-Elemente: nur der öffnende Tag (bzw. br/hr) erzeugt einen Umbruch, Zellen ein Leerzeichen
_HTML_LINE = re.compile(r"<(br|hr|li|tr|dt|dd|option)\b[^>]*>", re.IGNORECASE)
_HTML_CELL = re.compile(r"<(td|th)\b[^>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_XML_TAG_CLOSE = re.compile(r">\s*<")
_SLIDE_NAME = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
_CELL_REF = re.compile(r"^([A-Z]+)(\d*)$")
_WS_RUN = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT    NOT NULL,
    path        TEXT,
    kind        TEXT    NOT NULL DEFAULT 'text',
    size        INTEGER NOT NULL DEFAULT 0,
    hash        TEXT    NOT NULL,
    project     TEXT,
    added_at    REAL    NOT NULL,
    mtime       REAL,
    missing     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_documents_path ON documents(path);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(hash);
CREATE INDEX IF NOT EXISTS idx_documents_project ON documents(project);

CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    idx         INTEGER NOT NULL,
    title       TEXT    NOT NULL DEFAULT '',
    content     TEXT    NOT NULL,
    embedding   TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id, idx);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, title,
    content='chunks', content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content, title) VALUES (new.id, new.content, new.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, title)
        VALUES ('delete', old.id, old.content, old.title);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE OF content, title ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, content, title)
        VALUES ('delete', old.id, old.content, old.title);
    INSERT INTO chunks_fts(rowid, content, title) VALUES (new.id, new.content, new.title);
END;

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# ------------------------------------------------------------------ Hilfen
def _iso(ts: float | None) -> str | None:
    """Zeitstempel als ISO-8601 in lokaler Zeit (``None`` bleibt ``None``)."""
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _clip(text: str, n: int) -> str:
    """Kürzt ``text`` auf ``n`` Zeichen und hängt :data:`CLIP_SUFFIX` an (wie ``agents.clip``)."""
    text = "" if text is None else str(text)
    n = max(0, int(n))
    if len(text) <= n:
        return text
    return text[:n].rstrip() + CLIP_SUFFIX


def _one_line(text: str) -> str:
    """Faltet beliebigen Leerraum auf einzelne Leerzeichen zusammen."""
    return " ".join((text or "").split())


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("Bytes", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            if unit == "Bytes":
                return f"{int(n)} Bytes"
            return f"{n:.1f} {unit}".replace(".", ",")
        n /= 1024
    return f"{n:.1f} GB".replace(".", ",")


def _fts_query(query: str) -> str:
    """FTS5-Ausdruck aus freiem Text: OR-verknüpfte Wörter, Präfixsuche für längere Wörter."""
    words = [w for w in _WORD.findall((query or "").lower()) if len(w) > 1]
    parts = []
    for w in dict.fromkeys(words):
        w = w.replace('"', "")
        parts.append(f'"{w[:-1]}"*' if len(w) >= 5 else f'"{w}"')
    return " OR ".join(parts)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _local(tag: str) -> str:
    """Lokaler Name eines XML-Tags ohne Namespace (``{ns}p`` → ``p``)."""
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clean_text(text: str) -> str:
    """Normalisiert Zeilenenden und Leerraum, behält Absätze (maximal eine Leerzeile)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [_WS_RUN.sub(" ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


# ------------------------------------------------------------------ Datenklassen
@dataclass
class Document:
    """Ein indexiertes Dokument (Datei oder direkt übergebener Text)."""

    id: int
    title: str
    path: str | None
    kind: str
    size: int
    hash: str
    project: str | None
    added_at: float
    mtime: float | None
    chunks: int
    missing: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "titel": self.title,
            "pfad": self.path,
            "art": self.kind,
            "groesse": self.size,
            "projekt": self.project,
            "hinzugefuegt": _iso(self.added_at),
            "geaendert": _iso(self.mtime),
            "abschnitte": self.chunks,
            "fehlt": bool(self.missing),
        }

    def short(self) -> str:
        proj = f" · {self.project}" if self.project else ""
        flag = " · FEHLT" if self.missing else ""
        where = f" ({self.path})" if self.path else ""
        return f"#{self.id} [{self.kind}{proj}{flag}] {self.title}{where} – {self.chunks} Abschnitte"


@dataclass
class Chunk:
    """Ein Abschnitt eines Dokuments. ``idx`` zählt ab 1 und erscheint im Zitat als ``§idx``."""

    id: int
    doc_id: int
    title: str
    idx: int
    content: str
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "dokument_id": self.doc_id,
            "titel": self.title,
            "abschnitt": self.idx,
            "inhalt": self.content,
            "score": round(float(self.score), 4),
        }

    def cite(self) -> str:
        return f"[{self.title} §{self.idx}]"


# ------------------------------------------------------------------ Extraktion
def _check_file(path: Path) -> int:
    """Prüft Existenz, Typ und Größe; liefert die Dateigröße in Bytes."""
    if not path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Ist ein Verzeichnis: {path}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Datei zu groß ({_fmt_bytes(size)}, max. {_fmt_bytes(MAX_FILE_BYTES)}): {path.name}")
    return size


def _read_bytes(path: Path) -> bytes:
    _check_file(path)
    with open(path, "rb") as fh:
        return fh.read()


def _decode(data: bytes, name: str) -> str:
    """Dekodiert Textdaten (UTF-8 mit BOM-Erkennung, Fehler ersetzt). Nullbytes → Binärdatei."""
    if b"\x00" in data[:8192]:
        raise ValueError(f"Binärdatei ohne lesbaren Text: {name}")
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def _strip_html(raw: str) -> str:
    """Entfernt HTML-Tags: Skripte/Styles weg, Blockelemente werden Zeilenumbrüche, Entities aufgelöst."""
    text = _HTML_COMMENT.sub(" ", raw)
    text = _HTML_DROP.sub(" ", text)
    text = _HTML_BLOCK.sub("\n", text)
    text = _HTML_LINE.sub("\n", text)
    text = _HTML_CELL.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _clean_text("\n".join(line.strip() for line in text.split("\n")))


def _strip_xml(raw: str) -> str:
    """Entfernt XML-Tags; Elementgrenzen werden Zeilenumbrüche, Entities aufgelöst."""
    text = _HTML_COMMENT.sub(" ", raw)
    text = re.sub(r"<\?.*?\?>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", text, flags=re.DOTALL)
    text = _XML_TAG_CLOSE.sub(">\n<", text)
    text = _TAG.sub(" ", text)
    text = html.unescape(text)
    return _clean_text("\n".join(line.strip() for line in text.split("\n")))


def _zip_open(path: Path, kind: str) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        raise ValueError(f"Keine gültige {kind}-Datei (kein ZIP-Container): {path.name}") from None


def _zip_xml(zf: zipfile.ZipFile, name: str, kind: str) -> ET.Element:
    try:
        data = zf.read(name)
    except KeyError:
        raise ValueError(f"Ungültige {kind}-Datei, {name} fehlt: {os.path.basename(zf.filename or '')}") from None
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise ValueError(f"Fehlerhaftes XML in {kind}-Datei ({name}): {e}") from None


def _paragraph_text(para: ET.Element, text_tag: str, tab_tag: str | None = None, br_tag: str | None = None) -> str:
    """Text eines Absatzes: alle Textknoten in Dokumentreihenfolge, Tabs/Umbrüche berücksichtigt."""
    parts: list[str] = []
    for el in para.iter():
        name = _local(el.tag)
        if name == text_tag:
            parts.append(el.text or "")
        elif tab_tag and name == tab_tag:
            parts.append("\t")
        elif br_tag and name == br_tag:
            parts.append("\n")
    return "".join(parts)


def _extract_docx(path: Path) -> str:
    with _zip_open(path, "DOCX") as zf:
        root = _zip_xml(zf, "word/document.xml", "DOCX")
    lines: list[str] = []

    def walk(el: ET.Element) -> None:
        # ein Absatz wird komplett übernommen (inkl. verschachtelter Textfelder), nicht doppelt
        if _local(el.tag) == "p":
            lines.append(_paragraph_text(el, "t", tab_tag="tab", br_tag="br"))
            return
        for child in el:
            walk(child)

    walk(root)
    return "\n".join(lines)


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = _zip_xml(zf, "xl/sharedStrings.xml", "XLSX")
    out: list[str] = []
    for si in root:
        if _local(si.tag) != "si":
            continue
        out.append("".join(t.text or "" for t in si.iter() if _local(t.tag) == "t"))
    return out


def _xlsx_sheets(zf: zipfile.ZipFile) -> list[tuple[str, str]]:
    """Blattnamen und Pfade in Arbeitsmappen-Reihenfolge; Fallback: alle sheet*.xml sortiert."""
    names = set(zf.namelist())
    sheets: list[tuple[str, str]] = []
    if "xl/workbook.xml" in names and "xl/_rels/workbook.xml.rels" in names:
        try:
            wb = _zip_xml(zf, "xl/workbook.xml", "XLSX")
            rels = _zip_xml(zf, "xl/_rels/workbook.xml.rels", "XLSX")
            targets: dict[str, str] = {}
            for rel in rels:
                rid = rel.get("Id")
                target = rel.get("Target") or ""
                if rid and target:
                    target = target.lstrip("/")
                    if not target.startswith("xl/"):
                        target = "xl/" + target
                    targets[rid] = target
            for sheet in wb.iter():
                if _local(sheet.tag) != "sheet":
                    continue
                rid = next((v for k, v in sheet.attrib.items() if _local(k) == "id"), None)
                target = targets.get(rid or "")
                if target and target in names:
                    sheets.append((sheet.get("name") or f"Blatt{len(sheets) + 1}", target))
        except ValueError:
            sheets = []
    if not sheets:
        def _num(n: str) -> int:
            m = re.search(r"(\d+)\.xml$", n)
            return int(m.group(1)) if m else 0
        for name in sorted((n for n in names if re.match(r"^xl/worksheets/sheet\d*\.xml$", n)), key=_num):
            sheets.append((f"Blatt{len(sheets) + 1}", name))
    return sheets


def _col_index(ref: str | None) -> int | None:
    """Spaltenindex (0-basiert) aus einer Zellreferenz wie ``C7``."""
    if not ref:
        return None
    m = _CELL_REF.match(ref.strip().upper())
    if not m:
        return None
    n = 0
    for ch in m.group(1):
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def _xlsx_cell_value(cell: ET.Element, shared: list[str]) -> str:
    typ = cell.get("t") or ""
    value: str | None = None
    inline_parts: list[str] = []
    for child in cell:
        name = _local(child.tag)
        if name == "v":
            value = child.text or ""
        elif name == "is":
            inline_parts.extend(t.text or "" for t in child.iter() if _local(t.tag) == "t")
    if typ == "inlineStr":
        return "".join(inline_parts)
    if value is None:
        return ""
    if typ == "s":
        try:
            return shared[int(value)]
        except (ValueError, IndexError):
            return value
    if typ == "b":
        return "WAHR" if value.strip() == "1" else "FALSCH"
    # Zahlen: 3.0 -> "3", sonst unverändert (Datumswerte bleiben serielle Zahlen)
    try:
        f = float(value)
        if f.is_integer() and abs(f) < 1e15:
            return str(int(f))
    except ValueError:
        pass
    return value


def _extract_xlsx(path: Path) -> str:
    with _zip_open(path, "XLSX") as zf:
        shared = _xlsx_shared_strings(zf)
        sheets = _xlsx_sheets(zf)
        if not sheets:
            raise ValueError(f"Ungültige XLSX-Datei, keine Tabellenblätter: {path.name}")
        blocks: list[str] = []
        for sheet_name, member in sheets:
            root = _zip_xml(zf, member, "XLSX")
            lines = [f"Blatt: {sheet_name}"]
            for row in root.iter():
                if _local(row.tag) != "row":
                    continue
                cells: list[str] = []
                for cell in row:
                    if _local(cell.tag) != "c":
                        continue
                    col = _col_index(cell.get("r"))
                    if col is None:
                        col = len(cells)
                    while len(cells) < col:
                        cells.append("")
                    text = _xlsx_cell_value(cell, shared).replace("\n", " ").strip()
                    if col < len(cells):
                        cells[col] = text
                    else:
                        cells.append(text)
                while cells and not cells[-1]:
                    cells.pop()
                if any(cells):
                    lines.append(";".join(cells))
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _extract_pptx(path: Path) -> str:
    with _zip_open(path, "PPTX") as zf:
        slides: list[tuple[int, str]] = []
        for name in zf.namelist():
            m = _SLIDE_NAME.match(name)
            if m:
                slides.append((int(m.group(1)), name))
        if not slides:
            raise ValueError(f"Ungültige PPTX-Datei, keine Folien: {path.name}")
        slides.sort()
        blocks: list[str] = []
        for number, member in slides:
            root = _zip_xml(zf, member, "PPTX")
            lines = [f"Folie {number}:"]
            for para in root.iter():
                if _local(para.tag) != "p":
                    continue
                text = _paragraph_text(para, "t", br_tag="br").strip()
                if text:
                    lines.append(text)
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _extract_pdf(path: Path) -> str:
    try:
        import pypdf  # type: ignore[import-not-found]
    except ImportError:
        raise ValueError("PDF-Unterstützung: pip install pypdf") from None
    try:
        reader = pypdf.PdfReader(str(path))
        if getattr(reader, "is_encrypted", False):
            try:
                reader.decrypt("")
            except Exception:  # noqa: BLE001
                raise ValueError(f"PDF ist verschlüsselt: {path.name}") from None
        pages: list[str] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - einzelne kaputte Seite überspringen
                text = ""
            text = text.strip()
            if text:
                pages.append(f"Seite {i}:\n{text}")
    except ValueError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"PDF konnte nicht gelesen werden ({path.name}): {e}") from None
    return "\n\n".join(pages)


def extract_text(path: str | Path) -> str:
    """Holt lesbaren Text aus einer Datei.

    * Text-/Code-Dateien: direkt (UTF-8, Fehler ersetzt); Nullbytes → ``ValueError`` (Binärdatei).
    * ``.html``/``.htm``/``.xml``: Tags entfernt, Entities aufgelöst.
    * ``.docx``: Absätze aus ``word/document.xml`` → Zeilen.
    * ``.xlsx``: je Blatt ``Blatt: Name`` und Zeilen mit ``;`` getrennt.
    * ``.pptx``: Texte je Folie (``Folie N:``).
    * ``.pdf``: über ``pypdf`` (lazy); fehlt es → ``ValueError("PDF-Unterstützung: pip install pypdf")``.

    Unbekannte Endungen werden als Text gelesen. Leeres Ergebnis → ``ValueError``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".docx":
        _check_file(p)
        text = _extract_docx(p)
    elif suffix == ".xlsx":
        _check_file(p)
        text = _extract_xlsx(p)
    elif suffix == ".pptx":
        _check_file(p)
        text = _extract_pptx(p)
    elif suffix == ".pdf":
        _check_file(p)
        text = _extract_pdf(p)
    elif suffix in (".html", ".htm"):
        text = _strip_html(_decode(_read_bytes(p), p.name))
    elif suffix == ".xml":
        text = _strip_xml(_decode(_read_bytes(p), p.name))
    else:
        text = _decode(_read_bytes(p), p.name)
    text = _clean_text(text)
    if not text:
        raise ValueError(f"Datei enthält keinen lesbaren Text: {p.name}")
    return text


# ------------------------------------------------------------------ Chunking
def _overlap_tail(text: str, limit: int) -> str:
    """Letzte höchstens ``limit`` Zeichen von ``text``, beginnend an einer Wortgrenze."""
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text.strip()
    tail = text[-limit:]
    if not text[-limit - 1].isspace():
        # mitten im Wort abgeschnitten -> erstes angeschnittenes Wort verwerfen
        m = re.search(r"\s", tail)
        if not m:
            return ""
        tail = tail[m.end():]
    return tail.strip()


def _split_units(text: str, size: int) -> list[tuple[str, str]]:
    """Zerlegt Text in ``(trenner, einheit)``-Paare: Absätze, bei Bedarf Zeilen, Sätze, Wörter.

    Jede Einheit passt in ``size`` Zeichen, außer einzelne Wörter, die länger sind."""
    units: list[tuple[str, str]] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    for pi, para in enumerate(paragraphs):
        para = para.strip()
        psep = "\n\n" if pi else ""
        if len(para) <= size:
            units.append((psep, para))
            continue
        lines = [ln for ln in para.split("\n") if ln.strip()]
        for li, line in enumerate(lines):
            line = line.strip()
            lsep = psep if li == 0 else "\n"
            if len(line) <= size:
                units.append((lsep, line))
                continue
            sentences = [s for s in _SENTENCE_END.split(line) if s.strip()]
            for si, sentence in enumerate(sentences):
                sentence = sentence.strip()
                ssep = lsep if si == 0 else " "
                if len(sentence) <= size:
                    units.append((ssep, sentence))
                    continue
                for wi, word in enumerate(sentence.split()):
                    units.append((ssep if wi == 0 else " ", word))
    return units


def chunk_text(text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Teilt ``text`` deterministisch in Abschnitte von höchstens ``size`` Zeichen.

    Grenzen liegen bevorzugt an Absätzen, dann Zeilen, Sätzen, zuletzt Wörtern – nie mitten
    im Wort. Aufeinanderfolgende Abschnitte überlappen um bis zu ``overlap`` Zeichen (ganze
    Wörter vom Ende des Vorgängers). Ein einzelnes Wort länger als ``size`` bildet einen
    eigenen, überlangen Abschnitt."""
    text = _clean_text(text or "")
    if not text:
        return []
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size // 2))
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    current = ""
    for sep, unit in _split_units(text, size):
        if not current:
            current = unit
            continue
        candidate = current + sep + unit
        if len(candidate) <= size:
            current = candidate
            continue
        chunks.append(current)
        # Überlappung: ganze Wörter vom Ende des letzten Abschnitts, soweit Platz bleibt
        tail = _overlap_tail(current, min(overlap, size - len(unit) - len(sep)))
        current = tail + sep + unit if tail else unit
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


# ------------------------------------------------------------------ Speicher
class KnowledgeStore:
    """Thread-sicherer Dokumentenspeicher mit hybrider Suche (Volltext + Vektoren)."""

    def __init__(self, path: str | Path, embedder: Embedder | None = None):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
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
        """Setzt oder entfernt das Embedding-Modell. Vektoren abweichender Dimension werden in
        der Suche ignoriert, bis :meth:`reindex` sie neu berechnet."""
        with self._lock:
            self.embedder = embedder

    def _embed(self, texts: Sequence[str]) -> list[list[float]] | None:
        if not self.embedder or not texts:
            return None
        try:
            vecs = self.embedder(list(texts))
        except Exception:  # noqa: BLE001 - ohne Vektor weiterarbeiten
            return None
        if not vecs or len(vecs) != len(texts):
            return None
        dim = len(vecs[0])
        if dim and dim != self.embed_dim:
            with self._lock:
                self.embed_dim = dim
                self._set_meta("embed_dim", str(dim))
        return vecs

    def _embed_all(self, texts: Sequence[str]) -> list[list[float] | None]:
        """Bettet in Stapeln ein; schlägt ein Stapel fehl, bleiben seine Vektoren ``None``."""
        out: list[list[float] | None] = []
        for i in range(0, len(texts), EMBED_BATCH):
            batch = texts[i:i + EMBED_BATCH]
            vecs = self._embed(batch)
            out.extend(vecs if vecs else [None] * len(batch))
        return out

    def _vec_ok(self, raw: str | None) -> list[float] | None:
        if not raw:
            return None
        try:
            vec = json.loads(raw)
        except (TypeError, ValueError):
            return None
        if self.embed_dim and len(vec) != self.embed_dim:
            return None
        return vec

    def _doc_row(self, row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"], title=row["title"], path=row["path"], kind=row["kind"], size=row["size"],
            hash=row["hash"], project=row["project"], added_at=row["added_at"], mtime=row["mtime"],
            chunks=int(row["n_chunks"]) if "n_chunks" in row.keys() else self._chunk_count(row["id"]),
            missing=bool(row["missing"]),
        )

    def _chunk_count(self, doc_id: int) -> int:
        return self._db.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", (doc_id,)).fetchone()[0]

    @staticmethod
    def _chunk_row(row: sqlite3.Row, score: float = 0.0) -> Chunk:
        return Chunk(id=row["id"], doc_id=row["doc_id"], title=row["title"], idx=row["idx"],
                     content=row["content"], score=score)

    _DOC_SELECT = ("SELECT d.*, (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS n_chunks"
                   " FROM documents d")

    def _write_chunks(self, doc_id: int, title: str, chunks: Sequence[str],
                      vecs: Sequence[list[float] | None]) -> None:
        """Ersetzt alle Abschnitte eines Dokuments (Aufrufer hält die Sperre)."""
        self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self._db.executemany(
            "INSERT INTO chunks(doc_id, idx, title, content, embedding) VALUES (?,?,?,?,?)",
            [(doc_id, i + 1, title, c, json.dumps(v) if v else None)
             for i, (c, v) in enumerate(zip(chunks, vecs))],
        )

    def _store(self, *, title: str, text: str, path: str | None, kind: str, size: int, digest: str,
               project: str | None, mtime: float | None, existing: int | None = None) -> Document:
        """Legt ein Dokument an oder ersetzt Inhalt/Abschnitte eines vorhandenen (``existing``)."""
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError(f"Dokument »{title}« enthält keinen Text.")
        vecs = self._embed_all(chunks)          # außerhalb der Sperre: kann lange dauern
        now = time.time()
        with self._lock:
            try:
                if existing is None:
                    cur = self._db.execute(
                        "INSERT INTO documents(title, path, kind, size, hash, project, added_at, mtime, missing)"
                        " VALUES (?,?,?,?,?,?,?,?,0)",
                        (title, path, kind, size, digest, project, now, mtime),
                    )
                    doc_id = int(cur.lastrowid)
                else:
                    doc_id = int(existing)
                    self._db.execute(
                        "UPDATE documents SET title = ?, path = ?, kind = ?, size = ?, hash = ?, project = ?,"
                        " mtime = ?, missing = 0 WHERE id = ?",
                        (title, path, kind, size, digest, project, mtime, doc_id),
                    )
                self._write_chunks(doc_id, title, chunks, vecs)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.get(doc_id)  # type: ignore[return-value]

    # ------------------------------------------------------------ schreiben
    def add_text(self, title: str, text: str, *, project: str | None = None, kind: str = "text",
                 source_path: str | None = None) -> Document:
        """Indexiert direkt übergebenen Text. Gleicher Inhalt im gleichen Projekt → vorhandenes Dokument."""
        title = _one_line(title or "")
        if not title:
            raise ValueError("Titel darf nicht leer sein.")
        text = _clean_text(text or "")
        if not text:
            raise ValueError("Leerer Text kann nicht indexiert werden.")
        kind = (kind or "text").strip().lower().lstrip(".") or "text"
        data = text.encode("utf-8")
        digest = _sha256(data)
        with self._lock:
            row = self._db.execute(
                "SELECT id FROM documents WHERE hash = ? AND project IS ? AND path IS ? LIMIT 1",
                (digest, project, source_path),
            ).fetchone()
        if row:
            return self.get(row["id"])  # type: ignore[return-value]
        return self._store(title=title, text=text, path=source_path, kind=kind, size=len(data),
                           digest=digest, project=project, mtime=None)

    def _add_file(self, path: str | Path, *, project: str | None = None,
                  title: str | None = None) -> tuple[Document, str]:
        """Wie :meth:`add_file`, liefert zusätzlich ``"hinzugefuegt" | "aktualisiert" | "unveraendert"``."""
        real = os.path.realpath(str(path))
        p = Path(real)
        data = _read_bytes(p)
        digest = _sha256(data)
        st = p.stat()
        with self._lock:
            row = self._db.execute(
                "SELECT id, hash, title, project FROM documents WHERE path = ? ORDER BY id LIMIT 1", (real,)
            ).fetchone()
            if row and row["hash"] == digest:
                updates = {"mtime": st.st_mtime, "missing": 0}
                if project is not None:
                    updates["project"] = project
                if title and title != row["title"]:
                    updates["title"] = title
                    self._db.execute("UPDATE chunks SET title = ? WHERE doc_id = ?", (title, row["id"]))
                sets = ", ".join(f"{k} = ?" for k in updates)
                self._db.execute(f"UPDATE documents SET {sets} WHERE id = ?", (*updates.values(), row["id"]))
                self._db.commit()
                return self.get(row["id"]), "unveraendert"  # type: ignore[return-value]
        text = extract_text(real)
        doc_title = _one_line(title or "") or p.name
        kind = p.suffix.lower().lstrip(".") or "text"
        if row:
            doc = self._store(title=doc_title, text=text, path=real, kind=kind, size=st.st_size, digest=digest,
                              project=project if project is not None else row["project"], mtime=st.st_mtime,
                              existing=row["id"])
            return doc, "aktualisiert"
        doc = self._store(title=doc_title, text=text, path=real, kind=kind, size=st.st_size, digest=digest,
                          project=project, mtime=st.st_mtime)
        return doc, "hinzugefuegt"

    def add_file(self, path: str | Path, *, project: str | None = None, title: str | None = None) -> Document:
        """Indexiert eine Datei. Ist dieselbe Datei (Pfad) unverändert (SHA-256), wird das vorhandene
        Dokument zurückgegeben; hat sich der Inhalt geändert, werden die Abschnitte ersetzt."""
        doc, _ = self._add_file(path, project=project, title=title)
        return doc

    @staticmethod
    def _matches(name: str, patterns: Sequence[str]) -> bool:
        lower = name.lower()
        for pat in patterns:
            pat = (pat or "").strip().lower()
            if not pat:
                continue
            if pat.startswith(".") and not any(ch in pat for ch in "*?["):
                if lower.endswith(pat):
                    return True
            elif fnmatch.fnmatch(lower, pat):
                return True
        return False

    def add_directory(self, path: str | Path, *, project: str | None = None,
                      patterns: Iterable[str] | None = None, recursive: bool = True,
                      max_files: int = 500) -> dict:
        """Indexiert alle passenden Dateien eines Verzeichnisses.

        ``patterns``: Endungen (``".md"``) oder Glob-Muster (``"*.md"``), Standard :data:`SUPPORTED`.
        Versteckte Verzeichnisse, ``.git``, ``__pycache__`` und ``node_modules`` werden nicht betreten;
        versteckte und nicht passende Dateien landen unter ``"uebersprungen"``. Nach ``max_files``
        indexierten Dateien wird abgebrochen (Hinweis unter ``"fehler"``).

        Rückgabe: ``{"hinzugefuegt": n, "unveraendert": n, "uebersprungen": [pfad…], "fehler": [(pfad, grund)…]}``
        (``hinzugefuegt`` zählt neue und wegen Änderung neu indexierte Dateien)."""
        root = Path(os.path.realpath(str(path)))
        if not root.exists():
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {path}")
        if not root.is_dir():
            raise NotADirectoryError(f"Kein Verzeichnis: {path}")
        pats = [p for p in (patterns if patterns is not None else sorted(SUPPORTED))]
        max_files = max(1, int(max_files))
        result: dict = {"hinzugefuegt": 0, "unveraendert": 0, "uebersprungen": [], "fehler": []}

        candidates: list[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
            if not recursive:
                dirnames[:] = []
            for name in sorted(filenames):
                full = os.path.join(dirpath, name)
                if name.startswith(".") or not self._matches(name, pats):
                    result["uebersprungen"].append(full)
                    continue
                candidates.append(full)

        processed = 0
        for full in candidates:
            if processed >= max_files:
                rest = len(candidates) - processed
                result["fehler"].append((str(root), f"Limit von {max_files} Dateien erreicht, {rest} nicht indexiert"))
                break
            processed += 1
            try:
                _, status = self._add_file(full, project=project)
            except (OSError, ValueError, sqlite3.Error) as e:
                result["fehler"].append((full, str(e) or e.__class__.__name__))
                continue
            if status == "unveraendert":
                result["unveraendert"] += 1
            else:
                result["hinzugefuegt"] += 1
        return result

    def sync(self) -> dict:
        """Prüft alle Datei-Dokumente: geändertes mtime/Größe → Hash vergleichen → neu indexieren;
        nicht mehr vorhandene Dateien werden als fehlend markiert (Inhalt bleibt durchsuchbar).

        Rückgabe: ``{"aktualisiert": n, "unveraendert": n, "fehlend": [pfad…], "fehler": [(pfad, grund)…]}``."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, path, hash, size, mtime, missing FROM documents WHERE path IS NOT NULL ORDER BY id"
            ).fetchall()
        result: dict = {"aktualisiert": 0, "unveraendert": 0, "fehlend": [], "fehler": []}
        for r in rows:
            path = r["path"]
            try:
                st = os.stat(path)
            except OSError:
                with self._lock:
                    self._db.execute("UPDATE documents SET missing = 1 WHERE id = ?", (r["id"],))
                    self._db.commit()
                result["fehlend"].append(path)
                continue
            same_meta = (r["mtime"] is not None and abs(st.st_mtime - r["mtime"]) < 1e-6
                         and st.st_size == r["size"] and not r["missing"])
            if same_meta:
                result["unveraendert"] += 1
                continue
            try:
                _, status = self._add_file(path)
            except (OSError, ValueError, sqlite3.Error) as e:
                result["fehler"].append((path, str(e) or e.__class__.__name__))
                continue
            if status == "unveraendert":
                result["unveraendert"] += 1
            else:
                result["aktualisiert"] += 1
        return result

    def remove(self, doc_id: int) -> bool:
        """Entfernt ein Dokument samt Abschnitten."""
        with self._lock:
            self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (int(doc_id),))
            cur = self._db.execute("DELETE FROM documents WHERE id = ?", (int(doc_id),))
            self._db.commit()
            return cur.rowcount > 0

    # ------------------------------------------------------------ lesen
    def get(self, doc_id: int) -> Document | None:
        with self._lock:
            row = self._db.execute(self._DOC_SELECT + " WHERE d.id = ?", (int(doc_id),)).fetchone()
        return self._doc_row(row) if row else None

    def get_by_path(self, path: str | Path) -> Document | None:
        """Dokument zu einem Dateipfad (realpath)."""
        real = os.path.realpath(str(path))
        with self._lock:
            row = self._db.execute(self._DOC_SELECT + " WHERE d.path = ? ORDER BY d.id LIMIT 1", (real,)).fetchone()
        return self._doc_row(row) if row else None

    def list(self, project: str | None = None) -> list[Document]:
        """Alle Dokumente (mit ``project``: die des Projekts plus projektlose), neueste zuerst."""
        sql = self._DOC_SELECT
        args: tuple = ()
        if project:
            sql += " WHERE d.project = ? OR d.project IS NULL"
            args = (project,)
        sql += " ORDER BY d.added_at DESC, d.id DESC"
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [self._doc_row(r) for r in rows]

    def chunks(self, doc_id: int) -> list[Chunk]:
        """Alle Abschnitte eines Dokuments in Reihenfolge."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM chunks WHERE doc_id = ? ORDER BY idx", (int(doc_id),)).fetchall()
        return [self._chunk_row(r) for r in rows]

    def search(self, query: str, k: int = 5, project: str | None = None, min_score: float = 0.05) -> list[Chunk]:
        """Hybride Suche über Abschnitte: BM25-Volltext (Inhalt, Titel) + Kosinus-Ähnlichkeit.

        Relevanz = ``0.6·Vektor + 0.4·Text`` bei vorhandenem Vektor, sonst nur Text (wie
        ``MemoryStore.search``). ``project`` liefert Abschnitte des Projekts und projektlose."""
        k = int(k)
        if k <= 0 or not (query or "").strip():
            return []
        candidates: dict[int, dict] = {}
        fts = _fts_query(query)
        proj_sql = " AND (d.project = ? OR d.project IS NULL)" if project else ""
        proj_args: tuple = (project,) if project else ()

        qvec = self._embed([query])
        with self._lock:
            if fts:
                rows = self._db.execute(
                    "SELECT c.*, bm25(chunks_fts, 1.0, 0.5) AS rank FROM chunks_fts"
                    " JOIN chunks c ON c.id = chunks_fts.rowid"
                    " JOIN documents d ON d.id = c.doc_id"
                    f" WHERE chunks_fts MATCH ?{proj_sql} ORDER BY rank LIMIT 100",
                    (fts,) + proj_args,
                ).fetchall()
                if rows:
                    ranks = [-r["rank"] for r in rows]  # bm25: kleiner = besser
                    top = max(ranks) or 1.0
                    for r, rk in zip(rows, ranks):
                        candidates[r["id"]] = {"row": r, "text": max(0.0, rk / top)}
            if qvec:
                for r in self._db.execute(
                    "SELECT c.* FROM chunks c JOIN documents d ON d.id = c.doc_id"
                    f" WHERE c.embedding IS NOT NULL{proj_sql}",
                    proj_args,
                ):
                    vec = self._vec_ok(r["embedding"])
                    if vec is None:
                        continue
                    sim = cosine(qvec[0], vec)
                    if sim > 0.3:
                        c = candidates.setdefault(r["id"], {"row": r, "text": 0.0})
                        c["vec"] = sim

        scored: list[Chunk] = []
        for c in candidates.values():
            text = c.get("text", 0.0)
            relevance = 0.6 * c["vec"] + 0.4 * text if "vec" in c else text
            if relevance > 0 and relevance >= min_score:
                scored.append(self._chunk_row(c["row"], relevance))
        scored.sort(key=lambda ch: (-ch.score, ch.doc_id, ch.idx))
        return scored[:k]

    def context_section(self, query: str, k: int = 4, max_chars: int = 1500, project: str | None = None) -> str:
        """Kontextabschnitt für den Prompt mit Zitaten, oder ``""`` ohne Treffer:

        ``Auszüge aus deinen Dokumenten (Quelle in eckigen Klammern):\\n[Titel §3] …``"""
        hits = self.search(query, k=k, project=project)
        if not hits:
            return ""
        max_chars = max(0, int(max_chars))
        lines = [CONTEXT_HEADER]
        used = len(CONTEXT_HEADER)
        for chunk in hits:
            line = f"{chunk.cite()} {_one_line(chunk.content)}"
            remaining = max_chars - used - 1
            if len(line) > remaining:
                head = len(chunk.cite()) + 1 + len(CLIP_SUFFIX)
                if remaining - head < 40:
                    break
                line = _clip(line, remaining - len(CLIP_SUFFIX))
            lines.append(line)
            used += len(line) + 1
        if len(lines) == 1:
            return ""
        return "\n".join(lines) + "\n"

    def reindex(self, progress: Callable[[int, int], None] | None = None, only_missing: bool = False) -> int:
        """Berechnet Vektoren neu: fehlende (immer) und – ohne ``only_missing`` – auch solche mit
        abweichender Dimension (Modellwechsel). Rückgabe: Anzahl aktualisierter Abschnitte."""
        if not self.embedder:
            return 0
        with self._lock:
            rows = self._db.execute("SELECT id, content, embedding FROM chunks ORDER BY id").fetchall()
        todo: list[tuple[int, str]] = []
        for r in rows:
            if r["embedding"] is None:
                todo.append((r["id"], r["content"]))
            elif not only_missing and self._vec_ok(r["embedding"]) is None:
                todo.append((r["id"], r["content"]))
        done = 0
        for i in range(0, len(todo), EMBED_BATCH):
            batch = todo[i:i + EMBED_BATCH]
            vecs = self._embed([c for _, c in batch])
            if not vecs:
                break
            with self._lock:
                self._db.executemany(
                    "UPDATE chunks SET embedding = ? WHERE id = ?",
                    [(json.dumps(v), cid) for (cid, _), v in zip(batch, vecs)],
                )
                self._db.commit()
            done += len(batch)
            if progress:
                progress(done, len(todo))
        return done

    def stats(self) -> dict:
        with self._lock:
            docs = self._db.execute("SELECT COUNT(*), COALESCE(SUM(size), 0) FROM documents").fetchone()
            missing = self._db.execute("SELECT COUNT(*) FROM documents WHERE missing = 1").fetchone()[0]
            n_chunks = self._db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            projects = [r[0] for r in self._db.execute(
                "SELECT DISTINCT project FROM documents WHERE project IS NOT NULL ORDER BY project")]
            with_vec = 0
            for r in self._db.execute("SELECT embedding FROM chunks WHERE embedding IS NOT NULL"):
                if self._vec_ok(r["embedding"]) is not None:
                    with_vec += 1
        return {
            "dokumente": int(docs[0]),
            "abschnitte": int(n_chunks),
            "mit_vektor": with_vec,
            "ohne_vektor": int(n_chunks) - with_vec,
            "projekte": projects,
            "groesse_bytes": int(docs[1]),
            "fehlend": int(missing),
            "embedding_dim": self.embed_dim,
        }

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True


# ------------------------------------------------------------------ Werkzeuge
def register_tools(registry: "ToolRegistry", store: KnowledgeStore) -> None:
    """Registriert ``dokumente_suchen`` und ``dokument_hinzufuegen`` in der ``ToolRegistry``.
    Pfade laufen über ``registry.resolve`` – nur Dateien im Arbeitsbereich können indexiert werden."""
    from .tools import Tool

    def dokumente_suchen(frage: str, projekt: str | None = None) -> str:
        if not frage or not str(frage).strip():
            raise ValueError("Leere Suchanfrage.")
        proj = str(projekt).strip() if projekt else None
        hits = store.search(str(frage), k=5, project=proj or None)
        if not hits:
            return "Keine passenden Dokument-Auszüge gefunden."
        lines = [f"{len(hits)} Treffer (Quelle in eckigen Klammern):"]
        for c in hits:
            score = f"{c.score:.2f}".replace(".", ",")
            lines.append(f"{c.cite()} (Relevanz {score}): {_clip(_one_line(c.content), MAX_SNIPPET_CHARS)}")
        return "\n".join(lines)

    def dokument_hinzufuegen(pfad: str, projekt: str | None = None) -> str:
        if not pfad or not str(pfad).strip():
            raise ValueError("Kein Pfad übergeben.")
        real = registry.resolve(str(pfad))
        if not os.path.exists(real):
            raise FileNotFoundError(f"Datei nicht gefunden: {pfad}")
        proj = str(projekt).strip() if projekt else None
        if os.path.isdir(real):
            res = store.add_directory(real, project=proj or None)
            text = (f"Verzeichnis {registry.relpath(real)} indexiert: {res['hinzugefuegt']} neu/aktualisiert, "
                    f"{res['unveraendert']} unverändert, {len(res['uebersprungen'])} übersprungen")
            if res["fehler"]:
                shown = "; ".join(f"{os.path.basename(p)}: {g}" for p, g in res["fehler"][:5])
                text += f", {len(res['fehler'])} Fehler ({shown})"
            return text
        doc, status = store._add_file(real, project=proj or None)
        word = {"hinzugefuegt": "indexiert", "aktualisiert": "neu indexiert", "unveraendert": "bereits vorhanden"}[status]
        proj_txt = f", Projekt »{doc.project}«" if doc.project else ""
        return f"Dokument »{doc.title}« {word}: {doc.chunks} Abschnitte (#{doc.id}{proj_txt})"

    registry.register(Tool(
        name="dokumente_suchen",
        description="Durchsucht die indexierten Dokumente des Nutzers (Datenzentrum) und liefert passende Auszüge mit Zitat.",
        parameters={"type": "object",
                    "properties": {"frage": {"type": "string", "description": "Suchbegriffe oder Frage",
                                             "example": "Maximalstrom des Reglers"},
                                   "projekt": {"type": "string", "description": "Nur Dokumente dieses Projekts (plus allgemeine)"}},
                    "required": ["frage"]},
        fn=dokumente_suchen,
    ))
    registry.register(Tool(
        name="dokument_hinzufuegen",
        description="Indexiert eine Datei oder ein Verzeichnis aus dem Arbeitsbereich im Datenzentrum "
                    "(" + ", ".join(sorted(SUPPORTED)) + ").",
        parameters={"type": "object",
                    "properties": {"pfad": {"type": "string", "description": "Pfad relativ zum Arbeitsbereich",
                                            "example": "docs/handbuch.md"},
                                   "projekt": {"type": "string", "description": "Projekt, dem das Dokument zugeordnet wird"}},
                    "required": ["pfad"]},
        fn=dokument_hinzufuegen,
    ))
