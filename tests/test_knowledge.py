import os
import sys
import tempfile
import time
import unittest
import zipfile

from obito.llm import FakeBackend
from obito.knowledge import (CONTEXT_HEADER, SUPPORTED, Chunk, Document, KnowledgeStore, chunk_text,
                             extract_text, register_tools)
from obito.tools import ToolRegistry

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


# ------------------------------------------------------------------ Testdateien
def write(path, text, encoding="utf-8"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding=encoding) as fh:
        fh.write(text)
    return path


def make_docx(path, paragraphs):
    body = "".join(
        f"<w:p><w:r><w:t xml:space=\"preserve\">{p}</w:t></w:r></w:p>" for p in paragraphs
    )
    doc = (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           f'<w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        zf.writestr("word/document.xml", doc)
    return path


def make_xlsx(path, sheets):
    """sheets: {"Name": [[zelle, …], …]} – Strings landen im sharedStrings, Zahlen direkt."""
    shared = []

    def sidx(s):
        if s not in shared:
            shared.append(s)
        return shared.index(s)

    sheet_xml = {}
    wb_sheets = []
    rels = []
    for n, (name, rows) in enumerate(sheets.items(), start=1):
        row_xml = []
        for ri, row in enumerate(rows, start=1):
            cells = []
            for ci, value in enumerate(row):
                col = chr(ord("A") + ci)
                if value is None:
                    continue
                if isinstance(value, (int, float)):
                    cells.append(f'<c r="{col}{ri}"><v>{value}</v></c>')
                else:
                    cells.append(f'<c r="{col}{ri}" t="s"><v>{sidx(value)}</v></c>')
            row_xml.append(f'<row r="{ri}">{"".join(cells)}</row>')
        sheet_xml[f"xl/worksheets/sheet{n}.xml"] = (
            f'<worksheet xmlns="{S_NS}"><sheetData>{"".join(row_xml)}</sheetData></worksheet>')
        wb_sheets.append(f'<sheet name="{name}" sheetId="{n}" r:id="rId{n}"/>')
        rels.append(f'<Relationship Id="rId{n}" Type="x" Target="worksheets/sheet{n}.xml"/>')
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("xl/workbook.xml",
                    f'<workbook xmlns="{S_NS}" xmlns:r="{R_NS}"><sheets>{"".join(wb_sheets)}</sheets></workbook>')
        zf.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{REL_NS}">{"".join(rels)}</Relationships>')
        zf.writestr("xl/sharedStrings.xml",
                    f'<sst xmlns="{S_NS}">' + "".join(f"<si><t>{s}</t></si>" for s in shared) + "</sst>")
        for name, xml in sheet_xml.items():
            zf.writestr(name, xml)
    return path


def make_pptx(path, slides):
    """slides: Liste von Listen mit Textzeilen je Folie."""
    with zipfile.ZipFile(path, "w") as zf:
        # absichtlich in verdrehter Reihenfolge schreiben: Extraktor muss numerisch sortieren
        for n in reversed(range(1, len(slides) + 1)):
            paras = "".join(f"<a:p><a:r><a:t>{t}</a:t></a:r></a:p>" for t in slides[n - 1])
            zf.writestr(f"ppt/slides/slide{n}.xml",
                        f'<p:sld xmlns:p="x" xmlns:a="{A_NS}"><p:cSld><p:spTree><p:sp><p:txBody>{paras}'
                        f"</p:txBody></p:sp></p:spTree></p:cSld></p:sld>")
    return path


class TempCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def p(self, *parts):
        return os.path.join(self.root, *parts)


# ------------------------------------------------------------------ Extraktion
class ExtractTextTest(TempCase):
    def test_supported_set(self):
        for ext in (".md", ".py", ".docx", ".xlsx", ".pptx", ".pdf", ".html", ".csv", ".ino"):
            self.assertIn(ext, SUPPORTED)
        self.assertTrue(all(e.startswith(".") for e in SUPPORTED))

    def test_plain_text_and_whitespace_cleanup(self):
        path = write(self.p("a.md"), "# Titel\r\n\r\n\r\n\r\nZeile   eins  \r\nZeile zwei\n")
        self.assertEqual(extract_text(path), "# Titel\n\nZeile eins\nZeile zwei")

    def test_utf8_bom_and_replacement(self):
        path = self.p("bom.txt")
        with open(path, "wb") as fh:
            fh.write(b"\xef\xbb\xbfHallo \xff Welt")
        text = extract_text(path)
        self.assertTrue(text.startswith("Hallo"))
        self.assertIn("Welt", text)
        self.assertNotIn("﻿", text)

    def test_binary_rejected(self):
        path = self.p("bild.txt")
        with open(path, "wb") as fh:
            fh.write(b"PNG\x00\x01\x02 viel binaer")
        with self.assertRaises(ValueError) as ctx:
            extract_text(path)
        self.assertIn("Binärdatei", str(ctx.exception))

    def test_empty_rejected(self):
        path = write(self.p("leer.txt"), "   \n\n  ")
        with self.assertRaises(ValueError):
            extract_text(path)

    def test_missing_and_directory(self):
        with self.assertRaises(FileNotFoundError):
            extract_text(self.p("gibtsnicht.txt"))
        with self.assertRaises(IsADirectoryError):
            extract_text(self.root)

    def test_html_stripping(self):
        path = write(self.p("seite.html"),
                     "<html><head><title>Kopf</title><style>p{color:red}</style></head><body>"
                     "<script>var x = '<b>nein</b>';</script><!-- Kommentar -->"
                     "<h1>Regler &amp; Motor</h1><p>Max. <b>40&nbsp;A</b> Dauerstrom.</p>"
                     "<ul><li>eins</li><li>zwei &lt;drei&gt;</li></ul>"
                     "<table><tr><td>Zelle A</td><td>Zelle B</td></tr><tr><td>Zelle C</td></tr></table>"
                     "</body></html>")
        text = extract_text(path)
        self.assertIn("Regler & Motor", text)
        self.assertIn("40\xa0A Dauerstrom.", text)
        self.assertIn("zwei <drei>", text)
        self.assertNotIn("<", text.replace("<drei>", ""))
        self.assertNotIn("color", text)
        self.assertNotIn("nein", text)
        self.assertNotIn("Kommentar", text)
        # Listenpunkte werden Zeilen, Tabellenzeilen bleiben zusammen
        self.assertIn("eins\nzwei", text)
        self.assertIn("Zelle A Zelle B\nZelle C", text)

    def test_xml_stripping(self):
        path = write(self.p("daten.xml"),
                     '<?xml version="1.0"?><root><item name="a">Wert &amp; mehr</item><item>zwei</item></root>')
        text = extract_text(path)
        self.assertEqual(text.split("\n"), ["Wert & mehr", "zwei"])

    def test_docx(self):
        path = make_docx(self.p("bericht.docx"), ["Überschrift", "Erster Absatz mit 12 V.", "", "Zweiter Absatz"])
        text = extract_text(path)
        self.assertEqual(text, "Überschrift\nErster Absatz mit 12 V.\n\nZweiter Absatz")

    def test_docx_nested_paragraph_not_duplicated(self):
        doc = (f'<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>Außen</w:t></w:r>'
               f"<w:txbxContent><w:p><w:r><w:t>Innen</w:t></w:r></w:p></w:txbxContent></w:p>"
               f"</w:body></w:document>")
        path = self.p("box.docx")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("word/document.xml", doc)
        self.assertEqual(extract_text(path), "AußenInnen")

    def test_docx_broken(self):
        path = write(self.p("kaputt.docx"), "kein zip")
        with self.assertRaises(ValueError):
            extract_text(path)
        path2 = self.p("ohne.docx")
        with zipfile.ZipFile(path2, "w") as zf:
            zf.writestr("irgendwas.txt", "x")
        with self.assertRaises(ValueError) as ctx:
            extract_text(path2)
        self.assertIn("word/document.xml", str(ctx.exception))

    def test_xlsx(self):
        path = make_xlsx(self.p("tabelle.xlsx"), {
            "Teile": [["Name", "Menge", "Preis"], ["Motor", 4, 12.5], [None, None, None], ["ESC", 4, 20]],
            "Notizen": [["Hinweis", None, "Spalte C"]],
        })
        text = extract_text(path)
        self.assertEqual(text, "Blatt: Teile\nName;Menge;Preis\nMotor;4;12.5\nESC;4;20\n\n"
                               "Blatt: Notizen\nHinweis;;Spalte C")

    def test_pptx(self):
        path = make_pptx(self.p("vortrag.pptx"), [["Titel", "Untertitel"], ["Folie zwei", "Punkt A"], ["Ende"]])
        text = extract_text(path)
        self.assertEqual(text, "Folie 1:\nTitel\nUntertitel\n\nFolie 2:\nFolie zwei\nPunkt A\n\nFolie 3:\nEnde")

    def test_pdf_without_pypdf(self):
        path = write(self.p("doku.pdf"), "%PDF-1.4 fake")
        saved = sys.modules.get("pypdf")
        sys.modules["pypdf"] = None  # erzwingt ImportError
        try:
            with self.assertRaises(ValueError) as ctx:
                extract_text(path)
            self.assertEqual(str(ctx.exception), "PDF-Unterstützung: pip install pypdf")
        finally:
            if saved is None:
                sys.modules.pop("pypdf", None)
            else:
                sys.modules["pypdf"] = saved

    def test_unknown_extension_read_as_text(self):
        path = write(self.p("Makefile"), "all:\n\techo hallo")
        self.assertIn("echo hallo", extract_text(path))


# ------------------------------------------------------------------ Chunking
class ChunkTextTest(unittest.TestCase):
    def test_empty_and_short(self):
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n "), [])
        self.assertEqual(chunk_text("kurz"), ["kurz"])
        self.assertEqual(chunk_text("a" * 800), ["a" * 800])

    def test_size_respected_and_never_mid_word(self):
        words = [f"wort{i:04d}" for i in range(600)]
        text = " ".join(words)
        chunks = chunk_text(text, size=120, overlap=30)
        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(len(c) <= 120 for c in chunks))
        vocab = set(words)
        for c in chunks:
            self.assertTrue(all(t in vocab for t in c.split()), c)
        # alle Wörter kommen in Reihenfolge vor
        seen = []
        for c in chunks:
            for t in c.split():
                if not seen or t > seen[-1]:
                    seen.append(t)
        self.assertEqual(seen, words)

    def test_overlap_repeats_tail_words(self):
        words = [f"w{i:03d}" for i in range(300)]
        chunks = chunk_text(" ".join(words), size=100, overlap=25)
        for a, b in zip(chunks, chunks[1:]):
            a_words, b_words = a.split(), b.split()
            # Anfang von b = Ende von a
            n = 0
            while n < len(b_words) and b_words[n] in a_words:
                n += 1
            self.assertGreaterEqual(n, 1, (a, b))
            self.assertEqual(a_words[-n:], b_words[:n])
            self.assertLessEqual(len(" ".join(b_words[:n])), 25)

    def test_no_overlap(self):
        words = [f"w{i:03d}" for i in range(300)]
        chunks = chunk_text(" ".join(words), size=100, overlap=0)
        joined = " ".join(chunks).split()
        self.assertEqual(joined, words)

    def test_paragraph_boundaries_preferred(self):
        paras = [f"Absatz {i}: " + "text " * 20 for i in range(8)]
        text = "\n\n".join(p.strip() for p in paras)
        chunks = chunk_text(text, size=260, overlap=0)
        for c in chunks:
            self.assertTrue(c.startswith("Absatz"), c)
            self.assertTrue(c.endswith("text"), c)
        self.assertIn("\n\n", chunks[0])          # zwei Absätze passen in einen Abschnitt

    def test_sentence_boundaries_for_long_paragraph(self):
        sentences = [f"Satz Nummer {i} endet hier." for i in range(40)]
        text = " ".join(sentences)
        chunks = chunk_text(text, size=120, overlap=0)
        for c in chunks:
            self.assertTrue(c.endswith("."), c)
            self.assertTrue(c.startswith("Satz"), c)

    def test_oversized_word_becomes_own_chunk(self):
        text = "kurz " + "x" * 500 + " ende"
        chunks = chunk_text(text, size=100, overlap=10)
        self.assertIn("x" * 500, chunks)
        self.assertEqual(len(chunks), 3)

    def test_deterministic(self):
        text = "\n\n".join("Absatz %d. " % i + "Wort " * (i * 7 % 50 + 5) for i in range(30))
        self.assertEqual(chunk_text(text, 300, 40), chunk_text(text, 300, 40))

    def test_overlap_clamped_to_half_size(self):
        words = [f"w{i:03d}" for i in range(100)]
        chunks = chunk_text(" ".join(words), size=50, overlap=500)
        self.assertTrue(all(len(c) <= 50 for c in chunks))
        self.assertGreater(len(chunks), 1)


# ------------------------------------------------------------------ Datenklassen
class DataclassTest(unittest.TestCase):
    def test_document_to_dict_keys(self):
        d = Document(id=1, title="T", path="/x", kind="md", size=10, hash="h", project="P",
                     added_at=1_700_000_000.0, mtime=None, chunks=3)
        dd = d.to_dict()
        for key in ("id", "titel", "pfad", "art", "groesse", "projekt", "hinzugefuegt", "geaendert", "abschnitte"):
            self.assertIn(key, dd)
        self.assertEqual(dd["abschnitte"], 3)
        self.assertIsNone(dd["geaendert"])
        self.assertRegex(dd["hinzugefuegt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        self.assertIn("T", d.short())

    def test_chunk_to_dict_and_cite(self):
        c = Chunk(id=5, doc_id=1, title="Handbuch", idx=3, content="Inhalt", score=0.54321)
        self.assertEqual(c.cite(), "[Handbuch §3]")
        self.assertEqual(c.to_dict(), {"id": 5, "dokument_id": 1, "titel": "Handbuch", "abschnitt": 3,
                                       "inhalt": "Inhalt", "score": 0.5432})


# ------------------------------------------------------------------ Speicher
class StoreBasicsTest(TempCase):
    def setUp(self):
        super().setUp()
        self.store = KnowledgeStore(":memory:")

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def test_add_text_get_list_remove(self):
        d = self.store.add_text("Notiz", "Der Regler verträgt 40 A.", project="Drohne")
        self.assertEqual((d.id, d.title, d.kind, d.project, d.path, d.chunks), (1, "Notiz", "text", "Drohne", None, 1))
        self.assertEqual(self.store.get(d.id).title, "Notiz")
        self.assertEqual([x.id for x in self.store.list()], [1])
        self.assertEqual(len(self.store.chunks(d.id)), 1)
        self.assertTrue(self.store.remove(d.id))
        self.assertFalse(self.store.remove(d.id))
        self.assertIsNone(self.store.get(d.id))
        self.assertEqual(self.store.stats()["abschnitte"], 0)
        self.assertEqual(self.store.search("Regler"), [])

    def test_add_text_validation(self):
        with self.assertRaises(ValueError):
            self.store.add_text("", "Text")
        with self.assertRaises(ValueError):
            self.store.add_text("Titel", "   ")

    def test_add_text_dedup_same_project(self):
        a = self.store.add_text("A", "Gleicher Inhalt", project="P")
        b = self.store.add_text("B", "Gleicher Inhalt", project="P")
        c = self.store.add_text("C", "Gleicher Inhalt", project="Q")
        self.assertEqual(a.id, b.id)
        self.assertNotEqual(a.id, c.id)

    def test_long_text_is_chunked_with_running_index(self):
        text = "\n\n".join(f"Absatz {i}. " + "Inhalt " * 60 for i in range(10))
        d = self.store.add_text("Lang", text)
        self.assertGreater(d.chunks, 3)
        chunks = self.store.chunks(d.id)
        self.assertEqual([c.idx for c in chunks], list(range(1, d.chunks + 1)))
        self.assertEqual(chunks[0].cite(), "[Lang §1]")
        self.assertTrue(all(c.title == "Lang" for c in chunks))

    def test_add_file_and_dedup(self):
        path = write(self.p("docs", "handbuch.md"), "# Handbuch\n\nDer Motor hat 2300 KV.")
        d1 = self.store.add_file(path, project="Drohne")
        self.assertEqual((d1.title, d1.kind, d1.project), ("handbuch.md", "md", "Drohne"))
        self.assertEqual(d1.path, os.path.realpath(path))
        self.assertIsNotNone(d1.mtime)
        self.assertEqual(d1.size, os.path.getsize(path))
        d2 = self.store.add_file(path)
        self.assertEqual(d1.id, d2.id)
        self.assertEqual(d2.project, "Drohne")        # Projekt bleibt, wenn keins übergeben wird
        self.assertEqual(self.store.stats()["dokumente"], 1)
        d3 = self.store.add_file(path, title="Motor-Handbuch")
        self.assertEqual(d3.id, d1.id)
        self.assertEqual(d3.title, "Motor-Handbuch")
        self.assertEqual(self.store.search("Motor KV")[0].title, "Motor-Handbuch")

    def test_add_file_changed_content_replaces_chunks(self):
        path = write(self.p("n.txt"), "Alter Inhalt über Propeller.")
        d1 = self.store.add_file(path)
        old_chunk_ids = [c.id for c in self.store.chunks(d1.id)]
        write(path, "Neuer Inhalt über Akkus.\n\n" + "Mehr Text. " * 200)
        d2 = self.store.add_file(path)
        self.assertEqual(d1.id, d2.id)
        self.assertNotEqual(d1.hash, d2.hash)
        self.assertGreater(d2.chunks, 1)
        self.assertEqual(self.store.stats()["dokumente"], 1)
        self.assertEqual(self.store.stats()["abschnitte"], d2.chunks)
        self.assertEqual(self.store.search("Propeller"), [])
        self.assertTrue(self.store.search("Akkus"))
        self.assertFalse(set(old_chunk_ids) & {c.id for c in self.store.chunks(d2.id)})

    def test_add_file_errors(self):
        with self.assertRaises(FileNotFoundError):
            self.store.add_file(self.p("fehlt.txt"))
        bad = self.p("bin.txt")
        with open(bad, "wb") as fh:
            fh.write(b"\x00\x01\x02")
        with self.assertRaises(ValueError):
            self.store.add_file(bad)
        self.assertEqual(self.store.stats()["dokumente"], 0)

    def test_get_by_path(self):
        path = write(self.p("x.txt"), "Inhalt")
        d = self.store.add_file(path)
        self.assertEqual(self.store.get_by_path(path).id, d.id)
        self.assertIsNone(self.store.get_by_path(self.p("y.txt")))

    def test_list_project_filter_includes_global(self):
        self.store.add_text("A", "Inhalt A", project="P")
        self.store.add_text("B", "Inhalt B", project="Q")
        self.store.add_text("G", "Inhalt global")
        self.assertEqual({d.title for d in self.store.list(project="P")}, {"A", "G"})
        self.assertEqual(len(self.store.list()), 3)

    def test_stats(self):
        self.store.add_text("A", "Inhalt A", project="P")
        self.store.add_text("B", "Inhalt B " * 50, project="Q")
        st = self.store.stats()
        self.assertEqual(st["dokumente"], 2)
        self.assertEqual(st["abschnitte"], 2)
        self.assertEqual((st["mit_vektor"], st["ohne_vektor"]), (0, 2))
        self.assertEqual(st["projekte"], ["P", "Q"])
        self.assertEqual(st["groesse_bytes"], len("Inhalt A") + len(("Inhalt B " * 50).strip()))

    def test_close_idempotent(self):
        self.store.close()
        self.store.close()


class DirectoryAndSyncTest(TempCase):
    def setUp(self):
        super().setUp()
        self.store = KnowledgeStore(":memory:")
        self.docs = self.p("docs")
        write(self.p("docs", "a.md"), "Dokument A über Rahmen aus Carbon.")
        write(self.p("docs", "sub", "b.txt"), "Dokument B über Motoren.")
        write(self.p("docs", "c.exe"), "nicht unterstützt")
        write(self.p("docs", ".versteckt.md"), "versteckte Datei")
        write(self.p("docs", ".git", "config"), "git")
        write(self.p("docs", "__pycache__", "x.py"), "print(1)")
        write(self.p("docs", "node_modules", "m.js"), "module.exports = 1")
        write(self.p("docs", ".cache", "d.md"), "cache")
        make_docx(self.p("docs", "d.docx"), ["Dokument D aus Word."])

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def test_add_directory_defaults(self):
        res = self.store.add_directory(self.docs, project="Drohne")
        self.assertEqual(res["hinzugefuegt"], 3)
        self.assertEqual(res["unveraendert"], 0)
        self.assertEqual(res["fehler"], [])
        skipped = {os.path.basename(p) for p in res["uebersprungen"]}
        self.assertEqual(skipped, {"c.exe", ".versteckt.md"})
        titles = {d.title for d in self.store.list()}
        self.assertEqual(titles, {"a.md", "b.txt", "d.docx"})
        self.assertTrue(all(d.project == "Drohne" for d in self.store.list()))
        # zweiter Lauf: alles unverändert
        res2 = self.store.add_directory(self.docs)
        self.assertEqual((res2["hinzugefuegt"], res2["unveraendert"]), (0, 3))

    def test_add_directory_non_recursive_and_patterns(self):
        res = self.store.add_directory(self.docs, recursive=False, patterns=[".md"])
        self.assertEqual(res["hinzugefuegt"], 1)
        self.assertEqual([d.title for d in self.store.list()], ["a.md"])
        res2 = self.store.add_directory(self.docs, patterns=["*.txt"])
        self.assertEqual(res2["hinzugefuegt"], 1)
        self.assertEqual({d.title for d in self.store.list()}, {"a.md", "b.txt"})

    def test_add_directory_max_files_and_errors(self):
        with open(self.p("docs", "bin.txt"), "wb") as fh:
            fh.write(b"\x00\x00binaer")
        res = self.store.add_directory(self.docs, max_files=2)
        self.assertEqual(res["hinzugefuegt"] + len([f for f in res["fehler"] if "bin.txt" in f[0]]), 2)
        self.assertTrue(any("Limit" in grund for _, grund in res["fehler"]))
        res_all = self.store.add_directory(self.docs)
        self.assertTrue(any(p.endswith("bin.txt") and "Binärdatei" in g for p, g in res_all["fehler"]))
        self.assertEqual(self.store.stats()["dokumente"], 3)

    def test_add_directory_errors(self):
        with self.assertRaises(FileNotFoundError):
            self.store.add_directory(self.p("nix"))
        with self.assertRaises(NotADirectoryError):
            self.store.add_directory(self.p("docs", "a.md"))

    def test_sync_detects_change_and_missing(self):
        self.store.add_directory(self.docs)
        a = self.p("docs", "a.md")
        b = self.p("docs", "sub", "b.txt")
        res0 = self.store.sync()
        self.assertEqual((res0["aktualisiert"], res0["unveraendert"], res0["fehlend"]), (0, 3, []))

        write(a, "Dokument A jetzt über Aluminium.")
        os.utime(a, (time.time() + 5, time.time() + 5))       # mtime sicher verändert
        os.remove(b)
        res = self.store.sync()
        self.assertEqual(res["aktualisiert"], 1)
        self.assertEqual(res["unveraendert"], 1)
        self.assertEqual(res["fehlend"], [os.path.realpath(b)])
        self.assertEqual(res["fehler"], [])
        self.assertEqual(self.store.search("Carbon"), [])
        self.assertEqual(self.store.search("Aluminium")[0].title, "a.md")
        doc_b = self.store.get_by_path(b)
        self.assertTrue(doc_b.missing)
        self.assertTrue(doc_b.to_dict()["fehlt"])
        self.assertEqual(self.store.stats()["fehlend"], 1)
        # Inhalt der fehlenden Datei bleibt durchsuchbar
        self.assertEqual(self.store.search("Motoren")[0].doc_id, doc_b.id)

        # Datei kommt zurück: Markierung verschwindet
        write(b, "Dokument B über Motoren.")
        res2 = self.store.sync()
        self.assertEqual(res2["fehlend"], [])
        self.assertFalse(self.store.get(doc_b.id).missing)

    def test_sync_touch_without_change_is_unchanged(self):
        path = write(self.p("t.txt"), "unverändert")
        self.store.add_file(path)
        os.utime(path, (time.time() + 10, time.time() + 10))
        res = self.store.sync()
        self.assertEqual((res["aktualisiert"], res["unveraendert"]), (0, 1))


class SearchTest(unittest.TestCase):
    TEXTS = {
        "Motor": ("Der Brushless-Motor hat 2300 KV und wiegt 28 g. Maximalstrom 25 A.", "Drohne"),
        "Akku": ("Der Akku ist ein 4S LiPo mit 1500 mAh und 100C Entladerate.", "Drohne"),
        "Kuchen": ("Für den Apfelkuchen braucht man 1 kg Äpfel, Zimt und Mürbeteig.", "Backen"),
        "Allgemein": ("Sicherheitshinweis: Akku nie unbeaufsichtigt laden.", None),
    }

    def fill(self, store):
        for title, (text, project) in self.TEXTS.items():
            store.add_text(title, text, project=project)

    def test_search_without_embedder(self):
        store = KnowledgeStore(":memory:")
        self.fill(store)
        hits = store.search("Motor Maximalstrom")
        self.assertEqual(hits[0].title, "Motor")
        self.assertEqual(hits[0].cite(), "[Motor §1]")
        self.assertGreater(hits[0].score, 0)
        self.assertEqual(store.search("Quantenphysik"), [])
        self.assertEqual(store.search(""), [])
        self.assertEqual(store.search("Motor", k=0), [])
        self.assertEqual(store.stats()["mit_vektor"], 0)
        store.close()

    def test_search_prefix_umlaut_and_title(self):
        store = KnowledgeStore(":memory:")
        self.fill(store)
        self.assertEqual(store.search("ÄPFEL")[0].title, "Kuchen")
        self.assertEqual(store.search("Apfelkuchen")[0].title, "Kuchen")   # Präfix "apfelkuche"*
        self.assertEqual(store.search("Kuchen")[0].title, "Kuchen")        # Titeltreffer
        store.close()

    def test_search_project_filter_includes_global(self):
        store = KnowledgeStore(":memory:")
        self.fill(store)
        hits = store.search("Akku", project="Drohne")
        self.assertEqual({h.title for h in hits}, {"Akku", "Allgemein"})
        self.assertEqual({h.title for h in store.search("Akku", project="Backen")}, {"Allgemein"})
        self.assertEqual({h.title for h in store.search("Akku")}, {"Akku", "Allgemein"})
        store.close()

    def test_search_with_embedder(self):
        backend = FakeBackend(embed_dim=256)
        store = KnowledgeStore(":memory:", embedder=backend.embed)
        self.fill(store)
        st = store.stats()
        self.assertEqual((st["mit_vektor"], st["ohne_vektor"], st["embedding_dim"]), (4, 0, 256))
        hits = store.search("Motor KV Maximalstrom")
        self.assertEqual(hits[0].title, "Motor")
        self.assertTrue(all(h.score <= hits[0].score for h in hits))
        # Vektor-Treffer ohne Volltext-Treffer: gleiche Wörter, aber nur über Kosinus
        only_vec = store.search("Äpfel Zimt Mürbeteig Apfelkuchen")
        self.assertEqual(only_vec[0].title, "Kuchen")
        store.close()

    def test_k_and_min_score(self):
        store = KnowledgeStore(":memory:")
        self.fill(store)
        self.assertEqual(len(store.search("Akku", k=1)), 1)
        self.assertEqual(store.search("Akku", min_score=1.1), [])
        store.close()

    def test_embedder_failure_tolerated_and_reindex(self):
        def broken(texts):
            raise RuntimeError("kaputt")
        store = KnowledgeStore(":memory:", embedder=broken)
        self.fill(store)
        self.assertEqual(store.stats()["mit_vektor"], 0)
        self.assertEqual(store.search("Motor")[0].title, "Motor")
        store.set_embedder(FakeBackend(embed_dim=64).embed)
        progress = []
        self.assertEqual(store.reindex(progress=lambda d, t: progress.append((d, t))), 4)
        self.assertEqual(progress[-1], (4, 4))
        self.assertEqual((store.stats()["mit_vektor"], store.stats()["embedding_dim"]), (4, 64))
        self.assertEqual(store.reindex(), 0)
        # Modellwechsel: alte Vektoren werden ignoriert, only_missing findet nichts, reindex alles
        store.set_embedder(FakeBackend(embed_dim=32).embed)
        store.add_text("Neu", "Neuer Eintrag mit anderem Modell")
        self.assertEqual((store.stats()["mit_vektor"], store.stats()["ohne_vektor"]), (1, 4))
        self.assertEqual(store.search("Motor")[0].title, "Motor")   # Volltext trägt weiter
        self.assertEqual(store.reindex(only_missing=True), 0)
        self.assertEqual(store.reindex(), 4)
        self.assertEqual(store.stats()["ohne_vektor"], 0)
        store.close()

    def test_reindex_without_embedder(self):
        store = KnowledgeStore(":memory:")
        self.fill(store)
        self.assertEqual(store.reindex(), 0)
        store.close()


class ContextSectionTest(unittest.TestCase):
    def setUp(self):
        self.store = KnowledgeStore(":memory:")
        self.store.add_text("Handbuch", "Der Regler verträgt maximal 40 A Dauerstrom.\nZweite Zeile.", project="Drohne")
        self.store.add_text("Lang", "Regler " + "Füllwort " * 300)

    def tearDown(self):
        self.store.close()

    def test_format_and_citation(self):
        text = self.store.context_section("Regler Dauerstrom", k=1)
        self.assertTrue(text.startswith(CONTEXT_HEADER + "\n"))
        self.assertTrue(text.endswith("\n"))
        lines = text.rstrip("\n").split("\n")
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[1].startswith("[Handbuch §1] Der Regler verträgt maximal 40 A Dauerstrom. Zweite Zeile."))
        self.assertNotIn("\n", lines[1])

    def test_empty_without_hits(self):
        self.assertEqual(self.store.context_section("Quantenphysik"), "")
        self.assertEqual(self.store.context_section(""), "")

    def test_budget(self):
        text = self.store.context_section("Regler", k=4, max_chars=400)
        self.assertLessEqual(len(text), 401)
        self.assertIn("[gekürzt]", text)
        self.assertIn("[Handbuch §1]", text)
        full = self.store.context_section("Regler", k=4, max_chars=100000)
        self.assertNotIn("[gekürzt]", full)
        self.assertGreater(full.count("\n"), 2)

    def test_tiny_budget_gives_empty(self):
        self.assertEqual(self.store.context_section("Regler", max_chars=len(CONTEXT_HEADER) + 20), "")

    def test_project_filter(self):
        self.assertIn("[Handbuch §1]", self.store.context_section("Dauerstrom", project="Drohne"))
        self.assertEqual(self.store.context_section("Dauerstrom", project="Backen"), "")


# ------------------------------------------------------------------ Werkzeuge
class ToolsTest(TempCase):
    def setUp(self):
        super().setUp()
        self.ws = self.p("arbeit")
        os.makedirs(self.ws)
        write(self.p("arbeit", "docs", "regler.md"), "Der Regler verträgt maximal 40 A Dauerstrom.")
        write(self.p("arbeit", "docs", "akku.txt"), "Akku: 4S 1500 mAh.")
        write(self.p("draussen", "geheim.txt"), "streng geheim")
        self.store = KnowledgeStore(":memory:")
        self.reg = ToolRegistry(self.ws)
        register_tools(self.reg, self.store)

    def tearDown(self):
        self.store.close()
        super().tearDown()

    def test_registered(self):
        names = {t.name for t in self.reg.list()}
        self.assertEqual(names, {"dokumente_suchen", "dokument_hinzufuegen"})
        self.assertFalse(any(t.dangerous for t in self.reg.list()))
        self.assertIn("dokumente_suchen(frage, projekt?)", self.reg.describe())

    def test_add_file_via_tool_and_search(self):
        res = self.reg.run("dokument_hinzufuegen", {"pfad": "docs/regler.md", "projekt": "Drohne"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("»regler.md« indexiert: 1 Abschnitte", res.output)
        self.assertIn("Projekt »Drohne«", res.output)
        self.assertEqual(self.store.list()[0].project, "Drohne")
        again = self.reg.run("dokument_hinzufuegen", {"pfad": "docs/regler.md"})
        self.assertIn("bereits vorhanden", again.output)

        hit = self.reg.run("dokumente_suchen", {"frage": "Dauerstrom Regler"})
        self.assertTrue(hit.ok)
        self.assertIn("[regler.md §1]", hit.output)
        self.assertIn("40 A Dauerstrom", hit.output)
        self.assertIn("Relevanz", hit.output)
        none = self.reg.run("dokumente_suchen", {"frage": "Quantenphysik"})
        self.assertTrue(none.ok)
        self.assertEqual(none.output, "Keine passenden Dokument-Auszüge gefunden.")
        bad = self.reg.run("dokumente_suchen", {"frage": "  "})
        self.assertFalse(bad.ok)
        self.assertFalse(self.reg.run("dokumente_suchen", {}).ok)

    def test_project_filter_through_tool(self):
        self.reg.run("dokument_hinzufuegen", {"pfad": "docs/regler.md", "projekt": "Drohne"})
        self.reg.run("dokument_hinzufuegen", {"pfad": "docs/akku.txt", "projekt": "Auto"})
        res = self.reg.run("dokumente_suchen", {"frage": "Regler Akku", "projekt": "Auto"})
        self.assertIn("akku.txt", res.output)
        self.assertNotIn("regler.md", res.output)

    def test_add_directory_via_tool(self):
        res = self.reg.run("dokument_hinzufuegen", {"pfad": "docs"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("2 neu/aktualisiert", res.output)
        self.assertEqual(self.store.stats()["dokumente"], 2)

    def test_outside_workspace_rejected(self):
        for pfad in ("../draussen/geheim.txt", self.p("draussen", "geheim.txt"), ".."):
            res = self.reg.run("dokument_hinzufuegen", {"pfad": pfad})
            self.assertFalse(res.ok, pfad)
            self.assertIn("außerhalb des Arbeitsbereichs", res.error)
        self.assertEqual(self.store.stats()["dokumente"], 0)

    def test_missing_file_and_empty_path(self):
        res = self.reg.run("dokument_hinzufuegen", {"pfad": "docs/fehlt.md"})
        self.assertFalse(res.ok)
        self.assertIn("nicht gefunden", res.error)
        self.assertFalse(self.reg.run("dokument_hinzufuegen", {"pfad": " "}).ok)
        self.assertFalse(self.reg.run("dokument_hinzufuegen", {}).ok)


# ------------------------------------------------------------------ Persistenz
class PersistenceTest(TempCase):
    def test_persists_across_reopen(self):
        db = self.p("daten", "wissen.db")
        path = write(self.p("doc.md"), "Persistenter Inhalt über Servos.")
        s1 = KnowledgeStore(db, embedder=FakeBackend(embed_dim=8).embed)
        d = s1.add_file(path, project="P")
        s1.add_text("Notiz", "Direkter Text.")
        s1.close()

        s2 = KnowledgeStore(db)
        st = s2.stats()
        self.assertEqual((st["dokumente"], st["abschnitte"], st["mit_vektor"], st["embedding_dim"]), (2, 2, 2, 8))
        self.assertEqual(s2.get(d.id).title, "doc.md")
        self.assertEqual(s2.search("Servos")[0].doc_id, d.id)
        # unveränderte Datei wird erkannt, kein zweites Dokument
        self.assertEqual(s2.add_file(path).id, d.id)
        self.assertEqual(s2.stats()["dokumente"], 2)
        s2.close()
        s2.close()


if __name__ == "__main__":
    unittest.main()
