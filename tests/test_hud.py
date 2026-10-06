"""Tests für obito/static/index.html – das HUD ist reines HTML/CSS/JS ohne externe Ressourcen.

Geprüft wird ohne Browser: Existenz, HTML-Grundgerüst (Doctype, ``lang="de"``, ``<title>``),
ausgeglichene Tags, keine externen ``http(s)://``-Ressourcen, alle benötigten API-Pfade,
die SSE-Ereignisnamen und die deutschen Kernelemente. Zusätzlich wird die Datei über den echten
``ObitoServer`` (``GET /``) ausgeliefert – ohne Modell, nur gegen 127.0.0.1.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

from obito import server as server_mod
from obito.config import Config
from obito.memory import KINDS

INDEX = Path(__file__).resolve().parent.parent / "obito" / "static" / "index.html"

API_PATHS = ("/api/status", "/api/frage", "/api/feedback", "/api/erinnerungen", "/api/lektionen", "/api/modelle")
VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
             "source", "track", "wbr"}


class _TagBalance(HTMLParser):
    """Zählt öffnende/schließende Tags (ohne Void-Elemente) und merkt sich Fehlpaarungen."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.titles: list[str] = []
        self._in_title = False
        self.ids: list[str] = []
        self.label_for: list[str] = []
        self.inputs_with_label: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "label" and attrs.get("for"):
            self.label_for.append(attrs["for"])
        if tag in ("input", "textarea", "select") and attrs.get("type") not in ("hidden", "submit", "button"):
            if attrs.get("id"):
                self.inputs_with_label.append(attrs["id"])
        if tag in VOID_TAGS:
            return
        if tag == "title":
            self._in_title = True
            self.titles.append("")
        self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in VOID_TAGS:
            self.stack.pop()

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        if tag == "title":
            self._in_title = False
        if not self.stack:
            self.errors.append(f"schließendes </{tag}> ohne Öffnung")
            return
        if self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            self.errors.append(f"</{tag}> schließt, aber <{self.stack[-1]}> ist noch offen")
            while self.stack and self.stack[-1] != tag:
                self.stack.pop()
            self.stack.pop()
        else:
            self.errors.append(f"</{tag}> ohne passendes öffnendes Tag")

    def handle_data(self, data):
        if self._in_title and self.titles:
            self.titles[-1] += data


class _StubTools:
    def set_policy(self, confirm, confirm_dangerous):
        self.policy = (confirm, confirm_dangerous)


class _StubBrain:
    """Minimaler Brain-Ersatz – ``GET /`` berührt den Denkkern nicht."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.tools = _StubTools()
        self.busy = False


class HudFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = INDEX.read_text(encoding="utf-8")
        cls.parser = _TagBalance()
        cls.parser.feed(cls.html)
        cls.parser.close()
        m = re.search(r"<script[^>]*>(.*?)</script>", cls.html, re.S)
        cls.script = m.group(1) if m else ""

    def test_file_exists_and_matches_server_path(self):
        self.assertTrue(INDEX.is_file(), f"{INDEX} fehlt")
        self.assertEqual(Path(server_mod.INDEX_PATH).resolve(), INDEX.resolve())
        self.assertGreater(INDEX.stat().st_size, 10_000, "index.html ist verdächtig klein")

    def test_basic_html_skeleton(self):
        head = self.html.lstrip()[:200].lower()
        self.assertTrue(head.startswith("<!doctype html>"), "Doctype fehlt oder steht nicht am Anfang")
        self.assertRegex(self.html, r'<html[^>]*\blang="de"')
        self.assertRegex(self.html, r'<meta\s+charset="utf-8"', )
        self.assertRegex(self.html, r'<meta\s+name="viewport"')
        self.assertEqual(len(self.parser.titles), 1)
        self.assertIn("OBITO", self.parser.titles[0])
        self.assertIn("</body>", self.html)
        self.assertTrue(self.html.rstrip().endswith("</html>"))

    def test_tags_balanced(self):
        self.assertEqual(self.parser.errors, [])
        self.assertEqual(self.parser.stack, [], f"nicht geschlossene Tags: {self.parser.stack}")

    def test_ids_unique(self):
        seen, dupes = set(), set()
        for i in self.parser.ids:
            (dupes if i in seen else seen).add(i)
        self.assertEqual(dupes, set(), f"doppelte IDs: {sorted(dupes)}")

    def test_form_fields_have_labels(self):
        """Barrierefreiheit: jedes sichtbare Eingabefeld mit ID hat ein <label for=…>."""
        for field_id in self.parser.inputs_with_label:
            if field_id.startswith("tiefe-"):
                continue  # Radio-Gruppe – Labels folgen direkt (per for=)
            self.assertIn(field_id, self.parser.label_for, f"Feld #{field_id} ohne Label")
        for field_id in ("frage", "projekt", "sitzung", "mem-search", "mem-inhalt", "mem-art", "sys-model-select"):
            self.assertIn(field_id, self.parser.label_for)
        for depth in ("auto", "schnell", "mittel", "tief"):
            self.assertIn(f"tiefe-{depth}", self.parser.label_for)

    def test_no_external_resources(self):
        self.assertNotRegex(self.html, r"https?://", "externe URL gefunden")
        self.assertNotRegex(self.html, r"<script[^>]+\bsrc=", "externes Skript")
        self.assertNotRegex(self.html, r'<link[^>]+rel="stylesheet"', "externes Stylesheet")
        self.assertNotIn("@import", self.html)
        self.assertNotRegex(self.html, r"url\(\s*['\"]?(?!data:)[a-z]+:", "externe CSS-Ressource")
        self.assertNotRegex(self.html, r"<(img|iframe|object|embed|video|audio)\b", "eingebettete Medien")
        # Die inline-SVG/-Icons kommen ohne http-Namensraum aus; ein Favicon ist höchstens data:
        for m in re.finditer(r'<link[^>]*href="([^"]+)"', self.html):
            self.assertTrue(m.group(1).startswith("data:"), m.group(0))

    def test_api_paths_referenced(self):
        for path in API_PATHS:
            self.assertIn(path, self.script, f"API-Pfad {path} wird nicht verwendet")
        # DELETE-Routen mit ID
        self.assertIn("'/api/erinnerungen/'", self.script)
        self.assertIn("'/api/lektionen/'", self.script)
        self.assertIn("'DELETE'", self.script)

    def test_question_request_and_sse_protocol(self):
        self.assertIn("'/api/frage'", self.script)
        self.assertIn("stream: true", self.script)
        for key in ("frage", "sitzung", "tiefe", "projekt"):
            self.assertRegex(self.script, rf"\b{key}\b")
        for event in ("schritt", "token", "antwort", "fehler"):
            self.assertIn(f"'{event}'", self.script, f"SSE-Ereignis {event} wird nicht behandelt")
        self.assertIn("getReader()", self.script)        # fetch + ReadableStream
        self.assertIn("TextDecoder", self.script)
        self.assertIn("'\\n\\n'", self.script)           # Leerzeile trennt Ereignisse
        self.assertIn("'data'", self.script)
        self.assertIn("'event'", self.script)
        self.assertNotIn("EventSource", self.script)     # POST-Streaming geht nur über fetch

    def test_feedback_protocol(self):
        self.assertIn("'/api/feedback'", self.script)
        self.assertIn("interaktion_id", self.script)
        self.assertIn("bewertung: 1", self.script)
        self.assertIn("bewertung: -1", self.script)
        self.assertIn("kommentar", self.script)
        self.assertIn("umschreiben: true", self.script)
        self.assertIn("korrektur_vorschlag", self.script)
        self.assertIn("lektionen", self.script)
        for glyph in ("👍", "👎", "✎ Korrektur"):
            self.assertIn(glyph, self.html)

    def test_answer_fields_used(self):
        for key in ("experten", "erinnerungen_neu", "werkzeuge", "interaktion_id", "kritik", "tiefe", "dauer"):
            self.assertIn(key, self.script, f"Answer-Feld {key} wird nicht angezeigt")
        for key in ("stufe", "wer", "zusammenfassung", "detail", "dauer", "tokens", "status"):
            self.assertIn(key, self.script, f"Step-Feld {key} wird nicht genutzt")
        for key in ("art", "inhalt", "wichtigkeit", "geholfen", "geschadet"):
            self.assertIn(key, self.script)

    def test_memory_kinds_and_depths_offered(self):
        for kind in KINDS:
            self.assertRegex(self.html, rf'<option value="{kind}"')
        for depth in server_mod.DEPTHS:
            self.assertRegex(self.html, rf'name="tiefe" id="tiefe-{depth}" value="{depth}"')

    def test_german_ui_and_header(self):
        for text in ("OBITO", "v4.0", "AI Engineering Nexus", "LOKAL", "OFFLINE", "Denk-Spur", "Gedächtnis",
                     "Lektionen", "System", "Senden", "Abbrechen", "Korrektur", "Erinnerung", "Modell"):
            self.assertIn(text, self.html, f"Text »{text}« fehlt")
        self.assertIn("OBITO v4.0 · AI Engineering Nexus · LOKAL · OFFLINE", self.html)

    def test_error_handling_messages(self):
        self.assertIn("409", self.script)
        self.assertIn("503", self.script)
        self.assertIn("beschäftigt", self.script.lower() + self.html.lower())
        self.assertIn("nicht erreichbar", self.script)
        self.assertIn("AbortController", self.script)

    def test_keyboard_and_responsive(self):
        self.assertIn("'Enter'", self.script)
        self.assertIn("shiftKey", self.script)
        self.assertRegex(self.html, r"@media\s*\(max-width:\s*1100px\)")
        self.assertRegex(self.html, r"prefers-reduced-motion")

    def test_no_placeholder_leftovers(self):
        self.assertNotRegex(self.html, r"\bTODO\b|\bFIXME\b|lorem ipsum", "Platzhalter gefunden")

    def test_html_escaping_present(self):
        """Modelltext wird escaped, bevor Markdown-Muster angewendet werden."""
        self.assertIn("&amp;", self.script)
        self.assertIn("&lt;", self.script)
        self.assertRegex(self.script, r"function esc\(")

    @unittest.skipUnless(shutil.which("node"), "node nicht installiert – JS-Syntaxprüfung übersprungen")
    def test_javascript_syntax(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "hud.js"
            p.write_text(self.script, encoding="utf-8")
            proc = subprocess.run(["node", "--check", str(p)], capture_output=True, text=True, timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)


class HudServedTest(unittest.TestCase):
    """``GET /`` des echten Servers liefert genau diese Datei aus."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = Config(data_dir=cls.tmp.name, backend="fake", model="fake-modell")
        cls.server = server_mod.ObitoServer(_StubBrain(cfg), "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.thread.start()
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.tmp.cleanup()

    def test_index_served(self):
        with self.opener.open(f"http://127.0.0.1:{self.server.port}/", timeout=10) as resp:
            body = resp.read().decode("utf-8")
            self.assertEqual(resp.status, 200)
            self.assertTrue(resp.headers["Content-Type"].startswith("text/html"))
            csp = resp.headers.get("Content-Security-Policy", "")
        self.assertEqual(body, INDEX.read_text(encoding="utf-8"))
        self.assertNotIn("nicht gefunden", body)
        # Die CSP des Servers erlaubt nur 'self'/inline/data: – das HUD muss damit auskommen.
        self.assertIn("default-src 'self'", csp)
        self.assertIn("'unsafe-inline'", csp)
        self.assertIn("data:", csp)
        self.assertIn("connect-src 'self'", csp)


if __name__ == "__main__":
    unittest.main()


class HudPhase2Test(unittest.TestCase):
    """Phase 2: Tab-Leiste und die Ansichten Projekte, Datenzentrum, Missionen, Automationen, Modelle."""

    @classmethod
    def setUpClass(cls):
        from obito.server import INDEX_PATH
        cls.html = INDEX_PATH.read_text(encoding="utf-8")

    def test_tabs_present(self):
        for label in ("Übersicht", "Projekte", "Datenzentrum", "Missionen", "Automationen", "Modelle"):
            self.assertIn(">" + label, self.html)
        for view in ("uebersicht", "projekte", "daten", "missionen", "automationen", "modelle"):
            self.assertIn('id="view-' + view + '"', self.html)

    def test_phase2_api_paths_referenced(self):
        for path in ("/api/projekte", "/api/notizen/", "/api/dokumente", "/api/dokumente/suche", "/api/dokumente/sync",
                     "/api/missionen", "/api/automationen", "/api/automationen/vorschlaege", "/api/modell",
                     "/api/modelle/pull", "/api/modelle/"):
            self.assertIn(path, self.html, path)

    def test_phase2_protocol_details(self):
        self.assertIn("ev === 'fortschritt'", self.html)
        self.assertIn("/api/missionen/' + id + '/' + path", self.html)
        self.assertIn("answer.dokumente", self.html)
        self.assertIn("encodeURIComponent(name)", self.html)
        self.assertIn("window.OBITO", self.html)

    def test_second_script_syntax(self):
        import re
        import shutil
        import subprocess
        import tempfile
        node = shutil.which("node")
        if not node:
            self.skipTest("node nicht installiert")
        scripts = re.findall(r"<script>(.*?)</script>", self.html, re.S)
        self.assertEqual(len(scripts), 3)
        for sc in scripts:
            with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
                fh.write(sc)
                path = fh.name
            try:
                res = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=30)
                self.assertEqual(res.returncode, 0, res.stderr)
            finally:
                os.unlink(path)


class HudPhase3Test(unittest.TestCase):
    """Phase 3: Tabs System, Geräte, 3D-Modellierung, Simulation, Weltkarte."""

    @classmethod
    def setUpClass(cls):
        from obito.server import INDEX_PATH
        cls.html = INDEX_PATH.read_text(encoding="utf-8")
        scripts = re.findall(r"<script>(.*?)</script>", cls.html, re.S)
        cls.script = scripts[2] if len(scripts) > 2 else ""

    def test_tabs_and_views(self):
        for label in ("System", "Geräte", "3D-Modellierung", "Simulation", "Weltkarte"):
            self.assertIn(">" + label, self.html)
        for view in ("system", "geraete", "3d", "simulation", "welt"):
            self.assertIn('id="view-' + view + '"', self.html)

    def test_phase3_api_paths_referenced(self):
        for path in ("/api/system", "/api/system/verlauf", "/api/geraete", "/api/geraete/scan", "/api/geraete/lesen",
                     "/api/geraete/notiz", "/api/modelle3d", "/api/modelle3d/arten", "/stl", "/mesh",
                     "/api/simulation/arten", "/api/simulation", "/api/geo/orte", "/api/geo/routen", "/api/geo/plan",
                     "/api/geo/sonne", "/api/geo/wetter"):
            self.assertIn(path, self.script, path)

    def test_offline_map_and_no_hardcoded_tiles(self):
        # Kachel-URL kommt nur vom Server (Status), nie aus dem HTML
        self.assertNotIn("openstreetmap.org/", self.html)
        self.assertIn("kacheln_url", self.script)
        self.assertIn("terminator", self.script)
        self.assertIn("OpenStreetMap-Mitwirkende", self.script)

    def test_renderers_present(self):
        for fn in ("function drawChart", "function drawViewer", "function drawMap", "function prepareMesh",
                   "requestAnimationFrame", "getContext('2d')"):
            self.assertIn(fn, self.script)
        self.assertIn("devicePixelRatio", self.script)

    def test_german_texts(self):
        for text in ("Geräte erkennen", "Bauteil erzeugen", "Wasserdicht", "Flugplan", "Tag-Nacht-Grenze", "keine Geräte erfunden"):
            self.assertIn(text, self.html)
