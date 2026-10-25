"""Studio: KI-Figuren nehmen Aufträge an und bauen Webseiten, Apps, Pläne, PC-Pläne."""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from obito.brain import Brain
from obito.config import Config
from obito.llm import FakeBackend
from obito.server import ObitoServer
from obito.studio import FIGUREN, Studio, StudioStore, extract_html, markdown_to_html

SEITE = "<!doctype html><html><head><title>X</title></head><body><h1>Imbiss</h1></body></html>"


class StudioUnitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, chat):
        return Studio(StudioStore(self.dir / "s.db"), self.dir / "out", chat=chat, autostart=False)

    def test_extract_html(self):
        self.assertEqual(extract_html("Hier:\n```html\n" + SEITE + "\n```\nViel Spaß"), SEITE)
        self.assertEqual(extract_html(SEITE), SEITE)
        self.assertIsNone(extract_html("kein html"))

    def test_markdown_escaped(self):
        out = markdown_to_html("# Titel\n- **a** <script>\n1. b")
        self.assertIn("<h1>Titel</h1>", out)
        self.assertIn("<strong>a</strong> &lt;script&gt;", out)
        self.assertIn("<ol><li>b</li>", out.replace("\n", ""))

    def test_webseite_gebaut(self):
        st = self.make(lambda msgs, n: "```html\n" + SEITE + "\n```")
        rec = st.submit("pixel", "Webseite für meinen Imbiss")
        done = st.run_once(rec["id"])
        self.assertEqual(done["status"], "fertig")
        self.assertFalse(done["vorlage"])
        self.assertEqual(done["vorschau"], f"/studio/{rec['id']}/")
        self.assertEqual(st.datei(rec["id"]).read_text(encoding="utf-8"), SEITE)
        st.close()

    def test_vorlage_ohne_modell(self):
        def kaputt(msgs, n):
            raise RuntimeError("weg")
        st = self.make(kaputt)
        rec = st.submit("byte", "Snake <b>Spiel</b>")
        done = st.run_once(rec["id"])
        self.assertEqual(done["status"], "fertig")
        self.assertTrue(done["vorlage"])
        text = st.datei(rec["id"]).read_text(encoding="utf-8")
        self.assertIn("&lt;b&gt;", text)          # Auftrag wird escaped
        st.close()

    def test_pc_plan_fuehrt_nichts_aus(self):
        plan = {"zusammenfassung": "Aufräumen", "schritte": [{"beschreibung": "Ordner ansehen", "befehl": "dir", "risiko": "niedrig"}]}
        st = self.make(lambda msgs, n: json.dumps(plan))
        done = st.run_once(st.submit("kernel", "Räume Downloads auf")["id"])
        self.assertEqual(done["status"], "fertig")
        text = st.datei(done["id"]).read_text(encoding="utf-8")
        self.assertIn("Nichts davon wurde ausgeführt", text)
        self.assertIn("<code>dir</code>", text)
        st.close()

    def test_eingaben_pruefen(self):
        st = self.make(None)
        with self.assertRaises(ValueError):
            st.submit("niemand", "x")
        with self.assertRaises(ValueError):
            st.submit("pixel", "   ")
        self.assertEqual({f["id"] for f in st.figuren()}, set(FIGUREN))
        st.close()

    def test_loeschen_und_neustart(self):
        st = self.make(None)
        rec = st.submit("nova", "Plan für eine App")
        st.store.update(rec["id"], status="arbeitet")
        st.close()
        st2 = self.make(None)                     # unterbrochener Auftrag wird wieder eingereiht
        self.assertEqual(st2.store.get(rec["id"])["status"], "wartend")
        st2.run_once(rec["id"])
        self.assertTrue(st2.delete(rec["id"]))
        self.assertFalse((self.dir / "out" / str(rec["id"])).exists())
        st2.close()


class StudioServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cfg = Config(backend="fake", data_dir=cls.tmp.name, workspace=cls.tmp.name)
        cls.brain = Brain(cfg, backend=FakeBackend(embed_dim=32, responder=lambda m, k: SEITE))
        cls.server = ObitoServer(cls.brain, "127.0.0.1", 0)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.server.port}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.brain.close()
        cls.tmp.cleanup()

    def call(self, path, body=None, method=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method or ("POST" if body is not None else "GET"))
        req.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(req, timeout=30) as r:
                return r.status, r.read(), dict(r.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def test_spielseite(self):
        status, body, headers = self.call("/spiel")
        self.assertEqual(status, 200)
        self.assertIn(b"OBITO Studio", body)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])

    def test_auftrag_ablauf(self):
        status, body, _ = self.call("/api/studio/figuren")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["figuren"]), len(FIGUREN))

        status, body, _ = self.call("/api/studio/auftraege", {"figur": "pixel", "auftrag": "Webseite für Imbiss"})
        self.assertEqual(status, 201)
        aid = json.loads(body)["auftrag"]["id"]
        rec = None
        for _ in range(100):
            rec = json.loads(self.call(f"/api/studio/auftraege/{aid}")[1])["auftrag"]
            if rec["status"] in ("fertig", "fehler"):
                break
            time.sleep(0.05)
        self.assertEqual(rec["status"], "fertig", rec)

        status, body, headers = self.call(rec["vorschau"])
        self.assertEqual(status, 200)
        self.assertEqual(body.decode("utf-8"), SEITE)
        self.assertTrue(headers["Content-Security-Policy"].startswith("sandbox"))
        self.assertNotIn("allow-same-origin", headers["Content-Security-Policy"])

        status, body, _ = self.call("/api/studio/auftraege")
        self.assertEqual(json.loads(body)["statistik"]["fertig"], 1)
        self.assertEqual(self.call(f"/api/studio/auftraege/{aid}", method="DELETE")[0], 200)
        self.assertEqual(self.call(f"/studio/{aid}/")[0], 404)

    def test_fehlerhafte_eingaben(self):
        self.assertEqual(self.call("/api/studio/auftraege", {"figur": "x", "auftrag": "y"})[0], 400)
        self.assertEqual(self.call("/api/studio/auftraege", {"figur": "pixel"})[0], 400)
        self.assertEqual(self.call("/api/studio/auftraege/999")[0], 404)
        self.assertEqual(self.call("/studio/999/")[0], 404)


if __name__ == "__main__":
    unittest.main()
