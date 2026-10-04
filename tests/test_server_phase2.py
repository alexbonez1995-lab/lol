"""Phase-2-Endpunkte des Servers gegen einen echten Brain mit FakeBackend."""

import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from obito.agents import stage_of
from obito.brain import Brain
from obito.config import Config
from obito.llm import FakeBackend
from obito.server import ObitoServer


def responder(msgs, kw):
    stage, who = stage_of(msgs)
    if stage == "routing":
        return '{"komplexitaet":"einfach","experten":[],"werkzeuge":false,"begruendung":"x"}'
    if stage == "schnell":
        return "Antwort: 42."
    if stage == "mission":
        return json.dumps({"titel": "Testmission", "schritte": [
            {"beschreibung": "Rechne 6*7", "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "6*7"}},
            {"beschreibung": "Erkläre das Ergebnis", "art": "frage", "werkzeug": None, "args": {}},
        ]})
    if stage == "bericht":
        return "Bericht: 42 berechnet und erklärt."
    if stage == "extraktion":
        return '{"erinnerungen":[]}'
    if stage == "konsolidierung":
        return '{"zusammenfassung":"Zusammenfassung","behalten":[]}'
    return f"[{stage}/{who}] ok"


class Phase2ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(cls.tmp.name, "notizen.md"), "w", encoding="utf-8") as fh:
            fh.write("# Notizen\nDer Motor hat 2300 KV und braucht einen 45-A-ESC.\n")
        cls.cfg = Config(backend="fake", data_dir=cls.tmp.name, workspace=cls.tmp.name)
        cls.brain = Brain(cls.cfg, backend=FakeBackend(responder=responder, embed_dim=64))
        cls.server = ObitoServer(cls.brain, "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
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
        req = urllib.request.Request(self.base + path, data=data,
                                     method=method or ("POST" if body is not None else "GET"))
        req.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(req, timeout=60) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_01_projects(self):
        status, body = self.call("/api/projekte", {"name": "Drohne", "beschreibung": "5 Zoll", "tags": ["fpv"]})
        self.assertEqual(status, 200)
        self.assertEqual(body["projekt"]["name"], "Drohne")
        self.assertEqual(self.call("/api/projekte", {"name": "Drohne"})[0], 400)       # doppelt
        self.assertEqual(self.call("/api/projekte", {"name": "X", "tags": 5})[0], 400)
        status, body = self.call("/api/projekte/Drohne/notizen", {"art": "aufgabe", "titel": "Motoren bestellen"})
        self.assertEqual(status, 200)
        note_id = body["notiz"]["id"]
        self.assertEqual(self.call("/api/projekte/Nope/notizen", {"art": "notiz", "titel": "x"})[0], 404)
        self.assertEqual(self.call("/api/projekte/Drohne/notizen", {"art": "unsinn", "titel": "x"})[0], 400)
        status, body = self.call("/api/projekte/Drohne")
        self.assertEqual(status, 200)
        self.assertIn("Offene Aufgaben: Motoren bestellen", body["zusammenfassung"])
        self.assertEqual(len(body["notizen"]), 1)
        status, body = self.call(f"/api/notizen/{note_id}/erledigt", {"erledigt": True})
        self.assertTrue(body["notiz"]["erledigt"])
        self.assertEqual(self.call("/api/notizen/999/erledigt", {})[0], 404)
        self.assertEqual(self.call("/api/projekte")[1]["projekte"][0]["name"], "Drohne")
        self.assertEqual(self.call("/api/projekte/Unbekannt")[0], 404)

    def test_02_documents(self):
        status, body = self.call("/api/dokumente", {"pfad": "notizen.md", "projekt": "Drohne"})
        self.assertEqual(status, 200)
        self.assertEqual(body["dokument"]["titel"], "notizen.md")
        self.assertEqual(self.call("/api/dokumente", {"titel": "Datenblatt", "text": "ESC 45 A Dauerstrom"})[0], 200)
        self.assertEqual(self.call("/api/dokumente", {"pfad": "/etc/passwd"})[0], 403)
        self.assertEqual(self.call("/api/dokumente", {"pfad": "gibtsnicht.txt"})[0], 404)
        self.assertEqual(self.call("/api/dokumente", {"titel": "nur Titel"})[0], 400)
        self.assertEqual(self.call("/api/dokumente/suche")[0], 400)
        status, body = self.call("/api/dokumente/suche?q=Motor+KV")
        self.assertEqual(status, 200)
        self.assertEqual(body["treffer"][0]["titel"], "notizen.md")
        self.assertEqual(len(self.call("/api/dokumente")[1]["dokumente"]), 2)
        self.assertEqual(len(self.call("/api/dokumente?projekt=Drohne")[1]["dokumente"]), 2)   # projektlose dabei
        status, body = self.call("/api/dokumente/sync", {})
        self.assertEqual(status, 200)
        self.assertIn("unveraendert", body)
        doc_id = self.call("/api/dokumente")[1]["dokumente"][-1]["id"]
        self.assertEqual(self.call(f"/api/dokumente/{doc_id}", None, "DELETE")[0], 200)
        self.assertEqual(self.call(f"/api/dokumente/{doc_id}", None, "DELETE")[0], 404)

    def test_03_missions(self):
        self.assertEqual(self.call("/api/missionen", {"ziel": ""})[0], 400)
        status, body = self.call("/api/missionen", {"ziel": "Rechne 6*7 und erkläre es", "projekt": "Drohne"})
        self.assertEqual(status, 200)
        m = body["mission"]
        self.assertEqual(m["status"], "geplant")
        self.assertEqual([s["art"] for s in m["schritte"]], ["werkzeug", "frage"])
        self.assertEqual(self.call(f"/api/missionen/{m['id']}/start", {})[0], 200)
        for _ in range(100):
            st = self.call(f"/api/missionen/{m['id']}")[1]
            if st["mission"]["status"] in ("fertig", "fehler", "abgebrochen"):
                break
            time.sleep(0.05)
        self.assertEqual(st["mission"]["status"], "fertig")
        self.assertEqual(st["mission"]["fortschritt"], 1.0)
        self.assertIn("42", st["mission"]["schritte"][0]["ergebnis"])
        self.assertTrue(st["mission"]["bericht"].startswith("Bericht"))
        notes = self.call("/api/projekte/Drohne")[1]["notizen"]
        self.assertIn("ergebnis", {n["art"] for n in notes})
        listing = self.call("/api/missionen")[1]
        self.assertEqual(listing["laufend"], [])
        self.assertEqual(listing["missionen"][0]["id"], m["id"])
        self.assertEqual(self.call("/api/missionen/999")[0], 404)
        self.assertEqual(self.call("/api/missionen/999/start", {})[0], 404)
        self.assertEqual(self.call(f"/api/missionen/{m['id']}/stop", {})[0], 200)
        self.assertEqual(self.call(f"/api/missionen/{m['id']}", None, "DELETE")[0], 200)
        self.assertEqual(self.call(f"/api/missionen/{m['id']}", None, "DELETE")[0], 404)

    def test_04_automations(self):
        self.assertEqual(self.call("/api/automationen", {"name": "x", "art": "unsinn", "intervall_minuten": 5})[0], 400)
        self.assertEqual(self.call("/api/automationen", {"name": "x", "art": "backup"})[0], 400)
        status, body = self.call("/api/automationen", {"name": "Backup", "art": "backup", "intervall_minuten": 1440,
                                                       "parameter": {"behalten": 3}})
        self.assertEqual(status, 200)
        aid = body["automation"]["id"]
        self.assertTrue(body["automation"]["aktiv"])
        status, body = self.call(f"/api/automationen/{aid}/jetzt", {})
        self.assertEqual((status, body["status"]), (200, "ok"))
        self.assertTrue(self.cfg.backups_dir.exists())
        runs = self.call(f"/api/automationen/{aid}/laeufe")[1]["laeufe"]
        self.assertEqual(runs[0]["status"], "ok")
        self.assertFalse(self.call(f"/api/automationen/{aid}/aktiv", {"aktiv": False})[1]["automation"]["aktiv"])
        self.assertEqual(self.call("/api/automationen/999/jetzt", {})[0], 404)
        self.assertEqual(len(self.call("/api/automationen/vorschlaege")[1]["vorschlaege"]), 3)
        status, body = self.call("/api/automationen/vorschlaege", {"aktiv": False})
        self.assertEqual(status, 200)
        self.assertEqual(len(body["automationen"]), 3)
        listing = self.call("/api/automationen")[1]
        self.assertEqual(len(listing["automationen"]), 4)
        self.assertFalse(listing["zeitplaner_aktiv"])
        self.assertEqual(self.call(f"/api/automationen/{aid}", None, "DELETE")[0], 200)
        self.assertEqual(self.call(f"/api/automationen/{aid}", None, "DELETE")[0], 404)

    def test_05_maintenance_and_models(self):
        status, body = self.call("/api/pflege/konsolidieren", {"tage": 7})
        self.assertEqual(status, 200)
        self.assertIn("zusammengefasst", body)
        status, body = self.call("/api/pflege/backup", {"behalten": 2})
        self.assertEqual(status, 200)
        self.assertTrue(os.path.isdir(body["pfad"]))
        self.assertEqual(self.call("/api/modell", {"name": "gibtsnicht"})[0], 404)
        self.assertEqual(self.call("/api/modell", {"name": "fake-modell"})[0], 200)
        status, body = self.call("/api/modell", {"name": "fake-modell", "schnell": True})
        self.assertEqual(body["schnell"], "fake-modell")
        self.assertEqual(self.call("/api/modell", {"name": "", "schnell": True})[1]["schnell"], "")
        self.assertEqual(self.call("/api/modell", {"name": ""})[0], 400)
        models = self.call("/api/modelle")[1]
        self.assertEqual(models["aktuell"], "fake-modell")
        self.assertEqual(self.call("/api/modelle/fake-embed", None, "DELETE")[0], 200)
        self.assertEqual(self.call("/api/modelle/fake-embed", None, "DELETE")[0], 404)
        # Pull als SSE
        req = urllib.request.Request(self.base + "/api/modelle/pull", data=b'{"name": "neu:1b"}', method="POST")
        req.add_header("Content-Type", "application/json")
        with self.opener.open(req, timeout=30) as r:
            self.assertTrue(r.headers["Content-Type"].startswith("text/event-stream"))
            text = r.read().decode("utf-8")
        self.assertIn("event: fortschritt", text)
        self.assertIn("event: fertig", text)
        self.assertTrue(self.brain.backend.has_model("neu:1b"))
        s = self.call("/api/status")[1]
        for key in ("wissen", "projekte", "missionen", "automationen"):
            self.assertIn(key, s)

    def test_06_background_scheduler_start_stop(self):
        self.assertTrue(self.server.start_background())
        self.assertTrue(self.brain.automation.running)
        self.server.stop_background()
        self.assertFalse(self.brain.automation.running)


if __name__ == "__main__":
    unittest.main()
