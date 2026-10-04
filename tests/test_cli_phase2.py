"""Phase-2-Unterbefehle und Chat-Befehle der Kommandozeile."""

import io
import json
import os
import tempfile
import unittest

from obito.agents import stage_of
from obito.brain import Brain
from obito.cli import ChatSession, main
from obito.config import Config
from obito.llm import FakeBackend


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


class SubcommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = os.path.join(self.tmp.name, "daten")
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)
        with open("notizen.md", "w", encoding="utf-8") as fh:
            fh.write("# Notizen\nDer Motor hat 2300 KV und braucht einen 45-A-ESC.\n")

    def run_cli(self, *argv, inp=None, backend=None):
        out = io.StringIO()
        be = backend if backend is not None else FakeBackend(responder=responder, embed_dim=64)
        code = main(["--daten", self.data, "--backend", "fake", *argv], out=out,
                    inp=inp or (lambda prompt: "j"), backend=be)
        return code, out.getvalue()

    def test_wissen(self):
        code, out = self.run_cli("wissen", "list")
        self.assertEqual(code, 0)
        self.assertIn("Keine Dokumente", out)
        code, out = self.run_cli("wissen", "add", "notizen.md", "--projekt", "Drohne")
        self.assertEqual(code, 0)
        self.assertIn("Indexiert: #1 notizen.md", out)
        code, out = self.run_cli("wissen", "search", "Motor KV")
        self.assertEqual(code, 0)
        self.assertIn("[notizen.md §1]", out)
        code, out = self.run_cli("wissen", "list")
        self.assertIn("1 Dokumente", out)
        code, out = self.run_cli("wissen", "sync")
        self.assertEqual(code, 0)
        self.assertIn("unverändert", out)
        code, out = self.run_cli("wissen", "dir", ".")
        self.assertEqual(code, 0)
        self.assertIn("unverändert", out)
        code, out = self.run_cli("wissen", "remove", "1")
        self.assertEqual(code, 0)
        code, out = self.run_cli("wissen", "remove", "1")
        self.assertEqual(code, 1)
        code, out = self.run_cli("wissen", "add", "gibtsnicht.txt")
        self.assertEqual(code, 1)
        code, out = self.run_cli("wissen")
        self.assertEqual(code, 2)

    def test_projekt(self):
        code, out = self.run_cli("projekt", "neu", "Drohne", "--beschreibung", "5 Zoll", "--tags", "fpv, carbon")
        self.assertEqual(code, 0)
        self.assertIn("Drohne (aktiv) [carbon, fpv]", out)
        self.assertEqual(self.run_cli("projekt", "neu", "Drohne")[0], 1)
        code, out = self.run_cli("projekt", "aufgabe", "Drohne", "Motoren bestellen")
        self.assertEqual(code, 0)
        self.assertIn("☐ Motoren bestellen", out)
        code, out = self.run_cli("projekt", "entscheidung", "Drohne", "Carbon-Rahmen", "--inhalt", "4 mm CFK")
        self.assertIn("[entscheidung", out)
        code, out = self.run_cli("projekt", "info", "Drohne")
        self.assertEqual(code, 0)
        self.assertIn("Offene Aufgaben: Motoren bestellen", out)
        self.assertIn("Letzte Entscheidungen", out)
        code, out = self.run_cli("projekt", "erledigt", "1")
        self.assertIn("☑", out)
        code, out = self.run_cli("projekt", "datei", "Drohne", "notizen.md", "--beschreibung", "Notizen")
        self.assertIn("Datei #1 zugeordnet", out)
        code, out = self.run_cli("projekt", "list")
        self.assertIn("#1 Drohne", out)
        self.assertEqual(self.run_cli("projekt", "aufgabe", "Nope", "x")[0], 1)
        code, out = self.run_cli("projekt", "loeschen", "Drohne", inp=lambda p: "n")
        self.assertEqual(code, 1)
        code, out = self.run_cli("projekt", "archiv", "Drohne")
        self.assertIn("archiviert", out)
        code, out = self.run_cli("projekt", "list")
        self.assertIn("Noch keine Projekte", out)
        code, out = self.run_cli("projekt", "loeschen", "Drohne", "--ja")
        self.assertEqual(code, 0)
        self.assertEqual(self.run_cli("projekt", "info", "Drohne")[0], 1)

    def test_mission(self):
        code, out = self.run_cli("mission", "neu", "Rechne 6*7 und erkläre es", "--projekt", "Drohne")
        self.assertEqual(code, 0)
        self.assertIn("#1 Testmission [geplant · Drohne]", out)
        self.assertIn("Starten: python -m obito mission start 1", out)
        code, out = self.run_cli("mission", "start", "1")
        self.assertEqual(code, 0, out)
        self.assertIn("1. Rechne 6*7 [fertig] → 6*7 = 42", out)
        self.assertIn("[fertig · Drohne] 2/2 Schritte (100 %)", out)
        self.assertIn("Bericht: 42", out)
        code, out = self.run_cli("mission", "status", "1")
        self.assertIn("2/2 Schritte", out)
        code, out = self.run_cli("mission", "list", "--status", "fertig")
        self.assertIn("#1 Testmission", out)
        code, out = self.run_cli("mission", "neu", "Nochmal", "--start")
        self.assertEqual(code, 0)
        self.assertIn("[fertig]", out)
        code, out = self.run_cli("mission", "neu", "Hintergrund", "--start", "--hintergrund")
        self.assertEqual(code, 0)
        self.assertIn("im Hintergrund", out)
        self.assertEqual(self.run_cli("mission", "status", "99")[0], 1)
        self.assertEqual(self.run_cli("mission", "loeschen", "1")[0], 0)
        self.assertEqual(self.run_cli("mission", "loeschen", "1")[0], 1)
        self.assertEqual(self.run_cli("mission", "neu", "")[0], 1)

    def test_automation(self):
        code, out = self.run_cli("automation", "list")
        self.assertIn("Keine Automationen", out)
        code, out = self.run_cli("automation", "vorschlaege")
        self.assertIn("Einrichten:", out)
        code, out = self.run_cli("automation", "vorschlaege", "--installieren")
        self.assertEqual(code, 0)
        self.assertEqual(out.count("Eingerichtet:"), 3)
        code, out = self.run_cli("automation", "neu", "Backup", "--art", "backup", "--intervall", "60",
                                 "--parameter", '{"behalten": 2}')
        self.assertEqual(code, 0)
        self.assertIn("#4 Backup [backup, alle 60 min, aktiv]", out)
        self.assertEqual(self.run_cli("automation", "neu", "x", "--art", "backup", "--intervall", "5",
                                      "--parameter", "kein json")[0], 1)
        code, out = self.run_cli("automation", "jetzt", "4")
        self.assertEqual(code, 0, out)
        self.assertIn("[ok]", out)
        self.assertTrue(os.path.isdir(os.path.join(self.data, "backups")))
        code, out = self.run_cli("automation", "laeufe", "4")
        self.assertIn("[ok]", out)
        code, out = self.run_cli("automation", "inaktiv", "4")
        self.assertIn("inaktiv]", out)
        code, out = self.run_cli("automation", "log", "-n", "5")
        self.assertEqual(code, 0)
        self.assertEqual(self.run_cli("automation", "loeschen", "4")[0], 0)
        self.assertEqual(self.run_cli("automation", "loeschen", "4")[0], 1)
        self.assertEqual(self.run_cli("automation", "jetzt", "99")[0], 1)

    def test_modelle(self):
        be = FakeBackend(responder=responder, embed_dim=64, models=["fake-modell", "fake-embed", "qwen2.5:7b"])
        code, out = self.run_cli("modelle", "list", backend=be)
        self.assertEqual(code, 0)
        self.assertIn("fake-modell", out)
        self.assertIn("Hauptmodell", out)
        code, out = self.run_cli("modelle", "pull", "neu:3b", backend=be)
        self.assertEqual(code, 0)
        self.assertIn("Fertig: neu:3b", out)
        self.assertTrue(be.has_model("neu:3b"))
        code, out = self.run_cli("modelle", "wechseln", "qwen2.5:7b", "--nicht-speichern", backend=be)
        self.assertEqual(code, 0)
        self.assertIn("Hauptmodell: qwen2.5:7b", out)
        cfg_path = os.path.join(self.tmp.name, "obito.json")
        code, out = self.run_cli("--config", cfg_path, "modelle", "wechseln", "qwen2.5:7b", backend=be)
        self.assertEqual(code, 0)
        self.assertIn("Gespeichert in", out)
        self.assertEqual(json.load(open(cfg_path, encoding="utf-8"))["model"], "qwen2.5:7b")
        self.assertEqual(self.run_cli("modelle", "wechseln", "gibtsnicht", backend=be)[0], 1)
        code, out = self.run_cli("modelle", "loeschen", "neu:3b", backend=be)
        self.assertEqual(code, 0)
        self.assertEqual(self.run_cli("modelle", "loeschen", "neu:3b", backend=be)[0], 1)
        code, out = self.run_cli("modelle", "empfehlen", "8")
        self.assertEqual(code, 0)
        self.assertIn("qwen2.5:7b", out)

    def test_rechner(self):
        code, out = self.run_cli("rechner")
        self.assertEqual(code, 0)
        self.assertIn("akku_rechner(", out)
        code, out = self.run_cli("rechner", "akku_rechner", "zellen=4", "mah=1500")
        self.assertEqual(code, 0)
        self.assertIn("22,2 Wh", out)
        code, out = self.run_cli("rechner", "material", "Alu 7075")
        self.assertEqual(code, 0)
        self.assertIn("7075", out)
        code, out = self.run_cli("rechner", "vergleich", "cfk,petg")
        self.assertEqual(code, 0)
        self.assertIn("PETG", out)
        self.assertEqual(self.run_cli("rechner", "material", "unobtainium")[0], 1)
        self.assertEqual(self.run_cli("rechner", "unbekannt")[0], 2)
        self.assertEqual(self.run_cli("rechner", "akku_rechner", "zellen")[0], 2)
        self.assertEqual(self.run_cli("rechner", "akku_rechner", "zellen=4")[0], 1)   # mah fehlt


class ChatCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(backend="fake", data_dir=os.path.join(self.tmp.name, "d"), workspace=self.tmp.name)
        self.backend = FakeBackend(responder=responder, embed_dim=64)
        self.brain = Brain(self.cfg, backend=self.backend)
        self.addCleanup(self.brain.close)
        self.out = io.StringIO()
        self.answers: list[str] = []
        self.session = ChatSession(self.brain, self.cfg, out=self.out, inp=lambda p: self.answers.pop(0) if self.answers else "j")
        with open(os.path.join(self.tmp.name, "doku.md"), "w", encoding="utf-8") as fh:
            fh.write("ESC 45 A Dauerstrom, Motor 2300 KV.\n")

    def text(self) -> str:
        value = self.out.getvalue()
        self.out.seek(0)
        self.out.truncate(0)
        return value

    def test_documents(self):
        self.session.handle_line("/dokumente")
        self.assertIn("Noch keine Dokumente", self.text())
        self.session.handle_line(f"/dokument {os.path.join(self.tmp.name, 'doku.md')}")
        self.assertIn("Indexiert: #1 doku.md", self.text())
        self.session.handle_line("/dokumente ESC Dauerstrom")
        self.assertIn("[doku.md §1]", self.text())
        self.session.handle_line("/dokument /gibt/es/nicht")
        self.assertIn("nicht gefunden", self.text())
        self.session.handle_line(f"/dokument {self.tmp.name}")
        self.assertIn("unverändert", self.text())

    def test_project_commands(self):
        self.session.handle_line("/aufgabe Motoren bestellen")
        self.assertIn("Kein Projekt gesetzt", self.text())
        self.session.handle_line("/projekt Drohne")
        self.assertIn("Projekt: Drohne (neu angelegt)", self.text())
        self.session.handle_line("/aufgabe Motoren bestellen")
        self.assertIn("☐ Motoren bestellen", self.text())
        self.session.handle_line("/entscheidung Rahmen aus 4 mm CFK")
        self.assertIn("[entscheidung", self.text())
        self.session.handle_line("/projektinfo")
        text = self.text()
        self.assertIn("Offene Aufgaben: Motoren bestellen", text)
        self.assertIn("Letzte Entscheidungen", text)
        self.session.handle_line("/projektinfo Unbekannt")
        self.assertIn("unbekannt", self.text())

    def test_mission_commands(self):
        self.session.handle_line("/missionen")
        self.assertIn("Keine Missionen", self.text())
        self.answers = ["n"]
        self.session.handle_line("/mission Rechne 6*7 und erkläre es")
        text = self.text()
        self.assertIn("#1 Testmission [geplant]", text)
        self.assertIn("nicht gestartet", text)
        self.answers = ["j"]
        self.session.handle_line("/mission Rechne 6*7 und erkläre es")
        text = self.text()
        self.assertIn("1. Rechne 6*7 [fertig]", text)
        self.assertIn("2/2 Schritte (100 %)", text)
        self.session.handle_line("/missionen")
        self.assertIn("#2 Testmission [fertig]", self.text())

    def test_misc_commands(self):
        self.session.handle_line("/automationen")
        self.assertIn("Keine Automationen", self.text())
        self.session.handle_line("/material cfk")
        self.assertIn("CFK", self.text())
        self.session.handle_line("/material nix")
        self.assertIn("Unbekanntes Material", self.text())
        self.session.handle_line("/rechner")
        self.assertIn("akku_rechner(", self.text())
        self.session.handle_line("/rechner elektro_rechner u=12 r=47")
        self.assertIn("0,2553 A", self.text())
        self.session.handle_line("/rechner elektro_rechner u")
        self.assertIn("schluessel=wert", self.text())
        self.session.handle_line("/backup")
        self.assertIn("Sicherung angelegt", self.text())
        self.session.handle_line("/konsolidieren 7")
        self.assertIn("Konsolidiert:", self.text())
        self.session.handle_line("/hilfe")
        text = self.text()
        for cmd in ("/dokument", "/mission", "/rechner", "/backup", "/konsolidieren", "/projektinfo"):
            self.assertIn(cmd, text)


if __name__ == "__main__":
    unittest.main()
