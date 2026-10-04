"""Phase-2-Integration des Denkkerns: Wissensbasis, Projekte, Konsolidierung, Backup, Pflege."""

import json
import tempfile
import time
import unittest

from obito.agents import stage_of
from obito.brain import Brain
from obito.config import Config
from obito.llm import FakeBackend


def responder(msgs, kw):
    stage, who = stage_of(msgs)
    if stage == "routing":
        return '{"komplexitaet":"einfach","experten":[],"werkzeuge":false,"begruendung":"x"}'
    if stage == "schnell":
        return "Der Rahmen braucht 4 mm CFK. Entscheidung: wir nehmen Carbon."
    if stage == "extraktion":
        return json.dumps({"erinnerungen": [
            {"inhalt": "Entscheidung: Drohnenrahmen wird aus 4 mm Carbon gebaut", "art": "entscheidung",
             "wichtigkeit": 0.8, "tags": ["carbon"]},
            {"inhalt": "Nutzer bevorzugt Carbon Rahmen für Drohnen", "art": "praeferenz", "wichtigkeit": 0.7,
             "tags": []},
        ]})
    if stage == "konsolidierung":
        if "Erinnerungen:" in msgs[-1]["content"]:
            return '{"zusammenfassung":"Nutzer baut Carbon-Drohnen; bevorzugt 4 mm CFK.","behalten":[]}'
        return "Es ging um Drohnenrahmen aus Carbon; Entscheidung für 4 mm CFK; offen: Kosten."
    return f"[{stage}/{who}] ok"


class Phase2BrainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(backend="fake", data_dir=self.tmp.name, workspace=self.tmp.name, depth="auto",
                          parallel_calls=1)
        self.backend = FakeBackend(responder=responder, embed_dim=64)
        self.brain = Brain(self.cfg, backend=self.backend)
        self.addCleanup(self.brain.close)

    def test_stores_created_and_tools_registered(self):
        names = {t.name for t in self.brain.tools.list()}
        self.assertIn("dokumente_suchen", names)
        self.assertIn("material_info", names)
        self.assertIn("akku_rechner", names)
        self.assertTrue(self.cfg.knowledge_db.exists())
        self.assertTrue(self.cfg.projects_db.exists())
        self.assertTrue(self.cfg.missions_db.exists())
        self.assertTrue(self.cfg.automation_db.exists())
        self.assertFalse(self.brain.automation.running)

    def test_document_and_project_context_flow_into_prompt(self):
        self.brain.knowledge.add_text(
            "Datenblatt CFK",
            "CFK-Platte 4 mm: Dichte 1,6 g/cm³, Zugfestigkeit 600 MPa, geeignet für Drohnenrahmen.",
            project="Drohne")
        self.brain.projects.create("Drohne", "5-Zoll-Quadcopter")
        self.brain.projects.add_note("Drohne", "aufgabe", "Motoren bestellen")
        a = self.brain.ask("Welche Plattenstärke für den Drohnenrahmen aus CFK?", project="Drohne")
        self.assertEqual(a.depth, "schnell")
        self.assertEqual([c.cite() for c in a.documents], ["[Datenblatt CFK §1]"])
        self.assertIn("1 Dokument-Auszüge", a.steps[0].summary)
        self.assertIn("Projektkontext", a.steps[0].summary)
        fast_call = [c for c in self.backend.calls if stage_of(c["messages"])[0] == "schnell"][0]
        system = fast_call["messages"][0]["content"]
        self.assertIn("Projektkontext:", system)
        self.assertIn("Motoren bestellen", system)
        self.assertIn("[Datenblatt CFK §1]", system)
        d = a.to_dict()
        self.assertEqual(d["dokumente"][0]["titel"], "Datenblatt CFK")
        self.assertEqual(d["dokumente"][0]["abschnitt"], 1)

    def test_unknown_project_is_created_and_decisions_become_notes(self):
        a = self.brain.ask("Welche Plattenstärke für den Drohnenrahmen aus CFK?", project="Neu")
        self.assertIsNotNone(self.brain.projects.get("Neu"))
        kinds = {m.kind for m in a.new_memories}
        self.assertEqual(kinds, {"entscheidung", "praeferenz"})
        notes = self.brain.projects.notes("Neu", kind="entscheidung")
        self.assertEqual(len(notes), 1)
        self.assertTrue(notes[0].title.startswith("Entscheidung: Drohnenrahmen"))
        self.assertTrue(any("Projektnotiz" in line for line in a.steps[-1].detail.splitlines()))

    def _age_ki_memories(self, days: int = 30):
        with self.brain.memory._lock:
            self.brain.memory._db.execute(
                "UPDATE memories SET created_at = created_at - ?, last_access = last_access - ? WHERE source = 'ki'",
                (days * 86400, days * 86400))
            self.brain.memory._db.commit()

    def test_consolidate_merges_old_ki_memories_but_keeps_user_memories(self):
        user = self.brain.remember("Nutzer heißt Alex", kind="fakt", project="Drohne", importance=0.2)
        for i in range(3):
            self.brain.memory.remember(f"Alte Notiz {i} über Carbon Rahmen", source="ki", importance=0.2,
                                       project="Drohne")
        self._age_ki_memories()
        # zu wenige Kandidaten ohne Projekt -> keine Gruppe
        self.brain.memory.remember("Einzelne alte Notiz", source="ki", importance=0.1)
        self._age_ki_memories()
        before_calls = len(self.backend.calls)
        result = self.brain.consolidate(days=7)
        self.assertEqual(result["zusammengefasst"], 1)
        self.assertEqual(result["geloescht"], 3)
        self.assertEqual(result["fehler"], [])
        self.assertIsNotNone(self.brain.memory.get(user.id))
        summaries = self.brain.memory.list(kind="zusammenfassung")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].project, "Drohne")
        self.assertIn("Carbon", summaries[0].content)
        self.assertGreater(len(self.backend.calls), before_calls)
        # zweiter Lauf: nichts mehr zu tun
        self.assertEqual(self.brain.consolidate(days=7)["zusammengefasst"], 0)

    def test_consolidate_respects_keep_list(self):
        mems = [self.brain.memory.remember(f"Notiz {i} Carbon", source="ki", importance=0.2) for i in range(3)]
        self._age_ki_memories()
        keep_id = mems[1].id
        self.backend.responder = lambda msgs, kw: (
            json.dumps({"zusammenfassung": "Zusammenfassung Carbon", "behalten": [keep_id]})
            if stage_of(msgs)[0] == "konsolidierung" else "x")
        result = self.brain.consolidate(days=7)
        self.assertEqual((result["zusammengefasst"], result["geloescht"]), (1, 2))
        self.assertIsNotNone(self.brain.memory.get(keep_id))

    def test_consolidate_summarises_long_sessions_once(self):
        for i in range(34):
            self.brain.memory.add_message("lang", "user" if i % 2 == 0 else "assistant", f"Nachricht {i} Drohne")
        result = self.brain.consolidate(days=7)
        self.assertEqual(result["sitzungen"], 1)
        summaries = self.brain.memory.list(kind="zusammenfassung")
        self.assertEqual(len(summaries), 1)
        self.assertIn("sitzung", summaries[0].tags)
        self.assertEqual(self.brain.consolidate(days=7)["sitzungen"], 0)     # Marker gesetzt
        for i in range(32):
            self.brain.memory.add_message("lang", "user", f"Weitere {i}")
        self.assertEqual(self.brain.consolidate(days=7)["sitzungen"], 1)     # erst nach > 30 neuen Nachrichten

    def test_consolidate_reports_model_errors(self):
        for i in range(3):
            self.brain.memory.remember(f"Notiz {i} Carbon", source="ki", importance=0.2)
        self._age_ki_memories()
        self.backend.up = False
        result = self.brain.consolidate(days=7)
        self.assertEqual(result["zusammengefasst"], 0)
        self.assertEqual(len(result["fehler"]), 1)
        self.assertEqual(self.brain.memory.count(), 3)

    def test_backup_and_pruning(self):
        self.brain.remember("Sicherung testen", kind="fakt")
        first = self.brain.backup(keep=5)
        names = sorted(p.name for p in first.iterdir())
        self.assertEqual(names, ["automationen.db", "config.json", "gedaechtnis.db", "gedaechtnis.json", "lernen.db",
                                 "manifest.json", "missionen.db", "projekte.db", "wissen.db"])
        manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["gedaechtnis"]["erinnerungen"], 1)
        self.assertIn("gedaechtnis.db", manifest["dateien"])
        exported = json.loads((first / "gedaechtnis.json").read_text(encoding="utf-8"))
        self.assertEqual(exported[0]["content"], "Sicherung testen")
        # Kopie ist eine gültige SQLite-Datenbank
        import sqlite3
        conn = sqlite3.connect(str(first / "gedaechtnis.db"))
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 1)
        conn.close()
        second = self.brain.backup(keep=1)
        remaining = sorted(p.name for p in self.cfg.backups_dir.iterdir() if p.is_dir())
        self.assertEqual(remaining, [second.name])

    def test_set_dangerous_policy(self):
        self.assertFalse(self.brain.automation.allow_dangerous)
        self.brain.set_dangerous_policy(True)
        self.assertTrue(self.brain.automation.allow_dangerous)
        res = self.brain.tools.run("python_ausfuehren", {"code": "print(2+3)"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("5", res.output)
        self.brain.set_dangerous_policy(False)
        res = self.brain.tools.run("python_ausfuehren", {"code": "print(1)"})
        self.assertFalse(res.ok)

    def test_status_and_close(self):
        s = self.brain.status()
        self.assertEqual(s["missionen"], {"laufend": [], "anzahl": 0})
        self.assertIn("dokumente", s["wissen"])
        self.assertIn("aktiv", s["automationen"])
        self.brain.close()
        self.brain.close()


if __name__ == "__main__":
    unittest.main()
