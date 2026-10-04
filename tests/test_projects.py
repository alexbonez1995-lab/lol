import os
import tempfile
import time
import unittest

from obito.projects import (KINDS, LABEL_DECISIONS, LABEL_OPEN_TASKS, LABEL_PROBLEM, LABEL_VERSION,
                            NOTE_KINDS, STATUSES, Note, Project, ProjectStore, normalize_name)


class ProjectCrudTest(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_create_and_get(self):
        p = self.store.create("Drohne X", "Racing-Quad 5 Zoll", tags=["Drohne", "FPV", "drohne"])
        self.assertIsInstance(p, Project)
        self.assertEqual(p.name, "Drohne X")
        self.assertEqual(p.description, "Racing-Quad 5 Zoll")
        self.assertEqual(p.status, "aktiv")
        self.assertEqual(p.tags, ["drohne", "fpv"])
        self.assertGreater(p.created_at, 0)
        self.assertEqual(p.created_at, p.updated_at)
        self.assertEqual((p.open_tasks, p.note_count), (0, 0))
        self.assertEqual(self.store.get("Drohne X").id, p.id)
        self.assertEqual(self.store.get_by_id(p.id).name, "Drohne X")
        self.assertIsNone(self.store.get("gibt es nicht"))
        self.assertIsNone(self.store.get_by_id(999))
        self.assertIsNone(self.store.get_by_id("x"))
        self.assertIsNone(self.store.get(""))

    def test_name_lookup_tolerates_case_and_whitespace(self):
        p = self.store.create("  Drohne   X ")
        self.assertEqual(p.name, "Drohne X")
        self.assertEqual(self.store.get("drohne x").id, p.id)
        self.assertEqual(self.store.get("DROHNE  X").id, p.id)
        self.assertEqual(normalize_name(" Drohne  X"), "drohne x")

    def test_create_validation(self):
        with self.assertRaises(ValueError):
            self.store.create("   ")
        with self.assertRaises(ValueError):
            self.store.create("")
        with self.assertRaises(ValueError):
            self.store.create("x" * 200)
        self.store.create("Doppelt")
        with self.assertRaises(ValueError):
            self.store.create("doppelt")
        self.assertEqual(len(self.store.list()), 1)

    def test_to_dict_keys_and_counts(self):
        p = self.store.create("P", "Beschreibung", tags=["a"])
        self.store.add_note("P", "aufgabe", "offen 1")
        t2 = self.store.add_note("P", "aufgabe", "offen 2")
        self.store.add_note("P", "notiz", "eine Notiz")
        self.store.complete(t2.id)
        d = self.store.get("P").to_dict()
        self.assertEqual(set(d), {"id", "name", "beschreibung", "status", "tags", "erstellt", "geaendert",
                                  "offene_aufgaben", "notizen"})
        self.assertEqual(d["id"], p.id)
        self.assertEqual(d["beschreibung"], "Beschreibung")
        self.assertEqual(d["tags"], ["a"])
        self.assertEqual(d["offene_aufgaben"], 1)
        self.assertEqual(d["notizen"], 3)
        self.assertRegex(d["erstellt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        self.assertRegex(d["geaendert"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        self.assertIn("P", p.short())

    def test_ensure_is_idempotent(self):
        a = self.store.ensure("Neu")
        b = self.store.ensure("neu")
        c = self.store.ensure("  NEU ")
        self.assertEqual(a.id, b.id)
        self.assertEqual(a.id, c.id)
        self.assertEqual(len(self.store.list()), 1)
        with self.assertRaises(ValueError):
            self.store.ensure(" ")

    def test_list_and_archive(self):
        self.store.create("A")
        time.sleep(0.01)
        self.store.create("B")
        time.sleep(0.01)
        self.store.create("C")
        self.assertEqual([p.name for p in self.store.list()], ["C", "B", "A"])
        arch = self.store.archive("B")
        self.assertEqual(arch.status, "archiviert")
        self.assertEqual({p.name for p in self.store.list()}, {"A", "C"})
        self.assertEqual({p.name for p in self.store.list(include_archived=True)}, {"A", "B", "C"})
        # idempotent
        self.assertEqual(self.store.archive("B").status, "archiviert")
        # "fertig" bleibt sichtbar
        self.store.update("C", status="fertig")
        self.assertEqual({p.name for p in self.store.list()}, {"A", "C"})
        with self.assertRaises(ValueError):
            self.store.archive("unbekannt")

    def test_update_fields_and_validation(self):
        self.store.create("P", "alt", tags=["x"])
        p = self.store.update("P", description="neu")
        self.assertEqual((p.description, p.status, p.tags), ("neu", "aktiv", ["x"]))
        p = self.store.update("P", status="pausiert", tags=["b", "A"])
        self.assertEqual((p.description, p.status, p.tags), ("neu", "pausiert", ["a", "b"]))
        p = self.store.update("P", tags=[])
        self.assertEqual(p.tags, [])
        p = self.store.update("P")          # nichts zu ändern -> unverändert, kein Fehler
        self.assertEqual(p.status, "pausiert")
        with self.assertRaises(ValueError):
            self.store.update("P", status="kaputt")
        with self.assertRaises(ValueError):
            self.store.update("fehlt", description="x")
        self.assertEqual(STATUSES, ("aktiv", "pausiert", "archiviert", "fertig"))

    def test_update_touches_updated_at(self):
        p = self.store.create("P")
        time.sleep(0.01)
        p2 = self.store.update("P", description="x")
        self.assertGreater(p2.updated_at, p.updated_at)
        self.assertEqual(p2.created_at, p.created_at)

    def test_rename_and_conflicts(self):
        a = self.store.create("Alt")
        self.store.create("Belegt")
        time.sleep(0.01)
        r = self.store.rename("alt", "Neu")
        self.assertEqual((r.id, r.name), (a.id, "Neu"))
        self.assertGreater(r.updated_at, a.updated_at)
        self.assertIsNone(self.store.get("Alt"))
        self.assertEqual(self.store.get("neu").id, a.id)
        with self.assertRaises(ValueError):
            self.store.rename("Neu", "belegt")
        with self.assertRaises(ValueError):
            self.store.rename("Neu", "   ")
        with self.assertRaises(ValueError):
            self.store.rename("gibt es nicht", "Egal")
        # Nur Schreibweise ändern ist erlaubt
        self.assertEqual(self.store.rename("Neu", "NEU").name, "NEU")
        self.assertEqual(len(self.store.list()), 2)

    def test_delete_removes_notes_and_files(self):
        self.store.create("P")
        self.store.create("Q")
        n = self.store.add_note("P", "notiz", "weg damit", "Inhalt zum Finden")
        self.store.add_note("Q", "notiz", "bleibt", "anderes Finden")
        self.store.add_file("P", "/tmp/datei.txt")
        self.assertTrue(self.store.delete("p"))
        self.assertFalse(self.store.delete("P"))
        self.assertFalse(self.store.delete(""))
        self.assertIsNone(self.store.get("P"))
        self.assertIsNone(self.store.get_note(n.id))
        self.assertEqual(self.store.notes("P"), [])
        self.assertEqual(self.store.files("P"), [])
        hits = self.store.search("Finden")
        self.assertEqual([h.title for h in hits], ["bleibt"])
        st = self.store.stats()
        self.assertEqual((st["projekte"], st["notizen"], st["dateien"]), (1, 1, 0))


class NotesTest(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(":memory:")
        self.store.create("P", "Testprojekt")

    def tearDown(self):
        self.store.close()

    def test_add_note_all_kinds(self):
        self.assertEqual(NOTE_KINDS, ("notiz", "entscheidung", "aufgabe", "version", "ergebnis", "problem"))
        self.assertIs(KINDS, NOTE_KINDS)
        for kind in NOTE_KINDS:
            n = self.store.add_note("P", kind, f"Titel {kind}", f"Inhalt {kind}")
            self.assertIsInstance(n, Note)
            self.assertEqual(n.kind, kind)
            self.assertEqual(n.title, f"Titel {kind}")
            self.assertEqual(n.content, f"Inhalt {kind}")
            self.assertFalse(n.done)
            self.assertGreater(n.created_at, 0)
        self.assertEqual(len(self.store.notes("P")), len(NOTE_KINDS))
        self.assertEqual(self.store.get("P").note_count, len(NOTE_KINDS))

    def test_add_note_validation(self):
        with self.assertRaises(ValueError):
            self.store.add_note("P", "idee", "Titel")
        with self.assertRaises(ValueError):
            self.store.add_note("P", "notiz", "   ")
        with self.assertRaises(ValueError):
            self.store.add_note("P", "notiz", "")
        with self.assertRaises(ValueError):
            self.store.add_note("unbekannt", "notiz", "Titel")
        self.assertEqual(self.store.notes("P"), [])

    def test_add_note_is_tolerant(self):
        n = self.store.add_note("p", " Aufgabe ", "  Motor   bestellen ", None)
        self.assertEqual((n.kind, n.title, n.content), ("aufgabe", "Motor bestellen", ""))
        long_title = "x" * 500
        n2 = self.store.add_note("P", "notiz", long_title)
        self.assertLessEqual(len(n2.title), 200)
        self.assertTrue(n2.title.endswith("[gekürzt]"))

    def test_note_to_dict_and_short(self):
        n = self.store.add_note("P", "aufgabe", "Rahmen fräsen", "CFK 3 mm")
        d = n.to_dict()
        self.assertEqual(set(d), {"id", "projekt_id", "art", "titel", "inhalt", "erledigt", "erstellt"})
        self.assertEqual(d["projekt_id"], self.store.get("P").id)
        self.assertEqual((d["art"], d["titel"], d["inhalt"], d["erledigt"]), ("aufgabe", "Rahmen fräsen", "CFK 3 mm", False))
        self.assertRegex(d["erstellt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
        self.assertIn("☐", n.short())
        self.assertIn("☑", self.store.complete(n.id).short())

    def test_notes_filters_and_order(self):
        a = self.store.add_note("P", "aufgabe", "A")
        time.sleep(0.005)
        b = self.store.add_note("P", "aufgabe", "B")
        time.sleep(0.005)
        c = self.store.add_note("P", "entscheidung", "C")
        self.store.complete(b.id)
        self.assertEqual([n.id for n in self.store.notes("P")], [c.id, b.id, a.id])
        self.assertEqual([n.id for n in self.store.notes("P", kind="aufgabe")], [b.id, a.id])
        self.assertEqual([n.id for n in self.store.notes("P", kind="aufgabe", include_done=False)], [a.id])
        self.assertEqual([n.id for n in self.store.notes("P", include_done=False)], [c.id, a.id])
        self.assertEqual([n.id for n in self.store.notes("P", limit=1)], [c.id])
        self.assertEqual(self.store.notes("P", limit=0), [])
        self.assertEqual(self.store.notes("gibt es nicht"), [])
        with self.assertRaises(ValueError):
            self.store.notes("P", kind="unsinn")

    def test_complete_and_reopen(self):
        t = self.store.add_note("P", "aufgabe", "Löten")
        self.assertEqual(self.store.get("P").open_tasks, 1)
        done = self.store.complete(t.id)
        self.assertTrue(done.done)
        self.assertEqual(self.store.get("P").open_tasks, 0)
        again = self.store.complete(t.id, done=False)
        self.assertFalse(again.done)
        self.assertEqual(self.store.get("P").open_tasks, 1)
        with self.assertRaises(ValueError):
            self.store.complete(12345)

    def test_delete_note(self):
        n = self.store.add_note("P", "notiz", "weg", "Suchwort Zirkonium")
        self.assertEqual(len(self.store.search("Zirkonium")), 1)
        self.assertTrue(self.store.delete_note(n.id))
        self.assertFalse(self.store.delete_note(n.id))
        self.assertIsNone(self.store.get_note(n.id))
        self.assertEqual(self.store.search("Zirkonium"), [])
        self.assertIsNone(self.store.get_note("abc"))

    def test_note_changes_touch_project(self):
        p0 = self.store.get("P")
        time.sleep(0.01)
        n = self.store.add_note("P", "aufgabe", "T")
        p1 = self.store.get("P")
        self.assertGreater(p1.updated_at, p0.updated_at)
        time.sleep(0.01)
        self.store.complete(n.id)
        p2 = self.store.get("P")
        self.assertGreater(p2.updated_at, p1.updated_at)
        time.sleep(0.01)
        self.store.delete_note(n.id)
        p3 = self.store.get("P")
        self.assertGreater(p3.updated_at, p2.updated_at)


class FilesTest(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(":memory:")
        self.store.create("P")

    def tearDown(self):
        self.store.close()

    def test_add_and_list_files(self):
        with tempfile.TemporaryDirectory() as d:
            real_file = os.path.join(d, "rahmen.step")
            with open(real_file, "w", encoding="utf-8") as fh:
                fh.write("x")
            given = os.path.join(d, ".", "rahmen.step")
            f = self.store.add_file("P", given, "CAD des Rahmens")
            self.assertEqual(set(f), {"id", "projekt_id", "pfad", "realpfad", "beschreibung", "hinzugefuegt",
                                      "existiert"})
            self.assertEqual(f["pfad"], given)
            self.assertEqual(f["realpfad"], os.path.realpath(real_file))
            self.assertEqual(f["beschreibung"], "CAD des Rahmens")
            self.assertTrue(f["existiert"])
            missing = self.store.add_file("P", "noch/nicht/da.txt")
            self.assertFalse(missing["existiert"])
            self.assertEqual(missing["pfad"], "noch/nicht/da.txt")
            self.assertTrue(os.path.isabs(missing["realpfad"]))
            files = self.store.files("P")
            self.assertEqual([x["id"] for x in files], [f["id"], missing["id"]])

    def test_same_file_once_per_project(self):
        a = self.store.add_file("P", "/tmp/a.txt", "erste")
        b = self.store.add_file("P", "/tmp//a.txt", "zweite Beschreibung")
        self.assertEqual(a["id"], b["id"])
        self.assertEqual(b["beschreibung"], "zweite Beschreibung")
        self.assertEqual(len(self.store.files("P")), 1)
        self.store.create("Q")
        c = self.store.add_file("Q", "/tmp/a.txt")
        self.assertNotEqual(a["id"], c["id"])

    def test_file_validation_and_remove(self):
        with self.assertRaises(ValueError):
            self.store.add_file("P", "   ")
        with self.assertRaises(ValueError):
            self.store.add_file("fehlt", "/tmp/x")
        self.assertEqual(self.store.files("fehlt"), [])
        f = self.store.add_file("P", "/tmp/x.txt")
        self.assertTrue(self.store.remove_file(f["id"]))
        self.assertFalse(self.store.remove_file(f["id"]))
        self.assertEqual(self.store.files("P"), [])

    def test_add_file_touches_project(self):
        p0 = self.store.get("P")
        time.sleep(0.01)
        self.store.add_file("P", "/tmp/y.txt")
        self.assertGreater(self.store.get("P").updated_at, p0.updated_at)


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_minimal_summary(self):
        self.store.create("Leer")
        self.assertEqual(self.store.summary("Leer"), "Projekt »Leer« (aktiv)")
        self.store.update("Leer", description="Nur Text", status="pausiert")
        self.assertEqual(self.store.summary("leer"), "Projekt »Leer« (pausiert): Nur Text")
        self.assertEqual(self.store.summary("gibt es nicht"), "")
        self.assertEqual(self.store.summary("Leer", max_chars=0), "")

    def test_summary_sections(self):
        self.store.create("Drohne", "5-Zoll-Quad für FPV")
        for i in range(7):
            self.store.add_note("Drohne", "aufgabe", f"Aufgabe {i}")
            time.sleep(0.002)
        done = self.store.add_note("Drohne", "aufgabe", "Erledigte Aufgabe")
        self.store.complete(done.id)
        for i in range(4):
            self.store.add_note("Drohne", "entscheidung", f"Entscheidung {i}", f"Begründung {i}")
            time.sleep(0.002)
        self.store.add_note("Drohne", "version", "v0.1", "erster Prototyp")
        time.sleep(0.002)
        self.store.add_note("Drohne", "version", "v0.2", "Rahmen aus CFK")
        self.store.add_note("Drohne", "problem", "Motor 3 wird heiß", "nach 2 min Flug 70 °C")
        self.store.add_note("Drohne", "notiz", "Nur eine Notiz, taucht nicht auf")
        self.store.add_note("Drohne", "ergebnis", "Missionsbericht, taucht nicht auf")

        s = self.store.summary("Drohne", max_chars=2000)
        self.assertTrue(s.startswith("Projekt »Drohne« (aktiv): 5-Zoll-Quad für FPV | "))
        self.assertIn(LABEL_OPEN_TASKS, s)
        self.assertIn(LABEL_DECISIONS, s)
        self.assertIn(LABEL_VERSION, s)
        self.assertIn(LABEL_PROBLEM, s)
        # Reihenfolge der Abschnitte
        self.assertLess(s.index(LABEL_OPEN_TASKS), s.index(LABEL_DECISIONS))
        self.assertLess(s.index(LABEL_DECISIONS), s.index(LABEL_VERSION))
        self.assertLess(s.index(LABEL_VERSION), s.index(LABEL_PROBLEM))
        # max. 5 offene Aufgaben (die neuesten), Hinweis auf weitere, erledigte nicht
        tasks_part = s.split(LABEL_OPEN_TASKS)[1].split(" | ")[0]
        for i in (6, 5, 4, 3, 2):
            self.assertIn(f"Aufgabe {i}", tasks_part)
        self.assertNotIn("Aufgabe 1", tasks_part)
        self.assertNotIn("Aufgabe 0", tasks_part)
        self.assertNotIn("Erledigte", s)
        self.assertIn("(+2 weitere)", tasks_part)
        # 3 neueste Entscheidungen
        dec_part = s.split(LABEL_DECISIONS)[1].split(" | ")[0]
        self.assertIn("Entscheidung 3", dec_part)
        self.assertIn("Entscheidung 1", dec_part)
        self.assertNotIn("Entscheidung 0", dec_part)
        # neueste Version mit Inhalt, neuestes Problem
        self.assertIn("v0.2 – Rahmen aus CFK", s)
        self.assertNotIn("v0.1", s)
        self.assertIn("Motor 3 wird heiß – nach 2 min Flug 70 °C", s)
        self.assertNotIn("Nur eine Notiz", s)
        self.assertNotIn("Missionsbericht", s)

    def test_only_sections_with_content(self):
        self.store.create("P")
        self.store.add_note("P", "entscheidung", "CFK statt Alu")
        t = self.store.add_note("P", "aufgabe", "fertig gemacht")
        self.store.complete(t.id)
        s = self.store.summary("P")
        self.assertEqual(s, f"Projekt »P« (aktiv) | {LABEL_DECISIONS}CFK statt Alu")
        self.assertNotIn(LABEL_OPEN_TASKS, s)
        self.assertNotIn(LABEL_VERSION, s)
        self.assertNotIn(LABEL_PROBLEM, s)

    def test_summary_budget(self):
        self.store.create("Budget", "B" * 400)
        for i in range(5):
            self.store.add_note("Budget", "aufgabe", f"Aufgabe {i} " + "lang " * 20)
            self.store.add_note("Budget", "entscheidung", f"Entscheidung {i} " + "lang " * 20)
        self.store.add_note("Budget", "version", "v9", "v" * 300)
        self.store.add_note("Budget", "problem", "Problem", "p" * 300)
        full = self.store.summary("Budget", max_chars=5000)
        self.assertGreater(len(full), 800)
        for budget in (800, 300, 120, 60, 30, 10):
            s = self.store.summary("Budget", max_chars=budget)
            self.assertLessEqual(len(s), budget, budget)
            self.assertTrue(s.startswith("Projekt »Budget«"[:budget]), (budget, s))
        s800 = self.store.summary("Budget")
        self.assertLessEqual(len(s800), 800)
        self.assertIn("[gekürzt]", s800)
        self.assertIn(LABEL_OPEN_TASKS, s800)
        # Beschreibung wird für die Zusammenfassung begrenzt
        self.assertNotIn("B" * 301, full)

    def test_summary_description_decision_identical_to_title(self):
        # Brain legt Entscheidungen mit titel = inhalt[:80], inhalt = inhalt an -> kein Doppel
        self.store.create("P")
        text = "Wir nehmen 6S statt 4S wegen der höheren Effizienz der Motoren"
        self.store.add_note("P", "entscheidung", text[:80], text)
        self.store.add_note("P", "version", "Wir nehmen 6S", text)
        s = self.store.summary("P")
        self.assertIn(LABEL_DECISIONS + text + " | ", s)       # Titel reicht, kein "Titel – Inhalt"
        self.assertIn(LABEL_VERSION + text, s)                 # Titel = Anfang des Inhalts -> Inhalt
        self.assertNotIn(" – ", s)
        self.assertEqual(s.count(text), 2)


class SearchTest(unittest.TestCase):
    def setUp(self):
        self.store = ProjectStore(":memory:")
        self.store.create("A")
        self.store.create("B")

    def tearDown(self):
        self.store.close()

    def test_search_title_and_content(self):
        n1 = self.store.add_note("A", "entscheidung", "Rahmen aus Carbon", "steifer als Aluminium")
        n2 = self.store.add_note("B", "notiz", "Einkaufsliste", "Carbonplatte 3 mm bestellen")
        self.store.add_note("B", "notiz", "Lieblingsfarbe", "blau")
        hits = self.store.search("Carbon")
        self.assertEqual({h.id for h in hits}, {n1.id, n2.id})
        self.assertTrue(all(0 < h.score <= 1.0 for h in hits))
        self.assertEqual(hits[0].score, 1.0)
        # Treffer im Titel wiegt mehr
        self.assertEqual(hits[0].id, n1.id)
        self.assertEqual([h.id for h in self.store.search("Aluminium")], [n1.id])
        self.assertEqual(len(self.store.search("Carbon", k=1)), 1)
        self.assertEqual(self.store.search("Carbon", k=0), [])

    def test_search_umlaut_prefix_and_empty(self):
        n = self.store.add_note("A", "problem", "Lötstation überhitzt", "Temperaturregler defekt")
        self.assertEqual([h.id for h in self.store.search("LÖTSTATION")], [n.id])
        self.assertEqual([h.id for h in self.store.search("Temperatur")], [n.id])
        self.assertEqual(self.store.search(""), [])
        self.assertEqual(self.store.search("   "), [])
        self.assertEqual(self.store.search("Quantenphysik"), [])
        self.assertEqual(self.store.search('"kaputt'), [])

    def test_search_reflects_updates(self):
        n = self.store.add_note("A", "aufgabe", "Propeller kaufen")
        self.assertEqual(len(self.store.search("Propeller")), 1)
        self.store.complete(n.id)
        self.assertTrue(self.store.search("Propeller")[0].done)
        self.store.delete("A")
        self.assertEqual(self.store.search("Propeller"), [])


class StatsAndPersistenceTest(unittest.TestCase):
    def test_stats(self):
        store = ProjectStore(":memory:")
        self.assertEqual(store.stats()["projekte"], 0)
        store.create("A")
        store.create("B")
        store.create("C")
        store.archive("C")
        store.update("B", status="fertig")
        t1 = store.add_note("A", "aufgabe", "offen")
        t2 = store.add_note("A", "aufgabe", "zu")
        store.complete(t2.id)
        store.add_note("B", "entscheidung", "E")
        store.add_file("A", "/tmp/f.txt")
        st = store.stats()
        self.assertEqual(st["projekte"], 3)
        self.assertEqual(st["aktiv"], 1)
        self.assertEqual(st["nach_status"], {"aktiv": 1, "pausiert": 0, "archiviert": 1, "fertig": 1})
        self.assertEqual(st["notizen"], 3)
        self.assertEqual(st["nach_art"]["aufgabe"], 2)
        self.assertEqual(st["nach_art"]["entscheidung"], 1)
        self.assertEqual(st["offene_aufgaben"], 1)
        self.assertEqual(st["dateien"], 1)
        self.assertIsNotNone(t1)
        store.close()

    def test_persistence_across_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "unter", "projekte.db")
            s1 = ProjectStore(path)
            p = s1.create("Persistent", "bleibt", tags=["t"])
            n = s1.add_note("Persistent", "aufgabe", "Aufgabe bleibt", "Inhalt")
            s1.add_note("Persistent", "entscheidung", "Entscheidung bleibt")
            s1.add_file("Persistent", "/tmp/datei.txt", "Beschreibung")
            s1.close()
            s1.close()      # idempotent

            s2 = ProjectStore(path)
            p2 = s2.get("persistent")
            self.assertIsNotNone(p2)
            self.assertEqual((p2.id, p2.name, p2.description, p2.tags), (p.id, "Persistent", "bleibt", ["t"]))
            self.assertEqual(p2.created_at, p.created_at)
            self.assertEqual(p2.open_tasks, 1)
            self.assertEqual(p2.note_count, 2)
            notes = s2.notes("Persistent")
            self.assertEqual({x.title for x in notes}, {"Aufgabe bleibt", "Entscheidung bleibt"})
            self.assertEqual(s2.get_note(n.id).content, "Inhalt")
            self.assertEqual([h.id for h in s2.search("Aufgabe")], [n.id])
            files = s2.files("Persistent")
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0]["beschreibung"], "Beschreibung")
            self.assertIn(LABEL_OPEN_TASKS + "Aufgabe bleibt", s2.summary("Persistent"))
            s2.close()

    def test_thread_safety_smoke(self):
        import threading
        store = ProjectStore(":memory:")
        store.create("T")
        errors: list[Exception] = []

        def work(i: int) -> None:
            try:
                for j in range(20):
                    store.add_note("T", "notiz", f"Notiz {i}-{j}", "parallel")
                    store.summary("T")
                    store.search("parallel", k=3)
            except Exception as exc:  # pragma: no cover - nur bei Fehler
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(store.get("T").note_count, 80)
        store.close()


if __name__ == "__main__":
    unittest.main()
