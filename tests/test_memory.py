import os
import tempfile
import time
import unittest

from obito.llm import FakeBackend
from obito.memory import MemoryStore, cosine, normalize


class MemoryBasicsTest(unittest.TestCase):
    def setUp(self):
        self.store = MemoryStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_remember_and_get(self):
        m = self.store.remember("Nutzer bevorzugt Carbon für Drohnenrahmen", kind="praeferenz",
                                tags=["Material", "drohne"], project="DJI Mod", importance=0.8)
        self.assertEqual(m.kind, "praeferenz")
        self.assertEqual(m.tags, ["drohne", "material"])
        self.assertEqual(m.project, "DJI Mod")
        self.assertEqual(self.store.get(m.id).content, m.content)
        self.assertEqual(self.store.count(), 1)

    def test_empty_content_rejected(self):
        with self.assertRaises(ValueError):
            self.store.remember("   ")

    def test_unknown_kind_becomes_note(self):
        m = self.store.remember("irgendwas", kind="unbekannt")
        self.assertEqual(m.kind, "notiz")

    def test_duplicates_merge(self):
        a = self.store.remember("Motor_J2 hat 12 Nm Drehmoment", importance=0.4)
        b = self.store.remember("motor_j2 hat 12 nm Drehmoment!", importance=0.6)
        self.assertEqual(a.id, b.id)
        self.assertEqual(self.store.count(), 1)
        self.assertGreaterEqual(b.importance, 0.6)
        self.assertEqual(b.access_count, 1)

    def test_same_text_different_project_is_separate(self):
        a = self.store.remember("Gewicht 249 g", project="A")
        b = self.store.remember("Gewicht 249 g", project="B")
        self.assertNotEqual(a.id, b.id)

    def test_forget(self):
        m = self.store.remember("vergiss mich")
        self.assertTrue(self.store.forget(m.id))
        self.assertFalse(self.store.forget(m.id))
        self.assertIsNone(self.store.get(m.id))
        self.assertEqual(self.store.search("vergiss"), [])

    def test_search_fulltext_and_prefix(self):
        self.store.remember("Drohnenrahmen aus Carbon ist steifer als Aluminium", kind="fakt")
        self.store.remember("Lieblingsfarbe ist blau", kind="praeferenz")
        hits = self.store.search("Drohne Material")
        self.assertEqual(len(hits), 1)
        self.assertIn("Carbon", hits[0].content)
        self.assertGreater(hits[0].score, 0)

    def test_search_umlaut_and_case(self):
        self.store.remember("Die Lötstation hat 60 W Leistung")
        self.assertEqual(len(self.store.search("LÖTSTATION")), 1)

    def test_search_project_filter(self):
        self.store.remember("Projekt A: Spannweite 1200 mm", project="A")
        self.store.remember("Projekt B: Spannweite 800 mm", project="B")
        self.store.remember("Allgemein: Spannweite messen vor dem Druck")
        hits = self.store.search("Spannweite", project="A")
        self.assertEqual({h.project for h in hits}, {"A", None})

    def test_search_touches_access(self):
        m = self.store.remember("Akku 4S 1500 mAh")
        before = m.last_access
        time.sleep(0.01)
        self.store.search("Akku")
        self.assertGreater(self.store.get(m.id).last_access, before)
        self.assertEqual(self.store.get(m.id).access_count, 1)
        self.store.search("Akku", touch=False)
        self.assertEqual(self.store.get(m.id).access_count, 1)

    def test_search_no_match(self):
        self.store.remember("Etwas über Motoren")
        self.assertEqual(self.store.search("Quantenphysik"), [])
        self.assertEqual(self.store.search(""), [])

    def test_importance_ranks_higher(self):
        low = self.store.remember("Schraube M3 für Deckel", importance=0.1)
        high = self.store.remember("Schraube M3 für Motorhalter", importance=1.0)
        hits = self.store.search("Schraube M3")
        self.assertEqual(hits[0].id, high.id)
        self.assertEqual(hits[1].id, low.id)

    def test_update_importance_clamped(self):
        m = self.store.remember("x" * 20, importance=0.9)
        self.store.update_importance(m.id, +0.5)
        self.assertEqual(self.store.get(m.id).importance, 1.0)
        self.store.update_importance(m.id, -5)
        self.assertEqual(self.store.get(m.id).importance, 0.0)

    def test_recent_and_stats(self):
        self.store.remember("eins", project="P")
        self.store.remember("zwei")
        self.store.remember("drei", kind="fakt", project="Q")
        self.assertEqual([m.content for m in self.store.recent(2)], ["drei", "zwei"])
        self.assertEqual({m.content for m in self.store.recent(10, project="P")}, {"eins", "zwei"})
        st = self.store.stats()
        self.assertEqual(st["erinnerungen"], 3)
        self.assertEqual(st["nach_art"]["fakt"], 1)
        self.assertEqual(st["projekte"], ["P", "Q"])
        self.assertEqual(st["mit_vektor"], 0)

    def test_history(self):
        for i in range(5):
            self.store.add_message("s1", "user" if i % 2 == 0 else "assistant", f"m{i}")
        self.store.add_message("s2", "user", "andere Sitzung")
        h = self.store.history("s1", limit=3)
        self.assertEqual([x["content"] for x in h], ["m2", "m3", "m4"])
        self.assertEqual(h[0]["role"], "user")
        self.assertEqual(self.store.history("leer"), [])
        self.assertEqual(self.store.stats()["nachrichten"], 6)

    def test_short(self):
        m = self.store.remember("Kurztext", kind="fakt", project="P")
        s = m.short()
        self.assertIn("#", s)
        self.assertIn("fakt", s)
        self.assertIn("P", s)
        self.assertIn("Kurztext", s)


class MemoryPersistenceTest(unittest.TestCase):
    def test_persists_across_instances_and_export_import(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "mem.db")
            s1 = MemoryStore(path)
            s1.remember("bleibt erhalten", kind="fakt", tags=["t1"])
            s1.close()
            s2 = MemoryStore(path)
            self.assertEqual(s2.count(), 1)
            export = os.path.join(d, "export.json")
            self.assertEqual(s2.export(export), 1)
            s2.close()
            s3 = MemoryStore(":memory:")
            self.assertEqual(s3.import_json(export), 1)
            self.assertEqual(s3.count(), 1)
            self.assertEqual(s3.recent(1)[0].tags, ["t1"])
            s3.close()


class MemoryVectorTest(unittest.TestCase):
    def setUp(self):
        self.backend = FakeBackend(embed_dim=256)   # große Dimension = kaum Hash-Kollisionen
        self.store = MemoryStore(":memory:", embedder=self.backend.embed)

    def tearDown(self):
        self.store.close()

    def test_embeddings_stored(self):
        self.store.remember("Vektor-Erinnerung")
        self.assertEqual(self.store.stats()["mit_vektor"], 1)

    def test_vector_duplicate_merge(self):
        a = self.store.remember("Drohne Carbon Rahmen leicht")
        b = self.store.remember("Drohne Carbon Rahmen leicht.")   # gleiche Wörter -> gleicher Vektor
        self.assertEqual(a.id, b.id)

    def test_vector_search_finds_overlapping_words(self):
        self.store.remember("Carbon Rahmen für die Drohne")
        self.store.remember("Rezept für Apfelkuchen")
        hits = self.store.search("Drohne Rahmen")
        self.assertGreaterEqual(len(hits), 1)
        self.assertIn("Carbon", hits[0].content)
        self.assertTrue(all(h.score < hits[0].score for h in hits[1:]))

    def test_min_importance_filters_recall(self):
        m = self.store.remember("Carbon Rahmen Drohne", importance=0.1)
        self.assertEqual(self.store.search("Carbon Drohne", min_importance=0.15), [])
        self.assertEqual([x.id for x in self.store.search("Carbon Drohne")], [m.id])

    def test_set_embedder_later_and_reindex(self):
        store = MemoryStore(":memory:")
        store.remember("Drohne Carbon Rahmen")
        store.remember("Apfelkuchen Rezept")
        self.assertEqual(store.stats()["ohne_vektor"], 2)
        store.set_embedder(FakeBackend(embed_dim=64).embed, "fake-64")
        self.assertEqual(store.reindex(), 2)
        st = store.stats()
        self.assertEqual((st["mit_vektor"], st["ohne_vektor"], st["embedding_modell"], st["embedding_dim"]),
                         (2, 0, "fake-64", 64))
        self.assertEqual(store.reindex(), 0)
        store.close()

    def test_model_switch_ignores_old_vectors_until_reindex(self):
        store = MemoryStore(":memory:", embedder=FakeBackend(embed_dim=32).embed, embed_model="fake-32")
        a = store.remember("Drohne Carbon Rahmen")
        store.set_embedder(FakeBackend(embed_dim=64).embed, "fake-64")
        b = store.remember("Neuer Eintrag mit anderem Modell")
        st = store.stats()
        self.assertEqual((st["mit_vektor"], st["ohne_vektor"]), (1, 1))
        # alte Erinnerung bleibt über Volltext auffindbar, nicht abgewertet
        hits = store.search("Carbon Drohne")
        self.assertEqual([h.id for h in hits], [a.id])
        self.assertEqual(store.reindex(only_missing=True), 0)   # alter Vektor ist vorhanden, nur falsch
        self.assertEqual(store.reindex(), 1)
        self.assertEqual(store.stats()["ohne_vektor"], 0)
        self.assertIsNotNone(store.get(b.id).embedding)
        store.close()

    def test_persisted_embed_model_survives_reopen(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "m.db")
            s = MemoryStore(p, embedder=FakeBackend(embed_dim=8).embed, embed_model="fake-8")
            s.remember("etwas")
            s.close()
            s2 = MemoryStore(p)
            st = s2.stats()
            self.assertEqual((st["embedding_dim"], st["mit_vektor"]), (8, 1))
            s2.close()
            s2.close()   # idempotent

    def test_embedder_failure_is_tolerated(self):
        def broken(texts):
            raise RuntimeError("kaputt")
        store = MemoryStore(":memory:", embedder=broken)
        m = store.remember("ohne Vektor")
        self.assertIsNone(m.embedding)
        self.assertEqual(len(store.search("Vektor")), 1)
        store.close()


class HelpersTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("  Hallo,  WELT! "), "hallo welt")

    def test_cosine(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine([], [1]), 0.0)
        self.assertEqual(cosine([1, 2], [1]), 0.0)


if __name__ == "__main__":
    unittest.main()
