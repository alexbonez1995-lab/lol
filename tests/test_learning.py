import json
import os
import tempfile
import unittest

from obito.llm import FakeBackend
from obito.learning import (FALLBACK_SYSTEM_PROMPT, Interaction, LearningStore, Lesson, keywords,
                            strip_tool_blocks)

LONG = "Lade den 4S-Akku mit 1C (1,5 A) über den Balancer-Anschluss und beende bei 4,2 V je Zelle."


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class SchemaRoundtripTest(unittest.TestCase):
    def setUp(self):
        self.store = LearningStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_record_and_get_all_fields(self):
        iid = self.store.record(
            "s1", "Wie lade ich den Akku?", LONG, project="Drohne", experts=["ALPHA", "BETA"],
            depth="tief", trace='[{"stufe": "routing"}]', model="qwen2.5:7b", context="Projekt: Drohne",
            history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hallo"},
                     {"role": "system", "content": "weg"}, "kaputt"],
            memories_used=[1, "2", "x"], new_memories=[3], lessons_used=[7], tools_used=(),
            tokens=123, duration=1.5,
        )
        self.assertEqual(iid, 1)
        it = self.store.get(iid)
        self.assertIsInstance(it, Interaction)
        self.assertEqual(it.session_id, "s1")
        self.assertEqual(it.question, "Wie lade ich den Akku?")
        self.assertEqual(it.answer, LONG)
        self.assertEqual(it.project, "Drohne")
        self.assertEqual(it.experts, ["ALPHA", "BETA"])
        self.assertEqual(it.depth, "tief")
        self.assertEqual(it.model, "qwen2.5:7b")
        self.assertEqual(it.context, "Projekt: Drohne")
        self.assertEqual(it.history, [{"role": "user", "content": "hi"},
                                      {"role": "assistant", "content": "hallo"}])
        self.assertEqual(it.memories_used, [1, 2])
        self.assertEqual(it.new_memories, [3])
        self.assertEqual(it.lessons_used, [7])
        self.assertEqual(it.tools_used, [])
        self.assertEqual((it.tokens, it.duration), (123, 1.5))
        self.assertEqual((it.rating, it.comment, it.correction, it.correction_full), (0, None, None, None))
        self.assertTrue(it.trainable)
        self.assertGreater(it.created_at, 0)
        self.assertEqual(it.trace, '[{"stufe": "routing"}]')

    def test_to_dict_keys(self):
        iid = self.store.record("s", "Frage?", "Antwort", depth="schnell", tools_used=["rechnen"])
        d = self.store.get(iid).to_dict()
        self.assertEqual(set(d), {"id", "sitzung", "frage", "antwort", "projekt", "experten", "tiefe", "modell",
                                  "bewertung", "kommentar", "korrektur", "korrektur_voll", "trainierbar",
                                  "erstellt", "werkzeuge", "tokens", "dauer"})
        self.assertEqual(d["werkzeuge"], ["rechnen"])
        self.assertFalse(d["trainierbar"])
        self.assertRegex(d["erstellt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")

    def test_trainable_rule(self):
        ok = self.store.record("s", "a?", "b", depth="schnell")
        tool = self.store.record("s", "a?", "b", depth="schnell", tools_used=["zeit"])
        err = self.store.record("s", "a?", "b", depth="fehler")
        self.assertTrue(self.store.get(ok).trainable)
        self.assertFalse(self.store.get(tool).trainable)
        self.assertFalse(self.store.get(err).trainable)

    def test_empty_question_rejected(self):
        with self.assertRaises(ValueError):
            self.store.record("s", "   ", "antwort")

    def test_get_unknown_and_last(self):
        self.assertIsNone(self.store.get(42))
        self.assertEqual(self.store.last(), [])
        a = self.store.record("s1", "eins?", "1")
        b = self.store.record("s2", "zwei?", "2")
        c = self.store.record("s1", "drei?", "3")
        self.assertEqual([i.id for i in self.store.last()], [c])
        self.assertEqual([i.id for i in self.store.last(n=5)], [c, b, a])
        self.assertEqual([i.id for i in self.store.last("s1", n=5)], [c, a])
        self.assertEqual(self.store.last("leer"), [])
        self.assertEqual(self.store.last(n=0), [])

    def test_persists_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "sub", "lernen.db")
            s1 = LearningStore(path)
            iid = s1.record("s", "bleibt?", LONG, depth="schnell")
            s1.add_lesson("Regel bleibt", scope="allgemein")
            s1.rate(iid, 1, "gut")
            s1.close()
            s1.close()   # idempotent
            s2 = LearningStore(path)
            it = s2.get(iid)
            self.assertEqual((it.rating, it.comment), (1, "gut"))
            self.assertEqual([l.rule for l in s2.list_lessons()], ["Regel bleibt"])
            self.assertEqual(s2.stats()["interaktionen"], 1)
            s2.close()

    def test_stats(self):
        a = self.store.record("s", "a?", LONG)
        b = self.store.record("s", "b?", LONG)
        c = self.store.record("s", "c?", LONG, tools_used=["zeit"])
        self.store.record("s", "d?", LONG)
        self.store.rate(a, 1)
        self.store.rate(b, -1)
        self.store.correct(c, "besser")
        self.store.add_lesson("eins", scope="allgemein")
        self.store.add_lesson("zwei")
        st = self.store.stats()
        self.assertEqual(st, {"interaktionen": 4, "bewertet": 3, "positiv": 1, "negativ": 2, "korrigiert": 1,
                              "trainierbar": 3, "lektionen": 2, "lektionen_aktiv": 2})


class RateCorrectTest(unittest.TestCase):
    def setUp(self):
        self.store = LearningStore(":memory:")
        self.iid = self.store.record("s", "Frage?", LONG, depth="schnell")

    def tearDown(self):
        self.store.close()

    def test_rate_sets_rating_and_comment(self):
        it = self.store.rate(self.iid, 1, "prima")
        self.assertEqual((it.rating, it.comment), (1, "prima"))
        it = self.store.rate(self.iid, -1)
        self.assertEqual((it.rating, it.comment), (-1, "prima"))   # Kommentar bleibt ohne neuen
        it = self.store.rate(self.iid, 0, "neutral")
        self.assertEqual((it.rating, it.comment), (0, "neutral"))

    def test_rate_validation(self):
        for bad in (2, -2, "x", None):
            with self.assertRaises(ValueError):
                self.store.rate(self.iid, bad)
        with self.assertRaises(ValueError):
            self.store.rate(999, 1)
        self.assertEqual(self.store.rate(self.iid, "1").rating, 1)   # Zahl als String ist ok

    def test_correct_sets_minus_one_only_if_unrated(self):
        it = self.store.correct(self.iid, "Besser so", "Vollständige bessere Antwort.")
        self.assertEqual((it.rating, it.correction, it.correction_full),
                         (-1, "Besser so", "Vollständige bessere Antwort."))
        other = self.store.record("s", "Frage 2?", LONG)
        self.store.rate(other, 1)
        it = self.store.correct(other, "Kleinigkeit")
        self.assertEqual((it.rating, it.correction, it.correction_full), (1, "Kleinigkeit", None))

    def test_correct_validation(self):
        with self.assertRaises(ValueError):
            self.store.correct(self.iid, "   ")
        with self.assertRaises(ValueError):
            self.store.correct(999, "x")

    def test_best_answer(self):
        it = self.store.get(self.iid)
        self.assertEqual(it.best_answer(), LONG)
        it = self.store.correct(self.iid, "kurz")
        self.assertEqual(it.best_answer(), "kurz")
        it = self.store.correct(self.iid, "kurz", "voll")
        self.assertEqual(it.best_answer(), "voll")


class LessonTest(unittest.TestCase):
    def setUp(self):
        self.store = LearningStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_add_lesson_and_dict(self):
        l = self.store.add_lesson("  Zahlen immer   mit Einheiten  ", scope="allgemein",
                                  topics=["Physik", "physik", " Einheiten "], source_interaction="3")
        self.assertIsInstance(l, Lesson)
        self.assertEqual(l.rule, "Zahlen immer mit Einheiten")
        self.assertEqual(l.scope, "allgemein")
        self.assertEqual(l.topics, ["physik", "einheiten"])
        self.assertEqual(l.source_interaction, 3)
        self.assertEqual((l.helped, l.hurt, l.active), (0, 0, True))
        d = l.to_dict()
        self.assertEqual(set(d), {"id", "regel", "bereich", "themen", "quelle", "erstellt", "geholfen",
                                  "geschadet", "aktiv"})
        self.assertEqual(d["regel"], "Zahlen immer mit Einheiten")

    def test_unknown_scope_becomes_thema_and_empty_rejected(self):
        self.assertEqual(self.store.add_lesson("x", scope="egal").scope, "thema")
        with self.assertRaises(ValueError):
            self.store.add_lesson("  ")

    def test_duplicate_returns_existing(self):
        a = self.store.add_lesson("Immer Einheiten angeben.", scope="allgemein")
        b = self.store.add_lesson("immer EINHEITEN angeben", scope="thema", topics=["x"])
        self.assertEqual(a.id, b.id)
        self.assertEqual(b.scope, "allgemein")
        self.assertEqual(len(self.store.list_lessons()), 1)

    def test_render(self):
        l = Lesson(1, "Regeltext", "thema", [], None, 0.0, 0, 0, True)
        self.assertEqual(l.render(), "Regel: Regeltext")
        l.helped = 4
        self.assertEqual(l.render(), "Regel (bestätigt 4×): Regeltext")
        l.hurt = 3
        l.helped = 2
        self.assertEqual(l.render(), "Regel (umstritten 2/3): Regeltext")
        l.rule = "lang " * 200
        self.assertLessEqual(len(l.render()), 300)
        self.assertTrue(l.render().endswith("…"))

    def test_list_and_delete(self):
        a = self.store.add_lesson("allgemeine Regel", scope="allgemein")
        b = self.store.add_lesson("Akku-Regel", topics=["akku"])
        self.assertEqual([l.id for l in self.store.list_lessons()], [a.id, b.id])
        self.assertTrue(self.store.delete_lesson(a.id))
        self.assertFalse(self.store.delete_lesson(a.id))
        self.assertEqual([l.id for l in self.store.list_lessons()], [b.id])
        self.assertEqual(self.store.lessons("Akku", k=3), [self.store.list_lessons()[0]])

    def test_helped_hurt_and_deactivation(self):
        l = self.store.add_lesson("Schlechte Regel", scope="allgemein")
        ids = []
        for i in range(5):
            ids.append(self.store.record("s", f"Frage {i}?", LONG, lessons_used=[l.id]))
        self.store.rate(ids[0], 1)
        self.store.rate(ids[1], -1)
        self.store.rate(ids[2], -1)
        cur = self.store.list_lessons(include_inactive=True)[0]
        self.assertEqual((cur.helped, cur.hurt, cur.active), (1, 2, True))
        self.store.rate(ids[3], -1)      # hurt 3 > 2*helped (2) -> deaktiviert
        cur = self.store.list_lessons(include_inactive=True)[0]
        self.assertEqual((cur.helped, cur.hurt, cur.active), (1, 3, False))
        self.assertEqual(self.store.list_lessons(), [])
        self.assertEqual(self.store.lessons("Frage"), [])
        self.assertEqual(self.store.stats()["lektionen_aktiv"], 0)
        # Bewertung 0 ändert nichts, 1 ohne Lektion auch nicht
        self.store.rate(ids[4], 0)
        cur = self.store.list_lessons(include_inactive=True)[0]
        self.assertEqual((cur.helped, cur.hurt), (1, 3))

    def test_deactivation_needs_hurt_dominance(self):
        l = self.store.add_lesson("Gute Regel", scope="allgemein")
        ids = [self.store.record("s", f"F{i}?", LONG, lessons_used=[l.id]) for i in range(6)]
        for i in ids[:3]:
            self.store.rate(i, 1)
        for i in ids[3:]:
            self.store.rate(i, -1)       # hurt 3, helped 3 -> 3 > 6 ist falsch -> bleibt aktiv
        cur = self.store.list_lessons()[0]
        self.assertEqual((cur.helped, cur.hurt, cur.active), (3, 3, True))

    def test_rerating_reverts_previous_count(self):
        l = self.store.add_lesson("Regel", scope="allgemein")
        iid = self.store.record("s", "F?", LONG, lessons_used=[l.id])
        self.store.rate(iid, 1)
        self.store.rate(iid, 1)
        self.assertEqual(self.store.list_lessons()[0].helped, 1)
        self.store.rate(iid, -1)
        cur = self.store.list_lessons()[0]
        self.assertEqual((cur.helped, cur.hurt), (0, 1))
        self.store.rate(iid, 0)
        cur = self.store.list_lessons()[0]
        self.assertEqual((cur.helped, cur.hurt), (0, 0))

    def test_correction_counts_as_hurt_when_unrated(self):
        l = self.store.add_lesson("Regel", scope="allgemein")
        iid = self.store.record("s", "F?", LONG, lessons_used=[l.id])
        self.store.correct(iid, "besser")
        self.assertEqual(self.store.list_lessons()[0].hurt, 1)
        self.store.correct(iid, "noch besser")     # schon -1: kein zweites Mal
        self.assertEqual(self.store.list_lessons()[0].hurt, 1)

    def test_lessons_general_limit_and_ranking(self):
        a = self.store.add_lesson("allgemein a", scope="allgemein")
        b = self.store.add_lesson("allgemein b", scope="allgemein")
        c = self.store.add_lesson("allgemein c", scope="allgemein")
        # b am besten, c zweitbeste (gleichstand mit a -> neuer gewinnt)
        for _ in range(2):
            iid = self.store.record("s", "x?", LONG, lessons_used=[b.id])
            self.store.rate(iid, 1)
        got = self.store.lessons("irgendwas ohne Treffer", k=5)
        self.assertEqual([l.id for l in got], [b.id, c.id])
        self.assertEqual([l.id for l in self.store.lessons("x", k=1)], [b.id])
        self.assertEqual(self.store.lessons("x", k=0), [])

    def test_lessons_topic_via_fts_rule_and_topics(self):
        g = self.store.add_lesson("Allgemeine Regel", scope="allgemein")
        t1 = self.store.add_lesson("Bei Akkus Ladestrom in C angeben", topics=["akku"])
        t2 = self.store.add_lesson("Brennweite in mm nennen", topics=["kamera", "optik"])
        t3 = self.store.add_lesson("Nicht relevant", topics=["kuchen"])
        got = self.store.lessons("Wie lade ich den Akku?", k=3)
        self.assertEqual([l.id for l in got], [g.id, t1.id])
        got = self.store.lessons("Welche Optik für die Kamera?", k=3)
        self.assertEqual([l.id for l in got], [g.id, t2.id])
        got = self.store.lessons("Akku Kamera Kuchen", k=2)
        self.assertEqual(len(got), 2)
        self.assertEqual(got[0].id, g.id)
        self.assertNotIn(t3.id, [l.id for l in self.store.lessons("Akku", k=3)])

    def test_lessons_ignore_inactive(self):
        t = self.store.add_lesson("Akku Regel", topics=["akku"])
        ids = [self.store.record("s", "F?", LONG, lessons_used=[t.id]) for _ in range(3)]
        for i in ids:
            self.store.rate(i, -1)
        self.assertFalse(self.store.list_lessons(include_inactive=True)[0].active)
        self.assertEqual(self.store.lessons("Akku"), [])

    def test_lessons_with_vectors(self):
        be = FakeBackend(embed_dim=256)
        store = LearningStore(":memory:", embedder=be.embed)
        t1 = store.add_lesson("Drohne Carbon Rahmen steif bauen", topics=["drohne"])
        store.add_lesson("Apfelkuchen Rezept Butter Zucker", topics=["kuchen"])
        got = store.lessons("Rahmen Drohne Carbon", k=3)
        self.assertEqual([l.id for l in got], [t1.id])
        store.close()

    def test_embedder_failure_tolerated(self):
        def broken(texts):
            raise RuntimeError("kaputt")
        store = LearningStore(":memory:", embedder=broken)
        t = store.add_lesson("Akku Regel", topics=["akku"])
        iid = store.record("s", "Akku?", LONG)
        self.assertEqual([l.id for l in store.lessons("Akku")], [t.id])
        self.assertEqual(store.get(iid).question, "Akku?")
        store.close()


class ExamplesTest(unittest.TestCase):
    def tearDown(self):
        self.store.close()

    def _fill(self, store):
        self.a = store.record("s", "Wie lade ich den Drohnen-Akku?", "Akku Antwort " + LONG)
        self.b = store.record("s", "Welcher Akku für die Drohne?", "Zweite Akku Antwort " + LONG)
        self.c = store.record("s", "Rezept für Apfelkuchen?", "Kuchen " + LONG)
        self.d = store.record("s", "Akku Ladestrom?", "mit Werkzeug", tools_used=["rechnen"])
        self.e = store.record("s", "Akku Spannung?", "unbewertet")
        store.rate(self.a, 1)
        store.correct(self.b, "Kurzkorrektur", "Vollständig korrigierte Akku-Antwort.")
        store.rate(self.c, 1)
        store.rate(self.d, 1)          # nicht trainierbar -> nie Beispiel

    def test_examples_without_vectors(self):
        self.store = LearningStore(":memory:")
        self._fill(self.store)
        ex = self.store.examples("Akku Drohne laden", k=3)
        self.assertEqual(len(ex), 2)
        self.assertEqual({q for q, _ in ex}, {"Wie lade ich den Drohnen-Akku?", "Welcher Akku für die Drohne?"})
        ans = dict(ex)
        self.assertEqual(ans["Welcher Akku für die Drohne?"], "Vollständig korrigierte Akku-Antwort.")
        self.assertEqual(ans["Wie lade ich den Drohnen-Akku?"], "Akku Antwort " + LONG)
        self.assertEqual(self.store.examples("Akku Drohne laden", k=1)[0][0], "Wie lade ich den Drohnen-Akku?")
        self.assertEqual(self.store.examples("Quantenphysik"), [])
        self.assertEqual(self.store.examples("Akku", k=0), [])

    def test_examples_with_vectors_rank_overlap(self):
        be = FakeBackend(embed_dim=256)
        self.store = LearningStore(":memory:", embedder=be.embed)
        self._fill(self.store)
        ex = self.store.examples("Drohnen-Akku laden", k=3)
        self.assertEqual(ex[0][0], "Wie lade ich den Drohnen-Akku?")
        self.assertEqual([q for q, _ in ex][1], "Welcher Akku für die Drohne?")
        self.assertNotIn("Rezept für Apfelkuchen?", [q for q, _ in ex])
        # reine Vektor-Nähe ohne FTS-Treffer: gleiche Wörter in anderer Form
        ex = self.store.examples("Apfelkuchen Rezept", k=1)
        self.assertEqual(ex[0][0], "Rezept für Apfelkuchen?")

    def test_examples_strip_tool_blocks(self):
        self.store = LearningStore(":memory:")
        iid = self.store.record("s", "Wie spät?", 'Es ist <werkzeug>{"name":"zeit","args":{}}</werkzeug> 12 Uhr.')
        self.store.rate(iid, 1)
        self.assertEqual(self.store.examples("spät"), [("Wie spät?", "Es ist  12 Uhr.")])


class ExportTest(unittest.TestCase):
    def setUp(self):
        self.store = LearningStore(":memory:")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "ds.jsonl")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _files(self):
        return self.path, self.path + ".eval.jsonl", self.path + ".fragen.jsonl"

    def test_empty_export_creates_files(self):
        res = self.store.export_dataset(self.path)
        self.assertEqual(res, {"train": 0, "eval": 0,
                               "verworfen": {"duplikat": 0, "zu_kurz": 0, "werkzeug": 0, "fehler": 0,
                                             "eigene_limit": 0}})
        for f in self._files():
            self.assertTrue(os.path.exists(f), f)
            self.assertEqual(os.path.getsize(f), 0)

    def test_chat_format_and_context(self):
        iid = self.store.record("s", "Frage?", LONG, context="Projekt: X",
                                history=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hallo"}])
        self.store.rate(iid, 1)
        res = self.store.export_dataset(self.path, eval_share=0.0, system_prompt="SYS")
        self.assertEqual((res["train"], res["eval"]), (1, 0))
        rows = read_jsonl(self.path)
        self.assertEqual(rows[0]["messages"], [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hallo"},
            {"role": "user", "content": "Projekt: X\n\nFrage?"},
            {"role": "assistant", "content": LONG},
        ])
        self.store.export_dataset(self.path, eval_share=0.0, include_context=False)
        rows = read_jsonl(self.path)
        self.assertEqual(rows[0]["messages"][3]["content"], "Frage?")
        sys_prompt = rows[0]["messages"][0]["content"]
        self.assertTrue(sys_prompt.strip())
        self.assertIn("Deutsch", sys_prompt + FALLBACK_SYSTEM_PROMPT)
        # fragen.jsonl aus dem Eval-Anteil, Fallback: alle
        fragen = read_jsonl(self._files()[2])
        self.assertEqual(fragen[0]["frage"], "Frage?")
        self.assertEqual(fragen[0]["erwartet"], LONG)
        self.assertEqual(fragen[0]["quelle_id"], iid)
        self.assertEqual(fragen[0]["stichworte"], keywords(LONG))
        self.assertLessEqual(len(fragen[0]["stichworte"]), 8)
        self.assertTrue(all(len(w) >= 5 for w in fragen[0]["stichworte"]))

    def test_alpaca_format(self):
        iid = self.store.record("s", "Frage?", LONG, context="Kontext")
        self.store.rate(iid, 1)
        self.store.export_dataset(self.path, fmt="alpaca", eval_share=0.0)
        self.assertEqual(read_jsonl(self.path), [{"instruction": "Frage?", "input": "Kontext", "output": LONG}])
        self.store.export_dataset(self.path, fmt="alpaca", eval_share=0.0, include_context=False)
        self.assertEqual(read_jsonl(self.path)[0]["input"], "")

    def test_dpo_only_with_corrections(self):
        good = self.store.record("s", "Gut?", LONG, context="K")
        self.store.rate(good, 1)
        bad = self.store.record("s", "Schlecht?", "Falsche Antwort mit genug Zeichen für den Export hier.")
        self.store.correct(bad, "kurz", "Korrigierte Antwort mit genug Zeichen für den Export hier.")
        same = self.store.record("s", "Gleich?", LONG)
        self.store.correct(same, LONG)      # chosen == rejected -> kein Paar
        res = self.store.export_dataset(self.path, fmt="dpo", eval_share=0.0)
        self.assertEqual((res["train"], res["eval"]), (1, 0))
        self.assertEqual(res["verworfen"]["duplikat"], 1)
        rows = read_jsonl(self.path)
        self.assertEqual(rows, [{"prompt": "Schlecht?",
                                 "chosen": "Korrigierte Antwort mit genug Zeichen für den Export hier.",
                                 "rejected": "Falsche Antwort mit genug Zeichen für den Export hier."}])
        with self.assertRaises(ValueError):
            self.store.export_dataset(self.path, fmt="xml")

    def test_selection_and_verworfen_counts(self):
        a = self.store.record("s", "A?", LONG)
        self.store.rate(a, 1)
        tool = self.store.record("s", "T?", LONG, tools_used=["rechnen"])
        self.store.rate(tool, 1)
        err = self.store.record("s", "E?", LONG, depth="fehler")
        self.store.rate(err, 1)
        short = self.store.record("s", "K?", "zu kurz")
        self.store.rate(short, 1)
        self.store.record("s", "U?", LONG)                    # unbewertet -> nicht ausgewählt
        neg = self.store.record("s", "N?", LONG)
        self.store.rate(neg, -1)                              # negativ ohne Korrektur -> nicht ausgewählt
        corr = self.store.record("s", "C?", "Falsch " + LONG)
        self.store.correct(corr, "Richtig " + LONG)
        res = self.store.export_dataset(self.path, eval_share=0.0)
        self.assertEqual(res["train"], 2)
        self.assertEqual(res["verworfen"], {"duplikat": 0, "zu_kurz": 1, "werkzeug": 1, "fehler": 1,
                                            "eigene_limit": 0})
        targets = [r["messages"][-1]["content"] for r in read_jsonl(self.path)]
        self.assertEqual(targets, [LONG, "Richtig " + LONG])
        # min_rating=0 nimmt auch Unbewertete, aber nicht Negative ohne Korrektur
        res = self.store.export_dataset(self.path, eval_share=0.0, min_rating=0)
        self.assertEqual(res["train"], 3)
        res = self.store.export_dataset(self.path, eval_share=0.0, min_rating=-1)
        self.assertEqual(res["train"], 4)

    def test_dedup_keeps_corrected_or_newest(self):
        for _ in range(3):
            iid = self.store.record("s", "Gleiche Frage?", LONG)
            self.store.rate(iid, 1)
        other = self.store.record("s", "gleiche FRAGE!", LONG + " Zusatz")
        self.store.rate(other, 1)
        corr = self.store.record("s", "Gleiche Frage?", "ganz anders " + LONG)
        self.store.correct(corr, LONG)      # gleiche Frage + gleiches Ziel wie die 3 oben
        res = self.store.export_dataset(self.path, eval_share=0.0)
        self.assertEqual(res["train"], 2)
        self.assertEqual(res["verworfen"]["duplikat"], 3)
        rows = read_jsonl(self.path)
        targets = {r["messages"][-1]["content"] for r in rows}
        self.assertEqual(targets, {LONG, LONG + " Zusatz"})
        # korrigierte Variante bleibt (quelle_id in fragen.jsonl)
        ids = {f["quelle_id"] for f in read_jsonl(self._files()[2])}
        self.assertEqual(ids, {corr, other})
        res = self.store.export_dataset(self.path, eval_share=0.0, dedup=False)
        self.assertEqual((res["train"], res["verworfen"]["duplikat"]), (5, 0))

    def test_max_own_cap_keeps_corrected(self):
        for i in range(6):
            iid = self.store.record("s", f"Eigene {i}?", f"{LONG} Nummer {i}")
            self.store.rate(iid, 1)
        for i in range(2):
            iid = self.store.record("s", f"Korrigiert {i}?", f"falsch {i} " + LONG)
            self.store.correct(iid, f"richtig {i} " + LONG)
        res = self.store.export_dataset(self.path, eval_share=0.0, max_own=2)
        self.assertEqual(res["train"], 4)
        self.assertEqual(res["verworfen"]["eigene_limit"], 4)
        questions = [r["messages"][-2]["content"] for r in read_jsonl(self.path)]
        self.assertEqual(questions, ["Eigene 4?", "Eigene 5?", "Korrigiert 0?", "Korrigiert 1?"])  # neueste eigene
        res = self.store.export_dataset(self.path, eval_share=0.0, max_own=0)
        self.assertEqual((res["train"], res["verworfen"]["eigene_limit"]), (2, 6))
        # Standard: mindestens 20 eigene erlaubt
        res = self.store.export_dataset(self.path, eval_share=0.0)
        self.assertEqual((res["train"], res["verworfen"]["eigene_limit"]), (8, 0))

    def test_split_deterministic_and_all_files(self):
        for i in range(40):
            iid = self.store.record("s", f"Frage Nummer {i} über Thema {i * 7}?", f"{LONG} Variante {i}")
            self.store.rate(iid, 1)
        # ohne Korrekturen gilt die Standard-Obergrenze von 20 eigenen Antworten
        capped = self.store.export_dataset(self.path, eval_share=0.3)
        self.assertEqual((capped["train"] + capped["eval"], capped["verworfen"]["eigene_limit"]), (20, 20))
        res1 = self.store.export_dataset(self.path, eval_share=0.3, max_own=100)
        self.assertEqual(res1["train"] + res1["eval"], 40)
        self.assertGreater(res1["train"], 0)
        self.assertGreater(res1["eval"], 0)
        contents1 = []
        for f in self._files():
            with open(f, encoding="utf-8") as fh:
                contents1.append(fh.read())
        self.assertEqual(len(read_jsonl(self._files()[0])), res1["train"])
        self.assertEqual(len(read_jsonl(self._files()[1])), res1["eval"])
        fragen = read_jsonl(self._files()[2])
        self.assertEqual(len(fragen), res1["eval"])
        self.assertEqual(set(fragen[0]), {"frage", "erwartet", "stichworte", "quelle_id"})
        # zweiter Export (auch nach weiteren Daten) verteilt gleich
        extra = self.store.record("s", "Frage Nummer 40 über Thema 280?", LONG + " Variante 40")
        self.store.rate(extra, 1)
        res2 = self.store.export_dataset(self.path, eval_share=0.3, seed=7, max_own=100)
        self.assertEqual(res2["train"] + res2["eval"], 41)
        eval1 = {r["messages"][-2]["content"] for r in json.loads("[" + ",".join(
            contents1[1].strip().splitlines()) + "]")}
        eval2 = {r["messages"][-2]["content"] for r in read_jsonl(self._files()[1])}
        self.assertTrue(eval1 <= eval2)
        train2 = {r["messages"][-2]["content"] for r in read_jsonl(self._files()[0])}
        self.assertEqual(eval2 & train2, set())
        # eval_share 1.0 -> alles in eval, 0 -> alles train
        res3 = self.store.export_dataset(self.path, eval_share=1.0, max_own=100)
        self.assertEqual((res3["train"], res3["eval"]), (0, 41))
        res4 = self.store.export_dataset(self.path, eval_share=0, max_own=100)
        self.assertEqual((res4["train"], res4["eval"]), (41, 0))
        self.assertEqual(len(read_jsonl(self._files()[2])), 41)    # Fallback: alle Fragen

    def test_export_creates_parent_dir_and_strips_tools(self):
        path = os.path.join(self.tmp.name, "tief", "er", "ds.jsonl")
        iid = self.store.record("s", "Frage?", 'Text <werkzeug>{"name":"zeit","args":{}}</werkzeug>\n\n' + LONG)
        self.store.rate(iid, 1)
        res = self.store.export_dataset(path, eval_share=0.0)
        self.assertEqual(res["train"], 1)
        self.assertEqual(read_jsonl(path)[0]["messages"][-1]["content"], "Text\n\n" + LONG)


class HelpersTest(unittest.TestCase):
    def test_strip_tool_blocks(self):
        self.assertEqual(strip_tool_blocks(""), "")
        self.assertEqual(strip_tool_blocks("nichts"), "nichts")
        self.assertEqual(strip_tool_blocks('a <werkzeug>{"name":"x","args":{}}</werkzeug> b'), "a  b")
        self.assertEqual(strip_tool_blocks('a <WERKZEUG >{"x":1}</Werkzeug>'), "a")
        self.assertEqual(strip_tool_blocks('Text <werkzeug>{"name": "offen"'), "Text")
        self.assertEqual(strip_tool_blocks('x\n```json\n{"name": "zeit", "args": {}}\n```\ny'), "x\n\ny")
        self.assertEqual(strip_tool_blocks("x\n```python\nprint(1)\n```\ny"), "x\n```python\nprint(1)\n```\ny")
        self.assertEqual(strip_tool_blocks('```json\n{"name": "zeit", "extra": 1}\n```'),
                         '```json\n{"name": "zeit", "extra": 1}\n```')

    def test_keywords(self):
        self.assertEqual(keywords("Der Akku lädt mit 1500 mAh Ladestrom, Ladestrom!"), ["ladestrom"])
        self.assertEqual(keywords(""), [])
        self.assertEqual(len(keywords(" ".join(f"wort{i}xyz" for i in range(20)))), 8)
        self.assertEqual(keywords("12345 Zahlen zählen nicht"), ["zahlen", "zählen", "nicht"])


if __name__ == "__main__":
    unittest.main()
