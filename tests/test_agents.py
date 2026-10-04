import json
import time
import unittest
from dataclasses import dataclass

from obito import agents
from obito.agents import (BASE_RULES, CRITIC, CRITIC_SCHEMA, EXPERTS, FILL_ORDER, JUDGE_SCHEMA,
                          LESSON_SCHEMA, MAX_ANSWER_IN_PROMPT, MAX_CONTEXT_CHARS, MAX_CRITIQUE_CHARS,
                          MAX_EXAMPLE_CHARS, MAX_HISTORY_CHARS, MEMORY_SCHEMA, OMEGA, PAIRWISE_SCHEMA,
                          ROUTING_SCHEMA, STAGE_TAG, Expert, clip, context_block, correction_rewrite_messages,
                          critic_messages, estimate_tokens, example_messages, expert_messages, fast_messages,
                          fit_messages, heuristic_complexity, judge_messages, lesson_extraction_messages,
                          memory_extraction_messages, pairwise_judge_messages, parse_critique, parse_judge,
                          parse_lesson, parse_memories, parse_pairwise, parse_routing, revision_messages,
                          routing_messages, select_experts, stage_of, tool_result_message)
from obito.llm import FakeBackend
from obito.memory import KINDS, Memory


# ------------------------------------------------------------ Attrappen
@dataclass
class FakeToolResult:
    ok: bool
    output: str
    error: str | None = None


class FakeLesson:
    def __init__(self, rule: str, helped: int = 4):
        self.rule = rule
        self.helped = helped

    def render(self) -> str:
        return f"Regel (bestätigt {self.helped}×): {self.rule}"[:300]


def mem(content: str, kind: str = "fakt", mid: int = 1) -> Memory:
    now = time.time()
    return Memory(id=mid, kind=kind, content=content, tags=[], project=None, source="nutzer",
                  importance=0.5, created_at=now, last_access=now, access_count=0)


HISTORY = [{"role": "user", "content": "Erste Frage"}, {"role": "assistant", "content": "Erste Antwort"}]
EXAMPLES = [("Beispielfrage", "Beispielantwort")]
ANSWERS = {"ALPHA": "Antwort von Alpha mit 12 mm.", "BETA": "Antwort von Beta mit 4S."}
CRITIQUE = {"bewertung": 6, "fehler": [{"experte": "ALPHA", "problem": "Zu dünn", "korrektur": "1,2 mm",
                                        "schwere": "hoch"}],
            "widersprueche": ["ALPHA vs BETA"], "fehlt": ["Temperatur"], "sicher": True, "revidiert": ["ALPHA"]}


def all_builders() -> list[tuple[str, str, str, list[dict]]]:
    """(Name, erwartete Stufe, erwarteter who, Nachrichten) für jeden Prompt-Bauer."""
    alpha = EXPERTS["ALPHA"]
    return [
        ("routing", "routing", "system", routing_messages("Frage?", ["ALPHA", "BETA"])),
        ("expert", "experte", "ALPHA", expert_messages(alpha, "Frage?", "Kontext", HISTORY)),
        ("critic", "kritiker", "KRITIKER", critic_messages("Frage?", ANSWERS, "Kontext")),
        ("revision", "revision", "ALPHA", revision_messages(alpha, "Frage?", "alt", CRITIQUE["fehler"])),
        ("synthesis", "synthese", "OMEGA",
         synthesis_messages_wrapper("Frage?", ANSWERS, CRITIQUE, "Kontext", HISTORY, EXAMPLES, "rechnen(ausdruck)")),
        ("synthesis_medium", "synthese", "OMEGA",
         synthesis_messages_wrapper("Frage?", ANSWERS, None, "", [], [], medium=True)),
        ("fast", "schnell", "OMEGA", fast_messages("Frage?", "Kontext", HISTORY, EXAMPLES, "rechnen(ausdruck)")),
        ("extraction", "extraktion", "system", memory_extraction_messages("Frage?", "Antwort", "Projekt X")),
        ("lesson", "lektion", "system", lesson_extraction_messages("Frage?", "Antwort", "falsch", "besser so")),
        ("correction", "korrektur", "OMEGA", correction_rewrite_messages("Frage?", "Antwort", "Korrektur")),
        ("judge", "richter", "system", judge_messages("Frage?", "erwartet", "Antwort")),
        ("pairwise", "richter", "system", pairwise_judge_messages("Frage?", "erwartet", "A1", "A2")),
    ]


def synthesis_messages_wrapper(*args, **kw):
    return agents.synthesis_messages(*args, **kw)


# ------------------------------------------------------------ Experten
class ExpertsTest(unittest.TestCase):
    def test_eleven_experts_with_expected_ids(self):
        self.assertEqual(set(EXPERTS), {"ALPHA", "BETA", "GAMMA", "DELTA", "EPSILON", "ZETA", "THETA", "IOTA",
                                        "KAPPA", "LAMBDA", "GENERALIST"})
        for eid, ex in EXPERTS.items():
            self.assertIsInstance(ex, Expert)
            self.assertEqual(ex.id, eid)
            self.assertTrue(ex.name and ex.role and ex.system_prompt)
            self.assertGreater(len(ex.system_prompt), 150, eid)
            self.assertGreaterEqual(len(ex.keywords), 20, eid)
            self.assertEqual(len(set(ex.keywords)), len(ex.keywords), f"doppelte Keywords bei {eid}")
            for kw in ex.keywords:
                self.assertEqual(kw, kw.lower(), kw)
                self.assertFalse(any(c in kw for c in "äöüß"), kw)
            self.assertTrue(0.0 <= ex.temperature <= 1.0)
        self.assertEqual(EXPERTS["ALPHA"].name, "3D-Konstruktion")

    def test_critic_and_omega(self):
        self.assertEqual(CRITIC.id, "KRITIKER")
        self.assertEqual(OMEGA.id, "OMEGA")
        self.assertNotIn("KRITIKER", EXPERTS)
        self.assertNotIn("OMEGA", EXPERTS)
        self.assertIn("Widerspr", CRITIC.system_prompt)

    def test_expert_is_frozen(self):
        with self.assertRaises(Exception):
            EXPERTS["ALPHA"].name = "x"  # type: ignore[misc]

    def test_base_rules_content(self):
        for needle in ("Deutsch", "Einheiten", "Unsicher:", "Lektionen", "Rückfragen", "Sicherheit"):
            self.assertIn(needle, BASE_RULES)

    def test_fill_order(self):
        self.assertEqual(FILL_ORDER[:5], ("GENERALIST", "KAPPA", "DELTA", "GAMMA", "ALPHA"))
        self.assertEqual(set(FILL_ORDER), set(EXPERTS))


# ------------------------------------------------------------ Marker
class StageMarkerTest(unittest.TestCase):
    def test_every_builder_has_marker_as_last_line(self):
        for name, stage, who, msgs in all_builders():
            with self.subTest(builder=name):
                self.assertIsInstance(msgs, list)
                self.assertGreaterEqual(len(msgs), 2)
                self.assertEqual(msgs[0]["role"], "system")
                last_line = msgs[0]["content"].rstrip().splitlines()[-1]
                self.assertEqual(last_line, STAGE_TAG.format(stage=stage, who=who))
                self.assertEqual(stage_of(msgs), (stage, who))
                self.assertEqual(msgs[-1]["role"], "user")
                for m in msgs:
                    self.assertIn(m["role"], ("system", "user", "assistant"))
                    self.assertIsInstance(m["content"], str)
                    self.assertTrue(m["content"])
                # Marker steht genau einmal in der System-Nachricht
                self.assertEqual(msgs[0]["content"].count("[OBITO:"), 1)

    def test_stage_of_fallbacks(self):
        self.assertEqual(stage_of([]), ("?", "?"))
        self.assertEqual(stage_of([{"role": "user", "content": "[OBITO:schnell:OMEGA]"}]), ("?", "?"))
        self.assertEqual(stage_of([{"role": "system", "content": "ohne Marker"}]), ("?", "?"))
        self.assertEqual(stage_of([{"role": "system", "content": "[OBITO:experte:ALPHA] nicht am Ende"}]), ("?", "?"))
        self.assertEqual(stage_of([{"role": "system", "content": "x\n[OBITO:experte:ALPHA]\n  "}]), ("experte", "ALPHA"))
        self.assertEqual(stage_of([{"role": "system", "content": None}]), ("?", "?"))
        self.assertEqual(stage_of(None), ("?", "?"))  # type: ignore[arg-type]
        self.assertEqual(stage_of([{"role": "user", "content": "x"},
                                   {"role": "system", "content": "[OBITO:richter:system]"}]), ("richter", "system"))

    def test_fake_backend_script_by_stage(self):
        script = {("experte", "ALPHA"): "Alpha sagt", ("synthese", "OMEGA"): "Omega fasst zusammen"}
        fb = FakeBackend(responder=lambda msgs, kw: script.get(stage_of(msgs), "?"))
        self.assertEqual(fb.chat(expert_messages(EXPERTS["ALPHA"], "f", "", [])).text, "Alpha sagt")
        self.assertEqual(fb.chat(agents.synthesis_messages("f", ANSWERS, None, "", [], [])).text,
                         "Omega fasst zusammen")
        self.assertEqual(fb.chat(expert_messages(EXPERTS["BETA"], "f", "", [])).text, "?")


# ------------------------------------------------------------ Budget
class ClipTokensTest(unittest.TestCase):
    def test_clip(self):
        self.assertEqual(clip("abc", 5), "abc")
        self.assertEqual(clip("abc", 3), "abc")
        self.assertEqual(clip("abcdef", 3), "abc … [gekürzt]")
        self.assertEqual(clip("ab  cdef", 4), "ab … [gekürzt]")
        self.assertEqual(clip(None, 3), "")  # type: ignore[arg-type]
        self.assertEqual(clip("abc", 0), " … [gekürzt]")
        self.assertEqual(clip("abc", -2), " … [gekürzt]")

    def test_estimate_tokens(self):
        self.assertEqual(estimate_tokens(""), 1)
        self.assertEqual(estimate_tokens("abc"), 2)
        self.assertEqual(estimate_tokens("a" * 300), 101)
        self.assertEqual(estimate_tokens(None), 1)  # type: ignore[arg-type]

    def test_constants(self):
        self.assertEqual(MAX_CONTEXT_CHARS, 3000)
        self.assertEqual(MAX_HISTORY_CHARS, 3000)
        self.assertEqual(MAX_EXAMPLE_CHARS, 400)
        self.assertEqual(MAX_ANSWER_IN_PROMPT, 2500)
        self.assertEqual(MAX_CRITIQUE_CHARS, 1500)
        self.assertEqual(agents.MAX_MEMORY_CONTEXT_CHARS, 1200)
        self.assertEqual(agents.MAX_LESSON_CONTEXT_CHARS, 800)


class FitMessagesTest(unittest.TestCase):
    def make(self, n_history=6, hist_len=300, last_len=300):
        msgs = [{"role": "system", "content": "S" * 100}]
        for i in range(n_history):
            role = "user" if i % 2 == 0 else "assistant"
            msgs.append({"role": role, "content": f"{role}-{i}-" + "h" * hist_len})
        msgs.append({"role": "user", "content": "LETZTE " + "q" * last_len})
        return msgs

    def tokens(self, msgs):
        return sum(estimate_tokens(m["content"]) for m in msgs)

    def test_untouched_when_fits_and_returns_copies(self):
        msgs = self.make()
        out = fit_messages(msgs, 10_000)
        self.assertEqual(out, msgs)
        self.assertIsNot(out, msgs)
        self.assertIsNot(out[0], msgs[0])

    def test_drops_oldest_history_first(self):
        msgs = self.make(n_history=6, hist_len=300, last_len=300)
        total = self.tokens(msgs)
        out = fit_messages(msgs, total - 150)
        self.assertEqual(out[0], msgs[0])
        self.assertEqual(out[-1], msgs[-1])
        self.assertLessEqual(self.tokens(out), total - 150)
        kept = [m["content"][:12] for m in out[1:-1]]
        # die ältesten (user-0, assistant-1) sind weg, die jüngsten bleiben
        self.assertNotIn("user-0-hhhhh", kept)
        self.assertIn("assistant-5-", kept)
        # Reihenfolge bleibt chronologisch
        self.assertEqual(kept, [m["content"][:12] for m in msgs[1:-1] if m["content"][:12] in kept])
        # Verlaufsnachrichten wurden nicht gekürzt, nur entfernt
        for m in out[1:-1]:
            self.assertNotIn("[gekürzt]", m["content"])

    def test_clips_assistant_history_after_dropping(self):
        msgs = [{"role": "system", "content": "S" * 30},
                {"role": "user", "content": "u" * 100},
                {"role": "assistant", "content": "a" * 3000},
                {"role": "user", "content": "LETZTE"}]
        # Budget: System (11) + user (34) + LETZTE (3) + gekürzter Assistant (~205) passt;
        # der ungekürzte Assistant (1001) nicht. Es wird zuerst entfernt (user), dann gekürzt.
        out = fit_messages(msgs, 11 + 3 + 210)
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[-1]["content"], "LETZTE")
        self.assertEqual(len(out), 3)
        self.assertEqual(out[1]["role"], "assistant")
        self.assertTrue(out[1]["content"].endswith(" … [gekürzt]"))
        self.assertLessEqual(len(out[1]["content"]), 600 + len(agents.CLIP_SUFFIX))

    def test_clips_last_user_last_keeps_head(self):
        msgs = [{"role": "system", "content": "S" * 30},
                {"role": "user", "content": "KOPF " + "x" * 5000}]
        out = fit_messages(msgs, 300)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["content"], "S" * 30)
        self.assertTrue(out[1]["content"].startswith("KOPF xxxx"))
        self.assertTrue(out[1]["content"].endswith("[gekürzt]"))
        self.assertLessEqual(self.tokens(out), 300)

    def test_never_drops_system_or_last_user(self):
        msgs = self.make(n_history=10, hist_len=500, last_len=5000)
        out = fit_messages(msgs, 5)   # absurd kleines Budget
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[0]["content"], "S" * 100)
        self.assertEqual(out[-1]["role"], "user")
        self.assertTrue(out[-1]["content"].startswith("LETZTE qqq"))
        self.assertGreaterEqual(len(out[-1]["content"]), 200)   # Mindest-Kopf
        self.assertEqual(len(out), 2)

    def test_keep_last_user_false_allows_dropping_it(self):
        msgs = [{"role": "system", "content": "S" * 30},
                {"role": "user", "content": "u" * 900},
                {"role": "assistant", "content": "a" * 900},
                {"role": "user", "content": "LETZTE " + "q" * 900}]
        out = fit_messages(msgs, 100, keep_last_user=False)
        self.assertEqual(out[0]["role"], "system")
        self.assertLessEqual(len(out), 2)
        if len(out) == 2:
            self.assertTrue(out[1]["content"].startswith("LETZTE"))
            self.assertIn("[gekürzt]", out[1]["content"])

    def test_without_system_message(self):
        msgs = [{"role": "user", "content": "u" * 600}, {"role": "assistant", "content": "a" * 600},
                {"role": "user", "content": "L" * 60}]
        out = fit_messages(msgs, 30)
        self.assertEqual(out, [{"role": "user", "content": "L" * 60}])

    def test_tool_loop_shape_keeps_tool_result_as_last_user(self):
        msgs = [{"role": "system", "content": "S"},
                {"role": "user", "content": "Frage " + "f" * 2000},
                {"role": "assistant", "content": "<werkzeug>{}</werkzeug>" + "x" * 2000},
                {"role": "user", "content": "Ergebnis von Werkzeug »rechnen«:\n42"}]
        out = fit_messages(msgs, 300)
        self.assertEqual(out[-1]["content"], "Ergebnis von Werkzeug »rechnen«:\n42")
        self.assertEqual(out[0]["content"], "S")

    def test_non_string_content_tolerated(self):
        out = fit_messages([{"role": "system", "content": None}, {"role": "user", "content": 42}], 1000)
        self.assertEqual(out[0]["content"], "")
        self.assertEqual(out[1]["content"], "42")

    def test_empty(self):
        self.assertEqual(fit_messages([], 10), [])


# ------------------------------------------------------------ Experten-Wahl
class SelectExpertsTest(unittest.TestCase):
    def ids(self, *a, **kw):
        return [e.id for e in select_experts(*a, **kw)]

    def test_hint_first_unknown_ignored(self):
        ids = self.ids("Irgendeine Frage", 3, hint=["ZETA", "unbekannt", "KRITIKER", "OMEGA", "iota"])
        self.assertEqual(ids[:2], ["ZETA", "IOTA"])
        self.assertEqual(len(ids), 3)
        self.assertNotIn("KRITIKER", ids)
        self.assertNotIn("OMEGA", ids)

    def test_hint_by_name(self):
        self.assertEqual(self.ids("x", 1, hint=["Code & Software"]), ["DELTA"])
        self.assertEqual(self.ids("x", 1, hint=["Experte ALPHA"]), ["ALPHA"])

    def test_keyword_matches_ranked(self):
        ids = self.ids("Welchen Motor und welchen Akku brauche ich für einen Quadcopter aus Carbon?", 3)
        self.assertEqual(ids[0], "BETA")      # motor + akku
        self.assertIn("IOTA", ids)           # carbon
        self.assertEqual(len(set(ids)), 3)

    def test_umlaut_and_case_tolerant_keywords(self):
        self.assertEqual(self.ids("WIE DICK MUSS DIE WANDSTÄRKE DES GEHÄUSES SEIN", 1), ["ALPHA"])
        self.assertEqual(self.ids("Welche Lötstation für Platinen?", 1), ["BETA"])

    def test_short_keywords_match_whole_words_only(self):
        # "pla" (IOTA) darf nicht in "Plane" treffen
        self.assertNotIn("IOTA", self.ids("Plane meinen Urlaub in Italien", 2))
        self.assertIn("IOTA", self.ids("Soll ich PLA oder PETG nehmen?", 1))

    def test_fill_order_without_matches(self):
        self.assertEqual(self.ids("Grüß dich", 5), ["GENERALIST", "KAPPA", "DELTA", "GAMMA", "ALPHA"])
        self.assertEqual(self.ids("", 11), list(FILL_ORDER))
        self.assertEqual(len(self.ids("", 20)), 11)

    def test_no_duplicates_with_hint_and_keywords(self):
        ids = self.ids("Python Code mit einem Bug", 4, hint=["DELTA", "DELTA"])
        self.assertEqual(ids[0], "DELTA")
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 4)

    def test_k_limits(self):
        self.assertEqual(self.ids("x", 0), [])
        self.assertEqual(self.ids("x", -1), [])
        self.assertEqual(len(self.ids("Motor Akku Carbon Python", 2, hint=["ZETA", "THETA", "LAMBDA"])), 2)

    def test_returns_expert_objects(self):
        for e in select_experts("Drohne", 5):
            self.assertIsInstance(e, Expert)


class HeuristicComplexityTest(unittest.TestCase):
    def test_einfach(self):
        for q in ["Hallo, wie geht es dir?", "Wie heißt die Hauptstadt von Frankreich?", "Danke!",
                  "Warum ist der Himmel blau?", "Berechne 17*23", "", "Wie spät ist es?",
                  "Hallo OBITO, ich wollte nur kurz fragen, ob du heute gut drauf bist?"]:
            self.assertEqual(heuristic_complexity(q), "einfach", q)

    def test_mittel(self):
        for q in ["Welchen Motor nehme ich für einen 250 g Quadcopter?",
                  "Wie dick sollte die Wandstärke bei PETG sein?",
                  "Mein Python-Skript wirft eine Exception beim Import.",
                  "Ich möchte gern wissen, wie ich am besten einen Brief an meinen Vermieter formuliere, der höflich bleibt."]:
            self.assertEqual(heuristic_complexity(q), "mittel", q)

    def test_komplex(self):
        for q in ["Entwirf einen Rahmen für eine 5-Zoll-Drohne.",
                  "Vergleiche Carbon und Aluminium für einen Rahmen.",
                  "Berechne die Durchbiegung eines Carbonrohrs mit 16 mm Durchmesser.",
                  "Optimiere das Gewicht meines Quadcopters.",
                  "Plane die Entwicklung meines Roboters.",
                  "Analysiere diese Messreihe auf Ausreißer.",
                  "Warum startet mein Python-Skript nicht?",
                  "Was kostet das? Und wie lange dauert es?",
                  "Ich brauche einen Motor, einen passenden Akku, einen Carbon-Rahmen und Python-Code für die Steuerung.",
                  " ".join(["Wort"] * 45)]:
            self.assertEqual(heuristic_complexity(q), "komplex", q)

    def test_only_valid_classes(self):
        for q in ["a", "Motor", "Warum?", "?" * 5, "12345", None]:
            self.assertIn(heuristic_complexity(q), ("einfach", "mittel", "komplex"))  # type: ignore[arg-type]


# ------------------------------------------------------------ Kontext & Beispiele
class ContextBlockTest(unittest.TestCase):
    def test_sections_and_order(self):
        ctx = context_block([mem("Nutzer fliegt 5-Zoll-Quads", "fakt"), mem("Mag Carbon", "praeferenz", 2)],
                            [FakeLesson("Immer Zellenzahl nennen.")], "DJI Mod", extra="Zusatz")
        self.assertIn("Projekt: DJI Mod", ctx)
        self.assertIn("Verbindliche Lektionen aus früherem Feedback (befolgen):", ctx)
        self.assertIn("- Regel (bestätigt 4×): Immer Zellenzahl nennen.", ctx)
        self.assertIn("Erinnerungen über den Nutzer/Projekt:", ctx)
        self.assertIn("- [fakt] Nutzer fliegt 5-Zoll-Quads", ctx)
        self.assertIn("- [praeferenz] Mag Carbon", ctx)
        self.assertIn("Zusatz", ctx)
        # Lektionen stehen vor den Erinnerungen (direkt nach den Regeln in der System-Nachricht)
        self.assertLess(ctx.index("Verbindliche Lektionen"), ctx.index("Erinnerungen über"))

    def test_empty(self):
        self.assertEqual(context_block([], [], None), "")
        self.assertEqual(context_block([], [], None, extra="   "), "")
        self.assertEqual(context_block([], [], "P"), "Projekt: P")

    def test_budgets(self):
        memories = [mem("M" * 500, "fakt", i) for i in range(10)]
        lessons = [FakeLesson("L" * 400) for _ in range(6)]
        ctx = context_block(memories, lessons, "Projekt", extra="E" * 5000)
        self.assertLessEqual(len(ctx), MAX_CONTEXT_CHARS)
        lesson_part = ctx.split("Verbindliche Lektionen aus früherem Feedback (befolgen):\n")[1].split("\n\n")[0]
        self.assertLessEqual(len(lesson_part), agents.MAX_LESSON_CONTEXT_CHARS)
        for line in lesson_part.splitlines():
            self.assertLessEqual(len(line), 300 + 2 + len(agents.CLIP_SUFFIX))
        mem_part = ctx.split("Erinnerungen über den Nutzer/Projekt:\n")[1].split("\n\n")[0]
        self.assertLessEqual(len(mem_part), agents.MAX_MEMORY_CONTEXT_CHARS)

    def test_lesson_without_render_and_odd_memories(self):
        class Plain:
            def __str__(self):
                return "nur str"
        ctx = context_block([mem("  mehr   Leerzeichen  ")], [Plain(), "roher String"], None)
        self.assertIn("- nur str", ctx)
        self.assertIn("- roher String", ctx)
        self.assertIn("- [fakt] mehr Leerzeichen", ctx)

    def test_example_messages(self):
        out = example_messages([("Frage 1", "A" * 1000), ("Frage 2", "kurz"), ("", "leer"), ("x", ""), None,
                                ("nur eins",)])  # type: ignore[list-item]
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user", "assistant"])
        self.assertEqual(out[0]["content"], "Frage 1")
        self.assertTrue(out[1]["content"].endswith("[gekürzt]"))
        self.assertLessEqual(len(out[1]["content"]), MAX_EXAMPLE_CHARS + len(agents.CLIP_SUFFIX))
        self.assertEqual(out[3]["content"], "kurz")
        self.assertEqual(example_messages([]), [])
        self.assertEqual(example_messages(None), [])  # type: ignore[arg-type]


# ------------------------------------------------------------ Prompt-Bauer
class PromptBuildersTest(unittest.TestCase):
    def test_expert_system_identical_across_experts(self):
        ctx = context_block([mem("Fakt")], [FakeLesson("Regel")], "P")
        systems = {}
        for ex in EXPERTS.values():
            msgs = expert_messages(ex, "Frage", ctx, HISTORY)
            lines = msgs[0]["content"].rstrip().splitlines()
            self.assertEqual(lines[-1], f"[OBITO:experte:{ex.id}]")
            systems[ex.id] = "\n".join(lines[:-1])
            self.assertNotIn(ex.system_prompt, msgs[0]["content"])   # Persona nicht im System-Präfix
            user = msgs[-1]["content"]
            self.assertTrue(user.startswith(f"Du antwortest als Experte {ex.id} – {ex.name}: {ex.role}"), user[:80])
            self.assertIn(ex.system_prompt, user)
            self.assertIn("Frage", user)
            self.assertIn("400 Wörter", user)
            self.assertIn("Risiken", user)
        self.assertEqual(len(set(systems.values())), 1)
        prefix = next(iter(systems.values()))
        self.assertTrue(prefix.startswith(BASE_RULES))
        self.assertIn("Projekt: P", prefix)
        self.assertIn("Regel", prefix)

    def test_expert_history_between_system_and_user(self):
        msgs = expert_messages(EXPERTS["BETA"], "Frage", "", HISTORY)
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user"])
        self.assertEqual(msgs[1]["content"], "Erste Frage")
        msgs = expert_messages(EXPERTS["BETA"], "Frage", "", None)
        self.assertEqual(len(msgs), 2)

    def test_history_budget_and_filtering(self):
        history = [{"role": "user", "content": "alt " + "x" * 2000},
                   {"role": "assistant", "content": "a" * 2000},
                   {"role": "system", "content": "ignorieren"},
                   {"role": "user", "content": "   "},
                   {"role": "user", "content": "neu " + "y" * 2000}]
        msgs = fast_messages("Frage", "", history, [])
        mids = msgs[1:-1]
        self.assertTrue(all(m["role"] in ("user", "assistant") for m in mids))
        self.assertTrue(mids[-1]["content"].startswith("neu "))
        self.assertLessEqual(sum(len(m["content"]) for m in mids), MAX_HISTORY_CHARS + len(agents.CLIP_SUFFIX) * 3)
        self.assertFalse(any(m["content"] == "ignorieren" for m in mids))

    def test_fast_messages_layout(self):
        msgs = fast_messages("Wie spät?", "Projekt: P", HISTORY, EXAMPLES, "rechnen(ausdruck) – rechnet")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "assistant", "user"])
        self.assertEqual(msgs[1]["content"], "Beispielfrage")      # Beispiele vor dem Verlauf
        self.assertEqual(msgs[3]["content"], "Erste Frage")
        self.assertEqual(msgs[-1]["content"], "Wie spät?")
        system = msgs[0]["content"]
        self.assertIn(OMEGA.system_prompt, system)
        self.assertIn(BASE_RULES, system)
        self.assertIn("Projekt: P", system)
        self.assertIn("<werkzeug>", system)
        self.assertIn("rechnen(ausdruck) – rechnet", system)
        self.assertNotIn("<werkzeug>", fast_messages("f", "", [], [])[0]["content"])

    def test_routing_messages(self):
        msgs = routing_messages("Welcher Motor?", ["ALPHA", "BETA", "unbekannt"])
        system, user = msgs[0]["content"], msgs[1]["content"]
        self.assertIn("ALPHA", system)
        self.assertIn("BETA", system)
        self.assertNotIn("GAMMA:", system)
        self.assertIn("einfach", system)
        self.assertIn("komplex", system)
        self.assertIn("Welcher Motor?", user)
        self.assertIn("Antworte nur mit JSON.", user)
        example = json.loads(user.split("Beispiel:\n", 1)[1])
        self.assertEqual(set(example), set(ROUTING_SCHEMA["properties"]))
        # ohne IDs: alle Experten gelistet
        self.assertIn("LAMBDA", routing_messages("x", None)[0]["content"])

    def test_critic_messages(self):
        msgs = critic_messages("Frage", {"ALPHA": "A" * 5000, "XYZ": ""}, "Kontext")
        system, user = msgs[0]["content"], msgs[1]["content"]
        self.assertIn(CRITIC.system_prompt, system)
        self.assertIn("Kontext", system)
        self.assertIn("ALPHA (3D-Konstruktion)", user)
        self.assertIn("XYZ", user)
        self.assertIn("(keine Antwort)", user)
        self.assertIn("[gekürzt]", user)
        self.assertLess(user.count("A" * 100), 30)   # Antwort auf 2500 gekürzt
        self.assertIn("Antworte nur mit JSON.", user)
        example = json.loads(user.split("Beispiel:\n", 1)[1])
        self.assertEqual(set(example), set(CRITIC_SCHEMA["properties"]))
        self.assertIn("schwere", example["fehler"][0])

    def test_revision_messages(self):
        items = [{"experte": "ALPHA", "problem": "P1", "korrektur": "K1", "schwere": "hoch"}, "nackter Text",
                 {"problem": "P2"}]
        msgs = revision_messages(EXPERTS["ALPHA"], "Frage", "alte Antwort", items)
        user = msgs[-1]["content"]
        self.assertTrue(user.startswith("Du antwortest als Experte ALPHA"))
        self.assertIn("alte Antwort", user)
        self.assertIn("[hoch] P1 → Korrektur: K1", user)
        self.assertIn("- nackter Text", user)
        self.assertIn("[mittel] P2", user)
        self.assertIn("vollständige", user)
        self.assertEqual(len(msgs), 2)
        self.assertIn("(keine konkreten Punkte", revision_messages(EXPERTS["ALPHA"], "f", "a", [])[-1]["content"])
        long_items = [{"problem": "x" * 500}] * 10
        self.assertIn("[gekürzt]", revision_messages(EXPERTS["ALPHA"], "f", "a", long_items)[-1]["content"])

    def test_synthesis_messages_deep(self):
        msgs = agents.synthesis_messages("Frage", ANSWERS, CRITIQUE, "Kontext", HISTORY, EXAMPLES, "rechnen()")
        self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant", "user", "assistant", "user"])
        system, user = msgs[0]["content"], msgs[-1]["content"]
        self.assertIn(OMEGA.system_prompt, system)
        self.assertIn("rechnen()", system)
        self.assertIn("ALPHA (3D-Konstruktion)", user)
        self.assertIn("BETA (Elektronik & Steuerung)", user)
        self.assertIn("Bewertung des Kritikers: 6/10", user)
        self.assertIn("[ALPHA, hoch] Zu dünn → 1,2 mm", user)
        self.assertIn("ALPHA vs BETA", user)
        self.assertIn("Temperatur", user)
        self.assertIn("Bereits überarbeitet: ALPHA", user)
        self.assertNotIn("Benenne Widersprüche zwischen den Experten selbst.", user)

    def test_synthesis_messages_medium_and_fallback_critique(self):
        msgs = agents.synthesis_messages("Frage", ANSWERS, None, "", [], [], medium=True)
        self.assertIn("Benenne Widersprüche zwischen den Experten selbst.", msgs[-1]["content"])
        self.assertEqual(len(msgs), 2)
        fallback = {"bewertung": None, "fehler": [], "widersprueche": [], "fehlt": [], "sicher": False,
                    "roh": "Der Kritiker schrieb Prosa."}
        user = agents.synthesis_messages("Frage", ANSWERS, fallback, "", [], [])[-1]["content"]
        self.assertIn("Kritik (unstrukturiert):\nDer Kritiker schrieb Prosa.", user)
        user = agents.synthesis_messages("Frage", ANSWERS, None, "", [], [])[-1]["content"]
        self.assertIn("keine verwertbare Kritik", user)
        user = agents.synthesis_messages("Frage", {}, None, "", [], [], medium=True)[-1]["content"]
        self.assertIn("(keine Expertenantworten vorhanden)", user)

    def test_tool_result_message_exact_texts(self):
        self.assertEqual(tool_result_message("rechnen", FakeToolResult(True, "42")),
                         {"role": "user", "content": "Ergebnis von Werkzeug »rechnen«:\n42"})
        self.assertEqual(tool_result_message("rechnen", FakeToolResult(False, "", "Nicht erlaubt: Name")),
                         {"role": "user", "content": "Fehler bei Werkzeug »rechnen«: Nicht erlaubt: Name\n"
                                                     "Antworte ohne dieses Ergebnis oder korrigiere den Aufruf."})
        self.assertEqual(tool_result_message("zeit", FakeToolResult(True, ""))["content"],
                         "Ergebnis von Werkzeug »zeit«:\n")
        self.assertIn("unbekannter Fehler", tool_result_message("x", FakeToolResult(False, "", None))["content"])

    def test_json_stage_prompts_end_with_instruction_and_example(self):
        cases = [
            (routing_messages("f", None), ROUTING_SCHEMA),
            (critic_messages("f", ANSWERS, ""), CRITIC_SCHEMA),
            (memory_extraction_messages("f", "a", None), MEMORY_SCHEMA),
            (lesson_extraction_messages("f", "a", "k", None), LESSON_SCHEMA),
            (judge_messages("f", "e", "a"), JUDGE_SCHEMA),
            (pairwise_judge_messages("f", "e", "a1", "a2"), PAIRWISE_SCHEMA),
        ]
        for msgs, schema in cases:
            user = msgs[-1]["content"]
            self.assertIn("Antworte nur mit JSON.", user)
            tail = user.rsplit("Antworte nur mit JSON.", 1)[1]
            example = json.loads(tail.split("Beispiel:", 1)[1])
            self.assertEqual(set(example), set(schema["properties"]))

    def test_extraction_lesson_correction_judge_contents(self):
        msgs = memory_extraction_messages("Frage F", "Antwort A", "Proj")
        self.assertIn("Projekt: Proj", msgs[-1]["content"])
        self.assertIn("Frage F", msgs[-1]["content"])
        self.assertIn("Antwort A", msgs[-1]["content"])
        for kind in KINDS:
            self.assertIn(kind, msgs[0]["content"])
        self.assertNotIn("Projekt:", memory_extraction_messages("f", "a", None)[-1]["content"])

        msgs = lesson_extraction_messages("F", "A", "Kommentar K", "Korrektur C")
        self.assertIn("Kommentar K", msgs[-1]["content"])
        self.assertIn("Korrektur C", msgs[-1]["content"])
        user = lesson_extraction_messages("F", "A", None, None)[-1]["content"]
        self.assertNotIn("Kommentar des Nutzers", user)
        self.assertNotIn("Korrektur des Nutzers", user)

        msgs = correction_rewrite_messages("F", "Alte Antwort", "Korrektur C")
        self.assertIn("Alte Antwort", msgs[-1]["content"])
        self.assertIn("Korrektur C", msgs[-1]["content"])
        self.assertIn("vollständige", msgs[-1]["content"])
        self.assertNotIn("Antworte nur mit JSON.", msgs[-1]["content"])

        msgs = judge_messages("F", "E", "A")
        for s in ("F", "E", "A", "Erwartete Antwort", "Zu bewertende Antwort"):
            self.assertIn(s, msgs[-1]["content"])
        msgs = pairwise_judge_messages("F", "E", "EINS", "ZWEI")
        self.assertIn("Antwort 1:\nEINS", msgs[-1]["content"])
        self.assertIn("Antwort 2:\nZWEI", msgs[-1]["content"])

    def test_long_inputs_are_clipped(self):
        big = "B" * 20000
        for msgs in (memory_extraction_messages(big, big, None), lesson_extraction_messages(big, big, big, big),
                     correction_rewrite_messages(big, big, big), judge_messages(big, big, big),
                     pairwise_judge_messages(big, big, big, big), critic_messages(big, {"ALPHA": big}, ""),
                     revision_messages(EXPERTS["ALPHA"], big, big, [{"problem": big}])):
            self.assertLess(len(msgs[-1]["content"]), 20000)


# ------------------------------------------------------------ Schemata
class SchemaTest(unittest.TestCase):
    def test_schemas_are_object_schemas(self):
        for schema in (ROUTING_SCHEMA, CRITIC_SCHEMA, MEMORY_SCHEMA, LESSON_SCHEMA, JUDGE_SCHEMA, PAIRWISE_SCHEMA):
            self.assertEqual(schema["type"], "object")
            self.assertIsInstance(schema["properties"], dict)
            self.assertEqual(set(schema["required"]), set(schema["properties"]))
            json.dumps(schema)   # serialisierbar
        self.assertEqual(set(ROUTING_SCHEMA["properties"]), {"komplexitaet", "experten", "werkzeuge", "begruendung"})
        self.assertEqual(ROUTING_SCHEMA["properties"]["komplexitaet"]["enum"], ["einfach", "mittel", "komplex"])
        self.assertEqual(set(CRITIC_SCHEMA["properties"]), {"bewertung", "fehler", "widersprueche", "fehlt", "sicher"})
        self.assertEqual(set(CRITIC_SCHEMA["properties"]["fehler"]["items"]["properties"]),
                         {"experte", "problem", "korrektur", "schwere"})
        self.assertEqual(CRITIC_SCHEMA["properties"]["fehler"]["items"]["properties"]["schwere"]["enum"],
                         ["hoch", "mittel", "niedrig"])
        self.assertEqual(set(MEMORY_SCHEMA["properties"]["erinnerungen"]["items"]["properties"]),
                         {"inhalt", "art", "wichtigkeit", "tags"})
        self.assertEqual(MEMORY_SCHEMA["properties"]["erinnerungen"]["items"]["properties"]["art"]["enum"], list(KINDS))
        self.assertEqual(set(LESSON_SCHEMA["properties"]), {"regel", "gilt_fuer", "allgemein"})
        self.assertEqual(set(JUDGE_SCHEMA["properties"]), {"punkte", "begruendung"})
        self.assertEqual(set(PAIRWISE_SCHEMA["properties"]), {"besser", "begruendung"})
        self.assertEqual(PAIRWISE_SCHEMA["properties"]["besser"]["enum"], [0, 1, 2])


# ------------------------------------------------------------ Normalisierer
GARBAGE = ["", "   ", None, "kein json", "{kaputt: }", "[]", "{}", "null", "true", '"string"', "[1, 2, 3]",
           '{"foo": "bar"}', b"\xff\xfe", 12.5, [], {}]


class ParseRoutingTest(unittest.TestCase):
    KNOWN = list(EXPERTS)

    def test_clean(self):
        r = parse_routing('{"komplexitaet": "mittel", "experten": ["ALPHA", "BETA"], "werkzeuge": false, '
                          '"begruendung": "ok"}', self.KNOWN)
        self.assertEqual(r, {"komplexitaet": "mittel", "experten": ["ALPHA", "BETA"], "werkzeuge": False,
                             "begruendung": "ok"})

    def test_tolerant(self):
        r = parse_routing('Hier mein Routing:\n```json\n{"Komplexität": "KOMPLEX", "Experten": "alpha, Code & Software, '
                          'kritiker, OMEGA, Experte GAMMA, nix", "Werkzeuge": "ja", "Begründung": 42}\n```', self.KNOWN)
        self.assertEqual(r["komplexitaet"], "komplex")
        self.assertEqual(r["experten"], ["ALPHA", "DELTA", "GAMMA"])
        self.assertIs(r["werkzeuge"], True)
        self.assertEqual(r["begruendung"], "42")

    def test_level_synonyms_and_numbers(self):
        for raw, want in [("Einfach", "einfach"), ("simple", "einfach"), ("schwierig", "komplex"), ("hoch", "komplex"),
                          ("medium", "mittel"), (1, "einfach"), (2, "mittel"), (3, "komplex"), ("3", "komplex"),
                          ("sehr komplex", "komplex"), ("einfach.", "einfach")]:
            r = parse_routing(json.dumps({"komplexitaet": raw, "experten": []}), self.KNOWN)
            self.assertIsNotNone(r, raw)
            self.assertEqual(r["komplexitaet"], want, raw)

    def test_unknown_experts_dropped_and_known_ids_restrict(self):
        r = parse_routing('{"komplexitaet": "mittel", "experten": ["ALPHA", "ZETA", 7, null, {"id": "BETA"}]}',
                          ["ALPHA", "BETA"])
        self.assertEqual(r["experten"], ["ALPHA", "BETA"])
        r = parse_routing('{"komplexitaet": "mittel", "experten": [["ALPHA"]]}', self.KNOWN)
        self.assertEqual(r["experten"], [])

    def test_defaults(self):
        r = parse_routing('{"komplexitaet": "einfach"}', self.KNOWN)
        self.assertEqual(r, {"komplexitaet": "einfach", "experten": [], "werkzeuge": False, "begruendung": ""})
        self.assertTrue(parse_routing('{"komplexitaet": "einfach", "werkzeuge": 1}', self.KNOWN)["werkzeuge"])
        self.assertFalse(parse_routing('{"komplexitaet": "einfach", "werkzeuge": "nein"}', self.KNOWN)["werkzeuge"])
        self.assertTrue(parse_routing('{"komplexitaet": "einfach", "werkzeuge": "true"}', self.KNOWN)["werkzeuge"])

    def test_accepts_dict_and_list_wrapper(self):
        self.assertEqual(parse_routing({"komplexitaet": "mittel"}, self.KNOWN)["komplexitaet"], "mittel")
        self.assertEqual(parse_routing('[{"komplexitaet": "mittel"}]', self.KNOWN)["komplexitaet"], "mittel")

    def test_garbage(self):
        for g in GARBAGE + ['{"komplexitaet": "egal"}', '{"komplexitaet": null}', '{"experten": ["ALPHA"]}']:
            self.assertIsNone(parse_routing(g, self.KNOWN), repr(g))


class ParseCritiqueTest(unittest.TestCase):
    IDS = ["ALPHA", "BETA", "GAMMA"]

    def test_clean_and_missing_keys_filled(self):
        r = parse_critique('{"bewertung": 8, "fehler": [{"experte": "ALPHA", "problem": "p", "korrektur": "k", '
                           '"schwere": "hoch"}]}', self.IDS)
        self.assertEqual(r, {"bewertung": 8,
                             "fehler": [{"experte": "ALPHA", "problem": "p", "korrektur": "k", "schwere": "hoch"}],
                             "widersprueche": [], "fehlt": [], "sicher": False})
        self.assertEqual(set(r), set(CRITIC_SCHEMA["properties"]))

    def test_tolerant(self):
        text = ('{"Bewertung": "7 von 10", "Fehler": [{"Experte": "beta", "Problem": "x", "Schwere": "HOCH!"}, '
                '{"experte": "Physik & Simulation", "problem": "y", "schwere": "gering"}, '
                '{"experte": "UNBEKANNT", "problem": "z", "schwere": "irgendwas"}, {"experte": "ALPHA"}, "ALPHA: nackt", 42],'
                ' "Widersprüche": "nur einer", "fehlt": ["a", "", "a", 3], "sicher": "Ja"}')
        r = parse_critique(text, self.IDS)
        self.assertEqual(r["bewertung"], 7)
        self.assertEqual([f["experte"] for f in r["fehler"]], ["BETA", "GAMMA", None, "ALPHA", None])
        self.assertEqual([f["schwere"] for f in r["fehler"]], ["hoch", "niedrig", "mittel", "mittel", "mittel"])
        self.assertEqual(r["fehler"][3]["problem"], "nackt")
        self.assertEqual(r["fehler"][4]["problem"], "42")
        self.assertEqual(r["widersprueche"], ["nur einer"])
        self.assertEqual(r["fehlt"], ["a", "3"])
        self.assertIs(r["sicher"], True)

    def test_rating_clamped_and_optional(self):
        self.assertEqual(parse_critique('{"bewertung": 15, "fehler": []}', self.IDS)["bewertung"], 10)
        self.assertEqual(parse_critique('{"bewertung": -3, "fehler": []}', self.IDS)["bewertung"], 1)
        self.assertEqual(parse_critique('{"bewertung": "8,4"}', self.IDS)["bewertung"], 8)
        self.assertIsNone(parse_critique('{"fehler": [], "sicher": true}', self.IDS)["bewertung"])
        self.assertIsNone(parse_critique('{"bewertung": "unbekannt"}', self.IDS)["bewertung"])

    def test_fehler_as_single_object_or_string(self):
        r = parse_critique('{"fehler": {"experte": "ALPHA", "problem": "p"}}', self.IDS)
        self.assertEqual(r["fehler"][0]["experte"], "ALPHA")
        self.assertEqual(r["fehler"][0]["korrektur"], "")
        r = parse_critique('{"fehler": "GAMMA – Einheit fehlt"}', self.IDS)
        self.assertEqual(r["fehler"], [{"experte": "GAMMA", "problem": "Einheit fehlt", "korrektur": "", "schwere": "mittel"}])

    def test_garbage(self):
        for g in GARBAGE + ['{"experte": "ALPHA"}', '[{"a": 1}, {"b": 2}]']:
            self.assertIsNone(parse_critique(g, self.IDS), repr(g))


class ParseMemoriesTest(unittest.TestCase):
    def test_valid_entries_only(self):
        text = ('{"erinnerungen": [{"inhalt": "Nutzer fliegt 5-Zoll-Quads mit 4S", "art": "Fakt", "wichtigkeit": "0,8", '
                '"tags": "Drohne"}, {"inhalt": "", "art": "fakt"}, {"art": "fakt"}, "nur text", 5, null, '
                '{"inhalt": "Mag Carbon", "art": "Vorliebe", "wichtigkeit": 3, "tags": ["A", "b", "A"]}, '
                '{"inhalt": "Mag Carbon"}, {"inhalt": "Fehler beim Druck: Warping", "art": "fehler", "wichtigkeit": -1}]}')
        r = parse_memories(text)
        self.assertEqual(r, [
            {"inhalt": "Nutzer fliegt 5-Zoll-Quads mit 4S", "art": "fakt", "wichtigkeit": 0.8, "tags": ["drohne"]},
            {"inhalt": "Mag Carbon", "art": "notiz", "wichtigkeit": 1.0, "tags": ["a", "b"]},
            {"inhalt": "Fehler beim Druck: Warping", "art": "fehler", "wichtigkeit": 0.0, "tags": []},
        ])
        for e in r:
            self.assertIn(e["art"], KINDS)

    def test_alternative_shapes(self):
        self.assertEqual(parse_memories('[{"inhalt": "x y z", "art": "praeferenz"}]')[0]["art"], "praeferenz")
        self.assertEqual(parse_memories('{"inhalt": "einzeln", "art": "notiz"}')[0]["inhalt"], "einzeln")
        self.assertEqual(parse_memories('{"Erinnerungen": {"inhalt": "eins"}}')[0]["inhalt"], "eins")
        self.assertEqual(parse_memories('{"memories": [{"content": "englisch", "kind": "fakt"}]}')[0]["art"], "fakt")
        self.assertEqual(parse_memories('{"erinnerungen": [{"inhalt": "x", "art": "Präferenz"}]}')[0]["art"], "praeferenz")
        self.assertEqual(parse_memories('{"erinnerungen": [{"inhalt": "x"}]}')[0]["wichtigkeit"], 0.5)

    def test_garbage_gives_empty_list(self):
        for g in GARBAGE + ['{"erinnerungen": "text"}', '{"erinnerungen": null}', '{"erinnerungen": [1, "a", []]}']:
            self.assertEqual(parse_memories(g), [], repr(g))


class ParseLessonTest(unittest.TestCase):
    def test_clean(self):
        self.assertEqual(parse_lesson('{"regel": "Einheiten nennen.", "gilt_fuer": ["akku"], "allgemein": false}'),
                         {"regel": "Einheiten nennen.", "gilt_fuer": ["akku"], "allgemein": False})

    def test_tolerant(self):
        r = parse_lesson('```json\n{"Regel": "  Immer   Zellenzahl nennen ", "Gilt für": "Akku", "Allgemein": "ja"}\n```')
        self.assertEqual(r, {"regel": "Immer Zellenzahl nennen", "gilt_fuer": ["akku"], "allgemein": True})
        r = parse_lesson('{"rule": "x", "topics": ["A", "B"], "general": 0}')
        self.assertEqual(r, {"regel": "x", "gilt_fuer": ["a", "b"], "allgemein": False})
        r = parse_lesson('{"regel": ["Teil 1.", "Teil 2."]}')
        self.assertEqual(r["regel"], "Teil 1. Teil 2.")
        self.assertEqual(r["gilt_fuer"], [])
        self.assertLessEqual(len(parse_lesson(json.dumps({"regel": "r" * 1000}))["regel"]), 300 + len(agents.CLIP_SUFFIX))

    def test_garbage(self):
        for g in GARBAGE + ['{"regel": ""}', '{"regel": null, "gilt_fuer": ["x"]}', '{"gilt_fuer": ["x"]}']:
            self.assertIsNone(parse_lesson(g), repr(g))


class ParseJudgeTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_judge('{"punkte": 8, "begruendung": "gut"}'), {"punkte": 8, "begruendung": "gut"})
        self.assertEqual(parse_judge('{"Punkte": "7/10"}'), {"punkte": 7, "begruendung": ""})
        self.assertEqual(parse_judge('{"punkte": 12}')["punkte"], 10)
        self.assertEqual(parse_judge('{"punkte": -1}')["punkte"], 0)
        self.assertEqual(parse_judge('{"punkte": 6.6}')["punkte"], 7)
        self.assertEqual(parse_judge('{"score": "9", "reason": "x"}'), {"punkte": 9, "begruendung": "x"})
        self.assertEqual(parse_judge("7"), {"punkte": 7, "begruendung": ""})
        self.assertEqual(parse_judge('Bewertung: {"punkte": 5, "begruendung": "ok"} Ende')["punkte"], 5)

    def test_garbage(self):
        for g in ["", None, "kein json", "{}", '{"punkte": "keine"}', '{"begruendung": "x"}', "[]", "true"]:
            self.assertIsNone(parse_judge(g), repr(g))


class ParsePairwiseTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_pairwise('{"besser": 1, "begruendung": "a"}'), {"besser": 1, "begruendung": "a"})
        for raw, want in [("2", 2), ("Antwort 2", 2), ("Antwort 1", 1), ("gleich", 0), ("unentschieden", 0),
                          (0, 0), (2.0, 2), ("A", 1), ("B", 2), ("beide", 0), ("die erste", 1), ("zweite", 2)]:
            r = parse_pairwise(json.dumps({"besser": raw}))
            self.assertIsNotNone(r, raw)
            self.assertEqual(r["besser"], want, raw)
        self.assertEqual(parse_pairwise('{"Besser": "2", "Begründung": "x"}'), {"besser": 2, "begruendung": "x"})
        self.assertEqual(parse_pairwise("1"), {"besser": 1, "begruendung": ""})

    def test_garbage(self):
        for g in ["", None, "{}", '{"besser": 3}', '{"besser": "egal"}', '{"besser": true}', "kein json", "[]",
                  '{"begruendung": "x"}']:
            self.assertIsNone(parse_pairwise(g), repr(g))


class ResolveExpertIdTest(unittest.TestCase):
    def test_resolution(self):
        self.assertEqual(agents.resolve_expert_id("alpha"), "ALPHA")
        self.assertEqual(agents.resolve_expert_id(" Experte DELTA "), "DELTA")
        self.assertEqual(agents.resolve_expert_id("Datenanalyse"), "EPSILON")
        self.assertEqual(agents.resolve_expert_id("3d-konstruktion"), "ALPHA")
        self.assertEqual(agents.resolve_expert_id("ALPHA", ["BETA"]), None)
        self.assertIsNone(agents.resolve_expert_id("KRITIKER"))
        self.assertIsNone(agents.resolve_expert_id(""))
        self.assertIsNone(agents.resolve_expert_id(None))
        self.assertIsNone(agents.resolve_expert_id(5))
        self.assertIsNone(agents.resolve_expert_id(["ALPHA"]))
        self.assertIsNone(agents.resolve_expert_id("x"))


if __name__ == "__main__":
    unittest.main()
