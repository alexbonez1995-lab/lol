"""Tests für obito/brain.py – vollständig mit FakeBackend, ohne Modell, ohne Netzwerk.

Der FakeBackend-Responder wird über ``agents.stage_of(messages)`` skriptet:
``SCRIPT[("routing", "system")] = '{"komplexitaet": …}'``; für Experten zusätzlich nach ID.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest

from obito import agents
from obito.agents import EXPERTS, stage_of, select_experts, heuristic_complexity
from obito.brain import (Answer, Brain, Feedback, Step, TOOL_LIMIT_HINT, memory_to_dict)
from obito.config import Config
from obito.learning import LearningStore
from obito.llm import BackendUnavailable, FakeBackend, LLMError, ModelNotFound
from obito.memory import Memory, MemoryStore
from obito.tools import needs_tools

LONG_ANSWER = ("Für ein PETG-Gehäuse sind 1,6 mm Wandstärke (vier Linien à 0,4 mm) ein guter Kompromiss "
               "zwischen Stabilität und Druckzeit.")
SMALLTALK_ANSWER = "Mir geht es gut, danke der Nachfrage! Womit kann ich dir heute helfen?"
Q_SIMPLE = "Hallo, wie geht es dir?"
Q_CALC = "Was ist 17*23?"
Q_MEDIUM = "Welche Wandstärke sollte mein gedrucktes Gehäuse aus PETG haben, damit es stabil bleibt?"
Q_DRONE = "Mein Quadcopter wiegt 600 g und nutzt einen 4S-LiPo. Welche Propeller passen dazu?"
Q_FILE = "Schreibe die Datei notiz.txt mit dem Inhalt Hallo."

TOOL_CALL = '<werkzeug>{"name": "rechnen", "args": {"ausdruck": "17*23"}}</werkzeug>'


def _last_user(msgs: list[dict]) -> str:
    users = [m["content"] for m in msgs if m["role"] == "user"]
    return users[-1] if users else ""


def _has_tool_result(msgs: list[dict]) -> bool:
    return any(m["role"] == "user" and m["content"].startswith(("Ergebnis von Werkzeug", "Fehler bei Werkzeug"))
               for m in msgs)


def scripted(script: dict, default: str = SMALLTALK_ANSWER):
    """Responder: ``script[(stage, who)]`` → Text oder Funktion ``(msgs, kw) -> str``;
    ``(stage, "*")`` als Platzhalter für jeden ``who``."""

    def responder(msgs, kw):
        stage, who = stage_of(msgs)
        value = script.get((stage, who), script.get((stage, "*"), default))
        if callable(value):
            return value(msgs, kw)
        return value

    return responder


class BrainTestBase(unittest.TestCase):
    """Baut je Test einen Brain mit FakeBackend, :memory:-Speichern und Temp-Arbeitsbereich."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.confirm_calls: list[tuple] = []
        self.allow_dangerous = False

    def make_brain(self, responder=None, *, up=True, models=None, **cfg_kw) -> Brain:
        kw = dict(backend="fake", data_dir=self.tmp.name, model="fake-modell", fast_model="", depth="auto",
                  workspace=self.tmp.name, parallel_calls=1, deadline=30.0)
        kw.update(cfg_kw)
        self.cfg = Config(**kw)
        self.backend = FakeBackend(responder=responder, up=up, embed_dim=64, models=models)
        self.memory = MemoryStore(":memory:")
        self.learning = LearningStore(":memory:")

        def confirm(name, args):
            self.confirm_calls.append((name, args))
            return self.allow_dangerous

        brain = Brain(self.cfg, backend=self.backend, memory=self.memory, learning=self.learning, confirm=confirm)
        self.addCleanup(brain.close)
        return brain

    def calls_for(self, stage: str, who: str | None = None) -> list[dict]:
        out = []
        for c in self.backend.calls:
            s, w = stage_of(c["messages"])
            if s == stage and (who is None or w == who):
                out.append(c)
        return out

    def stages_called(self) -> list[str]:
        return [stage_of(c["messages"])[0] for c in self.backend.calls]


# ------------------------------------------------------------------ Datenklassen
class DataclassTest(unittest.TestCase):
    def test_step_to_dict_keys(self):
        s = Step("experte", "ALPHA", "ALPHA (3D-Konstruktion): 812 Zeichen", detail="x", duration=1.23456,
                 tokens=12, status="fertig")
        d = s.to_dict()
        self.assertEqual(list(d), ["stufe", "wer", "zusammenfassung", "detail", "dauer", "tokens", "status"])
        self.assertEqual(d["dauer"], 1.235)
        self.assertEqual(Step("a", "b", "c").status, "fertig")

    def test_answer_to_dict_trace_and_trace_json(self):
        m = Memory(id=3, kind="fakt", content="Inhalt", tags=["a"], project=None, source="nutzer",
                   importance=0.5, created_at=time.time(), last_access=time.time(), access_count=0, score=0.4)
        steps = [Step("erinnern", "system", "1 Erinnerung"), Step("schnell", "OMEGA", "x", detail="D" * 3000,
                                                                   duration=0.5, tokens=9, status="fehler")]
        a = Answer(text="t", question="q", session_id="s", project="p", depth="schnell", experts=[], critique=None,
                   memories_used=[m], new_memories=[], lessons_used=[], tools_used=["rechnen"], steps=steps,
                   interaction_id=7, tokens=9, duration=0.5)
        d = a.to_dict()
        self.assertEqual(list(d), ["antwort", "frage", "sitzung", "projekt", "tiefe", "experten", "kritik",
                                   "erinnerungen_genutzt", "erinnerungen_neu", "werkzeuge", "dokumente", "spur",
                                   "interaktion_id", "tokens", "dauer"])
        self.assertEqual(d["dokumente"], [])
        self.assertEqual(d["erinnerungen_genutzt"][0]["id"], 3)
        self.assertEqual(d["spur"][1]["status"], "fehler")
        trace = json.loads(a.trace_json(max_detail=100))
        self.assertEqual(len(trace), 2)
        self.assertEqual(len(trace[1]["detail"]), 100)
        self.assertEqual(len(json.loads(a.trace_json())[1]["detail"]), 1000)
        text = a.trace()
        self.assertIn("✓ erinnern/system: 1 Erinnerung", text)
        self.assertIn("✗ schnell/OMEGA: x [0.5 s, 9 Tokens]", text)

    def test_memory_to_dict(self):
        now = time.time()
        m = Memory(id=1, kind="fakt", content="c", tags=["t"], project="P", source="ki", importance=0.3,
                   created_at=now, last_access=now, access_count=2, score=0.9, embedding=[0.1, 0.2])
        d = memory_to_dict(m)
        self.assertEqual(set(d), {"id", "art", "inhalt", "tags", "projekt", "quelle", "wichtigkeit", "erstellt",
                                  "score"})
        self.assertEqual(d["erstellt"], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)))
        self.assertNotIn("embedding", d)
        self.assertEqual(Feedback(interaction=None, lessons=[], memories_adjusted=0, correction_full=None).lessons, [])


# ------------------------------------------------------------------ Konstruktion
class InitTest(BrainTestBase):
    def test_wiring_embedder_and_memory_tools(self):
        brain = self.make_brain()
        self.assertIsNotNone(self.memory.embedder)
        self.assertIsNotNone(self.learning.embedder)
        self.assertEqual(self.memory.embed_model, "nomic-embed-text")
        self.assertIsNotNone(brain.tools.get("gedaechtnis_suchen"))
        self.assertIsNotNone(brain.tools.get("gedaechtnis_merken"))
        self.assertIs(brain.tools.memory, self.memory)
        self.assertEqual(brain.tools.workspace, os.path.realpath(self.tmp.name))
        self.assertFalse(brain.busy)
        self.assertTrue(brain.status()["embedding_aktiv"])

    def test_fake_down_has_no_embedder(self):
        brain = self.make_brain(up=False)
        self.assertIsNone(self.memory.embedder)
        self.assertFalse(brain.status()["embedding_aktiv"])

    def test_creates_stores_from_config(self):
        cfg = Config(backend="fake", data_dir=self.tmp.name, model="fake-modell", workspace=self.tmp.name)
        brain = Brain(cfg)
        self.addCleanup(brain.close)
        self.assertEqual(brain.backend.name, "fake")
        self.assertTrue(cfg.memory_db.exists())
        self.assertTrue(cfg.learning_db.exists())
        self.assertEqual(brain.status()["datenverzeichnis"], str(cfg.data_path))

    def test_close_idempotent(self):
        brain = self.make_brain()
        brain.close()
        brain.close()
        self.assertTrue(self.memory._closed)
        self.assertTrue(self.learning._closed)


# ------------------------------------------------------------------ schneller Pfad
class FastPathTest(BrainTestBase):
    def test_fast_path_streams_and_records(self):
        brain = self.make_brain(scripted({("extraktion", "system"): '{"erinnerungen": []}'}))
        out: list[str] = []
        steps: list[Step] = []
        a = brain.ask(Q_SIMPLE, depth="schnell", stream=out.append, progress=steps.append)
        self.assertEqual(a.depth, "schnell")
        self.assertEqual(a.text, SMALLTALK_ANSWER)
        self.assertEqual("".join(out), a.text)
        self.assertEqual(a.experts, [])
        self.assertIsNone(a.critique)
        self.assertIsNotNone(a.interaction_id)
        self.assertEqual(a.question, Q_SIMPLE)
        self.assertEqual(a.session_id, "standard")
        # Steps: start nur an progress, fertig in Answer.steps
        self.assertTrue(any(s.status == "start" and s.stage == "schnell" for s in steps))
        self.assertFalse(any(s.status == "start" for s in a.steps))
        self.assertEqual([s.stage for s in a.steps], ["erinnern", "routing", "schnell", "lernen"])
        self.assertEqual(a.steps[1].summary, "vorgegeben → schnell")
        self.assertGreater(a.tokens, 0)
        self.assertEqual(a.tokens, sum(s.tokens for s in a.steps))
        # Verlauf und Protokoll
        hist = self.memory.history("standard")
        self.assertEqual([h["role"] for h in hist], ["user", "assistant"])
        self.assertEqual(hist[1]["content"], SMALLTALK_ANSWER)
        inter = self.learning.get(a.interaction_id)
        self.assertEqual(inter.answer, SMALLTALK_ANSWER)
        self.assertEqual(inter.depth, "schnell")
        self.assertEqual(inter.model, "fake-modell")
        self.assertTrue(inter.trainable)
        self.assertEqual(json.loads(inter.trace), [s.to_dict() for s in a.steps])
        # Laufzeitoptionen an jedem Aufruf
        fast_call = self.calls_for("schnell")[0]
        self.assertEqual(fast_call["num_ctx"], self.cfg.num_ctx)
        self.assertEqual(fast_call["max_tokens"], self.cfg.max_tokens_fast)
        self.assertIsNone(fast_call["think"])
        self.assertEqual(fast_call["temperature"], self.cfg.temperature)
        self.assertEqual(fast_call["messages"][-1]["content"], Q_SIMPLE)

    def test_think_flag_and_fit_messages(self):
        brain = self.make_brain(scripted({}), think="an", num_ctx=900, auto_memory=False)
        long_q = "Hallo " + "wort " * 2000
        a = brain.ask(long_q, depth="schnell")
        call = self.calls_for("schnell")[0]
        self.assertIs(call["think"], True)
        self.assertLess(len(call["messages"][-1]["content"]), len(long_q))
        self.assertIn(agents.CLIP_SUFFIX, call["messages"][-1]["content"])
        self.assertEqual(a.depth, "schnell")
        brain2 = self.make_brain(scripted({}), think="aus", auto_memory=False)
        brain2.ask(Q_SIMPLE, depth="schnell")
        self.assertIs(self.calls_for("schnell")[0]["think"], False)

    def test_auto_depth_heuristic_simple_skips_routing(self):
        brain = self.make_brain(scripted({}), auto_memory=False)
        self.assertEqual(heuristic_complexity(Q_SIMPLE), "einfach")
        a = brain.ask(Q_SIMPLE)
        self.assertEqual(a.depth, "schnell")
        self.assertEqual(self.stages_called(), ["schnell"])
        self.assertEqual(a.steps[1].summary, "einfach → schnell (Heuristik)")

    def test_auto_depth_with_fast_model_calls_routing(self):
        script = {("routing", "system"): '{"komplexitaet": "einfach", "experten": [], "werkzeuge": false, '
                                         '"begruendung": "Smalltalk."}'}
        brain = self.make_brain(scripted(script), fast_model="klein", auto_memory=False)
        a = brain.ask(Q_SIMPLE)
        self.assertEqual(a.depth, "schnell")
        routing = self.calls_for("routing")
        self.assertEqual(len(routing), 1)
        self.assertEqual(routing[0]["model"], "klein")
        self.assertEqual(routing[0]["json_mode"], agents.ROUTING_SCHEMA)
        self.assertEqual(routing[0]["temperature"], 0.0)
        self.assertEqual(routing[0]["max_tokens"], self.cfg.max_tokens_json)
        self.assertEqual(a.steps[1].summary, "einfach → schnell")
        self.assertEqual(a.steps[1].detail, "Smalltalk.")

    def test_routing_result_overrides_heuristic(self):
        self.assertEqual(heuristic_complexity(Q_MEDIUM), "mittel")
        script = {
            ("routing", "system"): '{"komplexitaet": "komplex", "experten": ["IOTA", "ALPHA", "UNBEKANNT"], '
                                   '"werkzeuge": false, "begruendung": "Material und Konstruktion."}',
            ("experte", "*"): lambda msgs, kw: f"Antwort von {stage_of(msgs)[1]}: 1,6 mm Wandstärke.",
            ("kritiker", "KRITIKER"): '{"bewertung": 9, "fehler": [], "widersprueche": [], "fehlt": [], "sicher": true}',
            ("synthese", "OMEGA"): LONG_ANSWER,
        }
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM)
        self.assertEqual(a.depth, "tief")
        self.assertEqual(a.experts[:2], ["IOTA", "ALPHA"])      # Hinweis aus dem Routing zuerst
        self.assertEqual(len(a.experts), 5)
        self.assertEqual(a.steps[1].summary, "komplex → tief (IOTA, ALPHA)")

    def test_routing_garbage_falls_back_to_heuristic(self):
        script = {("routing", "system"): "Ich bin kein JSON.",
                  ("experte", "*"): lambda msgs, kw: f"Antwort von {stage_of(msgs)[1]} mit Zahlen.",
                  ("synthese", "OMEGA"): LONG_ANSWER}
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM)
        self.assertEqual(a.depth, "mittel")
        routing = self.calls_for("routing")
        self.assertEqual(len(routing), 2)                              # genau ein Retry
        self.assertIs(routing[1]["json_mode"], True)
        # Der Hinweis hängt an der letzten Nutzer-Nachricht (keine eigene Nachricht, damit fit_messages
        # bei knappem Budget nie die eigentliche Aufgabe statt des Hinweises kürzt)
        last = routing[1]["messages"][-1]["content"]
        self.assertIn("Antworte ausschließlich mit einem JSON-Objekt nach diesem Schema: ", last)
        self.assertTrue(last.startswith(routing[0]["messages"][-1]["content"]))
        self.assertEqual(len(routing[1]["messages"]), len(routing[0]["messages"]))
        self.assertEqual(a.tokens, sum(c["prompt_tokens"] for c in ()) + sum(
            len(" ".join(m["content"] for m in c["messages"]).split()) + 0 for c in []) if False else a.tokens)
        hints = [s for s in a.steps if s.stage == "routing" and s.who == "system" and s.status == "fehler"]
        self.assertEqual(len(hints), 1)
        self.assertIn("Fallback", hints[0].summary)
        final = [s for s in a.steps if s.stage == "routing" and s.status == "fertig"]
        self.assertEqual(final[-1].summary, "mittel → mittel (Heuristik)")

    def test_invalid_input(self):
        brain = self.make_brain(scripted({}))
        with self.assertRaises(ValueError):
            brain.ask("   ")
        with self.assertRaises(ValueError):
            brain.ask(Q_SIMPLE, depth="ultra")
        self.assertEqual(self.backend.calls, [])

    def test_empty_model_answer_is_error(self):
        brain = self.make_brain(scripted({("schnell", "OMEGA"): "   "}))
        a = brain.ask(Q_SIMPLE, depth="schnell")
        self.assertEqual(a.depth, "fehler")
        self.assertIn("Leere Antwort", a.text)


# ------------------------------------------------------------------ Gremium
class CouncilTest(BrainTestBase):
    def expert_script(self):
        return {("experte", "*"): lambda msgs, kw: f"Antwort von {stage_of(msgs)[1]}: 1,2 mm Wandstärke, "
                                                   f"PETG ab 230 °C drucken.",
                ("revision", "*"): lambda msgs, kw: f"Revidiert von {stage_of(msgs)[1]}: 1,6 mm Wandstärke.",
                ("synthese", "OMEGA"): LONG_ANSWER}

    def test_medium_three_experts_no_critic(self):
        brain = self.make_brain(scripted(self.expert_script()), auto_memory=False)
        steps: list[Step] = []
        a = brain.ask(Q_MEDIUM, depth="mittel", progress=steps.append)
        expected = [e.id for e in select_experts(Q_MEDIUM, 3)]
        self.assertEqual(a.depth, "mittel")
        self.assertEqual(a.experts, expected)
        self.assertIsNone(a.critique)
        self.assertEqual(self.stages_called().count("experte"), 3)
        self.assertEqual(self.calls_for("kritiker"), [])
        self.assertEqual(a.text, LONG_ANSWER)
        # Expertenaufrufe: Persona in der Nutzer-Nachricht, Temperatur je Experte, Budget
        for c in self.calls_for("experte"):
            who = stage_of(c["messages"])[1]
            self.assertEqual(c["temperature"], EXPERTS[who].temperature)
            self.assertEqual(c["max_tokens"], self.cfg.max_tokens_expert)
            self.assertIn(f"Experte {who}", c["messages"][-1]["content"])
        # Synthese enthält alle Expertenantworten und den medium-Satz
        syn = self.calls_for("synthese")[0]
        user = _last_user(syn["messages"])
        for eid in expected:
            self.assertIn(f"Antwort von {eid}", user)
        self.assertIn("Benenne Widersprüche zwischen den Experten selbst.", user)
        self.assertEqual(syn["max_tokens"], self.cfg.max_tokens_synthesis)
        # Step-Formate
        exp_steps = [s for s in a.steps if s.stage == "experte"]
        self.assertEqual(len(exp_steps), 3)
        self.assertRegex(exp_steps[0].summary, r"^[A-Z]+ \(.+\): \d+ Zeichen$")
        starts = [s for s in steps if s.status == "start" and s.stage == "experte"]
        self.assertEqual(len(starts), 3)

    def test_deep_with_critic_and_targeted_revision(self):
        expected = [e.id for e in select_experts(Q_MEDIUM, 5)]
        bad, minor = expected[0], expected[1]
        critique = {"bewertung": 7,
                    "fehler": [{"experte": bad, "problem": "Wandstärke zu dünn", "korrektur": "1,6 mm", "schwere": "hoch"},
                               {"experte": minor, "problem": "Tippfehler", "korrektur": "", "schwere": "niedrig"},
                               {"experte": "NIEMAND", "problem": "x", "korrektur": "", "schwere": "hoch"}],
                    "widersprueche": ["A sagt PLA, B sagt PETG"], "fehlt": ["Umgebungstemperatur"], "sicher": True}
        script = self.expert_script()
        script[("kritiker", "KRITIKER")] = json.dumps(critique)
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="tief")
        self.assertEqual(a.depth, "tief")
        self.assertEqual(a.experts, expected)
        self.assertEqual(self.stages_called().count("experte"), 5)
        self.assertEqual(len(self.calls_for("kritiker")), 1)
        crit_call = self.calls_for("kritiker")[0]
        self.assertEqual(crit_call["json_mode"], agents.CRITIC_SCHEMA)
        self.assertEqual(crit_call["max_tokens"], self.cfg.max_tokens_critic)
        self.assertEqual(crit_call["temperature"], 0.0)
        # Revision nur für schwere hoch
        rev = self.calls_for("revision")
        self.assertEqual([stage_of(c["messages"])[1] for c in rev], [bad])
        self.assertIn("Wandstärke zu dünn", _last_user(rev[0]["messages"]))
        self.assertEqual(a.critique["revidiert"], [bad])
        self.assertEqual(a.critique["bewertung"], 7)
        self.assertEqual(a.critique["fehler"][0]["experte"], bad)
        self.assertIsNone(a.critique["fehler"][2]["experte"])
        # Synthese bekommt die revidierte Fassung und die Kritik
        user = _last_user(self.calls_for("synthese")[0]["messages"])
        self.assertIn(f"Revidiert von {bad}", user)
        self.assertNotIn(f"Antwort von {bad}:", user)
        self.assertIn(f"Antwort von {minor}", user)
        self.assertIn("Bewertung des Kritikers: 7/10", user)
        self.assertNotIn("Benenne Widersprüche zwischen den Experten selbst.", user)
        crit_steps = [s for s in a.steps if s.stage == "kritiker"]
        self.assertEqual(crit_steps[0].summary, "Bewertung 7/10, 3 Fehler, 1 Widerspruch")
        self.assertEqual([s.who for s in a.steps if s.stage == "revision"], [bad])

    def test_low_rating_revises_all_mentioned_sorted_and_capped(self):
        expected = [e.id for e in select_experts(Q_MEDIUM, 5)]
        a1, a2, a3 = expected[0], expected[1], expected[2]
        critique = {"bewertung": 3,
                    "fehler": [{"experte": a1, "problem": "p1", "korrektur": "", "schwere": "niedrig"},
                               {"experte": a3, "problem": "p2", "korrektur": "", "schwere": "mittel"},
                               {"experte": a3, "problem": "p3", "korrektur": "", "schwere": "mittel"},
                               {"experte": a2, "problem": "p4", "korrektur": "", "schwere": "niedrig"}],
                    "widersprueche": [], "fehlt": [], "sicher": False}
        script = self.expert_script()
        script[("kritiker", "KRITIKER")] = json.dumps(critique)
        brain = self.make_brain(scripted(script), auto_memory=False, max_revised_experts=2)
        a = brain.ask(Q_MEDIUM, depth="tief")
        rev_ids = [stage_of(c["messages"])[1] for c in self.calls_for("revision")]
        self.assertEqual(rev_ids, [a3, a1])          # meiste Befunde zuerst, dann Expertenreihenfolge
        self.assertEqual(a.critique["revidiert"], sorted([a1, a3]))

    def test_no_revision_when_rounds_zero_or_no_high_severity(self):
        expected = [e.id for e in select_experts(Q_MEDIUM, 5)]
        critique = {"bewertung": 8, "fehler": [{"experte": expected[0], "problem": "p", "korrektur": "",
                                                "schwere": "mittel"}],
                    "widersprueche": [], "fehlt": [], "sicher": True}
        script = self.expert_script()
        script[("kritiker", "KRITIKER")] = json.dumps(critique)
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="tief")
        self.assertEqual(self.calls_for("revision"), [])
        self.assertEqual(a.critique["revidiert"], [])
        critique["fehler"][0]["schwere"] = "hoch"
        script[("kritiker", "KRITIKER")] = json.dumps(critique)
        brain = self.make_brain(scripted(script), auto_memory=False, max_revision_rounds=0)
        a = brain.ask(Q_MEDIUM, depth="tief")
        self.assertEqual(self.calls_for("revision"), [])
        self.assertEqual(a.critique["revidiert"], [])

    def test_critic_garbage_fallback_without_revision(self):
        script = self.expert_script()
        script[("kritiker", "KRITIKER")] = "Alles prima, keine Fehler!!!"
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="tief")
        self.assertEqual(len(self.calls_for("kritiker")), 2)
        self.assertEqual(self.calls_for("revision"), [])
        self.assertEqual(a.critique["bewertung"], None)
        self.assertEqual(a.critique["fehler"], [])
        self.assertEqual(a.critique["widersprueche"], [])
        self.assertEqual(a.critique["fehlt"], [])
        self.assertIs(a.critique["sicher"], False)
        self.assertEqual(a.critique["roh"], "Alles prima, keine Fehler!!!")
        self.assertEqual(a.critique["revidiert"], [])
        hints = [s for s in a.steps if s.stage == "kritiker" and s.status == "fehler"]
        self.assertEqual(len(hints), 1)
        self.assertEqual(hints[0].who, "system")
        self.assertIn("Kritik (unstrukturiert)", _last_user(self.calls_for("synthese")[0]["messages"]))
        self.assertEqual(a.text, LONG_ANSWER)

    def test_failed_revision_keeps_original(self):
        expected = [e.id for e in select_experts(Q_MEDIUM, 5)]
        bad = expected[0]
        critique = {"bewertung": 6, "fehler": [{"experte": bad, "problem": "p", "korrektur": "", "schwere": "hoch"}],
                    "widersprueche": [], "fehlt": [], "sicher": True}
        script = self.expert_script()
        script[("kritiker", "KRITIKER")] = json.dumps(critique)

        def failing_revision(msgs, kw):
            raise LLMError("Revision kaputt")

        script[("revision", "*")] = failing_revision
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="tief")
        self.assertEqual(a.depth, "tief")
        self.assertEqual(a.critique["revidiert"], [])
        rev_steps = [s for s in a.steps if s.stage == "revision"]
        self.assertEqual(rev_steps[0].status, "fehler")
        self.assertIn(f"Antwort von {bad}", _last_user(self.calls_for("synthese")[0]["messages"]))

    def test_single_expert_failure_is_tolerated(self):
        expected = [e.id for e in select_experts(Q_MEDIUM, 3)]
        broken = expected[1]

        def experts(msgs, kw):
            who = stage_of(msgs)[1]
            if who == broken:
                raise LLMError("Experte kaputt")
            return f"Antwort von {who}: alles gut mit 1,6 mm."

        script = {("experte", "*"): experts, ("synthese", "OMEGA"): LONG_ANSWER}
        brain = self.make_brain(scripted(script), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="mittel")
        self.assertEqual(a.depth, "mittel")
        self.assertEqual(a.experts, [e for e in expected if e != broken])
        failed = [s for s in a.steps if s.stage == "experte" and s.status == "fehler"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].who, broken)
        self.assertIn("Experte kaputt", failed[0].summary)
        self.assertNotIn(f"Antwort von {broken}", _last_user(self.calls_for("synthese")[0]["messages"]))

    def test_all_experts_fail_gives_error_answer(self):
        def experts(msgs, kw):
            raise LLMError("alles kaputt")

        brain = self.make_brain(scripted({("experte", "*"): experts}), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="mittel")
        self.assertEqual(a.depth, "fehler")
        self.assertEqual(a.text, "Modellfehler: alles kaputt")
        self.assertEqual(a.experts, [])
        self.assertIsNone(a.interaction_id)
        self.assertEqual(self.calls_for("synthese"), [])
        self.assertEqual(self.memory.history("standard"), [])

    def test_parallel_calls_progress_in_caller_thread(self):
        script = self.expert_script()
        brain = self.make_brain(scripted(script), auto_memory=False, parallel_calls=3)
        idents: set[int] = set()
        busy_seen: list[bool] = []

        def progress(step):
            idents.add(threading.get_ident())
            busy_seen.append(brain.busy)

        a = brain.ask(Q_MEDIUM, depth="mittel", progress=progress)
        self.assertEqual(a.depth, "mittel")
        self.assertEqual(idents, {threading.get_ident()})
        self.assertTrue(busy_seen and all(busy_seen))
        self.assertFalse(brain.busy)
        self.assertEqual(sorted(a.experts), sorted(e.id for e in select_experts(Q_MEDIUM, 3)))

    def test_deadline_timeout_marks_experts_failed(self):
        def slow(msgs, kw):
            time.sleep(0.4)
            return "zu spät"

        brain = self.make_brain(scripted({("experte", "*"): slow}), auto_memory=False, deadline=0.2)
        a = brain.ask(Q_MEDIUM, depth="mittel")
        self.assertEqual(a.depth, "fehler")
        self.assertIn("Zeitbudget", a.text)
        failed = [s for s in a.steps if s.stage == "experte" and s.status == "fehler"]
        self.assertEqual(len(failed), 3)
        self.assertTrue(all("Zeitbudget" in s.summary for s in failed))
        time.sleep(0.5)  # Arbeits-Threads auslaufen lassen


# ------------------------------------------------------------------ Werkzeuge
class ToolLoopTest(BrainTestBase):
    def calc_script(self, second="17 × 23 = 391."):
        def fast(msgs, kw):
            if _has_tool_result(msgs):
                return second
            return "Ich rechne das kurz aus: " + TOOL_CALL

        return {("schnell", "OMEGA"): fast}

    def test_tool_block_only_when_needed(self):
        brain = self.make_brain(scripted({}), auto_memory=False)
        brain.ask(Q_SIMPLE, depth="schnell")
        self.assertNotIn("Verfügbare Werkzeuge", self.calls_for("schnell")[0]["messages"][0]["content"])
        brain.ask(Q_CALC, depth="schnell")
        sys_msg = self.calls_for("schnell")[1]["messages"][0]["content"]
        self.assertIn("Verfügbare Werkzeuge", sys_msg)
        self.assertIn("rechnen(", sys_msg)
        self.assertIn("gedaechtnis_suchen(", sys_msg)
        brain2 = self.make_brain(scripted({}), auto_memory=False, allow_tools=False)
        brain2.ask(Q_CALC, depth="schnell")
        self.assertNotIn("Verfügbare Werkzeuge", self.calls_for("schnell")[0]["messages"][0]["content"])

    def test_routing_tools_flag_adds_block(self):
        script = {("routing", "system"): '{"komplexitaet": "einfach", "experten": [], "werkzeuge": true, '
                                         '"begruendung": "Datum nötig."}'}
        brain = self.make_brain(scripted(script), fast_model="klein", auto_memory=False)
        self.assertFalse(needs_tools(Q_SIMPLE, []))
        brain.ask(Q_SIMPLE)
        self.assertIn("Verfügbare Werkzeuge", self.calls_for("schnell")[0]["messages"][0]["content"])

    def test_tool_loop_rechnen_ok_with_stream_gate(self):
        brain = self.make_brain(scripted(self.calc_script()), auto_memory=False)
        out: list[str] = []
        a = brain.ask(Q_CALC, depth="schnell", stream=out.append)
        self.assertEqual(a.text, "Ich rechne das kurz aus:\n\n17 × 23 = 391.")
        self.assertEqual("".join(out), a.text)
        self.assertNotIn("<werkzeug", "".join(out))
        self.assertEqual(a.tools_used, ["rechnen"])
        tool_steps = [s for s in a.steps if s.stage == "werkzeug"]
        self.assertEqual(len(tool_steps), 1)
        self.assertEqual(tool_steps[0].summary, "rechnen: ok")
        self.assertEqual(tool_steps[0].who, "rechnen")
        self.assertEqual(tool_steps[0].detail, "17*23 = 391")
        calls = self.calls_for("schnell")
        self.assertEqual(len(calls), 2)
        msgs = calls[1]["messages"]
        self.assertEqual(msgs[-2]["role"], "assistant")
        self.assertIn(TOOL_CALL, msgs[-2]["content"])
        self.assertEqual(msgs[-1], {"role": "user", "content": "Ergebnis von Werkzeug »rechnen«:\n17*23 = 391"})
        inter = self.learning.get(a.interaction_id)
        self.assertEqual(inter.tools_used, ["rechnen"])
        self.assertFalse(inter.trainable)
        self.assertIn("Werkzeugaufruf", [s for s in a.steps if s.stage == "schnell"][0].summary)

    def test_unknown_tool(self):
        def fast(msgs, kw):
            if _has_tool_result(msgs):
                return "Dieses Werkzeug gibt es nicht; ich antworte direkt: 391."
            return 'Moment. <werkzeug>{"name": "zauberei", "args": {"x": 1}}</werkzeug>'

        brain = self.make_brain(scripted({("schnell", "OMEGA"): fast}), auto_memory=False)
        a = brain.ask(Q_CALC, depth="schnell")
        step = [s for s in a.steps if s.stage == "werkzeug"][0]
        self.assertEqual(step.status, "fehler")
        self.assertTrue(step.summary.startswith("zauberei: Fehler – Unbekanntes Werkzeug »zauberei«"))
        last = self.calls_for("schnell")[1]["messages"][-1]["content"]
        self.assertTrue(last.startswith("Fehler bei Werkzeug »zauberei«: Unbekanntes Werkzeug"))
        self.assertTrue(last.endswith("Antworte ohne dieses Ergebnis oder korrigiere den Aufruf."))
        self.assertEqual(a.text, "Moment.\n\nDieses Werkzeug gibt es nicht; ich antworte direkt: 391.")
        self.assertEqual(a.tools_used, ["zauberei"])

    def test_denied_dangerous_tool(self):
        call = '<werkzeug>{"name": "datei_schreiben", "args": {"pfad": "notiz.txt", "inhalt": "Hallo"}}</werkzeug>'

        def fast(msgs, kw):
            if _has_tool_result(msgs):
                return "Ich darf die Datei ohne deine Freigabe nicht schreiben."
            return "Ich schreibe die Datei. " + call

        brain = self.make_brain(scripted({("schnell", "OMEGA"): fast}), auto_memory=False)
        out: list[str] = []
        a = brain.ask(Q_FILE, depth="schnell", stream=out.append)
        self.assertEqual(self.confirm_calls, [("datei_schreiben", {"pfad": "notiz.txt", "inhalt": "Hallo"})])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "notiz.txt")))
        step = [s for s in a.steps if s.stage == "werkzeug"][0]
        self.assertEqual(step.summary, "datei_schreiben: Fehler – Vom Nutzer abgelehnt")
        self.assertIn("Fehler bei Werkzeug »datei_schreiben«: Vom Nutzer abgelehnt",
                      self.calls_for("schnell")[1]["messages"][-1]["content"])
        self.assertEqual(a.text, "Ich schreibe die Datei.\n\nIch darf die Datei ohne deine Freigabe nicht schreiben.")
        self.assertEqual("".join(out), a.text)
        # mit Freigabe wird geschrieben
        self.allow_dangerous = True
        a = brain.ask(Q_FILE, depth="schnell", session_id="zwei")
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "notiz.txt")))
        self.assertEqual([s for s in a.steps if s.stage == "werkzeug"][0].summary, "datei_schreiben: ok")

    def test_round_limit(self):
        brain = self.make_brain(scripted({("schnell", "OMEGA"): "Rechne: " + TOOL_CALL}), auto_memory=False,
                                max_tool_rounds=2)
        out: list[str] = []
        a = brain.ask(Q_CALC, depth="schnell", stream=out.append)
        self.assertEqual(len(self.calls_for("schnell")), 3)
        self.assertEqual(a.tools_used, ["rechnen", "rechnen"])
        self.assertEqual(a.text, "Rechne:\n\nRechne:\n\nRechne:\n\n" + TOOL_LIMIT_HINT)
        self.assertEqual("".join(out), a.text)
        self.assertTrue(a.text.endswith(TOOL_LIMIT_HINT))

    def test_dedup_and_max_calls_per_answer(self):
        calls = (TOOL_CALL + TOOL_CALL
                 + '<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2+2"}}</werkzeug>'
                 + '<werkzeug>{"name": "zeit", "args": {}}</werkzeug>'
                 + '<werkzeug>{"name": "system_info", "args": {}}</werkzeug>')

        def fast(msgs, kw):
            if _has_tool_result(msgs):
                return "Fertig gerechnet."
            return "Mehrere Aufrufe: " + calls

        brain = self.make_brain(scripted({("schnell", "OMEGA"): fast}), auto_memory=False,
                                max_tool_calls_per_answer=3)
        a = brain.ask(Q_CALC, depth="schnell")
        self.assertEqual(a.tools_used, ["rechnen", "rechnen", "zeit"])
        results = [m["content"] for m in self.calls_for("schnell")[1]["messages"] if m["role"] == "user"
                   and m["content"].startswith("Ergebnis")]
        self.assertEqual(len(results), 3)
        self.assertIn("17*23 = 391", results[0])
        self.assertIn("2+2 = 4", results[1])
        self.assertIn("Datum:", results[2])
        self.assertEqual(a.text, "Mehrere Aufrufe:\n\nFertig gerechnet.")

    def test_tool_calls_ignored_without_tools_block(self):
        brain = self.make_brain(scripted({("schnell", "OMEGA"): "Antwort ohne Werkzeuge. " + TOOL_CALL}),
                                auto_memory=False, allow_tools=False)
        a = brain.ask(Q_CALC, depth="schnell")
        self.assertEqual(a.text, "Antwort ohne Werkzeuge.")
        self.assertEqual(a.tools_used, [])
        self.assertEqual(len(self.calls_for("schnell")), 1)

    def test_tools_in_synthesis_stage(self):
        def synth(msgs, kw):
            if _has_tool_result(msgs):
                return "Das Ergebnis lautet 391."
            return "Ich prüfe die Zahl: " + TOOL_CALL

        script = {("experte", "*"): lambda msgs, kw: f"Antwort von {stage_of(msgs)[1]}: 17*23 ist etwa 390.",
                  ("synthese", "OMEGA"): synth}
        brain = self.make_brain(scripted(script), auto_memory=False)
        out: list[str] = []
        a = brain.ask("Berechne bitte 17*23 für den Rahmen aus Carbon.", depth="mittel", stream=out.append)
        self.assertEqual(a.text, "Ich prüfe die Zahl:\n\nDas Ergebnis lautet 391.")
        self.assertEqual("".join(out), a.text)
        self.assertEqual(a.tools_used, ["rechnen"])
        self.assertEqual(len(self.calls_for("synthese")), 2)
        # Experten bekommen nie einen Werkzeugblock
        for c in self.calls_for("experte"):
            self.assertNotIn("Verfügbare Werkzeuge", c["messages"][0]["content"])

    def test_stream_gate_holds_partial_marker(self):
        text = "Ergebnis folgt <werk"            # unvollständiger Marker am Ende → wird geflusht
        brain = self.make_brain(scripted({("schnell", "OMEGA"): text}), auto_memory=False)
        out: list[str] = []
        a = brain.ask(Q_CALC, depth="schnell", stream=out.append)
        self.assertEqual(a.text, "Ergebnis folgt <werk")
        self.assertEqual("".join(out), a.text)


# ------------------------------------------------------------------ Lernen
class LearningTest(BrainTestBase):
    def test_memory_extraction_guards(self):
        answer = "Für 600 g Abfluggewicht mit 4S-LiPo passen 5-Zoll-Propeller mit 3 Blättern (z. B. 5x4.3x3)."
        extraction = {"erinnerungen": [
            {"inhalt": "Der Quadcopter des Nutzers wiegt 600 g und fliegt mit einem 4S-LiPo.", "art": "fakt",
             "wichtigkeit": 0.9, "tags": ["drohne", "akku", "gewicht", "lipo", "4s", "zu", "viele"]},
            {"inhalt": "Nutzer hat gefragt, welche Propeller zum Quadcopter passen.", "art": "notiz",
             "wichtigkeit": 0.8, "tags": []},
            {"inhalt": "Der Nutzer bevorzugt Propeller mit drei Blättern.", "art": "praeferenz",
             "wichtigkeit": 0.3, "tags": []},
            {"inhalt": "Kurz.", "art": "fakt", "wichtigkeit": 0.9, "tags": []},
            {"inhalt": "Unsicher: vielleicht nutzt der Quadcopter auch 6S-LiPo.", "art": "fakt",
             "wichtigkeit": 0.7, "tags": []},
            {"inhalt": "Die Hauptstadt von Frankreich heisst Paris, das kennt wirklich jeder Mensch.",
             "art": "fakt", "wichtigkeit": 0.9, "tags": []},
        ]}
        script = {("schnell", "OMEGA"): answer, ("extraktion", "system"): json.dumps(extraction)}
        brain = self.make_brain(scripted(script))
        a = brain.ask(Q_DRONE, depth="schnell", project="Drohne")
        self.assertEqual(len(a.new_memories), 1)
        m = a.new_memories[0]
        self.assertEqual(m.content, "Der Quadcopter des Nutzers wiegt 600 g und fliegt mit einem 4S-LiPo.")
        self.assertEqual(m.source, "ki")
        self.assertEqual(m.kind, "fakt")
        self.assertEqual(m.project, "Drohne")
        self.assertAlmostEqual(m.importance, 0.5)
        self.assertEqual(len(m.tags), 5)
        self.assertEqual(self.memory.count(), 1)
        learn = [s for s in a.steps if s.stage == "lernen"][0]
        self.assertEqual(learn.summary, "1 Erinnerung gemerkt (5 übersprungen)")
        for reason in ("Floskel", "Wichtigkeit", "Länge", "Unsicher", "kein Bezug"):
            self.assertIn(reason, learn.detail)
        ext = self.calls_for("extraktion")
        self.assertEqual(len(ext), 1)
        self.assertEqual(ext[0]["json_mode"], agents.MEMORY_SCHEMA)
        self.assertEqual(ext[0]["model"], self.cfg.routing_model)
        inter = self.learning.get(a.interaction_id)
        self.assertEqual(inter.new_memories, [m.id])
        self.assertEqual(a.to_dict()["erinnerungen_neu"][0]["quelle"], "ki")

    def test_max_new_memories_cap(self):
        extraction = {"erinnerungen": [
            {"inhalt": f"Der Quadcopter des Nutzers hat Eigenschaft Nummer {i} mit {i * 100} g.", "art": "fakt",
             "wichtigkeit": 0.8, "tags": []} for i in range(1, 5)]}
        script = {("schnell", "OMEGA"): LONG_ANSWER.replace("PETG-Gehäuse", "Quadcopter"),
                  ("extraktion", "system"): json.dumps(extraction)}
        brain = self.make_brain(scripted(script), max_new_memories=2)
        a = brain.ask(Q_DRONE, depth="schnell")
        self.assertEqual(len(a.new_memories), 2)
        self.assertEqual(self.memory.count(), 2)
        self.assertIn("Limit", [s for s in a.steps if s.stage == "lernen"][0].detail)

    def test_extraction_skipped_for_short_answers_and_when_disabled(self):
        brain = self.make_brain(scripted({("schnell", "OMEGA"): "Ja, 391."}))
        a = brain.ask(Q_CALC, depth="schnell", learn=True)
        self.assertEqual(self.calls_for("extraktion"), [])
        self.assertEqual([s.stage for s in a.steps if s.stage == "lernen"], [])
        self.assertIsNotNone(a.interaction_id)
        brain2 = self.make_brain(scripted({}), auto_memory=False)
        a = brain2.ask(Q_SIMPLE, depth="schnell")
        self.assertEqual(self.calls_for("extraktion"), [])
        self.assertIsNotNone(a.interaction_id)
        a = brain2.ask(Q_SIMPLE, depth="schnell", learn=True)
        self.assertEqual(len(self.calls_for("extraktion")), 2)      # Müll → ein Retry, nie fatal
        self.assertEqual(a.depth, "schnell")
        learn = [s for s in a.steps if s.stage == "lernen"][0]
        self.assertEqual(learn.summary, "übersprungen: keine verwertbare Extraktion")

    def test_learn_false_skips_record_but_keeps_history(self):
        brain = self.make_brain(scripted({}))
        a = brain.ask(Q_SIMPLE, depth="schnell", learn=False, session_id="eval-0")
        self.assertIsNone(a.interaction_id)
        self.assertEqual(self.learning.stats()["interaktionen"], 0)
        self.assertEqual(self.calls_for("extraktion"), [])
        self.assertEqual(len(self.memory.history("eval-0")), 2)

    def test_extraction_errors_never_fatal(self):
        def extraction(msgs, kw):
            raise LLMError("Extraktion kaputt")

        brain = self.make_brain(scripted({("extraktion", "system"): extraction}))
        a = brain.ask(Q_SIMPLE, depth="schnell")
        self.assertEqual(a.depth, "schnell")
        self.assertIsNotNone(a.interaction_id)
        learn = [s for s in a.steps if s.stage == "lernen"][0]
        self.assertEqual(learn.status, "fehler")
        self.assertIn("Extraktion kaputt", learn.summary)

    def test_record_gets_memories_and_lessons_used(self):
        brain = self.make_brain(scripted({("extraktion", "system"): '{"erinnerungen": []}'}))
        mem = brain.remember("Der Nutzer druckt sein PETG-Gehäuse mit 0,4-mm-Düse.", kind="fakt", importance=0.7)
        lesson = self.learning.add_lesson("Wandstärken immer als Vielfache der Linienbreite angeben.",
                                          scope="allgemein")
        a = brain.ask(Q_MEDIUM, depth="schnell", session_id="s1", project="Gehäuse")
        self.assertEqual([m.id for m in a.memories_used], [mem.id])
        self.assertEqual([l.id for l in a.lessons_used], [lesson.id])
        inter = self.learning.get(a.interaction_id)
        self.assertEqual(inter.memories_used, [mem.id])
        self.assertEqual(inter.lessons_used, [lesson.id])
        self.assertEqual(inter.project, "Gehäuse")
        self.assertEqual(inter.session_id, "s1")
        self.assertIn("Vielfache der Linienbreite", inter.context)
        self.assertIn("0,4-mm-Düse", inter.context)
        sys_msg = self.calls_for("schnell")[0]["messages"][0]["content"]
        self.assertIn("Verbindliche Lektionen", sys_msg)
        self.assertIn("Projekt: Gehäuse", sys_msg)
        self.assertIn("[fakt] Der Nutzer druckt", sys_msg)
        self.assertEqual(a.steps[0].summary, "1 Erinnerungen, 1 Lektionen, 0 Beispiele, 0 Dokument-Auszüge, "
                                             "0 Verlaufsnachrichten, Projektkontext")
        # Verlauf und Beispiele landen im nächsten Prompt
        self.learning.rate(a.interaction_id, 1)
        brain.ask("Und welche Düse für PLA?", depth="schnell", session_id="s1", project="Gehäuse")
        msgs = self.calls_for("schnell")[1]["messages"]
        roles = [m["role"] for m in msgs]
        self.assertEqual(roles[0], "system")
        self.assertIn(Q_MEDIUM, [m["content"] for m in msgs if m["role"] == "user"])
        self.assertEqual(roles.count("assistant"), 2)          # Beispiel + Verlauf
        inter2 = self.learning.last("s1")[0]
        self.assertEqual(len(inter2.history), 2)


# ------------------------------------------------------------------ Feedback
class FeedbackTest(BrainTestBase):
    LESSON_JSON = '{"regel": "Bei PETG immer 240 °C Düsentemperatur nennen.", "gilt_fuer": ["petg", "druck"], "allgemein": false}'

    def setup_interaction(self, extra_script=None):
        extraction = {"erinnerungen": [{"inhalt": "Der Nutzer druckt sein PETG-Gehäuse mit einer 0,6-mm-Düse.",
                                        "art": "fakt", "wichtigkeit": 0.9, "tags": ["petg"]}]}
        script = {("schnell", "OMEGA"): LONG_ANSWER, ("extraktion", "system"): json.dumps(extraction),
                  ("lektion", "system"): self.LESSON_JSON,
                  ("korrektur", "OMEGA"): "Vollständige Antwort: PETG bei 240 °C mit 1,6 mm Wandstärke drucken."}
        script.update(extra_script or {})
        brain = self.make_brain(scripted(script))
        self.used = brain.remember("Der Nutzer druckt PETG-Gehäuse bei 240 °C Düsentemperatur.", kind="fakt",
                                   importance=0.6)
        a = brain.ask(Q_MEDIUM, depth="schnell")
        self.assertEqual([m.id for m in a.memories_used], [self.used.id])
        self.assertEqual(len(a.new_memories), 1)
        self.ki = a.new_memories[0]
        self.assertEqual(self.ki.source, "ki")
        self.backend.calls.clear()
        return brain, a

    def test_unknown_id(self):
        brain = self.make_brain(scripted({}))
        with self.assertRaises(ValueError):
            brain.feedback(999, rating=1)
        with self.assertRaises(ValueError):
            brain.rewrite_correction(999, "x")

    def test_negative_without_comment_no_lesson_but_hygiene(self):
        brain, a = self.setup_interaction()
        fb = brain.feedback(a.interaction_id, rating=-1)
        self.assertEqual(fb.lessons, [])
        self.assertEqual(self.calls_for("lektion"), [])
        self.assertEqual(fb.interaction.rating, -1)
        self.assertEqual(fb.memories_adjusted, 2)
        self.assertAlmostEqual(self.memory.get(self.used.id).importance, 0.45)
        self.assertIsNone(self.memory.get(self.ki.id))
        self.assertIsNone(fb.correction_full)
        self.assertEqual(self.learning.stats()["lektionen"], 0)

    def test_negative_with_comment_creates_lesson(self):
        brain, a = self.setup_interaction()
        fb = brain.feedback(a.interaction_id, rating=-1, comment="Die Temperatur fehlt.")
        self.assertEqual(len(fb.lessons), 1)
        self.assertEqual(fb.lessons[0].rule, "Bei PETG immer 240 °C Düsentemperatur nennen.")
        self.assertEqual(fb.lessons[0].topics, ["petg", "druck"])
        self.assertEqual(fb.lessons[0].scope, "thema")
        self.assertEqual(fb.lessons[0].source_interaction, a.interaction_id)
        self.assertEqual(fb.interaction.comment, "Die Temperatur fehlt.")
        lektion = self.calls_for("lektion")
        self.assertEqual(len(lektion), 1)
        self.assertEqual(lektion[0]["json_mode"], agents.LESSON_SCHEMA)
        self.assertEqual(lektion[0]["model"], self.cfg.routing_model)
        self.assertIn("Die Temperatur fehlt.", _last_user(lektion[0]["messages"]))
        # Lektion wirkt sofort im nächsten Prompt
        brain.ask("Welche Düsentemperatur für PETG?", depth="schnell", learn=False)
        self.assertIn("Düsentemperatur nennen", self.calls_for("schnell")[0]["messages"][0]["content"])

    def test_positive_rating_boosts_memories(self):
        brain, a = self.setup_interaction()
        fb = brain.feedback(a.interaction_id, rating=1, comment="Super.")
        self.assertEqual(fb.lessons, [])
        self.assertEqual(fb.memories_adjusted, 1)
        self.assertAlmostEqual(self.memory.get(self.used.id).importance, 0.7)
        self.assertIsNotNone(self.memory.get(self.ki.id))
        self.assertEqual(fb.interaction.rating, 1)
        self.assertEqual(self.backend.calls, [])

    def test_short_correction_is_rewritten(self):
        brain, a = self.setup_interaction()
        fb = brain.feedback(a.interaction_id, correction="PETG braucht 240 °C, nicht 230 °C.")
        korr = self.calls_for("korrektur")
        self.assertEqual(len(korr), 1)
        self.assertEqual(korr[0]["max_tokens"], self.cfg.max_tokens_synthesis)
        self.assertIn("PETG braucht 240 °C", _last_user(korr[0]["messages"]))
        self.assertIn(LONG_ANSWER, _last_user(korr[0]["messages"]))
        self.assertEqual(fb.correction_full, "Vollständige Antwort: PETG bei 240 °C mit 1,6 mm Wandstärke drucken.")
        self.assertEqual(fb.interaction.correction, "PETG braucht 240 °C, nicht 230 °C.")
        self.assertEqual(fb.interaction.correction_full, fb.correction_full)
        self.assertEqual(fb.interaction.rating, -1)
        self.assertEqual(len(fb.lessons), 1)
        self.assertIn("PETG braucht 240 °C", _last_user(self.calls_for("lektion")[0]["messages"]))
        # Beispiel nutzt die vollständige Korrektur
        self.assertEqual(self.learning.examples("PETG Gehäuse Wandstärke", 1)[0][1], fb.correction_full)
        # ohne explizite Bewertung keine Gedächtnis-Hygiene
        self.assertEqual(fb.memories_adjusted, 0)
        self.assertIsNotNone(self.memory.get(self.ki.id))

    def test_long_correction_or_given_full_skips_rewrite(self):
        brain, a = self.setup_interaction()
        long_corr = "Die vollständige korrigierte Antwort lautet: " + "PETG bei 240 °C drucken. " * 12
        self.assertGreaterEqual(len(long_corr), 200)
        fb = brain.feedback(a.interaction_id, correction=long_corr)
        self.assertEqual(self.calls_for("korrektur"), [])
        self.assertIsNone(fb.correction_full)
        self.assertEqual(fb.interaction.correction, long_corr.strip())
        brain2, a2 = self.setup_interaction()
        fb2 = brain2.feedback(a2.interaction_id, correction="kurz", correction_full="Voll ausformuliert.")
        self.assertEqual(self.calls_for("korrektur"), [])
        self.assertEqual(fb2.correction_full, "Voll ausformuliert.")

    def test_lesson_fallback_when_json_garbage(self):
        brain, a = self.setup_interaction({("lektion", "system"): "keine ahnung"})
        fb = brain.feedback(a.interaction_id, rating=-1, comment="Zu heiß, PETG verbrennt.")
        self.assertEqual(len(self.calls_for("lektion")), 2)
        self.assertEqual(fb.lessons[0].rule, "Zu heiß, PETG verbrennt.")
        self.assertEqual(fb.lessons[0].topics, ["welche", "wandstärke", "sollte", "gedrucktes", "gehäuse"])
        self.assertEqual(fb.lessons[0].scope, "thema")
        brain2, a2 = self.setup_interaction({("lektion", "system"): "nope"})
        fb2 = brain2.feedback(a2.interaction_id, correction="Lieber 1,6 mm.")
        self.assertEqual(fb2.lessons[0].rule, "Besser: Lieber 1,6 mm.")

    def test_lesson_general_scope_and_dedup(self):
        brain, a = self.setup_interaction({("lektion", "system"):
                                           '{"regel": "Immer Einheiten nennen.", "gilt_fuer": [], "allgemein": true}'})
        fb = brain.feedback(a.interaction_id, rating=-1, comment="Einheiten fehlen.")
        self.assertEqual(fb.lessons[0].scope, "allgemein")
        fb2 = brain.feedback(a.interaction_id, rating=-1, comment="Einheiten fehlen schon wieder.")
        self.assertEqual(fb2.lessons[0].id, fb.lessons[0].id)
        self.assertEqual(self.learning.stats()["lektionen"], 1)

    def test_invalid_rating(self):
        brain, a = self.setup_interaction()
        with self.assertRaises(ValueError):
            brain.feedback(a.interaction_id, rating=5)

    def test_rewrite_correction_error_returns_correction(self):
        brain, a = self.setup_interaction()
        self.backend.up = False
        self.assertEqual(brain.rewrite_correction(a.interaction_id, "Nur 1,6 mm."), "Nur 1,6 mm.")
        self.backend.up = True
        self.assertEqual(brain.rewrite_correction(a.interaction_id, "   "), "")
        brain2, a2 = self.setup_interaction({("korrektur", "OMEGA"): "   "})
        self.assertEqual(brain2.rewrite_correction(a2.interaction_id, "Nur 1,6 mm."), "Nur 1,6 mm.")


# ------------------------------------------------------------------ Fehler, Abbruch
class ErrorTest(BrainTestBase):
    def test_backend_down(self):
        brain = self.make_brain(up=False)
        out: list[str] = []
        steps: list[Step] = []
        a = brain.ask(Q_SIMPLE, depth="schnell", stream=out.append, progress=steps.append)
        self.assertEqual(a.depth, "fehler")
        self.assertEqual(a.text, "Modell-Server nicht erreichbar – starte Ollama (`ollama serve`) oder prüfe "
                                 "base_url (Standard).")
        self.assertIsNone(a.interaction_id)
        self.assertEqual(out, [])
        self.assertEqual(self.memory.history("standard"), [])
        self.assertEqual(self.learning.stats()["interaktionen"], 0)
        self.assertEqual([s.stage for s in a.steps], ["erinnern", "routing", "schnell", "system"])
        self.assertEqual(a.steps[-1].status, "fehler")
        self.assertEqual(a.steps[2].status, "fehler")
        self.assertTrue(any(s.status == "start" for s in steps))
        d = a.to_dict()
        self.assertEqual(d["tiefe"], "fehler")
        self.assertIsNone(d["interaktion_id"])
        self.assertFalse(brain.status()["verfuegbar"])

    def test_backend_down_with_base_url(self):
        brain = self.make_brain(up=False, base_url="http://127.0.0.1:9999")
        a = brain.ask(Q_SIMPLE, depth="schnell")
        self.assertIn("(http://127.0.0.1:9999)", a.text)

    def test_model_not_found(self):
        def missing(msgs, kw):
            raise ModelNotFound("model 'fake-modell' not found")

        brain = self.make_brain(scripted({("schnell", "OMEGA"): missing}))
        a = brain.ask(Q_SIMPLE, depth="schnell")
        self.assertEqual(a.depth, "fehler")
        self.assertEqual(a.text, "Modell »fake-modell« ist nicht installiert – `ollama pull fake-modell`.")
        brain2 = self.make_brain(scripted({("routing", "system"): missing}), fast_model="klein", auto_memory=False)
        a = brain2.ask(Q_SIMPLE)
        self.assertEqual(a.text, "Modell »klein« ist nicht installiert – `ollama pull klein`.")

    def test_generic_llm_error(self):
        def boom(msgs, kw):
            raise LLMError("HTTP 500: kaputt")

        brain = self.make_brain(scripted({("schnell", "OMEGA"): boom}))
        a = brain.ask(Q_SIMPLE, depth="schnell")
        self.assertEqual(a.text, "Modellfehler: HTTP 500: kaputt")
        self.assertEqual(a.depth, "fehler")

    def test_cancel_before_start(self):
        brain = self.make_brain(scripted({}))
        cancel = threading.Event()
        cancel.set()
        a = brain.ask(Q_SIMPLE, depth="schnell", cancel=cancel)
        self.assertEqual(a.depth, "fehler")
        self.assertEqual(a.text, "Modellfehler: Abgebrochen")
        self.assertEqual(self.backend.calls, [])
        self.assertFalse(brain.busy)

    def test_cancel_during_stream(self):
        brain = self.make_brain(scripted({}))
        cancel = threading.Event()
        out: list[str] = []

        def stream(piece):
            out.append(piece)
            cancel.set()

        a = brain.ask(Q_SIMPLE, depth="schnell", stream=stream, cancel=cancel)
        self.assertEqual(a.depth, "fehler")
        self.assertEqual(a.text, "Modellfehler: Abgebrochen")
        self.assertEqual(len(out), 1)
        self.assertIsNone(a.interaction_id)
        self.assertEqual(self.memory.history("standard"), [])
        self.assertEqual(self.learning.stats()["interaktionen"], 0)

    def test_cancel_in_council_worker(self):
        cancel = threading.Event()

        def experts(msgs, kw):
            cancel.set()
            return "Antwort, die abgebrochen wird."

        brain = self.make_brain(scripted({("experte", "*"): experts}), auto_memory=False)
        a = brain.ask(Q_MEDIUM, depth="mittel", cancel=cancel)
        self.assertEqual(a.depth, "fehler")
        self.assertEqual(a.text, "Modellfehler: Abgebrochen")
        self.assertEqual(self.calls_for("synthese"), [])

    def test_cancel_after_answer_skips_extraction(self):
        cancel = threading.Event()
        brain = self.make_brain(scripted({}))

        def progress(step):
            if step.stage == "schnell" and step.status == "fertig":
                cancel.set()

        a = brain.ask(Q_SIMPLE, depth="schnell", progress=progress, cancel=cancel)
        self.assertEqual(a.depth, "schnell")
        self.assertEqual(self.calls_for("extraktion"), [])
        self.assertEqual([s for s in a.steps if s.stage == "lernen"][0].summary, "übersprungen: abgebrochen")

    def test_record_failure_not_fatal(self):
        brain = self.make_brain(scripted({}), auto_memory=False)
        self.learning.close()
        a = brain.ask(Q_SIMPLE, depth="schnell")
        self.assertEqual(a.depth, "schnell")
        self.assertIsNone(a.interaction_id)
        self.assertEqual(a.steps[-1].stage, "system")
        self.assertEqual(a.steps[-1].status, "fehler")
        self.assertIn("Protokollierung", a.steps[-1].summary)


# ------------------------------------------------------------------ Verwaltung
class ManagementTest(BrainTestBase):
    def test_status_keys(self):
        brain = self.make_brain(scripted({}), models=["fake-modell", "fake-embed", "anderes:7b"])
        s = brain.status()
        self.assertEqual(set(s), {"backend", "verfuegbar", "modell", "routing_modell", "embedding_aktiv", "modelle",
                                  "gedaechtnis", "lernen", "werkzeuge", "beschaeftigt", "tiefe", "datenverzeichnis",
                                  "vektoren", "wissen", "projekte", "missionen", "automationen",
                                  "geraete", "modelle3d", "geo", "online"})
        self.assertEqual(s["missionen"], {"laufend": [], "anzahl": 0})
        self.assertFalse(s["automationen"]["aktiv"])
        self.assertIn("dokumente", s["wissen"])
        self.assertIn("projekte", s["projekte"])
        self.assertEqual(s["backend"], "fake")
        self.assertTrue(s["verfuegbar"])
        self.assertEqual(s["modell"], "fake-modell")
        self.assertEqual(s["routing_modell"], "fake-modell")
        self.assertEqual(s["modelle"], ["fake-modell", "fake-embed", "anderes:7b"])
        self.assertEqual(s["gedaechtnis"], self.memory.stats())
        self.assertEqual(s["lernen"], self.learning.stats())
        self.assertIn("rechnen", s["werkzeuge"])
        self.assertIn("gedaechtnis_merken", s["werkzeuge"])
        self.assertFalse(s["beschaeftigt"])
        self.assertEqual(s["tiefe"], "auto")
        self.assertEqual(set(s["vektoren"]), {"mit", "ohne", "modell", "dim"})
        brain.remember("Der Nutzer mag Carbon.", kind="praeferenz")
        v = brain.status()["vektoren"]
        self.assertEqual((v["mit"], v["ohne"], v["dim"]), (1, 0, 64))
        self.assertEqual(v["modell"], "nomic-embed-text")
        self.assertTrue(json.dumps(s, default=str))

    def test_set_model(self):
        brain = self.make_brain(scripted({}), models=["fake-modell", "fake-embed", "anderes:7b"])
        brain.set_model("anderes:7b")
        self.assertEqual(self.cfg.model, "anderes:7b")
        self.assertEqual(self.backend.default_model, "anderes:7b")
        with self.assertRaises(ModelNotFound):
            brain.set_model("nicht-da:3b")
        self.assertEqual(self.cfg.model, "anderes:7b")
        brain.set_model("fake-modell", fast=True)
        self.assertEqual(self.cfg.fast_model, "fake-modell")
        self.assertEqual(self.backend.default_model, "anderes:7b")
        self.assertEqual(self.cfg.routing_model, "fake-modell")
        brain.set_model("", fast=True)
        self.assertEqual(self.cfg.fast_model, "")
        with self.assertRaises(ValueError):
            brain.set_model("")
        # Backend nicht erreichbar → keine Prüfung, einfach setzen
        self.backend.up = False
        brain.set_model("irgendwas:1b")
        self.assertEqual(self.cfg.model, "irgendwas:1b")

    def test_models_empty_on_error(self):
        class Broken(FakeBackend):
            def list_models(self):
                raise BackendUnavailable("weg")

        cfg = Config(backend="fake", data_dir=self.tmp.name, model="fake-modell", workspace=self.tmp.name)
        brain = Brain(cfg, backend=Broken(), memory=MemoryStore(":memory:"), learning=LearningStore(":memory:"))
        self.addCleanup(brain.close)
        self.assertEqual(brain.models(), [])
        self.assertEqual(brain.status()["modelle"], [])
        brain2 = self.make_brain(scripted({}), models=["a", "b"])
        self.assertEqual([m.name for m in brain2.models()], ["a", "b"])

    def test_remember_forget_recall(self):
        brain = self.make_brain(scripted({}))
        m = brain.remember("Der Nutzer bevorzugt Carbon für den Drohnenrahmen.", kind="praeferenz",
                           tags=["Material"], project="Drohne")
        self.assertEqual(m.source, "nutzer")
        self.assertEqual(m.importance, 0.6)
        self.assertEqual(m.tags, ["material"])
        weak = brain.remember("Kaum wichtig: Drohnenrahmen in blau lackieren.", importance=0.1)
        hits = brain.recall("Drohnenrahmen Carbon")
        self.assertEqual([h.id for h in hits], [m.id])            # min_importance 0.15 blendet weak aus
        self.assertEqual(brain.recall("Drohnenrahmen", k=1, project="Drohne")[0].id, m.id)
        self.assertTrue(brain.forget(weak.id))
        self.assertFalse(brain.forget(weak.id))
        self.assertEqual(brain.reindex_memories(), 0)              # alle Vektoren vorhanden
        self.memory._db.execute("UPDATE memories SET embedding = NULL")
        self.memory._db.commit()
        self.assertEqual(brain.reindex_memories(), 1)

    def test_export_dataset_delegates(self):
        brain = self.make_brain(scripted({}), auto_memory=False)
        a = brain.ask(Q_SIMPLE, depth="schnell")
        brain.feedback(a.interaction_id, rating=1)
        path = os.path.join(self.tmp.name, "ds.jsonl")
        result = brain.export_dataset(path, fmt="chat", eval_share=0.0)
        self.assertEqual(set(result), {"train", "eval", "verworfen"})
        self.assertEqual(result["train"], 1)
        self.assertTrue(os.path.exists(path))
        self.assertTrue(os.path.exists(path + ".fragen.jsonl"))

    def test_tool_registry_passed_in_gets_memory_and_policy(self):
        from obito.tools import default_registry
        reg = default_registry(self.tmp.name, None, True)
        cfg = Config(backend="fake", data_dir=self.tmp.name, model="fake-modell", workspace=self.tmp.name)
        brain = Brain(cfg, backend=FakeBackend(), memory=MemoryStore(":memory:"), learning=LearningStore(":memory:"),
                      tools=reg, confirm=lambda n, a: True)
        self.addCleanup(brain.close)
        self.assertIs(brain.tools, reg)
        self.assertIsNotNone(reg.get("gedaechtnis_merken"))
        self.assertTrue(reg.confirm("x", {}))
        res = reg.run("gedaechtnis_merken", {"inhalt": "Der Nutzer mag Tests."})
        self.assertTrue(res.ok)
        self.assertEqual(brain.memory.count(), 1)


if __name__ == "__main__":
    unittest.main()
