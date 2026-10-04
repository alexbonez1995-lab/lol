"""Tests für obito/cli.py und obito/__main__.py – FakeBackend, :memory:/tempfile, kein Netzwerk.

``ChatSession.handle_line`` wird für jeden Befehl mit einem echten ``Brain`` geprüft, dessen
FakeBackend über ``agents.stage_of`` skriptet ist; Ausgaben landen in ``io.StringIO``, Eingaben
kommen aus einer skripteten ``inp``-Funktion. Unterbefehle laufen über ``cli.main`` mit
``--backend fake`` bzw. injiziertem Backend.
"""

from __future__ import annotations

import _thread
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest import mock

from obito import cli
from obito.agents import select_experts, stage_of
from obito.brain import Brain, Step
from obito.cli import ChatSession, build_parser, doctor, main, recommend_config, step_line
from obito.config import Config
from obito.learning import LearningStore
from obito.llm import BackendUnavailable, FakeBackend
from obito.memory import MemoryStore

LONG_ANSWER = ("Für ein PETG-Gehäuse sind 1,6 mm Wandstärke (vier Linien à 0,4 mm) ein guter Kompromiss "
               "zwischen Stabilität und Druckzeit.")
SMALLTALK = "Mir geht es gut, danke der Nachfrage! Womit kann ich dir heute helfen?"
Q_SIMPLE = "Hallo, wie geht es dir?"
Q_MEDIUM = "Welche Wandstärke sollte mein gedrucktes Gehäuse aus PETG haben, damit es stabil bleibt?"
Q_FILE = "Schreibe die Datei notiz.txt mit dem Inhalt Hallo."
TOOL_CALL = '<werkzeug>{"name": "datei_schreiben", "args": {"pfad": "notiz.txt", "inhalt": "Hallo"}}</werkzeug>'
MEMORY_JSON = ('{"erinnerungen": [{"inhalt": "Nutzer druckt Gehäuse aus PETG mit 1,6 mm Wandstärke", '
               '"art": "fakt", "wichtigkeit": 0.8, "tags": ["3d-druck"]}]}')
LESSON_JSON = ('{"regel": "Wandstärke immer als Vielfaches der Linienbreite angeben.", '
               '"gilt_fuer": ["3d-druck", "wandstärke"], "allgemein": false}')
CRITIC_JSON = '{"bewertung": 9, "fehler": [], "widersprueche": [], "fehlt": [], "sicher": true}'
REWRITE = "Umgeschrieben: Für PETG-Gehäuse sind 2,0 mm Wandstärke (fünf Linien) robuster."


def _has_tool_result(msgs: list[dict]) -> bool:
    return any(m["role"] == "user" and m["content"].startswith(("Ergebnis von Werkzeug", "Fehler bei Werkzeug"))
               for m in msgs)


def scripted(script: dict, default: str = SMALLTALK):
    """Responder: ``script[(stage, who)]`` → Text oder Funktion; ``(stage, "*")`` für jeden ``who``."""

    def responder(msgs, kw):
        stage, who = stage_of(msgs)
        value = script.get((stage, who), script.get((stage, "*"), default))
        if callable(value):
            return value(msgs, kw)
        return value

    return responder


def default_script() -> dict:
    return {
        ("extraktion", "system"): '{"erinnerungen": []}',
        ("experte", "*"): lambda msgs, kw: f"Antwort von {stage_of(msgs)[1]}: 1,6 mm Wandstärke.",
        ("kritiker", "KRITIKER"): CRITIC_JSON,
        ("synthese", "OMEGA"): LONG_ANSWER,
        ("schnell", "OMEGA"): SMALLTALK,
        ("lektion", "system"): LESSON_JSON,
        ("korrektur", "OMEGA"): REWRITE,
    }


class ScriptedInput:
    """Skriptete ``input``-Funktion: liefert die Antworten der Reihe nach, merkt sich die Prompts,
    wirft ``EOFError`` wenn nichts mehr da ist."""

    def __init__(self, *answers: str):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)


class CliTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def make_cfg(self, **kw) -> Config:
        base = dict(backend="fake", data_dir=self.tmp.name, model="fake-modell", fast_model="", depth="auto",
                    workspace=self.tmp.name, parallel_calls=1, deadline=30.0)
        base.update(kw)
        return Config(**base)

    def make_session(self, script: dict | None = None, *, inputs=(), up=True, models=None, **cfg_kw) -> ChatSession:
        self.cfg = self.make_cfg(**cfg_kw)
        self.backend = FakeBackend(responder=scripted(script if script is not None else default_script()), up=up,
                                   embed_dim=64, models=models)
        self.memory = MemoryStore(":memory:")
        self.learning = LearningStore(":memory:")
        self.brain = Brain(self.cfg, backend=self.backend, memory=self.memory, learning=self.learning)
        self.addCleanup(self.brain.close)
        self.out = io.StringIO()
        self.inp = ScriptedInput(*inputs)
        return ChatSession(self.brain, self.cfg, out=self.out, inp=self.inp)

    def text(self) -> str:
        return self.out.getvalue()

    def calls_for(self, stage: str) -> list[dict]:
        return [c for c in self.backend.calls if stage_of(c["messages"])[0] == stage]


# ------------------------------------------------------------------ Formatierung
class StepLineTest(unittest.TestCase):
    def test_start_fertig_fehler(self):
        self.assertEqual(step_line(Step("experte", "ALPHA", "ALPHA (3D-Konstruktion) denkt…", status="start")),
                         "⟳ ALPHA denkt…")
        self.assertEqual(step_line(Step("routing", "system", "Router ordnet die Frage ein…", status="start")),
                         "⟳ Router ordnet die Frage ein…")
        self.assertEqual(step_line(Step("kritiker", "KRITIKER", "Bewertung 8/10, 2 Fehler, 1 Widerspruch")),
                         "✓ KRITIKER: Bewertung 8/10, 2 Fehler, 1 Widerspruch")
        self.assertEqual(step_line(Step("experte", "ALPHA", "ALPHA (3D-Konstruktion): 812 Zeichen")),
                         "✓ ALPHA (3D-Konstruktion): 812 Zeichen")
        self.assertEqual(step_line(Step("werkzeug", "rechnen", "rechnen: ok")), "✓ rechnen: ok")
        self.assertEqual(step_line(Step("erinnern", "system", "2 Erinnerungen")), "✓ erinnern: 2 Erinnerungen")
        self.assertEqual(step_line(Step("routing", "system", "Fehler – kaputt", status="fehler")),
                         "✗ routing: Fehler – kaputt")

    def test_write_survives_non_utf8_stream(self):
        raw = io.BytesIO()
        out = io.TextIOWrapper(raw, encoding="ascii", errors="strict")
        cli._println(out, "⟳ ALPHA denkt… ✓ 💾")
        out.flush()
        self.assertEqual(raw.getvalue().decode("ascii"), "? ALPHA denkt? ? ?\n")


# ------------------------------------------------------------------ REPL: Fragen
class ChatAskTest(CliTestBase):
    def test_fast_question_streams_and_shows_footer(self):
        s = self.make_session()
        self.assertTrue(s.handle_line(Q_SIMPLE))
        text = self.text()
        self.assertIn("OBITO> " + SMALLTALK + "\n", text)
        self.assertIn("✓ erinnern: 0 Erinnerungen", text)
        self.assertIn("⟳ OMEGA denkt…", text)
        self.assertIn("✓ OMEGA: ", text)
        self.assertIn("— schnell · ", text)
        self.assertIn("Interaktion #1 (/gut, /schlecht, /korrektur)", text)
        self.assertEqual(s.last_interaction_id, 1)
        self.assertEqual(s.last_answer.text, SMALLTALK)
        self.assertEqual(self.calls_for("routing"), [])        # Heuristik „einfach“ → kein Router
        self.assertNotIn("💾", text)

    def test_new_memory_is_shown_as_unsicher(self):
        script = default_script()
        script[("extraktion", "system")] = MEMORY_JSON
        s = self.make_session(script)
        s.handle_line("/tief")
        s.handle_line(Q_MEDIUM)
        text = self.text()
        self.assertIn("💾 gemerkt (unsicher): Nutzer druckt Gehäuse aus PETG mit 1,6 mm Wandstärke", text)
        self.assertIn("✓ KRITIKER: Bewertung 9/10, 0 Fehler, 0 Widersprüche", text)
        self.assertIn("⟳ KRITIKER prüft die Antworten…", text)
        self.assertIn("— tief · ", text)
        self.assertEqual(len(s.last_answer.experts), 5)
        for eid in s.last_answer.experts:
            self.assertIn(f"⟳ {eid} denkt…", text)
            self.assertIn(f"✓ {eid} (", text)
        self.assertEqual(self.memory.count(), 1)

    def test_depth_commands_change_depth(self):
        s = self.make_session()
        self.assertIsNone(s.depth)
        self.assertEqual(s.effective_depth, "auto")
        s.handle_line("/mittel")
        self.assertEqual(s.depth, "mittel")
        s.handle_line(Q_MEDIUM)
        self.assertEqual(s.last_answer.depth, "mittel")
        self.assertEqual(len(s.last_answer.experts), 3)
        self.assertEqual(self.calls_for("kritiker"), [])
        s.handle_line("/schnell")
        self.assertEqual(s.depth, "schnell")
        s.handle_line("/auto")
        self.assertEqual(s.depth, "auto")
        s.handle_line("/tief")
        self.assertEqual(s.depth, "tief")
        self.assertIn("Tiefe: mittel", self.text())
        self.assertIn("Tiefe: tief", self.text())

    def test_error_answer_is_shown_without_prefix(self):
        s = self.make_session(up=False)
        s.handle_line(Q_SIMPLE)
        text = self.text()
        self.assertIn("✗ Modell-Server nicht erreichbar", text)
        self.assertEqual(text.count("Modell-Server nicht erreichbar"), 1)   # System-Step nicht doppelt
        self.assertIn("✗ OMEGA: Fehler – Fake-Backend ist abgeschaltet.", text)
        self.assertNotIn("OBITO> ", text)
        self.assertIsNone(s.last_interaction_id)
        self.assertEqual(s.last_answer.depth, "fehler")

    def test_banner_warns_when_backend_down(self):
        s = self.make_session(up=False)
        s.banner()
        self.assertIn("⚠ Modell-Server nicht erreichbar", self.text())
        self.assertIn("Modell fake-modell", self.text())

    def test_empty_and_unknown(self):
        s = self.make_session()
        self.assertTrue(s.handle_line(""))
        self.assertTrue(s.handle_line("   "))
        self.assertTrue(s.handle_line("/gibtsnicht 1 2"))
        self.assertIn("Unbekannter Befehl /gibtsnicht", self.text())
        self.assertEqual(self.backend.calls, [])

    def test_project_is_passed_to_brain(self):
        s = self.make_session()
        s.handle_line("/projekt Drohne")
        self.assertEqual(s.project, "Drohne")
        s.handle_line(Q_SIMPLE)
        self.assertEqual(s.last_answer.project, "Drohne")
        self.assertEqual(self.learning.get(1).project, "Drohne")
        s.handle_line("/projekt")
        self.assertIn("Projekt: Drohne", self.text())
        s.handle_line("/projekt -")
        self.assertIsNone(s.project)
        self.assertIn("Projekt gelöscht", self.text())

    def test_invalid_depth_rejected(self):
        self.make_session()
        with self.assertRaises(ValueError):
            ChatSession(self.brain, self.cfg, out=self.out, inp=self.inp, depth="ultra")


# ------------------------------------------------------------------ REPL: Werkzeuge
class ChatToolTest(CliTestBase):
    def tool_script(self) -> dict:
        script = default_script()

        def fast(msgs, kw):
            if _has_tool_result(msgs):
                return "Die Datei notiz.txt ist geschrieben – fertig und erledigt wie gewünscht."
            return "Ich schreibe die Datei. " + TOOL_CALL

        script[("schnell", "OMEGA")] = fast
        return script

    def test_dangerous_tool_denied_by_default_answer(self):
        s = self.make_session(self.tool_script(), inputs=["n"])
        s.handle_line("/schnell")
        s.handle_line(Q_FILE)
        self.assertTrue(any("[j/N]" in p for p in self.inp.prompts))
        self.assertIn("⚠ Werkzeug »datei_schreiben« will ausführen", self.text())
        self.assertIn("✗ datei_schreiben: Fehler – Vom Nutzer abgelehnt", self.text())
        self.assertFalse(Path(self.tmp.name, "notiz.txt").exists())

    def test_dangerous_tool_allowed(self):
        s = self.make_session(self.tool_script(), inputs=["j"])
        s.handle_line("/schnell")
        s.handle_line(Q_FILE)
        self.assertIn("✓ datei_schreiben: ok", self.text())
        self.assertEqual(Path(self.tmp.name, "notiz.txt").read_text(encoding="utf-8"), "Hallo")
        self.assertIn("Werkzeuge: datei_schreiben", self.text())

    def test_eof_at_confirmation_means_no(self):
        s = self.make_session(self.tool_script(), inputs=[])
        s.handle_line("/schnell")
        s.handle_line(Q_FILE)
        self.assertIn("Vom Nutzer abgelehnt", self.text())
        self.assertFalse(Path(self.tmp.name, "notiz.txt").exists())

    def test_tools_listing(self):
        s = self.make_session()
        s.handle_line("/werkzeuge")
        text = self.text()
        self.assertIn("rechnen:", text)
        self.assertIn("datei_schreiben:", text)
        self.assertIn("[gefährlich – Rückfrage]", text)
        self.assertIn("gedaechtnis_suchen", text)


# ------------------------------------------------------------------ REPL: Strg+C
class ChatCancelTest(CliTestBase):
    def setUp(self):
        super().setUp()
        if threading.current_thread() is not threading.main_thread():
            self.skipTest("Strg+C-Simulation braucht den Hauptthread")
        if signal.getsignal(signal.SIGINT) is not signal.default_int_handler:
            self.skipTest("SIGINT-Handler ist nicht der Standard – Strg+C nicht simulierbar")

    def test_first_ctrl_c_cancels_answer(self):
        holder: dict = {}

        def slow(msgs, kw):
            s = holder["session"]
            s._waiting.wait(5)
            _thread.interrupt_main()
            s.cancel_event.wait(5)
            return LONG_ANSWER

        script = default_script()
        script[("schnell", "OMEGA")] = slow
        s = self.make_session(script)
        holder["session"] = s
        s.handle_line("/schnell")
        self.assertTrue(s.handle_line(Q_SIMPLE))
        text = self.text()
        self.assertTrue(s.cancel_event.is_set())
        self.assertIn("⏹ Abbruch angefordert", text)
        self.assertIn("⏹ Abgebrochen.", text)
        self.assertNotIn("Modellfehler", text)
        self.assertNotIn("✗", text)                 # nach dem Abbruch kein Fehler-Rauschen der Stufen
        self.assertEqual(s.last_answer.depth, "fehler")
        self.assertIsNone(s.last_interaction_id)
        self.assertEqual(self.learning.stats()["interaktionen"], 0)

    def test_second_ctrl_c_ends_session(self):
        holder: dict = {}
        release = threading.Event()

        def slow(msgs, kw):
            s = holder["session"]
            s._waiting.wait(5)
            _thread.interrupt_main()
            s.cancel_event.wait(5)
            _thread.interrupt_main()
            release.wait(5)
            return LONG_ANSWER

        script = default_script()
        script[("schnell", "OMEGA")] = slow
        s = self.make_session(script)
        holder["session"] = s
        s.handle_line("/schnell")
        try:
            self.assertFalse(s.handle_line(Q_SIMPLE))
            self.assertIn("Beende OBITO.", self.text())
        finally:
            release.set()
            for t in threading.enumerate():
                if t.name == "obito-chat-frage":
                    t.join(5)


# ------------------------------------------------------------------ REPL: Gedächtnis-Befehle
class ChatMemoryCommandsTest(CliTestBase):
    def test_merk_vergiss_suche_erinnerungen(self):
        s = self.make_session()
        s.handle_line("/suche xyzzy nichts")
        self.assertIn("Keine passenden Erinnerungen", self.text())
        s.handle_line("/merk Mein Drucker ist ein Prusa MK4 mit 0,4-mm-Düse")
        self.assertIn("💾 gemerkt: #1 [notiz", self.text())
        m = self.memory.get(1)
        self.assertEqual(m.source, "nutzer")
        self.assertEqual(m.kind, "notiz")
        s.handle_line("/merk praeferenz: Ich bevorzuge metrische Einheiten")
        self.assertEqual(self.memory.get(2).kind, "praeferenz")
        s.handle_line("/merk")
        self.assertIn("Aufruf: /merk", self.text())

        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/suche Prusa Düse")
        self.assertIn("#1 [notiz", self.text())
        self.assertIn("Score", self.text())
        s.handle_line("/suche")
        self.assertIn("Aufruf: /suche", self.text())

        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/erinnerungen 1")
        self.assertIn("Die 1 jüngsten Erinnerungen", self.text())
        self.assertIn("#2 [praeferenz", self.text())
        self.assertNotIn("#1 [notiz", self.text())
        s.handle_line("/erinnerungen abc")
        self.assertIn("Aufruf: /erinnerungen", self.text())

        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/vergiss 1")
        self.assertIn("Erinnerung #1 gelöscht", self.text())
        self.assertIsNone(self.memory.get(1))
        s.handle_line("/vergiss #1")
        self.assertIn("Keine Erinnerung mit der ID #1", self.text())
        s.handle_line("/vergiss x")
        self.assertIn("Aufruf: /vergiss", self.text())
        s.handle_line("/vergiss 2")
        s.handle_line("/erinnerungen")
        self.assertIn("Noch keine Erinnerungen", self.text())

    def test_project_scopes_memories(self):
        s = self.make_session()
        s.handle_line("/projekt Rover")
        s.handle_line("/merk Der Rover hat vier Mecanum-Räder")
        self.assertEqual(self.memory.get(1).project, "Rover")
        s.handle_line("/erinnerungen")
        self.assertIn("(Projekt Rover)", self.text())


# ------------------------------------------------------------------ REPL: Modelle
class ChatModelCommandsTest(CliTestBase):
    def test_modelle_and_modell(self):
        s = self.make_session(models=["fake-modell", "klein", "fake-embed"])
        s.handle_line("/modelle")
        text = self.text()
        self.assertIn("* fake-modell", text)
        self.assertIn("  klein", text)
        s.handle_line("/modell")
        self.assertIn("Modell: fake-modell", self.text())
        s.handle_line("/modell gibtsnicht")
        self.assertIn("✗ Modell »gibtsnicht« ist nicht installiert", self.text())
        self.assertEqual(self.cfg.model, "fake-modell")
        s.handle_line("/modell klein")
        self.assertEqual(self.cfg.model, "klein")
        self.assertEqual(self.backend.default_model, "klein")
        s.handle_line("/modell schnell fake-modell")
        self.assertEqual(self.cfg.fast_model, "fake-modell")
        self.assertIn("Routing-Modell: fake-modell", self.text())
        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/modelle")
        self.assertIn("* klein", self.text())
        self.assertIn("» fake-modell", self.text())

    def test_modelle_when_backend_down(self):
        s = self.make_session(up=False)

        def broken():
            raise BackendUnavailable("Server weg")

        self.backend.list_models = broken
        s.handle_line("/modelle")
        self.assertIn("Keine Modelle abrufbar", self.text())
        s.handle_line("/modell irgendwas")          # nicht erreichbar -> ohne Prüfung gesetzt
        self.assertEqual(self.cfg.model, "irgendwas")


# ------------------------------------------------------------------ REPL: Feedback
class ChatFeedbackTest(CliTestBase):
    def test_feedback_without_answer(self):
        s = self.make_session()
        for cmd in ("/gut", "/schlecht egal", "/korrektur besser"):
            s.handle_line(cmd)
        self.assertEqual(self.text().count("Keine bewertbare Antwort"), 3)
        self.assertEqual(self.backend.calls, [])

    def test_gut_boosts_memories(self):
        script = default_script()
        s = self.make_session(script)
        self.memory.remember("PETG Gehäuse Wandstärke 1,6 mm bewährt", kind="fakt", importance=0.5)
        s.handle_line("/schnell")
        s.handle_line(Q_MEDIUM)
        self.assertEqual(len(s.last_answer.memories_used), 1)
        s.handle_line("/gut passt")
        self.assertIn("👍 Danke – Interaktion #1 als gut gespeichert.", self.text())
        self.assertIn("💾 1 Erinnerung(en) angepasst.", self.text())
        inter = self.learning.get(1)
        self.assertEqual(inter.rating, 1)
        self.assertEqual(inter.comment, "passt")
        self.assertAlmostEqual(self.memory.get(1).importance, 0.6, places=3)

    def test_schlecht_asks_what_was_wrong(self):
        s = self.make_session(inputs=["Die Einheiten fehlten"])
        s.handle_line(Q_SIMPLE)
        s.handle_line("/schlecht")
        self.assertEqual(self.inp.prompts, ["Was war falsch? "])
        text = self.text()
        self.assertIn("👎 Interaktion #1 als schlecht gespeichert.", text)
        self.assertIn("📚 Lektion #1 [thema]: Wandstärke immer als Vielfaches der Linienbreite angeben.", text)
        inter = self.learning.get(1)
        self.assertEqual(inter.rating, -1)
        self.assertEqual(inter.comment, "Die Einheiten fehlten")
        self.assertEqual(len(self.calls_for("lektion")), 1)
        self.assertEqual(len(self.learning.list_lessons()), 1)

    def test_schlecht_without_comment_creates_no_lesson(self):
        s = self.make_session(inputs=[""])
        s.handle_line(Q_SIMPLE)
        s.handle_line("/schlecht")
        self.assertIn("Ohne Kommentar entsteht keine Lektion", self.text())
        self.assertEqual(self.calls_for("lektion"), [])
        self.assertEqual(self.learning.get(1).rating, -1)

    def test_schlecht_with_inline_comment(self):
        s = self.make_session()
        s.handle_line(Q_SIMPLE)
        s.handle_line("/schlecht zu allgemein")
        self.assertEqual(self.inp.prompts, [])
        self.assertEqual(self.learning.get(1).comment, "zu allgemein")
        self.assertEqual(len(self.learning.list_lessons()), 1)

    def test_korrektur_accepts_proposal(self):
        s = self.make_session(inputs=[""])        # Enter = Ja
        s.handle_line("/schnell")
        s.handle_line(Q_MEDIUM)
        s.handle_line("/korrektur Besser 2,0 mm Wandstärke")
        text = self.text()
        self.assertIn("Vorschlag für die vollständige, korrigierte Antwort:", text)
        self.assertIn(REWRITE, text)
        self.assertEqual(self.inp.prompts, ["Übernehmen? [J/n] "])
        self.assertIn("✓ Korrektur zu Interaktion #1 gespeichert.", text)
        self.assertIn("📚 Lektion #1", text)
        inter = self.learning.get(1)
        self.assertEqual(inter.correction, "Besser 2,0 mm Wandstärke")
        self.assertEqual(inter.correction_full, REWRITE)
        self.assertEqual(inter.rating, -1)
        self.assertEqual(len(self.calls_for("korrektur")), 1)   # kein zweites Umschreiben im Brain

    def test_korrektur_declined_keeps_correction_verbatim(self):
        s = self.make_session(inputs=["n"])
        s.handle_line("/schnell")
        s.handle_line(Q_MEDIUM)
        s.handle_line("/korrektur Besser 2,0 mm Wandstärke")
        self.assertIn("Vorschlag verworfen", self.text())
        inter = self.learning.get(1)
        self.assertEqual(inter.correction_full, "Besser 2,0 mm Wandstärke")
        self.assertEqual(len(self.calls_for("korrektur")), 1)

    def test_korrektur_without_text(self):
        s = self.make_session()
        s.handle_line(Q_SIMPLE)
        s.handle_line("/korrektur")
        self.assertIn("Aufruf: /korrektur", self.text())

    def test_feedback_falls_back_to_last_interaction_of_session(self):
        s = self.make_session()
        iid = self.learning.record("standard", "Frühere Frage?", "Frühere Antwort mit genug Text für alles.")
        s.handle_line("/gut")
        self.assertIn(f"Interaktion #{iid} als gut", self.text())
        self.assertEqual(self.learning.get(iid).rating, 1)

    def test_lektionen_and_delete(self):
        s = self.make_session()
        s.handle_line("/lektionen")
        self.assertIn("Noch keine Lektionen", self.text())
        self.learning.add_lesson("Immer Einheiten nennen.", scope="allgemein")
        self.learning.add_lesson("Wandstärke in mm.", topics=["3d-druck"])
        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/lektionen")
        text = self.text()
        self.assertIn("Lektionen (2 von 2):", text)
        self.assertIn("#1 [allgemein] Immer Einheiten nennen.", text)
        self.assertIn("#2 [thema] Wandstärke in mm.", text)
        self.assertIn("Themen: 3d-druck", text)
        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/lektionen 1")
        self.assertIn("Lektionen (1 von 2):", self.text())
        self.assertNotIn("#1 [allgemein]", self.text())
        s.handle_line("/lektion-loeschen 1")
        self.assertIn("Lektion #1 gelöscht.", self.text())
        self.assertEqual(len(self.learning.list_lessons()), 1)
        s.handle_line("/lektion-löschen 1")
        self.assertIn("Keine Lektion mit der ID #1", self.text())
        s.handle_line("/lektion-loeschen")
        self.assertIn("Aufruf: /lektion-loeschen", self.text())


# ------------------------------------------------------------------ REPL: Sonstiges
class ChatMiscCommandsTest(CliTestBase):
    def test_hilfe_lists_every_command(self):
        s = self.make_session()
        s.handle_line("/hilfe")
        text = self.text()
        for usage in ("/hilfe", "/merk", "/vergiss", "/suche", "/erinnerungen", "/projekt", "/tief", "/mittel",
                      "/schnell", "/auto", "/modelle", "/modell", "/gut", "/schlecht", "/korrektur", "/lektionen",
                      "/lektion-loeschen", "/spur", "/werkzeuge", "/status", "/export", "/beenden"):
            self.assertIn(usage, text)
        # jeder Befehl der Tabelle ist auch im Dispatch erreichbar
        for names, method, _u, _h in ChatSession.COMMANDS:
            for n in names:
                self.assertIn(n, s._dispatch)
            self.assertTrue(callable(getattr(s, method)))

    def test_spur(self):
        s = self.make_session()
        s.handle_line("/spur")
        self.assertIn("Noch keine Antwort", self.text())
        s.handle_line(Q_SIMPLE)
        self.out.truncate(0), self.out.seek(0)
        s.handle_line("/spur")
        text = self.text()
        self.assertIn("Denk-Spur (schnell):", text)
        self.assertIn("✓ erinnern/system:", text)
        self.assertIn("✓ schnell/OMEGA:", text)

    def test_status(self):
        s = self.make_session()
        s.handle_line("/status")
        text = self.text()
        self.assertIn("Backend: fake (erreichbar)", text)
        self.assertIn("Modell: fake-modell", text)
        self.assertIn("Embedding: aktiv", text)
        self.assertIn("Gedächtnis: 0 Erinnerungen", text)
        self.assertIn("Lernen: 0 Interaktionen", text)
        self.assertIn("Werkzeuge: rechnen", text)
        self.assertIn(f"Datenverzeichnis: {self.cfg.data_path}", text)
        self.assertIn("Installierte Modelle: fake-modell, fake-embed", text)

    def test_export_default_and_explicit_path(self):
        s = self.make_session()
        s.handle_line(Q_SIMPLE)
        s.handle_line("/gut")
        s.handle_line("/export")
        default = Path(self.cfg.datasets_dir) / "obito.jsonl"
        self.assertTrue(default.exists())
        self.assertTrue(Path(str(default) + ".eval.jsonl").exists())
        self.assertTrue(Path(str(default) + ".fragen.jsonl").exists())
        self.assertIn("Datensatz: 1 Trainings-, 0 Eval-Beispiele", self.text())
        target = Path(self.tmp.name, "eigen", "daten.jsonl")
        s.handle_line(f"/export {target}")
        self.assertTrue(target.exists())
        self.assertIn(str(target), self.text())
        rows = [json.loads(l) for l in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["messages"][-1]["content"], SMALLTALK)

    def test_beenden_returns_false(self):
        s = self.make_session()
        for cmd in ("/beenden", "/exit", "/quit", "/q"):
            self.assertFalse(s.handle_line(cmd))
        self.assertIn("Bis bald.", self.text())

    def test_run_loop_until_eof(self):
        s = self.make_session(inputs=[Q_SIMPLE, "/status", "/beenden", "/status"])
        self.assertEqual(s.run(), 0)
        text = self.text()
        self.assertIn("OBITO – lokaler KI-Assistent", text)
        self.assertIn("OBITO> " + SMALLTALK, text)
        self.assertEqual(text.count("Backend: fake"), 1)      # nach /beenden kommt nichts mehr
        s2 = self.make_session(inputs=[])
        self.assertEqual(s2.run(), 0)

    def test_show_progress_off(self):
        s = self.make_session()
        s.show_progress = False
        s.handle_line(Q_SIMPLE)
        text = self.text()
        self.assertNotIn("⟳", text)
        self.assertNotIn("✓", text)
        self.assertIn("OBITO> " + SMALLTALK, text)


# ------------------------------------------------------------------ Parser
class ParserTest(unittest.TestCase):
    def test_subcommands_parse(self):
        p = build_parser()
        a = p.parse_args([])
        self.assertIsNone(a.befehl)
        a = p.parse_args(["--modell", "m", "chat", "--projekt", "P", "--tiefe", "tief", "--sitzung", "s1"])
        self.assertEqual((a.befehl, a.modell, a.projekt, a.tiefe, a.sitzung), ("chat", "m", "P", "tief", "s1"))
        a = p.parse_args(["doctor", "--training", "--backend", "fake"])
        self.assertTrue(a.training)
        self.assertEqual(a.backend, "fake")          # globale Option auch nach dem Unterbefehl
        a = p.parse_args(["--backend", "ollama", "doctor"])
        self.assertEqual(a.backend, "ollama")        # Unterbefehl überschreibt die globale Option nicht mit None
        a = p.parse_args(["serve", "--host", "0.0.0.0", "--port", "9000", "--gefaehrlich-erlauben"])
        self.assertEqual((a.host, a.port, a.gefaehrlich_erlauben), ("0.0.0.0", 9000, True))
        a = p.parse_args(["export-dataset", "--ausgabe", "x.jsonl", "--format", "dpo", "--min-bewertung", "0",
                          "--eval-anteil", "0.2", "--ohne-kontext", "--max-eigene", "5"])
        self.assertEqual((a.format, a.min_bewertung, a.eval_anteil, a.ohne_kontext, a.max_eigene),
                         ("dpo", 0, 0.2, True, 5))
        a = p.parse_args(["eval", "--datei", "f.jsonl", "--richter", "r", "--tiefe", "mittel", "--ausgabe", "b.json",
                          "--ohne-training", "d.jsonl"])
        self.assertEqual((a.datei, a.richter, a.tiefe, a.ausgabe, a.ohne_training),
                         ("f.jsonl", "r", "mittel", "b.json", "d.jsonl"))
        a = p.parse_args(["eval", "--vergleich", "a.json", "b.json"])
        self.assertEqual(a.vergleich, ["a.json", "b.json"])
        a = p.parse_args(["modelfile", "--name", "obito-v1", "--basis", "qwen2.5:7b", "--adapter", "a.gguf",
                          "--vorlage", "t.txt", "--erstellen", "--erzwingen"])
        self.assertEqual((a.name, a.basis, a.adapter, a.vorlage, a.erstellen, a.erzwingen),
                         ("obito-v1", "qwen2.5:7b", "a.gguf", "t.txt", True, True))
        a = p.parse_args(["memory", "search", "drucker", "-n", "3", "--projekt", "P"])
        self.assertEqual((a.aktion, a.frage, a.anzahl, a.projekt), ("search", "drucker", 3, "P"))
        a = p.parse_args(["memory", "reindex", "--alle"])
        self.assertTrue(a.alle)
        a = p.parse_args(["memory", "import", "x.json"])
        self.assertEqual(a.pfad, "x.json")
        a = p.parse_args(["config", "--schreiben", "c.json", "--empfehlen", "12"])
        self.assertEqual((a.schreiben, a.empfehlen), ("c.json", 12.0))

    def test_invalid_choices_exit_2(self):
        p = build_parser()
        with self.assertRaises(SystemExit) as cm:
            with mock.patch("sys.stderr", new=io.StringIO()):
                p.parse_args(["chat", "--tiefe", "ultra"])
        self.assertEqual(cm.exception.code, 2)
        with self.assertRaises(SystemExit):
            with mock.patch("sys.stderr", new=io.StringIO()):
                p.parse_args(["export-dataset"])          # --ausgabe fehlt

    def test_help_is_german(self):
        p = build_parser()
        text = p.format_help()
        self.assertIn("Aufruf:", text)
        self.assertIn("Unterbefehle:", text)
        self.assertIn("Optionen:", text)
        self.assertNotIn("usage:", text)
        out = io.StringIO()
        with mock.patch("sys.stdout", new=out):
            with self.assertRaises(SystemExit) as cm:
                main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("doctor", out.getvalue())

    def test_split_train_argv(self):
        self.assertEqual(cli._split_train_argv(["--modell", "m", "train", "--daten", "x", "--epochen", "2"]),
                         (["--modell", "m", "train"], ["--daten", "x", "--epochen", "2"]))
        self.assertEqual(cli._split_train_argv(["train"]), (["train"], []))
        self.assertEqual(cli._split_train_argv(["doctor", "train"]), (["doctor", "train"], None))
        self.assertEqual(cli._split_train_argv([]), ([], None))


# ------------------------------------------------------------------ doctor
class _OllamaLikeFake(FakeBackend):
    """FakeBackend, das sich wie Ollama ausgibt: ``show``/``running``/``base_url``."""

    name = "ollama"

    def __init__(self, *a, thinking=False, partial=False, **kw):
        super().__init__(*a, **kw)
        self.base_url = "http://127.0.0.1:11434"
        self.thinking = thinking
        self.partial = partial

    def show(self, name):
        return {"capabilities": ["completion", "thinking"] if self.thinking else ["completion"]}

    def running(self):
        size = 5_000_000_000
        return [{"name": self.default_model, "size": size, "size_vram": size // 2 if self.partial else size,
                 "expires_at": ""}]


class DoctorTest(CliTestBase):
    def test_up_exit_0(self):
        cfg = self.make_cfg()
        out = io.StringIO()
        code = doctor(cfg, out=out, backend=FakeBackend(up=True))
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("[OK]       Python", text)
        self.assertIn("Backend fake (Version fake) erreichbar", text)
        self.assertIn("[OK]       Hauptmodell fake-modell installiert", text)
        self.assertIn("Embedding-Modell nomic-embed-text fehlt – `ollama pull nomic-embed-text`", text)
        self.assertIn("Probeaufruf erfolgreich", text)
        self.assertIn("Ergebnis: Chat möglich.", text)
        self.assertNotIn("OLLAMA_NUM_PARALLEL", text)      # Umgebungshinweise nur für Ollama

    def test_down_exit_1(self):
        cfg = self.make_cfg()
        out = io.StringIO()
        code = doctor(cfg, out=out, backend=FakeBackend(up=False))
        text = out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("[FEHLER]   Modell-Server nicht erreichbar", text)
        self.assertIn("ollama serve", text)
        self.assertIn("Chat nicht möglich", text)

    def test_missing_model_exit_1(self):
        cfg = self.make_cfg(model="qwen2.5:7b")
        out = io.StringIO()
        code = doctor(cfg, out=out, backend=FakeBackend(models=["anderes"]))
        text = out.getvalue()
        self.assertEqual(code, 1)
        self.assertIn("Hauptmodell qwen2.5:7b fehlt – `ollama pull qwen2.5:7b`", text)

    def test_probe_uses_max_tokens_5(self):
        cfg = self.make_cfg(think="aus")
        fb = FakeBackend()
        doctor(cfg, out=io.StringIO(), backend=fb)
        probe = fb.calls[-1]
        self.assertEqual(probe["max_tokens"], 5)
        self.assertEqual(probe["num_ctx"], cfg.num_ctx)
        self.assertIs(probe["think"], False)

    def test_ollama_hints_speed_and_warnings(self):
        cfg = self.make_cfg(fast_model="klein", parallel_calls=2)
        fb = _OllamaLikeFake(models=["fake-modell", "klein", "nomic-embed-text"], thinking=True, partial=True)
        real_chat = fb.chat

        def chat(*a, **kw):
            res = real_chat(*a, **kw)
            res.completion_tokens = 5
            res.raw = {"eval_duration": 250_000_000}     # 5 Token in 0,25 s = 20 Tok/s
            return res

        fb.chat = chat
        out = io.StringIO()
        code = doctor(cfg, out=out, backend=fb, env={})
        text = out.getvalue()
        self.assertEqual(code, 0)
        self.assertIn("20,0 Tok/s", text)
        self.assertIn("Prognose je Stufe", text)
        self.assertIn("schnell ≈ 20 s", text)
        self.assertIn("läuft teilweise auf CPU", text)
        self.assertIn("fast_model ist nur sinnvoll", text)
        self.assertIn("Denk-Modell", text)
        self.assertIn("think: aus", text)
        for var in ("OLLAMA_NUM_PARALLEL", "OLLAMA_KEEP_ALIVE", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE"):
            self.assertIn(var, text)
        self.assertIn("Routing-Modell klein installiert", text)
        self.assertIn("Embedding-Modell nomic-embed-text installiert", text)

    def test_ollama_env_set_suppresses_hints(self):
        cfg = self.make_cfg(parallel_calls=2, think="aus")
        fb = _OllamaLikeFake(thinking=True)
        env = {"OLLAMA_NUM_PARALLEL": "2", "OLLAMA_KEEP_ALIVE": "30m", "OLLAMA_FLASH_ATTENTION": "1",
               "OLLAMA_KV_CACHE_TYPE": "q8_0"}
        out = io.StringIO()
        doctor(cfg, out=out, backend=fb, env=env)
        text = out.getvalue()
        self.assertNotIn("[INFO]     parallel_calls", text)
        self.assertNotIn("OLLAMA_FLASH_ATTENTION=1 spart", text)
        self.assertIn("vollständig im VRAM", text)
        self.assertIn("think: aus ist gesetzt", text)

    def test_memory_vectors_missing_hint(self):
        cfg = self.make_cfg()
        cfg.ensure_dirs()
        store = MemoryStore(cfg.memory_db)
        store.remember("Erinnerung ohne Vektor, weil kein Embedder gesetzt ist")
        store.close()
        learn = LearningStore(cfg.learning_db)
        learn.record("s", "Frage?", "Antwort mit ausreichend vielen Zeichen für den Test.")
        learn.close()
        out = io.StringIO()
        doctor(cfg, out=out, backend=FakeBackend())
        text = out.getvalue()
        self.assertIn("Vektoren fehlen: `python -m obito memory reindex`", text)
        self.assertIn("Lernen: 1 Interaktionen", text)

    def test_training_flag_lists_packages(self):
        cfg = self.make_cfg()
        out = io.StringIO()
        with mock.patch.object(cli.importlib.util, "find_spec", return_value=None):
            with mock.patch.dict(sys.modules, {"torch": None}):
                doctor(cfg, training=True, out=out, backend=FakeBackend())
        text = out.getvalue()
        self.assertIn("Training (LoRA):", text)
        self.assertIn("Paket torch fehlt", text)
        self.assertIn("pip install torch", text)
        self.assertIn("VRAM nicht prüfbar", text)

    def test_main_doctor_exit_codes(self):
        out = io.StringIO()
        self.assertEqual(main(["--backend", "fake", "--daten", self.tmp.name, "doctor"], out=out), 0)
        out = io.StringIO()
        self.assertEqual(main(["--daten", self.tmp.name, "doctor"], out=out, backend=FakeBackend(up=False)), 1)


# ------------------------------------------------------------------ config
class ConfigCommandTest(CliTestBase):
    def test_recommend_table(self):
        c = recommend_config(4)
        self.assertEqual((c.model, c.fast_model, c.depth, c.parallel_calls), ("qwen2.5:3b", "", "schnell", 1))
        c = recommend_config(8)
        self.assertEqual((c.model, c.fast_model, c.num_ctx, c.parallel_calls), ("qwen2.5:7b", "", 8192, 1))
        c = recommend_config(12)
        self.assertEqual((c.model, c.fast_model, c.num_ctx, c.parallel_calls), ("qwen2.5:7b", "qwen2.5:3b", 12288, 2))
        c = recommend_config(16)
        self.assertEqual((c.model, c.fast_model, c.parallel_calls), ("qwen2.5:14b", "qwen2.5:3b", 2))
        self.assertEqual(c.depth, "auto")
        c = recommend_config(24)
        self.assertEqual(c.model, "qwen2.5:14b")

    def test_empfehlen_writes_file(self):
        target = Path(self.tmp.name, "vorlage", "obito.json")
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "config", "--empfehlen", "8", "--schreiben", str(target)], out=out)
        self.assertEqual(code, 0)
        self.assertTrue(target.exists())
        data = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(data["model"], "qwen2.5:7b")
        self.assertEqual(data["fast_model"], "")
        self.assertEqual(data["num_ctx"], 8192)
        self.assertEqual(data["parallel_calls"], 1)
        self.assertIn("_hinweis", data)
        self.assertIn("Vorlage geschrieben", out.getvalue())
        self.assertIn("ollama pull qwen2.5:7b", out.getvalue())
        # Die Vorlage ist eine gültige Konfigurationsdatei
        cfg = cli.load_config(target, env={})
        self.assertEqual(cfg.model, "qwen2.5:7b")

    def test_empfehlen_default_target_never_overwrites(self):
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        Path("obito.json").write_text("{}", encoding="utf-8")
        out = io.StringIO()
        self.assertEqual(main(["--daten", self.tmp.name, "config", "--empfehlen", "16"], out=out), 0)
        self.assertEqual(Path("obito.json").read_text(encoding="utf-8"), "{}")
        data = json.loads(Path("obito.empfohlen.json").read_text(encoding="utf-8"))
        self.assertEqual(data["model"], "qwen2.5:14b")
        self.assertIn("nicht überschrieben", out.getvalue())

    def test_show_and_write(self):
        out = io.StringIO()
        code = main(["--backend", "fake", "--daten", self.tmp.name, "--modell", "m7", "config"], out=out)
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Konfigurationsdatei: keine", text)
        shown = json.loads(text.split("\n", 1)[1])
        self.assertEqual(shown["model"], "m7")
        self.assertEqual(shown["backend"], "fake")
        self.assertEqual(shown["data_dir"], self.tmp.name)
        target = Path(self.tmp.name, "c.json")
        out = io.StringIO()
        self.assertEqual(main(["--daten", self.tmp.name, "config", "--schreiben", str(target)], out=out), 0)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["data_dir"], self.tmp.name)
        out = io.StringIO()
        self.assertEqual(main(["--config", str(target), "config"], out=out), 0)
        self.assertIn(f"Konfigurationsdatei: {target}", out.getvalue())

    def test_broken_config_file(self):
        bad = Path(self.tmp.name, "kaputt.json")
        bad.write_text("[1, 2]", encoding="utf-8")
        out = io.StringIO()
        self.assertEqual(main(["--config", str(bad), "config"], out=out), 2)
        self.assertIn("Konfiguration konnte nicht geladen werden", out.getvalue())


# ------------------------------------------------------------------ export-dataset
class ExportDatasetCommandTest(CliTestBase):
    def test_without_db(self):
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "export-dataset", "--ausgabe", str(Path(self.tmp.name, "d.jsonl"))],
                    out=out)
        self.assertEqual(code, 1)
        self.assertIn("Noch keine Interaktionen", out.getvalue())

    def test_end_to_end(self):
        cfg = self.make_cfg()
        cfg.ensure_dirs()
        store = LearningStore(cfg.learning_db)
        for i in range(3):
            iid = store.record("s", f"Frage Nummer {i} zu Wandstärken?", f"Antwort {i}: " + LONG_ANSWER)
            store.rate(iid, 1)
        iid = store.correct(store.record("s", "Frage mit Korrektur?", "Falsche Antwort mit genug Zeichen für alle."),
                            "Richtig wäre 2 mm", "Richtig wäre eine Wandstärke von 2,0 mm bei PETG.").id
        store.close()
        target = Path(self.tmp.name, "daten", "obito.jsonl")
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "export-dataset", "--ausgabe", str(target), "--eval-anteil", "0",
                     "--ohne-kontext"], out=out)
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Datensatz: 4 Trainings-, 0 Eval-Beispiele", text)
        self.assertIn("verworfen:", text)
        self.assertIn(f"{target}.fragen.jsonl", text)
        rows = [json.loads(l) for l in target.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[-1]["messages"][-1]["content"], "Richtig wäre eine Wandstärke von 2,0 mm bei PETG.")
        self.assertTrue(Path(str(target) + ".eval.jsonl").exists())
        fragen = [json.loads(l) for l in Path(str(target) + ".fragen.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(fragen), 4)
        self.assertEqual(fragen[-1]["quelle_id"], iid)
        # DPO: nur korrigierte
        out = io.StringIO()
        target2 = Path(self.tmp.name, "dpo.jsonl")
        self.assertEqual(main(["--daten", self.tmp.name, "export-dataset", "--ausgabe", str(target2), "--format", "dpo"],
                              out=out), 0)
        rows = [json.loads(l) for l in target2.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {"prompt", "chosen", "rejected"})


# ------------------------------------------------------------------ memory
class MemoryCommandTest(CliTestBase):
    def run_main(self, *args, backend=None) -> tuple[int, str]:
        out = io.StringIO()
        argv = ["--backend", "fake", "--daten", self.tmp.name, "memory", *args]
        code = main(argv, out=out, backend=backend)
        return code, out.getvalue()

    def test_list_without_db(self):
        code, text = self.run_main("list")
        self.assertEqual(code, 0)
        self.assertIn("Noch kein Gedächtnis", text)
        code, text = self.run_main("export", str(Path(self.tmp.name, "x.json")))
        self.assertEqual(code, 1)

    def test_no_action_prints_usage(self):
        code, text = self.run_main()
        self.assertEqual(code, 2)
        self.assertIn("python -m obito memory", text)

    def test_import_list_search_export_reindex(self):
        src = Path(self.tmp.name, "quelle.json")
        store = MemoryStore(":memory:")
        store.remember("Der Drucker steht im Keller und hat eine 0,4-mm-Düse", kind="fakt", tags=["drucker"])
        store.remember("Ich bevorzuge metrische Einheiten", kind="praeferenz")
        store.export(src)
        store.close()

        code, text = self.run_main("import", str(src), backend=FakeBackend(up=False))
        self.assertEqual(code, 0)
        self.assertIn("2 Erinnerung(en) aus", text)
        self.assertIn("Ohne Embedding-Server: keine Vektoren", text)

        code, text = self.run_main("list", "-n", "1")
        self.assertEqual(code, 0)
        self.assertIn("1 Erinnerung(en) von 2", text)
        self.assertIn("[praeferenz", text)

        code, text = self.run_main("search", "Düse Drucker")
        self.assertEqual(code, 0)
        self.assertIn("0,4-mm-Düse", text)
        self.assertIn("Score", text)
        code, text = self.run_main("search", "xyzzy")
        self.assertIn("Keine passenden Erinnerungen", text)

        dst = Path(self.tmp.name, "export.json")
        code, text = self.run_main("export", str(dst))
        self.assertEqual(code, 0)
        self.assertEqual(len(json.loads(dst.read_text(encoding="utf-8"))), 2)

        code, text = self.run_main("reindex", backend=FakeBackend(up=False))
        self.assertEqual(code, 1)
        self.assertIn("nicht erreichbar", text)
        code, text = self.run_main("reindex")
        self.assertEqual(code, 0)
        self.assertIn("2 Erinnerung(en) neu eingebettet", text)
        self.assertIn("2 mit und 0 ohne Vektor", text)
        code, text = self.run_main("reindex", "--alle")
        self.assertEqual(code, 0)
        self.assertIn("0 Erinnerung(en) neu eingebettet", text)

        code, text = self.run_main("import", str(Path(self.tmp.name, "fehlt.json")))
        self.assertEqual(code, 1)
        self.assertIn("nicht gefunden", text)
        bad = Path(self.tmp.name, "kaputt.json")
        bad.write_text('[{"falsch": 1}]', encoding="utf-8")
        code, text = self.run_main("import", str(bad))
        self.assertEqual(code, 1)
        self.assertIn("keine gültige Export-Datei", text)


# ------------------------------------------------------------------ eval
class EvalCommandTest(CliTestBase):
    ANSWER = "Carbon ist leichter und steifer als Aluminium, dämpft Vibrationen, ist aber teurer und spröde."

    def write_items(self, name="fragen.jsonl", n=2) -> Path:
        p = Path(self.tmp.name, name)
        rows = [{"frage": "Welche Vorteile hat ein Drohnenrahmen aus Carbon gegenüber Aluminium?",
                 "erwartet": "leichter, steifer, teurer", "stichworte": ["leichter", "steif", "teurer"], "quelle_id": 7},
                {"frage": "Was ist 2+2?", "stichworte": ["4"], "muss_zahl": "4"}]
        with p.open("w", encoding="utf-8") as fh:
            for r in rows[:n]:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    def make_backend(self) -> FakeBackend:
        script = default_script()
        script[("schnell", "OMEGA")] = self.ANSWER
        return FakeBackend(responder=scripted(script, default=self.ANSWER), embed_dim=32)

    def test_direct_eval_writes_report(self):
        items = self.write_items()
        report = Path(self.tmp.name, "bericht.json")
        fb = self.make_backend()
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "--modell", "fake-modell", "eval", "--datei", str(items), "--ausgabe",
                     str(report)], out=out, backend=fb)
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("Eval: 2 Frage(n)", text)
        self.assertIn("(direkter Modellaufruf)", text)
        self.assertIn("[1/2] Stichwort 100 %", text)
        self.assertIn("[2/2] Stichwort 0 %", text)
        self.assertIn("Stichwort-Score: 50 %", text)
        self.assertIn(f"Bericht gespeichert: {report}", text)
        data = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(data["anzahl"], 2)
        self.assertEqual(data["modell"], "fake-modell")
        self.assertEqual(len(fb.calls), 2)
        self.assertEqual(fb.calls[0]["messages"][0]["role"], "system")       # Standard-System-Prompt
        self.assertEqual(stage_of(fb.calls[0]["messages"]), ("?", "?"))

    def test_eval_through_brain(self):
        items = self.write_items(n=1)
        fb = self.make_backend()
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "eval", "--datei", str(items), "--tiefe", "schnell"], out=out, backend=fb)
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("über den Denkkern (Tiefe schnell)", text)
        self.assertIn("Tiefe: schnell", text)
        self.assertEqual([stage_of(c["messages"])[0] for c in fb.calls], ["schnell"])
        # learn=False: nichts protokolliert
        store = LearningStore(Config(data_dir=self.tmp.name).learning_db)
        self.assertEqual(store.stats()["interaktionen"], 0)
        store.close()

    def test_ohne_training_excludes_questions(self):
        items = self.write_items()
        train = Path(self.tmp.name, "train.jsonl")
        train.write_text(json.dumps({"messages": [
            {"role": "system", "content": "x"},
            {"role": "user", "content": "Kontext\n\nWelche Vorteile hat ein Drohnenrahmen aus Carbon gegenüber Aluminium?"},
            {"role": "assistant", "content": "y"}]}, ensure_ascii=False) + "\n"
            + json.dumps({"instruction": "Was ist 2+2?", "input": "", "output": "4"}) + "\n", encoding="utf-8")
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "eval", "--datei", str(items), "--ohne-training", str(train)],
                    out=out, backend=self.make_backend())
        text = out.getvalue()
        self.assertEqual(code, 1)                      # alles ausgeschlossen -> nichts bewertet
        self.assertIn("1 Frage(n) ohne quelle_id kommen in", text)
        self.assertIn("1 Frage(n) sind in den Trainingsdaten enthalten", text)
        self.assertIn("Fragen: 0", text)

    def test_vergleich(self):
        from obito.training.evaluate import save_report
        a = {"modell": "alt", "tiefe": None, "anzahl": 2, "ergebnisse": [
            {"frage": "F1", "antwort": "a", "stichwort": 0.5, "richter": None, "dauer": 1.0, "tokens": 10},
            {"frage": "F2", "antwort": "a", "stichwort": 0.5, "richter": None, "dauer": 1.0, "tokens": 10}]}
        b = {"modell": "neu", "tiefe": None, "anzahl": 2, "ergebnisse": [
            {"frage": "F1", "antwort": "b", "stichwort": 1.0, "richter": None, "dauer": 2.0, "tokens": 20},
            {"frage": "F2", "antwort": "b", "stichwort": 1.0, "richter": None, "dauer": 2.0, "tokens": 20}]}
        pa, pb = Path(self.tmp.name, "a.json"), Path(self.tmp.name, "b.json")
        save_report(a, pa)
        save_report(b, pb)
        out = io.StringIO()
        result = Path(self.tmp.name, "vergleich.json")
        code = main(["--daten", self.tmp.name, "eval", "--vergleich", str(pa), str(pb), "--ausgabe", str(result)], out=out)
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("Vergleich A = alt", text)
        self.assertIn("B = neu", text)
        self.assertIn("Stichwort-Delta (B − A): +50 %", text)
        self.assertIn("Hinweis:", text)
        self.assertTrue(result.exists())

    def test_errors(self):
        out = io.StringIO()
        self.assertEqual(main(["--daten", self.tmp.name, "eval"], out=out), 2)
        self.assertIn("--datei", out.getvalue())
        out = io.StringIO()
        self.assertEqual(main(["--daten", self.tmp.name, "eval", "--datei", str(Path(self.tmp.name, "fehlt.jsonl"))],
                              out=out), 1)
        self.assertIn("Fehler:", out.getvalue())
        out = io.StringIO()
        items = self.write_items()
        code = main(["--daten", self.tmp.name, "eval", "--datei", str(items)], out=out, backend=FakeBackend(up=False))
        self.assertEqual(code, 1)
        self.assertIn("nicht erreichbar", out.getvalue())


# ------------------------------------------------------------------ modelfile / train / serve
class ModelfileCommandTest(CliTestBase):
    def test_prints_modelfile(self):
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "modelfile", "--basis", "qwen2.5:7b"], out=out)
        text = out.getvalue()
        self.assertEqual(code, 0, text)
        self.assertIn("FROM qwen2.5:7b", text)
        self.assertIn("SYSTEM", text)
        self.assertIn("PARAMETER num_ctx 8192", text)
        self.assertIn("Erstellen mit: python -m obito modelfile --name obito-v1 --basis qwen2.5:7b --erstellen", text)

    def test_missing_adapter_and_invalid_base(self):
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "modelfile", "--basis", "qwen2.5:7b", "--adapter",
                     str(Path(self.tmp.name, "fehlt.gguf"))], out=out)
        self.assertEqual(code, 1)
        self.assertIn("Adapter", out.getvalue())
        gguf = Path(self.tmp.name, "irgendwas.q4.gguf")
        gguf.write_bytes(b"GGUF")
        out = io.StringIO()
        code = main(["--daten", self.tmp.name, "modelfile", "--basis", str(gguf)], out=out)
        self.assertEqual(code, 1)                      # GGUF ohne Vorlage und ohne „qwen“ im Namen
        self.assertIn("Fehler:", out.getvalue())

    def test_create_with_patched_ollama(self):
        from obito.training import modelfile as mf
        adapter = Path(self.tmp.name, "adapter.gguf")
        adapter.write_bytes(b"GGUF")
        (adapter.parent / "obito_training.json").write_text(json.dumps({"hf_base": "Qwen/Qwen2.5-3B-Instruct",
                                                                          "ollama_base": "qwen2.5:3b"}), encoding="utf-8")
        out = io.StringIO()
        with mock.patch.object(mf, "create_model") as create:
            code = main(["--daten", self.tmp.name, "modelfile", "--name", "obito-v2", "--basis", "qwen2.5:7b",
                         "--adapter", str(adapter), "--erstellen"], out=out)
            self.assertEqual(code, 1)                  # Adapter passt nicht zur Basis -> Abbruch
            self.assertIn("Warnung: Der Adapter wurde auf", out.getvalue())
            self.assertIn("--erzwingen", out.getvalue())
            create.assert_not_called()
        out = io.StringIO()
        with mock.patch.object(mf, "create_model", return_value=CompletedProcess([], 0, "", "")) as create:
            code = main(["--daten", self.tmp.name, "modelfile", "--name", "obito-v2", "--basis", "qwen2.5:3b",
                         "--adapter", str(adapter), "--erstellen"], out=out)
            self.assertEqual(code, 0, out.getvalue())
            self.assertIn("Modell »obito-v2« erstellt", out.getvalue())
            name, text = create.call_args[0]
            self.assertEqual(name, "obito-v2")
            self.assertIn("ADAPTER", text)
            self.assertEqual(create.call_args[1]["cwd"], str(adapter.parent.resolve()))
        out = io.StringIO()
        with mock.patch.object(mf, "create_model", return_value=CompletedProcess([], 1, "", "kaputt")):
            code = main(["--daten", self.tmp.name, "modelfile", "--basis", "qwen2.5:3b", "--erstellen"], out=out)
            self.assertEqual(code, 1)
            self.assertIn("kaputt", out.getvalue())


class TrainCommandTest(CliTestBase):
    def test_train_help_passes_through(self):
        out = io.StringIO()
        with mock.patch("sys.stdout", new=out):
            with self.assertRaises(SystemExit) as cm:
                main(["train", "--help"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("--daten", out.getvalue())
        self.assertIn("--epochen", out.getvalue())

    def test_train_delegates_argv(self):
        from obito.training import train_lora
        with mock.patch.object(train_lora, "main", return_value=0) as tm:
            code = main(["--daten", self.tmp.name, "train", "--daten", "x.jsonl", "--epochen", "2"], out=io.StringIO())
        self.assertEqual(code, 0)
        tm.assert_called_once_with(["--daten", "x.jsonl", "--epochen", "2"])


class ServeCommandTest(CliTestBase):
    def test_serve_starts_and_prints_url(self):
        from obito import server as server_mod
        seen: dict = {}

        def fake_forever(self):
            seen["port"] = self.port
            seen["allow"] = self.allow_dangerous

        out = io.StringIO()
        with mock.patch.object(server_mod.ObitoServer, "serve_forever", fake_forever):
            code = main(["--backend", "fake", "--daten", self.tmp.name, "serve", "--port", "0", "--gefaehrlich-erlauben"],
                        out=out)
        self.assertEqual(code, 0)
        self.assertGreater(seen["port"], 0)
        self.assertTrue(seen["allow"])
        text = out.getvalue()
        self.assertIn(f"OBITO-HUD: http://127.0.0.1:{seen['port']}/", text)
        self.assertIn("gefährliche Werkzeuge FREIGEGEBEN", text)

    def test_serve_port_in_use(self):
        import socket
        from obito import server as server_mod
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        self.addCleanup(sock.close)
        port = sock.getsockname()[1]
        out = io.StringIO()
        with mock.patch.object(server_mod.ObitoServer, "serve_forever", lambda self: None):
            with mock.patch.object(server_mod.ObitoServer, "allow_reuse_address", False):
                code = main(["--backend", "fake", "--daten", self.tmp.name, "serve", "--port", str(port)], out=out)
        self.assertEqual(code, 1)
        self.assertIn("kann nicht auf", out.getvalue())


# ------------------------------------------------------------------ chat über main / __main__
class MainChatTest(CliTestBase):
    def test_default_subcommand_is_chat(self):
        out = io.StringIO()
        inp = ScriptedInput(Q_SIMPLE, "/beenden")
        fb = FakeBackend(responder=scripted(default_script()), embed_dim=32)
        code = main(["--daten", self.tmp.name], out=out, inp=inp, backend=fb)
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("OBITO – lokaler KI-Assistent", text)
        self.assertIn("OBITO> " + SMALLTALK, text)
        self.assertIn("Bis bald.", text)
        self.assertTrue(Config(data_dir=self.tmp.name).memory_db.exists())

    def test_chat_options(self):
        out = io.StringIO()
        inp = ScriptedInput(Q_MEDIUM, "/beenden")
        fb = FakeBackend(responder=scripted(default_script()), embed_dim=32)
        code = main(["--daten", self.tmp.name, "chat", "--projekt", "Gehäuse", "--tiefe", "mittel", "--sitzung", "s9"],
                    out=out, inp=inp, backend=fb)
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertIn("Projekt Gehäuse", text)
        self.assertIn("Sitzung s9", text)
        self.assertIn("— mittel · ", text)
        store = LearningStore(Config(data_dir=self.tmp.name).learning_db)
        inter = store.last("s9", 1)[0]
        store.close()
        self.assertEqual((inter.project, inter.depth), ("Gehäuse", "mittel"))

    def test_module_entry_point(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OBITO_CONFIG=str(Path(self.tmp.name, "keine.json")))
        proc = subprocess.run([sys.executable, "-m", "obito", "--backend", "fake", "--daten", self.tmp.name, "config"],
                              capture_output=True, text=True, encoding="utf-8", env=env,
                              cwd=str(Path(__file__).resolve().parent.parent), timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"backend": "fake"', proc.stdout)
        proc = subprocess.run([sys.executable, "-m", "obito", "--help"], capture_output=True, text=True,
                              encoding="utf-8", env=env, cwd=str(Path(__file__).resolve().parent.parent), timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Unterbefehle", proc.stdout)


if __name__ == "__main__":
    unittest.main()
