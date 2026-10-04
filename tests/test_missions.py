"""Tests für obito/missions.py – mit FakeBackend und echtem Brain, ohne Modell, ohne Netzwerk.

Der Responder wird über ``agents.stage_of(messages)`` skriptet: ``mission`` (Plan-JSON),
``experte``/``synthese`` (Teilfragen über den mittleren Pfad), ``bericht`` (Abschlussbericht).
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest

from obito import agents
from obito.agents import OMEGA, stage_of
from obito.brain import Brain
from obito.config import Config
from obito.llm import FakeBackend
from obito.missions import (MAX_FAILED_STEPS, MAX_STEPS, MISSION_SCHEMA, Mission, MissionRunner, MissionStep,
                            MissionStore, mission_plan_messages, mission_report_messages, parse_mission)

GOAL = "Wähle das Rahmenmaterial für einen 600-g-Quadcopter und berechne die Rahmenmasse."
REPORT = "## Ziel\nRahmenmaterial wählen.\n## Ergebnisse je Schritt\n1. ok\n## Offene Punkte\n- keine\n## Empfehlung\nCFK."
EXPERT_TEXT = "Antwort von {who}: CFK ist bei 1,6 g/cm³ steif und leicht, Aluminium 6061 günstiger."
SYNTH_TEXT = "Empfehlung: CFK-Platten 2 mm für Arme, Aluminium nur für Standoffs; Masse etwa 110 g."

PLAN_JSON = json.dumps({
    "titel": "Rahmenmaterial Quadcopter",
    "schritte": [
        {"beschreibung": "Anforderungen an den Rahmen ableiten.", "art": "frage", "werkzeug": None, "args": {}},
        {"beschreibung": "Masse eines 40-cm³-Rahmens aus CFK berechnen.", "art": "werkzeug",
         "werkzeug": "rechnen", "args": {"ausdruck": "40*1.6"}},
        {"beschreibung": "Material empfehlen.", "art": "frage", "werkzeug": None, "args": {}},
    ],
}, ensure_ascii=False)


def _last_user(msgs: list[dict]) -> str:
    users = [m["content"] for m in msgs if m["role"] == "user"]
    return users[-1] if users else ""


def scripted(script: dict, default: str = "Allgemeine Antwort ohne Stufe."):
    """Responder: ``script[(stage, who)]`` → Text oder ``(msgs, kw) -> str``; ``(stage, "*")`` als Platzhalter."""

    def responder(msgs, kw):
        stage, who = stage_of(msgs)
        value = script.get((stage, who), script.get((stage, "*"), default))
        if callable(value):
            return value(msgs, kw)
        return value

    return responder


def default_script(plan: str = PLAN_JSON) -> dict:
    return {
        ("mission", "system"): plan,
        ("experte", "*"): lambda msgs, kw: EXPERT_TEXT.format(who=stage_of(msgs)[1]),
        ("synthese", OMEGA.id): SYNTH_TEXT,
        ("bericht", OMEGA.id): REPORT,
    }


class FakeProjects:
    """Minimaler Ersatz für projects.ProjectStore (Phase 2, parallel entwickelt)."""

    def __init__(self, summary_text: str = "Projekt »Drohne« (aktiv): 5-Zoll-Quadcopter, Ziel 600 g."):
        self.summary_text = summary_text
        self.ensured: list[str] = []
        self.notes: list[tuple] = []

    def ensure(self, name):
        self.ensured.append(name)
        return {"name": name}

    def summary(self, name, max_chars=800):
        return self.summary_text

    def add_note(self, name, kind, title, content=""):
        self.notes.append((name, kind, title, content))
        return {"id": len(self.notes), "art": kind}


# ------------------------------------------------------------------ Datenklassen
class DataclassTest(unittest.TestCase):
    def test_step_to_dict_keys_and_roundtrip(self):
        s = MissionStep(idx=2, description="Rechnen", kind="werkzeug", tool="rechnen", args={"ausdruck": "1+1"},
                        status="fertig", result="2", duration=0.12345)
        d = s.to_dict()
        self.assertEqual(list(d), ["nr", "beschreibung", "art", "werkzeug", "args", "status", "ergebnis", "dauer"])
        self.assertEqual(d["dauer"], 0.123)
        back = MissionStep.from_dict(d)
        self.assertEqual((back.idx, back.kind, back.tool, back.args, back.status, back.result),
                         (2, "werkzeug", "rechnen", {"ausdruck": "1+1"}, "fertig", "2"))
        # kaputte Eingaben
        junk = MissionStep.from_dict({"nr": "x", "art": "unsinn", "status": "egal", "args": "nein"}, idx=4)
        self.assertEqual((junk.idx, junk.kind, junk.status, junk.args, junk.tool), (4, "frage", "offen", {}, None))
        self.assertEqual(MissionStep.from_dict("kein dict", idx=1).description, "")

    def test_mission_progress_and_to_dict(self):
        m = Mission(id=None, title="T", goal="G", project=None)
        self.assertEqual(m.progress, 0.0)
        m.steps = [MissionStep(1, "a", status="fertig"), MissionStep(2, "b", status="fehler"),
                   MissionStep(3, "c"), MissionStep(4, "d", status="fertig")]
        self.assertAlmostEqual(m.progress, 0.5)
        self.assertEqual(m.failed_steps(), 1)
        d = m.to_dict()
        self.assertEqual(list(d), ["id", "titel", "ziel", "projekt", "status", "fortschritt", "schritte", "bericht",
                                   "fehler", "erstellt", "geaendert"])
        self.assertEqual(d["fortschritt"], 0.5)
        self.assertEqual(len(d["schritte"]), 4)
        self.assertIsNone(d["erstellt"])
        m.created_at = m.updated_at = time.time()
        self.assertRegex(m.to_dict()["erstellt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


# ------------------------------------------------------------------ Speicher
class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "missionen.db")
        self.store = MissionStore(self.path)
        self.addCleanup(self.store.close)

    def make(self, goal="Ziel A", status="geplant", **kw) -> Mission:
        steps = kw.pop("steps", [MissionStep(1, "Schritt eins"), MissionStep(2, "Schritt zwei", kind="werkzeug",
                                                                             tool="rechnen", args={"ausdruck": "2*3"})])
        return Mission(id=None, title=kw.pop("title", ""), goal=goal, project=kw.pop("project", None),
                       status=status, steps=steps, **kw)

    def test_save_insert_sets_id_and_times_and_title_fallback(self):
        m = self.store.save(self.make(goal="  Ein   sehr langes Ziel " + "x" * 100))
        self.assertEqual(m.id, 1)
        self.assertGreater(m.created_at, 0)
        self.assertGreaterEqual(m.updated_at, m.created_at)
        self.assertTrue(m.title.startswith("Ein sehr langes Ziel"))
        self.assertTrue(m.title.endswith(agents.CLIP_SUFFIX))
        loaded = self.store.get(1)
        self.assertEqual(loaded.goal, m.goal)
        self.assertEqual([s.to_dict() for s in loaded.steps], [s.to_dict() for s in m.steps])
        self.assertEqual(loaded.steps[1].args, {"ausdruck": "2*3"})

    def test_save_upserts(self):
        m = self.store.save(self.make())
        m.status = "laeuft"
        m.steps[0].status = "fertig"
        m.steps[0].result = "Ergebnis"
        m.report = "B"
        before = m.updated_at
        time.sleep(0.01)
        self.store.save(m)
        self.assertEqual(self.store.count(), 1)
        loaded = self.store.get(m.id)
        self.assertEqual(loaded.status, "laeuft")
        self.assertEqual(loaded.steps[0].result, "Ergebnis")
        self.assertEqual(loaded.report, "B")
        self.assertGreater(loaded.updated_at, before)
        # gelöschte ID wird unter derselben ID neu angelegt
        self.store.delete(m.id)
        self.store.save(m)
        self.assertEqual(self.store.get(m.id).goal, m.goal)

    def test_save_validation(self):
        with self.assertRaises(ValueError):
            self.store.save(self.make(goal="   "))
        with self.assertRaises(ValueError):
            self.store.save(self.make(status="kaputt"))
        with self.assertRaises(TypeError):
            self.store.save("keine Mission")  # type: ignore[arg-type]

    def test_list_order_filter_limit(self):
        a = self.store.save(self.make(goal="A", status="fertig"))
        time.sleep(0.01)
        b = self.store.save(self.make(goal="B", status="geplant"))
        time.sleep(0.01)
        c = self.store.save(self.make(goal="C", status="fertig"))
        self.assertEqual([m.id for m in self.store.list()], [c.id, b.id, a.id])
        self.assertEqual([m.id for m in self.store.list(status="fertig")], [c.id, a.id])
        self.assertEqual([m.id for m in self.store.list(limit=1)], [c.id])
        self.assertEqual(self.store.list(status="abgebrochen"), [])
        self.assertEqual(self.store.count("fertig"), 2)
        self.assertEqual(self.store.stats(), {"missionen": 3, "nach_status": {"fertig": 2, "geplant": 1}, "aktiv": 0})

    def test_delete_get_unknown(self):
        m = self.store.save(self.make())
        self.assertTrue(self.store.delete(m.id))
        self.assertFalse(self.store.delete(m.id))
        self.assertIsNone(self.store.get(m.id))
        self.assertIsNone(self.store.get(999))

    def test_search(self):
        self.store.save(self.make(goal="Drohnenrahmen aus CFK auslegen", title="Rahmen"))
        self.store.save(self.make(goal="Netzteil für den 3D-Drucker wählen", title="Netzteil"))
        hits = self.store.search("drohne rahmen")
        self.assertEqual([m.title for m in hits], ["Rahmen"])
        self.assertEqual(self.store.search(""), [])

    def test_persistence_and_close_idempotent(self):
        m = self.store.save(self.make(goal="Bleibt", project="Drohne", report="Bericht", error="E"))
        self.store.close()
        self.store.close()
        again = MissionStore(self.path)
        self.addCleanup(again.close)
        loaded = again.get(m.id)
        self.assertEqual((loaded.goal, loaded.project, loaded.report, loaded.error), ("Bleibt", "Drohne", "Bericht", "E"))
        self.assertEqual(len(loaded.steps), 2)
        with self.assertRaises(RuntimeError):
            self.store.get(1)

    def test_broken_steps_json_is_tolerated(self):
        m = self.store.save(self.make())
        self.store._db.execute("UPDATE missions SET steps = ?, status = ? WHERE id = ?", ("kein json", "wild", m.id))
        self.store._db.commit()
        loaded = self.store.get(m.id)
        self.assertEqual(loaded.steps, [])
        self.assertEqual(loaded.status, "geplant")


# ------------------------------------------------------------------ Prompts
class PromptTest(unittest.TestCase):
    def test_plan_messages(self):
        msgs = mission_plan_messages(GOAL, "Projekt »Drohne« (aktiv): 600 g", "Verfügbare Werkzeuge:\n- rechnen(ausdruck)")
        self.assertEqual(stage_of(msgs), ("mission", "system"))
        self.assertEqual(msgs[0]["role"], "system")
        self.assertTrue(msgs[0]["content"].endswith(agents.STAGE_TAG.format(stage="mission", who="system")))
        self.assertIn("- rechnen(ausdruck)", msgs[0]["content"])
        self.assertIn(f"Höchstens {MAX_STEPS} Schritte", msgs[0]["content"])
        self.assertIn("Bevorzuge „frage“", msgs[0]["content"])
        user = _last_user(msgs)
        self.assertIn(GOAL, user)
        self.assertIn("Projektkontext:\nProjekt »Drohne«", user)
        self.assertIn("Antworte nur mit JSON.", user)
        example = json.loads(user.split("Beispiel:\n", 1)[1])
        self.assertIn("schritte", example)
        self.assertLessEqual(len(example["schritte"]), MAX_STEPS)
        # ohne Werkzeuge/Projekt
        msgs2 = mission_plan_messages(GOAL, "", "")
        self.assertIn("keine Werkzeuge verfügbar", msgs2[0]["content"])
        self.assertNotIn("Projektkontext", _last_user(msgs2))

    def test_report_messages(self):
        steps = [MissionStep(1, "Anforderungen", status="fertig", result="Steif, leicht, " + "x" * 2000),
                 MissionStep(2, "Rechnen", kind="werkzeug", tool="rechnen", status="fehler", result="Nicht erlaubt"),
                 MissionStep(3, "Empfehlen", status="uebersprungen")]
        msgs = mission_report_messages(GOAL, steps)
        self.assertEqual(stage_of(msgs), ("bericht", "OMEGA"))
        system = msgs[0]["content"]
        for part in ("## Ziel", "## Ergebnisse je Schritt", "## Offene Punkte", "## Empfehlung"):
            self.assertIn(part, system)
        user = _last_user(msgs)
        self.assertIn(GOAL, user)
        self.assertIn("1. [frage] Anforderungen – erledigt", user)
        self.assertIn("2. [werkzeug rechnen] Rechnen – fehlgeschlagen", user)
        self.assertIn("3. [frage] Empfehlen – übersprungen", user)
        self.assertIn(agents.CLIP_SUFFIX, user)           # langes Ergebnis gekürzt
        self.assertLess(len(user), 2500)
        self.assertIn("(keine Schritte ausgeführt)", _last_user(mission_report_messages(GOAL, [])))

    def test_schema_shape(self):
        self.assertEqual(MISSION_SCHEMA["required"], ["titel", "schritte"])
        item = MISSION_SCHEMA["properties"]["schritte"]["items"]
        self.assertEqual(item["properties"]["art"]["enum"], ["frage", "werkzeug"])
        json.dumps(MISSION_SCHEMA)


# ------------------------------------------------------------------ Normalisierer
class ParseMissionTest(unittest.TestCase):
    KNOWN = ["rechnen", "zeit", "datei_lesen"]

    def test_valid_plan(self):
        p = parse_mission(PLAN_JSON, self.KNOWN)
        self.assertEqual(p["titel"], "Rahmenmaterial Quadcopter")
        self.assertEqual(len(p["schritte"]), 3)
        self.assertEqual(p["schritte"][1], {"beschreibung": "Masse eines 40-cm³-Rahmens aus CFK berechnen.",
                                            "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "40*1.6"}})
        self.assertEqual(p["schritte"][0]["werkzeug"], None)
        self.assertEqual(set(p["schritte"][0]), {"beschreibung", "art", "werkzeug", "args"})

    def test_garbage_and_empty(self):
        self.assertIsNone(parse_mission("Ich bin kein JSON.", self.KNOWN))
        self.assertIsNone(parse_mission("", self.KNOWN))
        self.assertIsNone(parse_mission(None, self.KNOWN))
        self.assertIsNone(parse_mission('{"titel": "x", "schritte": []}', self.KNOWN))
        self.assertIsNone(parse_mission('{"foo": 1}', self.KNOWN))
        self.assertIsNone(parse_mission("42", self.KNOWN))

    def test_tolerant_keys_fence_and_unknown_tool(self):
        text = ('Hier der Plan:\n```json\n{"Titel": "Plan", "Schritte": ['
                '{"Beschreibung": "Datum holen", "Art": "Werkzeug", "Werkzeug": "Zeit", "Args": {}},'
                '{"beschreibung": "Wetter holen", "art": "werkzeug", "werkzeug": "wetter_api", "args": {"ort": "Berlin"}},'
                '{"beschreibung": "Rechnen", "art": "werkzeug", "werkzeug": "rechnen", "args": "kein dict"},'
                '{"beschreibung": "Nur Text", "art": "frage"},'
                '"Ein nackter String als Schritt",'
                '{"art": "frage"},'
                '{"beschreibung": "Rechnen 2", "art": "werkzeug", "werkzeug": "rechnen", "args": "{\\"ausdruck\\": \\"2*3\\"}"}'
                ']}\n```')
        p = parse_mission(text, self.KNOWN)
        self.assertEqual(p["titel"], "Plan")
        s = p["schritte"]
        self.assertEqual(len(s), 6)                           # Schritt ohne Beschreibung verworfen
        self.assertEqual((s[0]["art"], s[0]["werkzeug"]), ("werkzeug", "zeit"))          # Name normalisiert
        self.assertEqual((s[1]["art"], s[1]["werkzeug"], s[1]["args"]), ("frage", None, {}))  # unbekannt → frage
        self.assertEqual((s[2]["art"], s[2]["werkzeug"], s[2]["args"]), ("werkzeug", "rechnen", {}))  # args kein dict
        self.assertEqual((s[3]["art"], s[3]["werkzeug"]), ("frage", None))
        self.assertEqual(s[4], {"beschreibung": "Ein nackter String als Schritt", "art": "frage", "werkzeug": None,
                                "args": {}})
        self.assertEqual(s[5]["args"], {"ausdruck": "2*3"})  # args als JSON-String akzeptiert

    def test_max_steps_and_title_fallback(self):
        steps = [{"beschreibung": f"Schritt {i}", "art": "frage"} for i in range(12)]
        p = parse_mission(json.dumps({"schritte": steps}), self.KNOWN, goal="Z" * 100)
        self.assertEqual(len(p["schritte"]), MAX_STEPS)
        self.assertEqual(p["titel"], "Z" * 60 + agents.CLIP_SUFFIX)
        self.assertEqual(len(parse_mission(json.dumps(steps), self.KNOWN, max_steps=3)["schritte"]), 3)
        self.assertEqual(parse_mission(json.dumps(steps), self.KNOWN)["titel"], "")

    def test_top_level_list_and_single_step(self):
        p = parse_mission('[{"beschreibung": "A"}, {"beschreibung": "B", "art": "werkzeug", "werkzeug": "rechnen",'
                          ' "args": {"ausdruck": "1"}}]', self.KNOWN)
        self.assertEqual([s["art"] for s in p["schritte"]], ["frage", "werkzeug"])
        single = parse_mission('{"beschreibung": "Nur einer", "art": "frage"}', self.KNOWN)
        self.assertEqual(len(single["schritte"]), 1)
        # Werkzeug ohne bekannte Werkzeuge → frage
        p2 = parse_mission(PLAN_JSON, [])
        self.assertTrue(all(s["art"] == "frage" for s in p2["schritte"]))


# ------------------------------------------------------------------ Runner
class RunnerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.confirm_calls: list[tuple] = []
        self.allow_dangerous = False

    def make_runner(self, script: dict | None = None, *, projects=None, runner_confirm=None, **cfg_kw) -> MissionRunner:
        kw = dict(backend="fake", data_dir=self.tmp.name, model="fake-modell", fast_model="", depth="auto",
                  workspace=self.tmp.name, parallel_calls=1, deadline=30.0, auto_memory=False)
        kw.update(cfg_kw)
        self.cfg = Config(**kw)
        self.backend = FakeBackend(responder=scripted(script if script is not None else default_script()), embed_dim=32)

        def confirm(name, args):
            self.confirm_calls.append((name, args))
            return self.allow_dangerous

        self.brain = Brain(self.cfg, backend=self.backend, confirm=confirm)
        self.addCleanup(self.brain.close)
        if projects is not None:
            self.brain.projects = projects
        self.store = MissionStore(os.path.join(self.tmp.name, "missionen.db"))
        self.addCleanup(self.store.close)
        return MissionRunner(self.brain, self.store, confirm=runner_confirm)

    def calls_for(self, stage: str) -> list[dict]:
        return [c for c in self.backend.calls if stage_of(c["messages"])[0] == stage]


class PlanTest(RunnerTestBase):
    def test_plan_parses_and_persists(self):
        runner = self.make_runner()
        m = runner.plan("  " + GOAL + "  ")
        self.assertEqual(m.id, 1)
        self.assertEqual(m.status, "geplant")
        self.assertEqual(m.goal, GOAL)
        self.assertEqual(m.title, "Rahmenmaterial Quadcopter")
        self.assertEqual([s.kind for s in m.steps], ["frage", "werkzeug", "frage"])
        self.assertEqual([s.idx for s in m.steps], [1, 2, 3])
        self.assertEqual(m.steps[1].tool, "rechnen")
        self.assertEqual(m.steps[1].args, {"ausdruck": "40*1.6"})
        self.assertTrue(all(s.status == "offen" for s in m.steps))
        self.assertEqual(m.progress, 0.0)
        self.assertEqual(m.error, "")
        self.assertEqual(self.store.get(1).to_dict(), m.to_dict())
        # Aufruf: Stufe mission, Schema, temperature 0, Werkzeugblock im Prompt
        calls = self.calls_for("mission")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["json_mode"], MISSION_SCHEMA)
        self.assertEqual(calls[0]["temperature"], 0.0)
        self.assertGreaterEqual(calls[0]["max_tokens"], 900)
        self.assertIn("rechnen(", calls[0]["messages"][0]["content"])
        self.assertIn(GOAL, _last_user(calls[0]["messages"]))

    def test_plan_garbage_falls_back_to_single_step(self):
        script = default_script(plan="Ich plane lieber in Prosa.")
        runner = self.make_runner(script)
        m = runner.plan(GOAL)
        self.assertEqual(len(m.steps), 1)
        self.assertEqual((m.steps[0].kind, m.steps[0].description, m.steps[0].tool), ("frage", GOAL, None))
        self.assertEqual(m.title, agents.clip(GOAL, 60))
        self.assertEqual(len(self.calls_for("mission")), 2)              # genau ein Retry
        self.assertIs(self.calls_for("mission")[1]["json_mode"], True)
        self.assertEqual(m.status, "geplant")

    def test_plan_unknown_tool_becomes_question_and_limits_steps(self):
        steps = [{"beschreibung": f"S{i}", "art": "werkzeug", "werkzeug": "zauberei", "args": {"x": 1}} for i in range(10)]
        runner = self.make_runner(default_script(plan=json.dumps({"titel": "T", "schritte": steps})))
        m = runner.plan(GOAL)
        self.assertEqual(len(m.steps), MAX_STEPS)
        self.assertTrue(all(s.kind == "frage" and s.tool is None and s.args == {} for s in m.steps))

    def test_plan_backend_down_falls_back(self):
        runner = self.make_runner()
        self.backend.up = False
        m = runner.plan(GOAL, project="Drohne")
        self.assertEqual(len(m.steps), 1)
        self.assertEqual(m.status, "geplant")
        self.assertEqual(m.project, "Drohne")
        self.assertIn("Planung ohne Modell", m.error)

    def test_plan_validation_and_project_summary(self):
        projects = FakeProjects()
        runner = self.make_runner(projects=projects)
        with self.assertRaises(ValueError):
            runner.plan("   ")
        m = runner.plan(GOAL, project="  Drohne ")
        self.assertEqual(m.project, "Drohne")
        self.assertEqual(projects.ensured, ["Drohne"])
        self.assertIn("Projektkontext:\nProjekt »Drohne«", _last_user(self.calls_for("mission")[0]["messages"]))

    def test_plan_without_tools_block(self):
        runner = self.make_runner(allow_tools=False)
        runner.plan(GOAL)
        self.assertIn("keine Werkzeuge verfügbar", self.calls_for("mission")[0]["messages"][0]["content"])


class RunTest(RunnerTestBase):
    def test_run_tool_and_questions(self):
        runner = self.make_runner()
        m = runner.plan(GOAL)
        seen: list[tuple[str, float]] = []
        result = runner.run(m.id, progress=lambda mm: seen.append((mm.status, mm.progress)))
        self.assertEqual(result.status, "fertig")
        self.assertEqual(result.progress, 1.0)
        self.assertEqual([s.status for s in result.steps], ["fertig", "fertig", "fertig"])
        self.assertEqual(result.steps[1].result, "40*1.6 = 64")            # rechnen-Ausgabe
        self.assertEqual(result.steps[0].result, SYNTH_TEXT)
        self.assertEqual(result.report, REPORT)
        self.assertEqual(result.error, "")
        self.assertTrue(all(s.duration >= 0 for s in result.steps))
        # Fortschritt nach jedem Schritt + Start/Ende
        self.assertIn(("laeuft", 0.0), seen)
        self.assertEqual([(s, round(p, 3)) for s, p in seen],
                         [("laeuft", 0.0), ("laeuft", 0.333), ("laeuft", 0.667), ("laeuft", 1.0), ("fertig", 1.0)])
        # Teilfragen über brain.ask: Stufe mittel, Sitzung mission-<id>, learn=False
        self.assertEqual(len(self.calls_for("synthese")), 2)
        self.assertEqual(len(self.calls_for("experte")), 2 * self.cfg.experts_medium)
        self.assertEqual(self.calls_for("routing"), [])                    # Tiefe vorgegeben
        self.assertEqual(self.calls_for("extraktion"), [])                 # learn=False
        self.assertEqual(self.brain.learning.stats()["interaktionen"], 0)  # keine Protokollierung
        history = self.brain.memory.history(f"mission-{m.id}", 20)
        self.assertEqual([h["role"] for h in history], ["user", "assistant", "user", "assistant"])
        # zweite Frage enthält die bisherigen Ergebnisse (inkl. Werkzeugergebnis)
        second = history[2]["content"]
        self.assertTrue(second.startswith("Material empfehlen."))
        self.assertIn("Bisherige Ergebnisse:", second)
        self.assertIn("Schritt 1 (Anforderungen an den Rahmen ableiten.) – Ergebnis: " + SYNTH_TEXT, second)
        self.assertIn("Schritt 2 (Masse eines 40-cm³-Rahmens aus CFK berechnen.) – Ergebnis: 40*1.6 = 64", second)
        self.assertNotIn("Bisherige Ergebnisse", history[0]["content"])
        # Bericht: Stufe bericht mit allen Schritten
        report_calls = self.calls_for("bericht")
        self.assertEqual(len(report_calls), 1)
        self.assertIn("Ergebnis: 40*1.6 = 64", _last_user(report_calls[0]["messages"]))
        self.assertEqual(report_calls[0]["max_tokens"], self.cfg.max_tokens_synthesis)
        # Persistenz
        loaded = self.store.get(m.id)
        self.assertEqual(loaded.to_dict(), result.to_dict())
        self.assertEqual(loaded.report, REPORT)

    def test_previous_results_are_clipped(self):
        long_answer = "Z" * 4000
        script = default_script()
        script[("synthese", OMEGA.id)] = long_answer
        runner = self.make_runner(script)
        m = runner.plan(GOAL)
        runner.run(m.id)
        history = self.brain.memory.history(f"mission-{m.id}", 20)
        block = history[2]["content"].split("Bisherige Ergebnisse:\n", 1)[1]
        self.assertLessEqual(len(block), 1500 + len(agents.CLIP_SUFFIX))
        self.assertIn("Schritt 2", block)                                 # jüngstes Ergebnis bleibt

    def test_single_failed_step_mission_still_finishes(self):
        plan = json.dumps({"titel": "T", "schritte": [
            {"beschreibung": "Kaputt rechnen", "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "import os"}},
            {"beschreibung": "Weiter fragen", "art": "frage", "werkzeug": None, "args": {}},
        ]})
        runner = self.make_runner(default_script(plan))
        m = runner.plan(GOAL)
        result = runner.run(m.id)
        self.assertEqual(result.status, "fertig")
        self.assertEqual([s.status for s in result.steps], ["fehler", "fertig"])
        self.assertTrue(result.steps[0].result)
        self.assertAlmostEqual(result.progress, 0.5)
        self.assertEqual(result.report, REPORT)
        # Fehler steht als „Fehler“ in den bisherigen Ergebnissen der nächsten Frage
        history = self.brain.memory.history(f"mission-{m.id}", 20)
        self.assertIn("Schritt 1 (Kaputt rechnen) – Fehler:", history[0]["content"])

    def test_max_failures_abort_mission(self):
        plan = json.dumps({"titel": "T", "schritte": [
            {"beschreibung": "Fehler 1", "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "x+"}},
            {"beschreibung": "Fehler 2", "art": "werkzeug", "werkzeug": "rechnen", "args": {}},
            {"beschreibung": "Kommt nicht mehr dran", "art": "frage", "werkzeug": None, "args": {}},
            {"beschreibung": "Auch nicht", "art": "werkzeug", "werkzeug": "zeit", "args": {}},
        ]})
        runner = self.make_runner(default_script(plan))
        m = runner.plan(GOAL)
        result = runner.run(m.id)
        self.assertEqual(MAX_FAILED_STEPS, 2)
        self.assertEqual(result.status, "fehler")
        self.assertEqual([s.status for s in result.steps], ["fehler", "fehler", "uebersprungen", "uebersprungen"])
        self.assertIn("2 Schritte fehlgeschlagen", result.error)
        self.assertEqual(result.steps[1].result, "Fehlende Parameter: ausdruck")
        self.assertEqual(self.calls_for("synthese"), [])
        self.assertEqual(result.report, REPORT)                            # Bericht trotzdem
        self.assertEqual(self.store.get(m.id).status, "fehler")

    def test_unknown_tool_and_model_error_count_as_failures(self):
        plan = json.dumps({"titel": "T", "schritte": [
            {"beschreibung": "Frage", "art": "frage", "werkzeug": None, "args": {}},
            {"beschreibung": "Nochmal", "art": "frage", "werkzeug": None, "args": {}},
        ]})
        script = default_script(plan)
        script[("experte", "*")] = ""                                      # alle Experten leer → Fehler-Answer
        runner = self.make_runner(script)
        m = runner.plan(GOAL)
        # Werkzeug nachträglich unbekannt machen
        m.steps[1].kind, m.steps[1].tool = "werkzeug", "gibt_es_nicht"
        self.store.save(m)
        result = runner.run(m.id)
        self.assertEqual(result.status, "fehler")
        self.assertEqual([s.status for s in result.steps], ["fehler", "fehler"])
        self.assertIn("Modellfehler", result.steps[0].result)
        self.assertIn("Unbekanntes Werkzeug »gibt_es_nicht«", result.steps[1].result)

    def test_dangerous_tool_needs_confirmation(self):
        plan = json.dumps({"titel": "T", "schritte": [
            {"beschreibung": "Datei schreiben", "art": "werkzeug", "werkzeug": "datei_schreiben",
             "args": {"pfad": "notiz.txt", "inhalt": "Hallo"}},
        ]})
        runner = self.make_runner(default_script(plan))
        m = runner.plan(GOAL)
        result = runner.run(m.id)
        self.assertEqual(result.steps[0].status, "fehler")
        self.assertEqual(result.steps[0].result, "Vom Nutzer abgelehnt")
        self.assertEqual(self.confirm_calls, [("datei_schreiben", {"pfad": "notiz.txt", "inhalt": "Hallo"})])
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "notiz.txt")))
        # mit Zustimmung
        self.allow_dangerous = True
        result = runner.run(m.id)
        self.assertEqual(result.steps[0].status, "fertig")
        self.assertIn("geschrieben: notiz.txt", result.steps[0].result)
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, "notiz.txt")))

    def test_runner_confirm_is_additional_gate(self):
        plan = json.dumps({"titel": "T", "schritte": [
            {"beschreibung": "Datei schreiben", "art": "werkzeug", "werkzeug": "datei_schreiben",
             "args": {"pfad": "b.txt", "inhalt": "x"}},
            {"beschreibung": "Rechnen", "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "1+1"}},
        ]})
        asked: list[str] = []
        runner = self.make_runner(default_script(plan), runner_confirm=lambda n, a: asked.append(n) and False)
        self.allow_dangerous = True
        result = runner.run(runner.plan(GOAL).id)
        self.assertEqual(asked, ["datei_schreiben"])                       # rechnen fragt nicht
        self.assertEqual(result.steps[0].result, "Vom Nutzer abgelehnt")
        self.assertEqual(self.confirm_calls, [])                           # Registry wurde gar nicht gefragt
        self.assertEqual(result.steps[1].result, "1+1 = 2")

    def test_cancel_between_steps(self):
        runner = self.make_runner()
        m = runner.plan(GOAL)
        cancel = threading.Event()

        def progress(mm: Mission) -> None:
            if mm.steps[0].status == "fertig":
                cancel.set()

        result = runner.run(m.id, progress=progress, cancel=cancel)
        self.assertEqual(result.status, "abgebrochen")
        self.assertEqual(result.error, "Abgebrochen")
        self.assertEqual([s.status for s in result.steps], ["fertig", "uebersprungen", "uebersprungen"])
        self.assertEqual(result.report, "")
        self.assertEqual(self.calls_for("bericht"), [])
        self.assertEqual(self.store.get(m.id).status, "abgebrochen")
        # Fortsetzen: fertiger Schritt bleibt, der Rest läuft
        result = runner.run(m.id)
        self.assertEqual(result.status, "fertig")
        self.assertEqual(len(self.calls_for("synthese")), 2)               # 1 vor Abbruch + 1 danach

    def test_cancel_already_set_before_start(self):
        runner = self.make_runner()
        m = runner.plan(GOAL)
        cancel = threading.Event()
        cancel.set()
        result = runner.run(m.id, cancel=cancel)
        self.assertEqual(result.status, "abgebrochen")
        self.assertTrue(all(s.status == "uebersprungen" for s in result.steps))
        self.assertEqual(self.calls_for("synthese"), [])

    def test_report_fallback_when_model_fails(self):
        plan = json.dumps({"titel": "T", "schritte": [
            {"beschreibung": "Rechnen", "art": "werkzeug", "werkzeug": "rechnen", "args": {"ausdruck": "6*7"}}]})
        script = default_script(plan)
        script[("bericht", OMEGA.id)] = ""
        runner = self.make_runner(script)
        result = runner.run(runner.plan(GOAL).id)
        self.assertEqual(result.status, "fertig")
        self.assertIn("## Ziel", result.report)
        self.assertIn("1. Rechnen – erledigt: 6*7 = 42", result.report)
        self.assertIn("ohne Modell", result.report)
        self.assertIn("leere Modellantwort", result.error)

    def test_report_stored_as_project_note(self):
        projects = FakeProjects()
        runner = self.make_runner(projects=projects)
        m = runner.plan(GOAL, project="Drohne")
        result = runner.run(m.id)
        self.assertEqual(result.status, "fertig")
        self.assertEqual(len(projects.notes), 1)
        name, kind, title, content = projects.notes[0]
        self.assertEqual((name, kind, title, content), ("Drohne", "ergebnis", "Rahmenmaterial Quadcopter", REPORT))
        # ohne Projekt keine Notiz
        m2 = runner.plan(GOAL)
        runner.run(m2.id)
        self.assertEqual(len(projects.notes), 1)

    def test_project_note_failure_is_not_fatal(self):
        class Broken(FakeProjects):
            def add_note(self, *a, **k):
                raise RuntimeError("Datenbank weg")

        runner = self.make_runner(projects=Broken())
        result = runner.run(runner.plan(GOAL, project="Drohne").id)
        self.assertEqual(result.status, "fertig")
        self.assertIn("Projektnotiz fehlgeschlagen: Datenbank weg", result.error)

    def test_run_unknown_and_rerun_resets(self):
        runner = self.make_runner()
        with self.assertRaises(ValueError):
            runner.run(999)
        m = runner.plan(GOAL)
        first = runner.run(m.id)
        self.assertEqual(first.status, "fertig")
        second = runner.run(m.id)                                           # erneut: fertige bleiben, Bericht neu
        self.assertEqual(second.status, "fertig")
        self.assertEqual(len(self.calls_for("bericht")), 2)
        self.assertEqual(len(self.calls_for("synthese")), 2)

    def test_progress_callback_errors_are_ignored(self):
        runner = self.make_runner()
        m = runner.plan(GOAL)

        def bad(_m):
            raise RuntimeError("HUD weg")

        self.assertEqual(runner.run(m.id, progress=bad).status, "fertig")


class BackgroundTest(RunnerTestBase):
    def test_start_runs_in_background_and_finishes(self):
        runner = self.make_runner()
        m = runner.plan(GOAL)
        seen: list[str] = []
        thread = runner.start(m.id, progress=lambda mm: seen.append(mm.status))
        self.assertTrue(thread.daemon)
        self.assertEqual(thread.name, f"obito-mission-{m.id}")
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(runner.running(), [])
        self.assertTrue(runner.join(m.id, timeout=1))
        loaded = self.store.get(m.id)
        self.assertEqual(loaded.status, "fertig")
        self.assertEqual(loaded.report, REPORT)
        self.assertEqual(seen[-1], "fertig")

    def test_start_refuses_double_start_and_unknown(self):
        release = threading.Event()
        script = default_script()
        script[("experte", "*")] = lambda msgs, kw: (release.wait(5), EXPERT_TEXT.format(who="X"))[1]
        runner = self.make_runner(script)
        m = runner.plan(GOAL)
        thread = runner.start(m.id)
        try:
            self.assertEqual(runner.running(), [m.id])
            with self.assertRaises(RuntimeError):
                runner.start(m.id)
            with self.assertRaises(RuntimeError):
                runner.run(m.id)
            with self.assertRaises(ValueError):
                runner.start(999)
            self.assertEqual(self.store.get(m.id).status, "laeuft")
        finally:
            release.set()
            thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.get(m.id).status, "fertig")
        # danach darf sie erneut gestartet werden
        thread2 = runner.start(m.id)
        thread2.join(timeout=10)
        self.assertFalse(thread2.is_alive())

    def test_stop_cancels_background_run(self):
        started = threading.Event()
        release = threading.Event()
        script = default_script()

        def slow_expert(msgs, kw):
            started.set()
            release.wait(5)
            return EXPERT_TEXT.format(who=stage_of(msgs)[1])

        script[("experte", "*")] = slow_expert
        runner = self.make_runner(script)
        m = runner.plan(GOAL)
        thread = runner.start(m.id)
        self.assertTrue(started.wait(5))
        self.assertEqual(runner.running(), [m.id])
        self.assertTrue(runner.stop(m.id))                                  # setzt cancel
        release.set()                                                       # Experte liefert, Streaming sieht cancel
        thread.join(timeout=10)
        self.assertFalse(thread.is_alive())
        loaded = self.store.get(m.id)
        self.assertEqual(loaded.status, "abgebrochen")
        self.assertEqual(loaded.error, "Abgebrochen")
        self.assertEqual(loaded.steps[0].status, "fehler")                  # Frage brach ab
        self.assertEqual([s.status for s in loaded.steps[1:]], ["uebersprungen", "uebersprungen"])
        self.assertEqual(loaded.report, "")
        self.assertEqual(runner.running(), [])
        # stop auf fertiger/abgebrochener Mission: nichts zu tun
        self.assertFalse(runner.stop(m.id))
        self.assertFalse(runner.stop(999))

    def test_stop_marks_planned_mission_without_thread(self):
        runner = self.make_runner()
        m = runner.plan(GOAL)
        self.assertTrue(runner.stop(m.id))
        loaded = self.store.get(m.id)
        self.assertEqual(loaded.status, "abgebrochen")
        self.assertTrue(all(s.status == "uebersprungen" for s in loaded.steps))
        self.assertFalse(runner.stop(m.id))

    def test_two_missions_in_parallel_threads(self):
        runner = self.make_runner()
        a = runner.plan(GOAL)
        b = runner.plan("Zweites Ziel: Netzteil auslegen.")
        ta = runner.start(a.id)
        tb = runner.start(b.id)
        self.assertEqual(sorted(runner.running()), sorted([a.id, b.id]))
        ta.join(timeout=10)
        tb.join(timeout=10)
        self.assertEqual(runner.running(), [])
        self.assertEqual({self.store.get(a.id).status, self.store.get(b.id).status}, {"fertig"})


if __name__ == "__main__":
    unittest.main()
