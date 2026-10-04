"""Tests für obito/automation.py – ohne Modell, ohne Netzwerk, ohne echte Wartezeiten.

Der Denkkern wird durch einen Stellvertreter (``StubBrain``) ersetzt, der die Aufrufe
protokolliert; der Eval-Lauf nutzt einen echten ``FakeBackend``.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from obito.automation import (DEFAULT_EVAL_FILE, KINDS, KIND_LABELS, Automation, AutomationStore, Scheduler,
                              default_automations, install_defaults)
from obito.config import Config
from obito.llm import FakeBackend
from obito.tools import default_registry

T0 = 1_700_000_000.0        # fester Zeitpunkt für injizierte Zeitstempel


def wait_until(cond, timeout: float = 3.0, step: float = 0.01) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(step)
    return cond()


# ------------------------------------------------------------------ Stellvertreter
class StubMission:
    def __init__(self, mid: int, title: str, status: str = "fertig"):
        self.id = mid
        self.title = title
        self.status = status
        self.steps = [SimpleNamespace(status="fertig"), SimpleNamespace(status="fertig"),
                      SimpleNamespace(status="fehler" if status != "fertig" else "fertig")]
        self.report = "Bericht: Rahmen aus CFK, 5 Zoll Propeller."
        self.error = "" if status == "fertig" else "Schritt 3 gescheitert"


class StubMissions:
    def __init__(self, result_status: str = "fertig"):
        self.result_status = result_status
        self.calls: list[tuple] = []

    def plan(self, goal, *, project=None):
        self.calls.append(("plan", goal, project))
        return StubMission(7, f"Plan: {goal}", self.result_status)

    def run(self, mid, **kw):
        self.calls.append(("run", mid))
        return StubMission(mid, "Drohne bauen", self.result_status)


class StubBrain:
    """Stellvertreter für ``Brain`` mit allen Fähigkeiten, die Automationen brauchen."""

    def __init__(self, tmp: str, *, responder=None):
        self.cfg = Config(backend="fake", data_dir=tmp, model="fake-modell", workspace=tmp, num_ctx=4096)
        self.cfg.ensure_dirs()
        self.backend = FakeBackend(responder=responder, embed_dim=16)
        self.tools = default_registry(tmp, confirm=None, confirm_dangerous=True)
        self.calls: list[tuple] = []
        self.knowledge = SimpleNamespace(sync=self._sync)
        self.missions = StubMissions()
        self.fail_with: Exception | None = None

    def consolidate(self, days=7, project=None):
        self.calls.append(("consolidate", days, project))
        if self.fail_with is not None:
            raise self.fail_with
        return {"zusammengefasst": 2, "geloescht": 5, "sitzungen": 1}

    def backup(self, keep=7):
        self.calls.append(("backup", keep))
        p = Path(self.cfg.data_path) / "backups" / "20250101-1200"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def _sync(self):
        self.calls.append(("sync",))
        return {"aktualisiert": 3, "fehlend": ["a.md"], "unveraendert": 10}


class BareBrain:
    """Denkkern ohne Phase-2-Fähigkeiten (noch nicht verdrahtet)."""


def eval_answer(msgs, kw):
    """Antwort mit den erwarteten Stichworten der Test-Eval-Datei."""
    user = [m["content"] for m in msgs if m["role"] == "user"][-1]
    if "Akku" in user:
        return "4S bedeutet 14,8 V Nennspannung; 1,5 Ah ergeben 22,2 Wh."
    return "Carbon ist leichter und steifer, aber teurer und spröde."


def write_eval_file(path: Path) -> Path:
    items = [
        {"frage": "Wie viel Energie hat ein 4S-Akku mit 1500 mAh?", "erwartet": "22,2 Wh",
         "stichworte": ["14,8", "22", "Wh"]},
        {"frage": "Vorteile von Carbon gegenüber Aluminium?", "stichworte": ["leichter", "steifer", "teurer"]},
    ]
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items) + "\n", encoding="utf-8")
    return path


# ------------------------------------------------------------------ Datenklasse
class AutomationDataclassTest(unittest.TestCase):
    def test_to_dict_keys_and_iso(self):
        a = Automation(id=3, name="Backup", kind="backup", interval_minutes=1440, params={"behalten": 7},
                       enabled=True, last_run=T0, last_status="ok", last_message="gut", next_run=T0 + 86400,
                       created_at=T0 - 10)
        d = a.to_dict()
        self.assertEqual(list(d), ["id", "name", "art", "intervall_minuten", "parameter", "aktiv", "letzter_lauf",
                                   "letzter_status", "letzte_meldung", "naechster_lauf", "erstellt"])
        self.assertEqual(d["art"], "backup")
        self.assertEqual(d["parameter"], {"behalten": 7})
        self.assertTrue(d["aktiv"])
        self.assertEqual(d["letzter_lauf"], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(T0)))
        self.assertEqual(d["naechster_lauf"], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(T0 + 86400)))
        self.assertIsNot(d["parameter"], a.params)

    def test_to_dict_none_timestamps_and_is_due(self):
        a = Automation(id=1, name="x", kind="eval", interval_minutes=5, params={}, enabled=True, last_run=None,
                       last_status="", last_message="", next_run=None, created_at=T0)
        d = a.to_dict()
        self.assertIsNone(d["letzter_lauf"])
        self.assertIsNone(d["naechster_lauf"])
        self.assertFalse(a.is_due(T0))
        a.next_run = T0 + 10
        self.assertFalse(a.is_due(T0))
        self.assertTrue(a.is_due(T0 + 10))
        a.enabled = False
        self.assertFalse(a.is_due(T0 + 100))
        self.assertEqual(a.interval_seconds, 300.0)

    def test_kinds_and_labels(self):
        self.assertEqual(KINDS, ("gedaechtnis_konsolidieren", "backup", "eval", "wissen_sync", "mission", "werkzeug"))
        self.assertEqual(set(KIND_LABELS), set(KINDS))


# ------------------------------------------------------------------ Speicher
class StoreTest(unittest.TestCase):
    def setUp(self):
        self.store = AutomationStore(":memory:")
        self.addCleanup(self.store.close)

    def test_create_defaults_first_run_due_immediately(self):
        a = self.store.create("Backup täglich", "backup", 1440, {"behalten": 3}, now=T0)
        self.assertEqual(a.id, 1)
        self.assertEqual((a.name, a.kind, a.interval_minutes), ("Backup täglich", "backup", 1440))
        self.assertEqual(a.params, {"behalten": 3})
        self.assertTrue(a.enabled)
        self.assertIsNone(a.last_run)
        self.assertEqual((a.last_status, a.last_message), ("", ""))
        self.assertEqual(a.next_run, T0)
        self.assertEqual(a.created_at, T0)
        self.assertEqual([x.id for x in self.store.due(T0)], [1])
        self.assertEqual(self.store.due(T0 - 1), [])

    def test_create_validation(self):
        with self.assertRaises(ValueError):
            self.store.create("x", "unbekannt", 10)
        with self.assertRaises(ValueError):
            self.store.create("x", "backup", 0)
        with self.assertRaises(ValueError):
            self.store.create("x", "backup", "abc")
        with self.assertRaises(ValueError):
            self.store.create("x", "backup", True)
        with self.assertRaises(ValueError):
            self.store.create("   ", "backup", 10)
        with self.assertRaises(ValueError):
            self.store.create("x", "backup", 10, params=["liste"])
        with self.assertRaises(ValueError):
            self.store.create("x", "backup", 10, params={"obj": object()})
        self.assertEqual(self.store.list(), [])

    def test_create_normalizes(self):
        a = self.store.create("  Sync\n stündlich ", " Wissen_Sync ", "60", None, enabled=False)
        self.assertEqual(a.name, "Sync stündlich")
        self.assertEqual(a.kind, "wissen_sync")
        self.assertEqual(a.interval_minutes, 60)
        self.assertEqual(a.params, {})
        self.assertFalse(a.enabled)

    def test_get_list_delete(self):
        a = self.store.create("A", "backup", 10, now=T0)
        b = self.store.create("B", "eval", 20, now=T0)
        self.assertEqual([x.id for x in self.store.list()], [a.id, b.id])
        self.assertEqual(self.store.get(b.id).name, "B")
        self.assertIsNone(self.store.get(99))
        self.store.record_run(a.id, "ok", "lief", now=T0 + 1)
        self.assertEqual(len(self.store.runs(a.id)), 1)
        self.assertTrue(self.store.delete(a.id))
        self.assertFalse(self.store.delete(a.id))
        self.assertIsNone(self.store.get(a.id))
        self.assertEqual(self.store.runs(a.id), [])
        self.assertEqual([x.id for x in self.store.list()], [b.id])

    def test_update_fields(self):
        a = self.store.create("A", "backup", 10, {"behalten": 7}, now=T0)
        u = self.store.update(a.id, name="Neu", params={"behalten": 2}, enabled=False)
        self.assertEqual((u.name, u.params, u.enabled), ("Neu", {"behalten": 2}, False))
        self.assertEqual(u.next_run, T0)                    # kein Lauf bisher -> unverändert
        u = self.store.update(a.id, interval_minutes=30)
        self.assertEqual(u.interval_minutes, 30)
        self.assertEqual(u.next_run, T0)                    # ohne last_run bleibt next_run
        self.store.record_run(a.id, "ok", "x", now=T0 + 100)
        u = self.store.update(a.id, interval_minutes=5)
        self.assertEqual(u.next_run, T0 + 100 + 300)        # letzter Lauf + neues Intervall
        self.assertEqual(self.store.update(a.id).id, a.id)  # ohne Felder: unverändert
        self.assertEqual(self.store.due(T0 + 1000), [])    # weiterhin inaktiv
        self.store.update(a.id, enabled=True)
        self.assertEqual([x.id for x in self.store.due(T0 + 1000)], [a.id])

    def test_update_validation(self):
        a = self.store.create("A", "backup", 10)
        with self.assertRaises(ValueError):
            self.store.update(a.id, kind="eval")
        with self.assertRaises(ValueError):
            self.store.update(a.id, interval_minutes=0)
        with self.assertRaises(ValueError):
            self.store.update(a.id, name="")
        with self.assertRaises(ValueError):
            self.store.update(a.id, params="nein")
        with self.assertRaises(ValueError):
            self.store.update(42, name="x")
        self.assertEqual(self.store.get(a.id).name, "A")

    def test_due_logic_with_injected_timestamps(self):
        a = self.store.create("A", "backup", 60, now=T0)
        b = self.store.create("B", "eval", 10, now=T0 + 5)
        c = self.store.create("C", "wissen_sync", 10, enabled=False, now=T0)
        self.assertEqual([x.id for x in self.store.due(T0)], [a.id])
        self.assertEqual([x.id for x in self.store.due(T0 + 5)], [a.id, b.id])
        self.store.record_run(a.id, "ok", "lief", now=T0 + 10)
        self.assertEqual(self.store.get(a.id).next_run, T0 + 10 + 3600)
        self.assertEqual([x.id for x in self.store.due(T0 + 10)], [b.id])
        self.assertEqual([x.id for x in self.store.due(T0 + 10 + 3599)], [b.id])
        self.store.record_run(b.id, "fehler", "kaputt", now=T0 + 20)
        self.assertEqual(self.store.due(T0 + 20), [])
        # Reihenfolge: älteste Fälligkeit zuerst (b: T0+620, a: T0+3610)
        ids = [x.id for x in self.store.due(T0 + 4000)]
        self.assertEqual(ids, [b.id, a.id])
        self.assertNotIn(c.id, ids)

    def test_record_run_and_runs(self):
        a = self.store.create("A", "backup", 15, now=T0)
        self.store.record_run(a.id, "ok", "Backup erstellt:\n  /tmp/x", started=T0 + 1, now=T0 + 3.5)
        self.store.record_run(a.id, "komisch", "?", now=T0 + 10)
        got = self.store.get(a.id)
        self.assertEqual(got.last_run, T0 + 10)
        self.assertEqual(got.last_status, "fehler")          # unbekannter Status -> fehler
        self.assertEqual(got.next_run, T0 + 10 + 900)
        runs = self.store.runs(a.id)
        self.assertEqual(len(runs), 2)
        self.assertEqual(list(runs[0]), ["id", "automation_id", "gestartet", "beendet", "dauer", "status", "meldung"])
        self.assertEqual(runs[0]["status"], "fehler")         # neueste zuerst
        self.assertEqual(runs[1]["status"], "ok")
        self.assertEqual(runs[1]["meldung"], "Backup erstellt: /tmp/x")   # eine Zeile
        self.assertEqual(runs[1]["dauer"], 2.5)
        self.assertEqual(runs[1]["gestartet"], time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(T0 + 1)))
        self.assertEqual(len(self.store.runs(a.id, limit=1)), 1)
        with self.assertRaises(ValueError):
            self.store.record_run(99, "ok", "x")

    def test_record_run_clips_message(self):
        a = self.store.create("A", "backup", 15, now=T0)
        self.store.record_run(a.id, "ok", "x" * 5000, now=T0)
        msg = self.store.get(a.id).last_message
        self.assertLess(len(msg), 1100)
        self.assertTrue(msg.endswith("[gekürzt]"))

    def test_stats(self):
        a = self.store.create("A", "backup", 15, now=T0)
        self.store.create("B", "eval", 15, enabled=False, now=T0)
        self.store.record_run(a.id, "fehler", "x", now=T0)
        s = self.store.stats()
        self.assertEqual(s["automationen"], 2)
        self.assertEqual(s["aktiv"], 1)
        self.assertEqual(s["laeufe"], 1)
        self.assertEqual(s["fehler"], 1)
        self.assertIn("faellig", s)

    def test_persistence_and_close_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d, "sub", "automationen.db")
            s1 = AutomationStore(path)
            a = s1.create("A", "mission", 30, {"ziel": "Drohne"}, now=T0)
            s1.record_run(a.id, "ok", "fertig", now=T0 + 2)
            s1.close()
            s1.close()
            s2 = AutomationStore(path)
            try:
                got = s2.get(a.id)
                self.assertEqual(got.params, {"ziel": "Drohne"})
                self.assertEqual(got.last_status, "ok")
                self.assertEqual(len(s2.runs(a.id)), 1)
            finally:
                s2.close()

    def test_corrupt_params_tolerated(self):
        a = self.store.create("A", "backup", 15)
        self.store._db.execute("UPDATE automations SET params = ? WHERE id = ?", ("kein json", a.id))
        self.store._db.commit()
        self.assertEqual(self.store.get(a.id).params, {})

    def test_thread_safety_of_store(self):
        a = self.store.create("A", "backup", 1, now=T0)
        errors: list[Exception] = []

        def worker(i):
            try:
                for j in range(20):
                    self.store.record_run(a.id, "ok", f"{i}-{j}", now=T0 + i * 100 + j)
                    self.store.get(a.id)
                    self.store.list()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.store.runs(a.id, limit=200)), 80)


# ------------------------------------------------------------------ Vorschläge
class DefaultsTest(unittest.TestCase):
    def test_default_automations(self):
        defaults = default_automations()
        self.assertEqual(len(defaults), 3)
        by_name = {d["name"]: d for d in defaults}
        self.assertEqual(by_name["Konsolidierung täglich"]["kind"], "gedaechtnis_konsolidieren")
        self.assertEqual(by_name["Konsolidierung täglich"]["interval_minutes"], 1440)
        self.assertEqual(by_name["Backup täglich"]["kind"], "backup")
        self.assertEqual(by_name["Backup täglich"]["interval_minutes"], 1440)
        self.assertEqual(by_name["Wissens-Sync"]["kind"], "wissen_sync")
        self.assertEqual(by_name["Wissens-Sync"]["interval_minutes"], 60)
        for d in defaults:
            self.assertIn(d["kind"], KINDS)
            self.assertIsInstance(d["params"], dict)
            self.assertTrue(d["beschreibung"])

    def test_install_defaults_idempotent(self):
        store = AutomationStore(":memory:")
        self.addCleanup(store.close)
        created = install_defaults(store, enabled=False)
        self.assertEqual(len(created), 3)
        self.assertFalse(all(a.enabled for a in created))
        self.assertEqual(install_defaults(store), [])
        self.assertEqual(len(store.list()), 3)


# ------------------------------------------------------------------ Zeitplaner: run_once
class SchedulerBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = AutomationStore(":memory:")
        self.addCleanup(self.store.close)
        self.log = Path(self.tmp.name, "logs", "automation.log")

    def make(self, brain=None, **kw) -> Scheduler:
        brain = brain if brain is not None else StubBrain(self.tmp.name, responder=eval_answer)
        kw.setdefault("log_path", self.log)
        kw.setdefault("tick_seconds", 0.05)
        sched = Scheduler(brain, self.store, **kw)
        self.addCleanup(sched.stop)
        return sched

    def log_text(self) -> str:
        return self.log.read_text(encoding="utf-8") if self.log.is_file() else ""


class RunOnceTest(SchedulerBase):
    def test_consolidate(self):
        sched = self.make()
        a = self.store.create("Konsolidierung", "gedaechtnis_konsolidieren", 1440, {"tage": 3, "projekt": "Drohne"})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok")
        self.assertIn("2 zusammengefasst", msg)
        self.assertIn("5 gelöscht", msg)
        self.assertIn("1 Sitzungen", msg)
        self.assertEqual(sched.brain.calls, [("consolidate", 3, "Drohne")])
        got = self.store.get(a.id)
        self.assertEqual(got.last_status, "ok")
        self.assertEqual(got.last_message, msg)
        self.assertIsNotNone(got.last_run)
        self.assertEqual(len(self.store.runs(a.id)), 1)
        self.assertIn(f"[ok] #{a.id} Konsolidierung (gedaechtnis_konsolidieren): ", self.log_text())

    def test_consolidate_default_days_and_bad_param(self):
        sched = self.make()
        a = self.store.create("K", "gedaechtnis_konsolidieren", 1440)
        self.assertEqual(sched.run_once(a)[0], "ok")
        self.assertEqual(sched.brain.calls[-1], ("consolidate", 7, None))
        b = self.store.create("K2", "gedaechtnis_konsolidieren", 1440, {"tage": "viele"})
        status, msg = sched.run_once(b)
        self.assertEqual(status, "fehler")
        self.assertIn("tage", msg)
        self.assertEqual(len(sched.brain.calls), 1)      # kein zweiter Aufruf

    def test_run_once_by_id_and_unknown(self):
        sched = self.make()
        a = self.store.create("K", "gedaechtnis_konsolidieren", 1440)
        self.assertEqual(sched.run_once(a.id)[0], "ok")
        with self.assertRaises(ValueError):
            sched.run_once(999)

    def test_backup(self):
        sched = self.make()
        a = self.store.create("Backup", "backup", 1440, {"behalten": 2})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok")
        self.assertIn("Backup erstellt:", msg)
        self.assertIn("20250101-1200", msg)
        self.assertIn("behalten: 2", msg)
        self.assertEqual(sched.brain.calls, [("backup", 2)])

    def test_eval_writes_report(self):
        sched = self.make()
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        a = self.store.create("Eval", "eval", 10080, {"datei": str(eval_file)})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok", msg)
        self.assertIn("2 Fragen", msg)
        self.assertIn("Stichwort-Score 1.00", msg)
        self.assertIn("Modell fake-modell", msg)
        reports = sorted(Path(sched.brain.cfg.logs_dir).glob("eval-*.json"))
        self.assertEqual(len(reports), 1)
        self.assertIn(str(reports[0]), msg)
        report = json.loads(reports[0].read_text(encoding="utf-8"))
        self.assertEqual(report["anzahl"], 2)
        self.assertEqual(report["modell"], "fake-modell")
        self.assertEqual(report["stichwort_score"], 1.0)
        self.assertIsNone(report["richter_score"])
        self.assertEqual(len(report["ergebnisse"]), 2)
        # Backend wurde direkt (ohne Gremium) mit num_ctx aus cfg gerufen
        self.assertEqual(len(sched.brain.backend.calls), 2)
        self.assertEqual(sched.brain.backend.calls[0]["num_ctx"], 4096)

    def test_eval_with_judge_and_limit(self):
        def responder(msgs, kw):
            if msgs[0]["role"] == "system" and "richter" in msgs[0]["content"]:
                return '{"punkte": 8, "begruendung": "gut"}'
            return eval_answer(msgs, kw)

        brain = StubBrain(self.tmp.name, responder=responder)
        sched = self.make(brain)
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        a = self.store.create("Eval", "eval", 60, {"datei": str(eval_file), "richter": "richter", "max_fragen": 1,
                                                  "modell": "qwen2.5:7b"})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok", msg)
        self.assertIn("1 Fragen", msg)
        self.assertIn("Richter 8.0/10", msg)
        self.assertIn("Modell qwen2.5:7b", msg)

    def test_eval_report_names_unique(self):
        sched = self.make(clock=lambda: T0)       # gleicher Zeitstempel bei jedem Lauf
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        a = self.store.create("Eval", "eval", 60, {"datei": str(eval_file)})
        self.assertEqual(sched.run_once(a)[0], "ok")
        self.assertEqual(sched.run_once(a)[0], "ok")
        reports = sorted(Path(sched.brain.cfg.logs_dir).glob("eval-*.json"))
        self.assertEqual(len(reports), 2)

    def test_eval_default_file_exists(self):
        self.assertTrue(DEFAULT_EVAL_FILE.is_file(), DEFAULT_EVAL_FILE)
        self.assertEqual(DEFAULT_EVAL_FILE.name, "eval_fragen.jsonl")

    def test_eval_missing_file(self):
        sched = self.make()
        a = self.store.create("Eval", "eval", 60, {"datei": str(Path(self.tmp.name, "gibts_nicht.jsonl"))})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "fehler")
        self.assertIn("nicht gefunden", msg)
        self.assertEqual(self.store.get(a.id).last_status, "fehler")

    def test_eval_empty_file(self):
        sched = self.make()
        empty = Path(self.tmp.name, "leer.jsonl")
        empty.write_text("", encoding="utf-8")
        status, msg = sched.run_once(self.store.create("Eval", "eval", 60, {"datei": str(empty)}))
        self.assertEqual(status, "fehler")
        self.assertIn("keine Fragen", msg)

    def test_eval_through_brain_requires_ask(self):
        sched = self.make()
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        a = self.store.create("Eval", "eval", 60, {"datei": str(eval_file), "tiefe": "mittel"})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "fehler")
        self.assertIn("Brain.ask", msg)

    def test_eval_through_brain(self):
        brain = StubBrain(self.tmp.name, responder=eval_answer)
        asked: list[tuple] = []

        def ask(frage, *, session_id="standard", depth=None, learn=None, **kw):
            asked.append((frage, session_id, depth, learn))
            return SimpleNamespace(text=eval_answer([{"role": "user", "content": frage}], {}), tokens=5,
                                   duration=0.01, depth=depth)

        brain.ask = ask
        sched = self.make(brain)
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        a = self.store.create("Eval", "eval", 60, {"datei": str(eval_file), "tiefe": "mittel"})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok", msg)
        self.assertIn("Tiefe mittel", msg)
        self.assertEqual(len(asked), 2)
        self.assertEqual(asked[0][2:], ("mittel", False))
        self.assertEqual(brain.backend.calls, [])

    def test_eval_backend_down(self):
        brain = StubBrain(self.tmp.name)
        brain.backend = FakeBackend(up=False)
        sched = self.make(brain)
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        status, msg = sched.run_once(self.store.create("Eval", "eval", 60, {"datei": str(eval_file)}))
        self.assertEqual(status, "fehler")
        self.assertIn("Modell-Server nicht erreichbar", msg)

    def test_knowledge_sync(self):
        sched = self.make()
        a = self.store.create("Sync", "wissen_sync", 60)
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok")
        self.assertIn("Wissens-Sync:", msg)
        self.assertIn("aktualisiert: 3", msg)
        self.assertIn("fehlend: 1", msg)          # Listen werden gezählt
        self.assertEqual(sched.brain.calls, [("sync",)])

    def test_mission_ok(self):
        sched = self.make()
        a = self.store.create("Mission", "mission", 1440, {"ziel": "Drohne planen", "projekt": "Drohne"})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok")
        self.assertIn("Mission #7", msg)
        self.assertIn("fertig", msg)
        self.assertIn("3/3 Schritte", msg)
        self.assertIn("Bericht: Rahmen aus CFK", msg)
        self.assertEqual(sched.brain.missions.calls, [("plan", "Drohne planen", "Drohne"), ("run", 7)])

    def test_mission_failed_and_missing_goal(self):
        brain = StubBrain(self.tmp.name)
        brain.missions = StubMissions(result_status="fehler")
        sched = self.make(brain)
        status, msg = sched.run_once(self.store.create("M", "mission", 1440, {"ziel": "x"}))
        self.assertEqual(status, "fehler")
        self.assertIn("2/3 Schritte", msg)
        self.assertIn("Schritt 3 gescheitert", msg)
        status, msg = sched.run_once(self.store.create("M2", "mission", 1440, {}))
        self.assertEqual(status, "fehler")
        self.assertIn("ziel", msg)
        self.assertEqual(len(brain.missions.calls), 2)   # kein Plan ohne Ziel

    def test_tool_ok(self):
        sched = self.make()
        a = self.store.create("Rechnen", "werkzeug", 5, {"werkzeug": "rechnen", "args": {"ausdruck": "2*21"}})
        status, msg = sched.run_once(a)
        self.assertEqual(status, "ok")
        self.assertTrue(msg.startswith("rechnen: "), msg)
        self.assertIn("42", msg)

    def test_tool_errors(self):
        sched = self.make()
        status, msg = sched.run_once(self.store.create("W", "werkzeug", 5, {"werkzeug": "gibts_nicht"}))
        self.assertEqual(status, "fehler")
        self.assertIn("Unbekanntes Werkzeug »gibts_nicht«", msg)
        status, msg = sched.run_once(self.store.create("W2", "werkzeug", 5, {}))
        self.assertEqual(status, "fehler")
        self.assertIn("werkzeug", msg)
        status, msg = sched.run_once(self.store.create("W3", "werkzeug", 5, {"werkzeug": "rechnen", "args": "x"}))
        self.assertEqual(status, "fehler")
        self.assertIn("args", msg)
        status, msg = sched.run_once(self.store.create("W4", "werkzeug", 5, {"werkzeug": "rechnen", "args": {}}))
        self.assertEqual(status, "fehler")
        self.assertIn("Fehlende Parameter", msg)
        status, msg = sched.run_once(self.store.create("W5", "werkzeug", 5,
                                                       {"werkzeug": "rechnen", "args": {"ausdruck": "1/0"}}))
        self.assertEqual(status, "fehler")
        self.assertTrue(msg.startswith("rechnen: "))

    def test_dangerous_tool_policy(self):
        target = Path(self.tmp.name, "notiz.txt")
        params = {"werkzeug": "datei_schreiben", "args": {"pfad": "notiz.txt", "inhalt": "Hallo"}}
        sched = self.make()                                  # allow_dangerous=False
        status, msg = sched.run_once(self.store.create("Schreiben", "werkzeug", 5, params))
        self.assertEqual(status, "fehler")
        self.assertIn("gefährlich", msg)
        self.assertIn("allow_dangerous", msg)
        self.assertFalse(target.exists())

        allowed = self.make(allow_dangerous=True)
        status, msg = allowed.run_once(self.store.create("Schreiben2", "werkzeug", 5, params))
        self.assertEqual(status, "ok", msg)
        self.assertIn("geschrieben", msg)
        self.assertEqual(target.read_text(encoding="utf-8"), "Hallo")
        # Die Haupt-Registry behält ihre Freigabe-Regel (Rückfrage, hier ohne confirm -> Ablehnung)
        res = allowed.brain.tools.run("datei_schreiben", {"pfad": "zwei.txt", "inhalt": "x"})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "Vom Nutzer abgelehnt")

    def test_dangerous_tool_sandbox_still_applies(self):
        sched = self.make(allow_dangerous=True)
        outside = Path(self.tmp.name).parent / "obito_test_ausbruch.txt"
        params = {"werkzeug": "datei_schreiben", "args": {"pfad": str(outside), "inhalt": "x"}}
        status, msg = sched.run_once(self.store.create("S", "werkzeug", 5, params))
        self.assertEqual(status, "fehler")
        self.assertIn("außerhalb", msg)
        self.assertFalse(outside.exists())

    def test_missing_capabilities(self):
        sched = self.make(BareBrain())
        cases = {
            "gedaechtnis_konsolidieren": ({}, "Brain.consolidate"),
            "backup": ({}, "Brain.backup"),
            "eval": ({}, "Brain.backend"),
            "wissen_sync": ({}, "Brain.knowledge.sync"),
            "mission": ({"ziel": "x"}, "Brain.missions"),
            "werkzeug": ({"werkzeug": "rechnen"}, "Brain.tools"),
        }
        for kind, (params, attr) in cases.items():
            with self.subTest(kind=kind):
                a = self.store.create(kind, kind, 60, params)
                status, msg = sched.run_once(a)
                self.assertEqual(status, "fehler")
                self.assertIn(attr, msg)
                self.assertIn("nicht verfügbar", msg)
                self.assertEqual(self.store.get(a.id).last_status, "fehler")

    def test_partial_capability_not_callable(self):
        brain = StubBrain(self.tmp.name)
        brain.knowledge = SimpleNamespace(sync="kein Aufruf")
        sched = self.make(brain)
        status, msg = sched.run_once(self.store.create("S", "wissen_sync", 60))
        self.assertEqual(status, "fehler")
        self.assertIn("Brain.knowledge.sync", msg)

    def test_handler_exception_becomes_error_result(self):
        brain = StubBrain(self.tmp.name)
        brain.fail_with = RuntimeError("Datenbank gesperrt")
        sched = self.make(brain)
        a = self.store.create("K", "gedaechtnis_konsolidieren", 60)
        status, msg = sched.run_once(a)
        self.assertEqual(status, "fehler")
        self.assertIn("RuntimeError", msg)
        self.assertIn("Datenbank gesperrt", msg)
        self.assertEqual(self.store.get(a.id).last_status, "fehler")
        self.assertIn("Rückverfolgung", self.log_text())
        self.assertIsNone(sched.current)

    def test_unknown_kind_in_db(self):
        sched = self.make()
        a = self.store.create("K", "backup", 60)
        self.store._db.execute("UPDATE automations SET kind = 'fremd' WHERE id = ?", (a.id,))
        self.store._db.commit()
        status, msg = sched.run_once(self.store.get(a.id))
        self.assertEqual(status, "fehler")
        self.assertIn("fremd", msg)

    def test_handler_status_normalized(self):
        sched = self.make()
        sched._handlers["backup"] = lambda auto: ("vielleicht", "x\n" * 3000)
        a = self.store.create("B", "backup", 60)
        status, msg = sched.run_once(a)
        self.assertEqual(status, "fehler")
        self.assertLess(len(msg), 1100)

    def test_run_due(self):
        sched = self.make(clock=lambda: T0 + 5)
        a = self.store.create("A", "backup", 60, now=T0)
        b = self.store.create("B", "wissen_sync", 60, now=T0 + 100)
        results = sched.run_due()
        self.assertEqual([(r[0], r[1]) for r in results], [(a.id, "ok")])
        self.assertEqual(sched.run_due(), [])
        self.assertEqual([(r[0], r[1]) for r in sched.run_due(T0 + 100)], [(b.id, "ok")])
        self.assertEqual(self.store.get(a.id).next_run, T0 + 5 + 3600)

    def test_log_without_path_and_tail(self):
        sched = Scheduler(StubBrain(self.tmp.name), self.store, log_path=None)
        a = self.store.create("B", "backup", 60)
        self.assertEqual(sched.run_once(a)[0], "ok")
        self.assertEqual(sched.log_tail(), [])
        logged = self.make()
        logged.run_once(a)
        logged.run_once(a)
        tail = logged.log_tail(1)
        self.assertEqual(len(tail), 1)
        self.assertIn("[ok]", tail[0])
        self.assertEqual(len(logged.log_tail(50)), 2)

    def test_logs_dir_fallback_without_cfg(self):
        brain = SimpleNamespace(backend=FakeBackend(responder=eval_answer))
        sched = self.make(brain)
        eval_file = write_eval_file(Path(self.tmp.name, "fragen.jsonl"))
        status, msg = sched.run_once(self.store.create("E", "eval", 60, {"datei": str(eval_file)}))
        self.assertEqual(status, "ok", msg)
        self.assertEqual(len(list(self.log.parent.glob("eval-*.json"))), 1)


# ------------------------------------------------------------------ Zeitplaner: Thread
class SchedulerThreadTest(SchedulerBase):
    def test_background_thread_runs_due_and_stops_cleanly(self):
        sched = self.make()
        a = self.store.create("Backup", "backup", 60)
        self.assertFalse(sched.running)
        sched.start()
        sched.start()                                   # zweiter Start harmlos
        self.assertTrue(sched.running)
        self.assertTrue(wait_until(lambda: self.store.get(a.id).last_run is not None))
        got = self.store.get(a.id)
        self.assertEqual(got.last_status, "ok")
        self.assertGreater(got.next_run, got.last_run)
        self.assertEqual(len(self.store.runs(a.id)), 1)
        self.assertEqual(sched.brain.calls, [("backup", 7)])
        sched.stop()
        self.assertFalse(sched.running)
        sched.stop()                                    # zweiter Stop harmlos
        text = self.log_text()
        self.assertIn("Zeitplaner gestartet.", text)
        self.assertIn(f"[ok] #{a.id} Backup (backup): Backup erstellt:", text)
        self.assertIn("Zeitplaner gestoppt.", text)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for ln in lines:
            self.assertRegex(ln, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} ")
        self.assertEqual(lines[0].split(" ", 2)[2], "Zeitplaner gestartet.")
        self.assertEqual(lines[-1].split(" ", 2)[2], "Zeitplaner gestoppt.")

    def test_thread_runs_sequentially_and_survives_errors(self):
        brain = StubBrain(self.tmp.name)
        active = {"n": 0, "max": 0}
        lock = threading.Lock()

        def slow_sync():
            with lock:
                active["n"] += 1
                active["max"] = max(active["max"], active["n"])
            time.sleep(0.03)
            with lock:
                active["n"] -= 1
            return {"aktualisiert": 0}

        brain.knowledge = SimpleNamespace(sync=slow_sync)
        brain.fail_with = RuntimeError("kaputt")
        sched = self.make(brain)
        ids = [self.store.create(f"S{i}", "wissen_sync", 1).id for i in range(3)]
        bad = self.store.create("K", "gedaechtnis_konsolidieren", 1)
        sched.start()
        self.assertTrue(wait_until(lambda: all(self.store.get(i).last_run is not None for i in ids + [bad.id])))
        self.assertTrue(sched.running)                   # Fehler eines Laufs tötet den Thread nicht
        self.assertEqual(self.store.get(bad.id).last_status, "fehler")
        self.assertEqual(active["max"], 1)
        # manueller Lauf parallel zum Thread ist strikt sequenziell
        self.assertEqual(sched.run_once(ids[0])[0], "ok")
        self.assertEqual(active["max"], 1)
        sched.stop()
        self.assertFalse(sched.running)

    def test_loop_survives_store_exception(self):
        sched = self.make()
        a = self.store.create("B", "backup", 60)
        original = self.store.due
        state = {"fail": 2}

        def flaky(now=None):
            if state["fail"] > 0:
                state["fail"] -= 1
                raise sqlite3.OperationalError("database is locked")
            return original(now)

        self.store.due = flaky
        sched.start()
        self.assertTrue(wait_until(lambda: self.store.get(a.id).last_run is not None))
        self.assertTrue(sched.running)
        self.assertIn("database is locked", sched.last_error or "")
        self.assertIn("Fehler in der Zeitplaner-Schleife", self.log_text())
        sched.stop()

    def test_stop_from_inside_run_does_not_deadlock(self):
        brain = StubBrain(self.tmp.name)
        sched = self.make(brain)

        def stopping_sync():
            sched.stop(timeout=0.1)                      # aus dem eigenen Thread heraus
            return {"aktualisiert": 1}

        brain.knowledge = SimpleNamespace(sync=stopping_sync)
        a = self.store.create("S", "wissen_sync", 1)
        sched.start()
        self.assertTrue(wait_until(lambda: self.store.get(a.id).last_run is not None))
        self.assertTrue(wait_until(lambda: not sched.running))
        self.assertEqual(self.store.get(a.id).last_status, "ok")

    def test_disabled_automation_not_run(self):
        sched = self.make()
        a = self.store.create("B", "backup", 60, enabled=False)
        sched.start()
        time.sleep(0.15)
        sched.stop()
        self.assertIsNone(self.store.get(a.id).last_run)
        self.assertEqual(sched.brain.calls, [])


if __name__ == "__main__":
    unittest.main()
