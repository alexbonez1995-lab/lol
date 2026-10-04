"""Tests für obito/server.py – gegen einen FakeBrain, ohne Modell, ohne Netzwerk nach außen.

Der FakeBrain bildet exakt die in docs/ARCHITEKTUR.md beschriebene Brain-Oberfläche nach.
Wo möglich werden die echten Dataclasses aus obito.brain/obito.learning verwendet; fehlen sie
(oder sind noch nicht kompatibel), greifen formatgleiche lokale Ersatzklassen.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from obito.config import Config
from obito.llm import ModelInfo
from obito.memory import Memory, MemoryStore
from obito import server as server_mod
from obito.server import ObitoServer, memory_to_dict

try:  # echte Klassen bevorzugen, Ersatz nur wenn brain.py (noch) nicht importierbar ist
    from obito.brain import Answer as _RealAnswer, Feedback as _RealFeedback, Step as _RealStep
except Exception:  # noqa: BLE001
    _RealAnswer = _RealFeedback = _RealStep = None
try:
    from obito.learning import Interaction as _RealInteraction, Lesson as _RealLesson
except Exception:  # noqa: BLE001
    _RealInteraction = _RealLesson = None


# ------------------------------------------------------------ Ersatzklassen
@dataclass
class _Step:
    stage: str
    who: str
    summary: str
    detail: str = ""
    duration: float = 0.0
    tokens: int = 0
    status: str = "fertig"

    def to_dict(self) -> dict:
        return {"stufe": self.stage, "wer": self.who, "zusammenfassung": self.summary, "detail": self.detail,
                "dauer": self.duration, "tokens": self.tokens, "status": self.status}


@dataclass
class _Answer:
    text: str
    question: str
    session_id: str
    project: str | None
    depth: str
    experts: list
    critique: dict | None
    memories_used: list
    new_memories: list
    lessons_used: list
    tools_used: list
    steps: list
    interaction_id: int | None
    tokens: int
    duration: float

    def to_dict(self) -> dict:
        return {
            "antwort": self.text, "frage": self.question, "sitzung": self.session_id, "projekt": self.project,
            "tiefe": self.depth, "experten": list(self.experts), "kritik": self.critique,
            "erinnerungen_genutzt": [memory_to_dict(m) for m in self.memories_used],
            "erinnerungen_neu": [memory_to_dict(m) for m in self.new_memories],
            "werkzeuge": list(self.tools_used), "spur": [s.to_dict() for s in self.steps],
            "interaktion_id": self.interaction_id, "tokens": self.tokens, "dauer": self.duration,
        }

    def trace_json(self, max_detail: int = 1000) -> str:
        out = []
        for s in self.steps:
            d = s.to_dict()
            d["detail"] = d["detail"][:max_detail]
            out.append(d)
        return json.dumps(out, ensure_ascii=False)


@dataclass
class _Interaction:
    id: int
    session_id: str
    question: str
    answer: str
    project: str | None = None
    experts: list = field(default_factory=list)
    depth: str = "schnell"
    model: str = "fake-modell"
    context: str = ""
    history: list = field(default_factory=list)
    memories_used: list = field(default_factory=list)
    new_memories: list = field(default_factory=list)
    lessons_used: list = field(default_factory=list)
    tools_used: list = field(default_factory=list)
    tokens: int = 0
    duration: float = 0.0
    rating: int = 0
    comment: str | None = None
    correction: str | None = None
    correction_full: str | None = None
    trainable: bool = True
    created_at: float = field(default_factory=time.time)
    trace: str = "[]"

    def to_dict(self) -> dict:
        return {"id": self.id, "sitzung": self.session_id, "frage": self.question, "antwort": self.answer,
                "projekt": self.project, "experten": self.experts, "tiefe": self.depth, "modell": self.model,
                "bewertung": self.rating, "kommentar": self.comment, "korrektur": self.correction,
                "korrektur_voll": self.correction_full, "trainierbar": self.trainable,
                "erstellt": self.created_at, "werkzeuge": self.tools_used, "tokens": self.tokens,
                "dauer": self.duration}


@dataclass
class _Lesson:
    id: int
    rule: str
    scope: str = "thema"
    topics: list = field(default_factory=list)
    source_interaction: int | None = None
    created_at: float = field(default_factory=time.time)
    helped: int = 0
    hurt: int = 0
    active: bool = True

    def to_dict(self) -> dict:
        return {"id": self.id, "regel": self.rule, "bereich": self.scope, "themen": self.topics,
                "quelle": self.source_interaction, "erstellt": self.created_at, "geholfen": self.helped,
                "geschadet": self.hurt, "aktiv": self.active}

    def render(self) -> str:
        return f"Regel (bestätigt {self.helped}×): {self.rule}"[:300]


@dataclass
class _Feedback:
    interaction: object
    lessons: list
    memories_adjusted: int
    correction_full: str | None


def _make(real, fallback, **kw):
    """Baut bevorzugt die echte Dataclass; bei Signatur-Abweichung die lokale Ersatzklasse."""
    if real is not None:
        try:
            return real(**kw)
        except TypeError:
            pass
    return fallback(**kw)


def make_step(**kw):
    return _make(_RealStep, _Step, **kw)


def make_answer(**kw):
    return _make(_RealAnswer, _Answer, **kw)


def make_interaction(**kw):
    return _make(_RealInteraction, _Interaction, **kw)


def make_lesson(**kw):
    return _make(_RealLesson, _Lesson, **kw)


def make_feedback(**kw):
    return _make(_RealFeedback, _Feedback, **kw)


# --------------------------------------------------------------- FakeBrain
class FakeTools:
    def __init__(self):
        self.policy_calls: list[tuple] = []

    def set_policy(self, confirm, confirm_dangerous):
        self.policy_calls.append((confirm, confirm_dangerous))

    def list(self):
        return []


class FakeLearning:
    def __init__(self):
        self.lessons: dict[int, object] = {}
        self._next = 1

    def add(self, rule: str, active: bool = True):
        lesson = make_lesson(id=self._next, rule=rule, scope="thema", topics=["test"],
                             source_interaction=None, created_at=time.time(), helped=1, hurt=0, active=active)
        self.lessons[self._next] = lesson
        self._next += 1
        return lesson

    def list_lessons(self, include_inactive: bool = False):
        return [l for l in self.lessons.values() if include_inactive or l.active]

    def delete_lesson(self, lesson_id: int) -> bool:
        return self.lessons.pop(lesson_id, None) is not None

    def stats(self) -> dict:
        return {"interaktionen": 1, "bewertet": 0, "positiv": 0, "negativ": 0, "korrigiert": 0,
                "trainierbar": 1, "lektionen": len(self.lessons), "lektionen_aktiv": len(self.list_lessons())}


class FakeBrain:
    """Stand-in für obito.brain.Brain mit genau der vom Server genutzten Oberfläche."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.memory = MemoryStore(":memory:")
        self.learning = FakeLearning()
        self.tools = FakeTools()
        self._busy = False
        self.answer_text = "Die Antwort lautet 42 – ganz sicher."
        self.fail_text: str | None = None          # gesetzt -> Fehler-Answer (tiefe == "fehler")
        self.raise_in_ask: Exception | None = None
        self.raise_in_status: Exception | None = None
        self.ask_calls: list[dict] = []
        self.feedback_calls: list[dict] = []
        self.rewrite_calls: list[tuple] = []
        self.interactions: dict[int, object] = {}
        self.model_list = [ModelInfo("fake-modell", 1_000, "fake", "1B", "Q4"), ModelInfo("fake-embed")]
        self.models_fail = False
        self.closed = False

    @property
    def busy(self) -> bool:
        return self._busy

    def ask(self, question, *, session_id="standard", project=None, depth=None, stream=None,
            progress=None, learn=None, cancel=None):
        self.ask_calls.append({"question": question, "session_id": session_id, "project": project,
                               "depth": depth, "stream": stream is not None, "progress": progress is not None,
                               "cancel": cancel})
        if self.raise_in_ask is not None:
            raise self.raise_in_ask
        steps = []
        if self.fail_text is not None:
            steps.append(make_step(stage="erinnern", who="system", summary="0 Erinnerungen"))
            if progress:
                progress(steps[-1])
            return make_answer(text=self.fail_text, question=question, session_id=session_id, project=project,
                               depth="fehler", experts=[], critique=None, memories_used=[], new_memories=[],
                               lessons_used=[], tools_used=[], steps=steps, interaction_id=None, tokens=0,
                               duration=0.01)
        if progress:
            progress(make_step(stage="schnell", who="OMEGA", summary="denkt", status="start"))
        if stream:
            text = self.answer_text
            for i in range(0, len(text), 12):
                if cancel is not None and cancel.is_set():
                    break
                stream(text[i:i + 12])
        steps.append(make_step(stage="schnell", who="OMEGA", summary=f"{len(self.answer_text)} Zeichen",
                               detail=self.answer_text, duration=0.02, tokens=7, status="fertig"))
        if progress:
            progress(steps[-1])
        iid = len(self.interactions) + 1
        self.interactions[iid] = make_interaction(id=iid, session_id=session_id, question=question,
                                                  answer=self.answer_text, project=project, experts=["OMEGA"],
                                                  depth=depth or "schnell", model=self.cfg.model, context="",
                                                  history=[], memories_used=[], new_memories=[], lessons_used=[],
                                                  tools_used=[], tokens=7, duration=0.02, rating=0, comment=None,
                                                  correction=None, correction_full=None, trainable=True,
                                                  created_at=time.time(), trace="[]")
        return make_answer(text=self.answer_text, question=question, session_id=session_id, project=project,
                           depth=depth or "schnell", experts=["OMEGA"], critique=None, memories_used=[],
                           new_memories=[], lessons_used=[], tools_used=[], steps=steps, interaction_id=iid,
                           tokens=7, duration=0.02)

    def feedback(self, interaction_id, rating=None, comment=None, correction=None, correction_full=None):
        if interaction_id not in self.interactions:
            raise ValueError(f"Interaktion {interaction_id} unbekannt")
        self.feedback_calls.append({"interaction_id": interaction_id, "rating": rating, "comment": comment,
                                    "correction": correction, "correction_full": correction_full})
        inter = self.interactions[interaction_id]
        if rating is not None:
            inter.rating = rating
        inter.comment = comment
        if correction:
            inter.correction = correction
            inter.correction_full = correction_full
            if rating is None:
                inter.rating = -1
        lessons = []
        if (rating == -1 and comment) or correction:
            lessons.append(self.learning.add(comment or f"Besser: {correction}"))
        return make_feedback(interaction=inter, lessons=lessons, memories_adjusted=0, correction_full=correction_full)

    def rewrite_correction(self, interaction_id, correction) -> str:
        self.rewrite_calls.append((interaction_id, correction))
        return f"Vollständige Antwort mit Korrektur: {correction}"

    def remember(self, content, kind="notiz", tags=(), project=None, importance=0.6) -> Memory:
        return self.memory.remember(content, kind=kind, tags=tags, project=project, importance=importance)

    def forget(self, memory_id) -> bool:
        return self.memory.forget(memory_id)

    def recall(self, query, k=None, project=None) -> list[Memory]:
        return self.memory.search(query, k or self.cfg.memory_recall, project, min_importance=0.15)

    def models(self) -> list[ModelInfo]:
        return [] if self.models_fail else list(self.model_list)

    def status(self) -> dict:
        if self.raise_in_status is not None:
            raise self.raise_in_status
        return {"backend": "fake", "verfuegbar": True, "modell": self.cfg.model, "routing_modell": self.cfg.model,
                "embedding_aktiv": False, "modelle": [m.name for m in self.model_list],
                "gedaechtnis": self.memory.stats(), "lernen": self.learning.stats(), "werkzeuge": ["rechnen"],
                "beschaeftigt": self.busy, "tiefe": self.cfg.depth, "datenverzeichnis": self.cfg.data_dir,
                "vektoren": {"mit": 0, "ohne": 0, "modell": "", "dim": 0}}

    def close(self) -> None:
        if not self.closed:
            self.memory.close()
            self.closed = True


# ---------------------------------------------------------------- Helfer
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # nie über System-Proxy


class _Resp:
    def __init__(self, status: int, headers, body: bytes):
        self.status = status
        self.headers = headers
        self.body = body

    def json(self):
        return json.loads(self.body.decode("utf-8"))


def parse_sse(raw: bytes) -> list[tuple[str, dict]]:
    events = []
    for block in raw.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        ev, data = "message", ""
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev = line[6:].strip()
            elif line.startswith("data:"):
                data += line[5:].strip()
        events.append((ev, json.loads(data)))
    return events


class ServerTestBase(unittest.TestCase):
    allow_dangerous = False

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = Config(data_dir=cls.tmp.name, backend="fake", model="fake-modell")
        cls.brain = FakeBrain(cls.cfg)
        cls.server = ObitoServer(cls.brain, "127.0.0.1", 0, allow_dangerous=cls.allow_dangerous, max_body=4096)
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        cls.brain.close()
        cls.tmp.cleanup()

    def setUp(self):
        self.brain.fail_text = None
        self.brain.raise_in_ask = None
        self.brain.raise_in_status = None
        self.brain._busy = False
        self.brain.models_fail = False
        self.brain.ask_calls.clear()
        self.brain.feedback_calls.clear()
        self.brain.rewrite_calls.clear()
        self.server.index_path = server_mod.INDEX_PATH

    # -- HTTP
    def request(self, method: str, path: str, body=None, headers: dict | None = None, raw: bytes | None = None,
                json_ct: bool = True) -> _Resp:
        hdrs = dict(headers or {})
        data = raw
        if body is not None:
            data = json.dumps(body).encode("utf-8")
        if method in ("POST", "DELETE") and json_ct and "Content-Type" not in hdrs:
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, method=method, headers=hdrs)
        try:
            with _OPENER.open(req, timeout=10) as resp:
                return _Resp(resp.status, resp.headers, resp.read())
        except urllib.error.HTTPError as e:
            return _Resp(e.code, e.headers, e.read())

    def get(self, path, **kw):
        return self.request("GET", path, **kw)

    def post(self, path, body=None, **kw):
        return self.request("POST", path, body=body, **kw)

    def delete(self, path, **kw):
        return self.request("DELETE", path, **kw)

    def log_text(self) -> str:
        p = Path(self.cfg.logs_dir) / "server.log"
        return p.read_text(encoding="utf-8") if p.exists() else ""


# ------------------------------------------------------------------ Tests
class StartupTest(ServerTestBase):
    def test_port_and_policy(self):
        self.assertGreater(self.server.port, 0)
        self.assertEqual(len(self.brain.tools.policy_calls), 1)
        confirm, confirm_dangerous = self.brain.tools.policy_calls[0]
        self.assertTrue(confirm_dangerous)
        self.assertFalse(confirm("datei_schreiben", {"pfad": "x"}))

    def test_log_dir_created_and_requests_logged(self):
        self.assertTrue(Path(self.cfg.logs_dir).is_dir())
        self.get("/api/status")
        text = self.log_text()
        self.assertIn("Server gestartet", text)
        self.assertIn("GET /api/status", text)
        self.assertIn("127.0.0.1", text)

    def test_allow_dangerous_policy_and_warning(self):
        with tempfile.TemporaryDirectory() as d:
            brain = FakeBrain(Config(data_dir=d, backend="fake"))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                srv = ObitoServer(brain, "0.0.0.0", 0, allow_dangerous=True)
            try:
                self.assertIn("WARNUNG", err.getvalue())
                self.assertIn("0.0.0.0", err.getvalue())
                confirm, _ = brain.tools.policy_calls[0]
                self.assertTrue(confirm("befehl_ausfuehren", {}))
                self.assertIn("0.0.0.0", srv.allowed_hosts)
            finally:
                srv.server_close()
                brain.close()

    def test_loopback_without_warning(self):
        with tempfile.TemporaryDirectory() as d:
            brain = FakeBrain(Config(data_dir=d, backend="fake"))
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                srv = ObitoServer(brain, "localhost", 0)
            try:
                self.assertEqual(err.getvalue(), "")
            finally:
                srv.server_close()
                brain.close()

    def test_host_matching_rules(self):
        srv = self.server
        p = srv.port
        for ok in ("127.0.0.1", f"127.0.0.1:{p}", "localhost", f"LOCALHOST:{p}", "[::1]", f"[::1]:{p}"):
            self.assertTrue(srv.host_allowed(ok), ok)
        for bad in (None, "", "evil.example", f"evil.example:{p}", f"127.0.0.1:{p + 1}", "127.0.0.1:abc",
                    "::1", "[::1", "127.0.0.1:1:2"):
            self.assertFalse(srv.host_allowed(bad), repr(bad))
        self.assertTrue(srv.origin_allowed(f"http://localhost:{p}"))
        self.assertTrue(srv.origin_allowed(f"http://127.0.0.1:{p}/pfad"))
        for bad in ("null", "http://evil.example", f"ftp://127.0.0.1:{p}", f"http://user@127.0.0.1:{p}",
                    "127.0.0.1"):
            self.assertFalse(srv.origin_allowed(bad), bad)


class HardeningTest(ServerTestBase):
    def test_wrong_host_403(self):
        r = self.get("/api/status", headers={"Host": "evil.example"})
        self.assertEqual(r.status, 403)
        self.assertFalse(r.json()["ok"])
        self.assertIn("Host", r.json()["fehler"])

    def test_host_with_wrong_port_403_and_localhost_ok(self):
        r = self.get("/api/status", headers={"Host": f"127.0.0.1:{self.server.port + 1}"})
        self.assertEqual(r.status, 403)
        r = self.get("/api/status", headers={"Host": "localhost"})
        self.assertEqual(r.status, 200)

    def test_wrong_origin_403_and_good_origin_ok(self):
        r = self.get("/api/status", headers={"Origin": "http://evil.example"})
        self.assertEqual(r.status, 403)
        self.assertIn("Origin", r.json()["fehler"])
        r = self.post("/api/erinnerungen", {"inhalt": "Origin-Test Carbon"}, headers={"Origin": "null"})
        self.assertEqual(r.status, 403)
        r = self.get("/api/status", headers={"Origin": self.base})
        self.assertEqual(r.status, 200)

    def test_wrong_content_type_415(self):
        r = self.post("/api/frage", raw=b'{"frage": "x"}', headers={"Content-Type": "text/plain"})
        self.assertEqual(r.status, 415)
        self.assertFalse(r.json()["ok"])
        r = self.request("POST", "/api/frage", raw=b"{}", json_ct=False)
        self.assertEqual(r.status, 415)
        r = self.request("DELETE", "/api/erinnerungen/1", json_ct=False)
        self.assertEqual(r.status, 415)
        r = self.post("/api/frage", raw=b'{"frage": "x"}', headers={"Content-Type": "application/json; charset=utf-8"})
        self.assertEqual(r.status, 200)

    def test_too_large_413(self):
        big = json.dumps({"frage": "x" * 5000}).encode()
        r = self.post("/api/frage", raw=big)
        self.assertEqual(r.status, 413)
        self.assertFalse(r.json()["ok"])
        self.assertEqual(self.brain.ask_calls, [])

    def test_no_cors_headers(self):
        r = self.get("/api/status", headers={"Origin": self.base})
        self.assertEqual(r.status, 200)
        for h in r.headers:
            self.assertFalse(h.lower().startswith("access-control-"), h)
        r = self.get("/")
        for h in r.headers:
            self.assertFalse(h.lower().startswith("access-control-"), h)

    def test_404_unknown_path_and_405_wrong_method(self):
        r = self.get("/api/gibtsnicht")
        self.assertEqual(r.status, 404)
        self.assertFalse(r.json()["ok"])
        r = self.get("/static/index.html")
        self.assertEqual(r.status, 404)
        r = self.get("/../obito/server.py")
        self.assertNotEqual(r.status, 200)
        r = self.post("/api/status", {})
        self.assertEqual(r.status, 405)
        self.assertIn("GET", r.headers.get("Allow", ""))
        r = self.delete("/api/erinnerungen")
        self.assertEqual(r.status, 405)

    def test_internal_error_500_logged(self):
        self.brain.raise_in_status = RuntimeError("Kaputt im Kern")
        r = self.get("/api/status")
        self.assertEqual(r.status, 500)
        j = r.json()
        self.assertFalse(j["ok"])
        self.assertIn("Kaputt im Kern", j["fehler"])
        text = self.log_text()
        self.assertIn("RuntimeError: Kaputt im Kern", text)
        self.assertIn("Traceback", text)
        self.brain.raise_in_status = None
        self.assertEqual(self.get("/api/status").status, 200)


class IndexTest(ServerTestBase):
    def test_serves_index_html_if_present(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "index.html"
            p.write_text("<!doctype html><html lang='de'><body>HUD-Test ÄÖÜ</body></html>", encoding="utf-8")
            self.server.index_path = p
            r = self.get("/")
            self.assertEqual(r.status, 200)
            self.assertTrue(r.headers["Content-Type"].startswith("text/html"))
            self.assertIn("HUD-Test ÄÖÜ", r.body.decode("utf-8"))
            self.assertIn("Content-Security-Policy", r.headers)

    def test_placeholder_when_missing(self):
        self.server.index_path = Path(self.tmp.name) / "fehlt" / "index.html"
        r = self.get("/?x=1")
        self.assertEqual(r.status, 200)
        self.assertTrue(r.headers["Content-Type"].startswith("text/html"))
        self.assertIn("OBITO", r.body.decode("utf-8"))
        self.assertIn("nicht gefunden", r.body.decode("utf-8"))

    def test_real_index_path_points_into_static(self):
        self.assertEqual(server_mod.INDEX_PATH.parent.name, "static")
        self.assertEqual(server_mod.INDEX_PATH.name, "index.html")


class StatusAndModelsTest(ServerTestBase):
    def test_status(self):
        r = self.get("/api/status")
        self.assertEqual(r.status, 200)
        j = r.json()
        self.assertTrue(j["ok"])
        self.assertEqual(j["backend"], "fake")
        self.assertEqual(j["modell"], "fake-modell")
        self.assertIn("gedaechtnis", j)
        self.assertFalse(j["beschaeftigt"])
        self.assertTrue(r.headers["Content-Type"].startswith("application/json"))

    def test_models(self):
        r = self.get("/api/modelle")
        j = r.json()
        self.assertEqual(r.status, 200)
        self.assertTrue(j["ok"])
        self.assertEqual(j["aktuell"], "fake-modell")
        self.assertEqual(j["modelle"][0], {"name": "fake-modell", "groesse": 1000, "familie": "fake",
                                           "parameter": "1B", "quantisierung": "Q4"})
        self.brain.models_fail = True
        j = self.get("/api/modelle").json()
        self.assertEqual(j["modelle"], [])
        self.assertEqual(j["aktuell"], "fake-modell")


class FrageTest(ServerTestBase):
    def test_question_ok(self):
        r = self.post("/api/frage", {"frage": "Wie viel ist 6 mal 7?", "sitzung": "s1", "projekt": "Drohne",
                                     "tiefe": "schnell"})
        self.assertEqual(r.status, 200)
        j = r.json()
        self.assertTrue(j["ok"])
        self.assertEqual(j["antwort"], self.brain.answer_text)
        self.assertEqual(j["frage"], "Wie viel ist 6 mal 7?")
        self.assertEqual(j["sitzung"], "s1")
        self.assertEqual(j["projekt"], "Drohne")
        self.assertEqual(j["tiefe"], "schnell")
        self.assertIsInstance(j["interaktion_id"], int)
        for key in ("experten", "kritik", "erinnerungen_genutzt", "erinnerungen_neu", "werkzeuge", "spur",
                    "tokens", "dauer"):
            self.assertIn(key, j)
        self.assertEqual(j["spur"][0]["stufe"], "schnell")
        call = self.brain.ask_calls[-1]
        self.assertEqual((call["session_id"], call["project"], call["depth"]), ("s1", "Drohne", "schnell"))
        self.assertFalse(call["stream"])

    def test_defaults(self):
        r = self.post("/api/frage", {"frage": "Hallo"})
        self.assertEqual(r.status, 200)
        call = self.brain.ask_calls[-1]
        self.assertEqual(call["session_id"], "standard")
        self.assertIsNone(call["project"])
        self.assertIsNone(call["depth"])

    def test_bad_requests_400(self):
        self.assertEqual(self.post("/api/frage", {}).status, 400)
        self.assertEqual(self.post("/api/frage", {"frage": "   "}).status, 400)
        self.assertEqual(self.post("/api/frage", {"frage": 42}).status, 400)
        self.assertEqual(self.post("/api/frage", {"frage": "x", "tiefe": "ultratief"}).status, 400)
        self.assertEqual(self.post("/api/frage", {"frage": "x", "stream": "vielleicht"}).status, 400)
        r = self.post("/api/frage", raw=b"{kaputt")
        self.assertEqual(r.status, 400)
        self.assertIn("JSON", r.json()["fehler"])
        self.assertEqual(self.post("/api/frage", raw=b"[1, 2]").status, 400)
        self.assertEqual(self.post("/api/frage", raw=b"").status, 400)
        self.assertEqual(self.post("/api/frage", raw=b"\xff\xfe").status, 400)
        self.assertEqual(self.brain.ask_calls, [])

    def test_error_answer_503(self):
        self.brain.fail_text = "Modell-Server nicht erreichbar – starte Ollama (`ollama serve`)."
        r = self.post("/api/frage", {"frage": "Hallo"})
        self.assertEqual(r.status, 503)
        j = r.json()
        self.assertFalse(j["ok"])
        self.assertEqual(j["fehler"], self.brain.fail_text)
        self.assertEqual(j["tiefe"], "fehler")
        self.assertIsNone(j["interaktion_id"])

    def test_busy_409(self):
        self.brain._busy = True
        r = self.post("/api/frage", {"frage": "Hallo"})
        self.assertEqual(r.status, 409)
        self.assertFalse(r.json()["ok"])
        r = self.post("/api/frage", {"frage": "Hallo", "stream": True})
        self.assertEqual(r.status, 409)
        self.assertTrue(r.headers["Content-Type"].startswith("application/json"))
        self.brain._busy = False
        self.assertEqual(self.post("/api/frage", {"frage": "Hallo"}).status, 200)

    def test_exception_in_ask_500(self):
        self.brain.raise_in_ask = RuntimeError("Absturz")
        r = self.post("/api/frage", {"frage": "Hallo"})
        self.assertEqual(r.status, 500)
        self.assertIn("Absturz", r.json()["fehler"])
        self.assertIn("Absturz", self.log_text())


class StreamTest(ServerTestBase):
    def test_sse_events(self):
        r = self.post("/api/frage", {"frage": "Erzähl was", "stream": True, "sitzung": "sse"})
        self.assertEqual(r.status, 200)
        self.assertTrue(r.headers["Content-Type"].startswith("text/event-stream"))
        self.assertIn("chunked", (r.headers.get("Transfer-Encoding") or "").lower())
        for h in r.headers:
            self.assertFalse(h.lower().startswith("access-control-"), h)
        events = parse_sse(r.body)
        kinds = [e for e, _ in events]
        self.assertEqual(kinds[0], "schritt")
        self.assertEqual(kinds[-1], "antwort")
        steps = [d for e, d in events if e == "schritt"]
        self.assertEqual(steps[0]["status"], "start")
        self.assertEqual(steps[-1]["status"], "fertig")
        self.assertEqual(set(steps[0]), {"stufe", "wer", "zusammenfassung", "detail", "dauer", "tokens", "status"})
        tokens = [d["text"] for e, d in events if e == "token"]
        self.assertGreater(len(tokens), 1)
        self.assertEqual("".join(tokens), self.brain.answer_text)
        # Reihenfolge: start-Schritt, dann Tokens, dann fertig-Schritt, dann Antwort
        self.assertLess(kinds.index("token"), len(kinds) - 2)
        final = events[-1][1]
        self.assertEqual(final["antwort"], self.brain.answer_text)
        self.assertEqual(final["sitzung"], "sse")
        self.assertIsInstance(final["interaktion_id"], int)
        call = self.brain.ask_calls[-1]
        self.assertTrue(call["stream"] and call["progress"])
        self.assertIsInstance(call["cancel"], threading.Event)

    def test_sse_error_event(self):
        self.brain.fail_text = "Modellfehler: kaputt"
        r = self.post("/api/frage", {"frage": "x", "stream": True})
        self.assertEqual(r.status, 200)
        events = parse_sse(r.body)
        self.assertEqual(events[-1][0], "fehler")
        self.assertEqual(events[-1][1], {"fehler": "Modellfehler: kaputt"})
        self.assertEqual([e for e, _ in events if e == "token"], [])

    def test_sse_exception_event(self):
        self.brain.raise_in_ask = RuntimeError("Stream-Absturz")
        r = self.post("/api/frage", {"frage": "x", "stream": True})
        self.assertEqual(r.status, 200)
        events = parse_sse(r.body)
        self.assertEqual(events[-1][0], "fehler")
        self.assertIn("Stream-Absturz", events[-1][1]["fehler"])
        self.assertIn("Stream-Absturz", self.log_text())

    def test_sse_live_flush(self):
        """Der erste Event trifft ein, bevor die Antwort fertig ist (kein Puffern bis zum Ende)."""
        gate = threading.Event()
        original = self.brain.answer_text

        class SlowBrain:
            pass

        def slow_ask(question, *, session_id="standard", project=None, depth=None, stream=None,
                     progress=None, learn=None, cancel=None):
            progress(make_step(stage="schnell", who="OMEGA", summary="denkt", status="start"))
            gate.wait(5)
            return FakeBrain.ask(self.brain, question, session_id=session_id, project=project, depth=depth,
                                 stream=stream, progress=progress, learn=learn, cancel=cancel)

        self.brain.ask = slow_ask
        try:
            req = urllib.request.Request(self.base + "/api/frage", data=json.dumps({"frage": "x", "stream": True}).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
            resp = _OPENER.open(req, timeout=10)
            first = resp.readline() + resp.readline()
            self.assertIn(b"event: schritt", first)
            self.assertFalse(gate.is_set())
            gate.set()
            rest = resp.read()
            events = parse_sse(first + rest)
            self.assertEqual(events[-1][0], "antwort")
            self.assertEqual(events[-1][1]["antwort"], original)
        finally:
            del self.brain.ask
            gate.set()


class FeedbackTest(ServerTestBase):
    def _ask(self) -> int:
        return self.post("/api/frage", {"frage": "Wie schwer ist ein 5-Zoll-Quad?"}).json()["interaktion_id"]

    def test_rating_only(self):
        iid = self._ask()
        r = self.post("/api/feedback", {"interaktion_id": iid, "bewertung": 1})
        self.assertEqual(r.status, 200)
        j = r.json()
        self.assertTrue(j["ok"])
        self.assertEqual(j["interaktion"]["id"], iid)
        self.assertEqual(j["interaktion"]["bewertung"], 1)
        self.assertEqual(j["lektionen"], [])
        self.assertIsNone(j["korrektur_vorschlag"])
        self.assertEqual(self.brain.rewrite_calls, [])
        call = self.brain.feedback_calls[-1]
        self.assertEqual((call["rating"], call["comment"], call["correction"], call["correction_full"]),
                         (1, None, None, None))

    def test_negative_with_comment_creates_lesson(self):
        iid = self._ask()
        j = self.post("/api/feedback", {"interaktion_id": iid, "bewertung": -1, "kommentar": "Einheiten fehlen"}).json()
        self.assertEqual(len(j["lektionen"]), 1)
        self.assertEqual(j["lektionen"][0]["regel"], "Einheiten fehlen")
        self.assertIn("aktiv", j["lektionen"][0])

    def test_correction_with_rewrite(self):
        iid = self._ask()
        j = self.post("/api/feedback", {"interaktion_id": iid, "korrektur": "Es sind 650 g", "umschreiben": True}).json()
        self.assertTrue(j["ok"])
        self.assertEqual(j["korrektur_vorschlag"], "Vollständige Antwort mit Korrektur: Es sind 650 g")
        self.assertEqual(self.brain.rewrite_calls[-1], (iid, "Es sind 650 g"))
        call = self.brain.feedback_calls[-1]
        self.assertEqual(call["correction"], "Es sind 650 g")
        self.assertEqual(call["correction_full"], j["korrektur_vorschlag"])
        self.assertEqual(j["interaktion"]["korrektur"], "Es sind 650 g")
        self.assertEqual(len(j["lektionen"]), 1)

    def test_correction_full_given_wins_over_rewrite(self):
        iid = self._ask()
        j = self.post("/api/feedback", {"interaktion_id": iid, "korrektur": "650 g", "korrektur_voll": "Lange Fassung",
                                        "umschreiben": True}).json()
        self.assertEqual(j["korrektur_vorschlag"], "Vollständige Antwort mit Korrektur: 650 g")
        self.assertEqual(self.brain.feedback_calls[-1]["correction_full"], "Lange Fassung")

    def test_correction_without_rewrite(self):
        iid = self._ask()
        j = self.post("/api/feedback", {"interaktion_id": iid, "korrektur": "650 g"}).json()
        self.assertIsNone(j["korrektur_vorschlag"])
        self.assertEqual(self.brain.rewrite_calls, [])
        self.assertIsNone(self.brain.feedback_calls[-1]["correction_full"])

    def test_unknown_id_404(self):
        r = self.post("/api/feedback", {"interaktion_id": 99999, "bewertung": 1})
        self.assertEqual(r.status, 404)
        self.assertFalse(r.json()["ok"])
        r = self.post("/api/feedback", {"interaktion_id": 99999, "korrektur": "x", "umschreiben": True})
        self.assertEqual(r.status, 404)

    def test_bad_feedback_400(self):
        iid = self._ask()
        self.assertEqual(self.post("/api/feedback", {"bewertung": 1}).status, 400)
        self.assertEqual(self.post("/api/feedback", {"interaktion_id": "abc"}).status, 400)
        self.assertEqual(self.post("/api/feedback", {"interaktion_id": iid, "bewertung": 5}).status, 400)
        self.assertEqual(self.post("/api/feedback", {"interaktion_id": iid, "bewertung": True}).status, 400)
        self.assertEqual(self.post("/api/feedback", {"interaktion_id": iid, "umschreiben": True}).status, 400)
        self.assertEqual(self.post("/api/feedback", {"interaktion_id": iid, "kommentar": 7}).status, 400)


class ErinnerungenTest(ServerTestBase):
    def setUp(self):
        super().setUp()
        for m in self.brain.memory.recent(1000):
            self.brain.memory.forget(m.id)

    def test_post_get_delete(self):
        r = self.post("/api/erinnerungen", {"inhalt": "Nutzer bevorzugt Carbon für Drohnenrahmen", "art": "praeferenz",
                                            "projekt": "Drohne", "wichtigkeit": 0.9, "tags": ["Material", "drohne"]})
        self.assertEqual(r.status, 200)
        j = r.json()
        self.assertTrue(j["ok"])
        e = j["erinnerung"]
        self.assertEqual(set(e), {"id", "art", "inhalt", "tags", "projekt", "quelle", "wichtigkeit", "erstellt", "score"})
        self.assertEqual(e["art"], "praeferenz")
        self.assertEqual(e["tags"], ["drohne", "material"])
        self.assertEqual(e["projekt"], "Drohne")
        self.assertEqual(e["quelle"], "nutzer")
        self.assertAlmostEqual(e["wichtigkeit"], 0.9)
        self.assertNotIn("embedding", e)
        mid = e["id"]

        self.post("/api/erinnerungen", {"inhalt": "Lieblingsfarbe ist blau"})
        j = self.get("/api/erinnerungen").json()
        self.assertTrue(j["ok"])
        self.assertEqual([x["inhalt"] for x in j["erinnerungen"]],
                         ["Lieblingsfarbe ist blau", "Nutzer bevorzugt Carbon für Drohnenrahmen"])
        j = self.get("/api/erinnerungen?n=1").json()
        self.assertEqual(len(j["erinnerungen"]), 1)

        j = self.get("/api/erinnerungen?q=Carbon%20Drohne&n=5&projekt=Drohne").json()
        self.assertEqual([x["id"] for x in j["erinnerungen"]], [mid])
        j = self.get("/api/erinnerungen?q=Quantenphysik").json()
        self.assertEqual(j["erinnerungen"], [])

        r = self.delete(f"/api/erinnerungen/{mid}")
        self.assertEqual(r.status, 200)
        self.assertTrue(r.json()["ok"])
        r = self.delete(f"/api/erinnerungen/{mid}")
        self.assertEqual(r.status, 404)
        self.assertFalse(r.json()["ok"])
        self.assertEqual(self.delete("/api/erinnerungen/abc").status, 404)

    def test_default_importance_and_tags_as_string(self):
        j = self.post("/api/erinnerungen", {"inhalt": "Akku 4S 1500 mAh", "tags": "akku, lipo"}).json()
        self.assertAlmostEqual(j["erinnerung"]["wichtigkeit"], 0.6)
        self.assertEqual(j["erinnerung"]["tags"], ["akku", "lipo"])
        self.assertEqual(j["erinnerung"]["art"], "notiz")

    def test_bad_memory_requests(self):
        self.assertEqual(self.post("/api/erinnerungen", {}).status, 400)
        self.assertEqual(self.post("/api/erinnerungen", {"inhalt": ""}).status, 400)
        self.assertEqual(self.post("/api/erinnerungen", {"inhalt": "x", "wichtigkeit": 3}).status, 400)
        self.assertEqual(self.post("/api/erinnerungen", {"inhalt": "x", "wichtigkeit": "hoch"}).status, 400)
        self.assertEqual(self.post("/api/erinnerungen", {"inhalt": "x", "tags": [1, 2]}).status, 400)
        self.assertEqual(self.post("/api/erinnerungen", {"inhalt": "x", "art": "gefuehl"}).status, 400)
        self.assertEqual(self.get("/api/erinnerungen?n=abc").status, 400)
        self.assertEqual(self.get("/api/erinnerungen?n=0").status, 400)
        self.assertEqual(self.get("/api/erinnerungen?n=99999").status, 400)
        self.assertEqual(self.brain.memory.count(), 0)

    def test_memory_to_dict_format(self):
        m = self.brain.memory.remember("Format-Test", kind="fakt", tags=["a"], project="P", importance=0.7)
        d = memory_to_dict(m)
        self.assertEqual(d["id"], m.id)
        self.assertEqual(d["art"], "fakt")
        self.assertEqual(d["inhalt"], "Format-Test")
        self.assertEqual(d["projekt"], "P")
        self.assertIsInstance(d["erstellt"], str)
        self.assertRegex(d["erstellt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
        self.assertNotIn("embedding", d)


class LektionenTest(ServerTestBase):
    def test_list_and_delete(self):
        self.brain.learning.lessons.clear()
        a = self.brain.learning.add("Immer Einheiten angeben")
        b = self.brain.learning.add("Alte Regel", active=False)
        j = self.get("/api/lektionen").json()
        self.assertTrue(j["ok"])
        self.assertEqual([l["id"] for l in j["lektionen"]], [a.id])
        self.assertEqual(set(j["lektionen"][0]), {"id", "regel", "bereich", "themen", "quelle", "erstellt",
                                                  "geholfen", "geschadet", "aktiv"})
        j = self.get("/api/lektionen?alle=1").json()
        self.assertEqual([l["id"] for l in j["lektionen"]], [a.id, b.id])
        r = self.delete(f"/api/lektionen/{a.id}")
        self.assertEqual(r.status, 200)
        self.assertTrue(r.json()["ok"])
        self.assertEqual(self.delete(f"/api/lektionen/{a.id}").status, 404)
        self.assertEqual(self.get("/api/lektionen").json()["lektionen"], [])


class ConcurrencyTest(ServerTestBase):
    def test_parallel_requests_are_served(self):
        results: list[int] = []

        def worker():
            results.append(self.get("/api/status").status)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(results, [200] * 8)


if __name__ == "__main__":
    unittest.main()
