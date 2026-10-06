"""Denkkern von OBITO.

:class:`Brain` verbindet Modell-Anbindung, Gedächtnis, Lernen, Werkzeuge und das
Expertengremium zu einem Ablauf je Frage:

1. **Erinnern** – Verlauf, passende Erinnerungen, Lektionen und Beispiele laden.
2. **Routen** – Komplexität schätzen (Heuristik, bei Bedarf Modell) und die Tiefe wählen.
3. **Gremium** – Experten antworten parallel (``mittel``: 3, ``tief``: 5).
4. **Kritiker** und **Revision** (nur ``tief``).
5. **Synthese** – OMEGA formt die finale Antwort, mit Werkzeugen, gestreamt.
6. **Lernen** – Fakten merken, Interaktion protokollieren, Feedback verarbeiten.

Alles läuft lokal; jede Stufe hat ein Budget, einen Normalisierer, genau einen Retry und
einen definierten Fallback. ``progress``/``stream`` werden ausschließlich im aufrufenden
Thread gerufen.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from . import agents
from .agents import (CONSOLIDATION_SCHEMA, CRITIC_SCHEMA, EXPERTS, LESSON_SCHEMA, MEMORY_SCHEMA,
                     ROUTING_SCHEMA, Expert)
from .automation import AutomationStore, Scheduler
from .config import Config, find_config_file, save_config
from .devices import DeviceStore
from .devices import register_tools as register_device_tools
from .geo import WaypointStore
from .geo import register_tools as register_geo_tools
from .geometry import ModelStore
from .geometry import register_tools as register_geometry_tools
from .knowledge import Chunk, KnowledgeStore
from .knowledge import register_tools as register_knowledge_tools
from .learning import Interaction, LearningStore, Lesson, keywords
from .llm import (BackendUnavailable, ChatResult, LLMBackend, LLMError, ModelInfo, ModelNotFound,
                  StreamCallback, make_backend, parse_json)
from .memory import Memory, MemoryStore, normalize
from .missions import MissionRunner, MissionStore
from .projects import ProjectStore
from .simulation import register_tools as register_simulation_tools
from .sysmon import Sampler
from .sysmon import register_tools as register_sysmon_tools
from .tools import (ToolRegistry, ToolStreamFilter, default_registry, needs_tools, parse_tool_calls,
                    strip_tool_calls)

ProgressCallback = Callable[["Step"], None]

DEPTHS = ("auto", "schnell", "mittel", "tief")
COMPLEXITY_TO_DEPTH = {"einfach": "schnell", "mittel": "mittel", "komplex": "tief"}

MIN_RECALL_IMPORTANCE = 0.15        # Erinnerungen unterhalb werden nicht mehr abgerufen
MIN_LEARN_ANSWER_CHARS = 40         # kürzere Antworten lösen keine Gedächtnis-Extraktion aus
MIN_MEMORY_IMPORTANCE = 0.4         # Mindest-Wichtigkeit laut Modell
MAX_AUTO_MEMORY_IMPORTANCE = 0.5    # automatisch gemerkte Fakten bleiben „unsicher“
MEMORY_CONTENT_MIN_CHARS = 12
MEMORY_CONTENT_MAX_CHARS = 300
MAX_MEMORY_TAGS = 5
SHORT_CORRECTION_CHARS = 200        # kürzere Korrekturen werden zur vollen Antwort umgeschrieben
CONTEXT_RESERVE_TOKENS = 256
TOOL_LIMIT_HINT = "Hinweis: Werkzeug-Limit erreicht – Antwort ohne weiteres Werkzeug."
JSON_RETRY_HINT = "Antworte ausschließlich mit einem JSON-Objekt nach diesem Schema: "

# Floskeln, die das Gedächtnis-Modul kleiner Modelle gern als „Fakt“ ausgibt
MEMORY_PHRASE_BLOCKLIST = (
    "nutzer hat gefragt", "der nutzer fragt", "nutzer fragte", "die frage war", "die frage des nutzers",
    "wurde gefragt", "nutzer möchte wissen", "nutzer moechte wissen", "die antwort war",
    "die ki hat", "ich habe geantwortet",
)

_START_LABELS = {
    "routing": "Router ordnet die Frage ein…",
    "kritiker": "KRITIKER prüft die Antworten…",
    "extraktion": "Gedächtnis-Modul sucht merkwürdige Fakten…",
    "lektion": "Lern-Modul leitet eine Regel ab…",
}


def _iso(ts: float) -> str:
    """Zeitstempel als ISO-8601 in lokaler Zeit."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def memory_to_dict(m: Memory) -> dict:
    """Serialisiert eine Erinnerung für Server/CLI – ohne Embedding."""
    return {
        "id": m.id,
        "art": m.kind,
        "inhalt": m.content,
        "tags": list(m.tags or []),
        "projekt": m.project,
        "quelle": m.source,
        "wichtigkeit": m.importance,
        "erstellt": _iso(m.created_at),
        "score": m.score,
    }


# ------------------------------------------------------------------ Datenklassen
@dataclass
class Step:
    """Ein Schritt der Denk-Spur.

    ``stage`` ∈ erinnern|routing|experte|kritiker|revision|synthese|werkzeug|schnell|lernen|system,
    ``status`` ∈ start|fertig|fehler. Schritte mit ``status="start"`` gehen nur an ``progress``."""

    stage: str
    who: str
    summary: str
    detail: str = ""
    duration: float = 0.0
    tokens: int = 0
    status: str = "fertig"

    def to_dict(self) -> dict:
        return {
            "stufe": self.stage,
            "wer": self.who,
            "zusammenfassung": self.summary,
            "detail": self.detail,
            "dauer": round(float(self.duration), 3),
            "tokens": int(self.tokens),
            "status": self.status,
        }


@dataclass
class Answer:
    """Antwort des Denkkerns samt Spur, genutzten Erinnerungen und Protokoll-ID."""

    text: str
    question: str
    session_id: str
    project: str | None
    depth: str                      # schnell | mittel | tief | fehler
    experts: list[str]
    critique: dict | None           # Kritiker-JSON + "revidiert": [IDs]
    memories_used: list[Memory]
    new_memories: list[Memory]
    lessons_used: list[Lesson]
    tools_used: list[str]
    steps: list[Step]
    interaction_id: int | None
    tokens: int
    duration: float
    documents: list[Chunk] = field(default_factory=list)   # genutzte Dokument-Auszüge (Wissensbasis)

    def trace(self) -> str:
        """Lesbare Denk-Spur, eine Zeile je Schritt."""
        marks = {"fertig": "✓", "fehler": "✗", "start": "⟳"}
        lines = []
        for s in self.steps:
            extra = []
            if s.duration:
                extra.append(f"{s.duration:.1f} s")
            if s.tokens:
                extra.append(f"{s.tokens} Tokens")
            line = f"{marks.get(s.status, '•')} {s.stage}/{s.who}: {s.summary}"
            if extra:
                line += f" [{', '.join(extra)}]"
            lines.append(line)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "antwort": self.text,
            "frage": self.question,
            "sitzung": self.session_id,
            "projekt": self.project,
            "tiefe": self.depth,
            "experten": list(self.experts),
            "kritik": self.critique,
            "erinnerungen_genutzt": [memory_to_dict(m) for m in self.memories_used],
            "erinnerungen_neu": [memory_to_dict(m) for m in self.new_memories],
            "werkzeuge": list(self.tools_used),
            "dokumente": [c.to_dict() for c in self.documents],
            "spur": [s.to_dict() for s in self.steps],
            "interaktion_id": self.interaction_id,
            "tokens": self.tokens,
            "dauer": self.duration,
        }

    def trace_json(self, max_detail: int = 1000) -> str:
        """JSON-Liste der Schritte mit gekürztem ``detail`` – genau dieser String wird protokolliert."""
        max_detail = max(0, int(max_detail))
        out = []
        for s in self.steps:
            d = s.to_dict()
            d["detail"] = (d["detail"] or "")[:max_detail]
            out.append(d)
        return json.dumps(out, ensure_ascii=False)


@dataclass
class Feedback:
    """Ergebnis von :meth:`Brain.feedback`."""

    interaction: Interaction
    lessons: list[Lesson]
    memories_adjusted: int
    correction_full: str | None


@dataclass
class _JsonOutcome:
    """Ergebnis eines JSON-Stufenaufrufs (intern)."""

    value: Any
    raw: str
    duration: float
    tokens: int
    retried: bool


@dataclass
class _Run:
    """Zustand eines ``ask``-Durchlaufs (intern)."""

    question: str
    session_id: str
    project: str | None
    stream: StreamCallback | None
    progress: ProgressCallback | None
    cancel: threading.Event | None
    t0: float
    deadline_at: float
    steps: list[Step] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    experts: list[str] = field(default_factory=list)
    critique: dict | None = None
    memories: list[Memory] = field(default_factory=list)
    lessons: list[Lesson] = field(default_factory=list)
    new_memories: list[Memory] = field(default_factory=list)
    documents: list[Chunk] = field(default_factory=list)
    extra_context: str = ""          # Projektkontext + Dokument-Auszüge
    depth: str = "auto"


class _AnswerStream:
    """Formt den Strom der Werkzeugschleife so, dass der gestreamte Text exakt ``Answer.text``
    entspricht: führende/abschließende Leerzeichen eines Abschnitts werden zurückgehalten,
    zwischen nicht-leeren Abschnitten wird ``"\\n\\n"`` gesendet."""

    def __init__(self, emit: StreamCallback | None):
        self.emit = emit
        self._pending_ws = ""
        self._started = False
        self._need_sep = False

    def begin_part(self) -> None:
        self._pending_ws = ""
        self._started = False

    def feed(self, piece: str) -> None:
        if self.emit is None or not piece:
            return
        if not self._started:
            piece = piece.lstrip()
            if not piece:
                return
            self._started = True
            if self._need_sep:
                self.emit("\n\n")
        text = self._pending_ws + piece
        core = text.rstrip()
        self._pending_ws = text[len(core):]
        if core:
            self.emit(core)

    def end_part(self) -> None:
        self._pending_ws = ""
        if self._started:
            self._need_sep = True
        self._started = False


# ------------------------------------------------------------------ Brain
class Brain:
    """Der Denkkern: ein Objekt je Prozess, thread-sicher über ``_ask_lock`` (eine Frage zugleich)."""

    def __init__(self, cfg: Config, backend: LLMBackend | None = None, memory: MemoryStore | None = None,
                 learning: LearningStore | None = None, tools: ToolRegistry | None = None,
                 confirm: Callable[[str, dict], bool] | None = None):
        self.cfg = cfg
        self.backend: LLMBackend = backend if backend is not None else make_backend(cfg)
        cfg.ensure_dirs()
        self.memory: MemoryStore = memory if memory is not None else MemoryStore(cfg.memory_db)
        self.learning: LearningStore = learning if learning is not None else LearningStore(cfg.learning_db)
        if tools is None:
            tools = default_registry(cfg.workspace or ".", confirm, cfg.confirm_dangerous)
        elif confirm is not None:
            tools.set_policy(confirm, tools.confirm_dangerous)
        self.tools: ToolRegistry = tools
        self._confirm = confirm if confirm is not None else getattr(tools, "confirm", None)
        # Phase 2: Wissensbasis, Projekte, Missionen, Automationen
        self.knowledge = KnowledgeStore(cfg.knowledge_db)
        self.projects = ProjectStore(cfg.projects_db)
        self.missions = MissionRunner(self, MissionStore(cfg.missions_db), confirm=None)
        self.automation = Scheduler(self, AutomationStore(cfg.automation_db),
                                    log_path=cfg.logs_dir / "automation.log", allow_dangerous=False,
                                    tick_seconds=30)
        register_knowledge_tools(self.tools, self.knowledge)
        # Phase 3: Geräte, System, 3D-Modelle, Simulation, Welt
        self.devices = DeviceStore(cfg.devices_db)
        self.models3d = ModelStore(cfg.models3d_db, cfg.models3d_dir)
        self.geo = WaypointStore(cfg.geo_db)
        self.sysmon = Sampler(size=120)
        register_device_tools(self.tools, self.devices)
        register_sysmon_tools(self.tools, backend=self.backend, data_dir=str(cfg.data_path), sampler=self.sysmon)
        register_geometry_tools(self.tools, self.models3d)
        register_simulation_tools(self.tools)
        register_geo_tools(self.tools, self.geo, lambda: bool(self.cfg.online))

        # Embedder: echtes Backend immer, Fake nur wenn „oben“. Liefert das Backend None
        # (kein Embedding-Modell, Server weg), fällt die Suche auf Volltext zurück.
        self._embedder: Callable[[Sequence[str]], list[list[float]] | None] | None = None
        is_fake = getattr(self.backend, "name", "") == "fake"
        if not is_fake or bool(getattr(self.backend, "up", True)):
            backend_ref = self.backend
            embed_model = cfg.embed_model

            def embedder(texts: Sequence[str]) -> list[list[float]] | None:
                return backend_ref.embed(list(texts), model=embed_model)

            self._embedder = embedder
            self.memory.set_embedder(embedder, cfg.embed_model)
            set_le = getattr(self.learning, "set_embedder", None)
            if callable(set_le):
                set_le(embedder)
            self.knowledge.set_embedder(embedder)
        self.tools.set_memory(self.memory)

        self._ask_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------ Eigenschaften
    @property
    def busy(self) -> bool:
        """``True``, solange eine Frage bearbeitet wird."""
        return self._ask_lock.locked()

    # ------------------------------------------------------------ Hilfen
    @staticmethod
    def _emit(progress: ProgressCallback | None, step: Step) -> None:
        if progress is not None:
            progress(step)

    @staticmethod
    def _check_cancel(cancel: threading.Event | None) -> None:
        if cancel is not None and cancel.is_set():
            raise LLMError("Abgebrochen")

    def _think_flag(self) -> bool | None:
        think = str(self.cfg.think or "auto").strip().lower()
        return None if think == "auto" else think == "an"

    def _chat(self, messages: Sequence[dict], *, max_tokens: int, model: str | None = None,
              temperature: float | None = None, stream: StreamCallback | None = None,
              json_mode: bool | dict = False) -> ChatResult:
        """Ein Modellaufruf mit Budget: ``fit_messages`` vor jedem Aufruf, Laufzeitoptionen aus ``cfg``."""
        cfg = self.cfg
        budget = int(cfg.num_ctx) - int(max_tokens) - CONTEXT_RESERVE_TOKENS
        msgs = agents.fit_messages(messages, budget)
        model = model or cfg.model
        try:
            return self.backend.chat(
                msgs, model=model, temperature=temperature, max_tokens=int(max_tokens), json_mode=json_mode,
                stream=stream, timeout=cfg.timeout, num_ctx=cfg.num_ctx, keep_alive=cfg.keep_alive,
                think=self._think_flag(),
            )
        except ModelNotFound as e:
            if not getattr(e, "model_name", None):
                e.model_name = model  # type: ignore[attr-defined]
            raise

    @staticmethod
    def _safe_parse(parser: Callable[[str], Any], text: str) -> Any:
        try:
            return parser(text)
        except Exception:  # noqa: BLE001 – Normalisierer dürfen den Ablauf nie stoppen
            return None

    def _json_call(self, stage: str, messages: Sequence[dict], schema: dict, parser: Callable[[str], Any], *,
                   model: str, max_tokens: int, who: str = "system", steps: list[Step] | None = None,
                   progress: ProgressCallback | None = None,
                   cancel: threading.Event | None = None) -> _JsonOutcome:
        """JSON-Stufe: Versuch 1 mit Schema und ``temperature=0``; liefert der Normalisierer ``None``,
        genau ein Retry mit angehängter Nutzer-Nachricht und ``json_mode=True``. Bleibt es bei
        ``None``, wird ein Hinweis-Step angehängt; den Fallback bestimmt der Aufrufer."""
        steps = steps if steps is not None else []
        self._check_cancel(cancel)
        self._emit(progress, Step(stage, who, _START_LABELS.get(stage, f"Stufe {stage}…"), status="start"))
        t0 = time.time()
        try:
            res = self._chat(messages, max_tokens=max_tokens, model=model, temperature=0.0, json_mode=schema)
            raw = res.text or ""
            tokens = res.total_tokens
            value = self._safe_parse(parser, raw)
            retried = False
            if value is None:
                retried = True
                self._check_cancel(cancel)
                # Hinweis an die letzte Nutzer-Nachricht anhängen (keine eigene Nachricht, sonst würde
                # fit_messages bei knappem Budget die eigentliche Aufgabe statt des Hinweises kürzen)
                hint = JSON_RETRY_HINT + json.dumps(schema, ensure_ascii=False)
                retry_msgs = [dict(m) for m in messages]
                if retry_msgs and retry_msgs[-1].get("role") == "user":
                    retry_msgs[-1]["content"] = f"{retry_msgs[-1].get('content', '')}\n\n{hint}"
                else:
                    retry_msgs.append({"role": "user", "content": hint})
                res = self._chat(retry_msgs, max_tokens=max_tokens, model=model, temperature=0.0,
                                 json_mode=True)
                raw = res.text or ""
                tokens += res.total_tokens
                value = self._safe_parse(parser, raw)
        except LLMError as e:
            step = Step(stage, who, f"Fehler – {e}", duration=time.time() - t0, status="fehler")
            steps.append(step)
            self._emit(progress, step)
            raise
        duration = time.time() - t0
        if value is None:
            step = Step(stage, "system",
                        f"Hinweis: Stufe »{stage}« lieferte auch nach Wiederholung kein verwertbares JSON – "
                        f"Fallback wird verwendet.",
                        detail=raw[:2000], duration=duration, tokens=tokens, status="fehler")
            steps.append(step)
            self._emit(progress, step)
        return _JsonOutcome(value, raw, duration, tokens, retried)

    # ------------------------------------------------------------ ask
    def ask(self, question: str, *, session_id: str = "standard", project: str | None = None,
            depth: str | None = None, stream: StreamCallback | None = None,
            progress: ProgressCallback | None = None, learn: bool | None = None,
            cancel: threading.Event | None = None) -> Answer:
        """Beantwortet eine Frage über den vollen Denkablauf (siehe Modul-Doku).

        ``depth``: ``None`` = ``cfg.depth``; ``learn``: ``None`` = ``cfg.auto_memory`` für das
        automatische Merken, ``False`` schaltet zusätzlich die Protokollierung ab (Eval-Läufe).
        Modellfehler führen nie zu einer Exception, sondern zu einer Antwort mit ``depth="fehler"``."""
        question = (question or "").strip()
        if not question:
            raise ValueError("Leere Frage.")
        requested = str(depth or self.cfg.depth or "auto").strip().lower()
        if requested not in DEPTHS:
            raise ValueError(f"Unbekannte Tiefe {requested!r} – erlaubt: {', '.join(DEPTHS)}.")
        session_id = str(session_id or "standard")
        if project is not None:
            project = str(project).strip() or None
        with self._ask_lock:
            t0 = time.time()
            run = _Run(question=question, session_id=session_id, project=project, stream=stream,
                       progress=progress, cancel=cancel, t0=t0, deadline_at=t0 + float(self.cfg.deadline))
            return self._ask_locked(run, requested, learn)

    def _ask_locked(self, run: _Run, requested: str, learn: bool | None) -> Answer:
        cfg = self.cfg
        try:
            self._check_cancel(run.cancel)
            # 2. Erinnern (Gedächtnis, Lektionen, Beispiele, Projektkontext, Dokumente)
            history, examples = self._recall_stage(run)
            context = agents.context_block(run.memories, run.lessons, run.project, run.extra_context)
            # 3. Tiefe
            self._check_cancel(run.cancel)
            depth, hint, routing_tools = self._routing_stage(run, requested)
            run.depth = depth
            # 4. Werkzeugblock
            tools_block = ""
            if cfg.allow_tools and (needs_tools(run.question, history) or routing_tools):
                tools_block = self.tools.describe(compact=True)
            # 5./6. Antwort
            self._check_cancel(run.cancel)
            if depth == "schnell":
                messages = agents.fast_messages(run.question, context, history, examples, tools_block)
                text = self._tool_loop(messages, stage="schnell", max_tokens=cfg.max_tokens_fast,
                                       stream=run.stream, steps=run.steps, tools_used=run.tools_used,
                                       tools_block=tools_block, progress=run.progress, cancel=run.cancel)
            else:
                text = self._council_stage(run, depth, hint, context, history, examples, tools_block)
            text = (text or "").strip()
            if not text:
                raise LLMError("Leere Antwort vom Modell.")
        except LLMError as e:
            return self._error_answer(run, e)

        # 8. Verlauf und Lernen
        self.memory.add_message(run.session_id, "user", run.question, run.project)
        self.memory.add_message(run.session_id, "assistant", text, run.project)
        auto_memory = bool(cfg.auto_memory) if learn is None else bool(learn)
        if auto_memory and len(text) >= MIN_LEARN_ANSWER_CHARS:
            self._learn_stage(run, text)

        duration = time.time() - run.t0
        tokens = sum(int(s.tokens) for s in run.steps)
        answer = Answer(
            text=text, question=run.question, session_id=run.session_id, project=run.project, depth=depth,
            experts=list(run.experts), critique=run.critique, memories_used=list(run.memories),
            new_memories=list(run.new_memories), lessons_used=list(run.lessons), tools_used=list(run.tools_used),
            steps=run.steps, interaction_id=None, tokens=tokens, duration=duration,
            documents=list(run.documents),
        )
        # 9. Protokoll
        if learn is not False:
            try:
                answer.interaction_id = self.learning.record(
                    run.session_id, run.question, text, project=run.project, experts=answer.experts,
                    depth=depth, trace=answer.trace_json(), model=cfg.model, context=context, history=history,
                    memories_used=[m.id for m in run.memories], new_memories=[m.id for m in run.new_memories],
                    lessons_used=[l.id for l in run.lessons], tools_used=answer.tools_used, tokens=tokens,
                    duration=duration,
                )
            except Exception as e:  # noqa: BLE001 – Protokollfehler dürfen die Antwort nicht kosten
                step = Step("system", "system", f"Protokollierung fehlgeschlagen: {e}", status="fehler")
                run.steps.append(step)
                self._emit(run.progress, step)
        return answer

    # ------------------------------------------------------------ Stufen
    def _recall_stage(self, run: _Run) -> tuple[list[dict], list[tuple[str, str]]]:
        cfg = self.cfg
        t0 = time.time()
        problems: list[str] = []
        history: list[dict] = []
        examples: list[tuple[str, str]] = []
        try:
            history = self.memory.history(run.session_id, cfg.max_history)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Verlauf: {e}")
        try:
            run.memories = self.memory.search(run.question, cfg.memory_recall, run.project,
                                              min_importance=MIN_RECALL_IMPORTANCE)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Erinnerungen: {e}")
        try:
            run.lessons = self.learning.lessons(run.question, k=3)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Lektionen: {e}")
        try:
            examples = self.learning.examples(run.question, cfg.example_recall)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Beispiele: {e}")
        extra_parts: list[str] = []
        project_context = False
        if run.project:
            try:
                self.projects.ensure(run.project)
                project_summary = self.projects.summary(run.project)
                if project_summary:
                    extra_parts.append("Projektkontext:\n" + project_summary)
                    project_context = True
            except Exception as e:  # noqa: BLE001
                problems.append(f"Projekt: {e}")
        try:
            run.documents = self.knowledge.search(run.question, k=4, project=run.project)
            if run.documents:
                section = self.knowledge.context_section(run.question, k=4, max_chars=1500, project=run.project)
                if section:
                    extra_parts.append(section)
        except Exception as e:  # noqa: BLE001
            problems.append(f"Dokumente: {e}")
        run.extra_context = "\n\n".join(extra_parts)
        summary = (f"{len(run.memories)} Erinnerungen, {len(run.lessons)} Lektionen, {len(examples)} Beispiele, "
                   f"{len(run.documents)} Dokument-Auszüge, {len(history)} Verlaufsnachrichten")
        if project_context:
            summary += ", Projektkontext"
        detail_lines = ([m.short() for m in run.memories] + [l.render() for l in run.lessons]
                        + [f"{c.cite()} {c.content[:120]}" for c in run.documents])
        if problems:
            summary += " – Fehler: " + "; ".join(problems)
            detail_lines.extend(problems)
        step = Step("erinnern", "system", summary, detail="\n".join(detail_lines), duration=time.time() - t0,
                    status="fehler" if problems else "fertig")
        run.steps.append(step)
        self._emit(run.progress, step)
        return history, examples

    def _routing_stage(self, run: _Run, requested: str) -> tuple[str, list[str] | None, bool]:
        """Liefert ``(tiefe, experten_hinweis, werkzeuge_laut_routing)``."""
        cfg = self.cfg
        if requested != "auto":
            step = Step("routing", "system", f"vorgegeben → {requested}")
            run.steps.append(step)
            self._emit(run.progress, step)
            return requested, None, False

        heuristic = agents.heuristic_complexity(run.question)
        routing: dict | None = None
        outcome: _JsonOutcome | None = None
        # Modell-Routing für alles außer eindeutig einfachen Fragen: bei komplexen Fragen entscheidet
        # die Expertenwahl über die Qualität, ein kleiner JSON-Aufruf fällt dort nicht ins Gewicht.
        if heuristic != "einfach" or cfg.fast_model:
            outcome = self._json_call(
                "routing", agents.routing_messages(run.question, None), ROUTING_SCHEMA,
                lambda t: agents.parse_routing(t, EXPERTS.keys()), model=cfg.routing_model,
                max_tokens=cfg.max_tokens_json, steps=run.steps, progress=run.progress, cancel=run.cancel,
            )
            if isinstance(outcome.value, dict) and outcome.value.get("komplexitaet") in COMPLEXITY_TO_DEPTH:
                routing = outcome.value
        if routing is not None:
            complexity = routing["komplexitaet"]
            hint = [e for e in routing.get("experten") or [] if e in EXPERTS] or None
            tools = bool(routing.get("werkzeuge"))
            detail = routing.get("begruendung") or ""
        else:
            complexity, hint, tools = heuristic, None, False
            detail = f"Heuristik: {heuristic}" + (" (Routing ohne verwertbares Ergebnis)" if outcome else "")
        depth = COMPLEXITY_TO_DEPTH[complexity]
        summary = f"{complexity} → {depth}"
        if hint:
            summary += f" ({', '.join(hint)})"
        elif routing is None:
            summary += " (Heuristik)"
        step = Step("routing", "system", summary, detail=detail,
                    duration=outcome.duration if outcome else 0.0,
                    tokens=outcome.tokens if outcome and outcome.value is not None else 0)
        run.steps.append(step)
        self._emit(run.progress, step)
        return depth, hint, tools

    def _worker_chat(self, messages: Sequence[dict], max_tokens: int, temperature: float,
                     cancel: threading.Event | None,
                     stop: threading.Event | None = None) -> tuple[ChatResult, float, int]:
        """Modellaufruf in einem Arbeits-Thread: gestreamt mit internem Zähler, ohne Rückrufe nach außen.
        ``stop`` wird vom Gremium gesetzt (Zeitbudget/Abbruch), damit laufende Aufrufe beim nächsten
        Textstück aussteigen und das Backend freigeben."""
        counter = {"n": 0}

        def on_piece(piece: str) -> None:
            counter["n"] += 1
            if (cancel is not None and cancel.is_set()) or (stop is not None and stop.is_set()):
                raise LLMError("Abgebrochen")

        t0 = time.time()
        res = self._chat(messages, max_tokens=max_tokens, model=self.cfg.model, temperature=temperature,
                         stream=on_piece)
        return res, time.time() - t0, counter["n"]

    def _parallel(self, run: _Run, jobs: list[tuple[Expert, list[dict]]], *, stage: str,
                  max_tokens: int) -> tuple[dict[str, str], LLMError | None]:
        """Führt Experten-/Revisionsaufrufe parallel aus. Rückgabe: erfolgreiche Antworten in
        Job-Reihenfolge und der erste Modellfehler (falls einer auftrat)."""
        results: dict[str, str] = {}
        first_error: LLMError | None = None
        if not jobs:
            return results, None
        for ex, _ in jobs:
            self._emit(run.progress, Step(stage, ex.id, f"{ex.id} ({ex.name}) denkt…", status="start"))
        workers = max(1, min(len(jobs), int(self.cfg.parallel_calls or 1)))
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"obito-{stage}")
        futures: dict = {}
        stop = threading.Event()
        try:
            for ex, msgs in jobs:
                futures[pool.submit(self._worker_chat, msgs, max_tokens, ex.temperature, run.cancel, stop)] = ex
            pending = set(futures)
            remaining = max(0.05, run.deadline_at - time.time())
            try:
                for fut in as_completed(futures, timeout=remaining):
                    pending.discard(fut)
                    ex = futures[fut]
                    if run.cancel is not None and run.cancel.is_set():
                        stop.set()
                        for other in pending:
                            oex = futures[other]
                            step = Step(stage, oex.id, f"{oex.id} ({oex.name}): Fehler – Abgebrochen", status="fehler")
                            run.steps.append(step)
                            self._emit(run.progress, step)
                        raise LLMError("Abgebrochen")
                    try:
                        res, dur, _chunks = fut.result()
                    except LLMError as e:
                        if first_error is None:
                            first_error = e
                        step = Step(stage, ex.id, f"{ex.id} ({ex.name}): Fehler – {e}", status="fehler")
                    except Exception as e:  # noqa: BLE001 – ein defekter Experte stoppt nicht das Gremium
                        step = Step(stage, ex.id, f"{ex.id} ({ex.name}): Fehler – {e.__class__.__name__}: {e}",
                                    status="fehler")
                    else:
                        text = (res.text or "").strip()
                        if text:
                            results[ex.id] = text
                            step = Step(stage, ex.id, f"{ex.id} ({ex.name}): {len(text)} Zeichen", detail=text,
                                        duration=dur, tokens=res.total_tokens)
                        else:
                            step = Step(stage, ex.id, f"{ex.id} ({ex.name}): Fehler – leere Antwort",
                                        duration=dur, tokens=res.total_tokens, status="fehler")
                    run.steps.append(step)
                    self._emit(run.progress, step)
            except FuturesTimeout:
                stop.set()
                for fut in pending:
                    ex = futures[fut]
                    step = Step(stage, ex.id,
                                f"{ex.id} ({ex.name}): Fehler – Zeitbudget ({self.cfg.deadline:.0f} s) überschritten",
                                status="fehler")
                    run.steps.append(step)
                    self._emit(run.progress, step)
                if first_error is None:
                    first_error = LLMError(f"Zeitbudget ({self.cfg.deadline:.0f} s) überschritten")
        finally:
            stop.set()      # laufende Aufrufe steigen beim nächsten Textstück aus
            pool.shutdown(wait=False, cancel_futures=True)
        ordered = {ex.id: results[ex.id] for ex, _ in jobs if ex.id in results}
        return ordered, first_error

    def _council_stage(self, run: _Run, depth: str, hint: list[str] | None, context: str,
                       history: list[dict], examples: list[tuple[str, str]], tools_block: str) -> str:
        cfg = self.cfg
        k = cfg.experts_medium if depth == "mittel" else cfg.experts_per_question
        experts = agents.select_experts(run.question, k, hint)
        jobs = [(ex, agents.expert_messages(ex, run.question, context, history)) for ex in experts]
        answers, first_error = self._parallel(run, jobs, stage="experte", max_tokens=cfg.max_tokens_expert)
        self._check_cancel(run.cancel)
        run.experts = [ex.id for ex in experts if ex.id in answers]
        if not answers:
            raise first_error or LLMError("Kein Experte hat eine Antwort geliefert.")

        critique: dict | None = None
        if depth == "tief":
            critique = self._critic_stage(run, experts, answers, context)
            run.critique = critique
        self._check_cancel(run.cancel)
        messages = agents.synthesis_messages(run.question, answers, critique, context, history, examples,
                                             tools_block, medium=(depth == "mittel"))
        return self._tool_loop(messages, stage="synthese", max_tokens=cfg.max_tokens_synthesis,
                               stream=run.stream, steps=run.steps, tools_used=run.tools_used,
                               tools_block=tools_block, progress=run.progress, cancel=run.cancel)

    def _critic_stage(self, run: _Run, experts: list[Expert], answers: dict[str, str], context: str) -> dict:
        cfg = self.cfg
        outcome = self._json_call(
            "kritiker", agents.critic_messages(run.question, answers, context), CRITIC_SCHEMA,
            lambda t: agents.parse_critique(t, list(answers)), model=cfg.model, max_tokens=cfg.max_tokens_critic,
            who=agents.CRITIC.id, steps=run.steps, progress=run.progress, cancel=run.cancel,
        )
        valid = isinstance(outcome.value, dict)
        if valid:
            critique = dict(outcome.value)
            rating = critique.get("bewertung")
            n_err = len(critique.get("fehler") or [])
            n_con = len(critique.get("widersprueche") or [])
            summary = (f"Bewertung {rating if rating is not None else '–'}/10, {n_err} Fehler, "
                       f"{n_con} {'Widerspruch' if n_con == 1 else 'Widersprüche'}")
            step = Step("kritiker", agents.CRITIC.id, summary, detail=outcome.raw, duration=outcome.duration,
                        tokens=0 if outcome.value is None else outcome.tokens)
            run.steps.append(step)
            self._emit(run.progress, step)
        else:
            critique = {"bewertung": None, "fehler": [], "widersprueche": [], "fehlt": [], "sicher": False,
                        "roh": outcome.raw[:2000]}
        critique["revidiert"] = []
        if valid and int(cfg.max_revision_rounds or 0) > 0:
            self._check_cancel(run.cancel)
            revised = self._revision_stage(run, experts, answers, critique)
            critique["revidiert"] = sorted(revised)
        return critique

    def _revision_stage(self, run: _Run, experts: list[Expert], answers: dict[str, str], critique: dict) -> list[str]:
        """Gezielte Revision: Experten mit ``schwere == "hoch"`` oder – bei ``bewertung <= 4`` – alle
        genannten, nach Anzahl der Befunde sortiert, höchstens ``cfg.max_revised_experts``."""
        cfg = self.cfg
        by_expert: dict[str, list[dict]] = {}
        for item in critique.get("fehler") or []:
            if not isinstance(item, dict):
                continue
            eid = item.get("experte")
            if isinstance(eid, str) and eid in answers:
                by_expert.setdefault(eid, []).append(item)
        rating = critique.get("bewertung")
        if isinstance(rating, (int, float)) and rating <= 4:
            candidates = list(by_expert)
        else:
            candidates = [eid for eid, items in by_expert.items()
                          if any(str(it.get("schwere") or "").lower() == "hoch" for it in items)]
        order = {ex.id: i for i, ex in enumerate(experts)}
        candidates.sort(key=lambda e: (-len(by_expert[e]), order.get(e, 99)))
        candidates = candidates[:max(0, int(cfg.max_revised_experts or 0))]
        if not candidates:
            return []
        by_id = {ex.id: ex for ex in experts}
        jobs = [(by_id[eid], agents.revision_messages(by_id[eid], run.question, answers[eid], by_expert[eid]))
                for eid in candidates if eid in by_id]
        results, _ = self._parallel(run, jobs, stage="revision", max_tokens=cfg.max_tokens_expert)
        for eid, text in results.items():
            answers[eid] = text          # gescheiterte Revision behält das Original
        return list(results)

    def _tool_loop(self, messages: list[dict], *, stage: str, max_tokens: int, stream: StreamCallback | None,
                   steps: list[Step], tools_used: list[str], tools_block: str = "",
                   progress: ProgressCallback | None = None, cancel: threading.Event | None = None) -> str:
        """Modellaufruf mit Werkzeugschleife (Stufen ``schnell`` und ``synthese``).

        Invariante: der gestreamte Text entspricht dem Rückgabetext (bis auf Werkzeugblöcke in
        Zaun-Form, die das Gate nicht erkennt)."""
        cfg = self.cfg
        who = agents.OMEGA.id
        max_rounds = max(0, int(cfg.max_tool_rounds or 0))
        max_calls = max(0, int(cfg.max_tool_calls_per_answer or 0))
        joiner = _AnswerStream(stream)
        parts: list[str] = []
        for runde in range(max_rounds + 1):
            self._check_cancel(cancel)
            label = f"{who} denkt…" if runde == 0 else f"{who} wertet Werkzeugergebnisse aus…"
            self._emit(progress, Step(stage, who, label, status="start"))
            joiner.begin_part()
            gate = ToolStreamFilter(joiner.feed)

            def on_piece(piece: str, _gate: ToolStreamFilter = gate) -> None:
                self._check_cancel(cancel)
                _gate.feed(piece)

            t0 = time.time()
            try:
                res = self._chat(messages, max_tokens=max_tokens, model=cfg.model, temperature=cfg.temperature,
                                 stream=on_piece)
            except LLMError as e:
                step = Step(stage, who, f"Fehler – {e}", duration=time.time() - t0, status="fehler")
                steps.append(step)
                self._emit(progress, step)
                raise
            gate.finish()
            raw = res.text or ""
            calls = parse_tool_calls(raw) if tools_block else []
            if gate.tool_detected:
                # Gestreamt wurde nur der Text vor dem Block; Prosa nach einem gültigen Block hängen wir an
                # (und streamen sie nach), Protokollmüll ohne gültigen Aufruf fällt weg.
                head = gate.emitted
                tail = strip_tool_calls(raw[len(head):]).strip() if calls else ""
                if tail:
                    sep = "\n\n" if head.strip() else ""
                    joiner.feed(sep + tail)
                    parts.append((head.rstrip() + sep + tail).strip())
                else:
                    parts.append(head.strip())
            elif calls:
                parts.append(strip_tool_calls(raw).strip())      # Zaun-Form: Ausnahme von der Invariante
            else:
                parts.append(raw.strip())
            joiner.end_part()
            summary = f"{len(raw)} Zeichen"
            if calls:
                summary += f", {len(calls)} {'Werkzeugaufruf' if len(calls) == 1 else 'Werkzeugaufrufe'}"
            step = Step(stage, who, summary, detail=raw, duration=time.time() - t0, tokens=res.total_tokens)
            steps.append(step)
            self._emit(progress, step)
            if not calls:
                break
            if runde == max_rounds:
                parts.append(TOOL_LIMIT_HINT)
                joiner.begin_part()
                joiner.feed(TOOL_LIMIT_HINT)
                joiner.end_part()
                break
            messages.append({"role": "assistant", "content": raw})
            seen: set[tuple[str, str]] = set()
            unique: list[dict] = []
            for call in calls:
                args = call.get("args") or {}
                key = (str(call.get("name")), json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))
                if key in seen:
                    continue
                seen.add(key)
                unique.append(call)
            for call in unique[:max_calls]:
                self._check_cancel(cancel)
                name = str(call.get("name") or "")
                args = call.get("args") or {}
                t1 = time.time()
                result = self.tools.run(name, args)
                if result.ok:
                    tstep = Step("werkzeug", name, f"{name}: ok", detail=result.output, duration=time.time() - t1)
                else:
                    tstep = Step("werkzeug", name, f"{name}: Fehler – {result.error or 'unbekannter Fehler'}",
                                 detail=result.error or result.output or "", duration=time.time() - t1,
                                 status="fehler")
                steps.append(tstep)
                self._emit(progress, tstep)
                tools_used.append(name)
                messages.append(agents.tool_result_message(name, result))
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _memories_parser(text: str) -> list[dict] | None:
        """``None`` nur bei Nicht-JSON (löst den Retry aus), sonst die gültigen Einträge."""
        if parse_json(text) is None:
            return None
        return agents.parse_memories(text)

    def _learn_stage(self, run: _Run, text: str) -> None:
        """Automatisches Merken mit Schutzfiltern – Fehler hier sind nie fatal."""
        cfg = self.cfg
        t0 = time.time()
        if run.cancel is not None and run.cancel.is_set():
            step = Step("lernen", "system", "übersprungen: abgebrochen", duration=0.0)
            run.steps.append(step)
            self._emit(run.progress, step)
            return
        try:
            outcome = self._json_call(
                "extraktion", agents.memory_extraction_messages(run.question, text, run.project), MEMORY_SCHEMA,
                self._memories_parser, model=cfg.routing_model, max_tokens=cfg.max_tokens_json,
                steps=run.steps, progress=run.progress, cancel=run.cancel,
            )
            items = outcome.value if isinstance(outcome.value, list) else []
            qa_words = {w for w in normalize(run.question + " " + text).split() if len(w) > 4}
            skipped: dict[str, int] = {}
            details: list[str] = []
            limit = max(0, int(cfg.max_new_memories or 0))

            def skip(reason: str, content: str) -> None:
                skipped[reason] = skipped.get(reason, 0) + 1
                details.append(f"übersprungen ({reason}): {content[:120]}")

            for it in items:
                content = " ".join(str(it.get("inhalt") or "").split())
                if len(run.new_memories) >= limit:
                    skip("Limit", content)
                    continue
                try:
                    importance = float(it.get("wichtigkeit") or 0.0)
                except (TypeError, ValueError):
                    importance = 0.0
                if importance < MIN_MEMORY_IMPORTANCE:
                    skip("Wichtigkeit", content)
                    continue
                if not (MEMORY_CONTENT_MIN_CHARS <= len(content) <= MEMORY_CONTENT_MAX_CHARS):
                    skip("Länge", content)
                    continue
                low = content.lower()
                if any(phrase in low for phrase in MEMORY_PHRASE_BLOCKLIST):
                    skip("Floskel", content)
                    continue
                if "unsicher" in low:
                    skip("Unsicher", content)
                    continue
                words = {w for w in normalize(content).split() if len(w) > 4}
                if not (words & qa_words):
                    skip("kein Bezug", content)
                    continue
                tags = [str(t) for t in (it.get("tags") or []) if str(t).strip()][:MAX_MEMORY_TAGS]
                m = self.memory.remember(content, kind=str(it.get("art") or "notiz"), tags=tags,
                                         project=run.project, source="ki",
                                         importance=min(importance, MAX_AUTO_MEMORY_IMPORTANCE))
                if m.created_at < run.t0:
                    skip("bereits bekannt", content)
                    continue
                if any(x.id == m.id for x in run.new_memories):
                    skip("Duplikat", content)
                    continue
                run.new_memories.append(m)
                details.append(f"gemerkt: {m.short()}")
                if run.project and m.kind == "entscheidung":
                    try:
                        self.projects.ensure(run.project)
                        self.projects.add_note(run.project, "entscheidung", content[:80], content)
                        details.append("Projektnotiz: Entscheidung festgehalten")
                    except Exception as e:  # noqa: BLE001
                        details.append(f"Projektnotiz fehlgeschlagen: {e}")
            n = len(run.new_memories)
            n_skipped = sum(skipped.values())
            if n:
                summary = f"{n} {'Erinnerung' if n == 1 else 'Erinnerungen'} gemerkt"
                if n_skipped:
                    summary += f" ({n_skipped} übersprungen)"
            elif skipped:
                summary = "übersprungen: " + ", ".join(f"{k} {v}" for k, v in skipped.items())
            elif outcome.value is None:
                summary = "übersprungen: keine verwertbare Extraktion"
            else:
                summary = "übersprungen: nichts Merkwürdiges"
            step = Step("lernen", "system", summary, detail="\n".join(details), duration=time.time() - t0,
                        tokens=0 if outcome.value is None else outcome.tokens)
        except Exception as e:  # noqa: BLE001 – Lernen darf die Antwort nie kosten
            step = Step("lernen", "system", f"übersprungen: Fehler – {e}", duration=time.time() - t0,
                        status="fehler")
        run.steps.append(step)
        self._emit(run.progress, step)

    def _error_answer(self, run: _Run, error: LLMError) -> Answer:
        cfg = self.cfg
        if isinstance(error, BackendUnavailable):
            url = getattr(self.backend, "base_url", None) or cfg.base_url or "Standard"
            text = (f"Modell-Server nicht erreichbar – starte Ollama (`ollama serve`) oder prüfe base_url "
                    f"({url}).")
        elif isinstance(error, ModelNotFound):
            name = getattr(error, "model_name", None) or cfg.model
            text = f"Modell »{name}« ist nicht installiert – `ollama pull {name}`."
        else:
            text = f"Modellfehler: {error}"
        step = Step("system", "system", text, status="fehler")
        run.steps.append(step)
        self._emit(run.progress, step)
        return Answer(
            text=text, question=run.question, session_id=run.session_id, project=run.project, depth="fehler",
            experts=list(run.experts), critique=run.critique, memories_used=list(run.memories),
            new_memories=list(run.new_memories), lessons_used=list(run.lessons), tools_used=list(run.tools_used),
            steps=run.steps, interaction_id=None, tokens=sum(int(s.tokens) for s in run.steps),
            duration=time.time() - run.t0,
        )

    # ------------------------------------------------------------ Feedback
    def feedback(self, interaction_id: int, rating: int | None = None, comment: str | None = None,
                 correction: str | None = None, correction_full: str | None = None) -> Feedback:
        """Verarbeitet Nutzer-Feedback zu einer Interaktion.

        (a) kurze Korrektur → vollständige Antwort via :meth:`rewrite_correction`, (b) Bewertung/
        Korrektur speichern, (c) Lektion nur bei (``rating == -1`` und Kommentar) oder Korrektur,
        (d) Gedächtnis-Hygiene. ``ValueError`` bei unbekannter ID oder ungültiger Bewertung."""
        interaction_id = int(interaction_id)
        inter = self.learning.get(interaction_id)
        if inter is None:
            raise ValueError(f"Unbekannte Interaktion #{interaction_id}.")
        if rating is not None:
            try:
                rating = int(rating)
            except (TypeError, ValueError):
                raise ValueError(f"Ungültige Bewertung {rating!r} – erlaubt sind -1, 0, 1.") from None
            if rating not in (-1, 0, 1):
                raise ValueError(f"Ungültige Bewertung {rating!r} – erlaubt sind -1, 0, 1.")
        comment = (str(comment).strip() or None) if comment is not None else None
        correction = (str(correction).strip() or None) if correction is not None else None
        correction_full = (str(correction_full).strip() or None) if correction_full is not None else None

        # (a) kurze Korrektur zur vollständigen Antwort umschreiben
        if correction and correction_full is None and len(correction) < SHORT_CORRECTION_CHARS:
            correction_full = self.rewrite_correction(interaction_id, correction)

        # (b) speichern
        old_rating = int(inter.rating or 0)
        if rating is not None:
            inter = self.learning.rate(interaction_id, rating, comment)
        elif comment:
            inter = self.learning.rate(interaction_id, inter.rating, comment)
        if correction:
            inter = self.learning.correct(interaction_id, correction, correction_full)

        # (c) Lektion
        lessons: list[Lesson] = []
        if (rating == -1 and comment) or correction:
            lesson = self._lesson_stage(inter, comment, correction)
            if lesson is not None:
                lessons.append(lesson)

        # (d) Gedächtnis-Hygiene – nur beim Wechsel der Bewertung, alte Wirkung wird zurückgenommen
        adjusted = 0
        deltas = {1: 0.10, -1: -0.15, 0: 0.0}
        if rating is not None and rating != old_rating:
            change = deltas[rating] - deltas.get(old_rating, 0.0)
            if change:
                for mid in inter.memories_used:
                    if self.memory.get(mid) is not None:
                        self.memory.update_importance(mid, change)
                        adjusted += 1
            if rating == -1:
                for mid in inter.new_memories:
                    m = self.memory.get(mid)
                    if m is not None and m.source == "ki" and self.memory.forget(mid):
                        adjusted += 1
        refreshed = self.learning.get(interaction_id)
        return Feedback(interaction=refreshed or inter, lessons=lessons, memories_adjusted=adjusted,
                        correction_full=correction_full)

    def _lesson_stage(self, inter: Interaction, comment: str | None, correction: str | None) -> Lesson | None:
        cfg = self.cfg
        rule = ""
        topics: list[str] = []
        general = False
        try:
            outcome = self._json_call(
                "lektion", agents.lesson_extraction_messages(inter.question, inter.answer, comment, correction),
                LESSON_SCHEMA, agents.parse_lesson, model=cfg.routing_model, max_tokens=cfg.max_tokens_json,
            )
            if isinstance(outcome.value, dict):
                rule = str(outcome.value.get("regel") or "").strip()
                topics = [str(t) for t in outcome.value.get("gilt_fuer") or []]
                general = bool(outcome.value.get("allgemein"))
        except LLMError:
            rule = ""
        if not rule:
            rule = comment or ("Besser: " + agents.clip(correction or "", 300))
            topics = keywords(inter.question, limit=5)
            general = False
        rule = " ".join(rule.split())
        if not rule:
            return None
        return self.learning.add_lesson(rule, scope="allgemein" if general else "thema", topics=topics,
                                        source_interaction=inter.id)

    def rewrite_correction(self, interaction_id: int, correction: str) -> str:
        """Stufe ``korrektur``: schreibt die ursprüngliche Antwort mit eingearbeiteter Korrektur neu.
        Bei Modellfehlern (oder leerer Antwort) wird die Korrektur selbst zurückgegeben."""
        correction = (correction or "").strip()
        inter = self.learning.get(int(interaction_id))
        if inter is None:
            raise ValueError(f"Unbekannte Interaktion #{interaction_id}.")
        if not correction:
            return correction
        try:
            res = self._chat(agents.correction_rewrite_messages(inter.question, inter.answer, correction),
                             max_tokens=self.cfg.max_tokens_synthesis, model=self.cfg.model, temperature=0.2)
        except LLMError:
            return correction
        text = strip_tool_calls(res.text or "").strip()
        return text or correction

    # ------------------------------------------------------------ Gedächtnis
    def remember(self, content: str, kind: str = "notiz", tags: Sequence[str] = (), project: str | None = None,
                 importance: float = 0.6) -> Memory:
        """Speichert eine Erinnerung des Nutzers (``source="nutzer"``)."""
        return self.memory.remember(content, kind=kind, tags=tags, project=project, source="nutzer",
                                    importance=importance)

    def forget(self, memory_id: int) -> bool:
        return self.memory.forget(int(memory_id))

    def recall(self, query: str, k: int | None = None, project: str | None = None) -> list[Memory]:
        """Sucht Erinnerungen (``min_importance=0.15``)."""
        return self.memory.search(query, k or self.cfg.memory_recall, project, min_importance=MIN_RECALL_IMPORTANCE)

    def reindex_memories(self, progress: Callable[[int, int], None] | None = None) -> int:
        return self.memory.reindex(progress=progress)

    # ------------------------------------------------------------ Modelle
    def set_model(self, name: str, *, fast: bool = False) -> None:
        """Wechselt das Haupt- (oder mit ``fast=True`` das Routing-)Modell. Ist das Backend erreichbar
        und das Modell nicht installiert, folgt :class:`ModelNotFound`. ``fast=True`` mit leerem Namen
        schaltet das separate Routing-Modell ab."""
        name = (name or "").strip()
        if not name and not fast:
            raise ValueError("Leerer Modellname.")
        if name:
            available = False
            try:
                available = bool(self.backend.available())
            except Exception:  # noqa: BLE001
                available = False
            if available:
                try:
                    installed = self.backend.has_model(name)
                except LLMError:
                    installed = True        # Liste nicht abrufbar → nicht blockieren
                if not installed:
                    raise ModelNotFound(f"Modell »{name}« ist nicht installiert – `ollama pull {name}`.")
        if fast:
            self.cfg.fast_model = name
        else:
            self.cfg.model = name
            self.backend.default_model = name

    def models(self) -> list[ModelInfo]:
        """Installierte Modelle; leer (ohne Fehler), wenn das Backend nicht erreichbar ist."""
        try:
            return list(self.backend.list_models())
        except Exception:  # noqa: BLE001 – Server weg, Timeout, kaputte Antwort
            return []

    # ------------------------------------------------------------ Lernen
    def export_dataset(self, path: str, **kw: Any) -> dict:
        return self.learning.export_dataset(path, **kw)

    def set_dangerous_policy(self, allow: bool) -> None:
        """Erlaubt (``True``) oder verbietet gefährliche Werkzeuge ohne Rückfrage – für Server und
        Automationen. Werkzeuge im Chat fragen weiterhin über ``confirm`` nach, falls gesetzt."""
        self.automation.allow_dangerous = bool(allow)
        if allow:
            self.tools.set_policy(lambda _name, _args: True, True)
        else:
            self.tools.set_policy(self._confirm, True)

    # ------------------------------------------------------------ Pflege
    def consolidate(self, days: int = 7, project: str | None = None) -> dict:
        """Gedächtnis-Pflege: fasst alte, unsichere KI-Erinnerungen (``source == "ki"``,
        Wichtigkeit < 0,3, älter als ``days`` Tage) je Projekt zu einer Zusammenfassung zusammen und
        verdichtet lange Sitzungen (> 30 Nachrichten) zu je einer ``zusammenfassung``-Erinnerung.
        Nutzer-Erinnerungen werden nie angefasst."""
        cfg = self.cfg
        cutoff = time.time() - max(0, int(days)) * 86400
        result = {"zusammengefasst": 0, "geloescht": 0, "sitzungen": 0, "fehler": []}
        candidates = self.memory.list(source="ki", max_importance=0.3, older_than=cutoff)
        if project is not None:
            candidates = [m for m in candidates if m.project == project]
        groups: dict[str | None, list[Memory]] = {}
        for m in candidates:
            if m.kind == "zusammenfassung":
                continue
            groups.setdefault(m.project, []).append(m)
        for proj, mems in groups.items():
            if len(mems) < 3:
                continue
            batch = mems[:40]
            known = [m.id for m in batch]
            try:
                outcome = self._json_call(
                    "konsolidierung", agents.consolidation_messages(batch, proj), CONSOLIDATION_SCHEMA,
                    lambda t, known=known: agents.parse_consolidation(t, known), model=cfg.routing_model,
                    max_tokens=max(cfg.max_tokens_json, 600),
                )
            except LLMError as e:
                result["fehler"].append(f"{proj or 'allgemein'}: {e}")
                continue
            value = outcome.value if isinstance(outcome.value, dict) else None
            if not value:
                result["fehler"].append(f"{proj or 'allgemein'}: keine verwertbare Zusammenfassung")
                continue
            keep = set(value.get("behalten") or [])
            self.memory.remember(value["zusammenfassung"], kind="zusammenfassung", project=proj, source="ki",
                                 importance=0.5, tags=["konsolidierung"])
            for m in batch:
                if m.id not in keep and self.memory.forget(m.id):
                    result["geloescht"] += 1
            result["zusammengefasst"] += 1
        # Lange Sitzungen verdichten
        for sess in self.memory.sessions():
            n = int(sess["nachrichten"])
            if n <= 30:
                continue
            if project is not None and sess.get("project") != project:
                continue
            key = f"session_summary:{sess['session_id']}"
            try:
                done_at = int(self.memory.meta(key) or 0)
            except ValueError:
                done_at = 0
            if n - done_at <= 30:
                continue
            messages = self.memory.history(sess["session_id"], 60)
            try:
                res = self._chat(agents.session_summary_messages(messages), max_tokens=cfg.max_tokens_json,
                                 model=cfg.routing_model, temperature=0.2)
            except LLMError as e:
                result["fehler"].append(f"Sitzung {sess['session_id']}: {e}")
                continue
            text = " ".join((res.text or "").split())
            if len(text) < 20:
                result["fehler"].append(f"Sitzung {sess['session_id']}: leere Zusammenfassung")
                continue
            self.memory.remember(text[:800], kind="zusammenfassung", project=sess.get("project"), source="ki",
                                 importance=0.5, tags=["sitzung", str(sess["session_id"])])
            self.memory.set_meta(key, str(n))
            result["sitzungen"] += 1
        return result

    def backup(self, keep: int = 7) -> Path:
        """Sichert alle Datenbanken (SQLite-Online-Backup), das Gedächtnis als JSON und die
        Konfiguration nach ``backups/<JJJJMMTT-HHMMSS>/``; behält die ``keep`` neuesten Sicherungen."""
        cfg = self.cfg
        cfg.ensure_dirs()
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = cfg.backups_dir / stamp
        n = 2
        while target.exists():
            target = cfg.backups_dir / f"{stamp}-{n}"
            n += 1
        target.mkdir(parents=True)
        files: list[str] = []
        stores = {
            "gedaechtnis.db": self.memory, "lernen.db": self.learning, "wissen.db": self.knowledge,
            "projekte.db": self.projects, "missionen.db": self.missions.store, "automationen.db": self.automation.store,
        }
        for name, store in stores.items():
            conn = getattr(store, "_db", None)
            lock = getattr(store, "_lock", None)
            if conn is None:
                continue
            dest = sqlite3.connect(str(target / name))
            try:
                if lock is not None:
                    with lock:
                        conn.backup(dest)
                else:
                    conn.backup(dest)
            finally:
                dest.close()
            files.append(name)
        self.memory.export(target / "gedaechtnis.json")
        files.append("gedaechtnis.json")
        src_cfg = find_config_file()
        if src_cfg is not None:
            shutil.copy2(src_cfg, target / "config.quelle.json")
            files.append("config.quelle.json")
        save_config(cfg, target / "config.json")
        files.append("config.json")
        manifest = {
            "zeit": _iso(time.time()), "version": "4.0.0", "dateien": files,
            "gedaechtnis": self.memory.stats(), "lernen": self.learning.stats(),
            "wissen": self.knowledge.stats(), "projekte": self.projects.stats(),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        # Alte Sicherungen entfernen
        keep = max(1, int(keep))
        folders = sorted(p for p in cfg.backups_dir.iterdir() if p.is_dir() and (p / "manifest.json").exists())
        for old in folders[:-keep]:
            shutil.rmtree(old, ignore_errors=True)
        return target

    # ------------------------------------------------------------ Status
    def status(self) -> dict:
        cfg = self.cfg
        try:
            available = bool(self.backend.available())
        except Exception:  # noqa: BLE001
            available = False
        mem_stats = self.memory.stats()
        return {
            "backend": getattr(self.backend, "name", "?"),
            "verfuegbar": available,
            "modell": cfg.model,
            "routing_modell": cfg.routing_model,
            "embedding_aktiv": self._embedder is not None,
            "modelle": [m.name for m in self.models()],
            "gedaechtnis": mem_stats,
            "lernen": self.learning.stats(),
            "werkzeuge": [t.name for t in self.tools.list()],
            "beschaeftigt": self.busy,
            "tiefe": cfg.depth,
            "datenverzeichnis": str(cfg.data_path),
            "vektoren": {
                "mit": mem_stats.get("mit_vektor", 0),
                "ohne": mem_stats.get("ohne_vektor", 0),
                "modell": mem_stats.get("embedding_modell", ""),
                "dim": mem_stats.get("embedding_dim", 0),
            },
            "wissen": self.knowledge.stats(),
            "projekte": self.projects.stats(),
            "missionen": {"laufend": self.missions.running(), "anzahl": len(self.missions.store.list())},
            "automationen": {**self.automation.store.stats(), "aktiv": bool(self.automation.running)},
            "geraete": self.devices.stats(),
            "modelle3d": self.models3d.stats(),
            "geo": self.geo.stats(),
            "online": bool(cfg.online),
        }

    def close(self) -> None:
        """Schließt alle Speicher und stoppt den Zeitplaner (idempotent)."""
        if self._closed:
            return
        self._closed = True
        errors: list[Exception] = []

        def _call(obj: Any, method: str) -> None:
            fn = getattr(obj, method, None)
            if callable(fn):
                fn()

        for action in (
            lambda: _call(self.automation, "stop"),
            lambda: _call(getattr(self.automation, "store", None), "close"),
            lambda: _call(getattr(self.missions, "store", None), "close"),
            lambda: _call(self.projects, "close"),
            lambda: _call(self.knowledge, "close"),
            lambda: _call(getattr(self, "devices", None), "close"),
            lambda: _call(getattr(self, "models3d", None), "close"),
            lambda: _call(getattr(self, "geo", None), "close"),
            lambda: self.memory.close(),
            lambda: self.learning.close(),
        ):
            try:
                action()
            except Exception as e:  # noqa: BLE001 – alle Speicher trotzdem schließen
                errors.append(e)
        if errors:
            raise errors[0]


__all__ = [
    "Answer", "Brain", "Feedback", "ProgressCallback", "Step", "memory_to_dict",
    "DEPTHS", "COMPLEXITY_TO_DEPTH", "TOOL_LIMIT_HINT",
]
