"""Kommandozeile von OBITO.

``python -m obito [unterbefehl]`` – ohne Unterbefehl startet der Chat.

Unterbefehle: ``chat`` (REPL mit Streaming), ``doctor`` (Systemprüfung), ``serve`` (HUD/API),
``export-dataset``, ``eval``, ``modelfile``, ``train``, ``memory``, ``config``.

Die REPL-Logik steckt in :class:`ChatSession` (``handle_line``), damit sie ohne Terminal
testbar ist. Module, die parallel entstehen (``obito.training``, ``obito.server``), werden erst
innerhalb der jeweiligen Unterbefehle importiert – ``--help``, ``chat``, ``doctor``, ``memory``,
``config`` und ``export-dataset`` funktionieren auch ohne sie.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from . import cli_extra
from .brain import Answer, Brain, Step
from .config import Config, find_config_file, load_config, save_config
from .learning import LearningStore
from .llm import BackendUnavailable, LLMBackend, LLMError, ModelNotFound, make_backend
from .memory import KINDS, MemoryStore, normalize

DEPTH_CHOICES = ("auto", "schnell", "mittel", "tief")
EVAL_DEPTHS = ("schnell", "mittel", "tief")
DATASET_FORMATS = ("chat", "alpaca", "dpo")
GLOBAL_VALUE_OPTIONS = ("--config", "--backend", "--daten", "--modell")

# Symbole der REPL (werden bei nicht-UTF-8-Ausgaben ersetzt, siehe ``_write``)
SYM_START = "⟳"
SYM_OK = "✓"
SYM_FAIL = "✗"
SYM_MEMORY = "💾"
SYM_STOP = "⏹"
SYM_WARN = "⚠"
SYM_LESSON = "📚"
SYM_GOOD = "👍"
SYM_BAD = "👎"

ABORT_TEXT = "Modellfehler: Abgebrochen"
# Anfänge der Fehler-Antworten des Denkkerns (Brain._error_answer) – erscheinen auch als System-Step
ERROR_ANSWER_PREFIXES = ("Modellfehler:", "Modell-Server nicht erreichbar", "Modell »")

# Grobe Ausgabe-Token je Stufe für die Zeitprognose in ``doctor`` (sequenzielle Aufrufe)
EXPERT_TOKENS = 450
CRITIC_TOKENS = 300
FAST_TOKENS = 400
SYNTHESIS_TOKENS = 650

# Pakete für ``doctor --training``
TRAINING_PACKAGES = ("torch", "transformers", "peft", "trl", "bitsandbytes", "datasets")


# ------------------------------------------------------------------ Ausgabe
def _write(out: TextIO, text: str) -> None:
    """Schreibt Text; Zeichen, die die Ausgabe nicht kodieren kann (Windows-Konsole ohne UTF-8),
    werden ersetzt statt einen Fehler auszulösen."""
    try:
        out.write(text)
    except UnicodeEncodeError:
        enc = getattr(out, "encoding", None) or "ascii"
        out.write(text.encode(enc, "replace").decode(enc, "replace"))
    try:
        out.flush()
    except Exception:  # noqa: BLE001 – geschlossene Pipes dürfen die REPL nicht stoppen
        pass


def _println(out: TextIO, text: str = "") -> None:
    _write(out, text + "\n")


def _fmt_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min".replace(".", ",")
    return f"{seconds / 3600:.1f} h".replace(".", ",")


def _fmt_gb(size: int | float | None) -> str:
    if not size:
        return "?"
    return f"{float(size) / 1e9:.1f} GB".replace(".", ",")


def _yes(answer: str | None, default: bool) -> bool:
    """Deutsche Ja/Nein-Auswertung einer Eingabe (``j``/``ja``/``y`` = ja, ``n``/``nein`` = nein)."""
    if answer is None:
        return default
    a = answer.strip().lower()
    if not a:
        return default
    if a in ("j", "ja", "y", "yes"):
        return True
    if a in ("n", "nein", "no"):
        return False
    return default


def format_status(status: dict) -> str:
    """Lesbare Darstellung von ``Brain.status()`` für ``/status``."""
    mem = status.get("gedaechtnis") or {}
    learn = status.get("lernen") or {}
    vec = status.get("vektoren") or {}
    lines = [
        f"Backend: {status.get('backend')} ({'erreichbar' if status.get('verfuegbar') else 'nicht erreichbar'})"
        f" · Modell: {status.get('modell')} · Routing: {status.get('routing_modell')}"
        f" · Embedding: {'aktiv' if status.get('embedding_aktiv') else 'aus'}"
        + (f" ({vec.get('modell')}, {vec.get('dim')} Dim.)" if status.get("embedding_aktiv") and vec.get("modell") else ""),
        f"Tiefe: {status.get('tiefe')} · {'beschäftigt' if status.get('beschaeftigt') else 'bereit'}"
        f" · Werkzeuge: {', '.join(status.get('werkzeuge') or []) or 'keine'}",
        f"Gedächtnis: {mem.get('erinnerungen', 0)} Erinnerungen ({vec.get('mit', mem.get('mit_vektor', 0))} mit Vektor,"
        f" {vec.get('ohne', mem.get('ohne_vektor', 0))} ohne), {mem.get('nachrichten', 0)} Verlaufsnachrichten"
        + (f", Projekte: {', '.join(mem['projekte'])}" if mem.get("projekte") else ""),
        f"Lernen: {learn.get('interaktionen', 0)} Interaktionen, {learn.get('bewertet', 0)} bewertet"
        f" ({learn.get('positiv', 0)} positiv, {learn.get('negativ', 0)} negativ), {learn.get('korrigiert', 0)} korrigiert,"
        f" {learn.get('trainierbar', 0)} trainierbar, {learn.get('lektionen', 0)} Lektionen"
        f" ({learn.get('lektionen_aktiv', 0)} aktiv)",
        f"Datenverzeichnis: {status.get('datenverzeichnis')}",
    ]
    models = status.get("modelle") or []
    lines.append("Installierte Modelle: " + (", ".join(models) if models else "keine abrufbar"))
    return "\n".join(lines)


def step_line(step: Step) -> str:
    """Eine Fortschrittszeile je Step: ``⟳ ALPHA denkt…``, ``✓ KRITIKER: Bewertung 8/10 …``,
    ``✗ routing: Fehler – …``."""
    who = step.who if step.who != "system" else step.stage
    summary = (step.summary or "").strip()
    if step.status == "start":
        if summary.endswith("denkt…"):
            return f"{SYM_START} {step.who} denkt…"
        return f"{SYM_START} {summary}"
    mark = SYM_FAIL if step.status == "fehler" else SYM_OK
    if summary.startswith(step.who + " ") or summary.startswith(step.who + ":"):
        return f"{mark} {summary}"
    return f"{mark} {who}: {summary}"


# ------------------------------------------------------------------ REPL
class ChatSession:
    """Die Chat-REPL ohne Terminalbindung: ``out`` nimmt die Ausgabe, ``inp`` liefert Eingaben.

    ``handle_line(line) -> bool`` verarbeitet eine Zeile (Frage oder ``/befehl``) und liefert
    ``False``, wenn die Sitzung enden soll. Fragen laufen in einem Arbeits-Thread, damit Strg+C
    im Hauptthread ``cancel`` setzen kann (ein zweites Strg+C beendet die Sitzung). Gefährliche
    Werkzeuge werden über ``inp`` mit ``[j/N]`` bestätigt, sofern ``cfg.confirm_dangerous``.
    """

    PROMPT = "Du> "
    ANSWER_PREFIX = "OBITO> "

    # (Namen, Methode, Aufruf, Beschreibung)
    COMMANDS: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
        (("hilfe", "help", "?"), "cmd_help", "/hilfe", "diese Übersicht"),
        (("merk",), "cmd_remember", "/merk [art:] <text>", "Erinnerung speichern (art: fakt, praeferenz, entscheidung, loesung, fehler, notiz)"),
        (("vergiss",), "cmd_forget", "/vergiss <id>", "Erinnerung löschen"),
        (("suche",), "cmd_search", "/suche <frage>", "Erinnerungen suchen"),
        (("erinnerungen",), "cmd_memories", "/erinnerungen [n]", "die jüngsten Erinnerungen zeigen"),
        (("projekt",), "cmd_project", "/projekt [name|-]", "Projekt zeigen, setzen oder mit - löschen"),
        (("tief",), "cmd_depth_deep", "/tief", "Tiefe: Gremium + Kritiker + Revision"),
        (("mittel",), "cmd_depth_medium", "/mittel", "Tiefe: 3 Experten + Synthese"),
        (("schnell",), "cmd_depth_fast", "/schnell", "Tiefe: ein Aufruf mit Kontext"),
        (("auto",), "cmd_depth_auto", "/auto", "Tiefe automatisch wählen"),
        (("modelle",), "cmd_models", "/modelle", "installierte Modelle zeigen"),
        (("modell",), "cmd_model", "/modell [schnell] <name>", "Haupt- (oder Routing-)Modell wechseln"),
        (("gut",), "cmd_good", "/gut [kommentar]", "letzte Antwort positiv bewerten"),
        (("schlecht",), "cmd_bad", "/schlecht [kommentar]", "letzte Antwort negativ bewerten (Kommentar wird zur Lektion)"),
        (("korrektur",), "cmd_correction", "/korrektur <bessere antwort>", "letzte Antwort korrigieren (mit Umschreib-Vorschlag)"),
        (("lektionen",), "cmd_lessons", "/lektionen [n]", "gelernte Regeln zeigen"),
        (("lektion-loeschen", "lektion-löschen"), "cmd_delete_lesson", "/lektion-loeschen <id>", "Lektion löschen"),
        (("spur",), "cmd_trace", "/spur", "Denk-Spur der letzten Antwort"),
        (("werkzeuge",), "cmd_tools", "/werkzeuge", "verfügbare Werkzeuge"),
        (("status",), "cmd_status", "/status", "Systemstatus"),
        (("export",), "cmd_export", "/export [pfad]", "Trainingsdatensatz exportieren"),
        (("dokument",), "cmd_document", "/dokument <pfad>", "Datei oder Ordner ins Datenzentrum aufnehmen"),
        (("dokumente",), "cmd_documents", "/dokumente [frage]", "Dokumente zeigen oder durchsuchen"),
        (("aufgabe",), "cmd_task", "/aufgabe <text>", "Aufgabe im aktuellen Projekt notieren"),
        (("entscheidung",), "cmd_decision", "/entscheidung <text>", "Entscheidung im aktuellen Projekt festhalten"),
        (("projektinfo",), "cmd_project_info", "/projektinfo", "Zusammenfassung und Notizen des Projekts"),
        (("mission",), "cmd_mission", "/mission <ziel>", "Mission planen und ausführen"),
        (("missionen",), "cmd_missions", "/missionen", "Missionen zeigen"),
        (("automationen",), "cmd_automations", "/automationen", "Automationen zeigen"),
        (("material",), "cmd_material", "/material <name>", "Materialdaten (Richtwerte) zeigen"),
        (("rechner",), "cmd_calculator", "/rechner [name k=v …]", "Ingenieur-Rechner (ohne Argument: Liste)"),
        (("backup",), "cmd_backup", "/backup", "Sicherung aller Daten anlegen"),
        (("konsolidieren",), "cmd_consolidate", "/konsolidieren [tage]", "Gedächtnis-Pflege: alte KI-Erinnerungen verdichten"),
        (("beenden", "exit", "quit", "q"), "cmd_quit", "/beenden", "OBITO verlassen"),
    )

    def __init__(self, brain: Brain, cfg: Config, out: TextIO = sys.stdout, inp: Callable[[str], str] = input, *,
                 session_id: str = "standard", project: str | None = None, depth: str | None = None,
                 show_progress: bool = True):
        self.brain = brain
        self.cfg = cfg
        self.out = out
        self.inp = inp
        self.session_id = str(session_id or "standard")
        self.project: str | None = (project or "").strip() or None
        depth = (depth or "").strip().lower() or None
        if depth is not None and depth not in DEPTH_CHOICES:
            raise ValueError(f"Unbekannte Tiefe {depth!r} – erlaubt: {', '.join(DEPTH_CHOICES)}.")
        self.depth: str | None = depth            # None = cfg.depth
        self.show_progress = bool(show_progress)
        self.last_answer: Answer | None = None
        self.last_interaction_id: int | None = None
        self.cancel_event = threading.Event()
        self._waiting = threading.Event()         # gesetzt, solange auf eine Antwort gewartet wird
        self._stream_open = False
        self._abort = False
        self._dispatch: dict[str, Callable[[str], bool]] = {}
        for names, method, _usage, _help in self.COMMANDS:
            for n in names:
                self._dispatch[n] = getattr(self, method)
        if cfg.confirm_dangerous:
            brain.tools.set_policy(self.confirm, True)

    # ------------------------------------------------------------ Ausgabe
    def _print(self, text: str = "") -> None:
        _println(self.out, text)

    def _end_stream_line(self) -> None:
        if self._stream_open:
            _write(self.out, "\n")
            self._stream_open = False

    def _read(self, prompt: str) -> str | None:
        """Eine Zeile vom Nutzer; ``None`` bei Eingabeende oder Abbruch."""
        try:
            return self.inp(prompt)
        except (EOFError, KeyboardInterrupt):
            return None

    @property
    def effective_depth(self) -> str:
        return self.depth or self.cfg.depth or "auto"

    # ------------------------------------------------------------ Rückrufe
    def on_step(self, step: Step) -> None:
        if not self.show_progress:
            return
        if step.status == "fehler":
            # Den abschließenden Fehler-Step des Denkkerns zeigt ``_show_answer`` als Antwort;
            # nach einem Abbruch durch den Nutzer sind die Folgefehler der Stufen nur Rauschen.
            if step.stage == "system" and step.summary.startswith(ERROR_ANSWER_PREFIXES):
                return
            if self.cancel_event.is_set():
                return
        self._end_stream_line()
        self._print(step_line(step))

    def on_token(self, piece: str) -> None:
        if not piece:
            return
        if not self._stream_open:
            _write(self.out, self.ANSWER_PREFIX)
            self._stream_open = True
        _write(self.out, piece)

    def confirm(self, name: str, args: dict) -> bool:
        """Rückfrage für gefährliche Werkzeuge: ``[j/N]`` – Standard ist Nein."""
        self._end_stream_line()
        try:
            shown = json.dumps(args, ensure_ascii=False)
        except (TypeError, ValueError):
            shown = str(args)
        if len(shown) > 400:
            shown = shown[:400] + " …"
        self._print(f"{SYM_WARN} Werkzeug »{name}« will ausführen: {shown}")
        return _yes(self._read("Erlauben? [j/N] "), default=False)

    # ------------------------------------------------------------ Fragen
    def ask(self, question: str) -> Answer | None:
        """Stellt eine Frage mit Streaming und Fortschritt. Strg+C setzt ``cancel``; ein zweites
        Strg+C bricht die Sitzung ab (``handle_line`` liefert dann ``False``)."""
        question = (question or "").strip()
        if not question:
            return None
        cancel = threading.Event()
        self.cancel_event = cancel
        self._stream_open = False
        box: dict[str, Any] = {}
        done = threading.Event()

        def work() -> None:
            try:
                box["answer"] = self.brain.ask(
                    question, session_id=self.session_id, project=self.project, depth=self.effective_depth,
                    stream=self.on_token, progress=self.on_step, cancel=cancel,
                )
            except BaseException as e:  # noqa: BLE001 – im Thread darf nichts unbemerkt verloren gehen
                box["error"] = e
            finally:
                done.set()

        # Bewusst kein ``Thread.join(timeout)``: wird es von Strg+C unterbrochen, markiert CPython den
        # noch laufenden Thread als beendet. Ein eigenes Event ist davon nicht betroffen.
        worker = threading.Thread(target=work, name="obito-chat-frage", daemon=True)
        interrupts = 0
        worker.start()
        try:
            while True:
                try:
                    self._waiting.set()          # ab hier fängt die Schleife Strg+C ab
                    if done.wait(0.1):
                        break
                except KeyboardInterrupt:
                    interrupts += 1
                    if interrupts == 1:
                        cancel.set()
                        self._end_stream_line()
                        self._print(f"{SYM_STOP} Abbruch angefordert – warte auf das Modell … "
                                    f"(noch einmal Strg+C beendet OBITO)")
                    else:
                        self._end_stream_line()
                        self._print("Beende OBITO.")
                        self._abort = True
                        return None
        finally:
            self._waiting.clear()

        if "error" in box:
            self._end_stream_line()
            err = box["error"]
            self._print(f"{SYM_FAIL} Unerwarteter Fehler: {err.__class__.__name__}: {err}")
            return None
        answer: Answer = box["answer"]
        self._show_answer(answer)
        return answer

    def _show_answer(self, answer: Answer) -> None:
        self._end_stream_line()
        self.last_answer = answer
        if answer.depth == "fehler":
            if answer.text == ABORT_TEXT or answer.text.endswith("Abgebrochen"):
                self._print(f"{SYM_STOP} Abgebrochen.")
            else:
                self._print(f"{SYM_FAIL} {answer.text}")
            return
        for m in answer.new_memories:
            tag = " (unsicher)" if m.source == "ki" else ""
            self._print(f"{SYM_MEMORY} gemerkt{tag}: {m.content}")
        if answer.interaction_id is not None:
            self.last_interaction_id = answer.interaction_id
        parts = [answer.depth]
        if answer.experts:
            parts.append(", ".join(answer.experts))
        if answer.tools_used:
            parts.append("Werkzeuge: " + ", ".join(dict.fromkeys(answer.tools_used)))
        parts.append(f"{answer.tokens} Tokens")
        parts.append(f"{answer.duration:.1f} s".replace(".", ","))
        if answer.interaction_id is not None:
            parts.append(f"Interaktion #{answer.interaction_id} (/gut, /schlecht, /korrektur)")
        self._print("— " + " · ".join(parts))

    # ------------------------------------------------------------ Zeilen
    def handle_line(self, line: str) -> bool:
        """Verarbeitet eine Eingabezeile. ``False`` = Sitzung beenden."""
        line = (line or "").strip()
        if not line:
            return True
        if line.startswith("/"):
            cmd, _, arg = line[1:].partition(" ")
            cmd = cmd.strip().lower()
            arg = arg.strip()
            handler = self._dispatch.get(cmd)
            if handler is None:
                self._print(f"Unbekannter Befehl /{cmd} – /hilfe zeigt alle Befehle.")
                return True
            return bool(handler(arg))
        self.ask(line)
        return not self._abort

    def run(self) -> int:
        """Interaktive Schleife bis ``/beenden``, Eingabeende oder Strg+C am Prompt."""
        self.banner()
        while True:
            try:
                line = self.inp(self.PROMPT)
            except EOFError:
                self._print()
                break
            except KeyboardInterrupt:
                self._print("\nBis bald.")
                break
            try:
                if not self.handle_line(line):
                    break
            except KeyboardInterrupt:
                self._end_stream_line()
                self._print("\nBeende OBITO.")
                break
        return 0

    def banner(self) -> None:
        cfg = self.cfg
        self._print(f"OBITO – lokaler KI-Assistent · Modell {cfg.model} · Tiefe {self.effective_depth}"
                    + (f" · Projekt {self.project}" if self.project else "")
                    + f" · Sitzung {self.session_id}")
        self._print("Frage eingeben oder /hilfe für Befehle. /beenden verlässt OBITO.")
        try:
            if not self.brain.backend.available():
                url = getattr(self.brain.backend, "base_url", None) or cfg.base_url or "Standard"
                self._print(f"{SYM_WARN} Modell-Server nicht erreichbar ({url}) – starte Ollama (`ollama serve`). "
                            f"Befehle wie /erinnerungen funktionieren trotzdem.")
        except Exception as e:  # noqa: BLE001
            self._print(f"{SYM_WARN} Backend-Prüfung fehlgeschlagen: {e}")

    # ------------------------------------------------------------ Befehle
    def cmd_help(self, arg: str) -> bool:
        self._print("Befehle:")
        width = max(len(usage) for _n, _m, usage, _h in self.COMMANDS)
        for _names, _method, usage, help_text in self.COMMANDS:
            self._print(f"  {usage.ljust(width)}  {help_text}")
        self._print("Alles andere wird als Frage an OBITO gestellt. Strg+C bricht eine laufende Antwort ab.")
        return True

    def cmd_remember(self, arg: str) -> bool:
        if not arg:
            self._print("Aufruf: /merk [art:] <text>")
            return True
        kind = "notiz"
        head, sep, rest = arg.partition(":")
        if sep and head.strip().lower() in KINDS and rest.strip():
            kind = head.strip().lower()
            arg = rest.strip()
        try:
            m = self.brain.remember(arg, kind=kind, project=self.project)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        self._print(f"{SYM_MEMORY} gemerkt: {m.short()}")
        return True

    def cmd_forget(self, arg: str) -> bool:
        mid = self._parse_id(arg, "/vergiss <id>")
        if mid is None:
            return True
        if self.brain.forget(mid):
            self._print(f"Erinnerung #{mid} gelöscht.")
        else:
            self._print(f"Keine Erinnerung mit der ID #{mid}.")
        return True

    def cmd_search(self, arg: str) -> bool:
        if not arg:
            self._print("Aufruf: /suche <frage>")
            return True
        found = self.brain.recall(arg, project=self.project)
        if not found:
            self._print("Keine passenden Erinnerungen.")
            return True
        for m in found:
            self._print(f"  {m.short()}  (Score {m.score:.2f}, Wichtigkeit {m.importance:.2f}, {m.source})")
        return True

    def cmd_memories(self, arg: str) -> bool:
        n = self._parse_count(arg, 10)
        if n is None:
            self._print("Aufruf: /erinnerungen [n]")
            return True
        items = self.brain.memory.recent(n, self.project)
        if not items:
            self._print("Noch keine Erinnerungen.")
            return True
        self._print(f"Die {len(items)} jüngsten Erinnerungen" + (f" (Projekt {self.project})" if self.project else "") + ":")
        for m in items:
            flag = " (unsicher)" if m.source == "ki" else ""
            self._print(f"  {m.short()}{flag}")
        return True

    def cmd_project(self, arg: str) -> bool:
        if not arg:
            self._print(f"Projekt: {self.project or 'keins'}")
            return True
        if arg in ("-", "aus", "keins", "none"):
            self.project = None
            self._print("Projekt gelöscht – Fragen laufen ohne Projektbezug.")
            return True
        self.project = arg
        created = False
        try:
            created = self.brain.projects.get(arg) is None
            self.brain.projects.ensure(arg)
        except Exception as e:  # noqa: BLE001
            self._print(f"{SYM_WARN} Projektsystem: {e}")
        self._print(f"Projekt: {self.project}" + (" (neu angelegt)" if created else ""))
        return True

    def _set_depth(self, depth: str) -> bool:
        self.depth = depth
        labels = {"tief": "tief – 5 Experten, Kritiker, Revision, Synthese",
                  "mittel": "mittel – 3 Experten und Synthese",
                  "schnell": "schnell – ein Aufruf mit Kontext",
                  "auto": "auto – die Heuristik/der Router wählt je Frage"}
        self._print(f"Tiefe: {labels[depth]}")
        return True

    def cmd_depth_deep(self, arg: str) -> bool:
        return self._set_depth("tief")

    def cmd_depth_medium(self, arg: str) -> bool:
        return self._set_depth("mittel")

    def cmd_depth_fast(self, arg: str) -> bool:
        return self._set_depth("schnell")

    def cmd_depth_auto(self, arg: str) -> bool:
        return self._set_depth("auto")

    def cmd_models(self, arg: str) -> bool:
        models = self.brain.models()
        if not models:
            self._print("Keine Modelle abrufbar – ist der Modell-Server erreichbar?")
            return True
        self._print("Installierte Modelle (* = aktiv, » = Routing):")
        for m in models:
            mark = "*" if m.name == self.cfg.model else ("»" if self.cfg.fast_model and m.name == self.cfg.fast_model else " ")
            self._print(f"  {mark} {m.short()}")
        return True

    def cmd_model(self, arg: str) -> bool:
        if not arg:
            self._print(f"Modell: {self.cfg.model}" + (f" · Routing: {self.cfg.fast_model}" if self.cfg.fast_model else ""))
            return True
        fast = False
        parts = arg.split(None, 1)
        if parts[0].lower() in ("schnell", "fast", "routing") and len(parts) == 2:
            fast = True
            arg = parts[1].strip()
        try:
            self.brain.set_model(arg, fast=fast)
        except ModelNotFound as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        except (LLMError, ValueError) as e:
            self._print(f"{SYM_FAIL} Modellwechsel fehlgeschlagen: {e}")
            return True
        if fast:
            self._print(f"Routing-Modell: {self.cfg.fast_model or 'aus (Hauptmodell übernimmt)'}")
        else:
            self._print(f"Modell: {self.cfg.model}")
        return True

    def _target_interaction(self) -> int | None:
        """Interaktion für Feedback: die letzte Antwort dieser Sitzung, sonst die jüngste aus dem Protokoll."""
        if self.last_interaction_id is not None:
            return self.last_interaction_id
        try:
            last = self.brain.learning.last(self.session_id, 1)
        except Exception:  # noqa: BLE001
            last = []
        if last:
            self.last_interaction_id = last[0].id
            return last[0].id
        self._print("Keine bewertbare Antwort – stelle zuerst eine Frage.")
        return None

    def _report_feedback(self, fb: Any, head: str) -> None:
        self._print(head)
        for lesson in fb.lessons:
            self._print(f"{SYM_LESSON} Lektion #{lesson.id} [{lesson.scope}]: {lesson.rule}")
        if fb.memories_adjusted:
            self._print(f"{SYM_MEMORY} {fb.memories_adjusted} Erinnerung(en) angepasst.")

    def cmd_good(self, arg: str) -> bool:
        iid = self._target_interaction()
        if iid is None:
            return True
        try:
            fb = self.brain.feedback(iid, rating=1, comment=arg or None)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        self._report_feedback(fb, f"{SYM_GOOD} Danke – Interaktion #{iid} als gut gespeichert.")
        return True

    def cmd_bad(self, arg: str) -> bool:
        iid = self._target_interaction()
        if iid is None:
            return True
        comment = arg
        if not comment:
            answer = self._read("Was war falsch? ")
            comment = (answer or "").strip()
        try:
            fb = self.brain.feedback(iid, rating=-1, comment=comment or None)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        head = f"{SYM_BAD} Interaktion #{iid} als schlecht gespeichert."
        if not comment:
            head += " Ohne Kommentar entsteht keine Lektion – /schlecht <kommentar> hilft OBITO zu lernen."
        self._report_feedback(fb, head)
        return True

    def cmd_correction(self, arg: str) -> bool:
        if not arg:
            self._print("Aufruf: /korrektur <bessere antwort>")
            return True
        iid = self._target_interaction()
        if iid is None:
            return True
        proposal = self.brain.rewrite_correction(iid, arg)
        correction_full = arg
        if proposal.strip() and normalize(proposal) != normalize(arg):
            self._print("Vorschlag für die vollständige, korrigierte Antwort:")
            self._print(proposal)
            if _yes(self._read("Übernehmen? [J/n] "), default=True):
                correction_full = proposal
            else:
                self._print("Vorschlag verworfen – deine Korrektur wird wörtlich übernommen.")
        try:
            fb = self.brain.feedback(iid, correction=arg, correction_full=correction_full)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        self._report_feedback(fb, f"{SYM_OK} Korrektur zu Interaktion #{iid} gespeichert.")
        return True

    def cmd_lessons(self, arg: str) -> bool:
        n = self._parse_count(arg, 10)
        if n is None:
            self._print("Aufruf: /lektionen [n]")
            return True
        lessons = self.brain.learning.list_lessons(include_inactive=True)
        if not lessons:
            self._print("Noch keine Lektionen – /schlecht <kommentar> oder /korrektur erzeugen welche.")
            return True
        shown = lessons[-n:]
        self._print(f"Lektionen ({len(shown)} von {len(lessons)}):")
        for l in shown:
            state = "" if l.active else " [inaktiv]"
            topics = f" · Themen: {', '.join(l.topics)}" if l.topics else ""
            self._print(f"  #{l.id} [{l.scope}]{state} {l.rule}  (geholfen {l.helped}, geschadet {l.hurt}{topics})")
        return True

    def cmd_delete_lesson(self, arg: str) -> bool:
        lid = self._parse_id(arg, "/lektion-loeschen <id>")
        if lid is None:
            return True
        if self.brain.learning.delete_lesson(lid):
            self._print(f"Lektion #{lid} gelöscht.")
        else:
            self._print(f"Keine Lektion mit der ID #{lid}.")
        return True

    def cmd_trace(self, arg: str) -> bool:
        if self.last_answer is None:
            self._print("Noch keine Antwort in dieser Sitzung.")
            return True
        self._print(f"Denk-Spur ({self.last_answer.depth}):")
        self._print(self.last_answer.trace())
        return True

    def cmd_tools(self, arg: str) -> bool:
        tools = self.brain.tools.list()
        if not tools:
            self._print("Keine Werkzeuge registriert.")
            return True
        state = "aktiv" if self.cfg.allow_tools else "deaktiviert (allow_tools=false)"
        self._print(f"Werkzeuge ({state}):")
        for t in tools:
            flag = " [gefährlich – Rückfrage]" if t.dangerous else ""
            self._print(f"  {t.name}: {t.description}{flag}")
        return True

    def cmd_status(self, arg: str) -> bool:
        self._print(format_status(self.brain.status()))
        return True

    def cmd_export(self, arg: str) -> bool:
        path = Path(arg) if arg else Path(self.cfg.datasets_dir) / "obito.jsonl"
        try:
            stats = self.brain.export_dataset(str(path))
        except (OSError, ValueError) as e:
            self._print(f"{SYM_FAIL} Export fehlgeschlagen: {e}")
            return True
        self._print(format_export_stats(stats, path))
        return True

    def cmd_quit(self, arg: str) -> bool:
        self._print("Bis bald.")
        return False

    # ------------------------------------------------------------ Phase 2
    def cmd_document(self, arg: str) -> bool:
        if not arg:
            self._print("Aufruf: /dokument <pfad>")
            return True
        path = Path(arg).expanduser()
        if not path.exists():
            self._print(f"{SYM_FAIL} {arg} nicht gefunden.")
            return True
        try:
            if path.is_dir():
                r = self.brain.knowledge.add_directory(path, project=self.project)
                self._print(f"{r['hinzugefuegt']} hinzugefügt, {r['unveraendert']} unverändert, "
                            f"{len(r['uebersprungen'])} übersprungen, {len(r['fehler'])} Fehler.")
            else:
                doc = self.brain.knowledge.add_file(path, project=self.project)
                self._print(f"Indexiert: {cli_extra.fmt_document(doc)}")
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
        return True

    def cmd_documents(self, arg: str) -> bool:
        store = self.brain.knowledge
        if arg:
            hits = store.search(arg, k=5, project=self.project)
            if not hits:
                self._print("Keine Treffer in den Dokumenten.")
                return True
            for c in hits:
                self._print(f"{c.cite()} (Score {c.score:.2f})")
                self._print("  " + " ".join(c.content.split())[:300])
            return True
        docs = store.list(self.project)
        if not docs:
            self._print("Noch keine Dokumente. /dokument <pfad> nimmt Dateien oder Ordner auf.")
            return True
        for d in docs:
            self._print("  " + cli_extra.fmt_document(d))
        return True

    def _project_note(self, kind: str, arg: str, usage: str) -> bool:
        if not arg:
            self._print(f"Aufruf: {usage}")
            return True
        if not self.project:
            self._print("Kein Projekt gesetzt – zuerst /projekt <name>.")
            return True
        try:
            self.brain.projects.ensure(self.project)
            note = self.brain.projects.add_note(self.project, kind, arg[:80], arg)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        self._print(f"Festgehalten: {cli_extra.fmt_note(note)}")
        return True

    def cmd_task(self, arg: str) -> bool:
        return self._project_note("aufgabe", arg, "/aufgabe <text>")

    def cmd_decision(self, arg: str) -> bool:
        return self._project_note("entscheidung", arg, "/entscheidung <text>")

    def cmd_project_info(self, arg: str) -> bool:
        name = arg or self.project
        if not name:
            self._print("Kein Projekt gesetzt – /projekt <name> oder /projektinfo <name>.")
            return True
        p = self.brain.projects.get(name)
        if p is None:
            self._print(f"Projekt »{name}« unbekannt.")
            return True
        self._print(cli_extra.fmt_project(p))
        summary = self.brain.projects.summary(name)
        if summary:
            self._print(summary)
        for n in self.brain.projects.notes(name, limit=20):
            self._print("  " + cli_extra.fmt_note(n))
        return True

    def cmd_mission(self, arg: str) -> bool:
        if not arg:
            self._print("Aufruf: /mission <ziel>")
            return True
        try:
            m = self.brain.missions.plan(arg, project=self.project)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        self._print(cli_extra.fmt_mission(m, verbose=True))
        if m.error:
            self._print(f"{SYM_WARN} {m.error}")
        try:
            answer = self.inp("Starten? [J/n] ")
        except (EOFError, KeyboardInterrupt):
            answer = "n"
        if not _yes(answer, True):
            self._print(f"Geplant, nicht gestartet – später: python -m obito mission start {m.id}")
            return True
        self.cancel_event.clear()
        try:
            m = cli_extra.run_mission_foreground(self.brain, m.id, self.out, cancel=self.cancel_event)
        except (ValueError, RuntimeError) as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        self._print(cli_extra.fmt_mission(m, verbose=True))
        return True

    def cmd_missions(self, arg: str) -> bool:
        missions = self.brain.missions.store.list()
        if not missions:
            self._print("Keine Missionen. /mission <ziel> plant eine neue.")
            return True
        running = set(self.brain.missions.running())
        for m in missions:
            self._print("  " + cli_extra.fmt_mission(m) + ("  (läuft)" if m.id in running else ""))
        return True

    def cmd_automations(self, arg: str) -> bool:
        items = self.brain.automation.store.list()
        if not items:
            self._print("Keine Automationen. Einrichten: python -m obito automation vorschlaege --installieren")
            return True
        for a in items:
            self._print("  " + cli_extra.fmt_automation(a))
        self._print("Der Zeitplaner läuft nur mit `python -m obito serve`"
                    + (" – aktiv." if self.brain.automation.running else "."))
        return True

    def cmd_material(self, arg: str) -> bool:
        from . import engineering
        if not arg:
            self._print("Aufruf: /material <name>  (z. B. CFK, Alu 7075, PETG). Bekannt: "
                        + ", ".join(sorted(engineering.MATERIALS)))
            return True
        if engineering.find_material(arg) is None:
            self._print(f"Unbekanntes Material »{arg}«. Bekannt: " + ", ".join(sorted(engineering.MATERIALS)))
            return True
        self._print(engineering.material_info(arg))
        return True

    def cmd_calculator(self, arg: str) -> bool:
        from . import engineering
        tokens = arg.split()
        if not tokens:
            self._print("Rechner (Aufruf: /rechner <name> schluessel=wert …):")
            for name in engineering.TOOL_NAMES:
                t = self.brain.tools.get(name)
                if t is None:
                    continue
                props = t.parameters.get("properties", {})
                req = set(t.parameters.get("required", []))
                self._print(f"  {name}(" + ", ".join(f"{k}{'' if k in req else '?'}" for k in props) + ")")
            return True
        name, params = tokens[0], tokens[1:]
        if self.brain.tools.get(name) is None:
            self._print(f"Unbekannter Rechner »{name}« – /rechner zeigt die Liste.")
            return True
        try:
            kv = cli_extra._parse_kv(params)
        except ValueError as e:
            self._print(f"{SYM_FAIL} {e}")
            return True
        res = self.brain.tools.run(name, kv)
        self._print(res.output if res.ok else f"{SYM_FAIL} {res.error}")
        return True

    def cmd_backup(self, arg: str) -> bool:
        try:
            target = self.brain.backup()
        except OSError as e:
            self._print(f"{SYM_FAIL} Sicherung fehlgeschlagen: {e}")
            return True
        self._print(f"Sicherung angelegt: {target}")
        return True

    def cmd_consolidate(self, arg: str) -> bool:
        days = self._parse_count(arg, 7)
        if days is None:
            self._print("Aufruf: /konsolidieren [tage]")
            return True
        r = self.brain.consolidate(days=days, project=self.project)
        self._print(f"Konsolidiert: {r['zusammengefasst']} Gruppen zusammengefasst, {r['geloescht']} Erinnerungen "
                    f"verdichtet, {r['sitzungen']} Sitzungen zusammengefasst"
                    + (f"; Fehler: {'; '.join(r['fehler'])}" if r.get("fehler") else "."))
        return True

    # ------------------------------------------------------------ Hilfen
    def _parse_id(self, arg: str, usage: str) -> int | None:
        arg = arg.lstrip("#").strip()
        if not arg.isdigit():
            self._print(f"Aufruf: {usage}")
            return None
        return int(arg)

    @staticmethod
    def _parse_count(arg: str, default: int) -> int | None:
        if not arg:
            return default
        if not arg.isdigit() or int(arg) <= 0:
            return None
        return int(arg)


def format_export_stats(stats: dict, path: Path | str) -> str:
    rejected = stats.get("verworfen") or {}
    rej = ", ".join(f"{k} {v}" for k, v in rejected.items()) or "nichts"
    return (f"Datensatz: {stats.get('train', 0)} Trainings-, {stats.get('eval', 0)} Eval-Beispiele; verworfen: {rej}\n"
            f"Dateien: {path}, {path}.eval.jsonl, {path}.fragen.jsonl")


# ------------------------------------------------------------------ doctor
def _probe_speed(res: Any, wall: float) -> tuple[float | None, int]:
    """Tok/s aus dem Probeaufruf: bevorzugt Ollamas ``eval_duration``, sonst Wanduhr."""
    tokens = int(getattr(res, "completion_tokens", 0) or 0)
    raw = getattr(res, "raw", None) or {}
    eval_ns = raw.get("eval_duration") if isinstance(raw, dict) else None
    if tokens <= 0:
        return None, tokens
    if isinstance(eval_ns, (int, float)) and eval_ns > 0:
        return tokens / (float(eval_ns) / 1e9), tokens
    duration = float(getattr(res, "duration", 0.0) or 0.0) or wall
    if duration < 0.001:
        return None, tokens
    return tokens / duration, tokens


def _depth_estimates(cfg: Config, tok_s: float) -> dict[str, float]:
    fast = FAST_TOKENS
    medium = int(cfg.experts_medium) * EXPERT_TOKENS + SYNTHESIS_TOKENS
    deep = (int(cfg.experts_per_question) * EXPERT_TOKENS + CRITIC_TOKENS
            + (int(cfg.max_revised_experts) * EXPERT_TOKENS if int(cfg.max_revision_rounds) > 0 else 0)
            + SYNTHESIS_TOKENS)
    return {"schnell": fast / tok_s, "mittel": medium / tok_s, "tief": deep / tok_s}


def doctor(cfg: Config, *, training: bool = False, out: TextIO = sys.stdout, backend: LLMBackend | None = None,
           env: dict | None = None) -> int:
    """Systemprüfung. Exit-Code 0, wenn ein Chat möglich ist (Backend erreichbar, Probeaufruf ok)."""
    env = os.environ if env is None else env

    def say(text: str) -> None:
        _println(out, text)

    def ok(text: str) -> None:
        say(f"[OK]       {text}")

    def warn(text: str) -> None:
        say(f"[WARNUNG]  {text}")

    def fail(text: str) -> None:
        say(f"[FEHLER]   {text}")

    def info(text: str) -> None:
        say(f"[INFO]     {text}")

    say("OBITO Doctor")
    say("=" * 60)

    # Python
    v = sys.version_info
    if v >= (3, 10):
        ok(f"Python {platform.python_version()} ({platform.system()} {platform.machine()})")
    else:
        fail(f"Python {platform.python_version()} – OBITO braucht Python 3.10 oder neuer.")

    # Backend
    chat_possible = False
    if backend is None:
        try:
            backend = make_backend(cfg)
        except LLMError as e:
            fail(f"Backend-Konfiguration: {e}")
            say("=" * 60)
            say("Ergebnis: Chat nicht möglich.")
            return 1
    url = getattr(backend, "base_url", None) or cfg.base_url or "Standard"
    try:
        available = bool(backend.available())
    except Exception as e:  # noqa: BLE001
        available = False
        warn(f"Backend-Prüfung fehlgeschlagen: {e}")
    version = None
    if available:
        try:
            version = backend.version()
        except Exception:  # noqa: BLE001
            version = None
        ok(f"Backend {backend.name}" + (f" (Version {version})" if version else "") + f" erreichbar ({url})")
    else:
        fail(f"Modell-Server nicht erreichbar ({url}) – starte Ollama (`ollama serve`) oder prüfe base_url/backend "
             f"in der Konfiguration (backend={cfg.backend!r}).")

    # Modelle
    installed: list[str] = []
    model_ok = True
    if available:
        try:
            installed = [m.name for m in backend.list_models()]
        except LLMError as e:
            warn(f"Modellliste nicht abrufbar: {e}")

        def has(name: str) -> bool:
            try:
                return backend.has_model(name)
            except LLMError:
                return False

        if cfg.model and has(cfg.model):
            ok(f"Hauptmodell {cfg.model} installiert")
        elif cfg.model and not installed:
            info(f"Hauptmodell {cfg.model}: Installation nicht prüfbar (Modellliste leer) – der Probeaufruf entscheidet.")
        elif cfg.model:
            model_ok = False
            fail(f"Hauptmodell {cfg.model} fehlt – `ollama pull {cfg.model}`")
        else:
            model_ok = False
            fail("Kein Hauptmodell konfiguriert (model).")
        if cfg.fast_model:
            if has(cfg.fast_model):
                ok(f"Routing-Modell {cfg.fast_model} installiert")
            else:
                warn(f"Routing-Modell {cfg.fast_model} fehlt – `ollama pull {cfg.fast_model}` oder fast_model leeren "
                     f"(dann übernimmt das Hauptmodell).")
        if cfg.embed_model:
            if has(cfg.embed_model):
                ok(f"Embedding-Modell {cfg.embed_model} installiert")
            else:
                warn(f"Embedding-Modell {cfg.embed_model} fehlt – `ollama pull {cfg.embed_model}`; bis dahin nur "
                     f"Volltextsuche im Gedächtnis.")
        else:
            info("Kein Embedding-Modell konfiguriert – Gedächtnis nutzt nur Volltextsuche.")
        if installed:
            info(f"{len(installed)} Modell(e) installiert: {', '.join(installed[:12])}"
                 + (" …" if len(installed) > 12 else ""))

    # Probeaufruf
    tok_s: float | None = None
    if available and cfg.model:
        think = None if str(cfg.think).lower() == "auto" else str(cfg.think).lower() == "an"
        t0 = time.time()
        try:
            res = backend.chat([{"role": "user", "content": "Antworte nur mit: OK"}], model=cfg.model,
                               temperature=0.0, max_tokens=5, num_ctx=cfg.num_ctx, keep_alive=cfg.keep_alive,
                               think=think, timeout=min(float(cfg.timeout), 600.0))
        except ModelNotFound as e:
            fail(f"Probeaufruf: Modell nicht installiert ({e}) – `ollama pull {cfg.model}`")
        except BackendUnavailable as e:
            fail(f"Probeaufruf: {e}")
        except LLMError as e:
            fail(f"Probeaufruf fehlgeschlagen: {e}")
        else:
            wall = time.time() - t0
            chat_possible = True
            tok_s, tokens = _probe_speed(res, wall)
            wall_text = f"{wall:.1f}".replace(".", ",")
            if tok_s:
                speed_text = f"{tok_s:.1f}".replace(".", ",")
                ok(f"Probeaufruf: {wall_text} s (inkl. Laden), {speed_text} Tok/s bei {tokens} Ausgabe-Token")
                est = _depth_estimates(cfg, tok_s)
                info("Prognose je Stufe (nur Ausgabe, Aufrufe nacheinander): "
                     f"schnell ≈ {_fmt_seconds(est['schnell'])} · mittel ≈ {_fmt_seconds(est['mittel'])}"
                     f" · tief ≈ {_fmt_seconds(est['tief'])}"
                     + (f" – parallel_calls={cfg.parallel_calls} kann das Gremium beschleunigen, wenn "
                        f"OLLAMA_NUM_PARALLEL passt." if int(cfg.parallel_calls) > 1 else ""))
                if tok_s < 8:
                    warn("Unter 8 Tok/s wird die Stufe »tief« sehr langsam – kleineres Modell/Quantisierung erwägen "
                         "oder depth: schnell/mittel setzen.")
            else:
                ok(f"Probeaufruf erfolgreich ({wall_text} s) – Geschwindigkeit nicht messbar.")

    # /api/ps – geladene Modelle
    if available:
        try:
            running = backend.running()
        except Exception:  # noqa: BLE001
            running = []
        loaded_names = []
        for m in running:
            name = str(m.get("name") or "?")
            loaded_names.append(name)
            size = m.get("size")
            vram = m.get("size_vram")
            if isinstance(size, (int, float)) and isinstance(vram, (int, float)) and size > 0:
                if vram < size:
                    share = 100.0 * (1 - vram / size)
                    warn(f"{name} läuft teilweise auf CPU ({_fmt_gb(vram)} von {_fmt_gb(size)} im VRAM, ≈ {share:.0f} % "
                         f"ausgelagert) – deutlich langsamer. Kleinere Quantisierung/Modell oder num_ctx verringern.")
                else:
                    ok(f"{name} vollständig im VRAM ({_fmt_gb(vram)})")
        if cfg.fast_model and cfg.fast_model != cfg.model:
            both = all(any(n == x or n.split(":")[0] == x.split(":")[0] for n in loaded_names)
                       for x in (cfg.model, cfg.fast_model))
            if running and both:
                ok("Haupt- und Routing-Modell sind beide geladen")
            else:
                info("fast_model ist nur sinnvoll, wenn Haupt- und Routing-Modell gleichzeitig geladen bleiben "
                     "(genug VRAM, OLLAMA_MAX_LOADED_MODELS ≥ 2) – sonst wird vor jeder Frage umgeladen. Im Zweifel "
                     "fast_model leer lassen.")

    # Denk-Modell
    show = getattr(backend, "show", None)
    if available and callable(show) and cfg.model:
        try:
            details = show(cfg.model) or {}
        except Exception:  # noqa: BLE001
            details = {}
        caps = details.get("capabilities") if isinstance(details, dict) else None
        if isinstance(caps, (list, tuple)) and any(str(c).lower() == "thinking" for c in caps):
            if str(cfg.think).lower() == "aus":
                ok(f"{cfg.model} ist ein Denk-Modell; think: aus ist gesetzt")
            else:
                info(f"{cfg.model} ist ein Denk-Modell (thinking). Empfehlung: think: aus – das Gremium denkt "
                     f"bereits mehrstufig, Denk-Blöcke kosten sonst Zeit und Kontext.")

    # Ollama-Umgebung
    if getattr(backend, "name", "") == "ollama":
        par = env.get("OLLAMA_NUM_PARALLEL")
        if int(cfg.parallel_calls) > 1 and (not par or not par.isdigit() or int(par) < int(cfg.parallel_calls)):
            info(f"parallel_calls={cfg.parallel_calls}, aber OLLAMA_NUM_PARALLEL={par or 'nicht gesetzt'} – Ollama "
                 f"bearbeitet die Experten dann nacheinander. Setze OLLAMA_NUM_PARALLEL={cfg.parallel_calls} "
                 f"(braucht mehr VRAM) oder parallel_calls: 1.")
        if not env.get("OLLAMA_KEEP_ALIVE"):
            info(f"OLLAMA_KEEP_ALIVE ist nicht gesetzt – OBITO sendet keep_alive={cfg.keep_alive} je Aufruf; "
                 f"für andere Clients hilft OLLAMA_KEEP_ALIVE={cfg.keep_alive}.")
        if env.get("OLLAMA_FLASH_ATTENTION", "").lower() not in ("1", "true"):
            info("OLLAMA_FLASH_ATTENTION=1 spart Speicher und Zeit bei langen Kontexten (Ollama neu starten).")
        if not env.get("OLLAMA_KV_CACHE_TYPE"):
            info("OLLAMA_KV_CACHE_TYPE=q8_0 halbiert den KV-Cache (mit Flash Attention) – mehr Kontext bei gleichem VRAM.")

    # Datenverzeichnis
    data = cfg.data_path
    if data.exists():
        if os.access(str(data), os.W_OK):
            ok(f"Datenverzeichnis {data} (beschreibbar)")
        else:
            fail(f"Datenverzeichnis {data} ist nicht beschreibbar.")
    else:
        info(f"Datenverzeichnis {data} existiert noch nicht – wird beim ersten Start angelegt.")

    # Gedächtnis / Lernen
    if cfg.memory_db.exists():
        try:
            mem = MemoryStore(cfg.memory_db)
            try:
                stats = mem.stats()
            finally:
                mem.close()
            msg = (f"Gedächtnis: {stats['erinnerungen']} Erinnerungen ({stats['mit_vektor']} mit Vektor, "
                   f"{stats['ohne_vektor']} ohne), {stats['nachrichten']} Verlaufsnachrichten")
            if stats["ohne_vektor"] > 0 and stats["erinnerungen"] > 0:
                warn(msg + " – Vektoren fehlen: `python -m obito memory reindex` (Embedding-Modell nötig).")
            else:
                ok(msg)
        except Exception as e:  # noqa: BLE001
            fail(f"Gedächtnis-Datenbank {cfg.memory_db} nicht lesbar: {e}")
    else:
        info("Gedächtnis ist noch leer (keine Datenbank).")
    if cfg.learning_db.exists():
        try:
            learn = LearningStore(cfg.learning_db)
            try:
                ls = learn.stats()
            finally:
                learn.close()
            ok(f"Lernen: {ls['interaktionen']} Interaktionen, {ls['bewertet']} bewertet, {ls['korrigiert']} korrigiert, "
               f"{ls['lektionen_aktiv']} aktive Lektionen")
        except Exception as e:  # noqa: BLE001
            fail(f"Lern-Datenbank {cfg.learning_db} nicht lesbar: {e}")

    # Training
    if training:
        say("-" * 60)
        say("Training (LoRA):")
        missing = []
        for pkg in TRAINING_PACKAGES:
            try:
                found = importlib.util.find_spec(pkg) is not None
            except (ImportError, ValueError):
                found = False
            if found:
                ok(f"Paket {pkg} vorhanden")
            else:
                missing.append(pkg)
                (warn if pkg in ("trl", "bitsandbytes", "datasets") else fail)(f"Paket {pkg} fehlt")
        if missing:
            info("Installation: pip install " + " ".join(missing) + "  (NVIDIA-GPU mit CUDA empfohlen; unter Windows WSL2)")
        _doctor_vram(ok, warn, info)

    chat_possible = chat_possible and model_ok
    say("=" * 60)
    say("Ergebnis: " + ("Chat möglich." if chat_possible else "Chat nicht möglich – siehe [FEHLER] oben."))
    return 0 if chat_possible else 1


def _doctor_vram(ok: Callable[[str], None], warn: Callable[[str], None], info: Callable[[str], None]) -> None:
    try:
        import torch  # type: ignore  # noqa: WPS433 – nur bei --training
    except Exception:  # noqa: BLE001
        info("VRAM nicht prüfbar (torch fehlt).")
        return
    try:
        if not torch.cuda.is_available():
            warn("Keine CUDA-GPU gefunden – LoRA-Training auf CPU ist praktisch nicht machbar.")
            return
        free, total = torch.cuda.mem_get_info()
        name = torch.cuda.get_device_name(0)
        total_gb = total / 1e9
        ok(f"GPU {name}: {_fmt_gb(free)} frei von {_fmt_gb(total)}")
        if total_gb < 6:
            warn("Unter 6 GB VRAM reicht es höchstens für 1,5B-Modelle (QLoRA).")
        elif total_gb < 10:
            info("6–10 GB VRAM: 3B-Modelle mit QLoRA; 7B wird eng.")
        elif total_gb < 20:
            info("10–20 GB VRAM: 7B mit QLoRA (4-bit) möglich.")
        else:
            ok("≥ 20 GB VRAM: 7B-LoRA in bf16 möglich.")
    except Exception as e:  # noqa: BLE001
        warn(f"VRAM-Prüfung fehlgeschlagen: {e}")


# ------------------------------------------------------------------ config
def recommend_config(vram_gb: float, base: Config | None = None) -> Config:
    """Konfigurationsvorlage nach Grafikspeicher (siehe Tabelle in der Architektur)."""
    cfg = Config(**(base.to_dict() if base is not None else {}))
    cfg.embed_model = cfg.embed_model or "nomic-embed-text"
    cfg.depth = "auto"
    if vram_gb >= 16:
        cfg.model, cfg.fast_model, cfg.num_ctx, cfg.parallel_calls = "qwen2.5:14b", "qwen2.5:3b", 16384, 2
    elif vram_gb >= 12:
        cfg.model, cfg.fast_model, cfg.num_ctx, cfg.parallel_calls = "qwen2.5:7b", "qwen2.5:3b", 12288, 2
    elif vram_gb >= 8:
        cfg.model, cfg.fast_model, cfg.num_ctx, cfg.parallel_calls = "qwen2.5:7b", "", 8192, 1
    else:
        cfg.model, cfg.fast_model, cfg.num_ctx, cfg.parallel_calls = "qwen2.5:3b", "", 4096, 1
        cfg.depth = "schnell"
    return cfg


def _recommendation_note(vram_gb: float, cfg: Config) -> str:
    if vram_gb >= 16:
        return (f"{vram_gb:g} GB VRAM: {cfg.model} als Hauptmodell, {cfg.fast_model} für Routing/Extraktion, "
                f"num_ctx {cfg.num_ctx}, {cfg.parallel_calls} parallele Aufrufe.")
    if vram_gb >= 12:
        return (f"{vram_gb:g} GB VRAM: {cfg.model} plus kleines Routing-Modell {cfg.fast_model}, num_ctx {cfg.num_ctx}, "
                f"{cfg.parallel_calls} parallele Aufrufe.")
    if vram_gb >= 8:
        return (f"{vram_gb:g} GB VRAM: {cfg.model}, kein separates Routing-Modell, num_ctx {cfg.num_ctx}, "
                f"ein Aufruf zugleich.")
    return (f"{vram_gb:g} GB VRAM oder nur CPU: {cfg.model}, Tiefe »schnell« als Standard (Gremium per /tief "
            f"gezielt nutzen), num_ctx {cfg.num_ctx}.")


def cmd_config(cfg: Config, args: argparse.Namespace, out: TextIO, config_file: Path | None) -> int:
    if args.empfehlen is not None:
        vram = float(args.empfehlen)
        if vram < 0:
            _println(out, "Fehler: --empfehlen erwartet den Grafikspeicher in GB (0 = nur CPU).")
            return 2
        rec = recommend_config(vram, cfg)
        note = _recommendation_note(vram, rec)
        payload = {"_hinweis": "Von `python -m obito config --empfehlen` erzeugte Vorlage. " + note}
        payload.update(rec.to_dict())
        if args.schreiben:
            target = Path(args.schreiben)
        else:
            target = Path("obito.json")
            if target.exists():
                target = Path("obito.empfohlen.json")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except OSError as e:
            _println(out, f"Fehler: Vorlage konnte nicht geschrieben werden ({target}): {e}")
            return 1
        _println(out, f"Empfehlung: {note}")
        _println(out, f"Vorlage geschrieben: {target}")
        if target.name != "obito.json" and not args.schreiben:
            _println(out, "Hinweis: obito.json existiert bereits und wurde nicht überschrieben – Werte bei Bedarf übernehmen.")
        _println(out, f"Modelle laden: ollama pull {rec.model}" + (f" && ollama pull {rec.fast_model}" if rec.fast_model else "")
                 + f" && ollama pull {rec.embed_model}")
        return 0
    if args.schreiben:
        try:
            p = save_config(cfg, args.schreiben)
        except OSError as e:
            _println(out, f"Fehler: Konfiguration konnte nicht geschrieben werden: {e}")
            return 1
        _println(out, f"Konfiguration geschrieben: {p}")
        return 0
    _println(out, f"Konfigurationsdatei: {config_file if config_file else 'keine (Standardwerte + Umgebungsvariablen)'}")
    _println(out, json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    return 0


# ------------------------------------------------------------------ memory
def _attach_embedder(store: MemoryStore, cfg: Config, backend: LLMBackend | None) -> bool:
    """Hängt den Embedder des Backends an, wenn der Server erreichbar ist. Rückgabe: aktiv?"""
    backend = backend if backend is not None else make_backend(cfg)
    try:
        if not backend.available():
            return False
    except Exception:  # noqa: BLE001
        return False
    store.set_embedder(lambda texts: backend.embed(list(texts), model=cfg.embed_model), cfg.embed_model)
    return True


def cmd_memory(cfg: Config, args: argparse.Namespace, out: TextIO, backend: LLMBackend | None = None) -> int:
    action = args.aktion
    if action is None:
        _println(out, "Aufruf: python -m obito memory {list,search,export,import,reindex} … (--help für Details)")
        return 2
    if action in ("list", "export") and not cfg.memory_db.exists():
        _println(out, f"Noch kein Gedächtnis unter {cfg.memory_db}.")
        if action == "export":
            return 1
        return 0
    cfg.ensure_dirs()
    store = MemoryStore(cfg.memory_db)
    try:
        if action == "list":
            items = store.recent(args.anzahl, args.projekt)
            if not items:
                _println(out, "Keine Erinnerungen.")
                return 0
            for m in items:
                flag = " (unsicher)" if m.source == "ki" else ""
                _println(out, f"{m.short()}{flag}  [Wichtigkeit {m.importance:.2f}, {m.source}]")
            _println(out, f"{len(items)} Erinnerung(en) von {store.count()}.")
            return 0
        if action == "search":
            _attach_embedder(store, cfg, backend)
            found = store.search(args.frage, k=args.anzahl, project=args.projekt, touch=False, min_importance=0.0)
            if not found:
                _println(out, "Keine passenden Erinnerungen.")
                return 0
            for m in found:
                _println(out, f"{m.short()}  (Score {m.score:.2f}, Wichtigkeit {m.importance:.2f}, {m.source})")
            return 0
        if action == "export":
            try:
                n = store.export(args.pfad)
            except OSError as e:
                _println(out, f"Fehler: Export nach {args.pfad} fehlgeschlagen: {e}")
                return 1
            _println(out, f"{n} Erinnerung(en) nach {args.pfad} exportiert.")
            return 0
        if action == "import":
            active = _attach_embedder(store, cfg, backend)
            try:
                n = store.import_json(args.pfad)
            except FileNotFoundError:
                _println(out, f"Fehler: Datei {args.pfad} nicht gefunden.")
                return 1
            except (OSError, ValueError, KeyError, TypeError) as e:
                _println(out, f"Fehler: {args.pfad} ist keine gültige Export-Datei: {e}")
                return 1
            _println(out, f"{n} Erinnerung(en) aus {args.pfad} importiert (Duplikate zusammengeführt); "
                          f"Bestand: {store.count()}." + ("" if active else " Ohne Embedding-Server: keine Vektoren – "
                                                                             "später `memory reindex`."))
            return 0
        if action == "reindex":
            if not _attach_embedder(store, cfg, backend):
                _println(out, "Fehler: Modell-Server nicht erreichbar – ohne Embedding-Modell gibt es keine Vektoren.")
                return 1

            def progress(done: int, total: int) -> None:
                _println(out, f"  {done}/{total} Vektoren berechnet")

            n = store.reindex(progress=progress, only_missing=not args.alle)
            stats = store.stats()
            if n == 0 and stats["ohne_vektor"] > 0:
                _println(out, f"Keine Vektoren berechnet – liefert das Embedding-Modell {cfg.embed_model} Vektoren? "
                              f"(`ollama pull {cfg.embed_model}`)")
                return 1
            _println(out, f"{n} Erinnerung(en) neu eingebettet ({'alle veralteten/fehlenden' if args.alle else 'nur fehlende'}); "
                          f"jetzt {stats['mit_vektor']} mit und {stats['ohne_vektor']} ohne Vektor"
                          f" (Modell {stats['embedding_modell'] or '?'}, {stats['embedding_dim']} Dim.).")
            return 0
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        store.close()


# ------------------------------------------------------------------ export-dataset
def cmd_export_dataset(cfg: Config, args: argparse.Namespace, out: TextIO) -> int:
    if not cfg.learning_db.exists():
        _println(out, f"Noch keine Interaktionen protokolliert ({cfg.learning_db} fehlt) – zuerst chatten und bewerten.")
        return 1
    store = LearningStore(cfg.learning_db)
    try:
        stats = store.export_dataset(
            args.ausgabe, fmt=args.format, min_rating=args.min_bewertung, eval_share=args.eval_anteil,
            include_context=not args.ohne_kontext, max_own=args.max_eigene,
        )
    except (OSError, ValueError) as e:
        _println(out, f"Fehler: Export fehlgeschlagen: {e}")
        return 1
    finally:
        store.close()
    _println(out, format_export_stats(stats, args.ausgabe))
    if stats.get("train", 0) == 0:
        _println(out, "Hinweis: 0 Trainingsbeispiele – bewerte Antworten mit /gut oder korrigiere sie mit /korrektur.")
    return 0


# ------------------------------------------------------------------ eval
def _training_questions(path: str) -> set[str]:
    """Normalisierte Fragen aus einer Trainingsdatei (chat/alpaca/dpo-JSONL)."""
    found: set[str] = set()
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            text: str | None = None
            msgs = row.get("messages")
            if isinstance(msgs, list):
                users = [m.get("content") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
                if users and isinstance(users[-1], str):
                    text = users[-1]
            for key in ("instruction", "prompt", "frage"):
                if text is None and isinstance(row.get(key), str):
                    text = row[key]
            if text:
                found.add(normalize(text))
    return found


def _apply_training_exclusions(items: list[dict], path: str, out: TextIO) -> tuple[list[dict], set]:
    """Items, deren Frage in den Trainingsdaten vorkommt: mit ``quelle_id`` → ``exclude_ids``,
    ohne ``quelle_id`` → direkt aussortiert (sonst misst der Eval auswendig Gelerntes)."""
    train = _training_questions(path)
    exclude: set = set()
    kept: list[dict] = []
    dropped = 0
    for it in items:
        q = normalize(str(it.get("frage") or ""))
        hit = bool(q) and any(q == t or t.endswith(q) or q in t for t in train)
        if hit and it.get("quelle_id") is not None:
            exclude.add(it["quelle_id"])
            kept.append(it)
        elif hit:
            dropped += 1
        else:
            kept.append(it)
    if dropped:
        _println(out, f"{dropped} Frage(n) ohne quelle_id kommen in {path} vor und werden übersprungen.")
    if exclude:
        _println(out, f"{len(exclude)} Frage(n) sind in den Trainingsdaten enthalten und werden ausgeschlossen.")
    return kept, exclude


def _print_report(report: dict, out: TextIO) -> None:
    def pct(x: float | None) -> str:
        return "–" if x is None else f"{100 * x:.0f} %"

    judge = report.get("richter_score")
    judge_text = "–" if judge is None else f"{judge:.1f}/10"
    dauer = float(report.get("dauer_mittel") or 0.0)
    tokens = float(report.get("tokens_mittel") or 0.0)
    _println(out, f"Modell: {report.get('modell')}" + (f" · Tiefe: {report['tiefe']}" if report.get("tiefe") else ""))
    _println(out, f"Fragen: {report.get('anzahl', 0)} · Stichwort-Score: {pct(report.get('stichwort_score'))}"
                  f" · Richter: {judge_text} · Dauer/Frage: {dauer:.1f} s · Tokens/Frage: {tokens:.0f}")
    for w in report.get("warnungen") or []:
        _println(out, f"Warnung: {w}")


def cmd_eval(cfg: Config, args: argparse.Namespace, out: TextIO, backend: LLMBackend | None = None) -> int:
    try:
        from .training import evaluate as ev
    except Exception as e:  # noqa: BLE001 – Modul entsteht parallel / kann defekt sein
        _println(out, f"Fehler: Eval-Modul (obito.training.evaluate) nicht verfügbar: {e}")
        return 2

    if args.vergleich:
        try:
            report_a = ev.load_report(args.vergleich[0])
            report_b = ev.load_report(args.vergleich[1])
        except (OSError, ValueError) as e:
            _println(out, f"Fehler: Bericht nicht lesbar: {e}")
            return 1
        judge_backend = None
        if args.richter:
            judge_backend = backend if backend is not None else make_backend(cfg)
        try:
            result = ev.compare(report_a, report_b, backend=judge_backend, judge_model=args.richter)
        except LLMError as e:
            _println(out, f"Fehler beim Vergleich: {e}")
            return 1
        _println(out, f"Vergleich A = {report_a.get('modell')} ({report_a.get('tiefe') or 'direkt'}) "
                      f"gegen B = {report_b.get('modell')} ({report_b.get('tiefe') or 'direkt'}), n = {result.get('n')}")
        _println(out, f"Richter: A {result.get('siege_a')} · B {result.get('siege_b')} · gleich {result.get('gleich')}")
        delta = result.get("stichwort_delta")
        delta_text = "–" if delta is None else f"{100 * float(delta):+.0f} %"
        dauer_delta = float(result.get("dauer_delta") or 0.0)
        tokens_delta = float(result.get("tokens_delta") or 0.0)
        _println(out, f"Stichwort-Delta (B − A): {delta_text} · Dauer-Delta: {dauer_delta:+.1f} s"
                      f" · Token-Delta: {tokens_delta:+.0f}")
        if result.get("hinweis"):
            _println(out, f"Hinweis: {result['hinweis']}")
        if args.ausgabe:
            ev.save_report(result, args.ausgabe)
            _println(out, f"Vergleich gespeichert: {args.ausgabe}")
        return 0

    if not args.datei:
        _println(out, "Fehler: --datei FRAGEN.jsonl oder --vergleich A.json B.json angeben.")
        return 2
    try:
        items = ev.load_items(args.datei)
    except (OSError, ValueError) as e:
        _println(out, f"Fehler: {e}")
        return 1
    exclude: set = set()
    if args.ohne_training:
        try:
            items, exclude = _apply_training_exclusions(items, args.ohne_training, out)
        except OSError as e:
            _println(out, f"Fehler: Trainingsdaten {args.ohne_training} nicht lesbar: {e}")
            return 1
    if not items:
        _println(out, "Keine Fragen zu bewerten.")
        return 1

    model = cfg.model            # --modell (global) ist bereits in cfg.model eingeflossen
    backend = backend if backend is not None else make_backend(cfg)
    total = len(items)

    def progress(done: int, n: int, ergebnis: dict) -> None:
        score = ergebnis.get("stichwort")
        shown = "–" if score is None else f"{100 * score:.0f} %"
        _println(out, f"  [{done}/{n}] Stichwort {shown} · {ergebnis.get('dauer', 0):.1f} s · {ergebnis.get('frage', '')[:70]}")

    _println(out, f"Eval: {total} Frage(n) aus {args.datei} mit {model}"
                  + (f" über den Denkkern (Tiefe {args.tiefe})" if args.tiefe else " (direkter Modellaufruf)")
                  + (f", Richter {args.richter}" if args.richter else ""))
    brain: Brain | None = None
    try:
        if args.tiefe:
            brain = Brain(cfg, backend=backend)
            report = ev.run_eval(backend, model, items, judge_model=args.richter, progress=progress,
                                 through_brain=brain, depth=args.tiefe, exclude_ids=frozenset(exclude),
                                 num_ctx=cfg.num_ctx)
        else:
            system_prompt = None
            try:
                from .training.modelfile import default_system_prompt
                system_prompt = default_system_prompt()
            except Exception:  # noqa: BLE001
                system_prompt = None
            report = ev.run_eval(backend, model, items, judge_model=args.richter, system_prompt=system_prompt,
                                 progress=progress, exclude_ids=frozenset(exclude), num_ctx=cfg.num_ctx)
    except BackendUnavailable as e:
        _println(out, f"Fehler: Modell-Server nicht erreichbar – {e}")
        return 1
    except LLMError as e:
        _println(out, f"Fehler beim Eval: {e}")
        return 1
    finally:
        if brain is not None:
            brain.close()
    _print_report(report, out)
    if args.ausgabe:
        try:
            ev.save_report(report, args.ausgabe)
        except OSError as e:
            _println(out, f"Fehler: Bericht konnte nicht gespeichert werden: {e}")
            return 1
        _println(out, f"Bericht gespeichert: {args.ausgabe}")
    return 0 if report.get("anzahl", 0) > 0 else 1


# ------------------------------------------------------------------ modelfile
def cmd_modelfile(cfg: Config, args: argparse.Namespace, out: TextIO) -> int:
    try:
        from .training import modelfile as mf
    except Exception as e:  # noqa: BLE001
        _println(out, f"Fehler: Modelfile-Modul (obito.training.modelfile) nicht verfügbar: {e}")
        return 2
    template = None
    if args.vorlage:
        try:
            template = Path(args.vorlage).read_text(encoding="utf-8")
        except OSError as e:
            _println(out, f"Fehler: Vorlage {args.vorlage} nicht lesbar: {e}")
            return 1
    if args.adapter and not Path(args.adapter).exists():
        _println(out, f"Fehler: Adapter {args.adapter} nicht gefunden.")
        return 1
    try:
        text = mf.build_modelfile(args.basis, mf.default_system_prompt(), adapter=args.adapter, template=template,
                                  temperature=cfg.temperature, num_ctx=cfg.num_ctx)
    except ValueError as e:
        _println(out, f"Fehler: {e}")
        return 1
    warning = mf.check_adapter_compat(args.adapter, args.basis) if args.adapter else None
    if warning:
        _println(out, warning)
        if args.erstellen and not args.erzwingen:
            _println(out, "Abbruch – mit --erzwingen trotzdem erstellen.")
            return 1
    _println(out, text.rstrip("\n"))
    if not args.erstellen:
        _println(out, f"\nErstellen mit: python -m obito modelfile --name {args.name} --basis {args.basis}"
                      + (f" --adapter {args.adapter}" if args.adapter else "") + " --erstellen")
        return 0
    cwd = None
    for p in (args.adapter, args.basis):
        if p and Path(p).exists():
            cwd = str(Path(p).resolve().parent if Path(p).is_file() else Path(p).resolve())
            break
    try:
        proc = mf.create_model(args.name, text, cwd=cwd)
    except (ValueError, FileNotFoundError, PermissionError) as e:
        _println(out, f"Fehler: {e}")
        return 1
    if proc.returncode != 0:
        _println(out, f"Fehler: `ollama create` endete mit Code {proc.returncode}:")
        _println(out, (proc.stderr or proc.stdout or "").strip())
        return 1
    _println(out, f"Modell »{args.name}« erstellt. Testen: python -m obito --modell {args.name} chat")
    _println(out, f"Vergleichen: python -m obito eval --datei FRAGEN.jsonl --modell {args.name} --ausgabe neu.json")
    return 0


# ------------------------------------------------------------------ serve / chat / train
def cmd_serve(cfg: Config, args: argparse.Namespace, out: TextIO, backend: LLMBackend | None = None) -> int:
    try:
        from .server import ObitoServer
    except Exception as e:  # noqa: BLE001
        _println(out, f"Fehler: Server-Modul (obito.server) nicht verfügbar: {e}")
        return 2
    host = args.host or cfg.server_host
    port = cfg.server_port if args.port is None else int(args.port)
    brain = Brain(cfg, backend=backend)
    try:
        try:
            server = ObitoServer(brain, host, port, allow_dangerous=bool(args.gefaehrlich_erlauben))
        except OSError as e:
            _println(out, f"Fehler: Server kann nicht auf {host}:{port} lauschen: {e}")
            return 1
        shown_host = f"[{host}]" if ":" in host else host
        _println(out, f"OBITO-HUD: http://{shown_host}:{server.port}/  (Strg+C beendet)"
                      + ("  – gefährliche Werkzeuge FREIGEGEBEN" if args.gefaehrlich_erlauben else ""))
        if server.start_background():
            _println(out, "Automations-Zeitplaner läuft.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            _println(out, "\nServer wird beendet.")
        finally:
            server.server_close()
        return 0
    finally:
        brain.close()


def cmd_chat(cfg: Config, args: argparse.Namespace, out: TextIO, inp: Callable[[str], str],
             backend: LLMBackend | None = None) -> int:
    brain = Brain(cfg, backend=backend)
    try:
        session = ChatSession(brain, cfg, out=out, inp=inp, session_id=args.sitzung, project=args.projekt,
                              depth=args.tiefe)
        return session.run()
    finally:
        brain.close()


def cmd_train(train_argv: Sequence[str], out: TextIO) -> int:
    try:
        from .training import train_lora
    except Exception as e:  # noqa: BLE001
        _println(out, f"Fehler: Trainingsmodul (obito.training.train_lora) nicht verfügbar: {e}")
        return 2
    result = train_lora.main(list(train_argv))
    return int(result or 0)


# ------------------------------------------------------------------ argparse
class _Parser(argparse.ArgumentParser):
    """ArgumentParser mit deutschen Fehlermeldungen."""

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: Fehler: {message}\n")


class _Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog: str, **kw: Any) -> None:
        super().__init__(prog, max_help_position=30, **kw)

    def _format_usage(self, usage, actions, groups, prefix):  # type: ignore[override]
        return super()._format_usage(usage, actions, groups, "Aufruf: " if prefix is None else prefix)


def _germanize(p: argparse.ArgumentParser) -> None:
    p._positionals.title = "Argumente"
    p._optionals.title = "Optionen"
    p.add_argument("-h", "--hilfe", "--help", action="help", help="diese Hilfe anzeigen")


def _add_global_options(p: argparse.ArgumentParser, suppress: bool) -> None:
    d: Any = argparse.SUPPRESS if suppress else None
    p.add_argument("--config", default=d, metavar="PFAD", help="Konfigurationsdatei (Standard: obito.json oder ~/.obito/config.json)")
    p.add_argument("--backend", default=d, metavar="ART", help="Backend: auto, ollama, openai oder fake")
    p.add_argument("--daten", default=d, metavar="PFAD", help="Datenverzeichnis (Standard: ~/.obito)")
    p.add_argument("--modell", default=d, metavar="M", help="Hauptmodell, z. B. qwen2.5:7b")


def build_parser() -> argparse.ArgumentParser:
    """Baut den Argument-Parser (deutsch). Globale Optionen gelten vor und nach dem Unterbefehl."""
    parser = _Parser(
        prog="python -m obito", formatter_class=_Formatter, add_help=False,
        description="OBITO – lokaler, lernfähiger KI-Assistent (Ollama / OpenAI-kompatibel, 100 % offline).",
        epilog="Ohne Unterbefehl startet der Chat. Beispiele:\n"
               "  python -m obito                      Chat\n"
               "  python -m obito doctor               Systemprüfung\n"
               "  python -m obito serve                HUD im Browser (http://127.0.0.1:8765)\n"
               "  python -m obito config --empfehlen 8 Konfigurationsvorlage für 8 GB VRAM\n"
               "  python -m obito wissen add doku.pdf  Dokument ins Datenzentrum\n"
               "  python -m obito mission neu \"…\"      Mehrstufige Aufgabe planen\n"
               "  python -m obito rechner akku_rechner zellen=4 mah=1500 strom_a=20",
    )
    _germanize(parser)
    _add_global_options(parser, suppress=False)
    sub = parser.add_subparsers(dest="befehl", metavar="UNTERBEFEHL", title="Unterbefehle")

    def add(name: str, help_text: str, **kw: Any) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help_text, description=help_text, formatter_class=_Formatter, add_help=False, **kw)
        _germanize(sp)
        _add_global_options(sp, suppress=True)
        return sp

    p = add("chat", "Interaktiver Chat mit Streaming (Standard).")
    p.add_argument("--projekt", default=None, metavar="P", help="Projektbezug für Erinnerungen")
    p.add_argument("--tiefe", default=None, choices=DEPTH_CHOICES, help="Denk-Tiefe (Standard: aus der Konfiguration)")
    p.add_argument("--sitzung", default="standard", metavar="S", help="Sitzungs-ID für den Verlauf (Standard: standard)")

    p = add("doctor", "Systemprüfung: Python, Backend, Modelle, Probeaufruf, VRAM, Gedächtnis.")
    p.add_argument("--training", action="store_true", help="zusätzlich Trainingsumgebung prüfen (torch, peft, VRAM)")

    p = add("serve", "HTTP-Server mit HUD-Oberfläche starten.")
    p.add_argument("--host", default=None, help="Adresse (Standard: server_host, 127.0.0.1)")
    p.add_argument("--port", default=None, type=int, help="Port (Standard: server_port, 8765)")
    p.add_argument("--gefaehrlich-erlauben", action="store_true",
                   help="gefährliche Werkzeuge (Dateien schreiben, Code/Befehle ausführen) ohne Rückfrage erlauben")

    p = add("export-dataset", "Trainingsdatensatz aus bewerteten/korrigierten Antworten exportieren.")
    p.add_argument("--ausgabe", required=True, metavar="PFAD", help="Ziel-JSONL (dazu .eval.jsonl und .fragen.jsonl)")
    p.add_argument("--format", default="chat", choices=DATASET_FORMATS, help="Datensatzformat (Standard: chat)")
    p.add_argument("--min-bewertung", default=1, type=int, metavar="N", help="Mindestbewertung eigener Antworten (Standard: 1)")
    p.add_argument("--eval-anteil", default=0.1, type=float, metavar="ANTEIL", help="Anteil für den Eval-Split (Standard: 0.1)")
    p.add_argument("--ohne-kontext", action="store_true", help="Erinnerungen/Lektionen nicht in den Prompt schreiben")
    p.add_argument("--max-eigene", default=None, type=int, metavar="N", help="höchstens N eigene positive Antworten")

    p = add("eval", "Fragen-Set auswerten (Stichwort-Score, optional Richter-Modell) oder zwei Berichte vergleichen.")
    p.add_argument("--datei", default=None, metavar="FRAGEN.jsonl", help="Fragen (JSONL: frage, erwartet, stichworte …)")
    p.add_argument("--richter", default=None, metavar="M", help="Richter-Modell (muss ein anderes Modell sein)")
    p.add_argument("--tiefe", default=None, choices=EVAL_DEPTHS, help="durch den Denkkern statt direkt ans Modell")
    p.add_argument("--ausgabe", default=None, metavar="BERICHT.json", help="Bericht speichern")
    p.add_argument("--ohne-training", default=None, metavar="DATEN.jsonl",
                   help="Fragen ausschließen, die in dieser Trainingsdatei vorkommen")
    p.add_argument("--vergleich", nargs=2, default=None, metavar=("A.json", "B.json"), help="zwei Berichte vergleichen")

    p = add("modelfile", "Ollama-Modelfile für ein eigenes Modell (mit LoRA-Adapter) erzeugen.")
    p.add_argument("--name", default="obito-v1", metavar="N", help="Name des neuen Modells (Standard: obito-v1)")
    p.add_argument("--basis", required=True, metavar="B", help="Basis: Ollama-Tag (qwen2.5:7b) oder GGUF-Datei")
    p.add_argument("--adapter", default=None, metavar="PFAD", help="LoRA-Adapter (adapter.gguf oder Verzeichnis)")
    p.add_argument("--vorlage", default=None, metavar="PFAD", help="Chat-Template-Datei (Pflicht bei GGUF außer Qwen)")
    p.add_argument("--erstellen", action="store_true", help="`ollama create` ausführen")
    p.add_argument("--erzwingen", action="store_true", help="trotz Kompatibilitätswarnung erstellen")

    add("train", "LoRA-Feintuning (alle weiteren Optionen gehen an das Trainingsmodul, siehe `train --help`).")

    p = add("memory", "Gedächtnis verwalten.")
    msub = p.add_subparsers(dest="aktion", metavar="AKTION", title="Aktionen")

    def madd(name: str, help_text: str) -> argparse.ArgumentParser:
        mp = msub.add_parser(name, help=help_text, description=help_text, formatter_class=_Formatter, add_help=False)
        _germanize(mp)
        _add_global_options(mp, suppress=True)
        return mp

    mp = madd("list", "jüngste Erinnerungen anzeigen")
    mp.add_argument("-n", "--anzahl", default=20, type=int, help="Anzahl (Standard: 20)")
    mp.add_argument("--projekt", default=None, help="nur dieses Projekt (plus projektlose)")
    mp = madd("search", "Erinnerungen durchsuchen")
    mp.add_argument("frage", help="Suchtext")
    mp.add_argument("-n", "--anzahl", default=10, type=int, help="Anzahl (Standard: 10)")
    mp.add_argument("--projekt", default=None, help="nur dieses Projekt (plus projektlose)")
    mp = madd("export", "Erinnerungen als JSON exportieren")
    mp.add_argument("pfad", help="Zieldatei")
    mp = madd("import", "Erinnerungen aus JSON importieren")
    mp.add_argument("pfad", help="Quelldatei")
    mp = madd("reindex", "Vektoren neu berechnen")
    mp.add_argument("--alle", action="store_true", help="auch veraltete Vektoren (Modellwechsel), nicht nur fehlende")

    p = add("config", "Wirksame Konfiguration anzeigen, schreiben oder eine Vorlage empfehlen.")
    p.add_argument("--schreiben", default=None, metavar="PFAD", help="Konfiguration (oder Vorlage) in diese Datei schreiben")
    p.add_argument("--empfehlen", default=None, type=float, metavar="VRAM_GB",
                   help="Vorlage nach Grafikspeicher: <8 (CPU), 8, 12 oder 16 GB")

    cli_extra.add_parsers(add, _germanize, _add_global_options, _Formatter)
    return parser


def _split_train_argv(argv: list[str]) -> tuple[list[str], list[str] | None]:
    """Trennt ``train``-Argumente ab: alles nach dem Unterbefehl ``train`` geht an ``train_lora``."""
    i = 0
    while i < len(argv):
        tok = argv[i]
        if tok in GLOBAL_VALUE_OPTIONS:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if tok == "train":
            return argv[:i + 1], argv[i + 1:]
        return argv, None
    return argv, None


def _apply_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if getattr(args, "backend", None):
        cfg.backend = args.backend
    if getattr(args, "daten", None):
        cfg.data_dir = args.daten
    if getattr(args, "modell", None):
        cfg.model = args.modell


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None, inp: Callable[[str], str] = input,
         backend: LLMBackend | None = None) -> int:
    """Einstiegspunkt der Kommandozeile. ``out``/``inp``/``backend`` dienen Tests (FakeBackend)."""
    out = out if out is not None else sys.stdout
    argv = list(sys.argv[1:] if argv is None else argv)
    argv, train_argv = _split_train_argv(argv)
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except (OSError, ValueError) as e:
        _println(out, f"Fehler: Konfiguration konnte nicht geladen werden: {e}")
        return 2
    config_file = find_config_file(args.config)
    _apply_overrides(cfg, args)
    cmd = args.befehl or "chat"

    try:
        if cmd == "chat":
            if args.befehl is None:
                args.projekt, args.tiefe, args.sitzung = None, None, "standard"
            return cmd_chat(cfg, args, out, inp, backend=backend)
        if cmd == "doctor":
            return doctor(cfg, training=bool(args.training), out=out, backend=backend)
        if cmd == "serve":
            return cmd_serve(cfg, args, out, backend=backend)
        if cmd == "export-dataset":
            return cmd_export_dataset(cfg, args, out)
        if cmd == "eval":
            return cmd_eval(cfg, args, out, backend=backend)
        if cmd == "modelfile":
            return cmd_modelfile(cfg, args, out)
        if cmd == "train":
            return cmd_train(train_argv or [], out)
        if cmd == "memory":
            return cmd_memory(cfg, args, out, backend=backend)
        if cmd == "config":
            return cmd_config(cfg, args, out, config_file)
        if cmd in cli_extra.COMMANDS:
            return cli_extra.run(cmd, cfg, args, out, inp, backend)
    except LLMError as e:
        _println(out, f"Fehler: {e}")
        return 1
    except KeyboardInterrupt:
        _println(out, "\nAbgebrochen.")
        return 130
    parser.error(f"Unbekannter Unterbefehl {cmd!r}.")
    return 2


__all__ = [
    "ChatSession", "build_parser", "doctor", "format_status", "main", "recommend_config", "step_line",
    "cmd_chat", "cmd_config", "cmd_eval", "cmd_export_dataset", "cmd_memory", "cmd_modelfile", "cmd_serve",
    "cmd_train",
]
