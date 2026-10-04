# OBITO – Architektur-Spezifikation (KI-Kern)

Verbindlicher Vertrag für alle Module. Wer ein Modul implementiert, hält sich exakt an die
hier genannten Namen, Signaturen, Datenformate und Verhaltensregeln; andere Module werden
parallel dagegen gebaut. Abweichungen sind Integrationsfehler.

## Grundsätze

1. **100 % lokal.** Keine Cloud-KI. Modelle laufen über Ollama oder einen OpenAI-kompatiblen
   lokalen Server (llama.cpp, LM Studio, vLLM). Kein Netzwerk außer `127.0.0.1`.
2. **Nur Standardbibliothek** im Kern (`obito/*.py`). Python ≥ 3.10. Optionale schwere
   Abhängigkeiten (torch, transformers, peft, trl) nur in `obito/training/train_lora.py`, dort
   lazy importiert mit klarer deutscher Fehlermeldung + Installationshinweis.
3. **Kein Spielzeug.** Echte Fehlerbehandlung, keine erfundenen Daten, jede Unsicherheit
   wird benannt. Alle Nutzer-Texte, Prompts, Docstrings, CLI-Hilfe: **Deutsch**.
   Bezeichner im Code: Englisch (wie in `memory.py`, `llm.py`, `config.py`).
4. **Realität: 3B–14B-Modelle, 8–16 GB VRAM.** Kleine Modelle liefern unzuverlässiges JSON,
   Ollama kappt zu lange Prompts stillschweigend, fünf „parallele“ Aufrufe laufen bei
   `OLLAMA_NUM_PARALLEL=1` nacheinander. Jede Stufe hat deshalb ein Zeichen-/Token-Budget,
   einen Normalisierer, genau einen Retry und einen definierten Fallback.
5. **Testbar ohne Modell.** Alles wird mit `obito.llm.FakeBackend` getestet
   (`unittest`, Verzeichnis `tests/`, Dateien `tests/test_<modul>.py`, kein pytest nötig,
   kein Netzwerk, Datenbanken unter `tempfile`/`:memory:`).
6. **Die KI muss vom Nutzer verbesserbar sein** – in Stufen, die sofort wirken:
   Feedback → Lektionen (sofort im Kontext) → Beispiele (Few-Shot) → Eval-Set (messen) →
   Datensatz → LoRA → eigenes Ollama-Modell → Vergleichs-Eval (A/B).

## Warum „5-fach so schlau“

1. **Erinnern** – relevante Erinnerungen, Lektionen und Beispiele werden geladen.
2. **Routen** – Komplexität bestimmen (Heuristik, bei Bedarf Modell), passende Experten wählen.
3. **Gremium** – Experten antworten aus ihrer Fachperspektive (Stufe *mittel*: 3, *tief*: 5).
4. **Kritiker** (nur *tief*) – prüft alle Expertenantworten auf Fehler, Widersprüche, Lücken.
5. **Revision** (nur *tief*) – die am stärksten bemängelten Experten überarbeiten (max. 2).
6. **OMEGA-Synthese** – eine finale, geprüfte Antwort; Werkzeuge bei Bedarf; gestreamt.
7. **Lernen** – wichtige Fakten werden automatisch gemerkt, die Interaktion protokolliert,
   Feedback erzeugt Lektionen und justiert das Gedächtnis.

Einfache Fragen nehmen den schnellen Pfad (ein Aufruf mit Kontext). Stufen von `Answer.depth`:
`"schnell" | "mittel" | "tief" | "fehler"`.

---

## Bereits vorhanden (eingefroren – nur nutzen, nicht ändern)

### `obito/config.py`
`Config` (Dataclass) mit Feldern:
`data_dir, backend, base_url, model, fast_model, embed_model,
num_ctx=8192, keep_alive="30m", think="auto"|"an"|"aus", parallel_calls=2, timeout=300 (Sekunden ohne
Daten vom Modell), deadline=600 (Sekunden je Frage),
depth="auto"|"schnell"|"mittel"|"tief", experts_per_question=5, experts_medium=3,
max_revision_rounds=1, max_revised_experts=2, max_tool_rounds=4, max_tool_calls_per_answer=3,
max_history=12, memory_recall=6, example_recall=3, max_new_memories=3, temperature=0.4,
max_tokens_fast=1500, max_tokens_expert=900, max_tokens_critic=700, max_tokens_synthesis=1800,
max_tokens_json=500, allow_tools=True, confirm_dangerous=True, auto_memory=True, language="de",
workspace=".", server_host="127.0.0.1", server_port=8765`.
Properties: `data_path, memory_db, learning_db, datasets_dir, models_dir, logs_dir, routing_model`;
`ensure_dirs()`, `to_dict()`. Funktionen: `load_config(path=None, env=None)`, `save_config(cfg, path)`,
`find_config_file(explicit=None)`.

### `obito/llm.py`
- `LLMBackend.chat(messages, *, model=None, temperature=None, max_tokens=None,
  json_mode: bool | dict = False, stop=None, stream: Callable[[str],None] | None = None, timeout=None,
  num_ctx=None, seed=None, keep_alive=None, think: bool | None = None) -> ChatResult`.
  `json_mode=dict` = JSON-Schema (Ollama `format`, OpenAI `response_format json_schema`).
  `<think>…</think>`-Blöcke werden in allen Backends aus `text` entfernt und beim Streaming
  zurückgehalten (`ThinkFilter`). Beim Streaming ist `timeout` ein Leerlauf-Timeout.
- `embed(texts, *, model=None) -> list[list[float]] | None`, `list_models() -> list[ModelInfo]`,
  `has_model(name)`, `pull(name, progress=None)`, `running() -> list[dict]` (Ollama `/api/ps`:
  `name,size,size_vram,expires_at`), `version() -> str | None`, `info() -> dict`,
  `OllamaBackend.show(name) -> dict` (`capabilities`), Attribute `name`, `default_model`,
  `default_embed_model`, `base_url` (nicht bei Fake).
- `ChatResult(text, model, prompt_tokens, completion_tokens, duration, raw)`, `.total_tokens`.
- `ModelInfo(name, size, family, parameters, quantization)`, `.short()`, `.to_dict()`
  (`name, groesse, familie, parameter, quantisierung`).
- Fehler: `LLMError`, `BackendUnavailable`, `ModelNotFound`.
- `FakeBackend(responder=None, responses=None, model="fake-modell", embed_dim=16, models=None, up=True)`,
  `.push(*responses)`, `.calls` (je Aufruf `messages, model, temperature, max_tokens, json_mode, stop,
  num_ctx, seed, think`), streamt in 12-Zeichen-Stücken. `embed` = deterministische Wortsack-Vektoren.
- `make_backend(cfg)`, `parse_json(text) -> Any | None`, `strip_thinking(text)`, `ThinkFilter`.

### `obito/memory.py`
- `MemoryStore(path, embedder=None, embed_model="")`, `set_embedder(embedder, model_name)`,
  `reindex(batch_size=32, progress=None, only_missing=False) -> int`.
- `remember(content, kind="notiz", tags=(), project=None, source="nutzer", importance=0.5) -> Memory`
  (Duplikate werden zusammengeführt). `KINDS = ("fakt","praeferenz","entscheidung","loesung","fehler",
  "zusammenfassung","notiz")`.
- `search(query, k=6, project=None, min_score=0.05, touch=True, min_importance=0.0) -> list[Memory]`.
- `get(id)`, `forget(id) -> bool`, `update_importance(id, delta)`, `recent(limit, project)`, `count()`,
  `stats() -> {"erinnerungen","nach_art","projekte","nachrichten","mit_vektor","ohne_vektor",
  "embedding_modell","embedding_dim"}`, `export(path)`, `import_json(path)`, `close()` (idempotent).
- Verlauf: `add_message(session_id, role, content, project=None)`, `history(session_id, limit=12)
  -> list[{"role","content"}]`.
- `Memory` (Dataclass: `id, kind, content, tags, project, source, importance, created_at,
  last_access, access_count, score`) mit `.short()`. `normalize(text)`, `cosine(a, b)`.

---

## Gemeinsame Konventionen

### Stufen-Marker (Pflicht für deterministische Tests)
Jede von `agents.py` gebaute Nachrichtenliste beginnt mit einer System-Nachricht, deren
**letzte Zeile** exakt `[OBITO:<stage>:<who>]` ist. `stage` ∈ {`routing`, `schnell`, `experte`,
`kritiker`, `revision`, `synthese`, `extraktion`, `lektion`, `korrektur`, `richter`}, `who` =
Expert-ID oder `system`. `agents.stage_of(messages) -> tuple[str, str]` liest den Marker aus der
ersten System-Nachricht (Regex `\[OBITO:(\w+):([\w-]+)\]\s*$`), sonst `("?", "?")`.
Tests treiben den tiefen Pfad mit `FakeBackend(responder=lambda msgs, kw: SCRIPT[stage_of(msgs)])`.
Der Marker steht am **Ende** der System-Nachricht, damit der gemeinsame Präfix
(Regeln + Kontext) für alle Experten identisch bleibt (Prompt-Cache von llama.cpp/Ollama).

### Zeichen- und Token-Budget
`agents.py`: `clip(text, n) -> str` (kürzt auf n Zeichen + `" … [gekürzt]"`),
`estimate_tokens(text) -> int` (= `len(text) // 3 + 1`, konservativ für Deutsch),
`fit_messages(messages, budget_tokens, keep_last_user=True) -> list[dict]`: entfernt zuerst die
ältesten Verlaufs-/Beispiel-Nachrichten zwischen System-Nachricht und letzter Nutzer-Nachricht,
kürzt dann Assistant-Nachrichten im Verlauf auf 600 Zeichen, kürzt zuletzt die letzte
Nutzer-Nachricht (Kopf behalten). Konstanten: `MAX_CONTEXT_CHARS=3000` (Erinnerungen ≤ 1200,
Lektionen ≤ 800), `MAX_HISTORY_CHARS=3000`, `MAX_EXAMPLE_CHARS=400` (je Beispielantwort),
`MAX_ANSWER_IN_PROMPT=2500` (je Expertenantwort in Kritiker/Synthese), `MAX_CRITIQUE_CHARS=1500`.
Brain ruft vor **jedem** `chat` `fit_messages(messages, cfg.num_ctx - max_tokens_der_Stufe - 256)`
auf und übergibt `num_ctx=cfg.num_ctx`, `keep_alive=cfg.keep_alive`,
`think=None if cfg.think=="auto" else cfg.think=="an"`.

### JSON-Stufen (Routing, Kritiker, Extraktion, Lektion, Richter)
`agents.py` liefert je Stufe ein JSON-Schema (`ROUTING_SCHEMA`, `CRITIC_SCHEMA`, `MEMORY_SCHEMA`,
`LESSON_SCHEMA`, `JUDGE_SCHEMA`, `PAIRWISE_SCHEMA`) und einen Normalisierer (`parse_routing`,
`parse_critique`, `parse_memories`, `parse_lesson`, `parse_judge`, `parse_pairwise`), der
case-/umlaut-tolerant ist (`komplexität`→`komplexitaet`), Zahlen aus Strings zieht und klemmt,
Bools aus `ja/true/wahr/1`, Einzelstring → Liste, Experten-IDs `upper()` und gegen `EXPERTS`
(ID oder Name) auflöst, Unbekanntes verwirft und bei unbrauchbarem Inhalt `None` liefert.
Brain: `_json_call(stage, messages, schema, parser, *, model, max_tokens)`: Versuch 1 mit
`json_mode=schema, temperature=0`; bei `None` genau **ein** Retry mit angehängter Nutzer-Nachricht
„Antworte ausschließlich mit einem JSON-Objekt nach diesem Schema: …“ und `json_mode=True`;
danach Fallback (je Stufe definiert) und ein Step `who="system"` mit Hinweis.

### Werkzeug-Protokoll
Modell → Brain: `<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>`.
`parse_tool_calls` akzeptiert zusätzlich einen unverschlossenen Block am Textende, einen
```` ```json ````-Zaun mit Objekt `{"name","args"}` und ein nacktes JSON-Objekt mit genau diesen
Schlüsseln am Zeilenanfang. `name` muss str sein, `args` dict (sonst `{}`); Ungültiges → `[]`.
Brain → Modell: `tool_result_message(name, result)` liefert exakt
`{"role": "user", "content": "Ergebnis von Werkzeug »<name>«:\n<output>"}` bzw. bei `ok=False`
`"Fehler bei Werkzeug »<name>«: <error>\nAntworte ohne dieses Ergebnis oder korrigiere den Aufruf."`.
Werkzeuge nur in den Stufen `schnell` und `synthese`; nie bei Experten/Kritiker/Revision.

### Serialisierung (Server, CLI, Lernen nutzen genau diese Schlüssel)
- `Step.to_dict()` → `{"stufe","wer","zusammenfassung","detail","dauer","tokens","status"}`
- `brain.memory_to_dict(m)` → `{"id","art","inhalt","tags","projekt","quelle","wichtigkeit",
  "erstellt" (ISO-8601 lokal),"score"}` – **ohne** Embedding
- `Answer.to_dict()` → `{"antwort","frage","sitzung","projekt","tiefe","experten","kritik",
  "erinnerungen_genutzt","erinnerungen_neu","werkzeuge","spur","interaktion_id","tokens","dauer"}`
- `Answer.trace_json(max_detail=1000) -> str` = JSON-Liste der Steps mit gekürztem `detail`
  (genau dieser String geht in `learning.record(trace=…)`)
- `Interaction.to_dict()`, `Lesson.to_dict()` (Schlüssel = Feldnamen, deutsch wie unten)

---

## Zu implementierende Module

### `obito/tools.py` – Werkzeuge

```python
@dataclass
class ToolResult:
    ok: bool
    output: str                 # max. 4000 Zeichen, sonst Suffix "\n… [gekürzt, N Zeichen insgesamt]"
    error: str | None = None

@dataclass
class Tool:
    name: str; description: str; parameters: dict   # JSON-Schema {"type":"object","properties":…,"required":[…]}
    fn: Callable[..., str]; dangerous: bool = False

ConfirmCallback = Callable[[str, dict], bool]       # (tool_name, args) -> True = erlaubt

class ToolRegistry:
    def __init__(self, workspace: str = ".", confirm: ConfirmCallback | None = None,
                 confirm_dangerous: bool = True)      # self.workspace = os.path.realpath(workspace)
    def register(self, tool) / unregister(self, name) / get(self, name) -> Tool | None / list(self) -> list[Tool]
    def set_policy(self, confirm: ConfirmCallback | None, confirm_dangerous: bool) -> None
    def set_memory(self, memory: "MemoryStore | None") -> None    # aktiviert gedaechtnis_* Werkzeuge
    def resolve(self, pfad: str) -> str      # realpath innerhalb workspace, sonst PermissionError
    def describe(self, compact: bool = True) -> str   # kompakt: eine Zeile je Werkzeug + Aufrufformat
    def describe_one(self, name: str) -> str          # ausführlich mit Parametern
    def run(self, name: str, args: dict) -> ToolResult

def default_registry(workspace=".", confirm=None, confirm_dangerous=True) -> ToolRegistry
def parse_tool_calls(text: str) -> list[dict]     # [{"name":…, "args":{…}}]
def strip_tool_calls(text: str) -> str            # entfernt alle akzeptierten Formen
def needs_tools(question: str, history: list[dict]) -> bool
    # Regex: Zahlen+Operatoren, rechne|berechne|wie viel|datei|ordner|verzeichnis|uhrzeit|datum|heute|
    # system|ausführ|starte|speicher|lies|öffne … ODER im letzten Assistant-Turn kam ein Werkzeug vor

class ToolStreamFilter:
    """Streaming-Gate: reicht Text sofort weiter, hält nur ein mögliches Präfix von '<werkzeug'
    zurück; sobald der Marker vollständig ist, wird nichts mehr emittiert."""
    def __init__(self, emit: StreamCallback | None, marker: str = "<werkzeug")
    def feed(self, piece: str) -> None
    def finish(self) -> None            # leert den Rest nur, wenn kein Marker erkannt wurde
    tool_detected: bool; text: str      # gesamter Rohtext
```

`run`: (a) unbekannt → `ok=False, error="Unbekanntes Werkzeug »x«. Verfügbar: …"`; (b) Argumente nach
Schema-Typ koerzieren (`integer/number/boolean` aus Strings), unbekannte Schlüssel verwerfen;
(c) fehlende `required` → `error="Fehlende Parameter: …"`; (d) Freigabe exakt:
`erlaubt = (not tool.dangerous) or (not self.confirm_dangerous) or (self.confirm is not None and self.confirm(name, args))`,
sonst `error="Vom Nutzer abgelehnt"`; (e) jede Exception → `ok=False, error=str(e)`; (f) Ausgabe kürzen.

Eingebaute Werkzeuge (Namen genau so): `rechnen(ausdruck)` – `ast`-Whitelist: Expression, BinOp,
UnaryOp, Constant(int|float), Call mit `math.<name>`/`abs`/`round`/`min`/`max`, Namen `pi/e/tau`;
`**` nur mit |Exponent| ≤ 1000 und Operanden < 10**12; sonst `ValueError("Nicht erlaubt: …")`.
`zeit()`, `system_info()` (OS, Python, CPUs, RAM/Platte falls ermittelbar, keine Netzwerkaufrufe),
`verzeichnis(pfad=".")` (max. 200 Einträge, Ordner mit `/`), `datei_lesen(pfad, max_zeichen=4000)`
(1..20000, Dateien > 5 MB ablehnen, `errors="replace"`, Nullbytes → „Binärdatei“),
`datei_schreiben(pfad, inhalt)` **dangerous** (≤ 1 MB, Elternverzeichnisse via `resolve`, Rückgabe
„geschrieben: <relpfad> (N Bytes, überschrieben: ja/nein)“), `python_ausfuehren(code, timeout=20)`
**dangerous** (`[sys.executable, "-I", "-c", code]`, `cwd=workspace`, `timeout ≤ 120`,
`stdin=DEVNULL`, Minimal-Env: PATH, PYTHONIOENCODING, SYSTEMROOT/TEMP/TMP auf Windows; Ausgabe
`stdout + "\n[stderr]\n" + stderr + "\n[exit N]"`, Timeout → `error="Zeitlimit (N s) überschritten"`),
`befehl_ausfuehren(befehl, timeout=30)` **dangerous** (`shell=True`, timeout ≤ 300),
`gedaechtnis_suchen(frage)`, `gedaechtnis_merken(inhalt, art="notiz", wichtigkeit=0.6)` (`source="ki"`,
Wichtigkeit ≤ 0.8, `art` gegen `KINDS`) – nur mit `set_memory`.
Tests: Traversal (`../x`, absoluter Pfad, Symlink nach außen), Zeitlimit, Koerzion, Ablehnung,
Stream-Gate mit 12-Zeichen-Stücken („Ich rechne: <werkzeug>…“ → nur „Ich rechne: “ kommt an).

### `obito/agents.py` – Experten, Prompts, Normalisierer

```python
@dataclass(frozen=True)
class Expert:
    id: str; name: str; role: str; system_prompt: str; keywords: tuple[str, ...]; temperature: float = 0.4

EXPERTS: dict[str, Expert]   # ALPHA 3D-Konstruktion/CAD/Mechanik, BETA Elektronik & Steuerung,
    # GAMMA Physik & Simulation, DELTA Code & Software, EPSILON Datenanalyse, ZETA Vision & Sensorik,
    # THETA Optimierung, IOTA Materialien, KAPPA Recherche & Fakten, LAMBDA Planung & Projekte,
    # GENERALIST (Allgemeinwissen, Alltag, Sprache)
CRITIC: Expert               # KRITIKER
OMEGA: Expert                # Koordination & Synthese (auch Persona des schnellen Pfads)
BASE_RULES: str   # Deutsch; konkret; Zahlen mit Einheiten; keine erfundenen Fakten; "Unsicher:" markieren;
                  # Erinnerungen/Lektionen befolgen; bei fehlenden Infos gezielt nachfragen; Sicherheit
STAGE_TAG = "[OBITO:{stage}:{who}]"
def stage_of(messages) -> tuple[str, str]
def clip(text, n) -> str; def estimate_tokens(text) -> int
def fit_messages(messages, budget_tokens, keep_last_user=True) -> list[dict]

def select_experts(question, k=5, hint: list[str] | None = None) -> list[Expert]
    # hint-IDs zuerst (unbekannte ignorieren), dann Keyword-Treffer (Score), dann Auffüllen
    # in der Reihenfolge GENERALIST, KAPPA, DELTA, GAMMA, ALPHA, …; nie Duplikate, nie KRITIKER/OMEGA
def heuristic_complexity(question) -> str      # "einfach" | "mittel" | "komplex"
    # einfach: < 12 Wörter, keine Fachwörter, Smalltalk/Fakt; komplex: ≥ 2 Fragen, "entwirf|vergleiche|
    # berechne|optimiere|plane|analysiere|warum", lange Fragen, mehrere Fachgebiete; sonst mittel

def context_block(memories: list[Memory], lessons: list["Lesson"], project: str | None, extra="") -> str
    # Abschnitte: "Projekt: …", "Verbindliche Lektionen aus früherem Feedback (befolgen):" (direkt nach den
    # Regeln, "- Regel (bestätigt 4×): …", je ≤ 300 Zeichen), "Erinnerungen über den Nutzer/Projekt:"
    # ("- [art] inhalt"), Budget MAX_CONTEXT_CHARS
def example_messages(examples: list[tuple[str, str]]) -> list[dict]   # Few-Shot user/assistant-Paare,
    # Antwort je ≤ MAX_EXAMPLE_CHARS; stehen zwischen System-Nachricht und Verlauf
def routing_messages(question, expert_ids) -> list[dict]
def expert_messages(expert, question, context, history) -> list[dict]
    # System = BASE_RULES + context + Marker (identischer Präfix für alle Experten!);
    # Persona am Anfang der Nutzer-Nachricht: "Du antwortest als Experte ALPHA – 3D-Konstruktion: <role>…"
    # Aufgabe: Analyse, konkrete Vorschläge mit Zahlen, Risiken, offene Fragen; kompakt (≤ ~400 Wörter)
def critic_messages(question, answers: dict[str, str], context) -> list[dict]
def revision_messages(expert, question, previous, critique_items: list[dict]) -> list[dict]
def synthesis_messages(question, answers, critique: dict | None, context, history, examples,
                       tools_block="", medium=False) -> list[dict]
    # medium=True: "Benenne Widersprüche zwischen den Experten selbst." (kein Kritiker)
def fast_messages(question, context, history, examples, tools_block="") -> list[dict]
def tool_result_message(name, result: ToolResult) -> dict
def memory_extraction_messages(question, answer, project) -> list[dict]
def lesson_extraction_messages(question, answer, comment, correction) -> list[dict]
def correction_rewrite_messages(question, answer, correction) -> list[dict]   # Stufe "korrektur":
    # ursprüngliche Antwort so umschreiben, dass die Korrektur eingearbeitet ist (vollständige Antwort)
def judge_messages(frage, erwartet, antwort) -> list[dict]                   # {"punkte":0-10,"begruendung"}
def pairwise_judge_messages(frage, erwartet, antwort_1, antwort_2) -> list[dict]  # {"besser":1|2|0,"begruendung"}

ROUTING_SCHEMA  # {"komplexitaet":"einfach|mittel|komplex","experten":[IDs],"werkzeuge":bool,"begruendung":str}
CRITIC_SCHEMA   # {"bewertung":1-10,"fehler":[{"experte","problem","korrektur","schwere":"hoch|mittel|niedrig"}],
                #  "widersprueche":[str],"fehlt":[str],"sicher":bool}
MEMORY_SCHEMA   # {"erinnerungen":[{"inhalt","art","wichtigkeit":0-1,"tags":[str]}]}
LESSON_SCHEMA   # {"regel":str,"gilt_fuer":[str],"allgemein":bool}
JUDGE_SCHEMA, PAIRWISE_SCHEMA
def parse_routing(text, known_ids) -> dict | None
def parse_critique(text, expert_ids) -> dict | None      # Schlüssel genau wie CRITIC_SCHEMA, fehlende ergänzt
def parse_memories(text) -> list[dict]                   # nur gültige Einträge
def parse_lesson(text) -> dict | None
def parse_judge(text) -> dict | None; def parse_pairwise(text) -> dict | None
```

Alle Prompts deutsch, knapp, mit expliziter Ausgabe-Anweisung; JSON-Stufen enden mit
„Antworte nur mit JSON.“ und einem Beispielobjekt.

### `obito/brain.py` – Denkkern

```python
@dataclass
class Step:
    stage: str      # erinnern|routing|experte|kritiker|revision|synthese|werkzeug|schnell|lernen|system
    who: str; summary: str; detail: str = ""; duration: float = 0.0; tokens: int = 0
    status: str = "fertig"     # "start" | "fertig" | "fehler"
    def to_dict(self) -> dict
# summary-Formate: experte → "ALPHA (3D-Konstruktion): 812 Zeichen"; kritiker → "Bewertung 8/10, 2 Fehler,
# 1 Widerspruch"; routing → "komplex → tief (ALPHA, BETA, …)"; werkzeug → "rechnen: ok" / "rechnen: Fehler – …";
# lernen → "2 Erinnerungen gemerkt" / "übersprungen: …"

@dataclass
class Answer:
    text: str; question: str; session_id: str; project: str | None; depth: str
    experts: list[str]; critique: dict | None          # Kritiker-JSON + "revidiert": [IDs]
    memories_used: list[Memory]; new_memories: list[Memory]; lessons_used: list["Lesson"]
    tools_used: list[str]; steps: list[Step]; interaction_id: int | None; tokens: int; duration: float
    def trace(self) -> str; def to_dict(self) -> dict; def trace_json(self, max_detail=1000) -> str

@dataclass
class Feedback:
    interaction: "Interaction"; lessons: list["Lesson"]; memories_adjusted: int; correction_full: str | None

ProgressCallback = Callable[[Step], None]
def memory_to_dict(m: Memory) -> dict

class Brain:
    def __init__(self, cfg, backend=None, memory=None, learning=None, tools=None, confirm=None)
        # None → aus cfg erzeugen. Embedder: wenn backend.name != "fake" oder Fake mit up:
        # memory.set_embedder(lambda texts: backend.embed(texts, model=cfg.embed_model), cfg.embed_model)
        # (Backend liefert bei fehlendem Modell/Server None → Textsuche). tools.set_memory(memory).
    def ask(self, question, *, session_id="standard", project=None, depth=None, stream=None,
            progress=None, learn=None, cancel: threading.Event | None = None) -> Answer
    def feedback(self, interaction_id, rating: int | None = None, comment=None, correction=None,
                 correction_full: str | None = None) -> Feedback
    def rewrite_correction(self, interaction_id, correction) -> str     # Stufe "korrektur"; Fehler → correction
    def remember(self, content, kind="notiz", tags=(), project=None, importance=0.6) -> Memory
    def forget(self, memory_id) -> bool
    def recall(self, query, k=None, project=None) -> list[Memory]      # min_importance=0.15
    def set_model(self, name, *, fast=False) -> None   # ModelNotFound wenn verfügbar & nicht installiert;
                                                        # setzt cfg.model/fast_model UND backend.default_model
    def models(self) -> list[ModelInfo]                 # leer ohne Fehler, wenn Backend nicht erreichbar
    def reindex_memories(self, progress=None) -> int
    def export_dataset(self, path, **kw) -> dict        # Delegation an learning.export_dataset
    busy: bool                                          # property: self._ask_lock.locked()
    def status(self) -> dict   # {"backend","verfuegbar","modell","routing_modell","embedding_aktiv",
        # "modelle":[str],"gedaechtnis":memory.stats(),"lernen":learning.stats(),"werkzeuge":[str],
        # "beschaeftigt","tiefe":cfg.depth,"datenverzeichnis":str,"vektoren":{"mit","ohne","modell","dim"}}
    def close(self) -> None    # idempotent
```

Ablauf `ask` (verbindlich):
1. `with self._ask_lock` für den gesamten Durchlauf (`busy`). `progress`/`stream` werden
   **ausschließlich im aufrufenden Thread** gerufen. Vor jedem Modellaufruf ein Step mit
   `status="start"` an `progress` (nicht in `Answer.steps`), danach `fertig`/`fehler` (in `steps`).
   `cancel` wird zwischen Stufen und in Stream-Callbacks geprüft → `LLMError("Abgebrochen")`.
2. Erinnern: `history = memory.history(session_id, cfg.max_history)`;
   `memories = memory.search(question, cfg.memory_recall, project, min_importance=0.15)`;
   `lessons = learning.lessons(question, k=3)`; `examples = learning.examples(question, cfg.example_recall)`.
   Step `erinnern`.
3. Tiefe: `depth or cfg.depth`. Bei `auto`: `h = heuristic_complexity(question)`; LLM-Routing
   (`_json_call("routing", …, model=cfg.routing_model, max_tokens=cfg.max_tokens_json)`) **nur** wenn
   `h != "einfach"` oder `cfg.fast_model` gesetzt (einfache Fragen sparen den Aufruf; bei mittleren
   und komplexen Fragen entscheidet die Expertenwahl über die Qualität); Fallback = Heuristik. einfach→`schnell`,
   mittel→`mittel`, komplex→`tief`. `hint` = Experten aus dem Routing. Step `routing`.
4. Werkzeugblock: nur wenn `cfg.allow_tools` und (`needs_tools(question, history)` oder
   Routing `werkzeuge=True`) → `tools.describe(compact=True)`.
5. Schnell: `fast_messages(...)` → `_tool_loop` (Stufe `schnell`, `max_tokens_fast`).
6. Mittel/Tief: `experts = select_experts(question, cfg.experts_medium bzw. experts_per_question, hint)`;
   `ThreadPoolExecutor(max_workers=min(len(experts), cfg.parallel_calls))`, Aufrufe **gestreamt** mit
   internem Zähler (Token-Fortschritt), `as_completed(futures, timeout=verbleibende deadline)`;
   Exception/Timeout eines Experten → Step `experte` status `fehler`; `answers` enthält nur
   erfolgreiche; `len(answers)==0` → Fehler-Answer. **Tief**: Kritiker via `_json_call`
   (Fallback: `critique = {"bewertung": None, "fehler": [], "widersprueche": [], "fehlt": [],
   "sicher": False, "roh": text[:2000]}`, keine Revision). Revision wenn `max_revision_rounds > 0`:
   Experten mit `schwere=="hoch"` ODER (alle genannten, wenn `bewertung <= 4`), sortiert nach
   Anzahl Fehler, max. `cfg.max_revised_experts`, parallel wie Experten, gescheiterte Revision
   behält Original; `critique["revidiert"] = sorted(ids)`. Synthese: `synthesis_messages(...,
   medium=(depth=="mittel"))` → `_tool_loop` (Stufe `synthese`, `max_tokens_synthesis`).
7. `_tool_loop(messages, *, stage, max_tokens, stream, steps, tools_used) -> str`:
   ```
   parts = []
   for runde in range(cfg.max_tool_rounds + 1):
       gate = ToolStreamFilter(stream); res = chat(messages, stream=gate.feed, ...); gate.finish()
       calls = parse_tool_calls(res.text) if tools_block else []
       parts.append(strip_tool_calls(res.text).strip())
       if not calls: break
       if runde == cfg.max_tool_rounds:
           parts.append("Hinweis: Werkzeug-Limit erreicht – Antwort ohne weiteres Werkzeug."); break
       messages.append({"role": "assistant", "content": res.text})
       for call in dedupliziert(calls)[:cfg.max_tool_calls_per_answer]:
           result = tools.run(call["name"], call.get("args") or {})
           steps.append(Step("werkzeug", name, "rechnen: ok"/"…: Fehler – …", detail=output)); tools_used.append(name)
           messages.append(tool_result_message(name, result))
   return "\n\n".join(p for p in parts if p)
   ```
   Invariante: gestreamter Text == `Answer.text` (bis auf Werkzeug-Blöcke in Zaun-Form).
8. Verlauf: `add_message(user)`, `add_message(assistant, text)`. Lernen (wenn `learn` und
   `len(text) >= 40`): `_json_call("extraktion", memory_extraction_messages, MEMORY_SCHEMA,
   parse_memories, model=cfg.routing_model)`; Schutz: max `cfg.max_new_memories`, `wichtigkeit >= 0.4`,
   `inhalt` 12–300 Zeichen, mindestens ein Inhaltswort (> 4 Zeichen) gemeinsam mit Frage oder
   Antwort, Floskel-Filter („nutzer hat gefragt“, „die frage war“), kein „Unsicher“, max. 5 Tags;
   `memory.remember(..., source="ki", project=project, importance=min(w, 0.5))`. Step `lernen`.
   Fehler hier nie fatal.
9. `learning.record(session_id, question, text, project=…, experts=…, depth=…, trace=trace_json,
   model=…, context=…, history=…, memories_used=[ids], lessons_used=[ids], tools_used=…, tokens=…,
   duration=…)` → `interaction_id`.
10. Fehler-Answer (jede `LLMError` auf dem Hauptpfad): `depth="fehler"`, `interaction_id=None`,
    **kein** `add_message`, **kein** `record`, bisherige Steps enthalten. Texte:
    `BackendUnavailable` → „Modell-Server nicht erreichbar – starte Ollama (`ollama serve`) oder
    prüfe base_url (<url>).“; `ModelNotFound` → „Modell »<name>« ist nicht installiert –
    `ollama pull <name>`.“; sonst „Modellfehler: <msg>“. `stream` erhält den Fehlertext **nicht**.

`feedback`: `ValueError` bei unbekannter ID. `rating` ∈ {-1, 0, 1, None}. Reihenfolge:
(a) `correction` gesetzt und `correction_full is None` und `len(correction) < 200` →
`correction_full = rewrite_correction(...)`; (b) `learning.rate/correct`; (c) Lektion **nur** bei
(`rating == -1` und `comment`) oder `correction`: `_json_call("lektion", …, LESSON_SCHEMA,
parse_lesson, model=cfg.routing_model)`, Fallback `{"regel": comment or "Besser: "+clip(correction,300),
"gilt_fuer": Top-FTS-Wörter der Frage, "allgemein": False}` → `learning.add_lesson(...)`;
bare `-1` ohne Text erzeugt **keine** Lektion; (d) Gedächtnis-Hygiene: `rating == 1` →
`update_importance(id, +0.10)` für `memories_used`; `rating == -1` → `update_importance(id, -0.15)`
für `memories_used` und `forget(id)` für `new_memories` dieser Interaktion mit `source == "ki"`
(IDs aus `Interaction.memories_used`/`new_memories`); `memories_adjusted` = Anzahl.

### `obito/learning.py` – Lernen aus Feedback

SQLite (`cfg.learning_db`), `check_same_thread=False`, `RLock`, WAL, FTS5 mit Triggern wie `memory.py`.

```python
@dataclass
class Interaction:
    id: int; session_id: str; question: str; answer: str; project: str | None; experts: list[str]
    depth: str; model: str; context: str; history: list[dict]; memories_used: list[int]
    new_memories: list[int]; lessons_used: list[int]; tools_used: list[str]; tokens: int; duration: float
    rating: int; comment: str | None; correction: str | None; correction_full: str | None
    trainable: bool; created_at: float; trace: str
    def to_dict(self) -> dict   # Schlüssel: id, sitzung, frage, antwort, projekt, experten, tiefe, modell,
        # bewertung, kommentar, korrektur, korrektur_voll, trainierbar, erstellt, werkzeuge, tokens, dauer

@dataclass
class Lesson:
    id: int; rule: str; scope: str          # "allgemein" | "thema"
    topics: list[str]; source_interaction: int | None; created_at: float
    helped: int; hurt: int; active: bool
    def to_dict(self) -> dict   # id, regel, bereich, themen, quelle, erstellt, geholfen, geschadet, aktiv
    def render(self) -> str     # "Regel (bestätigt 4×): …" / "Regel (umstritten 2/3): …", ≤ 300 Zeichen

class LearningStore:
    def __init__(self, path, embedder=None)
    def record(self, session_id, question, answer, *, project=None, experts=(), depth="", trace="",
               model="", context="", history=(), memories_used=(), new_memories=(), lessons_used=(),
               tools_used=(), tokens=0, duration=0.0) -> int
        # trainable = depth != "fehler" and not tools_used; bettet question ein (Fehler → NULL)
    def rate(self, interaction_id, rating: int, comment=None) -> Interaction   # rating ∉ {-1,0,1} → ValueError;
        # aktualisiert helped/hurt aller lessons_used; deaktiviert Lektion bei hurt >= 3 and hurt > 2*helped
    def correct(self, interaction_id, correction, correction_full=None) -> Interaction  # rating=-1 falls 0
    def get(self, interaction_id) -> Interaction | None
    def last(self, session_id=None, n=1) -> list[Interaction]
    def add_lesson(self, rule, *, scope="thema", topics=(), source_interaction=None) -> Lesson
        # Duplikat (normalize(rule) gleich) → vorhandene zurück, nicht doppelt anlegen
    def lessons(self, query, k=3) -> list[Lesson]
        # bis 2 aktive "allgemein" mit bestem (helped-hurt) + "thema"-Lektionen per FTS (rule+topics)
        # und Vektor (0.6 vec + 0.4 bm25, ohne Vektor nur bm25); nur active; insgesamt ≤ k
    def list_lessons(self, include_inactive=False) -> list[Lesson]
    def delete_lesson(self, lesson_id) -> bool
    def examples(self, query, k=3) -> list[tuple[str, str]]
        # (frage, beste_antwort) aus rating>=1 OR correction; beste = correction_full or correction or answer;
        # Rang 0.6·Cosinus + 0.4·BM25 (ohne Vektor BM25); nur trainable
    def export_dataset(self, path, *, fmt="chat", min_rating=1, system_prompt=None, eval_share=0.1,
                       seed=42, dedup=True, min_answer_chars=40, include_context=True,
                       max_own: int | None = None) -> dict
        # Auswahl: trainable AND (rating >= min_rating OR correction IS NOT NULL)
        # Ziel = correction_full or correction or answer (strip_tool_calls); korrigierte immer rein,
        # eigene positive Antworten höchstens max_own (Standard 3 × Anzahl Korrekturen, mind. 20)
        # Dedup-Schlüssel normalize(question)+sha1(normalize(ziel)) – neueste/korrigierte behalten
        # Split deterministisch per sha1(normalize(question)) → eval_share in <path>.eval.jsonl
        # Zusätzlich <path>.fragen.jsonl: {"frage","erwartet":ziel,"stichworte":[bis 8 Inhaltswörter ≥5 Zeichen],
        #   "quelle_id": id} für evaluate.py (aus dem Eval-Anteil, Fallback: alle)
        # chat:  {"messages":[system(system_prompt or default_system_prompt()), *history,
        #          user((context + "\n\n" if include_context and context else "") + question), assistant(ziel)]}
        # alpaca: {"instruction": question, "input": context if include_context else "", "output": ziel}
        # dpo:    nur mit Korrektur: {"prompt": question (mit context), "chosen": ziel, "rejected": answer}
        # Rückgabe {"train": n, "eval": n, "verworfen": {"duplikat","zu_kurz","werkzeug","fehler","eigene_limit"}}
        # Dateien werden auch bei 0 Zeilen (leer) erzeugt
    def stats(self) -> dict   # interaktionen, bewertet, positiv, negativ, korrigiert, trainierbar,
                              # lektionen, lektionen_aktiv
    def close(self)           # idempotent
```

### `obito/training/` – Verbesserungs-Pipeline

- `__init__.py` leer.
- `modelfile.py`:
  `BASE_MAP: dict[str, str]` (Ollama-Tag → HF-ID: `qwen2.5:1.5b/3b/7b/14b` → `Qwen/Qwen2.5-{…}B-Instruct`,
  `qwen2.5-coder:7b` → `Qwen/Qwen2.5-Coder-7B-Instruct`, `qwen3:8b/14b` → `Qwen/Qwen3-{8,14}B`,
  `llama3.1:8b` → `meta-llama/Llama-3.1-8B-Instruct`, `llama3.2:3b` → `meta-llama/Llama-3.2-3B-Instruct`,
  `mistral:7b` → `mistralai/Mistral-7B-Instruct-v0.3`, `gemma3:4b/12b` → `google/gemma-3-{4,12}b-it`,
  `phi3.5:3.8b` → `microsoft/Phi-3.5-mini-instruct`); `hf_base_for(ollama_tag) -> str | None`
  (auch ohne `:latest`, Tag-Varianten wie `qwen2.5:7b-instruct-q4_K_M` auf `qwen2.5:7b` abbilden),
  `ollama_base_for(hf_id) -> str | None`. `CHATML_TEMPLATE: str` (Qwen).
  `build_modelfile(base, system_prompt, *, adapter=None, template=None, temperature=0.4, num_ctx=8192,
  extra_params=None) -> str`: `adapter` muss auf `.gguf` enden, sonst (Verzeichnis) nur bei Basis-Familie
  llama/mistral/gemma, sonst `ValueError` mit Hinweis auf `convert_lora_to_gguf.py --base <hf> --outfile
  adapter.gguf`; ist `base` eine `.gguf`-Datei, ist `template` Pflicht (Standard ChatML wenn Name `qwen`
  enthält, sonst `ValueError`); System-Prompt korrekt in `"""` gequotet; `PARAMETER num_ctx`,
  `temperature`, `stop`-Token je Template. `create_model(name, modelfile_text, *, ollama_bin="ollama",
  cwd=None) -> subprocess.CompletedProcess` (schreibt `Modelfile` in `cwd` = Verzeichnis des Adapters/
  GGUF, sonst Temp; `ollama create name -f Modelfile`). `default_system_prompt() -> str`
  (OMEGA-Persona + BASE_RULES). `check_adapter_compat(adapter_path, ollama_base) -> str | None`
  (liest `obito_training.json` neben dem Adapter; Warntext bei abweichender Basis).
- `evaluate.py`: Items `{"frage", "erwartet"?, "stichworte"?: [...], "verboten"?: [...], "muss_zahl"?: "4.2",
  "quelle_id"?}`. `load_items(path)`, `save_report(report, path)`.
  `keyword_score(answer, item) -> float` (0–1; `normalize`, Präfix-Treffer für Wörter ≥ 5 Zeichen wie
  `_fts_query`; `verboten`-Treffer → 0; `muss_zahl` ±2 % Toleranz, sonst 0).
  `run_eval(backend, model, items, *, judge_model=None, system_prompt=None, temperature=0.0, seed=42,
  progress=None, through_brain=None, depth=None, exclude_ids=frozenset(), num_ctx=None) -> dict`
  → `{"modell","tiefe","anzahl","stichwort_score","richter_score" (None ohne Richter),"dauer_mittel",
  "tokens_mittel","warnungen":[…],"ergebnisse":[{"frage","antwort","stichwort","richter":{"punkte","begruendung"}|None,
  "dauer","tokens"}]}`; `judge_model == model` → Richter aus, Warnung „Richter = geprüftes Modell …“;
  `through_brain` → Antworten via `brain.ask(frage, session_id=f"eval-{i}", depth=depth, learn=False)`.
  `compare(report_a, report_b, *, backend=None, judge_model=None) -> dict`: je Item
  `{"frage","a","b","stichwort_a","stichwort_b","richter":"A"|"B"|"gleich"}` mit paarweisem Richter
  (zweimal, vertauschte Reihenfolge; nur konsistente Urteile zählen); Summe `{"siege_a","siege_b",
  "gleich","stichwort_delta","dauer_delta","tokens_delta","n","hinweis"}` (Hinweis bei `n < 30` oder
  `|siege_a-siege_b| <= sqrt(n)`: nicht signifikant). Deterministischer Teil läuft ohne Richter.
- `train_lora.py`: `main(argv=None)` (argparse, deutsch): `--basis` (HF-ID, Pfad **oder Ollama-Tag**,
  Standard `hf_base_for(cfg.model)`), `--daten` (JSONL chat), `--eval-daten` (Standard `<daten>.eval.jsonl`
  falls vorhanden), `--ausgabe`, `--epochen 3`, `--lr 1e-4`, `--rang 16`, `--alpha 32`, `--max-laenge 1024`,
  `--batch 1`, `--grad-akkum 8`, `--qlora` (automatisch wenn CUDA-VRAM < 20 GB und bitsandbytes da),
  `--zusammenfuehren`, `--dpo` (trl, ≥ 100 Paare, sonst Abbruch), `--erzwingen`.
  Lazy-Import; fehlende Pakete → `SystemExit` mit `pip install …`. Preflight vor dem Laden: GPU/VRAM
  (`torch.cuda.mem_get_info`), Bedarf (7B 4-bit ≈ 10 GB, 3B ≈ 6 GB, 1.5B ≈ 4 GB), CPU-only → Abbruch mit
  Empfehlung; < 20 Beispiele → Abbruch, < 100 → Warnung. `target_modules` = alle Linear-Projektionen,
  Gradient-Checkpointing, dtype auto (bf16 wenn unterstützt). `build_example(tokenizer, messages,
  max_len) -> dict | None`: `prompt_ids = apply_chat_template(messages[:-1], add_generation_prompt=True,
  tokenize=True)`, `full_ids = apply_chat_template(messages, tokenize=True)`, gemeinsamer Präfix,
  Labels `[-100]*len(prompt) + rest`, Kürzung von **links** (Verlauf), nie das Ziel; voll gekürztes
  Ziel → `None`. Training mit `transformers.Trainer`, `eval_strategy="epoch"`,
  `load_best_model_at_end=True`, Loss je Epoche deutsch ausgeben, Warnung bei steigendem Eval-Loss.
  Schreibt `<ausgabe>/obito_training.json` `{hf_base, ollama_base, n_train, n_eval, epochen, lr, rang,
  max_laenge, chat_template_hash}`. Ausgabe am Ende genau zwei Routen:
  (A) `convert_lora_to_gguf.py <ausgabe> --base <hf_base> --outfile adapter.gguf` →
  `python -m obito modelfile --name obito-v1 --basis <ollama_base> --adapter adapter.gguf --erstellen`;
  (B) `--zusammenfuehren` → `convert_hf_to_gguf.py <merged> --outfile merged.f16.gguf` →
  `llama-quantize … Q4_K_M` → `python -m obito modelfile --basis ./merged.q4_k_m.gguf`.
  `load_jsonl(path)`, `build_example`, `preflight(...)`, `--help` ohne Abhängigkeiten.

### `obito/cli.py` + `obito/__main__.py`

`python -m obito [unterbefehl]`; ohne Unterbefehl = `chat`. Globale Optionen `--config PFAD`,
`--backend`, `--daten PFAD`, `--modell M`. Unterbefehle (argparse, deutsch):
- `chat [--projekt P] [--tiefe auto|schnell|mittel|tief] [--sitzung S]` – REPL mit Streaming.
  Befehle: `/hilfe`, `/merk <text>`, `/vergiss <id>`, `/suche <frage>`, `/erinnerungen [n]`,
  `/projekt [name]`, `/tief`, `/mittel`, `/schnell`, `/auto`, `/modelle`, `/modell <name>`,
  `/gut [kommentar]`, `/schlecht [kommentar]` (ohne Text: Rückfrage „Was war falsch?“),
  `/korrektur <bessere antwort>` (zeigt `rewrite_correction`-Vorschlag, „Übernehmen? [J/n]“),
  `/lektionen [n]`, `/lektion-loeschen <id>`, `/spur`, `/werkzeuge`, `/status`, `/export [pfad]`,
  `/beenden`. Fortschritt je Step: `⟳ ALPHA denkt…`, `✓ KRITIKER: Bewertung 8/10 …`;
  nach Antwort `💾 gemerkt: …` (bei `source=="ki"`: `💾 gemerkt (unsicher): …`). Strg+C während des
  Denkens setzt `cancel`; zweites Strg+C beendet. Bestätigung gefährlicher Werkzeuge `[j/N]`.
  REPL-Logik in `ChatSession(brain, cfg, out=sys.stdout, inp=input)` mit `handle_line(line) -> bool`.
- `doctor [--training]` – Python, Backend (Version), Modelle (Haupt/Embedding; `ollama pull`-Vorschlag),
  Probeaufruf (`max_tokens=5`, misst Tok/s, Prognose je Stufe), `/api/ps`: `size_vram < size` → Warnung
  „läuft teilweise auf CPU“, `fast_model` nur sinnvoll, wenn beide geladen bleiben, Denk-Modell
  (`show().capabilities` enthält `thinking`) → Empfehlung `think: aus`, Hinweis `OLLAMA_NUM_PARALLEL`,
  `OLLAMA_KEEP_ALIVE`, `OLLAMA_FLASH_ATTENTION`, `OLLAMA_KV_CACHE_TYPE`; Datenverzeichnis;
  Gedächtnis (Vektoren fehlen → `memory reindex`); `--training`: torch/transformers/peft/trl, VRAM.
  Exit-Code 0 wenn Chat möglich.
- `serve [--host] [--port] [--gefaehrlich-erlauben]`.
- `export-dataset --ausgabe PFAD [--format chat|alpaca|dpo] [--min-bewertung 1] [--eval-anteil 0.1]
  [--ohne-kontext] [--max-eigene N]` – gibt die Rückgabe-Statistik aus.
- `eval --datei FRAGEN.jsonl [--modell M] [--richter M] [--tiefe schnell|mittel|tief] [--ausgabe bericht.json]
  [--ohne-training DATEN.jsonl]`; `eval --vergleich A.json B.json [--richter M]`.
- `modelfile --name N --basis B [--adapter PFAD] [--vorlage PFAD] [--erstellen] [--erzwingen]`.
- `train …` → `train_lora.main`.
- `memory list|search|export|import|reindex [--alle] …`.
- `config [--schreiben PFAD] [--empfehlen VRAM_GB]` – zeigt wirksame Konfiguration; `--empfehlen`
  schreibt eine Vorlage (8 GB: `qwen2.5:7b`, fast leer, num_ctx 8192, parallel 1; 12 GB: + `qwen2.5:3b`,
  num_ctx 12288, parallel 2; 16 GB: `qwen2.5:14b`, fast `qwen2.5:3b`; < 8 GB/CPU: `qwen2.5:3b`,
  depth `schnell`).

### `obito/server.py` + `obito/static/index.html`

`ObitoServer(brain, host="127.0.0.1", port=8765, *, allow_dangerous=False, max_body=1_048_576)`,
`ThreadingHTTPServer`, `serve_forever()`, `shutdown()`, Attribut `port`. Beim Start
`brain.tools.set_policy(confirm=lambda n, a: allow_dangerous, confirm_dangerous=True)`; Warnung auf
stderr bei Nicht-Loopback-Host. Jede Anfrage: `Host` ∈ {`127.0.0.1`, `localhost`, `[::1]`, host}[:port]
sonst 403; `Origin` (falls vorhanden) auf dieselben Hosts sonst 403; POST/DELETE nur
`Content-Type: application/json` sonst 415; `Content-Length > max_body` → 413; keine CORS-Header;
`GET /` liefert ausschließlich `static/index.html`; unbehandelte Exception → 500
`{"ok": false, "fehler": …}` + Traceback in `cfg.logs_dir/server.log`; `log_message` → Log-Datei.
Antworten `{"ok": bool, …}`:
- `GET /api/status` → `{"ok": true, **brain.status()}`
- `POST /api/frage` `{"frage","sitzung"?,"projekt"?,"tiefe"?,"stream"?: bool}` → `{"ok": true, **answer.to_dict()}`;
  `tiefe=="fehler"` → 503; mit `"stream": true` → `text/event-stream` (chunked, `flush()` je Event):
  `event: schritt` / `data: <Step.to_dict()>`, `event: token` / `data: {"text": …}`, abschließend
  `event: antwort` / `data: <Answer.to_dict()>` bzw. `event: fehler` / `data: {"fehler": …}`; `busy` → 409.
- `POST /api/feedback` `{"interaktion_id","bewertung"?,"kommentar"?,"korrektur"?,"korrektur_voll"?,
  "umschreiben"?: true}` → `{"ok": true, "interaktion": Interaction.to_dict(), "lektionen": [Lesson.to_dict()],
  "korrektur_vorschlag": str|null}`; unbekannte ID → 404.
- `GET /api/erinnerungen?q=&n=&projekt=` → `{"ok": true, "erinnerungen": [memory_to_dict…]}` (ohne `q`: `recent`);
  `POST /api/erinnerungen` `{"inhalt","art"?,"projekt"?,"wichtigkeit"?,"tags"?}` → `{"ok": true, "erinnerung": …}`;
  `DELETE /api/erinnerungen/<id>` → 404 wenn unbekannt.
- `GET /api/lektionen` → `{"ok": true, "lektionen": [...]}`; `DELETE /api/lektionen/<id>`.
- `GET /api/modelle` → `{"ok": true, "modelle": [ModelInfo.to_dict()…], "aktuell": cfg.model}`.
- ungültiges JSON / fehlende `frage` → 400.
`index.html`: dunkles HUD (OBITO-Konzept): Chat links mit Streaming (`fetch` + `ReadableStream`,
SSE-Events parsen), rechts Panels „Denk-Spur“ (live je `schritt`), „Gedächtnis“ (Suche, Hinzufügen,
Löschen), „Lektionen“, „System“ (Status, Modelle); Feedback-Knöpfe 👍/👎/Korrektur unter jeder Antwort;
reines HTML/CSS/JS ohne externe Ressourcen, deutsch.

### Tests (`tests/`)
`unittest`, `python -m unittest discover -s tests -v` in < 60 s, kein Netzwerk, kein Modell.
Pflicht: `test_tools.py`, `test_agents.py` (Prompts enthalten Marker, `stage_of`, Normalisierer mit
kaputten Eingaben, `fit_messages`, `select_experts`, `heuristic_complexity`), `test_brain.py`
(schneller Pfad; mittel und tief mit skriptetem `responder` über `stage_of`; Kritiker-Fallback bei
Müll-JSON; Revision nur bei `schwere: hoch`; Werkzeugschleife inkl. Limit und Ablehnung; Streaming-Gate;
Lernen mit Schutzfiltern; Feedback → Lektion + Gedächtnis-Hygiene; Fehlerfall Backend down; `cancel`),
`test_learning.py` (Schema, Lektionen helped/hurt/deaktivieren, examples-Rang, export mit Dedup/Split/
Dateien, dpo), `test_training.py` (modelfile inkl. Fehlerfälle, hf_base_for, keyword_score, run_eval
mit FakeBackend, compare, `build_example` mit Tokenizer-Attrappe, `train_lora --help`, Preflight-Logik
ohne torch), `test_cli.py` (`ChatSession.handle_line` für alle Befehle, Unterbefehle mit FakeBackend),
`test_server.py` (Port 0, `urllib` gegen 127.0.0.1: alle Endpunkte, Härtung 403/415/413/404/400,
Streaming-Events).

### Dokumentation
`README.md` (deutsch): Was ist OBITO; Installation (Windows 11 + Ollama, `ollama pull qwen2.5:7b`,
`ollama pull nomic-embed-text`, Umgebungsvariablen); Modellmatrix nach VRAM; Schnellstart; wie das
Gremium funktioniert (Stufen); **„So verbesserst du deine KI“** als Stufenleiter (1 Feedback/Lektionen
sofort · 2 Beispiele ab ~10 · 3 Eval-Set ab ~30 · 4 LoRA ab ~200 Beispielen und 12 GB VRAM/WSL2) mit
exakten Befehlen und der Regel „Adapter passt nur zum exakt gleichen Basismodell“; Konfiguration;
API; Projektstruktur; Grenzen; Fahrplan (native HUD-App).
