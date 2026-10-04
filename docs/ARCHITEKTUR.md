# OBITO – Architektur-Spezifikation (KI-Kern)

Verbindlicher Vertrag für alle Module. Wer ein Modul implementiert, hält sich exakt an die
hier genannten Namen, Signaturen und Datenformate; andere Module werden parallel dagegen gebaut.

## Grundsätze

1. **100 % lokal.** Keine Cloud-KI. Modelle laufen über Ollama oder einen OpenAI-kompatiblen
   lokalen Server (llama.cpp, LM Studio, vLLM). Kein Netzwerk außer `127.0.0.1`.
2. **Nur Standardbibliothek** im Kern (`obito/*.py`). Python ≥ 3.10. Optionale schwere
   Abhängigkeiten (torch, transformers, peft) nur in `obito/training/train_lora.py`, dort
   lazy importiert mit klarer deutscher Fehlermeldung + Installationshinweis.
3. **Kein Spielzeug.** Echte Fehlerbehandlung, keine erfundenen Daten, jede Unsicherheit
   wird benannt. Alle Nutzer-Texte, Prompts, Docstrings, CLI-Hilfe: **Deutsch**.
   Bezeichner im Code: Englisch (wie in `memory.py`, `llm.py`, `config.py`).
4. **Testbar ohne Modell.** Alles wird mit `obito.llm.FakeBackend` getestet
   (`unittest`, Verzeichnis `tests/`, Dateien `tests/test_<modul>.py`, kein pytest nötig,
   kein Netzwerk, Datenbanken unter `tempfile`/`:memory:`).
5. **Die KI muss vom Nutzer verbesserbar sein**: Feedback → Lektionen (sofort wirksam über
   das Gedächtnis) → Trainingsdatensatz → LoRA-Training → eigenes Ollama-Modell → Evaluation.

## Warum „5-fach so schlau“

Eine Frage wird nicht von einem Modell beantwortet, sondern:

1. **Erinnern** – relevante Erinnerungen, Lektionen und Beispiele werden geladen.
2. **Routen** – Komplexität bestimmen, die 5 passendsten Experten wählen.
3. **Gremium** – 5 Experten antworten *parallel* aus ihrer Fachperspektive.
4. **Kritiker** – prüft alle Expertenantworten auf Fehler, Widersprüche, Lücken.
5. **Revision** – bemängelte Experten überarbeiten (max. `max_revision_rounds`).
6. **OMEGA-Synthese** – eine finale, geprüfte Antwort; Werkzeuge bei Bedarf.
7. **Lernen** – wichtige Fakten werden automatisch gemerkt, die Interaktion protokolliert.

Einfache Fragen (Routing: `einfach`) nehmen den schnellen Pfad (ein Aufruf mit Kontext).

---

## Bereits vorhanden (nicht ändern, nur nutzen)

### `obito/config.py`
`Config` (Dataclass) mit Feldern: `data_dir, backend, base_url, model, fast_model, embed_model,
depth ("auto"|"schnell"|"tief"), experts_per_question=5, max_revision_rounds=1, max_tool_rounds=4,
max_history=12, memory_recall=6, example_recall=3, temperature=0.4, timeout=300, allow_tools=True,
confirm_dangerous=True, auto_memory=True, language="de", workspace=".", server_host, server_port=8765`.
Properties: `data_path, memory_db, learning_db, datasets_dir, models_dir, logs_dir, routing_model`;
`ensure_dirs()`, `to_dict()`. Funktionen: `load_config(path=None, env=None)`, `save_config(cfg, path)`,
`find_config_file(explicit=None)`.

### `obito/llm.py`
- `LLMBackend` mit `available()`, `chat(messages, *, model=None, temperature=None, max_tokens=None,
  json_mode=False, stop=None, stream: Callable[[str],None]|None=None, timeout=None) -> ChatResult`,
  `embed(texts, *, model=None) -> list[list[float]] | None`, `list_models() -> list[ModelInfo]`,
  `has_model(name)`, `pull(name, progress=None)`, `info() -> dict`, Attribute `name`, `default_model`,
  `default_embed_model`.
- `ChatResult(text, model, prompt_tokens, completion_tokens, duration, raw)`, `.total_tokens`.
- `ModelInfo(name, size, family, parameters, quantization)`, `.short()`.
- Fehler: `LLMError`, `BackendUnavailable`, `ModelNotFound`.
- `OllamaBackend`, `OpenAICompatBackend`, `FakeBackend(responder=None, responses=None, model=...,
  embed_dim=16, models=None, up=True)` mit `.push(*responses)`, `.calls` (Liste aller Aufrufe,
  jeweils `{"messages": [...], "model", "temperature", "max_tokens", "json_mode", "stop"}`).
  `FakeBackend.embed` liefert deterministische Wortsack-Vektoren (gleiche Wörter → ähnlich).
- `make_backend(cfg) -> LLMBackend`, `parse_json(text) -> Any|None` (robust, Code-Zäune, Prosa).

### `obito/memory.py`
- `MemoryStore(path, embedder=None)` – `embedder(texts) -> list[list[float]]|None`.
- `remember(content, kind="notiz", tags=(), project=None, source="nutzer", importance=0.5) -> Memory`
  (Duplikate werden zusammengeführt). `KINDS = ("fakt","praeferenz","entscheidung","loesung","fehler",
  "zusammenfassung","notiz")`.
- `search(query, k=6, project=None, min_score=0.05, touch=True) -> list[Memory]` (hybride Suche).
- `get(id)`, `forget(id) -> bool`, `update_importance(id, delta)`, `recent(limit, project)`, `count()`,
  `stats() -> dict`, `export(path) -> int`, `import_json(path) -> int`, `close()`.
- Verlauf: `add_message(session_id, role, content, project=None)`, `history(session_id, limit=12)
  -> list[{"role","content"}]`.
- `Memory` (Dataclass: `id, kind, content, tags, project, source, importance, created_at,
  last_access, access_count, score`) mit `.short()`.

---

## Zu implementierende Module

### `obito/tools.py` – Werkzeuge

```python
@dataclass
class ToolResult:
    ok: bool
    output: str                 # für das Modell lesbar, max. ~4000 Zeichen (gekürzt mit Hinweis)
    error: str | None = None

@dataclass
class Tool:
    name: str                   # z. B. "rechnen"
    description: str            # deutsch, 1–2 Sätze
    parameters: dict            # JSON-Schema {"type":"object","properties":{...},"required":[...]}
    fn: Callable[..., str]      # fn(**args) -> str; Fehler -> Exception
    dangerous: bool = False     # erfordert Bestätigung

ConfirmCallback = Callable[[str, dict], bool]   # (tool_name, args) -> True = erlaubt

class ToolRegistry:
    def __init__(self, workspace: str = ".", confirm: ConfirmCallback | None = None,
                 confirm_dangerous: bool = True): ...
    def register(self, tool: Tool) -> None
    def unregister(self, name: str) -> None
    def get(self, name) -> Tool | None
    def list(self) -> list[Tool]
    def describe(self) -> str          # Prompt-Block (deutsch) mit allen Werkzeugen + Aufrufformat
    def run(self, name: str, args: dict) -> ToolResult
        # unbekanntes Werkzeug -> ok=False; gefährlich & keine Freigabe -> ok=False,
        # error="Vom Nutzer abgelehnt"; validiert required-Parameter; fängt alle Exceptions
    def set_memory(self, memory: "MemoryStore | None") -> None   # aktiviert Gedächtnis-Werkzeuge

def default_registry(workspace=".", confirm=None, confirm_dangerous=True) -> ToolRegistry

# Aufruf-Protokoll (Modell -> Brain):
TOOL_CALL_RE  # findet  <werkzeug>{"name": "...", "args": {...}}</werkzeug>
def parse_tool_calls(text: str) -> list[dict]         # [{"name":..., "args":{...}}], robust (parse_json)
def strip_tool_calls(text: str) -> str                 # Antworttext ohne die Aufruf-Blöcke
```

Eingebaute Werkzeuge (Namen genau so): `rechnen(ausdruck)` – sicherer Rechner via `ast`
(nur Zahlen, + - * / ** % //, Klammern, `math`-Funktionen; kein eval auf Namen);
`zeit()` – Datum/Uhrzeit; `system_info()` – OS, Python, CPU-Anzahl, RAM falls ermittelbar, Festplatte;
`verzeichnis(pfad=".")`, `datei_lesen(pfad, max_zeichen=4000)` – nur innerhalb `workspace`
(Pfad-Traversal verhindern, `realpath` prüfen); `datei_schreiben(pfad, inhalt)` – **dangerous**;
`python_ausfuehren(code, timeout=20)` – **dangerous**, `subprocess` mit `sys.executable`, Timeout,
Ausgabe begrenzt; `befehl_ausfuehren(befehl, timeout=30)` – **dangerous**, Shell im Workspace;
`gedaechtnis_suchen(frage)`, `gedaechtnis_merken(inhalt, art="notiz", wichtigkeit=0.6)` – nur wenn
`set_memory` gesetzt. `describe()` beschreibt das Format:

```
Wenn du ein Werkzeug brauchst, schreibe genau einen Block:
<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>
Danach wartest du auf das Ergebnis. Erfinde nie Ergebnisse.
```

### `obito/agents.py` – Experten & Prompts

```python
@dataclass(frozen=True)
class Expert:
    id: str          # "ALPHA" … "OMEGA", "KRITIKER", "GENERALIST"
    name: str        # "3D-Konstruktion (CAD)"
    role: str        # Kurzbeschreibung
    system_prompt: str
    keywords: tuple[str, ...]   # für die heuristische Zuordnung (Kleinbuchstaben)
    temperature: float = 0.4

EXPERTS: dict[str, Expert]   # ALPHA 3D-Konstruktion/CAD/Mechanik, BETA Elektronik & Steuerung,
    # GAMMA Physik & Simulation, DELTA Code-Generierung & Software, EPSILON Datenanalyse,
    # ZETA Vision & Erkennung/Sensorik, THETA Optimierung, IOTA Materialien, KAPPA Recherche & Fakten,
    # LAMBDA Planung & Projektmanagement, GENERALIST (Allgemeinwissen, Alltag, Sprache)
CRITIC: Expert               # KRITIKER
OMEGA: Expert                # Koordination & Synthese (auch Persona des schnellen Pfads)

BASE_RULES: str   # gemeinsame Regeln: Deutsch, konkret, Zahlen mit Einheiten, keine erfundenen Fakten,
                  # Unsicherheit markieren ("Unsicher:"), Nutzerkontext/Erinnerungen berücksichtigen,
                  # bei fehlenden Infos gezielt nachfragen statt raten, Sicherheit (Akku/Strom/Werkzeuge)

def select_experts(question: str, k: int = 5, hint: list[str] | None = None) -> list[Expert]
    # hint = Expert-IDs aus dem Routing (werden zuerst genommen, unbekannte ignoriert),
    # dann Keyword-Treffer, dann Auffüllen mit sinnvoller Reihenfolge (GENERALIST, KAPPA, DELTA …)

def heuristic_complexity(question: str) -> str   # "einfach" | "mittel" | "komplex" (Länge, Fachwörter,
                                                 # Mehrfachfragen, "warum/wie/vergleiche/entwirf/berechne")

# Prompt-Bausteine (alle liefern str; Kontext-Block = Erinnerungen + Lektionen + Beispiele + Projekt):
def context_block(memories: list, lessons: list, examples: list, project: str | None, extra: str = "") -> str
def routing_prompt(question: str, expert_ids: list[str]) -> list[dict]       # -> messages, json_mode
    # erwartetes JSON: {"komplexitaet":"einfach|mittel|komplex","experten":["ALPHA",...],
    #                   "werkzeuge": true|false, "begruendung":"..."}
def expert_messages(expert: Expert, question: str, context: str, history: list[dict]) -> list[dict]
def critic_messages(question: str, answers: dict[str, str], context: str) -> list[dict]  # json_mode
    # erwartetes JSON: {"bewertung": 1-10, "fehler":[{"experte":"ALPHA","problem":"...","korrektur":"..."}],
    #                   "widersprueche":["..."], "fehlt":["..."], "sicher": true|false}
def revision_messages(expert: Expert, question: str, previous: str, critique: list[dict]) -> list[dict]
def synthesis_messages(question: str, answers: dict[str, str], critique: dict, context: str,
                       history: list[dict], tools_block: str = "") -> list[dict]
def fast_messages(question: str, context: str, history: list[dict], tools_block: str = "") -> list[dict]
def tool_result_message(name: str, result: "ToolResult") -> dict     # {"role":"user","content":...}
def memory_extraction_messages(question: str, answer: str, project: str | None) -> list[dict]  # json_mode
    # erwartetes JSON: {"erinnerungen":[{"inhalt":"...","art":"fakt|praeferenz|entscheidung|loesung|fehler",
    #                   "wichtigkeit":0.0-1.0,"tags":["..."]}]}  – nur Dauerhaftes, keine Floskeln
```

### `obito/brain.py` – Denkkern

```python
@dataclass
class Step:                    # ein Schritt der Denk-Spur
    stage: str                 # "erinnern"|"routing"|"experte"|"kritiker"|"revision"|"synthese"|
                               # "werkzeug"|"schnell"|"lernen"
    who: str                   # Expert-ID, Werkzeugname, "system"
    summary: str               # kurz, für Anzeige
    detail: str = ""           # vollständiger Text/JSON
    duration: float = 0.0
    tokens: int = 0

@dataclass
class Answer:
    text: str
    question: str
    session_id: str
    project: str | None
    depth: str                  # "schnell" | "tief"
    experts: list[str]
    critique: dict | None
    memories_used: list[Memory]
    new_memories: list[Memory]
    tools_used: list[str]
    steps: list[Step]
    interaction_id: int | None
    tokens: int
    duration: float
    def trace(self) -> str      # lesbare deutsche Zusammenfassung der Schritte

ProgressCallback = Callable[[Step], None]

class Brain:
    def __init__(self, cfg: Config, backend: LLMBackend | None = None,
                 memory: MemoryStore | None = None, learning: "LearningStore | None" = None,
                 tools: ToolRegistry | None = None, confirm: ConfirmCallback | None = None)
        # None -> aus cfg erzeugen (make_backend, MemoryStore(cfg.memory_db, embedder), LearningStore,
        # default_registry). embedder = backend.embed nur wenn backend.available() und
        # has_model(cfg.embed_model), sonst None. tools.set_memory(memory).
    def ask(self, question: str, *, session_id: str = "standard", project: str | None = None,
            depth: str | None = None, stream: StreamCallback | None = None,
            progress: ProgressCallback | None = None, learn: bool | None = None) -> Answer
    def feedback(self, interaction_id: int, rating: int, comment: str | None = None,
                 correction: str | None = None) -> list[Memory]   # speichert Bewertung + erzeugt Lektion(en)
    def remember(self, content, kind="notiz", tags=(), project=None, importance=0.6) -> Memory
    def forget(self, memory_id) -> bool
    def recall(self, query, k=None, project=None) -> list[Memory]
    def status(self) -> dict     # backend-info, verfügbar?, modelle, gedächtnis-stats, lern-stats
    def close(self) -> None
```

Ablauf `ask` (genau so, mit Fehlerbehandlung):
1. `learn = cfg.auto_memory if learn is None`. Verlauf: `memory.history(session_id, cfg.max_history)`.
2. Erinnern: `memory.search(question, cfg.memory_recall, project)`, Lektionen =
   `learning.lessons(question, k=3)`, Beispiele = `learning.examples(question, cfg.example_recall)`.
   Step "erinnern".
3. Tiefe: `depth or cfg.depth`. Bei `"auto"`: Routing-Aufruf mit `cfg.routing_model`, `json_mode=True`,
   `temperature=0`; JSON via `parse_json`; bei Fehler/ungültig → `heuristic_complexity`. `einfach` →
   schnell, sonst tief. `"schnell"`/`"tief"` erzwingen. Step "routing".
4. Schneller Pfad: `fast_messages` (+ `tools.describe()` falls `allow_tools`) → `backend.chat`
   (streamt direkt an `stream`, wenn kein Werkzeug-Block erkannt wird – Streaming-Text also puffern,
   erst nach Abschluss an `stream` weitergeben, wenn Werkzeuge erlaubt sind; sonst direkt streamen).
5. Tiefer Pfad: Experten = `select_experts(question, cfg.experts_per_question, hint)`;
   `ThreadPoolExecutor` → `expert_messages` je Experte (Step "experte" je Antwort; Fehler eines Experten
   = Step mit Fehlermeldung, nicht Abbruch). Kritiker (`json_mode`), Step "kritiker". Wenn
   `bewertung < 6` oder `fehler` vorhanden und `max_revision_rounds > 0`: betroffene Experten
   überarbeiten (parallel), Step "revision". Synthese (`synthesis_messages` + Werkzeuge) mit Streaming
   wie in 4.
6. Werkzeugschleife (beide Pfade): `parse_tool_calls(text)`; je Aufruf `tools.run`, Step "werkzeug",
   `tool_result_message` anhängen, erneut `chat`; max `cfg.max_tool_rounds`; danach Text =
   `strip_tool_calls(text)`.
7. Verlauf speichern (`add_message` user+assistant). Wenn `learn`: `memory_extraction_messages`
   mit `routing_model`, `json_mode`; jede gültige Erinnerung (`inhalt` ≥ 12 Zeichen) →
   `memory.remember(..., source="ki", project=project)`; Step "lernen". Fehler hier nie fatal.
8. `learning.record(...)` → `interaction_id`. `Answer` zurück. Jeder Modellfehler
   (`LLMError`) auf dem Hauptpfad → `Answer` mit klarer deutscher Fehlermeldung als `text`
   (z. B. „Modell-Server nicht erreichbar – starte Ollama (`ollama serve`) …“), `depth="fehler"`.

### `obito/learning.py` – Lernen aus Feedback

SQLite (`cfg.learning_db`), eigene Verbindung, thread-sicher wie `memory.py`.

```python
@dataclass
class Interaction:
    id: int; session_id: str; question: str; answer: str; project: str | None
    experts: list[str]; depth: str; rating: int; comment: str | None; correction: str | None
    created_at: float; trace: str   # JSON der Steps (gekürzt)

class LearningStore:
    def __init__(self, path, embedder=None)
    def record(self, session_id, question, answer, *, project=None, experts=(), depth="", trace="") -> int
    def rate(self, interaction_id, rating: int, comment: str | None = None) -> Interaction  # -1/0/1
    def correct(self, interaction_id, correction: str) -> Interaction
    def get(self, interaction_id) -> Interaction | None
    def last(self, session_id=None, n=1) -> list[Interaction]
    def lessons(self, query, k=3) -> list[str]
        # Texte aus negativ bewerteten/korrigierten Interaktionen, ähnlich zur Frage (FTS):
        # "Frage: … | Problem: <kommentar> | Besser: <korrektur>"
    def examples(self, query, k=3) -> list[tuple[str, str]]   # (frage, beste_antwort) aus rating>=1 oder
                                                              # korrigiert, ähnlich zur Frage (FTS + Vektor)
    def export_dataset(self, path, *, fmt="chat"|"alpaca"|"dpo", min_rating=1, system_prompt=None) -> int
        # chat:  {"messages":[{"role":"system",…},{"role":"user",…},{"role":"assistant",…}]}
        # alpaca:{"instruction","input":"","output"}
        # dpo:   {"prompt","chosen","rejected"} – nur Interaktionen mit Korrektur (chosen=Korrektur,
        #        rejected=ursprüngliche Antwort)
        # Zeilen = JSONL; Rückgabe = Anzahl
    def stats(self) -> dict   # interaktionen, bewertet, positiv, negativ, korrigiert
    def close(self)
```

### `obito/training/` – Verbesserungs-Pipeline

- `__init__.py` leer.
- `modelfile.py`: `build_modelfile(base: str, system_prompt: str, *, adapter: str | None = None,
  temperature=0.4, num_ctx=8192, extra_params: dict | None = None) -> str` (Ollama-Modelfile-Text,
  System-Prompt korrekt in `"""` gequotet), `create_model(name, modelfile_text, *, ollama_bin="ollama",
  cwd=None) -> subprocess.CompletedProcess` (schreibt `Modelfile` in `cwd`/Temp, ruft
  `ollama create name -f Modelfile`), `default_system_prompt() -> str` (OMEGA-Persona + BASE_RULES aus
  `obito.agents`).
- `evaluate.py`: Datensatzformat JSONL `{"frage":…, "erwartet":… (optional), "stichworte":[…] (optional)}`.
  `run_eval(backend, model, items, *, judge_model=None, system_prompt=None, progress=None) -> dict`
  mit `{"modell", "anzahl", "stichwort_score" (0–1, Anteil gefundener Stichworte), "richter_score"
  (0–10 Mittel, None ohne Richter), "ergebnisse":[{"frage","antwort","stichworte_gefunden",
  "richter":{"punkte","begruendung"}}]}`. Richter = `backend.chat(json_mode)` mit Frage, erwartete
  Antwort, gegebene Antwort → `{"punkte":0-10,"begruendung":"…"}`. `load_items(path)`, `save_report(report, path)`.
  Deterministischer Teil muss ohne Richter funktionieren.
- `train_lora.py`: `main(argv=None)` mit argparse (deutsch): `--basis` (HF-Modell-ID oder Pfad),
  `--daten` (JSONL chat-Format), `--ausgabe`, `--epochen 3`, `--lr 2e-4`, `--rang 16`, `--alpha 32`,
  `--max-laenge 2048`, `--qlora` (4-bit, falls bitsandbytes), `--zusammenfuehren` (Adapter mergen),
  `--batch 1`, `--grad-akkum 8`. Lazy-Import von `torch/transformers/peft/datasets`; fehlt etwas →
  `SystemExit` mit deutscher Meldung und `pip install …`. Tokenisierung über
  `tokenizer.apply_chat_template`, Labels der Prompt-Tokens = -100. Training mit `transformers.Trainer`.
  Am Ende Hinweise: GGUF-Konvertierung (`convert_hf_to_gguf.py` aus llama.cpp) und
  `python -m obito modelfile --basis … --adapter …`. `--help` muss ohne Abhängigkeiten funktionieren.
  Funktionen `load_jsonl(path)`, `build_example(tokenizer, messages, max_len)` separat testbar
  (Tokenizer-Attrappe im Test).

### `obito/cli.py` + `obito/__main__.py`

`python -m obito [unterbefehl]`; ohne Unterbefehl = `chat`. Unterbefehle (argparse, deutsch):
- `chat [--projekt P] [--tiefe auto|schnell|tief] [--modell M] [--sitzung S]` – REPL mit Streaming.
  Befehle (Präfix `/`): `/hilfe`, `/merk <text>`, `/vergiss <id>`, `/suche <frage>`, `/erinnerungen [n]`,
  `/projekt [name]`, `/tief`, `/schnell`, `/auto`, `/modelle`, `/modell <name>`, `/gut [kommentar]`,
  `/schlecht [kommentar]`, `/korrektur <bessere antwort>`, `/spur` (letzte Denk-Spur), `/werkzeuge`,
  `/status`, `/export [pfad]` (Datensatz), `/beenden`. Unbekannter Befehl → Hinweis.
  Ausgabe während des Denkens: Progress-Zeilen je Step (z. B. `⟳ ALPHA denkt…`, `✓ KRITIKER: 8/10`).
  Nach Antwort: `💾 gemerkt: …` je neuer Erinnerung. Bestätigung gefährlicher Werkzeuge: `[j/N]`.
- `doctor` – prüft Python, Backend-Erreichbarkeit, Modelle (Haupt/Embedding vorhanden? sonst
  `ollama pull …`-Vorschlag), Datenverzeichnis, Gedächtnis-Statistik, optionale Trainings-Abhängigkeiten.
  Exit-Code 0 wenn Chat möglich.
- `serve [--host] [--port]` – startet `obito.server`.
- `export-dataset --ausgabe PFAD [--format chat|alpaca|dpo] [--min-bewertung 1]`.
- `eval --datei FRAGEN.jsonl [--modell M] [--richter M] [--ausgabe bericht.json]`.
- `modelfile --name N --basis B [--adapter PFAD] [--erstellen]`.
- `train …` → `obito.training.train_lora.main`.
- `memory list|search|export|import …`.
- `config [--schreiben PFAD]` – zeigt wirksame Konfiguration / schreibt Vorlage.
Globale Optionen: `--config PFAD`, `--backend`, `--daten PFAD` (data_dir).
Die REPL-Logik liegt in einer Klasse `ChatSession(brain, cfg, out=sys.stdout, inp=input)` mit
`handle_line(line) -> bool` (False = beenden), damit sie ohne Terminal testbar ist.

### `obito/server.py` + `obito/static/index.html`

`ObitoServer(brain, host, port)` (stdlib `http.server`, `ThreadingHTTPServer`), `serve_forever()`,
`shutdown()`, Attribut `port` (für Port 0 in Tests). JSON-API (alle Antworten `{"ok": bool, …}`):
- `GET /api/status` → `brain.status()`
- `POST /api/frage` `{"frage", "sitzung"?, "projekt"?, "tiefe"?}` → `{"antwort", "interaktion_id",
  "experten", "tiefe", "erinnerungen_neu", "spur": [Step…]}`
- `POST /api/feedback` `{"interaktion_id", "bewertung", "kommentar"?, "korrektur"?}`
- `GET /api/erinnerungen?q=…&n=…` ; `POST /api/erinnerungen` `{"inhalt","art"?,"projekt"?,"wichtigkeit"?}`;
  `DELETE /api/erinnerungen/<id>`
- `GET /api/modelle`
- `GET /` → `static/index.html` (dunkles HUD-Design wie die OBITO-Konzeptbilder: Chat links,
  rechts Panels „Gedächtnis“, „Denk-Spur“ (Experten, Kritiker-Bewertung), „System“; reines
  HTML/CSS/JS ohne externe Ressourcen, deutsch).
Gefährliche Werkzeuge im Server: keine interaktive Bestätigung → `confirm` liefert `False`.
Nur `127.0.0.1` als Standard; Fehler → `{"ok": false, "fehler": "…"}` mit passendem HTTP-Status.

### Tests (`tests/`)
`unittest`, laufen mit `python -m unittest discover -s tests -v` in < 30 s, kein Netzwerk.
Pflicht: `test_memory.py`, `test_llm.py`, `test_config.py`, `test_tools.py`, `test_agents.py`,
`test_brain.py` (schneller Pfad, tiefer Pfad mit skriptetem FakeBackend-`responder`, Werkzeugschleife,
Lernen, Fehlerfall Backend down), `test_learning.py`, `test_training.py`, `test_cli.py`,
`test_server.py` (Port 0, `urllib` gegen 127.0.0.1).

### Dokumentation
`README.md` (deutsch): Was ist OBITO, Installation (Windows 11 + Ollama, `ollama pull qwen2.5:7b`,
`ollama pull nomic-embed-text`), Schnellstart, wie das Gremium funktioniert, **„So verbesserst du
deine KI“** (Feedback → Lektionen → Datensatz → LoRA → Modelfile → Eval, mit Befehlen),
Konfiguration, API, Projektstruktur, Grenzen (Hardware, Modellqualität), Fahrplan (native HUD-App).
