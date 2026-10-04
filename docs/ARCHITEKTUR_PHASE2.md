# OBITO – Architektur Phase 2 (Engineering-Plattform)

Ergänzt `docs/ARCHITEKTUR.md` (Phase 1: KI-Kern). Phase 1 ist implementiert und getestet; die
dort beschriebenen Module (`config`, `llm`, `memory`, `tools`, `agents`, `learning`, `brain`,
`training/*`, `cli`, `server`, `static/index.html`) sind die Basis. Phase 2 fügt hinzu:

| Modul | Zweck (aus dem OBITO-Konzept) |
|---|---|
| `obito/engineering.py` | Ingenieur-Rechner + Materialdatenbank als Werkzeuge |
| `obito/knowledge.py` | Datenzentrum: Dokumente indexieren, semantisch durchsuchen (RAG) |
| `obito/projects.py` | Projektsystem: Notizen, Entscheidungen, Aufgaben, Versionen, Dateien |
| `obito/missions.py` | Missionen: autonome Mehrschritt-Aufgaben mit Plan, Ausführung, Bericht |
| `obito/automation.py` | Automationen: Zeitplaner (Konsolidierung, Backup, Eval, Wissens-Sync) |
| Paketierung | `pyproject.toml`, `OBITO.bat`, `start_obito.sh`, Modell-Hub |
| Integration | Brain/Server/CLI/HUD verdrahten alles |

Alle Grundsätze aus Phase 1 gelten (lokal, stdlib, deutsch, testbar mit `FakeBackend`, kein
Spielzeug). **Modul-Agenten schreiben nur ihre eigenen Dateien**; Brain/Server/CLI/HUD werden
danach von einem Integrations-Agenten verdrahtet. Deshalb hat jedes Modul klar definierte
**Integrationspunkte** (unten je Modul).

Zusätzliche Konvention: Jedes Modul mit Datenbank nutzt dieselbe Technik wie `memory.py`
(`sqlite3`, `check_same_thread=False`, `RLock`, WAL, FTS5 mit Triggern, idempotentes `close()`),
Datei unter `cfg.data_path / "<name>.db"`. Alle `to_dict()` liefern deutsche Schlüssel.

---

## `obito/engineering.py` – Ingenieur-Rechner & Materialdaten

Reine Funktionen mit SI-Einheiten; jede Rechnung liefert ein `dict` mit Ergebniswerten,
`"formel"` (lesbar) und `"annahmen"` (Liste). Werte sind **Richtwerte** und als solche
gekennzeichnet; keine erfundene Präzision.

```python
@dataclass(frozen=True)
class Material:
    key: str; name: str; category: str        # "metall" | "kunststoff" | "verbund" | "holz" | "sonstiges"
    density: float                            # g/cm³ (typisch)
    tensile_mpa: tuple[float, float]          # Zugfestigkeit min/max
    youngs_gpa: float                         # E-Modul (typisch)
    max_temp_c: float | None                  # Dauergebrauchstemperatur
    cost: str                                 # "günstig" | "mittel" | "teuer"
    printable: str | None                     # "FDM" | "SLA" | None
    notes: str; typical_use: str
    def to_dict(self) -> dict

MATERIALS: dict[str, Material]   # mindestens: cfk, gfk, alu_6061, alu_7075, stahl_s235, edelstahl_1_4301,
    # titan_grade5, pla, petg, abs, asa, nylon_pa12, pa_cf, tpu, resin_standard, pom, pc, balsa,
    # birkensperrholz, kupfer, messing, magnesium_az31
def find_material(name: str) -> Material | None        # tolerant: "Carbon", "CFK", "Alu 7075", "PETG"
def compare_materials(names: list[str]) -> list[Material]
def material_table(materials: list[Material]) -> str   # ASCII-Tabelle (deutsch)

# Rechner (alle -> dict mit Ergebnis + "formel" + "annahmen"; ValueError bei unsinnigen Eingaben)
def lipo_energy(cells: int, mah: float) -> dict                 # 3,7 V/Zelle nominal, Wh
def flight_time(mah: float, cells: int, avg_current_a: float, usable: float = 0.8) -> dict  # Minuten
def required_thrust(mass_g: float, twr: float = 2.0, motors: int = 4) -> dict  # Gesamtschub, je Motor, N & g
def thrust_to_weight(mass_g: float, thrust_per_motor_g: float, motors: int = 4) -> dict
def ohm(u: float | None = None, i: float | None = None, r: float | None = None, p: float | None = None) -> dict
    # genau zwei Größen gegeben -> die anderen beiden
def voltage_divider(u_in: float, r1: float, r2: float) -> dict
def torque(lever_m: float, force_n: float | None = None, mass_kg: float | None = None, safety: float = 1.5) -> dict
def motor_rpm(kv: float, voltage: float, load_factor: float = 0.85) -> dict
def prop_static_thrust(diameter_in: float, pitch_in: float, rpm: float) -> dict
    # empirische Näherung (Staples): T[N] = 4.392e-8 * rpm * d^3.5 / sqrt(pitch) * (4.233e-4 * rpm * pitch);
    # "annahmen": grobe Schätzung ±30 %
def beam_cantilever(force_n: float, length_m: float, youngs_gpa: float, width_m: float, height_m: float) -> dict
    # Durchbiegung f = F L³ / (3 E I), I = b h³/12, Biegespannung σ = M h/2 / I
def wire_size(current_a: float, length_m: float, voltage: float, max_drop_pct: float = 3.0) -> dict
    # Kupfer ρ = 0,0175 Ω·mm²/m, Hin- und Rückleiter; Querschnitt mm² und nächster AWG
def battery_c_check(mah: float, c_rating: float, current_a: float) -> dict
def unit_convert(value: float, from_unit: str, to_unit: str) -> dict
    # Länge (mm cm m km in ft), Masse (g kg lb oz), Kraft (N kgf lbf), Druck (Pa kPa bar psi),
    # Temperatur (C F K), Geschwindigkeit (m/s km/h mph kn), Energie (J Wh kWh mAh@V nicht), Leistung (W kW PS hp)
def mass_from_volume(volume_cm3: float, material: str) -> dict

def register_tools(registry: "ToolRegistry") -> None
    # Werkzeuge (nicht gefährlich), Ausgabe = deutscher Text mit Formel und Annahmen:
    # material_info(name), materialien_vergleichen(namen: "a, b, c"), akku_rechner(zellen, mah, strom_a=None),
    # schub_rechner(masse_g, motoren=4, schub_je_motor_g=None, twr=2.0), elektro_rechner(u=None,i=None,r=None,p=None),
    # spannungsteiler(u_in, r1, r2), drehmoment_rechner(hebel_m, kraft_n=None, masse_kg=None),
    # motor_rechner(kv, spannung, prop_zoll=None, steigung_zoll=None), balken_rechner(kraft_n, laenge_m, material, breite_m, hoehe_m),
    # kabel_rechner(strom_a, laenge_m, spannung, max_abfall_prozent=3), einheiten_umrechnen(wert, von, nach),
    # masse_aus_volumen(volumen_cm3, material)
```
Tests: jede Formel gegen handgerechnete Werte (z. B. 4S 1500 mAh → 22,2 Wh; 12 V/47 Ω → 0,255 A;
2 kg an 0,15 m → 2,94 Nm), Fehlerfälle, `find_material`-Toleranz, Werkzeuge über `ToolRegistry.run`.

**Integration**: `default_registry()` in `tools.py` ruft `engineering.register_tools(registry)`
(Integrations-Agent). `agents.needs_tools` erkennt zusätzlich Material-/Rechner-Fragen
(„material", „dichte", „festigkeit", „akku", „flugzeit", „schub", „drehmoment", „kabel", „awg",
„umrechn"). GAMMA/IOTA/BETA-Personas erwähnen die Rechner.

## `obito/knowledge.py` – Datenzentrum (Dokumente, RAG)

```python
@dataclass
class Document:
    id: int; title: str; path: str | None; kind: str; size: int; hash: str; project: str | None
    added_at: float; mtime: float | None; chunks: int
    def to_dict(self) -> dict   # id, titel, pfad, art, groesse, projekt, hinzugefuegt, geaendert, abschnitte

@dataclass
class Chunk:
    id: int; doc_id: int; title: str; idx: int; content: str; score: float = 0.0
    def to_dict(self) -> dict   # id, dokument_id, titel, abschnitt, inhalt, score
    def cite(self) -> str       # "[Titel §3]"

SUPPORTED = {".txt", ".md", ".rst", ".csv", ".tsv", ".json", ".yaml", ".yml", ".toml", ".ini", ".log",
             ".py", ".js", ".ts", ".html", ".htm", ".xml", ".c", ".cpp", ".h", ".ino", ".ps1", ".bat", ".sh",
             ".docx", ".xlsx", ".pptx", ".pdf"}
def extract_text(path) -> str
    # Text/Code direkt (errors="replace"); .html/.xml Tags entfernen; .docx: zipfile word/document.xml
    # (Absätze → Zeilen); .xlsx: sharedStrings + sheet-XML → "Blatt: …" + Zeilen mit ';' getrennt;
    # .pptx: ppt/slides/slideN.xml Texte je Folie; .pdf: lazy `pypdf` (sonst ValueError
    # "PDF-Unterstützung: pip install pypdf"); Binär/leer → ValueError
def chunk_text(text, size=800, overlap=100) -> list[str]   # an Absatz-/Satzgrenzen, nie mitten im Wort

class KnowledgeStore:
    def __init__(self, path, embedder=None)   # documents + chunks + chunks_fts (FTS5 content), meta embed_dim
    def add_text(self, title, text, *, project=None, kind="text", source_path=None) -> Document
    def add_file(self, path, *, project=None, title=None) -> Document   # Hash-Dedup: unverändert → vorhandenes
    def add_directory(self, path, *, project=None, patterns=None, recursive=True, max_files=500) -> dict
        # {"hinzugefuegt": n, "unveraendert": n, "uebersprungen": [pfad…], "fehler": [(pfad, grund)…]}
    def sync(self) -> dict        # Dateien mit geändertem mtime/hash neu indexieren, fehlende Dateien markieren
    def remove(self, doc_id) -> bool
    def get(self, doc_id) -> Document | None
    def list(self, project=None) -> list[Document]
    def search(self, query, k=5, project=None, min_score=0.05) -> list[Chunk]   # hybrid wie memory.search
    def context_section(self, query, k=4, max_chars=1500, project=None) -> str
        # "Auszüge aus deinen Dokumenten (Quelle in eckigen Klammern):\n[Titel §3] …\n" oder ""
    def reindex(self, progress=None, only_missing=False) -> int
    def stats(self) -> dict       # dokumente, abschnitte, mit_vektor, ohne_vektor, projekte, groesse_bytes
    def close(self)

def register_tools(registry, store) -> None
    # dokumente_suchen(frage, projekt=None) -> Treffer mit Zitat; dokument_hinzufuegen(pfad, projekt=None)
    # (Pfad via registry.resolve → nur Workspace)
```
Tests: Extraktoren mit selbst erzeugten .docx/.xlsx/.pptx (zipfile in Test erzeugen), Chunking an
Grenzen, Dedup, sync nach Dateiänderung, Suche mit/ohne Embedder (FakeBackend), context_section-Budget.

**Integration**: `Brain.__init__` erzeugt `self.knowledge = KnowledgeStore(cfg.knowledge_db, embedder)`
(neue `Config`-Property `knowledge_db = data_path/"wissen.db"` – `config.py` darf der Integrations-Agent
um Properties erweitern); `ask` Schritt 2 hängt `knowledge.context_section(question, project=project)`
an den Kontext (`agents.context_block(..., extra=…)`), Step `erinnern` meldet „N Dokument-Auszüge";
`Answer.to_dict()["dokumente"] = [chunk.to_dict()…]`. CLI `wissen add|dir|list|search|remove|sync`,
Chat `/dokument <pfad>`, `/dokumente [frage]`. API `GET /api/dokumente[?projekt]`,
`POST /api/dokumente {"pfad"} | {"titel","text"}`, `DELETE /api/dokumente/<id>`,
`GET /api/dokumente/suche?q=&n=`, `POST /api/dokumente/sync`. HUD-Tab „Datenzentrum".

## `obito/projects.py` – Projektsystem

```python
@dataclass
class Project:
    id: int; name: str; description: str; status: str   # "aktiv" | "pausiert" | "archiviert" | "fertig"
    tags: list[str]; created_at: float; updated_at: float
    def to_dict(self) -> dict   # id, name, beschreibung, status, tags, erstellt, geaendert, offene_aufgaben, notizen

@dataclass
class Note:
    id: int; project_id: int; kind: str   # "notiz" | "entscheidung" | "aufgabe" | "version" | "ergebnis" | "problem"
    title: str; content: str; done: bool; created_at: float
    def to_dict(self) -> dict   # id, projekt_id, art, titel, inhalt, erledigt, erstellt

class ProjectStore:
    def __init__(self, path)
    def create(self, name, description="", tags=()) -> Project     # ValueError bei leerem/doppeltem Namen
    def ensure(self, name) -> Project                                # vorhandenes oder neues
    def get(self, name) -> Project | None;  def get_by_id(self, pid) -> Project | None
    def list(self, include_archived=False) -> list[Project]
    def update(self, name, *, description=None, status=None, tags=None) -> Project
    def rename(self, name, new_name) -> Project
    def archive(self, name) -> Project;  def delete(self, name) -> bool
    def add_note(self, name, kind, title, content="") -> Note          # kind validiert
    def notes(self, name, kind=None, limit=50, include_done=True) -> list[Note]
    def complete(self, note_id, done=True) -> Note
    def delete_note(self, note_id) -> bool
    def add_file(self, name, path, description="") -> dict;  def files(self, name) -> list[dict]
    def summary(self, name, max_chars=800) -> str
        # "Projekt »X« (aktiv): Beschreibung … | Offene Aufgaben: … | Letzte Entscheidungen: … | Aktuelle Version: …"
    def search(self, query, k=10) -> list[Note]     # FTS über Notizen (titel+inhalt)
    def stats(self) -> dict                          # projekte, aktiv, notizen, offene_aufgaben
    def close(self)
```
Tests: CRUD, Validierung, summary-Budget, Suche, stats, Persistenz.

**Integration**: `Brain.projects = ProjectStore(cfg.projects_db)`; bei `project` in `ask`:
`projects.ensure(project)` und `projects.summary(project)` als Abschnitt „Projektkontext:" im
Kontext; Gedächtnis-Extraktion mit `art == "entscheidung"` und Projekt → zusätzlich
`projects.add_note(project, "entscheidung", inhalt[:80], inhalt)`. Missionen legen Berichte als
Notiz `ergebnis` ab. CLI `projekt list|neu|info|notiz|aufgabe|erledigt|entscheidung|datei|archiv|loeschen`,
Chat `/projekt [name]` (legt an), `/aufgabe <text>`, `/entscheidung <text>`, `/projektinfo`.
API `GET /api/projekte`, `POST /api/projekte {"name","beschreibung"?,"tags"?}`, `GET /api/projekte/<name>`,
`POST /api/projekte/<name>/notizen {"art","titel","inhalt"?}`, `POST /api/notizen/<id>/erledigt {"erledigt"}`,
`DELETE /api/notizen/<id>`, `DELETE /api/projekte/<name>` (archiviert). HUD-Tab „Projekte".

## `obito/missions.py` – Missionen

```python
@dataclass
class MissionStep:
    idx: int; description: str; kind: str      # "frage" | "werkzeug"
    tool: str | None; args: dict; status: str   # "offen" | "laeuft" | "fertig" | "fehler" | "uebersprungen"
    result: str; duration: float
    def to_dict(self) -> dict   # nr, beschreibung, art, werkzeug, args, status, ergebnis, dauer

@dataclass
class Mission:
    id: int; title: str; goal: str; project: str | None
    status: str                 # "geplant" | "laeuft" | "pausiert" | "fertig" | "fehler" | "abgebrochen"
    steps: list[MissionStep]; report: str; error: str; created_at: float; updated_at: float
    @property progress(self) -> float          # fertige Schritte / alle
    def to_dict(self) -> dict   # id, titel, ziel, projekt, status, fortschritt, schritte, bericht, fehler, erstellt, geaendert

class MissionStore:            # SQLite: missions (steps als JSON)
    def __init__(self, path);  def save(self, mission) -> Mission;  def get(self, mid) -> Mission | None
    def list(self, status=None, limit=50) -> list[Mission];  def delete(self, mid) -> bool;  def close(self)

MISSION_SCHEMA  # {"titel": str, "schritte": [{"beschreibung": str, "art": "frage|werkzeug", "werkzeug": str|null, "args": {}}]}
def mission_plan_messages(goal, project_summary, tools_block) -> list[dict]      # Marker [OBITO:mission:system]
def mission_report_messages(goal, steps: list[MissionStep]) -> list[dict]        # Marker [OBITO:bericht:OMEGA]
def parse_mission(text, known_tools: list[str]) -> dict | None   # max 8 Schritte, unbekannte Werkzeuge → art "frage"

class MissionRunner:
    def __init__(self, brain, store, *, confirm=None, max_steps=8)
    def plan(self, goal, *, project=None) -> Mission        # brain._json_call("mission", …); Fallback: 1 Schritt "frage" = Ziel
    def run(self, mission_id, *, progress=None, cancel=None) -> Mission
        # sequenziell; "frage" → brain.ask(f"{beschreibung}\n\nBisherige Ergebnisse:\n…", session_id=f"mission-{id}",
        # project=…, depth="mittel", learn=False); "werkzeug" → brain.tools.run(tool, args) (gefährlich: confirm);
        # Fehler eines Schritts → status "fehler", Mission läuft weiter (max 2 Fehler, dann Mission "fehler");
        # Bericht via brain._json_call? nein: brain.backend-Chat mit mission_report_messages (Text);
        # Projekt → projects.add_note(project, "ergebnis", titel, bericht); progress(mission) nach jedem Schritt
    def start(self, mission_id, *, progress=None) -> threading.Thread   # Hintergrund, daemon
    def stop(self, mission_id) -> bool          # setzt cancel; Status "abgebrochen"
    def running(self) -> list[int]
```
Tests mit FakeBackend-Responder über `agents.stage_of` (Stufen `mission`, `bericht`, `schnell`/`synthese`),
Werkzeugschritt mit `rechnen`, Fehlerschritt, Abbruch, Persistenz, Bericht als Projektnotiz.

**Integration**: `agents.STAGES` + `stage_of` kennen `mission`/`bericht` (die Builder liegen in
`missions.py`, nutzen aber `agents.STAGE_TAG`). `Brain.missions = MissionRunner(self, MissionStore(cfg.missions_db))`.
CLI `mission neu "Ziel" [--projekt] [--start]`, `mission list|status|start|stop|loeschen`; Chat `/mission <ziel>`
(plant, zeigt Schritte, fragt „Starten? [J/n]", zeigt Fortschritt). API `GET /api/missionen`,
`POST /api/missionen {"ziel","projekt"?,"start"?}`, `GET /api/missionen/<id>`,
`POST /api/missionen/<id>/start`, `POST /api/missionen/<id>/stop`, `DELETE /api/missionen/<id>`.
HUD-Panel „Aktive Missionen" mit Fortschrittsbalken (wie im Konzeptbild), Auto-Refresh 3 s.

## `obito/automation.py` – Automationen

```python
KINDS = ("gedaechtnis_konsolidieren", "backup", "eval", "wissen_sync", "mission", "werkzeug")

@dataclass
class Automation:
    id: int; name: str; kind: str; interval_minutes: int; params: dict; enabled: bool
    last_run: float | None; last_status: str; last_message: str; next_run: float | None; created_at: float
    def to_dict(self) -> dict   # id, name, art, intervall_minuten, parameter, aktiv, letzter_lauf, letzter_status,
                                # letzte_meldung, naechster_lauf, erstellt

class AutomationStore:   # SQLite + runs-Tabelle (automation_id, started, finished, status, message)
    def __init__(self, path); def create(self, name, kind, interval_minutes, params=None, enabled=True) -> Automation
    def get(self, aid); def list(self) -> list[Automation]; def update(self, aid, **fields) -> Automation
    def delete(self, aid) -> bool; def record_run(self, aid, status, message) -> None
    def runs(self, aid, limit=20) -> list[dict]; def due(self, now) -> list[Automation]; def close(self)

class Scheduler:
    def __init__(self, brain, store, *, log_path=None, allow_dangerous=False, tick_seconds=30)
    def run_once(self, automation) -> tuple[str, str]     # ("ok"|"fehler", meldung) – synchron, testbar
    def start(self) -> None;  def stop(self) -> None;  running: bool
    # Läufe sequenziell im Hintergrund-Thread; Fehler nie fatal; Log in cfg.logs_dir/automation.log

def default_automations() -> list[dict]   # Vorschläge: Konsolidierung täglich, Backup täglich, Wissens-Sync stündlich
```
Arten: `gedaechtnis_konsolidieren` → `brain.consolidate(days=params.get("tage", 7))`;
`backup` → `brain.backup(keep=params.get("behalten", 7)) -> Path` (Export Gedächtnis-JSON, Kopie aller .db via
`sqlite3.Connection.backup`, Konfiguration, nach `data_path/backups/<JJJJMMTT-HHMM>/`, alte löschen);
`eval` → `training.evaluate.run_eval` mit `params["datei"]` (Standard `beispiele/eval_fragen.jsonl`), Bericht nach
`logs_dir/eval-<zeit>.json`, Meldung mit Score; `wissen_sync` → `brain.knowledge.sync()`;
`mission` → `brain.missions.plan(params["ziel"])` + `run`; `werkzeug` → `brain.tools.run(params["werkzeug"],
params.get("args", {}))` (gefährlich nur mit `allow_dangerous`).

**Integration**: `Brain.consolidate(days=7, project=None) -> dict` (neu): fasst je Projekt alte
`source == "ki"`-Erinnerungen mit `importance < 0.3` und Alter > days zu einer Erinnerung `art="zusammenfassung"`
zusammen (`_json_call("konsolidierung")` mit `agents.consolidation_messages` + `CONSOLIDATION_SCHEMA`
`{"zusammenfassung": str, "behalten": [ids]}`), löscht die zusammengefassten; außerdem Sitzungen mit > 30
Nachrichten → eine `zusammenfassung`-Erinnerung je Sitzung; Rückgabe `{"zusammengefasst": n, "geloescht": n,
"sitzungen": n}`. `Brain.backup(keep=7) -> Path`. `Brain.automation = Scheduler(...)`; `serve` startet ihn;
CLI `automation list|neu|aktiv|inaktiv|jetzt|log|loeschen|vorschlaege`; Chat `/automationen`.
API `GET /api/automationen`, `POST /api/automationen`, `POST /api/automationen/<id>/jetzt`,
`POST /api/automationen/<id>/aktiv {"aktiv"}`, `DELETE /api/automationen/<id>`, `GET /api/automationen/<id>/laeufe`.
HUD-Panel „Automationen" mit Schaltern (wie im Konzeptbild).

## Paketierung, Starter, Modell-Hub

- `pyproject.toml` (setuptools): Name `obito`, Version `4.0.0`, `requires-python >= 3.10`, keine
  Pflichtabhängigkeiten, `[project.optional-dependencies] training = ["torch","transformers","peft",
  "datasets","accelerate"]`, `pdf = ["pypdf"]`, `[project.scripts] obito = "obito.cli:main"`,
  Paketdaten `obito/static/*.html`.
- `OBITO.bat` (Windows 11): prüft `python`/`py` und `ollama`; startet `ollama serve` im Hintergrund, falls
  Port 11434 nicht antwortet; `python -m obito doctor`; `python -m obito serve` im Hintergrund; öffnet
  `msedge --app=http://127.0.0.1:8765` (sonst `start http://127.0.0.1:8765`); deutsche Meldungen.
  `start_obito.sh` analog (Linux/macOS, `xdg-open`/`open`).
- Modell-Hub: CLI `modelle list|pull NAME|loeschen NAME|wechseln NAME [--schnell]|empfehlen VRAM` (pull mit
  Fortschrittsanzeige über `backend.pull(progress)`, VRAM-Passung je Modell aus `ModelInfo.size` × 1,2 + KV-Cache),
  API `POST /api/modell {"name","schnell"?}` → `brain.set_model`, `POST /api/modelle/pull {"name"}` (SSE
  `fortschritt`/`fertig`/`fehler`), `DELETE /api/modelle/<name>`. HUD: Modell-Dropdown aktiv, Pull-Feld.
  `llm.py` hat `delete(name)` (vorhanden).

## HUD (index.html) – Phase 2

Tab-Leiste oben wie im Konzept: **Übersicht** (Chat + Denk-Spur + Gedächtnis + System wie bisher),
**Projekte**, **Datenzentrum**, **Missionen**, **Automationen**, **Modelle**. Jede Ansicht nutzt die
oben definierten Endpunkte; Missionen/Automationen aktualisieren sich automatisch. Gleicher Stil
(dunkel, cyan-blaue Panels), weiterhin eine Datei, kein externes Laden, deutsch, responsiv.

## Tests

Je Modul `tests/test_<modul>.py`; Integration erweitert `test_brain.py` (Dokument-Kontext, Projektkontext,
Entscheidungs-Notiz, consolidate, backup), `test_server.py` (alle neuen Endpunkte), `test_cli.py`
(alle neuen Unterbefehle/Chat-Befehle), `test_hud.py` (neue Endpunkt-Referenzen, Tabs). Gesamtlaufzeit < 90 s.
