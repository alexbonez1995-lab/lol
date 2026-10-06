# OBITO v4.0 – AI Engineering Nexus

**Lokale, lernfähige KI-Engineering-Plattform. 100 % offline. Gehört dir.**

OBITO ist kein Chatbot, der an eine Cloud telefoniert. Es ist ein Denkkern, der auf deinem Rechner
mit offenen Modellen (Ollama / llama.cpp) läuft, sich dauerhaft erinnert, aus deinem Feedback lernt
und für Fragen aus Drohnenbau, 3D-Druck, Elektronik, Mechanik und Software ein ganzes Expertengremium
einsetzt – statt einer einzigen Antwort.

```
Frage ──▶ Erinnern ──▶ Routen ──▶ 5 Experten (parallel) ──▶ Kritiker ──▶ Revision ──▶ OMEGA-Synthese ──▶ Lernen
           Gedächtnis    Tiefe      ALPHA … IOTA              Fehler,       die zwei        eine geprüfte      Fakten merken,
           Dokumente     wählen     je Fachperspektive        Widersprüche  schwächsten     Antwort            Projekt-Notizen
           Projekt                                             Lücken        überarbeiten    (gestreamt)        Protokoll
```

Einfache Fragen gehen den schnellen Pfad (ein Aufruf mit Kontext), mittlere nutzen 3 Experten,
komplexe das volle Gremium mit Kritiker und Revision.

---

## Inhalt

1. [Was OBITO kann](#was-obito-kann)
2. [Installation (Windows 11)](#installation-windows-11)
3. [Schnellstart](#schnellstart)
4. [Modellwahl nach Grafikspeicher](#modellwahl-nach-grafikspeicher)
5. [So verbesserst du deine KI](#so-verbesserst-du-deine-ki)
6. [Die Bereiche](#die-bereiche)
7. [Kommandozeile](#kommandozeile)
8. [Konfiguration](#konfiguration)
9. [HTTP-API](#http-api)
10. [Projektstruktur](#projektstruktur)
11. [Grenzen und ehrliche Hinweise](#grenzen-und-ehrliche-hinweise)
12. [Fahrplan](#fahrplan)

---

## Was OBITO kann

| Bereich | Was dahinter steckt |
|---|---|
| **Gedächtnis** | SQLite, Volltext (FTS5) + Vektorsuche. Fakten, Präferenzen, Entscheidungen, Lösungen, Fehler – mit Wichtigkeit, Projektbezug, Duplikat-Erkennung. Automatisches Merken nach jeder Antwort (mit Schutzfiltern gegen Erfundenes). Konsolidierung alter Erinnerungen. |
| **Expertengremium** | 11 Experten (ALPHA 3D-Konstruktion, BETA Elektronik, GAMMA Physik, DELTA Code, EPSILON Daten, ZETA Sensorik, THETA Optimierung, IOTA Materialien, KAPPA Recherche, LAMBDA Planung, GENERALIST), KRITIKER und OMEGA. Routing wählt die passenden, Kritiker prüft, Revision korrigiert, Synthese entscheidet. |
| **Werkzeuge** | Sicherer Rechner, Datei-Zugriff im Arbeitsbereich (Sandbox), Python-/Shell-Ausführung mit Bestätigung und Zeitlimit, Gedächtnis-Werkzeuge, Dokumentensuche – und 12 **Ingenieur-Rechner**: Akku/Flugzeit, Schub, Drehmoment, Balken, Kabel/AWG, Motor/Propeller, Spannungsteiler, Einheiten, Materialdatenbank (22 Werkstoffe mit Richtwerten). |
| **Lernen** | 👍/👎/Korrektur je Antwort → **Lektionen** (sofort im Prompt wirksam, mit Erfolgs-Tracking), **Few-Shot-Beispiele**, **Trainingsdatensatz** (chat/alpaca/dpo, dedupliziert, train/eval-Split), **LoRA/QLoRA-Training**, eigenes **Ollama-Modell**, **A/B-Evaluation** zweier Modelle. |
| **Datenzentrum** | Eigene Dokumente (txt, md, csv, json, docx, xlsx, pptx, Code, pdf*) indexieren und semantisch durchsuchen; Auszüge fließen mit Quellenangabe in Antworten ein. |
| **Projekte** | Projekte mit Notizen, Aufgaben, Entscheidungen, Versionen, Problemen, Dateien. Der Projektkontext steht automatisch im Prompt; Entscheidungen aus Antworten werden als Notiz festgehalten. |
| **Missionen** | Mehrstufige Aufgaben: planen (JSON-Plan), ausführen (Fragen + Werkzeuge), Bericht, Ablage im Projekt. Im Hintergrund oder Vordergrund, abbrechbar. |
| **Automationen** | Zeitplaner für Gedächtnis-Konsolidierung, Backups, Eval-Läufe, Wissens-Sync, Missionen. |
| **Geräte & Sensoren** | Erkennt echte USB- und serielle Geräte (Flugsteuerungen, Mikrocontroller, GNSS-Empfänger, USB-Seriell-Wandler) über Betriebssystem-Quellen, merkt sich einen Verlauf, liest serielle Telemetrie (NMEA/GPS, Key=Value, JSON). Ohne Hardware: leere Liste, nichts Erfundenes. |
| **System & Performance** | Echte Messwerte: CPU-Last, RAM, Platte, NVIDIA-GPU/VRAM/Temperatur, geladene Modelle, eigener Prozess – mit Modellempfehlung nach VRAM und Live-Verlauf im HUD. |
| **3D-Modellierung** | Parametrische, wasserdichte Bauteile (Quader, Zylinder, Rohr, Kegel, Kugel, Lochplatte, Drohnenrahmen), STL/OBJ-Import/-Export, Volumen/Fläche/Masse/Schwerpunkt/Trägheit, Versionen je Projekt, 3D-Vorschau im HUD (eigener Renderer, keine Bibliothek). |
| **Simulation** | Zeitschritt-Modelle: Schwebeflug (LiPo-Kennlinie, Innenwiderstand, Peukert, Nutzlast), Steigflug mit Regler, Fall mit Luftwiderstand (RK4), Thermik (Ein-Knoten), PID-Sprungantwort, Akku-Entladung, Balken-Parameterstudie – mit Diagramm, Kennzahlen, Warnungen und Annahmen. |
| **Welt & Karten** | Offline-Geodäsie: Entfernung/Kurs (Haversine), Flugplan mit Wind je Abschnitt, Sonnenauf-/-untergang und Sonnenstand (NOAA), Tag-Nacht-Grenze, UTM, Orte und Routen je Projekt; Karte mit Gradnetz im HUD. Wetter und Kartenkacheln (Open-Meteo, OpenStreetMap) **nur** mit `online: true`. |
| **HUD** | Dunkle Command-Center-Oberfläche (eine HTML-Datei, kein CDN): Chat mit Live-Denk-Spur, Gedächtnis, Lektionen, Projekte, Datenzentrum, Missionen, Automationen, Modell-Hub, System, Geräte, 3D-Modellierung, Simulation, Weltkarte. |
| **App** | `python -m obito app` bzw. `OBITO.bat`: Server im Hintergrund + HUD als eigenes Fenster (Edge/Chrome-App-Modus oder pywebview); `build_exe.bat` baut `dist/OBITO/OBITO.exe`. |
| **CLI** | Vollständige Kommandozeile mit deutschem Chat (`/befehle`), `doctor`, Datensatz-Export, Eval, Training, Modell-Hub, Geräte, 3D, Simulation, Geo. |

Alles in reinem Python (Standardbibliothek). Optional: `pypdf` (PDF), `pyserial` (serielle Telemetrie unter Windows), `pywebview` (eigenes Fenster), `pyinstaller` (EXE); nur fürs LoRA-Training sind torch/transformers/peft nötig.

---

## Installation (Windows 11)

1. **Python 3.10 oder neuer** – <https://www.python.org/downloads/windows/> („Add python.exe to PATH" anhaken)
2. **Ollama** – <https://ollama.com/download/windows>
3. **Modelle laden** (einmalig, im Terminal):
   ```bat
   ollama pull qwen2.5:7b
   ollama pull nomic-embed-text
   ```
4. **OBITO holen** (ZIP entpacken oder `git clone`) und optional installieren:
   ```bat
   pip install -e .
   ```
   Danach steht der Befehl `obito` zur Verfügung (sonst immer `python -m obito`).
5. **Starten:** Doppelklick auf **`OBITO.bat`** – prüft Python/Ollama, startet Ollama und den Server,
   öffnet das HUD als App-Fenster (Edge). Oder manuell:
   ```bat
   python -m obito doctor      :: Systemprüfung
   python -m obito             :: Chat im Terminal
   python -m obito serve       :: HUD unter http://127.0.0.1:8765
   ```

Linux/macOS: `./start_obito.sh` oder dieselben `python -m obito …`-Befehle.

> **Ollama-Tipps:** `OLLAMA_NUM_PARALLEL=2` (zwei Experten gleichzeitig, braucht mehr VRAM),
> `OLLAMA_KEEP_ALIVE=30m`, `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` (mehr Kontext bei
> gleichem VRAM). `python -m obito doctor` sagt dir, was auf deinem Rechner sinnvoll ist.

---

## Schnellstart

`python -m obito app` startet Server und HUD-Fenster zusammen (das macht auch `OBITO.bat`); `serve` startet nur den Server für den Browser.

```text
$ python -m obito --projekt "Drohne"
OBITO – lokaler KI-Assistent · Modell qwen2.5:7b · Tiefe auto · Projekt Drohne
Du> Entwirf mir einen 5-Zoll-Rahmen aus CFK für ein Abfluggewicht von 650 g
⟳ Router ordnet die Frage ein…
✓ komplex → tief (ALPHA, GAMMA, IOTA, BETA, THETA)
⟳ ALPHA (3D-Konstruktion) denkt…
…
✓ KRITIKER: Bewertung 7/10, 2 Fehler, 1 Widerspruch
⟳ OMEGA denkt…
OBITO> Empfehlung: 4 mm CFK-Arme …
💾 gemerkt (unsicher): [fakt] Nutzer baut einen 5-Zoll-Quadcopter mit 650 g Abfluggewicht
Du> /schlecht Die Schraubenlänge fehlt
📚 Lektion gelernt: Bei Rahmen-Empfehlungen immer Schraubengröße und -länge angeben.
Du> /rechner akku_rechner zellen=4 mah=1500 strom_a=20
```

Wichtige Chat-Befehle: `/tief` `/mittel` `/schnell` `/auto` · `/merk <text>` `/erinnerungen` `/suche` ·
`/gut` `/schlecht <grund>` `/korrektur <bessere antwort>` `/lektionen` · `/projekt <name>` `/aufgabe` `/entscheidung`
`/projektinfo` · `/dokument <pfad>` `/dokumente [frage]` · `/mission <ziel>` `/missionen` · `/material cfk` `/rechner` ·
`/spur` `/status` `/backup` `/konsolidieren` · `/hilfe`

---

## Modellwahl nach Grafikspeicher

`python -m obito config --empfehlen <VRAM in GB>` schreibt eine passende Vorlage.

| VRAM | Hauptmodell | Routing-Modell | num_ctx | parallel_calls |
|---|---|---|---|---|
| CPU / < 8 GB | `qwen2.5:3b` | – | 4096 | 1 (Tiefe: schnell) |
| 8 GB | `qwen2.5:7b` | – | 8192 | 1 |
| 12 GB | `qwen2.5:7b` | `qwen2.5:3b` | 12288 | 2 |
| 16 GB | `qwen2.5:14b` | `qwen2.5:3b` | 16384 | 2 |

Embedding-Modell immer `nomic-embed-text` (274 MB). Qwen 2.5/3 und Gemma 3 sind im Deutschen stark;
Llama 3.1 8B ist schwächer. Denk-Modelle (qwen3, deepseek-r1) laufen, `think: aus` in der Konfiguration spart Zeit.
`python -m obito modelle list` zeigt installierte Modelle mit VRAM-Schätzung; `doctor` misst Tokens/s und warnt,
wenn ein Modell teilweise auf der CPU läuft.

---

## So verbesserst du deine KI

Die Stufenleiter – jede Stufe wirkt sofort, die nächste baut darauf auf:

### Stufe 1 – Feedback (ab der ersten Antwort)
`/gut`, `/schlecht <was war falsch>`, `/korrektur <bessere Antwort>` im Chat oder 👍/👎/✎ im HUD.
Aus Kritik und Korrekturen leitet OBITO **Lektionen** ab (`/lektionen`), die bei passenden Fragen als
verbindliche Regeln im Prompt stehen. Lektionen, die wiederholt zu schlechten Antworten führen, werden automatisch
deaktiviert. Negative Bewertungen senken die Wichtigkeit genutzter Erinnerungen und löschen daraus automatisch
gemerkte Fakten.

### Stufe 2 – Beispiele (ab ~10 bewerteten Antworten)
Positiv bewertete und korrigierte Antworten werden als Few-Shot-Beispiele zu ähnlichen Fragen eingeblendet –
kleine Modelle imitieren gute Beispiele deutlich besser als Anweisungen.

### Stufe 3 – Messen (ab ~30)
```bat
python -m obito export-dataset --ausgabe daten\obito.jsonl
python -m obito eval --datei daten\obito.jsonl.fragen.jsonl --ausgabe bericht-basis.json
python -m obito eval --datei beispiele\eval_fragen.jsonl --tiefe tief --ausgabe bericht-tief.json
python -m obito eval --vergleich bericht-basis.json bericht-tief.json --richter qwen2.5:3b
```
Der Export schreibt drei Dateien: Training, Eval-Split und ein Fragen-Set für `eval`. Der Vergleich zählt
Siege je Frage (paarweiser Richter, zweimal vertauscht) und sagt ehrlich, wann ein Unterschied nicht signifikant ist.

### Stufe 4 – LoRA-Training (ab ~200 Beispielen, ≥ 12 GB VRAM; unter Windows WSL2 empfohlen)
```bash
pip install -e ".[training]"            # torch, transformers, peft, datasets, accelerate
python -m obito train --basis qwen2.5:7b --daten daten/obito.jsonl --ausgabe lora-v1
# Route A – Adapter (klein, schnell):
python convert_lora_to_gguf.py lora-v1 --base Qwen/Qwen2.5-7B-Instruct --outfile adapter.gguf   # aus llama.cpp
python -m obito modelfile --name obito-v1 --basis qwen2.5:7b --adapter adapter.gguf --erstellen
# Route B – zusammengeführtes Modell:
python -m obito train … --zusammenfuehren   →  convert_hf_to_gguf.py  →  llama-quantize … Q4_K_M
python -m obito modelfile --name obito-v1 --basis ./merged.q4_k_m.gguf --erstellen
python -m obito modelle wechseln obito-v1
```
**Regel:** Ein Adapter passt nur zu exakt dem Basismodell, mit dem er trainiert wurde – OBITO prüft das
(`obito_training.json` neben dem Adapter). Danach Stufe 3 wiederholen: `eval --vergleich` mit altem und neuem Modell.
DPO (`--dpo`, braucht `trl`) erst ab ≥ 100 Korrektur-Paaren.

---

## Die Bereiche

**HUD** (`python -m obito serve`, dann <http://127.0.0.1:8765>): Tabs **Übersicht** (Chat, Denk-Spur, Gedächtnis,
Lektionen, System), **Projekte**, **Datenzentrum**, **Missionen** (Fortschrittsbalken, Schritte, Bericht),
**Automationen** (Schalter, „Jetzt", Läufe), **Modelle** (wechseln, laden mit Fortschritt, löschen).
Der Server lauscht nur auf 127.0.0.1, prüft Host/Origin und erlaubt gefährliche Werkzeuge nur mit
`serve --gefaehrlich-erlauben`.

**Datenzentrum:** `python -m obito wissen add docs/handbuch.md`, `wissen dir ./unterlagen`, `wissen search "ESC Dauerstrom"`,
`wissen sync`. PDF braucht `pip install pypdf`.

**Projekte:** `python -m obito projekt neu Drohne --tags fpv`, `projekt aufgabe Drohne "Motoren bestellen"`,
`projekt entscheidung Drohne "CFK-Rahmen" --inhalt "4 mm Arme"`, `projekt info Drohne`.

**Missionen:** `python -m obito mission neu "Prüfe Gewicht, Schub und Flugzeit meines Quads" --projekt Drohne --start`.

**Automationen:** `python -m obito automation vorschlaege --installieren` richtet Konsolidierung (täglich), Backup
(täglich) und Wissens-Sync (stündlich) ein; der Zeitplaner läuft, solange `serve` läuft.

**Backup/Pflege:** `/backup` sichert alle Datenbanken nach `~/.obito/backups/<zeit>/`; `/konsolidieren` verdichtet
alte, unsichere KI-Erinnerungen und lange Sitzungen.

**System:** `python -m obito system` (oder HUD-Tab **System**, Chat `/system`) zeigt CPU, RAM, Platte, GPU/VRAM,
geladene Modelle und Empfehlungen – alles gemessen, nicht geschätzt (keine GPU → „keine", nicht „0 %").

**Geräte:** `python -m obito geraete scan` erkennt angeschlossene Geräte (Windows: PnP/Registry, Linux: sysfs,
macOS: system_profiler) und ordnet Rollen zu (Flugsteuerung, Mikrocontroller, GNSS, USB-Seriell …);
`geraete lesen COM3 --baud 115200 --sekunden 3` liest Telemetrie und erkennt NMEA-Sätze (Position, Höhe,
Geschwindigkeit, Satelliten), Key=Value- und JSON-Zeilen. Unter Windows braucht das Lesen `pip install pyserial`.

**3D-Modellierung:** `python -m obito modell3d neu lochplatte l=100 b=50 t=3 "holes=-30/0/10;30/0/10" --material alu6061`
erzeugt ein wasserdichtes Netz samt Volumen, Masse und Schwerpunkt; `modell3d export 1 platte.stl`, `modell3d import teil.stl`.
Die KI nutzt dieselben Werkzeuge (`modell_erzeugen`, `modell_info`), wenn du sie nach einem Bauteil fragst.
Im HUD: Primitiv wählen, Parameter eintragen, Vorschau drehen/zoomen, STL herunterladen, Versionen je Projekt.

**Simulation:** `python -m obito simulation schwebeflug mass_g=1200 cells=4 mah=1500 hover_current_a=15 --csv flug.csv`;
ohne Argumente erscheint die Liste aller Arten mit Parametern. Jedes Ergebnis nennt Annahmen und Warnungen.

**Welt & Karten:** `python -m obito geo ort Feld "49.9576, 6.9294"`, `geo distanz Feld "50.1, 7.0"`,
`geo plan "Feld; 49.96, 6.94" --geschwindigkeit 12 --wind 20 --wind-aus 270`, `geo sonne Feld --zeit 2026-06-21`,
`geo utm Feld`, `geo wetter Feld` (nur mit `online: true`, Daten von Open-Meteo). Im HUD: Karte mit Gradnetz,
Tag-Nacht-Grenze, Orten, Routen und Flugplan; Klick setzt Wegpunkte.

---

## Kommandozeile

```text
python -m obito [--config PFAD] [--backend auto|ollama|openai] [--daten PFAD] [--modell M] UNTERBEFEHL

chat            Interaktiver Chat (Standard)           --projekt --tiefe --sitzung
doctor          Systemprüfung                          --training
serve           HUD + API                               --host --port --gefaehrlich-erlauben
export-dataset  Trainingsdatensatz                     --ausgabe --format chat|alpaca|dpo --min-bewertung --eval-anteil
eval            Fragen-Set auswerten / vergleichen     --datei --richter --tiefe --ausgabe --ohne-training --vergleich A B
train           LoRA/QLoRA/DPO (siehe train --help)
modelfile       Ollama-Modelfile erzeugen/erstellen    --name --basis --adapter --erstellen
modelle         list | pull | loeschen | wechseln | empfehlen
memory          list | search | export | import | reindex
wissen          add | dir | list | search | remove | sync | reindex
projekt         list | neu | info | notiz | aufgabe | entscheidung | version | problem | erledigt | datei | archiv | loeschen
mission         neu | list | status | start | stop | loeschen
automation      list | neu | aktiv | inaktiv | jetzt | laeufe | log | loeschen | vorschlaege
rechner         liste | material NAME | vergleich A,B | <rechner> schluessel=wert …
app             Server + HUD-Fenster in einem Prozess   --breite --hoehe --browser --ohne-fenster --ohne-webview
system          Messwerte und Empfehlungen             --json
geraete         scan | list | lesen PORT | notiz | vergessen
modell3d        arten | neu ART k=v … | import | list | info | export | loeschen
simulation      [ART k=v …]                            --json --csv PFAD
geo             distanz | plan | sonne | utm | ort | orte | route | routen | loeschen | wetter
config          --schreiben PFAD --empfehlen VRAM_GB
```

---

## Konfiguration

`obito.json` im Arbeitsverzeichnis oder `~/.obito/config.json` (Vorlage: `obito.example.json`,
`python -m obito config --empfehlen 8 --schreiben obito.json`). Jedes Feld auch als Umgebungsvariable `OBITO_<FELD>`.

| Feld | Bedeutung |
|---|---|
| `model`, `fast_model`, `embed_model` | Haupt-, Routing- und Embedding-Modell |
| `backend`, `base_url` | `auto`/`ollama`/`openai` (llama.cpp-Server, LM Studio, vLLM) |
| `num_ctx`, `keep_alive`, `think` | Kontextfenster, Modell im Speicher halten, Denk-Modelle an/aus |
| `depth` | `auto`, `schnell`, `mittel`, `tief` |
| `experts_per_question`, `experts_medium`, `parallel_calls` | Gremiumsgröße, gleichzeitige Modellaufrufe |
| `timeout`, `deadline` | Sekunden ohne Daten vom Modell / Gesamtbudget je Frage |
| `allow_tools`, `confirm_dangerous`, `workspace` | Werkzeuge, Rückfrage bei gefährlichen, Sandbox-Verzeichnis |
| `auto_memory`, `max_new_memories` | automatisches Merken |
| `max_tokens_*` | Ausgabebudget je Stufe |
| `online` | `false` (Standard): kein Internet. `true` erlaubt Wetter (Open-Meteo) und Kartenkacheln (OpenStreetMap) |
| `serial_baud` | Standard-Baudrate beim Lesen serieller Geräte (115200) |
| `data_dir` | Speicherort aller Datenbanken (Standard `~/.obito`) |

Alle Daten liegen in `data_dir`: `gedaechtnis.db`, `lernen.db`, `wissen.db`, `projekte.db`, `missionen.db`,
`automationen.db`, `geraete.db`, `modelle3d.db` + `modelle3d/` (STL), `geo.db`, `datensaetze/`, `modelle/`,
`backups/`, `logs/`, `fenster/` (Browserprofil des App-Fensters).

---

## HTTP-API

Alle Antworten `{"ok": true, …}` bzw. `{"ok": false, "fehler": "…"}`; POST/DELETE mit `Content-Type: application/json`.

| Endpunkt | Zweck |
|---|---|
| `GET /api/status` | Status, Modelle, Statistiken |
| `POST /api/frage` `{frage, sitzung?, projekt?, tiefe?, stream?}` | Antwort; mit `stream: true` als SSE (`schritt`, `token`, `antwort`, `fehler`) |
| `POST /api/feedback` `{interaktion_id, bewertung?, kommentar?, korrektur?, umschreiben?}` | Bewertung, Korrektur, Lektion |
| `GET/POST/DELETE /api/erinnerungen` | Gedächtnis |
| `GET/DELETE /api/lektionen` | Lektionen |
| `GET /api/modelle`, `POST /api/modell`, `POST /api/modelle/pull` (SSE), `DELETE /api/modelle/<name>` | Modell-Hub |
| `GET/POST/DELETE /api/dokumente`, `GET /api/dokumente/suche?q=`, `POST /api/dokumente/sync` | Datenzentrum |
| `GET/POST /api/projekte`, `GET/DELETE /api/projekte/<name>`, `POST /api/projekte/<name>/notizen`, `POST /api/notizen/<id>/erledigt`, `DELETE /api/notizen/<id>` | Projekte |
| `GET/POST /api/missionen`, `GET/DELETE /api/missionen/<id>`, `POST /api/missionen/<id>/start|stop` | Missionen |
| `GET/POST /api/automationen`, `POST /api/automationen/<id>/jetzt|aktiv`, `GET /api/automationen/<id>/laeufe`, `DELETE …`, `GET/POST /api/automationen/vorschlaege` | Automationen |
| `POST /api/pflege/konsolidieren`, `POST /api/pflege/backup` | Pflege |
| `GET /api/system`, `GET /api/system/verlauf` | Messwerte, Empfehlungen, Verlauf |
| `GET /api/geraete`, `POST /api/geraete/scan`, `POST /api/geraete/lesen` `{port, baud?, sekunden?}`, `POST /api/geraete/notiz`, `DELETE /api/geraete/<key>` | Geräte |
| `GET /api/modelle3d/arten`, `GET/POST /api/modelle3d` `{art, parameter, name?, projekt?, material?}` oder `{pfad}`, `GET /api/modelle3d/<id>`, `GET …/<id>/mesh`, `GET …/<id>/stl`, `DELETE …/<id>` | 3D-Modelle |
| `GET /api/simulation/arten`, `POST /api/simulation` `{art, parameter}` | Simulation |
| `GET/POST /api/geo/orte`, `DELETE /api/geo/orte/<id>`, `GET/POST /api/geo/routen`, `DELETE /api/geo/routen/<id>`, `POST /api/geo/plan`, `GET /api/geo/sonne?ort=|lat=&lon=`, `GET /api/geo/wetter` (403 wenn offline) | Welt |

---

## Projektstruktur

```
obito/
  config.py        Konfiguration (Datei + Umgebungsvariablen)
  llm.py           Ollama / OpenAI-kompatible Backends, Streaming, JSON-Schema, Denk-Filter, FakeBackend
  memory.py        Gedächtnis (SQLite, FTS5 + Vektoren, Verlauf, Konsolidierungs-Hilfen)
  agents.py        Experten, Prompts, Token-Budget, JSON-Schemata und tolerante Normalisierer
  tools.py         Werkzeug-Registry, Sandbox, Rechner, Prozess-Limits, Werkzeug-Protokoll, Streaming-Gate
  brain.py         Denkkern: Erinnern → Routen → Gremium → Kritiker → Revision → Synthese → Lernen; Feedback, Pflege
  learning.py      Interaktionen, Lektionen, Beispiele, Datensatz-Export
  engineering.py   Ingenieur-Rechner und Materialdatenbank
  knowledge.py     Datenzentrum (Dokumente, Chunking, Suche)
  projects.py      Projektsystem
  missions.py      Missionen (Plan → Ausführung → Bericht)
  automation.py    Zeitplaner
  devices.py       Geräte & Sensoren (USB/seriell, Telemetrie, Verlauf)
  sysmon.py        System & Performance (Messwerte, Empfehlungen, Verlauf)
  geometry.py      3D-Modellierung (Primitive, STL/OBJ, Statistik, Modellspeicher)
  simulation.py    Zeitschritt-Simulationen
  geo.py           Welt & Karten (Geodäsie, Sonne, UTM, Orte/Routen, Wetter online)
  desktop.py       App-Fenster (Server + Browser-App-Modus / pywebview)
  training/        modelfile.py (Ollama), evaluate.py (Eval + A/B), train_lora.py (LoRA/QLoRA/DPO)
  server.py        HTTP-API + SSE, gehärtet
  static/index.html  HUD (eine Datei)
  cli.py, cli_extra.py, __main__.py   Kommandozeile
packaging/         obito.spec, obito_app.py (PyInstaller → dist/OBITO/OBITO.exe)
tests/             unittest, ~1000 Tests ohne Modell und ohne Hardware (FakeBackend, gepatchte Quellen), < 40 s
docs/              ARCHITEKTUR.md, ARCHITEKTUR_PHASE2.md, ARCHITEKTUR_PHASE3.md (verbindliche Spezifikationen)
beispiele/         eval_fragen.jsonl (10 Ingenieur-Fragen mit Stichworten)
OBITO.bat, start_obito.sh, build_exe.bat, pyproject.toml, obito.example.json
```

Tests: `python -m unittest discover -s tests`.

---

## Grenzen und ehrliche Hinweise

- **Die Intelligenz hängt am Modell.** Ein 7B-Modell macht Fehler; das Gremium reduziert sie, ersetzt aber kein
  Fachwissen. Alle Zahlen prüfen, bevor etwas fliegt, fährt oder unter Strom steht.
- **Geschwindigkeit:** Stufe *tief* bedeutet 7–10 Modellaufrufe. Auf einer 8-GB-Karte mit 7B sind das 1–3 Minuten;
  `parallel_calls` nur erhöhen, wenn `OLLAMA_NUM_PARALLEL` und der VRAM es hergeben. `/schnell` ist immer eine Option.
- **Materialwerte** sind Richtwerte typischer Datenblätter, keine Prüfzeugnisse.
- **Werkzeuge** mit Dateischreiben/Code-Ausführung fragen im Chat nach; im Server sind sie standardmäßig gesperrt.
- **Training** braucht eine NVIDIA-GPU mit ≥ 12 GB (7B QLoRA) und unter Windows am besten WSL2; mit wenigen Dutzend
  Beispielen bringt ein LoRA kaum etwas – Lektionen und Beispiele wirken früher.
- **Kein Internet** ist Absicht: nichts verlässt den Rechner, außer du installierst Modelle.

## Fahrplan

- MAVLink-/DJI-SDK-Telemetrie (binäre Protokolle) über die bestehende Geräteerkennung
- Boolesche Operationen und STEP-Export in der 3D-Modellierung; Druckvorlagen im Projektsystem
- Tray-Icon und Autostart für die App; signierte EXE
- Vision-Modelle (Bilder von Platinen/Bauteilen erkennen) über Ollama-Multimodal

Lizenz: MIT.
