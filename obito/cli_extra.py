"""Phase-2-Unterbefehle der Kommandozeile: ``wissen``, ``projekt``, ``mission``, ``automation``,
``modelle``, ``rechner`` – plus Formatierungshilfen, die auch die Chat-Befehle nutzen.

Alle Funktionen schreiben nach ``out`` (Tests: ``io.StringIO``) und liefern einen Exit-Code.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from . import engineering
from .brain import Brain
from .config import Config, find_config_file, save_config
from .llm import LLMBackend, LLMError, ModelInfo, ModelNotFound

COMMANDS = ("wissen", "projekt", "mission", "automation", "modelle", "rechner",
            "system", "geraete", "modell3d", "simulation", "geo")

AUTOMATION_KINDS = ("gedaechtnis_konsolidieren", "backup", "eval", "wissen_sync", "mission", "werkzeug")
MISSION_STATUSES = ("geplant", "laeuft", "pausiert", "fertig", "fehler", "abgebrochen")
FINAL_MISSION_STATUSES = ("fertig", "fehler", "abgebrochen")
NOTE_KINDS = ("notiz", "entscheidung", "aufgabe", "version", "ergebnis", "problem")

STATUS_MARK = {"offen": "·", "laeuft": "⟳", "fertig": "✓", "fehler": "✗", "uebersprungen": "–",
               "geplant": "·", "pausiert": "‖", "abgebrochen": "⏹"}


def _println(out: TextIO, text: str = "") -> None:
    try:
        out.write(text + "\n")
    except UnicodeEncodeError:
        enc = getattr(out, "encoding", None) or "utf-8"
        out.write(text.encode(enc, "replace").decode(enc) + "\n")
    try:
        out.flush()
    except Exception:  # noqa: BLE001
        pass


def _fmt_bytes(n: int | float | None) -> str:
    if not n:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def _fmt_time(value: Any) -> str:
    if value is None or value == "":
        return "–"
    if isinstance(value, (int, float)):
        return time.strftime("%d.%m.%Y %H:%M", time.localtime(value))
    text = str(value)
    return text.replace("T", " ")[:16] if len(text) >= 16 and text[4:5] == "-" else text


def _parse_kv(tokens: Sequence[str]) -> dict:
    """``schluessel=wert`` → dict (Zahlen werden als Text übergeben, die Registry koerziert)."""
    args: dict = {}
    for tok in tokens:
        key, sep, value = tok.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"Parameter »{tok}« hat nicht die Form schluessel=wert")
        args[key.strip()] = value.strip()
    return args


# ------------------------------------------------------------------ Format
def fmt_document(d: Any) -> str:
    dd = d.to_dict() if hasattr(d, "to_dict") else dict(d)
    proj = f" · {dd.get('projekt')}" if dd.get("projekt") else ""
    missing = "  [Datei fehlt]" if dd.get("fehlt") else ""
    return (f"#{dd.get('id')} {dd.get('titel')} [{dd.get('art')}{proj}] – {dd.get('abschnitte')} Abschnitte, "
            f"{_fmt_bytes(dd.get('groesse'))}, {_fmt_time(dd.get('hinzugefuegt'))}{missing}")


def fmt_project(p: Any) -> str:
    pd = p.to_dict() if hasattr(p, "to_dict") else dict(p)
    tags = f" [{', '.join(pd.get('tags') or [])}]" if pd.get("tags") else ""
    desc = f" – {pd.get('beschreibung')}" if pd.get("beschreibung") else ""
    return (f"#{pd.get('id')} {pd.get('name')} ({pd.get('status')}){tags}{desc} · "
            f"{pd.get('offene_aufgaben', 0)} offene Aufgaben, {pd.get('notizen', 0)} Notizen")


def fmt_note(n: Any) -> str:
    nd = n.to_dict() if hasattr(n, "to_dict") else dict(n)
    box = ""
    if nd.get("art") == "aufgabe":
        box = "☑ " if nd.get("erledigt") else "☐ "
    content = f" – {nd.get('inhalt')}" if nd.get("inhalt") and nd.get("inhalt") != nd.get("titel") else ""
    return f"#{nd.get('id')} [{nd.get('art')} · {_fmt_time(nd.get('erstellt'))}] {box}{nd.get('titel')}{content}"


def fmt_mission(m: Any, verbose: bool = False) -> str:
    md = m.to_dict() if hasattr(m, "to_dict") else dict(m)
    done = sum(1 for s in md.get("schritte", []) if s.get("status") == "fertig")
    total = len(md.get("schritte", []))
    proj = f" · {md.get('projekt')}" if md.get("projekt") else ""
    head = (f"#{md.get('id')} {md.get('titel')} [{md.get('status')}{proj}] "
            f"{done}/{total} Schritte ({int(round(float(md.get('fortschritt') or 0) * 100))} %)")
    if not verbose:
        return head
    lines = [head, f"  Ziel: {md.get('ziel')}"]
    for s in md.get("schritte", []):
        mark = STATUS_MARK.get(s.get("status"), "·")
        tool = f" ({s.get('werkzeug')})" if s.get("art") == "werkzeug" and s.get("werkzeug") else ""
        lines.append(f"  {mark} {s.get('nr')}. {s.get('beschreibung')}{tool} – {s.get('status')}")
        if s.get("ergebnis"):
            res = " ".join(str(s["ergebnis"]).split())
            lines.append(f"      → {res[:200]}{'…' if len(res) > 200 else ''}")
    if md.get("bericht"):
        lines.append("  Bericht:")
        for line in str(md["bericht"]).splitlines():
            lines.append(f"    {line}")
    if md.get("fehler"):
        lines.append(f"  Fehler: {md['fehler']}")
    return "\n".join(lines)


def fmt_automation(a: Any) -> str:
    ad = a.to_dict() if hasattr(a, "to_dict") else dict(a)
    state = "aktiv" if ad.get("aktiv") else "inaktiv"
    last = ad.get("letzter_status") or "nie gelaufen"
    msg = f" – {ad.get('letzte_meldung')}" if ad.get("letzte_meldung") else ""
    return (f"#{ad.get('id')} {ad.get('name')} [{ad.get('art')}, alle {ad.get('intervall_minuten')} min, {state}] "
            f"letzter Lauf: {_fmt_time(ad.get('letzter_lauf'))} ({last}){msg} · nächster: {_fmt_time(ad.get('naechster_lauf'))}")


def _parse_params(value: str | None) -> dict:
    if not value:
        return {}
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("Parameter müssen ein JSON-Objekt sein, z. B. {\"tage\": 7}")
    return data


# ------------------------------------------------------------ Modell-Hub
_PARAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*B", re.IGNORECASE)


def model_params_b(info: ModelInfo) -> float | None:
    m = _PARAM_RE.search(info.parameters or "")
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"(\d+(?:\.\d+)?)b\b", info.name.lower())
    return float(m.group(1)) if m else None


def model_vram_estimate(info: ModelInfo, num_ctx: int) -> float | None:
    """Grobe VRAM-Schätzung in GB: Gewichte × 1,1 + KV-Cache (≈ 0,5 GB je 8k Kontext bei 7B,
    skaliert mit der Parameterzahl) + 0,4 GB Reserve. Heuristik, keine Messung."""
    if not info.size:
        return None
    params = model_params_b(info) or 7.0
    weights = info.size / 1e9 * 1.1
    kv = 0.5 * (max(1, int(num_ctx)) / 8192) * (params / 7.0)
    return round(weights + kv + 0.4, 1)


def fmt_model(info: ModelInfo, cfg: Config, current: bool = False, fast: bool = False) -> str:
    marks = []
    if current:
        marks.append("Hauptmodell")
    if fast:
        marks.append("Routing-Modell")
    mark = f"  ← {', '.join(marks)}" if marks else ""
    est = model_vram_estimate(info, cfg.num_ctx)
    vram = f" · braucht ≈ {est:.1f} GB VRAM bei num_ctx {cfg.num_ctx}" if est else ""
    return f"{info.short()}{vram}{mark}"


def pull_with_progress(backend: LLMBackend, name: str, out: TextIO) -> bool:
    last = {"line": "", "t": 0.0}

    def progress(chunk: dict) -> None:
        status = str(chunk.get("status") or "")
        total = chunk.get("total") or 0
        done = chunk.get("completed") or 0
        now = time.time()
        if total:
            pct = min(100, int(done * 100 / total))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            line = f"  {bar} {pct:3d} %  {_fmt_bytes(done)} / {_fmt_bytes(total)}  {status}"
        else:
            line = f"  {status}"
        if line != last["line"] and (now - last["t"] > 0.3 or pct_done(chunk)):
            _println(out, line)
            last["line"], last["t"] = line, now

    def pct_done(chunk: dict) -> bool:
        return chunk.get("status") == "success" or (chunk.get("total") and chunk.get("completed") == chunk.get("total"))

    return backend.pull(name, progress=progress)


def persist_model_choice(cfg: Config, out: TextIO, explicit_path: str | None = None) -> Path | None:
    target = Path(explicit_path) if explicit_path else (find_config_file() or Path("obito.json"))
    try:
        save_config(cfg, target)
    except OSError as e:
        _println(out, f"Hinweis: Konfiguration konnte nicht gespeichert werden ({target}): {e}")
        return None
    return target


# ------------------------------------------------------------- Missionen
def run_mission_foreground(brain: Brain, mission_id: int, out: TextIO, *, cancel: threading.Event | None = None):
    """Führt eine Mission im Vordergrund aus und schreibt je Schritt eine Zeile."""
    seen: dict[int, str] = {}

    def progress(mission: Any) -> None:
        for s in mission.steps:
            if seen.get(s.idx) != s.status and s.status in ("fertig", "fehler", "uebersprungen", "laeuft"):
                seen[s.idx] = s.status
                mark = STATUS_MARK.get(s.status, "·")
                res = " ".join(str(s.result or "").split())
                tail = f" → {res[:160]}{'…' if len(res) > 160 else ''}" if res and s.status != "laeuft" else ""
                _println(out, f"  {mark} {s.idx}. {s.description} [{s.status}]{tail}")

    return brain.missions.run(mission_id, progress=progress, cancel=cancel)


# ------------------------------------------------------------- Parser
def add_parsers(add: Callable[..., argparse.ArgumentParser], germanize: Callable[[argparse.ArgumentParser], None],
                add_global: Callable[[argparse.ArgumentParser, bool], None], formatter: Any) -> None:
    """Registriert die Phase-2-Unterbefehle am Hauptparser (``add(name, help)`` liefert den Unterparser)."""

    def nested(parent: argparse.ArgumentParser, dest: str):
        sub = parent.add_subparsers(dest=dest, metavar="AKTION", title="Aktionen")

        def make(name: str, help_text: str) -> argparse.ArgumentParser:
            sp = sub.add_parser(name, help=help_text, description=help_text, formatter_class=formatter, add_help=False)
            germanize(sp)
            add_global(sp, True)
            return sp

        return make

    # wissen
    p = add("wissen", "Datenzentrum: eigene Dokumente indexieren und durchsuchen (RAG).")
    w = nested(p, "aktion")
    sp = w("add", "Datei indexieren (txt, md, csv, json, docx, xlsx, pptx, pdf*, Code)")
    sp.add_argument("pfad")
    sp.add_argument("--projekt", default=None)
    sp.add_argument("--titel", default=None)
    sp = w("dir", "alle unterstützten Dateien eines Ordners indexieren")
    sp.add_argument("pfad")
    sp.add_argument("--projekt", default=None)
    sp.add_argument("--max", default=500, type=int, help="höchstens N Dateien (Standard: 500)")
    sp = w("list", "indexierte Dokumente anzeigen")
    sp.add_argument("--projekt", default=None)
    sp = w("search", "Dokumente semantisch durchsuchen")
    sp.add_argument("frage")
    sp.add_argument("-n", "--anzahl", default=5, type=int)
    sp.add_argument("--projekt", default=None)
    sp = w("remove", "Dokument aus dem Index entfernen")
    sp.add_argument("id", type=int)
    w("sync", "geänderte Dateien neu indexieren, fehlende markieren")
    sp = w("reindex", "Vektoren der Abschnitte neu berechnen")
    sp.add_argument("--alle", action="store_true", help="auch veraltete Vektoren (Modellwechsel)")

    # projekt
    p = add("projekt", "Projektsystem: Projekte, Notizen, Entscheidungen, Aufgaben, Versionen, Dateien.")
    pr = nested(p, "aktion")
    sp = pr("list", "Projekte anzeigen")
    sp.add_argument("--alle", action="store_true", help="auch archivierte")
    sp = pr("neu", "Projekt anlegen")
    sp.add_argument("name")
    sp.add_argument("--beschreibung", default="")
    sp.add_argument("--tags", default="", help="kommagetrennt")
    sp = pr("info", "Projekt mit Zusammenfassung, Notizen und Dateien anzeigen")
    sp.add_argument("name")
    for kind, help_text in (("notiz", "Notiz hinzufügen"), ("aufgabe", "Aufgabe hinzufügen"),
                            ("entscheidung", "Entscheidung festhalten"), ("version", "Version festhalten"),
                            ("problem", "Problem festhalten")):
        sp = pr(kind, help_text)
        sp.add_argument("name", help="Projektname")
        sp.add_argument("titel")
        sp.add_argument("--inhalt", default="")
    sp = pr("erledigt", "Aufgabe als erledigt (oder wieder offen) markieren")
    sp.add_argument("id", type=int)
    sp.add_argument("--offen", action="store_true")
    sp = pr("datei", "Datei dem Projekt zuordnen")
    sp.add_argument("name")
    sp.add_argument("pfad")
    sp.add_argument("--beschreibung", default="")
    sp = pr("archiv", "Projekt archivieren")
    sp.add_argument("name")
    sp = pr("loeschen", "Projekt samt Notizen löschen")
    sp.add_argument("name")
    sp.add_argument("--ja", action="store_true", help="ohne Rückfrage")

    # mission
    p = add("mission", "Missionen: mehrstufige Aufgaben planen, ausführen, verfolgen.")
    mi = nested(p, "aktion")
    sp = mi("neu", "Mission planen (und mit --start ausführen)")
    sp.add_argument("ziel")
    sp.add_argument("--projekt", default=None)
    sp.add_argument("--start", action="store_true")
    sp.add_argument("--hintergrund", action="store_true", help="im Hintergrund starten statt im Vordergrund laufen")
    sp = mi("list", "Missionen anzeigen")
    sp.add_argument("--status", default=None, choices=MISSION_STATUSES)
    sp = mi("status", "Mission mit Schritten und Bericht anzeigen")
    sp.add_argument("id", type=int)
    sp = mi("start", "geplante Mission ausführen")
    sp.add_argument("id", type=int)
    sp.add_argument("--hintergrund", action="store_true")
    sp = mi("stop", "laufende Mission abbrechen")
    sp.add_argument("id", type=int)
    sp = mi("loeschen", "Mission löschen")
    sp.add_argument("id", type=int)

    # automation
    p = add("automation", "Automationen: Zeitplaner für Konsolidierung, Backup, Eval, Wissens-Sync, Missionen.")
    au = nested(p, "aktion")
    au("list", "Automationen anzeigen")
    sp = au("neu", "Automation anlegen")
    sp.add_argument("name")
    sp.add_argument("--art", required=True, choices=AUTOMATION_KINDS)
    sp.add_argument("--intervall", required=True, type=int, metavar="MINUTEN")
    sp.add_argument("--parameter", default=None, metavar="JSON", help='z. B. {"tage": 7}')
    sp.add_argument("--inaktiv", action="store_true")
    for name, help_text in (("aktiv", "Automation aktivieren"), ("inaktiv", "Automation deaktivieren"),
                            ("jetzt", "Automation sofort ausführen"), ("loeschen", "Automation löschen"),
                            ("laeufe", "letzte Läufe anzeigen")):
        sp = au(name, help_text)
        sp.add_argument("id", type=int)
    sp = au("log", "Protokoll des Zeitplaners anzeigen")
    sp.add_argument("-n", "--anzahl", default=30, type=int)
    sp = au("vorschlaege", "empfohlene Automationen anzeigen oder einrichten")
    sp.add_argument("--installieren", action="store_true")

    # modelle
    p = add("modelle", "Modell-Hub: installierte Modelle, Laden, Löschen, Wechseln, Empfehlung.")
    mo = nested(p, "aktion")
    mo("list", "installierte Modelle mit VRAM-Schätzung anzeigen")
    sp = mo("pull", "Modell laden (ollama pull)")
    sp.add_argument("name")
    sp = mo("loeschen", "Modell löschen")
    sp.add_argument("name")
    sp = mo("wechseln", "Haupt- oder Routing-Modell setzen und in der Konfiguration speichern")
    sp.add_argument("name")
    sp.add_argument("--schnell", action="store_true", help="Routing-/Extraktionsmodell statt Hauptmodell")
    sp.add_argument("--nicht-speichern", action="store_true")
    sp = mo("empfehlen", "Modellempfehlung nach Grafikspeicher")
    sp.add_argument("vram_gb", type=float)

    # rechner
    p = add("rechner", "Ingenieur-Rechner und Materialdatenbank direkt aufrufen.")
    p.add_argument("werkzeug", nargs="?", default=None,
                   help="material <name> | vergleich <a,b,c> | liste | <rechner> schluessel=wert …")
    p.add_argument("parameter", nargs="*", help="z. B. zellen=4 mah=1500 strom_a=20")

    # ---- Phase 3
    p = add("system", "System & Performance: CPU, RAM, Platte, GPU/VRAM, geladene Modelle, Empfehlungen.")
    p.add_argument("--json", action="store_true", help="Rohdaten als JSON ausgeben")

    p = add("geraete", "Geräte & Sensoren: angeschlossene USB-/serielle Geräte erkennen, Telemetrie lesen.")
    ge = nested(p, "aktion")
    sp = ge("scan", "Geräte erkennen und im Verlauf speichern")
    sp.add_argument("--json", action="store_true")
    sp = ge("list", "Geräteverlauf anzeigen")
    sp.add_argument("--verbunden", action="store_true", help="nur aktuell verbundene")
    sp = ge("lesen", "serielle Rohdaten lesen und Telemetrie (NMEA, Key=Value, JSON) auswerten")
    sp.add_argument("port")
    sp.add_argument("--baud", default=None, type=int)
    sp.add_argument("--sekunden", default=2.0, type=float)
    sp = ge("notiz", "Notiz zu einem Gerät speichern")
    sp.add_argument("key")
    sp.add_argument("text", nargs="*")
    sp = ge("vergessen", "Gerät aus dem Verlauf löschen")
    sp.add_argument("key")

    p = add("modell3d", "3D-Modellierung: parametrische Bauteile erzeugen, importieren, prüfen, exportieren.")
    m3 = nested(p, "aktion")
    m3("arten", "verfügbare Primitive und Parameter anzeigen")
    sp = m3("neu", "Modell aus Primitiv erzeugen (Parameter als name=wert)")
    sp.add_argument("art")
    sp.add_argument("parameter", nargs="*", help="z. B. l=100 b=50 h=10  (lochplatte: holes=x/y/d;x/y/d)")
    sp.add_argument("--name", default=None)
    sp.add_argument("--projekt", default=None)
    sp.add_argument("--material", default=None)
    sp = m3("import", "STL/OBJ-Datei in den Modellspeicher übernehmen")
    sp.add_argument("pfad")
    sp.add_argument("--name", default=None)
    sp.add_argument("--projekt", default=None)
    sp = m3("list", "gespeicherte Modelle anzeigen")
    sp.add_argument("--projekt", default=None)
    sp = m3("info", "Statistik eines Modells (Volumen, Masse, Schwerpunkt, Wasserdichtigkeit)")
    sp.add_argument("id", type=int)
    sp.add_argument("--material", default=None)
    sp = m3("export", "Modell als STL (binär/ascii) oder OBJ schreiben")
    sp.add_argument("id", type=int)
    sp.add_argument("ziel")
    sp.add_argument("--ascii", action="store_true")
    sp = m3("loeschen", "Modellversion löschen")
    sp.add_argument("id", type=int)

    p = add("simulation", "Simulation: Flugzeit, Steigflug, Fall, Thermik, Regler, Akku, Balkenstudie.")
    p.add_argument("art", nargs="?", default=None, help="Simulationsart (ohne Angabe: Liste)")
    p.add_argument("parameter", nargs="*", help="name=wert … (Listen mit Semikolon)")
    p.add_argument("--json", action="store_true", help="vollständiges Ergebnis mit Reihen als JSON")
    p.add_argument("--csv", default=None, metavar="PFAD", help="Verlauf als CSV schreiben")

    p = add("geo", "Welt & Karten: Entfernung, Flugplan, Sonnenstand, UTM, Orte/Routen, Wetter (online).")
    g = nested(p, "aktion")
    sp = g("distanz", "Entfernung und Kurs zwischen zwei Punkten (Koordinaten oder Ortsnamen)")
    sp.add_argument("von")
    sp.add_argument("nach")
    sp = g("plan", "Flugplan über Wegpunkte")
    sp.add_argument("punkte", help="»lat,lon; lat,lon; …« oder Ortsnamen")
    sp.add_argument("--geschwindigkeit", default=10.0, type=float, metavar="M_S")
    sp.add_argument("--wind", default=0.0, type=float, metavar="KMH")
    sp.add_argument("--wind-aus", default=0.0, type=float, metavar="GRAD")
    sp = g("sonne", "Sonnenaufgang/-untergang und Sonnenstand")
    sp.add_argument("ort")
    sp.add_argument("--zeit", default=None, help="ISO-Datum/-Zeit (Standard: jetzt)")
    sp = g("utm", "UTM-Koordinaten eines Punkts")
    sp.add_argument("ort")
    sp = g("ort", "Ort speichern")
    sp.add_argument("name")
    sp.add_argument("koordinate")
    sp.add_argument("--projekt", default=None)
    sp.add_argument("--notiz", default="")
    sp = g("orte", "gespeicherte Orte anzeigen")
    sp.add_argument("--projekt", default=None)
    sp = g("route", "Route speichern")
    sp.add_argument("name")
    sp.add_argument("punkte")
    sp.add_argument("--projekt", default=None)
    sp = g("routen", "gespeicherte Routen anzeigen")
    sp.add_argument("--projekt", default=None)
    sp = g("loeschen", "Ort oder Route löschen")
    sp.add_argument("art", choices=("ort", "route"))
    sp.add_argument("id", type=int)
    sp = g("wetter", "aktuelles Wetter und Flugtauglichkeit (nur mit online=true)")
    sp.add_argument("ort")


# ------------------------------------------------------------- Ausführung
def _brain(cfg: Config, backend: LLMBackend | None) -> Brain:
    cfg.ensure_dirs()
    return Brain(cfg, backend=backend)


def run(cmd: str, cfg: Config, args: argparse.Namespace, out: TextIO, inp: Callable[[str], str],
        backend: LLMBackend | None = None) -> int:
    handler = {
        "wissen": cmd_wissen, "projekt": cmd_projekt, "mission": cmd_mission,
        "automation": cmd_automation, "modelle": cmd_modelle, "rechner": cmd_rechner,
        "system": cmd_system, "geraete": cmd_geraete, "modell3d": cmd_modell3d,
        "simulation": cmd_simulation, "geo": cmd_geo,
    }[cmd]
    return handler(cfg, args, out, inp, backend)


def cmd_wissen(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito wissen {add,dir,list,search,remove,sync,reindex} … (--help für Details)")
        return 2
    brain = _brain(cfg, backend)
    store = brain.knowledge
    try:
        if action == "add":
            path = Path(args.pfad)
            if not path.exists():
                _println(out, f"Fehler: {args.pfad} nicht gefunden.")
                return 1
            if path.is_dir():
                result = store.add_directory(path, project=args.projekt)
                _println(out, f"{result['hinzugefuegt']} hinzugefügt, {result['unveraendert']} unverändert, "
                              f"{len(result['uebersprungen'])} übersprungen, {len(result['fehler'])} Fehler.")
                return 0
            try:
                doc = store.add_file(path, project=args.projekt, title=args.titel)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Indexiert: {fmt_document(doc)}")
            return 0
        if action == "dir":
            if not Path(args.pfad).is_dir():
                _println(out, f"Fehler: {args.pfad} ist kein Ordner.")
                return 1
            result = store.add_directory(args.pfad, project=args.projekt, max_files=args.max)
            _println(out, f"{result['hinzugefuegt']} hinzugefügt, {result['unveraendert']} unverändert, "
                          f"{len(result['uebersprungen'])} übersprungen, {len(result['fehler'])} Fehler.")
            for pfad, grund in result["fehler"][:10]:
                _println(out, f"  Fehler {pfad}: {grund}")
            return 0
        if action == "list":
            docs = store.list(args.projekt)
            if not docs:
                _println(out, "Keine Dokumente im Datenzentrum. Hinzufügen: python -m obito wissen add <pfad>")
                return 0
            for d in docs:
                _println(out, fmt_document(d))
            st = store.stats()
            _println(out, f"{st['dokumente']} Dokumente, {st['abschnitte']} Abschnitte, "
                          f"{st['mit_vektor']} mit Vektor, {_fmt_bytes(st['groesse_bytes'])}.")
            return 0
        if action == "search":
            hits = store.search(args.frage, k=args.anzahl, project=args.projekt)
            if not hits:
                _println(out, "Keine Treffer.")
                return 0
            for c in hits:
                _println(out, f"{c.cite()} (Score {c.score:.2f})")
                _println(out, "  " + " ".join(c.content.split())[:400])
            return 0
        if action == "remove":
            if store.remove(args.id):
                _println(out, f"Dokument #{args.id} entfernt.")
                return 0
            _println(out, f"Kein Dokument mit der ID #{args.id}.")
            return 1
        if action == "sync":
            r = store.sync()
            _println(out, f"{r['aktualisiert']} aktualisiert, {r['unveraendert']} unverändert, "
                          f"{len(r['fehlend'])} fehlend, {len(r['fehler'])} Fehler.")
            return 0
        if action == "reindex":
            n = store.reindex(progress=lambda d, t: _println(out, f"  {d}/{t} Abschnitte eingebettet"),
                              only_missing=not args.alle)
            st = store.stats()
            _println(out, f"{n} Abschnitte neu eingebettet; {st['mit_vektor']} mit und {st['ohne_vektor']} ohne Vektor.")
            if n == 0 and st["ohne_vektor"]:
                _println(out, "Hinweis: Kein Embedding-Modell erreichbar? (`ollama pull nomic-embed-text`, Ollama starten)")
                return 1
            return 0
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        brain.close()


def cmd_projekt(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito projekt {list,neu,info,notiz,aufgabe,entscheidung,version,problem,"
                      "erledigt,datei,archiv,loeschen} … (--help für Details)")
        return 2
    brain = _brain(cfg, backend)
    store = brain.projects
    try:
        if action == "list":
            projects = store.list(include_archived=args.alle)
            if not projects:
                _println(out, "Noch keine Projekte. Anlegen: python -m obito projekt neu <name>")
                return 0
            for p in projects:
                _println(out, fmt_project(p))
            return 0
        if action == "neu":
            tags = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
            try:
                p = store.create(args.name, args.beschreibung, tags)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Projekt angelegt: {fmt_project(p)}")
            return 0
        if action == "info":
            p = store.get(args.name)
            if p is None:
                _println(out, f"Projekt »{args.name}« unbekannt.")
                return 1
            _println(out, fmt_project(p))
            summary = store.summary(args.name)
            if summary:
                _println(out, summary)
            notes = store.notes(args.name, limit=50)
            if notes:
                _println(out, "Notizen:")
                for n in notes:
                    _println(out, "  " + fmt_note(n))
            files = store.files(args.name)
            if files:
                _println(out, "Dateien:")
                for f in files:
                    _println(out, f"  #{f['id']} {f['pfad']}" + (f" – {f['beschreibung']}" if f.get("beschreibung") else "")
                                  + ("" if f.get("existiert", True) else "  [fehlt]"))
            return 0
        if action in NOTE_KINDS:
            if store.get(args.name) is None:
                _println(out, f"Projekt »{args.name}« unbekannt – zuerst anlegen: python -m obito projekt neu {args.name}")
                return 1
            try:
                n = store.add_note(args.name, action, args.titel, args.inhalt)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Festgehalten: {fmt_note(n)}")
            return 0
        if action == "erledigt":
            try:
                n = store.complete(args.id, not args.offen)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, fmt_note(n))
            return 0
        if action == "datei":
            if store.get(args.name) is None:
                _println(out, f"Projekt »{args.name}« unbekannt.")
                return 1
            f = store.add_file(args.name, args.pfad, args.beschreibung)
            _println(out, f"Datei #{f['id']} zugeordnet: {f['pfad']}" + ("" if f.get("existiert", True) else "  [existiert nicht]"))
            return 0
        if action == "archiv":
            try:
                p = store.archive(args.name)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Archiviert: {fmt_project(p)}")
            return 0
        if action == "loeschen":
            if store.get(args.name) is None:
                _println(out, f"Projekt »{args.name}« unbekannt.")
                return 1
            if not args.ja:
                answer = inp(f"Projekt »{args.name}« mit allen Notizen löschen? [j/N] ")
                if (answer or "").strip().lower() not in ("j", "ja", "y", "yes"):
                    _println(out, "Abgebrochen.")
                    return 1
            store.delete(args.name)
            _println(out, f"Projekt »{args.name}« gelöscht.")
            return 0
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        brain.close()


def cmd_mission(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito mission {neu,list,status,start,stop,loeschen} … (--help für Details)")
        return 2
    brain = _brain(cfg, backend)
    runner = brain.missions
    try:
        if action == "neu":
            try:
                m = runner.plan(args.ziel, project=args.projekt)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, fmt_mission(m, verbose=True))
            if m.error:
                _println(out, f"Hinweis: {m.error}")
            if not args.start:
                _println(out, f"Starten: python -m obito mission start {m.id}")
                return 0
            return _start_mission(brain, m.id, args.hintergrund, out)
        if action == "list":
            missions = runner.store.list(status=args.status)
            if not missions:
                _println(out, "Keine Missionen.")
                return 0
            for m in missions:
                _println(out, fmt_mission(m))
            return 0
        if action == "status":
            m = runner.store.get(args.id)
            if m is None:
                _println(out, f"Mission #{args.id} unbekannt.")
                return 1
            _println(out, fmt_mission(m, verbose=True))
            return 0
        if action == "start":
            if runner.store.get(args.id) is None:
                _println(out, f"Mission #{args.id} unbekannt.")
                return 1
            return _start_mission(brain, args.id, args.hintergrund, out)
        if action == "stop":
            if runner.store.get(args.id) is None:
                _println(out, f"Mission #{args.id} unbekannt.")
                return 1
            _println(out, "Abbruch angefordert." if runner.stop(args.id) else "Mission läuft nicht (oder ist bereits beendet).")
            return 0
        if action == "loeschen":
            if runner.store.get(args.id) is None:
                _println(out, f"Mission #{args.id} unbekannt.")
                return 1
            runner.stop(args.id)
            runner.store.delete(args.id)
            _println(out, f"Mission #{args.id} gelöscht.")
            return 0
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        brain.close()


def _start_mission(brain: Brain, mission_id: int, background: bool, out: TextIO) -> int:
    runner = brain.missions
    if background:
        try:
            runner.start(mission_id)
        except RuntimeError as e:
            _println(out, f"Fehler: {e}")
            return 1
        _println(out, f"Mission #{mission_id} läuft im Hintergrund (nur solange dieser Prozess lebt) – "
                      f"Status: python -m obito mission status {mission_id}")
        join = getattr(runner, "join", None)
        if callable(join):
            join(mission_id, None)
        m = runner.store.get(mission_id)
        if m is not None:
            _println(out, fmt_mission(m, verbose=True))
        return 0 if m is not None and m.status == "fertig" else 1
    _println(out, f"Mission #{mission_id} startet …")
    try:
        m = run_mission_foreground(brain, mission_id, out)
    except (ValueError, RuntimeError) as e:
        _println(out, f"Fehler: {e}")
        return 1
    _println(out, fmt_mission(m, verbose=True))
    return 0 if m.status == "fertig" else 1


def cmd_automation(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito automation {list,neu,aktiv,inaktiv,jetzt,laeufe,log,loeschen,vorschlaege} …")
        return 2
    brain = _brain(cfg, backend)
    scheduler = brain.automation
    store = scheduler.store
    try:
        if action == "list":
            items = store.list()
            if not items:
                _println(out, "Keine Automationen. Vorschläge: python -m obito automation vorschlaege --installieren")
                return 0
            for a in items:
                _println(out, fmt_automation(a))
            _println(out, "Der Zeitplaner läuft, solange `python -m obito serve` läuft.")
            return 0
        if action == "neu":
            try:
                params = _parse_params(args.parameter)
                a = store.create(args.name, args.art, args.intervall, params, not args.inaktiv)
            except (ValueError, json.JSONDecodeError) as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Angelegt: {fmt_automation(a)}")
            return 0
        if action in ("aktiv", "inaktiv"):
            try:
                a = store.update(args.id, enabled=(action == "aktiv"))
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, fmt_automation(a))
            return 0
        if action == "jetzt":
            try:
                status, message = scheduler.run_once(args.id)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"[{status}] {message}")
            return 0 if status == "ok" else 1
        if action == "laeufe":
            if store.get(args.id) is None:
                _println(out, f"Automation #{args.id} unbekannt.")
                return 1
            runs = store.runs(args.id, 20)
            if not runs:
                _println(out, "Noch keine Läufe.")
                return 0
            for r in runs:
                _println(out, f"{_fmt_time(r.get('gestartet'))} [{r.get('status')}] {r.get('meldung')}")
            return 0
        if action == "log":
            lines = scheduler.log_tail(args.anzahl)
            if not lines:
                _println(out, "Noch kein Protokoll.")
                return 0
            for line in lines:
                _println(out, line.rstrip("\n"))
            return 0
        if action == "loeschen":
            if store.delete(args.id):
                _println(out, f"Automation #{args.id} gelöscht.")
                return 0
            _println(out, f"Automation #{args.id} unbekannt.")
            return 1
        if action == "vorschlaege":
            from .automation import default_automations, install_defaults
            if args.installieren:
                created = install_defaults(store)
                for a in created:
                    _println(out, f"Eingerichtet: {fmt_automation(a)}")
                if not created:
                    _println(out, "Alle Vorschläge sind bereits eingerichtet.")
                return 0
            for v in default_automations():
                _println(out, f"- {v['name']} [{v['kind']}, alle {v['interval_minutes']} min]: {v.get('beschreibung', '')}")
            _println(out, "Einrichten: python -m obito automation vorschlaege --installieren")
            return 0
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        brain.close()


def cmd_modelle(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito modelle {list,pull,loeschen,wechseln,empfehlen} … (--help für Details)")
        return 2
    if action == "empfehlen":
        from .cli import recommend_config
        rec = recommend_config(args.vram_gb, cfg)
        _println(out, f"Empfehlung für {args.vram_gb:g} GB VRAM:")
        _println(out, f"  Hauptmodell:    {rec.model}")
        _println(out, f"  Routing-Modell: {rec.fast_model or '(keins – Hauptmodell übernimmt)'}")
        _println(out, f"  Embedding:      {rec.embed_model}")
        _println(out, f"  num_ctx:        {rec.num_ctx}   parallel_calls: {rec.parallel_calls}   Tiefe: {rec.depth}")
        _println(out, f"Laden:  ollama pull {rec.model}" + (f" && ollama pull {rec.fast_model}" if rec.fast_model else "")
                      + f" && ollama pull {rec.embed_model}")
        _println(out, f"Vorlage schreiben: python -m obito config --empfehlen {args.vram_gb:g} --schreiben obito.json")
        return 0
    brain = _brain(cfg, backend)
    be = brain.backend
    try:
        if action == "list":
            try:
                models = be.list_models()
            except LLMError as e:
                _println(out, f"Fehler: {e}")
                return 1
            if not models:
                _println(out, "Keine Modelle installiert. Laden: python -m obito modelle pull qwen2.5:7b")
                return 0
            for m in models:
                _println(out, fmt_model(m, cfg, current=(m.name == cfg.model or m.name.split(":")[0] == cfg.model),
                                        fast=bool(cfg.fast_model) and (m.name == cfg.fast_model
                                                                       or m.name.split(":")[0] == cfg.fast_model)))
            running = be.running()
            if running:
                _println(out, "Geladen: " + ", ".join(f"{r['name']} ({_fmt_bytes(r.get('size_vram'))} VRAM)" for r in running))
            _println(out, "VRAM-Werte sind Schätzungen (Gewichte × 1,1 + KV-Cache + Reserve).")
            return 0
        if action == "pull":
            _println(out, f"Lade {args.name} …")
            try:
                ok = pull_with_progress(be, args.name, out)
            except LLMError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"{'Fertig' if ok else 'Nicht geladen'}: {args.name}")
            return 0 if ok else 1
        if action == "loeschen":
            try:
                ok = be.delete(args.name)
            except LLMError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Gelöscht: {args.name}" if ok else f"Modell {args.name} unbekannt oder Löschen nicht unterstützt.")
            return 0 if ok else 1
        if action == "wechseln":
            try:
                brain.set_model(args.name, fast=args.schnell)
            except ModelNotFound:
                _println(out, f"Fehler: Modell »{args.name}« ist nicht installiert – python -m obito modelle pull {args.name}")
                return 1
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            label = "Routing-Modell" if args.schnell else "Hauptmodell"
            _println(out, f"{label}: {args.name or '(keins)'}")
            if not args.nicht_speichern:
                target = persist_model_choice(cfg, out, getattr(args, "config", None))
                if target is not None:
                    _println(out, f"Gespeichert in {target}.")
            return 0
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        brain.close()


def cmd_rechner(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    from .tools import ToolRegistry
    reg = ToolRegistry(".")
    engineering.register_tools(reg)
    tool = (args.werkzeug or "").strip().lower()
    params = list(args.parameter or [])
    if not tool or tool == "liste":
        _println(out, "Rechner (Aufruf: python -m obito rechner <name> schluessel=wert …):")
        for t in reg.list():
            props = t.parameters.get("properties", {})
            req = set(t.parameters.get("required", []))
            sig = ", ".join(f"{k}{'' if k in req else '?'}" for k in props)
            _println(out, f"  {t.name}({sig})\n      {t.description}")
        _println(out, "Material: python -m obito rechner material <name> | vergleich <a,b,c>")
        return 0
    if tool == "material":
        name = " ".join(params).strip()
        if not name:
            _println(out, "Aufruf: python -m obito rechner material <name>")
            return 2
        mat = engineering.find_material(name)
        if mat is None:
            _println(out, f"Unbekanntes Material »{name}«. Bekannt: " + ", ".join(sorted(engineering.MATERIALS)))
            return 1
        _println(out, engineering.material_info(name))
        return 0
    if tool == "vergleich":
        try:
            mats = engineering.compare_materials(" ".join(params))
        except ValueError as e:
            _println(out, f"Fehler: {e}")
            return 1
        _println(out, engineering.material_table(mats))
        return 0
    if reg.get(tool) is None:
        _println(out, f"Unbekannter Rechner »{tool}«. Liste: python -m obito rechner liste")
        return 2
    try:
        kv = _parse_kv(params)
    except ValueError as e:
        _println(out, f"Fehler: {e}")
        return 2
    res = reg.run(tool, kv)
    if not res.ok:
        _println(out, f"Fehler: {res.error}")
        return 1
    _println(out, res.output)
    return 0


# ------------------------------------------------------------- Phase 3
def cmd_system(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    from . import sysmon
    from .llm import make_backend
    try:
        be = backend if backend is not None else make_backend(cfg)
    except Exception:  # noqa: BLE001 – Systemstatus auch ohne Modell-Backend
        be = None
    snap = sysmon.snapshot(backend=be, data_dir=str(cfg.data_path))
    if args.json:
        _println(out, json.dumps({"system": snap, "hinweise": sysmon.recommend(snap)}, ensure_ascii=False, indent=2,
                                 default=str))
        return 0
    _println(out, sysmon.format_snapshot(snap))
    return 0


def cmd_geraete(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    from . import devices as devmod
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito geraete {scan,list,lesen,notiz,vergessen} … (--help für Details)")
        return 2
    cfg.ensure_dirs()
    store = devmod.DeviceStore(cfg.devices_db)
    try:
        if action == "scan":
            found = devmod.scan()
            rows = store.update(found)
            if args.json:
                _println(out, json.dumps([d.to_dict() for d in found], ensure_ascii=False, indent=2))
            else:
                _println(out, devmod.format_scan(found, rows))
            return 0
        if action == "list":
            rows = store.list(connected_only=bool(args.verbunden))
            if not rows:
                _println(out, "Keine Geräte im Verlauf. Erkennen: python -m obito geraete scan")
                return 0
            for r in rows:
                _println(out, "  " + devmod.format_device(r))
            return 0
        if action == "lesen":
            try:
                result = devmod.read_serial(args.port, baud=cfg.serial_baud if args.baud is None else args.baud,
                                           seconds=args.sekunden)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            parsed = devmod.parse_telemetry(result["text"])
            _println(out, devmod.format_telemetry(result, parsed))
            return 0
        if action == "notiz":
            if store.get(args.key) is None:
                _println(out, f"Gerät »{args.key}« unbekannt – python -m obito geraete list zeigt die Schlüssel.")
                return 1
            store.note(args.key, " ".join(args.text))
            _println(out, "Notiz gespeichert.")
            return 0
        if action == "vergessen":
            ok = store.forget(args.key)
            _println(out, "Gelöscht." if ok else f"Gerät »{args.key}« unbekannt.")
            return 0 if ok else 1
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        store.close()


def _fmt_stats3d(st: dict) -> str:
    bb = st.get("bounding_box") or {}
    size = bb.get("groesse") or (0, 0, 0)
    masse = st.get("masse_g")
    return (f"Dreiecke {st.get('dreiecke')} · Volumen {engineering.fmt_number(float(st.get('volumen_cm3') or 0), 4)} cm³ · "
            f"Fläche {engineering.fmt_number(float(st.get('flaeche_cm2') or 0), 4)} cm² · "
            f"Maße {engineering.fmt_number(float(size[0]), 4)} × {engineering.fmt_number(float(size[1]), 4)} × "
            f"{engineering.fmt_number(float(size[2]), 4)} mm · wasserdicht {'ja' if st.get('wasserdicht') else 'nein'}"
            + (f" · Masse {engineering.fmt_number(float(masse), 4)} g" if masse is not None else ""))


def fmt_model3d(rec: dict) -> str:
    proj = f" [{rec['projekt']}]" if rec.get("projekt") else ""
    return (f"#{rec['id']} {rec['name']} v{rec['version']}{proj} ({rec.get('art') or 'import'}) – "
            + _fmt_stats3d(rec.get("statistik") or {}))


def cmd_modell3d(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    from . import geometry
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito modell3d {arten,neu,import,list,info,export,loeschen} … (--help für Details)")
        return 2
    if action == "arten":
        for key, spec in geometry.PRIMITIVES.items():
            _println(out, f"{key}: {spec['beschreibung']}")
            for name, desc, default in spec["parameter"]:
                d = "" if default is None else f" (Standard {default})"
                _println(out, f"    {name} – {desc}{d}")
        return 0
    cfg.ensure_dirs()
    store = geometry.ModelStore(cfg.models3d_db, cfg.models3d_dir)
    try:
        if action == "neu":
            try:
                params = _parse_kv(args.parameter)
                mesh = geometry.build(args.art, params)
                rec = store.save(mesh, args.name or args.art, kind=geometry.resolve_kind(args.art), params=params,
                                 project=args.projekt, material=args.material)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, "Erzeugt: " + fmt_model3d(rec))
            _println(out, f"Datei: {rec['datei']}")
            return 0
        if action == "import":
            try:
                rec = store.import_file(args.pfad, name=args.name, project=args.projekt)
            except (ValueError, FileNotFoundError, OSError) as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, "Importiert: " + fmt_model3d(rec))
            return 0
        if action == "list":
            rows = store.list(project=args.projekt)
            if not rows:
                _println(out, "Keine Modelle. Erzeugen: python -m obito modell3d neu quader l=100 b=50 h=10")
                return 0
            for r in rows:
                _println(out, "  " + fmt_model3d(r))
            return 0
        if action == "info":
            rec = store.get(args.id)
            if rec is None:
                _println(out, f"Modell {args.id} unbekannt.")
                return 1
            mesh = store.mesh(args.id)
            st = mesh.stats(args.material or rec.get("material"))
            _println(out, fmt_model3d({**rec, "statistik": st}))
            com = st.get("schwerpunkt") or (0, 0, 0)
            _println(out, f"  Schwerpunkt: ({engineering.fmt_number(float(com[0]), 4)}; {engineering.fmt_number(float(com[1]), 4)}; "
                          f"{engineering.fmt_number(float(com[2]), 4)}) mm")
            if rec.get("parameter"):
                _println(out, "  Parameter: " + ", ".join(f"{k}={v}" for k, v in rec["parameter"].items()))
            _println(out, f"  Datei: {rec['datei']}")
            return 0
        if action == "export":
            rec = store.get(args.id)
            if rec is None:
                _println(out, f"Modell {args.id} unbekannt.")
                return 1
            mesh = store.mesh(args.id)
            target = Path(args.ziel)
            try:
                if target.suffix.lower() == ".obj":
                    geometry.write_obj(mesh, target)
                else:
                    geometry.write_stl(mesh, target, binary=not args.ascii)
            except OSError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Geschrieben: {target}")
            return 0
        if action == "loeschen":
            ok = store.delete(args.id)
            _println(out, "Gelöscht." if ok else f"Modell {args.id} unbekannt.")
            return 0 if ok else 1
        _println(out, f"Unbekannte Aktion {action!r}.")
        return 2
    finally:
        store.close()


def cmd_simulation(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    from . import simulation
    if not args.art or args.art.lower() in ("liste", "arten", "list"):
        _println(out, "Simulationen (Aufruf: python -m obito simulation <art> name=wert …):")
        _println(out, simulation.describe())
        return 0
    try:
        params = _parse_kv(args.parameter)
        result = simulation.run(args.art, params)
    except ValueError as e:
        _println(out, f"Fehler: {e}")
        return 1
    if args.json:
        _println(out, json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _println(out, simulation.summary_text(result))
    if args.csv:
        axis = result.get("zeit_s")
        axis_name = "zeit_s"
        if axis is None:
            axis = result.get("x", [])
            axis_name = result.get("x_name", "x")
        names = list(result["reihen"].keys())
        try:
            with open(args.csv, "w", encoding="utf-8", newline="") as fh:
                fh.write(";".join([axis_name] + names) + "\n")
                for i, x in enumerate(axis):
                    row = [x] + [result["reihen"][n][i] for n in names]
                    fh.write(";".join(str(v).replace(".", ",") for v in row) + "\n")
        except OSError as e:
            _println(out, f"CSV nicht geschrieben: {e}")
            return 1
        _println(out, f"CSV geschrieben: {args.csv}")
    return 0


def cmd_geo(cfg: Config, args: argparse.Namespace, out: TextIO, inp, backend) -> int:
    from . import geo
    action = getattr(args, "aktion", None)
    if action is None:
        _println(out, "Aufruf: python -m obito geo {distanz,plan,sonne,utm,ort,orte,route,routen,loeschen,wetter} … "
                      "(--help für Details)")
        return 2
    cfg.ensure_dirs()
    store = geo.WaypointStore(cfg.geo_db)
    from .tools import ToolRegistry
    reg = ToolRegistry(".")
    geo.register_tools(reg, store, lambda: bool(cfg.online))
    try:
        if action == "distanz":
            res = reg.run("geo_distanz", {"von": args.von, "nach": args.nach})
        elif action == "plan":
            res = reg.run("geo_route", {"punkte": args.punkte, "geschwindigkeit_m_s": args.geschwindigkeit,
                                        "wind_kmh": args.wind, "wind_aus_deg": args.wind_aus})
        elif action == "sonne":
            res = reg.run("sonnenstand", {"ort": args.ort, "datum": args.zeit})
        elif action == "wetter":
            res = reg.run("wetter", {"ort": args.ort})
        elif action == "utm":
            try:
                lat, lon = geo.parse_coord(args.ort)
            except ValueError:
                place = store.find_place(args.ort)
                if place is None:
                    _println(out, f"»{args.ort}« ist weder Koordinate noch gespeicherter Ort.")
                    return 1
                lat, lon = place["lat"], place["lon"]
            try:
                u = geo.utm(lat, lon)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"{geo.format_coord(lat, lon)} → UTM {u['text']} (Zone {u['zone']}{u['band']}, "
                          f"Ostwert {u['ostwert_m']:.2f} m, Nordwert {u['nordwert_m']:.2f} m, {u['hemisphaere']})")
            return 0
        elif action == "ort":
            try:
                lat, lon = geo.parse_coord(args.koordinate)
                place = store.add_place(args.name, lat, lon, project=args.projekt, note=args.notiz)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Gespeichert: #{place['id']} {place['name']} {place['koordinate']}")
            return 0
        elif action == "orte":
            rows = store.list_places(project=args.projekt)
            if not rows:
                _println(out, "Keine Orte gespeichert. Anlegen: python -m obito geo ort <name> \"lat, lon\"")
                return 0
            for r in rows:
                proj = f" [{r['projekt']}]" if r.get("projekt") else ""
                _println(out, f"  #{r['id']} {r['name']}{proj} {r['koordinate']}" + (f" – {r['notiz']}" if r.get("notiz") else ""))
            return 0
        elif action == "route":
            chunks = [c for c in args.punkte.split(";") if c.strip()]
            pts = []
            for c in chunks:
                try:
                    pts.append(geo.parse_coord(c))
                except ValueError:
                    place = store.find_place(c.strip())
                    if place is None:
                        _println(out, f"»{c.strip()}« ist weder Koordinate noch gespeicherter Ort.")
                        return 1
                    pts.append((place["lat"], place["lon"]))
            try:
                route = store.add_route(args.name, pts, project=args.projekt)
            except ValueError as e:
                _println(out, f"Fehler: {e}")
                return 1
            _println(out, f"Gespeichert: #{route['id']} {route['name']} – {len(route['punkte'])} Punkte, "
                          f"{engineering.fmt_number(route['laenge_m'] / 1000.0, 4)} km")
            return 0
        elif action == "routen":
            rows = store.list_routes(project=args.projekt)
            if not rows:
                _println(out, "Keine Routen gespeichert.")
                return 0
            for r in rows:
                proj = f" [{r['projekt']}]" if r.get("projekt") else ""
                _println(out, f"  #{r['id']} {r['name']}{proj} – {len(r['punkte'])} Punkte, "
                              f"{engineering.fmt_number(r['laenge_m'] / 1000.0, 4)} km")
            return 0
        elif action == "loeschen":
            ok = store.delete_place(args.id) if args.art == "ort" else store.delete_route(args.id)
            _println(out, "Gelöscht." if ok else f"{args.art.capitalize()} {args.id} unbekannt.")
            return 0 if ok else 1
        else:
            _println(out, f"Unbekannte Aktion {action!r}.")
            return 2
        if not res.ok:
            _println(out, f"Fehler: {res.error}")
            return 1
        _println(out, res.output)
        return 0
    finally:
        store.close()


__all__ = [
    "COMMANDS", "add_parsers", "run", "fmt_document", "fmt_project", "fmt_note", "fmt_mission", "fmt_automation",
    "fmt_model", "model_vram_estimate", "pull_with_progress", "run_mission_foreground", "persist_model_choice",
    "fmt_model3d",
]
