"""Studio: KI-Figuren der Spielwelt nehmen Aufträge an und bauen Dinge.

Jede Figur ist eine Rolle mit eigenem System-Prompt. Ein Auftrag („Baue eine Webseite für meinen
Imbiss“) landet in einer Warteschlange; ein Hintergrund-Thread arbeitet sie ab, fragt das lokale
Modell und legt das Ergebnis als Datei unter ``<data_dir>/studio/<id>/`` ab.

* PIXEL – Webseiten (eine vollständige HTML-Datei)
* BYTE  – kleine Apps/Spiele (HTML + JavaScript in einer Datei)
* NOVA  – Pläne und Konzepte (Markdown, als HTML-Seite angezeigt)
* KERNEL – PC-Steuerung: erstellt nur einen Befehlsplan. Ausgeführt wird nichts – das ist die
  Schnittstelle für eine spätere „PC-Brücke“ mit ausdrücklicher Freigabe je Schritt.

Ist kein Modell erreichbar, baut die Figur eine ehrlich gekennzeichnete Vorlage statt zu scheitern.
Erzeugte Seiten werden vom Server in einer Sandbox (eigener Ursprung, kein API-Zugriff) ausgeliefert.
"""

from __future__ import annotations

import html
import json
import queue
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

STATUS = ("wartend", "arbeitet", "fertig", "fehler")
MAX_AUFTRAG = 2000
MAX_TOKENS_BAU = 4000


@dataclass(frozen=True)
class Figur:
    id: str
    name: str
    rolle: str
    art: str            # webseite | app | plan | pc
    farbe: str
    zone: str
    begruessung: str
    system: str

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "rolle": self.rolle, "art": self.art,
                "farbe": self.farbe, "zone": self.zone, "begruessung": self.begruessung}


_HTML_REGELN = (
    "Antworte NUR mit einer einzigen, vollständigen HTML-Datei (<!doctype html> … </html>). "
    "Alles inline (CSS in <style>, JavaScript in <script>), keine externen Dateien, keine CDNs, "
    "keine Netzwerkaufrufe. Modernes, sauberes Design, mobil nutzbar, Texte auf Deutsch."
)

FIGUREN: dict[str, Figur] = {f.id: f for f in (
    Figur("pixel", "PIXEL", "Webdesignerin", "webseite", "#ff6fb5", "web",
          "Hi! Ich baue dir Webseiten. Was soll drauf?",
          "Du bist PIXEL, eine erfahrene Webdesignerin. Du baust aus einem kurzen Auftrag eine "
          "fertige, ansprechende Webseite mit sinnvollen Abschnitten und echtem Beispieltext. " + _HTML_REGELN),
    Figur("byte", "BYTE", "App-Entwickler", "app", "#4fd1ff", "app",
          "Yo! Sag mir, welche App oder welches Mini-Spiel du willst.",
          "Du bist BYTE, ein App-Entwickler. Du baust aus einem kurzen Auftrag eine funktionierende "
          "Web-App oder ein kleines Spiel, das sofort im Browser läuft. Speichere Daten bei Bedarf "
          "in localStorage (mit try/catch). " + _HTML_REGELN),
    Figur("nova", "NOVA", "Planerin", "plan", "#ffd166", "plan",
          "Ich plane Projekte. Erzähl mir deine Idee!",
          "Du bist NOVA, eine Projektplanerin. Du machst aus einer Idee einen klaren Plan auf Deutsch "
          "in Markdown: Ziel, Funktionen, Schritte (nummeriert), Werkzeuge, Risiken, nächster Schritt. "
          "Antworte nur mit dem Markdown."),
    Figur("kernel", "KERNEL", "PC-Operator", "pc", "#7dff8a", "pc",
          "Ich steuere später deinen PC. Heute schreibe ich dir den Plan dafür.",
          "Du bist KERNEL, ein PC-Operator für Windows. Du führst NICHTS aus. Du erstellst einen "
          "Befehlsplan als JSON: {\"zusammenfassung\": str, \"schritte\": [{\"beschreibung\": str, "
          "\"befehl\": str, \"risiko\": \"niedrig\"|\"mittel\"|\"hoch\"}]}. Nur sichere, nachvollziehbare "
          "Schritte; nichts löschen ohne Hinweis. Antworte nur mit dem JSON."),
)}


# ------------------------------------------------------------------ Hilfen
_FENCE = re.compile(r"```(?:[a-zA-Z0-9_-]+)?\s*\n(.*?)```", re.S)


def extract_html(text: str) -> str | None:
    """Zieht ein HTML-Dokument aus einer Modellantwort (mit oder ohne Code-Zaun)."""
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for c in candidates:
        low = c.lower()
        start = low.find("<!doctype html")
        if start < 0:
            start = low.find("<html")
        end = low.rfind("</html>")
        if start >= 0 and end > start:
            return c[start:end + len("</html>")].strip()
    return None


def _strip_fence(text: str) -> str:
    m = _FENCE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()


def markdown_to_html(md: str) -> str:
    """Kleiner, sicherer Markdown-Umsetzer (Überschriften, Listen, fett, Code) – alles escaped."""
    out: list[str] = []
    liste: str | None = None

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        return s

    def close() -> None:
        nonlocal liste
        if liste:
            out.append(f"</{liste}>")
            liste = None

    for raw in (md or "").splitlines():
        line = raw.rstrip()
        m_h = re.match(r"^(#{1,4})\s+(.*)$", line)
        m_ul = re.match(r"^\s*[-*]\s+(.*)$", line)
        m_ol = re.match(r"^\s*\d+[.)]\s+(.*)$", line)
        if m_h:
            close()
            n = len(m_h.group(1))
            out.append(f"<h{n}>{inline(m_h.group(2))}</h{n}>")
        elif m_ul or m_ol:
            tag = "ul" if m_ul else "ol"
            if liste != tag:
                close()
                out.append(f"<{tag}>")
                liste = tag
            out.append(f"<li>{inline((m_ul or m_ol).group(1))}</li>")
        elif not line.strip():
            close()
        else:
            close()
            out.append(f"<p>{inline(line)}</p>")
    close()
    return "\n".join(out)


def _seite(titel: str, body: str, akzent: str = "#4fd1ff") -> str:
    return f"""<!doctype html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(titel)}</title>
<style>
body{{margin:0;font-family:system-ui,Segoe UI,Roboto,sans-serif;background:#0d1220;color:#e6edf7;line-height:1.55}}
main{{max-width:820px;margin:0 auto;padding:32px 18px}}
h1,h2,h3{{color:{akzent}}} code{{background:#1b2338;padding:1px 5px;border-radius:4px}}
.karte{{background:#151c2e;border:1px solid #26314d;border-radius:12px;padding:14px 16px;margin:12px 0}}
.hinweis{{color:#9fb0c9;font-size:.9em}}
</style></head><body><main>
{body}
</main></body></html>"""


def vorlage(figur: Figur, auftrag: str) -> str:
    """Ersatz-Ergebnis ohne Modell: klar als Vorlage gekennzeichnet."""
    t = html.escape(auftrag)
    hinweis = ("<p class='hinweis'>Vorlage ohne KI-Modell erstellt (Ollama nicht erreichbar). "
               "Starte Ollama und gib den Auftrag erneut, dann baut die Figur das echte Ergebnis.</p>")
    if figur.art == "app":
        body = (f"<h1>App-Entwurf</h1><div class='karte'><p>{t}</p></div>"
                "<div class='karte'><button id='b'>Klick mich</button> <span id='z'>0</span></div>"
                "<script>let n=0;document.getElementById('b').onclick=()=>{"
                "document.getElementById('z').textContent=++n}</script>" + hinweis)
    elif figur.art == "plan":
        body = (f"<h1>Plan</h1><div class='karte'><h2>Ziel</h2><p>{t}</p></div><div class='karte'><h2>Schritte</h2>"
                "<ol><li>Idee genauer beschreiben</li><li>Wichtigste Funktionen festlegen</li>"
                "<li>Ersten Prototyp bauen lassen</li><li>Testen und verbessern</li></ol></div>" + hinweis)
    elif figur.art == "pc":
        body = (f"<h1>PC-Befehlsplan</h1><div class='karte'><p>{t}</p></div>"
                "<p>Ohne Modell kann KERNEL keinen Plan erstellen.</p>" + hinweis)
    else:
        body = (f"<h1>{t[:80]}</h1><div class='karte'><h2>Über uns</h2><p>{t}</p></div>"
                "<div class='karte'><h2>Kontakt</h2><p>Hier kommen deine Kontaktdaten hin.</p></div>" + hinweis)
    return _seite(figur.name, body, figur.farbe)


def pc_plan_html(figur: Figur, auftrag: str, plan: Any) -> str:
    if not isinstance(plan, dict) or not isinstance(plan.get("schritte"), list):
        return _seite("PC-Befehlsplan", f"<h1>PC-Befehlsplan</h1><div class='karte'><pre>"
                      f"{html.escape(json.dumps(plan, ensure_ascii=False, indent=2) if plan else '–')}</pre></div>",
                      figur.farbe)
    rows = []
    for i, s in enumerate(plan["schritte"][:30], 1):
        if not isinstance(s, dict):
            continue
        rows.append(f"<div class='karte'><strong>{i}. {html.escape(str(s.get('beschreibung', '')))}</strong>"
                    f"<p><code>{html.escape(str(s.get('befehl', '')))}</code></p>"
                    f"<p class='hinweis'>Risiko: {html.escape(str(s.get('risiko', '?')))}</p></div>")
    body = (f"<h1>PC-Befehlsplan</h1><p>{html.escape(str(plan.get('zusammenfassung', auftrag)))}</p>"
            + "".join(rows) +
            "<p class='hinweis'>Nichts davon wurde ausgeführt. Die PC-Brücke (Ausführen mit Freigabe je "
            "Schritt) ist der nächste Ausbauschritt.</p>")
    return _seite("PC-Befehlsplan", body, figur.farbe)


# ------------------------------------------------------------------ Speicher
class StudioStore:
    """SQLite-Tabelle der Aufträge (thread-sicher über eine Sperre)."""

    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("""CREATE TABLE IF NOT EXISTS auftraege (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            figur TEXT NOT NULL,
            auftrag TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'wartend',
            datei TEXT,
            vorlage INTEGER NOT NULL DEFAULT 0,
            meldung TEXT NOT NULL DEFAULT '',
            erstellt REAL NOT NULL,
            fertig REAL)""")
        # Beim Neustart unterbrochene Aufträge wieder einreihen
        self._db.execute("UPDATE auftraege SET status='wartend' WHERE status='arbeitet'")
        self._db.commit()

    def add(self, figur: str, auftrag: str) -> dict:
        with self._lock:
            cur = self._db.execute("INSERT INTO auftraege (figur, auftrag, erstellt) VALUES (?, ?, ?)",
                                   (figur, auftrag, time.time()))
            self._db.commit()
            new_id = int(cur.lastrowid)
        return self.get(new_id)  # type: ignore[return-value]

    def update(self, auftrag_id: int, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE auftraege SET {cols} WHERE id=?", (*fields.values(), auftrag_id))
            self._db.commit()

    def get(self, auftrag_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM auftraege WHERE id=?", (auftrag_id,)).fetchone()
        return self._row(row) if row else None

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM auftraege ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [self._row(r) for r in rows]

    def pending(self) -> list[int]:
        with self._lock:
            rows = self._db.execute("SELECT id FROM auftraege WHERE status='wartend' ORDER BY id").fetchall()
        return [int(r["id"]) for r in rows]

    def delete(self, auftrag_id: int) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM auftraege WHERE id=?", (auftrag_id,))
            self._db.commit()
        return cur.rowcount > 0

    def stats(self) -> dict:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) AS n FROM auftraege GROUP BY status").fetchall()
        out = {s: 0 for s in STATUS}
        out.update({r["status"]: int(r["n"]) for r in rows})
        out["gesamt"] = sum(out[s] for s in STATUS)
        return out

    def close(self) -> None:
        with self._lock:
            self._db.close()

    @staticmethod
    def _row(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["vorlage"] = bool(d["vorlage"])
        d["vorschau"] = f"/studio/{d['id']}/" if d["status"] == "fertig" and d.get("datei") else None
        return d


# ------------------------------------------------------------------ Studio
class Studio:
    """Aufträge annehmen und im Hintergrund von den Figuren bauen lassen."""

    def __init__(self, store: StudioStore, out_dir: str | Path,
                 chat: Callable[[list[dict], int], str] | None = None, *, autostart: bool = True):
        self.store = store
        self.out_dir = Path(out_dir)
        self.chat = chat
        self._queue: "queue.Queue[int | None]" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.aktiv: dict[str, int] = {}          # figur -> laufender Auftrag
        for pid in store.pending():
            self._queue.put(pid)
        if autostart:
            self.start()

    # -------------------------------------------------------------- Ablauf
    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._thread = threading.Thread(target=self._loop, name="obito-studio", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        t = self._thread
        if t and t.is_alive():
            self._queue.put(None)
            t.join(timeout)

    def close(self) -> None:
        self.stop()
        self.store.close()

    def submit(self, figur_id: str, auftrag: str) -> dict:
        figur = FIGUREN.get((figur_id or "").strip().lower())
        if figur is None:
            raise ValueError(f"Unbekannte Figur: {figur_id}")
        auftrag = (auftrag or "").strip()
        if not auftrag:
            raise ValueError("Auftrag darf nicht leer sein")
        if len(auftrag) > MAX_AUFTRAG:
            raise ValueError(f"Auftrag zu lang (max. {MAX_AUFTRAG} Zeichen)")
        rec = self.store.add(figur.id, auftrag)
        self._queue.put(rec["id"])
        return rec

    def run_once(self, auftrag_id: int) -> dict | None:
        """Bearbeitet einen Auftrag sofort (auch für Tests)."""
        rec = self.store.get(auftrag_id)
        if rec is None or rec["status"] != "wartend":
            return rec
        figur = FIGUREN[rec["figur"]]
        self.aktiv[figur.id] = auftrag_id
        self.store.update(auftrag_id, status="arbeitet", meldung=f"{figur.name} arbeitet …")
        try:
            inhalt, ist_vorlage, meldung = self._bauen(figur, rec["auftrag"])
            ziel = self.out_dir / str(auftrag_id)
            ziel.mkdir(parents=True, exist_ok=True)
            datei = ziel / "index.html"
            datei.write_text(inhalt, encoding="utf-8")
            self.store.update(auftrag_id, status="fertig", datei=str(datei), vorlage=int(ist_vorlage),
                              meldung=meldung, fertig=time.time())
        except Exception as e:  # noqa: BLE001 – jeder Fehler landet sichtbar im Auftrag
            self.store.update(auftrag_id, status="fehler", meldung=f"Fehler: {e}", fertig=time.time())
        finally:
            self.aktiv.pop(figur.id, None)
        return self.store.get(auftrag_id)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            self.run_once(item)

    # -------------------------------------------------------------- Bauen
    def _bauen(self, figur: Figur, auftrag: str) -> tuple[str, bool, str]:
        if self.chat is None:
            return vorlage(figur, auftrag), True, "Vorlage (kein Modell angebunden)"
        messages = [{"role": "system", "content": figur.system},
                    {"role": "user", "content": f"Auftrag: {auftrag}"}]
        try:
            text = self.chat(messages, MAX_TOKENS_BAU)
        except Exception as e:  # noqa: BLE001 – Modell weg: Vorlage statt Absturz
            return vorlage(figur, auftrag), True, f"Vorlage – Modell nicht erreichbar ({e})"

        if figur.art in ("webseite", "app"):
            doc = extract_html(text)
            if doc is None:
                return vorlage(figur, auftrag), True, "Vorlage – Modell lieferte kein HTML"
            return doc, False, f"{figur.name} ist fertig."
        if figur.art == "plan":
            return (_seite("Plan", markdown_to_html(_strip_fence(text)), figur.farbe), False,
                    f"{figur.name} ist fertig.")
        # pc
        from .llm import parse_json
        try:
            plan = parse_json(text)
        except Exception:  # noqa: BLE001
            plan = None
        if plan is None:
            return _seite("PC-Befehlsplan", "<h1>PC-Befehlsplan</h1>" + markdown_to_html(_strip_fence(text)),
                          figur.farbe), False, "Plan als Text (kein gültiges JSON)"
        return pc_plan_html(figur, auftrag, plan), False, "Plan erstellt – nichts ausgeführt."

    # -------------------------------------------------------------- Abfragen
    def figuren(self) -> list[dict]:
        out = []
        for f in FIGUREN.values():
            d = f.to_dict()
            d["arbeitet_an"] = self.aktiv.get(f.id)
            out.append(d)
        return out

    def delete(self, auftrag_id: int) -> bool:
        if not self.store.delete(auftrag_id):
            return False
        import shutil
        shutil.rmtree(self.out_dir / str(int(auftrag_id)), ignore_errors=True)
        return True

    def datei(self, auftrag_id: int) -> Path | None:
        rec = self.store.get(auftrag_id)
        if not rec or rec["status"] != "fertig" or not rec.get("datei"):
            return None
        p = Path(rec["datei"]).resolve()
        base = self.out_dir.resolve()
        if base not in p.parents or not p.is_file():
            return None
        return p
