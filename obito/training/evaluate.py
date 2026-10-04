"""Eval-Set: messen statt raten.

Fragen mit Erwartung (``beispiele/eval_fragen.jsonl`` oder ``<datensatz>.fragen.jsonl`` aus
``learning.export_dataset``) werden einem Modell gestellt. Zwei Maße:

* **Stichwort-Score** (deterministisch, ohne Modell): Anteil erwarteter Stichworte in der
  Antwort, verbotene Begriffe und Pflichtzahlen.
* **Richter-Score** (optional): ein zweites Modell bewertet 0–10. Der Richter darf nicht das
  geprüfte Modell sein – sonst bewertet es sich selbst.

``compare`` stellt zwei Berichte gegenüber (A/B, z. B. Basis vs. eigenes LoRA-Modell), mit
paarweisem Richter in beiden Reihenfolgen; nur konsistente Urteile zählen.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from obito.llm import BackendUnavailable, LLMBackend, LLMError, parse_json
from obito.memory import normalize

ProgressCallback = Callable[[int, int, dict], None]

# ------------------------------------------------------------ Richter (lokal)
JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "punkte": {"type": "integer", "minimum": 0, "maximum": 10},
        "begruendung": {"type": "string"},
    },
    "required": ["punkte", "begruendung"],
}

PAIRWISE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "besser": {"type": "integer", "minimum": 0, "maximum": 2},
        "begruendung": {"type": "string"},
    },
    "required": ["besser", "begruendung"],
}

_JUDGE_RULES = (
    "Du bist ein strenger, fairer Prüfer für Antworten eines KI-Assistenten. Bewerte nur die "
    "fachliche Richtigkeit, Vollständigkeit und Nützlichkeit – nicht den Stil. Erfundene Fakten "
    "und falsche Zahlen führen zu wenigen Punkten. Antworte ausschließlich mit JSON."
)


def _local_judge_messages(frage: str, erwartet: str, antwort: str) -> list[dict]:
    ref = erwartet.strip() if erwartet and erwartet.strip() else "(keine Referenzantwort – bewerte Plausibilität und Vollständigkeit)"
    return [
        {"role": "system", "content": _JUDGE_RULES + "\n[OBITO:richter:system]"},
        {"role": "user", "content": (
            f"Frage:\n{frage.strip()}\n\nReferenzantwort:\n{ref}\n\nZu prüfende Antwort:\n{antwort.strip() or '(leer)'}\n\n"
            "Vergib 0–10 Punkte (10 = fachlich korrekt und vollständig, 0 = falsch oder leer). "
            'Antworte nur mit JSON. Beispiel: {"punkte": 7, "begruendung": "Zahlen korrekt, Nachteile fehlen."}'
        )},
    ]


def _local_pairwise_judge_messages(frage: str, erwartet: str, antwort_1: str, antwort_2: str) -> list[dict]:
    ref = erwartet.strip() if erwartet and erwartet.strip() else "(keine Referenzantwort – bewerte Plausibilität und Vollständigkeit)"
    return [
        {"role": "system", "content": _JUDGE_RULES + "\n[OBITO:richter:system]"},
        {"role": "user", "content": (
            f"Frage:\n{frage.strip()}\n\nReferenzantwort:\n{ref}\n\n"
            f"Antwort 1:\n{antwort_1.strip() or '(leer)'}\n\nAntwort 2:\n{antwort_2.strip() or '(leer)'}\n\n"
            "Welche Antwort ist fachlich besser? 1 = Antwort 1, 2 = Antwort 2, 0 = gleich gut. "
            'Antworte nur mit JSON. Beispiel: {"besser": 2, "begruendung": "Antwort 2 nennt die Einheiten korrekt."}'
        )},
    ]


_NUM_IN_TEXT = re.compile(r"-?\d+(?:[.,]\d+)?")
_TRUE_WORDS = {"ja", "true", "wahr", "1"}


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(round(value))
    if isinstance(value, str):
        m = _NUM_IN_TEXT.search(value)
        if m:
            try:
                return int(round(float(m.group(0).replace(",", "."))))
            except ValueError:
                return None
    return None


def _local_parse_judge(text: str) -> dict | None:
    data = parse_json(text)
    if not isinstance(data, dict):
        return None
    lowered = {str(k).lower(): v for k, v in data.items()}
    punkte = _to_int(lowered.get("punkte", lowered.get("score", lowered.get("bewertung"))))
    if punkte is None:
        return None
    begr = lowered.get("begruendung", lowered.get("begründung", lowered.get("grund", "")))
    return {"punkte": max(0, min(10, punkte)), "begruendung": str(begr or "").strip()}


def _local_parse_pairwise(text: str) -> dict | None:
    data = parse_json(text)
    if not isinstance(data, dict):
        return None
    lowered = {str(k).lower(): v for k, v in data.items()}
    raw = lowered.get("besser", lowered.get("winner", lowered.get("gewinner")))
    besser: int | None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("a", "1", "antwort 1", "antwort_1", "erste"):
            besser = 1
        elif s in ("b", "2", "antwort 2", "antwort_2", "zweite"):
            besser = 2
        elif s in ("0", "gleich", "unentschieden", "tie", "beide", "keine"):
            besser = 0
        else:
            besser = _to_int(s)
    else:
        besser = _to_int(raw)
    if besser not in (0, 1, 2):
        return None
    begr = lowered.get("begruendung", lowered.get("begründung", ""))
    return {"besser": besser, "begruendung": str(begr or "").strip()}


def _agents_attr(name: str, fallback):
    """Holt ``obito.agents.<name>``, wenn das Modul (schon) existiert, sonst die lokale Variante."""
    try:
        from obito import agents  # lazy: entsteht parallel
    except Exception:
        return fallback
    value = getattr(agents, name, None)
    return value if value is not None else fallback


# ------------------------------------------------------------- Items / Dateien
def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if str(v).strip()]
    return [str(value)]


def normalize_item(item: dict) -> dict:
    """Vereinheitlicht ein Eval-Item (Listen, Strings); ``ValueError`` bei fehlender Frage."""
    if not isinstance(item, dict):
        raise ValueError("Eval-Eintrag muss ein JSON-Objekt sein.")
    frage = item.get("frage")
    if not isinstance(frage, str) or not frage.strip():
        raise ValueError("Eval-Eintrag ohne »frage«.")
    out: dict = {"frage": frage.strip()}
    if item.get("erwartet") is not None:
        out["erwartet"] = str(item["erwartet"])
    out["stichworte"] = _as_str_list(item.get("stichworte"))
    out["verboten"] = _as_str_list(item.get("verboten"))
    if item.get("muss_zahl") is not None and str(item["muss_zahl"]).strip() != "":
        out["muss_zahl"] = str(item["muss_zahl"]).strip()
    if item.get("quelle_id") is not None:
        out["quelle_id"] = item["quelle_id"]
    return out


def load_items(path: str | Path) -> list[dict]:
    """Lädt Eval-Fragen aus JSONL (eine Zeile je Objekt) oder einem JSON-Array."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Eval-Datei nicht gefunden: {p}")
    text = p.read_text(encoding="utf-8-sig")
    items: list[dict] = []
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}: ungültiges JSON ({e}).") from None
        for i, raw in enumerate(data, 1):
            try:
                items.append(normalize_item(raw))
            except ValueError as e:
                raise ValueError(f"{p}, Eintrag {i}: {e}") from None
        return items
    for lineno, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{p}, Zeile {lineno}: ungültiges JSON ({e.msg}).") from None
        try:
            items.append(normalize_item(raw))
        except ValueError as e:
            raise ValueError(f"{p}, Zeile {lineno}: {e}") from None
    return items


def save_report(report: dict, path: str | Path) -> Path:
    """Schreibt einen Bericht als JSON (UTF-8, eingerückt); legt Elternverzeichnisse an."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_report(path: str | Path) -> dict:
    """Lädt einen mit :func:`save_report` geschriebenen Bericht."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("ergebnisse"), list):
        raise ValueError(f"{path}: kein Eval-Bericht (Schlüssel »ergebnisse« fehlt).")
    return data


# ------------------------------------------------------------ Stichwort-Score
def _parse_number(text: str) -> float | None:
    s = str(text).strip().replace(" ", "")
    if not s:
        return None
    # "1.500,5" (deutsch) -> 1500.5 ; "1,500.5" (englisch) -> 1500.5 ; "4,2" -> 4.2
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


_NUM_TOKEN = re.compile(r"-?\d+(?:[.,]\d+)*")
_GROUPED = re.compile(r"-?\d{1,3}(?:[.,]\d{3})+")


def _numbers_in(answer: str) -> list[float]:
    """Alle Zahlen einer Antwort. Mehrdeutige Gruppierungen wie ``1.500`` (1,5 oder 1500) liefern
    beide Lesarten, damit die Toleranzprüfung nicht an der Schreibweise scheitert."""
    out: list[float] = []
    for m in _NUM_TOKEN.finditer(answer):
        token = m.group(0)
        v = _parse_number(token)
        if v is not None:
            out.append(v)
        if _GROUPED.fullmatch(token):
            try:
                out.append(float(re.sub(r"[.,]", "", token)))
            except ValueError:
                pass
    return out


def _word_hits(word: str, tokens: set[str]) -> bool:
    """Treffer-Regel wie ``MemoryStore._fts_query``: Wörter ≥ 5 Zeichen als Präfix (ohne letztes
    Zeichen), kürzere exakt."""
    if len(word) >= 5:
        prefix = word[:-1]
        return any(t.startswith(prefix) for t in tokens)
    return word in tokens


def keyword_hit(keyword: str, answer: str, tokens: set[str] | None = None) -> bool:
    """Prüft ein einzelnes Stichwort (auch mehrwortig) gegen eine Antwort."""
    if tokens is None:
        tokens = set(normalize(answer).split())
    words = normalize(keyword).split()
    if not words:  # z. B. "% " -> nur Sonderzeichen: roher Teilstring-Vergleich
        return keyword.strip().lower() in answer.lower() if keyword.strip() else False
    if len(words) == 1:
        return _word_hits(words[0], tokens)
    # Mehrwort: alle Wörter müssen vorkommen, zusätzlich die Phrase in Reihenfolge
    if not all(_word_hits(w, tokens) for w in words):
        return False
    norm_answer = " " + normalize(answer) + " "
    phrase = " " + " ".join(words) + " "
    if phrase in norm_answer:
        return True
    # Präfix-Phrase: letztes Wort darf Flexionsendung haben ("return false" vs "return falsey")
    return (" " + " ".join(words[:-1]) + " " + (words[-1][:-1] if len(words[-1]) >= 5 else words[-1])) in norm_answer


def has_criteria(item: dict) -> bool:
    """Hat das Item etwas Messbares (Stichworte, erwartet, muss_zahl, verboten)?"""
    return bool(item.get("stichworte") or item.get("muss_zahl") or item.get("erwartet") or item.get("verboten"))


def expected_keywords(item: dict) -> list[str]:
    """Stichworte des Items; ohne ``stichworte`` die Inhaltswörter (≥ 5 Zeichen, max. 8) aus ``erwartet``."""
    kws = _as_str_list(item.get("stichworte"))
    if kws:
        return kws
    erwartet = item.get("erwartet")
    if isinstance(erwartet, str) and erwartet.strip():
        words = [w for w in normalize(erwartet).split() if len(w) >= 5]
        return list(dict.fromkeys(words))[:8]
    return []


def keyword_score(answer: str, item: dict) -> float:
    """Deterministischer Score 0–1.

    * ``verboten``: ein Treffer -> 0.
    * ``muss_zahl``: eine Zahl der Antwort muss innerhalb ±2 % liegen, sonst 0.
    * ``stichworte`` (sonst Inhaltswörter aus ``erwartet``): Anteil der Treffer; Wörter ≥ 5
      Zeichen zählen als Präfix-Treffer (``steif`` findet ``Steifigkeit``).
    Ohne jedes Kriterium -> 0 (nichts messbar).
    """
    answer = answer or ""
    tokens = set(normalize(answer).split())
    for bad in _as_str_list(item.get("verboten")):
        if keyword_hit(bad, answer, tokens):
            return 0.0
    muss = item.get("muss_zahl")
    has_number_rule = muss is not None and str(muss).strip() != ""
    if has_number_rule:
        wanted = _parse_number(str(muss))
        if wanted is None:
            raise ValueError(f"muss_zahl ist keine Zahl: {muss!r}")
        tol = max(abs(wanted) * 0.02, 1e-9)
        if not any(abs(n - wanted) <= tol for n in _numbers_in(answer)):
            return 0.0
    keywords = expected_keywords(item)
    if not keywords:
        return 1.0 if has_number_rule else 0.0
    hits = sum(1 for k in keywords if keyword_hit(k, answer, tokens))
    return round(hits / len(keywords), 4)


# ------------------------------------------------------------------- Eval
def _mean(values: Sequence[float]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _judge(backend: LLMBackend, judge_model: str, messages: list[dict], schema: dict,
           parser: Callable[[str], dict | None], seed: int | None, num_ctx: int | None) -> tuple[dict | None, str | None]:
    """Richteraufruf: Versuch 1 mit JSON-Schema, genau ein Retry mit ``json_mode=True``."""
    try:
        res = backend.chat(messages, model=judge_model, temperature=0.0, seed=seed, json_mode=schema,
                           max_tokens=400, num_ctx=num_ctx)
        parsed = parser(res.text)
        if parsed is None:
            retry = list(messages) + [{"role": "user", "content": (
                "Antworte ausschließlich mit einem JSON-Objekt nach diesem Schema: "
                + json.dumps(schema, ensure_ascii=False))}]
            res = backend.chat(retry, model=judge_model, temperature=0.0, seed=seed, json_mode=True,
                               max_tokens=400, num_ctx=num_ctx)
            parsed = parser(res.text)
    except BackendUnavailable:
        raise
    except LLMError as e:
        return None, f"Richter-Fehler: {e}"
    if parsed is None:
        return None, "Richter lieferte kein auswertbares JSON."
    return parsed, None


def run_eval(
    backend: LLMBackend,
    model: str,
    items: Iterable[dict],
    *,
    judge_model: str | None = None,
    system_prompt: str | None = None,
    temperature: float = 0.0,
    seed: int | None = 42,
    progress: ProgressCallback | None = None,
    through_brain: Any = None,
    depth: str | None = None,
    exclude_ids: frozenset | set | Sequence = frozenset(),
    num_ctx: int | None = None,
) -> dict:
    """Stellt alle Fragen und bewertet die Antworten.

    * Direkt: ``backend.chat([system?, user], model, temperature, seed, num_ctx)``.
    * ``through_brain``: ``brain.ask(frage, session_id="eval-<i>", depth=depth, learn=False)``
      – misst das ganze Gremium statt des nackten Modells.
    * ``judge_model``: zweites Modell bewertet 0–10; ist es das geprüfte Modell, bleibt der
      Richter aus (Warnung).
    * ``exclude_ids``: ``quelle_id``s, die im Training waren (werden übersprungen).
    * ``progress(done, total, ergebnis)`` nach jeder Frage.
    """
    excl = set(exclude_ids or ())
    all_items = [normalize_item(it) for it in items]
    todo = [it for it in all_items if it.get("quelle_id") is None or it["quelle_id"] not in excl]
    warnungen: list[str] = []
    skipped = len(all_items) - len(todo)
    if skipped:
        warnungen.append(f"{skipped} Frage(n) ausgeschlossen, weil sie in den Trainingsdaten vorkommen.")

    if through_brain is not None and not model:
        model = str(getattr(getattr(through_brain, "cfg", None), "model", "") or "")

    judge_active = bool(judge_model)
    if judge_model and judge_model == model:
        judge_active = False
        warnungen.append(
            f"Richter = geprüftes Modell ({model}) – Selbstbewertung ist wertlos, Richter deaktiviert. "
            "Nutze ein anderes Modell als Richter."
        )
    judge_messages = _agents_attr("judge_messages", _local_judge_messages)
    parse_judge = _agents_attr("parse_judge", _local_parse_judge)
    judge_schema = _agents_attr("JUDGE_SCHEMA", JUDGE_SCHEMA)

    ohne_kriterien = 0
    ergebnisse: list[dict] = []
    for i, item in enumerate(todo):
        frage = item["frage"]
        t0 = time.time()
        text = ""
        tokens = 0
        dauer: float | None = None
        try:
            if through_brain is not None:
                ans = through_brain.ask(frage, session_id=f"eval-{i}", depth=depth, learn=False)
                text = str(getattr(ans, "text", "") or "")
                tokens = int(getattr(ans, "tokens", 0) or 0)
                dauer = float(getattr(ans, "duration", 0.0) or 0.0) or None
                if getattr(ans, "depth", "") == "fehler":
                    warnungen.append(f"Frage {i + 1}: Fehler-Antwort des Denkkerns – {text[:160]}")
            else:
                messages: list[dict] = []
                if system_prompt and system_prompt.strip():
                    messages.append({"role": "system", "content": system_prompt.strip()})
                messages.append({"role": "user", "content": frage})
                res = backend.chat(messages, model=model, temperature=temperature, seed=seed, num_ctx=num_ctx)
                text = res.text or ""
                tokens = int(res.total_tokens or 0)
                dauer = float(res.duration or 0.0) or None
        except BackendUnavailable:
            raise
        except LLMError as e:
            warnungen.append(f"Frage {i + 1}: Modellfehler – {e}")
            text = ""
        if dauer is None:
            dauer = time.time() - t0

        if has_criteria(item):
            stichwort: float | None = keyword_score(text, item)
        else:
            stichwort = None
            ohne_kriterien += 1

        richter = None
        if judge_active and text.strip():
            msgs = judge_messages(frage, item.get("erwartet", ""), text)
            parsed, warn = _judge(backend, judge_model, msgs, judge_schema, parse_judge, seed, num_ctx)
            if parsed is not None:
                richter = {"punkte": int(parsed["punkte"]), "begruendung": str(parsed.get("begruendung", ""))}
            elif warn:
                warnungen.append(f"Frage {i + 1}: {warn}")
        elif judge_active:
            richter = {"punkte": 0, "begruendung": "Leere Antwort."}

        ergebnis = {
            "frage": frage,
            "antwort": text,
            "stichwort": stichwort,
            "richter": richter,
            "dauer": round(dauer, 3),
            "tokens": tokens,
        }
        if "erwartet" in item:
            ergebnis["erwartet"] = item["erwartet"]
        if "quelle_id" in item:
            ergebnis["quelle_id"] = item["quelle_id"]
        ergebnisse.append(ergebnis)
        if progress is not None:
            progress(i + 1, len(todo), ergebnis)

    if ohne_kriterien:
        warnungen.append(
            f"{ohne_kriterien} Frage(n) ohne Stichworte/erwartet/muss_zahl – dort kein Stichwort-Score."
        )
    if not todo:
        warnungen.append("Keine Fragen ausgewertet.")

    return {
        "modell": model,
        "tiefe": depth,
        "anzahl": len(ergebnisse),
        "stichwort_score": _mean([e["stichwort"] for e in ergebnisse if e["stichwort"] is not None]),
        "richter_score": _mean([e["richter"]["punkte"] for e in ergebnisse if e["richter"]]) if judge_active else None,
        "dauer_mittel": _mean([e["dauer"] for e in ergebnisse]),
        "tokens_mittel": _mean([e["tokens"] for e in ergebnisse]),
        "warnungen": warnungen,
        "ergebnisse": ergebnisse,
    }


# ---------------------------------------------------------------- Vergleich
def _pairwise_verdict(backend: LLMBackend, judge_model: str, frage: str, erwartet: str, a: str, b: str,
                      seed: int | None) -> tuple[str, str | None]:
    """Zwei Richteraufrufe mit vertauschter Reihenfolge; nur konsistente Urteile zählen."""
    pairwise_messages = _agents_attr("pairwise_judge_messages", _local_pairwise_judge_messages)
    parse_pairwise = _agents_attr("parse_pairwise", _local_parse_pairwise)
    schema = _agents_attr("PAIRWISE_SCHEMA", PAIRWISE_SCHEMA)
    first, w1 = _judge(backend, judge_model, pairwise_messages(frage, erwartet, a, b), schema, parse_pairwise, seed, None)
    second, w2 = _judge(backend, judge_model, pairwise_messages(frage, erwartet, b, a), schema, parse_pairwise, seed, None)
    if first is None or second is None:
        return "unklar", w1 or w2
    v1, v2 = first["besser"], second["besser"]
    if v1 == 1 and v2 == 2:
        return "A", None
    if v1 == 2 and v2 == 1:
        return "B", None
    if v1 == 0 and v2 == 0:
        return "gleich", None
    return "unklar", None   # Positions-Bias: Urteil kippt mit der Reihenfolge


def compare(report_a: dict, report_b: dict, *, backend: LLMBackend | None = None,
            judge_model: str | None = None, seed: int | None = 42) -> dict:
    """A/B-Vergleich zweier Berichte (gleiche Fragen).

    Deterministisch: Stichwort-, Dauer- und Token-Deltas (jeweils B − A). Mit ``backend`` und
    ``judge_model``: paarweiser Richter je Frage (zweimal, vertauscht; inkonsistente Urteile
    zählen als »gleich« und werden unter ``inkonsistent`` gezählt). ``hinweis`` warnt, wenn das
    Ergebnis nicht signifikant ist (n < 30 oder |siege_a − siege_b| ≤ √n).
    """
    rows_a = {r["frage"]: r for r in report_a.get("ergebnisse", []) if isinstance(r, dict) and r.get("frage")}
    rows_b = {r["frage"]: r for r in report_b.get("ergebnisse", []) if isinstance(r, dict) and r.get("frage")}
    common = [f for f in rows_a if f in rows_b]
    hints: list[str] = []
    missing = (len(rows_a) - len(common)) + (len(rows_b) - len(common))
    if missing:
        hints.append(f"{missing} Frage(n) kommen nur in einem Bericht vor und wurden übersprungen.")

    judge_active = False
    if judge_model and backend is None:
        hints.append("Richter angegeben, aber kein Backend – nur Stichwort-Vergleich.")
    elif backend is not None:
        judge_model = judge_model or getattr(backend, "default_model", "") or None
        if not judge_model:
            hints.append("Kein Richter-Modell – nur Stichwort-Vergleich.")
        elif judge_model in (report_a.get("modell"), report_b.get("modell")):
            hints.append(
                f"Richter = geprüftes Modell ({judge_model}) – Richter deaktiviert, nur Stichwort-Vergleich."
            )
        else:
            judge_active = True

    items: list[dict] = []
    siege_a = siege_b = gleich = inkonsistent = 0
    for frage in common:
        ra, rb = rows_a[frage], rows_b[frage]
        row = {
            "frage": frage,
            "a": ra.get("antwort", ""),
            "b": rb.get("antwort", ""),
            "stichwort_a": ra.get("stichwort"),
            "stichwort_b": rb.get("stichwort"),
            "richter": None,
        }
        if judge_active:
            verdict, warn = _pairwise_verdict(backend, judge_model, frage, str(ra.get("erwartet") or rb.get("erwartet") or ""),
                                              row["a"], row["b"], seed)
            if warn:
                hints.append(f"»{frage[:60]}«: {warn}")
            if verdict == "A":
                siege_a += 1
            elif verdict == "B":
                siege_b += 1
            else:
                gleich += 1
                if verdict == "unklar":
                    inkonsistent += 1
                verdict = "gleich"
            row["richter"] = verdict
        items.append(row)

    n = len(items)

    def _delta(key_a: str, key_b: str, src_a: dict, src_b: dict) -> float | None:
        va = [src_a[f].get(key_a) for f in common if src_a[f].get(key_a) is not None]
        vb = [src_b[f].get(key_b) for f in common if src_b[f].get(key_b) is not None]
        ma, mb = _mean(va), _mean(vb)
        return round(mb - ma, 4) if ma is not None and mb is not None else None

    stichwort_delta = _delta("stichwort", "stichwort", rows_a, rows_b)
    dauer_delta = _delta("dauer", "dauer", rows_a, rows_b)
    tokens_delta = _delta("tokens", "tokens", rows_a, rows_b)

    if judge_active:
        diff = abs(siege_a - siege_b)
        root = math.sqrt(n) if n else 0.0
        if n < 30:
            hints.append(f"Nicht signifikant: nur n={n} Fragen (mindestens 30 nötig).")
        elif diff <= root:
            hints.append(f"Nicht signifikant: |{siege_a}−{siege_b}| = {diff} ≤ √{n} ≈ {root:.1f}.")
        else:
            winner = "A" if siege_a > siege_b else "B"
            hints.append(f"Unterschied vermutlich signifikant: {winner} gewinnt {max(siege_a, siege_b)}:{min(siege_a, siege_b)} "
                         f"bei n={n} (Differenz {diff} > √n ≈ {root:.1f}).")
        if inkonsistent:
            hints.append(f"{inkonsistent} Urteil(e) kippten mit der Reihenfolge und zählen als »gleich«.")
    else:
        if not any("Richter" in h for h in hints):
            hints.append("Ohne Richter: nur deterministischer Stichwort-Vergleich (Siege werden nicht gezählt).")
        if n < 30:
            hints.append(f"Nicht signifikant: nur n={n} Fragen (mindestens 30 nötig).")

    return {
        "modell_a": report_a.get("modell"),
        "modell_b": report_b.get("modell"),
        "items": items,
        "siege_a": siege_a,
        "siege_b": siege_b,
        "gleich": gleich,
        "inkonsistent": inkonsistent,
        "stichwort_delta": stichwort_delta,
        "dauer_delta": dauer_delta,
        "tokens_delta": tokens_delta,
        "n": n,
        "hinweis": " ".join(hints),
    }
