"""LoRA-/QLoRA-Feintuning auf dem eigenen OBITO-Datensatz (Stufe 4 der Verbesserungsleiter).

Eingabe: ``datensatz.jsonl`` im Chat-Format aus ``python -m obito export-dataset``
(``{"messages": [{"role": "system"|"user"|"assistant", "content": …}, …]}``), optional
``<daten>.eval.jsonl``. Ausgabe: ein PEFT-Adapter (``adapter_model.safetensors`` + Konfiguration)
sowie ``obito_training.json`` mit den Trainingsfakten, damit ``modelfile.check_adapter_compat``
später die passende Basis prüfen kann.

Die schweren Abhängigkeiten (torch, transformers, peft, optional bitsandbytes/trl/datasets) werden
erst in :func:`main` importiert – ``--help``, :func:`load_jsonl`, :func:`build_example` und
:func:`preflight` funktionieren ohne sie und sind so testbar.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from obito.training.modelfile import BASE_MAP, TRAINING_INFO_FILE, hf_base_for, ollama_base_for

# ------------------------------------------------------------- Konstanten
MIN_EXAMPLES = 20          # darunter Abbruch
FEW_EXAMPLES = 100         # darunter Warnung
MIN_DPO_PAIRS = 100        # trl-DPO braucht genug Paare
QLORA_AUTO_VRAM_GB = 20.0  # unterhalb automatisch QLoRA (4-bit), wenn bitsandbytes da ist

#: Linear-Projektionen gängiger Architekturen (Fallback, wenn das Modell nicht gescannt werden kann)
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                          "qkv_proj", "gate_up_proj")

_INSTALL_HINTS: dict[str, str] = {
    "torch": "pip install torch  (CUDA-Variante: https://pytorch.org/get-started/locally/)",
    "transformers": "pip install transformers",
    "peft": "pip install peft",
    "bitsandbytes": "pip install bitsandbytes  (QLoRA/4-bit; unter Windows nur via WSL2 zuverlässig)",
    "trl": "pip install trl",
    "datasets": "pip install datasets",
}
_ALL_PACKAGES_HINT = "pip install torch transformers peft bitsandbytes   (für --dpo zusätzlich: pip install trl datasets)"


# ------------------------------------------------------------ Lazy-Import
def lazy_import(name: str):
    """Importiert ein optionales Paket; fehlt es, ``SystemExit`` mit deutschem Installationshinweis."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise SystemExit(
            f"Fehlendes Paket »{name}« – LoRA-Training ist ohne dieses Paket nicht möglich ({e}).\n"
            f"Installation: {_INSTALL_HINTS.get(name, 'pip install ' + name)}\n"
            f"Alles auf einmal: {_ALL_PACKAGES_HINT}"
        ) from None


def package_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ------------------------------------------------------------------ Daten
_ROLES = {"system", "user", "assistant"}


def _alpaca_to_messages(item: dict) -> list[dict] | None:
    instruction = item.get("instruction")
    output = item.get("output")
    if not isinstance(instruction, str) or not isinstance(output, str):
        return None
    user = instruction.strip()
    if isinstance(item.get("input"), str) and item["input"].strip():
        user = item["input"].strip() + "\n\n" + user
    return [{"role": "user", "content": user}, {"role": "assistant", "content": output.strip()}]


def load_jsonl(path: str | Path) -> list[dict]:
    """Lädt einen JSONL-Datensatz.

    Chat-Format (``messages``) wird geprüft (Rollen, letzte Nachricht vom Assistenten);
    Alpaca-Zeilen (``instruction``/``input``/``output``) werden in Chat-Form umgewandelt;
    DPO-Zeilen (``prompt``/``chosen``/``rejected``) bleiben unverändert. Fehlerhafte Zeilen
    führen zu ``ValueError`` mit Zeilennummer.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Datensatz nicht gefunden: {p}")
    out: list[dict] = []
    with p.open("r", encoding="utf-8-sig") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{p}, Zeile {lineno}: ungültiges JSON ({e.msg}).") from None
            if not isinstance(item, dict):
                raise ValueError(f"{p}, Zeile {lineno}: erwartet ein JSON-Objekt.")
            if "messages" not in item and "instruction" in item:
                msgs = _alpaca_to_messages(item)
                if msgs is None:
                    raise ValueError(f"{p}, Zeile {lineno}: Alpaca-Eintrag ohne instruction/output.")
                item = {"messages": msgs}
            if "messages" in item:
                msgs = item["messages"]
                if not isinstance(msgs, list) or len(msgs) < 2:
                    raise ValueError(f"{p}, Zeile {lineno}: »messages« braucht mindestens Nutzer- und Assistenten-Nachricht.")
                for m in msgs:
                    if not isinstance(m, dict) or m.get("role") not in _ROLES or not isinstance(m.get("content"), str):
                        raise ValueError(f"{p}, Zeile {lineno}: Nachricht braucht role (system/user/assistant) und content (Text).")
                if msgs[-1]["role"] != "assistant":
                    raise ValueError(f"{p}, Zeile {lineno}: letzte Nachricht muss vom Assistenten sein (Trainingsziel).")
                if not msgs[-1]["content"].strip():
                    raise ValueError(f"{p}, Zeile {lineno}: leere Assistenten-Antwort.")
            elif all(k in item for k in ("prompt", "chosen", "rejected")):
                for k in ("prompt", "chosen", "rejected"):
                    if not isinstance(item[k], str) or not item[k].strip():
                        raise ValueError(f"{p}, Zeile {lineno}: DPO-Feld »{k}« muss ein nicht-leerer Text sein.")
            else:
                raise ValueError(f"{p}, Zeile {lineno}: weder Chat- (messages), Alpaca- noch DPO-Format (prompt/chosen/rejected).")
            out.append(item)
    return out


def find_eval_path(data_path: str | Path, explicit: str | None = None) -> Path | None:
    """``--eval-daten`` oder ``<daten>.eval.jsonl`` (auch ``<stamm>.eval.jsonl``), falls vorhanden."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise FileNotFoundError(f"Eval-Datensatz nicht gefunden: {p}")
        return p
    p = Path(data_path)
    for cand in (Path(str(p) + ".eval.jsonl"), p.with_suffix(".eval.jsonl")):
        if cand.is_file() and cand != p:
            return cand
    return None


# ------------------------------------------------------------- Beispiele
def _ids(result: Any) -> list[int]:
    """Token-IDs aus dem Rückgabewert von ``apply_chat_template`` (Liste, verschachtelt oder dict)."""
    if result is None:
        return []
    if hasattr(result, "keys"):  # dict / BatchEncoding
        if "input_ids" not in result:
            raise ValueError("apply_chat_template lieferte ein Objekt ohne »input_ids«.")
        result = result["input_ids"]
    if hasattr(result, "tolist"):
        result = result.tolist()
    if isinstance(result, (list, tuple)) and result and isinstance(result[0], (list, tuple)):
        result = result[0]
    return [int(x) for x in result]


def _common_prefix_len(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_example(tokenizer: Any, messages: list[dict], max_len: int) -> dict | None:
    """Tokenisiert ein Chat-Beispiel; nur die letzte Assistenten-Antwort trägt Labels.

    ``prompt_ids = apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True)``,
    ``full_ids = apply_chat_template(messages, tokenize=True)``. Der Prompt-Anteil ist der
    gemeinsame Präfix beider Sequenzen (Labels ``-100``). Ist das Beispiel länger als ``max_len``,
    wird **von links** gekürzt (alter Verlauf) – nie das Ziel; passt das Ziel nicht vollständig,
    wird ``None`` geliefert.
    """
    if not messages or messages[-1].get("role") != "assistant":
        return None
    if max_len < 1:
        raise ValueError("max_len muss mindestens 1 Token betragen.")
    prompt_ids = _ids(tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True))
    full_ids = _ids(tokenizer.apply_chat_template(messages, tokenize=True))
    if not full_ids:
        return None
    if full_ids[:len(prompt_ids)] == prompt_ids:
        prompt_len = len(prompt_ids)
    else:
        prompt_len = _common_prefix_len(prompt_ids, full_ids)
    if prompt_len >= len(full_ids):
        return None  # kein Ziel-Token übrig

    labels = [-100] * prompt_len + full_ids[prompt_len:]
    target_len = len(full_ids) - prompt_len
    if target_len > max_len:
        return None  # das Ziel allein passt nicht – Kürzen würde das Ziel beschneiden
    if len(full_ids) > max_len:
        cut = len(full_ids) - max_len
        full_ids = full_ids[cut:]
        labels = labels[cut:]
    if all(l == -100 for l in labels):
        return None
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


# -------------------------------------------------------------- Preflight
_PARAM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*[bB](?![a-zA-Z])")


def params_billion(name: str | None) -> float | None:
    """Parameterzahl in Milliarden aus einem Modellnamen (``Qwen2.5-7B-Instruct`` -> 7.0)."""
    if not name:
        return None
    base = os.path.basename(str(name).rstrip("/\\"))
    best: float | None = None
    for m in _PARAM_RE.finditer(base):
        try:
            v = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if 0.1 <= v <= 1000 and (best is None or v > best):
            best = v
    return best


def vram_need_gb(params_b: float | None, qlora: bool) -> float | None:
    """Grober Bedarf fürs Training: 4-bit (QLoRA) ≈ 7B 10 GB, 3B 6 GB, 1.5B 4,5 GB;
    bf16-LoRA ≈ 2,2 GB je Milliarde + 4 GB."""
    if params_b is None:
        return None
    if qlora:
        return round(max(3.0, params_b * 1.0 + 3.0), 1)
    return round(params_b * 2.2 + 4.0, 1)


def preflight(
    n_train: int,
    n_eval: int,
    *,
    cuda: bool,
    vram_gb: float | None,
    params_b: float | None,
    qlora: bool,
    bitsandbytes: bool,
    force: bool = False,
) -> dict:
    """Prüft vor dem Laden, ob das Training Sinn hat – ohne torch, alle Fakten kommen als Parameter.

    Rückgabe ``{"ok", "qlora", "bedarf_gb", "warnungen", "fehler"}``. Regeln: CPU-only -> Abbruch;
    QLoRA automatisch bei VRAM < 20 GB und vorhandenem bitsandbytes; Bedarf > VRAM -> Abbruch;
    < 20 Beispiele -> Abbruch, < 100 -> Warnung. ``force`` (``--erzwingen``) macht aus Abbrüchen
    Warnungen – außer bei einem leeren Datensatz.
    """
    warnungen: list[str] = []
    fehler: list[str] = []

    if n_train <= 0:
        return {"ok": False, "qlora": qlora, "bedarf_gb": None, "warnungen": warnungen,
                "fehler": ["Der Datensatz ist leer – nichts zu trainieren."]}
    if n_train < MIN_EXAMPLES:
        fehler.append(
            f"Nur {n_train} Trainingsbeispiele – unter {MIN_EXAMPLES} lernt ein LoRA praktisch nur auswendig. "
            "Sammle mehr Feedback/Korrekturen (Stufen 1–3) und exportiere erneut."
        )
    elif n_train < FEW_EXAMPLES:
        warnungen.append(
            f"Nur {n_train} Trainingsbeispiele (empfohlen ≥ {FEW_EXAMPLES}). Erwartung: kleine Stil-/Formatgewinne, "
            "wenig Wissen. Niedrige Lernrate und wenige Epochen beibehalten."
        )
    if n_eval <= 0:
        warnungen.append("Keine Eval-Daten – Überanpassung ist während des Trainings nicht erkennbar.")

    if not cuda:
        fehler.append(
            "Keine CUDA-GPU gefunden – LoRA-Training auf der CPU dauert Tage und ist nicht sinnvoll. "
            "Empfehlung: NVIDIA-GPU mit ≥ 12 GB VRAM (Windows: WSL2 + CUDA-Treiber) oder bei Stufe 1–3 "
            "(Lektionen, Beispiele, Eval) bleiben, die ohne Training wirken."
        )
        vram_gb = None

    if qlora and not bitsandbytes:
        fehler.append(f"--qlora verlangt bitsandbytes. {_INSTALL_HINTS['bitsandbytes']}")
    elif not qlora and cuda and vram_gb is not None and vram_gb < QLORA_AUTO_VRAM_GB:
        if bitsandbytes:
            qlora = True
            warnungen.append(
                f"QLoRA (4-bit) automatisch aktiviert: {vram_gb:.1f} GB VRAM < {QLORA_AUTO_VRAM_GB:.0f} GB."
            )
        else:
            warnungen.append(
                f"{vram_gb:.1f} GB VRAM ohne bitsandbytes: volles bf16-LoRA braucht deutlich mehr Speicher. "
                f"Empfehlung: {_INSTALL_HINTS['bitsandbytes']}"
            )

    bedarf = vram_need_gb(params_b, qlora)
    if params_b is None:
        warnungen.append("Modellgröße aus dem Namen nicht erkennbar – VRAM-Bedarf nicht schätzbar.")
    elif cuda and vram_gb is not None and bedarf is not None and bedarf > vram_gb:
        fehler.append(
            f"VRAM reicht nicht: Basis mit ~{params_b:g}B Parametern braucht {'in 4-bit ' if qlora else ''}etwa "
            f"{bedarf:.0f} GB, vorhanden sind {vram_gb:.1f} GB. Kleinere Basis wählen (z. B. qwen2.5:3b → "
            f"~{vram_need_gb(3.0, True):.0f} GB, qwen2.5:1.5b → ~{vram_need_gb(1.5, True):.0f} GB)"
            + ("" if qlora else " oder --qlora nutzen") + "."
        )

    if force and fehler:
        warnungen.extend(f"(erzwungen) {f}" for f in fehler)
        fehler = []
    return {"ok": not fehler, "qlora": qlora, "bedarf_gb": bedarf, "warnungen": warnungen, "fehler": fehler}


# ------------------------------------------------------------- Basis-Auflösung
def resolve_base(basis: str) -> tuple[str, str | None]:
    """``--basis`` -> (HF-ID oder lokaler Pfad, Ollama-Tag oder None)."""
    b = (basis or "").strip()
    if not b:
        raise ValueError("Keine Basis: --basis HF-ID, Pfad oder Ollama-Tag angeben (z. B. qwen2.5:7b).")
    if os.path.isdir(b):
        return b, ollama_base_for(b)
    hf = hf_base_for(b)
    if hf:
        tag = next((t for t, h in BASE_MAP.items() if h == hf), b)
        return hf, tag
    if ":" in b and "/" not in b:
        raise ValueError(
            f"Ollama-Tag »{b}« ist nicht bekannt. Gib die Hugging-Face-ID direkt an (z. B. Qwen/Qwen2.5-7B-Instruct) "
            "– der Adapter passt später nur zu genau diesem Modell."
        )
    return b, ollama_base_for(b)


# ------------------------------------------------------------- Routen-Hinweis
def route_hints(ausgabe: str, hf_base: str, ollama_base: str | None, merged_dir: str | None = None) -> str:
    """Die beiden Wege vom Adapter zum Ollama-Modell (wird am Ende ausgegeben)."""
    tag = ollama_base or "<ollama-basis>"
    merged = merged_dir or os.path.join(ausgabe, "merged")
    return (
        "\nNächste Schritte – zwei Routen:\n"
        "(A) Adapter als GGUF (klein, Basis bleibt in Ollama):\n"
        f"    python convert_lora_to_gguf.py {ausgabe} --base {hf_base} --outfile adapter.gguf\n"
        f"    python -m obito modelfile --name obito-v1 --basis {tag} --adapter adapter.gguf --erstellen\n"
        "(B) Zusammengeführtes Modell (--zusammenfuehren; eigenständige GGUF-Datei):\n"
        f"    python convert_hf_to_gguf.py {merged} --outfile merged.f16.gguf\n"
        "    llama-quantize merged.f16.gguf merged.q4_k_m.gguf Q4_K_M\n"
        "    python -m obito modelfile --name obito-v1 --basis ./merged.q4_k_m.gguf --erstellen\n"
        "Die Skripte convert_*_to_gguf.py und llama-quantize stammen aus llama.cpp "
        "(https://github.com/ggml-org/llama.cpp). Ein Adapter passt nur zum exakt gleichen Basismodell.\n"
    )


# ---------------------------------------------------------------- argparse
def build_parser(default_base: str | None) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m obito train",
        description="LoRA-/QLoRA-Feintuning auf dem eigenen OBITO-Datensatz (lokal, NVIDIA-GPU nötig).",
        epilog="Beispiel: python -m obito train --daten ~/.obito/datensaetze/obito.jsonl --ausgabe ./lora-v1",
    )
    p.add_argument("--basis", default=default_base,
                   help="Basismodell: Hugging-Face-ID, lokaler Pfad oder Ollama-Tag (Standard: aus cfg.model abgeleitet%s)"
                        % (f": {default_base}" if default_base else ""))
    p.add_argument("--daten", required=True, help="Trainingsdaten (JSONL, Chat-Format aus export-dataset)")
    p.add_argument("--eval-daten", default=None, help="Eval-Daten (JSONL); Standard: <daten>.eval.jsonl, falls vorhanden")
    p.add_argument("--ausgabe", default="./obito-lora", help="Zielverzeichnis für den Adapter (Standard: ./obito-lora)")
    p.add_argument("--epochen", type=float, default=3, help="Trainingsepochen (Standard: 3)")
    p.add_argument("--lr", type=float, default=1e-4, help="Lernrate (Standard: 1e-4)")
    p.add_argument("--rang", type=int, default=16, help="LoRA-Rang r (Standard: 16)")
    p.add_argument("--alpha", type=int, default=32, help="LoRA-Alpha (Standard: 32)")
    p.add_argument("--max-laenge", type=int, default=1024, help="Maximale Sequenzlänge in Tokens (Standard: 1024)")
    p.add_argument("--batch", type=int, default=1, help="Batch-Größe je Schritt (Standard: 1)")
    p.add_argument("--grad-akkum", type=int, default=8, help="Gradienten-Akkumulation (Standard: 8)")
    p.add_argument("--qlora", action="store_true",
                   help="4-bit-Basis (QLoRA, braucht bitsandbytes); automatisch bei < 20 GB VRAM")
    p.add_argument("--zusammenfuehren", action="store_true",
                   help="Adapter nach dem Training in die Basis einrechnen (<ausgabe>/merged, Route B)")
    p.add_argument("--dpo", action="store_true",
                   help="DPO-Training (trl) auf Paaren prompt/chosen/rejected; braucht mindestens 100 Paare")
    p.add_argument("--erzwingen", action="store_true", help="Preflight-Abbrüche ignorieren (auf eigene Gefahr)")
    return p


def _default_base() -> str | None:
    try:
        from obito.config import load_config
        return hf_base_for(load_config().model)
    except Exception:
        return None


# ------------------------------------------------------------------ Hilfen
def _say(msg: str) -> None:
    print(msg, flush=True)


def _warn(msg: str) -> None:
    print(f"Warnung: {msg}", file=sys.stderr, flush=True)


def chat_template_hash(tokenizer: Any) -> str:
    tmpl = getattr(tokenizer, "chat_template", None) or ""
    if not isinstance(tmpl, str):
        tmpl = json.dumps(tmpl, sort_keys=True, default=str)
    return hashlib.sha1(tmpl.encode("utf-8")).hexdigest()[:16]


def write_training_info(ausgabe: str | Path, info: dict) -> Path:
    p = Path(ausgabe) / TRAINING_INFO_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def linear_module_names(model: Any, torch_mod: Any) -> list[str]:
    """Alle Linear-Projektionen (Blattnamen) außer ``lm_head`` – Ziel der LoRA-Matrizen."""
    names: set[str] = set()
    linear_cls = getattr(torch_mod.nn, "Linear", None)
    for full_name, module in model.named_modules():
        cls_name = type(module).__name__
        is_linear = (linear_cls is not None and isinstance(module, linear_cls)) or "Linear" in cls_name
        if not is_linear:
            continue
        leaf = full_name.rsplit(".", 1)[-1]
        if leaf in ("lm_head", "embed_out", "score") or "lm_head" in full_name:
            continue
        names.add(leaf)
    return sorted(names) or list(DEFAULT_TARGET_MODULES)


class _ListDataset:
    """Minimaler Datensatz für ``transformers.Trainer`` (kein ``datasets``-Paket nötig)."""

    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        return self.rows[idx]


def make_collator(pad_id: int, torch_mod: Any):
    def collate(batch: list[dict]) -> dict:
        width = max(len(b["input_ids"]) for b in batch)
        ids, mask, labels = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * pad)
            mask.append(b["attention_mask"] + [0] * pad)
            labels.append(b["labels"] + [-100] * pad)
        return {
            "input_ids": torch_mod.tensor(ids, dtype=torch_mod.long),
            "attention_mask": torch_mod.tensor(mask, dtype=torch_mod.long),
            "labels": torch_mod.tensor(labels, dtype=torch_mod.long),
        }
    return collate


class LossTracker:
    """Sammelt Train-/Eval-Loss je Epoche, meldet auf Deutsch und warnt bei Überanpassung."""

    def __init__(self, epochs: float):
        self.epochs = epochs
        self.train_losses: list[float] = []
        self.eval_losses: list[float] = []
        self.warned = False

    def on_train_log(self, loss: float, epoch: float) -> None:
        self.train_losses.append(loss)
        _say(f"  Epoche {epoch:.2f}/{self.epochs:g} – Trainings-Loss {loss:.4f}")

    def on_eval(self, loss: float, epoch: float) -> str | None:
        prev = self.eval_losses[-1] if self.eval_losses else None
        self.eval_losses.append(loss)
        _say(f"  Epoche {epoch:.2f}/{self.epochs:g} – Eval-Loss {loss:.4f}")
        if prev is not None and loss > prev + 1e-4:
            self.warned = True
            msg = (f"Eval-Loss steigt ({prev:.4f} → {loss:.4f}) – Überanpassung. Das beste Modell "
                   "wird am Ende geladen; weniger Epochen oder mehr Daten empfohlen.")
            _warn(msg)
            return msg
        return None


def _make_callback(transformers_mod: Any, tracker: LossTracker):
    class _GermanLog(transformers_mod.TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            epoch = float(logs.get("epoch", state.epoch or 0.0) or 0.0)
            if "loss" in logs:
                tracker.on_train_log(float(logs["loss"]), epoch)
            if "eval_loss" in logs:
                tracker.on_eval(float(logs["eval_loss"]), epoch)
    return _GermanLog()


def _training_arguments(transformers_mod: Any, **kw):
    """``TrainingArguments`` versionsfest: ``eval_strategy`` (neu) vs. ``evaluation_strategy`` (alt)."""
    try:
        return transformers_mod.TrainingArguments(**kw)
    except TypeError:
        if "eval_strategy" in kw:
            kw["evaluation_strategy"] = kw.pop("eval_strategy")
            return transformers_mod.TrainingArguments(**kw)
        raise


# -------------------------------------------------------------------- main
def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser(_default_base())
    args = parser.parse_args(argv)

    # 1) Daten (ohne schwere Pakete prüfbar)
    try:
        hf_base, ollama_base = resolve_base(args.basis or "")
        train_rows = load_jsonl(args.daten)
        eval_path = find_eval_path(args.daten, args.eval_daten)
        eval_rows = load_jsonl(eval_path) if eval_path else []
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"Fehler: {e}") from None
    if not train_rows:
        raise SystemExit(f"Fehler: Der Datensatz {args.daten} ist leer – nichts zu trainieren.")
    if args.dpo:
        pairs = [r for r in train_rows if all(k in r for k in ("prompt", "chosen", "rejected"))]
        if len(pairs) < MIN_DPO_PAIRS:
            raise SystemExit(
                f"Fehler: DPO braucht mindestens {MIN_DPO_PAIRS} Paare (prompt/chosen/rejected), gefunden: {len(pairs)}. "
                "Exportiere mit --format dpo und sammle mehr Korrekturen – oder trainiere ohne --dpo."
            )
        train_rows = pairs
    elif any("messages" not in r for r in train_rows):
        raise SystemExit("Fehler: Für LoRA-Training wird das Chat-Format (messages) benötigt; "
                         "DPO-Paare nur mit --dpo. Exportiere mit --format chat.")
    for a in ("epochen", "lr", "rang", "alpha", "max_laenge", "batch", "grad_akkum"):
        if getattr(args, a) <= 0:
            raise SystemExit(f"Fehler: --{a.replace('_', '-')} muss positiv sein.")

    # 2) Pakete
    torch = lazy_import("torch")
    transformers = lazy_import("transformers")
    peft = lazy_import("peft")
    if args.dpo:
        trl = lazy_import("trl")
        datasets = lazy_import("datasets")

    # 3) Preflight mit echten Fakten
    cuda = bool(torch.cuda.is_available())
    vram_gb: float | None = None
    if cuda:
        try:
            _free, total = torch.cuda.mem_get_info()
            vram_gb = total / 1e9
        except Exception as e:  # pragma: no cover – treiberabhängig
            _warn(f"VRAM nicht ermittelbar ({e}).")
    check = preflight(len(train_rows), len(eval_rows), cuda=cuda, vram_gb=vram_gb,
                      params_b=params_billion(hf_base), qlora=args.qlora,
                      bitsandbytes=package_available("bitsandbytes"), force=args.erzwingen)
    for w in check["warnungen"]:
        _warn(w)
    if not check["ok"]:
        raise SystemExit("Abbruch vor dem Laden:\n- " + "\n- ".join(check["fehler"])
                         + "\n(--erzwingen übergeht diese Prüfung.)")
    use_qlora = check["qlora"]
    _say(f"Basis: {hf_base}" + (f" (Ollama: {ollama_base})" if ollama_base else ""))
    _say(f"Beispiele: {len(train_rows)} Training, {len(eval_rows)} Eval · GPU: "
         + (f"{torch.cuda.get_device_name(0)}, {vram_gb:.1f} GB" if cuda and vram_gb else ("ja" if cuda else "nein"))
         + f" · QLoRA: {'ja' if use_qlora else 'nein'}")

    # 4) Tokenizer + Modell
    bf16 = bool(cuda and torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else torch.float16
    _say(f"Lade Tokenizer und Modell ({'bf16' if bf16 else 'fp16'}) …")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_base)
    except Exception as e:
        raise SystemExit(f"Tokenizer von »{hf_base}« konnte nicht geladen werden: {e}\n"
                         "Hinweis: Gated Modelle (Llama, Gemma) brauchen `huggingface-cli login`.") from None
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kw: dict[str, Any] = {"torch_dtype": dtype, "device_map": {"": 0} if cuda else None}
    if use_qlora:
        load_kw["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(hf_base, **load_kw)
    except Exception as e:
        raise SystemExit(f"Modell »{hf_base}« konnte nicht geladen werden: {e}") from None
    model.config.use_cache = False
    if use_qlora:
        model = peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    targets = linear_module_names(model, torch)
    lora_cfg = peft.LoraConfig(r=args.rang, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
                               task_type="CAUSAL_LM", target_modules=targets)
    model = peft.get_peft_model(model, lora_cfg)
    _say(f"LoRA-Ziele: {', '.join(targets)}")
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    ausgabe = str(Path(args.ausgabe))
    Path(ausgabe).mkdir(parents=True, exist_ok=True)
    tracker = LossTracker(args.epochen)
    t0 = time.time()

    common_args: dict[str, Any] = dict(
        output_dir=ausgabe, num_train_epochs=args.epochen, learning_rate=args.lr,
        per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_akkum, gradient_checkpointing=True,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.0,
        logging_strategy="epoch", save_strategy="epoch" if eval_rows else "no", save_total_limit=2,
        eval_strategy="epoch" if eval_rows else "no",
        load_best_model_at_end=bool(eval_rows), metric_for_best_model="eval_loss" if eval_rows else None,
        greater_is_better=False if eval_rows else None,
        bf16=bf16, fp16=not bf16 and cuda, optim="paged_adamw_8bit" if use_qlora else "adamw_torch",
        report_to=[], remove_unused_columns=False, seed=42, dataloader_pin_memory=False,
    )

    if args.dpo:
        _say("DPO-Training (trl) …")
        train_ds = datasets.Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in train_rows])
        eval_pairs = [r for r in eval_rows if all(k in r for k in ("prompt", "chosen", "rejected"))]
        eval_ds = datasets.Dataset.from_list([{k: r[k] for k in ("prompt", "chosen", "rejected")} for r in eval_pairs]) or None
        if eval_ds is not None and len(eval_ds) == 0:
            eval_ds = None
        dpo_kw = dict(common_args)
        dpo_kw.update(beta=0.1, max_length=args.max_laenge, max_prompt_length=max(64, args.max_laenge // 2))
        try:
            dpo_config = trl.DPOConfig(**dpo_kw)
        except TypeError:
            for k in ("max_length", "max_prompt_length", "beta", "eval_strategy"):
                dpo_kw.pop(k, None)
            dpo_config = trl.DPOConfig(**dpo_kw)
        trainer = trl.DPOTrainer(model=model, ref_model=None, args=dpo_config, train_dataset=train_ds,
                                 eval_dataset=eval_ds, processing_class=tokenizer)
        trainer.add_callback(_make_callback(transformers, tracker))
        n_train, n_eval = len(train_ds), len(eval_ds) if eval_ds is not None else 0
    else:
        _say("Tokenisiere Beispiele …")
        train_ex = [build_example(tokenizer, r["messages"], args.max_laenge) for r in train_rows]
        eval_ex = [build_example(tokenizer, r["messages"], args.max_laenge) for r in eval_rows if "messages" in r]
        skipped = sum(1 for e in train_ex if e is None)
        train_ex = [e for e in train_ex if e is not None]
        eval_ex = [e for e in eval_ex if e is not None]
        if skipped:
            _warn(f"{skipped} Beispiel(e) übersprungen: Zielantwort länger als --max-laenge {args.max_laenge}.")
        if len(train_ex) < MIN_EXAMPLES and not args.erzwingen:
            raise SystemExit(f"Abbruch: nach dem Tokenisieren bleiben nur {len(train_ex)} Beispiele (< {MIN_EXAMPLES}). "
                             "Erhöhe --max-laenge oder kürze die Antworten.")
        if not train_ex:
            raise SystemExit("Abbruch: kein Beispiel passt in --max-laenge.")
        training_args = _training_arguments(transformers, **common_args)
        trainer = transformers.Trainer(
            model=model, args=training_args, train_dataset=_ListDataset(train_ex),
            eval_dataset=_ListDataset(eval_ex) if eval_ex else None,
            data_collator=make_collator(tokenizer.pad_token_id, torch),
            callbacks=[_make_callback(transformers, tracker)],
        )
        n_train, n_eval = len(train_ex), len(eval_ex)

    _say(f"Training startet: {n_train} Beispiele, {args.epochen:g} Epochen, lr {args.lr:g}, r={args.rang}, alpha={args.alpha}")
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "Abbruch: GPU-Speicher reicht nicht (CUDA out of memory). Versuche --qlora, --max-laenge 512, "
            "--batch 1 --grad-akkum 16 oder eine kleinere Basis (qwen2.5:3b)."
        ) from None
    except KeyboardInterrupt:
        _warn("Training abgebrochen (Strg+C) – der aktuelle Stand wird gespeichert.")
    dauer = time.time() - t0

    # 5) Speichern
    model.save_pretrained(ausgabe)
    tokenizer.save_pretrained(ausgabe)
    info = {
        "hf_base": hf_base, "ollama_base": ollama_base, "n_train": n_train, "n_eval": n_eval,
        "epochen": args.epochen, "lr": args.lr, "rang": args.rang, "alpha": args.alpha,
        "max_laenge": args.max_laenge, "chat_template_hash": chat_template_hash(tokenizer),
        "qlora": use_qlora, "dpo": bool(args.dpo), "ziel_module": targets,
        "train_loss": tracker.train_losses, "eval_loss": tracker.eval_losses,
        "ueberanpassung_gewarnt": tracker.warned, "dauer_s": round(dauer, 1),
        "erstellt": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_training_info(ausgabe, info)
    _say(f"Adapter gespeichert: {ausgabe} ({dauer / 60:.1f} min)")
    if tracker.eval_losses:
        _say(f"Eval-Loss je Epoche: {', '.join(f'{l:.4f}' for l in tracker.eval_losses)}"
             + (" – bestes Modell geladen." if eval_rows else ""))

    merged_dir: str | None = None
    if args.zusammenfuehren:
        merged_dir = os.path.join(ausgabe, "merged")
        _say("Führe Adapter und Basis zusammen …")
        try:
            if use_qlora:
                # 4-bit-Gewichte lassen sich nicht sauber einrechnen: Basis in bf16/fp16 neu laden
                base = transformers.AutoModelForCausalLM.from_pretrained(hf_base, torch_dtype=dtype, device_map={"": "cpu"})
                merged = peft.PeftModel.from_pretrained(base, ausgabe).merge_and_unload()
            else:
                merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            _say(f"Zusammengeführtes Modell: {merged_dir}")
        except Exception as e:
            _warn(f"Zusammenführen fehlgeschlagen ({e}). Der Adapter in {ausgabe} ist trotzdem nutzbar (Route A).")
            merged_dir = None

    _say(route_hints(ausgabe, hf_base, ollama_base, merged_dir))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
