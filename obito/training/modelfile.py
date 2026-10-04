"""Ollama-Modelfiles für eigene OBITO-Modelle.

Nach einem LoRA-Training (``train_lora.py``) entsteht entweder ein Adapter (``adapter.gguf``)
oder ein zusammengeführtes Modell (``merged.q4_k_m.gguf``). Dieses Modul baut daraus ein
Modelfile (``FROM``/``ADAPTER``/``TEMPLATE``/``SYSTEM``/``PARAMETER``) und ruft
``ollama create`` auf. Nur Standardbibliothek; ``obito.agents`` wird – falls vorhanden –
lazy für den Standard-System-Prompt verwendet.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

# ------------------------------------------------------------ Basis-Zuordnung
#: Ollama-Tag -> Hugging-Face-ID (exakt das Modell, das Ollama unter diesem Tag ausliefert)
BASE_MAP: dict[str, str] = {
    "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen3:8b": "Qwen/Qwen3-8B",
    "qwen3:14b": "Qwen/Qwen3-14B",
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3:12b": "google/gemma-3-12b-it",
    "phi3.5:3.8b": "microsoft/Phi-3.5-mini-instruct",
}

#: Größe, die Ollama für ``name`` bzw. ``name:latest`` ausliefert
_DEFAULT_SIZE: dict[str, str] = {
    "qwen2.5": "7b",
    "qwen2.5-coder": "7b",
    "qwen3": "8b",
    "llama3.1": "8b",
    "llama3.2": "3b",
    "mistral": "7b",
    "gemma3": "4b",
    "phi3.5": "3.8b",
}

_HF_TO_OLLAMA: dict[str, str] = {hf.lower(): tag for tag, hf in BASE_MAP.items()}

# ------------------------------------------------------------------ Vorlagen
#: ChatML-Vorlage (Qwen 2.5 / Qwen 3) im Go-Template-Format von Ollama
CHATML_TEMPLATE: str = (
    "{{- if .Messages }}\n"
    "{{- if .System }}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}\n"
    "{{- range $i, $_ := .Messages }}\n"
    "{{- $last := eq (len (slice $.Messages $i)) 1 }}\n"
    "{{- if eq .Role \"user\" }}<|im_start|>user\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ else if eq .Role \"assistant\" }}<|im_start|>assistant\n"
    "{{ .Content }}{{ if not $last }}<|im_end|>\n"
    "{{ end }}\n"
    "{{- end }}\n"
    "{{- if and $last (ne .Role \"assistant\") }}<|im_start|>assistant\n"
    "{{ end }}\n"
    "{{- end }}\n"
    "{{- else }}\n"
    "{{- if .System }}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}{{ if .Prompt }}<|im_start|>user\n"
    "{{ .Prompt }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
    "{{ .Response }}{{ if .Response }}<|im_end|>{{ end }}\n"
    "{{- end }}"
)

#: Stop-Token je Prompt-Format
STOP_TOKENS: dict[str, list[str]] = {
    "chatml": ["<|im_start|>", "<|im_end|>"],
    "llama3": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    "mistral": ["[INST]", "[/INST]"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
    "phi3": ["<|end|>", "<|user|>", "<|assistant|>"],
}

#: Modellfamilie -> Prompt-Format (für Basen ohne eigene Vorlage)
_FAMILY_FORMAT: dict[str, str] = {
    "qwen": "chatml",
    "deepseek": "chatml",
    "llama": "llama3",
    "mistral": "mistral",
    "gemma": "gemma",
    "phi": "phi3",
}

#: Familien, deren safetensors-Adapter Ollama direkt (ohne GGUF-Konvertierung) laden kann
SAFETENSORS_ADAPTER_FAMILIES = ("llama", "mistral", "gemma")

_FALLBACK_SYSTEM_PROMPT = (
    "Du bist OBITO, ein lokaler KI-Assistent, der als OMEGA die Antworten eines Expertengremiums "
    "koordiniert und zu einer geprüften, vollständigen Antwort zusammenführt.\n\n"
    "Regeln:\n"
    "- Antworte auf Deutsch, konkret und knapp; keine Floskeln.\n"
    "- Zahlen immer mit Einheiten; Rechenwege kurz nachvollziehbar machen.\n"
    "- Erfinde keine Fakten, Quellen oder Messwerte. Markiere Unsicheres mit »Unsicher:«.\n"
    "- Befolge Erinnerungen und Lektionen aus früherem Feedback des Nutzers.\n"
    "- Fehlen wichtige Informationen, stelle gezielte Rückfragen statt zu raten.\n"
    "- Weise auf Sicherheitsrisiken (Strom, Mechanik, Chemie, Daten) ausdrücklich hin."
)


# ---------------------------------------------------------------- Hilfen
def _clean_tag(tag: str) -> str:
    """Kleinschreibung, ohne Registry-Präfix (``registry.ollama.ai/library/…``)."""
    t = (tag or "").strip().lower()
    if "/" in t:
        t = t.rsplit("/", 1)[1]
    return t


def hf_base_for(ollama_tag: str | None) -> str | None:
    """Hugging-Face-ID zu einem Ollama-Tag.

    Akzeptiert ``qwen2.5:7b``, ``qwen2.5`` / ``qwen2.5:latest`` (Standardgröße) und
    Varianten wie ``qwen2.5:7b-instruct-q4_K_M`` (werden auf ``qwen2.5:7b`` abgebildet).
    Unbekannt -> ``None``.
    """
    t = _clean_tag(ollama_tag or "")
    if not t:
        return None
    if t in BASE_MAP:
        return BASE_MAP[t]
    name, _, variant = t.partition(":")
    if not variant or variant == "latest":
        size = _DEFAULT_SIZE.get(name)
        return BASE_MAP.get(f"{name}:{size}") if size else None
    # Varianten: "7b-instruct-q4_K_M" -> "7b"
    size = variant.split("-", 1)[0]
    return BASE_MAP.get(f"{name}:{size}")


def ollama_base_for(hf_id: str | None) -> str | None:
    """Ollama-Tag zu einer Hugging-Face-ID (oder einem lokalen Pfad mit dem Modellnamen)."""
    h = (hf_id or "").strip().strip("/").replace("\\", "/").lower()
    if not h:
        return None
    if h in _HF_TO_OLLAMA:
        return _HF_TO_OLLAMA[h]
    tail = h.rsplit("/", 1)[-1]
    for hf, tag in _HF_TO_OLLAMA.items():
        if hf.rsplit("/", 1)[-1] == tail:
            return tag
    return None


def family_of(name: str | None) -> str | None:
    """Modellfamilie (``qwen``, ``llama``, ``mistral``, ``gemma``, ``phi``, ``deepseek``) aus
    Tag, HF-ID oder Dateiname; unbekannt -> ``None``."""
    n = os.path.basename((name or "").strip().rstrip("/\\")).lower()
    if not n:
        return None
    for fam in ("qwen", "llama", "mistral", "mixtral", "gemma", "phi", "deepseek"):
        if fam in n:
            return "mistral" if fam == "mixtral" else fam
    return None


def is_gguf(path: str | None) -> bool:
    return bool(path) and str(path).strip().lower().endswith(".gguf")


def stop_tokens_for(template: str | None, base: str) -> list[str]:
    """Stop-Token passend zur Vorlage (erkannt an den Markern im Text), sonst zur Basis-Familie."""
    if template:
        found: list[str] = []
        for fmt, toks in STOP_TOKENS.items():
            if any(tok in template for tok in toks):
                found.extend(t for t in toks if t not in found)
        return found
    fmt = _FAMILY_FORMAT.get(family_of(base) or "")
    return list(STOP_TOKENS.get(fmt or "", []))


def _triple_quote(text: str) -> str:
    """Quotet Text für ein Modelfile in ``\"\"\"``; innere ``\"\"\"`` werden entschärft."""
    return '"""' + text.replace('"""', "'''") + '"""'


def _format_param_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, int):
        return str(value)
    s = str(value)
    if not s or any(ch.isspace() for ch in s) or "#" in s:
        return json.dumps(s, ensure_ascii=False)
    return s


def default_system_prompt() -> str:
    """OMEGA-Persona + Grundregeln aus ``obito.agents``; Fallback-Text, wenn das Modul fehlt."""
    try:
        from obito import agents  # lazy: Modul entsteht parallel
        omega = getattr(agents, "OMEGA", None)
        rules = getattr(agents, "BASE_RULES", None)
        persona = getattr(omega, "system_prompt", None) if omega is not None else None
        if isinstance(persona, str) and persona.strip() and isinstance(rules, str) and rules.strip():
            return persona.strip() + "\n\n" + rules.strip()
    except Exception:
        pass
    return _FALLBACK_SYSTEM_PROMPT


# ------------------------------------------------------------ Modelfile
def build_modelfile(
    base: str,
    system_prompt: str | None,
    *,
    adapter: str | None = None,
    template: str | None = None,
    temperature: float = 0.4,
    num_ctx: int = 8192,
    extra_params: Mapping[str, Any] | None = None,
) -> str:
    """Baut den Text eines Ollama-Modelfiles.

    ``base``: Ollama-Tag (``qwen2.5:7b``), GGUF-Datei oder HF-Verzeichnis.
    ``adapter``: ``adapter.gguf`` (immer erlaubt) oder ein safetensors-Verzeichnis (nur für
    die Familien llama/mistral/gemma). Bei GGUF-Basis ist ``template`` Pflicht
    (Standard: ChatML, wenn der Name ``qwen`` enthält). Pfade werden absolut eingetragen,
    damit das Modelfile unabhängig vom Arbeitsverzeichnis funktioniert.
    """
    base = (base or "").strip()
    if not base:
        raise ValueError("Basis fehlt: Ollama-Tag (z. B. qwen2.5:7b) oder GGUF-Datei angeben.")
    if not (0.0 <= float(temperature) <= 2.0):
        raise ValueError(f"temperature muss zwischen 0 und 2 liegen (nicht {temperature}).")
    if int(num_ctx) < 512:
        raise ValueError(f"num_ctx muss mindestens 512 sein (nicht {num_ctx}).")

    base_is_file = is_gguf(base) or os.path.isdir(base)
    family = family_of(base)

    # Vorlage
    if is_gguf(base) and not template:
        if family == "qwen":
            template = CHATML_TEMPLATE
        else:
            raise ValueError(
                f"Für die GGUF-Basis »{os.path.basename(base)}« ist eine Vorlage (TEMPLATE) Pflicht, "
                "weil die Datei kein Prompt-Format mitbringt. Gib sie mit --vorlage PFAD an "
                "(Ollama-Go-Template; für Qwen wird ChatML automatisch gewählt)."
            )

    # Adapter
    adapter_line = None
    if adapter:
        adapter = adapter.strip()
        if not is_gguf(adapter):
            if family not in SAFETENSORS_ADAPTER_FAMILIES:
                hf = hf_base_for(base) or (base if not base_is_file else "<HF-ID der Basis>")
                raise ValueError(
                    f"Ollama kann einen safetensors-Adapter nur für die Familien "
                    f"{'/'.join(SAFETENSORS_ADAPTER_FAMILIES)} laden, nicht für »{family or 'unbekannt'}« "
                    f"({base}). Konvertiere den Adapter zuerst mit llama.cpp:\n"
                    f"  python convert_lora_to_gguf.py {adapter} --base {hf} --outfile adapter.gguf\n"
                    f"und gib dann --adapter adapter.gguf an."
                )
        adapter_line = f"ADAPTER {os.path.abspath(adapter)}"

    base_ref = os.path.abspath(base) if base_is_file else base

    lines = ["# Modelfile – erzeugt von OBITO", f"FROM {base_ref}"]
    if adapter_line:
        lines.append(adapter_line)
    if template:
        lines.append("TEMPLATE " + _triple_quote(template.strip("\n")))
    if system_prompt and system_prompt.strip():
        lines.append("SYSTEM " + _triple_quote(system_prompt.strip()))

    params: dict[str, Any] = {"temperature": float(temperature), "num_ctx": int(num_ctx)}
    stops = stop_tokens_for(template, base)
    for key, value in dict(extra_params or {}).items():
        k = str(key).strip().lower()
        if not re.fullmatch(r"[a-z_]+", k):
            raise ValueError(f"Ungültiger PARAMETER-Name: {key!r}")
        if value is None:
            continue
        if k == "stop":
            for s in (value if isinstance(value, (list, tuple)) else [value]):
                if str(s) and str(s) not in stops:
                    stops.append(str(s))
            continue
        params[k] = value
    for k, v in params.items():
        values = v if isinstance(v, (list, tuple)) else [v]
        for one in values:
            lines.append(f"PARAMETER {k} {_format_param_value(one)}")
    for s in stops:
        lines.append(f"PARAMETER stop {json.dumps(s, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


_NAME_RE = re.compile(r"^[A-Za-z0-9][\w.-]*(?::[\w.-]+)?$")


def create_model(
    name: str,
    modelfile_text: str,
    *,
    ollama_bin: str = "ollama",
    cwd: str | os.PathLike | None = None,
) -> subprocess.CompletedProcess:
    """Schreibt ``Modelfile`` nach ``cwd`` (Verzeichnis des Adapters/GGUF; ``None`` = temporär)
    und führt ``ollama create <name> -f Modelfile`` aus.

    Gibt das ``CompletedProcess`` zurück (``returncode``, ``stdout``, ``stderr``); der
    Aufrufer entscheidet, wie ein Fehlschlag gemeldet wird. Ein fehlendes ``ollama``-Programm
    wird als ``FileNotFoundError`` mit deutscher Meldung gemeldet.
    """
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise ValueError(
            f"Ungültiger Modellname »{name}«: erlaubt sind Buchstaben, Ziffern, ._- und optional ein Tag (name:tag)."
        )
    if not modelfile_text or "FROM " not in modelfile_text:
        raise ValueError("Modelfile-Text ist leer oder enthält keine FROM-Zeile.")
    ollama_bin = (ollama_bin or "ollama").strip() or "ollama"

    def _run(directory: str) -> subprocess.CompletedProcess:
        Path(directory, "Modelfile").write_text(modelfile_text, encoding="utf-8")
        try:
            return subprocess.run(
                [ollama_bin, "create", name, "-f", "Modelfile"],
                cwd=directory, capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Ollama-Programm »{ollama_bin}« nicht gefunden. Ist Ollama installiert und im PATH? "
                "(Download: https://ollama.com/download; alternativ --ollama-bin PFAD)"
            ) from None
        except PermissionError as e:
            raise PermissionError(f"Ollama-Programm »{ollama_bin}« darf nicht ausgeführt werden: {e}") from None

    if cwd is None:
        with tempfile.TemporaryDirectory(prefix="obito-modelfile-") as tmp:
            return _run(tmp)
    directory = Path(cwd)
    directory.mkdir(parents=True, exist_ok=True)
    return _run(str(directory))


# ------------------------------------------------------- Kompatibilität
TRAINING_INFO_FILE = "obito_training.json"


def _training_info_path(adapter_path: str) -> Path:
    p = Path(adapter_path)
    directory = p if p.is_dir() else p.parent
    return directory / TRAINING_INFO_FILE


def check_adapter_compat(adapter_path: str, ollama_base: str) -> str | None:
    """Prüft anhand von ``obito_training.json`` neben dem Adapter, ob er zur Basis passt.

    ``None`` = passt oder nicht prüfbar (keine Trainingsinfo vorhanden); sonst ein Warntext.
    Ein LoRA-Adapter funktioniert nur mit exakt dem Basismodell, auf dem er trainiert wurde.
    """
    if not adapter_path:
        return None
    info_path = _training_info_path(adapter_path)
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(info, dict):
        return None

    trained_hf = str(info.get("hf_base") or "").strip() or None
    trained_tag = str(info.get("ollama_base") or "").strip() or None
    trained_resolved = (trained_hf or hf_base_for(trained_tag) or "").lower() or None
    trained_label = trained_tag or trained_hf or "unbekannt"

    wanted_resolved = (hf_base_for(ollama_base) or "").lower() or None
    if wanted_resolved is None and os.path.isdir(ollama_base or ""):
        wanted_resolved = (ollama_base_for(ollama_base) and hf_base_for(ollama_base_for(ollama_base)) or "").lower() or None

    if trained_resolved and wanted_resolved:
        if trained_resolved == wanted_resolved:
            return None
        return (
            f"Warnung: Der Adapter wurde auf »{trained_label}« ({trained_resolved}) trainiert, "
            f"die Basis ist aber »{ollama_base}« ({wanted_resolved}). Ein LoRA-Adapter passt nur zum "
            f"exakt gleichen Basismodell – nutze --basis {trained_tag or trained_hf} oder trainiere neu."
        )
    if trained_resolved or trained_label != "unbekannt":
        if trained_tag and _clean_tag(trained_tag) == _clean_tag(ollama_base or ""):
            return None
        return (
            f"Hinweis: Die Basis »{ollama_base}« ist nicht bekannt, daher ist die Kompatibilität nicht "
            f"prüfbar. Der Adapter wurde auf »{trained_label}« trainiert und passt nur zu genau dieser Basis."
        )
    return None
