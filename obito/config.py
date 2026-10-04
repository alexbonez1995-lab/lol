"""Konfiguration von OBITO.

Reihenfolge der Quellen (spätere überschreiben frühere):
Standardwerte -> Konfigurationsdatei -> Umgebungsvariablen ``OBITO_<FELD>``.

Konfigurationsdatei: expliziter Pfad, sonst ``$OBITO_CONFIG``, sonst ``./obito.json``,
sonst ``<data_dir>/config.json``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_DATA_DIR = Path(os.environ.get("OBITO_HOME", str(Path.home() / ".obito")))


@dataclass
class Config:
    # Speicherort für Gedächtnis, Lern-Daten, Datensätze, Modelle, Logs
    data_dir: str = str(DEFAULT_DATA_DIR)

    # Modell-Anbindung: auto | ollama | openai | fake
    backend: str = "auto"
    base_url: str = ""                 # leer = Standard des Backends
    model: str = "qwen2.5:7b"          # Hauptmodell (Experten, Kritiker, Synthese)
    fast_model: str = ""               # kleines Modell für Routing/Extraktion (leer = model)
    embed_model: str = "nomic-embed-text"

    # Modell-Laufzeit
    num_ctx: int = 8192                # Kontextfenster in Tokens (8 GB VRAM: 8192 mit 7B-Q4; 16 GB: 16384)
    keep_alive: str = "30m"            # Ollama: Modell so lange im Speicher halten
    think: str = "auto"                # Denk-Modelle (qwen3, deepseek-r1): auto | an | aus
    parallel_calls: int = 2            # gleichzeitige Modellaufrufe (auf OLLAMA_NUM_PARALLEL abstimmen)
    timeout: float = 300.0             # Sekunden ohne Daten vom Modell (Leerlauf-Timeout)
    deadline: float = 600.0            # Sekunden Gesamtbudget pro Frage

    # Denken
    depth: str = "auto"                # auto | schnell | mittel | tief
    experts_per_question: int = 5      # Stufe "tief": Experten + Kritiker + Revision
    experts_medium: int = 3            # Stufe "mittel": Experten + Synthese, kein Kritiker
    max_revision_rounds: int = 1
    max_revised_experts: int = 2
    max_tool_rounds: int = 4
    max_tool_calls_per_answer: int = 3
    max_history: int = 12
    memory_recall: int = 6
    example_recall: int = 3
    max_new_memories: int = 3          # automatisch gemerkte Erinnerungen je Antwort
    temperature: float = 0.4

    # Ausgabe-Budget je Stufe (Tokens)
    max_tokens_fast: int = 1500
    max_tokens_expert: int = 900
    max_tokens_critic: int = 700
    max_tokens_synthesis: int = 1800
    max_tokens_json: int = 500

    # Verhalten
    allow_tools: bool = True
    confirm_dangerous: bool = True
    auto_memory: bool = True
    language: str = "de"
    workspace: str = "."               # Verzeichnis, auf das Datei-Werkzeuge zugreifen dürfen

    # Lokaler API-Server (für die HUD-Oberfläche)
    server_host: str = "127.0.0.1"
    server_port: int = 8765

    # ------------------------------------------------------------ Pfade
    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def memory_db(self) -> Path:
        return self.data_path / "gedaechtnis.db"

    @property
    def learning_db(self) -> Path:
        return self.data_path / "lernen.db"

    @property
    def datasets_dir(self) -> Path:
        return self.data_path / "datensaetze"

    @property
    def models_dir(self) -> Path:
        return self.data_path / "modelle"

    @property
    def logs_dir(self) -> Path:
        return self.data_path / "logs"

    @property
    def routing_model(self) -> str:
        return self.fast_model or self.model

    def ensure_dirs(self) -> None:
        for p in (self.data_path, self.datasets_dir, self.models_dir, self.logs_dir):
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


_TRUE = {"1", "true", "ja", "yes", "on", "wahr"}


def _coerce(value: str, typ: type):
    if typ is bool:
        return value.strip().lower() in _TRUE
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    return value


def find_config_file(explicit: str | Path | None = None) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("OBITO_CONFIG"):
        candidates.append(Path(os.environ["OBITO_CONFIG"]))
    candidates.append(Path("obito.json"))
    candidates.append(DEFAULT_DATA_DIR / "config.json")
    for c in candidates:
        if c.is_file():
            return c
    return None


def load_config(path: str | Path | None = None, env: dict | None = None) -> Config:
    """Lädt die Konfiguration. ``env`` erlaubt Tests ohne echte Umgebungsvariablen."""
    cfg = Config()
    file = find_config_file(path)
    if file:
        data = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Konfigurationsdatei {file} muss ein JSON-Objekt sein.")
        known = {f.name for f in fields(Config)}
        for k, v in data.items():
            if k in known:
                setattr(cfg, k, v)

    env = os.environ if env is None else env
    for f in fields(Config):
        key = f"OBITO_{f.name.upper()}"
        if key in env and env[key] != "":
            setattr(cfg, f.name, _coerce(env[key], f.type if isinstance(f.type, type) else type(getattr(cfg, f.name))))
    return cfg


def save_config(cfg: Config, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return p
