"""System- und Leistungsüberwachung von OBITO.

Echte Messwerte zu CPU, RAM, Platte, GPU/VRAM, eigenem Prozess und geladenen Modellen –
nur mit der Python-Standardbibliothek. Grundsatz: **nie erfundene Werte**. Was auf der
aktuellen Plattform nicht messbar ist, wird als ``None`` geliefert und in Texten als
»unbekannt« ausgegeben.

Alle externen Quellen (Dateien unter ``/proc``, Unterprozesse wie ``nvidia-smi`` oder
PowerShell, ctypes-Aufrufe) sind in kleinen Funktionen gekapselt (``_read_file``, ``_run``,
``_proc_stat``, ``_win_memory`` …), damit Tests sie per Monkeypatching ersetzen können, ohne
echte Hardware zu brauchen. Jeder Unterprozess läuft mit Timeout; ``FileNotFoundError``,
``TimeoutExpired`` und ``OSError`` werden abgefangen.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfer
    from .llm import LLMBackend
    from .tools import ToolRegistry

log = logging.getLogger("obito.sysmon")

# ------------------------------------------------------------------ Konstanten
CPU_SAMPLE_INTERVAL = 0.1        # Sekunden zwischen den zwei /proc/stat-Stichproben
NVIDIA_SMI_TIMEOUT = 5.0         # Sekunden
POWERSHELL_TIMEOUT = 10.0        # Sekunden
SYSCTL_TIMEOUT = 5.0             # Sekunden
MAX_DIR_FILES = 20_000           # Obergrenze für die rekursive Größenermittlung
GB = 1024 ** 3
MB = 1024 ** 2

RAM_WARN_BYTES = 8 * GB
DISK_WARN_FREE_BYTES = 5 * GB
GPU_TEMP_WARN_C = 85.0
VRAM_USED_WARN_PERCENT = 90.0

NVIDIA_SMI_CMD = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
                  "--format=csv,noheader,nounits"]
POWERSHELL_CPU_CMD = ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                      "Get-CimInstance Win32_Processor | Select Name,NumberOfCores,NumberOfLogicalProcessors,"
                      "LoadPercentage | ConvertTo-Json"]

# VRAM-Klassen (GB, aufsteigend): (Mindest-VRAM, Modell, num_ctx, Zusatz)
VRAM_CLASSES: tuple[tuple[int, str, int, str], ...] = (
    (24, "qwen2.5:32b", 16384, "32B-Q4 passt komplett in den VRAM"),
    (16, "qwen2.5:14b", 16384, "14B-Q4 mit großem Kontext; 32B nur mit Auslagerung ins RAM"),
    (12, "qwen2.5:14b", 8192, "14B-Q4 passt knapp – bei Speicherfehlern num_ctx verringern"),
    (8, "qwen2.5:7b", 8192, "7B-Q4 mit num_ctx 8192"),
    (6, "qwen2.5:7b", 4096, "7B-Q4 knapp – kleiner Kontext"),
    (4, "qwen2.5:3b", 4096, "3B-Q4; 7B nur teilweise auf der GPU"),
)

_START_TIME = time.time()


# ------------------------------------------------------------------ Quellen (monkeypatchbar)
def _system() -> str:
    """Name des Betriebssystems (``Linux``, ``Windows``, ``Darwin`` …)."""
    return platform.system()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _read_file(path: str) -> str | None:
    """Liest eine Textdatei vollständig; ``None`` bei jedem Fehler."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, ValueError):
        return None


def _run(cmd: Sequence[str], timeout: float) -> str | None:
    """Führt ``cmd`` aus und liefert die Standardausgabe; ``None`` bei Fehler, Timeout oder Rückgabewert ≠ 0."""
    kwargs: dict[str, Any] = {}
    if os.name == "nt":  # pragma: no cover - kein Konsolenfenster unter Windows
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        proc = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace", **kwargs)
    except FileNotFoundError:
        log.debug("Programm nicht gefunden: %s", cmd[0] if cmd else "?")
        return None
    except subprocess.TimeoutExpired:
        log.warning("Zeitüberschreitung (%.0f s) bei %s", timeout, cmd[0] if cmd else "?")
        return None
    except (OSError, ValueError) as e:
        log.debug("Fehler beim Start von %s: %s", cmd[0] if cmd else "?", e)
        return None
    if proc.returncode != 0:
        log.debug("%s endete mit Code %s", cmd[0] if cmd else "?", proc.returncode)
        return None
    return proc.stdout


def _proc_stat() -> str | None:
    return _read_file("/proc/stat")


def _proc_cpuinfo() -> str | None:
    return _read_file("/proc/cpuinfo")


def _proc_meminfo() -> str | None:
    return _read_file("/proc/meminfo")


def _proc_self_status() -> str | None:
    return _read_file("/proc/self/status")


def _loadavg() -> tuple[float, float, float] | None:
    getter = getattr(os, "getloadavg", None)
    if getter is None:
        return None
    try:
        return tuple(getter())  # type: ignore[return-value]
    except OSError:
        return None


def _cpu_count() -> int | None:
    try:
        return os.cpu_count()
    except Exception:  # noqa: BLE001 - defensiv, cpu_count sollte nie werfen
        return None


def _nvidia_smi() -> str | None:
    return _run(NVIDIA_SMI_CMD, NVIDIA_SMI_TIMEOUT)


def _powershell_cpu() -> str | None:
    return _run(POWERSHELL_CPU_CMD, POWERSHELL_TIMEOUT)


def _sysctl(key: str) -> str | None:
    out = _run(["sysctl", "-n", key], SYSCTL_TIMEOUT)
    return out.strip() if out is not None else None


def _vm_stat() -> str | None:
    return _run(["vm_stat"], SYSCTL_TIMEOUT)


def _win_memory() -> tuple[int, int] | None:
    """Windows: ``(gesamt_bytes, frei_bytes)`` über ``GlobalMemoryStatusEx``; sonst ``None``."""
    if os.name != "nt":
        return None
    try:  # pragma: no cover - nur unter Windows ausführbar
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
            return None
        return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
    except Exception as e:  # noqa: BLE001
        log.debug("GlobalMemoryStatusEx fehlgeschlagen: %s", e)
        return None


def _win_process_rss() -> int | None:
    """Windows: Arbeitsspeicher des eigenen Prozesses (WorkingSetSize) über ``GetProcessMemoryInfo``."""
    if os.name != "nt":
        return None
    try:  # pragma: no cover - nur unter Windows ausführbar
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()  # type: ignore[attr-defined]
        psapi = getattr(ctypes.windll, "psapi", None)  # type: ignore[attr-defined]
        if psapi is None:
            return None
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return None
        return int(counters.WorkingSetSize)
    except Exception as e:  # noqa: BLE001
        log.debug("GetProcessMemoryInfo fehlgeschlagen: %s", e)
        return None


def _unix_rss() -> int | None:
    """Unix: aktueller RSS in Bytes – Linux über ``/proc/self/status``, sonst ``resource.getrusage``."""
    if os.name == "nt":
        return None
    if _system() == "Linux":
        rss = parse_status_rss(_proc_self_status())
        if rss is not None:
            return rss
    try:
        import resource
    except ImportError:  # pragma: no cover
        return None
    try:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    if maxrss <= 0:
        return None
    # macOS liefert Bytes, Linux/BSD Kibibytes.
    return int(maxrss) if _system() == "Darwin" else int(maxrss) * 1024


# ------------------------------------------------------------------ Parser (rein, testbar)
def _to_float(value: Any) -> float | None:
    """Zahl aus einem Text/Wert; ``None`` bei ``[N/A]``, leer oder Unsinn."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if not text or text.startswith("[") or text.lower() in ("n/a", "na", "null", "none"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_int(value: Any) -> int | None:
    f = _to_float(value)
    return int(round(f)) if f is not None else None


def _clamp_percent(value: float | None) -> float | None:
    if value is None:
        return None
    return round(max(0.0, min(100.0, float(value))), 1)


def parse_proc_stat(text: str | None) -> tuple[int, int] | None:
    """Erste ``cpu``-Zeile aus ``/proc/stat`` → ``(leerlauf_ticks, gesamt_ticks)``; ``None`` wenn unlesbar."""
    if not text:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "cpu":
            try:
                values = [int(x) for x in parts[1:]]
            except ValueError:
                return None
            # user nice system idle iowait irq softirq steal (guest/guest_nice stecken schon in user/nice)
            fields = values[:8]
            idle = fields[3] + (fields[4] if len(fields) > 4 else 0)
            total = sum(fields)
            return idle, total
    return None


def cpu_load_from_samples(first: tuple[int, int] | None, second: tuple[int, int] | None) -> float | None:
    """Auslastung in Prozent aus zwei ``(leerlauf, gesamt)``-Stichproben; ``None`` ohne Fortschritt."""
    if first is None or second is None:
        return None
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta <= 0:
        return None
    return _clamp_percent((1.0 - idle_delta / total_delta) * 100.0)


def parse_cpuinfo_name(text: str | None) -> str | None:
    """Prozessorname aus ``/proc/cpuinfo`` (``model name``, ARM: ``Hardware``/``Processor``/``model``)."""
    if not text:
        return None
    candidates: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if value and key in ("model name", "hardware", "processor", "model", "cpu model") and key not in candidates:
            candidates[key] = value
    for key in ("model name", "hardware", "cpu model", "model", "processor"):
        value = candidates.get(key)
        if value and not value.isdigit():
            return value
    return None


def parse_cpuinfo_physical(text: str | None) -> int | None:
    """Physische Kerne aus ``/proc/cpuinfo`` (eindeutige ``physical id``/``core id``-Paare); ``None`` wenn unbekannt."""
    if not text:
        return None
    pairs: set[tuple[str, str]] = set()
    cores_per_socket: int | None = None
    sockets: set[str] = set()
    phys: str | None = None
    core: str | None = None
    for line in text.splitlines() + [""]:
        if not line.strip():
            if phys is not None and core is not None:
                pairs.add((phys, core))
            if phys is not None:
                sockets.add(phys)
            phys = core = None
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key == "physical id":
            phys = value
        elif key == "core id":
            core = value
        elif key == "cpu cores" and cores_per_socket is None:
            cores_per_socket = _to_int(value)
    if pairs:
        return len(pairs)
    if cores_per_socket:
        return cores_per_socket * max(1, len(sockets))
    return None


def parse_meminfo(text: str | None) -> dict | None:
    """``/proc/meminfo`` → ``{"gesamt_bytes", "frei_bytes"}`` (MemTotal/MemAvailable, Fallback MemFree+Buffers+Cached)."""
    if not text:
        return None
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            number = int(parts[0])
        except ValueError:
            continue
        unit = parts[1].lower() if len(parts) > 1 else "kb"
        factor = {"kb": 1024, "mb": MB, "gb": GB, "b": 1}.get(unit, 1024)
        values[key.strip()] = number * factor
    total = values.get("MemTotal")
    if total is None:
        return None
    avail = values.get("MemAvailable")
    if avail is None and "MemFree" in values:
        avail = values["MemFree"] + values.get("Buffers", 0) + values.get("Cached", 0)
    return {"gesamt_bytes": total, "frei_bytes": avail}


def parse_status_rss(text: str | None) -> int | None:
    """``VmRSS`` aus ``/proc/self/status`` in Bytes."""
    if not text:
        return None
    for line in text.splitlines():
        if line.startswith("VmRSS:"):
            parts = line.split()
            if len(parts) >= 2:
                kib = _to_int(parts[1])
                return kib * 1024 if kib is not None else None
    return None


def parse_nvidia_smi(text: str | None) -> list[dict]:
    """CSV-Ausgabe von ``nvidia-smi`` (name, memory.total, memory.used, utilization.gpu, temperature.gpu).

    Zeilen werden tolerant gelesen: fehlende Spalten und ``[N/A]`` ergeben ``None``; Leerzeilen und
    Zeilen ohne Namen werden übersprungen."""
    if not text:
        return []
    gpus: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        cols = [c.strip() for c in line.split(",")]
        name = cols[0] if cols and cols[0] and not cols[0].startswith("[") else None
        if name is None:
            continue

        def col(i: int) -> str | None:
            return cols[i] if len(cols) > i else None

        gpus.append({
            "name": name,
            "vram_gesamt_mb": _to_int(col(1)),
            "vram_belegt_mb": _to_int(col(2)),
            "auslastung_prozent": _clamp_percent(_to_float(col(3))),
            "temperatur_c": _to_float(col(4)),
        })
    return gpus


def parse_powershell_cpu(text: str | None) -> dict | None:
    """JSON von ``Get-CimInstance Win32_Processor | ConvertTo-Json`` (Objekt oder Liste) → CPU-Dict.

    Mehrere Sockel werden summiert (Kerne/logisch), die Last gemittelt."""
    if not text:
        return None
    cleaned = text.strip().lstrip("﻿")
    if not cleaned:
        return None
    try:
        data = json.loads(cleaned)
    except ValueError:
        return None
    if isinstance(data, dict):
        items: list[dict] = [data]
    elif isinstance(data, list):
        items = [x for x in data if isinstance(x, dict)]
    else:
        return None
    if not items:
        return None
    names: list[str] = []
    cores = 0
    logical = 0
    cores_known = logical_known = False
    loads: list[float] = []
    for item in items:
        name = item.get("Name")
        if isinstance(name, str) and name.strip():
            name = " ".join(name.split())
            if name not in names:
                names.append(name)
        c = _to_int(item.get("NumberOfCores"))
        if c is not None:
            cores += c
            cores_known = True
        lp = _to_int(item.get("NumberOfLogicalProcessors"))
        if lp is not None:
            logical += lp
            logical_known = True
        load = _to_float(item.get("LoadPercentage"))
        if load is not None:
            loads.append(load)
    return {
        "name": " / ".join(names) if names else None,
        "kerne": cores if cores_known else None,
        "logisch": logical if logical_known else None,
        "last_prozent": _clamp_percent(sum(loads) / len(loads)) if loads else None,
    }


def parse_vm_stat(text: str | None) -> int | None:
    """macOS ``vm_stat`` → verfügbare Bytes (frei + inaktiv + spekulativ) × Seitengröße."""
    if not text:
        return None
    page_size = 4096
    pages: dict[str, int] = {}
    for line in text.splitlines():
        low = line.lower()
        if "page size of" in low:
            digits = "".join(ch for ch in low.split("page size of", 1)[1] if ch.isdigit())
            if digits:
                page_size = int(digits)
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        number = _to_int(value)
        if number is not None:
            pages[key.strip().lower()] = number
    if "pages free" not in pages:
        return None
    free = pages["pages free"] + pages.get("pages inactive", 0) + pages.get("pages speculative", 0)
    return free * page_size


# ------------------------------------------------------------------ Messungen
def cpu() -> dict:
    """CPU: ``{"name", "kerne" (physisch, sonst None), "logisch", "last_prozent"}``."""
    system = _system()
    result: dict[str, Any] = {"name": None, "kerne": None, "logisch": _cpu_count(), "last_prozent": None}
    try:
        if system == "Linux":
            info = _proc_cpuinfo()
            result["name"] = parse_cpuinfo_name(info)
            result["kerne"] = parse_cpuinfo_physical(info)
            first = parse_proc_stat(_proc_stat())
            if first is not None:
                _sleep(CPU_SAMPLE_INTERVAL)
                result["last_prozent"] = cpu_load_from_samples(first, parse_proc_stat(_proc_stat()))
        elif system == "Windows":
            parsed = parse_powershell_cpu(_powershell_cpu())
            if parsed:
                result["name"] = parsed["name"]
                result["kerne"] = parsed["kerne"]
                if parsed["logisch"] is not None:
                    result["logisch"] = parsed["logisch"]
                result["last_prozent"] = parsed["last_prozent"]
        elif system == "Darwin":
            result["name"] = _sysctl("machdep.cpu.brand_string") or None
            result["kerne"] = _to_int(_sysctl("hw.physicalcpu"))
        if result["last_prozent"] is None and system != "Windows":
            load = _loadavg()
            logical = result["logisch"]
            if load is not None and logical:
                result["last_prozent"] = _clamp_percent(load[0] / logical * 100.0)
    except Exception as e:  # noqa: BLE001 - Messung darf OBITO nie zum Absturz bringen
        log.warning("CPU-Messung fehlgeschlagen: %s", e)
    return result


def memory() -> dict:
    """RAM: ``{"gesamt_bytes", "frei_bytes", "belegt_prozent"}`` – ``None``, wenn nicht messbar."""
    total: int | None = None
    avail: int | None = None
    system = _system()
    try:
        if system == "Linux":
            parsed = parse_meminfo(_proc_meminfo())
            if parsed:
                total, avail = parsed["gesamt_bytes"], parsed["frei_bytes"]
        elif system == "Windows":
            win = _win_memory()
            if win:
                total, avail = win
        elif system == "Darwin":
            total = _to_int(_sysctl("hw.memsize"))
            avail = parse_vm_stat(_vm_stat())
    except Exception as e:  # noqa: BLE001
        log.warning("RAM-Messung fehlgeschlagen: %s", e)
    percent = None
    if total and avail is not None:
        percent = _clamp_percent((total - avail) / total * 100.0)
    return {"gesamt_bytes": total, "frei_bytes": avail, "belegt_prozent": percent}


def disk(path: str | None = None) -> dict:
    """Platte des Datenträgers von ``path`` (Standard: Arbeitsverzeichnis) über ``shutil.disk_usage``."""
    target = path or os.getcwd()
    probe = target
    for _ in range(32):                      # noch nicht angelegtes Datenverzeichnis → Elternpfad
        if os.path.exists(probe):
            break
        parent = os.path.dirname(probe.rstrip("/\\")) or probe
        if parent == probe:
            break
        probe = parent
    try:
        usage = shutil.disk_usage(probe)
    except (OSError, ValueError) as e:
        log.debug("disk_usage(%s) fehlgeschlagen: %s", target, e)
        return {"pfad": target, "gesamt_bytes": None, "frei_bytes": None, "belegt_prozent": None}
    percent = _clamp_percent(usage.used / usage.total * 100.0) if usage.total else None
    return {"pfad": target, "gesamt_bytes": int(usage.total), "frei_bytes": int(usage.free), "belegt_prozent": percent}


def gpu() -> list[dict]:
    """NVIDIA-GPUs über ``nvidia-smi`` (Timeout 5 s). Ohne nvidia-smi (auch AMD/Intel): ``[]``."""
    try:
        return parse_nvidia_smi(_nvidia_smi())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as e:
        log.debug("nvidia-smi nicht nutzbar: %s", e)
        return []
    except Exception as e:  # noqa: BLE001
        log.warning("GPU-Abfrage fehlgeschlagen: %s", e)
        return []


def process() -> dict:
    """Eigener Prozess: ``{"pid", "rss_bytes", "threads", "laufzeit_s"}``."""
    rss: int | None = None
    try:
        rss = _win_process_rss() if os.name == "nt" else _unix_rss()
    except Exception as e:  # noqa: BLE001
        log.debug("RSS nicht ermittelbar: %s", e)
    return {
        "pid": os.getpid(),
        "rss_bytes": rss,
        "threads": threading.active_count(),
        "laufzeit_s": round(max(0.0, time.time() - _START_TIME), 1),
    }


_DIR_CACHE: dict[str, tuple[float, int | None]] = {}
DIR_CACHE_TTL_S = 30.0


def dir_size(path: str | None, limit: int = MAX_DIR_FILES, *, cache: bool = True) -> int | None:
    """Rekursive Größe eines Verzeichnisses in Bytes (Symlinks nicht gefolgt), höchstens ``limit`` Dateien.
    Das Ergebnis wird 30 s zwischengespeichert, damit der Live-Verlauf im HUD die Platte nicht dauernd durchsucht."""
    if not path or not os.path.isdir(path):
        return None
    key = os.path.abspath(path)
    now = time.monotonic()
    if cache:
        hit = _DIR_CACHE.get(key)
        if hit and now - hit[0] < DIR_CACHE_TTL_S:
            return hit[1]
    total = 0
    count = 0
    for root, dirs, files in os.walk(path, followlinks=False, onerror=lambda e: None):
        for name in files:
            if count >= limit:
                log.debug("Größenermittlung nach %d Dateien abgebrochen: %s", limit, path)
                _DIR_CACHE[key] = (now, total)
                return total
            count += 1
            try:
                st = os.lstat(os.path.join(root, name))
            except OSError:
                continue
            total += st.st_size
    _DIR_CACHE[key] = (now, total)
    return total


def loaded_models(backend: "LLMBackend | None") -> list[dict]:
    """Geladene Modelle aus ``backend.running()`` als ``{"name", "groesse", "vram"}``; bei Fehler ``[]``."""
    if backend is None:
        return []
    runner = getattr(backend, "running", None)
    if not callable(runner):
        return []
    try:
        entries = runner() or []
    except Exception as e:  # noqa: BLE001 - Backend darf die Messung nicht abbrechen
        log.debug("backend.running() fehlgeschlagen: %s", e)
        return []
    models: list[dict] = []
    for entry in entries:
        if isinstance(entry, dict):
            name = entry.get("name") or entry.get("model")
            size = entry.get("size", entry.get("groesse"))
            vram = entry.get("size_vram")
        else:
            name = getattr(entry, "name", None)
            size = getattr(entry, "size", None)
            vram = getattr(entry, "size_vram", None)
        if not name:
            continue
        models.append({"name": str(name), "groesse": _to_int(size), "vram": _to_int(vram)})
    return models


def snapshot(backend: "LLMBackend | None" = None, data_dir: str | None = None) -> dict:
    """Vollständige Momentaufnahme: Plattform, CPU, RAM, Platte, GPU, Prozess, Modelle, Datenverzeichnis."""
    return {
        "zeit": time.time(),
        "plattform": {"system": _system(), "release": platform.release(), "python": platform.python_version()},
        "cpu": cpu(),
        "ram": memory(),
        "platte": disk(data_dir),
        "gpu": gpu(),
        "prozess": process(),
        "modelle_geladen": loaded_models(backend),
        "datenverzeichnis": data_dir,
        "datenverzeichnis_bytes": dir_size(data_dir),
    }


# ------------------------------------------------------------------ Empfehlungen
def _max_vram_mb(gpus: Sequence[dict]) -> int | None:
    values = [g.get("vram_gesamt_mb") for g in gpus if isinstance(g, dict) and g.get("vram_gesamt_mb")]
    return max(values) if values else None


def recommend(snap: dict) -> list[str]:
    """Deutsche Hinweise zu Modellwahl und Engpässen auf Basis eines Snapshots."""
    hints: list[str] = []
    gpus = snap.get("gpu") or []
    ram = snap.get("ram") or {}
    plate = snap.get("platte") or {}
    cpu_info = snap.get("cpu") or {}

    vram_mb = _max_vram_mb(gpus)
    if not gpus or vram_mb is None:
        if gpus:
            hints.append("GPU erkannt, aber VRAM-Größe unbekannt – Modellempfehlung nicht möglich.")
        else:
            hints.append("Keine NVIDIA-GPU erkannt (nvidia-smi fehlt oder keine Karte): Modelle laufen auf der CPU. "
                         "Kleine Modelle bevorzugen (qwen2.5:3b, bei ≥ 16 GB RAM qwen2.5:7b), Antworten dauern länger.")
    else:
        vram_gb = vram_mb / 1024.0
        chosen = None
        for min_gb, model, num_ctx, extra in VRAM_CLASSES:
            if vram_gb >= min_gb - 0.5:          # 7,8 GB nutzbarer VRAM zählt noch als 8-GB-Karte
                chosen = (min_gb, model, num_ctx, extra)
                break
        if chosen is None:
            hints.append(f"{_fmt_de(vram_gb, 1)} GB VRAM: zu wenig für 3B-Modelle auf der GPU – qwen2.5:1.5b oder "
                         "CPU-Betrieb mit num_ctx 2048.")
        else:
            min_gb, model, num_ctx, extra = chosen
            hints.append(f"{min_gb} GB VRAM: {model} mit num_ctx {num_ctx} ({extra}).")

    for g in gpus:
        if not isinstance(g, dict):
            continue
        temp = g.get("temperatur_c")
        if temp is not None and temp > GPU_TEMP_WARN_C:
            hints.append(f"GPU »{g.get('name') or '?'}« ist {_fmt_de(temp, 0)} °C heiß (> {GPU_TEMP_WARN_C:.0f} °C): "
                         "Kühlung prüfen, Dauerlast reduzieren.")
        total_mb = g.get("vram_gesamt_mb")
        used_mb = g.get("vram_belegt_mb")
        if total_mb and used_mb is not None and used_mb / total_mb * 100.0 > VRAM_USED_WARN_PERCENT:
            hints.append(f"VRAM von »{g.get('name') or '?'}« zu {_fmt_de(used_mb / total_mb * 100.0, 0)} % belegt – "
                         "kleineres Modell oder geringeren num_ctx wählen, sonst lagert Ollama ins RAM aus.")

    total_ram = ram.get("gesamt_bytes")
    if total_ram is not None and total_ram < RAM_WARN_BYTES:
        hints.append(f"Nur {fmt_gb(total_ram)} RAM (< 8 GB): 7B-Modelle passen nicht ohne GPU – qwen2.5:3b verwenden "
                     "und andere Programme schließen.")
    ram_pct = ram.get("belegt_prozent")
    if ram_pct is not None and ram_pct > 90:
        hints.append(f"RAM zu {_fmt_de(ram_pct, 0)} % belegt – Speicher wird knapp.")

    free_disk = plate.get("frei_bytes")
    if free_disk is not None and free_disk < DISK_WARN_FREE_BYTES:
        hints.append(f"Nur {fmt_gb(free_disk)} frei auf der Platte (< 5 GB): neue Modelle (4–20 GB) lassen sich nicht "
                     "laden – Speicher freigeben.")

    load = cpu_info.get("last_prozent")
    if load is not None and load > 90:
        hints.append(f"CPU-Last {_fmt_de(load, 0)} %: andere Prozesse bremsen das Modell.")

    return hints


# ------------------------------------------------------------------ Verlauf
class Sampler:
    """Thread-sicherer Ringpuffer der letzten ``size`` Snapshots (für den HUD-Verlauf)."""

    def __init__(self, size: int = 120):
        self.size = max(1, int(size))
        self._buf: collections.deque[dict] = collections.deque(maxlen=self.size)
        self._lock = threading.Lock()

    def add(self, snap: dict) -> None:
        if not isinstance(snap, dict):
            raise TypeError("Sampler.add erwartet einen Snapshot (dict).")
        with self._lock:
            self._buf.append(snap)

    def last(self) -> dict | None:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def series(self) -> dict:
        """``{"zeit", "cpu", "ram", "gpu", "vram"}`` – Listen gleicher Länge, ``None`` für fehlende Werte."""
        with self._lock:
            snaps = list(self._buf)
        out: dict[str, list] = {"zeit": [], "cpu": [], "ram": [], "gpu": [], "vram": []}
        for s in snaps:
            out["zeit"].append(s.get("zeit"))
            out["cpu"].append((s.get("cpu") or {}).get("last_prozent"))
            out["ram"].append((s.get("ram") or {}).get("belegt_prozent"))
            gpus = s.get("gpu") or []
            first = gpus[0] if gpus and isinstance(gpus[0], dict) else {}
            out["gpu"].append(first.get("auslastung_prozent"))
            total_mb = first.get("vram_gesamt_mb")
            used_mb = first.get("vram_belegt_mb")
            vram = _clamp_percent(used_mb / total_mb * 100.0) if total_mb and used_mb is not None else None
            out["vram"].append(vram)
        return out


# ------------------------------------------------------------------ Darstellung
def _fmt_de(value: float, digits: int = 1) -> str:
    """Zahl mit deutschem Dezimalkomma."""
    return f"{value:.{digits}f}".replace(".", ",")


def fmt_gb(value: int | float | None, digits: int = 1) -> str:
    """Bytes als Gigabyte mit Komma, z. B. ``15,6 GB``; ``None`` → ``unbekannt``."""
    if value is None:
        return "unbekannt"
    return f"{_fmt_de(value / GB, digits)} GB"


def fmt_mb(value: int | float | None) -> str:
    if value is None:
        return "unbekannt"
    return f"{_fmt_de(value / MB, 1)} MB"


def _fmt_percent(value: float | None, digits: int = 1) -> str:
    return "unbekannt" if value is None else f"{_fmt_de(value, digits)} %"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unbekannt"
    s = int(seconds)
    h, rest = divmod(s, 3600)
    m, sec = divmod(rest, 60)
    return f"{h}:{m:02d}:{sec:02d}"


def format_snapshot(snap: dict) -> str:
    """Mehrzeiliger deutscher Statusbericht aus einem Snapshot (inkl. Empfehlungen)."""
    lines: list[str] = []
    plat = snap.get("plattform") or {}
    lines.append(f"System: {plat.get('system') or 'unbekannt'} {plat.get('release') or ''}".rstrip()
                 + f" · Python {plat.get('python') or '?'}")

    c = snap.get("cpu") or {}
    cores = c.get("kerne")
    logical = c.get("logisch")
    core_txt = f"{cores} Kerne" if cores is not None else "Kerne unbekannt"
    if logical is not None:
        core_txt += f" / {logical} Threads"
    lines.append(f"CPU: {c.get('name') or 'unbekannt'} – {core_txt}, Last {_fmt_percent(c.get('last_prozent'))}")

    r = snap.get("ram") or {}
    lines.append(f"RAM: {fmt_gb(r.get('gesamt_bytes'))} gesamt, {fmt_gb(r.get('frei_bytes'))} frei "
                 f"({_fmt_percent(r.get('belegt_prozent'))} belegt)")

    p = snap.get("platte") or {}
    path_txt = f" ({p['pfad']})" if p.get("pfad") else ""
    lines.append(f"Platte{path_txt}: {fmt_gb(p.get('gesamt_bytes'))} gesamt, {fmt_gb(p.get('frei_bytes'))} frei "
                 f"({_fmt_percent(p.get('belegt_prozent'))} belegt)")

    gpus = snap.get("gpu") or []
    if not gpus:
        lines.append("GPU: keine NVIDIA-GPU erkannt (nvidia-smi nicht verfügbar)")
    for i, g in enumerate(gpus, 1):
        total_mb = g.get("vram_gesamt_mb")
        used_mb = g.get("vram_belegt_mb")
        vram_txt = (f"{_fmt_de(used_mb / 1024, 1) if used_mb is not None else '?'}/"
                    f"{_fmt_de(total_mb / 1024, 1) if total_mb is not None else '?'} GB")
        temp = g.get("temperatur_c")
        temp_txt = f"{_fmt_de(temp, 0)} °C" if temp is not None else "Temperatur unbekannt"
        lines.append(f"GPU {i}: {g.get('name') or 'unbekannt'} – VRAM {vram_txt}, "
                     f"Auslastung {_fmt_percent(g.get('auslastung_prozent'), 0)}, {temp_txt}")

    models = snap.get("modelle_geladen") or []
    if models:
        parts = []
        for m in models:
            size = m.get("groesse")
            parts.append(f"{m.get('name')} ({fmt_gb(size)})" if size is not None else str(m.get("name")))
        lines.append("Modelle geladen: " + ", ".join(parts))
    else:
        lines.append("Modelle geladen: keine")

    if snap.get("datenverzeichnis_bytes") is not None:
        lines.append(f"Datenverzeichnis: {fmt_gb(snap['datenverzeichnis_bytes'], 2)}")

    pr = snap.get("prozess") or {}
    lines.append(f"Prozess: RSS {fmt_mb(pr.get('rss_bytes'))}, {pr.get('threads', '?')} Threads, "
                 f"Laufzeit {_fmt_duration(pr.get('laufzeit_s'))}")

    hints = recommend(snap)
    if hints:
        lines.append("Hinweise:")
        lines.extend(f"- {h}" for h in hints)
    return "\n".join(lines)


# ------------------------------------------------------------------ Werkzeuge
def _schema(props: dict, required: Sequence[str]) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


TOOL_NAMES = ("system_status",)


def register_tools(registry: "ToolRegistry", backend: "LLMBackend | None" = None, data_dir: str | None = None,
                   sampler: Sampler | None = None) -> None:
    """Registriert ``system_status()`` (nicht gefährlich). Optional werden Backend, Datenverzeichnis und ein
    :class:`Sampler` mitgegeben, damit das Werkzeug geladene Modelle meldet und den Verlauf füllt."""
    from .tools import Tool

    def _system_status() -> str:
        snap = snapshot(backend=backend, data_dir=data_dir or getattr(registry, "workspace", None))
        if sampler is not None:
            sampler.add(snap)
        return format_snapshot(snap)

    registry.register(Tool(
        name="system_status",
        description="Systemstatus des Rechners: CPU, RAM, Platte, GPU/VRAM, geladene Modelle, eigener Prozess "
                    "und Empfehlungen zur Modellwahl (echte Messwerte, nicht messbare Werte als »unbekannt«).",
        parameters=_schema({}, []),
        fn=_system_status,
        dangerous=False,
    ))


__all__ = [
    "CPU_SAMPLE_INTERVAL", "GB", "MAX_DIR_FILES", "MB", "NVIDIA_SMI_CMD", "POWERSHELL_CPU_CMD", "TOOL_NAMES",
    "VRAM_CLASSES", "Sampler", "cpu", "cpu_load_from_samples", "dir_size", "disk", "fmt_gb", "fmt_mb",
    "format_snapshot", "gpu", "loaded_models", "memory", "parse_cpuinfo_name", "parse_cpuinfo_physical",
    "parse_meminfo", "parse_nvidia_smi", "parse_powershell_cpu", "parse_proc_stat", "parse_status_rss",
    "parse_vm_stat", "process", "recommend", "register_tools", "snapshot",
]
