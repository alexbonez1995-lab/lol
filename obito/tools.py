"""Werkzeuge von OBITO (100 % lokal, nur Standardbibliothek).

Das Modell fordert ein Werkzeug mit einem Block der Form

    <werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>

an. :func:`parse_tool_calls` erkennt diese Blöcke (und einige tolerante Varianten),
:class:`ToolRegistry` führt sie abgesichert aus (Sandbox auf ein Arbeitsverzeichnis,
Typ-Koerzion, Pflichtparameter, Freigabe gefährlicher Werkzeuge, Ausgabe-Kürzung) und
:class:`ToolStreamFilter` sorgt dafür, dass ein Werkzeug-Block beim Streaming nicht beim
Nutzer ankommt.
"""

from __future__ import annotations

import ast
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .llm import StreamCallback, parse_json

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfer
    from .memory import MemoryStore

MAX_OUTPUT_CHARS = 4000          # maximale Länge einer Werkzeugausgabe
MAX_DIR_ENTRIES = 200            # Einträge je Verzeichnisauflistung
MAX_READ_FILE_BYTES = 5 * 1024 * 1024     # datei_lesen lehnt größere Dateien ab
MAX_WRITE_BYTES = 1024 * 1024             # datei_schreiben schreibt höchstens 1 MB
MAX_READ_CHARS = 20000           # Obergrenze für max_zeichen bei datei_lesen
MAX_PYTHON_TIMEOUT = 120
MAX_SHELL_TIMEOUT = 300
MAX_EXPRESSION_CHARS = 500

ConfirmCallback = Callable[[str, dict], bool]


# ------------------------------------------------------------------ Datenklassen
@dataclass
class ToolResult:
    """Ergebnis eines Werkzeugaufrufs. ``output`` ist auf :data:`MAX_OUTPUT_CHARS` begrenzt."""

    ok: bool
    output: str
    error: str | None = None


@dataclass
class Tool:
    """Ein registrierbares Werkzeug mit JSON-Schema für die Parameter."""

    name: str
    description: str
    parameters: dict          # {"type": "object", "properties": {...}, "required": [...]}
    fn: Callable[..., str]
    dangerous: bool = False


# ------------------------------------------------------------------ Hilfen
def truncate_output(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    """Kürzt eine Ausgabe auf ``limit`` Zeichen und hängt einen Hinweis mit der Gesamtlänge an."""
    if text is None:
        return ""
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [gekürzt, {len(text)} Zeichen insgesamt]"


_TRUE_WORDS = {"1", "true", "ja", "wahr", "yes", "on", "an", "j", "y"}
_FALSE_WORDS = {"0", "false", "nein", "falsch", "no", "off", "aus", "n", ""}


def _schema_type(spec: Any) -> str | None:
    """Liest den (ersten nicht-null) Typ aus einem Parameter-Schema."""
    if not isinstance(spec, dict):
        return None
    typ = spec.get("type")
    if isinstance(typ, list):
        for t in typ:
            if t != "null":
                return str(t)
        return None
    return str(typ) if typ else None


def coerce_value(key: str, value: Any, typ: str | None) -> Any:
    """Wandelt ``value`` nach dem Schema-Typ (``integer``, ``number``, ``boolean``, ``string``,
    ``array``, ``object``). Strings werden geparst; Unpassendes löst ``ValueError`` aus."""
    if typ is None or value is None:
        return value
    if typ == "integer":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f"Parameter »{key}« muss eine ganze Zahl sein, nicht {value!r}")
        if isinstance(value, str):
            s = value.strip().replace(",", ".")
            try:
                return int(s)
            except ValueError:
                pass
            try:
                f = float(s)
            except ValueError:
                raise ValueError(f"Parameter »{key}« muss eine ganze Zahl sein, nicht {value!r}") from None
            if f.is_integer():
                return int(f)
        raise ValueError(f"Parameter »{key}« muss eine ganze Zahl sein, nicht {value!r}")
    if typ == "number":
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip().replace(",", "."))
            except ValueError:
                raise ValueError(f"Parameter »{key}« muss eine Zahl sein, nicht {value!r}") from None
        raise ValueError(f"Parameter »{key}« muss eine Zahl sein, nicht {value!r}")
    if typ == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            s = value.strip().lower()
            if s in _TRUE_WORDS:
                return True
            if s in _FALSE_WORDS:
                return False
        raise ValueError(f"Parameter »{key}« muss ja/nein sein, nicht {value!r}")
    if typ == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)
    if typ == "array":
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            parsed = parse_json(value)
            if isinstance(parsed, list):
                return parsed
        return [value]
    if typ == "object":
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            parsed = parse_json(value)
            if isinstance(parsed, dict):
                return parsed
        raise ValueError(f"Parameter »{key}« muss ein Objekt sein, nicht {value!r}")
    return value


def _fmt_bytes(n: int | float) -> str:
    """Formatiert Bytes lesbar (deutsches Dezimalkomma)."""
    n = float(n)
    for unit in ("Bytes", "kB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            if unit == "Bytes":
                return f"{int(n)} Bytes"
            return f"{n:.1f} {unit}".replace(".", ",")
        n /= 1024
    return f"{n:.1f} TB".replace(".", ",")


def _format_number(value: float | int) -> str:
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.12g}"
    return str(value)


# ------------------------------------------------------------------ Registry
class ToolRegistry:
    """Verwaltet Werkzeuge, beschreibt sie für das Modell und führt sie abgesichert aus.

    Alle Dateizugriffe laufen über :meth:`resolve` und bleiben damit innerhalb von
    ``self.workspace`` (realer, symlink-aufgelöster Pfad)."""

    def __init__(self, workspace: str = ".", confirm: ConfirmCallback | None = None,
                 confirm_dangerous: bool = True):
        self.workspace = os.path.realpath(workspace or ".")
        self.confirm = confirm
        self.confirm_dangerous = bool(confirm_dangerous)
        self.memory: "MemoryStore | None" = None
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------ Verwaltung
    def register(self, tool: Tool) -> None:
        if not isinstance(tool, Tool):
            raise TypeError("register erwartet ein Tool-Objekt.")
        if not tool.name or not isinstance(tool.name, str):
            raise ValueError("Werkzeug ohne Namen kann nicht registriert werden.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return [self._tools[k] for k in self._tools]

    def set_policy(self, confirm: ConfirmCallback | None, confirm_dangerous: bool) -> None:
        """Setzt die Freigabe-Regel für gefährliche Werkzeuge."""
        self.confirm = confirm
        self.confirm_dangerous = bool(confirm_dangerous)

    def set_memory(self, memory: "MemoryStore | None") -> None:
        """Verbindet ein Gedächtnis und aktiviert ``gedaechtnis_suchen`` / ``gedaechtnis_merken``.
        ``None`` entfernt beide Werkzeuge wieder."""
        self.memory = memory
        if memory is None:
            self.unregister("gedaechtnis_suchen")
            self.unregister("gedaechtnis_merken")
            return
        for tool in _memory_tools(self):
            self.register(tool)

    # ------------------------------------------------------------ Sandbox
    def resolve(self, pfad: str) -> str:
        """Löst ``pfad`` relativ zum Arbeitsbereich auf (realpath). Liegt das Ziel – auch über
        Symlinks oder ``..`` – außerhalb des Arbeitsbereichs, folgt ``PermissionError``."""
        if pfad is None:
            pfad = "."
        p = str(pfad).strip() or "."
        if "\x00" in p:
            raise ValueError("Pfad enthält ein Nullbyte.")
        candidate = p if os.path.isabs(p) else os.path.join(self.workspace, p)
        real = os.path.realpath(candidate)
        base = os.path.normcase(self.workspace)
        target = os.path.normcase(real)
        if target != base and not target.startswith(base.rstrip(os.sep) + os.sep):
            raise PermissionError(f"Pfad außerhalb des Arbeitsbereichs: {pfad}")
        return real

    def relpath(self, real: str) -> str:
        """Pfad relativ zum Arbeitsbereich (für Ausgaben)."""
        try:
            rel = os.path.relpath(real, self.workspace)
        except ValueError:
            return real
        return "." if rel == "." else rel.replace(os.sep, "/")

    # ------------------------------------------------------------ Beschreibung
    @staticmethod
    def _signature(tool: Tool) -> str:
        props = (tool.parameters or {}).get("properties") or {}
        required = list((tool.parameters or {}).get("required") or [])
        parts = []
        for name in required:
            if name in props:
                parts.append(name)
        for name, spec in props.items():
            if name in required:
                continue
            if isinstance(spec, dict) and "default" in spec:
                parts.append(f"{name}={json.dumps(spec['default'], ensure_ascii=False)}")
            else:
                parts.append(f"{name}?")
        return f"{tool.name}({', '.join(parts)})"

    @staticmethod
    def _example_args(tool: Tool) -> dict:
        props = (tool.parameters or {}).get("properties") or {}
        required = list((tool.parameters or {}).get("required") or [])
        example: dict[str, Any] = {}
        for name in required:
            spec = props.get(name) if isinstance(props.get(name), dict) else {}
            if "example" in spec:
                example[name] = spec["example"]
            else:
                typ = _schema_type(spec)
                example[name] = {"integer": 1, "number": 1.0, "boolean": True,
                                 "array": [], "object": {}}.get(typ or "string", "…")
        return example

    def describe(self, compact: bool = True) -> str:
        """Beschreibung aller Werkzeuge für den System-Prompt.

        ``compact=True``: eine Zeile je Werkzeug plus das exakte Aufrufformat."""
        tools = self.list()
        if not tools:
            return "Keine Werkzeuge verfügbar."
        lines = ["Verfügbare Werkzeuge:"]
        if compact:
            for t in tools:
                flag = " [gefährlich – Nutzer muss zustimmen]" if t.dangerous else ""
                lines.append(f"- {self._signature(t)}: {t.description}{flag}")
        else:
            for t in tools:
                lines.append("")
                lines.append(self.describe_one(t.name))
        lines.append("")
        lines.append("Aufruf: Schreibe genau einen Block pro Werkzeug am Ende deiner Antwort, exakt so:")
        lines.append('<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>')
        lines.append("Danach erhältst du das Ergebnis als Nachricht und antwortest dem Nutzer damit. "
                     "Nutze Werkzeuge nur, wenn sie wirklich nötig sind; erfinde nie Ergebnisse.")
        return "\n".join(lines)

    def describe_one(self, name: str) -> str:
        """Ausführliche Beschreibung eines Werkzeugs mit allen Parametern."""
        tool = self.get(name)
        if tool is None:
            return f"Unbekanntes Werkzeug »{name}«."
        props = (tool.parameters or {}).get("properties") or {}
        required = set((tool.parameters or {}).get("required") or [])
        lines = [f"{self._signature(tool)} – {tool.description}"]
        if props:
            lines.append("Parameter:")
            for pname, spec in props.items():
                spec = spec if isinstance(spec, dict) else {}
                typ = _schema_type(spec) or "string"
                kind = "Pflicht" if pname in required else "optional"
                if "default" in spec and pname not in required:
                    kind += f", Standard {json.dumps(spec['default'], ensure_ascii=False)}"
                desc = spec.get("description") or ""
                lines.append(f"  - {pname} ({typ}, {kind}){': ' + desc if desc else ''}")
        else:
            lines.append("Parameter: keine")
        lines.append("Gefährlich: " + ("ja (Bestätigung durch den Nutzer nötig)" if tool.dangerous else "nein"))
        call = {"name": tool.name, "args": self._example_args(tool)}
        lines.append("Aufruf: <werkzeug>" + json.dumps(call, ensure_ascii=False) + "</werkzeug>")
        return "\n".join(lines)

    # ------------------------------------------------------------ Ausführung
    def run(self, name: str, args: dict) -> ToolResult:
        """Führt ein Werkzeug aus. Liefert nie eine Exception, sondern immer ein ``ToolResult``."""
        name = str(name) if name is not None else ""
        tool = self.get(name)
        if tool is None:
            available = ", ".join(self._tools) or "keine"
            return ToolResult(False, "", f"Unbekanntes Werkzeug »{name}«. Verfügbar: {available}")
        if not isinstance(args, dict):
            args = {}
        try:
            props = (tool.parameters or {}).get("properties") or {}
            required = list((tool.parameters or {}).get("required") or [])
            clean: dict[str, Any] = {}
            for key, value in args.items():
                if key not in props:
                    continue                      # unbekannte Schlüssel verwerfen
                coerced = coerce_value(key, value, _schema_type(props[key]))
                if coerced is None:
                    continue
                clean[key] = coerced
            missing = [r for r in required if r not in clean]
            if missing:
                return ToolResult(False, "", "Fehlende Parameter: " + ", ".join(missing))
            allowed = (not tool.dangerous) or (not self.confirm_dangerous) or (
                self.confirm is not None and bool(self.confirm(name, clean)))
            if not allowed:
                return ToolResult(False, "", "Vom Nutzer abgelehnt")
            output = tool.fn(**clean)
            if output is None:
                output = ""
            elif not isinstance(output, str):
                output = str(output)
            return ToolResult(True, truncate_output(output))
        except Exception as e:  # noqa: BLE001 - jede Ausnahme wird zum Fehlerergebnis
            msg = str(e) or e.__class__.__name__
            return ToolResult(False, "", msg)


# ------------------------------------------------------------------ rechnen
_ALLOWED_BINOPS: dict[type, Callable[[Any, Any], Any] | None] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: None,  # Sonderbehandlung mit Limits (siehe _eval_node)
}
_ALLOWED_UNARY: dict[type, Callable[[Any], Any]] = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}
_ALLOWED_FUNCS: dict[str, Callable[..., Any]] = {"abs": abs, "round": round, "min": min, "max": max}
_ALLOWED_NAMES: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}
_POW_MAX_EXPONENT = 1000
_POW_MAX_OPERAND = 10 ** 12


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if _is_number(node.value):
            return node.value
        raise ValueError(f"Nicht erlaubt: Konstante {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Nicht erlaubt: Name »{node.id}«")
    if isinstance(node, ast.Attribute):
        # math.pi / math.e / math.tau als Konstanten
        if isinstance(node.value, ast.Name) and node.value.id == "math" and node.attr in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.attr]
        raise ValueError("Nicht erlaubt: Attributzugriff")
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARY.get(type(node.op))
        if op is None:
            raise ValueError(f"Nicht erlaubt: Operator {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        kind = type(node.op)
        if kind not in _ALLOWED_BINOPS:
            raise ValueError(f"Nicht erlaubt: Operator {kind.__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if kind is ast.Pow:
            if not (_is_number(left) and _is_number(right)):
                raise ValueError("Nicht erlaubt: Potenz mit nicht-numerischen Operanden")
            if abs(right) > _POW_MAX_EXPONENT:
                raise ValueError(f"Nicht erlaubt: Exponent {right!r} (max. ±{_POW_MAX_EXPONENT})")
            if abs(left) >= _POW_MAX_OPERAND or abs(right) >= _POW_MAX_OPERAND:
                raise ValueError(f"Nicht erlaubt: Operand ≥ 10^12 bei Potenz")
            try:
                return left ** right
            except OverflowError:
                raise ValueError("Nicht erlaubt: Ergebnis zu groß") from None
        op = _ALLOWED_BINOPS[kind]
        if op is None:  # pragma: no cover - Pow wurde oben behandelt
            raise ValueError(f"Nicht erlaubt: Operator {kind.__name__}")
        try:
            return op(left, right)
        except ZeroDivisionError:
            raise ValueError("Division durch null") from None
    if isinstance(node, ast.Call):
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise ValueError("Nicht erlaubt: Schlüsselwort- oder Stern-Argumente")
        func = node.func
        if isinstance(func, ast.Name):
            fn = _ALLOWED_FUNCS.get(func.id)
            if fn is None:
                # math.<name> auch ohne Präfix erlauben (z. B. sqrt(2))
                fn = _math_function(func.id)
            if fn is None:
                raise ValueError(f"Nicht erlaubt: Funktion »{func.id}«")
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "math":
            fn = _math_function(func.attr)
            if fn is None:
                raise ValueError(f"Nicht erlaubt: Funktion »math.{func.attr}«")
        else:
            raise ValueError("Nicht erlaubt: Aufruf")
        argv = [_eval_node(a) for a in node.args]
        if any(not _is_number(a) for a in argv):
            raise ValueError("Nicht erlaubt: nicht-numerische Argumente")
        try:
            return fn(*argv)
        except (ValueError, OverflowError, TypeError) as e:
            raise ValueError(f"Rechenfehler: {e}") from None
    raise ValueError(f"Nicht erlaubt: {type(node).__name__}")


def _math_function(attr: str) -> Callable[..., Any] | None:
    if attr.startswith("_"):
        return None
    fn = getattr(math, attr, None)
    return fn if callable(fn) else None


def calculate(ausdruck: str) -> str:
    """Rechnet einen arithmetischen Ausdruck sicher aus (``ast``-Whitelist, kein ``eval``)."""
    expr = (ausdruck or "").strip()
    if not expr:
        raise ValueError("Leerer Ausdruck.")
    if len(expr) > MAX_EXPRESSION_CHARS:
        raise ValueError(f"Ausdruck zu lang (max. {MAX_EXPRESSION_CHARS} Zeichen).")
    expr = expr.replace("^", "**").replace("×", "*").replace("÷", "/").replace("·", "*")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Ungültiger Ausdruck: {e.msg}") from None
    result = _eval_node(tree)
    if isinstance(result, complex):
        raise ValueError("Nicht erlaubt: komplexes Ergebnis")
    return f"{expr} = {_format_number(result)}"


# ------------------------------------------------------------------ zeit / system
_WEEKDAYS = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")


def current_time() -> str:
    """Aktuelles Datum und Uhrzeit (lokale Zeitzone)."""
    now = time.time()
    lt = time.localtime(now)
    offset = lt.tm_gmtoff if lt.tm_gmtoff is not None else 0
    sign = "+" if offset >= 0 else "-"
    hh, mm = divmod(abs(offset) // 60, 60)
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", lt) + f"{sign}{hh:02d}:{mm:02d}"
    week = time.strftime("%V", lt)
    return (
        f"Datum: {_WEEKDAYS[lt.tm_wday]}, {time.strftime('%d.%m.%Y', lt)}\n"
        f"Uhrzeit: {time.strftime('%H:%M:%S', lt)} ({lt.tm_zone or 'lokal'}, UTC{sign}{hh:02d}:{mm:02d})\n"
        f"Kalenderwoche: {week}\n"
        f"ISO: {iso}"
    )


def _memory_info() -> tuple[int | None, int | None]:
    """(gesamt, verfügbar) in Bytes – ohne Netzwerk, ohne Fremdpakete; ``None`` wenn unbekannt."""
    total: int | None = None
    avail: int | None = None
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        avail = int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    if total is None and hasattr(os, "sysconf"):
        try:
            total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
        except (OSError, ValueError, AttributeError):
            total = None
        try:
            if avail is None:
                avail = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_AVPHYS_PAGES"))
        except (OSError, ValueError, AttributeError):
            avail = None
    if total is None and sys.platform.startswith("win"):
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
                total = int(status.ullTotalPhys)
                avail = int(status.ullAvailPhys)
        except Exception:  # noqa: BLE001
            pass
    return total, avail


def system_info(workspace: str = ".") -> str:
    """Betriebssystem, Python, CPUs, RAM und Platte – ohne Netzwerkaufrufe."""
    lines = [
        f"Betriebssystem: {platform.system()} {platform.release()} ({platform.machine() or 'unbekannt'})",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"Prozessoren: {os.cpu_count() or 'unbekannt'}",
    ]
    total, avail = _memory_info()
    if total:
        ram = f"Arbeitsspeicher: {_fmt_bytes(total)}"
        if avail is not None:
            ram += f" (frei: {_fmt_bytes(avail)})"
        lines.append(ram)
    else:
        lines.append("Arbeitsspeicher: nicht ermittelbar")
    try:
        usage = shutil.disk_usage(workspace)
        lines.append(f"Festplatte (Arbeitsbereich): {_fmt_bytes(usage.total)} gesamt, {_fmt_bytes(usage.free)} frei")
    except OSError:
        lines.append("Festplatte: nicht ermittelbar")
    lines.append(f"Arbeitsbereich: {os.path.realpath(workspace)}")
    return "\n".join(lines)


# ------------------------------------------------------------------ Prozesse
def _minimal_env() -> dict[str, str]:
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONIOENCODING": "utf-8"}
    if sys.platform.startswith("win"):
        for key in ("SYSTEMROOT", "TEMP", "TMP"):
            if os.environ.get(key):
                env[key] = os.environ[key]
    return env


def _format_process_output(stdout: str, stderr: str, code: int) -> str:
    out = stdout.rstrip("\n") if stdout else ""
    if stderr and stderr.strip():
        out += "\n[stderr]\n" + stderr.rstrip("\n")
    return out + f"\n[exit {code}]"


def _run_process(cmd: Sequence[str] | str, *, cwd: str, timeout: int, shell: bool,
                 env: dict[str, str] | None) -> str:
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, shell=shell, env=env, timeout=timeout,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Zeitlimit ({timeout} s) überschritten") from None
    stdout = (proc.stdout or b"").decode("utf-8", errors="replace")
    stderr = (proc.stderr or b"").decode("utf-8", errors="replace")
    return _format_process_output(stdout, stderr, proc.returncode)


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(low, min(high, v))


# ------------------------------------------------------------------ eingebaute Werkzeuge
def _builtin_tools(reg: ToolRegistry) -> list[Tool]:
    """Erzeugt die eingebauten Werkzeuge, gebunden an die Sandbox der Registry."""

    def verzeichnis(pfad: str = ".") -> str:
        real = reg.resolve(pfad)
        if not os.path.isdir(real):
            if os.path.exists(real):
                raise NotADirectoryError(f"Kein Verzeichnis: {reg.relpath(real)}")
            raise FileNotFoundError(f"Verzeichnis nicht gefunden: {pfad}")
        entries = []
        with os.scandir(real) as it:
            for entry in it:
                try:
                    is_dir = entry.is_dir()
                except OSError:
                    is_dir = False
                size = None
                if not is_dir:
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = None
                entries.append((entry.name, is_dir, size))
        entries.sort(key=lambda e: (not e[1], e[0].lower()))
        total = len(entries)
        shown = entries[:MAX_DIR_ENTRIES]
        label = reg.relpath(real)
        lines = [f"Verzeichnis {label if label == '.' else label + '/'} ({total} Einträge):"]
        for name, is_dir, size in shown:
            if is_dir:
                lines.append(f"{name}/")
            else:
                lines.append(f"{name} ({_fmt_bytes(size)})" if size is not None else name)
        if total > MAX_DIR_ENTRIES:
            lines.append(f"… und {total - MAX_DIR_ENTRIES} weitere Einträge (max. {MAX_DIR_ENTRIES} angezeigt)")
        if total == 0:
            lines.append("(leer)")
        return "\n".join(lines)

    def datei_lesen(pfad: str, max_zeichen: int = 4000) -> str:
        real = reg.resolve(pfad)
        if not os.path.exists(real):
            raise FileNotFoundError(f"Datei nicht gefunden: {pfad}")
        if os.path.isdir(real):
            raise IsADirectoryError(f"Ist ein Verzeichnis: {reg.relpath(real)} (nutze »verzeichnis«)")
        size = os.path.getsize(real)
        if size > MAX_READ_FILE_BYTES:
            raise ValueError(f"Datei zu groß ({_fmt_bytes(size)}, max. {_fmt_bytes(MAX_READ_FILE_BYTES)})")
        limit = _clamp_int(max_zeichen, 1, MAX_READ_CHARS, 4000)
        with open(real, "rb") as fh:
            data = fh.read()
        if b"\x00" in data[:8192]:
            return f"Binärdatei: {reg.relpath(real)} ({_fmt_bytes(size)}) – Inhalt wird nicht angezeigt."
        text = data.decode("utf-8", errors="replace")
        if len(text) > limit:
            return text[:limit] + f"\n… [gekürzt, {len(text)} Zeichen insgesamt]"
        return text

    def datei_schreiben(pfad: str, inhalt: str) -> str:
        if inhalt is None:
            inhalt = ""
        data = str(inhalt).encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise ValueError(f"Inhalt zu groß ({_fmt_bytes(len(data))}, max. {_fmt_bytes(MAX_WRITE_BYTES)})")
        real = reg.resolve(pfad)
        if real == reg.workspace or os.path.isdir(real):
            raise IsADirectoryError(f"Ziel ist ein Verzeichnis: {reg.relpath(real)}")
        parent = reg.resolve(os.path.dirname(real) or ".")
        os.makedirs(parent, exist_ok=True)
        existed = os.path.exists(real)
        with open(real, "wb") as fh:
            fh.write(data)
        return f"geschrieben: {reg.relpath(real)} ({len(data)} Bytes, überschrieben: {'ja' if existed else 'nein'})"

    def python_ausfuehren(code: str, timeout: int = 20) -> str:
        if not code or not str(code).strip():
            raise ValueError("Kein Code übergeben.")
        t = _clamp_int(timeout, 1, MAX_PYTHON_TIMEOUT, 20)
        return _run_process([sys.executable, "-I", "-c", str(code)], cwd=reg.workspace, timeout=t,
                            shell=False, env=_minimal_env())

    def befehl_ausfuehren(befehl: str, timeout: int = 30) -> str:
        if not befehl or not str(befehl).strip():
            raise ValueError("Kein Befehl übergeben.")
        t = _clamp_int(timeout, 1, MAX_SHELL_TIMEOUT, 30)
        return _run_process(str(befehl), cwd=reg.workspace, timeout=t, shell=True, env=None)

    return [
        Tool(
            name="rechnen",
            description="Rechnet einen mathematischen Ausdruck exakt aus (+ - * / // % **, abs, round, min, max, math.sqrt, math.sin …, pi, e).",
            parameters={"type": "object",
                        "properties": {"ausdruck": {"type": "string", "description": "Ausdruck in Python-Schreibweise, z. B. 2*21 oder math.sqrt(2)",
                                                    "example": "2*21"}},
                        "required": ["ausdruck"]},
            fn=calculate,
        ),
        Tool(
            name="zeit",
            description="Aktuelles Datum, Uhrzeit, Wochentag und Kalenderwoche (lokale Zeitzone).",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=current_time,
        ),
        Tool(
            name="system_info",
            description="Informationen über diesen Rechner: Betriebssystem, Python, CPUs, Arbeitsspeicher, Festplatte.",
            parameters={"type": "object", "properties": {}, "required": []},
            fn=lambda: system_info(reg.workspace),
        ),
        Tool(
            name="verzeichnis",
            description="Listet den Inhalt eines Verzeichnisses im Arbeitsbereich (Ordner enden mit /).",
            parameters={"type": "object",
                        "properties": {"pfad": {"type": "string", "default": ".",
                                                "description": "Pfad relativ zum Arbeitsbereich"}},
                        "required": []},
            fn=verzeichnis,
        ),
        Tool(
            name="datei_lesen",
            description="Liest eine Textdatei aus dem Arbeitsbereich.",
            parameters={"type": "object",
                        "properties": {"pfad": {"type": "string", "description": "Pfad relativ zum Arbeitsbereich",
                                                "example": "notizen.txt"},
                                       "max_zeichen": {"type": "integer", "default": 4000,
                                                       "description": "Höchstens so viele Zeichen (1–20000)"}},
                        "required": ["pfad"]},
            fn=datei_lesen,
        ),
        Tool(
            name="datei_schreiben",
            description="Schreibt (überschreibt) eine Textdatei im Arbeitsbereich, legt fehlende Ordner an.",
            parameters={"type": "object",
                        "properties": {"pfad": {"type": "string", "description": "Zielpfad relativ zum Arbeitsbereich",
                                                "example": "ausgabe.txt"},
                                       "inhalt": {"type": "string", "description": "Vollständiger Dateiinhalt (max. 1 MB)",
                                                  "example": "Hallo"}},
                        "required": ["pfad", "inhalt"]},
            fn=datei_schreiben,
            dangerous=True,
        ),
        Tool(
            name="python_ausfuehren",
            description="Führt Python-Code in einem eigenen Prozess im Arbeitsbereich aus und liefert stdout/stderr.",
            parameters={"type": "object",
                        "properties": {"code": {"type": "string", "description": "Python-Code (print für Ausgaben)",
                                                "example": "print(2**10)"},
                                       "timeout": {"type": "integer", "default": 20,
                                                   "description": "Zeitlimit in Sekunden (1–120)"}},
                        "required": ["code"]},
            fn=python_ausfuehren,
            dangerous=True,
        ),
        Tool(
            name="befehl_ausfuehren",
            description="Führt einen Shell-Befehl im Arbeitsbereich aus und liefert stdout/stderr.",
            parameters={"type": "object",
                        "properties": {"befehl": {"type": "string", "description": "Shell-Befehl",
                                                  "example": "ls -la"},
                                       "timeout": {"type": "integer", "default": 30,
                                                   "description": "Zeitlimit in Sekunden (1–300)"}},
                        "required": ["befehl"]},
            fn=befehl_ausfuehren,
            dangerous=True,
        ),
    ]


def _memory_tools(reg: ToolRegistry) -> list[Tool]:
    """Gedächtnis-Werkzeuge; greifen zur Laufzeit auf ``reg.memory`` zu."""
    from .memory import KINDS

    def gedaechtnis_suchen(frage: str) -> str:
        if reg.memory is None:
            raise RuntimeError("Kein Gedächtnis verbunden.")
        if not frage or not str(frage).strip():
            raise ValueError("Leere Suchanfrage.")
        hits = reg.memory.search(str(frage), k=6)
        if not hits:
            return "Keine passenden Erinnerungen gefunden."
        return "\n".join(m.short() for m in hits)

    def gedaechtnis_merken(inhalt: str, art: str = "notiz", wichtigkeit: float = 0.6) -> str:
        if reg.memory is None:
            raise RuntimeError("Kein Gedächtnis verbunden.")
        if not inhalt or not str(inhalt).strip():
            raise ValueError("Leere Erinnerung kann nicht gespeichert werden.")
        kind = str(art or "notiz").strip().lower()
        if kind not in KINDS:
            kind = "notiz"
        try:
            w = float(wichtigkeit)
        except (TypeError, ValueError):
            w = 0.6
        w = max(0.0, min(0.8, w))
        m = reg.memory.remember(str(inhalt).strip(), kind=kind, source="ki", importance=w)
        return f"gemerkt: #{m.id} [{m.kind}] {m.content}"

    return [
        Tool(
            name="gedaechtnis_suchen",
            description="Durchsucht das Langzeitgedächtnis nach Erinnerungen über Nutzer und Projekte.",
            parameters={"type": "object",
                        "properties": {"frage": {"type": "string", "description": "Suchbegriffe oder Frage",
                                                 "example": "Lieblingsmaterial"}},
                        "required": ["frage"]},
            fn=gedaechtnis_suchen,
        ),
        Tool(
            name="gedaechtnis_merken",
            description="Speichert einen Fakt dauerhaft im Gedächtnis (Arten: " + ", ".join(KINDS) + ").",
            parameters={"type": "object",
                        "properties": {"inhalt": {"type": "string", "description": "Der zu merkende Fakt",
                                                  "example": "Nutzer arbeitet mit Fusion 360"},
                                       "art": {"type": "string", "default": "notiz",
                                               "description": "fakt | praeferenz | entscheidung | loesung | fehler | zusammenfassung | notiz"},
                                       "wichtigkeit": {"type": "number", "default": 0.6,
                                                       "description": "0 bis 0.8"}},
                        "required": ["inhalt"]},
            fn=gedaechtnis_merken,
        ),
    ]


def default_registry(workspace: str = ".", confirm: ConfirmCallback | None = None,
                     confirm_dangerous: bool = True) -> ToolRegistry:
    """Registry mit allen eingebauten Werkzeugen (Gedächtnis-Werkzeuge erst nach ``set_memory``)."""
    reg = ToolRegistry(workspace, confirm, confirm_dangerous)
    for tool in _builtin_tools(reg):
        reg.register(tool)
    from . import engineering  # Ingenieur-Rechner & Materialdaten (importiert tools.py selbst lazy)
    engineering.register_tools(reg)
    return reg


# ------------------------------------------------------------------ Protokoll
_TAG_RE = re.compile(r"<werkzeug\s*>(.*?)</werkzeug\s*>", re.DOTALL | re.IGNORECASE)
_OPEN_TAG_RE = re.compile(r"<werkzeug\s*>(?!.*?</werkzeug\s*>)(.*)\Z", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)
_NAKED_START_RE = re.compile(r"^[ \t]*\{", re.MULTILINE)


def _call_from_obj(obj: Any, strict_keys: bool) -> dict | None:
    """Macht aus einem geparsten Objekt einen Werkzeugaufruf oder ``None``."""
    if not isinstance(obj, dict):
        return None
    if strict_keys and set(obj.keys()) != {"name", "args"}:
        return None
    name = obj.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("args")
    if not isinstance(args, dict):
        args = {}
    return {"name": name.strip(), "args": args}


def _scan_object(text: str, start: int) -> tuple[Any, int] | None:
    """Liest ab ``start`` (eine ``{``) ein JSON-Objekt bis zur passenden Klammer.
    Rückgabe ``(objekt, ende)`` oder ``None``."""
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1]), i + 1
                except json.JSONDecodeError:
                    return None
    return None


def _find_tool_spans(text: str) -> list[tuple[int, int, dict]]:
    """Alle akzeptierten Werkzeug-Formen als ``(start, ende, aufruf)``, nach Position sortiert."""
    if not text or not isinstance(text, str):
        return []
    spans: list[tuple[int, int, dict]] = []
    covered: list[tuple[int, int]] = []

    def overlaps(a: int, b: int) -> bool:
        return any(a < e and b > s for s, e in covered)

    # 1) geschlossene Tags – immer entfernen, auch bei unbrauchbarem Inhalt
    for m in _TAG_RE.finditer(text):
        call = _call_from_obj(parse_json(m.group(1)), strict_keys=False)
        covered.append((m.start(), m.end()))
        if call:
            spans.append((m.start(), m.end(), call))
    # 2) offener Block am Textende
    m = _OPEN_TAG_RE.search(text)
    if m and not overlaps(m.start(), m.end()):
        call = _call_from_obj(parse_json(m.group(1)), strict_keys=False)
        if call:
            covered.append((m.start(), m.end()))
            spans.append((m.start(), m.end(), call))
    # 3) ```json-Zaun mit {"name", "args"}
    for m in _FENCE_RE.finditer(text):
        if overlaps(m.start(), m.end()):
            continue
        inner = m.group(1).strip()
        if not inner.startswith("{"):
            continue
        try:
            obj = json.loads(inner)
        except json.JSONDecodeError:
            continue
        call = _call_from_obj(obj, strict_keys=True)
        if call:
            covered.append((m.start(), m.end()))
            spans.append((m.start(), m.end(), call))
    # 4) nacktes Objekt am Zeilenanfang mit genau den Schlüsseln name/args
    for m in _NAKED_START_RE.finditer(text):
        start = m.end() - 1
        if overlaps(start, start + 1):
            continue
        scanned = _scan_object(text, start)
        if not scanned:
            continue
        obj, end = scanned
        call = _call_from_obj(obj, strict_keys=True)
        if call and not overlaps(start, end):
            covered.append((m.start(), end))
            spans.append((m.start(), end, call))
    spans.sort(key=lambda s: s[0])
    return spans


def parse_tool_calls(text: str) -> list[dict]:
    """Erkennt Werkzeugaufrufe im Modelltext: ``<werkzeug>{…}</werkzeug>``, einen offenen
    Block am Textende, einen ```` ```json ````-Zaun mit ``{"name","args"}`` und ein nacktes
    JSON-Objekt mit genau diesen Schlüsseln am Zeilenanfang. Ungültiges wird verworfen."""
    return [call for _, _, call in _find_tool_spans(text)]


def strip_tool_calls(text: str) -> str:
    """Entfernt alle akzeptierten Werkzeug-Formen aus dem Text."""
    if not text or not isinstance(text, str):
        return text or ""
    # geschlossene Tags immer entfernen (auch wenn der Inhalt kein gültiger Aufruf war)
    spans = [(s, e) for s, e, _ in _find_tool_spans(text)]
    spans.extend((m.start(), m.end()) for m in _TAG_RE.finditer(text))
    if not spans:
        return text
    spans.sort()
    out = []
    pos = 0
    for s, e in spans:
        if s < pos:
            pos = max(pos, e)
            continue
        out.append(text[pos:s])
        pos = e
    out.append(text[pos:])
    result = "".join(out)
    result = re.sub(r"[ \t]+\n", "\n", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result


_NEEDS_TOOLS_RE = re.compile(
    r"\d\s*(?:\*\*|[-+*/×÷^%:])\s*\d|\d\s*%"                # Zahlen mit Operatoren, Prozent
    r"|\b(?:rechne|berechne|ausrechnen|wie ?viel|wieviel|prozent|wurzel|quadrat)"
    r"|\b(?:datei|dateien|ordner|verzeichnis|pfad|speicher\w*|lies\b|liest|lese\b|öffne|oeffne)"
    r"|\b(?:uhrzeit|datum|heute|wochentag|kalenderwoche|wie ?spät|wie ?spaet|zeit\b)"
    r"|\b(?:system\w*|betriebssystem|arbeitsspeicher|festplatte|prozessor|cpu|ram\b)"
    r"|\b(?:ausführ\w*|ausfuehr\w*|führe\s+\w*\s*aus|starte|skript|script|python|befehl|kommando|shell|terminal)"
    r"|\b(?:merk\w*|erinner\w*|gedächtnis|gedaechtnis|vergiss)"
    # Ingenieur-Rechner, Materialdaten und Dokumente (Phase 2)
    r"|\b(?:material\w*|werkstoff\w*|dichte|festigkeit|e-modul|zugfestigkeit|akku\w*|lipo|flugzeit|schub\w*"
    r"|drehmoment|kabel\w*|awg|umrechn\w*|einheit\w*|spannungsteiler|durchbiegung|querschnitt|propeller"
    r"|dokument\w*|datenzentrum|handbuch|datenblatt|unterlagen)",
    re.IGNORECASE,
)


def needs_tools(question: str, history: list[dict]) -> bool:
    """Heuristik: lohnt sich der Werkzeugblock im Prompt? Zahlen mit Operatoren, Rechen-,
    Datei-, Zeit-, System- und Ausführungs-Wörter – oder ein Werkzeug im letzten Assistant-Turn."""
    q = question or ""
    if _NEEDS_TOOLS_RE.search(q):
        return True
    for msg in reversed(history or []):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = str(msg.get("content") or "")
        if "<werkzeug" in content.lower() or parse_tool_calls(content):
            return True
        break
    return False


# ------------------------------------------------------------------ Streaming-Gate
class ToolStreamFilter:
    """Streaming-Gate: reicht Text sofort weiter, hält nur ein mögliches Präfix von '<werkzeug'
    zurück; sobald der Marker vollständig ist, wird nichts mehr emittiert."""

    def __init__(self, emit: StreamCallback | None, marker: str = "<werkzeug"):
        self.emit = emit
        self.marker = marker
        self._marker_low = marker.lower()
        self.buf = ""
        self.text = ""
        self.tool_detected = False
        self.emitted = ""

    def _send(self, piece: str) -> None:
        if not piece:
            return
        self.emitted += piece
        if self.emit is not None:
            self.emit(piece)

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self.text += piece
        if self.tool_detected:
            return
        self.buf += piece
        idx = self.buf.lower().find(self._marker_low)
        if idx != -1:
            self._send(self.buf[:idx])
            self.buf = ""
            self.tool_detected = True
            return
        # Ende könnte Präfix des Markers sein -> zurückhalten
        keep = 0
        low = self.buf.lower()
        for n in range(min(len(self._marker_low) - 1, len(low)), 0, -1):
            if self._marker_low.startswith(low[-n:]):
                keep = n
                break
        cut = len(self.buf) - keep
        if cut > 0:
            self._send(self.buf[:cut])
        self.buf = self.buf[cut:]

    def finish(self) -> None:
        """Leert den zurückgehaltenen Rest – nur, wenn kein Marker erkannt wurde."""
        if not self.tool_detected and self.buf:
            self._send(self.buf)
        self.buf = ""


__all__ = [
    "ConfirmCallback", "MAX_OUTPUT_CHARS", "Tool", "ToolRegistry", "ToolResult", "ToolStreamFilter",
    "calculate", "coerce_value", "current_time", "default_registry", "needs_tools", "parse_tool_calls",
    "strip_tool_calls", "system_info", "truncate_output",
]
