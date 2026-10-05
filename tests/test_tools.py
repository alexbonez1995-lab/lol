import os
import sys
import tempfile
import time
import unittest

from obito.llm import FakeBackend
from obito.memory import MemoryStore
from obito.tools import (MAX_OUTPUT_CHARS, Tool, ToolRegistry, ToolResult, ToolStreamFilter,
                         calculate, coerce_value, default_registry, needs_tools, parse_tool_calls,
                         strip_tool_calls, truncate_output)


def _schema(**props):
    required = [k for k, v in props.items() if v.pop("required", False)]
    return {"type": "object", "properties": props, "required": required}


class WorkspaceCase(unittest.TestCase):
    """Basis: eigener temporärer Arbeitsbereich je Test."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self._tmp.name)
        self.ws = os.path.join(self.root, "arbeit")
        os.makedirs(self.ws)
        self.outside = os.path.join(self.root, "draussen")
        os.makedirs(self.outside)
        with open(os.path.join(self.outside, "geheim.txt"), "w", encoding="utf-8") as fh:
            fh.write("streng geheim")
        with open(os.path.join(self.ws, "notiz.txt"), "w", encoding="utf-8") as fh:
            fh.write("Hallo Welt\nZeile 2")
        self.reg = default_registry(self.ws)

    def tearDown(self):
        self._tmp.cleanup()


# ------------------------------------------------------------------ Sandbox
class ResolveTest(WorkspaceCase):
    def test_workspace_is_realpath(self):
        self.assertEqual(self.reg.workspace, os.path.realpath(self.ws))
        self.assertEqual(ToolRegistry(".").workspace, os.path.realpath("."))

    def test_inside_paths(self):
        self.assertEqual(self.reg.resolve("notiz.txt"), os.path.join(self.ws, "notiz.txt"))
        self.assertEqual(self.reg.resolve("."), self.ws)
        self.assertEqual(self.reg.resolve(""), self.ws)
        self.assertEqual(self.reg.resolve("sub/../notiz.txt"), os.path.join(self.ws, "notiz.txt"))
        # absoluter Pfad innerhalb des Arbeitsbereichs ist erlaubt
        self.assertEqual(self.reg.resolve(os.path.join(self.ws, "notiz.txt")), os.path.join(self.ws, "notiz.txt"))

    def test_traversal_dotdot(self):
        with self.assertRaises(PermissionError):
            self.reg.resolve("../draussen/geheim.txt")
        with self.assertRaises(PermissionError):
            self.reg.resolve("sub/../../draussen/geheim.txt")
        with self.assertRaises(PermissionError):
            self.reg.resolve("..")

    def test_traversal_absolute(self):
        with self.assertRaises(PermissionError):
            self.reg.resolve(os.path.join(self.outside, "geheim.txt"))
        with self.assertRaises(PermissionError):
            self.reg.resolve(os.path.abspath(os.sep))

    def test_sibling_prefix_is_not_inside(self):
        # /tmp/x/arbeit2 darf nicht als "innerhalb /tmp/x/arbeit" gelten
        sibling = self.ws + "2"
        os.makedirs(sibling)
        with self.assertRaises(PermissionError):
            self.reg.resolve(sibling)

    def test_traversal_symlink(self):
        link = os.path.join(self.ws, "link")
        try:
            os.symlink(self.outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks werden hier nicht unterstützt")
        with self.assertRaises(PermissionError):
            self.reg.resolve("link/geheim.txt")
        with self.assertRaises(PermissionError):
            self.reg.resolve("link")
        res = self.reg.run("datei_lesen", {"pfad": "link/geheim.txt"})
        self.assertFalse(res.ok)
        self.assertIn("außerhalb", res.error)
        # Symlink auf eine Datei außerhalb
        flink = os.path.join(self.ws, "datei_link.txt")
        os.symlink(os.path.join(self.outside, "geheim.txt"), flink)
        self.assertFalse(self.reg.run("datei_lesen", {"pfad": "datei_link.txt"}).ok)

    def test_nullbyte_rejected(self):
        res = self.reg.run("datei_lesen", {"pfad": "a\x00b"})
        self.assertFalse(res.ok)

    def test_relpath(self):
        self.assertEqual(self.reg.relpath(self.ws), ".")
        self.assertEqual(self.reg.relpath(os.path.join(self.ws, "a", "b.txt")), "a/b.txt")


# ------------------------------------------------------------------ Registry-Verwaltung
class RegistryTest(WorkspaceCase):
    def test_builtin_names(self):
        names = [t.name for t in self.reg.list()]
        from obito.engineering import TOOL_NAMES
        self.assertEqual(names, ["rechnen", "zeit", "system_info", "verzeichnis", "datei_lesen",
                                 "datei_schreiben", "python_ausfuehren", "befehl_ausfuehren", *TOOL_NAMES])
        self.assertTrue(all(not self.reg.get(n).dangerous for n in TOOL_NAMES))
        self.assertTrue(self.reg.get("datei_schreiben").dangerous)
        self.assertTrue(self.reg.get("python_ausfuehren").dangerous)
        self.assertTrue(self.reg.get("befehl_ausfuehren").dangerous)
        for n in ("rechnen", "zeit", "system_info", "verzeichnis", "datei_lesen"):
            self.assertFalse(self.reg.get(n).dangerous)

    def test_register_unregister_get(self):
        t = Tool("echo", "gibt zurück", _schema(text={"type": "string", "required": True}), lambda text: text)
        self.reg.register(t)
        self.assertIs(self.reg.get("echo"), t)
        self.assertEqual(self.reg.run("echo", {"text": "x"}), ToolResult(True, "x", None))
        self.assertTrue(self.reg.unregister("echo"))
        self.assertFalse(self.reg.unregister("echo"))
        self.assertIsNone(self.reg.get("echo"))
        with self.assertRaises(TypeError):
            self.reg.register("kein Tool")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            self.reg.register(Tool("", "x", {}, lambda: ""))

    def test_unknown_tool(self):
        res = self.reg.run("zaubern", {})
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("Unbekanntes Werkzeug »zaubern«. Verfügbar: "))
        self.assertIn("rechnen", res.error)
        self.assertEqual(res.output, "")

    def test_describe_compact(self):
        d = self.reg.describe()
        self.assertIn("Verfügbare Werkzeuge", d)
        self.assertIn("- rechnen(ausdruck):", d)
        self.assertIn("- datei_lesen(pfad, max_zeichen=4000):", d)
        self.assertIn('<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>', d)
        self.assertIn("gefährlich", d)
        # eine Zeile je Werkzeug
        tool_lines = [ln for ln in d.splitlines() if ln.startswith("- ")]
        self.assertEqual(len(tool_lines), len(self.reg.list()))

    def test_describe_full_and_one(self):
        full = self.reg.describe(compact=False)
        one = self.reg.describe_one("datei_lesen")
        self.assertIn(one, full)
        self.assertIn("pfad (string, Pflicht)", one)
        self.assertIn("max_zeichen (integer, optional, Standard 4000)", one)
        self.assertIn("Gefährlich: nein", one)
        self.assertIn('"name": "datei_lesen"', one)
        self.assertIn("Gefährlich: ja", self.reg.describe_one("python_ausfuehren"))
        self.assertIn("Parameter: keine", self.reg.describe_one("zeit"))
        self.assertIn("Unbekanntes Werkzeug", self.reg.describe_one("nix"))
        self.assertEqual(ToolRegistry(self.ws).describe(), "Keine Werkzeuge verfügbar.")

    def test_set_policy(self):
        self.reg.set_policy(None, False)
        self.assertIsNone(self.reg.confirm)
        self.assertFalse(self.reg.confirm_dangerous)
        cb = lambda n, a: True  # noqa: E731
        self.reg.set_policy(cb, True)
        self.assertIs(self.reg.confirm, cb)
        self.assertTrue(self.reg.confirm_dangerous)


# ------------------------------------------------------------------ run: Koerzion, Pflicht, Freigabe, Kürzung
class RunTest(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.seen = []

        def fn(zahl, anteil=0.5, an=False, name="x", liste=None):
            self.seen.append((zahl, anteil, an, name, liste))
            return f"{zahl!r} {anteil!r} {an!r} {name!r} {liste!r}"

        self.reg.register(Tool("probe", "Testwerkzeug", _schema(
            zahl={"type": "integer", "required": True}, anteil={"type": "number"},
            an={"type": "boolean"}, name={"type": "string"}, liste={"type": "array"}), fn))

    def test_coercion_from_strings(self):
        res = self.reg.run("probe", {"zahl": "42", "anteil": "0,25", "an": "ja", "name": 7, "liste": "a"})
        self.assertTrue(res.ok, res.error)
        self.assertEqual(self.seen[-1], (42, 0.25, True, "7", ["a"]))
        res = self.reg.run("probe", {"zahl": 3.0, "anteil": 1, "an": "nein", "liste": [1, 2]})
        self.assertTrue(res.ok)
        self.assertEqual(self.seen[-1], (3, 1.0, False, "x", [1, 2]))
        self.assertTrue(self.reg.run("probe", {"zahl": "12.0", "an": "0"}).ok)
        self.assertEqual(self.seen[-1][0], 12)
        self.assertFalse(self.seen[-1][2])

    def test_coercion_failure_is_error_result(self):
        res = self.reg.run("probe", {"zahl": "abc"})
        self.assertFalse(res.ok)
        self.assertIn("zahl", res.error)
        res = self.reg.run("probe", {"zahl": 1, "an": "vielleicht"})
        self.assertFalse(res.ok)
        self.assertIn("an", res.error)
        res = self.reg.run("probe", {"zahl": 1.5})
        self.assertFalse(res.ok)

    def test_unknown_keys_dropped(self):
        res = self.reg.run("probe", {"zahl": 1, "unbekannt": "weg", "noch_eins": 2})
        self.assertTrue(res.ok, res.error)
        self.assertEqual(self.seen[-1][0], 1)

    def test_required_missing(self):
        res = self.reg.run("probe", {"anteil": 0.1})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "Fehlende Parameter: zahl")
        res = self.reg.run("probe", {"zahl": None})
        self.assertEqual(res.error, "Fehlende Parameter: zahl")
        res = self.reg.run("datei_schreiben", {})
        self.assertEqual(res.error, "Fehlende Parameter: pfad, inhalt")
        res = self.reg.run("probe", "kein dict")  # type: ignore[arg-type]
        self.assertEqual(res.error, "Fehlende Parameter: zahl")

    def test_coerce_value_direct(self):
        self.assertEqual(coerce_value("k", "7", "integer"), 7)
        self.assertEqual(coerce_value("k", True, "integer"), 1)
        self.assertEqual(coerce_value("k", "1e2", "number"), 100.0)
        self.assertEqual(coerce_value("k", 0, "boolean"), False)
        self.assertEqual(coerce_value("k", {"a": 1}, "string"), '{"a": 1}')
        self.assertEqual(coerce_value("k", '{"a": 1}', "object"), {"a": 1})
        self.assertEqual(coerce_value("k", "[1, 2]", "array"), [1, 2])
        self.assertEqual(coerce_value("k", "x", None), "x")
        self.assertIsNone(coerce_value("k", None, "integer"))
        with self.assertRaises(ValueError):
            coerce_value("k", "nein", "object")
        with self.assertRaises(ValueError):
            coerce_value("k", [1], "number")
        # Typ-Liste ["integer", "null"]
        self.reg.register(Tool("opt", "x", {"type": "object", "properties": {"n": {"type": ["integer", "null"]}},
                                            "required": []}, lambda n=None: str(n)))
        self.assertEqual(self.reg.run("opt", {"n": "5"}).output, "5")
        self.assertEqual(self.reg.run("opt", {"n": None}).output, "None")

    def test_exception_becomes_error(self):
        self.reg.register(Tool("kaputt", "wirft", _schema(), lambda: 1 / 0))
        res = self.reg.run("kaputt", {})
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "division by zero")
        self.reg.register(Tool("leer", "wirft leer", _schema(), lambda: (_ for _ in ()).throw(RuntimeError())))
        self.assertEqual(self.reg.run("leer", {}).error, "RuntimeError")

    def test_non_string_output_is_stringified(self):
        self.reg.register(Tool("zahl", "liefert int", _schema(), lambda: 42))
        self.assertEqual(self.reg.run("zahl", {}).output, "42")
        self.reg.register(Tool("nichts", "liefert None", _schema(), lambda: None))
        self.assertEqual(self.reg.run("nichts", {}), ToolResult(True, "", None))

    def test_truncation_suffix(self):
        self.reg.register(Tool("lang", "viel Text", _schema(), lambda: "x" * 5000))
        res = self.reg.run("lang", {})
        self.assertTrue(res.ok)
        self.assertTrue(res.output.startswith("x" * MAX_OUTPUT_CHARS))
        self.assertTrue(res.output.endswith("\n… [gekürzt, 5000 Zeichen insgesamt]"))
        self.assertEqual(len(res.output), MAX_OUTPUT_CHARS + len("\n… [gekürzt, 5000 Zeichen insgesamt]"))
        self.reg.register(Tool("genau", "exakt", _schema(), lambda: "y" * MAX_OUTPUT_CHARS))
        self.assertEqual(len(self.reg.run("genau", {}).output), MAX_OUTPUT_CHARS)
        self.assertEqual(truncate_output("abc"), "abc")
        self.assertEqual(truncate_output("abcdef", 3), "abc\n… [gekürzt, 6 Zeichen insgesamt]")
        self.assertEqual(truncate_output(None), "")

    def test_dangerous_confirm_deny_policy(self):
        asked = []

        def deny(name, args):
            asked.append((name, dict(args)))
            return False

        def allow(name, args):
            asked.append((name, dict(args)))
            return True

        # kein Callback + Bestätigung nötig -> abgelehnt
        res = self.reg.run("datei_schreiben", {"pfad": "neu.txt", "inhalt": "x"})
        self.assertEqual((res.ok, res.error), (False, "Vom Nutzer abgelehnt"))
        self.assertFalse(os.path.exists(os.path.join(self.ws, "neu.txt")))
        # Callback lehnt ab
        self.reg.set_policy(deny, True)
        res = self.reg.run("datei_schreiben", {"pfad": "neu.txt", "inhalt": "x"})
        self.assertEqual(res.error, "Vom Nutzer abgelehnt")
        self.assertEqual(asked[-1], ("datei_schreiben", {"pfad": "neu.txt", "inhalt": "x"}))
        # Callback erlaubt
        self.reg.set_policy(allow, True)
        res = self.reg.run("datei_schreiben", {"pfad": "neu.txt", "inhalt": "x"})
        self.assertTrue(res.ok, res.error)
        # Policy: keine Bestätigung nötig -> Callback wird nicht gefragt
        asked.clear()
        self.reg.set_policy(deny, False)
        res = self.reg.run("datei_schreiben", {"pfad": "neu2.txt", "inhalt": "y"})
        self.assertTrue(res.ok, res.error)
        self.assertEqual(asked, [])
        # ungefährliche Werkzeuge fragen nie
        self.reg.set_policy(deny, True)
        self.assertTrue(self.reg.run("rechnen", {"ausdruck": "1+1"}).ok)
        self.assertEqual(asked, [])
        # Freigabe erst nach Koerzion/Pflichtprüfung
        res = self.reg.run("datei_schreiben", {"pfad": "x.txt"})
        self.assertEqual(res.error, "Fehlende Parameter: inhalt")
        self.assertEqual(asked, [])

    def test_confirm_exception_is_error(self):
        def boom(name, args):
            raise RuntimeError("Dialog abgebrochen")
        self.reg.set_policy(boom, True)
        res = self.reg.run("datei_schreiben", {"pfad": "x.txt", "inhalt": "y"})
        self.assertEqual((res.ok, res.error), (False, "Dialog abgebrochen"))

    def test_default_registry_policy_args(self):
        reg = default_registry(self.ws, confirm=lambda n, a: True, confirm_dangerous=False)
        self.assertFalse(reg.confirm_dangerous)
        self.assertTrue(reg.run("datei_schreiben", {"pfad": "p.txt", "inhalt": "q"}).ok)


# ------------------------------------------------------------------ rechnen
class CalculateTest(unittest.TestCase):
    def setUp(self):
        self.reg = default_registry(".")

    def calc(self, expr):
        return self.reg.run("rechnen", {"ausdruck": expr})

    def test_basic(self):
        self.assertEqual(self.calc("2*21").output, "2*21 = 42")
        self.assertEqual(self.calc("7/2").output, "7/2 = 3.5")
        self.assertEqual(self.calc("7//2 + 7%2").output, "7//2 + 7%2 = 4")
        self.assertEqual(self.calc("-3 + +2").output, "-3 + +2 = -1")
        self.assertEqual(self.calc("2**10").output, "2**10 = 1024")
        self.assertEqual(self.calc("2^10").output, "2**10 = 1024")   # ^ wird als Potenz gelesen
        self.assertEqual(self.calc("1.5 * 2").output, "1.5 * 2 = 3")
        self.assertEqual(self.calc("(1+2)*3").output, "(1+2)*3 = 9")

    def test_functions_and_names(self):
        self.assertEqual(self.calc("math.sqrt(16)").output, "math.sqrt(16) = 4")
        self.assertEqual(self.calc("sqrt(16)").output, "sqrt(16) = 4")
        self.assertEqual(self.calc("abs(-3) + round(2.6) + min(1, 2) + max(3, 4)").output,
                         "abs(-3) + round(2.6) + min(1, 2) + max(3, 4) = 11")
        self.assertEqual(self.calc("round(math.pi, 2)").output, "round(math.pi, 2) = 3.14")
        self.assertTrue(self.calc("math.e + math.tau").ok)
        self.assertFalse(self.calc("math.inf").ok)
        self.assertTrue(self.calc("pow(2, 3)").ok)   # math.pow auch ohne Präfix
        self.assertIn("3.14", self.calc("round(pi, 2)").output)
        self.assertTrue(self.calc("e + tau").ok)
        self.assertTrue(self.calc("math.sin(0) + math.log(1) + math.floor(2.7)").ok)
        self.assertEqual(self.calc("math.cos(0)").output, "math.cos(0) = 1")

    def test_forbidden(self):
        for expr in ("__import__('os').system('ls')", "open('x')", "x", "math.__dict__", "math.pi.real",
                     "'a' + 'b'", "[1, 2]", "1 if 1 else 2", "lambda: 1", "a = 1", "1 & 2", "1 << 3",
                     "math.sqrt(x=4)", "math.sqrt(*[4])", "print(1)", "exec('1')", "True + 1",
                     "math._something(1)", "zufall(1)"):
            res = self.calc(expr)
            self.assertFalse(res.ok, expr)
            self.assertTrue(res.error.startswith("Nicht erlaubt") or "Ungültiger Ausdruck" in res.error, (expr, res.error))

    def test_power_limits(self):
        self.assertTrue(self.calc("2**1000").ok)
        res = self.calc("2**1001")
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith("Nicht erlaubt"))
        self.assertFalse(self.calc("2**-1001").ok)
        self.assertFalse(self.calc("(10**12)**2").ok)
        self.assertTrue(self.calc("(10**11)**2").ok)
        self.assertFalse(self.calc("(-8)**0.5").ok)   # komplex

    def test_errors(self):
        self.assertEqual(self.calc("1/0").error, "Division durch null")
        self.assertEqual(self.calc("").error, "Leerer Ausdruck.")
        self.assertEqual(self.calc("   ").error, "Leerer Ausdruck.")
        self.assertEqual(self.reg.run("rechnen", {}).error, "Fehlende Parameter: ausdruck")
        self.assertIn("Ungültiger Ausdruck", self.calc("2 +* 3").error)
        self.assertIn("zu lang", self.calc("1+" * 300 + "1").error)
        self.assertIn("Rechenfehler", self.calc("math.sqrt(-1)").error)
        self.assertEqual(self.calc("10.0**400").error, "Nicht erlaubt: Ergebnis zu groß")

    def test_calculate_direct(self):
        self.assertEqual(calculate("3*3"), "3*3 = 9")
        with self.assertRaises(ValueError):
            calculate("import os")


# ------------------------------------------------------------------ zeit / system_info
class InfoToolsTest(unittest.TestCase):
    def test_zeit(self):
        res = default_registry(".").run("zeit", {"egal": 1})
        self.assertTrue(res.ok)
        self.assertIn("Datum:", res.output)
        self.assertIn("Uhrzeit:", res.output)
        self.assertIn("Kalenderwoche:", res.output)
        self.assertIn(time.strftime("%Y"), res.output)
        self.assertIn("ISO: " + time.strftime("%Y-%m-%d"), res.output)
        weekday = ("Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag")[time.localtime().tm_wday]
        self.assertIn(weekday, res.output)

    def test_system_info(self):
        res = default_registry(".").run("system_info", {})
        self.assertTrue(res.ok)
        for key in ("Betriebssystem:", "Python:", "Prozessoren:", "Arbeitsspeicher:", "Festplatte", "Arbeitsbereich:"):
            self.assertIn(key, res.output)
        self.assertIn(sys.version.split()[0], res.output)
        self.assertIn(os.path.realpath("."), res.output)


# ------------------------------------------------------------------ Dateien
class FileToolsTest(WorkspaceCase):
    def test_verzeichnis(self):
        os.makedirs(os.path.join(self.ws, "ordner"))
        res = self.reg.run("verzeichnis", {})
        self.assertTrue(res.ok, res.error)
        lines = res.output.splitlines()
        self.assertTrue(lines[0].startswith("Verzeichnis . (2 Einträge)"))
        self.assertIn("ordner/", lines)
        self.assertTrue(any(ln.startswith("notiz.txt (") for ln in lines))
        res = self.reg.run("verzeichnis", {"pfad": "ordner"})
        self.assertIn("(leer)", res.output)
        self.assertIn("Verzeichnis ordner/", res.output)
        self.assertFalse(self.reg.run("verzeichnis", {"pfad": "notiz.txt"}).ok)
        self.assertIn("nicht gefunden", self.reg.run("verzeichnis", {"pfad": "nix"}).error)
        self.assertIn("außerhalb", self.reg.run("verzeichnis", {"pfad": ".."}).error)

    def test_verzeichnis_limit(self):
        big = os.path.join(self.ws, "viele")
        os.makedirs(big)
        for i in range(250):
            open(os.path.join(big, f"f{i:03d}.txt"), "w").close()
        res = self.reg.run("verzeichnis", {"pfad": "viele"})
        self.assertTrue(res.ok)
        lines = res.output.splitlines()
        self.assertIn("(250 Einträge)", lines[0])
        self.assertEqual(len([ln for ln in lines if ln.startswith("f")]), 200)
        self.assertIn("50 weitere Einträge", lines[-1])

    def test_datei_lesen(self):
        res = self.reg.run("datei_lesen", {"pfad": "notiz.txt"})
        self.assertEqual(res, ToolResult(True, "Hallo Welt\nZeile 2", None))
        res = self.reg.run("datei_lesen", {"pfad": "notiz.txt", "max_zeichen": "5"})
        self.assertEqual(res.output, "Hallo\n… [gekürzt, 18 Zeichen insgesamt]")
        # Klemmen 1..20000
        res = self.reg.run("datei_lesen", {"pfad": "notiz.txt", "max_zeichen": 0})
        self.assertTrue(res.output.startswith("H\n… [gekürzt"))
        with open(os.path.join(self.ws, "gross.txt"), "w", encoding="utf-8") as fh:
            fh.write("a" * 30000)
        raw = self.reg.get("datei_lesen").fn("gross.txt", max_zeichen=99999)   # Klemme bei 20000
        self.assertTrue(raw.startswith("a" * 20000))
        self.assertEqual(raw, "a" * 20000 + "\n… [gekürzt, 30000 Zeichen insgesamt]")
        # run kürzt zusätzlich auf 4000
        res = self.reg.run("datei_lesen", {"pfad": "gross.txt", "max_zeichen": 99999})
        self.assertTrue(res.output.startswith("a" * 4000))
        self.assertIn(f"[gekürzt, {len(raw)} Zeichen insgesamt]", res.output)
        self.assertLessEqual(len(res.output), 4000 + 60)
        self.assertIn("nicht gefunden", self.reg.run("datei_lesen", {"pfad": "fehlt.txt"}).error)
        self.assertIn("Verzeichnis", self.reg.run("datei_lesen", {"pfad": "."}).error)
        self.assertIn("außerhalb", self.reg.run("datei_lesen", {"pfad": "../draussen/geheim.txt"}).error)

    def test_datei_lesen_binary_and_encoding(self):
        with open(os.path.join(self.ws, "bin.dat"), "wb") as fh:
            fh.write(b"\x00\x01\x02abc")
        res = self.reg.run("datei_lesen", {"pfad": "bin.dat"})
        self.assertTrue(res.ok)
        self.assertTrue(res.output.startswith("Binärdatei: bin.dat"))
        with open(os.path.join(self.ws, "latin.txt"), "wb") as fh:
            fh.write("Größe".encode("latin-1"))
        res = self.reg.run("datei_lesen", {"pfad": "latin.txt"})
        self.assertTrue(res.ok)
        self.assertIn("�", res.output)     # errors="replace"

    def test_datei_lesen_too_large(self):
        path = os.path.join(self.ws, "riesig.bin")
        with open(path, "wb") as fh:
            fh.truncate(5 * 1024 * 1024 + 1)
        res = self.reg.run("datei_lesen", {"pfad": "riesig.bin"})
        self.assertFalse(res.ok)
        self.assertIn("zu groß", res.error)

    def test_datei_schreiben(self):
        self.reg.set_policy(lambda n, a: True, True)
        res = self.reg.run("datei_schreiben", {"pfad": "neu/tief/ausgabe.txt", "inhalt": "Grüße"})
        self.assertEqual(res.output, "geschrieben: neu/tief/ausgabe.txt (7 Bytes, überschrieben: nein)")
        with open(os.path.join(self.ws, "neu", "tief", "ausgabe.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "Grüße")
        res = self.reg.run("datei_schreiben", {"pfad": "neu/tief/ausgabe.txt", "inhalt": "x"})
        self.assertEqual(res.output, "geschrieben: neu/tief/ausgabe.txt (1 Bytes, überschrieben: ja)")
        res = self.reg.run("datei_schreiben", {"pfad": "../draussen/neu.txt", "inhalt": "x"})
        self.assertIn("außerhalb", res.error)
        self.assertFalse(os.path.exists(os.path.join(self.outside, "neu.txt")))
        res = self.reg.run("datei_schreiben", {"pfad": "neu", "inhalt": "x"})
        self.assertIn("Verzeichnis", res.error)
        res = self.reg.run("datei_schreiben", {"pfad": "gross.txt", "inhalt": "x" * (1024 * 1024 + 1)})
        self.assertIn("zu groß", res.error)
        self.assertFalse(os.path.exists(os.path.join(self.ws, "gross.txt")))
        res = self.reg.run("datei_schreiben", {"pfad": "zahl.txt", "inhalt": 123})
        self.assertTrue(res.ok)
        with open(os.path.join(self.ws, "zahl.txt"), encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "123")

    def test_datei_schreiben_symlink_parent_outside(self):
        link = os.path.join(self.ws, "link")
        try:
            os.symlink(self.outside, link)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks werden hier nicht unterstützt")
        self.reg.set_policy(lambda n, a: True, True)
        res = self.reg.run("datei_schreiben", {"pfad": "link/boese.txt", "inhalt": "x"})
        self.assertFalse(res.ok)
        self.assertFalse(os.path.exists(os.path.join(self.outside, "boese.txt")))


# ------------------------------------------------------------------ Prozesse
class ProcessToolsTest(WorkspaceCase):
    def setUp(self):
        super().setUp()
        self.reg.set_policy(lambda n, a: True, True)

    def test_python_ausfuehren(self):
        res = self.reg.run("python_ausfuehren", {"code": "import os, sys\nprint(os.getcwd())\nprint(2**10)"})
        self.assertTrue(res.ok, res.error)
        self.assertIn(self.ws, res.output)
        self.assertIn("1024", res.output)
        self.assertTrue(res.output.endswith("\n[exit 0]"))
        self.assertNotIn("[stderr]", res.output)
        res = self.reg.run("python_ausfuehren", {"code": "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"})
        self.assertEqual(res.output, "out\n[stderr]\nerr\n[exit 3]")
        res = self.reg.run("python_ausfuehren", {"code": "raise ValueError('kaputt')"})
        self.assertTrue(res.ok)             # Prozessfehler sind Ausgabe, kein Werkzeugfehler
        self.assertIn("[stderr]", res.output)
        self.assertIn("ValueError: kaputt", res.output)
        self.assertTrue(res.output.endswith("[exit 1]"))

    def test_python_isolated_env_and_stdin(self):
        res = self.reg.run("python_ausfuehren", {
            "code": "import os, sys\nprint(sorted(k for k in os.environ if k in ('HOME','OBITO_TEST_X')))\n"
                    "print(repr(sys.stdin.read()))\nprint(sys.flags.isolated)"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("[]", res.output)
        self.assertIn("''", res.output)
        self.assertIn("1", res.output.splitlines()[2])

    def test_python_timeout(self):
        t0 = time.time()
        res = self.reg.run("python_ausfuehren", {"code": "import time; time.sleep(10)", "timeout": 1})
        self.assertLess(time.time() - t0, 8)
        self.assertFalse(res.ok)
        self.assertEqual(res.error, "Zeitlimit (1 s) überschritten")
        # Koerzion + Klemmen des Timeouts
        res = self.reg.run("python_ausfuehren", {"code": "import time; time.sleep(10)", "timeout": "0"})
        self.assertEqual(res.error, "Zeitlimit (1 s) überschritten")

    def test_python_rejected_and_empty(self):
        self.reg.set_policy(None, True)
        res = self.reg.run("python_ausfuehren", {"code": "print(1)"})
        self.assertEqual(res.error, "Vom Nutzer abgelehnt")
        self.reg.set_policy(None, False)
        self.assertIn("Kein Code", self.reg.run("python_ausfuehren", {"code": "  "}).error)

    def test_befehl_ausfuehren(self):
        res = self.reg.run("befehl_ausfuehren", {"befehl": "echo hallo"})
        self.assertTrue(res.ok, res.error)
        self.assertTrue(res.output.startswith("hallo"))
        self.assertTrue(res.output.endswith("[exit 0]"))
        res = self.reg.run("befehl_ausfuehren", {"befehl": f'"{sys.executable}" -c "import os; print(os.getcwd())"'})
        self.assertIn(self.ws, res.output)
        res = self.reg.run("befehl_ausfuehren", {"befehl": f'"{sys.executable}" -c "import time; time.sleep(10)"',
                                                 "timeout": 1})
        self.assertEqual(res.error, "Zeitlimit (1 s) überschritten")
        self.assertIn("Kein Befehl", self.reg.run("befehl_ausfuehren", {"befehl": ""}).error)


# ------------------------------------------------------------------ Gedächtnis
class MemoryToolsTest(unittest.TestCase):
    def setUp(self):
        self.reg = default_registry(".")
        self.mem = MemoryStore(":memory:", embedder=FakeBackend(embed_dim=64).embed)

    def tearDown(self):
        self.mem.close()

    def test_only_with_memory(self):
        self.assertIsNone(self.reg.get("gedaechtnis_suchen"))
        res = self.reg.run("gedaechtnis_suchen", {"frage": "x"})
        self.assertIn("Unbekanntes Werkzeug", res.error)
        self.reg.set_memory(self.mem)
        self.assertIsNotNone(self.reg.get("gedaechtnis_suchen"))
        self.assertIsNotNone(self.reg.get("gedaechtnis_merken"))
        self.assertFalse(self.reg.get("gedaechtnis_merken").dangerous)
        self.assertIn("gedaechtnis_suchen(frage)", self.reg.describe())
        self.reg.set_memory(None)
        self.assertIsNone(self.reg.get("gedaechtnis_suchen"))
        self.assertIsNone(self.reg.get("gedaechtnis_merken"))

    def test_merken_und_suchen(self):
        self.reg.set_memory(self.mem)
        res = self.reg.run("gedaechtnis_merken", {"inhalt": "Nutzer baut eine Carbon-Drohne", "art": "fakt",
                                                   "wichtigkeit": "0.95"})
        self.assertTrue(res.ok, res.error)
        self.assertTrue(res.output.startswith("gemerkt: #"))
        self.assertIn("[fakt]", res.output)
        m = self.mem.recent(1)[0]
        self.assertEqual((m.source, m.kind, m.importance), ("ki", "fakt", 0.8))   # ≤ 0.8
        res = self.reg.run("gedaechtnis_merken", {"inhalt": "Unbekannte Art", "art": "quatsch"})
        self.assertIn("[notiz]", res.output)
        self.assertEqual(self.mem.recent(1)[0].importance, 0.6)
        res = self.reg.run("gedaechtnis_merken", {"inhalt": "Negativ", "wichtigkeit": -3})
        self.assertEqual(self.mem.recent(1)[0].importance, 0.0)
        self.assertIn("Leere", self.reg.run("gedaechtnis_merken", {"inhalt": "  "}).error)

        res = self.reg.run("gedaechtnis_suchen", {"frage": "Drohne Carbon"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("Carbon-Drohne", res.output)
        self.assertIn("[fakt", res.output)
        res = self.reg.run("gedaechtnis_suchen", {"frage": "Quantenchromodynamik"})
        self.assertEqual(res.output, "Keine passenden Erinnerungen gefunden.")
        self.assertEqual(self.reg.run("gedaechtnis_suchen", {}).error, "Fehlende Parameter: frage")


# ------------------------------------------------------------------ Protokoll
class ParseToolCallsTest(unittest.TestCase):
    def test_closed_tag(self):
        text = 'Ich rechne das aus.\n<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>'
        self.assertEqual(parse_tool_calls(text), [{"name": "rechnen", "args": {"ausdruck": "2*21"}}])

    def test_multiple_and_case(self):
        text = ('<WERKZEUG>{"name": "zeit", "args": {}}</WERKZEUG> und '
                '<werkzeug >\n{"name": "rechnen", "args": {"ausdruck": "1"}}\n</werkzeug >')
        self.assertEqual(parse_tool_calls(text), [{"name": "zeit", "args": {}},
                                                  {"name": "rechnen", "args": {"ausdruck": "1"}}])

    def test_unclosed_at_end(self):
        text = 'Moment.\n<werkzeug>{"name": "zeit", "args": {}}'
        self.assertEqual(parse_tool_calls(text), [{"name": "zeit", "args": {}}])
        text = 'Moment.\n<werkzeug>{"name": "zeit", "args": {}}\n'
        self.assertEqual(parse_tool_calls(text), [{"name": "zeit", "args": {}}])
        # geschlossener plus offener Block
        text = ('<werkzeug>{"name": "zeit", "args": {}}</werkzeug>\n'
                '<werkzeug>{"name": "rechnen", "args": {"ausdruck": "3"}}')
        self.assertEqual([c["name"] for c in parse_tool_calls(text)], ["zeit", "rechnen"])
        # abgeschnittenes JSON -> nichts
        self.assertEqual(parse_tool_calls('<werkzeug>{"name": "zeit", "ar'), [])

    def test_json_fence(self):
        text = 'Ich nutze:\n```json\n{"name": "rechnen", "args": {"ausdruck": "5*5"}}\n```\nFertig.'
        self.assertEqual(parse_tool_calls(text), [{"name": "rechnen", "args": {"ausdruck": "5*5"}}])
        text = '```\n{"name": "zeit", "args": {}}\n```'
        self.assertEqual(parse_tool_calls(text), [{"name": "zeit", "args": {}}])
        # Zaun mit anderem Objekt ist kein Werkzeugaufruf
        self.assertEqual(parse_tool_calls('```json\n{"name": "Max", "alter": 3}\n```'), [])
        self.assertEqual(parse_tool_calls('```json\n{"name": "rechnen"}\n```'), [])
        self.assertEqual(parse_tool_calls('```python\nprint(1)\n```'), [])

    def test_naked_object(self):
        text = 'Antwort:\n{"name": "zeit", "args": {}}\nDanke.'
        self.assertEqual(parse_tool_calls(text), [{"name": "zeit", "args": {}}])
        text = '  {"name": "rechnen", "args": {"ausdruck": "1+1"}}'
        self.assertEqual(parse_tool_calls(text), [{"name": "rechnen", "args": {"ausdruck": "1+1"}}])
        # mehrzeilig
        text = '{\n  "name": "rechnen",\n  "args": {"ausdruck": "2+2"}\n}'
        self.assertEqual(parse_tool_calls(text), [{"name": "rechnen", "args": {"ausdruck": "2+2"}}])
        # genau diese Schlüssel
        self.assertEqual(parse_tool_calls('{"name": "zeit", "args": {}, "extra": 1}'), [])
        self.assertEqual(parse_tool_calls('{"name": "zeit"}'), [])
        # nicht am Zeilenanfang
        self.assertEqual(parse_tool_calls('Objekt {"name": "zeit", "args": {}} im Satz'), [])

    def test_invalid(self):
        self.assertEqual(parse_tool_calls(""), [])
        self.assertEqual(parse_tool_calls(None), [])  # type: ignore[arg-type]
        self.assertEqual(parse_tool_calls("kein Werkzeug"), [])
        self.assertEqual(parse_tool_calls("<werkzeug>kein json</werkzeug>"), [])
        self.assertEqual(parse_tool_calls('<werkzeug>{"name": 5, "args": {}}</werkzeug>'), [])
        self.assertEqual(parse_tool_calls('<werkzeug>{"args": {}}</werkzeug>'), [])
        self.assertEqual(parse_tool_calls('<werkzeug>["rechnen"]</werkzeug>'), [])
        self.assertEqual(parse_tool_calls('<werkzeug>{"name": "  ", "args": {}}</werkzeug>'), [])

    def test_args_not_dict_becomes_empty(self):
        self.assertEqual(parse_tool_calls('<werkzeug>{"name": "zeit", "args": "x"}</werkzeug>'),
                         [{"name": "zeit", "args": {}}])
        self.assertEqual(parse_tool_calls('<werkzeug>{"name": "zeit"}</werkzeug>'), [{"name": "zeit", "args": {}}])
        self.assertEqual(parse_tool_calls('<werkzeug>{"name": "zeit", "args": [1]}</werkzeug>'),
                         [{"name": "zeit", "args": {}}])

    def test_tag_with_fence_inside_and_prose(self):
        text = '<werkzeug>\n```json\n{"name": "zeit", "args": {}}\n```\n</werkzeug>'
        self.assertEqual(parse_tool_calls(text), [{"name": "zeit", "args": {}}])

    def test_no_double_counting(self):
        text = '<werkzeug>\n{"name": "zeit", "args": {}}\n</werkzeug>'
        self.assertEqual(len(parse_tool_calls(text)), 1)
        text = '```json\n{"name": "zeit", "args": {}}\n```'
        self.assertEqual(len(parse_tool_calls(text)), 1)


class StripToolCallsTest(unittest.TestCase):
    def test_strip_all_forms(self):
        text = ('Ich rechne: <werkzeug>{"name": "rechnen", "args": {"ausdruck": "1+1"}}</werkzeug>\n'
                'Zeit:\n```json\n{"name": "zeit", "args": {}}\n```\n'
                '{"name": "system_info", "args": {}}\n'
                'Ende <werkzeug>{"name": "verzeichnis", "args": {"pfad": "."}}')
        out = strip_tool_calls(text)
        self.assertNotIn("werkzeug", out)
        self.assertNotIn("```", out)
        self.assertNotIn('"name"', out)
        self.assertIn("Ich rechne:", out)
        self.assertIn("Zeit:", out)
        self.assertIn("Ende", out)

    def test_strip_keeps_normal_text_and_other_fences(self):
        text = 'Hier Code:\n```python\nprint(1)\n```\nund JSON ```json\n{"a": 1}\n```'
        self.assertEqual(strip_tool_calls(text), text)
        self.assertEqual(strip_tool_calls("nur Text"), "nur Text")
        self.assertEqual(strip_tool_calls(""), "")
        self.assertEqual(strip_tool_calls(None), "")  # type: ignore[arg-type]

    def test_strip_invalid_tag_content_still_removed(self):
        out = strip_tool_calls("Hallo <werkzeug>kaputt</werkzeug> Welt")
        self.assertEqual(out, "Hallo  Welt")

    def test_strip_collapses_blank_lines(self):
        text = 'A\n\n<werkzeug>{"name": "zeit", "args": {}}</werkzeug>\n\n\nB'
        self.assertEqual(strip_tool_calls(text), "A\n\nB")

    def test_roundtrip_with_parse(self):
        text = 'Antwort 42.\n<werkzeug>{"name": "zeit", "args": {}}</werkzeug>'
        self.assertEqual(parse_tool_calls(strip_tool_calls(text)), [])
        self.assertEqual(strip_tool_calls(text).strip(), "Antwort 42.")


class NeedsToolsTest(unittest.TestCase):
    def test_positive(self):
        for q in ("Was ist 12 * 7?", "Rechne 3+4", "Berechne die Fläche", "Wie viel ist 2 hoch 10?",
                  "Wieviel kostet das?", "Lies die Datei notizen.txt", "Zeig mir den Ordner",
                  "Welche Dateien sind im Verzeichnis?", "Wie spät ist es, welche Uhrzeit?",
                  "Welches Datum haben wir heute?", "Welcher Tag ist heute?", "Zeig mir Systeminfos",
                  "Führe das Skript aus", "Starte den Server", "Speicher das ab", "Öffne die config",
                  "Wie viel Arbeitsspeicher habe ich?", "Merk dir, dass ich Carbon mag",
                  "2+2", "100/4", "Was sind 15% von 200?", "Erinnerst du dich an mein Projekt?",
                  "Ausführen: python test.py", "Was ist 5 - 3?", "ausfuehren bitte"):
            self.assertTrue(needs_tools(q, []), q)

    def test_negative(self):
        for q in ("Hallo, wie geht es dir?", "Erkläre mir die Relativitätstheorie",
                  "Schreib ein Gedicht über den Herbst", "Was bedeutet das Wort Melancholie?",
                  "Warum ist der Himmel blau", "Wer war Lieselotte?", "Ich habe 3 Katzen und 2 Hunde",
                  "Was ist heute los?", "Welche Merkmale hat ein F7-Flight-Controller?",
                  "Eine Wandstärke von 3-5 mm reicht", "Welche Zeit braucht der Druck?",
                  "Die Speicherkarte ist voll"):
            self.assertFalse(needs_tools(q, []), q)
        self.assertFalse(needs_tools("", []))
        self.assertFalse(needs_tools(None, None))  # type: ignore[arg-type]

    def test_history_last_assistant_turn(self):
        hist = [{"role": "user", "content": "Rechne 2+2"},
                {"role": "assistant", "content": '<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2+2"}}</werkzeug>'},
                {"role": "user", "content": "Ergebnis von Werkzeug »rechnen«:\n4"},
                {"role": "assistant", "content": "Das Ergebnis ist 4."}]
        # letzter Assistant-Turn ohne Werkzeug -> nein
        self.assertFalse(needs_tools("Und danke dir", hist))
        # letzter Assistant-Turn mit Werkzeug -> ja
        hist2 = hist[:2]
        self.assertTrue(needs_tools("Und danke dir", hist2))
        hist3 = [{"role": "assistant", "content": '```json\n{"name": "zeit", "args": {}}\n```'}]
        self.assertTrue(needs_tools("ok", hist3))
        hist4 = [{"role": "assistant", "content": "normal"}, {"role": "user", "content": "<werkzeug> im Nutzertext"}]
        self.assertFalse(needs_tools("ok", hist4))
        self.assertFalse(needs_tools("ok", [{"role": "assistant"}, "kaputt"]))  # type: ignore[list-item]


# ------------------------------------------------------------------ Streaming-Gate
class ToolStreamFilterTest(unittest.TestCase):
    @staticmethod
    def feed12(f, text):
        for i in range(0, len(text), 12):
            f.feed(text[i:i + 12])

    def test_gate_holds_tool_block(self):
        out = []
        f = ToolStreamFilter(out.append)
        text = 'Ich rechne: <werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>'
        self.feed12(f, text)
        f.finish()
        self.assertEqual("".join(out), "Ich rechne: ")
        self.assertTrue(f.tool_detected)
        self.assertEqual(f.text, text)
        self.assertEqual(f.emitted, "Ich rechne: ")

    def test_marker_split_across_pieces(self):
        out = []
        f = ToolStreamFilter(out.append)
        text = 'Das Ergebnis kommt gleich <werkzeug>{"name": "zeit", "args": {}}</werkzeug> Ende'
        self.feed12(f, text)
        f.finish()
        self.assertEqual("".join(out), "Das Ergebnis kommt gleich ")
        self.assertNotIn("Ende", "".join(out))
        self.assertTrue(f.tool_detected)
        for n in range(1, 10):   # alle Stückgrößen
            out = []
            f = ToolStreamFilter(out.append)
            for i in range(0, len(text), n):
                f.feed(text[i:i + n])
            f.finish()
            self.assertEqual("".join(out), "Das Ergebnis kommt gleich ", n)

    def test_plain_text_passes_fully(self):
        out = []
        f = ToolStreamFilter(out.append)
        text = "Dies ist eine ganz normale Antwort ohne Werkzeuge, mit <b>HTML</b> und < Zeichen."
        self.feed12(f, text)
        f.finish()
        self.assertEqual("".join(out), text)
        self.assertFalse(f.tool_detected)
        self.assertEqual(f.text, text)

    def test_prefix_at_end_held_until_finish(self):
        out = []
        f = ToolStreamFilter(out.append)
        text = "Ich rechne: <werk"
        self.feed12(f, text)
        self.assertEqual("".join(out), "Ich rechne: ")   # Präfix zurückgehalten
        f.finish()
        self.assertEqual("".join(out), "Ich rechne: <werk")   # kein Marker -> Rest kommt
        self.assertFalse(f.tool_detected)
        out = []
        f = ToolStreamFilter(out.append)
        f.feed("Fast <")
        self.assertEqual("".join(out), "Fast ")
        f.feed("werkzeu")
        self.assertEqual("".join(out), "Fast ")
        f.feed("g>{")
        f.finish()
        self.assertEqual("".join(out), "Fast ")
        self.assertTrue(f.tool_detected)

    def test_false_prefix_released(self):
        out = []
        f = ToolStreamFilter(out.append)
        f.feed("a <wer")
        f.feed("t von 3 <")
        f.feed("x")
        f.finish()
        self.assertEqual("".join(out), "a <wert von 3 <x")
        self.assertFalse(f.tool_detected)

    def test_case_insensitive_marker_and_custom(self):
        out = []
        f = ToolStreamFilter(out.append)
        self.feed12(f, 'Hallo <WERKZEUG>{"name": "zeit"}')
        f.finish()
        self.assertEqual("".join(out), "Hallo ")
        out = []
        f = ToolStreamFilter(out.append, marker="<tool")
        self.feed12(f, "Text <tool>{}")
        f.finish()
        self.assertEqual("".join(out), "Text ")

    def test_without_emit(self):
        f = ToolStreamFilter(None)
        self.feed12(f, "Ohne Callback <werkzeug>{}")
        f.finish()
        self.assertTrue(f.tool_detected)
        self.assertEqual(f.text, "Ohne Callback <werkzeug>{}")
        self.assertEqual(f.emitted, "Ohne Callback ")
        f.feed("")
        f.feed("mehr")
        self.assertEqual(f.emitted, "Ohne Callback ")

    def test_with_fakebackend_stream(self):
        fb = FakeBackend(responses=['Antwort: <werkzeug>{"name": "zeit", "args": {}}</werkzeug>'])
        out = []
        gate = ToolStreamFilter(out.append)
        res = fb.chat([{"role": "user", "content": "x"}], stream=gate.feed)
        gate.finish()
        self.assertEqual("".join(out), "Antwort: ")
        self.assertEqual(parse_tool_calls(res.text), [{"name": "zeit", "args": {}}])
        self.assertEqual(strip_tool_calls(res.text).strip(), "Antwort:")


if __name__ == "__main__":
    unittest.main()
