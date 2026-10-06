"""Tests für obito.desktop – App-Start ohne echten Browser (Fake-Launcher) und ohne Modell (FakeBackend)."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from obito import desktop
from obito.config import Config
from obito.llm import FakeBackend


class _FakeProc:
    """Ersatz für subprocess.Popen: ``wait()`` kehrt nach ``delay`` Sekunden zurück."""

    def __init__(self, cmd, delay=0.05):
        self.cmd = list(cmd)
        self.delay = delay
        self.terminated = False
        self.started = time.monotonic()

    def wait(self):
        time.sleep(self.delay)
        return 0

    def poll(self):
        return None if time.monotonic() - self.started < self.delay else 0

    def terminate(self):
        self.terminated = True


class FindBrowserTest(unittest.TestCase):
    def test_windows_prefers_edge_path(self):
        found = desktop.find_browser("win32", which=lambda n: None,
                                     exists=lambda p: p.name.lower() == "msedge.exe")
        self.assertIsNotNone(found)
        self.assertTrue(found.lower().endswith("msedge.exe"))

    def test_falls_back_to_path_lookup(self):
        found = desktop.find_browser("linux", which=lambda n: "/usr/bin/chromium" if n == "chromium" else None,
                                     exists=lambda p: False)
        self.assertEqual(found, "/usr/bin/chromium")

    def test_none_when_nothing_found(self):
        self.assertIsNone(desktop.find_browser("linux", which=lambda n: None, exists=lambda p: False))

    def test_macos_candidates(self):
        found = desktop.find_browser("darwin", which=lambda n: None,
                                     exists=lambda p: "Google Chrome" in str(p))
        self.assertIn("Google Chrome", found)


class AppArgsTest(unittest.TestCase):
    def test_app_mode_flags(self):
        args = desktop.app_args("/x/edge", "http://127.0.0.1:8765", 1600, 950, "/tmp/profil")
        self.assertEqual(args[0], "/x/edge")
        self.assertIn("--app=http://127.0.0.1:8765", args)
        self.assertIn("--window-size=1600,950", args)
        self.assertIn("--user-data-dir=/tmp/profil", args)
        self.assertIn("--no-first-run", args)

    def test_size_is_clamped(self):
        args = desktop.app_args("e", "u", 10, 99999, "p")
        self.assertIn(f"--window-size={desktop.MIN_SIZE},{desktop.MAX_SIZE}", args)


class RunAppTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Config(data_dir=self.tmp.name, backend="fake", workspace=self.tmp.name,
                          server_host="127.0.0.1", server_port=0, parallel_calls=1)
        self.backend = FakeBackend()

    def tearDown(self):
        self.tmp.cleanup()

    def test_starts_server_opens_window_and_stops_after_window_closes(self):
        out = io.StringIO()
        launched: list[list[str]] = []

        def launcher(cmd):
            launched.append(list(cmd))
            # Während das Fenster "offen" ist, muss der Server antworten
            url = [a for a in cmd if a.startswith("--app=")][0][len("--app="):]
            self.assertTrue(desktop.server_alive(url))
            return _FakeProc(cmd, delay=0.05)

        rc = desktop.run_app(self.cfg, backend=self.backend, launcher=launcher, browser="/fake/msedge.exe",
                             out=out, use_webview=False)
        self.assertEqual(rc, 0)
        self.assertEqual(len(launched), 1)
        self.assertIn("--app=http://127.0.0.1:", launched[0][1])
        self.assertTrue(any(a.startswith("--user-data-dir=") and self.tmp.name in a for a in launched[0]))
        text = out.getvalue()
        self.assertIn("OBITO-Server gestartet", text)
        self.assertIn("App-Fenster geöffnet", text)
        self.assertIn("OBITO beendet", text)
        # Nach dem Ende antwortet der Server nicht mehr
        url = launched[0][1][len("--app="):]
        self.assertFalse(desktop.server_alive(url, timeout=0.5))

    def test_reuses_running_server(self):
        handle = desktop.start_server(self.cfg, self.backend)
        try:
            port = handle.server.port
            cfg2 = Config(data_dir=self.tmp.name, backend="fake", workspace=self.tmp.name,
                          server_host="127.0.0.1", server_port=port)
            out = io.StringIO()
            launched = []
            rc = desktop.run_app(cfg2, backend=self.backend, launcher=lambda c: (launched.append(c), _FakeProc(c))[1],
                                 browser="/fake/chrome", out=out, use_webview=False)
            self.assertEqual(rc, 0)
            self.assertIn("läuft bereits", out.getvalue())
            self.assertEqual(len(launched), 1)
            # Der fremde Server läuft weiter
            self.assertTrue(desktop.server_alive(f"http://127.0.0.1:{port}"))
        finally:
            handle.close()

    def test_browser_start_failure_falls_back_to_webbrowser(self):
        out = io.StringIO()
        opened = []
        orig = desktop.webbrowser.open
        desktop.webbrowser.open = lambda url: opened.append(url) or True
        try:
            def launcher(cmd):
                raise OSError("kein Browser")

            # wait=False: kehrt nach dem Öffnen zurück, Server läuft weiter → explizit nicht warten
            rc = desktop.run_app(self.cfg, backend=self.backend, launcher=launcher, browser="/fake/edge",
                                 out=out, use_webview=False, wait=False)
        finally:
            desktop.webbrowser.open = orig
        self.assertEqual(rc, 0)
        self.assertEqual(len(opened), 1)
        self.assertIn("startet nicht", out.getvalue())

    def test_without_window_returns_immediately_when_not_waiting(self):
        out = io.StringIO()
        rc = desktop.run_app(self.cfg, backend=self.backend, open_browser=False, out=out, wait=False)
        self.assertEqual(rc, 0)
        self.assertIn("OBITO-Server gestartet", out.getvalue())

    def test_port_in_use_reports_error(self):
        handle = desktop.start_server(self.cfg, self.backend)
        try:
            port = handle.server.port
            # Zweiter Start auf demselben Port, aber ohne laufenden OBITO (server_alive patchen)
            orig = desktop.server_alive
            desktop.server_alive = lambda url, timeout=1.5: False
            try:
                cfg2 = Config(data_dir=self.tmp.name, backend="fake", workspace=self.tmp.name,
                              server_host="127.0.0.1", server_port=port)
                out = io.StringIO()
                rc = desktop.run_app(cfg2, backend=self.backend, out=out, use_webview=False,
                                     launcher=lambda c: _FakeProc(c), browser="/fake/edge")
            finally:
                desktop.server_alive = orig
        finally:
            handle.close()
        self.assertEqual(rc, 1)
        self.assertIn("kann nicht", out.getvalue())

    def test_keyboard_interrupt_terminates_window(self):
        class _Proc(_FakeProc):
            def wait(self):
                raise KeyboardInterrupt

        procs = []

        def launcher(cmd):
            p = _Proc(cmd)
            procs.append(p)
            return p

        rc = desktop.run_app(self.cfg, backend=self.backend, launcher=launcher, browser="/fake/edge",
                             out=io.StringIO(), use_webview=False)
        self.assertEqual(rc, 0)
        self.assertTrue(procs[0].terminated)


class CliAppTest(unittest.TestCase):
    def test_cli_app_command_uses_desktop(self):
        from obito import cli

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        calls = {}
        orig = desktop.run_app

        def fake_run_app(cfg, **kw):
            calls["cfg"] = cfg
            calls.update(kw)
            return 0

        desktop.run_app = fake_run_app
        try:
            rc = cli.main(["--daten", tmp.name, "--backend", "fake", "app", "--port", "9999", "--breite", "1200",
                           "--ohne-webview", "--browser", "/x/edge"], out=io.StringIO(), backend=FakeBackend())
        finally:
            desktop.run_app = orig
        self.assertEqual(rc, 0)
        self.assertEqual(calls["cfg"].server_port, 9999)
        self.assertEqual(calls["width"], 1200)
        self.assertFalse(calls["use_webview"])
        self.assertEqual(calls["browser"], "/x/edge")
        self.assertTrue(calls["open_browser"])


class PackagedEntryTest(unittest.TestCase):
    def test_entry_delegates_to_cli_with_app_default(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("obito_app", Path(__file__).resolve().parent.parent / "packaging" / "obito_app.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        from obito import cli
        calls = []
        orig_main, orig_argv, orig_env = cli.main, sys.argv, os.environ.get("OBITO_CONFIG")
        cli.main = lambda argv: calls.append(list(argv)) or 0
        try:
            sys.argv = ["OBITO.exe"]
            self.assertEqual(mod.main(), 0)
            sys.argv = ["OBITO.exe", "doctor", "--training"]
            self.assertEqual(mod.main(), 0)
        finally:
            cli.main, sys.argv = orig_main, orig_argv
            if orig_env is None:
                os.environ.pop("OBITO_CONFIG", None)
        self.assertEqual(calls, [["app"], ["doctor", "--training"]])

    def test_spec_file_references_existing_files(self):
        root = Path(__file__).resolve().parent.parent
        spec = (root / "packaging" / "obito.spec").read_text(encoding="utf-8")
        for rel in ("obito/static/index.html", "beispiele/eval_fragen.jsonl", "obito.example.json", "packaging/obito_app.py"):
            self.assertTrue((root / rel).is_file(), rel)
        self.assertIn("obito_app.py", spec)
        self.assertIn('name="OBITO"', spec)
        bat = (root / "build_exe.bat").read_text(encoding="utf-8")
        self.assertIn("packaging\\obito.spec", bat)


if __name__ == "__main__":
    unittest.main()
