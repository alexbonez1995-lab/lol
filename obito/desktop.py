"""OBITO als native App: lokaler Server und App-Fenster in einem Prozess.

Ablauf von :func:`run_app`:

1. Läuft unter der konfigurierten Adresse bereits ein OBITO-Server, wird nur das Fenster geöffnet.
2. Sonst startet der Server im Hintergrund (eigener Thread, Zeitplaner inklusive).
3. Das Fenster ist – in dieser Reihenfolge – ein `pywebview`-Fenster (falls installiert), ein
   Chromium-„App-Fenster“ (Edge/Chrome/Chromium mit ``--app=URL``, eigenem Profil und ohne
   Browser-Leisten) oder der Standardbrowser.
4. Schließt der Nutzer das Fenster, wird der Server sauber beendet (bei Chromium-App-Fenstern
   über das Prozessende; beim Standardbrowser mit Strg+C).

Alles bleibt lokal (``127.0.0.1``); es werden keine Daten nach außen geschickt.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any, Callable, Sequence, TextIO

from .config import Config

log = logging.getLogger("obito.desktop")

APP_TITLE = "OBITO v4.0 – AI Engineering Nexus"
DEFAULT_WIDTH = 1600
DEFAULT_HEIGHT = 950
MIN_SIZE = 640
MAX_SIZE = 7680
SERVER_START_TIMEOUT_S = 15.0

# Chromium-Flags für ein ruhiges App-Fenster (keine Hinweise beim ersten Start, keine Übersetzungs-Leiste)
_CHROMIUM_FLAGS = ("--no-first-run", "--no-default-browser-check", "--disable-features=TranslateUI",
                   "--disable-session-crashed-bubble", "--disable-infobars")


# ------------------------------------------------------------------ Browser finden
def _windows_candidates() -> list[Path]:
    env = os.environ
    roots = [env.get("ProgramFiles(x86)") or r"C:\Program Files (x86)",
             env.get("ProgramFiles") or r"C:\Program Files",
             env.get("LocalAppData") or r"C:\Users\Default\AppData\Local"]
    rel = (Path("Microsoft") / "Edge" / "Application" / "msedge.exe",
           Path("Google") / "Chrome" / "Application" / "chrome.exe",
           Path("Chromium") / "Application" / "chrome.exe")
    out: list[Path] = []
    for r in roots:
        if not r:
            continue
        for p in rel:
            out.append(Path(r) / p)
    return out


def _macos_candidates() -> list[Path]:
    apps = Path("/Applications")
    return [apps / "Microsoft Edge.app" / "Contents" / "MacOS" / "Microsoft Edge",
            apps / "Google Chrome.app" / "Contents" / "MacOS" / "Google Chrome",
            apps / "Chromium.app" / "Contents" / "MacOS" / "Chromium"]


_PATH_NAMES = ("msedge", "microsoft-edge", "microsoft-edge-stable", "google-chrome", "google-chrome-stable",
               "chrome", "chromium", "chromium-browser", "brave-browser")


def find_browser(platform: str | None = None, which: Callable[[str], str | None] = shutil.which,
                 exists: Callable[[Path], bool] = Path.is_file) -> str | None:
    """Pfad zu Edge/Chrome/Chromium oder ``None``. ``platform``/``which``/``exists`` dienen Tests."""
    plat = platform or sys.platform
    candidates: list[Path] = []
    if plat.startswith("win"):
        candidates = _windows_candidates()
    elif plat == "darwin":
        candidates = _macos_candidates()
    for c in candidates:
        try:
            if exists(c):
                return str(c)
        except OSError:
            continue
    for name in _PATH_NAMES:
        found = which(name)
        if found:
            return found
    return None


def app_args(exe: str, url: str, width: int, height: int, profile_dir: Path | str) -> list[str]:
    """Kommandozeile für ein Chromium-App-Fenster mit eigenem Profil (bleibt an den Prozess gebunden)."""
    width = max(MIN_SIZE, min(MAX_SIZE, int(width)))
    height = max(MIN_SIZE, min(MAX_SIZE, int(height)))
    return [exe, f"--app={url}", f"--window-size={width},{height}", f"--user-data-dir={profile_dir}",
            *_CHROMIUM_FLAGS]


# ------------------------------------------------------------------ Server
def server_alive(url: str, timeout: float = 1.5) -> bool:
    """``True``, wenn unter ``url`` ein OBITO-Server antwortet (nur 127.0.0.1/localhost, Proxy umgangen)."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url.rstrip("/") + "/api/status", timeout=timeout) as resp:
            return resp.status == 200
    except Exception:  # noqa: BLE001 – nicht erreichbar ist ein normaler Fall
        return False


def _wait_for(pred: Callable[[], bool], timeout: float, step: float = 0.1) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


class _ServerHandle:
    """Hält Server und Brain und beendet beides genau einmal."""

    def __init__(self, server: Any, brain: Any, thread: threading.Thread):
        self.server = server
        self.brain = brain
        self.thread = thread
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.server.shutdown()
        except Exception as e:  # noqa: BLE001
            log.warning("Server-Shutdown: %s", e)
        try:
            self.server.server_close()
        except Exception as e:  # noqa: BLE001
            log.warning("Server-Close: %s", e)
        self.thread.join(timeout=5)
        try:
            self.brain.close()
        except Exception as e:  # noqa: BLE001
            log.warning("Brain-Close: %s", e)


def start_server(cfg: Config, backend: Any = None, *, allow_dangerous: bool = False,
                 brain_factory: Callable[..., Any] | None = None,
                 server_factory: Callable[..., Any] | None = None) -> _ServerHandle:
    """Startet Brain + Server in einem Hintergrund-Thread. ``*_factory`` dienen Tests."""
    if brain_factory is None:
        from .brain import Brain
        brain_factory = Brain
    if server_factory is None:
        from .server import ObitoServer
        server_factory = ObitoServer
    brain = brain_factory(cfg, backend=backend)
    try:
        server = server_factory(brain, cfg.server_host, cfg.server_port, allow_dangerous=allow_dangerous)
    except Exception:
        brain.close()
        raise
    thread = threading.Thread(target=server.serve_forever, name="obito-server", daemon=True)
    thread.start()
    start_bg = getattr(server, "start_background", None)
    if callable(start_bg):
        try:
            start_bg()
        except Exception as e:  # noqa: BLE001
            log.warning("Zeitplaner nicht gestartet: %s", e)
    return _ServerHandle(server, brain, thread)


# ------------------------------------------------------------------ Fenster
def _try_webview(url: str, width: int, height: int) -> bool:
    """Öffnet ein pywebview-Fenster (blockiert bis zum Schließen). ``False`` wenn nicht installiert."""
    try:
        import webview  # type: ignore  # optional
    except Exception:  # noqa: BLE001
        return False
    try:
        webview.create_window(APP_TITLE, url, width=width, height=height, min_size=(900, 600))
        webview.start()
        return True
    except Exception as e:  # noqa: BLE001
        log.warning("pywebview-Fenster fehlgeschlagen, nutze Browser: %s", e)
        return False


def run_app(cfg: Config, *, backend: Any = None, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
            allow_dangerous: bool = False, use_webview: bool = True, open_browser: bool = True,
            out: TextIO | None = None, launcher: Callable[[Sequence[str]], Any] | None = None,
            browser: str | None = None, server_factory: Callable[..., Any] | None = None,
            brain_factory: Callable[..., Any] | None = None, wait: bool = True) -> int:
    """Startet OBITO als App. Rückgabe: Exit-Code (0 = normal beendet).

    ``launcher`` (Tests) ersetzt ``subprocess.Popen`` und muss ein Objekt mit ``wait()``/``poll()``/
    ``terminate()`` liefern; ``browser`` erzwingt einen Browser-Pfad; ``wait=False`` kehrt nach dem
    Öffnen sofort zurück und lässt den Server laufen (der Aufrufer beendet über den Rückgabewert von
    :func:`start_server` – hier nur für eingebettete Nutzung).
    """
    out = out or sys.stdout
    url = f"http://{cfg.server_host}:{cfg.server_port}"
    if ":" in cfg.server_host and not cfg.server_host.startswith("["):
        url = f"http://[{cfg.server_host}]:{cfg.server_port}"
    handle: _ServerHandle | None = None

    if server_alive(url):
        print(f"OBITO-Server läuft bereits unter {url} – öffne nur das Fenster.", file=out)
    else:
        try:
            handle = start_server(cfg, backend, allow_dangerous=allow_dangerous,
                                  brain_factory=brain_factory, server_factory=server_factory)
        except OSError as e:
            print(f"Fehler: Server kann nicht auf {cfg.server_host}:{cfg.server_port} lauschen: {e}", file=out)
            return 1
        actual_port = getattr(handle.server, "port", cfg.server_port)
        url = url.rsplit(":", 1)[0] + f":{actual_port}"
        if not _wait_for(lambda: server_alive(url), SERVER_START_TIMEOUT_S):
            print("Fehler: Der OBITO-Server antwortet nicht.", file=out)
            handle.close()
            return 1
        print(f"OBITO-Server gestartet: {url}", file=out)

    try:
        if not open_browser:
            if not wait or handle is None:
                return 0
            print("Kein Fenster gewünscht – Server läuft (Strg+C beendet).", file=out)
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                return 0

        if use_webview and launcher is None and _try_webview(url, width, height):
            return 0

        exe = browser or find_browser()
        if exe:
            profile = cfg.window_dir
            try:
                profile.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                print(f"Hinweis: Fenster-Profil nicht anlegbar ({e}); Browser nutzt Standardprofil.", file=out)
            cmd = app_args(exe, url, width, height, profile)
            launch = launcher or (lambda c: subprocess.Popen(list(c)))
            try:
                proc = launch(cmd)
            except OSError as e:
                print(f"Hinweis: Browser {exe} startet nicht ({e}) – öffne Standardbrowser.", file=out)
                proc = None
            if proc is not None:
                print(f"App-Fenster geöffnet ({Path(exe).name}). Fenster schließen beendet OBITO.", file=out)
                if not wait:
                    return 0
                try:
                    proc.wait()
                except KeyboardInterrupt:
                    try:
                        proc.terminate()
                    except Exception:  # noqa: BLE001
                        pass
                return 0

        # Fallback: Standardbrowser – der Prozess bleibt bis Strg+C
        try:
            webbrowser.open(url)
        except Exception as e:  # noqa: BLE001
            print(f"Browser konnte nicht geöffnet werden: {e}", file=out)
        print(f"OBITO läuft unter {url} (Strg+C beendet).", file=out)
        if not wait or handle is None:
            return 0
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            return 0
    finally:
        if handle is not None and wait:
            handle.close()
            print("OBITO beendet.", file=out)
