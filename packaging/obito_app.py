"""Einstiegspunkt der gepackten OBITO-App (PyInstaller).

Verhält sich wie ``python -m obito app``; weitere Argumente werden durchgereicht
(``OBITO.exe doctor``, ``OBITO.exe serve`` …). Das Datenverzeichnis bleibt ``%USERPROFILE%\\.obito``
bzw. ``OBITO_HOME``; eine ``obito.json`` neben der EXE wird als Konfiguration genutzt.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    base = os.path.dirname(os.path.abspath(sys.executable if getattr(sys, "frozen", False) else __file__))
    cfg = os.path.join(base, "obito.json")
    if os.path.isfile(cfg) and not os.environ.get("OBITO_CONFIG"):
        os.environ["OBITO_CONFIG"] = cfg
    if getattr(sys, "frozen", False):
        # Das HUD liegt im Bündel unter obito/static – der Server findet es über das Paket
        os.chdir(base)
    from obito.cli import main as cli_main

    argv = sys.argv[1:]
    if not argv:
        argv = ["app"]
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
