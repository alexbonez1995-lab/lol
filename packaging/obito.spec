# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller-Spezifikation für OBITO v4.0 (Windows: dist/OBITO/OBITO.exe).

Bauen (im Projektverzeichnis):
    pip install pyinstaller
    pyinstaller packaging/obito.spec --noconfirm
oder einfach build_exe.bat.

Es entsteht ein Ordner ``dist/OBITO`` mit ``OBITO.exe`` (startet ``python -m obito app``: Server im
Hintergrund + HUD als App-Fenster). Ollama wird nicht eingebettet – es bleibt ein eigenes Programm.
Die Trainings-Abhängigkeiten (torch, peft …) gehören nicht in die EXE; Training läuft weiter über
``python -m obito train``.
"""

import os
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))
sys.path.insert(0, ROOT)

block_cipher = None

datas = [
    (os.path.join(ROOT, "obito", "static", "index.html"), os.path.join("obito", "static")),
    (os.path.join(ROOT, "beispiele", "eval_fragen.jsonl"), "beispiele"),
    (os.path.join(ROOT, "obito.example.json"), "."),
    (os.path.join(ROOT, "README.md"), "."),
]

hiddenimports = collect_submodules("obito") + ["sqlite3", "ctypes", "winreg", "termios", "select"]
excludes = ["torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes", "trl", "numpy",
            "tkinter", "matplotlib", "PIL", "scipy", "pandas"]

a = Analysis(
    [os.path.join(ROOT, "packaging", "obito_app.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="OBITO",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,            # Konsole bleibt sichtbar: Logs, Strg+C, Fehlermeldungen
    icon=None,
)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="OBITO")
