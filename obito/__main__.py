"""``python -m obito`` – Einstieg in die Kommandozeile von OBITO."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
