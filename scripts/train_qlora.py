"""Lanzador del QLoRA de ADA. Por defecto hace un ENSAYO; usa --execute para entrenar."""

import sys
from pathlib import Path

if sys.platform == "win32":  # la consola de Windows usa cp1252 y se caia al imprimir "≥", "≈", tildes...
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ada.training.qlora import run  # noqa: E402

if __name__ == "__main__":
    sys.exit(run())
