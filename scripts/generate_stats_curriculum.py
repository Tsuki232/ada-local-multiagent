"""Genera el curriculo de inferencia estadistica verificado en data/work/training/stats_curriculum.jsonl.

  python scripts/generate_stats_curriculum.py            # 900 ejemplos
  python scripts/generate_stats_curriculum.py --n 1500
Cada ejemplo solo se escribe si su resultado fue verificado por doble via. Es determinista (misma semilla -> mismos ejemplos).
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from ada.config import DATA_WORK_DIR, assert_safe_write_path  # noqa: E402
from ada.stats.curriculum import generate_curriculum  # noqa: E402

OUT = DATA_WORK_DIR / "training" / "stats_curriculum.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--seed", type=int, default=20260104)
    args = ap.parse_args()
    t0 = time.time()
    rows = generate_curriculum(args.n, args.seed)
    assert_safe_write_path(OUT)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"{len(rows)} ejemplos verificados en {time.time() - t0:.0f} s -> {OUT}")
    print("por análisis:", dict(collections.Counter(r["meta"]["analysis"] for r in rows)))
    print("por prueba:  ", dict(collections.Counter(r["meta"]["test"] for r in rows).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
