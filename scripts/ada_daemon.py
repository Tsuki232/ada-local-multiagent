"""Daemon del agente autonomo de ADA.

  python scripts/ada_daemon.py              # corre hasta que lo detengas
  python scripts/ada_daemon.py --once       # un solo ciclo y sale
  python scripts/ada_daemon.py --status     # estado del agente y del dataset
  python scripts/ada_daemon.py --stop       # freno de emergencia (crea data/work/autonomy/STOP)
  python scripts/ada_daemon.py --resume     # borra STOP
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ada.autonomy.agent import AutonomousAgent  # noqa: E402
from ada.autonomy.curator import CURATION_REPORT_FILE  # noqa: E402
from ada.autonomy.heart import Heart  # noqa: E402
from ada.autonomy.guard import PID_FILE, STOP_FILE, AutonomyGuard, clear_stop, pid_alive, request_stop  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--interval", type=float, default=25.0, help="segundos entre ciclos (defecto 25)")
    ap.add_argument("--max-per-hour", type=int, default=120)
    ap.add_argument("--allow-download", action="store_true",
                    help="AUTORIZA una vez que Ada descargue los pesos base Qwen3-14B (≈30 GB, el mismo modelo que sirve Ollama) para poder entrenarse sola")
    ap.add_argument("--auto-train", action="store_true", help="ACTIVA el autoentrenamiento (apagado por defecto: el QLoRA 14B no cabe en 12 GB; ademas requiere quitar TRAINING_DISABLED)")
    args = ap.parse_args()

    if args.allow_download:
        from ada.autonomy.auto_trainer import allow_download

        allow_download()
        print("Descarga de pesos base autorizada. Ada la hará sola cuando tenga datos suficientes para entrenar.")
        return 0

    if args.stop:
        print(f"STOP creado: {request_stop('cli')}")
        return 0
    if args.resume:
        clear_stop()
        print("STOP eliminado.")
        return 0
    if args.status:
        agent = AutonomousAgent()
        running_pid = None
        if PID_FILE.is_file():
            try:
                pid = int(PID_FILE.read_text(encoding="utf-8").strip())
                running_pid = pid if pid_alive(pid) else None
            except ValueError:
                pass
        out = {"daemon_pid": running_pid, "stop_requested": STOP_FILE.is_file(), **agent.status()}
        if CURATION_REPORT_FILE.is_file():
            out["dataset"] = json.loads(CURATION_REPORT_FILE.read_text(encoding="utf-8"))
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0

    def show(e: dict) -> None:
        st = e.get("status", "")
        line = e.get("detail") or (f"{e.get('kind')} nivel {e.get('level')}: {st}" if e.get("kind") else st)
        if st == "verified":
            line = f"verificado · {e['kind']} nivel {e['level']} · {(e.get('question') or '')[:80]}"
        print(f"[{time.strftime('%H:%M:%S')}] {st}: {line}", flush=True)
        if e.get("thought"):
            print(f"           Ada: {e['thought']}", flush=True)
        if e.get("reflection"):
            print(f"           Ada piensa sobre sí misma: {e['reflection']}", flush=True)
        if e.get("level_change"):
            print(f"           mejoró: {e['kind']} nivel {e['level_change'][0]} -> {e['level_change'][1]}", flush=True)

    agent = AutonomousAgent(
        interval_s=args.interval,
        guard=AutonomyGuard(max_cycles_per_hour=args.max_per_hour),
        narrate=True,
        announce=True,
        auto_train=args.auto_train,
        heart=Heart(),
        on_event=show,
    )
    if args.once:
        print(json.dumps(agent.run_cycle(), ensure_ascii=False, indent=2, default=str))
        return 0

    signal.signal(signal.SIGINT, lambda *_: agent.stop())
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, lambda *_: agent.stop())
    print(f"ADA autonoma activa (ciclo cada {args.interval:.0f}s, max {args.max_per_hour}/h). Ctrl+C o --stop para detener.")
    try:
        agent.run_forever()
    except RuntimeError as exc:
        print(f"[no iniciado] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
