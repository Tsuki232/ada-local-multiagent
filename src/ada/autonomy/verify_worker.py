"""Proceso aislado que ejecuta la verificacion por doble metodo.

Corre como subproceso para poder MATARLO por timeout: SymPy puede quedarse minutos en una integral
y un hilo de Python no se puede cancelar. Entrada: JSON por stdin. Salida: JSON en la ultima linea de stdout.
"""

from __future__ import annotations

import json
import sys


def _run(kind: str, p: dict) -> dict:
    import sympy as sp

    from ada.calculus.verified_math import (
        solve_and_verify_ode,
        verify_definite_integral,
        verify_derivative,
    )

    if kind == "derivative":
        res = verify_derivative(p["expression"], p["variable"], int(p["order"]), float(p["eval_point"]))
        var = sp.Symbol(p["variable"])
        res["expression_latex"] = sp.latex(sp.sympify(p["expression"], locals={p["variable"]: var}))
        return res
    if kind == "integral":
        res = verify_definite_integral(p["expression"], p["variable"], float(p["lower"]), float(p["upper"]))
        var = sp.Symbol(p["variable"])
        res["expression_latex"] = sp.latex(sp.sympify(p["expression"], locals={p["variable"]: var}))
        return res
    if kind == "ode":
        return solve_and_verify_ode(
            p["rhs"],
            order=int(p["order"]),
            t0=0.0,
            t_end=float(p["t_end"]),
            y0=float(p["y0"]),
            dy0=float(p.get("dy0", 0.0)),
        )
    raise ValueError(f"tipo desconocido: {kind}")


def main() -> int:
    try:
        job = json.loads(sys.stdin.read())
        result = _run(job["kind"], job["params"])
        out = {"ok": True, "result": result}
    except Exception as exc:  # noqa: BLE001 - se informa al padre, no se silencia
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
    sys.stdout.write("\n" + json.dumps(out, ensure_ascii=False, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
