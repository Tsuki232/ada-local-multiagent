"""Calculo simbolico + numerico doblemente verificado (SymPy + SciPy/mpmath) y lector OCR/VLM de ecuaciones."""

from __future__ import annotations

import base64
import re
import unicodedata
from pathlib import Path
from typing import Any

import httpx
import mpmath as mp
import numpy as np
import sympy as sp
from scipy import integrate

from ada.audit.ledger import record_audit_event
from ada.config import ModelRoleConfig


def _safe_sympify(expr_str: str, local_symbols: dict[str, Any]) -> sp.Expr:
    cleaned = unicodedata.normalize("NFKC", expr_str).strip().replace("^", "**")
    return sp.sympify(cleaned, locals=local_symbols)


def verify_derivative(
    expression: str,
    variable: str = "x",
    order: int = 1,
    eval_point: float = 0.75,
) -> dict[str, Any]:
    """Calcula la derivada simbolica con SymPy y la contrasta numericamente con diferencias centrales en mpmath (50 digitos)."""
    if order < 1 or order > 6:
        raise ValueError("El orden de derivacion debe estar entre 1 y 6.")
    var = sp.Symbol(variable, real=True)
    expr = _safe_sympify(expression, {variable: var, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})
    sym_deriv = sp.simplify(sp.diff(expr, var, order))

    f_num = sp.lambdify(var, expr, modules="mpmath")
    df_sym_num = sp.lambdify(var, sym_deriv, modules="mpmath")

    with mp.workdps(50):
        x0 = mp.mpf(str(eval_point))
        val_sym = float(mp.re(df_sym_num(x0)))
        val_num = float(mp.re(mp.diff(f_num, x0, n=order)))

    abs_err = abs(val_sym - val_num)
    rel_err = abs_err / max(1.0, abs(val_sym))
    verified = bool(np.isfinite(rel_err) and rel_err < 1e-8)

    result = {
        "verified": verified,
        "verification_methods": ["sympy.diff", "mpmath.diff_50dps"],
        "expression": str(expr),
        "variable": variable,
        "order": order,
        "symbolic_derivative": str(sym_deriv),
        "latex_derivative": sp.latex(sym_deriv),
        "eval_point": eval_point,
        "symbolic_value_at_point": val_sym,
        "numeric_value_at_point": val_num,
        "relative_discrepancy": rel_err,
    }
    record_audit_event("verify_derivative", result)
    return result


def verify_definite_integral(
    expression: str,
    variable: str,
    lower_limit: float,
    upper_limit: float,
) -> dict[str, Any]:
    """Calcula una integral definida simbolicamente (SymPy) y la contrasta con cuadratura adaptativa (SciPy quad)."""
    var = sp.Symbol(variable, real=True)
    expr = _safe_sympify(expression, {variable: var, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt})

    f_np = sp.lambdify(var, expr, modules="numpy")
    quad_val, quad_err = integrate.quad(lambda u: float(f_np(u)), float(lower_limit), float(upper_limit), limit=200)

    sym_integral = sp.integrate(expr, (var, sp.nsimplify(str(lower_limit)), sp.nsimplify(str(upper_limit))))
    sym_val = float(sym_integral.evalf(30))

    abs_diff = abs(sym_val - quad_val)
    rel_diff = abs_diff / max(1.0, abs(quad_val))
    verified = bool(np.isfinite(rel_diff) and rel_diff < 1e-7)

    result = {
        "verified": verified,
        "verification_methods": ["sympy.integrate", "scipy.integrate.quad"],
        "expression": str(expr),
        "limits": [lower_limit, upper_limit],
        "symbolic_exact": str(sym_integral),
        "symbolic_latex": sp.latex(sym_integral),
        "symbolic_float": sym_val,
        "scipy_quad_float": float(quad_val),
        "scipy_quad_est_error": float(quad_err),
        "relative_discrepancy": rel_diff,
    }
    record_audit_event("verify_definite_integral", result)
    return result


def solve_and_verify_ode(
    ode_rhs_order1_or_2: str,
    *,
    order: int = 2,
    t0: float = 0.0,
    t_end: float = 5.0,
    y0: float = 0.2,
    dy0: float = 0.0,
    parameters: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Resuelve una EDO de orden 1 (dy/dt = f(t, y)) o orden 2 (d2y/dt2 = f(t, y, dy)) por dos metodos independientes.

    Contrasta:
      1) Solucion analitica exacta con SymPy (si la EDO admite forma cerrada) o integrador Radau implícito,
      2) Integrador explicito Runge-Kutta de orden 8 (DOP853, rtol=1e-11, atol=1e-13) en SciPy.
    """
    if order not in {1, 2}:
        raise ValueError("order debe ser 1 o 2.")

    params = parameters or {}
    t_sym = sp.Symbol("t", real=True)
    y_sym = sp.Symbol("y", real=True)
    dy_sym = sp.Symbol("dy", real=True)
    y_func = sp.Function("y")(t_sym)

    loc: dict[str, Any] = {
        "t": t_sym,
        "y": y_sym,
        "dy": dy_sym,
        "theta": y_sym,
        "dtheta": dy_sym,
        "omega": dy_sym,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "exp": sp.exp,
        "sqrt": sp.sqrt,
    }
    for k, v in params.items():
        loc[k] = sp.Float(v)

    rhs_expr = _safe_sympify(ode_rhs_order1_or_2, loc)
    t_eval = np.linspace(float(t0), float(t_end), 200)

    if order == 1:
        f_num = sp.lambdify((t_sym, y_sym), rhs_expr, modules="numpy")
        rhs_vec = lambda t, s: [float(f_num(t, s[0]))]
        init_state = [float(y0)]
        ode_eq = sp.Eq(sp.diff(y_func, t_sym), rhs_expr.subs({y_sym: y_func}))
        ics = {y_func.subs(t_sym, t0): y0}
    else:
        f_num = sp.lambdify((t_sym, y_sym, dy_sym), rhs_expr, modules="numpy")
        rhs_vec = lambda t, s: [float(s[1]), float(f_num(t, s[0], s[1]))]
        init_state = [float(y0), float(dy0)]
        ode_eq = sp.Eq(sp.diff(y_func, t_sym, 2), rhs_expr.subs({dy_sym: sp.diff(y_func, t_sym), y_sym: y_func}))
        ics = {y_func.subs(t_sym, t0): y0, sp.diff(y_func, t_sym).subs(t_sym, t0): dy0}

    sol_dop = integrate.solve_ivp(
        rhs_vec,
        (float(t0), float(t_end)),
        init_state,
        method="DOP853",
        t_eval=t_eval,
        rtol=1e-11,
        atol=1e-13,
    )
    if not sol_dop.success:
        raise RuntimeError(f"DOP853 fallo al integrar la EDO: {sol_dop.message}")

    symbolic_solution_str = ""
    symbolic_latex = ""
    method_b_name = ""
    y_b: np.ndarray

    try:
        # Si la EDO es no lineal trascendente (ej. pendulo con sin(y)), SymPy dsolve no tiene forma cerrada
        # elemental y gastaria decenas de segundos buscando simetrias de Lie. Pasamos directamente a Radau implicito.
        if rhs_expr.has(sp.sin) or rhs_expr.has(sp.cos) or rhs_expr.has(sp.tan):
            raise ValueError("EDO no lineal trascendente; contrastar con Radau IIA implicito independiente.")
        sym_sol = sp.dsolve(ode_eq, y_func, ics=ics)
        if isinstance(sym_sol, list):
            sym_sol = sym_sol[0]
        rhs_closed = sym_sol.rhs
        if rhs_closed.has(sp.Integral):
            raise ValueError("Solucion contiene integrales no elementales; usar Radau independiente.")
        fn_closed = sp.lambdify(t_sym, rhs_closed, modules="numpy")
        y_b = np.asarray(fn_closed(t_eval), dtype=float)
        if y_b.ndim == 0:
            y_b = np.full_like(t_eval, float(y_b))
        symbolic_solution_str = str(rhs_closed)
        symbolic_latex = sp.latex(sym_sol)
        method_b_name = "sympy.dsolve_closed_form"
    except Exception:
        # Si la EDO es no lineal sin forma cerrada elemental (ej. pendulo amortiguado/forzado),
        # contrastamos DOP853 explicito contra Radau IIA implicito de orden 5 (metodo numerico independiente).
        sol_radau = integrate.solve_ivp(
            rhs_vec,
            (float(t0), float(t_end)),
            init_state,
            method="Radau",
            t_eval=t_eval,
            rtol=1e-11,
            atol=1e-13,
        )
        if not sol_radau.success:
            raise RuntimeError(f"Radau fallo al contrastar la EDO: {sol_radau.message}")
        y_b = sol_radau.y[0]
        method_b_name = "scipy.solve_ivp_Radau_implicit"

    max_abs_diff = float(np.max(np.abs(sol_dop.y[0] - y_b)))
    verified = bool(np.isfinite(max_abs_diff) and max_abs_diff < 1e-6)

    result = {
        "verified": verified,
        "verification_methods": ["scipy.solve_ivp_DOP853", method_b_name],
        "max_abs_discrepancy": max_abs_diff,
        "order": order,
        "ode_rhs": str(rhs_expr),
        "symbolic_closed_form": symbolic_solution_str or None,
        "symbolic_latex": symbolic_latex or None,
        "t_span": [t0, t_end],
        "initial_conditions": {"y0": y0, "dy0": dy0 if order == 2 else None},
        "final_state_dop853": {
            "y_end": float(sol_dop.y[0, -1]),
            "dy_end": float(sol_dop.y[1, -1]) if order == 2 else None,
        },
        "trajectory_preview": {
            "t": [round(float(v), 4) for v in t_eval[::20]],
            "y": [round(float(v), 6) for v in sol_dop.y[0, ::20]],
        },
    }
    record_audit_event("solve_and_verify_ode", {k: v for k, v in result.items() if k != "trajectory_preview"})
    return result


def _balance_parentheses(expr: str) -> str:
    """Elimina parentesis de cierre huerfanos producidos por cajas OCR solapadas y cierra los abiertos."""
    out: list[str] = []
    depth = 0
    for ch in expr:
        if ch == "(":
            depth += 1
            out.append(ch)
        elif ch == ")":
            if depth > 0:
                depth -= 1
                out.append(ch)
        else:
            out.append(ch)
    if depth > 0:
        out.extend(")" * depth)
    return "".join(out)


def _normalize_ocr_math_to_rhs(raw_text: str) -> dict[str, Any]:
    """Normaliza expresiones OCR comunes de EDOs de orden 1 o 2 hacia forma d2y/dt2 = rhs o dy/dt = rhs."""
    txt = unicodedata.normalize("NFKC", raw_text).strip()
    compact = (
        txt.replace("θ", "y")
        .replace("Θ", "y")
        .replace("ω", "dy")
        .replace('y"', "y''")
        .replace("−", "-")
        .replace("–", "-")
        .replace("×", "*")
        .replace("·", "*")
        .replace("sen", "sin")
    )
    # Corregir confusiones tipicas de OCR en numeros decimales (ej. '9.8l' -> '9.81', 'O.4' -> '0.4')
    compact = re.sub(r"(?<=\d)[lI](?=[\W_]|$)", "1", compact)
    compact = re.sub(r"\bO(?=\.\d)", "0", compact)
    compact_one_line = " ".join(compact.splitlines()).strip()

    cand_order = 2 if any(tok in compact_one_line.lower() for tok in ["''", "d2", "d^2", "ddot"]) else 1
    normalized_rhs = ""

    no_space = compact_one_line.replace(" ", "")
    no_space_std = re.sub(r"dy/dt", "dy", no_space, flags=re.IGNORECASE)
    m_second = re.search(
        r"(?:y''|d2y/dt2|d\^2y/dt\^2)([+\-].*?)=([^;]+)",
        no_space_std,
        flags=re.IGNORECASE,
    )
    if m_second:
        left_rest = _balance_parentheses(m_second.group(1))
        right_side = _balance_parentheses(m_second.group(2))
        normalized_rhs = f"({right_side}) - ({left_rest})".replace("y'", "dy")
        cand_order = 2
    else:
        m_first = re.search(
            r"(?:y'|dy)([+\-].*?)=([^;]+)",
            no_space_std,
            flags=re.IGNORECASE,
        )
        if m_first:
            left_rest = _balance_parentheses(m_first.group(1))
            right_side = _balance_parentheses(m_first.group(2))
            normalized_rhs = f"({right_side}) - ({left_rest})"
            cand_order = 1
        else:
            m_explicit = re.search(
                r"(?:y''|d2y/dt2|y'|dy)\s*=\s*(.+)",
                re.sub(r"dy/dt", "dy", compact_one_line, flags=re.IGNORECASE),
                flags=re.IGNORECASE,
            )
            if m_explicit:
                normalized_rhs = _balance_parentheses(m_explicit.group(1).strip().replace("y'", "dy"))

    return {
        "detected_order": cand_order,
        "normalized_text": compact_one_line,
        "candidate_ode_rhs": normalized_rhs,
    }


def _sort_ocr_boxes_reading_order(ocr_result: list[Any]) -> list[dict[str, Any]]:
    """Agrupa cajas OCR por renglon horizontal, normaliza NFKC y fusiona bordes duplicados por solapamiento."""
    items: list[dict[str, Any]] = []
    for box, text, score in ocr_result:
        xs = [float(pt[0]) for pt in box]
        ys = [float(pt[1]) for pt in box]
        norm_text = unicodedata.normalize("NFKC", str(text)).strip()
        if not norm_text:
            continue
        items.append(
            {
                "x_min": min(xs),
                "x_max": max(xs),
                "cy": 0.5 * (min(ys) + max(ys)),
                "h": max(1.0, max(ys) - min(ys)),
                "text": norm_text,
                "confidence": round(float(score), 4),
            }
        )
    items.sort(key=lambda d: d["cy"])
    rows: list[list[dict[str, Any]]] = []
    for it in items:
        placed = False
        for row in rows:
            avg_cy = sum(r["cy"] for r in row) / len(row)
            avg_h = sum(r["h"] for r in row) / len(row)
            if abs(it["cy"] - avg_cy) <= 0.65 * max(avg_h, it["h"]):
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    ordered_lines: list[dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda d: d["x_min"])
        merged_tokens: list[str] = []
        prev_xmax = -1e9
        for r in row:
            tok = r["text"]
            if merged_tokens and r["x_min"] <= prev_xmax + 6.0 and merged_tokens[-1][-1:] == tok[:1]:
                tok = tok[1:].lstrip()
            if tok:
                merged_tokens.append(tok)
            prev_xmax = max(prev_xmax, r["x_max"])
        ordered_lines.append(
            {
                "text": " ".join(merged_tokens),
                "confidence": round(sum(r["confidence"] for r in row) / len(row), 4),
            }
        )
    return ordered_lines


def extract_equation_from_image(image_path: str | Path) -> dict[str, Any]:
    """Extrae una ecuacion diferencial escrita graficamente en una imagen (OCR ONNX local + VLM Ollama si existe)."""
    resolved = Path(image_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"No existe la imagen: {resolved}")

    from rapidocr_onnxruntime import RapidOCR

    ocr_engine = RapidOCR()
    ocr_result, _ = ocr_engine(str(resolved))
    lines: list[dict[str, Any]] = _sort_ocr_boxes_reading_order(ocr_result) if ocr_result else []

    joined_ocr = "\n".join(item["text"] for item in lines)

    # Verificar si ya hay un modelo multimodal (VLM) cargado/instalado en Ollama local
    cfg = ModelRoleConfig()
    vlm_used: str | None = None
    vlm_extraction: str | None = None
    try:
        tags_resp = httpx.get(f"{cfg.ollama_api_host}/api/tags", timeout=3.0)
        if tags_resp.status_code == 200:
            installed_models = [m.get("name", "") for m in tags_resp.json().get("models", [])]
            vision_candidates = [
                m for m in installed_models if any(v in m.lower() for v in ("gemma4", "vl", "llava", "minicpm-v", "qwen3.5"))
            ]
            if vision_candidates:
                vlm_used = vision_candidates[0]
                img_b64 = base64.b64encode(resolved.read_bytes()).decode("ascii")
                prompt_vlm = (
                    "Transcribe EXACTAMENTE la ecuacion diferencial de la imagen en una sola linea ASCII limpia "
                    "usando y para la variable dependiente (o theta->y), y' para dy/dt, y'' para d2y/dt2, t para el tiempo. "
                    "Responde SOLO la ecuacion, sin explicaciones."
                )
                gen_resp = httpx.post(
                    f"{cfg.ollama_api_host}/api/generate",
                    json={"model": vlm_used, "prompt": prompt_vlm, "images": [img_b64], "stream": False},
                    timeout=45.0,
                )
                if gen_resp.status_code == 200:
                    vlm_extraction = gen_resp.json().get("response", "").strip()
    except Exception:
        vlm_used = None

    primary_text = vlm_extraction if vlm_extraction else joined_ocr
    parsed_hint = _normalize_ocr_math_to_rhs(primary_text)

    out = {
        "image_path": str(resolved),
        "ocr_engine": "rapidocr_onnxruntime_local",
        "ocr_lines": lines,
        "ocr_raw_text": joined_ocr,
        "vlm_model_used": vlm_used,
        "vlm_transcription": vlm_extraction,
        "parsed_ode_hint": parsed_hint,
        "multimodal_status_note": (
            f"VLM local activo ({vlm_used})."
            if vlm_used
            else "OCR ONNX local activo (<1 GB, ideal para ecuaciones impresas/claras). "
            "Para manuscritos matematicos complejos o diagramas fisicos a mano alzada, confirma la descarga de gemma4:12b (~8.0 GB, Apache-2.0)."
        ),
    }
    record_audit_event("extract_equation_from_image", out, input_path=resolved)
    return out
