"""Probabilidad de eventos, regla de la suma, condicional, independencia y Bayes — VERIFICADO por dos vias independientes.

Eventos sobre un dataset: un evento es un dict JSON (nada de eval/SQL libre):
    {"column": "loc", "op": ">", "value": 100}            comparaciones: == != > >= < <= in between isnull notnull
    {"column": "cc",  "op": ">", "quantile": 0.75}        umbral por cuantil fijado ANTES; el valor resultante se registra
    {"and": [e1, e2]} | {"or": [e1, e2]} | {"not": e1}
Semantica de dos valores (igual en ambos metodos): un dato faltante NO cumple ninguna comparacion (salvo isnull); `not` es el complemento exacto.

Verificacion: Metodo A = pandas (mascaras booleanas); Metodo B = DuckDB (COUNT(*) FILTER con parametros enlazados, columnas validadas).
Las cuentas deben coincidir EXACTAMENTE. Bayes: formula con fractions.Fraction contra la forma de razon de momios con SymPy Rational y el conteo directo.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import sympy as sp
from statsmodels.stats.proportion import proportion_confint

from ada.config import DATA_RAW_DIR, DATA_SAMPLES_DIR, DATA_WORK_DIR
from ada.ingest.loader import load_dataset_frame
from ada.stats.inference import chi_square_independence

OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "between", "isnull", "notnull"}
MAX_DEPTH, MAX_LEAVES = 6, 20


# ----------------------------------------------------------------------------- eventos


def _validate(event: Any, df: pd.DataFrame, depth: int = 0, leaves: list[int] | None = None) -> None:
    leaves = leaves if leaves is not None else [0]
    if depth > MAX_DEPTH:
        raise ValueError(f"Evento demasiado anidado (maximo {MAX_DEPTH} niveles).")
    if not isinstance(event, dict):
        raise ValueError("Un evento debe ser un objeto JSON.")
    keys = set(event) - {"name"}
    if keys in ({"and"}, {"or"}):
        k = next(iter(keys))
        if not isinstance(event[k], list) or len(event[k]) < 2:
            raise ValueError(f"'{k}' necesita una lista de al menos 2 eventos.")
        for e in event[k]:
            _validate(e, df, depth + 1, leaves)
    elif keys == {"not"}:
        _validate(event["not"], df, depth + 1, leaves)
    else:
        leaves[0] += 1
        if leaves[0] > MAX_LEAVES:
            raise ValueError(f"Demasiadas condiciones (maximo {MAX_LEAVES}).")
        col, op = event.get("column"), event.get("op")
        if col not in df.columns:
            raise ValueError(f"La columna {col!r} no existe. Columnas: {list(df.columns)[:15]}")
        if op not in OPS:
            raise ValueError(f"Operador {op!r} no permitido. Usa: {sorted(OPS)}")
        if op in {"isnull", "notnull"}:
            return
        if "quantile" in event:
            q = event["quantile"]
            if not isinstance(q, (int, float)) or not 0 < q < 1 or not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError("'quantile' debe estar en (0,1) y la columna ser numerica.")
            if op not in {">", ">=", "<", "<="}:
                raise ValueError("Con 'quantile' usa un operador de orden (> >= < <=).")
            return
        v = event.get("value")
        if op == "between" and not (isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
            raise ValueError("'between' necesita value=[minimo, maximo] numericos.")
        if op == "in" and not (isinstance(v, list) and 0 < len(v) <= 50):
            raise ValueError("'in' necesita una lista de 1 a 50 valores.")
        if op not in {"between", "in"} and (isinstance(v, (dict, list)) or v is None):
            raise ValueError(f"El operador {op!r} necesita un valor escalar (numero, texto o booleano).")


def _resolve(event: dict[str, Any], df: pd.DataFrame) -> dict[str, Any]:
    """Copia del evento con los umbrales por cuantil convertidos a su valor numerico (queda registrado: anti p-hacking)."""
    if "and" in event or "or" in event:
        k = "and" if "and" in event else "or"
        return {**{x: y for x, y in event.items() if x != k}, k: [_resolve(e, df) for e in event[k]]}
    if "not" in event:
        return {**{x: y for x, y in event.items() if x != "not"}, "not": _resolve(event["not"], df)}
    if "quantile" in event:
        return {"column": event["column"], "op": event["op"], "value": float(df[event["column"]].quantile(event["quantile"])), "quantile": event["quantile"]}
    return dict(event)


def _mask(event: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    if "and" in event:
        m = pd.Series(True, index=df.index)
        for e in event["and"]:
            m &= _mask(e, df)
        return m
    if "or" in event:
        m = pd.Series(False, index=df.index)
        for e in event["or"]:
            m |= _mask(e, df)
        return m
    if "not" in event:
        return ~_mask(event["not"], df)
    s, op, v = df[event["column"]], event["op"], event.get("value")
    if op == "isnull":
        return s.isna()
    if op == "notnull":
        return s.notna()
    if op == "==":
        return (s == v).fillna(False)
    if op == "!=":
        return (s != v) & s.notna()
    if op == ">":
        return (s > v).fillna(False)
    if op == ">=":
        return (s >= v).fillna(False)
    if op == "<":
        return (s < v).fillna(False)
    if op == "<=":
        return (s <= v).fillna(False)
    if op == "in":
        return s.isin(v)
    return s.between(v[0], v[1])


def _sql(event: dict[str, Any], params: list[Any]) -> str:
    if "and" in event:
        return "(" + " AND ".join(_sql(e, params) for e in event["and"]) + ")"
    if "or" in event:
        return "(" + " OR ".join(_sql(e, params) for e in event["or"]) + ")"
    if "not" in event:
        return f"(NOT {_sql(event['not'], params)})"
    col = '"' + event["column"].replace('"', '""') + '"'
    op, v = event["op"], event.get("value")
    if op == "isnull":
        return f"({col} IS NULL)"
    if op == "notnull":
        return f"({col} IS NOT NULL)"
    if op == "in":
        params.extend(v)
        return f"COALESCE({col} IN ({', '.join('?' for _ in v)}), FALSE)"
    if op == "between":
        params.extend(v)
        return f"COALESCE(({col} BETWEEN ? AND ?), FALSE)"
    params.append(v)
    return f"COALESCE(({col} {'=' if op == '==' else '<>' if op == '!=' else op} ?), FALSE)"


def _describe(event: dict[str, Any]) -> str:
    if "and" in event:
        return " Y ".join(f"({_describe(e)})" for e in event["and"])
    if "or" in event:
        return " O ".join(f"({_describe(e)})" for e in event["or"])
    if "not" in event:
        return f"NO ({_describe(event['not'])})"
    q = f" [cuantil {event['quantile']}]" if "quantile" in event else ""
    if event["op"] in {"isnull", "notnull"}:
        return f"{event['column']} {'es nulo' if event['op'] == 'isnull' else 'no es nulo'}"
    return f"{event['column']} {event['op']} {event.get('value')}{q}"


def _counts(df: pd.DataFrame, events: dict[str, dict[str, Any]], combos: dict[str, tuple[str, ...]]) -> tuple[dict[str, int], dict[str, int], int]:
    """Cuentas por Metodo A (pandas) y Metodo B (DuckDB). `combos`: nombre -> eventos que deben cumplirse a la vez (con prefijo '~' = no)."""
    masks = {k: _mask(e, df) for k, e in events.items()}
    a: dict[str, int] = {}
    for name, parts in combos.items():
        m = pd.Series(True, index=df.index)
        for p in parts:
            m &= ~masks[p[1:]] if p.startswith("~") else masks[p]
        a[name] = int(m.sum())
    sel, params = [], []
    for name, parts in combos.items():
        conds = []
        for p in parts:
            neg = p.startswith("~")
            sql = _sql(events[p[1:] if neg else p], params)
            conds.append(f"(NOT {sql})" if neg else sql)
        sel.append(f"COUNT(*) FILTER (WHERE {' AND '.join(conds)}) AS \"{name}\"")
    con = duckdb.connect()
    try:
        con.register("t", df)
        row = con.execute(f"SELECT {', '.join(sel)} FROM t", params).fetchone()
    finally:
        con.close()
    return a, {name: int(v) for name, v in zip(combos, row)}, len(df)


def _load(path: str | None, df: pd.DataFrame | None) -> tuple[pd.DataFrame, str | None]:
    if df is not None:
        return df, None
    if not path:
        raise ValueError("Falta `path` del dataset (csv/parquet/xlsx...).")
    resolved = Path(path).resolve()
    if not any(r in resolved.parents for r in (DATA_RAW_DIR, DATA_SAMPLES_DIR, DATA_WORK_DIR)):
        raise ValueError("Solo se leen datasets dentro de data/raw, data/samples o data/work.")
    frame, meta = load_dataset_frame(resolved)
    return frame, meta["sha256"]


def _base(analysis: str, test: str, df: pd.DataFrame, sha: str | None) -> dict[str, Any]:
    return {"analysis": analysis, "test": test, "n": int(len(df)), "dataset_sha256": sha, "warnings": [], "rationale": []}


def _finish(out: dict[str, Any], a: dict[str, int], b: dict[str, int]) -> dict[str, Any]:
    out["counts"] = a
    out["verified"] = a == b
    out["discrepancy"] = float(sum(abs(a[k] - b[k]) for k in a))
    out["verification_methods"] = ["pandas (mascaras booleanas)", "DuckDB (COUNT(*) FILTER con parametros enlazados)"]
    if not out["verified"]:
        out["warnings"].append(f"Los dos metodos DISCREPAN en las cuentas: pandas={a} duckdb={b}. No uses este resultado.")
    return out


def _wilson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    lo, hi = proportion_confint(k, n, alpha=alpha, method="wilson")
    return [float(lo), float(hi)]


def _need(df: pd.DataFrame, *evs: Any) -> None:
    if len(df) == 0:
        raise ValueError("El dataset no tiene filas.")
    for e in evs:
        _validate(e, df)


# ----------------------------------------------------------------------------- operaciones


def event_probability(df: pd.DataFrame, A: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    _need(df, A)
    ev = {"A": _resolve(A, df)}
    a, b, n = _counts(df, ev, {"A": ("A",)})
    k = a["A"]
    out = _base("probability", "Probabilidad de un evento", df, sha)
    out.update(events={"A": _describe(ev["A"])}, resolved=ev, probability=k / n, ci={"level": 0.95, "of": "P(A) (Wilson)", "low": _wilson(k, n)[0], "high": _wilson(k, n)[1]},
               formula=f"P(A) = n(A)/n = {k}/{n} = {k / n:.6g}")
    out["decision"] = f"P(A) = {k / n:.4g} ({k} de {n} filas)."
    if k < 5 or n - k < 5:
        out["warnings"].append("Evento muy raro o casi seguro (menos de 5 casos en alguno de los lados): la estimacion es inestable.")
    return _finish(out, a, b)


def complement(df: pd.DataFrame, A: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    _need(df, A)
    ev = {"A": _resolve(A, df)}
    a, b, n = _counts(df, ev, {"A": ("A",), "notA": ("~A",)})
    out = _base("probability", "Complemento", df, sha)
    pa = Fraction(a["A"], n)
    out.update(events={"A": _describe(ev["A"])}, resolved=ev, probability=float(1 - pa), p_A=float(pa), formula=f"P(A^c) = 1 - P(A) = 1 - {a['A']}/{n} = {a['notA']}/{n}")
    ok = a["notA"] == n - a["A"]
    out["decision"] = f"P(no A) = {a['notA'] / n:.4g}; comprobado: n(A) + n(no A) = {a['A'] + a['notA']} = n = {n}."
    out["rationale"].append("El complemento se cuenta directamente (filas que NO cumplen A) y se contrasta con 1 - P(A).")
    res = _finish(out, a, b)
    res["verified"] = bool(res["verified"] and ok)
    return res


def intersection(df: pd.DataFrame, A: dict[str, Any], B: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    _need(df, A, B)
    ev = {"A": _resolve(A, df), "B": _resolve(B, df)}
    a, b, n = _counts(df, ev, {"AB": ("A", "B"), "A": ("A",), "B": ("B",)})
    out = _base("probability", "Interseccion", df, sha)
    out.update(events={"A": _describe(ev["A"]), "B": _describe(ev["B"])}, resolved=ev, probability=a["AB"] / n, formula=f"P(A ∩ B) = n(A∩B)/n = {a['AB']}/{n}")
    out["decision"] = f"P(A ∩ B) = {a['AB'] / n:.4g}."
    return _finish(out, a, b)


def union(df: pd.DataFrame, A: dict[str, Any], B: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    _need(df, A, B)
    ev = {"A": _resolve(A, df), "B": _resolve(B, df)}
    a, b, n = _counts(df, ev, {"A": ("A",), "B": ("B",), "AB": ("A", "B"), "onlyA": ("A", "~B"), "onlyB": ("~A", "B")})
    direct = a["A"] + a["onlyB"]                         # n(A ∪ B) contado directamente
    incl_excl = a["A"] + a["B"] - a["AB"]               # regla de la suma
    out = _base("probability", "Union (regla de la suma)", df, sha)
    naive = (a["A"] + a["B"]) / n
    out.update(events={"A": _describe(ev["A"]), "B": _describe(ev["B"])}, resolved=ev, probability=direct / n, naive_sum=naive, overlap=a["AB"] / n,
               formula=f"P(A ∪ B) = P(A) + P(B) - P(A ∩ B) = {a['A']}/{n} + {a['B']}/{n} - {a['AB']}/{n} = {direct}/{n}")
    out["decision"] = f"P(A ∪ B) = {direct / n:.4g}. Sumar P(A)+P(B) daria {naive:.4g}, que cuenta dos veces la interseccion ({a['AB'] / n:.4g})."
    out["rationale"].append("Se cuenta la union directamente y se contrasta con la regla de inclusion-exclusion.")
    res = _finish(out, a, b)
    res["verified"] = bool(res["verified"] and direct == incl_excl)
    return res


def conditional(df: pd.DataFrame, A: dict[str, Any], given: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    _need(df, A, given)
    ev = {"A": _resolve(A, df), "B": _resolve(given, df)}
    a, b, n = _counts(df, ev, {"AB": ("A", "B"), "B": ("B",), "A": ("A",)})
    if a["B"] == 0:
        raise ValueError("P(A|B) esta indefinida: ninguna fila cumple B.")
    out = _base("probability", "Probabilidad condicional", df, sha)
    p = Fraction(a["AB"], a["B"])
    out.update(events={"A": _describe(ev["A"]), "B (dado)": _describe(ev["B"])}, resolved=ev, probability=float(p), p_A=a["A"] / n, n_given=a["B"],
               ci={"level": 0.95, "of": "P(A|B) (Wilson)", "low": _wilson(a["AB"], a["B"])[0], "high": _wilson(a["AB"], a["B"])[1]},
               formula=f"P(A | B) = n(A∩B)/n(B) = {a['AB']}/{a['B']} = {float(p):.6g}")
    out["decision"] = f"P(A | B) = {float(p):.4g} frente a P(A) = {a['A'] / n:.4g}."
    if a["B"] < 5:
        out["warnings"].append(f"Solo {a['B']} filas cumplen B: la condicional es muy inestable.")
    return _finish(out, a, b)


def independence(df: pd.DataFrame, A: dict[str, Any], B: dict[str, Any], sha: str | None = None, alpha: float = 0.05) -> dict[str, Any]:
    _need(df, A, B)
    ev = {"A": _resolve(A, df), "B": _resolve(B, df)}
    a, b, n = _counts(df, ev, {"AB": ("A", "B"), "AnB": ("A", "~B"), "nAB": ("~A", "B"), "nAnB": ("~A", "~B"), "A": ("A",), "B": ("B",)})
    table = [[a["AB"], a["AnB"]], [a["nAB"], a["nAnB"]]]
    out = _base("probability", "Independencia de dos eventos", df, sha)
    pa, pb, pab = a["A"] / n, a["B"] / n, a["AB"] / n
    chi = chi_square_independence(table, alpha=alpha) if min(min(r) for r in table) >= 0 and all(sum(r) > 0 for r in table) and all(sum(c) > 0 for c in zip(*table)) else None
    odds = (a["AB"] * a["nAnB"]) / (a["AnB"] * a["nAB"]) if a["AnB"] * a["nAB"] else float("inf")
    out.update(events={"A": _describe(ev["A"]), "B": _describe(ev["B"])}, resolved=ev, table=table, p_A=pa, p_B=pb, p_AB=pab, p_A_times_p_B=pa * pb, odds_ratio=odds,
               formula=f"Si fueran independientes: P(A∩B) = P(A)·P(B) = {pa:.4g}·{pb:.4g} = {pa * pb:.4g}; observado {pab:.4g}")
    if chi is None:
        out["warnings"].append("Algun evento no ocurre o ocurre siempre: la independencia no es evaluable.")
        out["decision"] = "No evaluable."
    else:
        out.update(p_value=chi["p_value"], statistic=chi["statistic"], test_used=chi["test"], alpha=alpha)
        sig = chi["p_value"] < alpha
        out["decision"] = (f"Se rechaza la independencia ({chi['test']}, p = {chi['p_value']:.4g}): A y B parecen asociados." if sig else
                           f"No se rechaza la independencia ({chi['test']}, p = {chi['p_value']:.4g}). Eso NO demuestra que sean independientes: la independencia se evalua, no se asume.")
        out["rationale"].append(chi["rationale"][0])
        out["warnings"] += chi.get("warnings", [])
    res = _finish(out, a, b)
    if chi is not None:
        res["verified"] = bool(res["verified"] and chi["verified"])
        res["verification_methods"] += chi["verification_methods"]
    return res


def _bayes_terms(prior: Fraction, sens: Fraction, fpr: Fraction) -> dict[str, Fraction]:
    pos = sens * prior + fpr * (1 - prior)
    if pos == 0:
        raise ValueError("P(prueba positiva) = 0: el posterior esta indefinido.")
    return {"p_pos": pos, "tp": sens * prior, "fp": fpr * (1 - prior), "posterior": sens * prior / pos}


def _frac(x: Any, name: str) -> Fraction:
    try:
        f = Fraction(str(x))
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"{name} no es un numero valido: {x!r}") from None
    if not 0 <= f <= 1:
        raise ValueError(f"{name} debe estar entre 0 y 1 (recibido {x}).")
    return f


def _odds_posterior(prior: Fraction, sens: Fraction, fpr: Fraction) -> sp.Rational:
    """Metodo B: forma de razon de momios con SymPy: odds posterior = odds previa * (sens/fpr)."""
    pr, s, f = sp.Rational(prior.numerator, prior.denominator), sp.Rational(sens.numerator, sens.denominator), sp.Rational(fpr.numerator, fpr.denominator)
    if f == 0:
        return sp.Integer(1) if s > 0 and pr > 0 else sp.Integer(0)
    odds = pr / (1 - pr) * (s / f) if pr < 1 else sp.oo
    return sp.Integer(1) if odds is sp.oo else odds / (1 + odds)


def bayes_from_priors(prior: Any, sensitivity: Any, false_positive_rate: Any, per: int = 10_000) -> dict[str, Any]:
    """Teorema de Bayes sin dataset (ejercicios): P(D|+) con Fraction exacta, contrastado con la forma de razon de momios (SymPy)."""
    pr, se, fp = _frac(prior, "prior"), _frac(sensitivity, "sensibilidad"), _frac(false_positive_rate, "tasa de falsos positivos")
    t = _bayes_terms(pr, se, fp)
    alt = _odds_posterior(pr, se, fp)
    post = t["posterior"]
    verified = sp.Rational(post.numerator, post.denominator) == alt
    tp, fn_, fpc, tn = per * pr * se, per * pr * (1 - se), per * (1 - pr) * fp, per * (1 - pr) * (1 - fp)
    return {"analysis": "probability", "test": "Teorema de Bayes (sin dataset)", "posterior": float(post), "posterior_fraction": f"{post.numerator}/{post.denominator}", "prior": float(pr),
            "sensitivity": float(se), "false_positive_rate": float(fp), "p_positive": float(t["p_pos"]),
            "natural_frequencies": {"por_cada": per, "enfermos_positivos": float(tp), "enfermos_negativos": float(fn_), "sanos_positivos": float(fpc), "sanos_negativos": float(tn)},
            "formula": f"P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|no D)P(no D)] = {float(t['tp']):.6g} / ({float(t['tp']):.6g} + {float(t['fp']):.6g}) = {float(post):.6g}",
            "verified": bool(verified), "verification_methods": ["Fraction exacta (formula de Bayes)", "SymPy Rational (forma de razon de momios)"], "discrepancy": 0.0 if verified else 1.0,
            "rationale": [f"De {per} personas, {float(tp):.4g} enfermas dan positivo y {float(fpc):.4g} sanas tambien: solo {float(post):.1%} de los positivos esta enfermo."],
            "warnings": ["La probabilidad previa (prevalencia) domina el resultado: con prevalencia baja, un test bueno aun da muchos falsos positivos."] if pr < Fraction(1, 10) and post < Fraction(1, 2) else [],
            "decision": f"P(D | prueba positiva) = {float(post):.4g} ({post.numerator}/{post.denominator})."}


def bayes_from_counts(df: pd.DataFrame, D: dict[str, Any], C: dict[str, Any], sha: str | None = None) -> dict[str, Any]:
    """P(D|C) con las probabilidades estimadas del dataset (Fraction exacta), contrastado con el conteo directo n(D∩C)/n(C)."""
    _need(df, D, C)
    ev = {"D": _resolve(D, df), "C": _resolve(C, df)}
    a, b, n = _counts(df, ev, {"D": ("D",), "DC": ("D", "C"), "nDC": ("~D", "C"), "nD": ("~D",), "C": ("C",)})
    if a["D"] == 0 or a["nD"] == 0 or a["C"] == 0:
        raise ValueError("Bayes necesita que D, no-D y C ocurran al menos una vez en los datos.")
    pD, pCgD, pCgnD = Fraction(a["D"], n), Fraction(a["DC"], a["D"]), Fraction(a["nDC"], a["nD"])
    t = _bayes_terms(pD, pCgD, pCgnD)
    direct = Fraction(a["DC"], a["C"])
    out = _base("probability", "Teorema de Bayes (desde los datos)", df, sha)
    out.update(events={"D": _describe(ev["D"]), "C": _describe(ev["C"])}, resolved=ev, posterior=float(t["posterior"]), prior=float(pD), p_C_given_D=float(pCgD), p_C_given_notD=float(pCgnD),
               p_C=float(t["p_pos"]), direct_conditional=float(direct),
               formula=f"P(D|C) = P(C|D)P(D) / P(C) = {float(pCgD):.4g}·{float(pD):.4g} / {float(t['p_pos']):.4g} = {float(t['posterior']):.6g}   (conteo directo: {a['DC']}/{a['C']})")
    out["decision"] = f"P(D | C) = {float(t['posterior']):.4g} frente a P(D) = {float(pD):.4g}."
    out["warnings"].append("Estimacion empirica: describe ESTA muestra; no es una probabilidad poblacional ni una relacion causal.")
    if a["DC"] < 5:
        out["warnings"].append(f"Solo {a['DC']} filas cumplen D y C a la vez: estimacion inestable.")
    res = _finish(out, a, b)
    res["verified"] = bool(res["verified"] and t["posterior"] == direct)
    res["verification_methods"].append("conteo directo n(D∩C)/n(C) = Bayes (identidad exacta)")
    return res


def bayes_sequential(prior: Any, evidences: list[dict[str, Any]]) -> dict[str, Any]:
    """Actualiza la probabilidad con varias pruebas encadenadas. SUPUESTO EXPLICITO: pruebas condicionalmente independientes dada la enfermedad."""
    if not evidences:
        raise ValueError("Se necesita al menos una evidencia.")
    pr = _frac(prior, "prior")
    cur, steps = pr, []
    for i, e in enumerate(evidences, 1):
        se, fp = _frac(e.get("sensitivity"), f"sensibilidad {i}"), _frac(e.get("false_positive_rate"), f"tasa de falsos positivos {i}")
        positive = bool(e.get("positive", True))
        if positive:
            nxt = _bayes_terms(cur, se, fp)["posterior"]
        else:  # resultado negativo: P(D|-) con P(-|D)=1-se y P(-|no D)=1-fp
            nxt = _bayes_terms(cur, 1 - se, 1 - fp)["posterior"]
        steps.append({"paso": i, "resultado": "positivo" if positive else "negativo", "antes": float(cur), "despues": float(nxt)})
        cur = nxt
    # Metodo B: producto de razones de verosimilitud sobre la razon de momios previa.
    odds = sp.Rational(pr.numerator, pr.denominator) / (1 - sp.Rational(pr.numerator, pr.denominator)) if pr < 1 else sp.oo
    for e in evidences:
        se, fp, positive = _frac(e["sensitivity"], "s"), _frac(e["false_positive_rate"], "f"), bool(e.get("positive", True))
        a_, b_ = (se, fp) if positive else (1 - se, 1 - fp)
        odds = odds * (sp.Rational(a_.numerator, a_.denominator) / sp.Rational(b_.numerator, b_.denominator)) if b_ != 0 else sp.oo
    alt = sp.Integer(1) if odds is sp.oo else odds / (1 + odds)
    verified = sp.Rational(cur.numerator, cur.denominator) == alt
    return {"analysis": "probability", "test": "Bayes secuencial", "posterior": float(cur), "posterior_fraction": f"{cur.numerator}/{cur.denominator}", "prior": float(pr), "steps": steps,
            "verified": bool(verified), "verification_methods": ["Fraction: actualizacion paso a paso", "SymPy: producto de razones de verosimilitud sobre la razon de momios previa"],
            "discrepancy": 0.0 if verified else 1.0, "rationale": ["Cada resultado actualiza la probabilidad previa de la siguiente."],
            "warnings": ["SUPUESTO: las pruebas son condicionalmente independientes dada la condicion. Si comparten causas, el posterior real es MENOR que este."],
            "decision": f"Tras {len(evidences)} evidencias, P(D) pasa de {float(pr):.4g} a {float(cur):.4g}."}


def compact_probability(res: dict[str, Any]) -> dict[str, Any]:
    """Salida que ve Ada: sin eventos resueltos ni hash completo (ahorra contexto), cifras a 6 cifras significativas."""
    def rnd(o: Any) -> Any:
        if isinstance(o, float):
            return float(f"{o:.6g}") if o == o and o not in (float("inf"), float("-inf")) else None
        if isinstance(o, dict):
            return {k: rnd(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [rnd(v) for v in o]
        return o

    out = {k: v for k, v in res.items() if k not in {"resolved", "dataset_sha256"}}
    if res.get("dataset_sha256"):
        out["dataset_sha256"] = res["dataset_sha256"][:12]
    return rnd(out)


OPERATIONS = ("event", "complement", "intersection", "union", "conditional", "independence", "bayes", "bayes_priors", "bayes_sequential")


def run_probability(operation: str, *, path: str | None = None, A: dict | None = None, B: dict | None = None, params: dict | None = None, df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Punto de entrada unico (CLI, MCP y pruebas)."""
    p = params or {}
    if operation == "bayes_priors":
        return bayes_from_priors(p.get("prior"), p.get("sensitivity"), p.get("false_positive_rate"), int(p.get("per", 10_000)))
    if operation == "bayes_sequential":
        return bayes_sequential(p.get("prior"), p.get("evidences") or [])
    if operation not in OPERATIONS:
        raise ValueError(f"operation desconocida: {operation!r}. Disponibles: {', '.join(OPERATIONS)}")
    frame, sha = _load(path, df)
    need_b = operation in {"intersection", "union", "conditional", "independence", "bayes"}
    if A is None or (need_b and B is None):
        raise ValueError(f"'{operation}' necesita {'A y B' if need_b else 'A'} (eventos JSON).")
    if operation == "event":
        return event_probability(frame, A, sha)
    if operation == "complement":
        return complement(frame, A, sha)
    if operation == "intersection":
        return intersection(frame, A, B, sha)
    if operation == "union":
        return union(frame, A, B, sha)
    if operation == "conditional":
        return conditional(frame, A, B, sha)
    if operation == "independence":
        return independence(frame, A, B, sha, alpha=float(p.get("alpha", 0.05)))
    return bayes_from_counts(frame, A, B, sha)

