"""Suite pytest de aceptacion para la Fase 1 (Entorno reproducible, seguridad de rutas, GPU y doble metodo)."""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
import scipy.integrate
import statsmodels.api as sm
import sympy as sp
import torch

from ada.config import (
    DATA_RAW_DIR,
    DATA_WORK_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    SecurityViolationError,
    assert_safe_write_path,
    set_global_seed,
)


def test_deterministic_seed_reproducibility() -> None:
    set_global_seed(42)
    a_np = np.random.normal(size=10)
    a_th = torch.randn(10)

    set_global_seed(42)
    b_np = np.random.normal(size=10)
    b_th = torch.randn(10)

    assert np.array_equal(a_np, b_np)
    assert torch.equal(a_th, b_th)


def test_filesystem_security_boundaries() -> None:
    # Rutas autorizadas deben pasar
    assert assert_safe_write_path(DATA_WORK_DIR / "clean_dataset.parquet") == (
        DATA_WORK_DIR / "clean_dataset.parquet"
    ).resolve()
    assert assert_safe_write_path(OUTPUTS_DIR / "plot.png") == (OUTPUTS_DIR / "plot.png").resolve()
    assert assert_safe_write_path(REPORTS_DIR / "audit.md") == (REPORTS_DIR / "audit.md").resolve()

    # data/raw es estrictamente inmutable
    with pytest.raises(SecurityViolationError):
        assert_safe_write_path(DATA_RAW_DIR / "original.csv")

    # src/, config/ y rutas fuera del workspace estan bloqueadas
    with pytest.raises(SecurityViolationError):
        assert_safe_write_path(PROJECT_ROOT / "src" / "ada" / "hack.py")
    with pytest.raises(SecurityViolationError):
        assert_safe_write_path(Path("C:/Windows/Temp/outside.txt"))


def test_rtx5070_blackwell_cuda_and_bf16() -> None:
    assert torch.cuda.is_available(), "CUDA debe estar disponible en la RTX 5070"
    major, minor = torch.cuda.get_device_capability(0)
    assert (major, minor) == (12, 0), f"Se esperaba Blackwell sm_120 (12, 0), obtenido ({major}, {minor})"
    assert torch.cuda.is_bf16_supported(), "Blackwell debe soportar BFloat16"

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda", dtype=torch.float64)
    inv_gpu = torch.linalg.inv(x).cpu().numpy()
    inv_cpu = np.linalg.inv(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    assert np.allclose(inv_gpu, inv_cpu, atol=1e-12)


def test_dual_method_calculus_and_data_engines() -> None:
    # Calculo verificado: integral definida de x^2 * exp(-x) en [0, 3]
    x = sp.Symbol("x", real=True)
    exact_sym = float(sp.integrate(x**2 * sp.exp(-x), (x, 0, 3)).evalf(25))
    num_quad, _ = scipy.integrate.quad(lambda t: (t**2) * np.exp(-t), 0.0, 3.0, epsabs=1e-13)
    assert abs(exact_sym - num_quad) < 1e-12

    # Polars + DuckDB + Statsmodels OLS verificado contra ecuacion normal matricial
    set_global_seed(42)
    n = 200
    t_vals = np.linspace(0.5, 2.5, n)
    # Simulacion lineal T^2 = (4*pi^2 / g) * L con g = 9.80665
    g_true = 9.80665
    slope_true = (4.0 * np.pi**2) / g_true
    y_vals = slope_true * t_vals + np.random.normal(0.0, 0.005, size=n)

    df = pl.DataFrame({"L": t_vals, "T2": y_vals})
    rows = duckdb.sql("SELECT COUNT(*) AS n, AVG(L) AS mean_l FROM df").fetchall()
    assert len(rows) == 1 and rows[0][0] == n

    X = sm.add_constant(df["L"].to_numpy())
    y = df["T2"].to_numpy()
    ols_sm = sm.OLS(y, X).fit()
    beta_lstsq, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(ols_sm.params, beta_lstsq, atol=1e-12)
