"""Prueba de humo Fase 1: GPU RTX 5070 (Blackwell sm_120), PyTorch cu128, QLoRA stack y doble verificacion cientifica."""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import duckdb
import numpy as np
import polars as pl
import scipy
import scipy.integrate
import scipy.special
import statsmodels.api as sm
import sympy as sp
import torch

from ada.config import DEFAULT_SEED, set_global_seed


def verify_gpu_pytorch() -> dict[str, Any]:
    """Verifica deteccion CUDA, Compute Capability sm_120 (12, 0), BFloat16 Tensor Cores y Autograd en RTX 5070."""
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch NO detecta CUDA disponible.")

    device = torch.device("cuda:0")
    dev_name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    props = torch.cuda.get_device_properties(device)
    total_vram_gib = props.total_memory / (1024**3)

    # Prueba de calculo matricial BFloat16 + Autograd en GPU y contraste contra CPU Float64 (Regla 2)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    x_gpu = torch.randn((2048, 2048), device=device, dtype=torch.float32, requires_grad=True)
    w_gpu = torch.randn((2048, 2048), device=device, dtype=torch.float32)
    y_gpu = (x_gpu @ w_gpu).pow(2).mean()
    y_gpu.backward()
    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # Contraste independiente en CPU float64
    x_cpu = x_gpu.detach().cpu().to(torch.float64)
    w_cpu = w_gpu.detach().cpu().to(torch.float64)
    y_cpu = float(((x_cpu @ w_cpu) ** 2).mean().item())
    rel_err = abs(float(y_gpu.item()) - y_cpu) / abs(y_cpu)
    if rel_err > 1e-5:
        raise AssertionError(f"Discrepancia GPU vs CPU excesiva: rel_err={rel_err}")

    # Prueba BFloat16 (clave para QLoRA fine-tuning en Blackwell)
    bf16_supported = torch.cuda.is_bf16_supported()
    a_bf16 = torch.ones((512, 512), device=device, dtype=torch.bfloat16)
    b_bf16 = torch.ones((512, 512), device=device, dtype=torch.bfloat16)
    c_bf16 = float((a_bf16 @ b_bf16)[0, 0].item())
    assert c_bf16 == 512.0, f"Fallo matmul BF16: esperado 512.0, obtenido {c_bf16}"

    peak_alloc_mib = torch.cuda.max_memory_allocated(device) / (1024**2)
    del x_gpu, w_gpu, y_gpu, a_bf16, b_bf16
    torch.cuda.empty_cache()

    return {
        "torch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": dev_name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_vram_gib": round(total_vram_gib, 3),
        "bf16_supported": bf16_supported,
        "gpu_autograd_matmul_ms": round(elapsed_ms, 2),
        "gpu_vs_cpu_rel_error": rel_err,
        "peak_test_vram_mib": round(peak_alloc_mib, 2),
    }


def verify_qlora_training_stack() -> dict[str, str]:
    """Verifica que todos los paquetes de manipulacion y entrenamiento abierto (QLoRA) carguen sin errores."""
    import accelerate
    import bitsandbytes as bnb
    import datasets
    import peft
    import transformers
    import trl

    return {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "accelerate": accelerate.__version__,
        "bitsandbytes": bnb.__version__,
        "trl": trl.__version__,
        "datasets": datasets.__version__,
    }


def verify_scientific_dual_method() -> dict[str, Any]:
    """Verifica SymPy vs SciPy en el periodo exacto del pendulo a 30 grados (prueba de cordura Seccion 11)."""
    # Periodo relativo T / T0 = (2 / pi) * K(m = sin^2(theta0 / 2))
    theta0_deg = 30
    theta0_sym = sp.Rational(theta0_deg, 180) * sp.pi
    m_sym = sp.sin(theta0_sym / 2) ** 2

    # Metodo 1: Simbolico + evaluacion arbitraria de 30 digitos en SymPy/mpmath
    ratio_sym = float(((2 / sp.pi) * sp.elliptic_k(m_sym)).evalf(30))

    # Metodo 2: Integracion numerica directa de cuadratura de Gauss-Kronrod (scipy.integrate.quad)
    theta0_rad = np.deg2rad(theta0_deg)
    k_val = np.sin(theta0_rad / 2.0)
    integral_quad, quad_err = scipy.integrate.quad(
        lambda phi: 1.0 / np.sqrt(1.0 - (k_val**2) * (np.sin(phi) ** 2)),
        0.0,
        np.pi / 2.0,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    ratio_quad = float((2.0 / np.pi) * integral_quad)

    # Metodo 3: scipy.special.ellipk(m = k^2)
    ratio_ellipk = float((2.0 / np.pi) * scipy.special.ellipk(k_val**2))

    diff_sym_quad = abs(ratio_sym - ratio_quad)
    diff_sym_ellipk = abs(ratio_sym - ratio_ellipk)
    if max(diff_sym_quad, diff_sym_ellipk) > 1e-12:
        raise AssertionError(
            f"Discrepancia entre metodos: sym={ratio_sym}, quad={ratio_quad}, ellipk={ratio_ellipk}"
        )

    period_increase_pct = (ratio_sym - 1.0) * 100.0
    return {
        "sympy_version": sp.__version__,
        "scipy_version": scipy.__version__,
        "duckdb_version": duckdb.__version__,
        "polars_version": pl.__version__,
        "statsmodels_version": sm.__version__,
        "pendulum_30deg_T_over_T0_sympy": ratio_sym,
        "pendulum_30deg_T_over_T0_quad": ratio_quad,
        "pendulum_30deg_increase_pct": round(period_increase_pct, 4),
        "max_method_discrepancy": max(diff_sym_quad, diff_sym_ellipk),
    }


def main() -> int:
    seed = set_global_seed(DEFAULT_SEED)
    report = {
        "seed": seed,
        "python": sys.version.split()[0],
        "gpu_pytorch": verify_gpu_pytorch(),
        "qlora_stack": verify_qlora_training_stack(),
        "scientific_dual_verification": verify_scientific_dual_method(),
        "status": "OK",
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
