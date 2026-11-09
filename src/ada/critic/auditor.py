"""Agente Critico Estadistico deterministicamente verificado: cuestiona supuestos, tamano muestral y fugas."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats


def audit_regression_results(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    best_r2: float,
    dummy_r2: float,
    best_rmse: float,
    dummy_rmse: float,
    residuals_test: np.ndarray,
    y_pred_test: np.ndarray,
    design_condition_number: float | None = None,
) -> dict[str, Any]:
    """Audita resultados de regresion cuestionando fuga de datos, potencia, normalidad y heterocedasticidad."""
    warnings_list: list[str] = []
    passed_checks: list[str] = []
    severity = "OK"

    if n_train + n_test < 50 or n_train < max(20, 5 * n_features):
        warnings_list.append(
            f"POTENCIA MUESTRAL BAJA: n_train={n_train} para {n_features} predictores (recomendado >= {max(30, 10 * n_features)}). "
            "Los intervalos de confianza seran anchos y hay riesgo de sobreajuste."
        )
        severity = "CAUTION"
    else:
        passed_checks.append(f"Relacion muestras/predictores aceptable (n_train={n_train}, p={n_features}).")

    if best_r2 >= 0.995:
        warnings_list.append(
            f"ALERTA DE POSIBLE DATA LEAKAGE: R^2={best_r2:.5f} es sospechosamente cercano a 1.0. "
            "Revisa si una columna predictora es derivada directa del target o contiene informacion futura."
        )
        severity = "HIGH_RISK"

    rmse_gain_pct = 100.0 * (dummy_rmse - best_rmse) / max(dummy_rmse, 1e-12)
    if best_r2 <= max(0.05, dummy_r2 + 0.02) or rmse_gain_pct < 2.0:
        warnings_list.append(
            f"PODER PREDICTIVO DEBIL: El mejor modelo apenas mejora el baseline ingenuo en {rmse_gain_pct:.2f}% de RMSE (R^2={best_r2:.4f}). "
            "No tomes decisiones fuertes basadas en estas predicciones sin mejores variables."
        )
        if severity == "OK":
            severity = "CAUTION"
    else:
        passed_checks.append(f"Supera al baseline ingenuo con reduccion de {rmse_gain_pct:.2f}% en RMSE.")

    diagnostics: dict[str, Any] = {"rmse_improvement_vs_baseline_pct": round(float(rmse_gain_pct), 3)}

    res = np.asarray(residuals_test, dtype=float)
    res = res[np.isfinite(res)]
    if res.size >= 8:
        sample_res = res[:5000]
        sw_stat, sw_p = stats.shapiro(sample_res)
        jb_stat, jb_p = stats.jarque_bera(sample_res)
        diagnostics["shapiro_wilk"] = {"statistic": round(float(sw_stat), 5), "p_value": float(sw_p)}
        diagnostics["jarque_bera"] = {"statistic": round(float(jb_stat), 5), "p_value": float(jb_p)}
        if sw_p < 0.01 and jb_p < 0.01:
            warnings_list.append(
                f"RESIDUOS NO NORMALES (Shapiro p={sw_p:.3g}, Jarque-Bera p={jb_p:.3g}): "
                "Usa intervalos de prediccion no parametricos/cuantilicos (incluidos por ADA), no intervalos gaussianos clasicos."
            )
        else:
            passed_checks.append("Normalidad de residuos no rechazada al 1% en ambos tests (Shapiro-Wilk y Jarque-Bera).")

        preds = np.asarray(y_pred_test, dtype=float)[: res.size]
        if np.std(preds) > 1e-12 and np.std(np.abs(res)) > 1e-12:
            rho, p_het = stats.spearmanr(np.abs(preds), np.abs(res))
            diagnostics["heteroscedasticity_spearman"] = {"rho": round(float(rho), 5), "p_value": float(p_het)}
            if abs(rho) > 0.30 and p_het < 0.05:
                warnings_list.append(
                    f"HETEROCEDASTICIDAD DETECTADA (Spearman |y_hat| vs |residuo| rho={rho:.3f}, p={p_het:.3g}): "
                    "La varianza del error cambia segun la magnitud predicha; considera transformar el target (ej. log1p)."
                )

    if design_condition_number is not None:
        diagnostics["design_condition_number"] = round(float(design_condition_number), 3)
        if design_condition_number > 100.0:
            warnings_list.append(
                f"MULTICOLINEALIDAD ELEVADA (numero de condicion={design_condition_number:.1f} > 100): "
                "Los coeficientes lineales individuales son inestables; interpreta con regularizacion Ridge o importancia por permutacion."
            )

    return {
        "severity": severity,
        "warnings": warnings_list,
        "passed_checks": passed_checks,
        "diagnostics": diagnostics,
    }


def audit_classification_results(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    class_distribution: dict[str, int],
    best_f1_macro: float,
    dummy_f1_macro: float,
    best_accuracy: float,
    dummy_accuracy: float,
    roc_auc: float | None,
) -> dict[str, Any]:
    """Audita resultados de clasificacion cuestionando desbalance de clases, accuracy enganosa y fuga."""
    warnings_list: list[str] = []
    passed_checks: list[str] = []
    severity = "OK"

    counts = list(class_distribution.values())
    total = max(sum(counts), 1)
    min_class = min(counts) if counts else 0
    imbalance_ratio = (max(counts) / max(min_class, 1)) if counts else 1.0

    if n_train + n_test < 60 or min_class < 15:
        warnings_list.append(
            f"CLASE MINORITARIA PEQUENA (min_count={min_class}, n_total={n_train + n_test}): "
            "Alta varianza en metricas de test; interpreta el intervalo bootstrap con cautela."
        )
        severity = "CAUTION"

    if imbalance_ratio >= 3.0:
        warnings_list.append(
            f"DESBALANCE DE CLASES ({imbalance_ratio:.1f}:1): La exactitud global (Accuracy={best_accuracy:.3f}) es enganosa; "
            f"prioriza F1-Macro ({best_f1_macro:.3f}) y ROC-AUC."
        )

    if (roc_auc is not None and roc_auc >= 0.998) or best_f1_macro >= 0.995:
        warnings_list.append(
            "ALERTA DE POSIBLE DATA LEAKAGE: Desempeno casi perfecto (>= 0.995). "
            "Verifica que ninguna columna filtre la etiqueta real post-evento."
        )
        severity = "HIGH_RISK"

    if best_f1_macro <= dummy_f1_macro + 0.03:
        warnings_list.append(
            f"GANANCIA MARGINAL SOBRE BASELINE: F1-Macro={best_f1_macro:.3f} vs Dummy={dummy_f1_macro:.3f}. "
            "El modelo no encontro senal discriminante robusta."
        )
        if severity == "OK":
            severity = "CAUTION"
    else:
        passed_checks.append(
            f"Mejora sustancial sobre clasificador ingenuo (+{best_f1_macro - dummy_f1_macro:.3f} en F1-Macro)."
        )

    return {
        "severity": severity,
        "imbalance_ratio": round(float(imbalance_ratio), 3),
        "warnings": warnings_list,
        "passed_checks": passed_checks,
    }
