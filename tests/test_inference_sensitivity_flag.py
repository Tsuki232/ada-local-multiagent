"""`compare_two_groups(..., sensitivity=False)`: omite el analisis de sensibilidad automatico (que no escala a decenas de miles de filas)
sin cambiar el resultado de la prueba pedida y avisando en `warnings`. Por defecto el comportamiento anterior no cambia."""
from __future__ import annotations

import inspect

import numpy as np

from ada.stats.inference import compare_two_groups


def _skewed(seed: int = 7, n: int = 40):
    rng = np.random.default_rng(seed)
    return rng.lognormal(0.0, 1.2, n), rng.lognormal(0.5, 1.2, n)  # asimetricos: Shapiro rechaza normalidad


def test_por_defecto_la_sensibilidad_sigue_activa():
    assert inspect.signature(compare_two_groups).parameters["sensitivity"].default is True
    x, y = _skewed()
    assert "sensitivity" in compare_two_groups(x, y, method="welch")


def test_sin_sensibilidad_no_cambia_la_prueba_y_lo_avisa():
    x, y = _skewed()
    con = compare_two_groups(x, y, method="welch")
    sin = compare_two_groups(x, y, method="welch", sensitivity=False)
    assert "sensitivity" not in sin
    for k in ("test", "statistic", "df", "p_value", "estimate", "verified"):
        assert sin[k] == con[k], k
    assert any("sensitivity=False" in w for w in sin["warnings"])
    assert sin["verified"] is True


def test_con_datos_normales_no_hay_aviso_de_sensibilidad():
    rng = np.random.default_rng(3)
    r = compare_two_groups(rng.normal(0, 1, 60), rng.normal(0.3, 1, 60), method="welch", sensitivity=False)
    assert "sensitivity" not in r and not any("sensitivity=False" in w for w in r["warnings"])
