"""Generador determinista de activos de prueba en data/samples/ para los Proyectos 1 y 2."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ada.config import DATA_SAMPLES_DIR, set_global_seed


def create_sample_sqlite_database() -> Path:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    DATA_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_SAMPLES_DIR / "industrial_energy_sensors.sqlite"
    if db_path.exists():
        db_path.unlink()

    n = 240
    zones = rng.choice(["Norte", "Sur", "Centro", "Norte "], size=n, p=[0.35, 0.30, 0.25, 0.10])
    temp_c = rng.normal(26.5, 4.8, size=n)
    rpm = rng.uniform(900.0, 2400.0, size=n)
    vib = 0.8 + 0.0014 * rpm + rng.gamma(2.0, 0.35, size=n)
    voltage = rng.normal(220.0, 4.5, size=n)

    true_kw = (
        12.5
        + 0.038 * rpm
        + 0.000011 * (rpm**2)
        - 0.42 * np.maximum(0.0, temp_c - 25.0)
        - 1.85 * vib
        + 0.09 * (voltage - 220.0)
        + np.where(np.char.strip(zones.astype(str)) == "Norte", 3.4, 0.0)
    )
    noise = rng.normal(0.0, 1.35, size=n)
    power_kw = np.round(true_kw + noise, 3)

    df = pd.DataFrame(
        {
            "sensor_uid": [f"SNS-{1000 + i}" for i in range(n)],
            "plant_zone": zones,
            "ambient_temp_c": np.round(temp_c, 2),
            "motor_rpm": np.round(rpm, 1),
            "vibration_mms": np.round(vib, 3),
            "line_voltage_v": np.round(voltage, 2),
            "power_output_kw": power_kw,
        }
    )

    df.loc[rng.choice(n - 10, size=11, replace=False), "ambient_temp_c"] = np.nan
    df.loc[rng.choice(n - 10, size=7, replace=False), "line_voltage_v"] = np.nan
    df.loc[[5, 42], "vibration_mms"] = [14.8, 16.2]
    df.loc[n - 6 :, "power_output_kw"] = np.nan

    df = pd.concat([df, df.iloc[[2, 10, 18, 25, 33]]], ignore_index=True)

    conn = sqlite3.connect(str(db_path))
    try:
        df.to_sql("turbine_telemetry", conn, index=False, if_exists="replace")
    finally:
        conn.close()
    return db_path


def create_graphical_ode_image() -> Path:
    DATA_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    img_path = DATA_SAMPLES_DIR / "graphical_pendulum_ode.png"
    fig, ax = plt.subplots(figsize=(7.5, 2.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.axis("off")
    ax.text(
        0.5,
        0.52,
        "d2y/dt2 + 0.4*dy/dt + 9.81*sin(y) = 0",
        fontsize=21,
        fontfamily="monospace",
        fontweight="bold",
        color="black",
        ha="center",
        va="center",
    )
    fig.savefig(img_path, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return img_path
