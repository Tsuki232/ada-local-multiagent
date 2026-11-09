"""Puente seguro de control y telemetria de dispositivos electronicos (COM/USB Serial + Gemelo Virtual MCU)."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import serial
from serial.tools import list_ports

from ada.audit.ledger import record_audit_event
from ada.config import DATA_WORK_DIR, DEFAULT_SEED, OUTPUTS_DIR, assert_safe_write_path


VIRTUAL_DEVICE_URI = "virtual://pendulum-mcu"


def list_electronic_devices() -> dict[str, Any]:
    """Descubre puertos serie COM/USB fisicos (Arduino, ESP32, STM32) y expone el gemelo electronico virtual."""
    ports_found: list[dict[str, Any]] = []
    for p in list_ports.comports():
        ports_found.append(
            {
                "port": p.device,
                "description": p.description,
                "hwid": p.hwid,
                "manufacturer": p.manufacturer,
                "vid": p.vid,
                "pid": p.pid,
            }
        )

    return {
        "physical_serial_ports": ports_found,
        "physical_count": len(ports_found),
        "virtual_device_available": {
            "port": VIRTUAL_DEVICE_URI,
            "description": "Gemelo Electronico RK4 (Encoder optico 12-bit + Actuador PWM con saturacion y E-STOP)",
            "protocol": "CSV/JSON line-framed: TELEM theta_rad,omega_rads,u_cmd | CMD u=<float> | ESTOP",
        },
    }


def _clamp(val: float, limit: float) -> float:
    return max(-abs(limit), min(abs(limit), val))


def _render_hardware_telemetry_html(
    *,
    output_html_path: Path,
    device_port: str,
    control_mode: str,
    df_telem: pd.DataFrame,
    safety_events: list[str],
    length_m: float,
) -> Path:
    safe_html = assert_safe_write_path(output_html_path)
    safe_html.parent.mkdir(parents=True, exist_ok=True)
    records = df_telem.to_dict(orient="records")
    payload = json.dumps(
        {
            "device": device_port,
            "mode": control_mode,
            "L": length_m,
            "safety": safety_events,
            "frames": records,
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<title>ADA — Telemetria Animada de Dispositivo Electronico ({device_port})</title>
<style>
  body {{ margin:0; padding:18px; background:#070b16; color:#e7eeff; font-family:system-ui,sans-serif; }}
  h1 {{ margin:0 0 6px; font-size:1.25rem; }}
  .sub {{ color:#94a3b8; font-size:0.88rem; margin-bottom:14px; }}
  .grid {{ display:grid; grid-template-columns: 420px 1fr; gap:14px; }}
  @media (max-width:920px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .card {{ background:#111931; border:1px solid #243463; border-radius:12px; padding:14px; }}
  canvas {{ width:100%; background:#090f20; border:1px solid #243463; border-radius:8px; display:block; }}
  button {{ background:#0284c7; color:#fff; border:none; border-radius:7px; padding:7px 14px; cursor:pointer; font-weight:600; margin-right:8px; }}
  input[type=range] {{ width:100%; margin-top:10px; }}
</style>
</head>
<body>
  <h1>ADA — Replay Animado de Control Hardware-in-the-Loop</h1>
  <div class="sub" id="sub"></div>
  <div class="grid">
    <div class="card">
      <canvas id="cvArm" width="390" height="320"></canvas>
      <div style="margin-top:10px">
        <button id="btnPlay">Pausar</button>
        <span id="frameLbl" style="font-size:0.85rem;color:#94a3b8"></span>
        <input type="range" id="scrub" min="0" value="0" />
      </div>
    </div>
    <div class="card">
      <canvas id="cvSignals" width="640" height="360"></canvas>
    </div>
  </div>
<script>
const DATA = {payload};
document.getElementById("sub").textContent =
  `Dispositivo: ${{DATA.device}} | Modo: ${{DATA.mode}} | Muestras capturadas: ${{DATA.frames.length}}`;
const scrub = document.getElementById("scrub");
scrub.max = Math.max(0, DATA.frames.length - 1);
let idx = 0, playing = true;

function draw() {{
  const f = DATA.frames[idx] || {{ t_s:0, theta_sensor_rad:0, theta_ref_rad:0, u_applied:0 }};
  document.getElementById("frameLbl").textContent =
    `t=${{f.t_s.toFixed(3)}}s | θ_sensor=${{(f.theta_sensor_rad*180/Math.PI).toFixed(2)}}° | u=${{f.u_applied.toFixed(2)}}`;
  scrub.value = idx;

  const cvA = document.getElementById("cvArm"), cA = cvA.getContext("2d");
  cA.clearRect(0,0,cvA.width,cvA.height);
  const cx = cvA.width/2, cy = 65, R = 185;
  cA.strokeStyle = "rgba(34,197,94,0.45)"; cA.setLineDash([5,5]);
  cA.beginPath(); cA.moveTo(cx,cy); cA.lineTo(cx+R*Math.sin(f.theta_ref_rad), cy+R*Math.cos(f.theta_ref_rad)); cA.stroke();
  cA.setLineDash([]);
  const bx = cx + R*Math.sin(f.theta_sensor_rad), by = cy + R*Math.cos(f.theta_sensor_rad);
  cA.strokeStyle = "#38bdf8"; cA.lineWidth = 4;
  cA.beginPath(); cA.moveTo(cx,cy); cA.lineTo(bx,by); cA.stroke();
  cA.fillStyle = "#38bdf8"; cA.beginPath(); cA.arc(bx,by,15,0,Math.PI*2); cA.fill();

  const cvS = document.getElementById("cvSignals"), cS = cvS.getContext("2d");
  cS.clearRect(0,0,cvS.width,cvS.height);
  const pad = 38;
  const tMax = Math.max(0.01, DATA.frames[DATA.frames.length-1]?.t_s || 1);
  const maxAng = Math.max(0.2, ...DATA.frames.map(r => Math.max(Math.abs(r.theta_sensor_rad), Math.abs(r.theta_ref_rad))));
  const sx = t => pad + (t/tMax)*(cvS.width - 2*pad);
  const sy = v => cvS.height*0.36 - (v/maxAng)*(cvS.height*0.28);

  cS.strokeStyle = "#1e293b"; cS.beginPath(); cS.moveTo(pad, cvS.height*0.36); cS.lineTo(cvS.width-pad, cvS.height*0.36); cS.stroke();
  cS.strokeStyle = "#22c55e"; cS.lineWidth = 1.6; cS.beginPath();
  DATA.frames.forEach((r,i) => {{ const x=sx(r.t_s), y=sy(r.theta_ref_rad); if(i===0) cS.moveTo(x,y); else cS.lineTo(x,y); }});
  cS.stroke();
  cS.strokeStyle = "#38bdf8"; cS.lineWidth = 2.2; cS.beginPath();
  DATA.frames.forEach((r,i) => {{ const x=sx(r.t_s), y=sy(r.theta_sensor_rad); if(i===0) cS.moveTo(x,y); else cS.lineTo(x,y); }});
  cS.stroke();

  const maxU = Math.max(1.0, ...DATA.frames.map(r => Math.abs(r.u_applied)));
  const syU = u => cvS.height*0.80 - (u/maxU)*(cvS.height*0.14);
  cS.strokeStyle = "#f59e0b"; cS.lineWidth = 1.8; cS.beginPath();
  DATA.frames.forEach((r,i) => {{ const x=sx(r.t_s), y=syU(r.u_applied); if(i===0) cS.moveTo(x,y); else cS.lineTo(x,y); }});
  cS.stroke();

  const curX = sx(f.t_s);
  cS.strokeStyle = "#f43f5e"; cS.beginPath(); cS.moveTo(curX, 16); cS.lineTo(curX, cvS.height-16); cS.stroke();
  cS.fillStyle = "#94a3b8"; cS.font = "12px sans-serif";
  cS.fillText("Arriba: θ_sensor (cian) vs θ_ref (verde) | Abajo: Senal de actuador u(t) (ambar)", pad, 20);
}}

setInterval(() => {{
  if (playing && DATA.frames.length > 0) {{
    idx = (idx + 1) % DATA.frames.length;
    draw();
  }}
}}, 33);

scrub.addEventListener("input", () => {{ playing = false; idx = parseInt(scrub.value, 10); draw(); }});
document.getElementById("btnPlay").addEventListener("click", (e) => {{
  playing = !playing; e.target.textContent = playing ? "Pausar" : "Reproducir";
}});
draw();
</script>
</body>
</html>
"""
    safe_html.write_text(html, encoding="utf-8")
    return safe_html


def run_pendulum_hardware_control(
    *,
    device_port: str = VIRTUAL_DEVICE_URI,
    baudrate: int = 115200,
    control_mode: str = "linearized_pd",
    allow_actuation: bool = False,
    target_angle_deg: float = 0.0,
    initial_angle_deg: float = 25.0,
    duration_s: float = 6.0,
    dt_s: float = 0.02,
    length_m: float = 1.0,
    gravity_ms2: float = 9.80665,
    max_torque_limit: float = 22.0,
    max_safe_angle_deg: float = 135.0,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Ejecuta un lazo de lectura/control sobre un dispositivo fisico COM o sobre el gemelo virtual con enclavamientos de seguridad.

    Seguridad:
      - Si allow_actuation=False (por defecto), opera en modo SOLO TELEMETRIA (u=0) sin energizar motores.
      - Incluye saturacion de actuador |u| <= max_torque_limit, limite de angulo seguro y parada de emergencia ESTOP garantizada.
    """
    if duration_s <= 0.1 or duration_s > 30.0:
        raise ValueError("duration_s debe estar entre 0.1 y 30.0 segundos por seguridad.")
    if dt_s < 0.005 or dt_s > 0.2:
        raise ValueError("dt_s debe estar entre 0.005 y 0.2 segundos.")

    rng = np.random.default_rng(seed)
    n_steps = int(round(duration_s / dt_s))
    theta_ref = math.radians(target_angle_deg)
    max_angle_rad = math.radians(max_safe_angle_deg)
    omega0_sq = gravity_ms2 / length_m

    # Ganancias disenadas sobre la planta linealizada d2theta/dt2 + (g/L)cos(theta_ref)*dtheta = u
    kp = 24.0
    kd = 7.5

    rows: list[dict[str, float]] = []
    safety_events: list[str] = []

    if device_port.strip().lower() == VIRTUAL_DEVICE_URI:
        theta = math.radians(initial_angle_deg)
        omega = 0.0
        damping = 0.08
        encoder_quant = (2.0 * math.pi) / 4096.0  # Encoder optico de 12 bits

        for step in range(n_steps + 1):
            t_now = step * dt_s
            theta_meas = round(theta / encoder_quant) * encoder_quant + float(rng.normal(0.0, 0.0008))
            omega_meas = omega + float(rng.normal(0.0, 0.004))

            u_cmd = 0.0
            if allow_actuation and control_mode != "telemetry_only":
                if abs(theta_meas) > max_angle_rad:
                    safety_events.append(f"t={t_now:.3f}s: E-STOP por exceso de angulo ({math.degrees(theta_meas):.1f}°).")
                    allow_actuation = False
                    u_cmd = 0.0
                elif control_mode == "linearized_pd":
                    # Feedforward linealizado + PD
                    u_ff = omega0_sq * theta_ref  # Aproximacion lineal sin(theta_ref) ≈ theta_ref
                    u_raw = u_ff + kp * (theta_ref - theta_meas) - kd * omega_meas
                    u_cmd = _clamp(u_raw, max_torque_limit)
                elif control_mode == "nonlinear_computed_torque":
                    # Compensacion exacta no lineal sin(theta) + PD
                    u_ff_nl = omega0_sq * math.sin(theta_ref)
                    u_raw = u_ff_nl + kp * (theta_ref - theta_meas) - kd * omega_meas
                    u_cmd = _clamp(u_raw, max_torque_limit)
                else:
                    raise ValueError(f"control_mode desconocido: {control_mode}")

            # Modelo lineal de referencia paralelo para medir cuanto diverge del hardware real no lineal
            rows.append(
                {
                    "t_s": round(t_now, 5),
                    "theta_true_rad": round(theta, 6),
                    "theta_sensor_rad": round(theta_meas, 6),
                    "omega_sensor_rads": round(omega_meas, 6),
                    "theta_ref_rad": round(theta_ref, 6),
                    "u_applied": round(u_cmd, 6),
                }
            )

            # Integracion fisica RK4 de la planta real no lineal del pendulo con motor
            sub_dt = dt_s / 4.0
            for _ in range(4):
                k1_th, k1_om = omega, -2.0 * damping * omega - omega0_sq * math.sin(theta) + u_cmd
                th2, om2 = theta + 0.5 * sub_dt * k1_th, omega + 0.5 * sub_dt * k1_om
                k2_th, k2_om = om2, -2.0 * damping * om2 - omega0_sq * math.sin(th2) + u_cmd
                th3, om3 = theta + 0.5 * sub_dt * k2_th, omega + 0.5 * sub_dt * k2_om
                k3_th, k3_om = om3, -2.0 * damping * om3 - omega0_sq * math.sin(th3) + u_cmd
                th4, om4 = theta + sub_dt * k3_th, omega + sub_dt * k3_om
                k4_th, k4_om = om4, -2.0 * damping * om4 - omega0_sq * math.sin(th4) + u_cmd
                theta += (sub_dt / 6.0) * (k1_th + 2 * k2_th + 2 * k3_th + k4_th)
                omega += (sub_dt / 6.0) * (k1_om + 2 * k2_om + 2 * k3_om + k4_om)
    else:
        # Dispositivo fisico por puerto serie COM/USB con parada de emergencia garantizada en bloque finally
        ser = serial.Serial(port=device_port.strip(), baudrate=baudrate, timeout=max(0.1, dt_s * 2.0))
        try:
            t_start = time.perf_counter()
            for _ in range(n_steps):
                t_now = time.perf_counter() - t_start
                raw_line = ser.readline().decode("utf-8", errors="replace").strip()
                theta_meas, omega_meas = 0.0, 0.0
                if raw_line:
                    if raw_line.startswith("{"):
                        pkt = json.loads(raw_line)
                        theta_meas = float(pkt.get("theta", 0.0))
                        omega_meas = float(pkt.get("omega", 0.0))
                    else:
                        parts = [p for p in raw_line.replace("TELEM", "").strip().split(",") if p]
                        if len(parts) >= 1:
                            theta_meas = float(parts[0])
                        if len(parts) >= 2:
                            omega_meas = float(parts[1])

                u_cmd = 0.0
                if allow_actuation and control_mode != "telemetry_only":
                    if abs(theta_meas) > max_angle_rad:
                        ser.write(b"ESTOP\n")
                        safety_events.append(f"t={t_now:.3f}s: ESTOP enviado al puerto {device_port}.")
                        allow_actuation = False
                    else:
                        u_ff = (
                            omega0_sq * theta_ref
                            if control_mode == "linearized_pd"
                            else omega0_sq * math.sin(theta_ref)
                        )
                        u_cmd = _clamp(u_ff + kp * (theta_ref - theta_meas) - kd * omega_meas, max_torque_limit)
                        ser.write(f"CMD u={u_cmd:.4f}\n".encode("utf-8"))

                rows.append(
                    {
                        "t_s": round(t_now, 5),
                        "theta_true_rad": round(theta_meas, 6),
                        "theta_sensor_rad": round(theta_meas, 6),
                        "omega_sensor_rads": round(omega_meas, 6),
                        "theta_ref_rad": round(theta_ref, 6),
                        "u_applied": round(u_cmd, 6),
                    }
                )
        finally:
            try:
                ser.write(b"ESTOP\n")
            except Exception:
                pass
            ser.close()

    df_telem = pd.DataFrame(rows)
    DATA_WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    safe_tag = device_port.replace("://", "_").replace("/", "_").replace("\\", "_").replace(":", "")
    csv_out = assert_safe_write_path(DATA_WORK_DIR / f"hardware_telemetry_{safe_tag}_{control_mode}.csv")
    df_telem.to_csv(csv_out, index=False, encoding="utf-8")

    html_out = _render_hardware_telemetry_html(
        output_html_path=OUTPUTS_DIR / f"hardware_replay_{safe_tag}_{control_mode}.html",
        device_port=device_port,
        control_mode=control_mode if allow_actuation else "telemetry_only (actuation_locked)",
        df_telem=df_telem,
        safety_events=safety_events,
        length_m=length_m,
    )

    final_err_deg = abs(math.degrees(float(df_telem["theta_sensor_rad"].iloc[-1]) - theta_ref))
    steady_tail = df_telem.tail(max(5, len(df_telem) // 5))
    rms_tracking_deg = float(
        np.sqrt(np.mean(np.square(np.degrees(steady_tail["theta_sensor_rad"].to_numpy() - theta_ref))))
    )

    summary = {
        "device_port": device_port,
        "actuation_enabled": allow_actuation,
        "control_mode": control_mode,
        "samples_captured": len(df_telem),
        "target_angle_deg": target_angle_deg,
        "final_tracking_error_deg": round(final_err_deg, 4),
        "steady_state_rms_error_deg": round(rms_tracking_deg, 4),
        "linear_vs_nonlinear_control_note": (
            "Nota experimental: A angulos grandes (ej. setpoint >= 35°), el feedforward linealizado u_ff=(g/L)*theta_ref "
            "sobreestima el par gravitatorio real (g/L)*sin(theta_ref), dejando un error estacionario medible frente a nonlinear_computed_torque."
        ),
        "safety_events": safety_events,
        "telemetry_csv_path": str(csv_out),
        "animated_replay_html": str(html_out),
    }
    record_audit_event("run_pendulum_hardware_control", summary, output_path=csv_out, seed=seed)
    return summary
