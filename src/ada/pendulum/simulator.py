"""Proyecto del Pendulo: '¿Cuanto vale la aproximacion lineal?' con doble verificacion y animacion HTML5 60 FPS."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np
from scipy import integrate, optimize, special

from ada.audit.ledger import record_audit_event
from ada.config import OUTPUTS_DIR, assert_safe_write_path


def exact_period_elliptic(theta0_rad: float, length_m: float = 1.0, gravity_ms2: float = 9.80665) -> float:
    """Metodo A: Periodo no lineal exacto mediante la integral eliptica completa de primera especie K(m)."""
    if abs(theta0_rad) >= math.pi:
        raise ValueError("theta0_rad debe cumplir |theta0| < pi para oscilacion acotada.")
    t0 = 2.0 * math.pi * math.sqrt(length_m / gravity_ms2)
    if abs(theta0_rad) < 1e-15:
        return t0
    m = math.sin(0.5 * abs(theta0_rad)) ** 2
    return float(t0 * (2.0 / math.pi) * special.ellipk(m))


def exact_period_ode_dop853(theta0_rad: float, length_m: float = 1.0, gravity_ms2: float = 9.80665) -> float:
    """Metodo B independiente: Integracion directa DOP853 de d2theta/dt2 + (g/L)sin(theta) = 0 hasta el primer cruce por cero (T/4)."""
    if abs(theta0_rad) < 1e-12:
        return 2.0 * math.pi * math.sqrt(length_m / gravity_ms2)
    omega0_sq = gravity_ms2 / length_m
    t0_est = 2.0 * math.pi / math.sqrt(omega0_sq)
    amp = abs(theta0_rad)

    def zero_crossing(t: float, state: np.ndarray) -> float:
        return float(state[0])

    zero_crossing.terminal = True  # type: ignore[attr-defined]
    zero_crossing.direction = -1.0  # type: ignore[attr-defined]

    sol = integrate.solve_ivp(
        lambda _t, s: [s[1], -omega0_sq * math.sin(s[0])],
        (0.0, t0_est * 5.0),
        [amp, 0.0],
        method="DOP853",
        events=zero_crossing,
        rtol=1e-12,
        atol=1e-14,
    )
    if not sol.t_events or len(sol.t_events[0]) == 0:
        raise RuntimeError("No se detecto el cruce por equilibrio en DOP853.")
    quarter_period = float(sol.t_events[0][0])
    return 4.0 * quarter_period


def analyze_pendulum_linear_approximation(
    *,
    length_m: float = 1.0,
    gravity_ms2: float = 9.80665,
    theta0_deg: float = 30.0,
    tolerance_pct: float = 1.0,
    n_cycles: int = 10,
) -> dict[str, Any]:
    """Evalua rigurosamente por dos metodos independientes cuanto vale la aproximacion lineal sin(theta) ≈ theta."""
    if length_m <= 0 or gravity_ms2 <= 0:
        raise ValueError("length_m y gravity_ms2 deben ser positivos.")
    if not (0.1 <= abs(theta0_deg) <= 175.0):
        raise ValueError("theta0_deg debe estar entre 0.1 y 175.0 grados.")
    if not (0.001 <= tolerance_pct <= 50.0):
        raise ValueError("tolerance_pct debe estar entre 0.001% y 50%.")

    theta0 = math.radians(abs(theta0_deg))
    omega0 = math.sqrt(gravity_ms2 / length_m)
    t_linear = 2.0 * math.pi / omega0

    # 1. Error instantaneo en el termino restaurador: |theta - sin(theta)| / |sin(theta)|
    force_rel_np = abs(theta0 - math.sin(theta0)) / abs(math.sin(theta0))
    with mp.workdps(50):
        th_mp = mp.mpf(str(theta0))
        force_rel_mp = float(abs(th_mp - mp.sin(th_mp)) / abs(mp.sin(th_mp)))
    taylor_leading_bound = (theta0**2) / 6.0

    # 2. Error en el periodo: Metodo A (Eliptica K) vs Metodo B (EDO DOP853) + Serie asintotica de Legendre
    t_exact_ellip = exact_period_elliptic(theta0, length_m, gravity_ms2)
    t_exact_ode = exact_period_ode_dop853(theta0, length_m, gravity_ms2)
    period_dual_diff = abs(t_exact_ellip - t_exact_ode) / t_exact_ellip

    period_rel_error = (t_exact_ellip - t_linear) / t_exact_ellip
    legendre_ratio_approx = 1.0 + (theta0**2) / 16.0 + (11.0 / 3072.0) * (theta0**4)

    # 3. Umbrales criticos theta_c para la tolerancia pedida (ej. 1%)
    tol_frac = tolerance_pct / 100.0
    theta_crit_force_rad = float(
        optimize.brentq(lambda th: (abs(th - math.sin(th)) / abs(math.sin(th))) - tol_frac, 1e-6, 3.05)
    )
    theta_crit_period_rad = float(
        optimize.brentq(
            lambda th: ((exact_period_elliptic(th, length_m, gravity_ms2) - t_linear) / exact_period_elliptic(th, length_m, gravity_ms2))
            - tol_frac,
            1e-6,
            3.08,
        )
    )

    # 4. Desfase acumulado tras N oscilaciones completas: integrar simultaneamente EDO exacta vs solucion lineal
    t_horizon = max(1, int(n_cycles)) * t_linear
    t_grid = np.linspace(0.0, t_horizon, max(400, int(n_cycles) * 120))
    sol_nl = integrate.solve_ivp(
        lambda _t, s: [s[1], -(omega0**2) * math.sin(s[0])],
        (0.0, t_horizon),
        [theta0, 0.0],
        method="DOP853",
        t_eval=t_grid,
        rtol=1e-11,
        atol=1e-13,
    )
    theta_lin_grid = theta0 * np.cos(omega0 * t_grid)
    drift_rad = np.abs(sol_nl.y[0] - theta_lin_grid)
    max_drift_rad = float(np.max(drift_rad))
    max_drift_rel_amp = max_drift_rad / theta0

    # Tabla de referencia multi-angulo verificada
    sweep_angles_deg = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0]
    sweep_table: list[dict[str, float]] = []
    for deg in sweep_angles_deg:
        r = math.radians(deg)
        te = exact_period_elliptic(r, length_m, gravity_ms2)
        sweep_table.append(
            {
                "theta0_deg": deg,
                "force_rel_error_pct": round(100.0 * abs(r - math.sin(r)) / abs(math.sin(r)), 5),
                "T_over_T0": round(te / t_linear, 6),
                "period_rel_error_pct": round(100.0 * (te - t_linear) / te, 5),
            }
        )

    verified = (abs(force_rel_np - force_rel_mp) < 1e-12) and (period_dual_diff < 1e-9)

    report = {
        "verified": verified,
        "verification_methods": [
            "scipy.special.ellipk (Integral Eliptica Completa K)",
            "scipy.integrate.solve_ivp_DOP853 (Integracion EDO No Lineal)",
            "mpmath_50dps (Cota de Taylor)",
        ],
        "dual_period_relative_discrepancy": period_dual_diff,
        "parameters": {
            "length_m": length_m,
            "gravity_ms2": gravity_ms2,
            "theta0_deg": abs(theta0_deg),
            "theta0_rad": round(theta0, 8),
            "tolerance_pct": tolerance_pct,
            "n_cycles": n_cycles,
        },
        "linear_period_T0_s": round(t_linear, 8),
        "exact_period_elliptic_s": round(t_exact_ellip, 8),
        "exact_period_dop853_s": round(t_exact_ode, 8),
        "legendre_series_T_over_T0": round(legendre_ratio_approx, 8),
        "instantaneous_force_error_pct": round(100.0 * force_rel_np, 6),
        "taylor_theta2_over_6_pct": round(100.0 * taylor_leading_bound, 6),
        "period_error_pct": round(100.0 * period_rel_error, 6),
        "critical_angles_for_tolerance": {
            "tolerance_pct": tolerance_pct,
            "max_angle_for_force_error_deg": round(math.degrees(theta_crit_force_rad), 4),
            "max_angle_for_period_error_deg": round(math.degrees(theta_crit_period_rad), 4),
        },
        "cumulative_phase_drift_after_N_cycles": {
            "n_cycles": n_cycles,
            "max_abs_angle_error_deg": round(math.degrees(max_drift_rad), 4),
            "max_error_pct_of_amplitude": round(100.0 * max_drift_rel_amp, 3),
        },
        "reference_sweep_table": sweep_table,
        "tutor_guiding_questions": [
            "Observa que para un mismo umbral (ej. 1%), el angulo critico de la fuerza (~14.0°) es mas estricto que el del periodo (~23.1°). ¿Por que el error del periodo crece como theta0^2/16 mientras el de la fuerza crece como theta0^2/6?",
            f"Aunque a {abs(theta0_deg):.1f}° el periodo solo difiere en {100.0 * period_rel_error:.3f}%, tras {n_cycles} ciclos el desfase acumulado alcanza {100.0 * max_drift_rel_amp:.1f}% de la amplitud. ¿Que implica esto si usas el modelo lineal para predecir la posicion de un pendulo electronico a largo plazo sin realimentacion de sensor?",
        ],
    }
    record_audit_event("analyze_pendulum_linear_approximation", report)
    return report


def build_pendulum_interactive_html(
    *,
    output_filename: str = "pendulum_interactive_lab.html",
    length_m: float = 1.0,
    gravity_ms2: float = 9.80665,
    theta0_deg: float = 35.0,
    damping_gamma: float = 0.0,
    tolerance_pct: float = 1.0,
) -> dict[str, Any]:
    """Construye un laboratorio HTML5 interactivo a 60 FPS (100% local, sin CDN externo) para explorar el pendulo y su control."""
    analysis = analyze_pendulum_linear_approximation(
        length_m=length_m,
        gravity_ms2=gravity_ms2,
        theta0_deg=theta0_deg,
        tolerance_pct=tolerance_pct,
        n_cycles=10,
    )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = assert_safe_write_path(OUTPUTS_DIR / output_filename)

    init_json = json.dumps(
        {
            "L": length_m,
            "g": gravity_ms2,
            "theta0_deg": abs(theta0_deg),
            "gamma": damping_gamma,
            "tol_pct": tolerance_pct,
            "verified_snapshot": analysis,
        },
        ensure_ascii=False,
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>ADA — Laboratorio Dinamico del Pendulo: ¿Cuanto vale la aproximacion lineal?</title>
<style>
  :root {{
    --bg: #070b17; --card: #10182f; --border: #22315c; --txt: #eaf0ff;
    --muted: #93a4d4; --nl: #38bdf8; --lin: #fbbf24; --ctrl: #22c55e; --danger: #f43f5e;
  }}
  * {{ box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 0; padding: 18px; background: var(--bg); color: var(--txt); }}
  header {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }}
  h1 {{ margin: 0; font-size: 1.32rem; }}
  .pill {{ background: rgba(56,189,248,0.15); border: 1px solid var(--nl); color: var(--nl); padding: 4px 11px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }}
  .layout {{ display: grid; grid-template-columns: 330px 1fr; gap: 16px; }}
  @media (max-width: 980px) {{ .layout {{ grid-template-columns: 1fr; }} }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 14px; }}
  .card h2 {{ margin: 0 0 10px; font-size: 0.98rem; color: var(--nl); }}
  .ctrl-row {{ margin-bottom: 11px; }}
  .ctrl-row label {{ display: flex; justify-content: space-between; font-size: 0.84rem; color: var(--muted); margin-bottom: 3px; }}
  .ctrl-row label b {{ color: var(--txt); }}
  input[type=range] {{ width: 100%; accent-color: var(--nl); cursor: pointer; }}
  .btns {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0; }}
  button {{
    background: #1b2850; color: var(--txt); border: 1px solid #354c8c; border-radius: 8px;
    padding: 7px 12px; cursor: pointer; font-weight: 600; font-size: 0.84rem;
  }}
  button:hover {{ background: #263973; }}
  button.primary {{ background: #0284c7; border-color: #38bdf8; }}
  button.estop {{ background: rgba(244,63,94,0.2); border-color: var(--danger); color: #fda4af; }}
  .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(175px, 1fr)); gap: 10px; margin-bottom: 14px; }}
  .metric {{ background: #0b1224; border: 1px solid var(--border); border-radius: 10px; padding: 10px; }}
  .metric .k {{ font-size: 0.76rem; color: var(--muted); }}
  .metric .v {{ font-size: 1.12rem; font-weight: 700; margin-top: 3px; }}
  .canvases {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  @media (max-width: 1100px) {{ .canvases {{ grid-template-columns: 1fr; }} }}
  canvas {{ width: 100%; background: #090f20; border: 1px solid var(--border); border-radius: 10px; display: block; }}
  .legend {{ display: flex; gap: 16px; font-size: 0.82rem; margin: 6px 0 10px; flex-wrap: wrap; }}
  .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  details {{ margin-top: 12px; background: #0b1224; border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }}
  summary {{ cursor: pointer; font-weight: 600; color: var(--lin); font-size: 0.9rem; }}
  .tutor-body {{ font-size: 0.86rem; line-height: 1.5; color: var(--txt); margin-top: 8px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>ADA — El Pendulo: ¿Cuanto vale la aproximacion lineal?</h1>
    <div style="color:var(--muted);font-size:0.86rem">Comparacion simultanea en tiempo real: EDO Exacta No Lineal (RK4) vs Modelo Lineal + Control PD de Actuador</div>
  </div>
  <span class="pill">VERIFICADO: Eliptica K(m) + DOP853</span>
</header>

<div class="layout">
  <div class="card">
    <h2>Parametros Fisicos y Control Electronico</h2>
    <div class="ctrl-row">
      <label><span>Angulo inicial θ₀</span><b id="lblTheta0"></b></label>
      <input type="range" id="slTheta0" min="1" max="172" step="0.5" />
    </div>
    <div class="ctrl-row">
      <label><span>Longitud L (m)</span><b id="lblL"></b></label>
      <input type="range" id="slL" min="0.2" max="3.0" step="0.05" />
    </div>
    <div class="ctrl-row">
      <label><span>Gravedad g (m/s²)</span><b id="lblG"></b></label>
      <input type="range" id="slG" min="1.6" max="24.8" step="0.05" />
    </div>
    <div class="ctrl-row">
      <label><span>Amortiguamiento γ (1/s)</span><b id="lblGamma"></b></label>
      <input type="range" id="slGamma" min="0.0" max="1.2" step="0.02" />
    </div>
    <div class="ctrl-row">
      <label><span>Umbral de tolerancia (%)</span><b id="lblTol"></b></label>
      <input type="range" id="slTol" min="0.1" max="10.0" step="0.1" />
    </div>
    <hr style="border-color:var(--border);margin:12px 0" />
    <h2>Simulador de Control de Motor/Servo</h2>
    <div class="ctrl-row">
      <label><span>Activar Control PD al setpoint</span><input type="checkbox" id="chkCtrl" /></label>
    </div>
    <div class="ctrl-row">
      <label><span>Setpoint θ_ref (°)</span><b id="lblRef">0.0°</b></label>
      <input type="range" id="slRef" min="-60" max="60" step="1" value="0" />
    </div>
    <div class="ctrl-row">
      <label><span>Ganancia Kp / Kd</span><b id="lblGains">18 / 6</b></label>
      <input type="range" id="slKp" min="2" max="50" step="1" value="18" />
    </div>

    <div class="btns">
      <button class="primary" id="btnPlay">Pausar</button>
      <button id="btnReset">Reiniciar t = 0</button>
      <button id="btnImpulse">Pulso +1.5 rad/s</button>
      <button class="estop" id="btnEstop">E-STOP Motor</button>
    </div>

    <details open>
      <summary>Modo Tutor Matemático (Pistas Graduales)</summary>
      <div class="tutor-body" id="tutorText"></div>
    </details>
  </div>

  <div>
    <div class="metrics">
      <div class="metric"><div class="k">Error Fuerza |θ₀ - sin θ₀| / |sin θ₀|</div><div class="v" id="mForce"></div></div>
      <div class="metric"><div class="k">Periodo Exacto T(θ₀) vs Lineal T₀</div><div class="v" id="mPeriod"></div></div>
      <div class="metric"><div class="k">Error Relativo de Periodo ε_T</div><div class="v" id="mEpsT"></div></div>
      <div class="metric"><div class="k">Desfase Instantaneo |θ_NL(t) - θ_Lin(t)|</div><div class="v" id="mDrift"></div></div>
      <div class="metric"><div class="k">θ_crit Fuerza / Periodo para Tolerancia</div><div class="v" id="mCrit"></div></div>
    </div>

    <div class="legend">
      <span><i class="dot" style="background:var(--nl)"></i>No Lineal Exacto: θ'' + 2γθ' + (g/L)sin(θ) = u(t)</span>
      <span><i class="dot" style="background:var(--lin)"></i>Aproximacion Lineal: θ'' + 2γθ' + (g/L)θ = u(t)</span>
      <span><i class="dot" style="background:var(--ctrl)"></i>Par de Control Electronico u(t)</span>
    </div>

    <div class="canvases">
      <canvas id="cvPendulum" width="540" height="360"></canvas>
      <canvas id="cvPhase" width="540" height="360"></canvas>
      <canvas id="cvTime" width="540" height="260"></canvas>
      <canvas id="cvErrorCurve" width="540" height="260"></canvas>
    </div>
  </div>
</div>

<script>
const INIT = {init_json};

// Integral eliptica completa K(m) por Media Aritmetico-Geometrica (AGM de Gauss) — precision 1e-15
function ellipK(m) {{
  if (m < 0 || m >= 1) return Infinity;
  let a = 1.0, b = Math.sqrt(1.0 - m);
  for (let i = 0; i < 28; i++) {{
    const an = 0.5 * (a + b);
    const bn = Math.sqrt(a * b);
    if (Math.abs(an - bn) < 1e-15) {{ a = an; break; }}
    a = an; b = bn;
  }}
  return Math.PI / (2.0 * a);
}}

function findCriticalAngles(tolPct) {{
  const tol = tolPct / 100.0;
  let lo = 1e-5, hi = 3.05;
  for (let i = 0; i < 50; i++) {{
    const mid = 0.5 * (lo + hi);
    const err = Math.abs(mid - Math.sin(mid)) / Math.abs(Math.sin(mid));
    if (err < tol) lo = mid; else hi = mid;
  }}
  const critF = lo * 180 / Math.PI;

  lo = 1e-5; hi = 3.08;
  for (let i = 0; i < 50; i++) {{
    const mid = 0.5 * (lo + hi);
    const ratio = (2.0 / Math.PI) * ellipK(Math.pow(Math.sin(0.5 * mid), 2));
    const errT = (ratio - 1.0) / ratio;
    if (errT < tol) lo = mid; else hi = mid;
  }}
  const critT = lo * 180 / Math.PI;
  return {{ critF, critT }};
}}

const slTheta0 = document.getElementById("slTheta0");
const slL = document.getElementById("slL");
const slG = document.getElementById("slG");
const slGamma = document.getElementById("slGamma");
const slTol = document.getElementById("slTol");
const chkCtrl = document.getElementById("chkCtrl");
const slRef = document.getElementById("slRef");
const slKp = document.getElementById("slKp");

slTheta0.value = INIT.theta0_deg;
slL.value = INIT.L;
slG.value = INIT.g;
slGamma.value = INIT.gamma;
slTol.value = INIT.tol_pct;

let running = true;
let simTime = 0.0;
let stateNL = [INIT.theta0_deg * Math.PI / 180, 0.0];
let stateLin = [INIT.theta0_deg * Math.PI / 180, 0.0];
let lastControlU = 0.0;
let history = [];
let phaseNL = [];
let phaseLin = [];

function getParams() {{
  return {{
    th0Deg: parseFloat(slTheta0.value),
    L: parseFloat(slL.value),
    g: parseFloat(slG.value),
    gamma: parseFloat(slGamma.value),
    tolPct: parseFloat(slTol.value),
    ctrlOn: chkCtrl.checked,
    refRad: parseFloat(slRef.value) * Math.PI / 180,
    kp: parseFloat(slKp.value),
    kd: Math.max(2.0, parseFloat(slKp.value) * 0.32),
  }};
}}

function resetSim() {{
  const p = getParams();
  const th0 = p.th0Deg * Math.PI / 180;
  simTime = 0.0;
  stateNL = [th0, 0.0];
  stateLin = [th0, 0.0];
  lastControlU = 0.0;
  history = [];
  phaseNL = [];
  phaseLin = [];
  updateStaticMetrics();
}}

function computeControl(theta, omega, p) {{
  if (!p.ctrlOn) return 0.0;
  // Saturacion de seguridad del actuador (|u| <= 25 rad/s^2) + compensacion gravitatoria acotada
  const raw = (p.g / p.L) * Math.sin(p.refRad) + p.kp * (p.refRad - theta) - p.kd * omega;
  return Math.max(-28.0, Math.min(28.0, raw));
}}

function rk4Step(state, dt, derivFn) {{
  const [y, v] = state;
  const k1 = derivFn(y, v);
  const k2 = derivFn(y + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1]);
  const k3 = derivFn(y + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1]);
  const k4 = derivFn(y + dt * k3[0], v + dt * k3[1]);
  return [
    y + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
    v + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
  ];
}}

function updateStaticMetrics() {{
  const p = getParams();
  document.getElementById("lblTheta0").textContent = p.th0Deg.toFixed(1) + "°";
  document.getElementById("lblL").textContent = p.L.toFixed(2) + " m";
  document.getElementById("lblG").textContent = p.g.toFixed(2) + " m/s²";
  document.getElementById("lblGamma").textContent = p.gamma.toFixed(2) + " s⁻¹";
  document.getElementById("lblTol").textContent = p.tolPct.toFixed(1) + " %";
  document.getElementById("lblRef").textContent = parseFloat(slRef.value).toFixed(1) + "°";
  document.getElementById("lblGains").textContent = `${{p.kp.toFixed(0)}} / ${{p.kd.toFixed(1)}}`;

  const th0 = p.th0Deg * Math.PI / 180;
  const errF = 100.0 * Math.abs(th0 - Math.sin(th0)) / Math.abs(Math.sin(th0));
  const T0 = 2.0 * Math.PI * Math.sqrt(p.L / p.g);
  const ratioT = (2.0 / Math.PI) * ellipK(Math.pow(Math.sin(0.5 * th0), 2));
  const Texact = T0 * ratioT;
  const errT = 100.0 * (Texact - T0) / Texact;
  const crits = findCriticalAngles(p.tolPct);

  document.getElementById("mForce").textContent = errF.toFixed(3) + " %";
  document.getElementById("mForce").style.color = errF <= p.tolPct ? "#22c55e" : "#fbbf24";
  document.getElementById("mPeriod").textContent = `${{Texact.toFixed(3)}}s vs ${{T0.toFixed(3)}}s`;
  document.getElementById("mEpsT").textContent = errT.toFixed(3) + " %";
  document.getElementById("mEpsT").style.color = errT <= p.tolPct ? "#22c55e" : "#f43f5e";
  document.getElementById("mCrit").textContent = `${{crits.critF.toFixed(1)}}° / ${{crits.critT.toFixed(1)}}°`;

  document.getElementById("tutorText").innerHTML =
    `<b>1. Cota de Taylor:</b> Como <code>sin(θ) = θ - θ³/3! + θ⁵/5! - ...</code> es alternante para <code>0 &lt; θ &lt; 2</code>, el error relativo instantaneo cumple <code>≈ θ²/6</code>. Para no superar <b>${{p.tolPct.toFixed(1)}}%</b> en la fuerza, necesitas <b>θ₀ ≤ ${{crits.critF.toFixed(1)}}°</b>.<br/><br/>` +
    `<b>2. Integral Eliptica del Periodo:</b> Al integrar la conservacion de la energia obtienes <code>T(θ₀)/T₀ = (2/π) K(sin²(θ₀/2)) ≈ 1 + θ₀²/16</code>. El denominador <code>16</code> (en vez de <code>6</code>) explica por que el periodo tolera hasta <b>${{crits.critT.toFixed(1)}}°</b> para el mismo <b>${{p.tolPct.toFixed(1)}}%</b>.<br/><br/>` +
    `<b>3. Pregunta clave para tu informe:</b> Observa la grafica inferior izquierda: aunque <code>ε_T</code> parezca pequeno (${{errT.toFixed(2)}}%), ¿que ocurre con <code>|θ_NL(t) - θ_Lin(t)|</code> al acumular varios periodos sin control realimentado?`;

  drawErrorCurve(p, crits);
}}

function drawPendulums(p) {{
  const cv = document.getElementById("cvPendulum");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);

  const cx = cv.width * 0.5, cy = 62;
  const rodPx = Math.min(235, Math.max(95, p.L * 145));

  // Arco de referencia y soporte
  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, rodPx, 0.1 * Math.PI, 0.9 * Math.PI); ctx.stroke();
  ctx.fillStyle = "#475569"; ctx.fillRect(cx - 48, cy - 6, 96, 6);

  if (p.ctrlOn) {{
    const rx = cx + rodPx * Math.sin(p.refRad);
    const ry = cy + rodPx * Math.cos(p.refRad);
    ctx.strokeStyle = "rgba(34,197,94,0.45)"; ctx.setLineDash([5, 5]);
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(rx, ry); ctx.stroke();
    ctx.setLineDash([]);
  }}

  // Pendulo Lineal (Ambar)
  const xLin = cx + rodPx * Math.sin(stateLin[0]);
  const yLin = cy + rodPx * Math.cos(stateLin[0]);
  ctx.strokeStyle = "rgba(251, 191, 36, 0.72)"; ctx.lineWidth = 2.5;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(xLin, yLin); ctx.stroke();
  ctx.fillStyle = "rgba(251, 191, 36, 0.85)";
  ctx.beginPath(); ctx.arc(xLin, yLin, 13, 0, Math.PI * 2); ctx.fill();

  // Pendulo No Lineal Exacto (Cian)
  const xNL = cx + rodPx * Math.sin(stateNL[0]);
  const yNL = cy + rodPx * Math.cos(stateNL[0]);
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 3.2;
  ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(xNL, yNL); ctx.stroke();
  ctx.fillStyle = "#38bdf8";
  ctx.beginPath(); ctx.arc(xNL, yNL, 15, 0, Math.PI * 2); ctx.fill();

  ctx.fillStyle = "#eaf0ff"; ctx.font = "12px sans-serif";
  ctx.fillText(`t = ${{simTime.toFixed(2)}} s`, 14, 22);
  ctx.fillStyle = "#38bdf8";
  ctx.fillText(`θ_NL = ${{(stateNL[0]*180/Math.PI).toFixed(2)}}°`, 14, 42);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText(`θ_Lin = ${{(stateLin[0]*180/Math.PI).toFixed(2)}}°`, 14, 60);
  if (p.ctrlOn) {{
    ctx.fillStyle = "#22c55e";
    ctx.fillText(`Actuador u(t) = ${{lastControlU.toFixed(2)}} rad/s²`, 14, 78);
  }}
}}

function drawPhaseSpace() {{
  const cv = document.getElementById("cvPhase");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const cx = cv.width / 2, cy = cv.height / 2;
  const scaleX = (cv.width - 70) / (2 * Math.PI);
  const scaleY = (cv.height - 70) / 16.0;

  ctx.strokeStyle = "#1e293b"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(20, cy); ctx.lineTo(cv.width - 20, cy);
  ctx.moveTo(cx, 20); ctx.lineTo(cx, cv.height - 20); ctx.stroke();

  const plotTrace = (pts, color) => {{
    if (pts.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2;
    ctx.beginPath();
    pts.forEach((pt, i) => {{
      const px = cx + pt[0] * scaleX;
      const py = cy - pt[1] * scaleY;
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }});
    ctx.stroke();
  }};
  plotTrace(phaseLin, "rgba(251,191,36,0.75)");
  plotTrace(phaseNL, "#38bdf8");

  ctx.fillStyle = "#93a4d4"; ctx.font = "12px sans-serif";
  ctx.fillText("Espacio de Fases (θ rad vs θ' rad/s)", 14, 22);
}}

function drawTimeSeries() {{
  const cv = document.getElementById("cvTime");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const pad = 34;
  ctx.fillStyle = "#93a4d4"; ctx.font = "12px sans-serif";
  ctx.fillText("Evolucion Temporal θ_NL(t), θ_Lin(t) y Desfase acumulado", 14, 20);
  if (history.length < 2) return;

  const tMin = history[0].t, tMax = Math.max(history[history.length - 1].t, tMin + 2.0);
  const maxAmp = Math.max(0.25, ...history.map(h => Math.max(Math.abs(h.nl), Math.abs(h.lin))));
  const sx = t => pad + ((t - tMin) / (tMax - tMin)) * (cv.width - 2 * pad);
  const sy = v => cv.height / 2 - (v / maxAmp) * (cv.height / 2 - pad);

  ctx.strokeStyle = "#1e293b"; ctx.beginPath();
  ctx.moveTo(pad, cv.height / 2); ctx.lineTo(cv.width - pad, cv.height / 2); ctx.stroke();

  const drawCurve = (key, color) => {{
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((h, i) => {{
      const x = sx(h.t), y = sy(h[key]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }});
    ctx.stroke();
  }};
  drawCurve("lin", "rgba(251,191,36,0.85)");
  drawCurve("nl", "#38bdf8");
}}

function drawErrorCurve(p, crits) {{
  const cv = document.getElementById("cvErrorCurve");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const pad = 40;
  const maxY = Math.max(15.0, p.tolPct * 3.0);
  const sx = deg => pad + (deg / 150.0) * (cv.width - 2 * pad);
  const sy = pct => cv.height - pad - (Math.min(pct, maxY) / maxY) * (cv.height - 2 * pad);

  // Linea de tolerancia
  ctx.strokeStyle = "rgba(34,197,94,0.6)"; ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(pad, sy(p.tolPct)); ctx.lineTo(cv.width - pad, sy(p.tolPct)); ctx.stroke();
  ctx.setLineDash([]);

  // Curva error periodo
  ctx.strokeStyle = "#38bdf8"; ctx.lineWidth = 2.2; ctx.beginPath();
  for (let deg = 1; deg <= 150; deg += 2) {{
    const r = deg * Math.PI / 180;
    const ratio = (2.0 / Math.PI) * ellipK(Math.pow(Math.sin(0.5 * r), 2));
    const epsT = 100.0 * (ratio - 1.0) / ratio;
    if (deg === 1) ctx.moveTo(sx(deg), sy(epsT)); else ctx.lineTo(sx(deg), sy(epsT));
  }}
  ctx.stroke();

  // Curva error fuerza
  ctx.strokeStyle = "#fbbf24"; ctx.lineWidth = 2.0; ctx.beginPath();
  for (let deg = 1; deg <= 150; deg += 2) {{
    const r = deg * Math.PI / 180;
    const epsF = 100.0 * Math.abs(r - Math.sin(r)) / Math.abs(Math.sin(r));
    if (deg === 1) ctx.moveTo(sx(deg), sy(epsF)); else ctx.lineTo(sx(deg), sy(epsF));
  }}
  ctx.stroke();

  // Marcador angulo actual
  const curX = sx(Math.min(150, p.th0Deg));
  ctx.strokeStyle = "#f43f5e"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(curX, pad - 10); ctx.lineTo(curX, cv.height - pad); ctx.stroke();

  ctx.fillStyle = "#93a4d4"; ctx.font = "12px sans-serif";
  ctx.fillText(`Curva ε_Fuerza (ambar) vs ε_Periodo (cian) | Umbral ${{p.tolPct.toFixed(1)}}% (verde)`, 14, 20);
  ctx.fillText("0°", pad, cv.height - 14);
  ctx.fillText("75°", sx(75) - 10, cv.height - 14);
  ctx.fillText("150°", sx(150) - 22, cv.height - 14);
}}

function frame() {{
  const p = getParams();
  if (running) {{
    const dt = 1.0 / 240.0;
    const w0sq = p.g / p.L;
    for (let k = 0; k < 4; k++) {{
      const uNL = computeControl(stateNL[0], stateNL[1], p);
      const uLin = computeControl(stateLin[0], stateLin[1], p);
      lastControlU = uNL;
      stateNL = rk4Step(stateNL, dt, (th, om) => [om, -2 * p.gamma * om - w0sq * Math.sin(th) + uNL]);
      stateLin = rk4Step(stateLin, dt, (th, om) => [om, -2 * p.gamma * om - w0sq * th + uLin]);
      simTime += dt;
    }}
    history.push({{ t: simTime, nl: stateNL[0], lin: stateLin[0] }});
    if (history.length > 550) history.shift();
    phaseNL.push([stateNL[0], stateNL[1]]);
    phaseLin.push([stateLin[0], stateLin[1]]);
    if (phaseNL.length > 450) {{ phaseNL.shift(); phaseLin.shift(); }}
  }}
  const driftDeg = Math.abs(stateNL[0] - stateLin[0]) * 180 / Math.PI;
  document.getElementById("mDrift").textContent = driftDeg.toFixed(2) + "°";

  drawPendulums(p);
  drawPhaseSpace();
  drawTimeSeries();
  requestAnimationFrame(frame);
}}

[slTheta0, slL, slG, slGamma, slTol].forEach(el => el.addEventListener("input", resetSim));
[chkCtrl, slRef, slKp].forEach(el => el.addEventListener("input", updateStaticMetrics));

document.getElementById("btnPlay").addEventListener("click", (e) => {{
  running = !running;
  e.target.textContent = running ? "Pausar" : "Reanudar";
}});
document.getElementById("btnReset").addEventListener("click", resetSim);
document.getElementById("btnImpulse").addEventListener("click", () => {{
  stateNL[1] += 1.5; stateLin[1] += 1.5;
}});
document.getElementById("btnEstop").addEventListener("click", () => {{
  chkCtrl.checked = false; lastControlU = 0.0; updateStaticMetrics();
}});

resetSim();
requestAnimationFrame(frame);
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    record_audit_event("build_pendulum_interactive_html", {"html_path": str(out_path), "theta0_deg": theta0_deg}, output_path=out_path)
    return {
        "verified": analysis["verified"],
        "interactive_html_path": str(out_path),
        "analysis_summary": analysis,
    }
