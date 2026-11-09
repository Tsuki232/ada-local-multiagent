"""Motor de Vision por Computadora, Seguimiento Fisico y Calculos en Tiempo Real de ADA."""

from __future__ import annotations

import collections
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from scipy import fft, optimize

from ada.calculus.verified_math import solve_and_verify_ode


@dataclass
class TrackingPoint:
    timestamp: float
    x: float
    y: float
    theta: float
    omega: float = 0.0
    alpha: float = 0.0


@dataclass
class KinematicsState:
    timestamp: float
    theta_rad: float
    theta_deg: float
    omega_rad_s: float
    alpha_rad_s2: float
    period_s: float
    frequency_hz: float
    estimated_length_m: float
    damping_factor: float
    energy_kinetic: float
    energy_potential: float
    energy_total: float
    verified: bool = False
    verification_methods: list[str] = field(default_factory=list)
    discrepancy: float = 0.0


class RealTimePhysicsTracker:
    """Rastreador cinematico y analizador fisico en tiempo real sobre frames de video o camara ADB."""

    def __init__(
        self,
        history_len: int = 300,
        pivot_x: float | None = None,
        pivot_y: float | None = None,
        pixels_per_meter: float = 400.0,
        pendulum_mass_kg: float = 0.25,
    ) -> None:
        self.history_len = history_len
        self.pivot_x = pivot_x
        self.pivot_y = pivot_y
        self.pixels_per_meter = pixels_per_meter
        self.mass = pendulum_mass_kg
        self.g = 9.80665

        self.history: collections.deque[TrackingPoint] = collections.deque(maxlen=history_len)
        self.last_kinematics: KinematicsState | None = None
        self._last_verified_ode_result: dict[str, Any] | None = None
        self._last_verification_time: float = 0.0

    def set_pivot(self, x: float, y: float) -> None:
        self.pivot_x = x
        self.pivot_y = y

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        target_hsv_lower: tuple[int, int, int] = (5, 120, 120),
        target_hsv_upper: tuple[int, int, int] = (25, 255, 255),
        auto_detect_target: bool = True,
    ) -> tuple[np.ndarray, KinematicsState | None]:
        """Procesa un frame: detecta el cuerpo en movimiento, calcula la cinematica y dibuja la telemetria."""
        now = time.time()
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()

        # Establecer pivote por defecto en el centro superior si no se ha calibrado
        if self.pivot_x is None or self.pivot_y is None:
            self.pivot_x = w / 2.0
            self.pivot_y = h * 0.15

        # Deteccion de posicion (x, y)
        detected_pos = None

        # 1. Intento por mascara de color HSV (por defecto naranja/rojo vibrante o marcador brillante)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(target_hsv_lower), np.array(target_hsv_upper))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 50:
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    detected_pos = (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))

        # 2. Fallback a deteccion de contraste / contorno mas dinamico si auto_detect esta activo
        if detected_pos is None and auto_detect_target:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (9, 9), 0)
            _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
            gen_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid = [c for c in gen_contours if 100 < cv2.contourArea(c) < (w * h * 0.1)]
            if valid:
                c = max(valid, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] > 0:
                    detected_pos = (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))

        kinematics = None
        if detected_pos:
            px, py = detected_pos
            dx = px - self.pivot_x
            dy = py - self.pivot_y
            theta = float(np.arctan2(dx, dy))

            omega = 0.0
            alpha = 0.0
            if len(self.history) >= 2:
                prev = self.history[-1]
                dt = max(1e-4, now - prev.timestamp)
                omega = float((theta - prev.theta) / dt)
                alpha = float((omega - prev.omega) / dt)

            pt = TrackingPoint(timestamp=now, x=px, y=py, theta=theta, omega=omega, alpha=alpha)
            self.history.append(pt)

            kinematics = self._compute_kinematics(now)
            self.last_kinematics = kinematics

        self._draw_hud_telemetry(annotated, kinematics)
        return annotated, kinematics

    def _compute_kinematics(self, current_time: float) -> KinematicsState:
        """Calcula parametros fisicos rigurosos mediante diferencias finitas, FFT y EDO no lineal."""
        pts = list(self.history)
        cur = pts[-1]
        theta = cur.theta
        omega = cur.omega
        alpha = cur.alpha

        # Estimacion de frecuencia y longitud mediante FFT
        freq_hz = 0.0
        period_s = 0.0
        est_length = 0.5
        damping = 0.05

        if len(pts) >= 40:
            times = np.array([p.timestamp for p in pts])
            thetas = np.array([p.theta for p in pts])
            dt_mean = float(np.mean(np.diff(times)))

            if dt_mean > 0:
                centered = thetas - np.mean(thetas)
                fft_vals = np.abs(fft.rfft(centered))
                fft_freqs = fft.rfftfreq(len(centered), d=dt_mean)

                # Ignorar componente continua (0 Hz)
                if len(fft_vals) > 1:
                    peak_idx = int(np.argmax(fft_vals[1:])) + 1
                    freq_hz = float(fft_freqs[peak_idx])
                    if freq_hz > 0.05:
                        period_s = 1.0 / freq_hz
                        omega_0 = 2.0 * np.pi * freq_hz
                        est_length = float(self.g / max(1e-4, omega_0**2))

        # Energetica
        L = max(0.05, est_length)
        v = L * omega
        e_k = 0.5 * self.mass * (v**2)
        h_pendulum = L * (1.0 - np.cos(theta))
        e_p = self.mass * self.g * h_pendulum
        e_tot = e_k + e_p

        # Verificacion con EDO no lineal (SciPy DOP853 vs SymPy) cada 5 segundos
        verified = False
        methods: list[str] = []
        discrepancy = 0.0

        if current_time - self._last_verification_time > 4.0 and est_length > 0:
            try:
                # EDO del pendulo no lineal amortiguado: d2y/dt2 = -gamma*dy - (g/L)*sin(y)
                omega_sq = self.g / L
                ode_str = f"-{damping:.4f}*dy - {omega_sq:.4f}*sin(y)"
                res = solve_and_verify_ode(
                    ode_rhs_order1_or_2=ode_str,
                    order=2,
                    t0=0.0,
                    t_end=max(1.0, period_s * 2 if period_s > 0 else 2.0),
                    y0=float(theta),
                    dy0=float(omega),
                )
                self._last_verified_ode_result = res
                self._last_verification_time = current_time
                verified = res.get("verified", False)
                methods = res.get("verification_methods", [])
                discrepancy = res.get("max_abs_discrepancy", 0.0)
            except Exception:
                pass
        elif self._last_verified_ode_result:
            verified = self._last_verified_ode_result.get("verified", False)
            methods = self._last_verified_ode_result.get("verification_methods", [])
            discrepancy = self._last_verified_ode_result.get("max_abs_discrepancy", 0.0)

        return KinematicsState(
            timestamp=current_time,
            theta_rad=theta,
            theta_deg=float(np.degrees(theta)),
            omega_rad_s=omega,
            alpha_rad_s2=alpha,
            period_s=period_s,
            frequency_hz=freq_hz,
            estimated_length_m=est_length,
            damping_factor=damping,
            energy_kinetic=e_k,
            energy_potential=e_p,
            energy_total=e_tot,
            verified=verified,
            verification_methods=methods,
            discrepancy=discrepancy,
        )

    def _draw_hud_telemetry(self, img: np.ndarray, state: KinematicsState | None) -> None:
        """Dibuja interfaz Bento HUD de alta precision sobre el frame."""
        h, w = img.shape[:2]
        p_x, p_y = int(self.pivot_x or w / 2), int(self.pivot_y or h * 0.15)

        # 1. Dibujar pivote
        cv2.circle(img, (p_x, p_y), 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (p_x, p_y), 14, (0, 220, 255), 1, cv2.LINE_AA)

        # 2. Dibujar estela de trayectoria
        pts = list(self.history)
        if len(pts) >= 2:
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                color = (int(255 * alpha), int(180 * alpha), int(40 * alpha))
                pt1 = (int(pts[i - 1].x), int(pts[i - 1].y))
                pt2 = (int(pts[i].x), int(pts[i].y))
                thickness = max(1, int(3 * alpha))
                cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

        # 3. Dibujar objetivo actual y vector velocidad
        if pts:
            cur = pts[-1]
            cx, cy = int(cur.x), int(cur.y)

            # Cuerda / varilla del pendulo
            cv2.line(img, (p_x, p_y), (cx, cy), (100, 255, 180), 2, cv2.LINE_AA)

            # Punto de masa / objetivo
            cv2.circle(img, (cx, cy), 10, (50, 100, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), 18, (0, 255, 255), 2, cv2.LINE_AA)

            # Vector velocidad tangencial
            v_scale = 15.0
            vx = -cur.omega * (cy - p_y) * (v_scale / max(1.0, self.pixels_per_meter))
            vy = cur.omega * (cx - p_x) * (v_scale / max(1.0, self.pixels_per_meter))
            cv2.arrowedLine(img, (cx, cy), (int(cx + vx), int(cy + vy)), (0, 255, 120), 2, tipLength=0.25)

        # 4. Banner superior Bento HUD
        cv2.rectangle(img, (15, 15), (360, 140), (12, 14, 20), -1)
        cv2.rectangle(img, (15, 15), (360, 140), (45, 52, 70), 1)

        cv2.putText(img, "ADA LIVE VISION & KINEMATICS", (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)

        if state:
            theta_str = f"Theta: {state.theta_deg:+.1f} deg  ({state.theta_rad:+.3f} rad)"
            omega_str = f"Omega: {state.omega_rad_s:+.2f} rad/s"
            freq_str = f"f0: {state.frequency_hz:.2f} Hz  |  L_est: {state.estimated_length_m:.2f} m"
            cv2.putText(img, theta_str, (25, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 225, 230), 1)
            cv2.putText(img, omega_str, (25, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 180), 1)
            cv2.putText(img, freq_str, (25, 108), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 190, 210), 1)

            verif_badge = "[VERIFICADO: SciPy DOP853 + SymPy]" if state.verified else "[CALCULANDO TELEMETRIA...]"
            badge_color = (120, 255, 120) if state.verified else (140, 160, 180)
            cv2.putText(img, verif_badge, (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.38, badge_color, 1)
        else:
            cv2.putText(img, "Buscando objetivo fisico...", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 170, 190), 1)
