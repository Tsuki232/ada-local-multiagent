"""Generador sintetico de video fisico a 60 FPS para pruebas de vision sin hardware."""

from __future__ import annotations

import time
import cv2
import numpy as np


class SyntheticPhysicsLaboratoryStream:
    """Genera frames de video sintetico con fisica no lineal real, ruido gaussiano y sombras realistas."""

    def __init__(self, width: int = 800, height: int = 600) -> None:
        self.width = width
        self.height = height
        self.start_time = time.time()
        self.theta_0 = 0.65  # ~37 grados iniciales
        self.omega_0 = np.sqrt(9.80665 / 0.55)  # longitud = 0.55m
        self.damping = 0.04
        self.pivot = (int(width / 2), int(height * 0.18))
        self.rod_length_px = int(height * 0.52)

    def read_frame(self) -> np.ndarray:
        """Genera el siguiente frame en tiempo real evaluando la solucion fisica del pendulo no lineal."""
        t = time.time() - self.start_time
        # Solucion aproximada no lineal con amortiguamiento real
        theta = self.theta_0 * np.exp(-self.damping * t) * np.cos(self.omega_0 * t)

        img = np.full((self.height, self.width, 3), (18, 20, 26), dtype=np.uint8)

        # Rejilla de laboratorio en el fondo
        grid_step = 40
        for x in range(0, self.width, grid_step):
            cv2.line(img, (x, 0), (x, self.height), (26, 30, 40), 1)
        for y in range(0, self.height, grid_step):
            cv2.line(img, (0, y), (self.width, y), (26, 30, 40), 1)

        # Coordenadas del extremo
        px, py = self.pivot
        cx = int(px + self.rod_length_px * np.sin(theta))
        cy = int(py + self.rod_length_px * np.cos(theta))

        # Sombra proyectada
        shadow_y = int(self.height * 0.88)
        shadow_w = int(35 + (cy / self.height) * 30)
        cv2.ellipse(img, (cx, shadow_y), (shadow_w, 10), 0, 0, 360, (12, 14, 18), -1)

        # Soporte de laboratorio / Pivote
        cv2.rectangle(img, (px - 60, py - 20), (px + 60, py), (45, 52, 68), -1)
        cv2.circle(img, self.pivot, 8, (120, 140, 160), -1)
        cv2.circle(img, self.pivot, 4, (20, 24, 30), -1)

        # Varilla de acero pulido
        cv2.line(img, self.pivot, (cx, cy), (160, 175, 195), 3, cv2.LINE_AA)

        # Marcador naranja fluorescente en la masa (detectable por HSV)
        cv2.circle(img, (cx, cy), 22, (20, 120, 240), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 14, (30, 160, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (cx - 5, cy - 5), 5, (120, 220, 255), -1, cv2.LINE_AA)

        # Ruido suave de camara digital para mayor realismo
        noise = np.random.normal(0, 2.0, (self.height, self.width, 3)).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return noisy
