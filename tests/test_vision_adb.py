"""Pruebas unitarias e integracion de vision en vivo, puente ADB y telemetria cinematica de ADA."""

import pytest
import numpy as np

from ada.vision.adb_bridge import (
    find_adb_executable,
    find_scrcpy_executable,
    list_adb_devices,
)
from ada.vision.realtime_tracker import (
    RealTimePhysicsTracker,
)
from ada.vision.synthetic_stream import SyntheticPhysicsLaboratoryStream
from ada.viz.live_vision_studio import build_live_vision_dashboard_html
from ada.viz.web_studio import audit_html_quality


def test_adb_binaries_discovery():
    adb = find_adb_executable()
    assert adb is not None
    assert "adb" in adb.lower()

    scrcpy = find_scrcpy_executable()
    assert scrcpy is not None
    assert "scrcpy" in scrcpy.lower()


def test_list_adb_devices_no_crash():
    devs = list_adb_devices()
    assert isinstance(devs, list)
    # Puede ser 0 si no hay telefono conectado fisicamente en este instante, pero no debe fallar


def test_synthetic_stream_generates_valid_frames():
    sim = SyntheticPhysicsLaboratoryStream(width=640, height=480)
    frame = sim.read_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8


def test_realtime_physics_tracker_kinematics_and_verification():
    sim = SyntheticPhysicsLaboratoryStream(width=800, height=600)
    tracker = RealTimePhysicsTracker()

    # Procesar 60 frames sucesivos
    last_kin = None
    for _ in range(60):
        frame = sim.read_frame()
        ann, kin = tracker.process_frame(frame)
        if kin:
            last_kin = kin

    assert last_kin is not None
    assert isinstance(last_kin.theta_deg, float)
    assert isinstance(last_kin.omega_rad_s, float)
    assert last_kin.frequency_hz >= 0.0
    # Verificacion matematica de EDO no lineal (DOP853)
    assert last_kin.verified is True
    assert "scipy.solve_ivp_DOP853" in last_kin.verification_methods


def test_live_vision_studio_generation_and_audit():
    p = build_live_vision_dashboard_html("test_ada_vision_dashboard.html")
    assert p.is_file()

    audit = audit_html_quality(p)
    assert audit["verified"] is True
    assert audit["grade"] in {"A+", "A"}
    assert audit["quality_score"] >= 90
    assert len(audit["critical_errors"]) == 0
