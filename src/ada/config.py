"""Configuracion central reproducible, semillas fijas y barreras de seguridad de rutas para ADA."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

DEFAULT_SEED = int(os.environ.get("ADA_SEED", "42"))

DATA_RAW_DIR = (PROJECT_ROOT / "data" / "raw").resolve()
DATA_WORK_DIR = (PROJECT_ROOT / "data" / "work").resolve()
DATA_SAMPLES_DIR = (PROJECT_ROOT / "data" / "samples").resolve()
OUTPUTS_DIR = (PROJECT_ROOT / "outputs").resolve()
REPORTS_DIR = (PROJECT_ROOT / "reports").resolve()
DSH_HOME_DIR = (PROJECT_ROOT / os.environ.get("DSH_HOME", ".dsh_home")).resolve()
OLLAMA_MODELS_DIR = (PROJECT_ROOT / os.environ.get("OLLAMA_MODELS", ".ollama/models")).resolve()
PORTABLE_NODE_DIR = (PROJECT_ROOT / ".tools" / "node-v24.22.2-win-x64").resolve()
LOCAL_DSH_BIN_JS = (PROJECT_ROOT / "node_modules" / "@deepseek-ai" / "dsh" / "lib" / "bin.js").resolve()


def resolve_dsh_launch_args(
    *,
    profile: str = "sdk",
    patches: tuple[str | Path, ...] = (),
) -> tuple[tuple[str, ...], dict[str, str]]:
    """Construye los argumentos de arranque directo sobre Node v24.22.2 portable y @deepseek-ai/dsh local.

    Evita el error Win32 126 de koffi.node dentro del snapshot virtual de deepseek-harness-runtime-bin==0.2.1rc2.
    """
    node_exe = PORTABLE_NODE_DIR / "node.exe"
    if not node_exe.is_file():
        raise FileNotFoundError(f"No se encontro Node.js portable en {node_exe}. Ejecuta scripts/reproduce_env.ps1.")
    if not LOCAL_DSH_BIN_JS.is_file():
        raise FileNotFoundError(
            f"No se encontro @deepseek-ai/dsh en {LOCAL_DSH_BIN_JS}. Ejecuta scripts/reproduce_env.ps1."
        )

    patch_args: list[str] = []
    for p in patches:
        patch_args.extend(["--patch", str(Path(p).resolve())])

    launch_args = (str(node_exe), str(LOCAL_DSH_BIN_JS), "--profile", profile, *patch_args)
    extra_env = {
        "DSH_HOME": str(DSH_HOME_DIR),
        "PATH": f"{PORTABLE_NODE_DIR}{os.pathsep}{os.environ.get('PATH', '')}",
    }
    return launch_args, extra_env

ALLOWED_WRITE_ROOTS: tuple[Path, ...] = (
    DATA_WORK_DIR,
    OUTPUTS_DIR,
    REPORTS_DIR,
)


class SecurityViolationError(PermissionError):
    """Excepcion lanzada cuando una herramienta intenta escribir fuera de las rutas permitidas o en data/raw."""


def set_global_seed(seed: int = DEFAULT_SEED) -> int:
    """Fija todas las semillas pseudoaleatorias (Python, NumPy y PyTorch si esta cargado) para reproducibilidad."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def assert_safe_write_path(target: str | Path) -> Path:
    """Verifica que una ruta de escritura caiga estrictamente dentro de data/work, outputs o reports y nunca en data/raw."""
    resolved = Path(target).resolve()
    if resolved == DATA_RAW_DIR or DATA_RAW_DIR in resolved.parents:
        raise SecurityViolationError(f"VIOLACION DE INMUTABILIDAD: 'data/raw' es de solo lectura ({resolved}).")
    for allowed_root in ALLOWED_WRITE_ROOTS:
        if resolved == allowed_root or allowed_root in resolved.parents:
            return resolved
    allowed_str = ", ".join(str(p) for p in ALLOWED_WRITE_ROOTS)
    raise SecurityViolationError(
        f"ESCRITURA BLOQUEADA: '{resolved}' esta fuera de los directorios autorizados ({allowed_str})."
    )


@dataclass(frozen=True)
class ModelRoleConfig:
    """Contrato de configuracion de modelos locales libres por rol."""

    orchestrator_ollama: str = os.environ.get("ADA_MODEL_ORCHESTRATOR", "ada-q")
    orchestrator_hf: str = "Qwen/Qwen3-14B"
    orchestrator_license: str = "Apache-2.0"

    reasoner_ollama: str = os.environ.get("ADA_MODEL_REASONER", "ada-r1")
    reasoner_hf: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    reasoner_license: str = "MIT"

    qlora_agile_hf: str = "Qwen/Qwen3-8B"
    ollama_base_url: str = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    ollama_api_host: str = f"http://{os.environ.get('OLLAMA_HOST', '127.0.0.1:11434')}"
    context_length: int = int(os.environ.get("OLLAMA_CONTEXT_LENGTH", "24576"))
