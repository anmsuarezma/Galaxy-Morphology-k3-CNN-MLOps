"""Utilidades transversales: device, semillas y rutas.

Este módulo no conoce el modelo, los datos ni MLflow.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""Raíz del proyecto Galaxy3 (carpeta ``galaxy3/``), independiente del directorio actual."""

VALID_DEVICE_PREFERENCES: tuple[str, ...] = ("auto", "cuda", "mps", "cpu")


def _mps_available() -> bool:
    return torch.backends.mps.is_available()


def get_device(preference: str = "auto") -> torch.device:
    """Selecciona el device de cómputo.

    Con ``"auto"`` el orden es CUDA -> MPS -> CPU. Con un device explícito no hay
    fallback silencioso: si no está disponible se lanza un error.

    Args:
        preference: Uno de ``"auto"``, ``"cuda"``, ``"mps"`` o ``"cpu"``.

    Returns:
        El ``torch.device`` seleccionado.

    Raises:
        ValueError: Si ``preference`` no es un valor válido.
        RuntimeError: Si se pide explícitamente un device no disponible.
    """
    if preference not in VALID_DEVICE_PREFERENCES:
        raise ValueError(
            f"device_preference inválido: {preference!r}; "
            f"opciones válidas: {VALID_DEVICE_PREFERENCES}"
        )
    available = {"cuda": torch.cuda.is_available(), "mps": _mps_available(), "cpu": True}
    if preference == "auto":
        return torch.device(next(name for name, ok in available.items() if ok))
    if not available[preference]:
        raise RuntimeError(f"Se pidió device {preference!r}, pero no está disponible.")
    return torch.device(preference)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Siembra los generadores globales de ``random``, NumPy y PyTorch (incluida CUDA).

    Se llama con ``train_seed`` al inicio de cada run. Garantiza reproducibilidad
    razonable en el mismo hardware, no identidad bit a bit entre CUDA, MPS y CPU.

    Args:
        seed: Semilla entera no negativa.
        deterministic: Si es ``True``, fuerza algoritmos deterministas de cuDNN
            (``cudnn.deterministic=True``, ``cudnn.benchmark=False``). Costo: puede
            hacer más lentas las convoluciones en GPU, porque desactiva la búsqueda
            del algoritmo más rápido.

    Raises:
        ValueError: Si ``seed`` es negativa.
    """
    if seed < 0:
        raise ValueError(f"seed debe ser no negativa; se recibió {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # también siembra todas las GPUs CUDA
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_generator(seed: int) -> torch.Generator:
    """Crea un ``torch.Generator`` sembrado, para el shuffle del ``DataLoader``.

    Al tener su propio generador, el orden de los batches no depende del RNG global
    (que consumen, por ejemplo, las máscaras de Dropout).

    Args:
        seed: Semilla entera no negativa (``train_seed``).

    Returns:
        Un generador nuevo en CPU.

    Raises:
        ValueError: Si ``seed`` es negativa.
    """
    if seed < 0:
        raise ValueError(f"seed debe ser no negativa; se recibió {seed}")
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` para el ``DataLoader``: siembra ``random`` y NumPy en cada worker.

    PyTorch ya asigna a cada worker una semilla distinta derivada del generador del
    ``DataLoader``; aquí se propaga a ``random`` y NumPy.

    Args:
        worker_id: Índice del worker (lo pasa el ``DataLoader``; no se usa).
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class OutputDirs:
    """Carpetas de salida de un experimento."""

    checkpoints: Path
    figures: Path
    metrics: Path


def ensure_dir(path: Path) -> Path:
    """Crea ``path`` (y sus padres) si no existe y lo devuelve. Es idempotente."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(output_root: Path) -> OutputDirs:
    """Crea ``checkpoints/``, ``figures/`` y ``metrics/`` dentro de ``output_root``.

    Args:
        output_root: Carpeta raíz de salidas.

    Returns:
        Las rutas creadas.
    """
    return OutputDirs(
        checkpoints=ensure_dir(output_root / "checkpoints"),
        figures=ensure_dir(output_root / "figures"),
        metrics=ensure_dir(output_root / "metrics"),
    )
