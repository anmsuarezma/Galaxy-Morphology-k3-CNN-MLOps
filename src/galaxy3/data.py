"""Datos de Galaxy3: transformaciones, estadísticas y DataLoaders.

Este módulo no sabe nada del modelo. Las transformaciones se construyen a partir de la
configuración (resolución, crop) y de ``dataset_stats.json`` (normalización); no quedan
fijadas aquí.
"""

from __future__ import annotations

import json
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from galaxy3.config import ExperimentConfig
from galaxy3.utils import get_device, make_generator, seed_worker

DATASET_STATS_NAME = "dataset_stats.json"
"""Archivo que escribe la preparación en ``data_root`` (split_seed, clases, mean/std)."""

SPLITS: tuple[str, ...] = ("train", "val", "test")


def worker_context(num_workers: int) -> str | None:
    """Contexto de multiprocessing para los workers del ``DataLoader``.

    En Linux el default es ``fork``, y hacer fork de un proceso con hilos (CUDA, MLflow,
    OpenMP) puede dejar un worker bloqueado (``docs/PILOTO.md``). Por eso en Linux se usa
    ``forkserver``: los workers nacen de un proceso servidor sin hilos. macOS y Windows ya
    usan ``spawn`` por defecto y conservan su contexto. Con ``forkserver`` o ``spawn`` el
    dataset, sus transforms y ``worker_init_fn`` deben ser picklables.

    Args:
        num_workers: Workers del ``DataLoader`` (>= 0).

    Returns:
        ``"forkserver"`` en Linux con ``num_workers > 0``; ``None`` (contexto por defecto)
        en otro caso. ``DataLoader`` no acepta un contexto con ``num_workers=0``.
    """
    return "forkserver" if num_workers > 0 and sys.platform.startswith("linux") else None


def _loader_context(num_workers: int) -> str | None:
    """``worker_context`` + precarga de los módulos de datos en el servidor ``forkserver``.

    Sin precarga, cada worker importa de nuevo torch y torchvision (medido con 8 workers
    en Jupyter: ~16 s hasta el primer batch frente a ~2 s con precarga). Los módulos
    precargados se importan una vez en el servidor y los workers los heredan. Solo se
    precargan módulos que no crean hilos al importarse: precargar, por ejemplo, MLflow
    dejaría hilos en el servidor y volvería el riesgo de fork. La precarga solo tiene
    efecto antes de que el servidor arranque (con el primer loader con workers).
    """
    context = worker_context(num_workers)
    if context == "forkserver":
        multiprocessing.set_forkserver_preload(["__main__", "torch", "torchvision", "galaxy3.data"])
    return context


def build_base_transform(image_size: int, center_crop: int | None = None) -> transforms.Compose:
    """Transformación geométrica + ``ToTensor``, sin normalizar.

    Es la parte del pipeline previa a ``Normalize``: con ella se calculan
    ``mean_train`` y ``std_train`` en escala ``[0, 1]``.

    Args:
        image_size: Lado de la imagen cuadrada de salida (>= 1).
        center_crop: Lado del ``CenterCrop`` aplicado antes del ``Resize``, o ``None``.

    Returns:
        ``[CenterCrop] -> Resize((image_size, image_size)) -> ToTensor``. Produce
        tensores ``float32`` de forma ``(C, image_size, image_size)`` en ``[0, 1]``.

    Raises:
        ValueError: Si ``image_size`` o ``center_crop`` no son positivos.
    """
    if image_size < 1:
        raise ValueError(f"image_size debe ser >= 1 (es {image_size})")
    if center_crop is not None and center_crop < 1:
        raise ValueError(f"center_crop debe ser None o >= 1 (es {center_crop})")
    crop = [transforms.CenterCrop(center_crop)] if center_crop is not None else []
    return transforms.Compose(
        [*crop, transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )


def compute_channel_stats(
    dataset: Dataset,
    batch_size: int = 256,
    num_workers: int = 0,
) -> tuple[list[float], list[float]]:
    """Media y desviación estándar por canal sobre todos los píxeles de ``dataset``.

    Debe llamarse **solo con train**, con ``build_base_transform`` como transformación,
    para que las estadísticas estén en la misma escala que recibirá ``Normalize``.
    Acumula sumas en ``float64`` por batch (sin loops sobre píxeles ni imágenes).

    Args:
        dataset: Dataset que devuelve pares ``(X, y)`` con ``X`` de forma ``(C, H, W)``;
            todas las imágenes con el mismo ``H, W``.
        batch_size: Tamaño de batch para recorrer el dataset (>= 1).
        num_workers: Workers del ``DataLoader`` (>= 0).

    Returns:
        ``(mean, std)``, listas de ``C`` floats. ``std`` es la desviación poblacional.

    Raises:
        ValueError: Si el dataset está vacío o los argumentos son inválidos.
    """
    if batch_size < 1 or num_workers < 0:
        raise ValueError(f"batch_size >= 1 y num_workers >= 0 (son {batch_size}, {num_workers})")
    if len(dataset) == 0:  # type: ignore[arg-type]
        raise ValueError("El dataset está vacío; no se pueden calcular estadísticas.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        multiprocessing_context=_loader_context(num_workers),
    )
    total = sq_total = None
    n_pixels = 0
    for X, _ in loader:
        X = X.to(torch.float64)
        batch_sum, batch_sq = X.sum(dim=(0, 2, 3)), (X * X).sum(dim=(0, 2, 3))
        total = batch_sum if total is None else total + batch_sum
        sq_total = batch_sq if sq_total is None else sq_total + batch_sq
        n_pixels += X.shape[0] * X.shape[2] * X.shape[3]
    mean = total / n_pixels
    std = (sq_total / n_pixels - mean * mean).clamp_min(0).sqrt()
    return mean.tolist(), std.tolist()


def load_dataset_stats(data_root: Path) -> dict[str, Any]:
    """Lee ``data_root / dataset_stats.json``, escrito por la preparación.

    Raises:
        FileNotFoundError: Si el archivo no existe (falta ejecutar ``prepare_galaxy3.py``).
    """
    path = Path(data_root) / DATASET_STATS_NAME
    if not path.is_file():
        raise FileNotFoundError(f"No existe {path}; ejecuta primero scripts/prepare_galaxy3.py.")
    return json.loads(path.read_text())


def load_original_samples(
    data_root: Path, split: str = "train", per_class: int = 4, seed: int = 0
) -> dict[str, list[Image.Image]]:
    """Carga imágenes originales por clase, **sin ninguna transformación** (para visualizar).

    Las imágenes son las exportadas por la preparación (256x256, composición en falso color
    de las bandas g, r, z). La selección es reproducible con ``seed`` y no consume el RNG
    global.

    Args:
        data_root: Carpeta con ``train/``, ``val/`` y ``test/`` en formato ``ImageFolder``.
        split: ``"train"``, ``"val"`` o ``"test"``.
        per_class: Número de imágenes por clase (>= 1).
        seed: Semilla de la selección.

    Returns:
        ``{class_name: [PIL.Image, ...]}`` en el orden de las clases (índice = etiqueta).

    Raises:
        ValueError: Si ``split`` no es válido, ``per_class < 1`` o alguna clase tiene menos
            de ``per_class`` imágenes.
    """
    if split not in ("train", "val", "test"):
        raise ValueError(f"split debe ser 'train', 'val' o 'test'; se recibió {split!r}")
    if per_class < 1:
        raise ValueError(f"per_class debe ser >= 1; se recibió {per_class}")
    dataset = ImageFolder(Path(data_root) / split)
    targets = np.asarray(dataset.targets)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[Image.Image]] = {}
    for label, name in enumerate(dataset.classes):
        indices = np.flatnonzero(targets == label)
        if len(indices) < per_class:
            raise ValueError(f"La clase {name!r} tiene {len(indices)} imágenes (< {per_class})")
        chosen = np.sort(rng.choice(indices, size=per_class, replace=False))
        samples[name] = [dataset.loader(dataset.samples[i][0]) for i in chosen]
    return samples


def check_stats_match_config(stats: dict[str, Any], config: ExperimentConfig) -> None:
    """Verifica que ``dataset_stats.json`` corresponda a la configuración.

    ``split_seed`` debe coincidir (el split se fijó en la preparación), y también
    ``image_size`` y ``center_crop``: ``mean_train``/``std_train`` solo son válidas
    para las transformaciones con que se calcularon.

    Raises:
        ValueError: Si algún valor no coincide.
    """
    transform = stats["transform"]
    pairs = {
        "split_seed": (stats["split_seed"], config.split_seed),
        "image_size": (transform["image_size"], config.image_size),
        "center_crop": (transform["center_crop"], config.center_crop),
    }
    errors = [f"{k}: stats={s!r}, config={c!r}" for k, (s, c) in pairs.items() if s != c]
    if errors:
        raise ValueError(
            "dataset_stats.json no coincide con la configuración ("
            + "; ".join(errors)
            + "). Ajusta la configuración o regenera los datos con prepare_galaxy3.py."
        )


def build_transform(
    config: ExperimentConfig, mean: list[float], std: list[float]
) -> transforms.Compose:
    """Pipeline de entrada al modelo: base (crop, ``Resize``, ``ToTensor``) + ``Normalize``.

    Args:
        config: Aporta ``image_size`` y ``center_crop``.
        mean: Media por canal de train (``mean_train``), longitud 3.
        std: Desviación por canal de train (``std_train``), longitud 3, valores > 0.

    Returns:
        Transformación que produce tensores ``(3, image_size, image_size)`` normalizados.

    Raises:
        ValueError: Si ``mean``/``std`` no tienen 3 valores o algún ``std`` no es positivo.
    """
    if len(mean) != 3 or len(std) != 3 or any(s <= 0 for s in std):
        raise ValueError(f"mean/std deben tener 3 valores y std > 0: mean={mean}, std={std}")
    base = build_base_transform(config.image_size, config.center_crop)
    return transforms.Compose([*base.transforms, transforms.Normalize(mean, std)])


@dataclass(frozen=True)
class DataLoaders:
    """Loaders de los tres splits y nombres de clase (índice = etiqueta)."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]


def _make_loader(
    dataset: Dataset, config: ExperimentConfig, device: torch.device, shuffle: bool
) -> DataLoader:
    # Cada loader tiene su propio generador: el shuffle y las semillas de los workers no
    # consumen el RNG global (que usan la inicialización del modelo y el Dropout).
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.num_workers > 0,
        multiprocessing_context=_loader_context(config.num_workers),
        worker_init_fn=seed_worker,
        generator=make_generator(config.train_seed),
    )


def make_split_loader(
    config: ExperimentConfig, split: str, device: torch.device | None = None
) -> DataLoader:
    """Construye el loader de **un solo** split desde ``config.data_root / split``.

    Solo se crea el ``ImageFolder`` del split pedido: ``run_experiment`` pide train y val,
    y nunca toca la carpeta de test. Train usa ``shuffle=True`` con un generador sembrado
    con ``train_seed`` (misma ``train_seed`` -> mismo orden de batches); val y test usan
    ``shuffle=False``. ``pin_memory`` solo se activa en CUDA; ``persistent_workers``
    solo con ``num_workers > 0``.

    Args:
        config: Configuración del experimento.
        split: ``"train"``, ``"val"`` o ``"test"``.
        device: Device de entrenamiento; solo decide ``pin_memory``. Si es ``None``,
            se resuelve con ``get_device(config.device_preference)``.

    Returns:
        Loader cuyo ``dataset`` es un ``ImageFolder`` (``dataset.classes`` = nombres de
        clase, índice = etiqueta), con batches ``X`` de forma
        ``(B, 3, image_size, image_size)`` (``float32``, normalizados) e ``y`` de forma
        ``(B,)`` (``int64``).

    Raises:
        FileNotFoundError: Si falta ``dataset_stats.json``.
        ValueError: Si ``split`` no es válido, ``dataset_stats.json`` no coincide con
            ``config`` o las clases de la carpeta no coinciden con las registradas en la
            preparación o con ``config.num_classes``.
    """
    if split not in SPLITS:
        raise ValueError(f"split debe ser uno de {SPLITS}; se recibió {split!r}")
    stats = load_dataset_stats(config.data_root)
    check_stats_match_config(stats, config)
    device = device if device is not None else get_device(config.device_preference)
    transform = build_transform(config, stats["mean_train"], stats["std_train"])
    dataset = ImageFolder(config.data_root / split, transform=transform)
    if dataset.class_to_idx != stats["class_to_label"]:
        raise ValueError(
            f"Clases de {split} ({dataset.class_to_idx}) no coinciden con "
            f"dataset_stats.json ({stats['class_to_label']})"
        )
    if len(dataset.classes) != config.num_classes:
        raise ValueError(
            f"config.num_classes={config.num_classes}, pero hay "
            f"{len(dataset.classes)} clases en {config.data_root / split}"
        )
    return _make_loader(dataset, config, device, shuffle=split == "train")


def make_dataloaders(config: ExperimentConfig, device: torch.device | None = None) -> DataLoaders:
    """Construye los loaders de train, val y test (ver ``make_split_loader``).

    Útil para inspección en el notebook. ``run_experiment`` no la usa: construye solo
    train y val. El loader de test es para la evaluación final explícita
    (``evaluate_on_test``); no debe pasarse a ``fit``.

    Args:
        config: Configuración del experimento.
        device: Device de entrenamiento; solo decide ``pin_memory``.

    Returns:
        ``DataLoaders`` con los tres splits y los nombres de clase.

    Raises:
        FileNotFoundError: Si falta ``dataset_stats.json``.
        ValueError: Si ``dataset_stats.json`` no coincide con ``config`` o las clases de
            las carpetas no coinciden con las registradas en la preparación.
    """
    loaders = {split: make_split_loader(config, split, device) for split in SPLITS}
    return DataLoaders(**loaders, class_names=list(loaders["train"].dataset.classes))
