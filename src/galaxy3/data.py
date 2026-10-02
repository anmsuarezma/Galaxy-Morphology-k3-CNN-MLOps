"""Transformaciones y estadísticas de imágenes para Galaxy3.

Este módulo no sabe nada del modelo. Las transformaciones se construyen a partir de
parámetros explícitos (resolución y crop), no quedan fijadas aquí.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
