"""Engine de entrenamiento: ``train_one_epoch``, ``evaluate`` y ``fit``.

El engine solo asume el contrato externo del modelo (sección 7 de la spec)::

    logits = model(X_batch)   # X_batch: (B, C, H, W) -> logits: (B, num_classes)

No accede a submódulos internos, no conoce MLflow y nunca recibe el loader de test.
Protocolo: número fijo de épocas, sin scheduler ni early stopping; se conserva el
checkpoint de la época con menor ``val_loss``.

Las métricas se acumulan como tensores en el device y se convierten a ``float`` una
sola vez por época: llamar ``.item()`` en cada batch forzaría una sincronización
CPU-GPU por batch.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch
from torch import nn
from torch.utils.data import DataLoader

from galaxy3.config import ExperimentConfig

BatchCallback = Callable[[torch.Tensor, torch.Tensor], None]
"""Recibe ``(logits, y_batch)`` de cada batch, ya en el device: ``(B, K)`` y ``(B,)``."""


@dataclass(frozen=True)
class EpochMetrics:
    """Métricas de una pasada completa sobre un loader.

    Attributes:
        loss: Pérdida promedio por muestra.
        accuracy: Fracción de aciertos, en ``[0, 1]``.
    """

    loss: float
    accuracy: float


class History(TypedDict):
    """Historia de ``fit``. Las listas tienen una entrada por época, en orden.

    ``train_loss`` y ``train_acc`` se acumulan durante la época en modo train (Dropout
    activo, BN con estadísticas del batch, pesos cambiando): sirven para curvas, no
    para el generalization gap. ``val_*`` se miden en modo eval al final de la época.
    ``best_epoch`` está numerada desde 1.
    """

    train_loss: list[float]
    train_acc: list[float]
    val_loss: list[float]
    val_acc: list[float]
    lr: list[float]
    epoch_time: list[float]
    best_epoch: int
    best_val_loss: float


def _check_mean_reduction(loss_fn: nn.Module) -> None:
    reduction = getattr(loss_fn, "reduction", "mean")
    if reduction != "mean":
        raise ValueError(
            f"loss_fn debe promediar sobre el batch (reduction='mean'); tiene {reduction!r}"
        )


def _synchronize(device: torch.device) -> None:
    """Espera a que terminen los kernels CUDA pendientes, para medir tiempos reales."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    on_batch: BatchCallback | None = None,
) -> EpochMetrics:
    """Recorre ``loader`` una vez; si hay ``optimizer``, actualiza los pesos en cada batch."""
    _check_mean_reduction(loss_fn)
    non_blocking = device.type == "cuda"
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    num_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=non_blocking)
        y_batch = y_batch.to(device, non_blocking=non_blocking)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        if on_batch is not None:
            on_batch(logits.detach(), y_batch)

        batch_size = y_batch.size(0)
        loss_sum += loss.detach() * batch_size
        correct += (logits.argmax(dim=1) == y_batch).sum()
        num_samples += batch_size

    if num_samples == 0:
        raise ValueError("El loader no produjo ninguna muestra.")
    return EpochMetrics(
        loss=loss_sum.item() / num_samples,
        accuracy=correct.item() / num_samples,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    """Entrena una época: ``zero_grad -> forward -> loss -> backward -> step`` por batch.

    Pone el modelo en modo train. Modifica los pesos del modelo, el estado del
    optimizer y, si hay BN, sus running statistics.

    Args:
        model: Modelo que cumple ``(B, C, H, W) -> (B, num_classes)``, ya en ``device``.
        loader: Loader de ``(X_batch, y_batch)`` con ``y_batch`` de forma ``(B,)``.
        loss_fn: Pérdida con ``reduction='mean'`` sobre logits (p. ej. ``CrossEntropyLoss``).
        optimizer: Optimizer construido sobre los parámetros de ``model``.
        device: Device del modelo; los batches se transfieren a él.

    Returns:
        Pérdida y accuracy promedio por muestra, acumuladas durante la época.

    Raises:
        ValueError: Si ``loader`` está vacío o ``loss_fn`` no promedia sobre el batch.
    """
    model.train()
    return _run_epoch(model, loader, loss_fn, device, optimizer)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    on_batch: BatchCallback | None = None,
) -> EpochMetrics:
    """Evalúa en modo eval y bajo ``torch.inference_mode()``: sin backward ni step.

    Deja el modelo en modo eval y no modifica pesos ni running statistics.

    Args:
        model: Modelo que cumple ``(B, C, H, W) -> (B, num_classes)``, ya en ``device``.
        loader: Loader de ``(X_batch, y_batch)`` con ``y_batch`` de forma ``(B,)``.
        loss_fn: Pérdida con ``reduction='mean'`` sobre logits.
        device: Device del modelo.
        on_batch: Si se da, se llama en cada batch con ``(logits, y_batch)`` en el
            device, en el orden del loader. Permite recolectar predicciones sin
            duplicar el bucle (ver ``metrics.collect_predictions``).

    Returns:
        Pérdida y accuracy promedio por muestra.

    Raises:
        ValueError: Si ``loader`` está vacío o ``loss_fn`` no promedia sobre el batch.
    """
    model.eval()
    with torch.inference_mode():
        return _run_epoch(model, loader, loss_fn, device, optimizer=None, on_batch=on_batch)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_loss: float,
    config: ExperimentConfig,
) -> None:
    """Guarda un checkpoint cargable con ``torch.load(path, weights_only=True)``.

    Solo contiene tipos primitivos, tensores y ``state_dict``; la configuración se
    guarda como ``config.to_dict()``, nunca como dataclass. Crea la carpeta padre.

    Args:
        path: Archivo de destino (se sobrescribe).
        epoch: Época del checkpoint, numerada desde 1.
        model: Modelo cuyo ``state_dict`` se guarda.
        optimizer: Optimizer cuyo ``state_dict`` se guarda.
        val_loss: Pérdida de validación de esa época.
        config: Configuración del run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": float(val_loss),
            "config": config.to_dict(),
        },
        path,
    )


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: ExperimentConfig,
    checkpoint_path: Path,
    verbose: bool = True,
) -> History:
    """Entrena exactamente ``config.epochs`` épocas y guarda el mejor checkpoint por ``val_loss``.

    Sin scheduler ni early stopping. Recibe solo train y validation: el test nunca entra
    aquí. El checkpoint se sobrescribe cada vez que ``val_loss < best_val_loss``, de modo
    que al terminar contiene la época con menor ``val_loss``. Al final, ``model`` queda
    con los pesos de la **última** época; para usar el mejor modelo hay que recargar el
    checkpoint.

    ``epoch_time`` mide train + validation de cada época con ``time.perf_counter()``,
    sincronizando CUDA antes de cada lectura del reloj.

    Args:
        model: Modelo ya movido a ``device``.
        train_loader: Loader de entrenamiento.
        val_loader: Loader de validación.
        loss_fn: Pérdida con ``reduction='mean'`` sobre logits.
        optimizer: Optimizer sobre los parámetros de ``model``.
        device: Device de cómputo.
        config: Configuración del run; aporta ``epochs`` y se guarda en el checkpoint.
        checkpoint_path: Archivo del mejor checkpoint.
        verbose: Si es ``True``, imprime una línea por época.

    Returns:
        La historia del entrenamiento (ver ``History``).

    Raises:
        ValueError: Si algún loader está vacío o ``loss_fn`` no promedia sobre el batch.
        RuntimeError: Si ``val_loss`` no fue finita en ninguna época (no hay checkpoint).
    """
    history: History = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epoch_time": [],
        "best_epoch": 0,
        "best_val_loss": math.inf,
    }

    for epoch in range(1, config.epochs + 1):
        lr = optimizer.param_groups[0]["lr"]
        _synchronize(device)
        start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        _synchronize(device)
        epoch_time = time.perf_counter() - start

        history["train_loss"].append(train_metrics.loss)
        history["train_acc"].append(train_metrics.accuracy)
        history["val_loss"].append(val_metrics.loss)
        history["val_acc"].append(val_metrics.accuracy)
        history["lr"].append(lr)
        history["epoch_time"].append(epoch_time)

        improved = val_metrics.loss < history["best_val_loss"]
        if improved:
            history["best_epoch"] = epoch
            history["best_val_loss"] = val_metrics.loss
            save_checkpoint(checkpoint_path, epoch, model, optimizer, val_metrics.loss, config)
        if verbose:
            print(
                f"Época {epoch:>3}/{config.epochs} | "
                f"train loss {train_metrics.loss:.4f} acc {train_metrics.accuracy:.4f} | "
                f"val loss {val_metrics.loss:.4f} acc {val_metrics.accuracy:.4f} | "
                f"{epoch_time:.1f}s{' *' if improved else ''}"
            )

    if history["best_epoch"] == 0:
        raise RuntimeError("val_loss no fue finita en ninguna época; no se guardó checkpoint.")
    return history
