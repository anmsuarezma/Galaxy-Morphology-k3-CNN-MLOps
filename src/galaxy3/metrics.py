"""Métricas de clasificación y generalization gap.

Todo se calcula con tensores de PyTorch a partir de etiquetas enteras ``(N,)``. Este
módulo no conoce MLflow ni Matplotlib: devuelve números y tensores que luego consumen
``plots.py`` y ``experiment.py``.

Convención de la matriz de confusión: ``cm[i, j]`` = número de muestras de clase real
``i`` predichas como clase ``j`` (filas = real, columnas = predicha).

Convención ante divisiones por cero (igual que ``zero_division=0`` en scikit-learn): si
una clase nunca se predice, su precision es 0; si no tiene muestras reales, su recall es
0; si precision + recall = 0, su F1 es 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from galaxy3.engine import evaluate


@dataclass(frozen=True)
class Predictions:
    """Salida de ``collect_predictions``, en CPU y en el orden del loader.

    Attributes:
        y_true: Etiquetas reales, ``(N,)`` de tipo ``long``.
        y_pred: Clase predicha (argmax de los logits), ``(N,)`` de tipo ``long``.
        probabilities: Softmax de los logits, ``(N, K)``; cada fila suma 1.
        loss: Pérdida promedio por muestra, calculada por ``engine.evaluate``.
        accuracy: Accuracy calculada por ``engine.evaluate``.
    """

    y_true: torch.Tensor
    y_pred: torch.Tensor
    probabilities: torch.Tensor
    loss: float
    accuracy: float


@dataclass(frozen=True)
class ClassificationReport:
    """Métricas de clasificación de un split.

    Attributes:
        accuracy: Fracción de aciertos, en ``[0, 1]``.
        confusion_matrix: ``(K, K)`` de tipo ``long``; filas = real, columnas = predicha.
        precision: Precision por clase, ``(K,)``.
        recall: Recall por clase, ``(K,)``.
        f1: F1 por clase, ``(K,)``.
        macro_f1: Promedio no ponderado de ``f1`` sobre las ``K`` clases.
    """

    accuracy: float
    confusion_matrix: torch.Tensor
    precision: torch.Tensor
    recall: torch.Tensor
    f1: torch.Tensor
    macro_f1: float


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> Predictions:
    """Recorre ``loader`` con ``engine.evaluate`` y guarda etiquetas, predicciones y probabilidades.

    Reutiliza el bucle de ``evaluate`` (modo eval, ``torch.inference_mode()``) mediante su
    parámetro ``on_batch``: los logits se acumulan en el device y se pasan a CPU una sola
    vez al final. El softmax se aplica solo aquí, para análisis; el modelo sigue
    devolviendo logits. Deja el modelo en modo eval y no modifica sus pesos.

    Para que la posición ``i`` corresponda a ``loader.dataset[i]``, el loader no debe
    barajar (val y test usan ``shuffle=False``).

    Args:
        model: Modelo que cumple ``(B, C, H, W) -> (B, K)``, ya en ``device``.
        loader: Loader de ``(X_batch, y_batch)``.
        loss_fn: Pérdida con ``reduction='mean'`` sobre logits.
        device: Device del modelo.

    Returns:
        Las predicciones del split (ver ``Predictions``).

    Raises:
        ValueError: Si ``loader`` está vacío o ``loss_fn`` no promedia sobre el batch.
    """
    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []

    def store(logits: torch.Tensor, y_batch: torch.Tensor) -> None:
        logits_batches.append(logits)
        label_batches.append(y_batch)

    metrics = evaluate(model, loader, loss_fn, device, on_batch=store)
    with torch.inference_mode():
        logits = torch.cat(logits_batches).float().cpu()
        y_true = torch.cat(label_batches).long().cpu()
        probabilities = torch.softmax(logits, dim=1)
    return Predictions(
        y_true=y_true,
        y_pred=logits.argmax(dim=1),
        probabilities=probabilities,
        loss=metrics.loss,
        accuracy=metrics.accuracy,
    )


def _check_labels(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int | None) -> None:
    """Exige tensores enteros 1D, no vacíos, de igual forma y (si se da K) en ``[0, K)``."""
    if num_classes is not None and num_classes < 1:
        raise ValueError(f"num_classes debe ser >= 1; se recibió {num_classes}")
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if y.ndim != 1 or y.dtype.is_floating_point or y.dtype == torch.bool:
            raise ValueError(f"{name} debe ser un tensor entero 1D; tiene {y.dtype}, {y.shape}")
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {tuple(y_true.shape)} y y_pred {tuple(y_pred.shape)} difieren")
    if y_true.numel() == 0:
        raise ValueError("y_true y y_pred están vacíos")
    if num_classes is None:
        return
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if y.min() < 0 or y.max() >= num_classes:
            raise ValueError(f"{name} tiene etiquetas fuera de [0, {num_classes})")


def accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Fracción de posiciones donde ``y_pred == y_true``.

    Args:
        y_true: Etiquetas reales, tensor entero ``(N,)``, ``N > 0``.
        y_pred: Etiquetas predichas, misma forma que ``y_true``.

    Returns:
        Accuracy en ``[0, 1]``.

    Raises:
        ValueError: Si las formas difieren, no son enteros 1D o están vacíos.
    """
    _check_labels(y_true, y_pred, num_classes=None)
    return (y_true == y_pred).float().mean().item()


def confusion_matrix(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Matriz de confusión ``(K, K)`` con filas = clase real y columnas = clase predicha.

    Se calcula sin bucles: el par ``(i, j)`` se codifica como ``i * K + j`` y se cuenta
    con ``torch.bincount``.

    Args:
        y_true: Etiquetas reales, tensor entero ``(N,)`` con valores en ``[0, K)``.
        y_pred: Etiquetas predichas, misma forma y rango.
        num_classes: ``K``; fija el tamaño aunque alguna clase no aparezca.

    Returns:
        Tensor ``long`` ``(K, K)`` en el device de las entradas; suma ``N``.

    Raises:
        ValueError: Si las entradas no cumplen el contrato.
    """
    _check_labels(y_true, y_pred, num_classes)
    codes = y_true.long() * num_classes + y_pred.long()
    counts = torch.bincount(codes, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """División elemento a elemento que devuelve 0 donde el denominador es 0."""
    numerator = numerator.double()
    denominator = denominator.double()
    positive = denominator > 0
    ratio = numerator / torch.where(positive, denominator, 1.0)
    return torch.where(positive, ratio, 0.0)


def per_class_metrics(cm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precision, recall y F1 por clase a partir de una matriz de confusión.

    Para la clase ``k``: ``TP = cm[k, k]``, ``precision = TP / sum(cm[:, k])`` (sobre lo
    predicho como ``k``), ``recall = TP / sum(cm[k, :])`` (sobre lo que realmente es
    ``k``) y ``F1 = 2 * P * R / (P + R)``. Divisiones por cero dan 0.

    Args:
        cm: Matriz ``(K, K)`` no negativa, filas = real, columnas = predicha.

    Returns:
        ``(precision, recall, f1)``, cada uno ``(K,)`` de tipo ``float64``.

    Raises:
        ValueError: Si ``cm`` no es cuadrada o tiene entradas negativas.
    """
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm debe ser cuadrada (K, K); tiene forma {tuple(cm.shape)}")
    if (cm < 0).any():
        raise ValueError("cm tiene entradas negativas")
    true_positives = cm.diagonal()
    precision = _safe_divide(true_positives, cm.sum(dim=0))
    recall = _safe_divide(true_positives, cm.sum(dim=1))
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def macro_f1(cm: torch.Tensor) -> float:
    """Promedio no ponderado del F1 por clase: cada clase pesa igual, sin importar su tamaño.

    Args:
        cm: Matriz de confusión ``(K, K)``.

    Returns:
        Macro F1 en ``[0, 1]``.

    Raises:
        ValueError: Si ``cm`` no es una matriz de confusión válida.
    """
    _, _, f1 = per_class_metrics(cm)
    return f1.mean().item()


def classification_report(
    y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int
) -> ClassificationReport:
    """Calcula todas las métricas de clasificación de un split.

    Args:
        y_true: Etiquetas reales, tensor entero ``(N,)`` con valores en ``[0, K)``.
        y_pred: Etiquetas predichas, misma forma y rango.
        num_classes: ``K``.

    Returns:
        El reporte con accuracy, matriz de confusión, métricas por clase y macro F1. Los
        tensores quedan en CPU.

    Raises:
        ValueError: Si las entradas no cumplen el contrato.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes).cpu()
    precision, recall, f1 = per_class_metrics(cm)
    return ClassificationReport(
        accuracy=(cm.diagonal().sum() / cm.sum()).item(),
        confusion_matrix=cm,
        precision=precision,
        recall=recall,
        f1=f1,
        macro_f1=f1.mean().item(),
    )


def generalization_gap(train_acc_eval: float, val_acc: float) -> float:
    """``train_acc_eval - val_acc`` (sección 17 de la spec).

    ``train_acc_eval`` debe medirse recargando el best checkpoint y evaluando **todo
    train en modo eval**, igual que validation. No usar el ``train_acc`` del history: se
    acumula con Dropout activo, BN en modo train y pesos cambiando, que son justamente
    las variables que se comparan. Un gap positivo indica que el modelo acierta más en
    train que en validation.

    Args:
        train_acc_eval: Accuracy de train en modo eval, en ``[0, 1]``.
        val_acc: Accuracy de validation en la mejor época, en ``[0, 1]``.

    Returns:
        El gap, en ``[-1, 1]``.

    Raises:
        ValueError: Si alguna accuracy está fuera de ``[0, 1]`` o no es finita.
    """
    for name, value in (("train_acc_eval", train_acc_eval), ("val_acc", val_acc)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} debe estar en [0, 1]; se recibió {value}")
    return train_acc_eval - val_acc
