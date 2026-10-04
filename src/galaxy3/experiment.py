"""Orquestación de un run del Experimento 1: ``run_experiment`` y ``evaluate_on_test``.

``run_experiment`` sigue el flujo de la spec (sección 9)::

    config -> train_seed -> device -> loaders (train, val) -> model -> loss -> optimizer
    -> fit -> recargar best checkpoint -> evaluar train y val en modo eval -> metrics
    -> plots -> archivos en outputs/

Nunca construye ni recorre el loader de test. La evaluación final en test es una función
aparte, explícita, que se llama **una sola vez** después de elegir la configuración con
validation (sección 17).

Salidas por run (``<run_name>`` como prefijo, para que runs paralelos no colisionen)::

    outputs/checkpoints/<run_name>_best.pt
    outputs/figures/<run_name>_training_curves.png
    outputs/figures/<run_name>_val_confusion_matrix.png
    outputs/metrics/<run_name>.json
    # solo con evaluate_on_test:
    outputs/figures/<run_name>_test_confusion_matrix.png
    outputs/figures/<run_name>_test_errors.png
    outputs/metrics/<run_name>_test.json

Este módulo todavía no registra nada en MLflow (Fase 9): devuelve resultados
estructurados cuyo ``to_dict()`` / ``final_metrics()`` usará esa integración.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from torch import nn
from torch.utils.data import DataLoader

from galaxy3.config import ExperimentConfig
from galaxy3.data import load_dataset_stats, make_split_loader
from galaxy3.engine import History, fit
from galaxy3.metrics import (
    ClassificationReport,
    Predictions,
    classification_report,
    collect_predictions,
    generalization_gap,
    misclassified_indices,
)
from galaxy3.models import GalaxyCNN, count_parameters, count_trainable_parameters
from galaxy3.plots import plot_confusion_matrix, plot_predictions, plot_training_curves
from galaxy3.utils import OutputDirs, get_device, prepare_output_dirs, set_seed

MODEL_NAME = "GalaxyCNN"
OPTIMIZER_NAME = "AdamW"
MAX_ERROR_EXAMPLES = 16
FIGURE_DPI = 120

CHECKPOINT_COMPATIBLE_FIELDS: tuple[str, ...] = (
    "split_seed",
    "num_classes",
    "image_size",
    "center_crop",
    "use_batchnorm",
    "dropout_p",
)
"""Campos que deben coincidir entre el checkpoint y la config de ``evaluate_on_test``."""


@dataclass(frozen=True)
class SplitEvaluation:
    """Evaluación de un split en modo eval con el best checkpoint.

    Attributes:
        loss: Pérdida promedio por muestra.
        report: Accuracy, matriz de confusión, métricas por clase y macro F1.
    """

    loss: float
    report: ClassificationReport

    def to_dict(self, class_names: list[str]) -> dict[str, Any]:
        """Serializa a tipos primitivos (apto para JSON)."""
        report = self.report
        per_class = {
            name: {
                "precision": float(report.precision[k]),
                "recall": float(report.recall[k]),
                "f1": float(report.f1[k]),
            }
            for k, name in enumerate(class_names)
        }
        return {
            "loss": self.loss,
            "accuracy": report.accuracy,
            "macro_f1": report.macro_f1,
            "per_class": per_class,
            "confusion_matrix": report.confusion_matrix.tolist(),
        }


@dataclass(frozen=True)
class ExperimentResult:
    """Resultado de ``run_experiment``. No contiene nada de test.

    Attributes:
        config: Configuración del run.
        device: Device usado (``"cuda"``, ``"mps"`` o ``"cpu"``).
        class_names: Nombres de clase (índice = etiqueta).
        history: Historia de ``fit``.
        total_parameters: Parámetros totales del modelo.
        trainable_parameters: Parámetros entrenables.
        train_samples: Imágenes de train.
        val_samples: Imágenes de validation.
        train_eval: Train evaluado en modo eval con el best checkpoint.
        val_eval: Validation evaluado en modo eval con el best checkpoint.
        generalization_gap: ``train_eval.accuracy - val_eval.accuracy`` (sección 17).
        checkpoint_path: Best checkpoint (menor ``val_loss``).
        metrics_path: JSON con ``to_dict()``.
        figure_paths: ``{nombre: ruta}`` de las figuras guardadas.
    """

    config: ExperimentConfig
    device: str
    class_names: list[str]
    history: History
    total_parameters: int
    trainable_parameters: int
    train_samples: int
    val_samples: int
    train_eval: SplitEvaluation
    val_eval: SplitEvaluation
    generalization_gap: float
    checkpoint_path: Path
    metrics_path: Path
    figure_paths: dict[str, Path]

    def final_metrics(self) -> dict[str, float]:
        """Métricas finales del run, con los nombres de la sección 14 de la spec."""
        return {
            "best_epoch": float(self.history["best_epoch"]),
            "best_val_loss": self.history["best_val_loss"],
            "train_loss_eval": self.train_eval.loss,
            "train_acc_eval": self.train_eval.report.accuracy,
            "val_acc_best": self.val_eval.report.accuracy,
            "val_macro_f1": self.val_eval.report.macro_f1,
            "generalization_gap": self.generalization_gap,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializa a tipos primitivos (apto para JSON)."""
        return {
            "config": self.config.to_dict(),
            "model_name": MODEL_NAME,
            "optimizer": OPTIMIZER_NAME,
            "device": self.device,
            "class_names": self.class_names,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "train_samples": self.train_samples,
            "val_samples": self.val_samples,
            "history": dict(self.history),
            "final_metrics": self.final_metrics(),
            "train_eval": self.train_eval.to_dict(self.class_names),
            "val_eval": self.val_eval.to_dict(self.class_names),
            "checkpoint_path": str(self.checkpoint_path),
            "figure_paths": {name: str(path) for name, path in self.figure_paths.items()},
        }


@dataclass(frozen=True)
class FinalEvaluation:
    """Resultado de ``evaluate_on_test``.

    Attributes:
        checkpoint_path: Checkpoint evaluado.
        checkpoint_epoch: Época del checkpoint (la de menor ``val_loss``).
        class_names: Nombres de clase.
        test_samples: Imágenes de test.
        test_eval: Métricas de test en modo eval.
        metrics_path: JSON con ``to_dict()``.
        figure_paths: ``{nombre: ruta}``; ``test_errors`` falta si no hubo errores.
    """

    checkpoint_path: Path
    checkpoint_epoch: int
    class_names: list[str]
    test_samples: int
    test_eval: SplitEvaluation
    metrics_path: Path
    figure_paths: dict[str, Path]

    def final_metrics(self) -> dict[str, float]:
        """Métricas escalares de test."""
        return {
            "test_loss": self.test_eval.loss,
            "test_acc": self.test_eval.report.accuracy,
            "test_macro_f1": self.test_eval.report.macro_f1,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serializa a tipos primitivos (apto para JSON)."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint_epoch,
            "class_names": self.class_names,
            "test_samples": self.test_samples,
            "final_metrics": self.final_metrics(),
            "test_eval": self.test_eval.to_dict(self.class_names),
            "figure_paths": {name: str(path) for name, path in self.figure_paths.items()},
        }


def build_model(config: ExperimentConfig) -> GalaxyCNN:
    """Construye la GalaxyCNN descrita por ``config`` (en CPU, sin mover al device)."""
    return GalaxyCNN(
        num_classes=config.num_classes,
        use_batchnorm=config.use_batchnorm,
        dropout_p=config.dropout_p,
    )


def checkpoint_path_for(config: ExperimentConfig) -> Path:
    """Ruta del best checkpoint del run: ``output_root/checkpoints/<run_name>_best.pt``.

    Raises:
        ValueError: Si ``run_name`` no sirve como nombre de archivo.
    """
    return config.output_root / "checkpoints" / f"{_file_prefix(config)}_best.pt"


def _file_prefix(config: ExperimentConfig) -> str:
    """``run_name`` como prefijo de archivo; rechaza separadores de ruta."""
    name = config.run_name
    if Path(name).name != name or name in (".", ".."):
        raise ValueError(f"run_name no puede contener separadores de ruta: {name!r}")
    return name


def _num_samples(loader: DataLoader) -> int:
    return len(loader.dataset)  # type: ignore[arg-type]


def _load_weights(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    """Carga ``model_state_dict`` en ``model`` (``weights_only=True``); devuelve el checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def _evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[SplitEvaluation, Predictions]:
    """Evalúa ``loader`` en modo eval y calcula su reporte de clasificación."""
    predictions = collect_predictions(model, loader, loss_fn, device)
    report = classification_report(predictions.y_true, predictions.y_pred, num_classes)
    return SplitEvaluation(loss=predictions.loss, report=report), predictions


def _save_figures(figures: dict[str, Figure], directory: Path, prefix: str) -> dict[str, Path]:
    """Guarda cada figura como ``<prefix>_<nombre>.png`` y la cierra."""
    paths = {}
    for name, fig in figures.items():
        path = directory / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=FIGURE_DPI)
        plt.close(fig)
        paths[name] = path
    return paths


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _train(
    config: ExperimentConfig,
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
    checkpoint_path: Path,
    verbose: bool,
) -> tuple[nn.Module, History]:
    """Protocolo de la sección 15: CrossEntropy + AdamW, épocas fijas, sin scheduler."""
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    train_loader, val_loader = loaders
    history = fit(
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        config,
        checkpoint_path,
        verbose=verbose,
    )
    return loss_fn, history


@dataclass(frozen=True)
class RunData:
    """Datos intermedios de ``run_experiment`` que ``_summarize_run`` convierte en salidas."""

    device: str
    class_names: list[str]
    history: History
    model: nn.Module
    train_samples: int
    val_samples: int
    train_eval: SplitEvaluation
    val_eval: SplitEvaluation
    checkpoint_path: Path


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """Entrena, selecciona por ``val_loss`` y evalúa un run, **sin tocar test**.

    Pasos: siembra ``train_seed``; resuelve el device; construye solo los loaders de
    train y val; entrena ``config.epochs`` épocas con ``CrossEntropyLoss`` y
    ``AdamW(lr=learning_rate, weight_decay=weight_decay)``; recarga el best checkpoint con
    ``torch.load(..., weights_only=True)``; evalúa **todo train y val en modo eval** con
    esos pesos; calcula métricas y el generalization gap (``train_acc_eval - val_acc``);
    guarda figuras y un JSON de métricas en ``config.output_root``.

    Args:
        config: Configuración del run. ``run_name`` es el prefijo de los archivos de
            salida: un run con el mismo ``run_name`` los sobrescribe.
        verbose: Si es ``True``, ``fit`` imprime una línea por época.

    Returns:
        El resultado estructurado del run (ver ``ExperimentResult``).

    Raises:
        FileNotFoundError: Si falta ``dataset_stats.json`` en ``config.data_root``.
        ValueError: Si ``split_seed``, ``image_size`` o ``center_crop`` no coinciden con
            ``dataset_stats.json``, si las clases no coinciden, o si ``run_name`` contiene
            separadores de ruta.
        RuntimeError: Si un device pedido explícitamente no está disponible, o si
            ``val_loss`` no fue finita en ninguna época.
    """
    checkpoint_path = checkpoint_path_for(config)
    set_seed(config.train_seed)
    device = get_device(config.device_preference)
    train_loader = make_split_loader(config, "train", device)
    val_loader = make_split_loader(config, "val", device)
    model = build_model(config).to(device)
    dirs = prepare_output_dirs(config.output_root)

    loss_fn, history = _train(
        config, model, (train_loader, val_loader), device, checkpoint_path, verbose
    )
    _load_weights(model, checkpoint_path, device)
    train_eval, _ = _evaluate_split(model, train_loader, loss_fn, device, config.num_classes)
    val_eval, _ = _evaluate_split(model, val_loader, loss_fn, device, config.num_classes)

    class_names = list(train_loader.dataset.classes)  # type: ignore[attr-defined]
    return _summarize_run(
        config,
        dirs,
        RunData(
            device=device.type,
            class_names=class_names,
            history=history,
            model=model,
            train_samples=_num_samples(train_loader),
            val_samples=_num_samples(val_loader),
            train_eval=train_eval,
            val_eval=val_eval,
            checkpoint_path=checkpoint_path,
        ),
    )


def _summarize_run(config: ExperimentConfig, dirs: OutputDirs, data: RunData) -> ExperimentResult:
    """Calcula el gap, guarda figuras y JSON, y arma el ``ExperimentResult``."""
    prefix = _file_prefix(config)
    figures = {
        "training_curves": plot_training_curves(data.history),
        "val_confusion_matrix": plot_confusion_matrix(
            data.val_eval.report.confusion_matrix,
            data.class_names,
            title=f"Validation — mejor época ({data.history['best_epoch']})",
        ),
    }
    result = ExperimentResult(
        config=config,
        device=data.device,
        class_names=data.class_names,
        history=data.history,
        total_parameters=count_parameters(data.model),
        trainable_parameters=count_trainable_parameters(data.model),
        train_samples=data.train_samples,
        val_samples=data.val_samples,
        train_eval=data.train_eval,
        val_eval=data.val_eval,
        generalization_gap=generalization_gap(
            data.train_eval.report.accuracy, data.val_eval.report.accuracy
        ),
        checkpoint_path=data.checkpoint_path,
        metrics_path=dirs.metrics / f"{prefix}.json",
        figure_paths=_save_figures(figures, dirs.figures, prefix),
    )
    _write_json(result.metrics_path, result.to_dict())
    return result


def _check_checkpoint_matches(checkpoint_config: dict[str, Any], config: ExperimentConfig) -> None:
    """Exige que el checkpoint y ``config`` describan el mismo modelo y los mismos datos."""
    current = config.to_dict()
    errors = [
        f"{name}: checkpoint={checkpoint_config.get(name)!r}, config={current[name]!r}"
        for name in CHECKPOINT_COMPATIBLE_FIELDS
        if checkpoint_config.get(name) != current[name]
    ]
    if errors:
        raise ValueError("El checkpoint no corresponde a la configuración: " + "; ".join(errors))


def _plot_test_errors(
    predictions: Predictions,
    loader: DataLoader,
    class_names: list[str],
    stats: dict[str, Any],
) -> Figure | None:
    """Errores de test con mayor confianza primero; ``None`` si no hubo errores."""
    indices = misclassified_indices(predictions, MAX_ERROR_EXAMPLES)
    if indices.numel() == 0:
        return None
    X = torch.stack([loader.dataset[i][0] for i in indices.tolist()])
    return plot_predictions(
        X,
        predictions.y_true[indices],
        predictions.probabilities[indices],
        class_names,
        stats["mean_train"],
        stats["std_train"],
        title="Errores en test (mayor confianza primero)",
    )


def evaluate_on_test(checkpoint_path: Path, config: ExperimentConfig) -> FinalEvaluation:
    """Evaluación final en test del checkpoint elegido con validation.

    **Llamar una sola vez**, después de seleccionar la configuración con validation
    (sección 17). Carga el checkpoint con ``torch.load(..., weights_only=True)``, construye
    solo el loader de test, evalúa en modo eval y guarda la matriz de confusión, ejemplos
    de errores y un JSON de métricas en ``config.output_root``. No modifica el checkpoint.

    Args:
        checkpoint_path: Best checkpoint producido por ``run_experiment``.
        config: Configuración del run que produjo el checkpoint. Los campos de
            ``CHECKPOINT_COMPATIBLE_FIELDS`` deben coincidir con los guardados en él;
            ``data_root``, ``output_root``, ``batch_size``, ``num_workers`` y
            ``device_preference`` pueden diferir.

    Returns:
        Métricas y rutas de la evaluación final (ver ``FinalEvaluation``).

    Raises:
        FileNotFoundError: Si no existe el checkpoint o ``dataset_stats.json``.
        ValueError: Si el checkpoint no corresponde a ``config`` o los datos no coinciden
            con ``dataset_stats.json``.
    """
    prefix = _file_prefix(config)
    device = get_device(config.device_preference)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _check_checkpoint_matches(checkpoint["config"], config)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loader = make_split_loader(config, "test", device)
    class_names = list(test_loader.dataset.classes)  # type: ignore[attr-defined]
    test_eval, predictions = _evaluate_split(
        model, test_loader, nn.CrossEntropyLoss(), device, config.num_classes
    )

    figures = {
        "test_confusion_matrix": plot_confusion_matrix(
            test_eval.report.confusion_matrix, class_names, title="Test — evaluación final"
        )
    }
    stats = load_dataset_stats(config.data_root)
    errors = _plot_test_errors(predictions, test_loader, class_names, stats)
    if errors is not None:
        figures["test_errors"] = errors

    dirs = prepare_output_dirs(config.output_root)
    result = FinalEvaluation(
        checkpoint_path=Path(checkpoint_path),
        checkpoint_epoch=int(checkpoint["epoch"]),
        class_names=class_names,
        test_samples=_num_samples(test_loader),
        test_eval=test_eval,
        metrics_path=dirs.metrics / f"{prefix}_test.json",
        figure_paths=_save_figures(figures, dirs.figures, prefix),
    )
    _write_json(result.metrics_path, result.to_dict())
    return result
