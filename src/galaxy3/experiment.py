"""Orquestación de un run del Experimento 1: ``run_experiment`` y ``evaluate_on_test``.

``run_experiment`` sigue el flujo de la spec (sección 9)::

    config -> train_seed -> device -> loaders (train, val) -> model -> loss -> optimizer
    -> fit -> recargar best checkpoint -> evaluar train y val en modo eval -> metrics
    -> plots -> archivos en outputs/ -> MLflow

Nunca construye ni recorre el loader de test. La evaluación final en test es una función
aparte, explícita, que se llama **una sola vez** después de elegir la configuración con
validation (sección 17), y registra sus resultados en el mismo run de MLflow.

Salidas por run (``<run_name>`` como prefijo, para que runs paralelos no colisionen)::

    outputs/checkpoints/<run_name>_best.pt
    outputs/checkpoints/<run_name>_best.mlflow.json   # run_id del run que lo produjo
    outputs/figures/<run_name>_training_curves.png
    outputs/figures/<run_name>_val_confusion_matrix.png
    outputs/metrics/<run_name>.json
    # solo con evaluate_on_test:
    outputs/figures/<run_name>_test_confusion_matrix.png
    outputs/figures/<run_name>_test_errors.png
    outputs/metrics/<run_name>_test.json

MLflow (sección 14) se usa **solo en este módulo**, con logging explícito (sin autolog):
params al iniciar el run; métricas por época (``step=epoch``) recorriendo el history
después de ``fit``; métricas finales sin step; config, métricas, figuras y checkpoint
como artifacts. ``engine``, ``metrics`` y ``plots`` no conocen MLflow.

Tracking URI: ``MLFLOW_TRACKING_URI`` si existe; si no, ``sqlite:///<galaxy3>/mlflow.db``
(el backend local por defecto de MLflow 3). Con un backend SQLite los artifacts se guardan
en ``mlruns/`` junto a la base, sin depender del directorio actual.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import torch
from matplotlib.figure import Figure
from mlflow.exceptions import MlflowException
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
from galaxy3.utils import PROJECT_ROOT, OutputDirs, get_device, prepare_output_dirs, set_seed

MODEL_NAME = "GalaxyCNN"
OPTIMIZER_NAME = "AdamW"
MAX_ERROR_EXAMPLES = 16
FIGURE_DPI = 120

LOCAL_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
"""Backend local por defecto: SQLite dentro de ``galaxy3/`` (ignorado por git)."""

EPOCH_METRICS: dict[str, str] = {
    "train_loss": "train_loss",
    "train_acc": "train_acc",
    "val_loss": "val_loss",
    "val_acc": "val_acc",
    "learning_rate": "lr",
    "epoch_time": "epoch_time",
}
"""Nombre de la métrica MLflow por época -> clave del history."""

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
        run_id: Run de MLflow donde quedó registrado.
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
    run_id: str

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
            "run_id": self.run_id,
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
        run_id: Run de MLflow (el que produjo el checkpoint) donde se registró.
    """

    checkpoint_path: Path
    checkpoint_epoch: int
    class_names: list[str]
    test_samples: int
    test_eval: SplitEvaluation
    metrics_path: Path
    figure_paths: dict[str, Path]
    run_id: str

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
            "run_id": self.run_id,
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


# --------------------------------------------------------------------------- MLflow


def resolve_tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` si está definida; si no, ``LOCAL_TRACKING_URI`` (SQLite local)."""
    return os.environ.get("MLFLOW_TRACKING_URI") or LOCAL_TRACKING_URI


def _artifact_location(tracking_uri: str) -> str | None:
    """Con SQLite, ``mlruns/`` junto a la base; con otros backends, el default del servidor."""
    prefix = "sqlite:///"
    if not tracking_uri.startswith(prefix):
        return None
    database = Path(tracking_uri.removeprefix(prefix)).resolve()
    return (database.parent / "mlruns").as_uri()


def _use_experiment(tracking_uri: str, name: str) -> str:
    """Apunta MLflow a ``tracking_uri`` y devuelve el id del experimento ``name`` (lo crea).

    Si dos procesos lo crean a la vez, el segundo recibe un error y simplemente lo lee.
    """
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    try:
        return mlflow.create_experiment(name, artifact_location=_artifact_location(tracking_uri))
    except MlflowException:
        experiment = mlflow.get_experiment_by_name(name)
        if experiment is None:
            raise
        return experiment.experiment_id


def run_reference_path(checkpoint_path: Path) -> Path:
    """Archivo junto al checkpoint con el run de MLflow que lo produjo.

    ``<run_name>_best.pt`` -> ``<run_name>_best.mlflow.json``.
    """
    return Path(checkpoint_path).with_suffix(".mlflow.json")


def _write_run_reference(checkpoint_path: Path, run_id: str, tracking_uri: str) -> None:
    _write_json(
        run_reference_path(checkpoint_path),
        {"run_id": run_id, "tracking_uri": tracking_uri},
    )


def _read_run_reference(checkpoint_path: Path) -> dict[str, str]:
    """Lee el run de MLflow del checkpoint y exige que esté en el backend actual.

    Raises:
        FileNotFoundError: Si el checkpoint no tiene archivo de referencia.
        ValueError: Si el run se registró en otro tracking URI.
    """
    path = run_reference_path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"No existe {path}: el checkpoint no fue producido por run_experiment."
        )
    reference = json.loads(path.read_text())
    current = resolve_tracking_uri()
    if reference["tracking_uri"] != current:
        raise ValueError(
            f"El run {reference['run_id']} se registró en {reference['tracking_uri']!r}, "
            f"pero el tracking URI actual es {current!r}. Define MLFLOW_TRACKING_URI."
        )
    return reference


def _run_params(
    config: ExperimentConfig,
    device: torch.device,
    model: nn.Module,
    sample_counts: dict[str, int],
) -> dict[str, Any]:
    """Params del run (sección 14): la config completa más lo que se conoce al iniciar."""
    return {
        **config.to_dict(),
        "model_name": MODEL_NAME,
        "optimizer": OPTIMIZER_NAME,
        "device": device.type,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_trainable_parameters(model),
        **{f"{split}_samples": n for split, n in sample_counts.items()},
    }


def _split_counts_from_stats(stats: dict[str, Any]) -> dict[str, int]:
    """Imágenes por split según ``dataset_stats.json``: no abre ninguna carpeta (ni test)."""
    return {split: sum(counts.values()) for split, counts in stats["counts"].items()}


def _log_run_results(result: ExperimentResult) -> None:
    """Registra en el run activo las métricas por época, las finales y los artifacts."""
    history = result.history
    for index in range(len(history["val_loss"])):
        mlflow.log_metrics(
            {name: float(history[key][index]) for name, key in EPOCH_METRICS.items()},
            step=index + 1,
        )
    mlflow.log_metrics(result.final_metrics())
    mlflow.log_dict(result.config.to_dict(), "config.json")
    mlflow.log_artifact(str(result.metrics_path), "metrics")
    mlflow.log_artifact(str(result.checkpoint_path), "checkpoints")
    for path in result.figure_paths.values():
        mlflow.log_artifact(str(path), "figures")


def _check_not_evaluated_on_test(reference: dict[str, str]) -> None:
    """Exige que el run del checkpoint todavía no tenga métricas de test.

    Raises:
        RuntimeError: Si el run ya fue evaluado en test (``test_acc`` registrada).
    """
    mlflow.set_tracking_uri(reference["tracking_uri"])
    run_id = reference["run_id"]
    if "test_acc" in mlflow.get_run(run_id).data.metrics:
        raise RuntimeError(
            f"El run {run_id} ya tiene métricas de test: evaluate_on_test se usa una sola vez."
        )


def _log_test_results(result: FinalEvaluation) -> None:
    """Reabre el run del checkpoint y registra las métricas y artifacts de test.

    El run se reabre solo para registrar: si la evaluación fallara antes, el run de
    entrenamiento conserva su estado ``FINISHED``.
    """
    with mlflow.start_run(run_id=result.run_id):
        mlflow.log_metrics(result.final_metrics())
        mlflow.log_artifact(str(result.metrics_path), "test")
        for path in result.figure_paths.values():
            mlflow.log_artifact(str(path), "test")


# ---------------------------------------------------------------- entrenamiento


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
    run_id: str


def _train_and_evaluate(
    config: ExperimentConfig,
    model: nn.Module,
    loaders: tuple[DataLoader, DataLoader],
    device: torch.device,
    run_id: str,
    verbose: bool,
) -> RunData:
    """``fit`` -> recargar el best checkpoint -> evaluar train y val en modo eval."""
    checkpoint_path = checkpoint_path_for(config)
    train_loader, val_loader = loaders
    loss_fn, history = _train(config, model, loaders, device, checkpoint_path, verbose)
    _load_weights(model, checkpoint_path, device)
    train_eval, _ = _evaluate_split(model, train_loader, loss_fn, device, config.num_classes)
    val_eval, _ = _evaluate_split(model, val_loader, loss_fn, device, config.num_classes)
    return RunData(
        device=device.type,
        class_names=list(train_loader.dataset.classes),  # type: ignore[attr-defined]
        history=history,
        model=model,
        train_samples=_num_samples(train_loader),
        val_samples=_num_samples(val_loader),
        train_eval=train_eval,
        val_eval=val_eval,
        checkpoint_path=checkpoint_path,
        run_id=run_id,
    )


def run_experiment(config: ExperimentConfig, verbose: bool = True) -> ExperimentResult:
    """Entrena, selecciona por ``val_loss``, evalúa y registra un run, **sin tocar test**.

    Pasos: siembra ``train_seed``; resuelve el device; construye solo los loaders de
    train y val; abre un run de MLflow (``run_name`` = ``config.run_name``) en el
    experimento ``config.mlflow_experiment_name`` y registra los params; entrena
    ``config.epochs`` épocas con ``CrossEntropyLoss`` y
    ``AdamW(lr=learning_rate, weight_decay=weight_decay)``; recarga el best checkpoint con
    ``torch.load(..., weights_only=True)``; evalúa **todo train y val en modo eval** con
    esos pesos; calcula métricas y el generalization gap (``train_acc_eval - val_acc``);
    guarda figuras, un JSON de métricas y el ``run_id`` (junto al checkpoint) en
    ``config.output_root``; registra métricas y artifacts en MLflow. Si algo falla
    dentro del run, MLflow lo marca como ``FAILED``.

    ``test_samples`` se registra como param leyendo los conteos de ``dataset_stats.json``;
    la carpeta de test no se abre.

    Args:
        config: Configuración del run. ``run_name`` es el prefijo de los archivos de
            salida: un run con el mismo ``run_name`` los sobrescribe (en MLflow queda un
            run nuevo).
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
    loaders = (
        make_split_loader(config, "train", device),
        make_split_loader(config, "val", device),
    )
    model = build_model(config).to(device)
    dirs = prepare_output_dirs(config.output_root)
    sample_counts = _split_counts_from_stats(load_dataset_stats(config.data_root))

    tracking_uri = resolve_tracking_uri()
    experiment_id = _use_experiment(tracking_uri, config.mlflow_experiment_name)
    with mlflow.start_run(experiment_id=experiment_id, run_name=config.run_name) as run:
        mlflow.log_params(_run_params(config, device, model, sample_counts))
        data = _train_and_evaluate(config, model, loaders, device, run.info.run_id, verbose)
        result = _summarize_run(config, dirs, data)
        _write_run_reference(checkpoint_path, run.info.run_id, tracking_uri)
        _log_run_results(result)
    return result


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
        run_id=data.run_id,
    )
    _write_json(result.metrics_path, result.to_dict())
    return result


# ------------------------------------------------------------- evaluación final


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


def _evaluate_test_split(
    config: ExperimentConfig, model: nn.Module, device: torch.device
) -> tuple[SplitEvaluation, list[str], int, dict[str, Figure]]:
    """Construye el loader de test, lo evalúa en modo eval y arma sus figuras."""
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
    return test_eval, class_names, _num_samples(test_loader), figures


def evaluate_on_test(checkpoint_path: Path, config: ExperimentConfig) -> FinalEvaluation:
    """Evaluación final en test del checkpoint elegido con validation.

    **Llamar una sola vez**, después de seleccionar la configuración con validation
    (sección 17). Carga el checkpoint con ``torch.load(..., weights_only=True)``, construye
    solo el loader de test, evalúa en modo eval y guarda la matriz de confusión, ejemplos
    de errores y un JSON de métricas en ``config.output_root``. Registra métricas
    (``test_loss``, ``test_acc``, ``test_macro_f1``) y artifacts (carpeta ``test/``) en el
    run de MLflow que produjo el checkpoint, leído de ``run_reference_path``. No modifica
    el checkpoint.

    Args:
        checkpoint_path: Best checkpoint producido por ``run_experiment``.
        config: Configuración del run que produjo el checkpoint. Los campos de
            ``CHECKPOINT_COMPATIBLE_FIELDS`` deben coincidir con los guardados en él;
            ``data_root``, ``output_root``, ``batch_size``, ``num_workers`` y
            ``device_preference`` pueden diferir.

    Returns:
        Métricas y rutas de la evaluación final (ver ``FinalEvaluation``).

    Raises:
        FileNotFoundError: Si no existe el checkpoint, su referencia de MLflow o
            ``dataset_stats.json``.
        ValueError: Si el checkpoint no corresponde a ``config``, los datos no coinciden
            con ``dataset_stats.json`` o el run está en otro tracking URI.
        RuntimeError: Si el run ya fue evaluado en test. La comprobación ocurre antes de
            abrir la carpeta de test.
    """
    prefix = _file_prefix(config)
    device = get_device(config.device_preference)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _check_checkpoint_matches(checkpoint["config"], config)
    reference = _read_run_reference(checkpoint_path)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _check_not_evaluated_on_test(reference)

    test_eval, class_names, test_samples, figures = _evaluate_test_split(config, model, device)
    dirs = prepare_output_dirs(config.output_root)
    result = FinalEvaluation(
        checkpoint_path=Path(checkpoint_path),
        checkpoint_epoch=int(checkpoint["epoch"]),
        class_names=class_names,
        test_samples=test_samples,
        test_eval=test_eval,
        metrics_path=dirs.metrics / f"{prefix}_test.json",
        figure_paths=_save_figures(figures, dirs.figures, prefix),
        run_id=reference["run_id"],
    )
    _write_json(result.metrics_path, result.to_dict())
    _log_test_results(result)
    return result
