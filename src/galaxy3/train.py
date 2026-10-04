"""Entry point CLI del Experimento 1: ``uv run python -m galaxy3.train``.

Solo traduce argumentos de línea de comandos a un ``ExperimentConfig`` y llama a
``run_experiment`` (sección 9). No contiene training loop ni evalúa test: la evaluación
final es ``evaluate_on_test``, explícita y posterior a la selección con validation.

Los defaults viven únicamente en ``ExperimentConfig``. Los flags usan
``argparse.SUPPRESS``: un flag omitido no llega a la configuración, y el ``--help``
muestra el default leyéndolo de los campos de la dataclass.

Ejemplo::

    uv run python -m galaxy3.train \\
        --run-name batchnorm_dropout --use-batchnorm --dropout-p 0.3 --train-seed 2026
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any

from galaxy3.config import ExperimentConfig
from galaxy3.experiment import ExperimentResult, resolve_tracking_uri, run_experiment
from galaxy3.utils import VALID_DEVICE_PREFERENCES

# flag -> (campo de ExperimentConfig, kwargs de add_argument, ayuda)
_FLAGS: dict[str, tuple[str, dict[str, Any], str]] = {
    "--run-name": ("run_name", {"type": str}, "nombre del run en MLflow y prefijo de archivos"),
    "--use-batchnorm": (
        "use_batchnorm",
        {"action": argparse.BooleanOptionalAction},
        "BatchNorm2d tras cada convolución",
    ),
    "--dropout-p": ("dropout_p", {"type": float}, "Dropout del clasificador, en [0, 1)"),
    "--train-seed": ("train_seed", {"type": int}, "semilla de inicialización y batches"),
    "--epochs": ("epochs", {"type": int}, "número fijo de épocas"),
    "--learning-rate": ("learning_rate", {"type": float}, "learning rate de AdamW"),
    "--batch-size": ("batch_size", {"type": int}, "tamaño de batch"),
    "--num-workers": ("num_workers", {"type": int}, "workers del DataLoader (0 = sin workers)"),
    "--device": (
        "device_preference",
        {"choices": VALID_DEVICE_PREFERENCES},
        "device preferido",
    ),
    "--center-crop": ("center_crop", {"type": int}, "lado del CenterCrop previo al Resize"),
    "--mlflow-experiment": ("mlflow_experiment_name", {"type": str}, "experimento MLflow"),
    "--data-root": ("data_root", {"type": Path}, "carpeta ImageFolder con dataset_stats.json"),
    "--output-root": ("output_root", {"type": Path}, "carpeta de checkpoints, figuras y métricas"),
}


def _config_defaults() -> dict[str, Any]:
    """Defaults de ``ExperimentConfig`` (solo para mostrarlos en ``--help``)."""
    defaults = {}
    for f in fields(ExperimentConfig):
        if f.default is not MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not MISSING:
            defaults[f.name] = f.default_factory()
    return defaults


def build_parser() -> argparse.ArgumentParser:
    """Construye el parser; los flags omitidos no aparecen en el ``Namespace``."""
    parser = argparse.ArgumentParser(
        prog="python -m galaxy3.train",
        description="Entrena GalaxyCNN desde cero (train + validation, sin test) y lo "
        "registra en MLflow.",
        argument_default=argparse.SUPPRESS,
    )
    defaults = _config_defaults()
    for flag, (field_name, kwargs, help_text) in _FLAGS.items():
        parser.add_argument(
            flag,
            dest=field_name,
            help=f"{help_text} (default: {defaults[field_name]})",
            **kwargs,
        )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    """Convierte argumentos CLI en un ``ExperimentConfig`` validado.

    Args:
        argv: Argumentos sin el nombre del programa; ``None`` usa ``sys.argv[1:]``.

    Returns:
        La configuración, con los defaults de ``ExperimentConfig`` en los campos omitidos.

    Raises:
        SystemExit: Con código 2 si un flag es desconocido o un valor es inválido
            (incluida la validación de ``ExperimentConfig``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return ExperimentConfig(**vars(args))
    except ValueError as error:
        parser.error(str(error))
        raise  # inalcanzable: parser.error lanza SystemExit


def format_summary(result: ExperimentResult) -> str:
    """Resumen final de un run, en texto plano."""
    history = result.history
    lines = [
        ("run_name", result.config.run_name),
        ("experimento", result.config.mlflow_experiment_name),
        ("run_id", result.run_id),
        ("tracking_uri", resolve_tracking_uri()),
        ("device", result.device),
        ("épocas", len(history["val_loss"])),
        ("best_epoch", history["best_epoch"]),
        ("best_val_loss", f"{history['best_val_loss']:.4f}"),
        ("val_acc", f"{result.val_eval.report.accuracy:.4f}"),
        ("val_macro_f1", f"{result.val_eval.report.macro_f1:.4f}"),
        ("train_acc_eval", f"{result.train_eval.report.accuracy:.4f}"),
        ("generalization_gap", f"{result.generalization_gap:+.4f}"),
        ("checkpoint", result.checkpoint_path),
    ]
    width = max(len(name) for name, _ in lines)
    body = "\n".join(f"  {name:<{width}}  {value}" for name, value in lines)
    return f"Resumen del run (sin test)\n{body}"


def main(argv: Sequence[str] | None = None) -> ExperimentResult:
    """Parsea argumentos, ejecuta ``run_experiment`` e imprime el resumen con el ``run_id``."""
    config = parse_config(argv)
    result = run_experiment(config)
    print(format_summary(result))
    return result


if __name__ == "__main__":
    main()
