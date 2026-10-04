"""Configuración tipada del experimento Galaxy3."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from galaxy3.utils import PROJECT_ROOT, VALID_DEVICE_PREFERENCES

_PATH_FIELDS: tuple[str, ...] = ("data_root", "output_root")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuración de un run.

    Es inmutable: para variantes usar ``dataclasses.replace(config, ...)``.
    Los valores por defecto son puntos de partida, no óptimos.

    Attributes:
        run_name: Nombre del run (MLflow y archivos de salida).
        split_seed: Semilla del split train/val/test. Solo la usa la preparación de
            datos; debe coincidir con la registrada en ``dataset_stats.json``.
        train_seed: Semilla de inicialización, orden de batches y máscaras de Dropout.
            Cambiarla nunca cambia el split.
        deterministic: Si se fuerzan algoritmos deterministas de cuDNN (ver
            ``utils.set_seed``). Puede hacer más lento el entrenamiento en GPU.
        num_classes: Número de clases (>= 2).
        image_size: Lado de la imagen cuadrada de entrada al modelo, en píxeles.
        center_crop: Lado del ``CenterCrop`` previo al ``Resize``, o ``None`` (baseline).
        batch_size: Tamaño de batch (>= 1).
        num_workers: Workers del ``DataLoader`` (>= 0; 0 = proceso principal).
        use_batchnorm: Si se usa ``BatchNorm2d`` tras cada convolución.
        dropout_p: Probabilidad de Dropout en el clasificador, en ``[0, 1)``.
        learning_rate: Learning rate de AdamW (> 0).
        weight_decay: Weight decay de AdamW (>= 0).
        epochs: Número fijo de épocas (>= 1). Provisional hasta el piloto.
        device_preference: ``"auto"``, ``"cuda"``, ``"mps"`` o ``"cpu"``.
        data_root: Carpeta con ``train/``, ``val/`` y ``test/`` en formato ``ImageFolder``.
        output_root: Carpeta de checkpoints, figuras y métricas.
        mlflow_experiment_name: Experimento MLflow donde se registra el run.

    Raises:
        ValueError: Si algún campo está fuera de su dominio.
    """

    run_name: str = "baseline"
    split_seed: int = 2026
    train_seed: int = 2026
    deterministic: bool = False
    num_classes: int = 3
    image_size: int = 128
    center_crop: int | None = None
    batch_size: int = 64
    num_workers: int = 8
    use_batchnorm: bool = False
    dropout_p: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 30
    device_preference: str = "auto"
    data_root: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "galaxy3")
    output_root: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    mlflow_experiment_name: str = "galaxy3-cnn-from-scratch"

    def __post_init__(self) -> None:
        for name in _PATH_FIELDS:
            object.__setattr__(self, name, Path(getattr(self, name)))
        errors = [message for invalid, message in self._checks() if invalid]
        if errors:
            raise ValueError("ExperimentConfig inválida: " + "; ".join(errors))

    def _checks(self) -> list[tuple[bool, str]]:
        return [
            (not self.run_name.strip(), "run_name no puede estar vacío"),
            (self.split_seed < 0, f"split_seed debe ser >= 0 (es {self.split_seed})"),
            (self.train_seed < 0, f"train_seed debe ser >= 0 (es {self.train_seed})"),
            (self.num_classes < 2, f"num_classes debe ser >= 2 (es {self.num_classes})"),
            (self.image_size < 1, f"image_size debe ser >= 1 (es {self.image_size})"),
            (
                self.center_crop is not None and self.center_crop < 1,
                f"center_crop debe ser None o >= 1 (es {self.center_crop})",
            ),
            (self.batch_size < 1, f"batch_size debe ser >= 1 (es {self.batch_size})"),
            (self.num_workers < 0, f"num_workers debe ser >= 0 (es {self.num_workers})"),
            (
                not 0.0 <= self.dropout_p < 1.0,
                f"dropout_p debe estar en [0, 1) (es {self.dropout_p})",
            ),
            (self.learning_rate <= 0, f"learning_rate debe ser > 0 (es {self.learning_rate})"),
            (self.weight_decay < 0, f"weight_decay debe ser >= 0 (es {self.weight_decay})"),
            (self.epochs < 1, f"epochs debe ser >= 1 (es {self.epochs})"),
            (
                self.device_preference not in VALID_DEVICE_PREFERENCES,
                f"device_preference debe ser uno de {VALID_DEVICE_PREFERENCES} "
                f"(es {self.device_preference!r})",
            ),
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serializa a un ``dict`` de tipos primitivos (``Path`` -> ``str``).

        Apto para el checkpoint (``torch.load(..., weights_only=True)``) y para MLflow.
        """
        data = asdict(self)
        for name in _PATH_FIELDS:
            data[name] = str(data[name])
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentConfig:
        """Reconstruye la configuración desde ``to_dict()`` (por ejemplo, desde un checkpoint).

        Raises:
            ValueError: Si ``data`` contiene claves desconocidas o valores inválidos.
        """
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Claves desconocidas en la configuración: {sorted(unknown)}")
        return cls(**data)
