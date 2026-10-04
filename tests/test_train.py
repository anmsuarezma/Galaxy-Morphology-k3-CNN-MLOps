"""Tests del entry point CLI ``python -m galaxy3.train``.

Parseo de flags, ``--help`` y una ejecución corta de punta a punta sobre un ImageFolder
sintético. El CLI no expone ``image_size`` (es parte del diseño de GalaxyCNN), así que las
imágenes sintéticas son de 128x128; con pocas imágenes, CPU y ``num_workers=0`` sigue
siendo rápido. El backend de MLflow es un SQLite temporal (ver ``conftest.py``).
"""

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402

from galaxy3 import train  # noqa: E402
from galaxy3.config import ExperimentConfig  # noqa: E402
from galaxy3.data import DATASET_STATS_NAME  # noqa: E402
from galaxy3.experiment import resolve_tracking_uri  # noqa: E402

CLASSES = ["barred_spiral", "edge_on_no_bulge", "round_smooth"]
PER_CLASS = {"train": 4, "val": 2, "test": 2}
IMAGE_SIZE = ExperimentConfig().image_size


def _make_data_root(root: Path) -> Path:
    """ImageFolder sintético de train y val (test solo existe en los conteos)."""
    rng = np.random.default_rng(0)
    for split in ("train", "val"):
        for name in CLASSES:
            folder = root / split / name
            folder.mkdir(parents=True)
            for i in range(PER_CLASS[split]):
                pixels = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i:05d}.png")
    stats = {
        "split_seed": ExperimentConfig().split_seed,
        "class_to_label": {name: i for i, name in enumerate(CLASSES)},
        "counts": {split: dict.fromkeys(CLASSES, n) for split, n in PER_CLASS.items()},
        "transform": {"image_size": IMAGE_SIZE, "center_crop": None},
        "mean_train": [0.5, 0.5, 0.5],
        "std_train": [0.25, 0.25, 0.25],
    }
    (root / DATASET_STATS_NAME).write_text(json.dumps(stats))
    return root


# ------------------------------------------------------------------ parseo


def test_no_flags_gives_config_defaults():
    assert train.parse_config([]) == ExperimentConfig()


def test_flags_map_to_config_fields(tmp_path):
    config = train.parse_config(
        [
            "--run-name", "batchnorm_dropout",
            "--use-batchnorm",
            "--dropout-p", "0.3",
            "--train-seed", "11",
            "--deterministic",
            "--epochs", "5",
            "--learning-rate", "5e-4",
            "--batch-size", "32",
            "--num-workers", "0",
            "--device", "cpu",
            "--center-crop", "192",
            "--mlflow-experiment", "galaxy3-pilot",
            "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    expected = ExperimentConfig(
        run_name="batchnorm_dropout",
        use_batchnorm=True,
        dropout_p=0.3,
        train_seed=11,
        deterministic=True,
        epochs=5,
        learning_rate=5e-4,
        batch_size=32,
        num_workers=0,
        device_preference="cpu",
        center_crop=192,
        mlflow_experiment_name="galaxy3-pilot",
        data_root=tmp_path / "data",
        output_root=tmp_path / "out",
    )
    assert config == expected


def test_omitted_flags_keep_defaults():
    config = train.parse_config(["--run-name", "batchnorm", "--use-batchnorm"])
    defaults = ExperimentConfig()
    assert config.use_batchnorm is True
    assert config.dropout_p == defaults.dropout_p
    assert config.epochs == defaults.epochs
    assert config.mlflow_experiment_name == defaults.mlflow_experiment_name


def test_no_use_batchnorm_flag():
    assert train.parse_config(["--no-use-batchnorm"]).use_batchnorm is False


def test_deterministic_flag_defaults_to_false():
    assert train.parse_config([]).deterministic is False
    assert train.parse_config(["--no-deterministic"]).deterministic is False


def test_parser_defaults_are_not_duplicated():
    """El Namespace vacío prueba que los defaults vienen solo de ``ExperimentConfig``."""
    assert vars(train.build_parser().parse_args([])) == {}


@pytest.mark.parametrize(
    "argv",
    [
        ["--dropout-p", "1.0"],
        ["--epochs", "0"],
        ["--device", "tpu"],
        ["--batch-size", "dos"],
        ["--weight-decay", "0.1"],
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        train.parse_config(argv)
    assert excinfo.value.code == 2
    assert "usage:" in capsys.readouterr().err


# ------------------------------------------------------------------- --help


def test_help_lists_flags_and_config_defaults(capsys):
    with pytest.raises(SystemExit) as excinfo:
        train.parse_config(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in train._FLAGS:
        assert flag in out
    assert f"(default: {ExperimentConfig().mlflow_experiment_name})" in out


def test_module_is_executable_with_python_m():
    completed = subprocess.run(
        [sys.executable, "-m", "galaxy3.train", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--run-name" in completed.stdout


# ------------------------------------------------------------- punta a punta


def test_end_to_end_short_run(tmp_path, capsys):
    data_root = _make_data_root(tmp_path / "data")
    result = train.main(
        [
            "--run-name", "cli_smoke",
            "--use-batchnorm",
            "--dropout-p", "0.3",
            "--epochs", "2",
            "--batch-size", "4",
            "--num-workers", "0",
            "--device", "cpu",
            "--mlflow-experiment", "galaxy3-cli-test",
            "--data-root", str(data_root),
            "--output-root", str(tmp_path / "outputs"),
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert "Resumen del run" in out
    assert result.run_id in out
    assert len(result.history["val_loss"]) == 2
    assert result.checkpoint_path.is_file()

    mlflow.set_tracking_uri(resolve_tracking_uri())
    run = mlflow.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == "cli_smoke"
    assert run.data.params["use_batchnorm"] == "True"
    assert run.data.params["dropout_p"] == "0.3"
    experiment = mlflow.get_experiment(run.info.experiment_id)
    assert experiment.name == "galaxy3-cli-test"
    assert not any(key.startswith("test_") for key in run.data.metrics)
