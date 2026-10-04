"""Tests de ``experiment.py`` con un ImageFolder sintético pequeño.

Imágenes de ruido de 32x32 (GalaxyCNN funciona con cualquier lado múltiplo de 16), CPU y
``num_workers=0`` para que sean rápidos. No se evalúa la calidad del modelo: solo el
contrato de la orquestación y lo que queda registrado en MLflow. El backend de MLflow es
un SQLite temporal (ver ``conftest.py``).
"""

import json
import math
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.datasets import ImageFolder  # noqa: E402

from galaxy3 import data, experiment  # noqa: E402
from galaxy3.config import ExperimentConfig  # noqa: E402
from galaxy3.data import DATASET_STATS_NAME  # noqa: E402
from galaxy3.experiment import (  # noqa: E402
    checkpoint_path_for,
    evaluate_on_test,
    resolve_tracking_uri,
    run_experiment,
    run_reference_path,
)
from galaxy3.utils import PROJECT_ROOT  # noqa: E402

CLASSES = ["barred_spiral", "edge_on_no_bulge", "round_smooth"]
PER_CLASS = {"train": 6, "val": 3, "test": 2}
IMAGE_SIZE = 32
SPLIT_SEED = 2026


def _write_split(root, split: str, n: int, rng: np.random.Generator) -> None:
    for name in CLASSES:
        folder = root / split / name
        folder.mkdir(parents=True)
        for i in range(n):
            pixels = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / f"{i:05d}.png")


def _make_data_root(root, splits=("train", "val", "test")):
    rng = np.random.default_rng(0)
    for split in splits:
        _write_split(root, split, PER_CLASS[split], rng)
    stats = {
        "split_seed": SPLIT_SEED,
        "class_to_label": {name: i for i, name in enumerate(CLASSES)},
        "counts": {split: dict.fromkeys(CLASSES, n) for split, n in PER_CLASS.items()},
        "transform": {"image_size": IMAGE_SIZE, "center_crop": None},
        "mean_train": [0.5, 0.5, 0.5],
        "std_train": [0.25, 0.25, 0.25],
    }
    (root / DATASET_STATS_NAME).write_text(json.dumps(stats))
    return root


@pytest.fixture
def config(tmp_path):
    return ExperimentConfig(
        run_name="test_run",
        image_size=IMAGE_SIZE,
        batch_size=8,
        num_workers=0,
        epochs=2,
        use_batchnorm=True,
        device_preference="cpu",
        data_root=_make_data_root(tmp_path / "data"),
        output_root=tmp_path / "outputs",
    )


@pytest.fixture
def result(config):
    return run_experiment(config, verbose=False)


def test_result_structure_and_outputs(config, result):
    assert result.class_names == CLASSES
    assert result.device == "cpu"
    assert len(result.history["val_loss"]) == config.epochs
    assert result.train_samples == PER_CLASS["train"] * len(CLASSES)
    assert result.val_samples == PER_CLASS["val"] * len(CLASSES)
    assert result.trainable_parameters == result.total_parameters > 0
    assert result.checkpoint_path == checkpoint_path_for(config)
    assert result.checkpoint_path.is_file()
    assert set(result.figure_paths) == {"training_curves", "val_confusion_matrix"}
    assert all(path.is_file() for path in result.figure_paths.values())
    assert result.val_eval.report.confusion_matrix.sum() == result.val_samples
    assert result.train_eval.report.confusion_matrix.sum() == result.train_samples


def test_final_metrics_are_finite_and_named_as_spec(result):
    metrics = result.final_metrics()
    assert set(metrics) == {
        "best_epoch",
        "best_val_loss",
        "train_loss_eval",
        "train_acc_eval",
        "val_acc_best",
        "val_macro_f1",
        "generalization_gap",
    }
    assert all(math.isfinite(v) for v in metrics.values())


def test_generalization_gap_uses_train_eval(result):
    expected = result.train_eval.report.accuracy - result.val_eval.report.accuracy
    assert result.generalization_gap == pytest.approx(expected)
    assert result.final_metrics()["train_acc_eval"] == result.train_eval.report.accuracy


def test_val_metrics_come_from_best_checkpoint(config, tmp_path):
    # Con más épocas, la mejor no tiene por qué ser la última: la val_loss reevaluada
    # debe coincidir con la de la mejor época, no con la de la última.
    result = run_experiment(replace(config, epochs=4), verbose=False)
    best = result.history["best_epoch"]
    assert result.val_eval.loss == pytest.approx(result.history["val_loss"][best - 1], rel=1e-5)
    assert result.val_eval.loss == pytest.approx(result.history["best_val_loss"], rel=1e-5)


def test_checkpoint_loads_with_weights_only(config, result):
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == result.history["best_epoch"]
    assert checkpoint["config"] == config.to_dict()
    model = experiment.build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])


def test_optimizer_uses_config_weight_decay(config):
    result = run_experiment(replace(config, epochs=1, weight_decay=0.05), verbose=False)
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu", weights_only=True)
    group = checkpoint["optimizer_state_dict"]["param_groups"][0]
    assert group["weight_decay"] == 0.05
    assert group["lr"] == result.config.learning_rate


def test_metrics_json_is_written(result):
    payload = json.loads(result.metrics_path.read_text())
    assert payload["final_metrics"] == pytest.approx(result.final_metrics())
    assert payload["config"]["run_name"] == "test_run"
    assert set(payload["val_eval"]["per_class"]) == set(CLASSES)
    assert "test" not in json.dumps(payload["final_metrics"])


def test_run_experiment_never_builds_or_reads_test(config, monkeypatch):
    """El ImageFolder de test no se construye ni se lee, y no se pide su loader."""
    roots, read_from = [], []

    class RecordingImageFolder(ImageFolder):
        def __init__(self, root, *args, **kwargs):
            roots.append(str(root))
            super().__init__(root, *args, **kwargs)

        def __getitem__(self, index):
            read_from.append(str(self.root))
            return super().__getitem__(index)

    requested = []
    original = data.make_split_loader

    def spy(cfg, split, device=None):
        requested.append(split)
        return original(cfg, split, device)

    monkeypatch.setattr(data, "ImageFolder", RecordingImageFolder)
    monkeypatch.setattr(experiment, "make_split_loader", spy)
    run_experiment(config, verbose=False)

    test_root = str(config.data_root / "test")
    assert sorted(set(requested)) == ["train", "val"]
    assert test_root not in roots
    assert read_from  # el espía sí registra lecturas de train y val
    assert test_root not in set(read_from)


def test_run_experiment_works_without_test_folder(tmp_path):
    data_root = _make_data_root(tmp_path / "data", splits=("train", "val"))
    config = ExperimentConfig(
        image_size=IMAGE_SIZE,
        batch_size=8,
        num_workers=0,
        epochs=1,
        device_preference="cpu",
        data_root=data_root,
        output_root=tmp_path / "outputs",
    )
    result = run_experiment(config, verbose=False)
    assert result.checkpoint_path.is_file()


@pytest.mark.parametrize("deterministic", [False, True])
def test_run_experiment_passes_deterministic_to_set_seed(config, monkeypatch, deterministic):
    calls = []
    original = experiment.set_seed

    def spy(seed, deterministic=False):
        calls.append((seed, deterministic))
        original(seed, deterministic=deterministic)

    monkeypatch.setattr(experiment, "set_seed", spy)
    run_experiment(replace(config, epochs=1, deterministic=deterministic), verbose=False)
    assert calls == [(config.train_seed, deterministic)]
    torch.backends.cudnn.deterministic = False


def test_split_seed_mismatch_raises(config):
    with pytest.raises(ValueError, match="split_seed"):
        run_experiment(replace(config, split_seed=SPLIT_SEED + 1), verbose=False)


@pytest.mark.parametrize("name", ["a/b", "..", "../escape"])
def test_run_name_with_path_separators_raises(config, name):
    with pytest.raises(ValueError, match="run_name"):
        run_experiment(replace(config, run_name=name), verbose=False)


def test_evaluate_on_test_with_synthetic_data(config, result):
    final = evaluate_on_test(result.checkpoint_path, config)
    n_test = PER_CLASS["test"] * len(CLASSES)
    assert final.test_samples == n_test
    assert final.checkpoint_epoch == result.history["best_epoch"]
    assert final.test_eval.report.confusion_matrix.shape == (len(CLASSES), len(CLASSES))
    assert final.test_eval.report.confusion_matrix.sum() == n_test
    assert set(final.final_metrics()) == {"test_loss", "test_acc", "test_macro_f1"}
    assert all(math.isfinite(v) for v in final.final_metrics().values())
    assert "test_confusion_matrix" in final.figure_paths
    assert all(path.is_file() for path in final.figure_paths.values())
    payload = json.loads(final.metrics_path.read_text())
    assert payload["test_samples"] == n_test


def test_evaluate_on_test_plots_errors_only_when_there_are_errors(config, result):
    final = evaluate_on_test(result.checkpoint_path, config)
    has_errors = final.test_eval.report.accuracy < 1.0
    assert ("test_errors" in final.figure_paths) == has_errors


def test_evaluate_on_test_does_not_modify_checkpoint(config, result):
    before = result.checkpoint_path.read_bytes()
    evaluate_on_test(result.checkpoint_path, config)
    assert result.checkpoint_path.read_bytes() == before


@pytest.mark.parametrize(
    "changes", [{"use_batchnorm": False}, {"dropout_p": 0.3}], ids=["bn", "dropout"]
)
def test_evaluate_on_test_rejects_mismatched_config(config, result, changes):
    with pytest.raises(ValueError, match=next(iter(changes))):
        evaluate_on_test(result.checkpoint_path, replace(config, **changes))


# --------------------------------------------------------------------------- MLflow

EXPECTED_ARTIFACTS = {
    "config.json",
    "metrics/test_run.json",
    "checkpoints/test_run_best.pt",
    "figures/test_run_training_curves.png",
    "figures/test_run_val_confusion_matrix.png",
}


def _artifact_paths(run_id: str, path: str | None = None) -> set[str]:
    """Rutas de todos los artifacts de un run (recorre carpetas)."""
    client = mlflow.MlflowClient()
    paths = set()
    for item in client.list_artifacts(run_id, path):
        paths |= _artifact_paths(run_id, item.path) if item.is_dir else {item.path}
    return paths


def _metric_history(run_id: str, name: str) -> list[tuple[int, float]]:
    history = mlflow.MlflowClient().get_metric_history(run_id, name)
    return sorted((m.step, m.value) for m in history)


def test_run_is_registered_with_params(config, result):
    run = mlflow.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.info.run_name == config.run_name
    experiment_name = mlflow.get_experiment(run.info.experiment_id).name
    assert experiment_name == config.mlflow_experiment_name
    params = run.data.params
    assert set(config.to_dict()) <= set(params)
    assert params["weight_decay"] == str(config.weight_decay)
    assert params["deterministic"] == str(config.deterministic)
    assert params["model_name"] == "GalaxyCNN"
    assert params["optimizer"] == "AdamW"
    assert params["device"] == "cpu"
    assert params["total_parameters"] == str(result.total_parameters)
    assert params["trainable_parameters"] == str(result.trainable_parameters)
    for split, n in PER_CLASS.items():
        assert params[f"{split}_samples"] == str(n * len(CLASSES))


def test_epoch_metrics_are_logged_with_step(config, result):
    history = result.history
    for name, key in experiment.EPOCH_METRICS.items():
        logged = _metric_history(result.run_id, name)
        assert [step for step, _ in logged] == list(range(1, config.epochs + 1))
        assert [value for _, value in logged] == pytest.approx(history[key])


def test_final_metrics_are_logged(result):
    logged = mlflow.get_run(result.run_id).data.metrics
    for name, value in result.final_metrics().items():
        assert logged[name] == pytest.approx(value)
    assert not any(name.startswith("test_") for name in logged)


def test_artifacts_are_logged_next_to_temporary_database(result, isolated_mlflow_backend):
    assert _artifact_paths(result.run_id) == EXPECTED_ARTIFACTS
    artifact_uri = mlflow.get_run(result.run_id).info.artifact_uri
    assert artifact_uri.startswith((isolated_mlflow_backend.parent / "mlruns").as_uri())


def test_run_id_is_saved_next_to_checkpoint(result):
    reference = json.loads(run_reference_path(result.checkpoint_path).read_text())
    assert run_reference_path(result.checkpoint_path).parent == result.checkpoint_path.parent
    assert reference == {"run_id": result.run_id, "tracking_uri": resolve_tracking_uri()}


def test_real_backend_is_never_touched(config):
    watched = [PROJECT_ROOT / "mlflow.db", PROJECT_ROOT / "mlruns"]

    def snapshot():
        return [(p.exists(), p.stat().st_mtime_ns if p.exists() else None) for p in watched]

    before = snapshot()
    result = run_experiment(replace(config, epochs=1), verbose=False)
    evaluate_on_test(result.checkpoint_path, replace(config, epochs=1))
    assert snapshot() == before


def test_failed_training_marks_run_as_failed(config, monkeypatch):
    def broken_fit(*args, **kwargs):
        raise RuntimeError("fallo simulado")

    monkeypatch.setattr(experiment, "fit", broken_fit)
    with pytest.raises(RuntimeError, match="fallo simulado"):
        run_experiment(replace(config, run_name="failing_run"), verbose=False)
    runs = mlflow.search_runs(
        experiment_names=[config.mlflow_experiment_name],
        filter_string="attributes.run_name = 'failing_run'",
        output_format="list",
    )
    assert [run.info.status for run in runs] == ["FAILED"]


def test_evaluate_on_test_logs_into_the_same_run(config, result):
    final = evaluate_on_test(result.checkpoint_path, config)
    assert final.run_id == result.run_id
    run = mlflow.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    for name, value in {**final.final_metrics(), **result.final_metrics()}.items():
        assert run.data.metrics[name] == pytest.approx(value)
    test_artifacts = {p for p in _artifact_paths(result.run_id) if p.startswith("test/")}
    expected = {f"test/{path.name}" for path in final.figure_paths.values()}
    assert test_artifacts == expected | {"test/test_run_test.json"}


def test_evaluate_on_test_runs_only_once(config, result, monkeypatch):
    evaluate_on_test(result.checkpoint_path, config)
    requested = []
    original = data.make_split_loader

    def spy(cfg, split, device=None):
        requested.append(split)
        return original(cfg, split, device)

    monkeypatch.setattr(experiment, "make_split_loader", spy)
    with pytest.raises(RuntimeError, match="una sola vez"):
        evaluate_on_test(result.checkpoint_path, config)
    assert requested == []  # el segundo intento no abre test


def test_evaluate_on_test_requires_run_reference(config, result):
    run_reference_path(result.checkpoint_path).unlink()
    with pytest.raises(FileNotFoundError, match="mlflow.json"):
        evaluate_on_test(result.checkpoint_path, config)


def test_evaluate_on_test_rejects_other_tracking_uri(config, result, monkeypatch, tmp_path):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{(tmp_path / 'other.db').as_posix()}")
    with pytest.raises(ValueError, match="tracking URI"):
        evaluate_on_test(result.checkpoint_path, config)
    assert not (tmp_path / "other.db").exists()
