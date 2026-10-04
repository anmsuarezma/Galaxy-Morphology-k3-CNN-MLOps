"""Tests del engine con datos sintéticos.

Se usan imágenes de 32x32 (no 128x128) para que sean rápidos; además comprueban que
el engine no asume la resolución de GalaxyCNN.
"""

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from galaxy3 import engine
from galaxy3.config import ExperimentConfig
from galaxy3.engine import EpochMetrics, evaluate, fit, train_one_epoch
from galaxy3.models import GalaxyCNN

CPU = torch.device("cpu")
NUM_CLASSES = 3
IMAGE_SIZE = 32


def _dataset(n: int, seed: int) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g)
    y = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return TensorDataset(X, y)


def _loader(n: int = 20, batch_size: int = 8, seed: int = 0, shuffle: bool = False) -> DataLoader:
    return DataLoader(_dataset(n, seed), batch_size=batch_size, shuffle=shuffle)


def _model(use_batchnorm: bool = True, seed: int = 0) -> GalaxyCNN:
    torch.manual_seed(seed)
    return GalaxyCNN(num_classes=NUM_CLASSES, use_batchnorm=use_batchnorm)


def _config(tmp_path, epochs: int = 3) -> ExperimentConfig:
    return ExperimentConfig(
        run_name="test_engine",
        epochs=epochs,
        learning_rate=1e-3,
        device_preference="cpu",
        data_root=tmp_path / "data",
        output_root=tmp_path / "outputs",
    )


def _fit(tmp_path, epochs: int = 3, model: nn.Module | None = None):
    model = model if model is not None else _model()
    config = _config(tmp_path, epochs=epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    checkpoint_path = tmp_path / "checkpoints" / "best.pt"
    history = fit(
        model,
        _loader(seed=0, shuffle=True),
        _loader(n=12, seed=1),
        nn.CrossEntropyLoss(),
        optimizer,
        CPU,
        config,
        checkpoint_path,
        verbose=False,
    )
    return model, config, history, checkpoint_path


def _snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def test_train_one_epoch_updates_weights_and_returns_finite_metrics():
    model = _model()
    before = _snapshot(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = train_one_epoch(model, _loader(), nn.CrossEntropyLoss(), optimizer, CPU)

    assert model.training
    assert math.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy <= 1.0
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before)


def test_evaluate_uses_eval_mode_and_does_not_modify_model():
    model = _model(use_batchnorm=True)
    model.train()
    before = _snapshot(model)
    metrics = evaluate(model, _loader(), nn.CrossEntropyLoss(), CPU)

    assert not model.training
    assert math.isfinite(metrics.loss)
    assert 0.0 <= metrics.accuracy <= 1.0
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)  # incluye running stats


def test_evaluate_averages_per_sample_with_uneven_last_batch():
    model = _model().eval()
    dataset = _dataset(20, seed=0)
    loss_fn = nn.CrossEntropyLoss()
    metrics = evaluate(model, DataLoader(dataset, batch_size=7), loss_fn, CPU)

    X, y = dataset.tensors
    with torch.inference_mode():
        logits = model(X)
    assert metrics.loss == pytest.approx(loss_fn(logits, y).item(), rel=1e-5)
    assert metrics.accuracy == pytest.approx((logits.argmax(1) == y).float().mean().item())


def test_empty_loader_raises():
    empty = DataLoader(TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="ninguna muestra"):
        evaluate(_model(), empty, nn.CrossEntropyLoss(), CPU)


def test_loss_without_mean_reduction_raises():
    with pytest.raises(ValueError, match="reduction"):
        evaluate(_model(), _loader(), nn.CrossEntropyLoss(reduction="sum"), CPU)


@pytest.mark.parametrize("epochs", [1, 3])
def test_fit_runs_exactly_epochs_and_history_is_consistent(tmp_path, epochs):
    _, config, history, _ = _fit(tmp_path, epochs=epochs)

    for key in ("train_loss", "train_acc", "val_loss", "val_acc", "lr", "epoch_time"):
        assert len(history[key]) == epochs, key
        assert all(math.isfinite(v) for v in history[key]), key
    assert all(0.0 <= a <= 1.0 for a in history["train_acc"] + history["val_acc"])
    assert all(t > 0 for t in history["epoch_time"])
    assert history["lr"] == [config.learning_rate] * epochs

    best_index = min(range(epochs), key=lambda i: history["val_loss"][i])
    assert history["best_epoch"] == best_index + 1
    assert history["best_val_loss"] == history["val_loss"][best_index]


def test_checkpoint_loads_with_weights_only_and_restores_model(tmp_path):
    _, config, history, checkpoint_path = _fit(tmp_path, epochs=3)

    checkpoint = torch.load(checkpoint_path, map_location=CPU, weights_only=True)
    assert set(checkpoint) == {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "val_loss",
        "config",
    }
    assert checkpoint["epoch"] == history["best_epoch"]
    assert checkpoint["val_loss"] == pytest.approx(history["best_val_loss"])
    assert checkpoint["config"] == config.to_dict()
    assert ExperimentConfig.from_dict(checkpoint["config"]) == config

    restored = _model(seed=123)
    restored.load_state_dict(checkpoint["model_state_dict"])
    for key, value in checkpoint["model_state_dict"].items():
        assert torch.equal(restored.state_dict()[key], value), key
    val_metrics = evaluate(restored, _loader(n=12, seed=1), nn.CrossEntropyLoss(), CPU)
    assert val_metrics.loss == pytest.approx(history["best_val_loss"], rel=1e-5)

    optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def test_checkpoint_saved_only_when_val_loss_improves(tmp_path, monkeypatch):
    scripted_val_losses = iter([1.0, 0.5, 0.8, 0.5])
    monkeypatch.setattr(
        engine, "evaluate", lambda *args, **kwargs: EpochMetrics(next(scripted_val_losses), 0.5)
    )
    saved_epochs = []
    original_save = engine.save_checkpoint

    def spy_save(path, epoch, *args, **kwargs):
        saved_epochs.append(epoch)
        original_save(path, epoch, *args, **kwargs)

    monkeypatch.setattr(engine, "save_checkpoint", spy_save)
    _, _, history, checkpoint_path = _fit(tmp_path, epochs=4)

    assert saved_epochs == [1, 2]  # 0.5 en la época 4 empata, no mejora estrictamente
    assert history["best_epoch"] == 2
    assert history["best_val_loss"] == 0.5
    assert torch.load(checkpoint_path, weights_only=True)["epoch"] == 2


def test_fit_raises_if_val_loss_never_finite(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "evaluate", lambda *a, **k: EpochMetrics(math.nan, 0.0))
    with pytest.raises(RuntimeError, match="finita"):
        _fit(tmp_path, epochs=2)
    assert not (tmp_path / "checkpoints" / "best.pt").exists()


def test_model_overfits_small_batch():
    """Prueba de sanidad: el modelo debe poder memorizar un batch pequeño."""
    model = _model(use_batchnorm=False)
    loader = _loader(n=16, batch_size=16, seed=7)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)

    initial = evaluate(model, loader, loss_fn, CPU).loss
    for _ in range(60):
        train_one_epoch(model, loader, loss_fn, optimizer, CPU)
    final = evaluate(model, loader, loss_fn, CPU).loss

    assert final < 0.1 * initial


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
def test_fit_on_cuda(tmp_path):
    device = torch.device("cuda")
    model = _model().to(device)
    config = _config(tmp_path, epochs=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loaders = [
        DataLoader(_dataset(n, seed), batch_size=8, pin_memory=True)
        for n, seed in ((20, 0), (12, 1))
    ]
    checkpoint_path = tmp_path / "best.pt"
    history = fit(
        model,
        *loaders,
        nn.CrossEntropyLoss(),
        optimizer,
        device,
        config,
        checkpoint_path,
        verbose=False,
    )
    assert len(history["val_loss"]) == 2
    checkpoint = torch.load(checkpoint_path, map_location=CPU, weights_only=True)
    assert all(t.device == CPU for t in checkpoint["model_state_dict"].values())
