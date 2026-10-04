"""Smoke test de punta a punta (Fase 6): data -> model -> loss -> backward -> optimizer.step.

A diferencia de ``test_engine.py`` (tensores sintéticos de 32x32), aquí los batches salen
del pipeline real de ``data.py``: ``ImageFolder`` + ``Resize(128)`` + ``ToTensor`` +
``Normalize`` construidos desde la configuración. El ``ImageFolder`` es sintético y
diminuto (imágenes de ruido en ``tmp_path``) para no depender del dataset completo.

No se comprueba que el modelo aprenda bien: solo que todas las piezas se conectan, que la
pérdida es finita y que el gradiente llega a todos los parámetros entrenables.
"""

import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from galaxy3.config import ExperimentConfig
from galaxy3.data import DATASET_STATS_NAME, make_dataloaders
from galaxy3.engine import fit
from galaxy3.models import GalaxyCNN, count_trainable_parameters
from galaxy3.utils import set_seed

CLASSES = ["barred_spiral", "edge_on_no_bulge", "round_smooth"]
PER_SPLIT = {"train": 4, "val": 2, "test": 1}
IMAGE_SIZE = 128
BATCH_SIZE = 6
CPU = torch.device("cpu")
DEVICES = [CPU] + ([torch.device("cuda")] if torch.cuda.is_available() else [])

RUNS = {
    "baseline": {"use_batchnorm": False, "dropout_p": 0.0},
    "batchnorm": {"use_batchnorm": True, "dropout_p": 0.0},
    "batchnorm_dropout": {"use_batchnorm": True, "dropout_p": 0.3},
}


@pytest.fixture
def config(tmp_path):
    """Config apuntando a un ``ImageFolder`` sintético con stats coherentes con ella."""
    data_root = tmp_path / "data"
    rng = np.random.default_rng(0)
    for split, n in PER_SPLIT.items():
        for name in CLASSES:
            folder = data_root / split / name
            folder.mkdir(parents=True)
            for i in range(n):
                pixels = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(folder / f"{i:05d}.png")
    stats = {
        "split_seed": 2026,
        "class_to_label": {name: i for i, name in enumerate(CLASSES)},
        "transform": {"image_size": IMAGE_SIZE, "center_crop": None},
        "mean_train": [0.5, 0.5, 0.5],
        "std_train": [0.29, 0.29, 0.29],
    }
    (data_root / DATASET_STATS_NAME).write_text(json.dumps(stats))
    return ExperimentConfig(
        run_name="smoke",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        num_workers=0,
        epochs=1,
        device_preference="cpu",
        data_root=data_root,
        output_root=tmp_path / "outputs",
    )


def _build(config: ExperimentConfig, device: torch.device):
    set_seed(config.train_seed)
    model = GalaxyCNN(config.num_classes, config.use_batchnorm, config.dropout_p).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    return model, optimizer


@pytest.mark.parametrize("device", DEVICES, ids=str)
@pytest.mark.parametrize("run", RUNS)
def test_one_training_step_from_dataloader(config, run, device):
    config = replace(config, run_name=run, **RUNS[run])
    loaders = make_dataloaders(config, device)
    model, optimizer = _build(config, device)
    model.train()

    # data
    X, y = next(iter(loaders.train))
    assert X.shape == (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert X.dtype == torch.float32
    assert y.shape == (BATCH_SIZE,)
    assert y.dtype == torch.long
    assert X.is_pinned() == (device.type == "cuda")
    X, y = X.to(device), y.to(device)

    # model -> loss
    optimizer.zero_grad(set_to_none=True)
    logits = model(X)
    assert logits.shape == (BATCH_SIZE, config.num_classes)
    loss = nn.CrossEntropyLoss()(logits, y)
    assert torch.isfinite(loss)

    # backward: todo parámetro entrenable recibe un gradiente finito y no nulo
    loss.backward()
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in trainable.values()) == count_trainable_parameters(model)
    for name, p in trainable.items():
        assert p.grad is not None, f"{name} no está conectado a la pérdida"
        assert torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"{name} recibió gradiente nulo"

    # optimizer.step: todos los parámetros entrenables cambian
    before = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer.step()
    for name, p in trainable.items():
        assert not torch.equal(before[name], p.detach()), f"{name} no se actualizó"

    batchnorms = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(batchnorms) == (4 if config.use_batchnorm else 0)
    assert all(bn.num_batches_tracked.item() == 1 for bn in batchnorms)


def test_fit_one_epoch_from_dataloaders(config):
    """Integración: los loaders de ``data.py`` alimentan ``fit`` y el checkpoint recarga."""
    config = replace(config, **RUNS["batchnorm_dropout"])
    loaders = make_dataloaders(config, CPU)
    model, optimizer = _build(config, CPU)
    checkpoint_path = config.output_root / "checkpoints" / "best.pt"

    history = fit(
        model,
        loaders.train,
        loaders.val,
        nn.CrossEntropyLoss(),
        optimizer,
        CPU,
        config,
        checkpoint_path,
        verbose=False,
    )

    assert history["best_epoch"] == 1
    assert all(math.isfinite(history[k][0]) for k in ("train_loss", "val_loss"))
    checkpoint = torch.load(checkpoint_path, map_location=CPU, weights_only=True)
    GalaxyCNN(config.num_classes, config.use_batchnorm, config.dropout_p).load_state_dict(
        checkpoint["model_state_dict"]
    )
