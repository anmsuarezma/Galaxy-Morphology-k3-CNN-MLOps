"""Tests de ``plots.py`` con backend no interactivo y datos sintéticos.

No se compara el aspecto visual: se verifica que cada función devuelva un ``Figure``
coherente con su entrada, que la desnormalización sea correcta y que el módulo no dependa
de MLflow.
"""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from PIL import Image  # noqa: E402

from galaxy3.plots import (  # noqa: E402
    FALSE_COLOR_NOTE,
    RUN_COLORS,
    RUN_LINESTYLES,
    denormalize,
    plot_confusion_matrix,
    plot_feature_maps,
    plot_image_batch,
    plot_original_galaxies,
    plot_predictions,
    plot_runs_comparison,
    plot_training_curves,
)

CLASSES = ["barred_spiral", "edge_on_no_bulge", "round_smooth"]
MEAN = [0.17, 0.16, 0.16]
STD = [0.12, 0.11, 0.10]


@pytest.fixture
def close_figures():
    """Cierra toda figura que el test haya dejado abierta, aunque el test falle."""
    yield
    plt.close("all")


def _normalized_batch(n: int = 6, size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    images = torch.rand(n, 3, size, size, generator=g)
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return images, (images - mean) / std


def _suptitle(fig: Figure) -> str:
    return fig._suptitle.get_text() if fig._suptitle is not None else ""


def test_denormalize_recovers_original_image():
    images, normalized = _normalized_batch()
    restored = denormalize(normalized, MEAN, STD)
    assert restored.shape == images.shape
    assert torch.allclose(restored, images, atol=1e-6)
    single = denormalize(normalized[0], MEAN, STD)
    assert torch.allclose(single, images[0], atol=1e-6)


def test_denormalize_clamps_to_unit_interval_and_does_not_mutate():
    X = torch.tensor([-100.0, 0.0, 100.0]).view(3, 1, 1).expand(3, 2, 2).clone()
    original = X.clone()
    out = denormalize(X, MEAN, STD)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert out[0].eq(0.0).all() and out[2].eq(1.0).all()
    assert torch.equal(X, original)


@pytest.mark.parametrize(
    ("X", "mean", "std"),
    [
        (torch.zeros(3, 4), MEAN, STD),
        (torch.zeros(3, 4, 4), MEAN[:2], STD),
        (torch.zeros(2, 3, 4, 4), MEAN, STD[:2]),
    ],
)
def test_denormalize_rejects_invalid_input(X, mean, std):
    with pytest.raises(ValueError):
        denormalize(X, mean, std)


def test_plot_image_batch_returns_figure(close_figures):
    _, X = _normalized_batch(n=6)
    y = torch.tensor([0, 1, 2, 0, 1, 2])
    fig = plot_image_batch(X, y, CLASSES, MEAN, STD, max_images=4, ncols=2)

    assert isinstance(fig, Figure)
    images = [ax for ax in fig.axes if ax.images]
    assert len(images) == 4
    assert [ax.get_title() for ax in images] == [CLASSES[i] for i in (0, 1, 2, 0)]
    shown = images[0].images[0].get_array()
    assert shown.shape == (16, 16, 3)
    assert shown.min() >= 0.0 and shown.max() <= 1.0
    assert FALSE_COLOR_NOTE in _suptitle(fig)


def test_plot_predictions_marks_hits_and_errors(close_figures):
    _, X = _normalized_batch(n=3)
    y_true = torch.tensor([0, 1, 2])
    probabilities = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]])
    fig = plot_predictions(X, y_true, probabilities, CLASSES, MEAN, STD)

    assert isinstance(fig, Figure)
    titles = [ax.get_title() for ax in fig.axes if ax.images]
    assert titles[0].startswith("✓") and "p=0.70" in titles[0]
    assert titles[1].startswith("✗") and "pred: barred_spiral (p=0.60)" in titles[1]
    assert titles[2].startswith("✓") and "p=0.80" in titles[2]
    assert FALSE_COLOR_NOTE in _suptitle(fig)


def test_plot_predictions_rejects_mismatched_probabilities(close_figures):
    _, X = _normalized_batch(n=3)
    with pytest.raises(ValueError, match="probabilities"):
        plot_predictions(X, torch.tensor([0, 1, 2]), torch.ones(3, 2) / 2, CLASSES, MEAN, STD)


def test_plot_original_galaxies_one_row_per_class(close_figures):
    images = {
        name: [Image.new("RGB", (32, 32), (40 * i, 80, 120)) for i in range(3)] for name in CLASSES
    }
    fig = plot_original_galaxies(images)

    assert isinstance(fig, Figure)
    assert len([ax for ax in fig.axes if ax.images]) == 9
    assert [ax.get_ylabel() for ax in fig.axes[::3]] == CLASSES
    assert FALSE_COLOR_NOTE in _suptitle(fig)
    assert "32x32" in _suptitle(fig)


def test_plot_original_galaxies_rejects_uneven_classes(close_figures):
    with pytest.raises(ValueError):
        plot_original_galaxies({"a": [Image.new("RGB", (4, 4))], "b": []})


def test_plot_feature_maps_grayscale_normalized_per_map(close_figures):
    maps = torch.rand(2, 20, 8, 8) * 50
    maps[1, 3] = 0.0  # canal muerto en la muestra 1
    fig = plot_feature_maps(maps, sample_index=1, max_channels=16, layer_name="features.block1.act")

    assert isinstance(fig, Figure)
    shown = [ax for ax in fig.axes if ax.images]
    assert len(shown) == 16
    for ax in shown:
        array = ax.images[0].get_array()
        assert array.shape == (8, 8)
        assert array.min() >= 0.0 and array.max() <= 1.0
        assert ax.images[0].get_cmap().name == "gray"
    assert shown[0].images[0].get_array().max() == pytest.approx(1.0)
    assert "constante" in shown[3].get_title()
    assert "features.block1.act" in _suptitle(fig)


@pytest.mark.parametrize(
    ("maps", "kwargs"),
    [
        (torch.zeros(3, 8, 8), {}),
        (torch.zeros(2, 3, 8, 8), {"sample_index": 2}),
        (torch.zeros(2, 3, 8, 8), {"max_channels": 0}),
    ],
)
def test_plot_feature_maps_rejects_invalid_input(maps, kwargs):
    with pytest.raises(ValueError):
        plot_feature_maps(maps, **kwargs)


def test_plot_training_curves(close_figures):
    history = {
        "train_loss": [1.0, 0.7, 0.5],
        "train_acc": [0.4, 0.6, 0.8],
        "val_loss": [0.9, 0.6, 0.7],
        "val_acc": [0.5, 0.7, 0.65],
        "lr": [1e-3] * 3,
        "epoch_time": [1.0] * 3,
        "best_epoch": 2,
        "best_val_loss": 0.6,
    }
    fig = plot_training_curves(history)

    assert isinstance(fig, Figure)
    ax_loss, ax_acc = fig.axes
    assert list(ax_loss.lines[1].get_ydata()) == history["val_loss"]
    assert list(ax_acc.lines[0].get_ydata()) == history["train_acc"]
    assert "mejor época = 2" in _suptitle(fig)


def _history(val_loss: list[float], val_acc: list[float], best_epoch: int) -> dict:
    n = len(val_loss)
    return {
        "train_loss": [0.5] * n,
        "train_acc": [0.8] * n,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "lr": [1e-3] * n,
        "epoch_time": [1.0] * n,
        "best_epoch": best_epoch,
        "best_val_loss": val_loss[best_epoch - 1],
    }


def test_plot_runs_comparison_one_line_and_marker_per_run(close_figures):
    histories = {
        "baseline": _history([0.9, 0.6, 0.7], [0.5, 0.7, 0.72], best_epoch=2),
        "batchnorm": _history([0.8, 0.5, 0.55, 0.6], [0.6, 0.8, 0.8, 0.81], best_epoch=2),
        "batchnorm_dropout": _history([0.85, 0.55, 0.5], [0.55, 0.75, 0.8], best_epoch=3),
    }
    fig = plot_runs_comparison(histories)

    assert isinstance(fig, Figure)
    ax_loss, ax_acc = fig.axes
    for ax, key in ((ax_loss, "val_loss"), (ax_acc, "val_acc")):
        curves, markers = ax.lines[0::2], ax.lines[1::2]
        assert [line.get_label() for line in curves] == list(histories)
        for line, history in zip(curves, histories.values(), strict=True):
            assert list(line.get_ydata()) == history[key]
        assert [list(m.get_ydata()) for m in markers] == [
            [h[key][h["best_epoch"] - 1]] for h in histories.values()
        ]
        assert [c.get_color() for c in curves] == list(RUN_COLORS[:3])
        assert [c.get_linestyle() for c in curves] == ["-", "--", ":"]
        legend = ax.get_legend()
        assert [t.get_text() for t in legend.get_texts()] == list(histories)
        assert [h.get_linestyle() for h in legend.get_lines()] == ["-", "--", ":"]


def test_run_styles_are_aligned_and_distinct():
    assert len(RUN_LINESTYLES) == len(RUN_COLORS)
    assert len(set(map(str, RUN_LINESTYLES))) == len(RUN_LINESTYLES)
    assert len(set(RUN_COLORS)) == len(RUN_COLORS)


@pytest.mark.parametrize(
    "histories",
    [
        {},
        {"a": _history([0.5], [0.5], best_epoch=1) | {"val_acc": []}},
        {f"run{i}": _history([0.5], [0.5], best_epoch=1) for i in range(len(RUN_COLORS) + 1)},
    ],
)
def test_plot_runs_comparison_rejects_invalid_input(histories):
    with pytest.raises(ValueError):
        plot_runs_comparison(histories)


def test_plot_confusion_matrix_annotates_counts(close_figures):
    cm = torch.tensor([[2, 1, 0], [0, 1, 1], [0, 0, 1]])
    fig = plot_confusion_matrix(cm, CLASSES)

    assert isinstance(fig, Figure)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert texts[0] == "2\n67%"  # fila 0: 2 de 3
    assert texts[4] == "1\n50%"  # fila 1, columna 1: 1 de 2
    with pytest.raises(ValueError):
        plot_confusion_matrix(cm, CLASSES[:2])


def test_plots_module_does_not_import_mlflow():
    code = "import sys, galaxy3.plots; assert 'mlflow' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
