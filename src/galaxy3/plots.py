"""Figuras de Galaxy3 con Matplotlib.

Todas las funciones ``plot_*`` devuelven un ``matplotlib.figure.Figure`` y no muestran,
guardan ni registran nada: la misma figura sirve para el notebook, para ``fig.savefig`` y
para ``mlflow.log_figure`` (que hará ``experiment.py``). Este módulo no conoce MLflow ni
lee archivos. Las figuras se crean con ``pyplot`` para que el notebook las muestre; quien
las guarde fuera de un notebook debe cerrarlas con ``plt.close(fig)``.

Las imágenes de Galaxy3 son composiciones en **falso color** de las bandas g, r, z, no
fotografías RGB; los títulos lo indican. Las imágenes normalizadas se desnormalizan con
``mean_train`` y ``std_train`` solo para visualizarlas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from PIL import Image

from galaxy3.engine import History

FALSE_COLOR_NOTE = "composición en falso color (bandas g, r, z)"

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#e4e3df"
TRAIN_COLOR = "#2a78d6"
VAL_COLOR = "#eb6834"
CORRECT_COLOR = "#0ca30c"
WRONG_COLOR = "#d03b3b"
RUN_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
"""Colores categóricos por run, en orden fijo (el run i siempre recibe ``RUN_COLORS[i]``)."""
RUN_LINESTYLES = ("-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (10, 3)))
"""Estilo de línea por run, en el mismo orden fijo que ``RUN_COLORS``: la identidad del run
no depende solo del color (proyector, escala de grises, daltonismo)."""
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "galaxy3_blues", ["#f5f9fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)


def denormalize(X: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Invierte ``Normalize``: ``x * std + mean`` por canal, seguido de ``clamp(0, 1)``.

    Solo para visualizar. No modifica ``X``.

    Args:
        X: Imagen ``(C, H, W)`` o batch ``(B, C, H, W)`` normalizado.
        mean: Media por canal usada en ``Normalize`` (``mean_train``), longitud ``C``.
        std: Desviación por canal usada en ``Normalize`` (``std_train``), longitud ``C``.

    Returns:
        Tensor ``float32`` en CPU con la forma de ``X`` y valores en ``[0, 1]``.

    Raises:
        ValueError: Si ``X`` no tiene 3 o 4 dimensiones o ``mean``/``std`` no tienen ``C``
            valores.
    """
    if X.ndim not in (3, 4):
        raise ValueError(f"X debe ser (C, H, W) o (B, C, H, W); tiene forma {tuple(X.shape)}")
    channels = X.shape[-3]
    if len(mean) != channels or len(std) != channels:
        raise ValueError(f"mean y std deben tener {channels} valores: mean={mean}, std={std}")
    mean_t = torch.tensor(mean, dtype=torch.float32).view(channels, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(channels, 1, 1)
    return (X.detach().float().cpu() * std_t + mean_t).clamp(0.0, 1.0)


def _to_hwc(image: torch.Tensor) -> np.ndarray:
    """``(C, H, W)`` en ``[0, 1]`` -> arreglo ``(H, W, C)`` para ``imshow``."""
    return image.permute(1, 2, 0).numpy()


def _grid(n: int, ncols: int, cell: float = 1.9) -> tuple[Figure, np.ndarray]:
    ncols = max(1, min(ncols, n))
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(cell * ncols, (cell + 0.35) * nrows + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.flat:
        ax.set_axis_off()
    return fig, axes


def _style_axes(ax: Axes) -> None:
    ax.grid(True, color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
    ax.tick_params(colors=TEXT_SECONDARY)


def _check_batch(X: torch.Tensor, n: int, name: str) -> None:
    if X.ndim != 4:
        raise ValueError(f"X debe ser (B, C, H, W); tiene forma {tuple(X.shape)}")
    if n != X.shape[0]:
        raise ValueError(f"{name} tiene {n} elementos y X tiene {X.shape[0]} imágenes")
    if X.shape[0] == 0:
        raise ValueError("X no contiene imágenes")


def plot_training_curves(history: History) -> Figure:
    """Curvas de loss y accuracy de train y validation por época.

    Loss y accuracy van en paneles separados (escalas distintas). Una línea vertical
    marca ``best_epoch`` (mínimo de ``val_loss``). ``train_*`` se acumula en modo train
    durante la época; ``val_*`` se mide en modo eval al final de la época.

    Args:
        history: Salida de ``engine.fit``.

    Returns:
        Figura con dos paneles.

    Raises:
        ValueError: Si el history está vacío o sus listas tienen longitudes distintas.
    """
    keys = ("train_loss", "val_loss", "train_acc", "val_acc")
    lengths = {len(history[k]) for k in keys}
    if len(lengths) != 1 or 0 in lengths:
        raise ValueError(f"history vacío o inconsistente: { ({k: len(history[k]) for k in keys}) }")
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    panels = ((ax_loss, "loss", "Loss (entropía cruzada)"), (ax_acc, "acc", "Accuracy"))
    for ax, suffix, ylabel in panels:
        _style_axes(ax)
        ax.plot(epochs, history[f"train_{suffix}"], color=TRAIN_COLOR, lw=2, label="train")
        ax.plot(epochs, history[f"val_{suffix}"], color=VAL_COLOR, lw=2, label="validation")
        if history["best_epoch"] >= 1:
            ax.axvline(history["best_epoch"], color=TEXT_SECONDARY, lw=1, ls="--")
        ax.set_xlabel("Época", color=TEXT_PRIMARY)
        ax.set_ylabel(ylabel, color=TEXT_PRIMARY)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    ax_acc.set_ylim(0, 1.0)
    fig.suptitle(
        f"Curvas de entrenamiento (línea discontinua: mejor época = {history['best_epoch']})",
        color=TEXT_PRIMARY,
    )
    return fig


def _check_run_histories(histories: Mapping[str, History]) -> None:
    if not histories or len(histories) > len(RUN_COLORS):
        raise ValueError(f"histories debe tener entre 1 y {len(RUN_COLORS)} runs")
    for name, history in histories.items():
        n_loss, n_acc = len(history["val_loss"]), len(history["val_acc"])
        if n_loss == 0 or n_loss != n_acc:
            raise ValueError(f"history de {name!r} vacío o inconsistente: {n_loss} vs {n_acc}")


def plot_runs_comparison(histories: Mapping[str, History]) -> Figure:
    """Superpone ``val_loss`` y ``val_acc`` de varios runs, con color y estilo fijos por run.

    El run ``i`` (en el orden de ``histories``) usa ``RUN_COLORS[i]`` y
    ``RUN_LINESTYLES[i]``, para que se distinga también sin color. Loss y accuracy van en
    paneles separados (escalas distintas). Un marcador señala la mejor época de cada run
    (mínimo de ``val_loss``) en ambos paneles: en accuracy marca la ``val_acc`` del
    checkpoint que se conserva, no el máximo de ``val_acc``. Los runs pueden tener
    distinto número de épocas.

    Args:
        histories: ``{nombre_run: history}`` en el orden en que se quieren los colores;
            cada history es la salida de ``engine.fit`` (o ``ExperimentResult.history``).

    Returns:
        Figura con dos paneles y una leyenda con los nombres de los runs.

    Raises:
        ValueError: Si no hay runs, hay más de ``len(RUN_COLORS)`` o algún history está
            vacío o tiene ``val_loss`` y ``val_acc`` de longitudes distintas.
    """
    _check_run_histories(histories)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    panels = ((ax_loss, "val_loss", "Loss de validation"), (ax_acc, "val_acc", "Accuracy"))
    for ax, key, ylabel in panels:
        _style_axes(ax)
        styles = zip(RUN_COLORS, RUN_LINESTYLES, strict=True)
        for (color, linestyle), (name, history) in zip(styles, histories.items(), strict=False):
            epochs = np.arange(1, len(history[key]) + 1)
            ax.plot(epochs, history[key], color=color, ls=linestyle, lw=2, label=name)
            best = history["best_epoch"]
            if 1 <= best <= len(history[key]):
                ax.plot(best, history[key][best - 1], "o", color=color, ms=8, mec="white")
        ax.set_xlabel("Época", color=TEXT_PRIMARY)
        ax.set_ylabel(ylabel, color=TEXT_PRIMARY)
        ax.legend(frameon=False, labelcolor=TEXT_PRIMARY)
    ax_acc.set_ylim(0, 1.0)
    fig.suptitle(
        "Comparación de runs en validation (marcador: mejor época por val_loss)",
        color=TEXT_PRIMARY,
    )
    return fig


def plot_confusion_matrix(
    cm: torch.Tensor, class_names: Sequence[str], title: str = "Matriz de confusión"
) -> Figure:
    """Matriz de confusión con conteos y porcentaje por fila (recall de cada clase).

    El color codifica el porcentaje por fila, para que las clases con menos muestras no se
    vean más claras solo por tener menos imágenes.

    Args:
        cm: Matriz ``(K, K)`` no negativa, filas = clase real, columnas = predicha.
        class_names: ``K`` nombres de clase (índice = etiqueta).
        title: Título de la figura.

    Returns:
        Figura con la matriz anotada.

    Raises:
        ValueError: Si ``cm`` no es ``(K, K)`` con ``K == len(class_names)``.
    """
    k = len(class_names)
    if cm.shape != (k, k):
        raise ValueError(f"cm debe ser ({k}, {k}); tiene forma {tuple(cm.shape)}")
    counts = cm.detach().cpu().numpy()
    totals = counts.sum(axis=1, keepdims=True)
    row_frac = np.divide(counts, totals, out=np.zeros(counts.shape), where=totals > 0)

    fig, ax = plt.subplots(figsize=(1.6 * k + 2.4, 1.6 * k + 1.2), constrained_layout=True)
    image = ax.imshow(row_frac, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0)
    for i in range(k):
        for j in range(k):
            ink = "white" if row_frac[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(
                j,
                i,
                f"{counts[i, j]}\n{row_frac[i, j]:.0%}",
                ha="center",
                va="center",
                color=ink,
                fontsize=10,
            )
    ax.set_xticks(range(k), class_names, rotation=30, ha="right", color=TEXT_PRIMARY)
    ax.set_yticks(range(k), class_names, color=TEXT_PRIMARY)
    ax.set_xlabel("Clase predicha", color=TEXT_PRIMARY)
    ax.set_ylabel("Clase real", color=TEXT_PRIMARY)
    fig.colorbar(image, ax=ax, label="Fracción de la fila", shrink=0.8)
    ax.set_title(title, color=TEXT_PRIMARY)
    return fig


def plot_original_galaxies(images_by_class: Mapping[str, Sequence[Image.Image]]) -> Figure:
    """Grilla de imágenes originales, antes de cualquier transformación: una fila por clase.

    Args:
        images_by_class: ``{class_name: [PIL.Image, ...]}``, por ejemplo la salida de
            ``data.load_original_samples``. Todas las clases con el mismo número (>= 1).

    Returns:
        Figura con ``len(images_by_class)`` filas.

    Raises:
        ValueError: Si no hay clases o alguna lista está vacía o tiene otra longitud.
    """
    per_class = {len(images) for images in images_by_class.values()}
    if not images_by_class or len(per_class) != 1 or 0 in per_class:
        raise ValueError(
            "images_by_class debe tener >= 1 clase y la misma cantidad (>= 1) por clase"
        )
    ncols = per_class.pop()
    fig, axes = plt.subplots(
        len(images_by_class),
        ncols,
        figsize=(2.0 * ncols + 1.4, 2.1 * len(images_by_class) + 0.5),
        squeeze=False,
        constrained_layout=True,
    )
    sizes = set()
    for row, (name, images) in zip(axes, images_by_class.items(), strict=True):
        for col, (ax, image) in enumerate(zip(row, images, strict=True)):
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            sizes.add(f"{image.size[0]}x{image.size[1]}")
            if col == 0:
                ax.set_ylabel(name, color=TEXT_PRIMARY, fontsize=11)
    fig.suptitle(
        f"Galaxias originales ({', '.join(sorted(sizes))} px, sin transformar) — "
        f"{FALSE_COLOR_NOTE}",
        color=TEXT_PRIMARY,
    )
    return fig


def plot_image_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    max_images: int = 16,
    ncols: int = 8,
) -> Figure:
    """Grilla de un batch transformado, desnormalizado solo para visualizar.

    Args:
        X: Batch normalizado ``(B, 3, H, W)``.
        y: Etiquetas ``(B,)``.
        class_names: Nombres de clase (índice = etiqueta).
        mean: ``mean_train`` usada en ``Normalize``.
        std: ``std_train`` usada en ``Normalize``.
        max_images: Máximo de imágenes a mostrar (las primeras del batch).
        ncols: Columnas de la grilla.

    Returns:
        Figura con ``min(B, max_images)`` imágenes tituladas con su clase.

    Raises:
        ValueError: Si las formas no son coherentes.
    """
    _check_batch(X, len(y), "y")
    n = min(len(y), max_images)
    images = denormalize(X[:n], mean, std)
    fig, axes = _grid(n, ncols)
    for i, ax in enumerate(axes.flat[:n]):
        ax.imshow(_to_hwc(images[i]))
        ax.set_title(class_names[int(y[i])], fontsize=8, color=TEXT_PRIMARY)
    height, width = X.shape[-2:]
    fig.suptitle(
        f"Batch transformado ({height}x{width}, desnormalizado para visualizar) — "
        f"{FALSE_COLOR_NOTE}",
        color=TEXT_PRIMARY,
    )
    return fig


def plot_predictions(
    X: torch.Tensor,
    y_true: torch.Tensor,
    probabilities: torch.Tensor,
    class_names: Sequence[str],
    mean: Sequence[float],
    std: Sequence[float],
    max_images: int = 16,
    ncols: int = 4,
    title: str = "Predicciones",
) -> Figure:
    """Imágenes con clase real, clase predicha y probabilidad de la predicción.

    ``probabilities`` es el softmax de los logits, calculado solo para análisis (por
    ejemplo, ``metrics.collect_predictions``); el modelo sigue devolviendo logits. Los
    aciertos llevan marco verde y ``✓``; los errores, marco rojo y ``✗``. Para mostrar
    ejemplos de errores, pasar solo las muestras de ``metrics.misclassified_indices``.

    Args:
        X: Imágenes normalizadas ``(N, 3, H, W)``.
        y_true: Etiquetas reales ``(N,)``.
        probabilities: ``(N, K)``, filas que suman 1.
        class_names: ``K`` nombres de clase.
        mean: ``mean_train``.
        std: ``std_train``.
        max_images: Máximo de imágenes a mostrar.
        ncols: Columnas de la grilla.
        title: Título principal; se le añade la nota de falso color.

    Returns:
        Figura con ``min(N, max_images)`` imágenes.

    Raises:
        ValueError: Si las formas no son coherentes o ``probabilities`` no tiene ``K``
            columnas.
    """
    _check_batch(X, len(y_true), "y_true")
    if probabilities.shape != (len(y_true), len(class_names)):
        raise ValueError(
            f"probabilities debe ser ({len(y_true)}, {len(class_names)}); "
            f"tiene forma {tuple(probabilities.shape)}"
        )
    n = min(len(y_true), max_images)
    images = denormalize(X[:n], mean, std)
    confidence, y_pred = probabilities[:n].max(dim=1)
    fig, axes = _grid(n, ncols, cell=2.4)
    for i, ax in enumerate(axes.flat[:n]):
        correct = int(y_pred[i]) == int(y_true[i])
        ax.imshow(_to_hwc(images[i]))
        ax.set_axis_on()
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(CORRECT_COLOR if correct else WRONG_COLOR)
            spine.set_linewidth(3)
        ax.set_title(
            f"{'✓' if correct else '✗'} real: {class_names[int(y_true[i])]}\n"
            f"pred: {class_names[int(y_pred[i])]} (p={float(confidence[i]):.2f})",
            fontsize=8,
            color=TEXT_PRIMARY,
        )
    fig.suptitle(f"{title} — {FALSE_COLOR_NOTE}", color=TEXT_PRIMARY)
    return fig


def _normalize_per_map(maps: torch.Tensor) -> torch.Tensor:
    """Escala cada mapa de ``(C, H, W)`` a ``[0, 1]`` por separado; los constantes quedan en 0."""
    low = maps.amin(dim=(1, 2), keepdim=True)
    span = maps.amax(dim=(1, 2), keepdim=True) - low
    return torch.where(span > 0, (maps - low) / span.clamp_min(1e-12), torch.zeros_like(maps))


def plot_feature_maps(
    feature_maps: torch.Tensor,
    sample_index: int = 0,
    max_channels: int = 16,
    ncols: int = 8,
    layer_name: str | None = None,
) -> Figure:
    """Primeros canales de los feature maps de una muestra, en escala de grises.

    Cada mapa se normaliza por separado a ``[0, 1]`` (normalización visual): el brillo
    solo es comparable dentro de un mismo mapa, no entre mapas. Un mapa constante (por
    ejemplo, un canal ReLU completamente en 0) se muestra negro y se marca como
    "constante". Los canales no tienen un significado semántico asignado.

    Args:
        feature_maps: Activaciones ``(B, C, H, W)``, por ejemplo de
            ``inspection.capture_feature_maps``.
        sample_index: Muestra del batch a visualizar, en ``[0, B)``.
        max_channels: Máximo de canales a mostrar (los primeros).
        ncols: Columnas de la grilla.
        layer_name: Nombre de la capa, para el título.

    Returns:
        Figura con ``min(C, max_channels)`` mapas.

    Raises:
        ValueError: Si ``feature_maps`` no es 4D, ``sample_index`` está fuera de rango o
            ``max_channels < 1``.
    """
    if feature_maps.ndim != 4:
        raise ValueError(f"feature_maps debe ser (B, C, H, W); tiene {tuple(feature_maps.shape)}")
    if not 0 <= sample_index < feature_maps.shape[0]:
        raise ValueError(f"sample_index={sample_index} fuera de [0, {feature_maps.shape[0]})")
    if max_channels < 1:
        raise ValueError(f"max_channels debe ser >= 1; se recibió {max_channels}")
    _, channels, height, width = feature_maps.shape
    n = min(channels, max_channels)
    maps = feature_maps[sample_index, :n].detach().float().cpu()
    normalized = _normalize_per_map(maps)
    constant = (maps.amax(dim=(1, 2)) == maps.amin(dim=(1, 2))).tolist()
    fig, axes = _grid(n, ncols, cell=1.6)
    for c, ax in enumerate(axes.flat[:n]):
        ax.imshow(normalized[c].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        label = f"canal {c}" + (" (constante)" if constant[c] else "")
        ax.set_title(label, fontsize=7, color=TEXT_PRIMARY)
    layer = f"{layer_name} — " if layer_name else ""
    fig.suptitle(
        f"Feature maps: {layer}muestra {sample_index}, {n} de {channels} canales, "
        f"{height}x{width}\n(escala de grises, normalización visual por mapa)",
        color=TEXT_PRIMARY,
    )
    return fig
