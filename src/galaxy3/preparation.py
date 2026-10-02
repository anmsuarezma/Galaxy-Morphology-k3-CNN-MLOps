"""Preparación de Galaxy3 a partir de Galaxy10 DECaLS (se ejecuta una sola vez).

Pasos de ``prepare``:
    1. lee las etiquetas del ``.h5`` fuente;
    2. selecciona las tres clases de Galaxy3 (todas sus imágenes, sin submuestrear);
    3. split estratificado train/val/test según ``SPLIT_FRACTIONS`` con ``split_seed``;
    4. exporta PNG sin pérdida, a la resolución original, en estructura ``ImageFolder``;
    5. escribe ``manifest.csv`` (una fila por imagen, con su índice en el ``.h5``);
    6. calcula ``mean_train``/``std_train`` solo con train y escribe ``dataset_stats.json``.

La salida se construye en ``<output>.partial`` y solo se renombra al terminar, para no
dejar un dataset a medias. Si ``<output>`` ya existe, falla salvo ``overwrite=True``.
El CLI está en ``scripts/prepare_galaxy3.py``.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder

from galaxy3.data import build_base_transform, compute_channel_stats

GALAXY3_CLASSES: dict[int, str] = {
    2: "round_smooth",
    5: "barred_spiral",
    8: "edge_on_no_bulge",
}
"""Índice de clase en Galaxy10 DECaLS (``ans``) -> nombre de carpeta en Galaxy3.

Los nombres provienen de la documentación de astroNN; el ``.h5`` no los almacena.
"""

EXPECTED_COUNTS: dict[int, int] = {2: 2645, 5: 2043, 8: 1423}
"""Conteos documentados por astroNN; se verifican contra el archivo real."""

SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}
H5_IMAGES_KEY = "images"
H5_LABELS_KEY = "ans"
READ_BLOCK = 1110  # múltiplo del chunk del .h5 en el eje 0 (555)
MANIFEST_NAME = "manifest.csv"
STATS_NAME = "dataset_stats.json"


def class_to_label(class_map: dict[int, str]) -> dict[str, int]:
    """Etiqueta entera de cada clase, igual a la que asigna ``ImageFolder`` (orden alfabético)."""
    return {name: label for label, name in enumerate(sorted(class_map.values()))}


def read_labels(source: Path) -> np.ndarray:
    """Lee el vector de etiquetas ``ans`` completo, forma ``(N,)``."""
    with h5py.File(source, "r") as f:
        return f[H5_LABELS_KEY][:]


def select_indices(labels: np.ndarray, class_map: dict[int, str]) -> np.ndarray:
    """Índices del ``.h5`` (ordenados) cuya clase pertenece a ``class_map``."""
    return np.flatnonzero(np.isin(labels, list(class_map)))


def check_counts(labels: np.ndarray, expected: dict[int, int]) -> None:
    """Verifica los conteos por clase.

    Raises:
        ValueError: Si algún conteo difiere del esperado.
    """
    actual = {c: int((labels == c).sum()) for c in expected}
    if actual != expected:
        raise ValueError(f"Conteos por clase inesperados: {actual}; esperados {expected}")


def _check_fractions(fractions: dict[str, float]) -> None:
    if set(fractions) != set(SPLITS):
        raise ValueError(f"Las fracciones deben tener las claves {SPLITS}: {sorted(fractions)}")
    if any(not 0.0 < v < 1.0 for v in fractions.values()):
        raise ValueError(f"Cada fracción debe estar en (0, 1): {fractions}")
    if not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError(f"Las fracciones deben sumar 1: {fractions}")


def split_indices(
    indices: np.ndarray,
    labels: np.ndarray,
    split_seed: int,
    fractions: dict[str, float] = SPLIT_FRACTIONS,
) -> dict[str, np.ndarray]:
    """Split estratificado train/val/test, reproducible con ``split_seed``.

    Dos etapas, ambas estratificadas: train vs resto, y luego el resto en val y test,
    estratificando sobre las etiquetas del propio resto. Los ``test_size`` se derivan
    de ``fractions``.

    Args:
        indices: Índices del ``.h5`` seleccionados, forma ``(M,)``.
        labels: Etiquetas ``ans`` completas, forma ``(N,)``.
        split_seed: Semilla del split (no es ``train_seed``).
        fractions: ``{"train", "val", "test"}`` -> fracción; deben sumar 1.

    Returns:
        ``{"train": idx, "val": idx, "test": idx}``, cada uno ordenado y disjunto.

    Raises:
        ValueError: Si las fracciones no son válidas.
    """
    _check_fractions(fractions)
    rest_size = 1.0 - fractions["train"]
    test_within_rest = fractions["test"] / rest_size
    train, rest = train_test_split(
        indices, test_size=rest_size, stratify=labels[indices], random_state=split_seed
    )
    val, test = train_test_split(
        rest, test_size=test_within_rest, stratify=labels[rest], random_state=split_seed
    )
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


def build_manifest(
    splits: dict[str, np.ndarray], labels: np.ndarray, class_map: dict[int, str]
) -> pd.DataFrame:
    """Una fila por imagen: ``h5_index, split, class_idx_h5, class_name, label, relpath``."""
    to_label = class_to_label(class_map)
    frames = []
    for split in SPLITS:
        idx = splits[split]
        names = [class_map[int(c)] for c in labels[idx]]
        relpaths = [f"{split}/{n}/{i:05d}.png" for i, n in zip(idx, names, strict=True)]
        frames.append(
            pd.DataFrame(
                {
                    "h5_index": idx,
                    "split": split,
                    "class_idx_h5": labels[idx].astype(int),
                    "class_name": names,
                    "label": [to_label[n] for n in names],
                    "relpath": relpaths,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def export_images(source: Path, manifest: pd.DataFrame, out_dir: Path) -> None:
    """Escribe cada imagen del manifest como PNG en ``out_dir / relpath``.

    Lee ``images`` por bloques secuenciales: el ``.h5`` está comprimido por chunks y el
    acceso aleatorio imagen por imagen es muy lento.

    Raises:
        ValueError: Si las imágenes no tienen forma ``(N, H, W, 3)`` y dtype ``uint8``.
    """
    for relpath in manifest["relpath"]:
        (out_dir / relpath).parent.mkdir(parents=True, exist_ok=True)
    wanted = manifest.set_index("h5_index")["relpath"]
    with h5py.File(source, "r") as f:
        images = f[H5_IMAGES_KEY]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f"Formato de imágenes inesperado: {images.shape} {images.dtype}")
        for start in range(0, images.shape[0], READ_BLOCK):
            stop = min(start + READ_BLOCK, images.shape[0])
            in_block = wanted[(wanted.index >= start) & (wanted.index < stop)]
            if in_block.empty:
                continue
            block = images[start:stop]
            for h5_index, relpath in in_block.items():
                Image.fromarray(block[h5_index - start]).save(out_dir / relpath)


def train_stats(
    out_dir: Path, image_size: int, center_crop: int | None, num_workers: int
) -> tuple[list[float], list[float]]:
    """``mean_train`` y ``std_train`` por canal, calculadas **solo** con ``out_dir/train``."""
    transform = build_base_transform(image_size, center_crop)
    dataset = ImageFolder(out_dir / "train", transform=transform)
    return compute_channel_stats(dataset, num_workers=num_workers)


def counts_table(manifest: pd.DataFrame) -> dict[str, dict[str, int]]:
    """``{split: {class_name: n}}`` a partir del manifest."""
    table = manifest.groupby(["split", "class_name"]).size()
    return {s: {c: int(n) for c, n in table[s].items()} for s in SPLITS}


def build_stats(
    manifest: pd.DataFrame,
    class_map: dict[int, str],
    source: Path,
    split_seed: int,
    image_size: int,
    center_crop: int | None,
    mean_std: tuple[list[float], list[float]],
) -> dict[str, Any]:
    """Contenido de ``dataset_stats.json`` (solo tipos JSON)."""
    to_label = class_to_label(class_map)
    classes = sorted(class_map.items(), key=lambda kv: to_label[kv[1]])
    return {
        "split_seed": split_seed,
        "split_fractions": SPLIT_FRACTIONS,
        "source_file": source.name,
        "classes": [
            {"class_name": name, "label": to_label[name], "h5_class_index": h5_idx}
            for h5_idx, name in classes
        ],
        "class_to_label": to_label,
        "counts": counts_table(manifest),
        "total": len(manifest),
        "transform": {
            "image_size": image_size,
            "center_crop": center_crop,
            "ops": (["CenterCrop"] if center_crop else []) + ["Resize", "ToTensor"],
        },
        "mean_train": mean_std[0],
        "std_train": mean_std[1],
    }


def check_imagefolder_labels(out_dir: Path, class_map: dict[int, str]) -> None:
    """Verifica que ``ImageFolder`` asigne las mismas etiquetas que el manifest.

    Raises:
        ValueError: Si los mapeos difieren.
    """
    expected = class_to_label(class_map)
    for split in SPLITS:
        found = ImageFolder(out_dir / split).class_to_idx
        if found != expected:
            raise ValueError(f"ImageFolder ({split}) asigna {found}; se esperaba {expected}")


def resolve_output(output: Path, overwrite: bool) -> Path:
    """Valida el destino y devuelve la carpeta temporal donde construir la salida.

    Raises:
        FileExistsError: Si ``output`` existe y no se pasó ``overwrite``, o si existe pero
            no parece generado por este módulo (no contiene ``manifest.csv``).
    """
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"{output} ya existe. No se sobrescribe en silencio; usa --overwrite para regenerarlo."
        )
    if output.exists() and not (output / MANIFEST_NAME).is_file():
        raise FileExistsError(
            f"{output} existe pero no contiene {MANIFEST_NAME}; no se borra por seguridad."
        )
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    return partial


def finalize(partial: Path, output: Path) -> None:
    """Reemplaza ``output`` por ``partial`` una vez que todo terminó bien."""
    if output.exists():
        shutil.rmtree(output)
    partial.rename(output)


def prepare(
    source: Path,
    output: Path,
    split_seed: int,
    image_size: int,
    center_crop: int | None = None,
    overwrite: bool = False,
    expected_counts: dict[int, int] | None = None,
    num_workers: int = 0,
    class_map: dict[int, str] = GALAXY3_CLASSES,
) -> dict[str, Any]:
    """Ejecuta la preparación completa y devuelve el contenido de ``dataset_stats.json``.

    Args:
        source: Ruta al ``.h5`` de Galaxy10 DECaLS.
        output: Carpeta final con ``train/``, ``val/``, ``test/``.
        split_seed: Semilla del split.
        image_size: Resolución usada para calcular ``mean_train``/``std_train``.
        center_crop: Crop previo al ``Resize`` para las estadísticas, o ``None``.
        overwrite: Si se permite reemplazar ``output`` existente.
        expected_counts: Conteos por clase a verificar, o ``None`` para no verificar.
        num_workers: Workers del ``DataLoader`` al calcular estadísticas.
        class_map: Índice de clase en el ``.h5`` -> nombre de carpeta.

    Returns:
        El diccionario escrito en ``dataset_stats.json``.

    Raises:
        FileExistsError: Ver ``resolve_output``.
        ValueError: Si los conteos o el formato del ``.h5`` no son los esperados.
    """
    partial = resolve_output(output, overwrite)
    labels = read_labels(source)
    if expected_counts is not None:
        check_counts(labels, expected_counts)
    splits = split_indices(select_indices(labels, class_map), labels, split_seed)
    manifest = build_manifest(splits, labels, class_map)
    export_images(source, manifest, partial)
    check_imagefolder_labels(partial, class_map)
    mean_std = train_stats(partial, image_size, center_crop, num_workers)
    stats = build_stats(manifest, class_map, source, split_seed, image_size, center_crop, mean_std)
    manifest.to_csv(partial / MANIFEST_NAME, index=False)
    (partial / STATS_NAME).write_text(json.dumps(stats, indent=2) + "\n")
    finalize(partial, output)
    return stats
