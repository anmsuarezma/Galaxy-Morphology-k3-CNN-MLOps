import json

import h5py
import numpy as np
import pandas as pd
import pytest
from torchvision.datasets import ImageFolder

from galaxy3.preparation import (
    GALAXY3_CLASSES,
    MANIFEST_NAME,
    SPLIT_FRACTIONS,
    SPLITS,
    STATS_NAME,
    class_to_label,
    prepare,
    split_indices,
)

SIZE = 16
# Clases de Galaxy3 más dos clases ajenas (0 y 3) que deben quedar fuera.
SYNTHETIC_COUNTS = {0: 10, 2: 40, 3: 15, 5: 60, 8: 100}
SEED = 2026


def _write_synthetic_h5(path, seed=0):
    """``.h5`` pequeño con el esquema de Galaxy10: cada imagen es un color constante."""
    rng = np.random.default_rng(seed)
    labels = np.concatenate([np.full(n, c, dtype=np.uint8) for c, n in SYNTHETIC_COUNTS.items()])
    rng.shuffle(labels)
    colors = rng.integers(0, 256, size=(labels.size, 1, 1, 3), dtype=np.uint8)
    images = np.broadcast_to(colors, (labels.size, SIZE, SIZE, 3)).copy()
    with h5py.File(path, "w") as f:
        f.create_dataset("images", data=images, chunks=(8, SIZE, SIZE, 1), compression="gzip")
        f.create_dataset("ans", data=labels)
    return labels, colors[:, 0, 0, :]


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "synthetic.h5"
    labels, colors = _write_synthetic_h5(path)
    return path, labels, colors


@pytest.fixture(scope="module")
def prepared(source, tmp_path_factory):
    path, labels, colors = source
    out = tmp_path_factory.mktemp("prep") / "galaxy3"
    stats = prepare(path, out, split_seed=SEED, image_size=SIZE)
    manifest = pd.read_csv(out / MANIFEST_NAME)
    return out, stats, manifest, labels, colors


def test_splits_are_disjoint_and_cover_selected_classes(prepared):
    _, _, manifest, labels, _ = prepared
    sets = {s: set(manifest.loc[manifest.split == s, "h5_index"]) for s in SPLITS}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]
    selected = set(np.flatnonzero(np.isin(labels, list(GALAXY3_CLASSES))))
    assert set().union(*sets.values()) == selected
    assert manifest["h5_index"].is_unique


def test_class_proportions_per_split(prepared):
    _, _, manifest, _, _ = prepared
    counts = manifest.groupby(["class_idx_h5", "split"]).size()
    for h5_class, n_class in SYNTHETIC_COUNTS.items():
        if h5_class not in GALAXY3_CLASSES:
            continue
        for split, frac in SPLIT_FRACTIONS.items():
            assert abs(counts[(h5_class, split)] - frac * n_class) <= 2


def test_manifest_labels_match_imagefolder(prepared):
    out, stats, manifest, labels, _ = prepared
    assert (manifest["class_idx_h5"] == labels[manifest["h5_index"]]).all()
    for split in SPLITS:
        dataset = ImageFolder(out / split)
        assert dataset.class_to_idx == stats["class_to_label"]
        assert len(dataset) == (manifest.split == split).sum()
    for row in manifest.itertuples():
        assert (out / row.relpath).is_file()
        assert row.label == stats["class_to_label"][row.class_name]


def test_stats_json_contents(prepared):
    out, stats, manifest, _, _ = prepared
    on_disk = json.loads((out / STATS_NAME).read_text())
    assert on_disk == stats
    assert stats["split_seed"] == SEED
    assert stats["class_to_label"] == class_to_label(GALAXY3_CLASSES)
    assert {c["h5_class_index"]: c["class_name"] for c in stats["classes"]} == GALAXY3_CLASSES
    assert stats["total"] == len(manifest)
    assert stats["transform"] == {
        "image_size": SIZE,
        "center_crop": None,
        "ops": ["Resize", "ToTensor"],
    }


def test_mean_std_use_only_train(prepared):
    _, stats, manifest, _, colors = prepared
    values = colors.astype(np.float64) / 255.0  # imágenes constantes: un color por imagen

    def mean_std(idx):
        return values[idx].mean(axis=0), values[idx].std(axis=0)

    train_mean, train_std = mean_std(manifest.loc[manifest.split == "train", "h5_index"])
    np.testing.assert_allclose(stats["mean_train"], train_mean, atol=1e-6)
    np.testing.assert_allclose(stats["std_train"], train_std, atol=1e-6)
    all_mean, _ = mean_std(manifest["h5_index"])
    assert not np.allclose(stats["mean_train"], all_mean, atol=1e-6)


def test_same_split_seed_same_split(source, prepared, tmp_path):
    path = source[0]
    manifest = prepared[2]
    prepare(path, tmp_path / "again", split_seed=SEED, image_size=SIZE)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "again" / MANIFEST_NAME), manifest)
    prepare(path, tmp_path / "other", split_seed=SEED + 1, image_size=SIZE)
    other = pd.read_csv(tmp_path / "other" / MANIFEST_NAME)
    train_ids = set(manifest.loc[manifest.split == "train", "h5_index"])
    assert set(other.loc[other.split == "train", "h5_index"]) != train_ids


def test_existing_output_requires_overwrite(source, tmp_path):
    path = source[0]
    out = tmp_path / "galaxy3"
    prepare(path, out, split_seed=SEED, image_size=SIZE)
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare(path, out, split_seed=SEED, image_size=SIZE)
    prepare(path, out, split_seed=SEED, image_size=SIZE, overwrite=True)
    assert (out / MANIFEST_NAME).is_file()
    assert not out.with_name("galaxy3.partial").exists()


def test_overwrite_refuses_foreign_directory(source, tmp_path):
    out = tmp_path / "galaxy3"
    out.mkdir()
    (out / "algo.txt").write_text("no es de prepare")
    with pytest.raises(FileExistsError, match="no se borra"):
        prepare(source[0], out, split_seed=SEED, image_size=SIZE, overwrite=True)
    assert (out / "algo.txt").is_file()


def test_unexpected_counts_raise(source, tmp_path):
    with pytest.raises(ValueError, match="Conteos"):
        prepare(
            source[0],
            tmp_path / "galaxy3",
            split_seed=SEED,
            image_size=SIZE,
            expected_counts={2: 2645, 5: 2043, 8: 1423},
        )


@pytest.mark.parametrize(
    "fractions",
    [
        {"train": 0.7, "val": 0.2, "test": 0.2},
        {"train": 0.7, "val": 0.3},
        {"train": 1.0, "val": 0.0, "test": 0.0},
    ],
)
def test_invalid_fractions_raise(fractions):
    labels = np.repeat([2, 5, 8], 20).astype(np.uint8)
    with pytest.raises(ValueError):
        split_indices(np.arange(labels.size), labels, SEED, fractions)


def test_split_sizes_follow_fractions():
    labels = np.repeat([2, 5, 8], 100).astype(np.uint8)
    fractions = {"train": 0.6, "val": 0.2, "test": 0.2}
    splits = split_indices(np.arange(labels.size), labels, SEED, fractions)
    assert {s: len(v) for s, v in splits.items()} == {"train": 180, "val": 60, "test": 60}
