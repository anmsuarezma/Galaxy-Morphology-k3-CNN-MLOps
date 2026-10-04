import json
from dataclasses import replace

import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from galaxy3.config import ExperimentConfig
from galaxy3.data import (
    DATASET_STATS_NAME,
    build_base_transform,
    compute_channel_stats,
    load_original_samples,
    make_dataloaders,
)

CLASSES = ["barred_spiral", "edge_on_no_bulge", "round_smooth"]
PER_CLASS = {"train": 10, "val": 4, "test": 3}
IMAGE_SIZE = 8
SPLIT_SEED = 2026


def _class_red(label: int) -> int:
    return 60 * (label + 1)


@pytest.fixture
def image_folder(tmp_path):
    """ImageFolder sintético: R codifica la clase, G identifica la imagen (único por split)."""
    for split, n in PER_CLASS.items():
        for label, name in enumerate(CLASSES):
            folder = tmp_path / split / name
            folder.mkdir(parents=True)
            for i in range(n):
                color = (_class_red(label), label * n + i, 0)
                Image.new("RGB", (24, 24), color).save(folder / f"{i:05d}.png")
    stats = {
        "split_seed": SPLIT_SEED,
        "class_to_label": {name: i for i, name in enumerate(CLASSES)},
        "transform": {"image_size": IMAGE_SIZE, "center_crop": None},
        "mean_train": [0.0, 0.0, 0.0],
        "std_train": [1.0, 1.0, 1.0],
    }
    (tmp_path / DATASET_STATS_NAME).write_text(json.dumps(stats))
    return tmp_path


@pytest.fixture
def config(image_folder):
    return ExperimentConfig(
        data_root=image_folder,
        image_size=IMAGE_SIZE,
        batch_size=4,
        num_workers=0,
        device_preference="cpu",
    )


def _image_ids(loader):
    """Identificador de cada imagen (canal G) en el orden en que llega."""
    return [round(v * 255) for X, _ in loader for v in X[:, 1, 0, 0].tolist()]


def _write_stats(root, **changes):
    path = root / DATASET_STATS_NAME
    stats = json.loads(path.read_text())
    stats.update(changes)
    path.write_text(json.dumps(stats))


def test_base_transform_shape_and_range():
    image = Image.new("RGB", (256, 256), color=(255, 0, 128))
    X = build_base_transform(128)(image)
    assert X.shape == (3, 128, 128)
    assert X.dtype == torch.float32
    assert X.min() >= 0.0
    assert X.max() <= 1.0


def test_base_transform_center_crop_runs_before_resize():
    image = Image.new("RGB", (256, 256))
    assert build_base_transform(64, center_crop=200)(image).shape == (3, 64, 64)


@pytest.mark.parametrize(("size", "crop"), [(0, None), (128, 0)])
def test_base_transform_rejects_invalid_sizes(size, crop):
    with pytest.raises(ValueError):
        build_base_transform(size, crop)


def test_channel_stats_match_direct_computation():
    X = torch.rand(10, 3, 8, 8)
    dataset = TensorDataset(X, torch.zeros(10, dtype=torch.long))
    mean, std = compute_channel_stats(dataset, batch_size=3)
    flat = X.to(torch.float64).permute(1, 0, 2, 3).reshape(3, -1)
    torch.testing.assert_close(torch.tensor(mean, dtype=torch.float64), flat.mean(dim=1))
    torch.testing.assert_close(
        torch.tensor(std, dtype=torch.float64), flat.std(dim=1, correction=0)
    )


def test_channel_stats_empty_dataset_raises():
    empty = TensorDataset(torch.empty(0, 3, 8, 8), torch.empty(0))
    with pytest.raises(ValueError, match="vacío"):
        compute_channel_stats(empty)


def test_batch_shapes_and_dtypes(config):
    loaders = make_dataloaders(config)
    for loader in (loaders.train, loaders.val, loaders.test):
        X, y = next(iter(loader))
        assert X.ndim == 4
        assert X.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert y.ndim == 1
        assert X.shape[0] == y.shape[0] == config.batch_size
        assert X.dtype == torch.float32
        assert y.dtype == torch.int64


def test_class_names_and_split_sizes(config):
    loaders = make_dataloaders(config)
    assert loaders.class_names == CLASSES
    assert len(loaders.class_names) == config.num_classes
    for split, n in PER_CLASS.items():
        assert len(getattr(loaders, split).dataset) == n * len(CLASSES)


def test_labels_match_images(config):
    loaders = make_dataloaders(config)
    for loader in (loaders.train, loaders.val, loaders.test):
        for X, y in loader:
            red = (X[:, 0, 0, 0] * 255).round().long()
            assert torch.equal(red, 60 * (y + 1))


def test_normalize_uses_stats_from_json(config):
    _write_stats(config.data_root, mean_train=[0.5, 0.5, 0.5], std_train=[0.25, 0.25, 0.25])
    X, y = next(iter(make_dataloaders(config).val))
    expected = (torch.tensor([_class_red(int(c)) for c in y]) / 255 - 0.5) / 0.25
    torch.testing.assert_close(X[:, 0, 0, 0], expected)


def test_same_train_seed_same_batch_order(config):
    first = _image_ids(make_dataloaders(config).train)
    second = _image_ids(make_dataloaders(config).train)
    assert first == second
    other = _image_ids(make_dataloaders(replace(config, train_seed=config.train_seed + 1)).train)
    assert other != first
    assert sorted(other) == sorted(first)


def test_train_order_independent_of_global_rng(config):
    torch.manual_seed(0)
    first = _image_ids(make_dataloaders(config).train)
    torch.manual_seed(123)
    assert _image_ids(make_dataloaders(config).train) == first


def test_train_shuffles_val_and_test_do_not(config):
    loaders = make_dataloaders(config)
    epoch1, epoch2 = _image_ids(loaders.train), _image_ids(loaders.train)
    assert epoch1 != epoch2  # el generador avanza entre épocas
    for loader in (loaders.val, loaders.test):
        ids = _image_ids(loader)
        assert ids == sorted(ids)


def test_order_with_workers_matches_seed(config):
    with_workers = replace(config, num_workers=2)
    loaders = make_dataloaders(with_workers)
    assert loaders.train.persistent_workers
    assert _image_ids(loaders.train) == _image_ids(make_dataloaders(config).train)


def test_loader_flags_depend_on_device_and_workers(config):
    cpu = make_dataloaders(config, torch.device("cpu"))
    assert not cpu.train.pin_memory
    assert not cpu.train.persistent_workers
    cuda = make_dataloaders(config, torch.device("cuda"))  # solo construye, no itera
    assert cuda.train.pin_memory
    assert cuda.val.pin_memory


def test_split_seed_mismatch_raises(config):
    with pytest.raises(ValueError, match="split_seed"):
        make_dataloaders(replace(config, split_seed=SPLIT_SEED + 1))


@pytest.mark.parametrize(
    "changes", [{"image_size": IMAGE_SIZE * 2}, {"center_crop": 16}], ids=["size", "crop"]
)
def test_transform_mismatch_raises(config, changes):
    with pytest.raises(ValueError, match=next(iter(changes))):
        make_dataloaders(replace(config, **changes))


def test_num_classes_mismatch_raises(config):
    with pytest.raises(ValueError, match="num_classes"):
        make_dataloaders(replace(config, num_classes=4))


def test_class_mapping_mismatch_raises(config):
    _write_stats(config.data_root, class_to_label={"a": 0, "b": 1, "c": 2})
    with pytest.raises(ValueError, match="Clases"):
        make_dataloaders(config)


def test_missing_stats_raises(config):
    (config.data_root / DATASET_STATS_NAME).unlink()
    with pytest.raises(FileNotFoundError, match="prepare_galaxy3"):
        make_dataloaders(config)


def test_load_original_samples_untransformed_and_reproducible(image_folder):
    samples = load_original_samples(image_folder, split="val", per_class=2, seed=0)

    assert list(samples) == CLASSES
    for label, images in enumerate(samples.values()):
        assert len(images) == 2
        assert all(image.size == (24, 24) and image.mode == "RGB" for image in images)
        assert all(image.getpixel((0, 0))[0] == _class_red(label) for image in images)
    again = load_original_samples(image_folder, split="val", per_class=2, seed=0)
    ids = [[im.getpixel((0, 0))[1] for im in v] for v in samples.values()]
    assert ids == [[im.getpixel((0, 0))[1] for im in v] for v in again.values()]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"split": "all"}, "split"), ({"per_class": 0}, "per_class"), ({"per_class": 5}, "< 5")],
)
def test_load_original_samples_rejects_invalid_arguments(image_folder, kwargs, match):
    with pytest.raises(ValueError, match=match):
        load_original_samples(image_folder, **{"split": "val", **kwargs})
