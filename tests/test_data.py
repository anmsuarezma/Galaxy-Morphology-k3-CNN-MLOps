import pytest
import torch
from PIL import Image
from torch.utils.data import TensorDataset

from galaxy3.data import build_base_transform, compute_channel_stats


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
