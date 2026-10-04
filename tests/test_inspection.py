"""Tests de ``capture_feature_maps`` con GalaxyCNN y datos sintéticos."""

import pytest
import torch

from galaxy3.inspection import capture_feature_maps
from galaxy3.models import GalaxyCNN

CPU = torch.device("cpu")


def _model(use_batchnorm: bool = True) -> GalaxyCNN:
    torch.manual_seed(0)
    return GalaxyCNN(num_classes=3, use_batchnorm=use_batchnorm, dropout_p=0.3)


def _batch(n: int = 2, size: int = 128) -> torch.Tensor:
    return torch.randn(n, 3, size, size, generator=torch.Generator().manual_seed(0))


def _hook_count(model: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) for m in model.modules())


@pytest.mark.parametrize(
    ("layer", "shape"),
    [
        ("features.block1.act", (2, 32, 128, 128)),
        ("features.block2.act", (2, 64, 64, 64)),
        ("features.block3.act", (2, 128, 32, 32)),
        ("features.block4.act", (2, 256, 16, 16)),
        ("features.block1", (2, 32, 64, 64)),  # bloque completo: después del pooling
    ],
)
def test_shapes_match_spec(layer, shape):
    feature_maps = capture_feature_maps(_model(), _batch(), layer, CPU)
    assert feature_maps.shape == shape


def test_output_is_cpu_detached_normal_tensor_and_matches_manual_forward():
    model = _model()
    X = _batch()
    feature_maps = capture_feature_maps(model, X, "features.block1.act", CPU)

    assert feature_maps.device == CPU
    assert not feature_maps.requires_grad
    assert not feature_maps.is_inference()
    assert (feature_maps >= 0).all()  # salida de ReLU
    block1 = model.features.block1
    model.eval()  # la captura usa BN en modo eval (running stats)
    with torch.no_grad():
        expected = block1.act(block1.norm(block1.conv(X)))
    assert torch.allclose(feature_maps, expected, atol=1e-6)


@pytest.mark.parametrize("training", [True, False])
def test_restores_previous_mode_and_removes_hooks(training):
    model = _model()
    model.train(training)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    capture_feature_maps(model, _batch(), "features.block2.act", CPU)

    assert model.training is training
    assert all(m.training is training for m in model.modules())
    assert _hook_count(model) == 0
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_removes_hook_and_restores_mode_when_forward_fails():
    model = _model().train()
    bad_batch = torch.randn(2, 4, 128, 128)  # 4 canales: falla la primera convolución

    with pytest.raises(RuntimeError):
        capture_feature_maps(model, bad_batch, "features.block1.act", CPU)

    assert model.training
    assert _hook_count(model) == 0


def test_unknown_layer_raises_value_error():
    model = _model()
    with pytest.raises(ValueError, match="features.block1.act"):
        capture_feature_maps(model, _batch(), "features.block9.act", CPU)
    with pytest.raises(ValueError):
        capture_feature_maps(model, _batch(), "", CPU)
    assert _hook_count(model) == 0


def test_non_4d_batch_raises_value_error():
    with pytest.raises(ValueError, match="X_batch"):
        capture_feature_maps(_model(), torch.randn(3, 128, 128), "features.block1.act", CPU)


def test_same_layer_names_in_all_configurations():
    """Los nombres de 6.5 existen en A, B y C, así que la misma llamada sirve para los tres."""
    X = _batch(n=1)
    for use_batchnorm, dropout_p in ((False, 0.0), (True, 0.0), (True, 0.3)):
        model = GalaxyCNN(3, use_batchnorm, dropout_p)
        maps = capture_feature_maps(model, X, "features.block3.act", CPU)
        assert maps.shape == (1, 128, 32, 32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requiere CUDA")
def test_capture_on_cuda_returns_cpu_tensor():
    device = torch.device("cuda")
    model = _model().to(device)
    feature_maps = capture_feature_maps(model, _batch(), "features.block1.act", device)
    assert feature_maps.device == CPU
    assert feature_maps.shape == (2, 32, 128, 128)
