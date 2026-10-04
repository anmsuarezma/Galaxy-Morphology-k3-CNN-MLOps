import pytest
import torch
from torch import nn

from galaxy3.models import GalaxyCNN, count_parameters, count_trainable_parameters

CONFIGS = {
    "A_baseline": {"use_batchnorm": False, "dropout_p": 0.0},
    "B_batchnorm": {"use_batchnorm": True, "dropout_p": 0.0},
    "C_batchnorm_dropout": {"use_batchnorm": True, "dropout_p": 0.3},
}
EXPECTED_PARAMS = {False: 520_003, True: 520_483}


def _model(name: str) -> GalaxyCNN:
    torch.manual_seed(0)
    return GalaxyCNN(num_classes=3, **CONFIGS[name])


def _has(model: nn.Module, module_type: type[nn.Module]) -> bool:
    return any(isinstance(m, module_type) for m in model.modules())


@pytest.mark.parametrize("name", CONFIGS)
def test_output_shape(name):
    logits = _model(name)(torch.randn(4, 3, 128, 128))
    assert logits.shape == (4, 3)


@pytest.mark.parametrize("num_classes", [2, 5])
def test_output_shape_follows_num_classes(num_classes):
    logits = GalaxyCNN(num_classes=num_classes)(torch.randn(2, 3, 128, 128))
    assert logits.shape == (2, num_classes)


def test_intermediate_shapes():
    model = _model("B_batchnorm").eval()
    x = torch.randn(4, 3, 128, 128)
    expected = [(4, 32, 64, 64), (4, 64, 32, 32), (4, 128, 16, 16), (4, 256, 8, 8)]
    with torch.inference_mode():
        for block, shape in zip(model.features, expected, strict=True):
            x = block(x)
            assert x.shape == shape
        assert model.pool(x).shape == (4, 256, 2, 2)


@pytest.mark.parametrize("name", CONFIGS)
def test_returns_logits_not_probabilities(name):
    model = _model(name).eval()
    assert not _has(model, nn.Softmax)
    assert not _has(model, nn.LogSoftmax)
    with torch.inference_mode():
        logits = model(torch.randn(8, 3, 128, 128))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8))
    assert (logits < 0).any()


@pytest.mark.parametrize("name", CONFIGS)
def test_parameter_count(name):
    model = _model(name)
    expected = EXPECTED_PARAMS[CONFIGS[name]["use_batchnorm"]]
    assert count_parameters(model) == expected
    assert count_trainable_parameters(model) == expected


def test_trainable_count_excludes_frozen_parameters():
    model = _model("A_baseline")
    for p in model.classifier.parameters():
        p.requires_grad_(False)
    assert count_parameters(model) == 520_003
    assert count_trainable_parameters(model) == 520_003 - (1024 * 128 + 128 + 128 * 3 + 3)


@pytest.mark.parametrize("name", CONFIGS)
def test_batchnorm_and_dropout_presence(name):
    model = _model(name)
    cfg = CONFIGS[name]
    assert _has(model, nn.BatchNorm2d) == cfg["use_batchnorm"]
    assert _has(model, nn.Dropout) == (cfg["dropout_p"] > 0)
    if cfg["dropout_p"] > 0:
        assert model.classifier.dropout.p == cfg["dropout_p"]


@pytest.mark.parametrize("name", CONFIGS)
def test_conv_bias_only_without_batchnorm(name):
    model = _model(name)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    assert len(convs) == 4
    for conv in convs:
        assert (conv.bias is None) == CONFIGS[name]["use_batchnorm"]


def test_module_names_identical_across_configs():
    names = {cfg: [n for n, _ in _model(cfg).named_modules()] for cfg in CONFIGS}
    reference = names["A_baseline"]
    assert all(n == reference for n in names.values())
    for i in range(1, 5):
        for part in ("conv", "norm", "act", "pool"):
            assert f"features.block{i}.{part}" in reference
    assert "pool" in reference
    assert "classifier.dropout" in reference


@pytest.mark.parametrize("dropout_p", [-0.1, 1.0, 1.5, float("nan")])
def test_invalid_dropout_raises(dropout_p):
    with pytest.raises(ValueError, match="dropout_p"):
        GalaxyCNN(dropout_p=dropout_p)


@pytest.mark.parametrize("num_classes", [0, 1])
def test_invalid_num_classes_raises(num_classes):
    with pytest.raises(ValueError, match="num_classes"):
        GalaxyCNN(num_classes=num_classes)
