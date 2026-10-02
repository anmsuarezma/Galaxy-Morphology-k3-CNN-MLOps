from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest
import torch

from galaxy3.config import ExperimentConfig
from galaxy3.utils import PROJECT_ROOT

PRIMITIVES = (str, int, float, bool, type(None))


def test_defaults_match_spec():
    config = ExperimentConfig()
    assert config.split_seed == 2026
    assert config.train_seed == 2026
    assert config.num_classes == 3
    assert config.image_size == 128
    assert config.center_crop is None
    assert config.batch_size == 64
    assert config.num_workers == 8
    assert config.use_batchnorm is False
    assert config.dropout_p == 0.0
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.0
    assert config.device_preference == "auto"
    assert config.mlflow_experiment_name == "galaxy3-cnn-from-scratch"
    assert config.data_root == PROJECT_ROOT / "data" / "galaxy3"
    assert config.output_root == PROJECT_ROOT / "outputs"


def test_seeds_are_independent():
    config = ExperimentConfig(train_seed=11)
    assert config.split_seed == 2026
    assert config.train_seed == 11


def test_is_immutable_and_replace_creates_variant():
    config = ExperimentConfig()
    with pytest.raises(FrozenInstanceError):
        config.dropout_p = 0.3  # type: ignore[misc]
    variant = replace(config, use_batchnorm=True, dropout_p=0.3)
    assert variant.dropout_p == 0.3
    assert config.dropout_p == 0.0


def test_str_paths_are_converted_to_path():
    config = ExperimentConfig(data_root="some/data", output_root="some/out")
    assert isinstance(config.data_root, Path)
    assert isinstance(config.output_root, Path)


def test_to_dict_has_only_primitive_types():
    data = ExperimentConfig().to_dict()
    assert all(isinstance(value, PRIMITIVES) for value in data.values())
    assert data["data_root"] == str(PROJECT_ROOT / "data" / "galaxy3")


def test_round_trip_through_dict():
    config = ExperimentConfig(run_name="batchnorm_dropout", use_batchnorm=True, dropout_p=0.3)
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_to_dict_survives_weights_only_load(tmp_path):
    config = ExperimentConfig(center_crop=200)
    path = tmp_path / "ckpt.pt"
    torch.save({"config": config.to_dict()}, path)
    loaded = torch.load(path, weights_only=True)
    assert ExperimentConfig.from_dict(loaded["config"]) == config


def test_from_dict_rejects_unknown_keys():
    data = ExperimentConfig().to_dict() | {"scheduler": "plateau"}
    with pytest.raises(ValueError, match="scheduler"):
        ExperimentConfig.from_dict(data)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_name": "  "},
        {"split_seed": -1},
        {"train_seed": -1},
        {"num_classes": 1},
        {"image_size": 0},
        {"center_crop": 0},
        {"batch_size": 0},
        {"num_workers": -1},
        {"dropout_p": 1.0},
        {"dropout_p": -0.1},
        {"learning_rate": 0.0},
        {"weight_decay": -1e-4},
        {"epochs": 0},
        {"device_preference": "tpu"},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(**kwargs)


def test_num_workers_zero_is_valid():
    assert ExperimentConfig(num_workers=0).num_workers == 0
