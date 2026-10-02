import random

import numpy as np
import pytest
import torch

from galaxy3 import utils
from galaxy3.utils import (
    PROJECT_ROOT,
    get_device,
    make_generator,
    prepare_output_dirs,
    seed_worker,
    set_seed,
)


def _fake_availability(monkeypatch, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(utils, "_mps_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_device_order(monkeypatch, cuda, mps, expected):
    _fake_availability(monkeypatch, cuda=cuda, mps=mps)
    assert get_device("auto").type == expected


def test_cpu_always_available():
    assert get_device("cpu") == torch.device("cpu")


def test_explicit_unavailable_device_raises(monkeypatch):
    _fake_availability(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError):
        get_device("cuda")
    with pytest.raises(RuntimeError):
        get_device("mps")


def test_invalid_device_preference_raises():
    with pytest.raises(ValueError):
        get_device("gpu")


def _draw() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.rand()), torch.rand(3)


def test_set_seed_is_reproducible():
    set_seed(123)
    first = _draw()
    set_seed(123)
    second = _draw()
    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])


def test_set_seed_rejects_negative():
    with pytest.raises(ValueError):
        set_seed(-1)


def test_generator_is_independent_of_global_rng():
    gen_a = make_generator(7)
    perm_a = torch.randperm(10, generator=gen_a)
    torch.manual_seed(999)
    torch.rand(5)  # consume RNG global, como harían las máscaras de Dropout
    gen_b = make_generator(7)
    perm_b = torch.randperm(10, generator=gen_b)
    assert torch.equal(perm_a, perm_b)


def test_seed_worker_derives_from_torch_seed():
    torch.manual_seed(42)
    seed_worker(0)
    first = (random.random(), float(np.random.rand()))
    torch.manual_seed(42)
    seed_worker(0)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_prepare_output_dirs_is_idempotent(tmp_path):
    dirs = prepare_output_dirs(tmp_path / "outputs")
    again = prepare_output_dirs(tmp_path / "outputs")
    assert dirs == again
    for path in (dirs.checkpoints, dirs.figures, dirs.metrics):
        assert path.is_dir()


def test_project_root_points_to_galaxy3():
    assert (PROJECT_ROOT / "src" / "galaxy3" / "utils.py").is_file()
