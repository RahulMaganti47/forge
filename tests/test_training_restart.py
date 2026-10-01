from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest

from forge.model import training_restart

torch = pytest.importorskip("torch")


class _MappedCudaState:
    """Stand in for a byte tensor moved to CUDA by ``torch.load(map_location=...)``."""

    def __init__(self, value: object) -> None:
        self.value = value
        self.cpu_called = False

    def detach(self) -> _MappedCudaState:
        return self

    def cpu(self) -> object:
        self.cpu_called = True
        return self.value


def _state(cuda_states: object) -> dict[str, object]:
    return {
        "python_random_state": random.getstate(),
        "numpy_legacy_state": np.random.get_state(),
        "numpy_training_state": np.random.default_rng(11).bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": cuda_states,
        "torch_training_generator_state": torch.Generator().get_state(),
    }


def test_cuda_rng_restore_normalizes_map_location_tensors_to_cpu(monkeypatch: object) -> None:
    expected = torch.arange(16, dtype=torch.uint8)
    mapped = _MappedCudaState(expected)
    restored: list[object] = []
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: restored.extend(values))

    training_restart.restore_training_random_state(
        _state([mapped]),
        np.random.default_rng(11),
        torch.Generator(),
        device=SimpleNamespace(type="cuda"),
    )

    assert mapped.cpu_called is True
    assert len(restored) == 1
    assert torch.equal(restored[0], expected)
    assert restored[0].device.type == "cpu"


def test_rng_restore_rejects_non_byte_cuda_state(monkeypatch: object) -> None:
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda values: None)

    with pytest.raises(
        training_restart.TrainingRestartError,
        match="CUDA RNG state 0 is not a one-dimensional CPU byte tensor",
    ):
        training_restart.restore_training_random_state(
            _state([torch.arange(4, dtype=torch.int64)]),
            np.random.default_rng(11),
            torch.Generator(),
            device=SimpleNamespace(type="cuda"),
        )
