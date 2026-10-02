"""Tensor-state serialization and deterministic training restart checkpoints."""

from __future__ import annotations

import base64
import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


class TensorCheckpointError(ValueError):
    """A serialized tensor state is malformed or unsupported."""


def encode_tensor_state(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Encode a tensor mapping as sorted JSON-compatible raw byte records."""

    output: dict[str, dict[str, Any]] = {}
    for key, value in sorted(state.items()):
        if not hasattr(value, "detach"):
            raise TensorCheckpointError(f"state value is not a tensor: {key}")
        array = value.detach().cpu().contiguous().numpy()
        output[str(key)] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    return output


def decode_tensor_state(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a JSON tensor mapping without executing pickle or arbitrary Python."""

    if torch is None:
        raise TensorCheckpointError("tensor-state decoding requires torch")
    output: dict[str, Any] = {}
    try:
        for key, record in raw.items():
            if not isinstance(record, Mapping) or set(record) != {
                "dtype",
                "shape",
                "data_base64",
            }:
                raise ValueError(f"malformed tensor record: {key}")
            dtype = np.dtype(str(record["dtype"]))
            if dtype.kind not in {"b", "f", "i", "u"}:
                raise ValueError(f"unsupported tensor dtype: {dtype}")
            shape = tuple(int(value) for value in record["shape"])
            if any(value < 0 for value in shape):
                raise ValueError(f"negative tensor shape: {key}")
            payload = base64.b64decode(str(record["data_base64"]), validate=True)
            array = np.frombuffer(payload, dtype=dtype)
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise ValueError(f"tensor shape differs from payload: {key}")
            output[str(key)] = torch.from_numpy(array.reshape(shape).copy())
    except (KeyError, TypeError, ValueError) as error:
        raise TensorCheckpointError("tensor state is malformed") from error
    return output


class TrainingRestartError(RuntimeError):
    """A local restart checkpoint cannot reproduce the interrupted run."""


def _cpu_byte_rng_state(value: Any, *, label: str) -> Any:
    """Normalize a serialized torch RNG state without changing its bytes.

    ``torch.load(..., map_location=device)`` moves every tensor in a restart payload to the
    training device.  CUDA RNG setters nevertheless require CPU ``ByteTensor`` state, so resumed
    CUDA training must move these bookkeeping tensors back to CPU explicitly.
    """

    if torch is None:
        raise TrainingRestartError("random-state restore requires torch")
    try:
        normalized = value.detach().cpu().contiguous()
    except (AttributeError, RuntimeError, TypeError) as error:
        raise TrainingRestartError(f"{label} is not a torch tensor") from error
    if not torch.is_tensor(normalized) or normalized.dtype != torch.uint8 or normalized.ndim != 1:
        raise TrainingRestartError(f"{label} is not a one-dimensional CPU byte tensor")
    return normalized


def atomic_torch_save(path: Path, value: Any) -> None:
    """Publish one torch checkpoint atomically in its destination directory."""

    if torch is None:
        raise TrainingRestartError("torch checkpointing requires torch")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def capture_training_random_state(
    numpy_generator: np.random.Generator,
    torch_generator: Any,
    *,
    device: Any,
) -> dict[str, Any]:
    """Capture every RNG consumed by the supported deterministic training loop."""

    if torch is None:
        raise TrainingRestartError("random-state capture requires torch")
    return {
        "python_random_state": random.getstate(),
        "numpy_legacy_state": np.random.get_state(),
        "numpy_training_state": numpy_generator.bit_generator.state,
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        ),
        "torch_training_generator_state": torch_generator.get_state(),
    }


def restore_training_random_state(
    state: Mapping[str, Any],
    numpy_generator: np.random.Generator,
    torch_generator: Any,
    *,
    device: Any,
) -> None:
    """Restore a state captured by :func:`capture_training_random_state`."""

    if torch is None:
        raise TrainingRestartError("random-state restore requires torch")
    required = {
        "python_random_state",
        "numpy_legacy_state",
        "numpy_training_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state_all",
        "torch_training_generator_state",
    }
    missing = sorted(required - set(state))
    if missing:
        raise TrainingRestartError(f"restart checkpoint lacks RNG fields: {missing}")
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_legacy_state"])
    numpy_generator.bit_generator.state = state["numpy_training_state"]
    torch.set_rng_state(_cpu_byte_rng_state(state["torch_cpu_rng_state"], label="CPU RNG state"))
    if device.type == "cuda":
        cuda_states = state["torch_cuda_rng_state_all"]
        if not isinstance(cuda_states, list):
            raise TrainingRestartError("restart checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state_all(
            [
                _cpu_byte_rng_state(value, label=f"CUDA RNG state {index}")
                for index, value in enumerate(cuda_states)
            ]
        )
    torch_generator.set_state(
        _cpu_byte_rng_state(
            state["torch_training_generator_state"],
            label="training-generator RNG state",
        )
    )


__all__ = [
    "TensorCheckpointError",
    "decode_tensor_state",
    "encode_tensor_state",
    "TrainingRestartError",
    "atomic_torch_save",
    "capture_training_random_state",
    "restore_training_random_state",
]
