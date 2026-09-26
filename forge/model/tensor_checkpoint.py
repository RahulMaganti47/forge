"""Deterministic, non-executable tensor-state serialization for local checkpoints."""

from __future__ import annotations

import base64
from collections.abc import Mapping
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


__all__ = ["TensorCheckpointError", "decode_tensor_state", "encode_tensor_state"]
