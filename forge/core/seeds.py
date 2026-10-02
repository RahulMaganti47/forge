"""Deterministic random-stream identities shared by scientific workflows."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def keyed_seed(base_seed: int, **coordinates: Any) -> int:
    """Derive an order-independent 63-bit seed from named coordinates."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base seed must be a nonnegative integer")
    payload = json.dumps(
        {"base_seed": base_seed, "coordinates": coordinates},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


__all__ = ["keyed_seed"]
