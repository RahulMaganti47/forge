"""Keyed random streams for call-order-independent reproducibility."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SeedPlan:
    """Derive deterministic independent seeds from one recorded root seed.

    A keyed stream is stable when another arm performs an extra random draw.  That property is
    essential for matched experiments: relying on one sequential global RNG couples an arm's later
    randomness to every earlier branch it happened to take.
    """

    root_seed: int
    namespace: str

    def __post_init__(self) -> None:
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise TypeError("root_seed must be an integer")
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        if not self.namespace:
            raise ValueError("seed namespace must not be empty")

    def derive(self, label: str, *parts: object) -> int:
        if not label:
            raise ValueError("seed label must not be empty")
        digest = hashlib.sha256()
        for item in (str(self.root_seed), self.namespace, label, *(str(part) for part in parts)):
            payload = item.encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return int.from_bytes(digest.digest()[:8], "big") & ((1 << 63) - 1)

    def python(self, label: str, *parts: object) -> random.Random:
        return random.Random(self.derive(label, *parts))

    def numpy(self, label: str, *parts: object) -> Any:
        import numpy as np

        return np.random.default_rng(self.derive(label, *parts))

    def torch(self, label: str, *parts: object, device: str = "cpu") -> Any:
        import torch

        generator = torch.Generator(device=device)
        generator.manual_seed(self.derive(label, *parts))
        return generator


__all__ = ["SeedPlan"]
