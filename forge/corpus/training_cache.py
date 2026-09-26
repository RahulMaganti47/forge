"""Read-only access to the authenticated Phase 1 tensor-cache schema."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

torch: Any
try:
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None
else:
    torch = _torch


class UgiTrainingCacheError(RuntimeError):
    """Raised when an authenticated Ugi training cache cannot be consumed."""


_FROZEN_MODULE_MOVES = {
    "forge.product.defog_feasibility": "forge.model.defog_feasibility",
    "forge.product.ugi_chemistry_corpus": "forge.corpus.ugi_chemistry_corpus",
    "forge.product.ugi_chemistry_interface": "forge.model.ugi_chemistry_interface",
    "forge.product.ugi_joint_sparse_flow": "forge.model.ugi_joint_sparse_flow",
    "forge.product.ugi_morphology_program": "forge.model.ugi_morphology_program",
}


class _FrozenCacheUnpickler(pickle.Unpickler):
    """Resolve the exact module paths embedded by the frozen v1 cache."""

    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(_FROZEN_MODULE_MOVES.get(module, module), name)


class _FrozenCachePickle:
    """Minimal module-shaped pickle interface required by ``torch.load``."""

    __name__ = "forge_frozen_cache_pickle"
    Unpickler = _FrozenCacheUnpickler


def load_ugi_training_cache_payload(path: Path) -> dict[str, Any]:
    """Load the complete trusted payload after the caller verifies its artifact hash."""

    if torch is None:
        raise UgiTrainingCacheError("training-cache loading requires torch")
    payload = torch.load(
        path,
        map_location="cpu",
        pickle_module=_FrozenCachePickle,
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise UgiTrainingCacheError("prepared training cache must contain a mapping")
    if payload.get("schema_version") != "phase1_ugi_training_cache.v1":
        raise UgiTrainingCacheError("unsupported prepared training cache")
    return payload


def load_ugi_training_cache(path: Path) -> tuple[Any, dict[str, tuple[Any, ...]]]:
    """Load the corpus and joint records from one trusted authenticated cache."""

    payload = load_ugi_training_cache_payload(path)
    return payload["corpus"], payload["joint_records_by_fold"]


__all__ = [
    "UgiTrainingCacheError",
    "load_ugi_training_cache",
    "load_ugi_training_cache_payload",
]
