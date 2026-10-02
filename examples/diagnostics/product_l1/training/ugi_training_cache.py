"""Prepare one hash-pinned tensor cache shared by matched Ugi training arms."""

from __future__ import annotations

from forge.corpus.training_cache import (
    UgiTrainingCacheError,
    load_ugi_training_cache,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


__all__ = ["UgiTrainingCacheError", "load_ugi_training_cache"]
