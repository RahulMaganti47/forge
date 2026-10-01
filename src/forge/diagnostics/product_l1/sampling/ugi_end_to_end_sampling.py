"""Compose generated Ugi morphology, sparse closures, and chemistry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.model.ugi_closure_placement import UgiSparseClosureScorer

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class UgiEndToEndSamplingError(RuntimeError):
    """Raised when checkpoint composition violates the generated-data contract."""


def _load_checkpoint(path: Path, schema: str | tuple[str, ...]) -> dict[str, Any]:
    if torch is None or not path.is_file():
        raise UgiEndToEndSamplingError(f"missing checkpoint: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    supported = (schema,) if isinstance(schema, str) else schema
    if value.get("schema_version") not in supported:
        raise UgiEndToEndSamplingError(f"unexpected checkpoint schema: {path}")
    return value


def _closure_model(checkpoint: dict[str, Any]) -> Any:
    model = UgiSparseClosureScorer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
