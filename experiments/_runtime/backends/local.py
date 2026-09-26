"""In-process local execution for CPU, MPS, or CUDA development runs."""

from __future__ import annotations

import os
import random

from experiments._runtime.errors import BackendError
from experiments._runtime.stage import RunContext, StageCallable, StageResult


class LocalBackend:
    """Execute the same stage callable the remote backend will invoke."""

    name = "local"

    def execute(self, function: StageCallable, context: RunContext) -> StageResult:
        seed = context.derive_seed("stage")
        random.seed(seed)
        previous_deterministic: bool | None = None
        try:
            import numpy as np

            np.random.seed(seed % (2**32))
        except ImportError:
            pass
        try:
            import torch

            if context.resources.device == "cuda" and not torch.cuda.is_available():
                raise BackendError("CUDA was requested but torch.cuda.is_available() is false")
            if context.resources.device == "mps":
                backend = getattr(torch.backends, "mps", None)
                if backend is None or not backend.is_available():
                    raise BackendError("MPS was requested but the torch MPS backend is unavailable")
                if context.resources.precision == "float64":
                    raise BackendError("MPS does not support the requested float64 precision")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            previous_deterministic = bool(torch.are_deterministic_algorithms_enabled())
            strict = context.stage.determinism.mode == "strict"
            if strict:
                os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(strict)
        except ImportError:
            if context.resources.device != "cpu":
                raise BackendError(
                    f"{context.resources.device.upper()} was requested but torch is not installed"
                ) from None

        try:
            result = function(context)
        finally:
            if previous_deterministic is not None:
                import torch

                torch.use_deterministic_algorithms(previous_deterministic)
        if not isinstance(result, StageResult):
            raise BackendError(
                f"stage {context.stage.stage_id!r} returned {type(result).__name__}; "
                "expected StageResult"
            )
        return result


__all__ = ["LocalBackend"]
