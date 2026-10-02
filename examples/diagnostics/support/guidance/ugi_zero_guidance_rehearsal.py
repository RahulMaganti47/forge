"""Additive zero-guidance rehearsal for matched Ugi generation and routing.

This module composes the hash-pinned matched runner with a dependency-injected
restartable generator/closure adapter, typed terminal-route assessment and two
isolated planner-cache overlays.  It exercises the real seams at guidance
strength zero while defining no synthesis scalar and performing no selection.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forge.synthesis.engine.planner import RoutePlanner
from forge.synthesis.engine.planner_cache import PlannerCacheContext
from forge.synthesis.engine.planner_cache_snapshot import (
    OverlayFilePlannerCache,
)
from forge.synthesis.matched import (
    LockedMatchedTerminal,
    MatchedAssessmentContext,
    MatchedGenerationRequest,
)

ZERO_GUIDANCE_REHEARSAL_SCHEMA_VERSION = "forge.ugi_zero_guidance_rehearsal.v1"
ZERO_GUIDANCE_REHEARSAL_PREFLIGHT_SCHEMA_VERSION = "forge.ugi_zero_guidance_rehearsal_preflight.v1"
HASH_PINNED_MATCHED_RUNNER_SHA256 = (
    "3554ca96038e7916c15de17e8544c4f43cd190537f4342e28b24d34c64426fe3"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiZeroGuidanceRehearsalError(RuntimeError):
    """Raised when a zero-guidance rehearsal contract is violated."""


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise UgiZeroGuidanceRehearsalError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiZeroGuidanceRehearsalError(f"{label} must be a nonempty string")
    return value


@dataclass(frozen=True)
class RestartableGeneratorClosureIdentity:
    """Hash ownership for the selected restartable generator/closure seam."""

    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    production_generator_manifest_sha256: str
    restartable_equivalence_receipt_sha256: str
    generator_implementation_sha256: str
    terminal_decoder_id: str

    def __post_init__(self) -> None:
        for name in (
            "generator_checkpoint_sha256",
            "closure_checkpoint_sha256",
            "production_generator_manifest_sha256",
            "restartable_equivalence_receipt_sha256",
            "generator_implementation_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        _require_nonempty(self.terminal_decoder_id, label="terminal_decoder_id")

    def to_dict(self) -> dict[str, str]:
        return {
            "generator_checkpoint_sha256": self.generator_checkpoint_sha256,
            "closure_checkpoint_sha256": self.closure_checkpoint_sha256,
            "production_generator_manifest_sha256": (self.production_generator_manifest_sha256),
            "restartable_equivalence_receipt_sha256": (self.restartable_equivalence_receipt_sha256),
            "generator_implementation_sha256": self.generator_implementation_sha256,
            "terminal_decoder_id": self.terminal_decoder_id,
        }


@dataclass(frozen=True)
class RestartableGeneratorClosureAdapter:
    """Dependency-injected productive terminal generator with frozen identity."""

    identity: RestartableGeneratorClosureIdentity
    generate_locked_terminal: Callable[[MatchedGenerationRequest], LockedMatchedTerminal]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, RestartableGeneratorClosureIdentity):
            raise UgiZeroGuidanceRehearsalError("generator/closure identity is malformed")
        if not callable(self.generate_locked_terminal):
            raise UgiZeroGuidanceRehearsalError("generate_locked_terminal must be callable")


RoutePlannerFactory = Callable[
    [
        OverlayFilePlannerCache,
        PlannerCacheContext,
        MatchedAssessmentContext,
        LockedMatchedTerminal,
    ],
    RoutePlanner,
]
