"""Development runner for matched Ugi synthesis-guidance experiments.

This module wires restartable product trajectories to terminal-only route
rollouts and within-morphology-program SMC ancestry.  It deliberately owns no
chemistry, model, cache, or cloud implementation.  Those dependencies are
injected through :class:`RestartableGuidanceLane`, which keeps the orchestration
device agnostic and suitable for a later local or remote wrapper.

Nonzero production execution is fail-closed behind a separate authenticated
execution-review receipt.  Small fake lanes may exercise the orchestration in
tests without claiming that any production gate has passed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from forge.synthesis.matched import (
    LockedMatchedTerminal,
    RouteComputeUsage,
)

RUNNER_CONFIG_SCHEMA_VERSION = "phase1_ugi_nonzero_guidance_runner_config.v1"
EXECUTION_REVIEW_SCHEMA_VERSION = "phase1_ugi_nonzero_guidance_execution_review.v1"
GROUPED_SMC_SCHEDULE_SCHEMA_VERSION = "phase1_ugi_grouped_smc_schedule_qualification.v1"
RUNNER_PLAN_SCHEMA_VERSION = "forge.ugi_nonzero_guidance_runner_plan.v1"
RUN_RESULT_SCHEMA_VERSION = "forge.ugi_nonzero_guidance_development_run.v1"


class UgiNonzeroGuidanceRunnerError(RuntimeError):
    """Raised when guidance orchestration violates the frozen contract."""


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_integer(value: Any, *, label: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        qualifier = "positive" if positive else "nonnegative"
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be a {qualifier} integer")
    return value


def _require_finite_nonnegative(value: Any, *, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        raise UgiNonzeroGuidanceRunnerError(f"{label} must be finite and nonnegative")
    return float(value)


@dataclass(frozen=True)
class GuidanceStateReceipt:
    """One restartable-lane state operation and its realized device compute."""

    state: Any
    product_transition_calls: int
    gpu_device_seconds: float = 0.0
    consumed_particle_seed_manifest_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.product_transition_calls,
            label="state-operation product_transition_calls",
        )
        _require_finite_nonnegative(
            self.gpu_device_seconds,
            label="state-operation gpu_device_seconds",
        )
        if self.consumed_particle_seed_manifest_sha256 is not None:
            _require_sha256(
                self.consumed_particle_seed_manifest_sha256,
                label="consumed_particle_seed_manifest_sha256",
            )


@dataclass(frozen=True)
class GuidanceTerminalCompletionReceipt:
    """One terminal-completion attempt and its realized transition/device use."""

    terminal: LockedMatchedTerminal | None
    product_transition_calls: int
    terminal_completions: int = 1
    gpu_device_seconds: float = 0.0
    error_detail: str | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_integer(
            self.product_transition_calls,
            label="terminal product_transition_calls",
        )
        if self.terminal_completions != 1:
            raise UgiNonzeroGuidanceRunnerError(
                "each terminal receipt must account for exactly one completion attempt"
            )
        _require_finite_nonnegative(
            self.gpu_device_seconds,
            label="terminal gpu_device_seconds",
        )
        if self.terminal is None:
            if not isinstance(self.error_detail, str) or not self.error_detail:
                raise UgiNonzeroGuidanceRunnerError(
                    "failed terminal completion requires a nonempty error detail"
                )
        elif self.error_detail is not None:
            raise UgiNonzeroGuidanceRunnerError(
                "successful terminal completion cannot carry an error detail"
            )


@dataclass(frozen=True)
class GuidanceRouteEvaluation:
    """Binary/neutral route utility plus realized logical and physical compute."""

    route_completion_utility: float | None
    value_policy_id: str
    usage: RouteComputeUsage
    assessment_receipt_sha256: str
    route_dossier_sha256: str | None
    wall_seconds: float = 0.0
    gpu_device_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.route_completion_utility not in {None, 0.0, 1.0}:
            raise UgiNonzeroGuidanceRunnerError(
                "route-completion utility must be binary or explicitly censored"
            )
        if not isinstance(self.value_policy_id, str) or not self.value_policy_id:
            raise UgiNonzeroGuidanceRunnerError("value_policy_id must be nonempty")
        if not isinstance(self.usage, RouteComputeUsage):
            raise UgiNonzeroGuidanceRunnerError("route usage must be typed")
        _require_sha256(
            self.assessment_receipt_sha256,
            label="assessment_receipt_sha256",
        )
        if self.route_dossier_sha256 is not None:
            _require_sha256(self.route_dossier_sha256, label="route_dossier_sha256")
        if self.route_completion_utility == 1.0 and self.route_dossier_sha256 is None:
            raise UgiNonzeroGuidanceRunnerError(
                "a positive exact-dossier utility requires a dossier hash"
            )
        _require_finite_nonnegative(self.wall_seconds, label="wall_seconds")
        _require_finite_nonnegative(self.gpu_device_seconds, label="gpu_device_seconds")

    @property
    def censored(self) -> bool:
        return self.route_completion_utility is None


@dataclass(frozen=True)
class GuidanceAssessmentContext:
    """One route call's matched budget, RNG, cache and lock provenance."""

    treatment_arm: str
    assessment_phase: str
    checkpoint: int
    route_seed: int
    reservation: RouteComputeUsage
    base_snapshot_sha256: str
    planner_context_sha256: str
    cache_preflight_sha256: str
    cache_clone_id: str
    post_hoc_productive_lock_sha256: str | None

    def __post_init__(self) -> None:
        if self.treatment_arm not in {"guided", "post_hoc"}:
            raise UgiNonzeroGuidanceRunnerError("unsupported route-assessment arm")
        if self.assessment_phase not in {"checkpoint_shadow", "productive_final"}:
            raise UgiNonzeroGuidanceRunnerError("unsupported route-assessment phase")
        _require_nonnegative_integer(self.checkpoint, label="checkpoint")
        _require_nonnegative_integer(self.route_seed, label="route_seed")
        if not isinstance(self.reservation, RouteComputeUsage):
            raise UgiNonzeroGuidanceRunnerError("route reservation must be typed")
        for name in (
            "base_snapshot_sha256",
            "planner_context_sha256",
            "cache_preflight_sha256",
        ):
            _require_sha256(getattr(self, name), label=name)
        if not isinstance(self.cache_clone_id, str) or not self.cache_clone_id:
            raise UgiNonzeroGuidanceRunnerError("cache_clone_id must be nonempty")
        if self.treatment_arm == "post_hoc":
            _require_sha256(
                self.post_hoc_productive_lock_sha256,
                label="post_hoc_productive_lock_sha256",
            )
        elif self.post_hoc_productive_lock_sha256 is not None:
            raise UgiNonzeroGuidanceRunnerError(
                "guided assessments cannot carry the post-hoc productive lock"
            )


RouteEvaluator = Callable[
    [LockedMatchedTerminal, GuidanceAssessmentContext],
    GuidanceRouteEvaluation,
]
CanonicalIdentity = Callable[[bytes], str]


__all__ = ["GuidanceRouteEvaluation", "UgiNonzeroGuidanceRunnerError"]
