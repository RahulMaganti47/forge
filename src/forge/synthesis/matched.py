"""Diagnostic matched-budget orchestration for future Ugi guidance experiments.

This module coordinates synthetic guided and post-hoc arms without defining a
synthesis scalar or running chemistry. Productive generation uses shared keyed
substreams, route assessment uses arm-specific keyed substreams, and post-hoc
assessment is deferred until every admitted terminal and trace has been sealed.

The contract is intentionally conservative: schedule entries reserve a common
per-arm route budget before either arm runs. Budget exhaustion therefore censors
one identical suffix in both arms without dummy calls or extra compute.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Any


class UgiMatchedBudgetError(RuntimeError):
    """Raised when diagnostic matched-arm execution violates its contract."""


class MatchedArm(str, Enum):
    """The two causally distinct arms in the matched comparison."""

    GUIDED = "guided"
    POST_HOC = "post_hoc"


def _nonnegative_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiMatchedBudgetError(f"{name} must be a nonnegative integer")
    return value


def _nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise UgiMatchedBudgetError(f"{name} must be a nonempty string")
    return value


def _sha256_string(name: str, value: Any) -> str:
    value = _nonempty_string(name, value)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise UgiMatchedBudgetError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class RouteComputeUsage:
    """Logical and physical route-compute calls for one assessment."""

    logical_planner_calls: int = 0
    physical_planner_calls: int = 0
    logical_verifier_calls: int = 0
    physical_verifier_calls: int = 0

    def __post_init__(self) -> None:
        for name in (
            "logical_planner_calls",
            "physical_planner_calls",
            "logical_verifier_calls",
            "physical_verifier_calls",
        ):
            _nonnegative_integer(name, getattr(self, name))
        if self.physical_planner_calls > self.logical_planner_calls:
            raise UgiMatchedBudgetError(
                "physical_planner_calls cannot exceed logical_planner_calls"
            )
        if self.physical_verifier_calls > self.logical_verifier_calls:
            raise UgiMatchedBudgetError(
                "physical_verifier_calls cannot exceed logical_verifier_calls"
            )

    @property
    def planner_cache_hits(self) -> int:
        """Logical planner calls served without physical execution."""

        return self.logical_planner_calls - self.physical_planner_calls

    @property
    def verifier_cache_hits(self) -> int:
        """Logical verifier calls served without physical execution."""

        return self.logical_verifier_calls - self.physical_verifier_calls

    def plus(self, other: RouteComputeUsage) -> RouteComputeUsage:
        """Return field-wise usage addition."""

        return RouteComputeUsage(
            logical_planner_calls=self.logical_planner_calls + other.logical_planner_calls,
            physical_planner_calls=self.physical_planner_calls + other.physical_planner_calls,
            logical_verifier_calls=self.logical_verifier_calls + other.logical_verifier_calls,
            physical_verifier_calls=self.physical_verifier_calls + other.physical_verifier_calls,
        )

    def fits_within(self, ceiling: RouteComputeUsage) -> bool:
        """Return whether every usage field is no larger than its ceiling."""

        return all(
            getattr(self, name) <= getattr(ceiling, name)
            for name in (
                "logical_planner_calls",
                "physical_planner_calls",
                "logical_verifier_calls",
                "physical_verifier_calls",
            )
        )

    def remaining_after(self, used: RouteComputeUsage) -> RouteComputeUsage:
        """Subtract used calls from a ceiling, failing on overuse."""

        if not used.fits_within(self):
            raise UgiMatchedBudgetError("route usage exceeds its declared ceiling")
        return RouteComputeUsage(
            logical_planner_calls=self.logical_planner_calls - used.logical_planner_calls,
            physical_planner_calls=min(
                self.logical_planner_calls - used.logical_planner_calls,
                self.physical_planner_calls - used.physical_planner_calls,
            ),
            logical_verifier_calls=self.logical_verifier_calls - used.logical_verifier_calls,
            physical_verifier_calls=min(
                self.logical_verifier_calls - used.logical_verifier_calls,
                self.physical_verifier_calls - used.physical_verifier_calls,
            ),
        )


@dataclass(frozen=True)
class MatchedScheduleEntry:
    """One common morphology/checkpoint unit with a frozen budget reservation."""

    unit_id: str
    morphology_program: bytes
    program_index: int
    particle_index: int
    checkpoint_index: int
    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    rollout_index: int
    productive_generation_calls: int
    route_reservation: RouteComputeUsage

    def __post_init__(self) -> None:
        _nonempty_string("unit_id", self.unit_id)
        if not isinstance(self.morphology_program, bytes) or not self.morphology_program:
            raise UgiMatchedBudgetError("morphology_program must be nonempty bytes")
        for name in (
            "program_index",
            "particle_index",
            "checkpoint_index",
            "rollout_index",
            "productive_generation_calls",
        ):
            _nonnegative_integer(name, getattr(self, name))
        _sha256_string("generator_checkpoint_sha256", self.generator_checkpoint_sha256)
        _sha256_string("closure_checkpoint_sha256", self.closure_checkpoint_sha256)
        if self.productive_generation_calls == 0:
            raise UgiMatchedBudgetError("productive_generation_calls must be positive")

    @property
    def morphology_program_sha256(self) -> str:
        """Return the exact shared morphology-program digest."""

        return hashlib.sha256(self.morphology_program).hexdigest()


@dataclass(frozen=True)
class MatchedGenerationRequest:
    """Shared productive-generation request passed to either arm."""

    arm: MatchedArm
    entry: MatchedScheduleEntry
    productive_seed: int


@dataclass(frozen=True)
class LockedMatchedTerminal:
    """A sealed terminal candidate and trace returned by a generation callback."""

    unit_id: str
    morphology_program_sha256: str
    checkpoint_index: int
    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    terminal_id: str
    terminal_locked: bool
    terminal_valid: bool
    exact_l1: bool
    terminal_bytes: bytes
    generation_trace_bytes: bytes
    payload: Any = None

    def __post_init__(self) -> None:
        _nonempty_string("unit_id", self.unit_id)
        _sha256_string("morphology_program_sha256", self.morphology_program_sha256)
        _sha256_string("generator_checkpoint_sha256", self.generator_checkpoint_sha256)
        _sha256_string("closure_checkpoint_sha256", self.closure_checkpoint_sha256)
        _nonempty_string("terminal_id", self.terminal_id)
        _nonnegative_integer("checkpoint_index", self.checkpoint_index)
        if not all(
            isinstance(value, bool)
            for value in (self.terminal_locked, self.terminal_valid, self.exact_l1)
        ):
            raise UgiMatchedBudgetError("terminal admission fields must be boolean")
        if self.exact_l1 and not self.terminal_valid:
            raise UgiMatchedBudgetError("an invalid terminal cannot be exact L1")
        for name in ("terminal_bytes", "generation_trace_bytes"):
            value = getattr(self, name)
            if not isinstance(value, bytes) or not value:
                raise UgiMatchedBudgetError(f"{name} must be nonempty bytes")

    @property
    def terminal_sha256(self) -> str:
        return hashlib.sha256(self.terminal_bytes).hexdigest()

    @property
    def generation_trace_sha256(self) -> str:
        return hashlib.sha256(self.generation_trace_bytes).hexdigest()


DiagnosticField = tuple[str, str]


@dataclass(frozen=True)
class MatchedAssessmentContext:
    """Arm-specific route substream and frozen call ceilings."""

    arm: MatchedArm
    route_seed: int
    remaining_budget: RouteComputeUsage
    unit_reservation: RouteComputeUsage
    cache_snapshot_sha256: str
    cache_clone_id: str
    post_hoc_lock_manifest_sha256: str | None
