"""The boundary between scientific stages and execution backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from experiments._runtime.errors import StageError
from experiments._runtime.seed import SeedPlan
from experiments._runtime.spec import ResourceSpec, StageSpec
from forge.core.io import read_json_object


@dataclass(frozen=True)
class ProducedArtifact:
    """One declared stage output written below ``RunContext.output_dir``."""

    label: str
    relative_path: str
    schema_version: str
    rows: int | None = None

    def __post_init__(self) -> None:
        if self.rows is not None and self.rows < 0:
            raise ValueError("artifact rows must be non-negative")


@dataclass(frozen=True)
class StageResult:
    """The small receipt returned by a stage after it writes its declared files."""

    artifacts: tuple[ProducedArtifact, ...]
    metrics: Mapping[str, Any] = field(default_factory=dict)
    summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        labels = [artifact.label for artifact in self.artifacts]
        if len(labels) != len(set(labels)):
            raise ValueError("stage result contains duplicate artifact labels")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "summary", MappingProxyType(dict(self.summary)))


@dataclass(frozen=True)
class DependencyArtifact:
    """A verified output from an already committed dependency stage."""

    stage_id: str
    label: str
    path: Path
    sha256: str
    schema_version: str
    rows: int | None = None


@dataclass
class RunContext:
    """Everything a stage may read or write during one isolated execution."""

    repo: Path
    experiment_id: str
    run_id: str
    profile: str
    replicate: int
    backend: str
    stage: StageSpec
    resources: ResourceSpec
    work_dir: Path
    output_dir: Path
    config_path: Path
    inputs: Mapping[str, Path]
    dependencies: Mapping[str, Mapping[str, DependencyArtifact]]
    seed_plan: SeedPlan
    resume: bool = False
    _derived_seeds: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.inputs = MappingProxyType(dict(self.inputs))
        self.dependencies = MappingProxyType(
            {
                stage_id: MappingProxyType(dict(artifacts))
                for stage_id, artifacts in self.dependencies.items()
            }
        )

    def config(self) -> dict[str, Any]:
        return read_json_object(
            self.config_path,
            error=StageError,
            label=f"stage {self.stage.stage_id} config",
        )

    def input(self, label: str) -> Path:
        try:
            return self.inputs[label]
        except KeyError as error:
            known = ", ".join(sorted(self.inputs)) or "none"
            raise StageError(
                f"stage {self.stage.stage_id!r} requested undeclared input {label!r}; "
                f"declared inputs: {known}"
            ) from error

    def dependency(self, stage_id: str, label: str) -> DependencyArtifact:
        if stage_id not in self.stage.needs:
            raise StageError(f"stage {self.stage.stage_id!r} requested non-dependency {stage_id!r}")
        try:
            return self.dependencies[stage_id][label]
        except KeyError as error:
            available = sorted(self.dependencies.get(stage_id, {}))
            raise StageError(
                f"dependency {stage_id!r} has no artifact {label!r}; available: {available}"
            ) from error

    def output_path(self, relative_path: str) -> Path:
        candidate = self.output_dir / relative_path
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.output_dir.resolve())
        except ValueError as error:
            raise StageError(
                f"stage output escapes its private directory: {relative_path!r}"
            ) from error
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def derive_seed(self, label: str, *parts: object) -> int:
        key = "/".join((label, *(str(part) for part in parts)))
        value = self.seed_plan.derive(label, *parts)
        previous = self._derived_seeds.setdefault(key, value)
        if (
            previous != value
        ):  # defensive: SeedPlan is deterministic, but fail if that ever changes.
            raise StageError(f"seed key {key!r} derived two different values")
        return value

    @property
    def derived_seeds(self) -> Mapping[str, int]:
        return MappingProxyType(dict(sorted(self._derived_seeds.items())))


def require_config_inputs(
    context: RunContext,
    config: Mapping[str, Any],
    *,
    labels: set[str] | None = None,
) -> None:
    """Require a stage config to repeat exactly the pins declared by its experiment spec."""

    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError(f"stage {context.stage.stage_id} config has no input mapping")
    expected = set(context.inputs) if labels is None else labels
    if set(configured) != expected:
        raise StageError(
            f"experiment inputs and stage config differ for {context.stage.stage_id}: "
            f"experiment={sorted(expected)}, config={sorted(configured)}"
        )
    for label in sorted(expected):
        pin = context.stage.inputs[label]
        record = configured[label]
        if not isinstance(record, dict) or record != pin.to_mapping():
            raise StageError(f"experiment pin differs from stage config for {label!r}")


StageCallable = Callable[[RunContext], StageResult]


__all__ = [
    "DependencyArtifact",
    "ProducedArtifact",
    "RunContext",
    "StageCallable",
    "StageResult",
    "require_config_inputs",
]
