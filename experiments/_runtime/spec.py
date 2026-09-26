"""Strict, typed experiment specifications.

Experiment JSON is deliberately small.  It composes hash-pinned stage configurations and inputs;
it does not duplicate scientific parameters from those configurations.  Unknown fields are errors,
because a misspelled budget or determinism key must not be silently ignored.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from experiments._runtime.errors import SpecError
from forge.core.hashing import is_sha256
from forge.core.io import read_json_object
from forge.core.records import PinnedInput

# The schema identifier is a persisted data contract and remains stable across the package move.
EXPERIMENT_SCHEMA_VERSION = "forge.experiment.v1"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
_DEVICES = frozenset({"cpu", "mps", "cuda"})
_PRECISIONS = frozenset({"float32", "float64", "bfloat16", "float16"})
_DETERMINISM_MODES = frozenset({"strict", "statistical"})


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{label} must be a JSON object")
    return value


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SpecError(f"{label} must be a JSON array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    missing = expected - set(value)
    unknown = set(value) - expected
    if missing or unknown:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise SpecError(f"{label} has invalid fields: {'; '.join(details)}")


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise SpecError(f"{label} must match {_IDENTIFIER.pattern!r}; found {value!r}")
    return value


def _nonnegative_int(value: object, label: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SpecError(f"{label} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise SpecError(f"{label} must be {qualifier}")
    return value


def _relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise SpecError(f"{label} must be a non-empty relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise SpecError(f"{label} must stay inside its stage directory: {value!r}")
    return path.as_posix()


def _pin(value: object, label: str) -> PinnedInput:
    record = _mapping(value, label)
    _exact_keys(record, {"path", "sha256"}, label)
    path = _relative_path(record["path"], f"{label}.path")
    if not is_sha256(record["sha256"]):
        raise SpecError(f"{label}.sha256 must be a lowercase 64-character digest")
    return PinnedInput.from_mapping({"path": path, "sha256": record["sha256"]})


@dataclass(frozen=True)
class OutputSpec:
    """One file a stage must produce before it can commit."""

    path: str
    schema_version: str

    @classmethod
    def from_mapping(cls, value: object, label: str) -> OutputSpec:
        record = _mapping(value, label)
        _exact_keys(record, {"path", "schema_version"}, label)
        schema = record["schema_version"]
        if not isinstance(schema, str) or not schema:
            raise SpecError(f"{label}.schema_version must be a non-empty string")
        return cls(
            path=_relative_path(record["path"], f"{label}.path"),
            schema_version=schema,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {"path": self.path, "schema_version": self.schema_version}


@dataclass(frozen=True)
class ResourceSpec:
    """Explicit runtime resources; nothing is inferred from the available machine."""

    device: str
    precision: str
    cpus: int
    workers: int
    memory_mb: int
    timeout_seconds: int
    gpu_type: str | None

    @classmethod
    def from_mapping(cls, value: object, label: str) -> ResourceSpec:
        record = _mapping(value, label)
        _exact_keys(
            record,
            {
                "device",
                "precision",
                "cpus",
                "workers",
                "memory_mb",
                "timeout_seconds",
                "gpu_type",
            },
            label,
        )
        device = record["device"]
        if device not in _DEVICES:
            raise SpecError(f"{label}.device must be one of {sorted(_DEVICES)}")
        precision = record["precision"]
        if precision not in _PRECISIONS:
            raise SpecError(f"{label}.precision must be one of {sorted(_PRECISIONS)}")
        gpu_type = record["gpu_type"]
        if gpu_type is not None and (not isinstance(gpu_type, str) or not gpu_type):
            raise SpecError(f"{label}.gpu_type must be null or a non-empty string")
        if device != "cuda" and gpu_type is not None:
            raise SpecError(f"{label}.gpu_type is only valid for a CUDA stage")
        return cls(
            device=str(device),
            precision=str(precision),
            cpus=_nonnegative_int(record["cpus"], f"{label}.cpus", positive=True),
            workers=_nonnegative_int(record["workers"], f"{label}.workers"),
            memory_mb=_nonnegative_int(record["memory_mb"], f"{label}.memory_mb", positive=True),
            timeout_seconds=_nonnegative_int(
                record["timeout_seconds"], f"{label}.timeout_seconds", positive=True
            ),
            gpu_type=gpu_type,
        )

    def with_device(self, device: str) -> ResourceSpec:
        if device not in _DEVICES:
            raise SpecError(f"device override must be one of {sorted(_DEVICES)}")
        return ResourceSpec(
            device=device,
            precision=self.precision,
            cpus=self.cpus,
            workers=self.workers,
            memory_mb=self.memory_mb,
            timeout_seconds=self.timeout_seconds,
            gpu_type=self.gpu_type if device == "cuda" else None,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cpus": self.cpus,
            "device": self.device,
            "gpu_type": self.gpu_type,
            "memory_mb": self.memory_mb,
            "precision": self.precision,
            "timeout_seconds": self.timeout_seconds,
            "workers": self.workers,
        }


@dataclass(frozen=True)
class DeterminismSpec:
    """How a stage's repeated outputs are compared."""

    mode: str
    stream: str

    @classmethod
    def from_mapping(cls, value: object, label: str) -> DeterminismSpec:
        record = _mapping(value, label)
        _exact_keys(record, {"mode", "stream"}, label)
        mode = record["mode"]
        if mode not in _DETERMINISM_MODES:
            raise SpecError(f"{label}.mode must be one of {sorted(_DETERMINISM_MODES)}")
        return cls(mode=str(mode), stream=_identifier(record["stream"], f"{label}.stream"))

    def to_mapping(self) -> dict[str, Any]:
        return {"mode": self.mode, "stream": self.stream}


@dataclass(frozen=True)
class StageSpec:
    """A registered stage plus its immutable external contract."""

    stage_id: str
    implementation: str
    needs: tuple[str, ...]
    config: PinnedInput
    inputs: Mapping[str, PinnedInput]
    outputs: Mapping[str, OutputSpec]
    resources: ResourceSpec
    determinism: DeterminismSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))

    @classmethod
    def from_mapping(cls, value: object, index: int) -> StageSpec:
        label = f"stages[{index}]"
        record = _mapping(value, label)
        _exact_keys(
            record,
            {
                "id",
                "implementation",
                "needs",
                "config",
                "inputs",
                "outputs",
                "resources",
                "determinism",
            },
            label,
        )
        needs = tuple(
            _identifier(item, f"{label}.needs[{need_index}]")
            for need_index, item in enumerate(_sequence(record["needs"], f"{label}.needs"))
        )
        if len(needs) != len(set(needs)):
            raise SpecError(f"{label}.needs contains duplicates")
        inputs_record = _mapping(record["inputs"], f"{label}.inputs")
        outputs_record = _mapping(record["outputs"], f"{label}.outputs")
        inputs = {
            _identifier(name, f"{label}.inputs key"): _pin(pin, f"{label}.inputs.{name}")
            for name, pin in inputs_record.items()
        }
        outputs = {
            _identifier(name, f"{label}.outputs key"): OutputSpec.from_mapping(
                output, f"{label}.outputs.{name}"
            )
            for name, output in outputs_record.items()
        }
        paths = [output.path for output in outputs.values()]
        if len(paths) != len(set(paths)):
            raise SpecError(f"{label}.outputs contains duplicate paths")
        return cls(
            stage_id=_identifier(record["id"], f"{label}.id"),
            implementation=_identifier(record["implementation"], f"{label}.implementation"),
            needs=needs,
            config=_pin(record["config"], f"{label}.config"),
            inputs=inputs,
            outputs=outputs,
            resources=ResourceSpec.from_mapping(record["resources"], f"{label}.resources"),
            determinism=DeterminismSpec.from_mapping(record["determinism"], f"{label}.determinism"),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "config": self.config.to_mapping(),
            "determinism": self.determinism.to_mapping(),
            "id": self.stage_id,
            "implementation": self.implementation,
            "inputs": {name: pin.to_mapping() for name, pin in sorted(self.inputs.items())},
            "needs": list(self.needs),
            "outputs": {name: output.to_mapping() for name, output in sorted(self.outputs.items())},
            "resources": self.resources.to_mapping(),
        }


@dataclass(frozen=True)
class ExperimentSpec:
    """A validated acyclic experiment DAG."""

    experiment_id: str
    description: str
    root_seed: int
    profiles: tuple[str, ...]
    replicates: Mapping[str, int]
    stages: tuple[StageSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    nonclaims: tuple[str, ...] = ()
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "replicates", MappingProxyType(dict(self.replicates)))
        self.topological_stages()

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentSpec:
        record = _mapping(value, "experiment")
        _exact_keys(
            record,
            {
                "schema_version",
                "experiment_id",
                "description",
                "root_seed",
                "profiles",
                "replicates",
                "stages",
                "metadata",
                "nonclaims",
            },
            "experiment",
        )
        if record["schema_version"] != EXPERIMENT_SCHEMA_VERSION:
            raise SpecError(
                f"unsupported experiment schema {record['schema_version']!r}; "
                f"expected {EXPERIMENT_SCHEMA_VERSION!r}"
            )
        description = record["description"]
        if not isinstance(description, str) or not description.strip():
            raise SpecError("experiment.description must be a non-empty string")
        profiles = tuple(
            _identifier(item, f"experiment.profiles[{index}]")
            for index, item in enumerate(_sequence(record["profiles"], "experiment.profiles"))
        )
        if not profiles or len(profiles) != len(set(profiles)):
            raise SpecError("experiment.profiles must contain unique profile names")
        replicate_record = _mapping(record["replicates"], "experiment.replicates")
        if set(replicate_record) != set(profiles):
            raise SpecError("experiment.replicates must define exactly every profile")
        replicates = {
            profile: _nonnegative_int(
                replicate_record[profile],
                f"experiment.replicates.{profile}",
                positive=True,
            )
            for profile in profiles
        }
        stages = tuple(
            StageSpec.from_mapping(item, index)
            for index, item in enumerate(_sequence(record["stages"], "experiment.stages"))
        )
        if not stages:
            raise SpecError("experiment.stages must not be empty")
        identifiers = [stage.stage_id for stage in stages]
        if len(identifiers) != len(set(identifiers)):
            raise SpecError("experiment.stages contains duplicate ids")
        metadata = _mapping(record["metadata"], "experiment.metadata")
        nonclaims_value = _sequence(record["nonclaims"], "experiment.nonclaims")
        if not all(isinstance(item, str) and item for item in nonclaims_value):
            raise SpecError("experiment.nonclaims must contain non-empty strings")
        return cls(
            experiment_id=_identifier(record["experiment_id"], "experiment.experiment_id"),
            description=description.strip(),
            root_seed=_nonnegative_int(record["root_seed"], "experiment.root_seed"),
            profiles=profiles,
            replicates=replicates,
            stages=stages,
            metadata=dict(metadata),
            nonclaims=tuple(nonclaims_value),
        )

    @classmethod
    def load(cls, path: Path) -> ExperimentSpec:
        return cls.from_mapping(read_json_object(path, error=SpecError, label="experiment spec"))

    def stage(self, stage_id: str) -> StageSpec:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        raise SpecError(f"unknown stage {stage_id!r}")

    def topological_stages(self) -> tuple[StageSpec, ...]:
        by_id = {stage.stage_id: stage for stage in self.stages}
        for stage in self.stages:
            missing = set(stage.needs) - set(by_id)
            if missing:
                raise SpecError(
                    f"stage {stage.stage_id!r} depends on unknown stages {sorted(missing)}"
                )
            if stage.stage_id in stage.needs:
                raise SpecError(f"stage {stage.stage_id!r} depends on itself")

        ordered: list[StageSpec] = []
        permanent: set[str] = set()
        temporary: set[str] = set()

        def visit(stage: StageSpec, trail: tuple[str, ...]) -> None:
            if stage.stage_id in permanent:
                return
            if stage.stage_id in temporary:
                cycle = " -> ".join((*trail, stage.stage_id))
                raise SpecError(f"experiment stage dependency cycle: {cycle}")
            temporary.add(stage.stage_id)
            for dependency in stage.needs:
                visit(by_id[dependency], (*trail, stage.stage_id))
            temporary.remove(stage.stage_id)
            permanent.add(stage.stage_id)
            ordered.append(stage)

        for stage in self.stages:
            visit(stage, ())
        return tuple(ordered)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "experiment_id": self.experiment_id,
            "metadata": dict(self.metadata),
            "nonclaims": list(self.nonclaims),
            "profiles": list(self.profiles),
            "replicates": dict(self.replicates),
            "root_seed": self.root_seed,
            "schema_version": self.schema_version,
            "stages": [stage.to_mapping() for stage in self.stages],
        }


__all__ = [
    "DETERMINISM_MODES",
    "EXPERIMENT_SCHEMA_VERSION",
    "DeterminismSpec",
    "ExperimentSpec",
    "OutputSpec",
    "ResourceSpec",
    "StageSpec",
]

# Public constant under a non-private name for schema/documentation tooling.
DETERMINISM_MODES = _DETERMINISM_MODES
