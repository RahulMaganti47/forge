"""Content identity for the productive selected Ugi generator implementation."""

from __future__ import annotations

import hashlib
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import rdBase

from forge.core.io import stable_json as _stable_json

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - productive generation requires torch
    torch = None

SELECTED_GENERATOR_IMPLEMENTATION_SCHEMA_VERSION = "forge.selected_ugi_generator_implementation.v1"
SELECTED_GENERATOR_SOURCE_PATHS = (
    "experiments/phase1/product_l1/sampling/ugi_end_to_end_sampling.py",
    "experiments/phase1/product_l1/sampling/ugi_joint_end_to_end_sampling.py",
    "experiments/phase1/product_l1/sampling/ugi_joint_sparse_sampling.py",
    "experiments/phase1/product_l1/sampling/ugi_selected_generator_implementation.py",
    "experiments/phase1/product_l1/sampling/ugi_selected_restartable_generator.py",
    "experiments/phase1/product_l1/training/ugi_training_cache.py",
    "experiments/phase1/synthesis_guidance/adapters/terminal_support.py",
    "experiments/phase1/synthesis_guidance/guidance/ugi_synthesis_guidance.py",
    "forge/chemistry/smiles.py",
    "forge/corpus/r0_splits.py",
    "forge/corpus/r1_prime_audit.py",
    "forge/corpus/ugi_chemistry_corpus.py",
    "forge/corpus/ugi_component_expansion.py",
    "forge/corpus/ugi_generated_components.py",
    "forge/corpus/ugi_generated_terminal_support.py",
    "forge/corpus/ugi_held_component_gate.py",
    "forge/corpus/ugi_morphology_corpus.py",
    "forge/model/adapter_node_conditioning.py",
    "forge/model/defog_feasibility.py",
    "forge/model/eligibility.py",
    "forge/model/phase1_flow.py",
    "forge/model/phase1_tree_topology_flow.py",
    "forge/model/ugi_adapter_features.py",
    "forge/model/ugi_chemistry_flow.py",
    "forge/model/ugi_chemistry_interface.py",
    "forge/model/ugi_closure_placement.py",
    "forge/model/ugi_joint_sparse_flow.py",
    "forge/model/ugi_morphology_flow.py",
    "forge/model/ugi_morphology_program.py",
    "forge/model/v5_morphology_program.py",
    "forge/model/v5_sparse_representation.py",
    "forge/potency/annotations.py",
    "forge/synthesis/matched.py",
    "forge/synthesis/terminals/terminal_assessment.py",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiSelectedGeneratorImplementationError(RuntimeError):
    """Raised when productive generator code/runtime identity cannot be sealed."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise UgiSelectedGeneratorImplementationError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


@dataclass(frozen=True, order=True)
class SelectedGeneratorSourceArtifact:
    """One source file loaded by the productive generation/support path."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise UgiSelectedGeneratorImplementationError("generator source path must be nonempty")
        _require_sha256(self.sha256, label=f"generator source {self.path}")

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class SelectedGeneratorImplementationQualification:
    """Exact productive source bundle and numerical runtime versions."""

    sources: tuple[SelectedGeneratorSourceArtifact, ...]
    runtime_versions: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.sources != tuple(sorted(set(self.sources))) or not self.sources:
            raise UgiSelectedGeneratorImplementationError(
                "generator sources must be a nonempty unique sorted tuple"
            )
        if tuple(item.path for item in self.sources) != SELECTED_GENERATOR_SOURCE_PATHS:
            raise UgiSelectedGeneratorImplementationError(
                "generator source bundle does not match the frozen productive path set"
            )
        if (
            not self.runtime_versions
            or self.runtime_versions != tuple(sorted(set(self.runtime_versions)))
            or any(not name or not version for name, version in self.runtime_versions)
        ):
            raise UgiSelectedGeneratorImplementationError(
                "generator runtime versions must be nonempty, unique and sorted"
            )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTED_GENERATOR_IMPLEMENTATION_SCHEMA_VERSION,
            "sources": [item.to_dict() for item in self.sources],
            "runtime_versions": [list(item) for item in self.runtime_versions],
        }

    @property
    def implementation_sha256(self) -> str:
        return _sha256_bytes(_stable_json(self._content_dict()).encode())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self._content_dict(),
            "implementation_sha256": self.implementation_sha256,
        }


def build_selected_generator_implementation_qualification(
    repository: Path,
) -> SelectedGeneratorImplementationQualification:
    """Hash every productive source and capture exact numerical runtime versions."""

    if torch is None:
        raise UgiSelectedGeneratorImplementationError(
            "productive generator implementation requires torch"
        )
    root = Path(repository).resolve()
    sources = []
    for relative_path in SELECTED_GENERATOR_SOURCE_PATHS:
        path = root / relative_path
        if not path.is_file() or path.is_symlink():
            raise UgiSelectedGeneratorImplementationError(
                f"generator source is missing or not a real file: {relative_path}"
            )
        sources.append(
            SelectedGeneratorSourceArtifact(
                path=relative_path,
                sha256=_sha256_bytes(path.read_bytes()),
            )
        )
    return SelectedGeneratorImplementationQualification(
        sources=tuple(sources),
        runtime_versions=tuple(
            sorted(
                (
                    ("numpy", np.__version__),
                    ("python", platform.python_version()),
                    ("rdkit", rdBase.rdkitVersion),
                    ("torch", torch.__version__),
                )
            )
        ),
    )


def require_selected_generator_implementation_unchanged(
    repository: Path,
    expected: SelectedGeneratorImplementationQualification,
) -> None:
    """Fail when productive code or runtime identity changed after preflight."""

    if not isinstance(expected, SelectedGeneratorImplementationQualification):
        raise UgiSelectedGeneratorImplementationError(
            "expected generator implementation qualification is malformed"
        )
    observed = build_selected_generator_implementation_qualification(repository)
    if observed != expected:
        raise UgiSelectedGeneratorImplementationError(
            "productive generator implementation changed after qualification"
        )


__all__ = [
    "SELECTED_GENERATOR_IMPLEMENTATION_SCHEMA_VERSION",
    "SELECTED_GENERATOR_SOURCE_PATHS",
    "SelectedGeneratorImplementationQualification",
    "SelectedGeneratorSourceArtifact",
    "UgiSelectedGeneratorImplementationError",
    "build_selected_generator_implementation_qualification",
    "require_selected_generator_implementation_unchanged",
]
