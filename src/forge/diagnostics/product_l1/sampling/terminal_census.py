"""Calibration-only dynamic terminal census for the frozen selected-v3 Ugi generator.

This module is additive.  It does not modify the selected-v2/v3 generator or
guidance adapters and it deliberately stops at native exact-L1 terminal
completion.  It never evaluates potency, routes, synthesis values, proposals,
or candidate selection.

The census has a nested design: frozen morphology programs, independent noisy
states within each program, and independent terminal continuations from exact
flow checkpoints.  Partial states are saved as primitive/tensor-only payloads
that can be loaded with ``torch.load(..., weights_only=True)``.
"""

from __future__ import annotations

import gzip
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.diagnostics.product_l1.sampling.ugi_joint_end_to_end_sampling import (
    complete_ugi_joint_terminals,
)
from forge.diagnostics.product_l1.sampling.ugi_joint_sparse_sampling import (
    UgiJointSparseTrajectoryState,
    advance_ugi_joint_sparse_state,
    finalize_ugi_joint_sparse_state,
)
from forge.diagnostics.product_l1.sampling.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.diagnostics.product_l1.sampling.ugi_selected_restartable_generator_v2 import (
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_TEMPERATURE,
)
from forge.diagnostics.support.adapters.selected_v1 import (
    SelectedGuidanceState,
)
from forge.model.defog_feasibility import sha256_file

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - productive census requires torch
    torch = None


CONFIG_SCHEMA_VERSION = "phase1_ugi_dynamic_frozen_prior_terminal_census_config.v1"
STATE_SCHEMA_VERSION = "forge.ugi_dynamic_frozen_prior_partial_state.v1"
SHARD_SCHEMA_VERSION = "phase1_ugi_dynamic_frozen_prior_terminal_census_shard.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_dynamic_frozen_prior_terminal_census.v1"

EXPECTED_INPUT_KEYS = frozenset(
    {
        "fresh_pool_config",
        "frozen_programs",
        "prior_terminal_reference",
        "production_generator_manifest",
        "generator_checkpoint",
        "closure_checkpoint",
        "selected_v3_adapter_source",
        "selected_v2_equivalence_result",
        "selected_v2_equivalence_rows",
        "census_source",
        "census_runner",
        "census_tests",
    }
)
EXPECTED_SCOPE = {
    "partition": "calibration",
    "candidate_selection": False,
    "guidance": False,
    "nonzero_guidance": False,
    "oracle_calls": 0,
    "potency_predictions": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "proposal_calls": 0,
    "sealed_holdout_access": False,
    "prospective_candidate_lock": False,
    "retries_or_repairs": False,
}


class UgiDynamicTerminalCensusError(RuntimeError):
    """Raised when the frozen-prior census contract is violated."""


def _stable_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiDynamicTerminalCensusError("value is not canonically serializable") from error


def _lower_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise UgiDynamicTerminalCensusError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise UgiDynamicTerminalCensusError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiDynamicTerminalCensusError(f"{label} must be a nonnegative integer")
    return value


def _jsonl_gzip_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    from io import BytesIO

    buffer = BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as handle:
        for row in rows:
            handle.write(_stable_json_bytes(dict(row)) + b"\n")
    return buffer.getvalue()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDynamicTerminalCensusError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiDynamicTerminalCensusError(f"{label} must be a JSON object")
    return value


def _resolve_pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiDynamicTerminalCensusError(f"{label} pin is malformed")
    expected = _lower_sha256(record["sha256"], label=f"{label} sha256")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiDynamicTerminalCensusError(f"{label} path escapes the repository") from error
    if not path.is_file() or path.is_symlink():
        raise UgiDynamicTerminalCensusError(f"{label} is missing or is not a real file")
    if sha256_file(path) != expected:
        raise UgiDynamicTerminalCensusError(f"{label} file hash changed")
    return path


@dataclass(frozen=True)
class CensusDesign:
    """Frozen nested sampling design."""

    population_programs: int
    selected_programs: int
    states_per_program: int
    checkpoints: tuple[int, ...]
    rollouts_per_state_checkpoint: int
    sample_steps: int
    shard_programs: int
    program_selection_seed: int
    particle_seed_base: int
    rollout_seed_base: int
    device: str

    def __post_init__(self) -> None:
        for label in (
            "population_programs",
            "selected_programs",
            "states_per_program",
            "rollouts_per_state_checkpoint",
            "sample_steps",
            "shard_programs",
        ):
            _positive_integer(getattr(self, label), label=label)
        for label in ("program_selection_seed", "particle_seed_base", "rollout_seed_base"):
            _nonnegative_integer(getattr(self, label), label=label)
        if self.selected_programs > self.population_programs:
            raise UgiDynamicTerminalCensusError(
                "selected_programs cannot exceed population_programs"
            )
        if self.device != "cpu":
            raise UgiDynamicTerminalCensusError("selected-v3 census is CPU-qualified only")
        if (
            not self.checkpoints
            or tuple(sorted(set(self.checkpoints))) != self.checkpoints
            or any(step <= 0 or step >= self.sample_steps for step in self.checkpoints)
        ):
            raise UgiDynamicTerminalCensusError(
                "checkpoints must be unique increasing interior flow steps"
            )

    @property
    def selected_partial_states(self) -> int:
        return self.selected_programs * self.states_per_program * len(self.checkpoints)

    @property
    def terminal_attempts(self) -> int:
        return self.selected_partial_states * self.rollouts_per_state_checkpoint

    @property
    def shard_count(self) -> int:
        return math.ceil(self.selected_programs / self.shard_programs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "population_programs": self.population_programs,
            "selected_programs": self.selected_programs,
            "states_per_program": self.states_per_program,
            "checkpoints": list(self.checkpoints),
            "rollouts_per_state_checkpoint": self.rollouts_per_state_checkpoint,
            "sample_steps": self.sample_steps,
            "shard_programs": self.shard_programs,
            "program_selection_seed": self.program_selection_seed,
            "particle_seed_base": self.particle_seed_base,
            "rollout_seed_base": self.rollout_seed_base,
            "device": self.device,
            "selected_partial_states": self.selected_partial_states,
            "terminal_attempts": self.terminal_attempts,
            "shard_count": self.shard_count,
        }


@dataclass(frozen=True)
class CensusContract:
    """Validated config and immutable input paths."""

    repo: Path
    config_path: Path
    config_sha256: str
    design: CensusDesign
    inputs: Mapping[str, Path]
    scope: Mapping[str, Any]


def _design_from_config(value: Any) -> CensusDesign:
    if not isinstance(value, dict):
        raise UgiDynamicTerminalCensusError("design must be a JSON object")
    expected = {
        "population_programs",
        "selected_programs",
        "states_per_program",
        "checkpoints",
        "rollouts_per_state_checkpoint",
        "sample_steps",
        "shard_programs",
        "program_selection_seed",
        "particle_seed_base",
        "rollout_seed_base",
        "device",
        "selection_method",
        "terminal_decoder_mode",
        "terminal_temperature",
        "maximum_adjacent_branch_runs",
        "expected_partial_states",
        "expected_terminal_attempts",
    }
    if set(value) != expected:
        raise UgiDynamicTerminalCensusError("design fields changed")
    if value["selection_method"] != "sha256_priority_without_replacement_v1":
        raise UgiDynamicTerminalCensusError("selection method changed")
    if value["terminal_decoder_mode"] != "bond_stochastic":
        raise UgiDynamicTerminalCensusError("terminal decoder changed")
    if float(value["terminal_temperature"]) != TERMINAL_TEMPERATURE:
        raise UgiDynamicTerminalCensusError("terminal temperature changed")
    if tuple(value["maximum_adjacent_branch_runs"]) != MAXIMUM_ADJACENT_BRANCH_RUNS:
        raise UgiDynamicTerminalCensusError("adjacent branch-run policy changed")
    design = CensusDesign(
        population_programs=value["population_programs"],
        selected_programs=value["selected_programs"],
        states_per_program=value["states_per_program"],
        checkpoints=tuple(value["checkpoints"]),
        rollouts_per_state_checkpoint=value["rollouts_per_state_checkpoint"],
        sample_steps=value["sample_steps"],
        shard_programs=value["shard_programs"],
        program_selection_seed=value["program_selection_seed"],
        particle_seed_base=value["particle_seed_base"],
        rollout_seed_base=value["rollout_seed_base"],
        device=value["device"],
    )
    if design.sample_steps != SAMPLE_STEPS:
        raise UgiDynamicTerminalCensusError("selected-v3 sample-step count changed")
    if value["expected_partial_states"] != design.selected_partial_states:
        raise UgiDynamicTerminalCensusError("expected partial-state count is inconsistent")
    if value["expected_terminal_attempts"] != design.terminal_attempts:
        raise UgiDynamicTerminalCensusError("expected terminal-attempt count is inconsistent")
    return design


def load_census_contract(repo: Path, config_path: Path) -> CensusContract:
    """Validate the frozen census config without constructing the model."""

    root = repo.resolve()
    resolved_config = config_path.resolve()
    try:
        resolved_config.relative_to(root)
    except ValueError as error:
        raise UgiDynamicTerminalCensusError("config path escapes the repository") from error
    config = _read_json(resolved_config, label="census config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiDynamicTerminalCensusError("unsupported census config schema")
    if config.get("status") != "frozen_before_calibration_only_census":
        raise UgiDynamicTerminalCensusError("census config is not frozen")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiDynamicTerminalCensusError("census scope changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or frozenset(inputs) != EXPECTED_INPUT_KEYS:
        raise UgiDynamicTerminalCensusError("census input set changed")
    paths = {label: _resolve_pin(root, pin, label=label) for label, pin in inputs.items()}
    design = _design_from_config(config.get("design"))
    return CensusContract(
        repo=root,
        config_path=resolved_config,
        config_sha256=sha256_file(resolved_config),
        design=design,
        inputs=paths,
        scope=dict(config["scope"]),
    )


def _replace_trajectory_generator_state(
    trajectory: UgiJointSparseTrajectoryState,
    generator_state: Any,
) -> UgiJointSparseTrajectoryState:
    return UgiJointSparseTrajectoryState(
        programs=trajectory.programs,
        layout={key: value.clone() for key, value in trajectory.layout.items()},
        sources={key: value.clone() for key, value in trajectory.sources.items()},
        channels={key: value.clone() for key, value in trajectory.channels.items()},
        sample_steps=trajectory.sample_steps,
        step=trajectory.step,
        generator_state=generator_state.clone(),
        device=trajectory.device,
    )


def complete_native_no_route_terminal(
    lane: Any,
    state: SelectedGuidanceState,
    *,
    particle_index: int,
    seed: int,
    checkpoint_index: int,
    rollout_index: int,
) -> dict[str, Any]:
    """Complete one clone to a native L1-annotated row with no value calls."""

    if torch is None:
        raise UgiDynamicTerminalCensusError("native terminal completion requires torch")
    _nonnegative_integer(particle_index, label="particle_index")
    _nonnegative_integer(seed, label="seed")
    _nonnegative_integer(rollout_index, label="rollout_index")
    if checkpoint_index not in (2, 4, 6) or checkpoint_index != state.step:
        raise UgiDynamicTerminalCensusError("native continuation requires checkpoint 2, 4 or 6")
    if particle_index >= len(state.particles):
        raise UgiDynamicTerminalCensusError("particle index exceeds partial state")
    if state.adapter_identity_sha256 != getattr(lane, "adapter_identity_sha256", None):
        raise UgiDynamicTerminalCensusError("lane and partial-state identities differ")
    particle = state.particles[particle_index]
    source_before = particle.state_sha256
    trajectory = particle.trajectory.clone()
    flow_state = torch.Generator(device=trajectory.device).manual_seed(seed).get_state()
    trajectory = _replace_trajectory_generator_state(trajectory, flow_state)
    trajectory = advance_ugi_joint_sparse_state(
        lane.callback.model,
        trajectory,
        target_step=SAMPLE_STEPS,
    )
    finalization = finalize_ugi_joint_sparse_state(
        lane.callback.model,
        trajectory,
        tree_generator_state=torch.Generator().manual_seed(seed + 1).get_state(),
        allowed_ring_sizes=lane.callback.allowed_ring_sizes,
        maximum_heavy_degree=lane.callback.maximum_heavy_degree,
        maximum_adjacent_branch_runs=MAXIMUM_ADJACENT_BRANCH_RUNS,
    )
    if len(finalization.terminals) != 1:
        raise UgiDynamicTerminalCensusError(
            "one partial particle did not finalize to exactly one terminal"
        )
    completion = complete_ugi_joint_terminals(
        lane.callback.model,
        lane.callback.closure_model,
        finalization.terminals,
        lane.callback.corpus,
        program_metadata=({},),
        closure_generator_state=torch.Generator().manual_seed(seed + 1).get_state(),
        allowed_ring_sizes=lane.callback.allowed_ring_sizes,
        maximum_heavy_degree=lane.callback.maximum_heavy_degree,
        l1_reaction=lane.callback.reaction,
        terminal_decoder_mode="bond_stochastic",
        terminal_generator_state=torch.Generator().manual_seed(seed + 2).get_state(),
        terminal_temperature=TERMINAL_TEMPERATURE,
    )
    if len(completion.rows) != 1:
        raise UgiDynamicTerminalCensusError(
            "one partial particle did not complete to exactly one native row"
        )
    native = dict(completion.rows[0])
    source_after = particle.state_sha256
    if source_after != source_before:
        raise UgiDynamicTerminalCensusError("native continuation mutated its partial state")
    identity = {
        "adapter_identity_sha256": state.adapter_identity_sha256,
        "particle_state_sha256": source_before,
        "particle_index": particle_index,
        "checkpoint_index": checkpoint_index,
        "rollout_index": rollout_index,
        "seed": seed,
    }
    return {
        "completion_id": _sha256_payload(identity),
        "status": "terminal",
        **identity,
        "source_state_unchanged": True,
        "product_transition_calls": SAMPLE_STEPS - checkpoint_index,
        "error_detail": None,
        "native_terminal": native,
    }


NativeCompleter = Callable[..., dict[str, Any]]


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "CensusContract",
    "CensusDesign",
    "EXPECTED_SCOPE",
    "RESULT_SCHEMA_VERSION",
    "SHARD_SCHEMA_VERSION",
    "STATE_SCHEMA_VERSION",
    "UgiDynamicTerminalCensusError",
    "complete_native_no_route_terminal",
    "load_census_contract",
]
