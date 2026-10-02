"""Equivalence-bound restartable guidance adapter for the selected Ugi v2 lane.

This additive adapter does not alter the frozen selected-v2 generator or its
guidance adapter.  It validates the repaired production-runtime
pooled/singleton equivalence receipt and every comparison row, replaces only
the v2 closure identity's pending-equivalence token, and binds the resulting
complete closure identity into a new guidance-adapter identity.

Construction and every state operation fail closed if the receipt, comparison
rows, this source, or either frozen selected-v2 source changes.  The module
executes no guidance, routing, biology, candidate selection, retry, or repair.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from forge.experiments.hela.adapters.selected_v1 import (
    SelectedGuidanceState,
    UgiSelectedGuidanceAdapterError,
    _sha256_payload,
)
from forge.experiments.hela.adapters.selected_v2 import (
    SelectedModelRestartableGuidanceLaneV2,
)
from forge.experiments.hela.guidance.ugi_zero_guidance_rehearsal import (
    RestartableGeneratorClosureAdapter,
    RestartableGeneratorClosureIdentity,
)
from forge.experiments.hela.sampling.ugi_selected_generator_implementation import (
    build_selected_generator_implementation_qualification,
)
from forge.experiments.hela.sampling.ugi_selected_restartable_generator import (
    SAMPLE_STEPS,
    SelectedRestartableGeneratorLane,
)
from forge.experiments.hela.sampling.ugi_selected_restartable_generator_v2 import (
    _PENDING_EQUIVALENCE_SHA256,
    CLOSURE_CHECKPOINT_SHA256,
    GENERATOR_CHECKPOINT_SHA256,
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_DECODER_ID,
    build_selected_step2000_bond_stochastic_lane,
)

SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION = "forge.selected_guidance_adapter.v3"
EQUIVALENCE_RESULT_SCHEMA_VERSION = "phase1_ugi_selected_v2_pool_singleton_equivalence.v2"
EQUIVALENCE_RESULT_STATUS = "selected_v2_production_runtime_pool_singleton_equivalent"
EQUIVALENCE_RESULT_DECISION = "qualified_for_equivalence_binding_only"
EQUIVALENCE_RESULT_PATH = "results/phase1/ugi_selected_v2_pool_singleton_equivalence_v2/result.json"
EQUIVALENCE_ROWS_PATH = (
    "results/phase1/ugi_selected_v2_pool_singleton_equivalence_v2/comparison_rows.json"
)
EQUIVALENCE_RESULT_FILE_SHA256 = "e13eb65ce1653acd1d7ca6443eecb7dbc85ba90dd1e299ae705b409cc6b6f11d"
EQUIVALENCE_RESULT_LOGICAL_SHA256 = (
    "3d2bd2aef30896cca413f5573000416fa3872b5763badba5d012c4128ea5f720"
)
EQUIVALENCE_ROWS_FILE_SHA256 = "4baf6a9a1bde3b017ad46f1bc39c2bf9d3ca4957421c18788ffbd9b646468691"
EQUIVALENCE_ROWS_LOGICAL_SHA256 = "71ba25bff3b2fa754dbdd368352af241c5ac8e23dce6549f0efdb8575266bad1"
EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256 = (
    "ea06a5b6300657757ddb70364ecc4944424d4fb9e69b7cdb6aed9f07f83d574a"
)
EXPECTED_V2_GENERATOR_SOURCE_SHA256 = (
    "b632f401dc8b6d313433647c52481a0d2fba3eced4c493cfaa114c84d2ee5f52"
)
EXPECTED_V2_GUIDANCE_SOURCE_SHA256 = (
    "61a223218b7d28c2324773c92333f96b7f61b2880b49230d97de2453f64707cb"
)
V2_GENERATOR_SOURCE_PATH = "src/forge/product/ugi_selected_restartable_generator_v2.py"
V2_GUIDANCE_SOURCE_PATH = "src/forge/product/ugi_selected_guidance_adapter_v2.py"
EXPECTED_SCOPE = {
    "biology": False,
    "candidate_selection": False,
    "guidance": False,
    "holdout_access": False,
    "nonzero_guidance": False,
    "nonzero_guidance_authorized": False,
    "production_execution": False,
    "production_sources_exercised": True,
    "routing": False,
}
EXPECTED_RECEIPT_KEYS = {
    "comparison_rows_artifact",
    "config",
    "decision",
    "execution",
    "initialization",
    "inputs",
    "prior_equivalence",
    "result_sha256",
    "runtime_qualification",
    "schema_version",
    "scope",
    "selected_generator",
    "status",
}
EXPECTED_INPUT_KEYS = {
    "audit_runner",
    "audit_source",
    "audit_tests",
    "grouped_schedule",
    "prior_selected_v2_equivalence",
    "production_generator_manifest",
    "qualified_runtime_receipt",
    "selected_guidance_adapter_v2",
    "selected_restartable_generator_v2",
}
EXPECTED_STATE_STEPS = (0, 2, 4, 6, 8)
EXPECTED_COMPLETION_STEPS = (2, 4, 6)
EXPECTED_PARTICLES = 64
EXPECTED_PARTICLES_PER_PROGRAM = 4
EXPECTED_PROGRAMS = 16
EXPECTED_ROWS = 576
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiSelectedGuidanceAdapterV3Error(UgiSelectedGuidanceAdapterError):
    """Raised when the equivalence-bound selected-v2 seam is invalid."""


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} is not a lowercase SHA-256 digest")
    return value


def _repository_file(repository: Path, relative_path: Any, *, label: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} path is malformed")
    path = (repository / relative_path).resolve()
    try:
        path.relative_to(repository)
    except ValueError as error:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink():
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} is missing or not a real file")
    return path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedGuidanceAdapterV3Error(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} must be a JSON object")
    return value


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedGuidanceAdapterV3Error(
            f"invalid equivalence comparison rows: {path}"
        ) from error
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise UgiSelectedGuidanceAdapterV3Error(
            "equivalence comparison rows must be a JSON array of objects"
        )
    return value


def _require_pin(
    repository: Path,
    record: Any,
    *,
    label: str,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} pin is malformed")
    if expected_path is not None and record["path"] != expected_path:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} path changed")
    claimed = _require_sha256(record["sha256"], label=f"{label} hash")
    if expected_sha256 is not None and claimed != expected_sha256:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} declared hash changed")
    path = _repository_file(repository, record["path"], label=label)
    if _file_sha256(path) != claimed:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} file hash changed")
    return path


def _require_logical_result_hash(receipt: dict[str, Any]) -> str:
    claimed = _require_sha256(receipt.get("result_sha256"), label="equivalence logical result")
    content = {key: value for key, value in receipt.items() if key != "result_sha256"}
    if _sha256_payload(content) != claimed or claimed != EQUIVALENCE_RESULT_LOGICAL_SHA256:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence logical result hash changed")
    return claimed


def _require_hash_fields(value: dict[str, Any], fields: tuple[str, ...], *, label: str) -> None:
    for field in fields:
        _require_sha256(value.get(field), label=f"{label}.{field}")


def _validate_normalized_terminal(value: Any, *, expected_calls: int, label: str) -> None:
    expected_keys = {
        "candidate_record_sha256",
        "exact_l1",
        "product_transition_calls",
        "status",
        "terminal_bytes_sha256",
        "terminal_valid",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} terminal record is malformed")
    if (
        value["status"] != "terminal"
        or not isinstance(value["terminal_valid"], bool)
        or not isinstance(value["exact_l1"], bool)
        or value["product_transition_calls"] != expected_calls
    ):
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} terminal semantics changed")
    _require_hash_fields(
        value,
        ("candidate_record_sha256", "terminal_bytes_sha256"),
        label=label,
    )


def _validate_exact_terminal(
    value: Any,
    *,
    program_sha256: str,
    normalized: dict[str, Any],
    label: str,
) -> None:
    expected_keys = {
        "candidate_record_sha256",
        "checkpoint_index",
        "closure_checkpoint_sha256",
        "exact_l1",
        "generation_trace_sha256",
        "generator_checkpoint_sha256",
        "morphology_program_sha256",
        "product_transition_calls",
        "status",
        "terminal_bytes_sha256",
        "terminal_id",
        "terminal_valid",
        "unit_id",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} exact terminal is malformed")
    for key, expected in normalized.items():
        if value.get(key) != expected:
            raise UgiSelectedGuidanceAdapterV3Error(
                f"{label} exact terminal differs from its normalized record"
            )
    if (
        value["checkpoint_index"] != SAMPLE_STEPS
        or value["generator_checkpoint_sha256"] != GENERATOR_CHECKPOINT_SHA256
        or value["closure_checkpoint_sha256"] != CLOSURE_CHECKPOINT_SHA256
        or value["morphology_program_sha256"] != program_sha256
        or not isinstance(value["unit_id"], str)
        or not value["unit_id"]
        or not isinstance(value["terminal_id"], str)
        or not value["terminal_id"].startswith(f"{value['unit_id']}:")
    ):
        raise UgiSelectedGuidanceAdapterV3Error(f"{label} exact terminal semantics changed")
    _require_hash_fields(
        value,
        (
            "candidate_record_sha256",
            "closure_checkpoint_sha256",
            "generation_trace_sha256",
            "generator_checkpoint_sha256",
            "morphology_program_sha256",
            "terminal_bytes_sha256",
        ),
        label=label,
    )


def _expected_row_coordinates() -> tuple[tuple[str, int, int], ...]:
    coordinates: list[tuple[str, int, int]] = []
    coordinates.extend(("state", 0, particle) for particle in range(EXPECTED_PARTICLES))
    for step in EXPECTED_STATE_STEPS[1:]:
        coordinates.extend(("state", step, particle) for particle in range(EXPECTED_PARTICLES))
        if step in EXPECTED_COMPLETION_STEPS:
            coordinates.extend(
                ("checkpoint_completion", step, particle) for particle in range(EXPECTED_PARTICLES)
            )
    coordinates.extend(
        ("productive_completion", SAMPLE_STEPS, particle) for particle in range(EXPECTED_PARTICLES)
    )
    return tuple(coordinates)


def _validate_comparison_rows(
    rows: list[dict[str, Any]],
    *,
    expected_logical_sha256: str,
) -> None:
    if len(rows) != EXPECTED_ROWS or _sha256_payload(rows) != expected_logical_sha256:
        raise UgiSelectedGuidanceAdapterV3Error(
            "equivalence comparison-row count or logical hash changed"
        )
    coordinates = tuple(
        (row.get("comparison"), row.get("step"), row.get("particle_index")) for row in rows
    )
    if coordinates != _expected_row_coordinates():
        raise UgiSelectedGuidanceAdapterV3Error(
            "equivalence comparison-row ordering or coverage changed"
        )

    particle_seeds: dict[int, int] = {}
    program_hashes: dict[int, str] = {}
    for row in rows:
        comparison = row["comparison"]
        step = row["step"]
        particle = row["particle_index"]
        expected_program = particle // EXPECTED_PARTICLES_PER_PROGRAM
        if row.get("program_index") != expected_program or row.get("equal") is not True:
            raise UgiSelectedGuidanceAdapterV3Error(
                "comparison row changed program grouping or equality status"
            )
        particle_seed = row.get("particle_seed")
        if (
            isinstance(particle_seed, bool)
            or not isinstance(particle_seed, int)
            or particle_seed < 0
        ):
            raise UgiSelectedGuidanceAdapterV3Error("comparison particle seed is malformed")
        previous_seed = particle_seeds.setdefault(particle, particle_seed)
        if previous_seed != particle_seed:
            raise UgiSelectedGuidanceAdapterV3Error("comparison particle seed changed across rows")

        if comparison == "state":
            expected_keys = {
                "comparison",
                "equal",
                "particle_index",
                "particle_seed",
                "pooled",
                "program_index",
                "singleton",
                "step",
            }
            pooled = row.get("pooled")
            if set(row) != expected_keys or pooled != row.get("singleton"):
                raise UgiSelectedGuidanceAdapterV3Error("state comparison row is malformed")
            state_keys = {
                "categorical_state_sha256",
                "program_sha256",
                "rng_state_sha256",
                "step",
            }
            if not isinstance(pooled, dict) or set(pooled) != state_keys or pooled["step"] != step:
                raise UgiSelectedGuidanceAdapterV3Error("state comparison payload is malformed")
            _require_hash_fields(
                pooled,
                ("categorical_state_sha256", "program_sha256", "rng_state_sha256"),
                label="state comparison",
            )
            previous_program = program_hashes.setdefault(particle, pooled["program_sha256"])
            if previous_program != pooled["program_sha256"]:
                raise UgiSelectedGuidanceAdapterV3Error(
                    "particle morphology program changed across state rows"
                )
            continue

        common_keys = {
            "comparison",
            "equal",
            "particle_index",
            "particle_seed",
            "pooled",
            "pooled_source_after_sha256",
            "pooled_source_before_sha256",
            "program_index",
            "program_sha256",
            "singleton",
            "singleton_source_after_sha256",
            "singleton_source_before_sha256",
            "source_states_unchanged",
            "step",
            "within_program_particle_index",
        }
        if (
            row.get("within_program_particle_index") != particle % EXPECTED_PARTICLES_PER_PROGRAM
            or row.get("source_states_unchanged") is not True
            or row.get("pooled_source_before_sha256") != row.get("pooled_source_after_sha256")
            or row.get("singleton_source_before_sha256") != row.get("singleton_source_after_sha256")
            or row.get("pooled") != row.get("singleton")
        ):
            raise UgiSelectedGuidanceAdapterV3Error(
                "completion comparison equality or source nonmutation changed"
            )
        _require_hash_fields(
            row,
            (
                "pooled_source_before_sha256",
                "pooled_source_after_sha256",
                "program_sha256",
                "singleton_source_before_sha256",
                "singleton_source_after_sha256",
            ),
            label="completion comparison",
        )
        if program_hashes.get(particle) != row["program_sha256"]:
            raise UgiSelectedGuidanceAdapterV3Error(
                "completion comparison program differs from state comparison"
            )

        if comparison == "checkpoint_completion":
            if set(row) != common_keys | {"rollout_seed"}:
                raise UgiSelectedGuidanceAdapterV3Error("checkpoint completion row is malformed")
            rollout_seed = row.get("rollout_seed")
            if (
                isinstance(rollout_seed, bool)
                or not isinstance(rollout_seed, int)
                or rollout_seed < 0
            ):
                raise UgiSelectedGuidanceAdapterV3Error(
                    "checkpoint completion rollout seed is malformed"
                )
            _validate_normalized_terminal(
                row["pooled"],
                expected_calls=SAMPLE_STEPS - step,
                label="checkpoint pooled",
            )
            continue

        if comparison != "productive_completion" or set(row) != common_keys | {
            "direct",
            "invocation_seed",
            "pooled_exact",
        }:
            raise UgiSelectedGuidanceAdapterV3Error("productive completion row is malformed")
        if row.get("invocation_seed") != particle_seed + 1:
            raise UgiSelectedGuidanceAdapterV3Error("productive completion invocation seed changed")
        _validate_normalized_terminal(
            row["pooled"],
            expected_calls=0,
            label="productive pooled",
        )
        if row.get("pooled_exact") != row.get("direct"):
            raise UgiSelectedGuidanceAdapterV3Error(
                "productive pooled and direct exact completions differ"
            )
        _validate_exact_terminal(
            row["pooled_exact"],
            program_sha256=row["program_sha256"],
            normalized=row["pooled"],
            label="productive pooled",
        )

    if len(set(particle_seeds.values())) != EXPECTED_PARTICLES:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence particle seeds are not unique")
    grouped_programs = {
        program: {program_hashes[index] for index in range(program * 4, program * 4 + 4)}
        for program in range(EXPECTED_PROGRAMS)
    }
    if (
        any(len(values) != 1 for values in grouped_programs.values())
        or len({next(iter(values)) for values in grouped_programs.values()}) != EXPECTED_PROGRAMS
    ):
        raise UgiSelectedGuidanceAdapterV3Error("equivalence morphology-program grouping changed")


@dataclass(frozen=True)
class SelectedV2EquivalenceBinding:
    """Semantically validated artifact identities bound by the v3 adapter."""

    repository: Path
    result_path: Path
    rows_path: Path
    result_file_sha256: str
    result_logical_sha256: str
    rows_file_sha256: str
    rows_logical_sha256: str
    v2_generator_source_path: Path
    v2_generator_source_sha256: str
    v2_guidance_source_path: Path
    v2_guidance_source_sha256: str
    base_v2_adapter_identity_sha256: str

    def identity_dict(self) -> dict[str, Any]:
        return {
            "result": {
                "path": str(self.result_path.relative_to(self.repository)),
                "file_sha256": self.result_file_sha256,
                "logical_sha256": self.result_logical_sha256,
            },
            "comparison_rows": {
                "path": str(self.rows_path.relative_to(self.repository)),
                "file_sha256": self.rows_file_sha256,
                "logical_sha256": self.rows_logical_sha256,
            },
            "frozen_v2_sources": {
                str(self.v2_generator_source_path.relative_to(self.repository)): (
                    self.v2_generator_source_sha256
                ),
                str(self.v2_guidance_source_path.relative_to(self.repository)): (
                    self.v2_guidance_source_sha256
                ),
            },
            "base_v2_adapter_identity_sha256": self.base_v2_adapter_identity_sha256,
        }


def load_selected_v2_equivalence_binding(repository: Path) -> SelectedV2EquivalenceBinding:
    """Load and completely validate the repaired selected-v2 equivalence evidence."""

    root = Path(repository).resolve()
    result_path = _repository_file(root, EQUIVALENCE_RESULT_PATH, label="equivalence result")
    if _file_sha256(result_path) != EQUIVALENCE_RESULT_FILE_SHA256:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence result file hash changed")
    receipt = _load_json_object(result_path, label="equivalence result")
    if set(receipt) != EXPECTED_RECEIPT_KEYS:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence result schema keys changed")
    logical_result_sha256 = _require_logical_result_hash(receipt)
    if (
        receipt.get("schema_version") != EQUIVALENCE_RESULT_SCHEMA_VERSION
        or receipt.get("status") != EQUIVALENCE_RESULT_STATUS
        or receipt.get("decision") != EQUIVALENCE_RESULT_DECISION
        or receipt.get("scope") != EXPECTED_SCOPE
    ):
        raise UgiSelectedGuidanceAdapterV3Error("equivalence result is not binding-qualified")

    config_path = _require_pin(root, receipt.get("config"), label="equivalence config")
    del config_path
    inputs = receipt.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != EXPECTED_INPUT_KEYS:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence input set changed")
    pinned_inputs = {
        label: _require_pin(root, record, label=f"equivalence input {label}")
        for label, record in inputs.items()
    }
    v2_generator_source = pinned_inputs["selected_restartable_generator_v2"]
    v2_guidance_source = pinned_inputs["selected_guidance_adapter_v2"]
    if (
        str(v2_generator_source.relative_to(root)) != V2_GENERATOR_SOURCE_PATH
        or inputs["selected_restartable_generator_v2"]["sha256"]
        != EXPECTED_V2_GENERATOR_SOURCE_SHA256
        or str(v2_guidance_source.relative_to(root)) != V2_GUIDANCE_SOURCE_PATH
        or inputs["selected_guidance_adapter_v2"]["sha256"] != EXPECTED_V2_GUIDANCE_SOURCE_SHA256
    ):
        raise UgiSelectedGuidanceAdapterV3Error("frozen selected-v2 source binding changed")

    rows_path = _require_pin(
        root,
        receipt.get("comparison_rows_artifact"),
        label="equivalence comparison rows",
        expected_path=EQUIVALENCE_ROWS_PATH,
        expected_sha256=EQUIVALENCE_ROWS_FILE_SHA256,
    )
    rows = _load_json_rows(rows_path)
    _validate_comparison_rows(rows, expected_logical_sha256=EQUIVALENCE_ROWS_LOGICAL_SHA256)

    selected = receipt.get("selected_generator")
    if selected != {
        "base_adapter_identity_sha256": EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256,
        "checkpoint_sha256": GENERATOR_CHECKPOINT_SHA256,
        "maximum_adjacent_branch_runs": list(MAXIMUM_ADJACENT_BRANCH_RUNS),
        "terminal_decoder_id": TERMINAL_DECODER_ID,
    }:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence selected-generator identity changed")
    execution = receipt.get("execution")
    if execution != {
        "all_equal": True,
        "assignment_seed": 20260821,
        "assignment_sha256": ("c0b30c929cca59e08e9746050856761d2304934f883c06f1fb3379376c5daa27"),
        "checkpoint_completion_comparisons": 192,
        "comparison_rows_sha256": EQUIVALENCE_ROWS_LOGICAL_SHA256,
        "particles": EXPECTED_PARTICLES,
        "productive_direct_callback_comparisons": EXPECTED_PARTICLES,
        "productive_singleton_comparisons": EXPECTED_PARTICLES,
        "source_state_mutations": 0,
        "state_comparisons": 320,
    }:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence execution semantics changed")
    initialization = receipt.get("initialization")
    if initialization != {
        "expanded_program_manifest_sha256": (
            "9affa5ff2598ffc99376800b78269f8cf714670e06be30901c6e1364ab1716d4"
        ),
        "pooled_particle_seed_manifest_sha256": (
            "09fbd24a9c035a049055d1f3df299a4a67c2f1bb1669da8d6ec8e6ae9a97dfa1"
        ),
        "singleton_particle_seed_manifests_sha256": (
            "6c82152eb0677d23d3338b82bfc3c37bd09527e2c8856c1570a20e882fbafaf5"
        ),
    }:
        raise UgiSelectedGuidanceAdapterV3Error("equivalence initialization semantics changed")

    current_implementation = build_selected_generator_implementation_qualification(root).to_dict()
    if receipt.get("runtime_qualification") != current_implementation:
        raise UgiSelectedGuidanceAdapterV3Error(
            "equivalence runtime qualification differs from the current implementation"
        )
    prior = receipt.get("prior_equivalence")
    if (
        not isinstance(prior, dict)
        or set(prior)
        != {
            "adapter_identity_sha256",
            "file_sha256",
            "result_sha256",
        }
        or prior.get("adapter_identity_sha256") != EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256
        or prior.get("file_sha256") != inputs["prior_selected_v2_equivalence"]["sha256"]
    ):
        raise UgiSelectedGuidanceAdapterV3Error("prior equivalence chain changed")
    _require_hash_fields(prior, ("file_sha256", "result_sha256"), label="prior equivalence")

    return SelectedV2EquivalenceBinding(
        repository=root,
        result_path=result_path,
        rows_path=rows_path,
        result_file_sha256=EQUIVALENCE_RESULT_FILE_SHA256,
        result_logical_sha256=logical_result_sha256,
        rows_file_sha256=EQUIVALENCE_ROWS_FILE_SHA256,
        rows_logical_sha256=EQUIVALENCE_ROWS_LOGICAL_SHA256,
        v2_generator_source_path=v2_generator_source,
        v2_generator_source_sha256=EXPECTED_V2_GENERATOR_SOURCE_SHA256,
        v2_guidance_source_path=v2_guidance_source,
        v2_guidance_source_sha256=EXPECTED_V2_GUIDANCE_SOURCE_SHA256,
        base_v2_adapter_identity_sha256=EXPECTED_BASE_V2_ADAPTER_IDENTITY_SHA256,
    )


def _bind_equivalence_receipt(
    selected_lane: SelectedRestartableGeneratorLane,
    binding: SelectedV2EquivalenceBinding,
) -> SelectedRestartableGeneratorLane:
    if not isinstance(selected_lane, SelectedRestartableGeneratorLane):
        raise UgiSelectedGuidanceAdapterV3Error("selected-v2 lane is malformed")
    old_identity = selected_lane.adapter.identity
    if old_identity.restartable_equivalence_receipt_sha256 != _PENDING_EQUIVALENCE_SHA256:
        raise UgiSelectedGuidanceAdapterV3Error(
            "selected-v2 closure identity no longer contains the exact pending token"
        )
    if selected_lane.adapter.generate_locked_terminal is not selected_lane.callback:
        raise UgiSelectedGuidanceAdapterV3Error(
            "selected-v2 closure adapter does not own the selected callback"
        )
    new_identity = replace(
        old_identity,
        restartable_equivalence_receipt_sha256=binding.result_file_sha256,
    )
    old_values = old_identity.to_dict()
    new_values = new_identity.to_dict()
    changed = {key for key in old_values if old_values[key] != new_values[key]}
    if changed != {"restartable_equivalence_receipt_sha256"}:
        raise UgiSelectedGuidanceAdapterV3Error(
            "equivalence binding changed more than the pending closure-identity token"
        )
    adapter = RestartableGeneratorClosureAdapter(
        identity=new_identity,
        generate_locked_terminal=selected_lane.callback,
    )
    return replace(selected_lane, adapter=adapter)


class SelectedModelRestartableGuidanceLaneV3(SelectedModelRestartableGuidanceLaneV2):
    """Selected-v2 lane with repaired equivalence evidence in its identity."""

    def __init__(
        self,
        selected_lane: SelectedRestartableGeneratorLane,
        binding: SelectedV2EquivalenceBinding,
    ) -> None:
        if not isinstance(binding, SelectedV2EquivalenceBinding):
            raise UgiSelectedGuidanceAdapterV3Error("equivalence binding is malformed")
        self.equivalence_binding = binding
        self._v3_source_sha256 = _source_sha256()
        super().__init__(selected_lane)
        base_v2_identity = self.adapter_identity_sha256
        if base_v2_identity != binding.base_v2_adapter_identity_sha256:
            raise UgiSelectedGuidanceAdapterV3Error(
                "selected-v2 guidance identity differs from the equivalence receipt"
            )
        closure_identity = selected_lane.adapter.identity
        if not isinstance(closure_identity, RestartableGeneratorClosureIdentity):
            raise UgiSelectedGuidanceAdapterV3Error("bound closure identity is malformed")
        if closure_identity.restartable_equivalence_receipt_sha256 != binding.result_file_sha256:
            raise UgiSelectedGuidanceAdapterV3Error(
                "bound closure identity does not contain the equivalence receipt hash"
            )
        self.bound_closure_identity = closure_identity
        self.bound_closure_identity_sha256 = _sha256_payload(closure_identity.to_dict())
        self.base_v2_adapter_identity_sha256 = base_v2_identity
        self.adapter_identity_sha256 = _sha256_payload(
            {
                "schema_version": SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION,
                "base_v2_adapter_identity_sha256": base_v2_identity,
                "bound_closure_identity": closure_identity.to_dict(),
                "bound_closure_identity_sha256": self.bound_closure_identity_sha256,
                "equivalence_binding": binding.identity_dict(),
                "selected_bindings_sha256": selected_lane.bindings.canonical_sha256,
                "v3_adapter_source_sha256": self._v3_source_sha256,
            }
        )
        self._require_bound_artifacts_unchanged()

    def _require_bound_artifacts_unchanged(self) -> None:
        checks = (
            (
                self.equivalence_binding.result_path,
                self.equivalence_binding.result_file_sha256,
                "equivalence receipt",
            ),
            (
                self.equivalence_binding.rows_path,
                self.equivalence_binding.rows_file_sha256,
                "equivalence comparison rows",
            ),
            (
                self.equivalence_binding.v2_generator_source_path,
                self.equivalence_binding.v2_generator_source_sha256,
                "selected-v2 generator source",
            ),
            (
                self.equivalence_binding.v2_guidance_source_path,
                self.equivalence_binding.v2_guidance_source_sha256,
                "selected-v2 guidance source",
            ),
        )
        for path, expected, label in checks:
            if not path.is_file() or path.is_symlink() or _file_sha256(path) != expected:
                raise UgiSelectedGuidanceAdapterV3Error(f"bound {label} changed after construction")
        if _source_sha256() != self._v3_source_sha256:
            raise UgiSelectedGuidanceAdapterV3Error(
                "selected-v3 guidance-adapter source changed after construction"
            )

    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> Any:
        self._require_bound_artifacts_unchanged()
        receipt = super().initialize(
            programs,
            seed=seed,
            particle_seeds=particle_seeds,
            device=device,
        )
        self._require_bound_artifacts_unchanged()
        return receipt

    def _require_current_state(self, state: Any) -> SelectedGuidanceState:
        self._require_bound_artifacts_unchanged()
        return super()._require_current_state(state)


def build_selected_model_restartable_guidance_lane_v3(
    repository: Path,
    *,
    selected_lane: SelectedRestartableGeneratorLane | None = None,
) -> SelectedModelRestartableGuidanceLaneV3:
    """Build selected v2 with its pending token bound to repaired evidence."""

    root = Path(repository).resolve()
    binding = load_selected_v2_equivalence_binding(root)
    base_lane = selected_lane or build_selected_step2000_bond_stochastic_lane(root)
    bound_lane = _bind_equivalence_receipt(base_lane, binding)
    return SelectedModelRestartableGuidanceLaneV3(bound_lane, binding)


__all__ = [
    "EQUIVALENCE_RESULT_FILE_SHA256",
    "EQUIVALENCE_RESULT_LOGICAL_SHA256",
    "EQUIVALENCE_ROWS_FILE_SHA256",
    "EQUIVALENCE_ROWS_LOGICAL_SHA256",
    "SELECTED_GUIDANCE_ADAPTER_V3_SCHEMA_VERSION",
    "SelectedModelRestartableGuidanceLaneV3",
    "SelectedV2EquivalenceBinding",
    "UgiSelectedGuidanceAdapterV3Error",
    "build_selected_model_restartable_guidance_lane_v3",
    "load_selected_v2_equivalence_binding",
]
