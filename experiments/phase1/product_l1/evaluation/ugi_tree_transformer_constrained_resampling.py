"""Paired frozen-checkpoint Ugi resampling with role-local terminal constraints."""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_failure_attribution import (
    _COMMON_MEMBER,
    _LOCAL_MEMBER,
    _jsonl_member,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_transformer_assessment import (
    assess_native_ugi_method,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_tree_transformer_constrained_resampling_config.v1"
CONFIG_SCHEMA_V2 = "forge.ugi_tree_transformer_constrained_resampling_config.v2"
RESULT_SCHEMA = "forge.ugi_tree_transformer_constrained_resampling.v1"
PROGRAM_ID = "ugi_3cr_agile"
TRAINING_SEED = 20260905
INPUT_LABELS = (
    "checkpoint",
    "closure_checkpoint",
    "prepared_cache",
    "program_draw",
    "qualified_reactions",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "role_local_policy",
    "intervention_readiness",
)
V2_EXTRA_INPUT_LABELS = ("broad_scope_smoke", "original_seed0_evaluation_details")


class UgiTreeTransformerConstrainedResamplingError(ValueError):
    """The paired constrained-resampling contract changed or failed closed."""


def _runtime(config: Mapping[str, Any], profile: str) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "status",
        "training_seed",
        "inputs",
        "profiles",
        "seed_derivation",
        "gates",
        "policy",
        "nonclaims",
    }
    schema_version = config.get("schema_version")
    if schema_version not in {CONFIG_SCHEMA, CONFIG_SCHEMA_V2} or set(config) != required:
        raise UgiTreeTransformerConstrainedResamplingError(
            "constrained-resampling config fields changed"
        )
    if (
        config.get("status")
        not in {
            "frozen_after_intervention_readiness_before_seed0_resampling",
            "frozen_after_broad_scope_smoke_before_edge_scope_resampling",
        }
        or config.get("training_seed") != TRAINING_SEED
        or not isinstance(config.get("inputs"), Mapping)
        or set(config["inputs"])
        != set(INPUT_LABELS)
        | (set(V2_EXTRA_INPUT_LABELS) if schema_version == CONFIG_SCHEMA_V2 else set())
    ):
        raise UgiTreeTransformerConstrainedResamplingError(
            "constrained-resampling frozen contract changed"
        )
    profiles = config.get("profiles")
    runtime = profiles.get(profile) if isinstance(profiles, Mapping) else None
    expected_count = 32 if profile == "smoke" else 3072 if profile == "full" else None
    if (
        expected_count is None
        or not isinstance(runtime, Mapping)
        or set(runtime)
        != {
            "program_count",
            "sample_steps",
            "batch_size",
            "terminal_decoder_mode",
            "terminal_temperature",
            "maximum_adjacent_branch_runs",
        }
        or runtime.get("program_count") != expected_count
        or runtime.get("sample_steps") != 8
        or runtime.get("batch_size") != 16
        or runtime.get("terminal_decoder_mode") != "stochastic"
        or runtime.get("terminal_temperature") != 1.0
        or runtime.get("maximum_adjacent_branch_runs") != [2, 1, 1]
    ):
        raise UgiTreeTransformerConstrainedResamplingError(
            f"{profile!r} constrained-resampling runtime changed"
        )
    if config.get("seed_derivation") != {
        "flow_seed_offset": 100000,
        "terminal_decoder_seed_offset": 200000,
    }:
        raise UgiTreeTransformerConstrainedResamplingError("paired seeds changed")
    expected_gates = {
        "all_attempts_retained": True,
        "all_terminal_failures_typed": True,
        "all_molecule_failures_typed": True,
        "maximum_local_unsupported_exact_l1_products": 0,
        "maximum_nitrogen_oxygen_bond_product_fraction_per_attempt": 0.0,
        "maximum_oxygen_oxygen_bond_product_fraction_per_attempt": 0.0,
    }
    if schema_version == CONFIG_SCHEMA_V2:
        expected_gates.update(
            {
                "minimum_paired_original_local_supported_exact_l1_retained_fraction": 1.0,
                "minimum_paired_original_exact_l1_retained_fraction": 0.9,
            }
        )
    else:
        expected_gates["maximum_small_oxygen_ring_product_fraction_per_attempt"] = 0.0
    if config.get("gates") != expected_gates:
        raise UgiTreeTransformerConstrainedResamplingError("resampling gates changed")
    expected_policy = {
        "frozen_checkpoint_only": True,
        "retraining": False,
        "same_ordered_programs_as_negative_production_seed0": True,
        "same_flow_and_terminal_seeds_as_negative_production_seed0": True,
        "role_local_policy_is_training_fold_only": True,
        "component_identities_exposed_to_decoder": False,
        "repairs_or_retries": False,
        "production_result_replaced": False,
        "candidate_selection": False,
        "route_calls": 0,
        "oracle_calls": 0,
    }
    if schema_version == CONFIG_SCHEMA_V2:
        expected_policy["local_chemistry_constraint_scope"] = "role_edges_only"
    if config.get("policy") != expected_policy:
        raise UgiTreeTransformerConstrainedResamplingError("resampling policy changed")
    return runtime


def _paired_original_counts(path: Path, *, attempts: int) -> dict[str, int]:
    with tarfile.open(path, "r") as archive:
        common = _jsonl_member(
            archive,
            _COMMON_MEMBER,
            schema_version="forge.common_ugi_assessed_attempts.v1",
            expected_rows=3072,
        )[:attempts]
        local = _jsonl_member(
            archive,
            _LOCAL_MEMBER,
            schema_version="forge.common_local_chemistry_attempt.v1",
            expected_rows=3072,
        )[:attempts]
    return {
        "attempts": attempts,
        "valid": sum(row.get("valid") is True for row in common),
        "exact_l1": sum(row.get("exact_l1_program") is True for row in common),
        "local_supported_exact_l1": sum(
            row.get("local_support_qualified_exact_l1") is True for row in local
        ),
    }


def run_tree_transformer_constrained_resampling(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
) -> dict[str, Any]:
    """Run one no-repair, no-retry paired constrained frozen-checkpoint sample."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise UgiTreeTransformerConstrainedResamplingError(
            f"constrained-resampling output already exists: {output_dir}"
        )
    config = read_json_object(
        config_path,
        error=UgiTreeTransformerConstrainedResamplingError,
        label="constrained-resampling config",
    )
    runtime = _runtime(config, profile)
    input_labels = (
        *INPUT_LABELS,
        *(V2_EXTRA_INPUT_LABELS if config["schema_version"] == CONFIG_SCHEMA_V2 else ()),
    )
    inputs = {
        label: resolve_pin(config["inputs"][label], repo, label=label)
        for label in input_labels
    }
    readiness = read_json_object(
        inputs["intervention_readiness"],
        error=UgiTreeTransformerConstrainedResamplingError,
        label="intervention readiness",
    )
    if readiness.get("status") != "ready_for_new_constrained_sampling_preflight":
        raise UgiTreeTransformerConstrainedResamplingError(
            "intervention readiness has not closed"
        )
    flow_seed = TRAINING_SEED + int(config["seed_derivation"]["flow_seed_offset"])
    terminal_seed = TRAINING_SEED + int(
        config["seed_derivation"]["terminal_decoder_seed_offset"]
    )
    constraint_scope = str(
        config["policy"].get(
            "local_chemistry_constraint_scope", "role_edges_cycles_bounds"
        )
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        # Keep the evaluation module renderer-free at import time.  The sampler's
        # optional rendering dependency is needed only while an experiment runs.
        from experiments.phase1.product_l1.sampling.ugi_joint_end_to_end_sampling import (
            sample_ugi_joint_end_to_end,
        )

        sampling_dir = partial / "sampling"
        sampling = sample_ugi_joint_end_to_end(
            repo,
            sampling_dir,
            joint_checkpoint_path=inputs["checkpoint"],
            closure_checkpoint_path=inputs["closure_checkpoint"],
            matched_staged_result_path=inputs["program_draw"],
            prepared_cache_path=inputs["prepared_cache"],
            sample_steps=int(runtime["sample_steps"]),
            batch_size=int(runtime["batch_size"]),
            seed=flow_seed,
            overwrite=False,
            maximum_adjacent_branch_runs=tuple(runtime["maximum_adjacent_branch_runs"]),
            qualified_reactions_path=inputs["qualified_reactions"],
            evaluate_exact_l1_terminal_admission=True,
            terminal_decoder_mode=str(runtime["terminal_decoder_mode"]),
            terminal_decoder_seed=terminal_seed,
            terminal_temperature=float(runtime["terminal_temperature"]),
            local_chemistry_policy_path=inputs["role_local_policy"],
            local_chemistry_program_id=PROGRAM_ID,
            local_chemistry_constraint_scope=constraint_scope,
            program_offset=0,
            program_limit=int(runtime["program_count"]),
            reference_comparison_mode="deferred",
            render=False,
            record_timing=False,
            device=device,
        )
        rows = sampling.get("samples")
        if not isinstance(rows, list) or len(rows) != int(runtime["program_count"]):
            raise UgiTreeTransformerConstrainedResamplingError(
                "constrained sampler changed the fixed attempt denominator"
            )
        assessment_dir = partial / "assessment"
        assessment = assess_native_ugi_method(
            rows,
            method_id=f"forge_tree_relational_role_local_seed_{TRAINING_SEED}",
            seed_label=TRAINING_SEED,
            repo=repo,
            output_dir=assessment_dir,
            common_ugi_assessment_config=inputs["common_ugi_assessment_config"],
            lipid_realism_config=inputs["lipid_realism_config"],
            local_chemistry_config=inputs["local_chemistry_config"],
            role_morphology_policy=inputs["role_local_policy"],
        )
        metrics = assessment["metrics"]
        terminal_failures = [
            row for row in rows if row.get("failure_type") == "TerminalSupportFailure"
        ]
        molecule_failures = [
            row for row in rows if row.get("failure_type") == "MoleculeSanitizationFailure"
        ]
        local_unsupported = int(round(
            int(runtime["program_count"])
            * (
                float(metrics["exact_l1_yield_per_attempt"])
                - float(metrics["local_support_qualified_exact_l1_yield_per_attempt"])
            )
        ))
        paired_original = (
            _paired_original_counts(
                inputs["original_seed0_evaluation_details"],
                attempts=int(runtime["program_count"]),
            )
            if config["schema_version"] == CONFIG_SCHEMA_V2
            else None
        )
        exact_l1_count = int(
            round(int(runtime["program_count"]) * float(metrics["exact_l1_yield_per_attempt"]))
        )
        local_supported_count = exact_l1_count - local_unsupported
        gates = {
            "all_attempts_retained": len(rows) == int(runtime["program_count"]),
            "all_terminal_failures_typed": all(
                isinstance(row.get("terminal_failure_detail"), Mapping)
                for row in terminal_failures
            ),
            "all_molecule_failures_typed": all(
                isinstance(row.get("molecule_failure_detail"), Mapping)
                for row in molecule_failures
            ),
            "maximum_local_unsupported_exact_l1_products": local_unsupported == 0,
            "maximum_nitrogen_oxygen_bond_product_fraction_per_attempt": (
                float(metrics["nitrogen_oxygen_bond_product_fraction_per_attempt"]) == 0.0
            ),
            "maximum_oxygen_oxygen_bond_product_fraction_per_attempt": (
                float(metrics["oxygen_oxygen_bond_product_fraction_per_attempt"]) == 0.0
            ),
        }
        if paired_original is None:
            gates["maximum_small_oxygen_ring_product_fraction_per_attempt"] = (
                float(metrics["small_oxygen_ring_product_fraction_per_attempt"]) == 0.0
            )
        else:
            gates[
                "minimum_paired_original_local_supported_exact_l1_retained_fraction"
            ] = local_supported_count >= paired_original["local_supported_exact_l1"]
            gates["minimum_paired_original_exact_l1_retained_fraction"] = (
                exact_l1_count
                >= 0.9 * paired_original["exact_l1"]
            )
        passed = all(gates.values())
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete",
            "decision": (
                "pass_constrained_sampling_preflight"
                if passed and profile == "smoke"
                else "complete_seed0_constrained_diagnostic"
                if passed
                else "negative_constrained_sampling_result"
            ),
            "profile": profile,
            "device": device,
            "training_seed": TRAINING_SEED,
            "attempts": len(rows),
            "flow_seed": flow_seed,
            "terminal_decoder_seed": terminal_seed,
            "local_chemistry_constraint_scope": constraint_scope,
            "metrics": metrics,
            "failure_counts": {
                "terminal_support": len(terminal_failures),
                "molecule_sanitization": len(molecule_failures),
                "local_unsupported_exact_l1": local_unsupported,
            },
            "paired_original": paired_original,
            "paired_counts": {
                "exact_l1": exact_l1_count,
                "local_supported_exact_l1": local_supported_count,
            },
            "gates": gates,
            "all_gates_pass": passed,
            "sampling": artifact_record(sampling_dir / "result.json"),
            "assessment": assessment,
            "inputs": {
                "config": pin_record(config_path, repo),
                **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
            },
            "implementation": {
                "runner": pin_record(Path(__file__), repo),
                "terminal_decoder": pin_record(
                    repo / "forge/model/ugi_chemistry_flow.py", repo
                ),
                "sampler": pin_record(
                    repo
                    / "experiments/phase1/product_l1/sampling/ugi_joint_end_to_end_sampling.py",
                    repo,
                ),
            },
            "retraining": False,
            "repairs_or_retries": False,
            "production_result_replaced": False,
            "candidate_selection": False,
            "calls": {"route": 0, "oracle": 0},
            "nonclaims": list(config["nonclaims"]),
        }
        write_json(partial / "result.json", result)
        partial.rename(output_dir)
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_V2",
    "RESULT_SCHEMA",
    "UgiTreeTransformerConstrainedResamplingError",
    "run_tree_transformer_constrained_resampling",
]
