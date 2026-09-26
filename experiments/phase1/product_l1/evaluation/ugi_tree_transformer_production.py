"""Post-selection Ugi production evaluation for fresh tree-Transformer seeds."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (
    sample_and_assess_checkpoint,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import iter_jsonl, read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_tree_transformer_production_evaluation_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_production_evaluation.v1"
SELECTED_ARM = "tree_relations_and_routing"
STATIC_INPUT_LABELS = (
    "program_draw",
    "closure_checkpoint",
    "prepared_cache",
    "qualified_reactions",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "role_morphology_policy",
)


class UgiTreeTransformerProductionEvaluationError(ValueError):
    """A production seed or component-family evaluation contract changed."""


def _runtime(config: Mapping[str, Any], *, profile: str) -> Mapping[str, Any]:
    required = {
        "schema_version",
        "status",
        "selected_model",
        "inputs",
        "profiles",
        "seed_derivation",
        "policy",
        "nonclaims",
    }
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != required:
        raise UgiTreeTransformerProductionEvaluationError(
            "production-evaluation config fields changed"
        )
    if config.get("status") != "frozen_before_fresh_production_seeds":
        raise UgiTreeTransformerProductionEvaluationError(
            "production-evaluation config is not frozen"
        )
    selected = config.get("selected_model")
    if selected != {
        "arm_id": SELECTED_ARM,
        "checkpoint_step": 2700,
        "candidate_id": f"{SELECTED_ARM}:step_2700",
    }:
        raise UgiTreeTransformerProductionEvaluationError("selected model changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != set(STATIC_INPUT_LABELS):
        raise UgiTreeTransformerProductionEvaluationError("evaluation inputs changed")
    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise UgiTreeTransformerProductionEvaluationError(
            f"production evaluation has no {profile!r} profile"
        )
    runtime = profiles[profile]
    expected_runtime = {
        "program_count",
        "sample_steps",
        "batch_size",
        "terminal_decoder_mode",
        "terminal_temperature",
        "maximum_adjacent_branch_runs",
    }
    if set(runtime) != expected_runtime:
        raise UgiTreeTransformerProductionEvaluationError(
            f"{profile} production-evaluation runtime changed"
        )
    if (
        int(runtime["program_count"]) != (3072 if profile == "full" else 4)
        or int(runtime["sample_steps"]) != 8
        or runtime["terminal_decoder_mode"] != "stochastic"
        or float(runtime["terminal_temperature"]) != 1.0
        or runtime["maximum_adjacent_branch_runs"] != [2, 1, 1]
    ):
        raise UgiTreeTransformerProductionEvaluationError(
            f"{profile} sampling contract changed"
        )
    if config.get("seed_derivation") != {
        "flow_seed_offset": 100000,
        "terminal_decoder_seed_offset": 200000,
        "derived_from_training_seed": True,
    }:
        raise UgiTreeTransformerProductionEvaluationError("sampling seed derivation changed")
    if config.get("policy") != {
        "same_ordered_programs_across_training_seeds": True,
        "component_family_stress_draw_previously_used": True,
        "heldout_rows_descriptive_only": True,
        "heldout_selects_architecture_checkpoint_or_threshold": False,
        "component_identifiers_exposed_to_model": False,
        "repairs_or_retries": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
        "reductive_amination_substructure_rate_reported": False,
    }:
        raise UgiTreeTransformerProductionEvaluationError("evaluation policy changed")
    return runtime


def _validate_training_config(
    training: Mapping[str, Any], *, profile: str, training_seed: int
) -> int:
    if training.get("schema_version") != "phase1_ugi_joint_sparse_training_config.v1":
        raise UgiTreeTransformerProductionEvaluationError("effective training schema changed")
    arm = training.get("experiment_arm")
    runtime = training.get(profile)
    if (
        not isinstance(arm, Mapping)
        or arm.get("arm_id") != SELECTED_ARM
        or arm.get("candidate_id") != f"{SELECTED_ARM}:step_2700"
        or int(arm.get("production_seed", -1)) != training_seed
        or int(training.get("seed", -1)) != training_seed
        or not isinstance(runtime, Mapping)
    ):
        raise UgiTreeTransformerProductionEvaluationError(
            "effective training config does not identify this production seed"
        )
    checkpoint_step = int(runtime.get("steps", -1))
    expected_step = 2700 if profile == "full" else 2
    if (
        checkpoint_step != expected_step
        or runtime.get("checkpoint_steps") != [expected_step]
        or runtime.get("device") != "cuda"
        or training.get("training_partition", {}).get("training_folds") != ["train"]
        or training.get("promotion_contract", {}).get(
            "heldout_selects_architecture_or_checkpoint"
        )
        is not False
        or training.get("model", {}).get("backbone") != "ugi_tree_program_transformer"
        or training.get("model", {}).get("tree_relation_attention") is not True
        or training.get("model", {}).get("role_routed_program_attention") is not True
        or int(training.get("model", {}).get("role_adapter_dim", -1)) != 0
    ):
        raise UgiTreeTransformerProductionEvaluationError(
            "effective production architecture or fixed step changed"
        )
    return checkpoint_step


def _validate_program_rows(
    program_draw: Mapping[str, Any],
    native_rows: Sequence[Mapping[str, Any]],
    *,
    expected: int,
) -> None:
    programs = program_draw.get("samples")
    if not isinstance(programs, list) or len(programs) < expected or len(native_rows) != expected:
        raise UgiTreeTransformerProductionEvaluationError(
            "component-family program denominator changed"
        )
    for index, (program, native) in enumerate(
        zip(programs[:expected], native_rows, strict=True)
    ):
        if not isinstance(program, Mapping) or not isinstance(native, Mapping):
            raise UgiTreeTransformerProductionEvaluationError(
                f"component-family program {index} is malformed"
            )
        for field in ("product_id", "source_stratum", "held_role_class"):
            if native.get(field) != program.get(field):
                raise UgiTreeTransformerProductionEvaluationError(
                    f"sample {index} changed program identity field {field!r}"
                )


def summarize_component_family_strata(
    native_rows: Sequence[Mapping[str, Any]], assessed_attempts_path: Path
) -> dict[str, dict[str, Any]]:
    """Summarize descriptive strata without treating conditioning as component recovery."""

    assessed = list(iter_jsonl(assessed_attempts_path))
    header = assessed.pop(0) if assessed else None
    if header != {
        "schema_version": "forge.common_ugi_assessed_attempts.v1",
        "rows": len(assessed),
    } or len(assessed) != len(native_rows):
        raise UgiTreeTransformerProductionEvaluationError(
            "assessed-attempt ledger denominator changed"
        )
    grouped: dict[str, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for index, (native, common) in enumerate(zip(native_rows, assessed, strict=True)):
        if int(common.get("attempt_index", -1)) != index:
            raise UgiTreeTransformerProductionEvaluationError(
                "assessed-attempt order changed"
            )
        source = str(native.get("source_stratum", ""))
        held_role = str(native.get("held_role_class", ""))
        if not source or not held_role:
            raise UgiTreeTransformerProductionEvaluationError(
                f"sample {index} lacks descriptive stratum metadata"
            )
        grouped[f"source:{source}"].append((native, common))
        grouped[f"held_role:{held_role}"].append((native, common))

    summaries: dict[str, dict[str, Any]] = {}
    for label, rows in sorted(grouped.items()):
        counts: Counter[str] = Counter(attempts=len(rows))
        exact_smiles: set[str] = set()
        open_smiles: set[str] = set()
        for _, common in rows:
            valid = common.get("valid") is True
            exact = common.get("exact_l1_program") is True
            open_ended = common.get("method_visible_open_ended_exact_l1") is True
            held_component = common.get("held_component_exact_l1") is True
            counts["valid"] += int(valid)
            counts["exact_l1"] += int(exact)
            counts["open_ended_exact_l1"] += int(exact and open_ended)
            counts["held_component_exact_l1"] += int(exact and held_component)
            smiles = common.get("canonical_smiles")
            if exact and isinstance(smiles, str) and smiles:
                exact_smiles.add(smiles)
                if open_ended:
                    open_smiles.add(smiles)
        attempts = int(counts["attempts"])
        summaries[label] = {
            "attempts": attempts,
            "valid_fraction_per_attempt": counts["valid"] / attempts,
            "exact_l1_yield_per_attempt": counts["exact_l1"] / attempts,
            "open_ended_exact_l1_yield_per_attempt": (
                counts["open_ended_exact_l1"] / attempts
            ),
            "held_component_exact_l1_products_per_1000_attempts": (
                1000.0 * counts["held_component_exact_l1"] / attempts
            ),
            "unique_exact_l1_products": len(exact_smiles),
            "unique_open_ended_exact_l1_products": len(open_smiles),
        }
    return summaries


def run_tree_transformer_production_evaluation(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    checkpoint_path: Path,
    effective_training_config_path: Path,
    training_result_path: Path,
    training_seed: int,
    profile: str,
    resume: bool,
    device: str,
) -> dict[str, Any]:
    """Evaluate one fresh production seed under the shared method-blind assessors."""

    config = read_json_object(
        config_path,
        error=UgiTreeTransformerProductionEvaluationError,
        label="tree-Transformer production evaluation config",
    )
    runtime = _runtime(config, profile=profile)
    training = read_json_object(
        effective_training_config_path,
        error=UgiTreeTransformerProductionEvaluationError,
        label="effective production training config",
    )
    checkpoint_step = _validate_training_config(
        training, profile=profile, training_seed=training_seed
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiTreeTransformerProductionEvaluationError(
            f"production evaluation output is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(config["inputs"][label], repo, label=label)
        for label in STATIC_INPUT_LABELS
    }
    program_draw = read_json_object(
        inputs["program_draw"],
        error=UgiTreeTransformerProductionEvaluationError,
        label="component-family program draw",
    )
    program_count = int(runtime["program_count"])
    runtime_with_seeds = {
        **dict(runtime),
        "flow_seed": training_seed + int(config["seed_derivation"]["flow_seed_offset"]),
        "terminal_decoder_seed": training_seed
        + int(config["seed_derivation"]["terminal_decoder_seed_offset"]),
    }
    method_id = f"forge_tree_relational_production_seed_{training_seed}"
    detail_dir = output_dir / "details"
    assessment = sample_and_assess_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_step=checkpoint_step,
        arm_id=SELECTED_ARM,
        method_id=method_id,
        assessment_seed_label=training_seed,
        runtime=runtime_with_seeds,
        inputs=inputs,
        repo=repo,
        output_dir=detail_dir,
        device=device,
        heldout_rows_used=True,
    )
    sampling = read_json_object(
        detail_dir / "sampling" / "result.json",
        error=UgiTreeTransformerProductionEvaluationError,
        label="production sampling result",
    )
    native_rows = sampling.get("samples")
    if not isinstance(native_rows, list):
        raise UgiTreeTransformerProductionEvaluationError(
            "production sampling result has no attempts"
        )
    _validate_program_rows(program_draw, native_rows, expected=program_count)
    strata = summarize_component_family_strata(
        native_rows,
        detail_dir / "assessment" / "common" / "assessed_attempts.jsonl.gz",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "arm_id": SELECTED_ARM,
        "candidate_id": f"{SELECTED_ARM}:step_2700",
        "profile": profile,
        "training_seed": training_seed,
        "checkpoint_step": checkpoint_step,
        "programs": program_count,
        "sampling_device": device,
        "method_id": method_id,
        "metrics": assessment["assessment"]["metrics"],
        "strata": strata,
        "inputs": {
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
            "checkpoint": artifact_record(checkpoint_path),
            "effective_training_config": artifact_record(effective_training_config_path),
            "training_result": artifact_record(training_result_path),
        },
        "paired_program_order_across_seeds": True,
        "heldout_rows_used": True,
        "heldout_selects_architecture_checkpoint_or_threshold": False,
        "candidate_selection": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SELECTED_ARM",
    "STATIC_INPUT_LABELS",
    "UgiTreeTransformerProductionEvaluationError",
    "run_tree_transformer_production_evaluation",
    "summarize_component_family_strata",
]
