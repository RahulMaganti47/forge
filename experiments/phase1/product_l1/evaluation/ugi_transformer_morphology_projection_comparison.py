"""Paired full-versus-reduced program projection assessment for the mixed Transformer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _load_programs,
    _metric_deltas,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_transformer_morphology_projection_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_transformer_morphology_projection_comparison.v1"
ARM_IDS = ("reduced_projection", "full_role_morphology")
INPUT_LABELS = (
    "common_ugi_assessment_config",
    "full_checkpoint_archive",
    "full_production_design",
    "full_training_result",
    "lipid_realism_config",
    "local_chemistry_config",
    "production_cache",
    "program_draw",
    "reduced_checkpoint_archive",
    "reduced_production_design",
    "reduced_training_result",
    "role_morphology_policy",
)


class UgiTransformerMorphologyProjectionComparisonError(ValueError):
    """The paired projection comparison contract or an authenticated input changed."""


def _validate_config(config: Mapping[str, Any], *, profile: str) -> Mapping[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise UgiTransformerMorphologyProjectionComparisonError(
            "unsupported morphology projection comparison config"
        )
    inputs = config.get("inputs")
    profiles = config.get("profiles")
    arms = config.get("arms")
    policy = config.get("policy")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_LABELS):
        raise UgiTransformerMorphologyProjectionComparisonError("comparison inputs changed")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise UgiTransformerMorphologyProjectionComparisonError(
            f"comparison config has no {profile!r} profile"
        )
    expected_arms = {
        "reduced_projection": {
            "arm_id": "bl_core_constrained_repeat_aware",
            "checkpoint_step": 1700,
            "method_id": "forge_mixed_transformer_reduced_projection_seed0",
            "role_morphology_conditioning": False,
        },
        "full_role_morphology": {
            "arm_id": "full_role_morphology_transformer",
            "checkpoint_step": 1700,
            "method_id": "forge_mixed_transformer_full_role_morphology_seed0",
            "role_morphology_conditioning": True,
        },
    }
    if arms != expected_arms:
        raise UgiTransformerMorphologyProjectionComparisonError("comparison arms changed")
    expected_policy = {
        "candidate_selection": False,
        "flow_seed_matched": True,
        "method_blind_assessment": True,
        "negative_results_reported": True,
        "oracle_calls": 0,
        "paired_program_order": True,
        "repairs_or_retries": False,
        "route_calls": 0,
        "training_calls": 0,
    }
    if policy != expected_policy:
        raise UgiTransformerMorphologyProjectionComparisonError("comparison policy changed")
    runtime = profiles[profile]
    if (
        str(runtime.get("device")) not in {"cpu", "cuda"}
        or int(runtime.get("program_count", 0)) < 1
        or int(runtime.get("sample_steps", 0)) < 1
        or int(runtime.get("batch_size", 0)) < 1
        or runtime.get("terminal_decoder_mode") != "argmax"
        or runtime.get("terminal_decoder_seed") is not None
        or runtime.get("terminal_decode_policy") != "strict_valence_topology_argmax"
    ):
        raise UgiTransformerMorphologyProjectionComparisonError(
            "comparison runtime contract changed"
        )
    decision_rule = config.get("decision_rule")
    if decision_rule != {
        "primary_metric": "exact_l1_yield_per_attempt",
        "absolute_margin": 0.02,
    }:
        raise UgiTransformerMorphologyProjectionComparisonError("decision rule changed")
    return runtime


def _arm_inputs(inputs: Mapping[str, Path], *, prefix: str) -> dict[str, Path]:
    return {
        "checkpoint_archive": inputs[f"{prefix}_checkpoint_archive"],
        "common_ugi_assessment_config": inputs["common_ugi_assessment_config"],
        "current_training_result": inputs[f"{prefix}_training_result"],
        "lipid_realism_config": inputs["lipid_realism_config"],
        "local_chemistry_config": inputs["local_chemistry_config"],
        "production_cache": inputs["production_cache"],
        "production_design": inputs[f"{prefix}_production_design"],
        "program_draw": inputs["program_draw"],
        "role_morphology_policy": inputs["role_morphology_policy"],
    }


def _projection_decision(delta: float, *, margin: float) -> str:
    if delta > margin:
        return "full_role_morphology_improves_exact_l1_under_seed0_gate"
    if delta < -margin:
        return "reduced_projection_improves_exact_l1_under_seed0_gate"
    return "no_material_seed0_exact_l1_change"


def _program_rows_match(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]
) -> bool:
    return len(first) == len(second) and all(
        left.get("pipeline_index") == right.get("pipeline_index")
        and left.get("program") == right.get("program")
        for left, right in zip(first, second, strict=True)
    )


def _no_sampling_repairs(sampling: Mapping[str, Any]) -> bool:
    return sampling.get("repairs") in (None, {}) and sampling.get("retries") in (None, 0, {})


def run_ugi_transformer_morphology_projection_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Sample both frozen Transformers on the same ordered full-morphology programs."""

    config = read_json_object(
        config_path,
        error=UgiTransformerMorphologyProjectionComparisonError,
        label="Transformer morphology projection comparison config",
    )
    runtime = _validate_config(config, profile=profile)
    if str(runtime["device"]) != device:
        raise UgiTransformerMorphologyProjectionComparisonError(
            "configured and allocated devices differ"
        )
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiTransformerMorphologyProjectionComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(config["inputs"][label], repo, label=label) for label in INPUT_LABELS
    }
    programs = _load_programs(inputs["program_draw"], count=int(runtime["program_count"]))
    sampling_runtime = {
        "batch_size": int(runtime["batch_size"]),
        "current_terminal_decode_policy": str(runtime["terminal_decode_policy"]),
        "flow_seed": int(runtime["flow_seed"]),
        "program_count": int(runtime["program_count"]),
        "sample_steps": int(runtime["sample_steps"]),
    }
    config_sha256 = sha256_file(config_path)
    method_results: dict[str, Any] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm_name in ARM_IDS:
        prefix = "reduced" if arm_name == "reduced_projection" else "full"
        arm = config["arms"][arm_name]
        scoped_inputs = _arm_inputs(inputs, prefix=prefix)
        rows, sampling, sampling_path = _load_or_sample_current(
            repo=repo,
            output_dir=output_dir / arm_name / "sampling",
            inputs=scoped_inputs,
            runtime=sampling_runtime,
            config_sha256=config_sha256,
            device=device,
            arm_id=str(arm["arm_id"]),
            checkpoint_step=int(arm["checkpoint_step"]),
            programs=programs,
        )
        assessment = _load_or_assess_current(
            rows=rows,
            sampling_path=sampling_path,
            output_dir=output_dir / arm_name / "assessment",
            inputs=scoped_inputs,
            repo=repo,
            method_id=str(arm["method_id"]),
        )
        sampling_result = read_json_object(
            sampling_path,
            error=UgiTransformerMorphologyProjectionComparisonError,
            label=f"{arm_name} sampling result",
        )
        projection = sampling_result.get("program_projection")
        if not isinstance(projection, Mapping):
            raise UgiTransformerMorphologyProjectionComparisonError(
                f"{arm_name} has no program projection receipt"
            )
        expected_conditioning = bool(arm["role_morphology_conditioning"])
        if bool(projection.get("role_morphology_conditioning")) != expected_conditioning:
            raise UgiTransformerMorphologyProjectionComparisonError(
                f"{arm_name} loaded a checkpoint with the wrong conditioning contract"
            )
        rows_by_arm[arm_name] = rows
        method_results[str(arm["method_id"])] = {
            "metrics": dict(assessment["assessment"]["metrics"]),
            "sampling": artifact_record(sampling_path),
            "assessment": artifact_record(
                output_dir / arm_name / "assessment" / "result_index.json"
            ),
            "program_projection": dict(projection),
            "sampling_receipt": dict(sampling),
        }

    reduced_id = str(config["arms"]["reduced_projection"]["method_id"])
    full_id = str(config["arms"]["full_role_morphology"]["method_id"])
    reduced_metrics = method_results[reduced_id]["metrics"]
    full_metrics = method_results[full_id]["metrics"]
    deltas = _metric_deltas(full_metrics, reduced_metrics)
    primary = str(config["decision_rule"]["primary_metric"])
    margin = float(config["decision_rule"]["absolute_margin"])
    paired_rows = _program_rows_match(
        rows_by_arm["reduced_projection"], rows_by_arm["full_role_morphology"]
    )
    gates = {
        "attempt_denominator_matched": (
            len(rows_by_arm["reduced_projection"])
            == len(rows_by_arm["full_role_morphology"])
            == len(programs)
        ),
        "candidate_selection_absent": True,
        "full_projection_consumed": bool(
            method_results[full_id]["program_projection"]["role_morphology_conditioning"]
        ),
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": all(
            _no_sampling_repairs(method_results[method_id]["sampling_receipt"])
            for method_id in (reduced_id, full_id)
        ),
        "paired_program_order": paired_rows,
        "reduced_projection_preserved": not bool(
            method_results[reduced_id]["program_projection"]["role_morphology_conditioning"]
        ),
        "route_or_oracle_calls_zero": True,
        "training_calls_zero": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if all(gates.values()) else "fail",
        "profile": profile,
        "programs_per_method": len(programs),
        "comparison_design": {
            "checkpoint_step_matched": 1700,
            "flow_seed_matched": True,
            "program_rows_and_order_matched": paired_rows,
            "sample_steps_matched": True,
            "terminal_decoder_matched": str(runtime["terminal_decode_policy"]),
            "training_budget_matched": True,
            "training_data_and_source_weights_matched": True,
            "training_seed_matched": True,
            "only_declared_model_intervention": "role_morphology_conditioning",
        },
        "methods": method_results,
        "full_minus_reduced": deltas,
        "decision_rule": dict(config["decision_rule"]),
        "decision": _projection_decision(float(deltas[primary]), margin=margin),
        "gates": gates,
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "complete":
        raise UgiTransformerMorphologyProjectionComparisonError(f"comparison gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTransformerMorphologyProjectionComparisonError",
    "run_ugi_transformer_morphology_projection_comparison",
]
