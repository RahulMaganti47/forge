"""Program-matched Ugi assessment of an exposure-matched specialist and frozen v0."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (
    sample_and_assess_checkpoint,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
    _load_or_assess_current,
    _load_or_sample_current,
    _load_programs,
    _metric_deltas,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.model.ugi_transformer_topology import UgiTransformerTopologyPolicy

CONFIG_SCHEMA = "forge.reaction_specialist_ugi_v0_comparison_config.v1"
RESULT_SCHEMA = "forge.reaction_specialist_ugi_v0_comparison_result.v1"
TOPOLOGY_CONFIG_SCHEMA = "forge.reaction_topology_specialist_ugi_v0_comparison_config.v2"
TOPOLOGY_RESULT_SCHEMA = "forge.reaction_topology_specialist_ugi_v0_comparison_result.v2"
INPUT_LABELS = (
    "base_checkpoint_archive",
    "base_training_result",
    "base_design",
    "closure_checkpoint",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "prepared_cache",
    "production_cache",
    "program_draw",
    "qualified_reactions",
    "role_morphology_policy",
    "v0_checkpoint",
)
RECOVERY_INPUT_LABELS = ("specialist_checkpoint", "specialist_result")
TOPOLOGY_INPUT_LABELS = ("topology_closure_config", "topology_morphology_config")


class ReactionSpecialistUgiV0ComparisonError(ValueError):
    """The paired specialist-versus-v0 assessment contract changed."""


def _validate(config: Mapping[str, Any], *, profile: str, device: str) -> Mapping[str, Any]:
    schema_version = config.get("schema_version")
    if schema_version not in {CONFIG_SCHEMA, TOPOLOGY_CONFIG_SCHEMA}:
        raise ReactionSpecialistUgiV0ComparisonError("unsupported specialist comparison config")
    topology_specialist = schema_version == TOPOLOGY_CONFIG_SCHEMA
    inputs = config.get("inputs")
    profiles = config.get("profiles")
    required_inputs = (
        (*INPUT_LABELS, *TOPOLOGY_INPUT_LABELS)
        if topology_specialist
        else INPUT_LABELS
    )
    admitted_input_sets = (set(required_inputs), set((*required_inputs, *RECOVERY_INPUT_LABELS)))
    if not isinstance(inputs, Mapping) or set(inputs) not in admitted_input_sets:
        raise ReactionSpecialistUgiV0ComparisonError("specialist comparison inputs changed")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise ReactionSpecialistUgiV0ComparisonError("specialist comparison profile is missing")
    runtime = profiles[profile]
    if (
        str(runtime.get("device")) != device
        or int(runtime.get("program_count", 0)) < 1
        or int(runtime.get("sample_steps", 0)) < 2
        or int(runtime.get("batch_size", 0)) < 1
        or runtime.get("terminal_decoder_mode")
        != (
            "conditional_topology_argmax_chemistry" if topology_specialist else "argmax"
        )
        or (
            int(runtime.get("terminal_decoder_seed", -1))
            != int(runtime.get("flow_seed", -2)) + 1
            if topology_specialist
            else runtime.get("terminal_decoder_seed") is not None
        )
        or runtime.get("current_terminal_decode_policy")
        != (
            "strict_ugi_program_coupled_conditional"
            if topology_specialist
            else "strict_program_topology_argmax"
        )
        or runtime.get("maximum_adjacent_branch_runs") != [2, 1, 1]
    ):
        raise ReactionSpecialistUgiV0ComparisonError("specialist comparison runtime changed")
    if config.get("policy") != {
        "candidate_selection": False,
        "method_blind_assessment": True,
        "negative_results_reported": True,
        "oracle_calls": 0,
        "paired_program_order": True,
        "repairs_or_retries": False,
        "route_calls": 0,
        "training_calls": 0,
    }:
        raise ReactionSpecialistUgiV0ComparisonError("specialist comparison policy changed")
    return runtime


def _load_topology_policy(inputs: Mapping[str, Path]) -> UgiTransformerTopologyPolicy:
    closure = read_json_object(
        inputs["topology_closure_config"],
        error=ReactionSpecialistUgiV0ComparisonError,
        label="Ugi closure topology support",
    )
    morphology = read_json_object(
        inputs["topology_morphology_config"],
        error=ReactionSpecialistUgiV0ComparisonError,
        label="Ugi morphology topology support",
    )
    try:
        return UgiTransformerTopologyPolicy.from_support_documents(closure, morphology)
    except ValueError as error:
        raise ReactionSpecialistUgiV0ComparisonError(
            "pinned Ugi topology policy is malformed"
        ) from error


def run_reaction_specialist_ugi_v0_comparison(
    config_path: Path,
    specialist_checkpoint_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Evaluate both methods on one ordered morphology draw without selection or retries."""

    config = read_json_object(
        config_path,
        error=ReactionSpecialistUgiV0ComparisonError,
        label="specialist Ugi-v0 comparison config",
    )
    runtime = _validate(config, profile=profile, device=device)
    topology_specialist = config["schema_version"] == TOPOLOGY_CONFIG_SCHEMA
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise ReactionSpecialistUgiV0ComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(config["inputs"].items())
    }
    if set(RECOVERY_INPUT_LABELS).issubset(inputs):
        if inputs["specialist_checkpoint"] != specialist_checkpoint_path.resolve():
            raise ReactionSpecialistUgiV0ComparisonError(
                "recovered specialist checkpoint differs from the runtime checkpoint"
            )
        expected_result = specialist_checkpoint_path.with_name("result.json").resolve()
        if inputs["specialist_result"] != expected_result:
            raise ReactionSpecialistUgiV0ComparisonError(
                "recovered specialist result is not adjacent to its checkpoint"
            )
    specialist = read_json_object(
        specialist_checkpoint_path.with_name("result.json"),
        error=ReactionSpecialistUgiV0ComparisonError,
        label="Ugi specialist result",
    )
    if (
        specialist.get("schema_version")
        != (
            "forge.reaction_program_topology_specialization_result.v2"
            if topology_specialist
            else "forge.reaction_program_specialization_result.v1"
        )
        or specialist.get("status") != "pass"
        or specialist.get("target_program") != "ugi_3cr_agile"
        or int(specialist.get("observed", {}).get("cumulative_examples", -1))
        != int(config["matched_ugi_exposure"])
        or specialist.get("checkpoint", {}).get("sha256")
        != sha256_file(specialist_checkpoint_path)
    ):
        raise ReactionSpecialistUgiV0ComparisonError("Ugi specialist result is not admissible")
    programs = _load_programs(inputs["program_draw"], count=int(runtime["program_count"]))
    topology_policy = _load_topology_policy(inputs) if topology_specialist else None
    current_inputs = {
        "checkpoint_archive": inputs["base_checkpoint_archive"],
        "common_ugi_assessment_config": inputs["common_ugi_assessment_config"],
        "current_training_result": inputs["base_training_result"],
        "lipid_realism_config": inputs["lipid_realism_config"],
        "local_chemistry_config": inputs["local_chemistry_config"],
        "production_cache": inputs["production_cache"],
        "production_design": inputs["base_design"],
        "program_draw": inputs["program_draw"],
        "role_morphology_policy": inputs["role_morphology_policy"],
    }
    current_rows, current_sampling, current_sampling_path = _load_or_sample_current(
        repo=repo,
        output_dir=output_dir / "specialist_sampling",
        inputs=current_inputs,
        runtime=runtime,
        config_sha256=sha256_file(config_path),
        device=device,
        arm_id=str(config["base_checkpoint"]["arm_id"]),
        checkpoint_step=int(config["base_checkpoint"]["step"]),
        programs=programs,
        specialist_checkpoint=specialist_checkpoint_path,
        ugi_topology_policy=topology_policy,
    )
    current_assessment = _load_or_assess_current(
        rows=current_rows,
        sampling_path=current_sampling_path,
        output_dir=output_dir / "specialist_assessment",
        inputs=current_inputs,
        repo=repo,
        method_id=(
            "forge_shared_plus_ugi_topology_specialist_exposure_matched_seed0"
            if topology_specialist
            else "forge_shared_plus_ugi_specialist_exposure_matched_seed0"
        ),
    )
    v0_inputs = {
        label: inputs[label]
        for label in (
            "program_draw",
            "closure_checkpoint",
            "prepared_cache",
            "qualified_reactions",
            "common_ugi_assessment_config",
            "lipid_realism_config",
            "local_chemistry_config",
            "role_morphology_policy",
        )
    }
    v0 = sample_and_assess_checkpoint(
        checkpoint_path=inputs["v0_checkpoint"],
        checkpoint_step=3000,
        arm_id="v0_reference",
        method_id="forge_v0_step_3000_program_matched",
        assessment_seed_label=0,
        runtime={
            "program_count": int(runtime["program_count"]),
            "sample_steps": int(runtime["sample_steps"]),
            "batch_size": int(runtime["batch_size"]),
            "flow_seed": int(runtime["flow_seed"]),
            "terminal_decoder_mode": "argmax",
            "terminal_decoder_seed": None,
            "terminal_temperature": 1.0,
            "maximum_adjacent_branch_runs": list(runtime["maximum_adjacent_branch_runs"]),
        },
        inputs=v0_inputs,
        repo=repo,
        output_dir=output_dir / "v0",
        device=device,
        heldout_rows_used=True,
    )
    current_metrics = dict(current_assessment["assessment"]["metrics"])
    v0_metrics = dict(v0["assessment"]["metrics"])
    deltas = _metric_deltas(current_metrics, v0_metrics)
    v0_sampling_path = output_dir / "v0" / "sampling" / "result.json"
    v0_sampling = read_json_object(
        v0_sampling_path,
        error=ReactionSpecialistUgiV0ComparisonError,
        label="v0 program-matched sampling result",
    )
    v0_rows = v0_sampling.get("samples")
    paired = bool(
        isinstance(v0_rows, list)
        and len(v0_rows) == len(current_rows) == len(programs)
        and all(
            current_rows[index]["program"] == v0_rows[index].get("program")
            for index in range(len(programs))
        )
    )
    primary = str(config["decision_rule"]["primary_metric"])
    margin = float(config["decision_rule"]["absolute_margin"])
    reliability = tuple(config["decision_rule"]["reliability_metrics"])
    required_decision_metrics = (primary, *reliability)
    missing_decision_metrics = sorted(set(required_decision_metrics) - set(deltas))
    if missing_decision_metrics:
        raise ReactionSpecialistUgiV0ComparisonError(
            f"decision metrics are absent from the shared assessor: {missing_decision_metrics}"
        )
    unavailable_decision_metrics = sorted(
        key for key in required_decision_metrics if deltas[key] is None
    )
    if unavailable_decision_metrics:
        raise ReactionSpecialistUgiV0ComparisonError(
            f"decision metrics are unavailable: {unavailable_decision_metrics}"
        )
    if all(float(deltas[key]) >= -margin for key in reliability) and float(
        deltas[primary]
    ) > margin:
        decision = "shared_plus_specialist_superior_under_frozen_gate"
    elif float(deltas[primary]) < -margin:
        decision = "v0_reference_better_on_primary_metric"
    else:
        decision = "no_superiority_under_frozen_gate"
    gates = {
        "attempt_denominator_matched": len(current_rows)
        == len(v0_rows or [])
        == int(runtime["program_count"]),
        "candidate_selection_absent": True,
        "exact_ugi_exposure_matched": int(config["matched_ugi_exposure"]) == 384000,
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": not current_sampling.get("repairs")
        and int(v0_sampling.get("sampling", {}).get("terminal_tree_repairs", -1)) == 0,
        "paired_program_order": paired,
        "program_topology_decoder_recorded": current_sampling.get("terminal_decode_policy")
        == str(runtime["current_terminal_decode_policy"]),
        "route_or_oracle_calls_zero": True,
        "training_calls_zero": True,
    }
    result = {
        "schema_version": TOPOLOGY_RESULT_SCHEMA if topology_specialist else RESULT_SCHEMA,
        "status": "complete" if all(gates.values()) else "fail",
        "profile": profile,
        "programs_per_method": len(programs),
        "comparison_design": {
            "program_rows_and_order_matched": paired,
            "ugi_training_exposure_matched": int(config["matched_ugi_exposure"]),
            "total_training_compute_matched": False,
            "sample_steps_matched": True,
            "model_native_constrained_argmax": not topology_specialist,
            "exact_program_coupled_topology": topology_specialist,
            "repairs_or_retries": False,
            "independently_trained_seed_replication": False,
        },
        "methods": {
            (
                "forge_shared_plus_ugi_topology_specialist_exposure_matched_seed0"
                if topology_specialist
                else "forge_shared_plus_ugi_specialist_exposure_matched_seed0"
            ): {
                "metrics": current_metrics,
                "sampling": artifact_record(current_sampling_path),
                "assessment": artifact_record(
                    output_dir / "specialist_assessment" / "result_index.json"
                ),
            },
            "forge_v0_step_3000_program_matched": {
                "metrics": v0_metrics,
                "sampling": artifact_record(v0_sampling_path),
                "assessment": artifact_record(
                    output_dir / "v0" / "assessment" / "result_index.json"
                ),
            },
        },
        "specialist_minus_v0": deltas,
        "decision_rule": dict(config["decision_rule"]),
        "decision": decision,
        "gates": gates,
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "specialist_checkpoint": artifact_record(specialist_checkpoint_path),
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "complete":
        raise ReactionSpecialistUgiV0ComparisonError(
            f"specialist comparison gates failed: {gates}"
        )
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "TOPOLOGY_CONFIG_SCHEMA",
    "TOPOLOGY_RESULT_SCHEMA",
    "ReactionSpecialistUgiV0ComparisonError",
    "RECOVERY_INPUT_LABELS",
    "run_reaction_specialist_ugi_v0_comparison",
]
