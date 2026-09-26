"""Fail-closed qualification of the bounded multi-reaction overfit experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.core.hashing import pin_record, sha256_file
from forge.core.io import atomic_write, pretty_json_bytes, read_json_object

CONFIG_SCHEMA = "forge.multireaction_overfit_qualification_config.v1"
RESULT_SCHEMA = "forge.multireaction_overfit_qualification.v1"


class MultiReactionQualificationError(ValueError):
    """The overfit artifacts do not satisfy their frozen comparison contract."""


def qualify_overfit_run(
    config_path: Path,
    repo: Path,
    training_result_path: Path,
    sampling_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate prespecified gates and preserve both passing and negative outcomes."""

    config = read_json_object(
        config_path,
        error=MultiReactionQualificationError,
        label="multi-reaction overfit qualification config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or config.get("inputs") != {}:
        raise MultiReactionQualificationError("unsupported overfit qualification config")
    training = read_json_object(
        training_result_path,
        error=MultiReactionQualificationError,
        label="multi-reaction training result",
    )
    sampling = read_json_object(
        sampling_result_path,
        error=MultiReactionQualificationError,
        label="multi-reaction sampling result",
    )
    if (
        training.get("schema_version") != "forge.multireaction_training_result.v2"
        or sampling.get("schema_version") != "forge.multireaction_sampling_result.v2"
        or training.get("run_kind") != "overfit_gate"
        or sampling.get("run_kind") != "overfit_gate"
    ):
        raise MultiReactionQualificationError("qualification received non-overfit artifacts")
    arm = training.get("arms", {}).get("program")
    metrics = sampling.get("metrics")
    gates = config.get("gates")
    if not isinstance(arm, dict) or not isinstance(metrics, dict) or not isinstance(gates, dict):
        raise MultiReactionQualificationError("overfit metrics or gates are missing")
    initial_loss = float(arm["initial_total_loss"])
    final_loss = float(arm["final_total_loss"])
    loss_ratio = final_loss / initial_loss
    evaluations = arm.get("evaluation_by_program")
    if not isinstance(evaluations, dict) or not evaluations:
        raise MultiReactionQualificationError("training result has no per-program evaluation")
    evaluated = sum(int(value["records"]) for value in evaluations.values())
    exact_tensors = sum(
        int(value["fixed_noise_reconstruction"]["exact_tensor_records"])
        for value in evaluations.values()
    )
    exact_tensor_fraction = exact_tensors / evaluated
    sample_count = int(metrics["samples"])
    if sample_count < 1:
        raise MultiReactionQualificationError("overfit sampler emitted no records")
    validity_fraction = int(metrics["valid"]) / sample_count
    exact_l1_fraction = int(metrics["exact_l1_program"]) / sample_count
    decisions = {
        "finite_training": bool(arm["all_losses_finite"]),
        "loss_reduction": loss_ratio <= float(gates["maximum_final_to_initial_loss_ratio"]),
        "fixed_noise_tensor_reconstruction": exact_tensor_fraction
        >= float(gates["minimum_exact_tensor_fraction"]),
        "native_validity": validity_fraction >= float(gates["minimum_valid_fraction"]),
        "exact_l1_generation": exact_l1_fraction >= float(gates["minimum_exact_l1_fraction"]),
    }
    passed = all(decisions.values())

    def dependency_record(path: Path, stage_id: str) -> dict[str, Any]:
        return {
            "path": f"{stage_id}/result.json",
            "sha256": str(sha256_file(path)),
            "bytes": path.stat().st_size,
        }

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "overfit_gate_pass" if passed else "overfit_gate_failed",
        "gate_pass": passed,
        "gates": decisions,
        "thresholds": gates,
        "measurements": {
            "initial_total_loss": initial_loss,
            "final_total_loss": final_loss,
            "final_to_initial_loss_ratio": loss_ratio,
            "evaluated_tensor_records": evaluated,
            "exact_tensor_records": exact_tensors,
            "exact_tensor_fraction": exact_tensor_fraction,
            "samples": sample_count,
            "valid_fraction": validity_fraction,
            "exact_l1_fraction": exact_l1_fraction,
        },
        "config": pin_record(config_path, repo),
        "inputs": {
            "training_result": dependency_record(training_result_path, "training"),
            "sampling_result": dependency_record(sampling_result_path, "sampling"),
        },
        "decision": (
            "A passing bounded overfit gate permits production-qualification implementation; "
            "it does not authorize a production launch or establish generation quality."
            if passed
            else "Production qualification remains blocked; retain this negative result and "
            "repair the representation or optimization before increasing compute."
        ),
        "nonclaims": [
            "Exact L1 is transform consistency, not synthesis success.",
            "The selected records are a pipeline overfit set, not heldout evidence.",
            "No biological, L2 routing or procurement information is used.",
        ],
    }
    atomic_write(output_path, pretty_json_bytes(result))
    return result


__all__ = ["MultiReactionQualificationError", "qualify_overfit_run"]
