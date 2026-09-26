"""Fail-closed qualification of the shared three-program integration gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.synthesis_program_integration_qualification_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_integration_qualification.v1"


class SharedSynthesisProgramQualificationError(ValueError):
    """Shared cache, training, and sampling artifacts do not form one qualified run."""


def qualify_shared_synthesis_program_integration(
    config_path: Path,
    repo: Path,
    cache_result_path: Path,
    training_result_path: Path,
    sampling_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Apply the frozen integration gates without promoting an overfit result."""

    config = read_json_object(
        config_path,
        error=SharedSynthesisProgramQualificationError,
        label="shared synthesis-program qualification config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SharedSynthesisProgramQualificationError("unsupported qualification config")
    cache = read_json_object(
        cache_result_path,
        error=SharedSynthesisProgramQualificationError,
        label="shared cache result",
    )
    training = read_json_object(
        training_result_path,
        error=SharedSynthesisProgramQualificationError,
        label="shared training result",
    )
    sampling = read_json_object(
        sampling_result_path,
        error=SharedSynthesisProgramQualificationError,
        label="shared sampling result",
    )
    if (
        cache.get("schema_version") != "forge.synthesis_program_cache_qualification.v1"
        or training.get("schema_version") != "forge.synthesis_program_training_result.v1"
        or sampling.get("schema_version") != "forge.synthesis_program_sampling_result.v1"
    ):
        raise SharedSynthesisProgramQualificationError("qualification input schema changed")
    if training["cache"]["sha256"] != cache["cache"]["sha256"]:
        raise SharedSynthesisProgramQualificationError("training did not consume qualified cache")
    if sampling["cache"]["sha256"] != cache["cache"]["sha256"]:
        raise SharedSynthesisProgramQualificationError("sampling did not consume qualified cache")
    if sampling["checkpoint"]["sha256"] != training["checkpoint"]["sha256"]:
        raise SharedSynthesisProgramQualificationError(
            "sampling did not consume the qualified training checkpoint"
        )
    gates = {
        "cache_contract_passed": cache["status"] == "pass" and all(cache["gates"].values()),
        "ugi_regression_control_passed": bool(cache["ugi_regression"]["atom_states_exact"])
        and bool(cache["ugi_regression"]["bond_states_and_endpoints_exact"]),
        "training_contract_passed": training["status"] == "pass"
        and all(training["gates"].values()),
        "zero_fixed_equivalence_passed": all(
            bool(training["zero_fixed_equivalence"][key])
            for key in ("noise_tensors_exact", "loss_exact", "metric_values_exact")
        ),
        "sampling_contract_passed": sampling["status"] == "pass"
        and all(sampling["gates"].values()),
        "fixed_states_never_changed": int(sampling["metrics"]["fixed_state_failures"]) == 0,
        "all_three_programs_sampled": set(sampling["metrics"]["by_program"])
        == set(cache["selected_records"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "integration_gate_pass" if all(gates.values()) else "integration_gate_fail",
        "config": pin_record(config_path, repo),
        "inputs": {
            "cache_result": artifact_record(cache_result_path, logical_path="cache/result.json"),
            "training_result": artifact_record(
                training_result_path, logical_path="training/result.json"
            ),
            "sampling_result": artifact_record(
                sampling_result_path, logical_path="sampling/result.json"
            ),
        },
        "gates": gates,
        "summary": {
            "programs": len(cache["selected_records"]),
            "selected_records": len(cache["selected_records"]),
            "samples": int(sampling["metrics"]["samples"]),
            "valid_samples": int(sampling["metrics"]["valid"]),
            "exact_tensor_samples": int(sampling["metrics"]["exact_tensor"]),
            "exact_graph_samples": int(sampling["metrics"]["exact_target_graph"]),
            "production_training_authorized": False,
        },
        "decision": config["decision"],
        "nonclaims": config["nonclaims"],
    }
    write_json(output_path, result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SharedSynthesisProgramQualificationError",
    "qualify_shared_synthesis_program_integration",
]
