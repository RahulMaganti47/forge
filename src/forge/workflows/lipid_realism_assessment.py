"""Run the frozen method-blind whole-lipid structural-realism assessment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json, write_jsonl
from forge.model.common_lipid_realism import (
    ATTEMPT_ASSESSMENT_SCHEMA,
    RealismPolicy,
    assess_lipid_realism,
    build_realism_reference,
)
from forge.model.common_ugi_benchmark import load_attempt_ledger

CONFIG_SCHEMA = "forge.common_lipid_realism_config.v1"
RESULT_SCHEMA = "forge.common_lipid_realism_complete_assessment.v1"


class LipidRealismAssessmentError(ValueError):
    """A ledger cannot be assessed under the frozen method-blind realism protocol."""


def run_lipid_realism_assessment(
    config_path: Path,
    repo: Path,
    attempts_path: Path,
    output_dir: Path,
    *,
    method_id: str,
    seed: int,
    expected_attempts: int | None = None,
) -> dict[str, Any]:
    """Assess one method/seed ledger without generation, routes, or oracle calls."""

    config = read_json_object(
        config_path, error=LipidRealismAssessmentError, label="lipid realism config"
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "policy",
        "metrics",
        "nonclaims",
    }:
        raise LipidRealismAssessmentError("unsupported or malformed lipid realism config")
    metrics = config["metrics"]
    if (
        not isinstance(metrics, dict)
        or metrics.get("attempt_denominator_includes_invalid_failed_duplicate_and_out_of_support")
        is not True
        or metrics.get("fingerprint_and_descriptor_manifolds_reported_separately") is not True
        or metrics.get("no_result_selected_thresholds") is not True
        or metrics.get("qed_excluded") is not True
    ):
        raise LipidRealismAssessmentError("lipid realism metric guardrails changed")
    nonclaims = config["nonclaims"]
    if (
        not isinstance(nonclaims, list)
        or not nonclaims
        or any(not isinstance(value, str) or not value for value in nonclaims)
    ):
        raise LipidRealismAssessmentError("lipid realism nonclaims must be explicit")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "r0_constitutional",
        "r0_fold_assignments",
    }:
        raise LipidRealismAssessmentError("lipid realism input pins changed")
    inputs = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    policy = RealismPolicy.from_mapping(config["policy"])
    attempts = load_attempt_ledger(
        attempts_path,
        expected_method=method_id,
        expected_seed=seed,
        expected_attempts=expected_attempts,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LipidRealismAssessmentError(f"output directory is not empty: {output_dir}")
    reference = build_realism_reference(
        inputs["r0_constitutional"], inputs["r0_fold_assignments"], policy
    )
    rows, assessment = assess_lipid_realism(attempts, reference, policy)

    output_dir.mkdir(parents=True, exist_ok=True)
    assessed_path = output_dir / "assessed_attempts.jsonl.gz"
    write_jsonl(
        assessed_path,
        [
            {"schema_version": ATTEMPT_ASSESSMENT_SCHEMA, "rows": len(rows)},
            *rows,
        ],
    )
    gates = {
        "attempt_denominator_preserved": len(attempts) == assessment["attempts"],
        "method_and_seed_preserved": assessment["method_id"] == method_id
        and int(assessment["seed"]) == seed,
        "source_study_heldout_reference": assessment["reference"][
            "reference_population"
        ].startswith("source-study-held-out"),
        "training_only_descriptor_scaling": assessment["reference"]["scaling_population"]
        == "R0_train only",
        "reference_selected_independently_of_method": assessment["reference"]["selection"].endswith(
            "independent of method outputs"
        ),
        "coverage_and_precision_reported_separately": assessment[
            "coverage_and_precision_reported_separately"
        ]
        is True,
        "invalid_failed_and_out_of_support_preserved": assessment[
            "attempt_denominator_includes_invalid_failed_and_out_of_support"
        ]
        is True,
        "qed_absent": assessment["qed_reported"] is False,
        "route_or_oracle_calls_during_assessment_zero": assessment["route_or_oracle_calls"] == 0,
        "candidate_selection_absent": assessment["candidate_selection"] is False,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "method_id": method_id,
        "seed": seed,
        "attempts": artifact_record(attempts_path),
        "assessed_attempts": artifact_record(assessed_path),
        "assessment": assessment,
        "config": pin_record(config_path, repo),
        "inputs": {
            "r0_constitutional": pin_record(inputs["r0_constitutional"], repo),
            "r0_fold_assignments": pin_record(inputs["r0_fold_assignments"], repo),
        },
        "gates": gates,
        "candidate_selection": False,
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise LipidRealismAssessmentError(f"lipid realism gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "LipidRealismAssessmentError",
    "run_lipid_realism_assessment",
]
