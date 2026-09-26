"""Apply the common Ugi molecular, exact-L1 and frozen route-evidence contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json, write_jsonl
from forge.model.common_ugi_benchmark import assess_common_ugi_attempts, load_attempt_ledger
from forge.synthesis.assessment.common_route_evidence import (
    assess_common_route_evidence,
    load_frozen_component_evidence,
)

CONFIG_SCHEMA = "forge.common_ugi_assessment_config.v1"
RESULT_SCHEMA = "forge.common_ugi_complete_assessment.v1"


class CommonUgiAssessmentError(ValueError):
    """A common method ledger cannot be assessed under the frozen protocol."""


def run_common_ugi_assessment(
    config_path: Path,
    repo: Path,
    attempts_path: Path,
    output_dir: Path,
    *,
    method_id: str,
    seed: int,
    expected_attempts: int | None = None,
) -> dict[str, Any]:
    """Assess one already frozen method/seed ledger without any generation or planner calls."""

    config = read_json_object(
        config_path, error=CommonUgiAssessmentError, label="common Ugi assessment config"
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise CommonUgiAssessmentError("unsupported common Ugi assessment config")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "common_protocol",
        "component_route_ledger",
    }:
        raise CommonUgiAssessmentError("common Ugi assessment inputs changed")
    inputs = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    protocol = read_json_object(
        inputs["common_protocol"], error=CommonUgiAssessmentError, label="common Ugi protocol"
    )
    if protocol.get("schema_version") != "forge.common_ugi_baseline_protocol.v1":
        raise CommonUgiAssessmentError("unsupported common Ugi protocol")
    protocol_inputs = protocol.get("inputs")
    if not isinstance(protocol_inputs, dict) or not {
        "ugi_assignments",
        "qualified_ugi_reactions",
    }.issubset(protocol_inputs):
        raise CommonUgiAssessmentError("common protocol input pins changed")
    assignments = resolve_pin(protocol_inputs["ugi_assignments"], repo, label="ugi_assignments")
    reaction_registry = resolve_pin(
        protocol_inputs["qualified_ugi_reactions"], repo, label="qualified_ugi_reactions"
    )
    attempts = load_attempt_ledger(
        attempts_path,
        expected_method=method_id,
        expected_seed=seed,
        expected_attempts=expected_attempts,
    )
    adapter = Ugi3AssemblyAdapter.from_registry(
        reaction_registry,
        expected_sha256=str(protocol_inputs["qualified_ugi_reactions"]["sha256"]),
    )
    assessed, common = assess_common_ugi_attempts(
        attempts, adapter=adapter, assignments_path=assignments
    )
    evidence = load_frozen_component_evidence(inputs["component_route_ledger"])
    route_rows, route = assess_common_route_evidence(assessed, component_evidence=evidence)
    output_dir.mkdir(parents=True, exist_ok=True)
    assessed_path = output_dir / "assessed_attempts.jsonl.gz"
    route_path = output_dir / "route_assessed_attempts.jsonl.gz"
    write_jsonl(
        assessed_path,
        [
            {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": len(assessed)},
            *assessed,
        ],
    )
    write_jsonl(
        route_path,
        [
            {
                "schema_version": "forge.common_ugi_route_assessed_attempts.v1",
                "rows": len(route_rows),
            },
            *route_rows,
        ],
    )
    gates = {
        "attempt_denominator_preserved": len(attempts) == common["attempts"],
        "method_and_seed_preserved": common["method_id"] == method_id
        and int(common["seed"]) == seed,
        "coverage_and_precision_reported": common["coverage_and_precision_reported"] is True,
        "forbidden_reductive_amination_metric_absent": common[
            "reductive_amination_substructure_rate_reported"
        ]
        is False,
        "route_or_oracle_calls_during_assessment_zero": route["route_or_oracle_calls"] == 0,
        "candidate_selection_absent": common["candidate_selection"] is False,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "method_id": method_id,
        "seed": seed,
        "attempts": artifact_record(attempts_path),
        "assessed_attempts": artifact_record(assessed_path),
        "route_assessed_attempts": artifact_record(route_path),
        "common_assessment": common,
        "route_evidence_assessment": route,
        "config": pin_record(config_path, repo),
        "inputs": {
            "common_protocol": pin_record(inputs["common_protocol"], repo),
            "component_route_ledger": pin_record(inputs["component_route_ledger"], repo),
            "ugi_assignments": pin_record(assignments, repo),
            "qualified_ugi_reactions": pin_record(reaction_registry, repo),
        },
        "gates": gates,
        "candidate_selection": False,
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise CommonUgiAssessmentError(f"common Ugi assessment gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "CommonUgiAssessmentError",
    "run_common_ugi_assessment",
]
