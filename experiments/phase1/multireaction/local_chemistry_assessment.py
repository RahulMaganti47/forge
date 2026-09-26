"""Method-blind local-chemistry assessment for frozen common Ugi attempt ledgers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import iter_jsonl, read_json_object, write_json, write_jsonl
from forge.model.common_ugi_benchmark import ASSESSED_ATTEMPT_SCHEMA, UGI_PROGRAM_ID
from forge.model.local_chemistry_support import (
    ASSESSMENT_SCHEMA,
    LocalChemistrySupport,
    assess_product_local_chemistry,
)

CONFIG_SCHEMA = "forge.common_local_chemistry_assessment_config.v1"
RESULT_SCHEMA = "forge.common_local_chemistry_complete_assessment.v1"
ATTEMPT_SCHEMA = "forge.common_local_chemistry_attempt.v1"


class LocalChemistryAssessmentError(ValueError):
    """A common attempt ledger violates the local-chemistry comparison contract."""


def _load_assessed_attempts(
    path: Path,
    *,
    method_id: str,
    seed: int,
    expected_attempts: int | None,
) -> list[dict[str, Any]]:
    records = list(iter_jsonl(path))
    if not records or not isinstance(records[0], Mapping):
        raise LocalChemistryAssessmentError("common assessed-attempt ledger is empty")
    header = records.pop(0)
    if header != {
        "schema_version": "forge.common_ugi_assessed_attempts.v1",
        "rows": len(records),
    }:
        raise LocalChemistryAssessmentError("common assessed-attempt header changed")
    if expected_attempts is not None and len(records) != expected_attempts:
        raise LocalChemistryAssessmentError(
            f"attempt denominator changed: expected {expected_attempts}, found {len(records)}"
        )
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(records):
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != ASSESSED_ATTEMPT_SCHEMA
            or value.get("method_id") != method_id
            or value.get("seed") != seed
            or value.get("attempt_index") != index
            or value.get("program_id") != UGI_PROGRAM_ID
        ):
            raise LocalChemistryAssessmentError(f"assessed attempt identity changed at row {index}")
        rows.append(dict(value))
    return rows


def run_local_chemistry_assessment(
    config_path: Path,
    repo: Path,
    assessed_attempts_path: Path,
    output_dir: Path,
    *,
    method_id: str,
    seed: int,
    expected_attempts: int | None = None,
) -> dict[str, Any]:
    """Assess one frozen method/seed ledger without generation, selection, routes, or oracles."""

    config = read_json_object(
        config_path,
        error=LocalChemistryAssessmentError,
        label="common local chemistry assessment config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "metrics",
        "nonclaims",
    }:
        raise LocalChemistryAssessmentError("unsupported local chemistry assessment config")
    metrics = config["metrics"]
    if not isinstance(metrics, dict) or metrics != {
        "primary": "local_support_qualified_exact_l1_yield_per_attempt",
        "raw_exact_l1_retained": True,
        "invalid_failed_and_duplicate_attempts_retained": True,
        "method_private_atom_roles_required": False,
        "candidate_selection": False,
    }:
        raise LocalChemistryAssessmentError("local chemistry metric guardrails changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {"local_chemistry_support"}:
        raise LocalChemistryAssessmentError("local chemistry assessment inputs changed")
    policy_path = resolve_pin(
        raw_inputs["local_chemistry_support"], repo, label="local_chemistry_support"
    )
    support = LocalChemistrySupport.from_mapping(
        read_json_object(
            policy_path,
            error=LocalChemistryAssessmentError,
            label="local chemistry support",
        )
    )
    rows = _load_assessed_attempts(
        assessed_attempts_path,
        method_id=method_id,
        seed=seed,
        expected_attempts=expected_attempts,
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LocalChemistryAssessmentError(f"output directory is not empty: {output_dir}")

    assessed: list[dict[str, Any]] = []
    failure_types: Counter[str] = Counter()
    raw_exact_l1 = 0
    qualified_exact_l1 = 0
    local_supported = 0
    unique_qualified: set[str] = set()
    for row in rows:
        local = assess_product_local_chemistry(
            row.get("canonical_smiles"),
            program_id=UGI_PROGRAM_ID,
            support=support,
        )
        if local["schema_version"] != ASSESSMENT_SCHEMA:
            raise LocalChemistryAssessmentError("local chemistry assessment schema changed")
        raw = row.get("exact_l1_program") is True
        qualified = raw and local["local_chemistry_supported"] is True
        raw_exact_l1 += int(raw)
        qualified_exact_l1 += int(qualified)
        local_supported += int(local["local_chemistry_supported"] is True)
        failure_types.update(local["failure_types"])
        if qualified and row.get("canonical_smiles") is not None:
            unique_qualified.add(str(row["canonical_smiles"]))
        assessed.append(
            {
                "schema_version": ATTEMPT_SCHEMA,
                "method_id": method_id,
                "seed": seed,
                "attempt_index": int(row["attempt_index"]),
                "program_id": UGI_PROGRAM_ID,
                "valid": bool(row.get("valid")),
                "canonical_smiles": row.get("canonical_smiles"),
                "raw_exact_l1_program": raw,
                "local_chemistry_supported": bool(local["local_chemistry_supported"]),
                "local_support_qualified_exact_l1": qualified,
                "unsupported_edge_count": int(local["unsupported_edge_count"]),
                "unsupported_three_membered_ring_count": int(
                    local["unsupported_three_membered_ring_count"]
                ),
                "failure_types": list(local["failure_types"]),
            }
        )

    denominator = len(rows)
    assessment = {
        "method_id": method_id,
        "seed": seed,
        "attempts": denominator,
        "raw_exact_l1_products": raw_exact_l1,
        "raw_exact_l1_yield_per_attempt": raw_exact_l1 / denominator,
        "local_chemistry_supported_products": local_supported,
        "local_chemistry_supported_products_per_attempt": local_supported / denominator,
        "local_support_qualified_exact_l1_products": qualified_exact_l1,
        "local_support_qualified_exact_l1_yield_per_attempt": qualified_exact_l1 / denominator,
        "local_support_precision_among_raw_exact_l1": (
            qualified_exact_l1 / raw_exact_l1 if raw_exact_l1 else None
        ),
        "unique_local_support_qualified_exact_l1_products": len(unique_qualified),
        "failure_types": dict(sorted(failure_types.items())),
        "attempt_denominator_includes_invalid_failed_duplicate_and_unsupported": True,
        "candidate_selection": False,
        "calls": {"generation": 0, "route": 0, "oracle": 0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_output = output_dir / "assessed_attempts.jsonl.gz"
    write_jsonl(
        attempts_output,
        [
            {"schema_version": ATTEMPT_SCHEMA, "rows": len(assessed)},
            *assessed,
        ],
    )
    gates = {
        "attempt_denominator_preserved": denominator == len(assessed),
        "raw_exact_l1_not_replaced": all(
            item["raw_exact_l1_program"] == bool(source.get("exact_l1_program"))
            for item, source in zip(assessed, rows, strict=True)
        ),
        "method_and_seed_preserved": all(
            item["method_id"] == method_id and item["seed"] == seed for item in assessed
        ),
        "candidate_selection_absent": True,
        "generation_route_or_oracle_calls_zero": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "config": pin_record(config_path, repo),
        "inputs": {
            "local_chemistry_support": pin_record(policy_path, repo),
            "common_assessed_attempts": artifact_record(assessed_attempts_path),
        },
        "assessed_attempts": artifact_record(attempts_output),
        "assessment": assessment,
        "gates": gates,
        "candidate_selection": False,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise LocalChemistryAssessmentError(f"local chemistry gates failed: {gates}")
    return result


__all__ = [
    "ATTEMPT_SCHEMA",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "LocalChemistryAssessmentError",
    "run_local_chemistry_assessment",
]
