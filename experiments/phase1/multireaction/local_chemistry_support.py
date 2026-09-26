"""Build and assess training-derived local chemistry support without selecting candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.local_chemistry_support import (
    LEGACY_POLICY_SCHEMA,
    POLICY_SCHEMA,
    LocalChemistrySupport,
    build_local_chemistry_support,
)

LEGACY_CONFIG_SCHEMA = "forge.local_chemistry_support_config.v1"
CONFIG_SCHEMA = "forge.local_chemistry_support_config.v2"
LEGACY_RESULT_SCHEMA = "forge.local_chemistry_support_result.v1"
RESULT_SCHEMA = "forge.local_chemistry_support_result.v2"


class LocalChemistrySupportBuildError(ValueError):
    """The local-support build differs from its frozen training-only contract."""


def build_local_chemistry_support_artifact(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build one deterministic policy from training-fold graph records only."""

    config = read_json_object(
        config_path,
        error=LocalChemistrySupportBuildError,
        label="local chemistry support config",
    )
    config_schema = config.get("schema_version")
    if config_schema not in {LEGACY_CONFIG_SCHEMA, CONFIG_SCHEMA} or set(config) != {
        "schema_version",
        "scientific_question",
        "inputs",
        "policy",
        "reporting",
        "nonclaims",
    }:
        raise LocalChemistrySupportBuildError("unsupported local chemistry support config")
    policy = config["policy"]
    expected_policy = {
        "source_fold": "train",
        "edge_support": "program_and_precursor_origin_role_conditioned",
        "three_membered_ring_support": "program_and_precursor_origin_role_conditioned",
        "component_hard_bounds": "observed_training_min_max",
        "component_diagnostic_bounds": "training_q01_q99_nonselecting",
        "store_component_identities": False,
    }
    if config_schema == CONFIG_SCHEMA:
        expected_policy.update(
            {
                "fundamental_cycle_support": (
                    "program_and_precursor_origin_role_conditioned_training_closures"
                ),
                "fixed_adapter_cycles_authorize_variable_closures": False,
            }
        )
    if not isinstance(policy, dict) or policy != expected_policy:
        raise LocalChemistrySupportBuildError("local chemistry policy changed")
    reporting = config["reporting"]
    if not isinstance(reporting, dict) or reporting != {
        "candidate_selection": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "negative_results_are_reported": True,
    }:
        raise LocalChemistrySupportBuildError("local chemistry reporting guardrails changed")
    raw_inputs = config["inputs"]
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {"production_cache"}:
        raise LocalChemistrySupportBuildError("local chemistry inputs changed")
    cache_path = resolve_pin(raw_inputs["production_cache"], repo, label="production_cache")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LocalChemistrySupportBuildError(f"output directory is not empty: {output_dir}")

    cache = SynthesisProgramProductionCache(cache_path)
    try:
        training_indices = cache.indices(fold="train")
        support = build_local_chemistry_support(
            (cache.record(int(index)) for index in training_indices),
            cache.atom_vocabulary,
            schema_version=(
                POLICY_SCHEMA if config_schema == CONFIG_SCHEMA else LEGACY_POLICY_SCHEMA
            ),
        )
        expected_training_counts = {
            program_id: int(cache.indices(program_id=program_id, fold="train").size)
            for program_id in cache.vocabulary.program_states[1:]
        }
    finally:
        cache.close()

    payload = support.to_mapping()
    roundtrip = LocalChemistrySupport.from_mapping(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = output_dir / "policy.json"
    write_json(policy_path, payload)
    gates = {
        "training_fold_only": payload["policy"]["training_fold_only"] is True,
        "all_training_records_counted": dict(support.training_records) == expected_training_counts,
        "component_identities_absent": payload["policy"]["complete_component_identities_stored"]
        is False,
        "deterministic_schema_roundtrip": roundtrip.to_mapping() == payload,
        "all_programs_have_edge_support": all(
            support.program_edges[program_id] and support.role_edges[program_id]
            for program_id in support.programs
        ),
        "candidate_selection_absent": True,
        "route_or_oracle_calls_zero": True,
    }
    if config_schema == CONFIG_SCHEMA:
        gates["full_role_cycle_support_frozen"] = support.enforces_role_cycles
    result = {
        "schema_version": (
            RESULT_SCHEMA if config_schema == CONFIG_SCHEMA else LEGACY_RESULT_SCHEMA
        ),
        "status": "pass" if all(gates.values()) else "fail",
        "config": pin_record(config_path, repo),
        "inputs": {"production_cache": pin_record(cache_path, repo)},
        "policy": artifact_record(policy_path),
        "programs": {
            program_id: {
                "training_records": int(support.training_records[program_id]),
                "program_edge_types": len(support.program_edges[program_id]),
                "role_conditioned_edge_types": len(support.role_edges[program_id]),
                "program_three_membered_ring_types": len(support.program_triangles[program_id]),
                "role_conditioned_three_membered_ring_types": len(
                    support.role_triangles[program_id]
                ),
                **(
                    {
                        "program_fundamental_cycle_types": len(support.program_cycles[program_id]),
                        "role_conditioned_fundamental_cycle_types": len(
                            support.role_cycles[program_id]
                        ),
                        "role_conditioned_cycle_sizes": sorted(
                            {len(cycle) for cycle in support.role_cycles[program_id]}
                        ),
                    }
                    if support.enforces_role_cycles
                    else {}
                ),
                "component_bounds": {
                    role: bounds.to_mapping()
                    for role, bounds in sorted(support.component_bounds[program_id].items())
                },
            }
            for program_id in support.programs
        },
        "gates": gates,
        "candidate_selection": False,
        "calls": {"route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise LocalChemistrySupportBuildError(f"local chemistry gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "LEGACY_CONFIG_SCHEMA",
    "LEGACY_RESULT_SCHEMA",
    "RESULT_SCHEMA",
    "LocalChemistrySupportBuildError",
    "build_local_chemistry_support_artifact",
]
