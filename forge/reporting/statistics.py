"""Strict aggregation and machine-generated v1 paper table/figure inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record
from forge.core.io import (
    read_json_object,
)

CONFIG_SCHEMA = "forge.renderer_config.v1"
LEDGER_SCHEMA = "forge.result_rows.v1"
COMMON_ROW_SCHEMA = "forge.common_seed_row.v2"
MECHANISM_ROW_SCHEMA = "forge.mechanism_seed_row.v1"
HELD_FAMILY_ROW_SCHEMA = "forge.held_family_seed_row.v1"

COMMON_METRICS = (
    "valid_per_1000",
    "exact_l1_per_1000",
    "unique_l1_per_1000",
    "open_ended_per_1000",
    "held_component_per_1000",
    "diversity",
    "decomposition_coverage",
    "replay_precision",
    "ambiguity_fraction",
    "verified_upstream_fraction",
    "terminal_evidence_fraction",
    "complete_dossier_fraction",
    "route_abstention_fraction",
)
MECHANISM_METRICS = (
    "held_family_loss",
    "ugi_l1_per_1000",
    "bl_l1_per_1000",
    "lx_l1_per_1000",
    "held_component_per_1000",
    "cross_role_fidelity",
    "diversity",
)


class PaperResultsV1Error(ValueError):
    """Verified experiment rows are incomplete or inconsistent for paper rendering."""


def mechanism_seed_rows(evaluation_path: Path, repo: Path) -> list[dict[str, Any]]:
    """Extract all architecture rows from one passing mechanism-study evaluation."""

    result = read_json_object(
        evaluation_path, error=PaperResultsV1Error, label="mechanism evaluation result"
    )
    if (
        result.get("schema_version") != "forge.synthesis_program_production_evaluation_result.v1"
        or result.get("status") != "pass"
    ):
        raise PaperResultsV1Error("mechanism evaluation is not a passing production evaluation")
    checkpoints = result["checkpoint_metrics"]
    component = result["component_disjoint_metrics"]
    held = result["ugi_held_component_metrics"]
    fidelity = result["cross_role_fidelity"]
    final_step = str(max(int(step) for arm in checkpoints.values() for step in arm))
    program_fields = {
        "ugi_l1_per_1000": "ugi_3cr_agile",
        "bl_l1_per_1000": "bl_2023_repeated_aza_michael",
        "lx_l1_per_1000": "lx_2024_repeated_reductive_amination",
    }
    rows = []
    for arm_id in sorted(checkpoints):
        final = checkpoints[arm_id][final_step]["heldout"]
        if set(final) != set(program_fields.values()):
            raise PaperResultsV1Error(f"mechanism arm {arm_id} does not cover all three programs")
        losses = [
            float(component[arm_id][program]["heldout_denoising_loss_at_t_0_5"])
            for program in program_fields.values()
        ]
        diversities = [final[program]["internal_diversity"] for program in final]
        rows.append(
            {
                "schema_version": MECHANISM_ROW_SCHEMA,
                "arm_id": arm_id,
                "seed": int(result["seed"]),
                "metrics": {
                    "held_family_loss": float(np.mean(losses)),
                    **{
                        field: 1000.0 * float(final[program]["exact_l1_yield_per_attempt"])
                        for field, program in program_fields.items()
                    },
                    "held_component_per_1000": float(
                        held[arm_id]["exact_l1_products_with_held_component_per_1000_attempts"]
                    ),
                    "cross_role_fidelity": fidelity[arm_id][
                        "residual_correlation_frobenius_distance"
                    ],
                    "diversity": (
                        float(np.mean([float(value) for value in diversities]))
                        if all(value is not None for value in diversities)
                        else None
                    ),
                },
                "source": pin_record(evaluation_path, repo),
            }
        )
    return rows


__all__ = [
    "COMMON_ROW_SCHEMA",
    "HELD_FAMILY_ROW_SCHEMA",
    "LEDGER_SCHEMA",
    "MECHANISM_ROW_SCHEMA",
    "PaperResultsV1Error",
    "mechanism_seed_rows",
]
