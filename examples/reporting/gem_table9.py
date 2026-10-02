"""Render GEM Table 9 from final FORGE and fixed finite-catalogue evidence."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.reporting.gem_table1 import (
    EXPECTED_SEEDS,
    PROGRAMS,
    load_gem_final_evaluations,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json

CONFIG_SCHEMA = "forge.gem_table9_catalogue_comparison_config.v1"
RESULT_SCHEMA = "forge.gem_table9_catalogue_comparison_render.v1"
CATALOGUE_SCHEMA = "forge.final_bl_core_production_adjudication.v1"
ATTEMPTS_PER_PROGRAM = 3072


class GemTable9Error(ValueError):
    """Final-model or finite-catalogue evidence is incomplete or inadmissible."""


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GemTable9Error(f"{label} is not finite")
    return float(value)


def _final_mean(rows: Sequence[Mapping[str, Any]], metric: str, *, program: str) -> float:
    if len(rows) != 3:
        raise GemTable9Error(f"{program} does not have three final-model rows")
    return statistics.fmean(
        _number(row.get(metric), label=f"FORGE.{program}.{metric}") for row in rows
    )


def _catalogue_mean(metrics: Mapping[str, Any], metric: str, *, program: str) -> float:
    summary = metrics.get(metric)
    if not isinstance(summary, Mapping):
        raise GemTable9Error(f"catalogue metric is missing: {program}.{metric}")
    values = summary.get("by_seed")
    if (
        not isinstance(values, list)
        or len(values) != 3
        or summary.get("all_seed_cells_defined") is not True
        or summary.get("defined_seed_count") != 3
        or summary.get("expected_seed_count") != 3
        or summary.get("undefined_cells_imputed") is not False
    ):
        raise GemTable9Error(f"catalogue seed contract changed: {program}.{metric}")
    parsed = [_number(value, label=f"catalogue.{program}.{metric}") for value in values]
    mean = statistics.fmean(parsed)
    if not math.isclose(
        mean,
        _number(summary.get("mean"), label=f"catalogue.{program}.{metric}.mean"),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise GemTable9Error(f"catalogue mean changed: {program}.{metric}")
    return mean


def _metric_cells(
    program: str,
    forge_rows: Sequence[Mapping[str, Any]],
    catalogue_metrics: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, dict[str, float]]]:
    forge = {
        "valid_per_1000": 1000.0 * _final_mean(forge_rows, "raw_valid_fraction", program=program),
        "exact_l1_per_1000": 1000.0
        * _final_mean(forge_rows, "exact_l1_yield_per_attempt", program=program),
        "distinct_exact_l1_per_1000": _final_mean(
            forge_rows, "unique_exact_l1_products_per_1000_attempts", program=program
        ),
        "component_novel_per_1000": _final_mean(
            forge_rows,
            "unique_open_ended_exact_l1_products_per_1000_attempts",
            program=program,
        ),
        "whole_novelty_percent": 100.0
        * _final_mean(forge_rows, "whole_lipid_novelty_fraction", program=program),
        "component_novelty_percent": 100.0
        * _final_mean(forge_rows, "component_novelty_fraction", program=program),
        "diversity": _final_mean(forge_rows, "internal_diversity", program=program),
        "effective_component_count": _final_mean(
            forge_rows, "effective_component_count", program=program
        ),
    }
    catalogue = {
        "valid_per_1000": 1000.0
        * _catalogue_mean(catalogue_metrics, "raw_valid_fraction", program=program),
        "exact_l1_per_1000": 1000.0
        * _catalogue_mean(catalogue_metrics, "exact_l1_yield_per_attempt", program=program),
        "distinct_exact_l1_per_1000": _catalogue_mean(
            catalogue_metrics, "unique_exact_l1_products_per_1000_attempts", program=program
        ),
        "component_novel_per_1000": _catalogue_mean(
            catalogue_metrics,
            "unique_open_ended_exact_l1_products_per_1000_attempts",
            program=program,
        ),
        "whole_novelty_percent": 100.0
        * _catalogue_mean(catalogue_metrics, "whole_product_novelty_fraction", program=program),
        "component_novelty_percent": 100.0
        * _catalogue_mean(catalogue_metrics, "component_novelty_fraction", program=program),
        "diversity": _catalogue_mean(catalogue_metrics, "internal_diversity", program=program),
        "effective_component_count": _catalogue_mean(
            catalogue_metrics, "effective_component_count", program=program
        ),
    }

    def render(values: Mapping[str, float]) -> tuple[str, ...]:
        return (
            f"{values['valid_per_1000']:.1f}",
            f"{values['exact_l1_per_1000']:.1f}",
            f"{values['distinct_exact_l1_per_1000']:.1f}",
            f"{values['component_novel_per_1000']:.1f}",
            f"{values['whole_novelty_percent']:.1f}",
            f"{values['component_novelty_percent']:.1f}",
            f"{values['diversity']:.3f}/{values['effective_component_count']:.1f}",
        )

    return render(forge), render(catalogue), {"forge": forge, "finite_catalogue": catalogue}


def render_gem_table9_catalogue_comparison(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Generate the final-model versus finite-catalogue transposed table."""

    config = read_json_object(config_path, error=GemTable9Error, label="GEM Table 9 config")
    expected_fields = {
        "schema_version",
        "status",
        "final_evidence_config",
        "catalogue_adjudication",
        "attempts_per_program_per_seed",
        "expected_seeds",
        "candidate_selection",
    }
    if (
        set(config) != expected_fields
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_after_final_model_evaluations"
        or config.get("attempts_per_program_per_seed") != ATTEMPTS_PER_PROGRAM
        or config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or config.get("candidate_selection") is not False
    ):
        raise GemTable9Error("GEM Table 9 config changed")

    final_config = resolve_pin(
        config["final_evidence_config"], repo, label="GEM final-evidence config"
    )
    try:
        heldout_by_seed, final_sources = load_gem_final_evaluations(final_config, repo)
    except ValueError as error:
        raise GemTable9Error(str(error)) from error
    catalogue_path = resolve_pin(
        config["catalogue_adjudication"], repo, label="finite-catalogue adjudication"
    )
    catalogue_result = read_json_object(
        catalogue_path, error=GemTable9Error, label="finite-catalogue adjudication"
    )
    comparison = catalogue_result.get("finite_component_catalogue_comparison")
    catalogue = catalogue_result.get("finite_component_catalogue_arm_summary")
    if (
        catalogue_result.get("schema_version") != CATALOGUE_SCHEMA
        or catalogue_result.get("status") != "complete"
        or catalogue_result.get("candidate_selection") is not False
        or not isinstance(comparison, Mapping)
        or comparison.get("catalogue_arm") != "finite_component_catalogue_oracle"
        or not isinstance(catalogue, Mapping)
        or set(catalogue) != set(PROGRAMS)
    ):
        raise GemTable9Error("finite-catalogue evidence is inadmissible")

    metric_labels = (
        "Valid/1k",
        "Verified exact L1/1k",
        "Distinct exact L1/1k",
        "Verifier-recovered component-novel/1k",
        "Whole novelty",
        "Component novelty",
        r"Diversity/$N_{\mathrm{eff}}$",
    )
    columns: list[tuple[str, ...]] = []
    summaries: dict[str, Any] = {}
    for program in PROGRAMS:
        forge_rows = [seed[program] for seed in heldout_by_seed]
        forge_cells, catalogue_cells, program_summary = _metric_cells(
            program, forge_rows, catalogue[program]
        )
        columns.extend((forge_cells, catalogue_cells))
        summaries[program] = program_summary

    lines = []
    for label, cells in zip(metric_labels, zip(*columns), strict=True):
        rendered = [
            rf"\cellcolor{{forgerow}}{{{value}}}" if index % 2 == 0 else value
            for index, value in enumerate(cells)
        ]
        lines.append(" & ".join((label, *rendered)) + r" \\")
    atomic_write(row_path, ("\n".join(lines) + "\n\\hline\n").encode("utf-8"))

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "final_evidence_config": pin_record(final_config, repo),
        "sources": [*final_sources, pin_record(catalogue_path, repo)],
        "attempts_per_program_per_seed": ATTEMPTS_PER_PROGRAM,
        "training_seeds": list(EXPECTED_SEEDS),
        "candidate_selection": False,
        "summaries": summaries,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "gates": {
            "final_model_three_seed_evidence": len(heldout_by_seed) == 3,
            "fixed_catalogue_baseline_three_seed_evidence": True,
            "fixed_attempt_budget": True,
            "candidate_selection_absent": True,
            "strong_catalogue_escape_not_claimed": True,
        },
        "ignored_catalogue_artifact_fields": [
            "superseded forge arm summaries",
            "superseded forge-versus-catalogue paired intervals",
        ],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable9Error", "render_gem_table9_catalogue_comparison"]
