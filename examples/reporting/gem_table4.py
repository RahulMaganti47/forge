"""Render GEM Table 4 from the matched three-arm production evaluations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from examples.reporting.gem_table1 import (
    EXPECTED_SEEDS,
    FINAL_STEP,
    PROGRAMS,
    load_gem_table1_evaluations,
)
from forge.core.hashing import artifact_record, pin_record
from forge.core.io import atomic_write, write_json

RESULT_SCHEMA = "forge.gem_table4_production_comparison_render.v1"
ARMS = ("conditioned", "shared_null", "cyclic_program")
ATTEMPTS_PER_PROGRAM = 3072


class GemTable4Error(ValueError):
    """Matched final-model evidence cannot support GEM Table 4."""


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GemTable4Error(f"{label} is not finite")
    return float(value)


def _mean(rows: Sequence[Mapping[str, Any]], metric: str, *, label: str) -> float:
    if len(rows) != len(EXPECTED_SEEDS):
        raise GemTable4Error(f"{label} does not have three training-seed rows")
    return statistics.fmean(_number(row.get(metric), label=f"{label}.{metric}") for row in rows)


def _mean_sd(rows: Sequence[Mapping[str, Any]], metric: str, *, label: str) -> tuple[float, float]:
    values = [100.0 * _number(row.get(metric), label=f"{label}.{metric}") for row in rows]
    if len(values) != len(EXPECTED_SEEDS):
        raise GemTable4Error(f"{label} does not have three training-seed rows")
    return statistics.fmean(values), statistics.stdev(values)


def _cells(rows: Sequence[Mapping[str, Any]], *, label: str) -> tuple[str, ...]:
    exact_mean, exact_sd = _mean_sd(rows, "exact_l1_yield_per_attempt", label=label)
    return (
        f"{100.0 * _mean(rows, 'raw_valid_fraction', label=label):.1f}/"
        f"{100.0 * _mean(rows, 'connected_fraction', label=label):.1f}",
        f"{100.0 * _mean(rows, 'exact_l1_decomposition_coverage', label=label):.1f}/"
        f"{100.0 * _mean(rows, 'exact_forward_replay_precision', label=label):.1f}",
        f"{100.0 * _mean(rows, 'decomposition_abstention_fraction', label=label):.1f}/"
        f"{100.0 * _mean(rows, 'decomposition_ambiguity_fraction', label=label):.1f}",
        rf"${exact_mean:.1f}\pm{exact_sd:.1f}$",
        f"{100.0 * _mean(rows, 'whole_lipid_novelty_fraction', label=label):.1f}/"
        f"{100.0 * _mean(rows, 'component_novelty_fraction', label=label):.1f}",
        f"{_mean(rows, 'internal_diversity', label=label):.3f}/"
        f"{_mean(rows, 'effective_component_count', label=label):.1f}",
    )


def render_gem_table4_production_comparison(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render the complete transposed production table from pinned seed evaluations."""

    try:
        heldout_by_arm, sources = load_gem_table1_evaluations(config_path, repo)
    except ValueError as error:
        raise GemTable4Error(str(error)) from error

    columns: list[tuple[str, ...]] = []
    records: dict[str, dict[str, dict[str, str]]] = {program: {} for program in PROGRAMS}
    for program in PROGRAMS:
        for arm in ARMS:
            program_rows = [seed[program] for seed in heldout_by_arm[arm]]
            if any(row.get("samples") != ATTEMPTS_PER_PROGRAM for row in program_rows):
                raise GemTable4Error(f"attempt denominator changed for {program}/{arm}")
            cells = _cells(program_rows, label=f"{program}.{arm}")
            columns.append(cells)
            records[program][arm] = {
                label: value
                for label, value in zip(
                    (
                        "valid_connected_percent",
                        "decomposition_replay_percent",
                        "abstention_ambiguity_percent",
                        "exact_l1_yield_percent",
                        "whole_component_novelty_percent",
                        "diversity_effective_component_count",
                    ),
                    cells,
                    strict=True,
                )
            }

    row_labels = (
        "Valid./Conn.",
        "Decomp./Replay",
        "Abst./Ambig.",
        "Exact-L1/attempt",
        "Whole/Comp. novelty",
        r"Diversity/$N_{\mathrm{eff}}$",
    )
    lines: list[str] = []
    for row_index, values in enumerate(zip(*columns, strict=True)):
        rendered = []
        for column_index, value in enumerate(values):
            if column_index % len(ARMS) == 0:
                rendered.append(rf"\cellcolor{{forgerow}} {value}")
            else:
                rendered.append(value)
        lines.append(f"{row_labels[row_index]} & " + " & ".join(rendered) + " \\\\")

    row_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(row_path, ("\n".join(lines) + "\n").encode("utf-8"))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "checkpoint_step": int(FINAL_STEP),
        "training_seeds": list(EXPECTED_SEEDS),
        "attempts_per_program_per_seed": ATTEMPTS_PER_PROGRAM,
        "candidate_selection": False,
        "arm_order": list(ARMS),
        "program_order": list(PROGRAMS),
        "records": records,
        "sources": sources,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "gates": {
            "three_independent_training_seeds_per_arm": all(
                len(heldout_by_arm[arm]) == len(EXPECTED_SEEDS) for arm in ARMS
            ),
            "all_nine_method_program_cells_defined": len(columns) == 9,
            "fixed_attempt_denominator": True,
            "candidate_selection_absent": True,
            "unmatched_ugi_only_column_absent": True,
        },
        "nonclaims": [
            "Mean plus or minus sample standard deviation is descriptive across three seeds.",
            "Replay is a verifier invariant, not learned performance.",
            "The cyclic control does not isolate every program coordinate independently.",
        ],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable4Error", "render_gem_table4_production_comparison"]
