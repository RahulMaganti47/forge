"""Render GEM Table 6 from the final three-seed conditioned-model evaluations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from examples.reporting.gem_table1 import (
    EXPECTED_SEEDS,
    FINAL_STEP,
    PROGRAM_NAMES,
    PROGRAMS,
    load_gem_final_evaluations,
)
from forge.core.hashing import artifact_record, pin_record
from forge.core.io import atomic_write, write_json

RESULT_SCHEMA = "forge.gem_table6_exact_l1_counts_render.v1"
ATTEMPTS_PER_PROGRAM = 3072


class GemTable6Error(ValueError):
    """Final-model evidence cannot support the requested exact-count table."""


def _probability(row: Mapping[str, Any], *, program: str, replicate: int) -> float:
    value = row.get("exact_l1_yield_per_attempt")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise GemTable6Error(
            f"invalid exact-L1 yield for {program} seed replicate {replicate}: {value!r}"
        )
    return float(value)


def render_gem_table6_exact_l1_counts(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Generate exact numerators, denominators and percentages for GEM Table 6."""

    try:
        heldout_by_seed, sources = load_gem_final_evaluations(config_path, repo)
    except ValueError as error:
        raise GemTable6Error(str(error)) from error

    rows: list[str] = []
    records: list[dict[str, Any]] = []
    for program in PROGRAMS:
        for replicate, seed_metrics in enumerate(heldout_by_seed):
            metric = seed_metrics[program]
            attempts = metric.get("samples")
            if attempts != ATTEMPTS_PER_PROGRAM:
                raise GemTable6Error(
                    f"attempt denominator changed for {program} seed replicate {replicate}"
                )
            yield_value = _probability(metric, program=program, replicate=replicate)
            exact_count = round(yield_value * ATTEMPTS_PER_PROGRAM)
            if not math.isclose(
                exact_count / ATTEMPTS_PER_PROGRAM,
                yield_value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise GemTable6Error(
                    f"yield does not encode an integer count for {program} seed replicate {replicate}"
                )
            rows.append(
                " & ".join(
                    (
                        PROGRAM_NAMES[program],
                        str(replicate),
                        f"{exact_count:,}".replace(",", "{,}"),
                        f"{ATTEMPTS_PER_PROGRAM:,}".replace(",", "{,}"),
                        f"{100.0 * yield_value:.2f}",
                    )
                )
                + r" \\"
            )
            records.append(
                {
                    "program_id": program,
                    "program_label": PROGRAM_NAMES[program],
                    "replicate": replicate,
                    "training_seed": EXPECTED_SEEDS[replicate],
                    "verified_exact_l1_count": exact_count,
                    "attempts": ATTEMPTS_PER_PROGRAM,
                    "yield_percent": f"{100.0 * yield_value:.2f}",
                }
            )

    # Keep the terminal rule inside the included fragment.  TeX rejects a
    # \noalign-based rule placed immediately after an alignment-ending \input.
    atomic_write(row_path, ("\n".join(rows) + "\n\\hline\n").encode("utf-8"))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "sources": sources,
        "checkpoint_step": int(FINAL_STEP),
        "attempts_per_program_per_seed": ATTEMPTS_PER_PROGRAM,
        "candidate_selection": False,
        "rows": records,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "gates": {
            "three_independent_training_seeds": len(heldout_by_seed) == 3,
            "all_rows_have_fixed_denominator": len(records) == 9,
            "counts_reconstruct_reported_yields": True,
            "candidate_selection_absent": True,
        },
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable6Error", "render_gem_table6_exact_l1_counts"]
