"""Render GEM Table 1 from matched final-model and control evaluations."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json

FINAL_CONFIG_SCHEMA = "forge.gem_table1_final_evidence_config.v1"
TABLE_CONFIG_SCHEMA = "forge.gem_table1_complete_config.v1"
RESULT_SCHEMA = "forge.gem_table1_final_evidence_render.v1"
EVALUATION_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"
FINAL_ARM = "shared_bias_program_role_source"
CONTROL_ARMS = {
    "shared_null": "shared_three_program_null",
    "cyclic_program": "shared_three_program_program_id_cyclic",
}
FINAL_STEP = "9143"
EXPECTED_SEEDS = (20260825, 20260826, 20260827)
PROGRAMS = (
    "ugi_3cr_agile",
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
)
PROGRAM_NAMES = {
    "ugi_3cr_agile": "Ugi",
    "bl_2023_repeated_aza_michael": "Aza-Michael",
    "lx_2024_repeated_reductive_amination": "Reductive amination",
}


class GemTable1Error(ValueError):
    """Pinned final-model evidence is missing or inadmissible for GEM Table 1."""


def _all_true(value: object) -> bool:
    return (
        isinstance(value, Mapping) and bool(value) and all(item is True for item in value.values())
    )


def _metric(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise GemTable1Error(f"final evaluation has invalid {name}: {value!r}")
    return float(value)


def _mean_sd(values: Sequence[float]) -> tuple[float, float]:
    if len(values) != 3:
        raise GemTable1Error("Table 1 requires exactly three independent training seeds")
    return statistics.fmean(values), statistics.stdev(values)


def _mean_sd_tex(values: Sequence[float]) -> str:
    mean, sd = _mean_sd(values)
    return f"${mean:.1f}\\pm{sd:.1f}$"


def validate_gem_evaluation(
    document: Mapping[str, Any], *, expected_replicate: int, expected_arm: str
) -> Mapping[str, Mapping[str, Any]]:
    selection = document.get("selection")
    calls = document.get("calls")
    if (
        document.get("schema_version") != EVALUATION_SCHEMA
        or document.get("status") != "pass"
        or document.get("profile") != "full"
        or document.get("run_kind") != "production"
        or int(document.get("replicate", -1)) != expected_replicate
        or int(document.get("seed", -1)) != EXPECTED_SEEDS[expected_replicate]
        or document.get("terminal_decode_policy") != "strict_reaction_core_saturation_argmax"
        or set(document.get("checkpoint_metrics", {})) != {expected_arm}
        or not _all_true(document.get("gates"))
        or calls != {"oracle": 0, "route": 0}
        or not isinstance(selection, Mapping)
        or selection.get("candidate_selection") is not False
        or selection.get("heldout_selects_model_or_threshold") is not False
    ):
        raise GemTable1Error(
            f"seed replicate {expected_replicate} is not an admissible {expected_arm} evaluation"
        )
    checkpoints = document["checkpoint_metrics"][expected_arm]
    final = checkpoints.get(FINAL_STEP) if isinstance(checkpoints, Mapping) else None
    heldout = final.get("heldout") if isinstance(final, Mapping) else None
    if not isinstance(heldout, Mapping) or set(heldout) != set(PROGRAMS):
        raise GemTable1Error(f"seed replicate {expected_replicate} omits a final held-out program")
    for program in PROGRAMS:
        row = heldout[program]
        if not isinstance(row, Mapping) or int(row.get("samples", -1)) != 3072:
            raise GemTable1Error(
                f"seed replicate {expected_replicate} changed the denominator for {program}"
            )
        if row.get("exact_forward_replay_precision") != 1.0:
            raise GemTable1Error(
                f"seed replicate {expected_replicate} violates the replay invariant for {program}"
            )
        if "reductive_amination_substructure_hit_rate" in row:
            raise GemTable1Error(
                "forbidden reductive-amination substructure statistic was reported"
            )
    return heldout


def validate_gem_final_evaluation(
    document: Mapping[str, Any], *, expected_replicate: int
) -> Mapping[str, Mapping[str, Any]]:
    """Validate one conditioned-model evaluation retained by downstream tables."""

    return validate_gem_evaluation(
        document,
        expected_replicate=expected_replicate,
        expected_arm=FINAL_ARM,
    )


def load_gem_final_evaluations(
    config_path: Path, repo: Path
) -> tuple[list[Mapping[str, Mapping[str, Any]]], list[dict[str, Any]]]:
    """Load and validate the three pinned final conditioned-model evaluations."""

    config = read_json_object(config_path, error=GemTable1Error, label="GEM final-evidence config")
    pins = config.get("final_evaluations")
    if (
        config.get("schema_version") != FINAL_CONFIG_SCHEMA
        or not isinstance(pins, list)
        or len(pins) != 3
        or config.get("candidate_selection") is not False
    ):
        raise GemTable1Error("GEM final-evidence config is malformed")

    heldout_by_seed: list[Mapping[str, Mapping[str, Any]]] = []
    sources: list[dict[str, Any]] = []
    for replicate, pin in enumerate(pins):
        if not isinstance(pin, Mapping):
            raise GemTable1Error(f"final evaluation pin {replicate} is malformed")
        path = resolve_pin(pin, repo, label=f"final evaluation seed replicate {replicate}")
        document = read_json_object(
            path, error=GemTable1Error, label=f"final evaluation seed replicate {replicate}"
        )
        heldout_by_seed.append(
            validate_gem_final_evaluation(document, expected_replicate=replicate)
        )
        sources.append(pin_record(path, repo))
    return heldout_by_seed, sources


def load_gem_table1_evaluations(
    config_path: Path, repo: Path
) -> tuple[dict[str, list[Mapping[str, Mapping[str, Any]]]], list[dict[str, Any]]]:
    """Load the matched conditioned, null and cyclic three-seed evaluations."""

    config = read_json_object(config_path, error=GemTable1Error, label="GEM Table 1 config")
    controls = config.get("control_evaluations")
    if (
        config.get("schema_version") != TABLE_CONFIG_SCHEMA
        or config.get("candidate_selection") is not False
        or config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or not isinstance(controls, Mapping)
        or set(controls) != set(CONTROL_ARMS)
    ):
        raise GemTable1Error("GEM Table 1 config is malformed")

    final_pin = config.get("final_evidence_config")
    if not isinstance(final_pin, Mapping):
        raise GemTable1Error("GEM Table 1 final-evidence config pin is missing")
    final_config_path = resolve_pin(final_pin, repo, label="GEM Table 1 final-evidence config")
    final_rows, final_sources = load_gem_final_evaluations(final_config_path, repo)
    rows_by_arm: dict[str, list[Mapping[str, Mapping[str, Any]]]] = {"conditioned": final_rows}
    sources = [pin_record(final_config_path, repo), *final_sources]

    for control_name, expected_arm in CONTROL_ARMS.items():
        pins = controls[control_name]
        if not isinstance(pins, list) or len(pins) != len(EXPECTED_SEEDS):
            raise GemTable1Error(f"{control_name} requires exactly three pinned evaluations")
        control_rows: list[Mapping[str, Mapping[str, Any]]] = []
        for replicate, pin in enumerate(pins):
            if not isinstance(pin, Mapping):
                raise GemTable1Error(f"{control_name} replicate {replicate} pin is malformed")
            path = resolve_pin(pin, repo, label=f"{control_name} seed replicate {replicate}")
            document = read_json_object(
                path,
                error=GemTable1Error,
                label=f"{control_name} seed replicate {replicate}",
            )
            control_rows.append(
                validate_gem_evaluation(
                    document,
                    expected_replicate=replicate,
                    expected_arm=expected_arm,
                )
            )
            sources.append(pin_record(path, repo))
        rows_by_arm[control_name] = control_rows
    return rows_by_arm, sources


def _macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}"


def _paired_summary(conditioned: Sequence[float], control: Sequence[float]) -> dict[str, Any]:
    if len(conditioned) != len(control) or len(conditioned) != len(EXPECTED_SEEDS):
        raise GemTable1Error("paired Table 1 contrast changed its seed count")
    differences = [left - right for left, right in zip(conditioned, control, strict=True)]
    return {
        "by_seed_percentage_points": differences,
        "mean_percentage_points": statistics.fmean(differences),
        "range_percentage_points": [min(differences), max(differences)],
    }


def render_gem_table1_final_evidence(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render the complete three-arm Table 1 from nine pinned production evaluations."""

    heldout_by_arm, sources = load_gem_table1_evaluations(config_path, repo)

    summaries: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in heldout_by_arm}
    paired_contrasts: dict[str, dict[str, Any]] = {
        "conditioned_minus_shared_null": {},
        "conditioned_minus_cyclic_program": {},
    }
    rows = []
    macros: dict[str, str] = {}
    program_macro_names = {
        "ugi_3cr_agile": "Ugi",
        "bl_2023_repeated_aza_michael": "BL",
        "lx_2024_repeated_reductive_amination": "LX",
    }
    arm_macro_names = {
        "conditioned": "Conditioned",
        "shared_null": "Null",
        "cyclic_program": "Cyclic",
    }
    for program in PROGRAMS:
        yields_by_arm = {
            arm: [
                100.0 * _metric(seed[program], "exact_l1_yield_per_attempt") for seed in seed_rows
            ]
            for arm, seed_rows in heldout_by_arm.items()
        }
        coverage = [
            100.0 * _metric(seed[program], "exact_l1_decomposition_coverage")
            for seed in heldout_by_arm["conditioned"]
        ]
        replay = [
            100.0 * _metric(seed[program], "exact_forward_replay_precision")
            for seed in heldout_by_arm["conditioned"]
        ]
        ambiguity = [
            100.0 * _metric(seed[program], "decomposition_ambiguity_fraction")
            for seed in heldout_by_arm["conditioned"]
        ]
        for arm, yields in yields_by_arm.items():
            yield_mean, yield_sd = _mean_sd(yields)
            summaries[arm][program] = {
                "exact_l1_yield_percent": {
                    "by_seed": yields,
                    "mean": yield_mean,
                    "sample_sd": yield_sd,
                }
            }
            macro_prefix = f"ForgeTableOne{arm_macro_names[arm]}{program_macro_names[program]}"
            macros[f"{macro_prefix}Yield"] = _mean_sd_tex(yields)
        summaries["conditioned"][program].update(
            {
                "decomposition_coverage_percent": {
                    "by_seed": coverage,
                    "mean": statistics.fmean(coverage),
                },
                "replay_invariant_percent": {
                    "by_seed": replay,
                    "mean": statistics.fmean(replay),
                },
                "exact_decomposition_ambiguity_percent": {
                    "by_seed": ambiguity,
                    "mean": statistics.fmean(ambiguity),
                },
            }
        )
        macros[f"ForgeTableOne{program_macro_names[program]}AmbiguityPercent"] = (
            f"{statistics.fmean(ambiguity):.1f}"
        )
        for control_name, contrast_name in (
            ("shared_null", "conditioned_minus_shared_null"),
            ("cyclic_program", "conditioned_minus_cyclic_program"),
        ):
            contrast = _paired_summary(yields_by_arm["conditioned"], yields_by_arm[control_name])
            paired_contrasts[contrast_name][program] = contrast
            control_macro = "Null" if control_name == "shared_null" else "Cyclic"
            prefix = f"ForgeTableOne{control_macro}{program_macro_names[program]}"
            macros[f"{prefix}DifferencePP"] = f"{contrast['mean_percentage_points']:.1f}"
            macros[f"{prefix}RangeLowPP"] = f"{contrast['range_percentage_points'][0]:.1f}"
            macros[f"{prefix}RangeHighPP"] = f"{contrast['range_percentage_points'][1]:.1f}"
        rows.append(
            " & ".join(
                (
                    PROGRAM_NAMES[program],
                    _mean_sd_tex(yields_by_arm["conditioned"]),
                    _mean_sd_tex(yields_by_arm["shared_null"]),
                    _mean_sd_tex(yields_by_arm["cyclic_program"]),
                    f"{statistics.fmean(coverage):.1f}/{statistics.fmean(replay):.1f}",
                )
            )
            + r" \\"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "shared_program_figure_rows.tex"
    atomic_write(rows_path, ("\n".join(rows) + "\n").encode("utf-8"))
    macros_path = output_dir / "gem_table1_macros.tex"
    atomic_write(
        macros_path,
        ("\n".join(_macro(name, macros[name]) for name in sorted(macros)) + "\n").encode("utf-8"),
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "candidate_selection": False,
        "final_checkpoint_step": int(FINAL_STEP),
        "attempts_per_program_per_seed": 3072,
        "training_seeds": list(EXPECTED_SEEDS),
        "summaries": summaries,
        "paired_contrasts": paired_contrasts,
        "pending_columns": [],
        "sources": sources,
        "artifacts": {
            "table_rows": artifact_record(rows_path, logical_path=rows_path.name),
            "macros": artifact_record(macros_path, logical_path=macros_path.name),
        },
        "nonclaims": [
            "Replay is a verifier invariant, not learned performance.",
            "Three training seeds support descriptive summaries, not population confidence intervals.",
            "The cyclic control does not isolate every program coordinate independently.",
        ],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = [
    "GemTable1Error",
    "load_gem_table1_evaluations",
    "load_gem_final_evaluations",
    "render_gem_table1_final_evidence",
    "validate_gem_evaluation",
    "validate_gem_final_evaluation",
]
