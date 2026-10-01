"""Render GEM Table 8 from the pinned three-seed mechanism study."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, iter_jsonl, read_json_object, write_json
from forge.reporting.results_v1 import MECHANISM_METRICS, MECHANISM_ROW_SCHEMA, mechanism_seed_rows

CONFIG_SCHEMA = "forge.gem_table8_architecture_ablations_config.v1"
RESULT_SCHEMA = "forge.gem_table8_architecture_ablations_render.v1"
LEDGER_SCHEMA = "forge.natbiotech_v1_result_rows.v1"
TRAINING_SCHEMA = "forge.synthesis_program_production_training_result.v1"
EVALUATION_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"
EXPECTED_SEEDS = (20260825, 20260826, 20260827)
ATTEMPTS_PER_PROGRAM = 3_072
ARM_ORDER = (
    "input_only_program",
    "no_role_loss",
    "no_core_loss",
    "no_routed_adapters",
    "no_gradient_conflict_control",
    "fact_matched",
    "fact_generous",
    "full_transformer",
)
ARM_NAMES = {
    "input_only_program": "Input-only program",
    "no_role_loss": "No role loss",
    "no_core_loss": "No core loss",
    "no_routed_adapters": "No routed adapters",
    "no_gradient_conflict_control": "No gradient conflict control",
    "fact_matched": "FACT-matched",
    "fact_generous": "FACT-generous",
    "full_transformer": "FORGE",
}
METRIC_DIGITS = {
    "held_family_loss": 2,
    "ugi_l1_per_1000": 1,
    "bl_l1_per_1000": 1,
    "lx_l1_per_1000": 1,
    "held_component_per_1000": 2,
    "cross_role_fidelity": 2,
    "diversity": 3,
}


class GemTable8Error(ValueError):
    """The mechanism-study evidence is incomplete, changed, or inadmissible."""


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GemTable8Error(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise GemTable8Error(f"{label} must be finite")
    return result


def _summary(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(EXPECTED_SEEDS):
        raise GemTable8Error("every Table 8 metric requires three independent training seeds")
    return {
        "by_seed": list(values),
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _tex_summary(summary: Mapping[str, Any], *, digits: int) -> str:
    return (
        rf"${_number(summary['mean'], label='mean'):.{digits}f}"
        rf"\pm{_number(summary['sample_sd'], label='sample sd'):.{digits}f}$"
    )


def _path_sha_pin(value: object, *, label: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise GemTable8Error(f"{label} lacks a path/SHA-256 pin")
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def _validate_training_results(
    pins: object,
    repo: Path,
    *,
    optimizer_steps: int,
    effective_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    if not isinstance(pins, list) or len(pins) != len(EXPECTED_SEEDS):
        raise GemTable8Error("Table 8 requires three pinned training results")
    sources: list[dict[str, Any]] = []
    training_sha_by_seed: dict[int, str] = {}
    reference_design_sha: str | None = None
    for index, pin in enumerate(pins):
        path = resolve_pin(pin, repo, label=f"mechanism training result {index}")
        result = read_json_object(path, error=GemTable8Error, label="mechanism training result")
        seed = result.get("seed")
        arms = result.get("arms")
        if (
            result.get("schema_version") != TRAINING_SCHEMA
            or result.get("status") != "pass"
            or result.get("profile") != "full"
            or seed not in EXPECTED_SEEDS
            or not isinstance(arms, Mapping)
            or set(arms) != set(ARM_ORDER)
        ):
            raise GemTable8Error(f"mechanism training result {index} is inadmissible")
        design = result.get("design")
        if not isinstance(design, Mapping) or not isinstance(design.get("sha256"), str):
            raise GemTable8Error("mechanism training design pin is missing")
        design_sha = str(design["sha256"])
        if reference_design_sha is None:
            reference_design_sha = design_sha
        elif design_sha != reference_design_sha:
            raise GemTable8Error("mechanism training design differs across seeds")
        for arm_id in ARM_ORDER:
            arm = arms[arm_id]
            if (
                not isinstance(arm, Mapping)
                or arm.get("optimizer_steps") != optimizer_steps
                or arm.get("effective_batch_size") != effective_batch_size
                or arm.get("examples_seen") != optimizer_steps * effective_batch_size
                or arm.get("candidate_selection") is not False
                or arm.get("route_calls") != 0
                or arm.get("oracle_calls") != 0
            ):
                raise GemTable8Error(f"training contract changed for {arm_id}, seed {seed}")
        if seed in training_sha_by_seed:
            raise GemTable8Error(f"duplicate mechanism training seed: {seed}")
        training_sha_by_seed[int(seed)] = str(pin["sha256"])
        sources.append(pin_record(path, repo))
    if tuple(sorted(training_sha_by_seed)) != EXPECTED_SEEDS:
        raise GemTable8Error("mechanism training seed set changed")
    return sources, training_sha_by_seed


def _validate_evaluation(
    path: Path,
    repo: Path,
    *,
    training_sha_by_seed: Mapping[int, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = read_json_object(path, error=GemTable8Error, label="mechanism evaluation result")
    seed = result.get("seed")
    selection = result.get("selection")
    gates = result.get("gates")
    training = result.get("training_result")
    if (
        result.get("schema_version") != EVALUATION_SCHEMA
        or result.get("status") != "pass"
        or result.get("profile") != "full"
        or result.get("execution_scope") != "production"
        or seed not in EXPECTED_SEEDS
        or result.get("terminal_decode_policy") != "strict_reaction_core_saturation_argmax"
        or not isinstance(selection, Mapping)
        or selection.get("candidate_selection") is not False
        or selection.get("checkpoint") != "fixed_final_step"
        or selection.get("calibration_selects_model") is not False
        or selection.get("heldout_selects_model_or_threshold") is not False
        or not isinstance(gates, Mapping)
        or not all(
            gates.get(name) is True
            for name in (
                "all_arms_evaluated",
                "all_fixed_checkpoints_evaluated",
                "candidate_selection_absent",
                "component_disjoint_metrics_complete",
                "coverage_and_precision_reported",
                "fixed_state_failures_zero",
                "heldout_is_nonselecting",
                "no_repairs_or_retries",
                "route_or_oracle_calls_zero",
                "terminal_decode_policy_recorded",
                "ugi_cross_role_fidelity_reported",
                "ugi_held_component_metrics_reported",
            )
        )
        or not isinstance(training, Mapping)
        or training.get("sha256") != training_sha_by_seed[int(seed)]
    ):
        raise GemTable8Error(f"mechanism evaluation for seed {seed} is inadmissible")
    checkpoints = result.get("checkpoint_metrics")
    if not isinstance(checkpoints, Mapping) or set(checkpoints) != set(ARM_ORDER):
        raise GemTable8Error(f"mechanism evaluation arms changed for seed {seed}")
    for arm_id in ARM_ORDER:
        arm_checkpoints = checkpoints[arm_id]
        if not isinstance(arm_checkpoints, Mapping) or not arm_checkpoints:
            raise GemTable8Error(f"mechanism checkpoints missing for {arm_id}, seed {seed}")
        final_step = str(max(int(step) for step in arm_checkpoints))
        heldout = arm_checkpoints[final_step].get("heldout")
        if not isinstance(heldout, Mapping) or len(heldout) != 3:
            raise GemTable8Error(f"held-out program set changed for {arm_id}, seed {seed}")
        if any(row.get("samples") != ATTEMPTS_PER_PROGRAM for row in heldout.values()):
            raise GemTable8Error(f"attempt budget changed for {arm_id}, seed {seed}")
    try:
        rows = mechanism_seed_rows(path, repo)
    except ValueError as error:
        raise GemTable8Error(str(error)) from error
    return rows, pin_record(path, repo)


def render_gem_table8_architecture_ablations(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Generate the matched eight-arm architecture table as mean plus or minus sample SD."""

    config = read_json_object(config_path, error=GemTable8Error, label="GEM Table 8 config")
    expected_fields = {
        "schema_version",
        "status",
        "mechanism_seed_rows",
        "mechanism_training_results",
        "expected_seeds",
        "arm_order",
        "attempts_per_program_per_seed",
        "optimizer_steps",
        "effective_batch_size",
        "candidate_selection",
    }
    if (
        set(config) != expected_fields
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_after_three_seed_core_saturation_evaluation"
        or config.get("expected_seeds") != list(EXPECTED_SEEDS)
        or config.get("arm_order") != list(ARM_ORDER)
        or config.get("attempts_per_program_per_seed") != ATTEMPTS_PER_PROGRAM
        or config.get("optimizer_steps") != 1_700
        or config.get("effective_batch_size") != 128
        or config.get("candidate_selection") is not False
    ):
        raise GemTable8Error("GEM Table 8 config changed")

    training_sources, training_sha_by_seed = _validate_training_results(
        config["mechanism_training_results"],
        repo,
        optimizer_steps=config["optimizer_steps"],
        effective_batch_size=config["effective_batch_size"],
    )
    ledger_path = resolve_pin(config["mechanism_seed_rows"], repo, label="mechanism seed rows")
    records = list(iter_jsonl(ledger_path))
    if len(records) != 1 + len(ARM_ORDER) * len(EXPECTED_SEEDS) or records[0] != {
        "rows": 24,
        "schema_version": LEDGER_SCHEMA,
    }:
        raise GemTable8Error("mechanism seed-row ledger shape changed")

    evaluation_cache: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    rows_by_arm: dict[str, dict[int, Mapping[str, Any]]] = {arm_id: {} for arm_id in ARM_ORDER}
    for record in records[1:]:
        if (
            not isinstance(record, Mapping)
            or set(record) != {"schema_version", "arm_id", "seed", "metrics", "source"}
            or record.get("schema_version") != MECHANISM_ROW_SCHEMA
            or record.get("arm_id") not in ARM_ORDER
            or record.get("seed") not in EXPECTED_SEEDS
            or not isinstance(record.get("metrics"), Mapping)
            or set(record["metrics"]) != set(MECHANISM_METRICS)
        ):
            raise GemTable8Error("mechanism seed row changed")
        arm_id = str(record["arm_id"])
        seed = int(record["seed"])
        if seed in rows_by_arm[arm_id]:
            raise GemTable8Error(f"duplicate mechanism row for {arm_id}, seed {seed}")
        source = _path_sha_pin(record["source"], label=f"mechanism evaluation {arm_id}/{seed}")
        source_path = resolve_pin(source, repo, label=f"mechanism evaluation {arm_id}/{seed}")
        source_key = str(source_path)
        if source_key not in evaluation_cache:
            evaluation_cache[source_key] = _validate_evaluation(
                source_path, repo, training_sha_by_seed=training_sha_by_seed
            )
        reconstructed = [
            row
            for row in evaluation_cache[source_key][0]
            if row["arm_id"] == arm_id and row["seed"] == seed
        ]
        if len(reconstructed) != 1:
            raise GemTable8Error(f"source evaluation lacks {arm_id}, seed {seed}")
        for metric in MECHANISM_METRICS:
            stored = _number(record["metrics"][metric], label=f"stored {arm_id}.{metric}")
            rebuilt = _number(
                reconstructed[0]["metrics"][metric], label=f"rebuilt {arm_id}.{metric}"
            )
            if not math.isclose(stored, rebuilt, rel_tol=0.0, abs_tol=1e-12):
                raise GemTable8Error(f"seed ledger drifted from source: {arm_id}.{metric}")
        rows_by_arm[arm_id][seed] = record

    if len(evaluation_cache) != len(EXPECTED_SEEDS):
        raise GemTable8Error("Table 8 does not resolve to exactly three evaluation sources")

    summaries: dict[str, dict[str, Any]] = {}
    lines: list[str] = []
    for arm_id in ARM_ORDER:
        if tuple(sorted(rows_by_arm[arm_id])) != EXPECTED_SEEDS:
            raise GemTable8Error(f"mechanism seed set changed for {arm_id}")
        arm_summary = {
            metric: _summary(
                [
                    _number(
                        rows_by_arm[arm_id][seed]["metrics"][metric],
                        label=f"{arm_id}.{seed}.{metric}",
                    )
                    for seed in EXPECTED_SEEDS
                ]
            )
            for metric in MECHANISM_METRICS
        }
        summaries[arm_id] = arm_summary
        cells = [
            ARM_NAMES[arm_id],
            *(
                _tex_summary(arm_summary[metric], digits=METRIC_DIGITS[metric])
                for metric in MECHANISM_METRICS
            ),
        ]
        if arm_id == "full_transformer":
            cells = [rf"\cellcolor{{forgerow}}{{{cell}}}" for cell in cells]
        if arm_id in {"fact_matched", "full_transformer"}:
            lines.append(r"\midrule")
        lines.append(" & ".join(cells) + r" \\")
    atomic_write(row_path, ("\n".join(lines) + "\n\\hline\n").encode("utf-8"))

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "sources": {
            "seed_rows": pin_record(ledger_path, repo),
            "training_results": training_sources,
            "evaluation_results": [value[1] for _, value in sorted(evaluation_cache.items())],
        },
        "training_seeds": list(EXPECTED_SEEDS),
        "arm_order": list(ARM_ORDER),
        "attempts_per_program_per_seed": ATTEMPTS_PER_PROGRAM,
        "optimizer_steps": config["optimizer_steps"],
        "effective_batch_size": config["effective_batch_size"],
        "candidate_selection": False,
        "summaries": summaries,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "gates": {
            "all_eight_arms_present": True,
            "three_independent_training_seeds": True,
            "seed_rows_reconstructed_from_source_evaluations": True,
            "training_contract_authenticated": True,
            "core_saturation_decoder_shared": True,
            "fixed_attempt_budget": True,
            "candidate_selection_absent": True,
            "route_and_oracle_calls_zero": True,
            "descriptive_sample_sd_not_confidence_interval": True,
        },
        "scope_note": (
            "The full_transformer row is the matched mechanism-study reference arm, not the "
            "independently trained production arm used in GEM Tables 1 and 9."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["GemTable8Error", "render_gem_table8_architecture_ablations"]
