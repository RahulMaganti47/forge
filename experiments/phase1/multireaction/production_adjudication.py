"""Aggregate the three matched production replicates without selecting a checkpoint.

The production DAG intentionally emits one independently verifiable run per replicate.  This
module is the cross-run boundary: it verifies those runs, binds them to one executable source and
one frozen design, and applies the design's paired-seed non-inferiority rule.  Generated molecules
are not treated as independent experimental replicates; the paired training seed is the resampling
unit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments._runtime import verify_run_directory
from experiments._runtime.source import source_fingerprint
from forge.core.hashing import pin_record, resolve_pin, sha256_bytes, sha256_file
from forge.core.io import atomic_write, iter_jsonl, read_json_object, write_json

from .production_evaluation import RESULT_SCHEMA as EVALUATION_RESULT_SCHEMA
from .production_training import RESULT_SCHEMA as TRAINING_RESULT_SCHEMA

ADJUDICATION_SCHEMA = "forge.synthesis_program_production_adjudication.v1"
TRANSFORMER_EXPERIMENT_IDS = frozenset(
    {
        "phase1-transformer-synthesis-program-production",
        "phase1-transformer-synthesis-program-production-h100",
    }
)
TRAINING_IMPLEMENTATION = "model.shared-synthesis-program-production-training.v1"
EVALUATION_IMPLEMENTATION = "model.shared-synthesis-program-production-evaluation.v1"
CATALOGUE_EXPERIMENT_ID = "phase1-finite-component-catalogue-baseline"
CATALOGUE_IMPLEMENTATION = "model.finite-component-catalogue-baseline.v1"
CATALOGUE_RESULT_SCHEMA = "forge.finite_component_catalogue_baseline_result.v1"
CATALOGUE_SAMPLES_SCHEMA = "forge.finite_component_catalogue_baseline_samples.v1"
UGI_PROGRAM = "ugi_3cr_agile"
CONDITIONED_ARM = "shared_three_program_conditioned"
POSTHOC_ARM = "shared_three_program_null"
CYCLIC_ARM = "shared_three_program_program_id_cyclic"


class SynthesisProgramProductionAdjudicationError(ValueError):
    """The run set cannot support the frozen production comparison."""


def _artifact_path(
    run_dir: Path,
    stage_id: str,
    manifest: Mapping[str, Any],
    label: str,
) -> Path:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or label not in artifacts:
        raise SynthesisProgramProductionAdjudicationError(
            f"{stage_id} stage omits required artifact {label!r}"
        )
    record = artifacts[label]
    if not isinstance(record, Mapping):
        raise SynthesisProgramProductionAdjudicationError(
            f"{stage_id}.{label} artifact record is malformed"
        )
    path = (run_dir / "stages" / stage_id / str(record.get("path", ""))).resolve()
    try:
        path.relative_to((run_dir / "stages" / stage_id).resolve())
    except ValueError as error:
        raise SynthesisProgramProductionAdjudicationError(
            f"{stage_id}.{label} artifact escapes its stage directory"
        ) from error
    if not path.is_file() or str(sha256_file(path)) != record.get("sha256"):
        raise SynthesisProgramProductionAdjudicationError(
            f"{stage_id}.{label} artifact is missing or changed"
        )
    return path


def _load_production_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="production run manifest",
    )
    if (
        run.get("experiment_id") not in TRANSFORMER_EXPERIMENT_IDS
        or run.get("profile") != "full"
        or run.get("status") != "complete"
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "adjudication requires a complete full-profile Transformer production run"
        )
    stages = run.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != {"training", "evaluation"}:
        raise SynthesisProgramProductionAdjudicationError(
            "production run must contain exactly training and evaluation stages"
        )
    manifests: dict[str, dict[str, Any]] = {}
    for stage_id, implementation in (
        ("training", TRAINING_IMPLEMENTATION),
        ("evaluation", EVALUATION_IMPLEMENTATION),
    ):
        manifest = read_json_object(
            run_dir / "stages" / stage_id / "manifest.json",
            error=SynthesisProgramProductionAdjudicationError,
            label=f"production {stage_id} manifest",
        )
        if manifest.get("implementation") != implementation:
            raise SynthesisProgramProductionAdjudicationError(
                f"production {stage_id} used the wrong implementation"
            )
        manifests[stage_id] = manifest

    training_path = _artifact_path(run_dir, "training", manifests["training"], "result")
    evaluation_path = _artifact_path(run_dir, "evaluation", manifests["evaluation"], "result")
    samples_path = _artifact_path(run_dir, "evaluation", manifests["evaluation"], "samples")
    training = read_json_object(
        training_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="production training result",
    )
    evaluation = read_json_object(
        evaluation_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="production evaluation result",
    )
    replicate = run.get("replicate")
    if (
        training.get("schema_version") != TRAINING_RESULT_SCHEMA
        or evaluation.get("schema_version") != EVALUATION_RESULT_SCHEMA
        or training.get("profile") != "full"
        or evaluation.get("profile") != "full"
        or training.get("replicate") != replicate
        or evaluation.get("replicate") != replicate
        or training.get("seed") != evaluation.get("seed")
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "production training/evaluation identity is inconsistent"
        )
    gates = evaluation.get("gates")
    if (
        evaluation.get("status") != "pass"
        or not isinstance(gates, Mapping)
        or not gates
        or not all(value is True for value in gates.values())
    ):
        raise SynthesisProgramProductionAdjudicationError(
            f"replicate {replicate} did not pass its evaluation gates"
        )
    training_inputs = manifests["training"].get("external_inputs")
    evaluation_inputs = manifests["evaluation"].get("external_inputs")
    if not isinstance(training_inputs, Mapping) or not isinstance(evaluation_inputs, Mapping):
        raise SynthesisProgramProductionAdjudicationError(
            "production stages have no pinned external inputs"
        )
    training_design_pin = training_inputs.get("production_design")
    evaluation_design_pin = evaluation_inputs.get("production_design")
    if (
        not isinstance(training_design_pin, Mapping)
        or not isinstance(evaluation_design_pin, Mapping)
        or dict(training_design_pin) != dict(evaluation_design_pin)
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "production stages do not share one production-design pin"
        )
    if (
        training.get("gates", {}).get("candidate_selection_absent") is not True
        or training.get("gates", {}).get("route_or_oracle_calls_zero") is not True
        or evaluation.get("gates", {}).get("candidate_selection_absent") is not True
        or evaluation.get("gates", {}).get("route_or_oracle_calls_zero") is not True
        or evaluation.get("calls") != {"oracle": 0, "route": 0}
        or evaluation.get("selection", {}).get("candidate_selection") is not False
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "production run performed or failed to exclude route, oracle or candidate selection"
        )
    return {
        "run_dir": run_dir,
        "run": run,
        "manifests": manifests,
        "training_path": training_path,
        "evaluation_path": evaluation_path,
        "samples_path": samples_path,
        "training": training,
        "evaluation": evaluation,
        "design_pin": dict(evaluation_design_pin),
    }


def _load_catalogue_run(run_dir: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    verify_run_directory(run_dir)
    run = read_json_object(
        run_dir / "run.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="catalogue baseline run manifest",
    )
    if (
        run.get("experiment_id") != CATALOGUE_EXPERIMENT_ID
        or run.get("profile") != "full"
        or run.get("status") != "complete"
        or set(run.get("stages", {})) != {"catalogue"}
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "catalogue comparison requires a complete full-profile catalogue run"
        )
    manifest = read_json_object(
        run_dir / "stages/catalogue/manifest.json",
        error=SynthesisProgramProductionAdjudicationError,
        label="catalogue stage manifest",
    )
    if manifest.get("implementation") != CATALOGUE_IMPLEMENTATION:
        raise SynthesisProgramProductionAdjudicationError(
            "catalogue run used the wrong implementation"
        )
    result_path = _artifact_path(run_dir, "catalogue", manifest, "result")
    samples_path = _artifact_path(run_dir, "catalogue", manifest, "samples")
    result = read_json_object(
        result_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="catalogue baseline result",
    )
    result_gates = result.get("gates")
    if (
        result.get("schema_version") != CATALOGUE_RESULT_SCHEMA
        or result.get("status") != "pass"
        or result.get("profile") != "full"
        or result.get("replicate") != run.get("replicate")
        or not isinstance(result_gates, Mapping)
        or not result_gates
        or not all(result_gates.values())
        or result.get("calls") != {"oracle": 0, "route": 0}
        or result.get("candidate_selection") is not False
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "catalogue baseline result did not pass its frozen gates"
        )
    header = next(iter_jsonl(samples_path), None)
    if not isinstance(header, Mapping) or header.get("schema_version") != CATALOGUE_SAMPLES_SCHEMA:
        raise SynthesisProgramProductionAdjudicationError(
            "catalogue baseline sample ledger has an unsupported schema"
        )
    inputs = manifest.get("external_inputs")
    design_pin = inputs.get("production_design") if isinstance(inputs, Mapping) else None
    if not isinstance(design_pin, Mapping):
        raise SynthesisProgramProductionAdjudicationError(
            "catalogue baseline has no production-design pin"
        )
    return {
        "run_dir": run_dir,
        "run": run,
        "manifest": manifest,
        "result_path": result_path,
        "samples_path": samples_path,
        "result": result,
        "design_pin": dict(design_pin),
    }


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SynthesisProgramProductionAdjudicationError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise SynthesisProgramProductionAdjudicationError(f"{label} is not finite")
    return result


def _optional_number(value: Any, *, label: str) -> float | None:
    """Validate a metric cell while preserving a scientifically undefined null value."""

    if value is None:
        return None
    return _number(value, label=label)


def _paired_descriptive_comparison(
    left: Sequence[float | None],
    right: Sequence[float | None],
    *,
    left_label: str,
    right_label: str,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Describe paired seed cells without imputing metrics undefined after zero survivors.

    Arm-specific means retain every defined metric cell.  A paired difference and its interval are
    reported only when every frozen seed pair is defined; complete-case deletion would otherwise
    turn a zero-survivor failure into an apparently better-conditioned comparison.
    """

    if len(left) != len(right) or len(left) < 2:
        raise SynthesisProgramProductionAdjudicationError(
            "paired descriptive comparison requires aligned vectors from at least two seeds"
        )
    left_values = list(left)
    right_values = list(right)
    defined_left = [value for value in left_values if value is not None]
    defined_right = [value for value in right_values if value is not None]
    complete_pairs = [
        (left_value, right_value)
        for left_value, right_value in zip(left_values, right_values, strict=True)
        if left_value is not None and right_value is not None
    ]
    all_pairs_defined = len(complete_pairs) == len(left_values)
    result: dict[str, Any] = {
        f"{left_label}_by_seed": left_values,
        f"{right_label}_by_seed": right_values,
        f"{left_label}_mean": float(np.mean(defined_left)) if defined_left else None,
        f"{right_label}_mean": float(np.mean(defined_right)) if defined_right else None,
        f"{left_label}_defined_seed_count": len(defined_left),
        f"{right_label}_defined_seed_count": len(defined_right),
        "paired_defined_seed_count": len(complete_pairs),
        "expected_seed_pair_count": len(left_values),
        "all_seed_pairs_defined": all_pairs_defined,
        f"{left_label}_minus_{right_label}_mean": None,
        "resampling_unit": "paired_independent_seed_metric_cell",
        "paired_seed_difference_interval": None,
        "undefined_cells_imputed": False,
    }
    if all_pairs_defined:
        finite_left = [float(value) for value in left_values if value is not None]
        finite_right = [float(value) for value in right_values if value is not None]
        result[f"{left_label}_minus_{right_label}_mean"] = float(
            np.mean(np.asarray(finite_left) - np.asarray(finite_right))
        )
        result["paired_seed_difference_interval"] = paired_seed_difference_interval(
            finite_left,
            finite_right,
            resamples=resamples,
            seed=seed,
            confidence_level=confidence_level,
        )
    else:
        result["paired_difference_unavailable_reason"] = (
            "one_or_more_frozen_seed_cells_are_undefined; no imputation or complete-case "
            "bootstrap was performed"
        )
    return result


def _final_ugi_metrics(
    evaluation: Mapping[str, Any], arm_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoints = evaluation.get("checkpoint_metrics", {}).get(arm_id)
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise SynthesisProgramProductionAdjudicationError(
            f"evaluation omits checkpoint metrics for {arm_id}"
        )
    try:
        final_step = max(int(step) for step in checkpoints)
        heldout = checkpoints[str(final_step)]["heldout"][UGI_PROGRAM]
        component = evaluation["component_disjoint_metrics"][arm_id][UGI_PROGRAM]
    except (KeyError, TypeError, ValueError) as error:
        raise SynthesisProgramProductionAdjudicationError(
            f"evaluation omits final heldout Ugi metrics for {arm_id}"
        ) from error
    if not isinstance(heldout, dict) or not isinstance(component, dict):
        raise SynthesisProgramProductionAdjudicationError(
            f"evaluation has malformed final Ugi metrics for {arm_id}"
        )
    return heldout, component


def _nearest_rank(values: np.ndarray, probability: float) -> float:
    if values.ndim != 1 or not len(values) or not 0.0 < probability < 1.0:
        raise SynthesisProgramProductionAdjudicationError("invalid bootstrap quantile request")
    ordered = np.sort(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return float(ordered[index])


def paired_seed_bootstrap(
    baseline: Sequence[float],
    challenger: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
    ratio: bool = False,
) -> dict[str, Any]:
    """Bootstrap paired independent-seed metric cells with no molecule pseudoreplication."""

    left = np.asarray(baseline, dtype=np.float64)
    right = np.asarray(challenger, dtype=np.float64)
    if (
        left.ndim != 1
        or right.shape != left.shape
        or len(left) < 2
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "paired bootstrap requires finite aligned vectors from at least two seeds"
        )
    if isinstance(resamples, bool) or resamples < 1_000:
        raise SynthesisProgramProductionAdjudicationError(
            "paired bootstrap requires at least 1,000 resamples"
        )
    if not 0.5 < confidence_level < 1.0:
        raise SynthesisProgramProductionAdjudicationError(
            "paired bootstrap confidence level must lie between 0.5 and 1"
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(left), size=(resamples, len(left)), endpoint=False)
    baseline_means = np.mean(left[draws], axis=1)
    challenger_means = np.mean(right[draws], axis=1)
    if ratio:
        if np.any(baseline_means <= 0.0) or float(np.mean(left)) <= 0.0:
            raise SynthesisProgramProductionAdjudicationError(
                "ratio bootstrap requires positive baseline values"
            )
        samples = challenger_means / baseline_means
        point = float(np.mean(right) / np.mean(left))
        one_sided_bound = _nearest_rank(samples, 1.0 - confidence_level)
        direction = "lower"
    else:
        samples = baseline_means - challenger_means
        point = float(np.mean(left) - np.mean(right))
        one_sided_bound = _nearest_rank(samples, confidence_level)
        direction = "upper"
    return {
        "baseline_by_seed": left.tolist(),
        "challenger_by_seed": right.tolist(),
        "point_estimate": point,
        "one_sided_bound": one_sided_bound,
        "bound_direction": direction,
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "draw_indices_sha256": str(sha256_bytes(draws.tobytes(order="C"))),
        "resampling_unit": "paired_independent_training_seed_metric_cell",
        "molecule_rows_treated_as_independent_replicates": False,
        "quantile": "nearest_rank",
    }


def paired_seed_difference_interval(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Return a two-sided paired-seed interval for a descriptive metric difference."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 1
        or right_values.shape != left_values.shape
        or len(left_values) < 2
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "paired difference interval requires finite aligned vectors from at least two seeds"
        )
    if isinstance(resamples, bool) or resamples < 1_000:
        raise SynthesisProgramProductionAdjudicationError(
            "paired difference interval requires at least 1,000 resamples"
        )
    if not 0.5 < confidence_level < 1.0:
        raise SynthesisProgramProductionAdjudicationError(
            "paired difference confidence level must lie between 0.5 and 1"
        )
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(left_values), size=(resamples, len(left_values)), endpoint=False)
    differences = left_values - right_values
    samples = np.mean(differences[draws], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return {
        "point_estimate": float(np.mean(differences)),
        "lower_bound": _nearest_rank(samples, alpha),
        "upper_bound": _nearest_rank(samples, 1.0 - alpha),
        "confidence_level": confidence_level,
        "resamples": resamples,
        "seed": seed,
        "draw_indices_sha256": str(sha256_bytes(draws.tobytes(order="C"))),
        "resampling_unit": "paired_independent_seed_metric_cell",
        "molecule_rows_treated_as_independent_replicates": False,
        "quantile": "nearest_rank",
    }


def _adjudicate_retention(
    evaluations: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    retention = design.get("ugi_retention")
    if not isinstance(retention, Mapping):
        raise SynthesisProgramProductionAdjudicationError(
            "production design omits the Ugi retention contract"
        )
    baseline_arm = str(retention.get("baseline_arm"))
    challenger_arm = str(retention.get("challenger_arm"))
    decision = retention.get("decision_rule")
    margins = retention.get("relative_noninferiority_gates")
    absolute = retention.get("absolute_gates")
    if (
        not isinstance(decision, Mapping)
        or decision.get("method") != "paired_seed_hierarchical_bootstrap"
        or not isinstance(margins, Mapping)
        or not isinstance(absolute, Mapping)
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "production design has an unsupported retention decision rule"
        )
    resamples = int(decision["resamples"])
    seed = int(decision["seed"])
    confidence = float(decision["confidence_level"])
    final_rows = [
        (
            _final_ugi_metrics(evaluation, baseline_arm),
            _final_ugi_metrics(evaluation, challenger_arm),
        )
        for evaluation in evaluations
    ]

    comparisons: dict[str, Any] = {}
    gates: dict[str, bool] = {}
    for metric, policy in margins.items():
        if not isinstance(policy, Mapping):
            raise SynthesisProgramProductionAdjudicationError(
                f"non-inferiority policy is malformed for {metric}"
            )
        if metric == "held_component_exact_l1_decomposition_coverage":
            field = "exact_l1_decomposition_coverage"
            baseline = [
                _number(row[0][1].get(field), label=f"baseline {metric}") for row in final_rows
            ]
            challenger = [
                _number(row[1][1].get(field), label=f"challenger {metric}") for row in final_rows
            ]
        else:
            baseline = [
                _number(row[0][0].get(metric), label=f"baseline {metric}") for row in final_rows
            ]
            challenger = [
                _number(row[1][0].get(metric), label=f"challenger {metric}") for row in final_rows
            ]

        if "minimum_ratio" in policy:
            floor = _number(policy["minimum_ratio"], label=f"{metric} minimum ratio")
            bootstrap = paired_seed_bootstrap(
                baseline,
                challenger,
                resamples=resamples,
                seed=seed,
                confidence_level=confidence,
                ratio=True,
            )
            point_pass = bootstrap["point_estimate"] >= floor
            bound_pass = bootstrap["one_sided_bound"] >= floor
            comparison = {
                **bootstrap,
                "decision_scale": "challenger_to_baseline_ratio",
                "minimum_ratio": floor,
                "point_pass": point_pass,
                "bound_pass": bound_pass,
            }
        elif "maximum_absolute_drop" in policy:
            margin = _number(policy["maximum_absolute_drop"], label=f"{metric} margin")
            bootstrap = paired_seed_bootstrap(
                baseline,
                challenger,
                resamples=resamples,
                seed=seed,
                confidence_level=confidence,
            )
            point_pass = bootstrap["point_estimate"] <= margin
            bound_pass = bootstrap["one_sided_bound"] <= margin
            comparison = {
                **bootstrap,
                "decision_scale": "baseline_minus_challenger",
                "maximum_absolute_drop": margin,
                "point_pass": point_pass,
                "bound_pass": bound_pass,
            }
            if "minimum_absolute_value" in policy:
                floor = _number(
                    policy["minimum_absolute_value"],
                    label=f"{metric} minimum absolute value",
                )
                challenger_mean = float(np.mean(challenger))
                absolute_pass = challenger_mean >= floor
                comparison.update(
                    {
                        "challenger_mean": challenger_mean,
                        "minimum_absolute_value": floor,
                        "absolute_point_pass": absolute_pass,
                    }
                )
                point_pass = point_pass and absolute_pass
        else:
            raise SynthesisProgramProductionAdjudicationError(
                f"non-inferiority policy has no supported threshold for {metric}"
            )
        comparison["pass"] = bool(point_pass and bound_pass)
        comparisons[str(metric)] = comparison
        gates[f"ugi_retention_{metric}"] = comparison["pass"]

    fixed_limit = int(absolute["fixed_state_failures"]["maximum"])
    overflow_limit = int(absolute["support_overflow_count"]["maximum"])
    fixed_observed = max(
        int(row.get("fixed_state_failures", 0))
        for evaluation in evaluations
        for arm in evaluation["checkpoint_metrics"].values()
        for checkpoint in arm.values()
        for split in checkpoint.values()
        for row in split.values()
    )
    overflow_observed = max(
        int(row.get("support_overflow_count", 0))
        for evaluation in evaluations
        for arm in evaluation["checkpoint_metrics"].values()
        for checkpoint in arm.values()
        for split in checkpoint.values()
        for row in split.values()
    )
    gates["fixed_state_failures_within_limit"] = fixed_observed <= fixed_limit
    gates["support_overflow_within_limit"] = overflow_observed <= overflow_limit
    return (
        {
            "baseline_arm": baseline_arm,
            "challenger_arm": challenger_arm,
            "comparisons": comparisons,
            "absolute_gates": {
                "fixed_state_failures": {
                    "observed_maximum": fixed_observed,
                    "maximum": fixed_limit,
                },
                "support_overflow_count": {
                    "observed_maximum": overflow_observed,
                    "maximum": overflow_limit,
                },
            },
            "decision_rule": dict(decision),
            "interpretation": (
                "Independent training seeds are the paired resampling unit. Molecules within a "
                "seed determine that seed's frozen metric cell and are not counted as independent "
                "replicates."
            ),
        },
        gates,
    )


def _catalogue_free_posthoc_comparison(
    loaded: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Compare semantic conditioning with exact-L1 post-hoc filtering at matched budgets."""

    final_step = int(design["training"]["checkpoint_steps"][-1])
    expected_attempts = int(
        design["evaluation"]["native_sampling"][
            "heldout_samples_per_supported_program_at_final_checkpoint_per_seed"
        ]
    )
    decision = design["ugi_retention"]["decision_rule"]
    bootstrap_resamples = int(decision["resamples"])
    bootstrap_seed = int(decision["seed"])
    confidence_level = float(decision["confidence_level"])
    comparison_metrics = (
        "raw_valid_fraction",
        "exact_l1_yield_per_attempt",
        "exact_l1_coverage_among_valid",
        "forward_replay_precision_among_exact_l1",
    )
    by_program: dict[str, Any] = {}
    budgets_match = True
    for program_id in sorted(design["programs"]):
        replicate_rows: list[dict[str, Any]] = []
        for item in loaded:
            groups: dict[str, list[Mapping[str, Any]]] = {
                CONDITIONED_ARM: [],
                POSTHOC_ARM: [],
            }
            header: Mapping[str, Any] | None = None
            for row_index, row in enumerate(iter_jsonl(item["samples_path"])):
                if row_index == 0 and isinstance(row, Mapping) and "schema_version" in row:
                    header = row
                    continue
                if not isinstance(row, Mapping):
                    raise SynthesisProgramProductionAdjudicationError(
                        "production sample ledger contains a non-object row"
                    )
                if (
                    row.get("arm_id") in groups
                    and row.get("program_id") == program_id
                    and row.get("evaluation_split") == "heldout"
                    and int(row.get("checkpoint_step", -1)) == final_step
                ):
                    groups[str(row["arm_id"])].append(row)
            if (
                not isinstance(header, Mapping)
                or header.get("schema_version") != "forge.synthesis_program_production_samples.v1"
            ):
                raise SynthesisProgramProductionAdjudicationError(
                    "production sample ledger has no supported schema header"
                )

            summaries: dict[str, Any] = {}
            for arm_id, rows in groups.items():
                attempts = len(rows)
                valid = sum(row.get("valid") is True for row in rows)
                exact_l1 = sum(
                    row.get("valid") is True and row.get("exact_l1_program") is True for row in rows
                )
                forward_verified = sum(
                    row.get("valid") is True
                    and row.get("exact_l1_program") is True
                    and int(row.get("forward_verified_trace_count", 0)) > 0
                    for row in rows
                )
                if attempts != expected_attempts:
                    budgets_match = False
                summaries[arm_id] = {
                    "attempts": attempts,
                    "valid": valid,
                    "exact_l1_survivors": exact_l1,
                    "forward_verified_survivors": forward_verified,
                    "raw_valid_fraction": valid / attempts if attempts else 0.0,
                    "exact_l1_yield_per_attempt": exact_l1 / attempts if attempts else 0.0,
                    "exact_l1_coverage_among_valid": exact_l1 / valid if valid else None,
                    "forward_replay_precision_among_exact_l1": (
                        forward_verified / exact_l1 if exact_l1 else None
                    ),
                }
            conditioned_yield = float(summaries[CONDITIONED_ARM]["exact_l1_yield_per_attempt"])
            posthoc_yield = float(summaries[POSTHOC_ARM]["exact_l1_yield_per_attempt"])
            replicate_rows.append(
                {
                    "replicate": int(item["run"]["replicate"]),
                    "seed": int(item["evaluation"]["seed"]),
                    "conditioned": summaries[CONDITIONED_ARM],
                    "posthoc_filtering": summaries[POSTHOC_ARM],
                    "conditioned_minus_posthoc_exact_l1_yield": (conditioned_yield - posthoc_yield),
                }
            )
        metric_comparisons: dict[str, Any] = {}
        for metric in comparison_metrics:
            conditioned_values = [
                _optional_number(
                    row["conditioned"].get(metric),
                    label=f"conditioned {program_id} {metric}",
                )
                for row in replicate_rows
            ]
            posthoc_values = [
                _optional_number(
                    row["posthoc_filtering"].get(metric),
                    label=f"posthoc {program_id} {metric}",
                )
                for row in replicate_rows
            ]
            metric_comparisons[metric] = _paired_descriptive_comparison(
                conditioned_values,
                posthoc_values,
                left_label="conditioned",
                right_label="posthoc",
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        yield_comparison = metric_comparisons["exact_l1_yield_per_attempt"]
        by_program[program_id] = {
            "replicates": replicate_rows,
            "metrics": metric_comparisons,
            "mean_conditioned_exact_l1_yield": yield_comparison["conditioned_mean"],
            "mean_posthoc_exact_l1_yield": yield_comparison["posthoc_mean"],
            "mean_conditioned_minus_posthoc_exact_l1_yield": yield_comparison[
                "conditioned_minus_posthoc_mean"
            ],
            "paired_seed_exact_l1_yield_difference_interval": yield_comparison[
                "paired_seed_difference_interval"
            ],
        }
    return (
        {
            "conditioned_arm": CONDITIONED_ARM,
            "posthoc_arm": POSTHOC_ARM,
            "posthoc_filter": "valid_and_exact_l1_program",
            "attempts_per_program_per_seed_per_arm": expected_attempts,
            "programs": by_program,
            "interpretation": (
                "Both arms generate complete graphs without component identifiers or fragment "
                "tokens. The post-hoc arm receives no program coordinate during training or "
                "sampling; the exact reaction adapter is applied only to completed products."
            ),
            "decision_role": "descriptive_causal_ablation_not_candidate_selection",
        },
        {"catalogue_free_posthoc_attempt_budgets_match": budgets_match},
    )


def _correct_program_vs_cyclic_comparison(
    loaded: Sequence[Mapping[str, Any]], design: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare the correct program identifier with the frozen cyclic-ID control."""

    final_step = str(int(design["training"]["checkpoint_steps"][-1]))
    decision = design["ugi_retention"]["decision_rule"]
    bootstrap_resamples = int(decision["resamples"])
    bootstrap_seed = int(decision["seed"])
    confidence_level = float(decision["confidence_level"])
    metric_fields = (
        "raw_valid_fraction",
        "exact_l1_yield_per_attempt",
        "exact_l1_decomposition_coverage",
        "exact_forward_replay_precision",
        "internal_diversity",
        "effective_component_count",
        "component_novelty_fraction",
        "whole_lipid_novelty_fraction",
    )
    programs: dict[str, Any] = {}
    for program_id in sorted(design["programs"]):
        try:
            correct_rows = [
                item["evaluation"]["checkpoint_metrics"][CONDITIONED_ARM][final_step]["heldout"][
                    program_id
                ]
                for item in loaded
            ]
            cyclic_rows = [
                item["evaluation"]["checkpoint_metrics"][CYCLIC_ARM][final_step]["heldout"][
                    program_id
                ]
                for item in loaded
            ]
        except (KeyError, TypeError) as error:
            raise SynthesisProgramProductionAdjudicationError(
                f"correct/cyclic comparison omits {program_id} final heldout metrics"
            ) from error
        metrics: dict[str, Any] = {}
        for metric in metric_fields:
            correct_values = [
                _optional_number(row.get(metric), label=f"correct {program_id} {metric}")
                for row in correct_rows
            ]
            cyclic_values = [
                _optional_number(row.get(metric), label=f"cyclic {program_id} {metric}")
                for row in cyclic_rows
            ]
            metrics[metric] = _paired_descriptive_comparison(
                correct_values,
                cyclic_values,
                left_label="correct_program",
                right_label="cyclic_program_id",
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        programs[program_id] = {"metrics": metrics}
    return {
        "correct_program_arm": CONDITIONED_ARM,
        "cyclic_program_id_arm": CYCLIC_ARM,
        "program_id_mapping": design["training"]["arms"][CYCLIC_ARM]["program_id_mapping"],
        "programs": programs,
        "interpretation": (
            "The control cyclically permutes only the program identifier while retaining role, "
            "core-position and depth coordinates. It tests whether the identifier contributes "
            "program-specific information; it is not a fully incorrect-program control."
        ),
        "decision_role": "descriptive_nonselecting_control",
    }


def _finite_catalogue_comparison(
    production_runs: Sequence[Mapping[str, Any]],
    catalogue_runs: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare the conditioned generator and strong catalogue assembler without a winner score."""

    final_step = str(int(design["training"]["checkpoint_steps"][-1]))
    decision = design["ugi_retention"]["decision_rule"]
    bootstrap_resamples = int(decision["resamples"])
    bootstrap_seed = int(decision["seed"])
    confidence_level = float(decision["confidence_level"])
    metric_fields = {
        "raw_valid_fraction": ("raw_valid_fraction", "valid_fraction"),
        "exact_l1_yield_per_attempt": (
            "exact_l1_yield_per_attempt",
            "exact_l1_yield_per_attempt",
        ),
        "unique_exact_l1_products_per_1000_attempts": (
            "unique_exact_l1_products_per_1000_attempts",
            "unique_exact_l1_products_per_1000_attempts",
        ),
        "unique_whole_product_novel_exact_l1_products_per_1000_attempts": (
            "unique_whole_product_novel_exact_l1_products_per_1000_attempts",
            "unique_whole_product_novel_exact_l1_products_per_1000_attempts",
        ),
        "unique_open_ended_exact_l1_products_per_1000_attempts": (
            "unique_open_ended_exact_l1_products_per_1000_attempts",
            "unique_open_ended_exact_l1_products_per_1000_attempts",
        ),
        "internal_diversity": ("internal_diversity", "mean_pairwise_ecfp4_distance"),
        "effective_component_count": (
            "effective_component_count",
            "effective_component_count",
        ),
        "component_novelty_fraction": (
            "component_novelty_fraction",
            "component_novelty_fraction",
        ),
        "whole_product_novelty_fraction": (
            "whole_lipid_novelty_fraction",
            "whole_product_novel_to_train_fraction",
        ),
    }
    programmes: dict[str, Any] = {}
    for program_id in sorted(design["programs"]):
        model_rows: list[Mapping[str, Any]] = []
        catalogue_rows: list[Mapping[str, Any]] = []
        for production, catalogue in zip(production_runs, catalogue_runs, strict=True):
            try:
                model = production["evaluation"]["checkpoint_metrics"][CONDITIONED_ARM][final_step][
                    "heldout"
                ][program_id]
                catalogue_metrics = dict(catalogue["result"]["metrics"]["per_program"][program_id])
                sampled_metrics = catalogue["result"]["sampled_component_metrics"][program_id]
                catalogue_metrics.update(
                    {
                        "effective_component_count": sampled_metrics["effective_component_count"],
                        "component_novelty_fraction": sampled_metrics["component_novelty_fraction"],
                    }
                )
            except (KeyError, TypeError) as error:
                raise SynthesisProgramProductionAdjudicationError(
                    f"model/catalogue comparison omits {program_id} metrics"
                ) from error
            if not isinstance(model, Mapping) or not isinstance(catalogue_metrics, Mapping):
                raise SynthesisProgramProductionAdjudicationError(
                    f"model/catalogue metrics are malformed for {program_id}"
                )
            model_rows.append(model)
            catalogue_rows.append(catalogue_metrics)
        comparisons: dict[str, Any] = {}
        for metric, (model_field, catalogue_field) in metric_fields.items():
            model_values = [
                _optional_number(row.get(model_field), label=f"FORGE {program_id} {metric}")
                for row in model_rows
            ]
            catalogue_values = [
                _optional_number(row.get(catalogue_field), label=f"catalogue {program_id} {metric}")
                for row in catalogue_rows
            ]
            comparisons[metric] = _paired_descriptive_comparison(
                model_values,
                catalogue_values,
                left_label="forge",
                right_label="catalogue",
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
                confidence_level=confidence_level,
            )
        programmes[program_id] = {
            "metrics": comparisons,
            "catalogue": catalogue_runs[0]["result"]["catalogue"][program_id],
        }
    return {
        "forge_arm": CONDITIONED_ARM,
        "catalogue_arm": "finite_component_catalogue_oracle",
        "primary_metric": "unique_open_ended_exact_l1_products_per_1000_attempts",
        "programs": programmes,
        "interpretation": (
            "The catalogue arm receives exact train-fold components and exact forward chemistry. "
            "It is expected to maximize transform validity but cannot leave its component support. "
            "The comparison therefore reports validity, breadth and diversity separately and does "
            "not collapse them into a single winner score."
        ),
        "decision_role": "descriptive_claim_matched_baseline_not_candidate_selection",
    }


def adjudicate_production_runs(
    run_dirs: Sequence[Path],
    repo: Path,
    output_path: Path,
    *,
    catalogue_run_dirs: Sequence[Path] = (),
) -> dict[str, Any]:
    """Verify, aggregate and preserve the complete three-replicate production comparison."""

    repo = repo.resolve()
    output_path = output_path.resolve()
    try:
        output_path.relative_to(repo)
    except ValueError as error:
        raise SynthesisProgramProductionAdjudicationError(
            "adjudication output must stay inside the repository"
        ) from error
    loaded = [_load_production_run(path) for path in run_dirs]
    if not loaded:
        raise SynthesisProgramProductionAdjudicationError("no production runs were supplied")
    replicates = [int(item["run"]["replicate"]) for item in loaded]
    if len(replicates) != len(set(replicates)):
        raise SynthesisProgramProductionAdjudicationError("production replicate ids are duplicated")
    loaded.sort(key=lambda item: int(item["run"]["replicate"]))

    source_hashes = {str(item["run"].get("source_sha256")) for item in loaded}
    spec_hashes = {str(item["run"].get("spec_sha256")) for item in loaded}
    experiment_ids = {str(item["run"].get("experiment_id")) for item in loaded}
    design_pins = {
        (str(item["design_pin"].get("path")), str(item["design_pin"].get("sha256")))
        for item in loaded
    }
    if (
        len(source_hashes) != 1
        or len(spec_hashes) != 1
        or len(experiment_ids) != 1
        or len(design_pins) != 1
    ):
        raise SynthesisProgramProductionAdjudicationError(
            "production runs differ in experiment, source, specification or scientific design"
        )
    design_relative, design_sha256 = next(iter(design_pins))
    design_path = resolve_pin(
        {"path": design_relative, "sha256": design_sha256},
        repo,
        label="production comparison design",
    )
    design = read_json_object(
        design_path,
        error=SynthesisProgramProductionAdjudicationError,
        label="production comparison design",
    )
    expected_replicates = list(range(len(design["training"]["replicate_seeds"])))
    observed_replicates = [int(item["run"]["replicate"]) for item in loaded]
    observed_seeds = [int(item["evaluation"]["seed"]) for item in loaded]
    if observed_replicates != expected_replicates:
        raise SynthesisProgramProductionAdjudicationError(
            f"production run set is incomplete: expected {expected_replicates}, found {observed_replicates}"
        )
    if observed_seeds != [int(value) for value in design["training"]["replicate_seeds"]]:
        raise SynthesisProgramProductionAdjudicationError(
            "production evaluation seeds differ from the frozen design"
        )

    catalogue_loaded = [_load_catalogue_run(path) for path in catalogue_run_dirs]
    catalogue_comparison: dict[str, Any] | None = None
    catalogue_evidence: list[dict[str, Any]] = []
    if catalogue_loaded:
        catalogue_loaded.sort(key=lambda item: int(item["run"]["replicate"]))
        catalogue_replicates = [int(item["run"]["replicate"]) for item in catalogue_loaded]
        catalogue_seeds = [int(item["result"]["seed"]) for item in catalogue_loaded]
        catalogue_sources = {str(item["run"].get("source_sha256")) for item in catalogue_loaded}
        catalogue_design_pins = {
            (str(item["design_pin"].get("path")), str(item["design_pin"].get("sha256")))
            for item in catalogue_loaded
        }
        if (
            catalogue_replicates != expected_replicates
            or catalogue_seeds != observed_seeds
            or catalogue_sources != source_hashes
            or catalogue_design_pins != design_pins
        ):
            raise SynthesisProgramProductionAdjudicationError(
                "catalogue runs do not share the production replicates, seeds, source and design"
            )
        catalogue_comparison = _finite_catalogue_comparison(loaded, catalogue_loaded, design)

    retention, gates = _adjudicate_retention([item["evaluation"] for item in loaded], design)
    catalogue_free_comparison, catalogue_free_gates = _catalogue_free_posthoc_comparison(
        loaded, design
    )
    cyclic_comparison = _correct_program_vs_cyclic_comparison(loaded, design)
    gates.update(catalogue_free_gates)
    gates.update(
        {
            "three_complete_replicates": len(loaded) == 3,
            "one_executable_source": len(source_hashes) == 1,
            "one_frozen_specification": len(spec_hashes) == 1,
            "one_frozen_design": len(design_pins) == 1,
            "all_per_replicate_gates_passed": True,
            "coverage_and_precision_reported": all(
                item["evaluation"]["gates"].get("coverage_and_precision_reported") is True
                for item in loaded
            ),
            "held_reaction_family_not_a_hard_gate": design["evaluation"]["held_reaction_family"][
                "hard_gate"
            ]
            is False,
            "candidate_selection_absent": True,
            "route_or_oracle_calls_zero": True,
            "reductive_amination_substructure_rate_absent": True,
            "finite_catalogue_comparison_complete": bool(catalogue_loaded),
        }
    )

    evidence: list[dict[str, Any]] = []
    for item in loaded:
        replicate = int(item["run"]["replicate"])
        evidence_dir = output_path.parent / "evidence" / f"replicate_{replicate}"
        sources = {
            "run_manifest": item["run_dir"] / "run.json",
            "training_stage_manifest": item["run_dir"] / "stages/training/manifest.json",
            "evaluation_stage_manifest": item["run_dir"] / "stages/evaluation/manifest.json",
            "training_result": item["training_path"],
            "evaluation_result": item["evaluation_path"],
            "evaluation_samples": item["samples_path"],
        }
        production_snapshots: dict[str, Path] = {}
        for label, source in sources.items():
            suffix = ".jsonl.gz" if label == "evaluation_samples" else ".json"
            destination = evidence_dir / f"{label}{suffix}"
            atomic_write(destination, source.read_bytes())
            production_snapshots[label] = destination
        evidence.append(
            {
                "replicate": replicate,
                "seed": int(item["evaluation"]["seed"]),
                "run_id": item["run"]["run_id"],
                **{label: pin_record(path, repo) for label, path in production_snapshots.items()},
            }
        )

    for item in catalogue_loaded:
        replicate = int(item["run"]["replicate"])
        evidence_dir = output_path.parent / "catalogue_evidence" / f"replicate_{replicate}"
        sources = {
            "run_manifest": item["run_dir"] / "run.json",
            "catalogue_stage_manifest": item["run_dir"] / "stages/catalogue/manifest.json",
            "catalogue_result": item["result_path"],
            "catalogue_samples": item["samples_path"],
        }
        catalogue_snapshots: dict[str, Path] = {}
        for label, source in sources.items():
            suffix = ".jsonl.gz" if label == "catalogue_samples" else ".json"
            destination = evidence_dir / f"{label}{suffix}"
            atomic_write(destination, source.read_bytes())
            catalogue_snapshots[label] = destination
        catalogue_evidence.append(
            {
                "replicate": replicate,
                "seed": int(item["result"]["seed"]),
                "run_id": item["run"]["run_id"],
                **{label: pin_record(path, repo) for label, path in catalogue_snapshots.items()},
            }
        )

    noninferiority_passed = (
        all(value for key, value in gates.items() if key.startswith("ugi_retention_"))
        and gates["fixed_state_failures_within_limit"]
        and gates["support_overflow_within_limit"]
    )
    result = {
        "schema_version": ADJUDICATION_SCHEMA,
        "status": (
            "adjudicated_noninferiority_pass"
            if noninferiority_passed
            else "adjudicated_noninferiority_fail"
        ),
        "scientific_decision": {
            "shared_three_program_ugi_retention_noninferior": noninferiority_passed,
            "negative_result_is_valid": not noninferiority_passed,
            "held_reaction_family_is_secondary": True,
        },
        "gates": gates,
        "ugi_retention": retention,
        "catalogue_free_conditioning_vs_posthoc_filtering": catalogue_free_comparison,
        "correct_program_coordinate_vs_cyclic_program_id_control": cyclic_comparison,
        "finite_component_catalogue_comparison": catalogue_comparison,
        "design": pin_record(design_path, repo),
        "input_source_sha256": next(iter(source_hashes)),
        "input_spec_sha256": next(iter(spec_hashes)),
        "adjudicator_source_sha256": source_fingerprint(repo),
        "implementation": pin_record(Path(__file__), repo),
        "evidence": evidence,
        "catalogue_evidence": catalogue_evidence,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "This aggregate is computational model evidence, not a synthesis-success probability.",
            "BL and LX exact replay does not inherit Ugi prospective validation.",
            "Held-reaction-family generalization is not a hard gate and is not fabricated here.",
        ],
    }
    write_json(output_path, result)
    return result


__all__ = [
    "ADJUDICATION_SCHEMA",
    "SynthesisProgramProductionAdjudicationError",
    "adjudicate_production_runs",
    "paired_seed_bootstrap",
    "paired_seed_difference_interval",
]
