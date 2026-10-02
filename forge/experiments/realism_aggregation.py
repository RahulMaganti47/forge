"""Aggregate method-blind lipid-realism seed results without molecule-level pseudoreplication."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, is_sha256, pin_record
from forge.core.io import read_json_object, write_csv, write_json

INPUT_SCHEMA = "forge.common_lipid_realism_complete_assessment.v1"
RESULT_SCHEMA = "forge.common_lipid_realism_aggregate.v1"

METRICS: dict[str, tuple[str, ...]] = {
    "connected_fraction_per_attempt": (
        "molecular_output",
        "connected_fraction_per_attempt",
    ),
    "unique_fraction_among_connected": (
        "molecular_output",
        "unique_fraction_among_connected",
    ),
    "effective_molecule_count": ("molecular_output", "effective_molecule_count"),
    "within_declared_support_fraction_per_attempt": (
        "molecular_output",
        "within_declared_support_fraction_per_attempt",
    ),
    "internal_diversity": (
        "molecular_output",
        "mean_pairwise_ecfp4_distance_among_unique",
    ),
    "fingerprint_manifold_precision_per_attempt": (
        "empirical_lipid_manifold",
        "fingerprint",
        "precision_per_requested_attempt",
    ),
    "fingerprint_manifold_coverage": (
        "empirical_lipid_manifold",
        "fingerprint",
        "coverage",
    ),
    "descriptor_manifold_precision_per_attempt": (
        "empirical_lipid_manifold",
        "descriptor",
        "precision_per_requested_attempt",
    ),
    "descriptor_manifold_coverage": (
        "empirical_lipid_manifold",
        "descriptor",
        "coverage",
    ),
    "normalized_descriptor_wasserstein": (
        "empirical_lipid_manifold",
        "normalized_descriptor_wasserstein",
        "mean_across_descriptors",
    ),
    "grouped_c2st_auc": (
        "empirical_lipid_manifold",
        "classifier_two_sample",
        "auc_mean",
    ),
}


class LipidRealismAggregationError(ValueError):
    """Seed results do not form one complete comparable experiment."""


def _nested_optional_number(value: Mapping[str, Any], path: Sequence[str]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    if current is None:
        return None
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise LipidRealismAggregationError(f"metric {'.'.join(path)} is not numeric")
    result = float(current)
    if not math.isfinite(result):
        raise LipidRealismAggregationError(f"metric {'.'.join(path)} is not finite")
    return result


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    if any(value is None for value in values):
        return {
            "status": "not_estimable",
            "reason": "metric unavailable for one or more seeds",
            "available_seeds": sum(value is not None for value in values),
            "required_seeds": len(values),
            "mean": None,
            "sample_standard_deviation": None,
        }
    vector = np.asarray(values, dtype=np.float64)
    return {
        "status": "estimated",
        "available_seeds": len(vector),
        "required_seeds": len(vector),
        "mean": float(vector.mean()),
        "sample_standard_deviation": float(vector.std(ddof=1)) if len(vector) > 1 else None,
    }


def aggregate_lipid_realism(
    result_paths: Sequence[Path],
    repo: Path,
    output_dir: Path,
    *,
    expected_seeds: Sequence[int],
) -> dict[str, Any]:
    """Aggregate complete seed sets, treating training seed as the independent unit."""

    if not result_paths:
        raise LipidRealismAggregationError("at least one realism result is required")
    seeds = tuple(expected_seeds)
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise LipidRealismAggregationError("expected seeds must be unique non-negative integers")

    grouped: dict[str, dict[int, tuple[Path, dict[str, Any]]]] = defaultdict(dict)
    config_sha: str | None = None
    reference_shas: dict[str, str] | None = None
    attempt_denominators: set[int] = set()
    for path in result_paths:
        value = read_json_object(
            path, error=LipidRealismAggregationError, label="lipid realism seed result"
        )
        if value.get("schema_version") != INPUT_SCHEMA or value.get("status") != "pass":
            raise LipidRealismAggregationError(f"input is not a passing realism result: {path}")
        method = value.get("method_id")
        seed = value.get("seed")
        assessment = value.get("assessment")
        if not isinstance(method, str) or not method or not isinstance(assessment, Mapping):
            raise LipidRealismAggregationError(f"seed result identity is malformed: {path}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise LipidRealismAggregationError(f"seed result has a malformed seed: {path}")
        if seed in grouped[method]:
            raise LipidRealismAggregationError(f"duplicate method/seed result: {method}/{seed}")
        if assessment.get("method_id") != method or assessment.get("seed") != seed:
            raise LipidRealismAggregationError(f"nested assessment identity changed: {path}")
        attempts = assessment.get("attempts")
        if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
            raise LipidRealismAggregationError(f"attempt denominator is malformed: {path}")
        attempt_denominators.add(attempts)
        gates = value.get("gates")
        if (
            not isinstance(gates, Mapping)
            or not gates
            or any(gate is not True for gate in gates.values())
        ):
            raise LipidRealismAggregationError(f"input assessment gates did not all pass: {path}")
        raw_config_sha = value.get("config", {}).get("sha256")
        observed_reference_shas: dict[str, str] = {}
        raw_reference_inputs = value.get("inputs")
        if isinstance(raw_reference_inputs, Mapping):
            for key, record in raw_reference_inputs.items():
                if isinstance(key, str) and isinstance(record, Mapping):
                    digest = record.get("sha256")
                    if is_sha256(digest):
                        observed_reference_shas[key] = str(digest)
        if not is_sha256(raw_config_sha) or set(observed_reference_shas) != {
            "r0_constitutional",
            "r0_fold_assignments",
        }:
            raise LipidRealismAggregationError(f"reference pins are malformed: {path}")
        observed_config_sha = str(raw_config_sha)
        if config_sha is None:
            config_sha = observed_config_sha
            reference_shas = observed_reference_shas
        elif config_sha != observed_config_sha or reference_shas != observed_reference_shas:
            raise LipidRealismAggregationError("seed results use different realism contracts")
        grouped[method][seed] = (path, value)

    if len(attempt_denominators) != 1:
        raise LipidRealismAggregationError(
            f"methods use different requested-attempt denominators: {sorted(attempt_denominators)}"
        )

    required_seed_set = set(seeds)
    if any(set(seed_results) != required_seed_set for seed_results in grouped.values()):
        observed = {method: sorted(seed_results) for method, seed_results in grouped.items()}
        raise LipidRealismAggregationError(
            f"every method must contain exactly seeds {sorted(seeds)}; observed {observed}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise LipidRealismAggregationError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_rows: list[dict[str, Any]] = []
    method_summaries: dict[str, Any] = {}
    for method in sorted(grouped):
        metric_values: dict[str, list[float | None]] = {name: [] for name in METRICS}
        for seed in seeds:
            _, result = grouped[method][seed]
            assessment = result["assessment"]
            row: dict[str, Any] = {
                "method_id": method,
                "seed": seed,
                "attempts": int(assessment["attempts"]),
            }
            for name, metric_path in METRICS.items():
                metric = _nested_optional_number(assessment, metric_path)
                row[name] = metric
                metric_values[name].append(metric)
            seed_rows.append(row)
        method_summaries[method] = {
            "seeds": list(seeds),
            "independent_unit": "training seed",
            "metrics": {name: _summary(values) for name, values in metric_values.items()},
        }

    summary_rows = [
        {
            "method_id": method,
            **{
                f"{metric}_{statistic}": summary["metrics"][metric][statistic]
                for metric in METRICS
                for statistic in ("mean", "sample_standard_deviation")
            },
        }
        for method, summary in method_summaries.items()
    ]
    seed_rows_path = output_dir / "seed_rows.csv"
    summary_rows_path = output_dir / "method_summary.csv"
    write_csv(seed_rows_path, seed_rows, list(seed_rows[0]))
    write_csv(summary_rows_path, summary_rows, list(summary_rows[0]))
    gates = {
        "complete_seed_set_per_method": True,
        "common_attempt_denominator": True,
        "common_config_and_reference_pins": True,
        "training_seed_is_independent_unit": True,
        "molecule_rows_not_treated_as_replicates": True,
        "missing_metrics_not_imputed": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "methods": method_summaries,
        "expected_seeds": list(seeds),
        "inputs": [
            pin_record(path, repo)
            for method in sorted(grouped)
            for seed in seeds
            for path, _ in [grouped[method][seed]]
        ],
        "contract": {
            "config_sha256": config_sha,
            "reference_sha256": reference_shas,
        },
        "seed_rows": artifact_record(seed_rows_path),
        "method_summary": artifact_record(summary_rows_path),
        "gates": gates,
        "candidate_selection": False,
        "nonclaims": [
            "Seed summaries do not turn structural realism into a property or activity claim.",
            "No molecule-level row is treated as an independent training replicate.",
        ],
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "INPUT_SCHEMA",
    "METRICS",
    "RESULT_SCHEMA",
    "LipidRealismAggregationError",
    "aggregate_lipid_realism",
]
