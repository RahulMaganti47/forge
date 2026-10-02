"""Test whether coarse Ugi morphology predicts observed HeLa potency."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

from examples.diagnostics.product_l1.training.ugi_training_cache import load_ugi_training_cache
from examples.diagnostics.support.adapters.terminal_support import (
    canonical_morphology_program_bytes,
)
from forge.core.hashing import sha256_file as _sha256_file
from forge.core.io import stable_json as _stable_json

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_potency_signal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_potency_signal.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_potency_signal_oof.v1"
PAIR_SCHEME = "held_aldehyde_isocyanide_pair_5fold"
FEATURE_FIELDS = (
    "amine_nodes",
    "aldehyde_nodes",
    "isocyanide_nodes",
    "amine_junctions",
    "aldehyde_junctions",
    "isocyanide_junctions",
    "amine_cycles",
    "aldehyde_cycles",
    "isocyanide_cycles",
    "amine_attachments",
    "aldehyde_attachments",
    "isocyanide_attachments",
)


class UgiMorphologyPotencySignalError(RuntimeError):
    """Raised when the frozen morphology-potency gate cannot be evaluated."""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise UgiMorphologyPotencySignalError(f"JSON object required: {path}")
    return value


def _read_csv_gzip(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiMorphologyPotencySignalError("proposal ledger must contain JSON objects")
    return rows


def _program_by_product(training_cache: Path) -> dict[str, str]:
    corpus, records_by_fold = load_ugi_training_cache(training_cache)
    output: dict[str, str] = {}
    for fold in sorted(corpus.assignments_by_fold):
        assignments = corpus.assignments_by_fold[fold]
        records = records_by_fold[fold]
        for assignment, record in zip(assignments, records, strict=True):
            product = str(assignment["canonical_product_smiles"])
            digest = hashlib.sha256(canonical_morphology_program_bytes(record.program)).hexdigest()
            previous = output.get(product)
            if previous is not None and previous != digest:
                raise UgiMorphologyPotencySignalError("conflicting morphology programs")
            output[product] = digest
    return output


def _role_state(program: Mapping[str, Any], role: int) -> str:
    return _stable_json(
        [
            int(program["node_counts"][role]),
            int(program["junction_budgets"][role]),
            int(program["cycle_ranks"][role]),
            int(program["attachment_counts"][role]),
        ]
    )


def _features(program: Mapping[str, Any]) -> list[float]:
    return [
        *[float(value) for value in program["node_counts"]],
        *[float(value) for value in program["junction_budgets"]],
        *[float(value) for value in program["cycle_ranks"]],
        *[float(value) for value in program["attachment_counts"]],
    ]


def _correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.all(observed == observed[0]) or np.all(predicted == predicted[0]):
        return 0.0
    left = rankdata(observed, method="average")
    right = rankdata(predicted, method="average")
    return float(np.corrcoef(left, right)[0, 1])


def prediction_metrics(
    observed: np.ndarray, predicted: np.ndarray, weights: np.ndarray | None = None
) -> dict[str, float]:
    """Return frozen prediction and top-quartile allocation metrics."""

    if len(observed) != len(predicted) or len(observed) < 2:
        raise UgiMorphologyPotencySignalError("invalid metric arrays")
    if weights is None:
        weights = np.ones(len(observed), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    residual = observed - predicted
    observed_mean = float(np.dot(weights, observed))
    denominator = float(np.dot(weights, (observed - observed_mean) ** 2))
    threshold = float(np.quantile(predicted, 0.75))
    top = predicted >= threshold
    top_weights = weights[top] / weights[top].sum()
    top_mean = float(np.dot(top_weights, observed[top]))
    return {
        "records": float(len(observed)),
        "mae": float(np.dot(weights, np.abs(residual))),
        "rmse": float(math.sqrt(np.dot(weights, residual**2))),
        "r2": 1.0 - float(np.dot(weights, residual**2)) / denominator if denominator > 0 else 0.0,
        "midrank_spearman": _correlation(observed, predicted),
        "observed_mean": observed_mean,
        "predicted_top_quartile_threshold": threshold,
        "observed_mean_in_predicted_top_quartile": top_mean,
        "top_quartile_observed_gain": top_mean - observed_mean,
    }


def _cohort(
    *,
    curated: Sequence[Mapping[str, str]],
    applicability: Sequence[Mapping[str, str]],
    program_by_product: Mapping[str, str],
    proposal_by_program: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    source_by_label = {str(row["label"]): row for row in curated}
    rows = [
        row
        for row in applicability
        if row.get("scheme") == PAIR_SCHEME and row.get("distribution_bin") == "interpolative"
    ]
    output = []
    for row in rows:
        label = str(row["label"])
        source = source_by_label[label]
        digest = program_by_product.get(str(source["model_smiles"]))
        proposal = proposal_by_program.get(str(digest)) if digest is not None else None
        if proposal is None:
            continue
        program = proposal["program"]
        role_states = [_role_state(program, role) for role in range(3)]
        output.append(
            {
                "label": label,
                "head_id": label.split("B", 1)[0],
                "program_sha256": digest,
                "amine_role_state": role_states[0],
                "aldehyde_role_state": role_states[1],
                "isocyanide_role_state": role_states[2],
                "tail_role_state_pair": role_states[1] + "|" + role_states[2],
                "role_factorized_features": [
                    role_states[0],
                    role_states[1],
                    role_states[2],
                    role_states[1] + "|" + role_states[2],
                ],
                "numeric_features": _features(program),
                "observed_hela_mtp": float(source["expt_Hela"]),
                "promoted_over_broad_weight": float(proposal["proposal_probability"])
                / float(proposal["broad_prior_probability"]),
            }
        )
    output.sort(key=lambda row: str(row["label"]))
    return output


__all__ = ["UgiMorphologyPotencySignalError", "prediction_metrics"]


__all__ = ["_sha256_file", "UgiMorphologyPotencySignalError", "prediction_metrics"]
