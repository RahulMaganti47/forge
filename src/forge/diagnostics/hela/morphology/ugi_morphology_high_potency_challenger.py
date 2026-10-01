"""Audit a decision-aligned high-potency morphology challenger.

This is one frozen, read-only fairness audit of the failed continuous-potency
proposal.  It asks whether a low-capacity classifier aimed at the upper observed
HeLa potency quartile transfers across exact programs and held role states.  It
does not tune or promote a production proposal on the same biological rows.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forge.core.io import stable_json as _stable_json
from forge.diagnostics.hela.morphology.ugi_morphology_potency_signal import (
    _cohort,
    _program_by_product,
    _read_csv_gzip,
    _read_json,
    _read_jsonl_gzip,
    _sha256_file,
    prediction_metrics,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_high_potency_challenger_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_high_potency_challenger.v1"


class UgiMorphologyHighPotencyChallengerError(RuntimeError):
    """Raised when the frozen challenger cannot be evaluated."""


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyHighPotencyChallengerError(f"invalid input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != record["sha256"]:
        raise UgiMorphologyHighPotencyChallengerError(f"input changed: {label}")
    return path


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    """Return decision-aligned binary metrics without choosing a hard cutoff."""

    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    if len(labels) < 2 or len(labels) != len(probabilities) or len(np.unique(labels)) != 2:
        raise UgiMorphologyHighPotencyChallengerError("binary metrics require both classes")
    return {
        "records": float(len(labels)),
        "prevalence": float(np.mean(labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
    }


def _model(kind: str, regularization: float) -> Any:
    if kind == "numeric_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization,
                solver="lbfgs",
                max_iter=2000,
                random_state=0,
            ),
        )
    if kind == "role_factorized_logistic":
        return make_pipeline(
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            LogisticRegression(
                C=regularization,
                solver="lbfgs",
                max_iter=2000,
                random_state=0,
            ),
        )
    raise UgiMorphologyHighPotencyChallengerError(f"unknown model: {kind}")


def _inner_regularization(
    source: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    values: Sequence[float],
) -> float:
    unique = np.unique(groups)
    if len(unique) < 3:
        return float(values[len(values) // 2])
    splitter = GroupKFold(n_splits=min(4, len(unique)))
    losses: dict[float, list[float]] = defaultdict(list)
    for train, validation in splitter.split(source, observed, groups):
        threshold = float(np.quantile(observed[train], 0.75))
        train_labels = (observed[train] >= threshold).astype(np.int64)
        validation_labels = (observed[validation] >= threshold).astype(np.int64)
        if len(np.unique(train_labels)) != 2:
            continue
        for value in values:
            fitted = _model(kind, float(value)).fit(source[train], train_labels)
            probability = fitted.predict_proba(source[validation])[:, 1]
            losses[float(value)].append(
                float(log_loss(validation_labels, probability, labels=[0, 1]))
            )
    candidates = [(float(np.mean(loss)), value) for value, loss in losses.items() if loss]
    if not candidates:
        return float(values[len(values) // 2])
    return min(candidates)[1]


def _oof_predictions(
    source: np.ndarray,
    observed: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    regularization: Sequence[float],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    unique = np.unique(groups)
    if len(unique) < 2:
        raise UgiMorphologyHighPotencyChallengerError(f"too few groups for {kind}")
    probabilities = np.zeros(len(observed), dtype=np.float64)
    labels = np.zeros(len(observed), dtype=np.int64)
    rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=min(5, len(unique)))
    for fold, (train, test) in enumerate(splitter.split(source, observed, groups)):
        threshold = float(np.quantile(observed[train], 0.75))
        train_labels = (observed[train] >= threshold).astype(np.int64)
        labels[test] = (observed[test] >= threshold).astype(np.int64)
        selected = _inner_regularization(
            source[train],
            observed[train],
            groups[train],
            kind=kind,
            values=regularization,
        )
        fitted = _model(kind, selected).fit(source[train], train_labels)
        probabilities[test] = fitted.predict_proba(source[test])[:, 1]
        rows.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_groups": len(np.unique(groups[test])),
                "training_potency_quartile_threshold": threshold,
                "selected_regularization": selected,
            }
        )
    return probabilities, labels, rows


def _top_probability_gain(observed: np.ndarray, probabilities: np.ndarray) -> float:
    threshold = float(np.quantile(probabilities, 0.75))
    return float(np.mean(observed[probabilities >= threshold]) - np.mean(observed))


def _cluster_bootstrap(
    observed: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
    groups: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    aucs: list[float] = []
    gains: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([indices[group] for group in sampled])
        if len(np.unique(labels[selected])) != 2:
            continue
        aucs.append(float(roc_auc_score(labels[selected], probabilities[selected])))
        gains.append(_top_probability_gain(observed[selected], probabilities[selected]))
    if len(aucs) < max(100, replicates // 2):
        raise UgiMorphologyHighPotencyChallengerError("bootstrap lost too many replicates")
    return {
        "roc_auc_ci95": [float(value) for value in np.quantile(aucs, [0.025, 0.975])],
        "top_probability_quartile_observed_gain_ci95": [
            float(value) for value in np.quantile(gains, [0.025, 0.975])
        ],
    }


def _head_stability(
    head_ids: Sequence[str],
    observed: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    by_head: dict[str, list[int]] = defaultdict(list)
    for index, head in enumerate(head_ids):
        by_head[str(head)].append(index)
    rows: list[dict[str, Any]] = []
    for head, indices in sorted(by_head.items()):
        selected = np.asarray(indices, dtype=np.int64)
        if len(selected) < 10 or len(np.unique(labels[selected])) != 2:
            continue
        metrics = classification_metrics(labels[selected], probabilities[selected])
        rows.append(
            {
                "head": head,
                **metrics,
                "top_probability_quartile_observed_gain": _top_probability_gain(
                    observed[selected], probabilities[selected]
                ),
            }
        )
    aucs = np.asarray([row["roc_auc"] for row in rows], dtype=np.float64)
    gains = np.asarray(
        [row["top_probability_quartile_observed_gain"] for row in rows], dtype=np.float64
    )
    if len(rows) == 0:
        raise UgiMorphologyHighPotencyChallengerError("no heads support stability analysis")
    return {
        "heads_evaluated": len(rows),
        "positive_auc_head_fraction": float(np.mean(aucs > 0.5)),
        "median_head_auc": float(np.median(aucs)),
        "positive_top_quartile_gain_head_fraction": float(np.mean(gains > 0.0)),
        "per_head": rows,
    }


def build_high_potency_challenger(repo: Path, config_path: Path) -> dict[str, Any]:
    """Run the frozen target-alignment audit on measured AGILE rows."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyHighPotencyChallengerError("unsupported config schema")
    pins = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    if set(pins) != {
        "curated_agile",
        "heldout_applicability",
        "training_cache",
        "promoted_proposal_ledger",
        "prior_potency_failure_audit",
    }:
        raise UgiMorphologyHighPotencyChallengerError("input set changed")
    failure = _read_json(pins["prior_potency_failure_audit"])
    if failure.get("decision", {}).get("current_potency_tilting_promoted") is not False:
        raise UgiMorphologyHighPotencyChallengerError("prior negative decision changed")

    program_by_product = _program_by_product(pins["training_cache"])
    proposal_rows = _read_jsonl_gzip(pins["promoted_proposal_ledger"])
    proposal_by_program = {str(row["program_sha256"]): row for row in proposal_rows}
    cohort = _cohort(
        curated=_read_csv_gzip(pins["curated_agile"]),
        applicability=_read_csv_gzip(pins["heldout_applicability"]),
        program_by_product=program_by_product,
        proposal_by_program=proposal_by_program,
    )
    if len(cohort) < int(config["analysis"]["minimum_cohort_records"]):
        raise UgiMorphologyHighPotencyChallengerError("cohort is too small")

    numeric = np.asarray([row["numeric_features"] for row in cohort], dtype=np.float64)
    categorical = np.asarray([row["role_factorized_features"] for row in cohort], dtype=object)
    observed = np.asarray([row["observed_hela_mtp"] for row in cohort], dtype=np.float64)
    group_arrays = {
        "exact_program_hash": np.asarray([row["program_sha256"] for row in cohort]),
        "aldehyde_role_state": np.asarray([row["aldehyde_role_state"] for row in cohort]),
        "isocyanide_role_state": np.asarray([row["isocyanide_role_state"] for row in cohort]),
        "tail_role_state_pair": np.asarray([row["tail_role_state_pair"] for row in cohort]),
    }
    models = tuple(config["analysis"]["models"])
    regularization = tuple(float(value) for value in config["analysis"]["regularization"])
    summaries: dict[str, Any] = {}
    stored: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for scheme, groups in group_arrays.items():
        model_rows: dict[str, Any] = {}
        for kind in models:
            source = numeric if kind == "numeric_logistic" else categorical
            probability, labels, folds = _oof_predictions(
                source,
                observed,
                groups,
                kind=kind,
                regularization=regularization,
            )
            stored[(scheme, kind)] = (probability, labels)
            model_rows[kind] = {
                **classification_metrics(labels, probability),
                "top_probability_quartile_observed_gain": _top_probability_gain(
                    observed, probability
                ),
                "continuous_potency_ranking": prediction_metrics(observed, probability),
                "folds": folds,
            }
        summaries[scheme] = {"groups": len(np.unique(groups)), "models": model_rows}

    primary_scheme = str(config["analysis"]["primary_scheme"])
    selected = min(
        models,
        key=lambda kind: (
            float(summaries[primary_scheme]["models"][kind]["log_loss"]),
            models.index(kind),
        ),
    )
    probability, labels = stored[(primary_scheme, selected)]
    bootstrap_policy = config["analysis"]["clustered_bootstrap"]
    bootstrap = _cluster_bootstrap(
        observed,
        labels,
        probability,
        group_arrays[primary_scheme],
        replicates=int(bootstrap_policy["replicates"]),
        seed=int(bootstrap_policy["seed"]),
    )
    primary = summaries[primary_scheme]["models"][selected]
    role_blocked = {
        scheme: summaries[scheme]["models"][selected]
        for scheme in ("aldehyde_role_state", "isocyanide_role_state", "tail_role_state_pair")
    }
    head_stability = _head_stability(
        [str(row["head_id"]) for row in cohort],
        observed,
        labels,
        probability,
    )
    gates = config["analysis"]["gates"]
    checks = {
        "primary_auc_ci": bootstrap["roc_auc_ci95"][0]
        > float(gates["minimum_primary_auc_ci_lower"]),
        "primary_average_precision": primary["average_precision"]
        >= primary["prevalence"] + float(gates["minimum_average_precision_above_prevalence"]),
        "top_quartile_gain_ci": bootstrap["top_probability_quartile_observed_gain_ci95"][0]
        > float(gates["minimum_top_quartile_gain_ci_lower"]),
        "role_state_transfer": all(
            float(metrics["roc_auc"]) >= float(gates["minimum_role_blocked_auc"])
            for metrics in role_blocked.values()
        ),
        "head_stability": head_stability["positive_auc_head_fraction"]
        >= float(gates["minimum_positive_auc_head_fraction"])
        and head_stability["median_head_auc"] >= float(gates["minimum_median_head_auc"])
        and head_stability["positive_top_quartile_gain_head_fraction"]
        >= float(gates["minimum_positive_gain_head_fraction"]),
    }
    passed = all(checks.values())
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "decision_aligned_morphology_potency_challenger_audited",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": _sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(pins.items())
        },
        "cohort": {
            "records": len(cohort),
            "target": "fold-local upper quartile of observed expt_Hela MTP",
            "molecular_or_oracle_predictions_used_as_target": False,
            "same_authorized_measured_cohort_as_original_signal_gate": True,
        },
        "cross_validation": summaries,
        "selection": {
            "primary_scheme": primary_scheme,
            "selected_low_capacity_classifier": selected,
            "primary_metrics": primary,
            "clustered_bootstrap": bootstrap,
            "role_state_blocked_metrics": role_blocked,
            "head_stability": head_stability,
        },
        "gates": {"criteria": dict(gates), "checks": checks, "all_pass": passed},
        "decision": {
            "target_mismatch_explains_prior_failure": passed,
            "potency_tilting_promoted": False,
            "new_generator_run_authorized": passed,
            "next_step": (
                "freeze_one_untouched_matched_classifier_proposal_replication"
                if passed
                else "close_morphology_potency_tilting_and_retain_terminal_ranking"
            ),
        },
        "nonclaims": [
            "This post-hoc target-alignment audit is not independent prospective evidence.",
            "Passing does not promote potency tilting on these same biological rows.",
            "The classifier does not authorize potency claims outside the frozen applicability lane.",
            "No generator, morphology proposal, support threshold, or terminal ranking rule was changed.",
        ],
    }
    content["result_sha256"] = hashlib.sha256(_stable_json(content).encode()).hexdigest()
    return content


__all__ = [
    "UgiMorphologyHighPotencyChallengerError",
    "build_high_potency_challenger",
    "classification_metrics",
]
