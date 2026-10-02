"""Freeze one decision-aligned morphology proposal for an untouched comparison."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.experiments.hela.morphology.ugi_morphology_high_potency_challenger import _model
from forge.experiments.hela.morphology.ugi_morphology_potency_proposal_sweep import (
    _candidate_distribution,
    _read_jsonl_gzip,
    _stable_state,
    proposal_metrics,
)
from forge.experiments.hela.morphology.ugi_morphology_potency_signal import (
    _cohort,
    _program_by_product,
    _read_csv_gzip,
    _read_json,
    _sha256_file,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_high_potency_proposal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_high_potency_proposal.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_high_potency_proposal_ledger.v1"


class UgiMorphologyHighPotencyProposalError(RuntimeError):
    """Raised when the frozen proposal calculation changes or fails."""


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMorphologyHighPotencyProposalError(f"invalid input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != record["sha256"]:
        raise UgiMorphologyHighPotencyProposalError(f"input changed: {label}")
    return path


def _csv_gzip_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(buffer.getvalue().encode())
    return raw.getvalue()


def build_high_potency_proposal(repo: Path, config_path: Path) -> tuple[dict[str, Any], bytes]:
    """Fit the frozen classifier and select one anti-collapse proposal."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _read_json(config_path)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMorphologyHighPotencyProposalError("unsupported config schema")
    paths = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    if set(paths) != {
        "challenger_result",
        "curated_agile",
        "heldout_applicability",
        "training_cache",
        "promoted_proposal_ledger",
        "promoted_proposal_result",
    }:
        raise UgiMorphologyHighPotencyProposalError("input set changed")
    challenger = _read_json(paths["challenger_result"])
    promoted = _read_json(paths["promoted_proposal_result"])
    if (
        challenger.get("decision", {}).get("new_generator_run_authorized") is not True
        or challenger.get("gates", {}).get("all_pass") is not True
        or promoted.get("decision", {}).get("applicability_proposal_promoted") is not True
    ):
        raise UgiMorphologyHighPotencyProposalError("prerequisite decision changed")

    proposal_rows = _read_jsonl_gzip(paths["promoted_proposal_ledger"])
    proposal_by_program = {str(row["program_sha256"]): row for row in proposal_rows}
    if len(proposal_by_program) != 57190:
        raise UgiMorphologyHighPotencyProposalError("complete morphology support changed")
    cohort = _cohort(
        curated=_read_csv_gzip(paths["curated_agile"]),
        applicability=_read_csv_gzip(paths["heldout_applicability"]),
        program_by_product=_program_by_product(paths["training_cache"]),
        proposal_by_program=proposal_by_program,
    )
    categorical = np.asarray([row["role_factorized_features"] for row in cohort], dtype=object)
    observed = np.asarray([row["observed_hela_mtp"] for row in cohort], dtype=np.float64)
    potency_threshold = float(np.quantile(observed, 0.75))
    labels = (observed >= potency_threshold).astype(np.int64)
    regularization = float(config["model"]["regularization"])
    model = _model("role_factorized_logistic", regularization).fit(categorical, labels)
    seen = [set(categorical[:, index]) for index in range(3)]

    rows = sorted(proposal_rows, key=lambda row: str(row["program_sha256"]))
    broad = np.asarray([float(row["broad_prior_probability"]) for row in rows])
    support = np.asarray([float(row["proposal_probability"]) for row in rows])
    support_scores = np.asarray([float(row["support_score"]) for row in rows])
    all_categorical = np.asarray(
        [
            [
                _stable_state(row["program"], 0),
                _stable_state(row["program"], 1),
                _stable_state(row["program"], 2),
                _stable_state(row["program"], 1) + "|" + _stable_state(row["program"], 2),
            ]
            for row in rows
        ],
        dtype=object,
    )
    authorized = np.asarray(
        [all(state[index] in seen[index] for index in range(3)) for state in all_categorical],
        dtype=bool,
    )
    probability = np.asarray(model.predict_proba(all_categorical)[:, 1], dtype=np.float64)
    utility = probability.copy()
    utility[~authorized] = 0.0
    expected_yield = support_scores * utility
    baseline = proposal_metrics(rows, broad, support, expected_yield, authorized)

    safeguards = config["safeguards"]
    candidates: list[dict[str, Any]] = []
    for gamma in config["sweep"]["gamma"]:
        for beta in config["sweep"]["beta"]:
            proposal = _candidate_distribution(
                support, expected_yield, gamma=float(gamma), beta=float(beta)
            )
            metrics = proposal_metrics(rows, broad, proposal, expected_yield, authorized)
            checks = {
                "full_support": bool(np.all(proposal > 0.0)),
                "effective_program_count": metrics["inverse_simpson_effective_program_count"]
                >= baseline["inverse_simpson_effective_program_count"]
                * float(safeguards["minimum_effective_program_fraction"]),
                "shannon_support": metrics["shannon_effective_program_count"]
                >= baseline["shannon_effective_program_count"]
                * float(safeguards["minimum_shannon_effective_fraction"]),
                "importance_ess": metrics["importance_ess_fraction_broad_over_proposal"]
                >= float(safeguards["minimum_importance_ess_fraction"]),
                "maximum_probability": metrics["maximum_program_probability"]
                <= baseline["maximum_program_probability"]
                * float(safeguards["maximum_program_probability_multiplier"]),
                "role_marginals": all(
                    metrics["role_marginal_effective_state_counts"][role]
                    >= baseline["role_marginal_effective_state_counts"][role]
                    * float(safeguards["minimum_role_effective_state_fraction"])
                    for role in ("amine", "aldehyde", "isocyanide")
                ),
                "expected_value_improvement": metrics["expected_supported_potency_value"]
                >= baseline["expected_supported_potency_value"]
                * (1.0 + float(safeguards["minimum_expected_value_relative_improvement"])),
            }
            candidates.append(
                {
                    "gamma": float(gamma),
                    "beta": float(beta),
                    "metrics": metrics,
                    "checks": checks,
                    "all_safeguards_pass": all(checks.values()),
                    "proposal": proposal,
                }
            )
    passing = [row for row in candidates if row["all_safeguards_pass"]]
    selected = max(
        passing,
        key=lambda row: (
            row["metrics"]["expected_supported_potency_value"],
            row["metrics"]["inverse_simpson_effective_program_count"],
            -row["gamma"],
            -row["beta"],
        ),
        default=None,
    )
    selected_proposal = support if selected is None else selected["proposal"]
    ledger_rows = [
        {
            "program_sha256": row["program_sha256"],
            "broad_prior_probability": broad[index],
            "support_proposal_probability": support[index],
            "potency_proposal_probability": selected_proposal[index],
            "authorized_morphology": bool(authorized[index]),
            "predicted_high_potency_probability": (probability[index] if authorized[index] else ""),
            "potency_utility": utility[index],
            "support_score": support_scores[index],
            "expected_supported_potency_value": expected_yield[index],
        }
        for index, row in enumerate(rows)
    ]
    fields = (
        "program_sha256",
        "broad_prior_probability",
        "support_proposal_probability",
        "potency_proposal_probability",
        "authorized_morphology",
        "predicted_high_potency_probability",
        "potency_utility",
        "support_score",
        "expected_supported_potency_value",
    )
    ledger = _csv_gzip_bytes(ledger_rows, fields)
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "decision_aligned_nested_potency_proposal_frozen",
        "config": {"path": str(config_path.relative_to(repo)), "sha256": _sha256_file(config_path)},
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": _sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "value_model": {
            "model": "role_factorized_logistic",
            "regularization": regularization,
            "fit_records": len(cohort),
            "target": "upper quartile of observed HeLa MTP",
            "potency_threshold": potency_threshold,
            "expected_yield": "frozen support score times high-potency probability",
            "unknown_role_state_policy": "zero potency utility without removing morphology support",
        },
        "authorized_morphology_support": {
            "programs": int(authorized.sum()),
            "fraction_of_programs": float(authorized.mean()),
            "broad_prior_mass": float(broad[authorized].sum()),
            "support_proposal_mass": float(support[authorized].sum()),
        },
        "baseline_support_proposal": baseline,
        "sweep": [
            {key: value for key, value in row.items() if key != "proposal"} for row in candidates
        ],
        "selection": (
            None
            if selected is None
            else {
                "gamma": selected["gamma"],
                "beta": selected["beta"],
                "metrics": selected["metrics"],
                "checks": selected["checks"],
                "expected_value_relative_improvement": selected["metrics"][
                    "expected_supported_potency_value"
                ]
                / baseline["expected_supported_potency_value"]
                - 1.0,
            }
        ),
        "artifacts": {
            "proposal_ledger.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(ledger_rows),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "decision": {
            "nested_potency_proposal_frozen_for_matched_diagnostic": selected is not None,
            "potency_tilting_promoted": False,
            "prospective_candidate_selection_authorized": False,
            "next_gate": (
                "one_untouched_matched_classifier_proposal_terminal_comparison"
                if selected is not None
                else "close_morphology_potency_tilting_and_retain_terminal_ranking"
            ),
        },
        "nonclaims": [
            "Proposal-value improvement is not realized terminal-potency improvement.",
            "This proposal retains the same conservative morphology authorization as the prior test.",
            "All qualified programs retain nonzero probability.",
        ],
    }
    content["result_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return content, ledger


__all__ = ["UgiMorphologyHighPotencyProposalError", "build_high_potency_proposal"]
