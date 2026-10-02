"""Read-only adjudication of the final Ugi morphology-potency challenger."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file as _sha256_file

RESULT_SCHEMA_VERSION = "forge.ugi_high_potency_challenger_adjudication.v1"


class HighPotencyChallengerAdjudicationError(ValueError):
    """Raised when a frozen challenger input violates the adjudication contract."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HighPotencyChallengerAdjudicationError(f"invalid input: {path}") from error
    if not isinstance(value, dict):
        raise HighPotencyChallengerAdjudicationError(f"input is not an object: {path}")
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _pattern(result: Mapping[str, Any], arm: str, pattern: str) -> Mapping[str, Any]:
    try:
        record = result["arms"][arm]["patterns"][pattern]
    except (KeyError, TypeError) as error:
        raise HighPotencyChallengerAdjudicationError(
            f"missing result for {arm}/{pattern}"
        ) from error
    if not isinstance(record, Mapping):
        raise HighPotencyChallengerAdjudicationError("pattern record is malformed")
    return record


def adjudicate_high_potency_challenger(
    *,
    signal_path: Path,
    proposal_path: Path,
    generation_path: Path,
    ranking_path: Path,
) -> dict[str, Any]:
    """Close the single pre-authorized potency challenger without retuning."""

    signal = _load(signal_path)
    proposal = _load(proposal_path)
    generation = _load(generation_path)
    ranking = _load(ranking_path)
    if signal.get("status") != "decision_aligned_morphology_potency_challenger_audited":
        raise HighPotencyChallengerAdjudicationError("signal audit status changed")
    if proposal.get("status") != "decision_aligned_nested_potency_proposal_frozen":
        raise HighPotencyChallengerAdjudicationError("proposal status changed")
    if generation.get("status") != "complete_matched_morphology_allocation_terminal_generation":
        raise HighPotencyChallengerAdjudicationError("generation status changed")
    if ranking.get("status") != "continuous_novelty_matched_ranking_complete":
        raise HighPotencyChallengerAdjudicationError("ranking status changed")

    tail_pattern = "aldehyde_isocyanide_pair"
    support = _pattern(ranking, "support_enriched", tail_pattern)
    potency = _pattern(ranking, "nested_potency", tail_pattern)
    comparison = ranking["support_vs_nested_potency"][tail_pattern]
    valid = generation["counts"]["valid_exact_l1_per_arm"]
    lower = float(comparison["confidence_interval_low"])
    point = float(comparison["point_difference_unique_high_products_per_generator_call"])
    promoted = lower > 0.0 and point > 0.0
    if ranking["interpretation"].get("potency_tilting_promoted") is not False:
        raise HighPotencyChallengerAdjudicationError("ranking made an unauthorized promotion")

    content: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "high_potency_challenger_adjudicated",
        "inputs": {
            "signal_audit": {"path": str(signal_path), "sha256": _sha256_file(signal_path)},
            "frozen_proposal": {
                "path": str(proposal_path),
                "sha256": _sha256_file(proposal_path),
            },
            "fresh_generation": {
                "path": str(generation_path),
                "sha256": _sha256_file(generation_path),
            },
            "fresh_ranking": {"path": str(ranking_path), "sha256": _sha256_file(ranking_path)},
        },
        "preterminal_signal": {
            "group_disjoint_roc_auc": signal["selection"]["primary_metrics"]["roc_auc"],
            "roc_auc_ci95": signal["selection"]["clustered_bootstrap"]["roc_auc_ci95"],
            "top_quartile_observed_gain": signal["selection"]["primary_metrics"][
                "top_probability_quartile_observed_gain"
            ],
            "top_quartile_gain_ci95": signal["selection"]["clustered_bootstrap"][
                "top_probability_quartile_observed_gain_ci95"
            ],
            "interpretation": (
                "Measured morphology carries population-level HeLa rank signal, but this is not "
                "sufficient evidence that reallocating terminal generation improves candidate yield."
            ),
        },
        "fresh_matched_terminal_gate": {
            "generator_calls_per_arm": generation["counts"]["draws_per_arm"],
            "valid_exact_l1": {
                "support_enriched": valid["support_enriched"],
                "nested_potency": valid["nested_potency"],
            },
            "equal_oracle_budget": ranking["matched_oracle_budgets_per_arm"][tail_pattern],
            "authorized_tail_lane": {
                "support_eligible_before_budget": support["eligible_before_budget"],
                "potency_eligible_before_budget": potency["eligible_before_budget"],
                "support_unique_conservative_high": support[
                    "unique_conservative_high_potency_products"
                ],
                "potency_unique_conservative_high": potency[
                    "unique_conservative_high_potency_products"
                ],
                "difference_per_generator_call": point,
                "difference_ci95": [
                    float(comparison["confidence_interval_low"]),
                    float(comparison["confidence_interval_high"]),
                ],
            },
            "promotion_gate_passed": promoted,
        },
        "diagnosis": {
            "target_mismatch_was_worth_testing": True,
            "target_mismatch_rescued_terminal_efficiency": False,
            "failure_is_not_validity_collapse": abs(
                int(valid["support_enriched"]) - int(valid["nested_potency"])
            )
            < 0.01 * int(generation["counts"]["draws_per_arm"]),
            "interpretation": (
                "The low-capacity classifier detects average potency differences among measured "
                "morphologies, but the selected nested proposal did not translate that weak signal "
                "into more eligible or conservative-high-potency generated products."
            ),
        },
        "decision": {
            "potency_morphology_tilting_promoted": promoted,
            "potency_morphology_tilting_branch_closed": not promoted,
            "production_biological_method": (
                "promoted applicability-enriched morphology proposal followed by frozen continuous "
                "support assessment and conservative terminal potency ranking"
            ),
            "additional_same_data_potency_proposals_authorized": False,
            "prospective_candidate_lock_authorized_by_this_result": False,
            "paper_blocked_without_potency_tilting": False,
        },
        "nonclaims": [
            "The negative generator comparison does not imply that morphology contains no biological signal.",
            "Terminal ranking does not establish calibrated absolute potency for exact-new components.",
            "Applicability enrichment remains independently promoted and is unaffected by this result.",
        ],
    }
    return {**content, "result_sha256": _stable_hash(content)}


__all__ = [
    "HighPotencyChallengerAdjudicationError",
    "RESULT_SCHEMA_VERSION",
    "adjudicate_high_potency_challenger",
]
