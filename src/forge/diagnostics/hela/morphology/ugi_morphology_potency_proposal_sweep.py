"""Freeze a conservative potency proposal nested inside the promoted support proposal."""

from __future__ import annotations

import gzip
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_potency_proposal_sweep_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_potency_proposal_sweep.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_potency_proposal_sweep_ledger.v1"
ROLE_FIELDS = ("amine_role_state", "aldehyde_role_state", "isocyanide_role_state")


class UgiMorphologyPotencyProposalSweepError(RuntimeError):
    """Raised when the frozen potency-proposal sweep contract changes."""


def _read_jsonl_gzip(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiMorphologyPotencyProposalSweepError("invalid proposal ledger")
    return rows


def _stable_state(program: Mapping[str, Any], role: int) -> str:
    return json.dumps(
        [
            int(program["node_counts"][role]),
            int(program["junction_budgets"][role]),
            int(program["cycle_ranks"][role]),
            int(program["attachment_counts"][role]),
        ],
        separators=(",", ":"),
    )


def _effective_count(probabilities: np.ndarray) -> float:
    probabilities = probabilities / probabilities.sum()
    return float(1.0 / np.square(probabilities).sum())


def _shannon_effective_count(probabilities: np.ndarray) -> float:
    probabilities = probabilities / probabilities.sum()
    positive = probabilities[probabilities > 0.0]
    return float(math.exp(-np.dot(positive, np.log(positive))))


def _importance_ess_fraction(broad: np.ndarray, proposal: np.ndarray) -> float:
    ratio = broad / proposal
    weighted = proposal * ratio
    return float(weighted.sum() ** 2 / np.dot(proposal, ratio**2))


def _marginal_effective_counts(
    rows: Sequence[Mapping[str, Any]], probabilities: np.ndarray
) -> dict[str, float]:
    output: dict[str, float] = {}
    for role, name in enumerate(("amine", "aldehyde", "isocyanide")):
        mass: dict[str, float] = defaultdict(float)
        for row, probability in zip(rows, probabilities, strict=True):
            mass[_stable_state(row["program"], role)] += float(probability)
        output[name] = _effective_count(np.asarray(tuple(mass.values()), dtype=np.float64))
    return output


def proposal_metrics(
    rows: Sequence[Mapping[str, Any]],
    broad: np.ndarray,
    proposal: np.ndarray,
    expected_yield: np.ndarray,
    authorized: np.ndarray,
) -> dict[str, Any]:
    """Summarize support, diversity and expected conservative yield."""

    return {
        "expected_supported_potency_value": float(np.dot(proposal, expected_yield)),
        "authorized_morphology_mass": float(proposal[authorized].sum()),
        "inverse_simpson_effective_program_count": _effective_count(proposal),
        "shannon_effective_program_count": _shannon_effective_count(proposal),
        "importance_ess_fraction_broad_over_proposal": _importance_ess_fraction(broad, proposal),
        "maximum_program_probability": float(proposal.max()),
        "role_marginal_effective_state_counts": _marginal_effective_counts(rows, proposal),
    }


def _candidate_distribution(
    support: np.ndarray,
    value: np.ndarray,
    *,
    gamma: float,
    beta: float,
) -> np.ndarray:
    tilted = support * np.exp(beta * value)
    tilted /= tilted.sum()
    output = (1.0 - gamma) * support + gamma * tilted
    return output / output.sum()


__all__ = ["UgiMorphologyPotencyProposalSweepError", "proposal_metrics"]
