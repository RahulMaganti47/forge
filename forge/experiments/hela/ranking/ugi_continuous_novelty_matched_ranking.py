"""Rank exact-new but continuously interpolative Ugi products.

This is an explicitly exploratory companion to the frozen confirmatory matched
adjudication.  Exact component identity is treated as a reporting/risk stratum,
not as chemical distance.  Only pre-existing pattern-specific calibration
scales are used, every molecular and role view must remain interpolative, and
the result cannot promote guidance or authorize prospective candidate lock.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.experiments.hela.morphology.ugi_morphology_potency_matched_adjudication import (
    EXPECTED_ARMS,
    POTENCY_ARM,
    SUPPORT_ARM,
    _candidate,
    _exact_l1,
)
from forge.experiments.hela.oracle import (
    ROLE_MAP,
    FrozenHeLaOracleWorker,
    HeLaBatchPredictor,
    HeLaPotencyDiagnosticPolicy,
)
from forge.potency.applicability import ugi_distributional_applicability as applicability

CONFIG_SCHEMA_VERSION = "phase1_ugi_continuous_novelty_matched_ranking_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_continuous_novelty_matched_ranking.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_continuous_novelty_matched_ranking_ledger.v1"


class UgiContinuousNoveltyMatchedRankingError(RuntimeError):
    """Raised when the frozen exploratory ranking contract changes."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiContinuousNoveltyMatchedRankingError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiContinuousNoveltyMatchedRankingError(f"{label} must be an object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiContinuousNoveltyMatchedRankingError(f"malformed input pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiContinuousNoveltyMatchedRankingError(f"input escapes repo: {label}") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiContinuousNoveltyMatchedRankingError(f"input changed: {label}")
    return path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise UgiContinuousNoveltyMatchedRankingError("terminal ledger malformed")
    return rows


def _gzip_csv(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fields})
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(buffer.getvalue().encode())
    return raw.getvalue()


def _canonical_context(
    row: Mapping[str, Any], policy: HeLaPotencyDiagnosticPolicy
) -> tuple[dict[str, str], tuple[str, ...], bool, dict[str, str]]:
    candidate = _candidate(row)
    canonical = {
        "product": applicability._canonical(candidate["product_smiles"]),
        **{role: applicability._canonical(candidate[f"{role}_smiles"]) for role in ROLE_MAP},
    }
    unseen = tuple(role for role in ROLE_MAP if canonical[role] not in policy.component_sets[role])
    exact_measured = "\x1f".join(canonical[role] for role in ROLE_MAP) in policy.measured_triples
    return canonical, unseen, exact_measured, candidate


def _classify(
    rows: Sequence[Mapping[str, Any]],
    policy: HeLaPotencyDiagnosticPolicy,
    role_patterns: Mapping[tuple[str, ...], str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        terminal = source.get("native_terminal")
        record: dict[str, Any] = {
            "arm_id": str(source.get("arm_id")),
            "draw_index": int(source.get("draw_index", -1)),
            "program_sha256": str(source.get("program_sha256", "")),
            "exact_l1": False,
            "pattern_id": "",
            "unseen_roles": "",
            "reason": "invalid_or_nonexact_l1",
            "overall_bin": "",
            "view_bins": "",
            "eligible": False,
            "oracle_selected": False,
            "canonical_product": "",
            "canonical_amine": "",
            "canonical_aldehyde": "",
            "canonical_isocyanide": "",
            "oracle_mean": "",
            "oracle_sd": "",
            "conformal_q90": "",
            "lcb90": "",
            "calibration_ecdf": "",
            "potency_utility": 0.0,
            "conservative_high_potency": False,
            "worker_receipt_sha256": "",
        }
        if not isinstance(terminal, Mapping) or not _exact_l1(terminal):
            output.append(record)
            continue
        canonical, unseen, exact_measured, candidate = _canonical_context(source, policy)
        pattern_id = role_patterns.get(unseen)
        record.update(
            {
                "exact_l1": True,
                "pattern_id": pattern_id or "",
                "unseen_roles": ",".join(unseen),
                "canonical_product": canonical["product"],
                "canonical_amine": canonical["amine"],
                "canonical_aldehyde": canonical["aldehyde"],
                "canonical_isocyanide": canonical["isocyanide"],
            }
        )
        if exact_measured:
            record["reason"] = "exact_measured_product_neutral"
        elif pattern_id is None:
            record["reason"] = "novelty_pattern_without_frozen_calibration"
        else:
            classification = policy.classify_candidate_mapping(candidate)
            if (
                classification["canonical"] != canonical
                or classification["unseen_roles"] != unseen
                or classification["pattern_id"] != pattern_id
            ):
                raise UgiContinuousNoveltyMatchedRankingError(
                    "authenticated classification changed"
                )
            record.update(
                {
                    "overall_bin": classification["overall_bin"],
                    "view_bins": ";".join(
                        f"{name}:{value}" for name, value in classification["view_bins"]
                    ),
                }
            )
            if classification["overall_bin"] != "interpolative" or any(
                value != "interpolative" for _, value in classification["view_bins"]
            ):
                record["reason"] = f"distribution_{classification['overall_bin']}"
            else:
                record["reason"] = "eligible_exact_new_continuously_interpolative"
                record["eligible"] = True
                record["candidate"] = candidate
        output.append(record)
    return output


def _select_equal_budgets(
    rows: Sequence[dict[str, Any]], patterns: Sequence[str]
) -> tuple[dict[str, int], dict[str, list[dict[str, Any]]]]:
    budgets: dict[str, int] = {}
    selected = {arm: [] for arm in EXPECTED_ARMS}
    for pattern in patterns:
        eligible = {
            arm: [
                row
                for row in rows
                if row["arm_id"] == arm and row["pattern_id"] == pattern and row["eligible"]
            ]
            for arm in EXPECTED_ARMS
        }
        budget = min(len(values) for values in eligible.values())
        budgets[pattern] = budget
        for arm, values in eligible.items():
            chosen = sorted(
                values,
                key=lambda row: hashlib.sha256(
                    (
                        f"continuous-novelty-v1|{pattern}|{row['draw_index']}|"
                        f"{row['canonical_product']}"
                    ).encode()
                ).digest(),
            )[:budget]
            for row in chosen:
                row["oracle_selected"] = True
            selected[arm].extend(chosen)
    return budgets, selected


def _score(
    selected: Mapping[str, Sequence[dict[str, Any]]],
    policy: HeLaPotencyDiagnosticPolicy,
    predictor: HeLaBatchPredictor,
) -> None:
    rows = [row for arm in EXPECTED_ARMS for row in selected[arm]]
    if not rows:
        return
    response = predictor.predict([row["candidate"] for row in rows])
    classifications = response.get("classifications")
    prediction = response.get("prediction")
    receipt = response.get("receipt_sha256")
    if (
        response.get("status") != "complete"
        or not isinstance(classifications, list)
        or len(classifications) != len(rows)
        or not isinstance(prediction, Mapping)
        or prediction.get("records") != len(rows)
        or prediction.get("endpoint") != "expt_Hela"
        or not isinstance(receipt, str)
        or len(receipt) != 64
    ):
        raise UgiContinuousNoveltyMatchedRankingError("oracle response changed")
    means = prediction.get("ensemble_mean")
    deviations = prediction.get("ensemble_standard_deviation")
    if not isinstance(means, list) or not isinstance(deviations, list):
        raise UgiContinuousNoveltyMatchedRankingError("oracle predictions absent")
    for row, classification, raw_mean, raw_sd in zip(
        rows, classifications, means, deviations, strict=True
    ):
        if not isinstance(classification, Mapping):
            raise UgiContinuousNoveltyMatchedRankingError("worker classification malformed")
        unseen = tuple(str(value) for value in classification.get("unseen_component_roles", []))
        expected_unseen = tuple(filter(None, str(row["unseen_roles"]).split(",")))
        if (
            classification.get("exact_forward_verified") is not True
            or unseen != expected_unseen
            or classification.get("combination_seen_in_measured_training") is not False
            or classification.get("canonical")
            != {
                "product": row["canonical_product"],
                "amine": row["canonical_amine"],
                "aldehyde": row["canonical_aldehyde"],
                "isocyanide": row["canonical_isocyanide"],
            }
        ):
            raise UgiContinuousNoveltyMatchedRankingError("worker identity changed")
        mean = float(raw_mean)
        deviation = float(raw_sd)
        if not math.isfinite(mean) or not math.isfinite(deviation) or deviation < 0.0:
            raise UgiContinuousNoveltyMatchedRankingError("oracle prediction invalid")
        scale = policy.scales[str(row["pattern_id"])]
        lcb90, cdf, utility = scale.score(mean)
        row.update(
            {
                "oracle_mean": mean,
                "oracle_sd": deviation,
                "conformal_q90": scale.max_q90,
                "lcb90": lcb90,
                "calibration_ecdf": cdf,
                "potency_utility": utility,
                "conservative_high_potency": cdf > 0.5,
                "worker_receipt_sha256": receipt,
            }
        )


def _metrics(rows: Sequence[dict[str, Any]], arm: str, patterns: Sequence[str]) -> dict[str, Any]:
    arm_rows = [row for row in rows if row["arm_id"] == arm]
    output: dict[str, Any] = {
        "generator_calls": len(arm_rows),
        "exact_l1_terminals": sum(bool(row["exact_l1"]) for row in arm_rows),
        "patterns": {},
    }
    for pattern in patterns:
        eligible = [row for row in arm_rows if row["pattern_id"] == pattern and row["eligible"]]
        scored = [row for row in eligible if row["oracle_selected"]]
        high = [row for row in scored if row["conservative_high_potency"]]
        output["patterns"][pattern] = {
            "eligible_before_budget": len(eligible),
            "eligible_fraction_per_generator_call": len(eligible) / len(arm_rows),
            "oracle_calls": len(scored),
            "conservative_high_potency_terminals": len(high),
            "unique_conservative_high_potency_products": len(
                {row["canonical_product"] for row in high}
            ),
            "mean_potency_utility": float(
                np.mean([float(row["potency_utility"]) for row in scored]) if scored else 0.0
            ),
        }
    output["abstention_reasons"] = dict(
        sorted(Counter(str(row["reason"]) for row in arm_rows).items())
    )
    return output


def _paired_bootstrap(
    rows: Sequence[dict[str, Any]], *, pattern: str | None, replicates: int, seed: int
) -> dict[str, float]:
    indexed: dict[str, dict[int, str | None]] = {arm: {} for arm in EXPECTED_ARMS}
    for row in rows:
        active = row["oracle_selected"] and row["conservative_high_potency"]
        if pattern is not None:
            active = active and row["pattern_id"] == pattern
        indexed[str(row["arm_id"])][int(row["draw_index"])] = (
            str(row["canonical_product"]) if active else None
        )
    draw_count = len(indexed[SUPPORT_ARM])
    rng = np.random.default_rng(seed)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        draws = rng.integers(0, draw_count, size=draw_count)
        support = {indexed[SUPPORT_ARM][int(index)] for index in draws}
        potency = {indexed[POTENCY_ARM][int(index)] for index in draws}
        support.discard(None)
        potency.discard(None)
        values[replicate] = (len(potency) - len(support)) / draw_count
    point = (
        len({value for value in indexed[POTENCY_ARM].values() if value is not None})
        - len({value for value in indexed[SUPPORT_ARM].values() if value is not None})
    ) / draw_count
    return {
        "point_difference_unique_high_products_per_generator_call": point,
        "confidence_interval_low": float(np.quantile(values, 0.025)),
        "confidence_interval_high": float(np.quantile(values, 0.975)),
    }


def build_continuous_novelty_matched_ranking(
    repo: Path,
    config_path: Path,
    *,
    predictor: HeLaBatchPredictor | None = None,
) -> tuple[dict[str, Any], bytes]:
    """Run the frozen exploratory exact-new ranking comparison."""

    repo = repo.resolve()
    config = _load(config_path.resolve(), label="continuous-novelty config")
    if (
        config.get("schema_version") != CONFIG_SCHEMA_VERSION
        or config.get("status")
        != "frozen_after_generation_and_confirmatory_null_before_novel_component_oracle_scoring"
    ):
        raise UgiContinuousNoveltyMatchedRankingError("unsupported config")
    paths = {
        label: _pin(repo, record, label=label) for label, record in config.get("inputs", {}).items()
    }
    if set(paths) != {
        "matched_generation_result",
        "matched_terminal_ledger",
        "hela_diagnostic_policy",
        "confirmatory_result",
    }:
        raise UgiContinuousNoveltyMatchedRankingError("input set changed")
    generation = _load(paths["matched_generation_result"], label="generation result")
    confirmatory = _load(paths["confirmatory_result"], label="confirmatory result")
    if (
        generation.get("status") != "complete_matched_morphology_allocation_terminal_generation"
        or generation.get("artifacts", {}).get("terminal_ledger.jsonl.gz", {}).get("sha256")
        != sha256_file(paths["matched_terminal_ledger"])
        or confirmatory.get("decision", {}).get("potency_tilting_promoted") is not False
        or confirmatory.get("matched_budgets", {}).get("oracle_calls_per_arm") != 0
    ):
        raise UgiContinuousNoveltyMatchedRankingError("prerequisite result changed")
    rows = _read_jsonl(paths["matched_terminal_ledger"])
    expected = int(config["matched_design"]["generator_calls_per_arm"])
    if len(rows) != expected * len(EXPECTED_ARMS):
        raise UgiContinuousNoveltyMatchedRankingError("terminal count changed")
    if Counter(str(row.get("arm_id")) for row in rows) != Counter(
        {arm: expected for arm in EXPECTED_ARMS}
    ):
        raise UgiContinuousNoveltyMatchedRankingError("arm budgets changed")
    policy = HeLaPotencyDiagnosticPolicy(repo, paths["hela_diagnostic_policy"])
    patterns = tuple(str(value) for value in config["patterns"])
    role_patterns = {
        tuple(str(role) for role in config["patterns"][pattern]["unseen_roles"]): pattern
        for pattern in patterns
    }
    if set(patterns) != set(policy.scales):
        raise UgiContinuousNoveltyMatchedRankingError("pattern calibration set changed")
    classified = _classify(rows, policy, role_patterns)
    budgets, selected = _select_equal_budgets(classified, patterns)
    if predictor is None:
        with FrozenHeLaOracleWorker(policy) as worker:
            _score(selected, policy, worker)
            worker_receipt = worker.ready_receipt_sha256
    else:
        _score(selected, policy, predictor)
        worker_receipt = "test_predictor"
    arm_metrics = {arm: _metrics(classified, arm, patterns) for arm in EXPECTED_ARMS}
    replicates = int(config["matched_design"]["common_draw_bootstrap_replicates"])
    seed = int(config["matched_design"]["bootstrap_seed"])
    comparisons = {
        pattern: _paired_bootstrap(
            classified,
            pattern=pattern,
            replicates=replicates,
            seed=seed + index,
        )
        for index, pattern in enumerate(patterns)
    }
    comparisons["combined"] = _paired_bootstrap(
        classified, pattern=None, replicates=replicates, seed=seed + len(patterns)
    )
    fields = (
        "arm_id",
        "draw_index",
        "program_sha256",
        "exact_l1",
        "pattern_id",
        "unseen_roles",
        "reason",
        "overall_bin",
        "view_bins",
        "eligible",
        "oracle_selected",
        "canonical_product",
        "canonical_amine",
        "canonical_aldehyde",
        "canonical_isocyanide",
        "oracle_mean",
        "oracle_sd",
        "conformal_q90",
        "lcb90",
        "calibration_ecdf",
        "potency_utility",
        "conservative_high_potency",
        "worker_receipt_sha256",
    )
    ledger = _gzip_csv(classified, fields)
    any_signal = any(
        value["point_difference_unique_high_products_per_generator_call"] > 0
        for value in comparisons.values()
    )
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "continuous_novelty_matched_ranking_complete",
        "scope": dict(config["decisions"]),
        "inputs": {
            label: {"path": str(path.relative_to(repo)), "sha256": sha256_file(path)}
            for label, path in sorted(paths.items())
        },
        "patterns": dict(config["patterns"]),
        "matched_oracle_budgets_per_arm": budgets,
        "worker_ready_receipt_sha256": worker_receipt,
        "arms": arm_metrics,
        "support_vs_nested_potency": comparisons,
        "interpretation": {
            "exploratory_ranking_signal_observed": any_signal,
            "potency_tilting_promoted": False,
            "prospective_candidate_selection_authorized": False,
            "reason": "exact-new identities were ranked only inside continuous support using pre-existing pattern scales; structured simultaneous-component or prospective evidence remains required",
        },
        "artifacts": {
            "terminal_ranking.csv.gz": {
                "schema_version": LEDGER_SCHEMA_VERSION,
                "rows": len(classified),
                "sha256": hashlib.sha256(ledger).hexdigest(),
            }
        },
        "nonclaims": [
            "Exact novelty alone neither establishes nor disproves oracle applicability.",
            "This exploratory ranking does not establish calibrated absolute potency for new components.",
            "This result cannot reverse the frozen confirmatory null or promote potency guidance.",
        ],
    }
    result["result_sha256"] = _sha256_payload(result)
    return result, ledger


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiContinuousNoveltyMatchedRankingError",
    "build_continuous_novelty_matched_ranking",
]
