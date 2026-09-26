"""Freeze the matched calibration-fold morphology programs for the Ugi challenger ladder."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.training_cache import load_ugi_training_cache_payload
from forge.potency.annotations import ROLE_NAMES

CONFIG_SCHEMA = "forge.ugi_tree_transformer_calibration_draw_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_calibration_program_draw.v1"


class UgiTreeTransformerCalibrationError(ValueError):
    """The calibration draw cannot be frozen under the declared split contract."""


def _apportion(total: int, masses: Mapping[str, float]) -> dict[str, int]:
    """Use deterministic largest-remainder apportionment."""

    if total < 1 or not masses or any(float(value) <= 0.0 for value in masses.values()):
        raise UgiTreeTransformerCalibrationError("calibration apportionment is invalid")
    normalizer = sum(float(value) for value in masses.values())
    raw = {key: total * float(value) / normalizer for key, value in masses.items()}
    quotas = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def _calibration_role_class(assignment: Mapping[str, Any]) -> str:
    roles = tuple(
        role for role in ROLE_NAMES if str(assignment.get(f"{role}_family_fold")) == "calibration"
    )
    if not roles:
        raise UgiTreeTransformerCalibrationError(
            "calibration product has no calibration-fold component family"
        )
    invalid = [
        role
        for role in ROLE_NAMES
        if str(assignment.get(f"{role}_family_fold")) not in {"train", "calibration"}
    ]
    if invalid:
        raise UgiTreeTransformerCalibrationError(
            f"calibration product contains an inadmissible family fold: {invalid}"
        )
    return "+".join(roles)


def _selection_key(seed: int, stratum: str, product_id: str) -> str:
    return hashlib.sha256(f"{seed}|{stratum}|{product_id}".encode()).hexdigest()


def select_calibration_programs(
    assignments: Sequence[Mapping[str, Any]],
    records: Sequence[Any],
    *,
    count: int,
    seed: int,
    source_mass: Mapping[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select paired calibration programs without replacement or component-ID model inputs."""

    if len(assignments) != len(records):
        raise UgiTreeTransformerCalibrationError("calibration assignments and records differ")
    by_product: dict[str, Any] = {}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for assignment, record in zip(assignments, records, strict=True):
        product_id = str(assignment.get("product_id", ""))
        if not product_id or product_id != str(getattr(record, "product_id", "")):
            raise UgiTreeTransformerCalibrationError("calibration product alignment changed")
        if product_id in by_product:
            raise UgiTreeTransformerCalibrationError("calibration product IDs are not unique")
        if str(assignment.get("primary_product_fold")) != "calibration":
            raise UgiTreeTransformerCalibrationError("non-calibration product entered the draw")
        source = str(assignment.get("source_stratum", ""))
        if source not in source_mass:
            raise UgiTreeTransformerCalibrationError(
                f"calibration source stratum is not declared: {source!r}"
            )
        role_class = _calibration_role_class(assignment)
        by_product[product_id] = record
        grouped[(source, role_class)].append(assignment)

    source_quotas = _apportion(count, source_mass)
    selected: list[tuple[Mapping[str, Any], str]] = []
    stratum_quotas: dict[str, int] = {}
    for source in sorted(source_quotas):
        role_classes = sorted(
            role for observed_source, role in grouped if observed_source == source
        )
        if not role_classes:
            raise UgiTreeTransformerCalibrationError(
                f"calibration source has no role strata: {source!r}"
            )
        role_quotas = _apportion(
            source_quotas[source], {role_class: 1.0 for role_class in role_classes}
        )
        for role_class in role_classes:
            label = f"{source}|{role_class}"
            ordered = sorted(
                grouped[(source, role_class)],
                key=lambda row: _selection_key(seed, label, str(row["product_id"])),
            )
            quota = role_quotas[role_class]
            if len(ordered) < quota:
                raise UgiTreeTransformerCalibrationError(
                    f"calibration stratum {label!r} has {len(ordered)} rows for quota {quota}"
                )
            selected.extend((row, role_class) for row in ordered[:quota])
            stratum_quotas[label] = quota

    selected.sort(key=lambda item: _selection_key(seed, "paired-order", str(item[0]["product_id"])))
    if len(selected) != count or len({str(row[0]["product_id"]) for row in selected}) != count:
        raise UgiTreeTransformerCalibrationError("calibration draw denominator changed")

    samples = []
    for assignment, role_class in selected:
        record = by_product[str(assignment["product_id"])]
        program = record.program
        samples.append(
            {
                "product_id": str(assignment["product_id"]),
                "evaluation_fold": "calibration",
                "source_stratum": str(assignment["source_stratum"]),
                "calibration_role_class": role_class,
                "component_novelty_class": str(assignment["component_novelty_class"]),
                "program": {
                    "node_counts": list(program.node_counts),
                    "junction_budgets": list(program.junction_budgets),
                    "cycle_ranks": list(program.cycle_ranks),
                    "attachment_counts": list(program.attachment_counts),
                },
            }
        )
    composition = {
        "source_strata": dict(sorted(Counter(row["source_stratum"] for row in samples).items())),
        "calibration_role_classes": dict(
            sorted(Counter(row["calibration_role_class"] for row in samples).items())
        ),
        "source_by_role_class": dict(
            sorted(
                Counter(
                    f"{row['source_stratum']}|{row['calibration_role_class']}" for row in samples
                ).items()
            )
        ),
    }
    return samples, {
        "source_quotas": dict(sorted(source_quotas.items())),
        "stratum_quotas": dict(sorted(stratum_quotas.items())),
        "composition": composition,
    }


def build_calibration_program_draw(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build and authenticate the calibration-only matched program draw."""

    config = read_json_object(
        config_path,
        error=UgiTreeTransformerCalibrationError,
        label="tree-Transformer calibration draw config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "status",
        "fold",
        "count",
        "seed",
        "source_mass",
        "inputs",
        "selection_policy",
        "nonclaims",
    }:
        raise UgiTreeTransformerCalibrationError("calibration draw config fields changed")
    if (
        config["status"] != "frozen_before_tree_transformer_training"
        or config["fold"] != "calibration"
    ):
        raise UgiTreeTransformerCalibrationError("calibration draw is not pretraining-frozen")
    policy = config["selection_policy"]
    if policy != {
        "without_replacement": True,
        "source_balanced": True,
        "calibration_role_class_balanced_within_source": True,
        "component_identifiers_exposed_to_model": False,
        "heldout_rows_allowed": False,
        "repairs_or_retries": False,
    }:
        raise UgiTreeTransformerCalibrationError("calibration selection policy changed")
    cache_path = resolve_pin(config["inputs"]["prepared_cache"], repo, label="prepared_cache")
    payload = load_ugi_training_cache_payload(cache_path)
    corpus = payload.get("corpus")
    records_by_fold = payload.get("joint_records_by_fold")
    if corpus is None or not isinstance(records_by_fold, Mapping):
        raise UgiTreeTransformerCalibrationError("prepared cache lacks aligned fold records")
    assignments = corpus.assignments_by_fold.get("calibration")
    records = records_by_fold.get("calibration")
    if not isinstance(assignments, tuple) or not isinstance(records, tuple):
        raise UgiTreeTransformerCalibrationError("prepared cache lacks the calibration fold")
    samples, selection = select_calibration_programs(
        assignments,
        records,
        count=int(config["count"]),
        seed=int(config["seed"]),
        source_mass={str(key): float(value) for key, value in config["source_mass"].items()},
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "frozen_before_tree_transformer_training",
        "fold": "calibration",
        "seed": int(config["seed"]),
        "samples": samples,
        "selection": selection,
        "paired_random_streams_required": True,
        "checkpoint_or_arm_selection_authorized": True,
        "candidate_selection": False,
        "heldout_rows_used": False,
        "component_identifiers_exposed_to_model": False,
        "config": pin_record(config_path, repo),
        "inputs": {"prepared_cache": pin_record(cache_path, repo)},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_path, result)
    result["artifact"] = artifact_record(output_path)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiTreeTransformerCalibrationError",
    "build_calibration_program_draw",
    "select_calibration_programs",
]
