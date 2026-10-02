"""Render decoder ablation from three pinned seed-0 evaluation artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, read_json_object, write_json

CONFIG_SCHEMA = "forge.decoder_ablation_config.v1"
RESULT_SCHEMA = "forge.decoder_ablation_render.v1"
EVALUATION_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"


class DecoderAblationError(ValueError):
    """A decoder/source ablation result is missing, mismatched, or inadmissible."""


def _format_percent(value: Any, *, label: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise DecoderAblationError(f"{label} is not a probability")
    return f"{100.0 * float(value):.2f}"


def _render_row(
    specification: Mapping[str, Any],
    payload: Mapping[str, Any],
    *,
    checkpoint_step: int,
    split: str,
    program_id: str,
    attempts: int,
    seed: int,
) -> tuple[str, dict[str, Any]]:
    arm_id = specification.get("arm_id")
    policy = specification.get("terminal_decode_policy")
    if (
        payload.get("schema_version") != EVALUATION_SCHEMA
        or payload.get("status") != "pass"
        or payload.get("replicate") != 0
        or payload.get("seed") != seed
        or payload.get("terminal_decode_policy") != policy
        or payload.get("selection", {}).get("candidate_selection") is not False
        or payload.get("gates", {}).get("coverage_and_precision_reported") is not True
        or payload.get("gates", {}).get("reductive_amination_substructure_rate_absent") is not True
    ):
        raise DecoderAblationError(f"inadmissible Table 5 evaluation for {arm_id}")
    checkpoint_metrics = payload.get("checkpoint_metrics")
    try:
        metrics = checkpoint_metrics[arm_id][str(checkpoint_step)][split][program_id]
    except (KeyError, TypeError) as error:
        raise DecoderAblationError(f"Table 5 metrics are missing for {arm_id}") from error
    if not isinstance(metrics, Mapping) or metrics.get("samples") != attempts:
        raise DecoderAblationError(f"Table 5 attempt denominator changed for {arm_id}")
    values = {
        "valid_percent": _format_percent(
            metrics.get("raw_valid_fraction"), label=f"{arm_id} validity"
        ),
        "coverage_percent": _format_percent(
            metrics.get("exact_l1_decomposition_coverage"), label=f"{arm_id} coverage"
        ),
        "exact_l1_percent": _format_percent(
            metrics.get("exact_l1_yield_per_attempt"), label=f"{arm_id} exact-L1"
        ),
    }
    labels = [
        str(specification.get("source_label")),
        str(specification.get("decoder_label")),
        rf"{values['valid_percent']}\%",
        rf"{values['coverage_percent']}\%",
        rf"{values['exact_l1_percent']}\%",
    ]
    if specification.get("highlight") is True:
        labels = [rf"\cellcolor{{forgerow}}{{{value}}}" for value in labels]
    row = " & ".join(labels) + r" \\"
    return row, values


def render_decoder_ablation_table(
    config_path: Path,
    repo: Path,
    row_path: Path,
    *,
    result_path: Path,
) -> dict[str, Any]:
    """Render the completed seed-0 decoder/source ablation rows."""

    config = read_json_object(
        config_path, error=DecoderAblationError, label="decoder ablation config"
    )
    expected_keys = {
        "schema_version",
        "status",
        "checkpoint_step",
        "program_id",
        "split",
        "attempts_per_row",
        "seed",
        "rows",
        "candidate_selection",
    }
    rows = config.get("rows")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "frozen_after_seed0_evaluations"
        or config.get("candidate_selection") is not False
        or set(config) != expected_keys
        or not isinstance(rows, list)
        or len(rows) != 3
    ):
        raise DecoderAblationError("decoder ablation config changed")
    rendered_rows = []
    rendered_values = []
    sources = []
    checkpoint_hashes = []
    for specification in rows:
        if not isinstance(specification, Mapping):
            raise DecoderAblationError("decoder ablation row specification is malformed")
        path = resolve_pin(specification.get("result"), repo, label="decoder ablation evaluation")
        payload = read_json_object(
            path, error=DecoderAblationError, label="decoder ablation evaluation"
        )
        row, values = _render_row(
            specification,
            payload,
            checkpoint_step=int(config["checkpoint_step"]),
            split=str(config["split"]),
            program_id=str(config["program_id"]),
            attempts=int(config["attempts_per_row"]),
            seed=int(config["seed"]),
        )
        rendered_rows.append(row)
        rendered_values.append(
            {
                "source_label": specification["source_label"],
                "decoder_label": specification["decoder_label"],
                **values,
            }
        )
        sources.append(pin_record(path, repo))
        checkpoint = payload.get("checkpoint_archive")
        if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("sha256"), str):
            raise DecoderAblationError("decoder ablation checkpoint receipt is missing")
        checkpoint_hashes.append(str(checkpoint["sha256"]))
    if checkpoint_hashes[0] != checkpoint_hashes[1]:
        raise DecoderAblationError("decoder ablation rows do not share trained weights")

    # Keep the terminal booktabs rule in the included fragment.  A \noalign-based
    # rule placed immediately after an alignment-ending \input is rejected by TeX.
    atomic_write(row_path, ("\n".join(rendered_rows) + "\n\\bottomrule\n").encode("utf-8"))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "config": pin_record(config_path, repo),
        "sources": sources,
        "seed": config["seed"],
        "checkpoint_step": config["checkpoint_step"],
        "attempts_per_row": config["attempts_per_row"],
        "rows": rendered_values,
        "artifact": artifact_record(row_path, logical_path=row_path.name),
        "gates": {
            "all_three_rows_hash_pinned": True,
            "same_weights_for_decoder_ablation": True,
            "fixed_attempt_denominator": True,
            "candidate_selection_absent": True,
            "coverage_and_precision_reported": True,
        },
        "candidate_selection": False,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(result_path, result)
    return result


__all__ = ["DecoderAblationError", "render_decoder_ablation_table"]
