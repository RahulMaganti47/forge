"""Matched molecular calibration over every checkpoint in one Ugi ablation arm."""

from __future__ import annotations

import shutil
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.phase1.product_l1.evaluation.ugi_v0_transformer_assessment import (
    assess_native_ugi_method,
)
from experiments.phase1.product_l1.sampling.ugi_joint_end_to_end_sampling import (
    joint_sampling_result_matches_request,
    sample_ugi_joint_end_to_end,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json

CONFIG_SCHEMA = "forge.ugi_tree_transformer_checkpoint_calibration_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_checkpoint_calibration.v1"
REFERENCE_RESULT_SCHEMA = "forge.ugi_reference_checkpoint_calibration.v1"
STATIC_INPUT_LABELS = (
    "program_draw",
    "closure_checkpoint",
    "prepared_cache",
    "qualified_reactions",
    "common_ugi_assessment_config",
    "lipid_realism_config",
    "local_chemistry_config",
    "role_morphology_policy",
)


class UgiTreeTransformerCheckpointCalibrationError(ValueError):
    """A checkpoint archive or calibration execution violated the matched contract."""


def extract_authenticated_checkpoint_archive(
    archive_path: Path,
    output_dir: Path,
    expected: Sequence[Mapping[str, Any]],
) -> dict[int, Path]:
    """Extract only declared regular checkpoint members and verify every digest."""

    expected_by_name: dict[str, Mapping[str, Any]] = {}
    for record in expected:
        name = str(record.get("member") or record.get("path") or "")
        if (
            not name.startswith("checkpoint_step_")
            or not name.endswith(".pt")
            or Path(name).name != name
            or not isinstance(record.get("sha256"), str)
        ):
            raise UgiTreeTransformerCheckpointCalibrationError(
                "training result contains an invalid checkpoint member"
            )
        if name in expected_by_name:
            raise UgiTreeTransformerCheckpointCalibrationError(
                "training result contains duplicate checkpoint members"
            )
        expected_by_name[name] = record
    if not expected_by_name:
        raise UgiTreeTransformerCheckpointCalibrationError("training result has no checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r") as archive:
        for member in archive.getmembers():
            if not member.isfile() or member.name not in expected_by_name:
                raise UgiTreeTransformerCheckpointCalibrationError(
                    f"checkpoint archive contains an undeclared member: {member.name!r}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise UgiTreeTransformerCheckpointCalibrationError(
                    f"checkpoint archive member cannot be read: {member.name!r}"
                )
            target = output_dir / member.name
            payload = source.read()
            target.write_bytes(payload)
            expected_sha = str(expected_by_name[member.name]["sha256"])
            if sha256_file(target) != expected_sha:
                target.unlink(missing_ok=True)
                raise UgiTreeTransformerCheckpointCalibrationError(
                    f"checkpoint archive member hash changed: {member.name!r}"
                )
            observed.add(member.name)
    if observed != set(expected_by_name):
        missing = sorted(set(expected_by_name).difference(observed))
        raise UgiTreeTransformerCheckpointCalibrationError(
            f"checkpoint archive is incomplete: {missing}"
        )
    return {
        int(expected_by_name[name]["step"]): output_dir / name for name in sorted(expected_by_name)
    }


def _validate_config(config: Mapping[str, Any], *, profile: str) -> Mapping[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise UgiTreeTransformerCheckpointCalibrationError(
            "unsupported checkpoint-calibration config"
        )
    inputs = config.get("inputs")
    profiles = config.get("profiles")
    policy = config.get("policy")
    if not isinstance(inputs, Mapping) or set(inputs) != set(STATIC_INPUT_LABELS):
        raise UgiTreeTransformerCheckpointCalibrationError(
            "checkpoint-calibration static inputs changed"
        )
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise UgiTreeTransformerCheckpointCalibrationError(
            f"checkpoint calibration has no {profile!r} profile"
        )
    if policy != {
        "paired_program_order": True,
        "paired_random_streams": True,
        "repairs_or_retries": False,
        "reference_comparison": "deferred",
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
        "heldout_rows_used": False,
        "checkpoint_selection_deferred_to_adjudicator": True,
    }:
        raise UgiTreeTransformerCheckpointCalibrationError("checkpoint-calibration policy changed")
    return profiles[profile]


def _native_rows(result: Mapping[str, Any], *, expected: int) -> list[dict[str, Any]]:
    rows = result.get("samples")
    if not isinstance(rows, list) or len(rows) != expected:
        raise UgiTreeTransformerCheckpointCalibrationError(
            "checkpoint sampling denominator changed"
        )
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or bool(row.get("valid")) != bool(row.get("smiles")):
            raise UgiTreeTransformerCheckpointCalibrationError(
                f"checkpoint sample {index} has inconsistent validity"
            )
    return [dict(row) for row in rows]


def _validate_program_draw(
    program_draw: Mapping[str, Any], *, program_count: int
) -> None:
    if (
        program_draw.get("fold") != "calibration"
        or program_draw.get("heldout_rows_used") is not False
        or len(program_draw.get("samples", [])) < program_count
    ):
        raise UgiTreeTransformerCheckpointCalibrationError(
            "checkpoint calibration draw is not calibration-only"
        )


def sample_and_assess_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_step: int,
    arm_id: str,
    method_id: str,
    assessment_seed_label: int,
    runtime: Mapping[str, Any],
    inputs: Mapping[str, Path],
    repo: Path,
    output_dir: Path,
    device: str,
    heldout_rows_used: bool,
) -> dict[str, Any]:
    """Run one checkpoint through the common paired sampler and assessors.

    Calibration and post-selection production evaluation share this exact execution path.  Their
    fold semantics remain explicit in ``heldout_rows_used`` rather than being inferred from a file
    name.
    """

    program_count = int(runtime["program_count"])
    flow_seed = int(runtime["flow_seed"])
    raw_terminal_seed = runtime.get("terminal_decoder_seed")
    terminal_seed = None if raw_terminal_seed is None else int(raw_terminal_seed)
    terminal_mode = str(runtime["terminal_decoder_mode"])
    if (terminal_mode == "argmax") != (terminal_seed is None):
        raise UgiTreeTransformerCheckpointCalibrationError(
            "argmax decoding requires a null terminal seed and stochastic decoding requires "
            "an explicit terminal seed"
        )
    sampling_dir = output_dir / "sampling"
    sampling_path = sampling_dir / "result.json"
    if sampling_path.is_file():
        sampling = read_json_object(
            sampling_path,
            error=UgiTreeTransformerCheckpointCalibrationError,
            label=f"step-{checkpoint_step} sampling result",
        )
        if not joint_sampling_result_matches_request(
            sampling,
            seed=flow_seed,
            program_offset=0,
            program_limit=program_count,
            terminal_decoder_mode=terminal_mode,
            terminal_decoder_seed=terminal_seed,
            terminal_temperature=float(runtime["terminal_temperature"]),
            checkpoint_filename=checkpoint_path.name,
            device=device,
        ):
            raise UgiTreeTransformerCheckpointCalibrationError(
                f"persisted step-{checkpoint_step} sampling request changed"
            )
    else:
        if sampling_dir.exists():
            shutil.rmtree(sampling_dir)
        sampling = sample_ugi_joint_end_to_end(
            repo,
            sampling_dir,
            joint_checkpoint_path=checkpoint_path,
            closure_checkpoint_path=inputs["closure_checkpoint"],
            matched_staged_result_path=inputs["program_draw"],
            prepared_cache_path=inputs["prepared_cache"],
            sample_steps=int(runtime["sample_steps"]),
            batch_size=int(runtime["batch_size"]),
            seed=flow_seed,
            overwrite=False,
            maximum_adjacent_branch_runs=tuple(runtime["maximum_adjacent_branch_runs"]),
            qualified_reactions_path=inputs["qualified_reactions"],
            evaluate_exact_l1_terminal_admission=True,
            terminal_decoder_mode=terminal_mode,
            terminal_decoder_seed=terminal_seed,
            terminal_temperature=float(runtime["terminal_temperature"]),
            program_offset=0,
            program_limit=program_count,
            reference_comparison_mode="deferred",
            render=False,
            record_timing=False,
            device=device,
        )
    native_rows = _native_rows(sampling, expected=program_count)
    assessment_dir = output_dir / "assessment"
    assessment_index = assessment_dir / "result_index.json"
    if assessment_index.is_file():
        assessment = read_json_object(
            assessment_index,
            error=UgiTreeTransformerCheckpointCalibrationError,
            label=f"step-{checkpoint_step} assessment index",
        )
        if assessment.get("sampling", {}).get("sha256") != sha256_file(
            sampling_path
        ) or assessment.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint_path):
            raise UgiTreeTransformerCheckpointCalibrationError(
                f"persisted step-{checkpoint_step} assessment inputs changed"
            )
    else:
        if assessment_dir.exists():
            shutil.rmtree(assessment_dir)
        assessed = assess_native_ugi_method(
            native_rows,
            method_id=method_id,
            seed_label=assessment_seed_label,
            repo=repo,
            output_dir=assessment_dir,
            common_ugi_assessment_config=inputs["common_ugi_assessment_config"],
            lipid_realism_config=inputs["lipid_realism_config"],
            local_chemistry_config=inputs["local_chemistry_config"],
            role_morphology_policy=inputs["role_morphology_policy"],
        )
        assessment = {
            "schema_version": "forge.ugi_tree_transformer_checkpoint_assessment.v1",
            "status": "complete",
            "arm_id": arm_id,
            "checkpoint_step": checkpoint_step,
            "checkpoint": artifact_record(checkpoint_path),
            "sampling": artifact_record(sampling_path),
            "assessment": assessed,
            "sampling_device": device,
            "checkpoint_selection": False,
            "candidate_selection": False,
            "heldout_rows_used": heldout_rows_used,
        }
        write_json(assessment_index, assessment)
    return assessment


def run_checkpoint_calibration(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    training_result_path: Path,
    checkpoint_archive_path: Path,
    profile: str,
    resume: bool,
    device: str = "cpu",
) -> dict[str, Any]:
    """Sample and assess every prespecified checkpoint under one paired program draw."""

    config = read_json_object(
        config_path,
        error=UgiTreeTransformerCheckpointCalibrationError,
        label="tree-Transformer checkpoint calibration config",
    )
    runtime = _validate_config(config, profile=profile)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiTreeTransformerCheckpointCalibrationError(
            f"checkpoint calibration output is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(config["inputs"][label], repo, label=label)
        for label in STATIC_INPUT_LABELS
    }
    program_draw = read_json_object(
        inputs["program_draw"],
        error=UgiTreeTransformerCheckpointCalibrationError,
        label="calibration program draw",
    )
    _validate_program_draw(program_draw, program_count=int(runtime["program_count"]))
    training = read_json_object(
        training_result_path,
        error=UgiTreeTransformerCheckpointCalibrationError,
        label="ablation training result",
    )
    arm = training.get("experiment_arm")
    snapshots = training.get("checkpoint_snapshots")
    if (
        training.get("status") != "complete"
        or not isinstance(arm, Mapping)
        or not isinstance(arm.get("arm_id"), str)
        or not isinstance(snapshots, list)
    ):
        raise UgiTreeTransformerCheckpointCalibrationError("ablation training result is incomplete")
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints = extract_authenticated_checkpoint_archive(
        checkpoint_archive_path, checkpoints_dir, snapshots
    )
    expected_steps = tuple(int(value) for value in runtime["checkpoint_steps"])
    if not set(expected_steps).issubset(checkpoints):
        raise UgiTreeTransformerCheckpointCalibrationError(
            "checkpoint archive lacks a prespecified step"
        )

    program_count = int(runtime["program_count"])
    assessments: dict[str, Any] = {}
    for step in expected_steps:
        step_dir = output_dir / f"step_{step:04d}"
        assessment = sample_and_assess_checkpoint(
            checkpoint_path=checkpoints[step],
            checkpoint_step=step,
            arm_id=str(arm["arm_id"]),
            method_id=f"forge_tree_{arm['arm_id']}_step_{step:04d}",
            assessment_seed_label=int(config["assessment_seed_label"]),
            runtime=runtime,
            inputs=inputs,
            repo=repo,
            output_dir=step_dir,
            device=device,
            heldout_rows_used=False,
        )
        assessment_index = step_dir / "assessment" / "result_index.json"
        assessments[str(step)] = {
            "result_index": artifact_record(assessment_index),
            "metrics": assessment["assessment"]["metrics"],
        }

    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete",
        "arm_id": arm["arm_id"],
        "programs_per_checkpoint": program_count,
        "checkpoint_steps": list(expected_steps),
        "paired_program_order": True,
        "paired_random_streams": True,
        "sampling_device": device,
        "inputs": {
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
            "training_result": artifact_record(training_result_path),
            "checkpoint_archive": artifact_record(checkpoint_archive_path),
        },
        "checkpoint_assessments": assessments,
        "checkpoint_selection": "deferred_to_cross_arm_v0_adjudicator",
        "candidate_selection": False,
        "heldout_rows_used": False,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    return result


def run_reference_checkpoint_calibration(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    checkpoint_path: Path,
    profile: str,
    resume: bool,
    device: str = "cpu",
) -> dict[str, Any]:
    """Sample and assess the frozen v0 checkpoint on the identical calibration programs."""

    config = read_json_object(
        config_path,
        error=UgiTreeTransformerCheckpointCalibrationError,
        label="reference checkpoint calibration config",
    )
    runtime = _validate_config(config, profile=profile)
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiTreeTransformerCheckpointCalibrationError(
            f"reference checkpoint calibration output is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(config["inputs"][label], repo, label=label)
        for label in STATIC_INPUT_LABELS
    }
    program_draw = read_json_object(
        inputs["program_draw"],
        error=UgiTreeTransformerCheckpointCalibrationError,
        label="calibration program draw",
    )
    _validate_program_draw(program_draw, program_count=int(runtime["program_count"]))
    step = 3000
    step_dir = output_dir / f"step_{step:04d}"
    assessment = sample_and_assess_checkpoint(
        checkpoint_path=checkpoint_path,
        checkpoint_step=step,
        arm_id="v0_reference",
        method_id="forge_v0_step_3000",
        assessment_seed_label=int(config["assessment_seed_label"]),
        runtime=runtime,
        inputs=inputs,
        repo=repo,
        output_dir=step_dir,
        device=device,
        heldout_rows_used=False,
    )
    assessment_index = step_dir / "assessment" / "result_index.json"
    result = {
        "schema_version": REFERENCE_RESULT_SCHEMA,
        "status": "complete",
        "arm_id": "v0_reference",
        "programs_per_checkpoint": int(runtime["program_count"]),
        "checkpoint_steps": [step],
        "paired_program_order": True,
        "paired_random_streams": True,
        "sampling_device": device,
        "inputs": {
            **{label: pin_record(path, repo) for label, path in sorted(inputs.items())},
            "reference_checkpoint": artifact_record(checkpoint_path),
        },
        "checkpoint_assessments": {
            str(step): {
                "result_index": artifact_record(assessment_index),
                "metrics": assessment["assessment"]["metrics"],
            }
        },
        "checkpoint_selection": False,
        "candidate_selection": False,
        "heldout_rows_used": False,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "REFERENCE_RESULT_SCHEMA",
    "UgiTreeTransformerCheckpointCalibrationError",
    "extract_authenticated_checkpoint_archive",
    "run_checkpoint_calibration",
    "run_reference_checkpoint_calibration",
    "sample_and_assess_checkpoint",
]
