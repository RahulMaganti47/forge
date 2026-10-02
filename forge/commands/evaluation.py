"""Evaluate released or freshly trained checkpoints through the authenticated evaluator."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import write_json


def evaluate(
    root: Path,
    output: Path,
    *,
    replicate: int,
    device: str,
    profile: str = "paper",
    config: Path | None = None,
    checkpoint: Path | None = None,
    training_result: Path | None = None,
    study_design: Path | None = None,
) -> dict[str, Any]:
    """Require a complete custom input set; preserve every existing evaluation gate."""
    supplied = (config, checkpoint, training_result, study_design)
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise ValueError(
            "custom evaluation requires --config, --checkpoint, --training-result, "
            "and --study-design together"
        )
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    failed = output.with_name(output.name + ".failed")
    if failed.exists() or failed.is_symlink():
        raise ValueError(f"previous failed evaluation exists: {failed}; choose a new output")
    if profile not in {"smoke", "paper"}:
        raise ValueError(f"unsupported evaluation profile: {profile}")
    if profile == "smoke" and device != "cpu":
        raise ValueError("smoke evaluation requires --device cpu")
    runtime_profile = "full" if profile == "paper" else "smoke"
    if config is not None:
        config, checkpoint, training_result, study_design = (
            (root / path).resolve() for path in supplied
        )
        configuration = json.loads(config.read_text())
        cache = resolve_pin(configuration["inputs"]["production_cache"], root, label="cache")
        training = json.loads(training_result.read_text())
        archive_pin = training["checkpoint_archive"]
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != archive_pin["sha256"]
            or checkpoint.stat().st_size != archive_pin["bytes"]
        ):
            raise ValueError("checkpoint archive differs from the supplied training result")
    else:
        from .generate import check_inputs

        report = check_inputs(root, replicate=replicate)
        if not report["ready"]:
            raise ValueError("checkpoint inputs are incomplete; run forge generate --check-inputs")
        paths = {key: Path(pin["path"]) for key, pin in report["inputs"].items()}
        config = (
            root
            / f"configs/multireaction/shared_bias_parallel_program_role_seed{replicate}_core_saturation_v2.json"
        )
        cache, checkpoint, training_result = (
            paths[key] for key in ("production_cache", "checkpoint", "training_result")
        )
        study_design = None

    from forge.experiments.evaluation import (
        run_synthesis_program_production_evaluation,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forge-evaluation-", dir=output.parent) as directory:
        staging = Path(directory) / "result"
        staging.mkdir()
        try:
            result = run_synthesis_program_production_evaluation(
                config,
                root,
                cache,
                checkpoint,
                training_result,
                staging,
                profile=runtime_profile,
                replicate=replicate,
                allocated_device=device,
                dynamic_production_design_path=study_design,
            )
        except Exception as error:
            write_json(staging / "FAILED.json", {"status": "fail", "error": str(error)})
            staging.rename(failed)
            raise
        if output.exists() or output.is_symlink():
            raise ValueError(f"output was created during evaluation: {output}")
        staging.rename(output)
    return result
