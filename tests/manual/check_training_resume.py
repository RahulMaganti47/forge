"""Interrupt CPU smoke training after step 1 and compare its resumed state with a full smoke run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from forge.commands.generate import source_identity
from forge.core.hashing import sha256_file
from forge.experiments import training as production_training
from forge.experiments.study import run_transformer_mechanism_study


class InjectedInterruption(RuntimeError):
    """Test-only interruption immediately after a durable restart checkpoint."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--reference", type=Path, required=True, help="completed smoke training/result.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output directory")
    config = (
        args.root
        / "configs/multireaction/shared_bias_parallel_shared_bias_program_role_source_v2.json"
    )
    kwargs = dict(
        config_path=config,
        repo=args.root,
        output_dir=args.output,
        work_dir=args.output / "work",
        profile="smoke",
        replicate=0,
        allocated_device="cpu",
    )
    original_save = production_training.atomic_torch_save

    def interrupt_after_first_restart(path: Path, payload: object) -> None:
        original_save(path, payload)
        if path.name == "restart_latest.pt" and payload["completed_steps"] == 1:
            raise InjectedInterruption("durable step-1 restart exists")

    production_training.atomic_torch_save = interrupt_after_first_restart
    interrupted = False
    try:
        run_transformer_mechanism_study(**kwargs, resume=False)
    except InjectedInterruption:
        interrupted = True
    finally:
        production_training.atomic_torch_save = original_save
    if not interrupted:
        raise ValueError("the test did not interrupt at step 1")
    run_transformer_mechanism_study(**kwargs, resume=True)
    reference = json.loads(args.reference.read_text())["arms"]
    resumed = json.loads((args.output / "training/result.json").read_text())["arms"]
    if reference != resumed:
        raise ValueError("resumed arm records differ from uninterrupted training")
    receipt = {
        "status": "pass",
        "interrupted_after_optimizer_step": 1,
        "completed_steps": 2,
        "complete_arm_records_equal": True,
        "reference_sha256": str(sha256_file(args.reference)),
        "source": source_identity(args.root),
        "scope": "CPU smoke model; no paid or full training",
    }
    (args.output / "resume_qualification.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        "Step-1 interruption and resumed step-2 training exactly match the uninterrupted smoke run."
    )


if __name__ == "__main__":
    main()
