"""Run all nine real-checkpoint/family pairs twice, preserving every attempt."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from forge.commands._generation_runtime import generate
from forge.commands.generate import PROGRAMS, check_inputs, source_identity, write_outputs
from forge.core.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new qualification directory")
    args.output.mkdir(parents=True)
    identity = source_identity(args.root)
    cells = []
    for replicate in range(3):
        report = check_inputs(args.root, replicate=replicate)
        for family, program in PROGRAMS.items():
            kwargs = dict(
                program_id=program, count=2, seed=42, device="cpu", batch_size=2, threads=2
            )
            rows, details = generate(report, **kwargs)
            repeated, again = generate(report, **kwargs)
            if rows != repeated or len(rows) != 2:
                raise ValueError(f"determinism/denominator failure: {replicate}/{family}")
            cell = {
                "replicate": replicate,
                "family": family,
                "request": kwargs,
                "deterministic_replay": True,
                "inputs": report,
                "runtime": details,
                "repeated_runtime": again,
                "counts": {
                    "attempts": len(rows),
                    "valid": sum(row["valid"] for row in rows),
                    "exact_l1": sum(row["exact_l1_program"] for row in rows),
                },
            }
            destination = args.output / f"seed{replicate}-{family}"
            write_outputs(destination, rows, cell)
            cells.append(
                {**cell, "attempts_sha256": str(sha256_file(destination / "attempts.jsonl"))}
            )
            print(f"seed{replicate} {family}: deterministic; {cell['counts']}", flush=True)
    if source_identity(args.root) != identity:
        raise ValueError("source changed during checkpoint qualification")
    receipt = {
        "status": "pass",
        "scope": "18 unique attempts, repeated; not paper metric replication",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "source": identity,
        "cells": cells,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
