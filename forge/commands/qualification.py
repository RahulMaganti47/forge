"""Bounded release checks using downloaded artifacts and CPU inference."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from forge.commands.artifacts import verify
from forge.commands.generate import source_identity
from forge.commands.reproduce import verify_manuscript_rows
from forge.core.hashing import sha256_file
from forge.core.io import write_json

GROUPS = ("paper-model-v1", "submission19337-evidence-v1")


def qualify(root: Path, output: Path) -> dict[str, Any]:
    """Verify inputs, replay tables, and repeat two Ugi attempts without training."""
    root, output = root.resolve(), output.resolve()
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists: {output}")
    artifacts = {}
    for group in GROUPS:
        manifest = root / f"manifests/{group}.json"
        report = verify(root, manifest)
        if not report["ready"]:
            failed = [row["path"] for row in report["files"] if row["status"] != "verified"]
            raise ValueError(f"fetch or restore {group} before qualification: {failed}")
        artifacts[group] = report
    source = source_identity(root)
    configs = {
        path.relative_to(root).as_posix(): str(sha256_file(path))
        for path in sorted((root / "configs").rglob("*.json"))
    }
    receipt: dict[str, Any] = {
        "schema_version": "forge.release.reader_qualification.v1",
        "scope": "frozen-table replay and two CPU Ugi attempts repeated; no retraining",
        "source": source,
        "script_sha256": str(sha256_file(root / "examples/check_reproduction.py")),
        "reference_pdf_sha256": str(sha256_file(root / "paper/submission.pdf")),
        "config_sha256s": configs,
        "uv_lock_sha256": str(sha256_file(root / "uv.lock")),
        "artifacts": artifacts,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("torch", "rdkit", "numpy", "pandas", "scikit-learn", "scipy")
            },
            "device": "cpu",
            "precision": "float32",
            "torch_threads": 2,
        },
        "commands": [],
    }
    output.mkdir(parents=True)
    started = time.perf_counter()

    def run(arguments: list[str], name: str) -> None:
        command = [sys.executable, "-m", "forge.cli", *arguments, "--root", str(root)]
        entry: dict[str, Any] = {"argv": command, "log": f"{name}.log"}
        receipt["commands"].append(entry)
        before = time.perf_counter()
        try:
            with (output / entry["log"]).open("w") as log:
                result = subprocess.run(command, cwd=root, stdout=log, stderr=log, check=False)
            entry["exit_code"] = result.returncode
            if result.returncode:
                raise RuntimeError(f"{name} failed; inspect {output / entry['log']}")
        finally:
            entry["elapsed_seconds"] = time.perf_counter() - before

    try:
        run(["reproduce", "--target", "all", "--output", str(output / "tables")], "tables")
        receipt["table_rows_matching_manuscript"] = verify_manuscript_rows(root, output / "tables")
        for name in ("generation", "generation-repeat"):
            run(
                [
                    "generate",
                    "--replicate",
                    "0",
                    "--family",
                    "ugi",
                    "--count",
                    "2",
                    "--seed",
                    "42",
                    "--device",
                    "cpu",
                    "--batch-size",
                    "2",
                    "--threads",
                    "2",
                    "--output",
                    str(output / name),
                ],
                name,
            )
        attempts = (output / "generation/attempts.jsonl").read_bytes()
        repeated = (output / "generation-repeat/attempts.jsonl").read_bytes()
        if attempts != repeated or len(attempts.splitlines()) != 2:
            raise ValueError("generation changed its attempt denominator or deterministic outputs")
        receipt["generation"] = {
            "attempts": 2,
            "deterministic_replay": True,
            "summary": json.loads((output / "generation/summary.json").read_text()),
        }
        if source_identity(root) != source:
            raise ValueError("source changed during qualification")
        if any(sha256_file(root / name) != digest for name, digest in configs.items()):
            raise ValueError("configuration changed during qualification")
        receipt["outputs"] = {
            path.relative_to(output).as_posix(): str(sha256_file(path))
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        receipt["status"] = "pass"
    except Exception as error:
        receipt.update(status="fail", error=str(error))
        write_json(output / "FAILED.json", {"status": "fail", "error": str(error)})
        raise
    finally:
        receipt["elapsed_seconds"] = time.perf_counter() - started
        write_json(output / "receipt.json", receipt)
    return receipt
