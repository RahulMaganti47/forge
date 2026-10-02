"""Bounded paper-model generation; --check-inputs never loads neural weights."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from forge.core.hashing import is_sha256, sha256_file, sha256_json

ROOT = Path.cwd()
MANIFEST = Path("manifests/paper-model-v1.json")
ARM = "shared_bias_program_role_source"
STEP = 9143
FLOW_STEPS = 32
DECODER = "strict_reaction_core_saturation_argmax"
PROGRAMS = {
    "ugi": "ugi_3cr_agile",
    "aza-michael": "bl_2023_repeated_aza_michael",
    "reductive-amination": "lx_2024_repeated_reductive_amination",
}


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _repo_path(root: Path, value: str) -> Path:
    relative = Path(value)
    path = root / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.resolve().is_relative_to(root.resolve())
    ):
        raise ValueError(f"input path escapes repository: {value}")
    return path


def check_inputs(
    root: Path, *, replicate: int = 0, checkpoint: Path | None = None
) -> dict[str, Any]:
    """Hash all direct inputs before importing Torch; relocated bundles retain their identities."""
    if replicate not in (0, 1, 2):
        raise ValueError("replicate must be 0, 1 or 2")
    config_path = (
        root
        / f"configs/multireaction/shared_bias_parallel_program_role_seed{replicate}_core_saturation_v2.json"
    )
    config = _read_object(config_path)
    runtime = config["full"]
    if (
        runtime["checkpoint_steps"] != [STEP]
        or runtime["sample_steps"] != FLOW_STEPS
        or runtime["terminal_decode_policy"] != DECODER
    ):
        raise ValueError("paper sampling settings changed in the evaluation configuration")
    manifest = _read_object(root / MANIFEST)
    pins = {row["repo_path"]: row["sha256"] for row in manifest["files"]}
    bundle = f"results/phase1/shared_bias_parallel_program_role_seed{replicate}_v2"
    declared = dict(config["inputs"])
    for name, filename in (
        ("checkpoint", "checkpoints.tar"),
        ("training_result", "training_result.json"),
    ):
        relative = f"{bundle}/{filename}"
        declared[name] = {"path": relative, "sha256": pins[relative]}
    design = declared["production_design"]
    if design["sha256"] != pins[design["path"]]:
        raise ValueError("production-design pins disagree")
    records: dict[str, Any] = {}
    for name, pin in sorted(declared.items()):
        path = _repo_path(root, pin["path"])
        if checkpoint is not None:
            if name == "checkpoint":
                path = checkpoint
            elif name in {"training_result", "production_design"}:
                path = checkpoint.parent / path.name
        expected = pin["sha256"]
        if not is_sha256(expected):
            raise ValueError(f"invalid expected SHA-256 for {name}")
        actual = str(sha256_file(path)) if path.is_file() else None
        records[name] = {
            "path": str(path.resolve()),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": (
                "missing" if actual is None else "verified" if actual == expected else "mismatch"
            ),
        }
    return {
        "schema_version": "forge.generation_inputs.v1",
        "ready": all(row["status"] == "verified" for row in records.values()),
        "replicate": replicate,
        "training_seed": 20260825 + replicate,
        "arm_id": ARM,
        "checkpoint_step": STEP,
        "sample_steps": FLOW_STEPS,
        "terminal_decode_policy": DECODER,
        "inputs": records,
        "configuration_sha256": str(sha256_file(config_path)),
        "artifact_manifest_sha256": str(sha256_file(root / MANIFEST)),
        "scope": "input identity only; no generation or paper reproduction performed",
    }


def source_identity(root: Path) -> dict[str, Any]:
    files = {
        path.relative_to(root).as_posix(): str(sha256_file(path))
        for directory in ("src",)
        for path in sorted((root / directory).rglob("*.py"))
    }
    return {"files": files, "sha256": str(sha256_json(files))}


def write_outputs(output: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """Publish complete outputs together, refusing an existing destination."""
    if output.exists() or output.is_symlink():
        raise ValueError(f"output already exists; choose a new directory: {output}")
    attempts = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".forge-generation-", dir=output.parent) as directory:
        staging = Path(directory) / "result"
        staging.mkdir()
        (staging / "attempts.jsonl").write_text(attempts)
        with (staging / "molecules.csv").open("w", newline="") as handle:
            fields = ("attempt", "program_id", "canonical_smiles", "valid", "exact_l1_program")
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows):
                writer.writerow({**{key: row.get(key) for key in fields}, "attempt": index})
        summary = {
            **summary,
            "output_sha256": {
                name: str(sha256_file(staging / name))
                for name in ("attempts.jsonl", "molecules.csv")
            },
        }
        (staging / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise ValueError(f"output appeared during generation: {output}")
        staging.rename(output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="only check files/hashes; no Torch or generation",
    )
    parser.add_argument(
        "--checkpoint", type=Path, help="paper checkpoints.tar; companions must be alongside it"
    )
    parser.add_argument(
        "--replicate",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="paper training replicate (default: 0)",
    )
    parser.add_argument("--family", choices=tuple(PROGRAMS), default="ugi")
    parser.add_argument(
        "--count", type=int, default=4, help="attempts, including failures (1–256; default: 4)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="demo sampling seed; not a paper evaluation seed"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2, help="Torch CPU threads")
    parser.add_argument("--output", type=Path, help="new output directory; required for generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if not 1 <= args.count <= 256 or not 1 <= args.batch_size <= 256 or args.threads < 1:
        parser.error("count and batch-size must be 1–256; threads must be positive")
    if not 0 <= args.seed < 2**63:
        parser.error("seed must be in [0, 2**63)")
    if not args.check_inputs and args.output is None:
        parser.error("--output is required for generation")
    try:
        report = check_inputs(args.root, replicate=args.replicate, checkpoint=args.checkpoint)
        if args.check_inputs:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["ready"] else 2
        if not report["ready"]:
            failures = [
                f"{name}: {row['status']} ({row['path']})"
                for name, row in report["inputs"].items()
                if row["status"] != "verified"
            ]
            raise ValueError(
                "model bundle is not ready:\n  "
                + "\n  ".join(failures)
                + "\nSee docs/ARTIFACTS.md; no generation was started."
            )
        if args.output.exists() or args.output.is_symlink():
            raise ValueError(f"output already exists: {args.output}")
        from forge.release._generation_runtime import generate

        rows, details = generate(
            report,
            program_id=PROGRAMS[args.family],
            count=args.count,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            threads=args.threads,
        )
        if len(rows) != args.count:
            raise ValueError("sampler did not retain exactly the requested number of attempts")
        summary = {
            "schema_version": "forge.reviewer_generation.v1",
            "scope": "bounded generation example",
            "full_reproduction_verified": False,
            "paper_checkpoint_end_to_end_qualified": False,
            "qualification_note": "See the versioned qualification receipt for the tested source, device and seeds.",
            "inputs": report,
            "source": source_identity(args.root),
            "python": platform.python_version(),
            "request": {
                "family": args.family,
                "program_id": PROGRAMS[args.family],
                "count": args.count,
                "seed": args.seed,
                "device": args.device,
                "batch_size": args.batch_size,
                "threads": args.threads,
            },
            "counts": {
                "attempts": len(rows),
                "valid": sum(row.get("valid") is True for row in rows),
                "exact_l1": sum(row.get("exact_l1_program") is True for row in rows),
            },
            "runtime": details,
            "limitations": "No top-up, ranking, property targeting, L2/L3 assessment or experimental validation.",
        }
        write_outputs(args.output, rows, summary)
        print(f"Saved {len(rows)} attempts to {args.output}")
        return 0
    except ModuleNotFoundError as error:
        print(
            f"forge generation: missing dependency {error.name!r}; "
            "install with uv sync --frozen --extra dev",
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        print(f"forge generation: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
