"""Reassess complete method-neutral Ugi attempt ledgers, retaining failures."""

from pathlib import Path
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import sha256_file
from forge.core.io import write_json, write_jsonl
from forge.evaluation.ugi_benchmark import assess_common_ugi_attempts, load_attempt_ledger
from forge.release.generate import check_inputs


def assess(root: Path, attempts: Path, output: Path) -> dict[str, Any]:
    report = check_inputs(root)
    pins = report["inputs"]
    for key in ("qualified_ugi_reactions", "ugi_assignments"):
        if pins[key]["status"] != "verified":
            raise ValueError(f"missing or changed assessment input: {key}")
    adapter = Ugi3AssemblyAdapter.from_registry(
        Path(pins["qualified_ugi_reactions"]["path"]),
        expected_sha256=pins["qualified_ugi_reactions"]["expected_sha256"],
    )
    ledger = load_attempt_ledger(attempts)
    rows, metrics = assess_common_ugi_attempts(
        ledger, adapter=adapter, assignments_path=Path(pins["ugi_assignments"]["path"])
    )
    if len(rows) != len(ledger):
        raise ValueError("assessment changed the attempt denominator")
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "assessed_attempts.jsonl.gz", rows)
    result = {
        "schema_version": "forge.release.common_assessment.v1",
        "assessment": metrics,
        "inputs": {
            "attempts": {"path": str(attempts), "sha256": str(sha256_file(attempts))},
            **{key: pins[key] for key in ("qualified_ugi_reactions", "ugi_assignments")},
        },
        "output_sha256": str(sha256_file(output / "assessed_attempts.jsonl.gz")),
    }
    write_json(output / "result.json", result)
    return result
