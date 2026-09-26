"""Read-only diagnosis of experiment inputs and local execution capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from experiments._runtime.environment import EnvironmentRecord
from experiments._runtime.spec import ExperimentSpec


def diagnose_experiment(repo: Path, spec_path: Path) -> dict[str, Any]:
    """Report every pin independently so one missing file does not hide the rest."""
    repo = repo.resolve()
    spec = ExperimentSpec.load(spec_path.resolve())
    stages: list[dict[str, Any]] = []
    complete = True
    for stage in spec.topological_stages():
        records: list[dict[str, Any]] = []
        for label, pin in (("config", stage.config), *sorted(stage.inputs.items())):
            try:
                path = pin.resolve(repo)
                record = {
                    "bytes": path.stat().st_size,
                    "label": label,
                    "path": pin.path,
                    "sha256": str(pin.sha256),
                    "status": "verified",
                }
            except (OSError, ValueError) as error:
                complete = False
                record = {
                    "error": str(error),
                    "label": label,
                    "path": pin.path,
                    "sha256": str(pin.sha256),
                    "status": "unavailable",
                }
            records.append(record)
        stages.append({"inputs": records, "stage_id": stage.stage_id})
    return {
        "environment": EnvironmentRecord.capture(repo).to_mapping(),
        "experiment_id": spec.experiment_id,
        "ready": complete,
        "replicates": dict(spec.replicates),
        "schema_version": "forge.experiment_diagnosis.v1",
        "spec": str(spec_path.resolve().relative_to(repo)),
        "stages": stages,
    }


__all__ = ["diagnose_experiment"]
