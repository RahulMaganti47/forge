"""Small infrastructure stages used to validate an experiment installation."""

from __future__ import annotations

from experiments._runtime.registry import stage
from experiments._runtime.stage import ProducedArtifact, RunContext, StageResult
from forge.core.hashing import sha256_file
from forge.core.io import write_json


@stage("experiment.inputs_snapshot.v1")
def inputs_snapshot(context: RunContext) -> StageResult:
    """Record the verified bytes visible to a stage without interpreting their science."""
    config = context.config()
    records = {
        label: {
            "bytes": path.stat().st_size,
            "path": str(path.relative_to(context.repo)),
            "sha256": str(sha256_file(path)),
        }
        for label, path in sorted(context.inputs.items())
    }
    output = context.output_path("snapshot.json")
    document = {
        "config_schema_version": config.get("schema_version"),
        "inputs": records,
        "profile": context.profile,
        "schema_version": "forge.inputs_snapshot.v1",
        "stage_id": context.stage.stage_id,
    }
    write_json(output, document)
    return StageResult(
        artifacts=(
            ProducedArtifact(
                label="snapshot",
                relative_path="snapshot.json",
                schema_version="forge.inputs_snapshot.v1",
            ),
        ),
        summary={"verified_inputs": len(records)},
    )


__all__ = ["inputs_snapshot"]
