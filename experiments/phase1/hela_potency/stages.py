"""Thin stage adapters for the consolidated potency-study data contract."""

from __future__ import annotations

from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)


@stage("potency.study-corpus.v2")
def build_potency_study_corpus_stage(context: RunContext) -> StageResult:
    """Materialize one row-preserving study corpus over the selected LNPDB records."""

    from forge.potency.study_data import build_potency_study_corpus

    config = context.config()
    require_config_inputs(context, config)
    result = build_potency_study_corpus(
        context.config_path,
        context.repo,
        ledger_path=context.output_path("observations.csv.gz"),
        result_path=context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "observations",
                "observations.csv.gz",
                "forge.potency_study_observations.v2",
                rows=int(result["artifact"]["rows"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.potency_study_corpus_result.v2",
            ),
        ),
        metrics={
            "observations": int(result["artifact"]["rows"]),
            "studies": len(result["studies"]),
            "excluded_observations": int(result["source_accounting"]["excluded_observations"]),
        },
        summary={
            "status": result["status"],
            "row_source": "lnpdb",
            "cross_study_label_pooling": False,
            "biological_guidance": False,
        },
    )


__all__ = ["build_potency_study_corpus_stage"]
