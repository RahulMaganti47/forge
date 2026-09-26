"""Fail-closed validation of every frozen factorized-layout evaluation draw."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutError,
    SynthesisProgramLayoutPrior,
)

from .production_evaluation import CONFIG_SCHEMA as EVALUATION_CONFIG_SCHEMA
from .production_randomness import production_seed

RESULT_SCHEMA = "forge.synthesis_program_layout_schedule_preflight.v1"


class LayoutSchedulePreflightError(ValueError):
    """The frozen evaluation layout schedule is invalid or outside declared support."""


def _resolve_evaluation_inputs(config: Mapping[str, Any], repo: Path) -> dict[str, Path]:
    required = {
        "production_design",
        "production_cache",
        "program_config",
        "qualified_reaction_families",
        "qualified_ugi_reactions",
        "ugi_assignments",
        "multireaction_atlas",
        "multireaction_splits",
    }
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != required:
        raise LayoutSchedulePreflightError("evaluation input closure changed")
    return {
        label: resolve_pin(pin, repo, label=label)
        for label, pin in sorted(inputs.items())
    }


def validate_layout_schedule(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    training_result_path: Path,
    output_path: Path,
    *,
    profile: str,
    replicate: int,
) -> dict[str, Any]:
    """Materialize every prescribed count-only layout before paid model sampling."""

    config = read_json_object(
        config_path,
        error=LayoutSchedulePreflightError,
        label="layout-schedule evaluation config",
    )
    if config.get("schema_version") != EVALUATION_CONFIG_SCHEMA:
        raise LayoutSchedulePreflightError("unsupported evaluation config")
    paths = _resolve_evaluation_inputs(config, repo)
    if paths["production_cache"].resolve() != cache_path.resolve():
        raise LayoutSchedulePreflightError("production cache path changed")
    design = read_json_object(
        paths["production_design"],
        error=LayoutSchedulePreflightError,
        label="frozen production design",
    )
    training = read_json_object(
        training_result_path,
        error=LayoutSchedulePreflightError,
        label="frozen production training result",
    )
    if (
        design.get("schema_version") != "forge.synthesis_program_production_design_config.v1"
        or training.get("schema_version")
        != "forge.synthesis_program_production_training_result.v1"
        or training.get("status") != "pass"
        or int(training.get("replicate", -1)) != replicate
    ):
        raise LayoutSchedulePreflightError("production design or training receipt changed")
    runtime = config.get(profile)
    if not isinstance(runtime, Mapping):
        raise LayoutSchedulePreflightError(f"evaluation profile is missing: {profile}")
    checkpoint_steps = [int(value) for value in runtime["checkpoint_steps"]]
    if checkpoint_steps != [int(value) for value in design["training"]["checkpoint_steps"]]:
        raise LayoutSchedulePreflightError("evaluation checkpoint schedule changed")
    seeds = [int(value) for value in design["training"]["replicate_seeds"]]
    if not 0 <= replicate < len(seeds) or int(training["seed"]) != seeds[replicate]:
        raise LayoutSchedulePreflightError("replicate seed changed")
    if set(training["arms"]) != set(design["training"]["arms"]):
        raise LayoutSchedulePreflightError("training arm set changed")

    cache = SynthesisProgramProductionCache(cache_path)
    try:
        prior = SynthesisProgramLayoutPrior(cache)
        maximum_node_count = 0
        cells = 0
        layouts = 0
        by_program: dict[str, dict[str, int]] = {}
        for arm_id, arm in design["training"]["arms"].items():
            supported = [
                str(program)
                for program, mass in arm["program_mass"].items()
                if float(mass) > 0
            ]
            evaluated = [str(value) for value in arm.get("evaluation_programs", supported)]
            if not evaluated or not set(evaluated).issubset(
                set(cache.vocabulary.program_states[1:])
            ):
                raise LayoutSchedulePreflightError("evaluation program set changed")
            snapshots = [int(value["step"]) for value in training["arms"][arm_id]["checkpoints"]]
            if snapshots != checkpoint_steps:
                raise LayoutSchedulePreflightError("authenticated checkpoint steps changed")
            for step in checkpoint_steps:
                split_counts = [("calibration", int(runtime["calibration_samples"]))]
                if step == checkpoint_steps[-1]:
                    split_counts.append(("heldout", int(runtime["heldout_samples"])))
                for split_name, count in split_counts:
                    for program_id in evaluated:
                        seed = production_seed(
                            int(training["seed"]),
                            arm_id,
                            step,
                            split_name,
                            program_id,
                            "layout",
                        )
                        try:
                            records = prior.sample(program_id, sample_count=count, seed=seed)
                        except SynthesisProgramLayoutError as error:
                            raise LayoutSchedulePreflightError(
                                "factorized layout schedule is outside declared support for "
                                f"arm={arm_id}, checkpoint={step}, split={split_name}, "
                                f"program={program_id}, seed={seed}"
                            ) from error
                        if len(records) != count:
                            raise LayoutSchedulePreflightError(
                                "factorized layout schedule returned the wrong draw count"
                            )
                        local_maximum = max(record.graph.node_count for record in records)
                        if local_maximum > prior.maximum_heavy_atoms:
                            raise LayoutSchedulePreflightError(
                                "factorized layout schedule exceeded declared atom support"
                            )
                        maximum_node_count = max(maximum_node_count, local_maximum)
                        cells += 1
                        layouts += count
                        metrics = by_program.setdefault(
                            program_id,
                            {"cells": 0, "layouts": 0, "maximum_node_count": 0},
                        )
                        metrics["cells"] += 1
                        metrics["layouts"] += count
                        metrics["maximum_node_count"] = max(
                            metrics["maximum_node_count"], local_maximum
                        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "pass",
            "profile": profile,
            "replicate": replicate,
            "seed": int(training["seed"]),
            "config": pin_record(config_path, repo),
            "design": pin_record(paths["production_design"], repo),
            "cache": pin_record(cache_path, repo),
            "training_result": pin_record(training_result_path, repo),
            "support": {
                "maximum_heavy_atoms": prior.maximum_heavy_atoms,
                "maximum_closures": prior.maximum_closures,
            },
            "schedule": {
                "cells": cells,
                "layouts": layouts,
                "maximum_node_count": maximum_node_count,
                "by_program": by_program,
            },
            "sampling_law": (
                "factorized training-fold role-size product conditioned exactly on declared "
                "maximum heavy-atom support"
            ),
            "gates": {
                "all_frozen_layout_cells_materialized": True,
                "all_layouts_within_declared_support": True,
                "clipping_absent": True,
                "repair_absent": True,
                "retry_absent": True,
                "dropped_attempts_zero": True,
                "heldout_structure_access_zero": True,
            },
        }
    finally:
        cache.close()
    write_json(output_path, result)
    return result


__all__ = [
    "LayoutSchedulePreflightError",
    "RESULT_SCHEMA",
    "validate_layout_schedule",
]
