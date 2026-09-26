"""Freeze fresh production seeds for the selected Ugi tree Transformer.

The calibration adjudicator selects an architecture and optimizer step.  This module applies that
decision to fresh training seeds without reopening the ablation ladder or exposing held-out data to
training or checkpoint selection.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

CONFIG_SCHEMA = "forge.ugi_tree_transformer_production_training.v1"
ABLATION_SCHEMA = "forge.ugi_tree_transformer_ablation_ladder.v1"
ADJUDICATION_SCHEMA = "forge.ugi_tree_transformer_calibration_adjudication.v1"
SELECTED_ARM = "tree_relations_and_routing"


class UgiTreeTransformerProductionTrainingError(ValueError):
    """The selected architecture cannot be materialized under the frozen contract."""


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise UgiTreeTransformerProductionTrainingError(f"{label} must be a mapping")
    return value


def build_production_training_config(
    design: Mapping[str, Any],
    base: Mapping[str, Any],
    ablation: Mapping[str, Any],
    adjudication: Mapping[str, Any],
    *,
    profile: str,
    replicate: int,
) -> tuple[int, dict[str, Any]]:
    """Return one seed-specific, fixed-step training config.

    The function is pure so the selection/seed contract can be tested without loading molecular
    data or starting a trainer.
    """

    required = {
        "schema_version",
        "status",
        "selected_model",
        "seed_schedule",
        "profiles",
        "inputs",
        "policy",
        "nonclaims",
    }
    if design.get("schema_version") != CONFIG_SCHEMA or set(design) != required:
        raise UgiTreeTransformerProductionTrainingError(
            "production-training design fields changed"
        )
    if design.get("status") != "frozen_after_calibration_before_production_training":
        raise UgiTreeTransformerProductionTrainingError(
            "production-training design is not frozen after calibration"
        )
    selected = _mapping(design.get("selected_model"), label="selected model")
    if selected != {
        "arm_id": SELECTED_ARM,
        "checkpoint_step": 2700,
        "candidate_id": f"{SELECTED_ARM}:step_2700",
    }:
        raise UgiTreeTransformerProductionTrainingError("selected model contract changed")

    if adjudication.get("schema_version") != ADJUDICATION_SCHEMA:
        raise UgiTreeTransformerProductionTrainingError("calibration adjudication schema changed")
    if (
        adjudication.get("status") != "complete"
        or adjudication.get("decision") != "promote_transformer"
        or adjudication.get("selected_model") != dict(selected)
        or adjudication.get("eligible_candidates") != [selected["candidate_id"]]
    ):
        raise UgiTreeTransformerProductionTrainingError(
            "calibration adjudication does not authorize the selected model"
        )

    if ablation.get("schema_version") != ABLATION_SCHEMA:
        raise UgiTreeTransformerProductionTrainingError("ablation design schema changed")
    arms = _mapping(ablation.get("arms"), label="ablation arms")
    arm = _mapping(arms.get(SELECTED_ARM), label="selected ablation arm")
    if set(arm) != {"scientific_question", "model_overrides", "objective_overrides"}:
        raise UgiTreeTransformerProductionTrainingError("selected ablation arm changed")

    schedule = _mapping(design.get("seed_schedule"), label="seed schedule")
    seeds = schedule.get(profile)
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
        or len(set(seeds)) != len(seeds)
        or replicate < 0
        or replicate >= len(seeds)
    ):
        raise UgiTreeTransformerProductionTrainingError(
            f"profile {profile!r} replicate {replicate} has no frozen seed"
        )
    seed = int(seeds[replicate])

    profiles = _mapping(design.get("profiles"), label="production profiles")
    runtime = _mapping(profiles.get(profile), label=f"{profile} production profile")
    required_runtime = {
        "device",
        "steps",
        "batch_size",
        "checkpoint_steps",
        "maximum_weighted_training_draws",
    }
    if set(runtime) != required_runtime:
        raise UgiTreeTransformerProductionTrainingError(
            f"{profile} production runtime fields changed"
        )
    steps = int(runtime["steps"])
    batch_size = int(runtime["batch_size"])
    checkpoints = [int(value) for value in runtime["checkpoint_steps"]]
    if (
        steps < 1
        or batch_size < 1
        or checkpoints != [steps]
        or int(runtime["maximum_weighted_training_draws"]) != steps * batch_size
    ):
        raise UgiTreeTransformerProductionTrainingError(
            f"{profile} fixed-duration contract is inconsistent"
        )
    if profile == "full" and (steps != 2700 or batch_size != 128):
        raise UgiTreeTransformerProductionTrainingError(
            "full production duration differs from the selected checkpoint contract"
        )
    if profile == "smoke" and steps != 2:
        raise UgiTreeTransformerProductionTrainingError("smoke must remain a two-step preflight")

    policy = design.get("policy")
    if policy != {
        "fresh_independent_training_seeds": True,
        "train_fold_only": True,
        "calibration_rows_select_nothing_after_freeze": True,
        "heldout_rows_used_for_training": False,
        "fixed_final_step": True,
        "retry_changes_seed_or_minibatch_order": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
    }:
        raise UgiTreeTransformerProductionTrainingError("production-training policy changed")

    effective = copy.deepcopy(dict(base))
    if effective.get("schema_version") != "phase1_ugi_joint_sparse_training_config.v1":
        raise UgiTreeTransformerProductionTrainingError("base training config schema changed")
    effective["task"] = "Fresh Ugi-only production training for selected tree-relational Transformer"
    effective["seed"] = seed
    effective["experiment_arm"] = {
        "arm_id": SELECTED_ARM,
        "candidate_id": selected["candidate_id"],
        "calibration_checkpoint_step": 2700,
        "production_seed": seed,
        "replicate": replicate,
        "profile": profile,
        "scientific_question": str(arm["scientific_question"]),
    }
    model = copy.deepcopy(dict(_mapping(effective.get("model"), label="base model")))
    model.update(copy.deepcopy(dict(_mapping(arm["model_overrides"], label="model overrides"))))
    effective["model"] = model
    effective["objective"] = copy.deepcopy(
        dict(_mapping(arm["objective_overrides"], label="objective overrides"))
    )
    effective_runtime = _mapping(effective.get(profile), label=f"base {profile} runtime")
    effective_runtime = copy.deepcopy(dict(effective_runtime))
    effective_runtime.update(
        {
            "device": str(runtime["device"]),
            "steps": steps,
            "batch_size": batch_size,
            "checkpoint_steps": checkpoints,
            "early_stopping": {"patience": 0, "min_delta": 0.0, "minimum_steps": 0},
        }
    )
    effective[profile] = effective_runtime
    effective["duration_contract"] = {
        "maximum_optimizer_steps": steps,
        "batch_size": batch_size,
        "maximum_weighted_training_draws": steps * batch_size,
        "reference": "calibration-selected tree-relational checkpoint step 2700",
        "fixed_checkpoint_grid": True,
        "production_checkpoint_selected_by_training_loss": False,
    }
    partition = copy.deepcopy(
        dict(_mapping(effective.get("training_partition"), label="training partition"))
    )
    if partition.get("training_folds") != ["train"]:
        raise UgiTreeTransformerProductionTrainingError("base config no longer trains on train only")
    partition.update(
        {
            "mode": "fixed_train_only",
            "selection_mode": "fixed_final_step",
            "diagnostic_use": (
                "calibration loss monitoring only; it cannot change the frozen architecture, "
                "duration or checkpoint"
            ),
        }
    )
    effective["training_partition"] = partition
    effective["promotion_contract"] = {
        "architecture_and_step_frozen_by": selected["candidate_id"],
        "fixed_final_checkpoint_step": steps,
        "fresh_seed_outputs_select_no_hyperparameter": True,
        "heldout_selects_architecture_or_checkpoint": False,
        "production_training_authorized_by_this_config": True,
    }
    effective["nonclaims"] = [
        "Fresh production seeds estimate stability; they do not reopen architecture or checkpoint selection.",
        "Exact L1 replay is transform consistency, not synthesis-success probability.",
        "No component identifier, route value or biological value enters the model.",
    ]
    return seed, effective


__all__ = [
    "CONFIG_SCHEMA",
    "SELECTED_ARM",
    "UgiTreeTransformerProductionTrainingError",
    "build_production_training_config",
]
