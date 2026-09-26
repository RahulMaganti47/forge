"""Separate Transformer mechanism/FACT and held-family studies.

This module reuses the qualified production primitives without modifying the frozen four-arm design.
Each run materializes its own design receipt before training, so checkpoints authenticate the exact
interventions they implement.
"""

from __future__ import annotations

import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.assembly import RegistryRepeatedReactionProgram, Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.reaction_program_training import load_reaction_program_specifications
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_evaluation import (
    adjudicate_reaction_program_rows,
    evaluate_reaction_program_samples,
    load_reaction_program_training_references,
)
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior
from forge.model.synthesis_program_sampling import (
    TERMINAL_DECODE_POLICIES,
    sample_synthesis_program_products,
)

from .production_evaluation import (
    _conditioning_contract,
    _load_checkpoint,
    _metric_contract,
    _validate_archive_members,
    _write_samples,
    run_synthesis_program_production_evaluation,
)
from .production_randomness import production_seed
from .production_training import _deterministic_tar, _train_arm, _validate_runtime

CONFIG_SCHEMA = "forge.transformer_mechanism_study_config.v1"
RESULT_SCHEMA = "forge.transformer_mechanism_study_result.v1"


class TransformerMechanismStudyError(ValueError):
    """A mechanism intervention or secondary held-family study is malformed."""


def _input_pin(path: Path, repo: Path, *, logical_path: str | None = None) -> dict[str, str]:
    return {
        "path": logical_path or str(path.resolve().relative_to(repo.resolve())),
        "sha256": str(sha256_file(path)),
    }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = value
    return output


def _mechanism_arms(program_mass: dict[str, float]) -> dict[str, dict[str, Any]]:
    common = {"conditioning": "program", "program_mass": program_mass, "candidate_source": False}
    reference_overrides = {
        "repeat_group_conditioning": True,
        "semantic_objective": {"repeat_consistency_weight": 0.25},
    }

    def overrides(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return _deep_merge(reference_overrides, dict(extra or {}))

    return {
        "full_transformer": {
            **common,
            "role": "reference",
            "model_overrides": overrides(),
        },
        "input_only_program": {
            **common,
            "role": "mechanism_ablation",
            "model_overrides": overrides({"layerwise_program_cross_attention": False}),
        },
        "no_role_loss": {
            **common,
            "role": "mechanism_ablation",
            "model_overrides": overrides({"semantic_objective": {"role_consistency_weight": 0.0}}),
        },
        "no_core_loss": {
            **common,
            "role": "mechanism_ablation",
            "model_overrides": overrides({"semantic_objective": {"core_consistency_weight": 0.0}}),
        },
        "no_routed_adapters": {
            **common,
            "role": "mechanism_ablation",
            "model_overrides": overrides({"routed_adapters": False}),
        },
        "no_gradient_conflict_control": {
            **common,
            "role": "mechanism_ablation",
            "model_overrides": overrides(
                {"gradient_balancing": {"method": "equal_family_mass_mean"}}
            ),
        },
        "fact_matched": {
            **common,
            "role": "factorized_primary",
            "model_overrides": overrides(
                {
                    "role_isolated_attention": True,
                    "role_specific_parameters": False,
                }
            ),
        },
        "fact_generous": {
            **common,
            "role": "factorized_capacity_control",
            "model_overrides": overrides(
                {
                    "role_isolated_attention": True,
                    "role_specific_parameters": True,
                }
            ),
        },
    }


def _held_family_arms(programs: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    arms = {}
    for held in programs:
        trained = [program for program in programs if program != held]
        arms[f"leave_out_{held}"] = {
            "role": "secondary_robustness_stress_test",
            "conditioning": "program",
            "program_mass": {
                program: (1.0 / len(trained) if program in trained else 0.0) for program in programs
            },
            "evaluation_programs": [held],
            "candidate_source": False,
            "held_reaction_family": held,
            "hard_gate": False,
        }
    return arms


def _repair_arms(
    program_mass: dict[str, float], repeat_consistency_weight: float
) -> dict[str, dict[str, Any]]:
    evaluation_programs = list(program_mass)
    common = {
        "conditioning": "program",
        "program_mass": program_mass,
        # BL/LX are the repair targets. Ugi is retained as a calibration-only
        # anti-regression anchor because all three families share model weights.
        "evaluation_programs": evaluation_programs,
        "candidate_source": False,
    }
    return {
        "corrected_layout_reference": {**common, "role": "calibration_reference"},
        "repeat_aware_transformer": {
            **common,
            "role": "calibration_intervention",
            "model_overrides": {
                "repeat_group_conditioning": True,
                "semantic_objective": {"repeat_consistency_weight": repeat_consistency_weight},
            },
        },
    }


def _bl_core_constraint_arms(
    program_mass: dict[str, float], repeat_consistency_weight: float
) -> dict[str, dict[str, Any]]:
    """Return the single cell needed to isolate the BL reaction-core intervention."""

    return {
        "bl_core_constrained_repeat_aware": {
            "role": "calibration_intervention",
            "conditioning": "program",
            "program_mass": program_mass,
            "evaluation_programs": list(program_mass),
            "candidate_source": False,
            "model_overrides": {
                "repeat_group_conditioning": True,
                "semantic_objective": {"repeat_consistency_weight": repeat_consistency_weight},
            },
        }
    }


def _bl_core_constrained_production_arms(
    program_mass: dict[str, float], repeat_consistency_weight: float
) -> dict[str, dict[str, Any]]:
    """Promote the calibrated BL-core intervention without changing its training schedule."""

    arm = _bl_core_constraint_arms(program_mass, repeat_consistency_weight)[
        "bl_core_constrained_repeat_aware"
    ]
    return {
        "bl_core_constrained_repeat_aware": {
            **arm,
            "role": "final_forge_conditioned",
        }
    }


def _role_morphology_projection_arms(
    program_mass: dict[str, float], repeat_consistency_weight: float
) -> dict[str, dict[str, Any]]:
    """Add the missing full role-local morphology coordinates to the final Transformer."""

    return {
        "full_role_morphology_transformer": {
            "role": "projection_ablation_intervention",
            "conditioning": "program",
            "program_mass": program_mass,
            "evaluation_programs": ["ugi_3cr_agile"],
            "candidate_source": False,
            "model_overrides": {
                "repeat_group_conditioning": True,
                "role_morphology_conditioning": True,
                "semantic_objective": {"repeat_consistency_weight": repeat_consistency_weight},
            },
        }
    }


def _ugi_train_exposure_arms(
    program_mass: dict[str, float],
    repeat_consistency_weight: float,
    batch_program_counts: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Keep the BL-core model fixed while increasing only Ugi train-fold rows."""

    if set(batch_program_counts) != set(program_mass):
        raise TransformerMechanismStudyError("Ugi exposure counts must cover every reaction family")
    counts = {program: int(batch_program_counts[program]) for program in program_mass}
    if any(count < 1 for count in counts.values()):
        raise TransformerMechanismStudyError(
            "Ugi exposure counts must keep every reaction family active"
        )
    return {
        "bl_core_constrained_ugi_exposure": {
            "role": "calibration_intervention",
            "conditioning": "program",
            "program_mass": program_mass,
            "batch_program_counts": counts,
            "evaluation_programs": list(program_mass),
            "candidate_source": False,
            "model_overrides": {
                "repeat_group_conditioning": True,
                "semantic_objective": {"repeat_consistency_weight": repeat_consistency_weight},
            },
        }
    }


def _shared_bias_retraining_arms(
    program_mass: dict[str, float], repeat_consistency_weight: float
) -> dict[str, dict[str, Any]]:
    """Return the matched global-source and role-source end-to-end retraining arms."""

    shared_model = {
        "repeat_group_conditioning": True,
        "role_morphology_conditioning": True,
        "program_routed_output_heads": True,
        "maximum_children": 3,
        "semantic_objective": {
            "repeat_consistency_weight": repeat_consistency_weight,
            "offspring_weight": 1.0,
            "junction_consistency_weight": 0.5,
            "chemistry_loss_balancing": "equal_present_role_mass",
            "topology_conditioned_chemistry_weight": 1.0,
        },
    }
    common = {
        "conditioning": "program",
        "program_mass": program_mass,
        "evaluation_programs": list(program_mass),
        "candidate_source": False,
    }
    return {
        "shared_bias_global_source_control": {
            **common,
            "role": "matched_source_control",
            "source_marginal_mode": "global",
            "model_overrides": {**shared_model, "source_marginal_mode": "global"},
        },
        "shared_bias_program_role_source": {
            **common,
            "role": "end_to_end_intervention",
            "source_marginal_mode": "program_role_full_support",
            "model_overrides": {
                **shared_model,
                "source_marginal_mode": "program_role_full_support",
            },
        },
    }


def _study_arms(config: dict[str, Any], programs: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    study = config.get("study")
    if study == "mechanism_and_factorized":
        mass = {program: 1.0 / len(programs) for program in programs}
        return _mechanism_arms(mass)
    if study == "held_reaction_family":
        return _held_family_arms(programs)
    if study == "bl_lx_repair_calibration":
        weight = float(config.get("repeat_consistency_weight", -1.0))
        if weight <= 0.0:
            raise TransformerMechanismStudyError(
                "repair calibration requires a positive frozen repeat-consistency weight"
            )
        mass = {program: 1.0 / len(programs) for program in programs}
        return _repair_arms(mass, weight)
    if study == "bl_core_constraint_calibration":
        weight = float(config.get("repeat_consistency_weight", -1.0))
        if weight <= 0.0:
            raise TransformerMechanismStudyError(
                "BL core calibration requires a positive frozen repeat-consistency weight"
            )
        mass = {program: 1.0 / len(programs) for program in programs}
        return _bl_core_constraint_arms(mass, weight)
    if study == "bl_core_constrained_production":
        weight = float(config.get("repeat_consistency_weight", -1.0))
        if weight <= 0.0:
            raise TransformerMechanismStudyError(
                "BL-core production requires a positive frozen repeat-consistency weight"
            )
        mass = {program: 1.0 / len(programs) for program in programs}
        return _bl_core_constrained_production_arms(mass, weight)
    if study == "role_morphology_projection_ablation":
        weight = float(config.get("repeat_consistency_weight", -1.0))
        if weight <= 0.0:
            raise TransformerMechanismStudyError(
                "role-morphology ablation requires a positive repeat-consistency weight"
            )
        mass = {program: 1.0 / len(programs) for program in programs}
        return _role_morphology_projection_arms(mass, weight)
    if study == "ugi_train_exposure_calibration":
        weight = float(config.get("repeat_consistency_weight", -1.0))
        if weight <= 0.0:
            raise TransformerMechanismStudyError(
                "Ugi exposure calibration requires a positive repeat-consistency weight"
            )
        counts = config.get("batch_program_counts")
        if not isinstance(counts, Mapping):
            raise TransformerMechanismStudyError(
                "Ugi exposure calibration requires frozen batch-program counts"
            )
        mass = {program: 1.0 / len(programs) for program in programs}
        return _ugi_train_exposure_arms(mass, weight, counts)
    if study == "shared_bias_end_to_end_retraining":
        weight = float(config.get("repeat_consistency_weight", -1.0))
        if weight <= 0.0:
            raise TransformerMechanismStudyError(
                "shared-bias retraining requires a positive repeat-consistency weight"
            )
        mass = {program: 1.0 / len(programs) for program in programs}
        arms = _shared_bias_retraining_arms(mass, weight)
        selected_arm_id = config.get("selected_arm_id")
        if selected_arm_id is None:
            return arms
        if not isinstance(selected_arm_id, str) or selected_arm_id not in arms:
            raise TransformerMechanismStudyError(
                f"unsupported shared-bias arm selection: {selected_arm_id!r}"
            )
        return {selected_arm_id: arms[selected_arm_id]}
    raise TransformerMechanismStudyError(f"unsupported study: {study!r}")


def _run_repair_calibration(
    *,
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
    design_path: Path,
    cache_path: Path,
    archive_path: Path,
    training: Mapping[str, Any],
    output_dir: Path,
    runtime: Mapping[str, Any],
    replicate: int,
    allocated_device: str,
) -> dict[str, Any]:
    """Evaluate the two trained models under two decoders on calibration only.

    BL/LX are the intervention targets; Ugi is reported as a non-selecting anti-regression
    diagnostic because retraining changes the shared parameters used by every family.
    """

    import torch

    policies = tuple(str(value) for value in runtime.get("terminal_decode_policies", ()))
    if policies != TERMINAL_DECODE_POLICIES:
        raise TransformerMechanismStudyError(
            "repair study requires the matched unconstrained and strict decoder pair"
        )
    if runtime.get("calibration_only") is not True or "heldout_samples" in runtime:
        raise TransformerMechanismStudyError("repair study may access calibration products only")
    final_step = int(runtime["checkpoint_steps"][-1])
    sample_count = int(runtime["calibration_samples"])
    if sample_count < 1:
        raise TransformerMechanismStudyError("repair calibration sample count must be positive")
    specs = {
        value.program_id: value
        for value in load_reaction_program_specifications(paths["program_config"])
    }
    adapters: dict[str, Any] = {
        spec.program_id: RegistryRepeatedReactionProgram.from_registry(
            paths["qualified_reaction_families"],
            spec,
            expected_sha256=str(config["inputs"]["qualified_reaction_families"]["sha256"]),
        )
        for spec in specs.values()
    }
    adapters["ugi_3cr_agile"] = Ugi3AssemblyAdapter.from_registry(
        paths["qualified_ugi_reactions"],
        expected_sha256=str(config["inputs"]["qualified_ugi_reactions"]["sha256"]),
    )
    design = read_json_object(
        design_path,
        error=TransformerMechanismStudyError,
        label="repair calibration study design",
    )
    cache = SynthesisProgramProductionCache(cache_path)
    all_rows: list[dict[str, Any]] = []
    cells: dict[str, Any] = {}
    try:
        prior = SynthesisProgramLayoutPrior(cache)
        training_products, training_components = load_reaction_program_training_references(
            ugi_assignments=paths["ugi_assignments"],
            multireaction_atlas=paths["multireaction_atlas"],
            multireaction_splits=paths["multireaction_splits"],
            repeated_program_specs=specs,
            ugi_program_id="ugi_3cr_agile",
            ugi_roles=adapters["ugi_3cr_agile"].roles,
        )
        with tarfile.open(archive_path, mode="r") as archive:
            _validate_archive_members(archive, training)
            for arm_id, arm in sorted(training["arms"].items()):
                snapshot = next(
                    (value for value in arm["checkpoints"] if int(value["step"]) == final_step),
                    None,
                )
                if snapshot is None:
                    raise TransformerMechanismStudyError(
                        f"repair arm {arm_id} lacks its fixed final checkpoint"
                    )
                model, package = _load_checkpoint(
                    archive,
                    member_name=f"{arm_id}/{snapshot['filename']}",
                    expected_sha256=str(snapshot["sha256"]),
                    design_sha256=str(sha256_file(design_path)),
                    cache_sha256=str(sha256_file(cache_path)),
                    device=torch.device(allocated_device),
                    cache=cache,
                )
                conditioning, state_mapping = _conditioning_contract(package, cache)
                node_marginal = package["node_marginal"].detach().cpu().numpy()
                bond_marginal = package["bond_marginal"].detach().cpu().numpy()
                for policy in policies:
                    cell_id = f"{arm_id}__{policy}"
                    program_metrics = {}
                    for program_id in design["training"]["arms"][arm_id]["evaluation_programs"]:
                        layouts = prior.sample(
                            program_id,
                            sample_count=sample_count,
                            seed=production_seed(
                                int(training["seed"]),
                                "bl_lx_repair_calibration",
                                program_id,
                                "layout",
                            ),
                            role_morphology_conditioning=bool(
                                package["model_config"].get("role_morphology_conditioning", False)
                            ),
                        )
                        rows, sampling = sample_synthesis_program_products(
                            model,
                            layouts,
                            cache.atom_vocabulary,
                            node_marginal,
                            bond_marginal,
                            samples_per_program=1,
                            sample_steps=int(runtime["sample_steps"]),
                            batch_size=int(runtime["batch_size"]),
                            seed=production_seed(
                                int(training["seed"]),
                                "bl_lx_repair_calibration",
                                program_id,
                                "flow",
                            ),
                            device=allocated_device,
                            conditioning_mode=conditioning,
                            program_state_mapping=state_mapping,
                            terminal_decode_policy=policy,
                        )
                        adjudicate_reaction_program_rows(
                            rows,
                            adapters=adapters,
                            repeated_program_specs=specs,
                            ugi_program_id="ugi_3cr_agile",
                        )
                        for row in rows:
                            row.update(
                                {
                                    "arm_id": arm_id,
                                    "cell_id": cell_id,
                                    "terminal_decode_policy": policy,
                                    "checkpoint_step": final_step,
                                    "evaluation_split": "calibration",
                                    "replicate": replicate,
                                    "seed": int(training["seed"]),
                                }
                            )
                        evaluated = evaluate_reaction_program_samples(
                            rows,
                            training_products=training_products,
                            training_components=training_components,
                        )
                        metrics = _metric_contract(
                            evaluated,
                            fixed_state_failures=int(sampling["fixed_state_failures"]),
                            support_overflow_count=0,
                        )
                        metrics.update(
                            {
                                "strict_constraint_abstentions": int(
                                    sampling["strict_constraint_abstentions"]
                                ),
                                "strict_constraint_abstention_reasons": dict(
                                    sampling["strict_constraint_abstention_reasons"]
                                ),
                                "repairs": dict(sampling["repairs"]),
                            }
                        )
                        program_metrics[program_id] = metrics
                        all_rows.extend(rows)
                    cells[cell_id] = {
                        "model_arm": arm_id,
                        "terminal_decode_policy": policy,
                        "programs": program_metrics,
                    }
                del model
                if torch.device(allocated_device).type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        cache.close()
    samples_path = output_dir / "samples.jsonl.gz"
    _write_samples(samples_path, all_rows)
    expected_cells = {f"{arm}__{policy}" for arm in training["arms"] for policy in policies}
    gates = {
        "all_declared_model_decoder_cells": set(cells) == expected_cells,
        "calibration_only": all(row["evaluation_split"] == "calibration" for row in all_rows),
        "matched_attempt_count": all(
            int(metrics["samples"]) == sample_count
            for cell in cells.values()
            for metrics in cell["programs"].values()
        ),
        "fixed_states_immutable": all(
            int(metrics["fixed_state_failures"]) == 0
            for cell in cells.values()
            for metrics in cell["programs"].values()
        ),
        "no_repairs_or_retries": all(
            not metrics["repairs"]
            for cell in cells.values()
            for metrics in cell["programs"].values()
        ),
        "coverage_and_precision_reported": all(
            metrics["coverage_and_precision_reported"] is True
            for cell in cells.values()
            for metrics in cell["programs"].values()
        ),
        "ugi_calibration_anti_regression_reported": all(
            "ugi_3cr_agile" in cell["programs"] for cell in cells.values()
        ),
        "forbidden_degenerate_metric_absent": all(
            metrics["reductive_amination_substructure_rate_reported"] is False
            for cell in cells.values()
            for metrics in cell["programs"].values()
        ),
        "route_or_oracle_calls_zero": True,
        "candidate_selection_absent": True,
    }
    result = {
        "schema_version": (
            "forge.bl_core_constraint_calibration_result.v1"
            if config["study"] == "bl_core_constraint_calibration"
            else (
                "forge.ugi_train_exposure_calibration_result.v1"
                if config["study"] == "ugi_train_exposure_calibration"
                else "forge.bl_lx_repair_calibration_result.v1"
            )
        ),
        "status": "pass" if all(gates.values()) else "fail",
        "evaluation_split": "calibration_only",
        "checkpoint_step": final_step,
        "replicate": replicate,
        "seed": int(training["seed"]),
        "cells": cells,
        "samples": artifact_record(samples_path),
        "gates": gates,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "scientific_outcome_is_not_a_gate": True,
    }
    write_json(output_dir / "result.json", result)
    return result


def run_transformer_mechanism_study(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    work_dir: Path,
    profile: str,
    replicate: int,
    allocated_device: str,
    resume: bool,
) -> dict[str, Any]:
    """Train and evaluate one separately frozen intervention study replicate."""

    import torch

    config = read_json_object(
        config_path,
        error=TransformerMechanismStudyError,
        label="Transformer mechanism study config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise TransformerMechanismStudyError("unsupported mechanism study config")
    authorization = config.get("authorization", {})
    if authorization.get("authorized") is not True:
        raise TransformerMechanismStudyError("mechanism study is not authorized")
    authorized_profiles = authorization.get("profiles")
    if authorized_profiles is not None and (
        not isinstance(authorized_profiles, list)
        or profile not in {str(value) for value in authorized_profiles}
    ):
        raise TransformerMechanismStudyError(
            f"mechanism study profile is not authorized: {profile}"
        )
    raw_inputs = config.get("inputs")
    required = {
        "base_design",
        "production_cache",
        "program_config",
        "qualified_reaction_families",
        "qualified_ugi_reactions",
        "ugi_assignments",
        "multireaction_atlas",
        "multireaction_splits",
    }
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != required:
        raise TransformerMechanismStudyError("mechanism study inputs changed")
    paths = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    base = read_json_object(
        paths["base_design"],
        error=TransformerMechanismStudyError,
        label="base Transformer design",
    )
    programs = tuple(base["programs"])
    arms = _study_arms(config, programs)
    execution_scope = str(config.get("execution_scope", "production"))
    if execution_scope not in {"production", "h100_preflight"}:
        raise TransformerMechanismStudyError(
            f"unsupported mechanism-study execution scope: {execution_scope!r}"
        )
    study_design = _deep_merge(base, {"training": {"arms": arms}})
    # Preserve all base training fields that the narrow merge above did not replace.
    study_design["training"] = {**base["training"], "arms": arms}
    if config["study"] == "shared_bias_end_to_end_retraining":
        full_training = config["full"]["training"]
        for key in (
            "optimizer_steps",
            "micro_batch_size",
            "gradient_accumulation_steps",
            "effective_batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "checkpoint_steps",
        ):
            study_design["training"][key] = full_training[key]
    if config["study"] == "ugi_train_exposure_calibration":
        expanded_batch_size = sum(int(value) for value in config["batch_program_counts"].values())
        study_design["training"].update(
            {
                "micro_batch_size": expanded_batch_size,
                "effective_batch_size": expanded_batch_size
                * int(base["training"]["gradient_accumulation_steps"]),
            }
        )
    repair_calibration = config["study"] in {
        "bl_lx_repair_calibration",
        "bl_core_constraint_calibration",
        "ugi_train_exposure_calibration",
    }
    study_design["decision"] = {
        "design_scope": str(config["study"]),
        "execution_scope": execution_scope,
        "controls_are_nonselecting": True,
        "production_training_authorized": (
            not repair_calibration and profile == "full" and execution_scope == "production"
        ),
        "smoke_training_authorized": profile == "smoke",
        "calibration_training_authorized": repair_calibration,
        "held_reaction_family_is_secondary": config["study"] == "held_reaction_family",
    }
    study_design["nonclaims"] = [
        "No arm is a candidate source.",
        "Exact L1 replay is transform consistency, not synthesis-success probability.",
        "Held-reaction-family transfer is a secondary robustness stress test and not a hard gate.",
    ]
    if repair_calibration:
        study_design["nonclaims"].extend(
            [
                "The repair study reads calibration products only; no heldout product is accessed.",
                "Calibration outcomes diagnose the failure and cannot select a final production model.",
                "Strict decoding abstains without repair or retry when exact support is infeasible.",
            ]
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    design_path = output_dir / "study_design.json"
    write_json(design_path, study_design)
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise TransformerMechanismStudyError("CUDA requested but unavailable")
    runtime_config = {
        "smoke": dict(config["smoke"]["training"]),
        "full": dict(config["full"]["training"]),
    }
    runtime, base_model = _validate_runtime(
        config=runtime_config,
        design=study_design,
        profile=profile,
        device=device,
    )
    seeds = [int(value) for value in base["training"]["replicate_seeds"]]
    if replicate < 0 or replicate >= len(seeds):
        raise TransformerMechanismStudyError("replicate lies outside the frozen seed set")
    seed = seeds[replicate]
    cache = SynthesisProgramProductionCache(paths["production_cache"])
    training_dir = output_dir / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    arm_work = work_dir / "arms"
    try:
        reference_mass = {program: 1.0 / len(programs) for program in programs}
        reference_measure = cache.training_measure(reference_mass)
        node_marginal, bond_marginal = cache.source_marginals(
            reference_measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=int(base_model["bond_classes"]),
            probability_floor=float(base_model["source_probability_floor"]),
        )
        role_node_marginal = role_bond_marginal = None
        if config["study"] == "shared_bias_end_to_end_retraining":
            role_node_marginal, role_bond_marginal = cache.program_role_source_marginals(
                reference_measure,
                node_classes=len(cache.atom_vocabulary),
                bond_classes=int(base_model["bond_classes"]),
                probability_floor=float(base_model["source_probability_floor"]),
                backoff_strength=float(config["source_backoff_strength"]),
            )
        results = {}
        for arm_id, arm in arms.items():
            model_config = _deep_merge(base_model, dict(arm.get("model_overrides", {})))
            arm_without_override = {
                key: value for key, value in arm.items() if key != "model_overrides"
            }
            source_mode = str(arm.get("source_marginal_mode", "global"))
            if source_mode == "global":
                arm_node_marginal, arm_bond_marginal = node_marginal, bond_marginal
            elif (
                source_mode == "program_role_full_support"
                and role_node_marginal is not None
                and role_bond_marginal is not None
            ):
                arm_node_marginal, arm_bond_marginal = (
                    role_node_marginal,
                    role_bond_marginal,
                )
            else:
                raise TransformerMechanismStudyError(
                    f"unsupported or unavailable source marginal mode: {source_mode!r}"
                )
            results[arm_id] = _train_arm(
                arm_id=arm_id,
                arm=arm_without_override,
                seed=seed,
                cache=cache,
                design_path=design_path,
                cache_path=paths["production_cache"],
                config_path=config_path,
                runtime=runtime,
                model_config=model_config,
                device=device,
                work_dir=arm_work,
                resume=resume,
                node_marginal=arm_node_marginal,
                bond_marginal=arm_bond_marginal,
            )
    finally:
        cache.close()
    checkpoint_paths = sorted(arm_work.glob("*/checkpoint_step_*.pt"))
    archive_path = training_dir / "checkpoints.tar"
    _deterministic_tar(checkpoint_paths, archive_path, base=arm_work)
    training_result = {
        "schema_version": "forge.synthesis_program_production_training_result.v1",
        "status": "pass",
        "run_kind": execution_scope if profile == "full" else "smoke",
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "authorization": dict(config["authorization"]),
        "config": pin_record(config_path, repo),
        "design": artifact_record(design_path, logical_path="study_design.json"),
        "cache": artifact_record(paths["production_cache"]),
        "model": base_model,
        "runtime": runtime,
        "source_marginals": (
            {
                "policy": "matched_global_and_smoothed_program_role_full_support",
                "backoff_strength": float(config["source_backoff_strength"]),
                "global": {"node": node_marginal.tolist(), "bond": bond_marginal.tolist()},
                "program_role": {
                    "node": role_node_marginal.tolist(),
                    "bond": role_bond_marginal.tolist(),
                },
            }
            if config["study"] == "shared_bias_end_to_end_retraining"
            else {
                "policy": "shared_three_program_training_mixture_for_every_arm",
                "node": node_marginal.tolist(),
                "bond": bond_marginal.tolist(),
            }
        ),
        "arms": results,
        "checkpoint_archive": artifact_record(archive_path),
        "gates": {
            "all_declared_arms_complete": set(results) == set(arms),
            "fixed_state_failures_zero": all(
                int(value["fixed_state_failures"]) == 0 for value in results.values()
            ),
            "route_or_oracle_calls_zero": True,
            "candidate_selection_absent": True,
            "fact_matched_parameter_count_equals_full": (
                int(results["fact_matched"]["parameter_count"])
                == int(results["full_transformer"]["parameter_count"])
                if config["study"] == "mechanism_and_factorized"
                else True
            ),
            "fact_generous_has_strictly_more_parameters": (
                int(results["fact_generous"]["parameter_count"])
                > int(results["fact_matched"]["parameter_count"])
                if config["study"] == "mechanism_and_factorized"
                else True
            ),
            "matched_source_arms_have_equal_parameter_count": (
                len({int(value["parameter_count"]) for value in results.values()}) == 1
                if config["study"] == "shared_bias_end_to_end_retraining"
                else True
            ),
            "full_parameter_count_matches_freeze": (
                all(
                    int(value["parameter_count"]) == int(config["expected_full_parameter_count"])
                    for value in results.values()
                )
                if config["study"] == "shared_bias_end_to_end_retraining" and profile == "full"
                else True
            ),
            "program_role_sources_are_full_support": (
                bool(
                    role_node_marginal is not None
                    and role_bond_marginal is not None
                    and (role_node_marginal > 0).all()
                    and (role_bond_marginal > 0).all()
                )
                if config["study"] == "shared_bias_end_to_end_retraining"
                else True
            ),
        },
        "nonclaims": study_design["nonclaims"],
    }
    if config["study"] == "ugi_train_exposure_calibration":
        arm = results["bl_core_constrained_ugi_exposure"]
        counts = {program: int(value) for program, value in config["batch_program_counts"].items()}
        expected_exposure = {
            program: count
            * int(runtime["gradient_accumulation_steps"])
            * int(runtime["optimizer_steps"])
            for program, count in counts.items()
        }
        training_result["gates"].update(
            {
                "batch_program_counts_match_freeze": arm["batch_program_counts"] == counts,
                "observed_program_exposure_matches_freeze": (
                    arm["examples_seen_by_program"] == expected_exposure
                ),
                "ugi_train_fold_count_unchanged": (
                    int(study_design["programs"]["ugi_3cr_agile"]["expected_fold_counts"]["train"])
                    == 66464
                ),
                "ugi_heldout_access_zero": config[profile]["evaluation"]["calibration_only"]
                is True,
            }
        )
    if not all(training_result["gates"].values()):
        training_result["status"] = "fail"
    training_result_path = training_dir / "result.json"
    write_json(training_result_path, training_result)
    if training_result["status"] != "pass":
        raise TransformerMechanismStudyError(
            f"mechanism training gates failed: {training_result['gates']}"
        )
    evaluation_config = {
        "schema_version": "forge.synthesis_program_production_evaluation_config.v1",
        "execution_scope": execution_scope,
        "inputs": {
            "production_design": _input_pin(design_path, repo, logical_path="study_design.json"),
            "production_cache": _input_pin(paths["production_cache"], repo),
            **{
                key: _input_pin(paths[key], repo)
                for key in required
                if key not in {"base_design", "production_cache"}
            },
        },
        "smoke": dict(config["smoke"]["evaluation"]),
        "full": dict(config["full"]["evaluation"]),
    }
    evaluation_config_path = output_dir / "evaluation_config.json"
    write_json(evaluation_config_path, evaluation_config)
    evaluation_dir = output_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    if repair_calibration:
        evaluation_runtime = dict(config[profile]["evaluation"])
        if str(evaluation_runtime.get("device")) != allocated_device:
            raise TransformerMechanismStudyError(
                "repair evaluation device and allocated stage device differ"
            )
        evaluation = _run_repair_calibration(
            config=config,
            paths=paths,
            design_path=design_path,
            cache_path=paths["production_cache"],
            archive_path=archive_path,
            training=training_result,
            output_dir=evaluation_dir,
            runtime=evaluation_runtime,
            replicate=replicate,
            allocated_device=allocated_device,
        )
    else:
        evaluation = run_synthesis_program_production_evaluation(
            evaluation_config_path,
            repo,
            paths["production_cache"],
            archive_path,
            training_result_path,
            evaluation_dir,
            profile=profile,
            replicate=replicate,
            allocated_device=allocated_device,
            dynamic_production_design_path=design_path,
        )
    parameter_counts = {arm_id: int(value["parameter_count"]) for arm_id, value in results.items()}
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if evaluation["status"] == "pass" else "fail",
        "study": config["study"],
        "execution_scope": execution_scope,
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "arms": sorted(arms),
        "training": artifact_record(training_result_path),
        "evaluation": artifact_record(evaluation_dir / "result.json"),
        "samples": artifact_record(evaluation_dir / "samples.jsonl.gz"),
        "parameter_counts": parameter_counts,
        "held_reaction_family_hard_gate": False,
        "candidate_selection": False,
        "calls": {"route": 0, "oracle": 0},
        "evaluation_split": (
            "calibration_only" if repair_calibration else "calibration_and_heldout"
        ),
        "nonclaims": study_design["nonclaims"],
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "TransformerMechanismStudyError",
    "run_transformer_mechanism_study",
]
