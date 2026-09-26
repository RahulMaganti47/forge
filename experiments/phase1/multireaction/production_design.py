"""Freeze the matched Ugi-only versus shared-program production design.

This module validates inputs and writes a design receipt.  It does not train a model, sample a
candidate, call a route engine, or authorize a production launch.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import iter_csv, read_json_object, write_json

CONFIG_SCHEMA = "forge.synthesis_program_production_design_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_production_design_result.v1"
MIXED_DESIGN_CONFIG_SCHEMA = "forge.mixed_repeat_training_design_config.v1"
PROGRAMS = (
    "ugi_3cr_agile",
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
)
FOLDS = ("train", "calibration", "heldout")
PRIMARY_ARMS = ("ugi_only_conditioned", "shared_three_program_conditioned")
CONTROL_ARMS = (
    "shared_three_program_null",
    "shared_three_program_program_id_cyclic",
)
REQUIRED_FAMILY_METRICS = {
    "raw_valid_fraction",
    "connected_fraction",
    "exact_l1_decomposition_coverage",
    "exact_forward_replay_precision",
    "decomposition_abstention_fraction",
    "decomposition_ambiguity_fraction",
    "internal_diversity",
    "unique_fraction",
    "effective_component_count",
    "component_novelty_fraction",
    "whole_lipid_novelty_fraction",
    "fixed_state_failures",
    "support_overflow_count",
}
REQUIRED_RETENTION_METRICS = {
    "raw_valid_fraction",
    "connected_fraction",
    "exact_l1_decomposition_coverage",
    "exact_forward_replay_precision",
    "internal_diversity",
    "effective_component_count",
    "component_novelty_fraction",
    "held_component_exact_l1_decomposition_coverage",
}


class SynthesisProgramProductionDesignError(ValueError):
    """The proposed matched-production design violates a frozen scientific constraint."""


def _auxiliary_corpus_passed(
    result: Mapping[str, Any],
    *,
    paths: Mapping[str, Path],
) -> bool:
    """Accept only the source corpus or its authenticated mixed-repeat successor."""

    if result.get("schema_version") == "forge.multireaction_lnpdb_result.v1":
        programs = result.get("programs")
        return (
            result.get("status") == "pass"
            and isinstance(programs, Mapping)
            and all(
                isinstance(row, Mapping) and row.get("gate") == "pass"
                for row in programs.values()
            )
        )
    if result.get("schema_version") != "forge.multireaction_mixed_expansion_result.v1":
        return False
    artifacts = result.get("artifacts")
    summary = result.get("summary")
    if not isinstance(artifacts, Mapping) or not isinstance(summary, Mapping):
        return False
    return (
        result.get("status") == "complete_bl_lx_mixed_repeat_expansion"
        and int(summary.get("total_products", -1)) == 92_000
        and isinstance(artifacts.get("splits"), Mapping)
        and artifacts["splits"].get("sha256")
        == sha256_file(paths["multireaction_splits"])
    )


def _mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SynthesisProgramProductionDesignError(f"{label} must be an object")
    return value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SynthesisProgramProductionDesignError(f"{label} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SynthesisProgramProductionDesignError(f"{label} must be finite")
    return numeric


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SynthesisProgramProductionDesignError(f"{label} must be a positive integer")
    return value


def _validate_program_mass(
    arm_name: str,
    arm: Mapping[str, Any],
    *,
    expected: Mapping[str, float],
) -> None:
    mass = _mapping(arm.get("program_mass"), label=f"{arm_name}.program_mass")
    if set(mass) != set(PROGRAMS):
        raise SynthesisProgramProductionDesignError(
            f"{arm_name} must declare mass for exactly {list(PROGRAMS)}"
        )
    observed = {
        program: _number(mass[program], label=f"{arm_name}.program_mass.{program}")
        for program in PROGRAMS
    }
    if any(value < 0 for value in observed.values()) or not math.isclose(
        sum(observed.values()), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SynthesisProgramProductionDesignError(
            f"{arm_name} program masses must be nonnegative and sum to one"
        )
    if any(
        not math.isclose(observed[program], expected[program], rel_tol=0.0, abs_tol=1e-12)
        for program in PROGRAMS
    ):
        raise SynthesisProgramProductionDesignError(
            f"{arm_name} does not use its frozen program prior"
        )


def validate_production_design_contract(config: Mapping[str, Any]) -> dict[str, bool]:
    """Validate the policy before touching corpus bytes."""

    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SynthesisProgramProductionDesignError("unsupported production-design config")
    programs = _mapping(config.get("programs"), label="programs")
    if set(programs) != set(PROGRAMS):
        raise SynthesisProgramProductionDesignError(
            f"production design must contain exactly {list(PROGRAMS)}"
        )
    expected_sources = {
        PROGRAMS[0]: (
            "ugi_assignments",
            "product_id",
            "primary_product_fold",
            "family_balance_weight_raw",
        ),
        PROGRAMS[1]: (
            "multireaction_splits",
            "record_id",
            "product_fold",
            "source_balanced_weight",
        ),
        PROGRAMS[2]: (
            "multireaction_splits",
            "record_id",
            "product_fold",
            "source_balanced_weight",
        ),
    }
    for program_id in PROGRAMS:
        program = _mapping(programs[program_id], label=f"programs.{program_id}")
        observed_source = (
            program.get("source"),
            program.get("record_id_field"),
            program.get("fold_field"),
            program.get("weight_field"),
        )
        if observed_source != expected_sources[program_id]:
            raise SynthesisProgramProductionDesignError(
                f"{program_id} must use its frozen split and source-balanced weight fields"
            )
        counts = _mapping(
            program.get("expected_fold_counts"),
            label=f"programs.{program_id}.expected_fold_counts",
        )
        if set(counts) != set(FOLDS):
            raise SynthesisProgramProductionDesignError(
                f"{program_id} must declare train, calibration and heldout counts"
            )
        for fold in FOLDS:
            _positive_int(counts[fold], label=f"{program_id}.{fold} count")

    model = _mapping(config.get("model"), label="model")
    architecture = model.get("architecture")
    if (
        architecture
        not in {
            "reaction_program_sparse_whole_lipid_flow",
            "reaction_program_graph_transformer",
        }
        or model.get("maximum_heavy_atoms") != 194
        or model.get("maximum_closures") != 3
        or model.get("bond_classes") != 4
        or model.get("component_ids_enter_neural_tensors") is not False
        or model.get("fragment_tokens_enter_neural_tensors") is not False
    ):
        raise SynthesisProgramProductionDesignError(
            "model must retain full qualified sparse support without component identities"
        )
    if architecture == "reaction_program_graph_transformer":
        hidden_dim = _positive_int(model.get("hidden_dim"), label="model.hidden_dim")
        heads = _positive_int(model.get("attention_heads"), label="model.attention_heads")
        objective = _mapping(model.get("semantic_objective"), label="semantic_objective")
        balancing = _mapping(model.get("gradient_balancing"), label="gradient_balancing")
        if (
            hidden_dim % heads != 0
            or _positive_int(model.get("layers"), label="model.layers") < 1
            or _positive_int(model.get("expert_count"), label="model.expert_count") < 2
            or _positive_int(model.get("adapter_dim"), label="model.adapter_dim") < 1
            or set(objective)
            not in (
                {
                    "role_consistency_weight",
                    "core_consistency_weight",
                    "state_balancing",
                },
                {
                    "role_consistency_weight",
                    "core_consistency_weight",
                    "repeat_consistency_weight",
                    "state_balancing",
                },
            )
            or _number(objective["role_consistency_weight"], label="role_consistency_weight") < 0
            or _number(objective["core_consistency_weight"], label="core_consistency_weight") < 0
            or _number(
                objective.get("repeat_consistency_weight", 0.0),
                label="repeat_consistency_weight",
            )
            < 0
            or objective["state_balancing"] != "equal_present_semantic_state_mass"
            or balancing.get("method") != "equal_family_mass_deterministic_pcgrad"
            or balancing.get("group_identity") != "source_program_state"
            or balancing.get("norm_amplification") is not False
        ):
            raise SynthesisProgramProductionDesignError(
                "Transformer model must retain the qualified semantic and family-balancing contract"
            )

    training = _mapping(config.get("training"), label="training")
    seeds = training.get("replicate_seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) < 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or len(seeds) != len(set(seeds))
    ):
        raise SynthesisProgramProductionDesignError(
            "matched production comparison requires at least three distinct integer seeds"
        )
    if training.get("folds") != ["train"]:
        raise SynthesisProgramProductionDesignError(
            "calibration and heldout products cannot enter matched training"
        )
    steps = _positive_int(training.get("optimizer_steps"), label="optimizer_steps")
    micro_batch = _positive_int(training.get("micro_batch_size"), label="micro_batch_size")
    accumulation = _positive_int(
        training.get("gradient_accumulation_steps"), label="gradient_accumulation_steps"
    )
    effective_batch = _positive_int(
        training.get("effective_batch_size"), label="effective_batch_size"
    )
    if micro_batch * accumulation != effective_batch:
        raise SynthesisProgramProductionDesignError(
            "micro-batch and accumulation do not reproduce the effective batch"
        )
    checkpoints = training.get("checkpoint_steps")
    if (
        not isinstance(checkpoints, list)
        or not checkpoints
        or any(isinstance(step, bool) or not isinstance(step, int) for step in checkpoints)
        or checkpoints != sorted(set(checkpoints))
        or checkpoints[-1] != steps
    ):
        raise SynthesisProgramProductionDesignError(
            "checkpoint steps must be unique, ordered and end at the fixed final step"
        )
    if (
        training.get("checkpoint_selection") != "fixed_final_step"
        or training.get("early_stopping") is not False
        or training.get("precision") != "float32"
        or training.get("deterministic_algorithms") is not True
        or training.get("source_marginal_policy")
        != "shared_three_program_training_mixture_for_every_arm"
    ):
        raise SynthesisProgramProductionDesignError(
            "training must be deterministic, fixed-step and source-marginal matched"
        )
    arms = _mapping(training.get("arms"), label="training.arms")
    if set(arms) != set(PRIMARY_ARMS + CONTROL_ARMS):
        raise SynthesisProgramProductionDesignError(
            "production design must contain two primary and two nonselecting control arms"
        )
    ugi_mass = {program: float(program == PROGRAMS[0]) for program in PROGRAMS}
    shared_mass = {program: 1.0 / len(PROGRAMS) for program in PROGRAMS}
    _validate_program_mass(
        PRIMARY_ARMS[0], _mapping(arms[PRIMARY_ARMS[0]], label=PRIMARY_ARMS[0]), expected=ugi_mass
    )
    for arm_name in (PRIMARY_ARMS[1], *CONTROL_ARMS):
        _validate_program_mass(
            arm_name, _mapping(arms[arm_name], label=arm_name), expected=shared_mass
        )
    if any(_mapping(arms[name], label=name).get("candidate_source") is not False for name in arms):
        raise SynthesisProgramProductionDesignError(
            "a design-only arm cannot select or lock candidates"
        )
    for name in CONTROL_ARMS:
        if _mapping(arms[name], label=name).get("role") != "nonselecting_control":
            raise SynthesisProgramProductionDesignError(f"{name} must remain nonselecting")
    mapping = _mapping(
        _mapping(arms[CONTROL_ARMS[1]], label=CONTROL_ARMS[1]).get("program_id_mapping"),
        label="program_id_mapping",
    )
    if (
        set(mapping) != set(PROGRAMS)
        or any(not isinstance(value, str) for value in mapping.values())
        or set(mapping.values()) != set(PROGRAMS)
        or any(mapping[program] == program for program in PROGRAMS)
    ):
        raise SynthesisProgramProductionDesignError(
            "program-ID control must be a complete derangement, not an identity mapping"
        )

    evaluation = _mapping(config.get("evaluation"), label="evaluation")
    metrics = evaluation.get("per_family_metrics")
    if not isinstance(metrics, list) or set(metrics) != REQUIRED_FAMILY_METRICS:
        raise SynthesisProgramProductionDesignError(
            "per-family reports do not contain the complete frozen metric set"
        )
    forbidden = evaluation.get("forbidden_metrics")
    if forbidden != ["reductive_amination_substructure_hit_rate"] or any(
        metric in REQUIRED_FAMILY_METRICS for metric in forbidden
    ):
        raise SynthesisProgramProductionDesignError(
            "the degenerate reductive-amination hit rate must remain forbidden"
        )
    held_family = _mapping(evaluation.get("held_reaction_family"), label="held_reaction_family")
    if held_family.get("hard_gate") is not False:
        raise SynthesisProgramProductionDesignError(
            "held reaction family cannot become a hard pass/fail gate"
        )
    if (
        any(evaluation.get(key) != 0 for key in ("route_calls", "oracle_calls"))
        or evaluation.get("candidate_selection") is not False
    ):
        raise SynthesisProgramProductionDesignError(
            "design preflight cannot call guidance systems or select candidates"
        )
    checkpoint_policy = _mapping(evaluation.get("checkpoint_policy"), label="checkpoint_policy")
    if (
        checkpoint_policy.get("selection_fold") != "none_fixed_final_step"
        or checkpoint_policy.get("calibration_is_diagnostic_only") is not True
        or checkpoint_policy.get("heldout_selects_model_or_threshold") is not False
    ):
        raise SynthesisProgramProductionDesignError(
            "checkpoint selection must remain fixed and heldout-blind"
        )
    native = _mapping(evaluation.get("native_sampling"), label="native_sampling")
    if (
        native.get("layout_source") != "training_fold_factorized_count_only_program_prior"
        or native.get("repairs_or_retries") is not False
    ):
        raise SynthesisProgramProductionDesignError(
            "native sampling cannot copy target graphs or use repairs/retries"
        )

    retention = _mapping(config.get("ugi_retention"), label="ugi_retention")
    if (
        retention.get("baseline_arm") != PRIMARY_ARMS[0]
        or retention.get("challenger_arm") != PRIMARY_ARMS[1]
    ):
        raise SynthesisProgramProductionDesignError(
            "Ugi retention must compare the two primary matched arms"
        )
    retention_metrics = _mapping(
        retention.get("relative_noninferiority_gates"),
        label="relative_noninferiority_gates",
    )
    if set(retention_metrics) != REQUIRED_RETENTION_METRICS:
        raise SynthesisProgramProductionDesignError(
            "Ugi retention gates do not cover the frozen metric set"
        )
    for metric, raw_gate in retention_metrics.items():
        gate = _mapping(raw_gate, label=f"Ugi retention gate {metric}")
        allowed = (
            {"minimum_ratio"}
            if metric == "effective_component_count"
            else {"maximum_absolute_drop"}
        )
        if metric == "exact_forward_replay_precision":
            allowed.add("minimum_absolute_value")
        if set(gate) != allowed:
            raise SynthesisProgramProductionDesignError(
                f"Ugi retention gate {metric} has an unexpected decision rule"
            )
        for key, value in gate.items():
            numeric = _number(value, label=f"Ugi retention gate {metric}.{key}")
            if not 0 <= numeric <= 1:
                raise SynthesisProgramProductionDesignError(
                    f"Ugi retention gate {metric}.{key} must lie in [0, 1]"
                )
    rule = _mapping(retention.get("decision_rule"), label="decision_rule")
    confidence = _number(rule.get("confidence_level"), label="confidence_level")
    if (
        rule.get("method") != "paired_seed_hierarchical_bootstrap"
        or not 0 < confidence < 1
        or _positive_int(rule.get("resamples"), label="bootstrap resamples") < 1000
        or isinstance(rule.get("seed"), bool)
        or not isinstance(rule.get("seed"), int)
    ):
        raise SynthesisProgramProductionDesignError(
            "Ugi retention requires the frozen paired hierarchical bootstrap"
        )
    absolute = _mapping(retention.get("absolute_gates"), label="absolute_gates")
    if set(absolute) != {"fixed_state_failures", "support_overflow_count"} or any(
        _mapping(absolute[name], label=name).get("maximum") != 0 for name in absolute
    ):
        raise SynthesisProgramProductionDesignError(
            "fixed-state changes and support truncation must have zero tolerance"
        )

    execution = _mapping(config.get("execution"), label="execution")
    decision = _mapping(config.get("decision"), label="decision")
    if (
        execution.get("production_launch_authorized") is not False
        or decision.get("production_training_authorized") is not False
        or decision.get("controls_are_nonselecting") is not True
    ):
        raise SynthesisProgramProductionDesignError(
            "the design receipt cannot authorize a production launch"
        )
    return {
        "full_sparse_support_frozen": True,
        "training_compute_matched": True,
        "program_priors_frozen": True,
        "checkpoint_and_holdout_policy_frozen": True,
        "per_family_metrics_frozen": True,
        "ugi_retention_gates_frozen": True,
        "controls_nonselecting": True,
        "production_launch_blocked": True,
    }


def _summarize_corpora(
    config: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    programs = _mapping(config["programs"], label="programs")
    counts: Counter[tuple[str, str]] = Counter()
    weight_sums: defaultdict[tuple[str, str], float] = defaultdict(float)
    weight_minima: dict[tuple[str, str], float] = {}
    weight_maxima: dict[tuple[str, str], float] = {}
    seen: defaultdict[str, set[str]] = defaultdict(set)

    for source_label in ("ugi_assignments", "multireaction_splits"):
        source_programs = [
            program_id
            for program_id in PROGRAMS
            if _mapping(programs[program_id], label=program_id).get("source") == source_label
        ]
        for row_index, row in enumerate(iter_csv(paths[source_label]), start=2):
            if source_label == "ugi_assignments":
                program_id = PROGRAMS[0]
            else:
                program_id = row.get("program_id", "")
                if program_id not in source_programs:
                    raise SynthesisProgramProductionDesignError(
                        f"unexpected program {program_id!r} in {source_label} row {row_index}"
                    )
            contract = _mapping(programs[program_id], label=program_id)
            required = {
                str(contract["record_id_field"]),
                str(contract["fold_field"]),
                str(contract["weight_field"]),
            }
            missing = sorted(required - set(row))
            if missing:
                raise SynthesisProgramProductionDesignError(
                    f"{source_label} row {row_index} is missing fields {missing}"
                )
            record_id = row[str(contract["record_id_field"])]
            fold = row[str(contract["fold_field"])]
            if not record_id or record_id in seen[program_id]:
                raise SynthesisProgramProductionDesignError(
                    f"{program_id} has a missing or duplicate record identifier: {record_id!r}"
                )
            if fold not in FOLDS:
                raise SynthesisProgramProductionDesignError(
                    f"{program_id} has unsupported fold {fold!r}"
                )
            try:
                weight = float(row[str(contract["weight_field"])])
            except ValueError as error:
                raise SynthesisProgramProductionDesignError(
                    f"{program_id} row {record_id} has a nonnumeric sampling weight"
                ) from error
            if not math.isfinite(weight) or weight <= 0:
                raise SynthesisProgramProductionDesignError(
                    f"{program_id} row {record_id} has a nonpositive or nonfinite weight"
                )
            seen[program_id].add(record_id)
            key = (program_id, fold)
            counts[key] += 1
            weight_sums[key] += weight
            weight_minima[key] = min(weight, weight_minima.get(key, weight))
            weight_maxima[key] = max(weight, weight_maxima.get(key, weight))

    summary: dict[str, dict[str, Any]] = {}
    for program_id in PROGRAMS:
        expected = _mapping(
            _mapping(programs[program_id], label=program_id)["expected_fold_counts"],
            label=f"{program_id}.expected_fold_counts",
        )
        folds: dict[str, Any] = {}
        for fold in FOLDS:
            key = (program_id, fold)
            if counts[key] != int(expected[fold]):
                raise SynthesisProgramProductionDesignError(
                    f"{program_id} {fold} count changed: expected {expected[fold]}, "
                    f"observed {counts[key]}"
                )
            folds[fold] = {
                "records": counts[key],
                "raw_weight_sum": weight_sums[key],
                "minimum_record_weight": weight_minima[key],
                "maximum_record_weight": weight_maxima[key],
                "normalized_within_program_mass": 1.0,
            }
        summary[program_id] = {
            "records": len(seen[program_id]),
            "weight_field": _mapping(programs[program_id], label=program_id)["weight_field"],
            "folds": folds,
        }
    for fold in FOLDS:
        bl = weight_sums[(PROGRAMS[1], fold)]
        lx = weight_sums[(PROGRAMS[2], fold)]
        if not math.isclose(bl, lx, rel_tol=1e-9, abs_tol=1e-9):
            raise SynthesisProgramProductionDesignError(
                f"auxiliary source-balanced weights diverge in {fold}: BL={bl}, LX={lx}"
            )
    return summary


def freeze_shared_production_comparison_design(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate and materialize the design without executing either production arm."""

    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionDesignError,
        label="shared production-comparison design",
    )
    contract_gates = validate_production_design_contract(config)
    raw_inputs = _mapping(config.get("inputs"), label="inputs")
    paths = {label: resolve_pin(record, repo, label=label) for label, record in raw_inputs.items()}
    representation = read_json_object(
        paths["representation_result"],
        error=SynthesisProgramProductionDesignError,
        label="shared representation result",
    )
    integration = read_json_object(
        paths["integration_qualification"],
        error=SynthesisProgramProductionDesignError,
        label="shared integration qualification",
    )
    transformer: Mapping[str, Any] | None = None
    if config["model"]["architecture"] == "reaction_program_graph_transformer":
        if "transformer_qualification" not in paths:
            raise SynthesisProgramProductionDesignError(
                "Transformer production design requires its qualification receipt"
            )
        transformer = read_json_object(
            paths["transformer_qualification"],
            error=SynthesisProgramProductionDesignError,
            label="Transformer qualification",
        )
    ugi_result = read_json_object(
        paths["ugi_corpus_result"],
        error=SynthesisProgramProductionDesignError,
        label="Ugi corpus result",
    )
    multireaction_result = read_json_object(
        paths["multireaction_corpus_result"],
        error=SynthesisProgramProductionDesignError,
        label="multi-reaction corpus result",
    )
    prerequisite_gates = {
        "representation_gate_passed": representation.get("status") == "pass"
        and all(
            bool(value) for value in _mapping(representation.get("gates"), label="gates").values()
        ),
        "integration_gate_passed": integration.get("status") == "integration_gate_pass"
        and all(
            bool(value) for value in _mapping(integration.get("gates"), label="gates").values()
        ),
        "ugi_corpus_passed": ugi_result.get("status") == "pass",
        "auxiliary_corpus_passed": _auxiliary_corpus_passed(
            multireaction_result,
            paths=paths,
        ),
    }
    if transformer is not None:
        prerequisite_gates["transformer_qualification_passed"] = transformer.get(
            "status"
        ) == "qualified_for_frozen_followup_preflight" and all(
            bool(value)
            for value in _mapping(transformer.get("gates"), label="Transformer gates").values()
        )
    if not all(prerequisite_gates.values()):
        failed = sorted(name for name, passed in prerequisite_gates.items() if not passed)
        raise SynthesisProgramProductionDesignError(
            f"production design prerequisites failed: {failed}"
        )
    corpus = _summarize_corpora(config, paths)
    training = _mapping(config["training"], label="training")
    arms = _mapping(training["arms"], label="training.arms")
    examples_per_arm = int(training["optimizer_steps"]) * int(training["effective_batch_size"])
    expected_weighted_examples = {
        arm_name: {
            program: examples_per_arm * float(_mapping(arm["program_mass"], label="mass")[program])
            for program in PROGRAMS
        }
        for arm_name, arm in sorted(arms.items())
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "design_frozen_launch_blocked",
        "seed": int(config["seed"]),
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "gates": {**prerequisite_gates, **contract_gates},
        "corpus": corpus,
        "matched_compute": {
            "arms": len(arms),
            "replicates": len(training["replicate_seeds"]),
            "optimizer_steps_per_arm_per_replicate": int(training["optimizer_steps"]),
            "effective_batch_size": int(training["effective_batch_size"]),
            "examples_per_arm_per_replicate": examples_per_arm,
            "total_planned_optimizer_steps": len(arms)
            * len(training["replicate_seeds"])
            * int(training["optimizer_steps"]),
            "expected_weighted_examples_per_arm_per_replicate": expected_weighted_examples,
            "checkpoint_steps": list(training["checkpoint_steps"]),
            "checkpoint_selection": training["checkpoint_selection"],
        },
        "evaluation_contract": config["evaluation"],
        "ugi_retention_contract": config["ugi_retention"],
        "decision": config["decision"],
        "production_training_authorized": False,
        "production_sampling_authorized": False,
        "nonclaims": config["nonclaims"],
    }
    write_json(output_path, result)
    return result


def freeze_mixed_repeat_training_design(
    config_path: Path,
    repo: Path,
    design_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Derive the expanded-data design from the qualified Transformer contract.

    This keeps architecture, compute, controls, evaluation and noninferiority rules identical while
    replacing only the authenticated auxiliary corpus and its exact fold counts.  The derived full
    design remains launch-blocked and is independently validated by the standard design freezer.
    """

    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionDesignError,
        label="mixed-repeat training design",
    )
    if config.get("schema_version") != MIXED_DESIGN_CONFIG_SCHEMA:
        raise SynthesisProgramProductionDesignError(
            "unsupported mixed-repeat training design schema"
        )
    raw_inputs = _mapping(config.get("inputs"), label="inputs")
    required_inputs = {
        "base_design",
        "representation_config",
        "representation_result",
        "multireaction_splits",
        "multireaction_corpus_result",
    }
    if set(raw_inputs) != required_inputs:
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat design inputs changed"
        )
    paths = {
        label: resolve_pin(record, repo, label=label)
        for label, record in raw_inputs.items()
    }
    base = read_json_object(
        paths["base_design"],
        error=SynthesisProgramProductionDesignError,
        label="base Transformer production design",
    )
    if (
        base.get("schema_version") != CONFIG_SCHEMA
        or base.get("model", {}).get("architecture")
        != "reaction_program_graph_transformer"
    ):
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat design requires the qualified Transformer base"
        )
    representation = read_json_object(
        paths["representation_result"],
        error=SynthesisProgramProductionDesignError,
        label="mixed-repeat representation result",
    )
    representation_config_pin = pin_record(paths["representation_config"], repo)
    if (
        representation.get("status") != "pass"
        or not all(bool(value) for value in representation.get("gates", {}).values())
        or {
            "path": representation.get("config", {}).get("path"),
            "sha256": representation.get("config", {}).get("sha256"),
        }
        != {
            "path": representation_config_pin["path"],
            "sha256": representation_config_pin["sha256"],
        }
    ):
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat representation has not passed for its pinned config"
        )
    mixed_result = read_json_object(
        paths["multireaction_corpus_result"],
        error=SynthesisProgramProductionDesignError,
        label="mixed-repeat corpus result",
    )
    if not _auxiliary_corpus_passed(mixed_result, paths=paths):
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat corpus has not passed its model-support contract"
        )
    counts = _mapping(config.get("expected_fold_counts"), label="expected_fold_counts")
    if set(counts) != {PROGRAMS[1], PROGRAMS[2]}:
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat fold counts must cover BL and LX"
        )
    expected_counts = {
        program_id: {
            fold: _positive_int(
                _mapping(counts[program_id], label=program_id).get(fold),
                label=f"{program_id}.{fold}",
            )
            for fold in FOLDS
        }
        for program_id in (PROGRAMS[1], PROGRAMS[2])
    }
    if any(
        value != {"train": 30_000, "calibration": 8_000, "heldout": 8_000}
        for value in expected_counts.values()
    ):
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat design must retain the qualified equal BL/LX support"
        )
    repeat_weight = _number(
        config.get("repeat_consistency_weight"),
        label="repeat_consistency_weight",
    )
    if repeat_weight != 0.25:
        raise SynthesisProgramProductionDesignError(
            "mixed-repeat design changed the calibrated repeat-consistency weight"
        )

    design = copy.deepcopy(base)
    design["seed"] = int(config["seed"])
    for label in (
        "representation_config",
        "representation_result",
        "multireaction_splits",
        "multireaction_corpus_result",
    ):
        # Design inputs are executable pins, not result-artifact records; keep the strict
        # two-field shape required by ``resolve_pin``.
        design["inputs"][label] = dict(raw_inputs[label])
    for program_id, values in expected_counts.items():
        design["programs"][program_id]["expected_fold_counts"] = values
        design["programs"][program_id]["evidence_role"] = (
            "reaction_enumerated_structural_support"
        )
    design["model"]["semantic_objective"]["repeat_consistency_weight"] = repeat_weight
    design["execution"]["production_launch_authorized"] = False
    design["decision"]["production_training_authorized"] = False
    design["decision"]["next_authority_required"] = (
        "explicit authorization for paid exact-H100 preflight and production training"
    )
    design["derivation"] = {
        "config": pin_record(config_path, repo),
        "base_design": pin_record(paths["base_design"], repo),
        "changed_fields": [
            "representation_config_and_result",
            "multireaction_corpus_result_and_splits",
            "bl_lx_expected_fold_counts",
            "repeat_consistency_weight",
        ],
    }
    design["nonclaims"] = [
        *base["nonclaims"],
        "The expanded BL/LX products are reaction-enumerated support, not observed syntheses or route-certified products.",
    ]
    write_json(design_path, design)
    result = freeze_shared_production_comparison_design(design_path, repo, result_path)
    # The runner moves stage outputs out of its ``.partial`` directory after this function
    # returns.  Record the derived design as a logical sibling artifact so the receipt never
    # contains a dead staging path.
    result["config"] = artifact_record(design_path, logical_path="design.json")
    result["derivation"] = design["derivation"]
    write_json(result_path, result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "MIXED_DESIGN_CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SynthesisProgramProductionDesignError",
    "freeze_shared_production_comparison_design",
    "freeze_mixed_repeat_training_design",
    "validate_production_design_contract",
]
