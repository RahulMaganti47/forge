"""Paired inference interventions for reaction-program semantic dependence.

The trained checkpoint, Ugi morphology layouts and flow noise are held fixed.  Only the semantic
coordinates presented to the model are changed.  This distinguishes a real inference-time mismatch
from the historical cyclic-label training control, which merely renamed categorical IDs
consistently and therefore preserved their information.
"""

from __future__ import annotations

import gzip
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.assembly import RegistryRepeatedReactionProgram, Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, stable_json, write_json
from forge.corpus.reaction_program_training import load_reaction_program_specifications
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_evaluation import (
    adjudicate_reaction_program_rows,
    evaluate_reaction_program_samples,
    load_reaction_program_training_references,
)
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior
from forge.model.synthesis_program_sampling import sample_synthesis_program_products

from .production_adjudication import (
    _optional_number,
    _paired_descriptive_comparison,
    paired_seed_difference_interval,
)
from .production_evaluation import (
    _load_checkpoint,
    _metric_contract,
    _validate_archive_members,
)
from .production_randomness import production_seed

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

CONFIG_SCHEMA = "forge.program_semantic_intervention_config.v1"
CONFIG_SCHEMA_V2 = "forge.program_semantic_intervention_config.v2"
RESULT_SCHEMA = "forge.program_semantic_intervention_result.v1"
RESULT_SCHEMA_V2 = "forge.program_semantic_intervention_result.v2"
SAMPLES_SCHEMA = "forge.program_semantic_intervention_samples.v1"
TARGET_ARM = "shared_three_program_conditioned"


class ProgramSemanticInterventionError(ValueError):
    """A paired semantic intervention violates its frozen inference contract."""


def _write_samples(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            compressed.write(
                (stable_json({"schema_version": SAMPLES_SCHEMA, "rows": len(rows)}) + "\n").encode()
            )
            for row in rows:
                compressed.write((stable_json(row) + "\n").encode())


def paired_intervention_summary(
    factual: Sequence[Mapping[str, Any]],
    control: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize paired outputs without treating attempts as independent training replicates."""

    if len(factual) != len(control) or not factual:
        raise ProgramSemanticInterventionError(
            "paired intervention rows must be nonempty and equal length"
        )
    exact_counts = {
        "both_exact_l1": 0,
        "factual_only_exact_l1": 0,
        "control_only_exact_l1": 0,
        "neither_exact_l1": 0,
    }
    output_changes = 0
    validity_changes = 0
    for factual_row, control_row in zip(factual, control, strict=True):
        identity = (
            factual_row.get("sample_index"),
            factual_row.get("layout_record_id"),
            factual_row.get("program_id"),
        )
        if identity != (
            control_row.get("sample_index"),
            control_row.get("layout_record_id"),
            control_row.get("program_id"),
        ):
            raise ProgramSemanticInterventionError("paired intervention attempt identity changed")
        factual_exact = factual_row.get("exact_l1_program") is True
        control_exact = control_row.get("exact_l1_program") is True
        if factual_exact and control_exact:
            exact_counts["both_exact_l1"] += 1
        elif factual_exact:
            exact_counts["factual_only_exact_l1"] += 1
        elif control_exact:
            exact_counts["control_only_exact_l1"] += 1
        else:
            exact_counts["neither_exact_l1"] += 1
        output_changes += factual_row.get("canonical_smiles") != control_row.get("canonical_smiles")
        validity_changes += bool(factual_row.get("valid")) != bool(control_row.get("valid"))
    attempts = len(factual)
    factual_exact_count = exact_counts["both_exact_l1"] + exact_counts["factual_only_exact_l1"]
    control_exact_count = exact_counts["both_exact_l1"] + exact_counts["control_only_exact_l1"]
    return {
        "attempts": attempts,
        **exact_counts,
        "factual_exact_l1_yield": factual_exact_count / attempts,
        "control_exact_l1_yield": control_exact_count / attempts,
        "factual_minus_control_exact_l1_yield": (factual_exact_count - control_exact_count)
        / attempts,
        "canonical_output_change_fraction": output_changes / attempts,
        "validity_change_fraction": validity_changes / attempts,
        "attempts_are_not_independent_training_replicates": True,
    }


def _condition_contracts(
    *,
    target_program: str,
    mismatch_programs: Sequence[str],
    ugi_roles: Sequence[str],
    vocabulary: Any,
) -> dict[str, dict[str, Any]]:
    if len(mismatch_programs) != 2 or target_program in mismatch_programs:
        raise ProgramSemanticInterventionError(
            "the Ugi intervention requires the two distinct non-Ugi program IDs"
        )
    if len(set(mismatch_programs)) != 2 or len(ugi_roles) < 2:
        raise ProgramSemanticInterventionError("semantic mismatch controls are degenerate")
    try:
        target_state = vocabulary.program_to_index[target_program]
        mismatch_states = [vocabulary.program_to_index[value] for value in mismatch_programs]
        role_states = [vocabulary.role_to_index[value] for value in ugi_roles]
    except KeyError as error:
        raise ProgramSemanticInterventionError(
            f"semantic intervention state is absent from the frozen vocabulary: {error}"
        ) from error

    def rotate(offset: int) -> dict[int, int]:
        return {
            state: role_states[(index + offset) % len(role_states)]
            for index, state in enumerate(role_states)
        }

    return {
        "factual": {
            "conditioning_mode": "program",
            "program_state_mapping": None,
            "role_state_mapping": None,
            "intervention": "none",
        },
        f"mismatched_program__{mismatch_programs[0]}": {
            "conditioning_mode": "program_mapped",
            "program_state_mapping": {target_state: mismatch_states[0]},
            "role_state_mapping": None,
            "intervention": "program_id_only",
        },
        f"mismatched_program__{mismatch_programs[1]}": {
            "conditioning_mode": "program_mapped",
            "program_state_mapping": {target_state: mismatch_states[1]},
            "role_state_mapping": None,
            "intervention": "program_id_only",
        },
        "mismatched_roles__forward_cycle": {
            "conditioning_mode": "program_mapped",
            "program_state_mapping": None,
            "role_state_mapping": rotate(1),
            "intervention": "precursor_roles_only",
        },
        "mismatched_roles__reverse_cycle": {
            "conditioning_mode": "program_mapped",
            "program_state_mapping": None,
            "role_state_mapping": rotate(-1),
            "intervention": "precursor_roles_only",
        },
        "null_all_program_coordinates": {
            "conditioning_mode": "null",
            "program_state_mapping": None,
            "role_state_mapping": None,
            "intervention": "remove_program_role_core_and_depth",
        },
    }


def _validate_training_input(
    training: Mapping[str, Any],
    *,
    replicate: int,
    seed: int,
    design_path: Path,
    cache_path: Path,
    archive_path: Path,
    final_step: int,
    target_arm: str = TARGET_ARM,
) -> Mapping[str, Any]:
    if (
        training.get("schema_version") != "forge.synthesis_program_production_training_result.v1"
        or training.get("status") != "pass"
        or training.get("profile") != "full"
        or int(training.get("replicate", -1)) != replicate
        or int(training.get("seed", -1)) != seed
    ):
        raise ProgramSemanticInterventionError(
            f"replicate {replicate} is not a passed full production training result"
        )
    if (
        training.get("design", {}).get("sha256") != str(sha256_file(design_path))
        or training.get("cache", {}).get("sha256") != str(sha256_file(cache_path))
        or training.get("checkpoint_archive", {}).get("sha256") != str(sha256_file(archive_path))
    ):
        raise ProgramSemanticInterventionError(
            f"replicate {replicate} training inputs or checkpoint archive changed"
        )
    arm = training.get("arms", {}).get(target_arm)
    if not isinstance(arm, Mapping):
        raise ProgramSemanticInterventionError(
            f"replicate {replicate} omits the full conditioned Transformer arm"
        )
    snapshots = [row for row in arm.get("checkpoints", []) if int(row["step"]) == final_step]
    if len(snapshots) != 1:
        raise ProgramSemanticInterventionError(
            f"replicate {replicate} lacks one frozen final checkpoint at step {final_step}"
        )
    return snapshots[0]


def run_program_semantic_intervention(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    allocated_device: str,
) -> dict[str, Any]:
    """Evaluate correct and mismatched semantics under paired layouts and flow noise."""

    if torch is None:
        raise ProgramSemanticInterventionError("semantic intervention evaluation requires torch")
    config = read_json_object(
        config_path,
        error=ProgramSemanticInterventionError,
        label="program semantic intervention config",
    )
    config_schema = config.get("schema_version")
    if config_schema not in {CONFIG_SCHEMA, CONFIG_SCHEMA_V2}:
        raise ProgramSemanticInterventionError("unsupported semantic intervention config")
    target_arm = str(config.get("target_arm", TARGET_ARM))
    if not target_arm:
        raise ProgramSemanticInterventionError("semantic intervention target arm is empty")
    if config.get("authorization", {}).get("authorized") is not True:
        raise ProgramSemanticInterventionError("semantic intervention study is not authorized")
    runtime = config.get(profile)
    if not isinstance(runtime, Mapping):
        raise ProgramSemanticInterventionError(f"unsupported intervention profile: {profile}")
    if runtime.get("device") != allocated_device:
        raise ProgramSemanticInterventionError(
            "semantic intervention runtime and allocated device differ"
        )
    if runtime.get("precision") != "float32" or runtime.get("deterministic_algorithms") is not True:
        raise ProgramSemanticInterventionError(
            "semantic intervention requires deterministic float32"
        )
    raw_inputs = config.get("inputs")
    common_inputs = {
        "production_design",
        "production_cache",
        "program_config",
        "qualified_reaction_families",
        "qualified_ugi_reactions",
        "ugi_assignments",
        "multireaction_atlas",
        "multireaction_splits",
    }
    if config_schema == CONFIG_SCHEMA_V2:
        common_inputs.add("checkpoint_design")
    run_inputs = {
        f"{kind}_r{replicate}"
        for replicate in range(3)
        for kind in ("training_result", "checkpoint_archive")
    }
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != common_inputs | run_inputs:
        raise ProgramSemanticInterventionError("semantic intervention input set changed")
    paths = {key: resolve_pin(value, repo, label=key) for key, value in raw_inputs.items()}
    design = read_json_object(
        paths["production_design"],
        error=ProgramSemanticInterventionError,
        label="frozen production design",
    )
    checkpoint_design_path = paths.get("checkpoint_design", paths["production_design"])
    target_program = str(config["target_program"])
    mismatch_programs = tuple(str(value) for value in config["mismatch_programs"])
    if set(mismatch_programs) != set(design["programs"]) - {target_program}:
        raise ProgramSemanticInterventionError(
            "mismatch programs must be exactly the two frozen non-target families"
        )
    seeds = [int(value) for value in design["training"]["replicate_seeds"]]
    if len(seeds) != 3:
        raise ProgramSemanticInterventionError("semantic intervention requires three seeds")
    final_step = int(config["checkpoint_step"])
    sample_count = int(runtime["samples"])
    if sample_count < 1 or int(runtime["sample_steps"]) < 2 or int(runtime["batch_size"]) < 1:
        raise ProgramSemanticInterventionError("semantic intervention runtime is invalid")
    torch.use_deterministic_algorithms(True)
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ProgramSemanticInterventionError("CUDA requested but unavailable")

    specs = {
        spec.program_id: spec
        for spec in load_reaction_program_specifications(paths["program_config"])
    }
    repeated_adapters = {
        spec.program_id: RegistryRepeatedReactionProgram.from_registry(
            paths["qualified_reaction_families"],
            spec,
            expected_sha256=str(config["inputs"]["qualified_reaction_families"]["sha256"]),
        )
        for spec in specs.values()
    }
    ugi_adapter = Ugi3AssemblyAdapter.from_registry(
        paths["qualified_ugi_reactions"],
        expected_sha256=str(config["inputs"]["qualified_ugi_reactions"]["sha256"]),
    )
    adapters = {**repeated_adapters, target_program: ugi_adapter}
    cache = SynthesisProgramProductionCache(paths["production_cache"])
    all_rows: list[dict[str, Any]] = []
    per_replicate: dict[str, Any] = {}
    try:
        prior = SynthesisProgramLayoutPrior(cache)
        training_products, training_components = load_reaction_program_training_references(
            ugi_assignments=paths["ugi_assignments"],
            multireaction_atlas=paths["multireaction_atlas"],
            multireaction_splits=paths["multireaction_splits"],
            repeated_program_specs=specs,
            ugi_program_id=target_program,
            ugi_roles=ugi_adapter.roles,
        )
        conditions = _condition_contracts(
            target_program=target_program,
            mismatch_programs=mismatch_programs,
            ugi_roles=ugi_adapter.roles,
            vocabulary=cache.vocabulary,
        )
        for replicate, seed in enumerate(seeds):
            training_path = paths[f"training_result_r{replicate}"]
            archive_path = paths[f"checkpoint_archive_r{replicate}"]
            training = read_json_object(
                training_path,
                error=ProgramSemanticInterventionError,
                label=f"replicate {replicate} training result",
            )
            snapshot = _validate_training_input(
                training,
                replicate=replicate,
                seed=seed,
                design_path=checkpoint_design_path,
                cache_path=paths["production_cache"],
                archive_path=archive_path,
                final_step=final_step,
                target_arm=target_arm,
            )
            with tarfile.open(archive_path, mode="r") as archive:
                _validate_archive_members(archive, training)
                model, package = _load_checkpoint(
                    archive,
                    member_name=f"{target_arm}/{snapshot['filename']}",
                    expected_sha256=str(snapshot["sha256"]),
                    design_sha256=str(sha256_file(checkpoint_design_path)),
                    cache_sha256=str(sha256_file(paths["production_cache"])),
                    device=device,
                    cache=cache,
                )
            if package.get("conditioning") != "program":
                raise ProgramSemanticInterventionError(
                    "semantic intervention requires the correctly conditioned checkpoint"
                )
            layouts = prior.sample(
                target_program,
                sample_count=sample_count,
                seed=production_seed(seed, "semantic_intervention", target_program, "layout"),
            )
            sampling_seed = production_seed(seed, "semantic_intervention", target_program, "flow")
            node_marginal = package["node_marginal"].detach().cpu().numpy()
            bond_marginal = package["bond_marginal"].detach().cpu().numpy()
            condition_rows: dict[str, list[dict[str, Any]]] = {}
            condition_metrics: dict[str, Any] = {}
            for condition_id, condition in conditions.items():
                rows, sampling = sample_synthesis_program_products(
                    model,
                    layouts,
                    cache.atom_vocabulary,
                    node_marginal,
                    bond_marginal,
                    samples_per_program=1,
                    sample_steps=int(runtime["sample_steps"]),
                    batch_size=int(runtime["batch_size"]),
                    seed=sampling_seed,
                    device=allocated_device,
                    conditioning_mode=str(condition["conditioning_mode"]),
                    program_state_mapping=condition["program_state_mapping"],
                    role_state_mapping=condition["role_state_mapping"],
                )
                adjudicate_reaction_program_rows(
                    rows,
                    adapters=adapters,
                    repeated_program_specs=specs,
                    ugi_program_id=target_program,
                )
                for row in rows:
                    row.update(
                        {
                            "arm_id": target_arm,
                            "checkpoint_step": final_step,
                            "condition_id": condition_id,
                            "intervention": condition["intervention"],
                            "replicate": replicate,
                            "seed": seed,
                        }
                    )
                evaluation = evaluate_reaction_program_samples(
                    rows,
                    training_products=training_products,
                    training_components=training_components,
                )
                condition_rows[condition_id] = rows
                condition_metrics[condition_id] = _metric_contract(
                    evaluation,
                    fixed_state_failures=int(sampling["fixed_state_failures"]),
                    support_overflow_count=0,
                )
                all_rows.extend(rows)
            factual = condition_rows["factual"]
            paired = {
                condition_id: paired_intervention_summary(factual, rows)
                for condition_id, rows in condition_rows.items()
                if condition_id != "factual"
            }
            per_replicate[str(replicate)] = {
                "seed": seed,
                "training_result": pin_record(training_path, repo),
                "checkpoint_archive": artifact_record(archive_path),
                "conditions": condition_metrics,
                "paired_against_factual": paired,
            }
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        cache.close()

    confidence = config["aggregate"]
    aggregate: dict[str, Any] = {}
    metric_names: tuple[str, ...] = (
        "raw_valid_fraction",
        "exact_l1_yield_per_attempt",
        "unique_open_ended_exact_l1_products_per_1000_attempts",
    )
    if config_schema == CONFIG_SCHEMA_V2:
        metric_names = (
            "raw_valid_fraction",
            "connected_fraction",
            "exact_l1_yield_per_attempt",
            "exact_l1_decomposition_coverage",
            "exact_forward_replay_precision",
            "decomposition_abstention_fraction",
            "decomposition_ambiguity_fraction",
            "unique_open_ended_exact_l1_products_per_1000_attempts",
        )
    condition_ids = [value for value in conditions if value != "factual"]
    for condition_id in condition_ids:
        if config_schema == CONFIG_SCHEMA_V2:
            aggregate[condition_id] = {
                metric: _paired_descriptive_comparison(
                    [
                        _optional_number(
                            per_replicate[str(index)]["conditions"]["factual"].get(metric),
                            label=f"factual {metric}",
                        )
                        for index in range(3)
                    ],
                    [
                        _optional_number(
                            per_replicate[str(index)]["conditions"][condition_id].get(metric),
                            label=f"{condition_id} {metric}",
                        )
                        for index in range(3)
                    ],
                    left_label="factual",
                    right_label="intervention",
                    resamples=int(confidence["resamples"]),
                    seed=production_seed(int(confidence["seed"]), condition_id, metric),
                    confidence_level=float(confidence["confidence_level"]),
                )
                for metric in metric_names
            }
        else:
            aggregate[condition_id] = {
                metric: paired_seed_difference_interval(
                    [
                        float(per_replicate[str(index)]["conditions"]["factual"][metric])
                        for index in range(3)
                    ],
                    [
                        float(per_replicate[str(index)]["conditions"][condition_id][metric])
                        for index in range(3)
                    ],
                    resamples=int(confidence["resamples"]),
                    seed=production_seed(int(confidence["seed"]), condition_id, metric),
                    confidence_level=float(confidence["confidence_level"]),
                )
                for metric in metric_names
            }
        aggregate[condition_id]["paired_output_change_fraction_by_seed"] = [
            per_replicate[str(index)]["paired_against_factual"][condition_id][
                "canonical_output_change_fraction"
            ]
            for index in range(3)
        ]

    gates = {
        "three_frozen_training_replicates_evaluated": set(per_replicate) == {"0", "1", "2"},
        "all_declared_conditions_evaluated": all(
            set(row["conditions"]) == set(conditions) for row in per_replicate.values()
        ),
        "paired_attempt_identity_complete": all(
            int(summary["attempts"]) == sample_count
            for row in per_replicate.values()
            for summary in row["paired_against_factual"].values()
        ),
        "fixed_state_failures_zero": all(
            int(metrics["fixed_state_failures"]) == 0
            for row in per_replicate.values()
            for metrics in row["conditions"].values()
        ),
        "coverage_and_precision_reported": all(
            metrics["coverage_and_precision_reported"] is True
            for row in per_replicate.values()
            for metrics in row["conditions"].values()
        ),
        "route_or_oracle_calls_zero": True,
        "candidate_selection_absent": True,
        "scientific_effect_is_not_an_execution_gate": True,
    }
    samples_path = output_dir / "samples.jsonl.gz"
    _write_samples(samples_path, all_rows)
    result = {
        "schema_version": RESULT_SCHEMA_V2 if config_schema == CONFIG_SCHEMA_V2 else RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "profile": profile,
        "target_arm": target_arm,
        "target_program": target_program,
        "checkpoint_step": final_step,
        "paired_contract": {
            "same_checkpoint": True,
            "same_layouts": True,
            "same_flow_noise": True,
            "only_semantic_coordinates_change": True,
            "training_seed_is_the_independent_replication_unit": True,
        },
        "conditions": conditions,
        "replicates": per_replicate,
        "aggregate": aggregate,
        "sample_rows": len(all_rows),
        "samples": artifact_record(samples_path),
        "gates": gates,
        "calls": {"route": 0, "oracle": 0},
        "selection": {"candidate_selection": False},
        "interpretation": {
            "program_id_controls": (
                "Test whether the trained model changes its Ugi generation when the family token "
                "is replaced at inference while all local role/core/depth coordinates stay factual."
            ),
            "role_controls": (
                "Test whether the trained model changes its Ugi generation when registry-defined "
                "precursor roles are cyclically mismatched at inference."
            ),
            "null_control": (
                "Diagnostic out-of-distribution removal of every learned program coordinate; the "
                "separately trained null arm remains the matched training ablation."
            ),
            "no_effect_gate": (
                "Completion status authenticates the measurement only. It does not force a "
                "positive semantic-dependence result."
            ),
        },
        "nonclaims": [
            "This inference intervention does not establish experimental synthesis success.",
            "A changed output is conditional dependence, not proof of chemical correctness.",
            "The null inference condition is out of the checkpoint's training distribution.",
            "No control selects a checkpoint, molecule or candidate.",
        ],
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise ProgramSemanticInterventionError(
            f"program semantic intervention execution gates failed: {gates}"
        )
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "CONFIG_SCHEMA_V2",
    "ProgramSemanticInterventionError",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_V2",
    "SAMPLES_SCHEMA",
    "paired_intervention_summary",
    "run_program_semantic_intervention",
]
