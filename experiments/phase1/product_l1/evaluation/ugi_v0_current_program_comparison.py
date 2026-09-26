"""Program-matched production assessment of the frozen v0 and mixed Transformer.

The two frozen model generations do not consume identical morphology tensors.  The v0 model
conditions on role-local exterior size, junction, cycle and attachment counts and generates terminal
decorations separately.  The original mixed Transformer checkpoint consumes complete component-
block sizes and a global closure count.  Every projected layout also carries the full node-aligned
coarse program so a separately trained morphology-conditioned Transformer can use the same semantic
information in a clean projection ablation.  Frozen checkpoints without those embeddings ignore it.
"""

from __future__ import annotations

import shutil
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.multireaction.production_evaluation import (
    _conditioning_contract,
    _load_checkpoint,
)
from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (
    sample_and_assess_checkpoint,
)
from experiments.phase1.product_l1.evaluation.ugi_v0_transformer_assessment import (
    assess_native_ugi_method,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_program_flow import derive_role_morphology_states
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutError,
    SynthesisProgramLayoutPrior,
)
from forge.model.synthesis_program_sampling import sample_synthesis_program_products
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.model.ugi_transformer_topology import UgiTransformerTopologyPolicy
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


CONFIG_SCHEMA = "forge.ugi_v0_current_program_comparison_config.v1"
RESULT_SCHEMA = "forge.ugi_v0_current_program_comparison.v1"
CURRENT_SAMPLING_SCHEMA = "forge.ugi_current_program_projection_sampling.v1"
PROGRAM_ID = "ugi_3cr_agile"
INPUT_LABELS = (
    "checkpoint_archive",
    "closure_checkpoint",
    "common_ugi_assessment_config",
    "current_training_result",
    "lipid_realism_config",
    "local_chemistry_config",
    "prepared_cache",
    "production_cache",
    "production_design",
    "program_draw",
    "qualified_reactions",
    "role_morphology_policy",
    "v0_checkpoint",
)


class UgiV0CurrentProgramComparisonError(ValueError):
    """The matched program comparison contract or an authenticated input changed."""


def _validate_config(config: Mapping[str, Any], *, profile: str) -> Mapping[str, Any]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise UgiV0CurrentProgramComparisonError("unsupported comparison config")
    inputs = config.get("inputs")
    profiles = config.get("profiles")
    policy = config.get("policy")
    if not isinstance(inputs, Mapping) or set(inputs) != set(INPUT_LABELS):
        raise UgiV0CurrentProgramComparisonError("comparison inputs changed")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(profile), Mapping):
        raise UgiV0CurrentProgramComparisonError(f"comparison config has no {profile!r} profile")
    required_policy = {
        "candidate_selection": False,
        "method_blind_assessment": True,
        "negative_results_reported": True,
        "oracle_calls": 0,
        "paired_program_order": True,
        "random_streams_matched": False,
        "repairs_or_retries": False,
        "route_calls": 0,
        "training_calls": 0,
    }
    if policy != required_policy:
        raise UgiV0CurrentProgramComparisonError("comparison policy changed")
    runtime = profiles[profile]
    if (
        int(runtime.get("program_count", 0)) < 1
        or int(runtime.get("sample_steps", 0)) < 1
        or int(runtime.get("batch_size", 0)) < 1
        or runtime.get("terminal_decoder_mode") != "argmax"
        or runtime.get("terminal_decoder_seed") is not None
        or runtime.get("current_terminal_decode_policy") != "strict_valence_topology_argmax"
        or runtime.get("maximum_adjacent_branch_runs") != [2, 1, 1]
    ):
        raise UgiV0CurrentProgramComparisonError("comparison runtime contract changed")
    return runtime


def _load_programs(path: Path, *, count: int) -> tuple[UgiMorphologyProgram, ...]:
    value = read_json_object(
        path,
        error=UgiV0CurrentProgramComparisonError,
        label="frozen Ugi program draw",
    )
    rows = value.get("samples")
    if not isinstance(rows, list) or len(rows) < count:
        raise UgiV0CurrentProgramComparisonError("program draw has the wrong denominator")
    programs: list[UgiMorphologyProgram] = []
    for index, row in enumerate(rows[:count]):
        raw = row.get("program") if isinstance(row, Mapping) else None
        if not isinstance(raw, Mapping):
            raise UgiV0CurrentProgramComparisonError(f"program draw row {index} is malformed")
        try:
            program = UgiMorphologyProgram(
                node_counts=tuple(int(value) for value in raw["node_counts"]),
                junction_budgets=tuple(int(value) for value in raw["junction_budgets"]),
                cycle_ranks=tuple(int(value) for value in raw["cycle_ranks"]),
                attachment_counts=tuple(int(value) for value in raw["attachment_counts"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise UgiV0CurrentProgramComparisonError(
                f"program draw row {index} has invalid fields"
            ) from error
        if any(
            len(values) != len(ROLE_NAMES)
            for values in (
                program.node_counts,
                program.junction_budgets,
                program.cycle_ranks,
                program.attachment_counts,
            )
        ):
            raise UgiV0CurrentProgramComparisonError(
                f"program draw row {index} has the wrong role count"
            )
        programs.append(program)
    return tuple(programs)


def _attachment_counts(record: Any) -> tuple[int, int, int]:
    counts: Counter[str] = Counter()
    core = np.asarray(record.core_position_states) > 1
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        parent = int(record.graph.parents[child])
        child_index = int(child)
        if bool(core[parent]) == bool(core[child_index]):
            continue
        role_state = int(record.role_states[child_index if not core[child_index] else parent])
        role = str(record.component_blocks[0].role)
        for block in record.component_blocks:
            if int(block.role_state) == role_state:
                role = str(block.role)
                break
        if role in ROLE_NAMES:
            counts[role] += 1
    return tuple(int(counts[role]) for role in ROLE_NAMES)


def project_program_for_mixed_transformer(
    prior: SynthesisProgramLayoutPrior,
    program: UgiMorphologyProgram,
    *,
    sample_index: int,
) -> Any:
    """Project one full v0 program into the mixed model's declared Ugi layout support."""

    distribution = prior._distributions.get(PROGRAM_ID)
    if distribution is None or 1 not in distribution.semantic_bundle_by_depth:
        raise UgiV0CurrentProgramComparisonError("mixed prior lacks depth-one Ugi support")
    precursor_counts = dict(zip(ROLE_NAMES, program.node_counts, strict=True))
    candidates = []
    for bundle in distribution.semantic_bundle_by_depth[1].values:
        multiplicity_pattern, fixed_signature = bundle
        blocks = []
        seen_roles: set[str] = set()
        for (role_state, core_signature), multiplicity in multiplicity_pattern:
            if int(multiplicity) != 1:
                raise UgiV0CurrentProgramComparisonError(
                    "common Ugi projection requires one component per role"
                )
            role = str(prior.vocabulary.role_states[int(role_state)])
            core_atoms = sum(int(amount) for _, amount in core_signature)
            exterior_atoms = int(precursor_counts.get(role, 0))
            blocks.append((int(role_state), core_atoms + exterior_atoms, core_signature))
            seen_roles.add(role)
        if not set(ROLE_NAMES).issubset(seen_roles):
            raise UgiV0CurrentProgramComparisonError(
                "mixed Ugi semantic bundle lacks a precursor role"
            )
        try:
            record = prior._fixed_core_layout(
                program_id=PROGRAM_ID,
                depth=1,
                closure_count=sum(program.cycle_ranks),
                blocks=blocks,
                fixed_signature=fixed_signature,
                sample_index=sample_index,
            )
        except SynthesisProgramLayoutError:
            continue
        if _attachment_counts(record) == program.attachment_counts:
            candidates.append(record)
    if len(candidates) != 1:
        raise UgiV0CurrentProgramComparisonError(
            "full Ugi program does not have one exact common-layout projection: "
            f"row={sample_index}, candidates={len(candidates)}"
        )
    record = candidates[0]
    morphology = derive_role_morphology_states(record)
    values_by_role = {
        role: (
            int(program.node_counts[index]),
            int(program.junction_budgets[index]),
            int(program.cycle_ranks[index]),
            int(program.attachment_counts[index]),
        )
        for index, role in enumerate(ROLE_NAMES)
    }
    for role, values in values_by_role.items():
        try:
            role_state = prior.vocabulary.role_states.index(role)
        except ValueError as error:
            raise UgiV0CurrentProgramComparisonError(
                f"mixed vocabulary lacks Ugi morphology role: {role}"
            ) from error
        morphology[record.role_states == role_state] = np.asarray(values, dtype=np.int64) + 1
    return replace(record, role_morphology_states=morphology)


def _current_sampling_request(
    *,
    config_sha256: str,
    checkpoint_sha256: str,
    program_draw_sha256: str,
    runtime: Mapping[str, Any],
    device: str,
    ugi_topology_policy: UgiTransformerTopologyPolicy | None = None,
) -> dict[str, Any]:
    request = {
        "batch_size": int(runtime["batch_size"]),
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "device": device,
        "flow_seed": int(runtime["flow_seed"]),
        "program_count": int(runtime["program_count"]),
        "program_draw_sha256": program_draw_sha256,
        "sample_steps": int(runtime["sample_steps"]),
        "terminal_decode_policy": str(runtime["current_terminal_decode_policy"]),
        "terminal_decoder_seed": runtime.get("terminal_decoder_seed"),
    }
    if ugi_topology_policy is not None:
        request["ugi_topology_policy"] = ugi_topology_policy.to_mapping()
    return request


def _load_or_sample_current(
    *,
    repo: Path,
    output_dir: Path,
    inputs: Mapping[str, Path],
    runtime: Mapping[str, Any],
    config_sha256: str,
    device: str,
    arm_id: str,
    checkpoint_step: int,
    programs: Sequence[UgiMorphologyProgram],
    specialist_checkpoint: Path | None = None,
    ugi_topology_policy: UgiTransformerTopologyPolicy | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    training = read_json_object(
        inputs["current_training_result"],
        error=UgiV0CurrentProgramComparisonError,
        label="mixed Transformer training result",
    )
    if (
        training.get("schema_version") != "forge.synthesis_program_production_training_result.v1"
        or training.get("status") != "pass"
        or arm_id not in training.get("arms", {})
    ):
        raise UgiV0CurrentProgramComparisonError("mixed Transformer training result changed")
    snapshots = training["arms"][arm_id].get("checkpoints")
    snapshot = next(
        (row for row in snapshots or [] if int(row.get("step", -1)) == checkpoint_step),
        None,
    )
    if not isinstance(snapshot, Mapping):
        raise UgiV0CurrentProgramComparisonError("mixed checkpoint step is missing")
    request = _current_sampling_request(
        config_sha256=config_sha256,
        checkpoint_sha256=str(snapshot["sha256"]),
        program_draw_sha256=sha256_file(inputs["program_draw"]),
        runtime=runtime,
        device=device,
        ugi_topology_policy=ugi_topology_policy,
    )
    if specialist_checkpoint is not None:
        request["specialist_checkpoint_sha256"] = sha256_file(specialist_checkpoint)
    if result_path.is_file():
        result = read_json_object(
            result_path,
            error=UgiV0CurrentProgramComparisonError,
            label="persisted mixed Transformer sampling",
        )
        rows = result.get("samples")
        if (
            result.get("schema_version") != CURRENT_SAMPLING_SCHEMA
            or result.get("status") != "complete"
            or result.get("request") != request
            or not isinstance(rows, list)
            or len(rows) != len(programs)
        ):
            raise UgiV0CurrentProgramComparisonError(
                "persisted mixed Transformer sampling request changed"
            )
        return [dict(row) for row in rows], dict(result["sampling"]), result_path

    cache = SynthesisProgramProductionCache(inputs["production_cache"])
    resolved_device = torch.device(device) if torch is not None else None
    if torch is None or resolved_device is None:
        raise UgiV0CurrentProgramComparisonError("mixed Transformer sampling requires torch")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise UgiV0CurrentProgramComparisonError("CUDA sampling was requested but unavailable")
    role_morphology_conditioning = False
    try:
        prior = SynthesisProgramLayoutPrior(cache)
        layouts = tuple(
            project_program_for_mixed_transformer(prior, program, sample_index=index)
            for index, program in enumerate(programs)
        )
        design_sha256 = sha256_file(inputs["production_design"])
        cache_sha256 = sha256_file(inputs["production_cache"])
        member = f"{arm_id}/{snapshot['filename']}"
        with tarfile.open(inputs["checkpoint_archive"], "r") as archive:
            model, package = _load_checkpoint(
                archive,
                member_name=member,
                expected_sha256=str(snapshot["sha256"]),
                design_sha256=design_sha256,
                cache_sha256=cache_sha256,
                device=resolved_device,
                cache=cache,
            )
        if specialist_checkpoint is not None:
            from experiments.phase1.multireaction.reaction_specialization import (
                load_reaction_program_specialist,
            )

            model, specialist_package = load_reaction_program_specialist(
                specialist_checkpoint,
                base_package=package,
                base_member_name=member,
                base_member_sha256=str(snapshot["sha256"]),
                cache=cache,
                cache_sha256=cache_sha256,
                device=resolved_device,
            )
            if specialist_package.get("target_program") != PROGRAM_ID:
                raise UgiV0CurrentProgramComparisonError(
                    "specialist checkpoint does not target Ugi"
                )
        conditioning, state_mapping = _conditioning_contract(package, cache)
        role_morphology_conditioning = bool(
            model.role_morphology_conditioning
        )
        node_marginal = package["node_marginal"].detach().cpu().numpy()
        bond_marginal = package["bond_marginal"].detach().cpu().numpy()
        rows, sampling = sample_synthesis_program_products(
            model,
            layouts,
            cache.atom_vocabulary,
            node_marginal,
            bond_marginal,
            samples_per_program=1,
            sample_steps=int(runtime["sample_steps"]),
            batch_size=int(runtime["batch_size"]),
            seed=int(runtime["flow_seed"]),
            device=device,
            conditioning_mode=conditioning,
            program_state_mapping=state_mapping,
            terminal_decode_policy=str(runtime["current_terminal_decode_policy"]),
            local_chemistry_support=None,
            ugi_topology_policy=ugi_topology_policy,
        )
    finally:
        cache.close()
    if len(rows) != len(programs):
        raise UgiV0CurrentProgramComparisonError("mixed sampler changed the attempt denominator")
    normalized = []
    for index, (row, program) in enumerate(zip(rows, programs, strict=True)):
        value = {
            "pipeline_index": index,
            "program": {
                "attachment_counts": list(program.attachment_counts),
                "cycle_ranks": list(program.cycle_ranks),
                "junction_budgets": list(program.junction_budgets),
                "node_counts": list(program.node_counts),
            },
            "smiles": row.get("canonical_smiles") if row.get("valid") is True else None,
            "valid": row.get("valid") is True,
        }
        normalized.append(value)
    result = {
        "schema_version": CURRENT_SAMPLING_SCHEMA,
        "status": "complete",
        "request": request,
        "samples": normalized,
        "sampling": sampling,
        "program_projection": {
            "consumed_fields": (
                [
                    "node_counts",
                    "junction_budgets",
                    "role_local_cycle_ranks",
                    "attachment_counts",
                ]
                if role_morphology_conditioning
                else ["node_counts", "sum(cycle_ranks)"]
            ),
            "ignored_fields": (
                []
                if role_morphology_conditioning
                else [
                    "junction_budgets",
                    "role_local_cycle_ranks",
                    "attachment_counts",
                ]
            ),
            "role_morphology_conditioning": role_morphology_conditioning,
            "terminal_decoration_factorization_matched": ugi_topology_policy is not None,
        },
    }
    write_json(result_path, result)
    return normalized, dict(sampling), result_path


def _load_or_assess_current(
    *,
    rows: Sequence[Mapping[str, Any]],
    sampling_path: Path,
    output_dir: Path,
    inputs: Mapping[str, Path],
    repo: Path,
    method_id: str = "forge_mixed_transformer_program_matched_seed0",
) -> dict[str, Any]:
    index_path = output_dir / "result_index.json"
    if index_path.is_file():
        index = read_json_object(
            index_path,
            error=UgiV0CurrentProgramComparisonError,
            label="persisted mixed Transformer assessment",
        )
        if index.get("sampling", {}).get("sha256") != sha256_file(sampling_path):
            raise UgiV0CurrentProgramComparisonError(
                "persisted mixed Transformer assessment input changed"
            )
        return index
    if output_dir.exists():
        shutil.rmtree(output_dir)
    assessed = assess_native_ugi_method(
        rows,
        method_id=method_id,
        seed_label=0,
        repo=repo,
        output_dir=output_dir,
        common_ugi_assessment_config=inputs["common_ugi_assessment_config"],
        lipid_realism_config=inputs["lipid_realism_config"],
        local_chemistry_config=inputs["local_chemistry_config"],
        role_morphology_policy=inputs["role_morphology_policy"],
    )
    index = {
        "schema_version": "forge.ugi_current_program_projection_assessment.v1",
        "status": "complete",
        "sampling": artifact_record(sampling_path),
        "assessment": assessed,
        "candidate_selection": False,
    }
    write_json(index_path, index)
    return index


def _metric_deltas(
    current: Mapping[str, Any], reference: Mapping[str, Any]
) -> dict[str, float | None]:
    if set(current) != set(reference):
        raise UgiV0CurrentProgramComparisonError("method assessment metric sets differ")
    deltas: dict[str, float | None] = {}
    for key in sorted(current):
        current_value = current[key]
        reference_value = reference[key]
        if current_value is None or reference_value is None:
            # Metric availability is an observed result, not part of the shared schema. Small
            # samples can leave diversity undefined for one method alone. Required decision
            # metrics are checked separately and still fail closed when unavailable.
            deltas[key] = None
        else:
            deltas[key] = float(current_value) - float(reference_value)
    return deltas


def run_ugi_v0_current_program_comparison(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    """Run both frozen checkpoints on one ordered 3,072-program Ugi draw."""

    config = read_json_object(
        config_path,
        error=UgiV0CurrentProgramComparisonError,
        label="v0 current-program comparison config",
    )
    runtime = _validate_config(config, profile=profile)
    if str(runtime.get("device")) != device:
        raise UgiV0CurrentProgramComparisonError("configured and allocated devices differ")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise UgiV0CurrentProgramComparisonError(
            f"comparison output directory is nonempty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        label: resolve_pin(config["inputs"][label], repo, label=label) for label in INPUT_LABELS
    }
    programs = _load_programs(inputs["program_draw"], count=int(runtime["program_count"]))
    current_rows, current_sampling, current_sampling_path = _load_or_sample_current(
        repo=repo,
        output_dir=output_dir / "current_sampling",
        inputs=inputs,
        runtime=runtime,
        config_sha256=sha256_file(config_path),
        device=device,
        arm_id=str(runtime["current_arm_id"]),
        checkpoint_step=int(runtime["current_checkpoint_step"]),
        programs=programs,
    )
    current_assessment = _load_or_assess_current(
        rows=current_rows,
        sampling_path=current_sampling_path,
        output_dir=output_dir / "current_assessment",
        inputs=inputs,
        repo=repo,
    )
    v0_runtime = {
        "program_count": int(runtime["program_count"]),
        "sample_steps": int(runtime["sample_steps"]),
        "batch_size": int(runtime["batch_size"]),
        "flow_seed": int(runtime["flow_seed"]),
        "terminal_decoder_mode": "argmax",
        "terminal_decoder_seed": None,
        "terminal_temperature": 1.0,
        "maximum_adjacent_branch_runs": list(runtime["maximum_adjacent_branch_runs"]),
    }
    v0_inputs = {
        label: inputs[label]
        for label in (
            "program_draw",
            "closure_checkpoint",
            "prepared_cache",
            "qualified_reactions",
            "common_ugi_assessment_config",
            "lipid_realism_config",
            "local_chemistry_config",
            "role_morphology_policy",
        )
    }
    v0 = sample_and_assess_checkpoint(
        checkpoint_path=inputs["v0_checkpoint"],
        checkpoint_step=3000,
        arm_id="v0_reference",
        method_id="forge_v0_step_3000_program_matched",
        assessment_seed_label=0,
        runtime=v0_runtime,
        inputs=v0_inputs,
        repo=repo,
        output_dir=output_dir / "v0",
        device=device,
        heldout_rows_used=True,
    )
    current_metrics = dict(current_assessment["assessment"]["metrics"])
    v0_metrics = dict(v0["assessment"]["metrics"])
    deltas = _metric_deltas(current_metrics, v0_metrics)
    primary = str(config["decision_rule"]["primary_metric"])
    margin = float(config["decision_rule"]["absolute_margin"])
    reliability = tuple(config["decision_rule"]["reliability_metrics"])
    current_reliable = all(deltas[key] >= -margin for key in reliability)
    current_primary_better = deltas[primary] > margin
    if current_reliable and current_primary_better:
        decision = "current_transformer_superior_under_frozen_gate"
    elif deltas[primary] < -margin:
        decision = "v0_reference_better_on_primary_metric"
    else:
        decision = "no_superiority_under_frozen_gate"
    v0_sampling_path = output_dir / "v0" / "sampling" / "result.json"
    v0_sampling = read_json_object(
        v0_sampling_path,
        error=UgiV0CurrentProgramComparisonError,
        label="v0 program-matched sampling result",
    )
    v0_rows = v0_sampling.get("samples")
    paired_programs = bool(
        isinstance(v0_rows, list)
        and len(v0_rows) == len(current_rows) == len(programs)
        and all(
            current_rows[index]["program"] == v0_rows[index].get("program")
            for index in range(len(programs))
        )
    )
    gates = {
        "attempt_denominator_matched": len(current_rows)
        == len(v0_rows or [])
        == int(runtime["program_count"]),
        "candidate_selection_absent": True,
        "method_blind_assessment_shared": True,
        "no_repairs_or_retries": not current_sampling.get("repairs")
        and int(v0_sampling.get("sampling", {}).get("terminal_tree_repairs", -1)) == 0,
        "paired_program_order": paired_programs,
        "route_or_oracle_calls_zero": True,
        "training_calls_zero": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if all(gates.values()) else "fail",
        "profile": profile,
        "programs_per_method": len(programs),
        "comparison_design": {
            "program_rows_and_order_matched": paired_programs,
            "sample_steps_matched": True,
            "terminal_decision_rule_matched": "model-native constrained argmax",
            "repairs_or_retries": False,
            "random_streams_matched": False,
            "training_budget_matched": False,
            "training_seed_replication": False,
            "program_projection": {
                "current_transformer": "role exterior node counts plus total cycle rank",
                "v0": (
                    "role exterior node counts, junction budgets, role-local cycle ranks and "
                    "attachment counts"
                ),
                "reason": "the frozen architectures expose different declared program tensors",
            },
        },
        "methods": {
            "forge_mixed_transformer_program_matched_seed0": {
                "metrics": current_metrics,
                "sampling": artifact_record(current_sampling_path),
                "assessment": artifact_record(
                    output_dir / "current_assessment" / "result_index.json"
                ),
            },
            "forge_v0_step_3000_program_matched": {
                "metrics": v0_metrics,
                "sampling": artifact_record(v0_sampling_path),
                "assessment": artifact_record(
                    output_dir / "v0" / "assessment" / "result_index.json"
                ),
            },
        },
        "current_minus_v0": deltas,
        "decision_rule": dict(config["decision_rule"]),
        "decision": decision,
        "gates": gates,
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "candidate_selection": False,
        "calls": {"training": 0, "route": 0, "oracle": 0},
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "complete":
        raise UgiV0CurrentProgramComparisonError(f"comparison gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "UgiV0CurrentProgramComparisonError",
    "project_program_for_mixed_transformer",
    "run_ugi_v0_current_program_comparison",
]
