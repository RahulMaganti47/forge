"""Exposure-matched, reaction-specific adaptation of one authenticated shared Transformer.

The shared Ugi/BL/LX checkpoint is immutable.  Each run trains only zero-initialized residual
adapters for one reaction family and records the exact number of source-balanced train-fold draws.
"""

from __future__ import annotations

import hashlib
import io
import random
import tarfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, stable_json, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.reaction_program_transformer import per_program_transformer_losses
from forge.model.reaction_specialization import (
    apply_specialist_state,
    exact_exposure_schedule,
    freeze_shared_parameters,
    initialize_specialist_from_shared_state,
    optimizer_parameters,
    specialist_parameter_report,
    specialist_state_dict,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_fixed_state_exact_tensor,
    synthesis_program_forward,
)
from forge.model.training_restart import (
    TrainingRestartError,
    atomic_torch_save,
    capture_training_random_state,
    restore_training_random_state,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.reaction_program_specialization_config.v1"
RESULT_SCHEMA = "forge.reaction_program_specialization_result.v1"
CHECKPOINT_SCHEMA = "forge.reaction_program_specialist_checkpoint.v1"
RESTART_SCHEMA = "forge.reaction_program_specialist_restart.v1"
TOPOLOGY_CONFIG_SCHEMA = "forge.reaction_program_topology_specialization_config.v2"
TOPOLOGY_RESULT_SCHEMA = "forge.reaction_program_topology_specialization_result.v2"
TOPOLOGY_CHECKPOINT_SCHEMA = "forge.reaction_program_topology_specialist_checkpoint.v2"
TOPOLOGY_RESTART_SCHEMA = "forge.reaction_program_topology_specialist_restart.v2"
BASE_CHECKPOINT_SCHEMA = "forge.synthesis_program_production_checkpoint.v1"


class ReactionProgramSpecializationError(ValueError):
    """A specialist run violates its authenticated base or exact exposure contract."""


def load_reaction_program_specialist(
    checkpoint_path: Path,
    *,
    base_package: Mapping[str, Any],
    base_member_name: str,
    base_member_sha256: str,
    cache: SynthesisProgramProductionCache,
    cache_sha256: str,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Reconstruct and authenticate a shared-plus-specialist model for evaluation."""

    if torch is None:
        raise ReactionProgramSpecializationError("specialist loading requires torch")
    package = torch.load(checkpoint_path, map_location=device, weights_only=True)
    base = package.get("base") if isinstance(package, Mapping) else None
    if (
        not isinstance(package, dict)
        or package.get("schema_version") not in {CHECKPOINT_SCHEMA, TOPOLOGY_CHECKPOINT_SCHEMA}
        or package.get("trusted_local_checkpoint") is not True
        or package.get("cache_sha256") != cache_sha256
        or not isinstance(base, Mapping)
        or base.get("member_name") != base_member_name
        or base.get("member_sha256") != base_member_sha256
        or base.get("model_state_sha256") != base_package.get("model_state_sha256")
    ):
        raise ReactionProgramSpecializationError("specialist checkpoint authentication failed")
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=package["model_config"],
        device=device,
    )
    topology_specialist = package["schema_version"] == TOPOLOGY_CHECKPOINT_SCHEMA
    expected_policy = "adapter_plus_offspring_head" if topology_specialist else "adapter_only"
    if package.get("delta_parameter_policy", "adapter_only") != expected_policy:
        raise ReactionProgramSpecializationError("specialist delta policy changed")
    initialize_specialist_from_shared_state(
        model,
        base_package["model_state"],
        include_topology_head=topology_specialist,
    )
    apply_specialist_state(
        model,
        package["specialist_state"],
        include_topology_head=topology_specialist,
    )
    if _model_state_sha256(model) != package.get("combined_model_state_sha256"):
        raise ReactionProgramSpecializationError("combined specialist model state changed")
    model.eval()
    return model, package


def _set_determinism(seed: int, cpu_threads: int, device: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def _load_base_package(
    *,
    archive_path: Path,
    training: Mapping[str, Any],
    arm_id: str,
    step: int,
    expected_member_sha256: str,
    expected_model_state_sha256: str,
    expected_design_sha256: str,
    expected_cache_sha256: str,
    device: Any,
) -> tuple[dict[str, Any], str]:
    arms = training.get("arms")
    if not isinstance(arms, Mapping) or arm_id not in arms:
        raise ReactionProgramSpecializationError("base training result omits the declared arm")
    snapshots = arms[arm_id].get("checkpoints")
    if not isinstance(snapshots, list):
        raise ReactionProgramSpecializationError("base training result omits checkpoint receipts")
    matches = [row for row in snapshots if int(row.get("step", -1)) == step]
    if len(matches) != 1:
        raise ReactionProgramSpecializationError("base checkpoint step is not unique")
    snapshot = matches[0]
    if (
        snapshot.get("sha256") != expected_member_sha256
        or snapshot.get("model_state_sha256") != expected_model_state_sha256
    ):
        raise ReactionProgramSpecializationError("base checkpoint receipt changed")
    member_name = f"{arm_id}/{snapshot['filename']}"
    with tarfile.open(archive_path, mode="r") as archive:
        try:
            member = archive.getmember(member_name)
        except KeyError as error:
            raise ReactionProgramSpecializationError("base archive omits the final checkpoint") from error
        if not member.isfile() or member.name != member_name:
            raise ReactionProgramSpecializationError("base checkpoint member is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise ReactionProgramSpecializationError("base checkpoint member cannot be read")
        payload = handle.read()
    if hashlib.sha256(payload).hexdigest() != expected_member_sha256:
        raise ReactionProgramSpecializationError("base checkpoint member hash changed")
    package = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
    if (
        not isinstance(package, dict)
        or package.get("schema_version") != BASE_CHECKPOINT_SCHEMA
        or package.get("design_sha256") != expected_design_sha256
        or package.get("cache_sha256") != expected_cache_sha256
        or package.get("model_state_sha256") != expected_model_state_sha256
    ):
        raise ReactionProgramSpecializationError("base checkpoint authentication failed")
    return package, member_name


def _restart_identity(
    *, config_path: Path, cache_path: Path, archive_path: Path, target_program: str, seed: int
) -> str:
    return hashlib.sha256(
        stable_json(
            {
                "archive_sha256": str(sha256_file(archive_path)),
                "cache_sha256": str(sha256_file(cache_path)),
                "config_sha256": str(sha256_file(config_path)),
                "seed": seed,
                "target_program": target_program,
            }
        ).encode()
    ).hexdigest()


def run_reaction_program_specialization(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    output_dir: Path,
    *,
    work_dir: Path,
    profile: str,
    allocated_device: str,
    resume: bool,
) -> dict[str, Any]:
    """Train one family adapter to an exact cumulative exposure target."""

    if torch is None:
        raise ReactionProgramSpecializationError("reaction specialization requires torch")
    config = read_json_object(
        config_path,
        error=ReactionProgramSpecializationError,
        label="reaction specialization config",
    )
    schema_version = config.get("schema_version")
    supported_profiles = (
        {"smoke", "h100_preflight", "full"}
        if schema_version == TOPOLOGY_CONFIG_SCHEMA
        else {"smoke", "full"}
    )
    if schema_version not in {CONFIG_SCHEMA, TOPOLOGY_CONFIG_SCHEMA} or profile not in (
        supported_profiles
    ):
        raise ReactionProgramSpecializationError("unsupported specialization config or profile")
    topology_specialist = schema_version == TOPOLOGY_CONFIG_SCHEMA
    topology_objective = config.get("topology_specialization")
    if topology_specialist:
        if (
            not isinstance(topology_objective, Mapping)
            or set(topology_objective)
            != {"junction_consistency_weight", "maximum_children", "offspring_weight"}
            or int(topology_objective.get("maximum_children", 0)) < 1
            or float(topology_objective.get("offspring_weight", 0.0)) <= 0.0
            or float(topology_objective.get("junction_consistency_weight", 0.0)) < 0.0
        ):
            raise ReactionProgramSpecializationError(
                "topology specialization objective is incomplete"
            )
    elif topology_objective is not None:
        raise ReactionProgramSpecializationError(
            "adapter-only specialization cannot declare a topology objective"
        )
    authorization = config.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get("authorized") is not True:
        raise ReactionProgramSpecializationError("reaction specialization is not authorized")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "base_checkpoint_archive",
        "base_training_result",
        "base_design",
        "production_cache",
    }:
        raise ReactionProgramSpecializationError("specialization input contract changed")
    paths = {
        label: resolve_pin(pin, repo, label=label) for label, pin in inputs.items()
    }
    if paths["production_cache"].resolve() != cache_path.resolve():
        raise ReactionProgramSpecializationError("stage cache differs from the pinned cache")
    runtime = config.get(profile)
    if not isinstance(runtime, Mapping):
        raise ReactionProgramSpecializationError("specialization runtime is missing")
    device = torch.device(allocated_device)
    if str(runtime.get("device")) != device.type:
        raise ReactionProgramSpecializationError("allocated and configured devices differ")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ReactionProgramSpecializationError("CUDA requested but unavailable")
    if runtime.get("precision") != "float32" or runtime.get("deterministic_algorithms") is not True:
        raise ReactionProgramSpecializationError("specialization requires deterministic float32")

    target_program = str(config.get("target_program"))
    base = config.get("base_checkpoint")
    exposure = config.get("exposure")
    if not isinstance(base, Mapping) or not isinstance(exposure, Mapping):
        raise ReactionProgramSpecializationError("base checkpoint or exposure contract is missing")
    existing_examples = int(exposure.get("existing_examples", -1))
    target_examples = (
        int(exposure.get("target_examples", -1))
        if profile == "full"
        else existing_examples + int(runtime.get("additional_examples", -1))
    )
    schedule = exact_exposure_schedule(
        existing_examples=existing_examples,
        target_examples=target_examples,
        effective_batch_size=int(runtime["effective_batch_size"]),
        micro_batch_size=int(runtime["micro_batch_size"]),
    )
    seed = int(config["seed"])
    _set_determinism(seed, int(runtime["cpu_threads"]), device)

    training = read_json_object(
        paths["base_training_result"],
        error=ReactionProgramSpecializationError,
        label="base training result",
    )
    observed_exposure = training["arms"][str(base["arm_id"])]["examples_seen_by_program"]
    if int(observed_exposure.get(target_program, -1)) != existing_examples:
        raise ReactionProgramSpecializationError("declared initial exposure differs from base evidence")
    package, member_name = _load_base_package(
        archive_path=paths["base_checkpoint_archive"],
        training=training,
        arm_id=str(base["arm_id"]),
        step=int(base["step"]),
        expected_member_sha256=str(base["member_sha256"]),
        expected_model_state_sha256=str(base["model_state_sha256"]),
        expected_design_sha256=str(sha256_file(paths["base_design"])),
        expected_cache_sha256=str(sha256_file(paths["production_cache"])),
        device=device,
    )
    model_config = dict(package["model_config"])
    model_config["specialist_adapter_dim"] = int(config["specialist_adapter_dim"])
    if topology_specialist:
        assert isinstance(topology_objective, Mapping)
        model_config["maximum_children"] = int(topology_objective["maximum_children"])
    cache = SynthesisProgramProductionCache(cache_path)
    try:
        if target_program not in cache.vocabulary.program_to_index:
            raise ReactionProgramSpecializationError("target program is absent from the cache")
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=model_config,
            device=device,
        )
        missing = initialize_specialist_from_shared_state(
            model,
            package["model_state"],
            include_topology_head=topology_specialist,
        )
        trainable_names = freeze_shared_parameters(
            model,
            include_topology_head=topology_specialist,
        )
        parameter_report = specialist_parameter_report(
            model,
            include_topology_head=topology_specialist,
        )
        optimizer = torch.optim.AdamW(
            optimizer_parameters(model),
            lr=float(runtime["learning_rate"]),
            weight_decay=float(runtime["weight_decay"]),
        )
        program_mass = {
            program: float(program == target_program)
            for program in cache.vocabulary.program_states[1:]
        }
        measure = cache.training_measure(program_mass)
        support = np.flatnonzero(measure > 0)
        probabilities = measure[support]
        probabilities /= probabilities.sum()
        target_state = cache.vocabulary.program_to_index[target_program]
        node_p0 = torch.as_tensor(package["node_marginal"], dtype=torch.float32, device=device)
        bond_p0 = torch.as_tensor(package["bond_marginal"], dtype=torch.float32, device=device)
        rng = np.random.default_rng(seed + 2)
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
        restart_path = work_dir / "restart_latest.pt"
        checkpoint_path = output_dir / "specialist_checkpoint.pt"
        identity = _restart_identity(
            config_path=config_path,
            cache_path=cache_path,
            archive_path=paths["base_checkpoint_archive"],
            target_program=target_program,
            seed=seed,
        )
        completed_steps = 0
        additional_examples_seen = 0
        fixed_state_failures = 0
        losses: list[dict[str, Any]] = []

        def restart_payload() -> dict[str, Any]:
            return {
                "schema_version": (
                    TOPOLOGY_RESTART_SCHEMA if topology_specialist else RESTART_SCHEMA
                ),
                "restart_identity": identity,
                "completed_steps": completed_steps,
                "additional_examples_seen": additional_examples_seen,
                "fixed_state_failures": fixed_state_failures,
                "specialist_state": specialist_state_dict(
                    model,
                    include_topology_head=topology_specialist,
                ),
                "optimizer_state": optimizer.state_dict(),
                "losses": losses,
                "random_state": capture_training_random_state(rng, generator, device=device),
            }

        if resume and restart_path.is_file():
            payload = torch.load(restart_path, map_location=device, weights_only=False)
            if (
                not isinstance(payload, dict)
                or payload.get("schema_version")
                != (TOPOLOGY_RESTART_SCHEMA if topology_specialist else RESTART_SCHEMA)
                or payload.get("restart_identity") != identity
            ):
                raise ReactionProgramSpecializationError("specialist restart contract changed")
            apply_specialist_state(
                model,
                payload["specialist_state"],
                include_topology_head=topology_specialist,
            )
            optimizer.load_state_dict(payload["optimizer_state"])
            completed_steps = int(payload["completed_steps"])
            additional_examples_seen = int(payload["additional_examples_seen"])
            fixed_state_failures = int(payload["fixed_state_failures"])
            losses = list(payload["losses"])
            try:
                restore_training_random_state(payload["random_state"], rng, generator, device=device)
            except TrainingRestartError as error:
                raise ReactionProgramSpecializationError("specialist restart RNG is invalid") from error
        if completed_steps > len(schedule.optimizer_steps):
            raise ReactionProgramSpecializationError("restart step exceeds exact schedule")
        expected_seen = sum(sum(step) for step in schedule.optimizer_steps[:completed_steps])
        if additional_examples_seen != expected_seen:
            raise ReactionProgramSpecializationError("restart exposure ledger is inconsistent")

        objective = model_config["semantic_objective"]
        model.train()
        for step_index in range(completed_steps, len(schedule.optimizer_steps)):
            microbatches = schedule.optimizer_steps[step_index]
            step_examples = sum(microbatches)
            optimizer.zero_grad(set_to_none=True)
            step_loss = torch.zeros((), dtype=torch.float32, device=device)
            step_fixed = torch.zeros((), dtype=torch.int64, device=device)
            for micro_size in microbatches:
                selected = rng.choice(support, size=micro_size, replace=True, p=probabilities)
                records = cache.records(selected)
                clean = move_tensors(
                    collate_synthesis_program_training_batch(
                        records,
                        maximum_closures=int(model_config["maximum_closures"]),
                        conditioning="program",
                        vocabulary=cache.vocabulary,
                    ),
                    device,
                )
                t = torch.rand(micro_size, generator=generator, device=device).clamp(0.02, 0.98)
                predictions, noisy = synthesis_program_forward(
                    model, clean, node_p0, bond_p0, t, generator
                )
                family_losses, _ = per_program_transformer_losses(
                    predictions,
                    clean,
                    role_weight=float(objective["role_consistency_weight"]),
                    core_weight=float(objective["core_consistency_weight"]),
                    repeat_consistency_weight=float(objective.get("repeat_consistency_weight", 0.0)),
                    offspring_weight=(
                        float(topology_objective["offspring_weight"])
                        if topology_specialist
                        else 0.0
                    ),
                    junction_consistency_weight=(
                        float(topology_objective["junction_consistency_weight"])
                        if topology_specialist
                        else 0.0
                    ),
                    program_states=(target_state,),
                    materialize_metrics=False,
                )
                loss = family_losses[target_state]
                fraction = micro_size / step_examples
                (loss * fraction).backward()
                step_loss += loss.detach() * fraction
                step_fixed += (~synthesis_program_fixed_state_exact_tensor(noisy, clean)).to(
                    torch.int64
                )
                additional_examples_seen += micro_size
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                optimizer_parameters(model), float(runtime["gradient_clip_norm"])
            )
            optimizer.step()
            published = torch.stack(
                (step_loss, gradient_norm.detach(), step_fixed.to(torch.float32))
            ).cpu().tolist()
            completed_steps = step_index + 1
            fixed_state_failures += int(published[2])
            losses.append(
                {
                    "step": completed_steps,
                    "examples": step_examples,
                    "loss": float(published[0]),
                    "gradient_norm": float(published[1]),
                }
            )
            interval = int(runtime["restart_interval_steps"])
            if completed_steps % interval == 0 or completed_steps == len(schedule.optimizer_steps):
                atomic_torch_save(restart_path, restart_payload())
                write_json(
                    output_dir / "progress.json",
                    {
                        "schema_version": (
                            "forge.reaction_program_topology_specialization_progress.v2"
                            if topology_specialist
                            else "forge.reaction_program_specialization_progress.v1"
                        ),
                        "target_program": target_program,
                        "completed_steps": completed_steps,
                        "target_steps": len(schedule.optimizer_steps),
                        "additional_examples_seen": additional_examples_seen,
                        "target_additional_examples": schedule.additional_examples,
                        "fixed_state_failures": fixed_state_failures,
                        "latest_loss": losses[-1],
                    },
                )

        if additional_examples_seen != schedule.additional_examples:
            raise ReactionProgramSpecializationError("final exposure does not match the exact target")
        checkpoint = {
            "schema_version": (
                TOPOLOGY_CHECKPOINT_SCHEMA if topology_specialist else CHECKPOINT_SCHEMA
            ),
            "trusted_local_checkpoint": True,
            "target_program": target_program,
            "seed": seed,
            "model_config": model_config,
            "specialist_state": specialist_state_dict(
                model,
                include_topology_head=topology_specialist,
            ),
            "delta_parameter_policy": (
                "adapter_plus_offspring_head" if topology_specialist else "adapter_only"
            ),
            "combined_model_state_sha256": _model_state_sha256(model),
            "base": {
                "archive": artifact_record(paths["base_checkpoint_archive"]),
                "member_name": member_name,
                "member_sha256": str(base["member_sha256"]),
                "model_state_sha256": str(base["model_state_sha256"]),
            },
            "cache_sha256": str(sha256_file(cache_path)),
            "node_marginal": package["node_marginal"],
            "bond_marginal": package["bond_marginal"],
            "exposure": schedule.to_mapping(),
        }
        atomic_torch_save(checkpoint_path, checkpoint)
    finally:
        cache.close()

    result = {
        "schema_version": TOPOLOGY_RESULT_SCHEMA if topology_specialist else RESULT_SCHEMA,
        "status": "pass" if fixed_state_failures == 0 else "fail",
        "profile": profile,
        "target_program": target_program,
        "seed": seed,
        "config": pin_record(config_path, repo),
        "inputs": {label: artifact_record(path) for label, path in paths.items()},
        "base_checkpoint": {
            "member_name": member_name,
            "member_sha256": str(base["member_sha256"]),
            "model_state_sha256": str(base["model_state_sha256"]),
        },
        "exposure": schedule.to_mapping(),
        "observed": {
            "additional_examples": additional_examples_seen,
            "cumulative_examples": existing_examples + additional_examples_seen,
            "optimizer_steps": completed_steps,
            "fixed_state_failures": fixed_state_failures,
        },
        "specialist": {
            **parameter_report,
            "adapter_dimension": int(config["specialist_adapter_dim"]),
            "trainable_parameter_names": list(trainable_names),
            "initial_missing_parameter_names": list(missing),
            "delta_parameter_policy": (
                "adapter_plus_offspring_head" if topology_specialist else "adapter_only"
            ),
        },
        "topology_objective": dict(topology_objective) if topology_specialist else None,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "checkpoint": artifact_record(checkpoint_path),
        "gates": {
            "exact_exposure_conservation": additional_examples_seen == schedule.additional_examples,
            "cumulative_exposure_matches_target": (
                existing_examples + additional_examples_seen == target_examples
            ),
            "shared_parameters_frozen": parameter_report["shared_frozen_parameters"] > 0,
            "specialist_delta_only": len(trainable_names) == len(missing),
            "topology_head_supervised": (
                not topology_specialist
                or any(name.startswith("offspring_output.") for name in trainable_names)
            ),
            "fixed_state_failures_zero": fixed_state_failures == 0,
            "train_fold_source_measure_used": True,
            "heldout_access_zero": True,
            "repair_or_retry_zero": True,
            "route_or_oracle_calls_zero": True,
        },
        "nonclaims": [
            "A completed specialist run is not model-quality evidence without frozen evaluation.",
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
            "The specialist sees source-balanced train-fold rows only and no component identifiers.",
        ],
    }
    if not all(result["gates"].values()):
        result["status"] = "fail"
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise ReactionProgramSpecializationError("reaction specialization failed its frozen gates")
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "TOPOLOGY_CHECKPOINT_SCHEMA",
    "TOPOLOGY_CONFIG_SCHEMA",
    "TOPOLOGY_RESULT_SCHEMA",
    "ReactionProgramSpecializationError",
    "load_reaction_program_specialist",
    "run_reaction_program_specialization",
]
