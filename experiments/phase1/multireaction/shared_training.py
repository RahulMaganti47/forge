"""Bounded training gate for the shared Ugi/BL/LX sparse whole-product flow."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, sha256_file
from forge.core.io import read_json_object, stable_json, write_json
from forge.corpus.synthesis_program_training import load_synthesis_program_training_cache
from forge.model.defog_feasibility import _model_state_sha256, set_determinism
from forge.model.reaction_program_flow import (
    collate_synthesis_program_records,
    decode_synthesis_program_argmax,
    noise_synthesis_program_batch,
    synthesis_program_flow_loss,
)
from forge.model.sparse_topology_feasibility import (
    _masked_sparse_losses,
    _noise_sparse_batch,
)
from forge.model.synthesis_program_sampling import (
    CHECKPOINT_SCHEMA,
    synthesis_program_source_marginals,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
)
from forge.model.tensor_checkpoint import encode_tensor_state
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

CONFIG_SCHEMA = "forge.synthesis_program_training_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_training_result.v1"
RESTART_SCHEMA = "forge.synthesis_program_training_restart.v1"


class SharedSynthesisProgramTrainingError(ValueError):
    """The shared three-program training gate violates its pinned contract."""


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _model(cache: Any, config: dict[str, Any], device: Any) -> Any:
    return build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=config["model"],
        device=device,
    )


def _forward(
    model: Any,
    clean: dict[str, Any],
    node_p0: Any,
    bond_p0: Any,
    t: Any,
    generator: Any,
) -> tuple[Any, dict[str, float], dict[str, Any], dict[str, Any]]:
    noisy = noise_synthesis_program_batch(clean, node_p0, bond_p0, t, generator)
    predictions = model(
        nodes=noisy["nodes"],
        parents=noisy["parents"],
        parent_bonds=noisy["parent_bonds"],
        closure_left=noisy["closure_left"],
        closure_right=noisy["closure_right"],
        closure_bonds=noisy["closure_bonds"],
        t=t,
        node_mask=clean["node_mask"],
        child_mask=clean["child_mask"],
        closure_mask=clean["closure_mask"],
        program_states=clean["program_states"],
        role_states=clean["role_states"],
        core_position_states=clean["core_position_states"],
        program_depths=clean["program_depths"],
        adapter_mask=clean["adapter_mask"],
        repeat_group_states=clean["repeat_group_states"],
        component_position_states=clean["component_position_states"],
        component_instance_states=clean["component_instance_states"],
        role_morphology_states=clean["role_morphology_states"],
    )
    loss, metrics = synthesis_program_flow_loss(predictions, clean)
    return loss, metrics, predictions, noisy


def _fixed_exact(state: dict[str, Any], clean: dict[str, Any]) -> bool:
    return all(
        torch.equal(state[field][clean[mask]], clean[field][clean[mask]])
        for field, mask in {
            "nodes": "fixed_atom_mask",
            "parents": "fixed_parent_mask",
            "parent_bonds": "fixed_parent_bond_mask",
            "closure_left": "fixed_closure_endpoint_mask",
            "closure_right": "fixed_closure_endpoint_mask",
            "closure_bonds": "fixed_closure_bond_mask",
        }.items()
    )


def _reconstruction(predictions: dict[str, Any], clean: dict[str, Any]) -> dict[str, Any]:
    terminal = decode_synthesis_program_argmax(predictions, clean)
    fields = {
        "nodes": "node_mask",
        "parents": "child_mask",
        "parent_bonds": "child_mask",
        "closure_left": "closure_mask",
        "closure_right": "closure_mask",
        "closure_bonds": "closure_mask",
    }
    exact = torch.ones(clean["nodes"].shape[0], dtype=torch.bool, device=clean["nodes"].device)
    accuracies: dict[str, float] = {}
    for field, mask_name in fields.items():
        mask = clean[mask_name]
        correct = terminal[field] == clean[field]
        accuracies[field] = (
            float(correct[mask].to(torch.float32).mean()) if bool(mask.any()) else 1.0
        )
        exact &= (correct | ~mask).all(dim=1)
    return {
        "field_accuracy": accuracies,
        "exact_tensor_records": int(exact.sum()),
        "exact_tensor_fraction": float(exact.to(torch.float32).mean()),
        "fixed_states_exact": _fixed_exact(terminal, clean),
    }


def _validation(
    model: Any,
    cache: Any,
    config: dict[str, Any],
    device: Any,
    node_p0: Any,
    bond_p0: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    model.eval()
    with torch.no_grad():
        for index, record in enumerate(cache.records):
            clean = _move(
                collate_synthesis_program_records(
                    (record,), maximum_closures=int(config["model"]["maximum_closures"])
                ),
                device,
            )
            generator = torch.Generator(device=device).manual_seed(
                int(config["seed"]) + 10_000 + index
            )
            t = torch.full((1,), float(config["evaluation"]["corruption_time"]), device=device)
            _, metrics, predictions, noisy = _forward(model, clean, node_p0, bond_p0, t, generator)
            output[record.program_id] = {
                "record_id": record.graph.structure_id,
                **metrics,
                "fixed_noising_exact": _fixed_exact(noisy, clean),
                "reconstruction": _reconstruction(predictions, clean),
            }
    return output


def _zero_fixed_equivalence(
    model: Any,
    cache: Any,
    config: dict[str, Any],
    device: Any,
    node_p0: Any,
    bond_p0: Any,
) -> dict[str, Any]:
    auxiliary = tuple(record for record in cache.records if not np.any(record.fixed_atom_mask))
    clean = _move(
        collate_synthesis_program_records(
            auxiliary, maximum_closures=int(config["model"]["maximum_closures"])
        ),
        device,
    )
    t = torch.full((len(auxiliary),), 0.37, device=device)
    first = torch.Generator(device=device).manual_seed(int(config["seed"]) + 20_000)
    second = torch.Generator(device=device).manual_seed(int(config["seed"]) + 20_000)
    shared_noise = noise_synthesis_program_batch(clean, node_p0, bond_p0, t, first)
    generic_noise = _noise_sparse_batch(clean, node_p0, bond_p0, t, second)
    noise_exact = all(torch.equal(shared_noise[key], generic_noise[key]) for key in shared_noise)
    model.eval()
    with torch.no_grad():
        predictions = model(
            nodes=shared_noise["nodes"],
            parents=shared_noise["parents"],
            parent_bonds=shared_noise["parent_bonds"],
            closure_left=shared_noise["closure_left"],
            closure_right=shared_noise["closure_right"],
            closure_bonds=shared_noise["closure_bonds"],
            t=t,
            node_mask=clean["node_mask"],
            child_mask=clean["child_mask"],
            closure_mask=clean["closure_mask"],
            program_states=clean["program_states"],
            role_states=clean["role_states"],
            core_position_states=clean["core_position_states"],
            program_depths=clean["program_depths"],
            adapter_mask=clean["adapter_mask"],
            repeat_group_states=clean["repeat_group_states"],
            component_position_states=clean["component_position_states"],
            component_instance_states=clean["component_instance_states"],
            role_morphology_states=clean["role_morphology_states"],
        )
        shared_loss, shared_metrics = synthesis_program_flow_loss(predictions, clean)
        generic_loss, generic_metrics = _masked_sparse_losses(predictions, clean)
    return {
        "programs": [record.program_id for record in auxiliary],
        "noise_tensors_exact": noise_exact,
        "loss_exact": bool(torch.equal(shared_loss, generic_loss)),
        "metric_values_exact": shared_metrics == generic_metrics,
    }


def run_shared_synthesis_program_training(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    output_dir: Path,
    *,
    work_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Train one deterministic three-record overfit gate and emit a safe JSON checkpoint."""

    if torch is None:
        raise SharedSynthesisProgramTrainingError("shared training requires torch")
    config = read_json_object(
        config_path,
        error=SharedSynthesisProgramTrainingError,
        label="shared synthesis-program training config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SharedSynthesisProgramTrainingError("unsupported shared training config")
    execution = config.get("execution")
    if not isinstance(execution, dict) or execution.get("precision") != "float32":
        raise SharedSynthesisProgramTrainingError("shared training requires explicit float32")
    if execution.get("deterministic_algorithms") is not True:
        raise SharedSynthesisProgramTrainingError("deterministic algorithms must remain enabled")
    device = torch.device(str(execution["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SharedSynthesisProgramTrainingError("CUDA training requested but unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SharedSynthesisProgramTrainingError("MPS training requested but unavailable")
    cache = load_synthesis_program_training_cache(cache_path)
    if len(cache.records) != 3 or any(fold != "train" for fold in cache.source_folds):
        raise SharedSynthesisProgramTrainingError(
            "overfit cache is not the three-program train set"
        )
    model_config = config["model"]
    if max(record.node_count for record in cache.records) > int(
        model_config["maximum_heavy_atoms"]
    ):
        raise SharedSynthesisProgramTrainingError("model silently truncates heavy-atom support")
    if max(record.graph.closure_count for record in cache.records) > int(
        model_config["maximum_closures"]
    ):
        raise SharedSynthesisProgramTrainingError("model silently truncates closure support")
    seed = int(config["seed"])
    set_determinism(seed, int(execution["cpu_threads"]))
    model = _model(cache, config, device)
    node_marginal, bond_marginal = synthesis_program_source_marginals(
        cache.records,
        cache.sampling_weights,
        node_classes=len(cache.atom_vocabulary),
        bond_classes=int(model_config["bond_classes"]),
    )
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=device)
    training = config["training"]
    if int(training["batch_size"]) != len(cache.records):
        raise SharedSynthesisProgramTrainingError(
            "bounded gate requires one equal-mass example from every program per step"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    target_steps = int(training["steps"])
    checkpoint_interval = int(execution["checkpoint_interval_steps"])
    if checkpoint_interval < 1 or target_steps < checkpoint_interval:
        raise SharedSynthesisProgramTrainingError("invalid restart checkpoint interval")
    restart_identity = hashlib.sha256(
        stable_json(
            {
                "config_sha256": str(sha256_file(config_path)),
                "cache_sha256": str(sha256_file(cache_path)),
            }
        ).encode()
    ).hexdigest()
    restart_path = work_dir / "shared_synthesis_program_training_restart.pt"
    completed_steps = 0
    initial_loss: float | None = None
    final_loss: float | None = None
    minimum_loss = float("inf")
    all_finite = True
    fixed_noising_exact = True
    initial_objective_metrics: dict[str, float] | None = None
    final_objective_metrics: dict[str, float] | None = None
    gradient_balancing: dict[str, Any] = {
        "method": (
            "equal_family_mass_deterministic_pcgrad"
            if model_config.get("architecture") == "reaction_program_graph_transformer"
            else "none"
        ),
        "steps": 0,
        "projected_conflicts": 0,
        "first_raw_gradient_norms": None,
        "final_raw_gradient_norms": None,
    }

    def restart_payload() -> dict[str, Any]:
        return {
            "schema_version": RESTART_SCHEMA,
            "restart_identity": restart_identity,
            "target_steps": target_steps,
            "completed_steps": completed_steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": {
                "initial_loss": initial_loss,
                "final_loss": final_loss,
                "minimum_loss": minimum_loss,
                "all_finite": all_finite,
                "fixed_noising_exact": fixed_noising_exact,
                "initial_objective_metrics": initial_objective_metrics,
                "final_objective_metrics": final_objective_metrics,
                "gradient_balancing": gradient_balancing,
            },
            "random_state": capture_training_random_state(
                np.random.default_rng(seed + 2), generator, device=device
            ),
        }

    if resume and restart_path.is_file():
        payload = torch.load(restart_path, map_location=device, weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != RESTART_SCHEMA
            or payload.get("restart_identity") != restart_identity
            or int(payload.get("target_steps", -1)) != target_steps
        ):
            raise SharedSynthesisProgramTrainingError("training restart contract changed")
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        completed_steps = int(payload["completed_steps"])
        metrics = payload["metrics"]
        initial_loss = metrics["initial_loss"]
        final_loss = metrics["final_loss"]
        minimum_loss = float(metrics["minimum_loss"])
        all_finite = bool(metrics["all_finite"])
        fixed_noising_exact = bool(metrics["fixed_noising_exact"])
        initial_objective_metrics = metrics.get("initial_objective_metrics")
        final_objective_metrics = metrics.get("final_objective_metrics")
        gradient_balancing = dict(metrics.get("gradient_balancing", gradient_balancing))
        try:
            restore_training_random_state(
                payload["random_state"],
                np.random.default_rng(seed + 2),
                generator,
                device=device,
            )
        except (KeyError, TrainingRestartError) as error:
            raise SharedSynthesisProgramTrainingError(
                "training restart RNG state is invalid"
            ) from error
    clean = _move(
        collate_synthesis_program_training_batch(
            cache.records,
            maximum_closures=int(model_config["maximum_closures"]),
            conditioning="program",
            vocabulary=cache.vocabulary,
        ),
        device,
    )
    model.train()
    for step in range(completed_steps + 1, target_steps + 1):
        t = torch.rand(len(cache.records), generator=generator, device=device).clamp(0.02, 0.98)
        optimizer.zero_grad(set_to_none=True)
        if model_config.get("architecture") == "reaction_program_graph_transformer":
            from forge.model.reaction_program_transformer import (
                balanced_pcgrad_backward,
                per_program_transformer_losses,
            )

            noisy = noise_synthesis_program_batch(clean, node_p0, bond_p0, t, generator)
            predictions = model(
                nodes=noisy["nodes"],
                parents=noisy["parents"],
                parent_bonds=noisy["parent_bonds"],
                closure_left=noisy["closure_left"],
                closure_right=noisy["closure_right"],
                closure_bonds=noisy["closure_bonds"],
                t=t,
                node_mask=clean["node_mask"],
                child_mask=clean["child_mask"],
                closure_mask=clean["closure_mask"],
                program_states=clean["program_states"],
                role_states=clean["role_states"],
                core_position_states=clean["core_position_states"],
                program_depths=clean["program_depths"],
                adapter_mask=clean["adapter_mask"],
                repeat_group_states=clean["repeat_group_states"],
                component_position_states=clean["component_position_states"],
                component_instance_states=clean["component_instance_states"],
                role_morphology_states=clean["role_morphology_states"],
            )
            objective = model_config["semantic_objective"]
            family_losses, metrics = per_program_transformer_losses(
                predictions,
                clean,
                role_weight=float(objective["role_consistency_weight"]),
                core_weight=float(objective["core_consistency_weight"]),
                repeat_consistency_weight=float(objective.get("repeat_consistency_weight", 0.0)),
            )
            loss = torch.stack(list(family_losses.values())).mean()
            diagnostic = balanced_pcgrad_backward(family_losses, model)
            gradient_balancing["steps"] += 1
            gradient_balancing["projected_conflicts"] += int(diagnostic["projected_conflicts"])
            gradient_balancing["first_raw_gradient_norms"] = (
                gradient_balancing["first_raw_gradient_norms"] or diagnostic["raw_gradient_norms"]
            )
            gradient_balancing["final_raw_gradient_norms"] = diagnostic["raw_gradient_norms"]
            metrics["total"] = float(loss.detach())
        else:
            loss, metrics, _, noisy = _forward(model, clean, node_p0, bond_p0, t, generator)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        value = float(metrics["total"])
        initial_objective_metrics = (
            dict(metrics) if initial_objective_metrics is None else initial_objective_metrics
        )
        final_objective_metrics = dict(metrics)
        initial_loss = value if initial_loss is None else initial_loss
        final_loss = value
        minimum_loss = min(minimum_loss, value)
        all_finite = all_finite and all(np.isfinite(item) for item in metrics.values())
        fixed_noising_exact = fixed_noising_exact and _fixed_exact(noisy, clean)
        completed_steps = step
        if step % checkpoint_interval == 0 or step == target_steps:
            atomic_torch_save(restart_path, restart_payload())
    if initial_loss is None or final_loss is None:
        raise SharedSynthesisProgramTrainingError("training completed no optimization step")
    validation = _validation(model, cache, config, device, node_p0, bond_p0)
    zero_fixed_equivalence = _zero_fixed_equivalence(model, cache, config, device, node_p0, bond_p0)
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "trusted_local_checkpoint": True,
        "run_kind": "overfit_gate",
        "config": pin_record(config_path, repo),
        "cache": artifact_record(cache_path, logical_path="cache/cache.json"),
        "model_config": model_config,
        "program_vocabulary": {
            "program_states": list(cache.vocabulary.program_states),
            "role_states": list(cache.vocabulary.role_states),
            "core_position_states": list(cache.vocabulary.core_position_states),
            "maximum_steps": cache.vocabulary.maximum_steps,
        },
        "atom_vocabulary": [
            {
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "explicit_hydrogens": state.explicit_hydrogens,
            }
            for state in cache.atom_vocabulary
        ],
        "node_marginal": node_marginal.tolist(),
        "bond_marginal": bond_marginal.tolist(),
        "model_state_sha256": _model_state_sha256(model),
        "model_state": encode_tensor_state(model.state_dict()),
        "layout_contract": "semantic_coordinates_plus_adapter_fixed_states_only",
    }
    checkpoint_path = output_dir / "checkpoint.json"
    write_json(checkpoint_path, checkpoint)
    loss_ratio = final_loss / initial_loss
    exact_tensor_records = sum(
        int(row["reconstruction"]["exact_tensor_records"]) for row in validation.values()
    )
    gates = {
        "all_losses_finite": all_finite,
        "fixed_states_never_noised": fixed_noising_exact
        and all(bool(row["fixed_noising_exact"]) for row in validation.values()),
        "fixed_states_restored_at_decode": all(
            bool(row["reconstruction"]["fixed_states_exact"]) for row in validation.values()
        ),
        "loss_reduction": loss_ratio
        <= float(config["gates"]["maximum_final_to_initial_loss_ratio"]),
        "three_program_exact_tensor_reconstruction": exact_tensor_records
        >= int(config["gates"]["minimum_exact_tensor_records"]),
        "zero_fixed_auxiliary_path_matches_generic_flow": all(
            bool(zero_fixed_equivalence[key])
            for key in ("noise_tensors_exact", "loss_exact", "metric_values_exact")
        ),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "run_kind": "overfit_gate",
        "seed": seed,
        "config": pin_record(config_path, repo),
        "cache": artifact_record(cache_path, logical_path="cache/cache.json"),
        "checkpoint": artifact_record(checkpoint_path),
        "training": {
            "steps": target_steps,
            "examples_seen": target_steps * len(cache.records),
            "initial_total_loss": initial_loss,
            "final_total_loss": final_loss,
            "minimum_total_loss": minimum_loss,
            "final_to_initial_loss_ratio": loss_ratio,
            "all_losses_finite": all_finite,
            "fixed_noising_exact": fixed_noising_exact,
            "model_state_sha256": checkpoint["model_state_sha256"],
            "architecture": str(model_config.get("architecture", "sparse_mpnn")),
            "semantic_objective": model_config.get("semantic_objective"),
            "initial_objective_metrics": initial_objective_metrics,
            "final_objective_metrics": final_objective_metrics,
            "gradient_balancing": gradient_balancing,
            "restart": {
                "schema_version": RESTART_SCHEMA,
                "completed_steps": completed_steps,
                "checkpoint_interval_steps": checkpoint_interval,
            },
        },
        "validation": validation,
        "zero_fixed_equivalence": zero_fixed_equivalence,
        "gates": gates,
        "nonclaims": [
            "Three-record overfitting is an integration gate, not evidence of generalization.",
            "This run does not authorize production multi-reaction training.",
        ],
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SharedSynthesisProgramTrainingError",
    "run_shared_synthesis_program_training",
]
