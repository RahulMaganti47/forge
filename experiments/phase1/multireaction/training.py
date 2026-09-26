"""Deterministic bounded training for the multi-reaction sparse flow."""

from __future__ import annotations

import base64
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import pin_record, resolve_pin, sha256_file
from forge.core.io import atomic_write, pretty_json_bytes, read_json_object
from forge.corpus.reaction_program_training import load_reaction_program_training_corpus
from forge.model.defog_feasibility import _model_state_sha256, set_determinism
from forge.model.reaction_program_flow import (
    ReactionProgramSparseFlow,
    collate_reaction_program_records,
)
from forge.model.reaction_program_sampling import reaction_program_source_marginals
from forge.model.sparse_topology_feasibility import (
    _endpoint_candidate_mask,
    _masked_sparse_losses,
    _noise_sparse_batch,
    _parent_candidate_mask,
)
from forge.model.training_restart import (
    TrainingRestartError,
    atomic_torch_save,
    capture_training_random_state,
    restore_training_random_state,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.multireaction_training_config.v2"
RESULT_SCHEMA = "forge.multireaction_training_result.v2"
CHECKPOINT_SCHEMA = "forge.multireaction_sparse_flow_checkpoint.v2"
RESTART_SCHEMA = "forge.multireaction_training_restart.v1"


class MultiReactionTrainingError(ValueError):
    """The multi-reaction training smoke violates its pinned contract."""


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _model(corpus: Any, config: dict[str, Any], device: Any) -> Any:
    model = config["model"]
    return ReactionProgramSparseFlow(
        vocabulary=corpus.vocabulary,
        node_classes=len(corpus.atom_vocabulary),
        hidden_dim=int(model["hidden_dim"]),
        layers=int(model["layers"]),
        maximum_closures=int(model["maximum_closures"]),
        maximum_heavy_atoms=int(model["maximum_heavy_atoms"]),
        dropout=float(model["dropout"]),
        bond_classes=int(model["bond_classes"]),
    ).to(device)


def _condition_batch(batch: dict[str, Any], arm: str) -> None:
    if arm == "program":
        return
    if arm == "null":
        batch["program_states"].zero_()
        batch["role_states"].zero_()
        batch["core_position_states"].zero_()
        batch["program_depths"].zero_()
        return
    if arm == "program_id_shuffled":
        # Exactly two source-qualified programs are frozen for this experiment. Swapping their
        # nonzero labels destroys family identity while retaining role/depth channels and capacity.
        states = batch["program_states"]
        if bool(((states != 1) & (states != 2)).any()):
            raise MultiReactionTrainingError(
                "program-id shuffle requires exactly two active labels"
            )
        batch["program_states"] = torch.where(states == 1, 2, 1)
        return
    raise MultiReactionTrainingError(f"unknown conditioning arm: {arm!r}")


def _forward_loss(
    model: Any,
    clean: dict[str, Any],
    node_p0: Any,
    bond_p0: Any,
    generator: Any,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    t = torch.rand(clean["nodes"].shape[0], generator=generator, device=clean["nodes"].device)
    t = t.clamp(0.02, 0.98)
    noisy = _noise_sparse_batch(clean, node_p0, bond_p0, t, generator)
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
    )
    loss, metrics = _masked_sparse_losses(predictions, clean)
    return loss, metrics, predictions


def _fixed_noise_reconstruction(
    predictions: dict[str, Any],
    clean: dict[str, Any],
) -> dict[str, Any]:
    fields = {
        "nodes": "node_mask",
        "parents": "child_mask",
        "parent_bonds": "child_mask",
        "closure_left": "closure_mask",
        "closure_right": "closure_mask",
        "closure_bonds": "closure_mask",
    }
    exact_by_record = torch.ones(
        clean["nodes"].shape[0], dtype=torch.bool, device=clean["nodes"].device
    )
    accuracies: dict[str, float] = {}
    for field, mask_name in fields.items():
        mask = clean[mask_name]
        logits = predictions[field]
        if field == "parents":
            logits = logits.masked_fill(~_parent_candidate_mask(clean["node_mask"]), -1e9)
        elif field in {"closure_left", "closure_right"}:
            logits = logits.masked_fill(
                ~_endpoint_candidate_mask(clean["node_mask"], clean["closure_left"].shape[1]),
                -1e9,
            )
        correct = logits.argmax(dim=-1) == clean[field]
        denominator = int(mask.sum())
        accuracies[field] = float(correct[mask].to(torch.float32).mean()) if denominator else 1.0
        exact_by_record &= (correct | ~mask).all(dim=1)
    return {
        "field_accuracy": accuracies,
        "exact_tensor_records": int(exact_by_record.sum()),
        "exact_tensor_fraction": float(exact_by_record.to(torch.float32).mean()),
    }


def _train_arm(
    corpus: Any,
    config: dict[str, Any],
    arm: str,
    *,
    records: tuple[Any, ...],
    weights: np.ndarray,
    device: Any,
    node_p0: Any,
    bond_p0: Any,
    restart_path: Path,
    restart_identity: str,
    resume: bool,
    interrupt_after_step: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    training = config["training"]
    seed = int(config["seed"])
    set_determinism(seed, int(config["execution"]["cpu_threads"]))
    model = _model(corpus, config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    rng = np.random.default_rng(seed + 1)
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    target_steps = int(training["steps"])
    checkpoint_interval = int(config["execution"].get("checkpoint_interval_steps", target_steps))
    if checkpoint_interval < 1 or checkpoint_interval > target_steps:
        raise MultiReactionTrainingError(
            "checkpoint_interval_steps must be within the training run"
        )
    completed_steps = 0
    initial_total_loss: float | None = None
    final_total_loss: float | None = None
    minimum_total_loss = float("inf")
    all_losses_finite = True
    sampled_programs: Counter[str] = Counter()

    def restart_payload() -> dict[str, Any]:
        return {
            "schema_version": RESTART_SCHEMA,
            "restart_identity": restart_identity,
            "arm": arm,
            "target_steps": target_steps,
            "completed_steps": completed_steps,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "metrics": {
                "initial_total_loss": initial_total_loss,
                "final_total_loss": final_total_loss,
                "minimum_total_loss": minimum_total_loss,
                "all_losses_finite": all_losses_finite,
                "sampled_programs": dict(sampled_programs),
            },
            "random_state": capture_training_random_state(
                rng,
                generator,
                device=device,
            ),
        }

    if resume:
        if not restart_path.is_file():
            raise MultiReactionTrainingError(
                f"resume requested but arm checkpoint is missing: {restart_path}"
            )
        checkpoint = torch.load(restart_path, map_location=device, weights_only=False)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("schema_version") != RESTART_SCHEMA
            or checkpoint.get("restart_identity") != restart_identity
            or checkpoint.get("arm") != arm
            or checkpoint.get("target_steps") != target_steps
        ):
            raise MultiReactionTrainingError("arm restart checkpoint contract changed")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        completed_steps = int(checkpoint["completed_steps"])
        if completed_steps < 0 or completed_steps > target_steps:
            raise MultiReactionTrainingError("arm restart step is outside the run")
        saved_metrics = checkpoint.get("metrics")
        if not isinstance(saved_metrics, dict):
            raise MultiReactionTrainingError("arm restart checkpoint lacks metrics")
        initial_total_loss = saved_metrics["initial_total_loss"]
        final_total_loss = saved_metrics["final_total_loss"]
        minimum_total_loss = float(saved_metrics["minimum_total_loss"])
        all_losses_finite = bool(saved_metrics["all_losses_finite"])
        sampled_programs.update(saved_metrics["sampled_programs"])
        try:
            restore_training_random_state(
                checkpoint["random_state"],
                rng,
                generator,
                device=device,
            )
        except (KeyError, TrainingRestartError) as error:
            raise MultiReactionTrainingError("arm restart RNG state is invalid") from error
    elif restart_path.exists():
        raise MultiReactionTrainingError(
            f"arm checkpoint exists but resume was not requested: {restart_path}"
        )

    model.train()
    for step in range(completed_steps + 1, target_steps + 1):
        indices = rng.choice(
            len(records),
            size=int(training["batch_size"]),
            replace=True,
            p=weights,
        )
        local = tuple(records[int(index)] for index in indices)
        for record in local:
            sampled_programs[record.program_id] += 1
        clean = _move(
            collate_reaction_program_records(
                local,
                maximum_closures=int(config["model"]["maximum_closures"]),
            ),
            device,
        )
        _condition_batch(clean, arm)
        loss, metrics, _ = _forward_loss(model, clean, node_p0, bond_p0, generator)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        row = {**metrics, "gradient_norm": float(gradient_norm.detach())}
        row_is_finite = all(np.isfinite(value) for value in row.values())
        total = float(row["total"])
        initial_total_loss = total if initial_total_loss is None else initial_total_loss
        final_total_loss = total
        minimum_total_loss = min(minimum_total_loss, total)
        all_losses_finite = all_losses_finite and row_is_finite
        completed_steps = step
        if step % checkpoint_interval == 0 or step == target_steps:
            atomic_torch_save(restart_path, restart_payload())
        if interrupt_after_step == step:
            atomic_torch_save(restart_path, restart_payload())
            raise MultiReactionTrainingError(f"test interruption after step {step}")
    if initial_total_loss is None or final_total_loss is None:
        raise MultiReactionTrainingError("training arm has no completed optimization step")
    return model, {
        "steps": target_steps,
        "examples_seen": target_steps * int(training["batch_size"]),
        "initial_total_loss": initial_total_loss,
        "final_total_loss": final_total_loss,
        "minimum_total_loss": minimum_total_loss,
        "all_losses_finite": all_losses_finite,
        "sampled_programs": dict(sorted(sampled_programs.items())),
        "model_state_sha256": _model_state_sha256(model),
        "restart": {
            "schema_version": RESTART_SCHEMA,
            "checkpoint_interval_steps": checkpoint_interval,
            "completed_steps": completed_steps,
            "published_result_independent_of_interruption": True,
        },
    }


def _selected_training_records(
    corpus: Any,
    config: dict[str, Any],
) -> tuple[tuple[Any, ...], np.ndarray, dict[str, Any]]:
    selection = config.get("selection", {"mode": "all_training_records"})
    if not isinstance(selection, dict):
        raise MultiReactionTrainingError("training selection must be an object")
    mode = selection.get("mode")
    records = corpus.records_by_fold["train"]
    weights = corpus.weights_by_fold["train"]
    if mode == "all_training_records":
        return (
            records,
            weights,
            {
                "mode": mode,
                "record_count": len(records),
                "record_ids": [record.graph.structure_id for record in records],
            },
        )
    if mode != "deterministic_records_per_program":
        raise MultiReactionTrainingError(f"unsupported training selection mode: {mode!r}")
    requested = selection.get("records_per_program")
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise MultiReactionTrainingError("records_per_program must be a positive integer")
    seed = int(config["seed"])
    selected: list[Any] = []
    by_program: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        by_program[record.program_id].append(record)
    for program_id in corpus.vocabulary.program_states[1:]:
        local = by_program[program_id]
        if len(local) < requested:
            raise MultiReactionTrainingError(
                f"{program_id} has {len(local)} training records; selection requires {requested}"
            )
        ranked = sorted(
            local,
            key=lambda record: hashlib.sha256(
                f"{seed}|{program_id}|{record.graph.structure_id}".encode()
            ).hexdigest(),
        )
        selected.extend(ranked[:requested])
    # The bounded overfit gate tests both programs at equal total mass. It must not recover the
    # raw source-family count imbalance through the selection path.
    selected_weights = np.asarray(
        [1.0 / (len(by_program) * requested)] * len(selected),
        dtype=np.float64,
    )
    return (
        tuple(selected),
        selected_weights,
        {
            "mode": mode,
            "records_per_program": requested,
            "record_count": len(selected),
            "record_ids": [record.graph.structure_id for record in selected],
            "equal_total_weight_per_program": True,
        },
    )


def _validation(
    model: Any,
    corpus: Any,
    config: dict[str, Any],
    arm: str,
    *,
    training_records: tuple[Any, ...],
    device: Any,
    node_p0: Any,
    bond_p0: Any,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(config["seed"]) + 10)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 11)
    output: dict[str, Any] = {}
    evaluation = config.get("evaluation", {"fold": "calibration"})
    if not isinstance(evaluation, dict) or evaluation.get("fold") not in {
        "calibration",
        "selected_training",
    }:
        raise MultiReactionTrainingError("evaluation fold is invalid")
    source_records = (
        training_records
        if evaluation["fold"] == "selected_training"
        else corpus.records_by_fold["calibration"]
    )
    model.eval()
    with torch.no_grad():
        for program_id in corpus.vocabulary.program_states[1:]:
            records = tuple(record for record in source_records if record.program_id == program_id)
            size = min(int(config["training"]["validation_batch_size"]), len(records))
            indices = rng.choice(len(records), size=size, replace=False)
            local = tuple(records[int(index)] for index in indices)
            clean = _move(
                collate_reaction_program_records(
                    local,
                    maximum_closures=int(config["model"]["maximum_closures"]),
                ),
                device,
            )
            _condition_batch(clean, arm)
            _, metrics, predictions = _forward_loss(model, clean, node_p0, bond_p0, generator)
            output[program_id] = {
                "records": size,
                **metrics,
                "fixed_noise_reconstruction": _fixed_noise_reconstruction(predictions, clean),
            }
    return output


def _encoded_model_state(model: Any) -> dict[str, Any]:
    encoded: dict[str, Any] = {}
    for key, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        encoded[key] = {
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    return encoded


def run_multireaction_training(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    work_dir: Path,
    resume: bool,
) -> dict[str, Any]:
    """Train one bounded, explicitly classified multi-reaction experiment."""

    if torch is None:
        raise MultiReactionTrainingError("multi-reaction training requires torch")
    config = read_json_object(
        config_path,
        error=MultiReactionTrainingError,
        label="multi-reaction training config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise MultiReactionTrainingError("unsupported multi-reaction training config")
    run_kind = config.get("run_kind")
    if run_kind not in {"diagnostic_smoke", "overfit_gate", "production_qualification"}:
        raise MultiReactionTrainingError("multi-reaction training run_kind is invalid")
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise MultiReactionTrainingError("training execution contract is missing")
    if execution.get("precision") != "float32":
        raise MultiReactionTrainingError("only float32 training is currently qualified")
    if execution.get("deterministic_algorithms") is not True:
        raise MultiReactionTrainingError("deterministic algorithms must remain enabled")
    cpu_threads = execution.get("cpu_threads")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise MultiReactionTrainingError("cpu_threads must be a positive integer")
    input_paths = {
        label: resolve_pin(record, repo, label=label) for label, record in config["inputs"].items()
    }
    requested_device = str(config["execution"]["device"])
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise MultiReactionTrainingError("CUDA training requested but CUDA is unavailable")
    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise MultiReactionTrainingError("MPS training requested but MPS is unavailable")
    device = torch.device(requested_device)
    corpus = load_reaction_program_training_corpus(
        program_config_path=input_paths["program_config"],
        atlas_path=input_paths["atlas"],
        semantic_atoms_path=input_paths["semantic_atoms"],
        splits_path=input_paths["splits"],
        declared_elements=set(config["model"]["declared_elements"]),
    )
    fold_counts = {fold: len(records) for fold, records in corpus.records_by_fold.items()}
    if (
        fold_counts != config["expected"]["fold_counts"]
        or corpus.maxima != config["expected"]["maxima"]
    ):
        raise MultiReactionTrainingError(
            f"training corpus support changed: folds={fold_counts}, maxima={corpus.maxima}"
        )
    if corpus.maxima["heavy_atoms"] > int(config["model"]["maximum_heavy_atoms"]):
        raise MultiReactionTrainingError("model silently truncates heavy-atom support")
    if corpus.maxima["closures"] > int(config["model"]["maximum_closures"]):
        raise MultiReactionTrainingError("model silently truncates closure support")
    node_marginal, bond_marginal = reaction_program_source_marginals(
        corpus.records_by_fold["train"],
        node_classes=len(corpus.atom_vocabulary),
        bond_classes=int(config["model"]["bond_classes"]),
    )
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=device)
    training_records, training_weights, selection = _selected_training_records(corpus, config)
    restart_identity = hashlib.sha256(
        pretty_json_bytes(
            {
                "config_sha256": str(sha256_file(config_path)),
                "inputs": {
                    label: pin_record(path, repo) for label, path in sorted(input_paths.items())
                },
                "selection": selection,
            }
        )
    ).hexdigest()
    work_dir.mkdir(parents=True, exist_ok=True)
    arm_results: dict[str, Any] = {}
    selected_model = None
    for arm in config["training"]["arms"]:
        restart_path = work_dir / f"{arm}.restart.pt"
        model, metrics = _train_arm(
            corpus,
            config,
            str(arm),
            records=training_records,
            weights=training_weights,
            device=device,
            node_p0=node_p0,
            bond_p0=bond_p0,
            restart_path=restart_path,
            restart_identity=restart_identity,
            resume=resume and restart_path.is_file(),
        )
        if not metrics["all_losses_finite"]:
            raise MultiReactionTrainingError(f"{arm} produced non-finite training loss")
        metrics["evaluation_fold"] = config.get("evaluation", {}).get("fold", "calibration")
        metrics["evaluation_by_program"] = _validation(
            model,
            corpus,
            config,
            str(arm),
            training_records=training_records,
            device=device,
            node_p0=node_p0,
            bond_p0=bond_p0,
        )
        arm_results[str(arm)] = metrics
        if arm == "program":
            selected_model = model
    if selected_model is None:
        raise MultiReactionTrainingError("training arms omit the program-conditioned model")

    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "trusted_local_checkpoint": True,
        "config_sha256": str(sha256_file(config_path)),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(input_paths.items())},
        "model_config": dict(config["model"]),
        "program_vocabulary": {
            "program_states": list(corpus.vocabulary.program_states),
            "role_states": list(corpus.vocabulary.role_states),
            "core_position_states": list(corpus.vocabulary.core_position_states),
            "maximum_steps": corpus.vocabulary.maximum_steps,
        },
        "atom_vocabulary": [
            {
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "explicit_hydrogens": state.explicit_hydrogens,
            }
            for state in corpus.atom_vocabulary
        ],
        "node_marginal": node_marginal.tolist(),
        "bond_marginal": bond_marginal.tolist(),
        "model_state_sha256": _model_state_sha256(selected_model),
        "training_record_ids": selection["record_ids"],
        "model_state": _encoded_model_state(selected_model),
    }
    atomic_write(checkpoint_path, pretty_json_bytes(checkpoint))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": f"{run_kind}_complete",
        "run_kind": run_kind,
        "seed": int(config["seed"]),
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(input_paths.items())},
        "corpus": {
            "fold_counts": fold_counts,
            "maxima": corpus.maxima,
            "atom_vocabulary_size": len(corpus.atom_vocabulary),
            "program_states": list(corpus.vocabulary.program_states),
            "role_states": list(corpus.vocabulary.role_states),
            "core_position_states": list(corpus.vocabulary.core_position_states),
        },
        "selection": selection,
        "restartability": {
            "schema_version": RESTART_SCHEMA,
            "checkpoint_interval_steps": int(
                config["execution"].get("checkpoint_interval_steps", config["training"]["steps"])
            ),
            "optimizer_and_rng_state_captured": True,
            "published_result_independent_of_interruption": True,
            "work_state_is_not_a_published_scientific_artifact": True,
        },
        "arms": arm_results,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": str(sha256_file(checkpoint_path)),
            "bytes": checkpoint_path.stat().st_size,
            "model_state_sha256": _model_state_sha256(selected_model),
        },
        "claim_boundary": str(config["claim_boundary"]),
    }
    atomic_write(output_dir / "result.json", pretty_json_bytes(result))
    return result


__all__ = ["MultiReactionTrainingError", "run_multireaction_training"]
