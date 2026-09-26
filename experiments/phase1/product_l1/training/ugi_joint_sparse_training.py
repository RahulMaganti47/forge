"""Matched smoke training for the chemistry-aware joint sparse-flow arm."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache
from forge.core.io import write_json as _atomic_json
from forge.corpus.ugi_chemistry_corpus import (
    load_expanded_ugi_chemistry_corpus,
    load_ugi_chemistry_corpus,
)
from forge.corpus.ugi_morphology_corpus import (
    balanced_product_weights,
    source_stratified_family_weights,
)
from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    apply_component_role_mask,
    collate_ugi_joint_sparse_records,
    joint_sparse_source_marginals,
    noise_ugi_joint_sparse_batch,
    project_joint_sparse_record,
    ugi_joint_sparse_loss,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class UgiJointSparseTrainingError(RuntimeError):
    """Raised when the matched joint-flow experiment violates its contract."""


@dataclass(frozen=True)
class TrainingPartition:
    """Validated fold use for development training or a fixed-duration production refit."""

    mode: str
    training_folds: tuple[str, ...]
    diagnostic_folds: tuple[str, ...]
    selection_mode: str

    @property
    def overlapping_folds(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.training_folds).intersection(self.diagnostic_folds)))


def _training_partition(
    config: dict[str, Any],
    available_folds: tuple[str, ...],
    runtime: dict[str, Any],
) -> TrainingPartition:
    """Validate that fold reuse cannot masquerade as checkpoint selection evidence."""

    policy = dict(config.get("training_partition", {}))
    mode = str(policy.get("mode", "development_split"))
    training_folds = tuple(str(value) for value in policy.get("training_folds", ("train",)))
    diagnostic_folds = tuple(
        str(value) for value in policy.get("diagnostic_folds", ("calibration",))
    )
    selection_mode = str(policy.get("selection_mode", "calibration_early_stopping"))
    available = set(available_folds)
    if not training_folds or not diagnostic_folds:
        raise UgiJointSparseTrainingError("training and diagnostic folds must be nonempty")
    if len(set(training_folds)) != len(training_folds) or len(set(diagnostic_folds)) != len(
        diagnostic_folds
    ):
        raise UgiJointSparseTrainingError("training partition contains duplicate folds")
    unknown = (set(training_folds) | set(diagnostic_folds)).difference(available)
    if unknown:
        raise UgiJointSparseTrainingError(
            f"training partition contains unknown folds: {sorted(unknown)}"
        )
    partition = TrainingPartition(
        mode=mode,
        training_folds=training_folds,
        diagnostic_folds=diagnostic_folds,
        selection_mode=selection_mode,
    )
    early_stopping = dict(runtime.get("early_stopping", {}))
    patience = int(early_stopping.get("patience", 0))
    if mode == "development_split":
        if selection_mode != "calibration_early_stopping":
            raise UgiJointSparseTrainingError(
                "development split requires calibration early-stopping selection"
            )
        if partition.overlapping_folds:
            raise UgiJointSparseTrainingError(
                "development training and calibration folds must be disjoint"
            )
    elif mode == "fixed_train_only":
        if training_folds != ("train",):
            raise UgiJointSparseTrainingError(
                "fixed train-only fitting requires exactly the frozen train fold"
            )
        if partition.overlapping_folds:
            raise UgiJointSparseTrainingError(
                "fixed train-only diagnostics must remain disjoint from fitting"
            )
        if selection_mode != "fixed_final_step":
            raise UgiJointSparseTrainingError(
                "fixed train-only fitting must select the prespecified final step"
            )
        if patience != 0:
            raise UgiJointSparseTrainingError(
                "fixed train-only fitting cannot use data-dependent early stopping"
            )
    elif mode == "production_refit_all_folds":
        if set(training_folds) != available:
            raise UgiJointSparseTrainingError(
                "production refit must train on every available structural fold"
            )
        if selection_mode != "fixed_final_step":
            raise UgiJointSparseTrainingError("production refit must select the fixed final step")
        if patience != 0:
            raise UgiJointSparseTrainingError(
                "production refit cannot use data-dependent early stopping"
            )
    else:
        raise UgiJointSparseTrainingError(f"unsupported training-partition mode: {mode}")
    return partition


def _merge_fold_values(
    values_by_fold: dict[str, tuple[Any, ...]], folds: tuple[str, ...]
) -> tuple[Any, ...]:
    """Merge folds in the declared order without changing within-fold record order."""

    return tuple(value for fold in folds for value in values_by_fold[fold])


def _atomic_checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _resolve(repo: Path, record: dict[str, str], label: str) -> tuple[Path, dict[str, str]]:
    path = Path(record["path"])
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise UgiJointSparseTrainingError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != record["sha256"]:
        raise UgiJointSparseTrainingError(f"{label} hash changed")
    return path, {"path": str(path), "sha256": observed}


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def _training_weights(
    assignments: tuple[Any, ...],
    sampling: dict[str, Any],
) -> np.ndarray:
    mode = sampling.get("mode", "role_component_raked")
    if mode == "role_component_raked":
        return balanced_product_weights(assignments)
    if mode == "source_stratified_role_family_raked":
        source_mass = sampling.get("source_mass")
        return source_stratified_family_weights(
            assignments,
            source_mass=(
                {str(key): float(value) for key, value in source_mass.items()}
                if source_mass is not None
                else None
            ),
            uniform_row_mixture=float(sampling.get("uniform_row_mixture", 0.5)),
        )
    raise UgiJointSparseTrainingError(f"unsupported training sampling mode: {mode}")


def _loss_on_records(
    model: Any,
    records: tuple[Any, ...],
    sources: dict[str, Any],
    model_config: dict[str, Any],
    runtime: dict[str, Any],
    *,
    seed: int,
    device: Any,
    semantic_organization: str = "role_structured",
    objective: dict[str, Any] | None = None,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    collected: dict[str, list[float]] = {}
    model.eval()
    with torch.no_grad():
        for _ in range(int(runtime["validation_batches"])):
            indices = rng.choice(
                len(records),
                size=min(int(runtime["batch_size"]), len(records)),
                replace=False,
            )
            local = tuple(records[int(index)] for index in indices)
            batch = _move(
                collate_ugi_joint_sparse_records(
                    local,
                    maximum_nodes=max(record.node_count for record in local),
                    maximum_children=int(model_config["maximum_children"]),
                    maximum_closures=int(model_config["maximum_cycle_rank"]) * 3,
                    maximum_decorations=int(model_config["maximum_decorations"]),
                    semantic_organization=semantic_organization,
                ),
                device,
            )
            t = torch.rand(len(local), generator=generator, device=device).clamp(0.02, 0.98)
            noisy = noise_ugi_joint_sparse_batch(batch, sources, t, generator)
            predictions = model(
                offspring=noisy["offspring"],
                nodes=noisy["nodes"],
                parent_bonds=noisy["parent_bonds"],
                role_states=batch["role_states"],
                within_role_positions=batch["within_role_positions"],
                programs=batch["programs"],
                node_mask=batch["node_mask"],
                t=t,
                closure_left=batch["closure_left"],
                closure_right=batch["closure_right"],
                decoration_anchors=noisy["decoration_anchors"],
                decoration_atoms=noisy["decoration_atoms"],
                decoration_bonds=noisy["decoration_bonds"],
            )
            _, metrics = ugi_joint_sparse_loss(
                predictions,
                batch,
                semantic_organization=semantic_organization,
                loss_weights=(objective or {}).get("loss_weights"),
                consistency_weights=(objective or {}).get("program_consistency_weights"),
            )
            for key, value in metrics.items():
                collected.setdefault(key, []).append(value)
    return {key: float(np.mean(values)) for key, values in collected.items()}


def _records_by_source(
    records: tuple[Any, ...],
    assignments: tuple[Any, ...],
) -> dict[str, tuple[Any, ...]]:
    grouped: dict[str, list[Any]] = {}
    for record, assignment in zip(records, assignments, strict=True):
        source = str(assignment.get("source_stratum") or "unspecified")
        grouped.setdefault(source, []).append(record)
    return {source: tuple(values) for source, values in sorted(grouped.items())}


def _macro_average_metrics(values: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = set.intersection(*(set(metrics) for metrics in values.values()))
    return {
        key: float(np.mean([metrics[key] for metrics in values.values()])) for key in sorted(keys)
    }


def _validated_checkpoint_steps(runtime: dict[str, Any]) -> tuple[int, ...]:
    """Return deterministic serial-evaluation checkpoints allowed by the run."""

    try:
        steps = int(runtime["steps"])
        eval_every = int(runtime["eval_every"])
        requested = tuple(sorted({int(step) for step in runtime.get("checkpoint_steps", ())}))
    except (KeyError, TypeError, ValueError) as error:
        raise UgiJointSparseTrainingError("invalid checkpoint-step policy") from error
    if steps <= 0 or eval_every <= 0:
        raise UgiJointSparseTrainingError("invalid checkpoint-step policy")
    if any(step <= 0 or step > steps or step % eval_every != 0 for step in requested):
        raise UgiJointSparseTrainingError(
            "checkpoint steps must be positive evaluation steps within the run"
        )
    return requested


def _training_objective(config: dict[str, Any]) -> dict[str, Any]:
    """Validate optional loss balancing and component-level denoising controls."""

    objective = dict(config.get("objective", {}))
    expected = {"loss_weights", "program_consistency_weights", "role_block_mask_probability"}
    unknown = set(objective).difference(expected)
    if unknown:
        raise UgiJointSparseTrainingError(f"unknown joint objective fields: {sorted(unknown)}")
    loss_weights = dict(objective.get("loss_weights", {}))
    consistency_weights = dict(objective.get("program_consistency_weights", {}))
    try:
        role_block_mask_probability = float(objective.get("role_block_mask_probability", 0.0))
    except (TypeError, ValueError) as error:
        raise UgiJointSparseTrainingError("invalid role-block mask probability") from error
    if not 0 <= role_block_mask_probability <= 1:
        raise UgiJointSparseTrainingError("role-block mask probability must lie in [0, 1]")
    return {
        "loss_weights": {str(key): float(value) for key, value in loss_weights.items()},
        "program_consistency_weights": {
            str(key): float(value) for key, value in consistency_weights.items()
        },
        "role_block_mask_probability": role_block_mask_probability,
    }


def _learning_rate_at_step(runtime: dict[str, Any], step: int) -> float:
    """Return the deterministic constant or warmup-cosine learning rate."""

    base = float(runtime["learning_rate"])
    schedule = dict(runtime.get("learning_rate_schedule", {}))
    mode = str(schedule.get("mode", "constant"))
    if mode == "constant":
        if set(schedule).difference({"mode"}):
            raise UgiJointSparseTrainingError("constant learning-rate schedule has extra fields")
        return base
    if mode != "warmup_cosine" or set(schedule) != {
        "mode",
        "warmup_steps",
        "minimum_learning_rate_ratio",
    }:
        raise UgiJointSparseTrainingError("invalid warmup-cosine learning-rate schedule")
    total = int(runtime["steps"])
    warmup = int(schedule["warmup_steps"])
    minimum_ratio = float(schedule["minimum_learning_rate_ratio"])
    if not 0 < warmup < total or not 0 <= minimum_ratio <= 1 or not 1 <= step <= total:
        raise UgiJointSparseTrainingError("invalid warmup-cosine learning-rate support")
    if step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / (total - warmup)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def train_ugi_joint_sparse(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    smoke: bool,
    overwrite: bool,
    resume: bool = False,
    progress_callback: Callable[[int, Path], None] | None = None,
) -> dict[str, Any]:
    """Train one preregistered joint arm on the expanded chemistry exemplars."""

    if torch is None:
        raise UgiJointSparseTrainingError("joint sparse training requires torch")
    config = json.loads(config_path.read_text())
    config_sha256 = sha256_file(config_path)
    # Semantics experiment knobs. Absent from every production config, so existing runs are
    # unaffected and reproduce bit-identically.
    coverage_split = config.get("coverage_split")
    if config.get("schema_version") != "phase1_ugi_joint_sparse_training_config.v1":
        raise UgiJointSparseTrainingError("unsupported joint sparse config")
    if overwrite and resume:
        raise UgiJointSparseTrainingError("overwrite and resume are mutually exclusive")
    if output_dir.exists() and any(output_dir.iterdir()) and not (overwrite or resume):
        raise UgiJointSparseTrainingError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    inputs = {}
    for label, value in config["inputs"].items():
        resolved[label], inputs[label] = _resolve(repo, value, label)
    corpus_kind = config.get("corpus_kind", "expanded_fold_clean")
    cached_records_by_fold = None
    if "prepared_cache" in resolved:
        corpus, cached_records_by_fold = load_ugi_training_cache(resolved["prepared_cache"])
    elif corpus_kind == "expanded_fold_clean":
        corpus = load_expanded_ugi_chemistry_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
    elif corpus_kind == "original_ugi_12276":
        corpus = load_ugi_chemistry_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
    else:
        raise UgiJointSparseTrainingError(f"unsupported joint corpus kind: {corpus_kind}")
    records_by_fold = (
        cached_records_by_fold
        if cached_records_by_fold is not None
        else {
            fold: tuple(project_joint_sparse_record(record) for record in records)
            for fold, records in corpus.records_by_fold.items()
        }
    )
    if {fold: len(records) for fold, records in records_by_fold.items()} != config[
        "expected_fold_counts"
    ]:
        raise UgiJointSparseTrainingError("joint sparse fold counts changed")
    mode = "smoke" if smoke else "full"
    runtime = dict(config[mode])
    objective = _training_objective(config)
    # Validate the complete schedule before any model or optimizer state is written.
    for schedule_step in (1, int(runtime["steps"])):
        _learning_rate_at_step(runtime, schedule_step)
    model_config = dict(config["model"])
    if smoke:
        model_config.update(runtime.pop("model_overrides"))
    top_level_semantic = config.get("semantic_organization")
    model_semantic = model_config.get("semantic_organization")
    if (
        top_level_semantic is not None
        and model_semantic is not None
        and str(top_level_semantic) != str(model_semantic)
    ):
        raise UgiJointSparseTrainingError(
            "top-level and checkpoint model semantic organizations disagree"
        )
    semantic_organization = str(
        top_level_semantic
        if top_level_semantic is not None
        else model_semantic if model_semantic is not None else "role_structured"
    )
    device = torch.device(runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiJointSparseTrainingError("CUDA requested but unavailable")
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    partition = _training_partition(config, tuple(records_by_fold), runtime)
    train_records = _merge_fold_values(records_by_fold, partition.training_folds)
    train_assignments = _merge_fold_values(corpus.assignments_by_fold, partition.training_folds)
    diagnostic_records = _merge_fold_values(records_by_fold, partition.diagnostic_folds)
    diagnostic_assignments = _merge_fold_values(
        corpus.assignments_by_fold, partition.diagnostic_folds
    )
    if coverage_split is not None:
        sealed = json.loads((repo / str(coverage_split["path"])).read_text())
        digest = hashlib.sha256((repo / str(coverage_split["path"])).read_bytes()).hexdigest()
        if digest != str(coverage_split["sha256"]):
            raise UgiJointSparseTrainingError("sealed coverage split bytes changed")
        alpha = f"{float(coverage_split['alpha']):.2f}"
        keep = set(sealed["training_product_ids"][alpha])
        before = len(train_records)
        train_records = tuple(r for r in train_records if r.product_id in keep)
        if len(train_records) != len(keep):
            raise UgiJointSparseTrainingError(
                f"coverage subset {alpha} selected {len(train_records)} of {len(keep)} sealed ids"
            )
        train_assignments = tuple(a for a in train_assignments if a["product_id"] in keep)
        print(f"coverage alpha={alpha}: {before} -> {len(train_records)} training records")
    diagnostic_by_source = _records_by_source(diagnostic_records, diagnostic_assignments)
    sampling = dict(config.get("sampling", {}))
    weights = _training_weights(train_assignments, sampling)
    sources_np = joint_sparse_source_marginals(
        train_records,
        atom_classes=len(corpus.atom_vocabulary),
        bond_classes=int(model_config["bond_classes"]),
        maximum_children=int(model_config["maximum_children"]),
        maximum_decorations=int(model_config["maximum_decorations"]),
        probability_floor=float(model_config["source_probability_floor"]),
        record_weights=weights,
        semantic_organization=semantic_organization,
    )
    sources = {
        key: torch.as_tensor(value, dtype=torch.float32, device=device)
        for key, value in sources_np.items()
    }
    architecture = dict(model_config)
    architecture.pop("source_probability_floor")
    architecture.pop("semantic_organization", None)
    model = UgiJointSparseFlow(
        atom_classes=len(corpus.atom_vocabulary),
        semantic_organization=semantic_organization,
        **architecture,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    rng = np.random.default_rng(seed + 2)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    role_mask_generator = torch.Generator(device=device).manual_seed(seed + 3)
    losses: list[dict[str, float]] = []
    evaluations: list[dict[str, Any]] = []
    latest_path = output_dir / "checkpoint_latest.pt"
    best_path = output_dir / "checkpoint_best.pt"
    best_loss: float | None = None
    best_step = 0
    early_stopping = dict(runtime.get("early_stopping", {}))
    early_stopping_patience = int(early_stopping.get("patience", 0))
    early_stopping_min_delta = float(early_stopping.get("min_delta", 0.0))
    early_stopping_minimum_steps = int(early_stopping.get("minimum_steps", 0))
    if (
        early_stopping_patience < 0
        or early_stopping_min_delta < 0
        or early_stopping_minimum_steps < 0
    ):
        raise UgiJointSparseTrainingError("invalid joint early-stopping policy")
    evaluations_without_improvement = 0
    stop_reason = ""
    completed_step = 0
    checkpoint_steps = _validated_checkpoint_steps(runtime)
    checkpoint_snapshots: list[dict[str, Any]] = []

    def resume_state(step: int) -> dict[str, Any]:
        return {
            "best_loss": best_loss,
            "best_step": best_step,
            "checkpoint_snapshots": checkpoint_snapshots,
            "completed_step": step,
            "evaluations": evaluations,
            "evaluations_without_improvement": evaluations_without_improvement,
            "numpy_legacy_state": np.random.get_state(),
            "numpy_training_state": rng.bit_generator.state,
            "python_random_state": random.getstate(),
            "stop_reason": stop_reason,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if device.type == "cuda" else None
            ),
            "torch_training_generator_state": generator.get_state(),
            "torch_role_mask_generator_state": role_mask_generator.get_state(),
            "losses": losses,
        }

    def package(step: int) -> dict[str, Any]:
        return {
            "schema_version": "phase1_ugi_joint_sparse_checkpoint.v1",
            "step": step,
            "model_config": model_config,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "source_marginals": sources_np,
            "inputs": inputs,
            "effective_config_sha256": config_sha256,
            "experiment_arm": config.get("experiment_arm"),
            "training_objective": objective,
            "learning_rate_schedule": dict(runtime.get("learning_rate_schedule", {})),
            "resume_state": resume_state(step),
        }

    def evaluate(step: int) -> dict[str, Any]:
        diagnostic_loss_by_source = {
            source: _loss_on_records(
                model,
                records,
                sources,
                model_config,
                runtime,
                seed=seed + 30_000 + source_index,
                device=device,
                semantic_organization=semantic_organization,
                objective=objective,
            )
            for source_index, (source, records) in enumerate(diagnostic_by_source.items())
        }
        value = {
            "step": step,
            "training_loss": _loss_on_records(
                model,
                train_records,
                sources,
                model_config,
                runtime,
                seed=seed + 20_000,
                device=device,
                semantic_organization=semantic_organization,
                objective=objective,
            ),
            "diagnostic_loss": _macro_average_metrics(diagnostic_loss_by_source),
            "diagnostic_loss_by_source": diagnostic_loss_by_source,
        }
        if partition.selection_mode == "calibration_early_stopping":
            value["calibration_loss"] = value["diagnostic_loss"]
            value["calibration_loss_by_source"] = value["diagnostic_loss_by_source"]
        evaluations.append(value)
        return value

    start_step = 1
    if resume:
        if not latest_path.is_file():
            raise UgiJointSparseTrainingError(
                f"resume requested but latest checkpoint is missing: {latest_path}"
            )
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != "phase1_ugi_joint_sparse_checkpoint.v1":
            raise UgiJointSparseTrainingError("resume checkpoint schema changed")
        if checkpoint.get("model_config") != model_config or checkpoint.get("inputs") != inputs:
            raise UgiJointSparseTrainingError("resume checkpoint contract differs from this run")
        if checkpoint.get("effective_config_sha256", config_sha256) != config_sha256:
            raise UgiJointSparseTrainingError("resume effective training config changed")
        if checkpoint.get("experiment_arm", config.get("experiment_arm")) != config.get(
            "experiment_arm"
        ):
            raise UgiJointSparseTrainingError("resume experiment arm changed")
        checkpoint_objective = checkpoint.get(
            "training_objective",
            _training_objective({}),
        )
        if checkpoint_objective != objective or checkpoint.get(
            "learning_rate_schedule", {}
        ) != dict(runtime.get("learning_rate_schedule", {})):
            raise UgiJointSparseTrainingError("resume objective or learning-rate schedule changed")
        state = checkpoint.get("resume_state")
        if not isinstance(state, dict):
            raise UgiJointSparseTrainingError("legacy checkpoint lacks deterministic resume state")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        losses = list(state["losses"])
        evaluations = list(state["evaluations"])
        best_loss = state["best_loss"]
        best_step = int(state["best_step"])
        evaluations_without_improvement = int(state["evaluations_without_improvement"])
        stop_reason = str(state["stop_reason"])
        completed_step = int(state["completed_step"])
        checkpoint_snapshots = list(state["checkpoint_snapshots"])
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_legacy_state"])
        rng.bit_generator.state = state["numpy_training_state"]
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        cuda_states = state.get("torch_cuda_rng_state_all")
        if device.type == "cuda":
            if not isinstance(cuda_states, list):
                raise UgiJointSparseTrainingError("resume checkpoint lacks CUDA RNG state")
            torch.cuda.set_rng_state_all(cuda_states)
        # Generator state is serialized as a CPU ByteTensor even when the generator itself
        # targets CUDA.  ``set_state`` performs the device-specific restore internally.
        generator.set_state(state["torch_training_generator_state"].cpu())
        role_mask_state = state.get("torch_role_mask_generator_state")
        if role_mask_state is None:
            if objective["role_block_mask_probability"]:
                raise UgiJointSparseTrainingError(
                    "resume checkpoint lacks component-role mask generator state"
                )
        else:
            role_mask_generator.set_state(role_mask_state.cpu())
        for snapshot in checkpoint_snapshots:
            snapshot_path = Path(str(snapshot["path"]))
            if not snapshot_path.is_absolute():
                if snapshot_path.name != str(snapshot["path"]):
                    raise UgiJointSparseTrainingError(
                        "resume checkpoint snapshot path is not one local filename"
                    )
                snapshot_path = output_dir / snapshot_path
            if not snapshot_path.is_file() or sha256_file(snapshot_path) != snapshot["sha256"]:
                raise UgiJointSparseTrainingError(
                    f"resume checkpoint snapshot is missing or changed: {snapshot_path}"
                )
        start_step = completed_step + 1
    else:
        initial = evaluate(0)
        if partition.selection_mode == "calibration_early_stopping":
            best_loss = float(initial["calibration_loss"]["total"])
        _atomic_checkpoint(latest_path, package(0))
        if partition.selection_mode == "calibration_early_stopping":
            _atomic_checkpoint(best_path, package(0))
        if progress_callback is not None:
            progress_callback(0, output_dir)
    model.train()
    for step in range(start_step, int(runtime["steps"]) + 1):
        completed_step = step
        learning_rate = _learning_rate_at_step(runtime, step)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        indices = rng.choice(
            len(train_records),
            size=int(runtime["batch_size"]),
            replace=True,
            p=weights,
        )
        local = tuple(train_records[int(index)] for index in indices)
        batch = _move(
            collate_ugi_joint_sparse_records(
                local,
                maximum_nodes=max(record.node_count for record in local),
                maximum_children=int(model_config["maximum_children"]),
                maximum_closures=int(model_config["maximum_cycle_rank"]) * 3,
                maximum_decorations=int(model_config["maximum_decorations"]),
                semantic_organization=semantic_organization,
            ),
            device,
        )
        t = torch.rand(len(local), generator=generator, device=device).clamp(0.02, 0.98)
        noisy = noise_ugi_joint_sparse_batch(batch, sources, t, generator)
        noisy, masked_roles = apply_component_role_mask(
            noisy,
            batch,
            sources,
            probability=float(objective["role_block_mask_probability"]),
            generator=role_mask_generator,
        )
        predictions = model(
            offspring=noisy["offspring"],
            nodes=noisy["nodes"],
            parent_bonds=noisy["parent_bonds"],
            role_states=batch["role_states"],
            within_role_positions=batch["within_role_positions"],
            programs=batch["programs"],
            node_mask=batch["node_mask"],
            t=t,
            closure_left=batch["closure_left"],
            closure_right=batch["closure_right"],
            decoration_anchors=noisy["decoration_anchors"],
            decoration_atoms=noisy["decoration_atoms"],
            decoration_bonds=noisy["decoration_bonds"],
        )
        loss, metrics = ugi_joint_sparse_loss(
            predictions,
            batch,
            semantic_organization=semantic_organization,
            loss_weights=objective["loss_weights"],
            consistency_weights=objective["program_consistency_weights"],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(runtime["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(
            {
                "step": step,
                **metrics,
                "gradient_norm": float(gradient_norm),
                "learning_rate": learning_rate,
                "role_block_masked_examples": int((masked_roles >= 0).sum()),
            }
        )
        if step % int(runtime["eval_every"]) == 0 or step == int(runtime["steps"]):
            model.eval()
            evaluation = evaluate(step)
            observed = (
                float(evaluation["calibration_loss"]["total"])
                if partition.selection_mode == "calibration_early_stopping"
                else None
            )
            improved = (
                observed is not None
                and best_loss is not None
                and observed < best_loss - early_stopping_min_delta
            )
            if improved:
                best_loss = observed
                best_step = step
                evaluations_without_improvement = 0
            elif (
                partition.selection_mode == "calibration_early_stopping"
                and step >= early_stopping_minimum_steps
            ):
                evaluations_without_improvement += 1
            should_stop = (
                partition.selection_mode == "calibration_early_stopping"
                and early_stopping_patience > 0
                and step >= early_stopping_minimum_steps
                and evaluations_without_improvement >= early_stopping_patience
            )
            if should_stop:
                stop_reason = "calibration_early_stopping"
            checkpoint = package(step)
            _atomic_checkpoint(latest_path, checkpoint)
            if improved:
                _atomic_checkpoint(best_path, checkpoint)
            if step in checkpoint_steps and not any(
                int(snapshot["step"]) == step for snapshot in checkpoint_snapshots
            ):
                snapshot_path = output_dir / f"checkpoint_step_{step}.pt"
                _atomic_checkpoint(snapshot_path, checkpoint)
                checkpoint_snapshots.append(
                    {
                        "step": step,
                        "path": snapshot_path.name,
                        "sha256": sha256_file(snapshot_path),
                    }
                )
                _atomic_checkpoint(latest_path, package(step))
            _atomic_json(
                output_dir / "progress.json",
                {
                    "step": step,
                    "selection_mode": partition.selection_mode,
                    "best_step": best_step,
                    "best_calibration_loss": best_loss,
                    "evaluations_without_improvement": evaluations_without_improvement,
                    "stop_reason": stop_reason,
                    "losses": losses,
                    "evaluations": evaluations,
                },
            )
            if progress_callback is not None:
                progress_callback(step, output_dir)
            model.train()
            if should_stop:
                break
    if partition.selection_mode == "fixed_final_step":
        best_step = completed_step
        _atomic_checkpoint(best_path, package(completed_step))
    result = {
        "schema_version": "phase1_ugi_joint_sparse_training_result.v1",
        "status": "complete",
        "mode": mode,
        "inputs": inputs,
        "corpus": {fold: len(records) for fold, records in records_by_fold.items()},
        "corpus_kind": corpus_kind,
        "training_partition": {
            "mode": partition.mode,
            "training_folds": list(partition.training_folds),
            "training_records": len(train_records),
            "diagnostic_folds": list(partition.diagnostic_folds),
            "diagnostic_records": len(diagnostic_records),
            "overlapping_folds": list(partition.overlapping_folds),
            "diagnostic_is_in_sample": bool(partition.overlapping_folds),
            "selection_mode": partition.selection_mode,
        },
        "model": model_config,
        "effective_config_sha256": config_sha256,
        "experiment_arm": config.get("experiment_arm"),
        "objective": objective,
        "runtime": runtime,
        "sampling": {
            **sampling,
            "observed_source_mass": {
                source: float(
                    weights[
                        np.asarray(
                            [
                                index
                                for index, row in enumerate(train_assignments)
                                if str(row.get("source_stratum") or "unspecified") == source
                            ],
                            dtype=np.int64,
                        )
                    ].sum()
                )
                for source in sorted(
                    {row.get("source_stratum", "unspecified") for row in train_assignments}
                )
            },
        },
        "selection": {
            "mode": partition.selection_mode,
            "best_step": best_step,
            "best_calibration_loss": best_loss,
            "completed_steps": completed_step,
            "stop_reason": stop_reason or "maximum_steps",
            "early_stopping_patience": early_stopping_patience,
            "early_stopping_min_delta": early_stopping_min_delta,
            "early_stopping_minimum_steps": early_stopping_minimum_steps,
            "diagnostic_loss_is_nonselecting": partition.selection_mode == "fixed_final_step",
        },
        "losses": losses,
        "evaluations": evaluations,
        "checkpoint": {"path": str(best_path), "sha256": sha256_file(best_path)},
        "checkpoint_latest": {
            "path": str(latest_path),
            "sha256": sha256_file(latest_path),
        },
        "checkpoint_snapshots": checkpoint_snapshots,
        "boundary": {
            "same_sparse_representation_as_staged_arm": True,
            "topology_and_chemistry_shared_backbone": True,
            "component_ids_in_model_state": False,
            "route_or_oracle_guidance": False,
        },
    }
    _atomic_json(output_dir / "result.json", result)
    return result
