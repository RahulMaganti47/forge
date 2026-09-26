"""Deterministic training for sparse Ugi closure-edge placement."""

from __future__ import annotations

import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.io import write_json as _atomic_json
from forge.corpus.ugi_morphology_corpus import (
    load_expanded_ugi_morphology_corpus,
    load_ugi_morphology_corpus,
)
from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_closure_placement import (
    UgiSparseClosureScorer,
    closure_set_loss,
    feasible_next_closures,
    sample_sparse_closures,
)
from forge.model.ugi_morphology_program import unique_component_morphologies
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class UgiClosureTrainingError(RuntimeError):
    """Raised when closure training violates its frozen sparse contract."""


def _closure_selection_key(evaluation: dict[str, Any]) -> tuple[float, float]:
    """Return a stable calibration key; lower is better.

    Mean negative log-likelihood is the primary selection statistic because
    exact-set recovery is a coarse, seed-dependent quantity on the small novel
    cyclic-component calibration set.  Recovery breaks exact NLL ties.
    """

    calibration = evaluation["calibration_novel"]
    if not calibration["components"]:
        calibration = evaluation["calibration_all"]
    mean_nll = calibration["mean_nll"]
    exact_fraction = calibration["exact_set_fraction"]
    if mean_nll is None or exact_fraction is None:
        raise UgiClosureTrainingError("closure calibration selection metric is undefined")
    return float(mean_nll), -float(exact_fraction)


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


def _resolve_input(repo: Path, record: dict[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise UgiClosureTrainingError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != str(record["sha256"]):
        raise UgiClosureTrainingError(
            f"{label} hash mismatch: expected {record['sha256']}, observed {observed}"
        )
    return path, {"path": str(path), "sha256": observed}


def _set_determinism(seed: int, device: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _cyclic_components(records: tuple[Any, ...]) -> tuple[Any, ...]:
    values = unique_component_morphologies(records).values()
    return tuple(
        sorted((value for value in values if value.cycle_rank), key=lambda x: x.component_key)
    )


def _expanded_fold_components(corpus: Any, fold: str) -> tuple[Any, ...]:
    values = {
        (component.role, component.component_key): component
        for role_families in corpus.families_by_fold_role[fold].values()
        for family in role_families.values()
        for component in family
    }
    return tuple(
        sorted(
            (component for component in values.values() if component.cycle_rank),
            key=lambda value: (value.role, value.component_key),
        )
    )


def _evaluate(
    model: Any,
    components: tuple[Any, ...],
    *,
    allowed_ring_sizes: tuple[int, ...],
    maximum_heavy_degree: int,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    if not components:
        return {
            "components": 0,
            "closure_edges": 0,
            "mean_nll": None,
            "exact_set_fraction": None,
            "exact_sets": 0,
            "mean_initial_candidate_count": None,
            "predictions": [],
        }
    losses = []
    exact = 0
    candidate_counts = []
    predictions = []
    generator = torch.Generator(device=device).manual_seed(seed)
    model.eval()
    for component in components:
        offspring = torch.as_tensor(component.offspring, dtype=torch.long, device=device)
        target = tuple(
            zip(
                component.closure_left.tolist(),
                component.closure_right.tolist(),
                strict=True,
            )
        )
        candidates = feasible_next_closures(
            component.offspring,
            remaining_closures_including_next=component.cycle_rank,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
            attachment_count=component.attachment_count,
        )
        candidate_counts.append(len(candidates.edges))
        with torch.no_grad():
            losses.append(
                float(
                    closure_set_loss(
                        model,
                        offspring,
                        role_index=ROLE_NAMES.index(component.role),
                        target_edges=target,
                        attachment_count=component.attachment_count,
                        allowed_ring_sizes=allowed_ring_sizes,
                        maximum_heavy_degree=maximum_heavy_degree,
                    )
                )
            )
            left, right = sample_sparse_closures(
                model,
                component.offspring,
                role_index=ROLE_NAMES.index(component.role),
                cycle_rank=component.cycle_rank,
                attachment_count=component.attachment_count,
                generator=generator,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                device=str(device),
                deterministic=True,
            )
        predicted = tuple(zip(left.tolist(), right.tolist(), strict=True))
        matched = set(predicted) == set(target)
        exact += int(matched)
        predictions.append(
            {
                "role": component.role,
                "component_smiles": component.component_key,
                "cycle_rank": component.cycle_rank,
                "initial_candidates": len(candidates.edges),
                "target": [list(edge) for edge in target],
                "predicted": [list(edge) for edge in predicted],
                "exact": matched,
            }
        )
    return {
        "components": len(components),
        "closure_edges": int(sum(component.cycle_rank for component in components)),
        "mean_nll": float(np.mean(losses)),
        "exact_set_fraction": exact / len(components),
        "exact_sets": exact,
        "mean_initial_candidate_count": float(np.mean(candidate_counts)),
        "predictions": predictions,
    }


def train_ugi_closure_scorer(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    smoke: bool,
    overwrite: bool,
    resume: bool = False,
) -> dict[str, Any]:
    """Train on strict-train unique cyclic components and audit novel heads."""

    if torch is None:
        raise UgiClosureTrainingError("closure training requires torch")
    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise UgiClosureTrainingError(f"invalid closure config: {exc}") from exc
    schema_version = config.get("schema_version")
    if schema_version not in {
        "phase1_ugi_sparse_closure_training_config.v1",
        "phase1_ugi_sparse_closure_training_config.v2",
    }:
        raise UgiClosureTrainingError("unsupported closure training config")
    if overwrite and resume:
        raise UgiClosureTrainingError("overwrite and resume are mutually exclusive")
    if output_dir.exists() and any(output_dir.iterdir()) and not (overwrite or resume):
        raise UgiClosureTrainingError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    inputs = {}
    for label, record in config["inputs"].items():
        resolved[label], inputs[label] = _resolve_input(repo, record, label)
    if schema_version == "phase1_ugi_sparse_closure_training_config.v2":
        corpus = load_expanded_ugi_morphology_corpus(
            resolved["component_exemplar_ledger"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        train_components = _expanded_fold_components(corpus, "train")
        calibration_components = _expanded_fold_components(corpus, "calibration")
        heldout_components = _expanded_fold_components(corpus, "heldout")
        expected = config.get("expected_cyclic_census")
        observed = {
            "train_components": len(train_components),
            "train_edges": int(sum(value.cycle_rank for value in train_components)),
            "calibration_components": len(calibration_components),
            "calibration_edges": int(sum(value.cycle_rank for value in calibration_components)),
            "heldout_components": len(heldout_components),
            "heldout_edges": int(sum(value.cycle_rank for value in heldout_components)),
        }
        if observed != expected:
            raise UgiClosureTrainingError(f"expanded cyclic-component census changed: {observed}")
        checkpoint_schema = "phase1_ugi_sparse_closure_checkpoint.v2"
        result_schema = "phase1_ugi_sparse_closure_result.v2"
    else:
        corpus = load_ugi_morphology_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        train_components = _cyclic_components(corpus.records_by_fold["train"])
        calibration_components = _cyclic_components(corpus.records_by_fold["calibration"])
        heldout_components = _cyclic_components(corpus.records_by_fold["heldout"])
        checkpoint_schema = "phase1_ugi_sparse_closure_checkpoint.v1"
        result_schema = "phase1_ugi_sparse_closure_result.v1"
    train_keys = {component.component_key for component in train_components}
    calibration_novel = tuple(
        component
        for component in calibration_components
        if component.component_key not in train_keys
    )
    heldout_novel = tuple(
        component for component in heldout_components if component.component_key not in train_keys
    )
    if schema_version.endswith(".v1") and (
        len(train_components) != 12 or sum(x.cycle_rank for x in train_components) != 12
    ):
        raise UgiClosureTrainingError("strict-train cyclic-component census changed")

    mode = "smoke" if smoke else "full"
    runtime = dict(config[mode])
    model_config = dict(config["model"])
    if smoke:
        model_config.update(runtime.pop("model_overrides"))
    device = torch.device(str(runtime["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise UgiClosureTrainingError("CUDA requested but unavailable")
    seed = int(config["seed"])
    _set_determinism(seed, device)
    allowed_ring_sizes = tuple(int(value) for value in config["support"]["ring_sizes"])
    maximum_heavy_degree = int(config["support"]["maximum_heavy_degree"])
    model = UgiSparseClosureScorer(**model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    rng = np.random.default_rng(seed + 1)
    losses = []
    evaluations = []
    checkpoint_path = output_dir / "checkpoint_best.pt"
    checkpoint_latest_path = output_dir / "checkpoint_latest.pt"
    best_key: tuple[float, float] | None = None
    best_step: int | None = None

    def checkpoint_value(step: int) -> dict[str, Any]:
        return {
            "schema_version": checkpoint_schema,
            "model_config": model_config,
            "model_state_dict": model.state_dict(),
            "allowed_ring_sizes": allowed_ring_sizes,
            "maximum_heavy_degree": maximum_heavy_degree,
            "seed": seed,
            "strict_train_component_keys": [
                component.component_key for component in train_components
            ],
            "inputs": inputs,
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "resume_state": {
                "best_key": best_key,
                "best_step": best_step,
                "evaluations": evaluations,
                "losses": losses,
                "numpy_legacy_state": np.random.get_state(),
                "numpy_training_state": rng.bit_generator.state,
                "python_random_state": random.getstate(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
        }

    def evaluate(step: int) -> None:
        nonlocal best_key, best_step
        evaluation = {
            "step": step,
            "train": _evaluate(
                model,
                train_components,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                device=device,
                seed=seed + step + 10,
            ),
            "calibration_all": _evaluate(
                model,
                calibration_components,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                device=device,
                seed=seed + step + 20,
            ),
            "calibration_novel": _evaluate(
                model,
                calibration_novel,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                device=device,
                seed=seed + step + 30,
            ),
        }
        evaluations.append(evaluation)
        selection_key = _closure_selection_key(evaluation)
        improved = best_key is None or selection_key < best_key
        if improved:
            best_key = selection_key
            best_step = step
        value = checkpoint_value(step)
        _atomic_checkpoint(checkpoint_latest_path, value)
        if improved:
            _atomic_checkpoint(checkpoint_path, value)

    start_step = 1
    if resume:
        if not checkpoint_latest_path.is_file():
            raise UgiClosureTrainingError(
                f"resume requested but latest checkpoint is missing: {checkpoint_latest_path}"
            )
        checkpoint = torch.load(checkpoint_latest_path, map_location=device, weights_only=False)
        if checkpoint.get("schema_version") != checkpoint_schema:
            raise UgiClosureTrainingError("resume checkpoint schema changed")
        if checkpoint.get("model_config") != model_config or checkpoint.get("inputs") != inputs:
            raise UgiClosureTrainingError("resume checkpoint contract differs from this run")
        state = checkpoint.get("resume_state")
        if not isinstance(state, dict):
            raise UgiClosureTrainingError("legacy checkpoint lacks deterministic resume state")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        losses = list(state["losses"])
        evaluations = list(state["evaluations"])
        stored_key = state["best_key"]
        best_key = tuple(stored_key) if stored_key is not None else None
        stored_step = state["best_step"]
        best_step = int(stored_step) if stored_step is not None else None
        random.setstate(state["python_random_state"])
        np.random.set_state(state["numpy_legacy_state"])
        rng.bit_generator.state = state["numpy_training_state"]
        torch.set_rng_state(state["torch_cpu_rng_state"].cpu())
        cuda_states = state.get("torch_cuda_rng_state_all")
        if device.type == "cuda":
            if not isinstance(cuda_states, list):
                raise UgiClosureTrainingError("resume checkpoint lacks CUDA RNG state")
            torch.cuda.set_rng_state_all(cuda_states)
        start_step = int(checkpoint["step"]) + 1
    else:
        evaluate(0)
    model.train()
    for step in range(start_step, int(runtime["steps"]) + 1):
        component = train_components[int(rng.integers(len(train_components)))]
        offspring = torch.as_tensor(component.offspring, dtype=torch.long, device=device)
        target = tuple(
            zip(
                component.closure_left.tolist(),
                component.closure_right.tolist(),
                strict=True,
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss = closure_set_loss(
            model,
            offspring,
            role_index=ROLE_NAMES.index(component.role),
            target_edges=target,
            attachment_count=component.attachment_count,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(runtime["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % int(runtime["eval_every"]) == 0 or step == int(runtime["steps"]):
            evaluate(step)
            model.train()
            _atomic_json(
                output_dir / "progress.json",
                {"status": "running", "mode": mode, "losses": losses, "evaluations": evaluations},
            )

    if best_key is None or best_step is None:
        raise UgiClosureTrainingError("closure training produced no selectable checkpoint")
    selected_evaluation = next(value for value in evaluations if value["step"] == best_step)
    selected_calibration = selected_evaluation["calibration_novel"]
    if not selected_calibration["components"]:
        selected_calibration = selected_evaluation["calibration_all"]
    selected_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(selected_checkpoint["model_state_dict"])
    heldout_selected = _evaluate(
        model,
        heldout_novel,
        allowed_ring_sizes=allowed_ring_sizes,
        maximum_heavy_degree=maximum_heavy_degree,
        device=device,
        seed=seed + best_step + 40,
    )
    result = {
        "schema_version": result_schema,
        "status": "complete",
        "mode": mode,
        "inputs": inputs,
        "support": {
            "ring_sizes": allowed_ring_sizes,
            "maximum_heavy_degree": maximum_heavy_degree,
            "dense_nonedge_tensor": False,
            "bond_order_generated_here": False,
        },
        "corpus": {
            "strict_train_cyclic_components": len(train_components),
            "strict_train_closure_edges": int(sum(x.cycle_rank for x in train_components)),
            "calibration_novel_cyclic_components": len(calibration_novel),
            "heldout_novel_cyclic_components": len(heldout_novel),
        },
        "runtime": runtime,
        "model": model_config,
        "losses": losses,
        "evaluations": evaluations,
        "heldout_selected": heldout_selected,
        "checkpoint": {"path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path)},
        "checkpoint_latest": {
            "path": str(checkpoint_latest_path),
            "sha256": sha256_file(checkpoint_latest_path),
        },
        "selection": {
            "metric": "calibration_novel_mean_nll",
            "best_step": best_step,
            "best_calibration_mean_nll": float(selected_calibration["mean_nll"]),
            "best_calibration_exact_set_fraction": float(
                selected_calibration["exact_set_fraction"]
            ),
            "completed_steps": int(runtime["steps"]),
        },
    }
    _atomic_json(output_dir / "result.json", result)
    _atomic_json(
        output_dir / "progress.json",
        {"status": "complete", "mode": mode, "losses": losses, "evaluations": evaluations},
    )
    return result
