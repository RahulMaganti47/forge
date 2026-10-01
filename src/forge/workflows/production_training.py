"""Restartable matched training for the frozen Ugi/BL/LX comparison."""

from __future__ import annotations

import hashlib
import random
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, stable_json, write_json
from forge.corpus.synthesis_program_production_cache import (
    SynthesisProgramProductionCache,
    SynthesisProgramProductionCacheError,
)
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_fixed_state_exact_tensor,
    synthesis_program_forward,
    synthesis_program_forward_loss,
    synthesis_program_paired_topology_forward,
    synthesis_program_topology_conditioned_forward,
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

CONFIG_SCHEMA = "forge.synthesis_program_production_training_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_production_training_result.v1"
CHECKPOINT_SCHEMA = "forge.synthesis_program_production_checkpoint.v1"
RESTART_SCHEMA = "forge.synthesis_program_production_restart.v1"

_TRAINING_OPTIMIZATION_DEFAULTS = {
    "padding_aware_quantile_bins": 1,
    "pcgrad_backend": "sequential",
    "topology_conditioned_forward_mode": "sequential",
}


class SynthesisProgramProductionTrainingError(ValueError):
    """A matched production arm violates its frozen training contract."""


def _deterministic_tar(paths: list[Path], target: Path, *, base: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths, key=lambda value: value.relative_to(base).as_posix()):
            info = tarfile.TarInfo(path.relative_to(base).as_posix())
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _set_determinism(seed: int, cpu_threads: int, device: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False


def _validate_runtime(
    *, config: dict[str, Any], design: dict[str, Any], profile: str, device: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile not in {"smoke", "full"}:
        raise SynthesisProgramProductionTrainingError(f"unsupported profile: {profile}")
    runtime = dict(config[profile])
    model = dict(design["model"])
    if profile == "smoke":
        model.update(runtime.pop("model_overrides"))
    else:
        training = design["training"]
        exact = {
            "optimizer_steps": int(training["optimizer_steps"]),
            "micro_batch_size": int(training["micro_batch_size"]),
            "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training["weight_decay"]),
            "gradient_clip_norm": float(training["gradient_clip_norm"]),
            "checkpoint_steps": [int(value) for value in training["checkpoint_steps"]],
        }
        for key, value in exact.items():
            if runtime.get(key) != value:
                raise SynthesisProgramProductionTrainingError(
                    f"full runtime differs from frozen design for {key}"
                )
        if runtime.get("device") != design["execution"]["device"]:
            raise SynthesisProgramProductionTrainingError(
                "full runtime device differs from the frozen design"
            )
        if model != design["model"]:
            raise SynthesisProgramProductionTrainingError("full model contract changed")
    if str(runtime.get("device")) != device.type:
        raise SynthesisProgramProductionTrainingError(
            "runtime device and allocated stage device differ"
        )
    if runtime.get("precision") != "float32" or runtime.get("deterministic_algorithms") is not True:
        raise SynthesisProgramProductionTrainingError(
            "production training requires deterministic float32"
        )
    if int(runtime["micro_batch_size"]) * int(runtime["gradient_accumulation_steps"]) != int(
        runtime["effective_batch_size"]
    ):
        raise SynthesisProgramProductionTrainingError("effective batch size is inconsistent")
    checkpoints = [int(value) for value in runtime["checkpoint_steps"]]
    steps = int(runtime["optimizer_steps"])
    if (
        checkpoints != sorted(set(checkpoints))
        or not checkpoints
        or checkpoints[-1] != steps
        or checkpoints[0] < 1
    ):
        raise SynthesisProgramProductionTrainingError("checkpoint schedule is invalid")
    if int(runtime["restart_interval_steps"]) < 1:
        raise SynthesisProgramProductionTrainingError("restart interval is invalid")
    _training_optimization(runtime)
    return runtime, model


def _training_optimization(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Return one explicit, validated execution-only optimization policy."""

    raw = runtime.get("optimization", {})
    if not isinstance(raw, Mapping) or set(raw).difference(_TRAINING_OPTIMIZATION_DEFAULTS):
        raise SynthesisProgramProductionTrainingError(
            "training optimization policy contains unsupported fields"
        )
    policy = {**_TRAINING_OPTIMIZATION_DEFAULTS, **dict(raw)}
    bins = policy["padding_aware_quantile_bins"]
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise SynthesisProgramProductionTrainingError(
            "padding-aware quantile bins must be a positive integer"
        )
    if policy["pcgrad_backend"] not in {"sequential", "batched_vjp"}:
        raise SynthesisProgramProductionTrainingError("unsupported PCGrad execution backend")
    if policy["topology_conditioned_forward_mode"] not in {
        "sequential",
        "paired_batch",
    }:
        raise SynthesisProgramProductionTrainingError(
            "unsupported topology-conditioned forward mode"
        )
    return policy


def _restart_identity(
    *, design_path: Path, cache_path: Path, config_path: Path, arm_id: str, seed: int
) -> str:
    return hashlib.sha256(
        stable_json(
            {
                "arm_id": arm_id,
                "cache_sha256": str(sha256_file(cache_path)),
                "config_sha256": str(sha256_file(config_path)),
                "design_sha256": str(sha256_file(design_path)),
                "seed": seed,
            }
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class _StratifiedProgramSampler:
    """Compiled source-balanced categorical groups for one production arm."""

    program_states: tuple[int, ...]
    program_ids: tuple[str, ...]
    indices: tuple[np.ndarray, ...]
    probabilities: tuple[np.ndarray, ...]
    fixed_batch_counts: tuple[int, ...] | None = None
    size_sorted_indices: tuple[np.ndarray, ...] | None = None
    size_cumulative_probabilities: tuple[np.ndarray, ...] | None = None

    def batch_counts(self, batch_size: int) -> tuple[int, ...]:
        """Return the deterministic per-family row allocation for one microbatch."""

        if self.fixed_batch_counts is not None:
            if sum(self.fixed_batch_counts) != batch_size:
                raise SynthesisProgramProductionTrainingError(
                    "configured program counts do not sum to the runtime microbatch size"
                )
            return self.fixed_batch_counts
        if batch_size < len(self.program_states):
            raise SynthesisProgramProductionTrainingError(
                "Transformer batch cannot represent every active reaction family"
            )
        base, remainder = divmod(batch_size, len(self.program_states))
        return tuple(base + int(index < remainder) for index in range(len(self.program_states)))

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        *,
        padding_aware_quantile_bins: int = 1,
    ) -> np.ndarray:
        counts = self.batch_counts(batch_size)
        selected: list[int] = []
        if padding_aware_quantile_bins < 1:
            raise SynthesisProgramProductionTrainingError(
                "padding-aware quantile bins must be positive"
            )
        if padding_aware_quantile_bins == 1:
            for count, indices, probabilities in zip(
                counts, self.indices, self.probabilities, strict=True
            ):
                selected.extend(
                    rng.choice(indices, size=count, replace=True, p=probabilities).tolist()
                )
        else:
            if self.size_sorted_indices is None or self.size_cumulative_probabilities is None:
                raise SynthesisProgramProductionTrainingError(
                    "padding-aware sampling was requested without compiled size distributions"
                )
            # Drawing one shared quantile interval aligns molecule sizes across families.  Because
            # the interval is uniform and draws are uniform inside it, every unconditional draw is
            # still Uniform(0, 1); inverse-CDF sampling therefore preserves each exact categorical
            # source measure and retains nonzero probability for every supported record.
            quantile_bin = int(rng.integers(padding_aware_quantile_bins))
            for count, indices, cumulative in zip(
                counts,
                self.size_sorted_indices,
                self.size_cumulative_probabilities,
                strict=True,
            ):
                draws = (quantile_bin + rng.random(count)) / padding_aware_quantile_bins
                positions = np.searchsorted(cumulative, draws, side="right")
                selected.extend(indices[positions].tolist())
        values = np.asarray(selected, dtype=np.int64)
        rng.shuffle(values)
        return values


def _compile_stratified_program_sampler(
    cache: SynthesisProgramProductionCache,
    measure: np.ndarray,
    *,
    batch_program_counts: Mapping[str, int] | None = None,
) -> _StratifiedProgramSampler:
    """Compile family indices and source weights once instead of rescanning every micro-batch."""

    program_states: list[int] = []
    program_ids: list[str] = []
    family_indices: list[np.ndarray] = []
    family_probabilities: list[np.ndarray] = []
    size_sorted_indices: list[np.ndarray] = []
    size_cumulative_probabilities: list[np.ndarray] = []
    node_counts = np.diff(cache.arrays["node_offsets"])
    for program_state, program in enumerate(cache.vocabulary.program_states[1:], start=1):
        indices = cache.indices(program_id=program, fold="train")
        weights = measure[indices]
        total = float(weights.sum())
        if total <= 0.0:
            continue
        program_states.append(program_state)
        program_ids.append(program)
        family_indices.append(indices)
        probabilities = weights / total
        family_probabilities.append(probabilities)
        order = np.lexsort((indices, node_counts[indices]))
        size_sorted_indices.append(indices[order])
        cumulative = np.cumsum(probabilities[order])
        cumulative[-1] = 1.0
        size_cumulative_probabilities.append(cumulative)
    if not program_states:
        raise SynthesisProgramProductionTrainingError(
            "Transformer arm has no active reaction family"
        )
    fixed_batch_counts = None
    if batch_program_counts is not None:
        expected = set(cache.vocabulary.program_states[1:])
        if set(batch_program_counts) != expected:
            raise SynthesisProgramProductionTrainingError(
                "configured program counts do not cover the cache vocabulary"
            )
        inactive = expected.difference(program_ids)
        if any(int(batch_program_counts[program]) != 0 for program in inactive):
            raise SynthesisProgramProductionTrainingError(
                "configured program counts allocate rows to an inactive reaction family"
            )
        fixed_batch_counts = tuple(int(batch_program_counts[program]) for program in program_ids)
        if any(count < 1 for count in fixed_batch_counts):
            raise SynthesisProgramProductionTrainingError(
                "every active reaction family requires a positive configured row count"
            )
    return _StratifiedProgramSampler(
        program_states=tuple(program_states),
        program_ids=tuple(program_ids),
        indices=tuple(family_indices),
        probabilities=tuple(family_probabilities),
        fixed_batch_counts=fixed_batch_counts,
        size_sorted_indices=tuple(size_sorted_indices),
        size_cumulative_probabilities=tuple(size_cumulative_probabilities),
    )


def _train_arm(
    *,
    arm_id: str,
    arm: dict[str, Any],
    seed: int,
    cache: SynthesisProgramProductionCache,
    design_path: Path,
    cache_path: Path,
    config_path: Path,
    runtime: dict[str, Any],
    model_config: dict[str, Any],
    device: Any,
    work_dir: Path,
    resume: bool,
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
) -> dict[str, Any]:
    arm_dir = work_dir / arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    completed_receipt = arm_dir / "result.json"
    if resume and completed_receipt.is_file():
        result = read_json_object(
            completed_receipt,
            error=SynthesisProgramProductionTrainingError,
            label=f"completed arm {arm_id}",
        )
        for snapshot in result["checkpoints"]:
            path = arm_dir / snapshot["filename"]
            if not path.is_file() or str(sha256_file(path)) != snapshot["sha256"]:
                raise SynthesisProgramProductionTrainingError(
                    f"completed arm checkpoint changed: {path}"
                )
        return result

    arm_model_config = {**model_config, **dict(arm.get("model_overrides", {}))}
    _set_determinism(seed, int(runtime["cpu_threads"]), device)
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=arm_model_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    measure = cache.training_measure(arm["program_mass"])
    stratified_sampler = (
        _compile_stratified_program_sampler(
            cache,
            measure,
            batch_program_counts=arm.get("batch_program_counts"),
        )
        if arm_model_config.get("architecture") == "reaction_program_graph_transformer"
        else None
    )
    support = np.flatnonzero(measure > 0)
    probabilities = measure[support]
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed + 2)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=device)
    restart_path = arm_dir / "restart_latest.pt"
    identity = _restart_identity(
        design_path=design_path,
        cache_path=cache_path,
        config_path=config_path,
        arm_id=arm_id,
        seed=seed,
    )
    completed_steps = 0
    examples_seen = 0
    examples_seen_by_program = {program: 0 for program in cache.vocabulary.program_states[1:]}
    losses: list[dict[str, Any]] = []
    fixed_state_failures = 0
    snapshots: list[dict[str, Any]] = []
    balancing_diagnostics = {"micro_batches": 0, "projected_conflicts": 0}

    def restart_payload() -> dict[str, Any]:
        return {
            "schema_version": RESTART_SCHEMA,
            "restart_identity": identity,
            "arm_id": arm_id,
            "seed": seed,
            "completed_steps": completed_steps,
            "examples_seen": examples_seen,
            "examples_seen_by_program": examples_seen_by_program,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "losses": losses,
            "fixed_state_failures": fixed_state_failures,
            "snapshots": snapshots,
            "balancing_diagnostics": balancing_diagnostics,
            "random_state": capture_training_random_state(rng, generator, device=device),
        }

    if resume and restart_path.is_file():
        payload = torch.load(restart_path, map_location=device, weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != RESTART_SCHEMA
            or payload.get("restart_identity") != identity
        ):
            raise SynthesisProgramProductionTrainingError(
                f"restart contract changed for arm {arm_id}"
            )
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        completed_steps = int(payload["completed_steps"])
        examples_seen = int(payload["examples_seen"])
        restored_examples = payload.get("examples_seen_by_program")
        if not isinstance(restored_examples, dict) or set(restored_examples) != set(
            examples_seen_by_program
        ):
            raise SynthesisProgramProductionTrainingError(
                "arm restart checkpoint lacks per-program exposure counts"
            )
        examples_seen_by_program = {
            program: int(restored_examples[program]) for program in examples_seen_by_program
        }
        losses = list(payload["losses"])
        fixed_state_failures = int(payload["fixed_state_failures"])
        snapshots = list(payload["snapshots"])
        balancing_diagnostics = dict(payload.get("balancing_diagnostics", balancing_diagnostics))
        try:
            restore_training_random_state(payload["random_state"], rng, generator, device=device)
        except TrainingRestartError as error:
            raise SynthesisProgramProductionTrainingError(
                f"restart RNG state is invalid for arm {arm_id}"
            ) from error
        for snapshot in snapshots:
            path = arm_dir / snapshot["filename"]
            if not path.is_file() or str(sha256_file(path)) != snapshot["sha256"]:
                raise SynthesisProgramProductionTrainingError(f"restart snapshot changed: {path}")

    conditioning = str(arm["conditioning"])
    mapping = arm.get("program_id_mapping")
    maximum_closures = int(arm_model_config["maximum_closures"])
    accumulation = int(runtime["gradient_accumulation_steps"])
    micro_batch = int(runtime["micro_batch_size"])
    target_steps = int(runtime["optimizer_steps"])
    checkpoint_steps = set(int(value) for value in runtime["checkpoint_steps"])
    restart_interval = int(runtime["restart_interval_steps"])
    optimization = _training_optimization(runtime)
    active_program_states = (
        stratified_sampler.program_states if stratified_sampler is not None else ()
    )
    model.train()
    for optimizer_step in range(completed_steps + 1, target_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_metrics: dict[str, Any] = {}
        step_conflicts = torch.zeros((), dtype=torch.int64, device=device)
        step_fixed_state_failures = torch.zeros((), dtype=torch.int64, device=device)
        for _ in range(accumulation):
            if arm_model_config.get("architecture") == "reaction_program_graph_transformer":
                if stratified_sampler is None:  # pragma: no cover - construction invariant
                    raise SynthesisProgramProductionTrainingError(
                        "Transformer sampler was not compiled"
                    )
                selected = stratified_sampler.sample(
                    micro_batch,
                    rng,
                    padding_aware_quantile_bins=int(optimization["padding_aware_quantile_bins"]),
                )
            else:
                selected = rng.choice(support, size=micro_batch, replace=True, p=probabilities)
            records = cache.records(selected)
            sampled_states, sampled_counts = np.unique(
                cache.arrays["program_states"][selected], return_counts=True
            )
            for program_state, count in zip(sampled_states, sampled_counts, strict=True):
                program_id = cache.vocabulary.program_states[int(program_state)]
                examples_seen_by_program[program_id] += int(count)
            clean = move_tensors(
                collate_synthesis_program_training_batch(
                    records,
                    maximum_closures=maximum_closures,
                    conditioning=conditioning,
                    vocabulary=cache.vocabulary,
                    program_id_mapping=mapping,
                ),
                device,
            )
            t = torch.rand(micro_batch, generator=generator, device=device).clamp(0.02, 0.98)
            if arm_model_config.get("architecture") == "reaction_program_graph_transformer":
                from forge.model.reaction_program_transformer import (
                    balanced_pcgrad_backward,
                    per_program_transformer_losses,
                )

                objective = arm_model_config["semantic_objective"]
                topology_conditioned_weight = float(
                    objective.get("topology_conditioned_chemistry_weight", 0.0)
                )
                if (
                    topology_conditioned_weight > 0.0
                    and optimization["topology_conditioned_forward_mode"] == "paired_batch"
                ):
                    predictions, noisy, topology_conditioned_predictions = (
                        synthesis_program_paired_topology_forward(
                            model, clean, node_p0, bond_p0, t, generator
                        )
                    )
                else:
                    predictions, noisy = synthesis_program_forward(
                        model, clean, node_p0, bond_p0, t, generator
                    )
                    topology_conditioned_predictions = (
                        synthesis_program_topology_conditioned_forward(model, clean, noisy, t)
                        if topology_conditioned_weight > 0.0
                        else None
                    )
                family_losses, metrics = per_program_transformer_losses(
                    predictions,
                    clean,
                    role_weight=float(objective["role_consistency_weight"]),
                    core_weight=float(objective["core_consistency_weight"]),
                    repeat_consistency_weight=float(
                        objective.get("repeat_consistency_weight", 0.0)
                    ),
                    offspring_weight=float(objective.get("offspring_weight", 0.0)),
                    junction_consistency_weight=float(
                        objective.get("junction_consistency_weight", 0.0)
                    ),
                    chemistry_loss_balancing=str(
                        objective.get("chemistry_loss_balancing", "pooled")
                    ),
                    topology_conditioned_predictions=topology_conditioned_predictions,
                    topology_conditioned_chemistry_weight=topology_conditioned_weight,
                    program_states=active_program_states,
                    materialize_metrics=False,
                )
                loss = torch.stack(list(family_losses.values())).mean()
                balancing_method = arm_model_config.get("gradient_balancing", {}).get(
                    "method", "equal_family_mass_deterministic_pcgrad"
                )
                if balancing_method not in {
                    "equal_family_mass_deterministic_pcgrad",
                    "equal_family_mass_mean",
                }:
                    raise SynthesisProgramProductionTrainingError(
                        f"unsupported gradient balancing method: {balancing_method!r}"
                    )
                if len(family_losses) == 1 or balancing_method == "equal_family_mass_mean":
                    (loss / accumulation).backward()
                else:
                    diagnostic = balanced_pcgrad_backward(
                        family_losses,
                        model,
                        scale=1.0 / accumulation,
                        materialize_diagnostics=False,
                        backend=str(optimization["pcgrad_backend"]),
                    )
                    balancing_diagnostics["micro_batches"] += 1
                    step_conflicts += diagnostic["projected_conflicts"]
            else:
                loss, metrics, noisy = synthesis_program_forward_loss(
                    model, clean, node_p0, bond_p0, t, generator
                )
                (loss / accumulation).backward()
            step_fixed_state_failures += (
                ~synthesis_program_fixed_state_exact_tensor(noisy, clean)
            ).to(torch.int64)
            examples_seen += micro_batch
            for key, value in metrics.items():
                detached = (
                    value.detach()
                    if isinstance(value, torch.Tensor)
                    else torch.as_tensor(value, device=device)
                )
                step_metrics[key] = step_metrics.get(key, 0.0) + detached / accumulation
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(runtime["gradient_clip_norm"])
        )
        # Launch parameter updates before the metrics cross the accelerator boundary.  The
        # diagnostics are detached from model state, so this preserves the exact update while
        # avoiding a host synchronization bubble between backward and AdamW.
        optimizer.step()
        metric_keys = sorted(step_metrics)
        published = (
            torch.stack(
                [
                    *(step_metrics[key] for key in metric_keys),
                    gradient_norm.detach(),
                    step_conflicts.to(torch.float32),
                    step_fixed_state_failures.to(torch.float32),
                ]
            )
            .cpu()
            .tolist()
        )
        step_metric_values = published[: len(metric_keys)]
        gradient_norm_value, conflict_value, fixed_failure_value = published[-3:]
        balancing_diagnostics["projected_conflicts"] += int(conflict_value)
        fixed_state_failures += int(fixed_failure_value)
        completed_steps = optimizer_step
        losses.append(
            {
                "step": optimizer_step,
                **dict(zip(metric_keys, step_metric_values, strict=True)),
                "gradient_norm": gradient_norm_value,
            }
        )
        if optimizer_step in checkpoint_steps:
            snapshot_path = arm_dir / f"checkpoint_step_{optimizer_step}.pt"
            package = {
                "schema_version": CHECKPOINT_SCHEMA,
                "arm_id": arm_id,
                "seed": seed,
                "step": optimizer_step,
                "model_config": arm_model_config,
                "model_state": model.state_dict(),
                "model_state_sha256": _model_state_sha256(model),
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
                "node_marginal": torch.as_tensor(node_marginal, dtype=torch.float32),
                "bond_marginal": torch.as_tensor(bond_marginal, dtype=torch.float32),
                "conditioning": conditioning,
                "program_id_mapping": mapping,
                "batch_program_counts": (
                    dict(arm["batch_program_counts"]) if "batch_program_counts" in arm else None
                ),
                "training_optimization": optimization,
                "design_sha256": str(sha256_file(design_path)),
                "cache_sha256": str(sha256_file(cache_path)),
            }
            atomic_torch_save(snapshot_path, package)
            snapshot = {
                "step": optimizer_step,
                "filename": snapshot_path.name,
                "sha256": str(sha256_file(snapshot_path)),
                "model_state_sha256": package["model_state_sha256"],
            }
            snapshots = [row for row in snapshots if int(row["step"]) != optimizer_step]
            snapshots.append(snapshot)
            snapshots.sort(key=lambda row: int(row["step"]))
        if optimizer_step % restart_interval == 0 or optimizer_step in checkpoint_steps:
            atomic_torch_save(restart_path, restart_payload())
            write_json(
                arm_dir / "progress.json",
                {
                    "schema_version": "forge.synthesis_program_production_progress.v1",
                    "arm_id": arm_id,
                    "seed": seed,
                    "completed_steps": completed_steps,
                    "target_steps": target_steps,
                    "examples_seen": examples_seen,
                    "examples_seen_by_program": examples_seen_by_program,
                    "fixed_state_failures": fixed_state_failures,
                    "latest_loss": losses[-1],
                    "checkpoints": snapshots,
                },
            )
    if [int(row["step"]) for row in snapshots] != sorted(checkpoint_steps):
        raise SynthesisProgramProductionTrainingError(
            f"arm {arm_id} did not publish every frozen checkpoint"
        )
    observed_mass = {
        program_id: float(measure[cache.indices(program_id=program_id, fold="train")].sum())
        for program_id in cache.vocabulary.program_states[1:]
    }
    result = {
        "schema_version": "forge.synthesis_program_production_arm_result.v1",
        "status": "complete",
        "arm_id": arm_id,
        "seed": seed,
        "conditioning": conditioning,
        "program_id_mapping": mapping,
        "program_mass": observed_mass,
        "batch_program_counts": (
            {
                program: count
                for program, count in zip(
                    stratified_sampler.program_ids,
                    stratified_sampler.batch_counts(micro_batch),
                    strict=True,
                )
            }
            if stratified_sampler is not None
            else None
        ),
        "optimizer_steps": completed_steps,
        "examples_seen": examples_seen,
        "examples_seen_by_program": examples_seen_by_program,
        "parameter_count": parameter_count,
        "effective_batch_size": micro_batch * accumulation,
        "training_optimization": optimization,
        "fixed_state_failures": fixed_state_failures,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "checkpoints": snapshots,
        "model_state_excludes_component_identity": True,
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
        "gradient_balancing": {
            "method": (
                arm_model_config.get("gradient_balancing", {}).get(
                    "method", "equal_family_mass_deterministic_pcgrad"
                )
                if arm_model_config.get("architecture") == "reaction_program_graph_transformer"
                else "none"
            ),
            **balancing_diagnostics,
        },
    }
    write_json(completed_receipt, result)
    return result


def run_synthesis_program_production_training(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    output_dir: Path,
    *,
    work_dir: Path,
    profile: str,
    replicate: int,
    allocated_device: str,
    resume: bool,
) -> dict[str, Any]:
    """Train every matched arm for one frozen replicate and publish restartable checkpoints."""

    if torch is None:
        raise SynthesisProgramProductionTrainingError("production training requires torch")
    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionTrainingError,
        label="synthesis-program production training config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SynthesisProgramProductionTrainingError("unsupported production training config")
    if config.get("authorization", {}).get("authorized") is not True:
        raise SynthesisProgramProductionTrainingError("production launch is not authorized")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != {
        "production_design",
        "production_cache",
    }:
        raise SynthesisProgramProductionTrainingError("production training inputs changed")
    design_path = resolve_pin(raw_inputs["production_design"], repo, label="production_design")
    resolved_cache = resolve_pin(raw_inputs["production_cache"], repo, label="production_cache")
    if resolved_cache.resolve() != cache_path.resolve():
        raise SynthesisProgramProductionTrainingError(
            "stage cache path differs from the pinned production cache"
        )
    design = read_json_object(
        design_path,
        error=SynthesisProgramProductionTrainingError,
        label="frozen production design",
    )
    if design.get("schema_version") != "forge.synthesis_program_production_design_config.v1":
        raise SynthesisProgramProductionTrainingError("production design schema changed")
    seeds = [int(value) for value in design["training"]["replicate_seeds"]]
    if replicate < 0 or replicate >= len(seeds):
        raise SynthesisProgramProductionTrainingError("replicate lies outside the frozen seeds")
    seed = seeds[replicate]
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SynthesisProgramProductionTrainingError("CUDA requested but unavailable")
    runtime, model_config = _validate_runtime(
        config=config, design=design, profile=profile, device=device
    )
    try:
        cache = SynthesisProgramProductionCache(cache_path)
    except SynthesisProgramProductionCacheError as error:
        raise SynthesisProgramProductionTrainingError(str(error)) from error
    try:
        if cache.fold_counts() != {
            program: {fold: int(value) for fold, value in contract["expected_fold_counts"].items()}
            for program, contract in design["programs"].items()
        }:
            raise SynthesisProgramProductionTrainingError(
                "cache fold counts differ from the frozen design"
            )
        shared_mass = design["training"]["arms"]["shared_three_program_conditioned"]["program_mass"]
        shared_measure = cache.training_measure(shared_mass)
        node_marginal, bond_marginal = cache.source_marginals(
            shared_measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=int(model_config["bond_classes"]),
            probability_floor=float(model_config["source_probability_floor"]),
        )
        arm_results: dict[str, Any] = {}
        for arm_id, arm in design["training"]["arms"].items():
            arm_results[arm_id] = _train_arm(
                arm_id=arm_id,
                arm=dict(arm),
                seed=seed,
                cache=cache,
                design_path=design_path,
                cache_path=cache_path,
                config_path=config_path,
                runtime=runtime,
                model_config=model_config,
                device=device,
                work_dir=work_dir,
                resume=resume,
                node_marginal=node_marginal,
                bond_marginal=bond_marginal,
            )
    finally:
        cache.close()
    snapshot_paths = sorted(work_dir.glob("*/checkpoint_step_*.pt"))
    archive_path = output_dir / "checkpoints.tar"
    _deterministic_tar(snapshot_paths, archive_path, base=work_dir)
    progress = {
        "schema_version": "forge.synthesis_program_production_progress.v1",
        "status": "complete",
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "arms": {
            arm_id: {
                "optimizer_steps": int(value["optimizer_steps"]),
                "examples_seen": int(value["examples_seen"]),
                "fixed_state_failures": int(value["fixed_state_failures"]),
            }
            for arm_id, value in arm_results.items()
        },
    }
    write_json(output_dir / "progress.json", progress)
    gates = {
        "all_four_arms_complete": set(arm_results) == set(design["training"]["arms"]),
        "matched_optimizer_steps": {int(value["optimizer_steps"]) for value in arm_results.values()}
        == {int(runtime["optimizer_steps"])},
        "matched_effective_batch_size": {
            int(value["effective_batch_size"]) for value in arm_results.values()
        }
        == {int(runtime["effective_batch_size"])},
        "all_frozen_checkpoints_present": all(
            [int(row["step"]) for row in value["checkpoints"]]
            == [int(step) for step in runtime["checkpoint_steps"]]
            for value in arm_results.values()
        ),
        "fixed_state_failures_zero": all(
            int(value["fixed_state_failures"]) == 0 for value in arm_results.values()
        ),
        "shared_source_marginals_for_every_arm": True,
        "route_or_oracle_calls_zero": True,
        "candidate_selection_absent": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "run_kind": "production" if profile == "full" else "smoke",
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "authorization": dict(config["authorization"]),
        "config": pin_record(config_path, repo),
        "design": pin_record(design_path, repo),
        "cache": artifact_record(cache_path),
        "model": model_config,
        "runtime": runtime,
        "source_marginals": {
            "policy": design["training"]["source_marginal_policy"],
            "node": node_marginal.tolist(),
            "bond": bond_marginal.tolist(),
        },
        "arms": arm_results,
        "checkpoint_archive": artifact_record(archive_path),
        "gates": gates,
        "nonclaims": [
            "Training completion is not evidence of generation quality or synthesis success.",
            "No arm is a candidate source and no biological or synthesis guidance is used.",
            "Auxiliary-family replay remains transform consistency, not route certification.",
        ],
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise SynthesisProgramProductionTrainingError(f"matched training gates failed: {gates}")
    return result


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SynthesisProgramProductionTrainingError",
    "run_synthesis_program_production_training",
]
