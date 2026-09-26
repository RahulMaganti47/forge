"""Bounded exact-H100 profile for the frozen mixed-program Transformer training step."""

from __future__ import annotations

import math
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.phase1.multireaction.production_training import (
    _compile_stratified_program_sampler,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_fixed_state_exact_tensor,
    synthesis_program_forward,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


CONFIG_SCHEMA = "forge.synthesis_program_h100_training_profile_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_h100_training_profile_result.v1"


class H100TrainingProfileError(ValueError):
    """The profiling request or runtime violates the frozen training contract."""


def _set_determinism(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _parameter_vector(model: Any) -> Any:
    return torch.cat(
        [parameter.detach().float().cpu().reshape(-1) for parameter in model.parameters()]
    )


def _fixed_program_draws(
    sampler: Any,
    *,
    seed: int,
    optimizer_steps: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> tuple[tuple[np.ndarray, ...], ...]:
    """Freeze production-shaped source-balanced draws before timing any candidate."""

    rng = np.random.default_rng(seed)
    return tuple(
        tuple(
            sampler.sample(micro_batch_size, rng)
            for _ in range(gradient_accumulation_steps)
        )
        for _ in range(optimizer_steps)
    )


def _repartition_draws(
    draws: Sequence[Sequence[np.ndarray]], *, micro_batch_size: int
) -> tuple[tuple[np.ndarray, ...], ...]:
    output: list[tuple[np.ndarray, ...]] = []
    for step in draws:
        flat = np.concatenate(step)
        if len(flat) % micro_batch_size:
            raise H100TrainingProfileError("candidate microbatch does not divide effective batch")
        output.append(
            tuple(
                flat[offset : offset + micro_batch_size]
                for offset in range(0, len(flat), micro_batch_size)
            )
        )
    return tuple(output)


def _candidate_runtime(
    baseline_runtime: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    runtime = dict(baseline_runtime)
    runtime["micro_batch_size"] = int(
        candidate.get("micro_batch_size", runtime["micro_batch_size"])
    )
    runtime["gradient_accumulation_steps"] = int(
        candidate.get(
            "gradient_accumulation_steps", runtime["gradient_accumulation_steps"]
        )
    )
    if (
        runtime["micro_batch_size"] * runtime["gradient_accumulation_steps"]
        != baseline_runtime["effective_batch_size"]
    ):
        raise H100TrainingProfileError("candidate changed the frozen effective batch size")
    return runtime


def _run_candidate(
    *,
    candidate: Mapping[str, Any],
    seed: int,
    cache: SynthesisProgramProductionCache,
    arm: Mapping[str, Any],
    model_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    draws: Sequence[Sequence[np.ndarray]],
    warmup_steps: int,
    measured_steps: int,
    active_program_states: tuple[int, ...] | None,
) -> tuple[dict[str, Any], Any, Any]:
    """Run one exact optimizer-step candidate and retain vectors only for adjudication."""

    device = torch.device("cuda")
    _set_determinism(seed, int(runtime["cpu_threads"]))
    allow_tf32 = bool(candidate.get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=model_config,
        device=device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    initial = _parameter_vector(model)
    executable = model
    compile_seconds = 0.0
    if bool(candidate.get("compile", False)):
        compile_started = time.perf_counter()
        executable = torch.compile(
            model,
            backend=str(candidate.get("compile_backend", "inductor")),
            mode=str(candidate.get("compile_mode", "default")),
            fullgraph=False,
            dynamic=False,
        )
        compile_seconds = time.perf_counter() - compile_started
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=device)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    autocast_enabled = str(candidate["precision"]) == "bfloat16"
    if str(candidate["precision"]) not in {"float32", "bfloat16"}:
        raise H100TrainingProfileError("unsupported profiling precision")
    if len(draws) != warmup_steps + measured_steps:
        raise H100TrainingProfileError("fixed draw count differs from profiling schedule")

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    measured_wall_seconds: list[float] = []
    measured_cuda_seconds: list[float] = []
    measured_losses: list[float] = []
    measured_gradient_norms: list[float] = []
    fixed_state_failures = 0
    observed_nodes = 0
    observed_closures = 0
    projected_conflicts = 0
    for optimizer_step, step_draws in enumerate(draws):
        wall_started = time.perf_counter()
        cuda_started = torch.cuda.Event(enable_timing=True)
        cuda_finished = torch.cuda.Event(enable_timing=True)
        cuda_started.record()
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[Any] = []
        step_failures = torch.zeros((), dtype=torch.int64, device=device)
        step_conflicts = torch.zeros((), dtype=torch.int64, device=device)
        for selected in step_draws:
            records = cache.records(selected)
            observed_nodes = max(observed_nodes, *(record.node_count for record in records))
            observed_closures = max(
                observed_closures, *(record.graph.closure_count for record in records)
            )
            clean = move_tensors(
                collate_synthesis_program_training_batch(
                    records,
                    maximum_closures=int(model_config["maximum_closures"]),
                    conditioning=str(arm["conditioning"]),
                    vocabulary=cache.vocabulary,
                    program_id_mapping=arm.get("program_id_mapping"),
                ),
                device,
            )
            t = torch.rand(len(selected), generator=generator, device=device).clamp(0.02, 0.98)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
            ):
                predictions, noisy = synthesis_program_forward(
                    executable, clean, node_p0, bond_p0, t, generator
                )
                from forge.model.reaction_program_transformer import (
                    balanced_pcgrad_backward,
                    per_program_transformer_losses,
                )

                objective = model_config["semantic_objective"]
                family_losses, _ = per_program_transformer_losses(
                    predictions,
                    clean,
                    role_weight=float(objective["role_consistency_weight"]),
                    core_weight=float(objective["core_consistency_weight"]),
                    repeat_consistency_weight=float(
                        objective.get("repeat_consistency_weight", 0.0)
                    ),
                    program_states=active_program_states,
                    materialize_metrics=False,
                )
                loss = torch.stack(list(family_losses.values())).mean()
            if not torch.isfinite(loss):
                raise H100TrainingProfileError(
                    f"non-finite loss for profiling candidate {candidate['id']}"
                )
            if len(family_losses) == 1:
                (loss / len(step_draws)).backward()
            else:
                diagnostic = balanced_pcgrad_backward(
                    family_losses,
                    model,
                    scale=1.0 / len(step_draws),
                    materialize_diagnostics=False,
                )
                step_conflicts += diagnostic["projected_conflicts"]
            step_losses.append(loss.detach())
            step_failures += (~synthesis_program_fixed_state_exact_tensor(noisy, clean)).to(
                torch.int64
            )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(runtime["gradient_clip_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise H100TrainingProfileError(
                f"non-finite gradient for profiling candidate {candidate['id']}"
            )
        optimizer.step()
        step_loss = torch.stack(step_losses).mean()
        cuda_finished.record()
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - wall_started
        cuda_seconds = cuda_started.elapsed_time(cuda_finished) / 1000.0
        published = torch.stack(
            (
                step_loss,
                gradient_norm.detach(),
                step_failures.to(torch.float32),
                step_conflicts.to(torch.float32),
            )
        ).cpu()
        fixed_state_failures += int(published[2])
        projected_conflicts += int(published[3])
        if optimizer_step >= warmup_steps:
            measured_wall_seconds.append(wall_seconds)
            measured_cuda_seconds.append(cuda_seconds)
            measured_losses.append(float(published[0]))
            measured_gradient_norms.append(float(published[1]))

    final = _parameter_vector(model)
    examples = int(runtime["effective_batch_size"]) * measured_steps
    wall_total = sum(measured_wall_seconds)
    cuda_total = sum(measured_cuda_seconds)
    result = {
        "id": str(candidate["id"]),
        "precision": str(candidate["precision"]),
        "allow_tf32": allow_tf32,
        "float32_matmul_precision": "high" if allow_tf32 else "highest",
        "compile": bool(candidate.get("compile", False)),
        "compile_backend": candidate.get("compile_backend"),
        "compile_mode": candidate.get("compile_mode"),
        "compile_wrapper_seconds": compile_seconds,
        "micro_batch_size": int(runtime["micro_batch_size"]),
        "gradient_accumulation_steps": int(runtime["gradient_accumulation_steps"]),
        "effective_batch_size": int(runtime["effective_batch_size"]),
        "warmup_optimizer_steps": warmup_steps,
        "measured_optimizer_steps": measured_steps,
        "step_wall_seconds": measured_wall_seconds,
        "step_cuda_seconds": measured_cuda_seconds,
        "median_wall_seconds_per_optimizer_step": statistics.median(measured_wall_seconds),
        "median_cuda_seconds_per_optimizer_step": statistics.median(measured_cuda_seconds),
        "examples_per_second_wall": examples / wall_total,
        "examples_per_second_cuda": examples / cuda_total,
        "losses": measured_losses,
        "gradient_norms": measured_gradient_norms,
        "fixed_state_failures": fixed_state_failures,
        "projected_conflicts": projected_conflicts,
        "maximum_nodes_observed": observed_nodes,
        "maximum_closures_observed": observed_closures,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "model_state_sha256": _model_state_sha256(model),
        "parameter_count": parameter_count,
    }
    del optimizer, executable, model
    torch.cuda.empty_cache()
    return result, initial, final


def _equivalence(
    reference_result: Mapping[str, Any],
    reference_initial: Any,
    reference_final: Any,
    candidate_result: Mapping[str, Any],
    candidate_initial: Any,
    candidate_final: Any,
) -> dict[str, Any]:
    reference_losses = torch.as_tensor(reference_result["losses"], dtype=torch.float64)
    candidate_losses = torch.as_tensor(candidate_result["losses"], dtype=torch.float64)
    loss_delta = (candidate_losses - reference_losses).abs()
    reference_update = reference_final - reference_initial
    candidate_update = candidate_final - candidate_initial
    update_delta = candidate_update - reference_update
    reference_norm = float(torch.linalg.vector_norm(reference_update))
    candidate_norm = float(torch.linalg.vector_norm(candidate_update))
    delta_norm = float(torch.linalg.vector_norm(update_delta))
    denominator = max(reference_norm, torch.finfo(torch.float32).eps)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            reference_update[None], candidate_update[None], dim=1, eps=1e-12
        )[0]
    )
    return {
        "initial_parameters_exact": bool(torch.equal(reference_initial, candidate_initial)),
        "loss_max_absolute_difference": float(loss_delta.max()),
        "loss_max_relative_difference": float(
            (loss_delta / reference_losses.abs().clamp(min=1e-12)).max()
        ),
        "parameter_update_reference_l2": reference_norm,
        "parameter_update_candidate_l2": candidate_norm,
        "parameter_update_delta_l2": delta_norm,
        "parameter_update_relative_l2_difference": delta_norm / denominator,
        "parameter_update_max_absolute_difference": float(update_delta.abs().max()),
        "parameter_update_cosine_similarity": cosine,
        "final_parameters_exact": bool(torch.equal(reference_final, candidate_final)),
    }


def _passes_tolerance(values: Mapping[str, Any], tolerance: Mapping[str, Any]) -> bool:
    return bool(
        values["initial_parameters_exact"]
        and float(values["loss_max_absolute_difference"])
        <= float(tolerance["loss_max_absolute_difference"])
        and float(values["loss_max_relative_difference"])
        <= float(tolerance["loss_max_relative_difference"])
        and float(values["parameter_update_relative_l2_difference"])
        <= float(tolerance["parameter_update_relative_l2_difference"])
        and float(values["parameter_update_max_absolute_difference"])
        <= float(tolerance["parameter_update_max_absolute_difference"])
        and float(values["parameter_update_cosine_similarity"])
        >= float(tolerance["parameter_update_cosine_similarity"])
    )


def run_h100_training_profile(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    output_path: Path,
    *,
    allocated_device: str,
    declared_gpu_type: str,
) -> dict[str, Any]:
    """Profile eager FP32 and bounded optimization candidates on one exact H100."""

    if torch is None or allocated_device != "cuda" or not torch.cuda.is_available():
        raise H100TrainingProfileError("exact-H100 profile requires allocated CUDA")
    config = read_json_object(
        config_path, error=H100TrainingProfileError, label="H100 training profile config"
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise H100TrainingProfileError("unsupported H100 training profile config")
    if config.get("authorization", {}).get("authorized") is not True:
        raise H100TrainingProfileError("H100 training profile is not authorized")
    raw_inputs = config.get("inputs")
    required_inputs = {
        "production_design",
        "production_cache",
        "production_training_config",
        "cpu_performance_result",
        "smoke_training_result",
    }
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != required_inputs:
        raise H100TrainingProfileError("H100 profile inputs changed")
    paths = {
        label: resolve_pin(pin, repo, label=label) for label, pin in raw_inputs.items()
    }
    if paths["production_cache"].resolve() != cache_path.resolve():
        raise H100TrainingProfileError("allocated production cache differs from the pinned cache")
    design = read_json_object(
        paths["production_design"], error=H100TrainingProfileError, label="production design"
    )
    training_config = read_json_object(
        paths["production_training_config"],
        error=H100TrainingProfileError,
        label="production training config",
    )
    cpu_result = read_json_object(
        paths["cpu_performance_result"],
        error=H100TrainingProfileError,
        label="CPU performance result",
    )
    smoke_result = read_json_object(
        paths["smoke_training_result"],
        error=H100TrainingProfileError,
        label="training smoke result",
    )
    runtime = dict(training_config["full"]["training"])
    frozen = design["training"]
    exact_runtime = {
        "optimizer_steps": int(frozen["optimizer_steps"]),
        "micro_batch_size": int(frozen["micro_batch_size"]),
        "gradient_accumulation_steps": int(frozen["gradient_accumulation_steps"]),
        "effective_batch_size": int(frozen["effective_batch_size"]),
        "learning_rate": float(frozen["learning_rate"]),
        "weight_decay": float(frozen["weight_decay"]),
        "gradient_clip_norm": float(frozen["gradient_clip_norm"]),
        "checkpoint_steps": [int(value) for value in frozen["checkpoint_steps"]],
    }
    runtime_matches_design = all(runtime.get(key) == value for key, value in exact_runtime.items())
    if not runtime_matches_design:
        raise H100TrainingProfileError("profiling runtime differs from the frozen design")
    if runtime.get("precision") != "float32" or runtime.get("deterministic_algorithms") is not True:
        raise H100TrainingProfileError("baseline must remain deterministic float32")

    policy = config["profile"]
    warmup_steps = int(policy["warmup_optimizer_steps"])
    measured_steps = int(policy["measured_optimizer_steps"])
    properties = torch.cuda.get_device_properties(0)
    memory_gib = int(properties.total_memory) / float(1024**3)
    exact_h100 = bool(
        declared_gpu_type == "H100!"
        and "H100" in str(properties.name).upper()
        and float(policy["minimum_memory_gib"])
        <= memory_gib
        <= float(policy["maximum_memory_gib"])
    )
    if not exact_h100:
        raise H100TrainingProfileError(
            f"runtime is not the declared exact H100: {properties.name}, {memory_gib:.2f} GiB"
        )

    model_config = dict(design["model"])
    model_config.update(dict(policy.get("model_overrides", {})))
    arm = {
        **design["training"]["arms"][str(policy["arm_id"])],
        **dict(policy.get("arm_overrides", {})),
    }
    seed = int(policy["seed"])
    cache = SynthesisProgramProductionCache(cache_path)
    try:
        measure = cache.training_measure(arm["program_mass"])
        sampler = _compile_stratified_program_sampler(cache, measure)
        node_marginal, bond_marginal = cache.source_marginals(
            measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=int(model_config["bond_classes"]),
            probability_floor=float(model_config["source_probability_floor"]),
        )
        baseline_draws = _fixed_program_draws(
            sampler,
            seed=seed + 2,
            optimizer_steps=warmup_steps + measured_steps,
            micro_batch_size=int(runtime["micro_batch_size"]),
            gradient_accumulation_steps=int(runtime["gradient_accumulation_steps"]),
        )
        candidates = list(policy["candidates"])
        baseline_policy = candidates[0]
        if baseline_policy.get("id") != "fp32_eager":
            raise H100TrainingProfileError("first profiling candidate must be fp32_eager")
        baseline, baseline_initial, baseline_final = _run_candidate(
            candidate=baseline_policy,
            seed=seed,
            cache=cache,
            arm=arm,
            model_config=model_config,
            runtime=runtime,
            node_marginal=node_marginal,
            bond_marginal=bond_marginal,
            draws=baseline_draws,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            active_program_states=sampler.program_states,
        )
        repeat, repeat_initial, repeat_final = _run_candidate(
            candidate=baseline_policy,
            seed=seed,
            cache=cache,
            arm=arm,
            model_config=model_config,
            runtime=runtime,
            node_marginal=node_marginal,
            bond_marginal=bond_marginal,
            draws=baseline_draws,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            active_program_states=sampler.program_states,
        )
        repeat_equivalence = _equivalence(
            baseline,
            baseline_initial,
            baseline_final,
            repeat,
            repeat_initial,
            repeat_final,
        )

        assessed: list[dict[str, Any]] = []
        retained_vectors: dict[str, tuple[Any, Any]] = {}
        for candidate in candidates[1:]:
            candidate_runtime = _candidate_runtime(runtime, candidate)
            candidate_draws = _repartition_draws(
                baseline_draws, micro_batch_size=int(candidate_runtime["micro_batch_size"])
            )
            try:
                measurement, initial, final = _run_candidate(
                    candidate=candidate,
                    seed=seed,
                    cache=cache,
                    arm=arm,
                    model_config=model_config,
                    runtime=candidate_runtime,
                    node_marginal=node_marginal,
                    bond_marginal=bond_marginal,
                    draws=candidate_draws,
                    warmup_steps=warmup_steps,
                    measured_steps=measured_steps,
                    active_program_states=sampler.program_states,
                )
                equivalence = _equivalence(
                    baseline,
                    baseline_initial,
                    baseline_final,
                    measurement,
                    initial,
                    final,
                )
                tolerance_name = str(candidate["equivalence_tolerance"])
                tolerance = policy["equivalence_tolerances"][tolerance_name]
                tolerance_passed = _passes_tolerance(equivalence, tolerance)
                partition_preserved = (
                    int(candidate_runtime["micro_batch_size"])
                    == int(runtime["micro_batch_size"])
                    and int(candidate_runtime["gradient_accumulation_steps"])
                    == int(runtime["gradient_accumulation_steps"])
                )
                speedup = float(baseline["median_wall_seconds_per_optimizer_step"]) / float(
                    measurement["median_wall_seconds_per_optimizer_step"]
                )
                candidate_eligible = bool(
                    tolerance_passed
                    and partition_preserved
                    and int(measurement["fixed_state_failures"]) == 0
                    and all(math.isfinite(value) for value in measurement["losses"])
                )
                measurement.update(
                    {
                        "status": "completed",
                        "speedup_over_fp32_eager": speedup,
                        "equivalence": equivalence,
                        "equivalence_tolerance": tolerance_name,
                        "equivalence_passed": tolerance_passed,
                        "pcgrad_microbatch_partition_preserved": partition_preserved,
                        "eligible_for_training_contract": candidate_eligible,
                        "ineligibility_reason": (
                            None
                            if candidate_eligible
                            else (
                                "PCGrad projection partition changes the scientific optimizer"
                                if not partition_preserved
                                else "prespecified numerical-equivalence gate failed"
                            )
                        ),
                    }
                )
                retained_vectors[str(candidate["id"])] = (initial, final)
            except Exception as error:  # candidate failure is an adjudicated negative result
                torch.cuda.empty_cache()
                measurement = {
                    "id": str(candidate["id"]),
                    "status": "failed",
                    "eligible_for_training_contract": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            assessed.append(measurement)

        eligible_rows = sorted(
            (
                row
                for row in assessed
                if row.get("eligible_for_training_contract") is True
                and float(row["speedup_over_fp32_eager"])
                >= float(policy["minimum_promotion_speedup"])
            ),
            key=lambda row: float(row["median_wall_seconds_per_optimizer_step"]),
        )
        selected = "fp32_eager"
        selection_reason = "no candidate cleared speed, equivalence, and optimizer-identity gates"
        selected_repeat: dict[str, Any] | None = None
        if eligible_rows:
            selected_row = eligible_rows[0]
            selected_id = str(selected_row["id"])
            selected_policy = next(row for row in candidates if row["id"] == selected_id)
            selected_runtime = _candidate_runtime(runtime, selected_policy)
            selected_draws = _repartition_draws(
                baseline_draws, micro_batch_size=int(selected_runtime["micro_batch_size"])
            )
            repeated, repeated_initial, repeated_final = _run_candidate(
                candidate=selected_policy,
                seed=seed,
                cache=cache,
                arm=arm,
                model_config=model_config,
                runtime=selected_runtime,
                node_marginal=node_marginal,
                bond_marginal=bond_marginal,
                draws=selected_draws,
                warmup_steps=warmup_steps,
                measured_steps=measured_steps,
                active_program_states=sampler.program_states,
            )
            original_initial, original_final = retained_vectors[selected_id]
            deterministic = _equivalence(
                selected_row,
                original_initial,
                original_final,
                repeated,
                repeated_initial,
                repeated_final,
            )
            deterministic_exact = bool(
                deterministic["final_parameters_exact"]
                and deterministic["loss_max_absolute_difference"] == 0.0
            )
            selected_repeat = {
                "candidate_id": selected_id,
                "measurement": repeated,
                "equivalence": deterministic,
                "deterministic_exact": deterministic_exact,
            }
            if deterministic_exact:
                selected = selected_id
                selection_reason = (
                    "fastest candidate clearing prespecified equivalence, PCGrad identity, "
                    "minimum speedup, and exact deterministic replay gates"
                )
            else:
                selected_row["eligible_for_training_contract"] = False
                selected_row["ineligibility_reason"] = "candidate deterministic replay was not exact"

        node_counts = np.diff(cache.arrays["node_offsets"])
        maximum_index = int(np.argmax(node_counts))
        stress_draws = tuple(
            (
                np.full(int(runtime["micro_batch_size"]), maximum_index, dtype=np.int64),
            )
            * int(runtime["gradient_accumulation_steps"])
        )
        stress, _, _ = _run_candidate(
            candidate=baseline_policy,
            seed=seed + 1000,
            cache=cache,
            arm=arm,
            model_config=model_config,
            runtime=runtime,
            node_marginal=node_marginal,
            bond_marginal=bond_marginal,
            draws=(stress_draws,),
            warmup_steps=0,
            measured_steps=1,
            active_program_states=None,
        )
    finally:
        cache.close()

    peak_reserved = max(
        [int(baseline["peak_reserved_bytes"]), int(repeat["peak_reserved_bytes"])]
        + [int(row.get("peak_reserved_bytes", 0)) for row in assessed]
        + [int(stress["peak_reserved_bytes"])]
    )
    gates = {
        "authorized_bounded_profile": True,
        "exact_h100_runtime": exact_h100,
        "runtime_matches_frozen_design": runtime_matches_design,
        "parameter_count_matches_profile": int(baseline["parameter_count"])
        == int(policy.get("expected_parameter_count", baseline["parameter_count"])),
        "deterministic_float32_baseline": runtime["precision"] == "float32"
        and runtime["deterministic_algorithms"] is True,
        "cpu_optimization_receipt_passed": cpu_result.get("status") == "pass",
        "authenticated_smoke_passed": smoke_result.get("status") == "pass",
        "three_source_balanced_programs_active": len(sampler.program_states) == 3,
        "baseline_losses_finite": all(math.isfinite(value) for value in baseline["losses"]),
        "baseline_fixed_states_exact": int(baseline["fixed_state_failures"]) == 0,
        "baseline_deterministic_replay_exact": bool(
            repeat_equivalence["final_parameters_exact"]
            and repeat_equivalence["loss_max_absolute_difference"] == 0.0
        ),
        "full_194_atom_support_present": int(np.max(node_counts))
        == int(model_config["maximum_heavy_atoms"]),
        "full_194_atom_support_exercised": int(stress["maximum_nodes_observed"])
        == int(model_config["maximum_heavy_atoms"]),
        "maximum_support_fixed_states_exact": int(stress["fixed_state_failures"]) == 0,
        "gpu_memory_headroom_at_least_ten_percent": peak_reserved
        <= int(properties.total_memory * 0.9),
        "all_candidates_adjudicated": len(assessed) == len(candidates) - 1,
        "route_or_oracle_calls_zero": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "cache": artifact_record(cache_path),
        "device": {
            "declared_gpu_type": declared_gpu_type,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "bfloat16_supported": bool(torch.cuda.is_bf16_supported()),
        },
        "scientific_contract": {
            "precision": "float32",
            "deterministic_algorithms": True,
            "maximum_heavy_atoms": int(model_config["maximum_heavy_atoms"]),
            "source_balancing": "stratified_equal_reaction_family_mass",
            "gradient_balancing": model_config["gradient_balancing"],
            "micro_batch_size": int(runtime["micro_batch_size"]),
            "gradient_accumulation_steps": int(runtime["gradient_accumulation_steps"]),
            "effective_batch_size": int(runtime["effective_batch_size"]),
        },
        "math_mode": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "baseline": baseline,
        "baseline_repeat": repeat,
        "baseline_repeat_equivalence": repeat_equivalence,
        "candidates": assessed,
        "selected_candidate": selected,
        "selection_reason": selection_reason,
        "selected_candidate_repeat": selected_repeat,
        "maximum_support_stress": stress,
        "peak_reserved_bytes": peak_reserved,
        "gates": gates,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "This is an execution profile, not a model-quality result or training run.",
            "No checkpoint, molecule, route, or biological candidate is produced or selected.",
            "A profiled optimization is not part of the frozen production contract until a "
            "separate versioned training config adopts it.",
        ],
    }
    write_json(output_path, result)
    if result["status"] != "pass":
        raise H100TrainingProfileError(f"exact-H100 profiling gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "H100TrainingProfileError",
    "RESULT_SCHEMA",
    "_candidate_runtime",
    "_equivalence",
    "_passes_tolerance",
    "_repartition_draws",
    "run_h100_training_profile",
]
