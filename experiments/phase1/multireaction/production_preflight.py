"""Fail-closed accelerator qualification for matched production training."""

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
    _validate_runtime,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutError,
    SynthesisProgramLayoutPrior,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_fixed_state_exact_tensor,
    synthesis_program_forward,
    synthesis_program_forward_loss,
)

from .production_randomness import production_seed

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

CONFIG_SCHEMA = "forge.synthesis_program_production_accelerator_benchmark_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_production_accelerator_benchmark_result.v1"


class SynthesisProgramProductionPreflightError(ValueError):
    """The production request is not safe to launch under the frozen contract."""


def _validate_molecule_rendering() -> bool:
    """Exercise the report-rendering dependency that production evaluation imports."""

    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        molecule = Chem.MolFromSmiles("CC")
        if molecule is None:
            raise RuntimeError("RDKit could not construct the rendering probe")
        drawer = rdMolDraw2D.MolDraw2DSVG(80, 60)
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        drawing = drawer.GetDrawingText()
    except Exception as error:
        raise SynthesisProgramProductionPreflightError(
            "production molecule-report rendering is unavailable"
        ) from error
    if "<svg" not in drawing:
        raise SynthesisProgramProductionPreflightError(
            "production molecule-report rendering returned no SVG"
        )
    return True


def _validate_factorized_layout_schedule(
    prior: SynthesisProgramLayoutPrior,
    design: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
) -> dict[str, int]:
    """Exercise every frozen production layout cell before any production training."""

    runtime = evaluation_config.get("full")
    if not isinstance(runtime, Mapping):
        raise SynthesisProgramProductionPreflightError(
            "production evaluation has no full layout schedule"
        )
    checkpoints = [int(value) for value in runtime["checkpoint_steps"]]
    final_checkpoint = checkpoints[-1]
    cells = 0
    layouts = 0
    for replicate_seed in design["training"]["replicate_seeds"]:
        for arm_id, arm in design["training"]["arms"].items():
            supported = [
                program_id for program_id, mass in arm["program_mass"].items() if float(mass) > 0.0
            ]
            for checkpoint in checkpoints:
                split_counts = [("calibration", int(runtime["calibration_samples"]))]
                if checkpoint == final_checkpoint:
                    split_counts.append(("heldout", int(runtime["heldout_samples"])))
                for split_name, count in split_counts:
                    for program_id in supported:
                        seed = production_seed(
                            int(replicate_seed),
                            arm_id,
                            checkpoint,
                            split_name,
                            program_id,
                            "layout",
                        )
                        try:
                            sampled = prior.sample(program_id, sample_count=count, seed=seed)
                        except SynthesisProgramLayoutError as error:
                            raise SynthesisProgramProductionPreflightError(
                                "production layout schedule is not closed for "
                                f"seed={replicate_seed}, arm={arm_id}, checkpoint={checkpoint}, "
                                f"split={split_name}, program={program_id}"
                            ) from error
                        if len(sampled) != count:
                            raise SynthesisProgramProductionPreflightError(
                                "production layout schedule returned the wrong sample count"
                            )
                        cells += 1
                        layouts += len(sampled)
    return {
        "replicates": len(design["training"]["replicate_seeds"]),
        "layout_cells": cells,
        "layouts": layouts,
    }


def _set_determinism(seed: int, cpu_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def _validate_static_contract(
    config: Mapping[str, Any],
    design: Mapping[str, Any],
    training_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    smoke_training: Mapping[str, Any],
    smoke_evaluation: Mapping[str, Any],
    test_baseline: Mapping[str, Any],
    *,
    expected_schema: str = CONFIG_SCHEMA,
) -> dict[str, bool]:
    if config.get("schema_version") != expected_schema:
        raise SynthesisProgramProductionPreflightError("unsupported GPU preflight config")
    if config.get("authorization", {}).get("authorized") is not True:
        raise SynthesisProgramProductionPreflightError("production GPU preflight is not authorized")
    if design.get("schema_version") != "forge.synthesis_program_production_design_config.v1":
        raise SynthesisProgramProductionPreflightError("frozen production design changed")
    if training_config.get("authorization", {}).get("authorized") is not True:
        raise SynthesisProgramProductionPreflightError("production training is not authorized")
    if (
        smoke_training.get("schema_version")
        != ("forge.synthesis_program_production_training_result.v1")
        or smoke_training.get("profile") != "smoke"
    ):
        raise SynthesisProgramProductionPreflightError("training smoke receipt changed")
    if (
        smoke_evaluation.get("schema_version")
        != ("forge.synthesis_program_production_evaluation_result.v1")
        or smoke_evaluation.get("profile") != "smoke"
    ):
        raise SynthesisProgramProductionPreflightError("evaluation smoke receipt changed")
    frozen = design["evaluation"]["native_sampling"]
    full_evaluation = evaluation_config.get("full", {})
    exact_evaluation = {
        "device": "cuda",
        "sample_steps": int(frozen["flow_steps"]),
        "calibration_samples": int(
            frozen["calibration_samples_per_supported_program_per_checkpoint_per_seed"]
        ),
        "heldout_samples": int(
            frozen["heldout_samples_per_supported_program_at_final_checkpoint_per_seed"]
        ),
        "checkpoint_steps": [int(value) for value in design["training"]["checkpoint_steps"]],
        "component_disjoint_record_limit": None,
    }
    evaluation_exact = all(
        full_evaluation.get(key) == value for key, value in exact_evaluation.items()
    )
    input_pins = config["inputs"]
    return {
        "explicit_authorization_present": True,
        "training_smoke_passed": smoke_training.get("status") == "pass"
        and all(smoke_training.get("gates", {}).values()),
        "evaluation_smoke_passed": smoke_evaluation.get("status") == "pass"
        and all(smoke_evaluation.get("gates", {}).values()),
        "smoke_uses_current_training_config": smoke_training.get("config", {}).get("sha256")
        == input_pins["production_training_config"]["sha256"],
        "smoke_uses_current_evaluation_config": smoke_evaluation.get("config", {}).get("sha256")
        == input_pins["production_evaluation_config"]["sha256"],
        "smoke_uses_current_cache": smoke_training.get("cache", {}).get("sha256")
        == input_pins["production_cache"]["sha256"]
        and smoke_evaluation.get("cache", {}).get("sha256")
        == input_pins["production_cache"]["sha256"],
        "full_evaluation_matches_frozen_budget": evaluation_exact,
        "candidate_selection_absent": smoke_evaluation.get("selection", {}).get(
            "candidate_selection"
        )
        is False,
        "route_or_oracle_calls_zero": smoke_evaluation.get("calls") == {"route": 0, "oracle": 0},
        "repository_has_no_new_test_failures": test_baseline.get("no_new_failures") is True
        and test_baseline.get("new_failures") == []
        and int(test_baseline.get("current_failure_count", -1))
        == int(test_baseline.get("baseline_failure_count", -2))
        and test_baseline.get("stale_cache_nodes") == [],
    }


def _positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SynthesisProgramProductionPreflightError(f"{label} must be a positive integer")
    return value


def _positive_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) <= 0.0:
        raise SynthesisProgramProductionPreflightError(f"{label} must be positive")
    return float(value)


def _accelerator_policy(config: Mapping[str, Any], declared_gpu_type: str) -> dict[str, Any]:
    """Resolve one immutable accelerator target without changing the scientific design."""

    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SynthesisProgramProductionPreflightError("unsupported GPU preflight config")

    benchmark = config.get("benchmark")
    if not isinstance(benchmark, Mapping):
        raise SynthesisProgramProductionPreflightError("accelerator benchmark policy is missing")
    targets = benchmark.get("targets")
    if not isinstance(targets, Mapping) or declared_gpu_type not in targets:
        allowed = sorted(targets) if isinstance(targets, Mapping) else []
        raise SynthesisProgramProductionPreflightError(
            f"GPU target {declared_gpu_type!r} is not authorized; allowed={allowed}"
        )
    target = targets[declared_gpu_type]
    if not isinstance(target, Mapping):
        raise SynthesisProgramProductionPreflightError(
            f"GPU target {declared_gpu_type!r} has no policy"
        )
    tokens = target.get("device_name_tokens")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, str) or not token for token in tokens)
    ):
        raise SynthesisProgramProductionPreflightError(
            f"GPU target {declared_gpu_type!r} has invalid device-name tokens"
        )
    minimum_memory_gib = _positive_float(
        target.get("minimum_memory_gib"),
        label=f"benchmark.targets.{declared_gpu_type}.minimum_memory_gib",
    )
    maximum_memory_gib = _positive_float(
        target.get("maximum_memory_gib"),
        label=f"benchmark.targets.{declared_gpu_type}.maximum_memory_gib",
    )
    if maximum_memory_gib <= minimum_memory_gib:
        raise SynthesisProgramProductionPreflightError(
            f"GPU target {declared_gpu_type!r} has an invalid memory interval"
        )
    pricing = benchmark.get("pricing")
    if not isinstance(pricing, Mapping):
        raise SynthesisProgramProductionPreflightError("accelerator pricing policy is missing")
    rates = pricing.get("gpu_usd_per_second")
    if not isinstance(rates, Mapping) or set(rates) != set(targets):
        raise SynthesisProgramProductionPreflightError(
            "accelerator prices must cover exactly the authorized GPU targets"
        )
    rate = _positive_float(
        rates[declared_gpu_type],
        label=f"benchmark.pricing.gpu_usd_per_second.{declared_gpu_type}",
    )
    return {
        "config_schema": CONFIG_SCHEMA,
        "result_schema": RESULT_SCHEMA,
        "declared_gpu_type": declared_gpu_type,
        "device_name_tokens": list(tokens),
        "minimum_memory_gib": minimum_memory_gib,
        "maximum_memory_gib": maximum_memory_gib,
        "warmup_optimizer_steps": _positive_int(
            benchmark.get("warmup_optimizer_steps"),
            label="benchmark.warmup_optimizer_steps",
        ),
        "measured_optimizer_steps": _positive_int(
            benchmark.get("measured_optimizer_steps"),
            label="benchmark.measured_optimizer_steps",
        ),
        "maximum_support_optimizer_steps": _positive_int(
            benchmark.get("maximum_support_optimizer_steps"),
            label="benchmark.maximum_support_optimizer_steps",
        ),
        "production_optimizer_steps_per_arm": _positive_int(
            benchmark.get("production_optimizer_steps_per_arm"),
            label="benchmark.production_optimizer_steps_per_arm",
        ),
        "production_replicates": _positive_int(
            benchmark.get("production_replicates"),
            label="benchmark.production_replicates",
        ),
        "gpu_usd_per_second": rate,
        "pricing": dict(pricing),
    }


def _runtime_matches_accelerator(properties: Any, policy: Mapping[str, Any]) -> bool:
    name = str(properties.name).upper()
    tokens_match = all(str(token).upper() in name for token in policy["device_name_tokens"])
    memory_gib = int(properties.total_memory) / float(1024**3)
    maximum = policy["maximum_memory_gib"]
    return bool(
        tokens_match
        and memory_gib >= float(policy["minimum_memory_gib"])
        and (maximum is None or memory_gib <= float(maximum))
    )


def _benchmark_step(
    *,
    arm_id: str,
    arm: Mapping[str, Any],
    seed: int,
    cache: SynthesisProgramProductionCache,
    runtime: Mapping[str, Any],
    model_config: Mapping[str, Any],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    selected_batches: Sequence[np.ndarray] | None = None,
    warmup_optimizer_steps: int = 0,
    measured_optimizer_steps: int = 1,
) -> dict[str, Any]:
    device = torch.device("cuda")
    _set_determinism(seed, int(runtime["cpu_threads"]))
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=model_config,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(runtime["learning_rate"]),
        weight_decay=float(runtime["weight_decay"]),
    )
    measure = cache.training_measure(arm["program_mass"])
    stratified_sampler = (
        _compile_stratified_program_sampler(cache, measure)
        if model_config.get("architecture") == "reaction_program_graph_transformer"
        else None
    )
    support = np.flatnonzero(measure > 0)
    probabilities = measure[support]
    probabilities /= probabilities.sum()
    rng = np.random.default_rng(seed + 2)
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=device)
    micro_batch = int(runtime["micro_batch_size"])
    accumulation = int(runtime["gradient_accumulation_steps"])
    if selected_batches is not None and len(selected_batches) != accumulation:
        raise SynthesisProgramProductionPreflightError(
            "capacity stress must exercise the full accumulation contract"
        )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    losses: list[float] = []
    measured_step_seconds: list[float] = []
    fixed_state_failures = 0
    balancing_diagnostics = {"micro_batches": 0, "projected_conflicts": 0}
    active_program_states = (
        stratified_sampler.program_states if stratified_sampler is not None else ()
    )
    observed_nodes = 0
    observed_closures = 0
    total_optimizer_steps = warmup_optimizer_steps + measured_optimizer_steps
    gradient_norm = torch.tensor(float("nan"), device=device)
    for optimizer_step in range(total_optimizer_steps):
        torch.cuda.synchronize(device)
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_losses: list[Any] = []
        step_finite = torch.ones((), dtype=torch.bool, device=device)
        step_conflicts = torch.zeros((), dtype=torch.int64, device=device)
        step_fixed_state_failures = torch.zeros((), dtype=torch.int64, device=device)
        for accumulation_index in range(accumulation):
            if (
                selected_batches is None
                and model_config.get("architecture") == "reaction_program_graph_transformer"
            ):
                if stratified_sampler is None:  # pragma: no cover - construction invariant
                    raise SynthesisProgramProductionPreflightError(
                        "Transformer sampler was not compiled"
                    )
                selected = stratified_sampler.sample(micro_batch, rng)
            elif selected_batches is None:
                selected = rng.choice(support, size=micro_batch, replace=True, p=probabilities)
            else:
                selected = selected_batches[accumulation_index]
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
            t = torch.rand(micro_batch, generator=generator, device=device).clamp(0.02, 0.98)
            if model_config.get("architecture") == "reaction_program_graph_transformer":
                from forge.model.reaction_program_transformer import (
                    balanced_pcgrad_backward,
                    per_program_transformer_losses,
                )

                predictions, noisy = synthesis_program_forward(
                    model, clean, node_p0, bond_p0, t, generator
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
            else:
                loss, _, noisy = synthesis_program_forward_loss(
                    model, clean, node_p0, bond_p0, t, generator
                )
            step_finite &= torch.isfinite(loss)
            if model_config.get("architecture") == "reaction_program_graph_transformer":
                if len(family_losses) == 1:
                    (loss / accumulation).backward()
                else:
                    diagnostic = balanced_pcgrad_backward(
                        family_losses,
                        model,
                        scale=1.0 / accumulation,
                        materialize_diagnostics=False,
                    )
                    balancing_diagnostics["micro_batches"] += 1
                    step_conflicts += diagnostic["projected_conflicts"]
            else:
                (loss / accumulation).backward()
            step_losses.append(loss.detach())
            step_fixed_state_failures += (
                ~synthesis_program_fixed_state_exact_tensor(noisy, clean)
            ).to(torch.int64)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(runtime["gradient_clip_norm"])
        )
        step_finite &= torch.isfinite(gradient_norm)
        optimizer.step()
        torch.cuda.synchronize(device)
        published = (
            torch.stack(
                [
                    *step_losses,
                    gradient_norm.detach(),
                    step_conflicts.to(torch.float32),
                    step_fixed_state_failures.to(torch.float32),
                    step_finite.to(torch.float32),
                ]
            )
            .cpu()
            .tolist()
        )
        losses.extend(published[:accumulation])
        balancing_diagnostics["projected_conflicts"] += int(published[-3])
        fixed_state_failures += int(published[-2])
        if not bool(published[-1]):
            raise SynthesisProgramProductionPreflightError(
                f"non-finite accelerator-preflight objective for {arm_id}"
            )
        if optimizer_step >= warmup_optimizer_steps:
            measured_step_seconds.append(time.perf_counter() - step_started)
    elapsed = time.perf_counter() - started
    result = {
        "arm_id": arm_id,
        "losses": losses,
        "gradient_norm": float(gradient_norm),
        "model_state_sha256": _model_state_sha256(model),
        "fixed_state_failures": fixed_state_failures,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds": elapsed,
        "measured_elapsed_seconds": sum(measured_step_seconds),
        "step_elapsed_seconds": measured_step_seconds,
        "seconds_per_optimizer_step": statistics.median(measured_step_seconds),
        "warmup_optimizer_steps": warmup_optimizer_steps,
        "measured_optimizer_steps": measured_optimizer_steps,
        "optimizer_steps_total": total_optimizer_steps,
        "examples": micro_batch * accumulation * measured_optimizer_steps,
        "examples_per_second": (
            micro_batch * accumulation * measured_optimizer_steps / sum(measured_step_seconds)
        ),
        "maximum_nodes_observed": observed_nodes,
        "maximum_closures_observed": observed_closures,
        "gradient_balancing": {
            "method": (
                "equal_family_mass_deterministic_pcgrad"
                if model_config.get("architecture") == "reaction_program_graph_transformer"
                else "none"
            ),
            **balancing_diagnostics,
        },
    }
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def run_synthesis_program_production_accelerator_benchmark(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    output_path: Path,
    *,
    allocated_device: str,
    declared_gpu_type: str,
) -> dict[str, Any]:
    """Exercise the exact production model and batch on one declared accelerator."""

    if torch is None or allocated_device != "cuda" or not torch.cuda.is_available():
        raise SynthesisProgramProductionPreflightError("GPU preflight requires allocated CUDA")
    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionPreflightError,
        label="production GPU preflight config",
    )
    policy = _accelerator_policy(config, declared_gpu_type)
    raw_inputs = config.get("inputs")
    required = {
        "production_design",
        "production_cache",
        "production_training_config",
        "production_evaluation_config",
        "smoke_training_result",
        "smoke_evaluation_result",
        "test_baseline_report",
    }
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != required:
        raise SynthesisProgramProductionPreflightError("GPU preflight inputs changed")
    paths = {label: resolve_pin(value, repo, label=label) for label, value in raw_inputs.items()}
    if paths["production_cache"].resolve() != cache_path.resolve():
        raise SynthesisProgramProductionPreflightError("preflight cache path changed")
    design = read_json_object(
        paths["production_design"],
        error=SynthesisProgramProductionPreflightError,
        label="frozen production design",
    )
    training_config = read_json_object(
        paths["production_training_config"],
        error=SynthesisProgramProductionPreflightError,
        label="production training config",
    )
    evaluation_config = read_json_object(
        paths["production_evaluation_config"],
        error=SynthesisProgramProductionPreflightError,
        label="production evaluation config",
    )
    smoke_training = read_json_object(
        paths["smoke_training_result"],
        error=SynthesisProgramProductionPreflightError,
        label="training smoke result",
    )
    smoke_evaluation = read_json_object(
        paths["smoke_evaluation_result"],
        error=SynthesisProgramProductionPreflightError,
        label="evaluation smoke result",
    )
    test_baseline = read_json_object(
        paths["test_baseline_report"],
        error=SynthesisProgramProductionPreflightError,
        label="repository test-baseline report",
    )
    gates = _validate_static_contract(
        config,
        design,
        training_config,
        evaluation_config,
        smoke_training,
        smoke_evaluation,
        test_baseline,
        expected_schema=str(policy["config_schema"]),
    )
    gates["molecule_report_rendering_available"] = _validate_molecule_rendering()
    gates["projection_matches_frozen_training"] = int(
        policy["production_optimizer_steps_per_arm"]
    ) == int(design["training"]["optimizer_steps"]) and int(policy["production_replicates"]) == len(
        design["training"]["replicate_seeds"]
    )
    runtime, model_config = _validate_runtime(
        config=training_config,
        design=design,
        profile="full",
        device=torch.device("cuda"),
    )
    cache = SynthesisProgramProductionCache(cache_path)
    try:
        expected_counts = {
            program: {fold: int(value) for fold, value in contract["expected_fold_counts"].items()}
            for program, contract in design["programs"].items()
        }
        gates["full_cache_fold_counts_exact"] = cache.fold_counts() == expected_counts
        node_counts = np.diff(cache.arrays["node_offsets"])
        closure_counts = np.diff(cache.arrays["closure_offsets"])
        gates["full_declared_support_present"] = int(node_counts.max()) == int(
            model_config["maximum_heavy_atoms"]
        ) and int(closure_counts.max()) == int(model_config["maximum_closures"])
        layout_prior = SynthesisProgramLayoutPrior(cache)
        layout_support = layout_prior.validate_support()
        layout_schedule = _validate_factorized_layout_schedule(
            layout_prior,
            design,
            evaluation_config,
        )
        gates["factorized_layout_support_closed"] = (
            int(layout_support["programs"]) == len(design["programs"])
            and int(layout_support["semantic_bundles"]) >= len(design["programs"])
            and int(layout_support["component_size_support_cells"])
            >= int(layout_support["semantic_bundles"])
        )
        gates["full_factorized_layout_schedule_passed"] = (
            int(layout_schedule["replicates"]) == len(design["training"]["replicate_seeds"])
            and int(layout_schedule["layout_cells"]) > 0
            and int(layout_schedule["layouts"]) > int(layout_schedule["layout_cells"])
        )
        shared_mass = design["training"]["arms"]["shared_three_program_conditioned"]["program_mass"]
        shared_measure = cache.training_measure(shared_mass)
        node_marginal, bond_marginal = cache.source_marginals(
            shared_measure,
            node_classes=len(cache.atom_vocabulary),
            bond_classes=int(model_config["bond_classes"]),
            probability_floor=float(model_config["source_probability_floor"]),
        )
        seed = int(design["training"]["replicate_seeds"][0])
        benchmarks = {
            arm_id: _benchmark_step(
                arm_id=arm_id,
                arm=arm,
                seed=seed,
                cache=cache,
                runtime=runtime,
                model_config=model_config,
                node_marginal=node_marginal,
                bond_marginal=bond_marginal,
                warmup_optimizer_steps=int(policy["warmup_optimizer_steps"]),
                measured_optimizer_steps=int(policy["measured_optimizer_steps"]),
            )
            for arm_id, arm in design["training"]["arms"].items()
        }
        repeated = _benchmark_step(
            arm_id="shared_three_program_conditioned_determinism_repeat",
            arm=design["training"]["arms"]["shared_three_program_conditioned"],
            seed=seed,
            cache=cache,
            runtime=runtime,
            model_config=model_config,
            node_marginal=node_marginal,
            bond_marginal=bond_marginal,
            warmup_optimizer_steps=int(policy["warmup_optimizer_steps"]),
            measured_optimizer_steps=int(policy["measured_optimizer_steps"]),
        )
        maximum_index = int(np.argmax(node_counts))
        stress_batches = tuple(
            np.full(int(runtime["micro_batch_size"]), maximum_index, dtype=np.int64)
            for _ in range(int(runtime["gradient_accumulation_steps"]))
        )
        stress = _benchmark_step(
            arm_id="maximum_194_atom_capacity_stress",
            arm=design["training"]["arms"]["shared_three_program_conditioned"],
            seed=seed,
            cache=cache,
            runtime=runtime,
            model_config=model_config,
            node_marginal=node_marginal,
            bond_marginal=bond_marginal,
            selected_batches=stress_batches,
            measured_optimizer_steps=int(policy["maximum_support_optimizer_steps"]),
        )
    finally:
        cache.close()
    reference = benchmarks["shared_three_program_conditioned"]
    properties = torch.cuda.get_device_properties(0)
    peak_reserved = max(
        [int(value["peak_reserved_bytes"]) for value in benchmarks.values()]
        + [int(repeated["peak_reserved_bytes"]), int(stress["peak_reserved_bytes"])]
    )
    gpu_identity_gate = _runtime_matches_accelerator(properties, policy)
    gates.update(
        {
            "all_four_arms_finite": len(benchmarks) == 4
            and all(
                all(math.isfinite(loss) for loss in row["losses"]) for row in benchmarks.values()
            ),
            "fixed_state_failures_zero": all(
                int(row["fixed_state_failures"]) == 0 for row in benchmarks.values()
            )
            and int(repeated["fixed_state_failures"]) == 0
            and int(stress["fixed_state_failures"]) == 0,
            "deterministic_repeat_exact": reference["model_state_sha256"]
            == repeated["model_state_sha256"]
            and reference["losses"] == repeated["losses"],
            "maximum_atom_capacity_exercised": int(stress["maximum_nodes_observed"])
            == int(model_config["maximum_heavy_atoms"]),
            "gpu_memory_headroom_at_least_ten_percent": peak_reserved
            <= int(properties.total_memory * 0.9),
            "declared_gpu_matches_runtime": gpu_identity_gate,
            "raw_family_count_sampling_absent": True,
        }
    )
    production_steps = int(policy["production_optimizer_steps_per_arm"])
    production_replicates = int(policy["production_replicates"])
    seconds_per_replicate = sum(
        float(value["seconds_per_optimizer_step"]) * production_steps
        for value in benchmarks.values()
    )
    rate = float(policy["gpu_usd_per_second"])
    projection = {
        "scope": "training_only_excludes_evaluation_and_startup",
        "optimizer_steps_per_arm": production_steps,
        "arms": len(benchmarks),
        "replicates": production_replicates,
        "seconds_per_replicate": seconds_per_replicate,
        "hours_per_replicate": seconds_per_replicate / 3600.0,
        "seconds_all_replicates": seconds_per_replicate * production_replicates,
        "gpu_usd_per_second": rate,
        "gpu_cost_usd_per_replicate": seconds_per_replicate * rate,
        "gpu_cost_usd_all_replicates": seconds_per_replicate * production_replicates * rate,
        "pricing": policy["pricing"],
    }
    result = {
        "schema_version": str(policy["result_schema"]),
        "status": "pass" if all(gates.values()) else "fail",
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "cache": artifact_record(cache_path),
        "design_sha256": str(sha256_file(paths["production_design"])),
        "device": {
            "declared_gpu_type": declared_gpu_type,
            "name": properties.name,
            "total_memory_bytes": int(properties.total_memory),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "runtime": runtime,
        "model": model_config,
        "benchmarks": benchmarks,
        "determinism_repeat": repeated,
        "maximum_support_stress": stress,
        "factorized_layout_support": layout_support,
        "factorized_layout_schedule": layout_schedule,
        "peak_reserved_bytes": peak_reserved,
        "projection": projection,
        "math_mode": {
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        },
        "gates": gates,
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "This preflight establishes execution capacity and determinism, not model quality.",
            "No training checkpoint or generated candidate is selected or published by this stage.",
        ],
    }
    write_json(output_path, result)
    if result["status"] != "pass":
        raise SynthesisProgramProductionPreflightError(
            f"production GPU preflight gates failed: {gates}"
        )
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SynthesisProgramProductionPreflightError",
    "_accelerator_policy",
    "_runtime_matches_accelerator",
    "run_synthesis_program_production_accelerator_benchmark",
    "_validate_molecule_rendering",
]
