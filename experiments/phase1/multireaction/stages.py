"""Thin orchestration adapters for the multi-reaction corpus experiment."""

from __future__ import annotations

from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)


def _run_local_support_stage(context: RunContext, *, morphology: bool) -> StageResult:
    from experiments.phase1.multireaction.local_chemistry_support import (
        build_local_chemistry_support_artifact,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = build_local_chemistry_support_artifact(
        context.config_path,
        context.repo,
        context.output_path("."),
    )
    metrics = {
        "programs": len(result["programs"]),
        "training_records": sum(
            int(value["training_records"]) for value in result["programs"].values()
        ),
    }
    if morphology:
        metrics["role_cycle_types"] = sum(
            int(value["role_conditioned_fundamental_cycle_types"])
            for value in result["programs"].values()
        )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "policy",
                "policy.json",
                (
                    "forge.local_chemistry_support.v2"
                    if morphology
                    else "forge.local_chemistry_support.v1"
                ),
            ),
            ProducedArtifact("result", "result.json", str(result["schema_version"])),
        ),
        metrics=metrics,
        summary={
            "status": result["status"],
            "candidate_selection": False,
            "component_vocabulary": False,
            "role_conditioned_ring_morphology": morphology,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.local-chemistry-support.v1")
def run_local_chemistry_support_stage(context: RunContext) -> StageResult:
    """Build the legacy training-only local graph support policy."""

    return _run_local_support_stage(context, morphology=False)


@stage("model.local-morphology-support.v2")
def run_local_morphology_support_stage(context: RunContext) -> StageResult:
    """Build the training-only local chemistry and ring-morphology support policy."""

    return _run_local_support_stage(context, morphology=True)


def _run_local_resampling_stage(context: RunContext) -> StageResult:
    from experiments.phase1.multireaction.production_evaluation import (
        run_synthesis_program_production_evaluation,
    )
    from forge.core.io import read_json_object

    config = context.config()
    configured_inputs = config.get("inputs")
    if not isinstance(configured_inputs, dict):
        raise ValueError("local-chemistry resampling config has no input mapping")
    require_config_inputs(context, config, labels=set(configured_inputs))
    training_result_path = context.input(f"training_result_r{context.replicate}")
    checkpoint_archive_path = context.input(f"checkpoint_archive_r{context.replicate}")
    training = read_json_object(
        training_result_path,
        error=ValueError,
        label="frozen production training result",
    )
    if int(training.get("replicate", -1)) != context.replicate:
        raise ValueError("resampling replicate and frozen training result disagree")
    result = run_synthesis_program_production_evaluation(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        checkpoint_archive_path,
        training_result_path,
        context.output_path("."),
        profile=context.profile,
        replicate=context.replicate,
        allocated_device=context.resources.device,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.jsonl.gz",
                "forge.synthesis_program_production_samples.v1",
                rows=int(result["sample_rows"]),
            ),
            ProducedArtifact(
                "molecule_report",
                "molecule_report.html",
                "forge.synthesis_program_production_molecule_report.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_evaluation_result.v1",
            ),
        ),
        metrics={
            "sample_rows": int(result["sample_rows"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "replicate": context.replicate,
            "checkpoint_weights_reused": True,
            "training_calls": 0,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.synthesis-program-local-chemistry-resampling.v1")
def run_local_chemistry_resampling_stage(context: RunContext) -> StageResult:
    """Resample one frozen production checkpoint with legacy local chemistry."""

    return _run_local_resampling_stage(context)


@stage("model.synthesis-program-local-morphology-resampling.v2")
def run_local_morphology_resampling_stage(context: RunContext) -> StageResult:
    """Resample frozen checkpoints with train-derived role-local ring morphology."""

    return _run_local_resampling_stage(context)


@stage("model.conditional-role-dependence.v1")
def run_conditional_role_dependence_stage(context: RunContext) -> StageResult:
    """Run the frozen within-context intact-versus-shuffled Ugi diagnostic."""

    from experiments.phase1.multireaction.benchmark_studies import (
        run_conditional_dependence_study,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_conditional_dependence_study(
        context.config_path, context.repo, context.output_path("result.json")
    )
    return StageResult(
        artifacts=(
            ProducedArtifact("result", "result.json", "forge.conditional_role_dependence.v1"),
        ),
        metrics={
            "eligible_rows": int(result["eligible_rows"]),
            "classifier_auc_mean": float(result["classifier_auc_mean"]),
        },
        summary={
            "claim_authorized": bool(result["interpretation_gate"]["claim_authorized"]),
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.learned-inventory-selector.v1")
def run_learned_inventory_selector_stage(context: RunContext) -> StageResult:
    """Train and assess the learned finite-vocabulary Ugi selector."""

    from experiments.phase1.multireaction.benchmark_studies import (
        run_learned_inventory_selector_study,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_learned_inventory_selector_study(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        replicate=context.replicate,
        allocated_device=context.resources.device,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "attempts", "attempts.jsonl.gz", "forge.common_ugi_baseline_attempt.v1"
            ),
            ProducedArtifact(
                "assessed_attempts",
                "assessed_attempts.jsonl.gz",
                "forge.common_ugi_assessed_attempts.v1",
            ),
            ProducedArtifact(
                "route_assessed_attempts",
                "route_assessed_attempts.jsonl.gz",
                "forge.common_ugi_route_assessed_attempts.v1",
            ),
            ProducedArtifact(
                "checkpoint", "checkpoint.pt", "forge.learned_inventory_selector_checkpoint.v1"
            ),
            ProducedArtifact("result", "result.json", "forge.learned_inventory_selector_result.v1"),
        ),
        metrics={
            "attempts": int(result["common_assessment"]["attempts"]),
            "exact_l1": int(result["common_assessment"]["metrics"]["exact_l1_program"]),
            "complete_dossiers": int(result["route_evidence_assessment"]["complete_dossier"]),
        },
        summary={
            "status": result["status"],
            "finite_inventory": True,
            "candidate_selection": False,
            "route_or_oracle_calls_during_generation": 0,
        },
    )


@stage("model.transformer-mechanism-study.v1")
def run_transformer_mechanism_study_stage(context: RunContext) -> StageResult:
    """Execute the separate mechanism/FACT or held-family study."""

    from experiments.phase1.multireaction.mechanism_study import (
        run_transformer_mechanism_study,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_transformer_mechanism_study(
        context.config_path,
        context.repo,
        context.output_path("."),
        work_dir=context.work_dir,
        profile=context.profile,
        replicate=context.replicate,
        allocated_device=context.resources.device,
        resume=context.resume,
    )
    calibration_schema = {
        "bl_lx_repair_calibration": "forge.bl_lx_repair_calibration_result.v1",
        "bl_core_constraint_calibration": "forge.bl_core_constraint_calibration_result.v1",
        "ugi_train_exposure_calibration": "forge.ugi_train_exposure_calibration_result.v1",
    }.get(result["study"])
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "design", "study_design.json", "forge.synthesis_program_production_design_config.v1"
            ),
            ProducedArtifact(
                "training_result",
                "training/result.json",
                "forge.synthesis_program_production_training_result.v1",
            ),
            ProducedArtifact(
                "checkpoints",
                "training/checkpoints.tar",
                "forge.synthesis_program_production_checkpoint_archive.v1",
            ),
            ProducedArtifact(
                "samples",
                "evaluation/samples.jsonl.gz",
                "forge.synthesis_program_production_samples.v1",
            ),
            ProducedArtifact(
                "evaluation_result",
                "evaluation/result.json",
                calibration_schema or "forge.synthesis_program_production_evaluation_result.v1",
            ),
            ProducedArtifact(
                "result", "result.json", "forge.transformer_mechanism_study_result.v1"
            ),
        ),
        metrics={"arms": len(result["arms"])},
        summary={
            "status": result["status"],
            "study": result["study"],
            "held_reaction_family_hard_gate": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.program-semantic-intervention.v1")
def run_program_semantic_intervention_stage(context: RunContext) -> StageResult:
    """Measure paired inference dependence on factual reaction-program coordinates."""

    from experiments.phase1.multireaction.program_semantic_intervention import (
        SAMPLES_SCHEMA,
        run_program_semantic_intervention,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_program_semantic_intervention(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        allocated_device=context.resources.device,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.jsonl.gz",
                SAMPLES_SCHEMA,
                rows=int(result["sample_rows"]),
            ),
            ProducedArtifact("result", "result.json", str(result["schema_version"])),
        ),
        metrics={
            "training_replicates": len(result["replicates"]),
            "conditions": len(result["conditions"]),
            "sample_rows": int(result["sample_rows"]),
        },
        summary={
            "status": result["status"],
            "paired_inference_only": True,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.external-ugi-common-export.v1")
def run_external_ugi_common_export_stage(context: RunContext) -> StageResult:
    """Export the shared split, component inventory and registry reaction for native ports."""

    from experiments.phase1.multireaction.external_baselines import run_external_ugi_export

    config = context.config()
    require_config_inputs(context, config)
    result = run_external_ugi_export(context.config_path, context.repo, context.output_path("."))
    return StageResult(
        artifacts=(
            ProducedArtifact("train", "train.csv.gz", "forge.external_ugi_products.v1"),
            ProducedArtifact("calibration", "calibration.csv.gz", "forge.external_ugi_products.v1"),
            ProducedArtifact("heldout", "heldout.csv.gz", "forge.external_ugi_products.v1"),
            ProducedArtifact(
                "train_components",
                "train_components.csv.gz",
                "forge.external_ugi_components.v1",
            ),
            ProducedArtifact(
                "ugi_reaction", "ugi_reaction.json", "forge.external_ugi_reaction_export.v1"
            ),
            ProducedArtifact("result", "result.json", "forge.external_ugi_common_input_export.v1"),
        ),
        metrics={
            "train_products": int(result["fold_counts"]["train"]),
            "heldout_products": int(result["fold_counts"]["heldout"]),
            "methods": len(result["methods"]),
        },
        summary={
            "completed_external_runs": 0,
            "native_ports_pending": len(result["methods"]),
            "candidate_selection": False,
        },
    )


@stage("model.finite-component-catalogue-baseline.v1")
def run_finite_component_catalogue_baseline_stage(context: RunContext) -> StageResult:
    """Execute the train-only oracle catalogue assembler at the model attempt budget."""

    from experiments.phase1.multireaction.catalogue_baseline import (
        COMMON_ATTEMPT_SCHEMA,
        RESULT_SCHEMA,
        SAMPLES_SCHEMA,
        run_finite_component_catalogue_baseline,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_finite_component_catalogue_baseline(
        context.config_path,
        context.repo,
        context.output_path("."),
        profile=context.profile,
        replicate=context.replicate,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.jsonl.gz",
                SAMPLES_SCHEMA,
                rows=int(result["attempts_per_program"]) * len(result["catalogue"]),
            ),
            ProducedArtifact("result", "result.json", RESULT_SCHEMA),
            ProducedArtifact(
                "ugi_attempts",
                "ugi_attempts.jsonl.gz",
                COMMON_ATTEMPT_SCHEMA,
                rows=int(result["attempts_per_program"]),
            ),
        ),
        metrics={
            "attempts_per_program": int(result["attempts_per_program"]),
            "programs": len(result["catalogue"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "arm_id": result["arm_id"],
            "train_catalogue_only": True,
            "route_or_oracle_calls": 0,
            "candidate_selection": False,
        },
    )


@stage("model.shared-synthesis-program-production-cache.v1")
def build_shared_synthesis_program_production_cache(context: RunContext) -> StageResult:
    """Pack every qualified program record under the frozen weighted design."""

    from forge.corpus.synthesis_program_production_cache import (
        build_synthesis_program_production_cache,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = build_synthesis_program_production_cache(
        context.config_path,
        context.repo,
        context.output_path("cache.npz"),
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "cache",
                "cache.npz",
                "forge.synthesis_program_production_cache.v1",
                rows=int(result["records"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_cache_result.v1",
            ),
        ),
        metrics={
            "records": int(result["records"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "raw_family_count_sampling": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.shared-synthesis-program-production-accelerator-benchmark.v1")
def benchmark_shared_synthesis_program_production_accelerator(
    context: RunContext,
) -> StageResult:
    """Qualify one explicit accelerator and project matched production training cost."""

    from experiments.phase1.multireaction.production_preflight import (
        RESULT_SCHEMA,
        run_synthesis_program_production_accelerator_benchmark,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_synthesis_program_production_accelerator_benchmark(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.output_path("result.json"),
        allocated_device=context.resources.device,
        declared_gpu_type=context.resources.gpu_type or "",
    )
    projection = result["projection"]
    if not isinstance(projection, dict):
        raise ValueError("accelerator benchmark did not produce a training projection")
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                RESULT_SCHEMA,
            ),
        ),
        metrics={
            "gates_passed": int(sum(result["gates"].values())),
            "peak_reserved_bytes": int(result["peak_reserved_bytes"]),
            "seconds_per_replicate": float(projection["seconds_per_replicate"]),
            "gpu_cost_usd_all_replicates": float(projection["gpu_cost_usd_all_replicates"]),
        },
        summary={
            "status": result["status"],
            "declared_gpu_type": result["device"]["declared_gpu_type"],
            "device": result["device"]["name"],
            "projection_scope": projection["scope"],
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.shared-synthesis-program-h100-training-profile.v1")
def profile_shared_synthesis_program_training_h100(context: RunContext) -> StageResult:
    """Profile the frozen mixed-program training step and bounded optimization candidates."""

    from experiments.phase1.multireaction.h100_training_profile import (
        RESULT_SCHEMA,
        run_h100_training_profile,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_h100_training_profile(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.output_path("result.json"),
        allocated_device=context.resources.device,
        declared_gpu_type=context.resources.gpu_type or "",
    )
    baseline = result["baseline"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                RESULT_SCHEMA,
            ),
        ),
        metrics={
            "gates_passed": int(sum(result["gates"].values())),
            "peak_reserved_bytes": int(result["peak_reserved_bytes"]),
            "baseline_seconds_per_optimizer_step": float(
                baseline["median_wall_seconds_per_optimizer_step"]
            ),
            "baseline_examples_per_second": float(baseline["examples_per_second_wall"]),
        },
        summary={
            "status": result["status"],
            "device": result["device"]["name"],
            "selected_candidate": result["selected_candidate"],
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.shared-synthesis-program-production-training.v1")
def train_shared_synthesis_program_production(context: RunContext) -> StageResult:
    """Train the four frozen matched arms for one explicit replicate."""

    from experiments.phase1.multireaction.production_training import (
        run_synthesis_program_production_training,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_synthesis_program_production_training(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.output_path("."),
        work_dir=context.work_dir,
        profile=context.profile,
        replicate=context.replicate,
        allocated_device=context.resources.device,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "checkpoints",
                "checkpoints.tar",
                "forge.synthesis_program_production_checkpoint_archive.v1",
                rows=sum(len(value["checkpoints"]) for value in result["arms"].values()),
            ),
            ProducedArtifact(
                "progress",
                "progress.json",
                "forge.synthesis_program_production_progress.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_training_result.v1",
            ),
        ),
        metrics={
            "arms": len(result["arms"]),
            "optimizer_steps_per_arm": int(result["runtime"]["optimizer_steps"]),
            "fixed_state_failures": sum(
                int(value["fixed_state_failures"]) for value in result["arms"].values()
            ),
        },
        summary={
            "status": result["status"],
            "replicate": context.replicate,
            "seed": int(result["seed"]),
            "route_or_oracle_calls": 0,
            "candidate_selection": False,
        },
    )


def _reaction_program_specialist_stage(
    context: RunContext,
    *,
    profile: str,
    topology_specialist: bool,
) -> StageResult:
    from experiments.phase1.multireaction.reaction_specialization import (
        run_reaction_program_specialization,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_reaction_program_specialization(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.output_path("."),
        work_dir=context.work_dir,
        profile=profile,
        allocated_device=context.resources.device,
        resume=context.resume,
    )
    suffix = "topology_specialization" if topology_specialist else "specialization"
    checkpoint_schema = (
        "forge.reaction_program_topology_specialist_checkpoint.v2"
        if topology_specialist
        else "forge.reaction_program_specialist_checkpoint.v1"
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "checkpoint",
                "specialist_checkpoint.pt",
                checkpoint_schema,
            ),
            ProducedArtifact(
                "progress",
                "progress.json",
                f"forge.reaction_program_{suffix}_progress."
                f"{'v2' if topology_specialist else 'v1'}",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                f"forge.reaction_program_{suffix}_result."
                f"{'v2' if topology_specialist else 'v1'}",
            ),
        ),
        metrics={
            "optimizer_steps": int(result["observed"]["optimizer_steps"]),
            "additional_examples": int(result["observed"]["additional_examples"]),
            "cumulative_examples": int(result["observed"]["cumulative_examples"]),
            "specialist_parameters": int(result["specialist"]["specialist_trainable_parameters"]),
        },
        summary={
            "status": result["status"],
            "target_program": result["target_program"],
            "preflight_only": profile != "full",
            "shared_parameters_frozen": True,
            "topology_head_supervised": topology_specialist,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.reaction-program-specialization.v1")
def train_reaction_program_specialist(context: RunContext) -> StageResult:
    """Fine-tune one lightweight family adapter from an authenticated shared checkpoint."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=False,
    )


@stage("model.reaction-program-specialization-h100-preflight.v1")
def preflight_reaction_program_specialist_h100(context: RunContext) -> StageResult:
    """Exercise authenticated specialist loading, a short final batch and checkpointing on H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="smoke",
        topology_specialist=False,
    )


@stage("model.reaction-program-topology-specialization.v2")
def train_reaction_program_topology_specialist(context: RunContext) -> StageResult:
    """Fine-tune a family adapter and explicit child-count head from one frozen base."""

    return _reaction_program_specialist_stage(
        context,
        profile=context.profile,
        topology_specialist=True,
    )


@stage("model.reaction-program-topology-specialization-h100-preflight.v2")
def preflight_reaction_program_topology_specialist_h100(
    context: RunContext,
) -> StageResult:
    """Qualify the topology-specialist delta and direct objective on the target H100."""

    return _reaction_program_specialist_stage(
        context,
        profile="h100_preflight",
        topology_specialist=True,
    )


@stage("model.shared-synthesis-program-production-evaluation.v1")
def evaluate_shared_synthesis_program_production(context: RunContext) -> StageResult:
    """Evaluate every frozen checkpoint without selecting models or candidates."""

    from experiments.phase1.multireaction.production_evaluation import (
        run_synthesis_program_production_evaluation,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_synthesis_program_production_evaluation(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.dependency("training", "checkpoints").path,
        context.dependency("training", "result").path,
        context.output_path("."),
        profile=context.profile,
        replicate=context.replicate,
        allocated_device=context.resources.device,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.jsonl.gz",
                "forge.synthesis_program_production_samples.v1",
                rows=int(result["sample_rows"]),
            ),
            ProducedArtifact(
                "molecule_report",
                "molecule_report.html",
                "forge.synthesis_program_production_molecule_report.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_evaluation_result.v1",
            ),
        ),
        metrics={
            "sample_rows": int(result["sample_rows"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "coverage_and_precision_reported": True,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.synthesis-program-layout-schedule-preflight.v1")
def preflight_synthesis_program_layout_schedule(context: RunContext) -> StageResult:
    """Materialize every frozen factorized layout before checkpoint evaluation."""

    from experiments.phase1.multireaction.layout_schedule_preflight import (
        RESULT_SCHEMA,
        validate_layout_schedule,
    )

    config = context.config()
    configured_inputs = config.get("inputs")
    if not isinstance(configured_inputs, dict):
        raise ValueError("layout-schedule preflight config has no input mapping")
    require_config_inputs(context, config, labels=set(configured_inputs))
    result = validate_layout_schedule(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.input("training_result"),
        context.output_path("result.json"),
        profile=context.profile,
        replicate=context.replicate,
    )
    return StageResult(
        artifacts=(ProducedArtifact("result", "result.json", RESULT_SCHEMA),),
        metrics={
            "layout_cells": int(result["schedule"]["cells"]),
            "layouts": int(result["schedule"]["layouts"]),
            "maximum_node_count": int(result["schedule"]["maximum_node_count"]),
        },
        summary={
            "status": result["status"],
            "repair_or_retry": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.shared-synthesis-program-production-evaluation-from-pins.v1")
def evaluate_shared_synthesis_program_production_from_pins(context: RunContext) -> StageResult:
    """Evaluate an authenticated external checkpoint package without retraining."""

    from experiments.phase1.multireaction.production_evaluation import (
        run_synthesis_program_production_evaluation,
    )

    config = context.config()
    configured_inputs = config.get("inputs")
    if not isinstance(configured_inputs, dict):
        raise ValueError("checkpoint evaluation config has no input mapping")
    require_config_inputs(context, config, labels=set(configured_inputs))
    result = run_synthesis_program_production_evaluation(
        context.config_path,
        context.repo,
        context.input("production_cache"),
        context.input("checkpoint_archive"),
        context.input("training_result"),
        context.output_path("."),
        profile=context.profile,
        replicate=context.replicate,
        allocated_device=context.resources.device,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.jsonl.gz",
                "forge.synthesis_program_production_samples.v1",
                rows=int(result["sample_rows"]),
            ),
            ProducedArtifact(
                "molecule_report",
                "molecule_report.html",
                "forge.synthesis_program_production_molecule_report.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_evaluation_result.v1",
            ),
        ),
        metrics={
            "sample_rows": int(result["sample_rows"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "checkpoint_weights_reused": True,
            "training_calls": 0,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("model.shared-synthesis-program-cache.v1")
def build_shared_synthesis_program_cache(context: RunContext) -> StageResult:
    """Build one equal-mass training-fold record per qualified program."""

    from forge.corpus.synthesis_program_training import (
        build_bounded_synthesis_program_training_cache,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = build_bounded_synthesis_program_training_cache(
        context.config_path,
        context.repo,
        context.output_path("cache.json"),
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "cache",
                "cache.json",
                "forge.synthesis_program_training_cache.v1",
                rows=len(result["selected_records"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_cache_qualification.v1",
            ),
        ),
        metrics={
            "programs": len(result["selected_records"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={"status": result["status"], "production_training_authorized": False},
    )


@stage("model.shared-synthesis-program-training.v1")
def train_shared_synthesis_program_flow(context: RunContext) -> StageResult:
    """Overfit the shared model on the bounded cache without launching production."""

    from experiments.phase1.multireaction.shared_training import (
        run_shared_synthesis_program_training,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_shared_synthesis_program_training(
        context.config_path,
        context.repo,
        context.dependency("cache", "cache").path,
        context.output_path("."),
        work_dir=context.work_dir,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "checkpoint",
                "checkpoint.json",
                "forge.synthesis_program_sparse_flow_checkpoint.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_training_result.v1",
            ),
        ),
        metrics={
            "steps": int(result["training"]["steps"]),
            "exact_tensor_programs": int(
                sum(
                    row["reconstruction"]["exact_tensor_records"]
                    for row in result["validation"].values()
                )
            ),
        },
        summary={"status": result["status"], "production_training_authorized": False},
    )


@stage("model.shared-synthesis-program-sampling.v1")
def sample_shared_synthesis_program_flow(context: RunContext) -> StageResult:
    """Sample an independent checkpoint from scrubbed semantic layouts."""

    from experiments.phase1.multireaction.shared_sampling import (
        run_shared_synthesis_program_sampling,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = run_shared_synthesis_program_sampling(
        context.config_path,
        context.repo,
        context.dependency("cache", "cache").path,
        context.dependency("training", "checkpoint").path,
        context.output_path("."),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.json",
                "forge.synthesis_program_samples.v1",
                rows=int(result["metrics"]["samples"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_sampling_result.v1",
            ),
        ),
        metrics={
            "samples": int(result["metrics"]["samples"]),
            "valid": int(result["metrics"]["valid"]),
            "exact_target_graph": int(result["metrics"]["exact_target_graph"]),
        },
        summary={"status": result["status"], "production_sampling_authorized": False},
    )


@stage("model.shared-synthesis-program-qualification.v1")
def qualify_shared_synthesis_program_integration(context: RunContext) -> StageResult:
    """Close the cache/training/sampling gate without promoting an overfit result."""

    from experiments.phase1.multireaction.shared_qualification import (
        qualify_shared_synthesis_program_integration as qualify,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = qualify(
        context.config_path,
        context.repo,
        context.dependency("cache", "result").path,
        context.dependency("training", "result").path,
        context.dependency("sampling", "result").path,
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_integration_qualification.v1",
            ),
        ),
        metrics={
            "gates_passed": int(sum(result["gates"].values())),
            "total_gates": len(result["gates"]),
        },
        summary={
            "status": result["status"],
            "production_training_authorized": False,
        },
    )


@stage("model.shared-synthesis-program-production-design.v1")
def freeze_shared_synthesis_program_production_design(context: RunContext) -> StageResult:
    """Freeze the matched experiment contract without launching either production arm."""

    from experiments.phase1.multireaction.production_design import (
        freeze_shared_production_comparison_design,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = freeze_shared_production_comparison_design(
        context.config_path,
        context.repo,
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_design_result.v1",
            ),
        ),
        metrics={
            "programs": len(result["corpus"]),
            "arms": int(result["matched_compute"]["arms"]),
            "replicates": int(result["matched_compute"]["replicates"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "production_training_authorized": False,
            "production_sampling_authorized": False,
        },
    )


@stage("model.mixed-repeat-training-design.v1")
def freeze_mixed_repeat_training_design_stage(context: RunContext) -> StageResult:
    """Derive and validate the expanded BL/LX Transformer training design."""

    from experiments.phase1.multireaction.production_design import (
        freeze_mixed_repeat_training_design,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = freeze_mixed_repeat_training_design(
        context.config_path,
        context.repo,
        context.output_path("design.json"),
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "design",
                "design.json",
                "forge.synthesis_program_production_design_config.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.synthesis_program_production_design_result.v1",
            ),
        ),
        metrics={
            "programs": len(result["corpus"]),
            "training_products": sum(
                int(value["folds"]["train"]["records"])
                for value in result["corpus"].values()
            ),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "production_training_authorized": False,
            "production_sampling_authorized": False,
        },
    )

@stage("model.shared-synthesis-program-representation.v1")
def qualify_shared_synthesis_program_representation(context: RunContext) -> StageResult:
    """Require lossless whole-product serialization across every admitted program record."""

    from forge.corpus.synthesis_program_representation import (
        qualify_shared_synthesis_program_representation as qualify,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = qualify(
        context.config_path,
        context.repo,
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.shared_synthesis_program_representation_result.v1",
            ),
        ),
        metrics={
            "programs": int(result["summary"]["programs"]),
            "records_represented": int(result["summary"]["records_represented"]),
            "atom_rows": int(result["summary"]["atom_rows"]),
            "gates_passed": int(sum(result["gates"].values())),
        },
        summary={
            "status": result["status"],
            "component_identifiers_used": False,
            "production_launch_authorized": False,
        },
    )


@stage("corpus.multireaction.lnpdb.v1")
def build_multireaction_corpus(context: RunContext) -> StageResult:
    """Build exact recursive programs and source/component-disjoint ledgers."""

    from forge.corpus.multireaction import build_multireaction_lnpdb_corpus

    config = context.config()
    require_config_inputs(context, config)
    outputs = {
        "atlas": context.output_path("reaction_program_atlas.csv.gz"),
        "steps": context.output_path("reaction_program_steps.csv.gz"),
        "semantic_atoms": context.output_path("semantic_atoms.csv.gz"),
        "provenance": context.output_path("source_provenance.csv.gz"),
        "splits": context.output_path("component_disjoint_splits.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = build_multireaction_lnpdb_corpus(
        context.config_path,
        context.repo,
        outputs=outputs,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "atlas",
                "reaction_program_atlas.csv.gz",
                "forge.multireaction_program_atlas.v1",
                rows=int(summary["unique_source_products"]),
            ),
            ProducedArtifact(
                "steps",
                "reaction_program_steps.csv.gz",
                "forge.multireaction_program_steps.v1",
                rows=int(summary["exact_program_steps"]),
            ),
            ProducedArtifact(
                "semantic_atoms",
                "semantic_atoms.csv.gz",
                "forge.multireaction_semantic_atoms.v2",
                rows=int(summary["semantic_atom_rows"]),
            ),
            ProducedArtifact(
                "provenance",
                "source_provenance.csv.gz",
                "forge.multireaction_source_provenance.v1",
                rows=int(summary["source_rows"]),
            ),
            ProducedArtifact(
                "splits",
                "component_disjoint_splits.csv.gz",
                "forge.multireaction_component_disjoint_splits.v1",
                rows=int(summary["admitted_products"]),
            ),
            ProducedArtifact("manifest", "manifest.json", "forge.multireaction_lnpdb_manifest.v1"),
            ProducedArtifact("result", "result.json", "forge.multireaction_lnpdb_result.v1"),
        ),
        metrics={
            "admitted_products": int(summary["admitted_products"]),
            "abstained_products": int(summary["abstained_products"]),
            "exact_program_steps": int(summary["exact_program_steps"]),
            "semantic_origin_products": int(summary["semantic_origin_products"]),
        },
        summary={
            "biological_labels_used": False,
            "reductive_amination_substructure_rate_reported": False,
            "sampling_policy": summary["sampling_policy"],
        },
    )


@stage("corpus.multireaction.reaction-enumerated-expansion.v1")
def build_multireaction_reaction_enumerated_expansion(context: RunContext) -> StageResult:
    """Expand BL/LX from source-linked components with exact forward programs."""

    from forge.corpus.multireaction_expansion import build_multireaction_expansion

    config = context.config()
    require_config_inputs(context, config)
    outputs = {
        "components": context.output_path("component_registry.csv.gz"),
        "attempts": context.output_path("enumeration_attempts.csv.gz"),
        "atlas": context.output_path("reaction_program_atlas.csv.gz"),
        "steps": context.output_path("reaction_program_steps.csv.gz"),
        "semantic_atoms": context.output_path("semantic_atoms.csv.gz"),
        "provenance": context.output_path("source_provenance.csv.gz"),
        "splits": context.output_path("component_family_splits.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = build_multireaction_expansion(
        context.config_path,
        context.repo,
        outputs=outputs,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "components",
                "component_registry.csv.gz",
                "forge.multireaction_expansion_components.v1",
                rows=int(summary["candidate_components"]),
            ),
            ProducedArtifact(
                "attempts",
                "enumeration_attempts.csv.gz",
                "forge.multireaction_expansion_attempts.v1",
                rows=int(summary["enumeration_attempts"]),
            ),
            ProducedArtifact(
                "atlas",
                "reaction_program_atlas.csv.gz",
                "forge.multireaction_program_atlas.v1",
                rows=int(summary["total_products"]),
            ),
            ProducedArtifact(
                "steps",
                "reaction_program_steps.csv.gz",
                "forge.multireaction_program_steps.v1",
                rows=int(summary["exact_program_steps"]),
            ),
            ProducedArtifact(
                "semantic_atoms",
                "semantic_atoms.csv.gz",
                "forge.multireaction_semantic_atoms.v2",
                rows=int(summary["semantic_atom_rows"]),
            ),
            ProducedArtifact(
                "provenance",
                "source_provenance.csv.gz",
                "forge.multireaction_source_provenance.v1",
            ),
            ProducedArtifact(
                "splits",
                "component_family_splits.csv.gz",
                "forge.multireaction_expanded_component_family_splits.v1",
                rows=int(summary["total_products"]),
            ),
            ProducedArtifact(
                "manifest",
                "manifest.json",
                "forge.multireaction_expansion_manifest.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_expansion_result.v1",
            ),
        ),
        metrics={
            "source_products": int(summary["source_products_preserved"]),
            "computed_products": int(summary["computed_products_admitted"]),
            "training_products": int(summary["product_folds"]["train"]),
            "calibration_products": int(summary["product_folds"]["calibration"]),
            "heldout_products": int(summary["product_folds"]["heldout"]),
        },
        summary={
            "status": result["status"],
            "source_executed_evidence_rewritten": False,
            "biological_labels_used": False,
            "computed_products_are_synthesis_success": False,
            "reductive_amination_substructure_rate_reported": False,
        },
    )


@stage("corpus.multireaction.mixed-repeat-expansion.v1")
def build_multireaction_mixed_repeat_expansion(context: RunContext) -> StageResult:
    """Expand BL/LX with distinct source-linked repeat components in one exact program."""

    from forge.corpus.multireaction_mixed_expansion import (
        build_multireaction_mixed_expansion,
    )

    config = context.config()
    require_config_inputs(context, config)
    outputs = {
        "components": context.output_path("component_registry.csv.gz"),
        "attempts": context.output_path("mixed_enumeration_attempts.csv.gz"),
        "atlas": context.output_path("reaction_program_atlas.csv.gz"),
        "steps": context.output_path("reaction_program_steps.csv.gz"),
        "semantic_atoms": context.output_path("semantic_atoms.csv.gz"),
        "provenance": context.output_path("source_provenance.csv.gz"),
        "splits": context.output_path("component_family_splits.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    strict_model_support = "atom_vocabulary" in config.get("inputs", {})
    if strict_model_support:
        outputs["support_exclusions"] = context.output_path(
            "model_support_exclusions.csv.gz"
        )
    result = build_multireaction_mixed_expansion(
        context.config_path,
        context.repo,
        outputs=outputs,
    )
    summary = result["summary"]
    artifacts = [
            ProducedArtifact(
                "components",
                "component_registry.csv.gz",
                "forge.multireaction_expansion_components.v1",
            ),
            ProducedArtifact(
                "attempts",
                "mixed_enumeration_attempts.csv.gz",
                "forge.multireaction_mixed_expansion_attempts.v1",
                rows=int(summary["mixed_enumeration_attempts"]),
            ),
            ProducedArtifact(
                "atlas",
                "reaction_program_atlas.csv.gz",
                "forge.multireaction_program_atlas.v2",
                rows=int(summary["total_products"]),
            ),
            ProducedArtifact(
                "steps",
                "reaction_program_steps.csv.gz",
                "forge.multireaction_program_steps.v1",
                rows=int(summary["exact_program_steps"]),
            ),
            ProducedArtifact(
                "semantic_atoms",
                "semantic_atoms.csv.gz",
                "forge.multireaction_semantic_atoms.v2",
                rows=int(summary["semantic_atom_rows"]),
            ),
            ProducedArtifact(
                "provenance",
                "source_provenance.csv.gz",
                "forge.multireaction_source_provenance.v1",
            ),
            ProducedArtifact(
                "splits",
                "component_family_splits.csv.gz",
                "forge.multireaction_mixed_component_family_splits.v1",
                rows=int(summary["total_products"]),
            ),
            ProducedArtifact(
                "manifest",
                "manifest.json",
                "forge.multireaction_mixed_expansion_manifest.v1",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_mixed_expansion_result.v1",
            ),
        ]
    if strict_model_support:
        artifacts.insert(
            -2,
            ProducedArtifact(
                "support_exclusions",
                "model_support_exclusions.csv.gz",
                "forge.multireaction_model_support_exclusions.v1",
                rows=int(summary["homogeneous_model_support_exclusions"]),
            ),
        )
    return StageResult(
        artifacts=tuple(artifacts),
        metrics={
            "v1_products": int(summary["v1_products_preserved"]),
            "mixed_products": int(summary["mixed_products_admitted"]),
            "training_products": int(summary["product_folds"]["train"]),
            "calibration_products": int(summary["product_folds"]["calibration"]),
            "heldout_products": int(summary["product_folds"]["heldout"]),
        },
        summary={
            "status": result["status"],
            "source_executed_evidence_rewritten": False,
            "biological_labels_used": False,
            "computed_products_are_synthesis_success": False,
            "component_family_assignment_precedes_enumeration": True,
            "reductive_amination_substructure_rate_reported": False,
        },
    )


@stage("model.multireaction.training.v2")
@stage("model.multireaction.training_smoke.v1")
def train_multireaction(context: RunContext) -> StageResult:
    """Exercise the model/checkpoint/sampler seam under matched semantic controls."""

    from experiments.phase1.multireaction.training import run_multireaction_training

    config = context.config()
    require_config_inputs(context, config)
    result = run_multireaction_training(
        context.config_path,
        context.repo,
        context.output_path("."),
        work_dir=context.work_dir,
        resume=context.resume,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "checkpoint",
                "checkpoint.json",
                "forge.multireaction_sparse_flow_checkpoint.v2",
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_training_result.v2",
            ),
        ),
        metrics={
            "training_arms": len(result["arms"]),
        },
        summary={
            "status": result["status"],
            "component_identifiers_used": False,
            "production_comparison": False,
        },
    )


@stage("model.multireaction.sampling.v2")
@stage("model.multireaction.sampling_smoke.v1")
def sample_multireaction(context: RunContext) -> StageResult:
    """Load the training dependency checkpoint and exercise program-conditioned sampling."""

    from experiments.phase1.multireaction.sampling import run_multireaction_sampling

    config = context.config()
    require_config_inputs(context, config)
    checkpoint = context.dependency("training", "checkpoint")
    result = run_multireaction_sampling(
        context.config_path,
        context.repo,
        checkpoint.path,
        context.output_path("."),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "samples",
                "samples.json",
                "forge.multireaction_samples.v2",
                rows=int(result["metrics"]["samples"]),
            ),
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_sampling_result.v2",
            ),
        ),
        metrics={
            "samples": int(result["metrics"]["samples"]),
            "valid_samples": int(result["metrics"]["valid"]),
            "exact_l1_samples": int(result["metrics"]["exact_l1_program"]),
        },
        summary={
            "status": result["status"],
            "component_identifiers_used": False,
            "production_comparison": False,
        },
    )


@stage("model.multireaction.overfit-qualification.v1")
def qualify_multireaction_overfit(context: RunContext) -> StageResult:
    """Apply frozen overfit gates without hiding a negative result."""

    from experiments.phase1.multireaction.qualification import qualify_overfit_run

    config = context.config()
    require_config_inputs(context, config)
    result = qualify_overfit_run(
        context.config_path,
        context.repo,
        context.dependency("training", "result").path,
        context.dependency("sampling", "result").path,
        context.output_path("result.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.multireaction_overfit_qualification.v1",
            ),
        ),
        metrics={
            "gate_pass": bool(result["gate_pass"]),
            "passed_gates": int(sum(result["gates"].values())),
            "total_gates": len(result["gates"]),
        },
        summary={
            "status": result["status"],
            "production_launch_authorized": False,
        },
    )


__all__ = [
    "build_shared_synthesis_program_cache",
    "build_multireaction_corpus",
    "freeze_shared_synthesis_program_production_design",
    "qualify_shared_synthesis_program_representation",
    "qualify_shared_synthesis_program_integration",
    "qualify_multireaction_overfit",
    "sample_shared_synthesis_program_flow",
    "sample_multireaction",
    "train_shared_synthesis_program_flow",
    "train_multireaction",
]
