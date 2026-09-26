"""Stage adapters for the authorized Phase 1 model pipeline.

Adapters live at the orchestration edge: they translate a verified ``RunContext`` into a domain
API call, but contain no chemistry or scientific policy of their own.
"""

from __future__ import annotations

import copy
import gc
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

from experiments._runtime.errors import StageError
from experiments._runtime.registry import stage
from experiments._runtime.stage import (
    ProducedArtifact,
    RunContext,
    StageResult,
    require_config_inputs,
)
from forge.core.hashing import sha256_file
from forge.core.io import write_json

UGI_TREE_ABLATION_SCHEMA = "forge.ugi_tree_transformer_ablation_ladder.v1"


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def _deterministic_tar(paths: list[Path], target: Path, *, base: Path) -> None:
    """Archive checkpoint shards without filesystem timestamps or ownership."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(paths, key=lambda value: value.relative_to(base).as_posix()):
            name = path.relative_to(base).as_posix()
            info = tarfile.TarInfo(name=name)
            info.size = path.stat().st_size
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)


def _resume_training_work(context: RunContext, work: Path) -> bool:
    """Resume an authenticated checkpoint, or restart unauthenticated scratch safely."""

    if not context.resume:
        return False
    if (work / "checkpoint_latest.pt").is_file():
        return True
    # A process can die before its first atomic checkpoint.  Nothing in that directory is a
    # resumable state, so rebuild it deterministically under the same stage fingerprint.
    if work.exists():
        shutil.rmtree(work)
    return False


def _publish_training_outputs(
    context: RunContext,
    work: Path,
    *,
    result_schema: str,
    checkpoint_schema: str,
    progress_schema: str,
) -> tuple[ProducedArtifact, ...]:
    result = json.loads((work / "result.json").read_text())
    for key, filename in (
        ("checkpoint", "checkpoint_best.pt"),
        ("checkpoint_latest", "checkpoint_latest.pt"),
    ):
        record = result.get(key)
        if not isinstance(record, dict) or record.get("sha256") != str(
            sha256_file(work / filename)
        ):
            raise StageError(f"training result does not authenticate {filename}")
        record["path"] = filename
    snapshots = sorted(work.glob("checkpoint_step_*.pt"))
    snapshot_records = []
    for path in snapshots:
        snapshot_records.append(
            {
                "member": path.name,
                "sha256": str(sha256_file(path)),
                "step": int(path.stem.rsplit("_", 1)[1]),
            }
        )
    result["checkpoint_snapshots"] = snapshot_records
    result["execution"] = {
        "backend": context.backend,
        "profile": context.profile,
        "resumed_partial_stage": context.resume,
    }
    write_json(context.output_path("result.json"), result)
    _copy(work / "checkpoint_best.pt", context.output_path("checkpoint_best.pt"))
    _copy(work / "checkpoint_latest.pt", context.output_path("checkpoint_latest.pt"))
    _copy(work / "progress.json", context.output_path("progress.json"))
    _deterministic_tar(snapshots, context.output_path("checkpoint_snapshots.tar"), base=work)
    return (
        ProducedArtifact("result", "result.json", result_schema),
        ProducedArtifact("checkpoint", "checkpoint_best.pt", checkpoint_schema),
        ProducedArtifact("checkpoint_latest", "checkpoint_latest.pt", checkpoint_schema),
        ProducedArtifact("progress", "progress.json", progress_schema),
        ProducedArtifact(
            "checkpoint_snapshots",
            "checkpoint_snapshots.tar",
            "forge.checkpoint_archive.v1",
            rows=len(snapshot_records),
        ),
    )


def _tree_ablation_effective_config(
    context: RunContext, design: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """Resolve one matched ablation arm without duplicating full training configurations."""

    if design.get("schema_version") != UGI_TREE_ABLATION_SCHEMA:
        raise StageError("unsupported Ugi tree-Transformer ablation design")
    profile_orders = design.get("profile_arm_order")
    arms = design.get("arms")
    if not isinstance(profile_orders, dict) or not isinstance(arms, dict):
        raise StageError("Ugi tree-Transformer ablation design is incomplete")
    order = profile_orders.get(context.profile)
    if (
        not isinstance(order, list)
        or not order
        or any(not isinstance(value, str) or value not in arms for value in order)
        or len(set(order)) != len(order)
        or context.replicate >= len(order)
    ):
        raise StageError("profile replicate does not map to one declared ablation arm")
    arm_id = order[context.replicate]

    base_pin = design.get("base_config")
    declared_base = context.stage.inputs.get("base_config")
    if not isinstance(base_pin, dict) or declared_base is None:
        raise StageError("ablation design does not bind its base configuration")
    if base_pin != declared_base.to_mapping():
        raise StageError("ablation design and experiment disagree on the base configuration")
    base = json.loads(context.input("base_config").read_text())
    configured_inputs = base.get("inputs")
    chemistry_inputs = {
        label: pin.to_mapping()
        for label, pin in context.stage.inputs.items()
        if label != "base_config"
    }
    if configured_inputs != chemistry_inputs:
        raise StageError("ablation base config and experiment chemistry inputs differ")

    matched = design.get("matched_contract")
    full = base.get("full")
    duration = base.get("duration_contract")
    if (
        not isinstance(matched, dict)
        or not isinstance(full, dict)
        or not isinstance(duration, dict)
    ):
        raise StageError("ablation matched-duration contract is incomplete")
    if (
        int(full.get("steps", -1)) != int(matched.get("optimizer_steps", -2))
        or int(full.get("batch_size", -1)) != int(matched.get("batch_size", -2))
        or duration.get("maximum_weighted_training_draws") != matched.get("weighted_training_draws")
        or full.get("checkpoint_steps") != matched.get("checkpoint_steps")
        or base.get("seed") != matched.get("initialization_and_minibatch_seed")
        or base.get("promotion_contract", {}).get("heldout_selects_architecture_or_checkpoint")
        is not False
        or base.get("promotion_contract", {}).get(
            "development_training_authorized_after_implementation_review"
        )
        is not True
    ):
        raise StageError("ablation base config violates the matched development contract")

    arm = arms[arm_id]
    if not isinstance(arm, dict) or set(arm) != {
        "scientific_question",
        "model_overrides",
        "objective_overrides",
    }:
        raise StageError(f"ablation arm {arm_id!r} is malformed")
    effective = copy.deepcopy(base)
    effective["task"] = f"Ugi tree-Transformer matched development arm: {arm_id}"
    effective["experiment_arm"] = {
        "arm_id": arm_id,
        "design_sha256": context.stage.config.sha256,
        "replicate": context.replicate,
        "scientific_question": arm["scientific_question"],
    }
    effective["model"].update(dict(arm["model_overrides"]))
    effective["objective"] = copy.deepcopy(arm["objective_overrides"])
    profile_overrides = design.get("execution_profile_overrides", {}).get(context.profile, {})
    if not isinstance(profile_overrides, dict):
        raise StageError("ablation execution-profile override is malformed")
    effective[context.profile].update(copy.deepcopy(profile_overrides))
    return arm_id, effective


@stage("corpus.ugi.calibration-program-draw.v1")
def build_ugi_calibration_program_draw_stage(context: RunContext) -> StageResult:
    """Freeze one source- and calibration-role-balanced program draw."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_calibration import (
        build_calibration_program_draw,
    )

    config = context.config()
    require_config_inputs(context, config)
    result = build_calibration_program_draw(
        context.config_path,
        context.repo,
        context.output_path("program_draw.json"),
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "program_draw",
                "program_draw.json",
                "forge.ugi_tree_transformer_calibration_program_draw.v1",
                rows=len(result["samples"]),
            ),
        ),
        metrics={"programs": len(result["samples"])},
        summary={
            "candidate_selection": False,
            "evaluation_fold": "calibration",
            "heldout_rows_used": False,
            "paired_random_streams_required": True,
        },
    )


@stage("corpus.phase1.freeze.v1")
def freeze_phase1_corpus(context: RunContext) -> StageResult:
    """Build the exact Phase 1 corpus contract inside an isolated stage directory."""

    from forge.corpus import freeze_phase1_data_contract

    config = context.config()
    require_config_inputs(context, config)
    paths = {
        "ugi_assignments": context.output_path("ugi_l1_assignments.csv.gz"),
        "ugi_provenance": context.output_path("ugi_l1_constitutional_provenance.csv.gz"),
        "manifest": context.output_path("manifest.json"),
        "result": context.output_path("result.json"),
    }
    result = freeze_phase1_data_contract(
        context.config_path,
        context.repo,
        output_paths=paths,
        output_display_root=context.output_dir,
    )
    summary = result["summary"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "assignments",
                "ugi_l1_assignments.csv.gz",
                "phase1_ugi_l1_assignments.v1",
                rows=int(summary["ugi_l1_products"]),
            ),
            ProducedArtifact(
                "provenance",
                "ugi_l1_constitutional_provenance.csv.gz",
                "phase1_ugi_l1_constitutional_provenance.v1",
                rows=int(summary["ugi_source_rows"]),
            ),
            ProducedArtifact("manifest", "manifest.json", "phase1_product_l1_split_manifest.v3"),
            ProducedArtifact("result", "result.json", "phase1_product_l1_data_result.v3"),
        ),
        metrics={
            "r0_rows": int(summary["r0_rows"]),
            "r1_rows": int(summary["r1_rows"]),
            "ugi_l1_products": int(summary["ugi_l1_products"]),
        },
        summary={
            "guidance_enabled": False,
            "r1_sampling_weight": "realism_weight",
            "status": result["status"],
        },
    )


@stage("corpus.ugi.training-cache-verify.v1")
def verify_ugi_training_cache(context: RunContext) -> StageResult:
    """Validate the frozen tensor cache before any trainer can consume it."""

    from forge.corpus.training_cache import load_ugi_training_cache_payload

    config = context.config()
    configured_labels = set(config.get("inputs", {}))
    require_config_inputs(context, config, labels=configured_labels)
    cache_path = context.input("prepared_cache")
    payload = load_ugi_training_cache_payload(cache_path)
    cached_inputs = payload.get("inputs")
    if not isinstance(cached_inputs, dict):
        raise StageError("prepared training cache has no authenticated inputs")
    for label in sorted(configured_labels):
        if cached_inputs.get(label, {}).get("sha256") != config["inputs"][label]["sha256"]:
            raise StageError(f"prepared training cache input changed for {label!r}")
    records = payload.get("joint_records_by_fold")
    if not isinstance(records, dict):
        raise StageError("prepared training cache has no fold records")
    counts = {fold: len(values) for fold, values in records.items()}
    if counts != config.get("expected_fold_counts"):
        raise StageError(f"prepared training cache fold counts changed: {counts}")
    receipt = {
        "cache": context.stage.inputs["prepared_cache"].to_mapping(),
        "fold_counts": counts,
        "inputs": {label: config["inputs"][label] for label in sorted(configured_labels)},
        "schema_version": "forge.ugi_training_cache_receipt.v1",
        "status": "verified",
    }
    del payload, records
    gc.collect()
    write_json(context.output_path("receipt.json"), receipt)
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "receipt",
                "receipt.json",
                "forge.ugi_training_cache_receipt.v1",
                rows=sum(counts.values()),
            ),
        ),
        metrics={"records": sum(counts.values())},
        summary={"cache_sha256": context.stage.inputs["prepared_cache"].sha256},
    )


@stage("generate.ugi.joint-train.v1")
def train_ugi_joint_stage(context: RunContext) -> StageResult:
    """Run the frozen all-fold joint-flow contract with deterministic resume state."""

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
        train_ugi_joint_sparse,
    )

    context.dependency("cache", "receipt")
    config = context.config()
    require_config_inputs(context, config)
    runtime = config.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("joint training config and declared device differ")
    work = context.work_dir / "joint_training"
    resume_training = _resume_training_work(context, work)
    result = train_ugi_joint_sparse(
        context.config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    return StageResult(
        artifacts=artifacts,
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
        },
        summary={
            "route_guidance": False,
            "oracle_guidance": False,
            "selection_mode": result["selection"]["mode"],
        },
    )


@stage("generate.ugi.joint-ablation-train.v1")
def train_ugi_joint_ablation_stage(context: RunContext) -> StageResult:
    """Materialize and train one replicate-indexed tree-Transformer ablation arm."""

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
        train_ugi_joint_sparse,
    )

    context.dependency("cache", "receipt")
    arm_id, effective = _tree_ablation_effective_config(context, context.config())
    runtime = effective.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("ablation training config and declared device differ")
    # Keep the generated config beside, rather than inside, the trainer output directory.  The
    # trainer deliberately refuses a nonempty fresh output directory; putting its own input config
    # there made a first run look like stale training state before step zero.
    work = context.work_dir / "joint_ablation_training"
    resume_training = _resume_training_work(context, work)
    effective_config_path = context.work_dir / "joint_ablation_effective_config.json"
    expected_effective = json.dumps(effective, indent=2, sort_keys=True) + "\n"
    if effective_config_path.is_file():
        if effective_config_path.read_text() != expected_effective:
            raise StageError("persisted effective ablation config changed")
    else:
        write_json(effective_config_path, effective)
    result = train_ugi_joint_sparse(
        effective_config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    _copy(effective_config_path, context.output_path("effective_config.json"))
    return StageResult(
        artifacts=(
            *artifacts,
            ProducedArtifact(
                "effective_config",
                "effective_config.json",
                "phase1_ugi_joint_sparse_training_config.v1",
            ),
        ),
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
        },
        summary={
            "ablation_arm": arm_id,
            "candidate_selection": False,
            "heldout_selects_nothing": True,
            "route_guidance": False,
            "oracle_guidance": False,
        },
    )


@stage("generate.ugi.selected-tree-production-train.v1")
def train_selected_ugi_tree_production_stage(context: RunContext) -> StageResult:
    """Train one fresh seed of the calibration-selected Ugi tree Transformer."""

    from experiments.phase1.product_l1.training.ugi_joint_sparse_training import (
        train_ugi_joint_sparse,
    )
    from experiments.phase1.product_l1.training.ugi_tree_transformer_production import (
        build_production_training_config,
    )

    context.dependency("cache", "receipt")
    design = context.config()
    configured = design.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("selected-tree production design has no inputs")
    for label, expected in configured.items():
        pin = context.stage.inputs.get(label)
        if pin is None or pin.to_mapping() != expected:
            raise StageError(f"selected-tree production input changed: {label}")
        context.input(label)
    base = json.loads(context.input("base_config").read_text())
    ablation = json.loads(context.input("ablation_design").read_text())
    adjudication = json.loads(context.input("calibration_adjudication").read_text())
    chemistry_labels = {
        "assignments",
        "semantic_products",
        "semantic_atoms",
        "atom_vocabulary",
        "prepared_cache",
    }
    if set(base.get("inputs", {})) != chemistry_labels or any(
        base["inputs"][label] != configured.get(label) for label in chemistry_labels
    ):
        raise StageError("selected-tree base config and production chemistry inputs differ")
    training_seed, effective = build_production_training_config(
        design,
        base,
        ablation,
        adjudication,
        profile=context.profile,
        replicate=context.replicate,
    )
    runtime = effective.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("selected-tree training config and declared device differ")
    work = context.work_dir / "selected_tree_production_training"
    resume_training = _resume_training_work(context, work)
    effective_config_path = context.work_dir / "selected_tree_effective_config.json"
    expected_effective = json.dumps(effective, indent=2, sort_keys=True) + "\n"
    if effective_config_path.is_file():
        if effective_config_path.read_text() != expected_effective:
            raise StageError("persisted selected-tree effective config changed")
    else:
        write_json(effective_config_path, effective)
    result = train_ugi_joint_sparse(
        effective_config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_joint_sparse_training_result.v1",
        checkpoint_schema="phase1_ugi_joint_sparse_checkpoint.v1",
        progress_schema="phase1_ugi_joint_sparse_progress.v1",
    )
    _copy(effective_config_path, context.output_path("effective_config.json"))
    return StageResult(
        artifacts=(
            *artifacts,
            ProducedArtifact(
                "effective_config",
                "effective_config.json",
                "phase1_ugi_joint_sparse_training_config.v1",
            ),
        ),
        metrics={
            "completed_steps": int(result["selection"]["completed_steps"]),
            "training_records": int(result["training_partition"]["training_records"]),
            "training_seed": training_seed,
        },
        summary={
            "arm_id": "tree_relations_and_routing",
            "fixed_final_step": int(runtime["steps"]),
            "heldout_selects_nothing": True,
            "route_guidance": False,
            "oracle_guidance": False,
        },
    )


@stage("evaluate.ugi.tree-transformer-checkpoints.v1")
def evaluate_ugi_tree_transformer_checkpoints_stage(context: RunContext) -> StageResult:
    """Sample and assess every frozen checkpoint from one matched training arm."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (  # noqa: E501
        STATIC_INPUT_LABELS,
        run_checkpoint_calibration,
    )

    config = context.config()
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("checkpoint calibration config has no static inputs")
    for label in STATIC_INPUT_LABELS:
        pin = context.stage.inputs.get(label)
        if pin is None or configured.get(label) != pin.to_mapping():
            raise StageError(f"checkpoint calibration input changed: {label}")
        context.input(label)
    for label in ("training_result", "checkpoint_snapshots"):
        if label not in context.stage.inputs:
            raise StageError(f"checkpoint calibration lacks dynamic input: {label}")
    work = context.work_dir / "checkpoint_calibration"
    result = run_checkpoint_calibration(
        context.config_path,
        context.repo,
        work,
        training_result_path=context.input("training_result"),
        checkpoint_archive_path=context.input("checkpoint_snapshots"),
        profile=context.profile,
        resume=context.resume,
        device=context.resources.device,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path
        for path in work.rglob("*")
        if path.is_file() and "checkpoints" not in path.relative_to(work).parts
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("checkpoint_calibrations.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_tree_transformer_checkpoint_calibration.v1",
            ),
            ProducedArtifact(
                "checkpoint_calibrations",
                "checkpoint_calibrations.tar",
                "forge.ugi_tree_transformer_checkpoint_calibration_archive.v1",
                rows=len(result["checkpoint_steps"]),
            ),
        ),
        metrics={
            "checkpoints": len(result["checkpoint_steps"]),
            "programs_per_checkpoint": int(result["programs_per_checkpoint"]),
        },
        summary={
            "arm_id": result["arm_id"],
            "checkpoint_selection": "deferred_to_cross_arm_v0_adjudicator",
            "candidate_selection": False,
            "heldout_rows_used": False,
        },
    )


@stage("evaluate.ugi.selected-tree-production.v1")
def evaluate_selected_ugi_tree_production_stage(context: RunContext) -> StageResult:
    """Evaluate one fresh selected-tree seed on the frozen component-family stress draw."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_production import (
        STATIC_INPUT_LABELS,
        run_tree_transformer_production_evaluation,
    )

    config = context.config()
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("selected-tree production evaluation has no static inputs")
    for label in STATIC_INPUT_LABELS:
        pin = context.stage.inputs.get(label)
        if pin is None or configured.get(label) != pin.to_mapping():
            raise StageError(f"selected-tree evaluation input changed: {label}")
        context.input(label)
    checkpoint = context.dependency("train", "checkpoint_latest")
    effective_config = context.dependency("train", "effective_config")
    training_result = context.dependency("train", "result")
    effective = json.loads(effective_config.path.read_text())
    training_seed = int(effective.get("seed", -1))
    work = context.work_dir / "selected_tree_production_evaluation"
    result = run_tree_transformer_production_evaluation(
        context.config_path,
        context.repo,
        work,
        checkpoint_path=checkpoint.path,
        effective_training_config_path=effective_config.path,
        training_result_path=training_result.path,
        training_seed=training_seed,
        profile=context.profile,
        resume=context.resume,
        device=context.resources.device,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [path for path in (work / "details").rglob("*") if path.is_file()]
    _deterministic_tar(
        detail_files,
        context.output_path("evaluation_details.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_tree_transformer_production_evaluation.v1",
            ),
            ProducedArtifact(
                "evaluation_details",
                "evaluation_details.tar",
                "forge.ugi_tree_transformer_production_evaluation_archive.v1",
                rows=int(result["programs"]),
            ),
        ),
        metrics={
            "programs": int(result["programs"]),
            "training_seed": training_seed,
            "exact_l1_yield_per_attempt": float(result["metrics"]["exact_l1_yield_per_attempt"]),
        },
        summary={
            "arm_id": result["arm_id"],
            "checkpoint_step": int(result["checkpoint_step"]),
            "candidate_selection": False,
            "heldout_rows_used": True,
            "heldout_selects_nothing": True,
        },
    )


@stage("evaluate.ugi.tree-relational-edge-constrained-resampling.v1")
def evaluate_ugi_tree_relational_edge_constrained_resampling_stage(
    context: RunContext,
) -> StageResult:
    """Resample one frozen tree-Transformer checkpoint with role-edge masks."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_constrained_resampling import (  # noqa: E501
        run_tree_transformer_constrained_resampling,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "tree_relational_edge_constrained_resampling"
    result_path = work / "result.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text())
    else:
        result = run_tree_transformer_constrained_resampling(
            context.config_path,
            context.repo,
            work,
            profile=context.profile,
            device=context.resources.device,
        )
    if (
        result.get("schema_version") != "forge.ugi_tree_transformer_constrained_resampling.v1"
        or result.get("status") != "complete"
        or int(result.get("attempts", -1)) != 3072
        or result.get("retraining") is not False
        or result.get("repairs_or_retries") is not False
    ):
        raise StageError("constrained resampling did not satisfy the frozen full-run contract")

    _copy(result_path, context.output_path("result.json"))
    detail_files = [path for path in work.rglob("*") if path.is_file() and path != result_path]
    _deterministic_tar(
        detail_files,
        context.output_path("diagnostic_details.tar"),
        base=work,
    )
    metrics = result["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_tree_transformer_constrained_resampling.v1",
                rows=int(result["attempts"]),
            ),
            ProducedArtifact(
                "diagnostic_details",
                "diagnostic_details.tar",
                "forge.ugi_tree_transformer_constrained_resampling_archive.v1",
                rows=int(result["attempts"]),
            ),
        ),
        metrics={
            "attempts": int(result["attempts"]),
            "valid_fraction_per_attempt": float(metrics["valid_fraction_per_attempt"]),
            "exact_l1_yield_per_attempt": float(metrics["exact_l1_yield_per_attempt"]),
            "local_support_qualified_exact_l1_yield_per_attempt": float(
                metrics["local_support_qualified_exact_l1_yield_per_attempt"]
            ),
            "local_unsupported_exact_l1": int(
                result["failure_counts"]["local_unsupported_exact_l1"]
            ),
        },
        summary={
            "all_gates_pass": bool(result["all_gates_pass"]),
            "decision": str(result["decision"]),
            "frozen_checkpoint_reused": True,
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.v0-calibration-checkpoint.v1")
def evaluate_ugi_v0_calibration_checkpoint_stage(context: RunContext) -> StageResult:
    """Sample the frozen v0 checkpoint under the exact tree-Transformer calibration contract."""

    from experiments.phase1.product_l1.evaluation.ugi_tree_transformer_checkpoint_calibration import (  # noqa: E501
        STATIC_INPUT_LABELS,
        run_reference_checkpoint_calibration,
    )

    config = context.config()
    configured = config.get("inputs")
    if not isinstance(configured, dict):
        raise StageError("v0 calibration config has no static inputs")
    for label in STATIC_INPUT_LABELS:
        pin = context.stage.inputs.get(label)
        if pin is None or configured.get(label) != pin.to_mapping():
            raise StageError(f"v0 calibration input changed: {label}")
        context.input(label)
    if "reference_checkpoint" not in context.stage.inputs:
        raise StageError("v0 calibration lacks its reference checkpoint")
    work = context.work_dir / "v0_checkpoint_calibration"
    result = run_reference_checkpoint_calibration(
        context.config_path,
        context.repo,
        work,
        checkpoint_path=context.input("reference_checkpoint"),
        profile=context.profile,
        resume=context.resume,
        device=context.resources.device,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [path for path in work.rglob("*") if path.is_file()]
    _deterministic_tar(
        detail_files,
        context.output_path("checkpoint_calibrations.tar"),
        base=work,
    )
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_reference_checkpoint_calibration.v1",
            ),
            ProducedArtifact(
                "checkpoint_calibrations",
                "checkpoint_calibrations.tar",
                "forge.ugi_tree_transformer_checkpoint_calibration_archive.v1",
                rows=1,
            ),
        ),
        metrics={
            "checkpoints": 1,
            "programs_per_checkpoint": int(result["programs_per_checkpoint"]),
        },
        summary={
            "arm_id": "v0_reference",
            "checkpoint_selection": False,
            "candidate_selection": False,
            "heldout_rows_used": False,
        },
    )


@stage("evaluate.ugi.v0-current-program-comparison.v1")
def evaluate_ugi_v0_current_program_comparison_stage(context: RunContext) -> StageResult:
    """Run the frozen v0 and mixed Transformer on one ordered Ugi program draw."""

    from experiments.phase1.product_l1.evaluation.ugi_v0_current_program_comparison import (
        run_ugi_v0_current_program_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "ugi_v0_current_program_comparison"
    result = run_ugi_v0_current_program_comparison(
        context.config_path,
        context.repo,
        work,
        profile=context.profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    current = result["methods"]["forge_mixed_transformer_program_matched_seed0"]["metrics"]
    reference = result["methods"]["forge_v0_step_3000_program_matched"]["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_v0_current_program_comparison.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_v0_current_program_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "current_exact_l1": float(current["exact_l1_yield_per_attempt"]),
            "v0_exact_l1": float(reference["exact_l1_yield_per_attempt"]),
            "current_open_ended_exact_l1": float(
                current["unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"]
            ),
            "v0_open_ended_exact_l1": float(
                reference["unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"]
            ),
        },
        summary={
            "decision": str(result["decision"]),
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.transformer-morphology-projection-comparison.v1")
def evaluate_ugi_transformer_morphology_projection_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare both frozen mixed Transformers on one ordered full Ugi program draw."""

    from experiments.phase1.product_l1.evaluation.ugi_transformer_morphology_projection_comparison import (
        run_ugi_transformer_morphology_projection_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    work = context.work_dir / "ugi_transformer_morphology_projection_comparison"
    result = run_ugi_transformer_morphology_projection_comparison(
        context.config_path,
        context.repo,
        work,
        profile=context.profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    reduced_id = "forge_mixed_transformer_reduced_projection_seed0"
    full_id = "forge_mixed_transformer_full_role_morphology_seed0"
    reduced = result["methods"][reduced_id]["metrics"]
    full = result["methods"][full_id]["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.ugi_transformer_morphology_projection_comparison.v1",
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                "forge.ugi_transformer_morphology_projection_comparison_archive.v1",
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "reduced_exact_l1": float(reduced["exact_l1_yield_per_attempt"]),
            "full_exact_l1": float(full["exact_l1_yield_per_attempt"]),
            "exact_l1_delta": float(result["full_minus_reduced"]["exact_l1_yield_per_attempt"]),
        },
        summary={
            "decision": str(result["decision"]),
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


def _reaction_specialist_v0_comparison_stage(
    context: RunContext,
    *,
    topology_specialist: bool,
) -> StageResult:
    from experiments.phase1.product_l1.evaluation.reaction_specialist_ugi_v0_comparison import (
        run_reaction_specialist_ugi_v0_comparison,
    )

    config = context.config()
    require_config_inputs(context, config)
    if "specialize_ugi" in context.stage.needs:
        specialist_checkpoint = context.dependency("specialize_ugi", "checkpoint").path
    else:
        required_recovery = {"specialist_checkpoint", "specialist_result"}
        if not required_recovery.issubset(context.inputs):
            raise StageError(
                "standalone specialist comparison requires pinned checkpoint and result inputs"
            )
        specialist_checkpoint = context.input("specialist_checkpoint")
        if context.input("specialist_result") != specialist_checkpoint.with_name("result.json"):
            raise StageError("standalone specialist result must be adjacent to its checkpoint")
    work = context.work_dir / (
        "reaction_topology_specialist_ugi_v0_comparison"
        if topology_specialist
        else "reaction_specialist_ugi_v0_comparison"
    )
    result = run_reaction_specialist_ugi_v0_comparison(
        context.config_path,
        specialist_checkpoint,
        context.repo,
        work,
        profile=context.profile,
        device=context.resources.device,
        resume=context.resume,
    )
    _copy(work / "result.json", context.output_path("result.json"))
    detail_files = [
        path for path in work.rglob("*") if path.is_file() and path.name != "result.json"
    ]
    _deterministic_tar(
        detail_files,
        context.output_path("comparison_details.tar"),
        base=work,
    )
    specialist_id = (
        "forge_shared_plus_ugi_topology_specialist_exposure_matched_seed0"
        if topology_specialist
        else "forge_shared_plus_ugi_specialist_exposure_matched_seed0"
    )
    v0_id = "forge_v0_step_3000_program_matched"
    specialist = result["methods"][specialist_id]["metrics"]
    v0 = result["methods"][v0_id]["metrics"]
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                (
                    "forge.reaction_topology_specialist_ugi_v0_comparison_result.v2"
                    if topology_specialist
                    else "forge.reaction_specialist_ugi_v0_comparison_result.v1"
                ),
            ),
            ProducedArtifact(
                "comparison_details",
                "comparison_details.tar",
                (
                    "forge.reaction_topology_specialist_ugi_v0_comparison_archive.v2"
                    if topology_specialist
                    else "forge.reaction_specialist_ugi_v0_comparison_archive.v1"
                ),
                rows=int(result["programs_per_method"]),
            ),
        ),
        metrics={
            "programs_per_method": int(result["programs_per_method"]),
            "specialist_exact_l1": float(specialist["exact_l1_yield_per_attempt"]),
            "v0_exact_l1": float(v0["exact_l1_yield_per_attempt"]),
            "open_ended_exact_l1_delta": float(
                result["specialist_minus_v0"][
                    "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt"
                ]
            ),
        },
        summary={
            "decision": str(result["decision"]),
            "training_calls": 0,
            "repairs_or_retries": False,
            "candidate_selection": False,
            "route_or_oracle_calls": 0,
        },
    )


@stage("evaluate.ugi.reaction-specialist-v0-comparison.v1")
def evaluate_ugi_reaction_specialist_v0_comparison_stage(context: RunContext) -> StageResult:
    """Compare the exposure-matched Ugi specialist with v0 on one ordered program draw."""

    return _reaction_specialist_v0_comparison_stage(
        context,
        topology_specialist=False,
    )


@stage("evaluate.ugi.reaction-topology-specialist-v0-comparison.v2")
def evaluate_ugi_reaction_topology_specialist_v0_comparison_stage(
    context: RunContext,
) -> StageResult:
    """Compare exact program-coupled Ugi topology specialization with v0."""

    return _reaction_specialist_v0_comparison_stage(
        context,
        topology_specialist=True,
    )


@stage("generate.ugi.closure-train.v1")
def train_ugi_closure_stage(context: RunContext) -> StageResult:
    """Train the sparse closure scorer with resumable optimizer and RNG state."""

    from experiments.phase1.product_l1.training.ugi_closure_training import train_ugi_closure_scorer

    config = context.config()
    require_config_inputs(context, config)
    runtime = config.get(context.profile)
    if not isinstance(runtime, dict) or runtime.get("device") != context.resources.device:
        raise StageError("closure training config and declared device differ")
    work = context.work_dir / "closure_training"
    resume_training = _resume_training_work(context, work)
    result = train_ugi_closure_scorer(
        context.config_path,
        context.repo,
        work,
        smoke=context.profile == "smoke",
        overwrite=False,
        resume=resume_training,
    )
    artifacts = _publish_training_outputs(
        context,
        work,
        result_schema="phase1_ugi_sparse_closure_result.v2",
        checkpoint_schema="phase1_ugi_sparse_closure_checkpoint.v2",
        progress_schema="phase1_ugi_sparse_closure_progress.v1",
    )
    return StageResult(
        artifacts=artifacts,
        metrics={
            "best_step": int(result["selection"]["best_step"]),
            "completed_steps": int(result["selection"]["completed_steps"]),
        },
        summary={"selection_metric": result["selection"]["metric"]},
    )


@stage("generate.ugi.sample-shards.v1")
def sample_ugi_shards(context: RunContext) -> StageResult:
    """Sample verified contiguous program shards and merge them without retries."""

    from experiments.phase1.product_l1.sampling.ugi_joint_end_to_end_sampling import (
        joint_sampling_result_matches_request,
        sample_ugi_joint_end_to_end,
    )

    config = context.config()
    require_config_inputs(context, config)
    runtime = config.get("profiles", {}).get(context.profile)
    if not isinstance(runtime, dict):
        raise StageError(f"sampling config has no {context.profile!r} profile")
    if context.resources.device != "cpu":
        raise StageError("the qualified end-to-end sampler currently runs on CPU")
    program_count = int(runtime["program_count"])
    shard_size = int(runtime["shard_size"])
    if program_count < 1 or shard_size < 1:
        raise StageError("sampling program and shard counts must be positive")
    program_document = json.loads(context.input("matched_programs").read_text())
    available = program_document.get("samples")
    if not isinstance(available, list) or program_count > len(available):
        raise StageError(
            f"sampling requested {program_count} programs but only "
            f"{len(available) if isinstance(available, list) else 0} are available"
        )

    work = context.work_dir / "sampling_shards"
    work.mkdir(exist_ok=context.resume)
    terminal_mode = str(runtime["terminal_decoder_mode"])
    merged_rows: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    shard_results: list[Path] = []
    for shard_index, offset in enumerate(range(0, program_count, shard_size)):
        limit = min(shard_size, program_count - offset)
        flow_seed = context.derive_seed("flow", shard_index)
        terminal_seed = (
            context.derive_seed("terminal", shard_index) if terminal_mode != "argmax" else None
        )
        shard_dir = work / f"shard_{shard_index:05d}"
        result_path = shard_dir / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if not joint_sampling_result_matches_request(
                result,
                seed=flow_seed,
                program_offset=offset,
                program_limit=limit,
                terminal_decoder_mode=terminal_mode,
                terminal_decoder_seed=terminal_seed,
                terminal_temperature=float(runtime["terminal_temperature"]),
                checkpoint_filename=context.input("joint_checkpoint").name,
            ):
                raise StageError(f"persisted sampling shard {shard_index} changed")
        else:
            if shard_dir.exists():
                # Only result.json authenticates a completed shard.  Discard a directory left by
                # interruption before that atomic receipt and replay its fixed seed from scratch.
                shutil.rmtree(shard_dir)
            result = sample_ugi_joint_end_to_end(
                context.repo,
                shard_dir,
                joint_checkpoint_path=context.input("joint_checkpoint"),
                closure_checkpoint_path=context.input("closure_checkpoint"),
                matched_staged_result_path=context.input("matched_programs"),
                prepared_cache_path=(
                    context.input("prepared_cache") if "prepared_cache" in context.inputs else None
                ),
                sample_steps=int(runtime["sample_steps"]),
                batch_size=int(runtime["batch_size"]),
                seed=flow_seed,
                overwrite=False,
                maximum_adjacent_branch_runs=runtime["maximum_adjacent_branch_runs"],
                qualified_reactions_path=context.input("qualified_reactions"),
                evaluate_exact_l1_terminal_admission=True,
                terminal_decoder_mode=terminal_mode,
                terminal_decoder_seed=terminal_seed,
                terminal_temperature=float(runtime["terminal_temperature"]),
                program_offset=offset,
                program_limit=limit,
                reference_comparison_mode="deferred",
                render=False,
                record_timing=False,
            )
        rows = result.get("samples")
        if not isinstance(rows, list) or len(rows) != limit:
            raise StageError(f"sampling shard {shard_index} returned the wrong row count")
        for local_index, row in enumerate(rows):
            normalized = dict(row)
            global_index = offset + local_index
            normalized["pipeline_index"] = global_index
            normalized["structure_id"] = f"pipeline_generated_{global_index:06d}"
            merged_rows.append(normalized)
        digest = str(sha256_file(result_path))
        shard_results.append(result_path)
        shard_records.append(
            {
                "flow_seed": flow_seed,
                "offset": offset,
                "rows": limit,
                "sha256": digest,
                "shard": shard_index,
                "terminal_seed": terminal_seed,
            }
        )

    exact_l1 = sum(
        bool((row.get("l1_forward_verification") or {}).get("exact_product_reconstructed"))
        for row in merged_rows
    )
    terminal_valid = sum(bool(row.get("terminal_valid")) for row in merged_rows)
    valid = sum(bool(row.get("valid")) for row in merged_rows)
    merged = {
        "inputs": {label: pin.to_mapping() for label, pin in sorted(context.stage.inputs.items())},
        "profile": context.profile,
        "samples": merged_rows,
        "schema_version": "forge.phase1_ugi_sampling_result.v1",
        "shards": shard_records,
        "statistics": {
            "exact_l1": exact_l1,
            "requested": program_count,
            "returned": len(merged_rows),
            "terminal_valid": terminal_valid,
            "valid": valid,
        },
        "status": "complete",
    }
    write_json(context.output_path("result.json"), merged)
    _deterministic_tar(shard_results, context.output_path("shards.tar"), base=work)
    return StageResult(
        artifacts=(
            ProducedArtifact(
                "result",
                "result.json",
                "forge.phase1_ugi_sampling_result.v1",
                rows=len(merged_rows),
            ),
            ProducedArtifact(
                "shards",
                "shards.tar",
                "forge.sampling_shard_archive.v1",
                rows=len(shard_records),
            ),
        ),
        metrics={
            "exact_l1": exact_l1,
            "terminal_valid": terminal_valid,
            "valid": valid,
        },
        summary={
            "candidate_selection": False,
            "route_calls": 0,
            "oracle_calls": 0,
            "retries_or_repairs": False,
        },
    )


__all__ = [
    "freeze_phase1_corpus",
    "sample_ugi_shards",
    "train_ugi_closure_stage",
    "train_ugi_joint_stage",
    "verify_ugi_training_cache",
]
