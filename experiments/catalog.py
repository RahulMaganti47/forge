"""Explicit allow-list of FORGE experiment stage implementations."""

from __future__ import annotations

from pathlib import Path

_LOADED = False

SPECIFICATIONS = {
    "installation-smoke": "experiments/installation_smoke/experiment.json",
    "phase1-corpus": "experiments/phase1/corpus/experiment.json",
    "phase1-multireaction-corpus": "experiments/phase1/multireaction/experiment.json",
    "phase1-bl-lx-reaction-enumerated-expansion": (
        "experiments/phase1/multireaction/bl_lx_reaction_enumerated_expansion.json"
    ),
    "phase1-bl-lx-mixed-repeat-expansion": (
        "experiments/phase1/multireaction/bl_lx_mixed_repeat_expansion.json"
    ),
    "phase1-bl-lx-model-support-mixed-repeat-expansion": (
        "experiments/phase1/multireaction/bl_lx_mixed_repeat_expansion_v2.json"
    ),
    "phase1-multireaction-training-smoke": ("experiments/phase1/multireaction/training_smoke.json"),
    "phase1-multireaction-overfit": "experiments/phase1/multireaction/overfit.json",
    "phase1-reaction-program-transformer-overfit": (
        "experiments/phase1/multireaction/transformer_overfit.json"
    ),
    "phase1-reaction-program-transformer-overfit-v2": (
        "experiments/phase1/multireaction/transformer_overfit_v2.json"
    ),
    "phase1-reaction-program-transformer-overfit-v3": (
        "experiments/phase1/multireaction/transformer_overfit_v3.json"
    ),
    "phase1-transformer-synthesis-program-production-design": (
        "experiments/phase1/multireaction/transformer_production_design.json"
    ),
    "phase1-transformer-synthesis-program-production": (
        "experiments/phase1/multireaction/transformer_production.json"
    ),
    "phase1-transformer-synthesis-program-production-h100": (
        "experiments/phase1/multireaction/transformer_production_h100.json"
    ),
    "phase1-transformer-production-accelerator-benchmark-l4": (
        "experiments/phase1/multireaction/transformer_production_accelerator_benchmark_l4.json"
    ),
    "phase1-transformer-production-accelerator-benchmark-a100-40gb": (
        "experiments/phase1/multireaction/transformer_production_accelerator_benchmark_a100_40gb.json"
    ),
    "phase1-transformer-production-accelerator-benchmark-h100": (
        "experiments/phase1/multireaction/transformer_production_accelerator_benchmark_h100.json"
    ),
    "phase1-finite-component-catalogue-baseline": (
        "experiments/phase1/multireaction/finite_component_catalogue_baseline.json"
    ),
    "phase1-conditional-role-dependence": (
        "experiments/phase1/multireaction/conditional_role_dependence.json"
    ),
    "phase1-learned-inventory-selector": (
        "experiments/phase1/multireaction/learned_inventory_selector.json"
    ),
    "phase1-learned-inventory-selector-h100-preflight": (
        "experiments/phase1/multireaction/learned_inventory_selector_h100_preflight.json"
    ),
    "phase1-local-chemistry-support": (
        "experiments/phase1/multireaction/local_chemistry_support.json"
    ),
    "phase1-local-morphology-support-v2": (
        "experiments/phase1/multireaction/local_morphology_support_v2.json"
    ),
    "phase1-local-morphology-resampling-smoke-v2": (
        "experiments/phase1/multireaction/local_morphology_resampling_smoke_v2.json"
    ),
    "phase1-local-morphology-resampling-h100-preflight-v2": (
        "experiments/phase1/multireaction/local_morphology_resampling_h100_preflight_v2.json"
    ),
    "phase1-local-morphology-resampling-seed0-h100-v2": (
        "experiments/phase1/multireaction/local_morphology_resampling_seed0_h100_v2.json"
    ),
    "phase1-local-chemistry-resampling": (
        "experiments/phase1/multireaction/local_chemistry_resampling.json"
    ),
    "phase1-local-chemistry-resampling-h100-preflight": (
        "experiments/phase1/multireaction/local_chemistry_resampling_h100_preflight.json"
    ),
    "phase1-transformer-mechanism-study": (
        "experiments/phase1/multireaction/transformer_mechanism_study.json"
    ),
    "phase1-transformer-mechanism-study-h100-preflight": (
        "experiments/phase1/multireaction/transformer_mechanism_study_h100_preflight.json"
    ),
    "phase1-bl-lx-repair-calibration-smoke": (
        "experiments/phase1/multireaction/bl_lx_repair_calibration_smoke.json"
    ),
    "phase1-bl-lx-repair-calibration-h100": (
        "experiments/phase1/multireaction/bl_lx_repair_calibration_h100.json"
    ),
    "phase1-bl-core-constraint-calibration-smoke": (
        "experiments/phase1/multireaction/bl_core_constraint_calibration_smoke.json"
    ),
    "phase1-bl-core-constraint-calibration-h100": (
        "experiments/phase1/multireaction/bl_core_constraint_calibration_h100.json"
    ),
    "phase1-bl-core-constraint-calibration-h100-preflight": (
        "experiments/phase1/multireaction/bl_core_constraint_calibration_h100_preflight.json"
    ),
    "phase1-ugi-train-exposure-calibration-smoke": (
        "experiments/phase1/multireaction/ugi_train_exposure_calibration_smoke.json"
    ),
    "phase1-ugi-train-exposure-calibration-h100-preflight": (
        "experiments/phase1/multireaction/ugi_train_exposure_calibration_h100_preflight.json"
    ),
    "phase1-ugi-train-exposure-calibration-h100": (
        "experiments/phase1/multireaction/ugi_train_exposure_calibration_h100.json"
    ),
    "phase1-reaction-specialist-preflight-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_preflight_seed0_h100.json"
    ),
    "phase1-reaction-specialist-ugi-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_ugi_seed0_h100.json"
    ),
    "phase1-reaction-specialist-ugi-v0-evaluation-recovery-seed0-h100": (
        "experiments/phase1/multireaction/"
        "reaction_specialist_ugi_v0_evaluation_recovery_seed0_h100.json"
    ),
    "phase1-reaction-specialist-bl-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_bl_seed0_h100.json"
    ),
    "phase1-reaction-specialist-lx-seed0-h100": (
        "experiments/phase1/multireaction/reaction_specialist_lx_seed0_h100.json"
    ),
    "phase1-reaction-topology-specialist-ugi-h100-preflight-v2": (
        "experiments/phase1/multireaction/"
        "reaction_topology_specialist_ugi_h100_preflight_v2.json"
    ),
    "phase1-reaction-topology-specialist-ugi-smoke-v2": (
        "experiments/phase1/multireaction/" "reaction_topology_specialist_ugi_smoke_v2.json"
    ),
    "phase1-reaction-topology-specialist-ugi-seed0-h100-v2": (
        "experiments/phase1/multireaction/" "reaction_topology_specialist_ugi_seed0_h100_v2.json"
    ),
    "phase1-bl-core-constrained-production-smoke": (
        "experiments/phase1/multireaction/bl_core_constrained_production_smoke.json"
    ),
    "phase1-bl-core-constrained-production-h100-preflight": (
        "experiments/phase1/multireaction/bl_core_constrained_production_h100_preflight.json"
    ),
    "phase1-bl-core-constrained-production-h100": (
        "experiments/phase1/multireaction/bl_core_constrained_production_h100.json"
    ),
    "phase1-program-semantic-intervention": (
        "experiments/phase1/multireaction/program_semantic_intervention.json"
    ),
    "phase1-final-program-semantic-intervention-h100-preflight": (
        "experiments/phase1/multireaction/final_program_semantic_intervention_h100_preflight.json"
    ),
    "phase1-final-program-semantic-intervention": (
        "experiments/phase1/multireaction/final_program_semantic_intervention.json"
    ),
    "phase1-held-reaction-family-study": (
        "experiments/phase1/multireaction/held_reaction_family_study.json"
    ),
    "phase1-external-ugi-common-export": (
        "experiments/phase1/multireaction/external_ugi_common_export.json"
    ),
    "phase1-potency-study-corpus": "experiments/phase1/hela_potency/data.json",
    "phase1-shared-synthesis-program-representation": (
        "experiments/phase1/multireaction/shared_representation.json"
    ),
    "phase1-shared-synthesis-program-mixed-representation": (
        "experiments/phase1/multireaction/shared_mixed_representation.json"
    ),
    "phase1-shared-synthesis-program-mixed-training-design": (
        "experiments/phase1/multireaction/shared_mixed_training_design.json"
    ),
    "phase1-shared-synthesis-program-mixed-production-cache": (
        "experiments/phase1/multireaction/shared_mixed_production_cache.json"
    ),
    "phase1-shared-synthesis-program-mixed-training-smoke": (
        "experiments/phase1/multireaction/shared_mixed_training_smoke.json"
    ),
    "phase1-shared-synthesis-program-mixed-h100-training-profile": (
        "experiments/phase1/multireaction/shared_mixed_h100_training_profile.json"
    ),
    "phase1-shared-synthesis-program-mixed-h100-tf32-profile": (
        "experiments/phase1/multireaction/shared_mixed_h100_tf32_profile.json"
    ),
    "phase1-shared-synthesis-program-mixed-primary-seed0-h100": (
        "experiments/phase1/multireaction/shared_mixed_primary_seed0_h100.json"
    ),
    "phase1-shared-synthesis-program-mixed-primary-seed0-evaluation-h100": (
        "experiments/phase1/multireaction/shared_mixed_primary_seed0_evaluation_h100.json"
    ),
    "phase1-shared-mixed-role-morphology-smoke": (
        "experiments/phase1/multireaction/shared_mixed_role_morphology_smoke.json"
    ),
    "phase1-shared-mixed-role-morphology-seed0-h100": (
        "experiments/phase1/multireaction/shared_mixed_role_morphology_seed0_h100.json"
    ),
    "phase1-shared-bias-end-to-end-smoke": (
        "experiments/phase1/multireaction/shared_bias_end_to_end_smoke.json"
    ),
    "phase1-shared-bias-end-to-end-seed0-h100": (
        "experiments/phase1/multireaction/shared_bias_end_to_end_seed0_h100.json"
    ),
    "phase1-shared-bias-parallel-global-seed0-h100-v2": (
        "experiments/phase1/multireaction/shared_bias_parallel_global_seed0_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-global-seed0-evaluation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_global_seed0_evaluation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed0-evaluation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_program_role_seed0_evaluation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed0-core-saturation-h100-v2": (
        "experiments/phase1/multireaction/"
        "shared_bias_parallel_program_role_seed0_core_saturation_h100_v2.json"
    ),
    "phase1-shared-bias-parallel-program-role-seed0-h100-v2": (
        "experiments/phase1/multireaction/shared_bias_parallel_program_role_seed0_h100_v2.json"
    ),
    "phase1-shared-synthesis-program-integration": (
        "experiments/phase1/multireaction/shared_integration.json"
    ),
    "phase1-shared-synthesis-program-production-design": (
        "experiments/phase1/multireaction/shared_production_design.json"
    ),
    "phase1-shared-synthesis-program-production-cache": (
        "experiments/phase1/multireaction/shared_production_cache.json"
    ),
    "phase1-shared-synthesis-program-production-training": (
        "experiments/phase1/multireaction/shared_production_training.json"
    ),
    "phase1-shared-synthesis-program-production-accelerator-benchmark-a100-40gb": (
        "experiments/phase1/multireaction/" "shared_production_accelerator_benchmark_a100_40gb.json"
    ),
    "phase1-shared-synthesis-program-production-accelerator-benchmark-l4": (
        "experiments/phase1/multireaction/shared_production_accelerator_benchmark_l4.json"
    ),
    "phase1-sampling": "experiments/phase1/product_l1/sampling.json",
    "phase1-training-production": "experiments/phase1/product_l1/training_production.json",
    "phase1-training-smoke": "experiments/phase1/product_l1/training_smoke.json",
    "phase1-ugi-v0-port-transformer-smoke-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_smoke_v1.json"
    ),
    "phase1-ugi-v0-port-transformer-h100-preflight-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_h100_preflight_v1.json"
    ),
    "phase1-ugi-v0-port-transformer-seed0-h100-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_seed0_h100_v1.json"
    ),
    "phase1-ugi-v0-port-transformer-seed0-sampling-v1": (
        "experiments/phase1/product_l1/ugi_v0_port_transformer_seed0_sampling_v1.json"
    ),
    "phase1-ugi-tree-transformer-challenger-smoke-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_challenger_smoke_v1.json"
    ),
    "phase1-ugi-tree-transformer-calibration-program-draw-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_calibration_program_draw_v1.json"
    ),
    "phase1-ugi-tree-transformer-development-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_transformer_development_h100_v1.json"
    ),
    "phase1-ugi-tree-transformer-calibration-h100-preflight-v1": (
        "experiments/phase1/product_l1/" "ugi_tree_transformer_calibration_h100_preflight_v1.json"
    ),
    "phase1-ugi-tree-calibration-dense-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_dense_h100_v1.json"
    ),
    "phase1-ugi-tree-calibration-relations-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_relations_h100_v1.json"
    ),
    "phase1-ugi-tree-calibration-consistency-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_consistency_h100_v1.json"
    ),
    "phase1-ugi-tree-calibration-masking-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_calibration_masking_h100_v1.json"
    ),
    "phase1-ugi-v0-calibration-h100-v1": (
        "experiments/phase1/product_l1/ugi_v0_calibration_h100_v1.json"
    ),
    "phase1-ugi-v0-current-program-comparison-h100-preflight-v1": (
        "experiments/phase1/product_l1/" "ugi_v0_current_program_comparison_h100_preflight_v1.json"
    ),
    "phase1-ugi-v0-current-program-comparison-h100-v1": (
        "experiments/phase1/product_l1/ugi_v0_current_program_comparison_h100_v1.json"
    ),
    "phase1-ugi-transformer-morphology-projection-comparison-h100-v1": (
        "experiments/phase1/product_l1/"
        "ugi_transformer_morphology_projection_comparison_h100_v1.json"
    ),
    "phase1-ugi-tree-relational-production-h100-v1": (
        "experiments/phase1/product_l1/ugi_tree_relational_production_h100_v1.json"
    ),
    "phase1-ugi-tree-relational-edge-constrained-resampling-seed0-h100-v1": (
        "experiments/phase1/product_l1/"
        "ugi_tree_relational_edge_constrained_resampling_seed0_h100_v1.json"
    ),
}


def load_catalog() -> None:
    """Register the stage adapters shipped by this repository exactly once."""

    global _LOADED
    if _LOADED:
        return

    # Importing this allow-listed module applies its @stage decorators. Experiment JSON is never
    # allowed to name an import path or execute arbitrary Python.
    import experiments.phase1.hela_potency.stages  # noqa: F401
    import experiments.phase1.multireaction.stages  # noqa: F401
    import experiments.phase1.product_l1.stages  # noqa: F401

    _LOADED = True


def specification_paths(repo: Path) -> tuple[Path, ...]:
    """Return every allow-listed experiment specification in stable identifier order."""

    return tuple((repo / SPECIFICATIONS[name]).resolve() for name in sorted(SPECIFICATIONS))


def resolve_specification(repo: Path, value: str) -> Path:
    """Resolve an allow-listed identifier or an explicit repository-local JSON path."""

    if value in SPECIFICATIONS:
        path = repo / SPECIFICATIONS[value]
    else:
        candidate = Path(value)
        path = candidate if candidate.is_absolute() else repo / candidate
    resolved = path.resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as error:
        raise ValueError(f"experiment specification escapes the repository: {value!r}") from error
    if not resolved.is_file():
        raise FileNotFoundError(f"experiment specification not found: {resolved}")
    return resolved


__all__ = ["SPECIFICATIONS", "load_catalog", "resolve_specification", "specification_paths"]
