"""Matched train-catalogue assembly baseline for the multi-reaction production study."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.assembly import RegistryRepeatedReactionProgram, Ugi3AssemblyAdapter
from forge.baselines.catalogue import (
    build_finite_component_catalogues,
    finite_component_catalogue_coverage,
    sample_finite_component_program,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import atomic_write, gzip_bytes, jsonl_bytes, read_json_object, write_json
from forge.corpus.reaction_program_training import load_reaction_program_specifications
from forge.evaluation.reaction_program import (
    adjudicate_reaction_program_rows,
    effective_count,
    evaluate_reaction_program_samples,
    load_reaction_program_training_references,
)
from forge.evaluation.ugi_benchmark import (
    ATTEMPT_SCHEMA as COMMON_ATTEMPT_SCHEMA,
)
from forge.evaluation.ugi_benchmark import (
    CommonUgiAttempt,
    write_attempt_ledger,
)

from .production_randomness import production_seed

CONFIG_SCHEMA = "forge.finite_component_catalogue_baseline_config.v1"
RESULT_SCHEMA = "forge.finite_component_catalogue_baseline_result.v1"
SAMPLES_SCHEMA = "forge.finite_component_catalogue_baseline_samples.v1"
UGI_PROGRAM = "ugi_3cr_agile"


class FiniteComponentCatalogueBaselineError(ValueError):
    """The matched finite-catalogue experiment violates its frozen contract."""


def _sampled_component_metrics(
    rows: list[dict[str, Any]],
    training_components: Mapping[str, Mapping[str, set[str]]],
) -> dict[str, dict[str, Any]]:
    """Measure the known input tuples without relying on a unique reverse decomposition."""

    output: dict[str, dict[str, Any]] = {}
    for program_id in sorted(training_components):
        values_by_role: dict[str, list[str]] = {
            role: [] for role in training_components[program_id]
        }
        novel = 0
        for row in rows:
            if row["program_id"] != program_id:
                continue
            sampled = row.get("sampled_components_by_role")
            if not isinstance(sampled, Mapping) or set(sampled) != set(values_by_role):
                raise FiniteComponentCatalogueBaselineError(
                    f"sampled component roles changed for {program_id}"
                )
            for role, raw in sampled.items():
                values = raw if isinstance(raw, list) else [raw]
                normalized = [str(value) for value in values]
                values_by_role[str(role)].extend(normalized)
                novel += sum(
                    value not in training_components[program_id][str(role)] for value in normalized
                )
        all_values = [value for role in sorted(values_by_role) for value in values_by_role[role]]
        output[program_id] = {
            "component_slots": len(all_values),
            "novel_component_slots": novel,
            "component_novelty_fraction": novel / len(all_values) if all_values else None,
            "effective_component_count": effective_count(all_values),
            "by_role": {
                role: {
                    "component_slots": len(values),
                    "effective_component_count": effective_count(values),
                }
                for role, values in sorted(values_by_role.items())
            },
            "identity_source": "known_sampled_train_catalogue_tuple",
        }
    return output


def _validate_policy(config: Mapping[str, Any], design: Mapping[str, Any], profile: str) -> int:
    if profile not in {"smoke", "full"} or not isinstance(config.get(profile), Mapping):
        raise FiniteComponentCatalogueBaselineError(f"unsupported baseline profile: {profile}")
    policy = config.get("policy")
    if not isinstance(policy, Mapping):
        raise FiniteComponentCatalogueBaselineError("catalogue baseline has no policy")
    expected = {
        "component_source": "train_fold_only",
        "component_sampling": "source_weighted_independent_role_marginals",
        "program_sampling": "equal_attempts_per_program",
        "outcome_resolution": "uniform_over_sorted_exact_forward_products",
        "products_per_attempt": 1,
        "repairs_or_retries": False,
        "route_calls": 0,
        "oracle_calls": 0,
        "candidate_selection": False,
    }
    if any(policy.get(key) != value for key, value in expected.items()):
        raise FiniteComponentCatalogueBaselineError("catalogue baseline policy changed")
    runtime = config[profile]
    attempts = runtime.get("attempts_per_program")
    maximum_outcomes = runtime.get("maximum_forward_outcomes")
    if (
        isinstance(attempts, bool)
        or not isinstance(attempts, int)
        or attempts < 1
        or isinstance(maximum_outcomes, bool)
        or not isinstance(maximum_outcomes, int)
        or maximum_outcomes < 2
    ):
        raise FiniteComponentCatalogueBaselineError("catalogue baseline budget is invalid")
    if profile == "full":
        expected_attempts = design["evaluation"]["native_sampling"][
            "heldout_samples_per_supported_program_at_final_checkpoint_per_seed"
        ]
        if attempts != expected_attempts:
            raise FiniteComponentCatalogueBaselineError(
                "full catalogue attempts do not match the model heldout attempt budget"
            )
    return attempts


def run_finite_component_catalogue_baseline(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    replicate: int,
) -> dict[str, Any]:
    """Execute one deterministic replicate of the strong catalogue-assembly baseline."""

    config = read_json_object(
        config_path,
        error=FiniteComponentCatalogueBaselineError,
        label="finite-component catalogue baseline config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise FiniteComponentCatalogueBaselineError("unsupported catalogue baseline config")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping):
        raise FiniteComponentCatalogueBaselineError("catalogue baseline has no input pins")
    required = {
        "production_design",
        "program_config",
        "qualified_reaction_families",
        "qualified_ugi_reactions",
        "ugi_assignments",
        "multireaction_atlas",
        "multireaction_splits",
    }
    if set(raw_inputs) != required:
        raise FiniteComponentCatalogueBaselineError(
            f"catalogue baseline inputs changed: {sorted(set(raw_inputs).symmetric_difference(required))}"
        )
    paths = {
        label: resolve_pin(pin, repo, label=label) for label, pin in sorted(raw_inputs.items())
    }
    design = read_json_object(
        paths["production_design"],
        error=FiniteComponentCatalogueBaselineError,
        label="production comparison design",
    )
    if design.get("schema_version") != "forge.synthesis_program_production_design_config.v1":
        raise FiniteComponentCatalogueBaselineError("production comparison design changed")
    attempts = _validate_policy(config, design, profile)
    seeds = design["training"]["replicate_seeds"]
    if (
        isinstance(replicate, bool)
        or not isinstance(replicate, int)
        or not 0 <= replicate < len(seeds)
    ):
        raise FiniteComponentCatalogueBaselineError(f"replicate must lie in [0, {len(seeds) - 1}]")
    seed = int(seeds[replicate])
    specs = {
        spec.program_id: spec
        for spec in load_reaction_program_specifications(paths["program_config"])
    }
    adapters: dict[str, Any] = {
        program_id: RegistryRepeatedReactionProgram.from_registry(
            paths["qualified_reaction_families"],
            spec,
            expected_sha256=str(raw_inputs["qualified_reaction_families"]["sha256"]),
        )
        for program_id, spec in specs.items()
    }
    ugi = Ugi3AssemblyAdapter.from_registry(
        paths["qualified_ugi_reactions"],
        expected_sha256=str(raw_inputs["qualified_ugi_reactions"]["sha256"]),
    )
    adapters[UGI_PROGRAM] = ugi
    catalogues = build_finite_component_catalogues(
        ugi_assignments=paths["ugi_assignments"],
        multireaction_atlas=paths["multireaction_atlas"],
        multireaction_splits=paths["multireaction_splits"],
        repeated_program_specs=specs,
        ugi_program_id=UGI_PROGRAM,
        ugi_roles=ugi.roles,
    )
    training_products, training_components = load_reaction_program_training_references(
        ugi_assignments=paths["ugi_assignments"],
        multireaction_atlas=paths["multireaction_atlas"],
        multireaction_splits=paths["multireaction_splits"],
        repeated_program_specs=specs,
        ugi_program_id=UGI_PROGRAM,
        ugi_roles=ugi.roles,
    )
    catalogue_coverage = finite_component_catalogue_coverage(
        catalogues,
        ugi_assignments=paths["ugi_assignments"],
        multireaction_atlas=paths["multireaction_atlas"],
        multireaction_splits=paths["multireaction_splits"],
        repeated_program_specs=specs,
        ugi_program_id=UGI_PROGRAM,
        ugi_roles=ugi.roles,
    )
    maximum_outcomes = int(config[profile]["maximum_forward_outcomes"])
    rows: list[dict[str, Any]] = []
    for program_id in sorted(catalogues):
        program_rows = sample_finite_component_program(
            catalogues[program_id],
            adapters[program_id],
            attempts=attempts,
            seed=production_seed(seed, "finite_component_catalogue", program_id),
            maximum_outcomes=maximum_outcomes,
        )
        for row in program_rows:
            row.update(
                {
                    "arm_id": "finite_component_catalogue_oracle",
                    "evaluation_split": "heldout_budget_train_catalogue_support",
                    "replicate": replicate,
                    "seed": seed,
                }
            )
        rows.extend(program_rows)
    adjudicate_reaction_program_rows(
        rows,
        adapters=adapters,
        repeated_program_specs=specs,
        ugi_program_id=UGI_PROGRAM,
    )
    evaluation = evaluate_reaction_program_samples(
        rows,
        training_products=training_products,
        training_components=training_components,
    )
    sampled_component_metrics = _sampled_component_metrics(rows, training_components)
    catalogue_summary = {
        program_id: {
            "assembly_kind": catalogue.assembly_kind,
            "role_component_counts": {
                role: len(support.values) for role, support in catalogue.role_supports
            },
            "supported_depths": [int(value) for value in catalogue.depth_support.values],
            "component_tuple_space_upper_bound": catalogue.tuple_space_upper_bound,
            "training_products": len(training_products[program_id]),
            **catalogue_coverage[program_id],
        }
        for program_id, catalogue in sorted(catalogues.items())
    }
    per_program = evaluation["per_program"]
    gates = {
        "three_programs_evaluated": set(per_program) == set(catalogues),
        "attempt_budget_matched": all(
            int(metrics["samples"]) == attempts for metrics in per_program.values()
        ),
        "train_catalogue_component_novelty_zero": all(
            metrics["component_novelty_fraction"] == 0.0
            for metrics in sampled_component_metrics.values()
        ),
        "all_valid_outputs_have_exact_l1_replay": all(
            int(metrics["valid"]) == int(metrics["exact_l1_program"])
            for metrics in per_program.values()
        ),
        "coverage_and_precision_reported": evaluation["coverage_and_precision_reported"] is True,
        "reductive_amination_substructure_rate_absent": (
            evaluation["reductive_amination_substructure_rate_reported"] is False
        ),
        "route_or_oracle_calls_zero": True,
        "candidate_selection_absent": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "arm_id": "finite_component_catalogue_oracle",
        "attempts_per_program": attempts,
        "policy": dict(config["policy"]),
        "catalogue": catalogue_summary,
        "metrics": evaluation,
        "sampled_component_metrics": sampled_component_metrics,
        "primary_metric": {
            "name": "unique_open_ended_exact_l1_products_per_1000_attempts",
            "definition": (
                "Distinct valid products with one exact L1 decomposition containing at least one "
                "component constitution absent from the train-fold catalogue, divided by all "
                "component-tuple attempts and multiplied by 1000."
            ),
            "interpretation": (
                "The finite-catalogue baseline is zero by construction; exact-L1 yield, product "
                "novelty and diversity are reported separately to expose the validity-breadth tradeoff."
            ),
        },
        "gates": gates,
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "implementation": pin_record(Path(__file__), repo),
        "calls": {"route": 0, "oracle": 0},
        "candidate_selection": False,
        "nonclaims": [
            "Catalogue assembly is not an open-ended component generator.",
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
            "The tuple-space value is an upper bound, not a count of unique valid products.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    visible_components = tuple(
        f"{role}:{smiles}"
        for role, values in sorted(training_components[UGI_PROGRAM].items())
        for smiles in sorted(values)
    )
    ugi_rows = [row for row in rows if row["program_id"] == UGI_PROGRAM]
    common_attempts = tuple(
        CommonUgiAttempt(
            method_id="finite_catalogue_oracle",
            seed=seed,
            attempt_index=index,
            status="generated" if row.get("valid") is True else "failed",
            product_smiles=str(row["canonical_smiles"]) if row.get("valid") is True else None,
            method_visible_component_ids=visible_components,
            generator_calls=1,
            reaction_calls=1,
            route_calls=0,
            oracle_calls=0,
            wall_seconds=0.0,
        )
        for index, row in enumerate(ugi_rows)
    )
    common_attempts_path = output_dir / "ugi_attempts.jsonl.gz"
    write_attempt_ledger(common_attempts_path, common_attempts)
    result["common_ugi_attempts"] = {
        **artifact_record(common_attempts_path),
        "method_id": "finite_catalogue_oracle",
        "seed": seed,
        "attempts": len(common_attempts),
        "wall_seconds_measurement": "not_recorded_do_not_interpret_zero",
    }
    records = [{"schema_version": SAMPLES_SCHEMA, "rows": len(rows)}, *rows]
    atomic_write(output_dir / "samples.jsonl.gz", gzip_bytes(jsonl_bytes(records)))
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "COMMON_ATTEMPT_SCHEMA",
    "RESULT_SCHEMA",
    "SAMPLES_SCHEMA",
    "FiniteComponentCatalogueBaselineError",
    "run_finite_component_catalogue_baseline",
]
