"""Matched assessment of the frozen v0 and Ugi-only Transformer sample ledgers.

The comparison is deliberately orchestration-only: it converts both native sampler outputs into
the same attempt ledger, runs the existing method-blind L1, route, realism and local-chemistry
assessors, and adds one train-fold-derived role-morphology audit over exact-L1 decompositions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from experiments.phase1.multireaction.common_assessment import run_common_ugi_assessment
from experiments.phase1.multireaction.lipid_realism_assessment import (
    run_lipid_realism_assessment,
)
from experiments.phase1.multireaction.local_chemistry_assessment import (
    run_local_chemistry_assessment,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import iter_jsonl, read_json_object, write_json, write_jsonl
from forge.model.common_ugi_benchmark import UGI_PROGRAM_ID, CommonUgiAttempt, write_attempt_ledger
from forge.model.local_chemistry_support import LocalChemistrySupport

CONFIG_SCHEMA = "forge.ugi_v0_transformer_assessment_config.v1"
RESULT_SCHEMA = "forge.ugi_v0_transformer_assessment.v1"
MORPHOLOGY_ATTEMPT_SCHEMA = "forge.ugi_role_morphology_attempt.v1"
METHOD_IDS = ("forge_v0_production", "forge_ugi_transformer")
PRECURSOR_ROLES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
TAIL_ROLES = frozenset({"isocyanide_tail", "oxoester_aldehyde_body_tail"})
PROGRAM_FIELDS = (
    "node_counts",
    "junction_budgets",
    "cycle_ranks",
    "attachment_counts",
)


class UgiV0TransformerAssessmentError(ValueError):
    """The frozen ledgers or comparison contract are not exactly comparable."""


def _canonical_program(value: object) -> tuple[tuple[int, ...], ...]:
    if not isinstance(value, Mapping) or set(value) != set(PROGRAM_FIELDS):
        raise UgiV0TransformerAssessmentError("sample morphology program fields changed")
    try:
        return tuple(tuple(int(item) for item in value[field]) for field in PROGRAM_FIELDS)
    except (TypeError, ValueError) as error:
        raise UgiV0TransformerAssessmentError("sample morphology program is malformed") from error


def _validate_sampling_config(config: Mapping[str, Any], *, expected_attempts: int) -> None:
    profiles = config.get("profiles")
    policy = config.get("policy")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get("full"), Mapping):
        raise UgiV0TransformerAssessmentError("sampler has no full profile")
    full = profiles["full"]
    required = {
        "program_count": expected_attempts,
        "sample_steps": 8,
        "maximum_adjacent_branch_runs": [2, 1, 1],
        "terminal_decoder_mode": "stochastic",
        "terminal_temperature": 1.0,
    }
    if any(full.get(key) != value for key, value in required.items()):
        raise UgiV0TransformerAssessmentError("sampler profile differs from the matched contract")
    if (
        not isinstance(policy, Mapping)
        or policy.get("retries_or_repairs") is not False
        or policy.get("exact_l1_terminal_admission") is not True
        or policy.get("candidate_selection") is not False
        or int(policy.get("route_calls", -1)) != 0
        or int(policy.get("oracle_calls", -1)) != 0
    ):
        raise UgiV0TransformerAssessmentError("sampler policy violates comparison guardrails")


def _validate_native_result(
    value: Mapping[str, Any],
    *,
    expected_attempts: int,
    expected_inputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    samples = value.get("samples")
    statistics = value.get("statistics")
    schema = value.get("schema_version")
    if (
        not isinstance(samples, list)
        or len(samples) != expected_attempts
        or not isinstance(statistics, Mapping)
    ):
        raise UgiV0TransformerAssessmentError("native sampler result is incomplete or malformed")
    if schema == "forge.phase1_ugi_sampling_result.v1":
        inputs = value.get("inputs")
        complete = (
            value.get("status") == "complete"
            and value.get("profile") == "full"
            and int(statistics.get("requested", -1)) == expected_attempts
            and int(statistics.get("returned", -1)) == expected_attempts
            and isinstance(inputs, Mapping)
        )
    elif schema == "phase1_ugi3_route_saturation_private_sample.v1":
        native_inputs = value.get("pinned_inputs")
        sampling = value.get("sampling")
        decoder = sampling.get("terminal_decoder") if isinstance(sampling, Mapping) else None
        complete = (
            value.get("status") == "sampled_once_headless"
            and int(statistics.get("samples", -1)) == expected_attempts
            and isinstance(native_inputs, Mapping)
            and isinstance(sampling, Mapping)
            and int(sampling.get("samples", -1)) == expected_attempts
            and int(sampling.get("matched_global_programs", -1)) == expected_attempts
            and int(sampling.get("sample_steps", -1)) == 8
            and sampling.get("maximum_adjacent_branch_runs") == [2, 1, 1]
            and int(sampling.get("terminal_tree_repairs", -1)) == 0
            and isinstance(decoder, Mapping)
            and decoder.get("mode") == "stochastic"
            and float(decoder.get("temperature", -1.0)) == 1.0
        )
        inputs = (
            {
                "joint_checkpoint": native_inputs.get("joint_checkpoint"),
                "closure_checkpoint": native_inputs.get("closure_checkpoint"),
                "matched_programs": native_inputs.get("program_draw"),
                "qualified_reactions": native_inputs.get("qualified_reactions"),
            }
            if isinstance(native_inputs, Mapping)
            else None
        )
    else:
        complete = False
        inputs = None
    if not complete or not isinstance(inputs, Mapping):
        raise UgiV0TransformerAssessmentError("native sampler result is incomplete or malformed")
    for label, expected in expected_inputs.items():
        observed = inputs.get(label)
        if not isinstance(observed, Mapping) or observed.get("sha256") != expected.get("sha256"):
            raise UgiV0TransformerAssessmentError(f"native sampler input changed: {label}")
    checked: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise UgiV0TransformerAssessmentError(f"sample {index} is malformed")
        smiles = sample.get("smiles")
        valid = sample.get("valid") is True
        if valid != (isinstance(smiles, str) and bool(smiles)):
            raise UgiV0TransformerAssessmentError(
                f"sample {index} has inconsistent native validity and molecular output"
            )
        if "pipeline_index" in sample and int(sample["pipeline_index"]) != index:
            raise UgiV0TransformerAssessmentError(f"sample index changed at row {index}")
        _canonical_program(sample.get("program"))
        checked.append(dict(sample))
    return checked


def _validate_pairing(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> None:
    if len(left) != len(right):
        raise UgiV0TransformerAssessmentError("native attempt denominators differ")
    identity_fields = ("product_id", "source_stratum", "held_role_class")
    for index, (first, second) in enumerate(zip(left, right, strict=True)):
        if _canonical_program(first.get("program")) != _canonical_program(second.get("program")):
            raise UgiV0TransformerAssessmentError(f"program mismatch at attempt {index}")
        for field in identity_fields:
            if first.get(field) != second.get(field):
                raise UgiV0TransformerAssessmentError(
                    f"matched-program identity {field} changed at attempt {index}"
                )


def native_samples_to_attempts(
    samples: Sequence[Mapping[str, Any]], *, method_id: str, seed_label: int
) -> tuple[CommonUgiAttempt, ...]:
    """Convert every native attempt, retaining failures in the denominator."""

    if not method_id or not method_id.strip():
        raise UgiV0TransformerAssessmentError("comparison method ID must be nonempty")
    return tuple(
        CommonUgiAttempt(
            method_id=method_id,
            seed=seed_label,
            attempt_index=index,
            status="generated" if row.get("valid") is True else "invalid",
            product_smiles=str(row["smiles"]) if row.get("valid") is True else None,
            method_visible_component_ids=(),
            generator_calls=1,
            reaction_calls=0,
            route_calls=0,
            oracle_calls=0,
            wall_seconds=0.0,
        )
        for index, row in enumerate(samples)
    )


def _parse_component(smiles: object, *, label: str) -> Chem.Mol:
    molecule = None
    with rdBase.BlockLogs():
        if isinstance(smiles, str) and smiles:
            molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiV0TransformerAssessmentError(f"{label} is not one valid component")
    return molecule


def _component_morphology(
    molecule: Chem.Mol,
    *,
    role: str,
    support: LocalChemistrySupport,
) -> dict[str, Any]:
    atoms = tuple(molecule.GetAtoms())
    heavy = molecule.GetNumHeavyAtoms()
    carbon = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    hetero = heavy - carbon
    bounds = support.component_support_bounds(UGI_PROGRAM_ID, role)
    cycles = tuple(molecule.GetRingInfo().AtomRings())
    unsupported_cycles = sum(
        not support.allows_role_cycle(
            UGI_PROGRAM_ID,
            ((role, molecule.GetAtomWithIdx(index).GetSymbol()) for index in cycle),
        )
        for cycle in cycles
    )
    oxygen_rings = sum(
        any(molecule.GetAtomWithIdx(index).GetAtomicNum() == 8 for index in cycle)
        for cycle in cycles
    )
    branch_atoms = sum(
        atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3 for atom in molecule.GetAtoms()
    )
    return {
        "heavy_atoms": heavy,
        "carbon_atoms": carbon,
        "heteroatoms": hetero,
        "branch_atoms": branch_atoms,
        "rings": len(cycles),
        "oxygen_containing_rings": oxygen_rings,
        "unsupported_role_ring_signatures": unsupported_cycles,
        "within_observed_hard_bounds": bounds.contains(
            heavy_atoms=heavy,
            carbon_atoms=carbon,
            heteroatoms=hetero,
        ),
        "within_training_q01_q99_diagnostic": (
            bounds.carbon_atoms_q01 <= carbon <= bounds.carbon_atoms_q99
            and bounds.heteroatoms_q01 <= hetero <= bounds.heteroatoms_q99
        ),
    }


def assess_role_morphology(
    assessed_attempts_path: Path,
    *,
    method_id: str,
    seed_label: int,
    support: LocalChemistrySupport,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Audit all exact-L1 traces under one shared, train-fold-derived role policy."""

    records = list(iter_jsonl(assessed_attempts_path))
    if not records or records.pop(0) != {
        "schema_version": "forge.common_ugi_assessed_attempts.v1",
        "rows": len(records),
    }:
        raise UgiV0TransformerAssessmentError("common assessed-attempt ledger header changed")
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter(attempts=len(records))
    role_totals: dict[str, Counter[str]] = {role: Counter(components=0) for role in PRECURSOR_ROLES}
    for index, source in enumerate(records):
        if (
            not isinstance(source, Mapping)
            or source.get("method_id") != method_id
            or source.get("seed") != seed_label
            or source.get("attempt_index") != index
        ):
            raise UgiV0TransformerAssessmentError(f"assessed attempt identity changed at {index}")
        product = None
        if source.get("valid") is True:
            product = _parse_component(
                source.get("canonical_smiles"), label=f"valid product at attempt {index}"
            )
        oxygen_oxygen = 0
        nitrogen_oxygen = 0
        small_oxygen_rings = 0
        if product is not None:
            totals["valid_products"] += 1
            for bond in product.GetBonds():
                symbols = {bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()}
                oxygen_oxygen += int(symbols == {"O"})
                nitrogen_oxygen += int(symbols == {"N", "O"})
            small_oxygen_rings = sum(
                len(cycle) in {3, 4}
                and any(product.GetAtomWithIdx(atom).GetAtomicNum() == 8 for atom in cycle)
                for cycle in product.GetRingInfo().AtomRings()
            )
            totals["products_with_oxygen_oxygen_bonds"] += int(oxygen_oxygen > 0)
            totals["products_with_nitrogen_oxygen_bonds"] += int(nitrogen_oxygen > 0)
            totals["products_with_three_or_four_membered_oxygen_rings"] += int(
                small_oxygen_rings > 0
            )

        exact = source.get("exact_l1_program") is True
        traces = source.get("exact_l1_traces") if exact else []
        if exact and (not isinstance(traces, list) or not traces):
            raise UgiV0TransformerAssessmentError(f"exact-L1 attempt {index} has no trace")
        trace_assessments: list[dict[str, Any]] = []
        for trace_index, trace in enumerate(traces):
            components = trace.get("components_by_role") if isinstance(trace, Mapping) else None
            if not isinstance(components, Mapping) or set(components) != set(PRECURSOR_ROLES):
                raise UgiV0TransformerAssessmentError(
                    f"exact-L1 trace roles changed at attempt {index}, trace {trace_index}"
                )
            by_role = {}
            for role, smiles in components.items():
                molecule = _parse_component(
                    smiles, label=f"attempt {index} trace {trace_index} role {role}"
                )
                morphology = _component_morphology(molecule, role=str(role), support=support)
                by_role[str(role)] = morphology
                role_summary = role_totals[str(role)]
                role_summary["components"] += 1
                role_summary["within_observed_hard_bounds"] += int(
                    morphology["within_observed_hard_bounds"]
                )
                role_summary["within_training_q01_q99_diagnostic"] += int(
                    morphology["within_training_q01_q99_diagnostic"]
                )
                role_summary["unsupported_role_ring_signatures"] += int(
                    morphology["unsupported_role_ring_signatures"]
                )
                role_summary["oxygen_containing_rings"] += int(
                    morphology["oxygen_containing_rings"]
                )
                role_summary["branch_atoms"] += int(morphology["branch_atoms"])
            tail_values = [by_role[role] for role in TAIL_ROLES]
            trace_assessments.append(
                {
                    "trace_index": trace_index,
                    "components_by_role": by_role,
                    "all_roles_within_observed_hard_bounds": all(
                        value["within_observed_hard_bounds"] for value in by_role.values()
                    ),
                    "all_role_ring_signatures_supported": all(
                        int(value["unsupported_role_ring_signatures"]) == 0
                        for value in by_role.values()
                    ),
                    "tails_within_observed_hard_bounds": all(
                        value["within_observed_hard_bounds"] for value in tail_values
                    ),
                    "tail_ring_signatures_supported": all(
                        int(value["unsupported_role_ring_signatures"]) == 0 for value in tail_values
                    ),
                    "oxygen_ring_outside_amine_head": any(
                        int(value["oxygen_containing_rings"]) > 0 for value in tail_values
                    ),
                }
            )
        if exact:
            totals["exact_l1_products"] += 1
            totals["exact_l1_products_with_any_fully_supported_trace"] += int(
                any(
                    trace["all_roles_within_observed_hard_bounds"]
                    and trace["all_role_ring_signatures_supported"]
                    for trace in trace_assessments
                )
            )
            totals["exact_l1_products_with_all_traces_fully_supported"] += int(
                all(
                    trace["all_roles_within_observed_hard_bounds"]
                    and trace["all_role_ring_signatures_supported"]
                    for trace in trace_assessments
                )
            )
            totals["exact_l1_products_with_any_supported_tail_trace"] += int(
                any(
                    trace["tails_within_observed_hard_bounds"]
                    and trace["tail_ring_signatures_supported"]
                    and not trace["oxygen_ring_outside_amine_head"]
                    for trace in trace_assessments
                )
            )
            totals["exact_l1_products_with_oxygen_ring_outside_amine_head"] += int(
                any(trace["oxygen_ring_outside_amine_head"] for trace in trace_assessments)
            )
        rows.append(
            {
                "schema_version": MORPHOLOGY_ATTEMPT_SCHEMA,
                "method_id": method_id,
                "seed": seed_label,
                "attempt_index": index,
                "valid": source.get("valid") is True,
                "exact_l1_program": exact,
                "oxygen_oxygen_bond_count": oxygen_oxygen,
                "nitrogen_oxygen_bond_count": nitrogen_oxygen,
                "three_or_four_membered_oxygen_ring_count": small_oxygen_rings,
                "trace_assessments": trace_assessments,
            }
        )
    denominator = len(rows)
    exact_count = totals["exact_l1_products"]
    assessment = {
        "method_id": method_id,
        "seed": seed_label,
        "attempts": denominator,
        "counts": dict(sorted(totals.items())),
        "fractions_per_attempt": {
            key: value / denominator
            for key, value in sorted(totals.items())
            if key not in {"attempts"}
        },
        "precision_among_exact_l1": {
            "any_fully_supported_trace": (
                totals["exact_l1_products_with_any_fully_supported_trace"] / exact_count
                if exact_count
                else None
            ),
            "all_traces_fully_supported": (
                totals["exact_l1_products_with_all_traces_fully_supported"] / exact_count
                if exact_count
                else None
            ),
            "any_supported_tail_trace": (
                totals["exact_l1_products_with_any_supported_tail_trace"] / exact_count
                if exact_count
                else None
            ),
        },
        "components_by_role": {
            role: {
                **dict(sorted(values.items())),
                "hard_bound_precision": (
                    values["within_observed_hard_bounds"] / values["components"]
                    if values["components"]
                    else None
                ),
                "q01_q99_diagnostic_precision": (
                    values["within_training_q01_q99_diagnostic"] / values["components"]
                    if values["components"]
                    else None
                ),
            }
            for role, values in sorted(role_totals.items())
        },
        "training_fold_only_policy": True,
        "complete_component_identities_stored_by_policy": False,
        "coverage_and_precision_reported_separately": True,
        "candidate_selection": False,
        "nonclaims": [
            "Training-supported role morphology is not synthesis success, stability, safety, activity or route closure.",
            "Absence from training support is an abstention, not proof of chemical impossibility.",
            "The q01-q99 component range is diagnostic and is not a generation or selection gate.",
        ],
    }
    return rows, assessment


def selected_ugi_metrics(
    common: Mapping[str, Any],
    realism: Mapping[str, Any],
    local: Mapping[str, Any],
    morphology: Mapping[str, Any],
) -> dict[str, float | int | None]:
    def optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    common_metrics = common["common_assessment"]["metrics"]
    realism_assessment = realism["assessment"]
    local_assessment = local["assessment"]
    manifold = realism_assessment["empirical_lipid_manifold"]
    molecular = realism_assessment["molecular_output"]
    return {
        "valid_fraction_per_attempt": float(common_metrics["valid_fraction"]),
        "exact_l1_yield_per_attempt": float(common_metrics["exact_l1_yield_per_attempt"]),
        "unique_exact_l1_products_per_attempt": float(
            common_metrics["unique_exact_l1_products_per_attempt"]
        ),
        "unique_open_ended_exact_l1_products_per_attempt": float(
            common_metrics["unique_open_ended_exact_l1_products_per_attempt"]
        ),
        "unique_whole_product_novel_exact_l1_products_per_attempt": float(
            common_metrics["unique_whole_product_novel_exact_l1_products_per_1000_attempts"]
        )
        / 1000.0,
        "unique_open_ended_whole_product_novel_exact_l1_products_per_attempt": float(
            common_metrics[
                "unique_open_ended_whole_product_novel_exact_l1_products_per_1000_attempts"
            ]
        )
        / 1000.0,
        "whole_product_novel_to_train_fraction": optional_float(
            common_metrics["whole_product_novel_to_train_fraction"]
        ),
        "component_novelty_fraction": optional_float(
            common_metrics["component_novelty_fraction"]
        ),
        "effective_component_count": optional_float(
            common_metrics["effective_component_count"]
        ),
        "held_component_exact_l1_products_per_1000_attempts": float(
            common_metrics["held_component_exact_l1_products_per_1000_attempts"]
        ),
        "mean_pairwise_ecfp4_distance": optional_float(
            common_metrics["mean_pairwise_ecfp4_distance"]
        ),
        "local_support_qualified_exact_l1_yield_per_attempt": float(
            local_assessment["local_support_qualified_exact_l1_yield_per_attempt"]
        ),
        "fingerprint_manifold_precision_per_attempt": float(
            manifold["fingerprint"]["precision_per_requested_attempt"]
        ),
        "fingerprint_manifold_coverage": float(manifold["fingerprint"]["coverage"]),
        "descriptor_manifold_precision_per_attempt": float(
            manifold["descriptor"]["precision_per_requested_attempt"]
        ),
        "descriptor_manifold_coverage": float(manifold["descriptor"]["coverage"]),
        "realism_c2st_auc": (
            float(manifold["classifier_two_sample"]["auc_mean"])
            if manifold["classifier_two_sample"]["status"] == "estimated"
            else None
        ),
        "realism_internal_diversity": optional_float(
            molecular["mean_pairwise_ecfp4_distance_among_unique"]
        ),
        "role_supported_exact_l1_yield_per_attempt": float(
            morphology["fractions_per_attempt"]["exact_l1_products_with_any_fully_supported_trace"]
        ),
        "tail_supported_exact_l1_yield_per_attempt": float(
            morphology["fractions_per_attempt"]["exact_l1_products_with_any_supported_tail_trace"]
        ),
        "small_oxygen_ring_product_fraction_per_attempt": float(
            morphology["fractions_per_attempt"].get(
                "products_with_three_or_four_membered_oxygen_rings", 0.0
            )
        ),
        "oxygen_oxygen_bond_product_fraction_per_attempt": float(
            morphology["fractions_per_attempt"].get("products_with_oxygen_oxygen_bonds", 0.0)
        ),
        "nitrogen_oxygen_bond_product_fraction_per_attempt": float(
            morphology["fractions_per_attempt"].get("products_with_nitrogen_oxygen_bonds", 0.0)
        ),
    }


def assess_native_ugi_method(
    native_rows: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    seed_label: int,
    repo: Path,
    output_dir: Path,
    common_ugi_assessment_config: Path,
    lipid_realism_config: Path,
    local_chemistry_config: Path,
    role_morphology_policy: Path,
    support: LocalChemistrySupport | None = None,
) -> dict[str, Any]:
    """Apply the shared assessors to one complete native sample ledger."""

    output_dir.mkdir(parents=True, exist_ok=False)
    attempts_path = output_dir / "attempts.jsonl.gz"
    write_attempt_ledger(
        attempts_path,
        native_samples_to_attempts(native_rows, method_id=method_id, seed_label=seed_label),
    )
    common_dir = output_dir / "common"
    common = run_common_ugi_assessment(
        common_ugi_assessment_config,
        repo,
        attempts_path,
        common_dir,
        method_id=method_id,
        seed=seed_label,
        expected_attempts=len(native_rows),
    )
    realism_dir = output_dir / "realism"
    realism = run_lipid_realism_assessment(
        lipid_realism_config,
        repo,
        attempts_path,
        realism_dir,
        method_id=method_id,
        seed=seed_label,
        expected_attempts=len(native_rows),
    )
    local_dir = output_dir / "local_chemistry"
    local = run_local_chemistry_assessment(
        local_chemistry_config,
        repo,
        common_dir / "assessed_attempts.jsonl.gz",
        local_dir,
        method_id=method_id,
        seed=seed_label,
        expected_attempts=len(native_rows),
    )
    resolved_support = support or LocalChemistrySupport.from_mapping(
        read_json_object(
            role_morphology_policy,
            error=UgiV0TransformerAssessmentError,
            label="role morphology policy",
        )
    )
    if not resolved_support.enforces_role_cycles:
        raise UgiV0TransformerAssessmentError("role morphology policy lacks complete cycles")
    morphology_rows, morphology = assess_role_morphology(
        common_dir / "assessed_attempts.jsonl.gz",
        method_id=method_id,
        seed_label=seed_label,
        support=resolved_support,
    )
    morphology_path = output_dir / "role_morphology_attempts.jsonl.gz"
    write_jsonl(
        morphology_path,
        [
            {"schema_version": MORPHOLOGY_ATTEMPT_SCHEMA, "rows": len(morphology_rows)},
            *morphology_rows,
        ],
    )
    morphology_result_path = output_dir / "role_morphology_result.json"
    morphology_result = {
        "schema_version": "forge.ugi_role_morphology_assessment.v1",
        "status": "pass",
        "method_id": method_id,
        "seed": seed_label,
        "attempts": artifact_record(attempts_path),
        "common_assessed_attempts": artifact_record(common_dir / "assessed_attempts.jsonl.gz"),
        "assessed_role_morphology": artifact_record(morphology_path),
        "policy": pin_record(role_morphology_policy, repo),
        "assessment": morphology,
        "candidate_selection": False,
    }
    write_json(morphology_result_path, morphology_result)
    return {
        "attempts": artifact_record(attempts_path),
        "common_assessment": artifact_record(common_dir / "result.json"),
        "lipid_realism": artifact_record(realism_dir / "result.json"),
        "local_chemistry": artifact_record(local_dir / "result.json"),
        "role_morphology": artifact_record(morphology_result_path),
        "metrics": selected_ugi_metrics(common, realism, local, morphology),
    }


def run_ugi_v0_transformer_assessment(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the complete frozen, method-blind v0-versus-Transformer assessment."""

    repo = repo.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise UgiV0TransformerAssessmentError(f"output directory is not empty: {output_dir}")
    config = read_json_object(
        config_path,
        error=UgiV0TransformerAssessmentError,
        label="v0 Transformer assessment config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != {
        "schema_version",
        "scientific_question",
        "expected_attempts",
        "seed_label",
        "inputs",
        "reporting",
        "nonclaims",
    }:
        raise UgiV0TransformerAssessmentError("assessment config fields changed")
    expected_attempts = int(config["expected_attempts"])
    seed_label = int(config["seed_label"])
    if expected_attempts != 3072 or seed_label != 0:
        raise UgiV0TransformerAssessmentError("frozen comparison denominator or seed label changed")
    reporting = config["reporting"]
    if reporting != {
        "candidate_selection": False,
        "single_seed_descriptive_comparison": True,
        "negative_results_are_reported": True,
        "reductive_amination_substructure_rate_reported": False,
    }:
        raise UgiV0TransformerAssessmentError("assessment reporting guardrails changed")
    raw_inputs = config["inputs"]
    expected_labels = {
        "v0_samples",
        "v0_sampler_config",
        "v0_training_config",
        "transformer_samples",
        "transformer_sampler_config",
        "transformer_training_config",
        "common_ugi_assessment_config",
        "lipid_realism_config",
        "local_chemistry_config",
        "role_morphology_policy",
    }
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != expected_labels:
        raise UgiV0TransformerAssessmentError("assessment input pins changed")
    inputs = {label: resolve_pin(pin, repo, label=label) for label, pin in raw_inputs.items()}
    v0_config = read_json_object(
        inputs["v0_sampler_config"],
        error=UgiV0TransformerAssessmentError,
        label="v0 sampler config",
    )
    transformer_config = read_json_object(
        inputs["transformer_sampler_config"],
        error=UgiV0TransformerAssessmentError,
        label="Transformer sampler config",
    )
    _validate_sampling_config(v0_config, expected_attempts=expected_attempts)
    _validate_sampling_config(transformer_config, expected_attempts=expected_attempts)
    v0_training = read_json_object(
        inputs["v0_training_config"],
        error=UgiV0TransformerAssessmentError,
        label="v0 training config",
    )
    transformer_training = read_json_object(
        inputs["transformer_training_config"],
        error=UgiV0TransformerAssessmentError,
        label="Transformer training config",
    )
    training_labels = {
        "assignments",
        "semantic_products",
        "semantic_atoms",
        "atom_vocabulary",
        "prepared_cache",
    }
    for value, label in (
        (v0_training, "v0"),
        (transformer_training, "Transformer"),
    ):
        if (
            value.get("schema_version") != "phase1_ugi_joint_sparse_training_config.v1"
            or not isinstance(value.get("inputs"), Mapping)
            or set(value["inputs"]) != training_labels
            or value.get("expected_fold_counts")
            != {"train": 66464, "calibration": 15800, "heldout": 30122}
            or value.get("training_partition", {}).get("training_folds") != ["train"]
        ):
            raise UgiV0TransformerAssessmentError(f"{label} training contract changed")
    if any(
        v0_training["inputs"][label]["sha256"] != transformer_training["inputs"][label]["sha256"]
        for label in training_labels
    ):
        raise UgiV0TransformerAssessmentError("v0 and Transformer training inputs differ")
    v0_full = v0_config["profiles"]["full"]
    transformer_full = transformer_config["profiles"]["full"]
    matched_profile_fields = (
        "program_count",
        "sample_steps",
        "maximum_adjacent_branch_runs",
        "terminal_decoder_mode",
        "terminal_temperature",
    )
    if any(v0_full[field] != transformer_full[field] for field in matched_profile_fields):
        raise UgiV0TransformerAssessmentError("native sampler profiles are not matched")
    shared_input_labels = ("closure_checkpoint", "matched_programs", "qualified_reactions")
    if any(
        v0_config["inputs"][label]["sha256"] != transformer_config["inputs"][label]["sha256"]
        for label in shared_input_labels
    ):
        raise UgiV0TransformerAssessmentError("native sampler shared inputs differ")
    v0_native = read_json_object(
        inputs["v0_samples"], error=UgiV0TransformerAssessmentError, label="v0 samples"
    )
    transformer_native = read_json_object(
        inputs["transformer_samples"],
        error=UgiV0TransformerAssessmentError,
        label="Transformer samples",
    )
    v0_rows = _validate_native_result(
        v0_native,
        expected_attempts=expected_attempts,
        expected_inputs=v0_config["inputs"],
    )
    transformer_rows = _validate_native_result(
        transformer_native,
        expected_attempts=expected_attempts,
        expected_inputs=transformer_config["inputs"],
    )
    _validate_pairing(v0_rows, transformer_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    support = LocalChemistrySupport.from_mapping(
        read_json_object(
            inputs["role_morphology_policy"],
            error=UgiV0TransformerAssessmentError,
            label="role morphology policy",
        )
    )
    if not support.enforces_role_cycles:
        raise UgiV0TransformerAssessmentError("role morphology policy lacks complete cycles")
    methods: dict[str, Any] = {}
    for method_id, native_rows in zip(METHOD_IDS, (v0_rows, transformer_rows), strict=True):
        methods[method_id] = assess_native_ugi_method(
            native_rows,
            method_id=method_id,
            seed_label=seed_label,
            repo=repo,
            output_dir=output_dir / method_id,
            common_ugi_assessment_config=inputs["common_ugi_assessment_config"],
            lipid_realism_config=inputs["lipid_realism_config"],
            local_chemistry_config=inputs["local_chemistry_config"],
            role_morphology_policy=inputs["role_morphology_policy"],
            support=support,
        )

    v0_metrics = methods[METHOD_IDS[0]]["metrics"]
    transformer_metrics = methods[METHOD_IDS[1]]["metrics"]
    deltas = {
        key: (
            float(transformer_metrics[key]) - float(v0_metrics[key])
            if transformer_metrics[key] is not None and v0_metrics[key] is not None
            else None
        )
        for key in v0_metrics
    }
    gates = {
        "attempt_denominator_3072_each": all(
            method["metrics"]["valid_fraction_per_attempt"] is not None
            for method in methods.values()
        ),
        "same_program_identity_at_every_attempt": True,
        "same_program_draw_closure_and_reaction_registry": True,
        "same_stochastic_terminal_decoder": True,
        "same_eight_step_no_repair_sampler": True,
        "same_training_inputs_and_train_fold": True,
        "method_blind_assessors_shared": True,
        "training_fold_only_role_morphology_policy": True,
        "candidate_selection_absent": True,
        "forbidden_reductive_amination_metric_absent": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete" if all(gates.values()) else "fail",
        "scientific_question": config["scientific_question"],
        "comparison_design": {
            "attempts_per_method": expected_attempts,
            "seed_label": seed_label,
            "program_matched": True,
            "random_stream_matched": False,
            "single_seed_descriptive": True,
            "training_inputs_matched": True,
            "training_budget_matched": False,
            "v0_optimizer_step": 3000,
            "transformer_optimizer_step": 5100,
            "variance_estimated_superiority_claim": False,
            "wall_time_comparison_admissible": False,
        },
        "inputs": {label: pin_record(path, repo) for label, path in sorted(inputs.items())},
        "methods": methods,
        "transformer_minus_v0": deltas,
        "gates": gates,
        "candidate_selection": False,
        "nonclaims": list(config["nonclaims"]),
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "complete":
        raise UgiV0TransformerAssessmentError(f"comparison gates failed: {gates}")
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "METHOD_IDS",
    "MORPHOLOGY_ATTEMPT_SCHEMA",
    "RESULT_SCHEMA",
    "UgiV0TransformerAssessmentError",
    "assess_native_ugi_method",
    "assess_role_morphology",
    "native_samples_to_attempts",
    "run_ugi_v0_transformer_assessment",
    "selected_ugi_metrics",
]
