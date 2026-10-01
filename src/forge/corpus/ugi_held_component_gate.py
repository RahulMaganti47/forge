"""Evaluate held-component and generated-component novelty for Ugi checkpoints."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

from forge.corpus.r0_splits import sha256_file
from forge.corpus.r1_prime_audit import compile_reactions, load_reaction_definitions
from forge.corpus.training_cache import load_ugi_training_cache
from forge.corpus.ugi_component_expansion import reaction_handle_qualification
from forge.corpus.ugi_generated_components import generated_ugi_component_smiles
from forge.model.ugi_chemistry_flow import UgiChemistrySample
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    collate_ugi_joint_sparse_records,
    noise_ugi_joint_sparse_batch,
    ugi_joint_sparse_loss,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


SCHEMA_VERSION = "phase1_ugi_held_component_gate.v1"


class UgiHeldComponentGateError(RuntimeError):
    """Raised when the held-component gate violates a frozen invariant."""


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_molecule(smiles: str) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiHeldComponentGateError(f"invalid generated component: {smiles!r}")
    return molecule


def _target_sample(target: Any) -> UgiChemistrySample:
    return UgiChemistrySample(
        atom_states=target.atom_states.copy(),
        parent_bond_states=target.parent_bond_states.copy(),
        closure_bond_states=target.closure_bond_states.copy(),
        decoration_anchor=(
            int(target.decorations.anchor_indices[0]) + 1 if target.decorations.count else 0
        ),
        decoration_anchors=target.decorations.anchor_indices.copy() + 1,
        decoration_atom_states=target.decorations.atom_states.copy(),
        decoration_bond_states=target.decorations.bond_states.copy(),
    )


def audit_reference_component_inverse(corpus: Any) -> dict[str, Any]:
    """Cover every admitted component with the exact inverse-Ugi implementation."""

    seen: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    checked_products: list[str] = []
    for fold in ("train", "calibration", "heldout"):
        assignments = corpus.assignments_by_fold[fold]
        records = corpus.records_by_fold[fold]
        for assignment, record in zip(assignments, records, strict=True):
            if all(str(assignment[f"{role}_smiles"]) in seen[role] for role in ROLE_NAMES):
                continue
            observed = generated_ugi_component_smiles(
                record.condition,
                _target_sample(record.target),
                corpus.atom_vocabulary,
            )
            for role in ROLE_NAMES:
                expected = str(assignment[f"{role}_smiles"])
                if observed[role] != expected:
                    raise UgiHeldComponentGateError(
                        f"inverse-Ugi mismatch for {record.product_id}/{role}: "
                        f"{expected!r} != {observed[role]!r}"
                    )
                seen[role].add(expected)
            checked_products.append(record.product_id)
    expected = {
        role: {
            str(assignment[f"{role}_smiles"])
            for fold in ("train", "calibration", "heldout")
            for assignment in corpus.assignments_by_fold[fold]
        }
        for role in ROLE_NAMES
    }
    if seen != expected:
        raise UgiHeldComponentGateError("inverse-Ugi audit did not cover every component")
    return {
        "status": "pass",
        "products_checked": len(checked_products),
        "unique_components_covered_by_role": {role: len(values) for role, values in seen.items()},
        "exact_component_reconstruction_fraction": 1.0,
    }


def held_role_class(assignment: Mapping[str, Any]) -> str:
    held = tuple(
        role for role in ROLE_NAMES if str(assignment.get(f"{role}_family_fold")) == "heldout"
    )
    if not held:
        raise UgiHeldComponentGateError("heldout product has no heldout component family")
    return "+".join(held)


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _fingerprint(molecule: Chem.Mol) -> Any:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
    )
    return generator.GetFingerprint(molecule)


def _nearest_similarity(fingerprint: Any, references: Sequence[Any]) -> float:
    if not references:
        return 0.0
    return float(max(DataStructs.BulkTanimotoSimilarity(fingerprint, list(references))))


def _catalog_by_role(registry_path: Path) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {role: {} for role in ROLE_NAMES}
    for row in _read_csv(registry_path):
        if row.get("l1_structural_admission") != "true":
            continue
        role = row["role"]
        if role not in catalog:
            continue
        smiles = row["canonical_smiles"]
        fold = row["family_fold"]
        if fold not in {"train", "calibration", "heldout"}:
            raise UgiHeldComponentGateError("admitted component lacks a frozen family fold")
        catalog[role][smiles] = fold
    return catalog


def _reaction_contract(qualified_reactions_path: Path) -> Any:
    value = json.loads(qualified_reactions_path.read_text())
    definitions = load_reaction_definitions(
        [qualified_reactions_path], expected_count=len(value["reactions"])
    )
    compiled = {item.definition.reaction_id: item for item in compile_reactions(definitions)}
    return compiled["ugi_3cr_agile"]


def load_ugi_reaction_contract(qualified_reactions_path: Path) -> Any:
    """Load the frozen qualified Ugi reaction contract for public reuse."""

    return _reaction_contract(qualified_reactions_path)


def _forward_reconstructs_product(
    reaction: Any,
    components: Mapping[str, str],
    product_smiles: str,
    *,
    maximum_outcomes: int = 64,
) -> tuple[bool, bool, int]:
    reactants = tuple(_canonical_molecule(components[role]) for role in ROLE_NAMES)
    with rdBase.BlockLogs():
        outcomes = reaction.forward.RunReactants(reactants, maxProducts=maximum_outcomes)
    saturated = len(outcomes) >= maximum_outcomes
    products = set()
    for outcome in outcomes:
        if len(outcome) != 1:
            continue
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(outcome[0])
            products.add(Chem.MolToSmiles(outcome[0], canonical=True, isomericSmiles=False))
        except (ValueError, RuntimeError):
            continue
    return product_smiles in products, saturated, len(outcomes)


def exact_forward_reconstructs_ugi_product(
    reaction: Any,
    components: Mapping[str, str],
    product_smiles: str,
    *,
    maximum_outcomes: int = 64,
) -> tuple[bool, bool, int]:
    """Verify that one role-qualified precursor tuple exactly rebuilds a product."""

    return _forward_reconstructs_product(
        reaction,
        components,
        product_smiles,
        maximum_outcomes=maximum_outcomes,
    )


def analyze_generated_components(
    sampling_result_path: Path,
    registry_path: Path,
    qualified_reactions_path: Path,
) -> dict[str, Any]:
    """Measure actual generated component identity, novelty and handle qualification."""

    result = json.loads(sampling_result_path.read_text())
    catalog = _catalog_by_role(registry_path)
    reaction = _reaction_contract(qualified_reactions_path)
    role_order = tuple(role.name for role in reaction.definition.reactant_roles)
    if role_order != ROLE_NAMES:
        raise UgiHeldComponentGateError("qualified Ugi role order changed")
    reference_fingerprints: dict[str, dict[str, tuple[Any, ...]]] = {}
    for role in ROLE_NAMES:
        by_fold = {
            fold: tuple(
                _fingerprint(_canonical_molecule(smiles))
                for smiles, observed_fold in catalog[role].items()
                if observed_fold == fold
            )
            for fold in ("train", "calibration", "heldout")
        }
        by_fold["all"] = tuple(
            _fingerprint(_canonical_molecule(smiles)) for smiles in catalog[role]
        )
        reference_fingerprints[role] = by_fold

    membership_counts: dict[str, Counter[str]] = {role: Counter() for role in ROLE_NAMES}
    similarities: dict[str, dict[str, list[float]]] = {
        role: {"train": [], "all": [], "outside_train": []} for role in ROLE_NAMES
    }
    handle_counts: dict[str, Counter[str]] = {role: Counter() for role in ROLE_NAMES}
    handle_failure_examples: dict[str, list[dict[str, Any]]] = {role: [] for role in ROLE_NAMES}
    unique_generated: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    unique_outside_catalog: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    product_classes: Counter[str] = Counter()
    held_role_classes: Counter[str] = Counter()
    exact_train_triples = 0
    all_handles_pass = 0
    forward_reconstructed = 0
    forward_outcome_saturated = 0
    forward_outcome_counts: list[float] = []
    reconstruction_valid = 0
    valid_products = 0
    cached: dict[tuple[str, str], tuple[str, float, float, dict[str, Any]]] = {}

    for row in result["samples"]:
        if not row.get("valid"):
            continue
        valid_products += 1
        if not row.get("component_reconstruction_valid"):
            continue
        components = row.get("component_smiles_by_role")
        if not isinstance(components, dict):
            continue
        reconstruction_valid += 1
        held_role_classes[str(row.get("held_role_class") or "unspecified")] += 1
        folds = []
        handles_pass = True
        for role_index, role in enumerate(ROLE_NAMES):
            smiles = str(components[role])
            key = (role, smiles)
            if key not in cached:
                molecule = _canonical_molecule(smiles)
                fingerprint = _fingerprint(molecule)
                fold = catalog[role].get(smiles, "outside_admitted_catalog")
                train_similarity = _nearest_similarity(
                    fingerprint, reference_fingerprints[role]["train"]
                )
                all_similarity = _nearest_similarity(
                    fingerprint, reference_fingerprints[role]["all"]
                )
                qualification = reaction_handle_qualification(
                    molecule,
                    query=reaction.handles[role_index],
                    forbidden=reaction.forbidden[role_index],
                    allowed_site_multiplicity=reaction.definition.reactant_roles[
                        role_index
                    ].allowed_site_multiplicity,
                )
                cached[key] = (fold, train_similarity, all_similarity, qualification)
            fold, train_similarity, all_similarity, qualification = cached[key]
            folds.append(fold)
            membership_counts[role][fold] += 1
            similarities[role]["train"].append(train_similarity)
            similarities[role]["all"].append(all_similarity)
            unique_generated[role].add(smiles)
            if fold == "outside_admitted_catalog":
                similarities[role]["outside_train"].append(train_similarity)
                unique_outside_catalog[role].add(smiles)
            passes = bool(qualification["passes_registry_handle_policy"])
            handle_counts[role]["pass" if passes else "fail"] += 1
            handle_counts[role][
                f"distinct_sites_{qualification['symmetry_distinct_handle_sites']}"
            ] += 1
            if not passes and len(handle_failure_examples[role]) < 5:
                handle_failure_examples[role].append(
                    {
                        "smiles": smiles,
                        **qualification,
                    }
                )
            handles_pass = handles_pass and passes
        if "outside_admitted_catalog" in folds:
            product_class = "genuinely_generated_component"
        elif "heldout" in folds:
            product_class = "exact_heldout_component"
        elif "calibration" in folds:
            product_class = "exact_calibration_component"
        else:
            product_class = "train_catalog_only"
            exact_train_triples += 1
        product_classes[product_class] += 1
        all_handles_pass += int(handles_pass)
        recovered, saturated, outcome_count = _forward_reconstructs_product(
            reaction,
            components,
            str(row["smiles"]),
        )
        forward_reconstructed += int(recovered)
        forward_outcome_saturated += int(saturated)
        forward_outcome_counts.append(float(outcome_count))

    denominator = reconstruction_valid or 1
    role_summary = {}
    for role in ROLE_NAMES:
        role_total = sum(membership_counts[role].values()) or 1
        role_summary[role] = {
            "samples": sum(membership_counts[role].values()),
            "unique_generated_components": len(unique_generated[role]),
            "unique_outside_admitted_catalog_components": len(unique_outside_catalog[role]),
            "membership_counts": dict(sorted(membership_counts[role].items())),
            "membership_fractions": {
                key: value / role_total for key, value in sorted(membership_counts[role].items())
            },
            "nearest_train_ecfp4_tanimoto": _quantiles(similarities[role]["train"]),
            "nearest_all_admitted_ecfp4_tanimoto": _quantiles(similarities[role]["all"]),
            "outside_catalog_nearest_train_ecfp4_tanimoto": _quantiles(
                similarities[role]["outside_train"]
            ),
            "outside_catalog_nearest_train_similarity_threshold_fractions": {
                f"below_{threshold:.1f}": (
                    sum(value < threshold for value in similarities[role]["outside_train"])
                    / len(similarities[role]["outside_train"])
                    if similarities[role]["outside_train"]
                    else 0.0
                )
                for threshold in (0.5, 0.7, 0.8, 0.9)
            },
            "handle_policy_counts": dict(sorted(handle_counts[role].items())),
            "handle_policy_pass_fraction": handle_counts[role]["pass"] / role_total,
            "handle_policy_failure_examples": handle_failure_examples[role],
        }
    return {
        "sampling_result": {
            "path": _display_path(sampling_result_path),
            "sha256": sha256_file(sampling_result_path),
        },
        "valid_products": valid_products,
        "component_reconstruction_valid": reconstruction_valid,
        "component_reconstruction_valid_fraction_of_valid_products": (
            reconstruction_valid / valid_products if valid_products else 0.0
        ),
        "product_component_novelty_counts": dict(sorted(product_classes.items())),
        "product_component_novelty_fractions": {
            key: value / denominator for key, value in sorted(product_classes.items())
        },
        "product_component_novelty_definitions": {
            "genuinely_generated_component": (
                "at least one inverse-Ugi precursor graph is absent from all 424 admitted "
                "L1 components; this is structural component novelty and does not imply "
                "product novelty, route closure or synthesis success"
            ),
            "exact_heldout_component": (
                "no outside-catalog precursor and at least one exact heldout-family component"
            ),
            "exact_calibration_component": (
                "no outside/heldout precursor and at least one exact calibration component"
            ),
            "train_catalog_only": "all three precursor graphs are exact training components",
        },
        "exact_train_component_triples": exact_train_triples,
        "all_three_handles_pass": all_handles_pass,
        "all_three_handles_pass_fraction": all_handles_pass / denominator,
        "exact_forward_product_reconstruction": forward_reconstructed,
        "exact_forward_product_reconstruction_fraction": (forward_reconstructed / denominator),
        "forward_outcome_saturated": forward_outcome_saturated,
        "forward_outcome_count": _quantiles(forward_outcome_counts),
        "held_role_program_counts": dict(sorted(held_role_classes.items())),
        "by_role": role_summary,
        "sampling_statistics": result["statistics"],
        "reference_comparison": result["reference_comparison"],
    }


def _stable_group_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _move(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def heldout_loss_by_component_role(
    checkpoint_path: Path,
    corpus: Any,
    joint_records: Mapping[str, tuple[Any, ...]],
    *,
    seed: int,
    batch_size: int,
    product_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate a frozen heldout set once under deterministic corruption."""

    if torch is None:
        raise UgiHeldComponentGateError("heldout loss requires torch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != "phase1_ugi_joint_sparse_checkpoint.v1":
        raise UgiHeldComponentGateError("unsupported joint checkpoint")
    architecture = dict(checkpoint["model_config"])
    source_floor = architecture.pop("source_probability_floor")
    if not 0 < float(source_floor) < 1:
        raise UgiHeldComponentGateError("checkpoint source floor is invalid")
    model = UgiJointSparseFlow(
        atom_classes=len(corpus.atom_vocabulary),
        **architecture,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    sources = {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in checkpoint["source_marginals"].items()
    }
    requested = set(product_ids) if product_ids is not None else None
    grouped: defaultdict[str, list[Any]] = defaultdict(list)
    selected_ids: set[str] = set()
    for assignment, record in zip(
        corpus.assignments_by_fold["heldout"],
        joint_records["heldout"],
        strict=True,
    ):
        if requested is not None and record.product_id not in requested:
            continue
        grouped[held_role_class(assignment)].append(record)
        selected_ids.add(record.product_id)
    if requested is not None and selected_ids != requested:
        missing = sorted(requested - selected_ids)
        raise UgiHeldComponentGateError(
            f"heldout loss probe contains unknown product IDs: {missing[:3]}"
        )

    start = time.perf_counter()
    groups = {}
    global_weighted: defaultdict[str, float] = defaultdict(float)
    total_records = 0
    with torch.no_grad():
        for label, records in sorted(grouped.items()):
            generator = torch.Generator(device="cpu").manual_seed(_stable_group_seed(seed, label))
            weighted: defaultdict[str, float] = defaultdict(float)
            observed = 0
            for start_index in range(0, len(records), batch_size):
                local = tuple(records[start_index : start_index + batch_size])
                batch = _move(
                    collate_ugi_joint_sparse_records(
                        local,
                        maximum_nodes=max(record.node_count for record in local),
                        maximum_children=int(architecture["maximum_children"]),
                        maximum_closures=int(architecture["maximum_cycle_rank"]) * 3,
                        maximum_decorations=int(architecture["maximum_decorations"]),
                    ),
                    "cpu",
                )
                t = torch.rand(len(local), generator=generator).clamp(0.02, 0.98)
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
                _, metrics = ugi_joint_sparse_loss(predictions, batch)
                for key, value in metrics.items():
                    weighted[key] += float(value) * len(local)
                    global_weighted[key] += float(value) * len(local)
                observed += len(local)
                total_records += len(local)
            groups[label] = {
                "records": observed,
                "metrics": {key: value / observed for key, value in sorted(weighted.items())},
            }
    expected_records = len(requested) if requested is not None else len(joint_records["heldout"])
    if total_records != expected_records:
        raise UgiHeldComponentGateError("heldout loss did not cover its frozen set exactly once")
    return {
        "checkpoint": {
            "path": _display_path(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": int(checkpoint["step"]),
        },
        "seed": seed,
        "batch_size": batch_size,
        "noise_draws_per_record": 1,
        "evaluation_set": (
            "frozen_held_component_probe" if requested is not None else "full_heldout"
        ),
        "evaluated_records": total_records,
        "full_heldout_records": len(joint_records["heldout"]),
        "heldout_coverage_fraction": total_records / len(joint_records["heldout"]),
        "by_held_role_class": groups,
        "record_weighted_metrics": {
            key: value / total_records for key, value in sorted(global_weighted.items())
        },
        "seconds": time.perf_counter() - start,
    }


def audit_checkpoint_alignment(checkpoints: Sequence[Path]) -> dict[str, Any]:
    """Require checkpoint comparisons to differ only by learned state and step."""

    if torch is None:
        raise UgiHeldComponentGateError("checkpoint alignment requires torch")
    payloads = [torch.load(path, map_location="cpu", weights_only=False) for path in checkpoints]
    reference = payloads[0]
    same_inputs = all(value["inputs"] == reference["inputs"] for value in payloads[1:])
    same_model_config = all(
        value["model_config"] == reference["model_config"] for value in payloads[1:]
    )
    same_sources = all(
        set(value["source_marginals"]) == set(reference["source_marginals"])
        and all(
            np.array_equal(
                np.asarray(value["source_marginals"][key]),
                np.asarray(reference["source_marginals"][key]),
            )
            for key in reference["source_marginals"]
        )
        for value in payloads[1:]
    )
    if not (same_inputs and same_model_config and same_sources):
        raise UgiHeldComponentGateError("checkpoint comparison is not matched")
    return {
        "status": "pass",
        "same_training_inputs": same_inputs,
        "same_model_config": same_model_config,
        "same_source_marginals": same_sources,
        "checkpoints": [
            {
                "path": _display_path(path),
                "sha256": sha256_file(path),
                "step": int(payload["step"]),
            }
            for path, payload in zip(checkpoints, payloads, strict=True)
        ],
    }


def evaluate_held_component_gate(
    *,
    cache_path: Path,
    registry_path: Path,
    qualified_reactions_path: Path,
    probe_path: Path,
    checkpoints: Sequence[Path],
    sampling_results: Sequence[Path],
    output_path: Path,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    """Build one provenance-complete, nonselecting checkpoint gate artifact."""

    if len(checkpoints) != len(sampling_results) or not checkpoints:
        raise UgiHeldComponentGateError("checkpoint and sampling-result inputs must align")
    corpus, joint_records = load_ugi_training_cache(cache_path)
    checkpoint_alignment = audit_checkpoint_alignment(checkpoints)
    probe = json.loads(probe_path.read_text())
    probe_ids = tuple(str(row["product_id"]) for row in probe["samples"])
    if len(probe_ids) != len(set(probe_ids)):
        raise UgiHeldComponentGateError("held-component probe product IDs are not unique")
    closure_paths = set()
    for sampling_result in sampling_results:
        sampling_payload = json.loads(sampling_result.read_text())
        closure_path = Path(str(sampling_payload["checkpoints"]["closure"]))
        if not closure_path.is_absolute():
            closure_path = Path.cwd() / closure_path
        closure_paths.add(closure_path.resolve())
    if len(closure_paths) != 1:
        raise UgiHeldComponentGateError("sampling results do not share one closure checkpoint")
    closure_path = closure_paths.pop()
    inverse_audit = audit_reference_component_inverse(corpus)
    generated = {
        str(index): analyze_generated_components(
            sampling_result,
            registry_path,
            qualified_reactions_path,
        )
        for index, sampling_result in enumerate(sampling_results)
    }
    losses = {}
    for index, checkpoint in enumerate(checkpoints):
        losses[str(index)] = {
            "frozen_probe": heldout_loss_by_component_role(
                checkpoint,
                corpus,
                joint_records,
                seed=seed,
                batch_size=batch_size,
                product_ids=probe_ids,
            ),
            "full_heldout": heldout_loss_by_component_role(
                checkpoint,
                corpus,
                joint_records,
                seed=seed,
                batch_size=batch_size,
            ),
        }
    by_step = {}
    for index in range(len(checkpoints)):
        step = str(losses[str(index)]["full_heldout"]["checkpoint"]["step"])
        by_step[step] = {
            "heldout_loss": losses[str(index)],
            "generated_component_analysis": generated[str(index)],
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete_nonselecting_gate",
        "seed": seed,
        "scope": {
            "product_and_l1_only": True,
            "l2_routes_evaluated": False,
            "biological_guidance_evaluated": False,
            "synthesis_guidance_evaluated": False,
            "stereochemistry": "excluded_constitutional_identity",
        },
        "inputs": {
            "training_cache": {
                "path": _display_path(cache_path),
                "sha256": sha256_file(cache_path),
            },
            "component_registry": {
                "path": _display_path(registry_path),
                "sha256": sha256_file(registry_path),
            },
            "qualified_reactions": {
                "path": _display_path(qualified_reactions_path),
                "sha256": sha256_file(qualified_reactions_path),
            },
            "held_component_probe": {
                "path": _display_path(probe_path),
                "sha256": sha256_file(probe_path),
            },
            "closure_checkpoint": {
                "path": _display_path(closure_path),
                "sha256": sha256_file(closure_path),
            },
        },
        "reference_inverse_audit": inverse_audit,
        "checkpoint_alignment_audit": checkpoint_alignment,
        "checkpoints_by_step": by_step,
        "decision": {
            "production_checkpoint_frozen": False,
            "reason": (
                "This gate reports held-component likelihood and actual generated-component "
                "novelty/handle tradeoffs; it does not select a production checkpoint."
            ),
        },
    }
    _atomic_json(output_path, result)
    return result


def evaluate_unconditional_component_gate(
    *,
    registry_path: Path,
    qualified_reactions_path: Path,
    program_probe_path: Path,
    checkpoints: Sequence[Path],
    sampling_results: Sequence[Path],
    output_path: Path,
    seed: int,
) -> dict[str, Any]:
    """Audit actual components from one frozen unconditional program draw."""

    if len(checkpoints) != len(sampling_results) or not checkpoints:
        raise UgiHeldComponentGateError("checkpoint and sampling-result inputs must align")
    alignment = audit_checkpoint_alignment(checkpoints)
    closure_paths = set()
    analyses = {}
    for checkpoint, sampling_result in zip(checkpoints, sampling_results, strict=True):
        checkpoint_payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        sampling_payload = json.loads(sampling_result.read_text())
        observed_probe = Path(str(sampling_payload["matched_staged_result"]))
        if not observed_probe.is_absolute():
            observed_probe = Path.cwd() / observed_probe
        if observed_probe.resolve() != program_probe_path.resolve():
            raise UgiHeldComponentGateError("unconditional sample used a different program draw")
        closure_path = Path(str(sampling_payload["checkpoints"]["closure"]))
        if not closure_path.is_absolute():
            closure_path = Path.cwd() / closure_path
        closure_paths.add(closure_path.resolve())
        step = str(int(checkpoint_payload["step"]))
        analyses[step] = analyze_generated_components(
            sampling_result,
            registry_path,
            qualified_reactions_path,
        )
    if len(closure_paths) != 1:
        raise UgiHeldComponentGateError("unconditional samples do not share one closure model")
    closure_path = closure_paths.pop()
    result = {
        "schema_version": "phase1_ugi_unconditional_component_gate.v1",
        "status": "complete_nonselecting_gate",
        "seed": seed,
        "scope": {
            "product_and_l1_only": True,
            "program_source": "training_fold_weighted_unconditional_prior",
            "l2_routes_evaluated": False,
            "biological_guidance_evaluated": False,
            "synthesis_guidance_evaluated": False,
        },
        "inputs": {
            "program_probe": {
                "path": _display_path(program_probe_path),
                "sha256": sha256_file(program_probe_path),
            },
            "component_registry": {
                "path": _display_path(registry_path),
                "sha256": sha256_file(registry_path),
            },
            "qualified_reactions": {
                "path": _display_path(qualified_reactions_path),
                "sha256": sha256_file(qualified_reactions_path),
            },
            "closure_checkpoint": {
                "path": _display_path(closure_path),
                "sha256": sha256_file(closure_path),
            },
        },
        "checkpoint_alignment_audit": alignment,
        "checkpoints_by_step": analyses,
        "decision": {
            "production_checkpoint_frozen": False,
            "reason": (
                "This production-like unconditional draw complements the held-component gate; "
                "checkpoint selection remains contingent on the matched architecture ablation."
            ),
        },
    }
    _atomic_json(output_path, result)
    return result
