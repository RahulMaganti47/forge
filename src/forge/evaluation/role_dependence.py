"""Frozen descriptor diagnostic for dependence between Ugi precursor roles."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from forge.core.io import iter_csv


class ConditionalRoleDependenceError(ValueError):
    """The dependence diagnostic cannot satisfy its frozen grouping contract."""


DESCRIPTOR_NAMES = (
    "amine_nitrogen_x_aldehyde_carbon",
    "amine_nitrogen_x_isocyanide_carbon",
    "amine_heteroatom_x_total_tail_carbon",
    "aldehyde_branch_x_isocyanide_carbon",
    "aldehyde_carbon_x_isocyanide_carbon",
    "aldehyde_unsaturation_x_isocyanide_unsaturation",
    "amine_logp_x_total_tail_carbon",
    "aldehyde_logp_minus_isocyanide_logp",
)


def _component_features(smiles: str) -> np.ndarray:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ConditionalRoleDependenceError("dependence input contains an invalid component")
    carbon = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    nitrogen = sum(atom.GetAtomicNum() == 7 for atom in molecule.GetAtoms())
    hetero = sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms())
    branch = sum(atom.GetDegree() >= 3 for atom in molecule.GetAtoms())
    unsaturation = sum(
        bond.GetBondTypeAsDouble() > 1.0 and not bond.GetIsAromatic()
        for bond in molecule.GetBonds()
    )
    return np.asarray(
        [
            molecule.GetNumHeavyAtoms(),
            carbon,
            nitrogen,
            hetero,
            molecule.GetRingInfo().NumRings(),
            branch,
            unsaturation,
            Crippen.MolLogP(molecule),  # type: ignore[attr-defined]
        ],
        dtype=np.float64,
    )


def _context(parts: Sequence[np.ndarray]) -> tuple[int, ...]:
    """Count-only context supplied to both intact and shuffled examples."""

    return tuple(
        int(value) for features in parts for value in (features[0], features[4], features[5])
    )


def _cross_role_descriptors(parts: Sequence[np.ndarray]) -> np.ndarray:
    amine, aldehyde, isocyanide = parts
    total_tail_carbon = aldehyde[1] + isocyanide[1]
    return np.asarray(
        [
            amine[2] * aldehyde[1],
            amine[2] * isocyanide[1],
            amine[3] * total_tail_carbon,
            aldehyde[5] * isocyanide[1],
            aldehyde[1] * isocyanide[1],
            aldehyde[6] * isocyanide[6],
            amine[7] * total_tail_carbon,
            aldehyde[7] - isocyanide[7],
        ],
        dtype=np.float64,
    )


def _residual_correlation(
    reference_contexts: np.ndarray,
    reference_descriptors: np.ndarray,
    target_contexts: np.ndarray,
    target_descriptors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((np.ones(len(reference_contexts)), reference_contexts))
    coefficients = np.linalg.lstsq(design, reference_descriptors, rcond=None)[0]
    reference_residuals = reference_descriptors - design @ coefficients
    target_design = np.column_stack((np.ones(len(target_contexts)), target_contexts))
    target_residuals = target_descriptors - target_design @ coefficients
    return (
        np.nan_to_num(np.corrcoef(reference_residuals, rowvar=False)),
        np.nan_to_num(np.corrcoef(target_residuals, rowvar=False)),
    )


def cross_role_fidelity_to_heldout(
    rows: Sequence[dict[str, Any]],
    assignments_path: Path,
    *,
    roles: Sequence[str],
) -> dict[str, Any]:
    """Compare generated unique exact-L1 role dependence to heldout Ugi tuples."""

    if len(roles) != 3:
        raise ConditionalRoleDependenceError("cross-role fidelity requires exactly three roles")
    feature_cache: dict[str, np.ndarray] = {}

    def features(smiles: str) -> np.ndarray:
        if smiles not in feature_cache:
            feature_cache[smiles] = _component_features(smiles)
        return feature_cache[smiles]

    reference_parts = [
        tuple(features(row[f"{role}_smiles"]) for role in roles)
        for row in iter_csv(assignments_path)
        if row.get("primary_product_fold") == "heldout"
    ]
    generated_parts = []
    for row in rows:
        if row.get("program_id") != "ugi_3cr_agile":
            continue
        if row.get("exact_l1_program") is not True or int(row["exact_l1_trace_count"]) != 1:
            continue
        components = row["exact_l1_traces"][0]["components_by_role"]
        generated_parts.append(tuple(features(str(components[role])) for role in roles))
    if len(reference_parts) < len(DESCRIPTOR_NAMES) + 2:
        raise ConditionalRoleDependenceError("heldout cross-role reference is too small")
    if len(generated_parts) < len(DESCRIPTOR_NAMES) + 2:
        return {
            "eligible_generated_products": len(generated_parts),
            "reference_products": len(reference_parts),
            "residual_correlation_frobenius_distance": None,
            "status": "insufficient_generated_exact_l1_support",
            "lower_is_better": True,
        }
    reference_contexts = np.asarray(
        [_context(parts) for parts in reference_parts], dtype=np.float64
    )
    reference_descriptors = np.asarray(
        [_cross_role_descriptors(parts) for parts in reference_parts], dtype=np.float64
    )
    generated_contexts = np.asarray(
        [_context(parts) for parts in generated_parts], dtype=np.float64
    )
    generated_descriptors = np.asarray(
        [_cross_role_descriptors(parts) for parts in generated_parts], dtype=np.float64
    )
    reference_matrix, generated_matrix = _residual_correlation(
        reference_contexts,
        reference_descriptors,
        generated_contexts,
        generated_descriptors,
    )
    return {
        "eligible_generated_products": len(generated_parts),
        "reference_products": len(reference_parts),
        "residual_correlation_frobenius_distance": float(
            np.linalg.norm(reference_matrix - generated_matrix, ord="fro")
        ),
        "status": "measured",
        "lower_is_better": True,
    }


def _fold_for_families(families: Sequence[str], *, seed: int, folds: int) -> int:
    digest = hashlib.sha256(f"{seed}|{'|'.join(sorted(families))}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def run_conditional_role_dependence(
    assignments_path: Path,
    *,
    roles: Sequence[str],
    seed: int,
    folds: int = 5,
) -> dict[str, Any]:
    """Distinguish intact from within-context shuffled role tuples.

    The classifier sees only cross-role descriptor interactions.  Every intact/shuffled pair stays
    in the same outer fold, which is assigned by the complete component-family tuple.  The result
    reports family overlap explicitly; a nonzero overlap fails the diagnostic instead of being
    silently accepted.
    """

    if len(roles) != 3 or folds < 2:
        raise ConditionalRoleDependenceError(
            "diagnostic requires exactly three roles and >=2 folds"
        )
    feature_cache: dict[str, np.ndarray] = {}
    records: list[dict[str, Any]] = []
    for row in iter_csv(assignments_path):
        if row.get("primary_product_fold") != "heldout":
            continue
        smiles = tuple(row[f"{role}_smiles"] for role in roles)
        for value in smiles:
            if value not in feature_cache:
                feature_cache[value] = _component_features(value)
        parts = tuple(feature_cache[value] for value in smiles)
        records.append(
            {
                "parts": parts,
                "context": _context(parts),
                "families": tuple(row[f"{role}_family_id"] for role in roles),
            }
        )
    if not records:
        raise ConditionalRoleDependenceError("heldout dependence population is empty")
    by_context: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        by_context[cast(tuple[int, ...], row["context"])].append(index)
    eligible = {key: values for key, values in by_context.items() if len(values) >= 2}
    if not eligible:
        raise ConditionalRoleDependenceError("no architectural context supports a role permutation")
    rng = np.random.default_rng(seed)
    intact: list[np.ndarray] = []
    shuffled: list[np.ndarray] = []
    groups: list[int] = []
    families_by_fold: dict[int, set[str]] = defaultdict(set)
    for context in sorted(eligible):
        indices = eligible[context]
        role_permutations = [rng.permutation(indices) for _ in roles]
        for local, source_index in enumerate(indices):
            source = records[source_index]
            shuffled_parts = tuple(
                records[int(role_permutations[role_index][local])]["parts"][role_index]
                for role_index in range(3)
            )
            # Count context must be exactly preserved. Permuting within the context enforces this.
            if _context(shuffled_parts) != context:
                raise ConditionalRoleDependenceError("within-context permutation changed context")
            fold = _fold_for_families(source["families"], seed=seed, folds=folds)
            families_by_fold[fold].update(source["families"])
            intact.append(_cross_role_descriptors(source["parts"]))
            shuffled.append(_cross_role_descriptors(shuffled_parts))
            groups.append(fold)
    overlap = {
        f"{left}-{right}": len(families_by_fold[left] & families_by_fold[right])
        for left in range(folds)
        for right in range(left + 1, folds)
    }
    # The tuple-group assignment is useful for uncertainty but cannot generally make a dense
    # combinatorial library component-disjoint. Report that limitation instead of claiming it away.
    x = np.vstack((intact, shuffled))
    y = np.concatenate((np.ones(len(intact)), np.zeros(len(shuffled))))
    group_array = np.asarray(groups + groups, dtype=np.int64)
    aucs: list[float] = []
    fold_rows = []
    for fold in range(folds):
        test = group_array == fold
        train = ~test
        if not test.any() or len(set(y[train])) != 2 or len(set(y[test])) != 2:
            raise ConditionalRoleDependenceError(f"diagnostic fold {fold} is empty or one-class")
        scaler = StandardScaler().fit(x[train])
        model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed + fold,
            solver="lbfgs",
        ).fit(scaler.transform(x[train]), y[train])
        probabilities = model.predict_proba(scaler.transform(x[test]))[:, 1]
        auc = float(roc_auc_score(y[test], probabilities))
        aucs.append(auc)
        fold_rows.append({"fold": fold, "rows": int(test.sum()), "auc": auc})
    context_matrix = np.asarray(
        [records[index]["context"] for context in sorted(eligible) for index in eligible[context]],
        dtype=np.float64,
    )
    intact_matrix, shuffled_matrix = _residual_correlation(
        context_matrix,
        np.asarray(intact),
        context_matrix,
        np.asarray(shuffled),
    )
    distance = float(np.linalg.norm(np.nan_to_num(intact_matrix - shuffled_matrix), ord="fro"))
    return {
        "schema_version": "forge.conditional_role_dependence.v1",
        "seed": seed,
        "roles": list(roles),
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "heldout_rows": len(records),
        "eligible_rows": len(intact),
        "eligible_contexts": len(eligible),
        "classifier_auc_mean": float(np.mean(aucs)),
        "classifier_auc_by_fold": fold_rows,
        "correlation_frobenius_intact_vs_shuffled": distance,
        "component_family_overlap_by_tuple_group_fold": overlap,
        "strict_component_disjoint_classifier": all(value == 0 for value in overlap.values()),
        "interpretation_gate": {
            "detectable_auc_threshold": 0.6,
            "dependence_detected": float(np.mean(aucs)) >= 0.6,
            "family_grouping_clean": all(value == 0 for value in overlap.values()),
            "claim_authorized": float(np.mean(aucs)) >= 0.6
            and all(value == 0 for value in overlap.values()),
        },
        "nonclaims": [
            "This diagnostic tests whether dependence exists; it does not show that FORGE learns it.",
            "A tuple-group fold is not called component-disjoint when component families overlap.",
        ],
    }


__all__ = [
    "ConditionalRoleDependenceError",
    "DESCRIPTOR_NAMES",
    "cross_role_fidelity_to_heldout",
    "run_conditional_role_dependence",
]
