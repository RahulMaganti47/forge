"""Deterministic component-family clustering and leakage-safe fold assignment.

Reaction-enumerated corpora must split components before products are assembled.  This module is
shared by Ugi and repeated-reaction expansions so the definition of a structural family and the
heldout-precedence rule cannot drift between chemistries.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

FOLDS = ("train", "calibration", "heldout")


def _molecule(smiles: str, *, error: type[ValueError]) -> Chem.Mol:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise error(f"family component is not one valid connected molecule: {smiles!r}")
    return molecule


def similarity_families(
    smiles: Sequence[str],
    fingerprint: Mapping[str, Any],
    role: str,
    *,
    error: type[ValueError] = ValueError,
) -> dict[str, str]:
    """Group every threshold-connected ECFP component into one structural family."""

    ordered = sorted(smiles)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(fingerprint["radius"]),
        fpSize=int(fingerprint["bits"]),
        includeChirality=bool(fingerprint["include_chirality"]),
    )
    fps = [generator.GetFingerprint(_molecule(value, error=error)) for value in ordered]
    adjacency: list[list[int]] = [[] for _ in ordered]
    threshold = float(fingerprint["similarity_threshold"])
    for index, current in enumerate(fps):
        similarities = DataStructs.BulkTanimotoSimilarity(current, fps[:index])
        for other, similarity in enumerate(similarities):
            if similarity >= threshold:
                adjacency[index].append(other)
                adjacency[other].append(index)

    components: list[list[int]] = []
    unseen = set(range(len(ordered)))
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))

    assignments: dict[str, str] = {}
    for component in components:
        members = [ordered[index] for index in component]
        digest = hashlib.sha256((role + "\0" + "\0".join(members)).encode()).hexdigest()[:16]
        family_id = f"{role}-{digest}"
        assignments.update({member: family_id for member in members})
    return assignments


def family_fold_map(
    family_members: Mapping[str, Sequence[str]],
    fractions: Mapping[str, Any],
    seed: int,
    role: str,
    *,
    error: type[ValueError] = ValueError,
) -> dict[str, str]:
    """Greedily balance complete structural families without splitting a family."""

    total = sum(len(values) for values in family_members.values())
    targets = {fold: total * float(fractions[fold]) for fold in FOLDS}
    counts = {fold: 0 for fold in FOLDS}
    ordered = sorted(
        family_members,
        key=lambda family: (
            -len(family_members[family]),
            hashlib.sha256(f"{seed}\0{role}\0{family}".encode()).hexdigest(),
        ),
    )
    assignments: dict[str, str] = {}
    for family in ordered:
        remaining = {fold: targets[fold] - counts[fold] for fold in FOLDS}
        fold = max(
            FOLDS,
            key=lambda value: (
                remaining[value],
                hashlib.sha256(f"{seed}\0{role}\0{family}\0{value}".encode()).hexdigest(),
            ),
        )
        assignments[family] = fold
        counts[fold] += len(family_members[family])
    if len(ordered) >= len(FOLDS) and set(assignments.values()) != set(FOLDS):
        raise error(f"family assignment left an empty {role} fold")
    return assignments


def product_fold(component_folds: Sequence[str], *, error: type[ValueError] = ValueError) -> str:
    """Apply heldout, then calibration, then training precedence to component folds."""

    if any(fold not in FOLDS for fold in component_folds):
        raise error(f"invalid component fold tuple: {tuple(component_folds)}")
    if "heldout" in component_folds:
        return "heldout"
    if "calibration" in component_folds:
        return "calibration"
    return "train"


__all__ = ["FOLDS", "family_fold_map", "product_fold", "similarity_families"]
