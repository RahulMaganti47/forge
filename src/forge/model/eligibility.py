"""Nonselecting chemistry-motif and declared-support diagnostics for Ugi products."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from rdkit import Chem

from forge.potency.annotations import ROLE_NAMES


class UgiCandidateEligibilityError(RuntimeError):
    """Raised when the candidate-eligibility audit contract is malformed."""


def compile_smarts(smarts_by_name: Mapping[str, str]) -> dict[str, Chem.Mol]:
    """Compile an explicitly named SMARTS registry, failing on any invalid pattern."""

    compiled = {}
    for name, smarts in smarts_by_name.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise UgiCandidateEligibilityError(f"invalid SMARTS for {name}: {smarts}")
        compiled[str(name)] = query
    return compiled


def motif_hits(molecule: Chem.Mol, queries: Mapping[str, Chem.Mol]) -> dict[str, bool]:
    """Return presence/absence for every frozen motif query."""

    return {name: molecule.HasSubstructMatch(query) for name, query in queries.items()}


def heteroatom_profile(molecule: Chem.Mol) -> tuple[int, float]:
    """Return non-carbon heavy-atom count and its fraction of all heavy atoms."""

    heavy_atoms = molecule.GetNumHeavyAtoms()
    if heavy_atoms <= 0:
        raise UgiCandidateEligibilityError("heteroatom profile requires a heavy atom")
    heteroatoms = sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms())
    return heteroatoms, heteroatoms / heavy_atoms


def heteroatom_thresholds(
    counts: np.ndarray,
    fractions: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, float]:
    """Freeze selection-reference quantile thresholds for extreme heteroatom flags."""

    if not len(counts) or len(counts) != len(fractions):
        raise UgiCandidateEligibilityError("heteroatom thresholds require aligned values")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise UgiCandidateEligibilityError("invalid heteroatom quantile interval")
    return {
        "count_q01": float(np.quantile(counts, lower_quantile, method="linear")),
        "count_q99": float(np.quantile(counts, upper_quantile, method="linear")),
        "fraction_q01": float(np.quantile(fractions, lower_quantile, method="linear")),
        "fraction_q99": float(np.quantile(fractions, upper_quantile, method="linear")),
    }


def heteroatom_extreme_flags(
    count: int,
    fraction: float,
    thresholds: Mapping[str, float],
) -> dict[str, bool]:
    """Flag strict excursions beyond the frozen q01-q99 reference interval."""

    flags = {
        "low_heteroatom_count": count < thresholds["count_q01"],
        "high_heteroatom_count": count > thresholds["count_q99"],
        "low_heteroatom_fraction": fraction < thresholds["fraction_q01"],
        "high_heteroatom_fraction": fraction > thresholds["fraction_q99"],
    }
    flags["any_extreme_heteroatom_pattern"] = any(flags.values())
    return flags


def declared_support_violations(
    sample: Mapping[str, Any],
    molecule: Chem.Mol,
    model_config: Mapping[str, Any],
    atom_vocabulary: set[tuple[str, int, bool, int]],
) -> list[str]:
    """Reapply the selected generator's declared graph-support contract."""

    violations = []
    if len(Chem.GetMolFrags(molecule)) != 1:
        violations.append("disconnected_product")
    if molecule.GetNumHeavyAtoms() > int(model_config["maximum_total_atoms"]):
        violations.append("maximum_total_atoms")
    for atom in molecule.GetAtoms():
        state = (
            atom.GetSymbol(),
            atom.GetFormalCharge(),
            atom.GetIsAromatic(),
            atom.GetNumExplicitHs(),
        )
        if state not in atom_vocabulary:
            violations.append("atom_state_vocabulary")
            break
    allowed_bonds = {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.TRIPLE,
        Chem.BondType.AROMATIC,
    }
    if any(bond.GetBondType() not in allowed_bonds for bond in molecule.GetBonds()):
        violations.append("bond_state_vocabulary")

    program = sample["program"]
    node_counts = tuple(int(value) for value in program["node_counts"])
    junctions = tuple(int(value) for value in program["junction_budgets"])
    cycles = tuple(int(value) for value in program["cycle_ranks"])
    attachments = tuple(int(value) for value in program["attachment_counts"])
    if len(node_counts) != 3 or any(
        value < 1 or value > int(model_config["maximum_component_atoms"]) for value in node_counts
    ):
        violations.append("component_atom_bound")
    if len(junctions) != 3 or any(
        value < 0 or value > int(model_config["maximum_junction_budget"]) for value in junctions
    ):
        violations.append("junction_budget_bound")
    if len(cycles) != 3 or any(
        value < 0 or value > int(model_config["maximum_cycle_rank"]) for value in cycles
    ):
        violations.append("cycle_rank_bound")
    if len(attachments) != 3 or any(
        value < 1 or value > int(model_config["maximum_attachment_count"]) for value in attachments
    ):
        violations.append("attachment_count_bound")
    for role, node_count, attachment_count in zip(
        ROLE_NAMES, node_counts, attachments, strict=True
    ):
        offspring = tuple(int(value) for value in sample["offspring_by_role"][role])
        if len(offspring) != node_count:
            violations.append(f"{role}:offspring_length")
        if any(value < 0 or value > int(model_config["maximum_children"]) for value in offspring):
            violations.append(f"{role}:offspring_bound")
        if sum(offspring) != node_count - attachment_count:
            violations.append(f"{role}:forest_edge_identity")
    return sorted(set(violations))
