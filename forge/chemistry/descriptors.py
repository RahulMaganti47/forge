"""Vocabulary-free graph descriptors shared by audits and applicability policies."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors


class DescriptorError(ValueError):
    """Raised when a descriptor cannot be computed for one connected molecule."""


CHEMOTYPE_SIGNATURE_FIELDS = (
    "carbon_atoms",
    "nitrogen_atoms",
    "oxygen_atoms",
    "sulfur_atoms",
    "phosphorus_atoms",
    "halogen_atoms",
    "carbon_carbon_double_bonds",
    "carbon_carbon_triple_bonds",
    "carbon_branch_atoms",
    "adjacent_carbon_branch_edges",
    "ring_count",
    "carbonyl_count",
    "ester_like_carbonyl_count",
    "amide_like_carbonyl_count",
    "ether_oxygen_count",
    "carbon_subgraph_diameter",
)
ARCHITECTURE_FIELDS = tuple(
    field
    for field in CHEMOTYPE_SIGNATURE_FIELDS
    if field not in {"carbon_atoms", "carbon_subgraph_diameter"}
)


def chemotype_signature(metrics: Mapping[str, int]) -> str:
    """Serialize the declared coarse chemotype signature deterministically."""

    return "|".join(f"{field}={int(metrics[field])}" for field in CHEMOTYPE_SIGNATURE_FIELDS)


def architecture_signature(metrics: Mapping[str, int]) -> str:
    """Serialize the length-independent functional architecture signature."""

    return "|".join(f"{field}={int(metrics[field])}" for field in ARCHITECTURE_FIELDS)


def connected_molecule(smiles: str, *, label: str = "molecule") -> Chem.Mol:
    """Parse one connected molecular graph while suppressing RDKit parser logging."""

    if not isinstance(smiles, str) or not smiles:
        raise DescriptorError(f"{label} must be non-empty SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise DescriptorError(f"{label} must be one valid connected graph")
    return molecule


def carbon_branch_metrics(smiles: str) -> dict[str, int | None]:
    """Return carbon-skeleton branch counts without fragment assumptions."""

    molecule = connected_molecule(smiles, label="component")
    branch_atoms = {
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.GetAtomicNum() == 6
        and sum(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()) >= 3
    }
    adjacency: dict[int, set[int]] = {index: set() for index in branch_atoms}
    adjacent_edges = 0
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in branch_atoms and right in branch_atoms:
            adjacency[left].add(right)
            adjacency[right].add(left)
            adjacent_edges += 1
    maximum_run = 0
    unseen = set(branch_atoms)
    while unseen:
        seed = unseen.pop()
        stack = [seed]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        maximum_run = max(maximum_run, size)
    minimum_distance: int | None = None
    distances = Chem.GetDistanceMatrix(molecule)
    branch_list = sorted(branch_atoms)
    for offset, left in enumerate(branch_list):
        for right in branch_list[offset + 1 :]:
            distance = int(distances[left][right])
            minimum_distance = (
                distance if minimum_distance is None else min(minimum_distance, distance)
            )
    return {
        "carbon_branch_atoms": len(branch_atoms),
        "adjacent_carbon_branch_edges": adjacent_edges,
        "maximum_adjacent_branch_run": maximum_run,
        "minimum_branch_graph_distance": minimum_distance,
    }


def _carbon_subgraph_diameter(molecule: Chem.Mol) -> int:
    carbon_indices = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6}
    if not carbon_indices:
        return 0
    adjacency = {
        index: {
            neighbor.GetIdx()
            for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors()
            if neighbor.GetIdx() in carbon_indices
        }
        for index in carbon_indices
    }
    diameter = 0
    for start in carbon_indices:
        distances = {start: 0}
        queue = [start]
        for current in queue:
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        diameter = max(diameter, max(distances.values()))
    return diameter


def component_chemotype_metrics(smiles: str) -> dict[str, int]:
    """Return coarse graph-derived component descriptors without a fragment vocabulary."""

    molecule = connected_molecule(smiles, label="component")
    element_counts = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    carbon_double_bonds = 0
    carbon_triple_bonds = 0
    for bond in molecule.GetBonds():
        left = bond.GetBeginAtom()
        right = bond.GetEndAtom()
        if left.GetAtomicNum() != 6 or right.GetAtomicNum() != 6 or bond.GetIsAromatic():
            continue
        if bond.GetBondType() == Chem.BondType.DOUBLE:
            carbon_double_bonds += 1
        elif bond.GetBondType() == Chem.BondType.TRIPLE:
            carbon_triple_bonds += 1

    carbonyl_count = 0
    ester_like_count = 0
    amide_like_count = 0
    carbonyl_adjacent_oxygens: set[int] = set()
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        neighbors = list(atom.GetNeighbors())
        bonds = [molecule.GetBondBetweenAtoms(atom.GetIdx(), item.GetIdx()) for item in neighbors]
        if not any(
            neighbor.GetAtomicNum() == 8 and bond.GetBondType() == Chem.BondType.DOUBLE
            for neighbor, bond in zip(neighbors, bonds, strict=True)
        ):
            continue
        carbonyl_count += 1
        for neighbor, bond in zip(neighbors, bonds, strict=True):
            if bond.GetBondType() != Chem.BondType.SINGLE:
                continue
            if neighbor.GetAtomicNum() == 8:
                ester_like_count += 1
                carbonyl_adjacent_oxygens.add(neighbor.GetIdx())
            elif neighbor.GetAtomicNum() == 7:
                amide_like_count += 1

    ether_oxygen_count = 0
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 8 or atom.GetIdx() in carbonyl_adjacent_oxygens:
            continue
        if (
            atom.GetDegree() == 2
            and all(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors())
            and all(
                molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx()).GetBondType()
                == Chem.BondType.SINGLE
                for neighbor in atom.GetNeighbors()
            )
        ):
            ether_oxygen_count += 1

    branch = carbon_branch_metrics(smiles)
    return {
        "heavy_atoms": molecule.GetNumHeavyAtoms(),
        "carbon_atoms": element_counts["C"],
        "nitrogen_atoms": element_counts["N"],
        "oxygen_atoms": element_counts["O"],
        "sulfur_atoms": element_counts["S"],
        "phosphorus_atoms": element_counts["P"],
        "halogen_atoms": sum(element_counts[symbol] for symbol in ("F", "Cl", "Br", "I")),
        "carbon_carbon_double_bonds": carbon_double_bonds,
        "carbon_carbon_triple_bonds": carbon_triple_bonds,
        "carbon_branch_atoms": int(branch["carbon_branch_atoms"] or 0),
        "adjacent_carbon_branch_edges": int(branch["adjacent_carbon_branch_edges"] or 0),
        "ring_count": molecule.GetRingInfo().NumRings(),
        "carbonyl_count": carbonyl_count,
        "ester_like_carbonyl_count": ester_like_count,
        "amide_like_carbonyl_count": amide_like_count,
        "ether_oxygen_count": ether_oxygen_count,
        "carbon_subgraph_diameter": _carbon_subgraph_diameter(molecule),
    }


def ring_signature(molecule: Chem.Mol, macrocycle_minimum: int) -> dict[str, Any]:
    """Describe ring topology without assigning synthesis or biological meaning."""

    rings = [tuple(int(atom) for atom in ring) for ring in Chem.GetSymmSSSR(molecule)]
    sizes = sorted(len(ring) for ring in rings)
    fused = any(
        len(set(left) & set(right)) >= 2
        for index, left in enumerate(rings)
        for right in rings[index + 1 :]
    )
    aromatic_rings = 0
    heterocyclic_rings = 0
    aromatic_heterocycles = 0
    for ring in rings:
        atoms = [molecule.GetAtomWithIdx(index) for index in ring]
        aromatic = all(atom.GetIsAromatic() for atom in atoms)
        heterocyclic = any(atom.GetAtomicNum() != 6 for atom in atoms)
        aromatic_rings += int(aromatic)
        heterocyclic_rings += int(heterocyclic)
        aromatic_heterocycles += int(aromatic and heterocyclic)
    return {
        "ring_count": len(rings),
        "ring_sizes": sizes,
        "cycle_rank": molecule.GetNumBonds() - molecule.GetNumAtoms() + 1,
        "maximum_ring_size": max(sizes, default=0),
        "macrocycle": any(size >= macrocycle_minimum for size in sizes),
        "fused": fused,
        "spiro": rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0,
        "bridgehead": rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 0,
        "aromatic_ring_count": aromatic_rings,
        "heterocyclic_ring_count": heterocyclic_rings,
        "aromatic_heterocycle_count": aromatic_heterocycles,
    }


__all__ = [
    "ARCHITECTURE_FIELDS",
    "CHEMOTYPE_SIGNATURE_FIELDS",
    "DescriptorError",
    "architecture_signature",
    "carbon_branch_metrics",
    "chemotype_signature",
    "component_chemotype_metrics",
    "connected_molecule",
    "ring_signature",
]
