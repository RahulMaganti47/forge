"""Canonical lipid support skeletons for morphology-first graph generation.

The support skeleton is deliberately not a carbon-only scaffold and not a
fragment vocabulary.  It retains every carbon, chemically important connector,
ring atom, ionizable nitrogen/phosphorus atom, and protected reaction-core atom.
Only terminal non-carbon decorations that do not carry charge or ring topology
are removed from the morphology graph.  Those atoms and their bonds remain
explicit atom-level targets for the later chemistry-realization stage.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem

from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX
from forge.model.v5_sparse_representation import canonical_constitutional_molecule

FULL_HEAVY = "full_heavy"
CARBON_INDUCED = "carbon_induced"
FUNCTIONAL_SUPPORT = "functional_support"
SKELETON_VARIANTS = (FULL_HEAVY, CARBON_INDUCED, FUNCTIONAL_SUPPORT)

_INDEX_TO_BOND_TYPE = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


@dataclass(frozen=True)
class LipidSupportSkeleton:
    """Exact atom-level decomposition into morphology support and decorations."""

    structure_id: str
    variant: str
    canonical_smiles: str
    atom_states: tuple[AtomState, ...]
    bonds: tuple[tuple[int, int, int], ...]
    retained_atoms: tuple[int, ...]
    removed_atoms: tuple[int, ...]

    @property
    def node_count(self) -> int:
        return len(self.atom_states)

    @property
    def retained_count(self) -> int:
        return len(self.retained_atoms)

    @property
    def removed_count(self) -> int:
        return len(self.removed_atoms)

    @property
    def skeleton_bonds(self) -> tuple[tuple[int, int, int], ...]:
        retained = set(self.retained_atoms)
        return tuple(bond for bond in self.bonds if bond[0] in retained and bond[1] in retained)

    @property
    def expansion_bonds(self) -> tuple[tuple[int, int, int], ...]:
        retained = set(self.retained_atoms)
        return tuple(
            bond for bond in self.bonds if bond[0] not in retained or bond[1] not in retained
        )


def _atom_state(atom: Chem.Atom) -> AtomState:
    return AtomState(
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.GetNumExplicitHs(),
    )


def _functional_support_atom(atom: Chem.Atom, protected_atoms: set[int]) -> bool:
    """Keep morphology-bearing atoms while pruning only terminal decorations.

    Connector heteroatoms are retained.  This is important for esters, ethers,
    amides, carbonates, heterocycles, and ionizable heads.  Nitrogen and
    phosphorus are retained even when terminal because a terminal amine can be
    the chemically defining polar anchor.  Formal charge and reaction-core
    protection always override pruning.
    """

    index = atom.GetIdx()
    symbol = atom.GetSymbol()
    return bool(
        index in protected_atoms
        or symbol == "C"
        or symbol in {"N", "P"}
        or atom.GetFormalCharge() != 0
        or atom.IsInRing()
        or atom.GetDegree() >= 2
    )


def encode_lipid_support_skeleton(
    molecule: Chem.Mol,
    *,
    structure_id: str,
    variant: str,
    protected_atoms: Iterable[int] = (),
) -> LipidSupportSkeleton:
    """Encode one connected constitutional graph into a canonical skeleton."""

    if variant not in SKELETON_VARIANTS:
        raise FeasibilityError(f"unsupported lipid skeleton variant: {variant}")
    if molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        raise FeasibilityError("lipid support skeleton requires one nonempty fragment")
    normalized = canonical_constitutional_molecule(molecule, structure_id)
    canonical_smiles = Chem.MolToSmiles(
        normalized,
        canonical=True,
        isomericSmiles=False,
    )
    protected = {int(index) for index in protected_atoms}
    if any(index < 0 or index >= normalized.GetNumAtoms() for index in protected):
        raise FeasibilityError(f"protected atom lies outside {structure_id}")

    if variant == FULL_HEAVY:
        retained = set(range(normalized.GetNumAtoms()))
    elif variant == CARBON_INDUCED:
        retained = {atom.GetIdx() for atom in normalized.GetAtoms() if atom.GetSymbol() == "C"}
        retained.update(protected)
    else:
        retained = {
            atom.GetIdx()
            for atom in normalized.GetAtoms()
            if _functional_support_atom(atom, protected)
        }
    if not retained:
        raise FeasibilityError(f"skeleton retains no atoms: {structure_id}: {variant}")

    atom_states = tuple(_atom_state(atom) for atom in normalized.GetAtoms())
    bonds = []
    for bond in normalized.GetBonds():
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(
                f"unsupported skeleton bond type: {structure_id}: {bond.GetBondType()}"
            )
        left, right = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        bonds.append((left, right, SPARSE_BOND_TO_INDEX[bond.GetBondType()]))
    bonds.sort()
    retained_atoms = tuple(sorted(retained))
    removed_atoms = tuple(
        index for index in range(normalized.GetNumAtoms()) if index not in retained
    )
    return LipidSupportSkeleton(
        structure_id=structure_id,
        variant=variant,
        canonical_smiles=canonical_smiles,
        atom_states=atom_states,
        bonds=tuple(bonds),
        retained_atoms=retained_atoms,
        removed_atoms=removed_atoms,
    )


def reconstruct_lipid_support_skeleton(encoding: LipidSupportSkeleton) -> Chem.Mol:
    """Reconstruct the exact constitutional graph from support plus expansions."""

    editable = Chem.RWMol()
    for state in encoding.atom_states:
        atom = Chem.Atom(state.symbol)
        atom.SetFormalCharge(state.formal_charge)
        atom.SetIsAromatic(state.aromatic)
        if state.explicit_hydrogens:
            atom.SetNumExplicitHs(state.explicit_hydrogens)
            atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for left, right, bond in encoding.bonds:
        editable.AddBond(left, right, _INDEX_TO_BOND_TYPE[bond])
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def skeleton_roundtrip_exact(encoding: LipidSupportSkeleton) -> bool:
    """Return whether support plus atom-level expansions restores the molecule."""

    reconstructed = reconstruct_lipid_support_skeleton(encoding)
    observed = Chem.MolToSmiles(reconstructed, canonical=True, isomericSmiles=False)
    return observed == encoding.canonical_smiles


def skeleton_signature(encoding: LipidSupportSkeleton) -> str:
    """Hash the canonical representation for atom-permutation stability tests."""

    payload = {
        "variant": encoding.variant,
        "canonical_smiles": encoding.canonical_smiles,
        "atom_states": [state.key() for state in encoding.atom_states],
        "bonds": encoding.bonds,
        "retained_atoms": encoding.retained_atoms,
        "removed_atoms": encoding.removed_atoms,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def _selected_adjacency(
    encoding: LipidSupportSkeleton,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    full = {index: set() for index in range(encoding.node_count)}
    skeleton = {index: set() for index in encoding.retained_atoms}
    for left, right, _ in encoding.bonds:
        full[left].add(right)
        full[right].add(left)
        if left in skeleton and right in skeleton:
            skeleton[left].add(right)
            skeleton[right].add(left)
    return full, skeleton


def _component_count(adjacency: dict[int, set[int]]) -> int:
    remaining = set(adjacency)
    components = 0
    while remaining:
        components += 1
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
    return components


def selected_component_count(
    encoding: LipidSupportSkeleton,
    selected_atoms: Iterable[int] | None = None,
) -> int:
    """Count induced components among retained atoms or a retained subset."""

    _, skeleton = _selected_adjacency(encoding)
    if selected_atoms is None:
        return _component_count(skeleton)
    selected = set(int(index) for index in selected_atoms) & set(skeleton)
    if not selected:
        return 0
    adjacency = {
        index: {neighbor for neighbor in skeleton[index] if neighbor in selected}
        for index in selected
    }
    return _component_count(adjacency)


def skeleton_statistics(encoding: LipidSupportSkeleton) -> dict[str, Any]:
    """Measure how a skeleton changes apparent morphology without losing chemistry."""

    full, skeleton = _selected_adjacency(encoding)
    components = _component_count(skeleton)
    skeleton_edges = sum(len(neighbors) for neighbors in skeleton.values()) // 2
    full_leaves = sum(len(neighbors) == 1 for neighbors in full.values())
    skeleton_leaves = sum(len(neighbors) == 1 for neighbors in skeleton.values())
    full_junctions = sum(len(neighbors) >= 3 for neighbors in full.values())
    skeleton_junctions = sum(len(neighbors) >= 3 for neighbors in skeleton.values())
    resolved_false_junctions = sum(
        len(full[index]) >= 3 and len(skeleton.get(index, ())) <= 2
        for index in encoding.retained_atoms
    )
    removed_symbols: dict[str, int] = {}
    removed_bond_types: dict[str, int] = {}
    attachment_counts: dict[int, int] = {}
    removed = set(encoding.removed_atoms)
    for index in encoding.removed_atoms:
        symbol = encoding.atom_states[index].symbol
        removed_symbols[symbol] = removed_symbols.get(symbol, 0) + 1
    for left, right, bond in encoding.expansion_bonds:
        label = str(bond)
        removed_bond_types[label] = removed_bond_types.get(label, 0) + 1
        if left in removed and right not in removed:
            attachment_counts[right] = attachment_counts.get(right, 0) + 1
        elif right in removed and left not in removed:
            attachment_counts[left] = attachment_counts.get(left, 0) + 1
    return {
        "full_atoms": encoding.node_count,
        "retained_atoms": encoding.retained_count,
        "removed_atoms": encoding.removed_count,
        "retained_fraction": encoding.retained_count / encoding.node_count,
        "components": components,
        "connected": components == 1,
        "skeleton_edges": skeleton_edges,
        "cycle_rank": skeleton_edges - encoding.retained_count + components,
        "full_leaves": full_leaves,
        "skeleton_leaves": skeleton_leaves,
        "removed_apparent_leaves": full_leaves - skeleton_leaves,
        "full_junctions": full_junctions,
        "skeleton_junctions": skeleton_junctions,
        "resolved_false_junctions": resolved_false_junctions,
        "maximum_decorations_per_support_atom": max(attachment_counts.values(), default=0),
        "removed_symbols": dict(sorted(removed_symbols.items())),
        "removed_bond_types": dict(sorted(removed_bond_types.items())),
        "roundtrip_exact": skeleton_roundtrip_exact(encoding),
    }


def protected_core_indices(atom_rows: Sequence[dict[str, str]]) -> set[int]:
    """Extract protected Ugi-core indices from one product's semantic atom rows."""

    return {
        int(row["product_atom_index"])
        for row in atom_rows
        if str(row["is_ugi_core"]).lower() == "true"
    }
