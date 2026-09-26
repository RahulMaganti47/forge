"""Canonical sparse representation selected for the Phase 1 V5 morphology flow.

This module is intentionally separate from the frozen V3 feasibility encoder.
Historical CUDA artifacts hash that source file and must remain reproducible.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.lipid_context import assign_lipid_regions, select_lipid_polar_root
from forge.model.phase1_tree_topology_flow import (
    TreeTopologyFlowError,
    offspring_to_parents,
    preorder_offspring_to_parents,
)
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX

_INDEX_TO_BOND_TYPE = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


@dataclass(frozen=True)
class V5SparseGraphRecord:
    """Offspring-word tree and closure record with no parent or dense-edge state."""

    structure_id: str
    canonical_smiles: str
    node_states: np.ndarray
    offspring: np.ndarray
    parent_bonds: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    closure_bonds: np.ndarray
    tree_traversal: str
    region_states: np.ndarray | None = None

    @property
    def node_count(self) -> int:
        return int(self.node_states.shape[0])

    @property
    def closure_count(self) -> int:
        return int(self.closure_left.shape[0])

    @property
    def parents(self) -> np.ndarray:
        """Derive parent indices exactly from the stored offspring word."""

        return v5_offspring_to_parents(self.offspring, self.tree_traversal)


def v5_offspring_to_parents(offspring: np.ndarray, tree_traversal: str) -> np.ndarray:
    """Decode the unique parent tree implied by a supported offspring language."""

    if tree_traversal == "breadth_first":
        return offspring_to_parents(offspring)
    if tree_traversal in {"breadth_first_tree_preorder", "depth_first_preorder"}:
        return preorder_offspring_to_parents(offspring)
    raise TreeTopologyFlowError(f"unsupported V5 tree traversal: {tree_traversal}")


def canonical_constitutional_molecule(molecule: Chem.Mol, structure_id: str) -> Chem.Mol:
    """Normalize atom order for the declared stereo-free constitutional state."""

    normalized_input = Chem.Mol(molecule)
    Chem.RemoveStereochemistry(normalized_input)
    for atom in normalized_input.GetAtoms():
        atom.SetAtomMapNum(0)
    canonical_smiles = Chem.MolToSmiles(
        normalized_input,
        canonical=True,
        isomericSmiles=False,
    )
    normalized = Chem.MolFromSmiles(canonical_smiles)
    if normalized is None:
        raise FeasibilityError(f"constitutional canonicalization failed: {structure_id}")
    if (
        normalized.GetNumAtoms() != molecule.GetNumAtoms()
        or normalized.GetNumBonds() != molecule.GetNumBonds()
    ):
        raise FeasibilityError(
            f"constitutional canonicalization changed graph size: {structure_id}"
        )
    return normalized


def _root(molecule: Chem.Mol, ranks: list[int], root_strategy: str) -> int:
    if root_strategy == "canonical":
        return min(range(molecule.GetNumAtoms()), key=lambda index: (ranks[index], index))
    if root_strategy == "lipid_polar":
        return select_lipid_polar_root(molecule)
    raise FeasibilityError(f"unsupported sparse root strategy: {root_strategy}")


def _breadth_first_tree(
    molecule: Chem.Mol,
    *,
    root_strategy: str,
) -> tuple[list[int], dict[int, int]]:
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    root = _root(molecule, ranks, root_strategy)
    order: list[int] = []
    parents = {root: root}
    queue = deque([root])
    seen = {root}
    while queue:
        atom_index = queue.popleft()
        order.append(atom_index)
        neighbors = sorted(
            (
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors()
                if neighbor.GetIdx() not in seen
            ),
            key=lambda index: (ranks[index], index),
        )
        for neighbor in neighbors:
            seen.add(neighbor)
            parents[neighbor] = atom_index
            queue.append(neighbor)
    if len(order) != molecule.GetNumAtoms():
        raise FeasibilityError("sparse topology requires a connected molecule")
    return order, parents


def _depth_first_tree(
    molecule: Chem.Mol,
    *,
    root_strategy: str,
) -> tuple[list[int], dict[int, int]]:
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    root = _root(molecule, ranks, root_strategy)
    order: list[int] = []
    parents = {root: root}
    seen = {root}
    stack: list[tuple[int, int]] = [(root, 0)]
    ordered_neighbors: dict[int, list[int]] = {}
    while stack:
        atom_index, next_neighbor = stack[-1]
        if next_neighbor == 0:
            order.append(atom_index)
            ordered_neighbors[atom_index] = sorted(
                (
                    neighbor.GetIdx()
                    for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors()
                ),
                key=lambda index: (ranks[index], index),
            )
        neighbors = ordered_neighbors[atom_index]
        if next_neighbor >= len(neighbors):
            stack.pop()
            continue
        neighbor = neighbors[next_neighbor]
        stack[-1] = (atom_index, next_neighbor + 1)
        if neighbor in seen:
            continue
        seen.add(neighbor)
        parents[neighbor] = atom_index
        stack.append((neighbor, 0))
    if len(order) != molecule.GetNumAtoms():
        raise FeasibilityError("sparse topology requires a connected molecule")
    return order, parents


def _breadth_first_tree_preorder(
    molecule: Chem.Mol,
    *,
    root_strategy: str,
) -> tuple[list[int], dict[int, int]]:
    breadth_first_order, parents = _breadth_first_tree(
        molecule,
        root_strategy=root_strategy,
    )
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    children: dict[int, list[int]] = {index: [] for index in breadth_first_order}
    for child in breadth_first_order[1:]:
        children[parents[child]].append(child)
    for values in children.values():
        values.sort(key=lambda index: (ranks[index], index))
    order: list[int] = []
    stack = [breadth_first_order[0]]
    while stack:
        atom_index = stack.pop()
        order.append(atom_index)
        stack.extend(reversed(children[atom_index]))
    if len(order) != molecule.GetNumAtoms():
        raise FeasibilityError("sparse topology requires a connected molecule")
    return order, parents


def canonical_tree(
    molecule: Chem.Mol,
    *,
    root_strategy: str,
    tree_traversal: str,
) -> tuple[list[int], dict[int, int]]:
    """Return the deterministic tree order and old-index parent map."""

    if tree_traversal == "breadth_first":
        return _breadth_first_tree(molecule, root_strategy=root_strategy)
    if tree_traversal == "breadth_first_tree_preorder":
        return _breadth_first_tree_preorder(molecule, root_strategy=root_strategy)
    if tree_traversal == "depth_first_preorder":
        return _depth_first_tree(molecule, root_strategy=root_strategy)
    raise FeasibilityError(f"unsupported sparse tree traversal: {tree_traversal}")


def tensorize_v5_sparse_row(
    row: Mapping[str, str],
    atom_to_index: Mapping[AtomState, int],
    *,
    preserve_aromaticity: bool = False,
    root_strategy: str = "canonical",
    region_scheme: str = "none",
    tree_traversal: str = "breadth_first_tree_preorder",
) -> V5SparseGraphRecord:
    """Encode one source row with the V5 canonical representation."""

    molecule = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
    if molecule is None:
        raise FeasibilityError(f"invalid SMILES: {row['canonical_isomeric_smiles']}")
    return tensorize_v5_sparse_molecule(
        molecule,
        atom_to_index,
        structure_id=row["r0_structure_id"],
        canonical_smiles=row["canonical_isomeric_smiles"],
        preserve_aromaticity=preserve_aromaticity,
        root_strategy=root_strategy,
        region_scheme=region_scheme,
        tree_traversal=tree_traversal,
    )


def tensorize_v5_sparse_molecule(
    molecule: Chem.Mol,
    atom_to_index: Mapping[AtomState, int],
    *,
    structure_id: str,
    canonical_smiles: str | None = None,
    preserve_aromaticity: bool = False,
    root_strategy: str = "canonical",
    region_scheme: str = "none",
    tree_traversal: str = "breadth_first_tree_preorder",
) -> V5SparseGraphRecord:
    """Encode an in-memory graph without a canonicalizing SMILES round trip first."""

    if molecule.GetNumAtoms() == 0:
        raise FeasibilityError(f"cannot tensorize an empty molecule: {structure_id}")
    original_smiles = canonical_smiles or Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    molecule = canonical_constitutional_molecule(molecule, structure_id)
    if not preserve_aromaticity:
        try:
            Chem.Kekulize(molecule, clearAromaticFlags=True)
        except (ValueError, RuntimeError) as exc:
            raise FeasibilityError(f"molecule cannot be kekulized: {structure_id}") from exc
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise FeasibilityError(f"sparse topology requires one fragment: {structure_id}")
    order, old_parent = canonical_tree(
        molecule,
        root_strategy=root_strategy,
        tree_traversal=tree_traversal,
    )
    if region_scheme == "none":
        region_states = None
    elif region_scheme == "polar_structural_v2":
        region_by_old_index = assign_lipid_regions(molecule, order[0])
        region_states = np.asarray(
            [region_by_old_index[old] for old in order],
            dtype=np.int64,
        )
    else:
        raise FeasibilityError(f"unsupported lipid region scheme: {region_scheme}")

    old_to_new = {old: new for new, old in enumerate(order)}
    node_states = np.asarray(
        [
            atom_to_index[
                AtomState(
                    molecule.GetAtomWithIdx(old).GetSymbol(),
                    molecule.GetAtomWithIdx(old).GetFormalCharge(),
                    (
                        molecule.GetAtomWithIdx(old).GetIsAromatic()
                        if preserve_aromaticity
                        else False
                    ),
                    (
                        molecule.GetAtomWithIdx(old).GetNumExplicitHs()
                        if preserve_aromaticity
                        else 0
                    ),
                )
            ]
            for old in order
        ],
        dtype=np.int64,
    )
    parents = np.zeros(len(order), dtype=np.int64)
    parent_bonds = np.zeros(len(order), dtype=np.int64)
    tree_edges: set[tuple[int, int]] = set()
    for new_child, old_child in enumerate(order[1:], start=1):
        old_parent_index = old_parent[old_child]
        new_parent = old_to_new[old_parent_index]
        parents[new_child] = new_parent
        pair = tuple(sorted((new_child, new_parent)))
        tree_edges.add(pair)
        bond = molecule.GetBondBetweenAtoms(old_child, old_parent_index)
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(f"unsupported bond type: {bond.GetBondType()}")
        parent_bonds[new_child] = SPARSE_BOND_TO_INDEX[bond.GetBondType()]

    closures: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        left = old_to_new[bond.GetBeginAtomIdx()]
        right = old_to_new[bond.GetEndAtomIdx()]
        pair = tuple(sorted((left, right)))
        if pair in tree_edges:
            continue
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(f"unsupported closure bond: {bond.GetBondType()}")
        closures.append((*pair, SPARSE_BOND_TO_INDEX[bond.GetBondType()]))
    closures.sort()
    closure_left = np.asarray([row[0] for row in closures], dtype=np.int64)
    closure_right = np.asarray([row[1] for row in closures], dtype=np.int64)
    closure_bonds = np.asarray([row[2] for row in closures], dtype=np.int64)
    offspring = np.bincount(parents[1:], minlength=len(order)).astype(np.int64)
    try:
        restored_parents = v5_offspring_to_parents(offspring, tree_traversal)
    except TreeTopologyFlowError as exc:
        raise FeasibilityError(
            f"canonical tree does not encode a valid offspring word: {structure_id}"
        ) from exc
    if not np.array_equal(restored_parents, parents):
        raise FeasibilityError(f"offspring word does not restore canonical tree: {structure_id}")

    return V5SparseGraphRecord(
        structure_id=structure_id,
        canonical_smiles=original_smiles,
        node_states=node_states,
        offspring=offspring,
        parent_bonds=parent_bonds,
        closure_left=closure_left,
        closure_right=closure_right,
        closure_bonds=closure_bonds,
        tree_traversal=tree_traversal,
        region_states=region_states,
    )


def v5_sparse_program_valid(
    record: V5SparseGraphRecord,
    *,
    atom_vocabulary_size: int | None = None,
    region_classes: int | None = None,
    maximum_children: int | None = None,
) -> bool:
    """Validate the exact linear-state tree and simple closure-edge program."""

    node_count = record.node_count
    integer_arrays = (
        record.node_states,
        record.offspring,
        record.parent_bonds,
        record.closure_left,
        record.closure_right,
        record.closure_bonds,
    )
    if record.region_states is not None:
        integer_arrays = (*integer_arrays, record.region_states)
    if (
        node_count < 1
        or record.node_states.shape != (node_count,)
        or record.offspring.shape != (node_count,)
        or record.parent_bonds.shape != (node_count,)
        or record.node_states.ndim != 1
        or any(not np.issubdtype(array.dtype, np.integer) for array in integer_arrays)
        or np.any(record.node_states < 0)
        or np.any(record.offspring < 0)
        or int(record.parent_bonds[0]) != 0
    ):
        return False
    if atom_vocabulary_size is not None and (
        atom_vocabulary_size < 1 or np.any(record.node_states >= atom_vocabulary_size)
    ):
        return False
    if maximum_children is not None and (
        maximum_children < 0 or np.any(record.offspring > maximum_children)
    ):
        return False
    if record.region_states is not None and (
        record.region_states.shape != (node_count,)
        or np.any(record.region_states < 0)
        or (
            region_classes is not None
            and (region_classes < 1 or np.any(record.region_states >= region_classes))
        )
    ):
        return False
    try:
        parents = record.parents
    except TreeTopologyFlowError:
        return False
    tree_edges: set[tuple[int, int]] = set()
    for child in range(1, node_count):
        parent = int(parents[child])
        if not 0 <= parent < child or int(record.parent_bonds[child]) not in _INDEX_TO_BOND_TYPE:
            return False
        tree_edges.add((parent, child))
    if len(tree_edges) != node_count - 1:
        return False
    if not (
        record.closure_left.shape == record.closure_right.shape == record.closure_bonds.shape
        and record.closure_left.ndim == 1
    ):
        return False
    closure_edges: set[tuple[int, int]] = set()
    closure_sequence: list[tuple[int, int]] = []
    for left, right, bond in zip(
        record.closure_left,
        record.closure_right,
        record.closure_bonds,
        strict=True,
    ):
        pair = (int(left), int(right))
        if (
            not 0 <= pair[0] < pair[1] < node_count
            or pair in tree_edges
            or pair in closure_edges
            or int(bond) not in _INDEX_TO_BOND_TYPE
        ):
            return False
        closure_edges.add(pair)
        closure_sequence.append(pair)
    return closure_sequence == sorted(closure_sequence)


def v5_graph_to_molecule(
    record: V5SparseGraphRecord,
    atom_vocabulary: tuple[AtomState, ...],
) -> Chem.Mol:
    """Construct and sanitize a molecule from only present sparse edges."""

    if not v5_sparse_program_valid(
        record,
        atom_vocabulary_size=len(atom_vocabulary),
    ):
        raise FeasibilityError(f"invalid V5 sparse program: {record.structure_id}")
    parents = record.parents
    editable = Chem.RWMol()
    for node in record.node_states:
        state = atom_vocabulary[int(node)]
        atom = Chem.Atom(state.symbol)
        atom.SetFormalCharge(state.formal_charge)
        atom.SetIsAromatic(state.aromatic)
        if state.explicit_hydrogens:
            atom.SetNumExplicitHs(state.explicit_hydrogens)
            atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for child in range(1, record.node_count):
        editable.AddBond(
            int(parents[child]),
            child,
            _INDEX_TO_BOND_TYPE[int(record.parent_bonds[child])],
        )
    for left, right, bond in zip(
        record.closure_left,
        record.closure_right,
        record.closure_bonds,
        strict=True,
    ):
        editable.AddBond(int(left), int(right), _INDEX_TO_BOND_TYPE[int(bond)])
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def v5_constitutional_roundtrip_exact(
    record: V5SparseGraphRecord,
    atom_vocabulary: tuple[AtomState, ...],
) -> bool:
    """Require the sparse program to restore the source constitutional graph."""

    original = Chem.MolFromSmiles(record.canonical_smiles)
    if original is None:
        raise FeasibilityError(f"invalid original SMILES: {record.canonical_smiles}")
    reconstructed = v5_graph_to_molecule(record, atom_vocabulary)
    original_smiles = Chem.MolToSmiles(original, canonical=True, isomericSmiles=False)
    reconstructed_smiles = Chem.MolToSmiles(
        reconstructed,
        canonical=True,
        isomericSmiles=False,
    )
    return original_smiles == reconstructed_smiles
