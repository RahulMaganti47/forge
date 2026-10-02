"""Role-blocked sparse graphs for vocabulary-free reaction-program conditioning."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from forge.model.conditioning.reaction_program import ReactionProgramVocabulary
from forge.model.networks.dense_flow import AtomState
from forge.model.networks.sparse_flow import (
    INDEX_TO_DENSE_BOND,
    SPARSE_BOND_TO_INDEX,
    SparseGraphRecord,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
)


class ReactionProgramGraphError(ValueError):
    """A product graph cannot be represented under its exact program semantics."""


@dataclass(frozen=True)
class ReactionProgramGraphRecord:
    """One complete product graph with role semantics but no component identity."""

    graph: SparseGraphRecord
    program_id: str
    program_state: int
    program_depth: int
    role_states: np.ndarray
    core_position_states: np.ndarray
    repeat_component_count: int
    accumulator_atom_count: int
    repeat_atom_counts: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.role_states.shape != (self.graph.node_count,)
            or self.core_position_states.shape != (self.graph.node_count,)
            or self.program_state < 1
            or self.program_depth < 1
            or self.repeat_component_count != self.program_depth
            or len(self.repeat_atom_counts) != self.program_depth
            or self.accumulator_atom_count + sum(self.repeat_atom_counts) != self.graph.node_count
        ):
            raise ReactionProgramGraphError("reaction-program graph metadata is inconsistent")

    @property
    def node_count(self) -> int:
        return self.graph.node_count


def _induced_components(molecule: Chem.Mol, atoms: set[int]) -> tuple[tuple[int, ...], ...]:
    components: list[tuple[int, ...]] = []
    remaining = set(atoms)
    while remaining:
        start = min(remaining)
        seen = {start}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                index = neighbor.GetIdx()
                if index in remaining and index not in seen:
                    seen.add(index)
                    queue.append(index)
        remaining.difference_update(seen)
        components.append(tuple(sorted(seen)))
    return tuple(components)


def _preorder_component(
    molecule: Chem.Mol,
    component: set[int],
    root: int,
    ranks: Sequence[int],
) -> tuple[list[int], dict[int, int]]:
    order: list[int] = []
    parents = {root: root}
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        order.append(current)
        neighbors = sorted(
            (
                atom.GetIdx()
                for atom in molecule.GetAtomWithIdx(current).GetNeighbors()
                if atom.GetIdx() in component and atom.GetIdx() not in seen
            ),
            key=lambda index: (ranks[index], index),
            reverse=True,
        )
        for neighbor in neighbors:
            seen.add(neighbor)
            parents[neighbor] = current
            stack.append(neighbor)
    if seen != component:
        raise ReactionProgramGraphError("precursor-origin component is disconnected")
    return order, parents


def build_reaction_program_atom_vocabulary(
    product_smiles: Sequence[str],
    *,
    declared_elements: set[str],
) -> tuple[AtomState, ...]:
    """Build a constitutional atom vocabulary from training products only."""

    states: set[AtomState] = set()
    for smiles in product_smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise ReactionProgramGraphError(f"invalid connected product graph: {smiles!r}")
        elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
        if not elements.issubset(declared_elements):
            raise ReactionProgramGraphError(
                f"product elements {sorted(elements)} exceed {sorted(declared_elements)}"
            )
        molecule = Chem.Mol(molecule)
        try:
            Chem.Kekulize(molecule, clearAromaticFlags=True)
        except (ValueError, RuntimeError) as error:
            raise ReactionProgramGraphError(f"product cannot be kekulized: {smiles!r}") from error
        states.update(
            AtomState(atom.GetSymbol(), atom.GetFormalCharge(), False, 0)
            for atom in molecule.GetAtoms()
        )
    if not states:
        raise ReactionProgramGraphError("training products yielded an empty atom vocabulary")
    return tuple(sorted(states, key=AtomState.key))


def tensorize_reaction_program_product(
    *,
    record_id: str,
    program_id: str,
    canonical_product_smiles: str,
    atom_roles: Sequence[str],
    atom_core_positions: Sequence[str] | None = None,
    program_depth: int,
    accumulator_role: str,
    repeat_role: str,
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[AtomState],
) -> ReactionProgramGraphRecord:
    """Serialize a complete graph as accumulator then exchangeable repeated-role components."""

    molecule = Chem.MolFromSmiles(canonical_product_smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ReactionProgramGraphError(f"invalid connected product graph: {record_id}")
    if len(atom_roles) != molecule.GetNumAtoms():
        raise ReactionProgramGraphError(f"semantic atom count changed: {record_id}")
    if atom_core_positions is None:
        atom_core_positions = ("exterior",) * molecule.GetNumAtoms()
    if len(atom_core_positions) != molecule.GetNumAtoms():
        raise ReactionProgramGraphError(f"core-position atom count changed: {record_id}")
    if set(atom_roles) != {accumulator_role, repeat_role}:
        raise ReactionProgramGraphError(f"product has unexpected precursor roles: {record_id}")
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except (ValueError, RuntimeError) as error:
        raise ReactionProgramGraphError(f"product cannot be kekulized: {record_id}") from error

    accumulator_atoms = {index for index, role in enumerate(atom_roles) if role == accumulator_role}
    repeat_atoms = {index for index, role in enumerate(atom_roles) if role == repeat_role}
    accumulator_components = _induced_components(molecule, accumulator_atoms)
    repeat_components = _induced_components(molecule, repeat_atoms)
    if len(accumulator_components) != 1:
        raise ReactionProgramGraphError(f"accumulator origin is not connected: {record_id}")
    if len(repeat_components) != program_depth:
        raise ReactionProgramGraphError(
            f"repeat-origin components ({len(repeat_components)}) differ from program depth "
            f"({program_depth}): {record_id}"
        )

    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    accumulator = set(accumulator_components[0])
    accumulator_root = min(accumulator, key=lambda index: (ranks[index], index))
    order, old_parents = _preorder_component(
        molecule,
        accumulator,
        accumulator_root,
        ranks,
    )
    cross_bonds: dict[int, tuple[int, Chem.Bond]] = {}
    for component_tuple in repeat_components:
        component = set(component_tuple)
        candidates = [
            (atom, neighbor.GetIdx(), molecule.GetBondBetweenAtoms(atom, neighbor.GetIdx()))
            for atom in component
            for neighbor in molecule.GetAtomWithIdx(atom).GetNeighbors()
            if neighbor.GetIdx() in accumulator
        ]
        if len(candidates) != 1:
            raise ReactionProgramGraphError(
                f"repeat component needs one accumulator attachment, found {len(candidates)}: "
                f"{record_id}"
            )
        root, parent, bond = candidates[0]
        cross_bonds[root] = (parent, bond)
    ordered_repeat_components = sorted(
        (set(component) for component in repeat_components),
        key=lambda component: (
            ranks[next(root for root in component if root in cross_bonds)],
            tuple(sorted(ranks[index] for index in component)),
        ),
    )
    repeat_atom_counts: list[int] = []
    for component in ordered_repeat_components:
        root = next(index for index in component if index in cross_bonds)
        component_order, component_parents = _preorder_component(
            molecule,
            component,
            root,
            ranks,
        )
        old_parents.update(component_parents)
        old_parents[root] = cross_bonds[root][0]
        order.extend(component_order)
        repeat_atom_counts.append(len(component_order))

    old_to_new = {old: new for new, old in enumerate(order)}
    atom_to_index = {state: index for index, state in enumerate(atom_vocabulary)}
    try:
        node_states = np.asarray(
            [
                atom_to_index[
                    AtomState(
                        molecule.GetAtomWithIdx(old).GetSymbol(),
                        molecule.GetAtomWithIdx(old).GetFormalCharge(),
                        False,
                        0,
                    )
                ]
                for old in order
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ReactionProgramGraphError(
            f"{record_id} contains an atom state absent from the training vocabulary"
        ) from error
    parents = np.zeros(len(order), dtype=np.int64)
    parent_bonds = np.zeros(len(order), dtype=np.int64)
    tree_edges: set[tuple[int, int]] = set()
    for new_child, old_child in enumerate(order[1:], start=1):
        old_parent = old_parents[old_child]
        new_parent = old_to_new[old_parent]
        if new_parent >= new_child:
            raise ReactionProgramGraphError("role-blocked parent does not precede its child")
        parents[new_child] = new_parent
        bond = molecule.GetBondBetweenAtoms(old_child, old_parent)
        if bond is None or bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise ReactionProgramGraphError(f"unsupported tree bond: {record_id}")
        parent_bonds[new_child] = SPARSE_BOND_TO_INDEX[bond.GetBondType()]
        tree_edges.add((min(new_child, new_parent), max(new_child, new_parent)))

    closures: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        begin = old_to_new[bond.GetBeginAtomIdx()]
        end = old_to_new[bond.GetEndAtomIdx()]
        pair = (min(begin, end), max(begin, end))
        if pair in tree_edges:
            continue
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise ReactionProgramGraphError(f"unsupported closure bond: {record_id}")
        closures.append((*pair, SPARSE_BOND_TO_INDEX[bond.GetBondType()]))
    closures.sort()
    edges = np.zeros((len(order), len(order)), dtype=np.int64)
    for child in range(1, len(order)):
        parent = int(parents[child])
        dense = INDEX_TO_DENSE_BOND[int(parent_bonds[child])]
        edges[child, parent] = edges[parent, child] = dense
    for left, right, bond_state in closures:
        dense = INDEX_TO_DENSE_BOND[bond_state]
        edges[left, right] = edges[right, left] = dense
    graph = SparseGraphRecord(
        structure_id=record_id,
        canonical_smiles=canonical_product_smiles,
        node_states=node_states,
        parents=parents,
        parent_bonds=parent_bonds,
        closure_left=np.asarray([row[0] for row in closures], dtype=np.int64),
        closure_right=np.asarray([row[1] for row in closures], dtype=np.int64),
        closure_bonds=np.asarray([row[2] for row in closures], dtype=np.int64),
        edges=edges,
    )
    if not sparse_roundtrip_exact(graph) or not sparse_constitutional_roundtrip_exact(
        graph, atom_vocabulary
    ):
        raise ReactionProgramGraphError(f"sparse constitutional round trip failed: {record_id}")
    role_to_index = vocabulary.role_to_index
    role_states = np.asarray([role_to_index[atom_roles[old]] for old in order], dtype=np.int64)
    core_to_index = vocabulary.core_position_to_index
    try:
        core_position_states = np.asarray(
            [core_to_index[atom_core_positions[old]] for old in order],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ReactionProgramGraphError(
            f"{record_id} contains a core position outside the program vocabulary"
        ) from error
    return ReactionProgramGraphRecord(
        graph=graph,
        program_id=program_id,
        program_state=vocabulary.program_to_index[program_id],
        program_depth=program_depth,
        role_states=role_states,
        core_position_states=core_position_states,
        repeat_component_count=len(ordered_repeat_components),
        accumulator_atom_count=len(accumulator),
        repeat_atom_counts=tuple(repeat_atom_counts),
    )


__all__ = [
    "ReactionProgramGraphError",
    "ReactionProgramGraphRecord",
    "build_reaction_program_atom_vocabulary",
    "tensorize_reaction_program_product",
]
