"""Shared whole-product graphs with reaction-program semantic coordinates.

The representation contains a complete molecular graph, an evidenced reaction-program identity,
per-atom precursor-origin roles, and namespaced reaction-core positions.  It deliberately contains
no component identifier, component fingerprint, or fragment token.  Component blocks below are
serialization spans derived from atom origins; they are not a purchasable-building-block vocabulary.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.sparse_topology_feasibility import (
    INDEX_TO_DENSE_BOND,
    SPARSE_BOND_TO_INDEX,
    SparseGraphRecord,
    sparse_constitutional_roundtrip_exact,
    sparse_roundtrip_exact,
)


class SynthesisProgramGraphError(ValueError):
    """A product cannot be represented under its declared program semantics."""


@dataclass(frozen=True)
class SynthesisProgramComponentBlock:
    """One contiguous atom-origin component in serialized node order."""

    role: str
    role_state: int
    start: int
    stop: int

    def __post_init__(self) -> None:
        if not self.role or self.role_state < 1 or self.start < 0 or self.stop <= self.start:
            raise SynthesisProgramGraphError("synthesis-program component block is invalid")

    @property
    def atom_count(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class SynthesisProgramGraphRecord:
    """A complete sparse product graph plus vocabulary-free program coordinates."""

    graph: SparseGraphRecord
    canonical_atom_order: np.ndarray
    program_id: str
    program_state: int
    program_depth: int
    role_states: np.ndarray
    core_position_states: np.ndarray
    component_blocks: tuple[SynthesisProgramComponentBlock, ...]
    fixed_atom_mask: np.ndarray
    fixed_parent_bond_mask: np.ndarray
    fixed_closure_bond_mask: np.ndarray
    # Optional node-aligned coarse program coordinates.  Column order is exterior-node count,
    # junction budget, cycle rank and core-attachment count.  Zero is the unconditioned state;
    # observed integer values are stored at value + 1.  Exact cache records derive this view at
    # collation time, while synthetic sampling layouts may supply an explicit program projection.
    role_morphology_states: np.ndarray | None = None

    def __post_init__(self) -> None:
        node_count = self.graph.node_count
        if (
            self.canonical_atom_order.shape != (node_count,)
            or sorted(self.canonical_atom_order.tolist()) != list(range(node_count))
            or self.role_states.shape != (node_count,)
            or self.core_position_states.shape != (node_count,)
            or self.fixed_atom_mask.shape != (node_count,)
            or self.fixed_parent_bond_mask.shape != (node_count,)
            or self.fixed_closure_bond_mask.shape != (self.graph.closure_count,)
            or self.fixed_atom_mask.dtype != np.bool_
            or self.fixed_parent_bond_mask.dtype != np.bool_
            or self.fixed_closure_bond_mask.dtype != np.bool_
            or self.program_state < 1
            or self.program_depth < 1
            or not self.component_blocks
        ):
            raise SynthesisProgramGraphError("synthesis-program graph metadata is inconsistent")
        if self.role_morphology_states is not None and (
            self.role_morphology_states.shape != (node_count, 4)
            or not np.issubdtype(self.role_morphology_states.dtype, np.integer)
            or np.any(self.role_morphology_states < 0)
        ):
            raise SynthesisProgramGraphError("role-local morphology states are inconsistent")
        cursor = 0
        for block in self.component_blocks:
            if block.start != cursor or block.stop > node_count:
                raise SynthesisProgramGraphError(
                    "synthesis-program component blocks do not partition the graph"
                )
            if not np.all(self.role_states[block.start : block.stop] == block.role_state):
                raise SynthesisProgramGraphError(
                    "synthesis-program component block and node roles disagree"
                )
            cursor = block.stop
        if cursor != node_count:
            raise SynthesisProgramGraphError(
                "synthesis-program component blocks do not cover the graph"
            )
        if self.fixed_parent_bond_mask[0]:
            raise SynthesisProgramGraphError("the root sentinel cannot be a fixed parent bond")
        if np.any(
            self.fixed_parent_bond_mask
            & ~(
                self.fixed_atom_mask
                | self.fixed_atom_mask[self.graph.parents]
                | (
                    (self.core_position_states > 1)
                    & (self.core_position_states[self.graph.parents] > 1)
                )
            )
        ):
            raise SynthesisProgramGraphError(
                "a fixed parent bond must touch an adapter-fixed atom or join declared "
                "reaction-core coordinates"
            )
        for slot, fixed in enumerate(self.fixed_closure_bond_mask.tolist()):
            if not fixed:
                continue
            left = int(self.graph.closure_left[slot])
            right = int(self.graph.closure_right[slot])
            if not self.fixed_atom_mask[left] or not self.fixed_atom_mask[right]:
                raise SynthesisProgramGraphError(
                    "a fixed closure bond must join two adapter-fixed atoms"
                )

    @property
    def node_count(self) -> int:
        return self.graph.node_count

    @property
    def component_count(self) -> int:
        return len(self.component_blocks)


@dataclass(frozen=True)
class _OriginComponent:
    index: int
    role: str
    atoms: tuple[int, ...]


def _origin_components(
    molecule: Chem.Mol,
    atom_roles: Sequence[str],
) -> tuple[_OriginComponent, ...]:
    components: list[_OriginComponent] = []
    for role in sorted(set(atom_roles)):
        remaining = {index for index, value in enumerate(atom_roles) if value == role}
        while remaining:
            root = min(remaining)
            seen = {root}
            queue = deque([root])
            while queue:
                current = queue.popleft()
                for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                    index = neighbor.GetIdx()
                    if index in remaining and index not in seen:
                        seen.add(index)
                        queue.append(index)
            remaining.difference_update(seen)
            components.append(
                _OriginComponent(
                    index=len(components),
                    role=role,
                    atoms=tuple(sorted(seen)),
                )
            )
    return tuple(components)


def _component_preorder(
    molecule: Chem.Mol,
    component: _OriginComponent,
    root: int,
    ranks: Sequence[int],
) -> tuple[list[int], dict[int, int]]:
    allowed = set(component.atoms)
    if root not in allowed:
        raise SynthesisProgramGraphError("component root lies outside its origin block")
    order: list[int] = []
    parents = {root: root}
    seen = {root}
    stack = [root]
    while stack:
        current = stack.pop()
        order.append(current)
        neighbors = sorted(
            (
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors()
                if neighbor.GetIdx() in allowed and neighbor.GetIdx() not in seen
            ),
            key=lambda index: (ranks[index], index),
            reverse=True,
        )
        for neighbor in neighbors:
            seen.add(neighbor)
            parents[neighbor] = current
            stack.append(neighbor)
    if seen != allowed:
        raise SynthesisProgramGraphError("origin component traversal is incomplete")
    return order, parents


def _serialize_components(
    molecule: Chem.Mol,
    atom_roles: Sequence[str],
    core_atoms: set[int],
    role_to_index: dict[str, int],
) -> tuple[list[int], dict[int, int], tuple[SynthesisProgramComponentBlock, ...]]:
    components = _origin_components(molecule, atom_roles)
    atom_to_component = {
        atom: component.index for component in components for atom in component.atoms
    }
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    root_atom = min(core_atoms, key=lambda index: (ranks[index], index))
    root_component = atom_to_component[root_atom]

    cross_bonds: dict[tuple[int, int], list[Chem.Bond]] = {}
    adjacency: dict[int, set[int]] = {component.index: set() for component in components}
    for bond in molecule.GetBonds():
        left = atom_to_component[bond.GetBeginAtomIdx()]
        right = atom_to_component[bond.GetEndAtomIdx()]
        if left == right:
            continue
        pair = (min(left, right), max(left, right))
        cross_bonds.setdefault(pair, []).append(bond)
        adjacency[left].add(right)
        adjacency[right].add(left)

    def component_key(index: int) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        component = components[index]
        return (
            role_to_index[component.role],
            tuple(sorted(ranks[atom] for atom in component.atoms)),
            component.atoms,
        )

    component_order: list[int] = []
    component_parent_bond: dict[int, Chem.Bond] = {}
    seen = {root_component}
    queue = deque([root_component])
    while queue:
        current = queue.popleft()
        component_order.append(current)
        for neighbor in sorted(adjacency[current], key=component_key):
            if neighbor in seen:
                continue
            pair = (min(current, neighbor), max(current, neighbor))
            candidates = cross_bonds[pair]

            def bond_key(bond: Chem.Bond) -> tuple[int, int, int, int, int]:
                begin = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                current_atom = begin if atom_to_component[begin] == current else end
                neighbor_atom = end if current_atom == begin else begin
                return (
                    ranks[current_atom],
                    ranks[neighbor_atom],
                    current_atom,
                    neighbor_atom,
                    SPARSE_BOND_TO_INDEX.get(bond.GetBondType(), 99),
                )

            component_parent_bond[neighbor] = min(candidates, key=bond_key)
            seen.add(neighbor)
            queue.append(neighbor)
    if len(component_order) != len(components):
        raise SynthesisProgramGraphError("origin-component graph is disconnected")

    atom_order: list[int] = []
    atom_parents: dict[int, int] = {}
    blocks: list[SynthesisProgramComponentBlock] = []
    for component_index in component_order:
        component = components[component_index]
        if component_index == root_component:
            component_root = root_atom
            parent_atom = None
        else:
            parent_bond = component_parent_bond[component_index]
            begin = parent_bond.GetBeginAtomIdx()
            end = parent_bond.GetEndAtomIdx()
            component_root = begin if atom_to_component[begin] == component_index else end
            parent_atom = end if component_root == begin else begin
        block_order, block_parents = _component_preorder(
            molecule,
            component,
            component_root,
            ranks,
        )
        if parent_atom is not None:
            atom_parents[component_root] = parent_atom
        atom_parents.update(
            {atom: parent for atom, parent in block_parents.items() if atom != component_root}
        )
        start = len(atom_order)
        atom_order.extend(block_order)
        blocks.append(
            SynthesisProgramComponentBlock(
                role=component.role,
                role_state=role_to_index[component.role],
                start=start,
                stop=len(atom_order),
            )
        )
    atom_parents[root_atom] = root_atom
    return atom_order, atom_parents, tuple(blocks)


def tensorize_synthesis_program_product(
    *,
    record_id: str,
    program_id: str,
    canonical_product_smiles: str,
    atom_roles: Sequence[str],
    atom_core_positions: Sequence[str],
    program_depth: int,
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[AtomState],
    fixed_atom_indices: Sequence[int] = (),
) -> SynthesisProgramGraphRecord:
    """Tensorize one exact semantic product without fragment or component identity tokens."""

    molecule = Chem.MolFromSmiles(canonical_product_smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise SynthesisProgramGraphError(f"invalid connected product graph: {record_id}")
    node_count = molecule.GetNumAtoms()
    if len(atom_roles) != node_count or len(atom_core_positions) != node_count:
        raise SynthesisProgramGraphError(f"semantic atom count changed: {record_id}")
    if program_id not in vocabulary.program_to_index:
        raise SynthesisProgramGraphError(f"unknown reaction program: {program_id}")
    if program_depth < 1 or program_depth > vocabulary.maximum_steps:
        raise SynthesisProgramGraphError(f"program depth lies outside support: {record_id}")
    role_to_index = vocabulary.role_to_index
    unknown_roles = sorted(set(atom_roles).difference(role_to_index))
    if unknown_roles or "unassigned" in atom_roles:
        raise SynthesisProgramGraphError(
            f"{record_id} contains roles outside the program vocabulary: {unknown_roles}"
        )
    core_to_index = vocabulary.core_position_to_index
    unknown_positions = sorted(set(atom_core_positions).difference(core_to_index))
    if unknown_positions or "unconditioned" in atom_core_positions:
        raise SynthesisProgramGraphError(
            f"{record_id} contains core positions outside the program vocabulary: "
            f"{unknown_positions}"
        )
    core_atoms = {
        index for index, position in enumerate(atom_core_positions) if position != "exterior"
    }
    if not core_atoms:
        raise SynthesisProgramGraphError(f"reaction program has no core atoms: {record_id}")
    fixed = {int(index) for index in fixed_atom_indices}
    if len(fixed) != len(tuple(fixed_atom_indices)) or any(
        index < 0 or index >= node_count for index in fixed
    ):
        raise SynthesisProgramGraphError(f"fixed atom indices are invalid: {record_id}")
    if not fixed.issubset(core_atoms):
        raise SynthesisProgramGraphError(f"fixed atoms must be reaction-core atoms: {record_id}")

    atom_order, old_parents, blocks = _serialize_components(
        molecule,
        atom_roles,
        core_atoms,
        role_to_index,
    )
    old_to_new = {old: new for new, old in enumerate(atom_order)}
    atom_to_index = {state: index for index, state in enumerate(atom_vocabulary)}
    try:
        node_states = np.asarray(
            [
                atom_to_index[
                    AtomState(
                        molecule.GetAtomWithIdx(old).GetSymbol(),
                        molecule.GetAtomWithIdx(old).GetFormalCharge(),
                        molecule.GetAtomWithIdx(old).GetIsAromatic(),
                        molecule.GetAtomWithIdx(old).GetNumExplicitHs(),
                    )
                ]
                for old in atom_order
            ],
            dtype=np.int64,
        )
    except KeyError as error:
        raise SynthesisProgramGraphError(
            f"{record_id} contains an atom state absent from the pinned vocabulary"
        ) from error

    parents = np.zeros(node_count, dtype=np.int64)
    parent_bonds = np.zeros(node_count, dtype=np.int64)
    tree_edges: set[tuple[int, int]] = set()
    for new_child, old_child in enumerate(atom_order[1:], start=1):
        old_parent = old_parents[old_child]
        new_parent = old_to_new[old_parent]
        if new_parent >= new_child:
            raise SynthesisProgramGraphError(
                f"serialized parent does not precede its child: {record_id}"
            )
        bond = molecule.GetBondBetweenAtoms(old_child, old_parent)
        if bond is None or bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise SynthesisProgramGraphError(f"unsupported tree bond: {record_id}")
        parents[new_child] = new_parent
        parent_bonds[new_child] = SPARSE_BOND_TO_INDEX[bond.GetBondType()]
        tree_edges.add((min(new_child, new_parent), max(new_child, new_parent)))

    closures: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        left = old_to_new[bond.GetBeginAtomIdx()]
        right = old_to_new[bond.GetEndAtomIdx()]
        pair = (min(left, right), max(left, right))
        if pair in tree_edges:
            continue
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise SynthesisProgramGraphError(f"unsupported closure bond: {record_id}")
        closures.append((*pair, SPARSE_BOND_TO_INDEX[bond.GetBondType()]))
    closures.sort()
    fixed_parent_bond_mask = np.zeros(node_count, dtype=np.bool_)
    if fixed:
        for new_child, old_child in enumerate(atom_order[1:], start=1):
            old_parent = old_parents[old_child]
            both_fixed = old_child in fixed and old_parent in fixed
            precursor_attachment = (
                (old_child in fixed) != (old_parent in fixed)
                and atom_roles[old_child] == atom_roles[old_parent]
                and parent_bonds[new_child] == 0
            )
            if both_fixed or precursor_attachment:
                fixed_parent_bond_mask[new_child] = True
    fixed_closure_bond_mask = np.asarray(
        [atom_order[left] in fixed and atom_order[right] in fixed for left, right, _ in closures],
        dtype=np.bool_,
    )
    edges = np.zeros((node_count, node_count), dtype=np.int64)
    for child in range(1, node_count):
        parent = int(parents[child])
        dense_bond = INDEX_TO_DENSE_BOND[int(parent_bonds[child])]
        edges[child, parent] = edges[parent, child] = dense_bond
    for left, right, bond_state in closures:
        dense_bond = INDEX_TO_DENSE_BOND[bond_state]
        edges[left, right] = edges[right, left] = dense_bond
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
        raise SynthesisProgramGraphError(f"sparse constitutional round trip failed: {record_id}")
    return SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.asarray(atom_order, dtype=np.int64),
        program_id=program_id,
        program_state=vocabulary.program_to_index[program_id],
        program_depth=program_depth,
        role_states=np.asarray(
            [role_to_index[atom_roles[old]] for old in atom_order],
            dtype=np.int64,
        ),
        core_position_states=np.asarray(
            [core_to_index[atom_core_positions[old]] for old in atom_order],
            dtype=np.int64,
        ),
        component_blocks=blocks,
        fixed_atom_mask=np.asarray([old in fixed for old in atom_order], dtype=np.bool_),
        fixed_parent_bond_mask=fixed_parent_bond_mask,
        fixed_closure_bond_mask=fixed_closure_bond_mask,
    )


__all__ = [
    "SynthesisProgramComponentBlock",
    "SynthesisProgramGraphError",
    "SynthesisProgramGraphRecord",
    "tensorize_synthesis_program_product",
]
