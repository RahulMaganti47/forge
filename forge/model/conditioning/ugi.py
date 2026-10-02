"""Exact adapter-defined node semantics for Ugi-conditioned lipid generation.

The canonical tree root is only a serialization coordinate.  Chemical position
is represented by atom-mapped precursor provenance, orthogonal reaction-core
position, role-anchor identity, and graph distances that are derived after a
valid topology exists.  No component catalog identifier enters model state.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from forge.model.networks.dense_flow import AtomState, FeasibilityError
from forge.model.networks.sparse_flow import SPARSE_BOND_TO_INDEX
from forge.model.representation.lipid_skeleton import (
    FUNCTIONAL_SUPPORT,
    LipidSupportSkeleton,
    encode_lipid_support_skeleton,
    skeleton_roundtrip_exact,
)
from forge.model.representation.sparse_graph import (
    V5SparseGraphRecord,
    canonical_constitutional_molecule,
    v5_offspring_to_parents,
)
from forge.potency.annotations import ROLE_NAMES

ASSEMBLY_INTRODUCED = "assembly_introduced"
ORIGIN_STATES = ("adapter_unspecified", *ROLE_NAMES, ASSEMBLY_INTRODUCED)
ORIGIN_TO_INDEX = {name: index for index, name in enumerate(ORIGIN_STATES)}
CORE_POSITION_STATES = (
    "not_core",
    "map_1",
    "map_2",
    "map_3",
    "map_4",
    "template_introduced_0",
)
CORE_POSITION_TO_INDEX = {name: index for index, name in enumerate(CORE_POSITION_STATES)}
PORT_STATES = ("not_port", *ROLE_NAMES)
PORT_TO_INDEX = {name: index for index, name in enumerate(PORT_STATES)}
NOT_APPLICABLE_DISTANCE = -1
_INDEX_TO_BOND_TYPE = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


def adapter_canonical_tree(
    molecule: Chem.Mol,
    *,
    core_candidates: Sequence[int],
    tree_traversal: str,
) -> tuple[list[int], dict[int, int]]:
    """Select a deterministic core root without modifying the shared serializer."""

    candidates = tuple(int(index) for index in core_candidates)
    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(index < 0 or index >= molecule.GetNumAtoms() for index in candidates)
    ):
        raise FeasibilityError("adapter core candidates are invalid")
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    root = min(candidates, key=lambda index: (ranks[index], index))

    if tree_traversal in {"breadth_first", "breadth_first_tree_preorder"}:
        breadth_order: list[int] = []
        parents = {root: root}
        queue = deque([root])
        seen = {root}
        while queue:
            atom_index = queue.popleft()
            breadth_order.append(atom_index)
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
        if tree_traversal == "breadth_first":
            order = breadth_order
        else:
            children: dict[int, list[int]] = {index: [] for index in breadth_order}
            for child in breadth_order[1:]:
                children[parents[child]].append(child)
            for values in children.values():
                values.sort(key=lambda index: (ranks[index], index))
            order = []
            stack = [root]
            while stack:
                atom_index = stack.pop()
                order.append(atom_index)
                stack.extend(reversed(children[atom_index]))
    elif tree_traversal == "depth_first_preorder":
        order = []
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
    else:
        raise FeasibilityError(f"unsupported adapter tree traversal: {tree_traversal}")
    if len(order) != molecule.GetNumAtoms():
        raise FeasibilityError("adapter topology requires a connected molecule")
    return order, parents


def _tensorize_adapter_sparse_molecule(
    molecule: Chem.Mol,
    atom_to_index: Mapping[AtomState, int],
    *,
    structure_id: str,
    canonical_smiles: str,
    core_candidates: Sequence[int],
    preserve_aromaticity: bool,
    tree_traversal: str,
) -> V5SparseGraphRecord:
    """Tensorize one canonical graph with adapter-owned root semantics."""

    molecule = canonical_constitutional_molecule(molecule, structure_id)
    if not preserve_aromaticity:
        try:
            Chem.Kekulize(molecule, clearAromaticFlags=True)
        except (ValueError, RuntimeError) as exc:
            raise FeasibilityError(f"molecule cannot be kekulized: {structure_id}") from exc
    order, old_parent = adapter_canonical_tree(
        molecule,
        core_candidates=core_candidates,
        tree_traversal=tree_traversal,
    )
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
        tree_edges.add(tuple(sorted((new_child, new_parent))))
        bond = molecule.GetBondBetweenAtoms(old_child, old_parent_index)
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(f"unsupported adapter tree bond: {bond.GetBondType()}")
        parent_bonds[new_child] = SPARSE_BOND_TO_INDEX[bond.GetBondType()]
    closures: list[tuple[int, int, int]] = []
    for bond in molecule.GetBonds():
        pair = tuple(
            sorted(
                (
                    old_to_new[bond.GetBeginAtomIdx()],
                    old_to_new[bond.GetEndAtomIdx()],
                )
            )
        )
        if pair in tree_edges:
            continue
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(f"unsupported adapter closure bond: {bond.GetBondType()}")
        closures.append((*pair, SPARSE_BOND_TO_INDEX[bond.GetBondType()]))
    closures.sort()
    offspring = np.bincount(parents[1:], minlength=len(order)).astype(np.int64)
    restored = v5_offspring_to_parents(offspring, tree_traversal)
    if not np.array_equal(restored, parents):
        raise FeasibilityError(f"adapter offspring word changed parents: {structure_id}")
    return V5SparseGraphRecord(
        structure_id=structure_id,
        canonical_smiles=canonical_smiles,
        node_states=node_states,
        offspring=offspring,
        parent_bonds=parent_bonds,
        closure_left=np.asarray([row[0] for row in closures], dtype=np.int64),
        closure_right=np.asarray([row[1] for row in closures], dtype=np.int64),
        closure_bonds=np.asarray([row[2] for row in closures], dtype=np.int64),
        tree_traversal=tree_traversal,
        region_states=None,
    )


@dataclass(frozen=True)
class UgiAdapterNodeFeatures:
    """Adapter semantics in the same atom order as one sparse graph record."""

    origin_states: np.ndarray
    core_position_states: np.ndarray
    port_states: np.ndarray
    distance_to_core: np.ndarray
    distance_to_own_port: np.ndarray
    distances_to_all_ports: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.origin_states.size)

    @property
    def core_membership(self) -> np.ndarray:
        return self.core_position_states != CORE_POSITION_TO_INDEX["not_core"]

    def take(self, indices: Sequence[int]) -> UgiAdapterNodeFeatures:
        """Select and reorder atoms while preserving every adapter channel."""

        selected = np.asarray(indices, dtype=np.int64)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= self.node_count):
            raise FeasibilityError("adapter feature selection lies outside the full graph")
        if len(set(selected.tolist())) != selected.size:
            raise FeasibilityError("adapter feature selection contains duplicate atoms")
        return UgiAdapterNodeFeatures(
            origin_states=self.origin_states[selected],
            core_position_states=self.core_position_states[selected],
            port_states=self.port_states[selected],
            distance_to_core=self.distance_to_core[selected],
            distance_to_own_port=self.distance_to_own_port[selected],
            distances_to_all_ports=self.distances_to_all_ports[selected],
        )


@dataclass(frozen=True)
class UgiL1SparseTrainingRecord:
    """Whole-product sparse graph plus exact Ugi adapter node semantics."""

    graph: V5SparseGraphRecord
    canonical_atom_order: np.ndarray
    adapter: UgiAdapterNodeFeatures


@dataclass(frozen=True)
class UgiL1SupportTrainingRecord:
    """Core-protected morphology support plus exact full-graph expansion target."""

    support_graph: V5SparseGraphRecord
    support_full_atom_order: np.ndarray
    support_adapter: UgiAdapterNodeFeatures
    full_adapter: UgiAdapterNodeFeatures
    skeleton: LipidSupportSkeleton


def _boolean(value: object, label: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise FeasibilityError(f"{label} must be a Boolean field")


def _json_object(value: object, label: str) -> dict[str, object]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise FeasibilityError(f"{label} is invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise FeasibilityError(f"{label} must contain a JSON object")
    return parsed


def ugi_adapter_features(
    product_row: Mapping[str, object],
    atom_rows: Sequence[Mapping[str, object]],
    *,
    atom_order: Sequence[int] | None = None,
) -> UgiAdapterNodeFeatures:
    """Validate and tensorize exact Ugi semantics without target inference."""

    product_id = str(product_row["product_id"])
    molecule = Chem.MolFromSmiles(str(product_row["product_smiles"]))
    if molecule is None:
        raise FeasibilityError(f"invalid Ugi product: {product_id}")
    node_count = molecule.GetNumAtoms()
    by_index = {int(row["product_atom_index"]): row for row in atom_rows}
    if len(by_index) != len(atom_rows) or sorted(by_index) != list(range(node_count)):
        raise FeasibilityError(f"Ugi semantic atoms do not cover {product_id} exactly")
    role_anchors_raw = _json_object(
        product_row["role_anchor_indices_json"],
        f"{product_id} role anchors",
    )
    if set(role_anchors_raw) != set(ROLE_NAMES):
        raise FeasibilityError(f"{product_id} role anchors do not match Ugi roles")
    role_anchors = {role: int(role_anchors_raw[role]) for role in ROLE_NAMES}
    if len(set(role_anchors.values())) != len(ROLE_NAMES):
        raise FeasibilityError(f"{product_id} role anchors must be distinct")

    origins = np.zeros(node_count, dtype=np.int64)
    core_positions = np.zeros(node_count, dtype=np.int64)
    ports = np.zeros(node_count, dtype=np.int64)
    core_distances = np.zeros(node_count, dtype=np.int64)
    own_distances = np.full(node_count, NOT_APPLICABLE_DISTANCE, dtype=np.int64)
    all_port_distances = np.zeros((node_count, len(ROLE_NAMES)), dtype=np.int64)

    for index in range(node_count):
        row = by_index[index]
        origin = str(row["origin_role"])
        if origin not in ORIGIN_TO_INDEX or origin == "adapter_unspecified":
            raise FeasibilityError(f"{product_id} has unsupported origin {origin!r}")
        origins[index] = ORIGIN_TO_INDEX[origin]
        is_core = _boolean(row["is_ugi_core"], f"{product_id} core membership")
        core_position = str(row["core_position"])
        if is_core != bool(core_position):
            raise FeasibilityError(f"{product_id} core flag and position disagree")
        core_position_state = core_position if is_core else "not_core"
        if core_position_state not in CORE_POSITION_TO_INDEX:
            raise FeasibilityError(f"{product_id} has unsupported core position {core_position!r}")
        core_positions[index] = CORE_POSITION_TO_INDEX[core_position_state]
        core_distances[index] = int(row["distance_to_nearest_core"])
        distances = _json_object(
            row["distances_to_role_anchors_json"],
            f"{product_id} atom {index} role-anchor distances",
        )
        if set(distances) != set(ROLE_NAMES):
            raise FeasibilityError(f"{product_id} atom {index} lacks role-anchor distances")
        for role_index, role in enumerate(ROLE_NAMES):
            all_port_distances[index, role_index] = int(distances[role])
        if origin in ROLE_NAMES:
            own_distance = int(row["distance_to_origin_anchor"])
            if own_distance != int(distances[origin]):
                raise FeasibilityError(f"{product_id} own-anchor distance disagrees")
            own_distances[index] = own_distance
        elif str(row["distance_to_origin_anchor"]):
            raise FeasibilityError(
                f"{product_id} assembly-introduced atom has an origin-anchor distance"
            )

    for role, anchor in role_anchors.items():
        if not 0 <= anchor < node_count:
            raise FeasibilityError(f"{product_id} role anchor lies outside the graph")
        if origins[anchor] != ORIGIN_TO_INDEX[role] or not bool(core_positions[anchor]):
            raise FeasibilityError(f"{product_id} role anchor has wrong origin or core state")
        ports[anchor] = PORT_TO_INDEX[role]
        if all_port_distances[anchor, ROLE_NAMES.index(role)] != 0:
            raise FeasibilityError(f"{product_id} role anchor is not at zero self-distance")

    if atom_order is None:
        order = np.arange(node_count, dtype=np.int64)
    else:
        order = np.asarray(atom_order, dtype=np.int64)
        if order.shape != (node_count,) or sorted(order.tolist()) != list(range(node_count)):
            raise FeasibilityError(f"{product_id} atom order is not a permutation")
    return UgiAdapterNodeFeatures(
        origin_states=origins[order],
        core_position_states=core_positions[order],
        port_states=ports[order],
        distance_to_core=core_distances[order],
        distance_to_own_port=own_distances[order],
        distances_to_all_ports=all_port_distances[order],
    )


def tensorize_ugi_l1_sparse_record(
    product_row: Mapping[str, object],
    atom_rows: Sequence[Mapping[str, object]],
    atom_to_index: Mapping[AtomState, int],
    *,
    preserve_aromaticity: bool = False,
    tree_traversal: str = "breadth_first_tree_preorder",
) -> UgiL1SparseTrainingRecord:
    """Encode one Ugi product with a core-rooted serialization and exact semantics."""

    product_id = str(product_row["product_id"])
    product_smiles = str(product_row["product_smiles"])
    molecule = Chem.MolFromSmiles(product_smiles)
    if molecule is None:
        raise FeasibilityError(f"invalid Ugi product: {product_id}")
    molecule = canonical_constitutional_molecule(molecule, product_id)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    if canonical != product_smiles:
        raise FeasibilityError(f"{product_id} is not in canonical constitutional atom order")
    canonical_features = ugi_adapter_features(product_row, atom_rows)
    core_candidates = np.flatnonzero(canonical_features.core_membership).tolist()
    atom_order, _ = adapter_canonical_tree(
        molecule,
        core_candidates=core_candidates,
        tree_traversal=tree_traversal,
    )
    graph = _tensorize_adapter_sparse_molecule(
        molecule,
        atom_to_index,
        structure_id=product_id,
        canonical_smiles=product_smiles,
        preserve_aromaticity=preserve_aromaticity,
        core_candidates=core_candidates,
        tree_traversal=tree_traversal,
    )
    features = ugi_adapter_features(product_row, atom_rows, atom_order=atom_order)
    if graph.node_count != features.node_count or not features.core_membership[0]:
        raise FeasibilityError(f"{product_id} core-rooted serialization lost alignment")
    return UgiL1SparseTrainingRecord(
        graph=graph,
        canonical_atom_order=np.asarray(atom_order, dtype=np.int64),
        adapter=features,
    )


def _support_molecule(
    skeleton: LipidSupportSkeleton,
) -> tuple[Chem.Mol, np.ndarray]:
    """Return canonical support graph and its atom indices in the full product."""

    retained = tuple(skeleton.retained_atoms)
    full_to_induced = {full: index for index, full in enumerate(retained)}
    editable = Chem.RWMol()
    for full_index in retained:
        state = skeleton.atom_states[full_index]
        atom = Chem.Atom(state.symbol)
        atom.SetFormalCharge(state.formal_charge)
        atom.SetIsAromatic(state.aromatic)
        if state.explicit_hydrogens:
            atom.SetNumExplicitHs(state.explicit_hydrogens)
            atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for left, right, bond in skeleton.skeleton_bonds:
        editable.AddBond(
            full_to_induced[left],
            full_to_induced[right],
            _INDEX_TO_BOND_TYPE[bond],
        )
    induced = editable.GetMol()
    Chem.SanitizeMol(induced)
    smiles = Chem.MolToSmiles(induced, canonical=True, isomericSmiles=False)
    if not induced.HasProp("_smilesAtomOutputOrder"):
        raise FeasibilityError("RDKit did not expose support-skeleton canonical order")
    try:
        output_order = [
            int(index) for index in json.loads(induced.GetProp("_smilesAtomOutputOrder"))
        ]
    except json.JSONDecodeError as exc:
        raise FeasibilityError("support-skeleton canonical order is invalid") from exc
    if sorted(output_order) != list(range(len(retained))):
        raise FeasibilityError("support-skeleton canonical order is not a permutation")
    canonical = Chem.MolFromSmiles(smiles)
    if canonical is None or canonical.GetNumAtoms() != len(retained):
        raise FeasibilityError("support-skeleton canonicalization changed atom count")
    canonical_full_indices = np.asarray(
        [retained[induced_index] for induced_index in output_order],
        dtype=np.int64,
    )
    return canonical, canonical_full_indices


def tensorize_ugi_l1_support_record(
    product_row: Mapping[str, object],
    atom_rows: Sequence[Mapping[str, object]],
    atom_to_index: Mapping[AtomState, int],
    *,
    preserve_aromaticity: bool = False,
    tree_traversal: str = "breadth_first_tree_preorder",
) -> UgiL1SupportTrainingRecord:
    """Build the morphology-first Ugi record without losing full chemistry."""

    product_id = str(product_row["product_id"])
    molecule = Chem.MolFromSmiles(str(product_row["product_smiles"]))
    if molecule is None:
        raise FeasibilityError(f"invalid Ugi product: {product_id}")
    full_features = ugi_adapter_features(product_row, atom_rows)
    core_indices = np.flatnonzero(full_features.core_membership).tolist()
    skeleton = encode_lipid_support_skeleton(
        molecule,
        structure_id=product_id,
        variant=FUNCTIONAL_SUPPORT,
        protected_atoms=core_indices,
    )
    if not skeleton_roundtrip_exact(skeleton):
        raise FeasibilityError(f"{product_id} support plus expansion does not roundtrip")
    support_molecule, canonical_full_indices = _support_molecule(skeleton)
    support_canonical = Chem.MolToSmiles(
        support_molecule,
        canonical=True,
        isomericSmiles=False,
    )
    canonical_core_candidates = [
        index
        for index, full_index in enumerate(canonical_full_indices.tolist())
        if full_features.core_membership[full_index]
    ]
    tree_order, _ = adapter_canonical_tree(
        support_molecule,
        core_candidates=canonical_core_candidates,
        tree_traversal=tree_traversal,
    )
    graph = _tensorize_adapter_sparse_molecule(
        support_molecule,
        atom_to_index,
        structure_id=f"{product_id}/support",
        canonical_smiles=support_canonical,
        preserve_aromaticity=preserve_aromaticity,
        core_candidates=canonical_core_candidates,
        tree_traversal=tree_traversal,
    )
    serialized_full_indices = canonical_full_indices[np.asarray(tree_order, dtype=np.int64)]
    support_features = full_features.take(serialized_full_indices)
    if graph.node_count != support_features.node_count or not support_features.core_membership[0]:
        raise FeasibilityError(f"{product_id} support semantics lost tree alignment")
    return UgiL1SupportTrainingRecord(
        support_graph=graph,
        support_full_atom_order=serialized_full_indices,
        support_adapter=support_features,
        full_adapter=full_features,
        skeleton=skeleton,
    )
