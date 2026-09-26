"""Non-leaking interface between Ugi morphology and chemistry realization.

The morphology stage generates a sparse support topology around the fixed Ugi
reaction core.  Chemistry realization may condition on that generated topology
and on adapter-owned synthesis semantics, but it must not receive the correct
non-core atom states, bond orders, aromatic states, or graph distances copied
from the target molecule.

This module makes that boundary structural.  ``ChemistryTopologyCondition``
contains only information available after topology generation.  Graph
distances are recomputed from that condition.  ``ChemistryRealizationTarget``
contains the withheld atom, bond, and terminal-decoration labels used by the
training loss.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.ugi_adapter_features import (
    CORE_POSITION_STATES,
    CORE_POSITION_TO_INDEX,
    NOT_APPLICABLE_DISTANCE,
    ORIGIN_TO_INDEX,
    PORT_TO_INDEX,
    UgiAdapterNodeFeatures,
    UgiL1SupportTrainingRecord,
)
from forge.model.ugi_morphology_program import preorder_attached_forest_to_parents
from forge.model.v5_sparse_representation import v5_offspring_to_parents
from forge.potency.annotations import ROLE_NAMES

WITHHELD_STATE = -1
ROOT_BOND_TARGET = -1
ADAPTER_ATTACHMENT_BOND_STATE = 0


@dataclass(frozen=True)
class UgiFixedCoreSchema:
    """Adapter-owned atom and bond chemistry for the five-atom Ugi core.

    Atom states are indexed by ``CORE_POSITION_STATES``; ``not_core`` remains
    withheld.  Bonds are keyed by sorted nonzero core-position state pairs.
    The schema is inferred once from qualified adapter records and must be
    invariant across the complete corpus before production use.
    """

    atom_state_by_core_position: tuple[int, ...]
    bond_state_by_core_positions: tuple[tuple[int, int, int], ...]

    @property
    def bond_lookup(self) -> dict[tuple[int, int], int]:
        return {(left, right): state for left, right, state in self.bond_state_by_core_positions}


@dataclass(frozen=True)
class ChemistryTopologyCondition:
    """Generated sparse topology and adapter semantics visible to chemistry."""

    structure_id: str
    tree_traversal: str
    offspring: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    origin_states: np.ndarray
    core_position_states: np.ndarray
    port_states: np.ndarray
    fixed_atom_mask: np.ndarray
    fixed_atom_states: np.ndarray
    fixed_parent_bond_mask: np.ndarray
    fixed_parent_bond_states: np.ndarray
    fixed_closure_bond_mask: np.ndarray
    fixed_closure_bond_states: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.offspring.size)

    @property
    def closure_count(self) -> int:
        return int(self.closure_left.size)

    @property
    def parents(self) -> np.ndarray:
        return v5_offspring_to_parents(self.offspring, self.tree_traversal)


@dataclass(frozen=True)
class DecorationTargets:
    """Canonical terminal atom decorations omitted from morphology support."""

    anchor_indices: np.ndarray
    atom_states: np.ndarray
    bond_states: np.ndarray

    @property
    def count(self) -> int:
        return int(self.anchor_indices.size)


@dataclass(frozen=True)
class ChemistryRealizationTarget:
    """Withheld chemistry labels on one verified generated support topology."""

    structure_id: str
    atom_states: np.ndarray
    parent_bond_states: np.ndarray
    closure_bond_states: np.ndarray
    decorations: DecorationTargets

    @property
    def node_count(self) -> int:
        return int(self.atom_states.size)


def _graph_edges(
    offspring: np.ndarray,
    tree_traversal: str,
    closure_left: np.ndarray,
    closure_right: np.ndarray,
) -> tuple[np.ndarray, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Validate sparse topology and return parents, tree edges, and closures."""

    if (
        offspring.ndim != 1
        or offspring.size < 1
        or not np.issubdtype(offspring.dtype, np.integer)
        or np.any(offspring < 0)
        or closure_left.shape != closure_right.shape
        or closure_left.ndim != 1
        or not np.issubdtype(closure_left.dtype, np.integer)
        or not np.issubdtype(closure_right.dtype, np.integer)
    ):
        raise FeasibilityError("chemistry condition contains an invalid sparse topology")
    parents = v5_offspring_to_parents(offspring, tree_traversal)
    tree_edges = tuple((int(parents[child]), child) for child in range(1, offspring.size))
    occupied = {tuple(sorted(edge)) for edge in tree_edges}
    closures: list[tuple[int, int]] = []
    for left_raw, right_raw in zip(closure_left, closure_right, strict=True):
        left, right = sorted((int(left_raw), int(right_raw)))
        pair = (left, right)
        if not 0 <= left < right < offspring.size or pair in occupied or pair in closures:
            raise FeasibilityError("chemistry condition contains an invalid closure edge")
        closures.append(pair)
    if closures != sorted(closures):
        raise FeasibilityError("chemistry closure edges are not in canonical order")
    return parents, tree_edges, tuple(closures)


def core_schema_from_record(record: UgiL1SupportTrainingRecord) -> UgiFixedCoreSchema:
    """Extract one candidate schema from a qualified Ugi support record."""

    graph = record.support_graph
    adapter = record.support_adapter
    if graph.node_count != adapter.node_count:
        raise FeasibilityError("Ugi support graph and adapter features are misaligned")
    _, tree_edges, closures = _graph_edges(
        graph.offspring,
        graph.tree_traversal,
        graph.closure_left,
        graph.closure_right,
    )
    core_state_by_position = [WITHHELD_STATE] * len(CORE_POSITION_STATES)
    for node in np.flatnonzero(adapter.core_membership):
        position = int(adapter.core_position_states[node])
        if position == CORE_POSITION_TO_INDEX["not_core"]:
            raise FeasibilityError("core member has the not-core position state")
        if core_state_by_position[position] != WITHHELD_STATE:
            raise FeasibilityError("Ugi core position is not unique")
        core_state_by_position[position] = int(graph.node_states[node])
    if core_state_by_position[0] != WITHHELD_STATE or any(
        state == WITHHELD_STATE for state in core_state_by_position[1:]
    ):
        raise FeasibilityError("Ugi core does not cover every adapter position exactly")

    bond_rows = []
    all_edges = (*tree_edges, *closures)
    all_bonds = np.concatenate((graph.parent_bonds[1:], graph.closure_bonds))
    if len(all_edges) != all_bonds.size:
        raise FeasibilityError("Ugi support edge chemistry is misaligned")
    for (left, right), bond_state in zip(all_edges, all_bonds, strict=True):
        left_position = int(adapter.core_position_states[left])
        right_position = int(adapter.core_position_states[right])
        if left_position == 0 or right_position == 0:
            continue
        position_pair = tuple(sorted((left_position, right_position)))
        bond_rows.append((*position_pair, int(bond_state)))
    bond_rows.sort()
    if len(bond_rows) != len({row[:2] for row in bond_rows}):
        raise FeasibilityError("Ugi core contains duplicate position-pair bonds")
    return UgiFixedCoreSchema(
        atom_state_by_core_position=tuple(core_state_by_position),
        bond_state_by_core_positions=tuple(bond_rows),
    )


def validate_record_against_core_schema(
    record: UgiL1SupportTrainingRecord,
    schema: UgiFixedCoreSchema,
) -> None:
    """Require adapter-owned core chemistry to be corpus invariant."""

    observed = core_schema_from_record(record)
    if observed != schema:
        raise FeasibilityError("Ugi fixed-core chemistry changed across records")


def project_chemistry_topology_condition(
    record: UgiL1SupportTrainingRecord,
    schema: UgiFixedCoreSchema,
) -> ChemistryTopologyCondition:
    """Project a rich source record without exposing non-core chemistry."""

    validate_record_against_core_schema(record, schema)
    graph = record.support_graph
    adapter = record.support_adapter
    parents, _, _ = _graph_edges(
        graph.offspring,
        graph.tree_traversal,
        graph.closure_left,
        graph.closure_right,
    )
    core_mask = adapter.core_membership.astype(bool, copy=True)
    fixed_atoms = np.full(graph.node_count, WITHHELD_STATE, dtype=np.int64)
    for node in np.flatnonzero(core_mask):
        position = int(adapter.core_position_states[node])
        fixed_atoms[node] = schema.atom_state_by_core_position[position]

    parent_mask = np.zeros(graph.node_count, dtype=bool)
    fixed_parent_bonds = np.full(graph.node_count, WITHHELD_STATE, dtype=np.int64)
    bond_lookup = schema.bond_lookup
    for child in range(1, graph.node_count):
        parent = int(parents[child])
        if core_mask[parent] and core_mask[child]:
            positions = tuple(
                sorted(
                    (
                        int(adapter.core_position_states[parent]),
                        int(adapter.core_position_states[child]),
                    )
                )
            )
            if positions not in bond_lookup:
                raise FeasibilityError("fixed Ugi tree edge is absent from core schema")
            parent_mask[child] = True
            fixed_parent_bonds[child] = bond_lookup[positions]
        elif (
            core_mask[parent] != core_mask[child]
            and adapter.origin_states[parent] == adapter.origin_states[child]
        ):
            # The precursor-to-core attachment is part of the qualified Ugi
            # assembly program.  Across the complete frozen product corpus it
            # is always a single bond; treating it as a free chemistry token
            # can produce products that cannot be decomposed back to a valid
            # amine, aldehyde or isocyanide precursor.
            parent_mask[child] = True
            fixed_parent_bonds[child] = ADAPTER_ATTACHMENT_BOND_STATE

    closure_mask = np.zeros(graph.closure_count, dtype=bool)
    fixed_closure_bonds = np.full(graph.closure_count, WITHHELD_STATE, dtype=np.int64)
    for index, (left_raw, right_raw) in enumerate(
        zip(graph.closure_left, graph.closure_right, strict=True)
    ):
        left, right = int(left_raw), int(right_raw)
        if core_mask[left] and core_mask[right]:
            positions = tuple(
                sorted(
                    (
                        int(adapter.core_position_states[left]),
                        int(adapter.core_position_states[right]),
                    )
                )
            )
            if positions not in bond_lookup:
                raise FeasibilityError("fixed Ugi closure edge is absent from core schema")
            closure_mask[index] = True
            fixed_closure_bonds[index] = bond_lookup[positions]

    condition = ChemistryTopologyCondition(
        structure_id=graph.structure_id,
        tree_traversal=graph.tree_traversal,
        offspring=graph.offspring.astype(np.int64, copy=True),
        closure_left=graph.closure_left.astype(np.int64, copy=True),
        closure_right=graph.closure_right.astype(np.int64, copy=True),
        origin_states=adapter.origin_states.astype(np.int64, copy=True),
        core_position_states=adapter.core_position_states.astype(np.int64, copy=True),
        port_states=adapter.port_states.astype(np.int64, copy=True),
        fixed_atom_mask=core_mask,
        fixed_atom_states=fixed_atoms,
        fixed_parent_bond_mask=parent_mask,
        fixed_parent_bond_states=fixed_parent_bonds,
        fixed_closure_bond_mask=closure_mask,
        fixed_closure_bond_states=fixed_closure_bonds,
    )
    validate_chemistry_topology_condition(condition)
    return condition


def assemble_ugi_chemistry_topology_condition(
    *,
    structure_id: str,
    offspring_by_role: Mapping[str, np.ndarray],
    attachment_counts_by_role: Mapping[str, int],
    closure_left_by_role: Mapping[str, np.ndarray],
    closure_right_by_role: Mapping[str, np.ndarray],
    schema: UgiFixedCoreSchema,
) -> ChemistryTopologyCondition:
    """Stitch three generated exteriors around the invariant Ugi core.

    This is the inference path from the morphology flow to chemistry
    realization.  The generated arrays carry no component identities or atom
    chemistry.  The resulting order is the frozen core-rooted hybrid preorder
    used by qualified training records.
    """

    required = set(ROLE_NAMES)
    if (
        set(offspring_by_role) != required
        or set(attachment_counts_by_role) != required
        or set(closure_left_by_role) != required
        or set(closure_right_by_role) != required
    ):
        raise FeasibilityError("generated Ugi topology must contain all three roles")
    local_closures: dict[str, tuple[tuple[int, int], ...]] = {}
    for role in ROLE_NAMES:
        offspring = np.asarray(offspring_by_role[role], dtype=np.int64)
        attachment_count = int(attachment_counts_by_role[role])
        left = np.asarray(closure_left_by_role[role], dtype=np.int64)
        right = np.asarray(closure_right_by_role[role], dtype=np.int64)
        try:
            local_parents = preorder_attached_forest_to_parents(
                offspring,
                attachment_count=attachment_count,
            )
        except RuntimeError as exc:
            raise FeasibilityError(
                f"generated {role} exterior is not a valid attached forest"
            ) from exc
        occupied = {
            tuple(sorted((int(parent), child)))
            for child, parent in enumerate(local_parents.tolist())
            if parent >= 0
        }
        closures = []
        for left_raw, right_raw in zip(left, right, strict=True):
            local_left, local_right = sorted((int(left_raw), int(right_raw)))
            pair = (local_left, local_right)
            if (
                not 0 <= local_left < local_right < offspring.size
                or pair in occupied
                or pair in closures
            ):
                raise FeasibilityError(f"generated {role} exterior contains an invalid closure")
            closures.append(pair)
        if closures != sorted(closures):
            raise FeasibilityError(f"generated {role} closures are not in canonical order")
        local_closures[role] = closures

    amine = "amine_head"
    aldehyde = "oxoester_aldehyde_body_tail"
    isocyanide = "isocyanide_tail"
    core_not_port = PORT_TO_INDEX["not_port"]
    nodes: list[tuple[int, int, int, int]] = [
        (
            ORIGIN_TO_INDEX["assembly_introduced"],
            CORE_POSITION_TO_INDEX["template_introduced_0"],
            core_not_port,
            1,
        ),
        (
            ORIGIN_TO_INDEX[isocyanide],
            CORE_POSITION_TO_INDEX["map_3"],
            core_not_port,
            2,
        ),
        (
            ORIGIN_TO_INDEX[isocyanide],
            CORE_POSITION_TO_INDEX["map_4"],
            PORT_TO_INDEX[isocyanide],
            int(attachment_counts_by_role[isocyanide]),
        ),
    ]
    offsets: dict[str, int] = {isocyanide: len(nodes)}
    nodes.extend(
        (ORIGIN_TO_INDEX[isocyanide], 0, core_not_port, int(children))
        for children in offspring_by_role[isocyanide]
    )
    aldehyde_core_index = len(nodes)
    nodes.append(
        (
            ORIGIN_TO_INDEX[aldehyde],
            CORE_POSITION_TO_INDEX["map_2"],
            PORT_TO_INDEX[aldehyde],
            int(attachment_counts_by_role[aldehyde]) + 1,
        )
    )
    offsets[aldehyde] = len(nodes)
    nodes.extend(
        (ORIGIN_TO_INDEX[aldehyde], 0, core_not_port, int(children))
        for children in offspring_by_role[aldehyde]
    )
    amine_core_index = len(nodes)
    nodes.append(
        (
            ORIGIN_TO_INDEX[amine],
            CORE_POSITION_TO_INDEX["map_1"],
            PORT_TO_INDEX[amine],
            int(attachment_counts_by_role[amine]),
        )
    )
    offsets[amine] = len(nodes)
    nodes.extend(
        (ORIGIN_TO_INDEX[amine], 0, core_not_port, int(children))
        for children in offspring_by_role[amine]
    )

    origin_states = np.asarray([row[0] for row in nodes], dtype=np.int64)
    core_positions = np.asarray([row[1] for row in nodes], dtype=np.int64)
    port_states = np.asarray([row[2] for row in nodes], dtype=np.int64)
    offspring = np.asarray([row[3] for row in nodes], dtype=np.int64)
    parents = v5_offspring_to_parents(offspring, "breadth_first_tree_preorder")
    expected_core_parents = {
        1: 0,
        2: 1,
        aldehyde_core_index: 1,
        amine_core_index: aldehyde_core_index,
    }
    if any(int(parents[child]) != parent for child, parent in expected_core_parents.items()):
        raise FeasibilityError("generated exteriors changed the invariant Ugi core tree")

    closure_pairs = sorted(
        (offsets[role] + left, offsets[role] + right)
        for role in ROLE_NAMES
        for left, right in local_closures[role]
    )
    closure_left = np.asarray([row[0] for row in closure_pairs], dtype=np.int64)
    closure_right = np.asarray([row[1] for row in closure_pairs], dtype=np.int64)
    core_mask = core_positions != CORE_POSITION_TO_INDEX["not_core"]
    fixed_atoms = np.full(len(nodes), WITHHELD_STATE, dtype=np.int64)
    for node in np.flatnonzero(core_mask):
        fixed_atoms[node] = schema.atom_state_by_core_position[int(core_positions[node])]
    if np.any(fixed_atoms[core_mask] < 0):
        raise FeasibilityError("generated Ugi core lacks fixed atom chemistry")

    bond_lookup = schema.bond_lookup
    parent_mask = np.zeros(len(nodes), dtype=bool)
    fixed_parent_bonds = np.full(len(nodes), WITHHELD_STATE, dtype=np.int64)
    for child in range(1, len(nodes)):
        parent = int(parents[child])
        if core_mask[parent] and core_mask[child]:
            positions = tuple(sorted((int(core_positions[parent]), int(core_positions[child]))))
            if positions not in bond_lookup:
                raise FeasibilityError("generated Ugi core edge is absent from schema")
            parent_mask[child] = True
            fixed_parent_bonds[child] = bond_lookup[positions]
        elif (
            core_mask[parent] != core_mask[child] and origin_states[parent] == origin_states[child]
        ):
            parent_mask[child] = True
            fixed_parent_bonds[child] = ADAPTER_ATTACHMENT_BOND_STATE
    condition = ChemistryTopologyCondition(
        structure_id=structure_id,
        tree_traversal="breadth_first_tree_preorder",
        offspring=offspring,
        closure_left=closure_left,
        closure_right=closure_right,
        origin_states=origin_states,
        core_position_states=core_positions,
        port_states=port_states,
        fixed_atom_mask=core_mask,
        fixed_atom_states=fixed_atoms,
        fixed_parent_bond_mask=parent_mask,
        fixed_parent_bond_states=fixed_parent_bonds,
        fixed_closure_bond_mask=np.zeros(len(closure_pairs), dtype=bool),
        fixed_closure_bond_states=np.full(
            len(closure_pairs),
            WITHHELD_STATE,
            dtype=np.int64,
        ),
    )
    validate_chemistry_topology_condition(condition)
    return condition


def validate_chemistry_topology_condition(condition: ChemistryTopologyCondition) -> None:
    """Reject topology conditions that leak chemistry or violate adapter semantics."""

    parents, tree_edges, closure_edges = _graph_edges(
        condition.offspring,
        condition.tree_traversal,
        condition.closure_left,
        condition.closure_right,
    )
    node_shape = (condition.node_count,)
    node_arrays = (
        condition.origin_states,
        condition.core_position_states,
        condition.port_states,
        condition.fixed_atom_mask,
        condition.fixed_atom_states,
        condition.fixed_parent_bond_mask,
        condition.fixed_parent_bond_states,
    )
    if any(array.shape != node_shape for array in node_arrays):
        raise FeasibilityError("chemistry node-conditioning arrays are misaligned")
    if (
        condition.fixed_atom_mask.dtype != np.bool_
        or condition.fixed_parent_bond_mask.dtype != np.bool_
    ):
        raise FeasibilityError("chemistry fixed-state masks must be Boolean")
    if (
        condition.fixed_closure_bond_mask.shape != (condition.closure_count,)
        or condition.fixed_closure_bond_states.shape != (condition.closure_count,)
        or condition.fixed_closure_bond_mask.dtype != np.bool_
    ):
        raise FeasibilityError("chemistry closure-conditioning arrays are misaligned")
    for values, classes, label in (
        (condition.origin_states, max(ORIGIN_TO_INDEX.values()) + 1, "origin"),
        (condition.core_position_states, len(CORE_POSITION_STATES), "core position"),
        (condition.port_states, max(PORT_TO_INDEX.values()) + 1, "port"),
    ):
        if (
            not np.issubdtype(values.dtype, np.integer)
            or np.any(values < 0)
            or np.any(values >= classes)
        ):
            raise FeasibilityError(f"chemistry {label} states lie outside support")
    core_mask = condition.core_position_states != CORE_POSITION_TO_INDEX["not_core"]
    if not np.array_equal(condition.fixed_atom_mask, core_mask) or int(core_mask.sum()) != 5:
        raise FeasibilityError("only the exact five-atom Ugi core may be fixed")
    if np.any(condition.fixed_atom_states[~core_mask] != WITHHELD_STATE):
        raise FeasibilityError("non-core atom chemistry leaked into conditioning")
    if np.any(condition.fixed_atom_states[core_mask] < 0):
        raise FeasibilityError("fixed Ugi core atom chemistry is incomplete")
    if (
        condition.fixed_parent_bond_mask[0]
        or condition.fixed_parent_bond_states[0] != WITHHELD_STATE
    ):
        raise FeasibilityError("the root sentinel cannot carry fixed bond chemistry")
    for child in range(1, condition.node_count):
        parent = int(parents[child])
        core_internal = bool(core_mask[parent] and core_mask[child])
        adapter_attachment = bool(
            core_mask[parent] != core_mask[child]
            and condition.origin_states[parent] == condition.origin_states[child]
        )
        expected = core_internal or adapter_attachment
        if bool(condition.fixed_parent_bond_mask[child]) != expected:
            raise FeasibilityError(
                "only core-internal or adapter-attachment tree bonds may be fixed"
            )
        if (
            adapter_attachment
            and condition.fixed_parent_bond_states[child] != ADAPTER_ATTACHMENT_BOND_STATE
        ):
            raise FeasibilityError("Ugi adapter attachment must be a fixed single bond")
    for index, (left, right) in enumerate(
        zip(condition.closure_left, condition.closure_right, strict=True)
    ):
        expected = bool(core_mask[int(left)] and core_mask[int(right)])
        if bool(condition.fixed_closure_bond_mask[index]) != expected:
            raise FeasibilityError("only core-internal closure bonds may be fixed")
    if np.any(
        condition.fixed_parent_bond_states[~condition.fixed_parent_bond_mask] != WITHHELD_STATE
    ) or np.any(
        condition.fixed_closure_bond_states[~condition.fixed_closure_bond_mask] != WITHHELD_STATE
    ):
        raise FeasibilityError("non-core bond chemistry leaked into conditioning")
    for role in ROLE_NAMES:
        role_port = PORT_TO_INDEX[role]
        ports = np.flatnonzero(condition.port_states == role_port)
        if ports.size != 1:
            raise FeasibilityError(f"chemistry condition lacks one {role} port")
        port = int(ports[0])
        if not core_mask[port] or condition.origin_states[port] != ORIGIN_TO_INDEX[role]:
            raise FeasibilityError(f"chemistry {role} port has invalid core semantics")

    all_edges = (*tree_edges, *closure_edges)
    adjacency = [set() for _ in range(condition.node_count)]
    for left, right in all_edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
        if condition.origin_states[left] != condition.origin_states[right] and not (
            core_mask[left] and core_mask[right]
        ):
            raise FeasibilityError("cross-origin chemistry edge lies outside the Ugi core")
    for role in ROLE_NAMES:
        origin = ORIGIN_TO_INDEX[role]
        selected = set(np.flatnonzero(condition.origin_states == origin).tolist())
        if not selected:
            raise FeasibilityError(f"chemistry condition has an empty {role} origin")
        start = min(selected)
        reached = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node] & selected:
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if reached != selected:
            raise FeasibilityError(f"chemistry {role} origin is disconnected")
        boundaries = [
            (left, right)
            for left, right in all_edges
            if condition.origin_states[left] == origin
            and condition.origin_states[right] == origin
            and bool(core_mask[left]) != bool(core_mask[right])
        ]
        if not boundaries:
            raise FeasibilityError(f"chemistry {role} origin lacks a core boundary")
        core_endpoints = {left if core_mask[left] else right for left, right in boundaries}
        if len(core_endpoints) != 1:
            raise FeasibilityError(f"chemistry {role} exterior uses multiple core ports")
        core_endpoint = next(iter(core_endpoints))
        if condition.port_states[core_endpoint] != PORT_TO_INDEX[role]:
            raise FeasibilityError(f"chemistry {role} exterior is attached at the wrong port")


def _distances(adjacency: tuple[tuple[int, ...], ...], sources: np.ndarray) -> np.ndarray:
    if sources.size < 1:
        raise FeasibilityError("graph-distance calculation requires a source")
    result = np.full(len(adjacency), NOT_APPLICABLE_DISTANCE, dtype=np.int64)
    queue: deque[int] = deque()
    for source_raw in sources:
        source = int(source_raw)
        result[source] = 0
        queue.append(source)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if result[neighbor] == NOT_APPLICABLE_DISTANCE:
                result[neighbor] = result[node] + 1
                queue.append(neighbor)
    if np.any(result < 0):
        raise FeasibilityError("generated chemistry topology is disconnected")
    return result


def recompute_adapter_distances(
    condition: ChemistryTopologyCondition,
) -> UgiAdapterNodeFeatures:
    """Derive positional channels from the generated graph, never the target."""

    validate_chemistry_topology_condition(condition)
    parents, _, closures = _graph_edges(
        condition.offspring,
        condition.tree_traversal,
        condition.closure_left,
        condition.closure_right,
    )
    neighbors = [set() for _ in range(condition.node_count)]
    for child in range(1, condition.node_count):
        parent = int(parents[child])
        neighbors[parent].add(child)
        neighbors[child].add(parent)
    for left, right in closures:
        neighbors[left].add(right)
        neighbors[right].add(left)
    adjacency = tuple(tuple(sorted(values)) for values in neighbors)
    core_mask = condition.core_position_states != CORE_POSITION_TO_INDEX["not_core"]
    core_distances = _distances(adjacency, np.flatnonzero(core_mask))
    all_port_distances = np.zeros((condition.node_count, len(ROLE_NAMES)), dtype=np.int64)
    own_distances = np.full(
        condition.node_count,
        NOT_APPLICABLE_DISTANCE,
        dtype=np.int64,
    )
    for role_index, role in enumerate(ROLE_NAMES):
        port = np.flatnonzero(condition.port_states == PORT_TO_INDEX[role])
        distances = _distances(adjacency, port)
        all_port_distances[:, role_index] = distances
        role_mask = condition.origin_states == ORIGIN_TO_INDEX[role]
        own_distances[role_mask] = distances[role_mask]
    return UgiAdapterNodeFeatures(
        origin_states=condition.origin_states.copy(),
        core_position_states=condition.core_position_states.copy(),
        port_states=condition.port_states.copy(),
        distance_to_core=core_distances,
        distance_to_own_port=own_distances,
        distances_to_all_ports=all_port_distances,
    )


def materialize_chemistry_target(
    record: UgiL1SupportTrainingRecord,
    atom_to_index: Mapping[AtomState, int],
) -> ChemistryRealizationTarget:
    """Materialize atom/bond/decorations labels withheld from conditioning."""

    graph = record.support_graph
    if graph.node_count != record.support_full_atom_order.size:
        raise FeasibilityError("support atom order and chemistry target are misaligned")
    full_to_support = {
        int(full_index): support_index
        for support_index, full_index in enumerate(record.support_full_atom_order.tolist())
    }
    removed = set(record.skeleton.removed_atoms)
    decoration_rows: list[tuple[int, int, int]] = []
    incident: dict[int, list[tuple[int, int]]] = {index: [] for index in removed}
    for left, right, bond_state in record.skeleton.expansion_bonds:
        if left in removed:
            incident[left].append((right, bond_state))
        if right in removed:
            incident[right].append((left, bond_state))
    for removed_index in sorted(removed):
        edges = incident[removed_index]
        if len(edges) != 1 or edges[0][0] not in full_to_support:
            raise FeasibilityError("functional-support decoration is not one terminal atom")
        anchor_full, bond_state = edges[0]
        atom_state = record.skeleton.atom_states[removed_index]
        if atom_state not in atom_to_index:
            raise FeasibilityError("decoration atom state is absent from vocabulary")
        decoration_rows.append(
            (full_to_support[anchor_full], int(atom_to_index[atom_state]), int(bond_state))
        )
    decoration_rows.sort()
    parent_bonds = graph.parent_bonds.astype(np.int64, copy=True)
    parent_bonds[0] = ROOT_BOND_TARGET
    return ChemistryRealizationTarget(
        structure_id=graph.structure_id,
        atom_states=graph.node_states.astype(np.int64, copy=True),
        parent_bond_states=parent_bonds,
        closure_bond_states=graph.closure_bonds.astype(np.int64, copy=True),
        decorations=DecorationTargets(
            anchor_indices=np.asarray([row[0] for row in decoration_rows], dtype=np.int64),
            atom_states=np.asarray([row[1] for row in decoration_rows], dtype=np.int64),
            bond_states=np.asarray([row[2] for row in decoration_rows], dtype=np.int64),
        ),
    )
