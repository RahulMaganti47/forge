"""Training-derived local chemistry support for vocabulary-free graph generation.

The policy in this module is deliberately local.  It records atom--bond neighborhoods and
fundamental ring-closure compositions observed in the training fold, conditioned on the reaction
program and precursor-origin role.  It never records a component identity, component fingerprint,
or complete molecular fragment.

Two views are retained:

* the role-conditioned view is used by FORGE's terminal constrained decoder; and
* the program-only view assesses complete molecular graphs from any generator without requiring
  access to method-private atom-origin labels.

Passing this policy is evidence of training-supported local chemistry, not synthesis success,
stability, safety, activity, or route closure.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.model.defog_feasibility import AtomState
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord

LEGACY_POLICY_SCHEMA = "forge.local_chemistry_support.v1"
POLICY_SCHEMA = "forge.local_chemistry_support.v2"
ASSESSMENT_SCHEMA = "forge.local_chemistry_assessment.v1"


class LocalChemistrySupportError(ValueError):
    """A local-support policy or assessed graph violates its declared contract."""


ProgramEdge = tuple[str, int, str]
RoleEdge = tuple[str, str, int, str, str]
ProgramTriangle = tuple[str, str, str]
RoleTriangle = tuple[tuple[str, str], tuple[str, str], tuple[str, str]]
ProgramCycle = tuple[str, ...]
RoleCycle = tuple[tuple[str, str], ...]


def _program_edge(left_symbol: str, bond: int, right_symbol: str) -> ProgramEdge:
    left, right = sorted((left_symbol, right_symbol))
    return left, int(bond), right


def _role_edge(
    left_role: str,
    left_symbol: str,
    bond: int,
    right_role: str,
    right_symbol: str,
) -> RoleEdge:
    left, right = sorted(((left_role, left_symbol), (right_role, right_symbol)))
    return left[0], left[1], int(bond), right[0], right[1]


def _program_triangle(symbols: Iterable[str]) -> ProgramTriangle:
    values = tuple(sorted(symbols))
    if len(values) != 3:
        raise LocalChemistrySupportError("a three-membered-ring signature needs three atoms")
    return values  # type: ignore[return-value]


def _role_triangle(values: Iterable[tuple[str, str]]) -> RoleTriangle:
    signature = tuple(sorted(values))
    if len(signature) != 3:
        raise LocalChemistrySupportError("a role-conditioned ring signature needs three atoms")
    return signature  # type: ignore[return-value]


def _program_cycle(symbols: Iterable[str]) -> ProgramCycle:
    values = tuple(sorted(symbols))
    if len(values) < 3:
        raise LocalChemistrySupportError("a ring signature needs at least three atoms")
    return values


def _role_cycle(values: Iterable[tuple[str, str]]) -> RoleCycle:
    signature = tuple(sorted(values))
    if len(signature) < 3:
        raise LocalChemistrySupportError("a role-conditioned ring needs at least three atoms")
    return signature


def tree_path_indices(
    parents: Sequence[int] | np.ndarray, left: int, right: int
) -> tuple[int, ...]:
    """Return the unique spanning-tree path between two nodes, including both endpoints."""

    parent_values = np.asarray(parents, dtype=np.int64)
    node_count = int(parent_values.size)
    if not 0 <= left < node_count or not 0 <= right < node_count or left == right:
        raise LocalChemistrySupportError("ring-closure endpoints are invalid")
    for child in range(1, node_count):
        parent = int(parent_values[child])
        if parent < 0 or parent >= child:
            raise LocalChemistrySupportError("spanning-tree parents are invalid")
    left_to_root = [left]
    while left_to_root[-1] != 0:
        left_to_root.append(int(parent_values[left_to_root[-1]]))
    right_to_root = [right]
    while right_to_root[-1] != 0:
        right_to_root.append(int(parent_values[right_to_root[-1]]))
    left_positions = {node: index for index, node in enumerate(left_to_root)}
    common = next((node for node in right_to_root if node in left_positions), None)
    if common is None:
        raise LocalChemistrySupportError("spanning tree is disconnected")
    left_segment = left_to_root[: left_positions[common] + 1]
    right_segment = right_to_root[: right_to_root.index(common)]
    return tuple((*left_segment, *reversed(right_segment)))


def fundamental_cycle_indices(
    record: SynthesisProgramGraphRecord,
    *,
    variable_only: bool,
) -> tuple[tuple[int, ...], ...]:
    """Return closure-defined cycles relative to the record's deterministic spanning tree."""

    cycles = []
    for slot, (left, right) in enumerate(
        zip(record.graph.closure_left, record.graph.closure_right, strict=True)
    ):
        if variable_only and bool(record.fixed_closure_bond_mask[slot]):
            continue
        cycles.append(tree_path_indices(record.graph.parents, int(left), int(right)))
    return tuple(cycles)


def _graph_edges(record: SynthesisProgramGraphRecord) -> tuple[tuple[int, int, int], ...]:
    edges = [
        (child, int(record.graph.parents[child]), int(record.graph.parent_bonds[child]))
        for child in range(1, record.node_count)
    ]
    edges.extend(
        (int(left), int(right), int(bond))
        for left, right, bond in zip(
            record.graph.closure_left,
            record.graph.closure_right,
            record.graph.closure_bonds,
            strict=True,
        )
    )
    return tuple(edges)


def _triangle_indices(
    node_count: int, edges: Iterable[tuple[int, int, int]]
) -> tuple[tuple[int, int, int], ...]:
    neighbors = [set() for _ in range(node_count)]
    for left, right, _ in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    triangles: list[tuple[int, int, int]] = []
    for left in range(node_count):
        for middle in sorted(value for value in neighbors[left] if value > left):
            for right in sorted(
                value for value in neighbors[left].intersection(neighbors[middle]) if value > middle
            ):
                triangles.append((left, middle, right))
    return tuple(triangles)


@dataclass(frozen=True)
class ComponentSupportBounds:
    """Observed hard support and nonselecting diagnostic quantiles for one semantic role."""

    records: int
    heavy_atoms_min: int
    heavy_atoms_max: int
    carbon_atoms_min: int
    carbon_atoms_max: int
    heteroatoms_min: int
    heteroatoms_max: int
    carbon_atoms_q01: float
    carbon_atoms_q99: float
    heteroatoms_q01: float
    heteroatoms_q99: float

    @classmethod
    def from_rows(cls, rows: Sequence[tuple[int, int, int]]) -> ComponentSupportBounds:
        if not rows:
            raise LocalChemistrySupportError("component support cannot be fit from zero rows")
        values = np.asarray(rows, dtype=np.int64)
        if values.ndim != 2 or values.shape[1] != 3 or np.any(values < 0):
            raise LocalChemistrySupportError("component support rows are malformed")
        heavy, carbon, hetero = values.T
        if np.any(carbon + hetero != heavy):
            raise LocalChemistrySupportError("component heavy-atom accounting changed")
        return cls(
            records=len(rows),
            heavy_atoms_min=int(heavy.min()),
            heavy_atoms_max=int(heavy.max()),
            carbon_atoms_min=int(carbon.min()),
            carbon_atoms_max=int(carbon.max()),
            heteroatoms_min=int(hetero.min()),
            heteroatoms_max=int(hetero.max()),
            carbon_atoms_q01=float(np.quantile(carbon, 0.01)),
            carbon_atoms_q99=float(np.quantile(carbon, 0.99)),
            heteroatoms_q01=float(np.quantile(hetero, 0.01)),
            heteroatoms_q99=float(np.quantile(hetero, 0.99)),
        )

    def to_mapping(self) -> dict[str, int | float]:
        return {
            "records": self.records,
            "heavy_atoms_min": self.heavy_atoms_min,
            "heavy_atoms_max": self.heavy_atoms_max,
            "carbon_atoms_min": self.carbon_atoms_min,
            "carbon_atoms_max": self.carbon_atoms_max,
            "heteroatoms_min": self.heteroatoms_min,
            "heteroatoms_max": self.heteroatoms_max,
            "carbon_atoms_q01": self.carbon_atoms_q01,
            "carbon_atoms_q99": self.carbon_atoms_q99,
            "heteroatoms_q01": self.heteroatoms_q01,
            "heteroatoms_q99": self.heteroatoms_q99,
        }

    @classmethod
    def from_mapping(cls, value: object) -> ComponentSupportBounds:
        if not isinstance(value, Mapping) or set(value) != set(cls.__dataclass_fields__):
            raise LocalChemistrySupportError("component support-bound fields changed")
        try:
            result = cls(
                records=int(value["records"]),
                heavy_atoms_min=int(value["heavy_atoms_min"]),
                heavy_atoms_max=int(value["heavy_atoms_max"]),
                carbon_atoms_min=int(value["carbon_atoms_min"]),
                carbon_atoms_max=int(value["carbon_atoms_max"]),
                heteroatoms_min=int(value["heteroatoms_min"]),
                heteroatoms_max=int(value["heteroatoms_max"]),
                carbon_atoms_q01=float(value["carbon_atoms_q01"]),
                carbon_atoms_q99=float(value["carbon_atoms_q99"]),
                heteroatoms_q01=float(value["heteroatoms_q01"]),
                heteroatoms_q99=float(value["heteroatoms_q99"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LocalChemistrySupportError("component support bounds are malformed") from error
        if (
            result.records < 1
            or result.heavy_atoms_min < 1
            or result.heavy_atoms_min > result.heavy_atoms_max
            or result.carbon_atoms_min > result.carbon_atoms_max
            or result.heteroatoms_min > result.heteroatoms_max
        ):
            raise LocalChemistrySupportError("component support bounds are inconsistent")
        return result

    def contains(self, *, heavy_atoms: int, carbon_atoms: int, heteroatoms: int) -> bool:
        return (
            self.heavy_atoms_min <= heavy_atoms <= self.heavy_atoms_max
            and self.carbon_atoms_min <= carbon_atoms <= self.carbon_atoms_max
            and self.heteroatoms_min <= heteroatoms <= self.heteroatoms_max
        )


@dataclass(frozen=True)
class LocalChemistrySupport:
    """Immutable train-fold local chemistry support with no component identities."""

    schema_version: str
    atom_states: tuple[AtomState, ...]
    program_edges: Mapping[str, frozenset[ProgramEdge]]
    role_edges: Mapping[str, frozenset[RoleEdge]]
    program_triangles: Mapping[str, frozenset[ProgramTriangle]]
    role_triangles: Mapping[str, frozenset[RoleTriangle]]
    program_cycles: Mapping[str, frozenset[ProgramCycle]]
    role_cycles: Mapping[str, frozenset[RoleCycle]]
    component_bounds: Mapping[str, Mapping[str, ComponentSupportBounds]]
    training_records: Mapping[str, int]

    @property
    def programs(self) -> tuple[str, ...]:
        return tuple(sorted(self.training_records))

    @property
    def enforces_role_cycles(self) -> bool:
        """Whether the policy contains complete train-fold closure morphology support."""

        return self.schema_version == POLICY_SCHEMA

    def _require_program(self, program_id: str) -> None:
        if program_id not in self.training_records:
            raise LocalChemistrySupportError(f"local chemistry has no support for {program_id}")

    def allows_program_edge(
        self, program_id: str, left_symbol: str, bond: int, right_symbol: str
    ) -> bool:
        self._require_program(program_id)
        return _program_edge(left_symbol, bond, right_symbol) in self.program_edges[program_id]

    def allows_role_edge(
        self,
        program_id: str,
        left_role: str,
        left_symbol: str,
        bond: int,
        right_role: str,
        right_symbol: str,
    ) -> bool:
        self._require_program(program_id)
        return (
            _role_edge(left_role, left_symbol, bond, right_role, right_symbol)
            in self.role_edges[program_id]
        )

    def allows_role_edge_for_any_bond(
        self,
        program_id: str,
        left_role: str,
        left_symbol: str,
        right_role: str,
        right_symbol: str,
    ) -> bool:
        return any(
            self.allows_role_edge(
                program_id,
                left_role,
                left_symbol,
                bond,
                right_role,
                right_symbol,
            )
            for bond in range(4)
        )

    def allows_role_triangle(self, program_id: str, values: Iterable[tuple[str, str]]) -> bool:
        self._require_program(program_id)
        return _role_triangle(values) in self.role_triangles[program_id]

    def allows_any_role_triangle(self, program_id: str, roles: Iterable[str]) -> bool:
        """Return whether any supported triangle has this semantic-role multiset."""

        self._require_program(program_id)
        expected = tuple(sorted(roles))
        if len(expected) != 3:
            raise LocalChemistrySupportError("a role triangle needs exactly three roles")
        return any(
            tuple(role for role, _ in signature) == expected
            for signature in self.role_triangles[program_id]
        )

    def allows_role_cycle(
        self,
        program_id: str,
        values: Iterable[tuple[str, str]],
    ) -> bool:
        """Return whether one complete role-and-element ring signature was observed."""

        self._require_program(program_id)
        if not self.enforces_role_cycles:
            raise LocalChemistrySupportError("legacy local chemistry has no full-cycle support")
        return _role_cycle(values) in self.role_cycles[program_id]

    def allows_any_role_cycle(self, program_id: str, roles: Iterable[str]) -> bool:
        """Return whether a generated closure's complete role multiset was observed."""

        self._require_program(program_id)
        if not self.enforces_role_cycles:
            raise LocalChemistrySupportError("legacy local chemistry has no full-cycle support")
        expected = tuple(sorted(roles))
        if len(expected) < 3:
            raise LocalChemistrySupportError("a ring needs at least three roles")
        return any(
            tuple(role for role, _ in signature) == expected
            for signature in self.role_cycles[program_id]
        )

    def component_support_bounds(self, program_id: str, role: str) -> ComponentSupportBounds:
        """Return the immutable observed bounds for one program-role component."""

        self._require_program(program_id)
        try:
            return self.component_bounds[program_id][role]
        except KeyError as error:
            raise LocalChemistrySupportError(
                f"local chemistry has no component support for {program_id}/{role}"
            ) from error

    def component_is_within_observed_support(
        self,
        program_id: str,
        role: str,
        *,
        heavy_atoms: int,
        carbon_atoms: int,
        heteroatoms: int,
    ) -> bool:
        bounds = self.component_support_bounds(program_id, role)
        return bounds.contains(
            heavy_atoms=heavy_atoms,
            carbon_atoms=carbon_atoms,
            heteroatoms=heteroatoms,
        )

    def to_mapping(self) -> dict[str, Any]:
        if self.schema_version not in {LEGACY_POLICY_SCHEMA, POLICY_SCHEMA}:
            raise LocalChemistrySupportError("unsupported local chemistry policy schema")
        programs: dict[str, dict[str, Any]] = {}
        for program_id in self.programs:
            program = {
                "training_records": int(self.training_records[program_id]),
                "program_edges": [list(value) for value in sorted(self.program_edges[program_id])],
                "role_edges": [list(value) for value in sorted(self.role_edges[program_id])],
                "program_triangles": [
                    list(value) for value in sorted(self.program_triangles[program_id])
                ],
                "role_triangles": [
                    [list(atom) for atom in value]
                    for value in sorted(self.role_triangles[program_id])
                ],
                "component_bounds": {
                    role: bounds.to_mapping()
                    for role, bounds in sorted(self.component_bounds[program_id].items())
                },
            }
            if self.enforces_role_cycles:
                program["program_cycles"] = [
                    list(value) for value in sorted(self.program_cycles[program_id])
                ]
                program["role_cycles"] = [
                    [list(atom) for atom in value] for value in sorted(self.role_cycles[program_id])
                ]
            programs[program_id] = program
        policy = {
            "training_fold_only": True,
            "complete_component_identities_stored": False,
            "program_and_role_conditioned_edges": True,
            "program_and_role_conditioned_three_membered_rings": True,
            "component_hard_bounds": "observed_training_min_max",
            "component_diagnostic_bounds": "training_q01_q99_nonselecting",
        }
        if self.enforces_role_cycles:
            policy.update(
                {
                    "program_and_role_conditioned_fundamental_cycles": True,
                    "cycle_source": "variable_training_closures_relative_to_spanning_tree",
                }
            )
        return {
            "schema_version": self.schema_version,
            "atom_states": [
                {
                    "symbol": state.symbol,
                    "formal_charge": state.formal_charge,
                    "aromatic": state.aromatic,
                    "explicit_hydrogens": state.explicit_hydrogens,
                }
                for state in self.atom_states
            ],
            "programs": programs,
            "policy": policy,
            "nonclaims": [
                "Local support is not synthesis success, stability, safety, activity or route closure.",
                "Absence from local training support is an abstention, not proof of impossibility.",
                "The policy stores no component identity, fingerprint or complete fragment.",
            ],
        }

    @classmethod
    def from_mapping(cls, value: object) -> LocalChemistrySupport:
        if not isinstance(value, Mapping) or set(value) != {
            "schema_version",
            "atom_states",
            "programs",
            "policy",
            "nonclaims",
        }:
            raise LocalChemistrySupportError("local chemistry policy fields changed")
        schema_version = value["schema_version"]
        if schema_version not in {LEGACY_POLICY_SCHEMA, POLICY_SCHEMA}:
            raise LocalChemistrySupportError("unsupported local chemistry policy schema")
        raw_atoms = value["atom_states"]
        raw_programs = value["programs"]
        if not isinstance(raw_atoms, Sequence) or not isinstance(raw_programs, Mapping):
            raise LocalChemistrySupportError("local chemistry policy payload is malformed")
        atoms: list[AtomState] = []
        for row in raw_atoms:
            if not isinstance(row, Mapping) or set(row) != {
                "symbol",
                "formal_charge",
                "aromatic",
                "explicit_hydrogens",
            }:
                raise LocalChemistrySupportError("local chemistry atom state is malformed")
            atoms.append(
                AtomState(
                    symbol=str(row["symbol"]),
                    formal_charge=int(row["formal_charge"]),
                    aromatic=bool(row["aromatic"]),
                    explicit_hydrogens=int(row["explicit_hydrogens"]),
                )
            )
        program_edges: dict[str, frozenset[ProgramEdge]] = {}
        role_edges: dict[str, frozenset[RoleEdge]] = {}
        program_triangles: dict[str, frozenset[ProgramTriangle]] = {}
        role_triangles: dict[str, frozenset[RoleTriangle]] = {}
        program_cycles: dict[str, frozenset[ProgramCycle]] = {}
        role_cycles: dict[str, frozenset[RoleCycle]] = {}
        component_bounds: dict[str, dict[str, ComponentSupportBounds]] = {}
        training_records: dict[str, int] = {}
        try:
            for program_id, raw in raw_programs.items():
                if not isinstance(program_id, str) or not isinstance(raw, Mapping):
                    raise TypeError
                expected_fields = {
                    "training_records",
                    "program_edges",
                    "role_edges",
                    "program_triangles",
                    "role_triangles",
                    "component_bounds",
                }
                if schema_version == POLICY_SCHEMA:
                    expected_fields |= {"program_cycles", "role_cycles"}
                if set(raw) != expected_fields:
                    raise ValueError
                training_records[program_id] = int(raw["training_records"])
                program_edges[program_id] = frozenset(
                    (str(row[0]), int(row[1]), str(row[2])) for row in raw["program_edges"]
                )
                role_edges[program_id] = frozenset(
                    (str(row[0]), str(row[1]), int(row[2]), str(row[3]), str(row[4]))
                    for row in raw["role_edges"]
                )
                program_triangles[program_id] = frozenset(
                    (str(row[0]), str(row[1]), str(row[2])) for row in raw["program_triangles"]
                )
                role_triangles[program_id] = frozenset(
                    tuple((str(atom[0]), str(atom[1])) for atom in row)
                    for row in raw["role_triangles"]
                )
                if schema_version == POLICY_SCHEMA:
                    program_cycles[program_id] = frozenset(
                        tuple(str(symbol) for symbol in row) for row in raw["program_cycles"]
                    )
                    role_cycles[program_id] = frozenset(
                        tuple((str(atom[0]), str(atom[1])) for atom in row)
                        for row in raw["role_cycles"]
                    )
                else:
                    program_cycles[program_id] = frozenset(program_triangles[program_id])
                    role_cycles[program_id] = frozenset(role_triangles[program_id])
                component_bounds[program_id] = {
                    str(role): ComponentSupportBounds.from_mapping(bounds)
                    for role, bounds in raw["component_bounds"].items()
                }
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise LocalChemistrySupportError(
                "local chemistry program support is malformed"
            ) from error
        result = cls(
            schema_version=str(schema_version),
            atom_states=tuple(atoms),
            program_edges=program_edges,
            role_edges=role_edges,
            program_triangles=program_triangles,
            role_triangles=role_triangles,
            program_cycles=program_cycles,
            role_cycles=role_cycles,
            component_bounds=component_bounds,
            training_records=training_records,
        )
        if not result.atom_states or not result.programs:
            raise LocalChemistrySupportError("local chemistry support is empty")
        for program_id in result.programs:
            if (
                result.training_records[program_id] < 1
                or not result.program_edges[program_id]
                or not result.role_edges[program_id]
                or not result.component_bounds[program_id]
            ):
                raise LocalChemistrySupportError(
                    f"local chemistry support is incomplete for {program_id}"
                )
        return result


def build_local_chemistry_support(
    records: Iterable[SynthesisProgramGraphRecord],
    atom_states: Sequence[AtomState],
    *,
    schema_version: str = POLICY_SCHEMA,
) -> LocalChemistrySupport:
    """Fit local support from already fold-filtered records without storing their identities."""

    if schema_version not in {LEGACY_POLICY_SCHEMA, POLICY_SCHEMA}:
        raise LocalChemistrySupportError("unsupported local chemistry policy schema")
    atoms = tuple(atom_states)
    if not atoms:
        raise LocalChemistrySupportError("local chemistry requires a non-empty atom vocabulary")
    program_edges: dict[str, set[ProgramEdge]] = defaultdict(set)
    role_edges: dict[str, set[RoleEdge]] = defaultdict(set)
    program_triangles: dict[str, set[ProgramTriangle]] = defaultdict(set)
    role_triangles: dict[str, set[RoleTriangle]] = defaultdict(set)
    program_cycles: dict[str, set[ProgramCycle]] = defaultdict(set)
    role_cycles: dict[str, set[RoleCycle]] = defaultdict(set)
    component_rows: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for record in records:
        counts[record.program_id] += 1
        try:
            symbols = [atoms[int(state)].symbol for state in record.graph.node_states]
        except IndexError as error:
            raise LocalChemistrySupportError(
                f"training record exceeds the atom vocabulary: {record.graph.structure_id}"
            ) from error
        role_by_state = {block.role_state: block.role for block in record.component_blocks}
        try:
            roles = [role_by_state[int(state)] for state in record.role_states]
        except KeyError as error:
            raise LocalChemistrySupportError(
                f"training record has an unnamed role: {record.graph.structure_id}"
            ) from error
        edges = _graph_edges(record)
        for left, right, bond in edges:
            program_edges[record.program_id].add(_program_edge(symbols[left], bond, symbols[right]))
            role_edges[record.program_id].add(
                _role_edge(roles[left], symbols[left], bond, roles[right], symbols[right])
            )
        for triangle in _triangle_indices(record.node_count, edges):
            program_triangles[record.program_id].add(
                _program_triangle(symbols[index] for index in triangle)
            )
            role_triangles[record.program_id].add(
                _role_triangle((roles[index], symbols[index]) for index in triangle)
            )
        for cycle in fundamental_cycle_indices(record, variable_only=True):
            program_cycles[record.program_id].add(_program_cycle(symbols[index] for index in cycle))
            role_cycles[record.program_id].add(
                _role_cycle((roles[index], symbols[index]) for index in cycle)
            )
        for block in record.component_blocks:
            component_symbols = symbols[block.start : block.stop]
            carbon = component_symbols.count("C")
            heavy = len(component_symbols)
            component_rows[(record.program_id, block.role)].append((heavy, carbon, heavy - carbon))
    if not counts:
        raise LocalChemistrySupportError("local chemistry received no training records")
    components: dict[str, dict[str, ComponentSupportBounds]] = defaultdict(dict)
    for (program_id, role), rows in sorted(component_rows.items()):
        components[program_id][role] = ComponentSupportBounds.from_rows(rows)
    return LocalChemistrySupport(
        schema_version=schema_version,
        atom_states=atoms,
        program_edges={key: frozenset(program_edges[key]) for key in counts},
        role_edges={key: frozenset(role_edges[key]) for key in counts},
        program_triangles={key: frozenset(program_triangles[key]) for key in counts},
        role_triangles={key: frozenset(role_triangles[key]) for key in counts},
        program_cycles={key: frozenset(program_cycles[key]) for key in counts},
        role_cycles={key: frozenset(role_cycles[key]) for key in counts},
        component_bounds={key: dict(components[key]) for key in counts},
        training_records=dict(counts),
    )


def assess_product_local_chemistry(
    smiles: str | None,
    *,
    program_id: str,
    support: LocalChemistrySupport,
) -> dict[str, Any]:
    """Assess one complete graph under the program-only, method-blind local-support view."""

    support._require_program(program_id)
    molecule = None
    with rdBase.BlockLogs():
        if smiles:
            molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        return {
            "schema_version": ASSESSMENT_SCHEMA,
            "program_id": program_id,
            "valid_connected": False,
            "local_chemistry_supported": False,
            "unsupported_edge_count": 0,
            "unsupported_three_membered_ring_count": 0,
            "failure_types": ["invalid_or_disconnected"],
        }
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except (RuntimeError, ValueError):
        return {
            "schema_version": ASSESSMENT_SCHEMA,
            "program_id": program_id,
            "valid_connected": False,
            "local_chemistry_supported": False,
            "unsupported_edge_count": 0,
            "unsupported_three_membered_ring_count": 0,
            "failure_types": ["kekulization_failure"],
        }
    unsupported_edges = 0
    for bond in molecule.GetBonds():
        try:
            bond_state = SPARSE_BOND_TO_INDEX[bond.GetBondType()]
        except KeyError:
            unsupported_edges += 1
            continue
        if not support.allows_program_edge(
            program_id,
            bond.GetBeginAtom().GetSymbol(),
            bond_state,
            bond.GetEndAtom().GetSymbol(),
        ):
            unsupported_edges += 1
    edges = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), 0) for bond in molecule.GetBonds())
    unsupported_triangles = sum(
        _program_triangle(molecule.GetAtomWithIdx(index).GetSymbol() for index in triangle)
        not in support.program_triangles[program_id]
        for triangle in _triangle_indices(molecule.GetNumAtoms(), edges)
    )
    failures = []
    if unsupported_edges:
        failures.append("unsupported_local_edge")
    if unsupported_triangles:
        failures.append("unsupported_three_membered_ring")
    return {
        "schema_version": ASSESSMENT_SCHEMA,
        "program_id": program_id,
        "valid_connected": True,
        "local_chemistry_supported": not failures,
        "unsupported_edge_count": unsupported_edges,
        "unsupported_three_membered_ring_count": unsupported_triangles,
        "failure_types": failures,
    }


__all__ = [
    "ASSESSMENT_SCHEMA",
    "LEGACY_POLICY_SCHEMA",
    "POLICY_SCHEMA",
    "ComponentSupportBounds",
    "LocalChemistrySupport",
    "LocalChemistrySupportError",
    "assess_product_local_chemistry",
    "build_local_chemistry_support",
    "fundamental_cycle_indices",
    "tree_path_indices",
]
