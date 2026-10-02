"""Topology-conditioned discrete flow for Ugi lipid chemistry realization."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from forge.model.conditioning.adapter_nodes import AdapterNodeConditioning
from forge.model.conditioning.ugi import ORIGIN_TO_INDEX
from forge.model.conditioning.ugi_chemistry import (
    ROOT_BOND_TARGET,
    ChemistryTopologyCondition,
)
from forge.model.networks.dense_flow import AtomState
from forge.model.networks.whole_lipid import DeterministicSparseFlowBlock
from forge.model.representation.sparse_graph import (
    _INDEX_TO_BOND_TYPE,
    V5SparseGraphRecord,
    v5_graph_to_molecule,
)
from forge.model.sampling.chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None


class UgiChemistryFlowError(RuntimeError):
    """Raised when Ugi chemistry tensors violate their topology contract."""


TERMINAL_DECODE_FAILURE_SCHEMA = "forge.ugi_terminal_decode_failure.v1"


class UgiTerminalDecodeError(UgiChemistryFlowError):
    """One typed, serializable terminal-support failure."""

    def __init__(self, code: str, stage: str, **context: int | str | bool) -> None:
        if not code or not stage:
            raise ValueError("terminal decode failure code and stage must be nonempty")
        self.code = code
        self.stage = stage
        self.context = dict(sorted(context.items()))
        super().__init__(f"{stage}:{code}")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TERMINAL_DECODE_FAILURE_SCHEMA,
            "stage": self.stage,
            "code": self.code,
            "context": self.context,
        }


TERMINAL_DECODER_MODES = (
    "argmax",
    "stochastic",
    "bond_stochastic",
    "atom_bond_stochastic",
    "decoration_bond_stochastic",
)


@dataclass(frozen=True)
class UgiChemistrySample:
    """Generated chemistry on one fixed sparse Ugi support topology."""

    atom_states: np.ndarray
    parent_bond_states: np.ndarray
    closure_bond_states: np.ndarray
    decoration_anchor: int
    decoration_anchors: np.ndarray | None = None
    decoration_atom_states: np.ndarray | None = None
    decoration_bond_states: np.ndarray | None = None


def _gather_nodes(hidden: Any, indices: Any) -> Any:
    return torch.gather(
        hidden,
        1,
        indices[:, :, None].expand(-1, -1, hidden.shape[-1]),
    )


if nn is not None:

    class UgiChemistryFlow(nn.Module):
        """Sparse graph denoiser for atoms, bonds, and one Ugi decoration."""

        def __init__(
            self,
            *,
            atom_classes: int,
            bond_classes: int,
            maximum_nodes: int,
            maximum_distance: int,
            hidden_dim: int,
            layers: int,
            dropout: float,
            maximum_decorations: int = 1,
        ) -> None:
            super().__init__()
            if (
                atom_classes < 1
                or bond_classes not in {3, 4}
                or maximum_nodes < 8
                or maximum_distance < 1
                or hidden_dim < 16
                or layers < 1
                or not 0 <= dropout < 1
                or maximum_decorations < 1
            ):
                raise UgiChemistryFlowError("invalid Ugi chemistry architecture")
            self.atom_classes = atom_classes
            self.bond_classes = bond_classes
            self.hidden_dim = hidden_dim
            self.maximum_nodes = maximum_nodes
            self.maximum_decorations = maximum_decorations
            self.atom_embedding = nn.Embedding(atom_classes, hidden_dim)
            self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
            self.adapter_conditioning = AdapterNodeConditioning(
                hidden_dim=hidden_dim,
                maximum_distance=maximum_distance,
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.blocks = nn.ModuleList(
                DeterministicSparseFlowBlock(hidden_dim, dropout) for _ in range(layers)
            )
            self.atom_output = nn.Linear(hidden_dim, atom_classes)
            self.parent_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.closure_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.decoration_query = nn.Linear(hidden_dim, hidden_dim)
            self.decoration_key = nn.Linear(hidden_dim, hidden_dim)
            self.no_decoration = nn.Parameter(torch.zeros(hidden_dim))
            if maximum_decorations > 1:
                self.decoration_slot_embedding = nn.Embedding(
                    maximum_decorations,
                    hidden_dim,
                )
                self.decoration_atom_output = nn.Linear(hidden_dim, atom_classes)
                self.decoration_bond_output = nn.Linear(hidden_dim, bond_classes)
            else:
                self.decoration_slot_embedding = None
                self.decoration_atom_output = None
                self.decoration_bond_output = None

        def forward(
            self,
            *,
            nodes: Any,
            parent_bonds: Any,
            closure_bonds: Any,
            t: Any,
            topology: dict[str, Any],
        ) -> dict[str, Any]:
            node_mask = topology["node_mask"]
            child_mask = topology["child_mask"]
            closure_mask = topology["closure_mask"]
            if (
                nodes.shape != node_mask.shape
                or parent_bonds.shape != node_mask.shape
                or closure_bonds.shape != closure_mask.shape
                or t.shape != (nodes.shape[0],)
            ):
                raise UgiChemistryFlowError("Ugi chemistry state shapes do not agree")
            adapter_hidden = self.adapter_conditioning(
                origin_states=topology["origin_states"],
                core_position_states=topology["core_position_states"],
                port_states=topology["port_states"],
                distance_to_core=topology["distance_to_core"],
                distance_to_own_port=topology["distance_to_own_port"],
                adapter_mask=topology["adapter_mask"],
            )
            hidden = (
                self.atom_embedding(nodes)
                + self.bond_embedding(parent_bonds) * child_mask[:, :, None]
                + adapter_hidden
                + self.time_embedding(t[:, None])[:, None, :]
            )
            hidden = self.input_norm(hidden) * node_mask[:, :, None]
            closure_hidden = self.bond_embedding(closure_bonds) * closure_mask[:, :, None]
            for block in self.blocks:
                hidden = block(
                    hidden,
                    topology["parents"],
                    topology["closure_left"],
                    topology["closure_right"],
                    closure_hidden,
                    node_mask,
                    child_mask,
                    closure_mask,
                )
            parent_hidden = _gather_nodes(hidden, topology["parents"])
            left_hidden = _gather_nodes(hidden, topology["closure_left"])
            right_hidden = _gather_nodes(hidden, topology["closure_right"])
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden /= node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            query = self.decoration_query(global_hidden)
            node_keys = self.decoration_key(hidden)
            output = {
                "nodes": self.atom_output(hidden),
                "parent_bonds": self.parent_bond_output(torch.cat((hidden, parent_hidden), dim=-1)),
                "closure_bonds": self.closure_bond_output(
                    torch.cat((left_hidden, right_hidden), dim=-1)
                ),
            }
            if self.maximum_decorations == 1:
                node_scores = torch.einsum("bd,bnd->bn", query, node_keys) / math.sqrt(
                    self.hidden_dim
                )
                node_scores = node_scores.masked_fill(
                    ~topology["atom_variable_mask"],
                    -torch.inf,
                )
                none_score = (query * self.no_decoration[None, :]).sum(
                    dim=1,
                    keepdim=True,
                )
                output["decoration_anchor"] = torch.cat((none_score, node_scores), dim=1)
                return output
            slots = torch.arange(self.maximum_decorations, device=hidden.device)
            slot_hidden = query[:, None, :] + self.decoration_slot_embedding(slots)[None, :, :]
            node_scores = torch.einsum("bsd,bnd->bsn", slot_hidden, node_keys) / math.sqrt(
                self.hidden_dim
            )
            node_scores = node_scores.masked_fill(
                ~topology["atom_variable_mask"][:, None, :],
                -torch.inf,
            )
            none_score = torch.einsum(
                "bsd,d->bs",
                slot_hidden,
                self.no_decoration,
            )[:, :, None]
            output.update(
                {
                    "decoration_anchors": torch.cat((none_score, node_scores), dim=2),
                    "decoration_atoms": self.decoration_atom_output(slot_hidden),
                    "decoration_bonds": self.decoration_bond_output(slot_hidden),
                }
            )
            return output

else:  # pragma: no cover

    class UgiChemistryFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise UgiChemistryFlowError("Ugi chemistry flow requires torch")


def _atom_valence_units(state: AtomState) -> int:
    """Return the declared heavy-bond capacity after explicit hydrogens."""

    if state.symbol == "C":
        maximum = 8
    elif state.symbol == "F":
        maximum = 2
    elif state.symbol == "N":
        maximum = 8 if state.formal_charge > 0 else 6
    elif state.symbol == "O":
        maximum = 2 if state.formal_charge < 0 else 4
    elif state.symbol == "P":
        maximum = 10
    elif state.symbol == "S":
        maximum = 12
    elif state.symbol == "Si":
        maximum = 8
    else:  # pragma: no cover - vocabulary construction rejects this first
        raise UgiChemistryFlowError(f"no valence policy for atom state {state}")
    return maximum - 2 * state.explicit_hydrogens


_BOND_VALENCE_UNITS = np.asarray([2, 4, 6, 3], dtype=np.int64)


def _tree_path_edges(parents: np.ndarray, left: int, right: int) -> set[tuple[int, int]]:
    adjacency = [[] for _ in range(len(parents))]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        adjacency[parent].append(child)
        adjacency[child].append(parent)
    previous = {left: -1}
    queue = [left]
    for node in queue:
        if node == right:
            break
        for neighbor in adjacency[node]:
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)
    if right not in previous:
        raise UgiChemistryFlowError("conditioned chemistry tree is disconnected")
    edges: set[tuple[int, int]] = set()
    node = right
    while previous[node] >= 0:
        parent = previous[node]
        edges.add(tuple(sorted((node, parent))))
        node = parent
    return edges


def _masked_terminal_choice(
    logits: Any,
    valid: Any,
    *,
    mode: str,
    generator: Any | None,
    temperature: float,
) -> int:
    """Choose one feasible terminal state without changing its support."""

    if torch is None or mode not in {"argmax", "stochastic"} or temperature <= 0:
        raise UgiChemistryFlowError("invalid terminal decoder configuration")
    if logits.ndim != 1 or valid.shape != logits.shape or not bool(valid.any()):
        raise UgiChemistryFlowError("terminal choice has no feasible state")
    masked = logits.masked_fill(~valid, -torch.inf)
    if mode == "argmax":
        return int(masked.argmax())
    if generator is None:
        raise UgiChemistryFlowError("stochastic terminal decoding requires a generator")
    probabilities = (masked / temperature).softmax(dim=0)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _terminal_choice_modes(mode: str) -> tuple[str, str, str]:
    """Resolve atom, bond and decoration readouts for one declared decoder mode."""

    policies = {
        "argmax": ("argmax", "argmax", "argmax"),
        "stochastic": ("stochastic", "stochastic", "stochastic"),
        "bond_stochastic": ("argmax", "stochastic", "argmax"),
        "atom_bond_stochastic": ("stochastic", "stochastic", "argmax"),
        "decoration_bond_stochastic": ("argmax", "stochastic", "stochastic"),
    }
    try:
        return policies[mode]
    except KeyError as error:
        raise UgiChemistryFlowError(f"unsupported terminal decoder mode: {mode}") from error


def _terminal_channel_temperatures(
    condition: ChemistryTopologyCondition,
    *,
    temperature: float,
    atom_temperature: float | None,
    bond_temperature: float | None,
    decoration_temperature: float | None,
    atom_temperatures_by_origin: Sequence[float] | None,
) -> tuple[np.ndarray, float, float]:
    """Resolve backward-compatible channel and role-conditional temperatures."""

    base = float(temperature)
    resolved_atom = base if atom_temperature is None else float(atom_temperature)
    resolved_bond = base if bond_temperature is None else float(bond_temperature)
    resolved_decoration = base if decoration_temperature is None else float(decoration_temperature)
    if min(base, resolved_atom, resolved_bond, resolved_decoration) <= 0:
        raise UgiChemistryFlowError("terminal temperatures must be positive")
    atom_by_node = np.full(condition.node_count, resolved_atom, dtype=np.float64)
    if atom_temperatures_by_origin is not None:
        per_origin = np.asarray(atom_temperatures_by_origin, dtype=np.float64)
        if (
            per_origin.ndim != 1
            or per_origin.size != len(ROLE_NAMES)
            or not np.all(np.isfinite(per_origin))
            or np.any(per_origin <= 0)
        ):
            raise UgiChemistryFlowError("invalid origin-conditional atom temperatures")
        for role_index, role in enumerate(ROLE_NAMES):
            atom_by_node[condition.origin_states == ORIGIN_TO_INDEX[role]] = per_origin[role_index]
    return atom_by_node, resolved_bond, resolved_decoration


def valence_constrained_terminal_sample(
    condition: ChemistryTopologyCondition,
    terminal: dict[str, Any],
    batch_index: int,
    atom_vocabulary: tuple[AtomState, ...],
    maximum_decorations: int,
    forbid_oxygen_oxygen_bonds: bool = True,
    *,
    mode: str = "argmax",
    generator: Any | None = None,
    temperature: float = 1.0,
    atom_temperature: float | None = None,
    bond_temperature: float | None = None,
    decoration_temperature: float | None = None,
    atom_temperatures_by_origin: Sequence[float] | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    program_id: str | None = None,
    local_chemistry_constraint_scope: str = "role_edges_cycles_bounds",
) -> UgiChemistrySample:
    """Decode terminal logits only within valence-feasible sparse support.

    This is a support-constrained decoder, not structural repair: topology is
    unchanged, no atom or edge is added except a decoration explicitly emitted
    by a learned slot, and every categorical decision is selected from its
    model logits after impossible states are masked.
    """

    node_count = condition.node_count
    closure_count = condition.closure_count
    if local_chemistry_constraint_scope not in {
        "role_edges_only",
        "role_edges_cycles_bounds",
    }:
        raise UgiChemistryFlowError("unsupported role-local chemistry constraint scope")
    if local_chemistry_support is not None:
        if program_id is None:
            raise UgiChemistryFlowError(
                "role-local terminal decoding requires an explicit program_id"
            )
        if tuple(local_chemistry_support.atom_states) != tuple(atom_vocabulary):
            raise UgiChemistryFlowError(
                "role-local chemistry atom vocabulary differs from the decoder vocabulary"
            )
        role_by_origin = {index: name for name, index in ORIGIN_TO_INDEX.items()}
        try:
            role_names = tuple(role_by_origin[int(value)] for value in condition.origin_states)
        except KeyError as error:
            raise UgiChemistryFlowError(
                "condition contains an origin absent from the role-local policy"
            ) from error
        if any(role == "adapter_unspecified" for role in role_names):
            raise UgiChemistryFlowError(
                "condition contains an adapter-unspecified role during Ugi decoding"
            )
        for role in sorted(set(role_names)):
            local_chemistry_support.component_support_bounds(program_id, role)
    else:
        role_names = tuple("" for _ in range(node_count))
    atom_choice_mode, bond_choice_mode, decoration_choice_mode = _terminal_choice_modes(mode)
    atom_temperatures, resolved_bond_temperature, resolved_decoration_temperature = (
        _terminal_channel_temperatures(
            condition,
            temperature=temperature,
            atom_temperature=atom_temperature,
            bond_temperature=bond_temperature,
            decoration_temperature=decoration_temperature,
            atom_temperatures_by_origin=atom_temperatures_by_origin,
        )
    )
    parents = condition.parents
    degrees = np.zeros(node_count, dtype=np.int64)
    for child in range(1, node_count):
        degrees[child] += 1
        degrees[int(parents[child])] += 1
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        degrees[int(left)] += 1
        degrees[int(right)] += 1

    fixed_extra = np.zeros(node_count, dtype=np.int64)
    for child in range(1, node_count):
        if condition.fixed_parent_bond_mask[child]:
            units = int(_BOND_VALENCE_UNITS[int(condition.fixed_parent_bond_states[child])])
            extra = units - 2
            fixed_extra[child] += extra
            fixed_extra[int(parents[child])] += extra
    for slot, (left, right) in enumerate(
        zip(condition.closure_left, condition.closure_right, strict=True)
    ):
        if condition.fixed_closure_bond_mask[slot]:
            units = int(_BOND_VALENCE_UNITS[int(condition.fixed_closure_bond_states[slot])])
            extra = units - 2
            fixed_extra[int(left)] += extra
            fixed_extra[int(right)] += extra
    minimum_units = 2 * degrees + fixed_extra

    cycle_edges: set[tuple[int, int]] = set()
    cycle_nodes_by_closure: list[set[int]] = []
    cycle_paths_by_closure: list[tuple[int, ...]] = []
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        path_edges = _tree_path_edges(parents, int(left), int(right))
        path_edges.add(tuple(sorted((int(left), int(right)))))
        cycle_edges.update(path_edges)
        cycle_nodes_by_closure.append({node for edge in path_edges for node in edge})
        cycle_paths_by_closure.append(tree_path_indices(parents, int(left), int(right)))
    nodes_in_cycles = {node for group in cycle_nodes_by_closure for node in group}

    node_logits = terminal["nodes"][batch_index, :node_count].detach().cpu()
    atom_states = np.zeros(node_count, dtype=np.int64)
    raw_states = np.zeros(node_count, dtype=np.int64)
    assigned = np.zeros(node_count, dtype=np.bool_)
    neighbors: list[set[int]] = [set() for _ in range(node_count)]
    for child in range(1, node_count):
        parent = int(parents[child])
        neighbors[child].add(parent)
        neighbors[parent].add(child)
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    for node in range(node_count):
        if condition.fixed_atom_mask[node]:
            state_index = int(condition.fixed_atom_states[node])
            atom_states[node] = raw_states[node] = state_index
            assigned[node] = True

    role_nodes = {
        role: tuple(index for index, observed in enumerate(role_names) if observed == role)
        for role in sorted(set(role_names))
    }

    def atom_is_role_locally_allowed(node: int, state: AtomState) -> bool:
        if local_chemistry_support is None:
            return True
        assert program_id is not None
        role = role_names[node]
        bounds = local_chemistry_support.component_support_bounds(program_id, role)
        nodes = role_nodes[role]
        if local_chemistry_constraint_scope == "role_edges_cycles_bounds":
            assigned_symbols = [
                atom_vocabulary[int(atom_states[index])].symbol
                for index in nodes
                if assigned[index] and index != node
            ]
            remaining_after = sum(not assigned[index] and index != node for index in nodes)
            carbon = assigned_symbols.count("C") + int(state.symbol == "C")
            hetero = len(assigned_symbols) - assigned_symbols.count("C") + int(state.symbol != "C")
            if (
                carbon > bounds.carbon_atoms_max
                or carbon + remaining_after < bounds.carbon_atoms_min
                or hetero > bounds.heteroatoms_max
                or hetero + remaining_after < bounds.heteroatoms_min
            ):
                return False
        for neighbor in neighbors[node]:
            if not assigned[neighbor] or neighbor == node:
                continue
            neighbor_state = atom_vocabulary[int(atom_states[neighbor])]
            if not local_chemistry_support.allows_role_edge_for_any_bond(
                program_id,
                role,
                state.symbol,
                role_names[neighbor],
                neighbor_state.symbol,
            ):
                return False
        if (
            local_chemistry_constraint_scope == "role_edges_cycles_bounds"
            and local_chemistry_support.enforces_role_cycles
        ):
            for cycle in cycle_paths_by_closure:
                if node not in cycle:
                    continue
                others = tuple(index for index in cycle if index != node)
                if not all(assigned[index] for index in others):
                    continue
                signature = (
                    (role, state.symbol),
                    *(
                        (
                            role_names[index],
                            atom_vocabulary[int(atom_states[index])].symbol,
                        )
                        for index in others
                    ),
                )
                if not local_chemistry_support.allows_role_cycle(program_id, signature):
                    return False
        return True

    def atom_is_allowed(node: int, state: AtomState) -> bool:
        if state.aromatic and node not in nodes_in_cycles:
            return False
        if not atom_is_role_locally_allowed(node, state):
            return False
        if not forbid_oxygen_oxygen_bonds or state.symbol != "O":
            return True
        return not any(
            assigned[neighbor] and atom_vocabulary[int(atom_states[neighbor])].symbol == "O"
            for neighbor in neighbors[node]
        )

    for node in range(node_count):
        if condition.fixed_atom_mask[node]:
            continue
        valence_valid = torch.tensor(
            [_atom_valence_units(state) >= minimum_units[node] for state in atom_vocabulary],
            dtype=torch.bool,
        )
        valid = torch.tensor(
            [
                bool(valence_valid[index]) and atom_is_allowed(node, state)
                for index, state in enumerate(atom_vocabulary)
            ],
            dtype=torch.bool,
        )
        if not bool(valid.any()):
            code = (
                "role_local_atom_state_unavailable"
                if bool(valence_valid.any()) and local_chemistry_support is not None
                else "no_valence_feasible_atom_state"
            )
            raise UgiTerminalDecodeError(
                code,
                "atom_state",
                node=node,
                origin=role_names[node] if local_chemistry_support is not None else "unconditioned",
                minimum_valence_units=int(minimum_units[node]),
            )
        raw_states[node] = _masked_terminal_choice(
            node_logits[node],
            valid,
            mode=atom_choice_mode,
            generator=generator,
            temperature=float(atom_temperatures[node]),
        )
        atom_states[node] = raw_states[node]
        assigned[node] = True

    # Aromaticity is a cycle-level state.  An isolated aromatic atom is never
    # emitted merely because its local logit is high.
    aromatic_cycle_nodes: set[int] = set()
    for group in cycle_nodes_by_closure:
        if all(atom_vocabulary[int(raw_states[node])].aromatic for node in group):
            aromatic_cycle_nodes.update(group)
    for node in range(node_count):
        if condition.fixed_atom_mask[node] or not atom_vocabulary[int(atom_states[node])].aromatic:
            continue
        if node not in aromatic_cycle_nodes:
            valid = torch.tensor(
                [
                    not state.aromatic
                    and _atom_valence_units(state) >= minimum_units[node]
                    and atom_is_allowed(node, state)
                    for state in atom_vocabulary
                ],
                dtype=torch.bool,
            )
            if not bool(valid.any()):
                raise UgiTerminalDecodeError(
                    "nonaromatic_state_unavailable",
                    "aromatic_cycle_consistency",
                    node=node,
                    origin=(
                        role_names[node] if local_chemistry_support is not None else "unconditioned"
                    ),
                )
            atom_states[node] = _masked_terminal_choice(
                node_logits[node],
                valid,
                mode=atom_choice_mode,
                generator=generator,
                temperature=float(atom_temperatures[node]),
            )

    capacities = np.asarray(
        [_atom_valence_units(atom_vocabulary[int(state)]) for state in atom_states],
        dtype=np.int64,
    )
    used_units = minimum_units.copy()
    parent_bonds = np.zeros(node_count, dtype=np.int64)
    parent_bonds[0] = ROOT_BOND_TARGET
    parent_logits = terminal["parent_bonds"][batch_index, :node_count].detach().cpu()
    closure_logits = terminal["closure_bonds"][batch_index, :closure_count].detach().cpu()

    def choose_bond(logits: Any, left: int, right: int, *, cycle_edge: bool) -> int:
        spare = min(capacities[left] - used_units[left], capacities[right] - used_units[right])
        valence_valid = torch.tensor(_BOND_VALENCE_UNITS <= 2 + max(0, int(spare)))
        valid = valence_valid.clone()
        left_aromatic = atom_vocabulary[int(atom_states[left])].aromatic
        right_aromatic = atom_vocabulary[int(atom_states[right])].aromatic
        if cycle_edge and left_aromatic and right_aromatic:
            valid[:] = False
            valid[3] = bool(spare >= 1)
        else:
            valid[3] = False
        valence_and_aromatic_valid = valid.clone()
        if local_chemistry_support is not None:
            assert program_id is not None
            left_symbol = atom_vocabulary[int(atom_states[left])].symbol
            right_symbol = atom_vocabulary[int(atom_states[right])].symbol
            for bond in range(len(valid)):
                if valid[bond] and not local_chemistry_support.allows_role_edge(
                    program_id,
                    role_names[left],
                    left_symbol,
                    bond,
                    role_names[right],
                    right_symbol,
                ):
                    valid[bond] = False
        if not bool(valid.any()):
            code = (
                "role_local_bond_state_unavailable"
                if bool(valence_and_aromatic_valid.any()) and local_chemistry_support is not None
                else "no_valence_feasible_bond_state"
            )
            raise UgiTerminalDecodeError(
                code,
                "bond_state",
                left=left,
                right=right,
                left_origin=(
                    role_names[left] if local_chemistry_support is not None else "unconditioned"
                ),
                right_origin=(
                    role_names[right] if local_chemistry_support is not None else "unconditioned"
                ),
                cycle_edge=cycle_edge,
            )
        selected = _masked_terminal_choice(
            logits,
            valid,
            mode=bond_choice_mode,
            generator=generator,
            temperature=resolved_bond_temperature,
        )
        extra = int(_BOND_VALENCE_UNITS[selected]) - 2
        used_units[left] += extra
        used_units[right] += extra
        return selected

    for child in range(1, node_count):
        parent = int(parents[child])
        if condition.fixed_parent_bond_mask[child]:
            parent_bonds[child] = int(condition.fixed_parent_bond_states[child])
        else:
            parent_bonds[child] = choose_bond(
                parent_logits[child],
                child,
                parent,
                cycle_edge=tuple(sorted((child, parent))) in cycle_edges,
            )
    closure_bonds = np.zeros(closure_count, dtype=np.int64)
    for slot, (left, right) in enumerate(
        zip(condition.closure_left, condition.closure_right, strict=True)
    ):
        if condition.fixed_closure_bond_mask[slot]:
            closure_bonds[slot] = int(condition.fixed_closure_bond_states[slot])
        else:
            closure_bonds[slot] = choose_bond(
                closure_logits[slot],
                int(left),
                int(right),
                cycle_edge=True,
            )

    if maximum_decorations == 1:
        anchor_logits = terminal["decoration_anchor"][batch_index].detach().cpu()
        if decoration_choice_mode == "argmax" and local_chemistry_support is None:
            decoration_anchor = int(anchor_logits.argmax())
            if decoration_anchor:
                anchor = decoration_anchor - 1
                if capacities[anchor] - used_units[anchor] < 4:
                    decoration_anchor = 0
        else:
            valid_anchors = torch.ones_like(anchor_logits, dtype=torch.bool)
            for encoded_anchor in range(1, anchor_logits.numel()):
                anchor = encoded_anchor - 1
                allowed = bool(capacities[anchor] - used_units[anchor] >= 4)
                if allowed and local_chemistry_support is not None:
                    assert program_id is not None
                    allowed = local_chemistry_support.allows_role_edge(
                        program_id,
                        role_names[anchor],
                        atom_vocabulary[int(atom_states[anchor])].symbol,
                        1,
                        role_names[anchor],
                        "O",
                    )
                valid_anchors[encoded_anchor] = allowed
            decoration_anchor = _masked_terminal_choice(
                anchor_logits,
                valid_anchors,
                mode=decoration_choice_mode,
                generator=generator,
                temperature=resolved_decoration_temperature,
            )
        return UgiChemistrySample(
            atom_states=atom_states,
            parent_bond_states=parent_bonds,
            closure_bond_states=closure_bonds,
            decoration_anchor=decoration_anchor,
        )

    decoration_anchors = np.zeros(maximum_decorations, dtype=np.int64)
    decoration_atoms = np.zeros(maximum_decorations, dtype=np.int64)
    decoration_bonds = np.zeros(maximum_decorations, dtype=np.int64)
    for slot in range(maximum_decorations):
        anchor_logits = terminal["decoration_anchors"][batch_index, slot].detach().cpu()
        atom_logits = terminal["decoration_atoms"][batch_index, slot].detach().cpu()
        bond_logits = terminal["decoration_bonds"][batch_index, slot].detach().cpu()
        remaining_anchors = torch.ones_like(anchor_logits, dtype=torch.bool)
        emitted = False
        while bool(remaining_anchors.any()):
            encoded_anchor = _masked_terminal_choice(
                anchor_logits,
                remaining_anchors,
                mode=decoration_choice_mode,
                generator=generator,
                temperature=resolved_decoration_temperature,
            )
            if encoded_anchor == 0:
                break
            anchor = int(encoded_anchor) - 1
            remaining_anchors[encoded_anchor] = False
            if anchor >= node_count or condition.fixed_atom_mask[anchor]:
                continue
            feasible_pairs: list[tuple[int, int]] = []
            pair_scores: list[float] = []
            for atom_index, state in enumerate(atom_vocabulary):
                if state.aromatic:
                    continue
                if (
                    forbid_oxygen_oxygen_bonds
                    and state.symbol == "O"
                    and atom_vocabulary[int(atom_states[anchor])].symbol == "O"
                ):
                    continue
                for bond_index, units in enumerate(_BOND_VALENCE_UNITS):
                    if bond_index == 3:
                        continue
                    locally_allowed = True
                    if local_chemistry_support is not None:
                        assert program_id is not None
                        locally_allowed = local_chemistry_support.allows_role_edge(
                            program_id,
                            role_names[anchor],
                            atom_vocabulary[int(atom_states[anchor])].symbol,
                            bond_index,
                            role_names[anchor],
                            state.symbol,
                        )
                    if (
                        locally_allowed
                        and int(units) <= capacities[anchor] - used_units[anchor]
                        and int(units) <= _atom_valence_units(state)
                    ):
                        feasible_pairs.append((atom_index, bond_index))
                        pair_scores.append(float(atom_logits[atom_index] + bond_logits[bond_index]))
            if not feasible_pairs:
                continue
            scores = torch.as_tensor(pair_scores, dtype=atom_logits.dtype)
            pair_index = _masked_terminal_choice(
                scores,
                torch.ones_like(scores, dtype=torch.bool),
                mode=decoration_choice_mode,
                generator=generator,
                temperature=resolved_decoration_temperature,
            )
            atom_index, bond_index = feasible_pairs[pair_index]
            decoration_anchors[slot] = encoded_anchor
            decoration_atoms[slot] = atom_index
            decoration_bonds[slot] = bond_index
            used_units[anchor] += int(_BOND_VALENCE_UNITS[bond_index])
            emitted = True
            break
        if not emitted:
            continue
    return UgiChemistrySample(
        atom_states=atom_states,
        parent_bond_states=parent_bonds,
        closure_bond_states=closure_bonds,
        decoration_anchor=0,
        decoration_anchors=decoration_anchors,
        decoration_atom_states=decoration_atoms,
        decoration_bond_states=decoration_bonds,
    )


def chemistry_sample_to_molecule(
    condition: ChemistryTopologyCondition,
    sample: UgiChemistrySample,
    atom_vocabulary: tuple[AtomState, ...],
) -> Chem.Mol:
    """Construct one complete molecule; sanitization failure is not repaired."""

    expanded_decorations = sample.decoration_anchors is not None
    if (
        sample.atom_states.shape != (condition.node_count,)
        or sample.parent_bond_states.shape != (condition.node_count,)
        or sample.closure_bond_states.shape != (condition.closure_count,)
        or not 0 <= sample.decoration_anchor <= condition.node_count
    ):
        raise UgiChemistryFlowError("generated chemistry does not match its topology")
    if expanded_decorations and (
        sample.decoration_atom_states is None
        or sample.decoration_bond_states is None
        or sample.decoration_anchors.ndim != 1
        or sample.decoration_atom_states.shape != sample.decoration_anchors.shape
        or sample.decoration_bond_states.shape != sample.decoration_anchors.shape
        or np.any(sample.decoration_anchors < 0)
        or np.any(sample.decoration_anchors > condition.node_count)
        or np.any(sample.decoration_atom_states < 0)
        or np.any(sample.decoration_atom_states >= len(atom_vocabulary))
        or np.any(sample.decoration_bond_states < 0)
        or np.any(sample.decoration_bond_states >= len(_INDEX_TO_BOND_TYPE))
    ):
        raise UgiChemistryFlowError("generated sparse decorations are invalid")
    parent_bonds = sample.parent_bond_states.copy()
    parent_bonds[0] = 0
    record = V5SparseGraphRecord(
        structure_id=condition.structure_id,
        canonical_smiles="",
        node_states=sample.atom_states.copy(),
        offspring=condition.offspring.copy(),
        parent_bonds=parent_bonds,
        closure_left=condition.closure_left.copy(),
        closure_right=condition.closure_right.copy(),
        closure_bonds=sample.closure_bond_states.copy(),
        tree_traversal=condition.tree_traversal,
    )
    molecule = v5_graph_to_molecule(record, atom_vocabulary)
    if expanded_decorations:
        editable = Chem.RWMol(molecule)
        for anchor, atom_index, bond_index in zip(
            sample.decoration_anchors,
            sample.decoration_atom_states,
            sample.decoration_bond_states,
            strict=True,
        ):
            if int(anchor) == 0:
                continue
            state = atom_vocabulary[int(atom_index)]
            atom = Chem.Atom(state.symbol)
            atom.SetFormalCharge(state.formal_charge)
            atom.SetIsAromatic(state.aromatic)
            if state.explicit_hydrogens:
                atom.SetNumExplicitHs(state.explicit_hydrogens)
                atom.SetNoImplicit(True)
            decoration_index = editable.AddAtom(atom)
            editable.AddBond(
                int(anchor) - 1,
                decoration_index,
                _INDEX_TO_BOND_TYPE[int(bond_index)],
            )
        molecule = editable.GetMol()
        Chem.SanitizeMol(molecule)
    elif sample.decoration_anchor:
        editable = Chem.RWMol(molecule)
        oxygen_index = editable.AddAtom(Chem.Atom("O"))
        editable.AddBond(sample.decoration_anchor - 1, oxygen_index, Chem.BondType.DOUBLE)
        molecule = editable.GetMol()
        Chem.SanitizeMol(molecule)
    return molecule
