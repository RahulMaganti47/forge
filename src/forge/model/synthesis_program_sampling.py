"""Fixed-state-safe sampling for the shared Ugi/BL/LX whole-product flow."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.core.io import read_json_object
from forge.flow import rstar_step
from forge.model.defog_feasibility import AtomState, _model_state_sha256, graph_to_molecule
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.reaction_core_saturation import (
    BoundReactionCoreSaturation,
    ReactionCoreSaturationPolicy,
)
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import (
    collate_synthesis_program_layouts,
    decode_synthesis_program_argmax,
    resolve_synthesis_program_source_marginals,
    restore_synthesis_program_fixed_states,
)
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    INDEX_TO_DENSE_BOND,
    _endpoint_candidate_mask,
    _maximum_valence_units,
    _parent_candidate_mask,
    pointer_rstar_step,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord
from forge.model.synthesis_program_training import build_synthesis_program_flow
from forge.model.tensor_checkpoint import TensorCheckpointError, decode_tensor_state
from forge.model.ugi_transformer_topology import (
    UgiTransformerTopologyError,
    UgiTransformerTopologyPolicy,
    decode_ugi_exact_topology,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

CHECKPOINT_SCHEMA = "forge.synthesis_program_sparse_flow_checkpoint.v1"
TERMINAL_DECODE_POLICIES = (
    "unconstrained_argmax",
    "strict_valence_topology_argmax",
)
LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY = "strict_local_chemistry_argmax"
PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY = "strict_program_topology_argmax"
COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY = "strict_ugi_program_coupled_conditional"
CORE_SATURATION_TERMINAL_DECODE_POLICY = "strict_reaction_core_saturation_argmax"
SUPPORTED_TERMINAL_DECODE_POLICIES = (
    *TERMINAL_DECODE_POLICIES,
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
    PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
    COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
    CORE_SATURATION_TERMINAL_DECODE_POLICY,
)
STRICT_TERMINAL_DECODE_POLICIES = frozenset(
    {
        "strict_valence_topology_argmax",
        LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
        PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
        COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
        CORE_SATURATION_TERMINAL_DECODE_POLICY,
    }
)


class SynthesisProgramSamplingError(ValueError):
    """A shared synthesis-program sampling request violates its fixed-state contract."""


def synthesis_program_source_marginals(
    records: Sequence[SynthesisProgramGraphRecord],
    weights: np.ndarray,
    *,
    node_classes: int,
    bond_classes: int,
    probability_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit full-support sources under explicit record weights, never raw family counts."""

    if (
        not records
        or weights.shape != (len(records),)
        or np.any(weights <= 0)
        or not np.isfinite(weights).all()
        or not np.isclose(weights.sum(), 1.0)
        or probability_floor <= 0
    ):
        raise SynthesisProgramSamplingError("source marginals require a finite normalized measure")
    nodes = np.full(node_classes, probability_floor, dtype=np.float64)
    bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
    for record, weight in zip(records, weights, strict=True):
        variable_atoms = record.graph.node_states[~record.fixed_atom_mask]
        variable_parent_bonds = record.graph.parent_bonds[1:][~record.fixed_parent_bond_mask[1:]]
        variable_closure_bonds = record.graph.closure_bonds[~record.fixed_closure_bond_mask]
        nodes += float(weight) * np.bincount(variable_atoms, minlength=node_classes)
        bonds += float(weight) * np.bincount(variable_parent_bonds, minlength=bond_classes)
        bonds += float(weight) * np.bincount(variable_closure_bonds, minlength=bond_classes)
    return nodes / nodes.sum(), bonds / bonds.sum()


def load_synthesis_program_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[
    Any,
    ReactionProgramVocabulary,
    tuple[AtomState, ...],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Load an authenticated non-executable tensor checkpoint."""

    if torch is None:
        raise SynthesisProgramSamplingError("checkpoint loading requires torch")
    package = read_json_object(
        checkpoint_path,
        error=SynthesisProgramSamplingError,
        label="shared synthesis-program checkpoint",
    )
    if (
        package.get("schema_version") != CHECKPOINT_SCHEMA
        or package.get("trusted_local_checkpoint") is not True
    ):
        raise SynthesisProgramSamplingError("checkpoint is not a trusted shared-flow checkpoint")
    raw_vocabulary = package.get("program_vocabulary")
    raw_atoms = package.get("atom_vocabulary")
    model_config = package.get("model_config")
    if (
        not isinstance(raw_vocabulary, Mapping)
        or not isinstance(raw_atoms, list)
        or not isinstance(model_config, Mapping)
    ):
        raise SynthesisProgramSamplingError("checkpoint is missing its model contract")
    vocabulary = ReactionProgramVocabulary(
        program_states=tuple(str(value) for value in raw_vocabulary["program_states"]),
        role_states=tuple(str(value) for value in raw_vocabulary["role_states"]),
        core_position_states=tuple(str(value) for value in raw_vocabulary["core_position_states"]),
        maximum_steps=int(raw_vocabulary["maximum_steps"]),
    )
    atom_vocabulary = tuple(
        AtomState(
            symbol=str(row["symbol"]),
            formal_charge=int(row["formal_charge"]),
            aromatic=bool(row["aromatic"]),
            explicit_hydrogens=int(row["explicit_hydrogens"]),
        )
        for row in raw_atoms
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise SynthesisProgramSamplingError("CUDA sampling requested but unavailable")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise SynthesisProgramSamplingError("MPS sampling requested but unavailable")
    model = build_synthesis_program_flow(
        vocabulary=vocabulary,
        node_classes=len(atom_vocabulary),
        model_config=model_config,
        device=resolved_device,
    )
    raw_state = package.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise SynthesisProgramSamplingError("checkpoint has no deterministic tensor state")
    try:
        state = decode_tensor_state(raw_state)
    except TensorCheckpointError as error:
        raise SynthesisProgramSamplingError(str(error)) from error
    model.load_state_dict(state, strict=True)
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise SynthesisProgramSamplingError("checkpoint model-state hash mismatch")
    node_marginal = np.asarray(package.get("node_marginal"), dtype=np.float64)
    bond_marginal = np.asarray(package.get("bond_marginal"), dtype=np.float64)
    expected_prefix = (len(vocabulary.program_states), len(vocabulary.role_states))
    global_sources = node_marginal.shape == (len(atom_vocabulary),) and bond_marginal.shape == (
        int(model_config["bond_classes"]),
    )
    program_role_sources = node_marginal.shape == (
        *expected_prefix,
        len(atom_vocabulary),
    ) and bond_marginal.shape == (*expected_prefix, int(model_config["bond_classes"]))
    if (
        not (global_sources or program_role_sources)
        or np.any(node_marginal <= 0)
        or np.any(bond_marginal <= 0)
        or not np.allclose(node_marginal.sum(axis=-1), 1.0)
        or not np.allclose(bond_marginal.sum(axis=-1), 1.0)
    ):
        raise SynthesisProgramSamplingError("checkpoint source marginals are invalid")
    model.eval()
    return model, vocabulary, atom_vocabulary, node_marginal, bond_marginal, package


def _move(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def _restore_fixed_states_in_place(
    state: dict[str, Any], layout: Mapping[str, Any]
) -> dict[str, Any]:
    """Restore immutable adapter states without cloning six complete state tensors."""

    fields = {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }
    for field, mask_name in fields.items():
        mask = layout[mask_name]
        state[field][mask] = layout[field][mask]
    return state


def _program_conditioning(model: Any, layout: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze one batch's clean program context, encoding it once where the model allows reuse.

    Every coordinate here is layout semantics: masks, program state, precursor roles, reaction-core
    positions, depth, repeat groups, component positions and role morphology.  None of them is
    denoised, so they are identical at all `sample_steps` calls and at the terminal call.  Models
    that expose ``prepare_program_memory`` additionally return their encoded program tokens, the
    per-block cross-attention projections of those tokens and the per-adapter routing weights, all
    of which are functions of these same fixed coordinates.  The model validates that the memory
    was built from these exact tensors and fails closed otherwise.
    """

    conditioning = {
        field: layout[field]
        for field in (
            "node_mask",
            "child_mask",
            "closure_mask",
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "adapter_mask",
            "repeat_group_states",
            "component_position_states",
            "component_instance_states",
            "role_morphology_states",
        )
    }
    prepare = getattr(model, "prepare_program_memory", None)
    if prepare is not None:
        conditioning["program_memory"] = prepare(
            program_states=layout["program_states"],
            role_states=layout["role_states"],
            core_position_states=layout["core_position_states"],
            program_depths=layout["program_depths"],
            adapter_mask=layout["adapter_mask"],
            repeat_group_states=layout["repeat_group_states"],
            component_position_states=layout["component_position_states"],
            role_morphology_states=layout["role_morphology_states"],
        )
    return conditioning


def _initial_state(
    layout: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    generator: Any,
) -> dict[str, Any]:
    batch, nodes = layout["node_mask"].shape
    closures = layout["closure_mask"].shape[1]
    node_source, parent_bond_source, closure_bond_source = (
        resolve_synthesis_program_source_marginals(layout, node_marginal, bond_marginal)
    )

    def draw(source: Any, shape: tuple[int, int]) -> Any:
        if source.ndim == 1:
            probabilities = source[None].expand(shape[0] * shape[1], -1)
        else:
            probabilities = source.reshape(shape[0] * shape[1], -1)
        return torch.multinomial(probabilities, 1, generator=generator).reshape(shape)

    state = {
        "nodes": draw(node_source, (batch, nodes)),
        "parent_bonds": draw(parent_bond_source, (batch, nodes)),
        "closure_bonds": draw(closure_bond_source, (batch, closures)),
    }
    parent_candidates = _parent_candidate_mask(layout["node_mask"])
    parent_probabilities = parent_candidates.to(torch.float32)
    parent_probabilities /= parent_probabilities.sum(dim=-1, keepdim=True).clamp(min=1)
    state["parents"] = torch.zeros((batch, nodes), dtype=torch.long, device=node_marginal.device)
    active_parents = layout["parent_variable_mask"]
    state["parents"][active_parents] = torch.multinomial(
        parent_probabilities[active_parents], 1, generator=generator
    ).squeeze(1)
    endpoint_candidates = _endpoint_candidate_mask(layout["node_mask"], closures)
    endpoint_probabilities = endpoint_candidates.to(torch.float32)
    endpoint_probabilities /= endpoint_probabilities.sum(dim=-1, keepdim=True).clamp(min=1)
    state["closure_left"] = torch.zeros(
        (batch, closures), dtype=torch.long, device=node_marginal.device
    )
    state["closure_right"] = torch.zeros_like(state["closure_left"])
    active_closures = layout["closure_endpoint_variable_mask"]
    for field in ("closure_left", "closure_right"):
        state[field][active_closures] = torch.multinomial(
            endpoint_probabilities[active_closures], 1, generator=generator
        ).squeeze(1)
    state["parent_bonds"][:, 0] = 0
    return _restore_fixed_states_in_place(state, layout)


def _fixed_state_exact(state: Mapping[str, Any], layout: Mapping[str, Any]) -> bool:
    return bool(_fixed_state_exact_tensor(state, layout).item())


def _fixed_state_exact_tensor(state: Mapping[str, Any], layout: Mapping[str, Any]) -> Any:
    """Return the fixed-state audit as a device scalar without forcing synchronization."""

    fields = {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }
    exact = torch.ones((), dtype=torch.bool, device=state["nodes"].device)
    for field, mask in fields.items():
        exact = exact & torch.all(state[field][layout[mask]] == layout[field][layout[mask]])
    return exact


def _fixed_state_exact_records(
    state: Mapping[str, Any], records: Sequence[SynthesisProgramGraphRecord]
) -> bool:
    """Audit a CPU terminal batch directly against its record-level immutable states."""

    fields = {
        "nodes": ("fixed_atom_mask", "node_states"),
        "parents": ("fixed_parent_bond_mask", "parents"),
        "parent_bonds": ("fixed_parent_bond_mask", "parent_bonds"),
        "closure_left": ("fixed_closure_bond_mask", "closure_left"),
        "closure_right": ("fixed_closure_bond_mask", "closure_right"),
        "closure_bonds": ("fixed_closure_bond_mask", "closure_bonds"),
    }
    # One host view per field, then per-record slicing of that view.  The audit previously built a
    # fresh NumPy array for every field of every record, which is six conversions per sample.
    views = {field: state[field].numpy() for field in fields}
    for index, record in enumerate(records):
        for field, (mask_name, target_name) in fields.items():
            mask = getattr(record, mask_name)
            target = (
                record.graph.node_states
                if target_name == "node_states"
                else getattr(record.graph, target_name)
            )
            observed = views[field][index, : len(target)]
            if not np.array_equal(observed[mask], target[mask]):
                return False
    return True


def _terminal_smiles(
    state: Mapping[str, Any],
    index: int,
    node_count: int,
    closure_count: int,
    atom_vocabulary: Sequence[AtomState],
) -> str | None:
    nodes = state["nodes"][index, :node_count].detach().cpu().numpy().astype(np.int64)
    parents = state["parents"][index, :node_count].detach().cpu().numpy().astype(np.int64)
    parent_bonds = state["parent_bonds"][index, :node_count].detach().cpu().numpy().astype(np.int64)
    left = state["closure_left"][index, :closure_count].detach().cpu().numpy().astype(np.int64)
    right = state["closure_right"][index, :closure_count].detach().cpu().numpy().astype(np.int64)
    closure_bonds = (
        state["closure_bonds"][index, :closure_count].detach().cpu().numpy().astype(np.int64)
    )
    edges = np.zeros((node_count, node_count), dtype=np.int64)
    try:
        for child in range(1, node_count):
            parent = int(parents[child])
            if parent < 0 or parent >= child:
                return None
            dense = INDEX_TO_DENSE_BOND[int(parent_bonds[child])]
            edges[child, parent] = edges[parent, child] = dense
        occupied = {tuple(sorted((child, int(parents[child])))) for child in range(1, node_count)}
        for slot in range(closure_count):
            pair = tuple(sorted((int(left[slot]), int(right[slot]))))
            if pair[0] < 0 or pair[1] >= node_count or pair[0] == pair[1] or pair in occupied:
                return None
            occupied.add(pair)
            dense = INDEX_TO_DENSE_BOND[int(closure_bonds[slot])]
            edges[pair[0], pair[1]] = edges[pair[1], pair[0]] = dense
        molecule = graph_to_molecule(nodes, edges, atom_vocabulary)
    except Exception:
        return None
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    # A graph may survive RDKit construction and serialization yet fail sanitizing parse on the
    # resulting SMILES.  Downstream metrics parse canonical strings again, so the sampler must not
    # label such a row valid and then fail the entire fixed-checkpoint evaluation later.
    reparsed = Chem.MolFromSmiles(smiles)
    if reparsed is None:
        return None
    return Chem.MolToSmiles(reparsed, canonical=True, isomericSmiles=False)


def _available_valence_units(state: AtomState) -> int:
    return _maximum_valence_units(state) - 2 * int(state.explicit_hydrogens)


_ATOM_CAPACITY_CACHE: dict[int, tuple[Sequence[AtomState], np.ndarray]] = {}
_BOND_UNIT_CACHE: dict[int, np.ndarray] = {}


def _atom_capacity_table(atom_vocabulary: Sequence[AtomState]) -> np.ndarray:
    """Return the frozen per-state valence capacities for one atom vocabulary.

    The table is a pure function of the vocabulary, but the strict decoder rebuilt it once per
    decoded record.  The vocabulary is held by reference in the cache value so its ``id`` cannot be
    recycled onto a different vocabulary while the entry is live.
    """

    entry = _ATOM_CAPACITY_CACHE.get(id(atom_vocabulary))
    if entry is None:
        table = np.asarray(
            [_available_valence_units(state) for state in atom_vocabulary], dtype=np.int64
        )
        table.setflags(write=False)
        entry = (atom_vocabulary, table)
        _ATOM_CAPACITY_CACHE[id(atom_vocabulary)] = entry
    return entry[1]


def _bond_unit_table(bond_classes: int) -> np.ndarray:
    """Return bond valence units on the host without a per-record device transfer.

    ``BOND_VALENCE_UNITS`` is a small constant torch tensor.  Reading it per decoded record forced
    a device-to-host copy for every sample in the batch, which on an accelerator is a full
    synchronization each time.  It is the same four numbers on every call.
    """

    table = _BOND_UNIT_CACHE.get(bond_classes)
    if table is None:
        table = BOND_VALENCE_UNITS[:bond_classes].cpu().numpy().astype(np.int64)
        table.setflags(write=False)
        _BOND_UNIT_CACHE[bond_classes] = table
    return table


def _argmax_allowed(logits: np.ndarray, valid: np.ndarray) -> int | None:
    if logits.ndim != 1 or valid.shape != logits.shape or not np.any(valid):
        return None
    masked = np.where(valid, logits, -np.inf)
    return int(np.argmax(masked))


def _exact_role_morphology_targets(
    record: SynthesisProgramGraphRecord,
) -> dict[int, tuple[int, int, int, int]]:
    """Read one explicit, internally consistent morphology target per semantic role."""

    states = record.role_morphology_states
    if states is None:
        raise SynthesisProgramSamplingError(
            "exact program-topology decoding requires explicit role morphology states"
        )
    targets: dict[int, tuple[int, int, int, int]] = {}
    for role_state in sorted(set(int(value) for value in record.role_states)):
        if role_state <= 0:
            continue
        values = states[record.role_states == role_state]
        unique = np.unique(values, axis=0)
        if unique.shape != (1, 4) or np.any(unique[0] < 1):
            raise SynthesisProgramSamplingError(
                "exact role morphology must be positive and constant within each role"
            )
        targets[role_state] = tuple(int(value) - 1 for value in unique[0])
    return targets


def _terminal_role_morphology(
    record: SynthesisProgramGraphRecord,
    *,
    parents: np.ndarray,
    closure_left: np.ndarray,
    closure_right: np.ndarray,
    parent_edge_mask: np.ndarray | None = None,
) -> dict[int, tuple[int, int, int, int]]:
    """Measure the same four coarse coordinates used by the conditioning tensor."""

    core = record.core_position_states > 1
    output: dict[int, tuple[int, int, int, int]] = {}
    for role_state in sorted(set(int(value) for value in record.role_states)):
        if role_state <= 0:
            continue
        role = record.role_states == role_state
        exterior_indices = set(np.flatnonzero(role & ~core).tolist())
        child_counts = {node: 0 for node in exterior_indices}
        attachments = 0
        for child in range(1, record.node_count):
            if parent_edge_mask is not None and not bool(parent_edge_mask[child]):
                continue
            parent = int(parents[child])
            if child in exterior_indices and parent in exterior_indices:
                child_counts[parent] += 1
            elif (
                child in exterior_indices
                and bool(core[parent])
                and int(record.role_states[parent]) == role_state
            ) or (
                parent in exterior_indices
                and bool(core[child])
                and int(record.role_states[child]) == role_state
            ):
                attachments += 1
        cycles = 0
        for left, right in zip(closure_left, closure_right, strict=True):
            left_index = int(left)
            right_index = int(right)
            if left_index in exterior_indices and right_index in exterior_indices:
                cycles += 1
            elif (
                left_index in exterior_indices
                and bool(core[right_index])
                and int(record.role_states[right_index]) == role_state
            ) or (
                right_index in exterior_indices
                and bool(core[left_index])
                and int(record.role_states[left_index]) == role_state
            ):
                attachments += 1
        output[role_state] = (
            len(exterior_indices),
            sum(max(children - 1, 0) for children in child_counts.values()),
            cycles,
            attachments,
        )
    return output


def _strict_terminal_record(
    predictions: Mapping[str, np.ndarray],
    index: int,
    record: SynthesisProgramGraphRecord,
    atom_vocabulary: Sequence[AtomState],
    local_chemistry_support: LocalChemistrySupport | None = None,
    *,
    enforce_program_topology: bool = False,
    core_saturation: BoundReactionCoreSaturation | None = None,
) -> tuple[dict[str, np.ndarray] | None, str | None]:
    """Decode one exact-size graph under topology and valence support, without fallback."""

    count = record.node_count
    closure_count = record.graph.closure_count
    bond_classes = predictions["parent_bonds"].shape[-1]
    if BOND_VALENCE_UNITS is None or bond_classes > len(BOND_VALENCE_UNITS):
        return None, "unsupported_bond_vocabulary"
    bond_units = _bond_unit_table(bond_classes)
    atom_capacities = _atom_capacity_table(atom_vocabulary)
    maximum_capacity = int(atom_capacities.max())
    maximum_capacities = np.full(record.node_count, maximum_capacity, dtype=np.int64)
    for node in np.flatnonzero(record.fixed_atom_mask):
        state = int(record.graph.node_states[node])
        if state >= len(atom_vocabulary):
            return None, "fixed_atom_state_outside_vocabulary"
        maximum_capacities[node] = atom_capacities[state]
    parents = np.zeros(count, dtype=np.int64)
    parent_bonds = np.zeros(count, dtype=np.int64)
    closure_left = np.zeros(closure_count, dtype=np.int64)
    closure_right = np.zeros(closure_count, dtype=np.int64)
    closure_bonds = np.zeros(closure_count, dtype=np.int64)
    minimum_used = np.zeros(count, dtype=np.int64)
    degrees = np.zeros(count, dtype=np.int64)
    occupied: set[tuple[int, int]] = set()
    component_instances = np.zeros(count, dtype=np.int64)
    for component_index, block in enumerate(record.component_blocks, start=1):
        component_instances[block.start : block.stop] = component_index
    core = record.core_position_states
    role_by_state = {block.role_state: block.role for block in record.component_blocks}
    try:
        role_names = tuple(role_by_state[int(value)] for value in record.role_states)
    except KeyError:
        return None, "unnamed_semantic_role"
    morphology_targets = (
        _exact_role_morphology_targets(record) if enforce_program_topology else None
    )

    # The qualified transform pins the hydrogen count, and therefore the exact heavy-atom
    # valence, of some reaction-core positions.  Reduce their capacity to that requirement so the
    # admissible sets below simply cannot place another neighbour there, and record the target so
    # under-saturation is caught too.  Precursor components meet only at the core, so a generated
    # edge stays inside one component block and a generated closure joins two exterior atoms.
    required_core_units: np.ndarray | None = None
    core_constrained = core_saturation is not None and core_saturation.applies_to(record)
    component_confined = (
        core_constrained and core_saturation.policy.component_confined_generated_edges
    )
    exterior_only_closures = (
        core_constrained and core_saturation.policy.exterior_only_generated_closures
    )
    if core_constrained:
        required_core_units = core_saturation.required_units(record)
        pinned = required_core_units >= 0
        if np.any(required_core_units[pinned] > maximum_capacities[pinned]):
            return None, "reaction_core_saturation_exceeds_atom_support"
        maximum_capacities[pinned] = required_core_units[pinned]

    # Reserve immutable adapter edges first so variable choices cannot consume their capacity.
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        child = int(child)
        if child == 0:
            return None, "fixed_root_parent"
        parent = int(record.graph.parents[child])
        bond = int(record.graph.parent_bonds[child])
        if parent < 0 or parent >= child or bond >= bond_classes:
            return None, "invalid_fixed_parent_edge"
        units = int(bond_units[bond])
        parents[child] = parent
        parent_bonds[child] = bond
        degrees[[child, parent]] += 1
        minimum_used[[child, parent]] += units
        occupied.add((parent, child))
    if np.any(minimum_used > maximum_capacities):
        return None, "fixed_parent_valence_exceeds_support"

    node_positions = np.arange(count, dtype=np.int64)
    parent_headroom = minimum_used + 2 <= maximum_capacities
    for child in range(1, count):
        if record.fixed_parent_bond_mask[child]:
            continue
        # The child's own headroom does not depend on the candidate parent, so the whole
        # admissible-parent row is one vectorized comparison rather than a Python scan over every
        # earlier node.  ``parent_headroom`` tracks ``minimum_used + 2 <= maximum_capacities``
        # incrementally; only the two endpoints of an accepted edge can change it.
        valid = np.zeros(count, dtype=np.bool_)
        if parent_headroom[child]:
            valid[:child] = parent_headroom[:child]
            # Core saturation confines generated tree edges to one component just as program
            # topology does, so both gates mask the admissible row the same way.
            if enforce_program_topology or component_confined:
                valid[:child] &= component_instances[:child] == component_instances[child]
        if enforce_program_topology:
            for parent in np.flatnonzero(valid).tolist():
                trial_parents = parents.copy()
                trial_parents[child] = parent
                observed = _terminal_role_morphology(
                    record,
                    parents=trial_parents,
                    closure_left=np.empty(0, dtype=np.int64),
                    closure_right=np.empty(0, dtype=np.int64),
                    parent_edge_mask=(record.fixed_parent_bond_mask | (node_positions <= child)),
                )
                assert morphology_targets is not None
                role_state = int(record.role_states[child])
                target = morphology_targets[role_state]
                # Counts and cycles are layout-level invariants.  During tree construction only
                # junction and core-attachment budgets can increase.
                if observed[role_state][1] > target[1] or observed[role_state][3] > target[3]:
                    valid[parent] = False
        parent = _argmax_allowed(predictions["parents"][index, child, :count], valid)
        if parent is None:
            if enforce_program_topology:
                reason = "program_topology_parent_unavailable"
            elif component_confined:
                reason = "reaction_core_component_parent_unavailable"
            else:
                reason = "parent_capacity_exhausted"
            return None, reason
        parents[child] = parent
        degrees[[child, parent]] += 1
        minimum_used[[child, parent]] += 2
        parent_headroom[[child, parent]] = (
            minimum_used[[child, parent]] + 2 <= maximum_capacities[[child, parent]]
        )
        occupied.add((parent, child))

    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        slot = int(slot)
        left = int(record.graph.closure_left[slot])
        right = int(record.graph.closure_right[slot])
        bond = int(record.graph.closure_bonds[slot])
        pair = tuple(sorted((left, right)))
        if left < 0 or right >= count or left == right or pair in occupied or bond >= bond_classes:
            return None, "invalid_fixed_closure"
        units = int(bond_units[bond])
        closure_left[slot] = left
        closure_right[slot] = right
        closure_bonds[slot] = bond
        degrees[[left, right]] += 1
        minimum_used[[left, right]] += units
        occupied.add(pair)
    if np.any(minimum_used > maximum_capacities):
        return None, "fixed_closure_valence_exceeds_support"

    topology_neighbors = [set() for _ in range(count)]
    for left, right in occupied:
        topology_neighbors[left].add(right)
        topology_neighbors[right].add(left)
    for slot in range(closure_count):
        if record.fixed_closure_bond_mask[slot]:
            continue
        pair_scores = (
            predictions["closure_left"][index, slot, :count, None]
            + predictions["closure_right"][index, slot, None, :count]
        )
        best: tuple[float, int, int] | None = None
        for left in range(count):
            for right in range(left + 1, count):
                if (left, right) in occupied:
                    continue
                semantic_pair = component_instances[left] == component_instances[right] or (
                    int(core[left]) > 1 and int(core[right]) > 1
                )
                if exterior_only_closures:
                    # A ring that reaches a reaction-core atom, or crosses two precursor
                    # components, cannot be cut back into that transform's precursors.
                    semantic_pair = (
                        component_instances[left] == component_instances[right]
                        and int(core[left]) == 1
                        and int(core[right]) == 1
                    )
                if enforce_program_topology:
                    role_state = int(record.role_states[left])
                    current = _terminal_role_morphology(
                        record,
                        parents=parents,
                        closure_left=closure_left[:slot],
                        closure_right=closure_right[:slot],
                    )
                    assert morphology_targets is not None
                    semantic_pair = (
                        component_instances[left] == component_instances[right]
                        and int(core[left]) == 1
                        and int(core[right]) == 1
                        and role_state == int(record.role_states[right])
                        and current[role_state][2] < morphology_targets[role_state][2]
                    )
                if (
                    not semantic_pair
                    or minimum_used[left] + 2 > maximum_capacities[left]
                    or minimum_used[right] + 2 > maximum_capacities[right]
                ):
                    continue
                if local_chemistry_support is not None:
                    if local_chemistry_support.enforces_role_cycles:
                        cycle = tree_path_indices(parents, left, right)
                        if not local_chemistry_support.allows_any_role_cycle(
                            record.program_id,
                            (role_names[node] for node in cycle),
                        ):
                            continue
                    elif any(
                        not local_chemistry_support.allows_any_role_triangle(
                            record.program_id,
                            (role_names[left], role_names[middle], role_names[right]),
                        )
                        for middle in topology_neighbors[left].intersection(
                            topology_neighbors[right]
                        )
                    ):
                        continue
                direct = float(pair_scores[left, right])
                reverse = float(pair_scores[right, left])
                candidate = (max(direct, reverse), left, right)
                if best is None or candidate > best:
                    best = candidate
                    if reverse > direct:
                        closure_left[slot], closure_right[slot] = right, left
                    else:
                        closure_left[slot], closure_right[slot] = left, right
        if best is None:
            return None, (
                "reaction_core_exterior_closure_unavailable"
                if exterior_only_closures and not enforce_program_topology
                else "closure_pair_unavailable"
            )
        _, left, right = best
        degrees[[left, right]] += 1
        minimum_used[[left, right]] += 2
        occupied.add((left, right))
        topology_neighbors[left].add(right)
        topology_neighbors[right].add(left)

    if enforce_program_topology:
        assert morphology_targets is not None
        observed_morphology = _terminal_role_morphology(
            record,
            parents=parents,
            closure_left=closure_left,
            closure_right=closure_right,
        )
        if observed_morphology != morphology_targets:
            return None, "program_morphology_exactness_failure"

    neighbors = [set() for _ in range(count)]
    for left, right in occupied:
        neighbors[left].add(right)
        neighbors[right].add(left)
    triangles: list[tuple[int, int, int]] = []
    for left in range(count):
        for middle in sorted(value for value in neighbors[left] if value > left):
            for right in sorted(
                value for value in neighbors[left].intersection(neighbors[middle]) if value > middle
            ):
                triangles.append((left, middle, right))
    generated_cycles = [
        tree_path_indices(parents, int(closure_left[slot]), int(closure_right[slot]))
        for slot in range(closure_count)
        if not record.fixed_closure_bond_mask[slot]
    ]

    node_states = np.zeros(count, dtype=np.int64)
    capacities = np.zeros(count, dtype=np.int64)
    assigned = np.zeros(count, dtype=np.bool_)
    block_by_node = {
        node: block for block in record.component_blocks for node in range(block.start, block.stop)
    }
    if local_chemistry_support is not None:
        for block in record.component_blocks:
            bounds = local_chemistry_support.component_support_bounds(record.program_id, block.role)
            if not bounds.heavy_atoms_min <= block.atom_count <= bounds.heavy_atoms_max:
                return None, "component_heavy_atoms_outside_observed_local_support"
    for node in range(count):
        valid = atom_capacities >= minimum_used[node]
        if local_chemistry_support is not None:
            block = block_by_node[node]
            bounds = local_chemistry_support.component_support_bounds(record.program_id, block.role)
            assigned_symbols = [
                atom_vocabulary[int(node_states[index])].symbol
                for index in range(block.start, node)
            ]
            assigned_carbons = assigned_symbols.count("C")
            assigned_heteroatoms = len(assigned_symbols) - assigned_carbons
            remaining_after_node = block.stop - node - 1
            for state, atom in enumerate(atom_vocabulary):
                if not valid[state]:
                    continue
                carbon_atoms = assigned_carbons + int(atom.symbol == "C")
                heteroatoms = assigned_heteroatoms + int(atom.symbol != "C")
                if (
                    carbon_atoms > bounds.carbon_atoms_max
                    or carbon_atoms + remaining_after_node < bounds.carbon_atoms_min
                    or heteroatoms > bounds.heteroatoms_max
                    or heteroatoms + remaining_after_node < bounds.heteroatoms_min
                ):
                    valid[state] = False
                    continue
                for neighbor in neighbors[node]:
                    if not assigned[neighbor]:
                        continue
                    neighbor_atom = atom_vocabulary[int(node_states[neighbor])]
                    if not local_chemistry_support.allows_role_edge_for_any_bond(
                        record.program_id,
                        role_names[node],
                        atom.symbol,
                        role_names[neighbor],
                        neighbor_atom.symbol,
                    ):
                        valid[state] = False
                        break
                if not valid[state]:
                    continue
                for triangle in triangles:
                    if node not in triangle:
                        continue
                    others = tuple(value for value in triangle if value != node)
                    if not all(assigned[value] for value in others):
                        continue
                    signature = (
                        (role_names[node], atom.symbol),
                        *(
                            (
                                role_names[value],
                                atom_vocabulary[int(node_states[value])].symbol,
                            )
                            for value in others
                        ),
                    )
                    if not local_chemistry_support.allows_role_triangle(
                        record.program_id, signature
                    ):
                        valid[state] = False
                        break
                if not valid[state] or not local_chemistry_support.enforces_role_cycles:
                    continue
                for cycle in generated_cycles:
                    if node not in cycle:
                        continue
                    others = tuple(value for value in cycle if value != node)
                    if not all(assigned[value] for value in others):
                        continue
                    signature = (
                        (role_names[node], atom.symbol),
                        *(
                            (
                                role_names[value],
                                atom_vocabulary[int(node_states[value])].symbol,
                            )
                            for value in others
                        ),
                    )
                    if not local_chemistry_support.allows_role_cycle(record.program_id, signature):
                        valid[state] = False
                        break
        if record.fixed_atom_mask[node]:
            state = int(record.graph.node_states[node])
            if state >= len(atom_vocabulary) or not valid[state]:
                reason = (
                    "fixed_atom_local_chemistry_exceeds_support"
                    if local_chemistry_support is not None
                    else "fixed_atom_valence_exceeds_support"
                )
                return None, reason
        else:
            selected = _argmax_allowed(predictions["nodes"][index, node], valid)
            if selected is None:
                reason = (
                    "atom_local_chemistry_state_unavailable"
                    if local_chemistry_support is not None
                    else "atom_valence_state_unavailable"
                )
                return None, reason
            state = selected
        node_states[node] = state
        capacities[node] = atom_capacities[state]
        assigned[node] = True

    if local_chemistry_support is not None:
        for block in record.component_blocks:
            symbols = [
                atom_vocabulary[int(node_states[node])].symbol
                for node in range(block.start, block.stop)
            ]
            carbon_atoms = symbols.count("C")
            heavy_atoms = len(symbols)
            if not local_chemistry_support.component_is_within_observed_support(
                record.program_id,
                block.role,
                heavy_atoms=heavy_atoms,
                carbon_atoms=carbon_atoms,
                heteroatoms=heavy_atoms - carbon_atoms,
            ):
                return None, "component_outside_observed_local_support"

    if required_core_units is not None:
        # Bond-order selection below must not spend a pinned core position's stated valence on a
        # higher bond order either, so the realized capacities carry the same ceiling.
        capacities = np.minimum(capacities, maximum_capacities)
    used = minimum_used.copy()
    variable_edges: list[tuple[str, int, int, int]] = []
    for child in range(1, count):
        if not record.fixed_parent_bond_mask[child]:
            variable_edges.append(("parent", child, int(parents[child]), child))
    for slot in range(closure_count):
        if not record.fixed_closure_bond_mask[slot]:
            variable_edges.append(
                ("closure", slot, int(closure_left[slot]), int(closure_right[slot]))
            )
    for kind, slot, left, right in variable_edges:
        spare = min(int(capacities[left] - used[left]), int(capacities[right] - used[right]))
        valid = bond_units <= 2 + spare
        for bond, units in enumerate(bond_units):
            if units == 3 and not (
                atom_vocabulary[node_states[left]].aromatic
                and atom_vocabulary[node_states[right]].aromatic
            ):
                valid[bond] = False
        if local_chemistry_support is not None:
            left_atom = atom_vocabulary[int(node_states[left])]
            right_atom = atom_vocabulary[int(node_states[right])]
            for bond in range(len(valid)):
                if valid[bond] and not local_chemistry_support.allows_role_edge(
                    record.program_id,
                    role_names[left],
                    left_atom.symbol,
                    bond,
                    role_names[right],
                    right_atom.symbol,
                ):
                    valid[bond] = False
        logits = predictions["parent_bonds" if kind == "parent" else "closure_bonds"][index, slot]
        bond = _argmax_allowed(logits, valid)
        if bond is None:
            qualifier = "local_chemistry" if local_chemistry_support is not None else "valence"
            return None, f"{kind}_bond_{qualifier}_unavailable"
        extra = int(bond_units[bond]) - 2
        used[[left, right]] += extra
        if kind == "parent":
            parent_bonds[slot] = bond
        else:
            closure_bonds[slot] = bond
    if np.any(used > capacities):
        return None, "terminal_valence_overflow"
    if required_core_units is not None:
        pinned = required_core_units >= 0
        if np.any(used[pinned] != required_core_units[pinned]):
            # Masking makes over-substitution unreachable; this catches under-substitution, which
            # would leave the core atom with an extra hydrogen and equally break the transform.
            return None, "reaction_core_saturation_unmet"
    if local_chemistry_support is not None:
        symbols = tuple(atom_vocabulary[int(state)].symbol for state in node_states)
        terminal_edges = [
            (int(parents[child]), child, int(parent_bonds[child])) for child in range(1, count)
        ]
        terminal_edges.extend(
            (int(closure_left[slot]), int(closure_right[slot]), int(closure_bonds[slot]))
            for slot in range(closure_count)
        )
        if any(
            not local_chemistry_support.allows_role_edge(
                record.program_id,
                role_names[left],
                symbols[left],
                bond,
                role_names[right],
                symbols[right],
            )
            for left, right, bond in terminal_edges
        ):
            return None, "terminal_edge_outside_observed_local_support"
        if any(
            not local_chemistry_support.allows_role_triangle(
                record.program_id,
                ((role_names[index], symbols[index]) for index in triangle),
            )
            for triangle in triangles
        ):
            return None, "terminal_triangle_outside_observed_local_support"
        if local_chemistry_support.enforces_role_cycles and any(
            not local_chemistry_support.allows_role_cycle(
                record.program_id,
                ((role_names[index], symbols[index]) for index in cycle),
            )
            for cycle in generated_cycles
        ):
            return None, "terminal_cycle_outside_observed_role_morphology_support"
    return {
        "nodes": node_states,
        "parents": parents,
        "parent_bonds": parent_bonds,
        "closure_left": closure_left,
        "closure_right": closure_right,
        "closure_bonds": closure_bonds,
    }, None


def decode_synthesis_program_strict_argmax(
    predictions: Mapping[str, Any],
    layout: Mapping[str, Any],
    records: Sequence[SynthesisProgramGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    local_chemistry_support: LocalChemistrySupport | None = None,
    *,
    enforce_program_topology: bool = False,
    core_saturation: BoundReactionCoreSaturation | None = None,
) -> tuple[dict[str, Any], tuple[str | None, ...]]:
    """Decode once under strict support; infeasible attempts abstain and are never repaired."""

    if len(records) != int(layout["node_mask"].shape[0]):
        raise SynthesisProgramSamplingError("strict decoder batch and record counts disagree")
    cpu_predictions = {
        key: value.detach().to("cpu").numpy()
        for key, value in predictions.items()
        if key
        in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        }
    }
    # Strict decoding is an RDKit/NumPy terminal operation.  Keep its output on CPU: the previous
    # implementation copied predictions GPU->CPU, then performed thousands of tiny decoded-state
    # copies CPU->GPU only for the caller to immediately copy every state GPU->CPU again.
    cpu_layout = {
        key: value.detach().to("cpu")
        for key, value in layout.items()
        if key
        in {
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
            "fixed_atom_mask",
            "fixed_parent_mask",
            "fixed_parent_bond_mask",
            "fixed_closure_endpoint_mask",
            "fixed_closure_bond_mask",
        }
    }
    terminal = {
        field: torch.zeros_like(cpu_layout[field])
        for field in (
            "nodes",
            "parents",
            "parent_bonds",
            "closure_left",
            "closure_right",
            "closure_bonds",
        )
    }
    terminal = restore_synthesis_program_fixed_states(terminal, cpu_layout)
    reasons: list[str | None] = []
    for index, record in enumerate(records):
        decoded, reason = _strict_terminal_record(
            cpu_predictions,
            index,
            record,
            atom_vocabulary,
            local_chemistry_support,
            enforce_program_topology=enforce_program_topology,
            core_saturation=core_saturation,
        )
        reasons.append(reason)
        if decoded is None:
            continue
        for field, values in decoded.items():
            terminal[field][index, : len(values)] = torch.as_tensor(
                values, dtype=terminal[field].dtype
            )
    return restore_synthesis_program_fixed_states(terminal, cpu_layout), tuple(reasons)


def sample_synthesis_program_products(
    model: Any,
    records: Sequence[SynthesisProgramGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    *,
    samples_per_program: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
    conditioning_mode: str = "program",
    program_state_mapping: Mapping[int, int] | None = None,
    role_state_mapping: Mapping[int, int] | None = None,
    terminal_decode_policy: str = "unconstrained_argmax",
    local_chemistry_support: LocalChemistrySupport | None = None,
    ugi_topology_policy: UgiTransformerTopologyPolicy | None = None,
    reaction_core_saturation_policy: ReactionCoreSaturationPolicy | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate from semantic layouts while exposing only Ugi adapter-fixed graph states."""

    if (
        torch is None
        or not records
        or samples_per_program < 1
        or sample_steps < 2
        or batch_size < 1
        or terminal_decode_policy not in SUPPORTED_TERMINAL_DECODE_POLICIES
    ):
        raise SynthesisProgramSamplingError("invalid shared synthesis-program sampling request")
    if (terminal_decode_policy == LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY) != (
        local_chemistry_support is not None
    ):
        raise SynthesisProgramSamplingError(
            "strict local-chemistry decoding and its support policy must be supplied together"
        )
    coupled_ugi_topology = terminal_decode_policy == COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY
    if coupled_ugi_topology != (ugi_topology_policy is not None):
        raise SynthesisProgramSamplingError(
            "coupled Ugi topology decoding and its explicit support policy must be supplied together"
        )
    if (terminal_decode_policy == CORE_SATURATION_TERMINAL_DECODE_POLICY) != (
        reaction_core_saturation_policy is not None
    ):
        raise SynthesisProgramSamplingError(
            "strict reaction-core saturation decoding and its registry contract must be supplied "
            "together"
        )
    core_saturation: BoundReactionCoreSaturation | None = None
    if reaction_core_saturation_policy is not None:
        vocabulary = getattr(model, "vocabulary", None)
        core_position_states = getattr(vocabulary, "core_position_states", None)
        if core_position_states is None:
            raise SynthesisProgramSamplingError(
                "reaction-core saturation decoding requires a model that declares its "
                "core-position vocabulary"
            )
        core_saturation = reaction_core_saturation_policy.bind(core_position_states)
    resolved_device = torch.device(device)
    repeated = tuple(record for record in records for _ in range(samples_per_program))
    maximum_closures = int(model.maximum_closures)
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=resolved_device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=resolved_device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    topology_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    outputs: list[dict[str, Any]] = []
    fixed_failures = 0
    strict_abstentions: Counter[str] = Counter()
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(repeated), batch_size):
            local = repeated[offset : offset + batch_size]
            cpu_layout = collate_synthesis_program_layouts(
                local,
                maximum_closures=maximum_closures,
                conditioning_mode=conditioning_mode,
                program_state_mapping=program_state_mapping,
                role_state_mapping=role_state_mapping,
            )
            has_variable_parents = bool(cpu_layout["parent_variable_mask"].any())
            has_variable_closure_endpoints = bool(
                cpu_layout["closure_endpoint_variable_mask"].any()
            )
            layout = _move(cpu_layout, resolved_device)
            node_source, parent_bond_source, closure_bond_source = (
                resolve_synthesis_program_source_marginals(layout, node_p0, bond_p0)
            )
            state = _initial_state(layout, node_p0, bond_p0, generator)
            fixed_failure_checks = [~_fixed_state_exact_tensor(state, layout)]
            parent_candidates = _parent_candidate_mask(layout["node_mask"])
            endpoint_candidates = _endpoint_candidate_mask(layout["node_mask"], maximum_closures)
            time_grid = torch.arange(
                sample_steps, dtype=torch.float32, device=resolved_device
            ) / float(sample_steps)
            terminal_time = torch.ones((len(local),), device=resolved_device)
            conditioning = _program_conditioning(model, layout)
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = time_grid[step].expand(len(local))
                predictions = model(
                    nodes=state["nodes"],
                    parents=state["parents"],
                    parent_bonds=state["parent_bonds"],
                    closure_left=state["closure_left"],
                    closure_right=state["closure_right"],
                    closure_bonds=state["closure_bonds"],
                    t=t,
                    **conditioning,
                )
                state["nodes"] = rstar_step(
                    state["nodes"],
                    predictions["nodes"].softmax(dim=-1),
                    node_source,
                    t_value,
                    1.0 / sample_steps,
                    layout["atom_variable_mask"],
                    generator,
                )
                if has_variable_parents:
                    state["parents"] = pointer_rstar_step(
                        state["parents"],
                        predictions["parents"],
                        parent_candidates,
                        layout["parent_variable_mask"],
                        t_value,
                        1.0 / sample_steps,
                        generator,
                    )
                for field, mask_name, source in (
                    ("parent_bonds", "parent_bond_variable_mask", parent_bond_source),
                    ("closure_bonds", "closure_bond_variable_mask", closure_bond_source),
                ):
                    state[field] = rstar_step(
                        state[field],
                        predictions[field].softmax(dim=-1),
                        source,
                        t_value,
                        1.0 / sample_steps,
                        layout[mask_name],
                        generator,
                    )
                if has_variable_closure_endpoints:
                    for field in ("closure_left", "closure_right"):
                        state[field] = pointer_rstar_step(
                            state[field],
                            predictions[field],
                            endpoint_candidates,
                            layout["closure_endpoint_variable_mask"],
                            t_value,
                            1.0 / sample_steps,
                            generator,
                        )
                state = _restore_fixed_states_in_place(state, layout)
                fixed_failure_checks.append(~_fixed_state_exact_tensor(state, layout))
            terminal_predictions = model(
                nodes=state["nodes"],
                parents=state["parents"],
                parent_bonds=state["parent_bonds"],
                closure_left=state["closure_left"],
                closure_right=state["closure_right"],
                closure_bonds=state["closure_bonds"],
                t=terminal_time,
                **conditioning,
            )
            topology_reasons: list[str | None] = [None] * len(local)
            if coupled_ugi_topology:
                assert ugi_topology_policy is not None
                topology_state = {key: value.clone() for key, value in state.items()}
                for index, record in enumerate(local):
                    try:
                        decoded_topology = decode_ugi_exact_topology(
                            terminal_predictions,
                            index=index,
                            record=record,
                            policy=ugi_topology_policy,
                            generator=topology_generator,
                        )
                    except UgiTransformerTopologyError as error:
                        topology_reasons[index] = f"ugi_topology_coupling_failure:{error}"
                        continue
                    count = record.node_count
                    closure_count = record.graph.closure_count
                    topology_state["parents"][index, :count] = torch.as_tensor(
                        decoded_topology.parents,
                        dtype=topology_state["parents"].dtype,
                        device=resolved_device,
                    )
                    if closure_count:
                        topology_state["closure_left"][index, :closure_count] = torch.as_tensor(
                            decoded_topology.closure_left,
                            dtype=topology_state["closure_left"].dtype,
                            device=resolved_device,
                        )
                        topology_state["closure_right"][index, :closure_count] = torch.as_tensor(
                            decoded_topology.closure_right,
                            dtype=topology_state["closure_right"].dtype,
                            device=resolved_device,
                        )
                topology_state = _restore_fixed_states_in_place(topology_state, layout)
                # Chemistry is predicted after, and therefore conditional on, the exact tree and
                # feasible closure endpoints.  This is the factorization that made the native Ugi
                # model reliable; the Transformer remains the shared denoiser.
                terminal_predictions = model(
                    nodes=topology_state["nodes"],
                    parents=topology_state["parents"],
                    parent_bonds=topology_state["parent_bonds"],
                    closure_left=topology_state["closure_left"],
                    closure_right=topology_state["closure_right"],
                    closure_bonds=topology_state["closure_bonds"],
                    t=terminal_time,
                    **conditioning,
                )
                # The exact topology is already decoded.  One-hot pointer scores let the common
                # strict chemistry decoder preserve it while retaining all valence checks.
                pointer_predictions = dict(terminal_predictions)
                parent_logits = torch.full_like(pointer_predictions["parents"], -1e9)
                parent_logits.scatter_(-1, topology_state["parents"].unsqueeze(-1), 1e9)
                pointer_predictions["parents"] = parent_logits
                for field in ("closure_left", "closure_right"):
                    endpoint_logits = torch.full_like(pointer_predictions[field], -1e9)
                    endpoint_logits.scatter_(-1, topology_state[field].unsqueeze(-1), 1e9)
                    pointer_predictions[field] = endpoint_logits
                terminal_predictions = pointer_predictions
            if terminal_decode_policy in STRICT_TERMINAL_DECODE_POLICIES:
                terminal, abstention_reasons = decode_synthesis_program_strict_argmax(
                    terminal_predictions,
                    layout,
                    local,
                    atom_vocabulary,
                    local_chemistry_support,
                    core_saturation=core_saturation,
                    enforce_program_topology=(
                        terminal_decode_policy
                        in {
                            PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
                            COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
                        }
                    ),
                )
                if coupled_ugi_topology:
                    abstention_reasons = tuple(
                        topology_reason if topology_reason is not None else chemistry_reason
                        for topology_reason, chemistry_reason in zip(
                            topology_reasons, abstention_reasons, strict=True
                        )
                    )
            else:
                terminal = decode_synthesis_program_argmax(terminal_predictions, layout)
                abstention_reasons = (None,) * len(local)
            if terminal_decode_policy in STRICT_TERMINAL_DECODE_POLICIES:
                fixed_failures += int(not _fixed_state_exact_records(terminal, local))
            else:
                fixed_failure_checks.append(~_fixed_state_exact_tensor(terminal, layout))
            fixed_failures += int(torch.stack(fixed_failure_checks).sum().item())
            terminal_cpu = {field: values.detach().to("cpu") for field, values in terminal.items()}
            for index, record in enumerate(local):
                count = record.node_count
                closure_count = record.graph.closure_count
                abstention_reason = abstention_reasons[index]
                if abstention_reason is not None:
                    strict_abstentions[abstention_reason] += 1
                    smiles = None
                else:
                    smiles = _terminal_smiles(
                        terminal_cpu,
                        index,
                        count,
                        closure_count,
                        atom_vocabulary,
                    )
                exact_fields = {
                    "nodes": np.array_equal(
                        terminal_cpu["nodes"][index, :count].numpy(),
                        record.graph.node_states,
                    ),
                    "parents": np.array_equal(
                        terminal_cpu["parents"][index, :count].numpy(), record.graph.parents
                    ),
                    "parent_bonds": np.array_equal(
                        terminal_cpu["parent_bonds"][index, :count].numpy(),
                        record.graph.parent_bonds,
                    ),
                    "closure_left": np.array_equal(
                        terminal_cpu["closure_left"][index, :closure_count].numpy(),
                        record.graph.closure_left,
                    ),
                    "closure_right": np.array_equal(
                        terminal_cpu["closure_right"][index, :closure_count].numpy(),
                        record.graph.closure_right,
                    ),
                    "closure_bonds": np.array_equal(
                        terminal_cpu["closure_bonds"][index, :closure_count].numpy(),
                        record.graph.closure_bonds,
                    ),
                }
                target_values = {
                    "nodes": record.graph.node_states,
                    "parents": record.graph.parents,
                    "parent_bonds": record.graph.parent_bonds,
                    "closure_left": record.graph.closure_left,
                    "closure_right": record.graph.closure_right,
                    "closure_bonds": record.graph.closure_bonds,
                }
                mismatch_positions = {}
                for field, exact in exact_fields.items():
                    if exact:
                        continue
                    size = closure_count if field.startswith("closure") else count
                    observed = terminal_cpu[field][index, :size].numpy()
                    mismatch_positions[field] = np.flatnonzero(
                        observed != target_values[field]
                    ).tolist()
                outputs.append(
                    {
                        "sample_index": offset + index,
                        "program_id": record.program_id,
                        "layout_record_id": record.graph.structure_id,
                        "canonical_smiles": smiles,
                        "valid": smiles is not None,
                        "constraint_abstention_reason": abstention_reason,
                        "local_chemistry_policy_applied": local_chemistry_support is not None,
                        "program_topology_policy_applied": (
                            terminal_decode_policy
                            in {
                                PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
                                COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
                            }
                        ),
                        "topology_coupling_second_pass_applied": coupled_ugi_topology,
                        "reaction_core_saturation_policy_applied": (
                            core_saturation is not None and core_saturation.applies_to(record)
                        ),
                        "exact_target_graph": smiles == record.graph.canonical_smiles,
                        "exact_tensor": all(exact_fields.values()),
                        "exact_fields": exact_fields,
                        "mismatch_positions": mismatch_positions,
                    }
                )
    by_program: dict[str, dict[str, int]] = {}
    for program_id in sorted({record.program_id for record in records}):
        program_rows = [row for row in outputs if row["program_id"] == program_id]
        by_program[program_id] = {
            "samples": len(program_rows),
            "valid": sum(bool(row["valid"]) for row in program_rows),
            "exact_target_graph": sum(bool(row["exact_target_graph"]) for row in program_rows),
            "exact_tensor": sum(bool(row["exact_tensor"]) for row in program_rows),
            "strict_constraint_abstentions": sum(
                row["constraint_abstention_reason"] is not None for row in program_rows
            ),
        }
    return outputs, {
        "samples": len(outputs),
        "valid": sum(bool(row["valid"]) for row in outputs),
        "exact_target_graph": sum(bool(row["exact_target_graph"]) for row in outputs),
        "exact_tensor": sum(bool(row["exact_tensor"]) for row in outputs),
        "fixed_state_failures": fixed_failures,
        "terminal_decode_policy": terminal_decode_policy,
        "strict_constraint_abstentions": sum(strict_abstentions.values()),
        "strict_constraint_abstention_reasons": dict(sorted(strict_abstentions.items())),
        "local_chemistry_policy_applied": local_chemistry_support is not None,
        "program_topology_policy_applied": (
            terminal_decode_policy
            in {
                PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
                COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY,
            }
        ),
        "topology_coupling_second_pass_applied": coupled_ugi_topology,
        "topology_selection": (
            "exact_program_conditional_sample_then_argmax_chemistry"
            if coupled_ugi_topology
            else None
        ),
        "topology_seed": seed + 1 if coupled_ugi_topology else None,
        "ugi_topology_policy": (
            None if ugi_topology_policy is None else ugi_topology_policy.to_mapping()
        ),
        "reaction_core_saturation_policy_applied": reaction_core_saturation_policy is not None,
        "reaction_core_saturation_policy": (
            None
            if reaction_core_saturation_policy is None
            else reaction_core_saturation_policy.to_mapping()
        ),
        "by_program": by_program,
        "repairs": dict(Counter()),
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "CORE_SATURATION_TERMINAL_DECODE_POLICY",
    "COUPLED_UGI_TOPOLOGY_TERMINAL_DECODE_POLICY",
    "LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY",
    "PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY",
    "STRICT_TERMINAL_DECODE_POLICIES",
    "SUPPORTED_TERMINAL_DECODE_POLICIES",
    "TERMINAL_DECODE_POLICIES",
    "SynthesisProgramSamplingError",
    "load_synthesis_program_checkpoint",
    "decode_synthesis_program_strict_argmax",
    "sample_synthesis_program_products",
    "synthesis_program_source_marginals",
]
