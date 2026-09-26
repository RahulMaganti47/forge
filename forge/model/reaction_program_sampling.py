"""Sampling utilities for program-conditioned whole-product sparse flow."""

from __future__ import annotations

import base64
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

from forge.assembly import ReactionProgramAdapter, ReactionProgramError, ReactionProgramSpec
from forge.flow import rstar_step
from forge.model.defog_feasibility import AtomState, _model_state_sha256, graph_to_molecule
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import ReactionProgramSparseFlow
from forge.model.reaction_program_graph import ReactionProgramGraphRecord
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    INDEX_TO_DENSE_BOND,
    _endpoint_candidate_mask,
    _maximum_valence_units,
    _parent_candidate_mask,
    _sample_from_logits,
    pointer_rstar_step,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]


class ReactionProgramSamplingError(ValueError):
    """A sampling request violates the program or graph support."""


@dataclass(frozen=True)
class ReactionProgramLayout:
    """Coarse semantic program with counts only and no stored molecular fragment."""

    program_id: str
    program_state: int
    program_depth: int
    accumulator_role_state: int
    repeat_role_state: int
    accumulator_atom_count: int
    repeat_atom_count: int
    closure_count: int
    core_position_word: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.core_position_word and len(self.core_position_word) != self.node_count:
            raise ReactionProgramSamplingError(
                "core-position word must align with the complete program layout"
            )

    @property
    def node_count(self) -> int:
        return self.accumulator_atom_count + self.program_depth * self.repeat_atom_count

    @property
    def role_states(self) -> tuple[int, ...]:
        return (
            *(self.accumulator_role_state for _ in range(self.accumulator_atom_count)),
            *(self.repeat_role_state for _ in range(self.program_depth * self.repeat_atom_count)),
        )

    @property
    def core_position_states(self) -> tuple[int, ...]:
        # Index one is the stable ``exterior`` state in ReactionProgramVocabulary.
        return self.core_position_word or (1,) * self.node_count


def load_reaction_program_checkpoint(
    checkpoint_path: Path,
    *,
    device: str,
) -> tuple[
    Any, ReactionProgramVocabulary, tuple[AtomState, ...], np.ndarray, np.ndarray, dict[str, Any]
]:
    """Load a trusted local checkpoint into the reusable sampling interface."""

    if torch is None:
        raise ReactionProgramSamplingError("checkpoint loading requires torch")
    try:
        package = json.loads(checkpoint_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReactionProgramSamplingError(
            f"reaction-program checkpoint could not be loaded: {checkpoint_path}"
        ) from error
    if (
        not isinstance(package, dict)
        or package.get("schema_version") != "forge.multireaction_sparse_flow_checkpoint.v2"
        or package.get("trusted_local_checkpoint") is not True
    ):
        raise ReactionProgramSamplingError("checkpoint is not a trusted multi-reaction checkpoint")
    raw_vocabulary = package.get("program_vocabulary")
    raw_atoms = package.get("atom_vocabulary")
    model_config = package.get("model_config")
    if (
        not isinstance(raw_vocabulary, dict)
        or not isinstance(raw_atoms, list)
        or not isinstance(model_config, dict)
    ):
        raise ReactionProgramSamplingError("checkpoint is missing its model vocabulary")
    vocabulary = ReactionProgramVocabulary(
        program_states=tuple(str(value) for value in raw_vocabulary["program_states"]),
        role_states=tuple(str(value) for value in raw_vocabulary["role_states"]),
        maximum_steps=int(raw_vocabulary["maximum_steps"]),
        core_position_states=tuple(str(value) for value in raw_vocabulary["core_position_states"]),
    )
    if (
        not vocabulary.program_states
        or vocabulary.program_states[0] != "unconditioned"
        or not vocabulary.role_states
        or vocabulary.role_states[0] != "unassigned"
    ):
        raise ReactionProgramSamplingError("checkpoint semantic null states changed")
    atom_vocabulary = tuple(
        AtomState(
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
            int(row["explicit_hydrogens"]),
        )
        for row in raw_atoms
    )
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ReactionProgramSamplingError("CUDA checkpoint loading requested but unavailable")
    if resolved_device.type == "mps" and not torch.backends.mps.is_available():
        raise ReactionProgramSamplingError("MPS checkpoint loading requested but unavailable")
    model = ReactionProgramSparseFlow(
        vocabulary=vocabulary,
        node_classes=len(atom_vocabulary),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        maximum_closures=int(model_config["maximum_closures"]),
        maximum_heavy_atoms=int(model_config["maximum_heavy_atoms"]),
        dropout=float(model_config["dropout"]),
        bond_classes=int(model_config["bond_classes"]),
    )
    raw_state = package.get("model_state")
    if not isinstance(raw_state, dict):
        raise ReactionProgramSamplingError("checkpoint has no deterministic tensor state")
    state: dict[str, Any] = {}
    try:
        for key, record in raw_state.items():
            if not isinstance(record, dict) or set(record) != {"dtype", "shape", "data_base64"}:
                raise ValueError(f"malformed tensor record: {key}")
            dtype = np.dtype(str(record["dtype"]))
            if dtype.kind not in {"b", "f", "i", "u"}:
                raise ValueError(f"unsupported tensor dtype: {dtype}")
            shape = tuple(int(value) for value in record["shape"])
            payload = base64.b64decode(str(record["data_base64"]), validate=True)
            array = np.frombuffer(payload, dtype=dtype)
            if array.size != int(np.prod(shape, dtype=np.int64)):
                raise ValueError(f"tensor shape differs from payload: {key}")
            state[str(key)] = torch.from_numpy(array.reshape(shape).copy())
    except (KeyError, TypeError, ValueError) as error:
        raise ReactionProgramSamplingError("checkpoint tensor state is malformed") from error
    model.load_state_dict(state, strict=True)
    model.to(resolved_device)
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise ReactionProgramSamplingError("checkpoint model-state hash mismatch")
    node_marginal = np.asarray(package.get("node_marginal"), dtype=np.float64)
    bond_marginal = np.asarray(package.get("bond_marginal"), dtype=np.float64)
    if (
        node_marginal.shape != (len(atom_vocabulary),)
        or bond_marginal.shape != (int(model_config["bond_classes"]),)
        or not np.isclose(node_marginal.sum(), 1.0)
        or not np.isclose(bond_marginal.sum(), 1.0)
        or np.any(node_marginal <= 0)
        or np.any(bond_marginal <= 0)
    ):
        raise ReactionProgramSamplingError("checkpoint source marginals are invalid")
    model.eval()
    return model, vocabulary, atom_vocabulary, node_marginal, bond_marginal, package


def _weighted_choice(
    values: Sequence[int],
    weights: np.ndarray,
    rng: np.random.Generator,
) -> int:
    grouped: dict[int, float] = {}
    for value, weight in zip(values, weights, strict=True):
        grouped[int(value)] = grouped.get(int(value), 0.0) + float(weight)
    support = np.asarray(sorted(grouped), dtype=np.int64)
    probabilities = np.asarray([grouped[int(value)] for value in support], dtype=np.float64)
    probabilities /= probabilities.sum()
    return int(rng.choice(support, p=probabilities))


def _factorized_core_position_word(
    records: Sequence[ReactionProgramGraphRecord],
    weights: np.ndarray,
    layout: ReactionProgramLayout,
    vocabulary: ReactionProgramVocabulary,
    rng: np.random.Generator,
) -> tuple[int, ...]:
    """Draw adapter semantics without selecting an atom, bond, component, or stored fragment."""

    same_depth = tuple(
        (record, float(weight))
        for record, weight in zip(records, weights, strict=True)
        if record.program_depth == layout.program_depth
    )
    if not same_depth:
        raise ReactionProgramSamplingError("sampled depth has no core-position supervision")
    accumulator_state = layout.accumulator_role_state
    repeat_state = layout.repeat_role_state
    counts_by_role: dict[int, Counter[int]] = {}
    for role_state in (accumulator_state, repeat_state):
        observed: dict[tuple[tuple[int, int], ...], float] = {}
        for record, weight in same_depth:
            counter = Counter(
                int(core)
                for role, core in zip(
                    record.role_states,
                    record.core_position_states,
                    strict=True,
                )
                if int(role) == role_state and int(core) > 1
            )
            pattern = tuple(sorted(counter.items()))
            observed[pattern] = observed.get(pattern, 0.0) + weight
        patterns = tuple(sorted(observed))
        probabilities = np.asarray([observed[value] for value in patterns], dtype=np.float64)
        probabilities /= probabilities.sum()
        selected = patterns[int(rng.choice(len(patterns), p=probabilities))]
        counts_by_role[role_state] = Counter(dict(selected))

    word = np.ones(layout.node_count, dtype=np.int64)
    accumulator_counts = counts_by_role[accumulator_state]
    accumulator_labels = [
        state for state, count in sorted(accumulator_counts.items()) for _ in range(count)
    ]
    if len(accumulator_labels) > layout.accumulator_atom_count:
        raise ReactionProgramSamplingError("accumulator count cannot hold its reaction core")
    if accumulator_labels:
        positions = rng.choice(
            layout.accumulator_atom_count,
            size=len(accumulator_labels),
            replace=False,
        )
        for position, state in zip(sorted(positions.tolist()), accumulator_labels, strict=True):
            word[position] = state

    repeat_counts = counts_by_role[repeat_state]
    per_repeat: list[int] = []
    for state, count in sorted(repeat_counts.items()):
        if count % layout.program_depth:
            raise ReactionProgramSamplingError(
                "repeat-role core multiplicity is not divisible by program depth"
            )
        per_repeat.extend(state for _ in range(count // layout.program_depth))
    if len(per_repeat) > layout.repeat_atom_count:
        raise ReactionProgramSamplingError("repeat count cannot hold its reaction core")
    for step in range(layout.program_depth):
        start = layout.accumulator_atom_count + step * layout.repeat_atom_count
        if per_repeat:
            positions = rng.choice(
                layout.repeat_atom_count,
                size=len(per_repeat),
                replace=False,
            )
            for position, state in zip(sorted(positions.tolist()), per_repeat, strict=True):
                word[start + position] = state
    if any(value >= len(vocabulary.core_position_states) for value in word):
        raise ReactionProgramSamplingError("sampled core position lies outside the vocabulary")
    return tuple(int(value) for value in word)


def sample_factorized_program_layouts(
    records: Sequence[ReactionProgramGraphRecord],
    weights: np.ndarray,
    specifications: Sequence[ReactionProgramSpec],
    vocabulary: ReactionProgramVocabulary,
    *,
    sample_count: int,
    seed: int,
) -> tuple[ReactionProgramLayout, ...]:
    """Sample independent coarse count fields under a uniform program prior."""

    if sample_count < 1 or len(records) != len(weights):
        raise ReactionProgramSamplingError("layout sampling requires aligned records and weights")
    if np.any(weights < 0) or not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ReactionProgramSamplingError("layout weights are not a finite positive measure")
    spec_by_program = {spec.program_id: spec for spec in specifications}
    records_by_program = {
        program_id: tuple(record for record in records if record.program_id == program_id)
        for program_id in sorted(spec_by_program)
    }
    indices_by_program = {
        program_id: np.asarray(
            [index for index, record in enumerate(records) if record.program_id == program_id],
            dtype=np.int64,
        )
        for program_id in records_by_program
    }
    if any(not values for values in records_by_program.values()):
        raise ReactionProgramSamplingError("every declared program needs training support")
    rng = np.random.default_rng(seed)
    program_ids = tuple(sorted(records_by_program))
    layouts: list[ReactionProgramLayout] = []
    for _ in range(sample_count):
        program_id = str(rng.choice(program_ids))
        local = records_by_program[program_id]
        local_weights = weights[indices_by_program[program_id]].astype(np.float64)
        local_weights /= local_weights.sum()
        repeat_sizes = []
        for record in local:
            if len(set(record.repeat_atom_counts)) != 1:
                raise ReactionProgramSamplingError(
                    "identical-repeat program contains unequal repeated-component atom counts"
                )
            repeat_sizes.append(record.repeat_atom_counts[0])
        spec = spec_by_program[program_id]
        depth = _weighted_choice([record.program_depth for record in local], local_weights, rng)
        if not spec.minimum_steps <= depth <= spec.maximum_steps:
            raise ReactionProgramSamplingError("sampled program depth lies outside adapter support")
        layout = ReactionProgramLayout(
            program_id=program_id,
            program_state=vocabulary.program_to_index[program_id],
            program_depth=depth,
            accumulator_role_state=vocabulary.role_to_index[spec.accumulator_role],
            repeat_role_state=vocabulary.role_to_index[spec.repeat_role],
            accumulator_atom_count=_weighted_choice(
                [record.accumulator_atom_count for record in local], local_weights, rng
            ),
            repeat_atom_count=_weighted_choice(repeat_sizes, local_weights, rng),
            closure_count=_weighted_choice(
                [record.graph.closure_count for record in local], local_weights, rng
            ),
        )
        layouts.append(
            ReactionProgramLayout(
                **{
                    **layout.__dict__,
                    "core_position_word": _factorized_core_position_word(
                        local, local_weights, layout, vocabulary, rng
                    ),
                }
            )
        )
    return tuple(layouts)


def sample_training_semantic_layouts(
    records: Sequence[ReactionProgramGraphRecord],
    vocabulary: ReactionProgramVocabulary,
    *,
    sample_count: int,
    seed: int,
) -> tuple[ReactionProgramLayout, ...]:
    """Draw only coarse semantic words from a balanced set of training records.

    This overfit diagnostic deliberately reuses node counts, role states, core-position states and
    closure counts. It never reads atom states, bonds, component SMILES or component identifiers.
    """

    if not records or sample_count < 1:
        raise ReactionProgramSamplingError("semantic-layout sampling requires records")
    by_program = {
        program_id: tuple(record for record in records if record.program_id == program_id)
        for program_id in vocabulary.program_states[1:]
    }
    if any(not local for local in by_program.values()):
        raise ReactionProgramSamplingError("semantic-layout source omits a declared program")
    rng = np.random.default_rng(seed)
    program_ids = tuple(sorted(by_program))
    layouts: list[ReactionProgramLayout] = []
    for _ in range(sample_count):
        program_id = str(rng.choice(program_ids))
        local = by_program[program_id]
        record = local[int(rng.integers(len(local)))]
        if len(set(record.repeat_atom_counts)) != 1:
            raise ReactionProgramSamplingError(
                "semantic-layout source has unequal repeated-component sizes"
            )
        accumulator_role = int(record.role_states[0])
        repeat_start = record.accumulator_atom_count
        repeat_role = int(record.role_states[repeat_start])
        layouts.append(
            ReactionProgramLayout(
                program_id=program_id,
                program_state=record.program_state,
                program_depth=record.program_depth,
                accumulator_role_state=accumulator_role,
                repeat_role_state=repeat_role,
                accumulator_atom_count=record.accumulator_atom_count,
                repeat_atom_count=record.repeat_atom_counts[0],
                closure_count=record.graph.closure_count,
                core_position_word=tuple(int(value) for value in record.core_position_states),
            )
        )
    return tuple(layouts)


def reaction_program_source_marginals(
    records: Sequence[ReactionProgramGraphRecord],
    *,
    node_classes: int,
    bond_classes: int,
    probability_floor: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit full-support categorical source marginals on the training fold."""

    if not records or probability_floor <= 0:
        raise ReactionProgramSamplingError("source marginals require records and a positive floor")
    nodes = np.full(node_classes, probability_floor, dtype=np.float64)
    bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
    for record in records:
        nodes += np.bincount(record.graph.node_states, minlength=node_classes)
        bonds += np.bincount(record.graph.parent_bonds[1:], minlength=bond_classes)
        bonds += np.bincount(record.graph.closure_bonds, minlength=bond_classes)
    return nodes / nodes.sum(), bonds / bonds.sum()


def _semantic_parent_candidates(layout: ReactionProgramLayout, child: int) -> range:
    accumulator_end = layout.accumulator_atom_count
    if child < accumulator_end:
        return range(0, child)
    relative = child - accumulator_end
    block_start = (
        accumulator_end + (relative // layout.repeat_atom_count) * layout.repeat_atom_count
    )
    if child == block_start:
        return range(0, accumulator_end)
    return range(block_start, child)


def _terminal_graph(
    predictions: Mapping[str, Any],
    index: int,
    layout: ReactionProgramLayout,
    atom_vocabulary: Sequence[AtomState],
    generator: Any,
) -> tuple[str | None, dict[str, int]]:
    count = layout.node_count
    device = predictions["nodes"].device
    repairs: Counter[str] = Counter()
    node_states = np.zeros(count, dtype=np.int64)
    capacities = np.zeros(count, dtype=np.int64)
    for node in range(count):
        state, _ = _sample_from_logits(
            predictions["nodes"][index, node],
            torch.ones(len(atom_vocabulary), dtype=torch.bool, device=device),
            generator,
        )
        node_states[node] = state
        capacities[node] = _maximum_valence_units(atom_vocabulary[state])
    parents = np.zeros(count, dtype=np.int64)
    used = np.zeros(count, dtype=np.int64)
    edges = np.zeros((count, count), dtype=np.int64)
    for child in range(1, count):
        candidates = list(_semantic_parent_candidates(layout, child))
        valid = torch.zeros(count, dtype=torch.bool, device=device)
        for parent in candidates:
            if used[parent] + 2 <= capacities[parent] and used[child] + 2 <= capacities[child]:
                valid[parent] = True
        if not bool(valid.any()):
            repairs["no_valence_parent"] += 1
            return None, dict(repairs)
        parent, _ = _sample_from_logits(
            predictions["parents"][index, child, :count], valid, generator
        )
        parents[child] = parent
        used[parent] += 2
        used[child] += 2
        spare = min(capacities[parent] - used[parent], capacities[child] - used[child])
        bond_units = BOND_VALENCE_UNITS[: predictions["parent_bonds"].shape[-1]].to(device)
        valid_bonds = bond_units <= 2 + max(0, int(spare))
        bond, repaired = _sample_from_logits(
            predictions["parent_bonds"][index, child], valid_bonds, generator
        )
        if repaired:
            repairs["bond_fallback"] += 1
        extra = int(bond_units[bond]) - 2
        used[parent] += extra
        used[child] += extra
        dense = INDEX_TO_DENSE_BOND[bond]
        edges[parent, child] = edges[child, parent] = dense
    for slot in range(layout.closure_count):
        valid_pairs = torch.zeros((count, count), dtype=torch.bool, device=device)
        boundaries = [
            (0, layout.accumulator_atom_count),
            *[
                (
                    layout.accumulator_atom_count + step * layout.repeat_atom_count,
                    layout.accumulator_atom_count + (step + 1) * layout.repeat_atom_count,
                )
                for step in range(layout.program_depth)
            ],
        ]
        for start, end in boundaries:
            for left in range(start, end):
                for right in range(left + 1, end):
                    if (
                        edges[left, right] == 0
                        and used[left] + 2 <= capacities[left]
                        and used[right] + 2 <= capacities[right]
                    ):
                        valid_pairs[left, right] = True
        if not bool(valid_pairs.any()):
            repairs["closure_omitted"] += layout.closure_count - slot
            break
        pair_logits = (
            predictions["closure_left"][index, slot, :count, None]
            + predictions["closure_right"][index, slot, None, :count]
        )
        pair, _ = _sample_from_logits(pair_logits.flatten(), valid_pairs.flatten(), generator)
        left, right = divmod(pair, count)
        spare = min(capacities[left] - used[left], capacities[right] - used[right])
        bond_units = BOND_VALENCE_UNITS[: predictions["closure_bonds"].shape[-1]].to(device)
        valid_bonds = bond_units <= max(0, int(spare))
        bond, repaired = _sample_from_logits(
            predictions["closure_bonds"][index, slot], valid_bonds, generator
        )
        if repaired:
            repairs["closure_bond_fallback"] += 1
        units = int(bond_units[bond])
        used[left] += units
        used[right] += units
        dense = INDEX_TO_DENSE_BOND[bond]
        edges[left, right] = edges[right, left] = dense
    try:
        molecule = graph_to_molecule(node_states, edges, atom_vocabulary)
    except Exception:
        repairs["sanitization_failed"] += 1
        return None, dict(repairs)
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False), dict(repairs)


def sample_reaction_program_products(
    model: Any,
    layouts: Sequence[ReactionProgramLayout],
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    *,
    adapters: Mapping[str, ReactionProgramAdapter] | None,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate whole graphs and audit exact open decomposition under the requested program."""

    if torch is None or not layouts or sample_steps < 2 or batch_size < 1:
        raise ReactionProgramSamplingError("invalid reaction-program sampling request")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    node_p0 = torch.as_tensor(node_marginal, dtype=torch.float32, device=resolved_device)
    bond_p0 = torch.as_tensor(bond_marginal, dtype=torch.float32, device=resolved_device)
    maximum_closures = int(model.maximum_closures)
    outputs: list[dict[str, Any]] = []
    repair_totals: Counter[str] = Counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(layouts), batch_size):
            local = tuple(layouts[offset : offset + batch_size])
            counts = torch.tensor([layout.node_count for layout in local], device=resolved_device)
            requested = torch.tensor(
                [layout.closure_count for layout in local], device=resolved_device
            )
            n_max = int(counts.max())
            node_mask = torch.arange(n_max, device=resolved_device)[None, :] < counts[:, None]
            child_mask = node_mask.clone()
            child_mask[:, 0] = False
            closure_mask = (
                torch.arange(maximum_closures, device=resolved_device)[None, :] < requested[:, None]
            )
            parent_candidates = _parent_candidate_mask(node_mask)
            endpoint_candidates = _endpoint_candidate_mask(node_mask, maximum_closures)
            state = {
                "nodes": torch.multinomial(
                    node_p0,
                    len(local) * n_max,
                    replacement=True,
                    generator=generator,
                ).reshape(len(local), n_max),
                "parent_bonds": torch.multinomial(
                    bond_p0,
                    len(local) * n_max,
                    replacement=True,
                    generator=generator,
                ).reshape(len(local), n_max),
                "closure_bonds": torch.multinomial(
                    bond_p0,
                    len(local) * maximum_closures,
                    replacement=True,
                    generator=generator,
                ).reshape(len(local), maximum_closures),
            }
            parent_probabilities = parent_candidates.to(torch.float32)
            parent_probabilities /= parent_probabilities.sum(dim=-1, keepdim=True).clamp(min=1)
            state["parents"] = torch.zeros((len(local), n_max), dtype=torch.long, device=device)
            state["parents"][child_mask] = torch.multinomial(
                parent_probabilities[child_mask], 1, generator=generator
            ).squeeze(1)
            endpoint_probabilities = endpoint_candidates.to(torch.float32)
            endpoint_probabilities /= endpoint_probabilities.sum(dim=-1, keepdim=True)
            state["closure_left"] = torch.multinomial(
                endpoint_probabilities.reshape(-1, n_max), 1, generator=generator
            ).reshape(len(local), maximum_closures)
            state["closure_right"] = torch.multinomial(
                endpoint_probabilities.reshape(-1, n_max), 1, generator=generator
            ).reshape(len(local), maximum_closures)
            program_states = torch.tensor(
                [layout.program_state for layout in local], device=resolved_device
            )
            program_depths = torch.tensor(
                [layout.program_depth for layout in local], device=resolved_device
            )
            role_states = torch.zeros((len(local), n_max), dtype=torch.long, device=device)
            core_position_states = torch.zeros((len(local), n_max), dtype=torch.long, device=device)
            for index, layout in enumerate(local):
                role_states[index, : layout.node_count] = torch.tensor(
                    layout.role_states, device=resolved_device
                )
                core_position_states[index, : layout.node_count] = torch.tensor(
                    layout.core_position_states, device=resolved_device
                )
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full((len(local),), t_value, device=resolved_device)
                predictions = model(
                    nodes=state["nodes"],
                    parents=state["parents"],
                    parent_bonds=state["parent_bonds"],
                    closure_left=state["closure_left"],
                    closure_right=state["closure_right"],
                    closure_bonds=state["closure_bonds"],
                    t=t,
                    node_mask=node_mask,
                    child_mask=child_mask,
                    closure_mask=closure_mask,
                    program_states=program_states,
                    role_states=role_states,
                    core_position_states=core_position_states,
                    program_depths=program_depths,
                    adapter_mask=node_mask,
                )
                state["nodes"] = rstar_step(
                    state["nodes"],
                    predictions["nodes"].softmax(dim=-1),
                    node_p0,
                    t_value,
                    1.0 / sample_steps,
                    node_mask,
                    generator,
                )
                state["parents"] = pointer_rstar_step(
                    state["parents"],
                    predictions["parents"],
                    parent_candidates,
                    child_mask,
                    t_value,
                    1.0 / sample_steps,
                    generator,
                )
                for key, mask in (
                    ("parent_bonds", child_mask),
                    ("closure_bonds", closure_mask),
                ):
                    state[key] = rstar_step(
                        state[key],
                        predictions[key].softmax(dim=-1),
                        bond_p0,
                        t_value,
                        1.0 / sample_steps,
                        mask,
                        generator,
                    )
                for key in ("closure_left", "closure_right"):
                    state[key] = pointer_rstar_step(
                        state[key],
                        predictions[key],
                        endpoint_candidates,
                        closure_mask,
                        t_value,
                        1.0 / sample_steps,
                        generator,
                    )
            terminal = model(
                nodes=state["nodes"],
                parents=state["parents"],
                parent_bonds=state["parent_bonds"],
                closure_left=state["closure_left"],
                closure_right=state["closure_right"],
                closure_bonds=state["closure_bonds"],
                t=torch.ones(len(local), device=resolved_device),
                node_mask=node_mask,
                child_mask=child_mask,
                closure_mask=closure_mask,
                program_states=program_states,
                role_states=role_states,
                core_position_states=core_position_states,
                program_depths=program_depths,
                adapter_mask=node_mask,
            )
            for index, layout in enumerate(local):
                smiles, repairs = _terminal_graph(
                    terminal, index, layout, atom_vocabulary, generator
                )
                exact_traces: tuple[Any, ...] = ()
                verified_trace_count = 0
                l1_abstention_reason = ""
                if smiles is not None and adapters is not None:
                    try:
                        traces = adapters[layout.program_id].decompose(smiles)
                        exact_traces = tuple(
                            trace
                            for trace in traces
                            if trace.step_count == layout.program_depth
                            and len(set(trace.repeated_component_smiles)) == 1
                        )
                        for trace in exact_traces:
                            check = adapters[layout.program_id].check_forward(
                                trace.terminal_head_smiles,
                                trace.repeated_component_smiles,
                                smiles,
                            )
                            if check.exact and not check.saturated:
                                verified_trace_count += 1
                    except ReactionProgramError as error:
                        repairs["l1_decomposition_abstained"] = 1
                        l1_abstention_reason = str(error)
                repair_totals.update(repairs)
                outputs.append(
                    {
                        "sample_index": offset + index,
                        "program_id": layout.program_id,
                        "program_depth": layout.program_depth,
                        "node_count": layout.node_count,
                        "closure_count": layout.closure_count,
                        "canonical_smiles": smiles,
                        "valid": smiles is not None,
                        "exact_l1_program": bool(exact_traces),
                        "exact_l1_trace_count": len(exact_traces),
                        "forward_verified_trace_count": verified_trace_count,
                        "exact_l1_traces": [
                            {
                                "terminal_head_smiles": trace.terminal_head_smiles,
                                "repeated_component_smiles": list(trace.repeated_component_smiles),
                            }
                            for trace in exact_traces
                        ],
                        "l1_abstention_reason": l1_abstention_reason,
                        "repairs": repairs,
                    }
                )
    return outputs, {
        "samples": len(outputs),
        "valid": sum(bool(row["valid"]) for row in outputs),
        "exact_l1_program": sum(bool(row["exact_l1_program"]) for row in outputs),
        "repairs": dict(sorted(repair_totals.items())),
    }


__all__ = [
    "ReactionProgramLayout",
    "ReactionProgramSamplingError",
    "load_reaction_program_checkpoint",
    "reaction_program_source_marginals",
    "sample_factorized_program_layouts",
    "sample_training_semantic_layouts",
    "sample_reaction_program_products",
]
