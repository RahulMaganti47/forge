"""Program-conditioned sparse whole-product flow shared across reaction families."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from forge.model.phase1_flow import SparseWholeLipidFlow
from forge.model.reaction_program_conditioning import (
    ReactionProgramConditioning,
    ReactionProgramVocabulary,
)
from forge.model.reaction_program_graph import ReactionProgramGraphRecord
from forge.model.sparse_topology_feasibility import (
    _endpoint_candidate_mask,
    _parent_candidate_mask,
    _sample_flat_interpolation,
    collate_sparse_records,
    sample_pointer_interpolation,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


class ReactionProgramFlowError(ValueError):
    """Reaction-program flow inputs violate the declared semantic support."""


ROLE_MORPHOLOGY_FIELDS = (
    "exterior_node_count",
    "junction_budget",
    "cycle_rank",
    "attachment_count",
)


def derive_role_morphology_states(record: SynthesisProgramGraphRecord) -> np.ndarray:
    """Broadcast exact coarse morphology to every node of its precursor role.

    The four coordinates reproduce the vocabulary-free Ugi morphology contract without exposing
    atom identities, component identifiers or fragments.  State zero is reserved for an absent
    condition, so every observed integer is encoded as ``value + 1``.
    """

    node_count = record.node_count
    core = record.core_position_states > 1
    output = np.zeros((node_count, len(ROLE_MORPHOLOGY_FIELDS)), dtype=np.int64)
    for role_state in sorted(set(int(value) for value in record.role_states)):
        if role_state <= 0:
            continue
        role = record.role_states == role_state
        exterior = role & ~core
        exterior_indices = set(np.flatnonzero(exterior).tolist())
        child_counts = {index: 0 for index in exterior_indices}
        attachments = 0
        for child in range(1, node_count):
            parent = int(record.graph.parents[child])
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
        for left, right in zip(record.graph.closure_left, record.graph.closure_right, strict=True):
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
        values = np.asarray(
            (
                len(exterior_indices),
                sum(max(children - 1, 0) for children in child_counts.values()),
                cycles,
                attachments,
            ),
            dtype=np.int64,
        )
        output[role] = values + 1
    return output


def collate_reaction_program_records(
    records: Sequence[ReactionProgramGraphRecord],
    *,
    maximum_nodes: int | None = None,
    maximum_closures: int,
    conditioning_mode: str = "program",
) -> dict[str, Any]:
    """Collate exact graph targets and clean program semantics.

    ``null`` removes all semantic coordinates. ``program_shuffled`` is applied by the experiment
    runner because it needs a seeded permutation across complete records.
    """

    if torch is None or not records:
        raise ReactionProgramFlowError("reaction-program collation requires records and torch")
    if conditioning_mode not in {"program", "null"}:
        raise ReactionProgramFlowError(f"unsupported conditioning mode: {conditioning_mode!r}")
    n_max = maximum_nodes or max(record.node_count for record in records)
    clean = collate_sparse_records(
        [record.graph for record in records],
        n_max,
        maximum_closures,
    )
    roles = torch.zeros((len(records), n_max), dtype=torch.long)
    core_positions = torch.zeros((len(records), n_max), dtype=torch.long)
    programs = torch.zeros(len(records), dtype=torch.long)
    depths = torch.zeros(len(records), dtype=torch.long)
    if conditioning_mode == "program":
        for index, record in enumerate(records):
            roles[index, : record.node_count] = torch.from_numpy(record.role_states.copy())
            core_positions[index, : record.node_count] = torch.from_numpy(
                record.core_position_states.copy()
            )
            programs[index] = record.program_state
            depths[index] = record.program_depth
    clean.update(
        {
            "program_states": programs,
            "role_states": roles,
            "core_position_states": core_positions,
            "program_depths": depths,
            "adapter_mask": clean["node_mask"].clone(),
        }
    )
    return clean


def collate_synthesis_program_records(
    records: Sequence[SynthesisProgramGraphRecord],
    *,
    maximum_nodes: int | None = None,
    maximum_closures: int,
    conditioning_mode: str = "program",
) -> dict[str, Any]:
    """Collate the shared Ugi/BL/LX contract with explicit immutable state masks.

    The fixed masks are adapter semantics, not copied component identities.  A fixed Ugi parent
    edge protects its pointer and bond state together; a fixed closure protects both endpoints and
    its bond.  Auxiliary-family core positions remain conditioning coordinates only.
    """

    if torch is None or not records:
        raise ReactionProgramFlowError("synthesis-program collation requires records and torch")
    if conditioning_mode not in {"program", "null"}:
        raise ReactionProgramFlowError(f"unsupported conditioning mode: {conditioning_mode!r}")
    n_max = maximum_nodes or max(record.node_count for record in records)
    if n_max < max(record.node_count for record in records):
        raise ReactionProgramFlowError("synthesis-program batch exceeds node capacity")
    clean = collate_sparse_records(
        [record.graph for record in records],
        n_max,
        maximum_closures,
    )
    shape = clean["nodes"].shape
    closure_shape = clean["closure_bonds"].shape
    roles = torch.zeros(shape, dtype=torch.long)
    core_positions = torch.zeros(shape, dtype=torch.long)
    programs = torch.zeros(len(records), dtype=torch.long)
    depths = torch.zeros(len(records), dtype=torch.long)
    fixed_atoms = torch.zeros(shape, dtype=torch.bool)
    fixed_parents = torch.zeros(shape, dtype=torch.bool)
    fixed_closures = torch.zeros(closure_shape, dtype=torch.bool)
    component_instances = torch.zeros(shape, dtype=torch.long)
    component_positions = torch.zeros(shape, dtype=torch.long)
    repeat_groups = torch.zeros(shape, dtype=torch.long)
    role_morphology = torch.zeros((*shape, len(ROLE_MORPHOLOGY_FIELDS)), dtype=torch.long)
    for index, record in enumerate(records):
        count = record.node_count
        closure_count = record.graph.closure_count
        if conditioning_mode == "program":
            roles[index, :count] = torch.from_numpy(record.role_states.copy())
            core_positions[index, :count] = torch.from_numpy(record.core_position_states.copy())
            programs[index] = record.program_state
            depths[index] = record.program_depth
            morphology = (
                derive_role_morphology_states(record)
                if record.role_morphology_states is None
                else record.role_morphology_states
            )
            role_morphology[index, :count] = torch.from_numpy(morphology.copy())
            role_multiplicities = Counter(block.role_state for block in record.component_blocks)
            for component_index, block in enumerate(record.component_blocks, start=1):
                component_instances[index, block.start : block.stop] = component_index
                component_positions[index, block.start : block.stop] = torch.arange(
                    1,
                    block.atom_count + 1,
                    dtype=torch.long,
                )
                if role_multiplicities[block.role_state] > 1:
                    # The role state is the stable equivalence-group identifier.  It does not
                    # number otherwise indistinguishable repeated reaction steps.
                    repeat_groups[index, block.start : block.stop] = block.role_state
        fixed_atoms[index, :count] = torch.from_numpy(record.fixed_atom_mask.copy())
        fixed_parents[index, :count] = torch.from_numpy(record.fixed_parent_bond_mask.copy())
        fixed_closures[index, :closure_count] = torch.from_numpy(
            record.fixed_closure_bond_mask.copy()
        )
    clean.update(
        {
            "program_states": programs,
            "role_states": roles,
            "core_position_states": core_positions,
            "program_depths": depths,
            "component_instance_states": component_instances,
            "component_position_states": component_positions,
            "repeat_group_states": repeat_groups,
            "role_morphology_states": role_morphology,
            "adapter_mask": clean["node_mask"].clone(),
            "fixed_atom_mask": fixed_atoms,
            "fixed_parent_mask": fixed_parents,
            "fixed_parent_bond_mask": fixed_parents.clone(),
            "fixed_closure_endpoint_mask": fixed_closures,
            "fixed_closure_bond_mask": fixed_closures.clone(),
        }
    )
    clean.update(
        {
            "atom_variable_mask": clean["node_mask"] & ~fixed_atoms,
            "parent_variable_mask": clean["child_mask"] & ~fixed_parents,
            "parent_bond_variable_mask": clean["child_mask"] & ~fixed_parents,
            "closure_endpoint_variable_mask": clean["closure_mask"] & ~fixed_closures,
            "closure_bond_variable_mask": clean["closure_mask"] & ~fixed_closures,
        }
    )
    return clean


def collate_synthesis_program_layouts(
    records: Sequence[SynthesisProgramGraphRecord],
    *,
    maximum_nodes: int | None = None,
    maximum_closures: int,
    conditioning_mode: str = "program",
    program_state_mapping: Mapping[int, int] | None = None,
    role_state_mapping: Mapping[int, int] | None = None,
) -> dict[str, Any]:
    """Project records to sampling-visible semantics and fixed states only.

    Nonfixed atom, pointer, and bond targets are overwritten with zero.  This prevents an overfit
    layout source from becoming an accidental product-template or component-vocabulary channel.
    """

    clean = collate_synthesis_program_records(
        records,
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
        conditioning_mode=("null" if conditioning_mode == "null" else "program"),
    )
    if conditioning_mode not in {"program", "null", "program_mapped"}:
        raise ReactionProgramFlowError(
            f"unsupported sampling conditioning mode: {conditioning_mode!r}"
        )
    if conditioning_mode == "program_mapped":
        if program_state_mapping is None and role_state_mapping is None:
            raise ReactionProgramFlowError(
                "mapped sampling requires a program- or role-state mapping"
            )
        for label, fields, mapping in (
            ("program", ("program_states",), program_state_mapping),
            ("role", ("role_states", "repeat_group_states"), role_state_mapping),
        ):
            if mapping is None:
                continue
            if any(int(source) <= 0 or int(target) <= 0 for source, target in mapping.items()):
                raise ReactionProgramFlowError(
                    f"{label}-state mappings may not rewrite the null state"
                )
            for field in fields:
                source_values = clean[field]
                mapped = source_values.clone()
                for source, target in mapping.items():
                    mapped[source_values == int(source)] = int(target)
                clean[field] = mapped
    elif program_state_mapping is not None or role_state_mapping is not None:
        raise ReactionProgramFlowError("state mappings are only valid for mapped sampling")
    projected = {
        key: value.clone()
        for key, value in clean.items()
        if key
        in {
            "node_mask",
            "child_mask",
            "closure_mask",
            "program_states",
            "role_states",
            "core_position_states",
            "program_depths",
            "component_instance_states",
            "component_position_states",
            "repeat_group_states",
            "role_morphology_states",
            "adapter_mask",
            "fixed_atom_mask",
            "fixed_parent_mask",
            "fixed_parent_bond_mask",
            "fixed_closure_endpoint_mask",
            "fixed_closure_bond_mask",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    for field, mask_name in {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }.items():
        projected[field] = torch.zeros_like(clean[field])
        mask = clean[mask_name]
        projected[field][mask] = clean[field][mask]
    return projected


def noise_synthesis_program_batch(
    clean: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    """Corrupt only learnable states while preserving adapter-fixed Ugi chemistry exactly."""

    parent_candidates = _parent_candidate_mask(clean["node_mask"])
    endpoint_candidates = _endpoint_candidate_mask(
        clean["node_mask"], clean["closure_left"].shape[1]
    )
    node_source, parent_bond_source, closure_bond_source = (
        resolve_synthesis_program_source_marginals(clean, node_marginal, bond_marginal)
    )
    return {
        "nodes": _sample_categorical_interpolation(
            clean["nodes"], node_source, t, clean["atom_variable_mask"], generator
        ),
        "parents": sample_pointer_interpolation(
            clean["parents"],
            parent_candidates,
            clean["parent_variable_mask"],
            t,
            generator,
        ),
        "parent_bonds": _sample_categorical_interpolation(
            clean["parent_bonds"],
            parent_bond_source,
            t,
            clean["parent_bond_variable_mask"],
            generator,
        ),
        "closure_left": sample_pointer_interpolation(
            clean["closure_left"],
            endpoint_candidates,
            clean["closure_endpoint_variable_mask"],
            t,
            generator,
        ),
        "closure_right": sample_pointer_interpolation(
            clean["closure_right"],
            endpoint_candidates,
            clean["closure_endpoint_variable_mask"],
            t,
            generator,
        ),
        "closure_bonds": _sample_categorical_interpolation(
            clean["closure_bonds"],
            closure_bond_source,
            t,
            clean["closure_bond_variable_mask"],
            generator,
        ),
    }


def resolve_synthesis_program_source_marginals(
    clean: Mapping[str, Any], node_marginal: Any, bond_marginal: Any
) -> tuple[Any, Any, Any]:
    """Resolve legacy global or program/role sources onto graph positions."""

    if node_marginal.ndim == 1 and bond_marginal.ndim == 1:
        return node_marginal, bond_marginal, bond_marginal
    if node_marginal.ndim != 3 or bond_marginal.ndim != 3:
        raise ReactionProgramFlowError(
            "node and bond sources must both be global or program/role conditioned"
        )
    programs = clean["program_states"]
    roles = clean["role_states"]
    if (
        node_marginal.shape[:2] != bond_marginal.shape[:2]
        or int(programs.max()) >= node_marginal.shape[0]
        or int(roles.max()) >= node_marginal.shape[1]
    ):
        raise ReactionProgramFlowError("program/role source support is incompatible with the batch")
    node_source = node_marginal[programs[:, None], roles]
    parent_bond_source = bond_marginal[programs[:, None], roles]
    closure_bond_source = bond_marginal[programs, 0][:, None, :].expand(
        -1, clean["closure_bonds"].shape[1], -1
    )
    return node_source, parent_bond_source, closure_bond_source


def _sample_categorical_interpolation(
    clean: Any, marginal: Any, t: Any, active_mask: Any, generator: Any
) -> Any:
    """Sample the categorical interpolant from one global or one source per position."""

    if marginal.ndim == 1:
        return _sample_flat_interpolation(clean, marginal, t, active_mask, generator)
    if marginal.ndim != clean.ndim + 1 or marginal.shape[:-1] != clean.shape:
        raise ReactionProgramFlowError("position-specific source marginal has invalid shape")
    probabilities = marginal[active_mask].clone()
    example_index = torch.arange(clean.shape[0], device=clean.device)[:, None].expand_as(clean)[
        active_mask
    ]
    probabilities *= 1.0 - t[example_index, None]
    probabilities.scatter_add_(1, clean[active_mask][:, None], t[example_index, None])
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    output = clean.clone()
    output[active_mask] = sampled
    return output


def restore_synthesis_program_fixed_states(
    state: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a cloned state with every adapter-fixed atom and edge restored exactly."""

    restored = {key: value.clone() for key, value in state.items()}
    fields = {
        "nodes": "fixed_atom_mask",
        "parents": "fixed_parent_mask",
        "parent_bonds": "fixed_parent_bond_mask",
        "closure_left": "fixed_closure_endpoint_mask",
        "closure_right": "fixed_closure_endpoint_mask",
        "closure_bonds": "fixed_closure_bond_mask",
    }
    for field, mask_name in fields.items():
        mask = clean[mask_name]
        restored[field][mask] = clean[field][mask]
    return restored


def decode_synthesis_program_argmax(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode clean targets under graph support, then reinstate the exact Ugi fixed state."""

    parent_candidates = _parent_candidate_mask(clean["node_mask"])
    endpoint_candidates = _endpoint_candidate_mask(
        clean["node_mask"], clean["closure_left"].shape[1]
    )
    terminal = {
        "nodes": predictions["nodes"].argmax(dim=-1),
        "parents": predictions["parents"].masked_fill(~parent_candidates, -1e9).argmax(dim=-1),
        "parent_bonds": predictions["parent_bonds"].argmax(dim=-1),
        "closure_left": predictions["closure_left"]
        .masked_fill(~endpoint_candidates, -1e9)
        .argmax(dim=-1),
        "closure_right": predictions["closure_right"]
        .masked_fill(~endpoint_candidates, -1e9)
        .argmax(dim=-1),
        "closure_bonds": predictions["closure_bonds"].argmax(dim=-1),
    }
    for field, mask_name in {
        "nodes": "node_mask",
        "parents": "child_mask",
        "parent_bonds": "child_mask",
        "closure_left": "closure_mask",
        "closure_right": "closure_mask",
        "closure_bonds": "closure_mask",
    }.items():
        inactive = ~clean[mask_name]
        terminal[field][inactive] = clean[field][inactive]
    return restore_synthesis_program_fixed_states(terminal, clean)


def _masked_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    """Mean cross entropy over selected states without synchronizing the accelerator.

    ``cross_entropy(..., reduction="sum")`` is exactly zero for an empty selection.  Dividing by a
    clamped tensor count therefore preserves the previous empty-mask behavior without evaluating a
    CUDA tensor as a Python boolean on every objective component.
    """

    selected = functional.cross_entropy(logits[mask], targets[mask], reduction="sum")
    return selected / mask.sum().clamp(min=1)


def _group_balanced_masked_cross_entropy(logits: Any, targets: Any, mask: Any, groups: Any) -> Any:
    """Give each present semantic group equal mass while retaining per-state supervision."""

    selected_groups = groups[mask]
    point_losses = functional.cross_entropy(logits[mask], targets[mask], reduction="none")
    group_count = int(groups.max()) + 1
    sums = logits.new_zeros(group_count).scatter_add(0, selected_groups, point_losses)
    counts = torch.bincount(selected_groups, minlength=group_count)
    present = counts > 0
    means = sums / counts.clamp(min=1)
    return (means * present).sum() / present.sum().clamp(min=1)


def synthesis_program_chemistry_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    balance_by_role: bool,
) -> tuple[Any, dict[str, Any]]:
    """Return atom and bond-state loss without topology-pointer terms."""

    cross_entropy = _masked_cross_entropy
    node_arguments: tuple[Any, ...] = ()
    parent_arguments: tuple[Any, ...] = ()
    closure_arguments: tuple[Any, ...] = ()
    if balance_by_role:
        cross_entropy = _group_balanced_masked_cross_entropy
        node_arguments = (clean["role_states"],)
        parent_arguments = (clean["role_states"],)
        batch = torch.arange(clean["role_states"].shape[0], device=clean["role_states"].device)[
            :, None
        ]
        closure_roles = clean["role_states"][batch, clean["closure_left"]]
        closure_arguments = (closure_roles,)
    losses = {
        "node_ce": cross_entropy(
            predictions["nodes"],
            clean["nodes"],
            clean["atom_variable_mask"],
            *node_arguments,
        ),
        "backbone_bond_ce": cross_entropy(
            predictions["parent_bonds"],
            clean["parent_bonds"],
            clean["parent_bond_variable_mask"],
            *parent_arguments,
        ),
        "closure_bond_ce": cross_entropy(
            predictions["closure_bonds"],
            clean["closure_bonds"],
            clean["closure_bond_variable_mask"],
            *closure_arguments,
        ),
    }
    return sum(losses.values(), start=predictions["nodes"].new_zeros(())), losses


def synthesis_program_flow_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    balance_chemistry_by_role: bool = False,
    materialize_metrics: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Train only variable graph states; adapter-fixed Ugi targets contribute zero loss."""

    parent_logits = predictions["parents"].masked_fill(
        ~_parent_candidate_mask(clean["node_mask"]), -1e9
    )
    endpoint_candidates = _endpoint_candidate_mask(
        clean["node_mask"], clean["closure_left"].shape[1]
    )
    left_logits = predictions["closure_left"].masked_fill(~endpoint_candidates, -1e9)
    right_logits = predictions["closure_right"].masked_fill(~endpoint_candidates, -1e9)
    _, chemistry_losses = synthesis_program_chemistry_loss(
        predictions,
        clean,
        balance_by_role=balance_chemistry_by_role,
    )
    losses = {
        "node_ce": chemistry_losses["node_ce"],
        "parent_pointer_ce": _masked_cross_entropy(
            parent_logits, clean["parents"], clean["parent_variable_mask"]
        ),
        "backbone_bond_ce": chemistry_losses["backbone_bond_ce"],
        "closure_left_ce": _masked_cross_entropy(
            left_logits,
            clean["closure_left"],
            clean["closure_endpoint_variable_mask"],
        ),
        "closure_right_ce": _masked_cross_entropy(
            right_logits,
            clean["closure_right"],
            clean["closure_endpoint_variable_mask"],
        ),
        "closure_bond_ce": chemistry_losses["closure_bond_ce"],
    }
    total = sum(losses.values(), start=predictions["nodes"].new_zeros(()))
    if materialize_metrics:
        metrics = {name: float(value.detach()) for name, value in losses.items()}
        metrics["total"] = float(total.detach())
    else:
        metrics = {name: value.detach() for name, value in losses.items()}
        metrics["total"] = total.detach()
    return total, metrics


if nn is not None:

    class ReactionProgramSparseFlow(nn.Module):
        """Whole-product sparse flow with reaction program coordinates as clean context."""

        def __init__(
            self,
            *,
            vocabulary: ReactionProgramVocabulary,
            node_classes: int,
            hidden_dim: int,
            layers: int,
            maximum_closures: int,
            maximum_heavy_atoms: int,
            dropout: float,
            bond_classes: int = 3,
        ) -> None:
            super().__init__()
            self.vocabulary = vocabulary
            self.maximum_heavy_atoms = maximum_heavy_atoms
            self.maximum_closures = maximum_closures
            self.conditioning = ReactionProgramConditioning(
                vocabulary=vocabulary,
                hidden_dim=hidden_dim,
            )
            self.backbone = SparseWholeLipidFlow(
                node_classes=node_classes,
                hidden_dim=hidden_dim,
                layers=layers,
                maximum_closures=maximum_closures,
                maximum_heavy_atoms=maximum_heavy_atoms,
                dropout=dropout,
                bond_classes=bond_classes,
                use_position_embedding=True,
            )

        def forward(
            self,
            *,
            nodes: Any,
            parents: Any,
            parent_bonds: Any,
            closure_left: Any,
            closure_right: Any,
            closure_bonds: Any,
            t: Any,
            node_mask: Any,
            child_mask: Any,
            closure_mask: Any,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            repeat_group_states: Any | None = None,
            component_position_states: Any | None = None,
            component_instance_states: Any | None = None,
        ) -> dict[str, Any]:
            del repeat_group_states, component_position_states, component_instance_states
            context = self.conditioning(
                program_states=program_states,
                role_states=role_states,
                core_position_states=core_position_states,
                program_depths=program_depths,
                adapter_mask=adapter_mask,
            )
            return self.backbone(
                nodes,
                parents,
                parent_bonds,
                closure_left,
                closure_right,
                closure_bonds,
                t,
                node_mask,
                child_mask,
                closure_mask,
                node_context=context,
            )

else:  # pragma: no cover

    class ReactionProgramSparseFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise ReactionProgramFlowError("reaction-program flow requires torch")


__all__ = [
    "ROLE_MORPHOLOGY_FIELDS",
    "ReactionProgramFlowError",
    "ReactionProgramSparseFlow",
    "collate_reaction_program_records",
    "collate_synthesis_program_layouts",
    "collate_synthesis_program_records",
    "derive_role_morphology_states",
    "decode_synthesis_program_argmax",
    "noise_synthesis_program_batch",
    "resolve_synthesis_program_source_marginals",
    "restore_synthesis_program_fixed_states",
    "synthesis_program_chemistry_loss",
    "synthesis_program_flow_loss",
]
