"""Restartable zero-guidance operations for the selected joint sparse flow.

This module is intentionally controller-free. It exposes cloneable trajectory
state and behavior-preserving advance/finalize operations so that exact
zero-guidance equivalence can be qualified before synthesis coupling is added.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.flow.rstar import rstar_step as _rstar_step
from forge.model.networks.ugi_joint_flow import (
    UgiJointSparseFlowError,
    UgiJointSparseTerminal,
    _decoration_anchor_source,
)
from forge.model.networks.ugi_morphology import _layout_from_programs
from forge.model.representation.ugi_morphology import (
    UgiMorphologyProgram,
    attached_tree_junction_contributions,
    sample_attached_offspring_with_exact_budget,
    sample_attached_offspring_with_exact_budget_and_cycle_rank,
    sample_attached_offspring_without_budget,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


@dataclass(frozen=True)
class UgiJointSparseTrajectoryState:
    """Cloneable categorical flow state at one exact step boundary."""

    programs: tuple[UgiMorphologyProgram, ...]
    layout: dict[str, Any]
    sources: dict[str, Any]
    channels: dict[str, Any]
    sample_steps: int
    step: int
    generator_state: Any
    device: str

    def clone(self) -> UgiJointSparseTrajectoryState:
        """Return an independent state suitable for deterministic rollouts."""

        return UgiJointSparseTrajectoryState(
            programs=self.programs,
            layout={key: value.clone() for key, value in self.layout.items()},
            sources={key: value.clone() for key, value in self.sources.items()},
            channels={key: value.clone() for key, value in self.channels.items()},
            sample_steps=self.sample_steps,
            step=self.step,
            generator_state=self.generator_state.clone(),
            device=self.device,
        )


@dataclass(frozen=True)
class UgiJointSparseFinalization:
    """Terminal records plus the advanced topology-decoder RNG state."""

    terminals: tuple[UgiJointSparseTerminal, ...]
    tree_generator_state: Any


def _resolved_branch_runs(
    maximum_adjacent_branch_runs: Sequence[int | None] | None,
) -> tuple[int | None, ...]:
    if maximum_adjacent_branch_runs is None:
        return (None,) * len(ROLE_NAMES)
    resolved = tuple(maximum_adjacent_branch_runs)
    if len(resolved) != len(ROLE_NAMES) or any(
        value is not None and value < 0 for value in resolved
    ):
        raise UgiJointSparseFlowError(
            "adjacent branch-run policy must provide one nonnegative bound per Ugi role"
        )
    return resolved


def _generator_from_state(device: Any, generator_state: Any) -> Any:
    generator = torch.Generator(device=device)
    generator.set_state(generator_state)
    return generator


def initialize_ugi_joint_sparse_state(
    model: Any,
    programs: Sequence[UgiMorphologyProgram],
    source_marginals: dict[str, Any],
    *,
    sample_steps: int,
    device: str,
    seed: int | None = None,
    generator_state: Any | None = None,
) -> UgiJointSparseTrajectoryState:
    """Sample the source state without advancing the flow trajectory."""

    if torch is None or not programs or sample_steps < 2:
        raise UgiJointSparseFlowError("invalid joint sparse initialization request")
    if (seed is None) == (generator_state is None):
        raise UgiJointSparseFlowError("provide exactly one seed or generator state")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    if generator_state is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(generator_state)
    sources = {
        key: (
            value.to(dtype=torch.float32, device=resolved_device)
            if torch.is_tensor(value)
            else torch.as_tensor(value, dtype=torch.float32, device=resolved_device)
        )
        for key, value in source_marginals.items()
    }
    local = tuple(programs)
    layout = {key: value.to(resolved_device) for key, value in _layout_from_programs(local).items()}
    roles = layout["role_states"]
    mask = layout["node_mask"]
    offspring_source = sources["offspring"][roles]
    atom_source = sources["atoms"][roles]
    bond_source = sources["bonds"][roles]
    channels = {
        "offspring": torch.multinomial(
            offspring_source.reshape(-1, offspring_source.shape[-1]),
            1,
            generator=generator,
        ).reshape(mask.shape),
        "nodes": torch.multinomial(
            atom_source.reshape(-1, atom_source.shape[-1]),
            1,
            generator=generator,
        ).reshape(mask.shape),
        "parent_bonds": torch.multinomial(
            bond_source.reshape(-1, bond_source.shape[-1]),
            1,
            generator=generator,
        ).reshape(mask.shape),
    }
    anchor_source = _decoration_anchor_source(mask, sources["decoration"])
    slots = model.maximum_decorations
    channels["decoration_anchors"] = torch.multinomial(
        anchor_source[:, None, :].expand(-1, slots, -1).reshape(-1, anchor_source.shape[-1]),
        1,
        generator=generator,
    ).reshape(len(local), slots)
    channels["decoration_atoms"] = torch.multinomial(
        sources["decoration_atoms"],
        len(local) * slots,
        replacement=True,
        generator=generator,
    ).reshape(len(local), slots)
    channels["decoration_bonds"] = torch.multinomial(
        sources["decoration_bonds"],
        len(local) * slots,
        replacement=True,
        generator=generator,
    ).reshape(len(local), slots)
    return UgiJointSparseTrajectoryState(
        programs=local,
        layout=layout,
        sources=sources,
        channels=channels,
        sample_steps=sample_steps,
        step=0,
        generator_state=generator.get_state().clone(),
        device=str(resolved_device),
    )


def advance_ugi_joint_sparse_state(
    model: Any,
    trajectory: UgiJointSparseTrajectoryState,
    *,
    target_step: int,
) -> UgiJointSparseTrajectoryState:
    """Advance an independent trajectory to an exact flow-step boundary."""

    if torch is None or not trajectory.step <= target_step <= trajectory.sample_steps:
        raise UgiJointSparseFlowError("invalid joint sparse target step")
    current = trajectory.clone()
    if target_step == current.step:
        return current
    resolved_device = torch.device(current.device)
    generator = _generator_from_state(resolved_device, current.generator_state)
    roles = current.layout["role_states"]
    mask = current.layout["node_mask"]
    offspring_source = current.sources["offspring"][roles]
    atom_source = current.sources["atoms"][roles]
    bond_source = current.sources["bonds"][roles]
    anchor_source = _decoration_anchor_source(mask, current.sources["decoration"])
    slots = model.maximum_decorations
    channels = current.channels
    model.eval()
    with torch.no_grad():
        for step in range(current.step, target_step):
            t_value = step / current.sample_steps
            t = torch.full(
                (len(current.programs),),
                t_value,
                dtype=torch.float32,
                device=resolved_device,
            )
            predictions = model(
                offspring=channels["offspring"],
                nodes=channels["nodes"],
                parent_bonds=channels["parent_bonds"],
                role_states=roles,
                within_role_positions=current.layout["within_role_positions"],
                programs=current.layout["programs"],
                node_mask=mask,
                t=t,
                decoration_anchors=channels["decoration_anchors"],
                decoration_atoms=channels["decoration_atoms"],
                decoration_bonds=channels["decoration_bonds"],
            )
            for key, source in (
                ("offspring", offspring_source),
                ("nodes", atom_source),
                ("parent_bonds", bond_source),
            ):
                channels[key] = _rstar_step(
                    channels[key],
                    predictions[key].softmax(dim=-1),
                    source,
                    t_value,
                    1.0 / current.sample_steps,
                    mask,
                    generator,
                )
            for key, source in (
                ("decoration_anchors", anchor_source[:, None, :].expand(-1, slots, -1)),
                ("decoration_atoms", current.sources["decoration_atoms"]),
                ("decoration_bonds", current.sources["decoration_bonds"]),
            ):
                channels[key] = _rstar_step(
                    channels[key],
                    predictions[key].softmax(dim=-1),
                    source,
                    t_value,
                    1.0 / current.sample_steps,
                    torch.ones_like(channels[key], dtype=torch.bool),
                    generator,
                )
    return UgiJointSparseTrajectoryState(
        programs=current.programs,
        layout=current.layout,
        sources=current.sources,
        channels=channels,
        sample_steps=current.sample_steps,
        step=target_step,
        generator_state=generator.get_state().clone(),
        device=current.device,
    )


def finalize_ugi_joint_sparse_state(
    model: Any,
    trajectory: UgiJointSparseTrajectoryState,
    *,
    tree_generator_state: Any,
    allowed_ring_sizes: Sequence[int] = (5, 6),
    maximum_heavy_degree: int = 4,
    maximum_adjacent_branch_runs: Sequence[int | None] | None = None,
) -> UgiJointSparseFinalization:
    """Constrained-decode one fully advanced categorical trajectory."""

    if torch is None or trajectory.step != trajectory.sample_steps:
        raise UgiJointSparseFlowError("only a fully advanced trajectory can be finalized")
    resolved_branch_runs = _resolved_branch_runs(maximum_adjacent_branch_runs)
    resolved_device = torch.device(trajectory.device)
    tree_generator = torch.Generator()
    tree_generator.set_state(tree_generator_state)
    local = trajectory.programs
    layout = trajectory.layout
    state = trajectory.channels
    roles = layout["role_states"]
    mask = layout["node_mask"]
    model.eval()
    with torch.no_grad():
        terminal = model(
            offspring=state["offspring"],
            nodes=state["nodes"],
            parent_bonds=state["parent_bonds"],
            role_states=roles,
            within_role_positions=layout["within_role_positions"],
            programs=layout["programs"],
            node_mask=mask,
            t=torch.ones(len(local), dtype=torch.float32, device=resolved_device),
            decoration_anchors=state["decoration_anchors"],
            decoration_atoms=state["decoration_atoms"],
            decoration_bonds=state["decoration_bonds"],
        )
        constrained_offspring = state["offspring"].clone()
        component_values: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        generated_attachments: list[tuple[int, int, int]] = []
        for batch_index, program in enumerate(local):
            sequence_offset = 0
            components: list[np.ndarray] = []
            attachments: list[int] = []
            for role_index, node_count in enumerate(program.node_counts):
                logits = terminal["offspring"][
                    batch_index, sequence_offset : sequence_offset + node_count
                ].cpu()
                if model.conditioning_mode == "size_only":
                    attachment_logits = terminal["attachment_counts"][batch_index, role_index].cpu()
                    allowed = torch.arange(attachment_logits.numel())
                    allowed = allowed[
                        (allowed >= 1)
                        & (allowed <= min(model.maximum_attachment_count, node_count))
                    ]
                    if not allowed.numel():
                        raise UgiJointSparseFlowError(
                            "size-only sample has no feasible attachment count"
                        )
                    selected = int(
                        torch.multinomial(
                            attachment_logits[allowed].softmax(dim=0),
                            1,
                            generator=tree_generator,
                        )
                    )
                    attachment_count = int(allowed[selected])
                    values = sample_attached_offspring_without_budget(
                        logits,
                        attachment_count=attachment_count,
                        generator=tree_generator,
                        maximum_adjacent_branch_run=resolved_branch_runs[role_index],
                    )
                else:
                    attachment_count = program.attachment_counts[role_index]
                    if program.cycle_ranks[role_index]:
                        values = sample_attached_offspring_with_exact_budget_and_cycle_rank(
                            logits,
                            junction_budget=program.junction_budgets[role_index],
                            cycle_rank=program.cycle_ranks[role_index],
                            attachment_count=attachment_count,
                            generator=tree_generator,
                            allowed_ring_sizes=allowed_ring_sizes,
                            maximum_heavy_degree=maximum_heavy_degree,
                            maximum_adjacent_branch_run=resolved_branch_runs[role_index],
                        )
                    else:
                        values = sample_attached_offspring_with_exact_budget(
                            logits,
                            junction_budget=program.junction_budgets[role_index],
                            attachment_count=attachment_count,
                            generator=tree_generator,
                            maximum_adjacent_branch_run=resolved_branch_runs[role_index],
                        )
                constrained_offspring[
                    batch_index, sequence_offset : sequence_offset + node_count
                ] = torch.from_numpy(values).to(resolved_device)
                components.append(values)
                attachments.append(attachment_count)
                sequence_offset += node_count
            component_values.append(tuple(components))  # type: ignore[arg-type]
            generated_attachments.append(tuple(attachments))  # type: ignore[arg-type]
        final = model(
            offspring=constrained_offspring,
            nodes=state["nodes"],
            parent_bonds=state["parent_bonds"],
            role_states=roles,
            within_role_positions=layout["within_role_positions"],
            programs=layout["programs"],
            node_mask=mask,
            t=torch.ones(len(local), dtype=torch.float32, device=resolved_device),
            decoration_anchors=state["decoration_anchors"],
            decoration_atoms=state["decoration_atoms"],
            decoration_bonds=state["decoration_bonds"],
        )
        output_programs = []
        for batch_index, program in enumerate(local):
            if model.conditioning_mode == "size_only":
                from forge.model.sampling.ugi_closures import feasible_next_closures

                cycle_ranks = []
                for role_index, offspring in enumerate(component_values[batch_index]):
                    cycle_logits = final["cycle_ranks"][batch_index, role_index].cpu()
                    remaining = list(range(model.maximum_cycle_rank + 1))
                    while remaining:
                        allowed = torch.as_tensor(remaining, dtype=torch.long)
                        selected = int(
                            torch.multinomial(
                                cycle_logits[allowed].softmax(dim=0),
                                1,
                                generator=tree_generator,
                            )
                        )
                        cycle_rank = int(allowed[selected])
                        if cycle_rank == 0:
                            cycle_ranks.append(0)
                            break
                        candidates = feasible_next_closures(
                            offspring,
                            remaining_closures_including_next=cycle_rank,
                            attachment_count=generated_attachments[batch_index][role_index],
                            allowed_ring_sizes=allowed_ring_sizes,
                            maximum_heavy_degree=maximum_heavy_degree,
                        )
                        if candidates.edges:
                            cycle_ranks.append(cycle_rank)
                            break
                        remaining.remove(cycle_rank)
                    else:  # pragma: no cover - rank zero is always feasible
                        raise UgiJointSparseFlowError("size-only sample has no feasible cycle rank")
                output_programs.append(
                    UgiMorphologyProgram(
                        node_counts=program.node_counts,
                        junction_budgets=tuple(
                            int(attached_tree_junction_contributions(values).sum())
                            for values in component_values[batch_index]
                        ),
                        cycle_ranks=tuple(cycle_ranks),  # type: ignore[arg-type]
                        attachment_counts=generated_attachments[batch_index],
                    )
                )
            else:
                output_programs.append(program)
        outputs = []
        for batch_index, program in enumerate(output_programs):
            count = program.node_count
            outputs.append(
                UgiJointSparseTerminal(
                    program=program,
                    offspring=component_values[batch_index],
                    atom_logits=final["nodes"][batch_index, :count].cpu().numpy(),
                    parent_bond_logits=final["parent_bonds"][batch_index, :count].cpu().numpy(),
                    decoration_anchor_logits=final["decoration_anchors"][
                        batch_index, :, : count + 1
                    ]
                    .cpu()
                    .numpy(),
                    decoration_atom_logits=final["decoration_atoms"][batch_index].cpu().numpy(),
                    decoration_bond_logits=final["decoration_bonds"][batch_index].cpu().numpy(),
                    hidden=final["hidden"][batch_index, :count].cpu().numpy(),
                    flow_endpoint_atom_states=state["nodes"][batch_index, :count].cpu().numpy(),
                    flow_endpoint_parent_bond_states=state["parent_bonds"][batch_index, :count]
                    .cpu()
                    .numpy(),
                    flow_endpoint_decoration_anchors=state["decoration_anchors"][batch_index]
                    .cpu()
                    .numpy(),
                    flow_endpoint_decoration_atom_states=state["decoration_atoms"][batch_index]
                    .cpu()
                    .numpy(),
                    flow_endpoint_decoration_bond_states=state["decoration_bonds"][batch_index]
                    .cpu()
                    .numpy(),
                )
            )
    return UgiJointSparseFinalization(
        terminals=tuple(outputs),
        tree_generator_state=tree_generator.get_state().clone(),
    )


def sample_restartable_terminals(
    model: Any,
    programs: Sequence[UgiMorphologyProgram],
    source_marginals: dict[str, np.ndarray],
    *,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
    allowed_ring_sizes: Sequence[int] = (5, 6),
    maximum_heavy_degree: int = 4,
    maximum_adjacent_branch_runs: Sequence[int | None] | None = None,
) -> tuple[list[UgiJointSparseTerminal], dict[str, Any]]:
    """Execute the legacy zero-guidance schedule through restartable states."""

    if torch is None or not programs or sample_steps < 2 or batch_size < 1:
        raise UgiJointSparseFlowError("invalid joint sparse sampling request")
    resolved_branch_runs = _resolved_branch_runs(maximum_adjacent_branch_runs)
    resolved_device = torch.device(device)
    flow_generator = torch.Generator(device=resolved_device).manual_seed(seed)
    tree_generator = torch.Generator().manual_seed(seed + 1)
    flow_generator_state = flow_generator.get_state().clone()
    tree_generator_state = tree_generator.get_state().clone()
    outputs: list[UgiJointSparseTerminal] = []
    model.eval()
    for offset in range(0, len(programs), batch_size):
        local = tuple(programs[offset : offset + batch_size])
        trajectory = initialize_ugi_joint_sparse_state(
            model,
            local,
            source_marginals,
            sample_steps=sample_steps,
            device=device,
            generator_state=flow_generator_state,
        )
        trajectory = advance_ugi_joint_sparse_state(
            model,
            trajectory,
            target_step=sample_steps,
        )
        flow_generator_state = trajectory.generator_state.clone()
        finalization = finalize_ugi_joint_sparse_state(
            model,
            trajectory,
            tree_generator_state=tree_generator_state,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
            maximum_adjacent_branch_runs=resolved_branch_runs,
        )
        outputs.extend(finalization.terminals)
        tree_generator_state = finalization.tree_generator_state.clone()
    return outputs, {
        "samples": len(outputs),
        "sample_steps": sample_steps,
        "terminal_tree_repairs": 0,
        "maximum_adjacent_branch_runs": list(resolved_branch_runs),
        "conditioning_mode": model.conditioning_mode,
        "program_fields_supplied": (
            ["node_counts", "junction_budgets", "cycle_ranks", "attachment_counts"]
            if model.conditioning_mode == "full_morphology"
            else ["node_counts"]
        ),
        "program_fields_generated": (
            []
            if model.conditioning_mode == "full_morphology"
            else ["junction_budgets", "cycle_ranks", "attachment_counts"]
        ),
        "jointly_flowed_channels": [
            "offspring",
            "atom_state",
            "parent_bond",
            "decoration_anchor",
            "decoration_atom",
            "decoration_bond",
        ],
    }
