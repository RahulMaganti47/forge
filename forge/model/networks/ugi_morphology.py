"""Discrete flow over Ugi-core-anchored precursor exterior trees.

The five-atom reaction core is supplied by the Ugi adapter.  A generated global
program declares only per-role exterior size, junction budget, and cycle rank.
The neural model jointly denoises every exterior topology state at atom
resolution in a shared sequence.  No component identifier, clean graph distance, atom state,
or bond state is available to this topology denoiser.
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.model.networks.dense_flow import _rstar_step
from forge.model.representation.ugi_morphology import (
    UgiMorphologyProgram,
    UgiProductMorphology,
    attached_tree_matches_program,
    preorder_attached_forest_to_parents,
    sample_attached_offspring_with_exact_budget,
    sample_attached_offspring_with_exact_budget_and_cycle_rank,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None
    pack_padded_sequence = None
    pad_packed_sequence = None


class UgiMorphologyFlowError(RuntimeError):
    """Raised when Ugi morphology tensors violate the generated program."""


@dataclass(frozen=True)
class UgiMorphologySample:
    """Three generated precursor exterior trees around the fixed Ugi core."""

    program: UgiMorphologyProgram
    offspring: tuple[np.ndarray, np.ndarray, np.ndarray]


def program_to_array(program: UgiMorphologyProgram) -> np.ndarray:
    """Serialize count, junction, and cycle triples in a stable order."""

    return np.asarray(
        (
            *program.node_counts,
            *program.junction_budgets,
            *program.cycle_ranks,
            *program.attachment_counts,
        ),
        dtype=np.int64,
    )


def _layout_from_programs(
    programs: Sequence[UgiMorphologyProgram],
    *,
    maximum_nodes: int | None = None,
) -> dict[str, Any]:
    if torch is None:
        raise UgiMorphologyFlowError("Ugi morphology layout requires torch")
    if not programs or any(any(count < 1 for count in program.node_counts) for program in programs):
        raise UgiMorphologyFlowError("every Ugi role requires a nonempty exterior")
    required = max(program.node_count for program in programs)
    capacity = required if maximum_nodes is None else maximum_nodes
    if capacity < required:
        raise UgiMorphologyFlowError("Ugi morphology capacity is too small")
    node_mask = torch.zeros((len(programs), capacity), dtype=torch.bool)
    role_states = torch.zeros((len(programs), capacity), dtype=torch.long)
    within_role_positions = torch.zeros_like(role_states)
    for batch_index, program in enumerate(programs):
        offset = 0
        for role_index, count in enumerate(program.node_counts):
            node_mask[batch_index, offset : offset + count] = True
            role_states[batch_index, offset : offset + count] = role_index
            within_role_positions[batch_index, offset : offset + count] = torch.arange(count)
            offset += count
    return {
        "node_mask": node_mask,
        "role_states": role_states,
        "within_role_positions": within_role_positions,
        "programs": torch.from_numpy(np.stack([program_to_array(value) for value in programs])),
    }


def collate_ugi_morphology_records(
    records: Sequence[UgiProductMorphology],
    *,
    maximum_nodes: int,
    maximum_children: int,
) -> dict[str, Any]:
    """Collate topology-only targets with role layout derived from the program."""

    if not records:
        raise UgiMorphologyFlowError("Ugi morphology collation requires records")
    programs = tuple(record.program for record in records)
    batch = _layout_from_programs(programs, maximum_nodes=maximum_nodes)
    offspring = torch.zeros_like(batch["role_states"])
    for batch_index, record in enumerate(records):
        offset = 0
        for component in record.components:
            if component.offspring.max(initial=0) > maximum_children:
                raise UgiMorphologyFlowError("offspring count exceeds declared support")
            offspring[batch_index, offset : offset + component.node_count] = torch.from_numpy(
                component.offspring.copy()
            )
            offset += component.node_count
    batch["offspring"] = offspring
    return batch


def _pending_by_role(
    offspring: Any,
    role_states: Any,
    node_mask: Any,
    maximum_nodes: int,
    attachment_counts: Any,
) -> Any:
    pending = torch.zeros_like(offspring)
    for role_index in range(len(ROLE_NAMES)):
        role_mask = node_mask & (role_states == role_index)
        active_delta = (offspring - 1) * role_mask
        balance = attachment_counts[:, role_index, None] + torch.cumsum(active_delta, dim=1)
        pending[role_mask] = balance.clamp(
            min=-maximum_nodes,
            max=maximum_nodes,
        )[role_mask]
    return pending + maximum_nodes


if nn is not None:

    class UgiMorphologyFlow(nn.Module):
        """Joint role-aware denoiser for the three atom-level exterior trees."""

        def __init__(
            self,
            *,
            maximum_children: int,
            maximum_component_atoms: int,
            maximum_total_atoms: int,
            maximum_junction_budget: int,
            maximum_cycle_rank: int,
            maximum_attachment_count: int | None = None,
            hidden_dim: int,
            layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if (
                maximum_children < 1
                or maximum_component_atoms < 1
                or maximum_total_atoms < 3
                or maximum_junction_budget < 0
                or maximum_cycle_rank < 0
                or (maximum_attachment_count is not None and maximum_attachment_count < 1)
                or hidden_dim < 16
                or hidden_dim % 2
                or layers < 1
                or not 0 <= dropout < 1
            ):
                raise UgiMorphologyFlowError("invalid Ugi morphology architecture")
            self.maximum_children = maximum_children
            self.maximum_component_atoms = maximum_component_atoms
            self.maximum_total_atoms = maximum_total_atoms
            self.maximum_junction_budget = maximum_junction_budget
            self.maximum_cycle_rank = maximum_cycle_rank
            self.maximum_attachment_count = maximum_attachment_count
            self.offspring_embedding = nn.Embedding(maximum_children + 1, hidden_dim)
            self.role_embedding = nn.Embedding(len(ROLE_NAMES), hidden_dim)
            self.within_role_position_embedding = nn.Embedding(
                maximum_component_atoms,
                hidden_dim,
            )
            self.pending_embedding = nn.Embedding(2 * maximum_total_atoms + 1, hidden_dim)
            self.count_embeddings = nn.ModuleList(
                nn.Embedding(maximum_component_atoms + 1, hidden_dim) for _ in ROLE_NAMES
            )
            self.junction_embeddings = nn.ModuleList(
                nn.Embedding(maximum_junction_budget + 1, hidden_dim) for _ in ROLE_NAMES
            )
            self.cycle_embeddings = nn.ModuleList(
                nn.Embedding(maximum_cycle_rank + 1, hidden_dim) for _ in ROLE_NAMES
            )
            self.attachment_embeddings = (
                nn.ModuleList(
                    nn.Embedding(maximum_attachment_count + 1, hidden_dim) for _ in ROLE_NAMES
                )
                if maximum_attachment_count is not None
                else None
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.sequence = nn.GRU(
                hidden_dim,
                hidden_dim // 2,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=True,
            )
            self.output = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.offspring_output = nn.Linear(hidden_dim, maximum_children + 1)

        def _program_context(self, programs: Any) -> Any:
            if programs.ndim != 2 or programs.shape[1] not in {9, 12}:
                raise UgiMorphologyFlowError("program tensor must be [batch, 9 or 12]")
            counts = programs[:, :3]
            junctions = programs[:, 3:6]
            cycles = programs[:, 6:9]
            attachments = programs[:, 9:12] if programs.shape[1] == 12 else torch.ones_like(counts)
            if (
                torch.any(counts < 1)
                or torch.any(counts > self.maximum_component_atoms)
                or torch.any(junctions < 0)
                or torch.any(junctions > self.maximum_junction_budget)
                or torch.any(cycles < 0)
                or torch.any(cycles > self.maximum_cycle_rank)
                or torch.any(attachments < 1)
                or (
                    self.maximum_attachment_count is not None
                    and torch.any(attachments > self.maximum_attachment_count)
                )
            ):
                raise UgiMorphologyFlowError("program value exceeds model support")
            context = torch.zeros(
                (programs.shape[0], self.offspring_embedding.embedding_dim),
                dtype=self.offspring_embedding.weight.dtype,
                device=programs.device,
            )
            for role_index in range(len(ROLE_NAMES)):
                context = (
                    context
                    + self.count_embeddings[role_index](counts[:, role_index])
                    + self.junction_embeddings[role_index](junctions[:, role_index])
                    + self.cycle_embeddings[role_index](cycles[:, role_index])
                )
                if self.attachment_embeddings is not None:
                    context = context + self.attachment_embeddings[role_index](
                        attachments[:, role_index]
                    )
            return context

        def forward(
            self,
            offspring: Any,
            role_states: Any,
            within_role_positions: Any,
            programs: Any,
            t: Any,
            node_mask: Any,
        ) -> dict[str, Any]:
            shape = offspring.shape
            if (
                offspring.ndim != 2
                or role_states.shape != shape
                or within_role_positions.shape != shape
                or node_mask.shape != shape
                or t.shape != (shape[0],)
            ):
                raise UgiMorphologyFlowError("Ugi morphology state shapes do not agree")
            if torch.any(within_role_positions[node_mask] >= self.maximum_component_atoms):
                raise UgiMorphologyFlowError("within-role position exceeds model support")
            pending = _pending_by_role(
                offspring,
                role_states,
                node_mask,
                self.maximum_total_atoms,
                programs[:, 9:12] if programs.shape[1] == 12 else torch.ones_like(programs[:, :3]),
            )
            hidden = (
                self.offspring_embedding(offspring)
                + self.role_embedding(role_states)
                + self.within_role_position_embedding(within_role_positions)
                + self.pending_embedding(pending)
                + self._program_context(programs)[:, None, :]
                + self.time_embedding(t[:, None])[:, None, :]
            )
            hidden = self.input_norm(hidden) * node_mask[:, :, None]
            lengths = node_mask.sum(dim=1).to("cpu")
            packed = pack_padded_sequence(
                hidden,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.sequence(packed)
            hidden, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=offspring.shape[1],
            )
            hidden = (hidden + self.output(hidden)) * node_mask[:, :, None]
            return {"offspring": self.offspring_output(hidden)}

else:  # pragma: no cover

    class UgiMorphologyFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise UgiMorphologyFlowError("Ugi morphology flow requires torch")


def noise_ugi_morphology_batch(
    clean: dict[str, Any],
    source_marginals: Any,
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    """Corrupt topology with role-conditioned, smoothed full-support sources."""

    mask = clean["node_mask"]
    if source_marginals.shape[0] != len(ROLE_NAMES):
        raise UgiMorphologyFlowError("source marginals must have one row per Ugi role")
    example_indices = torch.arange(mask.shape[0], device=mask.device)[:, None].expand_as(mask)[mask]
    roles = clean["role_states"][mask]
    probabilities = source_marginals[roles].clone()
    probabilities *= 1.0 - t[example_indices, None]
    probabilities.scatter_add_(
        1,
        clean["offspring"][mask][:, None],
        t[example_indices, None],
    )
    noisy = clean["offspring"].clone()
    noisy[mask] = torch.multinomial(
        probabilities,
        1,
        generator=generator,
    ).squeeze(1)
    return {"offspring": noisy}


def ugi_morphology_flow_loss(
    predictions: dict[str, Any],
    clean: dict[str, Any],
    *,
    label_smoothing: float = 0.0,
) -> tuple[Any, dict[str, float]]:
    """Use equal precursor-role loss mass despite unequal exterior lengths."""

    if not 0.0 <= label_smoothing < 1.0:
        raise UgiMorphologyFlowError("label smoothing must be in [0, 1)")

    token_losses = functional.cross_entropy(
        predictions["offspring"].transpose(1, 2),
        clean["offspring"],
        reduction="none",
        label_smoothing=label_smoothing,
    )
    role_losses = []
    metrics: dict[str, float] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        mask = clean["node_mask"] & (clean["role_states"] == role_index)
        if not bool(mask.any()):
            raise UgiMorphologyFlowError(f"batch lacks {role} topology targets")
        value = token_losses[mask].mean()
        role_losses.append(value)
        metrics[f"{role}_offspring_ce"] = float(value.detach())
    total = torch.stack(role_losses).mean()
    metrics["total"] = float(total.detach())
    return total, metrics


def sample_ugi_morphologies(
    model: Any,
    programs: Sequence[UgiMorphologyProgram],
    source_marginals: np.ndarray,
    *,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
    maximum_adjacent_branch_runs: Sequence[int] | None = None,
) -> tuple[list[UgiMorphologySample], dict[str, Any]]:
    """Sample three exact trees per product with no terminal repair."""

    if torch is None:
        raise UgiMorphologyFlowError("Ugi morphology sampling requires torch")
    if not programs or sample_steps < 2 or batch_size < 1:
        raise UgiMorphologyFlowError("invalid Ugi morphology sampling request")
    if maximum_adjacent_branch_runs is not None and (
        len(maximum_adjacent_branch_runs) != len(ROLE_NAMES)
        or any(value < 0 for value in maximum_adjacent_branch_runs)
    ):
        raise UgiMorphologyFlowError("invalid role-specific branch-run support")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    sources = torch.as_tensor(source_marginals, dtype=torch.float32, device=resolved_device)
    output: list[UgiMorphologySample] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(programs), batch_size):
            local = tuple(programs[offset : offset + batch_size])
            layout = {
                key: value.to(resolved_device)
                for key, value in _layout_from_programs(local).items()
            }
            offspring = torch.zeros_like(layout["role_states"])
            for role_index in range(len(ROLE_NAMES)):
                mask = layout["node_mask"] & (layout["role_states"] == role_index)
                offspring[mask] = torch.multinomial(
                    sources[role_index],
                    int(mask.sum()),
                    replacement=True,
                    generator=generator,
                )
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full(
                    (len(local),),
                    t_value,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                predictions = model(
                    offspring,
                    layout["role_states"],
                    layout["within_role_positions"],
                    layout["programs"],
                    t,
                    layout["node_mask"],
                )
                role_sources = sources[layout["role_states"]]
                offspring = _rstar_step(
                    offspring,
                    predictions["offspring"].softmax(dim=-1),
                    role_sources,
                    t_value,
                    1.0 / sample_steps,
                    layout["node_mask"],
                    generator,
                )
            terminal = model(
                offspring,
                layout["role_states"],
                layout["within_role_positions"],
                layout["programs"],
                torch.ones(len(local), dtype=torch.float32, device=resolved_device),
                layout["node_mask"],
            )["offspring"]
            for batch_index, program in enumerate(local):
                components = []
                component_offset = 0
                for role_index, node_count in enumerate(program.node_counts):
                    logits = (
                        terminal[
                            batch_index,
                            component_offset : component_offset + node_count,
                        ]
                        .detach()
                        .cpu()
                    )
                    cycle_rank = program.cycle_ranks[role_index]
                    if cycle_rank:
                        components.append(
                            sample_attached_offspring_with_exact_budget_and_cycle_rank(
                                logits,
                                junction_budget=program.junction_budgets[role_index],
                                cycle_rank=cycle_rank,
                                attachment_count=program.attachment_counts[role_index],
                                maximum_adjacent_branch_run=(
                                    maximum_adjacent_branch_runs[role_index]
                                    if maximum_adjacent_branch_runs is not None
                                    else None
                                ),
                                generator=cpu_generator,
                            )
                        )
                    else:
                        components.append(
                            sample_attached_offspring_with_exact_budget(
                                logits,
                                junction_budget=program.junction_budgets[role_index],
                                attachment_count=program.attachment_counts[role_index],
                                maximum_adjacent_branch_run=(
                                    maximum_adjacent_branch_runs[role_index]
                                    if maximum_adjacent_branch_runs is not None
                                    else None
                                ),
                                generator=cpu_generator,
                            )
                        )
                    component_offset += node_count
                output.append(
                    UgiMorphologySample(
                        program=program,
                        offspring=tuple(components),  # type: ignore[arg-type]
                    )
                )
    elapsed = time.perf_counter() - start
    return output, {
        "samples": len(output),
        "sampling_steps": sample_steps,
        "wall_seconds": elapsed,
        "graph_steps_per_second": len(output) * sample_steps / elapsed,
        "terminal_tree_repairs": 0,
        "closure_feasibility_in_terminal_support": True,
        "clean_target_graph_distances_used": False,
        "component_catalog_ids_used": False,
        "maximum_adjacent_branch_runs": (
            list(maximum_adjacent_branch_runs) if maximum_adjacent_branch_runs is not None else None
        ),
    }


def ugi_morphology_statistics(samples: Sequence[UgiMorphologySample]) -> dict[str, Any]:
    """Summarize morphology quality by chemically meaningful precursor role."""

    if not samples:
        raise UgiMorphologyFlowError("Ugi morphology statistics require samples")
    role_metrics: dict[str, dict[str, Any]] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        maximum_depths = []
        branch_atoms = []
        adjacent_branch_runs = []
        signatures = set()
        offspring_histogram: Counter[int] = Counter()
        for sample in samples:
            values = sample.offspring[role_index]
            if not attached_tree_matches_program(
                values,
                node_count=sample.program.node_counts[role_index],
                junction_budget=sample.program.junction_budgets[role_index],
                attachment_count=sample.program.attachment_counts[role_index],
            ):
                raise UgiMorphologyFlowError("sample violates its exact role program")
            parents = preorder_attached_forest_to_parents(
                values,
                attachment_count=sample.program.attachment_counts[role_index],
            )
            depths = np.ones(values.size, dtype=np.int64)
            for child in range(values.size):
                if int(parents[child]) >= 0:
                    depths[child] = depths[int(parents[child])] + 1
            branches = values >= 2
            longest_run = 0
            current_run = 0
            for is_branch in branches.tolist():
                current_run = current_run + 1 if is_branch else 0
                longest_run = max(longest_run, current_run)
            maximum_depths.append(int(depths.max(initial=1)))
            branch_atoms.append(int(branches.sum()))
            adjacent_branch_runs.append(longest_run)
            signatures.add(tuple(values.tolist()))
            offspring_histogram.update(int(value) for value in values)
        role_metrics[role] = {
            "mean_maximum_port_distance": float(np.mean(maximum_depths)),
            "mean_branch_atoms": float(np.mean(branch_atoms)),
            "maximum_adjacent_branch_run": int(max(adjacent_branch_runs)),
            "unique_tree_fraction": len(signatures) / len(samples),
            "offspring_fractions": {
                str(value): count / sum(offspring_histogram.values())
                for value, count in sorted(offspring_histogram.items())
            },
        }
    return {"samples": len(samples), "by_role": role_metrics}
