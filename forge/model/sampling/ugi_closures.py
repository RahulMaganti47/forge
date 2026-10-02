"""Sparse closure-edge placement for Ugi precursor exterior trees.

The Ugi morphology flow generates exact connected exterior trees and declares a
small cycle rank for each precursor role.  This module places only those K
non-tree edges.  It never predicts a dense bond/no-bond matrix: candidates are
the bounded tree-distance pairs that can form a supported ring while retaining
enough coarse heavy-degree capacity for every remaining closure.

The original qualified Ugi component corpus contains only five- and six-member
rings.  Expanded component support is read from its own frozen config and may
add an empirically observed size without opening arbitrary macrocycle support.
Bond order and exact atom valence remain the responsibility of the downstream
chemistry flow.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.model.representation.ugi_morphology import preorder_attached_forest_to_parents
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None


class UgiClosurePlacementError(RuntimeError):
    """Raised when a sparse Ugi closure program is invalid."""


Edge = tuple[int, int]


@dataclass(frozen=True)
class ClosureCandidateSet:
    """Feasible next closure edges and their tree-derived features."""

    edges: tuple[Edge, ...]
    ring_sizes: np.ndarray
    left_degrees: np.ndarray
    right_degrees: np.ndarray


def _tree_state(
    offspring: np.ndarray,
    *,
    attachment_count: int = 1,
) -> tuple[np.ndarray, tuple[Edge, ...], np.ndarray, np.ndarray]:
    if offspring.ndim != 1 or offspring.size < 1 or np.any(offspring < 0):
        raise UgiClosurePlacementError("offspring must be a nonempty nonnegative vector")
    parents = preorder_attached_forest_to_parents(
        offspring.astype(np.int64, copy=False),
        attachment_count=attachment_count,
    )
    tree_edges = tuple(
        (int(parents[child]), child) for child in range(offspring.size) if int(parents[child]) >= 0
    )
    virtual_core = int(offspring.size)
    adjacency = [[] for _ in range(offspring.size + 1)]
    for left, right in tree_edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    for root in np.flatnonzero(parents < 0).tolist():
        adjacency[virtual_core].append(int(root))
        adjacency[int(root)].append(virtual_core)
    # Every exterior root has one additional fixed-core parent.  Hence every
    # exterior atom has heavy tree degree offspring + 1 in the complete lipid.
    complete_tree_degrees = offspring.astype(np.int64, copy=True) + 1
    distances = np.full((offspring.size, offspring.size), -1, dtype=np.int64)
    for start in range(offspring.size):
        local = np.full(offspring.size + 1, -1, dtype=np.int64)
        local[start] = 0
        queue = [start]
        for node in queue:
            for neighbor in adjacency[node]:
                if local[neighbor] < 0:
                    local[neighbor] = local[node] + 1
                    queue.append(neighbor)
        distances[start] = local[: offspring.size]
    return parents, tree_edges, complete_tree_degrees, distances


def _normalize_selected(selected: Iterable[Sequence[int]], node_count: int) -> tuple[Edge, ...]:
    output = []
    for raw in selected:
        if len(raw) != 2:
            raise UgiClosurePlacementError("closure edge must contain two endpoints")
        left, right = sorted((int(raw[0]), int(raw[1])))
        if not 0 <= left < right < node_count:
            raise UgiClosurePlacementError("closure endpoint lies outside the exterior tree")
        output.append((left, right))
    if len(output) != len(set(output)):
        raise UgiClosurePlacementError("closure prefix contains duplicate edges")
    return tuple(output)


def _raw_candidates(
    offspring: np.ndarray,
    *,
    selected: tuple[Edge, ...],
    allowed_ring_sizes: tuple[int, ...],
    maximum_heavy_degree: int,
    attachment_count: int,
) -> tuple[list[Edge], np.ndarray, np.ndarray, np.ndarray]:
    _, tree_edges, base_degrees, distances = _tree_state(
        offspring,
        attachment_count=attachment_count,
    )
    occupied = set(tree_edges) | set(selected)
    degrees = base_degrees.copy()
    for left, right in selected:
        degrees[left] += 1
        degrees[right] += 1
    edges: list[Edge] = []
    ring_sizes: list[int] = []
    left_degrees: list[int] = []
    right_degrees: list[int] = []
    allowed = set(allowed_ring_sizes)
    for left in range(offspring.size - 1):
        if degrees[left] >= maximum_heavy_degree:
            continue
        for right in range(left + 1, offspring.size):
            edge = (left, right)
            ring_size = int(distances[left, right]) + 1
            if (
                edge in occupied
                or degrees[right] >= maximum_heavy_degree
                or ring_size not in allowed
            ):
                continue
            edges.append(edge)
            ring_sizes.append(ring_size)
            left_degrees.append(int(degrees[left]))
            right_degrees.append(int(degrees[right]))
    return (
        edges,
        np.asarray(ring_sizes, dtype=np.int64),
        np.asarray(left_degrees, dtype=np.int64),
        np.asarray(right_degrees, dtype=np.int64),
    )


def _completion_exists(
    offspring: np.ndarray,
    *,
    selected: tuple[Edge, ...],
    remaining: int,
    allowed_ring_sizes: tuple[int, ...],
    maximum_heavy_degree: int,
    attachment_count: int,
) -> bool:
    if remaining == 0:
        return True
    candidates, _, _, _ = _raw_candidates(
        offspring,
        selected=selected,
        allowed_ring_sizes=allowed_ring_sizes,
        maximum_heavy_degree=maximum_heavy_degree,
        attachment_count=attachment_count,
    )
    if len(candidates) < remaining:
        return False
    for edge in candidates:
        if _completion_exists(
            offspring,
            selected=(*selected, edge),
            remaining=remaining - 1,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
            attachment_count=attachment_count,
        ):
            return True
    return False


def feasible_next_closures(
    offspring: np.ndarray,
    *,
    selected: Iterable[Sequence[int]] = (),
    remaining_closures_including_next: int,
    allowed_ring_sizes: Sequence[int] = (5, 6),
    maximum_heavy_degree: int = 4,
    attachment_count: int = 1,
) -> ClosureCandidateSet:
    """Enumerate only next edges that retain an exact residual completion."""

    if remaining_closures_including_next < 1:
        raise UgiClosurePlacementError("at least one closure must remain")
    normalized_sizes = tuple(sorted(set(int(value) for value in allowed_ring_sizes)))
    if not normalized_sizes or normalized_sizes[0] < 3 or maximum_heavy_degree < 2:
        raise UgiClosurePlacementError("invalid closure support")
    prefix = _normalize_selected(selected, int(offspring.size))
    tree_edges = set(_tree_state(offspring, attachment_count=attachment_count)[1])
    if any(edge in tree_edges for edge in prefix):
        raise UgiClosurePlacementError("closure prefix duplicates a tree edge")
    raw_edges, raw_sizes, raw_left, raw_right = _raw_candidates(
        offspring,
        selected=prefix,
        allowed_ring_sizes=normalized_sizes,
        maximum_heavy_degree=maximum_heavy_degree,
        attachment_count=attachment_count,
    )
    kept_edges: list[Edge] = []
    kept_sizes: list[int] = []
    kept_left: list[int] = []
    kept_right: list[int] = []
    for edge, ring_size, left_degree, right_degree in zip(
        raw_edges,
        raw_sizes.tolist(),
        raw_left.tolist(),
        raw_right.tolist(),
        strict=True,
    ):
        if _completion_exists(
            offspring,
            selected=(*prefix, edge),
            remaining=remaining_closures_including_next - 1,
            allowed_ring_sizes=normalized_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
            attachment_count=attachment_count,
        ):
            kept_edges.append(edge)
            kept_sizes.append(ring_size)
            kept_left.append(left_degree)
            kept_right.append(right_degree)
    return ClosureCandidateSet(
        edges=tuple(kept_edges),
        ring_sizes=np.asarray(kept_sizes, dtype=np.int64),
        left_degrees=np.asarray(kept_left, dtype=np.int64),
        right_degrees=np.asarray(kept_right, dtype=np.int64),
    )


if nn is not None:

    class UgiSparseClosureScorer(nn.Module):
        """Small role-aware scorer over feasible sparse closure pairs only."""

        def __init__(
            self,
            *,
            maximum_children: int,
            maximum_component_atoms: int,
            maximum_cycle_rank: int,
            maximum_ring_size: int,
            hidden_dim: int,
            layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if (
                maximum_children < 1
                or maximum_component_atoms < 3
                or maximum_cycle_rank < 1
                or maximum_ring_size < 3
                or hidden_dim < 16
                or hidden_dim % 2
                or layers < 1
                or not 0 <= dropout < 1
            ):
                raise UgiClosurePlacementError("invalid sparse closure scorer")
            self.maximum_component_atoms = maximum_component_atoms
            self.offspring_embedding = nn.Embedding(maximum_children + 1, hidden_dim)
            self.position_embedding = nn.Embedding(maximum_component_atoms, hidden_dim)
            self.depth_embedding = nn.Embedding(maximum_component_atoms + 1, hidden_dim)
            self.degree_embedding = nn.Embedding(maximum_children + 3, hidden_dim)
            self.role_embedding = nn.Embedding(len(ROLE_NAMES), hidden_dim)
            self.maximum_cycle_rank = maximum_cycle_rank
            self.encoder = nn.GRU(
                hidden_dim,
                hidden_dim // 2,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=True,
            )
            self.ring_embedding = nn.Embedding(maximum_ring_size + 1, hidden_dim)
            self.pair = nn.Sequential(
                nn.LayerNorm(5 * hidden_dim),
                nn.Linear(5 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, 1),
            )

        def forward(
            self,
            offspring: Any,
            *,
            role_index: int,
            cycle_rank: int,
            attachment_count: int = 1,
            candidates: ClosureCandidateSet,
        ) -> Any:
            if offspring.ndim != 1 or offspring.numel() > self.maximum_component_atoms:
                raise UgiClosurePlacementError("closure scorer expects one supported tree")
            if not 1 <= cycle_rank <= self.maximum_cycle_rank:
                raise UgiClosurePlacementError("cycle rank exceeds closure scorer support")
            if not candidates.edges:
                raise UgiClosurePlacementError("closure scorer requires feasible candidates")
            parents = preorder_attached_forest_to_parents(
                offspring.detach().cpu().numpy().astype(np.int64),
                attachment_count=attachment_count,
            )
            depth = np.ones(offspring.numel(), dtype=np.int64)
            for child in range(offspring.numel()):
                if int(parents[child]) >= 0:
                    depth[child] = depth[int(parents[child])] + 1
            positions = torch.arange(offspring.numel(), device=offspring.device)
            depth_tensor = torch.as_tensor(depth, dtype=torch.long, device=offspring.device)
            degrees = offspring + 1
            context = self.role_embedding(torch.tensor(role_index, device=offspring.device))
            hidden = (
                self.offspring_embedding(offspring)
                + self.position_embedding(positions)
                + self.depth_embedding(depth_tensor)
                + self.degree_embedding(degrees)
                + context
            )
            encoded, _ = self.encoder(hidden[None, :, :])
            encoded = encoded[0]
            left = torch.as_tensor(
                [edge[0] for edge in candidates.edges], dtype=torch.long, device=offspring.device
            )
            right = torch.as_tensor(
                [edge[1] for edge in candidates.edges], dtype=torch.long, device=offspring.device
            )
            ring = torch.as_tensor(candidates.ring_sizes, dtype=torch.long, device=offspring.device)
            left_hidden = encoded[left]
            right_hidden = encoded[right]
            pair_state = torch.cat(
                (
                    left_hidden,
                    right_hidden,
                    torch.abs(left_hidden - right_hidden),
                    left_hidden * right_hidden,
                    self.ring_embedding(ring),
                ),
                dim=-1,
            )
            return self.pair(pair_state).squeeze(-1)

else:  # pragma: no cover

    class UgiSparseClosureScorer:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise UgiClosurePlacementError("sparse closure scoring requires torch")


def closure_set_loss(
    model: Any,
    offspring: Any,
    *,
    role_index: int,
    target_edges: Sequence[Sequence[int]],
    attachment_count: int = 1,
    allowed_ring_sizes: Sequence[int] = (5, 6),
    maximum_heavy_degree: int = 4,
) -> Any:
    """Permutation-tolerant teacher-forced loss for one sparse closure set."""

    if functional is None:
        raise UgiClosurePlacementError("closure loss requires torch")
    normalized_target = set(_normalize_selected(target_edges, int(offspring.numel())))
    if not normalized_target:
        raise UgiClosurePlacementError("closure placement loss requires a cyclic target")
    selected: list[Edge] = []
    losses = []
    while len(selected) < len(normalized_target):
        candidates = feasible_next_closures(
            offspring.detach().cpu().numpy().astype(np.int64),
            selected=selected,
            remaining_closures_including_next=len(normalized_target) - len(selected),
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
            attachment_count=attachment_count,
        )
        logits = model(
            offspring,
            role_index=role_index,
            cycle_rank=len(normalized_target),
            attachment_count=attachment_count,
            candidates=candidates,
        )
        remaining_targets = normalized_target - set(selected)
        target_indices = [
            index for index, edge in enumerate(candidates.edges) if edge in remaining_targets
        ]
        if not target_indices:
            raise UgiClosurePlacementError("target closure lies outside declared sparse support")
        log_probabilities = functional.log_softmax(logits, dim=0)
        indices = torch.as_tensor(target_indices, dtype=torch.long, device=offspring.device)
        losses.append(-torch.logsumexp(log_probabilities[indices], dim=0))
        selected.append(sorted(remaining_targets)[0])
    return torch.stack(losses).mean()


def sample_sparse_closures(
    model: Any,
    offspring: np.ndarray,
    *,
    role_index: int,
    cycle_rank: int,
    attachment_count: int = 1,
    generator: Any,
    temperature: float = 1.0,
    allowed_ring_sizes: Sequence[int] = (5, 6),
    maximum_heavy_degree: int = 4,
    device: str = "cpu",
    deterministic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample exactly K feasible closure edges with no terminal repair."""

    if torch is None or cycle_rank < 0 or temperature <= 0:
        raise UgiClosurePlacementError("invalid sparse closure sampling request")
    if cycle_rank == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy()
    resolved = torch.device(device)
    values = torch.as_tensor(offspring, dtype=torch.long, device=resolved)
    selected: list[Edge] = []
    model.eval()
    with torch.no_grad():
        for _ in range(cycle_rank):
            candidates = feasible_next_closures(
                offspring,
                selected=selected,
                remaining_closures_including_next=cycle_rank - len(selected),
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                attachment_count=attachment_count,
            )
            if not candidates.edges:
                raise UgiClosurePlacementError("generated tree cannot realize its cycle rank")
            logits = model(
                values,
                role_index=role_index,
                cycle_rank=cycle_rank,
                attachment_count=attachment_count,
                candidates=candidates,
            )
            probabilities = torch.softmax(logits / temperature, dim=0)
            choice = (
                int(torch.argmax(probabilities).item())
                if deterministic
                else int(torch.multinomial(probabilities, 1, generator=generator).item())
            )
            selected.append(candidates.edges[choice])
    selected.sort()
    return (
        np.asarray([edge[0] for edge in selected], dtype=np.int64),
        np.asarray([edge[1] for edge in selected], dtype=np.int64),
    )
