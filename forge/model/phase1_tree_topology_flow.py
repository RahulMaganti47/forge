"""Discrete flow over exact breadth-first offspring encodings of lipid trees."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.model.defog_feasibility import _rstar_step

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None


class TreeTopologyFlowError(RuntimeError):
    """Raised when a rooted tree-flow contract is violated."""


@dataclass(frozen=True)
class TreeTopologySample:
    """One exact rooted spanning tree and its BFS offspring sequence."""

    offspring: np.ndarray
    parents: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.offspring.size)


def record_offspring_counts(record: Any) -> np.ndarray:
    """Convert a canonical BFS parent sequence to exact offspring counts."""

    offspring = np.zeros(record.node_count, dtype=np.int64)
    previous_parent = 0
    for child in range(1, record.node_count):
        parent = int(record.parents[child])
        if parent < previous_parent or parent >= child:
            raise TreeTopologyFlowError("training parent sequence is not canonical breadth first")
        offspring[parent] += 1
        previous_parent = parent
    if offspring.sum() != record.node_count - 1:
        raise TreeTopologyFlowError("offspring sequence does not encode one tree")
    return offspring


def offspring_to_parents(offspring: np.ndarray) -> np.ndarray:
    """Decode one valid BFS offspring sequence to its unique parent sequence."""

    if offspring.ndim != 1 or offspring.size < 1 or np.any(offspring < 0):
        raise TreeTopologyFlowError("invalid offspring sequence")
    node_count = int(offspring.size)
    parents = np.zeros(node_count, dtype=np.int64)
    next_child = 1
    queue = 1
    for parent, child_count in enumerate(offspring):
        queue -= 1
        for _ in range(int(child_count)):
            if next_child >= node_count:
                raise TreeTopologyFlowError("offspring sequence overfills tree")
            parents[next_child] = parent
            next_child += 1
            queue += 1
        if parent < node_count - 1 and queue <= 0:
            raise TreeTopologyFlowError("offspring sequence exhausts BFS queue early")
    if next_child != node_count or queue != 0:
        raise TreeTopologyFlowError("offspring sequence does not close one tree")
    return parents


def preorder_offspring_to_parents(offspring: np.ndarray) -> np.ndarray:
    """Decode a valid depth-first preorder offspring word to unique parents."""

    if offspring.ndim != 1 or offspring.size < 1 or np.any(offspring < 0):
        raise TreeTopologyFlowError("invalid preorder offspring sequence")
    node_count = int(offspring.size)
    parents = np.zeros(node_count, dtype=np.int64)
    stack: list[list[int]] = [[0, int(offspring[0])]]
    for node in range(1, node_count):
        while stack and stack[-1][1] == 0:
            stack.pop()
        if not stack:
            raise TreeTopologyFlowError("preorder offspring sequence exhausts tree early")
        parents[node] = stack[-1][0]
        stack[-1][1] -= 1
        stack.append([node, int(offspring[node])])
    while stack and stack[-1][1] == 0:
        stack.pop()
    if stack or int(offspring.sum()) != node_count - 1:
        raise TreeTopologyFlowError("preorder offspring sequence does not close one tree")
    return parents


def preorder_record_offspring_counts(record: Any) -> np.ndarray:
    """Extract child counts from a canonical depth-first preorder record."""

    offspring = np.zeros(record.node_count, dtype=np.int64)
    for child in range(1, record.node_count):
        parent = int(record.parents[child])
        if not 0 <= parent < child:
            raise TreeTopologyFlowError("preorder parent must precede its child")
        offspring[parent] += 1
    if not np.array_equal(preorder_offspring_to_parents(offspring), record.parents):
        raise TreeTopologyFlowError("record is not a canonical depth-first preorder")
    return offspring


def collate_tree_records(
    records: Sequence[Any],
    *,
    maximum_nodes: int,
    maximum_children: int,
) -> dict[str, Any]:
    """Collate exact tree sequences while omitting atom and bond chemistry."""

    if torch is None:
        raise TreeTopologyFlowError("tree collation requires torch")
    if not records:
        raise TreeTopologyFlowError("tree collation requires records")
    if maximum_nodes < max(record.node_count for record in records):
        raise TreeTopologyFlowError("tree node capacity is too small")
    offspring = torch.zeros((len(records), maximum_nodes), dtype=torch.long)
    node_mask = torch.zeros((len(records), maximum_nodes), dtype=torch.bool)
    for index, record in enumerate(records):
        counts = record_offspring_counts(record)
        if counts.max(initial=0) > maximum_children:
            raise TreeTopologyFlowError("offspring count exceeds declared support")
        offspring[index, : record.node_count] = torch.from_numpy(counts)
        node_mask[index, : record.node_count] = True
    return {"offspring": offspring, "node_mask": node_mask}


def offspring_marginal(
    records: Sequence[Any],
    *,
    maximum_children: int,
    probability_floor: float = 1e-6,
) -> np.ndarray:
    """Estimate the R0 categorical source over per-atom offspring counts."""

    if maximum_children < 1 or not 0 < probability_floor < 1:
        raise TreeTopologyFlowError("invalid offspring marginal support")
    counts = np.full(maximum_children + 1, probability_floor, dtype=np.float64)
    for record in records:
        values = record_offspring_counts(record)
        if values.max(initial=0) > maximum_children:
            raise TreeTopologyFlowError("offspring count exceeds declared support")
        np.add.at(counts, values, 1.0)
    return counts / counts.sum()


def queue_balance(offspring: Any, node_mask: Any, maximum_nodes: int) -> Any:
    """Return clipped BFS queue balance for a possibly noisy sequence."""

    active_delta = (offspring - 1) * node_mask
    balance = 1 + torch.cumsum(active_delta, dim=1)
    balance = balance.clamp(min=-maximum_nodes, max=maximum_nodes)
    return balance + maximum_nodes


if nn is not None:

    class GlobalTreeBlock(nn.Module):
        """Globally coordinate a complete rooted topology sequence."""

        def __init__(
            self,
            hidden_dim: int,
            attention_heads: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.attention = nn.MultiheadAttention(
                hidden_dim,
                attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)

        def forward(self, hidden: Any, node_mask: Any) -> Any:
            attended, _ = self.attention(
                hidden,
                hidden,
                hidden,
                key_padding_mask=~node_mask,
                need_weights=False,
            )
            hidden = self.attention_norm(hidden + attended)
            hidden = self.output_norm(hidden + self.feedforward(hidden))
            return hidden * node_mask[:, :, None]

    class OffspringTreeFlow(nn.Module):
        """Discrete flow over full-tree BFS offspring states."""

        def __init__(
            self,
            *,
            maximum_children: int,
            hidden_dim: int,
            layers: int,
            attention_heads: int,
            maximum_heavy_atoms: int,
            dropout: float,
        ) -> None:
            super().__init__()
            if (
                maximum_children < 1
                or hidden_dim < 8
                or layers < 1
                or attention_heads < 1
                or hidden_dim % attention_heads
                or maximum_heavy_atoms < 2
            ):
                raise TreeTopologyFlowError("invalid tree-flow architecture")
            self.maximum_children = maximum_children
            self.maximum_heavy_atoms = maximum_heavy_atoms
            self.offspring_embedding = nn.Embedding(
                maximum_children + 1,
                hidden_dim,
            )
            self.position_embedding = nn.Embedding(
                maximum_heavy_atoms,
                hidden_dim,
            )
            self.queue_embedding = nn.Embedding(
                2 * maximum_heavy_atoms + 1,
                hidden_dim,
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.blocks = nn.ModuleList(
                GlobalTreeBlock(hidden_dim, attention_heads, dropout) for _ in range(layers)
            )
            self.offspring_output = nn.Linear(
                hidden_dim,
                maximum_children + 1,
            )
            self.node_count_logits = nn.Parameter(torch.zeros(maximum_heavy_atoms + 1))

        def forward(
            self,
            offspring: Any,
            t: Any,
            node_mask: Any,
        ) -> dict[str, Any]:
            positions = torch.arange(
                offspring.shape[1],
                device=offspring.device,
            )
            balances = queue_balance(
                offspring,
                node_mask,
                self.maximum_heavy_atoms,
            )
            hidden = (
                self.offspring_embedding(offspring)
                + self.position_embedding(positions)[None, :, :]
                + self.queue_embedding(balances)
                + self.time_embedding(t[:, None])[:, None, :]
            )
            hidden *= node_mask[:, :, None]
            for block in self.blocks:
                hidden = block(hidden, node_mask)
            return {
                "offspring": self.offspring_output(hidden),
                "node_count": self.node_count_logits[None, :].expand(
                    offspring.shape[0],
                    -1,
                ),
            }

else:  # pragma: no cover

    class OffspringTreeFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise TreeTopologyFlowError("tree flow requires torch")


def tree_flow_loss(
    predictions: dict[str, Any],
    clean: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Compute masked offspring and generated node-count losses."""

    node_mask = clean["node_mask"]
    offspring_ce = functional.cross_entropy(
        predictions["offspring"][node_mask],
        clean["offspring"][node_mask],
    )
    node_count_logits = predictions["node_count"].clone()
    node_count_logits[:, :2] = -1e9
    node_count_ce = functional.cross_entropy(
        node_count_logits,
        node_mask.sum(dim=1).long(),
    )
    total = offspring_ce + node_count_ce
    return total, {
        "total": float(total.detach()),
        "offspring_ce": float(offspring_ce.detach()),
        "node_count_ce": float(node_count_ce.detach()),
    }


def noise_tree_batch(
    clean: dict[str, Any],
    source_marginal: Any,
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    """Corrupt the full offspring sequence under its categorical source."""

    node_mask = clean["node_mask"]
    active_examples = torch.arange(
        node_mask.shape[0],
        device=node_mask.device,
    )[:, None].expand_as(
        node_mask
    )[node_mask]
    probabilities = source_marginal[None, :].repeat(int(node_mask.sum()), 1)
    probabilities *= 1.0 - t[active_examples, None]
    probabilities.scatter_add_(
        1,
        clean["offspring"][node_mask][:, None],
        t[active_examples, None],
    )
    sampled = torch.multinomial(
        probabilities,
        1,
        generator=generator,
    ).squeeze(1)
    output = clean["offspring"].clone()
    output[node_mask] = sampled
    return {"offspring": output}


def _valid_offspring_choices(
    *,
    position: int,
    queue: int,
    node_count: int,
    maximum_children: int,
) -> tuple[int, ...]:
    choices = []
    for children in range(maximum_children + 1):
        next_queue = queue - 1 + children
        remaining_positions = node_count - position - 1
        if next_queue < 0 or next_queue > remaining_positions:
            continue
        if remaining_positions and next_queue == 0:
            continue
        if not remaining_positions and next_queue != 0:
            continue
        choices.append(children)
    return tuple(choices)


def sample_valid_offspring(
    logits: Any,
    *,
    generator: Any,
) -> np.ndarray:
    """Sample exactly from factorized logits conditioned on one valid BFS tree."""

    if torch is None:
        raise TreeTopologyFlowError("valid tree sampling requires torch")
    if logits.ndim != 2 or logits.shape[0] < 2:
        raise TreeTopologyFlowError("offspring logits require at least two nodes")
    node_count, classes = logits.shape
    maximum_children = classes - 1
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)
    suffix: list[dict[int, Any]] = [dict() for _ in range(node_count + 1)]
    suffix[node_count][0] = logits.new_tensor(0.0, dtype=torch.float64)
    for position in range(node_count - 1, -1, -1):
        for queue in range(1, node_count - position + 1):
            terms = []
            for children in _valid_offspring_choices(
                position=position,
                queue=queue,
                node_count=node_count,
                maximum_children=maximum_children,
            ):
                next_queue = queue - 1 + children
                if next_queue not in suffix[position + 1]:
                    continue
                terms.append(
                    log_probabilities[position, children] + suffix[position + 1][next_queue]
                )
            if terms:
                suffix[position][queue] = torch.logsumexp(
                    torch.stack(terms),
                    dim=0,
                )
    if 1 not in suffix[0]:
        raise TreeTopologyFlowError("declared offspring support cannot form tree")

    offspring = np.zeros(node_count, dtype=np.int64)
    queue = 1
    for position in range(node_count):
        choices = []
        weights = []
        for children in _valid_offspring_choices(
            position=position,
            queue=queue,
            node_count=node_count,
            maximum_children=maximum_children,
        ):
            next_queue = queue - 1 + children
            if next_queue not in suffix[position + 1]:
                continue
            choices.append(children)
            weights.append(log_probabilities[position, children] + suffix[position + 1][next_queue])
        selected = int(
            torch.multinomial(
                torch.stack(weights).softmax(dim=0),
                1,
                generator=generator,
            )
        )
        offspring[position] = choices[selected]
        queue = queue - 1 + choices[selected]
    offspring_to_parents(offspring)
    return offspring


def _sample_count_distribution(
    distribution: np.ndarray,
    *,
    sample_count: int,
    maximum_heavy_atoms: int,
    generator: Any,
    device: Any,
) -> Any:
    probabilities = torch.as_tensor(
        distribution,
        dtype=torch.float32,
        device=device,
    )
    if (
        probabilities.shape != (maximum_heavy_atoms + 1,)
        or not torch.isfinite(probabilities).all()
        or torch.any(probabilities < 0)
        or torch.any(probabilities[:2] > 0)
        or not torch.isclose(
            probabilities.sum(),
            torch.tensor(1.0, device=device),
        )
    ):
        raise TreeTopologyFlowError("invalid node-count distribution")
    return torch.multinomial(
        probabilities,
        sample_count,
        replacement=True,
        generator=generator,
    )


def sample_tree_topologies(
    model: Any,
    source_marginal: np.ndarray,
    node_count_distribution: np.ndarray,
    *,
    sample_count: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[list[TreeTopologySample], dict[str, Any]]:
    """Generate complete valid rooted tree topologies through discrete flow."""

    if sample_count < 1 or sample_steps < 2 or batch_size < 1:
        raise TreeTopologyFlowError("invalid tree sampling counts")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    cpu_generator = torch.Generator().manual_seed(seed + 1)
    source = torch.as_tensor(
        source_marginal,
        dtype=torch.float32,
        device=resolved_device,
    )
    if (
        source.shape != (model.maximum_children + 1,)
        or not torch.isfinite(source).all()
        or torch.any(source <= 0)
        or not torch.isclose(
            source.sum(),
            torch.tensor(1.0, device=resolved_device),
        )
    ):
        raise TreeTopologyFlowError("invalid offspring source marginal")
    samples = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, sample_count, batch_size):
            local_batch = min(batch_size, sample_count - offset)
            counts = _sample_count_distribution(
                node_count_distribution,
                sample_count=local_batch,
                maximum_heavy_atoms=model.maximum_heavy_atoms,
                generator=generator,
                device=resolved_device,
            )
            maximum_nodes = int(counts.max())
            node_mask = (
                torch.arange(maximum_nodes, device=resolved_device)[None, :] < counts[:, None]
            )
            state = torch.zeros(
                (local_batch, maximum_nodes),
                dtype=torch.long,
                device=resolved_device,
            )
            state[node_mask] = torch.multinomial(
                source,
                int(node_mask.sum()),
                replacement=True,
                generator=generator,
            )
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full(
                    (local_batch,),
                    t_value,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                predictions = model(state, t, node_mask)
                state = _rstar_step(
                    state,
                    predictions["offspring"].softmax(dim=-1),
                    source,
                    t_value,
                    1.0 / sample_steps,
                    node_mask,
                    generator,
                )
            terminal = (
                model(
                    state,
                    torch.ones(
                        local_batch,
                        dtype=torch.float32,
                        device=resolved_device,
                    ),
                    node_mask,
                )["offspring"]
                .detach()
                .cpu()
            )
            for index, node_count in enumerate(counts.tolist()):
                offspring = sample_valid_offspring(
                    terminal[index, :node_count],
                    generator=cpu_generator,
                )
                samples.append(
                    TreeTopologySample(
                        offspring=offspring,
                        parents=offspring_to_parents(offspring),
                    )
                )
    elapsed = time.perf_counter() - start
    return samples, {
        "samples": len(samples),
        "sample_steps": sample_steps,
        "wall_seconds": elapsed,
        "graph_steps_per_second": len(samples) * sample_steps / elapsed,
        "terminal_interventions": {},
    }


def tree_topology_statistics(
    samples: Sequence[TreeTopologySample],
    *,
    maximum_children: int,
) -> dict[str, Any]:
    """Summarize global rooted tree organization without chemistry."""

    if not samples:
        raise TreeTopologyFlowError("tree statistics require samples")
    offspring_histogram: Counter[int] = Counter()
    degrees = []
    maximum_depths = []
    leaf_fractions = []
    for sample in samples:
        offspring_histogram.update(int(value) for value in sample.offspring)
        local_degree = sample.offspring.copy()
        local_degree[1:] += 1
        degrees.extend(int(value) for value in local_degree)
        depths = np.zeros(sample.node_count, dtype=np.int64)
        for child in range(1, sample.node_count):
            depths[child] = depths[int(sample.parents[child])] + 1
        maximum_depths.append(int(depths.max(initial=0)))
        leaf_fractions.append(float(np.mean(sample.offspring == 0)))
    offspring_total = sum(offspring_histogram.values())
    return {
        "samples": len(samples),
        "mean_maximum_root_distance": float(np.mean(maximum_depths)),
        "branch_atom_fraction": float(np.mean(np.asarray(degrees) >= 3)),
        "mean_leaf_atom_fraction": float(np.mean(leaf_fractions)),
        "offspring_fractions": {
            str(value): offspring_histogram[value] / max(1, offspring_total)
            for value in range(maximum_children + 1)
        },
    }
