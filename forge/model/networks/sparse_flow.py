"""Bounded sparse, hierarchical discrete-flow probe for molecular topology.

The probe factorizes a connected molecular graph into a rooted spanning tree
and a small set of residual ring-closure edges. Parent pointers and closure
endpoints retain full pair reachability, while bond-state losses are evaluated
only on active molecular edges. This avoids treating every absent atom pair as
an edge class without changing FORGE into fragment or building-block
generation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from forge.model.networks.dense_flow import (
    AtomState,
    FeasibilityError,
    graph_to_molecule,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None


SPARSE_BOND_TO_INDEX = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}
INDEX_TO_DENSE_BOND = {0: 1, 1: 2, 2: 3, 3: 4}
BOND_VALENCE_UNITS = torch.tensor([2, 4, 6, 3]) if torch is not None else None


@dataclass(frozen=True)
class SparseGraphRecord:
    """Canonical rooted-tree and closure representation of one molecule."""

    structure_id: str
    canonical_smiles: str
    node_states: np.ndarray
    parents: np.ndarray
    parent_bonds: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    closure_bonds: np.ndarray
    edges: np.ndarray
    region_states: np.ndarray | None = None

    @property
    def node_count(self) -> int:
        return int(self.node_states.shape[0])

    @property
    def closure_count(self) -> int:
        return int(self.closure_left.shape[0])


def sparse_roundtrip_exact(record: SparseGraphRecord) -> bool:
    """Require the sparse program to reconstruct every original edge label."""

    reconstructed = np.zeros_like(record.edges)
    for child in range(1, record.node_count):
        parent = int(record.parents[child])
        edge = INDEX_TO_DENSE_BOND[int(record.parent_bonds[child])]
        reconstructed[child, parent] = reconstructed[parent, child] = edge
    for left, right, bond in zip(
        record.closure_left,
        record.closure_right,
        record.closure_bonds,
        strict=True,
    ):
        edge = INDEX_TO_DENSE_BOND[int(bond)]
        reconstructed[int(left), int(right)] = reconstructed[int(right), int(left)] = edge
    return bool(np.array_equal(reconstructed, record.edges))


def sparse_constitutional_roundtrip_exact(
    record: SparseGraphRecord,
    atom_vocabulary: Sequence[AtomState],
) -> bool:
    """Require sanitization to restore the original constitutional graph.

    The flow operates on deterministic Kekule bond assignments and deliberately
    omits stereochemistry. RDKit sanitization of the reconstructed graph must
    therefore recover the same canonical constitutional SMILES, including
    aromaticity perception.
    """

    original = Chem.MolFromSmiles(record.canonical_smiles)
    if original is None:
        raise FeasibilityError(f"invalid original SMILES: {record.canonical_smiles}")
    reconstructed = graph_to_molecule(
        record.node_states,
        record.edges,
        atom_vocabulary,
    )
    original_smiles = Chem.MolToSmiles(
        original,
        canonical=True,
        isomericSmiles=False,
    )
    reconstructed_smiles = Chem.MolToSmiles(
        reconstructed,
        canonical=True,
        isomericSmiles=False,
    )
    return original_smiles == reconstructed_smiles


def collate_sparse_records(
    records: Sequence[SparseGraphRecord],
    n_max: int,
    maximum_closures: int,
) -> dict[str, Any]:
    """Collate fixed-size sparse state tensors without dense edge features."""

    batch = len(records)
    nodes = torch.zeros((batch, n_max), dtype=torch.long)
    parents = torch.zeros((batch, n_max), dtype=torch.long)
    parent_bonds = torch.zeros((batch, n_max), dtype=torch.long)
    closure_left = torch.zeros((batch, maximum_closures), dtype=torch.long)
    closure_right = torch.zeros((batch, maximum_closures), dtype=torch.long)
    closure_bonds = torch.zeros((batch, maximum_closures), dtype=torch.long)
    node_mask = torch.zeros((batch, n_max), dtype=torch.bool)
    closure_mask = torch.zeros((batch, maximum_closures), dtype=torch.bool)
    for index, record in enumerate(records):
        count = record.node_count
        closure_count = record.closure_count
        nodes[index, :count] = torch.from_numpy(record.node_states.copy())
        parents[index, :count] = torch.from_numpy(record.parents.copy())
        parent_bonds[index, :count] = torch.from_numpy(record.parent_bonds.copy())
        node_mask[index, :count] = True
        if closure_count:
            closure_left[index, :closure_count] = torch.from_numpy(record.closure_left.copy())
            closure_right[index, :closure_count] = torch.from_numpy(record.closure_right.copy())
            closure_bonds[index, :closure_count] = torch.from_numpy(record.closure_bonds.copy())
            closure_mask[index, :closure_count] = True
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    return {
        "nodes": nodes,
        "parents": parents,
        "parent_bonds": parent_bonds,
        "closure_left": closure_left,
        "closure_right": closure_right,
        "closure_bonds": closure_bonds,
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": closure_mask,
    }


def _gather_nodes(hidden: Any, indices: Any) -> Any:
    return hidden.gather(1, indices[:, :, None].expand(-1, -1, hidden.shape[-1]))


if nn is not None:

    class SparseTreeBlock(nn.Module):
        """Linear-state message passing over parent and child relations."""

        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.update = nn.Sequential(
                nn.Linear(4 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, hidden: Any, parents: Any, node_mask: Any, child_mask: Any) -> Any:
            parent_hidden = _gather_nodes(hidden, parents)
            child_sum = torch.zeros_like(hidden)
            child_sum.scatter_add_(
                1,
                parents[:, :, None].expand_as(hidden),
                hidden * child_mask[:, :, None],
            )
            child_count = torch.zeros_like(node_mask, dtype=hidden.dtype)
            child_count.scatter_add_(1, parents, child_mask.to(hidden.dtype))
            child_mean = child_sum / child_count[:, :, None].clamp(min=1.0)
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden = global_hidden / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_hidden = global_hidden[:, None, :].expand_as(hidden)
            update = self.update(
                torch.cat((hidden, parent_hidden, child_mean, global_hidden), dim=-1)
            )
            return self.norm(hidden + update) * node_mask[:, :, None]

    class SparseTopologyFlowProbe(nn.Module):
        """Joint clean-marginal predictor for nodes, tree pointers, and closures."""

        def __init__(
            self,
            node_classes: int,
            bond_classes: int,
            hidden_dim: int,
            layers: int,
            maximum_closures: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.hidden_dim = hidden_dim
            self.node_embedding = nn.Embedding(node_classes, hidden_dim)
            self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.blocks = nn.ModuleList(SparseTreeBlock(hidden_dim, dropout) for _ in range(layers))
            self.node_output = nn.Linear(hidden_dim, node_classes)
            self.parent_query = nn.Linear(hidden_dim, hidden_dim)
            self.parent_key = nn.Linear(hidden_dim, hidden_dim)
            self.backbone_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.closure_slots = nn.Embedding(maximum_closures, hidden_dim)
            self.closure_update = nn.Sequential(
                nn.Linear(4 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.closure_left_query = nn.Linear(hidden_dim, hidden_dim)
            self.closure_right_query = nn.Linear(hidden_dim, hidden_dim)
            self.closure_node_key = nn.Linear(hidden_dim, hidden_dim)
            self.closure_bond_output = nn.Linear(hidden_dim, bond_classes)

        def forward(
            self,
            nodes: Any,
            parents: Any,
            parent_bonds: Any,
            closure_left: Any,
            closure_right: Any,
            t: Any,
            node_mask: Any,
            child_mask: Any,
        ) -> dict[str, Any]:
            time_hidden = self.time_embedding(t[:, None])
            hidden = self.node_embedding(nodes) + time_hidden[:, None, :]
            hidden = hidden + self.bond_embedding(parent_bonds) * child_mask[:, :, None]
            hidden = hidden * node_mask[:, :, None]
            for block in self.blocks:
                hidden = block(hidden, parents, node_mask, child_mask)

            parent_query = self.parent_query(hidden)
            parent_key = self.parent_key(hidden)
            parent_logits = torch.einsum("bid,bjd->bij", parent_query, parent_key) / math.sqrt(
                self.hidden_dim
            )
            parent_hidden = _gather_nodes(hidden, parents)
            parent_bond_logits = self.backbone_bond_output(
                torch.cat((hidden, parent_hidden), dim=-1)
            )

            batch, maximum_closures = closure_left.shape
            slots = self.closure_slots(torch.arange(maximum_closures, device=nodes.device))[
                None, :, :
            ].expand(batch, -1, -1)
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden = global_hidden / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_hidden = global_hidden[:, None, :].expand_as(slots)
            left_hidden = _gather_nodes(hidden, closure_left)
            right_hidden = _gather_nodes(hidden, closure_right)
            closure_hidden = self.closure_update(
                torch.cat((slots, global_hidden, left_hidden, right_hidden), dim=-1)
            )
            closure_key = self.closure_node_key(hidden)
            left_logits = torch.einsum(
                "bkd,bnd->bkn",
                self.closure_left_query(closure_hidden),
                closure_key,
            ) / math.sqrt(self.hidden_dim)
            right_logits = torch.einsum(
                "bkd,bnd->bkn",
                self.closure_right_query(closure_hidden),
                closure_key,
            ) / math.sqrt(self.hidden_dim)
            return {
                "nodes": self.node_output(hidden),
                "parents": parent_logits,
                "parent_bonds": parent_bond_logits,
                "closure_left": left_logits,
                "closure_right": right_logits,
                "closure_bonds": self.closure_bond_output(closure_hidden),
            }

else:  # pragma: no cover - optional dependency fallback

    class SparseTopologyFlowProbe:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise FeasibilityError("sparse topology probe requires torch")


def _parent_candidate_mask(node_mask: Any) -> Any:
    n_max = node_mask.shape[1]
    previous = torch.tril(
        torch.ones(
            (n_max, n_max),
            dtype=torch.bool,
            device=node_mask.device,
        ),
        diagonal=-1,
    )
    return previous[None, :, :] & node_mask[:, :, None] & node_mask[:, None, :]


def _endpoint_candidate_mask(node_mask: Any, slots: int) -> Any:
    return node_mask[:, None, :].expand(-1, slots, -1)


def sample_pointer_interpolation(
    clean: Any,
    candidate_mask: Any,
    active_mask: Any,
    t: Any,
    generator: Any,
) -> Any:
    """Linear interpolation from a variable uniform pointer marginal."""

    probabilities = candidate_mask.to(torch.float32)
    probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp(min=1.0)
    flat_probabilities = probabilities[active_mask]
    noise = torch.multinomial(flat_probabilities, 1, generator=generator).squeeze(1)
    if clean.ndim == 2:
        example_index = torch.arange(clean.shape[0], device=clean.device)[:, None].expand_as(clean)[
            active_mask
        ]
    else:
        raise FeasibilityError("pointer labels must have shape batch by variables")
    keep = (
        torch.rand(
            noise.shape[0],
            generator=generator,
            device=clean.device,
        )
        < t[example_index]
    )
    sampled = torch.where(keep, clean[active_mask], noise)
    output = clean.clone()
    output[active_mask] = sampled
    return output


def pointer_rstar_step(
    current: Any,
    clean_logits: Any,
    candidate_mask: Any,
    active_mask: Any,
    t: float,
    dt: float,
    generator: Any,
) -> Any:
    """Euler R-star update for pointers with variable uniform marginals."""

    masked_logits = clean_logits.masked_fill(~candidate_mask, -1e9)
    clean_probabilities = masked_logits.softmax(dim=-1)
    flat_clean_probabilities = clean_probabilities[active_mask]
    sampled_clean = torch.multinomial(flat_clean_probabilities, 1, generator=generator).squeeze(1)
    flat_candidates = candidate_mask[active_mask]
    flat_current = current[active_mask]
    counts = flat_candidates.sum(dim=1)
    p0 = flat_candidates.to(torch.float32) / counts[:, None]
    derivative = -p0
    derivative.scatter_add_(
        1,
        sampled_clean[:, None],
        torch.ones(
            (sampled_clean.shape[0], 1),
            device=current.device,
        ),
    )
    derivative_current = derivative.gather(1, flat_current[:, None])
    numerator = torch.relu(derivative - derivative_current)
    numerator *= flat_candidates
    p_current = (1.0 - t) * p0.gather(1, flat_current[:, None]).squeeze(1)
    p_current += t * (flat_current == sampled_clean).to(torch.float32)
    rates = numerator / (counts[:, None] * p_current[:, None].clamp(min=1e-8))
    rates.scatter_(1, flat_current[:, None], 0.0)
    transition = rates * dt
    total = transition.sum(dim=1, keepdim=True)
    transition *= torch.where(total > 0.999, 0.999 / total, torch.ones_like(total))
    transition.scatter_(
        1,
        flat_current[:, None],
        1.0 - transition.sum(dim=1, keepdim=True),
    )
    sampled = torch.multinomial(transition, 1, generator=generator).squeeze(1)
    output = current.clone()
    output[active_mask] = sampled
    return output


def _sample_flat_interpolation(
    clean: Any,
    marginal: Any,
    t: Any,
    active_mask: Any,
    generator: Any,
) -> Any:
    probabilities = marginal[None, :].repeat(int(active_mask.sum()), 1)
    example_index = torch.arange(clean.shape[0], device=clean.device)[:, None].expand_as(clean)[
        active_mask
    ]
    probabilities *= 1.0 - t[example_index, None]
    probabilities.scatter_add_(1, clean[active_mask][:, None], t[example_index, None])
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    output = clean.clone()
    output[active_mask] = sampled
    return output


def _maximum_valence_units(state: AtomState) -> int:
    if state.symbol == "C":
        return 8
    if state.symbol == "F":
        return 2
    if state.symbol == "N":
        return 8 if state.formal_charge > 0 else 6
    if state.symbol == "O":
        return 2 if state.formal_charge < 0 else 4
    if state.symbol == "P":
        return 10
    if state.symbol == "S":
        return 12
    if state.symbol == "Si":
        return 8
    raise FeasibilityError(f"no valence policy for {state}")
