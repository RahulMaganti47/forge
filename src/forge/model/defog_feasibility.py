"""Bounded DeFoG-style dense graph-flow feasibility probe for M0-06.

This module is deliberately a probe, not the production FORGE product prior. It
implements the load-bearing mechanics needed to test the dense representation:

* independent linear-interpolation corruption of nodes and upper-triangle edges;
* an explicit no-bond edge class;
* a permutation-equivariant dense graph network that predicts clean marginals;
* continuous-time Markov chain Euler sampling using the DeFoG R-star rate; and
* molecular, sparsity, reconstruction, memory, and throughput diagnostics.

The node count is sampled from the empirical training-fold distribution, as in
the reference DeFoG implementation. Molecular stereochemistry is intentionally
flat in accordance with the frozen FORGE plan and is assigned at dossier time.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - exercised only without the optional extra
    torch = None
    nn = None
    functional = None


NO_BOND = 0
BOND_TYPE_TO_INDEX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}
INDEX_TO_BOND_TYPE = {value: key for key, value in BOND_TYPE_TO_INDEX.items()}
SIZE_STRATA = (
    ("01_32", 1, 32),
    ("33_48", 33, 48),
    ("49_64", 49, 64),
    ("65_80", 65, 80),
    ("81_96", 81, 96),
)


class FeasibilityError(RuntimeError):
    """Raised when the bounded feasibility contract cannot be evaluated."""


@dataclass(frozen=True)
class AtomState:
    """Flat atom state used by the bounded product graph."""

    symbol: str
    formal_charge: int
    aromatic: bool
    explicit_hydrogens: int = 0

    def key(self) -> tuple[str, int, int, int]:
        return (
            self.symbol,
            self.formal_charge,
            int(self.aromatic),
            self.explicit_hydrogens,
        )


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_torch() -> None:
    if torch is None:
        raise FeasibilityError(
            "M0-06 requires the optional torch dependency; install the project torch extra"
        )


def pair_mask(node_mask: Any, *, upper_only: bool) -> Any:
    """Return valid off-diagonal node-pair positions."""

    mask = node_mask[:, :, None] & node_mask[:, None, :]
    n_max = node_mask.shape[1]
    diagonal = torch.eye(n_max, dtype=torch.bool, device=node_mask.device)[None, :, :]
    mask = mask & ~diagonal
    if upper_only:
        upper = torch.triu(
            torch.ones((n_max, n_max), dtype=torch.bool, device=node_mask.device),
            diagonal=1,
        )
        mask = mask & upper[None, :, :]
    return mask


if nn is not None:

    class DenseEquivariantBlock(nn.Module):
        """A small symmetric dense message-passing block."""

        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.gate = nn.Linear(hidden_dim, hidden_dim)
            self.node_update = nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.edge_update = nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.node_norm = nn.LayerNorm(hidden_dim)
            self.edge_norm = nn.LayerNorm(hidden_dim)

        def forward(self, h: Any, e: Any, node_mask: Any) -> tuple[Any, Any]:
            full_pair_mask = pair_mask(node_mask, upper_only=False)
            values = self.value(h)
            messages = torch.sigmoid(self.gate(e)) * values[:, None, :, :]
            messages = messages * full_pair_mask[:, :, :, None]
            denominator = full_pair_mask.sum(dim=2, keepdim=True).clamp(min=1)
            aggregate = messages.sum(dim=2) / denominator
            masked_h = h * node_mask[:, :, None]
            global_h = masked_h.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_h = global_h[:, None, :].expand_as(h)
            h = self.node_norm(h + self.node_update(torch.cat((h, aggregate, global_h), dim=-1)))
            h = h * node_mask[:, :, None]

            pair_sum = h[:, :, None, :] + h[:, None, :, :]
            pair_difference = torch.abs(h[:, :, None, :] - h[:, None, :, :])
            update = self.edge_update(torch.cat((e, pair_sum, pair_difference), dim=-1))
            e = self.edge_norm(e + update)
            e = 0.5 * (e + e.transpose(1, 2))
            e = e * full_pair_mask[:, :, :, None]
            return h, e

    class DenseGraphFlowProbe(nn.Module):
        """Small DeFoG-style clean-marginal predictor for the bounded gate."""

        def __init__(
            self,
            node_classes: int,
            edge_classes: int,
            hidden_dim: int,
            layers: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.node_embedding = nn.Embedding(node_classes, hidden_dim)
            self.edge_embedding = nn.Embedding(edge_classes, hidden_dim)
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.blocks = nn.ModuleList(
                DenseEquivariantBlock(hidden_dim, dropout) for _ in range(layers)
            )
            self.node_output = nn.Linear(hidden_dim, node_classes)
            self.edge_output = nn.Sequential(
                nn.Linear(3 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Linear(2 * hidden_dim, edge_classes),
            )

        def forward(self, nodes: Any, edges: Any, t: Any, node_mask: Any) -> tuple[Any, Any]:
            time_embedding = self.time_embedding(t[:, None])
            h = self.node_embedding(nodes) + time_embedding[:, None, :]
            e = self.edge_embedding(edges) + time_embedding[:, None, None, :]
            h = h * node_mask[:, :, None]
            e = e * pair_mask(node_mask, upper_only=False)[:, :, :, None]
            for block in self.blocks:
                h, e = block(h, e, node_mask)
            node_logits = self.node_output(h)
            pair_sum = h[:, :, None, :] + h[:, None, :, :]
            pair_difference = torch.abs(h[:, :, None, :] - h[:, None, :, :])
            edge_logits = self.edge_output(torch.cat((e, pair_sum, pair_difference), dim=-1))
            edge_logits = 0.5 * (edge_logits + edge_logits.transpose(1, 2))
            return node_logits, edge_logits

else:  # pragma: no cover - optional dependency fallback

    class DenseGraphFlowProbe:  # type: ignore[no-redef]
        """Placeholder that reports the missing optional dependency."""

        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def _sample_categorical(probabilities: Any, generator: Any) -> Any:
    shape = probabilities.shape[:-1]
    sampled = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        1,
        generator=generator,
    )
    return sampled.reshape(shape)


def _rstar_step(
    current: Any,
    clean_probabilities: Any,
    marginal: Any,
    t: float,
    dt: float,
    valid_mask: Any,
    generator: Any,
) -> Any:
    """One Euler step using DeFoG's minimum R-star conditional rate."""

    sampled_clean = _sample_categorical(clean_probabilities, generator)
    flat_current = current[valid_mask]
    flat_clean = sampled_clean[valid_mask]
    if flat_current.numel() == 0:
        return current
    if marginal.ndim == 1:
        flat_marginal = marginal[None, :].repeat(flat_current.shape[0], 1)
    elif marginal.ndim == current.ndim + 1 and marginal.shape[:-1] == current.shape:
        flat_marginal = marginal[valid_mask]
    else:
        raise FeasibilityError("R-star marginal has incompatible support")
    classes = flat_marginal.shape[1]
    derivative = -flat_marginal
    derivative.scatter_add_(
        1,
        flat_clean[:, None],
        torch.ones(
            (flat_clean.shape[0], 1),
            dtype=derivative.dtype,
            device=derivative.device,
        ),
    )
    derivative_current = derivative.gather(1, flat_current[:, None])
    interpolant_support = ((1.0 - t) * flat_marginal) + t * functional.one_hot(
        flat_clean, num_classes=classes
    ) > 0
    numerator = torch.relu(derivative - derivative_current) * interpolant_support
    p_current = (1.0 - t) * flat_marginal.gather(
        1,
        flat_current[:, None],
    ).squeeze(1)
    p_current += t * (flat_current == flat_clean).float()
    nonzero_states = interpolant_support.sum(dim=1)
    rates = numerator / (nonzero_states[:, None] * p_current[:, None].clamp(min=1e-8))
    rates.scatter_(1, flat_current[:, None], 0.0)
    off_diagonal = rates * dt
    total = off_diagonal.sum(dim=1, keepdim=True)
    scale = torch.where(total > 0.999, 0.999 / total, torch.ones_like(total))
    off_diagonal = off_diagonal * scale
    probabilities = off_diagonal
    probabilities.scatter_(
        1,
        flat_current[:, None],
        1.0 - off_diagonal.sum(dim=1, keepdim=True),
    )
    sampled = _sample_categorical(probabilities, generator)
    output = current.clone()
    output[valid_mask] = sampled
    return output


def graph_to_molecule(
    nodes: np.ndarray,
    edges: np.ndarray,
    atom_vocabulary: Sequence[AtomState],
) -> Chem.Mol:
    """Construct and sanitize an RDKit molecule from one generated flat graph."""

    editable = Chem.RWMol()
    for node in nodes:
        state = atom_vocabulary[int(node)]
        atom = Chem.Atom(state.symbol)
        atom.SetFormalCharge(state.formal_charge)
        atom.SetIsAromatic(state.aromatic)
        if state.explicit_hydrogens:
            atom.SetNumExplicitHs(state.explicit_hydrogens)
            atom.SetNoImplicit(True)
        editable.AddAtom(atom)
    for begin in range(len(nodes)):
        for end in range(begin + 1, len(nodes)):
            edge_type = int(edges[begin, end])
            if edge_type == NO_BOND:
                continue
            editable.AddBond(begin, end, INDEX_TO_BOND_TYPE[edge_type])
    molecule = editable.GetMol()
    Chem.SanitizeMol(molecule)
    return molecule


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
