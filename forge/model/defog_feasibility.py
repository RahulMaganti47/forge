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

import csv
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

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


@dataclass(frozen=True)
class GraphRecord:
    """One tensorized heavy-atom molecular graph."""

    structure_id: str
    canonical_smiles: str
    node_states: np.ndarray
    edges: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.node_states.shape[0])


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


def _resolve_and_verify(repo: Path, record: Mapping[str, str], label: str) -> Path:
    path = Path(record["path"])
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise FeasibilityError(f"{label} input does not exist: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise FeasibilityError(
            f"{label} SHA-256 mismatch: expected {record['sha256']}, observed {actual}"
        )
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _atom_state(atom: Chem.Atom) -> AtomState:
    return AtomState(
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        atom.GetIsAromatic(),
        atom.GetNumExplicitHs(),
    )


def audit_input_support(
    rows: Sequence[Mapping[str, str]],
    declared_elements: set[str],
    documented_profile: Mapping[str, float | int],
    tolerance: float,
) -> dict[str, Any]:
    """Audit the real input instead of trusting stale support-profile prose."""

    total = len(rows)
    heavy_atoms = [int(row["heavy_atoms"]) for row in rows]
    element_exclusions: Counter[str] = Counter()
    eligible = []
    atom_states: set[tuple[str, int, int, int]] = set()
    bond_types: Counter[str] = Counter()
    bond_stereo: Counter[str] = Counter()
    invalid_smiles: list[str] = []

    for row in rows:
        elements = set(row["elements"].split("|"))
        unsupported = elements.difference(declared_elements)
        if unsupported:
            element_exclusions.update(unsupported)
        else:
            eligible.append(row)
        mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        if mol is None:
            invalid_smiles.append(row["r0_structure_id"])
            continue
        for atom in mol.GetAtoms():
            atom_states.add(_atom_state(atom).key())
        for bond in mol.GetBonds():
            bond_types[str(bond.GetBondType())] += 1
            bond_stereo[str(bond.GetStereo())] += 1

    actual_profile = {
        "rows": total,
        "n64_count": sum(value <= 64 for value in heavy_atoms),
        "n64_fraction": sum(value <= 64 for value in heavy_atoms) / total,
        "n96_count": sum(value <= 96 for value in heavy_atoms),
        "n96_fraction": sum(value <= 96 for value in heavy_atoms) / total,
        "maximum_heavy_atoms": max(heavy_atoms),
    }
    supported_heavy_atoms = [int(row["heavy_atoms"]) for row in eligible]
    supported_profile = {
        "rows": len(eligible),
        "n64_count": sum(value <= 64 for value in supported_heavy_atoms),
        "n64_fraction": sum(value <= 64 for value in supported_heavy_atoms) / len(eligible),
        "n96_count": sum(value <= 96 for value in supported_heavy_atoms),
        "n96_fraction": sum(value <= 96 for value in supported_heavy_atoms) / len(eligible),
        "maximum_heavy_atoms": max(supported_heavy_atoms),
    }
    comparisons = {
        "n64_fraction_matches": abs(
            actual_profile["n64_fraction"] - float(documented_profile["n64_fraction"])
        )
        <= tolerance,
        "n96_fraction_matches": abs(
            actual_profile["n96_fraction"] - float(documented_profile["n96_fraction"])
        )
        <= tolerance,
        "maximum_heavy_atoms_matches": actual_profile["maximum_heavy_atoms"]
        == int(documented_profile["maximum_heavy_atoms"]),
    }
    return {
        "actual_full_r0": actual_profile,
        "declared_element_supported_r0": supported_profile,
        "documented_profile": dict(documented_profile),
        "profile_comparisons": comparisons,
        "profile_matches": all(comparisons.values()),
        "declared_elements": sorted(declared_elements),
        "rows_outside_declared_element_vocabulary": total - len(eligible),
        "unsupported_element_row_incidence": dict(sorted(element_exclusions.items())),
        "observed_atom_states": [
            {
                "symbol": symbol,
                "formal_charge": charge,
                "aromatic": bool(aromatic),
                "explicit_hydrogens": explicit_hydrogens,
            }
            for symbol, charge, aromatic, explicit_hydrogens in sorted(atom_states)
        ],
        "observed_bond_types": dict(sorted(bond_types.items())),
        "observed_bond_stereochemistry": dict(sorted(bond_stereo.items())),
        "invalid_smiles": invalid_smiles,
    }


def build_atom_vocabulary(
    rows: Sequence[Mapping[str, str]], declared_elements: set[str]
) -> tuple[AtomState, ...]:
    """Freeze all observed flat atom states inside the declared element support."""

    states: set[AtomState] = set()
    for row in rows:
        if not set(row["elements"].split("|")).issubset(declared_elements):
            continue
        mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        if mol is None:
            raise FeasibilityError(f"invalid R0 SMILES: {row['r0_structure_id']}")
        states.update(_atom_state(atom) for atom in mol.GetAtoms())
    return tuple(sorted(states, key=AtomState.key))


def tensorize_row(
    row: Mapping[str, str],
    atom_to_index: Mapping[AtomState, int],
) -> GraphRecord:
    """Convert one R0 SMILES into a flat heavy-atom graph."""

    mol = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
    if mol is None:
        raise FeasibilityError(f"invalid R0 SMILES: {row['r0_structure_id']}")
    nodes = np.asarray(
        [atom_to_index[_atom_state(atom)] for atom in mol.GetAtoms()], dtype=np.int64
    )
    edges = np.zeros((len(nodes), len(nodes)), dtype=np.int64)
    for bond in mol.GetBonds():
        try:
            edge_type = BOND_TYPE_TO_INDEX[bond.GetBondType()]
        except KeyError as exc:
            raise FeasibilityError(
                f"unsupported bond type {bond.GetBondType()} in {row['r0_structure_id']}"
            ) from exc
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edges[begin, end] = edge_type
        edges[end, begin] = edge_type
    return GraphRecord(
        structure_id=row["r0_structure_id"],
        canonical_smiles=row["canonical_isomeric_smiles"],
        node_states=nodes,
        edges=edges,
    )


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _size_stratum(node_count: int) -> str:
    for name, lower, upper in SIZE_STRATA:
        if lower <= node_count <= upper:
            return name
    return "over_96"


def deterministic_stratified_subset(
    records: Sequence[GraphRecord], limit: int, seed: int
) -> tuple[GraphRecord, ...]:
    """Select a deterministic size-stratified subset without using held-out labels."""

    if len(records) <= limit:
        return tuple(sorted(records, key=lambda record: record.structure_id))
    buckets: dict[str, list[GraphRecord]] = defaultdict(list)
    for record in records:
        buckets[_size_stratum(record.node_count)].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: _stable_rank(record.structure_id, seed))

    selected: list[GraphRecord] = []
    names = sorted(buckets)
    positions = {name: 0 for name in names}
    while len(selected) < limit:
        progressed = False
        for name in names:
            position = positions[name]
            if position < len(buckets[name]):
                selected.append(buckets[name][position])
                positions[name] += 1
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
    return tuple(selected)


def prepare_records(
    rows: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, str]],
    atom_vocabulary: Sequence[AtomState],
    declared_elements: set[str],
    fold_column: str,
    n_max: int,
) -> tuple[dict[str, tuple[GraphRecord, ...]], dict[str, int]]:
    """Build fold-separated graph records with explicit exclusion counts."""

    fold_by_id = {row["r0_structure_id"]: row[fold_column] for row in assignments}
    atom_to_index = {state: index for index, state in enumerate(atom_vocabulary)}
    records: dict[str, list[GraphRecord]] = {
        "R0_train": [],
        "R0_cal": [],
        "R0_heldout": [],
    }
    exclusions = Counter()
    for row in rows:
        structure_id = row["r0_structure_id"]
        if structure_id not in fold_by_id:
            raise FeasibilityError(f"R0 structure absent from frozen splits: {structure_id}")
        elements = set(row["elements"].split("|"))
        if not elements.issubset(declared_elements):
            exclusions["outside_element_vocabulary"] += 1
            continue
        if int(row["heavy_atoms"]) > n_max:
            exclusions["over_n_max"] += 1
            continue
        record = tensorize_row(row, atom_to_index)
        if record.node_count != int(row["heavy_atoms"]):
            raise FeasibilityError(f"heavy-atom count mismatch: {structure_id}")
        records[fold_by_id[structure_id]].append(record)
    return (
        {
            fold: tuple(sorted(values, key=lambda record: record.structure_id))
            for fold, values in records.items()
        },
        dict(sorted(exclusions.items())),
    )


def compute_marginals(
    records: Sequence[GraphRecord], node_classes: int, edge_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute empirical node and upper-triangle edge initial distributions."""

    node_counts = np.zeros(node_classes, dtype=np.float64)
    edge_counts = np.zeros(edge_classes, dtype=np.float64)
    for record in records:
        node_counts += np.bincount(record.node_states, minlength=node_classes)
        upper = record.edges[np.triu_indices(record.node_count, 1)]
        edge_counts += np.bincount(upper, minlength=edge_classes)
    if np.any(node_counts == 0) or np.any(edge_counts == 0):
        raise FeasibilityError("every declared state must occur in the training fold")
    return node_counts / node_counts.sum(), edge_counts / edge_counts.sum()


def _permuted_record(record: GraphRecord, rng: np.random.Generator) -> GraphRecord:
    permutation = rng.permutation(record.node_count)
    return GraphRecord(
        structure_id=record.structure_id,
        canonical_smiles=record.canonical_smiles,
        node_states=record.node_states[permutation],
        edges=record.edges[np.ix_(permutation, permutation)],
    )


def collate_records(
    records: Sequence[GraphRecord],
    n_max: int,
    rng: np.random.Generator,
    *,
    permute: bool,
) -> tuple[Any, Any, Any]:
    """Create dense fixed-N tensors and masks for worst-case memory testing."""

    _require_torch()
    batch_size = len(records)
    nodes = torch.zeros((batch_size, n_max), dtype=torch.long)
    edges = torch.zeros((batch_size, n_max, n_max), dtype=torch.long)
    node_mask = torch.zeros((batch_size, n_max), dtype=torch.bool)
    for index, original in enumerate(records):
        record = _permuted_record(original, rng) if permute else original
        count = record.node_count
        nodes[index, :count] = torch.from_numpy(record.node_states.copy())
        edges[index, :count, :count] = torch.from_numpy(record.edges.copy())
        node_mask[index, :count] = True
    return nodes, edges, node_mask


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


def _mirror_upper(labels: Any) -> Any:
    """Mirror categorical upper-triangle labels without retaining stale lower values."""

    upper = torch.triu(labels, diagonal=1)
    return upper + upper.transpose(1, 2)


def sample_linear_interpolation(
    clean: Any,
    marginal: Any,
    t: Any,
    valid_mask: Any,
    generator: Any,
) -> Any:
    """Sample q_t(x_t|x_1) = t delta(x_1) + (1-t) p_0."""

    flat_clean = clean[valid_mask]
    if flat_clean.numel() == 0:
        return clean.clone()
    if clean.ndim == 2:
        example_index = torch.arange(clean.shape[0], device=clean.device)[:, None].expand_as(clean)[
            valid_mask
        ]
    elif clean.ndim == 3:
        example_index = torch.arange(clean.shape[0], device=clean.device)[:, None, None].expand_as(
            clean
        )[valid_mask]
    else:
        raise FeasibilityError("linear interpolation expects node or edge labels")
    probabilities = (1.0 - t[example_index, None]) * marginal[None, :]
    probabilities.scatter_add_(
        1,
        flat_clean[:, None],
        t[example_index, None],
    )
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    output = clean.clone()
    output[valid_mask] = sampled
    if clean.ndim == 3:
        output = _mirror_upper(output)
    return output


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


def set_determinism(seed: int, cpu_threads: int) -> Any:
    """Freeze RNGs and deterministic CPU execution."""

    _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    return torch.Generator(device="cpu").manual_seed(seed)


def _batch_records(
    records: Sequence[GraphRecord],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[GraphRecord, ...]:
    indices = rng.integers(0, len(records), size=batch_size)
    return tuple(records[int(index)] for index in indices)


def _losses(
    node_logits: Any,
    edge_logits: Any,
    clean_nodes: Any,
    clean_edges: Any,
    node_mask: Any,
    edge_weight: float,
    bond_auxiliary_weight: float,
) -> tuple[Any, dict[str, float]]:
    node_loss = functional.cross_entropy(node_logits[node_mask], clean_nodes[node_mask])
    upper_mask = pair_mask(node_mask, upper_only=True)
    edge_targets = clean_edges[upper_mask]
    edge_ce = functional.cross_entropy(edge_logits[upper_mask], edge_targets)
    bonded = edge_targets != NO_BOND
    if bonded.any():
        bonded_ce = functional.cross_entropy(edge_logits[upper_mask][bonded], edge_targets[bonded])
    else:
        bonded_ce = edge_ce * 0.0
    loss = node_loss + edge_weight * edge_ce + bond_auxiliary_weight * bonded_ce
    return loss, {
        "node_ce": float(node_loss.detach()),
        "edge_ce": float(edge_ce.detach()),
        "bonded_edge_aux_ce": float(bonded_ce.detach()),
        "total": float(loss.detach()),
    }


def train_probe(
    model: Any,
    records: Sequence[GraphRecord],
    node_marginal: np.ndarray,
    edge_marginal: np.ndarray,
    n_max: int,
    batch_size: int,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run the fixed-budget clean-marginal training probe."""

    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32)
    edge_p0 = torch.tensor(edge_marginal, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    losses = []
    start = time.perf_counter()
    examples = 0
    model.train()
    for _ in range(int(config["steps"])):
        batch = _batch_records(records, batch_size, rng)
        clean_nodes, clean_edges, node_mask = collate_records(batch, n_max, rng, permute=True)
        upper_mask = pair_mask(node_mask, upper_only=True)
        t = torch.rand(batch_size, generator=generator).clamp(0.02, 0.98)
        noisy_nodes = sample_linear_interpolation(clean_nodes, node_p0, t, node_mask, generator)
        noisy_edges = sample_linear_interpolation(clean_edges, edge_p0, t, upper_mask, generator)
        node_logits, edge_logits = model(noisy_nodes, noisy_edges, t, node_mask)
        loss, components = _losses(
            node_logits,
            edge_logits,
            clean_nodes,
            clean_edges,
            node_mask,
            float(config["edge_loss_weight"]),
            float(config["bond_auxiliary_weight"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        losses.append(components)
        examples += batch_size
    elapsed = time.perf_counter() - start
    return {
        "steps": int(config["steps"]),
        "examples_seen_with_replacement": examples,
        "wall_seconds": elapsed,
        "graphs_per_second": examples / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_20_loss": {
            key: float(np.mean([row[key] for row in losses[-20:]])) for key in losses[-1]
        },
    }


def _ece(probabilities: np.ndarray, targets: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        if upper == 1.0:
            mask = (probabilities >= lower) & (probabilities <= upper)
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
        if mask.any():
            result += float(mask.mean()) * abs(
                float(probabilities[mask].mean()) - float(targets[mask].mean())
            )
    return result


def evaluate_reconstruction(
    model: Any,
    records: Sequence[GraphRecord],
    node_marginal: np.ndarray,
    edge_marginal: np.ndarray,
    n_max: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Measure denoising and sparsity performance on frozen held-out graphs."""

    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32)
    edge_p0 = torch.tensor(edge_marginal, dtype=torch.float32)
    aggregate: dict[str, list[float]] = defaultdict(list)
    strata: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    bond_probabilities: list[np.ndarray] = []
    bond_targets: list[np.ndarray] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), batch_size):
            batch = tuple(records[offset : offset + batch_size])
            clean_nodes, clean_edges, node_mask = collate_records(batch, n_max, rng, permute=True)
            upper_mask = pair_mask(node_mask, upper_only=True)
            time_values = torch.tensor(
                [0.25 + 0.25 * ((offset + index) % 3) for index in range(len(batch))],
                dtype=torch.float32,
            )
            noisy_nodes = sample_linear_interpolation(
                clean_nodes, node_p0, time_values, node_mask, generator
            )
            noisy_edges = sample_linear_interpolation(
                clean_edges, edge_p0, time_values, upper_mask, generator
            )
            node_logits, edge_logits = model(noisy_nodes, noisy_edges, time_values, node_mask)
            node_predictions = node_logits.argmax(dim=-1)
            edge_prob = edge_logits.softmax(dim=-1)
            edge_predictions = edge_prob.argmax(dim=-1)
            for index, record in enumerate(batch):
                count = record.node_count
                local_nodes = node_mask[index]
                local_upper = upper_mask[index]
                node_correct = (
                    node_predictions[index][local_nodes] == clean_nodes[index][local_nodes]
                )
                edge_truth = clean_edges[index][local_upper]
                edge_pred = edge_predictions[index][local_upper]
                true_bond = edge_truth != NO_BOND
                node_accuracy = float(node_correct.float().mean())
                edge_accuracy = float((edge_pred == edge_truth).float().mean())
                bond_recall = (
                    float((edge_pred[true_bond] != NO_BOND).float().mean())
                    if true_bond.any()
                    else 0.0
                )
                null_mask = ~true_bond
                null_false_positive_rate = (
                    float((edge_pred[null_mask] != NO_BOND).float().mean())
                    if null_mask.any()
                    else 0.0
                )
                exact = float(bool(node_correct.all() and (edge_pred == edge_truth).all()))
                values = {
                    "node_accuracy": node_accuracy,
                    "edge_accuracy": edge_accuracy,
                    "bond_recall": bond_recall,
                    "null_false_positive_rate": null_false_positive_rate,
                    "exact_graph_reconstruction": exact,
                }
                stratum = _size_stratum(count)
                for key, value in values.items():
                    aggregate[key].append(value)
                    strata[stratum][key].append(value)
                bond_probabilities.append(
                    (1.0 - edge_prob[index][local_upper, NO_BOND]).cpu().numpy()
                )
                bond_targets.append(true_bond.cpu().numpy().astype(np.float64))
    elapsed = time.perf_counter() - start
    probabilities = np.concatenate(bond_probabilities)
    targets = np.concatenate(bond_targets)
    return {
        "rows": len(records),
        "wall_seconds": elapsed,
        "graphs_per_second": len(records) / elapsed,
        "metrics": {key: float(np.mean(values)) for key, values in aggregate.items()},
        "bond_sparsity_calibration": {
            "observed_bond_fraction": float(targets.mean()),
            "mean_predicted_bond_probability": float(probabilities.mean()),
            "absolute_density_error": abs(float(probabilities.mean() - targets.mean())),
            "brier_score": float(np.mean((probabilities - targets) ** 2)),
            "ece_10_bin": _ece(probabilities, targets),
            "edge_null_baseline_accuracy": float(1.0 - targets.mean()),
        },
        "by_size": {
            name: {
                "rows": len(next(iter(metrics.values()))),
                **{key: float(np.mean(values)) for key, values in metrics.items()},
            }
            for name, metrics in sorted(strata.items())
        },
    }


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


def sample_endpoints(
    model: Any,
    train_records: Sequence[GraphRecord],
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    edge_marginal: np.ndarray,
    n_max: int,
    sample_count: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Generate unconditional graph endpoints with empirical node-count sampling."""

    rng = np.random.default_rng(seed)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32)
    edge_p0 = torch.tensor(edge_marginal, dtype=torch.float32)
    train_node_counts = np.asarray([record.node_count for record in train_records])
    sampled_node_counts = rng.choice(train_node_counts, size=sample_count, replace=True)
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, sample_count, batch_size):
            counts = sampled_node_counts[offset : offset + batch_size]
            local_batch = len(counts)
            node_mask = (
                torch.arange(n_max)[None, :] < torch.tensor(counts, dtype=torch.long)[:, None]
            )
            full_pair = pair_mask(node_mask, upper_only=False)
            upper = pair_mask(node_mask, upper_only=True)
            nodes = torch.zeros((local_batch, n_max), dtype=torch.long)
            nodes[node_mask] = torch.multinomial(
                node_p0,
                int(node_mask.sum()),
                replacement=True,
                generator=generator,
            )
            edges = torch.zeros((local_batch, n_max, n_max), dtype=torch.long)
            initial_edges = torch.multinomial(
                edge_p0,
                int(upper.sum()),
                replacement=True,
                generator=generator,
            )
            edges[upper] = initial_edges
            edges = _mirror_upper(edges)
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full((local_batch,), t_value, dtype=torch.float32)
                node_logits, edge_logits = model(nodes, edges, t, node_mask)
                node_probabilities = node_logits.softmax(dim=-1)
                edge_probabilities = edge_logits.softmax(dim=-1)
                if step == sample_steps - 1:
                    nodes[node_mask] = _sample_categorical(node_probabilities[node_mask], generator)
                    edges[upper] = _sample_categorical(edge_probabilities[upper], generator)
                else:
                    dt = 1.0 / sample_steps
                    nodes = _rstar_step(
                        nodes,
                        node_probabilities,
                        node_p0,
                        t_value,
                        dt,
                        node_mask,
                        generator,
                    )
                    edges = _rstar_step(
                        edges,
                        edge_probabilities,
                        edge_p0,
                        t_value,
                        dt,
                        upper,
                        generator,
                    )
                edges = _mirror_upper(edges)
                edges[~full_pair] = NO_BOND
            for index, count in enumerate(counts):
                samples.append(
                    (
                        nodes[index, :count].cpu().numpy().copy(),
                        edges[index, :count, :count].cpu().numpy().copy(),
                    )
                )
    elapsed = time.perf_counter() - start
    return samples, {
        "samples": sample_count,
        "sampling_steps": sample_steps,
        "wall_seconds": elapsed,
        "graph_steps_per_second": sample_count * sample_steps / elapsed,
        "node_count_source": "empirical_R0_train_distribution_conditioned_on_n_max",
        "atom_vocabulary_size": len(atom_vocabulary),
    }


def _connected(edges: np.ndarray) -> bool:
    count = edges.shape[0]
    if count == 0:
        return False
    seen = {0}
    frontier = [0]
    while frontier:
        node = frontier.pop()
        for neighbor in np.flatnonzero(edges[node] != NO_BOND):
            neighbor = int(neighbor)
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return len(seen) == count


def topology_hash(edges: np.ndarray, iterations: int = 4) -> str:
    """Return a deterministic unlabeled Weisfeiler-Lehman topology hash."""

    labels = [str(int(np.count_nonzero(row))) for row in edges]
    adjacency = [np.flatnonzero(row != NO_BOND).tolist() for row in edges]
    for _ in range(iterations):
        labels = [
            hashlib.sha256(
                (labels[index] + "|" + "|".join(sorted(labels[j] for j in neighbors))).encode()
            ).hexdigest()[:16]
            for index, neighbors in enumerate(adjacency)
        ]
    graph_signature = "|".join(sorted(labels))
    return hashlib.sha256(graph_signature.encode()).hexdigest()


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


def _distribution(values: Sequence[int], n_max: int) -> np.ndarray:
    counts = np.bincount(np.asarray(values, dtype=np.int64), minlength=n_max + 1).astype(np.float64)
    return counts / counts.sum()


def _jensen_shannon(left: np.ndarray, right: np.ndarray) -> float:
    midpoint = 0.5 * (left + right)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(np.sum(first[mask] * np.log(first[mask] / second[mask])))

    return 0.5 * kl(left, midpoint) + 0.5 * kl(right, midpoint)


def _wasserstein_integer_support(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(np.cumsum(left) - np.cumsum(right)).sum())


def evaluate_endpoints(
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    train_records: Sequence[GraphRecord],
    heldout_records: Sequence[GraphRecord],
    atom_vocabulary: Sequence[AtomState],
    n_max: int,
) -> dict[str, Any]:
    """Evaluate molecular validity, topology novelty, count fidelity, and sparsity."""

    train_topologies = {topology_hash(record.edges) for record in train_records}
    connected_flags = []
    valid_flags = []
    valid_smiles = []
    valid_topologies = []
    generated_bond_fractions = []
    invalid_reasons: Counter[str] = Counter()
    for nodes, edges in samples:
        connected = _connected(edges)
        connected_flags.append(connected)
        upper = edges[np.triu_indices(len(nodes), 1)]
        generated_bond_fractions.append(float(np.mean(upper != NO_BOND)))
        try:
            with rdBase.BlockLogs():
                molecule = graph_to_molecule(nodes, edges, atom_vocabulary)
                smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        except (ValueError, RuntimeError) as exc:
            valid_flags.append(False)
            invalid_reasons[type(exc).__name__] += 1
            continue
        valid_flags.append(True)
        valid_smiles.append(smiles)
        valid_topologies.append(topology_hash(edges))

    train_counts = [record.node_count for record in train_records]
    generated_counts = [len(nodes) for nodes, _ in samples]
    train_distribution = _distribution(train_counts, n_max)
    generated_distribution = _distribution(generated_counts, n_max)
    heldout_bond_fractions = []
    for record in heldout_records:
        upper = record.edges[np.triu_indices(record.node_count, 1)]
        heldout_bond_fractions.append(float(np.mean(upper != NO_BOND)))
    generated_bond_fraction = float(np.mean(generated_bond_fractions))
    heldout_bond_fraction = float(np.mean(heldout_bond_fractions))
    return {
        "endpoint_validity": len(valid_smiles) / len(samples),
        "connectedness": float(np.mean(connected_flags)),
        "valid_and_connected": sum(
            valid and connected
            for valid, connected in zip(valid_flags, connected_flags, strict=True)
        )
        / len(samples),
        "unique_among_valid": len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0,
        "topology_novelty_among_valid": (
            sum(topology not in train_topologies for topology in valid_topologies)
            / len(valid_topologies)
            if valid_topologies
            else 0.0
        ),
        "atom_count_fidelity": {
            "jensen_shannon_divergence": _jensen_shannon(
                train_distribution, generated_distribution
            ),
            "wasserstein_heavy_atoms": _wasserstein_integer_support(
                train_distribution, generated_distribution
            ),
            "train_mean": float(np.mean(train_counts)),
            "generated_mean": float(np.mean(generated_counts)),
        },
        "bond_sparsity": {
            "heldout_mean_bond_fraction": heldout_bond_fraction,
            "generated_mean_bond_fraction": generated_bond_fraction,
            "absolute_density_error": abs(generated_bond_fraction - heldout_bond_fraction),
        },
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
    }


def _is_sanitizable(
    nodes: np.ndarray,
    edges: np.ndarray,
    atom_vocabulary: Sequence[AtomState],
) -> bool:
    try:
        with rdBase.BlockLogs():
            graph_to_molecule(nodes, edges, atom_vocabulary)
    except (ValueError, RuntimeError):
        return False
    return True


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value * 1024)


def _parameter_count(model: Any) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_one_probe(
    records: Mapping[str, tuple[GraphRecord, ...]],
    atom_vocabulary: Sequence[AtomState],
    config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    objective_config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Train, reconstruct, and sample one fixed N_max dense probe."""

    n_max = int(run_config["n_max"])
    batch_size = int(run_config["batch_size"])
    training_config = dict(config["training"])
    training_config.update(config["model"])
    training_config.update(objective_config)
    train_subset = deterministic_stratified_subset(
        records["R0_train"], int(training_config["train_example_limit"]), seed
    )
    cal_subset = deterministic_stratified_subset(
        records["R0_cal"], int(training_config["cal_example_limit"]), seed + 1
    )
    heldout_subset = deterministic_stratified_subset(
        records["R0_heldout"], int(training_config["heldout_example_limit"]), seed + 2
    )
    node_marginal, edge_marginal = compute_marginals(
        records["R0_train"], len(atom_vocabulary), len(BOND_TYPE_TO_INDEX) + 1
    )
    set_determinism(seed, int(training_config["cpu_threads"]))
    model = DenseGraphFlowProbe(
        node_classes=len(atom_vocabulary),
        edge_classes=len(BOND_TYPE_TO_INDEX) + 1,
        hidden_dim=int(config["model"]["hidden_dim"]),
        layers=int(config["model"]["layers"]),
        dropout=float(config["model"]["dropout"]),
    )
    rss_before = _peak_rss_bytes()
    training = train_probe(
        model,
        train_subset,
        node_marginal,
        edge_marginal,
        n_max,
        batch_size,
        training_config,
        seed + 10,
    )
    calibration = evaluate_reconstruction(
        model,
        cal_subset,
        node_marginal,
        edge_marginal,
        n_max,
        batch_size,
        seed + 20,
    )
    heldout = evaluate_reconstruction(
        model,
        heldout_subset,
        node_marginal,
        edge_marginal,
        n_max,
        batch_size,
        seed + 30,
    )
    samples, sampling = sample_endpoints(
        model,
        records["R0_train"],
        atom_vocabulary,
        node_marginal,
        edge_marginal,
        n_max,
        int(config["sampling"]["endpoint_samples"]),
        int(config["sampling"]["steps"]),
        int(config["sampling"]["batch_size"]),
        seed + 40,
    )
    endpoints = evaluate_endpoints(
        samples,
        records["R0_train"],
        heldout_subset,
        atom_vocabulary,
        n_max,
    )
    rss_after = _peak_rss_bytes()
    return {
        "name": f"{objective_config['name']}_{run_config['name']}",
        "objective": {
            "name": objective_config["name"],
            "purpose": objective_config["purpose"],
            "bond_auxiliary_weight": float(objective_config["bond_auxiliary_weight"]),
        },
        "n_max": n_max,
        "fold_counts_after_support_filters": {
            fold: len(values) for fold, values in records.items()
        },
        "bounded_training_rows": len(train_subset),
        "bounded_calibration_rows": len(cal_subset),
        "bounded_heldout_rows": len(heldout_subset),
        "node_marginal": node_marginal.tolist(),
        "edge_marginal": edge_marginal.tolist(),
        "model": {
            "parameter_count": _parameter_count(model),
            "state_sha256": _model_state_sha256(model),
        },
        "training": training,
        "calibration_reconstruction": calibration,
        "heldout_reconstruction": heldout,
        "sampling": sampling,
        "endpoints": endpoints,
        "memory": {
            "peak_rss_before_bytes": rss_before,
            "peak_rss_after_bytes": rss_after,
            "peak_rss_increment_bytes": max(0, rss_after - rss_before),
            "dense_ordered_edge_slots_per_graph": n_max * n_max,
            "dense_upper_edge_slots_per_graph": n_max * (n_max - 1) // 2,
        },
    }


def make_decision(
    runs: Sequence[Mapping[str, Any]],
    support_audit: Mapping[str, Any],
    thresholds: Mapping[str, float],
    primary_objective: str,
) -> dict[str, Any]:
    """Apply frozen dense-degradation and support-contract rules."""

    by_objective: dict[str, dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for run in runs:
        by_objective[run["objective"]["name"]][int(run["n_max"])] = run
    if primary_objective not in by_objective:
        raise FeasibilityError(f"primary objective is absent: {primary_objective}")
    by_n = by_objective[primary_objective]
    if set(by_n) != {64, 96}:
        raise FeasibilityError("M0-06 requires exactly N_max=64 and N_max=96")
    run64 = by_n[64]
    run96 = by_n[96]
    throughput_ratio = (
        run96["training"]["graphs_per_second"] / run64["training"]["graphs_per_second"]
    )
    recall_drop = (
        run64["heldout_reconstruction"]["metrics"]["bond_recall"]
        - run96["heldout_reconstruction"]["metrics"]["bond_recall"]
    )
    density_error_increase = (
        run96["endpoints"]["bond_sparsity"]["absolute_density_error"]
        - run64["endpoints"]["bond_sparsity"]["absolute_density_error"]
    )
    checks = {
        "n64_endpoint_validity_pass": run64["endpoints"]["endpoint_validity"]
        >= float(thresholds["minimum_endpoint_validity"]),
        "n96_endpoint_validity_pass": run96["endpoints"]["endpoint_validity"]
        >= float(thresholds["minimum_endpoint_validity"]),
        "n64_endpoint_connectedness_pass": run64["endpoints"]["connectedness"]
        >= float(thresholds["minimum_endpoint_connectedness"]),
        "n96_endpoint_connectedness_pass": run96["endpoints"]["connectedness"]
        >= float(thresholds["minimum_endpoint_connectedness"]),
        "n64_heldout_bond_recall_pass": run64["heldout_reconstruction"]["metrics"]["bond_recall"]
        >= float(thresholds["minimum_heldout_bond_recall"]),
        "n96_heldout_bond_recall_pass": run96["heldout_reconstruction"]["metrics"]["bond_recall"]
        >= float(thresholds["minimum_heldout_bond_recall"]),
        "n64_endpoint_bond_density_pass": run64["endpoints"]["bond_sparsity"][
            "absolute_density_error"
        ]
        <= float(thresholds["maximum_endpoint_bond_density_error"]),
        "n96_endpoint_bond_density_pass": run96["endpoints"]["bond_sparsity"][
            "absolute_density_error"
        ]
        <= float(thresholds["maximum_endpoint_bond_density_error"]),
        "n96_training_throughput_ratio_pass": throughput_ratio
        >= float(thresholds["minimum_n96_to_n64_training_throughput_ratio"]),
        "n96_bond_recall_drop_pass": recall_drop
        <= float(thresholds["maximum_n96_bond_recall_drop"]),
        "n96_bond_density_error_increase_pass": density_error_increase
        <= float(thresholds["maximum_n96_bond_density_error_increase"]),
        "memory_pass": max(run["memory"]["peak_rss_after_bytes"] for run in runs)
        <= int(thresholds["maximum_peak_rss_bytes"]),
    }
    absolute_probe_names = {
        "n64_endpoint_validity_pass",
        "n96_endpoint_validity_pass",
        "n64_endpoint_connectedness_pass",
        "n96_endpoint_connectedness_pass",
        "n64_heldout_bond_recall_pass",
        "n96_heldout_bond_recall_pass",
        "n64_endpoint_bond_density_pass",
        "n96_endpoint_bond_density_pass",
    }
    absolute_probe_checks = {
        key: value for key, value in checks.items() if key in absolute_probe_names
    }
    scaling_checks = {
        key: value
        for key, value in checks.items()
        if key
        in {
            "n96_training_throughput_ratio_pass",
            "n96_bond_recall_drop_pass",
            "n96_bond_density_error_increase_pass",
            "memory_pass",
        }
    }
    probe_quality_established = all(absolute_probe_checks.values())
    dense_degraded_at_96 = not all(scaling_checks.values())
    actual_supported = support_audit["declared_element_supported_r0"]
    material_full_corpus_tail = actual_supported["n96_fraction"] < 0.95
    sparse_required_for_declared_corpus = (
        dense_degraded_at_96 or material_full_corpus_tail or not probe_quality_established
    )
    if sparse_required_for_declared_corpus:
        recommendation = (
            "adopt_sparse_or_hierarchical_edge_parameterization_before_full_product_prior"
        )
    else:
        recommendation = "dense_parameterization_feasible_with_declared_n96_support"
    objective_diagnostics = {}
    for objective_name, objective_runs in sorted(by_objective.items()):
        if set(objective_runs) != {64, 96}:
            raise FeasibilityError(f"objective {objective_name} lacks an N=64 or N=96 run")
        objective_diagnostics[objective_name] = {
            str(n_max): {
                "heldout_bond_recall": run["heldout_reconstruction"]["metrics"]["bond_recall"],
                "heldout_bond_probability_density_error": run["heldout_reconstruction"][
                    "bond_sparsity_calibration"
                ]["absolute_density_error"],
                "endpoint_bond_density_error": run["endpoints"]["bond_sparsity"][
                    "absolute_density_error"
                ],
                "endpoint_validity": run["endpoints"]["endpoint_validity"],
                "endpoint_connectedness": run["endpoints"]["connectedness"],
            }
            for n_max, run in sorted(objective_runs.items())
        }
    return {
        "primary_objective": primary_objective,
        "dense_degraded_at_n96": dense_degraded_at_96,
        "bounded_probe_quality_established": probe_quality_established,
        "support_profile_drift": not support_audit["profile_matches"],
        "material_declared_vocabulary_corpus_above_n96": material_full_corpus_tail,
        "checks": checks,
        "comparisons": {
            "n96_to_n64_training_throughput_ratio": throughput_ratio,
            "n96_bond_recall_drop": recall_drop,
            "n96_endpoint_bond_density_error_increase": density_error_increase,
        },
        "objective_diagnostics": objective_diagnostics,
        "recommendation": recommendation,
        "reason": (
            "The bounded dense probe is not permission to discard larger real lipids. "
            "Sparse or hierarchical edges are required if N=96 degrades or if the current "
            "hash-pinned corpus has a material supported tail above N=96."
        ),
    }


def _relative_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def run_feasibility(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute the complete deterministic M0-06 gate."""

    _require_torch()
    config = json.loads(config_path.read_text())
    input_paths = {
        name: _resolve_and_verify(repo, record, name) for name, record in config["inputs"].items()
    }
    rows = _read_csv(input_paths["r0"])
    assignments = _read_csv(input_paths["split_assignments"])
    if len(rows) != len(assignments):
        raise FeasibilityError("R0 and frozen assignment row counts differ")
    declared_elements = set(config["declared_support"]["elements"])
    support_audit = audit_input_support(
        rows,
        declared_elements,
        config["declared_support"]["documented_profile_to_audit"],
        float(config["decision_thresholds"]["support_profile_fraction_tolerance"]),
    )
    if support_audit["invalid_smiles"]:
        raise FeasibilityError("R0 contains invalid SMILES")
    atom_vocabulary = build_atom_vocabulary(rows, declared_elements)
    runs = []
    exclusions = {}
    for objective_config in config["objective_variants"]:
        for size_index, run_config in enumerate(config["runs"]):
            fold_records, run_exclusions = prepare_records(
                rows,
                assignments,
                atom_vocabulary,
                declared_elements,
                f"{config['split_scheme']}_fold",
                int(run_config["n_max"]),
            )
            runs.append(
                run_one_probe(
                    fold_records,
                    atom_vocabulary,
                    config,
                    run_config,
                    objective_config,
                    int(config["seed"]) + size_index * 1000,
                )
            )
            exclusions[f"{objective_config['name']}_{run_config['name']}"] = run_exclusions
    decision = make_decision(
        runs,
        support_audit,
        config["decision_thresholds"],
        config["primary_objective"],
    )
    result = {
        "schema_version": "m0_06_defog_feasibility_result.v1",
        "status": "completed_bounded_probe",
        "scope": (
            "M0 feasibility probe only; this is not the production product-prior training run"
        ),
        "method": {
            "family": "DeFoG-style discrete flow matching",
            "corruption": "linear interpolation between empirical marginal and clean graph",
            "prediction_target": "clean node and upper-triangle edge marginals",
            "edge_representation": "dense symmetric matrix with explicit no-bond class",
            "sampler": "Euler CTMC with minimum R-star rate and direct terminal prediction",
            "node_count": "sampled from empirical R0_train distribution",
            "stereochemistry": config["declared_support"]["stereochemistry_policy"],
            "reference": {
                "paper": "https://openreview.net/forum?id=KPRIwWhqAZ",
                "code": "https://github.com/manuelmlmadeira/DeFoG",
            },
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cpu",
            "cpu_count": os.cpu_count(),
            "deterministic_algorithms": True,
        },
        "inputs": {
            name: {
                "path": _relative_path(path, repo),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for name, path in input_paths.items()
        },
        "config": {
            "path": _relative_path(config_path, repo),
            "sha256": sha256_file(config_path),
        },
        "support_audit": support_audit,
        "atom_vocabulary": [
            {
                "index": index,
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
            }
            for index, state in enumerate(atom_vocabulary)
        ],
        "bond_vocabulary": config["declared_support"]["bond_types"],
        "exclusions": exclusions,
        "runs": runs,
        "decision": decision,
        "limitations": [
            "The fixed training budget tests tractability and failure modes, not converged generative quality.",
            "Flat graph generation omits stereochemical labels under the frozen route-assigned stereo policy.",
            "No pKa, particle-size, PDI, encapsulation, or other unavailable formulation measurements are imputed.",
            "The empirical node-count prior makes count fidelity a sampler-contract check, not a learned node-count result.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
