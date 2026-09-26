"""Bounded sparse, hierarchical discrete-flow probe for molecular topology.

The probe factorizes a connected molecular graph into a rooted spanning tree
and a small set of residual ring-closure edges. Parent pointers and closure
endpoints retain full pair reachability, while bond-state losses are evaluated
only on active molecular edges. This avoids treating every absent atom pair as
an edge class without changing FORGE into fragment or building-block
generation.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import resource
import sys
import time
from collections import Counter, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.model.defog_feasibility import (
    AtomState,
    FeasibilityError,
    _connected,
    _distribution,
    _jensen_shannon,
    _model_state_sha256,
    _parameter_count,
    _rstar_step,
    _size_stratum,
    _wasserstein_integer_support,
    audit_input_support,
    graph_to_molecule,
    set_determinism,
    sha256_file,
    topology_hash,
)
from forge.model.lipid_context import assign_lipid_regions, select_lipid_polar_root

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


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


def _kekulized_molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise FeasibilityError(f"invalid SMILES: {smiles}")
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except (ValueError, RuntimeError) as exc:
        raise FeasibilityError(f"molecule cannot be kekulized: {smiles}") from exc
    return molecule


def build_sparse_atom_vocabulary(
    rows: Sequence[Mapping[str, str]],
    declared_elements: set[str],
    *,
    preserve_aromaticity: bool = False,
) -> tuple[AtomState, ...]:
    """Build atom states inside the declared element support."""

    states: set[AtomState] = set()
    for row in rows:
        if not set(row["elements"].split("|")).issubset(declared_elements):
            continue
        molecule = (
            Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
            if preserve_aromaticity
            else _kekulized_molecule(row["canonical_isomeric_smiles"])
        )
        if molecule is None:
            raise FeasibilityError(f"invalid SMILES: {row['canonical_isomeric_smiles']}")
        states.update(
            AtomState(
                atom.GetSymbol(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic() if preserve_aromaticity else False,
                atom.GetNumExplicitHs() if preserve_aromaticity else 0,
            )
            for atom in molecule.GetAtoms()
        )
    return tuple(sorted(states, key=AtomState.key))


def _canonical_bfs_tree(
    molecule: Chem.Mol,
    *,
    root_strategy: str = "canonical",
) -> tuple[list[int], dict[int, int]]:
    """Return a canonical breadth-first atom order and old-index parent map."""

    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    if root_strategy == "canonical":
        root = min(range(molecule.GetNumAtoms()), key=lambda index: (ranks[index], index))
    elif root_strategy == "lipid_polar":
        root = select_lipid_polar_root(molecule)
    else:
        raise FeasibilityError(f"unsupported sparse root strategy: {root_strategy}")
    order = []
    parents = {root: root}
    queue = deque([root])
    seen = {root}
    while queue:
        atom_index = queue.popleft()
        order.append(atom_index)
        neighbors = sorted(
            (
                neighbor.GetIdx()
                for neighbor in molecule.GetAtomWithIdx(atom_index).GetNeighbors()
                if neighbor.GetIdx() not in seen
            ),
            key=lambda index: (ranks[index], index),
        )
        for neighbor in neighbors:
            seen.add(neighbor)
            parents[neighbor] = atom_index
            queue.append(neighbor)
    if len(order) != molecule.GetNumAtoms():
        raise FeasibilityError("sparse topology requires a connected molecule")
    return order, parents


def tensorize_sparse_row(
    row: Mapping[str, str],
    atom_to_index: Mapping[AtomState, int],
    *,
    preserve_aromaticity: bool = False,
    root_strategy: str = "canonical",
    region_scheme: str = "none",
) -> SparseGraphRecord:
    """Convert a molecule into a canonical tree plus residual closures."""

    molecule = (
        Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        if preserve_aromaticity
        else _kekulized_molecule(row["canonical_isomeric_smiles"])
    )
    if molecule is None:
        raise FeasibilityError(f"invalid SMILES: {row['canonical_isomeric_smiles']}")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise FeasibilityError(f"sparse topology requires one fragment: {row['r0_structure_id']}")
    order, old_parent = _canonical_bfs_tree(
        molecule,
        root_strategy=root_strategy,
    )
    if region_scheme == "none":
        region_states = None
    elif region_scheme == "polar_structural_v2":
        region_by_old_index = assign_lipid_regions(molecule, order[0])
        region_states = np.asarray(
            [region_by_old_index[old] for old in order],
            dtype=np.int64,
        )
    else:
        raise FeasibilityError(f"unsupported lipid region scheme: {region_scheme}")
    old_to_new = {old: new for new, old in enumerate(order)}
    node_states = np.asarray(
        [
            atom_to_index[
                AtomState(
                    molecule.GetAtomWithIdx(old).GetSymbol(),
                    molecule.GetAtomWithIdx(old).GetFormalCharge(),
                    (
                        molecule.GetAtomWithIdx(old).GetIsAromatic()
                        if preserve_aromaticity
                        else False
                    ),
                    (
                        molecule.GetAtomWithIdx(old).GetNumExplicitHs()
                        if preserve_aromaticity
                        else 0
                    ),
                )
            ]
            for old in order
        ],
        dtype=np.int64,
    )
    parents = np.zeros(len(order), dtype=np.int64)
    parent_bonds = np.zeros(len(order), dtype=np.int64)
    tree_edges: set[tuple[int, int]] = set()
    for new_child, old_child in enumerate(order[1:], start=1):
        old_parent_index = old_parent[old_child]
        new_parent = old_to_new[old_parent_index]
        parents[new_child] = new_parent
        pair = tuple(sorted((new_child, new_parent)))
        tree_edges.add(pair)
        bond = molecule.GetBondBetweenAtoms(old_child, old_parent_index)
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(f"unsupported kekulized bond type: {bond.GetBondType()}")
        parent_bonds[new_child] = SPARSE_BOND_TO_INDEX[bond.GetBondType()]

    closures = []
    for bond in molecule.GetBonds():
        left = old_to_new[bond.GetBeginAtomIdx()]
        right = old_to_new[bond.GetEndAtomIdx()]
        pair = tuple(sorted((left, right)))
        if pair in tree_edges:
            continue
        if bond.GetBondType() not in SPARSE_BOND_TO_INDEX:
            raise FeasibilityError(f"unsupported kekulized closure bond: {bond.GetBondType()}")
        closures.append((*pair, SPARSE_BOND_TO_INDEX[bond.GetBondType()]))
    closures.sort()
    closure_left = np.asarray([row[0] for row in closures], dtype=np.int64)
    closure_right = np.asarray([row[1] for row in closures], dtype=np.int64)
    closure_bonds = np.asarray([row[2] for row in closures], dtype=np.int64)

    edges = np.zeros((len(order), len(order)), dtype=np.int64)
    for child in range(1, len(order)):
        parent = int(parents[child])
        edge = INDEX_TO_DENSE_BOND[int(parent_bonds[child])]
        edges[child, parent] = edges[parent, child] = edge
    for left, right, bond in closures:
        edge = INDEX_TO_DENSE_BOND[bond]
        edges[left, right] = edges[right, left] = edge
    return SparseGraphRecord(
        structure_id=row["r0_structure_id"],
        canonical_smiles=row["canonical_isomeric_smiles"],
        node_states=node_states,
        parents=parents,
        parent_bonds=parent_bonds,
        closure_left=closure_left,
        closure_right=closure_right,
        closure_bonds=closure_bonds,
        edges=edges,
        region_states=region_states,
    )


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


def _record_contains_aromatic_atom(record: SparseGraphRecord) -> bool:
    molecule = Chem.MolFromSmiles(record.canonical_smiles)
    if molecule is None:
        raise FeasibilityError(f"invalid original SMILES: {record.canonical_smiles}")
    return any(atom.GetIsAromatic() for atom in molecule.GetAtoms())


def prepare_sparse_records(
    rows: Sequence[Mapping[str, str]],
    assignments: Sequence[Mapping[str, str]],
    atom_vocabulary: Sequence[AtomState],
    declared_elements: set[str],
    fold_column: str,
    n_max: int | None,
    maximum_closures: int,
) -> tuple[dict[str, tuple[SparseGraphRecord, ...]], dict[str, int]]:
    """Tensorize fold-separated records and expose every support exclusion."""

    fold_by_id = {row["r0_structure_id"]: row[fold_column] for row in assignments}
    atom_to_index = {state: index for index, state in enumerate(atom_vocabulary)}
    records: dict[str, list[SparseGraphRecord]] = {
        "R0_train": [],
        "R0_cal": [],
        "R0_heldout": [],
    }
    exclusions = Counter()
    for row in rows:
        if not set(row["elements"].split("|")).issubset(declared_elements):
            exclusions["outside_element_vocabulary"] += 1
            continue
        if n_max is not None and int(row["heavy_atoms"]) > n_max:
            exclusions["over_n_max"] += 1
            continue
        record = tensorize_sparse_row(row, atom_to_index)
        if record.closure_count > maximum_closures:
            exclusions["over_closure_support"] += 1
            continue
        if not sparse_roundtrip_exact(record):
            raise FeasibilityError(f"sparse roundtrip failed: {record.structure_id}")
        records[fold_by_id[record.structure_id]].append(record)
    return (
        {
            fold: tuple(sorted(values, key=lambda record: record.structure_id))
            for fold, values in records.items()
        },
        dict(sorted(exclusions.items())),
    )


def _stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def deterministic_sparse_subset(
    records: Sequence[SparseGraphRecord], limit: int, seed: int
) -> tuple[SparseGraphRecord, ...]:
    if len(records) <= limit:
        return tuple(records)
    buckets: dict[str, list[SparseGraphRecord]] = {}
    for record in records:
        buckets.setdefault(_size_stratum(record.node_count), []).append(record)
    for values in buckets.values():
        values.sort(key=lambda record: _stable_rank(record.structure_id, seed))
    positions = {name: 0 for name in buckets}
    selected = []
    while len(selected) < limit:
        progressed = False
        for name in sorted(buckets):
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


def sparse_marginals(
    records: Sequence[SparseGraphRecord], node_classes: int
) -> tuple[np.ndarray, np.ndarray]:
    node_counts = np.zeros(node_classes, dtype=np.float64)
    bond_counts = np.zeros(len(SPARSE_BOND_TO_INDEX), dtype=np.float64)
    for record in records:
        node_counts += np.bincount(record.node_states, minlength=node_classes)
        bond_counts += np.bincount(
            np.concatenate((record.parent_bonds[1:], record.closure_bonds)),
            minlength=len(SPARSE_BOND_TO_INDEX),
        )
    if np.any(node_counts == 0) or np.any(bond_counts == 0):
        raise FeasibilityError("every sparse node and bond state must occur in training")
    return node_counts / node_counts.sum(), bond_counts / bond_counts.sum()


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


def _noise_sparse_batch(
    clean: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    node_mask = clean["node_mask"]
    child_mask = clean["child_mask"]
    closure_mask = clean["closure_mask"]
    parent_candidates = _parent_candidate_mask(node_mask)
    endpoint_candidates = _endpoint_candidate_mask(node_mask, clean["closure_left"].shape[1])
    return {
        "nodes": _sample_flat_interpolation(clean["nodes"], node_marginal, t, node_mask, generator),
        "parents": sample_pointer_interpolation(
            clean["parents"], parent_candidates, child_mask, t, generator
        ),
        "parent_bonds": _sample_flat_interpolation(
            clean["parent_bonds"], bond_marginal, t, child_mask, generator
        ),
        "closure_left": sample_pointer_interpolation(
            clean["closure_left"],
            endpoint_candidates,
            closure_mask,
            t,
            generator,
        ),
        "closure_right": sample_pointer_interpolation(
            clean["closure_right"],
            endpoint_candidates,
            closure_mask,
            t,
            generator,
        ),
        "closure_bonds": _sample_flat_interpolation(
            clean["closure_bonds"], bond_marginal, t, closure_mask, generator
        ),
    }


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


def _masked_sparse_losses(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> tuple[Any, dict[str, float]]:
    node_mask = clean["node_mask"]
    child_mask = clean["child_mask"]
    closure_mask = clean["closure_mask"]
    parent_candidates = _parent_candidate_mask(node_mask)
    endpoint_candidates = _endpoint_candidate_mask(node_mask, clean["closure_left"].shape[1])
    node_loss = functional.cross_entropy(predictions["nodes"][node_mask], clean["nodes"][node_mask])
    parent_logits = predictions["parents"].masked_fill(~parent_candidates, -1e9)
    parent_loss = functional.cross_entropy(parent_logits[child_mask], clean["parents"][child_mask])
    parent_bond_loss = functional.cross_entropy(
        predictions["parent_bonds"][child_mask],
        clean["parent_bonds"][child_mask],
    )
    if closure_mask.any():
        left_logits = predictions["closure_left"].masked_fill(~endpoint_candidates, -1e9)
        right_logits = predictions["closure_right"].masked_fill(~endpoint_candidates, -1e9)
        left_loss = functional.cross_entropy(
            left_logits[closure_mask], clean["closure_left"][closure_mask]
        )
        right_loss = functional.cross_entropy(
            right_logits[closure_mask], clean["closure_right"][closure_mask]
        )
        closure_bond_loss = functional.cross_entropy(
            predictions["closure_bonds"][closure_mask],
            clean["closure_bonds"][closure_mask],
        )
    else:
        left_loss = node_loss * 0.0
        right_loss = node_loss * 0.0
        closure_bond_loss = node_loss * 0.0
    total = node_loss + parent_loss + parent_bond_loss + left_loss + right_loss + closure_bond_loss
    components = {
        "node_ce": float(node_loss.detach()),
        "parent_pointer_ce": float(parent_loss.detach()),
        "backbone_bond_ce": float(parent_bond_loss.detach()),
        "closure_left_ce": float(left_loss.detach()),
        "closure_right_ce": float(right_loss.detach()),
        "closure_bond_ce": float(closure_bond_loss.detach()),
        "total": float(total.detach()),
    }
    return total, components


def _batch_records(
    records: Sequence[SparseGraphRecord],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[SparseGraphRecord, ...]:
    indices = rng.integers(0, len(records), size=batch_size)
    return tuple(records[int(index)] for index in indices)


def train_sparse_probe(
    model: Any,
    records: Sequence[SparseGraphRecord],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    n_max: int,
    maximum_closures: int,
    batch_size: int,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32)
    bond_p0 = torch.tensor(bond_marginal, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    losses = []
    start = time.perf_counter()
    model.train()
    for _ in range(int(config["steps"])):
        batch = _batch_records(records, batch_size, rng)
        clean = collate_sparse_records(batch, n_max, maximum_closures)
        t = torch.rand(batch_size, generator=generator).clamp(0.02, 0.98)
        noisy = _noise_sparse_batch(clean, node_p0, bond_p0, t, generator)
        predictions = model(
            noisy["nodes"],
            noisy["parents"],
            noisy["parent_bonds"],
            noisy["closure_left"],
            noisy["closure_right"],
            t,
            clean["node_mask"],
            clean["child_mask"],
        )
        loss, components = _masked_sparse_losses(predictions, clean)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["gradient_clip_norm"]))
        optimizer.step()
        losses.append(components)
    elapsed = time.perf_counter() - start
    return {
        "steps": int(config["steps"]),
        "examples_seen_with_replacement": int(config["steps"]) * batch_size,
        "wall_seconds": elapsed,
        "graphs_per_second": int(config["steps"]) * batch_size / elapsed,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "mean_last_20_loss": {
            key: float(np.mean([row[key] for row in losses[-20:]])) for key in losses[-1]
        },
    }


def evaluate_sparse_reconstruction(
    model: Any,
    records: Sequence[SparseGraphRecord],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    n_max: int,
    maximum_closures: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32)
    bond_p0 = torch.tensor(bond_marginal, dtype=torch.float32)
    metrics: dict[str, list[float]] = {}
    by_size: dict[str, dict[str, list[float]]] = {}
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(records), batch_size):
            batch = tuple(records[offset : offset + batch_size])
            clean = collate_sparse_records(batch, n_max, maximum_closures)
            t = torch.tensor([0.25 + 0.25 * ((offset + index) % 3) for index in range(len(batch))])
            noisy = _noise_sparse_batch(clean, node_p0, bond_p0, t, generator)
            predictions = model(
                noisy["nodes"],
                noisy["parents"],
                noisy["parent_bonds"],
                noisy["closure_left"],
                noisy["closure_right"],
                t,
                clean["node_mask"],
                clean["child_mask"],
            )
            parent_candidates = _parent_candidate_mask(clean["node_mask"])
            endpoint_candidates = _endpoint_candidate_mask(clean["node_mask"], maximum_closures)
            predicted = {
                "nodes": predictions["nodes"].argmax(dim=-1),
                "parents": predictions["parents"]
                .masked_fill(~parent_candidates, -1e9)
                .argmax(dim=-1),
                "parent_bonds": predictions["parent_bonds"].argmax(dim=-1),
                "closure_left": predictions["closure_left"]
                .masked_fill(~endpoint_candidates, -1e9)
                .argmax(dim=-1),
                "closure_right": predictions["closure_right"]
                .masked_fill(~endpoint_candidates, -1e9)
                .argmax(dim=-1),
                "closure_bonds": predictions["closure_bonds"].argmax(dim=-1),
            }
            for index, record in enumerate(batch):
                node_mask = clean["node_mask"][index]
                child_mask = clean["child_mask"][index]
                closure_mask = clean["closure_mask"][index]
                row = {
                    "node_accuracy": float(
                        (predicted["nodes"][index][node_mask] == clean["nodes"][index][node_mask])
                        .float()
                        .mean()
                    ),
                    "parent_accuracy": float(
                        (
                            predicted["parents"][index][child_mask]
                            == clean["parents"][index][child_mask]
                        )
                        .float()
                        .mean()
                    ),
                    "backbone_bond_accuracy": float(
                        (
                            predicted["parent_bonds"][index][child_mask]
                            == clean["parent_bonds"][index][child_mask]
                        )
                        .float()
                        .mean()
                    ),
                    "closure_pair_accuracy": _closure_pair_accuracy(
                        predicted, clean, index, closure_mask
                    ),
                    "closure_bond_accuracy": _closure_bond_accuracy(
                        predicted, clean, index, closure_mask
                    ),
                }
                row["exact_sparse_state"] = float(
                    all(
                        value == 1.0
                        for key, value in row.items()
                        if key != "closure_pair_accuracy" or bool(closure_mask.any())
                    )
                )
                stratum = _size_stratum(record.node_count)
                by_size.setdefault(stratum, {})
                for key, value in row.items():
                    metrics.setdefault(key, []).append(value)
                    by_size[stratum].setdefault(key, []).append(value)
    elapsed = time.perf_counter() - start
    return {
        "rows": len(records),
        "wall_seconds": elapsed,
        "graphs_per_second": len(records) / elapsed,
        "metrics": {key: float(np.mean(values)) for key, values in metrics.items()},
        "by_size": {
            stratum: {
                "rows": len(next(iter(values.values()))),
                **{key: float(np.mean(items)) for key, items in values.items()},
            }
            for stratum, values in sorted(by_size.items())
        },
    }


def _closure_pair_accuracy(
    predicted: Mapping[str, Any],
    clean: Mapping[str, Any],
    index: int,
    mask: Any,
) -> float:
    if not mask.any():
        return 1.0
    left = predicted["closure_left"][index][mask]
    right = predicted["closure_right"][index][mask]
    true_left = clean["closure_left"][index][mask]
    true_right = clean["closure_right"][index][mask]
    direct = (left == true_left) & (right == true_right)
    reverse = (left == true_right) & (right == true_left)
    return float((direct | reverse).float().mean())


def _closure_bond_accuracy(
    predicted: Mapping[str, Any],
    clean: Mapping[str, Any],
    index: int,
    mask: Any,
) -> float:
    if not mask.any():
        return 1.0
    return float(
        (predicted["closure_bonds"][index][mask] == clean["closure_bonds"][index][mask])
        .float()
        .mean()
    )


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


def _sample_from_logits(logits: Any, valid: Any, generator: Any) -> tuple[int, bool]:
    masked = logits.masked_fill(~valid, -1e9)
    if not valid.any():
        return 0, True
    probability = masked.softmax(dim=-1)
    return int(torch.multinomial(probability, 1, generator=generator)), False


def _construct_terminal_graph(
    predictions: Mapping[str, Any],
    index: int,
    node_count: int,
    requested_closures: int,
    atom_vocabulary: Sequence[AtomState],
    maximum_closures: int,
    generator: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Construct a connected valence-masked graph from terminal marginals."""

    max_degree = max(_maximum_valence_units(state) // 2 for state in atom_vocabulary)
    parents = np.zeros(node_count, dtype=np.int64)
    degrees = np.zeros(node_count, dtype=np.int64)
    repairs = Counter()
    for child in range(1, node_count):
        valid = torch.zeros(node_count, dtype=torch.bool)
        valid[:child] = torch.tensor(degrees[:child] < max_degree)
        parent, repaired = _sample_from_logits(
            predictions["parents"][index, child, :node_count], valid, generator
        )
        if repaired:
            parent = int(np.argmin(degrees[:child]))
            repairs["parent_capacity_fallback"] += 1
        parents[child] = parent
        degrees[child] += 1
        degrees[parent] += 1

    node_states = np.zeros(node_count, dtype=np.int64)
    capacities = np.zeros(node_count, dtype=np.int64)
    for node in range(node_count):
        valid_states = torch.tensor(
            [_maximum_valence_units(state) >= 2 * degrees[node] for state in atom_vocabulary],
            dtype=torch.bool,
        )
        state, repaired = _sample_from_logits(
            predictions["nodes"][index, node], valid_states, generator
        )
        if repaired:
            state = max(
                range(len(atom_vocabulary)),
                key=lambda candidate: _maximum_valence_units(atom_vocabulary[candidate]),
            )
            repairs["node_valence_fallback"] += 1
        node_states[node] = state
        capacities[node] = _maximum_valence_units(atom_vocabulary[state])

    edges = np.zeros((node_count, node_count), dtype=np.int64)
    used_units = 2 * degrees
    for child in range(1, node_count):
        parent = int(parents[child])
        spare = min(
            capacities[child] - used_units[child],
            capacities[parent] - used_units[parent],
        )
        bond_valence_units = BOND_VALENCE_UNITS[: predictions["parent_bonds"].shape[-1]]
        valid_bonds = bond_valence_units <= 2 + max(0, spare)
        bond, repaired = _sample_from_logits(
            predictions["parent_bonds"][index, child], valid_bonds, generator
        )
        if repaired:
            bond = 0
            repairs["backbone_bond_fallback"] += 1
        extra = int(bond_valence_units[bond]) - 2
        used_units[child] += extra
        used_units[parent] += extra
        dense_bond = INDEX_TO_DENSE_BOND[bond]
        edges[child, parent] = edges[parent, child] = dense_bond

    realized_closures = 0
    for slot in range(min(requested_closures, maximum_closures)):
        left_logits = predictions["closure_left"][index, slot, :node_count]
        right_logits = predictions["closure_right"][index, slot, :node_count]
        pair_logits = left_logits[:, None] + right_logits[None, :]
        valid_pairs = torch.triu(torch.ones((node_count, node_count), dtype=torch.bool), diagonal=1)
        valid_pairs &= torch.tensor(edges == 0)
        spare_nodes = torch.tensor(capacities - used_units >= 2)
        valid_pairs &= spare_nodes[:, None] & spare_nodes[None, :]
        if not valid_pairs.any():
            repairs["closure_omitted_no_valence_pair"] += requested_closures - slot
            break
        flat_pair, _ = _sample_from_logits(pair_logits.flatten(), valid_pairs.flatten(), generator)
        left, right = divmod(flat_pair, node_count)
        spare = min(
            capacities[left] - used_units[left],
            capacities[right] - used_units[right],
        )
        bond_valence_units = BOND_VALENCE_UNITS[: predictions["closure_bonds"].shape[-1]]
        valid_bonds = bond_valence_units <= spare
        bond, repaired = _sample_from_logits(
            predictions["closure_bonds"][index, slot], valid_bonds, generator
        )
        if repaired:
            bond = 0
            repairs["closure_bond_fallback"] += 1
        units = int(bond_valence_units[bond])
        used_units[left] += units
        used_units[right] += units
        dense_bond = INDEX_TO_DENSE_BOND[bond]
        edges[left, right] = edges[right, left] = dense_bond
        realized_closures += 1
    repairs["requested_closures"] = requested_closures
    repairs["realized_closures"] = realized_closures
    return node_states, edges, dict(repairs)


def sample_sparse_endpoints(
    model: Any,
    train_records: Sequence[SparseGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    n_max: int,
    maximum_closures: int,
    sample_count: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32)
    bond_p0 = torch.tensor(bond_marginal, dtype=torch.float32)
    node_counts = rng.choice(
        np.asarray([record.node_count for record in train_records]),
        size=sample_count,
        replace=True,
    )
    closure_counts = rng.choice(
        np.asarray([record.closure_count for record in train_records]),
        size=sample_count,
        replace=True,
    )
    samples = []
    repair_totals = Counter()
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, sample_count, batch_size):
            counts = node_counts[offset : offset + batch_size]
            requested = closure_counts[offset : offset + batch_size]
            local_batch = len(counts)
            node_mask = torch.arange(n_max)[None, :] < torch.tensor(counts)[:, None]
            child_mask = node_mask.clone()
            child_mask[:, 0] = False
            closure_mask = (
                torch.arange(maximum_closures)[None, :] < torch.tensor(requested)[:, None]
            )
            parent_candidates = _parent_candidate_mask(node_mask)
            endpoint_candidates = _endpoint_candidate_mask(node_mask, maximum_closures)
            nodes = torch.zeros((local_batch, n_max), dtype=torch.long)
            nodes[node_mask] = torch.multinomial(
                node_p0,
                int(node_mask.sum()),
                replacement=True,
                generator=generator,
            )
            parents = torch.zeros((local_batch, n_max), dtype=torch.long)
            parents[child_mask] = torch.multinomial(
                (
                    parent_candidates.to(torch.float32)
                    / parent_candidates.sum(dim=-1, keepdim=True).clamp(min=1)
                )[child_mask],
                1,
                generator=generator,
            ).squeeze(1)
            parent_bonds = torch.zeros((local_batch, n_max), dtype=torch.long)
            parent_bonds[child_mask] = torch.multinomial(
                bond_p0,
                int(child_mask.sum()),
                replacement=True,
                generator=generator,
            )
            closure_left = torch.zeros((local_batch, maximum_closures), dtype=torch.long)
            closure_right = torch.zeros_like(closure_left)
            if closure_mask.any():
                endpoint_p0 = endpoint_candidates.to(torch.float32) / endpoint_candidates.sum(
                    dim=-1, keepdim=True
                ).clamp(min=1)
                closure_left[closure_mask] = torch.multinomial(
                    endpoint_p0[closure_mask], 1, generator=generator
                ).squeeze(1)
                closure_right[closure_mask] = torch.multinomial(
                    endpoint_p0[closure_mask], 1, generator=generator
                ).squeeze(1)
            closure_bonds = torch.zeros_like(closure_left)
            if closure_mask.any():
                closure_bonds[closure_mask] = torch.multinomial(
                    bond_p0,
                    int(closure_mask.sum()),
                    replacement=True,
                    generator=generator,
                )

            predictions = None
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full((local_batch,), t_value)
                predictions = model(
                    nodes,
                    parents,
                    parent_bonds,
                    closure_left,
                    closure_right,
                    t,
                    node_mask,
                    child_mask,
                )
                if step == sample_steps - 1:
                    break
                dt = 1.0 / sample_steps
                nodes = _rstar_step(
                    nodes,
                    predictions["nodes"].softmax(dim=-1),
                    node_p0,
                    t_value,
                    dt,
                    node_mask,
                    generator,
                )
                parents = pointer_rstar_step(
                    parents,
                    predictions["parents"],
                    parent_candidates,
                    child_mask,
                    t_value,
                    dt,
                    generator,
                )
                parent_bonds = _rstar_step(
                    parent_bonds,
                    predictions["parent_bonds"].softmax(dim=-1),
                    bond_p0,
                    t_value,
                    dt,
                    child_mask,
                    generator,
                )
                if closure_mask.any():
                    closure_left = pointer_rstar_step(
                        closure_left,
                        predictions["closure_left"],
                        endpoint_candidates,
                        closure_mask,
                        t_value,
                        dt,
                        generator,
                    )
                    closure_right = pointer_rstar_step(
                        closure_right,
                        predictions["closure_right"],
                        endpoint_candidates,
                        closure_mask,
                        t_value,
                        dt,
                        generator,
                    )
                    closure_bonds = _rstar_step(
                        closure_bonds,
                        predictions["closure_bonds"].softmax(dim=-1),
                        bond_p0,
                        t_value,
                        dt,
                        closure_mask,
                        generator,
                    )
            assert predictions is not None
            for index, count in enumerate(counts):
                node_states, edges, repairs = _construct_terminal_graph(
                    predictions,
                    index,
                    int(count),
                    int(requested[index]),
                    atom_vocabulary,
                    maximum_closures,
                    generator,
                )
                samples.append((node_states, edges))
                repair_totals.update(repairs)
    elapsed = time.perf_counter() - start
    terminal_decisions = int(
        sum(len(nodes) for nodes, _ in samples)
        + sum(np.count_nonzero(np.triu(edges, 1)) for _, edges in samples)
    )
    fallback_repairs = int(
        sum(value for key, value in repair_totals.items() if "fallback" in key or "omitted" in key)
    )
    return samples, {
        "samples": sample_count,
        "sampling_steps": sample_steps,
        "wall_seconds": elapsed,
        "graph_steps_per_second": sample_count * sample_steps / elapsed,
        "node_count_source": "empirical_R0_train_distribution_conditioned_on_n_max",
        "closure_count_source": "empirical_R0_train_cycle_rank_distribution",
        "terminal_constraint_events": dict(sorted(repair_totals.items())),
        "terminal_decisions": terminal_decisions,
        "terminal_constraint_repair_fraction": fallback_repairs / max(1, terminal_decisions),
    }


def evaluate_sparse_endpoints(
    samples: Sequence[tuple[np.ndarray, np.ndarray]],
    train_records: Sequence[SparseGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    n_max: int,
) -> dict[str, Any]:
    train_topologies = {topology_hash(record.edges) for record in train_records}
    valid_flags = []
    connected_flags = []
    valid_smiles = []
    valid_topologies = []
    generated_cycles = []
    invalid_reasons = Counter()
    for nodes, edges in samples:
        connected_flags.append(_connected(edges))
        edge_count = int(np.count_nonzero(np.triu(edges, 1)))
        generated_cycles.append(edge_count - len(nodes) + 1)
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
    train_count_distribution = _distribution(train_counts, n_max)
    generated_count_distribution = _distribution(generated_counts, n_max)
    train_cycles = [record.closure_count for record in train_records]
    max_cycles = max(max(train_cycles), max(generated_cycles))
    train_cycle_distribution = _distribution(train_cycles, max_cycles)
    generated_cycle_distribution = _distribution(generated_cycles, max_cycles)
    valid_and_connected = sum(
        valid and connected for valid, connected in zip(valid_flags, connected_flags, strict=True)
    )
    return {
        "endpoint_validity": float(np.mean(valid_flags)),
        "connectedness": float(np.mean(connected_flags)),
        "valid_and_connected": valid_and_connected / len(samples),
        "unique_among_valid": len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0,
        "topology_novelty_among_valid": (
            sum(topology not in train_topologies for topology in valid_topologies)
            / len(valid_topologies)
            if valid_topologies
            else 0.0
        ),
        "atom_count_fidelity": {
            "jensen_shannon_divergence": _jensen_shannon(
                train_count_distribution, generated_count_distribution
            ),
            "wasserstein_heavy_atoms": _wasserstein_integer_support(
                train_count_distribution, generated_count_distribution
            ),
            "train_mean": float(np.mean(train_counts)),
            "generated_mean": float(np.mean(generated_counts)),
        },
        "cycle_rank_fidelity": {
            "jensen_shannon_divergence": _jensen_shannon(
                train_cycle_distribution, generated_cycle_distribution
            ),
            "wasserstein_cycle_rank": _wasserstein_integer_support(
                train_cycle_distribution, generated_cycle_distribution
            ),
            "train_mean": float(np.mean(train_cycles)),
            "generated_mean": float(np.mean(generated_cycles)),
            "generated_maximum": max(generated_cycles),
        },
        "invalid_reason_counts": dict(sorted(invalid_reasons.items())),
    }


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def run_sparse_probe(
    records: Mapping[str, tuple[SparseGraphRecord, ...]],
    atom_vocabulary: Sequence[AtomState],
    config: Mapping[str, Any],
    run_config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    n_max = int(run_config["n_max"])
    batch_size = int(run_config["batch_size"])
    maximum_closures = int(config["model"]["maximum_closure_slots"])
    train_subset = deterministic_sparse_subset(
        records["R0_train"], int(config["training"]["train_example_limit"]), seed
    )
    cal_subset = deterministic_sparse_subset(
        records["R0_cal"], int(config["training"]["cal_example_limit"]), seed + 1
    )
    heldout_subset = deterministic_sparse_subset(
        records["R0_heldout"],
        int(config["training"]["heldout_example_limit"]),
        seed + 2,
    )
    node_marginal, bond_marginal = sparse_marginals(records["R0_train"], len(atom_vocabulary))
    set_determinism(seed, int(config["training"]["cpu_threads"]))
    model = SparseTopologyFlowProbe(
        len(atom_vocabulary),
        len(SPARSE_BOND_TO_INDEX),
        int(config["model"]["hidden_dim"]),
        int(config["model"]["layers"]),
        maximum_closures,
        float(config["model"]["dropout"]),
    )
    rss_before = _peak_rss_bytes()
    training = train_sparse_probe(
        model,
        train_subset,
        node_marginal,
        bond_marginal,
        n_max,
        maximum_closures,
        batch_size,
        config["training"],
        seed + 10,
    )
    calibration = evaluate_sparse_reconstruction(
        model,
        cal_subset,
        node_marginal,
        bond_marginal,
        n_max,
        maximum_closures,
        batch_size,
        seed + 20,
    )
    heldout = evaluate_sparse_reconstruction(
        model,
        heldout_subset,
        node_marginal,
        bond_marginal,
        n_max,
        maximum_closures,
        batch_size,
        seed + 30,
    )
    samples, sampling = sample_sparse_endpoints(
        model,
        records["R0_train"],
        atom_vocabulary,
        node_marginal,
        bond_marginal,
        n_max,
        maximum_closures,
        int(config["sampling"]["endpoint_samples"]),
        int(config["sampling"]["steps"]),
        int(config["sampling"]["batch_size"]),
        seed + 40,
    )
    endpoints = evaluate_sparse_endpoints(samples, records["R0_train"], atom_vocabulary, n_max)
    rss_after = _peak_rss_bytes()
    return {
        "name": run_config["name"],
        "n_max": n_max,
        "fold_counts_after_support_filters": {
            fold: len(values) for fold, values in records.items()
        },
        "bounded_training_rows": len(train_subset),
        "bounded_calibration_rows": len(cal_subset),
        "bounded_heldout_rows": len(heldout_subset),
        "node_marginal": node_marginal.tolist(),
        "bond_marginal": bond_marginal.tolist(),
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
            "active_backbone_edges_per_graph": n_max - 1,
            "maximum_closure_edges_per_graph": maximum_closures,
            "parent_pointer_logits_per_graph": n_max * n_max,
            "closure_endpoint_logits_per_graph": 2 * maximum_closures * n_max,
            "dense_edge_hidden_states_eliminated": n_max * n_max,
        },
        "_model_object": model,
    }


def scaling_dry_runs(
    model: Any,
    settings: Sequence[Mapping[str, int]],
    maximum_closures: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Measure forward scaling through the current full-corpus maximum."""

    set_determinism(seed, min(8, os.cpu_count() or 1))
    results = []
    model.eval()
    for setting in settings:
        n_max = int(setting["n_max"])
        batch_size = int(setting["batch_size"])
        nodes = torch.zeros((batch_size, n_max), dtype=torch.long)
        parents = torch.arange(n_max)[None, :].repeat(batch_size, 1)
        parents[:, 1:] -= 1
        parent_bonds = torch.zeros_like(nodes)
        closure_left = torch.zeros((batch_size, maximum_closures), dtype=torch.long)
        closure_right = torch.ones_like(closure_left)
        node_mask = torch.ones_like(nodes, dtype=torch.bool)
        child_mask = node_mask.clone()
        child_mask[:, 0] = False
        t = torch.full((batch_size,), 0.5)
        start_rss = _peak_rss_bytes()
        start = time.perf_counter()
        iterations = 10
        with torch.no_grad():
            for _ in range(iterations):
                output = model(
                    nodes,
                    parents,
                    parent_bonds,
                    closure_left,
                    closure_right,
                    t,
                    node_mask,
                    child_mask,
                )
                _ = output["parents"].sum().item()
        elapsed = time.perf_counter() - start
        end_rss = _peak_rss_bytes()
        hidden_dim = int(model.hidden_dim)
        results.append(
            {
                "n_max": n_max,
                "batch_size": batch_size,
                "iterations": iterations,
                "graphs_per_second": batch_size * iterations / elapsed,
                "peak_rss_before_bytes": start_rss,
                "peak_rss_after_bytes": end_rss,
                "parent_pointer_logits_per_graph": n_max * n_max,
                "closure_endpoint_logits_per_graph": 2 * maximum_closures * n_max,
                "sparse_hidden_state_scalars_per_graph": n_max * hidden_dim,
                "dense_edge_hidden_state_scalars_avoided_per_graph": n_max * n_max * hidden_dim,
            }
        )
    return results


def make_sparse_decision(
    runs: Sequence[Mapping[str, Any]],
    scaling: Sequence[Mapping[str, Any]],
    representation_roundtrip: float,
    constitutional_roundtrip: float,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    by_n = {int(run["n_max"]): run for run in runs}
    run64 = by_n[64]
    run96 = by_n[96]
    scaling_282 = next(row for row in scaling if row["n_max"] == 282)
    checks = {
        "representation_roundtrip_pass": representation_roundtrip
        >= float(thresholds["minimum_representation_roundtrip"]),
        "constitutional_roundtrip_pass": constitutional_roundtrip
        >= float(
            thresholds.get(
                "minimum_constitutional_roundtrip",
                thresholds["minimum_representation_roundtrip"],
            )
        ),
        "n64_endpoint_validity_pass": run64["endpoints"]["endpoint_validity"]
        >= float(thresholds["minimum_endpoint_validity"]),
        "n96_endpoint_validity_pass": run96["endpoints"]["endpoint_validity"]
        >= float(thresholds["minimum_endpoint_validity"]),
        "n64_endpoint_connectedness_pass": run64["endpoints"]["connectedness"]
        >= float(thresholds["minimum_endpoint_connectedness"]),
        "n96_endpoint_connectedness_pass": run96["endpoints"]["connectedness"]
        >= float(thresholds["minimum_endpoint_connectedness"]),
        "n64_valid_and_connected_pass": run64["endpoints"]["valid_and_connected"]
        >= float(thresholds["minimum_valid_and_connected"]),
        "n96_valid_and_connected_pass": run96["endpoints"]["valid_and_connected"]
        >= float(thresholds["minimum_valid_and_connected"]),
        "n96_to_n64_throughput_pass": (
            run96["training"]["graphs_per_second"] / run64["training"]["graphs_per_second"]
        )
        >= float(thresholds["minimum_n96_to_n64_training_throughput_ratio"]),
        "n64_topology_novelty_pass": run64["endpoints"]["topology_novelty_among_valid"]
        >= float(thresholds["minimum_topology_novelty_among_valid"]),
        "n96_topology_novelty_pass": run96["endpoints"]["topology_novelty_among_valid"]
        >= float(thresholds["minimum_topology_novelty_among_valid"]),
        "n282_memory_pass": scaling_282["peak_rss_after_bytes"]
        <= int(thresholds["maximum_peak_rss_bytes_at_282"]),
        "n64_terminal_repair_pass": run64["sampling"]["terminal_constraint_repair_fraction"]
        <= float(thresholds["maximum_terminal_constraint_repair_fraction"]),
        "n96_terminal_repair_pass": run96["sampling"]["terminal_constraint_repair_fraction"]
        <= float(thresholds["maximum_terminal_constraint_repair_fraction"]),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    accepted = all(checks.values())
    return {
        "accepted_for_production_architecture_implementation": accepted,
        "checks": checks,
        "recommendation": (
            "implement_sparse_hierarchical_discrete_flow_product_prior"
            if accepted
            else "revise_sparse_probe_before_product_prior_implementation"
        ),
        "claim_boundary": (
            "Acceptance selects a representation for later full training. "
            "It is not a trained product prior or evidence of prospective lipid performance."
        ),
    }


def _relative_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def run_sparse_feasibility(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Execute the bounded sparse-topology follow-up to M0-06."""

    if torch is None:
        raise FeasibilityError("sparse topology probe requires torch")
    config = json.loads(config_path.read_text())
    input_paths = {
        name: _resolve_and_verify(repo, record, name) for name, record in config["inputs"].items()
    }
    rows = _read_csv(input_paths["r0"])
    assignments = _read_csv(input_paths["split_assignments"])
    expected_profile = config.get("expected_corpus_profile")
    if not isinstance(expected_profile, Mapping):
        raise FeasibilityError("sparse topology config lacks expected_corpus_profile")
    if len(rows) != int(expected_profile["rows"]):
        raise FeasibilityError(
            f"R0 row count changed: expected {expected_profile['rows']}, " f"observed {len(rows)}"
        )
    declared_elements = set(config["declared_support"]["elements"])
    support_audit = audit_input_support(
        rows,
        declared_elements,
        {
            "n64_fraction": (
                int(expected_profile["records_at_most_64_atoms"]) / int(expected_profile["rows"])
            ),
            "n96_fraction": (
                int(expected_profile["records_at_most_96_atoms"]) / int(expected_profile["rows"])
            ),
            "maximum_heavy_atoms": int(expected_profile["maximum_heavy_atoms"]),
        },
        1e-12,
    )
    atom_vocabulary = build_sparse_atom_vocabulary(rows, declared_elements)
    maximum_closures = int(config["model"]["maximum_closure_slots"])
    full_records, full_exclusions = prepare_sparse_records(
        rows,
        assignments,
        atom_vocabulary,
        declared_elements,
        f"{config['split_scheme']}_fold",
        None,
        maximum_closures,
    )
    full_flat = tuple(record for values in full_records.values() for record in values)
    roundtrip_count = sum(sparse_roundtrip_exact(record) for record in full_flat)
    constitutional_roundtrip_count = sum(
        sparse_constitutional_roundtrip_exact(record, atom_vocabulary) for record in full_flat
    )
    aromatic_records = tuple(
        record for record in full_flat if _record_contains_aromatic_atom(record)
    )
    aromatic_roundtrip_count = sum(
        sparse_constitutional_roundtrip_exact(record, atom_vocabulary)
        for record in aromatic_records
    )
    cycle_census = Counter(record.closure_count for record in full_flat)
    runs = []
    exclusions = {}
    retained_models = []
    for index, run_config in enumerate(config["runs"]):
        n_max = int(run_config["n_max"])
        fold_records = {
            fold: tuple(record for record in values if record.node_count <= n_max)
            for fold, values in full_records.items()
        }
        run_exclusions = dict(full_exclusions)
        run_exclusions["over_n_max"] = sum(record.node_count > n_max for record in full_flat)
        run = run_sparse_probe(
            fold_records,
            atom_vocabulary,
            config,
            run_config,
            int(config["seed"]) + index * 1000,
        )
        retained_models.append(run.pop("_model_object"))
        runs.append(run)
        exclusions[run_config["name"]] = run_exclusions
    scaling = scaling_dry_runs(
        retained_models[-1],
        config["scaling_dry_runs"],
        maximum_closures,
        int(config["seed"]) + 5000,
    )
    decision = make_sparse_decision(
        runs,
        scaling,
        roundtrip_count / len(full_flat),
        constitutional_roundtrip_count / len(full_flat),
        config["decision_thresholds"],
    )
    result = {
        "schema_version": "m0_06_sparse_topology_feasibility_result.v1",
        "status": "completed_bounded_sparse_probe",
        "scope": ("M0 representation feasibility only; no production product-prior training"),
        "method": {
            "family": "hierarchical sparse discrete flow matching",
            "whole_molecule_generation": True,
            "topology_factorization": (
                "connected rooted backbone plus factorized residual closure endpoints"
            ),
            "full_pair_reachability": True,
            "fixed_fragment_or_building_block_inventory": False,
            "node_count": "empirical R0_train prior",
            "closure_count": "empirical R0_train cycle-rank prior",
            "aromaticity": config["declared_support"]["aromaticity_policy"],
            "stereochemistry": config["declared_support"]["stereochemistry_policy"],
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
        "representation_audit": {
            "eligible_single_fragment_rows": len(full_flat),
            "roundtrip_exact_rows": roundtrip_count,
            "roundtrip_fraction": roundtrip_count / len(full_flat),
            "constitutional_roundtrip_exact_rows": constitutional_roundtrip_count,
            "constitutional_roundtrip_fraction": (constitutional_roundtrip_count / len(full_flat)),
            "aromatic_rows": len(aromatic_records),
            "aromatic_constitutional_roundtrip_exact_rows": aromatic_roundtrip_count,
            "aromatic_constitutional_roundtrip_fraction": (
                aromatic_roundtrip_count / len(aromatic_records) if aromatic_records else 1.0
            ),
            "maximum_observed_cycle_rank": max(cycle_census),
            "cycle_rank_counts": {str(key): value for key, value in sorted(cycle_census.items())},
            "full_support_exclusions": full_exclusions,
        },
        "atom_vocabulary": [
            {
                "index": index,
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "maximum_valence_units": _maximum_valence_units(state),
            }
            for index, state in enumerate(atom_vocabulary)
        ],
        "bond_vocabulary": config["declared_support"]["bond_types"],
        "exclusions": exclusions,
        "runs": runs,
        "scaling_dry_runs": scaling,
        "decision": decision,
        "limitations": [
            "The bounded optimization budget selects a representation, not converged generative quality.",
            "Terminal valence masks are deterministic chemistry constraints and every fallback is counted.",
            "The current pointer implementation materializes N by N parent logits but eliminates N by N by hidden-dimension edge states.",
            "Aromatic systems use deterministic Kekule graphs; stereochemistry remains route assigned.",
            "No unavailable pKa, particle-size, PDI, encapsulation, or formulation values are imputed.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result
