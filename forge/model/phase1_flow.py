"""GPU-ready sparse whole-lipid discrete-flow training for Phase 1."""

from __future__ import annotations

import csv
import gzip
import json
import math
import os
import tempfile
import time
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.corpus.r0_splits import sha256_file
from forge.model.defog_feasibility import (
    AtomState,
    _model_state_sha256,
    _parameter_count,
    set_determinism,
)
from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
    SparseGraphRecord,
    _masked_sparse_losses,
    _noise_sparse_batch,
    _sample_flat_interpolation,
    build_sparse_atom_vocabulary,
    collate_sparse_records,
    tensorize_sparse_row,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None

CONFIG_SCHEMA_VERSION = "phase1_product_pretrain_config.v1"
CONFIG_SCHEMA_VERSION_V2 = "phase1_product_pretrain_config.v2"
CONFIG_SCHEMA_VERSION_V3 = "phase1_product_pretrain_config.v3"
RESULT_SCHEMA_VERSION = "phase1_product_pretrain_result.v1"
SIZE_BUCKETS = ("le40", "41_64", "65_96", "97_128", "gt128")
R0_TO_R1_BUCKET = {
    "le40": "le40",
    "41_64": "41_64",
    "65_96": "65_96",
    "97_128": "97_128",
    "gt128": "gt128",
}


class Phase1FlowError(ValueError):
    """Raised when Phase 1 flow training violates its frozen contract."""


@dataclass(frozen=True)
class TrainingBatch:
    records: tuple[TrainingGraphRecord, ...]
    source_counts: dict[str, int]
    size_bucket: str


@dataclass(frozen=True)
class TrainingGraphRecord:
    """Sparse training state without the unused dense round-trip edge matrix."""

    structure_id: str
    canonical_smiles: str
    node_states: np.ndarray
    parents: np.ndarray
    parent_bonds: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    closure_bonds: np.ndarray
    region_states: np.ndarray | None = None

    @property
    def node_count(self) -> int:
        return int(self.node_states.shape[0])

    @property
    def closure_count(self) -> int:
        return int(self.closure_left.shape[0])


def _compact_training_record(record: SparseGraphRecord) -> TrainingGraphRecord:
    return TrainingGraphRecord(
        structure_id=record.structure_id,
        canonical_smiles=record.canonical_smiles,
        node_states=record.node_states,
        parents=record.parents,
        parent_bonds=record.parent_bonds,
        closure_left=record.closure_left,
        closure_right=record.closure_right,
        closure_bonds=record.closure_bonds,
        region_states=record.region_states,
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise Phase1FlowError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase1FlowError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise Phase1FlowError(f"{label} must be a JSON object")
    return value


def _resolve_and_verify(repo: Path, record: Mapping[str, Any], label: str) -> Path:
    if not {"path", "sha256"}.issubset(record):
        raise Phase1FlowError(f"{label} is missing path or sha256")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise Phase1FlowError(f"{label} not found: {path}")
    observed = sha256_file(path)
    if observed != str(record["sha256"]):
        raise Phase1FlowError(
            f"{label} SHA-256 mismatch: expected {record['sha256']}, observed {observed}"
        )
    return path


def _size_bucket(node_count: int) -> str:
    if node_count <= 40:
        return "le40"
    if node_count <= 64:
        return "41_64"
    if node_count <= 96:
        return "65_96"
    if node_count <= 128:
        return "97_128"
    return "gt128"


def _source_mixture_diagnostic(
    expected_fractions: Mapping[str, float],
    source_counts: Mapping[str, int],
    examples_seen: int,
) -> dict[str, Any]:
    """Expose source-mixture drift without turning a finite sample into a false gate."""

    counts = {str(key): int(value) for key, value in source_counts.items()}
    if any(value < 0 for value in counts.values()):
        raise Phase1FlowError("source counts must be nonnegative")
    if examples_seen < 0 or sum(counts.values()) != examples_seen:
        raise Phase1FlowError("source counts do not equal the number of training examples")
    expected = {str(key): float(value) for key, value in expected_fractions.items()}
    if set(expected) != {"r0", "r1"}:
        raise Phase1FlowError("Phase 1 source mixture must contain exactly r0 and r1")
    if not set(counts).issubset(expected):
        raise Phase1FlowError("source counts contain a source outside the Phase 1 mixture")
    if any(not math.isfinite(value) or value < 0 for value in expected.values()):
        raise Phase1FlowError("Phase 1 source fractions must be finite and nonnegative")
    if not math.isclose(sum(expected.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise Phase1FlowError("Phase 1 source fractions must sum to one")
    observed = {
        source: counts.get(source, 0) / max(1, examples_seen) for source in sorted(expected)
    }
    absolute_error = {
        source: abs(observed[source] - expected[source]) for source in sorted(expected)
    }
    return {
        "expected_fractions": dict(sorted(expected.items())),
        "observed_fractions": observed,
        "absolute_error": absolute_error,
        "maximum_absolute_error": max(absolute_error.values()),
        "is_diagnostic_not_acceptance_gate": True,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _portable(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def _row_for_tensorization(
    structure_id: str,
    smiles: str,
) -> dict[str, str]:
    return {
        "r0_structure_id": structure_id,
        "canonical_isomeric_smiles": smiles,
    }


def _cdf(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 1 or len(weights) == 0:
        raise Phase1FlowError("sampling weights must be a nonempty vector")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise Phase1FlowError("sampling weights must be finite and nonnegative")
    total = float(weights.sum())
    if total <= 0:
        raise Phase1FlowError("sampling weights have no positive mass")
    output = np.cumsum(weights / total, dtype=np.float64)
    output[-1] = 1.0
    return output


def _sample_cdf(cdf: np.ndarray, rng: np.random.Generator) -> int:
    return min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)


class BroadTrainingCorpus:
    """Source-balanced, size-bucketed R0/R1 record sampler."""

    def __init__(
        self,
        repo: Path,
        manifest: Mapping[str, Any],
        *,
        seed: int,
        preserve_aromaticity: bool = False,
        root_strategy: str = "canonical",
        region_scheme: str = "none",
        atom_vocabulary_extensions: Sequence[AtomState] = (),
        r0_limit_per_fold: int | None = None,
        r1_limit_per_bucket: int | None = None,
        cache_size: int = 50_000,
    ) -> None:
        inputs = manifest["inputs"]
        r0_path = _resolve_and_verify(repo, inputs["r0_constitutional"], "R0")
        split_path = _resolve_and_verify(repo, inputs["r0_assignments"], "R0 assignments")
        r1_path = _resolve_and_verify(repo, inputs["r1_reaction_enumerated"], "R1")
        r0_rows = _read_csv(r0_path)
        assignments = {row["r0_structure_id"]: row for row in _read_csv(split_path)}
        if len(assignments) != len(r0_rows):
            raise Phase1FlowError("R0 assignments no longer cover every R0 row")

        declared_elements = set(manifest["policy"]["representation"]["elements"])
        self.preserve_aromaticity = preserve_aromaticity
        self.root_strategy = root_strategy
        self.region_scheme = region_scheme
        self.atom_vocabulary = tuple(
            sorted(
                set(
                    build_sparse_atom_vocabulary(
                        r0_rows,
                        declared_elements,
                        preserve_aromaticity=preserve_aromaticity,
                    )
                )
                | set(atom_vocabulary_extensions),
                key=AtomState.key,
            )
        )
        self.atom_to_index = {state: index for index, state in enumerate(self.atom_vocabulary)}
        self.r0_by_fold_bucket: dict[str, dict[str, list[TrainingGraphRecord]]] = {
            fold: defaultdict(list) for fold in ("R0_train", "R0_cal", "R0_heldout")
        }
        for row in r0_rows:
            fold = assignments[row["r0_structure_id"]]["source_study_fold"]
            record = _compact_training_record(
                tensorize_sparse_row(
                    row,
                    self.atom_to_index,
                    preserve_aromaticity=self.preserve_aromaticity,
                    root_strategy=self.root_strategy,
                    region_scheme=self.region_scheme,
                )
            )
            self.r0_by_fold_bucket[fold][_size_bucket(record.node_count)].append(record)
        if r0_limit_per_fold is not None:
            for fold in self.r0_by_fold_bucket:
                for bucket in self.r0_by_fold_bucket[fold]:
                    self.r0_by_fold_bucket[fold][bucket] = self.r0_by_fold_bucket[fold][bucket][
                        :r0_limit_per_fold
                    ]

        self.r1_rows = _read_csv(r1_path)
        self.r1_indices_by_bucket: dict[str, np.ndarray] = {}
        self.r1_cdf_by_bucket: dict[str, np.ndarray] = {}
        for bucket in SIZE_BUCKETS:
            indices = [
                index
                for index, row in enumerate(self.r1_rows)
                if R0_TO_R1_BUCKET[row["size_bin"]] == bucket
            ]
            if r1_limit_per_bucket is not None:
                indices = indices[:r1_limit_per_bucket]
            array = np.asarray(indices, dtype=np.int64)
            weights = np.asarray(
                [float(self.r1_rows[int(index)]["realism_weight"]) for index in array],
                dtype=np.float64,
            )
            positive = weights > 0
            array = array[positive]
            weights = weights[positive]
            if len(array):
                self.r1_indices_by_bucket[bucket] = array
                self.r1_cdf_by_bucket[bucket] = _cdf(weights)
        self.r1_cache: OrderedDict[int, TrainingGraphRecord] = OrderedDict()
        self.cache_size = cache_size
        self.seed = seed

        broad = manifest["policy"]["broad_stream"]
        self.source_fractions = {
            "r0": float(broad["r0_fraction"]),
            "r1": float(broad["r1_fraction"]),
        }
        r0_bucket_counts = Counter()
        for bucket, records in self.r0_by_fold_bucket["R0_train"].items():
            r0_bucket_counts[bucket] = len(records)
        total_r0 = sum(r0_bucket_counts.values())
        r1_bucket_weight = {
            bucket: float(
                sum(
                    float(self.r1_rows[int(index)]["realism_weight"])
                    for index in self.r1_indices_by_bucket.get(
                        bucket, np.asarray([], dtype=np.int64)
                    )
                )
            )
            for bucket in SIZE_BUCKETS
        }
        total_r1 = sum(r1_bucket_weight.values())
        self.joint_bucket_source_mass: dict[str, dict[str, float]] = {}
        combined = []
        for bucket in SIZE_BUCKETS:
            source_mass = {
                "r0": self.source_fractions["r0"] * r0_bucket_counts[bucket] / max(1, total_r0),
                "r1": self.source_fractions["r1"] * r1_bucket_weight[bucket] / max(1e-12, total_r1),
            }
            self.joint_bucket_source_mass[bucket] = source_mass
            combined.append(sum(source_mass.values()))
        self.bucket_cdf = _cdf(np.asarray(combined, dtype=np.float64))
        self.r0_bucket_cdf_by_fold: dict[str, np.ndarray] = {}
        for fold, by_bucket in self.r0_by_fold_bucket.items():
            counts = np.asarray(
                [len(by_bucket.get(bucket, ())) for bucket in SIZE_BUCKETS],
                dtype=np.float64,
            )
            if counts.sum() <= 0:
                raise Phase1FlowError(f"{fold} contains no product records")
            self.r0_bucket_cdf_by_fold[fold] = _cdf(counts)

    def _sample_r1_record(self, bucket: str, rng: np.random.Generator) -> TrainingGraphRecord:
        indices = self.r1_indices_by_bucket[bucket]
        local_index = _sample_cdf(self.r1_cdf_by_bucket[bucket], rng)
        row_index = int(indices[local_index])
        cached = self.r1_cache.get(row_index)
        if cached is not None:
            self.r1_cache.move_to_end(row_index)
            return cached
        row = self.r1_rows[row_index]
        try:
            record = _compact_training_record(
                tensorize_sparse_row(
                    _row_for_tensorization(
                        f"R1-{row_index:06d}",
                        row["canonical_smiles"],
                    ),
                    self.atom_to_index,
                    preserve_aromaticity=self.preserve_aromaticity,
                    root_strategy=self.root_strategy,
                    region_scheme=self.region_scheme,
                )
            )
        except KeyError as exc:
            raise Phase1FlowError(
                f"R1 row {row_index} contains atom state outside frozen R0 vocabulary"
            ) from exc
        self.r1_cache[row_index] = record
        if len(self.r1_cache) > self.cache_size:
            self.r1_cache.popitem(last=False)
        return record

    def _sample_source(self, bucket: str, rng: np.random.Generator) -> str:
        mass = self.joint_bucket_source_mass[bucket]
        total = mass["r0"] + mass["r1"]
        if total <= 0:
            raise Phase1FlowError(f"size bucket {bucket} has no source mass")
        return "r0" if rng.random() < mass["r0"] / total else "r1"

    def sample_batch(
        self,
        rng: np.random.Generator,
        *,
        maximum_graphs: int,
        maximum_atoms: int,
        maximum_pointer_logits: int,
    ) -> TrainingBatch:
        bucket = SIZE_BUCKETS[_sample_cdf(self.bucket_cdf, rng)]
        records: list[TrainingGraphRecord] = []
        source_counts: Counter[str] = Counter()
        total_atoms = 0
        maximum_nodes = 0
        for _ in range(maximum_graphs):
            source = self._sample_source(bucket, rng)
            if source == "r0":
                candidates = self.r0_by_fold_bucket["R0_train"][bucket]
                if not candidates:
                    source = "r1"
                    record = self._sample_r1_record(bucket, rng)
                else:
                    record = candidates[int(rng.integers(0, len(candidates)))]
            else:
                record = self._sample_r1_record(bucket, rng)
            proposed_atoms = total_atoms + record.node_count
            proposed_maximum = max(maximum_nodes, record.node_count)
            proposed_pointer_logits = (len(records) + 1) * proposed_maximum * proposed_maximum
            if records and (
                proposed_atoms > maximum_atoms or proposed_pointer_logits > maximum_pointer_logits
            ):
                break
            if not records and (
                record.node_count > maximum_atoms
                or record.node_count * record.node_count > maximum_pointer_logits
            ):
                raise Phase1FlowError(
                    "batch budgets exclude a molecule inside declared heavy-atom support"
                )
            records.append(record)
            source_counts[source] += 1
            total_atoms = proposed_atoms
            maximum_nodes = proposed_maximum
        if not records:
            raise Phase1FlowError("batch sampler produced an empty batch")
        return TrainingBatch(tuple(records), dict(source_counts), bucket)

    def sample_r0_batch(
        self,
        fold: str,
        rng: np.random.Generator,
        *,
        maximum_graphs: int,
        maximum_atoms: int,
        maximum_pointer_logits: int,
    ) -> TrainingBatch:
        """Sample one source-held R0 batch without auxiliary R1 leakage."""

        if fold not in self.r0_bucket_cdf_by_fold:
            raise Phase1FlowError(f"unknown R0 fold: {fold}")
        bucket = SIZE_BUCKETS[_sample_cdf(self.r0_bucket_cdf_by_fold[fold], rng)]
        candidates = self.r0_by_fold_bucket[fold].get(bucket, ())
        if not candidates:
            raise Phase1FlowError(f"{fold} size bucket {bucket} is empty")
        records: list[TrainingGraphRecord] = []
        total_atoms = 0
        maximum_nodes = 0
        for _ in range(maximum_graphs):
            record = candidates[int(rng.integers(0, len(candidates)))]
            proposed_atoms = total_atoms + record.node_count
            proposed_maximum = max(maximum_nodes, record.node_count)
            proposed_pointer_logits = (len(records) + 1) * proposed_maximum * proposed_maximum
            if records and (
                proposed_atoms > maximum_atoms or proposed_pointer_logits > maximum_pointer_logits
            ):
                break
            if not records and (
                record.node_count > maximum_atoms
                or record.node_count * record.node_count > maximum_pointer_logits
            ):
                raise Phase1FlowError(
                    f"batch budgets exclude a molecule in the declared {fold} support"
                )
            records.append(record)
            total_atoms = proposed_atoms
            maximum_nodes = proposed_maximum
        if not records:
            raise Phase1FlowError(f"{fold} sampler produced an empty batch")
        return TrainingBatch(tuple(records), {"r0": len(records)}, bucket)


if nn is not None:

    def _gather_training_nodes(hidden: Any, indices: Any) -> Any:
        return hidden.gather(1, indices[:, :, None].expand(-1, -1, hidden.shape[-1]))

    def _noisy_topology_features(
        parents: Any,
        parent_bonds: Any,
        closure_left: Any,
        closure_right: Any,
        closure_bonds: Any,
        node_mask: Any,
        child_mask: Any,
        closure_mask: Any,
        bond_valence_units: Any | None = None,
    ) -> Any:
        """Compute bounded molecular context from only the current noisy state."""

        node_count = parents.shape[1]
        dtype = torch.float32
        parent_assignment = functional.one_hot(
            parents,
            num_classes=node_count,
        ).to(dtype)
        parent_assignment = parent_assignment * child_mask[:, :, None]
        child_count = parent_assignment.sum(dim=1)

        left_assignment = functional.one_hot(
            closure_left,
            num_classes=node_count,
        ).to(dtype)
        right_assignment = functional.one_hot(
            closure_right,
            num_classes=node_count,
        ).to(dtype)
        active_closures = closure_mask[:, :, None].to(dtype)
        left_assignment = left_assignment * active_closures
        right_assignment = right_assignment * active_closures
        closure_degree = left_assignment.sum(dim=1) + right_assignment.sum(dim=1)

        valence_lookup = (
            BOND_VALENCE_UNITS.to(parent_bonds.device)
            if bond_valence_units is None
            else bond_valence_units
        )
        bond_units = valence_lookup[parent_bonds].to(dtype)
        bond_units = bond_units * child_mask.to(dtype)
        valence_units = bond_units.clone()
        valence_units += torch.einsum(
            "bcp,bc->bp",
            parent_assignment,
            bond_units,
        )
        closure_units = valence_lookup[closure_bonds].to(dtype)
        closure_units = closure_units * closure_mask.to(dtype)
        valence_units += torch.einsum(
            "bkn,bk->bn",
            left_assignment + right_assignment,
            closure_units,
        )

        degree = child_count + child_mask.to(dtype) + closure_degree
        features = torch.stack(
            (
                child_count.clamp(max=4.0) / 4.0,
                degree.clamp(max=8.0) / 8.0,
                valence_units.clamp(max=16.0) / 16.0,
                closure_degree.clamp(max=4.0) / 4.0,
            ),
            dim=-1,
        )
        return features * node_mask[:, :, None]

    class DeterministicSparseFlowBlock(nn.Module):
        """Tree and residual-closure message passing without CUDA scatter reductions."""

        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.update = nn.Sequential(
                nn.Linear(5 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            hidden: Any,
            parents: Any,
            closure_left: Any,
            closure_right: Any,
            closure_bond_hidden: Any,
            node_mask: Any,
            child_mask: Any,
            closure_mask: Any,
        ) -> Any:
            node_count = hidden.shape[1]
            parent_hidden = _gather_training_nodes(hidden, parents)
            parent_assignment = functional.one_hot(
                parents,
                num_classes=node_count,
            ).to(hidden.dtype)
            parent_assignment = parent_assignment * child_mask[:, :, None]
            child_sum = torch.einsum("bcp,bcd->bpd", parent_assignment, hidden)
            child_count = parent_assignment.sum(dim=1)
            child_mean = child_sum / child_count[:, :, None].clamp(min=1.0)

            left_hidden = _gather_training_nodes(hidden, closure_left)
            right_hidden = _gather_training_nodes(hidden, closure_right)
            left_assignment = functional.one_hot(
                closure_left,
                num_classes=node_count,
            ).to(hidden.dtype)
            right_assignment = functional.one_hot(
                closure_right,
                num_classes=node_count,
            ).to(hidden.dtype)
            active_closures = closure_mask[:, :, None].to(hidden.dtype)
            left_assignment = left_assignment * active_closures
            right_assignment = right_assignment * active_closures
            message_to_left = right_hidden + closure_bond_hidden
            message_to_right = left_hidden + closure_bond_hidden
            closure_sum = torch.einsum(
                "bkn,bkd->bnd",
                left_assignment,
                message_to_left,
            )
            closure_sum += torch.einsum(
                "bkn,bkd->bnd",
                right_assignment,
                message_to_right,
            )
            closure_count = left_assignment.sum(dim=1) + right_assignment.sum(dim=1)
            closure_mean = closure_sum / closure_count[:, :, None].clamp(min=1.0)

            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden = global_hidden / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_hidden = global_hidden[:, None, :].expand_as(hidden)
            update = self.update(
                torch.cat(
                    (
                        hidden,
                        parent_hidden,
                        child_mean,
                        closure_mean,
                        global_hidden,
                    ),
                    dim=-1,
                )
            )
            return self.norm(hidden + update) * node_mask[:, :, None]

    class SparseWholeLipidFlow(nn.Module):
        """Production sparse whole-graph flow over tree and residual-closure state."""

        def __init__(
            self,
            *,
            node_classes: int,
            hidden_dim: int,
            layers: int,
            maximum_closures: int,
            maximum_heavy_atoms: int,
            dropout: float,
            bond_classes: int = 3,
            topology_context: bool = False,
            parent_distance_buckets: int = 0,
            closure_ring_size_buckets: int = 0,
            region_classes: int = 0,
            use_position_embedding: bool = False,
        ) -> None:
            super().__init__()
            if bond_classes not in {3, 4}:
                raise Phase1FlowError("sparse flow supports three or four bond classes")
            if topology_context and parent_distance_buckets < 1:
                raise Phase1FlowError(
                    "topology context requires at least one parent-distance bucket"
                )
            self.hidden_dim = hidden_dim
            self.topology_context = topology_context
            self.parent_distance_buckets = parent_distance_buckets
            self.closure_ring_size_buckets = closure_ring_size_buckets
            self.region_classes = region_classes
            self.bond_classes = bond_classes
            self.use_position_embedding = use_position_embedding
            self.node_embedding = nn.Embedding(node_classes, hidden_dim)
            self.position_embedding = (
                nn.Embedding(maximum_heavy_atoms, hidden_dim) if use_position_embedding else None
            )
            self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
            if region_classes:
                self.region_embedding = nn.Embedding(region_classes, hidden_dim)
                self.region_output = nn.Linear(hidden_dim, region_classes)
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if topology_context:
                self.register_buffer(
                    "bond_valence_units",
                    BOND_VALENCE_UNITS[:bond_classes].to(torch.float32).clone(),
                    persistent=False,
                )
                self.topology_context_embedding = nn.Sequential(
                    nn.Linear(4, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.parent_distance_bias = nn.Parameter(torch.zeros(parent_distance_buckets + 1))
            self.blocks = nn.ModuleList(
                DeterministicSparseFlowBlock(hidden_dim, dropout) for _ in range(layers)
            )
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
                nn.Linear(5 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.closure_left_query = nn.Linear(hidden_dim, hidden_dim)
            self.closure_right_query = nn.Linear(hidden_dim, hidden_dim)
            self.closure_node_key = nn.Linear(hidden_dim, hidden_dim)
            self.closure_bond_output = nn.Linear(hidden_dim, bond_classes)
            self.node_count_logits = nn.Parameter(torch.zeros(maximum_heavy_atoms + 1))
            self.closure_count_logits = nn.Parameter(torch.zeros(maximum_closures + 1))
            if closure_ring_size_buckets:
                self.register_buffer(
                    "closure_ring_size_log_probabilities",
                    torch.zeros(closure_ring_size_buckets + 1),
                )

        def forward(
            self,
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
            regions: Any | None = None,
            node_context: Any | None = None,
        ) -> dict[str, Any]:
            time_hidden = self.time_embedding(t[:, None])
            hidden = self.node_embedding(nodes) + time_hidden[:, None, :]
            if self.position_embedding is not None:
                if nodes.shape[1] > self.position_embedding.num_embeddings:
                    raise Phase1FlowError("node sequence exceeds position-embedding support")
                positions = torch.arange(nodes.shape[1], device=nodes.device)
                hidden += self.position_embedding(positions)[None, :, :]
            if node_context is not None:
                if node_context.shape != (*nodes.shape, self.hidden_dim):
                    raise Phase1FlowError(
                        "external node context must be [batch, nodes, hidden_dim]"
                    )
                hidden += node_context
            if self.region_classes:
                if regions is None:
                    raise Phase1FlowError("region-aware flow requires atom-region state")
                hidden += self.region_embedding(regions)
            hidden += self.bond_embedding(parent_bonds) * child_mask[:, :, None]
            if self.topology_context:
                hidden += self.topology_context_embedding(
                    _noisy_topology_features(
                        parents,
                        parent_bonds,
                        closure_left,
                        closure_right,
                        closure_bonds,
                        node_mask,
                        child_mask,
                        closure_mask,
                        self.bond_valence_units,
                    )
                )
            hidden = hidden * node_mask[:, :, None]
            closure_bond_hidden = self.bond_embedding(closure_bonds)
            closure_bond_hidden = closure_bond_hidden * closure_mask[:, :, None]
            for block in self.blocks:
                hidden = block(
                    hidden,
                    parents,
                    closure_left,
                    closure_right,
                    closure_bond_hidden,
                    node_mask,
                    child_mask,
                    closure_mask,
                )

            parent_query = self.parent_query(hidden)
            parent_key = self.parent_key(hidden)
            parent_logits = torch.einsum("bid,bjd->bij", parent_query, parent_key) / math.sqrt(
                self.hidden_dim
            )
            if self.topology_context:
                indices = torch.arange(hidden.shape[1], device=hidden.device)
                distances = indices[:, None] - indices[None, :]
                distance_buckets = torch.where(
                    distances > 0,
                    distances.clamp(max=self.parent_distance_buckets),
                    torch.zeros_like(distances),
                )
                parent_logits += self.parent_distance_bias[distance_buckets][None, :, :]
            parent_hidden = _gather_training_nodes(hidden, parents)
            parent_bond_logits = self.backbone_bond_output(
                torch.cat((hidden, parent_hidden), dim=-1)
            )

            batch, maximum_closures = closure_left.shape
            slots = self.closure_slots(
                torch.arange(
                    maximum_closures,
                    device=nodes.device,
                )
            )[
                None, :, :
            ].expand(batch, -1, -1)
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden = global_hidden / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_hidden = global_hidden[:, None, :].expand_as(slots)
            left_hidden = _gather_training_nodes(hidden, closure_left)
            right_hidden = _gather_training_nodes(hidden, closure_right)
            closure_hidden = self.closure_update(
                torch.cat(
                    (
                        slots,
                        global_hidden,
                        left_hidden,
                        right_hidden,
                        closure_bond_hidden,
                    ),
                    dim=-1,
                )
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
            output = {
                "nodes": self.node_output(hidden),
                "parents": parent_logits,
                "parent_bonds": parent_bond_logits,
                "closure_left": left_logits,
                "closure_right": right_logits,
                "closure_bonds": self.closure_bond_output(closure_hidden),
                "node_count": self.node_count_logits[None, :].expand(batch, -1),
                "closure_count": self.closure_count_logits[None, :].expand(batch, -1),
            }
            if self.region_classes:
                output["regions"] = self.region_output(hidden)
            return output

else:  # pragma: no cover

    class SparseWholeLipidFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise Phase1FlowError("Phase 1 flow training requires torch")


def _model_architecture_kwargs(model_config: Mapping[str, Any]) -> dict[str, Any]:
    topology_context = bool(model_config.get("topology_context", False))
    return {
        "bond_classes": int(model_config.get("bond_classes", 3)),
        "topology_context": topology_context,
        "parent_distance_buckets": (
            int(model_config.get("parent_distance_buckets", 0)) if topology_context else 0
        ),
        "closure_ring_size_buckets": int(model_config.get("closure_ring_size_buckets", 0)),
        "region_classes": int(model_config.get("region_classes", 0)),
    }


def _parent_distance_log_prior(
    records: Sequence[TrainingGraphRecord],
    *,
    buckets: int,
    probability_floor: float,
    strength: float,
) -> np.ndarray:
    """Estimate a training-only canonical-distance prior without closing graph support."""

    if not records or buckets < 1:
        raise Phase1FlowError("parent-distance prior requires records and positive support")
    if not 0.0 < probability_floor < 1.0 / buckets:
        raise Phase1FlowError("parent-distance probability floor is outside its valid range")
    if strength < 0.0:
        raise Phase1FlowError("parent-distance prior strength must be nonnegative")
    counts = np.zeros(buckets + 1, dtype=np.float64)
    for record in records:
        for child in range(1, record.node_count):
            distance = child - int(record.parents[child])
            if distance < 1:
                raise Phase1FlowError("clean parent pointers must precede their children")
            counts[min(distance, buckets)] += 1.0
    probabilities = counts[1:] / counts[1:].sum()
    probabilities = np.maximum(probabilities, probability_floor)
    probabilities /= probabilities.sum()
    valid_bias = strength * np.log(probabilities)
    valid_bias -= valid_bias.max()
    output = np.zeros(buckets + 1, dtype=np.float32)
    output[1:] = valid_bias.astype(np.float32)
    return output


def _initialize_parent_distance_prior(
    model: Any,
    records: Sequence[TrainingGraphRecord],
    model_config: Mapping[str, Any],
) -> np.ndarray | None:
    if not bool(model_config.get("topology_context", False)):
        return None
    bias = _parent_distance_log_prior(
        records,
        buckets=int(model_config["parent_distance_buckets"]),
        probability_floor=float(model_config["parent_distance_probability_floor"]),
        strength=float(model_config["parent_distance_prior_strength"]),
    )
    with torch.no_grad():
        model.parent_distance_bias.copy_(
            torch.as_tensor(
                bias,
                dtype=model.parent_distance_bias.dtype,
                device=model.parent_distance_bias.device,
            )
        )
    return bias


def _closure_ring_size_log_prior(
    records: Sequence[TrainingGraphRecord],
    *,
    buckets: int,
    probability_floor: float,
) -> np.ndarray:
    """Estimate the training-only closure ring-size distribution."""

    if not records or buckets < 3:
        raise Phase1FlowError("closure ring-size prior requires records and at least three buckets")
    if not 0.0 < probability_floor < 1.0 / (buckets - 1):
        raise Phase1FlowError("closure ring-size probability floor is invalid")
    counts = np.zeros(buckets + 1, dtype=np.float64)
    for record in records:
        for left, right in zip(
            record.closure_left,
            record.closure_right,
            strict=True,
        ):
            left_ancestors: dict[int, int] = {}
            node = int(left)
            distance = 0
            while True:
                left_ancestors[node] = distance
                parent = int(record.parents[node])
                if parent == node:
                    break
                node = parent
                distance += 1
            node = int(right)
            distance = 0
            while node not in left_ancestors:
                parent = int(record.parents[node])
                if parent == node:
                    raise Phase1FlowError("closure endpoints do not share a tree root")
                node = parent
                distance += 1
            ring_size = left_ancestors[node] + distance + 1
            counts[min(ring_size, buckets)] += 1.0
    if counts.sum() == 0:
        raise Phase1FlowError("training records contain no residual closures")
    probabilities = counts[2:] / counts[2:].sum()
    probabilities = np.maximum(probabilities, probability_floor)
    probabilities /= probabilities.sum()
    output = np.full(buckets + 1, -np.inf, dtype=np.float32)
    output[2:] = np.log(probabilities).astype(np.float32)
    return output


def _initialize_closure_ring_size_prior(
    model: Any,
    records: Sequence[TrainingGraphRecord],
    model_config: Mapping[str, Any],
) -> np.ndarray | None:
    buckets = int(model_config.get("closure_ring_size_buckets", 0))
    if buckets == 0:
        return None
    log_probabilities = _closure_ring_size_log_prior(
        records,
        buckets=buckets,
        probability_floor=float(model_config["closure_ring_size_probability_floor"]),
    )
    with torch.no_grad():
        model.closure_ring_size_log_probabilities.copy_(
            torch.as_tensor(
                log_probabilities,
                dtype=model.closure_ring_size_log_probabilities.dtype,
                device=model.closure_ring_size_log_probabilities.device,
            )
        )
    return log_probabilities


def _move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _flow_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> tuple[Any, dict[str, float]]:
    sparse_loss, components = _masked_sparse_losses(predictions, clean)
    if "regions" in predictions:
        region_mask = clean["region_supervision_mask"]
        if not region_mask.any():
            raise Phase1FlowError("region-aware flow batch lacks region supervision")
        region_loss = functional.cross_entropy(
            predictions["regions"][region_mask],
            clean["regions"][region_mask],
        )
    else:
        region_loss = sparse_loss * 0.0
    node_counts = clean["node_mask"].sum(dim=1).long()
    closure_counts = clean["closure_mask"].sum(dim=1).long()
    node_count_loss = functional.cross_entropy(predictions["node_count"], node_counts)
    closure_count_loss = functional.cross_entropy(
        predictions["closure_count"],
        closure_counts,
    )
    total = sparse_loss + region_loss + node_count_loss + closure_count_loss
    return total, {
        **components,
        "region_ce": float(region_loss.detach()),
        "node_count_ce": float(node_count_loss.detach()),
        "closure_count_ce": float(closure_count_loss.detach()),
        "total_with_counts": float(total.detach()),
    }


def _collate_training_records(
    records: Sequence[TrainingGraphRecord],
    n_max: int,
    maximum_closures: int,
) -> dict[str, Any]:
    clean = collate_sparse_records(records, n_max, maximum_closures)
    if any(record.region_states is not None for record in records):
        regions = torch.zeros((len(records), n_max), dtype=torch.long)
        supervision = torch.zeros((len(records), n_max), dtype=torch.bool)
        for index, record in enumerate(records):
            if record.region_states is None:
                continue
            if record.region_states.shape != (record.node_count,):
                raise Phase1FlowError("atom-region state no longer matches node count")
            regions[index, : record.node_count] = torch.from_numpy(record.region_states.copy())
            supervision[index, : record.node_count] = True
        clean["regions"] = regions
        clean["region_supervision_mask"] = supervision
    return clean


def _sample_region_conditioned_interpolation(
    clean: Any,
    marginal_by_region: Any,
    regions: Any,
    t: Any,
    active_mask: Any,
    generator: Any,
) -> Any:
    """Sample an atom bridge whose source marginal is conditioned on region."""

    probabilities = marginal_by_region[regions[active_mask]].clone()
    example_index = torch.arange(
        clean.shape[0],
        device=clean.device,
    )[:, None].expand_as(
        clean
    )[active_mask]
    probabilities *= 1.0 - t[example_index, None]
    probabilities.scatter_add_(
        1,
        clean[active_mask][:, None],
        t[example_index, None],
    )
    sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    output = clean.clone()
    output[active_mask] = sampled
    return output


def _loss_on_records(
    model: Any,
    records: Sequence[TrainingGraphRecord],
    *,
    device: Any,
    maximum_closures: int,
    node_marginal: Any,
    bond_marginal: Any,
    region_marginal: Any | None = None,
    region_atom_marginal: Any | None = None,
    generator: Any,
) -> tuple[Any, dict[str, float]]:
    n_max = max(record.node_count for record in records)
    clean = _move_batch(
        _collate_training_records(records, n_max, maximum_closures),
        device,
    )
    t = torch.rand(len(records), generator=generator, device=device).clamp(0.02, 0.98)
    noisy = _noise_sparse_batch(clean, node_marginal, bond_marginal, t, generator)
    if "regions" in clean:
        if region_marginal is None:
            raise Phase1FlowError("region-aware records require a source marginal")
        noisy["regions"] = _sample_flat_interpolation(
            clean["regions"],
            region_marginal,
            t,
            clean["node_mask"],
            generator,
        )
        if region_atom_marginal is not None:
            noisy["nodes"] = _sample_region_conditioned_interpolation(
                clean["nodes"],
                region_atom_marginal,
                noisy["regions"],
                t,
                clean["node_mask"],
                generator,
            )
    elif region_atom_marginal is not None:
        raise Phase1FlowError("region-conditioned atom source requires region state")
    predictions = model(
        noisy["nodes"],
        noisy["parents"],
        noisy["parent_bonds"],
        noisy["closure_left"],
        noisy["closure_right"],
        noisy["closure_bonds"],
        t,
        clean["node_mask"],
        clean["child_mask"],
        clean["closure_mask"],
        regions=noisy.get("regions"),
    )
    loss, components = _flow_loss(predictions, clean)
    return loss, components


def _train_one_step(
    model: Any,
    records: Sequence[TrainingGraphRecord],
    optimizer: Any,
    *,
    device: Any,
    maximum_closures: int,
    node_marginal: Any,
    bond_marginal: Any,
    region_marginal: Any | None = None,
    region_atom_marginal: Any | None = None,
    generator: Any,
    gradient_clip_norm: float,
) -> dict[str, float]:
    loss, components = _loss_on_records(
        model,
        records,
        device=device,
        maximum_closures=maximum_closures,
        node_marginal=node_marginal,
        bond_marginal=bond_marginal,
        region_marginal=region_marginal,
        region_atom_marginal=region_atom_marginal,
        generator=generator,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        gradient_clip_norm,
    )
    optimizer.step()
    components["gradient_norm"] = float(gradient_norm.detach())
    return components


def _build_validation_batches(
    corpus: BroadTrainingCorpus,
    training_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[tuple[TrainingGraphRecord, ...], ...]:
    rng = np.random.default_rng(seed)
    return tuple(
        corpus.sample_r0_batch(
            "R0_cal",
            rng,
            maximum_graphs=int(training_config["maximum_graphs_per_batch"]),
            maximum_atoms=int(training_config["maximum_atoms_per_batch"]),
            maximum_pointer_logits=int(training_config["maximum_parent_pointer_logits_per_batch"]),
        ).records
        for _ in range(int(training_config["validation_batches"]))
    )


def _evaluate_model(
    model: Any,
    validation_batches: Sequence[Sequence[TrainingGraphRecord]],
    *,
    device: Any,
    maximum_closures: int,
    node_marginal: Any,
    bond_marginal: Any,
    region_marginal: Any | None = None,
    region_atom_marginal: Any | None = None,
    seed: int,
) -> dict[str, float]:
    """Evaluate on fixed R0-cal batches with identical noise at every checkpoint."""

    generator = torch.Generator(device=device).manual_seed(seed)
    components: list[dict[str, float]] = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for records in validation_batches:
            _, row = _loss_on_records(
                model,
                records,
                device=device,
                maximum_closures=maximum_closures,
                node_marginal=node_marginal,
                bond_marginal=bond_marginal,
                region_marginal=region_marginal,
                region_atom_marginal=region_atom_marginal,
                generator=generator,
            )
            components.append(row)
    model.train(was_training)
    return {key: float(np.mean([row[key] for row in components])) for key in components[0]}


def _marginals(
    records: Sequence[TrainingGraphRecord],
    node_classes: int,
    bond_classes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    node_counts = np.zeros(node_classes, dtype=np.float64)
    bond_counts = np.zeros(bond_classes, dtype=np.float64)
    for record in records:
        np.add.at(node_counts, record.node_states, 1)
        np.add.at(bond_counts, record.parent_bonds[1:], 1)
        np.add.at(bond_counts, record.closure_bonds, 1)
    node_counts += 1e-6
    bond_counts += 1e-6
    return node_counts / node_counts.sum(), bond_counts / bond_counts.sum()


def _region_marginal(
    records: Sequence[TrainingGraphRecord],
    region_classes: int,
) -> np.ndarray | None:
    if region_classes == 0:
        return None
    counts = np.zeros(region_classes, dtype=np.float64)
    for record in records:
        if record.region_states is None:
            continue
        if np.any(record.region_states < 0) or np.any(record.region_states >= region_classes):
            raise Phase1FlowError("atom-region state lies outside configured support")
        np.add.at(counts, record.region_states, 1)
    if counts.sum() == 0 or np.any(counts == 0):
        raise Phase1FlowError("every configured atom region requires training support")
    return counts / counts.sum()


def _region_atom_marginal(
    records: Sequence[TrainingGraphRecord],
    node_classes: int,
    region_classes: int,
) -> np.ndarray | None:
    if region_classes == 0:
        return None
    counts = np.full((region_classes, node_classes), 1e-6, dtype=np.float64)
    for record in records:
        if record.region_states is None:
            continue
        np.add.at(counts, (record.region_states, record.node_states), 1.0)
    if np.any(counts.sum(axis=1) <= 0):
        raise Phase1FlowError("every atom region requires conditional source support")
    return counts / counts.sum(axis=1, keepdims=True)


def _record_degrees(record: TrainingGraphRecord) -> np.ndarray:
    """Return constitutional degrees from the tree plus residual closures."""

    degrees = np.zeros(record.node_count, dtype=np.int64)
    for child in range(1, record.node_count):
        parent = int(record.parents[child])
        degrees[child] += 1
        degrees[parent] += 1
    for left, right in zip(
        record.closure_left,
        record.closure_right,
        strict=True,
    ):
        degrees[int(left)] += 1
        degrees[int(right)] += 1
    return degrees


def _degree_continuation_log_prior(
    records: Sequence[TrainingGraphRecord],
    *,
    region_classes: int,
    maximum_degree: int,
    probability_floor: float,
) -> np.ndarray | None:
    """Estimate P(final degree >= d + 1 | final degree >= d) by lipid region.

    The continuation probability is used only while decoding parent pointers. It
    supplies global degree coordination that independent terminal pointer draws
    lack, while a positive floor preserves every degree inside model support.
    """

    if region_classes == 0:
        return None
    if not records or maximum_degree < 1:
        raise Phase1FlowError("degree-continuation prior requires records and degree support")
    if not 0.0 < probability_floor < 1.0:
        raise Phase1FlowError("degree-continuation probability floor is invalid")
    counts = np.zeros((region_classes, maximum_degree + 1), dtype=np.float64)
    for record in records:
        if record.region_states is None:
            continue
        degrees = _record_degrees(record)
        if np.any(degrees > maximum_degree):
            raise Phase1FlowError("training degree exceeds configured degree-continuation support")
        np.add.at(counts, (record.region_states, degrees), 1.0)
    if np.any(counts.sum(axis=1) == 0):
        raise Phase1FlowError("every lipid region requires degree-continuation support")
    output = np.full((region_classes, maximum_degree + 1), -np.inf, dtype=np.float32)
    for region in range(region_classes):
        survival = np.flip(np.cumsum(np.flip(counts[region])))
        for degree in range(maximum_degree):
            if survival[degree] <= 0:
                probability = probability_floor
            else:
                probability = survival[degree + 1] / survival[degree]
            output[region, degree] = np.log(np.clip(probability, probability_floor, 1.0))
    return output


def run_tiny_overfit_gate(
    records: Sequence[TrainingGraphRecord],
    atom_vocabulary: Sequence[AtomState],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    if torch is None:
        raise Phase1FlowError("tiny-overfit gate requires torch")
    set_determinism(seed, max(1, torch.get_num_threads()))
    gate = config["overfit_gate"]
    selected = tuple(records[: int(gate["records"])])
    if len(selected) != int(gate["records"]):
        raise Phase1FlowError("tiny-overfit gate lacks enough records")
    device = torch.device(str(gate["device"]))
    model = SparseWholeLipidFlow(
        node_classes=len(atom_vocabulary),
        hidden_dim=int(gate["hidden_dim"]),
        layers=int(gate["layers"]),
        maximum_closures=int(config["model"]["maximum_closure_slots"]),
        maximum_heavy_atoms=int(config["model"]["maximum_heavy_atoms"]),
        dropout=float(gate["dropout"]),
        **_model_architecture_kwargs(config["model"]),
    ).to(device)
    _initialize_parent_distance_prior(model, selected, config["model"])
    node_marginal, bond_marginal = _marginals(
        selected,
        len(atom_vocabulary),
        int(config["model"].get("bond_classes", 3)),
    )
    region_marginal = _region_marginal(
        selected,
        int(config["model"].get("region_classes", 0)),
    )
    region_atom_marginal = _region_atom_marginal(
        selected,
        len(atom_vocabulary),
        int(config["model"].get("region_classes", 0)),
    )
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.tensor(bond_marginal, dtype=torch.float32, device=device)
    region_p0 = (
        torch.tensor(region_marginal, dtype=torch.float32, device=device)
        if region_marginal is not None
        else None
    )
    region_atom_p0 = (
        torch.tensor(region_atom_marginal, dtype=torch.float32, device=device)
        if region_atom_marginal is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(gate.get("learning_rate", config["training"]["learning_rate"])),
        weight_decay=0.0,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    losses = []
    model.train()
    start = time.perf_counter()
    for _ in range(int(gate["steps"])):
        losses.append(
            _train_one_step(
                model,
                selected,
                optimizer,
                device=device,
                maximum_closures=int(config["model"]["maximum_closure_slots"]),
                node_marginal=node_p0,
                bond_marginal=bond_p0,
                region_marginal=region_p0,
                region_atom_marginal=region_atom_p0,
                generator=generator,
                gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
            )
        )
    elapsed = time.perf_counter() - start
    initial = float(np.mean([row["total_with_counts"] for row in losses[:10]]))
    final = float(np.mean([row["total_with_counts"] for row in losses[-10:]]))
    reduction = (initial - final) / initial
    threshold = float(gate["minimum_total_loss_reduction_fraction"])
    return {
        "status": "pass" if reduction >= threshold else "fail",
        "records": len(selected),
        "steps": int(gate["steps"]),
        "initial_mean_total_loss": initial,
        "final_mean_total_loss": final,
        "loss_reduction_fraction": reduction,
        "required_reduction_fraction": threshold,
        "wall_seconds": elapsed,
        "steps_per_second": int(gate["steps"]) / elapsed,
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    schema_version = config.get("schema_version")
    if schema_version not in {
        CONFIG_SCHEMA_VERSION,
        CONFIG_SCHEMA_VERSION_V2,
        CONFIG_SCHEMA_VERSION_V3,
    }:
        raise Phase1FlowError("unsupported Phase 1 product-pretraining config")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "phase1_manifest",
        "product_prelaunch_audit",
    }:
        raise Phase1FlowError("Phase 1 product-pretraining inputs changed")
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise Phase1FlowError("execution must be an object")
    if execution.get("mixed_precision") is not False or execution.get("compile") is not False:
        raise Phase1FlowError("first Phase 1 run must not enable unvalidated acceleration")
    if execution.get("deterministic_algorithms") is not True:
        raise Phase1FlowError("first Phase 1 run must require deterministic algorithms")
    if int(execution.get("num_workers", -1)) != 0:
        raise Phase1FlowError("initial deterministic sampler requires num_workers=0")
    model = config.get("model")
    if not isinstance(model, dict) or int(model.get("maximum_heavy_atoms", 0)) != 282:
        raise Phase1FlowError("maximum heavy-atom support must remain 282")
    if int(model.get("maximum_closure_slots", 0)) != 12:
        raise Phase1FlowError("maximum closure support must remain 12")
    topology_context = bool(model.get("topology_context", False))
    if schema_version == CONFIG_SCHEMA_VERSION and topology_context:
        raise Phase1FlowError("topology context requires the Phase 1 v2 config schema")
    if schema_version in {CONFIG_SCHEMA_VERSION_V2, CONFIG_SCHEMA_VERSION_V3}:
        if not topology_context:
            version = "v2" if schema_version == CONFIG_SCHEMA_VERSION_V2 else "v3"
            raise Phase1FlowError(f"Phase 1 {version} requires noisy-state topology context")
        distance_buckets = int(model.get("parent_distance_buckets", 0))
        if not 1 <= distance_buckets <= 32:
            raise Phase1FlowError("parent-distance buckets must remain between 1 and 32")
        probability_floor = float(model.get("parent_distance_probability_floor", 0.0))
        if not 0.0 < probability_floor < 1.0 / distance_buckets:
            raise Phase1FlowError("parent-distance probability floor is invalid")
        if not 0.0 <= float(model.get("parent_distance_prior_strength", -1.0)) <= 4.0:
            raise Phase1FlowError("parent-distance prior strength is outside the frozen range")
    if schema_version == CONFIG_SCHEMA_VERSION_V3:
        if int(model.get("bond_classes", 0)) != 4:
            raise Phase1FlowError("Phase 1 v3 requires explicit aromatic bond support")
        if model.get("preserve_aromaticity") is not True:
            raise Phase1FlowError("Phase 1 v3 requires explicit aromatic atom states")
        if model.get("root_strategy") != "lipid_polar":
            raise Phase1FlowError("Phase 1 v3 requires the lipid-polar root strategy")
        ring_buckets = int(model.get("closure_ring_size_buckets", 0))
        if not 9 <= ring_buckets <= int(model["maximum_heavy_atoms"]):
            raise Phase1FlowError("Phase 1 v3 closure ring-size support is invalid")
        ring_floor = float(model.get("closure_ring_size_probability_floor", 0.0))
        if not 0.0 < ring_floor < 1.0 / (ring_buckets - 1):
            raise Phase1FlowError("Phase 1 v3 closure ring-size probability floor is invalid")
        if not 0.0 <= float(model.get("closure_ring_size_prior_strength", -1.0)) <= 4.0:
            raise Phase1FlowError("Phase 1 v3 closure ring-size prior strength is invalid")
        if int(model.get("region_classes", 0)) != 3:
            raise Phase1FlowError("Phase 1 v3 requires three structural lipid regions")
        if model.get("region_scheme") != "polar_structural_v2":
            raise Phase1FlowError("Phase 1 v3 requires the frozen structural region scheme")
        if model.get("aromatic_cycle_sizes") != [5, 6]:
            raise Phase1FlowError("Phase 1 v3 aromatic cycle support must remain five and six")
        aromatic_threshold = float(model.get("aromatic_cycle_probability_threshold", 0.0))
        if not 0.0 < aromatic_threshold < 1.0:
            raise Phase1FlowError("Phase 1 v3 aromatic cycle threshold is invalid")
        if model.get("region_conditioned_atom_source") is not True:
            raise Phase1FlowError("Phase 1 v3 requires region-conditioned atom marginals")
        maximum_degree = int(model.get("degree_prior_maximum_degree", 0))
        if not 4 <= maximum_degree <= 8:
            raise Phase1FlowError(
                "Phase 1 v3 degree-continuation support must remain between four and eight"
            )
        degree_floor = float(model.get("degree_continuation_probability_floor", 0.0))
        if not 0.0 < degree_floor < 1.0:
            raise Phase1FlowError("Phase 1 v3 degree-continuation probability floor is invalid")
        if not 0.0 <= float(model.get("degree_continuation_prior_strength", -1.0)) <= 4.0:
            raise Phase1FlowError("Phase 1 v3 degree-continuation prior strength is invalid")
        if model.get("r1_atom_vocabulary_extensions") != [
            {
                "symbol": "O",
                "formal_charge": 0,
                "aromatic": True,
                "explicit_hydrogens": 0,
            }
        ]:
            raise Phase1FlowError("Phase 1 v3 R1 atom-vocabulary extension changed")
    elif any(
        key in model
        for key in (
            "bond_classes",
            "preserve_aromaticity",
            "root_strategy",
            "region_classes",
            "region_scheme",
            "aromatic_cycle_sizes",
            "aromatic_cycle_probability_threshold",
            "region_conditioned_atom_source",
            "degree_prior_maximum_degree",
            "degree_continuation_probability_floor",
            "degree_continuation_prior_strength",
            "r1_atom_vocabulary_extensions",
        )
    ):
        raise Phase1FlowError("explicit aromatic and lipid-root settings require Phase 1 v3")
    training = config.get("training")
    required_positive_training_fields = (
        "steps",
        "learning_rate",
        "gradient_clip_norm",
        "maximum_graphs_per_batch",
        "maximum_atoms_per_batch",
        "maximum_parent_pointer_logits_per_batch",
        "log_every_steps",
        "validation_every_steps",
        "validation_batches",
        "checkpoint_every_steps",
    )
    if not isinstance(training, dict) or any(
        float(training.get(field, 0)) <= 0 for field in required_positive_training_fields
    ):
        raise Phase1FlowError("training fields must be present and positive")
    if float(training.get("weight_decay", -1)) < 0:
        raise Phase1FlowError("weight decay must be nonnegative")
    overfit = config.get("overfit_gate")
    if (
        not isinstance(overfit, dict)
        or int(overfit.get("steps", 0)) < 10
        or int(overfit.get("records", 0)) < 1
        or not 0.0 < float(overfit.get("minimum_total_loss_reduction_fraction", 0.0)) < 1.0
    ):
        raise Phase1FlowError("tiny-overfit gate contract is invalid")
    outputs = config.get("outputs")
    expected_output_keys = {
        "default_full_directory",
        "checkpoint_filename",
        "latest_checkpoint_filename",
        "best_checkpoint_filename",
        "result_filename",
        "progress_filename",
    }
    if not isinstance(outputs, dict) or set(outputs) != expected_output_keys:
        raise Phase1FlowError("output contract changed")
    for key in expected_output_keys - {"default_full_directory"}:
        value = Path(str(outputs[key]))
        if value.name != str(outputs[key]) or value.is_absolute() or ".." in value.parts:
            raise Phase1FlowError(f"{key} must be one safe filename")
    default_directory = Path(str(outputs["default_full_directory"]))
    if default_directory.is_absolute() or ".." in default_directory.parts:
        raise Phase1FlowError("default full output directory must be repository-relative")


def _verify_prelaunch_audit(
    config: Mapping[str, Any],
    repo: Path,
) -> dict[str, Any]:
    audit_path = _resolve_and_verify(
        repo,
        config["inputs"]["product_prelaunch_audit"],
        "Phase 1 product prelaunch audit",
    )
    audit = _load_json(audit_path, "Phase 1 product prelaunch audit")
    if (
        audit.get("schema_version") != "phase1_product_prelaunch_audit.v1"
        or audit.get("status") != "pass"
    ):
        raise Phase1FlowError("full-corpus product prelaunch audit has not passed")
    if audit.get("decision", {}).get("gpu_training_authorized_by_representation_audit") is not True:
        raise Phase1FlowError("prelaunch audit does not authorize GPU representation support")
    manifest_input = audit.get("inputs", {}).get("phase1_manifest", {})
    if manifest_input.get("sha256") != config["inputs"]["phase1_manifest"]["sha256"]:
        raise Phase1FlowError("prelaunch audit was computed from a different Phase 1 manifest")
    r1_audit = audit.get("r1_audit", {})
    if (
        r1_audit.get("all_rows_supported") is not True
        or r1_audit.get("failure_counts") != {}
        or int(r1_audit.get("maximum_heavy_atoms", 10**9))
        > int(config["model"]["maximum_heavy_atoms"])
        or int(r1_audit.get("maximum_closures", 10**9))
        > int(config["model"]["maximum_closure_slots"])
    ):
        raise Phase1FlowError("prelaunch audit no longer satisfies model support")
    return audit


def build_corpus_from_config(
    config_path: Path,
    repo: Path,
    *,
    smoke: bool,
) -> tuple[dict[str, Any], dict[str, Any], BroadTrainingCorpus]:
    config = _load_json(config_path, "Phase 1 product-pretraining config")
    _validate_config(config)
    _verify_prelaunch_audit(config, repo)
    manifest_path = _resolve_and_verify(
        repo,
        config["inputs"]["phase1_manifest"],
        "Phase 1 data manifest",
    )
    manifest = _load_json(manifest_path, "Phase 1 data manifest")
    smoke_config = config["smoke"] if smoke else {}
    model_config = config["model"]
    atom_vocabulary_extensions = tuple(
        AtomState(
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
            int(row["explicit_hydrogens"]),
        )
        for row in model_config.get("r1_atom_vocabulary_extensions", ())
    )
    corpus = BroadTrainingCorpus(
        repo,
        manifest,
        seed=int(config["seed"]),
        preserve_aromaticity=bool(model_config.get("preserve_aromaticity", False)),
        root_strategy=str(model_config.get("root_strategy", "canonical")),
        region_scheme=str(model_config.get("region_scheme", "none")),
        atom_vocabulary_extensions=atom_vocabulary_extensions,
        r0_limit_per_fold=(int(smoke_config["r0_records_per_fold"]) if smoke else None),
        r1_limit_per_bucket=(int(smoke_config["r1_records_per_bucket"]) if smoke else None),
    )
    return config, manifest, corpus


def _effective_training_config(config: Mapping[str, Any], *, smoke: bool) -> dict[str, Any]:
    model = dict(config["model"])
    training = dict(config["training"])
    execution = dict(config["execution"])
    if smoke:
        overrides = config["smoke"]
        for key in ("hidden_dim", "layers", "dropout"):
            model[key] = overrides[key]
        for key in (
            "steps",
            "maximum_graphs_per_batch",
            "maximum_atoms_per_batch",
            "maximum_parent_pointer_logits_per_batch",
        ):
            training[key] = overrides[key]
        training["validation_every_steps"] = max(1, int(training["steps"]))
        training["checkpoint_every_steps"] = max(1, int(training["steps"]))
        execution["device"] = overrides["device"]
    return {"model": model, "training": training, "execution": execution}


def _validate_effective_training_config(effective: Mapping[str, Any]) -> None:
    model = effective["model"]
    training = effective["training"]
    maximum_heavy_atoms = int(model["maximum_heavy_atoms"])
    if int(training["maximum_graphs_per_batch"]) < 1:
        raise Phase1FlowError("batch graph budget must admit at least one molecule")
    if int(training["maximum_atoms_per_batch"]) < maximum_heavy_atoms:
        raise Phase1FlowError(
            "batch atom budget must admit one molecule at the declared heavy-atom maximum"
        )
    if (
        int(training["maximum_parent_pointer_logits_per_batch"])
        < maximum_heavy_atoms * maximum_heavy_atoms
    ):
        raise Phase1FlowError(
            "parent-pointer budget must admit one molecule at the declared heavy-atom maximum"
        )


def _resolve_device(execution: Mapping[str, Any]) -> Any:
    if torch is None:
        raise Phase1FlowError("Phase 1 flow training requires torch")
    requested = str(execution["device"])
    if requested == "cuda" and not torch.cuda.is_available():
        raise Phase1FlowError("CUDA training was requested but no CUDA device is available")
    device = torch.device(requested)
    if str(execution.get("precision")) != "float32":
        raise Phase1FlowError("the first Phase 1 run requires float32 precision")
    return device


def _training_records(corpus: BroadTrainingCorpus) -> tuple[TrainingGraphRecord, ...]:
    return tuple(
        record
        for bucket in SIZE_BUCKETS
        for record in corpus.r0_by_fold_bucket["R0_train"].get(bucket, ())
    )


def _save_checkpoint_atomic(path: Path, package: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(package), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _optimizer_to_device(optimizer: Any, device: Any) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _checkpoint_package(
    *,
    config_sha256: str,
    data_manifest_sha256: str,
    model: Any,
    optimizer: Any,
    model_config: Mapping[str, Any],
    atom_vocabulary: Sequence[AtomState],
    node_marginal: np.ndarray,
    bond_marginal: np.ndarray,
    step: int,
    generator: Any,
    rng: np.random.Generator,
    losses: Sequence[Mapping[str, float]],
    source_totals: Mapping[str, int],
    bucket_totals: Mapping[str, int],
    examples_seen: int,
    validation_history: Sequence[Mapping[str, Any]],
    best_validation_step: int,
    best_validation_loss: float,
    elapsed_wall_seconds: float,
    region_marginal: np.ndarray | None = None,
    region_atom_marginal: np.ndarray | None = None,
    degree_continuation_log_prior: np.ndarray | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "phase1_product_pretrain_checkpoint.v2",
        "trusted_local_checkpoint": True,
        "config_sha256": config_sha256,
        "data_manifest_sha256": data_manifest_sha256,
        "step": step,
        "model_state_sha256": _model_state_sha256(model),
        "model_config": dict(model_config),
        "atom_vocabulary": [
            {
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "explicit_hydrogens": state.explicit_hydrogens,
            }
            for state in atom_vocabulary
        ],
        "node_marginal": node_marginal.tolist(),
        "bond_marginal": bond_marginal.tolist(),
        "region_marginal": (region_marginal.tolist() if region_marginal is not None else None),
        "region_atom_marginal": (
            region_atom_marginal.tolist() if region_atom_marginal is not None else None
        ),
        "degree_continuation_log_prior": (
            degree_continuation_log_prior.tolist()
            if degree_continuation_log_prior is not None
            else None
        ),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": {
            "noise_generator_state": generator.get_state(),
            "numpy_rng_state": rng.bit_generator.state,
            "torch_cpu_rng_state": torch.get_rng_state(),
            "torch_cuda_rng_state_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "losses": [dict(row) for row in losses],
            "source_totals": dict(source_totals),
            "bucket_totals": dict(bucket_totals),
            "examples_seen": examples_seen,
            "validation_history": [dict(row) for row in validation_history],
            "best_validation_step": best_validation_step,
            "best_validation_loss": best_validation_loss,
            "elapsed_wall_seconds": elapsed_wall_seconds,
        },
    }


def _load_resume_checkpoint(
    path: Path,
    *,
    output_dir: Path,
    config_sha256: str,
    data_manifest_sha256: str,
    model: Any,
    optimizer: Any,
    device: Any,
    generator: Any,
    rng: np.random.Generator,
) -> dict[str, Any]:
    resolved = path.resolve()
    output_root = output_dir.resolve()
    if not resolved.is_relative_to(output_root) or not resolved.is_file():
        raise Phase1FlowError(
            "resume checkpoint must be an existing file inside the output directory"
        )
    try:
        package = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise Phase1FlowError(f"resume checkpoint could not be loaded: {resolved}") from exc
    if (
        not isinstance(package, dict)
        or package.get("schema_version") != "phase1_product_pretrain_checkpoint.v2"
        or package.get("trusted_local_checkpoint") is not True
        or package.get("config_sha256") != config_sha256
        or package.get("data_manifest_sha256") != data_manifest_sha256
    ):
        raise Phase1FlowError("resume checkpoint contract or pinned inputs changed")
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.to(device)
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise Phase1FlowError("resume checkpoint model-state hash mismatch")
    optimizer.load_state_dict(package["optimizer_state_dict"])
    _optimizer_to_device(optimizer, device)
    state = package.get("training_state")
    if not isinstance(state, dict):
        raise Phase1FlowError("resume checkpoint lacks training state")
    generator.set_state(state["noise_generator_state"])
    rng.bit_generator.state = state["numpy_rng_state"]
    torch.set_rng_state(state["torch_cpu_rng_state"])
    if device.type == "cuda":
        cuda_states = state.get("torch_cuda_rng_state_all")
        if not isinstance(cuda_states, list) or not cuda_states:
            raise Phase1FlowError("CUDA resume checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_states)
    return package


def train_product_pretrain(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    smoke: bool,
    overfit_only: bool = False,
    resume_checkpoint: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the mandatory overfit gate and a bounded broad product-pretraining job."""

    if overfit_only and resume_checkpoint is not None:
        raise Phase1FlowError("overfit-only mode cannot resume a product-training checkpoint")
    config, manifest, corpus = build_corpus_from_config(config_path, repo, smoke=smoke)
    effective = _effective_training_config(config, smoke=smoke)
    _validate_effective_training_config(effective)
    if resume_checkpoint is not None and overwrite:
        raise Phase1FlowError("resume and overwrite are mutually exclusive")
    output_names = config["outputs"]
    output_dir = output_dir.resolve()
    result_path = output_dir / output_names["result_filename"]
    checkpoint_path = output_dir / output_names["checkpoint_filename"]
    latest_checkpoint_path = output_dir / output_names["latest_checkpoint_filename"]
    best_checkpoint_path = output_dir / output_names["best_checkpoint_filename"]
    progress_path = output_dir / output_names["progress_filename"]
    managed_paths = (
        result_path,
        checkpoint_path,
        latest_checkpoint_path,
        best_checkpoint_path,
        progress_path,
    )
    if resume_checkpoint is None and not overwrite:
        existing = [path for path in managed_paths if path.exists()]
        if existing:
            raise Phase1FlowError(
                "output artifacts already exist; use --resume-checkpoint or explicit --overwrite: "
                + ", ".join(str(path) for path in existing)
            )
    cpu_threads = max(1, min(8, os.cpu_count() or 1))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    set_determinism(int(config["seed"]), cpu_threads)
    train_records = _training_records(corpus)
    small_records = tuple(record for record in train_records if record.node_count <= 64)
    overfit = run_tiny_overfit_gate(
        small_records,
        corpus.atom_vocabulary,
        config,
        seed=int(config["seed"]) + 1,
    )
    if overfit["status"] != "pass":
        raise Phase1FlowError(
            "tiny-overfit gate failed: "
            f"loss reduction {overfit['loss_reduction_fraction']:.3f} is below "
            f"{overfit['required_reduction_fraction']:.3f}"
        )
    # Diagnostic training must not alter the initialization of the actual run.
    set_determinism(int(config["seed"]), cpu_threads)

    result_base = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "overfit_gate_passed" if overfit_only else "complete",
        "task": config["task"],
        "mode": "overfit_only" if overfit_only else ("smoke" if smoke else "full"),
        "config": {
            "path": _portable(config_path, repo),
            "sha256": sha256_file(config_path),
        },
        "data_manifest": {
            "path": config["inputs"]["phase1_manifest"]["path"],
            "sha256": config["inputs"]["phase1_manifest"]["sha256"],
        },
        "overfit_gate": overfit,
        "guidance": {
            "biological": False,
            "synthesis_value": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if overfit_only:
        payload = (
            json.dumps(result_base, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
        ).encode()
        _atomic_write(result_path, payload)
        return result_base

    device = _resolve_device(effective["execution"])
    model_config = effective["model"]
    training_config = effective["training"]
    model = SparseWholeLipidFlow(
        node_classes=len(corpus.atom_vocabulary),
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        maximum_closures=int(model_config["maximum_closure_slots"]),
        maximum_heavy_atoms=int(model_config["maximum_heavy_atoms"]),
        dropout=float(model_config["dropout"]),
        **_model_architecture_kwargs(model_config),
    ).to(device)
    parent_distance_prior = _initialize_parent_distance_prior(
        model,
        train_records,
        model_config,
    )
    closure_ring_size_prior = _initialize_closure_ring_size_prior(
        model,
        train_records,
        model_config,
    )
    node_marginal, bond_marginal = _marginals(
        train_records,
        len(corpus.atom_vocabulary),
        int(model_config.get("bond_classes", 3)),
    )
    region_marginal = _region_marginal(
        train_records,
        int(model_config.get("region_classes", 0)),
    )
    region_atom_marginal = _region_atom_marginal(
        train_records,
        len(corpus.atom_vocabulary),
        int(model_config.get("region_classes", 0)),
    )
    degree_continuation_log_prior = _degree_continuation_log_prior(
        train_records,
        region_classes=int(model_config.get("region_classes", 0)),
        maximum_degree=int(model_config.get("degree_prior_maximum_degree", 0)),
        probability_floor=float(model_config.get("degree_continuation_probability_floor", 0.01)),
    )
    node_p0 = torch.tensor(node_marginal, dtype=torch.float32, device=device)
    bond_p0 = torch.tensor(bond_marginal, dtype=torch.float32, device=device)
    region_p0 = (
        torch.tensor(region_marginal, dtype=torch.float32, device=device)
        if region_marginal is not None
        else None
    )
    region_atom_p0 = (
        torch.tensor(region_atom_marginal, dtype=torch.float32, device=device)
        if region_atom_marginal is not None
        else None
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 2)
    rng = np.random.default_rng(int(config["seed"]) + 3)
    validation_batches = _build_validation_batches(
        corpus,
        training_config,
        seed=int(config["seed"]) + 4,
    )
    config_sha256 = sha256_file(config_path)
    data_manifest_sha256 = config["inputs"]["phase1_manifest"]["sha256"]
    start_step = 0
    losses: list[dict[str, float]]
    source_totals: Counter[str]
    bucket_totals: Counter[str]
    examples_seen: int
    validation_history: list[dict[str, Any]]
    best_validation_step: int
    best_validation_loss: float
    elapsed_before_run = 0.0
    resumed_from: dict[str, Any] | None = None

    if resume_checkpoint is not None:
        package = _load_resume_checkpoint(
            resume_checkpoint,
            output_dir=output_dir,
            config_sha256=config_sha256,
            data_manifest_sha256=data_manifest_sha256,
            model=model,
            optimizer=optimizer,
            device=device,
            generator=generator,
            rng=rng,
        )
        state = package["training_state"]
        start_step = int(package["step"])
        if not 0 <= start_step <= int(training_config["steps"]):
            raise Phase1FlowError("resume checkpoint step lies outside the configured run")
        losses = [dict(row) for row in state["losses"]]
        if len(losses) != start_step:
            raise Phase1FlowError("resume checkpoint loss history does not match its step")
        source_totals = Counter(
            {str(key): int(value) for key, value in state["source_totals"].items()}
        )
        bucket_totals = Counter(
            {str(key): int(value) for key, value in state["bucket_totals"].items()}
        )
        examples_seen = int(state["examples_seen"])
        validation_history = [dict(row) for row in state["validation_history"]]
        best_validation_step = int(state["best_validation_step"])
        best_validation_loss = float(state["best_validation_loss"])
        elapsed_before_run = float(state["elapsed_wall_seconds"])
        resumed_from = {
            "path": _portable(resume_checkpoint.resolve(), repo),
            "sha256": sha256_file(resume_checkpoint.resolve()),
            "step": start_step,
        }
    else:
        validation_history = [
            {
                "step": 0,
                "metrics": _evaluate_model(
                    model,
                    validation_batches,
                    device=device,
                    maximum_closures=int(model_config["maximum_closure_slots"]),
                    node_marginal=node_p0,
                    bond_marginal=bond_p0,
                    region_marginal=region_p0,
                    region_atom_marginal=region_atom_p0,
                    seed=int(config["seed"]) + 5,
                ),
            }
        ]
        best_validation_step = 0
        best_validation_loss = validation_history[0]["metrics"]["total_with_counts"]
        losses = []
        source_totals = Counter()
        bucket_totals = Counter()
        examples_seen = 0

    run_start = time.perf_counter()

    def elapsed_wall_seconds() -> float:
        return elapsed_before_run + (time.perf_counter() - run_start)

    def save_training_checkpoint(path: Path, step: int) -> dict[str, Any]:
        package = _checkpoint_package(
            config_sha256=config_sha256,
            data_manifest_sha256=data_manifest_sha256,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            atom_vocabulary=corpus.atom_vocabulary,
            node_marginal=node_marginal,
            bond_marginal=bond_marginal,
            step=step,
            generator=generator,
            rng=rng,
            losses=losses,
            source_totals=source_totals,
            bucket_totals=bucket_totals,
            examples_seen=examples_seen,
            validation_history=validation_history,
            best_validation_step=best_validation_step,
            best_validation_loss=best_validation_loss,
            elapsed_wall_seconds=elapsed_wall_seconds(),
            region_marginal=region_marginal,
            region_atom_marginal=region_atom_marginal,
            degree_continuation_log_prior=degree_continuation_log_prior,
        )
        _save_checkpoint_atomic(path, package)
        return package

    if resume_checkpoint is None:
        save_training_checkpoint(best_checkpoint_path, 0)

    model.train()
    for step in range(start_step + 1, int(training_config["steps"]) + 1):
        batch = corpus.sample_batch(
            rng,
            maximum_graphs=int(training_config["maximum_graphs_per_batch"]),
            maximum_atoms=int(training_config["maximum_atoms_per_batch"]),
            maximum_pointer_logits=int(training_config["maximum_parent_pointer_logits_per_batch"]),
        )
        components = _train_one_step(
            model,
            batch.records,
            optimizer,
            device=device,
            maximum_closures=int(model_config["maximum_closure_slots"]),
            node_marginal=node_p0,
            bond_marginal=bond_p0,
            region_marginal=region_p0,
            region_atom_marginal=region_atom_p0,
            generator=generator,
            gradient_clip_norm=float(training_config["gradient_clip_norm"]),
        )
        losses.append(components)
        source_totals.update(batch.source_counts)
        bucket_totals[batch.size_bucket] += len(batch.records)
        examples_seen += len(batch.records)
        if step % int(training_config["log_every_steps"]) == 0:
            window = losses[-min(int(training_config["log_every_steps"]), len(losses)) :]
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "step": step,
                        "mean_total_with_counts": float(
                            np.mean([row["total_with_counts"] for row in window])
                        ),
                        "examples_seen": examples_seen,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % int(training_config["validation_every_steps"]) == 0 or step == int(
            training_config["steps"]
        ):
            metrics = _evaluate_model(
                model,
                validation_batches,
                device=device,
                maximum_closures=int(model_config["maximum_closure_slots"]),
                node_marginal=node_p0,
                bond_marginal=bond_p0,
                region_marginal=region_p0,
                region_atom_marginal=region_atom_p0,
                seed=int(config["seed"]) + 5,
            )
            validation_history.append({"step": step, "metrics": metrics})
            if metrics["total_with_counts"] < best_validation_loss:
                best_validation_loss = metrics["total_with_counts"]
                best_validation_step = step
                save_training_checkpoint(best_checkpoint_path, step)
            print(
                json.dumps(
                    {
                        "event": "validation",
                        "step": step,
                        "r0_cal_total_with_counts": metrics["total_with_counts"],
                        "best_step": best_validation_step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if step % int(training_config["checkpoint_every_steps"]) == 0 or step == int(
            training_config["steps"]
        ):
            latest_package = save_training_checkpoint(latest_checkpoint_path, step)
            source_mixture = _source_mixture_diagnostic(
                corpus.source_fractions,
                source_totals,
                examples_seen,
            )
            progress = {
                "schema_version": "phase1_product_pretrain_progress.v1",
                "status": "training" if step < int(training_config["steps"]) else "trained",
                "step": step,
                "configured_steps": int(training_config["steps"]),
                "examples_seen": examples_seen,
                "best_validation_step": best_validation_step,
                "best_validation_loss": best_validation_loss,
                "source_mixture": source_mixture,
                "latest_checkpoint": {
                    "path": _portable(latest_checkpoint_path, repo),
                    "sha256": sha256_file(latest_checkpoint_path),
                    "model_state_sha256": latest_package["model_state_sha256"],
                },
            }
            _atomic_write(
                progress_path,
                (
                    json.dumps(progress, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
                ).encode(),
            )
    wall_seconds = elapsed_wall_seconds()
    state_hash = _model_state_sha256(model)
    save_training_checkpoint(
        latest_checkpoint_path,
        int(training_config["steps"]),
    )
    final_package = save_training_checkpoint(
        checkpoint_path,
        int(training_config["steps"]),
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    latest_checkpoint_sha256 = sha256_file(latest_checkpoint_path)
    best_checkpoint_sha256 = sha256_file(best_checkpoint_path)
    final_progress = {
        "schema_version": "phase1_product_pretrain_progress.v1",
        "status": "complete",
        "step": int(training_config["steps"]),
        "configured_steps": int(training_config["steps"]),
        "examples_seen": examples_seen,
        "best_validation_step": best_validation_step,
        "best_validation_loss": best_validation_loss,
        "source_mixture": _source_mixture_diagnostic(
            corpus.source_fractions,
            source_totals,
            examples_seen,
        ),
        "latest_checkpoint": {
            "path": _portable(latest_checkpoint_path, repo),
            "sha256": latest_checkpoint_sha256,
            "model_state_sha256": final_package["model_state_sha256"],
        },
    }
    _atomic_write(
        progress_path,
        (
            json.dumps(final_progress, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
        ).encode(),
    )
    first_window = losses[: min(10, len(losses))]
    final_window = losses[-min(10, len(losses)) :]
    result = {
        **result_base,
        "status": "complete",
        "effective": effective,
        "environment": {
            "torch": torch.__version__,
            "device": str(device),
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "model": {
            "parameters": _parameter_count(model),
            "state_sha256": state_hash,
            "atom_classes": len(corpus.atom_vocabulary),
            "topology_context": bool(model_config.get("topology_context", False)),
            "initial_parent_distance_log_prior": (
                parent_distance_prior.tolist() if parent_distance_prior is not None else None
            ),
            "learned_parent_distance_bias": (
                model.parent_distance_bias.detach().cpu().tolist()
                if bool(model_config.get("topology_context", False))
                else None
            ),
            "initial_closure_ring_size_log_prior": (
                closure_ring_size_prior.tolist() if closure_ring_size_prior is not None else None
            ),
            "degree_continuation_log_prior": (
                degree_continuation_log_prior.tolist()
                if degree_continuation_log_prior is not None
                else None
            ),
        },
        "training": {
            "steps": int(training_config["steps"]),
            "start_step": start_step,
            "examples_seen": examples_seen,
            "source_counts": dict(sorted(source_totals.items())),
            "source_fractions": {
                source: count / max(1, examples_seen)
                for source, count in sorted(source_totals.items())
            },
            "source_mixture": _source_mixture_diagnostic(
                corpus.source_fractions,
                source_totals,
                examples_seen,
            ),
            "size_bucket_counts": dict(sorted(bucket_totals.items())),
            "wall_seconds": wall_seconds,
            "steps_per_second": int(training_config["steps"]) / wall_seconds,
            "graphs_per_second": examples_seen / wall_seconds,
            "initial_mean_loss": {
                key: float(np.mean([row[key] for row in first_window])) for key in first_window[0]
            },
            "final_mean_loss": {
                key: float(np.mean([row[key] for row in final_window])) for key in final_window[0]
            },
        },
        "validation": {
            "fold": "R0_cal",
            "heldout_fold_used": False,
            "fixed_batches": len(validation_batches),
            "fixed_noise_seed": int(config["seed"]) + 5,
            "history": validation_history,
            "best_step": best_validation_step,
            "best_total_with_counts": best_validation_loss,
        },
        "checkpoint": {
            "path": _portable(checkpoint_path, repo),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": checkpoint_sha256,
            "schema_version": final_package["schema_version"],
            "step": final_package["step"],
            "resumable": True,
        },
        "latest_checkpoint": {
            "path": _portable(latest_checkpoint_path, repo),
            "bytes": latest_checkpoint_path.stat().st_size,
            "sha256": latest_checkpoint_sha256,
        },
        "best_checkpoint": {
            "path": _portable(best_checkpoint_path, repo),
            "bytes": best_checkpoint_path.stat().st_size,
            "sha256": best_checkpoint_sha256,
            "step": best_validation_step,
        },
        "resumed_from": resumed_from,
    }
    payload = (json.dumps(result, indent=2, sort_keys=True, separators=(",", ": ")) + "\n").encode()
    _atomic_write(result_path, payload)
    return result
