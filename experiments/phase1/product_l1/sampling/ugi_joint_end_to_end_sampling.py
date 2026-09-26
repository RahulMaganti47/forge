"""Compose the matched joint sparse-flow arm into complete Ugi molecules."""

from __future__ import annotations

import csv
import gzip
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw

from experiments.phase1.product_l1.sampling.ugi_end_to_end_sampling import (
    _atomic_json,
    _closure_model,
    _load_checkpoint,
    _load_expanded_chemistry,
    _reference_comparison,
)
from experiments.phase1.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals as sample_ugi_joint_sparse_terminals,
)
from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache
from forge.corpus.ugi_generated_components import (
    UgiGeneratedComponentError,
    generated_ugi_component_smiles,
)
from forge.corpus.ugi_held_component_gate import (
    exact_forward_reconstructs_ugi_product,
    load_ugi_reaction_contract,
)
from forge.model.defog_feasibility import sha256_file
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.ugi_adapter_features import ORIGIN_TO_INDEX
from forge.model.ugi_chemistry_flow import (
    UgiChemistryFlowError,
    UgiChemistrySample,
    UgiTerminalDecodeError,
    chemistry_sample_statistics,
    chemistry_sample_to_molecule,
    valence_constrained_terminal_sample,
)
from forge.model.ugi_chemistry_interface import (
    assemble_ugi_chemistry_topology_condition,
    recompute_adapter_distances,
)
from forge.model.ugi_closure_placement import sample_sparse_closures
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseFlow,
    UgiJointSparseTerminal,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class UgiJointEndToEndSamplingError(RuntimeError):
    """Raised when the matched joint arm cannot be composed exactly."""


REFERENCE_COMPARISON_MODES = ("full", "deferred")
MOLECULE_FAILURE_SCHEMA = "forge.ugi_molecule_construction_failure.v1"


def _runtime_failure_detail(error: BaseException, *, stage: str) -> dict[str, Any]:
    """Return a stable subtype plus the original exception evidence."""

    message = str(error)
    lowered = message.lower()
    if "kekul" in lowered or "aromatic" in lowered:
        code = "aromaticity_or_kekulization_failure"
    elif "valence" in lowered:
        code = "atom_valence_failure"
    elif "sanitize" in lowered:
        code = "rdkit_sanitization_failure"
    elif isinstance(error, UgiChemistryFlowError):
        code = "chemistry_topology_contract_failure"
    else:
        code = "unclassified_runtime_failure"
    return {
        "schema_version": MOLECULE_FAILURE_SCHEMA,
        "stage": stage,
        "code": code,
        "exception_type": type(error).__name__,
        "message": message,
    }


def _terminal_failure_detail(error: BaseException) -> dict[str, Any]:
    if isinstance(error, UgiTerminalDecodeError):
        return error.to_mapping()
    return _runtime_failure_detail(error, stage="terminal_decode")


def _validate_reference_comparison_mode(mode: str) -> None:
    if mode not in REFERENCE_COMPARISON_MODES:
        raise UgiJointEndToEndSamplingError(
            "reference comparison mode must be one of "
            f"{REFERENCE_COMPARISON_MODES}, received {mode!r}"
        )


def _resolve_prepared_cache(
    repo: Path,
    joint_checkpoint: dict[str, Any],
    prepared_cache_path: Path | None,
) -> tuple[Path, str] | None:
    """Resolve and authenticate a checkpoint's pinned cache, returning its verified digest."""

    cache_record = joint_checkpoint.get("inputs", {}).get("prepared_cache")
    if cache_record is None:
        if prepared_cache_path is not None:
            raise UgiJointEndToEndSamplingError(
                "a prepared-cache override was supplied for a checkpoint with no cache pin"
            )
        return None
    if not isinstance(cache_record, dict) or not isinstance(cache_record.get("sha256"), str):
        raise UgiJointEndToEndSamplingError("joint checkpoint has an invalid prepared-cache pin")

    if prepared_cache_path is not None:
        cache_path = prepared_cache_path
    else:
        cache_path = Path(str(cache_record.get("path", "")))
        if not cache_path.is_file() and str(cache_path).startswith("/root/forge_repo/"):
            cache_path = repo / cache_path.relative_to("/root/forge_repo")
    if not cache_path.is_file():
        raise UgiJointEndToEndSamplingError(f"prepared training cache is missing: {cache_path}")
    observed = sha256_file(cache_path)
    expected = cache_record["sha256"]
    if observed != expected:
        raise UgiJointEndToEndSamplingError(
            "prepared training cache differs from the checkpoint pin: "
            f"expected {expected}, observed {observed}"
        )
    return cache_path, observed


def _resolve_prepared_cache_path(
    repo: Path,
    joint_checkpoint: dict[str, Any],
    prepared_cache_path: Path | None,
) -> Path | None:
    """Resolve a checkpoint's pinned cache without trusting a machine-local training path."""

    resolved = _resolve_prepared_cache(repo, joint_checkpoint, prepared_cache_path)
    return None if resolved is None else resolved[0]


_PREPARED_CORPUS_CACHE: dict[tuple[str, str], Any] = {}


def _load_prepared_corpus(cache_path: Path, cache_sha256: str) -> Any:
    """Return the pinned corpus, unpickling one prepared cache at most once per process.

    A sampling run is sharded, and each shard previously unpickled the whole multi-hundred-megabyte
    training cache again to read two small immutable attributes from it: the atom vocabulary and
    the Ugi core schema.  The key is the caller's already-verified SHA-256 of that file, so a hit
    means identical authenticated bytes and cannot produce a different corpus.  Only the corpus is
    retained; the per-fold training records are not, so the training folds are not held resident
    between shards.
    """

    key = (str(cache_path.resolve()), cache_sha256)
    corpus = _PREPARED_CORPUS_CACHE.get(key)
    if corpus is None:
        corpus, _ = load_ugi_training_cache(cache_path)
        _PREPARED_CORPUS_CACHE[key] = corpus
    return corpus


def _complete_reference_comparison(
    result: dict[str, Any],
    *,
    molecules: list[Chem.Mol],
    reference_corpus: Any | None,
    mode: str,
) -> dict[str, Any]:
    """Finish or explicitly defer the expensive frozen-corpus comparison.

    The caller persists the complete sampling ledger before entering this
    function. A reference-index failure therefore cannot erase generated
    molecules or leave a plausible-looking ``status=complete`` receipt.
    """

    _validate_reference_comparison_mode(mode)
    if mode == "deferred":
        result["status"] = "complete_sampling_reference_deferred"
        result["reference_comparison"] = None
        result["reference_comparison_status"] = "deferred"
        return result
    if reference_corpus is None:
        raise UgiJointEndToEndSamplingError(
            "full reference comparison requires a frozen reference corpus"
        )
    result["reference_comparison"] = (
        _reference_comparison(molecules, reference_corpus) if molecules else None
    )
    result["reference_comparison_status"] = "complete"
    result["status"] = "complete"
    return result


def joint_sampling_result_matches_request(
    result: dict[str, Any],
    *,
    seed: int,
    program_offset: int,
    program_limit: int,
    terminal_decoder_mode: str,
    terminal_decoder_seed: int | None,
    terminal_temperature: float,
    checkpoint_filename: str,
    device: str | None = None,
    local_chemistry_policy_sha256: str | None = None,
    local_chemistry_constraint_scope: str | None = None,
) -> bool:
    """Return whether a persisted receipt can safely satisfy one exact request."""

    sampling = result.get("sampling")
    decoder = sampling.get("terminal_decoder") if isinstance(sampling, dict) else None
    checkpoint = (result.get("checkpoints") or {}).get("joint")
    return bool(
        result.get("status") in {"complete", "complete_sampling_reference_deferred"}
        and result.get("seed") == seed
        and isinstance(result.get("samples"), list)
        and len(result["samples"]) == program_limit
        and isinstance(sampling, dict)
        and sampling.get("program_offset") == program_offset
        and sampling.get("program_limit") == program_limit
        and sampling.get("matched_global_programs") == program_limit
        and isinstance(decoder, dict)
        and decoder.get("mode") == terminal_decoder_mode
        and decoder.get("seed") == terminal_decoder_seed
        and decoder.get("temperature") == terminal_temperature
        and decoder.get("local_chemistry_policy_sha256")
        == local_chemistry_policy_sha256
        and decoder.get("local_chemistry_constraint_scope")
        == local_chemistry_constraint_scope
        and (device is None or sampling.get("device") == str(torch.device(device)))
        and Path(str(checkpoint)).name == checkpoint_filename
    )


def _slice_matched_programs(
    programs: Sequence[UgiMorphologyProgram],
    metadata: Sequence[dict[str, Any]],
    *,
    offset: int,
    limit: int | None,
) -> tuple[tuple[UgiMorphologyProgram, ...], tuple[dict[str, Any], ...]]:
    """Select a declared contiguous program block without seed-search retries."""

    if offset < 0:
        raise UgiJointEndToEndSamplingError("program offset must be nonnegative")
    if limit is not None and limit < 1:
        raise UgiJointEndToEndSamplingError("program limit must be positive")
    stop = None if limit is None else offset + limit
    return tuple(programs[offset:stop]), tuple(metadata[offset:stop])


def flow_endpoint_chemistry_sample(
    condition: Any,
    terminal: UgiJointSparseTerminal,
    closure_bond_states: np.ndarray,
) -> UgiChemistrySample:
    """Map the flowed time-one categorical state onto the complete Ugi topology.

    Sparse closure placement and its bond state are resolved outside the joint
    sequence flow.  Supplying the same closure completion used by a matched
    terminal decoder isolates whether re-decoding the channels that *were*
    flowed (atoms, parent bonds and decorations) changes their joint chemistry.
    No atom, bond or decoration state is repaired in this mapping.
    """

    endpoints = (
        terminal.flow_endpoint_atom_states,
        terminal.flow_endpoint_parent_bond_states,
        terminal.flow_endpoint_decoration_anchors,
        terminal.flow_endpoint_decoration_atom_states,
        terminal.flow_endpoint_decoration_bond_states,
    )
    if any(value is None for value in endpoints):
        raise UgiJointEndToEndSamplingError("joint terminal lacks a flowed time-one state")
    (
        endpoint_atoms,
        endpoint_parent_bonds,
        endpoint_anchors,
        endpoint_decoration_atoms,
        (endpoint_decoration_bonds),
    ) = endpoints
    assert endpoint_atoms is not None
    assert endpoint_parent_bonds is not None
    assert endpoint_anchors is not None
    assert endpoint_decoration_atoms is not None
    assert endpoint_decoration_bonds is not None

    exterior_full_indices: list[int] = []
    for role in ROLE_NAMES:
        exterior_full_indices.extend(
            np.flatnonzero(
                (condition.origin_states == ORIGIN_TO_INDEX[role]) & ~condition.fixed_atom_mask
            ).tolist()
        )
    if len(exterior_full_indices) != terminal.program.node_count:
        raise UgiJointEndToEndSamplingError("flow endpoint/full topology mismatch")
    if len(endpoint_atoms) != len(exterior_full_indices) or len(endpoint_parent_bonds) != len(
        exterior_full_indices
    ):
        raise UgiJointEndToEndSamplingError("flow endpoint channel lengths are misaligned")
    if not (
        len(endpoint_anchors) == len(endpoint_decoration_atoms) == len(endpoint_decoration_bonds)
    ):
        raise UgiJointEndToEndSamplingError("flow endpoint decoration channels are misaligned")
    if len(closure_bond_states) != condition.closure_count:
        raise UgiJointEndToEndSamplingError("closure completion is misaligned")

    atom_states = np.asarray(condition.fixed_atom_states, dtype=np.int64).copy()
    parent_bond_states = np.asarray(condition.fixed_parent_bond_states, dtype=np.int64).copy()
    for sequence, full in enumerate(exterior_full_indices):
        atom_states[full] = int(endpoint_atoms[sequence])
        parent_bond_states[full] = int(endpoint_parent_bonds[sequence])

    decoration_anchors = np.zeros(len(endpoint_anchors), dtype=np.int64)
    for slot, encoded_sequence_anchor in enumerate(endpoint_anchors):
        encoded = int(encoded_sequence_anchor)
        if encoded == 0:
            continue
        sequence = encoded - 1
        if sequence < 0 or sequence >= len(exterior_full_indices):
            raise UgiJointEndToEndSamplingError(
                "flow endpoint decoration anchor points outside the generated exterior"
            )
        decoration_anchors[slot] = int(exterior_full_indices[sequence]) + 1

    return UgiChemistrySample(
        atom_states=atom_states,
        parent_bond_states=parent_bond_states,
        closure_bond_states=np.asarray(closure_bond_states, dtype=np.int64).copy(),
        decoration_anchor=0,
        decoration_anchors=decoration_anchors,
        decoration_atom_states=np.asarray(endpoint_decoration_atoms, dtype=np.int64).copy(),
        decoration_bond_states=np.asarray(endpoint_decoration_bonds, dtype=np.int64).copy(),
    )


@dataclass(frozen=True)
class UgiJointTerminalCompletion:
    """Restartable terminal completion outputs and advanced closure RNG state."""

    conditions: tuple[Any, ...]
    samples: tuple[Any, ...]
    rows: tuple[dict[str, Any], ...]
    molecules: tuple[Any, ...]
    closure_generator_state: Any
    terminal_generator_state: Any | None


def _terminal_atom_state_summary(
    conditions: Sequence[Any],
    samples: Sequence[Any],
    atom_vocabulary: Sequence[Any],
    *,
    maximum_distance_bin: int = 8,
) -> dict[str, Any]:
    """Summarize sampled exterior atom states by chemical origin and port distance."""

    if len(conditions) != len(samples) or maximum_distance_bin < 1:
        raise UgiJointEndToEndSamplingError("terminal atom-state summary is misaligned")
    summary: dict[str, Any] = {
        role: {
            str(distance): {
                "atom_state_counts": [0] * len(atom_vocabulary),
                "atoms": 0,
            }
            for distance in range(maximum_distance_bin + 1)
        }
        for role in ROLE_NAMES
    }
    valid_samples = 0
    for condition, sample in zip(conditions, samples, strict=True):
        if sample is None:
            continue
        valid_samples += 1
        features = recompute_adapter_distances(condition)
        for role in ROLE_NAMES:
            nodes = np.flatnonzero(
                (condition.origin_states == ORIGIN_TO_INDEX[role]) & ~condition.fixed_atom_mask
            )
            for node in nodes:
                distance = min(int(features.distance_to_own_port[int(node)]), maximum_distance_bin)
                state = int(sample.atom_states[int(node)])
                bucket = summary[role][str(distance)]
                bucket["atom_state_counts"][state] += 1
                bucket["atoms"] += 1
    return {
        "valid_terminal_samples": valid_samples,
        "maximum_distance_bin": maximum_distance_bin,
        "distance_bin_policy": f"exact_0_to_{maximum_distance_bin - 1}_then_ge_{maximum_distance_bin}",
        "atom_vocabulary": [
            {
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "explicit_hydrogens": state.explicit_hydrogens,
            }
            for state in atom_vocabulary
        ],
        "by_role_distance": summary,
    }


def _flow_endpoint_atom_state_summary(
    conditions: Sequence[Any],
    terminals: Sequence[UgiJointSparseTerminal],
    atom_vocabulary: Sequence[Any],
    *,
    maximum_distance_bin: int = 8,
) -> dict[str, Any]:
    """Summarize the categorical time-one flow state before terminal correction."""

    if len(conditions) != len(terminals):
        raise UgiJointEndToEndSamplingError("flow endpoint summary is misaligned")
    summary = {
        role: {
            str(distance): {
                "atom_state_counts": [0] * len(atom_vocabulary),
                "atoms": 0,
            }
            for distance in range(maximum_distance_bin + 1)
        }
        for role in ROLE_NAMES
    }
    available = 0
    for condition, terminal in zip(conditions, terminals, strict=True):
        endpoint = terminal.flow_endpoint_atom_states
        if endpoint is None:
            continue
        exterior_full_indices = []
        for role in ROLE_NAMES:
            exterior_full_indices.extend(
                np.flatnonzero(
                    (condition.origin_states == ORIGIN_TO_INDEX[role]) & ~condition.fixed_atom_mask
                ).tolist()
            )
        if len(exterior_full_indices) != len(endpoint):
            raise UgiJointEndToEndSamplingError("flow endpoint/full topology mismatch")
        available += 1
        features = recompute_adapter_distances(condition)
        for sequence, node in enumerate(exterior_full_indices):
            role = next(
                name
                for name in ROLE_NAMES
                if condition.origin_states[node] == ORIGIN_TO_INDEX[name]
            )
            distance = min(int(features.distance_to_own_port[node]), maximum_distance_bin)
            state = int(endpoint[sequence])
            bucket = summary[role][str(distance)]
            bucket["atom_state_counts"][state] += 1
            bucket["atoms"] += 1
    return {
        "available_terminal_states": available,
        "maximum_distance_bin": maximum_distance_bin,
        "distance_bin_policy": f"exact_0_to_{maximum_distance_bin - 1}_then_ge_{maximum_distance_bin}",
        "by_role_distance": summary,
    }


def _original_reference_assignments(repo: Path) -> Any:
    by_fold = {fold: [] for fold in ("train", "calibration", "heldout")}
    with gzip.open(
        repo / "data/splits/phase1/ugi_l1_assignments.csv.gz",
        "rt",
        newline="",
    ) as handle:
        for row in csv.DictReader(handle):
            by_fold[row["primary_product_fold"]].append(row)
    return SimpleNamespace(
        assignments_by_fold={fold: tuple(rows) for fold, rows in by_fold.items()}
    )


def _load_matched_programs(
    path: Path,
) -> tuple[tuple[UgiMorphologyProgram, ...], tuple[dict[str, Any], ...]]:
    value = json.loads(path.read_text())
    programs = []
    metadata = []
    for row in value["samples"]:
        program = row["program"]
        programs.append(
            UgiMorphologyProgram(
                node_counts=tuple(program["node_counts"]),
                junction_budgets=tuple(program["junction_budgets"]),
                cycle_ranks=tuple(program["cycle_ranks"]),
                attachment_counts=tuple(program.get("attachment_counts", (1, 1, 1))),
            )
        )
        metadata.append(
            {
                key: row[key]
                for key in (
                    "product_id",
                    "source_stratum",
                    "branch_class",
                    "component_novelty_class",
                    "held_role_class",
                )
                if key in row
            }
        )
    if not programs:
        raise UgiJointEndToEndSamplingError("matched staged result has no programs")
    return tuple(programs), tuple(metadata)


def _annotate_l1_terminal_admission(row: dict[str, Any], reaction: Any) -> None:
    """Add terminal-admission state without changing raw validity denominators."""

    row["raw_molecule_valid"] = bool(row.get("valid"))
    if not row["raw_molecule_valid"]:
        row["terminal_valid"] = False
        row["terminal_failure_type"] = row.get("failure_type") or "RawMoleculeInvalid"
        return
    if not row.get("component_reconstruction_valid"):
        row["terminal_valid"] = False
        row["terminal_failure_type"] = "L1ComponentRecoveryFailure"
        return
    exact_forward, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
        reaction,
        row["component_smiles_by_role"],
        row["smiles"],
    )
    row["l1_forward_verification"] = {
        "exact_product_reconstructed": exact_forward,
        "maximum_outcomes_saturated": saturated,
        "outcome_count": outcome_count,
    }
    row["terminal_valid"] = exact_forward
    row["terminal_failure_type"] = None if exact_forward else "L1ForwardConsistencyFailure"


def complete_ugi_joint_terminals(
    model: Any,
    closure_model: Any,
    terminals: Sequence[UgiJointSparseTerminal],
    corpus: Any,
    *,
    program_metadata: Sequence[dict[str, Any]],
    closure_generator_state: Any,
    allowed_ring_sizes: Sequence[int],
    maximum_heavy_degree: int,
    l1_reaction: Any | None = None,
    terminal_decoder_mode: str = "argmax",
    terminal_generator_state: Any | None = None,
    terminal_temperature: float = 1.0,
    terminal_atom_temperature: float | None = None,
    terminal_bond_temperature: float | None = None,
    terminal_decoration_temperature: float | None = None,
    terminal_atom_temperatures_by_origin: Sequence[float] | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    program_id: str | None = None,
    local_chemistry_constraint_scope: str = "role_edges_cycles_bounds",
) -> UgiJointTerminalCompletion:
    """Complete sparse terminals into sanitized products and exact components."""

    if torch is None or len(terminals) != len(program_metadata):
        raise UgiJointEndToEndSamplingError("terminal completion inputs are misaligned")
    if local_chemistry_constraint_scope not in {
        "role_edges_only",
        "role_edges_cycles_bounds",
    }:
        raise UgiJointEndToEndSamplingError(
            "unsupported role-local chemistry constraint scope"
        )
    if local_chemistry_support is not None:
        if program_id is None:
            raise UgiJointEndToEndSamplingError(
                "role-local chemistry support requires an explicit program id"
            )
        if tuple(local_chemistry_support.atom_states) != tuple(corpus.atom_vocabulary):
            raise UgiJointEndToEndSamplingError(
                "role-local policy atom vocabulary differs from the sampling corpus"
            )
    elif program_id is not None:
        raise UgiJointEndToEndSamplingError(
            "a role-local chemistry program id requires a support policy"
        )
    closure_generator = torch.Generator()
    closure_generator.set_state(closure_generator_state)
    terminal_generator = None
    if terminal_decoder_mode != "argmax":
        if terminal_generator_state is None:
            raise UgiJointEndToEndSamplingError(
                "non-argmax terminal decoding requires an explicit generator state"
            )
        terminal_generator = torch.Generator()
        terminal_generator.set_state(terminal_generator_state)
    elif terminal_generator_state is not None:
        raise UgiJointEndToEndSamplingError(
            "argmax terminal decoding does not accept a generator state"
        )
    conditions = []
    samples = []
    rows = []
    for sample_index, terminal in enumerate(terminals):
        offspring_by_role = {
            role: terminal.offspring[role_index] for role_index, role in enumerate(ROLE_NAMES)
        }
        left_by_role = {}
        right_by_role = {}
        for role_index, role in enumerate(ROLE_NAMES):
            left, right = sample_sparse_closures(
                closure_model,
                terminal.offspring[role_index],
                role_index=role_index,
                cycle_rank=terminal.program.cycle_ranks[role_index],
                attachment_count=terminal.program.attachment_counts[role_index],
                generator=closure_generator,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                device="cpu",
            )
            left_by_role[role] = left
            right_by_role[role] = right
        condition = assemble_ugi_chemistry_topology_condition(
            structure_id=f"joint_generated_{sample_index:04d}",
            offspring_by_role=offspring_by_role,
            attachment_counts_by_role={
                role: terminal.program.attachment_counts[role_index]
                for role_index, role in enumerate(ROLE_NAMES)
            },
            closure_left_by_role=left_by_role,
            closure_right_by_role=right_by_role,
            schema=corpus.core_schema,
        )
        exterior_full_indices = []
        for role in ROLE_NAMES:
            exterior_full_indices.extend(
                np.flatnonzero(
                    (condition.origin_states == ORIGIN_TO_INDEX[role]) & ~condition.fixed_atom_mask
                ).tolist()
            )
        if len(exterior_full_indices) != terminal.program.node_count:
            raise UgiJointEndToEndSamplingError("joint exterior/full topology mismatch")
        sequence_by_full = {full: sequence for sequence, full in enumerate(exterior_full_indices)}
        atom_logits = torch.zeros(
            (1, condition.node_count, len(corpus.atom_vocabulary)), dtype=torch.float32
        )
        parent_logits = torch.zeros((1, condition.node_count, 4), dtype=torch.float32)
        for sequence, full in enumerate(exterior_full_indices):
            atom_logits[0, full] = torch.from_numpy(terminal.atom_logits[sequence])
            parent_logits[0, full] = torch.from_numpy(terminal.parent_bond_logits[sequence])
        closure_sequence_pairs = [
            (sequence_by_full[int(left)], sequence_by_full[int(right)])
            for left, right in zip(condition.closure_left, condition.closure_right, strict=True)
        ]
        if closure_sequence_pairs:
            hidden = torch.from_numpy(terminal.hidden)
            closure_features = torch.stack(
                [torch.cat((hidden[left], hidden[right])) for left, right in closure_sequence_pairs]
            )
            closure_logits = model.closure_bond_output(closure_features)[None, :, :]
        else:
            closure_logits = torch.zeros((1, 0, 4), dtype=torch.float32)
        decoration_anchor_logits = torch.full(
            (1, model.maximum_decorations, condition.node_count + 1),
            -torch.inf,
            dtype=torch.float32,
        )
        decoration_anchor_logits[:, :, 0] = torch.from_numpy(
            terminal.decoration_anchor_logits[:, 0]
        )
        for sequence, full in enumerate(exterior_full_indices):
            decoration_anchor_logits[0, :, full + 1] = torch.from_numpy(
                terminal.decoration_anchor_logits[:, sequence + 1]
            )
        terminal_chemistry = {
            "nodes": atom_logits,
            "parent_bonds": parent_logits,
            "closure_bonds": closure_logits,
        }
        if model.maximum_decorations == 1:
            terminal_chemistry["decoration_anchor"] = decoration_anchor_logits[:, 0, :]
        else:
            terminal_chemistry.update(
                {
                    "decoration_anchors": decoration_anchor_logits,
                    "decoration_atoms": torch.from_numpy(terminal.decoration_atom_logits)[
                        None, :, :
                    ],
                    "decoration_bonds": torch.from_numpy(terminal.decoration_bond_logits)[
                        None, :, :
                    ],
                }
            )
        decode_failure: dict[str, Any] | None = None
        try:
            decoded = valence_constrained_terminal_sample(
                condition,
                terminal_chemistry,
                0,
                corpus.atom_vocabulary,
                model.maximum_decorations,
                mode=terminal_decoder_mode,
                generator=terminal_generator,
                temperature=terminal_temperature,
                atom_temperature=terminal_atom_temperature,
                bond_temperature=terminal_bond_temperature,
                decoration_temperature=terminal_decoration_temperature,
                atom_temperatures_by_origin=terminal_atom_temperatures_by_origin,
                local_chemistry_support=local_chemistry_support,
                program_id=program_id,
                local_chemistry_constraint_scope=local_chemistry_constraint_scope,
            )
        except RuntimeError as error:
            decoded = None
            decode_failure = _terminal_failure_detail(error)
        conditions.append(condition)
        samples.append(decoded)
        rows.append(
            {
                "structure_id": condition.structure_id,
                **program_metadata[sample_index],
                "program": terminal.program.__dict__,
                "offspring_by_role": {
                    role: offspring_by_role[role].tolist() for role in ROLE_NAMES
                },
                "terminal_decoder_mode": terminal_decoder_mode,
                "terminal_temperature": terminal_temperature,
                "terminal_atom_temperature": terminal_atom_temperature,
                "terminal_bond_temperature": terminal_bond_temperature,
                "terminal_decoration_temperature": terminal_decoration_temperature,
                "terminal_atom_temperatures_by_origin": (
                    list(terminal_atom_temperatures_by_origin)
                    if terminal_atom_temperatures_by_origin is not None
                    else None
                ),
                "terminal_failure_detail": decode_failure,
                "molecule_failure_detail": None,
            }
        )
    molecules = []
    for row, condition, sample in zip(rows, conditions, samples, strict=True):
        if sample is None:
            row["smiles"] = None
            row["valid"] = False
            row["failure_type"] = "TerminalSupportFailure"
            row["component_smiles_by_role"] = None
            row["component_reconstruction_valid"] = False
            row["component_reconstruction_error"] = "terminal support failed"
            if l1_reaction is not None:
                _annotate_l1_terminal_admission(row, l1_reaction)
            continue
        try:
            molecule = chemistry_sample_to_molecule(
                condition,
                sample,
                corpus.atom_vocabulary,
            )
            row["smiles"] = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            row["valid"] = True
            row["failure_type"] = None
            try:
                row["component_smiles_by_role"] = generated_ugi_component_smiles(
                    condition,
                    sample,
                    corpus.atom_vocabulary,
                )
                row["component_reconstruction_valid"] = True
                row["component_reconstruction_error"] = None
            except UgiGeneratedComponentError as error:
                row["component_smiles_by_role"] = None
                row["component_reconstruction_valid"] = False
                row["component_reconstruction_error"] = str(error)
            if l1_reaction is not None:
                _annotate_l1_terminal_admission(row, l1_reaction)
            molecules.append(molecule)
        except (ValueError, RuntimeError) as error:
            row["smiles"] = None
            row["valid"] = False
            row["failure_type"] = "MoleculeSanitizationFailure"
            row["molecule_failure_detail"] = _runtime_failure_detail(
                error, stage="molecule_construction"
            )
            row["component_smiles_by_role"] = None
            row["component_reconstruction_valid"] = False
            row["component_reconstruction_error"] = "product molecule is invalid"
            if l1_reaction is not None:
                _annotate_l1_terminal_admission(row, l1_reaction)
    return UgiJointTerminalCompletion(
        conditions=tuple(conditions),
        samples=tuple(samples),
        rows=tuple(rows),
        molecules=tuple(molecules),
        closure_generator_state=closure_generator.get_state().clone(),
        terminal_generator_state=(
            terminal_generator.get_state().clone() if terminal_generator is not None else None
        ),
    )


def sample_ugi_joint_end_to_end(
    repo: Path,
    output_dir: Path,
    *,
    joint_checkpoint_path: Path,
    closure_checkpoint_path: Path,
    matched_staged_result_path: Path,
    prepared_cache_path: Path | None = None,
    sample_steps: int,
    batch_size: int,
    seed: int,
    overwrite: bool,
    maximum_adjacent_branch_runs: Sequence[int | None] | None = None,
    qualified_reactions_path: Path | None = None,
    evaluate_exact_l1_terminal_admission: bool = False,
    terminal_decoder_mode: str = "argmax",
    terminal_decoder_seed: int | None = None,
    terminal_temperature: float = 1.0,
    terminal_atom_temperature: float | None = None,
    terminal_bond_temperature: float | None = None,
    terminal_decoration_temperature: float | None = None,
    terminal_atom_temperatures_by_origin: Sequence[float] | None = None,
    local_chemistry_policy_path: Path | None = None,
    local_chemistry_program_id: str | None = None,
    local_chemistry_constraint_scope: str = "role_edges_cycles_bounds",
    program_offset: int = 0,
    program_limit: int | None = None,
    reference_comparison_mode: str = "full",
    render: bool = True,
    record_timing: bool = True,
    device: str = "cpu",
) -> dict[str, Any]:
    """Generate on the exact same global programs used by the staged probe."""

    if torch is None:
        raise UgiJointEndToEndSamplingError("joint end-to-end sampling requires torch")
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise UgiJointEndToEndSamplingError("CUDA sampling was requested but CUDA is unavailable")
    _validate_reference_comparison_mode(reference_comparison_mode)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise UgiJointEndToEndSamplingError(f"output directory is nonempty: {output_dir}")
    if evaluate_exact_l1_terminal_admission and qualified_reactions_path is None:
        raise UgiJointEndToEndSamplingError(
            "exact L1 forward validity requires a qualified reaction registry"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "result.json"
    if overwrite and result_path.is_file():
        result_path.unlink()
    joint_checkpoint = _load_checkpoint(
        joint_checkpoint_path,
        "phase1_ugi_joint_sparse_checkpoint.v1",
    )
    closure_checkpoint = _load_checkpoint(
        closure_checkpoint_path,
        (
            "phase1_ugi_sparse_closure_checkpoint.v1",
            "phase1_ugi_sparse_closure_checkpoint.v2",
        ),
    )
    assignment_path = str(joint_checkpoint["inputs"]["assignments"]["path"])
    resolved_cache = _resolve_prepared_cache(repo, joint_checkpoint, prepared_cache_path)
    if resolved_cache is not None:
        corpus = _load_prepared_corpus(*resolved_cache)
    else:
        corpus = _load_expanded_chemistry(repo)
    reference_corpus = None
    if reference_comparison_mode == "full":
        reference_corpus = (
            _original_reference_assignments(repo)
            if assignment_path.endswith("data/splits/phase1/ugi_l1_assignments.csv.gz")
            else corpus
        )
    programs, program_metadata = _load_matched_programs(matched_staged_result_path)
    programs, program_metadata = _slice_matched_programs(
        programs,
        program_metadata,
        offset=program_offset,
        limit=program_limit,
    )
    architecture = dict(joint_checkpoint["model_config"])
    architecture.pop("source_probability_floor")
    model = UgiJointSparseFlow(
        atom_classes=len(corpus.atom_vocabulary),
        **architecture,
    )
    model.load_state_dict(joint_checkpoint["model_state"])
    model.to(resolved_device)
    closure_model = _closure_model(closure_checkpoint)
    l1_reaction = (
        load_ugi_reaction_contract(qualified_reactions_path)
        if evaluate_exact_l1_terminal_admission and qualified_reactions_path is not None
        else None
    )
    if local_chemistry_policy_path is not None:
        if local_chemistry_program_id is None:
            raise UgiJointEndToEndSamplingError(
                "a local chemistry policy requires an explicit program id"
            )
        local_chemistry_support = LocalChemistrySupport.from_mapping(
            json.loads(local_chemistry_policy_path.read_text())
        )
        local_chemistry_policy_sha256 = sha256_file(local_chemistry_policy_path)
    else:
        if local_chemistry_program_id is not None:
            raise UgiJointEndToEndSamplingError(
                "a local chemistry program id requires a policy"
            )
        local_chemistry_support = None
        local_chemistry_policy_sha256 = None
    start = time.perf_counter()
    terminals, joint_sampling = sample_ugi_joint_sparse_terminals(
        model,
        programs,
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in joint_checkpoint["source_marginals"].items()
        },
        sample_steps=sample_steps,
        batch_size=batch_size,
        seed=seed,
        device=str(resolved_device),
        allowed_ring_sizes=closure_checkpoint["allowed_ring_sizes"],
        maximum_heavy_degree=int(closure_checkpoint["maximum_heavy_degree"]),
        maximum_adjacent_branch_runs=maximum_adjacent_branch_runs,
    )
    # Terminal molecule construction and RDKit admission are CPU work.  The restartable sampler has
    # already converted every flowed terminal tensor to NumPy, so moving the small output heads back
    # here avoids device mismatches without changing any categorical draw.
    model.to("cpu")
    closure_generator_state = torch.Generator().manual_seed(seed + 1).get_state()
    if terminal_decoder_mode != "argmax":
        resolved_terminal_seed = (
            seed + 2 if terminal_decoder_seed is None else terminal_decoder_seed
        )
        terminal_generator_state = torch.Generator().manual_seed(resolved_terminal_seed).get_state()
    else:
        if terminal_decoder_seed is not None:
            raise UgiJointEndToEndSamplingError(
                "terminal decoder seed is only valid for non-argmax decoding"
            )
        resolved_terminal_seed = None
        terminal_generator_state = None
    completion = complete_ugi_joint_terminals(
        model,
        closure_model,
        terminals,
        corpus,
        program_metadata=program_metadata,
        closure_generator_state=closure_generator_state,
        allowed_ring_sizes=closure_checkpoint["allowed_ring_sizes"],
        maximum_heavy_degree=int(closure_checkpoint["maximum_heavy_degree"]),
        l1_reaction=l1_reaction,
        terminal_decoder_mode=terminal_decoder_mode,
        terminal_generator_state=terminal_generator_state,
        terminal_temperature=terminal_temperature,
        terminal_atom_temperature=terminal_atom_temperature,
        terminal_bond_temperature=terminal_bond_temperature,
        terminal_decoration_temperature=terminal_decoration_temperature,
        terminal_atom_temperatures_by_origin=terminal_atom_temperatures_by_origin,
        local_chemistry_support=local_chemistry_support,
        program_id=local_chemistry_program_id,
        local_chemistry_constraint_scope=local_chemistry_constraint_scope,
    )
    rows = list(completion.rows)
    molecules = list(completion.molecules)
    statistics = chemistry_sample_statistics(
        completion.conditions,
        completion.samples,
        corpus.atom_vocabulary,
    )
    statistics["terminal_atom_states"] = _terminal_atom_state_summary(
        completion.conditions,
        completion.samples,
        corpus.atom_vocabulary,
    )
    statistics["flow_endpoint_atom_states"] = _flow_endpoint_atom_state_summary(
        completion.conditions,
        terminals,
        corpus.atom_vocabulary,
    )
    sampling_seconds = time.perf_counter() - start if record_timing else None
    render_path = output_dir / "samples.png"
    rendered_molecules = 0
    result = {
        "schema_version": "phase1_ugi_joint_end_to_end_sampling.v1",
        "status": "sampling_complete_reference_pending",
        "seed": seed,
        "samples": rows,
        "statistics": statistics,
        "reference_comparison": None,
        "reference_comparison_status": "pending",
        "sampling": {
            **joint_sampling,
            "device": str(resolved_device),
            "sampling_seconds": sampling_seconds,
            "total_seconds": sampling_seconds,
            "matched_global_programs": len(programs),
            "program_offset": program_offset,
            "program_limit": program_limit,
            "evaluate_exact_l1_terminal_admission": evaluate_exact_l1_terminal_admission,
            "terminal_decoder": {
                "mode": terminal_decoder_mode,
                "seed": resolved_terminal_seed,
                "temperature": terminal_temperature,
                "atom_temperature": terminal_atom_temperature,
                "bond_temperature": terminal_bond_temperature,
                "decoration_temperature": terminal_decoration_temperature,
                "atom_temperatures_by_origin": (
                    list(terminal_atom_temperatures_by_origin)
                    if terminal_atom_temperatures_by_origin is not None
                    else None
                ),
                "local_chemistry_policy_sha256": local_chemistry_policy_sha256,
                "local_chemistry_program_id": local_chemistry_program_id,
                "local_chemistry_constraint_scope": (
                    local_chemistry_constraint_scope
                    if local_chemistry_support is not None
                    else None
                ),
            },
            "local_chemistry_policy": (
                {
                    "path": str(local_chemistry_policy_path),
                    "sha256": local_chemistry_policy_sha256,
                }
                if local_chemistry_policy_path is not None
                else None
            ),
            "qualified_reactions": (
                {
                    "path": str(qualified_reactions_path),
                    "sha256": sha256_file(qualified_reactions_path),
                }
                if qualified_reactions_path is not None
                else None
            ),
        },
        "checkpoints": {
            "joint": str(joint_checkpoint_path),
            "closure": str(closure_checkpoint_path),
        },
        "matched_staged_result": str(matched_staged_result_path),
        "render": None,
        "rendered_molecules": rendered_molecules,
    }
    # Persist the scientifically complete molecule ledger before optional
    # rendering and the expensive many-to-many frozen-reference comparison.
    _atomic_json(result_path, result)
    if molecules and render:
        rendered_molecules = min(len(molecules), 60)
        image = Draw.MolsToGridImage(
            molecules[:rendered_molecules],
            molsPerRow=3,
            subImgSize=(480, 320),
            legends=[row["structure_id"] for row in rows if row["valid"]][:rendered_molecules],
        )
        image.save(render_path)
        result["render"] = str(render_path)
        result["rendered_molecules"] = rendered_molecules
        result["sampling"]["total_seconds"] = time.perf_counter() - start if record_timing else None
        _atomic_json(result_path, result)
    result = _complete_reference_comparison(
        result,
        molecules=molecules,
        reference_corpus=reference_corpus,
        mode=reference_comparison_mode,
    )
    result["sampling"]["total_seconds"] = time.perf_counter() - start if record_timing else None
    _atomic_json(result_path, result)
    return result
