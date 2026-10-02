"""Packed, deterministic cache for production synthesis-program training.

Store all qualified Ugi, BL, and LX products as sparse NumPy arrays in a
deterministic ZIP container. Records are reconstructed lazily by index, without
dense adjacency matrices or executable pickle payloads.
"""

from __future__ import annotations

import io
import itertools
import json
import math
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import iter_csv, read_json_object, stable_json, write_json
from forge.corpus.reaction_program_records import admits_reaction_program_structure
from forge.corpus.synthesis_program_representation import (
    SynthesisProgramRepresentationError,
    synthesis_program_contracts,
    synthesis_program_vocabulary,
)
from forge.model.conditioning.reaction_program import ReactionProgramVocabulary
from forge.model.networks.dense_flow import AtomState
from forge.model.networks.sparse_flow import SparseGraphRecord
from forge.model.representation.synthesis_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
    tensorize_synthesis_program_product,
)
from forge.model.representation.vocabulary import load_atom_vocabulary

CACHE_CONFIG_SCHEMA = "forge.synthesis_program_production_cache_config.v1"
CACHE_SCHEMA = "forge.synthesis_program_production_cache.v1"
CACHE_RESULT_SCHEMA = "forge.synthesis_program_production_cache_result.v1"

_FOLD_STATES = ("train", "calibration", "heldout")
_ARRAY_KEYS = frozenset(
    {
        "metadata",
        "node_offsets",
        "closure_offsets",
        "block_offsets",
        "record_id_offsets",
        "record_id_bytes",
        "smiles_offsets",
        "smiles_bytes",
        "node_states",
        "parents",
        "parent_bonds",
        "closure_left",
        "closure_right",
        "closure_bonds",
        "canonical_atom_order",
        "role_states",
        "core_position_states",
        "fixed_atom_mask",
        "fixed_parent_bond_mask",
        "fixed_closure_bond_mask",
        "block_role_states",
        "block_starts",
        "block_stops",
        "program_states",
        "program_depths",
        "fold_states",
        "source_weights",
    }
)


class SynthesisProgramProductionCacheError(ValueError):
    """The packed cache or its source ledgers violate the frozen design."""


def _validate_representation_intervention(
    *,
    config: Mapping[str, Any],
    design: Mapping[str, Any],
    representation: Mapping[str, Any],
    raw_inputs: Mapping[str, Any],
    paths: Mapping[str, Path],
    repo: Path,
) -> None:
    """Admit only the prespecified false-to-true BL semantic-core intervention."""

    intervention = config.get("representation_intervention")
    if not isinstance(intervention, dict) or set(intervention) != {
        "program_id",
        "change",
    }:
        raise SynthesisProgramProductionCacheError(
            "representation override lacks a bounded intervention contract"
        )
    program_id = str(intervention["program_id"])
    if intervention["change"] != "promote_all_semantic_core_atoms_to_fixed":
        raise SynthesisProgramProductionCacheError("unsupported representation intervention")
    if "representation_qualification" not in paths:
        raise SynthesisProgramProductionCacheError(
            "representation intervention has no pinned full-corpus qualification"
        )
    qualification = read_json_object(
        paths["representation_qualification"],
        error=SynthesisProgramProductionCacheError,
        label="representation intervention qualification",
    )
    qualified_config = qualification.get("config")
    expected_config = pin_record(paths["representation_config"], repo)
    if (
        qualification.get("status") != "pass"
        or not isinstance(qualified_config, dict)
        or qualified_config.get("path") != expected_config["path"]
        or qualified_config.get("sha256") != expected_config["sha256"]
        or not qualification.get("gates", {}).get("fixed_core_policy_exact", False)
    ):
        raise SynthesisProgramProductionCacheError(
            "representation intervention has not passed its full-corpus fixed-core gate"
        )
    base_path = resolve_pin(
        design["inputs"]["representation_config"],
        repo,
        label="base representation config",
    )
    base = read_json_object(
        base_path,
        error=SynthesisProgramProductionCacheError,
        label="base representation config",
    )
    for field in ("schema_version", "seed", "inputs", "support_bounds"):
        if base.get(field) != representation.get(field):
            raise SynthesisProgramProductionCacheError(
                f"representation intervention changed forbidden field {field}"
            )
    base_programs = {str(value["program_id"]): value for value in base.get("programs", ())}
    new_programs = {str(value["program_id"]): value for value in representation.get("programs", ())}
    if set(base_programs) != set(new_programs) or program_id not in base_programs:
        raise SynthesisProgramProductionCacheError(
            "representation intervention changed the admitted program set"
        )
    for current_program, base_contract in base_programs.items():
        expected = dict(base_contract)
        if current_program == program_id:
            if bool(expected.get("fix_core_atoms", False)):
                raise SynthesisProgramProductionCacheError(
                    "representation intervention target was already fixed"
                )
            expected["fix_core_atoms"] = True
        if new_programs[current_program] != expected:
            raise SynthesisProgramProductionCacheError(
                f"representation intervention changed forbidden program fields: {current_program}"
            )
    if raw_inputs["representation_config"] == design["inputs"]["representation_config"]:
        raise SynthesisProgramProductionCacheError(
            "representation intervention did not change the representation"
        )


@dataclass(frozen=True)
class SynthesisProgramSourceRecord:
    """One qualified tensor record plus its non-neural sampling metadata."""

    record: SynthesisProgramGraphRecord
    fold: str
    source_weight: float


def _metadata_array(value: Mapping[str, Any]) -> np.ndarray:
    return np.frombuffer(stable_json(value).encode("utf-8"), dtype=np.uint8).copy()


def _encode_strings(values: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    chunks: list[bytes] = []
    cursor = 0
    for index, value in enumerate(values, start=1):
        encoded = value.encode("utf-8")
        if b"\x00" in encoded:
            raise SynthesisProgramProductionCacheError("cache strings cannot contain NUL bytes")
        chunks.append(encoded)
        cursor += len(encoded)
        offsets[index] = cursor
    payload = np.frombuffer(b"".join(chunks), dtype=np.uint8).copy()
    return offsets, payload


def _decode_string(offsets: np.ndarray, payload: np.ndarray, index: int) -> str:
    start = int(offsets[index])
    stop = int(offsets[index + 1])
    return payload[start:stop].tobytes().decode("utf-8")


def _npy_bytes(array: np.ndarray) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(handle, np.ascontiguousarray(array), allow_pickle=False)
    return handle.getvalue()


def _write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a stable, safe NPZ without wall-clock ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            info.create_system = 3
            archive.writestr(info, _npy_bytes(arrays[name]))


def _group_rows(
    rows: Iterator[dict[str, str]], *, identifier: str, atom_index: str
) -> Iterator[tuple[str, list[dict[str, str]]]]:
    seen: set[str] = set()
    for record_id, group in itertools.groupby(rows, key=lambda row: row[identifier]):
        if record_id in seen:
            raise SynthesisProgramProductionCacheError(
                f"semantic rows are not contiguous for {record_id}"
            )
        seen.add(record_id)
        values = list(group)
        if [int(row[atom_index]) for row in values] != list(range(len(values))):
            raise SynthesisProgramProductionCacheError(
                f"semantic atom indices are not contiguous for {record_id}"
            )
        yield record_id, values


def _check_semantics(
    *,
    record_id: str,
    program_id: str,
    roles: Sequence[str],
    core_positions: Sequence[str],
    depth: int,
    contract: Mapping[str, Any],
) -> None:
    if set(roles) != set(contract["roles"]):
        raise SynthesisProgramProductionCacheError(
            f"observed roles changed for {record_id}: {sorted(set(roles))}"
        )
    observed_core = {
        value.removeprefix(f"{program_id}:") for value in core_positions if value != "exterior"
    }
    if not observed_core.issubset(set(contract["core_positions"])):
        raise SynthesisProgramProductionCacheError(
            f"observed core positions changed for {record_id}: {sorted(observed_core)}"
        )
    if depth not in contract["allowed_depths"]:
        raise SynthesisProgramProductionCacheError(
            f"observed program depth changed for {record_id}: {depth}"
        )


def _tensorize(
    *,
    record_id: str,
    program_id: str,
    smiles: str,
    roles: Sequence[str],
    core_positions: Sequence[str],
    depth: int,
    fixed_indices: Sequence[int],
    contract: Mapping[str, Any],
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[AtomState],
) -> SynthesisProgramGraphRecord:
    _check_semantics(
        record_id=record_id,
        program_id=program_id,
        roles=roles,
        core_positions=core_positions,
        depth=depth,
        contract=contract,
    )
    if bool(contract["fix_core_atoms"]) != bool(fixed_indices):
        raise SynthesisProgramProductionCacheError(f"fixed-core policy changed for {record_id}")
    try:
        return tensorize_synthesis_program_product(
            record_id=record_id,
            program_id=program_id,
            canonical_product_smiles=smiles,
            atom_roles=roles,
            atom_core_positions=core_positions,
            program_depth=depth,
            vocabulary=vocabulary,
            atom_vocabulary=atom_vocabulary,
            fixed_atom_indices=fixed_indices,
        )
    except SynthesisProgramGraphError as error:
        raise SynthesisProgramProductionCacheError(str(error)) from error


def _iter_ugi_records(
    *,
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[AtomState],
) -> Iterator[SynthesisProgramSourceRecord]:
    assignments = iter_csv(paths["ugi_assignments"])
    products = iter_csv(paths["ugi_semantic_products"])
    atoms = _group_rows(
        iter_csv(paths["ugi_semantic_atoms"]),
        identifier="product_id",
        atom_index="product_atom_index",
    )
    missing = object()
    for assignment, product, atom_group in itertools.zip_longest(
        assignments, products, atoms, fillvalue=missing
    ):
        if missing in (assignment, product, atom_group):
            raise SynthesisProgramProductionCacheError(
                "Ugi assignments, products and atoms have different lengths"
            )
        assert isinstance(assignment, dict) and isinstance(product, dict)
        assert isinstance(atom_group, tuple)
        record_id, atom_rows = atom_group
        if assignment["product_id"] != record_id or product["product_id"] != record_id:
            raise SynthesisProgramProductionCacheError(
                f"Ugi ledgers changed order or identity at {record_id}"
            )
        if assignment["canonical_product_smiles"] != product["product_smiles"]:
            raise SynthesisProgramProductionCacheError(f"Ugi SMILES changed for {record_id}")
        roles = [row["origin_role"] for row in atom_rows]
        program_id = str(contract["program_id"])
        core_positions = [
            f"{program_id}:{row['core_position']}" if row["core_position"] else "exterior"
            for row in atom_rows
        ]
        fixed = tuple(
            int(row["product_atom_index"])
            for row in atom_rows
            if row["is_ugi_core"].strip().lower() == "true"
        )
        if len(fixed) != 5:
            raise SynthesisProgramProductionCacheError(
                f"Ugi fixed core no longer has five atoms: {record_id}"
            )
        fold = assignment["primary_product_fold"]
        weight = float(assignment["family_balance_weight_raw"])
        if fold not in _FOLD_STATES or not math.isfinite(weight) or weight <= 0:
            raise SynthesisProgramProductionCacheError(
                f"invalid Ugi fold or family-balance weight: {record_id}"
            )
        yield SynthesisProgramSourceRecord(
            record=_tensorize(
                record_id=record_id,
                program_id=program_id,
                smiles=product["product_smiles"],
                roles=roles,
                core_positions=core_positions,
                depth=1,
                fixed_indices=fixed,
                contract=contract,
                vocabulary=vocabulary,
                atom_vocabulary=atom_vocabulary,
            ),
            fold=fold,
            source_weight=weight,
        )


def _iter_auxiliary_records(
    *,
    paths: Mapping[str, Path],
    contracts: Mapping[str, Mapping[str, Any]],
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[AtomState],
) -> Iterator[SynthesisProgramSourceRecord]:
    splits: dict[str, tuple[str, str, float]] = {}
    for row in iter_csv(paths["multireaction_splits"]):
        record_id = row["record_id"]
        if record_id in splits:
            raise SynthesisProgramProductionCacheError(f"duplicate split row: {record_id}")
        weight = float(row["source_balanced_weight"])
        fold = row["product_fold"]
        if fold not in _FOLD_STATES or not math.isfinite(weight) or weight <= 0:
            raise SynthesisProgramProductionCacheError(
                f"invalid auxiliary fold or source-balanced weight: {record_id}"
            )
        splits[record_id] = (row["program_id"], fold, weight)
    atlas = (
        row
        for row in iter_csv(paths["multireaction_atlas"])
        if admits_reaction_program_structure(row)
    )
    atoms = _group_rows(
        iter_csv(paths["multireaction_semantic_atoms"]),
        identifier="record_id",
        atom_index="atom_index",
    )
    observed: set[str] = set()
    missing = object()
    for product, atom_group in itertools.zip_longest(atlas, atoms, fillvalue=missing):
        if missing in (product, atom_group):
            raise SynthesisProgramProductionCacheError(
                "auxiliary atlas and semantic atoms have different lengths"
            )
        assert isinstance(product, dict) and isinstance(atom_group, tuple)
        record_id, atom_rows = atom_group
        if product["record_id"] != record_id:
            raise SynthesisProgramProductionCacheError(
                f"auxiliary ledgers changed order or identity at {record_id}"
            )
        if record_id not in splits:
            raise SynthesisProgramProductionCacheError(f"missing split row: {record_id}")
        split_program, fold, weight = splits[record_id]
        program_id = product["program_id"]
        if split_program != program_id or program_id not in contracts:
            raise SynthesisProgramProductionCacheError(f"program identity changed: {record_id}")
        contract = contracts[program_id]
        if contract["source"] != "multireaction":
            raise SynthesisProgramProductionCacheError(
                f"auxiliary record uses a non-auxiliary contract: {record_id}"
            )
        depths = {int(row["program_depth"]) for row in atom_rows}
        depth = int(product["step_count"])
        if depths != {depth}:
            raise SynthesisProgramProductionCacheError(f"program depth changed: {record_id}")
        roles = [row["origin_role"] for row in atom_rows]
        core_positions = [
            f"{program_id}:{row['core_position']}" if row["core_position"] else "exterior"
            for row in atom_rows
        ]
        fixed_indices = (
            tuple(int(row["atom_index"]) for row in atom_rows if row["core_position"])
            if contract["fix_core_atoms"]
            else ()
        )
        observed.add(record_id)
        yield SynthesisProgramSourceRecord(
            record=_tensorize(
                record_id=record_id,
                program_id=program_id,
                smiles=product["canonical_product_smiles"],
                roles=roles,
                core_positions=core_positions,
                depth=depth,
                fixed_indices=fixed_indices,
                contract=contract,
                vocabulary=vocabulary,
                atom_vocabulary=atom_vocabulary,
            ),
            fold=fold,
            source_weight=weight,
        )
    if observed != set(splits):
        raise SynthesisProgramProductionCacheError(
            f"split and semantic ledgers differ for {len(set(splits).symmetric_difference(observed))} records"
        )


class _PackedArrays:
    def __init__(self) -> None:
        self.records = 0
        self.record_ids: list[str] = []
        self.smiles: list[str] = []
        self.node_offsets = [0]
        self.closure_offsets = [0]
        self.block_offsets = [0]
        self.node_states: list[np.ndarray] = []
        self.parents: list[np.ndarray] = []
        self.parent_bonds: list[np.ndarray] = []
        self.closure_left: list[np.ndarray] = []
        self.closure_right: list[np.ndarray] = []
        self.closure_bonds: list[np.ndarray] = []
        self.canonical_atom_order: list[np.ndarray] = []
        self.role_states: list[np.ndarray] = []
        self.core_position_states: list[np.ndarray] = []
        self.fixed_atom_mask: list[np.ndarray] = []
        self.fixed_parent_bond_mask: list[np.ndarray] = []
        self.fixed_closure_bond_mask: list[np.ndarray] = []
        self.block_role_states: list[int] = []
        self.block_starts: list[int] = []
        self.block_stops: list[int] = []
        self.program_states: list[int] = []
        self.program_depths: list[int] = []
        self.fold_states: list[int] = []
        self.source_weights: list[float] = []

    def append(self, value: SynthesisProgramSourceRecord) -> None:
        record = value.record
        self.records += 1
        self.record_ids.append(record.graph.structure_id)
        self.smiles.append(record.graph.canonical_smiles)
        for target, array in (
            (self.node_states, record.graph.node_states),
            (self.parents, record.graph.parents),
            (self.parent_bonds, record.graph.parent_bonds),
            (self.canonical_atom_order, record.canonical_atom_order),
            (self.role_states, record.role_states),
            (self.core_position_states, record.core_position_states),
            (self.fixed_atom_mask, record.fixed_atom_mask),
            (self.fixed_parent_bond_mask, record.fixed_parent_bond_mask),
        ):
            target.append(array)
        for target, array in (
            (self.closure_left, record.graph.closure_left),
            (self.closure_right, record.graph.closure_right),
            (self.closure_bonds, record.graph.closure_bonds),
            (self.fixed_closure_bond_mask, record.fixed_closure_bond_mask),
        ):
            target.append(array)
        self.node_offsets.append(self.node_offsets[-1] + record.node_count)
        self.closure_offsets.append(self.closure_offsets[-1] + record.graph.closure_count)
        for block in record.component_blocks:
            self.block_role_states.append(block.role_state)
            self.block_starts.append(block.start)
            self.block_stops.append(block.stop)
        self.block_offsets.append(self.block_offsets[-1] + len(record.component_blocks))
        self.program_states.append(record.program_state)
        self.program_depths.append(record.program_depth)
        self.fold_states.append(_FOLD_STATES.index(value.fold))
        self.source_weights.append(value.source_weight)

    @staticmethod
    def _concat(values: Sequence[np.ndarray], dtype: Any) -> np.ndarray:
        if not values:
            return np.asarray([], dtype=dtype)
        return np.concatenate(values).astype(dtype, copy=False)

    def arrays(self, metadata: Mapping[str, Any]) -> dict[str, np.ndarray]:
        record_id_offsets, record_id_bytes = _encode_strings(self.record_ids)
        smiles_offsets, smiles_bytes = _encode_strings(self.smiles)
        return {
            "metadata": _metadata_array(metadata),
            "node_offsets": np.asarray(self.node_offsets, dtype=np.int64),
            "closure_offsets": np.asarray(self.closure_offsets, dtype=np.int64),
            "block_offsets": np.asarray(self.block_offsets, dtype=np.int64),
            "record_id_offsets": record_id_offsets,
            "record_id_bytes": record_id_bytes,
            "smiles_offsets": smiles_offsets,
            "smiles_bytes": smiles_bytes,
            "node_states": self._concat(self.node_states, np.uint16),
            "parents": self._concat(self.parents, np.uint16),
            "parent_bonds": self._concat(self.parent_bonds, np.uint8),
            "closure_left": self._concat(self.closure_left, np.uint16),
            "closure_right": self._concat(self.closure_right, np.uint16),
            "closure_bonds": self._concat(self.closure_bonds, np.uint8),
            "canonical_atom_order": self._concat(self.canonical_atom_order, np.uint16),
            "role_states": self._concat(self.role_states, np.uint16),
            "core_position_states": self._concat(self.core_position_states, np.uint16),
            "fixed_atom_mask": self._concat(self.fixed_atom_mask, np.uint8),
            "fixed_parent_bond_mask": self._concat(self.fixed_parent_bond_mask, np.uint8),
            "fixed_closure_bond_mask": self._concat(self.fixed_closure_bond_mask, np.uint8),
            "block_role_states": np.asarray(self.block_role_states, dtype=np.uint16),
            "block_starts": np.asarray(self.block_starts, dtype=np.uint16),
            "block_stops": np.asarray(self.block_stops, dtype=np.uint16),
            "program_states": np.asarray(self.program_states, dtype=np.uint8),
            "program_depths": np.asarray(self.program_depths, dtype=np.uint8),
            "fold_states": np.asarray(self.fold_states, dtype=np.uint8),
            "source_weights": np.asarray(self.source_weights, dtype=np.float64),
        }


class SynthesisProgramProductionCache:
    """Validated packed corpus with lazy graph-record materialization."""

    def __init__(self, path: Path) -> None:
        try:
            archive = np.load(path, allow_pickle=False)
        except (OSError, ValueError, zipfile.BadZipFile) as error:
            raise SynthesisProgramProductionCacheError(
                f"production cache could not be loaded: {path}"
            ) from error
        if set(archive.files) != _ARRAY_KEYS:
            archive.close()
            raise SynthesisProgramProductionCacheError("production cache arrays changed")
        self.path = path
        self._archive = archive
        self.arrays = {name: archive[name] for name in archive.files}
        try:
            self.metadata = json.loads(self.arrays["metadata"].tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            self.close()
            raise SynthesisProgramProductionCacheError("cache metadata is malformed") from error
        if self.metadata.get("schema_version") != CACHE_SCHEMA:
            self.close()
            raise SynthesisProgramProductionCacheError("unsupported production cache schema")
        vocab = self.metadata["program_vocabulary"]
        self.vocabulary = ReactionProgramVocabulary(
            program_states=tuple(str(value) for value in vocab["program_states"]),
            role_states=tuple(str(value) for value in vocab["role_states"]),
            core_position_states=tuple(str(value) for value in vocab["core_position_states"]),
            maximum_steps=int(vocab["maximum_steps"]),
        )
        self.atom_vocabulary = tuple(
            AtomState(
                symbol=str(row["symbol"]),
                formal_charge=int(row["formal_charge"]),
                aromatic=bool(row["aromatic"]),
                explicit_hydrogens=int(row["explicit_hydrogens"]),
            )
            for row in self.metadata["atom_vocabulary"]
        )
        self._validate()

    def close(self) -> None:
        self._archive.close()

    def __enter__(self) -> SynthesisProgramProductionCache:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __len__(self) -> int:
        return int(self.arrays["program_states"].size)

    def _validate(self) -> None:
        count = len(self)
        for key in (
            "node_offsets",
            "closure_offsets",
            "block_offsets",
            "record_id_offsets",
            "smiles_offsets",
        ):
            offsets = self.arrays[key]
            if (
                offsets.dtype != np.int64
                or offsets.shape != (count + 1,)
                or int(offsets[0]) != 0
                or np.any(np.diff(offsets) < 0)
            ):
                raise SynthesisProgramProductionCacheError(f"invalid packed offsets: {key}")
        aligned = ("program_depths", "fold_states", "source_weights")
        if any(self.arrays[key].shape != (count,) for key in aligned):
            raise SynthesisProgramProductionCacheError("record metadata arrays are misaligned")
        if (
            np.any(self.arrays["fold_states"] >= len(_FOLD_STATES))
            or np.any(self.arrays["program_states"] == 0)
            or np.any(self.arrays["program_states"] >= len(self.vocabulary.program_states))
            or not np.isfinite(self.arrays["source_weights"]).all()
            or np.any(self.arrays["source_weights"] <= 0)
        ):
            raise SynthesisProgramProductionCacheError("record sampling metadata is invalid")
        for offset_key, payload_keys in (
            (
                "node_offsets",
                (
                    "node_states",
                    "parents",
                    "parent_bonds",
                    "canonical_atom_order",
                    "role_states",
                    "core_position_states",
                    "fixed_atom_mask",
                    "fixed_parent_bond_mask",
                ),
            ),
            (
                "closure_offsets",
                ("closure_left", "closure_right", "closure_bonds", "fixed_closure_bond_mask"),
            ),
            ("block_offsets", ("block_role_states", "block_starts", "block_stops")),
            ("record_id_offsets", ("record_id_bytes",)),
            ("smiles_offsets", ("smiles_bytes",)),
        ):
            expected = int(self.arrays[offset_key][-1])
            if any(self.arrays[key].size != expected for key in payload_keys):
                raise SynthesisProgramProductionCacheError(
                    f"packed payload length differs for {offset_key}"
                )
        expected_counts = {
            program: {fold: int(value) for fold, value in folds.items()}
            for program, folds in self.metadata["fold_counts"].items()
        }
        if self.fold_counts() != expected_counts:
            raise SynthesisProgramProductionCacheError("packed cache fold counts changed")

    def record_id(self, index: int) -> str:
        return _decode_string(
            self.arrays["record_id_offsets"], self.arrays["record_id_bytes"], index
        )

    def canonical_smiles(self, index: int) -> str:
        return _decode_string(self.arrays["smiles_offsets"], self.arrays["smiles_bytes"], index)

    def program_id(self, index: int) -> str:
        return self.vocabulary.program_states[int(self.arrays["program_states"][index])]

    def fold(self, index: int) -> str:
        return _FOLD_STATES[int(self.arrays["fold_states"][index])]

    def indices(self, *, program_id: str | None = None, fold: str | None = None) -> np.ndarray:
        mask = np.ones(len(self), dtype=np.bool_)
        if program_id is not None:
            try:
                state = self.vocabulary.program_to_index[program_id]
            except KeyError as error:
                raise SynthesisProgramProductionCacheError(
                    f"unknown cache program: {program_id}"
                ) from error
            mask &= self.arrays["program_states"] == state
        if fold is not None:
            if fold not in _FOLD_STATES:
                raise SynthesisProgramProductionCacheError(f"unknown cache fold: {fold}")
            mask &= self.arrays["fold_states"] == _FOLD_STATES.index(fold)
        return np.flatnonzero(mask)

    def fold_counts(self) -> dict[str, dict[str, int]]:
        output: dict[str, dict[str, int]] = {}
        for program_id in self.vocabulary.program_states[1:]:
            output[program_id] = {
                fold: int(self.indices(program_id=program_id, fold=fold).size)
                for fold in _FOLD_STATES
            }
        return output

    def training_measure(self, program_mass: Mapping[str, float]) -> np.ndarray:
        """Return program-balanced row probabilities under the frozen source weights."""

        if set(program_mass) != set(self.vocabulary.program_states[1:]):
            raise SynthesisProgramProductionCacheError("program mass does not cover the cache")
        mass = np.asarray([float(program_mass[key]) for key in program_mass], dtype=np.float64)
        if not np.isfinite(mass).all() or np.any(mass < 0) or not np.isclose(mass.sum(), 1.0):
            raise SynthesisProgramProductionCacheError("program mass is not normalized")
        measure = np.zeros(len(self), dtype=np.float64)
        for program_id, family_mass in program_mass.items():
            if family_mass == 0:
                continue
            indices = self.indices(program_id=program_id, fold="train")
            source = self.arrays["source_weights"][indices].astype(np.float64)
            if not len(indices) or source.sum() <= 0:
                raise SynthesisProgramProductionCacheError(
                    f"training measure has no support for {program_id}"
                )
            measure[indices] = float(family_mass) * source / source.sum()
        if not np.isclose(measure.sum(), 1.0):
            raise SynthesisProgramProductionCacheError("training measure failed normalization")
        return measure

    def source_marginals(
        self,
        measure: np.ndarray,
        *,
        node_classes: int,
        bond_classes: int,
        probability_floor: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if (
            measure.shape != (len(self),)
            or np.any(measure < 0)
            or not np.isfinite(measure).all()
            or not np.isclose(measure.sum(), 1.0)
            or probability_floor <= 0
        ):
            raise SynthesisProgramProductionCacheError("invalid source-marginal measure")
        nodes = np.full(node_classes, probability_floor, dtype=np.float64)
        bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
        active = np.flatnonzero(measure > 0)
        for index in active:
            node_start, node_stop = self.arrays["node_offsets"][index : index + 2]
            closure_start, closure_stop = self.arrays["closure_offsets"][index : index + 2]
            weight = float(measure[index])
            node_states = self.arrays["node_states"][node_start:node_stop]
            atom_fixed = self.arrays["fixed_atom_mask"][node_start:node_stop].astype(bool)
            parent_bonds = self.arrays["parent_bonds"][node_start + 1 : node_stop]
            parent_fixed = self.arrays["fixed_parent_bond_mask"][node_start + 1 : node_stop].astype(
                bool
            )
            closure_bonds = self.arrays["closure_bonds"][closure_start:closure_stop]
            closure_fixed = self.arrays["fixed_closure_bond_mask"][
                closure_start:closure_stop
            ].astype(bool)
            nodes += weight * np.bincount(node_states[~atom_fixed], minlength=node_classes)
            bonds += weight * np.bincount(parent_bonds[~parent_fixed], minlength=bond_classes)
            bonds += weight * np.bincount(closure_bonds[~closure_fixed], minlength=bond_classes)
        return nodes / nodes.sum(), bonds / bonds.sum()

    def program_role_source_marginals(
        self,
        measure: np.ndarray,
        *,
        node_classes: int,
        bond_classes: int,
        probability_floor: float,
        backoff_strength: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit smoothed program/role sources with family and global backoff.

        These are noise distributions, not component inventories.  Every atom and bond class keeps
        positive probability in every program/role cell.  An unsupported cell inherits its program
        distribution; a program with no active mass inherits the global distribution.  Parent bonds
        are assigned to the child role, while closure bonds use role zero (the program-level pool).
        """

        if (
            measure.shape != (len(self),)
            or np.any(measure < 0)
            or not np.isfinite(measure).all()
            or not np.isclose(measure.sum(), 1.0)
            or probability_floor <= 0
            or backoff_strength <= 0
        ):
            raise SynthesisProgramProductionCacheError(
                "invalid program-role source-marginal measure"
            )
        programs = len(self.vocabulary.program_states)
        roles = len(self.vocabulary.role_states)
        global_nodes = np.full(node_classes, probability_floor, dtype=np.float64)
        global_bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
        program_nodes = np.zeros((programs, node_classes), dtype=np.float64)
        program_bonds = np.zeros((programs, bond_classes), dtype=np.float64)
        role_nodes = np.zeros((programs, roles, node_classes), dtype=np.float64)
        role_bonds = np.zeros((programs, roles, bond_classes), dtype=np.float64)

        for index in np.flatnonzero(measure > 0):
            node_start, node_stop = self.arrays["node_offsets"][index : index + 2]
            closure_start, closure_stop = self.arrays["closure_offsets"][index : index + 2]
            weight = float(measure[index])
            program = int(self.arrays["program_states"][index])
            node_states = self.arrays["node_states"][node_start:node_stop]
            node_roles = self.arrays["role_states"][node_start:node_stop]
            atom_variable = ~self.arrays["fixed_atom_mask"][node_start:node_stop].astype(bool)
            parent_bonds = self.arrays["parent_bonds"][node_start + 1 : node_stop]
            parent_roles = node_roles[1:]
            parent_variable = ~self.arrays["fixed_parent_bond_mask"][
                node_start + 1 : node_stop
            ].astype(bool)
            closure_bonds = self.arrays["closure_bonds"][closure_start:closure_stop]
            closure_variable = ~self.arrays["fixed_closure_bond_mask"][
                closure_start:closure_stop
            ].astype(bool)

            node_counts = np.bincount(node_states[atom_variable], minlength=node_classes)
            parent_counts = np.bincount(parent_bonds[parent_variable], minlength=bond_classes)
            closure_counts = np.bincount(closure_bonds[closure_variable], minlength=bond_classes)
            global_nodes += weight * node_counts
            global_bonds += weight * (parent_counts + closure_counts)
            program_nodes[program] += weight * node_counts
            program_bonds[program] += weight * (parent_counts + closure_counts)
            for role in np.unique(node_roles[atom_variable]):
                selected = atom_variable & (node_roles == role)
                role_nodes[program, int(role)] += weight * np.bincount(
                    node_states[selected], minlength=node_classes
                )
            for role in np.unique(parent_roles[parent_variable]):
                selected = parent_variable & (parent_roles == role)
                role_bonds[program, int(role)] += weight * np.bincount(
                    parent_bonds[selected], minlength=bond_classes
                )

        global_nodes /= global_nodes.sum()
        global_bonds /= global_bonds.sum()
        output_nodes = np.empty_like(role_nodes)
        output_bonds = np.empty_like(role_bonds)
        output_nodes[0] = global_nodes
        output_bonds[0] = global_bonds
        for program in range(1, programs):
            program_node_source = backoff_strength * global_nodes + program_nodes[program]
            program_bond_source = backoff_strength * global_bonds + program_bonds[program]
            program_node_source /= program_node_source.sum()
            program_bond_source /= program_bond_source.sum()
            output_nodes[program, 0] = program_node_source
            output_bonds[program, 0] = program_bond_source
            for role in range(1, roles):
                node_source = backoff_strength * program_node_source + role_nodes[program, role]
                bond_source = backoff_strength * program_bond_source + role_bonds[program, role]
                output_nodes[program, role] = node_source / node_source.sum()
                output_bonds[program, role] = bond_source / bond_source.sum()
        if (
            np.any(output_nodes <= 0)
            or np.any(output_bonds <= 0)
            or not np.allclose(output_nodes.sum(axis=-1), 1.0)
            or not np.allclose(output_bonds.sum(axis=-1), 1.0)
        ):
            raise SynthesisProgramProductionCacheError(
                "program-role source marginals lost full support or normalization"
            )
        return output_nodes, output_bonds

    def record(self, index: int) -> SynthesisProgramGraphRecord:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        node_start, node_stop = (
            int(value) for value in self.arrays["node_offsets"][index : index + 2]
        )
        closure_start, closure_stop = (
            int(value) for value in self.arrays["closure_offsets"][index : index + 2]
        )
        block_start, block_stop = (
            int(value) for value in self.arrays["block_offsets"][index : index + 2]
        )

        def node_values(key: str, dtype: Any = np.int64) -> np.ndarray:
            return self.arrays[key][node_start:node_stop].astype(dtype, copy=True)

        def closure_values(key: str, dtype: Any = np.int64) -> np.ndarray:
            return self.arrays[key][closure_start:closure_stop].astype(dtype, copy=True)

        graph = SparseGraphRecord(
            structure_id=self.record_id(index),
            canonical_smiles=self.canonical_smiles(index),
            node_states=node_values("node_states"),
            parents=node_values("parents"),
            parent_bonds=node_values("parent_bonds"),
            closure_left=closure_values("closure_left"),
            closure_right=closure_values("closure_right"),
            closure_bonds=closure_values("closure_bonds"),
            # Training and layout sampling never consume a dense adjacency matrix.  Keeping this
            # explicitly empty prevents the cache from paying O(N^2) memory per product.
            edges=np.empty((0, 0), dtype=np.int8),
        )
        blocks = tuple(
            SynthesisProgramComponentBlock(
                role=self.vocabulary.role_states[
                    int(self.arrays["block_role_states"][block_index])
                ],
                role_state=int(self.arrays["block_role_states"][block_index]),
                start=int(self.arrays["block_starts"][block_index]),
                stop=int(self.arrays["block_stops"][block_index]),
            )
            for block_index in range(block_start, block_stop)
        )
        return SynthesisProgramGraphRecord(
            graph=graph,
            canonical_atom_order=node_values("canonical_atom_order"),
            program_id=self.program_id(index),
            program_state=int(self.arrays["program_states"][index]),
            program_depth=int(self.arrays["program_depths"][index]),
            role_states=node_values("role_states"),
            core_position_states=node_values("core_position_states"),
            component_blocks=blocks,
            fixed_atom_mask=node_values("fixed_atom_mask", np.bool_),
            fixed_parent_bond_mask=node_values("fixed_parent_bond_mask", np.bool_),
            fixed_closure_bond_mask=closure_values("fixed_closure_bond_mask", np.bool_),
        )

    def records(
        self, indices: Sequence[int] | np.ndarray
    ) -> tuple[SynthesisProgramGraphRecord, ...]:
        return tuple(self.record(int(index)) for index in indices)


def _atom_vocabulary_payload(values: Sequence[AtomState]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": state.symbol,
            "formal_charge": state.formal_charge,
            "aromatic": state.aromatic,
            "explicit_hydrogens": state.explicit_hydrogens,
        }
        for state in values
    ]


def build_synthesis_program_production_cache(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Build the full qualified cache under the frozen production design."""

    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionCacheError,
        label="synthesis-program production cache config",
    )
    if config.get("schema_version") != CACHE_CONFIG_SCHEMA:
        raise SynthesisProgramProductionCacheError("unsupported production cache config")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise SynthesisProgramProductionCacheError("production cache has no input pins")
    paths = {
        label: resolve_pin(value, repo, label=label) for label, value in sorted(raw_inputs.items())
    }
    required = {
        "production_design",
        "representation_config",
        "atom_vocabulary",
        "ugi_assignments",
        "ugi_semantic_products",
        "ugi_semantic_atoms",
        "multireaction_atlas",
        "multireaction_semantic_atoms",
        "multireaction_splits",
    }
    if config.get("representation_intervention") is not None:
        required.add("representation_qualification")
    if set(paths) != required:
        raise SynthesisProgramProductionCacheError(
            f"production cache inputs changed: {sorted(set(paths).symmetric_difference(required))}"
        )
    design = read_json_object(
        paths["production_design"],
        error=SynthesisProgramProductionCacheError,
        label="frozen production design",
    )
    if design.get("schema_version") != "forge.synthesis_program_production_design_config.v1":
        raise SynthesisProgramProductionCacheError("production design schema changed")
    representation = read_json_object(
        paths["representation_config"],
        error=SynthesisProgramProductionCacheError,
        label="shared representation config",
    )
    if (
        representation.get("schema_version")
        != "forge.shared_synthesis_program_representation_config.v1"
    ):
        raise SynthesisProgramProductionCacheError("representation config schema changed")
    if design["inputs"]["representation_config"] != raw_inputs["representation_config"]:
        _validate_representation_intervention(
            config=config,
            design=design,
            representation=representation,
            raw_inputs=raw_inputs,
            paths=paths,
            repo=repo,
        )
    elif config.get("representation_intervention") is not None:
        raise SynthesisProgramProductionCacheError(
            "representation intervention is declared without a representation change"
        )
    for label in (
        "atom_vocabulary",
        "ugi_assignments",
        "ugi_semantic_products",
        "ugi_semantic_atoms",
        "multireaction_atlas",
        "multireaction_semantic_atoms",
    ):
        if representation["inputs"].get(label) != raw_inputs[label]:
            raise SynthesisProgramProductionCacheError(
                f"cache and representation disagree on {label}"
            )
    if design["inputs"]["multireaction_splits"] != raw_inputs["multireaction_splits"]:
        raise SynthesisProgramProductionCacheError(
            "cache and production design disagree on multireaction_splits"
        )
    try:
        contracts = synthesis_program_contracts(representation)
        vocabulary = synthesis_program_vocabulary(contracts)
    except SynthesisProgramRepresentationError as error:
        raise SynthesisProgramProductionCacheError(str(error)) from error
    if set(contracts) != set(design["programs"]):
        raise SynthesisProgramProductionCacheError("design and representation programs differ")
    atom_vocabulary = load_atom_vocabulary(paths["atom_vocabulary"])
    ugi = [value for value in contracts.values() if value["source"] == "ugi"]
    if len(ugi) != 1:
        raise SynthesisProgramProductionCacheError("cache requires exactly one Ugi program")
    packed = _PackedArrays()
    counts: Counter[tuple[str, str]] = Counter()
    maximum_heavy_atoms = 0
    maximum_closures = 0
    fixed_failures = 0
    for value in itertools.chain(
        _iter_ugi_records(
            paths=paths,
            contract=ugi[0],
            vocabulary=vocabulary,
            atom_vocabulary=atom_vocabulary,
        ),
        _iter_auxiliary_records(
            paths=paths,
            contracts=contracts,
            vocabulary=vocabulary,
            atom_vocabulary=atom_vocabulary,
        ),
    ):
        record = value.record
        packed.append(value)
        counts[(record.program_id, value.fold)] += 1
        maximum_heavy_atoms = max(maximum_heavy_atoms, record.node_count)
        maximum_closures = max(maximum_closures, record.graph.closure_count)
        expected_fixed = (
            record.core_position_states > 1
            if contracts[record.program_id]["fix_core_atoms"]
            else np.zeros(record.node_count, dtype=np.bool_)
        )
        fixed_failures += int(not np.array_equal(record.fixed_atom_mask, expected_fixed))
        fixed_edges = int(np.count_nonzero(record.fixed_parent_bond_mask)) + int(
            np.count_nonzero(record.fixed_closure_bond_mask)
        )
        fixed_failures += int(bool(expected_fixed.any()) != (fixed_edges > 0))
    fold_counts = {
        program_id: {fold: counts[(program_id, fold)] for fold in _FOLD_STATES}
        for program_id in vocabulary.program_states[1:]
    }
    expected_counts = {
        program_id: {
            fold: int(value)
            for fold, value in design["programs"][program_id]["expected_fold_counts"].items()
        }
        for program_id in vocabulary.program_states[1:]
    }
    support = design["model"]
    metadata = {
        "schema_version": CACHE_SCHEMA,
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "fold_states": list(_FOLD_STATES),
        "fold_counts": fold_counts,
        "program_vocabulary": {
            "program_states": list(vocabulary.program_states),
            "role_states": list(vocabulary.role_states),
            "core_position_states": list(vocabulary.core_position_states),
            "maximum_steps": vocabulary.maximum_steps,
        },
        "atom_vocabulary": _atom_vocabulary_payload(atom_vocabulary),
        "support": {
            "maximum_heavy_atoms": int(support["maximum_heavy_atoms"]),
            "maximum_closures": int(support["maximum_closures"]),
            "observed_maximum_heavy_atoms": maximum_heavy_atoms,
            "observed_maximum_closures": maximum_closures,
        },
        "weight_fields": {
            program_id: str(design["programs"][program_id]["weight_field"])
            for program_id in vocabulary.program_states[1:]
        },
        "model_state_excludes": [
            "component_identifiers",
            "component_smiles",
            "component_fingerprints",
            "fragment_tokens",
            "biological_labels",
        ],
    }
    gates = {
        "all_frozen_fold_counts_exact": fold_counts == expected_counts,
        "all_records_packed": packed.records
        == sum(sum(value.values()) for value in expected_counts.values()),
        "full_heavy_atom_support_preserved": maximum_heavy_atoms
        <= int(support["maximum_heavy_atoms"]),
        "full_closure_support_preserved": maximum_closures <= int(support["maximum_closures"]),
        "fixed_state_policy_exact": fixed_failures == 0,
        "raw_family_count_sampling_absent": all(
            value in {"family_balance_weight_raw", "source_balanced_weight"}
            for value in metadata["weight_fields"].values()
        ),
    }
    if not all(gates.values()):
        raise SynthesisProgramProductionCacheError(
            f"production cache qualification failed before publication: {gates}"
        )
    _write_deterministic_npz(cache_path, packed.arrays(metadata))
    # Reload through the public validation path before authenticating the artifact.
    with SynthesisProgramProductionCache(cache_path) as cache:
        if cache.fold_counts() != expected_counts:
            raise SynthesisProgramProductionCacheError("published cache failed fold validation")
    result = {
        "schema_version": CACHE_RESULT_SCHEMA,
        "status": "pass",
        "config": pin_record(config_path, repo),
        "inputs": metadata["inputs"],
        "cache": artifact_record(cache_path),
        "records": packed.records,
        "fold_counts": fold_counts,
        "support": metadata["support"],
        "fixed_state_failures": fixed_failures,
        "gates": gates,
        "nonclaims": [
            "This cache is model input infrastructure, not evidence of model generalization.",
            "The cache contains reaction-enumerated support, not route-certified products.",
            "No route, oracle, candidate-selection or reductive-amination substructure metric is present.",
        ],
    }
    write_json(result_path, result)
    return result


__all__ = [
    "CACHE_CONFIG_SCHEMA",
    "CACHE_RESULT_SCHEMA",
    "CACHE_SCHEMA",
    "SynthesisProgramProductionCache",
    "SynthesisProgramProductionCacheError",
    "build_synthesis_program_production_cache",
]
