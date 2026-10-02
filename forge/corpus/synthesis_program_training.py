"""Model-ready shared training cache for qualified synthesis programs.

The cache contains complete sparse product targets and categorical synthesis-program coordinates.
It deliberately excludes precursor SMILES, component identifiers, fingerprints, and fragment tokens.
The bounded builder selects one training-fold record per program with equal program mass; it is an
integration/overfit gate, not the production training corpus.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import iter_csv, read_json_object, write_json
from forge.corpus.synthesis_program_representation import (
    SynthesisProgramRepresentationError,
    synthesis_program_contracts,
    synthesis_program_vocabulary,
)
from forge.model.conditioning.reaction_program import ReactionProgramVocabulary
from forge.model.conditioning.ugi import tensorize_ugi_l1_support_record
from forge.model.conditioning.ugi_chemistry import (
    core_schema_from_record,
    project_chemistry_topology_condition,
)
from forge.model.networks.dense_flow import AtomState
from forge.model.networks.sparse_flow import INDEX_TO_DENSE_BOND, SparseGraphRecord
from forge.model.representation.synthesis_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
    tensorize_synthesis_program_product,
)
from forge.model.representation.vocabulary import load_atom_vocabulary

CACHE_SCHEMA = "forge.synthesis_program_training_cache.v1"
RESULT_SCHEMA = "forge.synthesis_program_cache_qualification.v1"
CONFIG_SCHEMA = "forge.synthesis_program_cache_config.v1"


class SynthesisProgramTrainingError(ValueError):
    """Pinned shared-program inputs cannot produce the declared model cache."""


@dataclass(frozen=True)
class SynthesisProgramTrainingCache:
    """A bounded set of complete graph targets with equal mass per reaction program."""

    vocabulary: ReactionProgramVocabulary
    atom_vocabulary: tuple[AtomState, ...]
    records: tuple[SynthesisProgramGraphRecord, ...]
    sampling_weights: np.ndarray
    source_folds: tuple[str, ...]
    source_weights: np.ndarray
    inputs: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        count = len(self.records)
        if (
            count < 1
            or self.sampling_weights.shape != (count,)
            or self.source_weights.shape != (count,)
            or len(self.source_folds) != count
            or not np.isfinite(self.sampling_weights).all()
            or not np.isfinite(self.source_weights).all()
            or np.any(self.sampling_weights <= 0)
            or np.any(self.source_weights <= 0)
            or not np.isclose(self.sampling_weights.sum(), 1.0)
        ):
            raise SynthesisProgramTrainingError("shared training cache metadata is inconsistent")


def _rank(seed: int, program_id: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}|{program_id}|{record_id}".encode()).hexdigest()


def _selected_training_ids(
    *,
    assignments_path: Path,
    splits_path: Path,
    program_ids: Sequence[str],
    ugi_program_id: str,
    seed: int,
) -> tuple[dict[str, str], dict[str, tuple[str, float]]]:
    candidates: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
    for row in iter_csv(assignments_path):
        if row["primary_product_fold"] != "train":
            continue
        record_id = row["product_id"]
        weight = float(row["family_balance_weight_raw"])
        candidates[ugi_program_id].append(
            (_rank(seed, ugi_program_id, record_id), record_id, weight)
        )
    for row in iter_csv(splits_path):
        if row["product_fold"] != "train":
            continue
        program_id = row["program_id"]
        if program_id == ugi_program_id:
            raise SynthesisProgramTrainingError("auxiliary split reused the Ugi program identity")
        record_id = row["record_id"]
        weight = float(row["source_balanced_weight"])
        candidates[program_id].append((_rank(seed, program_id, record_id), record_id, weight))
    if set(candidates) != set(program_ids):
        raise SynthesisProgramTrainingError(
            f"training-fold programs changed: {sorted(candidates)} != {sorted(program_ids)}"
        )
    selected: dict[str, str] = {}
    metadata: dict[str, tuple[str, float]] = {}
    for program_id in sorted(candidates):
        rows = candidates[program_id]
        if not rows or any(not np.isfinite(weight) or weight <= 0 for _, _, weight in rows):
            raise SynthesisProgramTrainingError(
                f"{program_id} has no finite positive training-fold measure"
            )
        _, record_id, weight = min(rows)
        selected[program_id] = record_id
        metadata[record_id] = ("train", weight)
    return selected, metadata


def _selected_rows(
    path: Path,
    *,
    identifier: str,
    selected_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    output: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in iter_csv(path):
        record_id = row[identifier]
        if record_id in selected_ids:
            output[record_id].append(row)
    if set(output) != selected_ids:
        raise SynthesisProgramTrainingError(
            f"{path.name} omitted selected records: {sorted(selected_ids.difference(output))}"
        )
    return dict(output)


def _one_row_by_id(
    path: Path,
    *,
    identifier: str,
    selected_ids: set[str],
    predicate: Any | None = None,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in iter_csv(path):
        if predicate is not None and not predicate(row):
            continue
        record_id = row[identifier]
        if record_id not in selected_ids:
            continue
        if record_id in output:
            raise SynthesisProgramTrainingError(f"duplicate selected record: {record_id}")
        output[record_id] = row
    if set(output) != selected_ids:
        raise SynthesisProgramTrainingError(
            f"{path.name} omitted selected records: {sorted(selected_ids.difference(output))}"
        )
    return output


def _validate_semantics(
    *,
    record_id: str,
    program_id: str,
    roles: Sequence[str],
    core_positions: Sequence[str],
    depth: int,
    contract: Mapping[str, Any],
) -> None:
    if set(roles) != set(contract["roles"]):
        raise SynthesisProgramTrainingError(
            f"observed roles changed for {record_id}: {sorted(set(roles))}"
        )
    observed_core = {
        position.removeprefix(f"{program_id}:")
        for position in core_positions
        if position != "exterior"
    }
    if not observed_core.issubset(set(contract["core_positions"])):
        raise SynthesisProgramTrainingError(
            f"observed core positions changed for {record_id}: {sorted(observed_core)}"
        )
    if depth not in contract["allowed_depths"]:
        raise SynthesisProgramTrainingError(
            f"observed program depth changed for {record_id}: {depth}"
        )


def _tensorize_selected(
    *,
    selected: Mapping[str, str],
    contracts: Mapping[str, Mapping[str, Any]],
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[AtomState],
    paths: Mapping[str, Path],
) -> tuple[
    tuple[SynthesisProgramGraphRecord, ...],
    dict[str, dict[str, str]],
    dict[str, list[dict[str, str]]],
]:
    ugi_contracts = [value for value in contracts.values() if value["source"] == "ugi"]
    if len(ugi_contracts) != 1:
        raise SynthesisProgramTrainingError("cache requires exactly one qualified Ugi program")
    ugi_program = str(ugi_contracts[0]["program_id"])
    ugi_id = selected[ugi_program]
    ugi_products = _one_row_by_id(
        paths["ugi_semantic_products"],
        identifier="product_id",
        selected_ids={ugi_id},
    )
    ugi_atoms = _selected_rows(
        paths["ugi_semantic_atoms"],
        identifier="product_id",
        selected_ids={ugi_id},
    )
    aux_ids = {record_id for program, record_id in selected.items() if program != ugi_program}
    aux_products = _one_row_by_id(
        paths["multireaction_atlas"],
        identifier="record_id",
        selected_ids=aux_ids,
        predicate=lambda row: row["disposition"] == "admit_exact"
        and row["semantic_origin_status"] == "exact",
    )
    aux_atoms = _selected_rows(
        paths["multireaction_semantic_atoms"],
        identifier="record_id",
        selected_ids=aux_ids,
    )
    records: list[SynthesisProgramGraphRecord] = []
    for program_id in sorted(selected):
        record_id = selected[program_id]
        contract = contracts[program_id]
        if program_id == ugi_program:
            product = ugi_products[record_id]
            atoms = sorted(ugi_atoms[record_id], key=lambda row: int(row["product_atom_index"]))
            if [int(row["product_atom_index"]) for row in atoms] != list(range(len(atoms))):
                raise SynthesisProgramTrainingError(
                    f"Ugi atom indices are not contiguous: {record_id}"
                )
            roles = [row["origin_role"] for row in atoms]
            core_positions = [
                f"{program_id}:{row['core_position']}" if row["core_position"] else "exterior"
                for row in atoms
            ]
            fixed_indices = tuple(
                int(row["product_atom_index"])
                for row in atoms
                if row["is_ugi_core"].strip().lower() == "true"
            )
            depth = 1
            smiles = product["product_smiles"]
        else:
            product = aux_products[record_id]
            atoms = sorted(aux_atoms[record_id], key=lambda row: int(row["atom_index"]))
            if [int(row["atom_index"]) for row in atoms] != list(range(len(atoms))):
                raise SynthesisProgramTrainingError(
                    f"auxiliary atom indices are not contiguous: {record_id}"
                )
            if {row["program_id"] for row in atoms} != {program_id}:
                raise SynthesisProgramTrainingError(f"program identity changed: {record_id}")
            roles = [row["origin_role"] for row in atoms]
            core_positions = [
                f"{program_id}:{row['core_position']}" if row["core_position"] else "exterior"
                for row in atoms
            ]
            depths = {int(row["program_depth"]) for row in atoms}
            depth = int(product["step_count"])
            if depths != {depth}:
                raise SynthesisProgramTrainingError(f"program depth changed: {record_id}")
            smiles = product["canonical_product_smiles"]
            fixed_indices = (
                tuple(int(row["atom_index"]) for row in atoms if row["core_position"])
                if contract["fix_core_atoms"]
                else ()
            )
        _validate_semantics(
            record_id=record_id,
            program_id=program_id,
            roles=roles,
            core_positions=core_positions,
            depth=depth,
            contract=contract,
        )
        if bool(contract["fix_core_atoms"]) != bool(fixed_indices):
            raise SynthesisProgramTrainingError(f"fixed-core policy changed: {record_id}")
        try:
            record = tensorize_synthesis_program_product(
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
            raise SynthesisProgramTrainingError(str(error)) from error
        records.append(record)
    return tuple(records), ugi_products, ugi_atoms


def _fixed_edges_from_shared(record: SynthesisProgramGraphRecord) -> dict[tuple[int, int], int]:
    edges: dict[tuple[int, int], int] = {}
    order = record.canonical_atom_order
    for child in np.flatnonzero(record.fixed_parent_bond_mask):
        parent = int(record.graph.parents[child])
        left, right = sorted((int(order[child]), int(order[parent])))
        pair = (left, right)
        edges[pair] = int(record.graph.parent_bonds[child])
    for slot in np.flatnonzero(record.fixed_closure_bond_mask):
        left = int(record.graph.closure_left[slot])
        right = int(record.graph.closure_right[slot])
        first, second = sorted((int(order[left]), int(order[right])))
        pair = (first, second)
        edges[pair] = int(record.graph.closure_bonds[slot])
    return edges


def _ugi_regression(
    record: SynthesisProgramGraphRecord,
    product: Mapping[str, str],
    atom_rows: Sequence[Mapping[str, str]],
    atom_vocabulary: Sequence[AtomState],
) -> dict[str, Any]:
    atom_to_index = {state: index for index, state in enumerate(atom_vocabulary)}
    support = tensorize_ugi_l1_support_record(
        product,
        atom_rows,
        atom_to_index,
        preserve_aromaticity=True,
    )
    condition = project_chemistry_topology_condition(
        support,
        core_schema_from_record(support),
    )
    shared_atoms = {
        int(record.canonical_atom_order[index]): int(record.graph.node_states[index])
        for index in np.flatnonzero(record.fixed_atom_mask)
    }
    established_atoms = {
        int(support.support_full_atom_order[index]): int(condition.fixed_atom_states[index])
        for index in np.flatnonzero(condition.fixed_atom_mask)
    }
    established_edges: dict[tuple[int, int], int] = {}
    parents = condition.parents
    order = support.support_full_atom_order
    for child in np.flatnonzero(condition.fixed_parent_bond_mask):
        left, right = sorted((int(order[child]), int(order[int(parents[child])])))
        pair = (left, right)
        established_edges[pair] = int(condition.fixed_parent_bond_states[child])
    for slot in np.flatnonzero(condition.fixed_closure_bond_mask):
        left = int(condition.closure_left[slot])
        right = int(condition.closure_right[slot])
        first, second = sorted((int(order[left]), int(order[right])))
        pair = (first, second)
        established_edges[pair] = int(condition.fixed_closure_bond_states[slot])
    shared_edges = _fixed_edges_from_shared(record)
    return {
        "product_id": record.graph.structure_id,
        "fixed_atom_count": len(shared_atoms),
        "fixed_edge_count": len(shared_edges),
        "atom_states_exact": shared_atoms == established_atoms,
        "bond_states_and_endpoints_exact": shared_edges == established_edges,
        "established_fixed_atom_count": len(established_atoms),
        "established_fixed_edge_count": len(established_edges),
    }


def _record_payload(record: SynthesisProgramGraphRecord) -> dict[str, Any]:
    return {
        "structure_id": record.graph.structure_id,
        "canonical_smiles": record.graph.canonical_smiles,
        "node_states": record.graph.node_states.tolist(),
        "parents": record.graph.parents.tolist(),
        "parent_bonds": record.graph.parent_bonds.tolist(),
        "closure_left": record.graph.closure_left.tolist(),
        "closure_right": record.graph.closure_right.tolist(),
        "closure_bonds": record.graph.closure_bonds.tolist(),
        "canonical_atom_order": record.canonical_atom_order.tolist(),
        "program_id": record.program_id,
        "program_state": record.program_state,
        "program_depth": record.program_depth,
        "role_states": record.role_states.tolist(),
        "core_position_states": record.core_position_states.tolist(),
        "component_blocks": [
            {
                "role": block.role,
                "role_state": block.role_state,
                "start": block.start,
                "stop": block.stop,
            }
            for block in record.component_blocks
        ],
        "fixed_atom_mask": record.fixed_atom_mask.tolist(),
        "fixed_parent_bond_mask": record.fixed_parent_bond_mask.tolist(),
        "fixed_closure_bond_mask": record.fixed_closure_bond_mask.tolist(),
    }


def _record_from_payload(raw: Mapping[str, Any]) -> SynthesisProgramGraphRecord:
    nodes = np.asarray(raw["node_states"], dtype=np.int64)
    parents = np.asarray(raw["parents"], dtype=np.int64)
    parent_bonds = np.asarray(raw["parent_bonds"], dtype=np.int64)
    closure_left = np.asarray(raw["closure_left"], dtype=np.int64)
    closure_right = np.asarray(raw["closure_right"], dtype=np.int64)
    closure_bonds = np.asarray(raw["closure_bonds"], dtype=np.int64)
    edges = np.zeros((nodes.size, nodes.size), dtype=np.int64)
    for child in range(1, nodes.size):
        parent = int(parents[child])
        dense = INDEX_TO_DENSE_BOND[int(parent_bonds[child])]
        edges[child, parent] = edges[parent, child] = dense
    for left, right, bond in zip(closure_left, closure_right, closure_bonds, strict=True):
        dense = INDEX_TO_DENSE_BOND[int(bond)]
        edges[int(left), int(right)] = edges[int(right), int(left)] = dense
    graph = SparseGraphRecord(
        structure_id=str(raw["structure_id"]),
        canonical_smiles=str(raw["canonical_smiles"]),
        node_states=nodes,
        parents=parents,
        parent_bonds=parent_bonds,
        closure_left=closure_left,
        closure_right=closure_right,
        closure_bonds=closure_bonds,
        edges=edges,
    )
    return SynthesisProgramGraphRecord(
        graph=graph,
        canonical_atom_order=np.asarray(raw["canonical_atom_order"], dtype=np.int64),
        program_id=str(raw["program_id"]),
        program_state=int(raw["program_state"]),
        program_depth=int(raw["program_depth"]),
        role_states=np.asarray(raw["role_states"], dtype=np.int64),
        core_position_states=np.asarray(raw["core_position_states"], dtype=np.int64),
        component_blocks=tuple(
            SynthesisProgramComponentBlock(
                role=str(block["role"]),
                role_state=int(block["role_state"]),
                start=int(block["start"]),
                stop=int(block["stop"]),
            )
            for block in raw["component_blocks"]
        ),
        fixed_atom_mask=np.asarray(raw["fixed_atom_mask"], dtype=np.bool_),
        fixed_parent_bond_mask=np.asarray(raw["fixed_parent_bond_mask"], dtype=np.bool_),
        fixed_closure_bond_mask=np.asarray(raw["fixed_closure_bond_mask"], dtype=np.bool_),
    )


def build_bounded_synthesis_program_training_cache(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Build and qualify one deterministic training-fold record per admitted program."""

    config = read_json_object(
        config_path,
        error=SynthesisProgramTrainingError,
        label="shared synthesis-program cache config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SynthesisProgramTrainingError("unsupported shared cache config schema")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict):
        raise SynthesisProgramTrainingError("shared cache config has no pinned inputs")
    paths = {
        label: resolve_pin(pin, repo, label=label) for label, pin in sorted(raw_inputs.items())
    }
    required = {
        "representation_config",
        "atom_vocabulary",
        "ugi_assignments",
        "ugi_semantic_products",
        "ugi_semantic_atoms",
        "multireaction_atlas",
        "multireaction_semantic_atoms",
        "multireaction_splits",
    }
    if set(paths) != required:
        raise SynthesisProgramTrainingError(
            f"shared cache inputs changed: {sorted(set(paths).symmetric_difference(required))}"
        )
    representation = read_json_object(
        paths["representation_config"],
        error=SynthesisProgramTrainingError,
        label="shared representation config",
    )
    if representation.get("schema_version") != (
        "forge.shared_synthesis_program_representation_config.v1"
    ):
        raise SynthesisProgramTrainingError("shared representation config changed schema")
    for label in required.difference({"representation_config", "multireaction_splits"}):
        if representation["inputs"].get(label) != raw_inputs[label]:
            raise SynthesisProgramTrainingError(
                f"cache and qualified representation disagree on {label}"
            )
    try:
        contracts = synthesis_program_contracts(representation)
    except SynthesisProgramRepresentationError as error:
        raise SynthesisProgramTrainingError(str(error)) from error
    vocabulary = synthesis_program_vocabulary(contracts)
    atom_vocabulary = load_atom_vocabulary(paths["atom_vocabulary"])
    ugi_programs = [key for key, value in contracts.items() if value["source"] == "ugi"]
    if len(ugi_programs) != 1:
        raise SynthesisProgramTrainingError("cache requires exactly one Ugi program")
    seed = int(config["seed"])
    selected, metadata = _selected_training_ids(
        assignments_path=paths["ugi_assignments"],
        splits_path=paths["multireaction_splits"],
        program_ids=tuple(sorted(contracts)),
        ugi_program_id=ugi_programs[0],
        seed=seed,
    )
    records, ugi_products, ugi_atoms = _tensorize_selected(
        selected=selected,
        contracts=contracts,
        vocabulary=vocabulary,
        atom_vocabulary=atom_vocabulary,
        paths=paths,
    )
    ugi_record = next(record for record in records if record.program_id == ugi_programs[0])
    regression = _ugi_regression(
        ugi_record,
        ugi_products[ugi_record.graph.structure_id],
        ugi_atoms[ugi_record.graph.structure_id],
        atom_vocabulary,
    )
    fixed_core_policy_exact = all(
        np.array_equal(
            record.fixed_atom_mask,
            (
                record.core_position_states > 1
                if contracts[record.program_id]["fix_core_atoms"]
                else np.zeros(record.node_count, dtype=np.bool_)
            ),
        )
        for record in records
    )
    if not fixed_core_policy_exact:
        raise SynthesisProgramTrainingError("record masks disagree with the fixed-core contract")
    program_count = len(records)
    sampling_weights = np.full(program_count, 1.0 / program_count, dtype=np.float64)
    source_weights = np.asarray(
        [metadata[record.graph.structure_id][1] for record in records], dtype=np.float64
    )
    source_folds = tuple(metadata[record.graph.structure_id][0] for record in records)
    payload = {
        "schema_version": CACHE_SCHEMA,
        "seed": seed,
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "program_vocabulary": {
            "program_states": list(vocabulary.program_states),
            "role_states": list(vocabulary.role_states),
            "core_position_states": list(vocabulary.core_position_states),
            "maximum_steps": vocabulary.maximum_steps,
        },
        "atom_vocabulary": [
            {
                "symbol": state.symbol,
                "formal_charge": state.formal_charge,
                "aromatic": state.aromatic,
                "explicit_hydrogens": state.explicit_hydrogens,
            }
            for state in atom_vocabulary
        ],
        "records": [_record_payload(record) for record in records],
        "sampling_weights": sampling_weights.tolist(),
        "source_folds": list(source_folds),
        "source_weights": source_weights.tolist(),
        "model_state_excludes": [
            "component_identifiers",
            "component_smiles",
            "component_fingerprints",
            "fragment_tokens",
            "biological_labels",
        ],
    }
    write_json(cache_path, payload)
    gates = {
        "one_training_record_per_program": len(records) == len(contracts) == 3
        and set(record.program_id for record in records) == set(contracts),
        "equal_program_sampling_mass": np.allclose(
            sampling_weights, np.full(program_count, 1.0 / program_count)
        ),
        "ugi_five_fixed_atoms": int(np.count_nonzero(ugi_record.fixed_atom_mask)) == 5,
        "ugi_fixed_atoms_match_existing_adapter": bool(regression["atom_states_exact"]),
        "ugi_fixed_edges_match_existing_adapter": bool(
            regression["bond_states_and_endpoints_exact"]
        ),
        "fixed_core_policy_exact": fixed_core_policy_exact,
        "aromatic_state_space_preserved": any(state.aromatic for state in atom_vocabulary),
        "full_declared_size_support_preserved": max(record.node_count for record in records)
        <= int(config["model_support"]["maximum_heavy_atoms"]),
        "full_declared_closure_support_preserved": max(
            record.graph.closure_count for record in records
        )
        <= int(config["model_support"]["maximum_closures"]),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "seed": seed,
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "cache": artifact_record(cache_path),
        "selected_records": {
            record.program_id: {
                "record_id": record.graph.structure_id,
                "fold": fold,
                "source_weight": float(weight),
                "training_weight": float(training_weight),
                "heavy_atoms": record.node_count,
                "closures": record.graph.closure_count,
                "fixed_atoms": int(np.count_nonzero(record.fixed_atom_mask)),
                "fixed_parent_edges": int(np.count_nonzero(record.fixed_parent_bond_mask)),
                "fixed_closure_edges": int(np.count_nonzero(record.fixed_closure_bond_mask)),
            }
            for record, fold, weight, training_weight in zip(
                records,
                source_folds,
                source_weights,
                sampling_weights,
                strict=True,
            )
        },
        "ugi_regression": regression,
        "gates": gates,
        "nonclaims": [
            "The three-record cache is a bounded integration and overfit set, not generalization evidence.",
            "Exact adapter agreement is not route certification or synthesis-success probability.",
            "This cache does not authorize production multi-reaction training.",
        ],
    }
    write_json(result_path, result)
    if result["status"] != "pass":
        raise SynthesisProgramTrainingError(f"shared cache qualification failed: {gates}")
    return result


def load_synthesis_program_training_cache(path: Path) -> SynthesisProgramTrainingCache:
    """Load a hash-verified cache after the experiment runtime authenticates its artifact."""

    raw = read_json_object(
        path,
        error=SynthesisProgramTrainingError,
        label="shared synthesis-program training cache",
    )
    if raw.get("schema_version") != CACHE_SCHEMA:
        raise SynthesisProgramTrainingError("unsupported shared training cache schema")
    vocabulary_raw = raw["program_vocabulary"]
    vocabulary = ReactionProgramVocabulary(
        program_states=tuple(str(value) for value in vocabulary_raw["program_states"]),
        role_states=tuple(str(value) for value in vocabulary_raw["role_states"]),
        core_position_states=tuple(str(value) for value in vocabulary_raw["core_position_states"]),
        maximum_steps=int(vocabulary_raw["maximum_steps"]),
    )
    atom_vocabulary = tuple(
        AtomState(
            symbol=str(row["symbol"]),
            formal_charge=int(row["formal_charge"]),
            aromatic=bool(row["aromatic"]),
            explicit_hydrogens=int(row["explicit_hydrogens"]),
        )
        for row in raw["atom_vocabulary"]
    )
    records = tuple(_record_from_payload(record) for record in raw["records"])
    if set(record.program_id for record in records) != set(vocabulary.program_states[1:]):
        raise SynthesisProgramTrainingError("cache records do not cover the program vocabulary")
    return SynthesisProgramTrainingCache(
        vocabulary=vocabulary,
        atom_vocabulary=atom_vocabulary,
        records=records,
        sampling_weights=np.asarray(raw["sampling_weights"], dtype=np.float64),
        source_folds=tuple(str(value) for value in raw["source_folds"]),
        source_weights=np.asarray(raw["source_weights"], dtype=np.float64),
        inputs=raw["inputs"],
    )


__all__ = [
    "CACHE_SCHEMA",
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SynthesisProgramTrainingCache",
    "SynthesisProgramTrainingError",
    "build_bounded_synthesis_program_training_cache",
    "load_synthesis_program_training_cache",
]
