"""Qualify one shared semantic graph representation across admitted reaction programs."""

from __future__ import annotations

import hashlib
import itertools
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from forge.core.io import iter_csv, read_json_object, stable_json, write_json
from forge.corpus.reaction_program_records import admits_reaction_program_structure
from forge.model.defog_feasibility import sha256_file
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    SynthesisProgramGraphRecord,
    tensorize_synthesis_program_product,
)
from forge.model.vocabulary import load_atom_vocabulary


class SynthesisProgramRepresentationError(ValueError):
    """Pinned semantic corpora disagree with the shared representation contract."""


@dataclass
class _ProgramCensus:
    expected_records: int
    records_attempted: int = 0
    records_represented: int = 0
    atom_rows: int = 0
    core_atom_rows: int = 0
    fixed_atom_rows: int = 0
    fixed_policy_failures: int = 0
    aromatic_products: int = 0
    aromatic_atom_rows: int = 0
    aromatic_bond_rows: int = 0
    minimum_heavy_atoms: int | None = None
    maximum_heavy_atoms: int = 0
    maximum_closures: int = 0
    maximum_origin_components: int = 0
    depth_distribution: Counter[int] = field(default_factory=Counter)
    core_position_distribution: Counter[str] = field(default_factory=Counter)
    role_component_distribution: Counter[str] = field(default_factory=Counter)
    errors: list[dict[str, str]] = field(default_factory=list)
    digest: Any = field(default_factory=hashlib.sha256)

    def observe(
        self,
        record: SynthesisProgramGraphRecord,
        atom_vocabulary: Sequence[Any],
        program_vocabulary: ReactionProgramVocabulary,
        *,
        fix_core_atoms: bool,
    ) -> None:
        self.records_represented += 1
        self.atom_rows += record.node_count
        self.core_atom_rows += int(np.count_nonzero(record.core_position_states > 1))
        self.fixed_atom_rows += int(np.count_nonzero(record.fixed_atom_mask))
        expected_fixed = (
            record.core_position_states > 1
            if fix_core_atoms
            else np.zeros(record.node_count, dtype=np.bool_)
        )
        self.fixed_policy_failures += int(
            not np.array_equal(record.fixed_atom_mask, expected_fixed)
        )
        self.minimum_heavy_atoms = (
            record.node_count
            if self.minimum_heavy_atoms is None
            else min(self.minimum_heavy_atoms, record.node_count)
        )
        self.maximum_heavy_atoms = max(self.maximum_heavy_atoms, record.node_count)
        self.maximum_closures = max(self.maximum_closures, record.graph.closure_count)
        self.maximum_origin_components = max(self.maximum_origin_components, record.component_count)
        self.depth_distribution[record.program_depth] += 1
        self.core_position_distribution.update(
            program_vocabulary.core_position_states[int(state)]
            for state in record.core_position_states
            if int(state) > 1
        )
        role_counts = Counter(block.role for block in record.component_blocks)
        self.role_component_distribution.update(
            {f"{role}:{count}": 1 for role, count in sorted(role_counts.items())}
        )
        aromatic_states = np.asarray(
            [bool(atom_vocabulary[int(state)].aromatic) for state in record.graph.node_states],
            dtype=np.bool_,
        )
        aromatic_atoms = int(np.count_nonzero(aromatic_states))
        aromatic_bonds = int(np.count_nonzero(record.graph.parent_bonds[1:] == 3)) + int(
            np.count_nonzero(record.graph.closure_bonds == 3)
        )
        self.aromatic_atom_rows += aromatic_atoms
        self.aromatic_bond_rows += aromatic_bonds
        self.aromatic_products += int(aromatic_atoms > 0)
        _update_representation_digest(self.digest, record)

    def result(self) -> dict[str, Any]:
        return {
            "expected_records": self.expected_records,
            "records_attempted": self.records_attempted,
            "records_represented": self.records_represented,
            "atom_rows": self.atom_rows,
            "core_atom_rows": self.core_atom_rows,
            "fixed_atom_rows": self.fixed_atom_rows,
            "fixed_policy_failures": self.fixed_policy_failures,
            "aromatic_products": self.aromatic_products,
            "aromatic_atom_rows": self.aromatic_atom_rows,
            "aromatic_bond_rows": self.aromatic_bond_rows,
            "minimum_heavy_atoms": self.minimum_heavy_atoms,
            "maximum_heavy_atoms": self.maximum_heavy_atoms,
            "maximum_closures": self.maximum_closures,
            "maximum_origin_components": self.maximum_origin_components,
            "depth_distribution": {
                str(key): value for key, value in sorted(self.depth_distribution.items())
            },
            "core_position_distribution": dict(sorted(self.core_position_distribution.items())),
            "role_component_distribution": dict(sorted(self.role_component_distribution.items())),
            "representation_sha256": self.digest.hexdigest(),
            "errors": self.errors,
        }


def _resolve_inputs(
    config: Mapping[str, Any],
    repo: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, dict) or not raw_inputs:
        raise SynthesisProgramRepresentationError("representation config has no pinned inputs")
    paths: dict[str, Path] = {}
    receipts: dict[str, dict[str, Any]] = {}
    for label, raw in sorted(raw_inputs.items()):
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
            raise SynthesisProgramRepresentationError(f"invalid input pin: {label}")
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = repo / path
        if not path.is_file():
            raise SynthesisProgramRepresentationError(f"missing input {label}: {path}")
        actual = sha256_file(path)
        expected = str(raw["sha256"])
        if actual != expected:
            raise SynthesisProgramRepresentationError(
                f"input SHA-256 changed for {label}: expected {expected}, observed {actual}"
            )
        paths[label] = path
        receipts[label] = {
            "path": str(raw["path"]),
            "sha256": actual,
            "bytes": path.stat().st_size,
        }
    return paths, receipts


def _validate_corpus_result(
    *,
    label: str,
    path: Path,
    paths: Mapping[str, Path],
) -> None:
    """Validate either a source corpus or its admitted mixed-repeat expansion.

    The representation census predates the heterogeneous repeated-component expansion.  Keeping
    the input label stable preserves the downstream cache contract, while the schema check below
    prevents an arbitrary ``status`` string from being treated as qualified data.  Expansion
    receipts must also authenticate the exact atlas and semantic-atom files being censused.
    """

    result = read_json_object(
        path,
        error=SynthesisProgramRepresentationError,
        label=label,
    )
    if label == "ugi_corpus_result":
        if result.get("status") != "pass":
            raise SynthesisProgramRepresentationError(f"source corpus is not passed: {label}")
        return
    if result.get("schema_version") == "forge.multireaction_lnpdb_result.v1":
        if result.get("status") != "pass":
            raise SynthesisProgramRepresentationError(f"source corpus is not passed: {label}")
        return
    if result.get("schema_version") != "forge.multireaction_mixed_expansion_result.v1":
        raise SynthesisProgramRepresentationError(
            "unsupported multi-reaction corpus result schema"
        )
    if result.get("status") != "complete_bl_lx_mixed_repeat_expansion":
        raise SynthesisProgramRepresentationError(
            "mixed-repeat expansion is not complete"
        )
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, dict):
        raise SynthesisProgramRepresentationError(
            "mixed-repeat expansion has no artifact receipts"
        )
    for artifact_label, input_label in (
        ("atlas", "multireaction_atlas"),
        ("semantic_atoms", "multireaction_semantic_atoms"),
    ):
        receipt = artifacts.get(artifact_label)
        if (
            not isinstance(receipt, dict)
            or receipt.get("sha256") != sha256_file(paths[input_label])
        ):
            raise SynthesisProgramRepresentationError(
                f"mixed-repeat expansion does not authenticate {input_label}"
            )


def synthesis_program_contracts(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_programs = config.get("programs")
    if not isinstance(raw_programs, list) or not raw_programs:
        raise SynthesisProgramRepresentationError("representation config has no programs")
    contracts: dict[str, dict[str, Any]] = {}
    for raw in raw_programs:
        if not isinstance(raw, dict):
            raise SynthesisProgramRepresentationError("program contract must be an object")
        program_id = str(raw.get("program_id", ""))
        if not program_id or program_id in contracts:
            raise SynthesisProgramRepresentationError("program identifiers must be unique")
        roles = tuple(str(value) for value in raw.get("roles", ()))
        core_positions = tuple(str(value) for value in raw.get("core_positions", ()))
        depths = tuple(int(value) for value in raw.get("allowed_depths", ()))
        expected_records = int(raw.get("expected_records", 0))
        source = str(raw.get("source", ""))
        if (
            not roles
            or len(set(roles)) != len(roles)
            or not core_positions
            or len(set(core_positions)) != len(core_positions)
            or not depths
            or min(depths) < 1
            or expected_records < 1
            or source not in {"ugi", "multireaction"}
        ):
            raise SynthesisProgramRepresentationError(f"invalid program contract: {program_id}")
        contracts[program_id] = {
            "program_id": program_id,
            "roles": roles,
            "core_positions": core_positions,
            "allowed_depths": depths,
            "expected_records": expected_records,
            "source": source,
            "fix_core_atoms": bool(raw.get("fix_core_atoms", False)),
        }
    return contracts


def synthesis_program_vocabulary(
    contracts: Mapping[str, Mapping[str, Any]],
) -> ReactionProgramVocabulary:
    """Build the one shared categorical vocabulary from qualified program contracts."""

    return ReactionProgramVocabulary.from_semantics(
        program_ids=tuple(sorted(contracts)),
        roles=tuple(sorted({str(role) for value in contracts.values() for role in value["roles"]})),
        core_positions=tuple(
            sorted(
                f"{program_id}:{position}"
                for program_id, value in contracts.items()
                for position in value["core_positions"]
            )
        ),
        maximum_steps=max(
            max(int(depth) for depth in value["allowed_depths"]) for value in contracts.values()
        ),
    )


def _group_semantic_rows(
    rows: Iterator[dict[str, str]],
    *,
    identifier: str,
    atom_index: str,
) -> Iterator[tuple[str, list[dict[str, str]]]]:
    seen: set[str] = set()
    for record_id, group in itertools.groupby(rows, key=lambda row: row[identifier]):
        if record_id in seen:
            raise SynthesisProgramRepresentationError(
                f"semantic rows are not contiguous for {record_id}"
            )
        seen.add(record_id)
        values = list(group)
        indices = [int(row[atom_index]) for row in values]
        if indices != list(range(len(values))):
            raise SynthesisProgramRepresentationError(
                f"semantic atom indices are not contiguous for {record_id}"
            )
        yield record_id, values


def _update_bytes(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "little"))
    digest.update(value)


def _update_representation_digest(digest: Any, record: SynthesisProgramGraphRecord) -> None:
    for value in (record.graph.structure_id, record.program_id, str(record.program_depth)):
        _update_bytes(digest, value.encode())
    for values in (
        record.canonical_atom_order,
        record.graph.node_states,
        record.graph.parents,
        record.graph.parent_bonds,
        record.graph.closure_left,
        record.graph.closure_right,
        record.graph.closure_bonds,
        record.role_states,
        record.core_position_states,
        record.fixed_atom_mask.astype(np.uint8),
    ):
        _update_bytes(digest, np.asarray(values, dtype="<i8").tobytes())
    _update_bytes(
        digest,
        stable_json(
            [
                {
                    "role": block.role,
                    "role_state": block.role_state,
                    "start": block.start,
                    "stop": block.stop,
                }
                for block in record.component_blocks
            ]
        ).encode(),
    )


def _validate_contract_semantics(
    *,
    program_id: str,
    roles: Sequence[str],
    core_positions: Sequence[str],
    depth: int,
    contract: Mapping[str, Any],
    record_id: str,
) -> None:
    if set(roles) != set(contract["roles"]):
        raise SynthesisProgramRepresentationError(
            f"observed roles changed for {record_id}: {sorted(set(roles))}"
        )
    observed_core = {
        position.removeprefix(f"{program_id}:")
        for position in core_positions
        if position != "exterior"
    }
    if not observed_core.issubset(set(contract["core_positions"])):
        raise SynthesisProgramRepresentationError(
            f"observed core positions changed for {record_id}: {sorted(observed_core)}"
        )
    if depth not in contract["allowed_depths"]:
        raise SynthesisProgramRepresentationError(
            f"observed program depth changed for {record_id}: {depth}"
        )


def _try_tensorize(
    *,
    census: _ProgramCensus,
    atom_vocabulary: Sequence[Any],
    kwargs: dict[str, Any],
    fix_core_atoms: bool,
) -> None:
    census.records_attempted += 1
    try:
        program_vocabulary = kwargs.pop("vocabulary")
        record = tensorize_synthesis_program_product(
            vocabulary=program_vocabulary,
            atom_vocabulary=atom_vocabulary,
            **kwargs,
        )
    except SynthesisProgramGraphError as error:
        if len(census.errors) < 20:
            census.errors.append({"record_id": str(kwargs["record_id"]), "reason": str(error)})
        return
    census.observe(
        record,
        atom_vocabulary,
        program_vocabulary,
        fix_core_atoms=fix_core_atoms,
    )


def _census_ugi(
    *,
    paths: Mapping[str, Path],
    contract: Mapping[str, Any],
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[Any],
    census: _ProgramCensus,
) -> None:
    assignments = iter_csv(paths["ugi_assignments"])
    products = iter_csv(paths["ugi_semantic_products"])
    atom_groups = _group_semantic_rows(
        iter_csv(paths["ugi_semantic_atoms"]),
        identifier="product_id",
        atom_index="product_atom_index",
    )
    missing = object()
    for assignment, product, atom_group in itertools.zip_longest(
        assignments, products, atom_groups, fillvalue=missing
    ):
        if missing in (assignment, product, atom_group):
            raise SynthesisProgramRepresentationError(
                "Ugi assignments, products, and semantic atom groups have different lengths"
            )
        assert isinstance(assignment, dict) and isinstance(product, dict)
        assert isinstance(atom_group, tuple)
        product_id, atoms = atom_group
        if assignment["product_id"] != product_id or product["product_id"] != product_id:
            raise SynthesisProgramRepresentationError(
                f"Ugi semantic ledgers changed order or identity at {product_id}"
            )
        if assignment["canonical_product_smiles"] != product["product_smiles"]:
            raise SynthesisProgramRepresentationError(f"Ugi product SMILES changed: {product_id}")
        if len(atoms) != int(product["atom_rows"]):
            raise SynthesisProgramRepresentationError(
                f"Ugi semantic atom count changed: {product_id}"
            )
        roles = [row["origin_role"] for row in atoms]
        core_positions = [
            (
                f"{contract['program_id']}:{row['core_position']}"
                if row["core_position"]
                else "exterior"
            )
            for row in atoms
        ]
        _validate_contract_semantics(
            program_id=str(contract["program_id"]),
            roles=roles,
            core_positions=core_positions,
            depth=1,
            contract=contract,
            record_id=product_id,
        )
        fixed_indices = tuple(
            int(row["product_atom_index"])
            for row in atoms
            if row["is_ugi_core"].strip().lower() == "true"
        )
        if contract["fix_core_atoms"] and len(fixed_indices) != 5:
            raise SynthesisProgramRepresentationError(
                f"Ugi fixed core no longer contains five atoms: {product_id}"
            )
        _try_tensorize(
            census=census,
            atom_vocabulary=atom_vocabulary,
            fix_core_atoms=bool(contract["fix_core_atoms"]),
            kwargs={
                "record_id": product_id,
                "program_id": contract["program_id"],
                "canonical_product_smiles": product["product_smiles"],
                "atom_roles": roles,
                "atom_core_positions": core_positions,
                "program_depth": 1,
                "vocabulary": vocabulary,
                "fixed_atom_indices": fixed_indices if contract["fix_core_atoms"] else (),
            },
        )


def _census_multireaction(
    *,
    paths: Mapping[str, Path],
    contracts: Mapping[str, Mapping[str, Any]],
    vocabulary: ReactionProgramVocabulary,
    atom_vocabulary: Sequence[Any],
    censuses: Mapping[str, _ProgramCensus],
) -> None:
    atlas = (
        row
        for row in iter_csv(paths["multireaction_atlas"])
        if admits_reaction_program_structure(row)
    )
    atom_groups = _group_semantic_rows(
        iter_csv(paths["multireaction_semantic_atoms"]),
        identifier="record_id",
        atom_index="atom_index",
    )
    missing = object()
    for product, atom_group in itertools.zip_longest(atlas, atom_groups, fillvalue=missing):
        if missing in (product, atom_group):
            raise SynthesisProgramRepresentationError(
                "admitted multi-reaction atlas and semantic atom groups have different lengths"
            )
        assert isinstance(product, dict) and isinstance(atom_group, tuple)
        record_id, atoms = atom_group
        program_id = product["program_id"]
        if product["record_id"] != record_id:
            raise SynthesisProgramRepresentationError(
                f"multi-reaction semantic ledgers changed order or identity at {record_id}"
            )
        if program_id not in contracts or contracts[program_id]["source"] != "multireaction":
            raise SynthesisProgramRepresentationError(
                f"unqualified multi-reaction program: {program_id}"
            )
        if {row["program_id"] for row in atoms} != {program_id}:
            raise SynthesisProgramRepresentationError(f"program identity changed: {record_id}")
        depths = {int(row["program_depth"]) for row in atoms}
        depth = int(product["step_count"])
        if depths != {depth}:
            raise SynthesisProgramRepresentationError(f"program depth changed: {record_id}")
        contract = contracts[program_id]
        roles = [row["origin_role"] for row in atoms]
        core_positions = [
            f"{program_id}:{row['core_position']}" if row["core_position"] else "exterior"
            for row in atoms
        ]
        _validate_contract_semantics(
            program_id=program_id,
            roles=roles,
            core_positions=core_positions,
            depth=depth,
            contract=contract,
            record_id=record_id,
        )
        _try_tensorize(
            census=censuses[program_id],
            atom_vocabulary=atom_vocabulary,
            fix_core_atoms=bool(contract["fix_core_atoms"]),
            kwargs={
                "record_id": record_id,
                "program_id": program_id,
                "canonical_product_smiles": product["canonical_product_smiles"],
                "atom_roles": roles,
                "atom_core_positions": core_positions,
                "program_depth": depth,
                "vocabulary": vocabulary,
                "fixed_atom_indices": (
                    tuple(int(row["atom_index"]) for row in atoms if row["core_position"])
                    if contract["fix_core_atoms"]
                    else ()
                ),
            },
        )


def qualify_shared_synthesis_program_representation(
    config_path: Path,
    repo: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run an exact full-corpus census and write one hash-attributed qualification result."""

    config = read_json_object(
        config_path,
        error=SynthesisProgramRepresentationError,
        label="shared synthesis-program representation config",
    )
    if config.get("schema_version") != "forge.shared_synthesis_program_representation_config.v1":
        raise SynthesisProgramRepresentationError("unsupported representation config schema")
    paths, input_receipts = _resolve_inputs(config, repo)
    required_inputs = {
        "atom_vocabulary",
        "ugi_assignments",
        "ugi_corpus_result",
        "ugi_semantic_products",
        "ugi_semantic_atoms",
        "multireaction_corpus_result",
        "multireaction_atlas",
        "multireaction_semantic_atoms",
    }
    if set(paths) != required_inputs:
        raise SynthesisProgramRepresentationError(
            f"representation inputs changed: {sorted(set(paths).symmetric_difference(required_inputs))}"
        )
    for label in ("ugi_corpus_result", "multireaction_corpus_result"):
        _validate_corpus_result(label=label, path=paths[label], paths=paths)

    contracts = synthesis_program_contracts(config)
    ugi_contracts = [value for value in contracts.values() if value["source"] == "ugi"]
    if len(ugi_contracts) != 1:
        raise SynthesisProgramRepresentationError("exactly one Ugi program is required")
    vocabulary = synthesis_program_vocabulary(contracts)
    atom_vocabulary = load_atom_vocabulary(paths["atom_vocabulary"])
    censuses = {
        program_id: _ProgramCensus(expected_records=int(value["expected_records"]))
        for program_id, value in contracts.items()
    }
    _census_ugi(
        paths=paths,
        contract=ugi_contracts[0],
        vocabulary=vocabulary,
        atom_vocabulary=atom_vocabulary,
        census=censuses[str(ugi_contracts[0]["program_id"])],
    )
    _census_multireaction(
        paths=paths,
        contracts=contracts,
        vocabulary=vocabulary,
        atom_vocabulary=atom_vocabulary,
        censuses=censuses,
    )

    program_results = {
        program_id: census.result() for program_id, census in sorted(censuses.items())
    }
    limits = config.get("support_bounds")
    if not isinstance(limits, dict):
        raise SynthesisProgramRepresentationError("representation support bounds are missing")
    maximum_heavy_atoms = max(
        int(value["maximum_heavy_atoms"]) for value in program_results.values()
    )
    maximum_closures = max(int(value["maximum_closures"]) for value in program_results.values())
    all_records_represented = all(
        value["records_attempted"] == value["records_represented"] == value["expected_records"]
        for value in program_results.values()
    )
    all_core_positions_observed = all(
        set(value["core_position_distribution"])
        == {f"{program_id}:{position}" for position in contracts[program_id]["core_positions"]}
        for program_id, value in program_results.items()
    )
    record_fields = {value.name for value in fields(SynthesisProgramGraphRecord)}
    forbidden_identity_fields = {
        "component_id",
        "component_ids",
        "component_smiles",
        "component_fingerprints",
        "fragment_tokens",
    }
    gates = {
        "all_declared_records_attempted_and_represented": all_records_represented,
        "all_declared_core_positions_observed": all_core_positions_observed,
        "aromatic_atoms_and_bonds_preserved": any(
            value["aromatic_atom_rows"] > 0 and value["aromatic_bond_rows"] > 0
            for value in program_results.values()
        ),
        "fixed_ugi_core_is_explicit": (
            program_results[str(ugi_contracts[0]["program_id"])]["fixed_atom_rows"]
            == 5 * int(ugi_contracts[0]["expected_records"])
        ),
        "fixed_core_policy_exact": all(
            int(value["fixed_policy_failures"]) == 0 for value in program_results.values()
        ),
        "full_heavy_atom_support_preserved": maximum_heavy_atoms
        <= int(limits.get("maximum_heavy_atoms", 0)),
        "full_closure_support_preserved": maximum_closures
        <= int(limits.get("maximum_closures", -1)),
        "no_component_identity_fields": not record_fields.intersection(forbidden_identity_fields),
    }
    global_digest = hashlib.sha256()
    for program_id, value in program_results.items():
        _update_bytes(global_digest, program_id.encode())
        _update_bytes(global_digest, str(value["representation_sha256"]).encode())
    result: dict[str, Any] = {
        "schema_version": "forge.shared_synthesis_program_representation_result.v1",
        "status": "pass" if all(gates.values()) else "fail",
        "seed": int(config["seed"]),
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": input_receipts,
        "vocabulary": {
            "atom_states": len(atom_vocabulary),
            "program_states": list(vocabulary.program_states),
            "role_states": list(vocabulary.role_states),
            "core_position_states": list(vocabulary.core_position_states),
            "maximum_steps": vocabulary.maximum_steps,
        },
        "support": {
            "declared_maximum_heavy_atoms": int(limits["maximum_heavy_atoms"]),
            "observed_maximum_heavy_atoms": maximum_heavy_atoms,
            "declared_maximum_closures": int(limits["maximum_closures"]),
            "observed_maximum_closures": maximum_closures,
        },
        "programs": program_results,
        "summary": {
            "programs": len(program_results),
            "records_attempted": sum(
                int(value["records_attempted"]) for value in program_results.values()
            ),
            "records_represented": sum(
                int(value["records_represented"]) for value in program_results.values()
            ),
            "atom_rows": sum(int(value["atom_rows"]) for value in program_results.values()),
            "representation_sha256": global_digest.hexdigest(),
            "component_identifiers_used": False,
            "fragment_tokens_used": False,
            "biological_labels_used": False,
        },
        "gates": gates,
        "nonclaims": list(config.get("nonclaims", ())),
    }
    write_json(output_path, result)
    return result


__all__ = [
    "SynthesisProgramRepresentationError",
    "qualify_shared_synthesis_program_representation",
    "synthesis_program_contracts",
    "synthesis_program_vocabulary",
]
