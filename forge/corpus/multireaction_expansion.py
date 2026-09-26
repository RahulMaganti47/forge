"""Expand BL/LX reaction-program supervision from source-linked LNPDB components.

The source-executed BL and LX products remain immutable evidence.  This module adds a separate
``computed_transform_consistency`` layer by splitting source-linked components into structural
families *before* exact forward enumeration.  It never promotes a computed product to an observed
synthesis, route closure, procurement closure, or biological observation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.assembly import (
    ReactionProgramError,
    ReactionProgramSpec,
    ReactionProgramTrace,
    RegistryRepeatedReactionProgram,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import (
    iter_csv,
    read_csv_rows,
    read_json_object,
    write_csv_iter,
    write_json,
)
from forge.corpus.component_splits import (
    FOLDS,
    family_fold_map,
    product_fold,
    similarity_families,
)
from forge.corpus.lnpdb import LNPDBComponent, LNPDBRow, load_lnpdb
from forge.corpus.multireaction import (
    ATLAS_FIELDS,
    PROVENANCE_FIELDS,
    SEMANTIC_ATOM_FIELDS,
    STEP_FIELDS,
)
from forge.corpus.reaction_program_records import admits_reaction_program_structure

CONFIG_SCHEMA = "forge.multireaction_expansion_config.v1"
RESULT_SCHEMA = "forge.multireaction_expansion_result.v1"
MANIFEST_SCHEMA = "forge.multireaction_expansion_manifest.v1"
COMPONENT_SCHEMA = "forge.multireaction_expansion_components.v1"
ATTEMPT_SCHEMA = "forge.multireaction_expansion_attempts.v1"
ATLAS_SCHEMA = "forge.multireaction_program_atlas.v1"
STEPS_SCHEMA = "forge.multireaction_program_steps.v1"
SEMANTIC_SCHEMA = "forge.multireaction_semantic_atoms.v2"
PROVENANCE_SCHEMA = "forge.multireaction_source_provenance.v1"
SPLITS_SCHEMA = "forge.multireaction_expanded_component_family_splits.v1"

COMPONENT_FIELDS = (
    "component_id",
    "program_id",
    "reaction_id",
    "program_role",
    "reaction_role",
    "split_namespace",
    "canonical_smiles",
    "heavy_atoms",
    "elements_json",
    "source_classes_json",
    "source_slots_json",
    "source_studies_json",
    "source_record_ids_json",
    "source_occurrences",
    "is_source_executed_component",
    "raw_handle_matches",
    "symmetry_distinct_handle_sites",
    "reactive_hydrogen_capacity",
    "within_declared_component_support",
    "program_structural_admission",
    "admission_reason",
    "family_id",
    "family_size",
    "family_fold",
)
ATTEMPT_FIELDS = (
    "attempt_id",
    "program_id",
    "head_component_id",
    "repeat_component_id",
    "step_count",
    "disposition",
    "forward_products",
    "canonical_product_smiles",
    "reason",
)
SPLIT_FIELDS = (
    "record_id",
    "program_id",
    "head_component_id",
    "head_component_family_id",
    "head_component_fold",
    "repeat_component_id",
    "repeat_component_family_id",
    "repeat_component_fold",
    "product_fold",
    "evidence_stratum",
    "family_balance_weight_raw",
    "source_balanced_weight",
)


class MultiReactionExpansionError(ValueError):
    """The BL/LX expansion violates its evidence, split, or chemistry contract."""


@dataclass
class _Candidate:
    program_id: str
    reaction_id: str
    program_role: str
    reaction_role: str
    split_namespace: str
    canonical_smiles: str
    molecule: Chem.Mol
    source_classes: set[str] = field(default_factory=set)
    source_slots: set[str] = field(default_factory=set)
    source_studies: set[str] = field(default_factory=set)
    source_record_ids: set[str] = field(default_factory=set)
    source_occurrences: int = 0
    source_executed: bool = False


@dataclass(frozen=True)
class _Component:
    component_id: str
    program_id: str
    reaction_id: str
    program_role: str
    reaction_role: str
    canonical_smiles: str
    family_id: str
    family_size: int
    family_fold: str
    step_count: int


@dataclass(frozen=True)
class _Product:
    record_id: str
    program_id: str
    canonical_product_smiles: str
    head: _Component
    repeat: _Component
    evidence_stratum: str
    atlas_row: Mapping[str, Any]
    step_rows: tuple[Mapping[str, Any], ...]
    semantic_rows: tuple[Mapping[str, Any], ...]


def _identifier(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _canonical(smiles: str) -> tuple[str, Chem.Mol] | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    with rdBase.BlockLogs():
        normalized = Chem.MolFromSmiles(canonical)
    return (canonical, normalized) if normalized is not None else None


def _json_list(values: Sequence[str] | set[str]) -> str:
    return json.dumps(sorted(set(values)), separators=(",", ":"))


def _config(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = read_json_object(path, error=MultiReactionExpansionError, label="expansion config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise MultiReactionExpansionError(
            f"unsupported expansion config schema: {config.get('schema_version')!r}"
        )
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise MultiReactionExpansionError("expansion seed must be an integer")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise MultiReactionExpansionError("expansion policy is missing")
    if policy.get("identity") != "canonical_constitutional_smiles":
        raise MultiReactionExpansionError("expanded product identity must remain constitutional")
    if policy.get("source_activity_labels_inherited") is not False:
        raise MultiReactionExpansionError("source activity labels must never be inherited")
    if policy.get("computed_product_is_observed_synthesis") is not False:
        raise MultiReactionExpansionError("computed products are not observed syntheses")
    if policy.get("computed_product_is_route_closed") is not False:
        raise MultiReactionExpansionError("computed products are not route closed")
    if policy.get("uniform_product_row_sampling_allowed") is not False:
        raise MultiReactionExpansionError("uniform Cartesian product sampling is prohibited")
    fractions = policy.get("family_split_fractions")
    if not isinstance(fractions, dict) or tuple(fractions) != FOLDS:
        raise MultiReactionExpansionError(f"family split must define {FOLDS} in order")
    if abs(sum(float(fractions[fold]) for fold in FOLDS) - 1.0) > 1e-12:
        raise MultiReactionExpansionError("family split fractions must sum to one")
    masses = policy.get("evidence_stratum_mass")
    if not isinstance(masses, dict) or set(masses) != {
        "source_executed",
        "computed_transform_consistency",
    }:
        raise MultiReactionExpansionError("evidence_stratum_mass has the wrong strata")
    if abs(sum(float(value) for value in masses.values()) - 1.0) > 1e-12:
        raise MultiReactionExpansionError("evidence stratum masses must sum to one")
    if any(float(value) <= 0 for value in masses.values()):
        raise MultiReactionExpansionError("evidence stratum masses must be positive")

    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise MultiReactionExpansionError("expansion input pins are missing")
    paths: dict[str, Path] = {}
    for label, pin in sorted(inputs.items()):
        try:
            paths[label] = resolve_pin(pin, repo, label=label)
        except ValueError as exc:
            raise MultiReactionExpansionError(str(exc)) from exc
    required = {
        "lnpdb",
        "program_config",
        "qualified_reaction_families",
        "source_atlas",
        "source_steps",
        "source_semantic_atoms",
        "source_splits",
        "source_provenance",
        "source_result",
    }
    if set(paths) != required:
        raise MultiReactionExpansionError(
            f"expansion inputs differ from contract: {sorted(set(paths) ^ required)}"
        )
    return config, paths


def _program_specs(config: Mapping[str, Any], path: Path) -> dict[str, ReactionProgramSpec]:
    source = read_json_object(
        path,
        error=MultiReactionExpansionError,
        label="source reaction-program config",
    )
    raw_by_id = {
        str(raw["program_id"]): raw
        for raw in source.get("programs", [])
        if isinstance(raw, dict) and isinstance(raw.get("program_id"), str)
    }
    requested = config.get("programs")
    if not isinstance(requested, list) or not requested:
        raise MultiReactionExpansionError("expansion config has no programs")
    result: dict[str, ReactionProgramSpec] = {}
    for contract in requested:
        if not isinstance(contract, dict):
            raise MultiReactionExpansionError("expansion program contract must be an object")
        program_id = str(contract.get("program_id", ""))
        if program_id not in raw_by_id:
            raise MultiReactionExpansionError(f"unknown source program: {program_id!r}")
        raw = raw_by_id[program_id]
        result[program_id] = ReactionProgramSpec(
            program_id=program_id,
            reaction_id=str(raw["reaction_id"]),
            accumulator_role=str(raw["accumulator_role"]),
            repeat_role=str(raw["repeat_role"]),
            minimum_steps=int(raw["minimum_steps"]),
            maximum_steps=int(raw["maximum_steps"]),
        )
    if set(result) != {str(value["program_id"]) for value in requested}:
        raise MultiReactionExpansionError("expansion program identities are duplicated")
    return result


def _slot(row: LNPDBRow, name: str) -> LNPDBComponent:
    if name == "head":
        return row.head
    if name == "linker":
        return row.linker
    if name.startswith("tail") and name[4:].isdigit():
        return row.tail(int(name[4:]))
    raise MultiReactionExpansionError(f"unknown LNPDB component slot: {name!r}")


def _add_candidate(
    candidates: dict[tuple[str, str, str], _Candidate],
    *,
    spec: ReactionProgramSpec,
    program_role: str,
    split_namespace: str,
    smiles: str,
    source_class: str,
    source_slot: str,
    source_study: str,
    source_record_id: str,
    source_executed: bool,
) -> bool:
    normalized = _canonical(smiles)
    if normalized is None:
        return False
    canonical, molecule = normalized
    reaction_role = spec.accumulator_role if program_role == "accumulator" else spec.repeat_role
    key = (spec.program_id, program_role, canonical)
    entry = candidates.get(key)
    if entry is None:
        entry = _Candidate(
            program_id=spec.program_id,
            reaction_id=spec.reaction_id,
            program_role=program_role,
            reaction_role=reaction_role,
            split_namespace=split_namespace,
            canonical_smiles=canonical,
            molecule=molecule,
        )
        candidates[key] = entry
    entry.source_classes.add(source_class)
    if source_slot:
        entry.source_slots.add(source_slot)
    if source_study:
        entry.source_studies.add(source_study)
    if source_record_id:
        entry.source_record_ids.add(source_record_id)
    entry.source_occurrences += 1
    entry.source_executed = entry.source_executed or source_executed
    return True


def _head_capacity(molecule: Chem.Mol, query: Chem.Mol) -> tuple[int, int, int]:
    matches = molecule.GetSubstructMatches(query, uniquify=True)
    if query.GetNumAtoms() != 1:
        raise MultiReactionExpansionError("repeated-program accumulator handle must be one atom")
    indices = {match[0] for match in matches}
    ranks = Chem.CanonicalRankAtoms(molecule, breakTies=False)
    distinct = len({ranks[index] for index in indices})
    capacity = sum(int(molecule.GetAtomWithIdx(index).GetTotalNumHs()) for index in indices)
    return len(matches), distinct, capacity


def _component_rows(
    *,
    config: Mapping[str, Any],
    specs: Mapping[str, ReactionProgramSpec],
    adapters: Mapping[str, RegistryRepeatedReactionProgram],
    lnpdb_path: Path,
    source_atlas_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], _Component], dict[str, Any]]:
    candidates: dict[tuple[str, str, str], _Candidate] = {}
    invalid_source_strings: Counter[tuple[str, str, str]] = Counter()
    catalogue = load_lnpdb(lnpdb_path, error=MultiReactionExpansionError)
    contract_by_program = {str(value["program_id"]): value for value in config["programs"]}
    for program_id, spec in specs.items():
        contract = contract_by_program[program_id]
        slot_contracts = (
            ("accumulator", spec.accumulator_role, contract.get("accumulator_source_slots")),
            ("repeat", spec.repeat_role, contract.get("repeat_source_slots")),
        )
        for program_role, namespace, raw_slots in slot_contracts:
            if not isinstance(raw_slots, list) or not raw_slots:
                raise MultiReactionExpansionError(
                    f"{program_id}/{program_role} has no LNPDB source slots"
                )
            for row in catalogue.rows:
                for raw_slot in raw_slots:
                    slot_name = str(raw_slot)
                    component = _slot(row, slot_name)
                    if component.smiles is None:
                        continue
                    if not _add_candidate(
                        candidates,
                        spec=spec,
                        program_role=program_role,
                        split_namespace=namespace,
                        smiles=component.smiles,
                        source_class="lnpdb_reported_component",
                        source_slot=slot_name,
                        source_study=row.experiment_id,
                        source_record_id=row.lnp_id,
                        source_executed=False,
                    ):
                        invalid_source_strings[(program_id, program_role, slot_name)] += 1

    source_atlas = read_csv_rows(
        source_atlas_path,
        error=MultiReactionExpansionError,
        label="source multi-reaction atlas",
        required_fields=ATLAS_FIELDS,
    )
    exact_source_rows = [
        source_row
        for source_row in source_atlas
        if source_row["disposition"] == "admit_exact"
        and source_row["semantic_origin_status"] == "exact"
    ]
    for atlas_row in exact_source_rows:
        program_id = atlas_row["program_id"]
        if program_id not in specs:
            continue
        spec = specs[program_id]
        for program_role, namespace, field_name in (
            ("accumulator", spec.accumulator_role, "terminal_head_smiles"),
            ("repeat", spec.repeat_role, "repeat_component_smiles"),
        ):
            if not _add_candidate(
                candidates,
                spec=spec,
                program_role=program_role,
                split_namespace=namespace,
                smiles=atlas_row[field_name],
                source_class="source_executed_component",
                source_slot="source_review_resolved",
                source_study=atlas_row["source_study"],
                source_record_id=atlas_row["record_id"],
                source_executed=True,
            ):
                raise MultiReactionExpansionError(
                    f"source-executed component no longer parses: {atlas_row['record_id']}"
                )

    policy = config["policy"]
    allowed_elements = set(str(value) for value in policy["allowed_elements"])
    maximum_component_atoms = int(policy["maximum_component_heavy_atoms"])
    preliminary: list[dict[str, Any]] = []
    admitted_candidates: list[_Candidate] = []
    for candidate in sorted(
        candidates.values(),
        key=lambda value: (value.program_id, value.program_role, value.canonical_smiles),
    ):
        adapter = adapters[candidate.program_id]
        role_index = 0 if candidate.program_role == "accumulator" else 1
        query = adapter._reaction.handles[role_index]
        forbidden = adapter._reaction.forbidden[role_index]
        raw_matches = len(candidate.molecule.GetSubstructMatches(query, uniquify=True))
        if candidate.program_role == "accumulator":
            raw_matches, distinct_sites, capacity = _head_capacity(candidate.molecule, query)
            handle_ok = adapter.spec.minimum_steps <= capacity <= adapter.spec.maximum_steps
        else:
            distinct_sites = raw_matches
            capacity = 0
            handle_ok = raw_matches == 1
        forbidden_match = any(candidate.molecule.HasSubstructMatch(item) for item in forbidden)
        elements = {atom.GetSymbol() for atom in candidate.molecule.GetAtoms()}
        within_support = (
            candidate.molecule.GetNumHeavyAtoms() <= maximum_component_atoms
            and elements.issubset(allowed_elements)
            and all(atom.GetNumRadicalElectrons() == 0 for atom in candidate.molecule.GetAtoms())
        )
        if not within_support:
            admitted = False
            reason = "outside_declared_component_support"
        elif forbidden_match or not handle_ok:
            admitted = False
            reason = "fails_repeated_program_handle_policy"
        else:
            admitted = True
            reason = (
                "source_executed_component"
                if candidate.source_executed
                else "computed_transform_candidate"
            )
        if candidate.source_executed and not admitted:
            raise MultiReactionExpansionError(
                f"source-executed component failed expansion policy: "
                f"{candidate.program_id}/{candidate.program_role}/{candidate.canonical_smiles}"
            )
        component_id = _identifier(
            "mrc",
            candidate.program_id,
            candidate.program_role,
            candidate.canonical_smiles,
        )
        preliminary.append(
            {
                "component_id": component_id,
                "program_id": candidate.program_id,
                "reaction_id": candidate.reaction_id,
                "program_role": candidate.program_role,
                "reaction_role": candidate.reaction_role,
                "split_namespace": candidate.split_namespace,
                "canonical_smiles": candidate.canonical_smiles,
                "heavy_atoms": candidate.molecule.GetNumHeavyAtoms(),
                "elements_json": _json_list(elements),
                "source_classes_json": _json_list(candidate.source_classes),
                "source_slots_json": _json_list(candidate.source_slots),
                "source_studies_json": _json_list(candidate.source_studies),
                "source_record_ids_json": _json_list(candidate.source_record_ids),
                "source_occurrences": candidate.source_occurrences,
                "is_source_executed_component": str(candidate.source_executed).lower(),
                "raw_handle_matches": raw_matches,
                "symmetry_distinct_handle_sites": distinct_sites,
                "reactive_hydrogen_capacity": capacity,
                "within_declared_component_support": str(within_support).lower(),
                "program_structural_admission": str(admitted).lower(),
                "admission_reason": reason,
                "family_id": "",
                "family_size": "",
                "family_fold": "",
            }
        )
        if admitted:
            admitted_candidates.append(candidate)

    fingerprint = policy["family_fingerprint"]
    fractions = policy["family_split_fractions"]
    family_by_namespace_smiles: dict[tuple[str, str], str] = {}
    size_by_family: dict[str, int] = {}
    fold_by_family: dict[str, str] = {}
    namespaces = sorted({candidate.split_namespace for candidate in admitted_candidates})
    for namespace in namespaces:
        values = sorted(
            {
                candidate.canonical_smiles
                for candidate in admitted_candidates
                if candidate.split_namespace == namespace
            }
        )
        families = similarity_families(
            values,
            fingerprint,
            namespace,
            error=MultiReactionExpansionError,
        )
        members: dict[str, list[str]] = defaultdict(list)
        for smiles, family_id in families.items():
            members[family_id].append(smiles)
            family_by_namespace_smiles[(namespace, smiles)] = family_id
        folds = family_fold_map(
            members,
            fractions,
            int(config["seed"]),
            namespace,
            error=MultiReactionExpansionError,
        )
        for family_id, family_members in members.items():
            size_by_family[family_id] = len(family_members)
            fold_by_family[family_id] = folds[family_id]

    components: dict[tuple[str, str, str], _Component] = {}
    rows: list[dict[str, Any]] = []
    for component_row in preliminary:
        if component_row["program_structural_admission"] == "true":
            key = (component_row["split_namespace"], component_row["canonical_smiles"])
            family_id = family_by_namespace_smiles[key]
            component_row["family_id"] = family_id
            component_row["family_size"] = size_by_family[family_id]
            component_row["family_fold"] = fold_by_family[family_id]
            expanded_component = _Component(
                component_id=str(component_row["component_id"]),
                program_id=str(component_row["program_id"]),
                reaction_id=str(component_row["reaction_id"]),
                program_role=str(component_row["program_role"]),
                reaction_role=str(component_row["reaction_role"]),
                canonical_smiles=str(component_row["canonical_smiles"]),
                family_id=family_id,
                family_size=int(component_row["family_size"]),
                family_fold=str(component_row["family_fold"]),
                step_count=int(component_row["reactive_hydrogen_capacity"]),
            )
            components[
                (
                    expanded_component.program_id,
                    expanded_component.program_role,
                    expanded_component.canonical_smiles,
                )
            ] = expanded_component
        rows.append(component_row)

    summary = {
        "candidate_components": len(rows),
        "admitted_components": len(components),
        "source_executed_components": sum(
            row["is_source_executed_component"] == "true" for row in rows
        ),
        "admission_reasons": dict(Counter(str(row["admission_reason"]) for row in rows)),
        "admitted_by_program_role": {
            f"{program_id}|{program_role}": count
            for (program_id, program_role), count in sorted(
                Counter(
                    (value.program_id, value.program_role) for value in components.values()
                ).items()
            )
        },
        "shared_split_namespaces": {
            namespace: len(
                {
                    component.canonical_smiles
                    for component in components.values()
                    if component.reaction_role == namespace
                }
            )
            for namespace in namespaces
        },
        "family_fold_counts": dict(
            Counter(
                f"{row['split_namespace']}|{row['family_fold']}"
                for row in rows
                if row["program_structural_admission"] == "true"
            )
        ),
        "invalid_lnpdb_component_strings": {
            "|".join(key): value for key, value in sorted(invalid_source_strings.items())
        },
    }
    return rows, components, summary


def _group(rows: Sequence[Mapping[str, str]], key: str) -> dict[str, tuple[Mapping[str, str], ...]]:
    grouped: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return {value: tuple(items) for value, items in grouped.items()}


def _source_products(
    *,
    specs: Mapping[str, ReactionProgramSpec],
    components: Mapping[tuple[str, str, str], _Component],
    atlas_path: Path,
    steps_path: Path,
    semantic_path: Path,
) -> list[_Product]:
    atlas = read_csv_rows(
        atlas_path,
        error=MultiReactionExpansionError,
        label="source atlas",
        required_fields=ATLAS_FIELDS,
    )
    step_groups = _group(
        read_csv_rows(
            steps_path,
            error=MultiReactionExpansionError,
            label="source steps",
            required_fields=STEP_FIELDS,
        ),
        "record_id",
    )
    semantic_groups = _group(
        read_csv_rows(
            semantic_path,
            error=MultiReactionExpansionError,
            label="source semantic atoms",
            required_fields=SEMANTIC_ATOM_FIELDS,
        ),
        "record_id",
    )
    products: list[_Product] = []
    for row in atlas:
        if not admits_reaction_program_structure(row) or row["disposition"] != "admit_exact":
            continue
        program_id = row["program_id"]
        if program_id not in specs:
            continue
        head_key = (program_id, "accumulator", row["terminal_head_smiles"])
        repeat_key = (program_id, "repeat", row["repeat_component_smiles"])
        if head_key not in components or repeat_key not in components:
            raise MultiReactionExpansionError(
                f"source product components are absent from expanded registry: {row['record_id']}"
            )
        record_id = row["record_id"]
        if record_id not in step_groups or record_id not in semantic_groups:
            raise MultiReactionExpansionError(
                f"source semantic ledgers are incomplete: {record_id}"
            )
        products.append(
            _Product(
                record_id=record_id,
                program_id=program_id,
                canonical_product_smiles=row["canonical_product_smiles"],
                head=components[head_key],
                repeat=components[repeat_key],
                evidence_stratum="source_executed",
                atlas_row=dict(row),
                step_rows=tuple(step_groups[record_id]),
                semantic_rows=tuple(semantic_groups[record_id]),
            )
        )
    return products


def _within_product_support(molecule: Chem.Mol, policy: Mapping[str, Any]) -> bool:
    return (
        molecule.GetNumHeavyAtoms() <= int(policy["maximum_product_heavy_atoms"])
        and {atom.GetSymbol() for atom in molecule.GetAtoms()}.issubset(
            set(str(value) for value in policy["allowed_elements"])
        )
        and all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    )


def _virtual_product(
    *,
    adapter: RegistryRepeatedReactionProgram,
    head: _Component,
    repeat: _Component,
    trace: ReactionProgramTrace,
) -> _Product:
    origins = adapter.atom_origins(trace)
    product_smiles = trace.intermediate_product_smiles[-1]
    if origins.canonical_product_smiles != product_smiles:
        raise MultiReactionExpansionError("forward trace and canonical atom origins disagree")
    record_id = _identifier("mre", adapter.spec.program_id, product_smiles)
    accumulator = trace.terminal_head_smiles
    step_rows: list[dict[str, Any]] = []
    for step_index, (repeated, step_product) in enumerate(
        zip(trace.repeated_component_smiles, trace.intermediate_product_smiles, strict=True),
        start=1,
    ):
        step_rows.append(
            {
                "record_id": record_id,
                "program_id": adapter.spec.program_id,
                "step_index": step_index,
                "reaction_id": adapter.spec.reaction_id,
                "accumulator_role": adapter.spec.accumulator_role,
                "accumulator_input_smiles": accumulator,
                "repeat_role": adapter.spec.repeat_role,
                "repeat_component_smiles": repeated,
                "product_smiles": step_product,
                "exact_forward_roundtrip": "true",
            }
        )
        accumulator = step_product
    semantic_rows = tuple(
        {
            "record_id": record_id,
            "program_id": adapter.spec.program_id,
            "atom_index": atom_index,
            "origin_role": (
                adapter.spec.accumulator_role
                if origin == "accumulator"
                else adapter.spec.repeat_role
            ),
            "core_position": core_position,
            "program_depth": origins.step_count,
        }
        for atom_index, (origin, core_position) in enumerate(
            zip(origins.atom_origins, origins.core_positions, strict=True)
        )
    )
    return _Product(
        record_id=record_id,
        program_id=adapter.spec.program_id,
        canonical_product_smiles=product_smiles,
        head=head,
        repeat=repeat,
        evidence_stratum="computed_transform_consistency",
        atlas_row={
            "record_id": record_id,
            "program_id": adapter.spec.program_id,
            "reaction_id": adapter.spec.reaction_id,
            "source_study": "LNPDB_COMPONENT_ENUMERATION",
            "source_product_labels": "",
            "canonical_product_smiles": product_smiles,
            "terminal_head_smiles": trace.terminal_head_smiles,
            "repeat_component_smiles": repeat.canonical_smiles,
            "step_count": trace.step_count,
            "source_row_count": 0,
            "evidence_basis": "computed_transform_consistency",
            "disposition": "admit_transform_consistency",
            "abstention_reason": "",
            "exact_forward_roundtrip": "true",
            "source_locator": "source-linked LNPDB components plus frozen registry transform",
            "semantic_origin_status": "exact",
            "semantic_origin_reason": "",
        },
        step_rows=tuple(step_rows),
        semantic_rows=semantic_rows,
    )


def _enumerate_products(
    *,
    policy: Mapping[str, Any],
    specs: Mapping[str, ReactionProgramSpec],
    adapters: Mapping[str, RegistryRepeatedReactionProgram],
    components: Mapping[tuple[str, str, str], _Component],
    source_products: Sequence[_Product],
) -> tuple[list[_Product], list[dict[str, Any]], dict[str, Any]]:
    source_keys = {(value.program_id, value.canonical_product_smiles) for value in source_products}
    virtual_by_key: dict[tuple[str, str], list[tuple[_Product, int]]] = defaultdict(list)
    attempts: list[dict[str, Any]] = []
    maximum_outcomes = int(policy["maximum_forward_outcomes"])
    maximum_states = int(policy["maximum_forward_states"])
    for program_id, spec in sorted(specs.items()):
        adapter = adapters[program_id]
        heads = sorted(
            (
                value
                for value in components.values()
                if value.program_id == program_id and value.program_role == "accumulator"
            ),
            key=lambda value: value.canonical_smiles,
        )
        repeats = sorted(
            (
                value
                for value in components.values()
                if value.program_id == program_id and value.program_role == "repeat"
            ),
            key=lambda value: value.canonical_smiles,
        )
        for head in heads:
            if not spec.minimum_steps <= head.step_count <= spec.maximum_steps:
                raise MultiReactionExpansionError(
                    "admitted head has invalid repeated-program depth"
                )
            for repeat in repeats:
                attempt_id = _identifier("mra", program_id, head.component_id, repeat.component_id)
                attempt: dict[str, Any] = {
                    "attempt_id": attempt_id,
                    "program_id": program_id,
                    "head_component_id": head.component_id,
                    "repeat_component_id": repeat.component_id,
                    "step_count": head.step_count,
                    "disposition": "",
                    "forward_products": 0,
                    "canonical_product_smiles": "",
                    "reason": "",
                }
                try:
                    traces = adapter.forward_traces(
                        head.canonical_smiles,
                        (repeat.canonical_smiles,) * head.step_count,
                        maximum_outcomes=maximum_outcomes,
                        maximum_states=maximum_states,
                    )
                except ReactionProgramError as exc:
                    attempt["disposition"] = "abstain"
                    attempt["reason"] = f"forward_enumeration_error:{exc}"
                    attempts.append(attempt)
                    continue
                attempt["forward_products"] = len(traces)
                if len(traces) != 1:
                    attempt["disposition"] = "abstain"
                    attempt["reason"] = "forward_program_did_not_resolve_one_product"
                    attempts.append(attempt)
                    continue
                trace = traces[0]
                product_smiles = trace.intermediate_product_smiles[-1]
                attempt["canonical_product_smiles"] = product_smiles
                normalized = _canonical(product_smiles)
                if normalized is None or normalized[0] != product_smiles:
                    attempt["disposition"] = "abstain"
                    attempt["reason"] = "forward_product_failed_constitutional_normalization"
                    attempts.append(attempt)
                    continue
                if not _within_product_support(normalized[1], policy):
                    attempt["disposition"] = "abstain"
                    attempt["reason"] = "forward_product_outside_declared_model_support"
                    attempts.append(attempt)
                    continue
                try:
                    product = _virtual_product(
                        adapter=adapter,
                        head=head,
                        repeat=repeat,
                        trace=trace,
                    )
                except (ReactionProgramError, MultiReactionExpansionError) as exc:
                    attempt["disposition"] = "abstain"
                    attempt["reason"] = f"semantic_origin_error:{exc}"
                    attempts.append(attempt)
                    continue
                attempt["disposition"] = "candidate"
                attempt_index = len(attempts)
                attempts.append(attempt)
                virtual_by_key[(program_id, product_smiles)].append((product, attempt_index))

    admitted_virtual: list[_Product] = []
    duplicate_source_products = 0
    ambiguous_products = 0
    for key, candidates in sorted(virtual_by_key.items()):
        if key in source_keys:
            duplicate_source_products += 1
            for _, index in candidates:
                attempts[index]["disposition"] = "source_product_precedence"
                attempts[index]["reason"] = "computed product duplicates source-executed product"
            continue
        component_pairs = {
            (value.head.component_id, value.repeat.component_id) for value, _ in candidates
        }
        if len(component_pairs) != 1:
            ambiguous_products += 1
            for _, index in candidates:
                attempts[index]["disposition"] = "abstain"
                attempts[index]["reason"] = "multiple_component_factorizations"
            continue
        product, selected_index = candidates[0]
        attempts[selected_index]["disposition"] = "admit_transform_consistency"
        for _, duplicate_index in candidates[1:]:
            attempts[duplicate_index]["disposition"] = "duplicate"
            attempts[duplicate_index]["reason"] = "duplicate component/product enumeration"
        admitted_virtual.append(product)

    products = sorted(
        [*source_products, *admitted_virtual],
        key=lambda value: (value.program_id, value.canonical_product_smiles, value.record_id),
    )
    if len({value.record_id for value in products}) != len(products):
        raise MultiReactionExpansionError("expanded products contain duplicate record identifiers")
    summary = {
        "enumeration_attempts": len(attempts),
        "attempt_dispositions": dict(Counter(str(row["disposition"]) for row in attempts)),
        "attempt_abstention_reasons": dict(
            Counter(str(row["reason"]) for row in attempts if row["reason"])
        ),
        "computed_products_admitted": len(admitted_virtual),
        "source_products_preserved": len(source_products),
        "computed_duplicates_of_source_products": duplicate_source_products,
        "ambiguous_component_factorization_products": ambiguous_products,
    }
    return products, attempts, summary


def _split_rows(
    products: Sequence[_Product],
    *,
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows: list[dict[str, Any]] = []
    group_totals: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    for product in products:
        fold = product_fold(
            (product.head.family_fold, product.repeat.family_fold),
            error=MultiReactionExpansionError,
        )
        raw_weight = 1.0 / (product.head.family_size * product.repeat.family_size)
        group = (fold, product.program_id, product.evidence_stratum)
        group_totals[group] += raw_weight
        raw_rows.append(
            {
                "record_id": product.record_id,
                "program_id": product.program_id,
                "head_component_id": product.head.component_id,
                "head_component_family_id": product.head.family_id,
                "head_component_fold": product.head.family_fold,
                "repeat_component_id": product.repeat.component_id,
                "repeat_component_family_id": product.repeat.family_id,
                "repeat_component_fold": product.repeat.family_fold,
                "product_fold": fold,
                "evidence_stratum": product.evidence_stratum,
                "family_balance_weight_raw": raw_weight,
                "source_balanced_weight": 0.0,
            }
        )
    expected_groups = {
        (fold, program_id, stratum)
        for fold in FOLDS
        for program_id in {value.program_id for value in products}
        for stratum in ("source_executed", "computed_transform_consistency")
    }
    missing = sorted(expected_groups - set(group_totals))
    if missing:
        raise MultiReactionExpansionError(
            f"one or more fold/program/evidence groups are empty: {missing}"
        )
    masses = policy["evidence_stratum_mass"]
    for row in raw_rows:
        group = (row["product_fold"], row["program_id"], row["evidence_stratum"])
        row["source_balanced_weight"] = (
            float(masses[row["evidence_stratum"]])
            * float(row["family_balance_weight_raw"])
            / group_totals[group]
        )
        row["family_balance_weight_raw"] = f"{float(row['family_balance_weight_raw']):.12g}"
        row["source_balanced_weight"] = f"{float(row['source_balanced_weight']):.12g}"

    program_mass: defaultdict[tuple[str, str], float] = defaultdict(float)
    stratum_mass: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    for row in raw_rows:
        weight = float(row["source_balanced_weight"])
        program_mass[(row["product_fold"], row["program_id"])] += weight
        stratum_mass[(row["product_fold"], row["program_id"], row["evidence_stratum"])] += weight
    if any(abs(value - 1.0) > 1e-8 for value in program_mass.values()):
        raise MultiReactionExpansionError("program mass is not balanced within every fold")
    for (_, _, stratum), value in stratum_mass.items():
        if abs(value - float(masses[stratum])) > 1e-8:
            raise MultiReactionExpansionError("evidence stratum mass is not balanced")
    summary = {
        "product_folds": dict(Counter(str(row["product_fold"]) for row in raw_rows)),
        "products_by_program_fold": dict(
            Counter(f"{row['program_id']}|{row['product_fold']}" for row in raw_rows)
        ),
        "products_by_evidence": dict(Counter(str(row["evidence_stratum"]) for row in raw_rows)),
        "sampling_policy": "equal program mass, fixed evidence-stratum mass, inverse family size",
    }
    return raw_rows, summary


def _artifact_rows(products: Sequence[_Product], kind: str) -> Iterator[Mapping[str, Any]]:
    for product in products:
        if kind == "atlas":
            yield product.atlas_row
        elif kind == "steps":
            yield from product.step_rows
        elif kind == "semantic_atoms":
            yield from product.semantic_rows
        else:
            raise AssertionError(kind)


def build_multireaction_expansion(
    config_path: Path,
    repo: Path,
    *,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the component registry and exact BL/LX reaction-enumerated corpus."""

    required_outputs = {
        "components",
        "attempts",
        "atlas",
        "steps",
        "semantic_atoms",
        "provenance",
        "splits",
        "manifest",
        "result",
    }
    if set(outputs) != required_outputs:
        raise MultiReactionExpansionError(
            f"expansion outputs differ from contract: {sorted(set(outputs) ^ required_outputs)}"
        )
    config, paths = _config(config_path, repo)
    specs = _program_specs(config, paths["program_config"])
    registry_sha = str(sha256_file(paths["qualified_reaction_families"]))
    adapters = {
        program_id: RegistryRepeatedReactionProgram.from_registry(
            paths["qualified_reaction_families"],
            spec,
            expected_sha256=registry_sha,
        )
        for program_id, spec in specs.items()
    }
    component_rows, components, component_summary = _component_rows(
        config=config,
        specs=specs,
        adapters=adapters,
        lnpdb_path=paths["lnpdb"],
        source_atlas_path=paths["source_atlas"],
    )
    source_products = _source_products(
        specs=specs,
        components=components,
        atlas_path=paths["source_atlas"],
        steps_path=paths["source_steps"],
        semantic_path=paths["source_semantic_atoms"],
    )
    products, attempts, enumeration_summary = _enumerate_products(
        policy=config["policy"],
        specs=specs,
        adapters=adapters,
        components=components,
        source_products=source_products,
    )
    splits, split_summary = _split_rows(products, policy=config["policy"])

    old_splits = {row["record_id"]: row["product_fold"] for row in iter_csv(paths["source_splits"])}
    new_splits = {row["record_id"]: row["product_fold"] for row in splits}
    changed_source_folds = sum(
        old_splits[product.record_id] != new_splits[product.record_id]
        for product in source_products
    )
    provenance = [
        row
        for row in read_csv_rows(
            paths["source_provenance"],
            error=MultiReactionExpansionError,
            label="source provenance",
            required_fields=PROVENANCE_FIELDS,
        )
        if row["record_id"] in {product.record_id for product in source_products}
    ]

    write_csv_iter(outputs["components"], component_rows, COMPONENT_FIELDS)
    write_csv_iter(outputs["attempts"], attempts, ATTEMPT_FIELDS)
    write_csv_iter(outputs["atlas"], _artifact_rows(products, "atlas"), ATLAS_FIELDS)
    write_csv_iter(outputs["steps"], _artifact_rows(products, "steps"), STEP_FIELDS)
    write_csv_iter(
        outputs["semantic_atoms"],
        _artifact_rows(products, "semantic_atoms"),
        SEMANTIC_ATOM_FIELDS,
    )
    write_csv_iter(outputs["provenance"], provenance, PROVENANCE_FIELDS)
    write_csv_iter(outputs["splits"], splits, SPLIT_FIELDS)

    artifact_schemas = {
        "components": COMPONENT_SCHEMA,
        "attempts": ATTEMPT_SCHEMA,
        "atlas": ATLAS_SCHEMA,
        "steps": STEPS_SCHEMA,
        "semantic_atoms": SEMANTIC_SCHEMA,
        "provenance": PROVENANCE_SCHEMA,
        "splits": SPLITS_SCHEMA,
    }
    artifacts = {
        label: {
            **artifact_record(path, logical_path=path.name),
            "schema_version": artifact_schemas[label],
        }
        for label, path in sorted(outputs.items())
        if label in artifact_schemas
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": artifacts,
    }
    write_json(outputs["manifest"], manifest)

    input_records = {label: pin_record(path, repo) for label, path in sorted(paths.items())}
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_bl_lx_reaction_enumerated_expansion",
        "task": config["task"],
        "seed": int(config["seed"]),
        "config": pin_record(config_path, repo),
        "inputs": input_records,
        "policy": config["policy"],
        "summary": {
            **component_summary,
            **enumeration_summary,
            **split_summary,
            "total_products": len(products),
            "exact_program_steps": sum(len(product.step_rows) for product in products),
            "semantic_atom_rows": sum(len(product.semantic_rows) for product in products),
            "source_products_with_new_family_fold": changed_source_folds,
        },
        "claims_boundary": {
            "source_executed_evidence_rewritten": False,
            "computed_transform_consistency_is_observed_synthesis": False,
            "computed_transform_consistency_is_synthesis_success": False,
            "computed_transform_consistency_is_route_closure": False,
            "source_activity_labels_inherited": False,
            "component_family_assignment_precedes_enumeration": True,
            "reductive_amination_substructure_hit_rate_reported": False,
            "training_must_use_source_balanced_weight": True,
        },
        "artifacts": artifacts,
    }
    write_json(outputs["result"], result)
    return result


__all__ = [
    "ATTEMPT_FIELDS",
    "COMPONENT_FIELDS",
    "SPLIT_FIELDS",
    "MultiReactionExpansionError",
    "build_multireaction_expansion",
]
