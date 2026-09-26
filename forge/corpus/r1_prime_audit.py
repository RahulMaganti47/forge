"""Non-circular R1-prime anchoring audit for M0-04.

This module creates a reaction-enumerated diagnostic reference set from blocks
retro-decomposed from each frozen R0 training fold. It is not a molecular
generator. The product model remains a whole-graph discrete flow.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import itertools
import json
import math
import os
import platform
import shutil
import sqlite3
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdChemReactions, rdFingerprintGenerator

from forge.chemistry.reactive_sites import (
    RAW_SUBSTRUCTURE_MATCHES,
    SUPPORTED_MULTIPLICITY_SEMANTICS,
    SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES,
    audit_reactive_site_multiplicity,
)
from forge.corpus.r0_splits import FOLDS, SCHEMES, load_frozen_r0_splits

LEGACY_CONFIG_SCHEMA_VERSION = "m0_04_r1_prime_audit_config.v2"
CONFIG_SCHEMA_VERSION = "m0_04_r1_prime_audit_config.v3"
RESULT_SCHEMA_VERSION = "m0_04_r1_prime_audit.v3"
CHECKPOINT_SCHEMA_VERSION = "m0_04_scheme_checkpoint.v3"
ALGORITHM_VERSION = "parallel_scheme_checkpoint_v3_constitutional_r0"
QUALIFIED_STATUS = "qualified_for_enumeration"
AGILE_SOURCE = "agile_measured1200"


class AuditError(ValueError):
    """Raised when M0-04 cannot satisfy its scientific or data contract."""


@dataclass(frozen=True)
class ReactionRole:
    """One registry-defined reactant role."""

    name: str
    required_handle_smarts: str
    forbidden_smarts: tuple[str, ...]
    allowed_site_multiplicity: tuple[int, ...]
    site_multiplicity_semantics: str


@dataclass(frozen=True)
class ReactionDefinition:
    """Serializable qualified reaction definition loaded from the registries."""

    reaction_id: str
    reaction_version: int
    status: str
    atom_mapped_reaction_smarts: str
    selectivity_policy: str
    reactant_roles: tuple[ReactionRole, ...]
    known_positive_examples: tuple[Mapping[str, Any], ...]
    known_negative_examples: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class DecompositionCandidate:
    """One exact forward-reconstructing retrosynthetic candidate."""

    scheme: str
    source_structure_id: str
    reaction_id: str
    reactant_smiles: tuple[str, ...]
    source_studies: tuple[str, ...]


@dataclass(frozen=True)
class BlockRecord:
    """One role-typed, train-derived component-pool record."""

    scheme: str
    block_id: str
    reaction_id: str
    role: str
    canonical_smiles: str
    route_occurrence_count: int
    source_structure_count: int
    source_studies: tuple[str, ...]


@dataclass
class CompiledReaction:
    """RDKit executors and role queries for one reaction definition."""

    definition: ReactionDefinition
    forward: rdChemReactions.ChemicalReaction
    reverse: rdChemReactions.ChemicalReaction
    handles: tuple[Chem.Mol, ...]
    forbidden: tuple[tuple[Chem.Mol, ...], ...]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return a file's SHA-256 digest."""

    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except FileNotFoundError as exc:
        raise AuditError(f"required input not found: {path}") from exc
    with handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    """Return a byte payload's SHA-256 digest."""

    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise AuditError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"{description} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"{description} must contain a JSON object: {path}")
    return value


def _role_policy_override_index(
    raw_overrides: Any,
) -> dict[tuple[str, str], dict[str, str]]:
    if not isinstance(raw_overrides, list):
        raise AuditError("role_policy_overrides must be a list")
    overrides: dict[tuple[str, str], dict[str, str]] = {}
    required_fields = {
        "reaction_id",
        "role",
        "site_multiplicity_semantics",
        "evidence_asset",
        "evidence_schema_version",
    }
    for index, raw in enumerate(raw_overrides):
        if not isinstance(raw, dict) or set(raw) != required_fields:
            raise AuditError(
                f"role_policy_overrides[{index}] must contain exactly " f"{sorted(required_fields)}"
            )
        if any(not isinstance(raw[field], str) or not raw[field] for field in required_fields):
            raise AuditError(f"role_policy_overrides[{index}] fields must be nonempty strings")
        semantics = raw["site_multiplicity_semantics"]
        if semantics not in SUPPORTED_MULTIPLICITY_SEMANTICS:
            raise AuditError(
                f"role_policy_overrides[{index}] has unsupported semantics {semantics!r}"
            )
        key = (raw["reaction_id"], raw["role"])
        if key in overrides:
            raise AuditError(f"duplicate role-policy override for {key[0]}/{key[1]}")
        overrides[key] = {
            field: raw[field]
            for field in (
                "site_multiplicity_semantics",
                "evidence_asset",
                "evidence_schema_version",
            )
        }
    return overrides


def load_config(path: Path) -> dict[str, Any]:
    """Load and structurally validate the M0-04 configuration."""

    config = _load_json(path, "M0-04 config")
    schema_version = config.get("schema_version")
    if schema_version not in {LEGACY_CONFIG_SCHEMA_VERSION, CONFIG_SCHEMA_VERSION}:
        raise AuditError(
            f"unsupported M0-04 config schema {config.get('schema_version')!r}; "
            f"expected {LEGACY_CONFIG_SCHEMA_VERSION!r} or {CONFIG_SCHEMA_VERSION!r}"
        )
    if tuple(config.get("schemes", ())) != SCHEMES:
        raise AuditError(f"config schemes must be ordered exactly as {SCHEMES}")
    if config.get("randomness_used") is not False:
        raise AuditError("M0-04 is deterministic; randomness_used must be false")
    fingerprint = config.get("fingerprint", {})
    if (
        fingerprint.get("kind") != "ECFP4"
        or fingerprint.get("radius") != 2
        or not isinstance(fingerprint.get("bits"), int)
        or fingerprint["bits"] <= 0
    ):
        raise AuditError("fingerprint config must define ECFP4 with radius 2 and positive bits")
    evidence_inputs = config.get("policy_evidence_inputs")
    if not isinstance(evidence_inputs, dict) or any(
        not isinstance(asset, str)
        or not asset
        or not isinstance(relative_path, str)
        or not relative_path
        or Path(relative_path).is_absolute()
        or ".." in Path(relative_path).parts
        for asset, relative_path in evidence_inputs.items()
    ):
        raise AuditError(
            "policy_evidence_inputs must map asset names to safe repository-relative paths"
        )
    overrides = _role_policy_override_index(config.get("role_policy_overrides"))
    for override in overrides.values():
        if override["evidence_asset"] not in evidence_inputs:
            raise AuditError(
                "every role-policy override must reference a declared policy evidence asset"
            )
    if schema_version == CONFIG_SCHEMA_VERSION:
        repository_inputs = config.get("repository_inputs")
        dataset = config.get("dataset")
        if not isinstance(repository_inputs, dict) or any(
            not isinstance(asset, str)
            or not asset
            or not isinstance(relative_path, str)
            or not relative_path
            or Path(relative_path).is_absolute()
            or ".." in Path(relative_path).parts
            for asset, relative_path in repository_inputs.items()
        ):
            raise AuditError(
                "v3 repository_inputs must map asset names to safe repository-relative paths"
            )
        required_dataset_fields = {
            "current_r0_asset",
            "historical_r0_asset",
            "r1_control_asset",
            "split_assignments_asset",
            "split_manifest_asset",
            "reaction_families_asset",
            "reactions_asset",
            "graph_identity",
        }
        if not isinstance(dataset, dict) or not required_dataset_fields.issubset(dataset):
            raise AuditError(f"v3 dataset must contain {sorted(required_dataset_fields)}")
        referenced_assets = {value for key, value in dataset.items() if key.endswith("_asset")}
        missing_assets = referenced_assets.difference(repository_inputs)
        if missing_assets:
            raise AuditError(
                f"v3 dataset references undeclared repository inputs: {sorted(missing_assets)}"
            )
        if dataset["graph_identity"] != "canonical_constitutional_smiles":
            raise AuditError("v3 M0-04 requires canonical constitutional graph identity")
        controls = config.get("controls")
        required_controls = {
            "historical_non_agile_rows",
            "historical_non_agile_recovered",
            "current_non_agile_rows",
            "current_non_agile_recovered",
        }
        if not isinstance(controls, dict) or not required_controls.issubset(controls):
            raise AuditError(f"v3 controls must contain {sorted(required_controls)}")
    return config


def _validate_role_policy_evidence(
    config: Mapping[str, Any],
    locations: Mapping[str, Path],
) -> None:
    overrides = _role_policy_override_index(config["role_policy_overrides"])
    for (reaction_id, role), override in overrides.items():
        evidence_asset = override["evidence_asset"]
        evidence = _load_json(
            locations[evidence_asset],
            f"role-policy evidence for {reaction_id}/{role}",
        )
        if evidence.get("schema_version") != override["evidence_schema_version"]:
            raise AuditError(f"{reaction_id}/{role}: role-policy evidence schema mismatch")
        policy = evidence.get("multiplicity_policy")
        decision = evidence.get("decision")
        if not isinstance(policy, dict) or not isinstance(decision, dict):
            raise AuditError(
                f"{reaction_id}/{role}: role-policy evidence is missing policy or decision"
            )
        if (
            policy.get("reaction_id") != reaction_id
            or policy.get("role") != role
            or policy.get("semantics") != override["site_multiplicity_semantics"]
            or decision.get("all_measured_products_pass") is not True
            or evidence.get("failed_row_labels") != []
        ):
            raise AuditError(
                f"{reaction_id}/{role}: role-policy evidence does not qualify the override"
            )


def verify_inputs(
    config_path: Path, vendor_dir: Path, split_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Path]]:
    """Verify every hash-pinned M0-04 input."""

    config = load_config(config_path)
    expected = config.get("expected_inputs", {})
    repository_root = config_path.resolve().parents[2]
    if config["schema_version"] == CONFIG_SCHEMA_VERSION:
        locations = {
            asset: repository_root / relative_path
            for asset, relative_path in config["repository_inputs"].items()
        }
    else:
        locations = {
            "r0_observed_real_structures.csv": (vendor_dir / "r0_observed_real_structures.csv"),
            "r1_reaction_enumerated_support_v1.csv": (
                vendor_dir / "r1_reaction_enumerated_support_v1.csv"
            ),
            "qualified_reaction_families_v1.json": (
                vendor_dir / "qualified_reaction_families_v1.json"
            ),
            "qualified_reactions_v1.json": vendor_dir / "qualified_reactions_v1.json",
            "r0_fold_assignments.csv": split_dir / "r0_fold_assignments.csv",
            "m0_03_manifest.json": split_dir / "manifest.json",
        }
    locations.update(
        {
            asset: repository_root / relative_path
            for asset, relative_path in config["policy_evidence_inputs"].items()
        }
    )
    if set(expected) != set(locations):
        raise AuditError(f"expected_inputs must contain exactly {sorted(locations)}")
    records = [
        {
            "asset": config_path.name,
            "role": "audit_config",
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        }
    ]
    for asset, path in locations.items():
        actual = sha256_file(path)
        if actual != expected[asset]:
            raise AuditError(
                f"hash mismatch for {asset}: expected {expected[asset]}, observed {actual}"
            )
        records.append({"asset": asset, "bytes": path.stat().st_size, "sha256": actual})
    _validate_role_policy_evidence(config, locations)
    return config, records, locations


def _parse_role(
    raw: Mapping[str, Any],
    reaction_id: str,
    site_multiplicity_semantics: str,
) -> ReactionRole:
    name = raw.get("name")
    handle = raw.get("required_handle_smarts")
    forbidden = raw.get("forbidden_smarts")
    multiplicity = raw.get("allowed_site_multiplicity")
    if not isinstance(name, str) or not name:
        raise AuditError(f"{reaction_id}: every role must have a nonempty name")
    if raw.get("count") != 1:
        raise AuditError(f"{reaction_id}/{name}: M0-04 currently requires role count 1")
    handle_query = Chem.MolFromSmarts(handle) if isinstance(handle, str) else None
    if not isinstance(handle, str) or handle_query is None:
        raise AuditError(f"{reaction_id}/{name}: invalid required_handle_smarts")
    if site_multiplicity_semantics not in SUPPORTED_MULTIPLICITY_SEMANTICS:
        raise AuditError(
            f"{reaction_id}/{name}: unsupported site multiplicity semantics "
            f"{site_multiplicity_semantics!r}"
        )
    if (
        site_multiplicity_semantics == SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES
        and handle_query.GetNumAtoms() != 1
    ):
        raise AuditError(
            f"{reaction_id}/{name}: symmetry-qualified multiplicity requires "
            "a single-atom handle"
        )
    if not isinstance(forbidden, list) or any(
        not isinstance(item, str) or Chem.MolFromSmarts(item) is None for item in forbidden
    ):
        raise AuditError(f"{reaction_id}/{name}: invalid forbidden_smarts")
    if (
        not isinstance(multiplicity, list)
        or not multiplicity
        or any(not isinstance(value, int) or value < 1 for value in multiplicity)
    ):
        raise AuditError(f"{reaction_id}/{name}: invalid allowed_site_multiplicity")
    return ReactionRole(
        name=name,
        required_handle_smarts=handle,
        forbidden_smarts=tuple(forbidden),
        allowed_site_multiplicity=tuple(sorted(set(multiplicity))),
        site_multiplicity_semantics=site_multiplicity_semantics,
    )


def load_reaction_definitions(
    registry_paths: Sequence[Path],
    expected_count: int,
    role_policy_overrides: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[ReactionDefinition, ...]:
    """Load all qualified transformations without importing the source repository."""

    override_index = _role_policy_override_index(
        list(role_policy_overrides) if role_policy_overrides is not None else []
    )
    seen_role_keys: set[tuple[str, str]] = set()
    definitions: list[ReactionDefinition] = []
    for path in registry_paths:
        registry = _load_json(path, "reaction registry")
        reactions = registry.get("reactions")
        if not isinstance(reactions, list):
            raise AuditError(f"reaction registry has no reaction list: {path}")
        for raw in reactions:
            if not isinstance(raw, dict):
                raise AuditError(f"reaction registry contains a non-object record: {path}")
            reaction_id = raw.get("reaction_id")
            if not isinstance(reaction_id, str) or not reaction_id:
                raise AuditError(f"reaction registry has an invalid reaction_id: {path}")
            smarts = raw.get("atom_mapped_reaction_smarts")
            selectivity = raw.get("selectivity_policy")
            if raw.get("status") != QUALIFIED_STATUS:
                raise AuditError(f"{reaction_id}: status is not {QUALIFIED_STATUS}")
            if not isinstance(smarts, str) or smarts.count(">>") != 1:
                raise AuditError(
                    f"{reaction_id}: atom-mapped reaction SMARTS is missing or invalid"
                )
            if not isinstance(selectivity, str) or not selectivity.strip():
                raise AuditError(f"{reaction_id}: selectivity_policy is missing")
            roles = raw.get("reactant_roles")
            positives = raw.get("known_positive_examples")
            negatives = raw.get("known_negative_examples")
            if not isinstance(roles, list) or len(roles) < 2:
                raise AuditError(f"{reaction_id}: at least two reactant roles are required")
            if not isinstance(positives, list) or not positives:
                raise AuditError(f"{reaction_id}: known positive examples are required")
            if not isinstance(negatives, list) or not negatives:
                raise AuditError(f"{reaction_id}: known negative examples are required")
            parsed_roles = []
            for role in roles:
                if not isinstance(role, dict):
                    raise AuditError(f"{reaction_id}: reactant role must be an object")
                role_name = role.get("name")
                role_key = (reaction_id, role_name)
                semantics = override_index.get(role_key, {}).get(
                    "site_multiplicity_semantics",
                    RAW_SUBSTRUCTURE_MATCHES,
                )
                parsed_roles.append(_parse_role(role, reaction_id, semantics))
                if isinstance(role_name, str):
                    seen_role_keys.add((reaction_id, role_name))
            definitions.append(
                ReactionDefinition(
                    reaction_id=reaction_id,
                    reaction_version=int(raw["reaction_version"]),
                    status=str(raw["status"]),
                    atom_mapped_reaction_smarts=smarts,
                    selectivity_policy=selectivity,
                    reactant_roles=tuple(parsed_roles),
                    known_positive_examples=tuple(positives),
                    known_negative_examples=tuple(negatives),
                )
            )
    ids = [definition.reaction_id for definition in definitions]
    if len(definitions) != expected_count:
        raise AuditError(
            f"loaded {len(definitions)} qualified reactions; expected {expected_count}"
        )
    if len(ids) != len(set(ids)):
        raise AuditError("qualified registries contain duplicate reaction_id values")
    unresolved_overrides = set(override_index).difference(seen_role_keys)
    if unresolved_overrides:
        rendered = ", ".join(
            f"{reaction_id}/{role}" for reaction_id, role in sorted(unresolved_overrides)
        )
        raise AuditError(f"role-policy overrides did not resolve registry roles: {rendered}")
    return tuple(sorted(definitions, key=lambda definition: definition.reaction_id))


def _compile_reaction(definition: ReactionDefinition) -> CompiledReaction:
    left, right = definition.atom_mapped_reaction_smarts.split(">>")
    forward = rdChemReactions.ReactionFromSmarts(definition.atom_mapped_reaction_smarts)
    reverse = rdChemReactions.ReactionFromSmarts(f"{right}>>{left}")
    if forward is None or reverse is None:
        raise AuditError(f"{definition.reaction_id}: reaction SMARTS did not compile")
    if forward.GetNumReactantTemplates() != len(definition.reactant_roles):
        raise AuditError(
            f"{definition.reaction_id}: transform expects "
            f"{forward.GetNumReactantTemplates()} reactants but registry declares "
            f"{len(definition.reactant_roles)} roles"
        )
    if reverse.GetNumReactantTemplates() != 1:
        raise AuditError(f"{definition.reaction_id}: reverse transform must accept one product")
    handles = tuple(
        Chem.MolFromSmarts(role.required_handle_smarts) for role in definition.reactant_roles
    )
    if any(handle is None for handle in handles):
        raise AuditError(f"{definition.reaction_id}: a required handle did not compile")
    forbidden = tuple(
        tuple(Chem.MolFromSmarts(smarts) for smarts in role.forbidden_smarts)
        for role in definition.reactant_roles
    )
    if any(pattern is None for patterns in forbidden for pattern in patterns):
        raise AuditError(f"{definition.reaction_id}: a forbidden SMARTS did not compile")
    return CompiledReaction(
        definition=definition,
        forward=forward,
        reverse=reverse,
        handles=handles,  # type: ignore[arg-type]
        forbidden=forbidden,  # type: ignore[arg-type]
    )


def compile_reactions(
    definitions: Sequence[ReactionDefinition],
) -> tuple[CompiledReaction, ...]:
    """Compile deterministic forward and reverse RDKit executors."""

    return tuple(_compile_reaction(definition) for definition in definitions)


def _canonicalize_molecule(molecule: Chem.Mol) -> tuple[str, Chem.Mol] | None:
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(molecule)
        if len(Chem.GetMolFrags(molecule)) != 1:
            return None
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
        with rdBase.BlockLogs():
            parsed = Chem.MolFromSmiles(smiles)
        if parsed is None:
            return None
        return smiles, parsed
    except Exception:
        return None


def _forward_products(
    reaction: CompiledReaction,
    reactants: Sequence[Chem.Mol],
    max_outcomes: int,
) -> set[str]:
    with rdBase.BlockLogs():
        outcomes = reaction.forward.RunReactants(tuple(reactants), maxProducts=max_outcomes)
    if len(outcomes) >= max_outcomes:
        raise AuditError(
            f"{reaction.definition.reaction_id}: forward reconstruction reached "
            f"max_outcomes={max_outcomes}"
        )
    products: set[str] = set()
    for outcome in outcomes:
        if len(outcome) != 1:
            continue
        canonicalized = _canonicalize_molecule(outcome[0])
        if canonicalized is not None:
            products.add(canonicalized[0])
    return products


def validate_registry_examples(
    reactions: Sequence[CompiledReaction], max_forward_outcomes: int
) -> dict[str, dict[str, int]]:
    """Require every registry positive and negative example to behave as declared."""

    summary: dict[str, dict[str, int]] = {}
    for reaction in reactions:
        positive_count = 0
        negative_count = 0
        for example in reaction.definition.known_positive_examples:
            reactant_smiles = example.get("reactants")
            expected = example.get("expected")
            if not isinstance(reactant_smiles, list) or not isinstance(expected, str):
                raise AuditError(f"{reaction.definition.reaction_id}: malformed positive example")
            reactants = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles]
            expected_molecule = Chem.MolFromSmiles(expected)
            if expected_molecule is None or any(molecule is None for molecule in reactants):
                raise AuditError(
                    f"{reaction.definition.reaction_id}: invalid positive example SMILES"
                )
            expected_canonical = Chem.MolToSmiles(
                expected_molecule, canonical=True, isomericSmiles=True
            )
            products = _forward_products(
                reaction,
                reactants,  # type: ignore[arg-type]
                max_forward_outcomes,
            )
            if expected_canonical not in products:
                raise AuditError(
                    f"{reaction.definition.reaction_id}: positive example did not reconstruct"
                )
            positive_count += 1
        for example in reaction.definition.known_negative_examples:
            reactant_smiles = example.get("reactants")
            if not isinstance(reactant_smiles, list):
                raise AuditError(f"{reaction.definition.reaction_id}: malformed negative example")
            reactants = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles]
            if any(molecule is None for molecule in reactants):
                raise AuditError(
                    f"{reaction.definition.reaction_id}: invalid negative example SMILES"
                )
            products = _forward_products(
                reaction,
                reactants,  # type: ignore[arg-type]
                max_forward_outcomes,
            )
            if products:
                raise AuditError(
                    f"{reaction.definition.reaction_id}: negative example produced a product"
                )
            negative_count += 1
        summary[reaction.definition.reaction_id] = {
            "positive_examples_passed": positive_count,
            "negative_examples_passed": negative_count,
        }
    return summary


def _role_accepts(
    molecule: Chem.Mol,
    role: ReactionRole,
    handle: Chem.Mol,
    forbidden: Sequence[Chem.Mol],
) -> bool:
    if role.site_multiplicity_semantics == RAW_SUBSTRUCTURE_MATCHES:
        multiplicity = len(molecule.GetSubstructMatches(handle, uniquify=True))
    else:
        multiplicity = audit_reactive_site_multiplicity(
            molecule,
            handle,
        ).count(role.site_multiplicity_semantics)
    if multiplicity not in role.allowed_site_multiplicity:
        return False
    return not any(molecule.HasSubstructMatch(pattern) for pattern in forbidden)


def _source_study_labels(row: Mapping[str, str]) -> tuple[str, ...]:
    try:
        studies = json.loads(row["study_split_groups_json"])
    except json.JSONDecodeError as exc:
        raise AuditError(
            f"{row['r0_structure_id']} has invalid study_split_groups_json: {exc}"
        ) from exc
    if not isinstance(studies, dict):
        raise AuditError(f"{row['r0_structure_id']} study_split_groups_json must be an object")
    labels: set[str] = set()
    represented_sources: set[str] = set()
    for source, values in studies.items():
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise AuditError(f"{row['r0_structure_id']} has malformed study values for {source}")
        clean_values = [value.strip() for value in values if value.strip()]
        labels.update(f"{source}:{value}" for value in clean_values)
        if clean_values:
            represented_sources.add(source)
    sources = {source for source in row["observed_source_ids"].split("|") if source}
    labels.update(f"fallback-source:{source}" for source in sources.difference(represented_sources))
    if not labels:
        raise AuditError(f"{row['r0_structure_id']} has no source-study provenance")
    return tuple(sorted(labels))


def decompose_structure(
    row: Mapping[str, str],
    scheme: str,
    reactions: Sequence[CompiledReaction],
    max_reverse_outcomes: int,
    max_forward_outcomes: int,
    rejection_counts: Counter[str] | None = None,
) -> list[DecompositionCandidate]:
    """Return exact forward-reconstructing candidates for one training structure."""

    if scheme not in SCHEMES:
        raise AuditError(f"unknown M0-04 scheme {scheme!r}")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
    if molecule is None:
        raise AuditError(f"{row['r0_structure_id']} has an invalid canonical structure")
    target = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    studies = _source_study_labels(row)
    candidates: list[DecompositionCandidate] = []
    rejections = rejection_counts if rejection_counts is not None else Counter()
    for reaction in reactions:
        reaction_id = reaction.definition.reaction_id
        with rdBase.BlockLogs():
            outcomes = reaction.reverse.RunReactants((molecule,), maxProducts=max_reverse_outcomes)
        if len(outcomes) >= max_reverse_outcomes:
            raise AuditError(
                f"{reaction_id}/{row['r0_structure_id']}: reverse decomposition reached "
                f"max_outcomes={max_reverse_outcomes}"
            )
        seen: set[tuple[str, ...]] = set()
        for outcome in outcomes:
            if len(outcome) != len(reaction.definition.reactant_roles):
                rejections[f"{reaction_id}:wrong_product_count"] += 1
                continue
            reactant_smiles: list[str] = []
            reactant_molecules: list[Chem.Mol] = []
            valid = True
            for index, raw_reactant in enumerate(outcome):
                canonicalized = _canonicalize_molecule(raw_reactant)
                if canonicalized is None:
                    rejections[f"{reaction_id}:invalid_reactant"] += 1
                    valid = False
                    break
                smiles, parsed = canonicalized
                if not _role_accepts(
                    parsed,
                    reaction.definition.reactant_roles[index],
                    reaction.handles[index],
                    reaction.forbidden[index],
                ):
                    rejections[f"{reaction_id}:role_policy"] += 1
                    valid = False
                    break
                reactant_smiles.append(smiles)
                reactant_molecules.append(parsed)
            key = tuple(reactant_smiles)
            if not valid or key in seen:
                continue
            seen.add(key)
            reconstructed = _forward_products(reaction, reactant_molecules, max_forward_outcomes)
            if target not in reconstructed:
                rejections[f"{reaction_id}:no_exact_reconstruction"] += 1
                continue
            candidates.append(
                DecompositionCandidate(
                    scheme=scheme,
                    source_structure_id=row["r0_structure_id"],
                    reaction_id=reaction_id,
                    reactant_smiles=key,
                    source_studies=studies,
                )
            )
    return candidates


def _read_r0_rows(path: Path, expected_rows: int) -> dict[str, dict[str, str]]:
    required = {
        "r0_structure_id",
        "canonical_isomeric_smiles",
        "observed_source_ids",
        "study_split_groups_json",
    }
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        handle = opener(path, "rt", newline="")
    except FileNotFoundError as exc:
        raise AuditError(f"R0 corpus not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise AuditError(f"R0 corpus is missing columns: {missing}")
        rows = list(reader)
    if len(rows) != expected_rows:
        raise AuditError(f"R0 corpus has {len(rows)} rows; expected {expected_rows}")
    by_id = {row["r0_structure_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise AuditError("R0 corpus contains duplicate r0_structure_id values")
    return by_id


def build_block_records(
    candidates: Sequence[DecompositionCandidate],
    reactions: Sequence[ReactionDefinition],
) -> tuple[BlockRecord, ...]:
    """Aggregate role-typed block provenance from decomposition candidates."""

    roles_by_reaction = {
        reaction.reaction_id: tuple(role.name for role in reaction.reactant_roles)
        for reaction in reactions
    }
    route_occurrences: Counter[tuple[str, str, str, str]] = Counter()
    source_structures: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    source_studies: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for candidate in candidates:
        roles = roles_by_reaction[candidate.reaction_id]
        if len(roles) != len(candidate.reactant_smiles):
            raise AuditError(f"{candidate.reaction_id}: candidate role count mismatch")
        for role, smiles in zip(roles, candidate.reactant_smiles, strict=True):
            key = (candidate.scheme, candidate.reaction_id, role, smiles)
            route_occurrences[key] += 1
            source_structures[key].add(candidate.source_structure_id)
            source_studies[key].update(candidate.source_studies)
    records = []
    for key in sorted(route_occurrences):
        scheme, reaction_id, role, smiles = key
        block_id = f"r0d-{hashlib.sha256(smiles.encode()).hexdigest()[:20]}"
        records.append(
            BlockRecord(
                scheme=scheme,
                block_id=block_id,
                reaction_id=reaction_id,
                role=role,
                canonical_smiles=smiles,
                route_occurrence_count=route_occurrences[key],
                source_structure_count=len(source_structures[key]),
                source_studies=tuple(sorted(source_studies[key])),
            )
        )
    return tuple(records)


def decompose_scheme(
    scheme: str,
    r0_rows: Mapping[str, Mapping[str, str]],
    train_ids: set[str],
    heldout_ids: set[str],
    definitions: Sequence[ReactionDefinition],
    max_reverse_outcomes: int,
    max_forward_outcomes: int,
) -> tuple[tuple[DecompositionCandidate, ...], tuple[BlockRecord, ...], dict[str, Any]]:
    """Decompose only the frozen R0 training fold for one scheme."""

    if train_ids & heldout_ids:
        raise AuditError(f"{scheme}: train and heldout IDs overlap")
    missing = sorted(train_ids.difference(r0_rows))
    if missing:
        raise AuditError(f"{scheme}: train ID missing from R0 corpus: {missing[0]}")
    reactions = compile_reactions(definitions)
    candidates: list[DecompositionCandidate] = []
    rejections: Counter[str] = Counter()
    structures_with_candidates: set[str] = set()
    for structure_id in sorted(train_ids):
        row_candidates = decompose_structure(
            r0_rows[structure_id],
            scheme,
            reactions,
            max_reverse_outcomes,
            max_forward_outcomes,
            rejections,
        )
        if row_candidates:
            structures_with_candidates.add(structure_id)
            candidates.extend(row_candidates)
    if any(candidate.source_structure_id not in train_ids for candidate in candidates):
        raise AuditError(f"{scheme}: a decomposition candidate escaped the training fold")
    blocks = build_block_records(candidates, definitions)
    by_reaction = Counter(candidate.reaction_id for candidate in candidates)
    summary = {
        "train_structures": len(train_ids),
        "heldout_structures": len(heldout_ids),
        "train_structures_with_candidate_decomposition": len(structures_with_candidates),
        "candidate_decompositions": len(candidates),
        "candidate_decompositions_by_reaction": dict(sorted(by_reaction.items())),
        "role_typed_pool_records": len(blocks),
        "unique_block_structures": len({block.block_id for block in blocks}),
        "rejected_reverse_outcomes": dict(sorted(rejections.items())),
        "heldout_structures_decomposed_for_harvesting": 0,
        "all_harvest_sources_are_r0_train": True,
    }
    return tuple(candidates), blocks, summary


def _decompose_scheme_worker(
    scheme: str,
    r0_path: str,
    expected_rows: int,
    train_ids: tuple[str, ...],
    heldout_ids: tuple[str, ...],
    definitions: tuple[ReactionDefinition, ...],
    max_reverse_outcomes: int,
    max_forward_outcomes: int,
) -> tuple[str, tuple[DecompositionCandidate, ...], tuple[BlockRecord, ...], dict[str, Any]]:
    rows = _read_r0_rows(Path(r0_path), expected_rows)
    candidates, blocks, summary = decompose_scheme(
        scheme,
        rows,
        set(train_ids),
        set(heldout_ids),
        definitions,
        max_reverse_outcomes,
        max_forward_outcomes,
    )
    return scheme, candidates, blocks, summary


class ProductIndex:
    """Temporary disk-backed exact index for one scheme's R1-prime products."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=OFF")
        self.connection.execute("PRAGMA synchronous=OFF")
        self.connection.execute("PRAGMA temp_store=MEMORY")
        self.connection.execute(
            "CREATE TABLE products (canonical_smiles TEXT PRIMARY KEY) WITHOUT ROWID"
        )
        self.connection.execute("""
            CREATE TABLE product_families (
                canonical_smiles TEXT NOT NULL,
                reaction_id TEXT NOT NULL,
                PRIMARY KEY (canonical_smiles, reaction_id)
            ) WITHOUT ROWID
            """)

    def add(self, canonical_smiles: str, reaction_id: str) -> bool:
        cursor = self.connection.execute(
            "INSERT OR IGNORE INTO products VALUES (?)", (canonical_smiles,)
        )
        self.connection.execute(
            "INSERT OR IGNORE INTO product_families VALUES (?, ?)",
            (canonical_smiles, reaction_id),
        )
        return cursor.rowcount == 1

    def commit(self) -> None:
        self.connection.commit()

    def count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def family_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT reaction_id, COUNT(*) FROM product_families GROUP BY reaction_id"
        )
        return {str(reaction_id): int(count) for reaction_id, count in rows}

    def contains(self, canonical_smiles: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM products WHERE canonical_smiles = ?", (canonical_smiles,)
        ).fetchone()
        return row is not None

    def smiles_batches(self, batch_size: int) -> Iterator[list[str]]:
        cursor = self.connection.execute(
            "SELECT canonical_smiles FROM products ORDER BY canonical_smiles"
        )
        while rows := cursor.fetchmany(batch_size):
            yield [str(row[0]) for row in rows]

    def close(self) -> None:
        self.connection.close()


def enumerate_r1_prime(
    scheme: str,
    block_records: Sequence[BlockRecord],
    definitions: Sequence[ReactionDefinition],
    database_path: Path,
    max_combinations: int,
    commit_interval: int,
) -> tuple[ProductIndex, dict[str, Any]]:
    """Exhaustively enumerate train-derived pools without sampling combinations."""

    blocks_by_reaction_role: dict[tuple[str, str], dict[str, Chem.Mol]] = defaultdict(dict)
    for block in block_records:
        if block.scheme != scheme:
            raise AuditError(f"{scheme}: received block from scheme {block.scheme}")
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(block.canonical_smiles)
        if molecule is None:
            raise AuditError(f"{scheme}: invalid pooled block {block.block_id}")
        blocks_by_reaction_role[(block.reaction_id, block.role)][block.canonical_smiles] = molecule

    compiled = {
        reaction.definition.reaction_id: reaction for reaction in compile_reactions(definitions)
    }
    predicted_by_reaction: dict[str, int] = {}
    total_combinations = 0
    for definition in definitions:
        sizes = [
            len(blocks_by_reaction_role[(definition.reaction_id, role.name)])
            for role in definition.reactant_roles
        ]
        combinations = math.prod(sizes)
        predicted_by_reaction[definition.reaction_id] = combinations
        total_combinations += combinations
    if total_combinations > max_combinations:
        raise AuditError(
            f"{scheme}: predicted {total_combinations:,} reactant combinations exceeds "
            f"the declared safety limit {max_combinations:,}; do not sample or truncate"
        )

    index = ProductIndex(database_path)
    generated_outcomes: Counter[str] = Counter()
    invalid_products: Counter[str] = Counter()
    new_products: Counter[str] = Counter()
    combinations_processed: Counter[str] = Counter()
    since_commit = 0
    try:
        for definition in definitions:
            reaction_id = definition.reaction_id
            reaction = compiled[reaction_id]
            role_blocks = [
                [
                    blocks_by_reaction_role[(reaction_id, role.name)][smiles]
                    for smiles in sorted(blocks_by_reaction_role[(reaction_id, role.name)])
                ]
                for role in definition.reactant_roles
            ]
            if any(not blocks for blocks in role_blocks):
                continue
            for reactants in itertools.product(*role_blocks):
                combinations_processed[reaction_id] += 1
                with rdBase.BlockLogs():
                    outcomes = reaction.forward.RunReactants(reactants)
                for outcome in outcomes:
                    generated_outcomes[reaction_id] += 1
                    if len(outcome) != 1:
                        invalid_products[reaction_id] += 1
                        continue
                    canonicalized = _canonicalize_molecule(outcome[0])
                    if canonicalized is None:
                        invalid_products[reaction_id] += 1
                        continue
                    if index.add(canonicalized[0], reaction_id):
                        new_products[reaction_id] += 1
                since_commit += 1
                if since_commit >= commit_interval:
                    index.commit()
                    since_commit = 0
        index.commit()
        observed_combinations = sum(combinations_processed.values())
        if observed_combinations != total_combinations:
            raise AuditError(
                f"{scheme}: processed {observed_combinations} combinations but predicted "
                f"{total_combinations}"
            )
        summary = {
            "predicted_reactant_combinations": total_combinations,
            "reactant_combinations_by_reaction": dict(sorted(predicted_by_reaction.items())),
            "reactant_combinations_processed": observed_combinations,
            "generated_product_outcomes": sum(generated_outcomes.values()),
            "generated_product_outcomes_by_reaction": dict(sorted(generated_outcomes.items())),
            "invalid_product_outcomes_by_reaction": dict(sorted(invalid_products.items())),
            "globally_unique_products": index.count(),
            "unique_products_by_reaction_nonexclusive": dict(sorted(index.family_counts().items())),
            "first_seen_unique_products_by_reaction": dict(sorted(new_products.items())),
            "sampling_used": False,
            "provenance": "r0_train_derived_reaction_enumerated_reference",
            "generator_role": "diagnostic_only_not_the_whole_graph_product_generator",
        }
        return index, summary
    except Exception:
        index.close()
        raise


def _is_agile(row: Mapping[str, str]) -> bool:
    return AGILE_SOURCE in row["observed_source_ids"].split("|")


def load_r1_control(path: Path) -> set[str]:
    """Load the existing R1 reaction-enumerated support as an exact control set."""

    try:
        handle = path.open(newline="")
    except FileNotFoundError as exc:
        raise AuditError(f"R1 control not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "canonical_smiles" not in reader.fieldnames:
            raise AuditError("R1 control CSV lacks canonical_smiles")
        structures = {row["canonical_smiles"] for row in reader if row["canonical_smiles"]}
    if not structures:
        raise AuditError("R1 control contains no structures")
    return structures


def reproduce_non_agile_control(
    r0_rows: Mapping[str, Mapping[str, str]],
    r1_structures: set[str],
    expected_non_agile: int,
    expected_recovered: int,
) -> dict[str, Any]:
    """Mechanically reproduce the frozen 108/14,233 non-AGILE control."""

    non_agile = [row for row in r0_rows.values() if not _is_agile(row)]
    recovered = sum(row["canonical_isomeric_smiles"] in r1_structures for row in non_agile)
    if len(non_agile) != expected_non_agile or recovered != expected_recovered:
        raise AuditError(
            f"R1 control mismatch: observed {recovered}/{len(non_agile)}, expected "
            f"{expected_recovered}/{expected_non_agile}"
        )
    return {
        "recovered": recovered,
        "denominator": len(non_agile),
        "recovery_rate": recovered / len(non_agile),
        "expected_recovered": expected_recovered,
        "expected_denominator": expected_non_agile,
        "reproduced": True,
    }


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> list[float | None]:
    if total == 0:
        return [None, None]
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def _recovery_summary(recovered: int, denominator: int) -> dict[str, Any]:
    return {
        "recovered": recovered,
        "denominator": denominator,
        "recovery_rate": recovered / denominator if denominator else None,
        "wilson_95_interval": _wilson_interval(recovered, denominator),
    }


def exact_recovery_against_index(
    heldout_rows: Sequence[Mapping[str, str]], index: ProductIndex
) -> tuple[dict[str, Any], list[Mapping[str, str]]]:
    """Compute exact heldout recovery and return unrecovered rows."""

    recovered_rows = [
        row for row in heldout_rows if index.contains(row["canonical_isomeric_smiles"])
    ]
    unrecovered = [
        row for row in heldout_rows if not index.contains(row["canonical_isomeric_smiles"])
    ]
    non_agile = [row for row in heldout_rows if not _is_agile(row)]
    non_agile_recovered = sum(index.contains(row["canonical_isomeric_smiles"]) for row in non_agile)
    agile = [row for row in heldout_rows if _is_agile(row)]
    agile_recovered = sum(index.contains(row["canonical_isomeric_smiles"]) for row in agile)
    return (
        {
            "all_heldout": _recovery_summary(len(recovered_rows), len(heldout_rows)),
            "non_agile_heldout": _recovery_summary(non_agile_recovered, len(non_agile)),
            "agile_heldout": _recovery_summary(agile_recovered, len(agile)),
        },
        unrecovered,
    )


def exact_control_by_scheme(
    heldout_rows: Sequence[Mapping[str, str]], r1_structures: set[str]
) -> dict[str, Any]:
    """Report existing-R1 exact recovery on one M0-03 heldout fold."""

    recovered = sum(row["canonical_isomeric_smiles"] in r1_structures for row in heldout_rows)
    non_agile = [row for row in heldout_rows if not _is_agile(row)]
    non_agile_recovered = sum(
        row["canonical_isomeric_smiles"] in r1_structures for row in non_agile
    )
    agile = [row for row in heldout_rows if _is_agile(row)]
    agile_recovered = sum(row["canonical_isomeric_smiles"] in r1_structures for row in agile)
    return {
        "all_heldout": _recovery_summary(recovered, len(heldout_rows)),
        "non_agile_heldout": _recovery_summary(non_agile_recovered, len(non_agile)),
        "agile_heldout": _recovery_summary(agile_recovered, len(agile)),
    }


def _best_similarities_for_slice(
    query_fingerprints: Sequence[DataStructs.ExplicitBitVect],
    reference_fingerprints: Sequence[DataStructs.ExplicitBitVect],
) -> list[float]:
    neighbors = DataStructs.TanimotoSimilarityNeighbors(
        query_fingerprints,
        reference_fingerprints,
    )
    return [float(similarity) for _, similarity in neighbors]


def nearest_reachable_distances(
    unrecovered_rows: Sequence[Mapping[str, str]],
    index: ProductIndex,
    radius: int,
    bits: int,
    include_chirality: bool,
    reference_batch_size: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Compute exact nearest ECFP4 Tanimoto distance with bounded memory."""

    if not unrecovered_rows:
        return []
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=bits,
        includeChirality=include_chirality,
    )
    query_fingerprints = []
    for row in unrecovered_rows:
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
        if molecule is None:
            raise AuditError(f"{row['r0_structure_id']}: invalid heldout structure")
        query_fingerprints.append(generator.GetFingerprint(molecule))
    best = [0.0] * len(query_fingerprints)
    worker_count = max(1, min(workers, len(query_fingerprints)))
    chunk_size = math.ceil(len(query_fingerprints) / worker_count)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        for smiles_batch in index.smiles_batches(reference_batch_size):
            reference_fingerprints = []
            for smiles in smiles_batch:
                with rdBase.BlockLogs():
                    molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    raise AuditError("R1-prime index contains an invalid generated structure")
                reference_fingerprints.append(generator.GetFingerprint(molecule))
            futures = []
            slices = []
            for start in range(0, len(query_fingerprints), chunk_size):
                stop = min(start + chunk_size, len(query_fingerprints))
                slices.append((start, stop))
                futures.append(
                    executor.submit(
                        _best_similarities_for_slice,
                        query_fingerprints[start:stop],
                        reference_fingerprints,
                    )
                )
            for (start, stop), future in zip(slices, futures, strict=True):
                batch_best = future.result()
                for local_index, similarity in enumerate(batch_best):
                    index_value = start + local_index
                    if similarity > best[index_value]:
                        best[index_value] = similarity
                if len(batch_best) != stop - start:
                    raise AuditError("nearest-neighbor worker returned an unexpected result count")
    return [
        {
            "r0_structure_id": row["r0_structure_id"],
            "is_agile": _is_agile(row),
            "max_tanimoto_similarity": similarity,
            "nearest_reachable_distance": 1.0 - similarity,
        }
        for row, similarity in zip(unrecovered_rows, best, strict=True)
    ]


def summarize_distribution(values: Sequence[float]) -> dict[str, Any]:
    """Return a full histogram and standard quantiles for a numeric distribution."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "standard_deviation": None,
            "quantiles": {},
            "histogram": [],
        }
    ordered = sorted(values)

    def quantile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    edges = [index / 20 for index in range(21)]
    counts = [0] * 20
    for value in ordered:
        bin_index = min(19, int(value * 20))
        counts[bin_index] += 1
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "standard_deviation": statistics.pstdev(ordered),
        "quantiles": {
            key: quantile(probability)
            for key, probability in (
                ("min", 0.0),
                ("q01", 0.01),
                ("q05", 0.05),
                ("q10", 0.10),
                ("q25", 0.25),
                ("median", 0.50),
                ("q75", 0.75),
                ("q90", 0.90),
                ("q95", 0.95),
                ("q99", 0.99),
                ("max", 1.0),
            )
        },
        "histogram": [
            {"lower": edges[index], "upper": edges[index + 1], "count": count}
            for index, count in enumerate(counts)
        ],
    }


def _gzip_csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(csv_buffer.getvalue().encode())
    return output.getvalue()


def _plain_csv_bytes(rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _serialize_decompositions(candidates: Sequence[DecompositionCandidate]) -> bytes:
    rows = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.scheme,
            item.source_structure_id,
            item.reaction_id,
            item.reactant_smiles,
        ),
    ):
        route_payload = "\0".join(
            (candidate.scheme, candidate.source_structure_id, candidate.reaction_id)
            + candidate.reactant_smiles
        )
        rows.append(
            {
                "scheme": candidate.scheme,
                "decomposition_id": (
                    f"decomp-{hashlib.sha256(route_payload.encode()).hexdigest()[:20]}"
                ),
                "source_structure_id": candidate.source_structure_id,
                "reaction_id": candidate.reaction_id,
                "reactant_smiles_json": json.dumps(candidate.reactant_smiles),
                "source_studies_json": json.dumps(candidate.source_studies),
                "provenance": "r0_train_derived",
                "precision_status": "pending_m0_05_human_audit",
            }
        )
    return _gzip_csv_bytes(
        rows,
        (
            "scheme",
            "decomposition_id",
            "source_structure_id",
            "reaction_id",
            "reactant_smiles_json",
            "source_studies_json",
            "provenance",
            "precision_status",
        ),
    )


def _serialize_blocks(records: Sequence[BlockRecord]) -> bytes:
    rows = [
        {
            "scheme": block.scheme,
            "block_id": block.block_id,
            "reaction_id": block.reaction_id,
            "role": block.role,
            "canonical_smiles": block.canonical_smiles,
            "route_occurrence_count": block.route_occurrence_count,
            "source_structure_count": block.source_structure_count,
            "source_studies_json": json.dumps(block.source_studies),
            "provenance": "r0_derived",
        }
        for block in sorted(
            records,
            key=lambda item: (
                item.scheme,
                item.reaction_id,
                item.role,
                item.canonical_smiles,
            ),
        )
    ]
    return _gzip_csv_bytes(
        rows,
        (
            "scheme",
            "block_id",
            "reaction_id",
            "role",
            "canonical_smiles",
            "route_occurrence_count",
            "source_structure_count",
            "source_studies_json",
            "provenance",
        ),
    )


def _serialize_distances(rows: Sequence[Mapping[str, Any]]) -> bytes:
    ordered = sorted(rows, key=lambda row: (str(row["scheme"]), str(row["r0_structure_id"])))
    return _plain_csv_bytes(
        ordered,
        (
            "scheme",
            "r0_structure_id",
            "is_agile",
            "max_tanimoto_similarity",
            "nearest_reachable_distance",
        ),
    )


def _artifact_record(path: str, payload: bytes) -> dict[str, Any]:
    return {"path": path, "bytes": len(payload), "sha256": sha256_bytes(payload)}


def _scheme_checkpoint_bytes(value: Mapping[str, Any]) -> bytes:
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        compressed.write(_json_bytes(value))
    return output.getvalue()


def _load_scheme_checkpoint(
    path: Path,
    run_signature: str,
    scheme: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rt") as handle:
            payload = json.load(handle)
    except (gzip.BadGzipFile, json.JSONDecodeError, OSError) as exc:
        raise AuditError(f"invalid M0-04 checkpoint {path}: {exc}") from exc
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
        or payload.get("algorithm_version") != ALGORITHM_VERSION
        or payload.get("run_signature") != run_signature
        or payload.get("scheme") != scheme
    ):
        raise AuditError(f"M0-04 checkpoint metadata mismatch: {path}")
    result = payload.get("result")
    distances = payload.get("distances")
    if not isinstance(result, dict) or not isinstance(distances, list):
        raise AuditError(f"M0-04 checkpoint payload is malformed: {path}")
    return result, distances


def _write_scheme_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_scheme_checkpoint_bytes(payload))
    os.replace(temporary, path)


def _audit_scheme_worker(
    task: tuple[
        str,
        tuple[BlockRecord, ...],
        tuple[ReactionDefinition, ...],
        str,
        int,
        int,
        tuple[Mapping[str, str], ...],
        Mapping[str, Any],
        Mapping[str, Any],
        int,
        str,
        str,
    ],
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    (
        scheme,
        blocks,
        definitions,
        database_path_raw,
        max_combinations,
        commit_interval,
        heldout_rows,
        decomposition_summary,
        fingerprint,
        neighbor_workers,
        checkpoint_path_raw,
        run_signature,
    ) = task
    checkpoint_path = Path(checkpoint_path_raw)
    checkpoint = _load_scheme_checkpoint(checkpoint_path, run_signature, scheme)
    if checkpoint is not None:
        print(f"M0-04 {scheme}: reused completed checkpoint", flush=True)
        result, distances = checkpoint
        return scheme, result, distances

    started = time.monotonic()
    print(f"M0-04 {scheme}: enumerating exact R1-prime support", flush=True)
    database_path = Path(database_path_raw)
    index, enumeration_summary = enumerate_r1_prime(
        scheme,
        blocks,
        definitions,
        database_path,
        max_combinations,
        commit_interval,
    )
    try:
        recovery, unrecovered = exact_recovery_against_index(heldout_rows, index)
        print(
            f"M0-04 {scheme}: exact support complete; scoring "
            f"{len(unrecovered):,} unrecovered structures",
            flush=True,
        )
        distance_rows = nearest_reachable_distances(
            unrecovered,
            index,
            fingerprint["radius"],
            fingerprint["bits"],
            fingerprint["include_chirality"],
            fingerprint["reference_batch_size"],
            neighbor_workers,
        )
        all_values = [float(row["nearest_reachable_distance"]) for row in distance_rows]
        non_agile_values = [
            float(row["nearest_reachable_distance"]) for row in distance_rows if not row["is_agile"]
        ]
        result = {
            "decomposition": dict(decomposition_summary),
            "r1_prime_enumeration": enumeration_summary,
            "r1_prime_exact_recovery": recovery,
            "r1_prime_nearest_reachable_distance": {
                "metric": "1 - maximum ECFP4 Tanimoto similarity",
                "all_unrecovered": summarize_distribution(all_values),
                "non_agile_unrecovered": summarize_distribution(non_agile_values),
            },
        }
    finally:
        index.close()

    checkpoint_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "run_signature": run_signature,
        "scheme": scheme,
        "result": result,
        "distances": distance_rows,
    }
    _write_scheme_checkpoint(checkpoint_path, checkpoint_payload)
    database_path.unlink(missing_ok=True)
    print(
        f"M0-04 {scheme}: checkpoint complete in {(time.monotonic() - started) / 60:.1f} min",
        flush=True,
    )
    return scheme, result, distance_rows


def run_audit(
    config_path: Path,
    vendor_dir: Path,
    split_dir: Path,
    output_dir: Path,
    workers: int,
    checkpoint_dir: Path | None = None,
) -> dict[str, Any]:
    """Run all four M0-04 schemes and atomically write the completed audit."""

    if output_dir.exists():
        raise AuditError(
            f"refusing to overwrite existing M0-04 output {output_dir}; move it or use a new path"
        )
    if workers < 1:
        raise AuditError("workers must be at least one")
    config, input_records, input_locations = verify_inputs(
        config_path,
        vendor_dir,
        split_dir,
    )
    repository_root = config_path.resolve().parents[2]
    if config["schema_version"] == CONFIG_SCHEMA_VERSION:
        dataset = config["dataset"]
        r0_path = input_locations[dataset["current_r0_asset"]]
        historical_r0_path = input_locations[dataset["historical_r0_asset"]]
        r1_control_path = input_locations[dataset["r1_control_asset"]]
        assignments_path = input_locations[dataset["split_assignments_asset"]]
        manifest_path = input_locations[dataset["split_manifest_asset"]]
        if assignments_path.parent != manifest_path.parent:
            raise AuditError("v3 split assignments and manifest must share one directory")
        runtime_split_dir = assignments_path.parent
        reaction_paths = (
            input_locations[dataset["reaction_families_asset"]],
            input_locations[dataset["reactions_asset"]],
        )
    else:
        r0_path = vendor_dir / "r0_observed_real_structures.csv"
        historical_r0_path = r0_path
        r1_control_path = vendor_dir / "r1_reaction_enumerated_support_v1.csv"
        runtime_split_dir = split_dir
        reaction_paths = (
            vendor_dir / "qualified_reaction_families_v1.json",
            vendor_dir / "qualified_reactions_v1.json",
        )
    definitions = load_reaction_definitions(
        reaction_paths,
        config["expected_reaction_count"],
        config["role_policy_overrides"],
    )
    compiled = compile_reactions(definitions)
    registry_validation = validate_registry_examples(
        compiled,
        config["decomposition"]["max_forward_outcomes_per_reconstruction"],
    )
    frozen = load_frozen_r0_splits(runtime_split_dir)
    r0_rows = _read_r0_rows(r0_path, config["expected_r0_rows"])
    frozen_ids = {row["r0_structure_id"] for row in frozen.assignments}
    if frozen_ids != set(r0_rows):
        raise AuditError("frozen M0-03 IDs do not match the R0 corpus")

    fold_ids = {
        scheme: {fold: set(frozen.ids(scheme, fold)) for fold in FOLDS} for scheme in SCHEMES
    }
    tasks = [
        (
            scheme,
            str(r0_path),
            config["expected_r0_rows"],
            tuple(sorted(fold_ids[scheme]["R0_train"])),
            tuple(sorted(fold_ids[scheme]["R0_heldout"])),
            definitions,
            config["decomposition"]["max_reverse_outcomes_per_structure_reaction"],
            config["decomposition"]["max_forward_outcomes_per_reconstruction"],
        )
        for scheme in SCHEMES
    ]
    decompositions_by_scheme: dict[str, tuple[DecompositionCandidate, ...]] = {}
    blocks_by_scheme: dict[str, tuple[BlockRecord, ...]] = {}
    decomposition_summaries: dict[str, dict[str, Any]] = {}
    if workers == 1:
        decomposition_results = [_decompose_scheme_worker(*task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(SCHEMES))) as executor:
            decomposition_results = list(executor.map(_run_worker_task, tasks))
    for scheme, candidates, blocks, summary in decomposition_results:
        decompositions_by_scheme[scheme] = candidates
        blocks_by_scheme[scheme] = blocks
        decomposition_summaries[scheme] = summary

    r1_control = load_r1_control(r1_control_path)
    if config["schema_version"] == CONFIG_SCHEMA_VERSION:
        controls = config["controls"]
        historical_r0_rows = _read_r0_rows(
            historical_r0_path,
            config["expected_historical_r0_rows"],
        )
        historical_control = reproduce_non_agile_control(
            historical_r0_rows,
            r1_control,
            controls["historical_non_agile_rows"],
            controls["historical_non_agile_recovered"],
        )
        current_control = reproduce_non_agile_control(
            r0_rows,
            r1_control,
            controls["current_non_agile_rows"],
            controls["current_non_agile_recovered"],
        )
    else:
        historical_control = reproduce_non_agile_control(
            r0_rows,
            r1_control,
            config["expected_non_agile_r0_rows"],
            config["expected_control_non_agile_recovered"],
        )
        current_control = historical_control

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    checkpoint_root = checkpoint_dir or output_dir.parents[1] / "checkpoints/m0_04"
    run_signature = hashlib.sha256(
        _json_bytes(
            {
                "algorithm_version": ALGORITHM_VERSION,
                "inputs": input_records,
                "schemes": list(SCHEMES),
            }
        )
    ).hexdigest()[:24]
    checkpoint_run_dir = checkpoint_root / run_signature
    all_distances: list[dict[str, Any]] = []
    scheme_results: dict[str, Any] = {}
    try:
        work_dir = staging / ".work"
        work_dir.mkdir()
        scheme_worker_count = min(workers, len(SCHEMES))
        neighbor_workers = max(1, workers // scheme_worker_count)
        scheme_tasks = [
            (
                scheme,
                blocks_by_scheme[scheme],
                definitions,
                str(work_dir / f"{scheme}.sqlite"),
                config["enumeration"]["max_reactant_combinations_per_scheme"],
                config["enumeration"]["sqlite_commit_interval"],
                tuple(
                    r0_rows[structure_id] for structure_id in sorted(fold_ids[scheme]["R0_heldout"])
                ),
                decomposition_summaries[scheme],
                config["fingerprint"],
                neighbor_workers,
                str(checkpoint_run_dir / f"{scheme}.json.gz"),
                run_signature,
            )
            for scheme in SCHEMES
        ]
        if scheme_worker_count == 1:
            completed_schemes = [_audit_scheme_worker(task) for task in scheme_tasks]
        else:
            with ProcessPoolExecutor(max_workers=scheme_worker_count) as executor:
                completed_schemes = list(executor.map(_audit_scheme_worker, scheme_tasks))
        for scheme, scheme_result, distance_rows in completed_schemes:
            heldout_rows = [
                r0_rows[structure_id] for structure_id in sorted(fold_ids[scheme]["R0_heldout"])
            ]
            scheme_result["existing_r1_exact_control"] = exact_control_by_scheme(
                heldout_rows,
                r1_control,
            )
            scheme_results[scheme] = scheme_result
            all_distances.extend({"scheme": scheme, **row} for row in distance_rows)

        all_candidates = tuple(
            candidate for scheme in SCHEMES for candidate in decompositions_by_scheme[scheme]
        )
        all_blocks = tuple(block for scheme in SCHEMES for block in blocks_by_scheme[scheme])
        decomposition_payload = _serialize_decompositions(all_candidates)
        block_payload = _serialize_blocks(all_blocks)
        distance_payload = _serialize_distances(all_distances)
        artifact_payloads = {
            "decomposition_candidates.csv.gz": decomposition_payload,
            "component_pool.csv.gz": block_payload,
            "nearest_reachable_distances.csv": distance_payload,
        }
        try:
            output_relative = output_dir.resolve().relative_to(repository_root)
        except ValueError as exc:
            raise AuditError("M0-04 output directory must be inside the repository") from exc
        artifact_records = {
            name: _artifact_record(str(output_relative / name), payload)
            for name, payload in artifact_payloads.items()
        }
        result: dict[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task": "M0-04",
            "generated_utc": config["generated_utc"],
            "seed": config["seed"],
            "randomness_used": False,
            "software": {
                "python": platform.python_version(),
                "rdkit": rdBase.rdkitVersion,
                "sqlite": sqlite3.sqlite_version,
            },
            "inputs": input_records,
            "parameters": {
                "schemes": list(SCHEMES),
                "workers": workers,
                "scheme_processes": scheme_worker_count,
                "neighbor_threads_per_scheme": neighbor_workers,
                "algorithm_version": ALGORITHM_VERSION,
                "role_policy_overrides": config["role_policy_overrides"],
                "decomposition": config["decomposition"],
                "enumeration": config["enumeration"],
                "fingerprint": config["fingerprint"],
            },
            "registry_validation": registry_validation,
            "control": {
                "historical_existing_r1_non_agile_full_r0": historical_control,
                "current_existing_r1_non_agile_full_r0": current_control,
                "terminology": "reaction-enumerated support",
                "identity_note": (
                    "The historical control reproduces the frozen isomeric-string result. "
                    "The current control compares the corrected constitutional R0 strings "
                    "against the unchanged legacy R1 strings; the primary R1-prime audit "
                    "uses the current constitutional identity throughout."
                ),
            },
            "schemes": scheme_results,
            "artifacts": artifact_records,
            "scientific_status": {
                "all_harvest_sources_are_r0_train": all(
                    summary["all_harvest_sources_are_r0_train"]
                    for summary in decomposition_summaries.values()
                ),
                "heldout_structures_decomposed_for_harvesting": 0,
                "decomposition_precision": (
                    "diagnostic_only; M0-05 source-grounded evidence is the independent "
                    "training-admission authority"
                ),
                "generator_boundary": (
                    "R1-prime is a diagnostic reaction-enumerated reference set. "
                    "The product generator remains a whole-graph discrete flow."
                ),
            },
        }
        result_payload = _json_bytes(result)
        for name, payload in artifact_payloads.items():
            (staging / name).write_bytes(payload)
        (staging / "result.json").write_bytes(result_payload)
        shutil.rmtree(work_dir)
        os.replace(staging, output_dir)
        if checkpoint_run_dir.exists():
            shutil.rmtree(checkpoint_run_dir)
        return result
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _run_worker_task(
    task: tuple[
        str,
        str,
        int,
        tuple[str, ...],
        tuple[str, ...],
        tuple[ReactionDefinition, ...],
        int,
        int,
    ],
) -> tuple[str, tuple[DecompositionCandidate, ...], tuple[BlockRecord, ...], dict[str, Any]]:
    return _decompose_scheme_worker(*task)
