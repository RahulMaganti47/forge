"""Non-circular R1-prime anchoring audit for M0-04.

This module creates a reaction-enumerated diagnostic reference set from blocks
retro-decomposed from each frozen R0 training fold. It is not a molecular
generator. The product model remains a whole-graph discrete flow.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.chemistry.reactive_sites import (
    RAW_SUBSTRUCTURE_MATCHES,
    SUPPORTED_MULTIPLICITY_SEMANTICS,
    SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES,
)

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
