"""Typed loading and compilation of one qualified reaction-registry entry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdChemReactions

from forge.chemistry.reactive_sites import RAW_SUBSTRUCTURE_MATCHES

QUALIFIED_STATUS = "qualified_for_enumeration"


class ReactionRegistryError(ValueError):
    """A qualified reaction registry does not satisfy the assembly contract."""


@dataclass(frozen=True)
class ReactionRolePolicy:
    """Registry policy for one ordered reactant role."""

    name: str
    required_handle_smarts: str
    forbidden_smarts: tuple[str, ...]
    allowed_site_multiplicity: tuple[int, ...]
    site_multiplicity_semantics: str = RAW_SUBSTRUCTURE_MATCHES


@dataclass(frozen=True)
class ReactionRegistryEntry:
    """The fields needed to execute one reaction without importing a corpus workflow."""

    reaction_id: str
    reaction_version: int
    atom_mapped_reaction_smarts: str
    reactant_roles: tuple[ReactionRolePolicy, ...]


@dataclass(frozen=True)
class CompiledRegistryReaction:
    """Forward/reverse RDKit transforms and their validated role queries."""

    definition: ReactionRegistryEntry
    forward: rdChemReactions.ChemicalReaction
    reverse: rdChemReactions.ChemicalReaction
    handles: tuple[Chem.Mol, ...]
    forbidden: tuple[tuple[Chem.Mol, ...], ...]


def _role(raw: object, reaction_id: str) -> ReactionRolePolicy:
    if not isinstance(raw, dict):
        raise ReactionRegistryError(f"{reaction_id}: reactant role must be an object")
    name = raw.get("name")
    handle = raw.get("required_handle_smarts")
    forbidden = raw.get("forbidden_smarts")
    multiplicity = raw.get("allowed_site_multiplicity")
    if not isinstance(name, str) or not name:
        raise ReactionRegistryError(f"{reaction_id}: role name is invalid")
    if not isinstance(handle, str) or Chem.MolFromSmarts(handle) is None:
        raise ReactionRegistryError(f"{reaction_id}/{name}: handle SMARTS is invalid")
    if not isinstance(forbidden, list) or any(
        not isinstance(value, str) or Chem.MolFromSmarts(value) is None for value in forbidden
    ):
        raise ReactionRegistryError(f"{reaction_id}/{name}: forbidden SMARTS are invalid")
    if (
        not isinstance(multiplicity, list)
        or not multiplicity
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in multiplicity
        )
    ):
        raise ReactionRegistryError(f"{reaction_id}/{name}: site multiplicity is invalid")
    return ReactionRolePolicy(
        name=name,
        required_handle_smarts=handle,
        forbidden_smarts=tuple(forbidden),
        allowed_site_multiplicity=tuple(sorted(set(multiplicity))),
    )


def load_compiled_registry_reaction(
    registry_path: Path,
    *,
    reaction_id: str,
) -> CompiledRegistryReaction:
    """Resolve and compile exactly one qualified reaction from a registry."""

    try:
        document = json.loads(registry_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReactionRegistryError(
            f"reaction registry could not be read: {registry_path}"
        ) from exc
    reactions = document.get("reactions") if isinstance(document, dict) else None
    if not isinstance(reactions, list):
        raise ReactionRegistryError("reaction registry has no reaction list")
    matches = [
        raw for raw in reactions if isinstance(raw, dict) and raw.get("reaction_id") == reaction_id
    ]
    if len(matches) != 1:
        raise ReactionRegistryError(f"reaction {reaction_id!r} must resolve exactly once")
    raw = matches[0]
    if raw.get("status") != QUALIFIED_STATUS:
        raise ReactionRegistryError(f"reaction {reaction_id!r} is not qualified")
    smarts = raw.get("atom_mapped_reaction_smarts")
    roles = raw.get("reactant_roles")
    version = raw.get("reaction_version")
    if not isinstance(smarts, str) or smarts.count(">>") != 1:
        raise ReactionRegistryError(f"{reaction_id}: reaction SMARTS is invalid")
    if not isinstance(roles, list) or len(roles) < 2:
        raise ReactionRegistryError(f"{reaction_id}: at least two reactant roles are required")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ReactionRegistryError(f"{reaction_id}: reaction version is invalid")
    definition = ReactionRegistryEntry(
        reaction_id=reaction_id,
        reaction_version=version,
        atom_mapped_reaction_smarts=smarts,
        reactant_roles=tuple(_role(role, reaction_id) for role in roles),
    )
    left, right = smarts.split(">>")
    forward = rdChemReactions.ReactionFromSmarts(smarts)
    reverse = rdChemReactions.ReactionFromSmarts(f"{right}>>{left}")
    if forward is None or reverse is None:
        raise ReactionRegistryError(f"{reaction_id}: reaction SMARTS did not compile")
    if forward.GetNumReactantTemplates() != len(definition.reactant_roles):
        raise ReactionRegistryError(f"{reaction_id}: transform and role arity disagree")
    handles = tuple(
        Chem.MolFromSmarts(role.required_handle_smarts) for role in definition.reactant_roles
    )
    forbidden = tuple(
        tuple(Chem.MolFromSmarts(smarts) for smarts in role.forbidden_smarts)
        for role in definition.reactant_roles
    )
    if any(value is None for value in handles) or any(
        value is None for values in forbidden for value in values
    ):
        raise ReactionRegistryError(f"{reaction_id}: role queries did not compile")
    return CompiledRegistryReaction(
        definition=definition,
        forward=forward,
        reverse=reverse,
        handles=handles,
        forbidden=forbidden,
    )


__all__ = [
    "CompiledRegistryReaction",
    "QUALIFIED_STATUS",
    "ReactionRegistryEntry",
    "ReactionRegistryError",
    "ReactionRolePolicy",
    "load_compiled_registry_reaction",
]
