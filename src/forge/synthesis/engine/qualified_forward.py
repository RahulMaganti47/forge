"""Shared loader and executor for hash-qualified forward reaction transforms."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions


class QualifiedForwardError(ValueError):
    """Raised when a qualified forward transform violates its contract."""


@dataclass(frozen=True)
class QualifiedForwardReaction:
    """A compiled registry reaction with a verified role order."""

    reaction_id: str
    role_names: tuple[str, ...]
    reaction: rdChemReactions.ChemicalReaction


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    try:
        if path.suffix in {".yaml", ".yml"}:
            value = yaml.safe_load(path.read_text())
        else:
            value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise QualifiedForwardError(f"{label} not found: {path}") from exc
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise QualifiedForwardError(f"{label} could not be parsed: {path}") from exc
    if not isinstance(value, dict):
        raise QualifiedForwardError(f"{label} must contain a mapping")
    return value


def load_qualified_forward_reaction(
    registry_path: Path,
    variant_path: Path,
    *,
    reaction_id: str,
) -> QualifiedForwardReaction:
    """Load one registry transform and verify it against its variant contract."""

    registry = _load_mapping(registry_path, label="qualified reaction registry")
    variant = _load_mapping(variant_path, label="reaction variant")
    if variant.get("variant") != reaction_id or variant.get("reaction_id") != reaction_id:
        raise QualifiedForwardError(
            f"reaction variant does not declare reaction_id {reaction_id!r}"
        )

    reactions = registry.get("reactions")
    if not isinstance(reactions, list):
        raise QualifiedForwardError("qualified reaction registry has no reactions")
    matches = [
        item
        for item in reactions
        if isinstance(item, dict) and item.get("reaction_id") == reaction_id
    ]
    if len(matches) != 1:
        raise QualifiedForwardError(f"qualified reaction {reaction_id!r} must resolve exactly once")
    definition = matches[0]
    registry_roles = definition.get("reactant_roles")
    variant_roles = variant.get("roles")
    if not isinstance(registry_roles, list) or not isinstance(variant_roles, list):
        raise QualifiedForwardError("qualified reaction roles are missing")
    role_names = tuple(role.get("name") for role in registry_roles if isinstance(role, dict))
    variant_role_names = tuple(role.get("name") for role in variant_roles if isinstance(role, dict))
    if (
        len(role_names) != len(registry_roles)
        or role_names != variant_role_names
        or len(role_names) != variant.get("expected_reactant_count")
    ):
        raise QualifiedForwardError("registry and variant reactant-role orders disagree")

    reaction = rdChemReactions.ReactionFromSmarts(definition.get("atom_mapped_reaction_smarts", ""))
    if reaction is None or reaction.GetNumReactantTemplates() != len(role_names):
        raise QualifiedForwardError(f"qualified reaction {reaction_id!r} did not compile")
    return QualifiedForwardReaction(
        reaction_id=reaction_id,
        role_names=role_names,
        reaction=reaction,
    )


def unique_forward_products(
    compiled: QualifiedForwardReaction,
    reactant_smiles: Sequence[str],
    *,
    max_products: int,
    isomeric_smiles: bool,
) -> tuple[str, ...]:
    """Return unique sanitized products in deterministic canonical order."""

    if len(reactant_smiles) != len(compiled.role_names):
        raise QualifiedForwardError(
            f"{compiled.reaction_id} expected {len(compiled.role_names)} reactants, "
            f"received {len(reactant_smiles)}"
        )
    if isinstance(max_products, bool) or not isinstance(max_products, int) or max_products <= 0:
        raise QualifiedForwardError("max_products must be a positive integer")
    molecules = []
    for role_name, smiles in zip(compiled.role_names, reactant_smiles, strict=True):
        with rdBase.BlockLogs():
            molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise QualifiedForwardError(
                f"{compiled.reaction_id} {role_name} reactant is invalid SMILES"
            )
        molecules.append(molecule)

    with rdBase.BlockLogs():
        outcomes = compiled.reaction.RunReactants(
            tuple(molecules),
            maxProducts=max_products,
        )
    if len(outcomes) >= max_products:
        raise QualifiedForwardError(f"{compiled.reaction_id} reached max_products={max_products}")
    products: set[str] = set()
    for index, outcome in enumerate(outcomes):
        if len(outcome) != 1:
            raise QualifiedForwardError(
                f"{compiled.reaction_id} outcome {index} did not contain one product"
            )
        product = outcome[0]
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(product)
        except Exception as exc:
            raise QualifiedForwardError(
                f"{compiled.reaction_id} outcome {index} could not be sanitized"
            ) from exc
        products.add(
            Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=isomeric_smiles,
            )
        )
    return tuple(sorted(products))
