"""Registry-derived decode-time saturation contract for reaction-core positions.

The qualified registry states each reaction as an atom-mapped transform.  Its product template
pins the substitution of some mapped core atoms exactly: ``[CH1:2]`` and ``[NH1+0:4]`` in the
AGILE-type Ugi 3CR say that, in *every* product of that transform, the aldehyde-derived alpha
carbon carries exactly one hydrogen and the isocyanide-derived amide nitrogen carries exactly
one hydrogen.  A generated graph that puts another heavy neighbour on either atom is not a
product of the transform, so it cannot decompose through it, even when it is a perfectly valid
molecule.

The generated product graph is hydrogen-implicit, so this contract is not expressible as an atom
state: it is a statement about the heavy-atom bond-order sum at those positions.  This module
reads that statement out of the hash-pinned registry (never out of retyped SMARTS) and exposes
it in the valence units the sparse decoder already uses, so the terminal decoder can mask it
rather than reject afterwards.

It also carries the two structural coordinates the same transform implies: precursor components
meet only at the reaction core, so a generated edge that is not adapter-fixed must stay inside
one component block, and a generated ring closure must join two exterior atoms of one component
block.  Both are stated here rather than inferred from data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import rdChemReactions

from forge.core.hashing import sha256_file
from forge.core.io import read_json_object
from forge.model.networks.dense_flow import AtomState
from forge.model.networks.sparse_flow import _maximum_valence_units
from forge.model.representation.synthesis_graph import SynthesisProgramGraphRecord

QUALIFIED_STATUS = "qualified_for_enumeration"


class ReactionCoreSaturationError(ValueError):
    """The qualified registry cannot supply a core-saturation contract."""


@dataclass(frozen=True)
class ReactionCoreSaturationPolicy:
    """One reaction's decode-time core contract, read from the qualified registry."""

    reaction_id: str
    registry_path: Path
    registry_sha256: str
    # Core-position state name -> exact heavy-atom valence units the product template fixes.
    core_position_valence_units: Mapping[str, int]
    component_confined_generated_edges: bool = True
    exterior_only_generated_closures: bool = True

    @classmethod
    def from_qualified_registry(
        cls,
        registry_path: Path,
        *,
        reaction_id: str = "ugi_3cr_agile",
        expected_sha256: str | None = None,
    ) -> ReactionCoreSaturationPolicy:
        """Derive the contract from the hash-pinned atom-mapped transform."""

        resolved = Path(registry_path).resolve()
        observed = str(sha256_file(resolved))
        if expected_sha256 is not None and observed != expected_sha256:
            raise ReactionCoreSaturationError(
                f"qualified reaction registry changed: expected {expected_sha256}, "
                f"found {observed}"
            )
        registry = read_json_object(
            resolved, error=ReactionCoreSaturationError, label="qualified reaction registry"
        )
        definitions = [
            value
            for value in registry.get("reactions", [])
            if str(value.get("reaction_id")) == reaction_id
        ]
        if len(definitions) != 1:
            raise ReactionCoreSaturationError(
                f"registry does not define exactly one reaction {reaction_id!r}"
            )
        definition = definitions[0]
        if str(definition.get("status")) != QUALIFIED_STATUS:
            raise ReactionCoreSaturationError(f"reaction {reaction_id!r} is not qualified")
        smarts = str(definition["atom_mapped_reaction_smarts"])
        with rdBase.BlockLogs():
            reaction = rdChemReactions.ReactionFromSmarts(smarts)
        if reaction is None or reaction.GetNumProductTemplates() != 1:
            raise ReactionCoreSaturationError(
                f"reaction {reaction_id!r} does not expose one atom-mapped product template"
            )
        template = reaction.GetProductTemplate(0)
        requirements: dict[str, int] = {}
        for atom in template.GetAtoms():
            map_number = int(atom.GetAtomMapNum())
            # ``NoImplicit`` is set on exactly those product-template atoms whose hydrogen count
            # the transform states explicitly.  Every other atom's substitution is inherited from
            # the substrate and must stay free.
            if map_number < 1 or not atom.GetNoImplicit():
                continue
            state = AtomState(
                symbol=str(atom.GetSymbol()),
                formal_charge=int(atom.GetFormalCharge()),
                aromatic=bool(atom.GetIsAromatic()),
                explicit_hydrogens=0,
            )
            units = _maximum_valence_units(state) - 2 * int(atom.GetNumExplicitHs())
            if units < 2:
                raise ReactionCoreSaturationError(
                    f"{reaction_id}:map_{map_number} leaves no heavy-atom valence"
                )
            requirements[f"{reaction_id}:map_{map_number}"] = int(units)
        if not requirements:
            raise ReactionCoreSaturationError(
                f"reaction {reaction_id!r} pins no core hydrogen count; the contract is empty"
            )
        return cls(
            reaction_id=reaction_id,
            registry_path=resolved,
            registry_sha256=observed,
            core_position_valence_units=dict(sorted(requirements.items())),
        )

    def bind(self, core_position_states: Sequence[str]) -> BoundReactionCoreSaturation:
        """Resolve the contract against one model's core-position vocabulary, fail-closed."""

        names = tuple(str(value) for value in core_position_states)
        missing = sorted(set(self.core_position_valence_units) - set(names))
        if missing:
            raise ReactionCoreSaturationError(
                f"core-position vocabulary omits {missing}; the contract cannot be enforced"
            )
        units = np.full(len(names), -1, dtype=np.int64)
        for name, value in self.core_position_valence_units.items():
            units[names.index(name)] = int(value)
        return BoundReactionCoreSaturation(policy=self, units_by_state=units)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "registry_path": str(self.registry_path),
            "registry_sha256": self.registry_sha256,
            "core_position_valence_units": dict(self.core_position_valence_units),
            "component_confined_generated_edges": self.component_confined_generated_edges,
            "exterior_only_generated_closures": self.exterior_only_generated_closures,
            "derivation": "atom_mapped_product_template_explicit_hydrogen_count",
        }


@dataclass(frozen=True)
class BoundReactionCoreSaturation:
    """A core contract resolved against one core-position vocabulary."""

    policy: ReactionCoreSaturationPolicy
    units_by_state: np.ndarray

    def applies_to(self, record: SynthesisProgramGraphRecord) -> bool:
        return str(record.program_id) == self.policy.reaction_id

    def required_units(self, record: SynthesisProgramGraphRecord) -> np.ndarray:
        """Return per-node required heavy-atom valence units, -1 where unconstrained."""

        if not self.applies_to(record):
            return np.full(record.node_count, -1, dtype=np.int64)
        states = np.asarray(record.core_position_states, dtype=np.int64)
        if states.size and int(states.max()) >= self.units_by_state.shape[0]:
            raise ReactionCoreSaturationError(
                "record core-position states exceed the bound vocabulary"
            )
        return self.units_by_state[states]


def qualified_product_core_pattern(
    registry_path: Path,
    *,
    reaction_id: str = "ugi_3cr_agile",
    expected_sha256: str | None = None,
) -> Chem.Mol:
    """Return the registry product template as a query, for auditing generated products."""

    resolved = Path(registry_path).resolve()
    observed = str(sha256_file(resolved))
    if expected_sha256 is not None and observed != expected_sha256:
        raise ReactionCoreSaturationError(
            f"qualified reaction registry changed: expected {expected_sha256}, found {observed}"
        )
    registry = read_json_object(
        resolved, error=ReactionCoreSaturationError, label="qualified reaction registry"
    )
    definitions = [
        value
        for value in registry.get("reactions", [])
        if str(value.get("reaction_id")) == reaction_id
    ]
    if len(definitions) != 1:
        raise ReactionCoreSaturationError(
            f"registry does not define exactly one reaction {reaction_id!r}"
        )
    smarts = str(definitions[0]["atom_mapped_reaction_smarts"])
    product_smarts = smarts.split(">>")[-1]
    with rdBase.BlockLogs():
        pattern = Chem.MolFromSmarts(product_smarts)
    if pattern is None:
        raise ReactionCoreSaturationError(
            f"reaction {reaction_id!r} product template is not a usable query"
        )
    return pattern


__all__ = [
    "BoundReactionCoreSaturation",
    "ReactionCoreSaturationError",
    "ReactionCoreSaturationPolicy",
    "qualified_product_core_pattern",
]
