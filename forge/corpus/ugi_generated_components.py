"""Recover exact Ugi precursor graphs from generated product semantics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from rdkit import Chem

from forge.model.defog_feasibility import AtomState, FeasibilityError
from forge.model.ugi_adapter_features import CORE_POSITION_TO_INDEX, ORIGIN_TO_INDEX
from forge.model.ugi_chemistry_flow import UgiChemistrySample, chemistry_sample_to_molecule
from forge.model.ugi_chemistry_interface import ChemistryTopologyCondition
from forge.model.v5_sparse_representation import canonical_constitutional_molecule
from forge.potency.annotations import ROLE_NAMES


class UgiGeneratedComponentError(RuntimeError):
    """Raised when generated Ugi semantics cannot recover three precursors."""


def _origin_submolecule(
    product: Chem.Mol,
    selected: set[int],
) -> tuple[Chem.RWMol, dict[int, int]]:
    editable = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old in sorted(selected):
        atom = Chem.Atom(product.GetAtomWithIdx(old))
        atom.SetAtomMapNum(0)
        old_to_new[old] = editable.AddAtom(atom)
    for bond in product.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        if left in selected and right in selected:
            editable.AddBond(old_to_new[left], old_to_new[right], bond.GetBondType())
    return editable, old_to_new


def precursor_components_from_product_semantics(
    product: Chem.Mol,
    origin_states: Sequence[int],
    core_position_states: Sequence[int],
) -> dict[str, str]:
    """Reverse the qualified Ugi core using exact generated atom semantics.

    This is not unconstrained retrosynthesis.  It is the deterministic inverse
    of the fixed AGILE-type three-component assembly representation: the amine
    boundary bond is removed, the aldehyde oxygen is restored, and the
    isocyanide C/N bond and charges are restored.
    """

    if product.GetNumAtoms() != len(origin_states) or len(origin_states) != len(
        core_position_states
    ):
        raise UgiGeneratedComponentError("product and semantic atom arrays are misaligned")
    components: dict[str, str] = {}
    for role in ROLE_NAMES:
        origin = ORIGIN_TO_INDEX[role]
        selected = {index for index, state in enumerate(origin_states) if int(state) == origin}
        if not selected:
            raise UgiGeneratedComponentError(f"generated product has an empty {role} origin")
        editable, old_to_new = _origin_submolecule(product, selected)

        if role == "amine_head":
            core = [
                index
                for index in selected
                if int(core_position_states[index]) == CORE_POSITION_TO_INDEX["map_1"]
            ]
            if len(core) != 1:
                raise UgiGeneratedComponentError("amine origin lacks one map-1 atom")
            atom = editable.GetAtomWithIdx(old_to_new[core[0]])
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        elif role == "oxoester_aldehyde_body_tail":
            core = [
                index
                for index in selected
                if int(core_position_states[index]) == CORE_POSITION_TO_INDEX["map_2"]
            ]
            if len(core) != 1:
                raise UgiGeneratedComponentError("aldehyde origin lacks one map-2 atom")
            carbon = editable.GetAtomWithIdx(old_to_new[core[0]])
            carbon.SetNumExplicitHs(0)
            carbon.SetNoImplicit(False)
            oxygen = editable.AddAtom(Chem.Atom("O"))
            editable.AddBond(old_to_new[core[0]], oxygen, Chem.BondType.DOUBLE)
        elif role == "isocyanide_tail":
            carbon = [
                index
                for index in selected
                if int(core_position_states[index]) == CORE_POSITION_TO_INDEX["map_3"]
            ]
            nitrogen = [
                index
                for index in selected
                if int(core_position_states[index]) == CORE_POSITION_TO_INDEX["map_4"]
            ]
            if len(carbon) != 1 or len(nitrogen) != 1:
                raise UgiGeneratedComponentError("isocyanide origin lacks map-3/map-4 atoms")
            carbon_new = old_to_new[carbon[0]]
            nitrogen_new = old_to_new[nitrogen[0]]
            bond = editable.GetBondBetweenAtoms(carbon_new, nitrogen_new)
            if bond is None:
                raise UgiGeneratedComponentError("isocyanide core atoms are disconnected")
            bond.SetBondType(Chem.BondType.TRIPLE)
            carbon_atom = editable.GetAtomWithIdx(carbon_new)
            nitrogen_atom = editable.GetAtomWithIdx(nitrogen_new)
            carbon_atom.SetFormalCharge(-1)
            carbon_atom.SetNumExplicitHs(0)
            carbon_atom.SetNoImplicit(True)
            nitrogen_atom.SetFormalCharge(1)
            nitrogen_atom.SetNumExplicitHs(0)
            nitrogen_atom.SetNoImplicit(True)

        molecule = editable.GetMol()
        try:
            Chem.SanitizeMol(molecule)
            normalized = canonical_constitutional_molecule(molecule, f"generated/{role}")
        except (ValueError, RuntimeError, FeasibilityError) as error:
            raise UgiGeneratedComponentError(
                f"generated {role} precursor is not chemically valid"
            ) from error
        components[role] = Chem.MolToSmiles(
            normalized,
            canonical=True,
            isomericSmiles=False,
        )
    return components


def generated_ugi_component_smiles(
    condition: ChemistryTopologyCondition,
    sample: UgiChemistrySample,
    atom_vocabulary: tuple[AtomState, ...],
) -> dict[str, str]:
    """Recover generated precursor SMILES, including emitted decorations."""

    product = chemistry_sample_to_molecule(condition, sample, atom_vocabulary)
    origins = condition.origin_states.astype(np.int64).tolist()
    positions = condition.core_position_states.astype(np.int64).tolist()
    if sample.decoration_anchors is not None:
        for anchor in sample.decoration_anchors:
            if int(anchor) == 0:
                continue
            origins.append(int(condition.origin_states[int(anchor) - 1]))
            positions.append(CORE_POSITION_TO_INDEX["not_core"])
    elif sample.decoration_anchor:
        origins.append(int(condition.origin_states[sample.decoration_anchor - 1]))
        positions.append(CORE_POSITION_TO_INDEX["not_core"])
    return precursor_components_from_product_semantics(product, origins, positions)
