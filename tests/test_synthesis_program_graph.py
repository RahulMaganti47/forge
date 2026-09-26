from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.sparse_topology_feasibility import sparse_roundtrip_exact
from forge.model.synthesis_program_graph import (
    SynthesisProgramGraphError,
    tensorize_synthesis_program_product,
)


def _atom_vocabulary(smiles: str) -> tuple[AtomState, ...]:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return tuple(
        sorted(
            {
                AtomState(
                    atom.GetSymbol(),
                    atom.GetFormalCharge(),
                    atom.GetIsAromatic(),
                    atom.GetNumExplicitHs(),
                )
                for atom in molecule.GetAtoms()
            },
            key=AtomState.key,
        )
    )


def test_shared_serializer_preserves_aromatic_graph_and_program_semantics() -> None:
    smiles = "c1ccccc1CN(C)CC(=O)NCC"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    roles = [
        "aldehyde_body" if atom.GetIsAromatic() or index == 6 else "amine_head"
        for index, atom in enumerate(molecule.GetAtoms())
    ]
    roles[12] = "assembly_introduced"
    roles[13] = "isocyanide_tail"
    roles[14] = "isocyanide_tail"
    core_positions = ["exterior"] * molecule.GetNumAtoms()
    core_positions[6] = "ugi:map_2"
    core_positions[7] = "ugi:map_1"
    core_positions[11] = "ugi:map_3"
    core_positions[12] = "ugi:template_introduced_0"
    core_positions[13] = "ugi:map_4"
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=("ugi",),
        roles=("aldehyde_body", "amine_head", "assembly_introduced", "isocyanide_tail"),
        core_positions=tuple(position for position in core_positions if position != "exterior"),
        maximum_steps=1,
    )

    record = tensorize_synthesis_program_product(
        record_id="aromatic-ugi",
        program_id="ugi",
        canonical_product_smiles=smiles,
        atom_roles=roles,
        atom_core_positions=core_positions,
        program_depth=1,
        vocabulary=vocabulary,
        atom_vocabulary=_atom_vocabulary(smiles),
        fixed_atom_indices=(6, 7, 11, 12, 13),
    )

    assert sparse_roundtrip_exact(record.graph)
    assert any(state == 3 for state in record.graph.parent_bonds[1:]) or any(
        state == 3 for state in record.graph.closure_bonds
    )
    assert record.fixed_atom_mask.sum() == 5
    assert sorted(record.canonical_atom_order.tolist()) == list(range(molecule.GetNumAtoms()))
    assert sum(block.atom_count for block in record.component_blocks) == record.node_count
    for block in record.component_blocks:
        assert np.unique(record.role_states[block.start : block.stop]).tolist() == [
            block.role_state
        ]


def test_shared_serializer_supports_repeated_same_role_components() -> None:
    smiles = "CCN(CC)CC"
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    nitrogen = next(atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == "N")
    roles = ["amine_head" if index == nitrogen else "aldehyde_tail" for index in range(7)]
    core_positions = ["exterior"] * 7
    core_positions[nitrogen] = "lx:c_n_formed"
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=("lx",),
        roles=roles,
        core_positions=("lx:c_n_formed",),
        maximum_steps=3,
    )

    record = tensorize_synthesis_program_product(
        record_id="three-tails",
        program_id="lx",
        canonical_product_smiles=smiles,
        atom_roles=roles,
        atom_core_positions=core_positions,
        program_depth=3,
        vocabulary=vocabulary,
        atom_vocabulary=_atom_vocabulary(smiles),
    )

    repeat_blocks = [block for block in record.component_blocks if block.role == "aldehyde_tail"]
    assert len(repeat_blocks) == 3
    assert sorted(block.atom_count for block in repeat_blocks) == [2, 2, 2]
    assert record.component_count == 4


def test_shared_serializer_rejects_unqualified_semantics() -> None:
    vocabulary = ReactionProgramVocabulary.from_semantics(
        program_ids=("qualified",),
        roles=("head", "tail"),
        core_positions=("qualified:core",),
        maximum_steps=1,
    )
    with pytest.raises(SynthesisProgramGraphError, match="roles outside"):
        tensorize_synthesis_program_product(
            record_id="bad-role",
            program_id="qualified",
            canonical_product_smiles="CN",
            atom_roles=("head", "unsupported"),
            atom_core_positions=("qualified:core", "exterior"),
            program_depth=1,
            vocabulary=vocabulary,
            atom_vocabulary=_atom_vocabulary("CN"),
        )
    with pytest.raises(SynthesisProgramGraphError, match="fixed atoms must"):
        tensorize_synthesis_program_product(
            record_id="bad-fixed-mask",
            program_id="qualified",
            canonical_product_smiles="CN",
            atom_roles=("head", "tail"),
            atom_core_positions=("qualified:core", "exterior"),
            program_depth=1,
            vocabulary=vocabulary,
            atom_vocabulary=_atom_vocabulary("CN"),
            fixed_atom_indices=(1,),
        )


def test_vocabulary_from_semantics_rejects_duplicate_program_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        ReactionProgramVocabulary.from_semantics(
            program_ids=("same", "same"),
            roles=("head",),
            maximum_steps=1,
        )
