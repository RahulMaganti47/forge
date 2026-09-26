from __future__ import annotations

import numpy as np
import pytest

from forge.assembly import ReactionProgramSpec
from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import (
    ReactionProgramSparseFlow,
    collate_reaction_program_records,
)
from forge.model.reaction_program_graph import tensorize_reaction_program_product
from forge.model.reaction_program_sampling import (
    ReactionProgramLayout,
    sample_reaction_program_products,
)

torch = pytest.importorskip("torch")


def _record():
    vocabulary = ReactionProgramVocabulary.from_specs(
        (ReactionProgramSpec("aza", "aza", "head", "tail", 1, 2),)
    )
    atom_vocabulary = (
        AtomState("C", 0, False, 0),
        AtomState("N", 0, False, 0),
    )
    # NCCN: the first two canonical atoms belong to one introduced tail and the latter two to head.
    # Obtain the canonical atom order explicitly so this fixture tests the representation contract.
    from rdkit import Chem

    molecule = Chem.MolFromSmiles("NCCN")
    assert molecule is not None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    parsed = Chem.MolFromSmiles(canonical)
    assert parsed is not None
    roles = ["tail" if atom.GetIdx() < 2 else "head" for atom in parsed.GetAtoms()]
    # The index split above need not remain connected under future RDKit canonicalization; build the
    # roles from the two terminal halves of the path instead.
    adjacency = [
        [neighbor.GetIdx() for neighbor in atom.GetNeighbors()] for atom in parsed.GetAtoms()
    ]
    terminals = sorted(index for index, neighbors in enumerate(adjacency) if len(neighbors) == 1)
    tail = {terminals[0], adjacency[terminals[0]][0]}
    roles = ["tail" if index in tail else "head" for index in range(parsed.GetNumAtoms())]
    return (
        vocabulary,
        atom_vocabulary,
        tensorize_reaction_program_product(
            record_id="example",
            program_id="aza",
            canonical_product_smiles=canonical,
            atom_roles=roles,
            program_depth=1,
            accumulator_role="head",
            repeat_role="tail",
            vocabulary=vocabulary,
            atom_vocabulary=atom_vocabulary,
        ),
    )


def test_program_flow_consumes_clean_semantics_without_component_ids() -> None:
    vocabulary, atom_vocabulary, record = _record()
    batch = collate_reaction_program_records((record,), maximum_closures=1)
    model = ReactionProgramSparseFlow(
        vocabulary=vocabulary,
        node_classes=len(atom_vocabulary),
        hidden_dim=16,
        layers=1,
        maximum_closures=1,
        maximum_heavy_atoms=8,
        dropout=0.0,
    )
    assert model.backbone.position_embedding is not None
    predictions = model(
        nodes=batch["nodes"],
        parents=batch["parents"],
        parent_bonds=batch["parent_bonds"],
        closure_left=batch["closure_left"],
        closure_right=batch["closure_right"],
        closure_bonds=batch["closure_bonds"],
        t=torch.tensor([0.5]),
        node_mask=batch["node_mask"],
        child_mask=batch["child_mask"],
        closure_mask=batch["closure_mask"],
        program_states=batch["program_states"],
        role_states=batch["role_states"],
        core_position_states=batch["core_position_states"],
        program_depths=batch["program_depths"],
        adapter_mask=batch["adapter_mask"],
    )
    assert predictions["nodes"].shape == (1, record.node_count, len(atom_vocabulary))
    assert bool(np.all(record.graph.parents[1:] < np.arange(1, record.node_count)))


def test_null_control_zeros_every_program_coordinate() -> None:
    _, _, record = _record()
    batch = collate_reaction_program_records(
        (record,), maximum_closures=1, conditioning_mode="null"
    )
    assert not bool(batch["program_states"].any())
    assert not bool(batch["role_states"].any())
    assert not bool(batch["core_position_states"].any())
    assert not bool(batch["program_depths"].any())


def test_program_sampler_runs_without_a_component_catalog() -> None:
    vocabulary, atom_vocabulary, _ = _record()
    model = ReactionProgramSparseFlow(
        vocabulary=vocabulary,
        node_classes=len(atom_vocabulary),
        hidden_dim=16,
        layers=1,
        maximum_closures=1,
        maximum_heavy_atoms=8,
        dropout=0.0,
    )
    layout = ReactionProgramLayout(
        program_id="aza",
        program_state=vocabulary.program_to_index["aza"],
        program_depth=1,
        accumulator_role_state=vocabulary.role_to_index["head"],
        repeat_role_state=vocabulary.role_to_index["tail"],
        accumulator_atom_count=2,
        repeat_atom_count=2,
        closure_count=0,
    )
    rows, metrics = sample_reaction_program_products(
        model,
        (layout,),
        atom_vocabulary,
        np.asarray([0.5, 0.5]),
        np.asarray([1.0, 0.0, 0.0]),
        adapters=None,
        sample_steps=2,
        batch_size=1,
        seed=7,
        device="cpu",
    )
    assert len(rows) == metrics["samples"] == 1
    assert rows[0]["program_id"] == "aza"
    assert "component_id" not in rows[0]
