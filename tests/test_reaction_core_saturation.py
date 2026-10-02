from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forge.model.networks.dense_flow import AtomState
from forge.model.networks.reaction_flow import collate_synthesis_program_layouts
from forge.model.networks.sparse_flow import SparseGraphRecord
from forge.model.representation.synthesis_graph import (
    SynthesisProgramComponentBlock,
    SynthesisProgramGraphRecord,
)
from forge.model.sampling.core_saturation import (
    ReactionCoreSaturationError,
    ReactionCoreSaturationPolicy,
)
from forge.model.sampling.synthesis import (
    CORE_SATURATION_TERMINAL_DECODE_POLICY,
    SynthesisProgramSamplingError,
    decode_synthesis_program_strict_argmax,
    sample_synthesis_program_products,
)

REGISTRY = Path(__file__).parent / "fixtures/qualified_reactions_v1.json"
REGISTRY_SHA256 = "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf"
PROGRAM = "ugi_3cr_agile"
CORE_POSITION_STATES = (
    "unconditioned",
    "exterior",
    f"{PROGRAM}:map_1",
    f"{PROGRAM}:map_2",
    f"{PROGRAM}:map_3",
    f"{PROGRAM}:map_4",
    f"{PROGRAM}:template_introduced_0",
)
ATOM_VOCABULARY = (
    AtomState("C", 0, False, 0),
    AtomState("N", 0, False, 0),
    AtomState("O", 0, False, 0),
)
CARBON, NITROGEN, OXYGEN = 0, 1, 2
SINGLE, DOUBLE = 0, 1

# One synthetic AGILE-type Ugi layout.  Node 4 and node 8 are the only nodes whose parent the
# decoder chooses; every other edge is adapter-fixed, exactly as the production layouts are.
#   0 map_1 N   1 amine tail   2 map_2 alpha C   3,4 aldehyde tail
#   5 map_3 C   6 map_4 N      7,8 isocyanide tail            9 assembly-introduced O
NODE_STATES = (NITROGEN, CARBON, CARBON, CARBON, CARBON, CARBON, NITROGEN, CARBON, CARBON, OXYGEN)
PARENTS = (0, 0, 0, 2, 3, 2, 5, 6, 7, 5)
PARENT_BONDS = (SINGLE,) * 9 + (DOUBLE,)
CORE_STATES = (2, 1, 3, 1, 1, 4, 5, 1, 1, 6)
ROLE_STATES = (1, 1, 2, 2, 2, 3, 3, 3, 3, 4)
FIXED_ATOMS = (True, False, True, False, False, True, True, False, False, True)
FIXED_PARENT_BONDS = (False, True, True, True, False, True, True, True, False, True)
BLOCKS = (
    SynthesisProgramComponentBlock("amine_head", 1, 0, 2),
    SynthesisProgramComponentBlock("oxoester_aldehyde_body_tail", 2, 2, 5),
    SynthesisProgramComponentBlock("isocyanide_tail", 3, 5, 9),
    SynthesisProgramComponentBlock("assembly_introduced", 4, 9, 10),
)
VARIABLE_NODES = (4, 8)


def _record() -> SynthesisProgramGraphRecord:
    count = len(NODE_STATES)
    edges = np.zeros((count, count), dtype=np.int8)
    for child in range(1, count):
        parent = PARENTS[child]
        order = 2 if PARENT_BONDS[child] == DOUBLE else 1
        edges[child, parent] = edges[parent, child] = order
    return SynthesisProgramGraphRecord(
        graph=SparseGraphRecord(
            structure_id="synthetic-ugi",
            canonical_smiles="CCC(NC)C(=O)NCC",
            node_states=np.asarray(NODE_STATES, dtype=np.int64),
            parents=np.asarray(PARENTS, dtype=np.int64),
            parent_bonds=np.asarray(PARENT_BONDS, dtype=np.int64),
            closure_left=np.zeros(0, dtype=np.int64),
            closure_right=np.zeros(0, dtype=np.int64),
            closure_bonds=np.zeros(0, dtype=np.int64),
            edges=edges,
        ),
        canonical_atom_order=np.arange(count, dtype=np.int64),
        program_id=PROGRAM,
        program_state=1,
        program_depth=1,
        role_states=np.asarray(ROLE_STATES, dtype=np.int64),
        core_position_states=np.asarray(CORE_STATES, dtype=np.int64),
        component_blocks=BLOCKS,
        fixed_atom_mask=np.asarray(FIXED_ATOMS, dtype=np.bool_),
        fixed_parent_bond_mask=np.asarray(FIXED_PARENT_BONDS, dtype=np.bool_),
        fixed_closure_bond_mask=np.zeros(0, dtype=np.bool_),
    )


def _predictions(layout, preferred_parents: dict[int, int]):
    batch, nodes = layout["nodes"].shape
    closures = layout["closure_bonds"].shape[1]
    predictions = {
        "nodes": torch.zeros((batch, nodes, len(ATOM_VOCABULARY))),
        "parents": torch.zeros((batch, nodes, nodes)),
        "parent_bonds": torch.zeros((batch, nodes, 4)),
        "closure_left": torch.zeros((batch, closures, nodes)),
        "closure_right": torch.zeros((batch, closures, nodes)),
        "closure_bonds": torch.zeros((batch, closures, 4)),
    }
    predictions["parents"].scatter_(-1, layout["parents"].unsqueeze(-1), 5.0)
    predictions["parent_bonds"].scatter_(-1, layout["parent_bonds"].unsqueeze(-1), 5.0)
    for child, parent in preferred_parents.items():
        predictions["parents"][0, child, parent] = 50.0
    return predictions


def _decode(preferred_parents: dict[int, int], *, constrained: bool):
    record = _record()
    layout = collate_synthesis_program_layouts((record,), maximum_closures=0)
    predictions = _predictions(layout, preferred_parents)
    core_saturation = None
    if constrained:
        core_saturation = ReactionCoreSaturationPolicy.from_qualified_registry(
            REGISTRY, expected_sha256=REGISTRY_SHA256
        ).bind(CORE_POSITION_STATES)
    terminal, reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (record,),
        ATOM_VOCABULARY,
        core_saturation=core_saturation,
    )
    return terminal, reasons, record


def test_policy_is_derived_from_the_qualified_product_template() -> None:
    policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        REGISTRY, expected_sha256=REGISTRY_SHA256
    )
    # [CH1:2] on carbon leaves 3 heavy bonds (6 units); [NH1+0:4] on nitrogen leaves 2 (4 units).
    assert policy.core_position_valence_units == {
        f"{PROGRAM}:map_2": 6,
        f"{PROGRAM}:map_4": 4,
    }
    # The amine nitrogen and the amide carbon state no hydrogen count, so they stay free.
    assert f"{PROGRAM}:map_1" not in policy.core_position_valence_units
    assert f"{PROGRAM}:map_3" not in policy.core_position_valence_units


def test_registry_hash_and_vocabulary_mismatches_fail_closed() -> None:
    with pytest.raises(ReactionCoreSaturationError):
        ReactionCoreSaturationPolicy.from_qualified_registry(REGISTRY, expected_sha256="0" * 64)
    policy = ReactionCoreSaturationPolicy.from_qualified_registry(REGISTRY)
    with pytest.raises(ReactionCoreSaturationError):
        policy.bind(("unconditioned", "exterior"))


def test_unconstrained_decode_puts_a_fourth_neighbour_on_the_alpha_carbon() -> None:
    _, reasons, _ = _decode({4: 2}, constrained=False)
    assert reasons == (None,)
    terminal, _, _ = _decode({4: 2}, constrained=False)
    assert int(terminal["parents"][0, 4]) == 2


def test_constrained_decode_refuses_a_fourth_neighbour_on_the_alpha_carbon() -> None:
    terminal, reasons, _ = _decode({4: 2}, constrained=True)
    assert reasons == (None,)
    # Node 3 is the only remaining aldehyde-component parent: the alpha carbon keeps its one H.
    assert int(terminal["parents"][0, 4]) == 3


def test_constrained_decode_refuses_a_third_neighbour_on_the_amide_nitrogen() -> None:
    terminal, reasons, _ = _decode({8: 6}, constrained=True)
    assert reasons == (None,)
    assert int(terminal["parents"][0, 8]) == 7
    unconstrained, _, _ = _decode({8: 6}, constrained=False)
    assert int(unconstrained["parents"][0, 8]) == 6


def test_constrained_decode_keeps_a_generated_edge_inside_its_component() -> None:
    unconstrained, _, _ = _decode({4: 1}, constrained=False)
    assert int(unconstrained["parents"][0, 4]) == 1
    terminal, reasons, _ = _decode({4: 1}, constrained=True)
    assert reasons == (None,)
    assert int(terminal["parents"][0, 4]) == 3


def test_constrained_decode_preserves_every_adapter_fixed_state() -> None:
    terminal, reasons, record = _decode({4: 2, 8: 6}, constrained=True)
    assert reasons == (None,)
    fixed = record.fixed_parent_bond_mask
    assert np.array_equal(
        terminal["parents"][0, : record.node_count].numpy()[fixed],
        record.graph.parents[fixed],
    )
    assert np.array_equal(
        terminal["nodes"][0, : record.node_count].numpy()[record.fixed_atom_mask],
        record.graph.node_states[record.fixed_atom_mask],
    )


def test_saturation_holds_exactly_on_every_constrained_decode() -> None:
    units = {SINGLE: 2, DOUBLE: 4}
    for preferred in ({4: 2}, {8: 6}, {4: 1, 8: 6}, {4: 3, 8: 7}):
        terminal, reasons, record = _decode(preferred, constrained=True)
        assert reasons == (None,)
        used = np.zeros(record.node_count, dtype=np.int64)
        parents = terminal["parents"][0, : record.node_count].numpy()
        bonds = terminal["parent_bonds"][0, : record.node_count].numpy()
        for child in range(1, record.node_count):
            value = units[int(bonds[child])]
            used[child] += value
            used[int(parents[child])] += value
        assert int(used[2]) == 6, preferred  # map_2 alpha carbon keeps exactly one hydrogen
        assert int(used[6]) == 4, preferred  # map_4 amide nitrogen stays N-H


def test_policy_and_terminal_decoder_must_be_supplied_together() -> None:
    policy = ReactionCoreSaturationPolicy.from_qualified_registry(REGISTRY)
    with pytest.raises(SynthesisProgramSamplingError):
        sample_synthesis_program_products(
            object(),
            (_record(),),
            ATOM_VOCABULARY,
            np.full(len(ATOM_VOCABULARY), 1 / len(ATOM_VOCABULARY)),
            np.full(4, 0.25),
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=0,
            device="cpu",
            terminal_decode_policy=CORE_SATURATION_TERMINAL_DECODE_POLICY,
        )
    with pytest.raises(SynthesisProgramSamplingError):
        sample_synthesis_program_products(
            object(),
            (_record(),),
            ATOM_VOCABULARY,
            np.full(len(ATOM_VOCABULARY), 1 / len(ATOM_VOCABULARY)),
            np.full(4, 0.25),
            samples_per_program=1,
            sample_steps=2,
            batch_size=1,
            seed=0,
            device="cpu",
            terminal_decode_policy="strict_valence_topology_argmax",
            reaction_core_saturation_policy=policy,
        )


def test_the_contract_does_not_touch_another_program() -> None:
    record = _record()
    other = SynthesisProgramGraphRecord(
        graph=record.graph,
        canonical_atom_order=record.canonical_atom_order,
        program_id="bl_2023_repeated_aza_michael",
        program_state=1,
        program_depth=1,
        role_states=record.role_states,
        core_position_states=record.core_position_states,
        component_blocks=record.component_blocks,
        fixed_atom_mask=record.fixed_atom_mask,
        fixed_parent_bond_mask=record.fixed_parent_bond_mask,
        fixed_closure_bond_mask=record.fixed_closure_bond_mask,
    )
    layout = collate_synthesis_program_layouts((other,), maximum_closures=0)
    predictions = _predictions(layout, {4: 2})
    core_saturation = ReactionCoreSaturationPolicy.from_qualified_registry(REGISTRY).bind(
        CORE_POSITION_STATES
    )
    terminal, reasons = decode_synthesis_program_strict_argmax(
        predictions,
        layout,
        (other,),
        ATOM_VOCABULARY,
        core_saturation=core_saturation,
    )
    assert reasons == (None,)
    assert int(terminal["parents"][0, 4]) == 2
