from __future__ import annotations

import json
from pathlib import Path

import pytest
from rdkit import Chem

from forge.assembly import Ugi3AssemblyAdapter, Ugi3AssemblyError
from forge.core.hashing import sha256_file

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "data/vendor/qualified_reactions_v1.json"


def _example() -> tuple[dict[str, str], str]:
    document = json.loads(REGISTRY.read_text())
    reaction = next(row for row in document["reactions"] if row["reaction_id"] == "ugi_3cr_agile")
    roles = [row["name"] for row in reaction["reactant_roles"]]
    example = reaction["known_positive_examples"][0]
    return dict(zip(roles, example["reactants"], strict=True)), example["expected"]


def test_registry_example_reconstructs_exactly() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(
        REGISTRY,
        expected_sha256=str(sha256_file(REGISTRY)),
    )
    components, expected = _example()
    check = adapter.check_forward(components, expected)
    assert check.reaction_id == "ugi_3cr_agile"
    assert check.exact
    assert not check.saturated
    assert check.enumerated_outcomes == 1
    products = adapter.forward_products(components)
    assert products.products == (
        Chem.MolToSmiles(Chem.MolFromSmiles(expected), isomericSmiles=False),
    )
    assert not products.saturated


def test_adapter_rejects_an_incomplete_role_mapping() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    components, expected = _example()
    components.pop(next(iter(components)))
    with pytest.raises(Ugi3AssemblyError, match="roles differ"):
        adapter.check_forward(components, expected)


def test_registry_example_has_an_exact_open_decomposition() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    components, expected = _example()
    traces = adapter.decompose(expected)
    assert len(traces) == 1
    assert traces[0].as_mapping() == {
        role: Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
        for role, smiles in components.items()
    }
    assert adapter.check_forward(traces[0].as_mapping(), expected).exact


def test_transform_consistent_candidate_does_not_bypass_registry_handle_policy() -> None:
    adapter = Ugi3AssemblyAdapter.from_registry(REGISTRY)
    product = "CCCCCCCCCCC(=O)CCCCCCC(NC1CNC(N)OC1)C(=O)NCCC(C)CCC"

    candidates = adapter.transform_consistent_decomposition_candidates(product)

    assert adapter.decompose(product) == ()
    assert len(candidates) == 1
    assert candidates[0].registry_handle_qualified is False
    assessments = {
        value.role: value for value in candidates[0].handle_assessments
    }
    assert assessments["amine_head"].symmetry_distinct_handle_sites == 3
    assert assessments["amine_head"].passes_registry_handle_policy is False
    assert assessments["oxoester_aldehyde_body_tail"].passes_registry_handle_policy is True
    assert assessments["isocyanide_tail"].passes_registry_handle_policy is True
    assert candidates[0].as_mapping()["forward_replay_exact"] is True
