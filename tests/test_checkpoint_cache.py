"""Frozen cache class coordinates remain readable after package reorganization."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from forge.corpus.training_cache import load_ugi_training_cache_payload
from forge.model.networks.dense_flow import AtomState


@pytest.mark.parametrize(
    "module_name", ["forge.product.defog_feasibility", "forge.model.defog_feasibility"]
)
def test_frozen_cache_resolves_historical_atom_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, module_name: str
) -> None:
    path = tmp_path / "cache.pt"
    atom = AtomState("C", 0, False)
    with monkeypatch.context() as patch:
        parent = module_name.rsplit(".", 1)[0]
        if parent not in sys.modules:
            patch.setitem(sys.modules, parent, ModuleType(parent))
        historical = ModuleType(module_name)
        historical.AtomState = AtomState
        patch.setitem(sys.modules, module_name, historical)
        patch.setattr(AtomState, "__module__", module_name)
        torch.save(
            {
                "schema_version": "phase1_ugi_training_cache.v1",
                "corpus": {"atom": atom},
                "joint_records_by_fold": {},
            },
            path,
        )
    assert module_name not in sys.modules
    restored = load_ugi_training_cache_payload(path)
    assert type(restored["corpus"]["atom"]) is AtomState
    assert restored["corpus"]["atom"] == atom
