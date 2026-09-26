"""Sparse discrete-flow representations and neural architectures.

The public surface is lazy so importing a lightweight representation module does not eagerly import
PyTorch or create corpus/model initialization cycles.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PUBLIC = {
    "AtomVocabularyError": "forge.model.vocabulary",
    "ClosureCandidateSet": "forge.model.ugi_closure_placement",
    "UgiClosurePlacementError": "forge.model.ugi_closure_placement",
    "UgiJointSparseFlow": "forge.model.ugi_joint_sparse_flow",
    "UgiJointSparseFlowError": "forge.model.ugi_joint_sparse_flow",
    "UgiJointSparseRecord": "forge.model.ugi_joint_sparse_flow",
    "UgiJointSparseTerminal": "forge.model.ugi_joint_sparse_flow",
    "UgiSparseClosureScorer": "forge.model.ugi_closure_placement",
    "closure_set_loss": "forge.model.ugi_closure_placement",
    "collate_ugi_joint_sparse_records": "forge.model.ugi_joint_sparse_flow",
    "feasible_next_closures": "forge.model.ugi_closure_placement",
    "load_atom_vocabulary": "forge.model.vocabulary",
    "project_joint_sparse_record": "forge.model.ugi_joint_sparse_flow",
    "sample_sparse_closures": "forge.model.ugi_closure_placement",
    "ugi_joint_sparse_loss": "forge.model.ugi_joint_sparse_flow",
}

__all__ = sorted(_PUBLIC)


def __getattr__(name: str) -> Any:
    module_name = _PUBLIC.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
