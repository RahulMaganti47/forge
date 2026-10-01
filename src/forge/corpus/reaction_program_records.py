"""Shared admission predicate for reaction-program structure supervision."""

from __future__ import annotations

import json
from collections.abc import Mapping

STRUCTURE_SUPERVISION_DISPOSITIONS = frozenset(
    {
        "admit_exact",
        "admit_transform_consistency",
    }
)


def admits_reaction_program_structure(row: Mapping[str, str]) -> bool:
    """Return whether exact graph and program semantics may supervise the model.

    Experimental execution and deterministic transform consistency stay separate evidence labels.
    Neither implies route closure, procurement, activity, or synthesis-success probability.
    """

    return (
        row.get("disposition") in STRUCTURE_SUPERVISION_DISPOSITIONS
        and row.get("semantic_origin_status") == "exact"
    )


def repeat_component_smiles(row: Mapping[str, str]) -> tuple[str, ...]:
    """Return the exact ordered repeat-component sequence for one program row.

    Atlas v1 represents homogeneous repeated programs with one SMILES plus a depth. Atlas v2 adds
    an ordered JSON sequence so different source-linked components can occupy different steps. This
    compatibility helper makes that schema distinction explicit instead of letting consumers
    silently repeat the first component for a mixed program.
    """

    depth_text = row.get("step_count", "")
    if not depth_text:
        raise ValueError("reaction-program row has no step_count")
    depth = int(depth_text)
    if depth < 1:
        raise ValueError("reaction-program step_count must be positive")
    encoded = row.get("repeat_component_smiles_json", "")
    if encoded:
        try:
            values = json.loads(encoded)
        except json.JSONDecodeError as exc:
            raise ValueError("repeat_component_smiles_json is malformed") from exc
        if (
            not isinstance(values, list)
            or len(values) != depth
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError("repeat_component_smiles_json does not match program depth")
        return tuple(values)
    homogeneous = row.get("repeat_component_smiles", "")
    if not homogeneous:
        raise ValueError("reaction-program row has no repeat component")
    return (homogeneous,) * depth


__all__ = [
    "STRUCTURE_SUPERVISION_DISPOSITIONS",
    "admits_reaction_program_structure",
    "repeat_component_smiles",
]
