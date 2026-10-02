"""Exact Ugi topology coupling for the shared reaction-program Transformer.

The Transformer supplies local child-count and closure-endpoint scores.  This module conditions
those scores on the coarse Ugi program and returns one exact feasible topology.  It never selects a
component, copies a fragment, repairs an accepted molecule, or changes the requested program.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.model.representation.synthesis_graph import SynthesisProgramGraphRecord
from forge.model.representation.ugi_morphology import (
    UgiMorphologyProgramError,
    preorder_attached_forest_to_parents,
    sample_attached_offspring_with_exact_budget,
    sample_attached_offspring_with_exact_budget_and_cycle_rank,
)
from forge.model.sampling.ugi_closures import feasible_next_closures
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


UGI_PROGRAM_ID = "ugi_3cr_agile"


class UgiTransformerTopologyError(ValueError):
    """A Transformer topology request violates the frozen Ugi program contract."""


@dataclass(frozen=True)
class UgiTransformerTopologyPolicy:
    """Training-supported structural bounds used by the exact conditional decoder."""

    allowed_ring_sizes: tuple[int, ...]
    maximum_heavy_degree: int
    maximum_adjacent_branch_run_by_role: tuple[int, int, int]

    def __post_init__(self) -> None:
        if (
            not self.allowed_ring_sizes
            or tuple(sorted(set(self.allowed_ring_sizes))) != self.allowed_ring_sizes
            or self.allowed_ring_sizes[0] < 3
            or self.maximum_heavy_degree < 2
            or len(self.maximum_adjacent_branch_run_by_role) != len(ROLE_NAMES)
            or any(value < 0 for value in self.maximum_adjacent_branch_run_by_role)
        ):
            raise UgiTransformerTopologyError("invalid exact Ugi topology policy")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> UgiTransformerTopologyPolicy:
        branch = value.get("maximum_adjacent_branch_run_by_role")
        if not isinstance(branch, Mapping) or set(branch) != set(ROLE_NAMES):
            raise UgiTransformerTopologyError(
                "Ugi topology policy must name every precursor role exactly once"
            )
        raw_ring_sizes = value.get("allowed_ring_sizes")
        if not isinstance(raw_ring_sizes, Sequence) or isinstance(raw_ring_sizes, (str, bytes)):
            raise UgiTransformerTopologyError("Ugi topology ring-size support is missing")
        return cls(
            allowed_ring_sizes=tuple(sorted(set(int(item) for item in raw_ring_sizes))),
            maximum_heavy_degree=int(value.get("maximum_heavy_degree", -1)),
            maximum_adjacent_branch_run_by_role=tuple(int(branch[role]) for role in ROLE_NAMES),
        )

    @classmethod
    def from_support_documents(
        cls,
        closure_config: Mapping[str, Any],
        morphology_config: Mapping[str, Any],
    ) -> UgiTransformerTopologyPolicy:
        """Read topology support from the two frozen Ugi training contracts."""

        try:
            support = closure_config["support"]
            branch = morphology_config["sampling"]["maximum_adjacent_branch_run_by_role"]
            return cls.from_mapping(
                {
                    "allowed_ring_sizes": support["ring_sizes"],
                    "maximum_heavy_degree": support["maximum_heavy_degree"],
                    "maximum_adjacent_branch_run_by_role": branch,
                }
            )
        except (KeyError, TypeError, ValueError) as error:
            raise UgiTransformerTopologyError(
                "frozen Ugi topology support documents are malformed"
            ) from error

    def to_mapping(self) -> dict[str, Any]:
        return {
            "allowed_ring_sizes": list(self.allowed_ring_sizes),
            "maximum_heavy_degree": self.maximum_heavy_degree,
            "maximum_adjacent_branch_run_by_role": dict(
                zip(ROLE_NAMES, self.maximum_adjacent_branch_run_by_role, strict=True)
            ),
        }


@dataclass(frozen=True)
class UgiExactTopology:
    """One exact global sparse topology plus its role-local offspring words."""

    parents: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    offspring_by_role: tuple[np.ndarray, np.ndarray, np.ndarray]


def _role_targets(record: SynthesisProgramGraphRecord) -> dict[str, tuple[int, int, int, int]]:
    states = record.role_morphology_states
    if states is None:
        raise UgiTransformerTopologyError("exact Ugi topology requires role morphology states")
    output: dict[str, tuple[int, int, int, int]] = {}
    for block in record.component_blocks:
        if block.role not in ROLE_NAMES:
            continue
        values = np.unique(states[block.start : block.stop], axis=0)
        if values.shape != (1, 4) or np.any(values[0] < 1):
            raise UgiTransformerTopologyError(f"role morphology is not constant for {block.role}")
        output[block.role] = tuple(int(item) - 1 for item in values[0])
    if set(output) != set(ROLE_NAMES):
        raise UgiTransformerTopologyError("Ugi layout does not contain all precursor roles")
    return output


def _root_aligned_permutation(
    parents: np.ndarray,
    fixed_root_positions: Sequence[int],
) -> np.ndarray:
    """Map forest roots onto anonymous fixed attachment slots while preserving parent order."""

    roots = np.flatnonzero(parents < 0).tolist()
    fixed_roots = [int(value) for value in fixed_root_positions]
    if len(roots) != len(fixed_roots) or len(set(fixed_roots)) != len(fixed_roots):
        raise UgiTransformerTopologyError("decoded and adapter-fixed attachment counts disagree")
    node_count = len(parents)
    remaining_source = [index for index in range(node_count) if index not in set(roots)]
    remaining_target = [index for index in range(node_count) if index not in set(fixed_roots)]
    permutation = np.empty(node_count, dtype=np.int64)
    for source, target in zip(roots, fixed_roots, strict=True):
        permutation[source] = target
    for source, target in zip(remaining_source, remaining_target, strict=True):
        permutation[source] = target
    for child, parent in enumerate(parents.tolist()):
        if parent >= 0 and int(permutation[parent]) >= int(permutation[child]):
            raise UgiTransformerTopologyError("root alignment would violate sparse parent ordering")
    return permutation


def _select_role_closures(
    *,
    offspring: np.ndarray,
    attachment_count: int,
    cycle_rank: int,
    exterior: np.ndarray,
    permutation: np.ndarray,
    closure_left_logits: np.ndarray,
    closure_right_logits: np.ndarray,
    first_slot: int,
    policy: UgiTransformerTopologyPolicy,
) -> tuple[list[int], list[int]]:
    selected: list[tuple[int, int]] = []
    oriented: list[tuple[int, int]] = []
    for local_slot in range(cycle_rank):
        candidates = feasible_next_closures(
            offspring,
            selected=selected,
            remaining_closures_including_next=cycle_rank - local_slot,
            allowed_ring_sizes=policy.allowed_ring_sizes,
            maximum_heavy_degree=policy.maximum_heavy_degree,
            attachment_count=attachment_count,
        )
        if not candidates.edges:
            raise UgiTransformerTopologyError(
                "exact Ugi tree cannot realize its declared cycle rank"
            )
        slot = first_slot + local_slot
        scored: list[tuple[float, int, int, int, int]] = []
        for left, right in candidates.edges:
            global_left = int(exterior[int(permutation[left])])
            global_right = int(exterior[int(permutation[right])])
            direct = float(
                closure_left_logits[slot, global_left] + closure_right_logits[slot, global_right]
            )
            reverse = float(
                closure_left_logits[slot, global_right] + closure_right_logits[slot, global_left]
            )
            if reverse > direct:
                scored.append((reverse, -left, -right, global_right, global_left))
            else:
                scored.append((direct, -left, -right, global_left, global_right))
        _, negative_left, negative_right, global_left, global_right = max(scored)
        selected.append((-negative_left, -negative_right))
        oriented.append((global_left, global_right))
    return [left for left, _ in oriented], [right for _, right in oriented]


def decode_ugi_exact_topology(
    predictions: Mapping[str, Any],
    *,
    index: int,
    record: SynthesisProgramGraphRecord,
    policy: UgiTransformerTopologyPolicy,
    generator: Any,
) -> UgiExactTopology:
    """Condition one Transformer endpoint on an exact feasible Ugi morphology program."""

    if torch is None or record.program_id != UGI_PROGRAM_ID:
        raise UgiTransformerTopologyError("exact coupled topology currently supports Ugi only")
    if "offspring" not in predictions:
        raise UgiTransformerTopologyError("Transformer checkpoint has no offspring topology head")
    targets = _role_targets(record)
    parents = record.graph.parents.copy()
    closure_left = np.zeros(record.graph.closure_count, dtype=np.int64)
    closure_right = np.zeros(record.graph.closure_count, dtype=np.int64)
    offspring_rows: list[np.ndarray] = []
    closure_cursor = 0
    block_by_role = {
        block.role: block for block in record.component_blocks if block.role in ROLE_NAMES
    }
    for role_index, role in enumerate(ROLE_NAMES):
        block = block_by_role[role]
        exterior = np.flatnonzero(
            (np.arange(record.node_count) >= block.start)
            & (np.arange(record.node_count) < block.stop)
            & (record.core_position_states == 1)
        ).astype(np.int64)
        node_count, junction_budget, cycle_rank, attachment_count = targets[role]
        if len(exterior) != node_count or node_count < 1:
            raise UgiTransformerTopologyError(f"{role} exterior size changed")
        fixed_roots = [
            local
            for local, node in enumerate(exterior.tolist())
            if bool(record.fixed_parent_bond_mask[node])
            and int(record.core_position_states[int(record.graph.parents[node])]) > 1
        ]
        if len(fixed_roots) != attachment_count:
            raise UgiTransformerTopologyError(f"{role} attachment contract changed")
        logits = predictions["offspring"][index, exterior].detach().to("cpu")
        try:
            if cycle_rank:
                offspring = sample_attached_offspring_with_exact_budget_and_cycle_rank(
                    logits,
                    junction_budget=junction_budget,
                    cycle_rank=cycle_rank,
                    attachment_count=attachment_count,
                    generator=generator,
                    allowed_ring_sizes=policy.allowed_ring_sizes,
                    maximum_heavy_degree=policy.maximum_heavy_degree,
                    maximum_adjacent_branch_run=(
                        policy.maximum_adjacent_branch_run_by_role[role_index]
                    ),
                )
            else:
                offspring = sample_attached_offspring_with_exact_budget(
                    logits,
                    junction_budget=junction_budget,
                    attachment_count=attachment_count,
                    generator=generator,
                    maximum_adjacent_branch_run=(
                        policy.maximum_adjacent_branch_run_by_role[role_index]
                    ),
                )
        except UgiMorphologyProgramError as error:
            raise UgiTransformerTopologyError(str(error)) from error
        local_parents = preorder_attached_forest_to_parents(
            offspring,
            attachment_count=attachment_count,
        )
        permutation = _root_aligned_permutation(local_parents, fixed_roots)
        for source_child, source_parent in enumerate(local_parents.tolist()):
            target_child = int(exterior[int(permutation[source_child])])
            if source_parent < 0:
                if not bool(record.fixed_parent_bond_mask[target_child]):
                    raise UgiTransformerTopologyError("decoded root lost its fixed core attachment")
                continue
            if bool(record.fixed_parent_bond_mask[target_child]):
                raise UgiTransformerTopologyError("decoded non-root overlaps a fixed attachment")
            target_parent = int(exterior[int(permutation[source_parent])])
            if target_parent >= target_child:
                raise UgiTransformerTopologyError("decoded Ugi parent does not precede its child")
            parents[target_child] = target_parent
        local_left, local_right = _select_role_closures(
            offspring=offspring,
            attachment_count=attachment_count,
            cycle_rank=cycle_rank,
            exterior=exterior,
            permutation=permutation,
            closure_left_logits=predictions["closure_left"][index].detach().to("cpu").numpy(),
            closure_right_logits=predictions["closure_right"][index].detach().to("cpu").numpy(),
            first_slot=closure_cursor,
            policy=policy,
        )
        if cycle_rank:
            closure_left[closure_cursor : closure_cursor + cycle_rank] = local_left
            closure_right[closure_cursor : closure_cursor + cycle_rank] = local_right
        closure_cursor += cycle_rank
        offspring_rows.append(offspring)
    if closure_cursor != record.graph.closure_count:
        raise UgiTransformerTopologyError("role cycle ranks do not conserve closure slots")
    return UgiExactTopology(
        parents=parents,
        closure_left=closure_left,
        closure_right=closure_right,
        offspring_by_role=tuple(offspring_rows),  # type: ignore[arg-type]
    )


__all__ = [
    "UGI_PROGRAM_ID",
    "UgiExactTopology",
    "UgiTransformerTopologyError",
    "UgiTransformerTopologyPolicy",
    "decode_ugi_exact_topology",
]
