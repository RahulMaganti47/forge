"""Exact Ugi-core-anchored morphology programs for Phase 1 generation.

The qualified five-atom Ugi product core is fixed by the adapter.  This module
represents each precursor-derived exterior as one atom-level rooted tree whose
parent is its adapter-defined core port, plus sparse within-origin closures.
Component identifiers are retained only as data provenance and split keys;
they never enter model tensors.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

import numpy as np

from forge.model.phase1_tree_topology_flow import TreeTopologyFlowError
from forge.model.ugi_adapter_features import (
    ORIGIN_TO_INDEX,
    UgiL1SupportTrainingRecord,
)
from forge.potency.annotations import ROLE_NAMES


class UgiMorphologyProgramError(RuntimeError):
    """Raised when a Ugi morphology cannot be represented exactly."""


ROLE_TO_INDEX = {role: index for index, role in enumerate(ROLE_NAMES)}


@dataclass(frozen=True)
class UgiComponentMorphology:
    """Topology target for one generated precursor exterior.

    The first exterior atom has an implicit parent at the fixed Ugi core port.
    Every other parent is encoded by the preorder offspring word.  Closure
    indices are local to this exterior sequence.
    """

    role: str
    component_key: str
    offspring: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    attachment_count: int = 1

    @property
    def node_count(self) -> int:
        return int(self.offspring.size)

    @property
    def junction_budget(self) -> int:
        # Every exterior atom has one parent in the complete molecular tree;
        # the first atom's parent is the fixed reaction-core port.
        return int(np.maximum(self.offspring.astype(np.int64) - 1, 0).sum())

    @property
    def cycle_rank(self) -> int:
        return int(self.closure_left.size)

    @property
    def signature(self) -> tuple[Any, ...]:
        return (
            self.attachment_count,
            tuple(self.offspring.tolist()),
            tuple(zip(self.closure_left.tolist(), self.closure_right.tolist(), strict=True)),
        )


@dataclass(frozen=True)
class UgiMorphologyProgram:
    """Generated global program for three atom-level precursor exteriors."""

    node_counts: tuple[int, int, int]
    junction_budgets: tuple[int, int, int]
    cycle_ranks: tuple[int, int, int]
    attachment_counts: tuple[int, int, int] = (1, 1, 1)

    @property
    def node_count(self) -> int:
        return sum(self.node_counts)


@dataclass(frozen=True)
class UgiProductMorphology:
    """One complete Ugi morphology target with no atom or bond chemistry."""

    product_id: str
    components: tuple[UgiComponentMorphology, UgiComponentMorphology, UgiComponentMorphology]

    @property
    def program(self) -> UgiMorphologyProgram:
        return UgiMorphologyProgram(
            node_counts=tuple(component.node_count for component in self.components),
            junction_budgets=tuple(component.junction_budget for component in self.components),
            cycle_ranks=tuple(component.cycle_rank for component in self.components),
            attachment_counts=tuple(component.attachment_count for component in self.components),
        )


def attached_tree_junction_contributions(offspring: np.ndarray) -> np.ndarray:
    """Return degree excess for a tree attached to one fixed core parent."""

    if (
        offspring.ndim != 1
        or offspring.size < 1
        or not np.issubdtype(offspring.dtype, np.integer)
        or np.any(offspring < 0)
    ):
        raise UgiMorphologyProgramError("attached offspring must be nonempty nonnegative integers")
    return np.maximum(offspring.astype(np.int64) - 1, 0)


def attached_tree_matches_program(
    offspring: np.ndarray,
    *,
    node_count: int,
    junction_budget: int,
    attachment_count: int = 1,
) -> bool:
    """Check one exact core-attached exterior forest and its junction budget."""

    if offspring.shape != (node_count,):
        return False
    try:
        preorder_attached_forest_to_parents(offspring, attachment_count=attachment_count)
        contributions = attached_tree_junction_contributions(offspring)
    except (TreeTopologyFlowError, UgiMorphologyProgramError):
        return False
    return int(contributions.sum()) == junction_budget


def preorder_attached_forest_to_parents(
    offspring: np.ndarray,
    *,
    attachment_count: int,
) -> np.ndarray:
    """Decode preorder trees whose roots share one implicit reaction-core port.

    Root atoms receive parent ``-1``.  The implicit core atom is not a neural
    node, but its declared child count makes secondary amines and cyclic heads
    representable without introducing a fragment vocabulary.
    """

    if (
        offspring.ndim != 1
        or offspring.size < 1
        or np.any(offspring < 0)
        or not 1 <= attachment_count <= offspring.size
    ):
        raise UgiMorphologyProgramError("invalid core-attached offspring forest")
    parents = np.full(offspring.size, -1, dtype=np.int64)
    stack: list[list[int]] = [[-1, int(attachment_count)]]
    for node in range(offspring.size):
        while stack and stack[-1][1] == 0:
            stack.pop()
        if not stack:
            raise UgiMorphologyProgramError("offspring forest exhausts its core roots early")
        parent = stack[-1][0]
        stack[-1][1] -= 1
        parents[node] = parent
        stack.append([node, int(offspring[node])])
    while stack and stack[-1][1] == 0:
        stack.pop()
    if stack or int(offspring.sum()) != offspring.size - attachment_count:
        raise UgiMorphologyProgramError("offspring forest does not close at the core port")
    return parents


def split_ugi_support_morphology(
    record: UgiL1SupportTrainingRecord,
    component_keys: Mapping[str, str],
    *,
    product_id: str,
) -> UgiProductMorphology:
    """Project a chemistry-rich support record to three topology-only targets."""

    if set(component_keys) != set(ROLE_NAMES):
        raise UgiMorphologyProgramError("component keys must match the three Ugi roles")
    graph = record.support_graph
    adapter = record.support_adapter
    parents = graph.parents
    if graph.node_count != adapter.node_count:
        raise UgiMorphologyProgramError("support graph and adapter semantics are misaligned")

    components: list[UgiComponentMorphology] = []
    covered_exterior: set[int] = set()
    for role in ROLE_NAMES:
        origin = ORIGIN_TO_INDEX[role]
        exterior = np.flatnonzero(
            (adapter.origin_states == origin) & ~adapter.core_membership
        ).astype(np.int64)
        if exterior.size < 1:
            raise UgiMorphologyProgramError(f"{product_id} has an empty {role} exterior")
        # The selected hybrid serialization was accepted because every Ugi
        # precursor origin forms one contiguous preorder block.
        if not np.array_equal(exterior, np.arange(exterior[0], exterior[-1] + 1)):
            raise UgiMorphologyProgramError(f"{product_id} has a noncontiguous {role} exterior")
        selected = set(exterior.tolist())
        covered_exterior.update(selected)
        old_to_local = {old: local for local, old in enumerate(exterior.tolist())}
        local_parents = np.full(exterior.size, -1, dtype=np.int64)
        root_count = 0
        for local, old in enumerate(exterior.tolist()):
            parent = int(parents[old])
            if parent in selected:
                local_parent = old_to_local[parent]
                if local_parent >= local:
                    raise UgiMorphologyProgramError(
                        f"{product_id} {role} parent does not precede its child"
                    )
                local_parents[local] = local_parent
            else:
                if not adapter.core_membership[parent]:
                    raise UgiMorphologyProgramError(
                        f"{product_id} {role} exterior root is not core-attached"
                    )
                root_count += 1
        if root_count < 1:
            raise UgiMorphologyProgramError(f"{product_id} {role} does not have a core boundary")
        offspring = np.zeros(exterior.size, dtype=np.int64)
        for parent in local_parents:
            if parent >= 0:
                offspring[parent] += 1
        if not np.array_equal(
            preorder_attached_forest_to_parents(offspring, attachment_count=root_count),
            local_parents,
        ):
            raise UgiMorphologyProgramError(
                f"{product_id} {role} exterior is not one canonical preorder forest"
            )

        closures: list[tuple[int, int]] = []
        for left, right in zip(graph.closure_left, graph.closure_right, strict=True):
            left_index = int(left)
            right_index = int(right)
            left_selected = left_index in selected
            right_selected = right_index in selected
            if left_selected != right_selected:
                raise UgiMorphologyProgramError(
                    f"{product_id} contains a cross-boundary residual closure"
                )
            if left_selected:
                closures.append(
                    tuple(sorted((old_to_local[left_index], old_to_local[right_index])))
                )
        closures.sort()
        if len(closures) != len(set(closures)):
            raise UgiMorphologyProgramError(f"{product_id} contains duplicate {role} closures")
        components.append(
            UgiComponentMorphology(
                role=role,
                component_key=str(component_keys[role]),
                offspring=offspring,
                closure_left=np.asarray([left for left, _ in closures], dtype=np.int64),
                closure_right=np.asarray([right for _, right in closures], dtype=np.int64),
                attachment_count=root_count,
            )
        )

    expected_exterior = set(np.flatnonzero(~adapter.core_membership).tolist())
    if covered_exterior != expected_exterior:
        raise UgiMorphologyProgramError(
            f"{product_id} exterior atoms are not partitioned by precursor origin"
        )
    return UgiProductMorphology(
        product_id=product_id,
        components=tuple(components),  # type: ignore[arg-type]
    )


def unique_component_morphologies(
    records: Sequence[UgiProductMorphology],
) -> dict[tuple[str, str], UgiComponentMorphology]:
    """Deduplicate topology supervision without allowing catalog IDs into a model."""

    unique: dict[tuple[str, str], UgiComponentMorphology] = {}
    for record in records:
        for component in record.components:
            key = (component.role, component.component_key)
            previous = unique.get(key)
            if previous is not None and previous.signature != component.signature:
                raise UgiMorphologyProgramError(
                    f"component morphology changes across products: {component.role}"
                )
            unique.setdefault(key, component)
    return unique


def component_weighted_offspring_marginals(
    records: Sequence[UgiProductMorphology],
    *,
    maximum_children: int,
    probability_floor: float = 1e-5,
) -> np.ndarray:
    """Estimate role-specific full-support sources with equal component mass."""

    if maximum_children < 1 or not 0 < probability_floor < 1:
        raise UgiMorphologyProgramError("invalid offspring marginal support")
    unique = unique_component_morphologies(records)
    return component_offspring_marginals(
        tuple(unique.values()),
        maximum_children=maximum_children,
        probability_floor=probability_floor,
    )


def component_offspring_marginals(
    components: Sequence[UgiComponentMorphology],
    *,
    maximum_children: int,
    probability_floor: float = 1e-5,
    component_weights: Mapping[tuple[str, str], float] | None = None,
) -> np.ndarray:
    """Estimate equal-component source marginals from an explicit fold subset."""

    if maximum_children < 1 or not 0 < probability_floor < 1 or not components:
        raise UgiMorphologyProgramError("invalid component offspring marginal request")
    marginals = np.full(
        (len(ROLE_NAMES), maximum_children + 1),
        probability_floor,
        dtype=np.float64,
    )
    role_component_counts = Counter()
    seen: set[tuple[str, str]] = set()
    for component in components:
        key = (component.role, component.component_key)
        if key in seen:
            continue
        seen.add(key)
        role = component.role
        if component.offspring.max(initial=0) > maximum_children:
            raise UgiMorphologyProgramError("offspring count exceeds declared support")
        histogram = np.bincount(
            component.offspring,
            minlength=maximum_children + 1,
        ).astype(np.float64)
        histogram /= histogram.sum()
        weight = 1.0 if component_weights is None else float(component_weights.get(key, -1.0))
        if weight <= 0:
            raise UgiMorphologyProgramError("component source marginal lacks positive weight")
        marginals[ROLE_TO_INDEX[role]] += weight * histogram
        role_component_counts[role] += 1
    if set(role_component_counts) != set(ROLE_NAMES):
        raise UgiMorphologyProgramError("component-weighted marginals lack a Ugi role")
    return marginals / marginals.sum(axis=1, keepdims=True)


def component_weighted_program_pool(
    records: Sequence[UgiProductMorphology],
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Return one topology program per unique component and precursor role."""

    unique = unique_component_morphologies(records)
    return component_program_pool(tuple(unique.values()))


def component_program_pool(
    components: Sequence[UgiComponentMorphology],
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Return role pools from one explicit, already split component census."""

    pools: list[list[tuple[int, int, int, int]]] = [[] for _ in ROLE_NAMES]
    seen: set[tuple[str, str]] = set()
    for component in sorted(components, key=lambda value: (value.role, value.component_key)):
        key = (component.role, component.component_key)
        if key in seen:
            continue
        seen.add(key)
        role = component.role
        pools[ROLE_TO_INDEX[role]].append(
            (
                component.node_count,
                component.junction_budget,
                component.cycle_rank,
                component.attachment_count,
            )
        )
    if any(not pool for pool in pools):
        raise UgiMorphologyProgramError("program pool lacks a Ugi precursor role")
    return tuple(tuple(pool) for pool in pools)


def sample_component_weighted_programs(
    pools: tuple[tuple[tuple[int, int, int, int], ...], ...],
    *,
    count: int,
    rng: np.random.Generator,
) -> tuple[UgiMorphologyProgram, ...]:
    """Sample role programs independently without selecting component identities."""

    if len(pools) != len(ROLE_NAMES) or count < 1 or any(not pool for pool in pools):
        raise UgiMorphologyProgramError("invalid component-weighted program request")
    output = []
    for _ in range(count):
        selected = [pool[int(rng.integers(len(pool)))] for pool in pools]
        output.append(
            UgiMorphologyProgram(
                node_counts=tuple(value[0] for value in selected),
                junction_budgets=tuple(value[1] for value in selected),
                cycle_ranks=tuple(value[2] for value in selected),
                attachment_counts=tuple(value[3] for value in selected),
            )
        )
    return tuple(output)


def _attached_choice_valid(
    *,
    position: int,
    pending: int,
    remaining_budget: int,
    children: int,
    node_count: int,
) -> tuple[int, int] | None:
    next_pending = pending - 1 + children
    positions_after = node_count - position - 1
    next_budget = remaining_budget - max(0, children - 1)
    if (
        next_budget < 0
        or next_pending < 0
        or next_pending > positions_after
        or (positions_after > 0 and next_pending == 0)
        or (positions_after == 0 and next_pending != 0)
    ):
        return None
    return next_pending, next_budget


def _unbudgeted_attached_choice_valid(
    *,
    position: int,
    pending: int,
    children: int,
    node_count: int,
) -> int | None:
    """Advance one attached-forest word without prescribing its branch count."""

    next_pending = pending - 1 + children
    positions_after = node_count - position - 1
    if (
        next_pending < 0
        or next_pending > positions_after
        or (positions_after > 0 and next_pending == 0)
        or (positions_after == 0 and next_pending != 0)
    ):
        return None
    return next_pending


def _sample_attached_offspring_without_budget_once(
    logits: Any,
    *,
    attachment_count: int = 1,
    generator: Any,
) -> np.ndarray:
    """Sample one valid attached forest without a graph-branch constraint.

    This is the exact size-only topology decoder.  Neural scores determine the
    offspring word while a dynamic-programming mask enforces only closure of
    the declared number of core-attached roots.  The resulting junction budget
    is measured from the sampled word rather than supplied as a condition.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise UgiMorphologyProgramError("attached-tree sampling requires torch") from exc
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise UgiMorphologyProgramError("offspring logits must be [nodes, child classes]")
    node_count, child_classes = logits.shape
    if not 1 <= attachment_count <= node_count:
        raise UgiMorphologyProgramError("attachment count exceeds the size-only tree support")

    maximum_children = child_classes - 1
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)
    suffix: list[dict[int, Any]] = [dict() for _ in range(node_count + 1)]
    suffix[node_count][0] = logits.new_tensor(0.0, dtype=torch.float64)
    for position in range(node_count - 1, -1, -1):
        positions_including_current = node_count - position
        for pending in range(1, positions_including_current + 1):
            terms = []
            for children in range(maximum_children + 1):
                next_pending = _unbudgeted_attached_choice_valid(
                    position=position,
                    pending=pending,
                    children=children,
                    node_count=node_count,
                )
                if next_pending is not None and next_pending in suffix[position + 1]:
                    terms.append(
                        log_probabilities[position, children] + suffix[position + 1][next_pending]
                    )
            if terms:
                suffix[position][pending] = torch.logsumexp(torch.stack(terms), dim=0)
    if attachment_count not in suffix[0]:
        raise UgiMorphologyProgramError("size-only attached-tree decoder found no valid completion")

    output = np.zeros(node_count, dtype=np.int64)
    pending = attachment_count
    for position in range(node_count):
        choices = []
        states = []
        weights = []
        for children in range(maximum_children + 1):
            next_pending = _unbudgeted_attached_choice_valid(
                position=position,
                pending=pending,
                children=children,
                node_count=node_count,
            )
            if next_pending is None or next_pending not in suffix[position + 1]:
                continue
            choices.append(children)
            states.append(next_pending)
            weights.append(
                log_probabilities[position, children] + suffix[position + 1][next_pending]
            )
        selected = int(
            torch.multinomial(torch.stack(weights).softmax(dim=0), 1, generator=generator)
        )
        output[position] = choices[selected]
        pending = states[selected]
    try:
        preorder_attached_forest_to_parents(output, attachment_count=attachment_count)
    except UgiMorphologyProgramError as error:
        raise UgiMorphologyProgramError(
            "size-only attached-tree decoder violated its contract"
        ) from error
    return output


def maximum_adjacent_branch_graph_run(
    offspring: Sequence[int] | np.ndarray,
    *,
    attachment_count: int = 1,
) -> int:
    """Return the largest connected set of adjacent branch nodes in the tree.

    Branch adjacency is evaluated on decoded parent-child edges, never on
    neighboring positions in the preorder serialization.  A branch node has
    at least two tree children; because every exterior root is attached to the
    reaction core, this corresponds to heavy degree at least three before
    sparse closure edges are added.
    """

    values = np.asarray(offspring, dtype=np.int64)
    parents = preorder_attached_forest_to_parents(values, attachment_count=attachment_count)
    branched = values >= 2
    if not np.any(branched):
        return 0
    component_size = np.ones(len(values), dtype=np.int64)
    maximum = 1
    for index in range(len(values) - 1, -1, -1):
        parent = int(parents[index])
        if parent >= 0 and branched[index] and branched[parent]:
            component_size[parent] += component_size[index]
            maximum = max(maximum, int(component_size[parent]))
    return maximum


def sample_attached_offspring_without_budget(
    logits: Any,
    *,
    attachment_count: int = 1,
    generator: Any,
    maximum_adjacent_branch_run: int | None = None,
    maximum_rejection_attempts: int = 64,
) -> np.ndarray:
    """Sample size-only topology conditional on decoded tree-branch spacing."""

    if maximum_adjacent_branch_run is not None and maximum_adjacent_branch_run < 0:
        raise UgiMorphologyProgramError("adjacent branch-run support cannot be negative")
    if maximum_rejection_attempts < 1:
        raise UgiMorphologyProgramError("exact program rejection count must be positive")
    for _ in range(maximum_rejection_attempts):
        candidate = _sample_attached_offspring_without_budget_once(
            logits,
            attachment_count=attachment_count,
            generator=generator,
        )
        if (
            maximum_adjacent_branch_run is None
            or maximum_adjacent_branch_graph_run(candidate, attachment_count=attachment_count)
            <= maximum_adjacent_branch_run
        ):
            return candidate
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise UgiMorphologyProgramError("attached-tree sampling requires torch") from exc
    neutral_logits = torch.zeros_like(logits)
    for _ in range(4 * maximum_rejection_attempts):
        candidate = _sample_attached_offspring_without_budget_once(
            neutral_logits,
            attachment_count=attachment_count,
            generator=generator,
        )
        if (
            maximum_adjacent_branch_graph_run(candidate, attachment_count=attachment_count)
            <= maximum_adjacent_branch_run
        ):
            return candidate
    raise UgiMorphologyProgramError(
        "size-only attached-tree decoder exhausted branch-spacing support"
    )


@cache
def attached_program_feasible(
    node_count: int,
    junction_budget: int,
    maximum_children: int,
    attachment_count: int = 1,
) -> bool:
    """Return whether the declared child alphabet can realize a program."""

    if not 1 <= attachment_count <= node_count:
        return False
    states = {(attachment_count, junction_budget)}
    for position in range(node_count):
        next_states = set()
        for pending, budget in states:
            for children in range(maximum_children + 1):
                state = _attached_choice_valid(
                    position=position,
                    pending=pending,
                    remaining_budget=budget,
                    children=children,
                    node_count=node_count,
                )
                if state is not None:
                    next_states.add(state)
        states = next_states
    return (0, 0) in states


@dataclass(frozen=True)
class _AttachedProgramSchedule:
    """Transition structure of the exact attached-tree dynamic program.

    Which ``(pending, budget, incoming_run)`` states are reachable, and which child count moves
    between them, is decided entirely by the declared program coordinates.  It does not depend on
    a single neural score.  Enumerating it once per program shape and reusing it turns the
    per-sample cost from a Python walk over every state-and-child pair into a handful of tensor
    operations, without changing which terms are combined or in what order.
    """

    states: tuple[tuple[tuple[int, int, int], ...], ...]
    transitions: tuple[np.ndarray, ...]
    initial_index: int | None


@cache
def _attached_program_schedule(
    *,
    node_count: int,
    junction_budget: int,
    maximum_children: int,
    attachment_count: int,
    branch_limit: int,
) -> _AttachedProgramSchedule:
    """Enumerate the reachable suffix states and their child-indexed successors."""

    # A branch-run budget at least as large as the tree can never bind: the run counter increases
    # by at most one per position, so ``next_run`` is bounded by ``node_count``.  Every state that
    # the traceback can actually reach then has a suffix value independent of its run coordinate,
    # by backward induction from the all-zero terminal row.  Carrying the coordinate anyway
    # multiplies the state space by ``node_count`` and computes each value ``node_count`` times.
    unbounded_runs = branch_limit >= node_count
    run_values = (0,) if unbounded_runs else tuple(range(branch_limit + 1))
    terminal_states = tuple((0, 0, run) for run in run_values)
    states: list[tuple[tuple[int, int, int], ...]] = [()] * (node_count + 1)
    states[node_count] = terminal_states
    transitions: list[np.ndarray] = [
        np.empty((0, maximum_children + 1), dtype=np.int64) for _ in range(node_count)
    ]
    index_by_state: dict[tuple[int, int, int], int] = {
        state: index for index, state in enumerate(terminal_states)
    }
    for position in range(node_count - 1, -1, -1):
        positions_including_current = node_count - position
        current: list[tuple[int, int, int]] = []
        rows: list[list[int]] = []
        for pending in range(1, positions_including_current + 1):
            for budget in range(junction_budget + 1):
                for incoming_run in run_values:
                    row = [-1] * (maximum_children + 1)
                    present = False
                    for children in range(maximum_children + 1):
                        state = _attached_choice_valid(
                            position=position,
                            pending=pending,
                            remaining_budget=budget,
                            children=children,
                            node_count=node_count,
                        )
                        next_run = incoming_run + 1 if children >= 2 else 0
                        successor_run = 0 if unbounded_runs else next_run
                        suffix_state = None if state is None else (*state, successor_run)
                        if (
                            (unbounded_runs or next_run <= branch_limit)
                            and suffix_state is not None
                            and suffix_state in index_by_state
                        ):
                            row[children] = index_by_state[suffix_state]
                            present = True
                    if present:
                        current.append((pending, budget, incoming_run))
                        rows.append(row)
        states[position] = tuple(current)
        transitions[position] = (
            np.asarray(rows, dtype=np.int64)
            if rows
            else np.empty((0, maximum_children + 1), dtype=np.int64)
        )
        index_by_state = {state: index for index, state in enumerate(current)}
    initial_state = (attachment_count, junction_budget, 0)
    initial_index = states[0].index(initial_state) if initial_state in set(states[0]) else None
    return _AttachedProgramSchedule(
        states=tuple(states),
        transitions=tuple(transitions),
        initial_index=initial_index,
    )


def _sample_attached_offspring_with_exact_budget(
    logits: Any,
    *,
    junction_budget: int,
    attachment_count: int = 1,
    generator: Any,
    maximum_adjacent_branch_run: int | None = None,
) -> np.ndarray:
    """Sample one exact core-attached tree from conditional neural scores."""

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise UgiMorphologyProgramError("attached-tree sampling requires torch") from exc
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise UgiMorphologyProgramError("offspring logits must be [nodes, child classes]")
    node_count, child_classes = logits.shape
    maximum_children = child_classes - 1
    if not attached_program_feasible(
        node_count,
        junction_budget,
        maximum_children,
        attachment_count,
    ):
        raise UgiMorphologyProgramError("declared attached-tree program is infeasible")
    if maximum_adjacent_branch_run is not None and maximum_adjacent_branch_run < 0:
        raise UgiMorphologyProgramError("adjacent branch-run support cannot be negative")
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)
    branch_limit = (
        node_count if maximum_adjacent_branch_run is None else maximum_adjacent_branch_run
    )
    schedule = _attached_program_schedule(
        node_count=int(node_count),
        junction_budget=int(junction_budget),
        maximum_children=int(maximum_children),
        attachment_count=int(attachment_count),
        branch_limit=int(branch_limit),
    )
    if schedule.initial_index is None:
        raise UgiMorphologyProgramError(
            "exact attached-tree decoder found no branch-run-compatible completion"
        )

    # Backward pass.  Every reachable state at one position is combined in a single reduction over
    # the child axis.  Absent moves carry -inf, which contributes exp(-inf) = 0 to the logsumexp
    # sum and is therefore an exact additive identity: the surviving terms are the same float64
    # values, in the same child order, as the per-state stack this replaces.
    suffix_values: list[Any] = [None] * (node_count + 1)
    suffix_values[node_count] = log_probabilities.new_zeros(len(schedule.states[node_count]))
    for position in range(node_count - 1, -1, -1):
        transition = torch.from_numpy(schedule.transitions[position])
        valid = transition >= 0
        gathered = suffix_values[position + 1][transition.clamp(min=0)]
        terms = log_probabilities[position][None, :] + gathered
        suffix_values[position] = torch.logsumexp(terms.masked_fill(~valid, -torch.inf), dim=1)

    output = np.zeros(node_count, dtype=np.int64)
    state_index = schedule.initial_index
    for position in range(node_count):
        transition = schedule.transitions[position][state_index]
        choices = np.flatnonzero(transition >= 0)
        # The same two float64 numbers, added in the same ascending-child order as the per-choice
        # stack this replaces, so the softmax and the generator draw are unchanged.
        weights = (
            log_probabilities[position][torch.from_numpy(choices)]
            + suffix_values[position + 1][torch.from_numpy(transition[choices])]
        )
        selected = int(
            torch.multinomial(
                weights.softmax(dim=0),
                1,
                generator=generator,
            )
        )
        output[position] = int(choices[selected])
        state_index = int(transition[choices[selected]])
    if not attached_tree_matches_program(
        output,
        node_count=node_count,
        junction_budget=junction_budget,
        attachment_count=attachment_count,
    ):
        raise UgiMorphologyProgramError("exact attached-tree decoder violated its contract")
    return output


def decode_attached_offspring_with_exact_budget(
    logits: Any,
    *,
    junction_budget: int,
    attachment_count: int = 1,
) -> np.ndarray:
    """Return the highest-scoring exactly feasible attached forest deterministically.

    This is the max-product counterpart of the conditional sampler above.  Feasibility is carried
    in the dynamic-programming state, so the result is constructed under the requested size,
    attachment and junction coordinates rather than repaired after an unconstrained decode.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise UgiMorphologyProgramError("attached-tree decoding requires torch") from exc
    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise UgiMorphologyProgramError("offspring logits must be [nodes, child classes]")
    node_count, child_classes = logits.shape
    maximum_children = child_classes - 1
    if not attached_program_feasible(
        node_count,
        junction_budget,
        maximum_children,
        attachment_count,
    ):
        raise UgiMorphologyProgramError("declared attached-tree program is infeasible")
    scores = logits.to(torch.float64)
    suffix: list[dict[tuple[int, int], Any]] = [dict() for _ in range(node_count + 1)]
    suffix[node_count][(0, 0)] = scores.new_tensor(0.0)
    for position in range(node_count - 1, -1, -1):
        positions_including_current = node_count - position
        for pending in range(1, positions_including_current + 1):
            for budget in range(junction_budget + 1):
                candidates = []
                for children in range(maximum_children + 1):
                    state = _attached_choice_valid(
                        position=position,
                        pending=pending,
                        remaining_budget=budget,
                        children=children,
                        node_count=node_count,
                    )
                    if state is not None and state in suffix[position + 1]:
                        candidates.append(scores[position, children] + suffix[position + 1][state])
                if candidates:
                    suffix[position][(pending, budget)] = torch.stack(candidates).max()
    initial = (attachment_count, junction_budget)
    if initial not in suffix[0]:
        raise UgiMorphologyProgramError("exact attached-tree decoder found no completion")

    output = np.zeros(node_count, dtype=np.int64)
    pending, remaining_budget = initial
    for position in range(node_count):
        candidates: list[tuple[float, int, tuple[int, int]]] = []
        for children in range(maximum_children + 1):
            state = _attached_choice_valid(
                position=position,
                pending=pending,
                remaining_budget=remaining_budget,
                children=children,
                node_count=node_count,
            )
            if state is None or state not in suffix[position + 1]:
                continue
            value = float((scores[position, children] + suffix[position + 1][state]).item())
            # Lower child counts win exact ties.  This makes the contract deterministic across
            # devices without changing any non-tied neural preference.
            candidates.append((value, -children, state))
        if not candidates:
            raise UgiMorphologyProgramError("exact attached-tree traceback lost feasibility")
        _, negative_children, state = max(candidates)
        children = -negative_children
        output[position] = children
        pending, remaining_budget = state
    if not attached_tree_matches_program(
        output,
        node_count=node_count,
        junction_budget=junction_budget,
        attachment_count=attachment_count,
    ):
        raise UgiMorphologyProgramError("exact attached-tree decoder violated its contract")
    return output


def sample_attached_offspring_with_exact_budget(
    logits: Any,
    *,
    junction_budget: int,
    attachment_count: int = 1,
    generator: Any,
    maximum_adjacent_branch_run: int | None = None,
    maximum_rejection_attempts: int = 64,
) -> np.ndarray:
    """Sample the exact program conditional on decoded tree-branch spacing."""

    if maximum_adjacent_branch_run is not None and maximum_adjacent_branch_run < 0:
        raise UgiMorphologyProgramError("adjacent branch-run support cannot be negative")
    if maximum_rejection_attempts < 1:
        raise UgiMorphologyProgramError("exact program rejection count must be positive")
    for _ in range(maximum_rejection_attempts):
        candidate = _sample_attached_offspring_with_exact_budget(
            logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=generator,
            maximum_adjacent_branch_run=None,
        )
        if (
            maximum_adjacent_branch_run is None
            or maximum_adjacent_branch_graph_run(candidate, attachment_count=attachment_count)
            <= maximum_adjacent_branch_run
        ):
            return candidate
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise UgiMorphologyProgramError("attached-tree sampling requires torch") from exc
    neutral_logits = torch.zeros_like(logits)
    for _ in range(4 * maximum_rejection_attempts):
        candidate = _sample_attached_offspring_with_exact_budget(
            neutral_logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=generator,
            maximum_adjacent_branch_run=None,
        )
        if (
            maximum_adjacent_branch_graph_run(candidate, attachment_count=attachment_count)
            <= maximum_adjacent_branch_run
        ):
            return candidate
    raise UgiMorphologyProgramError("exact attached-tree decoder exhausted branch-spacing support")


def sample_attached_offspring_with_exact_budget_and_cycle_rank(
    logits: Any,
    *,
    junction_budget: int,
    cycle_rank: int,
    attachment_count: int = 1,
    generator: Any,
    allowed_ring_sizes: Sequence[int] = (5, 6),
    maximum_heavy_degree: int = 4,
    maximum_enumerated_nodes: int = 12,
    maximum_rejection_attempts: int = 64,
    maximum_adjacent_branch_run: int | None = None,
) -> np.ndarray:
    """Sample an exact tree conditional on realizability of its sparse cycles.

    Exact program-constrained rejection samples the neural distribution
    conditioned on closure feasibility.  Exhaustive enumeration remains a
    deterministic fallback for small exteriors; neither path repairs a sampled
    topology after acceptance.
    """

    if cycle_rank == 0:
        return sample_attached_offspring_with_exact_budget(
            logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=generator,
            maximum_adjacent_branch_run=maximum_adjacent_branch_run,
        )
    if cycle_rank < 0 or maximum_rejection_attempts < 1:
        raise UgiMorphologyProgramError("invalid cycle-aware sampling request")
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise UgiMorphologyProgramError("attached-tree sampling requires torch") from exc
    from forge.model.ugi_closure_placement import feasible_next_closures

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise UgiMorphologyProgramError("offspring logits must be [nodes, child classes]")
    node_count, child_classes = logits.shape
    maximum_children = child_classes - 1
    if not attached_program_feasible(
        node_count,
        junction_budget,
        maximum_children,
        attachment_count,
    ):
        raise UgiMorphologyProgramError("declared attached-tree program is infeasible")
    for _ in range(maximum_rejection_attempts):
        candidate = sample_attached_offspring_with_exact_budget(
            logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=generator,
            maximum_adjacent_branch_run=maximum_adjacent_branch_run,
        )
        feasible = feasible_next_closures(
            candidate,
            remaining_closures_including_next=cycle_rank,
            attachment_count=attachment_count,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
        )
        if feasible.edges:
            return candidate
    # A trained model can become sharply concentrated on a valid tree that is
    # not closure-realizable.  Rejection from those logits may then repeat the
    # same word even though the declared program has broad feasible support.
    # Fall back to the exact program distribution with neutral local scores;
    # this is constrained resampling, not post-hoc repair of an accepted tree.
    neutral_logits = torch.zeros_like(logits)
    for _ in range(4 * maximum_rejection_attempts):
        candidate = sample_attached_offspring_with_exact_budget(
            neutral_logits,
            junction_budget=junction_budget,
            attachment_count=attachment_count,
            generator=generator,
            maximum_adjacent_branch_run=maximum_adjacent_branch_run,
        )
        feasible = feasible_next_closures(
            candidate,
            remaining_closures_including_next=cycle_rank,
            attachment_count=attachment_count,
            allowed_ring_sizes=allowed_ring_sizes,
            maximum_heavy_degree=maximum_heavy_degree,
        )
        if feasible.edges:
            return candidate
    if node_count > maximum_enumerated_nodes:
        raise UgiMorphologyProgramError(
            "cyclic exterior exhausted exact conditional rejection support"
        )
    log_probabilities = logits.to(torch.float64).log_softmax(dim=-1)
    words: list[np.ndarray] = []
    weights: list[Any] = []
    partial = np.zeros(node_count, dtype=np.int64)

    def enumerate_words(position: int, pending: int, remaining_budget: int) -> None:
        if position == node_count:
            if pending != 0 or remaining_budget != 0:
                return
            if (
                maximum_adjacent_branch_run is not None
                and maximum_adjacent_branch_graph_run(partial, attachment_count=attachment_count)
                > maximum_adjacent_branch_run
            ):
                return
            candidates = feasible_next_closures(
                partial,
                remaining_closures_including_next=cycle_rank,
                attachment_count=attachment_count,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
            )
            if candidates.edges:
                word = partial.copy()
                words.append(word)
                indices = torch.as_tensor(word, dtype=torch.long, device=logits.device)
                weights.append(log_probabilities[torch.arange(node_count), indices].sum())
            return
        for children in range(maximum_children + 1):
            state = _attached_choice_valid(
                position=position,
                pending=pending,
                remaining_budget=remaining_budget,
                children=children,
                node_count=node_count,
            )
            if state is None:
                continue
            partial[position] = children
            enumerate_words(position + 1, *state)

    enumerate_words(0, attachment_count, junction_budget)
    if not words:
        raise UgiMorphologyProgramError(
            "declared cyclic program has no closure-feasible offspring word"
        )
    selected = int(
        torch.multinomial(
            torch.stack(weights).softmax(dim=0),
            1,
            generator=generator,
        )
    )
    output = words[selected]
    if not attached_tree_matches_program(
        output,
        node_count=node_count,
        junction_budget=junction_budget,
        attachment_count=attachment_count,
    ):
        raise UgiMorphologyProgramError("cycle-aware decoder violated its tree contract")
    return output
