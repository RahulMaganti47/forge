"""Reaction-program semantic objectives and per-program loss balancing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forge.model.networks.reaction_flow import (
    synthesis_program_chemistry_loss,
    synthesis_program_flow_loss,
)
from forge.model.networks.transformer import ReactionProgramTransformerError

try:
    import torch
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


def _balanced_state_cross_entropy(logits: Any, targets: Any, mask: Any) -> Any:
    """Average semantic classification loss across present states, not atom frequency."""

    selected_targets = targets[mask]
    point_losses = functional.cross_entropy(logits[mask], selected_targets, reduction="none")
    classes = logits.shape[-1]
    state_sums = logits.new_zeros(classes).scatter_add(0, selected_targets, point_losses)
    state_counts = torch.bincount(selected_targets, minlength=classes)
    present = state_counts > 0
    state_means = state_sums / state_counts.clamp(min=1)
    return (state_means * present).sum() / present.sum().clamp(min=1)


def _repeat_component_consistency(
    predictions: Mapping[str, Any], clean: Mapping[str, Any]
) -> tuple[Any, Any]:
    """Align matched positions across repeated components without dense state broadcasts.

    The equivalence relation itself is only ``[batch, nodes, nodes]``.  Materializing atom- and
    bond-state differences for every possible pair adds a final class dimension even though almost
    every pair is masked out.  Gather the admitted pairs first so memory and backward work scale
    with the number of supervised repeat pairs rather than ``nodes**2 * classes``.
    """

    repeat_groups = clean.get("repeat_group_states")
    component_positions = clean.get("component_position_states")
    component_instances = clean.get("component_instance_states")
    if repeat_groups is None or component_positions is None or component_instances is None:
        zero = predictions["nodes"].new_zeros(())
        return zero, zero.to(dtype=torch.int64)
    active = (repeat_groups > 0) & clean["node_mask"] & (clean["core_position_states"] == 1)
    pair_mask = (
        active[:, :, None]
        & active[:, None, :]
        & (repeat_groups[:, :, None] == repeat_groups[:, None, :])
        & (component_positions[:, :, None] == component_positions[:, None, :])
        & (component_instances[:, :, None] != component_instances[:, None, :])
    )
    nodes = repeat_groups.shape[1]
    upper = torch.triu(
        torch.ones((nodes, nodes), dtype=torch.bool, device=repeat_groups.device),
        diagonal=1,
    )
    pair_mask &= upper[None]
    node_probabilities = torch.softmax(predictions["nodes"], dim=-1)
    pair_count = pair_mask.sum()
    pair_batch, pair_left, pair_right = pair_mask.nonzero(as_tuple=True)
    node_difference = (
        node_probabilities[pair_batch, pair_left] - node_probabilities[pair_batch, pair_right]
    )
    node_loss = node_difference.square().mean(dim=-1).sum() / pair_count.clamp(min=1)

    bond_pair_mask = (
        clean["child_mask"][pair_batch, pair_left] & clean["child_mask"][pair_batch, pair_right]
    )
    bond_probabilities = torch.softmax(predictions["parent_bonds"], dim=-1)
    bond_difference = (
        bond_probabilities[pair_batch[bond_pair_mask], pair_left[bond_pair_mask]]
        - bond_probabilities[pair_batch[bond_pair_mask], pair_right[bond_pair_mask]]
    )
    bond_count = bond_pair_mask.sum()
    bond_loss = bond_difference.square().mean(dim=-1).sum() / bond_count.clamp(min=1)
    return node_loss + bond_loss, pair_count


def synthesis_program_offspring_targets(
    clean: Mapping[str, Any],
    *,
    maximum_children: int,
) -> tuple[Any, Any]:
    """Derive exterior child counts from sparse parents without fragment identities.

    The target is node-local topology in the existing serialization.  Only edges between exterior
    atoms in the same anonymous origin component contribute; adapter-owned core attachments and
    reaction-core edges remain outside the learned offspring channel.
    """

    if maximum_children < 1:
        raise ReactionProgramTransformerError("offspring supervision requires positive support")
    required = {
        "parents",
        "child_mask",
        "node_mask",
        "core_position_states",
        "component_instance_states",
    }
    if not required.issubset(clean):
        raise ReactionProgramTransformerError("offspring supervision is missing graph coordinates")
    parents = clean["parents"]
    batch, nodes = parents.shape
    batch_indices = torch.arange(batch, device=parents.device)[:, None].expand(batch, nodes)
    parent_core = clean["core_position_states"].gather(1, parents)
    parent_components = clean["component_instance_states"].gather(1, parents)
    exterior = clean["core_position_states"] == 1
    internal_children = (
        clean["child_mask"]
        & exterior
        & (parent_core == 1)
        & (clean["component_instance_states"] == parent_components)
    )
    targets = torch.zeros_like(parents)
    targets.index_put_(
        (batch_indices[internal_children], parents[internal_children]),
        torch.ones_like(parents[internal_children]),
        accumulate=True,
    )
    mask = clean["node_mask"] & exterior & (clean["component_instance_states"] > 0)
    if torch.any(targets[mask] > maximum_children):
        raise ReactionProgramTransformerError(
            "observed offspring target exceeds declared child-count support"
        )
    return targets, mask


def _offspring_program_consistency(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Match expected role-local junction budgets to the supplied coarse program."""

    logits = predictions.get("offspring")
    morphology = clean.get("role_morphology_states")
    if logits is None or morphology is None:
        raise ReactionProgramTransformerError(
            "program topology consistency requires offspring logits and morphology states"
        )
    maximum_children = int(logits.shape[-1]) - 1
    _, exterior_mask = synthesis_program_offspring_targets(
        clean,
        maximum_children=maximum_children,
    )
    child_states = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    junction_contributions = torch.clamp(child_states - 1, min=0)
    expected_by_node = torch.einsum(
        "bnc,c->bn", torch.softmax(logits, dim=-1), junction_contributions
    )
    losses = []
    comparisons = logits.new_zeros((), dtype=torch.int64)
    maximum_role = int(clean["role_states"].max().item())
    for role_state in range(1, maximum_role + 1):
        role_mask = exterior_mask & (clean["role_states"] == role_state)
        active = role_mask.any(dim=1)
        conditioned = (morphology[:, :, 1] > 0) & (clean["role_states"] == role_state)
        active &= conditioned.any(dim=1)
        if not bool(active.any()):
            continue
        predicted = (expected_by_node * role_mask).sum(dim=1)
        # Morphology coordinates store zero as unconditioned and observed values at value + 1.
        target = morphology[:, :, 1].masked_fill(~conditioned, 0).max(dim=1).values - 1
        node_counts = role_mask.sum(dim=1).clamp(min=1).to(logits.dtype)
        losses.append(
            functional.smooth_l1_loss(
                predicted[active] / node_counts[active],
                target[active].to(logits.dtype) / node_counts[active],
            )
        )
        comparisons = comparisons + active.sum()
    if not losses:
        return logits.new_zeros(()), comparisons
    return torch.stack(losses).mean(), comparisons


def reaction_program_transformer_loss(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    role_weight: float,
    core_weight: float,
    repeat_consistency_weight: float = 0.0,
    offspring_weight: float = 0.0,
    junction_consistency_weight: float = 0.0,
    chemistry_loss_balancing: str = "pooled",
    topology_conditioned_predictions: Mapping[str, Any] | None = None,
    topology_conditioned_chemistry_weight: float = 0.0,
    materialize_metrics: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Combine graph flow with state-balanced precursor-role and reaction-core consistency."""

    if any(
        value < 0.0
        for value in (
            role_weight,
            core_weight,
            repeat_consistency_weight,
            offspring_weight,
            junction_consistency_weight,
            topology_conditioned_chemistry_weight,
        )
    ):
        raise ReactionProgramTransformerError("semantic loss weights must be nonnegative")
    if chemistry_loss_balancing not in {"pooled", "equal_present_role_mass"}:
        raise ReactionProgramTransformerError(
            f"unsupported chemistry loss balancing: {chemistry_loss_balancing!r}"
        )
    base, metrics = synthesis_program_flow_loss(
        predictions,
        clean,
        balance_chemistry_by_role=chemistry_loss_balancing == "equal_present_role_mass",
        materialize_metrics=materialize_metrics,
    )
    role = _balanced_state_cross_entropy(
        predictions["role_states"], clean["role_states"], clean["node_mask"]
    )
    core = _balanced_state_cross_entropy(
        predictions["core_position_states"],
        clean["core_position_states"],
        clean["node_mask"],
    )
    repeat_consistency, repeat_pairs = _repeat_component_consistency(predictions, clean)
    offspring = base.new_zeros(())
    junction_consistency = base.new_zeros(())
    junction_comparisons = base.new_zeros((), dtype=torch.int64)
    topology_conditioned_chemistry = base.new_zeros(())
    if topology_conditioned_chemistry_weight > 0.0:
        if topology_conditioned_predictions is None:
            raise ReactionProgramTransformerError(
                "topology-conditioned chemistry was enabled without its second prediction pass"
            )
        topology_conditioned_chemistry, _ = synthesis_program_chemistry_loss(
            topology_conditioned_predictions,
            clean,
            balance_by_role=chemistry_loss_balancing == "equal_present_role_mass",
        )
    if offspring_weight > 0.0 or junction_consistency_weight > 0.0:
        offspring_logits = predictions.get("offspring")
        if offspring_logits is None:
            raise ReactionProgramTransformerError(
                "topology objective was enabled without an offspring prediction head"
            )
        offspring_targets, offspring_mask = synthesis_program_offspring_targets(
            clean,
            maximum_children=int(offspring_logits.shape[-1]) - 1,
        )
        offspring = _balanced_state_cross_entropy(
            offspring_logits,
            offspring_targets,
            offspring_mask,
        )
        junction_consistency, junction_comparisons = _offspring_program_consistency(
            predictions, clean
        )
    total = (
        base
        + role_weight * role
        + core_weight * core
        + repeat_consistency_weight * repeat_consistency
        + offspring_weight * offspring
        + junction_consistency_weight * junction_consistency
        + topology_conditioned_chemistry_weight * topology_conditioned_chemistry
    )
    semantic_metrics = {
        "role_consistency_ce": role.detach(),
        "core_consistency_ce": core.detach(),
        "repeat_consistency_mse": repeat_consistency.detach(),
        "repeat_consistency_pairs": repeat_pairs.detach(),
        "offspring_ce": offspring.detach(),
        "junction_budget_consistency": junction_consistency.detach(),
        "junction_budget_comparisons": junction_comparisons.detach(),
        "topology_conditioned_chemistry_ce": topology_conditioned_chemistry.detach(),
        "semantic_total": total.detach(),
    }
    if materialize_metrics:
        semantic_metrics = {key: float(value) for key, value in semantic_metrics.items()}
    return total, {**metrics, **semantic_metrics}


def _slice_batch(values: Mapping[str, Any], indices: Any) -> dict[str, Any]:
    batch = int(indices.shape[0])
    output: dict[str, Any] = {}
    for key, value in values.items():
        if hasattr(value, "shape") and value.ndim > 0 and value.shape[0] == batch:
            output[key] = value[indices]
        else:
            output[key] = value
    return output


def per_program_transformer_losses(
    predictions: Mapping[str, Any],
    clean: Mapping[str, Any],
    *,
    role_weight: float,
    core_weight: float,
    repeat_consistency_weight: float = 0.0,
    offspring_weight: float = 0.0,
    junction_consistency_weight: float = 0.0,
    chemistry_loss_balancing: str = "pooled",
    topology_conditioned_predictions: Mapping[str, Any] | None = None,
    topology_conditioned_chemistry_weight: float = 0.0,
    program_states: tuple[int, ...] | None = None,
    materialize_metrics: bool = True,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Return one equally weighted objective per source reaction program."""

    source_programs = clean.get("source_program_states")
    if source_programs is None:
        raise ReactionProgramTransformerError(
            "source program identities are required for balancing"
        )
    losses: dict[int, Any] = {}
    metrics: dict[str, Any] = {}
    observed_programs = (
        tuple(int(value) for value in torch.unique(source_programs, sorted=True).tolist())
        if program_states is None
        else program_states
    )
    for program_state in observed_programs:
        indices = source_programs == int(program_state)
        loss, values = reaction_program_transformer_loss(
            _slice_batch(predictions, indices),
            _slice_batch(clean, indices),
            role_weight=role_weight,
            core_weight=core_weight,
            repeat_consistency_weight=repeat_consistency_weight,
            offspring_weight=offspring_weight,
            junction_consistency_weight=junction_consistency_weight,
            chemistry_loss_balancing=chemistry_loss_balancing,
            topology_conditioned_predictions=(
                _slice_batch(topology_conditioned_predictions, indices)
                if topology_conditioned_predictions is not None
                else None
            ),
            topology_conditioned_chemistry_weight=topology_conditioned_chemistry_weight,
            materialize_metrics=materialize_metrics,
        )
        losses[int(program_state)] = loss
        for key, value in values.items():
            metrics[f"program_{int(program_state)}_{key}"] = value
    return losses, metrics


__all__ = [
    "per_program_transformer_losses",
    "reaction_program_transformer_loss",
    "synthesis_program_offspring_targets",
]
