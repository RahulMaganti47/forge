"""Lightweight reaction-specialist deltas for the shared program Transformer.

The shared Ugi/BL/LX checkpoint remains the authenticated base model.  A specialist checkpoint
contains only zero-initialized residual adapters and an exact exposure ledger; it never duplicates
or silently modifies the frozen shared parameters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


class ReactionSpecializationError(ValueError):
    """A specialist model or exposure schedule violates its frozen contract."""


SPECIALIST_PARAMETER_TOKEN = ".specialist_adapter."
TOPOLOGY_PARAMETER_PREFIX = "offspring_output."


@dataclass(frozen=True)
class ExactExposureSchedule:
    """One exact weighted-draw target expressed as bounded optimizer microbatches."""

    existing_examples: int
    target_examples: int
    effective_batch_size: int
    micro_batch_size: int
    optimizer_steps: tuple[tuple[int, ...], ...]

    @property
    def additional_examples(self) -> int:
        return self.target_examples - self.existing_examples

    @property
    def final_microbatches(self) -> tuple[int, ...]:
        return self.optimizer_steps[-1]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "existing_examples": self.existing_examples,
            "target_examples": self.target_examples,
            "additional_examples": self.additional_examples,
            "effective_batch_size": self.effective_batch_size,
            "micro_batch_size": self.micro_batch_size,
            "optimizer_steps": len(self.optimizer_steps),
            "full_optimizer_steps": sum(
                sum(step) == self.effective_batch_size for step in self.optimizer_steps
            ),
            "final_microbatches": list(self.final_microbatches),
        }


def exact_exposure_schedule(
    *,
    existing_examples: int,
    target_examples: int,
    effective_batch_size: int,
    micro_batch_size: int,
) -> ExactExposureSchedule:
    """Build an exact, no-padding draw schedule, including a potentially short final update."""

    values = (existing_examples, target_examples, effective_batch_size, micro_batch_size)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ReactionSpecializationError("exposure schedule fields must be integers")
    if (
        existing_examples < 0
        or target_examples <= existing_examples
        or micro_batch_size < 1
        or effective_batch_size < micro_batch_size
        or effective_batch_size % micro_batch_size
    ):
        raise ReactionSpecializationError("exposure schedule geometry is invalid")
    additional = target_examples - existing_examples
    full_steps, remainder = divmod(additional, effective_batch_size)
    full_microbatches = tuple(
        micro_batch_size for _ in range(effective_batch_size // micro_batch_size)
    )
    steps = [full_microbatches for _ in range(full_steps)]
    if remainder:
        final: list[int] = []
        while remainder:
            value = min(micro_batch_size, remainder)
            final.append(value)
            remainder -= value
        steps.append(tuple(final))
    if not steps or sum(sum(step) for step in steps) != additional:
        raise ReactionSpecializationError("exact exposure schedule failed conservation")
    return ExactExposureSchedule(
        existing_examples=existing_examples,
        target_examples=target_examples,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        optimizer_steps=tuple(steps),
    )


def specialist_parameter_names(
    model: Any,
    *,
    include_topology_head: bool = False,
) -> tuple[str, ...]:
    """Return the complete and only trainable state of a reaction specialist."""

    names = tuple(
        name
        for name, _ in model.named_parameters()
        if SPECIALIST_PARAMETER_TOKEN in name
        or (include_topology_head and name.startswith(TOPOLOGY_PARAMETER_PREFIX))
    )
    if not names:
        raise ReactionSpecializationError("model has no reaction-specialist adapters")
    if include_topology_head and not any(
        name.startswith(TOPOLOGY_PARAMETER_PREFIX) for name in names
    ):
        raise ReactionSpecializationError("model has no child-count topology head")
    return names


def initialize_specialist_from_shared_state(
    model: Any,
    shared_state: Mapping[str, Any],
    *,
    include_topology_head: bool = False,
) -> tuple[str, ...]:
    """Load a shared checkpoint and require every new key to be in the declared delta."""

    if torch is None:
        raise ReactionSpecializationError("specialist initialization requires torch")
    incompatible = model.load_state_dict(shared_state, strict=False)
    missing = tuple(sorted(incompatible.missing_keys))
    unexpected = tuple(sorted(incompatible.unexpected_keys))
    selected = set(
        specialist_parameter_names(model, include_topology_head=include_topology_head)
    )
    # Parameters and persistent state currently coincide for both declared modules.  Compare the
    # complete state-key set explicitly so a later buffer cannot enter the delta silently.
    expected = tuple(sorted(name for name in model.state_dict() if name in selected))
    if unexpected or missing != expected:
        raise ReactionSpecializationError(
            "shared checkpoint differs from the specialist model outside the declared adapters"
        )
    return missing


def freeze_shared_parameters(
    model: Any,
    *,
    include_topology_head: bool = False,
) -> tuple[str, ...]:
    """Freeze the authenticated backbone and enable only the declared specialist delta."""

    names = specialist_parameter_names(model, include_topology_head=include_topology_head)
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    observed = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if observed != names:
        raise ReactionSpecializationError("specialist trainable-parameter set changed")
    return names


def specialist_state_dict(
    model: Any,
    *,
    include_topology_head: bool = False,
) -> dict[str, Any]:
    """Copy only the lightweight trained delta for authenticated persistence."""

    expected = set(
        specialist_parameter_names(model, include_topology_head=include_topology_head)
    )
    state = model.state_dict()
    observed = {
        name
        for name in state
        if SPECIALIST_PARAMETER_TOKEN in name
        or (include_topology_head and name.startswith(TOPOLOGY_PARAMETER_PREFIX))
    }
    if observed != expected:
        raise ReactionSpecializationError("specialist parameter and state keys disagree")
    return {name: state[name].detach().cpu().clone() for name in sorted(observed)}


def apply_specialist_state(
    model: Any,
    delta: Mapping[str, Any],
    *,
    include_topology_head: bool = False,
) -> None:
    """Overlay one specialist delta while rejecting missing, extra, or malformed tensors."""

    expected = set(
        specialist_parameter_names(model, include_topology_head=include_topology_head)
    )
    if set(delta) != expected:
        raise ReactionSpecializationError("specialist delta keys changed")
    state = model.state_dict()
    for name in sorted(expected):
        value = delta[name]
        if not hasattr(value, "shape") or tuple(value.shape) != tuple(state[name].shape):
            raise ReactionSpecializationError(f"specialist delta tensor changed shape: {name}")
        state[name].copy_(value.to(device=state[name].device, dtype=state[name].dtype))


def specialist_parameter_report(
    model: Any,
    *,
    include_topology_head: bool = False,
) -> dict[str, int | float]:
    """Report the exact delta size relative to the shared-plus-specialist model."""

    selected = set(
        specialist_parameter_names(model, include_topology_head=include_topology_head)
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    specialist = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name in selected
    )
    if total < 1 or specialist < 1 or specialist >= total:
        raise ReactionSpecializationError("specialist parameter accounting is invalid")
    return {
        "total_parameters": total,
        "shared_frozen_parameters": total - specialist,
        "specialist_trainable_parameters": specialist,
        "specialist_fraction": specialist / total,
    }


def optimizer_parameters(model: Any) -> Sequence[Any]:
    """Return only declared specialist parameters for optimizer construction."""

    values = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not values:
        raise ReactionSpecializationError("specialist optimizer has no trainable parameters")
    return values


__all__ = [
    "ExactExposureSchedule",
    "ReactionSpecializationError",
    "SPECIALIST_PARAMETER_TOKEN",
    "TOPOLOGY_PARAMETER_PREFIX",
    "apply_specialist_state",
    "exact_exposure_schedule",
    "freeze_shared_parameters",
    "initialize_specialist_from_shared_state",
    "optimizer_parameters",
    "specialist_parameter_names",
    "specialist_parameter_report",
    "specialist_state_dict",
]
