"""Deterministic equal-family gradient projection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forge.model.networks.transformer import ReactionProgramTransformerError

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


def balanced_pcgrad_backward(
    losses: Mapping[int, Any],
    model: Any,
    *,
    scale: float = 1.0,
    materialize_diagnostics: bool = True,
    backend: str = "sequential",
) -> dict[str, Any]:
    """Give families equal loss mass and deterministically project conflicting gradients.

    Families are balanced by stratified sampling plus an equal-weight gradient mean.  Deliberately
    do not normalize each gradient to a common norm: that operation amplifies numerical noise from
    an already converged family and can destabilize the remaining objectives.

    Submitted configs use ``backend="sequential"``. The optional ``batched_vjp`` backend has
    an unresolved Linux bitwise-equivalence failure.
    """

    if len(losses) < 2 or scale <= 0.0:
        raise ReactionProgramTransformerError(
            "PCGrad requires at least two programs and positive scale"
        )
    if backend not in {"sequential", "batched_vjp"}:
        raise ReactionProgramTransformerError(f"unsupported PCGrad backend: {backend!r}")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ordered = sorted(losses)
    if backend == "batched_vjp":
        stacked_losses = torch.stack([losses[program] for program in ordered])
        batched = torch.autograd.grad(
            stacked_losses,
            parameters,
            grad_outputs=torch.eye(
                len(ordered), dtype=stacked_losses.dtype, device=stacked_losses.device
            ),
            is_grads_batched=True,
            allow_unused=True,
        )
        # The production Transformer is a shared parameter graph: every parameter participating in
        # one family participates in every family.  A tensor absent from the batched VJP is therefore
        # absent from all rows.  Keeping this backend explicit prevents it from being silently used
        # by future family-exclusive architectures.
        availability = [[gradient is not None for gradient in batched] for _ in ordered]
        flat_gradients = torch.cat(
            [
                (
                    gradient.reshape(len(ordered), -1)
                    if gradient is not None
                    else parameter.new_zeros((len(ordered), parameter.numel()))
                )
                for gradient, parameter in zip(batched, parameters, strict=True)
            ],
            dim=1,
        )
        del batched
    else:
        raw = [
            torch.autograd.grad(
                losses[program],
                parameters,
                retain_graph=index + 1 < len(ordered),
                allow_unused=True,
            )
            for index, program in enumerate(ordered)
        ]
        availability = [[gradient is not None for gradient in gradients] for gradients in raw]
        flat_gradients = torch.stack(
            [
                torch.cat(
                    [
                        (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(
                            -1
                        )
                        for gradient, parameter in zip(gradients, parameters, strict=True)
                    ]
                )
                for gradients in raw
            ]
        )
        del raw
    norms = torch.linalg.vector_norm(flat_gradients, dim=1)
    conflicts = torch.zeros((), dtype=torch.int64, device=flat_gradients.device)
    pairwise_dots = flat_gradients @ flat_gradients.transpose(0, 1)
    off_diagonal = ~torch.eye(len(ordered), dtype=torch.bool, device=flat_gradients.device)
    if not bool(((pairwise_dots < 0.0) & off_diagonal).any()):
        combined = flat_gradients.mean(dim=0)
    else:
        scalar_availability = torch.as_tensor(
            availability, dtype=torch.bool, device=flat_gradients.device
        ).repeat_interleave(
            torch.as_tensor(
                [parameter.numel() for parameter in parameters], device=flat_gradients.device
            ),
            dim=1,
        )
        denominators = pairwise_dots.diagonal().clamp(min=1e-12)
        combined = torch.zeros_like(flat_gradients[0])
        for left_index, gradients in enumerate(flat_gradients):
            current = gradients.clone()
            for right_index, reference in enumerate(flat_gradients):
                if left_index == right_index:
                    continue
                dot = torch.dot(current, reference)
                negative = dot < 0.0
                conflicts.add_(negative)
                coefficient = torch.where(
                    negative,
                    dot / denominators[right_index],
                    dot.new_zeros(()),
                )
                current = current - coefficient * reference * (
                    scalar_availability[left_index] & scalar_availability[right_index]
                )
            combined.add_(current, alpha=1.0 / len(ordered))
    combined.mul_(scale).detach_()
    additions: list[Any] = []
    existing: list[Any] = []
    offset = 0
    for parameter_index, parameter in enumerate(parameters):
        elements = parameter.numel()
        update = combined[offset : offset + elements].view_as(parameter)
        offset += elements
        if not any(row[parameter_index] for row in availability):
            continue
        if parameter.grad is None:
            parameter.grad = update
        else:
            existing.append(parameter.grad)
            additions.append(update)
    if existing:
        torch._foreach_add_(existing, additions)
    if materialize_diagnostics:
        values = torch.cat((norms, conflicts.to(norms.dtype)[None])).detach().cpu().tolist()
        raw_gradient_norms: Any = values[:-1]
        projected_conflicts: Any = int(values[-1])
    else:
        raw_gradient_norms = norms.detach()
        projected_conflicts = conflicts.detach()
    return {
        "program_states": ordered,
        "raw_gradient_norms": raw_gradient_norms,
        "family_weighting": "equal_loss_mass_without_norm_amplification",
        "projected_conflicts": projected_conflicts,
        "backend": backend,
    }


__all__ = ["balanced_pcgrad_backward"]
