"""Euler sampling with DeFoG's minimum R-star conditional rate.

The historical feasibility probe retains its own implementation. This module
provides the shared primitive and raises SamplingError for unsupported inputs.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

__all__ = ["SamplingError", "rstar_step", "sample_categorical"]


class SamplingError(RuntimeError):
    """A sampling step was asked for something its declared support cannot represent."""


def sample_categorical(probabilities: Any, generator: Any) -> Any:
    """Draw one category per row using the supplied random generator."""
    shape = probabilities.shape[:-1]
    sampled = torch.multinomial(
        probabilities.reshape(-1, probabilities.shape[-1]),
        1,
        generator=generator,
    )
    return sampled.reshape(shape)


def rstar_step(
    current: Any,
    clean_probabilities: Any,
    marginal: Any,
    t: float,
    dt: float,
    valid_mask: Any,
    generator: Any,
) -> Any:
    """One Euler step using DeFoG's minimum R-star conditional rate.

    Advances `current` one step of size `dt` at time `t`, given the model's predicted distribution
    over clean states. Only positions selected by `valid_mask` move; everything else is returned
    untouched, which is what lets one call advance a ragged batch of molecules of different sizes.

    `marginal` may be either a single distribution shared across positions, or one per position
    matching the shape of `current`. Anything else raises rather than broadcasting silently, since a
    mis-shaped marginal would produce plausible samples from the wrong prior.
    """
    sampled_clean = sample_categorical(clean_probabilities, generator)
    flat_current = current[valid_mask]
    flat_clean = sampled_clean[valid_mask]
    if flat_current.numel() == 0:
        return current
    if marginal.ndim == 1:
        flat_marginal = marginal[None, :].repeat(flat_current.shape[0], 1)
    elif marginal.ndim == current.ndim + 1 and marginal.shape[:-1] == current.shape:
        flat_marginal = marginal[valid_mask]
    else:
        raise SamplingError("R-star marginal has incompatible support")
    classes = flat_marginal.shape[1]
    derivative = -flat_marginal
    derivative.scatter_add_(
        1,
        flat_clean[:, None],
        torch.ones(
            (flat_clean.shape[0], 1),
            dtype=derivative.dtype,
            device=derivative.device,
        ),
    )
    derivative_current = derivative.gather(1, flat_current[:, None])
    interpolant_support = ((1.0 - t) * flat_marginal) + t * functional.one_hot(
        flat_clean, num_classes=classes
    ) > 0
    numerator = torch.relu(derivative - derivative_current) * interpolant_support
    p_current = (1.0 - t) * flat_marginal.gather(
        1,
        flat_current[:, None],
    ).squeeze(1)
    p_current += t * (flat_current == flat_clean).float()
    nonzero_states = interpolant_support.sum(dim=1)
    rates = numerator / (nonzero_states[:, None] * p_current[:, None].clamp(min=1e-8))
    rates.scatter_(1, flat_current[:, None], 0.0)
    off_diagonal = rates * dt
    total = off_diagonal.sum(dim=1, keepdim=True)
    # Cap the total off-diagonal mass just below 1 so the diagonal stays a valid probability.
    scale = torch.where(total > 0.999, 0.999 / total, torch.ones_like(total))
    off_diagonal = off_diagonal * scale
    probabilities = off_diagonal
    probabilities.scatter_(
        1,
        flat_current[:, None],
        1.0 - off_diagonal.sum(dim=1, keepdim=True),
    )
    sampled = sample_categorical(probabilities, generator)
    output = current.clone()
    output[valid_mask] = sampled
    return output
