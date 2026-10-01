"""The discrete-flow sampling primitive: one Euler step of the generative sampler.

`rstar_step` is one Euler step of the generative model's sampler, using DeFoG's minimum R-star
conditional rate. It is the core primitive of every flow in this package: ten modules call it, for
node states, parent bonds, closure bonds, decoration anchors, regions and offspring channels.

It did not have a home. It lived as `_rstar_step` inside `design/flow/defog_feasibility.py` -- a
completed M0-06 *feasibility probe*, one bounded experiment asking whether dense edge flow was
viable at 96 atoms -- and every production sampler reached into that finished experiment's privates
to get it. The probe also carries a training loop, ECE/Jensen-Shannon/Wasserstein metrics, a
peak-RSS meter and a policy decision, so importing the sampler dragged all of it in.

**The frozen original still exists, and that is deliberate.** `defog_feasibility.py` is hash-pinned
by the artifact its probe produced, so its bytes may not change; it keeps its own copy of this code
and continues to use it. This module is the canonical one for everything else. The duplication is
the price of not invalidating a frozen result, and it resolves whenever that artifact is regenerated.

Six of the ten callers are inside the blinded-execution dependency manifest, which proves which
modules loaded during a sealed holdout by exact set equality. Repointing them would add
`forge.flow.rstar` to that set and break the proof, so they keep importing the frozen copy until the
sealed protocol is revisited.

Behaviour is identical to the original, and `tests/test_generate_sampling.py` asserts that directly
against the frozen module rather than trusting the transcription. The one difference is the error
type: the original raises `FeasibilityError`, which named the probe rather than the operation. This
raises `SamplingError`. Both derive from `RuntimeError`, and no caller catches either.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

__all__ = ["SamplingError", "rstar_step", "sample_categorical"]


class SamplingError(RuntimeError):
    """A sampling step was asked for something its declared support cannot represent."""


def sample_categorical(probabilities: Any, generator: Any) -> Any:
    """Draw one category per row from a batch of distributions.

    `generator` is threaded through rather than left to global RNG state so a run is reproducible
    from its recorded seed -- the artifact contract requires that the numbers can be regenerated.
    """
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
