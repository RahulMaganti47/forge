"""Reaction-program semantic conditioning without component identities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from forge.assembly import ReactionProgramSpec

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class ReactionProgramConditioningError(ValueError):
    """Program-conditioning states lie outside the declared vocabulary."""


@dataclass(frozen=True)
class ReactionProgramVocabulary:
    """Stable categorical coordinates shared by corpus, trainer, and sampler."""

    program_states: tuple[str, ...]
    role_states: tuple[str, ...]
    maximum_steps: int
    core_position_states: tuple[str, ...] = ("unconditioned", "exterior")

    def __post_init__(self) -> None:
        if (
            not self.program_states
            or self.program_states[0] != "unconditioned"
            or len(set(self.program_states)) != len(self.program_states)
            or not self.role_states
            or self.role_states[0] != "unassigned"
            or len(set(self.role_states)) != len(self.role_states)
            or self.maximum_steps < 1
            or len(self.core_position_states) < 2
            or self.core_position_states[:2] != ("unconditioned", "exterior")
            or len(set(self.core_position_states)) != len(self.core_position_states)
        ):
            raise ReactionProgramConditioningError("reaction-program vocabulary is invalid")

    @classmethod
    def from_specs(
        cls,
        specs: tuple[ReactionProgramSpec, ...],
        *,
        core_positions: Sequence[str] = (),
    ) -> ReactionProgramVocabulary:
        if not specs:
            raise ReactionProgramConditioningError("at least one reaction program is required")
        program_ids = tuple(sorted({spec.program_id for spec in specs}))
        if len(program_ids) != len(specs):
            raise ReactionProgramConditioningError("reaction program identifiers must be unique")
        roles = tuple(
            sorted({role for spec in specs for role in (spec.accumulator_role, spec.repeat_role)})
        )
        return cls.from_semantics(
            program_ids=program_ids,
            roles=roles,
            core_positions=core_positions,
            maximum_steps=max(spec.maximum_steps for spec in specs),
        )

    @classmethod
    def from_semantics(
        cls,
        *,
        program_ids: Sequence[str],
        roles: Sequence[str],
        core_positions: Sequence[str] = (),
        maximum_steps: int,
    ) -> ReactionProgramVocabulary:
        """Build one vocabulary from arbitrary evidenced program semantics.

        ``from_specs`` remains the convenience constructor for repeated two-role programs.  This
        constructor is the shared seam for Ugi and other exact adapters whose precursor roles do
        not fit the accumulator/repeat shape.  Values are categorical coordinates only; component
        identifiers and molecular fragments are deliberately not accepted.
        """

        programs = tuple(sorted(set(program_ids)))
        semantic_roles = tuple(sorted(set(roles)))
        positions = tuple(sorted(set(core_positions)))
        if not programs or len(programs) != len(tuple(program_ids)):
            raise ReactionProgramConditioningError(
                "reaction program identifiers must be nonempty and unique"
            )
        if not semantic_roles or any(not role for role in semantic_roles):
            raise ReactionProgramConditioningError("reaction roles must be nonempty")
        if "unconditioned" in programs or "unassigned" in semantic_roles:
            raise ReactionProgramConditioningError(
                "reaction programs and roles use reserved null labels"
            )
        if {"unconditioned", "exterior"}.intersection(positions):
            raise ReactionProgramConditioningError(
                "reaction core positions use reserved null/exterior labels"
            )
        return cls(
            program_states=("unconditioned", *programs),
            role_states=("unassigned", *semantic_roles),
            maximum_steps=maximum_steps,
            core_position_states=("unconditioned", "exterior", *positions),
        )

    @property
    def program_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.program_states)}

    @property
    def role_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.role_states)}

    @property
    def core_position_to_index(self) -> dict[str, int]:
        return {value: index for index, value in enumerate(self.core_position_states)}


if nn is not None:

    class ReactionProgramConditioning(nn.Module):
        """Embed program identity, precursor role and complete program depth.

        These are semantic coordinates only.  Exact component graphs, component identifiers,
        source product labels and biological measurements are intentionally absent.
        """

        def __init__(self, *, vocabulary: ReactionProgramVocabulary, hidden_dim: int) -> None:
            super().__init__()
            if hidden_dim < 1:
                raise ReactionProgramConditioningError("hidden_dim must be positive")
            self.vocabulary = vocabulary
            self.program_embedding = nn.Embedding(len(vocabulary.program_states), hidden_dim)
            self.role_embedding = nn.Embedding(len(vocabulary.role_states), hidden_dim)
            self.core_position_embedding = nn.Embedding(
                len(vocabulary.core_position_states), hidden_dim
            )
            # Zero is unconditioned.  Depth k means that the complete program has k repeated steps.
            self.depth_embedding = nn.Embedding(vocabulary.maximum_steps + 1, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            *,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
        ) -> Any:
            if program_states.ndim != 1 or role_states.ndim != 2 or core_position_states.ndim != 2:
                raise ReactionProgramConditioningError(
                    "program states must be [batch] and node semantics [batch, nodes]"
                )
            if (
                program_depths.ndim != 1
                or adapter_mask.shape != role_states.shape
                or core_position_states.shape != role_states.shape
                or program_states.shape[0] != role_states.shape[0]
                or program_depths.shape[0] != role_states.shape[0]
            ):
                raise ReactionProgramConditioningError("program-conditioning shapes disagree")
            if adapter_mask.dtype != torch.bool:
                raise ReactionProgramConditioningError("adapter mask must be Boolean")
            for values, classes, label in (
                (program_states, len(self.vocabulary.program_states), "program"),
                (role_states, len(self.vocabulary.role_states), "role"),
                (
                    core_position_states,
                    len(self.vocabulary.core_position_states),
                    "core position",
                ),
                (program_depths, self.vocabulary.maximum_steps + 1, "program depth"),
            ):
                if torch.any(values < 0) or torch.any(values >= classes):
                    raise ReactionProgramConditioningError(
                        f"{label} state lies outside the declared vocabulary"
                    )
            hidden = (
                self.program_embedding(program_states)[:, None, :]
                + self.role_embedding(role_states)
                + self.core_position_embedding(core_position_states)
                + self.depth_embedding(program_depths)[:, None, :]
            )
            return self.norm(hidden) * adapter_mask[..., None]

else:  # pragma: no cover

    class ReactionProgramConditioning:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise ReactionProgramConditioningError("reaction-program conditioning requires torch")


__all__ = [
    "ReactionProgramConditioning",
    "ReactionProgramConditioningError",
    "ReactionProgramVocabulary",
]
