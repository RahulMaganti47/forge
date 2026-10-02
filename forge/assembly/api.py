"""Public contracts for exact L1 assembly checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ForwardAssemblyCheck:
    """Outcome of replaying a registry-defined L1 transform.

    ``exact`` establishes forward consistency only.  It is deliberately not named success,
    synthesis probability, or route certification because L2/L3 may still be open.
    """

    reaction_id: str
    roles: tuple[str, ...]
    exact: bool
    saturated: bool
    enumerated_outcomes: int

    def __post_init__(self) -> None:
        if not self.reaction_id or not self.roles:
            raise ValueError("assembly checks require a reaction id and role order")
        if self.enumerated_outcomes < 0:
            raise ValueError("enumerated outcome count must be non-negative")


@dataclass(frozen=True)
class ForwardAssemblyProducts:
    """Unique sanitized products emitted by an exact registry-backed assembly program."""

    assembly_id: str
    reaction_id: str
    roles: tuple[str, ...]
    products: tuple[str, ...]
    saturated: bool
    enumerated_outcomes_by_step: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.assembly_id or not self.reaction_id or not self.roles:
            raise ValueError("forward products require assembly, reaction and role identifiers")
        if not self.enumerated_outcomes_by_step or any(
            value < 0 for value in self.enumerated_outcomes_by_step
        ):
            raise ValueError("forward products require non-negative per-step outcome counts")
        if tuple(sorted(set(self.products))) != self.products:
            raise ValueError("forward products must be unique and deterministically ordered")


@dataclass(frozen=True)
class ReactionProgramSpec:
    """One repeatable, registry-backed synthesis-program topology.

    The accumulator is the molecule modified at each step.  ``repeat_role`` supplies one new
    component per step.  This deliberately represents programs such as successive aza-Michael
    additions to a polyamine without pretending they are a single fixed-arity reaction.
    """

    program_id: str
    reaction_id: str
    accumulator_role: str
    repeat_role: str
    minimum_steps: int
    maximum_steps: int

    def __post_init__(self) -> None:
        if not self.program_id or not self.reaction_id:
            raise ValueError("reaction programs require program and reaction identifiers")
        if not self.accumulator_role or not self.repeat_role:
            raise ValueError("reaction programs require accumulator and repeated roles")
        if self.accumulator_role == self.repeat_role:
            raise ValueError("accumulator and repeated roles must differ")
        if self.minimum_steps < 1 or self.maximum_steps < self.minimum_steps:
            raise ValueError("reaction-program step bounds are invalid")


@dataclass(frozen=True)
class ReactionProgramTrace:
    """An exact decomposition of a product into a terminal head and ordered step inputs."""

    program_id: str
    reaction_id: str
    terminal_head_smiles: str
    repeated_component_smiles: tuple[str, ...]
    intermediate_product_smiles: tuple[str, ...]

    def __post_init__(self) -> None:
        steps = len(self.repeated_component_smiles)
        if steps < 1 or len(self.intermediate_product_smiles) != steps:
            raise ValueError("a reaction-program trace requires one intermediate per step")

    @property
    def step_count(self) -> int:
        return len(self.repeated_component_smiles)


@dataclass(frozen=True)
class ReactionProgramCheck:
    """Outcome of exact forward replay for a complete multi-step program."""

    program_id: str
    reaction_id: str
    exact: bool
    saturated: bool
    steps: int
    enumerated_outcomes_by_step: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.steps < 1 or len(self.enumerated_outcomes_by_step) != self.steps:
            raise ValueError("program checks require one outcome count per step")
        if any(value < 0 for value in self.enumerated_outcomes_by_step):
            raise ValueError("program outcome counts must be non-negative")


@dataclass(frozen=True)
class ReactionProgramAtomOrigins:
    """Precursor-role and reaction-core states in canonical product atom order.

    The complete program depth is recorded once.  Individual copies of an identical repeated
    precursor are deliberately not numbered because graph symmetry can make that numbering
    non-identifiable in the final product.  Core labels are the registry product-template map
    positions (for example ``map_1``); repeated executions may therefore contain the same core
    label more than once without inventing a non-identifiable step identity.
    """

    program_id: str
    canonical_product_smiles: str
    atom_origins: tuple[str, ...]
    core_positions: tuple[str, ...]
    step_count: int

    def __post_init__(self) -> None:
        if (
            not self.canonical_product_smiles
            or not self.atom_origins
            or len(self.core_positions) != len(self.atom_origins)
            or self.step_count < 1
        ):
            raise ValueError("reaction-program atom origins are incomplete")


class AssemblyAdapter(Protocol):
    """The stable seam used by generation and routing for one L1 transform."""

    @property
    def reaction_id(self) -> str: ...

    @property
    def roles(self) -> tuple[str, ...]: ...

    def check_forward(
        self,
        components: Mapping[str, str],
        product_smiles: str,
        *,
        maximum_outcomes: int = 64,
    ) -> ForwardAssemblyCheck: ...

    def forward_products(
        self,
        components: Mapping[str, str],
        *,
        maximum_outcomes: int = 64,
    ) -> ForwardAssemblyProducts: ...


class ReactionProgramAdapter(Protocol):
    """Stable seam for a repeatable final-assembly program."""

    @property
    def spec(self) -> ReactionProgramSpec: ...

    def decompose(
        self,
        product_smiles: str,
        *,
        terminal_head_smiles: str | None = None,
        expected_repeat_smiles: str | None = None,
        maximum_outcomes: int = 512,
        maximum_states: int = 4096,
    ) -> tuple[ReactionProgramTrace, ...]: ...

    def check_forward(
        self,
        terminal_head_smiles: str,
        repeated_component_smiles: Sequence[str],
        product_smiles: str,
        *,
        maximum_outcomes: int = 512,
    ) -> ReactionProgramCheck: ...

    def forward_products(
        self,
        terminal_head_smiles: str,
        repeated_component_smiles: Sequence[str],
        *,
        maximum_outcomes: int = 512,
    ) -> ForwardAssemblyProducts: ...

    def forward_traces(
        self,
        terminal_head_smiles: str,
        repeated_component_smiles: Sequence[str],
        *,
        maximum_outcomes: int = 512,
        maximum_states: int = 4096,
    ) -> tuple[ReactionProgramTrace, ...]: ...

    def atom_origins(self, trace: ReactionProgramTrace) -> ReactionProgramAtomOrigins: ...


__all__ = [
    "AssemblyAdapter",
    "ForwardAssemblyCheck",
    "ForwardAssemblyProducts",
    "ReactionProgramAdapter",
    "ReactionProgramAtomOrigins",
    "ReactionProgramCheck",
    "ReactionProgramSpec",
    "ReactionProgramTrace",
]
