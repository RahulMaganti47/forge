"""Typed, fail-closed foundation for bounded recursive synthesis assessment.

This module does not infer reaction chemistry or qualify a learned planner.  A
``RouteKnowledgeSource`` must supply already classified evidence.  The
recursive assessor only composes those decisions under explicit budgets while
preserving every blocker and evidence record in the returned route tree and
trace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

ROUTE_ASSESSMENT_SCHEMA_VERSION = "forge.synthesis_assessment.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class RoutePlannerError(ValueError):
    """Raised when a route-planning record violates its typed contract."""


class AssessmentOutcome(str, Enum):
    """Mutually exclusive top-level outcomes for one bounded assessment."""

    COMPLETE = "complete"
    INCOMPATIBLE = "incompatible"
    OUTSIDE_SUPPORT = "outside_support"
    MISSING_KNOWLEDGE = "missing_knowledge"
    BUDGET_EXHAUSTED = "budget_exhausted"
    INVALID_INPUT = "invalid_input"
    EXECUTION_ERROR = "execution_error"


class KnowledgeDisposition(str, Enum):
    """Decision returned by an evidence source for one route-tree node."""

    TERMINAL = "terminal"
    EXPAND = "expand"
    INCOMPATIBLE = "incompatible"
    OUTSIDE_SUPPORT = "outside_support"
    MISSING_KNOWLEDGE = "missing_knowledge"
    INVALID_INPUT = "invalid_input"
    EXECUTION_ERROR = "execution_error"


class EvidenceTier(str, Enum):
    """Evidence tiers retained without promotion by recursive composition."""

    ACCEPTED_TERMINAL = "accepted_terminal"
    EXACT_SOURCE = "exact_source"
    FAMILY_PROJECTED = "family_projected"
    PROVENANCE_ONLY = "provenance_only"


class ForwardVerificationState(str, Enum):
    """State of deterministic forward verification for one exact route step."""

    VERIFIED_EXACT_PRODUCT_UNIQUE = "verified_exact_product_unique"
    NOT_APPLICABLE = "not_applicable"
    NOT_RUN = "not_run"
    AMBIGUOUS = "ambiguous"
    MISMATCHED = "mismatched"


class AvailabilityState(str, Enum):
    """Current L3 state attached to terminal-material evidence."""

    CURRENT_CLOSED = "current_closed"
    UNAVAILABLE = "unavailable"
    EXPIRED = "expired"
    UNASSESSED = "unassessed"


class TraceAction(str, Enum):
    """Stable action vocabulary for an evidence-bearing assessment trace."""

    QUERY = "query"
    DECISION = "decision"
    EXPANSION = "expansion"
    TERMINAL = "terminal"
    BUDGET = "budget"
    CYCLE = "cycle"
    COMPOSITION = "composition"


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RoutePlannerError(f"{label} must be a nonempty string")
    return value


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoutePlannerError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class RouteTarget:
    """One role-qualified constitutional component to assess recursively."""

    role: str
    canonical_smiles: str
    product_context_smiles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.role, label="route target role")
        _nonempty(self.canonical_smiles, label="route target canonical_smiles")
        if not isinstance(self.product_context_smiles, tuple):
            raise RoutePlannerError("product_context_smiles must be a tuple")
        if any(not isinstance(value, str) or not value for value in self.product_context_smiles):
            raise RoutePlannerError("product context SMILES must be nonempty strings")

    @property
    def identity(self) -> tuple[str, str, tuple[str, ...]]:
        return self.role, self.canonical_smiles, self.product_context_smiles

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "canonical_smiles": self.canonical_smiles,
            "product_context_smiles": list(self.product_context_smiles),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RouteTarget:
        if not isinstance(value, dict):
            raise RoutePlannerError("route target must be an object")
        context = value.get("product_context_smiles", [])
        if not isinstance(context, list):
            raise RoutePlannerError("product_context_smiles must be a list")
        return cls(
            role=value.get("role"),
            canonical_smiles=value.get("canonical_smiles"),
            product_context_smiles=tuple(context),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """Source- and state-bearing evidence attached to a route decision."""

    evidence_id: str
    tier: EvidenceTier
    source_sha256: str
    source_locator: str
    exact_substrate: bool
    forward_verification: ForwardVerificationState
    availability: AvailabilityState = AvailabilityState.UNASSESSED

    def __post_init__(self) -> None:
        _nonempty(self.evidence_id, label="evidence_id")
        _nonempty(self.source_locator, label="source_locator")
        if not isinstance(self.tier, EvidenceTier):
            raise RoutePlannerError("tier must be an EvidenceTier")
        if not _SHA256_PATTERN.fullmatch(self.source_sha256):
            raise RoutePlannerError(
                "source_sha256 must contain 64 lowercase hexadecimal characters"
            )
        if not isinstance(self.exact_substrate, bool):
            raise RoutePlannerError("exact_substrate must be boolean")
        if not isinstance(self.forward_verification, ForwardVerificationState):
            raise RoutePlannerError("forward_verification has an unsupported state")
        if not isinstance(self.availability, AvailabilityState):
            raise RoutePlannerError("availability has an unsupported state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "tier": self.tier.value,
            "source_sha256": self.source_sha256,
            "source_locator": self.source_locator,
            "exact_substrate": self.exact_substrate,
            "forward_verification": self.forward_verification.value,
            "availability": self.availability.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> EvidenceRecord:
        if not isinstance(value, dict):
            raise RoutePlannerError("evidence record must be an object")
        try:
            tier = EvidenceTier(value.get("tier"))
            forward = ForwardVerificationState(value.get("forward_verification"))
            availability = AvailabilityState(value.get("availability"))
        except ValueError as exc:
            raise RoutePlannerError("evidence record contains an unsupported state") from exc
        return cls(
            evidence_id=value.get("evidence_id"),
            tier=tier,
            source_sha256=value.get("source_sha256"),
            source_locator=value.get("source_locator"),
            exact_substrate=value.get("exact_substrate"),
            forward_verification=forward,
            availability=availability,
        )


@dataclass(frozen=True)
class RouteStepProposal:
    """One source-supplied disconnection; no reaction is inferred here."""

    reaction_id: str
    reactants: tuple[RouteTarget, ...]
    evidence: tuple[EvidenceRecord, ...]
    forward_product_count: int
    verifier_calls_required: int = 0
    product_candidates_considered: int = 1

    def __post_init__(self) -> None:
        _nonempty(self.reaction_id, label="reaction_id")
        if not isinstance(self.reactants, tuple) or not self.reactants:
            raise RoutePlannerError("route step must contain at least one reactant")
        if any(not isinstance(target, RouteTarget) for target in self.reactants):
            raise RoutePlannerError("route-step reactants must be RouteTarget records")
        if not isinstance(self.evidence, tuple) or not self.evidence:
            raise RoutePlannerError("route step must preserve at least one evidence record")
        if any(not isinstance(record, EvidenceRecord) for record in self.evidence):
            raise RoutePlannerError("route-step evidence must contain EvidenceRecord values")
        _nonnegative_integer(self.forward_product_count, label="forward_product_count")
        _nonnegative_integer(self.verifier_calls_required, label="verifier_calls_required")
        _nonnegative_integer(
            self.product_candidates_considered,
            label="product_candidates_considered",
        )

    @property
    def is_exact_source_forward_verified(self) -> bool:
        return self.forward_product_count == 1 and any(
            record.tier is EvidenceTier.EXACT_SOURCE
            and record.exact_substrate
            and record.forward_verification
            is ForwardVerificationState.VERIFIED_EXACT_PRODUCT_UNIQUE
            for record in self.evidence
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reaction_id": self.reaction_id,
            "reactants": [target.to_dict() for target in self.reactants],
            "evidence": [record.to_dict() for record in self.evidence],
            "forward_product_count": self.forward_product_count,
            "verifier_calls_required": self.verifier_calls_required,
            "product_candidates_considered": self.product_candidates_considered,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RouteStepProposal:
        if not isinstance(value, dict):
            raise RoutePlannerError("route step must be an object")
        reactants = value.get("reactants")
        evidence = value.get("evidence")
        if not isinstance(reactants, list) or not isinstance(evidence, list):
            raise RoutePlannerError("route step reactants and evidence must be lists")
        return cls(
            reaction_id=value.get("reaction_id"),
            reactants=tuple(RouteTarget.from_dict(target) for target in reactants),
            evidence=tuple(EvidenceRecord.from_dict(record) for record in evidence),
            forward_product_count=value.get("forward_product_count"),
            verifier_calls_required=value.get("verifier_calls_required", 0),
            product_candidates_considered=value.get("product_candidates_considered", 1),
        )


@dataclass(frozen=True)
class KnowledgeResult:
    """One explicit decision returned by a route-knowledge source."""

    disposition: KnowledgeDisposition
    evidence: tuple[EvidenceRecord, ...]
    detail: str
    proposal: RouteStepProposal | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, KnowledgeDisposition):
            raise RoutePlannerError("disposition must be a KnowledgeDisposition")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(record, EvidenceRecord) for record in self.evidence
        ):
            raise RoutePlannerError("knowledge evidence must be a tuple of EvidenceRecord values")
        _nonempty(self.detail, label="knowledge detail")
        if self.disposition is KnowledgeDisposition.EXPAND and self.proposal is None:
            raise RoutePlannerError("expand disposition requires a route-step proposal")
        if self.disposition is not KnowledgeDisposition.EXPAND and self.proposal is not None:
            raise RoutePlannerError("only expand disposition may carry a route-step proposal")


class RouteKnowledgeSource(Protocol):
    """Evidence lookup used by the recursive assessor.

    Implementations must classify evidence explicitly.  This protocol does not
    authorize family-template expansion or learned retrosynthetic inference.
    """

    def lookup(self, target: RouteTarget) -> KnowledgeResult: ...


@dataclass(frozen=True)
class PlannerBudgetLimits:
    """Frozen per-assessment logical and search-resource ceilings."""

    maximum_depth: int
    maximum_logical_planner_calls: int
    maximum_expansions: int
    maximum_product_candidates: int
    maximum_verifier_calls: int
    maximum_elapsed_milliseconds: int

    def __post_init__(self) -> None:
        for field_name, value in self.to_dict().items():
            _nonnegative_integer(value, label=field_name)
        if self.maximum_logical_planner_calls == 0:
            raise RoutePlannerError("maximum_logical_planner_calls must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "maximum_depth": self.maximum_depth,
            "maximum_logical_planner_calls": self.maximum_logical_planner_calls,
            "maximum_expansions": self.maximum_expansions,
            "maximum_product_candidates": self.maximum_product_candidates,
            "maximum_verifier_calls": self.maximum_verifier_calls,
            "maximum_elapsed_milliseconds": self.maximum_elapsed_milliseconds,
        }

    @classmethod
    def from_dict(cls, value: Any) -> PlannerBudgetLimits:
        if not isinstance(value, dict):
            raise RoutePlannerError("planner budget limits must be an object")
        return cls(**{field_name: value.get(field_name) for field_name in cls.__annotations__})


@dataclass
class PlannerBudgetLedger:
    """Deterministic ledger separating logical calls from realized work."""

    limits: PlannerBudgetLimits
    logical_planner_calls: int = 0
    physical_cache_hits: int = 0
    physical_cache_misses: int = 0
    expansions: int = 0
    product_candidates: int = 0
    verifier_calls: int = 0
    elapsed_milliseconds: int = 0
    exhaustion_events: list[str] = field(default_factory=list)

    def _consume(self, counter: str, maximum: str, count: int, *, reason: str) -> bool:
        _nonnegative_integer(count, label=f"{counter} count")
        current = getattr(self, counter)
        if current + count > getattr(self.limits, maximum):
            self.exhaustion_events.append(reason)
            return False
        setattr(self, counter, current + count)
        return True

    def consume_logical_planner_call(self) -> bool:
        return self._consume(
            "logical_planner_calls",
            "maximum_logical_planner_calls",
            1,
            reason="maximum_logical_planner_calls",
        )

    def consume_expansion(self) -> bool:
        return self._consume(
            "expansions",
            "maximum_expansions",
            1,
            reason="maximum_expansions",
        )

    def consume_product_candidates(self, count: int) -> bool:
        return self._consume(
            "product_candidates",
            "maximum_product_candidates",
            count,
            reason="maximum_product_candidates",
        )

    def consume_verifier_calls(self, count: int) -> bool:
        return self._consume(
            "verifier_calls",
            "maximum_verifier_calls",
            count,
            reason="maximum_verifier_calls",
        )

    def record_cache_hit(self) -> None:
        self.physical_cache_hits += 1

    def record_cache_miss(self) -> None:
        self.physical_cache_misses += 1

    def record_elapsed_milliseconds(self, count: int) -> bool:
        return self._consume(
            "elapsed_milliseconds",
            "maximum_elapsed_milliseconds",
            count,
            reason="maximum_elapsed_milliseconds",
        )

    def cache_key_usage(self) -> dict[str, int]:
        """Return usage that can change an uncached assessment's result."""

        return {
            "expansions": self.expansions,
            "product_candidates": self.product_candidates,
            "verifier_calls": self.verifier_calls,
            "elapsed_milliseconds": self.elapsed_milliseconds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": self.limits.to_dict(),
            "logical_planner_calls": self.logical_planner_calls,
            "physical_cache_hits": self.physical_cache_hits,
            "physical_cache_misses": self.physical_cache_misses,
            "expansions": self.expansions,
            "product_candidates": self.product_candidates,
            "verifier_calls": self.verifier_calls,
            "elapsed_milliseconds": self.elapsed_milliseconds,
            "exhaustion_events": list(self.exhaustion_events),
        }


@dataclass(frozen=True)
class AssessmentTraceEvent:
    """One ordered, evidence-bearing event in a recursive assessment."""

    sequence: int
    depth: int
    action: TraceAction
    target: RouteTarget
    outcome: AssessmentOutcome | None
    evidence_ids: tuple[str, ...]
    detail: str

    def __post_init__(self) -> None:
        _nonnegative_integer(self.sequence, label="trace sequence")
        _nonnegative_integer(self.depth, label="trace depth")
        if not isinstance(self.action, TraceAction):
            raise RoutePlannerError("trace action has an unsupported type")
        if self.outcome is not None and not isinstance(self.outcome, AssessmentOutcome):
            raise RoutePlannerError("trace outcome has an unsupported type")
        if not isinstance(self.evidence_ids, tuple) or any(
            not isinstance(value, str) or not value for value in self.evidence_ids
        ):
            raise RoutePlannerError("trace evidence_ids must be a tuple of nonempty strings")
        _nonempty(self.detail, label="trace detail")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "depth": self.depth,
            "action": self.action.value,
            "target": self.target.to_dict(),
            "outcome": None if self.outcome is None else self.outcome.value,
            "evidence_ids": list(self.evidence_ids),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, value: Any) -> AssessmentTraceEvent:
        if not isinstance(value, dict):
            raise RoutePlannerError("trace event must be an object")
        outcome = value.get("outcome")
        evidence_ids = value.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            raise RoutePlannerError("trace evidence_ids must be a list")
        try:
            action = TraceAction(value.get("action"))
            parsed_outcome = None if outcome is None else AssessmentOutcome(outcome)
        except ValueError as exc:
            raise RoutePlannerError("trace event contains an unsupported state") from exc
        return cls(
            sequence=value.get("sequence"),
            depth=value.get("depth"),
            action=action,
            target=RouteTarget.from_dict(value.get("target")),
            outcome=parsed_outcome,
            evidence_ids=tuple(evidence_ids),
            detail=value.get("detail"),
        )


@dataclass(frozen=True)
class SynthesisRouteNode:
    """One node in the complete recursive assessment tree."""

    target: RouteTarget
    outcome: AssessmentOutcome
    evidence: tuple[EvidenceRecord, ...]
    step: RouteStepProposal | None = None
    children: tuple[SynthesisRouteNode, ...] = ()
    detail: str = "assessment recorded"

    def __post_init__(self) -> None:
        _nonempty(self.detail, label="route-node detail")
        if not isinstance(self.outcome, AssessmentOutcome):
            raise RoutePlannerError("route-node outcome has an unsupported type")
        if not isinstance(self.evidence, tuple) or any(
            not isinstance(record, EvidenceRecord) for record in self.evidence
        ):
            raise RoutePlannerError("route-node evidence must be a tuple of EvidenceRecord values")
        if not isinstance(self.children, tuple) or any(
            not isinstance(child, SynthesisRouteNode) for child in self.children
        ):
            raise RoutePlannerError("route-node children must be a tuple of route nodes")
        if self.step is None and self.children:
            raise RoutePlannerError("route node without a step cannot contain children")
        if self.step is not None and len(self.step.reactants) != len(self.children):
            raise RoutePlannerError("route step reactants and child nodes must align")

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "outcome": self.outcome.value,
            "evidence": [record.to_dict() for record in self.evidence],
            "step": None if self.step is None else self.step.to_dict(),
            "children": [child.to_dict() for child in self.children],
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, value: Any) -> SynthesisRouteNode:
        if not isinstance(value, dict):
            raise RoutePlannerError("route node must be an object")
        evidence = value.get("evidence")
        children = value.get("children")
        if not isinstance(evidence, list) or not isinstance(children, list):
            raise RoutePlannerError("route node evidence and children must be lists")
        try:
            outcome = AssessmentOutcome(value.get("outcome"))
        except ValueError as exc:
            raise RoutePlannerError("route node contains an unsupported outcome") from exc
        step = value.get("step")
        return cls(
            target=RouteTarget.from_dict(value.get("target")),
            outcome=outcome,
            evidence=tuple(EvidenceRecord.from_dict(record) for record in evidence),
            step=None if step is None else RouteStepProposal.from_dict(step),
            children=tuple(cls.from_dict(child) for child in children),
            detail=value.get("detail"),
        )


@dataclass(frozen=True)
class SynthesisAssessment:
    """Complete structured result suitable for content-addressed caching."""

    target: RouteTarget
    outcome: AssessmentOutcome
    route_tree: SynthesisRouteNode
    trace: tuple[AssessmentTraceEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, AssessmentOutcome):
            raise RoutePlannerError("assessment outcome has an unsupported type")
        if not isinstance(self.trace, tuple) or any(
            not isinstance(event, AssessmentTraceEvent) for event in self.trace
        ):
            raise RoutePlannerError("assessment trace must be a tuple of trace events")
        if self.route_tree.target != self.target or self.route_tree.outcome is not self.outcome:
            raise RoutePlannerError("assessment root does not match target and outcome")
        if [event.sequence for event in self.trace] != list(range(len(self.trace))):
            raise RoutePlannerError("assessment trace sequence must be contiguous from zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROUTE_ASSESSMENT_SCHEMA_VERSION,
            "target": self.target.to_dict(),
            "outcome": self.outcome.value,
            "route_tree": self.route_tree.to_dict(),
            "trace": [event.to_dict() for event in self.trace],
        }

    @classmethod
    def from_dict(cls, value: Any) -> SynthesisAssessment:
        if not isinstance(value, dict) or value.get("schema_version") != (
            ROUTE_ASSESSMENT_SCHEMA_VERSION
        ):
            raise RoutePlannerError("unsupported route-assessment schema")
        trace = value.get("trace")
        if not isinstance(trace, list):
            raise RoutePlannerError("assessment trace must be a list")
        try:
            outcome = AssessmentOutcome(value.get("outcome"))
        except ValueError as exc:
            raise RoutePlannerError("assessment contains an unsupported outcome") from exc
        return cls(
            target=RouteTarget.from_dict(value.get("target")),
            outcome=outcome,
            route_tree=SynthesisRouteNode.from_dict(value.get("route_tree")),
            trace=tuple(AssessmentTraceEvent.from_dict(event) for event in trace),
        )


class RoutePlanner(Protocol):
    """Public protocol for one budgeted route assessment."""

    def assess(
        self,
        target: RouteTarget,
        budget: PlannerBudgetLedger,
    ) -> SynthesisAssessment: ...


_DIRECT_OUTCOMES = {
    KnowledgeDisposition.INCOMPATIBLE: AssessmentOutcome.INCOMPATIBLE,
    KnowledgeDisposition.OUTSIDE_SUPPORT: AssessmentOutcome.OUTSIDE_SUPPORT,
    KnowledgeDisposition.MISSING_KNOWLEDGE: AssessmentOutcome.MISSING_KNOWLEDGE,
    KnowledgeDisposition.INVALID_INPUT: AssessmentOutcome.INVALID_INPUT,
    KnowledgeDisposition.EXECUTION_ERROR: AssessmentOutcome.EXECUTION_ERROR,
}
_BLOCKER_PRECEDENCE = (
    AssessmentOutcome.EXECUTION_ERROR,
    AssessmentOutcome.INVALID_INPUT,
    AssessmentOutcome.INCOMPATIBLE,
    AssessmentOutcome.OUTSIDE_SUPPORT,
    AssessmentOutcome.MISSING_KNOWLEDGE,
    AssessmentOutcome.BUDGET_EXHAUSTED,
)
