"""Production support-boundary policy for route-knowledge adapter results.

Frozen evidence adapters answer the narrower question "is this identity in my
index?"  A miss in such an index is not, by itself, evidence that an exact-L1
Ugi component is outside the declared molecular or reaction-program support.
This module supplies an additive normalization layer without changing any
historical evidence source.

The wrapper never promotes evidence.  It can only replace an unsubstantiated
``OUTSIDE_SUPPORT`` decision with ``MISSING_KNOWLEDGE`` or reject malformed
qualification as ``INVALID_INPUT``.  Exact terminal and route decisions, as
well as all other typed blockers, pass through unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from forge.synthesis.engine.planner import (
    KnowledgeDisposition,
    KnowledgeResult,
    RouteKnowledgeSource,
    RouteTarget,
)

UGI_COMPONENT_ROLES = frozenset(
    {
        "amine_head",
        "oxoester_aldehyde_body_tail",
        "isocyanide_tail",
    }
)
_EXCLUSION_CODE = re.compile(r"^[a-z][a-z0-9_]{2,127}$")
_INTERNAL_ROLE = re.compile(r"^[a-z][a-z0-9_:+.-]{2,255}$")


class Ugi3SupportBoundaryError(ValueError):
    """Raised when explicit support-boundary metadata is inconsistent."""


class MolecularSupportState(str, Enum):
    """Declared molecular/model-support state for one exact target."""

    WITHIN_DECLARED_SUPPORT = "within_declared_support"
    UNASSESSED = "unassessed"
    OUTSIDE_DECLARED_SUPPORT = "outside_declared_support"


@dataclass(frozen=True)
class AuthenticatedInternalRoleRegistry:
    """Exact internal route roles authenticated by the source composer.

    Membership is deliberately exact.  Prefixes, patterns, and wildcard
    entries are not accepted because a lookalike role must never bypass root
    qualification merely by resembling an authenticated adapter role.
    """

    roles: frozenset[str]

    def __post_init__(self) -> None:
        if not isinstance(self.roles, frozenset):
            raise Ugi3SupportBoundaryError(
                "authenticated internal roles must be supplied as a frozenset"
            )
        for role in self.roles:
            if (
                not isinstance(role, str)
                or _INTERNAL_ROLE.fullmatch(role) is None
                or "*" in role
                or "?" in role
            ):
                raise Ugi3SupportBoundaryError(
                    "authenticated internal role must be an exact machine-readable name"
                )
            if role in UGI_COMPONENT_ROLES:
                raise Ugi3SupportBoundaryError(
                    "root Ugi component roles cannot bypass qualification"
                )

    def __contains__(self, role: object) -> bool:
        return role in self.roles


@dataclass(frozen=True)
class TargetQualification:
    """Explicit preconditions for interpreting a route-adapter index miss.

    ``declared_exclusion_code`` and ``declared_exclusion_policy_locator`` are
    an all-or-none pair.  Their presence is the only authorization for a
    qualified target to retain an ``OUTSIDE_SUPPORT`` result.
    """

    exact_l1_eligible: bool
    supported_ugi_role: bool
    role_handle_qualified: bool
    molecular_support_state: MolecularSupportState
    declared_exclusion_code: str | None = None
    declared_exclusion_policy_locator: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "exact_l1_eligible",
            "supported_ugi_role",
            "role_handle_qualified",
        ):
            if not isinstance(getattr(self, name), bool):
                raise Ugi3SupportBoundaryError(f"{name} must be boolean")
        if not isinstance(self.molecular_support_state, MolecularSupportState):
            raise Ugi3SupportBoundaryError(
                "molecular_support_state must be a MolecularSupportState"
            )
        code = self.declared_exclusion_code
        locator = self.declared_exclusion_policy_locator
        if (code is None) != (locator is None):
            raise Ugi3SupportBoundaryError(
                "declared exclusion code and policy locator must be supplied together"
            )
        if code is not None:
            if not isinstance(code, str) or _EXCLUSION_CODE.fullmatch(code) is None:
                raise Ugi3SupportBoundaryError(
                    "declared exclusion code must be machine-readable snake_case"
                )
            if not isinstance(locator, str) or not locator.strip():
                raise Ugi3SupportBoundaryError("declared exclusion policy locator must be nonempty")
            if (
                self.role_handle_qualified
                and self.molecular_support_state is MolecularSupportState.WITHIN_DECLARED_SUPPORT
            ):
                raise Ugi3SupportBoundaryError(
                    "declared exclusion contradicts handle-qualified, within-support state"
                )

    @property
    def has_declared_exclusion(self) -> bool:
        """Whether a complete, machine-readable exclusion is present."""

        return self.declared_exclusion_code is not None


QualificationKey = tuple[str, str, tuple[str, ...]]


def qualification_key(target: RouteTarget) -> QualificationKey:
    """Return the exact role, constitution, and context key used by the wrapper."""

    return target.identity


def _invalid(detail: str) -> KnowledgeResult:
    return KnowledgeResult(
        disposition=KnowledgeDisposition.INVALID_INPUT,
        evidence=(),
        detail=detail,
    )


def validate_root_qualification(
    target: RouteTarget,
    qualification: TargetQualification,
) -> KnowledgeResult | None:
    """Return an invalid-input decision when root qualification is malformed."""

    if target.role not in UGI_COMPONENT_ROLES or not qualification.supported_ugi_role:
        return _invalid("target role is not a supported Ugi component role")
    if not qualification.exact_l1_eligible:
        return _invalid("target is not eligible under the exact-L1 Ugi contract")
    if qualification.molecular_support_state is MolecularSupportState.UNASSESSED:
        return _invalid("target molecular/model-support qualification is unassessed")
    return None


def normalize_support_boundary_result(
    *,
    target: RouteTarget,
    qualification: TargetQualification,
    delegate_result: KnowledgeResult,
) -> KnowledgeResult:
    """Normalize only an unsubstantiated support-boundary decision.

    Terminal/expand decisions and every non-support blocker are returned
    byte-for-byte as the same object.  An otherwise qualified target with no
    explicit exclusion turns an adapter/index miss into missing knowledge.
    """

    invalid = validate_root_qualification(target, qualification)
    if invalid is not None:
        return invalid
    if delegate_result.disposition is not KnowledgeDisposition.OUTSIDE_SUPPORT:
        return delegate_result
    if qualification.has_declared_exclusion:
        return delegate_result
    if (
        not qualification.role_handle_qualified
        or qualification.molecular_support_state is MolecularSupportState.OUTSIDE_DECLARED_SUPPORT
    ):
        return _invalid("support exclusion lacks a machine-readable code and policy locator")
    return KnowledgeResult(
        disposition=KnowledgeDisposition.MISSING_KNOWLEDGE,
        evidence=delegate_result.evidence,
        detail=(
            "qualified exact-L1 target lacks closing route knowledge; "
            f"delegate support-boundary detail: {delegate_result.detail}"
        ),
    )


class SupportBoundaryNormalizedUgi3Source:
    """Wrap one frozen source with explicit per-root support qualifications.

    Supported Ugi component roots require an explicit qualification.  Internal
    exact-route nodes use adapter-specific upstream roles and pass through, so
    authenticated exact overlays retain their original behavior.
    """

    def __init__(
        self,
        delegate: RouteKnowledgeSource,
        qualifications: Mapping[QualificationKey, TargetQualification],
        *,
        authenticated_internal_roles: AuthenticatedInternalRoleRegistry,
    ):
        self._delegate = delegate
        self._qualifications = dict(qualifications)
        if not isinstance(
            authenticated_internal_roles,
            AuthenticatedInternalRoleRegistry,
        ):
            raise Ugi3SupportBoundaryError(
                "composer must supply an AuthenticatedInternalRoleRegistry"
            )
        self._authenticated_internal_roles = authenticated_internal_roles
        if any(
            not isinstance(value, TargetQualification) for value in self._qualifications.values()
        ):
            raise Ugi3SupportBoundaryError(
                "all support-boundary qualifications must be TargetQualification values"
            )

    def lookup(self, target: RouteTarget) -> KnowledgeResult:
        if target.role in self._authenticated_internal_roles:
            return self._delegate.lookup(target)
        qualification = self._qualifications.get(qualification_key(target))
        if qualification is None:
            if target.role in UGI_COMPONENT_ROLES:
                return _invalid("explicit support-boundary qualification is missing")
            return _invalid("target role is not a supported Ugi component role")
        invalid = validate_root_qualification(target, qualification)
        if invalid is not None:
            return invalid
        delegate_result = self._delegate.lookup(target)
        return normalize_support_boundary_result(
            target=target,
            qualification=qualification,
            delegate_result=delegate_result,
        )


__all__ = [
    "AuthenticatedInternalRoleRegistry",
    "MolecularSupportState",
    "SupportBoundaryNormalizedUgi3Source",
    "TargetQualification",
    "UGI_COMPONENT_ROLES",
    "Ugi3SupportBoundaryError",
    "normalize_support_boundary_result",
    "qualification_key",
    "validate_root_qualification",
]
