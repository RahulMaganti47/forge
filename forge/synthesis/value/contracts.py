"""Structured pre-prospective synthesis values without success probabilities.

The records in this module summarize typed route assessments for later
guidance experiments.  They preserve blocker taxonomy and expose only a
Pareto-style partial order.  They deliberately do not define a scalar, a route
likelihood, or a probability of experimental synthesis success.
"""

from __future__ import annotations

from enum import Enum

SYNTHESIS_VALUE_SCHEMA_VERSION = "forge.synthesis_value.v1"
PRODUCT_SYNTHESIS_VALUE_SCHEMA_VERSION = "forge.product_synthesis_value.v1"


class EvidenceSupport(str, Enum):
    """Weakest evidence class retained anywhere in an assessment."""

    MISSING = "missing"
    PROVENANCE_ONLY = "provenance_only"
    FAMILY_PROJECTED = "family_projected"
    EXACT_IDENTITY = "exact_identity"
    NOT_APPLICABLE = "not_applicable"


class ForwardConsistency(str, Enum):
    """Worst forward-verification state retained by an assessment."""

    FAILED = "failed"
    UNVERIFIED = "unverified"
    NOT_APPLICABLE = "not_applicable"
    EXACT_UNIQUE = "exact_unique"


_FORWARD_RANK = {
    ForwardConsistency.FAILED: 0,
    ForwardConsistency.UNVERIFIED: 1,
    ForwardConsistency.NOT_APPLICABLE: 2,
    ForwardConsistency.EXACT_UNIQUE: 2,
}
_EVIDENCE_RANK = {
    EvidenceSupport.MISSING: 0,
    EvidenceSupport.PROVENANCE_ONLY: 1,
    EvidenceSupport.FAMILY_PROJECTED: 2,
    EvidenceSupport.EXACT_IDENTITY: 3,
    EvidenceSupport.NOT_APPLICABLE: 3,
}
