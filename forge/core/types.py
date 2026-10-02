"""Molecular identities, artifact identifiers, and evidence vocabularies.

NewType distinguishes string values for static checking without changing their
runtime representation. Enums define accepted spellings for serialized records.
"""

from __future__ import annotations

from enum import Enum
from typing import NewType

# --------------------------------------------------------------------------- chemical identity

Smiles = NewType("Smiles", str)
"""A SMILES string as it arrived, of unknown canonical form."""

CanonicalSmiles = NewType("CanonicalSmiles", str)
"""A SMILES string canonicalized under the caller's declared chemistry policy."""

InChIKey = NewType("InChIKey", str)

# --------------------------------------------------------------------------- identifiers

Sha256 = NewType("Sha256", str)
"""A lowercase hex SHA-256 digest, 64 characters. Validate with `forge.core.hashing.is_sha256`."""

ProductId = NewType("ProductId", str)
ComponentId = NewType("ComponentId", str)
ReactionId = NewType("ReactionId", str)
SchemaVersion = NewType("SchemaVersion", str)

# --------------------------------------------------------------------------- vocabularies


class RoleName(str, Enum):
    """The three AGILE-type Ugi 3CR precursor roles.

    The ester comes from the aldehyde component; ester construction belongs to L2,
    not final assembly. There is no carboxylic-acid reactant.

    Parse either the short or long spelling with ``RoleName.parse``. Serialize the
    long form used by recorded artifacts.
    """

    AMINE = "amine_head"
    ALDEHYDE = "oxoester_aldehyde_body_tail"
    ISOCYANIDE = "isocyanide_tail"

    @property
    def short(self) -> str:
        """The abbreviated spelling, for the modules and configs that use it."""
        return {"amine_head": "amine", "oxoester_aldehyde_body_tail": "aldehyde"}.get(
            self.value, "isocyanide"
        )

    @classmethod
    def parse(cls, value: str) -> RoleName:
        """Accept either spelling. Raises on anything else rather than guessing."""
        for role in cls:
            if value == role.value or value == role.short:
                return role
        raise ValueError(f"unknown Ugi role {value!r}; expected one of {cls.spellings()}")

    @classmethod
    def spellings(cls) -> tuple[str, ...]:
        """Every accepted spelling, long and short, for error messages and validation."""
        return tuple(s for role in cls for s in (role.value, role.short))


class SupportTier(str, Enum):
    """Open-endedness tiers: E2 is the primary claim; E3 is exploratory."""

    E0 = "E0"
    """Exact product in the frozen enumeration."""

    E1 = "E1"
    """Known assembly, all components already accepted terminal blocks."""

    E2 = "E2"
    """Known assembly, at least one component needs a generated L2 route."""

    E3 = "E3"
    """New assembly family. The paper does not depend on it."""


class SynthesisLayer(str, Enum):
    """Final assembly, subcomponent synthesis, and procurement.

    Final-assembly consistency alone does not establish precursor makeability or
    procurement closure.
    """

    L1_ASSEMBLY = "L1"
    L2_SUBCOMPONENT = "L2"
    L3_PROCUREMENT = "L3"


class EvidenceStatus(str, Enum):
    """Evidence status and permitted use of a result.

    A nonselecting or diagnostic audit cannot serve as a candidate-selection criterion.
    """

    VERIFIED_FROZEN = "verified_frozen"
    VERIFIED_NONSELECTING = "verified_nonselecting"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    TRAINED_PENDING_EVALUATION = "trained_pending_evaluation"
    PROPOSED_PENDING = "proposed_pending"
    HISTORICAL_STALE = "historical_stale"


class ClaimClass(str, Enum):
    """Claim vocabulary from the paper-writing contract."""

    MEASURED = "Measured"
    COMPUTED = "Computed"
    REPORTED = "Reported"
    INFERRED = "Inferred"
    PROPOSED = "Proposed"


__all__ = [
    "CanonicalSmiles",
    "ClaimClass",
    "ComponentId",
    "EvidenceStatus",
    "InChIKey",
    "ProductId",
    "ReactionId",
    "RoleName",
    "SchemaVersion",
    "Sha256",
    "Smiles",
    "SupportTier",
    "SynthesisLayer",
]
