"""Domain types.

Annotation coverage in this codebase is already near-total, so the problem these types solve is
not missing annotations -- it is that the annotations say almost nothing. `dict[str, Any]` appears
around two thousand times, and every domain value is a bare `str`: a SMILES string, a SHA-256
digest, a component id and a role name are all indistinguishable to a reader and to a type checker.
Passing a product id where a component id belongs is a bug the tooling cannot currently see.

`NewType` fixes that at zero runtime cost. At runtime `Smiles("CCO")` *is* the string `"CCO"`, so
these are free to adopt incrementally: annotate a signature and nothing changes for its callers
until they are annotated too.

The enums replace stringly-typed vocabularies that AGENTS.md treats as settled. Spelling one of
these wrong currently fails silently at whatever downstream comparison happens to run.
"""

from __future__ import annotations

from enum import Enum
from typing import NewType

# --------------------------------------------------------------------------- chemical identity

Smiles = NewType("Smiles", str)
"""A SMILES string as it arrived, of unknown canonical form."""

CanonicalSmiles = NewType("CanonicalSmiles", str)
"""A SMILES string canonicalized by `forge.chemistry`.

Distinct from `Smiles` on purpose. Roughly eighty separate canonicalizers exist across this
codebase with differing flags, so "canonical" has not been a single well-defined state: two
modules could disagree about whether two structures are the same molecule. Anything typed
`CanonicalSmiles` has been through the one canonicalizer and is safe to compare by equality.
"""

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
    """The three Ugi-3CR precursor roles.

    AGENTS.md fixes the variant as an AGILE-type amine-aldehyde-isocyanide three-component
    reaction with no carboxylic-acid reactant, so the vocabulary is closed. Note the ester comes
    from the aldehyde component, which is why ester construction is an L2 problem rather than part
    of final assembly -- hence the aldehyde role's long name.

    **Two spellings exist and both are load-bearing.** The role vocabulary is declared 17 times
    across the package under four constant names (`ROLE_NAMES`, `ROLES`, `ROLE_ORDER`, `VIEWS`) in
    two incompatible forms. The long form dominates the serialized data by roughly 500:1, so it is
    the enum's value and what round-trips to an artifact. The short form appears in a comparable
    number of *modules* but barely reaches disk. No module mixes them, so the split is clean along
    module lines — which is exactly why a third spelling would go unnoticed without this type.

    Parse either with `RoleName.parse`; serialize only the long form.
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
    """Open-endedness tiers. E2 is the paper's primary claim; E3 is exploratory.

    Replaces the older I/F/B/N vocabulary, which AGENTS.md lists under settled decisions.
    """

    E0 = "E0"
    """Exact product in the frozen enumeration."""

    E1 = "E1"
    """Known assembly, all components already accepted terminal blocks."""

    E2 = "E2"
    """Known assembly, at least one component needs a generated L2 route."""

    E3 = "E3"
    """New assembly family. The paper does not depend on it."""


class SynthesisLayer(str, Enum):
    """The three layers "synthesizable" decomposes into.

    Preserving a Ugi core is insufficient: if final coupling works but an ester-bearing tail
    precursor cannot be made, the lipid is not executable. L2 is where candidate variance lives.
    """

    L1_ASSEMBLY = "L1"
    L2_SUBCOMPONENT = "L2"
    L3_PROCUREMENT = "L3"


class EvidenceStatus(str, Enum):
    """How far a result may be leaned on, from the evidence matrix vocabulary.

    `VERIFIED_NONSELECTING` and `DIAGNOSTIC_ONLY` exist to stop a descriptive audit being read as
    a selection criterion -- a distinction the manuscript contract cares about a great deal.
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
