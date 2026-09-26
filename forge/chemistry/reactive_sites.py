"""Shared, deterministic chemistry-policy helpers."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

RAW_SUBSTRUCTURE_MATCHES = "raw_substructure_matches"
SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES = "symmetry_distinct_required_handle_matches"
SUPPORTED_MULTIPLICITY_SEMANTICS = frozenset(
    {
        RAW_SUBSTRUCTURE_MATCHES,
        SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES,
    }
)


class ChemistryPolicyError(ValueError):
    """Raised when a chemistry-policy calculation cannot be performed safely."""


@dataclass(frozen=True)
class ReactiveSiteMultiplicity:
    """Raw handle matches and their symmetry-distinct signatures."""

    raw_matches: tuple[tuple[int, ...], ...]
    symmetry_class_signatures: tuple[tuple[int, ...], ...]

    @property
    def raw_match_count(self) -> int:
        return len(self.raw_matches)

    @property
    def symmetry_distinct_match_count(self) -> int:
        return len(self.symmetry_class_signatures)

    def count(self, semantics: str) -> int:
        """Return the multiplicity under a declared, supported interpretation."""

        if semantics == RAW_SUBSTRUCTURE_MATCHES:
            return self.raw_match_count
        if semantics == SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES:
            return self.symmetry_distinct_match_count
        raise ChemistryPolicyError(
            f"unsupported reactive-site multiplicity semantics {semantics!r}"
        )


def audit_reactive_site_multiplicity(
    molecule: Chem.Mol,
    required_handle: Chem.Mol,
) -> ReactiveSiteMultiplicity:
    """Count raw and symmetry-distinct required-handle matches.

    RDKit canonical atom ranks with ``breakTies=False`` identify atoms related
    by molecular symmetry. A query match is represented by the symmetry-rank
    tuple of its matched atoms. This is exact for the single-atom Ugi amine
    handle qualified in M0-09 and deterministic for connected role queries.
    """

    if molecule is None:
        raise ChemistryPolicyError("molecule is required")
    if required_handle is None or required_handle.GetNumAtoms() < 1:
        raise ChemistryPolicyError("required_handle must contain at least one atom")
    raw_matches = tuple(sorted(molecule.GetSubstructMatches(required_handle, uniquify=True)))
    symmetry_ranks = tuple(Chem.CanonicalRankAtoms(molecule, breakTies=False))
    symmetry_signatures = tuple(
        sorted({tuple(symmetry_ranks[atom_index] for atom_index in match) for match in raw_matches})
    )
    return ReactiveSiteMultiplicity(
        raw_matches=raw_matches,
        symmetry_class_signatures=symmetry_signatures,
    )
