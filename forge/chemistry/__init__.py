"""The chemistry boundary: where RDKit is used directly.

Two concerns, both of which need RDKit and neither of which any other package should reach past
this one to reach:

    smiles          molecular identity -- one canonicalizer, so "the same molecule" means one thing
    reactive_sites  reactive-site policy -- how many times a handle really matches, given symmetry

`reactive_sites` arrived here from a top-level `forge/chemistry.py` that sat beside this package
importing RDKit itself, which made the sentence above false and left two nearly identical names
with no way to tell which held what.

The claim is still aspirational rather than enforced: roughly a hundred `MolToSmiles` call sites
remain outside this package. Narrowing that is what the boundary is for.
"""

from forge.chemistry.descriptors import (
    DescriptorError,
    carbon_branch_metrics,
    component_chemotype_metrics,
    connected_molecule,
    ring_signature,
)
from forge.chemistry.reactive_sites import (
    RAW_SUBSTRUCTURE_MATCHES,
    SUPPORTED_MULTIPLICITY_SEMANTICS,
    SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES,
    ChemistryPolicyError,
    ReactiveSiteMultiplicity,
    audit_reactive_site_multiplicity,
)
from forge.chemistry.smiles import (
    ChemError,
    cache_stats,
    canonical_connected_constitution,
    canonical_constitution,
    canonical_isomeric,
    clear_caches,
    is_valid,
    parse_smiles,
    same_constitution,
)

__all__ = [
    "RAW_SUBSTRUCTURE_MATCHES",
    "SUPPORTED_MULTIPLICITY_SEMANTICS",
    "SYMMETRY_DISTINCT_REQUIRED_HANDLE_MATCHES",
    "ChemError",
    "ChemistryPolicyError",
    "DescriptorError",
    "ReactiveSiteMultiplicity",
    "audit_reactive_site_multiplicity",
    "cache_stats",
    "carbon_branch_metrics",
    "canonical_connected_constitution",
    "canonical_constitution",
    "canonical_isomeric",
    "clear_caches",
    "component_chemotype_metrics",
    "connected_molecule",
    "is_valid",
    "parse_smiles",
    "ring_signature",
    "same_constitution",
]
