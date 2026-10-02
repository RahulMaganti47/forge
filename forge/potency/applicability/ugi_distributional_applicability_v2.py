"""Corrected distribution-aware applicability audit.

Version 1 correctly separated provenance from chemical distance but revealed a
calibration degeneracy: a calibration product's component commonly occurs in
other training-fold products, making its component distance exactly zero.  V2
calibrates the distance of each unique calibration component after excluding
that exact component identity from the fold reference.  It also uses Morgan
count fingerprints so long-chain homologues that share the same local binary
bits remain distinguishable.

The correction is additive.  V1 artifacts are retained as a diagnostic and are
not rewritten.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from forge.potency.applicability import ugi_distributional_applicability as v1

CONFIG_SCHEMA_VERSION = "phase1_ugi_distributional_applicability_config.v2"
RESULT_SCHEMA_VERSION = "phase1_ugi_distributional_applicability.v2"
GENERATED_LEDGER_SCHEMA_VERSION = "phase1_ugi_generated_distributional_applicability.v2"
HELDOUT_LEDGER_SCHEMA_VERSION = "phase1_ugi_heldout_distributional_applicability.v2"

_COUNT_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


class CountChemicalReference:
    """V1 descriptor geometry with count-based Morgan similarity."""

    def __init__(self, smiles: Iterable[str]) -> None:
        canonical = tuple(sorted({v1._canonical(str(value)) for value in smiles}))
        if not canonical:
            raise v1.UgiDistributionalApplicabilityError("chemical reference is empty")
        self.canonical = canonical
        self.canonical_set = frozenset(canonical)
        self.fingerprints = tuple(
            _COUNT_MORGAN.GetCountFingerprint(v1._molecule(value)) for value in canonical
        )
        matrix = np.asarray([v1._descriptors(value) for value in canonical], dtype=np.float64)
        self.center = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        scale = q75 - q25
        standard = np.std(matrix, axis=0)
        self.scale = np.where(scale > 1e-12, scale, np.where(standard > 1e-12, standard, 1.0))
        self.standardized = (matrix - self.center) / self.scale

    def distance(self, smiles: str) -> v1.DistancePair:
        canonical = v1._canonical(smiles)
        fingerprint = _COUNT_MORGAN.GetCountFingerprint(v1._molecule(canonical))
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, list(self.fingerprints))
        fingerprint_distance = 1.0 - float(max(similarities))
        query = (
            np.asarray(v1._descriptors(canonical), dtype=np.float64) - self.center
        ) / self.scale
        descriptor_distance = float(
            np.min(np.linalg.norm(self.standardized - query, axis=1))
            / math.sqrt(len(v1.DESCRIPTOR_NAMES))
        )
        if not (math.isfinite(fingerprint_distance) and math.isfinite(descriptor_distance)):
            raise v1.UgiDistributionalApplicabilityError("chemical distance is not finite")
        return v1.DistancePair(
            fingerprint=fingerprint_distance,
            descriptor=descriptor_distance,
            exact_identity_seen=canonical in self.canonical_set,
        )


def _references(rows: Sequence[Mapping[str, str]]) -> dict[str, CountChemicalReference]:
    return {
        "product": CountChemicalReference(row["product_smiles"] for row in rows),
        **{
            role: CountChemicalReference(row[field] for row in rows)
            for role, field in v1.ROLE_FIELDS.items()
        },
    }


def _identity_excluded_reference(
    smiles: Iterable[str],
    query_smiles: str,
) -> CountChemicalReference:
    query = v1._canonical(query_smiles)
    retained = [value for value in smiles if v1._canonical(str(value)) != query]
    return CountChemicalReference(retained)


__all__ = ["CountChemicalReference"]
