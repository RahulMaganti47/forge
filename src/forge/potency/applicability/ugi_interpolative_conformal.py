"""Bin-conditional conformal audit for the fixed HeLa oracle.

This additive audit does not refit or select an oracle.  It reuses the
calibration predictions written by the already-frozen graph-oracle fits,
classifies calibration structures with the version-3 distributional policy,
and estimates a fold-specific split-conformal radius from calibration rows in
the interpolative bin.  The radius is then evaluated once on outer-test rows
that the frozen version-3 audit also classified as interpolative.

The audit is deliberately nonauthorizing.  It was motivated after inspection
of the version-3 outer-test diagnostic and therefore cannot, by itself,
retroactively create a pristine guidance test.  It establishes whether
conditional calibration is technically coherent and whether a later bounded,
versioned authorization review is warranted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from forge.corpus.r1_prime_audit import sha256_file
from forge.potency.applicability import ugi_distributional_applicability as v1
from forge.potency.applicability import ugi_distributional_applicability_v2 as v2

CONFIG_SCHEMA_VERSION = "phase1_ugi_interpolative_conformal_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_interpolative_conformal.v1"

_SCHEME_ROLES = {
    "held_head_5fold": ("amine",),
    "held_aldehyde_5fold": ("aldehyde",),
    "held_isocyanide_5fold": ("isocyanide",),
    "held_head_aldehyde_pair_5fold": ("amine", "aldehyde"),
    "held_head_isocyanide_pair_5fold": ("amine", "isocyanide"),
    "held_aldehyde_isocyanide_pair_5fold": ("aldehyde", "isocyanide"),
}


class UgiInterpolativeConformalError(ValueError):
    """Raised when conditional calibration violates its frozen contract."""


def _product_reference(
    rows: Sequence[Mapping[str, str]],
    query: Mapping[str, str],
    held_roles: Sequence[str],
) -> v2.CountChemicalReference:
    """Exclude products sharing any exact held-role component identity."""

    retained = []
    for candidate in rows:
        shares_identity = any(
            v1._canonical(candidate[v1.ROLE_FIELDS[role]])
            == v1._canonical(query[v1.ROLE_FIELDS[role]])
            for role in held_roles
        )
        if not shares_identity:
            retained.append(candidate["product_smiles"])
    return v2.CountChemicalReference(retained)


def _calibration_bins(
    rows: Sequence[Mapping[str, str]],
    training_rows: Sequence[Mapping[str, str]],
    held_roles: Sequence[str],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> list[str]:
    """Classify a fold with cached exact-identity-excluded references."""

    standard_components = {
        role: v2.CountChemicalReference(candidate[field] for candidate in training_rows)
        for role, field in v1.ROLE_FIELDS.items()
    }
    component_cache: dict[tuple[str, str], v2.CountChemicalReference] = {}
    product_cache: dict[tuple[str, ...], v2.CountChemicalReference] = {}
    output = []
    for row in rows:
        held_identity = tuple(v1._canonical(row[v1.ROLE_FIELDS[role]]) for role in held_roles)
        if held_identity not in product_cache:
            product_cache[held_identity] = _product_reference(training_rows, row, held_roles)
        references = {"product": product_cache[held_identity]}
        for role, field in v1.ROLE_FIELDS.items():
            if role not in held_roles:
                references[role] = standard_components[role]
                continue
            component = v1._canonical(row[field])
            key = (role, component)
            if key not in component_cache:
                component_cache[key] = v2._identity_excluded_reference(
                    (candidate[field] for candidate in training_rows), component
                )
            references[role] = component_cache[key]
        distances = v1._distance_fields(references, row, generated=False)
        _, overall = v1._bins(distances, thresholds)
        output.append(overall)
    return output


def _selected_calibration_ensembles(
    repo: Path,
    graph_result: Mapping[str, Any],
    *,
    endpoint: str,
    representation: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    entries = graph_result.get("fit_sources", {}).get("entries")
    if not isinstance(entries, list):
        raise UgiInterpolativeConformalError("graph fit-source registry is missing")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise UgiInterpolativeConformalError("invalid graph fit-source entry")
        relative = str(entry["path"])
        if f"/{representation}/{endpoint}/" not in relative or not any(
            f"/{scheme}/" in relative for scheme in _SCHEME_ROLES
        ):
            continue
        path = repo / relative
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise UgiInterpolativeConformalError("graph fit-source hash mismatch")
        fit = v1._read_json(path, "graph fit")
        job = fit.get("job", {})
        if (
            job.get("architecture") != representation
            or job.get("endpoint") != endpoint
            or job.get("scheme") not in _SCHEME_ROLES
        ):
            continue
        grouped[(str(job["scheme"]), int(job["fold"]))].append(fit)

    ensembles: dict[tuple[str, int], dict[str, Any]] = {}
    for key, fits in sorted(grouped.items()):
        fits.sort(key=lambda fit: int(fit["job"]["seed"]))
        if len(fits) != 3:
            raise UgiInterpolativeConformalError(f"selected ensemble lacks three seeds: {key}")
        labels = fits[0]["calibration"]["labels"]
        truth = fits[0]["calibration"]["truth"]
        if any(
            fit["calibration"]["labels"] != labels or fit["calibration"]["truth"] != truth
            for fit in fits[1:]
        ):
            raise UgiInterpolativeConformalError(f"calibration rows differ by seed: {key}")
        predictions = np.asarray(
            [fit["calibration"]["prediction"] for fit in fits], dtype=np.float64
        )
        ensembles[key] = {
            "labels": list(labels),
            "truth": np.asarray(truth, dtype=np.float64),
            "prediction": np.mean(predictions, axis=0),
            "seeds": [int(fit["job"]["seed"]) for fit in fits],
        }
    return ensembles


__all__ = ["UgiInterpolativeConformalError"]
