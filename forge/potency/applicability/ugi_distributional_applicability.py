"""Distribution-aware applicability audit for generated Ugi lipids.

The earlier fresh-pool oracle audit records whether exact precursor graphs were
present in the measured AGILE corpus.  Exact identity is provenance, not an
applicability domain.  This additive audit keeps that provenance field and
independently measures chemical distance in four views: complete product,
amine, aldehyde and isocyanide.

Applicability thresholds are learned without targets, predictions or errors.
For each held-component split, calibration structures are compared with that
fold's training structures.  Pooled calibration-distance quantiles then define
``interpolative``, ``boundary`` and ``extrapolative`` bins.  Only after those
thresholds are frozen in memory are the selected oracle's outer-test
predictions summarized within bins.  Generated candidates are compared with
the full measured corpus.  The audit is descriptive and cannot authorize
biological guidance or candidate selection.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

CONFIG_SCHEMA_VERSION = "phase1_ugi_distributional_applicability_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_distributional_applicability.v1"
GENERATED_LEDGER_SCHEMA_VERSION = "phase1_ugi_generated_distributional_applicability.v1"
HELDOUT_LEDGER_SCHEMA_VERSION = "phase1_ugi_heldout_distributional_applicability.v1"

SELECTED_REPRESENTATION = "ugi_component_role_aware_dmpnn"
SELECTED_MODEL = "neural_3seed_ensemble"
ENDPOINT = "expt_Hela"

ROLE_FIELDS = {
    "amine": "A_smiles",
    "aldehyde": "B_smiles",
    "isocyanide": "C_smiles",
}
GENERATED_ROLE_FIELDS = {
    "amine": "amine_smiles",
    "aldehyde": "aldehyde_smiles",
    "isocyanide": "isocyanide_smiles",
}
ROLE_SCHEMES = {
    "amine": "held_head_5fold",
    "aldehyde": "held_aldehyde_5fold",
    "isocyanide": "held_isocyanide_5fold",
}
EVALUATION_SCHEMES = (
    "held_head_5fold",
    "held_aldehyde_5fold",
    "held_isocyanide_5fold",
    "held_head_aldehyde_pair_5fold",
    "held_head_isocyanide_pair_5fold",
    "held_aldehyde_isocyanide_pair_5fold",
    "lantern_scaffold_balanced",
)
VIEWS = ("product", "amine", "aldehyde", "isocyanide")
DISTANCE_TYPES = ("fingerprint", "descriptor")
BIN_ORDER = {"interpolative": 0, "boundary": 1, "extrapolative": 2}

DESCRIPTOR_NAMES = (
    "heavy_atoms",
    "carbon_atoms",
    "hetero_atoms",
    "molecular_weight",
    "logp",
    "tpsa",
    "rotatable_bonds",
    "ring_count",
    "formal_charge_abs",
    "double_bonds",
    "triple_bonds",
    "branch_excess",
    "ester_count",
    "amide_count",
    "ether_count",
)

GENERATED_FIELDS = (
    "sample_index",
    "product_id",
    "product_smiles",
    "amine_smiles",
    "aldehyde_smiles",
    "isocyanide_smiles",
    "exact_identity_provenance",
    "exact_unseen_roles_json",
    "product_fingerprint_distance",
    "product_descriptor_distance",
    "amine_fingerprint_distance",
    "amine_descriptor_distance",
    "aldehyde_fingerprint_distance",
    "aldehyde_descriptor_distance",
    "isocyanide_fingerprint_distance",
    "isocyanide_descriptor_distance",
    "product_distribution_bin",
    "amine_distribution_bin",
    "aldehyde_distribution_bin",
    "isocyanide_distribution_bin",
    "overall_distribution_bin",
    "ensemble_mean_descriptive_only",
    "ensemble_standard_deviation",
    "guidance_action",
)

HELDOUT_FIELDS = (
    "scheme",
    "fold",
    "label",
    "distribution_bin",
    "product_fingerprint_distance",
    "product_descriptor_distance",
    "amine_fingerprint_distance",
    "amine_descriptor_distance",
    "aldehyde_fingerprint_distance",
    "aldehyde_descriptor_distance",
    "isocyanide_fingerprint_distance",
    "isocyanide_descriptor_distance",
    "y_true",
    "y_pred",
    "absolute_error",
    "conformal_q90",
    "covered90",
)


class UgiDistributionalApplicabilityError(ValueError):
    """Raised when the distribution-aware audit violates its frozen contract."""


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiDistributionalApplicabilityError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiDistributionalApplicabilityError(f"{label} must contain one JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        if path.suffix == ".gz":
            handle = gzip.open(path, "rt", newline="")
        else:
            handle = path.open("r", newline="")
        with handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise UgiDistributionalApplicabilityError(f"invalid CSV: {path}") from error
    if not rows:
        raise UgiDistributionalApplicabilityError(f"CSV has no rows: {path}")
    return rows


@cache
def _molecule(smiles: str) -> Chem.Mol:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiDistributionalApplicabilityError(f"invalid molecular graph: {smiles!r}")
    return molecule


@cache
def _canonical(smiles: str) -> str:
    return Chem.MolToSmiles(_molecule(smiles), canonical=True, isomericSmiles=False)


_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_AMIDE = Chem.MolFromSmarts("[CX3](=O)[NX3]")
_ETHER = Chem.MolFromSmarts("[#6][OX2][#6]")


@cache
def _fingerprint(smiles: str):
    return _MORGAN.GetFingerprint(_molecule(smiles))


@cache
def _descriptors(smiles: str) -> tuple[float, ...]:
    molecule = _molecule(smiles)
    atoms = list(molecule.GetAtoms())
    bonds = list(molecule.GetBonds())
    heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in atoms)
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    hetero_atoms = sum(atom.GetAtomicNum() not in (1, 6) for atom in atoms)
    double_bonds = sum(bond.GetBondType() == Chem.BondType.DOUBLE for bond in bonds)
    triple_bonds = sum(bond.GetBondType() == Chem.BondType.TRIPLE for bond in bonds)
    branch_excess = sum(max(0, atom.GetDegree() - 2) for atom in atoms if atom.GetAtomicNum() > 1)
    return (
        float(heavy_atoms),
        float(carbon_atoms),
        float(hetero_atoms),
        float(Descriptors.MolWt(molecule)),
        float(Crippen.MolLogP(molecule)),
        float(rdMolDescriptors.CalcTPSA(molecule)),
        float(Lipinski.NumRotatableBonds(molecule)),
        float(rdMolDescriptors.CalcNumRings(molecule)),
        float(sum(abs(atom.GetFormalCharge()) for atom in atoms)),
        float(double_bonds),
        float(triple_bonds),
        float(branch_excess),
        float(len(molecule.GetSubstructMatches(_ESTER))),
        float(len(molecule.GetSubstructMatches(_AMIDE))),
        float(len(molecule.GetSubstructMatches(_ETHER))),
    )


@dataclass(frozen=True)
class DistancePair:
    fingerprint: float
    descriptor: float
    exact_identity_seen: bool


class ChemicalReference:
    """Nearest-neighbor chemical reference with robust descriptor scaling."""

    def __init__(self, smiles: Iterable[str]) -> None:
        canonical = tuple(sorted({_canonical(str(value)) for value in smiles}))
        if not canonical:
            raise UgiDistributionalApplicabilityError("chemical reference is empty")
        self.canonical = canonical
        self.canonical_set = frozenset(canonical)
        self.fingerprints = tuple(_fingerprint(value) for value in canonical)
        matrix = np.asarray([_descriptors(value) for value in canonical], dtype=np.float64)
        self.descriptors = matrix
        self.center = np.median(matrix, axis=0)
        q25 = np.quantile(matrix, 0.25, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        scale = q75 - q25
        standard = np.std(matrix, axis=0)
        self.scale = np.where(scale > 1e-12, scale, np.where(standard > 1e-12, standard, 1.0))
        self.standardized = (matrix - self.center) / self.scale

    def distance(self, smiles: str) -> DistancePair:
        canonical = _canonical(smiles)
        similarities = DataStructs.BulkTanimotoSimilarity(
            _fingerprint(canonical), self.fingerprints
        )
        fingerprint = 1.0 - float(max(similarities))
        query = (np.asarray(_descriptors(canonical), dtype=np.float64) - self.center) / self.scale
        descriptor = float(
            np.min(np.linalg.norm(self.standardized - query, axis=1))
            / math.sqrt(len(DESCRIPTOR_NAMES))
        )
        if not (math.isfinite(fingerprint) and math.isfinite(descriptor)):
            raise UgiDistributionalApplicabilityError("chemical distance is not finite")
        return DistancePair(
            fingerprint=fingerprint,
            descriptor=descriptor,
            exact_identity_seen=canonical in self.canonical_set,
        )


def _distance_fields(
    references: Mapping[str, ChemicalReference],
    row: Mapping[str, str],
    *,
    generated: bool,
) -> dict[str, DistancePair]:
    output = {"product": references["product"].distance(row["product_smiles"])}
    fields = GENERATED_ROLE_FIELDS if generated else ROLE_FIELDS
    for role, field in fields.items():
        output[role] = references[role].distance(row[field])
    return output


def _training_rows(
    curated: Mapping[str, Mapping[str, str]],
    assignments: Mapping[tuple[str, int, str], str],
    scheme: str,
    fold: int,
    stage: str,
) -> list[Mapping[str, str]]:
    labels = [
        label
        for (candidate_scheme, candidate_fold, label), candidate_stage in assignments.items()
        if candidate_scheme == scheme and candidate_fold == fold and candidate_stage == stage
    ]
    if not labels:
        raise UgiDistributionalApplicabilityError(f"split has no {stage} records: {scheme}/{fold}")
    try:
        return [curated[label] for label in sorted(labels)]
    except KeyError as error:
        raise UgiDistributionalApplicabilityError(
            "split label is absent from curated AGILE"
        ) from error


def _split_index(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, int, str], str]:
    output: dict[tuple[str, int, str], str] = {}
    for row in rows:
        key = (str(row["scheme"]), int(row["fold"]), str(row["label"]))
        if key in output:
            raise UgiDistributionalApplicabilityError("duplicate split assignment")
        output[key] = str(row["stage"])
    return output


def _view_bin(distance: DistancePair, thresholds: Mapping[str, Mapping[str, float]]) -> str:
    if (
        distance.fingerprint <= thresholds["fingerprint"]["interpolative_max"]
        and distance.descriptor <= thresholds["descriptor"]["interpolative_max"]
    ):
        return "interpolative"
    if (
        distance.fingerprint <= thresholds["fingerprint"]["boundary_max"]
        and distance.descriptor <= thresholds["descriptor"]["boundary_max"]
    ):
        return "boundary"
    return "extrapolative"


def _bins(
    distances: Mapping[str, DistancePair],
    thresholds: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[dict[str, str], str]:
    views = {view: _view_bin(distances[view], thresholds[view]) for view in VIEWS}
    overall = max(views.values(), key=BIN_ORDER.__getitem__)
    return views, overall


__all__ = ["ChemicalReference", "DistancePair", "UgiDistributionalApplicabilityError"]
