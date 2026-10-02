"""Method-blind structural realism metrics for generated ionizable lipids.

Metrics compare complete graphs with held-out observed lipid structures. Every
method uses the same ledger, retaining invalid, failed, duplicate, and out-of-support
attempts in the denominator.

These metrics do not measure ionization, formulation, delivery, or biological
activity. QED is omitted because its medicinal-chemistry prior does not measure
ionizable-lipid realism.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits

from forge.core.hashing import sha256_file, sha256_json
from forge.core.io import iter_csv
from forge.evaluation.ugi_benchmark import CommonUgiAttempt, validate_attempt_ledger
from forge.model.representation.lipid_context import (
    assign_lipid_regions,
    rooted_distances,
    select_lipid_polar_root,
)

# Thread pool used for the frozen classifier two-sample test.  This is a wall-time setting only:
# the estimator is deterministic for a fixed random state and returns bit-identical fold AUCs at
# every thread count, and this benchmark's matrices are too small for its OpenMP regions to pay off.
C2ST_OPENMP_THREADS = 1

REFERENCE_SCHEMA = "forge.common_lipid_realism_reference.v1"
ASSESSMENT_SCHEMA = "forge.common_lipid_realism_assessment.v1"
ATTEMPT_ASSESSMENT_SCHEMA = "forge.common_lipid_realism_attempt.v1"

DESCRIPTOR_NAMES = (
    "heavy_atoms",
    "molecular_weight",
    "clogp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "ring_count",
    "aromatic_ring_count",
    "fraction_csp3",
    "formal_charge",
    "absolute_formal_charge",
    "carbon_fraction",
    "nitrogen_atoms",
    "oxygen_atoms",
    "sulfur_atoms",
    "phosphorus_atoms",
    "branch_atoms",
    "unsaturated_bonds",
    "maximum_polar_root_distance",
    "head_atom_fraction",
    "interface_atom_fraction",
    "tail_atom_fraction",
    "tail_branch_atoms",
)


class CommonLipidRealismError(ValueError):
    """A realism input or calculation violates the frozen comparison contract."""


@dataclass(frozen=True)
class RealismPolicy:
    """Frozen support, reference and estimator settings."""

    allowed_elements: frozenset[str]
    maximum_heavy_atoms: int
    train_reference_limit: int
    heldout_reference_limit: int
    selection_seed: int
    manifold_neighbors: int
    fingerprint_radius: int
    fingerprint_bits: int
    distance_chunk_size: int
    internal_diversity_limit: int
    c2st_folds: int
    c2st_minimum_rows_per_class: int
    c2st_maximum_rows_per_class: int
    c2st_max_iterations: int
    c2st_seed: int

    @classmethod
    def from_mapping(cls, value: object) -> RealismPolicy:
        required = {
            "allowed_elements",
            "maximum_heavy_atoms",
            "train_reference_limit",
            "heldout_reference_limit",
            "selection_seed",
            "manifold_neighbors",
            "fingerprint_radius",
            "fingerprint_bits",
            "distance_chunk_size",
            "internal_diversity_limit",
            "c2st_folds",
            "c2st_minimum_rows_per_class",
            "c2st_maximum_rows_per_class",
            "c2st_max_iterations",
            "c2st_seed",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CommonLipidRealismError(f"realism policy must define exactly {sorted(required)}")
        elements = value["allowed_elements"]
        if (
            not isinstance(elements, Sequence)
            or isinstance(elements, (str, bytes))
            or any(not isinstance(item, str) or not item for item in elements)
            or len(set(elements)) != len(elements)
        ):
            raise CommonLipidRealismError("allowed_elements must contain unique element symbols")

        def positive(name: str) -> int:
            raw = value[name]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise CommonLipidRealismError(f"{name} must be a positive integer")
            return raw

        selection_seed = value["selection_seed"]
        c2st_seed = value["c2st_seed"]
        for name, raw in (("selection_seed", selection_seed), ("c2st_seed", c2st_seed)):
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise CommonLipidRealismError(f"{name} must be a non-negative integer")
        policy = cls(
            allowed_elements=frozenset(elements),
            maximum_heavy_atoms=positive("maximum_heavy_atoms"),
            train_reference_limit=positive("train_reference_limit"),
            heldout_reference_limit=positive("heldout_reference_limit"),
            selection_seed=selection_seed,
            manifold_neighbors=positive("manifold_neighbors"),
            fingerprint_radius=positive("fingerprint_radius"),
            fingerprint_bits=positive("fingerprint_bits"),
            distance_chunk_size=positive("distance_chunk_size"),
            internal_diversity_limit=positive("internal_diversity_limit"),
            c2st_folds=positive("c2st_folds"),
            c2st_minimum_rows_per_class=positive("c2st_minimum_rows_per_class"),
            c2st_maximum_rows_per_class=positive("c2st_maximum_rows_per_class"),
            c2st_max_iterations=positive("c2st_max_iterations"),
            c2st_seed=c2st_seed,
        )
        if policy.train_reference_limit <= policy.manifold_neighbors:
            raise CommonLipidRealismError("train reference limit must exceed manifold_neighbors")
        if policy.heldout_reference_limit <= policy.manifold_neighbors:
            raise CommonLipidRealismError("heldout reference limit must exceed manifold_neighbors")
        if policy.c2st_folds < 2:
            raise CommonLipidRealismError("c2st_folds must be at least two")
        if policy.c2st_maximum_rows_per_class < policy.c2st_minimum_rows_per_class:
            raise CommonLipidRealismError("C2ST maximum rows must not be below its minimum")
        return policy


@dataclass(frozen=True)
class ReferenceMolecule:
    structure_id: str
    canonical_smiles: str
    group_id: str


@dataclass(frozen=True)
class RobustDescriptorScale:
    center: np.ndarray
    scale: np.ndarray

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.ndim != 2 or matrix.shape[1] != len(self.center):
            raise CommonLipidRealismError("descriptor matrix shape changed")
        return (matrix - self.center) / self.scale


@dataclass(frozen=True)
class RealismReference:
    training: tuple[ReferenceMolecule, ...]
    heldout: tuple[ReferenceMolecule, ...]
    scale: RobustDescriptorScale
    heldout_descriptors: np.ndarray
    heldout_fingerprints: tuple[Any, ...]
    descriptor_radii: np.ndarray
    fingerprint_radii: np.ndarray
    audit: dict[str, Any]


def _canonical_connected(smiles: str) -> tuple[str, Chem.Mol] | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0 or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False), molecule


def _molecule_elements(molecule: Chem.Mol) -> frozenset[str]:
    return frozenset(atom.GetSymbol() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1)


def _within_support(molecule: Chem.Mol, policy: RealismPolicy) -> bool:
    return molecule.GetNumHeavyAtoms() <= policy.maximum_heavy_atoms and _molecule_elements(
        molecule
    ).issubset(policy.allowed_elements)


def _descriptor_vector(molecule: Chem.Mol) -> np.ndarray:
    atoms = list(molecule.GetAtoms())
    heavy_atoms = molecule.GetNumHeavyAtoms()
    if heavy_atoms <= 0:
        raise CommonLipidRealismError("descriptor input contains no heavy atoms")
    root = select_lipid_polar_root(molecule)
    distances = rooted_distances(molecule, root)
    regions = assign_lipid_regions(molecule, root, distances=distances)
    region_counts = np.bincount(regions, minlength=3)
    # One pass over the atoms instead of five.  Every accumulator below is the same sum over the
    # same atom order as the per-descriptor comprehensions it replaces; only the number of calls
    # back into RDKit for the same symbol, degree and charge changes.
    element_counts: Counter[str] = Counter()
    branches = 0
    tail_branches = 0
    formal_charge = 0
    absolute_formal_charge = 0
    for atom in atoms:
        element_counts[atom.GetSymbol()] += 1
        charge = atom.GetFormalCharge()
        formal_charge += charge
        absolute_formal_charge += abs(charge)
        if atom.GetAtomicNum() > 1 and atom.GetDegree() >= 3:
            branches += 1
            if int(regions[atom.GetIdx()]) == 2:
                tail_branches += 1
    unsaturated = sum(
        not bond.GetIsAromatic() and bond.GetBondTypeAsDouble() > 1.0
        for bond in molecule.GetBonds()
    )
    values = (
        heavy_atoms,
        Descriptors.MolWt(molecule),  # type: ignore[attr-defined]
        Crippen.MolLogP(molecule),  # type: ignore[attr-defined]
        rdMolDescriptors.CalcTPSA(molecule),
        rdMolDescriptors.CalcNumHBD(molecule),
        rdMolDescriptors.CalcNumHBA(molecule),
        Lipinski.NumRotatableBonds(molecule),  # type: ignore[attr-defined]
        rdMolDescriptors.CalcNumRings(molecule),
        rdMolDescriptors.CalcNumAromaticRings(molecule),
        rdMolDescriptors.CalcFractionCSP3(molecule),
        formal_charge,
        absolute_formal_charge,
        element_counts["C"] / heavy_atoms,
        element_counts["N"],
        element_counts["O"],
        element_counts["S"],
        element_counts["P"],
        branches,
        unsaturated,
        max(distances),
        region_counts[0] / heavy_atoms,
        region_counts[1] / heavy_atoms,
        region_counts[2] / heavy_atoms,
        tail_branches,
    )
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (len(DESCRIPTOR_NAMES),) or not np.isfinite(vector).all():
        raise CommonLipidRealismError("non-finite whole-lipid descriptor vector")
    return vector


def _robust_scale(matrix: np.ndarray) -> RobustDescriptorScale:
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != len(DESCRIPTOR_NAMES):
        raise CommonLipidRealismError("descriptor scaling reference is too small or malformed")
    center = np.median(matrix, axis=0)
    q25, q75 = np.quantile(matrix, (0.25, 0.75), axis=0)
    scale = q75 - q25
    standard = matrix.std(axis=0)
    scale = np.where(scale > 1e-12, scale, np.where(standard > 1e-12, standard, 1.0))
    return RobustDescriptorScale(center=center, scale=scale)


def _ranked_sample(
    rows: Sequence[ReferenceMolecule], *, population: str, seed: int, limit: int
) -> tuple[ReferenceMolecule, ...]:
    def key(row: ReferenceMolecule) -> tuple[str, str]:
        digest = hashlib.sha256(
            f"{seed}|{population}|{row.structure_id}|{row.canonical_smiles}".encode()
        ).hexdigest()
        return digest, row.structure_id

    return tuple(sorted(rows, key=key)[:limit])


def _reference_rows(
    r0_path: Path,
    splits_path: Path,
    policy: RealismPolicy,
) -> tuple[
    tuple[ReferenceMolecule, ...],
    tuple[ReferenceMolecule, ...],
    dict[str, Any],
    dict[str, Chem.Mol],
]:
    split_by_id: dict[str, dict[str, str]] = {}
    for row in iter_csv(splits_path):
        required = {"r0_structure_id", "source_study_group_id", "source_study_fold"}
        if not required.issubset(row) or row["r0_structure_id"] in split_by_id:
            raise CommonLipidRealismError("R0 split assignment schema or identity changed")
        split_by_id[row["r0_structure_id"]] = row
    populations: dict[str, list[ReferenceMolecule]] = {"R0_train": [], "R0_heldout": []}
    counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    seen_smiles: set[str] = set()
    # Every eligible constitution is parsed and canonicality-checked here.  Keeping the accepted
    # molecules lets the descriptor and fingerprint passes reuse them instead of re-parsing and
    # re-canonicalizing the selected rows; only the selected ones survive this function.
    eligible_molecules: dict[str, Chem.Mol] = {}
    for row in iter_csv(r0_path):
        required = {
            "r0_structure_id",
            "canonical_constitutional_smiles",
            "r0_pretraining_eligible",
        }
        if not required.issubset(row):
            raise CommonLipidRealismError("constitutional R0 schema changed")
        structure_id = row["r0_structure_id"]
        if structure_id in seen_ids:
            raise CommonLipidRealismError("constitutional R0 contains duplicate identities")
        seen_ids.add(structure_id)
        assignment = split_by_id.get(structure_id)
        if assignment is None:
            raise CommonLipidRealismError(f"R0 structure lacks a split assignment: {structure_id}")
        fold = assignment["source_study_fold"]
        counts[f"source_{fold}"] += 1
        if fold not in populations:
            continue
        if row["r0_pretraining_eligible"] != "True":
            counts[f"excluded_{fold}_not_pretraining_eligible"] += 1
            continue
        parsed = _canonical_connected(row["canonical_constitutional_smiles"])
        if parsed is None:
            raise CommonLipidRealismError(f"R0 contains an invalid constitution: {structure_id}")
        canonical, molecule = parsed
        if canonical != row["canonical_constitutional_smiles"]:
            raise CommonLipidRealismError(f"R0 constitution is not canonical: {structure_id}")
        if not _within_support(molecule, policy):
            counts[f"excluded_{fold}_outside_declared_support"] += 1
            continue
        if canonical in seen_smiles:
            raise CommonLipidRealismError("constitutional R0 contains duplicate molecular graphs")
        seen_smiles.add(canonical)
        eligible_molecules[structure_id] = molecule
        populations[fold].append(
            ReferenceMolecule(
                structure_id=structure_id,
                canonical_smiles=canonical,
                group_id=assignment["source_study_group_id"],
            )
        )
    training = _ranked_sample(
        populations["R0_train"],
        population="R0_train",
        seed=policy.selection_seed,
        limit=policy.train_reference_limit,
    )
    heldout = _ranked_sample(
        populations["R0_heldout"],
        population="R0_heldout",
        seed=policy.selection_seed,
        limit=policy.heldout_reference_limit,
    )
    if len(training) <= policy.manifold_neighbors or len(heldout) <= policy.manifold_neighbors:
        raise CommonLipidRealismError("R0 reference is too small for the frozen manifold estimator")
    audit = {
        "schema_version": REFERENCE_SCHEMA,
        "reference_population": (
            "source-study-held-out observed constitutional R0 lipids within the declared common "
            "atom and size support"
        ),
        "scaling_population": "R0_train only",
        "source_counts": dict(sorted(counts.items())),
        "eligible_within_support": {
            "R0_train": len(populations["R0_train"]),
            "R0_heldout": len(populations["R0_heldout"]),
        },
        "selected": {"R0_train": len(training), "R0_heldout": len(heldout)},
        "selection_sha256": {
            "R0_train": str(
                sha256_json([(row.structure_id, row.canonical_smiles) for row in training])
            ),
            "R0_heldout": str(
                sha256_json([(row.structure_id, row.canonical_smiles) for row in heldout])
            ),
        },
        "selection": "SHA-256 rank without replacement; independent of method outputs",
        "support": {
            "allowed_elements": sorted(policy.allowed_elements),
            "maximum_heavy_atoms": policy.maximum_heavy_atoms,
            "outside-support reference rows are counted and excluded, not silently dropped": True,
        },
    }
    selected = {
        row.structure_id: eligible_molecules[row.structure_id] for row in (*training, *heldout)
    }
    return training, heldout, audit, selected


def _fingerprint_radii(fingerprints: tuple[Any, ...], neighbors: int) -> np.ndarray:
    if len(fingerprints) <= neighbors:
        raise CommonLipidRealismError("fingerprint reference is too small")
    radii = np.empty(len(fingerprints), dtype=np.float64)
    references = list(fingerprints)
    for index, fingerprint in enumerate(fingerprints):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprint, references), dtype=np.float64
        )
        similarities[index] = -1.0
        kth_similarity = float(np.partition(similarities, -neighbors)[-neighbors])
        radii[index] = 1.0 - kth_similarity
    return radii


@lru_cache(maxsize=8)
def _build_realism_reference_cached(
    r0_path: Path,
    splits_path: Path,
    policy: RealismPolicy,
    r0_sha256: str,
    splits_sha256: str,
) -> RealismReference:
    """Build one immutable reference per pinned input pair and policy."""

    # Digests participate in the cache key so replacing a file at the same path cannot reuse an
    # earlier reference.  The files were hash-pinned by the caller before reaching this function.
    del r0_sha256, splits_sha256

    training, heldout, audit, molecules = _reference_rows(r0_path, splits_path, policy)
    training_matrix = np.asarray(
        [_descriptor_vector(molecules[row.structure_id]) for row in training]
    )
    scale = _robust_scale(training_matrix)
    # Use each held-out reference once instead of parsing it for its descriptors and again for its
    # fingerprint.  The fingerprint is still taken from a molecule that no descriptor pass has
    # touched, so the reference manifold is built from exactly the same bit vectors as before.
    heldout_molecules = [molecules[row.structure_id] for row in heldout]
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=policy.fingerprint_radius, fpSize=policy.fingerprint_bits
    )
    fingerprints = tuple(generator.GetFingerprint(molecule) for molecule in heldout_molecules)
    heldout_matrix = scale.transform(
        np.asarray([_descriptor_vector(molecule) for molecule in heldout_molecules])
    )
    neighbors = NearestNeighbors(n_neighbors=policy.manifold_neighbors + 1, metric="euclidean")
    descriptor_distances = neighbors.fit(heldout_matrix).kneighbors(
        heldout_matrix, return_distance=True
    )[0]
    descriptor_radii = descriptor_distances[:, -1]
    fingerprint_radii = _fingerprint_radii(fingerprints, policy.manifold_neighbors)
    audit = {
        **audit,
        "descriptors": list(DESCRIPTOR_NAMES),
        "descriptor_scaling": {
            "center": "R0_train median",
            "scale": "R0_train IQR, then standard deviation, then one for constant features",
        },
        "manifold": {
            "neighbors": policy.manifold_neighbors,
            "fingerprint": {
                "kind": "Morgan bit vector",
                "radius": policy.fingerprint_radius,
                "bits": policy.fingerprint_bits,
                "distance": "one minus Tanimoto similarity",
            },
            "descriptor_distance": "Euclidean after frozen robust R0_train scaling",
        },
    }
    return RealismReference(
        training=training,
        heldout=heldout,
        scale=scale,
        heldout_descriptors=heldout_matrix,
        heldout_fingerprints=fingerprints,
        descriptor_radii=descriptor_radii,
        fingerprint_radii=fingerprint_radii,
        audit=audit,
    )


def build_realism_reference(
    r0_path: Path,
    splits_path: Path,
    policy: RealismPolicy,
) -> RealismReference:
    """Build or reuse the deterministic, content-addressed observed-lipid reference."""

    resolved_r0 = r0_path.resolve()
    resolved_splits = splits_path.resolve()
    return _build_realism_reference_cached(
        resolved_r0,
        resolved_splits,
        policy,
        str(sha256_file(resolved_r0)),
        str(sha256_file(resolved_splits)),
    )


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    vector = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(vector.min()),
        "q10": float(np.quantile(vector, 0.1)),
        "median": float(np.quantile(vector, 0.5)),
        "q90": float(np.quantile(vector, 0.9)),
        "maximum": float(vector.max()),
        "mean": float(vector.mean()),
    }


def _effective_count(canonical_smiles: Sequence[str]) -> float:
    if not canonical_smiles:
        return 0.0
    counts = np.asarray(list(Counter(canonical_smiles).values()), dtype=np.float64)
    probabilities = counts / counts.sum()
    return float(math.exp(-float(np.sum(probabilities * np.log(probabilities)))))


def _internal_diversity(
    canonical_smiles: Sequence[str],
    policy: RealismPolicy,
    fingerprints_by_smiles: Mapping[str, Any],
) -> tuple[float | None, int]:
    """Mean pairwise ECFP distance over a frozen sample of the unique generated constitutions.

    The caller has already fingerprinted every connected generated molecule under this policy's
    Morgan settings, so those bit vectors are reused rather than re-parsed and recomputed.
    """

    unique = sorted(set(canonical_smiles))
    selected = _ranked_sample(
        [ReferenceMolecule(value, value, value) for value in unique],
        population="generated_internal_diversity",
        seed=policy.selection_seed,
        limit=policy.internal_diversity_limit,
    )
    if len(selected) < 2:
        return None, len(selected)
    fingerprints = [fingerprints_by_smiles[row.canonical_smiles] for row in selected]
    distances: list[float] = []
    for index, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[index + 1 :])
        distances.extend(1.0 - float(value) for value in similarities)
    return float(np.mean(distances)), len(selected)


def _normalized_wasserstein(generated: np.ndarray, reference: np.ndarray) -> dict[str, Any] | None:
    if len(generated) == 0:
        return None
    quantile_grid = np.linspace(0.0, 1.0, 101)
    distances = np.mean(
        np.abs(
            np.quantile(generated, quantile_grid, axis=0)
            - np.quantile(reference, quantile_grid, axis=0)
        ),
        axis=0,
    )
    return {
        "mean_across_descriptors": float(distances.mean()),
        "by_descriptor": {
            name: float(distances[index]) for index, name in enumerate(DESCRIPTOR_NAMES)
        },
        "scale": "R0_train robust standardized descriptor units",
    }


def _classifier_two_sample(
    unique_generated: Mapping[str, np.ndarray],
    reference: RealismReference,
    policy: RealismPolicy,
) -> dict[str, Any]:
    available = min(len(unique_generated), len(reference.heldout))
    if available < policy.c2st_minimum_rows_per_class:
        return {
            "status": "not_estimable",
            "reason": "too_few_unique_generated_molecules",
            "unique_generated": len(unique_generated),
            "minimum_rows_per_class": policy.c2st_minimum_rows_per_class,
        }
    rows_per_class = min(available, policy.c2st_maximum_rows_per_class)
    generated_rows = _ranked_sample(
        [ReferenceMolecule(value, value, value) for value in unique_generated],
        population="generated_c2st",
        seed=policy.c2st_seed,
        limit=rows_per_class,
    )
    reference_rows = _ranked_sample(
        reference.heldout,
        population="heldout_c2st",
        seed=policy.c2st_seed,
        limit=rows_per_class,
    )
    generated_matrix = np.asarray(
        [unique_generated[row.canonical_smiles] for row in generated_rows], dtype=np.float64
    )
    reference_by_smiles = {
        row.canonical_smiles: reference.heldout_descriptors[index]
        for index, row in enumerate(reference.heldout)
    }
    reference_matrix = np.asarray(
        [reference_by_smiles[row.canonical_smiles] for row in reference_rows], dtype=np.float64
    )
    features = np.vstack((reference_matrix, generated_matrix))
    labels = np.concatenate(
        (np.zeros(rows_per_class, dtype=np.int64), np.ones(rows_per_class, dtype=np.int64))
    )
    groups = np.asarray(
        [f"reference:{row.group_id}" for row in reference_rows]
        + [f"generated:{row.canonical_smiles}" for row in generated_rows]
    )
    reference_groups = len({row.group_id for row in reference_rows})
    folds = min(policy.c2st_folds, reference_groups, rows_per_class)
    if folds < 2:
        return {
            "status": "not_estimable",
            "reason": "too_few_reference_groups",
            "reference_groups": reference_groups,
        }
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=policy.c2st_seed)
    aucs = []
    # The frozen C2ST fits a boosted tree on a few thousand rows and two dozen features.  That is
    # far below the size where the estimator's OpenMP regions pay for themselves: on a 16-core host
    # the default thread count spends its time in fork/join barriers and the estimator runs an order
    # of magnitude slower than on one thread.  The estimator is deterministic for a fixed
    # ``random_state`` and its fold AUCs are bit-identical at 1, 2, 4, 8 and 16 threads, so limiting
    # the pool changes only the wall time.  See tests/test_assessment_output_identity.py.
    with threadpool_limits(limits=C2ST_OPENMP_THREADS, user_api="openmp"):
        for train_indices, test_indices in splitter.split(features, labels, groups):
            if len(set(labels[train_indices])) != 2 or len(set(labels[test_indices])) != 2:
                continue
            classifier = HistGradientBoostingClassifier(
                max_iter=policy.c2st_max_iterations, random_state=policy.c2st_seed
            )
            classifier.fit(features[train_indices], labels[train_indices])
            aucs.append(
                float(
                    roc_auc_score(
                        labels[test_indices], classifier.predict_proba(features[test_indices])[:, 1]
                    )
                )
            )
    if not aucs:
        return {"status": "not_estimable", "reason": "no_valid_grouped_folds"}
    return {
        "status": "estimated",
        "auc_mean": float(np.mean(aucs)),
        "auc_folds": aucs,
        "rows_per_class": rows_per_class,
        "folds": len(aucs),
        "features": list(DESCRIPTOR_NAMES),
        "grouping": (
            "real lipids by frozen source-study group; generated molecules by exact constitution"
        ),
        "interpretation": "0.5 is indistinguishable; larger values are easier to distinguish",
    }


def assess_lipid_realism(
    attempts: Sequence[CommonUgiAttempt],
    reference: RealismReference,
    policy: RealismPolicy,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assess one method/seed ledger against the fixed observed-lipid reference."""

    checked = validate_attempt_ledger([attempt.to_mapping() for attempt in attempts])
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=policy.fingerprint_radius, fpSize=policy.fingerprint_bits
    )
    rows: list[dict[str, Any]] = []
    generated: list[dict[str, Any]] = []
    for attempt in checked:
        parsed = (
            _canonical_connected(attempt.product_smiles)
            if attempt.status == "generated" and attempt.product_smiles is not None
            else None
        )
        row = {
            "schema_version": ATTEMPT_ASSESSMENT_SCHEMA,
            "method_id": attempt.method_id,
            "seed": attempt.seed,
            "attempt_index": attempt.attempt_index,
            "native_status": attempt.status,
            "connected": parsed is not None,
            "canonical_smiles": parsed[0] if parsed else None,
            "within_declared_support": False,
            "fingerprint_manifold_member": False,
            "descriptor_manifold_member": False,
            "nearest_reference_tanimoto": None,
            "nearest_reference_descriptor_distance": None,
        }
        if parsed is not None:
            canonical, molecule = parsed
            within_support = _within_support(molecule, policy)
            descriptor = reference.scale.transform(_descriptor_vector(molecule)[None, :])[0]
            generated.append(
                {
                    "row": row,
                    "canonical": canonical,
                    "within_support": within_support,
                    "descriptor": descriptor,
                    "fingerprint": generator.GetFingerprint(molecule),
                }
            )
            row["within_declared_support"] = within_support
        rows.append(row)

    fingerprint_covered = np.zeros(len(reference.heldout), dtype=bool)
    heldout_fingerprints = list(reference.heldout_fingerprints)
    for item in generated:
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(item["fingerprint"], heldout_fingerprints),
            dtype=np.float64,
        )
        distances = 1.0 - similarities
        item["row"]["nearest_reference_tanimoto"] = float(similarities.max())
        if item["within_support"]:
            contained = distances <= reference.fingerprint_radii + 1e-12
            item["row"]["fingerprint_manifold_member"] = bool(contained.any())
            fingerprint_covered |= contained

    descriptor_covered = np.zeros(len(reference.heldout), dtype=bool)
    for start in range(0, len(generated), policy.distance_chunk_size):
        chunk = generated[start : start + policy.distance_chunk_size]
        if not chunk:
            continue
        matrix = np.asarray([item["descriptor"] for item in chunk], dtype=np.float64)
        distances = np.linalg.norm(
            matrix[:, None, :] - reference.heldout_descriptors[None, :, :], axis=2
        )
        for offset, item in enumerate(chunk):
            item["row"]["nearest_reference_descriptor_distance"] = float(distances[offset].min())
            if item["within_support"]:
                contained = distances[offset] <= reference.descriptor_radii + 1e-12
                item["row"]["descriptor_manifold_member"] = bool(contained.any())
                descriptor_covered |= contained

    denominator = len(checked)
    connected = len(generated)
    canonical_smiles = [str(item["canonical"]) for item in generated]
    unique_items: dict[str, dict[str, Any]] = {}
    for item in generated:
        unique_items.setdefault(str(item["canonical"]), item)
    within_support_count = sum(bool(item["within_support"]) for item in generated)

    def manifold_summary(field: str, covered: np.ndarray) -> dict[str, Any]:
        member_attempts = sum(bool(item["row"][field]) for item in generated)
        member_unique = sum(bool(item["row"][field]) for item in unique_items.values())
        return {
            "member_attempts": member_attempts,
            "precision_per_requested_attempt": member_attempts / denominator,
            "precision_among_connected": member_attempts / connected if connected else None,
            "unique_members": member_unique,
            "precision_among_unique_connected": (
                member_unique / len(unique_items) if unique_items else None
            ),
            "heldout_reference_covered": int(covered.sum()),
            "heldout_reference_size": len(reference.heldout),
            "coverage": float(covered.mean()),
        }

    supported_descriptors = np.asarray(
        [item["descriptor"] for item in generated if item["within_support"]], dtype=np.float64
    )
    if supported_descriptors.size == 0:
        supported_descriptors = np.empty((0, len(DESCRIPTOR_NAMES)), dtype=np.float64)
    unique_supported_descriptors = {
        smiles: item["descriptor"]
        for smiles, item in unique_items.items()
        if item["within_support"]
    }
    diversity, diversity_rows = _internal_diversity(
        canonical_smiles,
        policy,
        {smiles: item["fingerprint"] for smiles, item in unique_items.items()},
    )
    result = {
        "schema_version": ASSESSMENT_SCHEMA,
        "status": "pass",
        "method_id": checked[0].method_id,
        "seed": checked[0].seed,
        "attempts": denominator,
        "molecular_output": {
            "native_generated_status": sum(attempt.status == "generated" for attempt in checked),
            "connected": connected,
            "connected_fraction_per_attempt": connected / denominator,
            "unique_connected": len(unique_items),
            "unique_fraction_among_connected": len(unique_items) / connected if connected else None,
            "effective_molecule_count": _effective_count(canonical_smiles),
            "within_declared_support": within_support_count,
            "within_declared_support_fraction_per_attempt": within_support_count / denominator,
            "within_declared_support_fraction_among_connected": (
                within_support_count / connected if connected else None
            ),
            "mean_pairwise_ecfp4_distance_among_unique": diversity,
            "internal_diversity_unique_molecules_used": diversity_rows,
        },
        "empirical_lipid_manifold": {
            "reference": "source-study-held-out observed constitutional R0 lipids",
            "fingerprint": manifold_summary("fingerprint_manifold_member", fingerprint_covered),
            "descriptor": manifold_summary("descriptor_manifold_member", descriptor_covered),
            "nearest_reference_tanimoto": _quantiles(
                [
                    float(item["row"]["nearest_reference_tanimoto"])
                    for item in generated
                    if item["row"]["nearest_reference_tanimoto"] is not None
                ]
            ),
            "nearest_reference_descriptor_distance": _quantiles(
                [
                    float(item["row"]["nearest_reference_descriptor_distance"])
                    for item in generated
                    if item["row"]["nearest_reference_descriptor_distance"] is not None
                ]
            ),
            "normalized_descriptor_wasserstein": _normalized_wasserstein(
                supported_descriptors, reference.heldout_descriptors
            ),
            "classifier_two_sample": _classifier_two_sample(
                unique_supported_descriptors, reference, policy
            ),
        },
        "reference": reference.audit,
        "coverage_and_precision_reported_separately": True,
        "attempt_denominator_includes_invalid_failed_and_out_of_support": True,
        "qed_reported": False,
        "candidate_selection": False,
        "route_or_oracle_calls": 0,
        "nonclaims": [
            "Structural realism is not ionization, formulation, delivery or biological activity.",
            "Manifold membership is proximity to held-out observed lipid structures, not proof that "
            "a molecule is synthesizable.",
            "Fingerprint and descriptor manifolds are reported separately rather than selected "
            "after observing method results.",
            "The grouped classifier is secondary and must be interpreted with precision, coverage "
            "and effective molecule count.",
        ],
    }
    return rows, result


__all__ = [
    "ASSESSMENT_SCHEMA",
    "ATTEMPT_ASSESSMENT_SCHEMA",
    "DESCRIPTOR_NAMES",
    "CommonLipidRealismError",
    "RealismPolicy",
    "RealismReference",
    "assess_lipid_realism",
    "build_realism_reference",
]
