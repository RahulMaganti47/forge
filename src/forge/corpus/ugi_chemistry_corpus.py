"""Load exact Ugi topology conditions and withheld chemistry targets by fold."""

from __future__ import annotations

import csv
import gzip
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from forge.model.conditioning.ugi import tensorize_ugi_l1_support_record
from forge.model.conditioning.ugi_chemistry import (
    ChemistryRealizationTarget,
    ChemistryTopologyCondition,
    UgiFixedCoreSchema,
    core_schema_from_record,
    materialize_chemistry_target,
    project_chemistry_topology_condition,
)
from forge.model.networks.dense_flow import AtomState
from forge.model.representation.vocabulary import load_atom_vocabulary


class UgiChemistryCorpusError(RuntimeError):
    """Raised when chemistry supervision disagrees with the frozen corpus."""


@dataclass(frozen=True)
class UgiChemistryRecord:
    """One topology-conditioned chemistry example with no component IDs."""

    product_id: str
    condition: ChemistryTopologyCondition
    target: ChemistryRealizationTarget


@dataclass(frozen=True)
class UgiChemistryCorpus:
    """Strict product folds and the invariant Ugi-core adapter schema."""

    records_by_fold: Mapping[str, tuple[UgiChemistryRecord, ...]]
    assignments_by_fold: Mapping[str, tuple[Mapping[str, str], ...]]
    atom_vocabulary: tuple[AtomState, ...]
    core_schema: UgiFixedCoreSchema


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise UgiChemistryCorpusError(f"could not read {path}: {exc}") from exc


def load_ugi_chemistry_corpus(
    assignments_path: Path,
    semantic_products_path: Path,
    semantic_atoms_path: Path,
    atom_vocabulary_path: Path,
) -> UgiChemistryCorpus:
    """Materialize strict-fold topology conditions and chemistry-only labels."""

    assignments = _read_csv(assignments_path)
    if not assignments or "primary_product_fold" not in assignments[0]:
        raise UgiChemistryCorpusError("Ugi assignments lack the strict product fold")
    assignment_by_id = {row["product_id"]: row for row in assignments}
    if len(assignment_by_id) != len(assignments):
        raise UgiChemistryCorpusError("Ugi assignment product IDs are not unique")
    products = _read_csv(semantic_products_path)
    product_by_id = {row["product_id"]: row for row in products}
    if set(product_by_id) != set(assignment_by_id):
        raise UgiChemistryCorpusError("semantic products and assignments do not match")
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(semantic_atoms_path):
        atoms_by_product[row["product_id"]].append(row)
    if set(atoms_by_product) != set(assignment_by_id):
        raise UgiChemistryCorpusError("semantic atoms and assignments do not match")

    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    records_by_fold: defaultdict[str, list[UgiChemistryRecord]] = defaultdict(list)
    assignments_by_fold: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    schema: UgiFixedCoreSchema | None = None
    for assignment in assignments:
        product_id = assignment["product_id"]
        fold = assignment["primary_product_fold"]
        if fold not in {"train", "calibration", "heldout"}:
            raise UgiChemistryCorpusError(f"unsupported product fold: {fold}")
        support = tensorize_ugi_l1_support_record(
            product_by_id[product_id],
            atoms_by_product[product_id],
            atom_to_index,
            preserve_aromaticity=True,
        )
        observed_schema = core_schema_from_record(support)
        if schema is None:
            schema = observed_schema
        if observed_schema != schema:
            raise UgiChemistryCorpusError("qualified Ugi core chemistry is not invariant")
        condition = project_chemistry_topology_condition(support, schema)
        target = materialize_chemistry_target(support, atom_to_index)
        records_by_fold[fold].append(
            UgiChemistryRecord(
                product_id=product_id,
                condition=condition,
                target=target,
            )
        )
        assignments_by_fold[fold].append(assignment)
    observed_counts = {fold: len(rows) for fold, rows in records_by_fold.items()}
    if observed_counts != {"train": 4362, "calibration": 3350, "heldout": 4674}:
        raise UgiChemistryCorpusError(f"strict chemistry folds changed: {observed_counts}")
    if schema is None:
        raise UgiChemistryCorpusError("Ugi chemistry corpus is empty")
    return UgiChemistryCorpus(
        records_by_fold={fold: tuple(rows) for fold, rows in records_by_fold.items()},
        assignments_by_fold={fold: tuple(rows) for fold, rows in assignments_by_fold.items()},
        atom_vocabulary=vocabulary,
        core_schema=schema,
    )


def load_expanded_ugi_chemistry_corpus(
    assignments_path: Path,
    semantic_products_path: Path,
    semantic_atoms_path: Path,
    atom_vocabulary_path: Path,
) -> UgiChemistryCorpus:
    """Load the compact fold-clean chemistry cover for expanded components."""

    assignments = _read_csv(assignments_path)
    required_assignment_fields = {
        "product_id",
        "primary_product_fold",
        *(
            f"{role}_smiles"
            for role in ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
        ),
    }
    if not assignments or not required_assignment_fields.issubset(assignments[0]):
        raise UgiChemistryCorpusError("expanded chemistry assignments lack required fields")
    assignment_by_id = {row["product_id"]: row for row in assignments}
    if len(assignment_by_id) != len(assignments):
        raise UgiChemistryCorpusError("expanded chemistry product IDs are not unique")
    products = _read_csv(semantic_products_path)
    product_by_id = {row["product_id"]: row for row in products}
    if set(product_by_id) != set(assignment_by_id):
        raise UgiChemistryCorpusError("expanded chemistry products and assignments do not match")
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(semantic_atoms_path):
        atoms_by_product[row["product_id"]].append(row)
    if set(atoms_by_product) != set(assignment_by_id):
        raise UgiChemistryCorpusError("expanded chemistry atoms and assignments do not match")

    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    records_by_fold: defaultdict[str, list[UgiChemistryRecord]] = defaultdict(list)
    assignments_by_fold: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    schema: UgiFixedCoreSchema | None = None
    for assignment in assignments:
        product_id = assignment["product_id"]
        fold = assignment["primary_product_fold"]
        if fold not in {"train", "calibration", "heldout"}:
            raise UgiChemistryCorpusError(f"unsupported expanded chemistry fold: {fold}")
        support = tensorize_ugi_l1_support_record(
            product_by_id[product_id],
            atoms_by_product[product_id],
            atom_to_index,
            preserve_aromaticity=True,
        )
        observed_schema = core_schema_from_record(support)
        if schema is None:
            schema = observed_schema
        if observed_schema != schema:
            raise UgiChemistryCorpusError("expanded Ugi core chemistry is not invariant")
        records_by_fold[fold].append(
            UgiChemistryRecord(
                product_id=product_id,
                condition=project_chemistry_topology_condition(support, schema),
                target=materialize_chemistry_target(support, atom_to_index),
            )
        )
        assignments_by_fold[fold].append(assignment)
    if schema is None or set(records_by_fold) != {"train", "calibration", "heldout"}:
        raise UgiChemistryCorpusError("expanded Ugi chemistry corpus is incomplete")
    return UgiChemistryCorpus(
        records_by_fold={fold: tuple(values) for fold, values in records_by_fold.items()},
        assignments_by_fold={fold: tuple(values) for fold, values in assignments_by_fold.items()},
        atom_vocabulary=vocabulary,
        core_schema=schema,
    )
