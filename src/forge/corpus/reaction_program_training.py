"""Load exact reaction programs into a role-blocked sparse training corpus."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.assembly import ReactionProgramSpec
from forge.core.io import read_csv_rows, read_json_object
from forge.corpus.reaction_program_records import admits_reaction_program_structure
from forge.model.defog_feasibility import AtomState
from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_graph import (
    ReactionProgramGraphRecord,
    build_reaction_program_atom_vocabulary,
    tensorize_reaction_program_product,
)


class ReactionProgramTrainingCorpusError(ValueError):
    """Persisted program artifacts disagree with their model contract."""


@dataclass(frozen=True)
class ReactionProgramTrainingCorpus:
    """Model-ready records and source-balanced sampling weights by disjoint fold."""

    specifications: tuple[ReactionProgramSpec, ...]
    vocabulary: ReactionProgramVocabulary
    atom_vocabulary: tuple[AtomState, ...]
    records_by_fold: Mapping[str, tuple[ReactionProgramGraphRecord, ...]]
    weights_by_fold: Mapping[str, np.ndarray]

    @property
    def maxima(self) -> dict[str, int]:
        records = tuple(record for values in self.records_by_fold.values() for record in values)
        return {
            "heavy_atoms": max(record.node_count for record in records),
            "closures": max(record.graph.closure_count for record in records),
            "program_depth": max(record.program_depth for record in records),
            "accumulator_atoms": max(record.accumulator_atom_count for record in records),
            "repeat_component_atoms": max(
                count for record in records for count in record.repeat_atom_counts
            ),
        }


def reaction_program_specifications(
    config: Mapping[str, Any],
) -> tuple[ReactionProgramSpec, ...]:
    """Parse the shared repeated-program definitions from a validated config mapping."""

    raw_programs = config.get("programs")
    if not isinstance(raw_programs, list) or not raw_programs:
        raise ReactionProgramTrainingCorpusError("program config has no reaction programs")
    return tuple(
        ReactionProgramSpec(
            program_id=str(raw["program_id"]),
            reaction_id=str(raw["reaction_id"]),
            accumulator_role=str(raw["accumulator_role"]),
            repeat_role=str(raw["repeat_role"]),
            minimum_steps=int(raw["minimum_steps"]),
            maximum_steps=int(raw["maximum_steps"]),
        )
        for raw in raw_programs
    )


def load_reaction_program_specifications(path: Path) -> tuple[ReactionProgramSpec, ...]:
    """Load the authoritative multi-reaction config without loading its graph corpus."""

    config = read_json_object(
        path,
        error=ReactionProgramTrainingCorpusError,
        label="reaction-program config",
    )
    if config.get("schema_version") != "forge.multireaction_lnpdb_config.v1":
        raise ReactionProgramTrainingCorpusError("reaction-program config schema changed")
    return reaction_program_specifications(config)


def load_reaction_program_training_corpus(
    *,
    program_config_path: Path,
    atlas_path: Path,
    semantic_atoms_path: Path,
    splits_path: Path,
    declared_elements: set[str],
) -> ReactionProgramTrainingCorpus:
    """Load model inputs without importing the corpus-producing experiment."""

    config = read_json_object(
        program_config_path,
        error=ReactionProgramTrainingCorpusError,
        label="reaction-program config",
    )
    specs = reaction_program_specifications(config)
    spec_by_program = {spec.program_id: spec for spec in specs}
    atlas_rows = read_csv_rows(
        atlas_path,
        error=ReactionProgramTrainingCorpusError,
        label="reaction-program atlas",
        required_fields=(
            "record_id",
            "program_id",
            "canonical_product_smiles",
            "step_count",
            "disposition",
            "semantic_origin_status",
        ),
    )
    semantic_rows = read_csv_rows(
        semantic_atoms_path,
        error=ReactionProgramTrainingCorpusError,
        label="reaction-program semantic atoms",
        required_fields=(
            "record_id",
            "program_id",
            "atom_index",
            "origin_role",
            "core_position",
            "program_depth",
        ),
    )
    vocabulary = ReactionProgramVocabulary.from_specs(
        specs,
        core_positions=tuple(
            sorted(
                {
                    f"{row['program_id']}:{row['core_position']}"
                    for row in semantic_rows
                    if row["core_position"]
                }
            )
        ),
    )
    split_rows = read_csv_rows(
        splits_path,
        error=ReactionProgramTrainingCorpusError,
        label="reaction-program splits",
        required_fields=(
            "record_id",
            "program_id",
            "product_fold",
            "source_balanced_weight",
        ),
    )
    atlas = {row["record_id"]: row for row in atlas_rows if admits_reaction_program_structure(row)}
    splits = {row["record_id"]: row for row in split_rows}
    if set(atlas) != set(splits):
        raise ReactionProgramTrainingCorpusError("atlas and split records do not agree")
    atoms_by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in semantic_rows:
        atoms_by_record[row["record_id"]].append(row)
    if set(atoms_by_record) != set(atlas):
        raise ReactionProgramTrainingCorpusError("semantic atoms do not cover admitted products")
    train_smiles = [
        atlas[record_id]["canonical_product_smiles"]
        for record_id, split in splits.items()
        if split["product_fold"] == "train"
    ]
    atom_vocabulary = build_reaction_program_atom_vocabulary(
        train_smiles,
        declared_elements=declared_elements,
    )

    records_by_fold: dict[str, list[ReactionProgramGraphRecord]] = defaultdict(list)
    weights_by_fold: dict[str, list[float]] = defaultdict(list)
    for record_id in sorted(atlas):
        row = atlas[record_id]
        program_id = row["program_id"]
        if program_id not in spec_by_program or splits[record_id]["program_id"] != program_id:
            raise ReactionProgramTrainingCorpusError(f"program identity changed: {record_id}")
        semantic = sorted(atoms_by_record[record_id], key=lambda value: int(value["atom_index"]))
        if [int(value["atom_index"]) for value in semantic] != list(range(len(semantic))):
            raise ReactionProgramTrainingCorpusError(
                f"semantic atom indices are not contiguous: {record_id}"
            )
        depths = {int(value["program_depth"]) for value in semantic}
        depth = int(row["step_count"])
        if depths != {depth}:
            raise ReactionProgramTrainingCorpusError(f"program depth changed: {record_id}")
        spec = spec_by_program[program_id]
        record = tensorize_reaction_program_product(
            record_id=record_id,
            program_id=program_id,
            canonical_product_smiles=row["canonical_product_smiles"],
            atom_roles=[value["origin_role"] for value in semantic],
            atom_core_positions=[
                (f"{program_id}:{value['core_position']}" if value["core_position"] else "exterior")
                for value in semantic
            ],
            program_depth=depth,
            accumulator_role=spec.accumulator_role,
            repeat_role=spec.repeat_role,
            vocabulary=vocabulary,
            atom_vocabulary=atom_vocabulary,
        )
        fold = splits[record_id]["product_fold"]
        if fold not in {"train", "calibration", "heldout"}:
            raise ReactionProgramTrainingCorpusError(f"unknown product fold: {fold!r}")
        weight = float(splits[record_id]["source_balanced_weight"])
        if not np.isfinite(weight) or weight <= 0:
            raise ReactionProgramTrainingCorpusError(f"invalid sampling weight: {record_id}")
        records_by_fold[fold].append(record)
        weights_by_fold[fold].append(weight)
    if set(records_by_fold) != {"train", "calibration", "heldout"}:
        raise ReactionProgramTrainingCorpusError("one or more component-disjoint folds is empty")
    normalized_weights = {
        fold: np.asarray(values, dtype=np.float64) / np.sum(values)
        for fold, values in weights_by_fold.items()
    }
    return ReactionProgramTrainingCorpus(
        specifications=specs,
        vocabulary=vocabulary,
        atom_vocabulary=atom_vocabulary,
        records_by_fold={fold: tuple(values) for fold, values in records_by_fold.items()},
        weights_by_fold=normalized_weights,
    )


__all__ = [
    "ReactionProgramTrainingCorpus",
    "ReactionProgramTrainingCorpusError",
    "load_reaction_program_specifications",
    "load_reaction_program_training_corpus",
    "reaction_program_specifications",
]
