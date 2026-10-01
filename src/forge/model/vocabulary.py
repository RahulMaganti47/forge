"""Validated atom-state vocabularies for bounded molecular graph models."""

from __future__ import annotations

from pathlib import Path

from forge.core.io import read_json_object
from forge.model.defog_feasibility import AtomState


class AtomVocabularyError(ValueError):
    """Raised when an atom-vocabulary artifact is malformed or not admitted."""


def load_atom_vocabulary(path: Path) -> tuple[AtomState, ...]:
    """Load the contiguous, unique atom states admitted by a passed artifact."""

    payload = read_json_object(path, error=AtomVocabularyError, label="atom vocabulary")
    rows = payload.get("atom_vocabulary")
    if payload.get("status") != "pass" or not isinstance(rows, list) or not rows:
        raise AtomVocabularyError("atom vocabulary is not a passed nonempty artifact")
    if [int(row["index"]) for row in rows] != list(range(len(rows))):
        raise AtomVocabularyError("atom vocabulary indices must be contiguous and sorted")
    vocabulary = tuple(
        AtomState(
            str(row["symbol"]),
            int(row["formal_charge"]),
            bool(row["aromatic"]),
            int(row.get("explicit_hydrogens", 0)),
        )
        for row in rows
    )
    if len(set(vocabulary)) != len(vocabulary):
        raise AtomVocabularyError("atom vocabulary contains duplicate states")
    return vocabulary


__all__ = ["AtomVocabularyError", "load_atom_vocabulary"]
