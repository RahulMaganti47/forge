"""Create and load the frozen, leakage-safe R0 splits for M0-03.

Split construction is an M0 curation operation. Downstream tasks must call
``load_frozen_r0_splits`` and must not reconstruct groups from the source corpus.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CONFIG_SCHEMA_VERSION = "m0_03_r0_split_config.v2"
MANIFEST_SCHEMA_VERSION = "m0_03_r0_splits.v2"
RESULT_SCHEMA_VERSION = "m0_03_result.v2"
ASSIGNMENT_FILENAME = "r0_fold_assignments.csv"
MANIFEST_FILENAME = "manifest.json"
SCHEMES = ("source_study", "headgroup", "linker_scaffold", "component_family")
FOLDS = ("R0_train", "R0_cal", "R0_heldout")
ASSIGNMENT_FIELDS = (
    "r0_structure_id",
    "leakage_group_id",
    *(field for scheme in SCHEMES for field in (f"{scheme}_group_id", f"{scheme}_fold")),
)


class SplitError(ValueError):
    """Raised when source data or a frozen split violates the M0-03 contract."""


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except FileNotFoundError as exc:
        raise SplitError(f"required input not found: {path}") from exc
    with handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()
