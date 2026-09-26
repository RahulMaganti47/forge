"""Shared primitives every FORGE module depends on.

This package exists because the concepts that carry the science here -- a hash-pinned input, a
canonical structure identity, a result artifact with a declared schema -- had a consistent
*convention* across the codebase and no shared *implementation*. The same helper was retyped in
dozens of modules, and the copies had drifted apart: some `_pin` variants re-hash the file and
reject symlinks, others only check path containment, so the guarantee you got depended on which
module you happened to be in.

Nothing here invents behavior. Each function is the union of what the existing copies did, chosen
to stay byte-compatible with artifacts already on disk, because those artifacts' hashes are the
paper's evidence chain and may not move. See `docs/REFACTOR_BASELINE.md`.
"""

from forge.core.hashing import (
    PinError,
    pin_record,
    resolve_pin,
    sha256_bytes,
    sha256_file,
    sha256_json,
    sha256_tree,
)
from forge.core.io import (
    atomic_write,
    read_csv,
    read_json,
    read_jsonl,
    stable_json,
    write_csv,
    write_json,
    write_jsonl,
)
from forge.core.records import ArtifactRef, PinnedInput, RecordError, is_pin
from forge.core.seeds import keyed_seed
from forge.core.types import (
    ComponentId,
    EvidenceStatus,
    ProductId,
    RoleName,
    Sha256,
    Smiles,
    SupportTier,
)

__all__ = [
    "ArtifactRef",
    "ComponentId",
    "EvidenceStatus",
    "PinError",
    "PinnedInput",
    "RecordError",
    "ProductId",
    "RoleName",
    "Sha256",
    "Smiles",
    "SupportTier",
    "atomic_write",
    "is_pin",
    "keyed_seed",
    "pin_record",
    "read_csv",
    "read_json",
    "read_jsonl",
    "resolve_pin",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_tree",
    "stable_json",
    "write_csv",
    "write_json",
    "write_jsonl",
]
