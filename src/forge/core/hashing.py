"""Content hashing, and the pinned-input check built on it.

Provenance is the load-bearing concern in this repository -- a result is only usable if you can
say which bytes produced it -- and it is also the most duplicated: roughly 238 local hash helpers
and 50 separate `_pin` implementations. The copies are not equivalent. Some re-hash the file and
reject symlinks; others only check that the path sits inside the repository. Which guarantee you
got depended on which module you were in.

`verify_pin` adopts the strictest behavior found among them, deliberately. Loosening a provenance
check to make a call site pass would be a silent downgrade of the evidence chain.

The signatures match the existing `sha256_file(path, chunk_size=1 << 20)` and `sha256_bytes(payload)`
in `forge.corpus.r0_splits` and `forge.corpus.r1_prime_audit`, so migrating a module is an import
change and nothing else.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.core.types import Sha256

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

DEFAULT_CHUNK_SIZE = 1 << 20


class PinError(ValueError):
    """A declared input pin is malformed, unreachable, or does not match its recorded digest."""


def is_sha256(value: object) -> bool:
    """True when `value` is a lowercase 64-character hex digest.

    Uppercase is rejected on purpose: digests are compared by string equality throughout the
    codebase, so accepting both cases would let two spellings of the same digest miss each other.
    """
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def sha256_bytes(payload: bytes) -> Sha256:
    """Return a byte payload's SHA-256 digest."""
    return Sha256(hashlib.sha256(payload).hexdigest())


def sha256_file(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Sha256:
    """Return a file's SHA-256 digest, read in bounded chunks.

    Chunked so that hashing the 96 MB R1 corpus or a 768 MB tensor cache does not depend on
    holding the whole file in memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return Sha256(digest.hexdigest())


def sha256_json(value: Any) -> Sha256:
    """Digest a JSON-able value by way of its canonical serialization.

    Used to fingerprint a config or a result body rather than a file, so two structurally equal
    documents hash the same regardless of key order or incidental whitespace. Defined here rather
    than left to call sites because the digest is only stable if the serialization is, and pairing
    it with a different serializer silently produces a different -- but equally plausible -- hash.
    """
    from forge.core.io import stable_json

    return sha256_bytes(stable_json(value).encode())


def sha256_tree(root: Path, *, pattern: str = "*.py") -> Sha256:
    """Digest a source tree by relative path and file content.

    Hashing file bytes alone is ambiguous: two trees with the same files under different names
    would otherwise collide.  Each entry therefore contributes its UTF-8 relative path length,
    relative path, and content digest.  Files are ordered by POSIX relative path so filesystem
    enumeration order never enters an experiment fingerprint.

    Symlinks are rejected.  A source fingerprint must describe bytes inside the declared tree,
    not whatever an external link happens to reference when the run starts.
    """
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise PinError(f"source tree is missing: {resolved_root}")
    digest = hashlib.sha256()
    files = sorted(
        (candidate for candidate in resolved_root.rglob(pattern) if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(resolved_root).as_posix(),
    )
    for candidate in files:
        if candidate.is_symlink():
            raise PinError(f"source tree contains a symlink: {candidate}")
        relative = candidate.relative_to(resolved_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(sha256_file(candidate))))
    return Sha256(digest.hexdigest())


def resolve_pin(record: Mapping[str, Any], repo: Path, *, label: str) -> Path:
    """Validate one `{"path", "sha256"}` record and return the file it names.

    Fails closed on every count: the record must have exactly those two keys, the digest must be
    well formed, the path must resolve inside `repo`, it must be a real file rather than a symlink
    pointing elsewhere, and its contents must hash to the recorded digest.

    The symlink and containment checks matter together. Without both, a pinned input could be made
    to satisfy its own hash while actually reading bytes from outside the repository.
    """
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise PinError(f"malformed input pin for {label}: expected exactly path and sha256")

    declared = record["sha256"]
    if not is_sha256(declared):
        raise PinError(f"malformed sha256 for {label}: {declared!r}")

    path = Path(record["path"])
    candidate = path if path.is_absolute() else repo / path

    # Check for a symlink before resolving, not after: resolve() follows the link, so asking a
    # resolved path whether it is a symlink always answers no.
    if candidate.is_symlink():
        raise PinError(f"pinned input for {label} is a symlink: {candidate}")

    resolved = candidate.resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError as error:
        raise PinError(
            f"pinned input for {label} resolves outside the repository: {resolved}"
        ) from error

    if not resolved.is_file():
        raise PinError(f"pinned input for {label} is missing: {resolved}")

    observed = sha256_file(resolved)
    if observed != declared:
        raise PinError(
            f"pinned input for {label} changed: expected {declared}, found {observed} at {resolved}"
        )
    return resolved


def pin_record(path: Path, repo: Path) -> dict[str, Any]:
    """Build the `{"path", "sha256", "bytes"}` record an artifact writes for one of its inputs.

    The path is recorded repository-relative so an artifact stays portable between checkouts --
    an absolute path baked into a result is what makes it unverifiable on another machine, which
    is exactly how this project lost track of its inputs once already.
    """
    resolved = path.resolve()
    return {
        "path": str(resolved.relative_to(repo.resolve())),
        "sha256": str(sha256_file(resolved)),
        "bytes": resolved.stat().st_size,
    }


def artifact_record(path: Path, *, logical_path: str | None = None) -> dict[str, Any]:
    """Describe produced bytes without leaking a staging or reproduction directory.

    Experiment outputs are first written below a temporary ``.partial`` directory and then moved
    into their final stage directory.  Recording that physical path makes otherwise identical runs
    differ and leaves a dead path after publication.  A logical artifact name plus content digest
    and size is stable across both operations.
    """

    if logical_path is not None and (not logical_path or Path(logical_path).is_absolute()):
        raise PinError("artifact logical path must be a nonempty relative path")
    return {
        "logical_path": logical_path or path.name,
        "sha256": str(sha256_file(path)),
        "bytes": path.stat().st_size,
    }


__all__ = [
    "artifact_record",
    "DEFAULT_CHUNK_SIZE",
    "PinError",
    "is_sha256",
    "pin_record",
    "resolve_pin",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "sha256_tree",
]
