"""The provenance records that carry the evidence chain.

A result artifact records what it read and what it produced. Those two records are the most
pervasive structures in the codebase and both are currently untyped dicts.

**Unknown keys are preserved, not enumerated, and that is the central design decision.** A survey
of every `sha256`-bearing record in the tracked artifacts found 1,803 path-bearing instances
carrying **32 distinct extra-key combinations**. Only 536 are the bare `{path, sha256}`; the most
common shape is actually `{job_id, path, sha256}` at 960, and the tail includes `step`, `role`,
`seed`, `rows`, `input_id`, `note`, `review_id` and `chemistry_pages`. Naming each as an optional
field would be brittle and, worse, would silently drop anything not yet seen -- and dropping a key
on round-trip corrupts an artifact whose hash is pinned. So the core pair is typed and everything
else travels verbatim in `extra`, which makes losslessness a property of the design rather than of
the field list being complete.

**These are not the only `sha256`-bearing records.** The same survey found 289 download/cache
records (keyed by `url`), 108 vendored-asset records (keyed by `asset`) and 36 tensor references
(`{dtype, shape, sha256}`), none of which name a path. They are separate concepts and deliberately
not modelled here -- forcing them into one type would produce a record where most fields are
inapplicable most of the time.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from forge.core.types import Sha256

CORE_PIN_KEYS = frozenset({"path", "sha256"})
CORE_REF_KEYS = frozenset({"schema_version", "sha256"})


class RecordError(ValueError):
    """A serialized provenance record is missing a required field or malformed."""


def _frozen(extra: Mapping[str, Any]) -> Mapping[str, Any]:
    """Make the extension mapping read-only so a frozen record is actually immutable."""
    return MappingProxyType(dict(extra))


@dataclass(frozen=True)
class PinnedInput:
    """One file a stage read, identified by path and content digest.

    The record that makes a result attributable to the bytes it came from. `bytes` is modelled
    explicitly because it appears on 214 instances and is genuinely part of the pin; every other
    extension travels in `extra`.

    Not hashable when it carries extras -- the extension mapping has no stable hash. Compare by
    equality instead, which is what call sites do.
    """

    path: str
    sha256: Sha256
    size: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _frozen(self.extra))

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> PinnedInput:
        """Parse a serialized pin, keeping every key it carries.

        Only `path` and `sha256` are required; anything else is preserved as-is so the record can
        be written back byte-identically.
        """
        missing = CORE_PIN_KEYS - set(record)
        if missing:
            raise RecordError(f"pin is missing {sorted(missing)}: {dict(record)!r}")
        return cls(
            path=str(record["path"]),
            sha256=Sha256(str(record["sha256"])),
            size=record.get("bytes"),
            extra={k: v for k, v in record.items() if k not in {"path", "sha256", "bytes"}},
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize back to the shape it was parsed from, losing nothing.

        `bytes` is emitted only when it was present, so a record that never carried a size does not
        acquire one -- that would change the artifact.
        """
        record: dict[str, Any] = {"path": self.path, "sha256": str(self.sha256)}
        if self.size is not None:
            record["bytes"] = self.size
        record.update(self.extra)
        return record

    def resolve(self, repo: Path) -> Path:
        """Verify this pin against the file it names and return it.

        Delegates to `core.hashing.resolve_pin` so there is exactly one implementation of the
        provenance check, and it stays the strict one: exact key set, well-formed digest, path
        contained in the repository, not a symlink, contents matching the digest.

        Note this validates the *core pair only*, which is what `resolve_pin` requires. A record
        carrying extras is checked on its path and digest, since those are what the guarantee rests
        on; the extras are metadata about the pin, not part of it.
        """
        from forge.core.hashing import resolve_pin

        return resolve_pin({"path": self.path, "sha256": str(self.sha256)}, repo, label=self.path)


@dataclass(frozen=True)
class ArtifactRef:
    """One file a stage produced, identified by schema version and content digest.

    Distinct from `PinnedInput` rather than an optional-field variant of it: a produced artifact is
    described by what schema it conforms to and how many rows it holds, where a consumed one is
    described by where it lives and how many bytes. Conflating them would give a record whose
    `path` is empty half the time and whose `schema_version` is empty the other half.
    """

    schema_version: str
    sha256: Sha256
    rows: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "extra", _frozen(self.extra))

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> ArtifactRef:
        missing = CORE_REF_KEYS - set(record)
        if missing:
            raise RecordError(f"artifact ref is missing {sorted(missing)}: {dict(record)!r}")
        return cls(
            schema_version=str(record["schema_version"]),
            sha256=Sha256(str(record["sha256"])),
            rows=record.get("rows"),
            extra={
                k: v for k, v in record.items() if k not in {"schema_version", "sha256", "rows"}
            },
        )

    def to_mapping(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sha256": str(self.sha256),
        }
        if self.rows is not None:
            record["rows"] = self.rows
        record.update(self.extra)
        return record


def is_pin(record: Mapping[str, Any]) -> bool:
    """Whether a mapping looks like a pinned input rather than one of the other digest records.

    Path-bearing is the discriminator. Download records key on `url`, vendored assets on `asset`,
    and tensor references carry `dtype`/`shape` with no path at all.
    """
    return CORE_PIN_KEYS <= set(record)


__all__ = [
    "CORE_PIN_KEYS",
    "CORE_REF_KEYS",
    "ArtifactRef",
    "PinnedInput",
    "RecordError",
    "is_pin",
]
