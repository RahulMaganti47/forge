"""Source identity for executable experiment code."""

from __future__ import annotations

from pathlib import Path

from forge.core.hashing import sha256_file, sha256_json, sha256_tree

SOURCE_DIRECTORIES = (
    "forge",
    "experiments/_runtime",
    "experiments/installation_smoke",
    "experiments/phase1",
)
SOURCE_FILES = ("experiments/__init__.py", "experiments/catalog.py")


def source_manifest(
    repo: Path,
    paths: tuple[str, ...] = SOURCE_DIRECTORIES + SOURCE_FILES,
) -> dict[str, str]:
    """Hash exactly the library, runtime, catalog, and active experiment applications."""

    manifest: dict[str, str] = {}
    for relative in paths:
        path = repo / relative
        if path.is_dir():
            manifest[relative] = str(sha256_tree(path))
        elif path.is_file():
            manifest[relative] = str(sha256_file(path))
        else:
            raise FileNotFoundError(f"experiment source path is missing: {path}")
    return dict(sorted(manifest.items()))


def source_fingerprint(
    repo: Path,
    paths: tuple[str, ...] = SOURCE_DIRECTORIES + SOURCE_FILES,
) -> str:
    """Return one digest for the complete executable experiment source set."""

    return str(sha256_json(source_manifest(repo, paths)))


__all__ = ["SOURCE_DIRECTORIES", "SOURCE_FILES", "source_fingerprint", "source_manifest"]
