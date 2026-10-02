"""Verify and restore immutable paper bundles without trusting local download contents."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from forge.core.hashing import is_sha256, sha256_file


def safe_path(root: Path, value: str) -> Path:
    relative = Path(value)
    target = root / relative
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe artifact path: {value!r}")
    if not target.resolve().is_relative_to(root.resolve()) or target.is_symlink():
        raise ValueError(f"artifact path escapes root or is a symlink: {value}")
    return target


def manifest_files(manifest: Path) -> list[dict[str, Any]]:
    records = json.loads(manifest.read_text())["files"]
    seen: set[str] = set()
    for row in records:
        for key in ("repo_path", "bundle_path"):
            safe_path(Path.cwd(), row[key])
        if "checkout_path" in row:
            safe_path(Path.cwd(), row["checkout_path"])
        if row["repo_path"] in seen or not is_sha256(row["sha256"]):
            raise ValueError("duplicate artifact destination or malformed digest")
        seen.add(row["repo_path"])
        if not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("invalid artifact byte count")
    return records


def verify(root: Path, manifest: Path, *, bundle: bool = False) -> dict[str, Any]:
    records = _verify_files(
        root, manifest_files(manifest), "bundle_path" if bundle else "repo_path"
    )
    return {
        "schema_version": "forge.release.artifact_verification.v1",
        "manifest_sha256": str(sha256_file(manifest)),
        "ready": all(row["status"] == "verified" for row in records),
        "files": records,
    }


def _verify_files(root: Path, rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    records = []
    for row in rows:
        path = safe_path(root, row[key])
        actual = str(sha256_file(path)) if path.is_file() else None
        status = (
            "missing"
            if actual is None
            else (
                "verified"
                if (actual == row["sha256"] and path.stat().st_size == row["bytes"])
                else "mismatch"
            )
        )
        records.append(
            {
                "path": row["repo_path"],
                "status": status,
                "expected_sha256": row["sha256"],
                "actual_sha256": actual,
            }
        )
    return records


def restore(root: Path, manifest: Path, bundle: Path) -> dict[str, Any]:
    """Check the entire source bundle and all conflicts before copying any payload."""
    _restore_files(root, manifest_files(manifest), bundle, "bundle_path")
    return verify(root, manifest)


def _restore_files(root: Path, rows: list[dict[str, Any]], bundle: Path, key: str) -> None:
    failures = [row for row in _verify_files(bundle, rows, key) if row["status"] != "verified"]
    if failures:
        raise ValueError(f"download is incomplete or changed: {failures}")
    for row in rows:
        target = safe_path(root, row["repo_path"])
        if target.exists() and (not target.is_file() or sha256_file(target) != row["sha256"]):
            raise ValueError(f"refusing to overwrite a different local artifact: {target}")
    for row in rows:
        target = safe_path(root, row["repo_path"])
        if target.exists():
            continue
        source = safe_path(bundle, row[key])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
        try:
            shutil.copyfile(source, temporary)
            if sha256_file(temporary) != row["sha256"]:
                raise ValueError(f"artifact changed during copy: {source}")
            # Exclusive hard link refuses a destination created concurrently.
            os.link(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


def install(root: Path, manifest: Path) -> dict[str, Any]:
    """Verify committed data and copy it to the frozen experiment input paths."""
    rows = manifest_files(manifest)
    committed = [row for row in rows if "checkout_path" in row]
    _restore_files(root, committed, root, "checkout_path")
    records = _verify_files(root, committed, "repo_path")
    return {
        "schema_version": "forge.release.data_installation.v1",
        "scope": "committed data only; checkpoints and caches may still require download",
        "manifest_sha256": str(sha256_file(manifest)),
        "ready": all(row["status"] == "verified" for row in records),
        "files": records,
        "download_paths": [row["repo_path"] for row in rows if "checkout_path" not in row],
    }


def fetch(
    root: Path,
    manifest: Path,
    *,
    profile: str,
    environment: str,
    backend: str = "github",
    downloads: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(manifest.read_text())
    if backend == "github":
        return _fetch_github(root, manifest, config["storage"]["github"], downloads)
    if backend != "modal":
        raise ValueError(f"unknown artifact backend: {backend}")
    name = manifest.stem
    location = config.get(
        "storage", {"volume": "forge-paper-artifacts", "prefix": "/paper-model-v1"}
    )
    with tempfile.TemporaryDirectory(prefix=f"forge-{name}-") as directory:
        # Modal requires an existing destination directory.
        destination = Path(directory)
        subprocess.run(
            [
                "modal",
                "volume",
                "get",
                location["volume"],
                location["prefix"],
                str(destination),
                "--env",
                environment,
            ],
            env={**os.environ, "MODAL_PROFILE": profile},
            check=True,
        )
        bundle = destination / Path(location["prefix"]).name
        return restore(root, manifest, bundle)


def _fetch_github(
    root: Path, manifest: Path, location: dict[str, Any], downloads: Path | None
) -> dict[str, Any]:
    rows = manifest_files(manifest)
    committed = [row for row in rows if "checkout_path" in row]
    failures = [
        row
        for row in _verify_files(root, committed, "checkout_path")
        if row["status"] != "verified"
    ]
    if failures:
        raise ValueError(f"committed data is missing or changed; run git lfs pull: {failures}")
    external = {row["bundle_path"]: row for row in rows if "checkout_path" not in row}
    members = [member for archive in location["archives"] for member in archive["members"]]
    if len(members) != len(set(members)) or set(members) != set(external):
        raise ValueError("release archives do not cover the declared download files exactly")
    with tempfile.TemporaryDirectory(prefix=f"forge-{manifest.stem}-") as directory:
        destination = Path(directory)
        bundle = destination / "bundle"
        bundle.mkdir()
        for row in committed:
            target = safe_path(bundle, row["bundle_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(root, row["checkout_path"]), target)
        for archive in location["archives"]:
            name = archive["name"]
            if Path(name).name != name or not name.endswith(".tar.gz"):
                raise ValueError(f"invalid release archive name: {name}")
            path = destination / name
            if downloads is not None:
                shutil.copyfile(safe_path(downloads, name), path)
            else:
                subprocess.run(
                    [
                        "gh",
                        "release",
                        "download",
                        location["release"],
                        "--repo",
                        location["repository"],
                        "--pattern",
                        name,
                        "--dir",
                        str(destination),
                    ],
                    check=True,
                )
            if path.stat().st_size != archive["bytes"] or sha256_file(path) != archive["sha256"]:
                raise ValueError(f"release archive differs from its manifest: {name}")
            _unpack_archive(path, bundle, {key: external[key] for key in archive["members"]})
        return restore(root, manifest, bundle)


def _unpack_archive(path: Path, bundle: Path, expected: dict[str, dict[str, Any]]) -> None:
    seen = set()
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            target = safe_path(bundle, member.name)
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f"unexpected release archive member: {member.name}")
            if member.size != expected[member.name]["bytes"]:
                raise ValueError(f"release member has a different size: {member.name}")
            seen.add(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
    if seen != set(expected):
        raise ValueError(f"release archive is incomplete: {sorted(set(expected) - seen)}")
