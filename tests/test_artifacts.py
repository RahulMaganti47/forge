import hashlib
import json
from pathlib import Path

import pytest

from forge.release.artifacts import restore, safe_path, verify


def bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    download, destination = tmp_path / "download", tmp_path / "repo"
    download.mkdir()
    (download / "weights.bin").write_bytes(b"authenticated test payload")
    payload = (download / "weights.bin").read_bytes()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "repo_path": "data/weights.bin",
                        "bundle_path": "weights.bin",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ]
            }
        )
    )
    return download, destination, manifest


def test_restore_verifies_download_and_is_idempotent(tmp_path: Path) -> None:
    download, destination, manifest = bundle(tmp_path)
    assert restore(destination, manifest, download)["ready"]
    assert restore(destination, manifest, download)["ready"]
    (destination / "data/weights.bin").write_bytes(b"changed")
    assert not verify(destination, manifest)["ready"]
    with pytest.raises(ValueError, match="refusing to overwrite"):
        restore(destination, manifest, download)
    assert (destination / "data/weights.bin").read_bytes() == b"changed"


def test_corrupt_download_writes_nothing(tmp_path: Path) -> None:
    download, destination, manifest = bundle(tmp_path)
    (download / "weights.bin").write_bytes(b"different")
    with pytest.raises(ValueError, match="incomplete or changed"):
        restore(destination, manifest, download)
    assert not destination.exists()


@pytest.mark.parametrize("relative", ["../outside", "/absolute", "."])
def test_artifact_paths_cannot_escape(tmp_path: Path, relative: str) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        safe_path(tmp_path, relative)


def test_artifact_symlink_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "link").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError, match="symlink"):
        safe_path(tmp_path, "link")
