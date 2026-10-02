import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.commands import artifacts
from forge.commands.artifacts import fetch, install, restore, safe_path, verify
from forge.core.hashing import sha256_file


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


def _archive(path: Path, entries: list[tuple[str, bytes, bool]]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        for name, payload, symlink in entries:
            member = tarfile.TarInfo(name)
            if symlink:
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
                stream.addfile(member)
            else:
                member.size = len(payload)
                stream.addfile(member, io.BytesIO(payload))


@pytest.fixture
def github_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    (root / "data").mkdir(parents=True)
    data = root / "data/training.csv"
    data.write_bytes(b"training data\n")
    archive = tmp_path / "weights.tar.gz"
    _archive(archive, [("weights.bin", b"weights", False)])
    config = {
        "files": [
            {
                "repo_path": "results/training.csv",
                "bundle_path": "training.csv",
                "checkout_path": "data/training.csv",
                "bytes": data.stat().st_size,
                "sha256": str(sha256_file(data)),
            },
            {
                "repo_path": "results/weights.bin",
                "bundle_path": "weights.bin",
                "bytes": 7,
                "sha256": hashlib.sha256(b"weights").hexdigest(),
            },
        ],
        "storage": {
            "github": {
                "repository": "owner/repo",
                "release": "v1",
                "archives": [
                    {
                        "name": archive.name,
                        "bytes": archive.stat().st_size,
                        "sha256": str(sha256_file(archive)),
                        "members": ["weights.bin"],
                    }
                ],
            }
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(config))

    def download(command, **kwargs):
        assert command[:4] == ["gh", "release", "download", "v1"]
        target = Path(command[command.index("--dir") + 1]) / archive.name
        shutil.copyfile(archive, target)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(artifacts.subprocess, "run", download)
    return root, manifest, archive, config


def test_install_committed_data_keeps_missing_weights_explicit(github_bundle):
    root, manifest, _, _ = github_bundle
    result = install(root, manifest)
    assert result["ready"]
    assert (root / "results/training.csv").read_bytes() == b"training data\n"
    assert result["download_paths"] == ["results/weights.bin"]
    assert not verify(root, manifest)["ready"]


def test_github_fetch_combines_committed_data_and_download(github_bundle):
    root, manifest, _, _ = github_bundle
    assert fetch(root, manifest)["ready"]
    assert (root / "results/training.csv").read_bytes() == b"training data\n"
    assert (root / "results/weights.bin").read_bytes() == b"weights"


def test_offline_fetch_uses_downloaded_archives_without_network(github_bundle, monkeypatch):
    root, manifest, archive, _ = github_bundle

    def forbidden(*args, **kwargs):
        raise AssertionError("offline restoration must not contact GitHub")

    monkeypatch.setattr(artifacts.subprocess, "run", forbidden)
    assert fetch(root, manifest, downloads=archive.parent)["ready"]


def test_changed_committed_data_stops_before_download(github_bundle, monkeypatch):
    root, manifest, _, _ = github_bundle
    (root / "data/training.csv").write_bytes(b"version https://git-lfs.github.com/spec/v1\n")

    def forbidden(*args, **kwargs):
        raise AssertionError("must verify committed data before download")

    monkeypatch.setattr(artifacts.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="git lfs pull"):
        fetch(root, manifest)
    assert not (root / "results").exists()


def test_changed_archive_stops_before_installation(github_bundle):
    root, manifest, archive, _ = github_bundle
    archive.write_bytes(b"corrupt download")
    with pytest.raises(ValueError, match="differs from its manifest"):
        fetch(root, manifest)
    assert not (root / "results").exists()


@pytest.mark.parametrize(
    "entries",
    [
        [("../outside", b"weights", False)],
        [("weights.bin", b"", True)],
        [("weights.bin", b"weights", False), ("weights.bin", b"weights", False)],
        [("other.bin", b"weights", False)],
        [],
        [("weights.bin", b"changed", False)],
    ],
)
def test_authenticated_archive_cannot_bypass_payload_checks(github_bundle, entries):
    root, manifest, archive, config = github_bundle
    _archive(archive, entries)
    record = config["storage"]["github"]["archives"][0]
    record.update(bytes=archive.stat().st_size, sha256=str(sha256_file(archive)))
    manifest.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        fetch(root, manifest)
    assert not (root / "results").exists()
    assert not (root.parent / "outside").exists()


@pytest.mark.parametrize("action", ["install", "fetch"])
def test_committed_hela_group_is_available_without_downloads(
    github_bundle, monkeypatch, capsys, action
):
    from forge.cli import main

    root, _, _, config = github_bundle
    config = {"files": config["files"][:1]}
    (root / "manifests").mkdir()
    (root / "manifests/paper-model-v1.json").write_text(json.dumps({"files": []}))
    manifest = root / "manifests/hela-oracle-v1.json"
    manifest.write_text(json.dumps(config))

    def forbidden(*args, **kwargs):
        raise AssertionError("included HeLa artifacts must not require downloads")

    monkeypatch.setattr(artifacts.subprocess, "run", forbidden)
    assert main(["artifacts", action, "--group", "hela-oracle-v1", "--root", str(root)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ready"]
    assert (root / "results/training.csv").read_bytes() == b"training data\n"
