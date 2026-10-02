from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.commands import qualification
from forge.commands.reproduce import verify_manuscript_rows
from forge.core.hashing import sha256_file
from forge.core.io import write_json


@pytest.fixture
def reader_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(qualification.platform, "platform", lambda: "fixture-platform")
    root = tmp_path / "release"
    for directory in ("forge", "configs", "manifests", "paper/source", "examples"):
        (root / directory).mkdir(parents=True)
    for file in (
        "forge/model.py",
        "configs/run.json",
        "paper/submission.pdf",
        "examples/check_reproduction.py",
        "uv.lock",
    ):
        (root / file).write_text("fixture")
    payload = root / "payload"
    payload.write_bytes(b"fixture")
    for group in qualification.GROUPS:
        write_json(
            root / f"manifests/{group}.json",
            {
                "files": [
                    {
                        "repo_path": "payload",
                        "bundle_path": "payload",
                        "bytes": payload.stat().st_size,
                        "sha256": str(sha256_file(payload)),
                    }
                ]
            },
        )
    return root


def test_missing_inputs_do_not_start_subprocesses(reader_root, tmp_path, monkeypatch):
    (reader_root / "payload").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("must verify before execution")

    monkeypatch.setattr(qualification.subprocess, "run", forbidden)
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="fetch or restore"):
        qualification.qualify(reader_root, output)
    assert not output.exists()


def test_subprocess_failure_records_failed_receipt(reader_root, tmp_path, monkeypatch):
    monkeypatch.setattr(
        qualification.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=2)
    )
    output = tmp_path / "result"
    with pytest.raises(RuntimeError, match="tables failed"):
        qualification.qualify(reader_root, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "fail"
    assert receipt["commands"][0]["exit_code"] == 2
    assert (output / "FAILED.json").is_file()


@pytest.mark.parametrize("different_repeat", [False, True])
def test_repeated_attempts_keep_failures_and_detect_changes(
    reader_root,
    tmp_path,
    monkeypatch,
    different_repeat,
):
    def run(command, **kwargs):
        output = Path(command[command.index("--output") + 1])
        output.mkdir()
        if command[3] == "generate":
            rows = [{"valid": False, "attempt": i} for i in range(2)]
            if different_repeat and output.name == "generation-repeat":
                rows[0]["valid"] = True
            (output / "attempts.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
            write_json(output / "summary.json", {"counts": {"attempts": 2, "valid": 0}})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(qualification.subprocess, "run", run)
    monkeypatch.setattr(qualification, "verify_manuscript_rows", lambda *args: 89)
    output = tmp_path / "result"
    if different_repeat:
        with pytest.raises(ValueError, match="deterministic"):
            qualification.qualify(reader_root, output)
        assert json.loads((output / "receipt.json").read_text())["status"] == "fail"
    else:
        receipt = qualification.qualify(reader_root, output)
        assert receipt["status"] == "pass"
        assert receipt["generation"]["summary"]["counts"]["valid"] == 0
        assert receipt["generation"]["attempts"] == 2
        assert len(receipt["commands"]) == 3
        assert receipt["outputs"]["generation/attempts.jsonl"]
        with pytest.raises(ValueError, match="already exists"):
            qualification.qualify(reader_root, output)


def test_manuscript_check_rejects_changed_numbers(reader_root, tmp_path):
    (reader_root / "paper/source/main.tex").write_text("FORGE & 963.9 \\\\")
    tables = tmp_path / "tables"
    tables.mkdir()
    (tables / "table-1.tex").write_text("FORGE & 964.0 \\\\")
    with pytest.raises(ValueError, match="differs"):
        verify_manuscript_rows(reader_root, tables)


def test_manuscript_check_rejects_incomplete_row_set(reader_root, tmp_path):
    (reader_root / "paper/source/main.tex").write_text("FORGE & 963.9 \\\\")
    tables = tmp_path / "tables"
    tables.mkdir()
    for number in range(1, 12):
        (tables / f"table-{number}.tex").write_text("FORGE & 963.9 \\\\")
    with pytest.raises(ValueError, match="expected 89 numerical rows, found 11"):
        verify_manuscript_rows(reader_root, tables)
