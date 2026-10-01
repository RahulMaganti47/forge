"""Reviewer entry-point checks. Fixture/smoke weights never stand in for paper weights."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from forge.core.hashing import sha256_file
from forge.release import generate as command


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "fixture-repo"
    base = "results/phase1/shared_bias_parallel_program_role_seed0_v2"
    inputs = {
        "production_cache": "cache.npz",
        "production_design": f"{base}/study_design.json",
    }
    declared = {}
    for name, relative in inputs.items():
        _write(root / relative, {"fixture": name})
        declared[name] = {"path": relative, "sha256": str(sha256_file(root / relative))}
    pins = [declared["production_design"]]
    for name in ("checkpoints.tar", "training_result.json"):
        relative = f"{base}/{name}"
        _write(root / relative, {"fixture": name})
        pins.append({"path": relative, "sha256": str(sha256_file(root / relative))})
    _write(
        root / command.MANIFEST,
        {"files": [{"repo_path": row["path"], "sha256": row["sha256"]} for row in pins]},
    )
    _write(
        root
        / "configs/multireaction/shared_bias_parallel_program_role_seed0_core_saturation_v2.json",
        {
            "inputs": declared,
            "full": {
                "checkpoint_steps": [9143],
                "sample_steps": 32,
                "terminal_decode_policy": "strict_reaction_core_saturation_argmax",
            },
        },
    )
    monkeypatch.setattr(command, "ROOT", root)
    return root


def test_preflight_never_imports_runtime(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "forge.release._generation_runtime", None)
    assert command.main(["--check-inputs"]) == 0
    assert json.loads(capsys.readouterr().out)["ready"] is True


@pytest.mark.parametrize("mode", ["missing", "mismatch"])
def test_incomplete_or_changed_bundle_stops_before_runtime(
    bundle: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    path = Path(command.check_inputs(bundle)["inputs"]["checkpoint"]["path"])
    if mode == "missing":
        path.unlink()
    else:
        path.write_bytes(b"wrong checkpoint")
    monkeypatch.setitem(sys.modules, "forge.release._generation_runtime", None)
    output = bundle / "output"
    assert command.main(["--output", str(output)]) == 2
    assert mode in capsys.readouterr().err
    assert not output.exists()


def test_relocated_bundle_must_match_recorded_hashes(bundle: Path, tmp_path: Path) -> None:
    original = Path(command.check_inputs(bundle)["inputs"]["checkpoint"]["path"])
    destination = tmp_path / "download"
    shutil.copytree(original.parent, destination)
    assert command.check_inputs(bundle, checkpoint=destination / "checkpoints.tar")["ready"]
    (destination / "training_result.json").write_text("different training run")
    report = command.check_inputs(bundle, checkpoint=destination / "checkpoints.tar")
    assert not report["ready"]
    assert report["inputs"]["training_result"]["status"] == "mismatch"


def test_config_escape_and_decoder_changes_are_rejected(bundle: Path) -> None:
    path = (
        bundle
        / "configs/multireaction/shared_bias_parallel_program_role_seed0_core_saturation_v2.json"
    )
    config = json.loads(path.read_text())
    config["full"]["sample_steps"] = 2
    _write(path, config)
    with pytest.raises(ValueError, match="sampling settings"):
        command.check_inputs(bundle)
    config["full"]["sample_steps"] = 32
    config["inputs"]["production_cache"]["path"] = "../outside"
    _write(path, config)
    with pytest.raises(ValueError, match="escapes"):
        command.check_inputs(bundle)


def test_all_attempts_and_provenance_are_published(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "program_id": "ugi_3cr_agile",
            "valid": True,
            "canonical_smiles": "CC",
            "exact_l1_program": False,
        },
        {
            "program_id": "ugi_3cr_agile",
            "valid": False,
            "canonical_smiles": None,
            "exact_l1_program": False,
        },
    ]
    generate = Mock(return_value=(rows, {"fixture": True}))
    monkeypatch.setitem(
        sys.modules, "forge.release._generation_runtime", SimpleNamespace(generate=generate)
    )
    output = bundle / "output"
    assert command.main(["--count", "2", "--seed", "7", "--output", str(output)]) == 0
    assert generate.call_count == 1  # no retries/top-up
    assert generate.call_args.kwargs["seed"] == 7
    assert [
        json.loads(line) for line in (output / "attempts.jsonl").read_text().splitlines()
    ] == rows
    with (output / "molecules.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["counts"] == {"attempts": 2, "valid": 1, "exact_l1": 0}
    assert summary["full_reproduction_verified"] is False
    for name, digest in summary["output_sha256"].items():
        assert sha256_file(output / name) == digest
    assert command.main(["--count", "2", "--output", str(output)]) == 2
    assert generate.call_count == 1  # refuse overwrite before sampling


def test_runtime_failure_and_wrong_count_leave_no_output(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generate = Mock(side_effect=RuntimeError("fixture failure"))
    monkeypatch.setitem(
        sys.modules, "forge.release._generation_runtime", SimpleNamespace(generate=generate)
    )
    output = bundle / "output"
    assert command.main(["--output", str(output)]) == 2
    assert not output.exists()
    generate.side_effect = None
    generate.return_value = ([], {})
    assert command.main(["--output", str(output)]) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "args", [["--count", "0"], ["--seed", "-1"], ["--family", "other"], ["--replicate", "3"]]
)
def test_invalid_request_rejected_before_preflight(args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        command.main(["--check-inputs", *args])
    assert error.value.code == 2


@pytest.mark.parametrize("replicate", [0, 1, 2])
def test_real_repository_inputs_have_no_mismatches(replicate: int) -> None:
    report = command.check_inputs(Path(__file__).resolve().parents[1], replicate=replicate)
    assert all(row["status"] != "mismatch" for row in report["inputs"].values())
    assert report["checkpoint_step"] == 9143
    assert report["sample_steps"] == 32
