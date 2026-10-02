from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.baselines import requests as native_baseline_ports
from forge.cli import main
from forge.commands.evaluation import evaluate
from forge.core.hashing import artifact_record, sha256_file
from forge.core.io import write_csv, write_json
from forge.experiments import evaluation as production_evaluation

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def custom_run(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "release"
    (root / "manifests").mkdir(parents=True)
    write_json(root / "manifests/paper-model-v1.json", {"files": []})
    inputs = {}
    for name in (
        "production_cache",
        "program_config",
        "qualified_reaction_families",
        "qualified_ugi_reactions",
        "ugi_assignments",
        "multireaction_atlas",
        "multireaction_splits",
    ):
        path = root / (name + ".json")
        path.write_text("{}")
        inputs[name] = {"path": path.name, "sha256": str(sha256_file(path))}
    run = root / "run"
    run.mkdir()
    design = run / "study_design.json"
    design.write_text("{}")
    inputs["production_design"] = {"path": "study_design.json", "sha256": str(sha256_file(design))}
    checkpoint = run / "checkpoints.tar"
    checkpoint.write_bytes(b"fixture archive")
    training = run / "training_result.json"
    write_json(
        training,
        {
            "schema_version": "forge.synthesis_program_production_training_result.v1",
            "status": "pass",
            "replicate": 0,
            "checkpoint_archive": artifact_record(checkpoint),
        },
    )
    config = run / "evaluation_config.json"
    write_json(
        config,
        {
            "schema_version": production_evaluation.CONFIG_SCHEMA,
            "inputs": inputs,
            "execution_scope": "production",
            "smoke": {"device": "cpu"},
        },
    )
    return root, {
        "config": config,
        "checkpoint": checkpoint,
        "training_result": training,
        "study_design": design,
    }


def test_custom_cli_resolves_paths_from_release_root(
    custom_run: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root, paths = custom_run
    called = {}

    def run(
        config: Path,
        repo: Path,
        cache: Path,
        checkpoint: Path,
        training: Path,
        output: Path,
        **kwargs: object,
    ) -> dict[str, object]:
        called.update(
            config=config,
            repo=repo,
            cache=cache,
            checkpoint=checkpoint,
            training=training,
            **kwargs,
        )
        write_json(output / "result.json", {"status": "pass"})
        return {"status": "pass"}

    monkeypatch.setattr(production_evaluation, "run_synthesis_program_production_evaluation", run)
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "evaluated"
    arguments = [
        "evaluate",
        "--root",
        str(root),
        "--profile",
        "smoke",
        "--device",
        "cpu",
        "--output",
        str(output),
    ]
    for name, path in paths.items():
        arguments.extend(["--" + name.replace("_", "-"), str(path.relative_to(root))])
    assert main(arguments) == 0
    assert called["config"] == paths["config"]
    assert called["checkpoint"] == paths["checkpoint"]
    assert called["dynamic_production_design_path"] == paths["study_design"]
    assert called["profile"] == "smoke"
    assert json.loads((output / "result.json").read_text())["status"] == "pass"


@pytest.mark.parametrize("missing", ["config", "checkpoint", "training_result", "study_design"])
def test_custom_input_set_cannot_be_partial(custom_run: tuple[Path, dict[str, Path]], missing: str):
    root, paths = custom_run
    paths = {name: path for name, path in paths.items() if name != missing}
    with pytest.raises(ValueError, match="together"):
        evaluate(root, root / "output", replicate=0, device="cpu", profile="smoke", **paths)
    assert not (root / "output").exists()


@pytest.mark.parametrize("changed", ["checkpoint", "cache"])
def test_changed_custom_input_is_rejected(custom_run: tuple[Path, dict[str, Path]], changed: str):
    root, paths = custom_run
    path = paths["checkpoint"] if changed == "checkpoint" else root / "production_cache.json"
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs|changed"):
        evaluate(root, root / "output", replicate=0, device="cpu", profile="smoke", **paths)
    assert not (root / "output").exists()


@pytest.mark.parametrize("wrong", ["replicate", "design", "device"])
def test_kernel_authentication_remains_active(custom_run: tuple[Path, dict[str, Path]], wrong: str):
    root, paths = custom_run
    if wrong == "design":
        paths["study_design"].write_text('{"changed": true}')
    if wrong == "device":
        config = json.loads(paths["config"].read_text())
        config["smoke"]["device"] = "cuda"
        write_json(paths["config"], config)
    with pytest.raises(production_evaluation.SynthesisProgramProductionEvaluationError):
        evaluate(
            root,
            root / "output",
            replicate=1 if wrong == "replicate" else 0,
            device="cpu",
            profile="smoke",
            **paths,
        )
    assert not (root / "output").exists()
    assert (root / "output.failed/FAILED.json").is_file()


def test_smoke_does_not_silently_use_cuda(custom_run: tuple[Path, dict[str, Path]]):
    root, paths = custom_run
    with pytest.raises(ValueError, match="--device cpu"):
        evaluate(root, root / "output", replicate=0, device="cuda", profile="smoke", **paths)


def test_existing_released_checkpoint_evaluation_keeps_full_profile(
    custom_run: tuple[Path, dict[str, Path]],
    monkeypatch: pytest.MonkeyPatch,
):
    root, paths = custom_run
    from forge.commands import generate

    monkeypatch.setattr(
        generate,
        "check_inputs",
        lambda root, replicate: {
            "ready": True,
            "inputs": {
                key: {"path": str(path)}
                for key, path in {
                    "production_cache": root / "production_cache.json",
                    "checkpoint": paths["checkpoint"],
                    "training_result": paths["training_result"],
                }.items()
            },
        },
    )
    called = {}

    def run(*args, **kwargs):
        called.update(kwargs)
        return {"status": "pass"}

    monkeypatch.setattr(production_evaluation, "run_synthesis_program_production_evaluation", run)
    evaluate(root, root / "output", replicate=0, device="cuda")
    assert called["profile"] == "full"
    assert called["dynamic_production_design_path"] is None


@pytest.mark.parametrize("steps", [[3], [2, 1], [1, 1]])
def test_checkpoint_selection_rejects_missing_or_invalid_schedule(steps):
    with pytest.raises(production_evaluation.SynthesisProgramProductionEvaluationError):
        production_evaluation._select_evaluation_snapshots(
            [{"step": 1}, {"step": 2}],
            steps,
            arm_id="fixture",
        )


def test_native_request_command_uses_public_cli_from_other_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    export = tmp_path / "common"
    export.mkdir()
    for filename in native_baseline_ports._COMMON_FILES.values():
        path = export / filename
        if filename in ("train.csv.gz", "calibration.csv.gz", "heldout.csv.gz"):
            write_csv(
                path,
                [
                    {
                        "product_id": "p",
                        "canonical_product_smiles": "CCO",
                        "family_balance_weight_raw": "1",
                    }
                ],
                ["product_id", "canonical_product_smiles", "family_balance_weight_raw"],
            )
        else:
            path.write_bytes(b"fixture")
    write_json(
        export / "result.json",
        {
            "schema_version": "forge.external_ugi_common_input_export.v1",
            "artifacts": {
                key: artifact_record(export / value)
                for key, value in native_baseline_ports._COMMON_FILES.items()
            },
            "sampling_measure": "fixture",
        },
    )
    monkeypatch.setattr(
        native_baseline_ports,
        "verify_external_checkout",
        lambda method, path: {"commit": method["commit"]},
    )
    output = tmp_path / "request"
    native_baseline_ports.prepare_native_baseline_run(
        ROOT / "configs/baselines/external_ugi_v1.json",
        export,
        tmp_path / "upstream",
        output,
        method_id="rgfn",
        seed=0,
        attempts=2,
        profile="smoke",
    )
    command = json.loads((output / "run_command.json").read_text())["argv"]
    monkeypatch.chdir(tmp_path)
    from forge.baselines import runtime as native_baseline_runtime

    captured = {}

    def run(request, checkout, output, **kwargs):
        captured.update(request=request, checkout=checkout)

    monkeypatch.setattr(native_baseline_runtime, "run_native_baseline", run)
    assert main(command[1:]) == 0
    assert captured["request"] == output / "request.json"
    assert captured["checkout"] == tmp_path / "upstream"
