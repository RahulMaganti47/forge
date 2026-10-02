"""Freeze reproducible native-run packages for the retained external Ugi baselines.

The package builder runs in the FORGE environment and never imports an upstream project.  Native
training runs in that project's own environment through :mod:`native_baseline_runtime`.  This
separation keeps dependency conflicts out of FORGE while retaining exact inputs, commits, seeds,
budgets and output semantics in one request receipt.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.baselines.contract import (
    ExternalBaselineError,
    load_external_baseline_manifest,
    verify_external_checkout,
)
from forge.core.hashing import artifact_record, sha256_file
from forge.core.io import read_csv, read_json_object, write_json

REQUEST_SCHEMA = "forge.external_ugi_native_request.v1"
READY_METHODS = ("rgfn", "defog_unconditional", "genmol_safe")
TOKENIZER_REVISION = "34bc4afa8705fe7c3a6193e2a1174961f10686bf"

_COMMON_FILES = {
    "train": "train.csv.gz",
    "calibration": "calibration.csv.gz",
    "heldout": "heldout.csv.gz",
    "train_components": "train_components.csv.gz",
    "ugi_reaction": "ugi_reaction.json",
}

_PROFILES: dict[str, dict[str, dict[str, Any]]] = {
    "smoke": {
        "rgfn": {"training_iterations": 2, "trajectories_per_iteration": 4, "batch_size": 4},
        "defog_unconditional": {
            "training_epochs": 1,
            "training_batches": 2,
            "training_steps": 2,
            "batch_size": 4,
            "validation_batches": 0,
            "test_batches": 1,
            "sampling_steps": 4,
            "sampling_batch_size": 2,
        },
        "genmol_safe": {
            "training_steps": 2,
            "global_batch_size": 4,
            "sampling_batch_size": 4,
        },
    },
    "full": {
        "rgfn": {
            "training_iterations": 5002,
            "trajectories_per_iteration": 100,
            "batch_size": 100,
        },
        "defog_unconditional": {
            "training_epochs": 1000,
            "training_steps": 50000,
            "batch_size": 16,
            "validation_batches": 0,
            "test_batches": 1,
            "sampling_steps": 1000,
            "sampling_batch_size": 16,
        },
        "genmol_safe": {
            "training_steps": 50000,
            "global_batch_size": 128,
            "sampling_batch_size": 32,
        },
    },
}


class NativeBaselinePortError(ExternalBaselineError):
    """A native run package is incomplete, drifted or scientifically inadmissible."""


def _verify_common_export(export_dir: Path) -> dict[str, Any]:
    result = read_json_object(
        export_dir / "result.json",
        error=NativeBaselinePortError,
        label="external Ugi common export result",
    )
    if result.get("schema_version") != "forge.external_ugi_common_input_export.v1":
        raise NativeBaselinePortError("unsupported external Ugi common export")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(_COMMON_FILES):
        raise NativeBaselinePortError("external Ugi common export artifacts changed")
    for key, filename in _COMMON_FILES.items():
        path = export_dir / filename
        record = artifacts[key]
        if (
            not path.is_file()
            or not isinstance(record, Mapping)
            or record.get("logical_path") != filename
            or record.get("sha256") != str(sha256_file(path))
            or record.get("bytes") != path.stat().st_size
        ):
            raise NativeBaselinePortError(f"common export artifact changed: {key}")
    return result


def _write_smiles_csv_as_lines(source: Path, destination: Path) -> None:
    rows = read_csv(source)
    required = {"product_id", "canonical_product_smiles", "family_balance_weight_raw"}
    if not rows or any(set(row) != required for row in rows):
        raise NativeBaselinePortError(f"common product export schema changed: {source.name}")
    payload = "".join(f"{row['canonical_product_smiles']}\n" for row in rows).encode()
    from forge.core.io import atomic_write

    atomic_write(destination, payload)


def prepare_native_baseline_run(
    manifest_path: Path,
    common_export_dir: Path,
    checkout: Path,
    output_dir: Path,
    *,
    method_id: str,
    seed: int,
    attempts: int,
    profile: str,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Create one immutable native-method request without invoking third-party code."""

    if profile not in _PROFILES:
        raise NativeBaselinePortError(f"unsupported native profile: {profile}")
    repo = (repo or manifest_path.resolve().parents[2]).resolve()
    if not (repo / "manifests/paper-model-v1.json").is_file():
        raise NativeBaselinePortError("repo must name a FORGE checkout")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise NativeBaselinePortError("native seed must be a non-negative integer")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise NativeBaselinePortError("native attempts must be a positive integer")
    methods = load_external_baseline_manifest(manifest_path)
    method = methods.get(method_id)
    if method is None:
        raise NativeBaselinePortError(f"external method is not declared: {method_id}")
    if method_id not in READY_METHODS or method["integration_status"] != "native_port_ready":
        raise NativeBaselinePortError(
            f"external method is not admitted for native execution: {method_id} "
            f"({method['integration_status']})"
        )
    checkout_receipt = verify_external_checkout(method, checkout)
    common_result = _verify_common_export(common_export_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NativeBaselinePortError(f"native request output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "inputs"
    input_dir.mkdir()
    for filename in _COMMON_FILES.values():
        shutil.copyfile(common_export_dir / filename, input_dir / filename)
    for fold in ("train", "calibration", "heldout"):
        _write_smiles_csv_as_lines(input_dir / _COMMON_FILES[fold], input_dir / f"{fold}.smi")

    method_parameters = dict(_PROFILES[profile][method_id])
    method_parameters.update(
        {
            "device": "cuda",
            "precision": "float32",
            "deterministic_algorithms": True,
            "max_heavy_atoms": 194,
            "checkpoint_selection": "fixed_final_step",
            "repair_or_retry": False,
            "biological_or_property_reward": False,
            "candidate_selection": False,
        }
    )
    if method_id == "rgfn":
        method_parameters.update(
            {
                "max_reactions": 1,
                "reward": "constant_one_no_property_oracle",
                "component_inventory": "common_train_only",
            }
        )
    elif method_id == "defog_unconditional":
        method_parameters.update(
            {
                "dataset_statistics": "recomputed_from_common_train_and_calibration",
                "time_distortion": "identity",
                "eta": 0.0,
                "omega": 0.0,
                "direct_graph_decode": True,
            }
        )
    else:
        method_parameters.update(
            {
                "pretrained_weights": False,
                "safe_tokenizer_repository": "datamol-io/safe-gpt",
                "safe_tokenizer_revision": TOKENIZER_REVISION,
                "safe_decode_ignore_errors": False,
                "safe_fix": False,
            }
        )

    inputs = {
        filename: artifact_record(input_dir / filename, logical_path=f"inputs/{filename}")
        for filename in sorted(path.name for path in input_dir.iterdir())
    }
    request = {
        "schema_version": REQUEST_SCHEMA,
        "method": {
            "method_id": method_id,
            "repository": method["repository"],
            "commit": method["commit"],
            "license": method["license"],
        },
        "checkout": checkout_receipt,
        "profile": profile,
        "seed": seed,
        "requested_attempts": attempts,
        "parameters": method_parameters,
        "inputs": inputs,
        "common_export_result": artifact_record(
            common_export_dir / "result.json", logical_path="common_export/result.json"
        ),
        "common_export_sampling_measure": common_result["sampling_measure"],
        "output_contract": method["output_contract"],
        "call_budget": {
            "generator_calls": attempts,
            "reaction_calls": "measured_native_count",
            "route_calls": 0,
            "oracle_calls": 0,
        },
        "nonclaims": [
            "A prepared request is not a completed native run.",
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
            "Native invalid and failed attempts remain in the requested-attempt denominator.",
        ],
    }
    request_path = output_dir / "request.json"
    write_json(request_path, request)
    run_command = {
        "schema_version": "forge.external_ugi_native_command.v1",
        "argv": [
            "forge",
            "baseline",
            "native",
            "--root",
            str(repo),
            "--request",
            str(request_path.resolve()),
            "--checkout",
            str(checkout.resolve()),
            "--output",
            str((output_dir / "native").resolve()),
        ],
        "environment": method["native_environment"],
        "run_from_native_environment": True,
    }
    write_json(output_dir / "run_command.json", run_command)
    result = {
        "schema_version": "forge.external_ugi_native_preparation.v1",
        "status": "ready",
        "method_id": method_id,
        "seed": seed,
        "profile": profile,
        "requested_attempts": attempts,
        "request": artifact_record(request_path),
        "run_command": artifact_record(output_dir / "run_command.json"),
        "inputs": inputs,
        "checkout": checkout_receipt,
        "candidate_selection": False,
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "NativeBaselinePortError",
    "READY_METHODS",
    "REQUEST_SCHEMA",
    "TOKENIZER_REVISION",
    "prepare_native_baseline_run",
]
