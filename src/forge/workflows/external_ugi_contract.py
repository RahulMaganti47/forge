"""Fail-closed experiment interoperability for third-party molecular generators.

FORGE never treats an upstream repository name as an implementation.  An external row requires an
immutable checkout, a common-split export, and one receipt row per requested native attempt.  The
importer then converts that receipt to the same ledger used by every internal method.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, sha256_file
from forge.core.io import iter_csv, read_csv, read_json, read_json_object, write_csv, write_json
from forge.evaluation.ugi_benchmark import (
    CommonUgiAttempt,
    validate_attempt_ledger,
    write_attempt_ledger,
)

MANIFEST_SCHEMA = "forge.external_ugi_baseline_manifest.v1"
METHOD_FIELDS = {
    "method_id",
    "display_name",
    "comparison_class",
    "repository",
    "commit",
    "license",
    "license_status",
    "integration_status",
    "native_environment",
    "required_port",
    "finite_component_vocabulary",
    "output_contract",
}


class ExternalBaselineError(ValueError):
    """An external checkout, receipt, or native output is not common-benchmark admissible."""


def load_external_baseline_manifest(path: Path) -> dict[str, dict[str, Any]]:
    value = read_json_object(path, error=ExternalBaselineError, label="external baseline manifest")
    if set(value) != {"schema_version", "as_of", "methods", "nonclaims"}:
        raise ExternalBaselineError("external baseline manifest fields changed")
    if value["schema_version"] != MANIFEST_SCHEMA or not isinstance(value["methods"], list):
        raise ExternalBaselineError("unsupported external baseline manifest")
    methods: dict[str, dict[str, Any]] = {}
    for raw in value["methods"]:
        if not isinstance(raw, dict) or set(raw) != METHOD_FIELDS:
            raise ExternalBaselineError(
                f"external method must define exactly {sorted(METHOD_FIELDS)}"
            )
        method_id = raw["method_id"]
        commit = raw["commit"]
        if (
            not isinstance(method_id, str)
            or not method_id
            or method_id in methods
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
        ):
            raise ExternalBaselineError("external method id or commit is invalid")
        if raw["integration_status"] not in {
            "native_port_ready",
            "excluded_no_author_implementation",
            "excluded_incompatible_reaction_arity",
            "excluded_unlicensed",
        }:
            raise ExternalBaselineError(f"unsupported integration status for {method_id}")
        contract = raw["output_contract"]
        if contract != {
            "format": "csv",
            "required_columns": ["attempt_index", "status", "product_smiles"],
            "one_row_per_requested_attempt": True,
            "success_only_output_forbidden": True,
        }:
            raise ExternalBaselineError(f"output contract changed for {method_id}")
        methods[method_id] = dict(raw)
    return methods


def verify_external_checkout(method: Mapping[str, Any], checkout: Path) -> dict[str, Any]:
    """Require the exact clean upstream commit before a method may run."""

    if not (checkout / ".git").is_dir():
        raise ExternalBaselineError(f"external checkout is missing: {checkout}")
    try:
        commit = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(checkout), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExternalBaselineError(f"cannot inspect external checkout: {checkout}") from error
    if commit != method["commit"]:
        raise ExternalBaselineError(
            f"external checkout commit changed: expected {method['commit']}, found {commit}"
        )
    if dirty:
        raise ExternalBaselineError(
            "upstream checkout contains unreviewed modifications; method ports belong in FORGE adapters"
        )
    return {
        "method_id": method["method_id"],
        "repository": method["repository"],
        "commit": commit,
        "clean": True,
    }


def load_visible_component_ids(path: Path) -> tuple[str, ...]:
    """Load the exact generation-visible role/component inventory from the common export."""

    rows = read_csv(path)
    required = {"role", "canonical_component_smiles", "component_family_id"}
    if not rows or any(set(row) != required for row in rows):
        raise ExternalBaselineError("common train-component export schema changed")
    identities = {f"{row['role']}:{row['canonical_component_smiles']}" for row in rows}
    if len(identities) != len(rows):
        raise ExternalBaselineError("common train-component export contains duplicates")
    return tuple(sorted(identities))


def export_common_ugi_inputs(
    assignments_path: Path,
    reaction_registry_path: Path,
    output_dir: Path,
    *,
    expected_reaction_registry_sha256: str | None = None,
) -> dict[str, Any]:
    """Export identical constitutional products and train-only components for native ports."""

    fold_rows: dict[str, list[dict[str, str]]] = {
        "train": [],
        "calibration": [],
        "heldout": [],
    }
    components: dict[tuple[str, str], dict[str, str]] = {}
    required = {
        "product_id",
        "canonical_product_smiles",
        "primary_product_fold",
        "family_balance_weight_raw",
    }
    roles = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
    for row in iter_csv(assignments_path):
        if not required.issubset(row) or row["primary_product_fold"] not in fold_rows:
            raise ExternalBaselineError("Ugi common-split source schema changed")
        fold = row["primary_product_fold"]
        fold_rows[fold].append(
            {
                "product_id": row["product_id"],
                "canonical_product_smiles": row["canonical_product_smiles"],
                "family_balance_weight_raw": row["family_balance_weight_raw"],
            }
        )
        if fold == "train":
            for role in roles:
                key = (role, row[f"{role}_smiles"])
                components[key] = {
                    "role": role,
                    "canonical_component_smiles": row[f"{role}_smiles"],
                    "component_family_id": row[f"{role}_family_id"],
                }
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for fold, rows in fold_rows.items():
        path = output_dir / f"{fold}.csv.gz"
        write_csv(
            path,
            rows,
            ["product_id", "canonical_product_smiles", "family_balance_weight_raw"],
        )
        artifacts[fold] = artifact_record(path)
    components_path = output_dir / "train_components.csv.gz"
    write_csv(
        components_path,
        [components[key] for key in sorted(components)],
        ["role", "canonical_component_smiles", "component_family_id"],
    )
    artifacts["train_components"] = artifact_record(components_path)
    adapter = Ugi3AssemblyAdapter.from_registry(
        reaction_registry_path, expected_sha256=expected_reaction_registry_sha256
    )
    registry = read_json(reaction_registry_path)
    if not isinstance(registry, Mapping) or not isinstance(registry.get("reactions"), list):
        raise ExternalBaselineError("qualified Ugi reaction registry schema changed")
    matched = [
        row
        for row in registry["reactions"]
        if isinstance(row, Mapping) and row.get("reaction_id") == adapter.reaction_id
    ]
    if len(matched) != 1:
        raise ExternalBaselineError("qualified Ugi reaction could not be isolated for export")
    reaction_path = output_dir / "ugi_reaction.json"
    write_json(
        reaction_path,
        {
            "schema_version": "forge.external_ugi_reaction_export.v1",
            "registry_version": registry.get("registry_version"),
            "reaction": matched[0],
        },
    )
    artifacts["ugi_reaction"] = artifact_record(reaction_path)
    result = {
        "schema_version": "forge.external_ugi_common_input_export.v1",
        "source": artifact_record(assignments_path),
        "reaction_registry": artifact_record(reaction_registry_path),
        "fold_counts": {fold: len(rows) for fold, rows in fold_rows.items()},
        "train_component_counts": {
            role: sum(key[0] == role for key in components) for role in roles
        },
        "artifacts": artifacts,
        "sampling_measure": "family_balance_weight_raw_not_raw_reaction_family_counts",
    }
    write_json(output_dir / "result.json", result)
    return result


def import_external_attempts(
    method: Mapping[str, Any],
    receipt_path: Path,
    samples_path: Path,
    *,
    expected_seed: int,
    expected_attempts: int,
    method_visible_component_ids: tuple[str, ...] = (),
) -> tuple[CommonUgiAttempt, ...]:
    """Convert a native method receipt without forgiving missing or success-only rows."""

    receipt, rows, counts, total_wall = _verify_native_output(
        method,
        receipt_path,
        samples_path,
        expected_seed=expected_seed,
        expected_attempts=expected_attempts,
    )
    del receipt
    if bool(method["finite_component_vocabulary"]) and not method_visible_component_ids:
        raise ExternalBaselineError(
            "finite-vocabulary external methods must disclose every generation-visible component"
        )
    attempts, corrected = _convert_native_rows(
        method,
        rows,
        counts=counts,
        total_wall=total_wall,
        expected_seed=expected_seed,
        expected_attempts=expected_attempts,
        method_visible_component_ids=method_visible_component_ids,
        supersede_empty_generated=False,
    )
    if corrected != 0:
        raise ExternalBaselineError("strict external import unexpectedly changed native labels")
    return attempts


def _verify_native_output(
    method: Mapping[str, Any],
    receipt_path: Path,
    samples_path: Path,
    *,
    expected_seed: int,
    expected_attempts: int,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, int], float]:
    """Verify immutable native bytes before strict import or explicit supersession."""

    receipt = read_json_object(
        receipt_path, error=ExternalBaselineError, label="external native run receipt"
    )
    required = {
        "schema_version",
        "method_id",
        "upstream_commit",
        "seed",
        "requested_attempts",
        "repairs_or_retries",
        "generator_calls",
        "reaction_calls",
        "route_calls",
        "oracle_calls",
        "wall_seconds",
        "samples_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != required:
        raise ExternalBaselineError("external native receipt fields changed")
    if (
        receipt["schema_version"] != "forge.external_ugi_native_run_receipt.v1"
        or receipt["method_id"] != method["method_id"]
        or receipt["upstream_commit"] != method["commit"]
        or receipt["seed"] != expected_seed
        or receipt["requested_attempts"] != expected_attempts
        or receipt["repairs_or_retries"] is not False
        or receipt["samples_sha256"] != str(sha256_file(samples_path))
    ):
        raise ExternalBaselineError("external native receipt does not match the benchmark request")
    rows = read_csv(samples_path)
    if len(rows) != expected_attempts or any(
        set(row) != {"attempt_index", "status", "product_smiles"} for row in rows
    ):
        raise ExternalBaselineError(
            "native output must retain exactly one row per requested attempt"
        )
    total_wall = float(receipt["wall_seconds"])
    if not total_wall >= 0.0:
        raise ExternalBaselineError("external wall time is invalid")
    counts = {}
    for field in ("generator_calls", "reaction_calls", "route_calls", "oracle_calls"):
        value = receipt[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ExternalBaselineError(f"external receipt {field} is invalid")
        counts[field] = value
    return receipt, rows, counts, total_wall


def _convert_native_rows(
    method: Mapping[str, Any],
    rows: list[dict[str, str]],
    *,
    counts: Mapping[str, int],
    total_wall: float,
    expected_seed: int,
    expected_attempts: int,
    method_visible_component_ids: tuple[str, ...],
    supersede_empty_generated: bool,
) -> tuple[tuple[CommonUgiAttempt, ...], int]:
    """Convert verified rows, optionally applying the one admitted label-only correction."""

    attempts = []
    corrected = 0
    for row in rows:
        try:
            index = int(row["attempt_index"])
        except (TypeError, ValueError) as error:
            raise ExternalBaselineError("external attempt index is invalid") from error
        status = row["status"]
        product = row["product_smiles"] or None
        if status == "generated" and product is None and supersede_empty_generated:
            status = "invalid"
            corrected += 1
        attempts.append(
            CommonUgiAttempt.from_mapping(
                {
                    "method_id": method["method_id"],
                    "seed": expected_seed,
                    "attempt_index": index,
                    "status": status,
                    "product_smiles": product,
                    "method_visible_component_ids": list(method_visible_component_ids),
                    **{
                        field: counts[field] // expected_attempts
                        + int(index < counts[field] % expected_attempts)
                        for field in (
                            "generator_calls",
                            "reaction_calls",
                            "route_calls",
                            "oracle_calls",
                        )
                    },
                    "wall_seconds": total_wall / expected_attempts,
                }
            )
        )
    checked = validate_attempt_ledger(
        [attempt.to_mapping() for attempt in attempts],
        expected_method=str(method["method_id"]),
        expected_seed=expected_seed,
        expected_attempts=expected_attempts,
    )
    for field, expected in counts.items():
        if sum(getattr(attempt, field) for attempt in checked) != expected:
            raise ExternalBaselineError(f"external receipt {field} was not preserved")
    return checked, corrected


def supersede_genmol_empty_labels(
    method: Mapping[str, Any],
    receipt_path: Path,
    samples_path: Path,
    output_dir: Path,
    *,
    expected_seed: int,
    expected_attempts: int,
) -> dict[str, Any]:
    """Create a self-contained, hash-pinned correction without rerunning generation.

    The original native receipt and samples remain immutable.  Only attempts carrying the
    contradictory pair ``status=generated`` and an empty molecular payload are reclassified as
    invalid.  Every molecular payload, attempt index and compute-accounting field is preserved.
    """

    if method.get("method_id") != "genmol_safe":
        raise ExternalBaselineError("the empty-decode label supersession is GenMol/SAFE-specific")
    receipt, rows, counts, total_wall = _verify_native_output(
        method,
        receipt_path,
        samples_path,
        expected_seed=expected_seed,
        expected_attempts=expected_attempts,
    )
    attempts, corrected = _convert_native_rows(
        method,
        rows,
        counts=counts,
        total_wall=total_wall,
        expected_seed=expected_seed,
        expected_attempts=expected_attempts,
        method_visible_component_ids=(),
        supersede_empty_generated=True,
    )
    if corrected == 0:
        raise ExternalBaselineError("native GenMol output has no empty generated labels to correct")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ExternalBaselineError(f"label-supersession output is not empty: {output_dir}")
    source_dir = output_dir / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_receipt = source_dir / "receipt.json"
    source_samples = source_dir / "samples.csv"
    shutil.copyfile(receipt_path, source_receipt)
    shutil.copyfile(samples_path, source_samples)
    attempts_path = output_dir / "attempts.jsonl.gz"
    write_attempt_ledger(attempts_path, attempts)
    gates = {
        "source_receipt_and_samples_hash_verified": True,
        "attempt_denominator_preserved": len(attempts) == expected_attempts,
        "attempt_indices_preserved": [attempt.attempt_index for attempt in attempts]
        == list(range(expected_attempts)),
        "only_empty_generated_labels_changed": all(
            attempt.product_smiles == (row["product_smiles"] or None)
            and (
                attempt.status == row["status"]
                or (
                    row["status"] == "generated"
                    and row["product_smiles"] == ""
                    and attempt.status == "invalid"
                )
            )
            for attempt, row in zip(attempts, rows, strict=True)
        ),
        "molecular_payloads_preserved": all(
            attempt.product_smiles == (row["product_smiles"] or None)
            for attempt, row in zip(attempts, rows, strict=True)
        ),
        "compute_accounting_preserved": all(
            sum(getattr(attempt, field) for attempt in attempts) == value
            for field, value in counts.items()
        ),
        "generation_not_rerun": True,
        "model_not_retrained": True,
    }
    result = {
        "schema_version": "forge.external_ugi_genmol_label_supersession.v1",
        "status": "pass" if all(value is True for value in gates.values()) else "fail",
        "method_id": method["method_id"],
        "seed": expected_seed,
        "requested_attempts": expected_attempts,
        "corrected_attempts": corrected,
        "correction": "generated_empty_to_invalid_null",
        "source": {
            "receipt": artifact_record(source_receipt, logical_path="source/receipt.json"),
            "samples": artifact_record(source_samples, logical_path="source/samples.csv"),
        },
        "artifacts": {
            "attempts": artifact_record(attempts_path),
        },
        "native_receipt": receipt,
        "gates": gates,
        "nonclaims": [
            "This correction does not retrain the model or rerun molecular generation.",
            "Reclassifying an empty payload as invalid does not repair or create a molecule.",
        ],
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise ExternalBaselineError(f"GenMol label-supersession gates failed: {gates}")
    return result


def write_external_attempt_ledger(
    path: Path, attempts: tuple[CommonUgiAttempt, ...]
) -> dict[str, Any]:
    """Write an imported ledger while keeping the CLI above the scientific library boundary."""

    write_attempt_ledger(path, attempts)
    return artifact_record(path)


__all__ = [
    "ExternalBaselineError",
    "MANIFEST_SCHEMA",
    "export_common_ugi_inputs",
    "import_external_attempts",
    "load_visible_component_ids",
    "load_external_baseline_manifest",
    "supersede_genmol_empty_labels",
    "verify_external_checkout",
    "write_external_attempt_ledger",
]
