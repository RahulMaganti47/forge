"""Prepare one hash-pinned tensor cache shared by matched Ugi training arms."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from forge.corpus.training_cache import (
    UgiTrainingCacheError,
    load_ugi_training_cache,
    load_ugi_training_cache_payload,
)
from forge.corpus.ugi_chemistry_corpus import load_expanded_ugi_chemistry_corpus
from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_joint_sparse_flow import project_joint_sparse_record

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


def _resolve(repo: Path, record: dict[str, str], label: str) -> tuple[Path, dict[str, str]]:
    path = Path(record["path"])
    if not path.is_absolute():
        path = repo / path
    if not path.is_file():
        raise UgiTrainingCacheError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != record["sha256"]:
        raise UgiTrainingCacheError(f"{label} hash changed")
    return path, {"path": str(path), "sha256": observed}


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def prepare_ugi_training_cache(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    reuse_existing: bool = False,
) -> dict[str, Any]:
    """Project the exact corpus once for reuse by staged and joint arms."""

    if torch is None:
        raise UgiTrainingCacheError("training-cache preparation requires torch")
    config = json.loads(config_path.read_text())
    if config.get("schema_version") != "phase1_ugi_training_cache_config.v1":
        raise UgiTrainingCacheError("unsupported training-cache config")
    if not output_dir.is_absolute():
        output_dir = repo / output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise UgiTrainingCacheError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {}
    inputs = {}
    for label, value in config["inputs"].items():
        resolved[label], inputs[label] = _resolve(repo, value, label)
    cache_path = output_dir / "ugi_training_cache.pt"
    if reuse_existing:
        if not cache_path.is_file():
            raise UgiTrainingCacheError("requested prepared cache does not exist")
        payload = load_ugi_training_cache_payload(cache_path)
        if (
            payload.get("schema_version") != "phase1_ugi_training_cache.v1"
            or payload.get("inputs") != inputs
        ):
            raise UgiTrainingCacheError("existing prepared cache disagrees with inputs")
        corpus = payload["corpus"]
        joint_records_by_fold = payload["joint_records_by_fold"]
        maxima = payload["maxima"]
    else:
        corpus = load_expanded_ugi_chemistry_corpus(
            resolved["assignments"],
            resolved["semantic_products"],
            resolved["semantic_atoms"],
            resolved["atom_vocabulary"],
        )
        joint_records_by_fold = {
            fold: tuple(project_joint_sparse_record(record) for record in records)
            for fold, records in corpus.records_by_fold.items()
        }
        maxima = {
            "total_exterior_nodes": 0,
            "component_exterior_nodes": 0,
            "junction_budget": 0,
            "cycle_rank": 0,
            "attachment_count": 0,
            "decorations": 0,
            "children": 0,
        }
    decoration_histogram: dict[int, int] = {}
    for records in joint_records_by_fold.values():
        for record in records:
            if not reuse_existing:
                maxima["total_exterior_nodes"] = max(
                    maxima["total_exterior_nodes"], record.node_count
                )
                maxima["component_exterior_nodes"] = max(
                    maxima["component_exterior_nodes"], *record.program.node_counts
                )
                maxima["junction_budget"] = max(
                    maxima["junction_budget"], *record.program.junction_budgets
                )
                maxima["cycle_rank"] = max(maxima["cycle_rank"], *record.program.cycle_ranks)
                maxima["attachment_count"] = max(
                    maxima["attachment_count"], *record.program.attachment_counts
                )
            decoration_count = int(record.decoration_anchors.size)
            if not reuse_existing:
                maxima["decorations"] = max(maxima["decorations"], decoration_count)
            decoration_histogram[decoration_count] = (
                decoration_histogram.get(decoration_count, 0) + 1
            )
            if not reuse_existing and record.offspring.size:
                maxima["children"] = max(maxima["children"], int(record.offspring.max()))
    expected_counts = {fold: len(records) for fold, records in corpus.records_by_fold.items()}
    if expected_counts != config["expected_fold_counts"]:
        raise UgiTrainingCacheError(f"training-cache fold counts changed: {expected_counts}")
    if not reuse_existing:
        payload = {
            "schema_version": "phase1_ugi_training_cache.v1",
            "inputs": inputs,
            "corpus": corpus,
            "joint_records_by_fold": joint_records_by_fold,
            "maxima": maxima,
        }
        _atomic_torch_save(cache_path, payload)
    result = {
        "schema_version": "phase1_ugi_training_cache_result.v1",
        "status": "pass",
        "inputs": inputs,
        "fold_counts": expected_counts,
        "maxima": maxima,
        "decoration_histogram": {
            str(key): value for key, value in sorted(decoration_histogram.items())
        },
        "artifact": {
            "path": str(cache_path.relative_to(repo)),
            "sha256": sha256_file(cache_path),
            "bytes": cache_path.stat().st_size,
        },
    }
    _atomic_json(output_dir / "result.json", result)
    return result


__all__ = ["UgiTrainingCacheError", "load_ugi_training_cache", "prepare_ugi_training_cache"]
