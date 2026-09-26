"""Checkpoint-separated sampling gate for the shared Ugi/BL/LX flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.core.hashing import artifact_record, pin_record, sha256_file
from forge.core.io import read_json_object, write_json
from forge.corpus.synthesis_program_training import load_synthesis_program_training_cache
from forge.model.synthesis_program_sampling import (
    load_synthesis_program_checkpoint,
    sample_synthesis_program_products,
)

CONFIG_SCHEMA = "forge.synthesis_program_sampling_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_sampling_result.v1"
SAMPLES_SCHEMA = "forge.synthesis_program_samples.v1"


class SharedSynthesisProgramSamplingError(ValueError):
    """The shared three-program sampler violates its checkpoint contract."""


def run_shared_synthesis_program_sampling(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load an independent checkpoint and sample only from scrubbed semantic layouts."""

    config = read_json_object(
        config_path,
        error=SharedSynthesisProgramSamplingError,
        label="shared synthesis-program sampling config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SharedSynthesisProgramSamplingError("unsupported shared sampling config")
    execution = config.get("execution")
    if not isinstance(execution, dict) or execution.get("precision") != "float32":
        raise SharedSynthesisProgramSamplingError("shared sampling requires explicit float32")
    if execution.get("deterministic_algorithms") is not True:
        raise SharedSynthesisProgramSamplingError("deterministic algorithms must remain enabled")
    cache = load_synthesis_program_training_cache(cache_path)
    model, vocabulary, atom_vocabulary, node_p0, bond_p0, checkpoint = (
        load_synthesis_program_checkpoint(
            checkpoint_path,
            device=str(execution["device"]),
        )
    )
    if vocabulary != cache.vocabulary or atom_vocabulary != cache.atom_vocabulary:
        raise SharedSynthesisProgramSamplingError("checkpoint and cache vocabularies changed")
    if checkpoint.get("cache", {}).get("sha256") != str(sha256_file(cache_path)):
        raise SharedSynthesisProgramSamplingError("checkpoint was trained from a different cache")
    sampling = config["sampling"]
    rows, metrics = sample_synthesis_program_products(
        model,
        cache.records,
        atom_vocabulary,
        node_p0,
        bond_p0,
        samples_per_program=int(sampling["samples_per_program"]),
        sample_steps=int(sampling["steps"]),
        batch_size=int(sampling["batch_size"]),
        seed=int(config["seed"]),
        device=str(execution["device"]),
    )
    samples_path = output_dir / "samples.json"
    write_json(
        samples_path,
        {
            "schema_version": SAMPLES_SCHEMA,
            "seed": int(config["seed"]),
            "cache": artifact_record(cache_path, logical_path="cache/cache.json"),
            "checkpoint": artifact_record(checkpoint_path, logical_path="training/checkpoint.json"),
            "records": rows,
        },
    )
    expected_samples = len(cache.records) * int(sampling["samples_per_program"])
    gates = {
        "expected_balanced_sample_count": metrics["samples"] == expected_samples
        and all(
            row["samples"] == int(sampling["samples_per_program"])
            for row in metrics["by_program"].values()
        ),
        "fixed_states_immutable_through_sampling": metrics["fixed_state_failures"] == 0,
        "minimum_valid_fraction": metrics["valid"] / metrics["samples"]
        >= float(config["gates"]["minimum_valid_fraction"]),
        "minimum_exact_tensor_fraction": metrics["exact_tensor"] / metrics["samples"]
        >= float(config["gates"]["minimum_exact_tensor_fraction"]),
        "every_program_has_exact_graph": all(
            row["exact_target_graph"] >= int(config["gates"]["minimum_exact_graphs_per_program"])
            for row in metrics["by_program"].values()
        ),
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "run_kind": "overfit_gate",
        "seed": int(config["seed"]),
        "config": pin_record(config_path, repo),
        "cache": artifact_record(cache_path, logical_path="cache/cache.json"),
        "checkpoint": artifact_record(checkpoint_path, logical_path="training/checkpoint.json"),
        "samples": artifact_record(samples_path),
        "metrics": metrics,
        "gates": gates,
        "sampling_contract": {
            "layout_visible_state": "program_role_core_position_counts_and_adapter_fixed_states",
            "variable_target_atoms_visible": False,
            "variable_target_bonds_visible": False,
            "component_identifiers_visible": False,
            "fragment_tokens_visible": False,
        },
        "nonclaims": [
            "Samples reuse three overfit semantic layouts and are not generalization evidence.",
            "Exact target recovery is not synthesis-success probability or route certification.",
            "This run does not authorize production multi-reaction sampling.",
        ],
    }
    write_json(output_dir / "result.json", result)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SAMPLES_SCHEMA",
    "SharedSynthesisProgramSamplingError",
    "run_shared_synthesis_program_sampling",
]
