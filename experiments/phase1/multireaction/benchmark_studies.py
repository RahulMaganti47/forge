"""Executable internal studies built on the common Ugi attempt/evidence contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json, write_jsonl
from forge.model.common_ugi_benchmark import (
    assess_common_ugi_attempts,
    write_attempt_ledger,
)
from forge.model.conditional_role_dependence import run_conditional_role_dependence
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.learned_inventory_selector import (
    load_inventory_training_data,
    sample_inventory_selector,
    train_inventory_selector,
)
from forge.model.training_restart import atomic_torch_save
from forge.synthesis.assessment.common_route_evidence import (
    assess_common_route_evidence,
    load_frozen_component_evidence,
)

CONDITIONAL_CONFIG_SCHEMA = "forge.conditional_role_dependence_config.v1"
INVENTORY_CONFIG_SCHEMA = "forge.learned_inventory_selector_config.v1"


class BenchmarkStudyError(ValueError):
    """An internal benchmark study violates its frozen protocol."""


def run_conditional_dependence_study(
    config_path: Path, repo: Path, output_path: Path
) -> dict[str, Any]:
    config = read_json_object(
        config_path, error=BenchmarkStudyError, label="conditional dependence config"
    )
    if config.get("schema_version") != CONDITIONAL_CONFIG_SCHEMA:
        raise BenchmarkStudyError("unsupported conditional dependence config")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"ugi_assignments"}:
        raise BenchmarkStudyError("conditional dependence inputs changed")
    assignments = resolve_pin(inputs["ugi_assignments"], repo, label="ugi_assignments")
    result = run_conditional_role_dependence(
        assignments,
        roles=tuple(config["roles"]),
        seed=int(config["seed"]),
        folds=int(config["classifier_folds"]),
    )
    result.update(
        {
            "config": pin_record(config_path, repo),
            "inputs": {"ugi_assignments": pin_record(assignments, repo)},
            "candidate_selection": False,
            "calls": {"route": 0, "oracle": 0},
        }
    )
    write_json(output_path, result)
    return result


def run_learned_inventory_selector_study(
    config_path: Path,
    repo: Path,
    output_dir: Path,
    *,
    profile: str,
    replicate: int,
    allocated_device: str,
) -> dict[str, Any]:
    import torch

    config = read_json_object(
        config_path, error=BenchmarkStudyError, label="learned inventory selector config"
    )
    if config.get("schema_version") != INVENTORY_CONFIG_SCHEMA:
        raise BenchmarkStudyError("unsupported learned inventory selector config")
    inputs = config.get("inputs")
    required = {
        "common_protocol",
        "ugi_assignments",
        "qualified_ugi_reactions",
        "component_route_ledger",
    }
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise BenchmarkStudyError("learned inventory inputs changed")
    paths = {key: resolve_pin(value, repo, label=key) for key, value in inputs.items()}
    protocol = read_json_object(
        paths["common_protocol"], error=BenchmarkStudyError, label="common Ugi protocol"
    )
    seeds = [int(value) for value in protocol["randomness"]["training_and_sampling_seeds"]]
    if replicate < 0 or replicate >= len(seeds):
        raise BenchmarkStudyError("replicate lies outside the common seed set")
    if profile not in {"smoke", "full"}:
        raise BenchmarkStudyError(f"unsupported learned inventory profile: {profile}")
    runtime = dict(config[profile])
    if runtime["device"] != allocated_device:
        raise BenchmarkStudyError("allocated device differs from learned inventory config")
    expected_attempts = (
        int(protocol["sampling"]["attempts_per_seed"])
        if profile == "full"
        else int(runtime["attempts"])
    )
    if profile == "full" and int(runtime["attempts"]) != expected_attempts:
        raise BenchmarkStudyError("full attempt budget differs from the common protocol")
    adapter = Ugi3AssemblyAdapter.from_registry(
        paths["qualified_ugi_reactions"],
        expected_sha256=str(inputs["qualified_ugi_reactions"]["sha256"]),
    )
    data = load_inventory_training_data(paths["ugi_assignments"], roles=adapter.roles)
    seed = seeds[replicate]
    model, training = train_inventory_selector(
        data,
        seed=seed,
        steps=int(runtime["optimizer_steps"]),
        batch_size=int(runtime["batch_size"]),
        hidden_dim=int(config["model"]["hidden_dim"]),
        embedding_dim=int(config["model"]["embedding_dim"]),
        learning_rate=float(runtime["learning_rate"]),
        device=allocated_device,
    )
    attempts = sample_inventory_selector(
        model,
        data,
        adapter,
        method_id="learned_inventory_selector",
        seed=seed,
        attempts=expected_attempts,
        maximum_forward_outcomes=int(config["maximum_forward_outcomes"]),
        device=allocated_device,
    )
    assessed, common = assess_common_ugi_attempts(
        attempts,
        adapter=adapter,
        assignments_path=paths["ugi_assignments"],
    )
    route_index = load_frozen_component_evidence(paths["component_route_ledger"])
    route_rows, route = assess_common_route_evidence(assessed, component_evidence=route_index)
    output_dir.mkdir(parents=True, exist_ok=True)
    attempts_path = output_dir / "attempts.jsonl.gz"
    assessed_path = output_dir / "assessed_attempts.jsonl.gz"
    route_path = output_dir / "route_assessed_attempts.jsonl.gz"
    checkpoint_path = output_dir / "checkpoint.pt"
    write_attempt_ledger(attempts_path, attempts)
    write_jsonl(
        assessed_path,
        [
            {"schema_version": "forge.common_ugi_assessed_attempts.v1", "rows": len(assessed)},
            *assessed,
        ],
    )
    write_jsonl(
        route_path,
        [
            {
                "schema_version": "forge.common_ugi_route_assessed_attempts.v1",
                "rows": len(route_rows),
            },
            *route_rows,
        ],
    )
    package = {
        "schema_version": "forge.learned_inventory_selector_checkpoint.v1",
        "seed": seed,
        "roles": list(data.roles),
        "inventories": [list(values) for values in data.inventories],
        "model": dict(config["model"]),
        "model_state": model.state_dict(),
        "model_state_sha256": _model_state_sha256(model),
        "trusted_local_checkpoint": True,
    }
    atomic_torch_save(checkpoint_path, package)
    del model
    if torch.cuda.is_available() and allocated_device == "cuda":
        torch.cuda.empty_cache()
    gates = {
        "attempt_denominator_exact": len(attempts) == expected_attempts,
        "finite_inventory_escape_zero_by_construction": True,
        "source_balanced_training_measure": training["sampling_measure"]
        == "family_balance_weight_raw_normalized_within_train_fold",
        "coverage_and_precision_reported": common["coverage_and_precision_reported"] is True,
        "route_or_oracle_calls_zero_during_generation": all(
            attempt.route_calls == attempt.oracle_calls == 0 for attempt in attempts
        ),
        "candidate_selection_absent": True,
    }
    result = {
        "schema_version": "forge.learned_inventory_selector_result.v1",
        "status": "pass",
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "training": training,
        "inventory": {
            role: len(values) for role, values in zip(data.roles, data.inventories, strict=True)
        },
        "attempts": artifact_record(attempts_path),
        "assessed_attempts": artifact_record(assessed_path),
        "route_assessed_attempts": artifact_record(route_path),
        "checkpoint": artifact_record(checkpoint_path),
        "common_assessment": common,
        "route_evidence_assessment": route,
        "wall_seconds_measurement": "not_recorded_do_not_interpret_zero",
        "inputs": {key: pin_record(path, repo) for key, path in sorted(paths.items())},
        "config": pin_record(config_path, repo),
        "gates": gates,
        "candidate_selection": False,
        "nonclaims": [
            "The learned selector cannot generate a component outside its train inventory.",
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
        ],
    }
    if not all(gates.values()):
        result["status"] = "fail"
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise BenchmarkStudyError(f"learned inventory gates failed: {result['gates']}")
    return result


__all__ = [
    "BenchmarkStudyError",
    "CONDITIONAL_CONFIG_SCHEMA",
    "INVENTORY_CONFIG_SCHEMA",
    "run_conditional_dependence_study",
    "run_learned_inventory_selector_study",
]
