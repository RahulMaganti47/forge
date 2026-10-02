"""Restartable native terminal generation for the matched morphology arms.

This module deliberately contains no oracle, applicability, potency, route or
synthesis evaluator.  The three frozen morphology allocations are advanced by
the same selected-v3 generator with common draw-keyed random seeds.  Terminal
scoring is a separate, downstream read-only experiment.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.io import atomic_write as _atomic_write
from forge.core.io import write_json as _atomic_json
from forge.corpus.r1_prime_audit import sha256_file
from forge.experiments.hela.adapters.selected_v1 import SelectedGuidanceState
from forge.experiments.hela.adapters.selected_v3 import (
    build_selected_model_restartable_guidance_lane_v3,
)
from forge.experiments.hela.adapters.terminal_support import (
    canonical_morphology_program_bytes,
)
from forge.experiments.hela.allocation.ugi_matched_morphology_allocation_schedule import (
    ARM_IDS,
)
from forge.experiments.hela.sampling.terminal_census import (
    _jsonl_gzip_bytes,
    _sha256_payload,
    complete_native_no_route_terminal,
    load_census_contract,
)
from forge.experiments.hela.sampling.ugi_selected_restartable_generator import SAMPLE_STEPS

CONFIG_SCHEMA_VERSION = "phase1_ugi_matched_morphology_terminal_generation_config.v1"
SHARD_SCHEMA_VERSION = "phase1_ugi_matched_morphology_terminal_generation_shard.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_matched_morphology_terminal_generation.v1"
TERMINAL_LEDGER_SCHEMA_VERSION = "forge.ugi_matched_morphology_terminal_ledger.v1"
EXPECTED_SCOPE = {
    "matched_morphology_allocation_only": True,
    "native_selected_v3_terminal_completion": True,
    "common_particle_and_terminal_randomness_across_arms": True,
    "terminal_scoring": False,
    "potency_model_calls": 0,
    "applicability_model_calls": 0,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "nonzero_trajectory_guidance": False,
    "candidate_selection": False,
    "sealed_holdout_access": False,
    "retries_or_repairs": False,
}
EXPECTED_INPUTS = {
    "base_census_config",
    "runner",
    "schedule",
    "schedule_result",
    "source",
    "tests",
}
TERMINAL_LEDGER_REQUIRED_FIELDS = {
    "arm_id",
    "draw_index",
    "common_uniform",
    "support_index",
    "program_sha256",
    "program",
    "selected_probability",
    "broad_prior_probability",
    "support_proposal_probability",
    "potency_proposal_probability",
    "importance_ratio_broad_over_arm",
    "support_score",
    "authorized_morphology",
    "potency_utility",
    "particle_seed",
    "terminal_seed",
    "shard_index",
    "particle_index",
    "native_terminal",
}


class UgiMatchedMorphologyTerminalGenerationError(RuntimeError):
    """Raised when matched terminal generation is not exact and restartable."""


@dataclass(frozen=True)
class MatchedTerminalDesign:
    draws_per_arm: int
    shard_draws: int
    particle_seed_base: int
    terminal_seed_base: int
    device: str
    checkpoint: int

    @property
    def shard_count(self) -> int:
        if self.shard_draws < 1 or self.draws_per_arm % self.shard_draws:
            raise UgiMatchedMorphologyTerminalGenerationError(
                "draws_per_arm must divide exactly into positive shards"
            )
        return self.draws_per_arm // self.shard_draws

    @property
    def terminal_attempts(self) -> int:
        return self.draws_per_arm * len(ARM_IDS)


@dataclass(frozen=True)
class MatchedTerminalContract:
    repo: Path
    config_path: Path
    config_sha256: str
    inputs: Mapping[str, Path]
    design: MatchedTerminalDesign
    schedule: tuple[Mapping[str, Any], ...]
    schedule_sha256: str


NativeCompleter = Callable[..., dict[str, Any]]


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiMatchedMorphologyTerminalGenerationError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiMatchedMorphologyTerminalGenerationError(f"{label} must contain one object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiMatchedMorphologyTerminalGenerationError(f"malformed pin: {label}")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiMatchedMorphologyTerminalGenerationError(
            f"pin escapes repository: {label}"
        ) from error
    if path.is_symlink() or not path.is_file() or sha256_file(path) != record["sha256"]:
        raise UgiMatchedMorphologyTerminalGenerationError(f"pin changed: {label}")
    return path


def matched_draw_seed(base: int, *, purpose: str, draw_index: int) -> int:
    """Return an arm-independent deterministic seed for one draw coordinate."""

    if base < 0 or draw_index < 0 or not purpose:
        raise UgiMatchedMorphologyTerminalGenerationError("invalid matched seed coordinate")
    payload = json.dumps(
        {"base": base, "purpose": purpose, "draw_index": draw_index},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def _design(value: Any) -> MatchedTerminalDesign:
    expected = {
        "draws_per_arm",
        "shard_draws",
        "particle_seed_base",
        "terminal_seed_base",
        "device",
        "sample_steps",
        "checkpoint",
        "terminals_per_draw_per_arm",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UgiMatchedMorphologyTerminalGenerationError("terminal design changed")
    if (
        int(value["draws_per_arm"]) != 3072
        or int(value["sample_steps"]) != SAMPLE_STEPS
        or int(value["checkpoint"]) != 6
        or int(value["terminals_per_draw_per_arm"]) != 1
        or str(value["device"]) != "cpu"
    ):
        raise UgiMatchedMorphologyTerminalGenerationError(
            "terminal design differs from the selected-v3 preregistration"
        )
    design = MatchedTerminalDesign(
        draws_per_arm=int(value["draws_per_arm"]),
        shard_draws=int(value["shard_draws"]),
        particle_seed_base=int(value["particle_seed_base"]),
        terminal_seed_base=int(value["terminal_seed_base"]),
        device=str(value["device"]),
        checkpoint=int(value["checkpoint"]),
    )
    if design.particle_seed_base == design.terminal_seed_base:
        raise UgiMatchedMorphologyTerminalGenerationError("particle and terminal seeds overlap")
    _ = design.shard_count
    return design


def _validate_schedule_records(records: Any, *, draws: int) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(records, list) or len(records) != draws:
        raise UgiMatchedMorphologyTerminalGenerationError("schedule draw count changed")
    for draw_index, row in enumerate(records):
        if (
            not isinstance(row, Mapping)
            or int(row.get("draw_index", -1)) != draw_index
            or not 0.0 <= float(row.get("common_uniform", -1.0)) < 1.0
            or not isinstance(row.get("arms"), Mapping)
            or set(row["arms"]) != set(ARM_IDS)
        ):
            raise UgiMatchedMorphologyTerminalGenerationError("schedule draw or arm order changed")
        for arm in ARM_IDS:
            arm_record = row["arms"][arm]
            required = {
                "support_index",
                "program_sha256",
                "program",
                "selected_probability",
                "broad_prior_probability",
                "support_proposal_probability",
                "potency_proposal_probability",
                "importance_ratio_broad_over_arm",
                "support_score",
                "authorized_morphology",
                "potency_utility",
            }
            if not isinstance(arm_record, Mapping) or set(arm_record) != required:
                raise UgiMatchedMorphologyTerminalGenerationError(
                    f"{arm} schedule row schema changed"
                )
            canonical = canonical_morphology_program_bytes(dict(arm_record["program"]))
            if (
                hashlib.sha256(canonical).hexdigest() != arm_record["program_sha256"]
                or float(arm_record["selected_probability"]) <= 0.0
                or float(arm_record["broad_prior_probability"]) <= 0.0
                or float(arm_record["support_proposal_probability"]) <= 0.0
                or float(arm_record["potency_proposal_probability"]) <= 0.0
            ):
                raise UgiMatchedMorphologyTerminalGenerationError(
                    f"{arm} selected program identity or support changed"
                )
    return tuple(records)


def load_matched_terminal_contract(repo: Path, config_path: Path) -> MatchedTerminalContract:
    repo = repo.resolve()
    config_path = config_path.resolve()
    config = _load_json(config_path, label="matched terminal config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiMatchedMorphologyTerminalGenerationError("unsupported terminal config schema")
    if config.get("scope") != EXPECTED_SCOPE:
        raise UgiMatchedMorphologyTerminalGenerationError("terminal scope changed")
    raw_inputs = config.get("inputs")
    if not isinstance(raw_inputs, Mapping) or set(raw_inputs) != EXPECTED_INPUTS:
        raise UgiMatchedMorphologyTerminalGenerationError("terminal input pins changed")
    paths = {label: _pin(repo, record, label=label) for label, record in raw_inputs.items()}
    design = _design(config["design"])
    schedule_result = _load_json(paths["schedule_result"], label="schedule result")
    schedule = _load_json(paths["schedule"], label="schedule")
    if (
        schedule_result.get("decision", {}).get("schedule_frozen_before_generation") is not True
        or schedule_result.get("decision", {}).get("potency_tilting_promoted") is not False
        or schedule_result.get("artifacts", {}).get("schedule.json", {}).get("schedule_sha256")
        != schedule.get("schedule_sha256")
        or schedule.get("status") != "frozen_before_matched_terminal_generation"
        or schedule.get("design", {}).get("draws_per_arm") != design.draws_per_arm
        or tuple(schedule.get("design", {}).get("arms", ())) != ARM_IDS
        or schedule.get("population", {}).get("qualified_programs") != 57190
        or schedule.get("population", {}).get("all_arms_positive_on_all_programs") is not True
    ):
        raise UgiMatchedMorphologyTerminalGenerationError("frozen schedule identity changed")
    records = _validate_schedule_records(schedule.get("records"), draws=design.draws_per_arm)
    base = load_census_contract(repo, paths["base_census_config"])
    if base.design.sample_steps != SAMPLE_STEPS:
        raise UgiMatchedMorphologyTerminalGenerationError("selected-v3 generator schedule changed")
    return MatchedTerminalContract(
        repo=repo,
        config_path=config_path,
        config_sha256=sha256_file(config_path),
        inputs=paths,
        design=design,
        schedule=records,
        schedule_sha256=str(schedule["schedule_sha256"]),
    )


def matched_terminal_plan(repo: Path, config_path: Path) -> dict[str, Any]:
    contract = load_matched_terminal_contract(repo, config_path)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "planned_not_executed",
        "config_sha256": contract.config_sha256,
        "schedule_sha256": contract.schedule_sha256,
        "arms": list(ARM_IDS),
        "draws_per_arm": contract.design.draws_per_arm,
        "terminal_attempts": contract.design.terminal_attempts,
        "shards": contract.design.shard_count,
        "terminal_ledger_schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
        "terminal_ledger_required_fields": sorted(TERMINAL_LEDGER_REQUIRED_FIELDS),
        "scope": dict(EXPECTED_SCOPE),
    }


def _shard_records(
    contract: MatchedTerminalContract, shard_index: int
) -> tuple[Mapping[str, Any], ...]:
    if not 0 <= shard_index < contract.design.shard_count:
        raise UgiMatchedMorphologyTerminalGenerationError("shard index is outside the design")
    start = shard_index * contract.design.shard_draws
    return contract.schedule[start : start + contract.design.shard_draws]


def _validate_shard(
    contract: MatchedTerminalContract, output_dir: Path, *, shard_index: int
) -> dict[str, Any]:
    shard_dir = output_dir / "shards" / f"shard_{shard_index:04d}"
    receipt = _load_json(shard_dir / "receipt.json", label="matched terminal shard receipt")
    logical = {key: value for key, value in receipt.items() if key != "result_sha256"}
    expected_draws = _shard_records(contract, shard_index)
    expected_indices = [int(row["draw_index"]) for row in expected_draws]
    if (
        receipt.get("result_sha256") != _sha256_payload(logical)
        or receipt.get("schema_version") != SHARD_SCHEMA_VERSION
        or receipt.get("status") != "complete_matched_morphology_terminal_generation_shard"
        or receipt.get("config_sha256") != contract.config_sha256
        or receipt.get("schedule_sha256") != contract.schedule_sha256
        or receipt.get("shard_index") != shard_index
        or receipt.get("draw_indices") != expected_indices
        or receipt.get("arms") != list(ARM_IDS)
        or receipt.get("scope") != EXPECTED_SCOPE
        or receipt.get("counts")
        != {
            "draws": len(expected_draws),
            "terminal_attempts": len(expected_draws) * len(ARM_IDS),
            "terminal_attempts_per_arm": {arm: len(expected_draws) for arm in ARM_IDS},
            "product_transition_calls": len(expected_draws) * len(ARM_IDS) * SAMPLE_STEPS,
        }
    ):
        raise UgiMatchedMorphologyTerminalGenerationError("matched shard contract changed")
    artifact = receipt.get("terminal_ledger")
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "path",
        "sha256",
        "logical_sha256",
        "rows",
        "schema_version",
    }:
        raise UgiMatchedMorphologyTerminalGenerationError("shard terminal artifact changed")
    ledger_path = shard_dir / str(artifact["path"])
    if (
        artifact.get("schema_version") != TERMINAL_LEDGER_SCHEMA_VERSION
        or not ledger_path.is_file()
        or ledger_path.is_symlink()
        or sha256_file(ledger_path) != artifact["sha256"]
        or int(artifact["rows"]) != len(expected_draws) * len(ARM_IDS)
    ):
        raise UgiMatchedMorphologyTerminalGenerationError("shard terminal ledger changed")
    return receipt


def execute_matched_terminal_shard(
    contract: MatchedTerminalContract,
    output_dir: Path,
    *,
    shard_index: int,
    lane: Any,
    native_completer: NativeCompleter = complete_native_no_route_terminal,
) -> dict[str, Any]:
    """Execute or resume one atomic matched draw shard."""

    output_dir = output_dir.resolve()
    final_dir = output_dir / "shards" / f"shard_{shard_index:04d}"
    if final_dir.exists():
        return {
            **_validate_shard(contract, output_dir, shard_index=shard_index),
            "resumed": True,
        }
    draws = _shard_records(contract, shard_index)
    draw_indices = tuple(int(row["draw_index"]) for row in draws)
    particle_seeds = tuple(
        matched_draw_seed(
            contract.design.particle_seed_base,
            purpose="matched_morphology_particle",
            draw_index=draw_index,
        )
        for draw_index in draw_indices
    )
    terminal_seeds = tuple(
        matched_draw_seed(
            contract.design.terminal_seed_base,
            purpose="matched_morphology_terminal",
            draw_index=draw_index,
        )
        for draw_index in draw_indices
    )
    if (
        len(set(particle_seeds)) != len(draws)
        or len(set(terminal_seeds)) != len(draws)
        or set(particle_seeds) & set(terminal_seeds)
    ):
        raise UgiMatchedMorphologyTerminalGenerationError("matched seed collision")

    rows = []
    adapter_identity: str | None = None
    for arm in ARM_IDS:
        arm_programs = tuple(
            canonical_morphology_program_bytes(dict(row["arms"][arm]["program"])) for row in draws
        )
        initialized = lane.initialize(
            arm_programs,
            seed=contract.design.particle_seed_base,
            particle_seeds=particle_seeds,
            device=contract.design.device,
        )
        state = initialized.state
        if not isinstance(state, SelectedGuidanceState):
            raise UgiMatchedMorphologyTerminalGenerationError(
                "selected-v3 initialization returned an invalid state"
            )
        if adapter_identity is None:
            adapter_identity = state.adapter_identity_sha256
        elif adapter_identity != state.adapter_identity_sha256:
            raise UgiMatchedMorphologyTerminalGenerationError(
                "selected-v3 adapter identity changed between arms"
            )
        advanced = lane.advance(state, target_step=contract.design.checkpoint)
        if advanced.product_transition_calls != len(draws) * contract.design.checkpoint:
            raise UgiMatchedMorphologyTerminalGenerationError(
                "selected-v3 transition accounting changed"
            )
        state = advanced.state
        for particle_index, (draw, particle_seed, terminal_seed) in enumerate(
            zip(draws, particle_seeds, terminal_seeds, strict=True)
        ):
            completion = native_completer(
                lane,
                state,
                particle_index=particle_index,
                seed=terminal_seed,
                checkpoint_index=contract.design.checkpoint,
                rollout_index=0,
            )
            arm_record = draw["arms"][arm]
            row = {
                "arm_id": arm,
                "draw_index": int(draw["draw_index"]),
                "common_uniform": float(draw["common_uniform"]),
                "support_index": int(arm_record["support_index"]),
                "program_sha256": str(arm_record["program_sha256"]),
                "program": dict(arm_record["program"]),
                "selected_probability": float(arm_record["selected_probability"]),
                "broad_prior_probability": float(arm_record["broad_prior_probability"]),
                "support_proposal_probability": float(arm_record["support_proposal_probability"]),
                "potency_proposal_probability": float(arm_record["potency_proposal_probability"]),
                "importance_ratio_broad_over_arm": float(
                    arm_record["importance_ratio_broad_over_arm"]
                ),
                "support_score": float(arm_record["support_score"]),
                "authorized_morphology": bool(arm_record["authorized_morphology"]),
                "potency_utility": float(arm_record["potency_utility"]),
                "particle_seed": particle_seed,
                "terminal_seed": terminal_seed,
                "shard_index": shard_index,
                "particle_index": particle_index,
                **completion,
            }
            if not TERMINAL_LEDGER_REQUIRED_FIELDS.issubset(row):
                raise UgiMatchedMorphologyTerminalGenerationError(
                    "native terminal output lacks a required scoring field"
                )
            rows.append(row)
    arm_order = {arm: index for index, arm in enumerate(ARM_IDS)}
    rows.sort(key=lambda row: (int(row["draw_index"]), arm_order[str(row["arm_id"])]))

    shard_root = output_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".shard_{shard_index:04d}.", dir=shard_root))
    try:
        ledger_path = work_dir / "terminals.jsonl.gz"
        _atomic_write(ledger_path, _jsonl_gzip_bytes(rows))
        content = {
            "schema_version": SHARD_SCHEMA_VERSION,
            "status": "complete_matched_morphology_terminal_generation_shard",
            "config_sha256": contract.config_sha256,
            "schedule_sha256": contract.schedule_sha256,
            "shard_index": shard_index,
            "draw_indices": list(draw_indices),
            "arms": list(ARM_IDS),
            "adapter_identity_sha256": adapter_identity,
            "particle_seed_manifest_sha256": _sha256_payload(particle_seeds),
            "terminal_seed_manifest_sha256": _sha256_payload(terminal_seeds),
            "counts": {
                "draws": len(draws),
                "terminal_attempts": len(rows),
                "terminal_attempts_per_arm": {arm: len(draws) for arm in ARM_IDS},
                "product_transition_calls": len(rows) * SAMPLE_STEPS,
            },
            "terminal_ledger": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "logical_sha256": _sha256_payload(rows),
                "rows": len(rows),
                "schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
            },
            "scope": dict(EXPECTED_SCOPE),
        }
        receipt = {**content, "result_sha256": _sha256_payload(content)}
        _atomic_json(work_dir / "receipt.json", receipt)
        os.replace(work_dir, final_dir)
        return {**receipt, "resumed": False}
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise


def run_all_matched_terminal_shards(
    repo: Path, config_path: Path, output_dir: Path
) -> list[dict[str, Any]]:
    contract = load_matched_terminal_contract(repo, config_path)
    lane = build_selected_model_restartable_guidance_lane_v3(contract.repo)
    return [
        execute_matched_terminal_shard(
            contract,
            output_dir,
            shard_index=shard_index,
            lane=lane,
        )
        for shard_index in range(contract.design.shard_count)
    ]


def run_one_matched_terminal_shard(
    repo: Path, config_path: Path, output_dir: Path, *, shard_index: int
) -> dict[str, Any]:
    contract = load_matched_terminal_contract(repo, config_path)
    lane = build_selected_model_restartable_guidance_lane_v3(contract.repo)
    return execute_matched_terminal_shard(
        contract,
        output_dir,
        shard_index=shard_index,
        lane=lane,
    )


def aggregate_matched_terminal_generation(
    repo: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    contract = load_matched_terminal_contract(repo, config_path)
    output_dir = output_dir.resolve()
    receipts = []
    rows: list[dict[str, Any]] = []
    for shard_index in range(contract.design.shard_count):
        receipt = _validate_shard(contract, output_dir, shard_index=shard_index)
        receipts.append(receipt)
        ledger_path = (
            output_dir
            / "shards"
            / f"shard_{shard_index:04d}"
            / str(receipt["terminal_ledger"]["path"])
        )
        with gzip.open(ledger_path, "rt") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    if len(rows) != contract.design.terminal_attempts:
        raise UgiMatchedMorphologyTerminalGenerationError("terminal aggregate is incomplete")
    arm_order = {arm: index for index, arm in enumerate(ARM_IDS)}
    rows.sort(key=lambda row: (int(row["draw_index"]), arm_order[str(row["arm_id"])]))
    by_draw: dict[int, list[dict[str, Any]]] = defaultdict(list)
    arm_counts = defaultdict(int)
    valid_counts = defaultdict(int)
    for row in rows:
        if not TERMINAL_LEDGER_REQUIRED_FIELDS.issubset(row):
            raise UgiMatchedMorphologyTerminalGenerationError("aggregate row lacks required fields")
        by_draw[int(row["draw_index"])].append(row)
        arm = str(row["arm_id"])
        arm_counts[arm] += 1
        valid_counts[arm] += int(bool(row["native_terminal"].get("terminal_valid")))
    if set(by_draw) != set(range(contract.design.draws_per_arm)):
        raise UgiMatchedMorphologyTerminalGenerationError("aggregate draw support changed")
    for draw_index, grouped in by_draw.items():
        if [row["arm_id"] for row in grouped] != list(ARM_IDS):
            raise UgiMatchedMorphologyTerminalGenerationError(
                f"draw {draw_index} lacks one canonical arm"
            )
        if (
            len({float(row["common_uniform"]) for row in grouped}) != 1
            or len({int(row["particle_seed"]) for row in grouped}) != 1
            or len({int(row["terminal_seed"]) for row in grouped}) != 1
        ):
            raise UgiMatchedMorphologyTerminalGenerationError(
                f"draw {draw_index} lost common random numbers"
            )
    if dict(arm_counts) != {arm: contract.design.draws_per_arm for arm in ARM_IDS}:
        raise UgiMatchedMorphologyTerminalGenerationError("arm budgets are not matched")
    adapter_identities = {receipt["adapter_identity_sha256"] for receipt in receipts}
    if len(adapter_identities) != 1:
        raise UgiMatchedMorphologyTerminalGenerationError(
            "selected-v3 adapter changed across shards"
        )

    ledger_path = output_dir / "terminal_ledger.jsonl.gz"
    _atomic_write(ledger_path, _jsonl_gzip_bytes(rows))
    content = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete_matched_morphology_allocation_terminal_generation",
        "config": {
            "path": str(contract.config_path.relative_to(contract.repo)),
            "sha256": contract.config_sha256,
        },
        "schedule_sha256": contract.schedule_sha256,
        "adapter_identity_sha256": next(iter(adapter_identities)),
        "counts": {
            "shards": len(receipts),
            "draws_per_arm": contract.design.draws_per_arm,
            "terminal_attempts": len(rows),
            "terminal_attempts_per_arm": dict(arm_counts),
            "valid_exact_l1_per_arm": dict(valid_counts),
            "product_transition_calls": len(rows) * SAMPLE_STEPS,
        },
        "common_random_numbers": {
            "common_uniforms_across_arms": True,
            "common_particle_seeds_across_arms": True,
            "common_terminal_seeds_across_arms": True,
        },
        "artifacts": {
            "terminal_ledger.jsonl.gz": {
                "path": ledger_path.name,
                "sha256": sha256_file(ledger_path),
                "logical_sha256": _sha256_payload(rows),
                "rows": len(rows),
                "schema_version": TERMINAL_LEDGER_SCHEMA_VERSION,
                "required_fields": sorted(TERMINAL_LEDGER_REQUIRED_FIELDS),
            },
            "shard_receipts_sha256": _sha256_payload(
                [receipt["result_sha256"] for receipt in receipts]
            ),
        },
        "scope": dict(EXPECTED_SCOPE),
        "next_gate": "read_only_frozen_terminal_applicability_and_potency_scoring",
        "nonclaims": [
            "No terminal was scored, ranked, filtered, repaired or retried during generation.",
            "The potency arm changes only the frozen morphology allocation.",
            "Potency tilting remains unpromoted until matched terminal scoring succeeds.",
        ],
    }
    result = {**content, "result_sha256": _sha256_payload(content)}
    _atomic_json(output_dir / "result.json", result)
    return result


__all__ = [
    "MatchedTerminalContract",
    "MatchedTerminalDesign",
    "TERMINAL_LEDGER_REQUIRED_FIELDS",
    "UgiMatchedMorphologyTerminalGenerationError",
    "aggregate_matched_terminal_generation",
    "execute_matched_terminal_shard",
    "load_matched_terminal_contract",
    "matched_draw_seed",
    "matched_terminal_plan",
    "run_all_matched_terminal_shards",
    "run_one_matched_terminal_shard",
]
