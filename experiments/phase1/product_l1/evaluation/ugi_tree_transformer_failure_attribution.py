"""Attribute the frozen three-seed Ugi tree-Transformer production failures."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.hashing import artifact_record, pin_record, resolve_pin
from forge.core.io import read_json_object, write_json, write_jsonl
from forge.model.common_ugi_benchmark import UGI_PROGRAM_ID
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.sparse_topology_feasibility import SPARSE_BOND_TO_INDEX

CONFIG_SCHEMA = "forge.ugi_tree_transformer_failure_attribution_config.v1"
RESULT_SCHEMA = "forge.ugi_tree_transformer_failure_attribution.v1"
ATTEMPT_SCHEMA = "forge.ugi_tree_transformer_failure_attribution_attempt.v1"
EXPERIMENT_ID = "phase1-ugi-tree-relational-production-h100-v1"
SOURCE_SHA256 = "94bb0384b14accf8cd8bce866a3b43d3d1707c471abf6e9b64fb6f035479811c"
EXPECTED_SEEDS = (20260905, 20260906, 20260907)
ATTEMPTS_PER_SEED = 3072
FLOW_SEED_OFFSET = 100000
ROLE_NAMES = ("amine_head", "oxoester_aldehyde_body_tail", "isocyanide_tail")
PRIMARY_CATEGORIES = (
    "invalid_terminal_support",
    "invalid_molecule_sanitization",
    "valid_native_forward_exact_retro_abstention",
    "exact_l1_unsupported_local_chemistry",
    "exact_l1_local_supported",
)

_SAMPLING_MEMBER = "details/sampling/result.json"
_COMMON_MEMBER = "details/assessment/common/assessed_attempts.jsonl.gz"
_LOCAL_MEMBER = "details/assessment/local_chemistry/assessed_attempts.jsonl.gz"
_ROLE_MEMBER = "details/assessment/role_morphology_attempts.jsonl.gz"


class UgiTreeTransformerFailureAttributionError(ValueError):
    """The frozen run evidence cannot support deterministic failure attribution."""


def _member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    try:
        member = archive.getmember(name)
    except KeyError as error:
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive is missing {name!r}"
        ) from error
    if not member.isfile():
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive member is not a regular file: {name}"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive member cannot be read: {name}"
        )
    return handle.read()


def _json_member(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    try:
        value = json.loads(_member_bytes(archive, name))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive member is invalid JSON: {name}"
        ) from error
    if not isinstance(value, dict):
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive JSON is not an object: {name}"
        )
    return value


def _jsonl_member(
    archive: tarfile.TarFile,
    name: str,
    *,
    schema_version: str,
    expected_rows: int,
) -> list[dict[str, Any]]:
    try:
        payload = gzip.decompress(_member_bytes(archive, name)).decode()
        values = [json.loads(line) for line in payload.splitlines() if line.strip()]
    except (gzip.BadGzipFile, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive member is invalid JSONL gzip: {name}"
        ) from error
    if not values or not isinstance(values[0], dict):
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive member has no header: {name}"
        )
    header = values[0]
    rows = values[1:]
    if header != {"schema_version": schema_version, "rows": expected_rows}:
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive header changed: {name}"
        )
    if len(rows) != expected_rows or any(not isinstance(row, dict) for row in rows):
        raise UgiTreeTransformerFailureAttributionError(
            f"evaluation archive row count changed: {name}"
        )
    return rows


def _ordered_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> None:
    observed = [row.get("attempt_index") for row in rows]
    if observed != list(range(len(rows))):
        raise UgiTreeTransformerFailureAttributionError(
            f"{label} attempt order is not the exact zero-based program order"
        )


def _native_forward_exact(sample: Mapping[str, Any]) -> bool:
    forward = sample.get("l1_forward_verification")
    return bool(
        sample.get("valid") is True
        and sample.get("component_reconstruction_valid") is True
        and isinstance(forward, Mapping)
        and forward.get("exact_product_reconstructed") is True
    )


def _amine_nitrogen_count(sample: Mapping[str, Any]) -> int | None:
    components = sample.get("component_smiles_by_role")
    if not isinstance(components, Mapping) or not isinstance(components.get("amine_head"), str):
        return None
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(str(components["amine_head"]))
    if molecule is None:
        raise UgiTreeTransformerFailureAttributionError(
            "native exact sample contains an invalid amine component"
        )
    return sum(atom.GetSymbol() == "N" for atom in molecule.GetAtoms())


def unsupported_edge_signatures(
    smiles: str,
    support: LocalChemistrySupport,
) -> list[str]:
    """Return every program-unsupported whole-product edge in stable bond order."""

    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise UgiTreeTransformerFailureAttributionError(
            "exact-L1 local-support row has an invalid molecule"
        )
    molecule = Chem.Mol(molecule)
    try:
        Chem.Kekulize(molecule, clearAromaticFlags=True)
    except (RuntimeError, ValueError) as error:
        raise UgiTreeTransformerFailureAttributionError(
            "exact-L1 local-support row cannot be Kekulized"
        ) from error
    signatures: list[str] = []
    for bond in molecule.GetBonds():
        left, right = sorted(
            (
                bond.GetBeginAtom().GetSymbol(),
                bond.GetEndAtom().GetSymbol(),
            )
        )
        try:
            bond_state = SPARSE_BOND_TO_INDEX[bond.GetBondType()]
        except KeyError:
            signatures.append(f"{left}-{right}:{bond.GetBondType()}:unmapped")
            continue
        if not support.allows_program_edge(UGI_PROGRAM_ID, left, bond_state, right):
            signatures.append(f"{left}-{right}:{bond.GetBondType()}")
    return signatures


def classify_attempt(
    sample: Mapping[str, Any],
    common: Mapping[str, Any],
    local: Mapping[str, Any],
) -> str:
    """Assign one mutually exclusive primary failure category."""

    valid = sample.get("valid") is True
    if not valid:
        failure_type = sample.get("failure_type")
        if failure_type == "TerminalSupportFailure":
            return "invalid_terminal_support"
        if failure_type == "MoleculeSanitizationFailure":
            return "invalid_molecule_sanitization"
        raise UgiTreeTransformerFailureAttributionError(
            f"unsupported invalid sampling failure type: {failure_type!r}"
        )
    if common.get("valid") is not True:
        raise UgiTreeTransformerFailureAttributionError(
            "sampling and method-blind validity disagree"
        )
    if common.get("exact_l1_program") is not True:
        if not _native_forward_exact(sample):
            raise UgiTreeTransformerFailureAttributionError(
                "valid retro abstention is not native-forward exact"
            )
        return "valid_native_forward_exact_retro_abstention"
    if local.get("raw_exact_l1_program") is not True:
        raise UgiTreeTransformerFailureAttributionError(
            "local assessment changed the raw exact-L1 label"
        )
    if local.get("local_support_qualified_exact_l1") is True:
        return "exact_l1_local_supported"
    return "exact_l1_unsupported_local_chemistry"


def _count_summary(counter: Mapping[str, int], denominator: int) -> dict[str, Any]:
    return {
        key: {"count": int(counter.get(key, 0)), "fraction": counter.get(key, 0) / denominator}
        for key in PRIMARY_CATEGORIES
    }


def _group_summary(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        label: {
            "attempts": len(group),
            "primary_categories": _count_summary(
                Counter(str(row["primary_attribution"]) for row in group), len(group)
            ),
        }
        for label, group in sorted(grouped.items())
    }


def _binary_rate_by_feature(
    rows: Sequence[Mapping[str, Any]],
    feature: str,
    category: str,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(feature)
        if value is not None:
            grouped[str(value)].append(row)
    return {
        value: {
            "eligible": len(group),
            "count": sum(row["primary_attribution"] == category for row in group),
            "fraction": (
                sum(row["primary_attribution"] == category for row in group) / len(group)
            ),
        }
        for value, group in sorted(grouped.items(), key=lambda item: int(item[0]))
    }


def _attempt_rows_for_run(
    archive_path: Path,
    seed: int,
    support: LocalChemistrySupport,
) -> list[dict[str, Any]]:
    with tarfile.open(archive_path, "r") as archive:
        sampling = _json_member(archive, _SAMPLING_MEMBER)
        samples = sampling.get("samples")
        if (
            sampling.get("schema_version") != "phase1_ugi_joint_end_to_end_sampling.v1"
            or sampling.get("seed") != seed + FLOW_SEED_OFFSET
            or not isinstance(samples, list)
            or len(samples) != ATTEMPTS_PER_SEED
        ):
            raise UgiTreeTransformerFailureAttributionError(
                "sampling result changed or has the wrong seed"
            )
        common_rows = _jsonl_member(
            archive,
            _COMMON_MEMBER,
            schema_version="forge.common_ugi_assessed_attempts.v1",
            expected_rows=ATTEMPTS_PER_SEED,
        )
        local_rows = _jsonl_member(
            archive,
            _LOCAL_MEMBER,
            schema_version="forge.common_local_chemistry_attempt.v1",
            expected_rows=ATTEMPTS_PER_SEED,
        )
        role_rows = _jsonl_member(
            archive,
            _ROLE_MEMBER,
            schema_version="forge.ugi_role_morphology_attempt.v1",
            expected_rows=ATTEMPTS_PER_SEED,
        )
    _ordered_rows(common_rows, label="common assessment")
    _ordered_rows(local_rows, label="local chemistry assessment")
    _ordered_rows(role_rows, label="role morphology assessment")

    rows: list[dict[str, Any]] = []
    for attempt_index, (sample, common, local, role) in enumerate(
        zip(samples, common_rows, local_rows, role_rows, strict=True)
    ):
        if (
            common.get("attempt_index") != attempt_index
            or local.get("attempt_index") != attempt_index
            or role.get("attempt_index") != attempt_index
            or common.get("seed") != seed
            or local.get("seed") != seed
            or role.get("seed") != seed
        ):
            raise UgiTreeTransformerFailureAttributionError(
                "attempt ledgers disagree on order or seed"
            )
        category = classify_attempt(sample, common, local)
        raw_signatures: list[str] = []
        if category == "exact_l1_unsupported_local_chemistry":
            smiles = common.get("canonical_smiles")
            if not isinstance(smiles, str):
                raise UgiTreeTransformerFailureAttributionError(
                    "local-chemistry failure has no canonical product"
                )
            raw_signatures = unsupported_edge_signatures(smiles, support)
            if len(raw_signatures) != int(local.get("unsupported_edge_count", -1)):
                raise UgiTreeTransformerFailureAttributionError(
                    "recomputed unsupported-edge count differs from frozen assessment"
                )
        program = sample.get("program")
        if not isinstance(program, Mapping):
            raise UgiTreeTransformerFailureAttributionError(
                "sampling row has no morphology program"
            )
        node_counts = program.get("node_counts")
        junctions = program.get("junction_budgets")
        cycles = program.get("cycle_ranks")
        attachments = program.get("attachment_counts")
        if any(
            not isinstance(values, list) or len(values) != len(ROLE_NAMES)
            for values in (node_counts, junctions, cycles, attachments)
        ):
            raise UgiTreeTransformerFailureAttributionError(
                "sampling morphology program dimensions changed"
            )
        forward = sample.get("l1_forward_verification")
        outcome_count = (
            int(forward["outcome_count"])
            if isinstance(forward, Mapping) and isinstance(forward.get("outcome_count"), int)
            else None
        )
        trace_assessments = role.get("trace_assessments")
        all_role_supported = bool(
            isinstance(trace_assessments, list)
            and trace_assessments
            and any(
                trace.get("all_roles_within_observed_hard_bounds") is True
                and trace.get("all_role_ring_signatures_supported") is True
                for trace in trace_assessments
                if isinstance(trace, Mapping)
            )
        )
        tail_supported = bool(
            isinstance(trace_assessments, list)
            and trace_assessments
            and any(
                trace.get("tails_within_observed_hard_bounds") is True
                and trace.get("tail_ring_signatures_supported") is True
                for trace in trace_assessments
                if isinstance(trace, Mapping)
            )
        )
        row: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "seed": seed,
            "attempt_index": attempt_index,
            "product_id": sample.get("product_id"),
            "source_stratum": sample.get("source_stratum"),
            "held_role_class": sample.get("held_role_class"),
            "primary_attribution": category,
            "sampling_failure_type": sample.get("failure_type"),
            "valid": sample.get("valid") is True,
            "native_forward_exact": _native_forward_exact(sample),
            "method_blind_exact_l1": common.get("exact_l1_program") is True,
            "local_support_qualified_exact_l1": (
                local.get("local_support_qualified_exact_l1") is True
            ),
            "local_failure_types": list(local.get("failure_types", [])),
            "unsupported_edge_signatures": sorted(raw_signatures),
            "nitrogen_oxygen_bond_count": int(role.get("nitrogen_oxygen_bond_count", 0)),
            "all_role_morphology_supported": all_role_supported,
            "tail_morphology_supported": tail_supported,
            "amine_nitrogen_count": _amine_nitrogen_count(sample),
            "native_forward_outcome_count": outcome_count,
            "amine_attachment_count": int(attachments[0]),
        }
        for role_index, role_name in enumerate(ROLE_NAMES):
            row[f"{role_name}_node_count"] = int(node_counts[role_index])
            row[f"{role_name}_junction_budget"] = int(junctions[role_index])
            row[f"{role_name}_cycle_rank"] = int(cycles[role_index])
        row["canonical_smiles"] = (
            common.get("canonical_smiles")
            if category != "exact_l1_local_supported"
            else None
        )
        rows.append(row)
    return rows


def _load_and_validate_config(config_path: Path, repo: Path) -> tuple[dict[str, Any], list[Any]]:
    config = read_json_object(
        config_path,
        error=UgiTreeTransformerFailureAttributionError,
        label="tree-Transformer failure-attribution config",
    )
    required = {
        "schema_version",
        "status",
        "experiment_id",
        "source_sha256",
        "expected_training_seeds",
        "attempts_per_seed",
        "inputs",
        "policy",
        "nonclaims",
    }
    if config.get("schema_version") != CONFIG_SCHEMA or set(config) != required:
        raise UgiTreeTransformerFailureAttributionError("diagnostic config fields changed")
    if (
        config.get("status") != "frozen_after_negative_full_ugi_evaluation"
        or config.get("experiment_id") != EXPERIMENT_ID
        or config.get("source_sha256") != SOURCE_SHA256
        or config.get("expected_training_seeds") != list(EXPECTED_SEEDS)
        or config.get("attempts_per_seed") != ATTEMPTS_PER_SEED
    ):
        raise UgiTreeTransformerFailureAttributionError("diagnostic frozen contract changed")
    expected_policy = {
        "primary_categories": list(PRIMARY_CATEGORIES),
        "attempt_order_must_be_exact": True,
        "native_forward_and_method_blind_retro_reported_separately": True,
        "unsupported_edge_signatures_recomputed_from_pinned_policy": True,
        "top_edge_signatures": 20,
        "heldout_used_for_model_checkpoint_threshold_or_policy_selection": False,
        "gate_or_threshold_changes": False,
        "candidate_selection": False,
        "route_calls": 0,
        "oracle_calls": 0,
    }
    if config.get("policy") != expected_policy:
        raise UgiTreeTransformerFailureAttributionError("diagnostic policy changed")
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "aggregate",
        "local_chemistry_support",
        "runs",
    }:
        raise UgiTreeTransformerFailureAttributionError("diagnostic inputs changed")
    aggregate_path = resolve_pin(inputs["aggregate"], repo, label="production aggregate")
    aggregate = read_json_object(
        aggregate_path,
        error=UgiTreeTransformerFailureAttributionError,
        label="production aggregate",
    )
    if (
        aggregate.get("schema_version") != "forge.ugi_tree_transformer_production_aggregate.v1"
        or aggregate.get("decision") != "negative_full_ugi_evaluation"
        or aggregate.get("training_seeds") != list(EXPECTED_SEEDS)
        or aggregate.get("source_sha256") != SOURCE_SHA256
    ):
        raise UgiTreeTransformerFailureAttributionError(
            "diagnostic requires the frozen negative aggregate"
        )
    raw_runs = inputs.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) != len(EXPECTED_SEEDS):
        raise UgiTreeTransformerFailureAttributionError("diagnostic run set changed")
    return config, raw_runs


def diagnose_tree_transformer_failures(
    config_path: Path,
    repo: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Produce a method-separating, nonselecting failure-attribution dossier."""

    repo = repo.resolve()
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise UgiTreeTransformerFailureAttributionError(
            f"diagnostic output already exists: {output_dir}"
        )
    config, raw_runs = _load_and_validate_config(config_path, repo)
    inputs = config["inputs"]
    support_path = resolve_pin(
        inputs["local_chemistry_support"], repo, label="local chemistry support"
    )
    support = LocalChemistrySupport.from_mapping(
        read_json_object(
            support_path,
            error=UgiTreeTransformerFailureAttributionError,
            label="local chemistry support",
        )
    )

    all_rows: list[dict[str, Any]] = []
    input_runs: dict[str, Any] = {}
    for replicate, (expected_seed, raw) in enumerate(
        zip(EXPECTED_SEEDS, raw_runs, strict=True)
    ):
        if not isinstance(raw, Mapping) or set(raw) != {
            "seed",
            "run_manifest",
            "evaluation_manifest",
            "evaluation_details",
        }:
            raise UgiTreeTransformerFailureAttributionError("diagnostic run record changed")
        if raw.get("seed") != expected_seed:
            raise UgiTreeTransformerFailureAttributionError("diagnostic run seed order changed")
        run_path = resolve_pin(raw["run_manifest"], repo, label=f"run {expected_seed}")
        manifest_path = resolve_pin(
            raw["evaluation_manifest"], repo, label=f"evaluation manifest {expected_seed}"
        )
        archive_path = resolve_pin(
            raw["evaluation_details"], repo, label=f"evaluation details {expected_seed}"
        )
        run = read_json_object(
            run_path,
            error=UgiTreeTransformerFailureAttributionError,
            label=f"run {expected_seed}",
        )
        manifest = read_json_object(
            manifest_path,
            error=UgiTreeTransformerFailureAttributionError,
            label=f"evaluation manifest {expected_seed}",
        )
        if (
            run.get("schema_version") != "forge.run_manifest.v1"
            or run.get("status") != "complete"
            or run.get("profile") != "full"
            or run.get("experiment_id") != EXPERIMENT_ID
            or run.get("source_sha256") != SOURCE_SHA256
            or run.get("replicate") != replicate
            or manifest.get("schema_version") != "forge.stage_manifest.v1"
            or manifest.get("status") != "complete"
            or manifest.get("stage_id") != "evaluate"
            or manifest.get("source_sha256") != SOURCE_SHA256
            or manifest.get("artifacts", {}).get("evaluation_details", {}).get("sha256")
            != raw["evaluation_details"]["sha256"]
        ):
            raise UgiTreeTransformerFailureAttributionError(
                f"run {expected_seed} violates the completed production contract"
            )
        rows = _attempt_rows_for_run(archive_path, expected_seed, support)
        all_rows.extend(rows)
        input_runs[str(expected_seed)] = {
            "run_manifest": pin_record(run_path, repo),
            "evaluation_manifest": pin_record(manifest_path, repo),
            "evaluation_details": pin_record(archive_path, repo),
        }

    denominator = len(all_rows)
    expected_denominator = len(EXPECTED_SEEDS) * ATTEMPTS_PER_SEED
    if denominator != expected_denominator:
        raise UgiTreeTransformerFailureAttributionError("diagnostic denominator changed")
    category_counts = Counter(str(row["primary_attribution"]) for row in all_rows)
    if set(category_counts).difference(PRIMARY_CATEGORIES) or sum(category_counts.values()) != denominator:
        raise UgiTreeTransformerFailureAttributionError(
            "primary attribution is not mutually exclusive and exhaustive"
        )
    by_seed = {
        str(seed): {
            "attempts": ATTEMPTS_PER_SEED,
            "primary_categories": _count_summary(
                Counter(
                    str(row["primary_attribution"])
                    for row in all_rows
                    if row["seed"] == seed
                ),
                ATTEMPTS_PER_SEED,
            ),
        }
        for seed in EXPECTED_SEEDS
    }

    unsupported_rows = [
        row
        for row in all_rows
        if row["primary_attribution"] == "exact_l1_unsupported_local_chemistry"
    ]
    edge_occurrences: Counter[str] = Counter()
    edge_products: Counter[str] = Counter()
    for row in unsupported_rows:
        signatures = list(row["unsupported_edge_signatures"])
        edge_occurrences.update(signatures)
        edge_products.update(set(signatures))
    top_limit = int(config["policy"]["top_edge_signatures"])
    ordered_signatures = sorted(
        edge_products,
        key=lambda signature: (-edge_products[signature], -edge_occurrences[signature], signature),
    )[:top_limit]
    edge_summary = [
        {
            "signature": signature,
            "products": edge_products[signature],
            "edge_occurrences": edge_occurrences[signature],
            "fraction_of_local_unsupported_products": (
                edge_products[signature] / len(unsupported_rows) if unsupported_rows else 0.0
            ),
        }
        for signature in ordered_signatures
    ]

    terminal_features: dict[str, Any] = {}
    for role_name in ROLE_NAMES:
        for feature in ("junction_budget", "cycle_rank", "node_count"):
            key = f"{role_name}_{feature}"
            terminal_features[key] = _binary_rate_by_feature(
                all_rows, key, "invalid_terminal_support"
            )
    retro_features = {
        key: _binary_rate_by_feature(
            [row for row in all_rows if row["native_forward_exact"] is True],
            key,
            "valid_native_forward_exact_retro_abstention",
        )
        for key in (
            "amine_nitrogen_count",
            "amine_attachment_count",
            "native_forward_outcome_count",
        )
    }
    invalid_count = (
        category_counts["invalid_terminal_support"]
        + category_counts["invalid_molecule_sanitization"]
    )
    exact_misses = invalid_count + category_counts[
        "valid_native_forward_exact_retro_abstention"
    ]
    failing_categories = {
        key: value for key, value in category_counts.items() if key != "exact_l1_local_supported"
    }
    largest_failure = max(failing_categories, key=failing_categories.get)
    unsupported_with_no = sum(row["nitrogen_oxygen_bond_count"] > 0 for row in unsupported_rows)
    unsupported_with_role_morphology = sum(
        row["all_role_morphology_supported"] is True for row in unsupported_rows
    )
    unsupported_with_tail_morphology = sum(
        row["tail_morphology_supported"] is True for row in unsupported_rows
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=str(output_dir.parent))
    )
    try:
        ledger_path = partial / "attempt_attributions.jsonl.gz"
        write_jsonl(
            ledger_path,
            [
                {"schema_version": ATTEMPT_SCHEMA, "rows": denominator},
                *all_rows,
            ],
        )
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "complete_negative_result_attributed",
            "decision": "preserve_negative_production_result",
            "attempts": denominator,
            "training_seeds": list(EXPECTED_SEEDS),
            "primary_attribution": {
                "overall": _count_summary(category_counts, denominator),
                "by_seed": by_seed,
                "mutually_exclusive_and_exhaustive": True,
            },
            "invalidity": {
                "invalid_attempts": invalid_count,
                "invalid_fraction_per_attempt": invalid_count / denominator,
                "terminal_support_failures": category_counts["invalid_terminal_support"],
                "molecule_sanitization_failures": category_counts[
                    "invalid_molecule_sanitization"
                ],
                "terminal_support_fraction_among_invalid": (
                    category_counts["invalid_terminal_support"] / invalid_count
                ),
                "by_source_stratum": _group_summary(all_rows, "source_stratum"),
                "by_held_role_class": _group_summary(all_rows, "held_role_class"),
                "terminal_support_topology_associations": terminal_features,
                "internal_exception_subtype_persisted": False,
            },
            "exact_l1": {
                "method_blind_exact_l1_misses": exact_misses,
                "invalid_misses": invalid_count,
                "valid_native_forward_exact_retro_abstentions": category_counts[
                    "valid_native_forward_exact_retro_abstention"
                ],
                "invalid_fraction_of_exact_l1_misses": invalid_count / exact_misses,
                "retro_abstention_fraction_of_exact_l1_misses": (
                    category_counts["valid_native_forward_exact_retro_abstention"]
                    / exact_misses
                ),
                "all_valid_samples_native_forward_exact": all(
                    row["native_forward_exact"] is True for row in all_rows if row["valid"] is True
                ),
                "retro_abstention_associations": retro_features,
                "interpretation": (
                    "A valid method-blind exact-L1 miss is a retro-decomposer abstention despite an "
                    "independently persisted native component trace whose forward assembly exactly "
                    "reconstructed the sampled product."
                ),
            },
            "local_chemistry": {
                "exact_l1_unsupported_products": len(unsupported_rows),
                "fraction_per_attempt": len(unsupported_rows) / denominator,
                "unsupported_edge_occurrences": sum(edge_occurrences.values()),
                "top_unsupported_edge_signatures": edge_summary,
                "products_with_nitrogen_oxygen_bonds": unsupported_with_no,
                "products_with_all_role_morphology_supported": unsupported_with_role_morphology,
                "products_with_tail_morphology_supported": unsupported_with_tail_morphology,
                "all_stored_unsupported_edge_counts_reproduced": True,
            },
            "findings": {
                "largest_nonpassing_category": largest_failure,
                "largest_nonpassing_category_count": failing_categories[largest_failure],
                "invalidity_is_mostly_terminal_support_failure": (
                    category_counts["invalid_terminal_support"]
                    > category_counts["invalid_molecule_sanitization"]
                ),
                "method_blind_retro_and_native_forward_are_distinct_failure_axes": (
                    category_counts["valid_native_forward_exact_retro_abstention"] > 0
                ),
                "local_unsupported_edges_are_the_largest_gate_loss": (
                    largest_failure == "exact_l1_unsupported_local_chemistry"
                ),
            },
            "recommended_order": [
                "Instrument terminal decoding to persist the exact no-feasible-atom, no-feasible-bond, aromatic-state or sanitization exception without changing acceptance.",
                "Qualify role-local support masks or losses for the dominant unsupported edge signatures without introducing a finite component vocabulary.",
                "Freeze and test a broader method-blind Ugi retro-decomposition policy against native forward traces before using it in a new comparison.",
                "Only then freeze a new sampler or training experiment; do not rerun the unchanged model blindly.",
            ],
            "inputs": {
                "config": pin_record(config_path, repo),
                "aggregate": pin_record(
                    resolve_pin(inputs["aggregate"], repo, label="production aggregate"), repo
                ),
                "local_chemistry_support": pin_record(support_path, repo),
                "runs": input_runs,
            },
            "attempt_attributions": artifact_record(
                ledger_path, logical_path="attempt_attributions.jsonl.gz"
            ),
            "heldout_used_for_model_checkpoint_threshold_or_policy_selection": False,
            "gate_or_threshold_changes": False,
            "candidate_selection": False,
            "calls": {"generation": 0, "route": 0, "oracle": 0},
            "implementation": pin_record(Path(__file__), repo),
            "nonclaims": list(config["nonclaims"]),
        }
        write_json(partial / "result.json", result)
        os.replace(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return result


__all__ = [
    "ATTEMPT_SCHEMA",
    "CONFIG_SCHEMA",
    "PRIMARY_CATEGORIES",
    "RESULT_SCHEMA",
    "UgiTreeTransformerFailureAttributionError",
    "classify_attempt",
    "diagnose_tree_transformer_failures",
    "unsupported_edge_signatures",
]
