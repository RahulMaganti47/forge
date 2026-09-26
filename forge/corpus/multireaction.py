"""Source-adjudicated multi-reaction program corpus from hash-pinned LNPDB records."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.assembly import (
    ReactionProgramError,
    ReactionProgramSpec,
    ReactionProgramTrace,
    RegistryRepeatedReactionProgram,
)
from forge.chemistry.smiles import canonical_connected_constitution, canonical_constitution
from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import atomic_write, csv_gz_bytes, read_json_object, stable_json
from forge.corpus.lnpdb import LNPDBRow, load_lnpdb

CONFIG_SCHEMA = "forge.multireaction_lnpdb_config.v1"
ATLAS_SCHEMA = "forge.multireaction_program_atlas.v1"
STEPS_SCHEMA = "forge.multireaction_program_steps.v1"
SEMANTIC_ATOMS_SCHEMA = "forge.multireaction_semantic_atoms.v2"
PROVENANCE_SCHEMA = "forge.multireaction_source_provenance.v1"
SPLITS_SCHEMA = "forge.multireaction_component_disjoint_splits.v1"
RESULT_SCHEMA = "forge.multireaction_lnpdb_result.v1"
MANIFEST_SCHEMA = "forge.multireaction_lnpdb_manifest.v1"

ATLAS_FIELDS = (
    "record_id",
    "program_id",
    "reaction_id",
    "source_study",
    "source_product_labels",
    "canonical_product_smiles",
    "terminal_head_smiles",
    "repeat_component_smiles",
    "step_count",
    "source_row_count",
    "evidence_basis",
    "disposition",
    "abstention_reason",
    "exact_forward_roundtrip",
    "source_locator",
    "semantic_origin_status",
    "semantic_origin_reason",
)
STEP_FIELDS = (
    "record_id",
    "program_id",
    "step_index",
    "reaction_id",
    "accumulator_role",
    "accumulator_input_smiles",
    "repeat_role",
    "repeat_component_smiles",
    "product_smiles",
    "exact_forward_roundtrip",
)
SEMANTIC_ATOM_FIELDS = (
    "record_id",
    "program_id",
    "atom_index",
    "origin_role",
    "core_position",
    "program_depth",
)
PROVENANCE_FIELDS = (
    "source_row_id",
    "record_id",
    "program_id",
    "source_study",
    "lnp_id",
    "formulation_id",
    "source_product_label",
    "canonical_product_smiles",
)
SPLIT_FIELDS = (
    "record_id",
    "program_id",
    "head_component_id",
    "head_component_fold",
    "repeat_component_id",
    "repeat_component_fold",
    "product_fold",
    "source_balanced_weight",
)


class MultiReactionCorpusError(ValueError):
    """The multi-reaction corpus cannot satisfy its frozen evidence contract."""


@dataclass(frozen=True)
class _AdmittedRecord:
    record_id: str
    program_id: str
    reaction_id: str
    head_smiles: str
    repeat_smiles: str
    step_products: tuple[str, ...]


def _identifier(prefix: str, *values: str) -> str:
    payload = "\x1f".join(values).encode()
    return f"{prefix}-{hashlib.sha256(payload).hexdigest()[:16]}"


def _config(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = read_json_object(path, error=MultiReactionCorpusError, label="multi-reaction config")
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise MultiReactionCorpusError(
            f"unsupported config schema: {config.get('schema_version')!r}"
        )
    if config.get("graph_identity") != "canonical_constitutional_smiles":
        raise MultiReactionCorpusError("multi-reaction model identity must be constitutional")
    seed = config.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MultiReactionCorpusError("config seed must be a non-negative integer")
    inputs = config.get("inputs")
    programs = config.get("programs")
    if not isinstance(inputs, dict) or not isinstance(programs, list) or not programs:
        raise MultiReactionCorpusError("config requires input pins and at least one program")
    resolved = {
        str(label): resolve_pin(pin, repo, label=str(label)) for label, pin in inputs.items()
    }
    return config, resolved


def _source_review_index(path: Path) -> dict[str, dict[str, Any]]:
    document = read_json_object(path, error=MultiReactionCorpusError, label="source reviews")
    reviews = document.get("reviews")
    if not isinstance(reviews, list):
        raise MultiReactionCorpusError("source reviews have no review list")
    index: dict[str, dict[str, Any]] = {}
    for value in reviews:
        if not isinstance(value, dict) or not isinstance(value.get("review_id"), str):
            raise MultiReactionCorpusError("source review record is malformed")
        index[value["review_id"]] = value
    return index


def _lx_component_overlay(review: Mapping[str, Any]) -> dict[str, str]:
    overlay: dict[str, str] = {}
    route_families = review.get("l2_route_families")
    if not isinstance(route_families, list):
        raise MultiReactionCorpusError("LX source review has no L2 route families")
    for family in route_families:
        if not isinstance(family, dict):
            continue
        members = family.get("members", [])
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, dict):
                continue
            label = member.get("label")
            smiles = member.get("product_smiles")
            if isinstance(label, str) and isinstance(smiles, str):
                overlay[label.replace("-", "_T", 1)] = str(
                    canonical_connected_constitution(smiles, error=MultiReactionCorpusError)
                )
    if len(overlay) != 15:
        raise MultiReactionCorpusError(
            f"LX source overlay resolved {len(overlay)} members, expected 15"
        )
    return overlay


def _component_folds(
    components: Sequence[str],
    *,
    seed: int,
    program_id: str,
    role: str,
    fractions: Mapping[str, float],
) -> dict[str, str]:
    unique = sorted(set(components))
    if len(unique) < 3:
        raise MultiReactionCorpusError(
            f"{program_id}/{role} needs at least three components for disjoint folds"
        )
    ranked = sorted(
        unique,
        key=lambda smiles: hashlib.sha256(
            f"{seed}|{program_id}|{role}|{smiles}".encode()
        ).hexdigest(),
    )
    heldout_count = max(1, round(len(ranked) * fractions["heldout"]))
    calibration_count = max(1, round(len(ranked) * fractions["calibration"]))
    if heldout_count + calibration_count >= len(ranked):
        raise MultiReactionCorpusError(f"{program_id}/{role} leaves no training components")
    result = {smiles: "train" for smiles in ranked}
    for smiles in ranked[:heldout_count]:
        result[smiles] = "heldout"
    for smiles in ranked[heldout_count : heldout_count + calibration_count]:
        result[smiles] = "calibration"
    return result


def _product_fold(head_fold: str, repeat_fold: str) -> str:
    if "heldout" in {head_fold, repeat_fold}:
        return "heldout"
    if "calibration" in {head_fold, repeat_fold}:
        return "calibration"
    return "train"


def _render_outputs(
    outputs: Mapping[str, Path],
    *,
    atlas: Sequence[Mapping[str, Any]],
    steps: Sequence[Mapping[str, Any]],
    provenance: Sequence[Mapping[str, Any]],
    splits: Sequence[Mapping[str, Any]],
    semantic_atoms: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    payloads = {
        "atlas": csv_gz_bytes(atlas, ATLAS_FIELDS),
        "steps": csv_gz_bytes(steps, STEP_FIELDS),
        "provenance": csv_gz_bytes(provenance, PROVENANCE_FIELDS),
        "splits": csv_gz_bytes(splits, SPLIT_FIELDS),
        "semantic_atoms": csv_gz_bytes(semantic_atoms, SEMANTIC_ATOM_FIELDS),
    }
    for label, payload in payloads.items():
        atomic_write(outputs[label], payload)
    artifact_schemas = {
        "atlas": ATLAS_SCHEMA,
        "steps": STEPS_SCHEMA,
        "provenance": PROVENANCE_SCHEMA,
        "splits": SPLITS_SCHEMA,
        "semantic_atoms": SEMANTIC_ATOMS_SCHEMA,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "artifacts": {
            label: {
                "path": outputs[label].name,
                "rows": len(
                    {
                        "atlas": atlas,
                        "steps": steps,
                        "provenance": provenance,
                        "semantic_atoms": semantic_atoms,
                        "splits": splits,
                    }[label]
                ),
                "schema_version": artifact_schemas[label],
                "sha256": str(sha256_file(outputs[label])),
            }
            for label in sorted(payloads)
        },
    }
    atomic_write(outputs["manifest"], f"{stable_json(manifest)}\n".encode())
    atomic_write(outputs["result"], f"{stable_json(result)}\n".encode())
    return manifest


def build_multireaction_lnpdb_corpus(
    config_path: Path,
    repo: Path,
    *,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Build the admitted atlas, exact steps, provenance ledger and disjoint splits."""

    required_outputs = {
        "atlas",
        "steps",
        "semantic_atoms",
        "provenance",
        "splits",
        "manifest",
        "result",
    }
    if set(outputs) != required_outputs:
        raise MultiReactionCorpusError(
            f"outputs differ from contract: expected {sorted(required_outputs)}, got {sorted(outputs)}"
        )
    config, inputs = _config(config_path, repo)
    reviews = _source_review_index(inputs["source_reviews"])
    lnpdb = load_lnpdb(inputs["lnpdb"], error=MultiReactionCorpusError)
    registry_path = inputs["qualified_reaction_families"]
    limits = config.get("limits")
    if not isinstance(limits, dict):
        raise MultiReactionCorpusError("config limits are missing")
    maximum_outcomes = int(limits["maximum_outcomes_per_step"])
    maximum_states = int(limits["maximum_reverse_states"])
    atlas: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    semantic_atoms: list[dict[str, Any]] = []
    admitted: list[_AdmittedRecord] = []
    summaries: dict[str, dict[str, Any]] = {}

    for raw_program in config["programs"]:
        if not isinstance(raw_program, dict):
            raise MultiReactionCorpusError("program entries must be objects")
        program_id = str(raw_program["program_id"])
        source_study = str(raw_program["source_study"])
        source_review_id = str(raw_program["source_review_id"])
        if source_review_id not in reviews:
            raise MultiReactionCorpusError(f"unknown source review {source_review_id!r}")
        review = reviews[source_review_id]
        source_asset = review.get("source_asset")
        if not isinstance(source_asset, dict) or source_asset.get("expected_sha256") != str(
            config["inputs"][raw_program["source_asset"]]["sha256"]
        ):
            raise MultiReactionCorpusError(f"{program_id} source review and source pin disagree")
        spec = ReactionProgramSpec(
            program_id=program_id,
            reaction_id=str(raw_program["reaction_id"]),
            accumulator_role=str(raw_program["accumulator_role"]),
            repeat_role=str(raw_program["repeat_role"]),
            minimum_steps=int(raw_program["minimum_steps"]),
            maximum_steps=int(raw_program["maximum_steps"]),
        )
        adapter = RegistryRepeatedReactionProgram.from_registry(
            registry_path,
            spec,
            expected_sha256=str(config["inputs"]["qualified_reaction_families"]["sha256"]),
        )
        source_rows = lnpdb.study(source_study)
        if len(source_rows) != int(raw_program["expected_source_rows"]):
            raise MultiReactionCorpusError(
                f"{program_id} source row count changed: {len(source_rows)}"
            )
        grouped: dict[str, list[LNPDBRow]] = defaultdict(list)
        invalid_products = 0
        for source_row in source_rows:
            try:
                product = str(
                    canonical_connected_constitution(
                        source_row.lipid_smiles, error=MultiReactionCorpusError
                    )
                )
            except MultiReactionCorpusError:
                invalid_products += 1
                continue
            grouped[product].append(source_row)
        if len(grouped) != int(raw_program["expected_unique_lnpdb_products"]):
            raise MultiReactionCorpusError(
                f"{program_id} unique product count changed: {len(grouped)}"
            )
        overlay = _lx_component_overlay(review) if source_study == "LX_2024" else {}
        abstain_labels = raw_program.get("abstain_component_labels", {})
        if not isinstance(abstain_labels, dict):
            raise MultiReactionCorpusError(f"{program_id} abstention labels must be an object")
        program_counts: Counter[str] = Counter()
        step_counts: Counter[int] = Counter()
        semantic_products = 0
        semantic_core_products = 0

        for product, rows in sorted(grouped.items()):
            labels = tuple(sorted({row.lipid_name for row in rows if row.lipid_name}))
            tail_labels = tuple(
                sorted({str(row.tail(1).name) for row in rows if row.tail(1).name is not None})
            )
            record_id = _identifier("mrp", program_id, product)
            try:
                heads = {
                    str(canonical_constitution(row.head.smiles))
                    for row in rows
                    if row.head.smiles is not None
                }
            except Exception:
                heads = set()
            expected_repeat: str | None = None
            abstention_reason = ""
            if len(heads) != 1:
                abstention_reason = "source rows do not resolve one terminal head structure"
            elif source_study == "LX_2024":
                if len(tail_labels) != 1 or tail_labels[0] not in overlay:
                    abstention_reason = "source review does not resolve one aldehyde component"
                else:
                    expected_repeat = overlay[tail_labels[0]]
            traces: tuple[ReactionProgramTrace, ...] = ()
            if not abstention_reason:
                traces = adapter.decompose(
                    product,
                    terminal_head_smiles=next(iter(heads)),
                    expected_repeat_smiles=expected_repeat,
                    maximum_outcomes=maximum_outcomes,
                    maximum_states=maximum_states,
                )
                if not traces:
                    abstention_reason = "no exact source-compatible recursive decomposition"
                elif len(traces) != 1:
                    abstention_reason = "multiple exact recursive decompositions remain"
            trace = traces[0] if len(traces) == 1 else None
            if trace is not None and raw_program["require_identical_repeat_components"]:
                if len(set(trace.repeated_component_smiles)) != 1:
                    abstention_reason = "recursive decomposition uses non-identical repeated inputs"
                    trace = None
            conflict_reasons = [
                str(abstain_labels[label]) for label in tail_labels if label in abstain_labels
            ]
            if conflict_reasons:
                abstention_reason = "; ".join(conflict_reasons)
                trace = None
            disposition = str(raw_program["disposition"]) if trace is not None else "abstain"
            repeat_smiles = trace.repeated_component_smiles[0] if trace is not None else ""
            semantic_status = "not_admitted"
            semantic_reason = ""
            origins = None
            if trace is not None:
                try:
                    origins = adapter.atom_origins(trace)
                    if origins.canonical_product_smiles != product:
                        raise MultiReactionCorpusError(
                            f"{record_id} semantic replay changed constitutional product identity"
                        )
                    semantic_status = "exact"
                except ReactionProgramError as exc:
                    semantic_status = "abstain"
                    semantic_reason = str(exc)
            atlas.append(
                {
                    "record_id": record_id,
                    "program_id": program_id,
                    "reaction_id": spec.reaction_id,
                    "source_study": source_study,
                    "source_product_labels": "|".join(labels),
                    "canonical_product_smiles": product,
                    "terminal_head_smiles": next(iter(heads)) if len(heads) == 1 else "",
                    "repeat_component_smiles": repeat_smiles,
                    "step_count": trace.step_count if trace is not None else 0,
                    "source_row_count": len(rows),
                    "evidence_basis": str(raw_program["evidence_basis"]),
                    "disposition": disposition,
                    "abstention_reason": abstention_reason,
                    "exact_forward_roundtrip": str(trace is not None).lower(),
                    "source_locator": str(raw_program["source_locator"]),
                    "semantic_origin_status": semantic_status,
                    "semantic_origin_reason": semantic_reason,
                }
            )
            program_counts[disposition] += 1
            if trace is not None:
                step_counts[trace.step_count] += 1
                accumulator = trace.terminal_head_smiles
                for step_index, (repeated, step_product) in enumerate(
                    zip(
                        trace.repeated_component_smiles,
                        trace.intermediate_product_smiles,
                        strict=True,
                    ),
                    start=1,
                ):
                    steps.append(
                        {
                            "record_id": record_id,
                            "program_id": program_id,
                            "step_index": step_index,
                            "reaction_id": spec.reaction_id,
                            "accumulator_role": spec.accumulator_role,
                            "accumulator_input_smiles": accumulator,
                            "repeat_role": spec.repeat_role,
                            "repeat_component_smiles": repeated,
                            "product_smiles": step_product,
                            "exact_forward_roundtrip": "true",
                        }
                    )
                    accumulator = step_product
                admitted.append(
                    _AdmittedRecord(
                        record_id=record_id,
                        program_id=program_id,
                        reaction_id=spec.reaction_id,
                        head_smiles=trace.terminal_head_smiles,
                        repeat_smiles=repeat_smiles,
                        step_products=trace.intermediate_product_smiles,
                    )
                )
                if origins is not None:
                    semantic_products += 1
                    if any(origins.core_positions):
                        semantic_core_products += 1
                    semantic_atoms.extend(
                        {
                            "record_id": record_id,
                            "program_id": program_id,
                            "atom_index": atom_index,
                            "origin_role": (
                                spec.accumulator_role
                                if origin == "accumulator"
                                else spec.repeat_role
                            ),
                            "core_position": core_position,
                            "program_depth": origins.step_count,
                        }
                        for atom_index, (origin, core_position) in enumerate(
                            zip(origins.atom_origins, origins.core_positions, strict=True)
                        )
                    )
            for source_row in sorted(rows, key=lambda value: value.index):
                if source_row.formulation_id is None:
                    raise MultiReactionCorpusError(
                        f"{source_row.lnp_id} has no formulation identifier"
                    )
                provenance.append(
                    {
                        "source_row_id": f"lnpdb-row-{source_row.index}",
                        "record_id": record_id,
                        "program_id": program_id,
                        "source_study": source_study,
                        "lnp_id": source_row.lnp_id,
                        "formulation_id": source_row.formulation_id,
                        "source_product_label": source_row.lipid_name,
                        "canonical_product_smiles": product,
                    }
                )
        admitted_count = program_counts["admit_exact"]
        minimum = int(raw_program["minimum_admitted_products"])
        if admitted_count < minimum:
            raise MultiReactionCorpusError(
                f"{program_id} admitted {admitted_count} products; gate requires {minimum}"
            )
        semantic_fraction = semantic_products / admitted_count
        if semantic_fraction < float(limits["minimum_semantic_origin_fraction"]):
            raise MultiReactionCorpusError(
                f"{program_id} semantic-origin fraction {semantic_fraction:.6f} is below gate"
            )
        semantic_core_fraction = semantic_core_products / admitted_count
        if semantic_core_fraction < float(limits["minimum_semantic_core_fraction"]):
            raise MultiReactionCorpusError(
                f"{program_id} semantic-core fraction {semantic_core_fraction:.6f} is below gate"
            )
        summaries[program_id] = {
            "source_rows": len(source_rows),
            "invalid_source_products": invalid_products,
            "unique_lnpdb_products": len(grouped),
            "source_reported_library_size": int(raw_program["source_reported_library_size"]),
            "admitted_products": admitted_count,
            "abstained_products": program_counts["abstain"],
            "source_library_coverage": admitted_count
            / int(raw_program["source_reported_library_size"]),
            "lnpdb_decomposition_coverage": admitted_count / len(grouped),
            "source_adjudicated_precision": 1.0,
            "semantic_origin_products": semantic_products,
            "semantic_origin_fraction": semantic_fraction,
            "semantic_core_products": semantic_core_products,
            "semantic_core_fraction": semantic_core_fraction,
            "step_count_distribution": {
                str(step): count for step, count in sorted(step_counts.items())
            },
            "gate": "pass",
        }

    split_policy = config.get("split_policy")
    if not isinstance(split_policy, dict):
        raise MultiReactionCorpusError("split policy is missing")
    fractions = {
        fold: float(split_policy[f"{fold}_fraction"])
        for fold in ("train", "calibration", "heldout")
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise MultiReactionCorpusError("split fractions must sum to one")
    seed = int(config["seed"])
    head_folds: dict[tuple[str, str], str] = {}
    repeat_folds: dict[tuple[str, str], str] = {}
    for program_id in sorted({record.program_id for record in admitted}):
        records = [record for record in admitted if record.program_id == program_id]
        head_folds.update(
            {
                (program_id, smiles): fold
                for smiles, fold in _component_folds(
                    [record.head_smiles for record in records],
                    seed=seed,
                    program_id=program_id,
                    role="head",
                    fractions=fractions,
                ).items()
            }
        )
        repeat_folds.update(
            {
                (program_id, smiles): fold
                for smiles, fold in _component_folds(
                    [record.repeat_smiles for record in records],
                    seed=seed,
                    program_id=program_id,
                    role="repeat",
                    fractions=fractions,
                ).items()
            }
        )
    preliminary: list[dict[str, Any]] = []
    fold_program_counts: Counter[tuple[str, str]] = Counter()
    fold_counts: Counter[str] = Counter()
    program_count = len({record.program_id for record in admitted})
    for record in admitted:
        head_fold = head_folds[(record.program_id, record.head_smiles)]
        repeat_fold = repeat_folds[(record.program_id, record.repeat_smiles)]
        product_fold = _product_fold(head_fold, repeat_fold)
        preliminary.append(
            {
                "record_id": record.record_id,
                "program_id": record.program_id,
                "head_component_id": _identifier("head", record.program_id, record.head_smiles),
                "head_component_fold": head_fold,
                "repeat_component_id": _identifier(
                    "repeat", record.program_id, record.repeat_smiles
                ),
                "repeat_component_fold": repeat_fold,
                "product_fold": product_fold,
            }
        )
        fold_program_counts[(product_fold, record.program_id)] += 1
        fold_counts[product_fold] += 1
    splits: list[dict[str, Any]] = []
    for row in preliminary:
        denominator = fold_program_counts[(row["product_fold"], row["program_id"])]
        weight = fold_counts[row["product_fold"]] / (program_count * denominator)
        splits.append({**row, "source_balanced_weight": f"{weight:.12g}"})

    input_records = {
        label: {
            "path": str(path.relative_to(repo)),
            "sha256": str(sha256_file(path)),
        }
        for label, path in sorted(inputs.items())
    }
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass",
        "seed": seed,
        "inputs": input_records,
        "config": {
            "path": str(config_path.resolve().relative_to(repo.resolve())),
            "sha256": str(sha256_file(config_path)),
        },
        "summary": {
            "source_rows": len(provenance),
            "unique_source_products": len(atlas),
            "admitted_products": len(admitted),
            "abstained_products": sum(row["disposition"] == "abstain" for row in atlas),
            "exact_program_steps": len(steps),
            "semantic_atom_rows": len(semantic_atoms),
            "semantic_core_atom_rows": sum(bool(row["core_position"]) for row in semantic_atoms),
            "semantic_origin_products": sum(
                summary["semantic_origin_products"] for summary in summaries.values()
            ),
            "semantic_core_products": sum(
                summary["semantic_core_products"] for summary in summaries.values()
            ),
            "product_folds": dict(sorted(fold_counts.items())),
            "sampling_policy": split_policy["sampling"],
        },
        "programs": summaries,
        "nonclaims": list(config.get("nonclaims", [])),
    }
    _render_outputs(
        outputs,
        atlas=atlas,
        steps=steps,
        provenance=provenance,
        splits=splits,
        semantic_atoms=semantic_atoms,
        result=result,
    )
    return result


__all__ = [
    "ATLAS_SCHEMA",
    "MANIFEST_SCHEMA",
    "PROVENANCE_SCHEMA",
    "RESULT_SCHEMA",
    "SEMANTIC_ATOMS_SCHEMA",
    "SPLITS_SCHEMA",
    "STEPS_SCHEMA",
    "MultiReactionCorpusError",
    "build_multireaction_lnpdb_corpus",
]
