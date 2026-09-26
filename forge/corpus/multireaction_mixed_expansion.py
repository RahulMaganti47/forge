"""Build a deterministic mixed-repeat BL/LX reaction-enumerated corpus.

Version 1 deliberately repeated one component at every reactive site because that is the topology
observed in the source libraries.  This version adds a separately weighted computed layer in which
two or more source-linked repeat components occur in one exact registry-backed program.  It keeps
source execution, homogeneous transform consistency, and mixed transform consistency as distinct
evidence strata.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.assembly import (
    ReactionProgramError,
    ReactionProgramSpec,
    ReactionProgramTrace,
    RegistryRepeatedReactionProgram,
)
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import (
    atomic_write,
    iter_csv,
    read_csv_rows,
    read_json_object,
    write_csv_iter,
    write_json,
)
from forge.corpus.component_splits import FOLDS, product_fold
from forge.corpus.multireaction import (
    ATLAS_FIELDS,
    SEMANTIC_ATOM_FIELDS,
    STEP_FIELDS,
)
from forge.corpus.multireaction_expansion import COMPONENT_FIELDS
from forge.model.vocabulary import load_atom_vocabulary

CONFIG_SCHEMA = "forge.multireaction_mixed_expansion_config.v1"
RESULT_SCHEMA = "forge.multireaction_mixed_expansion_result.v1"
MANIFEST_SCHEMA = "forge.multireaction_mixed_expansion_manifest.v1"
ATTEMPT_SCHEMA = "forge.multireaction_mixed_expansion_attempts.v1"
ATLAS_SCHEMA = "forge.multireaction_program_atlas.v2"
SPLITS_SCHEMA = "forge.multireaction_mixed_component_family_splits.v1"
SUPPORT_EXCLUSIONS_SCHEMA = "forge.multireaction_model_support_exclusions.v1"

ATLAS_FIELDS_V2 = (*ATLAS_FIELDS, "repeat_component_smiles_json")
ATTEMPT_FIELDS = (
    "attempt_index",
    "attempt_id",
    "program_id",
    "requested_fold",
    "head_component_id",
    "repeat_component_ids_json",
    "step_count",
    "disposition",
    "forward_products",
    "canonical_product_smiles",
    "reason",
)
SPLIT_FIELDS = (
    "record_id",
    "program_id",
    "head_component_id",
    "head_component_family_id",
    "head_component_fold",
    "repeat_component_ids_json",
    "repeat_component_family_ids_json",
    "repeat_component_folds_json",
    "product_fold",
    "evidence_stratum",
    "family_balance_weight_raw",
    "source_balanced_weight",
)
SUPPORT_EXCLUSION_FIELDS = (
    "record_id",
    "program_id",
    "evidence_basis",
    "reason",
    "closure_count",
)

SOURCE_STRATUM = "source_executed"
HOMOGENEOUS_STRATUM = "computed_homogeneous_transform_consistency"
MIXED_STRATUM = "computed_mixed_transform_consistency"
EVIDENCE_STRATA = (SOURCE_STRATUM, HOMOGENEOUS_STRATUM, MIXED_STRATUM)


class MultiReactionMixedExpansionError(ValueError):
    """The mixed-repeat expansion violates its frozen data or chemistry contract."""


@dataclass(frozen=True)
class _Component:
    component_id: str
    program_id: str
    program_role: str
    reaction_role: str
    canonical_smiles: str
    family_id: str
    family_size: int
    family_fold: str
    step_count: int


@dataclass(frozen=True)
class _Request:
    attempt_index: int
    attempt_id: str
    program_id: str
    requested_fold: str
    head: _Component
    repeats: tuple[_Component, ...]


@dataclass(frozen=True)
class _ForwardResult:
    request: _Request
    disposition: str
    reason: str
    forward_products: int
    product_smiles: str
    intermediate_products: tuple[str, ...]


@dataclass(frozen=True)
class _SelectedProduct:
    record_id: str
    request: _Request
    product_smiles: str
    intermediate_products: tuple[str, ...]
    origin_codes: bytes
    core_position_codes: bytes


_WORKER_ADAPTERS: dict[str, RegistryRepeatedReactionProgram] = {}
_WORKER_POLICY: dict[str, Any] = {}


def _identifier(prefix: str, *values: str) -> str:
    digest = hashlib.sha256("\x1f".join(values).encode()).hexdigest()[:20]
    return f"{prefix}-{digest}"


def _json(values: Sequence[str]) -> str:
    return json.dumps(list(values), separators=(",", ":"))


def _canonical(smiles: str) -> tuple[str, Chem.Mol] | None:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    with rdBase.BlockLogs():
        normalized = Chem.MolFromSmiles(canonical)
    return (canonical, normalized) if normalized is not None else None


def _load_config(path: Path, repo: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    config = read_json_object(
        path,
        error=MultiReactionMixedExpansionError,
        label="mixed-repeat expansion config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise MultiReactionMixedExpansionError(
            f"unsupported mixed-repeat config schema: {config.get('schema_version')!r}"
        )
    if isinstance(config.get("seed"), bool) or not isinstance(config.get("seed"), int):
        raise MultiReactionMixedExpansionError("mixed-repeat seed must be an integer")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        raise MultiReactionMixedExpansionError("mixed-repeat input pins are missing")
    paths: dict[str, Path] = {}
    for label, pin in sorted(inputs.items()):
        try:
            paths[label] = resolve_pin(pin, repo, label=label)
        except ValueError as exc:
            raise MultiReactionMixedExpansionError(str(exc)) from exc
    required = {
        "component_registry",
        "homogeneous_atlas",
        "homogeneous_attempts",
        "homogeneous_provenance",
        "homogeneous_result",
        "homogeneous_semantic_atoms",
        "homogeneous_splits",
        "homogeneous_steps",
        "program_config",
        "qualified_reaction_families",
    }
    if frozenset(paths) not in {
        frozenset(required),
        frozenset({*required, "atom_vocabulary"}),
    }:
        raise MultiReactionMixedExpansionError(
            f"mixed-repeat inputs differ from contract: {sorted(set(paths) ^ required)}"
        )
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise MultiReactionMixedExpansionError("mixed-repeat policy is missing")
    if policy.get("identity") != "canonical_constitutional_smiles":
        raise MultiReactionMixedExpansionError("mixed-repeat identity must remain constitutional")
    if policy.get("component_family_assignment_precedes_enumeration") is not True:
        raise MultiReactionMixedExpansionError("component families must precede mixed enumeration")
    if policy.get("computed_product_is_observed_synthesis") is not False:
        raise MultiReactionMixedExpansionError("computed products are not observed syntheses")
    if policy.get("computed_product_is_route_closed") is not False:
        raise MultiReactionMixedExpansionError("computed products are not route closed")
    strict_model_support = "atom_vocabulary" in paths
    if strict_model_support and (
        int(policy.get("maximum_product_closures", -1)) != 3
        or policy.get("require_pinned_atom_vocabulary") is not True
    ):
        raise MultiReactionMixedExpansionError(
            "strict mixed-repeat expansion must retain the frozen atom vocabulary and three-closure support"
        )
    masses = policy.get("evidence_stratum_mass")
    if not isinstance(masses, dict) or tuple(masses) != EVIDENCE_STRATA:
        raise MultiReactionMixedExpansionError(
            f"evidence masses must define {EVIDENCE_STRATA} in order"
        )
    if abs(sum(float(value) for value in masses.values()) - 1.0) > 1e-12:
        raise MultiReactionMixedExpansionError("evidence masses must sum to one")
    if any(float(value) <= 0.0 for value in masses.values()):
        raise MultiReactionMixedExpansionError("evidence masses must be positive")
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise MultiReactionMixedExpansionError("mixed-repeat execution policy is missing")
    for field in (
        "candidate_multiplier_by_program_fold",
        "origin_batch_size",
        "workers",
        "worker_chunksize",
    ):
        if field not in execution:
            raise MultiReactionMixedExpansionError(f"mixed-repeat execution omits {field}")
    multipliers = execution["candidate_multiplier_by_program_fold"]
    expected_programs = {
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    }
    if not isinstance(multipliers, dict) or set(multipliers) != expected_programs:
        raise MultiReactionMixedExpansionError("candidate multiplier programs changed")
    for program_id, by_fold in multipliers.items():
        if not isinstance(by_fold, dict) or tuple(by_fold) != FOLDS:
            raise MultiReactionMixedExpansionError(
                f"candidate multiplier folds changed for {program_id}"
            )
        if any(float(value) <= 1.0 for value in by_fold.values()):
            raise MultiReactionMixedExpansionError("candidate multipliers must exceed one")
    if int(execution["origin_batch_size"]) < 1 or int(execution["worker_chunksize"]) < 1:
        raise MultiReactionMixedExpansionError("mixed-repeat batch sizes must be positive")
    if int(execution["workers"]) < 0:
        raise MultiReactionMixedExpansionError("mixed-repeat workers cannot be negative")
    return config, paths


def _program_specs(path: Path) -> dict[str, ReactionProgramSpec]:
    document = read_json_object(
        path,
        error=MultiReactionMixedExpansionError,
        label="reaction-program config",
    )
    result: dict[str, ReactionProgramSpec] = {}
    for raw in document.get("programs", []):
        if not isinstance(raw, dict):
            continue
        spec = ReactionProgramSpec(
            program_id=str(raw["program_id"]),
            reaction_id=str(raw["reaction_id"]),
            accumulator_role=str(raw["accumulator_role"]),
            repeat_role=str(raw["repeat_role"]),
            minimum_steps=int(raw["minimum_steps"]),
            maximum_steps=int(raw["maximum_steps"]),
        )
        result[spec.program_id] = spec
    if set(result) != {
        "bl_2023_repeated_aza_michael",
        "lx_2024_repeated_reductive_amination",
    }:
        raise MultiReactionMixedExpansionError("mixed-repeat programs changed")
    return result


def _components(
    path: Path, specs: Mapping[str, ReactionProgramSpec]
) -> tuple[list[dict[str, str]], dict[str, _Component]]:
    rows = read_csv_rows(
        path,
        error=MultiReactionMixedExpansionError,
        label="v1 component registry",
        required_fields=COMPONENT_FIELDS,
    )
    result: dict[str, _Component] = {}
    for row in rows:
        if row["program_structural_admission"] != "true":
            continue
        program_id = row["program_id"]
        if program_id not in specs:
            raise MultiReactionMixedExpansionError(f"unknown component program: {program_id}")
        component = _Component(
            component_id=row["component_id"],
            program_id=program_id,
            program_role=row["program_role"],
            reaction_role=row["reaction_role"],
            canonical_smiles=row["canonical_smiles"],
            family_id=row["family_id"],
            family_size=int(row["family_size"]),
            family_fold=row["family_fold"],
            step_count=int(row["reactive_hydrogen_capacity"]),
        )
        if component.component_id in result:
            raise MultiReactionMixedExpansionError(
                f"duplicate admitted component id: {component.component_id}"
            )
        result[component.component_id] = component
    return rows, result


def _counter_integer(seed: int, *values: str) -> int:
    payload = "\x1f".join((str(seed), *values)).encode()
    return int.from_bytes(hashlib.sha256(payload).digest(), "big")


def _unrank_combination(population: int, selections: int, rank: int) -> tuple[int, ...]:
    """Unrank a lexicographic combination with replacement."""

    if population < 1 or selections < 1:
        raise MultiReactionMixedExpansionError("multiset dimensions must be positive")
    total = math.comb(population + selections - 1, selections)
    if rank < 0 or rank >= total:
        raise MultiReactionMixedExpansionError("multiset rank is outside support")
    strict_population = population + selections - 1
    strict: list[int] = []
    start = 0
    remaining = rank
    for position in range(selections):
        suffix = selections - position - 1
        for value in range(start, strict_population):
            count = math.comb(strict_population - value - 1, suffix) if suffix else 1
            if remaining < count:
                strict.append(value)
                start = value + 1
                break
            remaining -= count
        else:  # pragma: no cover - protected by the rank check above
            raise MultiReactionMixedExpansionError("multiset unranking failed")
    return tuple(value - position for position, value in enumerate(strict))


def _candidate_fold_pools(
    components: Iterable[_Component],
    *,
    program_id: str,
    fold: str,
) -> tuple[list[_Component], list[_Component]]:
    allowed_folds = {
        "train": {"train"},
        "calibration": {"train", "calibration"},
        "heldout": set(FOLDS),
    }[fold]
    heads = sorted(
        (
            value
            for value in components
            if value.program_id == program_id
            and value.program_role == "accumulator"
            and value.step_count >= 2
            and value.family_fold in allowed_folds
        ),
        key=lambda value: value.component_id,
    )
    repeats = sorted(
        (
            value
            for value in components
            if value.program_id == program_id
            and value.program_role == "repeat"
            and value.family_fold in allowed_folds
        ),
        key=lambda value: value.component_id,
    )
    if not heads or len(repeats) < 2:
        raise MultiReactionMixedExpansionError(
            f"{program_id}/{fold} has insufficient mixed-repeat support"
        )
    return heads, repeats


def _requests(
    *,
    seed: int,
    program_id: str,
    fold: str,
    count: int,
    components: Iterable[_Component],
    maximum_counter: int,
    start_index: int,
) -> list[_Request]:
    heads, repeats = _candidate_fold_pools(components, program_id=program_id, fold=fold)
    requests: list[_Request] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    counter = 0
    while len(requests) < count and counter < maximum_counter:
        head = heads[_counter_integer(seed, program_id, fold, str(counter), "head") % len(heads)]
        total = math.comb(len(repeats) + head.step_count - 1, head.step_count)
        rank = _counter_integer(seed, program_id, fold, str(counter), "repeat") % total
        repeat_indices = _unrank_combination(len(repeats), head.step_count, rank)
        selected = tuple(repeats[index] for index in repeat_indices)
        counter += 1
        repeat_ids = tuple(value.component_id for value in selected)
        if len(set(repeat_ids)) < 2:
            continue
        observed_fold = product_fold(
            (head.family_fold, *(value.family_fold for value in selected)),
            error=MultiReactionMixedExpansionError,
        )
        if observed_fold != fold:
            continue
        key = (head.component_id, repeat_ids)
        if key in seen:
            continue
        seen.add(key)
        attempt_index = start_index + len(requests)
        requests.append(
            _Request(
                attempt_index=attempt_index,
                attempt_id=_identifier(
                    "mrm",
                    program_id,
                    head.component_id,
                    *repeat_ids,
                ),
                program_id=program_id,
                requested_fold=fold,
                head=head,
                repeats=selected,
            )
        )
    if len(requests) != count:
        raise MultiReactionMixedExpansionError(
            f"{program_id}/{fold} produced {len(requests)} of {count} requested candidates "
            f"within maximum_counter={maximum_counter}"
        )
    return requests


def _initialize_worker(
    registry_path: str,
    registry_sha256: str,
    raw_specs: Sequence[Mapping[str, Any]],
    policy: Mapping[str, Any],
) -> None:
    global _WORKER_ADAPTERS, _WORKER_POLICY
    specs = {
        str(raw["program_id"]): ReactionProgramSpec(
            program_id=str(raw["program_id"]),
            reaction_id=str(raw["reaction_id"]),
            accumulator_role=str(raw["accumulator_role"]),
            repeat_role=str(raw["repeat_role"]),
            minimum_steps=int(raw["minimum_steps"]),
            maximum_steps=int(raw["maximum_steps"]),
        )
        for raw in raw_specs
    }
    registry = Path(registry_path)
    _WORKER_ADAPTERS = {
        program_id: RegistryRepeatedReactionProgram.from_registry(
            registry,
            spec,
            expected_sha256=registry_sha256,
        )
        for program_id, spec in specs.items()
    }
    _WORKER_POLICY = dict(policy)


def _atom_state_key(atom: Chem.Atom) -> tuple[str, int, int, int]:
    return (
        atom.GetSymbol(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
        atom.GetNumExplicitHs(),
    )


def _model_support_reason(molecule: Chem.Mol, policy: Mapping[str, Any]) -> str:
    allowed_elements = set(str(value) for value in policy["allowed_elements"])
    if (
        molecule.GetNumHeavyAtoms() > int(policy["maximum_product_heavy_atoms"])
        or not {atom.GetSymbol() for atom in molecule.GetAtoms()}.issubset(allowed_elements)
        or any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    ):
        return "forward_product_outside_declared_model_support"
    raw_states = policy.get("_allowed_atom_states")
    if raw_states is not None:
        allowed_states = {tuple(value) for value in raw_states}
        if any(_atom_state_key(atom) not in allowed_states for atom in molecule.GetAtoms()):
            return "forward_product_outside_pinned_atom_vocabulary"
        closure_count = molecule.GetNumBonds() - molecule.GetNumAtoms() + 1
        if closure_count > int(policy["maximum_product_closures"]):
            return "forward_product_exceeds_declared_closure_support"
    return ""


def _screen_request(request: _Request) -> _ForwardResult:
    adapter = _WORKER_ADAPTERS[request.program_id]
    try:
        traces = adapter.forward_traces(
            request.head.canonical_smiles,
            tuple(value.canonical_smiles for value in request.repeats),
            maximum_outcomes=int(_WORKER_POLICY["maximum_forward_outcomes"]),
            maximum_states=int(_WORKER_POLICY["maximum_forward_states"]),
        )
    except ReactionProgramError as exc:
        return _ForwardResult(request, "abstain", f"forward_enumeration_error:{exc}", 0, "", ())
    if len(traces) != 1:
        return _ForwardResult(
            request,
            "abstain",
            "forward_program_did_not_resolve_one_product",
            len(traces),
            "",
            (),
        )
    trace = traces[0]
    product_smiles = trace.intermediate_product_smiles[-1]
    normalized = _canonical(product_smiles)
    if normalized is None or normalized[0] != product_smiles:
        return _ForwardResult(
            request,
            "abstain",
            "forward_product_failed_constitutional_normalization",
            1,
            product_smiles,
            (),
        )
    molecule = normalized[1]
    support_reason = _model_support_reason(molecule, _WORKER_POLICY)
    if support_reason:
        return _ForwardResult(
            request,
            "abstain",
            support_reason,
            1,
            product_smiles,
            (),
        )
    return _ForwardResult(
        request,
        "candidate",
        "",
        1,
        product_smiles,
        trace.intermediate_product_smiles,
    )


def _origin_result(
    result: _ForwardResult,
) -> tuple[_ForwardResult, tuple[str, ...], tuple[str, ...]]:
    adapter = _WORKER_ADAPTERS[result.request.program_id]
    trace = ReactionProgramTrace(
        program_id=result.request.program_id,
        reaction_id=adapter.spec.reaction_id,
        terminal_head_smiles=result.request.head.canonical_smiles,
        repeated_component_smiles=tuple(value.canonical_smiles for value in result.request.repeats),
        intermediate_product_smiles=result.intermediate_products,
    )
    try:
        origins = adapter.atom_origins(trace)
    except ReactionProgramError as exc:
        return result, (), (f"semantic_origin_error:{exc}",)
    if origins.canonical_product_smiles != result.product_smiles:
        return result, (), ("semantic_origin_error:canonical_product_changed",)
    return result, origins.atom_origins, origins.core_positions


def _attempt_row(
    result: _ForwardResult,
    *,
    disposition: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    request = result.request
    return {
        "attempt_index": request.attempt_index,
        "attempt_id": request.attempt_id,
        "program_id": request.program_id,
        "requested_fold": request.requested_fold,
        "head_component_id": request.head.component_id,
        "repeat_component_ids_json": _json(tuple(value.component_id for value in request.repeats)),
        "step_count": len(request.repeats),
        "disposition": disposition if disposition is not None else result.disposition,
        "forward_products": result.forward_products,
        "canonical_product_smiles": result.product_smiles,
        "reason": reason if reason is not None else result.reason,
    }


def _encode_origins(
    origins: Sequence[str],
    core_positions: Sequence[str],
) -> tuple[bytes, bytes]:
    if len(origins) != len(core_positions) or not origins:
        raise MultiReactionMixedExpansionError("selected product has incomplete semantic origins")
    allowed_origins = {"accumulator", "repeat"}
    if set(origins) - allowed_origins:
        raise MultiReactionMixedExpansionError("selected product has an unknown precursor role")
    origin_codes = bytes(1 if value == "repeat" else 0 for value in origins)
    encoded_core: list[int] = []
    for value in core_positions:
        if not value:
            encoded_core.append(0)
        elif value.startswith("map_") and value[4:].isdigit() and 1 <= int(value[4:]) <= 255:
            encoded_core.append(int(value[4:]))
        else:
            raise MultiReactionMixedExpansionError(f"unsupported core position: {value!r}")
    return origin_codes, bytes(encoded_core)


def _evaluate_group(
    screened: Sequence[_ForwardResult],
    *,
    quota: int,
    existing_products: set[tuple[str, str]],
    globally_ambiguous_products: set[tuple[str, str]],
    executor: ProcessPoolExecutor | None,
    worker_chunksize: int,
    origin_batch_size: int,
) -> tuple[list[_SelectedProduct], list[dict[str, Any]], dict[str, Any]]:
    product_candidates: dict[str, list[_ForwardResult]] = defaultdict(list)
    attempts_by_index: dict[int, dict[str, Any]] = {}
    for result in screened:
        attempts_by_index[result.request.attempt_index] = _attempt_row(result)
        if result.disposition != "candidate":
            continue
        key = (result.request.program_id, result.product_smiles)
        if key in existing_products:
            attempts_by_index[result.request.attempt_index] = _attempt_row(
                result,
                disposition="existing_product_precedence",
                reason="mixed product duplicates v1 corpus product",
            )
            continue
        if key in globally_ambiguous_products:
            attempts_by_index[result.request.attempt_index] = _attempt_row(
                result,
                disposition="abstain",
                reason="multiple_component_factorizations_in_enumerated_request_set",
            )
            continue
        product_candidates[result.product_smiles].append(result)

    eligible: list[_ForwardResult] = []
    for product_smiles, candidates in sorted(product_candidates.items()):
        component_programs = {
            (
                value.request.head.component_id,
                tuple(component.component_id for component in value.request.repeats),
            )
            for value in candidates
        }
        if len(component_programs) != 1:
            for result in candidates:
                attempts_by_index[result.request.attempt_index] = _attempt_row(
                    result,
                    disposition="abstain",
                    reason="multiple_component_factorizations_in_enumerated_request_set",
                )
            continue
        eligible.append(min(candidates, key=lambda value: value.request.attempt_index))
        for duplicate in sorted(candidates, key=lambda value: value.request.attempt_index)[1:]:
            attempts_by_index[duplicate.request.attempt_index] = _attempt_row(
                duplicate,
                disposition="duplicate",
                reason="duplicate component/product enumeration",
            )
    eligible.sort(key=lambda value: value.request.attempt_index)

    selected: list[_SelectedProduct] = []
    candidate_offset = 0
    while len(selected) < quota and candidate_offset < len(eligible):
        batch = eligible[candidate_offset : candidate_offset + origin_batch_size]
        candidate_offset += len(batch)
        origin_results = list(
            executor.map(_origin_result, batch, chunksize=worker_chunksize)
            if executor is not None
            else map(_origin_result, batch)
        )
        for result, origins, core_positions in origin_results:
            if not origins:
                attempts_by_index[result.request.attempt_index] = _attempt_row(
                    result,
                    disposition="abstain",
                    reason=core_positions[0],
                )
                continue
            if len(selected) >= quota:
                attempts_by_index[result.request.attempt_index] = _attempt_row(
                    result,
                    disposition="quota_not_selected",
                    reason="deterministic fold quota already filled",
                )
                continue
            origin_codes, core_codes = _encode_origins(origins, core_positions)
            record_id = _identifier(
                "mrx",
                result.request.program_id,
                result.product_smiles,
            )
            selected.append(
                _SelectedProduct(
                    record_id=record_id,
                    request=result.request,
                    product_smiles=result.product_smiles,
                    intermediate_products=result.intermediate_products,
                    origin_codes=origin_codes,
                    core_position_codes=core_codes,
                )
            )
            attempts_by_index[result.request.attempt_index] = _attempt_row(
                result,
                disposition="admit_mixed_transform_consistency",
            )
    if len(selected) != quota:
        raise MultiReactionMixedExpansionError(
            f"{screened[0].request.program_id}/{screened[0].request.requested_fold} admitted "
            f"{len(selected)} of {quota} required mixed products"
        )
    selected_indices = {value.request.attempt_index for value in selected}
    for result in eligible:
        index = result.request.attempt_index
        if index not in selected_indices and attempts_by_index[index]["disposition"] == "candidate":
            attempts_by_index[index] = _attempt_row(
                result,
                disposition="quota_not_selected",
                reason="outside deterministic fold quota",
            )
    summary = {
        "requested_candidates": len(screened),
        "selected_products": len(selected),
        "attempt_dispositions": dict(
            Counter(str(value["disposition"]) for value in attempts_by_index.values())
        ),
        "attempt_reasons": dict(
            Counter(str(value["reason"]) for value in attempts_by_index.values() if value["reason"])
        ),
    }
    return selected, [attempts_by_index[index] for index in sorted(attempts_by_index)], summary


def _existing_atlas_rows(
    path: Path,
    *,
    policy: Mapping[str, Any],
) -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, str]],
    list[dict[str, Any]],
]:
    rows = read_csv_rows(
        path,
        error=MultiReactionMixedExpansionError,
        label="v1 reaction-program atlas",
        required_fields=ATLAS_FIELDS,
    )
    admitted = [
        row
        for row in rows
        if row["disposition"] in {"admit_exact", "admit_transform_consistency"}
        and row["semantic_origin_status"] == "exact"
    ]
    if len({row["record_id"] for row in admitted}) != len(admitted):
        raise MultiReactionMixedExpansionError("v1 admitted atlas has duplicate record ids")
    if "_allowed_atom_states" not in policy:
        return admitted, {row["record_id"]: row for row in admitted}, []
    supported: list[dict[str, str]] = []
    exclusions: list[dict[str, Any]] = []
    for row in admitted:
        normalized = _canonical(row["canonical_product_smiles"])
        if normalized is None:
            raise MultiReactionMixedExpansionError(
                f"existing admitted product is no longer canonical: {row['record_id']}"
            )
        molecule = normalized[1]
        reason = _model_support_reason(molecule, policy)
        if not reason:
            supported.append(row)
            continue
        if row["disposition"] == "admit_exact":
            raise MultiReactionMixedExpansionError(
                f"source-executed product lies outside frozen model support: {row['record_id']}"
            )
        exclusions.append(
            {
                "record_id": row["record_id"],
                "program_id": row["program_id"],
                "evidence_basis": row["evidence_basis"],
                "reason": reason,
                "closure_count": molecule.GetNumBonds() - molecule.GetNumAtoms() + 1,
            }
        )
    return supported, {row["record_id"]: row for row in supported}, exclusions


def _new_atlas_row(product: _SelectedProduct, spec: ReactionProgramSpec) -> dict[str, Any]:
    return {
        "record_id": product.record_id,
        "program_id": product.request.program_id,
        "reaction_id": spec.reaction_id,
        "source_study": "LNPDB_COMPONENT_MIXED_ENUMERATION",
        "source_product_labels": "",
        "canonical_product_smiles": product.product_smiles,
        "terminal_head_smiles": product.request.head.canonical_smiles,
        "repeat_component_smiles": "",
        "step_count": len(product.request.repeats),
        "source_row_count": 0,
        "evidence_basis": MIXED_STRATUM,
        "disposition": "admit_transform_consistency",
        "abstention_reason": "",
        "exact_forward_roundtrip": "true",
        "source_locator": "source-linked LNPDB components plus frozen registry transform",
        "semantic_origin_status": "exact",
        "semantic_origin_reason": "",
        "repeat_component_smiles_json": _json(
            tuple(value.canonical_smiles for value in product.request.repeats)
        ),
    }


def _new_step_rows(
    product: _SelectedProduct, spec: ReactionProgramSpec
) -> Iterator[dict[str, Any]]:
    accumulator = product.request.head.canonical_smiles
    for step_index, (repeat, step_product) in enumerate(
        zip(product.request.repeats, product.intermediate_products, strict=True),
        start=1,
    ):
        yield {
            "record_id": product.record_id,
            "program_id": product.request.program_id,
            "step_index": step_index,
            "reaction_id": spec.reaction_id,
            "accumulator_role": spec.accumulator_role,
            "accumulator_input_smiles": accumulator,
            "repeat_role": spec.repeat_role,
            "repeat_component_smiles": repeat.canonical_smiles,
            "product_smiles": step_product,
            "exact_forward_roundtrip": "true",
        }
        accumulator = step_product


def _new_semantic_rows(
    product: _SelectedProduct, spec: ReactionProgramSpec
) -> Iterator[dict[str, Any]]:
    for atom_index, (origin_code, core_code) in enumerate(
        zip(product.origin_codes, product.core_position_codes, strict=True)
    ):
        yield {
            "record_id": product.record_id,
            "program_id": product.request.program_id,
            "atom_index": atom_index,
            "origin_role": spec.repeat_role if origin_code else spec.accumulator_role,
            "core_position": f"map_{core_code}" if core_code else "",
            "program_depth": len(product.request.repeats),
        }


def _existing_v2_atlas_rows(rows: Iterable[Mapping[str, str]]) -> Iterator[dict[str, Any]]:
    for row in rows:
        yield {
            **row,
            "repeat_component_smiles_json": _json(
                (row["repeat_component_smiles"],) * int(row["step_count"])
            ),
        }


def _raw_mixed_weight(product: _SelectedProduct) -> float:
    distinct_repeats = {value.component_id: value for value in product.request.repeats}.values()
    denominator = product.request.head.family_size
    for component in distinct_repeats:
        denominator *= component.family_size
    return 1.0 / denominator


def _split_rows(
    *,
    existing_splits_path: Path,
    existing_atlas: Mapping[str, Mapping[str, str]],
    components: Mapping[str, _Component],
    selected: Sequence[_SelectedProduct],
    policy: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for old in iter_csv(existing_splits_path):
        record_id = old["record_id"]
        if record_id not in existing_atlas:
            continue
        atlas = existing_atlas[record_id]
        head = components[old["head_component_id"]]
        repeat = components[old["repeat_component_id"]]
        depth = int(atlas["step_count"])
        stratum = SOURCE_STRATUM if atlas["disposition"] == "admit_exact" else HOMOGENEOUS_STRATUM
        rows.append(
            {
                "record_id": record_id,
                "program_id": old["program_id"],
                "head_component_id": head.component_id,
                "head_component_family_id": head.family_id,
                "head_component_fold": head.family_fold,
                "repeat_component_ids_json": _json((repeat.component_id,) * depth),
                "repeat_component_family_ids_json": _json((repeat.family_id,) * depth),
                "repeat_component_folds_json": _json((repeat.family_fold,) * depth),
                "product_fold": old["product_fold"],
                "evidence_stratum": stratum,
                "family_balance_weight_raw": float(old["family_balance_weight_raw"]),
                "source_balanced_weight": 0.0,
            }
        )
    for product in selected:
        rows.append(
            {
                "record_id": product.record_id,
                "program_id": product.request.program_id,
                "head_component_id": product.request.head.component_id,
                "head_component_family_id": product.request.head.family_id,
                "head_component_fold": product.request.head.family_fold,
                "repeat_component_ids_json": _json(
                    tuple(value.component_id for value in product.request.repeats)
                ),
                "repeat_component_family_ids_json": _json(
                    tuple(value.family_id for value in product.request.repeats)
                ),
                "repeat_component_folds_json": _json(
                    tuple(value.family_fold for value in product.request.repeats)
                ),
                "product_fold": product.request.requested_fold,
                "evidence_stratum": MIXED_STRATUM,
                "family_balance_weight_raw": _raw_mixed_weight(product),
                "source_balanced_weight": 0.0,
            }
        )
    if len({str(row["record_id"]) for row in rows}) != len(rows):
        raise MultiReactionMixedExpansionError("v2 split rows have duplicate records")
    raw_mass: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    for row in rows:
        raw_mass[(row["product_fold"], row["program_id"], row["evidence_stratum"])] += float(
            row["family_balance_weight_raw"]
        )
    expected = {
        (fold, program_id, stratum)
        for fold in FOLDS
        for program_id in {str(row["program_id"]) for row in rows}
        for stratum in EVIDENCE_STRATA
    }
    if set(raw_mass) != expected or any(value <= 0.0 for value in raw_mass.values()):
        raise MultiReactionMixedExpansionError("v2 fold/program/evidence support is incomplete")
    masses = policy["evidence_stratum_mass"]
    for row in rows:
        group = (row["product_fold"], row["program_id"], row["evidence_stratum"])
        row["source_balanced_weight"] = (
            float(masses[row["evidence_stratum"]])
            * float(row["family_balance_weight_raw"])
            / raw_mass[group]
        )
        row["family_balance_weight_raw"] = f"{float(row['family_balance_weight_raw']):.12g}"
        row["source_balanced_weight"] = f"{float(row['source_balanced_weight']):.12g}"

    observed_mass: defaultdict[tuple[str, str, str], float] = defaultdict(float)
    for row in rows:
        observed_mass[(row["product_fold"], row["program_id"], row["evidence_stratum"])] += float(
            row["source_balanced_weight"]
        )
    for (_, _, stratum), value in observed_mass.items():
        if abs(value - float(masses[stratum])) > 1e-8:
            raise MultiReactionMixedExpansionError("v2 evidence-stratum mass is not balanced")
    summary = {
        "product_folds": dict(Counter(str(row["product_fold"]) for row in rows)),
        "products_by_program_fold": dict(
            Counter(f"{row['program_id']}|{row['product_fold']}" for row in rows)
        ),
        "products_by_evidence": dict(Counter(str(row["evidence_stratum"]) for row in rows)),
        "sampling_policy": (
            "equal program mass; 0.5 source, 0.25 homogeneous computed, 0.25 mixed computed; "
            "inverse component-family size"
        ),
    }
    return rows, summary


def build_multireaction_mixed_expansion(
    config_path: Path,
    repo: Path,
    *,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Build and persist an exposure-balanced mixed-repeat BL/LX corpus."""

    required_outputs = {
        "components",
        "attempts",
        "atlas",
        "steps",
        "semantic_atoms",
        "provenance",
        "splits",
        "manifest",
        "result",
    }
    allowed_outputs = (required_outputs, {*required_outputs, "support_exclusions"})
    if set(outputs) not in allowed_outputs:
        raise MultiReactionMixedExpansionError(
            f"mixed-repeat outputs differ from contract: {sorted(set(outputs) ^ required_outputs)}"
        )
    config, paths = _load_config(config_path, repo)
    worker_policy = dict(config["policy"])
    if "atom_vocabulary" in paths:
        worker_policy["_allowed_atom_states"] = [
            list(state.key()) for state in load_atom_vocabulary(paths["atom_vocabulary"])
        ]
    specs = _program_specs(paths["program_config"])
    component_rows, components = _components(paths["component_registry"], specs)
    existing_rows, existing_atlas, support_exclusions = _existing_atlas_rows(
        paths["homogeneous_atlas"],
        policy=worker_policy,
    )
    existing_products = {
        (row["program_id"], row["canonical_product_smiles"]) for row in existing_rows
    }
    targets = config.get("target_products_by_program_fold")
    if not isinstance(targets, dict) or set(targets) != set(specs):
        raise MultiReactionMixedExpansionError("mixed-repeat program/fold targets are incomplete")
    current_counts = Counter(
        (row["program_id"], row["product_fold"])
        for row in iter_csv(paths["homogeneous_splits"])
        if row["record_id"] in existing_atlas
    )
    quotas: dict[tuple[str, str], int] = {}
    for program_id in sorted(specs):
        raw = targets[program_id]
        if not isinstance(raw, dict) or tuple(raw) != FOLDS:
            raise MultiReactionMixedExpansionError(f"{program_id} target folds changed")
        for fold in FOLDS:
            target = int(raw[fold])
            current = current_counts[(program_id, fold)]
            if target <= current:
                raise MultiReactionMixedExpansionError(
                    f"{program_id}/{fold} target {target} does not expand v1 count {current}"
                )
            quotas[(program_id, fold)] = target - current

    registry_sha = str(sha256_file(paths["qualified_reaction_families"]))
    raw_specs = [
        {
            "program_id": spec.program_id,
            "reaction_id": spec.reaction_id,
            "accumulator_role": spec.accumulator_role,
            "repeat_role": spec.repeat_role,
            "minimum_steps": spec.minimum_steps,
            "maximum_steps": spec.maximum_steps,
        }
        for spec in specs.values()
    ]
    execution = config["execution"]
    workers = int(execution["workers"])
    if workers == 0:
        _initialize_worker(
            str(paths["qualified_reaction_families"]),
            registry_sha,
            raw_specs,
            worker_policy,
        )
        executor_context: ProcessPoolExecutor | None = None
    else:
        executor_context = ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_worker,
            initargs=(
                str(paths["qualified_reaction_families"]),
                registry_sha,
                raw_specs,
                worker_policy,
            ),
        )

    group_requests: dict[tuple[str, str], list[_Request]] = {}
    attempt_offset = 0
    group_order = ("heldout", "calibration", "train")
    for program_id in sorted(specs):
        for fold in group_order:
            quota = quotas[(program_id, fold)]
            candidate_count = math.ceil(
                quota * float(execution["candidate_multiplier_by_program_fold"][program_id][fold])
            )
            requests = _requests(
                seed=int(config["seed"]),
                program_id=program_id,
                fold=fold,
                count=candidate_count,
                components=components.values(),
                maximum_counter=int(execution["maximum_candidate_counter"]),
                start_index=attempt_offset,
            )
            attempt_offset += len(requests)
            group_requests[(program_id, fold)] = requests

    group_screened: dict[tuple[str, str], list[_ForwardResult]] = {}
    try:
        for program_id in sorted(specs):
            for fold in group_order:
                requests = group_requests[(program_id, fold)]
                group_screened[(program_id, fold)] = list(
                    executor_context.map(
                        _screen_request,
                        requests,
                        chunksize=int(execution["worker_chunksize"]),
                    )
                    if executor_context is not None
                    else map(_screen_request, requests)
                )

        product_factorizations: dict[tuple[str, str], set[tuple[str, tuple[str, ...]]]] = (
            defaultdict(set)
        )
        for screened in group_screened.values():
            for value in screened:
                if value.disposition != "candidate":
                    continue
                key = (value.request.program_id, value.product_smiles)
                if key in existing_products:
                    continue
                product_factorizations[key].add(
                    (
                        value.request.head.component_id,
                        tuple(component.component_id for component in value.request.repeats),
                    )
                )
        globally_ambiguous_products = {
            key for key, factorizations in product_factorizations.items() if len(factorizations) > 1
        }

        selected: list[_SelectedProduct] = []
        attempts: list[dict[str, Any]] = []
        group_summaries: dict[str, Any] = {}
        for program_id in sorted(specs):
            for fold in group_order:
                group_selected, group_attempts, group_summary = _evaluate_group(
                    group_screened[(program_id, fold)],
                    quota=quotas[(program_id, fold)],
                    existing_products=existing_products,
                    globally_ambiguous_products=globally_ambiguous_products,
                    executor=executor_context,
                    worker_chunksize=int(execution["worker_chunksize"]),
                    origin_batch_size=int(execution["origin_batch_size"]),
                )
                for product in group_selected:
                    key = (product.request.program_id, product.product_smiles)
                    if key in existing_products:
                        raise MultiReactionMixedExpansionError(
                            "selected mixed products overlap an earlier corpus product"
                        )
                    existing_products.add(key)
                selected.extend(group_selected)
                attempts.extend(group_attempts)
                group_summaries[f"{program_id}|{fold}"] = group_summary
    finally:
        if executor_context is not None:
            executor_context.shutdown(wait=True, cancel_futures=True)

    split_rows, split_summary = _split_rows(
        existing_splits_path=paths["homogeneous_splits"],
        existing_atlas=existing_atlas,
        components=components,
        selected=selected,
        policy=config["policy"],
    )
    final_counts = Counter((row["program_id"], row["product_fold"]) for row in split_rows)
    for program_id in sorted(specs):
        for fold in FOLDS:
            if final_counts[(program_id, fold)] != int(targets[program_id][fold]):
                raise MultiReactionMixedExpansionError(
                    f"{program_id}/{fold} final target changed after enumeration"
                )

    selected.sort(
        key=lambda value: (
            value.request.program_id,
            value.request.requested_fold,
            value.request.attempt_index,
        )
    )
    atomic_write(outputs["components"], paths["component_registry"].read_bytes())
    write_csv_iter(outputs["attempts"], attempts, ATTEMPT_FIELDS)
    write_csv_iter(
        outputs["atlas"],
        chain(
            _existing_v2_atlas_rows(existing_rows),
            (_new_atlas_row(product, specs[product.request.program_id]) for product in selected),
        ),
        ATLAS_FIELDS_V2,
    )
    write_csv_iter(
        outputs["steps"],
        chain(
            (
                row
                for row in iter_csv(paths["homogeneous_steps"])
                if row["record_id"] in existing_atlas
            ),
            (
                row
                for product in selected
                for row in _new_step_rows(product, specs[product.request.program_id])
            ),
        ),
        STEP_FIELDS,
    )
    write_csv_iter(
        outputs["semantic_atoms"],
        chain(
            (
                row
                for row in iter_csv(paths["homogeneous_semantic_atoms"])
                if row["record_id"] in existing_atlas
            ),
            (
                row
                for product in selected
                for row in _new_semantic_rows(product, specs[product.request.program_id])
            ),
        ),
        SEMANTIC_ATOM_FIELDS,
    )
    atomic_write(outputs["provenance"], paths["homogeneous_provenance"].read_bytes())
    write_csv_iter(outputs["splits"], split_rows, SPLIT_FIELDS)
    if "support_exclusions" in outputs:
        write_csv_iter(
            outputs["support_exclusions"],
            support_exclusions,
            SUPPORT_EXCLUSION_FIELDS,
        )

    old_result = read_json_object(
        paths["homogeneous_result"],
        error=MultiReactionMixedExpansionError,
        label="v1 expansion result",
    )
    if old_result.get("status") != "complete_bl_lx_reaction_enumerated_expansion":
        raise MultiReactionMixedExpansionError("homogeneous expansion result is not complete")
    existing_steps = sum(
        1
        for row in iter_csv(paths["homogeneous_steps"])
        if row["record_id"] in existing_atlas
    )
    existing_atoms = sum(
        1
        for row in iter_csv(paths["homogeneous_semantic_atoms"])
        if row["record_id"] in existing_atlas
    )
    new_steps = sum(len(value.request.repeats) for value in selected)
    new_atoms = sum(len(value.origin_codes) for value in selected)
    artifact_schemas = {
        "components": "forge.multireaction_expansion_components.v1",
        "attempts": ATTEMPT_SCHEMA,
        "atlas": ATLAS_SCHEMA,
        "steps": "forge.multireaction_program_steps.v1",
        "semantic_atoms": "forge.multireaction_semantic_atoms.v2",
        "provenance": "forge.multireaction_source_provenance.v1",
        "splits": SPLITS_SCHEMA,
        "support_exclusions": SUPPORT_EXCLUSIONS_SCHEMA,
    }
    artifacts = {
        label: {
            **artifact_record(path, logical_path=path.name),
            "schema_version": artifact_schemas[label],
        }
        for label, path in sorted(outputs.items())
        if label in artifact_schemas
    }
    write_json(outputs["manifest"], {"schema_version": MANIFEST_SCHEMA, "artifacts": artifacts})
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "complete_bl_lx_mixed_repeat_expansion",
        "task": config["task"],
        "seed": int(config["seed"]),
        "config": pin_record(config_path, repo),
        "inputs": {label: pin_record(path, repo) for label, path in sorted(paths.items())},
        "policy": config["policy"],
        "execution": config["execution"],
        "summary": {
            **split_summary,
            "v1_products_preserved": len(existing_rows),
            "homogeneous_model_support_exclusions": len(support_exclusions),
            "homogeneous_model_support_exclusion_reasons": dict(
                Counter(str(row["reason"]) for row in support_exclusions)
            ),
            "mixed_products_admitted": len(selected),
            "total_products": len(split_rows),
            "mixed_enumeration_attempts": len(attempts),
            "exact_program_steps": existing_steps + new_steps,
            "semantic_atom_rows": existing_atoms + new_atoms,
            "group_summaries": group_summaries,
        },
        "claims_boundary": {
            "source_executed_evidence_rewritten": False,
            "homogeneous_v1_products_rewritten": False,
            "unsupported_homogeneous_products_excluded": bool(support_exclusions),
            "mixed_transform_consistency_is_observed_synthesis": False,
            "mixed_transform_consistency_is_synthesis_success": False,
            "mixed_transform_consistency_is_route_closure": False,
            "source_activity_labels_inherited": False,
            "component_family_assignment_precedes_enumeration": True,
            "reductive_amination_substructure_hit_rate_reported": False,
            "training_must_use_source_balanced_weight": True,
        },
        "artifacts": artifacts,
    }
    write_json(outputs["result"], result)
    return result


__all__ = [
    "ATLAS_FIELDS_V2",
    "ATTEMPT_FIELDS",
    "SPLIT_FIELDS",
    "MultiReactionMixedExpansionError",
    "build_multireaction_mixed_expansion",
]
