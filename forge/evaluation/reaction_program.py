"""Non-selecting metrics for generated reaction-program products."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence, Set
from copy import deepcopy
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from forge.assembly import (
    ReactionProgramError,
    ReactionProgramSpec,
    Ugi3AssemblyError,
)
from forge.core.io import iter_csv
from forge.corpus.reaction_program_records import repeat_component_smiles


class ReactionProgramEvaluationError(ValueError):
    """Generated rows or their reference populations violate the metric contract."""


def _canonical_component(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ReactionProgramEvaluationError(
            "training reference contains an invalid or disconnected molecular graph"
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


class _CanonicalMemo:
    """Reuse :func:`_canonical_component` per distinct source string within one reference build.

    Canonicalization has no state and no dependence on row order, and only accepted strings are
    stored, so this returns exactly what a fresh call would return and raises exactly where a
    fresh call would raise.
    """

    __slots__ = ("_values",)

    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    def __call__(self, smiles: str) -> str:
        canonical = self._values.get(smiles)
        if canonical is None:
            canonical = _canonical_component(smiles)
            self._values[smiles] = canonical
        return canonical


def load_reaction_program_training_references(
    *,
    ugi_assignments: Path,
    multireaction_atlas: Path,
    multireaction_splits: Path,
    repeated_program_specs: Mapping[str, ReactionProgramSpec],
    ugi_program_id: str,
    ugi_roles: Sequence[str],
) -> tuple[dict[str, set[str]], dict[str, dict[str, set[str]]]]:
    """Load train-fold product and component identities from the authoritative ledgers."""

    program_ids = {ugi_program_id, *repeated_program_specs}
    training_products: dict[str, set[str]] = {program_id: set() for program_id in program_ids}
    training_components: dict[str, dict[str, set[str]]] = {
        program_id: {} for program_id in program_ids
    }
    # These ledgers repeat a small component inventory across a large product table: 66,464
    # train-fold Ugi rows draw on 295 distinct precursor constitutions, and 60,000 multi-reaction
    # rows on 219 terminal heads and 54 repeat components.  Canonicalization is a pure function of
    # the source string, so memoize it.  Only validated strings are stored, so an invalid one still
    # raises on the first row that carries it, exactly as it did before.
    canonical = _CanonicalMemo()
    for row in iter_csv(ugi_assignments):
        if row.get("primary_product_fold") != "train":
            continue
        training_products[ugi_program_id].add(canonical(row["canonical_product_smiles"]))
        for role in ugi_roles:
            training_components[ugi_program_id].setdefault(role, set()).add(
                canonical(row[f"{role}_smiles"])
            )

    split_programs: dict[str, str] = {}
    for row in iter_csv(multireaction_splits):
        record_id = row["record_id"]
        if record_id in split_programs:
            raise ReactionProgramEvaluationError(
                f"duplicate multi-reaction split record: {record_id}"
            )
        if row.get("product_fold") == "train":
            split_programs[record_id] = row["program_id"]
    observed: set[str] = set()
    for row in iter_csv(multireaction_atlas):
        record_id = row["record_id"]
        if record_id not in split_programs:
            continue
        program_id = row["program_id"]
        if split_programs[record_id] != program_id or program_id not in repeated_program_specs:
            raise ReactionProgramEvaluationError(
                f"multi-reaction training identity changed for {record_id}"
            )
        spec = repeated_program_specs[program_id]
        training_products[program_id].add(canonical(row["canonical_product_smiles"]))
        training_components[program_id].setdefault(spec.accumulator_role, set()).add(
            canonical(row["terminal_head_smiles"])
        )
        training_components[program_id].setdefault(spec.repeat_role, set()).update(
            canonical(smiles) for smiles in repeat_component_smiles(row)
        )
        observed.add(record_id)
    if observed != set(split_programs):
        raise ReactionProgramEvaluationError("multi-reaction atlas omits a train-fold split record")
    if any(not training_products[program_id] for program_id in program_ids):
        raise ReactionProgramEvaluationError("a reaction program has no train-fold products")
    if any(
        not roles or any(not values for values in roles.values())
        for roles in training_components.values()
    ):
        raise ReactionProgramEvaluationError("a reaction program has empty component support")
    return training_products, training_components


def adjudicate_reaction_program_rows(
    rows: list[dict[str, Any]],
    *,
    adapters: Mapping[str, Any],
    repeated_program_specs: Mapping[str, ReactionProgramSpec],
    ugi_program_id: str,
) -> None:
    """Attach exact reverse-decomposition and forward-replay evidence to complete products."""

    for row in rows:
        row.update(
            {
                "exact_l1_program": False,
                "exact_l1_trace_count": 0,
                "forward_verified_trace_count": 0,
                "exact_l1_traces": [],
                "decomposition_error": None,
            }
        )
        if row.get("valid") is not True:
            continue
        program_id = str(row["program_id"])
        if program_id not in adapters:
            raise ReactionProgramEvaluationError(f"missing assembly adapter for {program_id}")
        smiles = str(row["canonical_smiles"])
        try:
            traces = adapters[program_id].decompose(smiles)
        except (ReactionProgramError, Ugi3AssemblyError) as error:
            row["decomposition_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            continue
        if program_id == ugi_program_id:
            trace_rows = [{"components_by_role": trace.as_mapping()} for trace in traces]
        else:
            try:
                spec = repeated_program_specs[program_id]
            except KeyError as error:
                raise ReactionProgramEvaluationError(
                    f"missing repeated-program specification for {program_id}"
                ) from error
            trace_rows = [
                {
                    "terminal_head_smiles": trace.terminal_head_smiles,
                    "repeated_component_smiles": list(trace.repeated_component_smiles),
                    "components_by_role": {
                        spec.accumulator_role: trace.terminal_head_smiles,
                        spec.repeat_role: list(trace.repeated_component_smiles),
                    },
                }
                for trace in traces
            ]
        row["exact_l1_program"] = bool(trace_rows)
        row["exact_l1_trace_count"] = len(trace_rows)
        row["forward_verified_trace_count"] = len(trace_rows)
        row["exact_l1_traces"] = trace_rows


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def effective_count(values: Sequence[str]) -> float | None:
    """Return the exponential Shannon entropy of a categorical sample."""

    if not values:
        return None
    counts = Counter(values)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return math.exp(entropy)


def _mean_pairwise_distance(
    smiles_values: Sequence[str], molecules: Mapping[str, Any] | None = None
) -> float | None:
    """Mean pairwise ECFP4 distance over the unique constitutions in a sample.

    ``molecules`` lets a caller that has already parsed these exact SMILES hand the graphs over.
    A molecule is still required for every unique value, so a missing or unparseable one fails the
    same way it would on a fresh parse.
    """

    unique = sorted(set(smiles_values))
    if len(unique) < 2:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = []
    for smiles in unique:
        molecule = Chem.MolFromSmiles(smiles) if molecules is None else molecules.get(smiles)
        if molecule is None:
            raise ReactionProgramEvaluationError("valid sample failed metric parsing")
        fingerprints.append(generator.GetFingerprint(molecule))
    similarity_sum = 0.0
    pair_count = 0
    for left, fingerprint in enumerate(fingerprints[:-1]):
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints[left + 1 :])
        similarity_sum += float(sum(similarities))
        pair_count += len(similarities)
    return 1.0 - similarity_sum / pair_count


def _reference_union(values: Mapping[str, Set[str]], program_ids: Set[str]) -> set[str]:
    missing = sorted(program_ids - set(values))
    if missing:
        raise ReactionProgramEvaluationError(f"missing training references for programs: {missing}")
    return set().union(*(values[program_id] for program_id in sorted(program_ids)))


def _evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    training_products: Mapping[str, Set[str]],
    training_components: Mapping[str, Mapping[str, Set[str]]],
) -> dict[str, Any]:
    sample_count = len(rows)
    program_ids = {str(row["program_id"]) for row in rows}
    product_reference = _reference_union(training_products, program_ids)
    component_roles = sorted(
        {role for program_id in program_ids for role in training_components.get(program_id, {})}
    )
    component_reference: dict[str, set[str]] = {}
    for role in component_roles:
        per_program = {
            program_id: training_components.get(program_id, {}).get(role, set())
            for program_id in program_ids
        }
        component_reference[role] = _reference_union(per_program, program_ids)
    valid_rows = [row for row in rows if row.get("valid") is True]
    smiles_values = [str(row["canonical_smiles"]) for row in valid_rows]
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_values]
    if any(molecule is None for molecule in molecules):
        raise ReactionProgramEvaluationError("valid sample failed metric parsing")
    connected_count = sum(
        len(Chem.GetMolFrags(molecule)) == 1 for molecule in molecules if molecule is not None
    )
    exact_rows = [row for row in valid_rows if row.get("exact_l1_program") is True]
    try:
        trace_count = sum(int(row["exact_l1_trace_count"]) for row in valid_rows)
        verified_count = sum(int(row["forward_verified_trace_count"]) for row in valid_rows)
    except (KeyError, TypeError, ValueError) as error:
        raise ReactionProgramEvaluationError("sample trace accounting is malformed") from error
    if any(int(row["exact_l1_trace_count"]) < 1 for row in exact_rows):
        raise ReactionProgramEvaluationError("exact-L1 row has no decomposition trace")
    if any(
        int(row["exact_l1_trace_count"]) > 0
        for row in valid_rows
        if row.get("exact_l1_program") is not True
    ):
        raise ReactionProgramEvaluationError("decomposition trace is hidden by an exact-L1 flag")
    if verified_count != trace_count:
        raise ReactionProgramEvaluationError(
            "retro-decomposition emitted a trace that failed exact forward replay"
        )
    unique_trace_rows = [row for row in exact_rows if int(row["exact_l1_trace_count"]) == 1]
    novel_component_rows = 0
    open_ended_products: set[str] = set()
    open_ended_novel_products: set[str] = set()
    components_by_role: dict[str, list[str]] = {role: [] for role in component_roles}
    novel_by_role: Counter[str] = Counter()
    for row in unique_trace_rows:
        try:
            trace = row["exact_l1_traces"][0]
        except (IndexError, KeyError, TypeError) as error:
            raise ReactionProgramEvaluationError("exact-L1 trace payload is malformed") from error
        raw_components = trace.get("components_by_role")
        if raw_components is None:
            try:
                terminal_head = str(trace["terminal_head_smiles"])
                repeated = [str(value) for value in trace["repeated_component_smiles"]]
            except (KeyError, TypeError) as error:
                raise ReactionProgramEvaluationError(
                    "exact-L1 trace payload is malformed"
                ) from error
            if not terminal_head or not repeated or len(set(repeated)) != 1:
                raise ReactionProgramEvaluationError(
                    "unique repeated-component program trace is internally inconsistent"
                )
            normalized: dict[str, list[str]] = {
                "terminal_head": [terminal_head],
                "repeat_component": [repeated[0]],
            }
        else:
            if not isinstance(raw_components, Mapping) or not raw_components:
                raise ReactionProgramEvaluationError(
                    "components_by_role must be a nonempty mapping"
                )
            normalized = {}
            for role, value in raw_components.items():
                values = (
                    [str(item) for item in value]
                    if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
                    else [str(value)]
                )
                if not values or any(not item for item in values):
                    raise ReactionProgramEvaluationError(
                        "components_by_role contains an empty component"
                    )
                normalized[str(role)] = values
        row_is_novel = False
        for role, values in normalized.items():
            if role not in component_reference:
                raise ReactionProgramEvaluationError(
                    f"missing training component reference for role {role!r}"
                )
            components_by_role.setdefault(role, []).extend(values)
            novel = sum(value not in component_reference[role] for value in values)
            novel_by_role[role] += novel
            row_is_novel = row_is_novel or novel > 0
        if row_is_novel:
            novel_component_rows += 1
            canonical_smiles = str(row["canonical_smiles"])
            open_ended_products.add(canonical_smiles)
            if canonical_smiles not in product_reference:
                open_ended_novel_products.add(canonical_smiles)
    unique_valid = len(set(smiles_values))
    novel_products = sum(smiles not in product_reference for smiles in smiles_values)
    unique_exact_l1_products = len({str(row["canonical_smiles"]) for row in exact_rows})
    unique_novel_exact_l1_products = len(
        {
            str(row["canonical_smiles"])
            for row in exact_rows
            if str(row["canonical_smiles"]) not in product_reference
        }
    )
    component_values = [
        value for role in sorted(components_by_role) for value in components_by_role[role]
    ]
    result = {
        "samples": sample_count,
        "valid": len(valid_rows),
        "valid_fraction": _fraction(len(valid_rows), sample_count),
        "connected": connected_count,
        "connected_fraction": _fraction(connected_count, sample_count),
        "connected_among_valid_fraction": _fraction(connected_count, len(valid_rows)),
        "unique_valid": unique_valid,
        "unique_fraction": _fraction(unique_valid, len(valid_rows)),
        "whole_product_novel_to_train": novel_products,
        "whole_product_novel_to_train_fraction": _fraction(novel_products, len(valid_rows)),
        "mean_pairwise_ecfp4_distance": _mean_pairwise_distance(
            smiles_values, dict(zip(smiles_values, molecules, strict=True))
        ),
        "exact_l1_program": len(exact_rows),
        "exact_l1_yield_per_attempt": _fraction(len(exact_rows), sample_count),
        "unique_exact_l1_products": unique_exact_l1_products,
        "unique_exact_l1_products_per_attempt": _fraction(unique_exact_l1_products, sample_count),
        "unique_exact_l1_products_per_1000_attempts": (
            1000.0 * unique_exact_l1_products / sample_count if sample_count else None
        ),
        "unique_whole_product_novel_exact_l1_products": unique_novel_exact_l1_products,
        "unique_whole_product_novel_exact_l1_products_per_1000_attempts": (
            1000.0 * unique_novel_exact_l1_products / sample_count if sample_count else None
        ),
        "retro_decomposition_coverage_among_valid": _fraction(len(exact_rows), len(valid_rows)),
        "exact_forward_verified_traces": verified_count,
        "retro_transform_precision": _fraction(verified_count, trace_count),
        "ambiguous_exact_decompositions": sum(
            int(row["exact_l1_trace_count"]) > 1 for row in valid_rows
        ),
        "unique_decomposition_rows": len(unique_trace_rows),
        "decomposed_products_with_any_novel_component": novel_component_rows,
        "decomposed_products_with_any_novel_component_fraction": _fraction(
            novel_component_rows, len(unique_trace_rows)
        ),
        "component_novelty_fraction": _fraction(novel_component_rows, len(unique_trace_rows)),
        "unique_open_ended_exact_l1_products": len(open_ended_products),
        "unique_open_ended_exact_l1_products_per_attempt": _fraction(
            len(open_ended_products), sample_count
        ),
        "unique_open_ended_exact_l1_products_per_1000_attempts": (
            1000.0 * len(open_ended_products) / sample_count if sample_count else None
        ),
        "unique_open_ended_whole_product_novel_exact_l1_products": len(open_ended_novel_products),
        "unique_open_ended_whole_product_novel_exact_l1_products_per_1000_attempts": (
            1000.0 * len(open_ended_novel_products) / sample_count if sample_count else None
        ),
        "component_slots_evaluated": len(component_values),
        "effective_generated_component_count": effective_count(component_values),
        "effective_component_count": effective_count(component_values),
        "component_metrics_by_role": {
            role: {
                "slots": len(values),
                "novel": novel_by_role[role],
                "novel_fraction": _fraction(novel_by_role[role], len(values)),
                "effective_count": effective_count(values),
            }
            for role, values in sorted(components_by_role.items())
        },
        "invalid_samples": sample_count - len(valid_rows),
        "valid_without_exact_decomposition": len(valid_rows) - len(exact_rows),
        "invalid_or_unresolved_l1": sample_count - len(exact_rows),
    }
    # Preserve established field names for repeated-program callers while supporting arbitrary
    # Ugi precursor roles through the general mapping above.
    if "terminal_head" in components_by_role:
        values = components_by_role["terminal_head"]
        result.update(
            {
                "novel_terminal_heads": novel_by_role["terminal_head"],
                "novel_terminal_head_fraction": _fraction(
                    novel_by_role["terminal_head"], len(values)
                ),
                "effective_generated_terminal_head_count": effective_count(values),
            }
        )
    if "repeat_component" in components_by_role:
        values = components_by_role["repeat_component"]
        result.update(
            {
                "novel_repeat_components": novel_by_role["repeat_component"],
                "novel_repeat_component_fraction": _fraction(
                    novel_by_role["repeat_component"], len(values)
                ),
                "effective_generated_repeat_component_count": effective_count(values),
            }
        )
    return result


def evaluate_reaction_program_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    training_products: Mapping[str, Set[str]],
    training_components: Mapping[str, Mapping[str, Set[str]]],
) -> dict[str, Any]:
    """Report overall and separate family metrics without candidate selection."""

    if not rows:
        raise ReactionProgramEvaluationError("sample evaluation requires at least one row")
    program_ids = sorted({str(row["program_id"]) for row in rows})
    overall = _evaluate_rows(
        rows,
        training_products=training_products,
        training_components=training_components,
    )
    if len(program_ids) == 1 and all(row["program_id"] == program_ids[0] for row in rows):
        # A single-program ledger's per-program filter reproduces the overall row sequence exactly,
        # and `_evaluate_rows` is a pure function of that sequence and the frozen references, so the
        # second pass -- including its O(n^2) pairwise diversity -- can only recompute this result.
        per_program = {program_ids[0]: deepcopy(overall)}
    else:
        per_program = {
            program_id: _evaluate_rows(
                [row for row in rows if row["program_id"] == program_id],
                training_products=training_products,
                training_components=training_components,
            )
            for program_id in program_ids
        }
    return {
        "overall": overall,
        "per_program": per_program,
        "coverage_and_precision_reported": True,
        "reductive_amination_substructure_rate_reported": False,
    }


__all__ = [
    "ReactionProgramEvaluationError",
    "adjudicate_reaction_program_rows",
    "effective_count",
    "evaluate_reaction_program_samples",
    "load_reaction_program_training_references",
]
