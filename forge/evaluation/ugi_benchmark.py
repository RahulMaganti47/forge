"""Method-neutral attempt accounting and exact Ugi-L1 assessment.

The ledger retains parser failures, invalid graphs, and native sampler failures
in the attempt denominator. After the ledger is frozen, every method uses the
same registry-backed Ugi adapter.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, cast

from rdkit import Chem, rdBase

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.hashing import sha256_file
from forge.core.io import iter_csv, iter_jsonl, write_jsonl
from forge.evaluation.reaction_program import (
    adjudicate_reaction_program_rows,
    evaluate_reaction_program_samples,
)

ATTEMPT_SCHEMA = "forge.common_ugi_baseline_attempt.v1"
ASSESSED_ATTEMPT_SCHEMA = "forge.common_ugi_assessed_attempt.v1"
UGI_PROGRAM_ID = "ugi_3cr_agile"
ATTEMPT_STATUSES = frozenset({"generated", "invalid", "failed"})


class CommonUgiBenchmarkError(ValueError):
    """A method output violates the common Ugi comparison contract."""


@dataclass(frozen=True)
class CommonUgiAttempt:
    """One native generation attempt, including failures that emit no molecule."""

    method_id: str
    seed: int
    attempt_index: int
    status: Literal["generated", "invalid", "failed"]
    product_smiles: str | None
    method_visible_component_ids: tuple[str, ...]
    generator_calls: int
    reaction_calls: int
    route_calls: int
    oracle_calls: int
    wall_seconds: float

    @classmethod
    def from_mapping(cls, value: object) -> CommonUgiAttempt:
        required = {
            "method_id",
            "seed",
            "attempt_index",
            "status",
            "product_smiles",
            "method_visible_component_ids",
            "generator_calls",
            "reaction_calls",
            "route_calls",
            "oracle_calls",
            "wall_seconds",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise CommonUgiBenchmarkError(f"attempt must define exactly {sorted(required)}")
        method_id = value["method_id"]
        seed = value["seed"]
        index = value["attempt_index"]
        status = value["status"]
        product = value["product_smiles"]
        visible = value["method_visible_component_ids"]
        if not isinstance(method_id, str) or not method_id:
            raise CommonUgiBenchmarkError("attempt method_id must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise CommonUgiBenchmarkError("attempt seed must be a non-negative integer")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise CommonUgiBenchmarkError("attempt_index must be a non-negative integer")
        if status not in ATTEMPT_STATUSES:
            raise CommonUgiBenchmarkError(f"unsupported attempt status: {status!r}")
        if status == "generated":
            if not isinstance(product, str) or not product:
                raise CommonUgiBenchmarkError("generated attempt must contain product_smiles")
        elif product is not None:
            raise CommonUgiBenchmarkError("invalid or failed attempt must retain a null product")
        if (
            not isinstance(visible, Sequence)
            or isinstance(visible, (str, bytes))
            or any(not isinstance(item, str) or not item for item in visible)
            or len(set(visible)) != len(visible)
        ):
            raise CommonUgiBenchmarkError(
                "method_visible_component_ids must contain unique non-empty strings"
            )
        counts: dict[str, int] = {}
        for field in ("generator_calls", "reaction_calls", "route_calls", "oracle_calls"):
            raw = value[field]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise CommonUgiBenchmarkError(f"{field} must be a non-negative integer")
            counts[field] = raw
        wall = value["wall_seconds"]
        if isinstance(wall, bool) or not isinstance(wall, (int, float)):
            raise CommonUgiBenchmarkError("wall_seconds must be numeric")
        wall = float(wall)
        if not math.isfinite(wall) or wall < 0.0:
            raise CommonUgiBenchmarkError("wall_seconds must be finite and non-negative")
        return cls(
            method_id=method_id,
            seed=seed,
            attempt_index=index,
            status=cast(Literal["generated", "invalid", "failed"], status),
            product_smiles=product,
            method_visible_component_ids=tuple(sorted(visible)),
            wall_seconds=wall,
            **counts,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "status": self.status,
            "product_smiles": self.product_smiles,
            "method_visible_component_ids": list(self.method_visible_component_ids),
            "generator_calls": self.generator_calls,
            "reaction_calls": self.reaction_calls,
            "route_calls": self.route_calls,
            "oracle_calls": self.oracle_calls,
            "wall_seconds": self.wall_seconds,
        }


def validate_attempt_ledger(
    values: Sequence[object],
    *,
    expected_method: str | None = None,
    expected_seed: int | None = None,
    expected_attempts: int | None = None,
) -> tuple[CommonUgiAttempt, ...]:
    """Validate one method/seed ledger and its exact, gap-free attempt denominator."""

    attempts = tuple(CommonUgiAttempt.from_mapping(value) for value in values)
    if not attempts:
        raise CommonUgiBenchmarkError("attempt ledger is empty")
    methods = {attempt.method_id for attempt in attempts}
    seeds = {attempt.seed for attempt in attempts}
    if len(methods) != 1 or (expected_method is not None and methods != {expected_method}):
        raise CommonUgiBenchmarkError("attempt ledger mixes or changes method identity")
    if len(seeds) != 1 or (expected_seed is not None and seeds != {expected_seed}):
        raise CommonUgiBenchmarkError("attempt ledger mixes or changes seed identity")
    indices = [attempt.attempt_index for attempt in attempts]
    if indices != list(range(len(attempts))):
        raise CommonUgiBenchmarkError(
            "attempt ledger must be sorted and contain every index from zero exactly once"
        )
    if expected_attempts is not None and len(attempts) != expected_attempts:
        raise CommonUgiBenchmarkError(
            f"attempt denominator changed: expected {expected_attempts}, found {len(attempts)}"
        )
    return attempts


def load_attempt_ledger(
    path: Path,
    *,
    expected_method: str | None = None,
    expected_seed: int | None = None,
    expected_attempts: int | None = None,
) -> tuple[CommonUgiAttempt, ...]:
    """Load a canonical JSONL ledger, accepting one optional schema header."""

    records = list(iter_jsonl(path))
    if (
        records
        and isinstance(records[0], Mapping)
        and set(records[0])
        == {
            "schema_version",
            "rows",
        }
    ):
        header = records.pop(0)
        if header["schema_version"] != ATTEMPT_SCHEMA or header["rows"] != len(records):
            raise CommonUgiBenchmarkError("attempt ledger header is inconsistent")
    return validate_attempt_ledger(
        records,
        expected_method=expected_method,
        expected_seed=expected_seed,
        expected_attempts=expected_attempts,
    )


def write_attempt_ledger(path: Path, attempts: Sequence[CommonUgiAttempt]) -> None:
    """Write a deterministic canonical ledger with its row-count receipt."""

    checked = validate_attempt_ledger([attempt.to_mapping() for attempt in attempts])
    write_jsonl(
        path,
        [
            {"schema_version": ATTEMPT_SCHEMA, "rows": len(checked)},
            *(attempt.to_mapping() for attempt in checked),
        ],
    )


@lru_cache(maxsize=8)
def _load_ugi_identity_references_cached(
    assignments_path: Path,
    roles: tuple[str, ...],
    assignments_sha256: str,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Parse one immutable assignment ledger once per process and content digest."""

    # The digest is part of the cache key.  Keep the name explicit even though the caller has
    # already calculated it: omitting it would let in-process file replacement return stale
    # scientific references for the same path.
    del assignments_sha256

    training_products: set[str] = set()
    training_components: dict[str, set[str]] = {role: set() for role in roles}
    held_components: dict[str, set[str]] = {role: set() for role in roles}
    seen = 0
    # A component ledger repeats a small precursor inventory across every product row: the balanced
    # Ugi corpus carries 112,386 rows but only a few hundred distinct component constitutions.
    # Canonicalization is a pure function of the source string, so memoize it per role.  The key
    # keeps the role so that an invalid component still fails on the same row with the same role
    # named, and only validated components are ever stored.
    component_canonical: dict[tuple[str, str], str] = {}
    with rdBase.BlockLogs():
        for row in iter_csv(assignments_path):
            seen += 1
            product = Chem.MolFromSmiles(row["canonical_product_smiles"])
            if product is None:
                raise CommonUgiBenchmarkError("Ugi assignment contains an invalid product")
            train_row = row["primary_product_fold"] == "train"
            if train_row:
                training_products.add(
                    Chem.MolToSmiles(product, canonical=True, isomericSmiles=False)
                )
            for role in roles:
                source = row[f"{role}_smiles"]
                canonical = component_canonical.get((role, source))
                if canonical is None:
                    molecule = Chem.MolFromSmiles(source)
                    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                        raise CommonUgiBenchmarkError(
                            f"Ugi assignment contains an invalid {role} component"
                        )
                    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
                    component_canonical[(role, source)] = canonical
                if train_row:
                    training_components[role].add(canonical)
                if row[f"{role}_family_fold"] == "heldout":
                    held_components[role].add(canonical)
    if (
        not seen
        or not training_products
        or any(not values for values in training_components.values())
    ):
        raise CommonUgiBenchmarkError("Ugi identity reference support is empty")
    return training_products, training_components, held_components


def load_ugi_identity_references(
    assignments_path: Path,
    *,
    roles: Sequence[str],
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    """Load train and held identities, reusing only content-addressed immutable references."""

    resolved = assignments_path.resolve()
    return _load_ugi_identity_references_cached(
        resolved,
        tuple(roles),
        str(sha256_file(resolved)),
    )


def assess_common_ugi_attempts(
    attempts: Sequence[CommonUgiAttempt],
    *,
    adapter: Ugi3AssemblyAdapter,
    assignments_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply molecular validity, exact L1 replay, novelty and held-component assessment."""

    checked = validate_attempt_ledger([attempt.to_mapping() for attempt in attempts])
    training_products, training_components, held_components = load_ugi_identity_references(
        assignments_path, roles=adapter.roles
    )
    rows: list[dict[str, Any]] = []
    with rdBase.BlockLogs():
        for attempt in checked:
            molecule = (
                Chem.MolFromSmiles(attempt.product_smiles)
                if attempt.status == "generated" and attempt.product_smiles is not None
                else None
            )
            connected = molecule is not None and len(Chem.GetMolFrags(molecule)) == 1
            row = {
                **attempt.to_mapping(),
                "schema_version": ASSESSED_ATTEMPT_SCHEMA,
                "program_id": UGI_PROGRAM_ID,
                "valid": bool(connected),
                "canonical_smiles": (
                    Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
                    if connected
                    else None
                ),
            }
            rows.append(row)
    adjudicate_reaction_program_rows(
        rows,
        adapters={UGI_PROGRAM_ID: adapter},
        repeated_program_specs={},
        ugi_program_id=UGI_PROGRAM_ID,
    )
    evaluated = evaluate_reaction_program_samples(
        rows,
        training_products={UGI_PROGRAM_ID: training_products},
        training_components={UGI_PROGRAM_ID: training_components},
    )
    held_products: set[str] = set()
    held_rows = 0
    held_by_role = {role: 0 for role in adapter.roles}
    method_open_ended_products: set[str] = set()
    for row in rows:
        row["held_component_exact_l1"] = False
        row["method_visible_open_ended_exact_l1"] = False
        if row.get("exact_l1_program") is not True or int(row["exact_l1_trace_count"]) != 1:
            continue
        components = row["exact_l1_traces"][0]["components_by_role"]
        canonical_components: dict[str, str] = {}
        for role in adapter.roles:
            molecule = Chem.MolFromSmiles(str(components[role]))
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                raise CommonUgiBenchmarkError("exact-L1 trace contains an invalid component")
            canonical_components[role] = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=False
            )
        held_roles = [
            role for role in adapter.roles if canonical_components[role] in held_components[role]
        ]
        held = bool(held_roles)
        row["held_component_exact_l1"] = held
        row["held_component_roles"] = held_roles
        if held:
            held_rows += 1
            held_products.add(str(row["canonical_smiles"]))
            for role in held_roles:
                held_by_role[role] += 1
        visible = set(row["method_visible_component_ids"])
        open_ended = any(
            f"{role}:{canonical_components[role]}" not in visible for role in adapter.roles
        )
        row["method_visible_open_ended_exact_l1"] = open_ended
        if open_ended:
            method_open_ended_products.add(str(row["canonical_smiles"]))
    overall = evaluated["overall"]
    denominator = len(rows)
    totals: dict[str, int | float] = {
        field: sum(int(getattr(attempt, field)) for attempt in checked)
        for field in ("generator_calls", "reaction_calls", "route_calls", "oracle_calls")
    }
    totals["wall_seconds"] = sum(attempt.wall_seconds for attempt in checked)
    result = {
        "schema_version": "forge.common_ugi_assessment.v1",
        "method_id": checked[0].method_id,
        "seed": checked[0].seed,
        "attempts": denominator,
        "metrics": {
            **overall,
            "valid_products_per_1000_attempts": 1000.0 * int(overall["valid"]) / denominator,
            "exact_l1_products_per_1000_attempts": (
                1000.0 * int(overall["exact_l1_program"]) / denominator
            ),
            "held_component_exact_l1_products": held_rows,
            "held_component_exact_l1_products_per_1000_attempts": (
                1000.0 * held_rows / denominator
            ),
            "unique_held_component_exact_l1_products": len(held_products),
            "held_component_exact_l1_products_by_role": dict(sorted(held_by_role.items())),
            "unique_method_visible_open_ended_exact_l1_products": len(method_open_ended_products),
            "unique_method_visible_open_ended_exact_l1_products_per_1000_attempts": (
                1000.0 * len(method_open_ended_products) / denominator
            ),
        },
        "calls": totals,
        "coverage_and_precision_reported": True,
        "reductive_amination_substructure_rate_reported": False,
        "candidate_selection": False,
        "nonclaims": [
            "Exact L1 replay is transform consistency, not synthesis-success probability.",
            "A missing decomposition is an abstention, not proof that a product is unsynthesizable.",
            "Open-endedness is relative to the components visible to that method during generation.",
        ],
    }
    return rows, result


__all__ = [
    "ASSESSED_ATTEMPT_SCHEMA",
    "ATTEMPT_SCHEMA",
    "CommonUgiAttempt",
    "CommonUgiBenchmarkError",
    "assess_common_ugi_attempts",
    "load_attempt_ledger",
    "load_ugi_identity_references",
    "validate_attempt_ledger",
    "write_attempt_ledger",
]
