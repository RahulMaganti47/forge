"""Strong finite-component baseline for reaction-program generation.

The baseline receives the exact train-fold component identities and the exact qualified forward
assembler. It samples source-weighted role marginals independently, then emits at most one
deterministically selected product per component-tuple attempt. It therefore represents the best
case for catalogue assembly without giving it calibration or heldout components.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rdkit import Chem

from forge.assembly import ReactionProgramSpec
from forge.core.io import iter_csv

AssemblyKind = Literal["fixed_arity", "repeated"]


class FiniteComponentCatalogueError(ValueError):
    """The train-only component catalogue or a forward assembly attempt is invalid."""


@dataclass(frozen=True)
class WeightedSupport:
    """One deterministically ordered, normalized finite categorical support."""

    values: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.values or len(self.values) != len(self.probabilities):
            raise FiniteComponentCatalogueError("weighted support is empty or misaligned")
        if tuple(sorted(set(self.values))) != self.values:
            raise FiniteComponentCatalogueError("weighted support values are not unique and sorted")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.probabilities):
            raise FiniteComponentCatalogueError("weighted support has a non-positive probability")
        if not math.isclose(sum(self.probabilities), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise FiniteComponentCatalogueError("weighted support probabilities do not sum to one")

    def sample(self, rng: np.random.Generator) -> str:
        index = int(rng.choice(len(self.values), p=np.asarray(self.probabilities)))
        return self.values[index]


@dataclass(frozen=True)
class FiniteComponentProgram:
    """Train-fold component support and depth prior for one exact assembly program."""

    program_id: str
    assembly_kind: AssemblyKind
    role_supports: tuple[tuple[str, WeightedSupport], ...]
    depth_support: WeightedSupport

    def __post_init__(self) -> None:
        roles = tuple(role for role, _ in self.role_supports)
        if not self.program_id or not roles or len(set(roles)) != len(roles):
            raise FiniteComponentCatalogueError("finite component program has invalid roles")
        if self.assembly_kind not in {"fixed_arity", "repeated"}:
            raise FiniteComponentCatalogueError(
                "finite component program has invalid assembly kind"
            )
        if any(not value.isdigit() or int(value) < 1 for value in self.depth_support.values):
            raise FiniteComponentCatalogueError("program depth support is invalid")
        if self.assembly_kind == "fixed_arity" and self.depth_support.values != ("1",):
            raise FiniteComponentCatalogueError("fixed-arity assembly must have depth one")

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role for role, _ in self.role_supports)

    @property
    def tuple_space_upper_bound(self) -> int:
        result = len(self.depth_support.values)
        for _, support in self.role_supports:
            result *= len(support.values)
        return result

    def sample_components(self, rng: np.random.Generator) -> tuple[dict[str, str], int]:
        components = {role: support.sample(rng) for role, support in self.role_supports}
        depth = int(self.depth_support.sample(rng))
        return components, depth


def _canonical(smiles: str, *, label: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise FiniteComponentCatalogueError(f"{label} is not a connected molecular graph")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _support(weights: Mapping[str, float], *, label: str) -> WeightedSupport:
    values = tuple(sorted(weights))
    if not values:
        raise FiniteComponentCatalogueError(f"{label} has no train-fold support")
    total = sum(float(weights[value]) for value in values)
    if not math.isfinite(total) or total <= 0.0:
        raise FiniteComponentCatalogueError(f"{label} has invalid total weight")
    return WeightedSupport(
        values=values,
        probabilities=tuple(float(weights[value]) / total for value in values),
    )


def build_finite_component_catalogues(
    *,
    ugi_assignments: Path,
    multireaction_atlas: Path,
    multireaction_splits: Path,
    repeated_program_specs: Mapping[str, ReactionProgramSpec],
    ugi_program_id: str,
    ugi_roles: Sequence[str],
) -> dict[str, FiniteComponentProgram]:
    """Build source-weighted role marginals from train-fold rows only."""

    role_weights: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(float))
    )
    depth_weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in iter_csv(ugi_assignments):
        if row.get("primary_product_fold") != "train":
            continue
        weight = float(row["family_balance_weight_raw"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise FiniteComponentCatalogueError("Ugi train row has invalid realism weight")
        for role in ugi_roles:
            smiles = _canonical(row[f"{role}_smiles"], label=f"Ugi {role}")
            role_weights[ugi_program_id][role][smiles] += weight
        depth_weights[ugi_program_id]["1"] += weight

    train_splits: dict[str, tuple[str, float]] = {}
    for row in iter_csv(multireaction_splits):
        record_id = row["record_id"]
        if record_id in train_splits:
            raise FiniteComponentCatalogueError(f"duplicate split record: {record_id}")
        if row.get("product_fold") != "train":
            continue
        weight = float(row["source_balanced_weight"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise FiniteComponentCatalogueError(
                f"multi-reaction train row has invalid source weight: {record_id}"
            )
        train_splits[record_id] = (row["program_id"], weight)

    observed: set[str] = set()
    for row in iter_csv(multireaction_atlas):
        record_id = row["record_id"]
        if record_id not in train_splits:
            continue
        program_id, weight = train_splits[record_id]
        if row["program_id"] != program_id or program_id not in repeated_program_specs:
            raise FiniteComponentCatalogueError(
                f"multi-reaction training identity changed for {record_id}"
            )
        if row.get("disposition") != "admit_exact" or row.get("exact_forward_roundtrip") != "true":
            raise FiniteComponentCatalogueError(
                f"multi-reaction catalogue includes nonexact record {record_id}"
            )
        spec = repeated_program_specs[program_id]
        head = _canonical(row["terminal_head_smiles"], label=f"{program_id} head")
        repeat = _canonical(row["repeat_component_smiles"], label=f"{program_id} repeat")
        depth = int(row["step_count"])
        if depth < spec.minimum_steps or depth > spec.maximum_steps:
            raise FiniteComponentCatalogueError(f"{record_id} has unsupported program depth")
        role_weights[program_id][spec.accumulator_role][head] += weight
        role_weights[program_id][spec.repeat_role][repeat] += weight
        depth_weights[program_id][str(depth)] += weight
        observed.add(record_id)
    if observed != set(train_splits):
        raise FiniteComponentCatalogueError(
            "multi-reaction atlas omits a train-fold catalogue record"
        )

    programs: dict[str, FiniteComponentProgram] = {}
    programs[ugi_program_id] = FiniteComponentProgram(
        program_id=ugi_program_id,
        assembly_kind="fixed_arity",
        role_supports=tuple(
            (role, _support(role_weights[ugi_program_id][role], label=f"{ugi_program_id}/{role}"))
            for role in ugi_roles
        ),
        depth_support=_support(depth_weights[ugi_program_id], label=f"{ugi_program_id}/depth"),
    )
    for program_id, spec in sorted(repeated_program_specs.items()):
        programs[program_id] = FiniteComponentProgram(
            program_id=program_id,
            assembly_kind="repeated",
            role_supports=tuple(
                (
                    role,
                    _support(role_weights[program_id][role], label=f"{program_id}/{role}"),
                )
                for role in (spec.accumulator_role, spec.repeat_role)
            ),
            depth_support=_support(depth_weights[program_id], label=f"{program_id}/depth"),
        )
    return programs


def finite_component_catalogue_coverage(
    catalogues: Mapping[str, FiniteComponentProgram],
    *,
    ugi_assignments: Path,
    multireaction_atlas: Path,
    multireaction_splits: Path,
    repeated_program_specs: Mapping[str, ReactionProgramSpec],
    ugi_program_id: str,
    ugi_roles: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Measure which source tuples are reachable without leaving each train-only catalogue."""

    supports = {
        program_id: {role: set(support.values) for role, support in catalogue.role_supports}
        for program_id, catalogue in catalogues.items()
    }
    depths = {
        program_id: {int(value) for value in catalogue.depth_support.values}
        for program_id, catalogue in catalogues.items()
    }
    fold_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"source_products": 0, "reachable_component_tuples": 0})
    )
    all_components: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    def observe(
        program_id: str,
        fold: str,
        components: Mapping[str, str],
        depth: int,
    ) -> None:
        if program_id not in catalogues or not fold:
            raise FiniteComponentCatalogueError("catalogue coverage row has unknown identity")
        counts = fold_counts[program_id][fold]
        counts["source_products"] += 1
        for role, value in components.items():
            all_components[program_id][role].add(value)
        if depth in depths[program_id] and all(
            value in supports[program_id].get(role, set()) for role, value in components.items()
        ):
            counts["reachable_component_tuples"] += 1

    for row in iter_csv(ugi_assignments):
        observe(
            ugi_program_id,
            row["primary_product_fold"],
            {role: _canonical(row[f"{role}_smiles"], label=f"Ugi {role}") for role in ugi_roles},
            1,
        )

    split_rows = {row["record_id"]: row for row in iter_csv(multireaction_splits)}
    observed_records: set[str] = set()
    for row in iter_csv(multireaction_atlas):
        record_id = row["record_id"]
        if row.get("disposition") != "admit_exact" or row.get("exact_forward_roundtrip") != "true":
            continue
        split = split_rows.get(record_id)
        if split is None:
            raise FiniteComponentCatalogueError(
                f"multi-reaction catalogue coverage omits split record {record_id}"
            )
        program_id = row["program_id"]
        if program_id not in repeated_program_specs or split["program_id"] != program_id:
            raise FiniteComponentCatalogueError(
                f"multi-reaction catalogue coverage changed identity for {record_id}"
            )
        spec = repeated_program_specs[program_id]
        observe(
            program_id,
            split["product_fold"],
            {
                spec.accumulator_role: _canonical(
                    row["terminal_head_smiles"], label=f"{program_id} head"
                ),
                spec.repeat_role: _canonical(
                    row["repeat_component_smiles"], label=f"{program_id} repeat"
                ),
            },
            int(row["step_count"]),
        )
        observed_records.add(record_id)
    if observed_records != set(split_rows):
        raise FiniteComponentCatalogueError(
            "multi-reaction catalogue coverage atlas/split records differ"
        )

    result: dict[str, dict[str, Any]] = {}
    for program_id, catalogue in sorted(catalogues.items()):
        by_fold = {
            fold: {
                **counts,
                "coverage_fraction": (
                    counts["reachable_component_tuples"] / counts["source_products"]
                    if counts["source_products"]
                    else None
                ),
            }
            for fold, counts in sorted(fold_counts[program_id].items())
        }
        result[program_id] = {
            "source_product_coverage_by_fold": by_fold,
            "unique_source_component_coverage_by_role": {
                role: {
                    "catalogue_components": len(supports[program_id][role]),
                    "all_source_components": len(all_components[program_id][role]),
                    "coverage_fraction": (
                        len(supports[program_id][role]) / len(all_components[program_id][role])
                        if all_components[program_id][role]
                        else None
                    ),
                }
                for role, _ in catalogue.role_supports
            },
            "maximum_attainable_component_novelty_fraction": 0.0,
            "whole_product_novelty_ceiling": (
                "not_inferred_from_cartesian_tuple_count; measured empirically per attempt"
            ),
        }
    return result


def sample_finite_component_program(
    catalogue: FiniteComponentProgram,
    adapter: Any,
    *,
    attempts: int,
    seed: int,
    maximum_outcomes: int,
) -> list[dict[str, Any]]:
    """Sample train-only component tuples and execute the exact forward assembly."""

    if isinstance(attempts, bool) or attempts < 1:
        raise FiniteComponentCatalogueError("catalogue baseline attempts must be positive")
    if isinstance(maximum_outcomes, bool) or maximum_outcomes < 2:
        raise FiniteComponentCatalogueError("maximum_outcomes must be at least two")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for attempt_index in range(attempts):
        components, depth = catalogue.sample_components(rng)
        if catalogue.assembly_kind == "fixed_arity":
            assembled = adapter.forward_products(components, maximum_outcomes=maximum_outcomes)
            sampled_components: dict[str, object] = dict(components)
        else:
            head_role, repeat_role = catalogue.roles
            repeated = [components[repeat_role]] * depth
            assembled = adapter.forward_products(
                components[head_role],
                repeated,
                maximum_outcomes=maximum_outcomes,
            )
            sampled_components = {
                head_role: components[head_role],
                repeat_role: repeated,
            }
        if assembled.saturated:
            raise FiniteComponentCatalogueError(
                f"{catalogue.program_id} attempt {attempt_index} saturated forward enumeration"
            )
        product_index = (
            int(rng.integers(0, len(assembled.products))) if assembled.products else None
        )
        canonical_smiles = assembled.products[product_index] if product_index is not None else None
        rows.append(
            {
                "attempt_id": f"{catalogue.program_id}:{attempt_index:08d}",
                "attempt_index": attempt_index,
                "program_id": catalogue.program_id,
                "program_depth": depth,
                "sampled_components_by_role": sampled_components,
                "forward_product_count": len(assembled.products),
                "forward_outcomes_by_step": list(assembled.enumerated_outcomes_by_step),
                "selected_forward_product_index": product_index,
                "valid": canonical_smiles is not None,
                "canonical_smiles": canonical_smiles,
                "candidate_selection": False,
            }
        )
    return rows


__all__ = [
    "FiniteComponentCatalogueError",
    "FiniteComponentProgram",
    "WeightedSupport",
    "build_finite_component_catalogues",
    "finite_component_catalogue_coverage",
    "sample_finite_component_program",
]
