"""Load the exact Ugi morphology corpus without replaying Cartesian frequency.

Only one verified topology target is materialized per unique precursor
component.  Product records reference those targets as data objects, while
component keys remain outside every neural tensor.  Training weights are
raked so each precursor role has an approximately uniform component marginal.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from forge.model.conditioning.ugi import tensorize_ugi_l1_support_record
from forge.model.representation.ugi_morphology import (
    UgiComponentMorphology,
    UgiProductMorphology,
    split_ugi_support_morphology,
)
from forge.model.representation.vocabulary import load_atom_vocabulary
from forge.potency.annotations import ROLE_NAMES


class UgiMorphologyCorpusError(RuntimeError):
    """Raised when morphology supervision disagrees with the frozen split."""


@dataclass(frozen=True)
class UgiMorphologyCorpus:
    """Component-deduplicated morphology targets and strict product folds."""

    records_by_fold: Mapping[str, tuple[UgiProductMorphology, ...]]
    assignments_by_fold: Mapping[str, tuple[Mapping[str, str], ...]]
    unique_components: Mapping[tuple[str, str], UgiComponentMorphology]


@dataclass(frozen=True)
class ExpandedUgiMorphologyCorpus:
    """Compact expanded targets grouped by frozen component-family folds.

    The 1.5-million exact product rows are deliberately absent.  One exact
    semantic exemplar supplies the topology of each component, while sampling
    selects a family and then a component independently within every Ugi role.
    """

    coverage_records: tuple[UgiProductMorphology, ...]
    unique_components: Mapping[tuple[str, str], UgiComponentMorphology]
    component_metadata: Mapping[tuple[str, str], Mapping[str, str]]
    families_by_fold_role: Mapping[
        str,
        Mapping[str, Mapping[str, tuple[UgiComponentMorphology, ...]]],
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise UgiMorphologyCorpusError(f"could not read {path}: {exc}") from exc


def _select_component_exemplars(
    assignments: Sequence[Mapping[str, str]],
) -> tuple[set[str], set[tuple[str, str]]]:
    required = {(role, row[f"{role}_smiles"]) for row in assignments for role in ROLE_NAMES}
    unseen = set(required)
    selected: set[str] = set()
    for row in assignments:
        local = {(role, row[f"{role}_smiles"]) for role in ROLE_NAMES}
        if local & unseen:
            selected.add(row["product_id"])
            unseen.difference_update(local)
        if not unseen:
            break
    if unseen:
        raise UgiMorphologyCorpusError(f"could not select exemplars for {len(unseen)} components")
    return selected, required


def load_ugi_morphology_corpus(
    assignments_path: Path,
    semantic_products_path: Path,
    semantic_atoms_path: Path,
    atom_vocabulary_path: Path,
) -> UgiMorphologyCorpus:
    """Build all product triplets from one exact topology per unique component."""

    assignments = _read_csv(assignments_path)
    if not assignments or "primary_product_fold" not in assignments[0]:
        raise UgiMorphologyCorpusError("Ugi assignments lack the strict primary product fold")
    selected_ids, required_components = _select_component_exemplars(assignments)
    semantic_products = {
        row["product_id"]: row
        for row in _read_csv(semantic_products_path)
        if row["product_id"] in selected_ids
    }
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    opener = gzip.open if semantic_atoms_path.suffix == ".gz" else open
    with opener(semantic_atoms_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["product_id"] in selected_ids:
                atoms_by_product[row["product_id"]].append(row)
    if set(semantic_products) != selected_ids or set(atoms_by_product) != selected_ids:
        raise UgiMorphologyCorpusError("selected products lack exact semantic annotations")

    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    assignment_by_id = {row["product_id"]: row for row in assignments}
    unique: dict[tuple[str, str], UgiComponentMorphology] = {}
    for product_id in sorted(selected_ids):
        product = semantic_products[product_id]
        assignment = assignment_by_id[product_id]
        component_keys = {role: assignment[f"{role}_smiles"] for role in ROLE_NAMES}
        support = tensorize_ugi_l1_support_record(
            product,
            atoms_by_product[product_id],
            atom_to_index,
            preserve_aromaticity=True,
        )
        morphology = split_ugi_support_morphology(
            support,
            component_keys,
            product_id=product_id,
        )
        for component in morphology.components:
            key = (component.role, component.component_key)
            previous = unique.get(key)
            if previous is not None and previous.signature != component.signature:
                raise UgiMorphologyCorpusError(
                    f"component topology changes between exemplars: {component.role}"
                )
            unique.setdefault(key, component)
    if set(unique) != required_components:
        raise UgiMorphologyCorpusError("unique component topology census is incomplete")

    records_by_fold: defaultdict[str, list[UgiProductMorphology]] = defaultdict(list)
    assignments_by_fold: defaultdict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in assignments:
        fold = row["primary_product_fold"]
        if fold not in {"train", "calibration", "heldout"}:
            raise UgiMorphologyCorpusError(f"unsupported primary fold: {fold}")
        components = tuple(unique[(role, row[f"{role}_smiles"])] for role in ROLE_NAMES)
        records_by_fold[fold].append(
            UgiProductMorphology(
                product_id=row["product_id"],
                components=components,  # type: ignore[arg-type]
            )
        )
        assignments_by_fold[fold].append(row)
    observed_counts = {fold: len(values) for fold, values in records_by_fold.items()}
    if observed_counts != {"train": 4362, "calibration": 3350, "heldout": 4674}:
        raise UgiMorphologyCorpusError(f"strict product folds changed: {observed_counts}")
    return UgiMorphologyCorpus(
        records_by_fold={fold: tuple(values) for fold, values in records_by_fold.items()},
        assignments_by_fold={fold: tuple(values) for fold, values in assignments_by_fold.items()},
        unique_components=unique,
    )


def load_expanded_ugi_morphology_corpus(
    component_ledger_path: Path,
    semantic_products_path: Path,
    semantic_atoms_path: Path,
    atom_vocabulary_path: Path,
) -> ExpandedUgiMorphologyCorpus:
    """Load one exact morphology target per admitted expanded component."""

    ledger = _read_csv(component_ledger_path)
    required_fields = {
        "role",
        "component_smiles",
        "family_id",
        "family_fold",
        "exemplar_product_id",
    }
    if not ledger or not required_fields.issubset(ledger[0]):
        raise UgiMorphologyCorpusError("expanded component ledger lacks required fields")
    component_metadata: dict[tuple[str, str], Mapping[str, str]] = {}
    exemplar_ids: set[str] = set()
    for row in ledger:
        role = row["role"]
        fold = row["family_fold"]
        if role not in ROLE_NAMES or fold not in {"train", "calibration", "heldout"}:
            raise UgiMorphologyCorpusError("expanded component ledger has invalid role or fold")
        key = (role, row["component_smiles"])
        if key in component_metadata:
            raise UgiMorphologyCorpusError("expanded component ledger contains duplicate keys")
        if not row["family_id"]:
            raise UgiMorphologyCorpusError("expanded component lacks a frozen family")
        component_metadata[key] = row
        exemplar_ids.add(row["exemplar_product_id"])

    semantic_products = {
        row["product_id"]: row
        for row in _read_csv(semantic_products_path)
        if row["product_id"] in exemplar_ids
    }
    atoms_by_product: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    opener = gzip.open if semantic_atoms_path.suffix == ".gz" else open
    with opener(semantic_atoms_path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["product_id"] in exemplar_ids:
                atoms_by_product[row["product_id"]].append(row)
    if set(semantic_products) != exemplar_ids or set(atoms_by_product) != exemplar_ids:
        raise UgiMorphologyCorpusError("expanded exemplars lack exact semantic annotations")

    vocabulary = load_atom_vocabulary(atom_vocabulary_path)
    atom_to_index = {state: index for index, state in enumerate(vocabulary)}
    unique: dict[tuple[str, str], UgiComponentMorphology] = {}
    coverage_records: list[UgiProductMorphology] = []
    for product_id in sorted(exemplar_ids):
        product = semantic_products[product_id]
        try:
            decoded = json.loads(product["component_smiles_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise UgiMorphologyCorpusError(
                f"expanded exemplar has invalid component mapping: {product_id}"
            ) from exc
        if set(decoded) != set(ROLE_NAMES):
            raise UgiMorphologyCorpusError(
                f"expanded exemplar roles disagree with adapter: {product_id}"
            )
        component_keys = {role: str(decoded[role]) for role in ROLE_NAMES}
        support = tensorize_ugi_l1_support_record(
            product,
            atoms_by_product[product_id],
            atom_to_index,
            preserve_aromaticity=True,
        )
        morphology = split_ugi_support_morphology(
            support,
            component_keys,
            product_id=product_id,
        )
        coverage_records.append(morphology)
        for component in morphology.components:
            key = (component.role, component.component_key)
            previous = unique.get(key)
            if previous is not None and previous.signature != component.signature:
                raise UgiMorphologyCorpusError(
                    f"expanded component topology changes between exemplars: {component.role}"
                )
            unique.setdefault(key, component)
    if set(unique) != set(component_metadata):
        missing = set(component_metadata).difference(unique)
        extra = set(unique).difference(component_metadata)
        raise UgiMorphologyCorpusError(
            f"expanded topology census mismatch: missing={len(missing)}, extra={len(extra)}"
        )

    grouped: defaultdict[
        str,
        defaultdict[str, defaultdict[str, list[UgiComponentMorphology]]],
    ] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    family_folds: dict[tuple[str, str], str] = {}
    for key, component in unique.items():
        row = component_metadata[key]
        family_key = (component.role, row["family_id"])
        previous_fold = family_folds.setdefault(family_key, row["family_fold"])
        if previous_fold != row["family_fold"]:
            raise UgiMorphologyCorpusError("one component family crosses frozen folds")
        grouped[row["family_fold"]][component.role][row["family_id"]].append(component)

    frozen_groups: dict[
        str,
        dict[str, dict[str, tuple[UgiComponentMorphology, ...]]],
    ] = {}
    for fold in ("train", "calibration", "heldout"):
        frozen_groups[fold] = {}
        for role in ROLE_NAMES:
            families = grouped[fold][role]
            if not families:
                raise UgiMorphologyCorpusError(f"expanded fold lacks {fold}/{role}")
            frozen_groups[fold][role] = {
                family_id: tuple(sorted(values, key=lambda value: value.component_key))
                for family_id, values in sorted(families.items())
            }
    return ExpandedUgiMorphologyCorpus(
        coverage_records=tuple(coverage_records),
        unique_components=unique,
        component_metadata=component_metadata,
        families_by_fold_role=frozen_groups,
    )


def sample_family_balanced_records(
    corpus: ExpandedUgiMorphologyCorpus,
    *,
    fold: str,
    count: int,
    rng: np.random.Generator,
    id_prefix: str = "family-balanced",
) -> tuple[UgiProductMorphology, ...]:
    """Sample one family then one component per role with exact fold isolation."""

    if fold not in corpus.families_by_fold_role or count < 1:
        raise UgiMorphologyCorpusError("invalid family-balanced sampling request")
    records = []
    for index in range(count):
        components = []
        for role in ROLE_NAMES:
            families = corpus.families_by_fold_role[fold][role]
            family_ids = tuple(families)
            family_id = family_ids[int(rng.integers(len(family_ids)))]
            family = families[family_id]
            components.append(family[int(rng.integers(len(family)))])
        records.append(
            UgiProductMorphology(
                product_id=f"{id_prefix}-{index:08d}",
                components=tuple(components),  # type: ignore[arg-type]
            )
        )
    return tuple(records)


def expanded_component_census(corpus: ExpandedUgiMorphologyCorpus) -> dict[str, object]:
    """Return frozen component and family counts without product-row statistics."""

    component_counts = Counter(role for role, _ in corpus.unique_components)
    fold_component_counts: dict[str, dict[str, int]] = {}
    fold_family_counts: dict[str, dict[str, int]] = {}
    for fold, by_role in corpus.families_by_fold_role.items():
        fold_family_counts[fold] = {role: len(by_role[role]) for role in ROLE_NAMES}
        fold_component_counts[fold] = {
            role: sum(len(values) for values in by_role[role].values()) for role in ROLE_NAMES
        }
    return {
        "unique_components": len(corpus.unique_components),
        "component_counts": dict(component_counts),
        "components_by_fold_role": fold_component_counts,
        "families_by_fold_role": fold_family_counts,
        "semantic_exemplar_products": len(corpus.coverage_records),
    }


def family_balanced_component_weights(
    corpus: ExpandedUgiMorphologyCorpus,
    *,
    fold: str,
) -> dict[tuple[str, str], float]:
    """Return P(family)P(component|family) weights, normalized per role."""

    if fold not in corpus.families_by_fold_role:
        raise UgiMorphologyCorpusError("unknown expanded component fold")
    output: dict[tuple[str, str], float] = {}
    for role in ROLE_NAMES:
        families = corpus.families_by_fold_role[fold][role]
        family_mass = 1.0 / len(families)
        for values in families.values():
            component_mass = family_mass / len(values)
            for component in values:
                output[(role, component.component_key)] = component_mass
    return output


def balanced_product_weights(
    assignments: Sequence[Mapping[str, str]],
    *,
    iterations: int = 300,
    uniform_row_mixture: float = 0.5,
) -> np.ndarray:
    """Regularize product-row mass while reducing component-frequency bias.

    Exact simultaneous uniform marginals need not be feasible when the observed
    component cross-product is incomplete.  Iterative raking finds the closest
    supported solution, then a frozen uniform-row mixture prevents vanishing
    training mass and pathological rare-component upweighting.
    """

    if not assignments or iterations < 1 or not 0 < uniform_row_mixture < 1:
        raise UgiMorphologyCorpusError("balanced product weighting requires assignments")
    weights = np.ones(len(assignments), dtype=np.float64)
    indices_by_role: dict[str, dict[str, np.ndarray]] = {}
    for role in ROLE_NAMES:
        grouped: defaultdict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(assignments):
            grouped[row[f"{role}_smiles"]].append(index)
        indices_by_role[role] = {
            component: np.asarray(indices, dtype=np.int64) for component, indices in grouped.items()
        }
    for _ in range(iterations):
        for role in ROLE_NAMES:
            groups = indices_by_role[role]
            target = 1.0 / len(groups)
            total = weights.sum()
            for indices in groups.values():
                current = weights[indices].sum() / total
                if current <= 0:
                    raise UgiMorphologyCorpusError("product raking encountered zero component mass")
                weights[indices] *= target / current
            weights /= weights.sum()
    raked = weights / weights.sum()
    uniform = np.full(len(assignments), 1.0 / len(assignments), dtype=np.float64)
    regularized = (1.0 - uniform_row_mixture) * raked + uniform_row_mixture * uniform
    return regularized / regularized.sum()


def source_stratified_family_weights(
    assignments: Sequence[Mapping[str, str]],
    *,
    source_mass: Mapping[str, float] | None = None,
    iterations: int = 300,
    uniform_row_mixture: float = 0.5,
) -> np.ndarray:
    """Balance precursor families within provenance strata.

    The production chemistry corpus deliberately mixes a realism anchor with
    a family-weighted expanded enumeration.  Raking all rows together would
    let the larger stratum determine the training distribution, while applying
    the selection weights again would square the expansion correction.  This
    helper instead assigns frozen mass to each source stratum and rakes the
    three role-family marginals *within* every stratum.  Family identifiers are
    sampler metadata only; the returned weights never enter model tensors.
    """

    if not assignments or iterations < 1 or not 0 < uniform_row_mixture < 1:
        raise UgiMorphologyCorpusError("source-stratified weighting requires assignments")
    required = {
        "source_stratum",
        *(f"{role}_family_id" for role in ROLE_NAMES),
    }
    if any(not required.issubset(row) for row in assignments):
        raise UgiMorphologyCorpusError(
            "source-stratified weighting requires source and role-family fields"
        )
    indices_by_source: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(assignments):
        source = str(row["source_stratum"])
        if not source:
            raise UgiMorphologyCorpusError("empty source stratum")
        indices_by_source[source].append(index)
    sources = tuple(sorted(indices_by_source))
    if source_mass is None:
        normalized_source_mass = {source: 1.0 / len(sources) for source in sources}
    else:
        if set(source_mass) != set(sources):
            raise UgiMorphologyCorpusError("source-mass keys disagree with observed strata")
        total = float(sum(source_mass.values()))
        if total <= 0 or any(
            not np.isfinite(value) or value <= 0 for value in source_mass.values()
        ):
            raise UgiMorphologyCorpusError("source masses must be finite and positive")
        normalized_source_mass = {source: float(source_mass[source]) / total for source in sources}

    output = np.zeros(len(assignments), dtype=np.float64)
    for source in sources:
        indices = np.asarray(indices_by_source[source], dtype=np.int64)
        local = [assignments[int(index)] for index in indices]
        weights = np.ones(len(local), dtype=np.float64)
        grouped_by_role: dict[str, dict[str, np.ndarray]] = {}
        for role in ROLE_NAMES:
            grouped: defaultdict[str, list[int]] = defaultdict(list)
            for local_index, row in enumerate(local):
                family = str(row[f"{role}_family_id"])
                if not family:
                    raise UgiMorphologyCorpusError("empty precursor family identifier")
                grouped[family].append(local_index)
            grouped_by_role[role] = {
                family: np.asarray(group_indices, dtype=np.int64)
                for family, group_indices in grouped.items()
            }
        for _ in range(iterations):
            for role in ROLE_NAMES:
                groups = grouped_by_role[role]
                target = 1.0 / len(groups)
                total = weights.sum()
                for group_indices in groups.values():
                    current = weights[group_indices].sum() / total
                    if current <= 0:
                        raise UgiMorphologyCorpusError("source-family raking encountered zero mass")
                    weights[group_indices] *= target / current
                weights /= weights.sum()
        raked = weights / weights.sum()
        uniform = np.full(len(local), 1.0 / len(local), dtype=np.float64)
        local_weights = (1.0 - uniform_row_mixture) * raked + uniform_row_mixture * uniform
        output[indices] = normalized_source_mass[source] * local_weights
    return output / output.sum()


def component_marginal_errors(
    assignments: Sequence[Mapping[str, str]],
    weights: np.ndarray,
) -> dict[str, float]:
    """Report the largest deviation from a uniform role-component marginal."""

    if weights.shape != (len(assignments),) or np.any(weights < 0):
        raise UgiMorphologyCorpusError("invalid product-weight vector")
    normalized = weights / weights.sum()
    errors = {}
    for role in ROLE_NAMES:
        mass: Counter[str] = Counter()
        for row, weight in zip(assignments, normalized, strict=True):
            mass[row[f"{role}_smiles"]] += float(weight)
        target = 1.0 / len(mass)
        errors[role] = max(abs(value - target) for value in mass.values())
    return errors


def component_program_ledger(corpus: UgiMorphologyCorpus) -> list[dict[str, object]]:
    """Materialize the hashable unique-component program PMF ledger."""

    counts = Counter(role for role, _ in corpus.unique_components)
    rows = []
    for (role, key), component in sorted(corpus.unique_components.items()):
        rows.append(
            {
                "role": role,
                "component_smiles": key,
                "component_weight": 1.0 / counts[role],
                "exterior_node_count": component.node_count,
                "junction_budget": component.junction_budget,
                "cycle_rank": component.cycle_rank,
                "attachment_count": component.attachment_count,
                "offspring_json": json.dumps(component.offspring.tolist(), separators=(",", ":")),
                "closure_pairs_json": json.dumps(
                    list(
                        zip(
                            component.closure_left.tolist(),
                            component.closure_right.tolist(),
                            strict=True,
                        )
                    ),
                    separators=(",", ":"),
                ),
            }
        )
    return rows
