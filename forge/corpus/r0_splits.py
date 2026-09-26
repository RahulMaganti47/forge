"""Create and load the frozen, leakage-safe R0 splits for M0-03.

Split construction is an M0 curation operation. Downstream tasks must call
``load_frozen_r0_splits`` and must not reconstruct groups from the source corpus.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import shutil
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

CONFIG_SCHEMA_VERSION = "m0_03_r0_split_config.v2"
MANIFEST_SCHEMA_VERSION = "m0_03_r0_splits.v2"
RESULT_SCHEMA_VERSION = "m0_03_result.v2"
ASSIGNMENT_FILENAME = "r0_fold_assignments.csv"
MANIFEST_FILENAME = "manifest.json"
SCHEMES = ("source_study", "headgroup", "linker_scaffold", "component_family")
FOLDS = ("R0_train", "R0_cal", "R0_heldout")
ASSIGNMENT_FIELDS = (
    "r0_structure_id",
    "leakage_group_id",
    *(field for scheme in SCHEMES for field in (f"{scheme}_group_id", f"{scheme}_fold")),
)


class SplitError(ValueError):
    """Raised when source data or a frozen split violates the M0-03 contract."""


@dataclass(frozen=True)
class FrozenR0Splits:
    """Validated, load-only view of the M0-03 split bundle."""

    manifest: dict[str, Any]
    assignments: tuple[dict[str, str], ...]

    def ids(self, scheme: str, fold: str) -> tuple[str, ...]:
        """Return R0 structure IDs assigned to one scheme and fold."""

        _validate_scheme_and_fold(scheme, fold)
        fold_field = f"{scheme}_fold"
        return tuple(row["r0_structure_id"] for row in self.assignments if row[fold_field] == fold)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        parent = self.parent[index]
        while parent != index:
            self.parent[index] = self.parent[parent]
            index = self.parent[index]
            parent = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def sha256_bytes(payload: bytes) -> str:
    """Return the SHA-256 digest of bytes."""

    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 digest of a file."""

    digest = hashlib.sha256()
    try:
        handle = path.open("rb")
    except FileNotFoundError as exc:
        raise SplitError(f"required input not found: {path}") from exc
    with handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(namespace: str, values: Iterable[str]) -> str:
    payload = "\0".join(sorted(set(values))).encode()
    return f"{namespace}-{hashlib.sha256(payload).hexdigest()[:20]}"


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise SplitError(f"{description} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SplitError(f"{description} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitError(f"{description} must contain a JSON object: {path}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the frozen M0-03 split configuration."""

    config = _load_json(path, "M0-03 split config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise SplitError(
            f"unsupported config schema {config.get('schema_version')!r}; "
            f"expected {CONFIG_SCHEMA_VERSION!r}"
        )
    if config.get("expected_r0_rows") != 15_229:
        raise SplitError("expected_r0_rows must remain fixed at 15229 for reconciled M0-03")
    if config.get("expected_original_r0_rows") != 15_433:
        raise SplitError("expected_original_r0_rows must remain fixed at 15433")
    if config.get("randomness_used") is not False:
        raise SplitError("M0-03 uses deterministic grouping; randomness_used must be false")
    fractions = config.get("target_fractions")
    if not isinstance(fractions, dict) or tuple(fractions) != FOLDS:
        raise SplitError(f"target_fractions must define folds in order {FOLDS}")
    if any(not isinstance(fractions[fold], (int, float)) for fold in FOLDS):
        raise SplitError("target_fractions must be numeric")
    if abs(sum(float(fractions[fold]) for fold in FOLDS) - 1.0) > 1e-12:
        raise SplitError("target_fractions must sum to one")
    if any(float(fractions[fold]) <= 0 for fold in FOLDS):
        raise SplitError("every target fraction must be positive")
    if set(config.get("upstream_split_columns", {})) != set(SCHEMES):
        raise SplitError(f"upstream_split_columns must define {SCHEMES}")
    artifact_paths = config.get("frozen_artifact_paths")
    if not isinstance(artifact_paths, dict) or set(artifact_paths) != {
        "assignments",
        "manifest",
        "result",
    }:
        raise SplitError("frozen_artifact_paths must define assignments, manifest, and result")
    return config


def verify_source_inputs(
    config_path: Path, source_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify hash-pinned split inputs and their cross-file claims."""

    config = load_config(config_path)
    records = [
        {
            "asset": config_path.name,
            "role": "split_config",
            "bytes": config_path.stat().st_size,
            "sha256": sha256_file(config_path),
        }
    ]
    expected_inputs = config.get("expected_inputs", {})
    required = {
        "r0_constitutional",
        "r0_reconciliation",
        "upstream_assignments",
        "upstream_manifest",
    }
    if set(expected_inputs) != required:
        raise SplitError(f"expected_inputs must contain exactly {sorted(required)}")
    for asset in sorted(required):
        specification = expected_inputs[asset]
        if not isinstance(specification, Mapping):
            raise SplitError(f"{asset} input specification must be an object")
        path = source_root / str(specification.get("path", ""))
        actual = sha256_file(path)
        expected = specification.get("sha256")
        if actual != expected:
            raise SplitError(f"hash mismatch for {asset}: expected {expected}, observed {actual}")
        records.append(
            {
                "asset": asset,
                "path": str(path.resolve().relative_to(source_root.resolve())),
                "bytes": path.stat().st_size,
                "sha256": actual,
            }
        )

    upstream_manifest = _load_json(
        source_root / expected_inputs["upstream_manifest"]["path"],
        "upstream split manifest",
    )
    assignment_claim = upstream_manifest.get("fold_assignments_csv", {})
    if assignment_claim.get("sha256") != expected_inputs["upstream_assignments"]["sha256"]:
        raise SplitError("upstream split manifest does not pin the vendored assignment CSV")
    if (
        upstream_manifest.get("layers", {}).get("R0_observed")
        != config["expected_original_r0_rows"]
    ):
        raise SplitError("upstream split manifest has an unexpected original R0 row count")
    if upstream_manifest.get("all_splits_leak_free") is not True:
        raise SplitError("upstream split manifest does not declare its source groups leak-free")
    reconciliation = _load_json(
        source_root / expected_inputs["r0_reconciliation"]["path"],
        "R0 reconciliation",
    )
    if (
        reconciliation.get("summary", {}).get("input_rows") != config["expected_original_r0_rows"]
        or reconciliation.get("summary", {}).get("output_constitutions")
        != config["expected_r0_rows"]
        or reconciliation.get("decision", {}).get("r0_ready_for_split_regeneration") is not True
    ):
        raise SplitError("R0 reconciliation does not authorize split regeneration")
    corpus_artifact = reconciliation.get("artifacts", {}).get(
        "r0_constitutional.csv.gz",
        {},
    )
    if corpus_artifact.get("sha256") != expected_inputs["r0_constitutional"]["sha256"]:
        raise SplitError("R0 reconciliation does not authenticate the constitutional corpus")
    return config, records


def _read_csv(path: Path, required_fields: set[str], description: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        handle = opener(path, "rt", newline="")
    except FileNotFoundError as exc:
        raise SplitError(f"{description} not found: {path}") from exc
    with handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required_fields.issubset(reader.fieldnames):
            missing = sorted(required_fields.difference(reader.fieldnames or ()))
            raise SplitError(f"{description} is missing columns: {missing}")
        return list(reader)


def _load_r0_rows(path: Path, expected_rows: int) -> list[dict[str, str]]:
    required = {
        "r0_structure_id",
        "canonical_isomeric_smiles",
        "leakage_group_id",
        "observed_source_ids",
        "study_split_groups_json",
        "component_holdout_groups_json",
        "reaction_family_holdout_groups",
        "original_isomeric_smiles_json",
        "source_r0_structure_ids_json",
    }
    rows = _read_csv(path, required, "R0 corpus")
    if len(rows) != expected_rows:
        raise SplitError(f"R0 corpus has {len(rows)} rows; expected {expected_rows}")
    for field in ("r0_structure_id", "canonical_isomeric_smiles", "leakage_group_id"):
        values = [row[field] for row in rows]
        if any(not value for value in values):
            raise SplitError(f"R0 corpus contains an empty {field}")
        if len(values) != len(set(values)):
            raise SplitError(f"R0 corpus contains duplicate {field} values")
    return rows


def _load_upstream_assignments(path: Path, expected_rows: int) -> dict[str, dict[str, str]]:
    required = {
        "canonical_smiles",
        "layer",
        "held_reaction_family",
        "held_scaffold",
        "held_head",
        "held_study",
    }
    all_rows = _read_csv(path, required, "upstream corpus fold assignments")
    rows = [row for row in all_rows if row["layer"] == "R0_observed"]
    if len(rows) != expected_rows:
        raise SplitError(f"upstream assignments have {len(rows)} R0 rows; expected {expected_rows}")
    by_smiles = {row["canonical_smiles"]: row for row in rows}
    if len(by_smiles) != len(rows):
        raise SplitError("upstream assignments contain duplicate R0 canonical_smiles")
    return by_smiles


def _parse_json_object(text: str, field: str, structure_id: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SplitError(f"{structure_id} has invalid {field}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitError(f"{structure_id} {field} must contain a JSON object")
    return value


def _values(value: Any, field: str, structure_id: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise SplitError(f"{structure_id} component field {field} must contain a list of strings")
    return [item.strip() for item in value if item.strip()]


def _canonical_annotation_token(value: str, namespace: str, parse_failures: Counter[str]) -> str:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None:
        parse_failures[namespace] += 1
        return f"{namespace}:unparsed:{value}"
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return f"{namespace}:smiles:{canonical}"


def _component_tokens(
    row: Mapping[str, str],
    selectors: Mapping[str, Mapping[str, Sequence[str]]],
    namespace: str,
    parse_failures: Counter[str],
) -> set[str]:
    structure_id = row["r0_structure_id"]
    components = _parse_json_object(
        row["component_holdout_groups_json"],
        "component_holdout_groups_json",
        structure_id,
    )
    tokens: set[str] = set()
    for source, field_kinds in selectors.items():
        source_components = components.get(source, {})
        if not isinstance(source_components, dict):
            raise SplitError(f"{structure_id} component source {source} must be a JSON object")
        for field in field_kinds.get("smiles", ()):
            for value in _values(source_components.get(field, []), field, structure_id):
                tokens.add(_canonical_annotation_token(value, namespace, parse_failures))
        for field in field_kinds.get("identifiers", ()):
            for value in _values(source_components.get(field, []), field, structure_id):
                tokens.add(f"{namespace}:identifier:{source}:{field}:{value}")
    return tokens


def _source_study_tokens(row: Mapping[str, str]) -> set[str]:
    structure_id = row["r0_structure_id"]
    studies = _parse_json_object(
        row["study_split_groups_json"], "study_split_groups_json", structure_id
    )
    tokens: set[str] = set()
    sources_with_studies: set[str] = set()
    for source, values in studies.items():
        source_values = _values(values, f"study:{source}", structure_id)
        for value in source_values:
            tokens.add(f"study:{source}:{value}")
        if source_values:
            sources_with_studies.add(source)
    observed_sources = {source for source in row["observed_source_ids"].split("|") if source}
    if not observed_sources:
        raise SplitError(f"{structure_id} has no observed sources")
    tokens.update(
        f"study:fallback-source:{source}"
        for source in observed_sources.difference(sources_with_studies)
    )
    return tokens


def heteroatom_connector_core(molecule: Chem.Mol) -> str:
    """Return the tail-stripped connector used by the upstream scaffold split.

    This compatibility implementation keeps every heteroatom and all shortest-path
    atoms connecting them. It is a grouping key, not a synthetic-block claim.
    """

    heteroatoms = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() != 6]
    if not heteroatoms:
        return "<NO_HETEROATOM>"
    keep = set(heteroatoms)
    for left_index, left in enumerate(heteroatoms):
        for right in heteroatoms[left_index + 1 :]:
            keep.update(Chem.GetShortestPath(molecule, left, right))

    editable = Chem.RWMol()
    old_to_new: dict[int, int] = {}
    for old_index in sorted(keep):
        old_to_new[old_index] = editable.AddAtom(Chem.Atom(molecule.GetAtomWithIdx(old_index)))
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if begin in keep and end in keep:
            editable.AddBond(old_to_new[begin], old_to_new[end], bond.GetBondType())
            new_bond = editable.GetBondBetweenAtoms(old_to_new[begin], old_to_new[end])
            if new_bond is not None:
                new_bond.SetIsAromatic(bond.GetIsAromatic())
    core = editable.GetMol()
    try:
        with rdBase.BlockLogs():
            Chem.SanitizeMol(core)
            return Chem.MolToSmiles(core, canonical=True, isomericSmiles=True)
    except Exception:
        atoms = ".".join(
            f"{molecule.GetAtomWithIdx(index).GetSymbol()}:{index}" for index in sorted(keep)
        )
        bonds = ".".join(
            sorted(
                f"{min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())}-"
                f"{max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())}:{bond.GetBondType()}"
                for bond in molecule.GetBonds()
                if bond.GetBeginAtomIdx() in keep and bond.GetEndAtomIdx() in keep
            )
        )
        return f"<UNSANITIZED_CONNECTOR>|{atoms}|{bonds}"


def _linker_scaffold_tokens(
    row: Mapping[str, str],
    selectors: Mapping[str, Mapping[str, Sequence[str]]],
    parse_failures: Counter[str],
) -> set[str]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(row["canonical_isomeric_smiles"])
    if molecule is None:
        raise SplitError(f"{row['r0_structure_id']} has an RDKit-invalid canonical structure")
    tokens = {f"linker-scaffold:heteroatom-connector:{heteroatom_connector_core(molecule)}"}
    tokens.update(_component_tokens(row, selectors, "linker", parse_failures))
    return tokens


def _component_family_tokens(row: Mapping[str, str]) -> set[str]:
    signature = row["reaction_family_holdout_groups"].strip()
    if not signature:
        raise SplitError(
            f"{row['r0_structure_id']} has an empty reaction_family_holdout_groups value"
        )
    return {f"component-family:signature:{signature}"}


def _connected_group_ids(
    token_sets: Sequence[set[str]], namespace: str
) -> tuple[list[str], dict[str, int]]:
    """Collapse overlapping annotation tokens into leak-safe connected groups."""

    union_find = _UnionFind(len(token_sets))
    token_owner: dict[str, int] = {}
    for row_index, tokens in enumerate(token_sets):
        if not tokens:
            raise SplitError(f"{namespace} row {row_index} has no grouping token")
        for token in sorted(tokens):
            owner = token_owner.setdefault(token, row_index)
            union_find.union(row_index, owner)

    component_tokens: dict[int, set[str]] = defaultdict(set)
    for row_index, tokens in enumerate(token_sets):
        component_tokens[union_find.find(row_index)].update(tokens)
    group_by_root = {
        root: _stable_id(namespace, tokens) for root, tokens in component_tokens.items()
    }
    group_ids = [group_by_root[union_find.find(index)] for index in range(len(token_sets))]
    return group_ids, dict(Counter(group_ids))


def _assign_group_folds(
    group_sizes: Mapping[str, int],
    target_fractions: Mapping[str, float],
    salt: str,
) -> dict[str, str]:
    """Assign whole groups with a deterministic largest-deficit policy."""

    if len(group_sizes) < len(FOLDS):
        raise SplitError(
            f"{salt} has {len(group_sizes)} groups; at least {len(FOLDS)} are required"
        )
    total = sum(group_sizes.values())
    targets = {fold: total * float(target_fractions[fold]) for fold in FOLDS}
    counts = dict.fromkeys(FOLDS, 0)

    def order_key(item: tuple[str, int]) -> tuple[int, str]:
        group_id, size = item
        tie_break = hashlib.sha256(f"{salt}\0{group_id}".encode()).hexdigest()
        return -size, tie_break

    assignments: dict[str, str] = {}
    for group_id, size in sorted(group_sizes.items(), key=order_key):
        fold = max(
            FOLDS,
            key=lambda candidate: (
                (targets[candidate] - counts[candidate]) / targets[candidate],
                -FOLDS.index(candidate),
            ),
        )
        assignments[group_id] = fold
        counts[fold] += size
    if any(counts[fold] == 0 for fold in FOLDS):
        raise SplitError(f"{salt} assignment unexpectedly produced an empty fold: {counts}")
    return assignments


def _leaked_groups(group_ids: Sequence[str], folds: Sequence[str]) -> dict[str, tuple[str, ...]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for group_id, fold in zip(group_ids, folds, strict=True):
        seen[group_id].add(fold)
    return {
        group_id: tuple(sorted(group_folds))
        for group_id, group_folds in seen.items()
        if len(group_folds) > 1
    }


def _csv_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ASSIGNMENT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def build_split_bundle(
    config_path: Path, source_root: Path
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Build the deterministic M0-03 assignment bytes, manifest, and result."""

    config, input_records = verify_source_inputs(config_path, source_root)
    expected_inputs = config["expected_inputs"]
    r0_rows = _load_r0_rows(
        source_root / expected_inputs["r0_constitutional"]["path"],
        config["expected_r0_rows"],
    )
    upstream = _load_upstream_assignments(
        source_root / expected_inputs["upstream_assignments"]["path"],
        config["expected_original_r0_rows"],
    )
    original_smiles_by_row = []
    missing_upstream = set()
    for row in r0_rows:
        try:
            values = json.loads(row["original_isomeric_smiles_json"])
        except json.JSONDecodeError as exc:
            raise SplitError(
                f"{row['r0_structure_id']} has invalid original_isomeric_smiles_json"
            ) from exc
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise SplitError(f"{row['r0_structure_id']} original isomeric identities are invalid")
        original_smiles_by_row.append(sorted(set(values)))
        missing_upstream.update(value for value in values if value not in upstream)
    if missing_upstream:
        raise SplitError(
            f"{len(missing_upstream)} R0 structures are absent from upstream assignments; "
            f"first: {sorted(missing_upstream)[0]}"
        )

    parse_failures: Counter[str] = Counter()
    token_sets: dict[str, list[set[str]]] = {scheme: [] for scheme in SCHEMES}
    for row in r0_rows:
        exact = {f"exact:{row['leakage_group_id']}"}
        token_sets["source_study"].append(_source_study_tokens(row) | exact)
        head_tokens = _component_tokens(row, config["headgroup_fields"], "head", parse_failures)
        token_sets["headgroup"].append((head_tokens or exact) | exact)
        token_sets["linker_scaffold"].append(
            _linker_scaffold_tokens(row, config["linker_fields"], parse_failures) | exact
        )
        token_sets["component_family"].append(_component_family_tokens(row) | exact)

    group_ids_by_scheme: dict[str, list[str]] = {}
    group_sizes_by_scheme: dict[str, dict[str, int]] = {}
    group_folds_by_scheme: dict[str, dict[str, str]] = {}
    for scheme in SCHEMES:
        group_ids, group_sizes = _connected_group_ids(token_sets[scheme], scheme)
        group_ids_by_scheme[scheme] = group_ids
        group_sizes_by_scheme[scheme] = group_sizes
        group_folds_by_scheme[scheme] = _assign_group_folds(
            group_sizes,
            config["target_fractions"],
            f"m0-03:{config['seed']}:{scheme}",
        )

    assignment_rows: list[dict[str, str]] = []
    scheme_summaries: dict[str, dict[str, Any]] = {}
    upstream_diagnostics: dict[str, Any] = {
        "source_row_fold_counts": {},
        "rows_with_source_fold_conflicts": {},
        "group_leakage": {},
    }
    for scheme in SCHEMES:
        group_ids = group_ids_by_scheme[scheme]
        folds = [group_folds_by_scheme[scheme][group_id] for group_id in group_ids]
        leaked = _leaked_groups(group_ids, folds)
        if leaked:
            first_group = next(iter(leaked))
            raise SplitError(f"{scheme} group {first_group} spans folds {leaked[first_group]}")
        fold_counts = Counter(folds)
        scheme_summaries[scheme] = {
            "fold_counts": {fold: fold_counts[fold] for fold in FOLDS},
            "fold_fractions": {fold: fold_counts[fold] / len(r0_rows) for fold in FOLDS},
            "groups": len(group_sizes_by_scheme[scheme]),
            "largest_group": max(group_sizes_by_scheme[scheme].values()),
            "groups_spanning_folds": 0,
        }

        upstream_column = config["upstream_split_columns"][scheme]
        upstream_fold_sets = [
            {upstream[smiles][upstream_column] for smiles in original_smiles}
            for original_smiles in original_smiles_by_row
        ]
        upstream_diagnostics["source_row_fold_counts"][scheme] = dict(
            sorted(
                Counter(
                    fold for folds_for_row in upstream_fold_sets for fold in folds_for_row
                ).items()
            )
        )
        upstream_diagnostics["rows_with_source_fold_conflicts"][scheme] = sum(
            len(folds_for_row) > 1 for folds_for_row in upstream_fold_sets
        )
        upstream_group_folds: dict[str, set[str]] = defaultdict(set)
        for group_id, folds_for_row in zip(
            group_ids,
            upstream_fold_sets,
            strict=True,
        ):
            upstream_group_folds[group_id].update(folds_for_row)
        upstream_leaks = {
            group_id: tuple(sorted(folds))
            for group_id, folds in upstream_group_folds.items()
            if len(folds) > 1
        }
        upstream_diagnostics["group_leakage"][scheme] = {
            "local_groups_spanning_upstream_folds": len(upstream_leaks)
        }

    for row_index, row in enumerate(r0_rows):
        assignment: dict[str, str] = {
            "r0_structure_id": row["r0_structure_id"],
            "leakage_group_id": row["leakage_group_id"],
        }
        for scheme in SCHEMES:
            group_id = group_ids_by_scheme[scheme][row_index]
            assignment[f"{scheme}_group_id"] = group_id
            assignment[f"{scheme}_fold"] = group_folds_by_scheme[scheme][group_id]
        assignment_rows.append(assignment)

    assignment_payload = _csv_bytes(assignment_rows)
    assignment_sha256 = sha256_bytes(assignment_payload)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "frozen_utc": config["frozen_utc"],
        "seed": config["seed"],
        "randomness_used": False,
        "software": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
        },
        "row_count": len(r0_rows),
        "folds": list(FOLDS),
        "inputs": input_records,
        "assignment_policy": {
            "algorithm": "whole-group deterministic largest normalized deficit",
            "target_fractions": config["target_fractions"],
            "multi_annotation_policy": (
                "Rows sharing any scheme token are collapsed into connected components before "
                "fold assignment."
            ),
            "exact_structure_policy": (
                "leakage_group_id is included in every scheme's connected components."
            ),
        },
        "group_definitions": {
            "source_study": (
                "study_split_groups_json; observed_source_ids is the fallback when study IDs "
                "are absent"
            ),
            "headgroup": (
                "configured head or amine fields from component_holdout_groups_json; parseable "
                "SMILES are canonicalized"
            ),
            "linker_scaffold": (
                "heteroatom connector core plus configured linker fields from "
                "component_holdout_groups_json"
            ),
            "component_family": "exact reaction_family_holdout_groups signature",
        },
        "annotation_parse_failures": dict(sorted(parse_failures.items())),
        "annotation_parse_failure_policy": (
            "Unparseable annotated component SMILES remain distinct raw-string grouping tokens; "
            "they are not dropped or treated as valid molecules."
        ),
        "schemes": scheme_summaries,
        "upstream_reference": {
            "format": "compose_lipid_corpus_splits_v1",
            **upstream_diagnostics,
        },
        "outputs": {
            "assignments": {
                "path": ASSIGNMENT_FILENAME,
                "bytes": len(assignment_payload),
                "sha256": assignment_sha256,
                "columns": list(ASSIGNMENT_FIELDS),
            }
        },
        "all_groups_leak_free": True,
        "downstream_contract": (
            "Load this bundle with forge.corpus.r0_splits.load_frozen_r0_splits. "
            "Do not reconstruct or repartition groups in M0-04 or later tasks."
        ),
    }
    manifest_payload = _json_bytes(manifest)
    frozen_paths = config["frozen_artifact_paths"]
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "task": "M0-03",
        "generated_utc": config["frozen_utc"],
        "seed": config["seed"],
        "randomness_used": False,
        "software": manifest["software"],
        "inputs": input_records,
        "frozen_artifacts": {
            "assignments": {
                "path": frozen_paths["assignments"],
                "sha256": assignment_sha256,
            },
            "manifest": {
                "path": frozen_paths["manifest"],
                "sha256": sha256_bytes(manifest_payload),
            },
        },
        "row_count": len(r0_rows),
        "schemes": scheme_summaries,
        "annotation_parse_failures": dict(sorted(parse_failures.items())),
        "annotation_parse_failure_policy": manifest["annotation_parse_failure_policy"],
        "upstream_reference_diagnostics": upstream_diagnostics,
        "decision": {
            "status": "frozen",
            "all_groups_leak_free": True,
            "downstream_rule": (
                "M0-04 must load the selected scheme from the frozen bundle and extract "
                "components from R0_train only."
            ),
            "component_family_limitation": (
                "R0 has three observed family signatures, so this stress split is necessarily "
                "coarse and remains a secondary robustness analysis."
            ),
        },
    }
    return assignment_payload, manifest, result


def _require_matching_artifact(path: Path, payload: bytes) -> None:
    if path.read_bytes() != payload:
        raise SplitError(f"refusing to modify frozen artifact {path}; create a new version instead")


def write_frozen_split_bundle(
    assignment_payload: bytes,
    manifest: Mapping[str, Any],
    result: Mapping[str, Any],
    output_dir: Path,
    result_path: Path,
) -> None:
    """Atomically create immutable split artifacts, or validate byte-identical reruns."""

    manifest_payload = _json_bytes(manifest)
    result_payload = _json_bytes(result)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        if not output_dir.is_dir():
            raise SplitError(f"frozen split path is not a directory: {output_dir}")
        for path, payload in (
            (output_dir / ASSIGNMENT_FILENAME, assignment_payload),
            (output_dir / MANIFEST_FILENAME, manifest_payload),
        ):
            if not path.exists():
                raise SplitError(f"existing frozen split bundle is incomplete: missing {path}")
            _require_matching_artifact(path, payload)
    if result_path.exists():
        _require_matching_artifact(result_path, result_payload)

    if not output_dir.exists():
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
        try:
            (staging / ASSIGNMENT_FILENAME).write_bytes(assignment_payload)
            (staging / MANIFEST_FILENAME).write_bytes(manifest_payload)
            os.replace(staging, output_dir)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    if result_path.exists():
        return
    result_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{result_path.name}.", dir=result_path.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(result_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, result_path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _validate_scheme_and_fold(scheme: str, fold: str) -> None:
    if scheme not in SCHEMES:
        raise SplitError(f"unknown split scheme {scheme!r}; choose one of {SCHEMES}")
    if fold not in FOLDS:
        raise SplitError(f"unknown R0 fold {fold!r}; choose one of {FOLDS}")


def load_frozen_r0_splits(output_dir: Path) -> FrozenR0Splits:
    """Load and fully validate a frozen split bundle without recomputing groups."""

    manifest_path = output_dir / MANIFEST_FILENAME
    assignment_path = output_dir / ASSIGNMENT_FILENAME
    manifest = _load_json(manifest_path, "frozen M0-03 manifest")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise SplitError(
            f"unsupported frozen split schema {manifest.get('schema_version')!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest.get("all_groups_leak_free") is not True:
        raise SplitError("frozen M0-03 manifest does not declare all groups leak-free")
    output_record = manifest.get("outputs", {}).get("assignments", {})
    if tuple(output_record.get("columns", ())) != ASSIGNMENT_FIELDS:
        raise SplitError("frozen assignment columns do not match the supported schema")
    expected_hash = output_record.get("sha256")
    actual_hash = sha256_file(assignment_path)
    if expected_hash != actual_hash:
        raise SplitError(
            f"hash mismatch for frozen {ASSIGNMENT_FILENAME}: "
            f"expected {expected_hash}, observed {actual_hash}"
        )
    rows = _read_csv(assignment_path, set(ASSIGNMENT_FIELDS), "frozen R0 assignments")
    if len(rows) != manifest.get("row_count"):
        raise SplitError(
            f"frozen assignment row count {len(rows)} does not match manifest "
            f"{manifest.get('row_count')}"
        )
    structure_ids = [row["r0_structure_id"] for row in rows]
    if len(structure_ids) != len(set(structure_ids)):
        raise SplitError("frozen assignments contain duplicate r0_structure_id values")
    for scheme in SCHEMES:
        fold_field = f"{scheme}_fold"
        group_field = f"{scheme}_group_id"
        invalid_folds = sorted({row[fold_field] for row in rows}.difference(FOLDS))
        if invalid_folds:
            raise SplitError(f"{scheme} contains invalid folds: {invalid_folds}")
        leaked = _leaked_groups(
            [row[group_field] for row in rows], [row[fold_field] for row in rows]
        )
        if leaked:
            first_group = next(iter(leaked))
            raise SplitError(f"frozen {scheme} group {first_group} spans folds")
        observed_counts = Counter(row[fold_field] for row in rows)
        expected_counts = manifest.get("schemes", {}).get(scheme, {}).get("fold_counts", {})
        if any(observed_counts[fold] != expected_counts.get(fold) for fold in FOLDS):
            raise SplitError(f"frozen {scheme} fold counts do not match the manifest")
        observed_groups = len({row[group_field] for row in rows})
        expected_groups = manifest.get("schemes", {}).get(scheme, {}).get("groups")
        if observed_groups != expected_groups:
            raise SplitError(f"frozen {scheme} group count does not match the manifest")
    return FrozenR0Splits(manifest=manifest, assignments=tuple(rows))
