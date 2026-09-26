"""Exact-source Ugi semantics for chemistry-native whole-graph supervision."""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.synthesis.engine.qualified_forward import load_qualified_forward_reaction

CONFIG_SCHEMA_VERSION = "m0_07_ugi_semantic_annotations_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_ugi_semantic_annotations.v1"

ROLE_NAMES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
COMPONENT_FIELDS = ("A_smiles", "B_smiles", "C_smiles")

PRODUCT_FIELDS = (
    "product_id",
    "product_smiles",
    "component_smiles_json",
    "source_evidence_record_id",
    "raw_forward_outcomes",
    "target_matching_outcomes",
    "semantic_signature_multiplicity",
    "source_mapping_multiplicity",
    "core_atom_indices_json",
    "role_anchor_indices_json",
    "maximum_distance_to_core",
    "atom_rows",
    "bond_rows",
    "component_mapping_rows",
    "annotation_sha256",
)
ATOM_FIELDS = (
    "product_id",
    "product_atom_index",
    "atomic_number",
    "formal_charge",
    "origin_role",
    "is_ugi_core",
    "core_position",
    "reaction_map_number",
    "is_role_anchor",
    "source_component_atom_indices_json",
    "source_component_symmetry_class",
    "distance_to_nearest_core",
    "distance_to_origin_anchor",
    "distances_to_role_anchors_json",
    "distances_to_core_positions_json",
)
BOND_FIELDS = (
    "product_id",
    "bond_index",
    "begin_atom_index",
    "end_atom_index",
    "bond_type",
    "is_aromatic",
    "is_conjugated",
    "is_in_ring",
    "is_core_internal",
    "is_core_boundary",
    "boundary_role",
)
COMPONENT_MAPPING_FIELDS = (
    "product_id",
    "role",
    "component_smiles",
    "component_atom_index",
    "atomic_number",
    "formal_charge",
    "symmetry_class",
    "mapping_status",
    "possible_product_atom_indices_json",
    "possible_reaction_map_numbers_json",
    "target_matching_outcomes_present",
)


class UgiSemanticAnnotationError(ValueError):
    """Raised when exact Ugi semantic annotation cannot be established."""


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise UgiSemanticAnnotationError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise UgiSemanticAnnotationError(f"{label} is invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise UgiSemanticAnnotationError(f"{label} must contain an object")
    return value


def _portable(path: Path, repo_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _verify_input(
    repo_root: Path,
    specification: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    relative = specification.get("path")
    expected = specification.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected, str) or len(expected) != 64:
        raise UgiSemanticAnnotationError(f"{label} input specification is incomplete")
    path = repo_root / relative
    if not path.is_file():
        raise UgiSemanticAnnotationError(f"{label} not found: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise UgiSemanticAnnotationError(
            f"{label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": relative, "sha256": observed, "bytes": path.stat().st_size}


def _read_csv(path: Path, required: set[str], label: str) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else Path.open
    try:
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise UgiSemanticAnnotationError(f"{label} is missing fields: {sorted(missing)}")
            return [dict(row) for row in reader]
    except FileNotFoundError as exc:
        raise UgiSemanticAnnotationError(f"{label} not found: {path}") from exc


def _molecule(smiles: str, label: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles:
        raise UgiSemanticAnnotationError(f"{label} lacks SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise UgiSemanticAnnotationError(f"{label} contains invalid SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiSemanticAnnotationError(f"{label} must contain one connected graph")
    return molecule


def _canonical_component(smiles: str, label: str) -> tuple[str, Chem.Mol]:
    molecule = _molecule(smiles, label)
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    reparsed = _molecule(canonical, f"{label} canonical form")
    return canonical, reparsed


def _gzip_csv(rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in fields})
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as archive:
        archive.write(text.getvalue().encode())
    return output.getvalue()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _artifact_metadata(payload: bytes) -> dict[str, Any]:
    return {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def _source_evidence_index(
    rows: Sequence[Mapping[str, str]],
    *,
    level: str,
    evidence_basis: str,
    disposition: str,
) -> dict[str, Mapping[str, str]]:
    selected: dict[str, Mapping[str, str]] = {}
    prefix = "agile-measured-"
    for row in rows:
        if (
            row["level"] != level
            or row["evidence_basis"] != evidence_basis
            or row["disposition"] != disposition
        ):
            continue
        target_id = row["target_id"]
        if not target_id.startswith(prefix):
            raise UgiSemanticAnnotationError(
                f"exact measured evidence target has unexpected ID: {target_id}"
            )
        label = target_id.removeprefix(prefix)
        if label in selected:
            raise UgiSemanticAnnotationError(f"duplicate exact evidence for {label}")
        selected[label] = row
    return selected


def _canonical_output_order(product: Chem.Mol) -> tuple[str, tuple[int, ...]]:
    smiles = Chem.MolToSmiles(
        product,
        canonical=True,
        isomericSmiles=False,
        ignoreAtomMapNumbers=True,
    )
    if not product.HasProp("_smilesAtomOutputOrder"):
        raise UgiSemanticAnnotationError("RDKit did not expose canonical SMILES atom order")
    try:
        order_raw = json.loads(product.GetProp("_smilesAtomOutputOrder"))
    except json.JSONDecodeError as exc:
        raise UgiSemanticAnnotationError("RDKit canonical atom order is invalid") from exc
    order = tuple(int(index) for index in order_raw)
    if sorted(order) != list(range(product.GetNumAtoms())):
        raise UgiSemanticAnnotationError("RDKit canonical atom order is not a permutation")
    reparsed = _molecule(smiles, "canonical product")
    if reparsed.GetNumAtoms() != product.GetNumAtoms():
        raise UgiSemanticAnnotationError("canonical product atom count changed")
    return smiles, order


def _component_symmetry_classes(component: Chem.Mol) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in Chem.CanonicalRankAtoms(
            component,
            breakTies=False,
            includeChirality=False,
            includeIsotopes=True,
            includeAtomMaps=False,
        )
    )


def _is_connected_region(
    product: Chem.Mol,
    atom_indices: set[int],
) -> bool:
    if not atom_indices:
        return False
    start = min(atom_indices)
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in product.GetAtomWithIdx(current).GetNeighbors():
            index = neighbor.GetIdx()
            if index in atom_indices and index not in seen:
                seen.add(index)
                stack.append(index)
    return seen == atom_indices


def _outcome_annotation(
    product: Chem.Mol,
    component_smiles: Sequence[str],
    components: Sequence[Chem.Mol],
    roles: Sequence[str],
) -> dict[str, Any]:
    product_smiles, order = _canonical_output_order(product)
    old_to_new = {old: new for new, old in enumerate(order)}
    symmetry = tuple(_component_symmetry_classes(component) for component in components)

    atom_records: list[dict[str, Any]] = []
    introduced_indices: list[int] = []
    core_map_to_index: dict[int, int] = {}
    source_present: set[tuple[int, int]] = set()
    for new_index, old_index in enumerate(order):
        atom = product.GetAtomWithIdx(old_index)
        properties = atom.GetPropsAsDict(includePrivate=True, includeComputed=False)
        reactant_index = int(properties.get("react_idx", -1))
        source_atom_index = int(properties.get("react_atom_idx", -1))
        reaction_map_number = int(properties.get("old_mapno", 0))
        if atom.GetAtomMapNum() != 0:
            raise UgiSemanticAnnotationError("ordinary product atom maps unexpectedly survived")
        if reactant_index >= 0:
            if reactant_index >= len(roles) or source_atom_index < 0:
                raise UgiSemanticAnnotationError("inherited product atom has invalid origin")
            if source_atom_index >= components[reactant_index].GetNumAtoms():
                raise UgiSemanticAnnotationError("product source atom index is out of range")
            origin_role = roles[reactant_index]
            source_symmetry_class: int | None = symmetry[reactant_index][source_atom_index]
            source_present.add((reactant_index, source_atom_index))
        else:
            if source_atom_index >= 0 or reaction_map_number:
                raise UgiSemanticAnnotationError("introduced atom carries inconsistent origin")
            origin_role = "assembly_introduced"
            source_atom_index = -1
            source_symmetry_class = None
            introduced_indices.append(new_index)
        if reaction_map_number:
            if reaction_map_number in core_map_to_index:
                raise UgiSemanticAnnotationError(
                    f"reaction map {reaction_map_number} appears more than once"
                )
            core_map_to_index[reaction_map_number] = new_index
        atom_records.append(
            {
                "product_atom_index": new_index,
                "old_atom_index": old_index,
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "origin_role": origin_role,
                "source_atom_index": source_atom_index,
                "source_symmetry_class": source_symmetry_class,
                "reaction_map_number": reaction_map_number,
            }
        )

    introduced_positions = {
        index: f"template_introduced_{ordinal}"
        for ordinal, index in enumerate(sorted(introduced_indices))
    }
    core_positions = {index: f"map_{map_number}" for map_number, index in core_map_to_index.items()}
    core_positions.update(introduced_positions)
    core_indices = set(core_positions)
    if not core_indices:
        raise UgiSemanticAnnotationError("reaction product contains no Ugi core")

    bond_records: list[dict[str, Any]] = []
    boundary_anchor_candidates: dict[str, set[int]] = defaultdict(set)
    for bond in product.GetBonds():
        begin = old_to_new[bond.GetBeginAtomIdx()]
        end = old_to_new[bond.GetEndAtomIdx()]
        begin, end = sorted((begin, end))
        begin_core = begin in core_indices
        end_core = end in core_indices
        is_boundary = begin_core != end_core
        boundary_role = ""
        if is_boundary:
            core_index = begin if begin_core else end
            exterior_index = end if begin_core else begin
            boundary_role = atom_records[exterior_index]["origin_role"]
            if atom_records[core_index]["origin_role"] != boundary_role:
                raise UgiSemanticAnnotationError("core boundary does not preserve reagent origin")
            boundary_anchor_candidates[boundary_role].add(core_index)
        bond_records.append(
            {
                "begin_atom_index": begin,
                "end_atom_index": end,
                "bond_type": str(bond.GetBondType()),
                "is_aromatic": bond.GetIsAromatic(),
                "is_conjugated": bond.GetIsConjugated(),
                "is_in_ring": bond.IsInRing(),
                "is_core_internal": begin_core and end_core,
                "is_core_boundary": is_boundary,
                "boundary_role": boundary_role,
            }
        )
    bond_records.sort(
        key=lambda row: (
            row["begin_atom_index"],
            row["end_atom_index"],
            row["bond_type"],
        )
    )

    role_anchor_indices: dict[str, int] = {}
    for role in roles:
        candidates = boundary_anchor_candidates.get(role, set())
        if len(candidates) != 1:
            raise UgiSemanticAnnotationError(
                f"{role} has {len(candidates)} product core anchors; expected one"
            )
        role_anchor_indices[role] = next(iter(candidates))

    distance_matrix = Chem.GetDistanceMatrix(product)
    core_old_indices = [order[index] for index in sorted(core_indices)]
    for record in atom_records:
        old_index = record["old_atom_index"]
        distances_to_core_positions = {
            core_positions[index]: int(distance_matrix[old_index, order[index]])
            for index in sorted(core_indices)
        }
        distances_to_role_anchors = {
            role: int(distance_matrix[old_index, order[anchor]])
            for role, anchor in role_anchor_indices.items()
        }
        record["distance_to_nearest_core"] = min(
            int(distance_matrix[old_index, core_old]) for core_old in core_old_indices
        )
        record["distances_to_core_positions"] = distances_to_core_positions
        record["distances_to_role_anchors"] = distances_to_role_anchors
        origin = record["origin_role"]
        record["distance_to_origin_anchor"] = (
            distances_to_role_anchors[origin] if origin in role_anchor_indices else None
        )
        record["core_position"] = core_positions.get(record["product_atom_index"], "")
        record["is_ugi_core"] = record["product_atom_index"] in core_indices
        record["is_role_anchor"] = record["product_atom_index"] in set(role_anchor_indices.values())

    connected_by_role = {}
    for role in roles:
        role_old_indices = {
            order[record["product_atom_index"]]
            for record in atom_records
            if record["origin_role"] == role
        }
        connected_by_role[role] = _is_connected_region(product, role_old_indices)

    component_mappings: list[dict[str, Any]] = []
    product_index_by_origin = {
        (roles.index(record["origin_role"]), record["source_atom_index"]): record[
            "product_atom_index"
        ]
        for record in atom_records
        if record["origin_role"] in roles
    }
    reaction_map_by_origin = {
        (roles.index(record["origin_role"]), record["source_atom_index"]): record[
            "reaction_map_number"
        ]
        for record in atom_records
        if record["origin_role"] in roles and record["reaction_map_number"]
    }
    for reactant_index, (role, smiles, component) in enumerate(
        zip(roles, component_smiles, components, strict=True)
    ):
        classes = symmetry[reactant_index]
        for atom in component.GetAtoms():
            source_index = atom.GetIdx()
            key = (reactant_index, source_index)
            component_mappings.append(
                {
                    "role": role,
                    "component_smiles": smiles,
                    "component_atom_index": source_index,
                    "atomic_number": atom.GetAtomicNum(),
                    "formal_charge": atom.GetFormalCharge(),
                    "symmetry_class": classes[source_index],
                    "product_atom_index": product_index_by_origin.get(key),
                    "reaction_map_number": reaction_map_by_origin.get(key),
                }
            )

    semantic_atom_signature = tuple(
        (
            record["atomic_number"],
            record["formal_charge"],
            record["origin_role"],
            record["source_symmetry_class"],
            record["reaction_map_number"],
            record["core_position"],
            record["distance_to_nearest_core"],
            tuple(sorted(record["distances_to_role_anchors"].items())),
            tuple(sorted(record["distances_to_core_positions"].items())),
        )
        for record in atom_records
    )
    source_index_signature = tuple(
        (
            record["origin_role"],
            record["source_atom_index"],
            record["reaction_map_number"],
        )
        for record in atom_records
    )
    semantic_bond_signature = tuple(
        (
            record["begin_atom_index"],
            record["end_atom_index"],
            record["bond_type"],
            record["is_core_internal"],
            record["is_core_boundary"],
            record["boundary_role"],
        )
        for record in bond_records
    )
    return {
        "product_smiles": product_smiles,
        "atom_records": atom_records,
        "bond_records": bond_records,
        "component_mappings": component_mappings,
        "core_positions": core_positions,
        "role_anchor_indices": role_anchor_indices,
        "connected_by_role": connected_by_role,
        "semantic_signature": (semantic_atom_signature, semantic_bond_signature),
        "source_index_signature": source_index_signature,
        "source_present": source_present,
    }


def _target_annotations(
    compiled: Any,
    component_smiles: Sequence[str],
    components: Sequence[Chem.Mol],
    target_smiles: str,
    *,
    max_outcomes: int,
) -> tuple[int, list[dict[str, Any]]]:
    with rdBase.BlockLogs():
        outcomes = compiled.reaction.RunReactants(
            tuple(components),
            maxProducts=max_outcomes,
        )
    if len(outcomes) >= max_outcomes:
        raise UgiSemanticAnnotationError(
            f"{compiled.reaction_id} reached max_products={max_outcomes}"
        )
    target = _canonical_component(target_smiles, "target product")[0]
    matched = []
    for outcome_index, outcome in enumerate(outcomes):
        if len(outcome) != 1:
            raise UgiSemanticAnnotationError(
                f"forward outcome {outcome_index} did not contain one product"
            )
        product = outcome[0]
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(product)
        except Exception as exc:
            raise UgiSemanticAnnotationError(
                f"forward outcome {outcome_index} did not sanitize"
            ) from exc
        product_smiles = Chem.MolToSmiles(
            product,
            canonical=True,
            isomericSmiles=False,
            ignoreAtomMapNumbers=True,
        )
        if product_smiles != target:
            continue
        matched.append(
            _outcome_annotation(
                product,
                component_smiles,
                components,
                compiled.role_names,
            )
        )
    if not matched:
        raise UgiSemanticAnnotationError("frozen transform did not reconstruct target")
    return len(outcomes), matched


def _aggregate_product(
    product_id: str,
    target_smiles: str,
    component_smiles: Sequence[str],
    components: Sequence[Chem.Mol],
    source_evidence_record_id: str,
    raw_outcomes: int,
    outcomes: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    semantic_signatures = {outcome["semantic_signature"] for outcome in outcomes}
    source_signatures = {outcome["source_index_signature"] for outcome in outcomes}
    if len(semantic_signatures) != 1:
        raise UgiSemanticAnnotationError(
            f"{product_id} has conflicting semantic annotations across target outcomes"
        )
    product_smiles = {outcome["product_smiles"] for outcome in outcomes}
    if product_smiles != {_canonical_component(target_smiles, product_id)[0]}:
        raise UgiSemanticAnnotationError(f"{product_id} target identity changed")

    first = outcomes[0]
    atom_rows: list[dict[str, Any]] = []
    for atom_index, record in enumerate(first["atom_records"]):
        source_indices = sorted(
            {
                outcome["atom_records"][atom_index]["source_atom_index"]
                for outcome in outcomes
                if outcome["atom_records"][atom_index]["source_atom_index"] >= 0
            }
        )
        atom_rows.append(
            {
                "product_id": product_id,
                "product_atom_index": atom_index,
                "atomic_number": record["atomic_number"],
                "formal_charge": record["formal_charge"],
                "origin_role": record["origin_role"],
                "is_ugi_core": record["is_ugi_core"],
                "core_position": record["core_position"],
                "reaction_map_number": record["reaction_map_number"] or "",
                "is_role_anchor": record["is_role_anchor"],
                "source_component_atom_indices_json": _compact(source_indices),
                "source_component_symmetry_class": (
                    ""
                    if record["source_symmetry_class"] is None
                    else record["source_symmetry_class"]
                ),
                "distance_to_nearest_core": record["distance_to_nearest_core"],
                "distance_to_origin_anchor": (
                    ""
                    if record["distance_to_origin_anchor"] is None
                    else record["distance_to_origin_anchor"]
                ),
                "distances_to_role_anchors_json": _compact(record["distances_to_role_anchors"]),
                "distances_to_core_positions_json": _compact(record["distances_to_core_positions"]),
            }
        )

    bond_rows = []
    for bond_index, record in enumerate(first["bond_records"]):
        bond_rows.append(
            {
                "product_id": product_id,
                "bond_index": bond_index,
                **record,
            }
        )

    mapping_aggregate: dict[tuple[str, int], dict[str, Any]] = {}
    for role, smiles, component in zip(
        ROLE_NAMES,
        component_smiles,
        components,
        strict=True,
    ):
        classes = _component_symmetry_classes(component)
        for atom in component.GetAtoms():
            mapping_aggregate[(role, atom.GetIdx())] = {
                "product_id": product_id,
                "role": role,
                "component_smiles": smiles,
                "component_atom_index": atom.GetIdx(),
                "atomic_number": atom.GetAtomicNum(),
                "formal_charge": atom.GetFormalCharge(),
                "symmetry_class": classes[atom.GetIdx()],
                "product_atom_indices": set(),
                "reaction_map_numbers": set(),
                "outcomes_present": 0,
            }
    for outcome in outcomes:
        for mapping in outcome["component_mappings"]:
            aggregate = mapping_aggregate[(mapping["role"], mapping["component_atom_index"])]
            if mapping["product_atom_index"] is not None:
                aggregate["product_atom_indices"].add(mapping["product_atom_index"])
                aggregate["outcomes_present"] += 1
            if mapping["reaction_map_number"] is not None:
                aggregate["reaction_map_numbers"].add(mapping["reaction_map_number"])

    component_rows = []
    for key in sorted(mapping_aggregate):
        record = mapping_aggregate[key]
        component_rows.append(
            {
                "product_id": product_id,
                "role": record["role"],
                "component_smiles": record["component_smiles"],
                "component_atom_index": record["component_atom_index"],
                "atomic_number": record["atomic_number"],
                "formal_charge": record["formal_charge"],
                "symmetry_class": record["symmetry_class"],
                "mapping_status": ("inherited" if record["product_atom_indices"] else "deleted"),
                "possible_product_atom_indices_json": _compact(
                    sorted(record["product_atom_indices"])
                ),
                "possible_reaction_map_numbers_json": _compact(
                    sorted(record["reaction_map_numbers"])
                ),
                "target_matching_outcomes_present": record["outcomes_present"],
            }
        )

    annotation_sha256 = _sha256_json(
        {
            "atoms": atom_rows,
            "bonds": bond_rows,
            "component_mappings": component_rows,
        }
    )
    product_row = {
        "product_id": product_id,
        "product_smiles": first["product_smiles"],
        "component_smiles_json": _compact(dict(zip(ROLE_NAMES, component_smiles, strict=True))),
        "source_evidence_record_id": source_evidence_record_id,
        "raw_forward_outcomes": raw_outcomes,
        "target_matching_outcomes": len(outcomes),
        "semantic_signature_multiplicity": len(semantic_signatures),
        "source_mapping_multiplicity": len(source_signatures),
        "core_atom_indices_json": _compact(
            dict(sorted(first["core_positions"].items(), key=lambda item: item[1]))
        ),
        "role_anchor_indices_json": _compact(first["role_anchor_indices"]),
        "maximum_distance_to_core": max(
            record["distance_to_nearest_core"] for record in first["atom_records"]
        ),
        "atom_rows": len(atom_rows),
        "bond_rows": len(bond_rows),
        "component_mapping_rows": len(component_rows),
        "annotation_sha256": annotation_sha256,
    }
    return product_row, atom_rows, bond_rows, component_rows


def annotate_qualified_ugi_product(
    compiled: Any,
    *,
    product_id: str,
    target_smiles: str,
    component_smiles_by_role: Mapping[str, str],
    source_evidence_record_id: str,
    max_outcomes: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create exact atom-origin semantics for one qualified Ugi product.

    This public single-product boundary lets Phase 1 annotate virtual products
    through the same atom-mapped reaction mechanics used for the source-exact
    measured ledger.  It does not infer origins from product substructures.
    """

    if tuple(compiled.role_names) != ROLE_NAMES:
        raise UgiSemanticAnnotationError(f"qualified role order changed: {compiled.role_names!r}")
    if set(component_smiles_by_role) != set(ROLE_NAMES):
        raise UgiSemanticAnnotationError(f"component roles must be exactly {ROLE_NAMES!r}")
    if not product_id or not source_evidence_record_id:
        raise UgiSemanticAnnotationError("product and evidence identifiers must be nonempty")
    if isinstance(max_outcomes, bool) or not isinstance(max_outcomes, int) or max_outcomes <= 0:
        raise UgiSemanticAnnotationError("max outcomes must be a positive integer")

    component_pairs = [
        _canonical_component(component_smiles_by_role[role], f"{product_id}/{role}")
        for role in ROLE_NAMES
    ]
    component_smiles = tuple(pair[0] for pair in component_pairs)
    components = tuple(pair[1] for pair in component_pairs)
    target = _canonical_component(target_smiles, f"{product_id} product")[0]
    raw_outcomes, outcomes = _target_annotations(
        compiled,
        component_smiles,
        components,
        target,
        max_outcomes=max_outcomes,
    )
    if any(
        set(outcome["connected_by_role"]) != set(ROLE_NAMES)
        or not all(outcome["connected_by_role"].values())
        for outcome in outcomes
    ):
        raise UgiSemanticAnnotationError(
            f"{product_id} contains a disconnected component-origin region"
        )
    return _aggregate_product(
        product_id,
        target,
        component_smiles,
        components,
        source_evidence_record_id,
        raw_outcomes,
        outcomes,
    )


def _validate_exact_source_record(
    label: str,
    evidence: Mapping[str, str],
    target_smiles: str,
    component_smiles: Sequence[str],
) -> None:
    evidence_target = _canonical_component(
        evidence["target_canonical_smiles"],
        f"{label} source-evidence target",
    )[0]
    target = _canonical_component(target_smiles, f"{label} target")[0]
    if evidence_target != target:
        raise UgiSemanticAnnotationError(f"{label} source-evidence target disagrees")
    try:
        evidence_components_raw = json.loads(evidence["component_smiles_json"])
    except json.JSONDecodeError as exc:
        raise UgiSemanticAnnotationError(f"{label} source-evidence components are invalid") from exc
    if not isinstance(evidence_components_raw, list) or len(evidence_components_raw) != 3:
        raise UgiSemanticAnnotationError(f"{label} source-evidence components must contain A/B/C")
    evidence_components = [
        _canonical_component(str(smiles), f"{label} evidence component {index}")[0]
        for index, smiles in enumerate(evidence_components_raw)
    ]
    if evidence_components != list(component_smiles):
        raise UgiSemanticAnnotationError(f"{label} source-evidence component mapping disagrees")


def _assert_expected(
    expected: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    for key, expected_value in expected.items():
        observed = summary.get(key)
        if observed != expected_value:
            raise UgiSemanticAnnotationError(
                f"{key} changed: expected {expected_value!r}, observed {observed!r}"
            )


def build_ugi_semantic_annotations(
    config_path: Path,
    output_dir: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Build deterministic whole-product Ugi semantic annotation ledgers."""

    config = _load_json(config_path, "Ugi semantic annotation config")
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise UgiSemanticAnnotationError("unsupported Ugi semantic annotation config")
    if config.get("seed") != 1729:
        raise UgiSemanticAnnotationError("Ugi semantic annotation seed must remain 1729")
    scope = config.get("scope")
    expected = config.get("expected")
    if not isinstance(scope, dict) or not isinstance(expected, dict):
        raise UgiSemanticAnnotationError("config must contain scope and expected objects")
    whole_graph_policy = scope.get("whole_graph_policy")
    if whole_graph_policy != {
        "one_connected_product_graph": True,
        "component_catalog_ids_in_model_state": False,
        "independent_fragment_generation": False,
        "region_labels_are_node_level_supervision": True,
        "distances_are_derived_not_generated": True,
    }:
        raise UgiSemanticAnnotationError("whole-graph policy changed")

    input_specs = config.get("inputs")
    if not isinstance(input_specs, dict):
        raise UgiSemanticAnnotationError("config lacks inputs")
    verified = {
        name: _verify_input(repo_root, specification, name)
        for name, specification in sorted(input_specs.items())
        if isinstance(specification, dict)
    }
    required_inputs = {
        "curated_oracle_data",
        "agile_reconciliation_result",
        "source_evidence_result",
        "source_evidence_ledger",
        "qualified_reactions",
        "ugi_variant",
        "ugi_assembly_qualification",
    }
    if set(verified) != required_inputs:
        raise UgiSemanticAnnotationError("Ugi semantic annotation inputs are incomplete")

    reconciliation = _load_json(
        repo_root / verified["agile_reconciliation_result"]["path"],
        "AGILE reconciliation result",
    )
    if reconciliation.get("summary", {}).get("curated_single_structure_records") != 1100:
        raise UgiSemanticAnnotationError("AGILE reconciliation no longer admits 1,100 records")
    source_result = _load_json(
        repo_root / verified["source_evidence_result"]["path"],
        "source evidence result",
    )
    if source_result.get("summary", {}).get("measured_l1_admit_exact") != 1100:
        raise UgiSemanticAnnotationError("source gate no longer admits 1,100 exact L1 records")
    qualification = _load_json(
        repo_root / verified["ugi_assembly_qualification"]["path"],
        "Ugi assembly qualification",
    )
    if (
        qualification.get("summary", {}).get("products_reconstructed_exactly") != 1200
        or qualification.get("decision", {}).get("all_measured_products_pass") is not True
    ):
        raise UgiSemanticAnnotationError("frozen Ugi assembly qualification no longer passes")

    curated = _read_csv(
        repo_root / verified["curated_oracle_data"]["path"],
        {
            "label",
            "model_smiles",
            "A_smiles",
            "B_smiles",
            "C_smiles",
        },
        "curated oracle data",
    )
    evidence_rows = _read_csv(
        repo_root / verified["source_evidence_ledger"]["path"],
        {
            "evidence_record_id",
            "level",
            "target_id",
            "reaction_family_id",
            "target_canonical_smiles",
            "component_smiles_json",
            "evidence_basis",
            "disposition",
        },
        "source evidence ledger",
    )
    evidence_index = _source_evidence_index(
        evidence_rows,
        level=str(scope["source_evidence_level"]),
        evidence_basis=str(scope["source_evidence_basis"]),
        disposition=str(scope["source_evidence_disposition"]),
    )
    if len(evidence_index) != expected["source_exact_records"]:
        raise UgiSemanticAnnotationError("exact source-evidence record count changed")

    reaction_id = str(scope["reaction_id"])
    compiled = load_qualified_forward_reaction(
        repo_root / verified["qualified_reactions"]["path"],
        repo_root / verified["ugi_variant"]["path"],
        reaction_id=reaction_id,
    )
    if compiled.role_names != ROLE_NAMES:
        raise UgiSemanticAnnotationError(f"qualified role order changed: {compiled.role_names!r}")
    max_outcomes = scope.get("max_forward_outcomes_per_product")
    if isinstance(max_outcomes, bool) or not isinstance(max_outcomes, int):
        raise UgiSemanticAnnotationError("max forward outcomes must be an integer")

    product_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    bond_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    raw_outcome_distribution: Counter[str] = Counter()
    target_outcome_distribution: Counter[str] = Counter()
    source_mapping_distribution: Counter[str] = Counter()
    semantic_distribution: Counter[str] = Counter()
    maximum_distance_distribution: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    deleted_by_role: Counter[str] = Counter()
    connected_role_regions = 0
    permutation_mismatches = 0

    labels = [row["label"] for row in curated]
    if len(curated) != expected["products"] or len(set(labels)) != expected["unique_labels"]:
        raise UgiSemanticAnnotationError("curated product label count changed")
    if set(labels) != set(evidence_index):
        raise UgiSemanticAnnotationError("curated and exact-source label sets disagree")

    for row in sorted(curated, key=lambda item: item["label"]):
        label = row["label"]
        component_pairs = [
            _canonical_component(row[field], f"{label}/{role}")
            for field, role in zip(COMPONENT_FIELDS, ROLE_NAMES, strict=True)
        ]
        component_smiles = tuple(pair[0] for pair in component_pairs)
        components = tuple(pair[1] for pair in component_pairs)
        target_smiles = _canonical_component(row["model_smiles"], f"{label} product")[0]
        evidence = evidence_index[label]
        if evidence["reaction_family_id"] != reaction_id:
            raise UgiSemanticAnnotationError(f"{label} source reaction family changed")
        _validate_exact_source_record(
            label,
            evidence,
            target_smiles,
            component_smiles,
        )

        raw_outcomes, outcomes = _target_annotations(
            compiled,
            component_smiles,
            components,
            target_smiles,
            max_outcomes=max_outcomes,
        )
        (
            product_row,
            product_atoms,
            product_bonds,
            product_components,
        ) = _aggregate_product(
            label,
            target_smiles,
            component_smiles,
            components,
            evidence["evidence_record_id"],
            raw_outcomes,
            outcomes,
        )

        reversed_components = tuple(
            Chem.RenumberAtoms(
                component,
                list(reversed(range(component.GetNumAtoms()))),
            )
            for component in components
        )
        _, reversed_outcomes = _target_annotations(
            compiled,
            component_smiles,
            reversed_components,
            target_smiles,
            max_outcomes=max_outcomes,
        )
        if {outcome["semantic_signature"] for outcome in outcomes} != {
            outcome["semantic_signature"] for outcome in reversed_outcomes
        }:
            permutation_mismatches += 1

        raw_outcome_distribution[str(raw_outcomes)] += 1
        target_outcome_distribution[str(len(outcomes))] += 1
        source_mapping_distribution[str(product_row["source_mapping_multiplicity"])] += 1
        semantic_distribution[str(product_row["semantic_signature_multiplicity"])] += 1
        maximum_distance_distribution[str(product_row["maximum_distance_to_core"])] += 1
        origin_counts.update(record["origin_role"] for record in product_atoms)
        deleted_by_role.update(
            record["role"] for record in product_components if record["mapping_status"] == "deleted"
        )
        for outcome in outcomes:
            if not all(outcome["connected_by_role"].values()):
                raise UgiSemanticAnnotationError(f"{label} contains a disconnected origin region")
        connected_role_regions += len(ROLE_NAMES)
        product_rows.append(product_row)
        atom_rows.extend(product_atoms)
        bond_rows.extend(product_bonds)
        component_rows.extend(product_components)

    product_rows.sort(key=lambda row: row["product_id"])
    atom_rows.sort(key=lambda row: (row["product_id"], row["product_atom_index"]))
    bond_rows.sort(key=lambda row: (row["product_id"], row["bond_index"]))
    component_rows.sort(
        key=lambda row: (
            row["product_id"],
            ROLE_NAMES.index(row["role"]),
            row["component_atom_index"],
        )
    )
    core_atoms = sum(bool(row["is_ugi_core"]) for row in atom_rows)
    core_boundary_bonds = sum(bool(row["is_core_boundary"]) for row in bond_rows)
    introduced_atoms = origin_counts["assembly_introduced"]
    deleted_atoms = sum(deleted_by_role.values())
    summary = {
        "products": len(product_rows),
        "unique_labels": len({row["product_id"] for row in product_rows}),
        "source_exact_records": len(evidence_index),
        "product_atoms": len(atom_rows),
        "product_bonds": len(bond_rows),
        "component_atom_rows": len(component_rows),
        "core_atoms": core_atoms,
        "core_boundary_bonds": core_boundary_bonds,
        "introduced_product_atoms": introduced_atoms,
        "deleted_component_atoms": deleted_atoms,
        "connected_role_regions": connected_role_regions,
        "raw_outcome_distribution": dict(sorted(raw_outcome_distribution.items())),
        "target_matching_outcome_distribution": dict(sorted(target_outcome_distribution.items())),
        "source_mapping_multiplicity_distribution": dict(
            sorted(source_mapping_distribution.items())
        ),
        "semantic_signature_multiplicity_distribution": dict(sorted(semantic_distribution.items())),
        "maximum_distance_to_core_distribution": dict(
            sorted(maximum_distance_distribution.items())
        ),
        "origin_atom_counts": dict(sorted(origin_counts.items())),
        "deleted_component_atoms_by_role": dict(sorted(deleted_by_role.items())),
        "permutation_annotation_mismatches": permutation_mismatches,
    }
    _assert_expected(expected, summary)
    if permutation_mismatches:
        raise UgiSemanticAnnotationError("semantic annotations depend on component atom order")

    product_payload = _gzip_csv(product_rows, PRODUCT_FIELDS)
    atom_payload = _gzip_csv(atom_rows, ATOM_FIELDS)
    bond_payload = _gzip_csv(bond_rows, BOND_FIELDS)
    component_payload = _gzip_csv(component_rows, COMPONENT_MAPPING_FIELDS)
    artifacts = {
        "ugi_semantic_products.csv.gz": _artifact_metadata(product_payload),
        "ugi_semantic_atoms.csv.gz": _artifact_metadata(atom_payload),
        "ugi_semantic_bonds.csv.gz": _artifact_metadata(bond_payload),
        "ugi_semantic_component_mappings.csv.gz": _artifact_metadata(component_payload),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed_exact_source_ugi_semantic_annotations",
        "seed": config["seed"],
        "software": {"rdkit": rdBase.rdkitVersion},
        "inputs": {
            "config": {
                "path": _portable(config_path, repo_root),
                "sha256": _sha256_file(config_path),
                "bytes": config_path.stat().st_size,
            },
            **verified,
        },
        "scope": scope,
        "summary": summary,
        "role_anchor_map_numbers": {
            role: sorted(
                {
                    int(core_position.removeprefix("map_"))
                    for product in product_rows
                    for atom_index, core_position in json.loads(
                        product["core_atom_indices_json"]
                    ).items()
                    if int(atom_index) == json.loads(product["role_anchor_indices_json"])[role]
                    and core_position.startswith("map_")
                }
            )
            for role in ROLE_NAMES
        },
        "decision": {
            "whole_graph_semantic_supervision_authorized": True,
            "component_ids_authorized_in_model_state": False,
            "independent_fragment_generation_authorized": False,
            "distances_must_be_derived_from_graph": True,
            "ordinary_atom_maps_survive_serialization": False,
            "capture_transient_reaction_properties": True,
            "oracle_model_frozen": False,
        },
        "artifacts": artifacts,
    }
    result_payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "ugi_semantic_products.csv.gz", product_payload)
    _atomic_write(output_dir / "ugi_semantic_atoms.csv.gz", atom_payload)
    _atomic_write(output_dir / "ugi_semantic_bonds.csv.gz", bond_payload)
    _atomic_write(
        output_dir / "ugi_semantic_component_mappings.csv.gz",
        component_payload,
    )
    _atomic_write(output_dir / "ugi_semantic_annotations_result.json", result_payload)
    return result
