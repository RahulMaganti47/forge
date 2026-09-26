"""Freeze the Phase 1 product and exact Ugi-L1 training data contract."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import atomic_write as _atomic_write
from forge.core.io import csv_gz_bytes as _csv_bytes
from forge.core.io import read_json_object
from forge.corpus.r0_splits import sha256_bytes, sha256_file

CONFIG_SCHEMA_VERSION = "phase1_product_l1_data_config.v3"
MANIFEST_SCHEMA_VERSION = "phase1_product_l1_split_manifest.v3"
RESULT_SCHEMA_VERSION = "phase1_product_l1_data_result.v3"
FOLDS = ("train", "calibration", "heldout")
ROLES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
ASSIGNMENT_FIELDS = (
    "product_id",
    "canonical_product_smiles",
    *(field for role in ROLES for field in (f"{role}_smiles", f"held_{role}_fold")),
    "primary_product_fold",
    "is_source_adjudicated_measured_product",
)
PROVENANCE_FIELDS = (
    "model_product_id",
    "canonical_product_smiles",
    "source_kind",
    "source_product_id",
    "source_product_smiles",
    "source_isomeric_canonical_smiles",
    "source_component_smiles_json",
    "constitutional_component_smiles_json",
    "is_selected_identity_source",
    "constitutional_group_size",
)


class Phase1DataError(ValueError):
    """Raised when a Phase 1 training input violates the frozen contract."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    return read_json_object(path, error=Phase1DataError, label=label)


def _resolve_and_verify(
    repo: Path,
    record: Mapping[str, Any],
    label: str,
) -> tuple[Path, dict[str, Any]]:
    if set(record) != {"path", "sha256"}:
        raise Phase1DataError(f"{label} must define exactly path and sha256")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = repo / path
    if not path.exists():
        raise Phase1DataError(f"{label} input not found: {path}")
    observed = sha256_file(path)
    expected = str(record["sha256"])
    if observed != expected:
        raise Phase1DataError(f"{label} SHA-256 mismatch: expected {expected}, observed {observed}")
    return path, {
        "path": str(path.relative_to(repo)),
        "bytes": path.stat().st_size,
        "sha256": observed,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise Phase1DataError(f"could not read CSV input {path}: {exc}") from exc


def _canonical_smiles_pair(smiles: str, label: str) -> tuple[str, str]:
    """Return canonical constitutional and isomeric identities for one graph."""

    if not isinstance(smiles, str) or not smiles:
        raise Phase1DataError(f"{label} lacks SMILES")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise Phase1DataError(f"{label} contains invalid SMILES")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise Phase1DataError(f"{label} must contain one connected graph")
    constitutional = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=False,
    )
    isomeric = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    return constitutional, isomeric


def _canonical_constitutional_smiles(smiles: str, label: str) -> str:
    return _canonical_smiles_pair(smiles, label)[0]


def _append_constitutional_product(
    grouped: dict[str, dict[str, Any]],
    product: dict[str, Any],
    *,
    source_label: str,
) -> None:
    """Collapse source rows only when their constitutional L1 mapping agrees."""

    identity = product["canonical_product_smiles"]
    existing = grouped.get(identity)
    if existing is None:
        grouped[identity] = product
        return
    for role in ROLES:
        if existing[f"{role}_smiles"] != product[f"{role}_smiles"]:
            raise Phase1DataError(
                f"{source_label} constitutional component mappings disagree for {identity}"
            )
    existing["_source_records"].extend(product["_source_records"])


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Phase1DataError(f"unsupported Phase 1 config schema {config.get('schema_version')!r}")
    inputs = config.get("inputs")
    expected_input_names = {
        "r0_constitutional",
        "r0_assignments",
        "r1_reaction_enumerated",
        "ugi_virtual_products",
        "ugi_measured_semantics",
        "qualified_reactions",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_names:
        raise Phase1DataError(f"inputs must define {sorted(expected_input_names)}")
    broad = config.get("broad_stream")
    if not isinstance(broad, dict):
        raise Phase1DataError("broad_stream must be an object")
    if broad.get("primary_split_scheme") != "source_study":
        raise Phase1DataError("Phase 1 primary broad split must remain source_study")
    if broad.get("r1_sampling_weight") != "realism_weight":
        raise Phase1DataError("R1 must be sampled through realism_weight")
    if broad.get("raw_reaction_family_sampling_allowed") is not False:
        raise Phase1DataError("raw reaction-family sampling must remain prohibited")
    r0_fraction = float(broad.get("r0_fraction", -1))
    r1_fraction = float(broad.get("r1_fraction", -1))
    if abs(r0_fraction + r1_fraction - 1.0) > 1e-12:
        raise Phase1DataError("broad R0 and R1 fractions must sum to one")
    if not (r0_fraction > r1_fraction > 0.0):
        raise Phase1DataError("R1 must remain a positive, lower-weight broad auxiliary")

    ugi = config.get("ugi_l1")
    if not isinstance(ugi, dict) or tuple(ugi.get("roles", ())) != ROLES:
        raise Phase1DataError(f"Ugi roles must remain ordered as {ROLES}")
    fractions = ugi.get("component_split_fractions")
    if not isinstance(fractions, dict) or tuple(fractions) != FOLDS:
        raise Phase1DataError(f"component_split_fractions must define {FOLDS} in order")
    if abs(sum(float(fractions[fold]) for fold in FOLDS) - 1.0) > 1e-12:
        raise Phase1DataError("component split fractions must sum to one")
    if ugi.get("duplicate_product_upweighting_allowed") is not False:
        raise Phase1DataError("duplicate Ugi product upweighting must remain prohibited")

    joint = config.get("joint_training")
    if not isinstance(joint, dict):
        raise Phase1DataError("joint_training must be an object")
    ratios = joint.get("broad_to_ugi_replay_ratios")
    expected_ratios = [[0.75, 0.25], [0.5, 0.5], [0.25, 0.75]]
    if ratios != expected_ratios:
        raise Phase1DataError(f"broad-to-Ugi replay ratios must remain {expected_ratios}")
    if joint.get("biological_guidance_enabled") is not False:
        raise Phase1DataError("Phase 1 biological guidance must remain disabled")
    if joint.get("synthesis_value_guidance_enabled") is not False:
        raise Phase1DataError("Phase 1 synthesis-value guidance must remain disabled")


def _stable_rank(namespace: str, value: str, seed: int) -> str:
    return hashlib.sha256(f"{namespace}\0{seed}\0{value}".encode()).hexdigest()


def _component_fold_map(
    components: Sequence[str],
    fractions: Mapping[str, Any],
    role: str,
    seed: int,
) -> dict[str, str]:
    ordered = sorted(components, key=lambda value: _stable_rank(role, value, seed))
    count = len(ordered)
    calibration_count = max(1, round(count * float(fractions["calibration"])))
    heldout_count = max(1, round(count * float(fractions["heldout"])))
    if calibration_count + heldout_count >= count:
        raise Phase1DataError(f"not enough {role} components for three nonempty folds")
    train_count = count - calibration_count - heldout_count
    fold_map = {}
    for index, value in enumerate(ordered):
        if index < train_count:
            fold = "train"
        elif index < train_count + calibration_count:
            fold = "calibration"
        else:
            fold = "heldout"
        fold_map[value] = fold
    return fold_map


def _parse_virtual_products(
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ugi = config["ugi_l1"]
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("decomposition_status") != ugi["required_decomposition_status"]:
            raise Phase1DataError(
                "Ugi virtual row does not have one exact qualified decomposition: "
                f"source_row_index={row.get('source_row_index')}"
            )
        if int(row.get("candidate_count", "0")) != int(ugi["required_candidate_count"]):
            raise Phase1DataError(
                f"Ugi virtual row candidate_count is not one: {row.get('source_row_index')}"
            )
        try:
            candidates = json.loads(row["candidate_routes_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise Phase1DataError(
                f"invalid candidate_routes_json at source row {row.get('source_row_index')}"
            ) from exc
        if not isinstance(candidates, list) or len(candidates) != 1:
            raise Phase1DataError("each Ugi virtual product must have exactly one candidate route")
        components = candidates[0].get("components")
        if not isinstance(components, dict) or set(components) != set(ROLES):
            raise Phase1DataError(f"Ugi component roles must be exactly {ROLES}")
        source_product = row["canonical_product_smiles"]
        product, isomeric_product = _canonical_smiles_pair(
            source_product,
            f"virtual product {row.get('source_row_index')}",
        )
        source_components = {role: str(components[role]) for role in ROLES}
        constitutional_components = {
            role: _canonical_constitutional_smiles(
                source_components[role],
                f"virtual product {row.get('source_row_index')} {role}",
            )
            for role in ROLES
        }
        product_id = f"VUGI-{int(row['source_row_index']):05d}"
        _append_constitutional_product(
            grouped,
            {
                "product_id": product_id,
                "canonical_product_smiles": product,
                **{f"{role}_smiles": constitutional_components[role] for role in ROLES},
                "_source_records": [
                    {
                        "source_kind": "virtual",
                        "source_product_id": product_id,
                        "source_product_smiles": source_product,
                        "source_isomeric_canonical_smiles": isomeric_product,
                        "source_component_smiles_json": json.dumps(
                            source_components,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "constitutional_component_smiles_json": json.dumps(
                            constitutional_components,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
            },
            source_label="virtual Ugi",
        )
    return [grouped[identity] for identity in sorted(grouped)]


def _parse_measured_products(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        try:
            components = json.loads(row["component_smiles_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise Phase1DataError(
                f"invalid measured component_smiles_json for {row.get('product_id')}"
            ) from exc
        if not isinstance(components, dict) or set(components) != set(ROLES):
            raise Phase1DataError(f"measured Ugi component roles must be exactly {ROLES}")
        source_product = row["product_smiles"]
        product, isomeric_product = _canonical_smiles_pair(
            source_product,
            f"measured product {row.get('product_id')}",
        )
        source_components = {role: str(components[role]) for role in ROLES}
        constitutional_components = {
            role: _canonical_constitutional_smiles(
                source_components[role],
                f"measured product {row.get('product_id')} {role}",
            )
            for role in ROLES
        }
        product_id = f"MUGI-{row['product_id']}"
        _append_constitutional_product(
            grouped,
            {
                "product_id": product_id,
                "canonical_product_smiles": product,
                **{f"{role}_smiles": constitutional_components[role] for role in ROLES},
                "_source_records": [
                    {
                        "source_kind": "measured",
                        "source_product_id": product_id,
                        "source_product_smiles": source_product,
                        "source_isomeric_canonical_smiles": isomeric_product,
                        "source_component_smiles_json": json.dumps(
                            source_components,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "constitutional_component_smiles_json": json.dumps(
                            constitutional_components,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    }
                ],
            },
            source_label="measured Ugi",
        )
    return [grouped[identity] for identity in sorted(grouped)]


def _merge_ugi_products(
    virtual_products: Sequence[Mapping[str, Any]],
    measured_products: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Deduplicate exact products while giving measured source evidence priority."""

    merged = {row["canonical_product_smiles"]: dict(row) for row in virtual_products}
    overlap = 0
    for row in measured_products:
        product = row["canonical_product_smiles"]
        existing = merged.get(product)
        if existing is not None:
            overlap += 1
            for role in ROLES:
                if existing[f"{role}_smiles"] != row[f"{role}_smiles"]:
                    raise Phase1DataError(
                        f"measured and virtual component mappings disagree for {product}"
                    )
        selected = dict(row)
        selected["_source_records"] = [
            *(existing or {}).get("_source_records", []),
            *row.get("_source_records", []),
        ]
        merged[product] = selected
    return [merged[product] for product in sorted(merged)], overlap


def _build_provenance_rows(
    products: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    rows = []
    for product in sorted(products, key=lambda row: row["canonical_product_smiles"]):
        sources = list(product.get("_source_records", ()))
        if not sources:
            raise Phase1DataError(
                f"Ugi product lacks source provenance: {product['canonical_product_smiles']}"
            )
        for source in sorted(
            sources,
            key=lambda row: (row["source_kind"], row["source_product_id"]),
        ):
            rows.append(
                {
                    "model_product_id": str(product["product_id"]),
                    "canonical_product_smiles": str(product["canonical_product_smiles"]),
                    "source_kind": str(source["source_kind"]),
                    "source_product_id": str(source["source_product_id"]),
                    "source_product_smiles": str(source["source_product_smiles"]),
                    "source_isomeric_canonical_smiles": str(
                        source["source_isomeric_canonical_smiles"]
                    ),
                    "source_component_smiles_json": str(source["source_component_smiles_json"]),
                    "constitutional_component_smiles_json": str(
                        source["constitutional_component_smiles_json"]
                    ),
                    "is_selected_identity_source": str(
                        source["source_product_id"] == product["product_id"]
                    ).lower(),
                    "constitutional_group_size": str(len(sources)),
                }
            )
    return rows


def _build_assignments(
    products: Sequence[Mapping[str, str]],
    measured_products: set[str],
    config: Mapping[str, Any],
) -> list[dict[str, str]]:
    fractions = config["ugi_l1"]["component_split_fractions"]
    components_by_role: dict[str, set[str]] = defaultdict(set)
    for product in products:
        for role in ROLES:
            components_by_role[role].add(product[f"{role}_smiles"])
    fold_maps = {
        role: _component_fold_map(
            sorted(components_by_role[role]),
            fractions,
            role,
            int(config["seed"]),
        )
        for role in ROLES
    }
    assignments = []
    for product in products:
        role_folds = {role: fold_maps[role][product[f"{role}_smiles"]] for role in ROLES}
        # This strict, mutually exclusive product partition is the default
        # training/evaluation boundary.  A product is held out if any of its
        # components is held out; calibration similarly requires no held-out
        # component.  Protocol-specific single-role evaluations retain the
        # three role-fold columns above.
        if "heldout" in role_folds.values():
            primary_product_fold = "heldout"
        elif "calibration" in role_folds.values():
            primary_product_fold = "calibration"
        else:
            primary_product_fold = "train"
        assignments.append(
            {
                "product_id": product["product_id"],
                "canonical_product_smiles": product["canonical_product_smiles"],
                **{
                    field: value
                    for role in ROLES
                    for field, value in (
                        (f"{role}_smiles", product[f"{role}_smiles"]),
                        (
                            f"held_{role}_fold",
                            role_folds[role],
                        ),
                    )
                },
                "primary_product_fold": primary_product_fold,
                "is_source_adjudicated_measured_product": str(
                    product["canonical_product_smiles"] in measured_products
                ).lower(),
            }
        )
    return assignments


def _relative_output(repo: Path, configured: str | Path) -> Path:
    path = Path(configured)
    return path if path.is_absolute() else repo / path


def _display_path(path: Path, repo: Path) -> str:
    try:
        return str(path.relative_to(repo))
    except ValueError:
        return str(path)


def freeze_phase1_data_contract(
    config_path: Path,
    repo: Path,
    *,
    output_paths: Mapping[str, str | Path] | None = None,
    output_display_root: Path | None = None,
) -> dict[str, Any]:
    """Validate inputs, freeze component holdouts, and write the Phase 1 manifest.

    ``output_paths`` lets the experiment runner isolate writes in its private staging directory.
    Omitting it preserves the original CLI behavior and configured artifact locations exactly.
    ``output_display_root`` controls only the stable paths recorded inside produced artifacts; it
    never changes where bytes are written.
    """

    config = _load_json(config_path, "Phase 1 data config")
    _validate_config(config)
    resolved: dict[str, Path] = {}
    input_records: dict[str, dict[str, Any]] = {}
    for label, record in config["inputs"].items():
        resolved[label], input_records[label] = _resolve_and_verify(repo, record, label)

    r0_rows = _read_csv(resolved["r0_constitutional"])
    r0_assignments = _read_csv(resolved["r0_assignments"])
    r1_rows = _read_csv(resolved["r1_reaction_enumerated"])
    virtual_rows = _read_csv(resolved["ugi_virtual_products"])
    measured_rows = _read_csv(resolved["ugi_measured_semantics"])
    expected = config["expected"]
    observed_counts = {
        "r0_rows": len(r0_rows),
        "r0_assignments": len(r0_assignments),
        "r1_rows": len(r1_rows),
        "ugi_virtual_products": len(virtual_rows),
        "ugi_measured_products": len(measured_rows),
    }
    for key in ("r0_rows", "r1_rows", "ugi_virtual_products", "ugi_measured_products"):
        if observed_counts[key] != int(expected[key]):
            raise Phase1DataError(
                f"{key} count mismatch: expected {expected[key]}, observed {observed_counts[key]}"
            )
    if observed_counts["r0_assignments"] != observed_counts["r0_rows"]:
        raise Phase1DataError("R0 rows and assignments must have equal cardinality")

    r0_ids = {row["r0_structure_id"] for row in r0_rows}
    assignment_ids = {row["r0_structure_id"] for row in r0_assignments}
    if r0_ids != assignment_ids or len(r0_ids) != len(r0_rows):
        raise Phase1DataError("R0 structure IDs are duplicated or do not match frozen assignments")
    source_study_counts = Counter(row["source_study_fold"] for row in r0_assignments)
    if dict(source_study_counts) != expected["r0_source_study_folds"]:
        raise Phase1DataError(
            "source-study fold counts changed: "
            f"expected {expected['r0_source_study_folds']}, observed {dict(source_study_counts)}"
        )

    family_rows = Counter()
    family_weight = Counter()
    total_r1_weight = 0.0
    for index, row in enumerate(r1_rows):
        try:
            weight = float(row["realism_weight"])
        except (KeyError, ValueError) as exc:
            raise Phase1DataError(f"invalid R1 realism_weight at row {index}") from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise Phase1DataError(f"nonfinite or negative R1 realism_weight at row {index}")
        family = row["reaction_family"]
        family_rows[family] += 1
        family_weight[family] += weight
        total_r1_weight += weight
    if total_r1_weight <= 0:
        raise Phase1DataError("R1 realism weights have no positive mass")

    virtual_products = _parse_virtual_products(virtual_rows, config)
    measured_product_rows = _parse_measured_products(measured_rows)
    if len(virtual_products) != int(expected["ugi_virtual_constitutions"]):
        raise Phase1DataError(
            "virtual Ugi constitutional count changed: "
            f"expected {expected['ugi_virtual_constitutions']}, "
            f"observed {len(virtual_products)}"
        )
    if len(measured_product_rows) != int(expected["ugi_measured_constitutions"]):
        raise Phase1DataError(
            "measured Ugi constitutional count changed: "
            f"expected {expected['ugi_measured_constitutions']}, "
            f"observed {len(measured_product_rows)}"
        )
    products, virtual_measured_overlap = _merge_ugi_products(
        virtual_products,
        measured_product_rows,
    )
    measured_products = {row["canonical_product_smiles"] for row in measured_product_rows}
    if len(products) != int(expected["ugi_union_products"]):
        raise Phase1DataError(
            "Ugi union count changed: "
            f"expected {expected['ugi_union_products']}, observed {len(products)}"
        )
    if virtual_measured_overlap != int(expected["ugi_virtual_measured_overlap"]):
        raise Phase1DataError(
            "virtual/measured Ugi overlap changed: "
            f"expected {expected['ugi_virtual_measured_overlap']}, "
            f"observed {virtual_measured_overlap}"
        )
    assignments = _build_assignments(products, measured_products, config)
    provenance_rows = _build_provenance_rows(products)
    expected_source_rows = len(virtual_rows) + len(measured_rows)
    if len(provenance_rows) != expected_source_rows:
        raise Phase1DataError(
            "Ugi provenance does not preserve every source row: "
            f"expected {expected_source_rows}, observed {len(provenance_rows)}"
        )

    component_counts = {role: len({row[f"{role}_smiles"] for row in assignments}) for role in ROLES}
    if component_counts != expected["ugi_component_counts"]:
        raise Phase1DataError(
            "Ugi component counts changed: "
            f"expected {expected['ugi_component_counts']}, observed {component_counts}"
        )
    scheme_counts = {
        f"held_{role}": dict(Counter(row[f"held_{role}_fold"] for row in assignments))
        for role in ROLES
    }
    component_fold_counts = {
        role: dict(
            Counter(
                {row[f"{role}_smiles"]: row[f"held_{role}_fold"] for row in assignments}.values()
            )
        )
        for role in ROLES
    }
    for role in ROLES:
        if set(component_fold_counts[role]) != set(FOLDS):
            raise Phase1DataError(f"{role} does not have nonempty train/calibration/heldout folds")
    primary_product_fold_counts = dict(Counter(row["primary_product_fold"] for row in assignments))
    if primary_product_fold_counts != expected["ugi_primary_product_folds"]:
        raise Phase1DataError(
            "Ugi primary product folds changed: "
            f"expected {expected['ugi_primary_product_folds']}, "
            f"observed {primary_product_fold_counts}"
        )

    assignment_payload = _csv_bytes(assignments, ASSIGNMENT_FIELDS)
    provenance_payload = _csv_bytes(provenance_rows, PROVENANCE_FIELDS)
    outputs = dict(config["outputs"] if output_paths is None else output_paths)
    expected_output_names = {"ugi_assignments", "ugi_provenance", "manifest", "result"}
    if set(outputs) != expected_output_names:
        raise Phase1DataError(f"output paths must define {sorted(expected_output_names)}")
    assignment_path = _relative_output(repo, outputs["ugi_assignments"])
    provenance_path = _relative_output(repo, outputs["ugi_provenance"])
    manifest_path = _relative_output(repo, outputs["manifest"])
    result_path = _relative_output(repo, outputs["result"])
    display_root = output_display_root or repo
    assignment_record = {
        "path": _display_path(assignment_path, display_root),
        "bytes": len(assignment_payload),
        "sha256": sha256_bytes(assignment_payload),
        "rows": len(assignments),
        "columns": list(ASSIGNMENT_FIELDS),
    }
    provenance_record = {
        "path": _display_path(provenance_path, display_root),
        "bytes": len(provenance_payload),
        "sha256": sha256_bytes(provenance_payload),
        "rows": len(provenance_rows),
        "columns": list(PROVENANCE_FIELDS),
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "seed": int(config["seed"]),
        "inputs": input_records,
        "policy": {
            "broad_primary_split": "source_study",
            "broad_stream": config["broad_stream"],
            "ugi_l1": config["ugi_l1"],
            "joint_training": config["joint_training"],
            "representation": config["representation"],
        },
        "counts": {
            **observed_counts,
            "ugi_virtual_constitutions": len(virtual_products),
            "ugi_measured_constitutions": len(measured_product_rows),
            "ugi_union_products": len(products),
            "ugi_virtual_measured_overlap": virtual_measured_overlap,
            "ugi_source_rows_collapsed_by_constitutional_identity": (
                len(provenance_rows) - len(products)
            ),
            "source_adjudicated_measured_products_in_union": len(measured_products),
            "ugi_components": component_counts,
            "ugi_component_folds": component_fold_counts,
            "ugi_product_folds": scheme_counts,
            "ugi_primary_product_folds": primary_product_fold_counts,
        },
        "r1_weighted_family_mass": {
            family: {
                "rows": family_rows[family],
                "weight_sum": family_weight[family],
                "normalized_weight_mass": family_weight[family] / total_r1_weight,
            }
            for family in sorted(family_rows)
        },
        "outputs": {
            "ugi_assignments": assignment_record,
            "ugi_provenance": provenance_record,
        },
        "randomness": {
            "used": False,
            "component_assignment": "SHA-256 rank by role and frozen seed",
        },
    }
    manifest_payload = (
        json.dumps(manifest, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode()
    manifest_record = {
        "path": _display_path(manifest_path, display_root),
        "bytes": len(manifest_payload),
        "sha256": sha256_bytes(manifest_payload),
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "complete",
        "task": config["task"],
        "config": {
            "path": _display_path(config_path, repo),
            "sha256": sha256_file(config_path),
        },
        "manifest": manifest_record,
        "summary": {
            "r0_rows": len(r0_rows),
            "r1_rows": len(r1_rows),
            "ugi_l1_products": len(assignments),
            "ugi_source_rows": len(provenance_rows),
            "ugi_source_rows_collapsed_by_constitutional_identity": (
                len(provenance_rows) - len(products)
            ),
            "ugi_virtual_measured_overlap": virtual_measured_overlap,
            "source_adjudicated_measured_ugi_products": len(measured_products),
            "ugi_components": component_counts,
            "broad_r0_fraction": float(config["broad_stream"]["r0_fraction"]),
            "broad_r1_fraction": float(config["broad_stream"]["r1_fraction"]),
            "replay_ratios": config["joint_training"]["broad_to_ugi_replay_ratios"],
            "biological_guidance_enabled": False,
            "synthesis_value_guidance_enabled": False,
        },
    }
    result_payload = (
        json.dumps(result, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode()
    _atomic_write(assignment_path, assignment_payload)
    _atomic_write(provenance_path, provenance_payload)
    _atomic_write(manifest_path, manifest_payload)
    _atomic_write(result_path, result_payload)
    return result
