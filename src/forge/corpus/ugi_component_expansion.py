"""Build the provenance-aware Phase 1 Ugi component-expansion registry.

This module qualifies structural L1 support only.  It preserves route and
procurement evidence as independent axes and never upgrades an L1-compatible
component into an E2 synthesis claim.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.corpus.component_splits import (
    FOLDS,
    family_fold_map,
    similarity_families,
)
from forge.corpus.r0_splits import sha256_bytes, sha256_file
from forge.corpus.r1_prime_audit import compile_reactions, load_reaction_definitions
from forge.model.vocabulary import load_atom_vocabulary

CONFIG_SCHEMA_VERSION = "phase1_ugi_component_expansion_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_component_expansion_result.v1"
REGISTRY_SCHEMA_VERSION = "phase1_ugi_component_expansion_registry.v1"
ROLES = (
    "amine_head",
    "oxoester_aldehyde_body_tail",
    "isocyanide_tail",
)
REGISTRY_FIELDS = (
    "component_id",
    "role",
    "canonical_smiles",
    "heavy_atoms",
    "elements_json",
    "source_classes_json",
    "source_record_ids_json",
    "source_platforms_json",
    "is_current_catalog",
    "source_scope_qualified",
    "within_model_support",
    "raw_handle_matches",
    "symmetry_distinct_handle_sites",
    "passes_registry_handle_policy",
    "site_resolution_required",
    "reference_forward_outcomes",
    "reference_forward_products",
    "reference_forward_compatible",
    "route_evidence_states_json",
    "procurement_evidence_states_json",
    "route_closure_states_json",
    "e2_route_closed",
    "l1_structural_admission",
    "admission_reason",
    "family_id",
    "family_size",
    "family_fold",
    "training_disposition",
)


class ComponentExpansionError(ValueError):
    """Raised when an expansion input or scientific invariant is violated."""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ComponentExpansionError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ComponentExpansionError(f"{label} is invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComponentExpansionError(f"{label} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise ComponentExpansionError(f"could not read CSV input {path}: {exc}") from exc


def _resolve_inputs(
    config: Mapping[str, Any], repo: Path
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    records: dict[str, dict[str, Any]] = {}
    for label, specification in sorted(config["inputs"].items()):
        if not isinstance(specification, dict) or set(specification) != {"path", "sha256"}:
            raise ComponentExpansionError(f"input {label!r} must define path and sha256")
        path = Path(str(specification["path"]))
        if not path.is_absolute():
            path = repo / path
        observed = sha256_file(path)
        expected = str(specification["sha256"])
        if observed != expected:
            raise ComponentExpansionError(
                f"input {label!r} SHA-256 mismatch: expected {expected}, observed {observed}"
            )
        paths[label] = path
        records[label] = {
            "path": str(path.relative_to(repo)),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    return paths, records


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ComponentExpansionError(
            f"unsupported component-expansion config schema {config.get('schema_version')!r}"
        )
    if not isinstance(config.get("seed"), int):
        raise ComponentExpansionError("seed must be an integer")
    policy = config.get("policy")
    if not isinstance(policy, dict) or policy.get("reaction_id") != "ugi_3cr_agile":
        raise ComponentExpansionError("policy must select the frozen ugi_3cr_agile reaction")
    if policy.get("identity") != "canonical_constitutional_smiles":
        raise ComponentExpansionError("Phase 1 component identity must remain constitutional")
    source_policy = policy.get("source_policy")
    if not isinstance(source_policy, dict):
        raise ComponentExpansionError("source_policy must be an object")
    if source_policy.get("source_activity_labels_inherited") is not False:
        raise ComponentExpansionError("source biological labels must never be inherited")
    if source_policy.get("route_closure_required_for_l1_structural_admission") is not False:
        raise ComponentExpansionError(
            "L1 structural support and E2 route closure must remain separate"
        )
    if source_policy.get("route_closure_required_for_e2_claim") is not True:
        raise ComponentExpansionError("E2 claims must require route closure")
    fingerprint = policy.get("family_fingerprint")
    if (
        not isinstance(fingerprint, dict)
        or fingerprint.get("kind") != "ECFP4"
        or fingerprint.get("radius") != 2
        or not isinstance(fingerprint.get("bits"), int)
        or fingerprint["bits"] <= 0
        or not 0.0 < float(fingerprint.get("similarity_threshold", 0.0)) < 1.0
        or fingerprint.get("include_chirality") is not False
    ):
        raise ComponentExpansionError("family_fingerprint must define constitutional ECFP4")
    fractions = policy.get("family_split_fractions")
    if not isinstance(fractions, dict) or tuple(fractions) != FOLDS:
        raise ComponentExpansionError(f"family_split_fractions must define {FOLDS} in order")
    if abs(sum(float(fractions[fold]) for fold in FOLDS) - 1.0) > 1e-12:
        raise ComponentExpansionError("family split fractions must sum to one")


def _canonical(smiles: str, label: str) -> tuple[str, Chem.Mol]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ComponentExpansionError(f"{label} is not one valid connected molecule: {smiles!r}")
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    with rdBase.BlockLogs():
        normalized = Chem.MolFromSmiles(canonical)
    if normalized is None:
        raise ComponentExpansionError(f"{label} failed constitutional normalization")
    return canonical, normalized


def _empty_candidate(role: str, smiles: str) -> dict[str, Any]:
    return {
        "role": role,
        "canonical_smiles": smiles,
        "source_classes": set(),
        "source_record_ids": set(),
        "source_platforms": set(),
        "route_evidence_states": set(),
        "procurement_evidence_states": set(),
        "route_closure_states": set(),
        "scope_votes": [],
        "site_resolution_votes": [],
        "e2_route_closed_votes": [],
    }


def _add_candidate(
    candidates: dict[tuple[str, str], dict[str, Any]],
    *,
    role: str,
    smiles: str,
    source_class: str,
    source_record_id: str = "",
    source_platforms: Sequence[str] = (),
    source_scope_qualified: bool,
    site_resolution_required: bool = False,
    route_evidence: Sequence[str] = (),
    procurement_evidence: Sequence[str] = (),
    route_closure: Sequence[str] = (),
    e2_route_closed: bool = False,
) -> None:
    canonical, _ = _canonical(smiles, f"{source_class}/{source_record_id or role}")
    entry = candidates.setdefault((role, canonical), _empty_candidate(role, canonical))
    entry["source_classes"].add(source_class)
    if source_record_id:
        entry["source_record_ids"].add(source_record_id)
    entry["source_platforms"].update(value for value in source_platforms if value)
    entry["scope_votes"].append(bool(source_scope_qualified))
    entry["site_resolution_votes"].append(bool(site_resolution_required))
    entry["route_evidence_states"].update(value for value in route_evidence if value)
    entry["procurement_evidence_states"].update(value for value in procurement_evidence if value)
    entry["route_closure_states"].update(value for value in route_closure if value)
    entry["e2_route_closed_votes"].append(bool(e2_route_closed))


def _json_list(value: Sequence[Any] | set[Any]) -> str:
    return json.dumps(sorted(value), separators=(",", ":"))


def _symmetry_distinct_handle_sites(molecule: Chem.Mol, query: Chem.Mol) -> int:
    matches = molecule.GetSubstructMatches(query, uniquify=True)
    if query.GetNumAtoms() != 1:
        return len(matches)
    ranks = Chem.CanonicalRankAtoms(molecule, breakTies=False)
    return len({ranks[match[0]] for match in matches})


def reaction_handle_qualification(
    molecule: Chem.Mol,
    *,
    query: Chem.Mol,
    forbidden: Sequence[Chem.Mol],
    allowed_site_multiplicity: Sequence[int],
) -> dict[str, Any]:
    """Apply the frozen registry handle policy to one generated component."""

    raw_matches = len(molecule.GetSubstructMatches(query, uniquify=True))
    distinct_sites = _symmetry_distinct_handle_sites(molecule, query)
    forbidden_match = any(molecule.HasSubstructMatch(pattern) for pattern in forbidden)
    return {
        "raw_handle_matches": raw_matches,
        "symmetry_distinct_handle_sites": distinct_sites,
        "forbidden_substructure_match": forbidden_match,
        "passes_registry_handle_policy": (
            distinct_sites in {int(value) for value in allowed_site_multiplicity}
            and not forbidden_match
        ),
    }


def _forward_products(
    reaction: Any,
    reactants: Sequence[Chem.Mol],
    maximum_outcomes: int,
) -> tuple[int, set[str]]:
    with rdBase.BlockLogs():
        outcomes = reaction.forward.RunReactants(tuple(reactants), maxProducts=maximum_outcomes)
    if len(outcomes) >= maximum_outcomes:
        return len(outcomes), set()
    products: set[str] = set()
    for outcome in outcomes:
        if len(outcome) != 1:
            continue
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(outcome[0])
            if len(Chem.GetMolFrags(outcome[0])) != 1:
                continue
            products.add(Chem.MolToSmiles(outcome[0], canonical=True, isomericSmiles=False))
        except Exception:
            continue
    return len(outcomes), products


def _similarity_families(
    smiles: Sequence[str], fingerprint: Mapping[str, Any], role: str
) -> dict[str, str]:
    """Return connected fingerprint families, guaranteeing no cross-family pair at threshold."""
    return similarity_families(
        smiles,
        fingerprint,
        role,
        error=ComponentExpansionError,
    )


def _family_fold_map(
    family_members: Mapping[str, Sequence[str]],
    fractions: Mapping[str, Any],
    seed: int,
    role: str,
) -> dict[str, str]:
    """Greedily balance complete families while keeping every family intact."""
    return family_fold_map(
        family_members,
        fractions,
        seed,
        role,
        error=ComponentExpansionError,
    )


def _csv_gzip_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row.get(field, "") for field in REGISTRY_FIELDS} for row in rows)
    return gzip.compress(stream.getvalue().encode(), compresslevel=9, mtime=0)


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


def build_ugi_component_expansion(
    config_path: Path, repo: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the component registry and family-disjoint split assignment."""

    config = _load_json(config_path, "component-expansion config")
    _validate_config(config)
    paths, input_records = _resolve_inputs(config, repo)
    policy = config["policy"]
    candidates: dict[tuple[str, str], dict[str, Any]] = {}

    current_rows = _read_csv(paths["current_assignments"])
    current_components: dict[str, set[str]] = {role: set() for role in ROLES}
    for row in current_rows:
        for role in ROLES:
            canonical, _ = _canonical(row[f"{role}_smiles"], f"current {role}")
            current_components[role].add(canonical)
    for role in ROLES:
        for smiles in sorted(current_components[role]):
            _add_candidate(
                candidates,
                role=role,
                smiles=smiles,
                source_class="current_phase1_catalog",
                source_record_id=f"current:{role}:{smiles}",
                source_scope_qualified=True,
            )

    head_aldehyde = _load_json(paths["aldehyde_head_capability"], "head/aldehyde audit")
    for record in head_aldehyde["head_candidate_audit"]:
        _add_candidate(
            candidates,
            role="amine_head",
            smiles=str(record["canonical_smiles"]),
            source_class="bounded_head_capability",
            source_record_id=str(record["block_id"]),
            source_scope_qualified=bool(record["passes_qualified_registry_handle_policy"]),
            site_resolution_required=int(record["qualified_site_multiplicity"]) != 1,
            route_evidence=tuple(str(x) for x in record.get("transformation_evidence", ())),
            procurement_evidence=(str(record.get("current_procurement_status", "")),),
            route_closure=(str(record.get("route_closure", "")),),
        )
    for record in head_aldehyde["aldehyde_candidate_audit"]:
        _add_candidate(
            candidates,
            role="oxoester_aldehyde_body_tail",
            smiles=str(record["canonical_smiles"]),
            source_class="bounded_aldehyde_capability",
            source_record_id=str(record["block_id"]),
            source_scope_qualified=bool(record["passes_qualified_registry_handle_policy"]),
            route_evidence=tuple(str(x) for x in record.get("transformation_evidence", ())),
            procurement_evidence=(str(record.get("operational_availability", "")),),
            route_closure=(str(record.get("route_closure", "")),),
        )

    precursor = _load_json(paths["precursor_capability"], "precursor audit")
    for record in precursor["isocyanide_candidates"]:
        _add_candidate(
            candidates,
            role="isocyanide_tail",
            smiles=str(record["canonical_smiles"]),
            source_class="bounded_isocyanide_capability",
            source_record_id=str(record["block_id"]),
            source_scope_qualified=True,
            route_evidence=tuple(str(x) for x in record.get("transformation_evidence", ())),
            procurement_evidence=(str(record.get("operational_availability", "")),),
            route_closure=(str(record.get("route_closure", "")),),
        )
    for role, key in (("amine_head", "heads"), ("isocyanide_tail", "isocyanides")):
        for record in precursor["miao_cross_assembly_audit"][key]:
            _add_candidate(
                candidates,
                role=role,
                smiles=str(record["canonical_smiles"]),
                source_class="cross_assembly_observed_component",
                source_record_id=str(record["component_id"]),
                source_platforms=("MIAO_2019",),
                source_scope_qualified=True,
                route_evidence=tuple(str(x) for x in record.get("transformation_evidence", ())),
                procurement_evidence=(str(record.get("operational_availability", "")),),
                route_closure=(str(record.get("route_closure", "")),),
            )

    for record in _read_csv(paths["lnpdb_head_transfer"]):
        if record.get("parse_status") != "parsed":
            continue
        source_ids = json.loads(record.get("source_ids_json") or "[]")
        disposition = record.get("disposition", "")
        _add_candidate(
            candidates,
            role="amine_head",
            smiles=record["canonical_smiles"],
            source_class="lnpdb_head_census",
            source_record_id=record["component_id"],
            source_platforms=tuple(str(x) for x in source_ids),
            source_scope_qualified=disposition
            in {
                "exact_agile_head",
                "lnpdb_ugi_head_transfer_candidate",
            },
            site_resolution_required=record.get("site_resolution_required") == "True",
            route_evidence=(record.get("route_evidence_status", ""),),
            procurement_evidence=(record.get("procurement_evidence_status", ""),),
            route_closure=(record.get("execution_closure_status", ""),),
        )

    for record in _read_csv(paths["hydrophobic_motif_transfer"]):
        if record.get("disposition") != "propose_ugi_aldehyde":
            continue
        _add_candidate(
            candidates,
            role="oxoester_aldehyde_body_tail",
            smiles=record["proposed_ugi_component_smiles"],
            source_class="cross_platform_hydrophobic_transfer",
            source_record_id=record["record_id"],
            source_platforms=(record.get("source_platform", ""),),
            source_scope_qualified=(
                record.get("frozen_ugi_handle_matches") == "1"
                and int(record.get("frozen_ugi_forward_products", "0")) >= 1
            ),
            route_evidence=(record.get("proposed_route_evidence_grade", ""),),
            procurement_evidence=(record.get("terminal_status", ""),),
            route_closure=(record.get("route_closure", ""),),
            e2_route_closed=record.get("computationally_route_complete") == "true",
        )

    registry_payload = _load_json(paths["qualified_reactions"], "reaction registry")
    definitions = load_reaction_definitions(
        [paths["qualified_reactions"]], expected_count=len(registry_payload["reactions"])
    )
    compiled = {item.definition.reaction_id: item for item in compile_reactions(definitions)}
    reaction = compiled[policy["reaction_id"]]
    role_order = tuple(role.name for role in reaction.definition.reactant_roles)
    if role_order != ROLES:
        raise ComponentExpansionError(f"unexpected Ugi role order: {role_order}")
    reference_smiles = tuple(
        str(value) for value in reaction.definition.known_positive_examples[0]["reactants"]
    )
    references = [_canonical(value, "reference reactant")[1] for value in reference_smiles]
    allowed_elements = set(policy["allowed_elements"])
    maximum_heavy_atoms = int(policy["maximum_heavy_atoms"])
    maximum_outcomes = int(policy["maximum_forward_outcomes"])
    supported_atom_states = {
        (state.symbol, state.formal_charge, state.aromatic, state.explicit_hydrogens)
        for state in load_atom_vocabulary(paths["atom_vocabulary"])
    }

    def precursor_within_declared_support(molecule: Chem.Mol) -> bool:
        return (
            molecule.GetNumHeavyAtoms() <= maximum_heavy_atoms
            and {atom.GetSymbol() for atom in molecule.GetAtoms()}.issubset(allowed_elements)
            and all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
        )

    def product_within_declared_support(molecule: Chem.Mol) -> bool:
        return precursor_within_declared_support(molecule) and all(
            (
                atom.GetSymbol(),
                atom.GetFormalCharge(),
                atom.GetIsAromatic(),
                atom.GetNumExplicitHs(),
            )
            in supported_atom_states
            for atom in molecule.GetAtoms()
        )

    rows: list[dict[str, Any]] = []
    for (role, smiles), entry in sorted(candidates.items()):
        molecule = _canonical(smiles, "candidate component")[1]
        elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
        heavy_atoms = molecule.GetNumHeavyAtoms()
        within_support = precursor_within_declared_support(molecule)
        role_index = role_order.index(role)
        query = reaction.handles[role_index]
        handle_qualification = reaction_handle_qualification(
            molecule,
            query=query,
            forbidden=reaction.forbidden[role_index],
            allowed_site_multiplicity=reaction.definition.reactant_roles[
                role_index
            ].allowed_site_multiplicity,
        )
        raw_matches = int(handle_qualification["raw_handle_matches"])
        distinct_sites = int(handle_qualification["symmetry_distinct_handle_sites"])
        passes_handle = bool(handle_qualification["passes_registry_handle_policy"])
        current = smiles in current_components[role]
        source_scope = any(entry["scope_votes"])
        source_requires_site = any(entry["site_resolution_votes"])
        unique_site_required = bool(policy["novel_amine_requires_one_symmetry_distinct_site"])
        # Current Phase 1 components already carry an exact product-level site
        # mapping.  A generic census flag must not erase that stronger evidence.
        # Novel heads, by contrast, remain deferred until one reacting site is
        # unambiguous or explicitly selected.
        site_resolution = not current and (
            source_requires_site
            or (role == "amine_head" and unique_site_required and distinct_sites != 1)
        )
        reactants = [Chem.Mol(value) for value in references]
        reactants[role_index] = Chem.Mol(molecule)
        raw_outcomes, products = _forward_products(reaction, reactants, maximum_outcomes)
        products_within_support = all(
            product_within_declared_support(_canonical(product, "forward product")[1])
            for product in products
        )
        forward_compatible = bool(products) and products_within_support
        if not source_scope:
            admitted = False
            reason = "outside_declared_candidate_sources"
        elif not within_support:
            admitted = False
            reason = "outside_declared_model_support"
        elif not passes_handle:
            admitted = False
            reason = "fails_registry_handle_policy"
        elif site_resolution:
            admitted = False
            reason = "reactive_site_resolution_required"
        elif not forward_compatible:
            admitted = False
            reason = "no_valid_reference_context_forward_product"
        else:
            admitted = True
            reason = "current_catalog" if current else "l1_structural_support_only"
        component_id = (
            "ugi-component-" + hashlib.sha256(f"{role}\0{smiles}".encode()).hexdigest()[:20]
        )
        rows.append(
            {
                "component_id": component_id,
                "role": role,
                "canonical_smiles": smiles,
                "heavy_atoms": heavy_atoms,
                "elements_json": _json_list(elements),
                "source_classes_json": _json_list(entry["source_classes"]),
                "source_record_ids_json": _json_list(entry["source_record_ids"]),
                "source_platforms_json": _json_list(entry["source_platforms"]),
                "is_current_catalog": str(current).lower(),
                "source_scope_qualified": str(source_scope).lower(),
                "within_model_support": str(within_support).lower(),
                "raw_handle_matches": raw_matches,
                "symmetry_distinct_handle_sites": distinct_sites,
                "passes_registry_handle_policy": str(passes_handle).lower(),
                "site_resolution_required": str(site_resolution).lower(),
                "reference_forward_outcomes": raw_outcomes,
                "reference_forward_products": len(products),
                "reference_forward_compatible": str(forward_compatible).lower(),
                "route_evidence_states_json": _json_list(entry["route_evidence_states"]),
                "procurement_evidence_states_json": _json_list(
                    entry["procurement_evidence_states"]
                ),
                "route_closure_states_json": _json_list(entry["route_closure_states"]),
                "e2_route_closed": str(any(entry["e2_route_closed_votes"])).lower(),
                "l1_structural_admission": str(admitted).lower(),
                "admission_reason": reason,
                "family_id": "",
                "family_size": "",
                "family_fold": "",
                "training_disposition": "deferred",
            }
        )

    fingerprint = policy["family_fingerprint"]
    fractions = policy["family_split_fractions"]
    for role in ROLES:
        admitted_rows = [
            row for row in rows if row["role"] == role and row["l1_structural_admission"] == "true"
        ]
        family_by_smiles = _similarity_families(
            [str(row["canonical_smiles"]) for row in admitted_rows], fingerprint, role
        )
        members: dict[str, list[str]] = defaultdict(list)
        for smiles, family in family_by_smiles.items():
            members[family].append(smiles)
        family_folds = _family_fold_map(members, fractions, int(config["seed"]), role)
        for row in admitted_rows:
            family = family_by_smiles[str(row["canonical_smiles"])]
            fold = family_folds[family]
            row["family_id"] = family
            row["family_size"] = len(members[family])
            row["family_fold"] = fold
            if fold == "train":
                row["training_disposition"] = (
                    "train_current_catalog"
                    if row["is_current_catalog"] == "true"
                    else "train_expanded_l1_structural_support"
                )
            else:
                row["training_disposition"] = f"reserve_{fold}_family"

    rows.sort(key=lambda row: (str(row["role"]), str(row["canonical_smiles"])))
    admitted = [row for row in rows if row["l1_structural_admission"] == "true"]
    current_admitted = [row for row in admitted if row["is_current_catalog"] == "true"]
    new_admitted = [row for row in admitted if row["is_current_catalog"] == "false"]
    family_counts = {
        role: len(
            {row["family_id"] for row in admitted if row["role"] == role and row["family_id"]}
        )
        for role in ROLES
    }
    fold_counts = {
        role: dict(Counter(row["family_fold"] for row in admitted if row["role"] == role))
        for role in ROLES
    }
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "complete_l1_structural_expansion_census",
        "task": config["task"],
        "config": {
            "path": str(config_path.relative_to(repo)),
            "sha256": sha256_file(config_path),
        },
        "inputs": input_records,
        "policy": policy,
        "summary": {
            "candidate_rows": len(rows),
            "current_catalog_components": {role: len(current_components[role]) for role in ROLES},
            "l1_admitted_components": dict(Counter(row["role"] for row in admitted)),
            "new_l1_structural_components": dict(Counter(row["role"] for row in new_admitted)),
            "current_components_retained": dict(Counter(row["role"] for row in current_admitted)),
            "admission_reasons": dict(Counter(row["admission_reason"] for row in rows)),
            "family_counts": family_counts,
            "component_fold_counts": fold_counts,
            "e2_route_closed_components": sum(row["e2_route_closed"] == "true" for row in admitted),
            "raw_cartesian_upper_bound": (
                sum(row["role"] == ROLES[0] for row in admitted)
                * sum(row["role"] == ROLES[1] for row in admitted)
                * sum(row["role"] == ROLES[2] for row in admitted)
            ),
        },
        "claims_boundary": {
            "l1_admission_is_synthesis_success": False,
            "l1_admission_is_route_closure": False,
            "reference_context_forward_compatibility_is_substrate_scope_proof": False,
            "source_activity_labels_inherited": False,
            "e2_requires_independent_complete_route": True,
            "family_assignment_precedes_product_enumeration": True,
        },
        "randomness": {
            "used": False,
            "seed": int(config["seed"]),
            "family_assignment": "deterministic ECFP4 similarity components and greedy balance",
        },
    }
    return result, rows


def write_ugi_component_expansion(
    config_path: Path,
    repo: Path,
    result: dict[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the registry and result atomically with deterministic hashes."""

    config = _load_json(config_path, "component-expansion config")
    registry_path = repo / config["outputs"]["registry"]
    result_path = repo / config["outputs"]["result"]
    registry_payload = _csv_gzip_bytes(rows)
    result = dict(result)
    result["artifacts"] = {
        "registry": {
            "path": str(registry_path.relative_to(repo)),
            "bytes": len(registry_payload),
            "sha256": sha256_bytes(registry_payload),
            "rows": len(rows),
            "columns": list(REGISTRY_FIELDS),
        }
    }
    result_payload = (
        json.dumps(result, indent=2, sort_keys=True, separators=(",", ": ")) + "\n"
    ).encode()
    _atomic_write(registry_path, registry_payload)
    _atomic_write(result_path, result_payload)
    return result
