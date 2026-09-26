"""Potency observations derived directly from the hash-pinned LNPDB catalogue.

The active potency pipeline has one row source: LNPDB.  It keeps each study and endpoint separate,
uses the LNPDB within-study normalized target exactly as published, excludes configured non-single-
compound observations, and canonicalizes only the complete lipid graph.  Reaction-program evidence
is deliberately absent from this schema and belongs to the reaction-program corpus.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.chemistry.smiles import canonical_connected_constitution
from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import atomic_write, csv_gz_bytes, read_csv_rows, read_json_object, stable_json
from forge.corpus.lnpdb import LNPDBComponent, LNPDBRow, load_lnpdb

CONFIG_SCHEMA = "forge.potency_study_corpus_config.v2"
LEDGER_SCHEMA = "forge.potency_study_observations.v2"
RESULT_SCHEMA = "forge.potency_study_corpus_result.v2"

LEDGER_FIELDS = (
    "record_id",
    "study_id",
    "publication_pmid",
    "source_record_id",
    "source_row_index",
    "source_lipid_name",
    "endpoint",
    "label_value",
    "label_semantics",
    "raw_source_product_smiles",
    "model_smiles",
)

_COMPONENT_SLOTS = {"head", "linker", "tail1", "tail2", "tail3", "tail4"}


class PotencyStudyDataError(ValueError):
    """The LNPDB potency view would violate its frozen modelling contract."""


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PotencyStudyDataError(f"{label} must be a non-negative integer")
    return value


def _count_map(value: Any, *, label: str) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise PotencyStudyDataError(f"{label} must be a non-empty object")
    output: dict[str, int] = {}
    for endpoint, count in value.items():
        if not isinstance(endpoint, str) or not endpoint:
            raise PotencyStudyDataError(f"{label} has an invalid endpoint")
        output[endpoint] = _integer(count, label=f"{label}.{endpoint}")
    return output


def _validate_config(config: Mapping[str, Any]) -> Mapping[str, Mapping[str, Any]]:
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise PotencyStudyDataError(
            f"unsupported study-corpus schema: {config.get('schema_version')!r}"
        )
    if config.get("graph_identity") != "canonical_connected_constitutional_smiles":
        raise PotencyStudyDataError("potency model identity must be connected and constitutional")
    policy = config.get("policy")
    required_policy = {
        "raw_cross_study_label_pooling": False,
        "lnpdb_is_authoritative_row_source": True,
        "mixture_measurements_excluded": True,
        "reaction_program_fields_in_potency_rows": False,
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != expected for key, expected in required_policy.items()
    ):
        raise PotencyStudyDataError("study-corpus scientific policy changed")
    studies = config.get("studies")
    if not isinstance(studies, dict) or set(studies) != {"YX_2024", "JC_2023", "LM_2019"}:
        raise PotencyStudyDataError("study corpus must declare YX_2024, JC_2023, and LM_2019")
    for study_id, raw_spec in studies.items():
        if not isinstance(raw_spec, dict) or raw_spec.get("study_id") != study_id:
            raise PotencyStudyDataError(f"{study_id} study specification is malformed")
        if raw_spec.get("label_semantics") != "lnpdb_within_study_endpoint_zscore":
            raise PotencyStudyDataError(f"{study_id} must preserve LNPDB target semantics")
        source_counts = _count_map(
            raw_spec.get("expected_source_endpoint_rows"),
            label=f"{study_id}.expected_source_endpoint_rows",
        )
        model_counts = _count_map(
            raw_spec.get("expected_model_endpoint_rows"),
            label=f"{study_id}.expected_model_endpoint_rows",
        )
        if set(source_counts) != set(model_counts):
            raise PotencyStudyDataError(f"{study_id} source and model endpoints differ")
        exclusions = raw_spec.get("excluded_component_labels", {})
        if not isinstance(exclusions, dict) or not set(exclusions).issubset(_COMPONENT_SLOTS):
            raise PotencyStudyDataError(f"{study_id} component exclusions are malformed")
        for slot, labels in exclusions.items():
            if not isinstance(labels, list) or any(
                not isinstance(label, str) or not label for label in labels
            ):
                raise PotencyStudyDataError(f"{study_id}.{slot} exclusion labels are malformed")
        expected_excluded = _integer(
            raw_spec.get("expected_excluded_rows", 0), label=f"{study_id} exclusions"
        )
        if sum(source_counts.values()) - sum(model_counts.values()) != expected_excluded:
            raise PotencyStudyDataError(
                f"{study_id} expected counts do not reconcile with exclusions"
            )
        if bool(exclusions) != bool(expected_excluded):
            raise PotencyStudyDataError(f"{study_id} exclusion labels and expected count disagree")
        if expected_excluded and not isinstance(raw_spec.get("exclusion_reason"), str):
            raise PotencyStudyDataError(f"{study_id} exclusions require a reason")
    return studies


def _input_path(config: Mapping[str, Any], repo: Path) -> Path:
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"lnpdb"}:
        raise PotencyStudyDataError("potency corpus input must be exactly the LNPDB pin")
    return resolve_pin(inputs["lnpdb"], repo, label="lnpdb")


def _component(row: LNPDBRow, slot: str) -> LNPDBComponent:
    if slot == "head":
        return row.head
    if slot == "linker":
        return row.linker
    return row.tail(int(slot.removeprefix("tail")))


def _excluded(row: LNPDBRow, spec: Mapping[str, Any]) -> bool:
    exclusions = spec.get("excluded_component_labels", {})
    if not isinstance(exclusions, dict):  # defensive if called outside the builder
        raise PotencyStudyDataError("excluded component labels must be an object")
    return any(_component(row, slot).name in labels for slot, labels in exclusions.items())


def _float_text(value: float) -> str:
    return format(value, ".12g")


def _summaries(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_study: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_study[str(row["study_id"])].append(row)
    return {
        study_id: {
            "rows": len(study_rows),
            "endpoints": dict(sorted(Counter(str(row["endpoint"]) for row in study_rows).items())),
            "label_semantics": sorted({str(row["label_semantics"]) for row in study_rows}),
            "unique_model_graphs": len({str(row["model_smiles"]) for row in study_rows}),
            "unique_source_records": len({str(row["source_record_id"]) for row in study_rows}),
        }
        for study_id, study_rows in sorted(by_study.items())
    }


def build_potency_study_corpus(
    config_path: Path,
    repo: Path,
    *,
    ledger_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Build the single-source modelling view over the selected LNPDB studies."""

    config = read_json_object(
        config_path,
        error=PotencyStudyDataError,
        label="potency study-corpus config",
    )
    studies = _validate_config(config)
    lnpdb_path = _input_path(config, repo)
    catalogue = load_lnpdb(lnpdb_path, error=PotencyStudyDataError)

    rows: list[dict[str, Any]] = []
    exclusions: dict[str, dict[str, Any]] = {}
    selected_count = 0
    for study_id, spec in sorted(studies.items()):
        source_rows = catalogue.study(study_id)
        selected_count += len(source_rows)
        incomplete = [
            row.lnp_id
            for row in source_rows
            if row.model_type is None
            or row.experiment_value is None
            or row.publication_pmid is None
        ]
        if incomplete:
            raise PotencyStudyDataError(
                f"{study_id} has incomplete potency metadata for {len(incomplete)} rows"
            )
        source_counts = Counter(row.model_type for row in source_rows)
        expected_source = _count_map(
            spec["expected_source_endpoint_rows"],
            label=f"{study_id}.expected_source_endpoint_rows",
        )
        if dict(source_counts) != expected_source:
            raise PotencyStudyDataError(
                f"{study_id} source endpoint counts changed: {dict(source_counts)}"
            )

        excluded_rows = [row for row in source_rows if _excluded(row, spec)]
        expected_excluded = int(spec.get("expected_excluded_rows", 0))
        if len(excluded_rows) != expected_excluded:
            raise PotencyStudyDataError(
                f"{study_id} excluded row count changed: {len(excluded_rows)}"
            )
        if excluded_rows:
            exclusions[study_id] = {
                "reason": str(spec["exclusion_reason"]),
                "source_observations": len(excluded_rows),
                "unique_source_products": len({row.lipid_name for row in excluded_rows}),
                "component_labels": spec["excluded_component_labels"],
            }

        excluded_ids = {row.lnp_id for row in excluded_rows}
        model_rows = [row for row in source_rows if row.lnp_id not in excluded_ids]
        model_counts = Counter(row.model_type for row in model_rows)
        expected_model = _count_map(
            spec["expected_model_endpoint_rows"],
            label=f"{study_id}.expected_model_endpoint_rows",
        )
        if dict(model_counts) != expected_model:
            raise PotencyStudyDataError(
                f"{study_id} model endpoint counts changed: {dict(model_counts)}"
            )
        for source in model_rows:
            if source.experiment_value is None:
                raise PotencyStudyDataError(
                    f"{source.lnp_id} has no LNPDB Experiment_value for potency modelling"
                )
            model_smiles = str(
                canonical_connected_constitution(
                    source.lipid_smiles,
                    error=PotencyStudyDataError,
                )
            )
            rows.append(
                {
                    "record_id": f"{study_id}::{source.lnp_id}",
                    "study_id": study_id,
                    "publication_pmid": source.publication_pmid,
                    "source_record_id": source.lnp_id,
                    "source_row_index": source.index,
                    "source_lipid_name": source.lipid_name,
                    "endpoint": source.model_type,
                    "label_value": _float_text(source.experiment_value),
                    "label_semantics": str(spec["label_semantics"]),
                    "raw_source_product_smiles": source.lipid_smiles,
                    "model_smiles": model_smiles,
                }
            )

    rows.sort(key=lambda row: (str(row["study_id"]), int(row["source_row_index"])))
    excluded_count = sum(item["source_observations"] for item in exclusions.values())
    if (
        len({str(row["record_id"]) for row in rows}) != len(rows)
        or len(rows) + excluded_count != selected_count
    ):
        raise PotencyStudyDataError("model and excluded rows do not account for selected LNPDB")

    ledger = csv_gz_bytes(rows, LEDGER_FIELDS)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "single_source_lnpdb_potency_corpus_complete",
        "config": {
            "path": str(config_path.resolve().relative_to(repo.resolve())),
            "sha256": str(sha256_file(config_path)),
        },
        "inputs": {
            "lnpdb": {
                "path": str(lnpdb_path.resolve().relative_to(repo.resolve())),
                "sha256": str(sha256_file(lnpdb_path)),
            }
        },
        "artifact": {
            "path": ledger_path.name,
            "rows": len(rows),
            "schema_version": LEDGER_SCHEMA,
            "sha256": hashlib.sha256(ledger).hexdigest(),
        },
        "studies": _summaries(rows),
        "source_accounting": {
            "selected_lnpdb_rows": selected_count,
            "model_observations": len(rows),
            "excluded_observations": excluded_count,
        },
        "exclusions": exclusions,
        "gates": {
            "all_selected_lnpdb_rows_accounted_for": True,
            "constitutional_single_graph_identity": True,
            "configured_mixtures_absent_from_model_corpus": True,
            "cross_study_label_pooling_disabled": True,
            "lnpdb_is_only_row_level_input": True,
            "reaction_program_fields_absent": True,
        },
        "nonclaims": [
            "LNPDB normalized values are not interchangeable raw assay measurements.",
            "A source row does not establish biological-guidance applicability.",
            "Reaction-program eligibility and evidence are owned by separate corpora.",
        ],
    }
    atomic_write(ledger_path, ledger)
    atomic_write(result_path, f"{stable_json(result)}\n".encode())
    return result


@dataclass(frozen=True, order=True)
class StudyEndpoint:
    study_id: str
    endpoint: str


@dataclass(frozen=True)
class PotencyObservation:
    record_id: str
    study: StudyEndpoint
    publication_pmid: str
    source_record_id: str
    source_row_index: int
    source_lipid_name: str
    label_value: float
    label_semantics: str
    raw_source_product_smiles: str
    model_smiles: str


@dataclass(frozen=True)
class PotencyStudyCorpus:
    """Typed access that forces callers to choose a study and endpoint."""

    observations: tuple[PotencyObservation, ...]

    def study_endpoints(self) -> tuple[StudyEndpoint, ...]:
        return tuple(sorted({row.study for row in self.observations}))

    def records(self, study: StudyEndpoint) -> tuple[PotencyObservation, ...]:
        return tuple(row for row in self.observations if row.study == study)


def load_potency_study_corpus(path: Path) -> PotencyStudyCorpus:
    """Load a produced corpus without exposing a cross-study pooled target array."""

    rows = read_csv_rows(
        path,
        error=PotencyStudyDataError,
        label="potency study corpus",
        required_fields=LEDGER_FIELDS,
    )
    observations: list[PotencyObservation] = []
    seen: set[str] = set()
    for row in rows:
        record_id = row["record_id"]
        if record_id in seen:
            raise PotencyStudyDataError(f"duplicate potency observation: {record_id}")
        seen.add(record_id)
        try:
            source_row_index = int(row["source_row_index"])
            label_value = float(row["label_value"])
        except ValueError as exc:
            raise PotencyStudyDataError(f"{record_id} has invalid numeric data") from exc
        if not row["model_smiles"]:
            raise PotencyStudyDataError(f"{record_id} model observation has no molecular graph")
        observations.append(
            PotencyObservation(
                record_id=record_id,
                study=StudyEndpoint(row["study_id"], row["endpoint"]),
                publication_pmid=row["publication_pmid"],
                source_record_id=row["source_record_id"],
                source_row_index=source_row_index,
                source_lipid_name=row["source_lipid_name"],
                label_value=label_value,
                label_semantics=row["label_semantics"],
                raw_source_product_smiles=row["raw_source_product_smiles"],
                model_smiles=row["model_smiles"],
            )
        )
    return PotencyStudyCorpus(tuple(observations))


__all__ = [
    "CONFIG_SCHEMA",
    "LEDGER_FIELDS",
    "LEDGER_SCHEMA",
    "RESULT_SCHEMA",
    "PotencyObservation",
    "PotencyStudyCorpus",
    "PotencyStudyDataError",
    "StudyEndpoint",
    "build_potency_study_corpus",
    "load_potency_study_corpus",
]
