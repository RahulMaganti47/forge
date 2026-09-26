"""Typed access to the single hash-pinned LNPDB row catalogue.

LNPDB supplies experimental rows, whole-lipid structures, and the component columns reported by
each source study.  It does not decide whether a component assignment is chemically trustworthy:
reaction adapters and source-evidence reviews own that separate admission decision.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from forge.core.io import read_csv_rows

LNPDB_FIELDS = (
    "Index",
    "LNP_ID",
    "Experiment_ID",
    "Formulation_ID",
    "IL_name",
    "IL_SMILES",
    "Model_type",
    "Experiment_value",
    "Publication_PMID",
    "IL_head_name",
    "IL_head_SMILES",
    "IL_linker_name",
    "IL_linker_SMILES",
    "IL_tail1_name",
    "IL_tail1_SMILES",
    "IL_tail2_name",
    "IL_tail2_SMILES",
    "IL_tail3_name",
    "IL_tail3_SMILES",
    "IL_tail4_name",
    "IL_tail4_SMILES",
)


class LNPDBDataError(ValueError):
    """The LNPDB catalogue is malformed or violates its row-identity contract."""


@dataclass(frozen=True)
class LNPDBComponent:
    """One source-reported component slot; either field may be unavailable."""

    name: str | None
    smiles: str | None


@dataclass(frozen=True)
class LNPDBRow:
    """One measured LNPDB observation with stable row and study identity."""

    index: int
    lnp_id: str
    experiment_id: str
    formulation_id: str | None
    lipid_name: str
    lipid_smiles: str
    model_type: str | None
    experiment_value: float | None
    publication_pmid: str | None
    head: LNPDBComponent
    linker: LNPDBComponent
    tails: tuple[LNPDBComponent, LNPDBComponent, LNPDBComponent, LNPDBComponent]

    def tail(self, position: int) -> LNPDBComponent:
        """Return a one-indexed tail slot, matching the source column names."""

        if position < 1 or position > len(self.tails):
            raise IndexError(f"LNPDB tail position must be 1..{len(self.tails)}, got {position}")
        return self.tails[position - 1]


@dataclass(frozen=True)
class LNPDBCatalogue:
    """Immutable LNPDB rows indexed once by study and source identifier."""

    rows: tuple[LNPDBRow, ...]
    _by_study: Mapping[str, tuple[LNPDBRow, ...]]
    _by_lnp_id: Mapping[str, LNPDBRow]

    def study(self, experiment_id: str) -> tuple[LNPDBRow, ...]:
        return self._by_study.get(experiment_id, ())

    def record(self, lnp_id: str) -> LNPDBRow:
        try:
            return self._by_lnp_id[lnp_id]
        except KeyError as exc:
            raise KeyError(f"unknown LNPDB record: {lnp_id}") from exc


def _text(value: str, *, field: str, row_number: int, error: type[ValueError]) -> str:
    normalized = value.strip()
    if not normalized or normalized == "NA":
        raise error(f"LNPDB row {row_number} has no {field}")
    return normalized


def _optional(value: str) -> str | None:
    normalized = value.strip()
    return None if not normalized or normalized == "NA" else normalized


def _component(row: Mapping[str, str], prefix: str) -> LNPDBComponent:
    return LNPDBComponent(
        name=_optional(row[f"{prefix}_name"]),
        smiles=_optional(row[f"{prefix}_SMILES"]),
    )


def load_lnpdb(
    path: Path,
    *,
    error: type[ValueError] = LNPDBDataError,
) -> LNPDBCatalogue:
    """Load and validate LNPDB without assigning reaction or biological meaning."""

    source_rows = read_csv_rows(
        path,
        error=error,
        label="LNPDB",
        required_fields=LNPDB_FIELDS,
    )
    rows: list[LNPDBRow] = []
    by_study: dict[str, list[LNPDBRow]] = defaultdict(list)
    by_lnp_id: dict[str, LNPDBRow] = {}
    seen_indices: set[int] = set()
    for row_number, source in enumerate(source_rows, start=2):
        raw_index = source["Index"].strip()
        try:
            index = int(raw_index)
        except ValueError as exc:
            raise error(f"LNPDB row {row_number} has invalid Index {raw_index!r}") from exc
        if index in seen_indices:
            raise error(f"LNPDB Index is duplicated: {index}")
        seen_indices.add(index)

        lnp_id = _text(source["LNP_ID"], field="LNP_ID", row_number=row_number, error=error)
        if lnp_id in by_lnp_id:
            raise error(f"LNPDB source record is duplicated: {lnp_id}")
        experiment_id = _text(
            source["Experiment_ID"], field="Experiment_ID", row_number=row_number, error=error
        )
        raw_value = source["Experiment_value"].strip()
        experiment_value: float | None = None
        if raw_value not in {"", "NA"}:
            try:
                experiment_value = float(raw_value)
            except ValueError as exc:
                raise error(f"LNPDB {lnp_id} has invalid Experiment_value {raw_value!r}") from exc
            if not math.isfinite(experiment_value):
                raise error(f"LNPDB {lnp_id} Experiment_value must be finite")

        row = LNPDBRow(
            index=index,
            lnp_id=lnp_id,
            experiment_id=experiment_id,
            formulation_id=_optional(source["Formulation_ID"]),
            lipid_name=_text(
                source["IL_name"], field="IL_name", row_number=row_number, error=error
            ),
            lipid_smiles=_text(
                source["IL_SMILES"], field="IL_SMILES", row_number=row_number, error=error
            ),
            model_type=_optional(source["Model_type"]),
            experiment_value=experiment_value,
            publication_pmid=_optional(source["Publication_PMID"]),
            head=_component(source, "IL_head"),
            linker=_component(source, "IL_linker"),
            tails=tuple(  # type: ignore[arg-type]
                _component(source, f"IL_tail{position}") for position in range(1, 5)
            ),
        )
        rows.append(row)
        by_study[experiment_id].append(row)
        by_lnp_id[lnp_id] = row

    return LNPDBCatalogue(
        rows=tuple(rows),
        _by_study=MappingProxyType(
            {study: tuple(study_rows) for study, study_rows in sorted(by_study.items())}
        ),
        _by_lnp_id=MappingProxyType(dict(by_lnp_id)),
    )


__all__ = [
    "LNPDB_FIELDS",
    "LNPDBCatalogue",
    "LNPDBComponent",
    "LNPDBDataError",
    "LNPDBRow",
    "load_lnpdb",
]
