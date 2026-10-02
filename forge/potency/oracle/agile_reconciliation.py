"""Reconcile AGILE assay labels and molecular identities before oracle training.

The published AGILE library contains 1,200 nominal formulation measurements.
LANTERN established that the B4 reagent was a cis/trans mixture while B5 was a
pure trans reagent. A single-graph oracle cannot represent the B4 mixtures.
This module therefore:

* verifies HeLa and RAW labels against the official AGILE source workbook;
* excludes the 100 B4 mixture observations from single-graph supervision;
* corrects the 100 B5 product and component structures to the pure-trans graph;
* cross-validates the resulting 1,100 HeLa records against LANTERN; and
* emits deterministic audit, exclusion, and oracle-training artifacts.

The correction is a representation-policy gate. It does not add stereochemical
generation to FORGE. The model-facing SMILES are constitutional canonical
SMILES, while isomeric identities are retained for provenance and chemistry.
"""

from __future__ import annotations

import re

CONFIG_SCHEMA_VERSION = "m0_07_agile_reconciliation_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_agile_reconciliation.v1"

CURATED_FIELDS = (
    "source_row_id",
    "label",
    "model_smiles",
    "isomeric_smiles",
    "A_smiles",
    "B_smiles",
    "C_smiles",
    "expt_Hela",
    "expt_Raw",
    "structure_policy",
    "hela_label_source",
    "raw_label_source",
)
EXCLUSION_FIELDS = (
    "source_row_id",
    "label",
    "model_smiles",
    "reported_mixture_smiles",
    "mixture_members_json",
    "A_smiles",
    "original_B_smiles",
    "C_smiles",
    "expt_Hela",
    "expt_Raw",
    "exclusion_reason",
)
LEDGER_FIELDS = (
    "source_row_id",
    "label",
    "component_A_id",
    "component_B_id",
    "component_C_id",
    "original_model_smiles",
    "original_isomeric_smiles",
    "reconciled_model_smiles",
    "reconciled_isomeric_smiles",
    "original_B_smiles",
    "reconciled_B_smiles",
    "action",
    "source_hela",
    "source_raw",
    "lantern_hela",
)

_LABEL_RE = re.compile(r"^A(?P<a>\d+)(?:B(?P<b>\d+)C(?P<c>\d+)|C(?P<c2>\d+)B(?P<b2>\d+))$")
_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
