"""Adjudicate the frozen broad/support/potency morphology experiment.

The terminal policy in this module is intentionally narrower than the older
HeLa diagnostic wrapper.  The held aldehyde--isocyanide split tested a new
pairing of component identities that were each observed elsewhere in the fit
population; it did not test two exact-new components.  This module therefore
uses the older artifact only for its authenticated oracle, four-view chemical
distances, and held-pair calibration scale.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.diagnostics.hela.evaluation import (
    ROLE_MAP,
)

CONFIG_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication_config.v1"
POLICY_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication_policy.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication.v1"
LEDGER_SCHEMA_VERSION = "phase1_ugi_morphology_potency_matched_adjudication_ledger.v1"
BROAD_ARM = "broad_prior"
SUPPORT_ARM = "support_enriched"
POTENCY_ARM = "nested_potency"
EXPECTED_ARMS = (BROAD_ARM, SUPPORT_ARM, POTENCY_ARM)
PAIR_SCALE = "aldehyde_isocyanide_pair"


class UgiMorphologyPotencyMatchedAdjudicationError(RuntimeError):
    """Raised when the frozen matched diagnostic cannot be reproduced."""


def _exact_l1(terminal: Mapping[str, Any]) -> bool:
    verification = terminal.get("l1_forward_verification")
    return bool(
        terminal.get("valid") is True
        and terminal.get("terminal_valid") is True
        and terminal.get("component_reconstruction_valid") is True
        and isinstance(verification, Mapping)
        and verification.get("exact_product_reconstructed") is True
        and verification.get("maximum_outcomes_saturated") is False
    )


def _candidate(row: Mapping[str, Any]) -> dict[str, str]:
    terminal = row.get("native_terminal")
    if not isinstance(terminal, Mapping) or not _exact_l1(terminal):
        raise UgiMorphologyPotencyMatchedAdjudicationError(
            "candidate requested from a nonexact terminal"
        )
    components = terminal.get("component_smiles_by_role")
    if not isinstance(components, Mapping):
        raise UgiMorphologyPotencyMatchedAdjudicationError("terminal components are absent")
    try:
        return {
            "label": _sha256_payload(
                [str(row["arm_id"]), int(row["draw_index"]), str(terminal["smiles"])]
            ),
            "product_smiles": str(terminal["smiles"]),
            **{
                f"{role}_smiles": str(components[native_role])
                for role, native_role in ROLE_MAP.items()
            },
        }
    except KeyError as error:
        raise UgiMorphologyPotencyMatchedAdjudicationError(
            "terminal component roles changed"
        ) from error


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "LEDGER_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "UgiMorphologyPotencyMatchedAdjudicationError",
]
