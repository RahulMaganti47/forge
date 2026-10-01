"""Typed, nonselecting Ugi terminal-to-route-assessor integration.

This module binds one sealed, chemically valid and exact-L1 Ugi terminal to
three independently budgeted component route assessments.  It returns only
structured pre-prospective diagnostics.  It does not define a scalar synthesis
value, resample particles, select candidates or invoke a biological oracle.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, rdBase

from forge.core.io import stable_json as _stable_json
from forge.corpus.ugi_generated_components import (
    generated_ugi_component_smiles,
    precursor_components_from_product_semantics,
)
from forge.corpus.ugi_held_component_gate import exact_forward_reconstructs_ugi_product
from forge.model.ugi_chemistry_flow import (
    UgiChemistrySample,
    chemistry_sample_to_molecule,
)
from forge.model.ugi_chemistry_interface import ChemistryTopologyCondition
from forge.potency.annotations import ROLE_NAMES

VALIDATED_TERMINAL_SCHEMA_VERSION = "forge.validated_ugi_terminal_payload.v1"
TERMINAL_ROUTE_ASSESSMENT_SCHEMA_VERSION = "forge.ugi_terminal_route_assessment.v1"
TARGET_CONTEXT_POLICY = "component_intrinsic_empty_product_context.v1"
DEFAULT_IDENTITY_POLICY = "canonical_constitutional_smiles"
DEFAULT_STEREOCHEMISTRY_POLICY = "phase1_stereo_free"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class UgiTerminalRouteAssessmentError(RuntimeError):
    """Raised when the typed terminal-to-route boundary fails closed."""


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise UgiTerminalRouteAssessmentError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise UgiTerminalRouteAssessmentError(
            f"{label} must contain 64 lowercase hexadecimal characters"
        )
    return value


def _require_nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiTerminalRouteAssessmentError(f"{label} must be a nonnegative integer")
    return value


def _canonical_constitution(smiles: Any, *, label: str) -> str:
    _require_nonempty(smiles, label=label)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiTerminalRouteAssessmentError(
            f"{label} must be one valid connected molecular constitution"
        )
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


@dataclass(frozen=True)
class UgiRoleComponent:
    """One exact precursor role and its canonical constitutional identity."""

    role: str
    canonical_smiles: str

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(f"unsupported Ugi role: {self.role!r}")
        canonical = _canonical_constitution(
            self.canonical_smiles,
            label=f"{self.role} component",
        )
        if self.canonical_smiles != canonical:
            raise UgiTerminalRouteAssessmentError(
                f"{self.role} component must already be canonical constitutional SMILES"
            )

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "canonical_smiles": self.canonical_smiles}

    @classmethod
    def from_dict(cls, value: Any) -> UgiRoleComponent:
        if not isinstance(value, dict) or set(value) != {"role", "canonical_smiles"}:
            raise UgiTerminalRouteAssessmentError("Ugi component has an unsupported schema")
        return cls(role=value.get("role"), canonical_smiles=value.get("canonical_smiles"))


@dataclass(frozen=True)
class ExactL1ForwardVerification:
    """Exact qualified Ugi reconstruction state retained at terminal admission."""

    exact_product_reconstructed: bool
    maximum_outcomes: int
    maximum_outcomes_saturated: bool
    outcome_count: int

    def __post_init__(self) -> None:
        if self.exact_product_reconstructed is not True:
            raise UgiTerminalRouteAssessmentError(
                "validated terminal payload requires exact L1 product reconstruction"
            )
        if (
            isinstance(self.maximum_outcomes, bool)
            or not isinstance(self.maximum_outcomes, int)
            or self.maximum_outcomes <= 0
        ):
            raise UgiTerminalRouteAssessmentError("maximum_outcomes must be a positive integer")
        _require_nonnegative_integer(self.outcome_count, label="L1 outcome_count")
        if self.outcome_count == 0:
            raise UgiTerminalRouteAssessmentError(
                "exact L1 reconstruction cannot contain zero forward outcomes"
            )
        if not isinstance(self.maximum_outcomes_saturated, bool):
            raise UgiTerminalRouteAssessmentError("maximum_outcomes_saturated must be boolean")
        expected_saturation = self.outcome_count >= self.maximum_outcomes
        if self.maximum_outcomes_saturated != expected_saturation:
            raise UgiTerminalRouteAssessmentError(
                "L1 saturation flag disagrees with outcome_count and maximum_outcomes"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "exact_product_reconstructed": self.exact_product_reconstructed,
            "maximum_outcomes": self.maximum_outcomes,
            "maximum_outcomes_saturated": self.maximum_outcomes_saturated,
            "outcome_count": self.outcome_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> ExactL1ForwardVerification:
        if not isinstance(value, dict) or set(value) != {
            "exact_product_reconstructed",
            "maximum_outcomes",
            "maximum_outcomes_saturated",
            "outcome_count",
        }:
            raise UgiTerminalRouteAssessmentError("L1 verification has an unsupported schema")
        return cls(**value)


@dataclass(frozen=True)
class QualifiedUgiL1Reverifier:
    """Qualified reaction contract bound to the L1 artifact it reverifies.

    This object exists so route admission never trusts a serialized exact-L1
    flag.  The caller must bind the compiled qualified reaction to the same
    content hash frozen in the planner and terminal contracts.
    """

    reaction_contract: Any
    l1_reaction_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.l1_reaction_sha256, label="l1_reaction_sha256")
        try:
            role_order = tuple(
                role.name for role in self.reaction_contract.definition.reactant_roles
            )
        except (AttributeError, TypeError) as error:
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reverifier reaction contract is malformed"
            ) from error
        if role_order != ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reverifier role order differs from the frozen Ugi adapter"
            )

    def require_exact(
        self,
        payload: ValidatedUgiTerminalPayload,
    ) -> ExactL1ForwardVerification:
        """Independently rerun the qualified forward transform and fail closed."""

        if not isinstance(payload, ValidatedUgiTerminalPayload):
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reverification requires a validated payload record"
            )
        if payload.l1_reaction_sha256 != self.l1_reaction_sha256:
            raise UgiTerminalRouteAssessmentError(
                "L1 reverifier and terminal payload use different reaction hashes"
            )
        exact, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
            self.reaction_contract,
            {component.role: component.canonical_smiles for component in payload.components},
            payload.product_smiles,
            maximum_outcomes=(payload.l1_forward_verification.maximum_outcomes),
        )
        if not exact:
            raise UgiTerminalRouteAssessmentError(
                "independent qualified L1 reverification did not reconstruct the terminal"
            )
        stored = payload.l1_forward_verification
        if saturated != stored.maximum_outcomes_saturated or outcome_count != stored.outcome_count:
            raise UgiTerminalRouteAssessmentError(
                "independent qualified L1 result disagrees with serialized verification"
            )
        return ExactL1ForwardVerification(
            exact_product_reconstructed=True,
            maximum_outcomes=stored.maximum_outcomes,
            maximum_outcomes_saturated=saturated,
            outcome_count=outcome_count,
        )


@dataclass(frozen=True)
class ValidatedUgiTerminalPayload:
    """Canonical exact-L1 terminal payload suitable for route assessment."""

    product_smiles: str
    components: tuple[UgiRoleComponent, ...]
    l1_forward_verification: ExactL1ForwardVerification
    l1_reaction_sha256: str
    component_recovery_contract_sha256: str
    identity_policy: str = DEFAULT_IDENTITY_POLICY
    stereochemistry_policy: str = DEFAULT_STEREOCHEMISTRY_POLICY

    def __post_init__(self) -> None:
        canonical_product = _canonical_constitution(
            self.product_smiles,
            label="terminal product",
        )
        if self.product_smiles != canonical_product:
            raise UgiTerminalRouteAssessmentError(
                "terminal product must already be canonical constitutional SMILES"
            )
        if (
            not isinstance(self.components, tuple)
            or tuple(component.role for component in self.components) != ROLE_NAMES
        ):
            raise UgiTerminalRouteAssessmentError(
                "terminal payload must contain exactly the three Ugi roles in frozen order"
            )
        if any(not isinstance(component, UgiRoleComponent) for component in self.components):
            raise UgiTerminalRouteAssessmentError(
                "terminal components must be UgiRoleComponent records"
            )
        if not isinstance(self.l1_forward_verification, ExactL1ForwardVerification):
            raise UgiTerminalRouteAssessmentError(
                "terminal payload requires typed exact-L1 verification"
            )
        _require_sha256(self.l1_reaction_sha256, label="l1_reaction_sha256")
        _require_sha256(
            self.component_recovery_contract_sha256,
            label="component_recovery_contract_sha256",
        )
        _require_nonempty(self.identity_policy, label="identity_policy")
        _require_nonempty(self.stereochemistry_policy, label="stereochemistry_policy")

    @classmethod
    def from_recovered_components(
        cls,
        *,
        product_smiles: str,
        components_by_role: Mapping[str, str],
        l1_reaction: Any,
        l1_reaction_sha256: str,
        component_recovery_contract_sha256: str,
        maximum_outcomes: int = 64,
        identity_policy: str = DEFAULT_IDENTITY_POLICY,
        stereochemistry_policy: str = DEFAULT_STEREOCHEMISTRY_POLICY,
    ) -> ValidatedUgiTerminalPayload:
        """Canonicalize exact recovered roles and independently verify L1."""

        if not isinstance(components_by_role, Mapping) or set(components_by_role) != set(
            ROLE_NAMES
        ):
            raise UgiTerminalRouteAssessmentError(
                "recovered components must contain exactly the three frozen Ugi roles"
            )
        role_order = tuple(role.name for role in l1_reaction.definition.reactant_roles)
        if role_order != ROLE_NAMES:
            raise UgiTerminalRouteAssessmentError(
                "qualified L1 reaction role order differs from the frozen Ugi adapter"
            )
        if (
            isinstance(maximum_outcomes, bool)
            or not isinstance(maximum_outcomes, int)
            or maximum_outcomes <= 0
        ):
            raise UgiTerminalRouteAssessmentError("maximum_outcomes must be a positive integer")
        canonical_product = _canonical_constitution(product_smiles, label="terminal product")
        canonical_components = {
            role: _canonical_constitution(
                components_by_role[role],
                label=f"recovered {role} component",
            )
            for role in ROLE_NAMES
        }
        exact, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
            l1_reaction,
            canonical_components,
            canonical_product,
            maximum_outcomes=maximum_outcomes,
        )
        if not exact:
            raise UgiTerminalRouteAssessmentError(
                "recovered Ugi components do not exactly reconstruct the terminal product"
            )
        return cls(
            product_smiles=canonical_product,
            components=tuple(
                UgiRoleComponent(role=role, canonical_smiles=canonical_components[role])
                for role in ROLE_NAMES
            ),
            l1_forward_verification=ExactL1ForwardVerification(
                exact_product_reconstructed=True,
                maximum_outcomes=maximum_outcomes,
                maximum_outcomes_saturated=saturated,
                outcome_count=outcome_count,
            ),
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            identity_policy=identity_policy,
            stereochemistry_policy=stereochemistry_policy,
        )

    @classmethod
    def from_product_semantics(
        cls,
        *,
        product: Chem.Mol,
        origin_states: Sequence[int],
        core_position_states: Sequence[int],
        l1_reaction: Any,
        l1_reaction_sha256: str,
        component_recovery_contract_sha256: str,
        maximum_outcomes: int = 64,
    ) -> ValidatedUgiTerminalPayload:
        """Recover the exact three Ugi roles from product atom semantics."""

        try:
            components = precursor_components_from_product_semantics(
                product,
                origin_states,
                core_position_states,
            )
            product_smiles = Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=False,
            )
        except (RuntimeError, ValueError) as error:
            raise UgiTerminalRouteAssessmentError(
                "exact Ugi precursor recovery from product semantics failed"
            ) from error
        return cls.from_recovered_components(
            product_smiles=product_smiles,
            components_by_role=components,
            l1_reaction=l1_reaction,
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            maximum_outcomes=maximum_outcomes,
        )

    @classmethod
    def from_generated_semantics(
        cls,
        *,
        condition: ChemistryTopologyCondition,
        sample: UgiChemistrySample,
        atom_vocabulary: tuple[Any, ...],
        l1_reaction: Any,
        l1_reaction_sha256: str,
        component_recovery_contract_sha256: str,
        maximum_outcomes: int = 64,
    ) -> ValidatedUgiTerminalPayload:
        """Build the payload directly from a completed generated terminal."""

        try:
            product = chemistry_sample_to_molecule(condition, sample, atom_vocabulary)
            components = generated_ugi_component_smiles(condition, sample, atom_vocabulary)
            product_smiles = Chem.MolToSmiles(
                product,
                canonical=True,
                isomericSmiles=False,
            )
        except (RuntimeError, ValueError) as error:
            raise UgiTerminalRouteAssessmentError(
                "generated terminal chemistry or exact component recovery failed"
            ) from error
        return cls.from_recovered_components(
            product_smiles=product_smiles,
            components_by_role=components,
            l1_reaction=l1_reaction,
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            maximum_outcomes=maximum_outcomes,
        )

    def by_role(self) -> dict[str, UgiRoleComponent]:
        return {component.role: component for component in self.components}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATED_TERMINAL_SCHEMA_VERSION,
            "product_smiles": self.product_smiles,
            "components": [component.to_dict() for component in self.components],
            "l1_forward_verification": self.l1_forward_verification.to_dict(),
            "l1_reaction_sha256": self.l1_reaction_sha256,
            "component_recovery_contract_sha256": self.component_recovery_contract_sha256,
            "identity_policy": self.identity_policy,
            "stereochemistry_policy": self.stereochemistry_policy,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return (_stable_json(self.to_dict()) + "\n").encode()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_dict(cls, value: Any) -> ValidatedUgiTerminalPayload:
        expected = {
            "schema_version",
            "product_smiles",
            "components",
            "l1_forward_verification",
            "l1_reaction_sha256",
            "component_recovery_contract_sha256",
            "identity_policy",
            "stereochemistry_policy",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("schema_version") != VALIDATED_TERMINAL_SCHEMA_VERSION
        ):
            raise UgiTerminalRouteAssessmentError(
                "validated Ugi terminal payload has an unsupported schema"
            )
        components = value.get("components")
        if not isinstance(components, list):
            raise UgiTerminalRouteAssessmentError("terminal components must be a list")
        return cls(
            product_smiles=value.get("product_smiles"),
            components=tuple(UgiRoleComponent.from_dict(item) for item in components),
            l1_forward_verification=ExactL1ForwardVerification.from_dict(
                value.get("l1_forward_verification")
            ),
            l1_reaction_sha256=value.get("l1_reaction_sha256"),
            component_recovery_contract_sha256=value.get("component_recovery_contract_sha256"),
            identity_policy=value.get("identity_policy"),
            stereochemistry_policy=value.get("stereochemistry_policy"),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> ValidatedUgiTerminalPayload:
        if not isinstance(payload, bytes) or not payload:
            raise UgiTerminalRouteAssessmentError("terminal payload bytes must be nonempty")
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UgiTerminalRouteAssessmentError(
                "terminal payload bytes are invalid JSON"
            ) from error
        record = cls.from_dict(value)
        if payload != record.canonical_bytes:
            raise UgiTerminalRouteAssessmentError("terminal payload bytes are not canonical")
        return record
