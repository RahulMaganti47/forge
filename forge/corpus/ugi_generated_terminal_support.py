"""Fail-closed support qualification for one generated Ugi terminal.

This module is the narrow seam between the frozen generated-candidate
eligibility contract and route assessment.  It does not infer route evidence,
assign a synthesis value, select candidates, or inspect biological or holdout
data.  A terminal receives route-root qualifications only after its sealed
payload, exact L1 reconstruction, declared graph support, component recovery,
and all three frozen Ugi handle policies are independently rechecked.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, rdBase

from forge.corpus.ugi_component_expansion import reaction_handle_qualification
from forge.evaluation.eligibility import declared_support_violations
from forge.potency.annotations import ROLE_NAMES
from forge.synthesis.assessment.ugi3_support_boundary import (
    AuthenticatedInternalRoleRegistry,
    MolecularSupportState,
    QualificationKey,
    SupportBoundaryNormalizedUgi3Source,
    TargetQualification,
    qualification_key,
)
from forge.synthesis.engine.planner import RouteKnowledgeSource, RouteTarget
from forge.synthesis.matched import LockedMatchedTerminal
from forge.synthesis.terminals.terminal_assessment import (
    ExactL1ForwardVerification,
    QualifiedUgiL1Reverifier,
    UgiTerminalRouteAssessmentError,
    ValidatedUgiTerminalPayload,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
DECLARED_GRAPH_SUPPORT_SCHEMA_VERSION = "forge.declared_graph_support_context.v1"
_GRAPH_SUPPORT_FIELDS = (
    "maximum_total_atoms",
    "maximum_component_atoms",
    "maximum_junction_budget",
    "maximum_cycle_rank",
    "maximum_attachment_count",
    "maximum_children",
)


class UgiGeneratedTerminalSupportError(RuntimeError):
    """Raised when generated-terminal route qualification cannot be established."""


@dataclass(frozen=True)
class DeclaredGraphSupportContext:
    """Current generator support inputs used by the frozen eligibility check."""

    generator_checkpoint_sha256: str
    model_config: Mapping[str, Any]
    atom_vocabulary: frozenset[tuple[str, int, bool, int]]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generator_checkpoint_sha256, str)
            or _SHA256.fullmatch(self.generator_checkpoint_sha256) is None
        ):
            raise UgiGeneratedTerminalSupportError(
                "generator_checkpoint_sha256 must contain 64 lowercase hexadecimal characters"
            )
        if not isinstance(self.model_config, Mapping):
            raise UgiGeneratedTerminalSupportError("model_config must be a mapping")
        missing = tuple(field for field in _GRAPH_SUPPORT_FIELDS if field not in self.model_config)
        if missing:
            raise UgiGeneratedTerminalSupportError(
                f"declared graph-support fields are missing: {', '.join(missing)}"
            )
        for field in _GRAPH_SUPPORT_FIELDS:
            value = self.model_config[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise UgiGeneratedTerminalSupportError(
                    f"declared graph-support field {field} must be an integer"
                )
        if (
            self.model_config["maximum_total_atoms"] <= 0
            or self.model_config["maximum_component_atoms"] <= 0
            or self.model_config["maximum_junction_budget"] < 0
            or self.model_config["maximum_cycle_rank"] < 0
            or self.model_config["maximum_attachment_count"] <= 0
            or self.model_config["maximum_children"] < 0
        ):
            raise UgiGeneratedTerminalSupportError(
                "declared graph-support bounds contain an invalid value"
            )
        if not isinstance(self.atom_vocabulary, frozenset) or not self.atom_vocabulary:
            raise UgiGeneratedTerminalSupportError("atom_vocabulary must be a nonempty frozenset")
        for state in self.atom_vocabulary:
            if (
                not isinstance(state, tuple)
                or len(state) != 4
                or not isinstance(state[0], str)
                or not state[0]
                or isinstance(state[1], bool)
                or not isinstance(state[1], int)
                or not isinstance(state[2], bool)
                or isinstance(state[3], bool)
                or not isinstance(state[3], int)
                or state[3] < 0
            ):
                raise UgiGeneratedTerminalSupportError(
                    "atom_vocabulary contains a malformed atom-state tuple"
                )


def declared_graph_support_context_sha256(value: DeclaredGraphSupportContext) -> str:
    """Return the one canonical identity for eligibility-relevant graph support."""

    if not isinstance(value, DeclaredGraphSupportContext):
        raise UgiGeneratedTerminalSupportError(
            "declared graph support must be a DeclaredGraphSupportContext"
        )
    payload = {
        "schema_version": DECLARED_GRAPH_SUPPORT_SCHEMA_VERSION,
        "generator_checkpoint_sha256": value.generator_checkpoint_sha256,
        "model_config": {field: value.model_config[field] for field in _GRAPH_SUPPORT_FIELDS},
        "atom_vocabulary": [list(item) for item in sorted(value.atom_vocabulary)],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RoleHandleRecheck:
    """Exact frozen handle-policy result retained without route interpretation."""

    role: str
    raw_handle_matches: int
    symmetry_distinct_handle_sites: int
    forbidden_substructure_match: bool
    passes_registry_handle_policy: bool

    def __post_init__(self) -> None:
        if self.role not in ROLE_NAMES:
            raise UgiGeneratedTerminalSupportError(f"unsupported Ugi role: {self.role!r}")
        for field in ("raw_handle_matches", "symmetry_distinct_handle_sites"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise UgiGeneratedTerminalSupportError(
                    f"{self.role} {field} must be a nonnegative integer"
                )
        if not isinstance(self.forbidden_substructure_match, bool) or not isinstance(
            self.passes_registry_handle_policy, bool
        ):
            raise UgiGeneratedTerminalSupportError(
                f"{self.role} handle-policy states must be boolean"
            )
        if self.forbidden_substructure_match or not self.passes_registry_handle_policy:
            raise UgiGeneratedTerminalSupportError(
                f"{self.role} does not pass the exact frozen handle policy"
            )


@dataclass(frozen=True)
class QualifiedGeneratedUgiTerminalSupport:
    """Immutable qualifications for the three exact route roots of one terminal."""

    terminal_sha256: str
    generator_checkpoint_sha256: str
    product_smiles: str
    l1_reverification: ExactL1ForwardVerification
    handle_rechecks: tuple[RoleHandleRecheck, ...]
    root_targets: tuple[RouteTarget, ...]
    root_qualifications: tuple[TargetQualification, ...]

    def __post_init__(self) -> None:
        for label, value in (
            ("terminal_sha256", self.terminal_sha256),
            ("generator_checkpoint_sha256", self.generator_checkpoint_sha256),
        ):
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise UgiGeneratedTerminalSupportError(
                    f"{label} must contain 64 lowercase hexadecimal characters"
                )
        if tuple(item.role for item in self.handle_rechecks) != ROLE_NAMES:
            raise UgiGeneratedTerminalSupportError(
                "handle rechecks must contain the three frozen Ugi roles in order"
            )
        if tuple(target.role for target in self.root_targets) != ROLE_NAMES:
            raise UgiGeneratedTerminalSupportError(
                "route roots must contain the three frozen Ugi roles in order"
            )
        if len(self.root_qualifications) != len(ROLE_NAMES) or any(
            not isinstance(item, TargetQualification) for item in self.root_qualifications
        ):
            raise UgiGeneratedTerminalSupportError(
                "route-root qualifications must contain three typed records"
            )

    @property
    def qualification_mapping(self) -> dict[QualificationKey, TargetQualification]:
        """Return the exact root-key mapping consumed by the normalized source."""

        return {
            qualification_key(target): qualification
            for target, qualification in zip(
                self.root_targets,
                self.root_qualifications,
                strict=True,
            )
        }

    def wrap_source(
        self,
        delegate: RouteKnowledgeSource,
        *,
        authenticated_internal_roles: AuthenticatedInternalRoleRegistry,
    ) -> SupportBoundaryNormalizedUgi3Source:
        """Bind these exact roots to the existing typed support-boundary source."""

        return SupportBoundaryNormalizedUgi3Source(
            delegate,
            self.qualification_mapping,
            authenticated_internal_roles=authenticated_internal_roles,
        )


def _canonical_molecule(value: Any, *, label: str) -> tuple[Chem.Mol, str]:
    if not isinstance(value, str) or not value:
        raise UgiGeneratedTerminalSupportError(f"{label} must be a nonempty SMILES string")
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(value)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise UgiGeneratedTerminalSupportError(
            f"{label} must be one valid connected molecular constitution"
        )
    return molecule, Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _validated_payload(terminal: LockedMatchedTerminal) -> ValidatedUgiTerminalPayload:
    if not isinstance(terminal, LockedMatchedTerminal):
        raise UgiGeneratedTerminalSupportError(
            "support qualification requires a LockedMatchedTerminal"
        )
    if not terminal.terminal_locked:
        raise UgiGeneratedTerminalSupportError("support qualification requires a sealed terminal")
    if not terminal.terminal_valid:
        raise UgiGeneratedTerminalSupportError("support qualification requires a valid terminal")
    if not terminal.exact_l1:
        raise UgiGeneratedTerminalSupportError(
            "support qualification requires exact L1 reconstruction"
        )
    if not isinstance(terminal.payload, ValidatedUgiTerminalPayload):
        raise UgiGeneratedTerminalSupportError(
            "sealed terminal payload must be ValidatedUgiTerminalPayload"
        )
    if terminal.terminal_bytes != terminal.payload.canonical_bytes:
        raise UgiGeneratedTerminalSupportError(
            "sealed terminal bytes do not match the validated payload"
        )
    return terminal.payload


def _require_candidate_record(
    candidate_record: Mapping[str, Any],
    payload: ValidatedUgiTerminalPayload,
) -> tuple[Chem.Mol, dict[str, Chem.Mol]]:
    if not isinstance(candidate_record, Mapping):
        raise UgiGeneratedTerminalSupportError("candidate_record must be a mapping")
    if candidate_record.get("valid") is not True:
        raise UgiGeneratedTerminalSupportError(
            "candidate valid state is missing, false, or unassessed"
        )
    if candidate_record.get("component_reconstruction_valid") is not True:
        raise UgiGeneratedTerminalSupportError(
            "candidate component reconstruction is missing, false, or unassessed"
        )
    for field in ("program", "offspring_by_role"):
        if not isinstance(candidate_record.get(field), Mapping):
            raise UgiGeneratedTerminalSupportError(f"candidate {field} is missing or unassessed")

    product, canonical_product = _canonical_molecule(
        candidate_record.get("smiles"),
        label="candidate product",
    )
    if canonical_product != payload.product_smiles:
        raise UgiGeneratedTerminalSupportError(
            "candidate product does not match the sealed terminal payload"
        )

    components = candidate_record.get("component_smiles_by_role")
    if not isinstance(components, Mapping) or set(components) != set(ROLE_NAMES):
        raise UgiGeneratedTerminalSupportError(
            "candidate components are missing, unassessed, or do not contain exactly three roles"
        )
    payload_components = payload.by_role()
    component_molecules: dict[str, Chem.Mol] = {}
    for role in ROLE_NAMES:
        molecule, canonical = _canonical_molecule(
            components[role],
            label=f"candidate {role} component",
        )
        if canonical != payload_components[role].canonical_smiles:
            raise UgiGeneratedTerminalSupportError(
                f"candidate {role} component does not match the sealed terminal payload"
            )
        component_molecules[role] = molecule
    return product, component_molecules


def qualify_locked_generated_ugi_terminal_support(
    terminal: LockedMatchedTerminal,
    *,
    candidate_record: Mapping[str, Any],
    graph_support: DeclaredGraphSupportContext,
    l1_reverifier: QualifiedUgiL1Reverifier,
) -> QualifiedGeneratedUgiTerminalSupport:
    """Recheck a locked generated terminal and derive exact route-root qualifications.

    Every input is selection-independent.  Any missing state, graph-support
    violation, component mismatch, failed exact L1 rerun, or handle-policy
    failure raises ``UgiGeneratedTerminalSupportError`` before a route source is
    constructed.
    """

    if not isinstance(graph_support, DeclaredGraphSupportContext):
        raise UgiGeneratedTerminalSupportError(
            "graph_support must be a DeclaredGraphSupportContext"
        )
    payload = _validated_payload(terminal)
    if terminal.generator_checkpoint_sha256 != graph_support.generator_checkpoint_sha256:
        raise UgiGeneratedTerminalSupportError(
            "terminal and declared graph-support contexts use different generator checkpoints"
        )
    if not isinstance(l1_reverifier, QualifiedUgiL1Reverifier):
        raise UgiGeneratedTerminalSupportError("l1_reverifier must be a QualifiedUgiL1Reverifier")
    product, component_molecules = _require_candidate_record(candidate_record, payload)

    try:
        l1_reverification = l1_reverifier.require_exact(payload)
    except UgiTerminalRouteAssessmentError as error:
        raise UgiGeneratedTerminalSupportError(
            "independent qualified L1 reverification failed"
        ) from error

    try:
        violations = declared_support_violations(
            candidate_record,
            product,
            graph_support.model_config,
            set(graph_support.atom_vocabulary),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise UgiGeneratedTerminalSupportError(
            "candidate graph-support fields are missing or malformed"
        ) from error
    if violations:
        raise UgiGeneratedTerminalSupportError(
            "candidate violates the declared graph-support contract: " + ", ".join(violations)
        )

    reaction = l1_reverifier.reaction_contract
    handle_rechecks: list[RoleHandleRecheck] = []
    for role_index, role in enumerate(ROLE_NAMES):
        try:
            result = reaction_handle_qualification(
                component_molecules[role],
                query=reaction.handles[role_index],
                forbidden=reaction.forbidden[role_index],
                allowed_site_multiplicity=(
                    reaction.definition.reactant_roles[role_index].allowed_site_multiplicity
                ),
            )
            recheck = RoleHandleRecheck(role=role, **result)
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise UgiGeneratedTerminalSupportError(
                f"{role} exact handle qualification failed"
            ) from error
        handle_rechecks.append(recheck)

    targets = tuple(
        RouteTarget(role=role, canonical_smiles=payload.by_role()[role].canonical_smiles)
        for role in ROLE_NAMES
    )
    qualifications = tuple(
        TargetQualification(
            exact_l1_eligible=True,
            supported_ugi_role=True,
            role_handle_qualified=True,
            molecular_support_state=MolecularSupportState.WITHIN_DECLARED_SUPPORT,
        )
        for _ in ROLE_NAMES
    )
    return QualifiedGeneratedUgiTerminalSupport(
        terminal_sha256=terminal.terminal_sha256,
        generator_checkpoint_sha256=terminal.generator_checkpoint_sha256,
        product_smiles=payload.product_smiles,
        l1_reverification=l1_reverification,
        handle_rechecks=tuple(handle_rechecks),
        root_targets=targets,
        root_qualifications=qualifications,
    )


__all__ = [
    "DECLARED_GRAPH_SUPPORT_SCHEMA_VERSION",
    "DeclaredGraphSupportContext",
    "QualifiedGeneratedUgiTerminalSupport",
    "RoleHandleRecheck",
    "UgiGeneratedTerminalSupportError",
    "declared_graph_support_context_sha256",
    "qualify_locked_generated_ugi_terminal_support",
]
