"""Seal native restartable-generator outputs for route-support qualification.

The selected sparse generator already emits every field required by the
frozen candidate-eligibility contract.  This module preserves those native
fields at the matched-generation boundary, binds them to the exact schedule
program and checkpoints, independently constructs the exact-L1 payload, and
then invokes the typed generated-terminal support seam.  It never infers a
program, offspring word, product, or precursor identity from similarity or a
post-hoc molecular decomposition.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from forge.corpus.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    QualifiedGeneratedUgiTerminalSupport,
    UgiGeneratedTerminalSupportError,
    qualify_locked_generated_ugi_terminal_support,
)
from forge.model.ugi_morphology_program import UgiMorphologyProgram
from forge.potency.annotations import ROLE_NAMES
from forge.synthesis.matched import (
    LockedMatchedTerminal,
    MatchedGenerationRequest,
)
from forge.synthesis.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    UgiTerminalRouteAssessmentError,
    ValidatedUgiTerminalPayload,
)

MORPHOLOGY_PROGRAM_SCHEMA_VERSION = "forge.ugi_morphology_program_bytes.v1"
RESTARTABLE_TERMINAL_TRACE_SCHEMA_VERSION = "forge.ugi_restartable_terminal_trace.v2"
UNQUALIFIED_TERMINAL_SCHEMA_VERSION = "forge.ugi_restartable_unqualified_terminal.v1"
_CANDIDATE_FIELDS = (
    "valid",
    "component_reconstruction_valid",
    "smiles",
    "component_smiles_by_role",
    "program",
    "offspring_by_role",
)


class UgiRestartableTerminalSupportAdapterError(RuntimeError):
    """Raised when a native restartable output cannot be sealed exactly."""


def _stable_json_bytes(value: Any) -> bytes:
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    except (TypeError, ValueError) as error:
        raise UgiRestartableTerminalSupportAdapterError(
            "restartable terminal record is not canonically serializable"
        ) from error


def _integer_triplet(value: Any, *, label: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != len(ROLE_NAMES)
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            f"{label} must contain three integer role values"
        )
    return tuple(value)  # type: ignore[return-value]


def _canonical_program_dict(value: Any) -> dict[str, list[int]]:
    if isinstance(value, UgiMorphologyProgram):
        program = value
    elif isinstance(value, Mapping):
        expected = {
            "node_counts",
            "junction_budgets",
            "cycle_ranks",
            "attachment_counts",
        }
        if set(value) != expected:
            raise UgiRestartableTerminalSupportAdapterError(
                "native morphology program has an unsupported field set"
            )
        try:
            program = UgiMorphologyProgram(
                node_counts=_integer_triplet(value["node_counts"], label="node_counts"),
                junction_budgets=_integer_triplet(
                    value["junction_budgets"], label="junction_budgets"
                ),
                cycle_ranks=_integer_triplet(value["cycle_ranks"], label="cycle_ranks"),
                attachment_counts=_integer_triplet(
                    value["attachment_counts"], label="attachment_counts"
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise UgiRestartableTerminalSupportAdapterError(
                "native morphology program is malformed"
            ) from error
    else:
        raise UgiRestartableTerminalSupportAdapterError(
            "native morphology program must be a UgiMorphologyProgram or mapping"
        )
    fields = {
        "node_counts": list(program.node_counts),
        "junction_budgets": list(program.junction_budgets),
        "cycle_ranks": list(program.cycle_ranks),
        "attachment_counts": list(program.attachment_counts),
    }
    if (
        any(item < 1 for item in program.node_counts)
        or any(item < 0 for item in program.junction_budgets)
        or any(item < 0 for item in program.cycle_ranks)
        or any(item < 1 for item in program.attachment_counts)
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "native morphology program contains invalid count values"
        )
    return fields


def canonical_morphology_program_bytes(value: Any) -> bytes:
    """Serialize one exact native program for matched-schedule binding."""

    return _stable_json_bytes(
        {
            "schema_version": MORPHOLOGY_PROGRAM_SCHEMA_VERSION,
            "program": _canonical_program_dict(value),
        }
    )


def decode_canonical_morphology_program_bytes(payload: bytes) -> UgiMorphologyProgram:
    """Decode one schedule program only when its byte representation is canonical."""

    if not isinstance(payload, bytes) or not payload:
        raise UgiRestartableTerminalSupportAdapterError(
            "morphology-program payload must be nonempty bytes"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiRestartableTerminalSupportAdapterError(
            "morphology-program payload is invalid JSON"
        ) from error
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "program"}
        or value.get("schema_version") != MORPHOLOGY_PROGRAM_SCHEMA_VERSION
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "morphology-program payload has an unsupported schema"
        )
    program_fields = _canonical_program_dict(value.get("program"))
    program = UgiMorphologyProgram(
        node_counts=tuple(program_fields["node_counts"]),  # type: ignore[arg-type]
        junction_budgets=tuple(program_fields["junction_budgets"]),  # type: ignore[arg-type]
        cycle_ranks=tuple(program_fields["cycle_ranks"]),  # type: ignore[arg-type]
        attachment_counts=tuple(program_fields["attachment_counts"]),  # type: ignore[arg-type]
    )
    if payload != canonical_morphology_program_bytes(program):
        raise UgiRestartableTerminalSupportAdapterError(
            "morphology-program payload is not canonically encoded"
        )
    return program


def native_completion_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact assessed native completion fields without promotion."""

    if not isinstance(row, Mapping):
        raise UgiRestartableTerminalSupportAdapterError(
            "restartable completion row must be a mapping"
        )
    missing = tuple(field for field in _CANDIDATE_FIELDS if field not in row)
    if missing:
        raise UgiRestartableTerminalSupportAdapterError(
            f"restartable completion row is missing native fields: {', '.join(missing)}"
        )
    if not isinstance(row["valid"], bool) or not isinstance(
        row["component_reconstruction_valid"], bool
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "native validity fields must be assessed booleans"
        )
    if row["valid"] and (not isinstance(row["smiles"], str) or not row["smiles"]):
        raise UgiRestartableTerminalSupportAdapterError(
            "a valid native completion must contain a nonempty product identity"
        )
    if not row["valid"] and row["smiles"] is not None:
        raise UgiRestartableTerminalSupportAdapterError(
            "an invalid native completion must not claim a product identity"
        )
    if row["component_reconstruction_valid"] and not row["valid"]:
        raise UgiRestartableTerminalSupportAdapterError(
            "an invalid native completion cannot claim component reconstruction"
        )
    components = row["component_smiles_by_role"]
    if row["component_reconstruction_valid"]:
        if not isinstance(components, Mapping) or set(components) != set(ROLE_NAMES):
            raise UgiRestartableTerminalSupportAdapterError(
                "native component identities must contain exactly the three Ugi roles"
            )
        if any(
            not isinstance(components[role], str) or not components[role] for role in ROLE_NAMES
        ):
            raise UgiRestartableTerminalSupportAdapterError(
                "native component identities must be nonempty SMILES strings"
            )
        canonical_components: dict[str, str] | None = {
            role: components[role] for role in ROLE_NAMES
        }
    else:
        if components is not None:
            raise UgiRestartableTerminalSupportAdapterError(
                "a failed component reconstruction must not claim component identities"
            )
        canonical_components = None
    program = _canonical_program_dict(row["program"])
    offspring = row["offspring_by_role"]
    if not isinstance(offspring, Mapping) or set(offspring) != set(ROLE_NAMES):
        raise UgiRestartableTerminalSupportAdapterError(
            "native offspring words must contain exactly the three Ugi roles"
        )
    canonical_offspring: dict[str, list[int]] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        values = offspring[role]
        if (
            not isinstance(values, (list, tuple))
            or len(values) != program["node_counts"][role_index]
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in values
            )
        ):
            raise UgiRestartableTerminalSupportAdapterError(
                f"native {role} offspring word is missing or malformed"
            )
        canonical_offspring[role] = list(values)
    return {
        "valid": row["valid"],
        "component_reconstruction_valid": row["component_reconstruction_valid"],
        "smiles": row["smiles"],
        "component_smiles_by_role": canonical_components,
        "program": program,
        "offspring_by_role": canonical_offspring,
    }


def native_candidate_eligibility_record(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract native fields only when the frozen eligibility inputs exist."""

    record = native_completion_record(row)
    if record["valid"] is not True:
        raise UgiRestartableTerminalSupportAdapterError(
            "native candidate eligibility requires a valid completed molecule"
        )
    if record["component_reconstruction_valid"] is not True:
        raise UgiRestartableTerminalSupportAdapterError(
            "native candidate eligibility requires exact component reconstruction"
        )
    return record


def native_completion_record_bytes(row: Mapping[str, Any]) -> bytes:
    """Return canonical bytes for valid or invalid assessed native output fields."""

    return _stable_json_bytes(native_completion_record(row))


def _restartable_terminal_trace_bytes(
    candidate: Mapping[str, Any],
    *,
    generation_request: MatchedGenerationRequest,
) -> bytes:
    candidate_bytes = _stable_json_bytes(candidate)
    entry = generation_request.entry
    return _stable_json_bytes(
        {
            "schema_version": RESTARTABLE_TERMINAL_TRACE_SCHEMA_VERSION,
            "unit_id": entry.unit_id,
            "program_index": entry.program_index,
            "particle_index": entry.particle_index,
            "checkpoint_index": entry.checkpoint_index,
            "rollout_index": entry.rollout_index,
            "productive_seed": generation_request.productive_seed,
            "morphology_program_sha256": entry.morphology_program_sha256,
            "generator_checkpoint_sha256": entry.generator_checkpoint_sha256,
            "closure_checkpoint_sha256": entry.closure_checkpoint_sha256,
            "candidate_record": dict(candidate),
            "candidate_record_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        }
    )


def native_completion_record_from_locked_terminal(
    terminal: LockedMatchedTerminal,
) -> dict[str, Any]:
    """Resolve and authenticate the native completion owned by a locked trace."""

    if not isinstance(terminal, LockedMatchedTerminal) or not terminal.terminal_locked:
        raise UgiRestartableTerminalSupportAdapterError(
            "native completion resolution requires a locked terminal"
        )
    try:
        trace = json.loads(terminal.generation_trace_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiRestartableTerminalSupportAdapterError(
            "locked generation trace is invalid JSON"
        ) from error
    expected = {
        "schema_version",
        "unit_id",
        "program_index",
        "particle_index",
        "checkpoint_index",
        "rollout_index",
        "productive_seed",
        "morphology_program_sha256",
        "generator_checkpoint_sha256",
        "closure_checkpoint_sha256",
        "candidate_record",
        "candidate_record_sha256",
    }
    if (
        not isinstance(trace, Mapping)
        or set(trace) != expected
        or trace.get("schema_version") != RESTARTABLE_TERMINAL_TRACE_SCHEMA_VERSION
        or terminal.generation_trace_bytes != _stable_json_bytes(trace)
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "locked generation trace has an unsupported or noncanonical schema"
        )
    if (
        trace["unit_id"] != terminal.unit_id
        or trace["checkpoint_index"] != terminal.checkpoint_index
        or trace["morphology_program_sha256"] != terminal.morphology_program_sha256
        or trace["generator_checkpoint_sha256"] != terminal.generator_checkpoint_sha256
        or trace["closure_checkpoint_sha256"] != terminal.closure_checkpoint_sha256
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "locked generation trace disagrees with terminal identity"
        )
    raw_candidate = trace.get("candidate_record")
    if not isinstance(raw_candidate, Mapping) or set(raw_candidate) != set(_CANDIDATE_FIELDS):
        raise UgiRestartableTerminalSupportAdapterError(
            "locked generation trace candidate record has an unsupported schema"
        )
    candidate = native_completion_record(raw_candidate)
    candidate_bytes = _stable_json_bytes(candidate)
    if hashlib.sha256(candidate_bytes).hexdigest() != trace["candidate_record_sha256"]:
        raise UgiRestartableTerminalSupportAdapterError(
            "locked generation trace candidate-record checksum mismatch"
        )
    if (
        hashlib.sha256(canonical_morphology_program_bytes(candidate["program"])).hexdigest()
        != terminal.morphology_program_sha256
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "locked generation trace candidate program differs from the scheduled program"
        )
    if (
        terminal.terminal_id != f"{terminal.unit_id}:{trace['productive_seed']}"
        or terminal.terminal_valid is not candidate["valid"]
        or (terminal.exact_l1 and candidate["component_reconstruction_valid"] is not True)
    ):
        raise UgiRestartableTerminalSupportAdapterError(
            "locked terminal admission state disagrees with its native completion record"
        )
    return candidate


def lock_unqualified_restartable_completion_row(
    row: Mapping[str, Any],
    *,
    generation_request: MatchedGenerationRequest,
    nonqualification_reason: str,
) -> LockedMatchedTerminal:
    """Seal a native invalid or nonexact completion so routing skips it."""

    if not isinstance(generation_request, MatchedGenerationRequest):
        raise UgiRestartableTerminalSupportAdapterError(
            "generation_request must be a MatchedGenerationRequest"
        )
    candidate = native_completion_record(row)
    expected_reason = (
        "native_molecule_invalid"
        if candidate["valid"] is not True
        else (
            "component_reconstruction_failed"
            if candidate["component_reconstruction_valid"] is not True
            else "independent_l1_forward_failed"
        )
    )
    if nonqualification_reason != expected_reason:
        raise UgiRestartableTerminalSupportAdapterError(
            "unqualified terminal reason disagrees with the assessed native completion"
        )
    program_bytes = canonical_morphology_program_bytes(candidate["program"])
    entry = generation_request.entry
    if entry.morphology_program != program_bytes:
        raise UgiRestartableTerminalSupportAdapterError(
            "matched schedule morphology bytes differ from the native completed program"
        )
    candidate_bytes = _stable_json_bytes(candidate)
    terminal_bytes = _stable_json_bytes(
        {
            "schema_version": UNQUALIFIED_TERMINAL_SCHEMA_VERSION,
            "candidate_record_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
            "terminal_valid": candidate["valid"],
            "exact_l1": False,
            "nonqualification_reason": nonqualification_reason,
        }
    )
    return LockedMatchedTerminal(
        unit_id=entry.unit_id,
        morphology_program_sha256=entry.morphology_program_sha256,
        checkpoint_index=entry.checkpoint_index,
        generator_checkpoint_sha256=entry.generator_checkpoint_sha256,
        closure_checkpoint_sha256=entry.closure_checkpoint_sha256,
        terminal_id=f"{entry.unit_id}:{generation_request.productive_seed}",
        terminal_locked=True,
        terminal_valid=candidate["valid"],
        exact_l1=False,
        terminal_bytes=terminal_bytes,
        generation_trace_bytes=_restartable_terminal_trace_bytes(
            candidate,
            generation_request=generation_request,
        ),
        payload=None,
    )


@dataclass(frozen=True)
class AdaptedRestartableGeneratedTerminal:
    """One sealed matched terminal plus its exact route-root qualifications."""

    locked_terminal: LockedMatchedTerminal
    candidate_record_bytes: bytes
    support: QualifiedGeneratedUgiTerminalSupport

    def __post_init__(self) -> None:
        if not isinstance(self.locked_terminal, LockedMatchedTerminal):
            raise UgiRestartableTerminalSupportAdapterError(
                "adapted terminal requires a LockedMatchedTerminal"
            )
        if not isinstance(self.candidate_record_bytes, bytes) or not self.candidate_record_bytes:
            raise UgiRestartableTerminalSupportAdapterError(
                "adapted terminal requires canonical candidate-record bytes"
            )
        if not isinstance(self.support, QualifiedGeneratedUgiTerminalSupport):
            raise UgiRestartableTerminalSupportAdapterError(
                "adapted terminal requires typed support qualifications"
            )
        if self.support.terminal_sha256 != self.locked_terminal.terminal_sha256:
            raise UgiRestartableTerminalSupportAdapterError(
                "support qualifications are bound to a different terminal"
            )
        if self.candidate_record_bytes != _stable_json_bytes(self.candidate_record):
            raise UgiRestartableTerminalSupportAdapterError(
                "candidate-record bytes are not canonical"
            )

    @property
    def candidate_record(self) -> dict[str, Any]:
        """Decode the immutable native record for the support-boundary seam."""

        try:
            value = json.loads(self.candidate_record_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise UgiRestartableTerminalSupportAdapterError(
                "candidate-record bytes are invalid JSON"
            ) from error
        if not isinstance(value, dict):
            raise UgiRestartableTerminalSupportAdapterError(
                "candidate-record bytes must decode to an object"
            )
        return value


def adapt_restartable_completion_row_for_route_support(
    row: Mapping[str, Any],
    *,
    generation_request: MatchedGenerationRequest,
    l1_reaction: Any,
    l1_reaction_sha256: str,
    component_recovery_contract_sha256: str,
    graph_support: DeclaredGraphSupportContext,
    l1_reverifier: QualifiedUgiL1Reverifier,
    maximum_outcomes: int = 64,
) -> AdaptedRestartableGeneratedTerminal:
    """Seal one native completion row and derive its route-support qualifications."""

    if not isinstance(generation_request, MatchedGenerationRequest):
        raise UgiRestartableTerminalSupportAdapterError(
            "generation_request must be a MatchedGenerationRequest"
        )
    candidate = native_candidate_eligibility_record(row)
    candidate_bytes = _stable_json_bytes(candidate)
    program_bytes = canonical_morphology_program_bytes(candidate["program"])
    entry = generation_request.entry
    if entry.morphology_program != program_bytes:
        raise UgiRestartableTerminalSupportAdapterError(
            "matched schedule morphology bytes differ from the native completed program"
        )
    if entry.generator_checkpoint_sha256 != graph_support.generator_checkpoint_sha256:
        raise UgiRestartableTerminalSupportAdapterError(
            "matched schedule and graph-support context use different generator checkpoints"
        )
    try:
        payload = ValidatedUgiTerminalPayload.from_recovered_components(
            product_smiles=candidate["smiles"],
            components_by_role=candidate["component_smiles_by_role"],
            l1_reaction=l1_reaction,
            l1_reaction_sha256=l1_reaction_sha256,
            component_recovery_contract_sha256=component_recovery_contract_sha256,
            maximum_outcomes=maximum_outcomes,
        )
    except UgiTerminalRouteAssessmentError as error:
        raise UgiRestartableTerminalSupportAdapterError(
            "native product and components failed independent exact-L1 payload construction"
        ) from error
    trace_bytes = _restartable_terminal_trace_bytes(
        candidate,
        generation_request=generation_request,
    )
    terminal = LockedMatchedTerminal(
        unit_id=entry.unit_id,
        morphology_program_sha256=entry.morphology_program_sha256,
        checkpoint_index=entry.checkpoint_index,
        generator_checkpoint_sha256=entry.generator_checkpoint_sha256,
        closure_checkpoint_sha256=entry.closure_checkpoint_sha256,
        terminal_id=f"{entry.unit_id}:{generation_request.productive_seed}",
        terminal_locked=True,
        terminal_valid=True,
        exact_l1=True,
        terminal_bytes=payload.canonical_bytes,
        generation_trace_bytes=trace_bytes,
        payload=payload,
    )
    try:
        support = qualify_locked_generated_ugi_terminal_support(
            terminal,
            candidate_record=candidate,
            graph_support=graph_support,
            l1_reverifier=l1_reverifier,
        )
    except UgiGeneratedTerminalSupportError as error:
        raise UgiRestartableTerminalSupportAdapterError(
            "native restartable terminal failed route-support qualification"
        ) from error
    return AdaptedRestartableGeneratedTerminal(
        locked_terminal=terminal,
        candidate_record_bytes=candidate_bytes,
        support=support,
    )


__all__ = [
    "AdaptedRestartableGeneratedTerminal",
    "MORPHOLOGY_PROGRAM_SCHEMA_VERSION",
    "RESTARTABLE_TERMINAL_TRACE_SCHEMA_VERSION",
    "UNQUALIFIED_TERMINAL_SCHEMA_VERSION",
    "UgiRestartableTerminalSupportAdapterError",
    "adapt_restartable_completion_row_for_route_support",
    "canonical_morphology_program_bytes",
    "decode_canonical_morphology_program_bytes",
    "lock_unqualified_restartable_completion_row",
    "native_candidate_eligibility_record",
    "native_completion_record",
    "native_completion_record_bytes",
    "native_completion_record_from_locked_terminal",
]
