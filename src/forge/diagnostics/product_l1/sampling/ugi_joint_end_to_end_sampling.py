"""Compose the matched joint sparse-flow arm into complete Ugi molecules."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from forge.corpus.ugi_generated_components import (
    UgiGeneratedComponentError,
    generated_ugi_component_smiles,
)
from forge.corpus.ugi_held_component_gate import (
    exact_forward_reconstructs_ugi_product,
)
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.ugi_adapter_features import ORIGIN_TO_INDEX
from forge.model.ugi_chemistry_flow import (
    UgiChemistryFlowError,
    UgiTerminalDecodeError,
    chemistry_sample_to_molecule,
    valence_constrained_terminal_sample,
)
from forge.model.ugi_chemistry_interface import (
    assemble_ugi_chemistry_topology_condition,
)
from forge.model.ugi_closure_placement import sample_sparse_closures
from forge.model.ugi_joint_sparse_flow import (
    UgiJointSparseTerminal,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


class UgiJointEndToEndSamplingError(RuntimeError):
    """Raised when the matched joint arm cannot be composed exactly."""


REFERENCE_COMPARISON_MODES = ("full", "deferred")
MOLECULE_FAILURE_SCHEMA = "forge.ugi_molecule_construction_failure.v1"


def _runtime_failure_detail(error: BaseException, *, stage: str) -> dict[str, Any]:
    """Return a stable subtype plus the original exception evidence."""

    message = str(error)
    lowered = message.lower()
    if "kekul" in lowered or "aromatic" in lowered:
        code = "aromaticity_or_kekulization_failure"
    elif "valence" in lowered:
        code = "atom_valence_failure"
    elif "sanitize" in lowered:
        code = "rdkit_sanitization_failure"
    elif isinstance(error, UgiChemistryFlowError):
        code = "chemistry_topology_contract_failure"
    else:
        code = "unclassified_runtime_failure"
    return {
        "schema_version": MOLECULE_FAILURE_SCHEMA,
        "stage": stage,
        "code": code,
        "exception_type": type(error).__name__,
        "message": message,
    }


def _terminal_failure_detail(error: BaseException) -> dict[str, Any]:
    if isinstance(error, UgiTerminalDecodeError):
        return error.to_mapping()
    return _runtime_failure_detail(error, stage="terminal_decode")


_PREPARED_CORPUS_CACHE: dict[tuple[str, str], Any] = {}


@dataclass(frozen=True)
class UgiJointTerminalCompletion:
    """Restartable terminal completion outputs and advanced closure RNG state."""

    conditions: tuple[Any, ...]
    samples: tuple[Any, ...]
    rows: tuple[dict[str, Any], ...]
    molecules: tuple[Any, ...]
    closure_generator_state: Any
    terminal_generator_state: Any | None


def _annotate_l1_terminal_admission(row: dict[str, Any], reaction: Any) -> None:
    """Add terminal-admission state without changing raw validity denominators."""

    row["raw_molecule_valid"] = bool(row.get("valid"))
    if not row["raw_molecule_valid"]:
        row["terminal_valid"] = False
        row["terminal_failure_type"] = row.get("failure_type") or "RawMoleculeInvalid"
        return
    if not row.get("component_reconstruction_valid"):
        row["terminal_valid"] = False
        row["terminal_failure_type"] = "L1ComponentRecoveryFailure"
        return
    exact_forward, saturated, outcome_count = exact_forward_reconstructs_ugi_product(
        reaction,
        row["component_smiles_by_role"],
        row["smiles"],
    )
    row["l1_forward_verification"] = {
        "exact_product_reconstructed": exact_forward,
        "maximum_outcomes_saturated": saturated,
        "outcome_count": outcome_count,
    }
    row["terminal_valid"] = exact_forward
    row["terminal_failure_type"] = None if exact_forward else "L1ForwardConsistencyFailure"


def complete_ugi_joint_terminals(
    model: Any,
    closure_model: Any,
    terminals: Sequence[UgiJointSparseTerminal],
    corpus: Any,
    *,
    program_metadata: Sequence[dict[str, Any]],
    closure_generator_state: Any,
    allowed_ring_sizes: Sequence[int],
    maximum_heavy_degree: int,
    l1_reaction: Any | None = None,
    terminal_decoder_mode: str = "argmax",
    terminal_generator_state: Any | None = None,
    terminal_temperature: float = 1.0,
    terminal_atom_temperature: float | None = None,
    terminal_bond_temperature: float | None = None,
    terminal_decoration_temperature: float | None = None,
    terminal_atom_temperatures_by_origin: Sequence[float] | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    program_id: str | None = None,
    local_chemistry_constraint_scope: str = "role_edges_cycles_bounds",
) -> UgiJointTerminalCompletion:
    """Complete sparse terminals into sanitized products and exact components."""

    if torch is None or len(terminals) != len(program_metadata):
        raise UgiJointEndToEndSamplingError("terminal completion inputs are misaligned")
    if local_chemistry_constraint_scope not in {
        "role_edges_only",
        "role_edges_cycles_bounds",
    }:
        raise UgiJointEndToEndSamplingError("unsupported role-local chemistry constraint scope")
    if local_chemistry_support is not None:
        if program_id is None:
            raise UgiJointEndToEndSamplingError(
                "role-local chemistry support requires an explicit program id"
            )
        if tuple(local_chemistry_support.atom_states) != tuple(corpus.atom_vocabulary):
            raise UgiJointEndToEndSamplingError(
                "role-local policy atom vocabulary differs from the sampling corpus"
            )
    elif program_id is not None:
        raise UgiJointEndToEndSamplingError(
            "a role-local chemistry program id requires a support policy"
        )
    closure_generator = torch.Generator()
    closure_generator.set_state(closure_generator_state)
    terminal_generator = None
    if terminal_decoder_mode != "argmax":
        if terminal_generator_state is None:
            raise UgiJointEndToEndSamplingError(
                "non-argmax terminal decoding requires an explicit generator state"
            )
        terminal_generator = torch.Generator()
        terminal_generator.set_state(terminal_generator_state)
    elif terminal_generator_state is not None:
        raise UgiJointEndToEndSamplingError(
            "argmax terminal decoding does not accept a generator state"
        )
    conditions = []
    samples = []
    rows = []
    for sample_index, terminal in enumerate(terminals):
        offspring_by_role = {
            role: terminal.offspring[role_index] for role_index, role in enumerate(ROLE_NAMES)
        }
        left_by_role = {}
        right_by_role = {}
        for role_index, role in enumerate(ROLE_NAMES):
            left, right = sample_sparse_closures(
                closure_model,
                terminal.offspring[role_index],
                role_index=role_index,
                cycle_rank=terminal.program.cycle_ranks[role_index],
                attachment_count=terminal.program.attachment_counts[role_index],
                generator=closure_generator,
                allowed_ring_sizes=allowed_ring_sizes,
                maximum_heavy_degree=maximum_heavy_degree,
                device="cpu",
            )
            left_by_role[role] = left
            right_by_role[role] = right
        condition = assemble_ugi_chemistry_topology_condition(
            structure_id=f"joint_generated_{sample_index:04d}",
            offspring_by_role=offspring_by_role,
            attachment_counts_by_role={
                role: terminal.program.attachment_counts[role_index]
                for role_index, role in enumerate(ROLE_NAMES)
            },
            closure_left_by_role=left_by_role,
            closure_right_by_role=right_by_role,
            schema=corpus.core_schema,
        )
        exterior_full_indices = []
        for role in ROLE_NAMES:
            exterior_full_indices.extend(
                np.flatnonzero(
                    (condition.origin_states == ORIGIN_TO_INDEX[role]) & ~condition.fixed_atom_mask
                ).tolist()
            )
        if len(exterior_full_indices) != terminal.program.node_count:
            raise UgiJointEndToEndSamplingError("joint exterior/full topology mismatch")
        sequence_by_full = {full: sequence for sequence, full in enumerate(exterior_full_indices)}
        atom_logits = torch.zeros(
            (1, condition.node_count, len(corpus.atom_vocabulary)), dtype=torch.float32
        )
        parent_logits = torch.zeros((1, condition.node_count, 4), dtype=torch.float32)
        for sequence, full in enumerate(exterior_full_indices):
            atom_logits[0, full] = torch.from_numpy(terminal.atom_logits[sequence])
            parent_logits[0, full] = torch.from_numpy(terminal.parent_bond_logits[sequence])
        closure_sequence_pairs = [
            (sequence_by_full[int(left)], sequence_by_full[int(right)])
            for left, right in zip(condition.closure_left, condition.closure_right, strict=True)
        ]
        if closure_sequence_pairs:
            hidden = torch.from_numpy(terminal.hidden)
            closure_features = torch.stack(
                [torch.cat((hidden[left], hidden[right])) for left, right in closure_sequence_pairs]
            )
            closure_logits = model.closure_bond_output(closure_features)[None, :, :]
        else:
            closure_logits = torch.zeros((1, 0, 4), dtype=torch.float32)
        decoration_anchor_logits = torch.full(
            (1, model.maximum_decorations, condition.node_count + 1),
            -torch.inf,
            dtype=torch.float32,
        )
        decoration_anchor_logits[:, :, 0] = torch.from_numpy(
            terminal.decoration_anchor_logits[:, 0]
        )
        for sequence, full in enumerate(exterior_full_indices):
            decoration_anchor_logits[0, :, full + 1] = torch.from_numpy(
                terminal.decoration_anchor_logits[:, sequence + 1]
            )
        terminal_chemistry = {
            "nodes": atom_logits,
            "parent_bonds": parent_logits,
            "closure_bonds": closure_logits,
        }
        if model.maximum_decorations == 1:
            terminal_chemistry["decoration_anchor"] = decoration_anchor_logits[:, 0, :]
        else:
            terminal_chemistry.update(
                {
                    "decoration_anchors": decoration_anchor_logits,
                    "decoration_atoms": torch.from_numpy(terminal.decoration_atom_logits)[
                        None, :, :
                    ],
                    "decoration_bonds": torch.from_numpy(terminal.decoration_bond_logits)[
                        None, :, :
                    ],
                }
            )
        decode_failure: dict[str, Any] | None = None
        try:
            decoded = valence_constrained_terminal_sample(
                condition,
                terminal_chemistry,
                0,
                corpus.atom_vocabulary,
                model.maximum_decorations,
                mode=terminal_decoder_mode,
                generator=terminal_generator,
                temperature=terminal_temperature,
                atom_temperature=terminal_atom_temperature,
                bond_temperature=terminal_bond_temperature,
                decoration_temperature=terminal_decoration_temperature,
                atom_temperatures_by_origin=terminal_atom_temperatures_by_origin,
                local_chemistry_support=local_chemistry_support,
                program_id=program_id,
                local_chemistry_constraint_scope=local_chemistry_constraint_scope,
            )
        except RuntimeError as error:
            decoded = None
            decode_failure = _terminal_failure_detail(error)
        conditions.append(condition)
        samples.append(decoded)
        rows.append(
            {
                "structure_id": condition.structure_id,
                **program_metadata[sample_index],
                "program": terminal.program.__dict__,
                "offspring_by_role": {
                    role: offspring_by_role[role].tolist() for role in ROLE_NAMES
                },
                "terminal_decoder_mode": terminal_decoder_mode,
                "terminal_temperature": terminal_temperature,
                "terminal_atom_temperature": terminal_atom_temperature,
                "terminal_bond_temperature": terminal_bond_temperature,
                "terminal_decoration_temperature": terminal_decoration_temperature,
                "terminal_atom_temperatures_by_origin": (
                    list(terminal_atom_temperatures_by_origin)
                    if terminal_atom_temperatures_by_origin is not None
                    else None
                ),
                "terminal_failure_detail": decode_failure,
                "molecule_failure_detail": None,
            }
        )
    molecules = []
    for row, condition, sample in zip(rows, conditions, samples, strict=True):
        if sample is None:
            row["smiles"] = None
            row["valid"] = False
            row["failure_type"] = "TerminalSupportFailure"
            row["component_smiles_by_role"] = None
            row["component_reconstruction_valid"] = False
            row["component_reconstruction_error"] = "terminal support failed"
            if l1_reaction is not None:
                _annotate_l1_terminal_admission(row, l1_reaction)
            continue
        try:
            molecule = chemistry_sample_to_molecule(
                condition,
                sample,
                corpus.atom_vocabulary,
            )
            row["smiles"] = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            row["valid"] = True
            row["failure_type"] = None
            try:
                row["component_smiles_by_role"] = generated_ugi_component_smiles(
                    condition,
                    sample,
                    corpus.atom_vocabulary,
                )
                row["component_reconstruction_valid"] = True
                row["component_reconstruction_error"] = None
            except UgiGeneratedComponentError as error:
                row["component_smiles_by_role"] = None
                row["component_reconstruction_valid"] = False
                row["component_reconstruction_error"] = str(error)
            if l1_reaction is not None:
                _annotate_l1_terminal_admission(row, l1_reaction)
            molecules.append(molecule)
        except (ValueError, RuntimeError) as error:
            row["smiles"] = None
            row["valid"] = False
            row["failure_type"] = "MoleculeSanitizationFailure"
            row["molecule_failure_detail"] = _runtime_failure_detail(
                error, stage="molecule_construction"
            )
            row["component_smiles_by_role"] = None
            row["component_reconstruction_valid"] = False
            row["component_reconstruction_error"] = "product molecule is invalid"
            if l1_reaction is not None:
                _annotate_l1_terminal_admission(row, l1_reaction)
    return UgiJointTerminalCompletion(
        conditions=tuple(conditions),
        samples=tuple(samples),
        rows=tuple(rows),
        molecules=tuple(molecules),
        closure_generator_state=closure_generator.get_state().clone(),
        terminal_generator_state=(
            terminal_generator.get_state().clone() if terminal_generator is not None else None
        ),
    )
