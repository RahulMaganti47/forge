"""Hash-bound production callback for the selected step-1000 Ugi generator.

This module is deliberately a thin composition layer.  It authenticates the
frozen product-plus-L1 generator, its sparse-closure checkpoint, its exact
training cache and atom vocabulary, the qualified Ugi reaction, the declared
graph-support context, the generated-component recovery implementation, and
the restartable-sampler equivalence receipt.  A callback then executes one
and only one native eight-step completion for the schedule's canonical
morphology program and productive seed before sealing the native completion
row through :mod:`ugi_restartable_terminal_support_adapter`.

It performs no route search, synthesis-value calculation, biological scoring,
candidate selection, repair, retry, or holdout inspection.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from forge.corpus.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    UgiGeneratedTerminalSupportError,
    declared_graph_support_context_sha256,
    qualify_locked_generated_ugi_terminal_support,
)
from forge.diagnostics.product_l1.sampling.ugi_joint_end_to_end_sampling import (
    complete_ugi_joint_terminals,
)
from forge.diagnostics.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals,
)
from forge.diagnostics.product_l1.sampling.ugi_selected_generator_implementation import (
    SelectedGeneratorImplementationQualification,
    require_selected_generator_implementation_unchanged,
)
from forge.diagnostics.support.adapters.terminal_support import (
    AdaptedRestartableGeneratedTerminal,
    UgiRestartableTerminalSupportAdapterError,
    adapt_restartable_completion_row_for_route_support,
    decode_canonical_morphology_program_bytes,
    lock_unqualified_restartable_completion_row,
    native_completion_record,
    native_completion_record_bytes,
    native_completion_record_from_locked_terminal,
)
from forge.diagnostics.support.guidance.ugi_zero_guidance_rehearsal import (
    RestartableGeneratorClosureAdapter,
)
from forge.model.networks.dense_flow import sha256_file
from forge.synthesis.matched import (
    LockedMatchedTerminal,
    MatchedGenerationRequest,
)
from forge.synthesis.terminals.terminal_assessment import (
    QualifiedUgiL1Reverifier,
    UgiTerminalRouteAssessmentError,
    ValidatedUgiTerminalPayload,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - product sampling requires torch
    torch = None


SELECTED_RESTARTABLE_GENERATOR_SCHEMA_VERSION = "forge.selected_step1000_restartable_generator.v1"
SAMPLE_STEPS = 8
TERMINAL_DECODER_ID = "ugi_joint_terminal_completion:argmax:v1"

GENERATOR_CHECKPOINT_SHA256 = "90f5f0bd3e41e2884f6588d9875db0b7bf34e5c5ea7fdc1f9d8565fba8c0c532"
CLOSURE_CHECKPOINT_SHA256 = "a97507ac6a9eeba41d0cc351666cffdd21069d13bc78c0db671ab3a30c710b5d"
PRODUCTION_GENERATOR_MANIFEST_SHA256 = (
    "c27352c11a2e210fd76a6e4ae510bbbb89e51c92c28514fd9b4693f82a50e68a"
)
TRAINING_CACHE_SHA256 = "b862b7a54c0cb325f0a962fea42a78138df1a81c88b9c0e86d56b6448e519c46"
ATOM_VOCABULARY_SHA256 = "90ab7430354a9ffd91cfd6e19f2a125020a6547c67cd462a11f77074fd2a0cc5"
QUALIFIED_REACTION_REGISTRY_SHA256 = (
    "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf"
)
L1_REACTION_SHA256 = "5b97e062b115fcc137b4a05d8f72b74e9cc67a584c05c054ef5f983969bf1427"
MODEL_CONFIG_SHA256 = "f3279a9b64aac4dfae7372c784a863fb93f25cfb4a0f1b5cb1ceb8c4c4a77f52"
DECLARED_GRAPH_SUPPORT_SHA256 = "331f40d5bf1ffe61688def4d9c971d79ae7607e6139bcc3ceb355871e097f505"
COMPONENT_RECOVERY_CONTRACT_SHA256 = (
    "14a84998891a814db287ed808d650a7c8e79175f85ee1508645fa5de0e10b944"
)


class UgiSelectedRestartableGeneratorError(RuntimeError):
    """Raised when the selected productive generator lane cannot fail closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "selected generator binding is not canonically serializable"
        ) from error


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _require_file_hash(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise UgiSelectedRestartableGeneratorError(f"missing {label}: {path}")
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise UgiSelectedRestartableGeneratorError(
            f"{label} hash changed: expected {expected_sha256}, observed {observed}"
        )


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiSelectedRestartableGeneratorError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise UgiSelectedRestartableGeneratorError(f"{label} must contain one JSON object")
    return value


def _atom_vocabulary_states(path: Path) -> frozenset[tuple[str, int, bool, int]]:
    value = _read_json(path, label="atom vocabulary")
    rows = value.get("atom_vocabulary")
    if not isinstance(rows, list) or not rows:
        raise UgiSelectedRestartableGeneratorError("atom vocabulary is empty or malformed")
    output: set[tuple[str, int, bool, int]] = set()
    try:
        for row in rows:
            if not isinstance(row, Mapping):
                raise TypeError
            state = (
                str(row["symbol"]),
                int(row["formal_charge"]),
                bool(row["aromatic"]),
                int(row["explicit_hydrogens"]),
            )
            if not state[0] or state[3] < 0:
                raise ValueError
            output.add(state)
    except (KeyError, TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError("atom vocabulary is malformed") from error
    if len(output) != len(rows):
        raise UgiSelectedRestartableGeneratorError(
            "atom vocabulary contains duplicate constitutional atom states"
        )
    return frozenset(output)


@dataclass(frozen=True)
class SelectedRestartableGeneratorBindings:
    """Content identities rechecked before model construction."""

    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    production_generator_manifest_sha256: str
    restartable_equivalence_receipt_sha256: str
    generator_implementation_sha256: str
    training_cache_sha256: str
    atom_vocabulary_sha256: str
    qualified_reaction_registry_sha256: str
    l1_reaction_sha256: str
    model_config_sha256: str
    declared_graph_support_sha256: str
    component_recovery_contract_sha256: str
    sample_steps: int
    terminal_decoder_id: str

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": SELECTED_RESTARTABLE_GENERATOR_SCHEMA_VERSION,
                **self.__dict__,
            }
        )


def _corpus_atom_states(corpus: Any) -> frozenset[tuple[str, int, bool, int]]:
    try:
        return frozenset(
            (
                str(state.symbol),
                int(state.formal_charge),
                bool(state.aromatic),
                int(state.explicit_hydrogens),
            )
            for state in corpus.atom_vocabulary
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "prepared cache contains a malformed atom vocabulary"
        ) from error


@dataclass
class SelectedRestartableGeneratorCallback:
    """One callable native completion lane with retained support-bound records."""

    model: Any
    closure_model: Any
    source_marginals: dict[str, np.ndarray]
    corpus: Any
    reaction: Any
    graph_support: DeclaredGraphSupportContext
    l1_reverifier: QualifiedUgiL1Reverifier
    bindings: SelectedRestartableGeneratorBindings
    allowed_ring_sizes: tuple[int, ...]
    maximum_heavy_degree: int
    repository: Path
    implementation_qualification: SelectedGeneratorImplementationQualification

    def __post_init__(self) -> None:
        if not isinstance(self.repository, Path) or not self.repository.is_dir():
            raise UgiSelectedRestartableGeneratorError(
                "selected generator repository must be a directory"
            )
        if not isinstance(
            self.implementation_qualification,
            SelectedGeneratorImplementationQualification,
        ):
            raise UgiSelectedRestartableGeneratorError(
                "selected generator implementation qualification is malformed"
            )
        if self.bindings.generator_implementation_sha256 != (
            self.implementation_qualification.implementation_sha256
        ):
            raise UgiSelectedRestartableGeneratorError(
                "generator bindings and implementation qualification differ"
            )

    def __call__(self, request: MatchedGenerationRequest) -> LockedMatchedTerminal:
        if torch is None:
            raise UgiSelectedRestartableGeneratorError(
                "selected productive generation requires torch"
            )
        if not isinstance(request, MatchedGenerationRequest):
            raise UgiSelectedRestartableGeneratorError(
                "productive request must be a MatchedGenerationRequest"
            )
        require_selected_generator_implementation_unchanged(
            self.repository,
            self.implementation_qualification,
        )
        entry = request.entry
        if entry.generator_checkpoint_sha256 != self.bindings.generator_checkpoint_sha256:
            raise UgiSelectedRestartableGeneratorError(
                "schedule generator checkpoint differs from the selected checkpoint"
            )
        if entry.closure_checkpoint_sha256 != self.bindings.closure_checkpoint_sha256:
            raise UgiSelectedRestartableGeneratorError(
                "schedule closure checkpoint differs from the selected closure checkpoint"
            )
        if entry.productive_generation_calls != 1:
            raise UgiSelectedRestartableGeneratorError(
                "one scheduled unit must reserve exactly one productive generation call"
            )
        if (
            isinstance(request.productive_seed, bool)
            or not isinstance(request.productive_seed, int)
            or request.productive_seed < 0
        ):
            raise UgiSelectedRestartableGeneratorError(
                "productive seed must be a nonnegative integer"
            )
        try:
            program = decode_canonical_morphology_program_bytes(entry.morphology_program)
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                "schedule morphology program is not canonical"
            ) from error

        terminals, metadata = sample_restartable_terminals(
            self.model,
            (program,),
            self.source_marginals,
            sample_steps=SAMPLE_STEPS,
            batch_size=1,
            seed=request.productive_seed,
            device="cpu",
            allowed_ring_sizes=self.allowed_ring_sizes,
            maximum_heavy_degree=self.maximum_heavy_degree,
            maximum_adjacent_branch_runs=(None, None, None),
        )
        if (
            len(terminals) != 1
            or metadata.get("sample_steps") != SAMPLE_STEPS
            or metadata.get("terminal_tree_repairs") != 0
            or metadata.get("samples") != 1
        ):
            raise UgiSelectedRestartableGeneratorError(
                "native restartable sampler violated the one-terminal eight-step contract"
            )
        completion = complete_ugi_joint_terminals(
            self.model,
            self.closure_model,
            terminals,
            self.corpus,
            program_metadata=({},),
            closure_generator_state=(
                torch.Generator().manual_seed(request.productive_seed + 1).get_state()
            ),
            allowed_ring_sizes=self.allowed_ring_sizes,
            maximum_heavy_degree=self.maximum_heavy_degree,
            l1_reaction=self.reaction,
            terminal_decoder_mode="argmax",
            terminal_generator_state=None,
            terminal_temperature=1.0,
        )
        if len(completion.rows) != 1:
            raise UgiSelectedRestartableGeneratorError(
                "native closure/chemistry completion did not return exactly one row"
            )
        row = completion.rows[0]
        try:
            native = native_completion_record(row)
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                f"native terminal for {entry.unit_id} is not an assessed completion row"
            ) from error
        if native["valid"] is not True:
            return lock_unqualified_restartable_completion_row(
                native,
                generation_request=request,
                nonqualification_reason="native_molecule_invalid",
            )
        if native["component_reconstruction_valid"] is not True:
            return lock_unqualified_restartable_completion_row(
                native,
                generation_request=request,
                nonqualification_reason="component_reconstruction_failed",
            )
        try:
            ValidatedUgiTerminalPayload.from_recovered_components(
                product_smiles=native["smiles"],
                components_by_role=native["component_smiles_by_role"],
                l1_reaction=self.reaction,
                l1_reaction_sha256=self.bindings.l1_reaction_sha256,
                component_recovery_contract_sha256=(
                    self.bindings.component_recovery_contract_sha256
                ),
            )
        except UgiTerminalRouteAssessmentError:
            return lock_unqualified_restartable_completion_row(
                native,
                generation_request=request,
                nonqualification_reason="independent_l1_forward_failed",
            )
        try:
            return adapt_restartable_completion_row_for_route_support(
                native,
                generation_request=request,
                l1_reaction=self.reaction,
                l1_reaction_sha256=self.bindings.l1_reaction_sha256,
                component_recovery_contract_sha256=(
                    self.bindings.component_recovery_contract_sha256
                ),
                graph_support=self.graph_support,
                l1_reverifier=self.l1_reverifier,
            ).locked_terminal
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                f"native exact-L1 terminal for {entry.unit_id} violated support; no retry was attempted"
            ) from error

    def adapted_for_terminal(
        self,
        terminal: LockedMatchedTerminal,
    ) -> AdaptedRestartableGeneratedTerminal:
        """Recreate support from the immutable trace, without a side registry."""

        if not isinstance(terminal, LockedMatchedTerminal):
            raise UgiSelectedRestartableGeneratorError(
                "support lookup requires a LockedMatchedTerminal"
            )
        if (
            terminal.generator_checkpoint_sha256 != self.bindings.generator_checkpoint_sha256
            or terminal.closure_checkpoint_sha256 != self.bindings.closure_checkpoint_sha256
        ):
            raise UgiSelectedRestartableGeneratorError(
                "terminal checkpoint identity differs from this callback lane"
            )
        if not terminal.terminal_valid or not terminal.exact_l1:
            raise UgiSelectedRestartableGeneratorError(
                "invalid or nonexact terminals have no route-root support"
            )
        try:
            candidate = native_completion_record_from_locked_terminal(terminal)
            support = qualify_locked_generated_ugi_terminal_support(
                terminal,
                candidate_record=candidate,
                graph_support=self.graph_support,
                l1_reverifier=self.l1_reverifier,
            )
        except (
            UgiGeneratedTerminalSupportError,
            UgiRestartableTerminalSupportAdapterError,
        ) as error:
            raise UgiSelectedRestartableGeneratorError(
                "locked terminal failed immutable native-record support resolution"
            ) from error
        return AdaptedRestartableGeneratedTerminal(
            locked_terminal=terminal,
            candidate_record_bytes=native_completion_record_bytes(candidate),
            support=support,
        )


@dataclass(frozen=True)
class SelectedRestartableGeneratorLane:
    """Factory product exposed to zero-guidance orchestration and route seams."""

    adapter: RestartableGeneratorClosureAdapter
    callback: SelectedRestartableGeneratorCallback
    bindings: SelectedRestartableGeneratorBindings
    graph_support: DeclaredGraphSupportContext
    implementation_qualification: SelectedGeneratorImplementationQualification


__all__ = [
    "ATOM_VOCABULARY_SHA256",
    "CLOSURE_CHECKPOINT_SHA256",
    "COMPONENT_RECOVERY_CONTRACT_SHA256",
    "DECLARED_GRAPH_SUPPORT_SHA256",
    "GENERATOR_CHECKPOINT_SHA256",
    "L1_REACTION_SHA256",
    "MODEL_CONFIG_SHA256",
    "PRODUCTION_GENERATOR_MANIFEST_SHA256",
    "QUALIFIED_REACTION_REGISTRY_SHA256",
    "SAMPLE_STEPS",
    "SELECTED_RESTARTABLE_GENERATOR_SCHEMA_VERSION",
    "SelectedRestartableGeneratorBindings",
    "SelectedRestartableGeneratorCallback",
    "SelectedRestartableGeneratorLane",
    "TERMINAL_DECODER_ID",
    "TRAINING_CACHE_SHA256",
    "UgiSelectedRestartableGeneratorError",
    "declared_graph_support_context_sha256",
]
