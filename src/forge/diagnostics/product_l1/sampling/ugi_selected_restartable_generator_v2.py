"""Versioned restartable callback for the selected v2 Ugi generator.

The historical production callback is intentionally left unchanged because its
receipts bind the step-1000 masked-argmax lane.  This module binds the current
production manifest instead: step 2000, bond-stochastic terminal bond decoding,
and the qualified per-role adjacent-branch ceilings.  It performs no routing,
guidance, biology, selection, retry, or repair.

The callback uses independent particle streams.  For one productive particle
seed ``s``, flow uses ``s``, sparse closure placement uses ``s + 1`` and
bond-stochastic terminal decoding uses ``s + 2``.  This stream split is the
contract exercised by the grouped lambda-zero seam qualification.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from forge.corpus.ugi_generated_terminal_support import (
    DeclaredGraphSupportContext,
    declared_graph_support_context_sha256,
)
from forge.corpus.ugi_held_component_gate import load_ugi_reaction_contract
from forge.diagnostics.product_l1.sampling.ugi_end_to_end_sampling import (
    _closure_model,
    _load_checkpoint,
)
from forge.diagnostics.product_l1.sampling.ugi_joint_end_to_end_sampling import (
    complete_ugi_joint_terminals,
)
from forge.diagnostics.product_l1.sampling.ugi_joint_sparse_sampling import (
    sample_restartable_terminals,
)
from forge.diagnostics.product_l1.sampling.ugi_selected_generator_implementation import (
    build_selected_generator_implementation_qualification,
    require_selected_generator_implementation_unchanged,
)
from forge.diagnostics.product_l1.sampling.ugi_selected_restartable_generator import (
    SAMPLE_STEPS,
    SelectedRestartableGeneratorCallback,
    SelectedRestartableGeneratorLane,
    UgiSelectedRestartableGeneratorError,
    _atom_vocabulary_states,
    _canonical_sha256,
    _corpus_atom_states,
    _read_json,
    _require_file_hash,
)
from forge.diagnostics.product_l1.training.ugi_training_cache import load_ugi_training_cache
from forge.diagnostics.support.adapters.terminal_support import (
    UgiRestartableTerminalSupportAdapterError,
    adapt_restartable_completion_row_for_route_support,
    decode_canonical_morphology_program_bytes,
    lock_unqualified_restartable_completion_row,
    native_completion_record,
)
from forge.diagnostics.support.guidance.ugi_zero_guidance_rehearsal import (
    RestartableGeneratorClosureAdapter,
    RestartableGeneratorClosureIdentity,
)
from forge.model.ugi_joint_sparse_flow import UgiJointSparseFlow
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
except ModuleNotFoundError:  # pragma: no cover - productive generation requires torch
    torch = None


SELECTED_RESTARTABLE_GENERATOR_V2_SCHEMA_VERSION = (
    "forge.selected_step2000_bond_stochastic_restartable_generator.v1"
)
GENERATOR_CHECKPOINT_SHA256 = "e1a0092a6425d19b09a46ab49660d144eccce20785177544dbc5308ab489d21a"
CLOSURE_CHECKPOINT_SHA256 = "a97507ac6a9eeba41d0cc351666cffdd21069d13bc78c0db671ab3a30c710b5d"
PRODUCTION_GENERATOR_MANIFEST_SHA256 = (
    "7fd0e763a1460290601d6accf31ecb96b2e9ebea325f2b2cdceb8174316a3059"
)
TRAINING_CACHE_SHA256 = "b862b7a54c0cb325f0a962fea42a78138df1a81c88b9c0e86d56b6448e519c46"
ATOM_VOCABULARY_SHA256 = "90ab7430354a9ffd91cfd6e19f2a125020a6547c67cd462a11f77074fd2a0cc5"
QUALIFIED_REACTION_REGISTRY_SHA256 = (
    "296bf06238ef22acc1f55117f5ce0adaee21b1bafaf5a83f89182b0f31cc4fcf"
)
L1_REACTION_SHA256 = "5b97e062b115fcc137b4a05d8f72b74e9cc67a584c05c054ef5f983969bf1427"
MODEL_CONFIG_SHA256 = "f3279a9b64aac4dfae7372c784a863fb93f25cfb4a0f1b5cb1ceb8c4c4a77f52"
DECLARED_GRAPH_SUPPORT_SHA256 = "2fb6253fc1e6237d94029ca8196bb026262f48dea3eca12930a80e5cd0dc8ae7"
COMPONENT_RECOVERY_CONTRACT_SHA256 = (
    "14a84998891a814db287ed808d650a7c8e79175f85ee1508645fa5de0e10b944"
)
TERMINAL_DECODER_ID = "ugi_joint_terminal_completion:bond_stochastic:v1"
TERMINAL_TEMPERATURE = 1.0
MAXIMUM_ADJACENT_BRANCH_RUNS = (2, 1, 1)
_PENDING_EQUIVALENCE_SHA256 = hashlib.sha256(
    b"selected-v2-restartable-equivalence-pending-combined-zero-guidance-seam"
).hexdigest()


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class SelectedStep2000BondStochasticArtifacts:
    """Exact filesystem bindings for the selected v2 productive lane."""

    repository: Path
    generator_checkpoint: Path
    closure_checkpoint: Path
    production_generator_manifest: Path
    training_cache: Path
    atom_vocabulary: Path
    qualified_reaction_registry: Path
    l1_reaction_variant: Path
    component_recovery_contract: Path

    @classmethod
    def from_repository(cls, repository: Path) -> SelectedStep2000BondStochasticArtifacts:
        root = Path(repository).resolve()
        return cls(
            repository=root,
            generator_checkpoint=(
                root / "results/phase1/ugi_joint_sparse_balanced_v2_full/checkpoint_step_2000.pt"
            ),
            closure_checkpoint=(
                root / "results/phase1/ugi_closure_expanded_full/checkpoint_best.pt"
            ),
            production_generator_manifest=(
                root / "results/phase1/ugi_product_l1_production_generator_v2.json"
            ),
            training_cache=(
                root / "results/phase1/ugi_balanced_training_cache_v2/ugi_training_cache.pt"
            ),
            atom_vocabulary=root / "results/phase1/product_v3_atom_vocabulary.json",
            qualified_reaction_registry=root / "data/vendor/qualified_reactions_v1.json",
            l1_reaction_variant=root / "configs/assembly/ugi_variant.yaml",
            component_recovery_contract=root / "forge/corpus/ugi_generated_components.py",
        )


@dataclass(frozen=True)
class SelectedRestartableGeneratorBindingsV2:
    """Content identities for the versioned step-2000 lane."""

    generator_checkpoint_sha256: str
    closure_checkpoint_sha256: str
    production_generator_manifest_sha256: str
    generator_implementation_sha256: str
    versioned_callback_source_sha256: str
    training_cache_sha256: str
    atom_vocabulary_sha256: str
    qualified_reaction_registry_sha256: str
    l1_reaction_sha256: str
    model_config_sha256: str
    declared_graph_support_sha256: str
    component_recovery_contract_sha256: str
    sample_steps: int
    terminal_decoder_id: str
    terminal_temperature: float
    maximum_adjacent_branch_runs: tuple[int, int, int]

    @property
    def canonical_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": SELECTED_RESTARTABLE_GENERATOR_V2_SCHEMA_VERSION,
                **self.__dict__,
            }
        )


def _require_manifest_contract(value: dict[str, Any]) -> None:
    try:
        identity = value["identity"]
        model = value["model"]
        frozen = value["frozen_inputs"]
        sampling = value["sampling_implementation"]
        valid = (
            value["schema_version"] == "phase1_ugi_product_l1_production_generator.v2"
            and value["status"] == "frozen_after_independent_decoder_confirmation"
            and identity["architecture"] == "full_morphology_program_conditioning"
            and identity["checkpoint_step"] == 2000
            and identity["terminal_decoder"] == "bond_stochastic"
            and identity["terminal_temperature"] == TERMINAL_TEMPERATURE
            and tuple(identity["maximum_adjacent_branch_runs_by_role"])
            == MAXIMUM_ADJACENT_BRANCH_RUNS
            and model["checkpoint"]["sha256"] == GENERATOR_CHECKPOINT_SHA256
            and frozen["prepared_training_cache"]["sha256"] == TRAINING_CACHE_SHA256
            and frozen["atom_vocabulary"]["sha256"] == ATOM_VOCABULARY_SHA256
            and sampling["closure_checkpoint"]["sha256"] == CLOSURE_CHECKPOINT_SHA256
            and frozen["qualified_reaction_registry"]["sha256"]
            == QUALIFIED_REACTION_REGISTRY_SHA256
            and value["independent_confirmation_metrics"]["all_confirmation_gates_pass"] is True
        )
    except (KeyError, TypeError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "v2 production generator manifest is malformed"
        ) from error
    if not valid:
        raise UgiSelectedRestartableGeneratorError(
            "v2 production manifest does not bind the selected step-2000 lane"
        )


@dataclass
class SelectedStep2000BondStochasticCallback(SelectedRestartableGeneratorCallback):
    """One independent-stream v2 productive completion callback."""

    versioned_callback_source_sha256: str

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.versioned_callback_source_sha256 != _source_sha256():
            raise UgiSelectedRestartableGeneratorError(
                "versioned v2 callback source changed during construction"
            )
        if self.bindings.terminal_decoder_id != TERMINAL_DECODER_ID:
            raise UgiSelectedRestartableGeneratorError("v2 terminal decoder identity changed")

    def _require_versioned_source_unchanged(self) -> None:
        if _source_sha256() != self.versioned_callback_source_sha256:
            raise UgiSelectedRestartableGeneratorError(
                "versioned v2 callback source changed after preflight"
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
        self._require_versioned_source_unchanged()
        require_selected_generator_implementation_unchanged(
            self.repository,
            self.implementation_qualification,
        )
        entry = request.entry
        if entry.generator_checkpoint_sha256 != self.bindings.generator_checkpoint_sha256:
            raise UgiSelectedRestartableGeneratorError(
                "schedule generator checkpoint differs from the selected v2 checkpoint"
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
            maximum_adjacent_branch_runs=MAXIMUM_ADJACENT_BRANCH_RUNS,
        )
        if (
            len(terminals) != 1
            or metadata.get("sample_steps") != SAMPLE_STEPS
            or metadata.get("terminal_tree_repairs") != 0
            or metadata.get("samples") != 1
        ):
            raise UgiSelectedRestartableGeneratorError(
                "v2 restartable sampler violated the one-terminal eight-step contract"
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
            terminal_decoder_mode="bond_stochastic",
            terminal_generator_state=(
                torch.Generator().manual_seed(request.productive_seed + 2).get_state()
            ),
            terminal_temperature=TERMINAL_TEMPERATURE,
        )
        if len(completion.rows) != 1:
            raise UgiSelectedRestartableGeneratorError(
                "v2 closure/chemistry completion did not return exactly one row"
            )
        try:
            native = native_completion_record(completion.rows[0])
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedRestartableGeneratorError(
                f"native v2 terminal for {entry.unit_id} is not an assessed completion row"
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
                f"native exact-L1 v2 terminal for {entry.unit_id} violated support; "
                "no retry was attempted"
            ) from error


def build_selected_step2000_bond_stochastic_lane(
    repository: Path,
    *,
    artifacts: SelectedStep2000BondStochasticArtifacts | None = None,
) -> SelectedRestartableGeneratorLane:
    """Authenticate and construct the selected v2 independent-stream callback."""

    if torch is None:
        raise UgiSelectedRestartableGeneratorError("selected v2 generation requires torch")
    resolved = artifacts or SelectedStep2000BondStochasticArtifacts.from_repository(repository)
    expected_root = Path(repository).resolve()
    if resolved.repository.resolve() != expected_root:
        raise UgiSelectedRestartableGeneratorError(
            "artifact repository differs from the requested repository"
        )
    implementation_qualification = build_selected_generator_implementation_qualification(
        resolved.repository
    )
    checks = (
        (resolved.generator_checkpoint, GENERATOR_CHECKPOINT_SHA256, "v2 generator checkpoint"),
        (resolved.closure_checkpoint, CLOSURE_CHECKPOINT_SHA256, "closure checkpoint"),
        (
            resolved.production_generator_manifest,
            PRODUCTION_GENERATOR_MANIFEST_SHA256,
            "v2 production generator manifest",
        ),
        (resolved.training_cache, TRAINING_CACHE_SHA256, "prepared training cache"),
        (resolved.atom_vocabulary, ATOM_VOCABULARY_SHA256, "atom vocabulary"),
        (
            resolved.qualified_reaction_registry,
            QUALIFIED_REACTION_REGISTRY_SHA256,
            "qualified Ugi reaction registry",
        ),
        (resolved.l1_reaction_variant, L1_REACTION_SHA256, "qualified Ugi L1 variant"),
        (
            resolved.component_recovery_contract,
            COMPONENT_RECOVERY_CONTRACT_SHA256,
            "generated-component recovery contract",
        ),
    )
    for path, expected, label in checks:
        _require_file_hash(path, expected, label=label)
    _require_manifest_contract(
        _read_json(resolved.production_generator_manifest, label="v2 production manifest")
    )

    joint_checkpoint = _load_checkpoint(
        resolved.generator_checkpoint,
        "phase1_ugi_joint_sparse_checkpoint.v1",
    )
    closure_checkpoint = _load_checkpoint(
        resolved.closure_checkpoint,
        (
            "phase1_ugi_sparse_closure_checkpoint.v1",
            "phase1_ugi_sparse_closure_checkpoint.v2",
        ),
    )
    model_config = dict(joint_checkpoint.get("model_config", {}))
    if _canonical_sha256(model_config) != MODEL_CONFIG_SHA256:
        raise UgiSelectedRestartableGeneratorError("selected v2 model configuration changed")
    checkpoint_inputs = joint_checkpoint.get("inputs")
    if not isinstance(checkpoint_inputs, Mapping):
        raise UgiSelectedRestartableGeneratorError("selected v2 checkpoint inputs are malformed")
    try:
        input_valid = (
            checkpoint_inputs["prepared_cache"]["sha256"] == TRAINING_CACHE_SHA256
            and checkpoint_inputs["atom_vocabulary"]["sha256"] == ATOM_VOCABULARY_SHA256
        )
    except (KeyError, TypeError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "selected v2 checkpoint lacks cache or vocabulary bindings"
        ) from error
    if not input_valid:
        raise UgiSelectedRestartableGeneratorError(
            "selected v2 checkpoint cache or vocabulary binding changed"
        )

    corpus, _ = load_ugi_training_cache(resolved.training_cache)
    vocabulary_states = _atom_vocabulary_states(resolved.atom_vocabulary)
    if _corpus_atom_states(corpus) != vocabulary_states:
        raise UgiSelectedRestartableGeneratorError(
            "prepared cache atom vocabulary differs from the frozen vocabulary artifact"
        )
    graph_support = DeclaredGraphSupportContext(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        model_config=MappingProxyType(model_config.copy()),
        atom_vocabulary=vocabulary_states,
    )
    graph_support_sha256 = declared_graph_support_context_sha256(graph_support)
    if graph_support_sha256 != DECLARED_GRAPH_SUPPORT_SHA256:
        raise UgiSelectedRestartableGeneratorError("selected v2 graph-support context changed")

    architecture = model_config.copy()
    source_probability_floor = architecture.pop("source_probability_floor", None)
    if source_probability_floor != 1e-5:
        raise UgiSelectedRestartableGeneratorError("selected v2 source probability floor changed")
    model = UgiJointSparseFlow(atom_classes=len(corpus.atom_vocabulary), **architecture)
    model.load_state_dict(joint_checkpoint["model_state"])
    model.eval()
    closure_model = _closure_model(closure_checkpoint)
    closure_model.eval()
    reaction = load_ugi_reaction_contract(resolved.qualified_reaction_registry)
    reverifier = QualifiedUgiL1Reverifier(
        reaction_contract=reaction,
        l1_reaction_sha256=L1_REACTION_SHA256,
    )
    try:
        allowed_ring_sizes = tuple(int(value) for value in closure_checkpoint["allowed_ring_sizes"])
        maximum_heavy_degree = int(closure_checkpoint["maximum_heavy_degree"])
    except (KeyError, TypeError, ValueError) as error:
        raise UgiSelectedRestartableGeneratorError(
            "closure checkpoint support bounds are malformed"
        ) from error
    if allowed_ring_sizes != (5, 6, 7) or maximum_heavy_degree != 4:
        raise UgiSelectedRestartableGeneratorError("closure checkpoint support bounds changed")
    source_marginals = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in joint_checkpoint["source_marginals"].items()
    }
    versioned_source_sha256 = _source_sha256()
    bindings = SelectedRestartableGeneratorBindingsV2(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        production_generator_manifest_sha256=PRODUCTION_GENERATOR_MANIFEST_SHA256,
        generator_implementation_sha256=(implementation_qualification.implementation_sha256),
        versioned_callback_source_sha256=versioned_source_sha256,
        training_cache_sha256=TRAINING_CACHE_SHA256,
        atom_vocabulary_sha256=ATOM_VOCABULARY_SHA256,
        qualified_reaction_registry_sha256=QUALIFIED_REACTION_REGISTRY_SHA256,
        l1_reaction_sha256=L1_REACTION_SHA256,
        model_config_sha256=MODEL_CONFIG_SHA256,
        declared_graph_support_sha256=graph_support_sha256,
        component_recovery_contract_sha256=COMPONENT_RECOVERY_CONTRACT_SHA256,
        sample_steps=SAMPLE_STEPS,
        terminal_decoder_id=TERMINAL_DECODER_ID,
        terminal_temperature=TERMINAL_TEMPERATURE,
        maximum_adjacent_branch_runs=MAXIMUM_ADJACENT_BRANCH_RUNS,
    )
    callback = SelectedStep2000BondStochasticCallback(
        model=model,
        closure_model=closure_model,
        source_marginals=source_marginals,
        corpus=corpus,
        reaction=reaction,
        graph_support=graph_support,
        l1_reverifier=reverifier,
        bindings=bindings,
        allowed_ring_sizes=allowed_ring_sizes,
        maximum_heavy_degree=maximum_heavy_degree,
        repository=resolved.repository,
        implementation_qualification=implementation_qualification,
        versioned_callback_source_sha256=versioned_source_sha256,
    )
    identity = RestartableGeneratorClosureIdentity(
        generator_checkpoint_sha256=GENERATOR_CHECKPOINT_SHA256,
        closure_checkpoint_sha256=CLOSURE_CHECKPOINT_SHA256,
        production_generator_manifest_sha256=PRODUCTION_GENERATOR_MANIFEST_SHA256,
        restartable_equivalence_receipt_sha256=_PENDING_EQUIVALENCE_SHA256,
        generator_implementation_sha256=(implementation_qualification.implementation_sha256),
        terminal_decoder_id=TERMINAL_DECODER_ID,
    )
    return SelectedRestartableGeneratorLane(
        adapter=RestartableGeneratorClosureAdapter(
            identity=identity, generate_locked_terminal=callback
        ),
        callback=callback,
        bindings=bindings,
        graph_support=graph_support,
        implementation_qualification=implementation_qualification,
    )


__all__ = [
    "MAXIMUM_ADJACENT_BRANCH_RUNS",
    "SELECTED_RESTARTABLE_GENERATOR_V2_SCHEMA_VERSION",
    "SelectedRestartableGeneratorBindingsV2",
    "SelectedStep2000BondStochasticArtifacts",
    "SelectedStep2000BondStochasticCallback",
    "TERMINAL_DECODER_ID",
    "TERMINAL_TEMPERATURE",
    "build_selected_step2000_bond_stochastic_lane",
]
