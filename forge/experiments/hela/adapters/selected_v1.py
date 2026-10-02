"""Restartable independent-stream guidance lane for the selected Ugi generator.

The matched guidance runner treats a particle pool as one state, whereas the
qualified selected generator defines its stochastic contract through
``batch_size=1`` calls keyed by one seed per particle.  This adapter preserves
that contract exactly: every particle owns an independent flow RNG stream,
ancestry copies categorical state without copying the destination RNG stream,
and terminal shadow rollouts operate on deep clones.

The implementation deliberately uses one model call per particle for now.
Batching neural inference is a future optimization and must be separately
qualified because different batch shapes are not assumed bitwise equivalent.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_json as _sha256_payload
from forge.experiments.hela.adapters.terminal_support import (
    UgiRestartableTerminalSupportAdapterError,
    adapt_restartable_completion_row_for_route_support,
    decode_canonical_morphology_program_bytes,
    lock_unqualified_restartable_completion_row,
    native_completion_record,
)
from forge.experiments.hela.allocation.ugi_nonzero_guidance_runner import (
    GuidanceStateReceipt,
    GuidanceTerminalCompletionReceipt,
)
from forge.experiments.hela.sampling.ugi_joint_end_to_end_sampling import (
    complete_ugi_joint_terminals,
)
from forge.experiments.hela.sampling.ugi_joint_sparse_sampling import (
    UgiJointSparseTrajectoryState,
    advance_ugi_joint_sparse_state,
    finalize_ugi_joint_sparse_state,
    initialize_ugi_joint_sparse_state,
)
from forge.experiments.hela.sampling.ugi_selected_generator_implementation import (
    require_selected_generator_implementation_unchanged,
)
from forge.experiments.hela.sampling.ugi_selected_restartable_generator import (
    SAMPLE_STEPS,
    SelectedRestartableGeneratorLane,
)
from forge.synthesis.matched import (
    MatchedArm,
    MatchedGenerationRequest,
    MatchedScheduleEntry,
    RouteComputeUsage,
)
from forge.synthesis.terminals.terminal_assessment import (
    UgiTerminalRouteAssessmentError,
    ValidatedUgiTerminalPayload,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - productive generation requires torch
    torch = None


SELECTED_GUIDANCE_ADAPTER_SCHEMA_VERSION = "forge.selected_guidance_adapter.v1"


class UgiSelectedGuidanceAdapterError(RuntimeError):
    """Raised when the selected guidance lane cannot preserve its contract."""


def _stable_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise UgiSelectedGuidanceAdapterError(
            "guidance-adapter record is not canonically serializable"
        ) from error


def _nonnegative_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UgiSelectedGuidanceAdapterError(f"{label} must be a nonnegative integer")
    return value


def _tensor_sha256(tensor: Any) -> str:
    if torch is None or not torch.is_tensor(tensor):
        raise UgiSelectedGuidanceAdapterError("trajectory state contains a non-tensor value")
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(_stable_json_bytes(list(value.shape)))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _tensor_mapping_sha256(values: dict[str, Any]) -> str:
    return _sha256_payload({key: _tensor_sha256(values[key]) for key in sorted(values)})


def _particle_seed_manifest_sha256(particle_seeds: Sequence[int]) -> str:
    """Match the runner's canonical JSON digest of the frozen seed tuple."""

    return _sha256_payload(tuple(particle_seeds))


def _replace_generator_state(
    trajectory: UgiJointSparseTrajectoryState,
    generator_state: Any,
) -> UgiJointSparseTrajectoryState:
    return UgiJointSparseTrajectoryState(
        programs=trajectory.programs,
        layout={key: value.clone() for key, value in trajectory.layout.items()},
        sources={key: value.clone() for key, value in trajectory.sources.items()},
        channels={key: value.clone() for key, value in trajectory.channels.items()},
        sample_steps=trajectory.sample_steps,
        step=trajectory.step,
        generator_state=generator_state.clone(),
        device=trajectory.device,
    )


@dataclass(frozen=True)
class SelectedGuidanceParticleState:
    """One categorical particle plus its destination-owned stochastic stream."""

    global_particle_index: int
    program_index: int
    program_bytes: bytes
    original_particle_seed: int
    trajectory: UgiJointSparseTrajectoryState
    founder_particle_index: int
    ancestry_path: tuple[int, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer(self.global_particle_index, label="global_particle_index")
        _nonnegative_integer(self.program_index, label="program_index")
        _nonnegative_integer(self.original_particle_seed, label="original_particle_seed")
        _nonnegative_integer(self.founder_particle_index, label="founder_particle_index")
        if not isinstance(self.program_bytes, bytes) or not self.program_bytes:
            raise UgiSelectedGuidanceAdapterError("program_bytes must be nonempty bytes")
        if not isinstance(self.trajectory, UgiJointSparseTrajectoryState):
            raise UgiSelectedGuidanceAdapterError("particle trajectory must be typed")
        if len(self.trajectory.programs) != 1:
            raise UgiSelectedGuidanceAdapterError(
                "each independent-stream particle must own exactly one trajectory row"
            )
        decoded = decode_canonical_morphology_program_bytes(self.program_bytes)
        if self.trajectory.programs != (decoded,):
            raise UgiSelectedGuidanceAdapterError(
                "particle program bytes differ from its trajectory program"
            )
        if not self.ancestry_path or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.ancestry_path
        ):
            raise UgiSelectedGuidanceAdapterError(
                "ancestry_path must contain nonnegative particle indices"
            )

    def clone(self) -> SelectedGuidanceParticleState:
        return SelectedGuidanceParticleState(
            global_particle_index=self.global_particle_index,
            program_index=self.program_index,
            program_bytes=self.program_bytes,
            original_particle_seed=self.original_particle_seed,
            trajectory=self.trajectory.clone(),
            founder_particle_index=self.founder_particle_index,
            ancestry_path=self.ancestry_path,
        )

    @property
    def program_sha256(self) -> str:
        return hashlib.sha256(self.program_bytes).hexdigest()

    @property
    def categorical_state_sha256(self) -> str:
        return _sha256_payload(
            {
                "step": self.trajectory.step,
                "program_sha256": self.program_sha256,
                "channels_sha256": _tensor_mapping_sha256(self.trajectory.channels),
            }
        )

    @property
    def rng_state_sha256(self) -> str:
        return _tensor_sha256(self.trajectory.generator_state)

    @property
    def provenance_sha256(self) -> str:
        return _sha256_payload(
            {
                "global_particle_index": self.global_particle_index,
                "program_index": self.program_index,
                "program_sha256": self.program_sha256,
                "original_particle_seed": self.original_particle_seed,
            }
        )

    @property
    def lineage_sha256(self) -> str:
        return _sha256_payload(
            {
                "destination_particle_index": self.global_particle_index,
                "founder_particle_index": self.founder_particle_index,
                "ancestry_path": self.ancestry_path,
            }
        )

    @property
    def state_sha256(self) -> str:
        return _sha256_payload(
            {
                "categorical_state_sha256": self.categorical_state_sha256,
                "rng_state_sha256": self.rng_state_sha256,
                "provenance_sha256": self.provenance_sha256,
                "lineage_sha256": self.lineage_sha256,
            }
        )


@dataclass(frozen=True)
class SelectedGuidanceState:
    """Authenticated collection of independently keyed selected-model particles."""

    base_seed: int
    adapter_identity_sha256: str
    particles: tuple[SelectedGuidanceParticleState, ...]

    def __post_init__(self) -> None:
        _nonnegative_integer(self.base_seed, label="base_seed")
        if (
            not isinstance(self.adapter_identity_sha256, str)
            or len(self.adapter_identity_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.adapter_identity_sha256)
        ):
            raise UgiSelectedGuidanceAdapterError(
                "adapter_identity_sha256 must be a lowercase SHA-256 digest"
            )
        if not self.particles:
            raise UgiSelectedGuidanceAdapterError("guidance state requires particles")
        if tuple(value.global_particle_index for value in self.particles) != tuple(
            range(len(self.particles))
        ):
            raise UgiSelectedGuidanceAdapterError(
                "guidance particles must remain in canonical global-index order"
            )
        steps = {value.trajectory.step for value in self.particles}
        sample_steps = {value.trajectory.sample_steps for value in self.particles}
        devices = {value.trajectory.device for value in self.particles}
        if len(steps) != 1 or sample_steps != {SAMPLE_STEPS} or len(devices) != 1:
            raise UgiSelectedGuidanceAdapterError(
                "guidance particles must share one device and exact flow-step boundary"
            )

    def clone(self) -> SelectedGuidanceState:
        return SelectedGuidanceState(
            base_seed=self.base_seed,
            adapter_identity_sha256=self.adapter_identity_sha256,
            particles=tuple(value.clone() for value in self.particles),
        )

    @property
    def step(self) -> int:
        return self.particles[0].trajectory.step

    @property
    def device(self) -> str:
        return self.particles[0].trajectory.device

    @property
    def categorical_state_sha256(self) -> str:
        return _sha256_payload(tuple(value.categorical_state_sha256 for value in self.particles))

    @property
    def provenance_sha256(self) -> str:
        return _sha256_payload(
            {
                "base_seed": self.base_seed,
                "adapter_identity_sha256": self.adapter_identity_sha256,
                "particles": tuple(value.provenance_sha256 for value in self.particles),
            }
        )

    @property
    def lineage_sha256(self) -> str:
        return _sha256_payload(tuple(value.lineage_sha256 for value in self.particles))

    @property
    def state_sha256(self) -> str:
        return _sha256_payload(
            {
                "categorical_state_sha256": self.categorical_state_sha256,
                "rng_state_sha256s": tuple(value.rng_state_sha256 for value in self.particles),
                "provenance_sha256": self.provenance_sha256,
                "lineage_sha256": self.lineage_sha256,
                "step": self.step,
            }
        )


class SelectedModelRestartableGuidanceLane:
    """Production selected-model implementation of ``RestartableGuidanceLane``."""

    def __init__(self, selected_lane: SelectedRestartableGeneratorLane) -> None:
        if not isinstance(selected_lane, SelectedRestartableGeneratorLane):
            raise UgiSelectedGuidanceAdapterError(
                "selected guidance adapter requires a selected restartable lane"
            )
        self.selected_lane = selected_lane
        callback = selected_lane.callback
        source_path = Path(__file__)
        self._adapter_source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        self.adapter_identity_sha256 = _sha256_payload(
            {
                "schema_version": SELECTED_GUIDANCE_ADAPTER_SCHEMA_VERSION,
                "adapter_source_sha256": self._adapter_source_sha256,
                "selected_bindings": asdict(selected_lane.bindings),
                "allowed_ring_sizes": callback.allowed_ring_sizes,
                "maximum_heavy_degree": callback.maximum_heavy_degree,
            }
        )

    @property
    def callback(self) -> Any:
        return self.selected_lane.callback

    def _require_current_state(self, state: Any) -> SelectedGuidanceState:
        if not isinstance(state, SelectedGuidanceState):
            raise UgiSelectedGuidanceAdapterError("guidance state must be typed")
        if state.adapter_identity_sha256 != self.adapter_identity_sha256:
            raise UgiSelectedGuidanceAdapterError(
                "guidance state belongs to a different adapter implementation"
            )
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._adapter_source_sha256:
            raise UgiSelectedGuidanceAdapterError(
                "guidance-adapter source changed after state initialization"
            )
        require_selected_generator_implementation_unchanged(
            self.callback.repository,
            self.callback.implementation_qualification,
        )
        return state

    def initialize(
        self,
        programs: tuple[bytes, ...],
        *,
        seed: int,
        particle_seeds: tuple[int, ...],
        device: str,
    ) -> GuidanceStateReceipt:
        if torch is None:
            raise UgiSelectedGuidanceAdapterError("selected guidance requires torch")
        _nonnegative_integer(seed, label="seed")
        if not programs or len(programs) != len(particle_seeds):
            raise UgiSelectedGuidanceAdapterError(
                "programs and particle_seeds must be nonempty and aligned"
            )
        if len(set(particle_seeds)) != len(particle_seeds):
            raise UgiSelectedGuidanceAdapterError("particle_seeds must be unique")
        for value in particle_seeds:
            _nonnegative_integer(value, label="particle_seed")
        try:
            resolved_device = torch.device(device)
        except (TypeError, RuntimeError) as error:
            raise UgiSelectedGuidanceAdapterError("device is invalid") from error
        if resolved_device.type != "cpu":
            raise UgiSelectedGuidanceAdapterError(
                "the selected closure/chemistry completion path is currently CPU-qualified only"
            )
        require_selected_generator_implementation_unchanged(
            self.callback.repository,
            self.callback.implementation_qualification,
        )
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != self._adapter_source_sha256:
            raise UgiSelectedGuidanceAdapterError(
                "guidance-adapter source changed after construction"
            )
        unique_program_indices: dict[bytes, int] = {}
        particles = []
        for global_index, (program_bytes, particle_seed) in enumerate(
            zip(programs, particle_seeds, strict=True)
        ):
            try:
                program = decode_canonical_morphology_program_bytes(program_bytes)
            except UgiRestartableTerminalSupportAdapterError as error:
                raise UgiSelectedGuidanceAdapterError(
                    f"particle {global_index} morphology program is not canonical"
                ) from error
            if program_bytes not in unique_program_indices:
                unique_program_indices[program_bytes] = len(unique_program_indices)
            trajectory = initialize_ugi_joint_sparse_state(
                self.callback.model,
                (program,),
                self.callback.source_marginals,
                sample_steps=SAMPLE_STEPS,
                device=str(resolved_device),
                seed=particle_seed,
            )
            particles.append(
                SelectedGuidanceParticleState(
                    global_particle_index=global_index,
                    program_index=unique_program_indices[program_bytes],
                    program_bytes=program_bytes,
                    original_particle_seed=particle_seed,
                    trajectory=trajectory,
                    founder_particle_index=global_index,
                    ancestry_path=(global_index,),
                )
            )
        state = SelectedGuidanceState(
            base_seed=seed,
            adapter_identity_sha256=self.adapter_identity_sha256,
            particles=tuple(particles),
        )
        return GuidanceStateReceipt(
            state=state,
            product_transition_calls=0,
            consumed_particle_seed_manifest_sha256=(_particle_seed_manifest_sha256(particle_seeds)),
        )

    def advance(self, state: Any, *, target_step: int) -> GuidanceStateReceipt:
        current = self._require_current_state(state)
        _nonnegative_integer(target_step, label="target_step")
        if not current.step <= target_step <= SAMPLE_STEPS:
            raise UgiSelectedGuidanceAdapterError("target_step is outside the flow schedule")
        advanced = tuple(
            SelectedGuidanceParticleState(
                global_particle_index=particle.global_particle_index,
                program_index=particle.program_index,
                program_bytes=particle.program_bytes,
                original_particle_seed=particle.original_particle_seed,
                trajectory=advance_ugi_joint_sparse_state(
                    self.callback.model,
                    particle.trajectory,
                    target_step=target_step,
                ),
                founder_particle_index=particle.founder_particle_index,
                ancestry_path=particle.ancestry_path,
            )
            for particle in current.particles
        )
        return GuidanceStateReceipt(
            state=SelectedGuidanceState(
                base_seed=current.base_seed,
                adapter_identity_sha256=current.adapter_identity_sha256,
                particles=advanced,
            ),
            product_transition_calls=len(advanced) * (target_step - current.step),
        )

    def snapshot(self, state: Any) -> GuidanceStateReceipt:
        current = self._require_current_state(state)
        return GuidanceStateReceipt(state=current.clone(), product_transition_calls=0)

    def apply_ancestry(
        self,
        state: Any,
        ancestors: Sequence[int],
    ) -> GuidanceStateReceipt:
        current = self._require_current_state(state)
        if len(ancestors) != len(current.particles):
            raise UgiSelectedGuidanceAdapterError(
                "ancestry requires one source index per destination particle"
            )
        for destination, source_index in enumerate(ancestors):
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or source_index < 0
                or source_index >= len(current.particles)
            ):
                raise UgiSelectedGuidanceAdapterError("ancestry contains an invalid index")
            if (
                current.particles[destination].program_bytes
                != current.particles[source_index].program_bytes
            ):
                raise UgiSelectedGuidanceAdapterError(
                    "ancestry cannot cross frozen morphology-program groups"
                )
        if tuple(ancestors) == tuple(range(len(current.particles))):
            return GuidanceStateReceipt(state=current.clone(), product_transition_calls=0)

        resampled = []
        for destination_index, source_index in enumerate(ancestors):
            destination = current.particles[destination_index]
            source = current.particles[source_index]
            if destination_index == source_index:
                resampled.append(destination.clone())
                continue
            source_trajectory = source.trajectory.clone()
            trajectory = _replace_generator_state(
                source_trajectory,
                destination.trajectory.generator_state,
            )
            resampled.append(
                SelectedGuidanceParticleState(
                    global_particle_index=destination.global_particle_index,
                    program_index=destination.program_index,
                    program_bytes=destination.program_bytes,
                    original_particle_seed=destination.original_particle_seed,
                    trajectory=trajectory,
                    founder_particle_index=source.founder_particle_index,
                    ancestry_path=source.ancestry_path + (destination.global_particle_index,),
                )
            )
        return GuidanceStateReceipt(
            state=SelectedGuidanceState(
                base_seed=current.base_seed,
                adapter_identity_sha256=current.adapter_identity_sha256,
                particles=tuple(resampled),
            ),
            product_transition_calls=0,
        )

    def completion_unit_id(
        self,
        state: SelectedGuidanceState,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> str:
        """Return the immutable state/provenance/lineage-bound completion ID."""

        current = self._require_current_state(state)
        _nonnegative_integer(particle_index, label="particle_index")
        _nonnegative_integer(seed, label="seed")
        _nonnegative_integer(checkpoint_index, label="checkpoint_index")
        if particle_index >= len(current.particles):
            raise UgiSelectedGuidanceAdapterError("particle_index exceeds the state")
        particle = current.particles[particle_index]
        return (
            "selected-guidance"
            f":p{particle_index}:c{checkpoint_index}:s{seed}"
            f":pool_state={current.state_sha256}"
            f":pool_provenance={current.provenance_sha256}"
            f":pool_lineage={current.lineage_sha256}"
            f":particle_state={particle.state_sha256}"
            f":particle_provenance={particle.provenance_sha256}"
            f":particle_lineage={particle.lineage_sha256}"
            f":adapter={current.adapter_identity_sha256}"
        )

    def completion_generation_request(
        self,
        state: SelectedGuidanceState,
        *,
        particle_index: int,
        invocation_seed: int,
        checkpoint_index: int,
    ) -> MatchedGenerationRequest:
        """Expose the exact terminal request for independent equivalence audit."""

        current = self._require_current_state(state)
        if checkpoint_index != current.step:
            raise UgiSelectedGuidanceAdapterError(
                "completion request checkpoint differs from the state's flow step"
            )
        _nonnegative_integer(particle_index, label="particle_index")
        _nonnegative_integer(invocation_seed, label="invocation_seed")
        if particle_index >= len(current.particles):
            raise UgiSelectedGuidanceAdapterError("particle_index exceeds the state")
        state = current
        particle = state.particles[particle_index]
        productive_seed = (
            particle.original_particle_seed if checkpoint_index == SAMPLE_STEPS else invocation_seed
        )
        entry = MatchedScheduleEntry(
            unit_id=self.completion_unit_id(
                state,
                particle_index=particle_index,
                seed=invocation_seed,
                checkpoint_index=checkpoint_index,
            ),
            morphology_program=particle.program_bytes,
            program_index=particle.program_index,
            particle_index=particle.global_particle_index,
            checkpoint_index=checkpoint_index,
            generator_checkpoint_sha256=(self.selected_lane.bindings.generator_checkpoint_sha256),
            closure_checkpoint_sha256=(self.selected_lane.bindings.closure_checkpoint_sha256),
            rollout_index=0,
            productive_generation_calls=1,
            route_reservation=RouteComputeUsage(),
        )
        return MatchedGenerationRequest(
            arm=MatchedArm.GUIDED,
            entry=entry,
            productive_seed=productive_seed,
        )

    def _lock_completion_row(
        self,
        row: Any,
        request: MatchedGenerationRequest,
    ) -> Any:
        try:
            native = native_completion_record(row)
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedGuidanceAdapterError(
                "native terminal is not an assessed completion row"
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
                l1_reaction=self.callback.reaction,
                l1_reaction_sha256=self.selected_lane.bindings.l1_reaction_sha256,
                component_recovery_contract_sha256=(
                    self.selected_lane.bindings.component_recovery_contract_sha256
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
                l1_reaction=self.callback.reaction,
                l1_reaction_sha256=self.selected_lane.bindings.l1_reaction_sha256,
                component_recovery_contract_sha256=(
                    self.selected_lane.bindings.component_recovery_contract_sha256
                ),
                graph_support=self.callback.graph_support,
                l1_reverifier=self.callback.l1_reverifier,
            ).locked_terminal
        except UgiRestartableTerminalSupportAdapterError as error:
            raise UgiSelectedGuidanceAdapterError(
                "native exact-L1 terminal violated selected route-root support"
            ) from error

    def complete_terminal(
        self,
        state: Any,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt:
        if torch is None:
            raise UgiSelectedGuidanceAdapterError("selected guidance requires torch")
        current = self._require_current_state(state)
        _nonnegative_integer(particle_index, label="particle_index")
        _nonnegative_integer(seed, label="seed")
        _nonnegative_integer(checkpoint_index, label="checkpoint_index")
        if particle_index >= len(current.particles):
            raise UgiSelectedGuidanceAdapterError("particle_index exceeds the state")
        if checkpoint_index != current.step:
            raise UgiSelectedGuidanceAdapterError(
                "completion checkpoint differs from the state's flow step"
            )
        particle = current.particles[particle_index]
        transition_calls = SAMPLE_STEPS - current.step
        trajectory = particle.trajectory.clone()
        if transition_calls:
            rollout_state = torch.Generator(device=trajectory.device).manual_seed(seed).get_state()
            trajectory = _replace_generator_state(trajectory, rollout_state)
            trajectory = advance_ugi_joint_sparse_state(
                self.callback.model,
                trajectory,
                target_step=SAMPLE_STEPS,
            )
            productive_seed = seed
        else:
            # Preserve the frozen batch_size=1 reference: final tree and closure
            # streams are keyed by the original particle seed + 1.  The runner's
            # final invocation seed is still bound into the unit identifier.
            productive_seed = particle.original_particle_seed
        tree_and_closure_seed = productive_seed + 1
        try:
            finalization = finalize_ugi_joint_sparse_state(
                self.callback.model,
                trajectory,
                tree_generator_state=(
                    torch.Generator().manual_seed(tree_and_closure_seed).get_state()
                ),
                allowed_ring_sizes=self.callback.allowed_ring_sizes,
                maximum_heavy_degree=self.callback.maximum_heavy_degree,
                maximum_adjacent_branch_runs=(None, None, None),
            )
            if len(finalization.terminals) != 1:
                raise UgiSelectedGuidanceAdapterError(
                    "one guidance particle did not finalize to exactly one sparse terminal"
                )
            completion = complete_ugi_joint_terminals(
                self.callback.model,
                self.callback.closure_model,
                finalization.terminals,
                self.callback.corpus,
                program_metadata=({},),
                closure_generator_state=(
                    torch.Generator().manual_seed(tree_and_closure_seed).get_state()
                ),
                allowed_ring_sizes=self.callback.allowed_ring_sizes,
                maximum_heavy_degree=self.callback.maximum_heavy_degree,
                l1_reaction=self.callback.reaction,
                terminal_decoder_mode="argmax",
                terminal_generator_state=None,
                terminal_temperature=1.0,
            )
            if len(completion.rows) != 1:
                raise UgiSelectedGuidanceAdapterError(
                    "one guidance particle did not complete to exactly one native row"
                )
            request = self.completion_generation_request(
                current,
                particle_index=particle_index,
                invocation_seed=seed,
                checkpoint_index=checkpoint_index,
            )
            terminal = self._lock_completion_row(completion.rows[0], request)
        except Exception as error:
            return GuidanceTerminalCompletionReceipt(
                terminal=None,
                product_transition_calls=transition_calls,
                error_detail=f"{type(error).__name__}: {error}",
            )
        return GuidanceTerminalCompletionReceipt(
            terminal=terminal,
            product_transition_calls=transition_calls,
        )


__all__ = [
    "SELECTED_GUIDANCE_ADAPTER_SCHEMA_VERSION",
    "SelectedGuidanceParticleState",
    "SelectedGuidanceState",
    "SelectedModelRestartableGuidanceLane",
    "UgiSelectedGuidanceAdapterError",
]
