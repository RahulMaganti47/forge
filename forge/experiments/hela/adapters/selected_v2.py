"""Restartable guidance adapter for the selected v2 Ugi generator.

This versioned adapter leaves the historical step-1000/argmax adapter intact.
It changes only the authenticated productive identity and terminal realization:
step 2000, the qualified per-role branch ceilings and bond-stochastic terminal
bond decoding with an independent particle terminal stream.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from forge.experiments.hela.adapters.selected_v1 import (
    SelectedGuidanceState,
    SelectedModelRestartableGuidanceLane,
    UgiSelectedGuidanceAdapterError,
    _replace_generator_state,
    _sha256_payload,
)
from forge.experiments.hela.allocation.ugi_nonzero_guidance_runner import (
    GuidanceTerminalCompletionReceipt,
)
from forge.experiments.hela.sampling.ugi_joint_end_to_end_sampling import (
    complete_ugi_joint_terminals,
)
from forge.experiments.hela.sampling.ugi_joint_sparse_sampling import (
    advance_ugi_joint_sparse_state,
    finalize_ugi_joint_sparse_state,
)
from forge.experiments.hela.sampling.ugi_selected_generator_implementation import (
    require_selected_generator_implementation_unchanged,
)
from forge.experiments.hela.sampling.ugi_selected_restartable_generator import SAMPLE_STEPS
from forge.experiments.hela.sampling.ugi_selected_restartable_generator_v2 import (
    MAXIMUM_ADJACENT_BRANCH_RUNS,
    TERMINAL_TEMPERATURE,
)

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - productive generation requires torch
    torch = None


SELECTED_GUIDANCE_ADAPTER_V2_SCHEMA_VERSION = "forge.selected_guidance_adapter.v2"


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


class SelectedModelRestartableGuidanceLaneV2(SelectedModelRestartableGuidanceLane):
    """Selected step-2000, bond-stochastic implementation of the runner lane."""

    def __init__(self, selected_lane: Any) -> None:
        super().__init__(selected_lane)
        self._versioned_adapter_source_sha256 = _source_sha256()
        self.adapter_identity_sha256 = _sha256_payload(
            {
                "schema_version": SELECTED_GUIDANCE_ADAPTER_V2_SCHEMA_VERSION,
                "shared_adapter_identity_sha256": self.adapter_identity_sha256,
                "versioned_adapter_source_sha256": self._versioned_adapter_source_sha256,
                "selected_bindings_sha256": selected_lane.bindings.canonical_sha256,
                "maximum_adjacent_branch_runs": MAXIMUM_ADJACENT_BRANCH_RUNS,
                "terminal_decoder": "bond_stochastic",
                "terminal_temperature": TERMINAL_TEMPERATURE,
            }
        )

    def _require_current_state(self, state: Any) -> SelectedGuidanceState:
        current = super()._require_current_state(state)
        if _source_sha256() != self._versioned_adapter_source_sha256:
            raise UgiSelectedGuidanceAdapterError(
                "versioned v2 guidance-adapter source changed after initialization"
            )
        callback = self.selected_lane.callback
        callback._require_versioned_source_unchanged()
        require_selected_generator_implementation_unchanged(
            callback.repository,
            callback.implementation_qualification,
        )
        return current

    def complete_terminal(
        self,
        state: Any,
        *,
        particle_index: int,
        seed: int,
        checkpoint_index: int,
    ) -> GuidanceTerminalCompletionReceipt:
        if torch is None:
            raise UgiSelectedGuidanceAdapterError("selected v2 guidance requires torch")
        current = self._require_current_state(state)
        if (
            isinstance(particle_index, bool)
            or not isinstance(particle_index, int)
            or particle_index < 0
            or particle_index >= len(current.particles)
        ):
            raise UgiSelectedGuidanceAdapterError("particle_index is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise UgiSelectedGuidanceAdapterError("seed must be a nonnegative integer")
        if checkpoint_index != current.step:
            raise UgiSelectedGuidanceAdapterError(
                "completion checkpoint differs from the state's flow step"
            )
        particle = current.particles[particle_index]
        transition_calls = SAMPLE_STEPS - current.step
        trajectory = particle.trajectory.clone()
        if transition_calls:
            flow_seed = seed
            rollout_state = (
                torch.Generator(device=trajectory.device).manual_seed(flow_seed).get_state()
            )
            trajectory = _replace_generator_state(trajectory, rollout_state)
            trajectory = advance_ugi_joint_sparse_state(
                self.callback.model,
                trajectory,
                target_step=SAMPLE_STEPS,
            )
        else:
            flow_seed = particle.original_particle_seed
        try:
            finalization = finalize_ugi_joint_sparse_state(
                self.callback.model,
                trajectory,
                tree_generator_state=(torch.Generator().manual_seed(flow_seed + 1).get_state()),
                allowed_ring_sizes=self.callback.allowed_ring_sizes,
                maximum_heavy_degree=self.callback.maximum_heavy_degree,
                maximum_adjacent_branch_runs=MAXIMUM_ADJACENT_BRANCH_RUNS,
            )
            if len(finalization.terminals) != 1:
                raise UgiSelectedGuidanceAdapterError(
                    "one v2 guidance particle did not finalize to exactly one sparse terminal"
                )
            completion = complete_ugi_joint_terminals(
                self.callback.model,
                self.callback.closure_model,
                finalization.terminals,
                self.callback.corpus,
                program_metadata=({},),
                closure_generator_state=(torch.Generator().manual_seed(flow_seed + 1).get_state()),
                allowed_ring_sizes=self.callback.allowed_ring_sizes,
                maximum_heavy_degree=self.callback.maximum_heavy_degree,
                l1_reaction=self.callback.reaction,
                terminal_decoder_mode="bond_stochastic",
                terminal_generator_state=(torch.Generator().manual_seed(flow_seed + 2).get_state()),
                terminal_temperature=TERMINAL_TEMPERATURE,
            )
            if len(completion.rows) != 1:
                raise UgiSelectedGuidanceAdapterError(
                    "one v2 guidance particle did not complete to exactly one native row"
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


__all__ = ["SELECTED_GUIDANCE_ADAPTER_V2_SCHEMA_VERSION", "SelectedModelRestartableGuidanceLaneV2"]
