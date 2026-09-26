"""Generic adapter-defined node conditioning for synthesis-program instances."""

from __future__ import annotations

from typing import Any

from forge.model.ugi_adapter_features import (
    CORE_POSITION_STATES,
    ORIGIN_STATES,
    PORT_STATES,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class AdapterConditioningError(ValueError):
    """Raised when adapter node semantics do not match the model contract."""


if nn is not None:

    class AdapterNodeConditioning(nn.Module):
        """Embed exact origin/core/port semantics without component identities.

        Core and own-port distances are used only when they are available from
        a valid current topology.  In broad pretraining, ``adapter_mask`` is
        false and the complete contribution is exactly zero.
        """

        def __init__(
            self,
            *,
            hidden_dim: int,
            maximum_distance: int,
            use_all_port_distances: bool = False,
            origin_state_count: int = len(ORIGIN_STATES),
            core_position_state_count: int = len(CORE_POSITION_STATES),
            port_state_count: int = len(PORT_STATES),
            role_count: int = len(ROLE_NAMES),
        ) -> None:
            super().__init__()
            if (
                hidden_dim < 1
                or maximum_distance < 1
                or origin_state_count < 1
                or core_position_state_count < 1
                or port_state_count < 1
                or role_count < 1
            ):
                raise AdapterConditioningError("invalid adapter-conditioning support")
            self.maximum_distance = maximum_distance
            self.use_all_port_distances = use_all_port_distances
            self.origin_state_count = origin_state_count
            self.core_position_state_count = core_position_state_count
            self.port_state_count = port_state_count
            self.role_count = role_count
            self.origin_embedding = nn.Embedding(origin_state_count, hidden_dim)
            self.core_position_embedding = nn.Embedding(core_position_state_count, hidden_dim)
            self.port_embedding = nn.Embedding(port_state_count, hidden_dim)
            # Index zero means not applicable; graph distance d occupies d + 1.
            self.core_distance_embedding = nn.Embedding(maximum_distance + 2, hidden_dim)
            self.own_port_distance_embedding = nn.Embedding(maximum_distance + 2, hidden_dim)
            self.all_port_distance_embeddings = (
                nn.ModuleList(
                    nn.Embedding(maximum_distance + 2, hidden_dim) for _ in range(role_count)
                )
                if use_all_port_distances
                else None
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def _distance_states(self, values: Any, *, allow_not_applicable: bool) -> Any:
            minimum = -1 if allow_not_applicable else 0
            if torch.any(values < minimum):
                raise AdapterConditioningError("adapter graph distance is below support")
            return torch.where(
                values < 0,
                torch.zeros_like(values),
                values.clamp(max=self.maximum_distance) + 1,
            )

        def forward(
            self,
            *,
            origin_states: Any,
            core_position_states: Any,
            port_states: Any,
            distance_to_core: Any,
            distance_to_own_port: Any,
            adapter_mask: Any,
            distances_to_all_ports: Any | None = None,
        ) -> Any:
            shape = origin_states.shape
            if (
                origin_states.ndim != 2
                or core_position_states.shape != shape
                or port_states.shape != shape
                or distance_to_core.shape != shape
                or distance_to_own_port.shape != shape
                or adapter_mask.shape != shape
            ):
                raise AdapterConditioningError(
                    "adapter node channels must share [batch, nodes] shape"
                )
            if adapter_mask.dtype != torch.bool:
                raise AdapterConditioningError("adapter mask must be Boolean")
            for values, classes, label in (
                (origin_states, self.origin_state_count, "origin"),
                (core_position_states, self.core_position_state_count, "core position"),
                (port_states, self.port_state_count, "port"),
            ):
                if torch.any(values < 0) or torch.any(values >= classes):
                    raise AdapterConditioningError(f"adapter {label} state is outside support")
            core_distance_states = self._distance_states(
                distance_to_core,
                allow_not_applicable=False,
            )
            own_distance_states = self._distance_states(
                distance_to_own_port,
                allow_not_applicable=True,
            )
            hidden = (
                self.origin_embedding(origin_states)
                + self.core_position_embedding(core_position_states)
                + self.port_embedding(port_states)
                + self.core_distance_embedding(core_distance_states)
                + self.own_port_distance_embedding(own_distance_states)
            )
            if self.use_all_port_distances:
                if distances_to_all_ports is None or distances_to_all_ports.shape != (
                    *shape,
                    self.role_count,
                ):
                    raise AdapterConditioningError(
                        "all-port ablation requires [batch, nodes, roles] distances"
                    )
                for role_index, embedding in enumerate(self.all_port_distance_embeddings or ()):
                    states = self._distance_states(
                        distances_to_all_ports[..., role_index],
                        allow_not_applicable=False,
                    )
                    hidden = hidden + embedding(states)
            elif distances_to_all_ports is not None:
                raise AdapterConditioningError(
                    "all-port distances were provided to the minimal conditioner"
                )
            return self.norm(hidden) * adapter_mask[..., None]

else:  # pragma: no cover

    class AdapterNodeConditioning:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise AdapterConditioningError("adapter conditioning requires torch")
