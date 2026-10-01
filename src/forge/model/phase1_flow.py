"""GPU-ready sparse whole-lipid discrete-flow training for Phase 1."""

from __future__ import annotations

import math
from typing import Any

from forge.model.sparse_topology_feasibility import (
    BOND_VALENCE_UNITS,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None

CONFIG_SCHEMA_VERSION = "phase1_product_pretrain_config.v1"
CONFIG_SCHEMA_VERSION_V2 = "phase1_product_pretrain_config.v2"
CONFIG_SCHEMA_VERSION_V3 = "phase1_product_pretrain_config.v3"
RESULT_SCHEMA_VERSION = "phase1_product_pretrain_result.v1"
SIZE_BUCKETS = ("le40", "41_64", "65_96", "97_128", "gt128")
R0_TO_R1_BUCKET = {
    "le40": "le40",
    "41_64": "41_64",
    "65_96": "65_96",
    "97_128": "97_128",
    "gt128": "gt128",
}


class Phase1FlowError(ValueError):
    """Raised when Phase 1 flow training violates its frozen contract."""


if nn is not None:

    def _gather_training_nodes(hidden: Any, indices: Any) -> Any:
        return hidden.gather(1, indices[:, :, None].expand(-1, -1, hidden.shape[-1]))

    def _noisy_topology_features(
        parents: Any,
        parent_bonds: Any,
        closure_left: Any,
        closure_right: Any,
        closure_bonds: Any,
        node_mask: Any,
        child_mask: Any,
        closure_mask: Any,
        bond_valence_units: Any | None = None,
    ) -> Any:
        """Compute bounded molecular context from only the current noisy state."""

        node_count = parents.shape[1]
        dtype = torch.float32
        parent_assignment = functional.one_hot(
            parents,
            num_classes=node_count,
        ).to(dtype)
        parent_assignment = parent_assignment * child_mask[:, :, None]
        child_count = parent_assignment.sum(dim=1)

        left_assignment = functional.one_hot(
            closure_left,
            num_classes=node_count,
        ).to(dtype)
        right_assignment = functional.one_hot(
            closure_right,
            num_classes=node_count,
        ).to(dtype)
        active_closures = closure_mask[:, :, None].to(dtype)
        left_assignment = left_assignment * active_closures
        right_assignment = right_assignment * active_closures
        closure_degree = left_assignment.sum(dim=1) + right_assignment.sum(dim=1)

        valence_lookup = (
            BOND_VALENCE_UNITS.to(parent_bonds.device)
            if bond_valence_units is None
            else bond_valence_units
        )
        bond_units = valence_lookup[parent_bonds].to(dtype)
        bond_units = bond_units * child_mask.to(dtype)
        valence_units = bond_units.clone()
        valence_units += torch.einsum(
            "bcp,bc->bp",
            parent_assignment,
            bond_units,
        )
        closure_units = valence_lookup[closure_bonds].to(dtype)
        closure_units = closure_units * closure_mask.to(dtype)
        valence_units += torch.einsum(
            "bkn,bk->bn",
            left_assignment + right_assignment,
            closure_units,
        )

        degree = child_count + child_mask.to(dtype) + closure_degree
        features = torch.stack(
            (
                child_count.clamp(max=4.0) / 4.0,
                degree.clamp(max=8.0) / 8.0,
                valence_units.clamp(max=16.0) / 16.0,
                closure_degree.clamp(max=4.0) / 4.0,
            ),
            dim=-1,
        )
        return features * node_mask[:, :, None]

    class DeterministicSparseFlowBlock(nn.Module):
        """Tree and residual-closure message passing without CUDA scatter reductions."""

        def __init__(self, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.update = nn.Sequential(
                nn.Linear(5 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            hidden: Any,
            parents: Any,
            closure_left: Any,
            closure_right: Any,
            closure_bond_hidden: Any,
            node_mask: Any,
            child_mask: Any,
            closure_mask: Any,
        ) -> Any:
            node_count = hidden.shape[1]
            parent_hidden = _gather_training_nodes(hidden, parents)
            parent_assignment = functional.one_hot(
                parents,
                num_classes=node_count,
            ).to(hidden.dtype)
            parent_assignment = parent_assignment * child_mask[:, :, None]
            child_sum = torch.einsum("bcp,bcd->bpd", parent_assignment, hidden)
            child_count = parent_assignment.sum(dim=1)
            child_mean = child_sum / child_count[:, :, None].clamp(min=1.0)

            left_hidden = _gather_training_nodes(hidden, closure_left)
            right_hidden = _gather_training_nodes(hidden, closure_right)
            left_assignment = functional.one_hot(
                closure_left,
                num_classes=node_count,
            ).to(hidden.dtype)
            right_assignment = functional.one_hot(
                closure_right,
                num_classes=node_count,
            ).to(hidden.dtype)
            active_closures = closure_mask[:, :, None].to(hidden.dtype)
            left_assignment = left_assignment * active_closures
            right_assignment = right_assignment * active_closures
            message_to_left = right_hidden + closure_bond_hidden
            message_to_right = left_hidden + closure_bond_hidden
            closure_sum = torch.einsum(
                "bkn,bkd->bnd",
                left_assignment,
                message_to_left,
            )
            closure_sum += torch.einsum(
                "bkn,bkd->bnd",
                right_assignment,
                message_to_right,
            )
            closure_count = left_assignment.sum(dim=1) + right_assignment.sum(dim=1)
            closure_mean = closure_sum / closure_count[:, :, None].clamp(min=1.0)

            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden = global_hidden / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_hidden = global_hidden[:, None, :].expand_as(hidden)
            update = self.update(
                torch.cat(
                    (
                        hidden,
                        parent_hidden,
                        child_mean,
                        closure_mean,
                        global_hidden,
                    ),
                    dim=-1,
                )
            )
            return self.norm(hidden + update) * node_mask[:, :, None]

    class SparseWholeLipidFlow(nn.Module):
        """Production sparse whole-graph flow over tree and residual-closure state."""

        def __init__(
            self,
            *,
            node_classes: int,
            hidden_dim: int,
            layers: int,
            maximum_closures: int,
            maximum_heavy_atoms: int,
            dropout: float,
            bond_classes: int = 3,
            topology_context: bool = False,
            parent_distance_buckets: int = 0,
            closure_ring_size_buckets: int = 0,
            region_classes: int = 0,
            use_position_embedding: bool = False,
        ) -> None:
            super().__init__()
            if bond_classes not in {3, 4}:
                raise Phase1FlowError("sparse flow supports three or four bond classes")
            if topology_context and parent_distance_buckets < 1:
                raise Phase1FlowError(
                    "topology context requires at least one parent-distance bucket"
                )
            self.hidden_dim = hidden_dim
            self.topology_context = topology_context
            self.parent_distance_buckets = parent_distance_buckets
            self.closure_ring_size_buckets = closure_ring_size_buckets
            self.region_classes = region_classes
            self.bond_classes = bond_classes
            self.use_position_embedding = use_position_embedding
            self.node_embedding = nn.Embedding(node_classes, hidden_dim)
            self.position_embedding = (
                nn.Embedding(maximum_heavy_atoms, hidden_dim) if use_position_embedding else None
            )
            self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
            if region_classes:
                self.region_embedding = nn.Embedding(region_classes, hidden_dim)
                self.region_output = nn.Linear(hidden_dim, region_classes)
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            if topology_context:
                self.register_buffer(
                    "bond_valence_units",
                    BOND_VALENCE_UNITS[:bond_classes].to(torch.float32).clone(),
                    persistent=False,
                )
                self.topology_context_embedding = nn.Sequential(
                    nn.Linear(4, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.parent_distance_bias = nn.Parameter(torch.zeros(parent_distance_buckets + 1))
            self.blocks = nn.ModuleList(
                DeterministicSparseFlowBlock(hidden_dim, dropout) for _ in range(layers)
            )
            self.node_output = nn.Linear(hidden_dim, node_classes)
            self.parent_query = nn.Linear(hidden_dim, hidden_dim)
            self.parent_key = nn.Linear(hidden_dim, hidden_dim)
            self.backbone_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.closure_slots = nn.Embedding(maximum_closures, hidden_dim)
            self.closure_update = nn.Sequential(
                nn.Linear(5 * hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.closure_left_query = nn.Linear(hidden_dim, hidden_dim)
            self.closure_right_query = nn.Linear(hidden_dim, hidden_dim)
            self.closure_node_key = nn.Linear(hidden_dim, hidden_dim)
            self.closure_bond_output = nn.Linear(hidden_dim, bond_classes)
            self.node_count_logits = nn.Parameter(torch.zeros(maximum_heavy_atoms + 1))
            self.closure_count_logits = nn.Parameter(torch.zeros(maximum_closures + 1))
            if closure_ring_size_buckets:
                self.register_buffer(
                    "closure_ring_size_log_probabilities",
                    torch.zeros(closure_ring_size_buckets + 1),
                )

        def forward(
            self,
            nodes: Any,
            parents: Any,
            parent_bonds: Any,
            closure_left: Any,
            closure_right: Any,
            closure_bonds: Any,
            t: Any,
            node_mask: Any,
            child_mask: Any,
            closure_mask: Any,
            regions: Any | None = None,
            node_context: Any | None = None,
        ) -> dict[str, Any]:
            time_hidden = self.time_embedding(t[:, None])
            hidden = self.node_embedding(nodes) + time_hidden[:, None, :]
            if self.position_embedding is not None:
                if nodes.shape[1] > self.position_embedding.num_embeddings:
                    raise Phase1FlowError("node sequence exceeds position-embedding support")
                positions = torch.arange(nodes.shape[1], device=nodes.device)
                hidden += self.position_embedding(positions)[None, :, :]
            if node_context is not None:
                if node_context.shape != (*nodes.shape, self.hidden_dim):
                    raise Phase1FlowError(
                        "external node context must be [batch, nodes, hidden_dim]"
                    )
                hidden += node_context
            if self.region_classes:
                if regions is None:
                    raise Phase1FlowError("region-aware flow requires atom-region state")
                hidden += self.region_embedding(regions)
            hidden += self.bond_embedding(parent_bonds) * child_mask[:, :, None]
            if self.topology_context:
                hidden += self.topology_context_embedding(
                    _noisy_topology_features(
                        parents,
                        parent_bonds,
                        closure_left,
                        closure_right,
                        closure_bonds,
                        node_mask,
                        child_mask,
                        closure_mask,
                        self.bond_valence_units,
                    )
                )
            hidden = hidden * node_mask[:, :, None]
            closure_bond_hidden = self.bond_embedding(closure_bonds)
            closure_bond_hidden = closure_bond_hidden * closure_mask[:, :, None]
            for block in self.blocks:
                hidden = block(
                    hidden,
                    parents,
                    closure_left,
                    closure_right,
                    closure_bond_hidden,
                    node_mask,
                    child_mask,
                    closure_mask,
                )

            parent_query = self.parent_query(hidden)
            parent_key = self.parent_key(hidden)
            parent_logits = torch.einsum("bid,bjd->bij", parent_query, parent_key) / math.sqrt(
                self.hidden_dim
            )
            if self.topology_context:
                indices = torch.arange(hidden.shape[1], device=hidden.device)
                distances = indices[:, None] - indices[None, :]
                distance_buckets = torch.where(
                    distances > 0,
                    distances.clamp(max=self.parent_distance_buckets),
                    torch.zeros_like(distances),
                )
                parent_logits += self.parent_distance_bias[distance_buckets][None, :, :]
            parent_hidden = _gather_training_nodes(hidden, parents)
            parent_bond_logits = self.backbone_bond_output(
                torch.cat((hidden, parent_hidden), dim=-1)
            )

            batch, maximum_closures = closure_left.shape
            slots = self.closure_slots(
                torch.arange(
                    maximum_closures,
                    device=nodes.device,
                )
            )[
                None, :, :
            ].expand(batch, -1, -1)
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden = global_hidden / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            global_hidden = global_hidden[:, None, :].expand_as(slots)
            left_hidden = _gather_training_nodes(hidden, closure_left)
            right_hidden = _gather_training_nodes(hidden, closure_right)
            closure_hidden = self.closure_update(
                torch.cat(
                    (
                        slots,
                        global_hidden,
                        left_hidden,
                        right_hidden,
                        closure_bond_hidden,
                    ),
                    dim=-1,
                )
            )
            closure_key = self.closure_node_key(hidden)
            left_logits = torch.einsum(
                "bkd,bnd->bkn",
                self.closure_left_query(closure_hidden),
                closure_key,
            ) / math.sqrt(self.hidden_dim)
            right_logits = torch.einsum(
                "bkd,bnd->bkn",
                self.closure_right_query(closure_hidden),
                closure_key,
            ) / math.sqrt(self.hidden_dim)
            output = {
                "nodes": self.node_output(hidden),
                "parents": parent_logits,
                "parent_bonds": parent_bond_logits,
                "closure_left": left_logits,
                "closure_right": right_logits,
                "closure_bonds": self.closure_bond_output(closure_hidden),
                "node_count": self.node_count_logits[None, :].expand(batch, -1),
                "closure_count": self.closure_count_logits[None, :].expand(batch, -1),
            }
            if self.region_classes:
                output["regions"] = self.region_output(hidden)
            return output

else:  # pragma: no cover

    class SparseWholeLipidFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise Phase1FlowError("Phase 1 flow training requires torch")
