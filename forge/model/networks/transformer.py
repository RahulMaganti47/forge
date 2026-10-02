"""Reaction-program cross-attentive graph Transformer for vocabulary-free product flow."""

from __future__ import annotations

import math
from typing import Any, cast

from forge.model.conditioning.reaction_program import ReactionProgramVocabulary
from forge.model.networks.whole_lipid import SparseWholeLipidFlow, _gather_training_nodes

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


class ReactionProgramTransformerError(ValueError):
    """Transformer inputs or objective violate the declared semantic contract."""


if nn is not None:

    class _MaskedMultiheadAttention(nn.Module):
        def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
            super().__init__()
            if hidden_dim % heads:
                raise ReactionProgramTransformerError("hidden_dim must be divisible by heads")
            self.heads = heads
            self.head_dim = hidden_dim // heads
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def project_memory(self, memory: Any) -> tuple[Any, Any]:
            """Project one memory tensor into per-head keys and values.

            Split out so a caller whose memory is constant across many queries can project it once
            and hand the identical tensors back through ``memory_key_value``.  The arithmetic is
            the same expression that ``forward`` would evaluate, so reuse is bit-exact rather than
            merely equivalent.
            """

            batch, keys, _ = memory.shape
            k = self.key(memory).reshape(batch, keys, self.heads, self.head_dim).transpose(1, 2)
            v = self.value(memory).reshape(batch, keys, self.heads, self.head_dim).transpose(1, 2)
            return k, v

        def forward(
            self,
            query: Any,
            memory: Any,
            *,
            query_mask: Any,
            memory_mask: Any,
            attention_bias: Any | None = None,
            memory_key_value: tuple[Any, Any] | None = None,
        ) -> Any:
            batch, queries, hidden_dim = query.shape
            q = self.query(query).reshape(batch, queries, self.heads, self.head_dim).transpose(1, 2)
            if memory_key_value is None:
                k, v = self.project_memory(memory)
            else:
                k, v = memory_key_value
            keys = k.shape[2]
            if attention_bias is not None:
                if attention_bias.shape != (batch, self.heads, queries, keys):
                    raise ReactionProgramTransformerError("attention bias shape is inconsistent")
                attention_mask = attention_bias.masked_fill(
                    ~memory_mask[:, None, None, :], -torch.inf
                )
            else:
                attention_mask = memory_mask[:, None, None, :]
            context = functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            context = context.transpose(1, 2).reshape(batch, queries, hidden_dim)
            return self.output(context) * query_mask[:, :, None]

    class ReactionProgramTokenEncoder(nn.Module):
        """Encode global and atom-aligned reaction-program semantics as reusable memory tokens."""

        def __init__(
            self,
            vocabulary: ReactionProgramVocabulary,
            hidden_dim: int,
            heads: int,
            dropout: float,
            maximum_heavy_atoms: int,
            maximum_closures: int,
            repeat_group_conditioning: bool = False,
            role_morphology_conditioning: bool = False,
        ) -> None:
            super().__init__()
            self.vocabulary = vocabulary
            self.program = nn.Embedding(len(vocabulary.program_states), hidden_dim)
            self.role = nn.Embedding(len(vocabulary.role_states), hidden_dim)
            self.core = nn.Embedding(len(vocabulary.core_position_states), hidden_dim)
            self.depth = nn.Embedding(vocabulary.maximum_steps + 1, hidden_dim)
            self.token_type = nn.Embedding(3, hidden_dim)
            self.position = nn.Embedding(maximum_heavy_atoms, hidden_dim)
            self.repeat_group_conditioning = repeat_group_conditioning
            self.role_morphology_conditioning = role_morphology_conditioning
            self.role_morphology_embeddings = (
                nn.ModuleList(
                    (
                        nn.Embedding(maximum_heavy_atoms + 2, hidden_dim),
                        nn.Embedding(maximum_heavy_atoms + 2, hidden_dim),
                        nn.Embedding(maximum_closures + 2, hidden_dim),
                        nn.Embedding(maximum_heavy_atoms + 2, hidden_dim),
                    )
                )
                if role_morphology_conditioning
                else None
            )
            self.repeat_group = (
                nn.Embedding(len(vocabulary.role_states), hidden_dim)
                if repeat_group_conditioning
                else None
            )
            self.component_position = (
                nn.Embedding(maximum_heavy_atoms + 1, hidden_dim)
                if repeat_group_conditioning
                else None
            )
            self.self_attention = _MaskedMultiheadAttention(hidden_dim, heads, dropout)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.norm2 = nn.LayerNorm(hidden_dim)

        def forward(
            self,
            *,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            role_isolated_attention: bool = False,
            repeat_group_states: Any | None = None,
            component_position_states: Any | None = None,
            role_morphology_states: Any | None = None,
        ) -> tuple[Any, Any, Any]:
            batch, nodes = role_states.shape
            if (
                program_states.shape != (batch,)
                or core_position_states.shape != (batch, nodes)
                or program_depths.shape != (batch,)
                or adapter_mask.shape != (batch, nodes)
                or adapter_mask.dtype != torch.bool
            ):
                raise ReactionProgramTransformerError("reaction-program token shapes disagree")
            program_token = self.program(program_states) + self.token_type.weight[0]
            depth_token = self.depth(program_depths) + self.token_type.weight[1]
            positions = torch.arange(nodes, device=role_states.device)
            node_tokens = (
                self.role(role_states)
                + self.core(core_position_states)
                + self.position(positions)[None]
                + self.token_type.weight[2]
            )
            if self.repeat_group_conditioning:
                if (
                    repeat_group_states is None
                    or component_position_states is None
                    or repeat_group_states.shape != (batch, nodes)
                    or component_position_states.shape != (batch, nodes)
                ):
                    raise ReactionProgramTransformerError(
                        "repeat-aware reaction-program token shapes disagree"
                    )
                if self.repeat_group is None or self.component_position is None:
                    raise ReactionProgramTransformerError(
                        "repeat-aware embeddings are unexpectedly absent"
                    )
                node_tokens = (
                    node_tokens
                    + self.repeat_group(repeat_group_states)
                    + self.component_position(component_position_states)
                )
            if self.role_morphology_conditioning:
                if (
                    role_morphology_states is None
                    or role_morphology_states.shape != (batch, nodes, 4)
                    or self.role_morphology_embeddings is None
                ):
                    raise ReactionProgramTransformerError(
                        "role-local morphology token shapes disagree"
                    )
                for field, embedding in enumerate(self.role_morphology_embeddings):
                    values = role_morphology_states[:, :, field]
                    if torch.any(values < 0) or torch.any(values >= embedding.num_embeddings):
                        raise ReactionProgramTransformerError(
                            "role-local morphology state lies outside declared support"
                        )
                    node_tokens = node_tokens + embedding(values)
            tokens = torch.cat((program_token[:, None], depth_token[:, None], node_tokens), dim=1)
            token_mask = torch.cat(
                (
                    torch.ones((batch, 2), dtype=torch.bool, device=adapter_mask.device),
                    adapter_mask,
                ),
                dim=1,
            )
            normalized = self.norm1(tokens)
            attention_bias = None
            if role_isolated_attention:
                # Global program/depth tokens may not pool role-local state: otherwise they become
                # a hidden B/C -> A communication channel in the factorized control.  A role token
                # may read the shared globals and tokens from its own role only.
                token_roles = torch.cat(
                    (
                        role_states.new_full((batch, 2), -1),
                        role_states,
                    ),
                    dim=1,
                )
                query_roles = token_roles[:, :, None]
                memory_roles = token_roles[:, None, :]
                allowed = torch.where(
                    query_roles < 0,
                    memory_roles < 0,
                    (memory_roles < 0) | (memory_roles == query_roles),
                )
                attention_bias = torch.where(
                    allowed[:, None],
                    tokens.new_zeros(()),
                    tokens.new_full((), -1e9),
                ).expand(-1, self.self_attention.heads, -1, -1)
            attended = self.self_attention(
                normalized,
                normalized,
                query_mask=token_mask,
                memory_mask=token_mask,
                attention_bias=attention_bias,
            )
            tokens = tokens + attended
            tokens = tokens + self.ffn(self.norm2(tokens)) * token_mask[:, :, None]
            return tokens, token_mask, tokens[:, 0]

    class _RoutedAdapter(nn.Module):
        def __init__(
            self, hidden_dim: int, expert_count: int, adapter_dim: int, dropout: float
        ) -> None:
            super().__init__()
            if expert_count < 2 or adapter_dim < 1:
                raise ReactionProgramTransformerError("routed adapter support is invalid")
            self.experts = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(hidden_dim, adapter_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(adapter_dim, hidden_dim),
                )
                for _ in range(expert_count)
            )
            self.gate = nn.Linear(hidden_dim, expert_count)

        def gate_weights(self, program_summary: Any) -> Any:
            """Route on the program summary alone, so the routing is constant per program batch."""

            return torch.softmax(self.gate(program_summary), dim=-1)

        def forward(
            self, hidden: Any, program_summary: Any, *, weights: Any | None = None
        ) -> tuple[Any, Any]:
            if weights is None:
                weights = self.gate_weights(program_summary)
            expert_values = torch.stack([expert(hidden) for expert in self.experts], dim=2)
            update = torch.einsum("be,bned->bnd", weights, expert_values)
            return update, weights

    class _SharedAdapter(nn.Module):
        """One residual adapter used by the no-routing mechanism control."""

        def __init__(self, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
            super().__init__()
            self.adapter = nn.Sequential(
                nn.Linear(hidden_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, hidden_dim),
            )

        def gate_weights(self, program_summary: Any) -> Any:
            return program_summary.new_ones((program_summary.shape[0], 1))

        def forward(
            self, hidden: Any, program_summary: Any, *, weights: Any | None = None
        ) -> tuple[Any, Any]:
            if weights is None:
                weights = self.gate_weights(program_summary)
            return self.adapter(hidden), weights

    class _ZeroInitializedSpecialistAdapter(nn.Module):
        """One lightweight reaction specialist that is initially an exact identity delta."""

        def __init__(self, hidden_dim: int, adapter_dim: int, dropout: float) -> None:
            super().__init__()
            if adapter_dim < 1:
                raise ReactionProgramTransformerError(
                    "specialist adapter dimension must be positive"
                )
            self.adapter = nn.Sequential(
                nn.Linear(hidden_dim, adapter_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(adapter_dim, hidden_dim),
            )
            # A newly attached specialist must reproduce the authenticated shared checkpoint
            # exactly before its first update.  Zeroing only the terminal projection preserves a
            # useful random input projection while making the residual identically zero.
            nn.init.zeros_(self.adapter[-1].weight)
            nn.init.zeros_(self.adapter[-1].bias)

        def forward(self, hidden: Any) -> Any:
            return self.adapter(hidden)

    class ReactionProgramMemory:
        """Encoded program semantics plus every projection of them that a batch can reuse.

        The reaction program is clean context: program state, precursor roles, reaction-core
        positions, depth, adapter mask, repeat groups, component positions and role morphology.
        None of those move while the graph state is denoised, so for one sampling batch the encoded
        memory, each block's cross-attention keys and values, and each routed adapter's gate are
        the same tensors at every one of the flow steps.  Computing them once and handing back the
        identical tensors is bit-exact reuse, not an approximation.

        Modules are held by strong reference in the cache keys so an ``id`` can never be recycled
        onto a different module while this memory is alive.
        """

        __slots__ = ("tokens", "mask", "summary", "_key_values", "_adapter_weights", "_sources")

        def __init__(self, tokens: Any, mask: Any, summary: Any, sources: tuple[Any, ...]) -> None:
            self.tokens = tokens
            self.mask = mask
            self.summary = summary
            self._sources = sources
            self._key_values: dict[int, tuple[Any, tuple[Any, Any]]] = {}
            self._adapter_weights: dict[int, tuple[Any, Any]] = {}

        def matches(self, sources: tuple[Any, ...]) -> bool:
            """Require the exact conditioning tensors this memory was encoded from."""

            return len(sources) == len(self._sources) and all(
                left is right for left, right in zip(sources, self._sources, strict=True)
            )

        def key_value(self, attention: Any) -> tuple[Any, Any]:
            entry = self._key_values.get(id(attention))
            if entry is None:
                entry = (attention, attention.project_memory(self.tokens))
                self._key_values[id(attention)] = entry
            return entry[1]

        def adapter_weights(self, adapter: Any) -> Any:
            entry = self._adapter_weights.get(id(adapter))
            if entry is None:
                entry = (adapter, adapter.gate_weights(self.summary))
                self._adapter_weights[id(adapter)] = entry
            return entry[1]

    class ReactionProgramTransformerBlock(nn.Module):
        """Graph self-attention, program cross-attention and program-routed adaptation."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            heads: int,
            expert_count: int,
            adapter_dim: int,
            dropout: float,
            program_cross_attention: bool,
            routed_adapters: bool,
            role_isolated_attention: bool,
            specialist_adapter_dim: int = 0,
        ) -> None:
            super().__init__()
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attention = _MaskedMultiheadAttention(hidden_dim, heads, dropout)
            self.cross_norm = nn.LayerNorm(hidden_dim)
            self.cross_attention = _MaskedMultiheadAttention(hidden_dim, heads, dropout)
            self.ffn_norm = nn.LayerNorm(hidden_dim)
            self.ffn = nn.Sequential(
                nn.Linear(hidden_dim, 4 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * hidden_dim, hidden_dim),
            )
            self.program_cross_attention = program_cross_attention
            self.role_isolated_attention = role_isolated_attention
            self.routed_adapter = (
                _RoutedAdapter(hidden_dim, expert_count, adapter_dim, dropout)
                if routed_adapters
                else _SharedAdapter(hidden_dim, adapter_dim, dropout)
            )
            self.specialist_adapter = (
                _ZeroInitializedSpecialistAdapter(
                    hidden_dim,
                    specialist_adapter_dim,
                    dropout,
                )
                if specialist_adapter_dim > 0
                else None
            )
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            hidden: Any,
            *,
            node_mask: Any,
            program_tokens: Any,
            program_mask: Any,
            program_summary: Any,
            graph_bias: Any,
            role_states: Any,
            program_memory: Any | None = None,
        ) -> tuple[Any, Any]:
            normalized = self.self_norm(hidden)
            if self.role_isolated_attention:
                # Role zero is adapter-owned/unassigned state.  It remains isolated as well: a
                # shared assembly coordinate must not become a hidden cross-role message bus.
                same_role = role_states[:, :, None] == role_states[:, None, :]
                role_bias = torch.where(
                    same_role[:, None],
                    graph_bias.new_zeros(()),
                    graph_bias.new_full((), -1e9),
                )
                graph_bias = graph_bias + role_bias
            hidden = hidden + self.dropout(
                self.self_attention(
                    normalized,
                    normalized,
                    query_mask=node_mask,
                    memory_mask=node_mask,
                    attention_bias=graph_bias,
                )
            )
            if self.program_cross_attention:
                cross_bias = None
                if self.role_isolated_attention:
                    batch, nodes = role_states.shape
                    token_roles = torch.cat(
                        (
                            role_states.new_full((batch, 2), -1),
                            role_states,
                        ),
                        dim=1,
                    )
                    allowed = (token_roles[:, None, :] < 0) | (
                        token_roles[:, None, :] == role_states[:, :, None]
                    )
                    cross_bias = torch.where(
                        allowed[:, None],
                        hidden.new_zeros(()),
                        hidden.new_full((), -1e9),
                    ).expand(-1, self.cross_attention.heads, -1, -1)
                hidden = hidden + self.dropout(
                    self.cross_attention(
                        self.cross_norm(hidden),
                        program_tokens,
                        query_mask=node_mask,
                        memory_mask=program_mask,
                        attention_bias=cross_bias,
                        memory_key_value=(
                            None
                            if program_memory is None
                            else program_memory.key_value(self.cross_attention)
                        ),
                    )
                )
            normalized = self.ffn_norm(hidden)
            routed, weights = self.routed_adapter(
                normalized,
                program_summary,
                weights=(
                    None
                    if program_memory is None
                    else program_memory.adapter_weights(self.routed_adapter)
                ),
            )
            update = self.ffn(normalized) + routed
            if self.specialist_adapter is not None:
                update = update + self.specialist_adapter(normalized)
            hidden = hidden + self.dropout(update)
            return hidden * node_mask[:, :, None], weights

    class ReactionProgramGraphTransformer(nn.Module):
        """Sparse whole-product flow conditioned on reaction-program memory at every layer."""

        def __init__(
            self,
            *,
            vocabulary: ReactionProgramVocabulary,
            node_classes: int,
            hidden_dim: int,
            layers: int,
            heads: int,
            expert_count: int,
            adapter_dim: int,
            maximum_closures: int,
            maximum_heavy_atoms: int,
            dropout: float,
            bond_classes: int = 4,
            layerwise_program_cross_attention: bool = True,
            routed_adapters: bool = True,
            role_isolated_attention: bool = False,
            role_specific_parameters: bool = False,
            repeat_group_conditioning: bool = False,
            role_morphology_conditioning: bool = False,
            specialist_adapter_dim: int = 0,
            maximum_children: int = 0,
            program_routed_output_heads: bool = False,
        ) -> None:
            super().__init__()
            if layers < 1:
                raise ReactionProgramTransformerError("Transformer requires at least one layer")
            self.vocabulary = vocabulary
            self.hidden_dim = hidden_dim
            self.heads = heads
            self.maximum_heavy_atoms = maximum_heavy_atoms
            self.maximum_closures = maximum_closures
            self.layers = layers
            self.layerwise_program_cross_attention = layerwise_program_cross_attention
            self.routed_adapters = routed_adapters
            self.role_isolated_attention = role_isolated_attention
            self.role_specific_parameters = role_specific_parameters
            self.repeat_group_conditioning = repeat_group_conditioning
            self.role_morphology_conditioning = role_morphology_conditioning
            self.specialist_adapter_dim = specialist_adapter_dim
            self.maximum_children = maximum_children
            self.program_routed_output_heads = program_routed_output_heads
            if specialist_adapter_dim < 0:
                raise ReactionProgramTransformerError(
                    "specialist adapter dimension cannot be negative"
                )
            if maximum_children < 0:
                raise ReactionProgramTransformerError(
                    "maximum child-count support cannot be negative"
                )
            if role_specific_parameters and not role_isolated_attention:
                raise ReactionProgramTransformerError(
                    "role-specific parameters require role-isolated attention"
                )
            self.program_encoder = ReactionProgramTokenEncoder(
                vocabulary,
                hidden_dim,
                heads,
                dropout,
                maximum_heavy_atoms,
                maximum_closures,
                repeat_group_conditioning=repeat_group_conditioning,
                role_morphology_conditioning=role_morphology_conditioning,
            )
            # Reuse the qualified sparse state embeddings and output parameterization, but not its
            # message-passing blocks.
            self.state = SparseWholeLipidFlow(
                node_classes=node_classes,
                hidden_dim=hidden_dim,
                layers=0,
                maximum_closures=maximum_closures,
                maximum_heavy_atoms=maximum_heavy_atoms,
                dropout=dropout,
                bond_classes=bond_classes,
                use_position_embedding=True,
            )
            self.relation_bias = nn.Embedding(5, heads)
            self.blocks = (
                nn.ModuleList(
                    ReactionProgramTransformerBlock(
                        hidden_dim=hidden_dim,
                        heads=heads,
                        expert_count=expert_count,
                        adapter_dim=adapter_dim,
                        dropout=dropout,
                        # The input-only control receives the program once, in the first block.
                        program_cross_attention=(layerwise_program_cross_attention or index == 0),
                        routed_adapters=routed_adapters,
                        role_isolated_attention=role_isolated_attention,
                        specialist_adapter_dim=specialist_adapter_dim,
                    )
                    for index in range(layers)
                )
                if not role_specific_parameters
                else nn.ModuleList()
            )
            self.role_blocks = (
                nn.ModuleList(
                    nn.ModuleList(
                        ReactionProgramTransformerBlock(
                            hidden_dim=hidden_dim,
                            heads=heads,
                            expert_count=expert_count,
                            adapter_dim=adapter_dim,
                            dropout=dropout,
                            program_cross_attention=(
                                layerwise_program_cross_attention or index == 0
                            ),
                            routed_adapters=routed_adapters,
                            role_isolated_attention=True,
                            specialist_adapter_dim=specialist_adapter_dim,
                        )
                        for index in range(layers)
                    )
                    for _ in vocabulary.role_states
                )
                if role_specific_parameters
                else None
            )
            self.role_output = nn.Linear(hidden_dim, len(vocabulary.role_states))
            self.core_output = nn.Linear(hidden_dim, len(vocabulary.core_position_states))
            self.offspring_output = (
                nn.Linear(hidden_dim, maximum_children + 1) if maximum_children > 0 else None
            )
            self.terminal_chemistry_adapter = (
                _RoutedAdapter(hidden_dim, expert_count, adapter_dim, dropout)
                if program_routed_output_heads
                else None
            )
            self.closure_output_adapter = (
                _RoutedAdapter(hidden_dim, expert_count, adapter_dim, dropout)
                if program_routed_output_heads
                else None
            )

        def _graph_bias(
            self,
            parents: Any,
            closure_left: Any,
            closure_right: Any,
            child_mask: Any,
            closure_mask: Any,
        ) -> Any:
            batch, nodes = parents.shape
            relations = torch.zeros((batch, nodes, nodes), dtype=torch.long, device=parents.device)
            diagonal = torch.arange(nodes, device=parents.device)
            relations[:, diagonal, diagonal] = 1
            batch_nodes = torch.arange(batch, device=parents.device)[:, None].expand(batch, nodes)
            child_nodes = diagonal[None].expand(batch, nodes)
            relations[batch_nodes[child_mask], child_nodes[child_mask], parents[child_mask]] = 2
            relations[batch_nodes[child_mask], parents[child_mask], child_nodes[child_mask]] = 3
            closure_slots = closure_left.shape[1]
            closure_batches = torch.arange(batch, device=parents.device)[:, None].expand(
                batch, closure_slots
            )
            active_batches = closure_batches[closure_mask]
            active_left = closure_left[closure_mask]
            active_right = closure_right[closure_mask]
            relations[active_batches, active_left, active_right] = 4
            relations[active_batches, active_right, active_left] = 4
            return self.relation_bias(relations).permute(0, 3, 1, 2)

        @staticmethod
        def _program_memory_sources(
            *,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            repeat_group_states: Any | None,
            component_position_states: Any | None,
            role_morphology_states: Any | None,
        ) -> tuple[Any, ...]:
            return (
                program_states,
                role_states,
                core_position_states,
                program_depths,
                adapter_mask,
                repeat_group_states,
                component_position_states,
                role_morphology_states,
            )

        def prepare_program_memory(
            self,
            *,
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            repeat_group_states: Any | None = None,
            component_position_states: Any | None = None,
            role_morphology_states: Any | None = None,
        ) -> Any:
            """Encode the batch's reaction program once for reuse across its denoising steps.

            Hand the result to ``forward(..., program_memory=...)`` with the *same* conditioning
            tensor objects.  ``forward`` verifies that identity and fails closed otherwise, so a
            stale memory cannot silently condition a different batch.
            """

            sources = self._program_memory_sources(
                program_states=program_states,
                role_states=role_states,
                core_position_states=core_position_states,
                program_depths=program_depths,
                adapter_mask=adapter_mask,
                repeat_group_states=repeat_group_states,
                component_position_states=component_position_states,
                role_morphology_states=role_morphology_states,
            )
            tokens, mask, summary = self.program_encoder(
                program_states=program_states,
                role_states=role_states,
                core_position_states=core_position_states,
                program_depths=program_depths,
                adapter_mask=adapter_mask,
                role_isolated_attention=self.role_isolated_attention,
                repeat_group_states=repeat_group_states,
                component_position_states=component_position_states,
                role_morphology_states=role_morphology_states,
            )
            return ReactionProgramMemory(tokens, mask, summary, sources)

        def forward(
            self,
            *,
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
            program_states: Any,
            role_states: Any,
            core_position_states: Any,
            program_depths: Any,
            adapter_mask: Any,
            repeat_group_states: Any | None = None,
            component_position_states: Any | None = None,
            component_instance_states: Any | None = None,
            role_morphology_states: Any | None = None,
            program_memory: Any | None = None,
        ) -> dict[str, Any]:
            del component_instance_states  # Loss-only coordinate; never a learned identity token.
            if program_memory is None:
                program_memory = self.prepare_program_memory(
                    program_states=program_states,
                    role_states=role_states,
                    core_position_states=core_position_states,
                    program_depths=program_depths,
                    adapter_mask=adapter_mask,
                    repeat_group_states=repeat_group_states,
                    component_position_states=component_position_states,
                    role_morphology_states=role_morphology_states,
                )
            elif not program_memory.matches(
                self._program_memory_sources(
                    program_states=program_states,
                    role_states=role_states,
                    core_position_states=core_position_states,
                    program_depths=program_depths,
                    adapter_mask=adapter_mask,
                    repeat_group_states=repeat_group_states,
                    component_position_states=component_position_states,
                    role_morphology_states=role_morphology_states,
                )
            ):
                raise ReactionProgramTransformerError(
                    "supplied program memory was encoded from different conditioning tensors"
                )
            program_tokens = program_memory.tokens
            program_mask = program_memory.mask
            program_summary = program_memory.summary
            hidden = (
                self.state.node_embedding(nodes) + self.state.time_embedding(t[:, None])[:, None]
            )
            positions = torch.arange(nodes.shape[1], device=nodes.device)
            position_embedding = self.state.position_embedding
            if position_embedding is None:
                raise ReactionProgramTransformerError("position embedding is unexpectedly absent")
            hidden = hidden + position_embedding(positions)[None]
            hidden = hidden + self.state.bond_embedding(parent_bonds) * child_mask[:, :, None]
            hidden = hidden * node_mask[:, :, None]
            graph_bias = self._graph_bias(
                parents, closure_left, closure_right, child_mask, closure_mask
            )
            expert_weights = []
            if self.role_blocks is None:
                for block in self.blocks:
                    hidden, weights = block(
                        hidden,
                        node_mask=node_mask,
                        program_tokens=program_tokens,
                        program_mask=program_mask,
                        program_summary=program_summary,
                        graph_bias=graph_bias,
                        role_states=role_states,
                        program_memory=program_memory,
                    )
                    expert_weights.append(weights)
            else:
                # Each semantic role has its own full-depth denoiser. Roles are scattered back only
                # onto their own nodes, so no role-specific parameters can alter another role.
                for layer_index in range(self.layers):
                    layer_weights = []
                    for role_index, role_stack in enumerate(self.role_blocks):
                        role_stack = cast(Any, role_stack)
                        role_mask = node_mask & (role_states == role_index)
                        candidate, weights = role_stack[layer_index](
                            hidden,
                            node_mask=role_mask,
                            program_tokens=program_tokens,
                            program_mask=program_mask,
                            program_summary=program_summary,
                            graph_bias=graph_bias,
                            role_states=role_states,
                            program_memory=program_memory,
                        )
                        hidden = torch.where(role_mask[:, :, None], candidate, hidden)
                        layer_weights.append(weights)
                    expert_weights.append(torch.stack(layer_weights, dim=1).mean(dim=1))

            chemistry_hidden = hidden
            terminal_expert_weights = None
            if self.terminal_chemistry_adapter is not None:
                child_counts = torch.zeros_like(parents)
                batch_indices = torch.arange(parents.shape[0], device=parents.device)[:, None]
                batch_indices = batch_indices.expand_as(parents)
                child_counts.index_put_(
                    (batch_indices[child_mask], parents[child_mask]),
                    torch.ones_like(parents[child_mask]),
                    accumulate=True,
                )
                terminal_mask = node_mask & (child_counts == 0)
                terminal_update, terminal_expert_weights = self.terminal_chemistry_adapter(
                    hidden,
                    program_summary,
                    weights=program_memory.adapter_weights(self.terminal_chemistry_adapter),
                )
                chemistry_hidden = hidden + terminal_update * terminal_mask[:, :, None]

            parent_logits = torch.einsum(
                "bid,bjd->bij", self.state.parent_query(hidden), self.state.parent_key(hidden)
            ) / math.sqrt(self.hidden_dim)
            if self.role_isolated_attention:
                same_role = role_states[:, :, None] == role_states[:, None, :]
                # Independent precursor regions are joined only at declared reaction-core atoms.
                # Exterior cross-role parent choices remain forbidden.
                core_attachment = (core_position_states[:, :, None] > 1) & (
                    core_position_states[:, None, :] > 1
                )
                parent_logits = parent_logits.masked_fill(~(same_role | core_attachment), -1e9)
            parent_hidden = _gather_training_nodes(chemistry_hidden, parents)
            parent_bond_logits = self.state.backbone_bond_output(
                torch.cat((chemistry_hidden, parent_hidden), dim=-1)
            )
            batch, maximum_closures = closure_left.shape
            slots = self.state.closure_slots(torch.arange(maximum_closures, device=nodes.device))[
                None
            ].expand(batch, -1, -1)
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden /= node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            left_hidden = _gather_training_nodes(hidden, closure_left)
            right_hidden = _gather_training_nodes(hidden, closure_right)
            closure_bond_hidden = self.state.bond_embedding(closure_bonds)
            closure_bond_hidden *= closure_mask[:, :, None]
            closure_hidden = self.state.closure_update(
                torch.cat(
                    (
                        slots,
                        global_hidden[:, None].expand_as(slots),
                        left_hidden,
                        right_hidden,
                        closure_bond_hidden,
                    ),
                    dim=-1,
                )
            )
            closure_expert_weights = None
            if self.closure_output_adapter is not None:
                closure_update, closure_expert_weights = self.closure_output_adapter(
                    closure_hidden,
                    program_summary,
                    weights=program_memory.adapter_weights(self.closure_output_adapter),
                )
                closure_hidden = closure_hidden + closure_update
            closure_key = self.state.closure_node_key(hidden)
            left_logits = torch.einsum(
                "bkd,bnd->bkn", self.state.closure_left_query(closure_hidden), closure_key
            ) / math.sqrt(self.hidden_dim)
            right_logits = torch.einsum(
                "bkd,bnd->bkn", self.state.closure_right_query(closure_hidden), closure_key
            ) / math.sqrt(self.hidden_dim)
            output = {
                "nodes": self.state.node_output(chemistry_hidden),
                "parents": parent_logits,
                "parent_bonds": parent_bond_logits,
                "closure_left": left_logits,
                "closure_right": right_logits,
                "closure_bonds": self.state.closure_bond_output(closure_hidden),
                "node_count": self.state.node_count_logits[None].expand(batch, -1),
                "closure_count": self.state.closure_count_logits[None].expand(batch, -1),
                "role_states": self.role_output(hidden),
                "core_position_states": self.core_output(hidden),
                "expert_weights": torch.stack(expert_weights, dim=1),
            }
            if self.offspring_output is not None:
                output["offspring"] = self.offspring_output(hidden)
            if terminal_expert_weights is not None and closure_expert_weights is not None:
                output["terminal_chemistry_expert_weights"] = terminal_expert_weights
                output["closure_output_expert_weights"] = closure_expert_weights
            return output

else:  # pragma: no cover

    class ReactionProgramGraphTransformer:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise ReactionProgramTransformerError("graph Transformer requires torch")


__all__ = ["ReactionProgramGraphTransformer", "ReactionProgramTransformerError"]
