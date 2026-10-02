"""Chemistry-aware joint sparse flow for Ugi product exteriors.

This arm tests one precise architectural question.  It retains the same
core-anchored preorder tree language and global morphology program as the
staged generator, but denoises offspring, atom, and parent-bond states in one
shared sequence model.  Component identifiers and route labels are excluded.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forge.model.networks.ugi_morphology import (
    _layout_from_programs,
    _pending_by_role,
)
from forge.model.representation.ugi_morphology import (
    UgiMorphologyProgram,
    preorder_attached_forest_to_parents,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None
    pack_padded_sequence = None
    pad_packed_sequence = None


class UgiJointSparseFlowError(RuntimeError):
    """Raised when joint sparse tensors violate the adapter contract."""


UGI_TREE_RELATIONS = (
    "unrelated",
    "self",
    "query_child_of_memory",
    "query_parent_of_memory",
    "siblings",
    "query_descendant_of_memory",
    "query_ancestor_of_memory",
    "same_role",
    "different_role",
)
(
    TREE_UNRELATED,
    TREE_SELF,
    TREE_CHILD,
    TREE_PARENT,
    TREE_SIBLING,
    TREE_DESCENDANT,
    TREE_ANCESTOR,
    TREE_SAME_ROLE,
    TREE_DIFFERENT_ROLE,
) = range(len(UGI_TREE_RELATIONS))


@dataclass(frozen=True)
class UgiJointSparseRecord:
    """One core-excluded sparse sequence with joint topology and chemistry."""

    product_id: str
    program: UgiMorphologyProgram
    full_indices: np.ndarray
    offspring: np.ndarray
    atom_states: np.ndarray
    parent_bond_states: np.ndarray
    closure_left: np.ndarray
    closure_right: np.ndarray
    closure_bond_states: np.ndarray
    decoration_anchors: np.ndarray
    decoration_atom_states: np.ndarray
    decoration_bond_states: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.offspring.size)


@dataclass(frozen=True)
class UgiJointSparseTerminal:
    """Generated valid trees, flow endpoints and terminal correction logits."""

    program: UgiMorphologyProgram
    offspring: tuple[np.ndarray, np.ndarray, np.ndarray]
    atom_logits: np.ndarray
    parent_bond_logits: np.ndarray
    decoration_anchor_logits: np.ndarray
    decoration_atom_logits: np.ndarray
    decoration_bond_logits: np.ndarray
    hidden: np.ndarray
    flow_endpoint_atom_states: np.ndarray | None = None
    flow_endpoint_parent_bond_states: np.ndarray | None = None
    flow_endpoint_decoration_anchors: np.ndarray | None = None
    flow_endpoint_decoration_atom_states: np.ndarray | None = None
    flow_endpoint_decoration_bond_states: np.ndarray | None = None


# "none" is retained as an alias of flat_no_role so earlier callers keep working.
SEMANTIC_ORGANIZATIONS = (
    "role_structured",  # production: role-blocked layout plus the role map
    "flat_no_role",  # flat layout, constant null tag, full embedding machinery retained
    "flat_true_role",  # flat layout plus the true role map
    "flat_misaligned_role",  # flat layout, labels permuted across nodes at matched counts
    "none",
)
FLAT_ORGANIZATIONS = ("flat_no_role", "flat_true_role", "flat_misaligned_role", "none")
NULL_ROLE_INDEX = len(ROLE_NAMES)


def _validated_semantic_organization(value: str) -> str:
    if value not in SEMANTIC_ORGANIZATIONS:
        raise UgiJointSparseFlowError(
            f"unsupported semantic_organization {value!r}; expected one of {SEMANTIC_ORGANIZATIONS}"
        )
    return value


def flat_subtree_permutation(record: UgiJointSparseRecord) -> np.ndarray:
    """Reorder whole preorder subtrees so role stops being a contiguous block.

    The role-structured projection concatenates three per-role preorder attached forests in role
    order, so roles occupy contiguous spans purely as an artefact of that concatenation. A
    concatenation of preorder forests is itself a preorder forest, so reordering whole trees yields
    another valid forest over the same atoms.

    Trees are keyed on their own CONTENT: size, then atom states, then parent-bond states, then
    child counts. Content is the right key and the product atom index is not. Measured on the
    calibration fold, ordering by root atom index left role recoverable at 0.69 against a 0.44
    baseline, a lift of +0.26, because product atom indexing is template-derived and therefore
    correlates with role. Content alone carries a lift of only +0.04, so a content-derived ordering
    is close to role-agnostic while still being a canonical serialization of the kind any
    whole-molecule generator would use. Verified as a pure permutation over the entire train fold in
    `scripts/phase1_verify_flat_projection_invariant_v1.py`: molecular identity, atom, bond and
    offspring targets, closures, decorations and all four program vectors are preserved, and 0 of
    66,464 flat layouts retain program-implied role blocks.
    """
    attachment_total = int(sum(record.program.attachment_counts))
    parents = preorder_attached_forest_to_parents(
        record.offspring, attachment_count=attachment_total
    )
    roots = [index for index, parent in enumerate(parents.tolist()) if parent < 0]
    if len(roots) != attachment_total:
        raise UgiJointSparseFlowError("flat layout found the wrong number of exterior roots")
    bounds = roots + [int(record.offspring.size)]
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(roots))]

    def content_key(span: tuple[int, int]) -> tuple:
        start, end = span
        return (
            end - start,
            tuple(record.atom_states[start:end].tolist()),
            tuple(record.parent_bond_states[start:end].tolist()),
            tuple(record.offspring[start:end].tolist()),
        )

    ordered = sorted(spans, key=content_key)
    return np.concatenate([np.arange(start, end, dtype=np.int64) for start, end in ordered])


def _true_role_of_position(record: UgiJointSparseRecord) -> np.ndarray:
    return np.concatenate(
        [
            np.full(count, index, dtype=np.int64)
            for index, count in enumerate(record.program.node_counts)
        ]
    )


def collate_ugi_joint_sparse_records(
    records: Sequence[UgiJointSparseRecord],
    *,
    maximum_nodes: int,
    maximum_children: int,
    maximum_closures: int,
    maximum_decorations: int,
    semantic_organization: str = "role_structured",
) -> dict[str, Any]:
    """Collate joint targets without inserting the adapter-owned core atoms.

    Under `semantic_organization="none"` the layout is flattened in place: whole preorder subtrees
    are reordered by a molecular key, positions become absolute rather than per-role, and
    `role_states` carries the true permuted role for analysis only. Both arms therefore see the
    identical records in the identical order, which is what the parity contract requires.
    """

    if torch is None or not records:
        raise UgiJointSparseFlowError("joint sparse collation requires records and torch")
    organization = _validated_semantic_organization(semantic_organization)
    flat = organization in FLAT_ORGANIZATIONS
    programs = tuple(record.program for record in records)
    batch = _layout_from_programs(programs, maximum_nodes=maximum_nodes)
    shape = batch["role_states"].shape
    offspring = torch.zeros(shape, dtype=torch.long)
    nodes = torch.zeros(shape, dtype=torch.long)
    parent_bonds = torch.zeros(shape, dtype=torch.long)
    closure_mask = torch.zeros((len(records), maximum_closures), dtype=torch.bool)
    closure_left = torch.zeros((len(records), maximum_closures), dtype=torch.long)
    closure_right = torch.zeros_like(closure_left)
    closure_bonds = torch.zeros_like(closure_left)
    decoration_anchors = torch.zeros((len(records), maximum_decorations), dtype=torch.long)
    decoration_atoms = torch.zeros_like(decoration_anchors)
    decoration_bonds = torch.zeros_like(decoration_anchors)
    for index, record in enumerate(records):
        if (
            record.node_count > maximum_nodes
            or record.offspring.max(initial=0) > maximum_children
            or record.closure_left.size > maximum_closures
            or record.decoration_anchors.size > maximum_decorations
        ):
            raise UgiJointSparseFlowError("joint target exceeds declared tensor support")
        count = record.node_count
        if flat:
            permutation = flat_subtree_permutation(record)
            inverse = np.empty_like(permutation)
            inverse[permutation] = np.arange(permutation.size, dtype=np.int64)
            record_offspring = record.offspring[permutation]
            record_atoms = record.atom_states[permutation]
            record_bonds = record.parent_bond_states[permutation]
            record_left = (
                inverse[record.closure_left] if record.closure_left.size else record.closure_left
            )
            record_right = (
                inverse[record.closure_right] if record.closure_right.size else record.closure_right
            )
            record_anchors = (
                inverse[record.decoration_anchors]
                if record.decoration_anchors.size
                else record.decoration_anchors
            )
            # Absolute sequence position, and the true role kept for analysis only.
            batch["within_role_positions"][index, :count] = torch.arange(count)
            true_roles = _true_role_of_position(record)[permutation]
            if organization == "flat_misaligned_role":
                # Permute labels ACROSS nodes while preserving each label's count exactly, fixed
                # deterministically per example by the product id. Dimensionality and marginal
                # frequencies survive; scientific alignment does not. Per-example RELABELLING was
                # rejected: the program carries the ordered per-role sizes, distinct in 91.5% of
                # molecules, so the model could reverse-engineer which renamed block is which.
                seed = int(hashlib.sha256(record.product_id.encode()).hexdigest()[:8], 16)
                shuffled = np.random.default_rng(seed).permutation(true_roles)
                batch["role_states"][index, :count] = torch.from_numpy(shuffled.copy())
            else:
                batch["role_states"][index, :count] = torch.from_numpy(true_roles.copy())
        else:
            record_offspring = record.offspring
            record_atoms = record.atom_states
            record_bonds = record.parent_bond_states
            record_left = record.closure_left
            record_right = record.closure_right
            record_anchors = record.decoration_anchors
        offspring[index, :count] = torch.from_numpy(record_offspring.copy())
        nodes[index, :count] = torch.from_numpy(record_atoms.copy())
        parent_bonds[index, :count] = torch.from_numpy(record_bonds.copy())
        closure_count = record.closure_left.size
        closure_mask[index, :closure_count] = True
        closure_left[index, :closure_count] = torch.from_numpy(record_left.copy())
        closure_right[index, :closure_count] = torch.from_numpy(record_right.copy())
        closure_bonds[index, :closure_count] = torch.from_numpy(record.closure_bond_states.copy())
        decoration_count = record.decoration_anchors.size
        decoration_anchors[index, :decoration_count] = torch.from_numpy(record_anchors.copy()) + 1
        decoration_atoms[index, :decoration_count] = torch.from_numpy(
            record.decoration_atom_states.copy()
        )
        decoration_bonds[index, :decoration_count] = torch.from_numpy(
            record.decoration_bond_states.copy()
        )
    batch.update(
        {
            "offspring": offspring,
            "nodes": nodes,
            "parent_bonds": parent_bonds,
            "closure_mask": closure_mask,
            "closure_left": closure_left,
            "closure_right": closure_right,
            "closure_bonds": closure_bonds,
            "decoration_anchors": decoration_anchors,
            "decoration_atoms": decoration_atoms,
            "decoration_bonds": decoration_bonds,
            "decoration_present_mask": decoration_anchors > 0,
        }
    )
    return batch


def noisy_preorder_relation_states(
    offspring: Any,
    role_states: Any,
    node_mask: Any,
) -> Any:
    """Infer fail-soft preorder relations from the current noisy offspring state.

    A discrete-flow intermediate is not guaranteed to be a valid forest.  Relation construction
    therefore cannot call the strict terminal decoder or use the clean target tree.  For every
    current node, this function finds the first preorder balance closure within the same role.  If
    none exists, the subtree is conservatively extended to that role's final active node.  The
    resulting relation tensor is deterministic, uses only model-visible state, and becomes exact
    whenever the offspring word is a valid core-attached preorder forest.
    """

    if torch is None:
        raise UgiJointSparseFlowError("tree relations require torch")
    if (
        offspring.ndim != 2
        or role_states.shape != offspring.shape
        or node_mask.shape != offspring.shape
        or node_mask.dtype != torch.bool
    ):
        raise UgiJointSparseFlowError("tree-relation tensors are misaligned")
    batch, nodes = offspring.shape
    positions = torch.arange(nodes, device=offspring.device)
    active = node_mask
    same_role = role_states[:, :, None] == role_states[:, None, :]
    active_pairs = active[:, :, None] & active[:, None, :]

    delta = (offspring.to(torch.long) - 1) * active
    prefix = torch.cumsum(delta, dim=1)
    target = prefix - delta - 1
    possible_end = (
        active_pairs
        & same_role
        & (positions[None, None, :] >= positions[None, :, None])
        & (prefix[:, None, :] == target[:, :, None])
    )
    end_candidates = torch.where(
        possible_end,
        positions[None, None, :],
        positions.new_full((), nodes),
    )
    first_end = end_candidates.amin(dim=2)
    last_role_node = torch.where(
        active_pairs & same_role,
        positions[None, None, :],
        positions.new_full((), -1),
    ).amax(dim=2)
    subtree_end = torch.where(first_end < nodes, first_end, last_role_node)

    before = positions[None, :, None] < positions[None, None, :]
    descendants = (
        active_pairs & same_role & before & (positions[None, None, :] <= subtree_end[:, :, None])
    )
    ancestor_indices = torch.where(
        descendants,
        positions[None, :, None],
        positions.new_full((), -1),
    )
    parent_by_child = ancestor_indices.amax(dim=1)

    query_positions = positions[None, :, None]
    memory_positions = positions[None, None, :]
    query_parent = parent_by_child[:, :, None]
    memory_parent = parent_by_child[:, None, :]
    query_is_child = query_parent == memory_positions
    query_is_parent = memory_parent == query_positions
    siblings = (
        (query_parent >= 0)
        & (query_parent == memory_parent)
        & (query_positions != memory_positions)
    )
    query_is_descendant = descendants.transpose(1, 2)
    query_is_ancestor = descendants

    relations = torch.full(
        (batch, nodes, nodes),
        TREE_UNRELATED,
        dtype=torch.long,
        device=offspring.device,
    )
    relations = torch.where(active_pairs & ~same_role, TREE_DIFFERENT_ROLE, relations)
    relations = torch.where(active_pairs & same_role, TREE_SAME_ROLE, relations)
    relations = torch.where(query_is_descendant, TREE_DESCENDANT, relations)
    relations = torch.where(query_is_ancestor, TREE_ANCESTOR, relations)
    relations = torch.where(siblings, TREE_SIBLING, relations)
    relations = torch.where(query_is_parent, TREE_PARENT, relations)
    relations = torch.where(query_is_child, TREE_CHILD, relations)
    diagonal = positions[None, :, None] == positions[None, None, :]
    relations = torch.where(active_pairs & diagonal, TREE_SELF, relations)
    return relations


if nn is not None:

    class _UgiProgramTransformerBlock(nn.Module):
        """Bidirectional exterior attention with morphology-program cross-attention."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            heads: int,
            dropout: float,
            feedforward_multiplier: int,
        ) -> None:
            super().__init__()
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attention = nn.MultiheadAttention(
                hidden_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.program_norm = nn.LayerNorm(hidden_dim)
            self.program_attention = nn.MultiheadAttention(
                hidden_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            self.feedforward_norm = nn.LayerNorm(hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_dim, feedforward_multiplier * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_multiplier * hidden_dim, hidden_dim),
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, hidden: Any, *, node_mask: Any, program_tokens: Any) -> Any:
            normalized = self.self_norm(hidden)
            attended, _ = self.self_attention(
                normalized,
                normalized,
                normalized,
                key_padding_mask=~node_mask,
                need_weights=False,
            )
            hidden = (hidden + self.dropout(attended)) * node_mask[:, :, None]
            normalized = self.program_norm(hidden)
            attended, _ = self.program_attention(
                normalized,
                program_tokens,
                program_tokens,
                need_weights=False,
            )
            hidden = (hidden + self.dropout(attended)) * node_mask[:, :, None]
            hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
            return hidden * node_mask[:, :, None]

    class _UgiTreeProgramTransformerBlock(nn.Module):
        """Tree-biased exterior attention with role-routed program memory."""

        def __init__(
            self,
            *,
            hidden_dim: int,
            heads: int,
            dropout: float,
            feedforward_multiplier: int,
            maximum_relative_position: int,
            use_tree_relations: bool,
            role_routed_program_attention: bool,
            role_adapter_dim: int,
        ) -> None:
            super().__init__()
            if maximum_relative_position < 1 or role_adapter_dim < 0:
                raise UgiJointSparseFlowError("invalid tree-Transformer structural support")
            # Preserve the import surface of historical Ugi runners that load this module but never
            # construct the challenger.  The structural primitive is needed only by this backbone.
            from forge.model.networks.attention import BiasedMultiheadAttention

            self.heads = heads
            self.maximum_relative_position = maximum_relative_position
            self.use_tree_relations = use_tree_relations
            self.role_routed_program_attention = role_routed_program_attention
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attention = BiasedMultiheadAttention(hidden_dim, heads, dropout)
            self.relation_bias = nn.Embedding(len(UGI_TREE_RELATIONS), heads)
            # Signed within-role sequence distance plus one cross-role state.
            self.relative_position_bias = nn.Embedding(2 * maximum_relative_position + 2, heads)
            self.program_norm = nn.LayerNorm(hidden_dim)
            self.program_attention = BiasedMultiheadAttention(hidden_dim, heads, dropout)
            self.adapter_norm = nn.LayerNorm(hidden_dim)
            self.role_adapters = (
                nn.ModuleList(
                    nn.Sequential(
                        nn.Linear(hidden_dim, role_adapter_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                        nn.Linear(role_adapter_dim, hidden_dim),
                    )
                    for _ in ROLE_NAMES
                )
                if role_adapter_dim
                else None
            )
            self.feedforward_norm = nn.LayerNorm(hidden_dim)
            self.feedforward = nn.Sequential(
                nn.Linear(hidden_dim, feedforward_multiplier * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feedforward_multiplier * hidden_dim, hidden_dim),
            )
            self.dropout = nn.Dropout(dropout)
            nn.init.zeros_(self.relation_bias.weight)
            nn.init.zeros_(self.relative_position_bias.weight)

        def _self_attention_bias(
            self,
            relation_states: Any,
            within_role_positions: Any,
            role_states: Any,
        ) -> Any:
            batch, nodes = role_states.shape
            if relation_states.shape != (batch, nodes, nodes):
                raise UgiJointSparseFlowError("tree relation states are misaligned")
            if not self.use_tree_relations:
                return relation_states.new_zeros(
                    (batch, self.heads, nodes, nodes), dtype=self.relation_bias.weight.dtype
                )
            relative = within_role_positions[:, None, :] - within_role_positions[:, :, None]
            relative = (
                relative.clamp(
                    min=-self.maximum_relative_position,
                    max=self.maximum_relative_position,
                )
                + self.maximum_relative_position
            )
            same_role = role_states[:, :, None] == role_states[:, None, :]
            cross_role_state = 2 * self.maximum_relative_position + 1
            relative = torch.where(same_role, relative, cross_role_state)
            relation = self.relation_bias(relation_states).permute(0, 3, 1, 2)
            position = self.relative_position_bias(relative).permute(0, 3, 1, 2)
            return relation + position

        def _program_attention_bias(self, role_states: Any, program_tokens: Any) -> Any | None:
            if not self.role_routed_program_attention:
                return None
            if program_tokens.shape[1] != len(ROLE_NAMES) + 1:
                raise UgiJointSparseFlowError("tree Transformer program memory changed")
            token_indices = torch.arange(program_tokens.shape[1], device=role_states.device)
            allowed = (token_indices[None, None, :] == 0) | (
                token_indices[None, None, :] == role_states[:, :, None] + 1
            )
            return torch.where(
                allowed[:, None],
                program_tokens.new_zeros(()),
                program_tokens.new_full((), -torch.inf),
            )

        def _role_adapter(self, hidden: Any, role_states: Any) -> Any:
            if self.role_adapters is None:
                return torch.zeros_like(hidden)
            normalized = self.adapter_norm(hidden)
            values = torch.stack([adapter(normalized) for adapter in self.role_adapters], dim=2)
            selected = role_states[:, :, None, None].expand(-1, -1, 1, hidden.shape[-1])
            return torch.gather(values, 2, selected).squeeze(2)

        def forward(
            self,
            hidden: Any,
            *,
            node_mask: Any,
            program_tokens: Any,
            relation_states: Any,
            within_role_positions: Any,
            role_states: Any,
        ) -> Any:
            normalized = self.self_norm(hidden)
            hidden = hidden + self.dropout(
                self.self_attention(
                    normalized,
                    normalized,
                    query_mask=node_mask,
                    memory_mask=node_mask,
                    attention_bias=self._self_attention_bias(
                        relation_states,
                        within_role_positions,
                        role_states,
                    ),
                )
            )
            hidden = hidden * node_mask[:, :, None]
            hidden = hidden + self.dropout(
                self.program_attention(
                    self.program_norm(hidden),
                    program_tokens,
                    query_mask=node_mask,
                    memory_mask=torch.ones(
                        program_tokens.shape[:2], dtype=torch.bool, device=program_tokens.device
                    ),
                    attention_bias=self._program_attention_bias(role_states, program_tokens),
                )
            )
            hidden = hidden + self.dropout(self._role_adapter(hidden, role_states))
            hidden = hidden + self.dropout(self.feedforward(self.feedforward_norm(hidden)))
            return hidden * node_mask[:, :, None]

    class UgiJointSparseFlow(nn.Module):
        """Shared sequence denoiser for topology and chemistry variables."""

        def __init__(
            self,
            *,
            maximum_children: int,
            atom_classes: int,
            bond_classes: int,
            maximum_component_atoms: int,
            maximum_total_atoms: int,
            maximum_junction_budget: int,
            maximum_cycle_rank: int,
            maximum_attachment_count: int,
            maximum_decorations: int,
            hidden_dim: int,
            layers: int,
            dropout: float,
            conditioning_mode: str = "full_morphology",
            decoration_state_conditioning: str = "legacy_global",
            semantic_organization: str = "role_structured",
            backbone: str = "bidirectional_gru",
            attention_heads: int = 8,
            transformer_feedforward_multiplier: int = 4,
            tree_relation_attention: bool = True,
            role_routed_program_attention: bool = True,
            role_adapter_dim: int = 0,
        ) -> None:
            super().__init__()
            if (
                maximum_children < 1
                or atom_classes < 1
                or bond_classes not in {3, 4}
                or maximum_component_atoms < 1
                or maximum_total_atoms < 3
                or maximum_junction_budget < 0
                or maximum_cycle_rank < 0
                or maximum_attachment_count < 1
                or maximum_decorations < 1
                or hidden_dim < 16
                or hidden_dim % 2
                or layers < 1
                or not 0 <= dropout < 1
                or conditioning_mode not in {"full_morphology", "size_only"}
                or decoration_state_conditioning
                not in {"legacy_global", "bidirectional_anchor_local"}
                or backbone
                not in {
                    "bidirectional_gru",
                    "ugi_program_transformer",
                    "ugi_tree_program_transformer",
                }
                or attention_heads < 1
                or transformer_feedforward_multiplier < 1
                or not isinstance(tree_relation_attention, bool)
                or not isinstance(role_routed_program_attention, bool)
                or role_adapter_dim < 0
            ):
                raise UgiJointSparseFlowError("invalid joint sparse architecture")
            if backbone in {"ugi_program_transformer", "ugi_tree_program_transformer"} and (
                hidden_dim % attention_heads
                or semantic_organization != "role_structured"
                or conditioning_mode != "full_morphology"
            ):
                raise UgiJointSparseFlowError(
                    "Ugi program Transformer requires role-structured full morphology and "
                    "a hidden dimension divisible by its attention heads"
                )
            self.maximum_children = maximum_children
            self.atom_classes = atom_classes
            self.bond_classes = bond_classes
            self.maximum_component_atoms = maximum_component_atoms
            self.maximum_total_atoms = maximum_total_atoms
            self.maximum_junction_budget = maximum_junction_budget
            self.maximum_cycle_rank = maximum_cycle_rank
            self.maximum_attachment_count = maximum_attachment_count
            self.maximum_decorations = maximum_decorations
            self.hidden_dim = hidden_dim
            self.conditioning_mode = conditioning_mode
            self.decoration_state_conditioning = decoration_state_conditioning
            self.backbone = backbone
            self.attention_heads = attention_heads
            self.transformer_feedforward_multiplier = transformer_feedforward_multiplier
            self.tree_relation_attention = tree_relation_attention
            self.role_routed_program_attention = role_routed_program_attention
            self.role_adapter_dim = role_adapter_dim
            self.offspring_embedding = nn.Embedding(maximum_children + 1, hidden_dim)
            self.atom_embedding = nn.Embedding(atom_classes, hidden_dim)
            self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
            self.semantic_organization = _validated_semantic_organization(semantic_organization)
            flat = self.semantic_organization in FLAT_ORGANIZATIONS
            # Role identity is the treatment and goes. Generic sequence position is modelling
            # capacity and stays, widened from a per-role counter to an absolute index so the flat
            # arm is ablated rather than crippled.
            # Capacity parity is exact: every flat arm carries the same table including the null
            # slot, so the no-role arm differs from the true-role arm only in which index it feeds.
            self.role_embedding = nn.Embedding(
                len(ROLE_NAMES) + 1 if flat else len(ROLE_NAMES), hidden_dim
            )
            self.position_embedding = nn.Embedding(
                maximum_total_atoms if flat else maximum_component_atoms, hidden_dim
            )
            self.pending_embedding = (
                nn.Embedding(2 * maximum_total_atoms + 1, hidden_dim)
                if conditioning_mode == "full_morphology" and not flat
                else None
            )
            # The flat arm receives the identical twelve-element program, consumed by one MLP
            # instead of role-indexed tables, so the information is matched but not the structure.
            self.program_projection = (
                nn.Sequential(
                    nn.Linear(12, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
                )
                if flat
                else None
            )
            self.count_embeddings = (
                None
                if flat
                else nn.ModuleList(
                    nn.Embedding(maximum_component_atoms + 1, hidden_dim) for _ in ROLE_NAMES
                )
            )
            self.junction_embeddings = (
                nn.ModuleList(
                    nn.Embedding(maximum_junction_budget + 1, hidden_dim) for _ in ROLE_NAMES
                )
                if conditioning_mode == "full_morphology" and not flat
                else None
            )
            self.cycle_embeddings = (
                nn.ModuleList(nn.Embedding(maximum_cycle_rank + 1, hidden_dim) for _ in ROLE_NAMES)
                if conditioning_mode == "full_morphology" and not flat
                else None
            )
            self.attachment_embeddings = (
                nn.ModuleList(
                    nn.Embedding(maximum_attachment_count + 1, hidden_dim) for _ in ROLE_NAMES
                )
                if conditioning_mode == "full_morphology" and not flat
                else None
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.sequence = (
                nn.GRU(
                    hidden_dim,
                    hidden_dim // 2,
                    num_layers=layers,
                    batch_first=True,
                    dropout=dropout if layers > 1 else 0.0,
                    bidirectional=True,
                )
                if backbone == "bidirectional_gru"
                else None
            )
            self.program_global_token = (
                nn.Parameter(torch.zeros(hidden_dim))
                if backbone in {"ugi_program_transformer", "ugi_tree_program_transformer"}
                else None
            )
            self.core_port_embedding = (
                nn.Embedding(len(ROLE_NAMES), hidden_dim)
                if backbone == "ugi_tree_program_transformer"
                else None
            )
            self.transformer_blocks = (
                nn.ModuleList(
                    (
                        _UgiTreeProgramTransformerBlock(
                            hidden_dim=hidden_dim,
                            heads=attention_heads,
                            dropout=dropout,
                            feedforward_multiplier=transformer_feedforward_multiplier,
                            maximum_relative_position=max(1, maximum_component_atoms - 1),
                            use_tree_relations=tree_relation_attention,
                            role_routed_program_attention=role_routed_program_attention,
                            role_adapter_dim=role_adapter_dim,
                        )
                        if backbone == "ugi_tree_program_transformer"
                        else _UgiProgramTransformerBlock(
                            hidden_dim=hidden_dim,
                            heads=attention_heads,
                            dropout=dropout,
                            feedforward_multiplier=transformer_feedforward_multiplier,
                        )
                    )
                    for _ in range(layers)
                )
                if backbone in {"ugi_program_transformer", "ugi_tree_program_transformer"}
                else None
            )
            self.output = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            self.offspring_output = nn.Linear(hidden_dim, maximum_children + 1)
            self.atom_output = nn.Linear(hidden_dim, atom_classes)
            self.parent_bond_output = nn.Linear(hidden_dim, bond_classes)
            self.closure_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.decoration_query = nn.Linear(hidden_dim, hidden_dim)
            self.decoration_key = nn.Linear(hidden_dim, hidden_dim)
            self.no_decoration = nn.Parameter(torch.zeros(hidden_dim))
            self.decoration_slot_embedding = nn.Embedding(maximum_decorations, hidden_dim)
            self.decoration_atom_output = nn.Linear(hidden_dim, atom_classes)
            self.decoration_bond_output = nn.Linear(hidden_dim, bond_classes)
            if decoration_state_conditioning == "bidirectional_anchor_local":
                self.decoration_atom_embedding = nn.Embedding(atom_classes, hidden_dim)
                self.decoration_bond_embedding = nn.Embedding(bond_classes, hidden_dim)
                self.decoration_to_node = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                self.decoration_slot_output = nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                )
                self.no_anchor_context = nn.Parameter(torch.zeros(hidden_dim))
            else:
                self.decoration_atom_embedding = None
                self.decoration_bond_embedding = None
                self.decoration_to_node = None
                self.decoration_slot_output = None
                self.no_anchor_context = None
            self.cycle_outputs = (
                nn.ModuleList(nn.Linear(hidden_dim, maximum_cycle_rank + 1) for _ in ROLE_NAMES)
                if conditioning_mode == "size_only"
                else None
            )
            self.attachment_outputs = (
                nn.ModuleList(
                    nn.Linear(hidden_dim, maximum_attachment_count + 1) for _ in ROLE_NAMES
                )
                if conditioning_mode == "size_only"
                else None
            )

        def _program_context(self, programs: Any) -> Any:
            if self.semantic_organization in FLAT_ORGANIZATIONS:
                return self.program_projection(programs.to(self.offspring_embedding.weight.dtype))
            counts = programs[:, :3]
            context = torch.zeros(
                (programs.shape[0], self.hidden_dim),
                dtype=self.offspring_embedding.weight.dtype,
                device=programs.device,
            )
            for role_index in range(len(ROLE_NAMES)):
                context = context + self.count_embeddings[role_index](counts[:, role_index])
                if self.conditioning_mode == "full_morphology":
                    context = (
                        context
                        + self.junction_embeddings[role_index](programs[:, 3 + role_index])
                        + self.cycle_embeddings[role_index](programs[:, 6 + role_index])
                        + self.attachment_embeddings[role_index](programs[:, 9 + role_index])
                    )
            return context

        def _program_tokens(self, programs: Any, t: Any) -> Any:
            """Return one global and three role-local coarse-program tokens."""

            if (
                self.backbone not in {"ugi_program_transformer", "ugi_tree_program_transformer"}
                or self.program_global_token is None
                or self.count_embeddings is None
                or self.junction_embeddings is None
                or self.cycle_embeddings is None
                or self.attachment_embeddings is None
            ):
                raise UgiJointSparseFlowError("program tokens require the Ugi program Transformer")
            role_tokens = []
            for role_index in range(len(ROLE_NAMES)):
                token = (
                    self.role_embedding.weight[role_index][None]
                    + self.count_embeddings[role_index](programs[:, role_index])
                    + self.junction_embeddings[role_index](programs[:, 3 + role_index])
                    + self.cycle_embeddings[role_index](programs[:, 6 + role_index])
                    + self.attachment_embeddings[role_index](programs[:, 9 + role_index])
                )
                if self.core_port_embedding is not None:
                    token = token + self.core_port_embedding.weight[role_index][None]
                role_tokens.append(token)
            roles = torch.stack(role_tokens, dim=1)
            global_token = roles.mean(dim=1) + self.program_global_token[None]
            tokens = torch.cat((global_token[:, None], roles), dim=1)
            return tokens + self.time_embedding(t[:, None])[:, None, :]

        def forward(
            self,
            *,
            offspring: Any,
            nodes: Any,
            parent_bonds: Any,
            role_states: Any,
            within_role_positions: Any,
            programs: Any,
            node_mask: Any,
            t: Any,
            closure_left: Any | None = None,
            closure_right: Any | None = None,
            decoration_anchors: Any | None = None,
            decoration_atoms: Any | None = None,
            decoration_bonds: Any | None = None,
        ) -> dict[str, Any]:
            shape = offspring.shape
            if (
                nodes.shape != shape
                or parent_bonds.shape != shape
                or role_states.shape != shape
                or within_role_positions.shape != shape
                or node_mask.shape != shape
                or programs.shape != (shape[0], 12)
                or t.shape != (shape[0],)
            ):
                raise UgiJointSparseFlowError("joint sparse state shapes do not agree")
            hidden = (
                self.offspring_embedding(offspring)
                + self.atom_embedding(nodes)
                + self.bond_embedding(parent_bonds)
                + self.position_embedding(within_role_positions)
                + self._program_context(programs)[:, None, :]
                + self.time_embedding(t[:, None])[:, None, :]
            )
            if self.semantic_organization == "role_structured":
                hidden = hidden + self.role_embedding(role_states)
            elif self.semantic_organization == "flat_true_role":
                hidden = hidden + self.role_embedding(role_states)
            elif self.semantic_organization == "flat_misaligned_role":
                # The collate has already placed the misaligned labels in role_states, so the arm
                # differs from flat_true_role only in the contents of that channel.
                hidden = hidden + self.role_embedding(role_states)
            else:
                # Constant null tag. The table is present and the same size; only the index differs.
                hidden = hidden + self.role_embedding(torch.full_like(role_states, NULL_ROLE_INDEX))
            if self.conditioning_mode == "full_morphology" and self.pending_embedding is not None:
                pending = _pending_by_role(
                    offspring,
                    role_states,
                    node_mask,
                    self.maximum_total_atoms,
                    programs[:, 9:12],
                )
                hidden = hidden + self.pending_embedding(pending)
            decoration_slot_state = None
            if self.decoration_state_conditioning == "bidirectional_anchor_local":
                expected_decoration_shape = (shape[0], self.maximum_decorations)
                if (
                    decoration_anchors is None
                    or decoration_atoms is None
                    or decoration_bonds is None
                    or decoration_anchors.shape != expected_decoration_shape
                    or decoration_atoms.shape != expected_decoration_shape
                    or decoration_bonds.shape != expected_decoration_shape
                ):
                    raise UgiJointSparseFlowError(
                        "anchor-local decoration conditioning requires the complete noisy slot state"
                    )
                if bool((decoration_anchors < 0).any()) or bool(
                    (decoration_anchors > shape[1]).any()
                ):
                    raise UgiJointSparseFlowError(
                        "decoration anchor state points outside the padded sequence"
                    )
                present_anchor_indices = (decoration_anchors - 1).clamp(min=0)
                anchors_active = torch.gather(node_mask, 1, present_anchor_indices)
                if bool(((decoration_anchors > 0) & ~anchors_active).any()):
                    raise UgiJointSparseFlowError(
                        "decoration anchor state points to a padded sequence position"
                    )
                slots = torch.arange(self.maximum_decorations, device=hidden.device)
                decoration_slot_state = (
                    self.decoration_atom_embedding(decoration_atoms)
                    + self.decoration_bond_embedding(decoration_bonds)
                    + self.decoration_slot_embedding(slots)[None, :, :]
                )
                present = decoration_anchors > 0
                anchor_indices = present_anchor_indices
                decoration_node_context = torch.zeros_like(hidden)
                decoration_node_context.scatter_add_(
                    1,
                    anchor_indices[:, :, None].expand(-1, -1, self.hidden_dim),
                    decoration_slot_state * present[:, :, None],
                )
                hidden = hidden + self.decoration_to_node(decoration_node_context)
            hidden = self.input_norm(hidden) * node_mask[:, :, None]
            if self.backbone == "bidirectional_gru":
                if self.sequence is None:
                    raise UgiJointSparseFlowError("GRU backbone is unexpectedly absent")
                packed = pack_padded_sequence(
                    hidden,
                    node_mask.sum(dim=1).to("cpu"),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_output, _ = self.sequence(packed)
                hidden, _ = pad_packed_sequence(
                    packed_output,
                    batch_first=True,
                    total_length=shape[1],
                )
            else:
                if self.transformer_blocks is None:
                    raise UgiJointSparseFlowError("Transformer backbone is unexpectedly absent")
                program_tokens = self._program_tokens(programs, t)
                relation_states = (
                    noisy_preorder_relation_states(offspring, role_states, node_mask)
                    if self.backbone == "ugi_tree_program_transformer"
                    else None
                )
                for block in self.transformer_blocks:
                    if self.backbone == "ugi_tree_program_transformer":
                        hidden = block(
                            hidden,
                            node_mask=node_mask,
                            program_tokens=program_tokens,
                            relation_states=relation_states,
                            within_role_positions=within_role_positions,
                            role_states=role_states,
                        )
                    else:
                        hidden = block(
                            hidden,
                            node_mask=node_mask,
                            program_tokens=program_tokens,
                        )
            hidden = (hidden + self.output(hidden)) * node_mask[:, :, None]
            global_hidden = hidden.sum(dim=1) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            query = self.decoration_query(global_hidden)
            slots = torch.arange(self.maximum_decorations, device=hidden.device)
            slot_hidden = query[:, None, :] + self.decoration_slot_embedding(slots)[None, :, :]
            if self.decoration_state_conditioning == "bidirectional_anchor_local":
                assert decoration_slot_state is not None
                assert decoration_anchors is not None
                anchor_indices = (decoration_anchors - 1).clamp(min=0)
                anchor_hidden = torch.gather(
                    hidden,
                    1,
                    anchor_indices[:, :, None].expand(-1, -1, self.hidden_dim),
                )
                anchor_hidden = torch.where(
                    (decoration_anchors > 0)[:, :, None],
                    anchor_hidden,
                    self.no_anchor_context[None, None, :],
                )
                slot_hidden = self.decoration_slot_output(
                    slot_hidden + decoration_slot_state + anchor_hidden
                )
            node_scores = torch.einsum(
                "bsd,bnd->bsn",
                slot_hidden,
                self.decoration_key(hidden),
            ) / np.sqrt(self.hidden_dim)
            node_scores = node_scores.masked_fill(~node_mask[:, None, :], -torch.inf)
            none_score = torch.einsum("bsd,d->bs", slot_hidden, self.no_decoration)[:, :, None]
            result = {
                "offspring": self.offspring_output(hidden),
                "nodes": self.atom_output(hidden),
                "parent_bonds": self.parent_bond_output(hidden),
                "decoration_anchors": torch.cat((none_score, node_scores), dim=2),
                "decoration_atoms": self.decoration_atom_output(slot_hidden),
                "decoration_bonds": self.decoration_bond_output(slot_hidden),
                "hidden": hidden,
            }
            if self.conditioning_mode == "size_only":
                role_hidden = []
                for role_index in range(len(ROLE_NAMES)):
                    role_mask = node_mask & (role_states == role_index)
                    role_hidden.append(
                        (hidden * role_mask[:, :, None]).sum(dim=1)
                        / role_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    )
                result["cycle_ranks"] = torch.stack(
                    [
                        self.cycle_outputs[role_index](role_hidden[role_index])
                        for role_index in range(len(ROLE_NAMES))
                    ],
                    dim=1,
                )
                result["attachment_counts"] = torch.stack(
                    [
                        self.attachment_outputs[role_index](role_hidden[role_index])
                        for role_index in range(len(ROLE_NAMES))
                    ],
                    dim=1,
                )
            if closure_left is not None or closure_right is not None:
                if (
                    closure_left is None
                    or closure_right is None
                    or closure_left.shape != closure_right.shape
                ):
                    raise UgiJointSparseFlowError("closure endpoint tensors are misaligned")
                left_hidden = torch.gather(
                    hidden,
                    1,
                    closure_left[:, :, None].expand(-1, -1, self.hidden_dim),
                )
                right_hidden = torch.gather(
                    hidden,
                    1,
                    closure_right[:, :, None].expand(-1, -1, self.hidden_dim),
                )
                result["closure_bonds"] = self.closure_bond_output(
                    torch.cat((left_hidden, right_hidden), dim=-1)
                )
            return result

else:  # pragma: no cover

    class UgiJointSparseFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise UgiJointSparseFlowError("joint sparse flow requires torch")


def _sample_interpolation(
    clean: Any,
    marginal: Any,
    t: Any,
    mask: Any,
    generator: Any,
) -> Any:
    output = clean.clone()
    if not bool(mask.any()):
        return output
    examples = torch.arange(clean.shape[0], device=clean.device)[:, None].expand_as(clean)[mask]
    if marginal.ndim == 1:
        probabilities = marginal[None, :].expand(int(mask.sum()), -1).clone()
    elif marginal.shape[:2] == clean.shape:
        probabilities = marginal[mask].clone()
    else:
        raise UgiJointSparseFlowError("joint interpolation source is misaligned")
    probabilities *= 1.0 - t[examples, None]
    probabilities.scatter_add_(1, clean[mask][:, None], t[examples, None])
    output[mask] = torch.multinomial(
        probabilities,
        1,
        generator=generator,
    ).squeeze(1)
    return output


def _decoration_anchor_source(node_mask: Any, binary_source: Any) -> Any:
    output = torch.zeros(
        (node_mask.shape[0], node_mask.shape[1] + 1),
        dtype=binary_source.dtype,
        device=node_mask.device,
    )
    output[:, 0] = binary_source[0]
    output[:, 1:] = node_mask.to(output.dtype) * (
        binary_source[1] / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
    )
    return output


def noise_ugi_joint_sparse_batch(
    clean: dict[str, Any],
    sources: dict[str, Any],
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    """Corrupt topology and chemistry together at one shared flow time."""

    mask = clean["node_mask"]
    roles = clean["role_states"]
    slots = clean["decoration_anchors"].shape[1]
    anchor_source = _decoration_anchor_source(mask, sources["decoration"])
    return {
        "offspring": _sample_interpolation(
            clean["offspring"], sources["offspring"][roles], t, mask, generator
        ),
        "nodes": _sample_interpolation(clean["nodes"], sources["atoms"][roles], t, mask, generator),
        "parent_bonds": _sample_interpolation(
            clean["parent_bonds"], sources["bonds"][roles], t, mask, generator
        ),
        "closure_bonds": _sample_interpolation(
            clean["closure_bonds"],
            sources["closure_bonds"],
            t,
            clean["closure_mask"],
            generator,
        ),
        "decoration_anchors": _sample_interpolation(
            clean["decoration_anchors"],
            anchor_source[:, None, :].expand(-1, slots, -1),
            t,
            torch.ones_like(clean["decoration_anchors"], dtype=torch.bool),
            generator,
        ),
        "decoration_atoms": _sample_interpolation(
            clean["decoration_atoms"],
            sources["decoration_atoms"],
            t,
            torch.ones_like(clean["decoration_atoms"], dtype=torch.bool),
            generator,
        ),
        "decoration_bonds": _sample_interpolation(
            clean["decoration_bonds"],
            sources["decoration_bonds"],
            t,
            torch.ones_like(clean["decoration_bonds"], dtype=torch.bool),
            generator,
        ),
    }


def _role_balanced_ce(logits: Any, target: Any, batch: dict[str, Any], label: str) -> Any:
    losses = []
    for role_index, role in enumerate(ROLE_NAMES):
        mask = batch["node_mask"] & (batch["role_states"] == role_index)
        if not bool(mask.any()):
            raise UgiJointSparseFlowError(f"joint batch lacks {role} {label} supervision")
        losses.append(functional.cross_entropy(logits[mask], target[mask]))
    return torch.stack(losses).mean()


def _pooled_ce(logits: Any, target: Any, batch: dict[str, Any], label: str) -> Any:
    """Pooled objective for the flat arms.

    A role-balanced objective is itself role structure, so it must not survive into an arm whose
    treatment is the absence of the role channel. Pooling over the whole molecule is the matched
    alternative and is applied identically to every flat arm, including the true-role one, so the
    arms differ only in the role channel and not in the objective.
    """
    mask = batch["node_mask"]
    if not bool(mask.any()):
        raise UgiJointSparseFlowError(f"joint batch lacks {label} supervision")
    return functional.cross_entropy(logits[mask], target[mask])


BASE_JOINT_LOSS_TERMS = (
    "offspring_ce",
    "atom_ce",
    "parent_bond_ce",
    "closure_bond_ce",
    "decoration_anchor_ce",
    "decoration_atom_ce",
    "decoration_bond_ce",
)
PROGRAM_CONSISTENCY_TERMS = (
    "attachment_count_consistency",
    "junction_budget_consistency",
)


def _objective_weights(
    supplied: Mapping[str, float] | None,
    *,
    names: Sequence[str],
    default: float,
    label: str,
) -> dict[str, float]:
    values = {name: default for name in names}
    if supplied is not None:
        unknown = set(supplied).difference(names)
        if unknown:
            raise UgiJointSparseFlowError(f"unknown {label} terms: {sorted(unknown)}")
        values.update({str(name): float(value) for name, value in supplied.items()})
    if any(not np.isfinite(value) or value < 0 for value in values.values()):
        raise UgiJointSparseFlowError(f"{label} weights must be finite and nonnegative")
    return values


def _program_consistency_losses(
    predictions: dict[str, Any], clean: dict[str, Any]
) -> dict[str, Any]:
    """Compare soft offspring topology with the supplied coarse role program."""

    logits = predictions["offspring"]
    probabilities = logits.softmax(dim=-1)
    child_values = torch.arange(logits.shape[-1], dtype=logits.dtype, device=logits.device)
    expected_children = torch.einsum("bnc,c->bn", probabilities, child_values)
    expected_junctions = torch.einsum(
        "bnc,c->bn",
        probabilities,
        (child_values - 1).clamp(min=0),
    )
    attachment_losses = []
    junction_losses = []
    for role_index in range(len(ROLE_NAMES)):
        role_mask = clean["node_mask"] & (clean["role_states"] == role_index)
        node_counts = clean["programs"][:, role_index].to(logits.dtype).clamp(min=1)
        target_children = (
            clean["programs"][:, role_index] - clean["programs"][:, 9 + role_index]
        ).to(logits.dtype)
        predicted_children = (expected_children * role_mask).sum(dim=1)
        attachment_losses.append(
            functional.smooth_l1_loss(
                predicted_children / node_counts,
                target_children / node_counts,
            )
        )
        target_junctions = clean["programs"][:, 3 + role_index].to(logits.dtype)
        predicted_junctions = (expected_junctions * role_mask).sum(dim=1)
        junction_losses.append(
            functional.smooth_l1_loss(
                predicted_junctions / node_counts,
                target_junctions / node_counts,
            )
        )
    return {
        "attachment_count_consistency": torch.stack(attachment_losses).mean(),
        "junction_budget_consistency": torch.stack(junction_losses).mean(),
    }


def ugi_joint_sparse_loss(
    predictions: dict[str, Any],
    clean: dict[str, Any],
    *,
    semantic_organization: str = "role_structured",
    loss_weights: Mapping[str, float] | None = None,
    consistency_weights: Mapping[str, float] | None = None,
) -> tuple[Any, dict[str, float]]:
    """Balance topology and chemistry while sharing one denoising backbone."""

    organization = _validated_semantic_organization(semantic_organization)
    node_ce = _pooled_ce if organization in FLAT_ORGANIZATIONS else _role_balanced_ce
    offspring_loss = node_ce(predictions["offspring"], clean["offspring"], clean, "offspring")
    atom_loss = node_ce(predictions["nodes"], clean["nodes"], clean, "atom")
    parent_bond_loss = node_ce(predictions["parent_bonds"], clean["parent_bonds"], clean, "bond")
    closure_mask = clean["closure_mask"]
    closure_loss = (
        functional.cross_entropy(
            predictions["closure_bonds"][closure_mask], clean["closure_bonds"][closure_mask]
        )
        if bool(closure_mask.any())
        else atom_loss * 0.0
    )
    decoration_anchor_loss = functional.cross_entropy(
        predictions["decoration_anchors"].flatten(0, 1),
        clean["decoration_anchors"].flatten(),
    )
    decoration_mask = clean["decoration_present_mask"]
    decoration_atom_loss = functional.cross_entropy(
        predictions["decoration_atoms"][decoration_mask],
        clean["decoration_atoms"][decoration_mask],
    )
    decoration_bond_loss = functional.cross_entropy(
        predictions["decoration_bonds"][decoration_mask],
        clean["decoration_bonds"][decoration_mask],
    )
    terms = {
        "offspring_ce": offspring_loss,
        "atom_ce": atom_loss,
        "parent_bond_ce": parent_bond_loss,
        "closure_bond_ce": closure_loss,
        "decoration_anchor_ce": decoration_anchor_loss,
        "decoration_atom_ce": decoration_atom_loss,
        "decoration_bond_ce": decoration_bond_loss,
    }
    resolved_loss_weights = _objective_weights(
        loss_weights,
        names=BASE_JOINT_LOSS_TERMS,
        default=1.0,
        label="joint-loss",
    )
    resolved_consistency_weights = _objective_weights(
        consistency_weights,
        names=PROGRAM_CONSISTENCY_TERMS,
        default=0.0,
        label="program-consistency",
    )
    consistency = _program_consistency_losses(predictions, clean)
    total = sum(resolved_loss_weights[name] * terms[name] for name in BASE_JOINT_LOSS_TERMS)
    total = total + sum(
        resolved_consistency_weights[name] * consistency[name] for name in PROGRAM_CONSISTENCY_TERMS
    )
    metrics = {name: float(value.detach()) for name, value in terms.items()}
    metrics.update({name: float(value.detach()) for name, value in consistency.items()})
    generated_global_states = {
        "cycle_ranks",
        "attachment_counts",
    }
    present_global_states = generated_global_states.intersection(predictions)
    if present_global_states and present_global_states != generated_global_states:
        raise UgiJointSparseFlowError("size-only global topology outputs are incomplete")
    if present_global_states:
        cycle_loss = functional.cross_entropy(
            predictions["cycle_ranks"].flatten(0, 1),
            clean["programs"][:, 6:9].flatten(),
        )
        attachment_loss = functional.cross_entropy(
            predictions["attachment_counts"].flatten(0, 1),
            clean["programs"][:, 9:12].flatten(),
        )
        total = total + cycle_loss + attachment_loss
        metrics.update(
            {
                "cycle_rank_ce": float(cycle_loss.detach()),
                "attachment_count_ce": float(attachment_loss.detach()),
            }
        )
    metrics["unweighted_base_total"] = float(sum(terms.values()).detach())
    metrics["total"] = float(total.detach())
    return total, metrics
