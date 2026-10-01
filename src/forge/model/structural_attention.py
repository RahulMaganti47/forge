"""Small attention primitives with explicit structural relation biases."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


class StructuralAttentionError(ValueError):
    """Attention masks or relation biases disagree with the supplied tensors."""


if nn is not None:

    class BiasedMultiheadAttention(nn.Module):
        """Multi-head attention with a batch-specific additive relation bias.

        PyTorch's stock ``MultiheadAttention`` accepts only a limited collection of mask layouts.
        Molecular relations vary per graph, query and key, so this compact implementation makes the
        required ``[batch, heads, queries, keys]`` bias an explicit part of the interface.
        """

        def __init__(self, hidden_dim: int, heads: int, dropout: float) -> None:
            super().__init__()
            if hidden_dim < 1 or heads < 1 or hidden_dim % heads or not 0 <= dropout < 1:
                raise StructuralAttentionError("invalid biased-attention dimensions")
            self.heads = heads
            self.head_dim = hidden_dim // heads
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.output = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(
            self,
            query: Any,
            memory: Any,
            *,
            query_mask: Any,
            memory_mask: Any,
            attention_bias: Any | None = None,
        ) -> Any:
            if query.ndim != 3 or memory.ndim != 3:
                raise StructuralAttentionError("attention inputs must be batched sequences")
            batch, queries, hidden_dim = query.shape
            memory_batch, keys, memory_hidden = memory.shape
            if (
                memory_batch != batch
                or memory_hidden != hidden_dim
                or query_mask.shape != (batch, queries)
                or memory_mask.shape != (batch, keys)
                or query_mask.dtype != torch.bool
                or memory_mask.dtype != torch.bool
            ):
                raise StructuralAttentionError("attention input and mask shapes disagree")
            # CUDA training masks are produced by validated collators.  Reading a device boolean
            # here would synchronize the host once per attention layer.  Retain the defensive
            # content check for direct CPU callers without putting it on the accelerator hot path.
            if memory_mask.device.type == "cpu" and bool((~memory_mask).all(dim=1).any()):
                raise StructuralAttentionError("attention memory cannot be empty")

            q = self.query(query).reshape(batch, queries, self.heads, self.head_dim).transpose(1, 2)
            k = self.key(memory).reshape(batch, keys, self.heads, self.head_dim).transpose(1, 2)
            v = self.value(memory).reshape(batch, keys, self.heads, self.head_dim).transpose(1, 2)
            if attention_bias is not None:
                if attention_bias.ndim != 4 or attention_bias.shape[0] != batch:
                    raise StructuralAttentionError("attention bias must be a batched four-tensor")
                if attention_bias.shape[1:] == (1, queries, keys):
                    attention_bias = attention_bias.expand(-1, self.heads, -1, -1)
                if attention_bias.shape != (batch, self.heads, queries, keys):
                    raise StructuralAttentionError("attention bias shape disagrees with attention")
                attention_mask = attention_bias.masked_fill(
                    ~memory_mask[:, None, None, :], -torch.inf
                )
            else:
                attention_mask = memory_mask[:, None, None, :]
            context = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
            )
            context = context.transpose(1, 2).reshape(batch, queries, hidden_dim)
            return self.output(context) * query_mask[:, :, None]

else:  # pragma: no cover

    class BiasedMultiheadAttention:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise StructuralAttentionError("biased attention requires torch")


__all__ = ["BiasedMultiheadAttention", "StructuralAttentionError"]
