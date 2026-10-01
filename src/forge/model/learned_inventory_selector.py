"""Autoregressive learned selector over the frozen train-only Ugi component inventory."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase

from forge.assembly import Ugi3AssemblyAdapter
from forge.core.io import iter_csv
from forge.model.common_ugi_benchmark import CommonUgiAttempt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    functional = None  # type: ignore[assignment]


class LearnedInventorySelectorError(ValueError):
    """The learned finite inventory or training contract is invalid."""


def _features(smiles: str) -> tuple[float, ...]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise LearnedInventorySelectorError("inventory contains an invalid component")
    heavy = molecule.GetNumHeavyAtoms()
    return (
        heavy / 64.0,
        molecule.GetRingInfo().NumRings() / 8.0,
        sum(atom.GetDegree() >= 3 for atom in molecule.GetAtoms()) / 16.0,
        sum(atom.GetAtomicNum() not in {1, 6} for atom in molecule.GetAtoms()) / 16.0,
    )


@dataclass(frozen=True)
class InventoryTrainingData:
    roles: tuple[str, ...]
    inventories: tuple[tuple[str, ...], ...]
    contexts: np.ndarray
    targets: np.ndarray
    weights: np.ndarray

    @property
    def visible_component_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{role}:{smiles}"
            for role, inventory in zip(self.roles, self.inventories, strict=True)
            for smiles in inventory
        )


def load_inventory_training_data(
    assignments_path: Path, *, roles: Sequence[str]
) -> InventoryTrainingData:
    """Tensorize only source-weighted train-fold identities and count context."""

    if len(roles) != 3:
        raise LearnedInventorySelectorError("learned inventory requires exactly three Ugi roles")
    rows = [row for row in iter_csv(assignments_path) if row["primary_product_fold"] == "train"]
    if not rows:
        raise LearnedInventorySelectorError("learned inventory training fold is empty")
    inventories = tuple(tuple(sorted({row[f"{role}_smiles"] for row in rows})) for role in roles)
    indices = tuple(
        {smiles: index for index, smiles in enumerate(values)} for values in inventories
    )
    feature_cache = {smiles: _features(smiles) for values in inventories for smiles in values}
    contexts = []
    targets = []
    weights = []
    for row in rows:
        components = tuple(row[f"{role}_smiles"] for role in roles)
        contexts.append([value for component in components for value in feature_cache[component]])
        targets.append([indices[index][component] for index, component in enumerate(components)])
        weight = float(row["family_balance_weight_raw"])
        if not math.isfinite(weight) or weight <= 0.0:
            raise LearnedInventorySelectorError("train row has invalid realism weight")
        weights.append(weight)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array /= weight_array.sum()
    return InventoryTrainingData(
        roles=tuple(roles),
        inventories=inventories,
        contexts=np.asarray(contexts, dtype=np.float32),
        targets=np.asarray(targets, dtype=np.int64),
        weights=weight_array,
    )


if nn is not None:

    class AutoregressiveInventorySelector(nn.Module):
        """Predict A, then B conditioned on A, then C conditioned on A/B."""

        def __init__(
            self,
            *,
            context_dim: int,
            inventory_sizes: Sequence[int],
            hidden_dim: int,
            embedding_dim: int,
        ) -> None:
            super().__init__()
            if len(inventory_sizes) != 3 or min(inventory_sizes) < 1:
                raise LearnedInventorySelectorError("inventory support is invalid")
            self.context = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.embeddings = nn.ModuleList(
                nn.Embedding(size, embedding_dim) for size in inventory_sizes[:2]
            )
            self.outputs = nn.ModuleList(
                (
                    nn.Linear(hidden_dim, inventory_sizes[0]),
                    nn.Linear(hidden_dim + embedding_dim, inventory_sizes[1]),
                    nn.Linear(hidden_dim + 2 * embedding_dim, inventory_sizes[2]),
                )
            )

        def forward(self, context: Any, targets: Any | None = None) -> tuple[Any, ...]:
            hidden = self.context(context)
            first = self.outputs[0](hidden)
            selected_first = first.argmax(dim=-1) if targets is None else targets[:, 0]
            first_embedding = self.embeddings[0](selected_first)
            second = self.outputs[1](torch.cat((hidden, first_embedding), dim=-1))
            selected_second = second.argmax(dim=-1) if targets is None else targets[:, 1]
            second_embedding = self.embeddings[1](selected_second)
            third = self.outputs[2](torch.cat((hidden, first_embedding, second_embedding), dim=-1))
            return first, second, third

else:  # pragma: no cover

    class AutoregressiveInventorySelector:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise LearnedInventorySelectorError("learned inventory selector requires torch")


def train_inventory_selector(
    data: InventoryTrainingData,
    *,
    seed: int,
    steps: int,
    batch_size: int,
    hidden_dim: int,
    embedding_dim: int,
    learning_rate: float,
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Train with source-balanced sampling; raw family counts never define the measure."""

    if torch is None:
        raise LearnedInventorySelectorError("learned inventory selector requires torch")
    if functional is None:
        raise LearnedInventorySelectorError("learned inventory selector requires torch functions")
    if min(steps, batch_size, hidden_dim, embedding_dim) < 1 or learning_rate <= 0:
        raise LearnedInventorySelectorError("learned inventory runtime is invalid")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise LearnedInventorySelectorError("CUDA requested but unavailable")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed + 1)
    model = AutoregressiveInventorySelector(
        context_dim=data.contexts.shape[1],
        inventory_sizes=[len(values) for values in data.inventories],
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
    ).to(resolved)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses = []
    model.train()
    for step in range(1, steps + 1):
        selected = rng.choice(len(data.contexts), size=batch_size, replace=True, p=data.weights)
        context = torch.as_tensor(data.contexts[selected], device=resolved)
        targets = torch.as_tensor(data.targets[selected], device=resolved)
        logits = model(context, targets)
        loss = torch.stack(
            tuple(
                functional.cross_entropy(value, targets[:, index])
                for index, value in enumerate(logits)
            )
        ).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return model, {
        "steps": steps,
        "examples_seen": steps * batch_size,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "sampling_measure": "family_balance_weight_raw_normalized_within_train_fold",
    }


def sample_inventory_selector(
    model: Any,
    data: InventoryTrainingData,
    adapter: Ugi3AssemblyAdapter,
    *,
    method_id: str,
    seed: int,
    attempts: int,
    maximum_forward_outcomes: int,
    device: str,
) -> tuple[CommonUgiAttempt, ...]:
    """Sample one product at most per attempt; retain all native failures."""

    if torch is None or attempts < 1:
        raise LearnedInventorySelectorError("inventory sampling runtime is invalid")
    rng = np.random.default_rng(seed)
    resolved = torch.device(device)
    model.eval()
    records: list[CommonUgiAttempt] = []
    visible = data.visible_component_ids
    generator = torch.Generator(device=resolved).manual_seed(seed)
    with torch.no_grad():
        for attempt_index in range(attempts):
            context_index = int(rng.choice(len(data.contexts), p=data.weights))
            context = torch.as_tensor(data.contexts[[context_index]], device=resolved)
            selected: list[int] = []
            hidden = model.context(context)
            first = model.outputs[0](hidden)
            first_index = int(
                torch.multinomial(torch.softmax(first, dim=-1), 1, generator=generator).item()
            )
            selected.append(first_index)
            first_embedding = model.embeddings[0](torch.tensor([first_index], device=resolved))
            second = model.outputs[1](torch.cat((hidden, first_embedding), dim=-1))
            second_index = int(
                torch.multinomial(torch.softmax(second, dim=-1), 1, generator=generator).item()
            )
            selected.append(second_index)
            second_embedding = model.embeddings[1](torch.tensor([second_index], device=resolved))
            third = model.outputs[2](torch.cat((hidden, first_embedding, second_embedding), dim=-1))
            third_index = int(
                torch.multinomial(torch.softmax(third, dim=-1), 1, generator=generator).item()
            )
            selected.append(third_index)
            components = {
                role: data.inventories[index][component_index]
                for index, (role, component_index) in enumerate(
                    zip(data.roles, selected, strict=True)
                )
            }
            products = adapter.forward_products(
                components, maximum_outcomes=maximum_forward_outcomes
            )
            if products.saturated:
                status = "failed"
                product = None
            elif products.products:
                status = "generated"
                product = products.products[int(rng.integers(0, len(products.products)))]
            else:
                status = "failed"
                product = None
            records.append(
                CommonUgiAttempt(
                    method_id=method_id,
                    seed=seed,
                    attempt_index=attempt_index,
                    status=status,  # type: ignore[arg-type]
                    product_smiles=product,
                    method_visible_component_ids=visible,
                    generator_calls=1,
                    reaction_calls=1,
                    route_calls=0,
                    oracle_calls=0,
                    # Runtime telemetry is deliberately kept out of the strict scientific ledger.
                    # A separate performance benchmark may record it without invalidating byte
                    # reproducibility of the chemistry result.
                    wall_seconds=0.0,
                )
            )
    return tuple(records)


__all__ = [
    "AutoregressiveInventorySelector",
    "InventoryTrainingData",
    "LearnedInventorySelectorError",
    "load_inventory_training_data",
    "sample_inventory_selector",
    "train_inventory_selector",
]
