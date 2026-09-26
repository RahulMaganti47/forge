"""Topology-conditioned discrete flow for Ugi lipid chemistry realization."""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from rdkit import Chem

from forge.corpus.ugi_chemistry_corpus import UgiChemistryRecord
from forge.model.adapter_node_conditioning import AdapterNodeConditioning
from forge.model.defog_feasibility import AtomState, _rstar_step
from forge.model.local_chemistry_support import LocalChemistrySupport, tree_path_indices
from forge.model.phase1_flow import DeterministicSparseFlowBlock
from forge.model.ugi_adapter_features import ORIGIN_TO_INDEX
from forge.model.ugi_chemistry_interface import (
    ROOT_BOND_TARGET,
    ChemistryTopologyCondition,
    recompute_adapter_distances,
)
from forge.model.v5_sparse_representation import (
    _INDEX_TO_BOND_TYPE,
    V5SparseGraphRecord,
    v5_graph_to_molecule,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    functional = None


class UgiChemistryFlowError(RuntimeError):
    """Raised when Ugi chemistry tensors violate their topology contract."""


TERMINAL_DECODE_FAILURE_SCHEMA = "forge.ugi_terminal_decode_failure.v1"


class UgiTerminalDecodeError(UgiChemistryFlowError):
    """One typed, serializable terminal-support failure."""

    def __init__(self, code: str, stage: str, **context: int | str | bool) -> None:
        if not code or not stage:
            raise ValueError("terminal decode failure code and stage must be nonempty")
        self.code = code
        self.stage = stage
        self.context = dict(sorted(context.items()))
        super().__init__(f"{stage}:{code}")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": TERMINAL_DECODE_FAILURE_SCHEMA,
            "stage": self.stage,
            "code": self.code,
            "context": self.context,
        }


TERMINAL_DECODER_MODES = (
    "argmax",
    "stochastic",
    "bond_stochastic",
    "atom_bond_stochastic",
    "decoration_bond_stochastic",
)


@dataclass(frozen=True)
class UgiChemistrySample:
    """Generated chemistry on one fixed sparse Ugi support topology."""

    atom_states: np.ndarray
    parent_bond_states: np.ndarray
    closure_bond_states: np.ndarray
    decoration_anchor: int
    decoration_anchors: np.ndarray | None = None
    decoration_atom_states: np.ndarray | None = None
    decoration_bond_states: np.ndarray | None = None


def _empty_batch(batch_size: int, maximum_nodes: int, maximum_closures: int) -> dict[str, Any]:
    if torch is None:
        raise UgiChemistryFlowError("Ugi chemistry collation requires torch")
    return {
        "node_mask": torch.zeros((batch_size, maximum_nodes), dtype=torch.bool),
        "child_mask": torch.zeros((batch_size, maximum_nodes), dtype=torch.bool),
        "closure_mask": torch.zeros((batch_size, maximum_closures), dtype=torch.bool),
        "parents": torch.zeros((batch_size, maximum_nodes), dtype=torch.long),
        "closure_left": torch.zeros((batch_size, maximum_closures), dtype=torch.long),
        "closure_right": torch.zeros((batch_size, maximum_closures), dtype=torch.long),
        "origin_states": torch.zeros((batch_size, maximum_nodes), dtype=torch.long),
        "core_position_states": torch.zeros((batch_size, maximum_nodes), dtype=torch.long),
        "port_states": torch.zeros((batch_size, maximum_nodes), dtype=torch.long),
        "distance_to_core": torch.zeros((batch_size, maximum_nodes), dtype=torch.long),
        "distance_to_own_port": torch.full(
            (batch_size, maximum_nodes),
            -1,
            dtype=torch.long,
        ),
        "fixed_atom_mask": torch.zeros((batch_size, maximum_nodes), dtype=torch.bool),
        "fixed_parent_bond_mask": torch.zeros((batch_size, maximum_nodes), dtype=torch.bool),
        "fixed_closure_bond_mask": torch.zeros((batch_size, maximum_closures), dtype=torch.bool),
        "fixed_atom_states": torch.full((batch_size, maximum_nodes), -1, dtype=torch.long),
        "fixed_parent_bond_states": torch.full((batch_size, maximum_nodes), -1, dtype=torch.long),
        "fixed_closure_bond_states": torch.full(
            (batch_size, maximum_closures), -1, dtype=torch.long
        ),
    }


def collate_ugi_chemistry_conditions(
    conditions: Sequence[ChemistryTopologyCondition],
    *,
    maximum_nodes: int,
    maximum_closures: int,
) -> dict[str, Any]:
    if not conditions:
        raise UgiChemistryFlowError("Ugi chemistry collation requires conditions")
    if max(value.node_count for value in conditions) > maximum_nodes:
        raise UgiChemistryFlowError("chemistry batch exceeds node capacity")
    if max(value.closure_count for value in conditions) > maximum_closures:
        raise UgiChemistryFlowError("chemistry batch exceeds closure capacity")
    batch = _empty_batch(len(conditions), maximum_nodes, maximum_closures)
    for index, condition in enumerate(conditions):
        node_count = condition.node_count
        closure_count = condition.closure_count
        distances = recompute_adapter_distances(condition)
        batch["node_mask"][index, :node_count] = True
        batch["child_mask"][index, 1:node_count] = True
        batch["closure_mask"][index, :closure_count] = True
        batch["parents"][index, :node_count] = torch.from_numpy(condition.parents.copy())
        batch["closure_left"][index, :closure_count] = torch.from_numpy(
            condition.closure_left.copy()
        )
        batch["closure_right"][index, :closure_count] = torch.from_numpy(
            condition.closure_right.copy()
        )
        for label, values in (
            ("origin_states", condition.origin_states),
            ("core_position_states", condition.core_position_states),
            ("port_states", condition.port_states),
            ("distance_to_core", distances.distance_to_core),
            ("distance_to_own_port", distances.distance_to_own_port),
            ("fixed_atom_mask", condition.fixed_atom_mask),
            ("fixed_parent_bond_mask", condition.fixed_parent_bond_mask),
            ("fixed_atom_states", condition.fixed_atom_states),
            ("fixed_parent_bond_states", condition.fixed_parent_bond_states),
        ):
            batch[label][index, :node_count] = torch.from_numpy(values.copy())
        batch["fixed_closure_bond_mask"][index, :closure_count] = torch.from_numpy(
            condition.fixed_closure_bond_mask.copy()
        )
        batch["fixed_closure_bond_states"][index, :closure_count] = torch.from_numpy(
            condition.fixed_closure_bond_states.copy()
        )
    batch["atom_variable_mask"] = batch["node_mask"] & ~batch["fixed_atom_mask"]
    batch["parent_bond_variable_mask"] = batch["child_mask"] & ~batch["fixed_parent_bond_mask"]
    batch["closure_bond_variable_mask"] = batch["closure_mask"] & ~batch["fixed_closure_bond_mask"]
    batch["adapter_mask"] = batch["node_mask"].clone()
    return batch


def collate_ugi_chemistry_records(
    records: Sequence[UgiChemistryRecord],
    *,
    maximum_nodes: int,
    maximum_closures: int,
    maximum_decorations: int = 1,
) -> dict[str, Any]:
    """Collate exact topologies and withheld chemistry labels."""

    conditions = tuple(record.condition for record in records)
    batch = collate_ugi_chemistry_conditions(
        conditions,
        maximum_nodes=maximum_nodes,
        maximum_closures=maximum_closures,
    )
    nodes = torch.zeros((len(records), maximum_nodes), dtype=torch.long)
    parent_bonds = torch.zeros_like(nodes)
    closure_bonds = torch.zeros((len(records), maximum_closures), dtype=torch.long)
    if maximum_decorations < 1:
        raise UgiChemistryFlowError("chemistry requires positive decoration capacity")
    decoration_anchor = torch.zeros(len(records), dtype=torch.long)
    decoration_anchors = torch.zeros(
        (len(records), maximum_decorations),
        dtype=torch.long,
    )
    decoration_atoms = torch.zeros_like(decoration_anchors)
    decoration_bonds = torch.zeros_like(decoration_anchors)
    for index, record in enumerate(records):
        target = record.target
        node_count = record.condition.node_count
        closure_count = record.condition.closure_count
        if target.node_count != node_count:
            raise UgiChemistryFlowError("chemistry target changed generated node count")
        nodes[index, :node_count] = torch.from_numpy(target.atom_states.copy())
        if int(target.parent_bond_states[0]) != ROOT_BOND_TARGET:
            raise UgiChemistryFlowError("chemistry target lacks the root bond sentinel")
        parent_bonds[index, 1:node_count] = torch.from_numpy(target.parent_bond_states[1:].copy())
        closure_bonds[index, :closure_count] = torch.from_numpy(target.closure_bond_states.copy())
        if target.decorations.count > maximum_decorations:
            raise UgiChemistryFlowError(
                "chemistry target exceeds declared terminal-decoration capacity"
            )
        if maximum_decorations == 1 and target.decorations.count > 1:
            raise UgiChemistryFlowError(
                "single-decoration chemistry mode received multiple terminal atoms"
            )
        if target.decorations.count:
            decoration_anchor[index] = int(target.decorations.anchor_indices[0]) + 1
            count = target.decorations.count
            decoration_anchors[index, :count] = (
                torch.from_numpy(target.decorations.anchor_indices.copy()) + 1
            )
            decoration_atoms[index, :count] = torch.from_numpy(
                target.decorations.atom_states.copy()
            )
            decoration_bonds[index, :count] = torch.from_numpy(
                target.decorations.bond_states.copy()
            )
    batch.update(
        {
            "nodes": nodes,
            "parent_bonds": parent_bonds,
            "closure_bonds": closure_bonds,
            # Zero means no terminal O= decoration; node i occupies state i+1.
            "decoration_anchor": decoration_anchor,
        }
    )
    if maximum_decorations > 1:
        batch.update(
            {
                "decoration_anchors": decoration_anchors,
                "decoration_atoms": decoration_atoms,
                "decoration_bonds": decoration_bonds,
                "decoration_present_mask": decoration_anchors > 0,
            }
        )
    return batch


def chemistry_source_marginals(
    records: Sequence[UgiChemistryRecord],
    *,
    atom_classes: int,
    bond_classes: int,
    probability_floor: float,
    record_weights: np.ndarray | None = None,
    maximum_decorations: int = 1,
) -> dict[str, np.ndarray]:
    """Estimate full-support role-aware chemistry sources from training only."""

    if (
        not records
        or atom_classes < 1
        or bond_classes < 1
        or maximum_decorations < 1
        or not 0 < probability_floor < 1
    ):
        raise UgiChemistryFlowError("invalid chemistry source-marginal request")
    if record_weights is None:
        weights = np.ones(len(records), dtype=np.float64)
    else:
        weights = np.asarray(record_weights, dtype=np.float64)
        if weights.shape != (len(records),) or np.any(weights < 0) or weights.sum() <= 0:
            raise UgiChemistryFlowError("invalid chemistry source record weights")
        weights = weights * (len(records) / weights.sum())
    atoms = np.full((len(ROLE_NAMES), atom_classes), probability_floor, dtype=np.float64)
    bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
    decoration = np.full(2, probability_floor, dtype=np.float64)
    decoration_atoms = np.full(atom_classes, probability_floor, dtype=np.float64)
    decoration_bonds = np.full(bond_classes, probability_floor, dtype=np.float64)
    for record, weight in zip(records, weights, strict=True):
        condition = record.condition
        target = record.target
        for role_index, role in enumerate(ROLE_NAMES):
            selected = (
                condition.origin_states == ORIGIN_TO_INDEX[role]
            ) & ~condition.fixed_atom_mask
            atoms[role_index] += weight * np.bincount(
                target.atom_states[selected], minlength=atom_classes
            )
        bonds += weight * np.bincount(
            target.parent_bond_states[1:][~condition.fixed_parent_bond_mask[1:]],
            minlength=bond_classes,
        )
        bonds += weight * np.bincount(
            target.closure_bond_states[~condition.fixed_closure_bond_mask],
            minlength=bond_classes,
        )
        if target.decorations.count > maximum_decorations:
            raise UgiChemistryFlowError("chemistry target exceeds the decoration source capacity")
        # In the legacy one-slot model this is the molecule-level presence
        # marginal.  In the expanded model it must be the slot-level
        # presence marginal; otherwise every one of the sparse slots is
        # initialized as present whenever every molecule has at least one
        # decoration.
        decoration[1] += weight * target.decorations.count
        decoration[0] += weight * (maximum_decorations - target.decorations.count)
        decoration_atoms += weight * np.bincount(
            target.decorations.atom_states,
            minlength=atom_classes,
        )
        decoration_bonds += weight * np.bincount(
            target.decorations.bond_states,
            minlength=bond_classes,
        )
    atoms /= atoms.sum(axis=1, keepdims=True)
    bonds /= bonds.sum()
    decoration /= decoration.sum()
    decoration_atoms /= decoration_atoms.sum()
    decoration_bonds /= decoration_bonds.sum()
    return {
        "atoms": atoms,
        "bonds": bonds,
        "decoration": decoration,
        "decoration_atoms": decoration_atoms,
        "decoration_bonds": decoration_bonds,
    }


def _gather_nodes(hidden: Any, indices: Any) -> Any:
    return torch.gather(
        hidden,
        1,
        indices[:, :, None].expand(-1, -1, hidden.shape[-1]),
    )


if nn is not None:

    class UgiChemistryFlow(nn.Module):
        """Sparse graph denoiser for atoms, bonds, and one Ugi decoration."""

        def __init__(
            self,
            *,
            atom_classes: int,
            bond_classes: int,
            maximum_nodes: int,
            maximum_distance: int,
            hidden_dim: int,
            layers: int,
            dropout: float,
            maximum_decorations: int = 1,
        ) -> None:
            super().__init__()
            if (
                atom_classes < 1
                or bond_classes not in {3, 4}
                or maximum_nodes < 8
                or maximum_distance < 1
                or hidden_dim < 16
                or layers < 1
                or not 0 <= dropout < 1
                or maximum_decorations < 1
            ):
                raise UgiChemistryFlowError("invalid Ugi chemistry architecture")
            self.atom_classes = atom_classes
            self.bond_classes = bond_classes
            self.hidden_dim = hidden_dim
            self.maximum_nodes = maximum_nodes
            self.maximum_decorations = maximum_decorations
            self.atom_embedding = nn.Embedding(atom_classes, hidden_dim)
            self.bond_embedding = nn.Embedding(bond_classes, hidden_dim)
            self.adapter_conditioning = AdapterNodeConditioning(
                hidden_dim=hidden_dim,
                maximum_distance=maximum_distance,
            )
            self.time_embedding = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.blocks = nn.ModuleList(
                DeterministicSparseFlowBlock(hidden_dim, dropout) for _ in range(layers)
            )
            self.atom_output = nn.Linear(hidden_dim, atom_classes)
            self.parent_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.closure_bond_output = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, bond_classes),
            )
            self.decoration_query = nn.Linear(hidden_dim, hidden_dim)
            self.decoration_key = nn.Linear(hidden_dim, hidden_dim)
            self.no_decoration = nn.Parameter(torch.zeros(hidden_dim))
            if maximum_decorations > 1:
                self.decoration_slot_embedding = nn.Embedding(
                    maximum_decorations,
                    hidden_dim,
                )
                self.decoration_atom_output = nn.Linear(hidden_dim, atom_classes)
                self.decoration_bond_output = nn.Linear(hidden_dim, bond_classes)
            else:
                self.decoration_slot_embedding = None
                self.decoration_atom_output = None
                self.decoration_bond_output = None

        def forward(
            self,
            *,
            nodes: Any,
            parent_bonds: Any,
            closure_bonds: Any,
            t: Any,
            topology: dict[str, Any],
        ) -> dict[str, Any]:
            node_mask = topology["node_mask"]
            child_mask = topology["child_mask"]
            closure_mask = topology["closure_mask"]
            if (
                nodes.shape != node_mask.shape
                or parent_bonds.shape != node_mask.shape
                or closure_bonds.shape != closure_mask.shape
                or t.shape != (nodes.shape[0],)
            ):
                raise UgiChemistryFlowError("Ugi chemistry state shapes do not agree")
            adapter_hidden = self.adapter_conditioning(
                origin_states=topology["origin_states"],
                core_position_states=topology["core_position_states"],
                port_states=topology["port_states"],
                distance_to_core=topology["distance_to_core"],
                distance_to_own_port=topology["distance_to_own_port"],
                adapter_mask=topology["adapter_mask"],
            )
            hidden = (
                self.atom_embedding(nodes)
                + self.bond_embedding(parent_bonds) * child_mask[:, :, None]
                + adapter_hidden
                + self.time_embedding(t[:, None])[:, None, :]
            )
            hidden = self.input_norm(hidden) * node_mask[:, :, None]
            closure_hidden = self.bond_embedding(closure_bonds) * closure_mask[:, :, None]
            for block in self.blocks:
                hidden = block(
                    hidden,
                    topology["parents"],
                    topology["closure_left"],
                    topology["closure_right"],
                    closure_hidden,
                    node_mask,
                    child_mask,
                    closure_mask,
                )
            parent_hidden = _gather_nodes(hidden, topology["parents"])
            left_hidden = _gather_nodes(hidden, topology["closure_left"])
            right_hidden = _gather_nodes(hidden, topology["closure_right"])
            global_hidden = (hidden * node_mask[:, :, None]).sum(dim=1)
            global_hidden /= node_mask.sum(dim=1, keepdim=True).clamp(min=1)
            query = self.decoration_query(global_hidden)
            node_keys = self.decoration_key(hidden)
            output = {
                "nodes": self.atom_output(hidden),
                "parent_bonds": self.parent_bond_output(torch.cat((hidden, parent_hidden), dim=-1)),
                "closure_bonds": self.closure_bond_output(
                    torch.cat((left_hidden, right_hidden), dim=-1)
                ),
            }
            if self.maximum_decorations == 1:
                node_scores = torch.einsum("bd,bnd->bn", query, node_keys) / math.sqrt(
                    self.hidden_dim
                )
                node_scores = node_scores.masked_fill(
                    ~topology["atom_variable_mask"],
                    -torch.inf,
                )
                none_score = (query * self.no_decoration[None, :]).sum(
                    dim=1,
                    keepdim=True,
                )
                output["decoration_anchor"] = torch.cat((none_score, node_scores), dim=1)
                return output
            slots = torch.arange(self.maximum_decorations, device=hidden.device)
            slot_hidden = query[:, None, :] + self.decoration_slot_embedding(slots)[None, :, :]
            node_scores = torch.einsum("bsd,bnd->bsn", slot_hidden, node_keys) / math.sqrt(
                self.hidden_dim
            )
            node_scores = node_scores.masked_fill(
                ~topology["atom_variable_mask"][:, None, :],
                -torch.inf,
            )
            none_score = torch.einsum(
                "bsd,d->bs",
                slot_hidden,
                self.no_decoration,
            )[:, :, None]
            output.update(
                {
                    "decoration_anchors": torch.cat((none_score, node_scores), dim=2),
                    "decoration_atoms": self.decoration_atom_output(slot_hidden),
                    "decoration_bonds": self.decoration_bond_output(slot_hidden),
                }
            )
            return output

else:  # pragma: no cover

    class UgiChemistryFlow:  # type: ignore[no-redef]
        def __init__(self, **_: Any) -> None:
            raise UgiChemistryFlowError("Ugi chemistry flow requires torch")


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
    example = torch.arange(clean.shape[0], device=clean.device)[:, None].expand_as(clean)[mask]
    probabilities = marginal.clone()
    if probabilities.ndim == 1:
        probabilities = probabilities[None, :].expand(int(mask.sum()), -1).clone()
    elif probabilities.shape[:2] == (*clean.shape,):
        probabilities = probabilities[mask].clone()
    elif probabilities.ndim != 2 or probabilities.shape[0] != int(mask.sum()):
        raise UgiChemistryFlowError("chemistry interpolation marginal is misaligned")
    probabilities *= 1.0 - t[example, None]
    probabilities.scatter_add_(1, clean[mask][:, None], t[example, None])
    output[mask] = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
    return output


def noise_ugi_chemistry_batch(
    clean: dict[str, Any],
    sources: dict[str, Any],
    t: Any,
    generator: Any,
) -> dict[str, Any]:
    """Corrupt only chemistry variables; topology and fixed core remain exact."""

    atom_sources = atom_source_by_node(clean, sources["atoms"])
    nodes = _sample_interpolation(
        clean["nodes"],
        atom_sources,
        t,
        clean["atom_variable_mask"],
        generator,
    )
    parent_bonds = _sample_interpolation(
        clean["parent_bonds"],
        sources["bonds"],
        t,
        clean["parent_bond_variable_mask"],
        generator,
    )
    closure_bonds = _sample_interpolation(
        clean["closure_bonds"],
        sources["bonds"],
        t,
        clean["closure_bond_variable_mask"],
        generator,
    )
    output = {
        "nodes": nodes,
        "parent_bonds": parent_bonds,
        "closure_bonds": closure_bonds,
    }
    decoration_source = decoration_source_marginal(clean, sources["decoration"])
    if "decoration_anchors" not in clean:
        decoration_probabilities = decoration_source * (1.0 - t[:, None])
        decoration_probabilities.scatter_add_(
            1,
            clean["decoration_anchor"][:, None],
            t[:, None],
        )
        output["decoration_anchor"] = torch.multinomial(
            decoration_probabilities,
            1,
            generator=generator,
        ).squeeze(1)
        return output
    slots = clean["decoration_anchors"].shape[1]
    anchor_sources = decoration_source[:, None, :].expand(-1, slots, -1)
    slot_mask = torch.ones_like(clean["decoration_anchors"], dtype=torch.bool)
    output["decoration_anchors"] = _sample_interpolation(
        clean["decoration_anchors"],
        anchor_sources,
        t,
        slot_mask,
        generator,
    )
    output["decoration_atoms"] = _sample_interpolation(
        clean["decoration_atoms"],
        sources["decoration_atoms"],
        t,
        clean["decoration_present_mask"],
        generator,
    )
    output["decoration_bonds"] = _sample_interpolation(
        clean["decoration_bonds"],
        sources["decoration_bonds"],
        t,
        clean["decoration_present_mask"],
        generator,
    )
    return output


def decoration_source_marginal(
    topology: dict[str, Any],
    binary_source: Any,
) -> Any:
    """Expand none/present mass over each example's valid anchor support."""

    if binary_source.shape != (2,):
        raise UgiChemistryFlowError("decoration source must contain none/present mass")
    valid = topology["atom_variable_mask"]
    output = torch.zeros(
        (valid.shape[0], valid.shape[1] + 1),
        dtype=binary_source.dtype,
        device=valid.device,
    )
    output[:, 0] = binary_source[0]
    counts = valid.sum(dim=1, keepdim=True)
    if torch.any(counts < 1):
        raise UgiChemistryFlowError("decoration source lacks a valid non-core anchor")
    output[:, 1:] = valid.to(output.dtype) * (binary_source[1] / counts)
    return output


def atom_source_by_node(topology: dict[str, Any], role_sources: Any) -> Any:
    """Broadcast the three role sources to adapter origin-state indices."""

    if role_sources.ndim != 2 or role_sources.shape[0] != len(ROLE_NAMES):
        raise UgiChemistryFlowError("atom sources must contain one row per Ugi role")
    output = (
        role_sources[0][None, None, :]
        .expand(
            *topology["origin_states"].shape,
            -1,
        )
        .clone()
    )
    for role_index, role in enumerate(ROLE_NAMES):
        mask = topology["origin_states"] == ORIGIN_TO_INDEX[role]
        output[mask] = role_sources[role_index]
    return output


def ugi_chemistry_flow_loss(
    predictions: dict[str, Any],
    clean: dict[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Balance atom and bond learning across the three precursor roles."""

    atom_losses = []
    parent_losses = []
    metrics: dict[str, float] = {}
    for role_index, role in enumerate(ROLE_NAMES):
        origin = ORIGIN_TO_INDEX[role]
        atom_mask = clean["atom_variable_mask"] & (clean["origin_states"] == origin)
        parent_mask = clean["parent_bond_variable_mask"] & (clean["origin_states"] == origin)
        if not bool(atom_mask.any()) or not bool(parent_mask.any()):
            raise UgiChemistryFlowError(f"chemistry batch lacks {role} supervision")
        atom_loss = functional.cross_entropy(
            predictions["nodes"][atom_mask],
            clean["nodes"][atom_mask],
        )
        parent_loss = functional.cross_entropy(
            predictions["parent_bonds"][parent_mask],
            clean["parent_bonds"][parent_mask],
        )
        atom_losses.append(atom_loss)
        parent_losses.append(parent_loss)
        metrics[f"{role}_atom_ce"] = float(atom_loss.detach())
        metrics[f"{role}_parent_bond_ce"] = float(parent_loss.detach())
    atom_total = torch.stack(atom_losses).mean()
    parent_total = torch.stack(parent_losses).mean()
    closure_mask = clean["closure_bond_variable_mask"]
    if bool(closure_mask.any()):
        closure_loss = functional.cross_entropy(
            predictions["closure_bonds"][closure_mask],
            clean["closure_bonds"][closure_mask],
        )
    else:
        closure_loss = atom_total * 0.0
    if "decoration_anchors" not in clean:
        decoration_anchor_loss = functional.cross_entropy(
            predictions["decoration_anchor"],
            clean["decoration_anchor"],
        )
        decoration_atom_loss = atom_total * 0.0
        decoration_bond_loss = atom_total * 0.0
    else:
        decoration_anchor_loss = functional.cross_entropy(
            predictions["decoration_anchors"].flatten(0, 1),
            clean["decoration_anchors"].flatten(),
        )
        decoration_mask = clean["decoration_present_mask"]
        if bool(decoration_mask.any()):
            decoration_atom_loss = functional.cross_entropy(
                predictions["decoration_atoms"][decoration_mask],
                clean["decoration_atoms"][decoration_mask],
            )
            decoration_bond_loss = functional.cross_entropy(
                predictions["decoration_bonds"][decoration_mask],
                clean["decoration_bonds"][decoration_mask],
            )
        else:
            decoration_atom_loss = atom_total * 0.0
            decoration_bond_loss = atom_total * 0.0
    decoration_loss = decoration_anchor_loss + decoration_atom_loss + decoration_bond_loss
    total = atom_total + parent_total + closure_loss + decoration_loss
    metrics.update(
        {
            "atom_ce": float(atom_total.detach()),
            "parent_bond_ce": float(parent_total.detach()),
            "closure_bond_ce": float(closure_loss.detach()),
            "decoration_ce": float(decoration_loss.detach()),
            "decoration_anchor_ce": float(decoration_anchor_loss.detach()),
            "decoration_atom_ce": float(decoration_atom_loss.detach()),
            "decoration_bond_ce": float(decoration_bond_loss.detach()),
            "total": float(total.detach()),
        }
    )
    return total, metrics


def rstar_ugi_chemistry_step(
    state: dict[str, Any],
    predictions: dict[str, Any],
    topology: dict[str, Any],
    sources: dict[str, Any],
    *,
    t: float,
    dt: float,
    generator: Any,
) -> dict[str, Any]:
    """Apply one discrete-flow step while clamping adapter-fixed chemistry."""

    atom_sources = atom_source_by_node(topology, sources["atoms"])
    output = {
        "nodes": _rstar_step(
            state["nodes"],
            predictions["nodes"].softmax(dim=-1),
            atom_sources,
            t,
            dt,
            topology["atom_variable_mask"],
            generator,
        ),
        "parent_bonds": _rstar_step(
            state["parent_bonds"],
            predictions["parent_bonds"].softmax(dim=-1),
            sources["bonds"],
            t,
            dt,
            topology["parent_bond_variable_mask"],
            generator,
        ),
        "closure_bonds": _rstar_step(
            state["closure_bonds"],
            predictions["closure_bonds"].softmax(dim=-1),
            sources["bonds"],
            t,
            dt,
            topology["closure_bond_variable_mask"],
            generator,
        ),
    }
    decoration_source = decoration_source_marginal(topology, sources["decoration"])
    if "decoration_anchors" not in state:
        output["decoration_anchor"] = _rstar_step(
            state["decoration_anchor"],
            predictions["decoration_anchor"].softmax(dim=-1),
            decoration_source,
            t,
            dt,
            torch.ones_like(state["decoration_anchor"], dtype=torch.bool),
            generator,
        )
        return output
    slots = state["decoration_anchors"].shape[1]
    output["decoration_anchors"] = _rstar_step(
        state["decoration_anchors"],
        predictions["decoration_anchors"].softmax(dim=-1),
        decoration_source[:, None, :].expand(-1, slots, -1),
        t,
        dt,
        torch.ones_like(state["decoration_anchors"], dtype=torch.bool),
        generator,
    )
    output["decoration_atoms"] = _rstar_step(
        state["decoration_atoms"],
        predictions["decoration_atoms"].softmax(dim=-1),
        sources["decoration_atoms"],
        t,
        dt,
        torch.ones_like(state["decoration_atoms"], dtype=torch.bool),
        generator,
    )
    output["decoration_bonds"] = _rstar_step(
        state["decoration_bonds"],
        predictions["decoration_bonds"].softmax(dim=-1),
        sources["decoration_bonds"],
        t,
        dt,
        torch.ones_like(state["decoration_bonds"], dtype=torch.bool),
        generator,
    )
    return output


def _initial_chemistry_state(
    topology: dict[str, Any],
    sources: dict[str, Any],
    generator: Any,
) -> dict[str, Any]:
    """Draw a source state and clamp only adapter-owned Ugi chemistry."""

    atom_probabilities = atom_source_by_node(topology, sources["atoms"])
    nodes = torch.multinomial(
        atom_probabilities.reshape(-1, atom_probabilities.shape[-1]),
        1,
        generator=generator,
    ).reshape(topology["node_mask"].shape)
    parent_bonds = torch.multinomial(
        sources["bonds"],
        int(topology["node_mask"].numel()),
        replacement=True,
        generator=generator,
    ).reshape(topology["node_mask"].shape)
    if topology["closure_mask"].numel():
        closure_bonds = torch.multinomial(
            sources["bonds"],
            int(topology["closure_mask"].numel()),
            replacement=True,
            generator=generator,
        ).reshape(topology["closure_mask"].shape)
    else:
        closure_bonds = torch.zeros_like(topology["closure_mask"], dtype=torch.long)
    fixed_atoms = topology["fixed_atom_mask"]
    fixed_parent = topology["fixed_parent_bond_mask"]
    fixed_closure = topology["fixed_closure_bond_mask"]
    nodes[fixed_atoms] = topology["fixed_atom_states"][fixed_atoms]
    parent_bonds[fixed_parent] = topology["fixed_parent_bond_states"][fixed_parent]
    closure_bonds[fixed_closure] = topology["fixed_closure_bond_states"][fixed_closure]
    parent_bonds[:, 0] = 0
    output = {
        "nodes": nodes,
        "parent_bonds": parent_bonds,
        "closure_bonds": closure_bonds,
    }
    decoration_source = decoration_source_marginal(topology, sources["decoration"])
    maximum_decorations = int(topology.get("maximum_decorations", 1))
    if maximum_decorations == 1:
        output["decoration_anchor"] = torch.multinomial(
            decoration_source,
            1,
            generator=generator,
        ).squeeze(1)
        return output
    anchor_source = decoration_source[:, None, :].expand(-1, maximum_decorations, -1)
    output["decoration_anchors"] = torch.multinomial(
        anchor_source.reshape(-1, anchor_source.shape[-1]),
        1,
        generator=generator,
    ).reshape(anchor_source.shape[:2])
    output["decoration_atoms"] = torch.multinomial(
        sources["decoration_atoms"],
        int(topology["node_mask"].shape[0] * maximum_decorations),
        replacement=True,
        generator=generator,
    ).reshape(topology["node_mask"].shape[0], maximum_decorations)
    output["decoration_bonds"] = torch.multinomial(
        sources["decoration_bonds"],
        int(topology["node_mask"].shape[0] * maximum_decorations),
        replacement=True,
        generator=generator,
    ).reshape(topology["node_mask"].shape[0], maximum_decorations)
    return output


def _atom_valence_units(state: AtomState) -> int:
    """Return the declared heavy-bond capacity after explicit hydrogens."""

    if state.symbol == "C":
        maximum = 8
    elif state.symbol == "F":
        maximum = 2
    elif state.symbol == "N":
        maximum = 8 if state.formal_charge > 0 else 6
    elif state.symbol == "O":
        maximum = 2 if state.formal_charge < 0 else 4
    elif state.symbol == "P":
        maximum = 10
    elif state.symbol == "S":
        maximum = 12
    elif state.symbol == "Si":
        maximum = 8
    else:  # pragma: no cover - vocabulary construction rejects this first
        raise UgiChemistryFlowError(f"no valence policy for atom state {state}")
    return maximum - 2 * state.explicit_hydrogens


_BOND_VALENCE_UNITS = np.asarray([2, 4, 6, 3], dtype=np.int64)


def _tree_path_edges(parents: np.ndarray, left: int, right: int) -> set[tuple[int, int]]:
    adjacency = [[] for _ in range(len(parents))]
    for child in range(1, len(parents)):
        parent = int(parents[child])
        adjacency[parent].append(child)
        adjacency[child].append(parent)
    previous = {left: -1}
    queue = [left]
    for node in queue:
        if node == right:
            break
        for neighbor in adjacency[node]:
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)
    if right not in previous:
        raise UgiChemistryFlowError("conditioned chemistry tree is disconnected")
    edges: set[tuple[int, int]] = set()
    node = right
    while previous[node] >= 0:
        parent = previous[node]
        edges.add(tuple(sorted((node, parent))))
        node = parent
    return edges


def _masked_terminal_choice(
    logits: Any,
    valid: Any,
    *,
    mode: str,
    generator: Any | None,
    temperature: float,
) -> int:
    """Choose one feasible terminal state without changing its support."""

    if torch is None or mode not in {"argmax", "stochastic"} or temperature <= 0:
        raise UgiChemistryFlowError("invalid terminal decoder configuration")
    if logits.ndim != 1 or valid.shape != logits.shape or not bool(valid.any()):
        raise UgiChemistryFlowError("terminal choice has no feasible state")
    masked = logits.masked_fill(~valid, -torch.inf)
    if mode == "argmax":
        return int(masked.argmax())
    if generator is None:
        raise UgiChemistryFlowError("stochastic terminal decoding requires a generator")
    probabilities = (masked / temperature).softmax(dim=0)
    return int(torch.multinomial(probabilities, 1, generator=generator).item())


def _terminal_choice_modes(mode: str) -> tuple[str, str, str]:
    """Resolve atom, bond and decoration readouts for one declared decoder mode."""

    policies = {
        "argmax": ("argmax", "argmax", "argmax"),
        "stochastic": ("stochastic", "stochastic", "stochastic"),
        "bond_stochastic": ("argmax", "stochastic", "argmax"),
        "atom_bond_stochastic": ("stochastic", "stochastic", "argmax"),
        "decoration_bond_stochastic": ("argmax", "stochastic", "stochastic"),
    }
    try:
        return policies[mode]
    except KeyError as error:
        raise UgiChemistryFlowError(f"unsupported terminal decoder mode: {mode}") from error


def _terminal_channel_temperatures(
    condition: ChemistryTopologyCondition,
    *,
    temperature: float,
    atom_temperature: float | None,
    bond_temperature: float | None,
    decoration_temperature: float | None,
    atom_temperatures_by_origin: Sequence[float] | None,
) -> tuple[np.ndarray, float, float]:
    """Resolve backward-compatible channel and role-conditional temperatures."""

    base = float(temperature)
    resolved_atom = base if atom_temperature is None else float(atom_temperature)
    resolved_bond = base if bond_temperature is None else float(bond_temperature)
    resolved_decoration = base if decoration_temperature is None else float(decoration_temperature)
    if min(base, resolved_atom, resolved_bond, resolved_decoration) <= 0:
        raise UgiChemistryFlowError("terminal temperatures must be positive")
    atom_by_node = np.full(condition.node_count, resolved_atom, dtype=np.float64)
    if atom_temperatures_by_origin is not None:
        per_origin = np.asarray(atom_temperatures_by_origin, dtype=np.float64)
        if (
            per_origin.ndim != 1
            or per_origin.size != len(ROLE_NAMES)
            or not np.all(np.isfinite(per_origin))
            or np.any(per_origin <= 0)
        ):
            raise UgiChemistryFlowError("invalid origin-conditional atom temperatures")
        for role_index, role in enumerate(ROLE_NAMES):
            atom_by_node[condition.origin_states == ORIGIN_TO_INDEX[role]] = per_origin[role_index]
    return atom_by_node, resolved_bond, resolved_decoration


def valence_constrained_terminal_sample(
    condition: ChemistryTopologyCondition,
    terminal: dict[str, Any],
    batch_index: int,
    atom_vocabulary: tuple[AtomState, ...],
    maximum_decorations: int,
    forbid_oxygen_oxygen_bonds: bool = True,
    *,
    mode: str = "argmax",
    generator: Any | None = None,
    temperature: float = 1.0,
    atom_temperature: float | None = None,
    bond_temperature: float | None = None,
    decoration_temperature: float | None = None,
    atom_temperatures_by_origin: Sequence[float] | None = None,
    local_chemistry_support: LocalChemistrySupport | None = None,
    program_id: str | None = None,
    local_chemistry_constraint_scope: str = "role_edges_cycles_bounds",
) -> UgiChemistrySample:
    """Decode terminal logits only within valence-feasible sparse support.

    This is a support-constrained decoder, not structural repair: topology is
    unchanged, no atom or edge is added except a decoration explicitly emitted
    by a learned slot, and every categorical decision is selected from its
    model logits after impossible states are masked.
    """

    node_count = condition.node_count
    closure_count = condition.closure_count
    if local_chemistry_constraint_scope not in {
        "role_edges_only",
        "role_edges_cycles_bounds",
    }:
        raise UgiChemistryFlowError("unsupported role-local chemistry constraint scope")
    if local_chemistry_support is not None:
        if program_id is None:
            raise UgiChemistryFlowError(
                "role-local terminal decoding requires an explicit program_id"
            )
        if tuple(local_chemistry_support.atom_states) != tuple(atom_vocabulary):
            raise UgiChemistryFlowError(
                "role-local chemistry atom vocabulary differs from the decoder vocabulary"
            )
        role_by_origin = {index: name for name, index in ORIGIN_TO_INDEX.items()}
        try:
            role_names = tuple(role_by_origin[int(value)] for value in condition.origin_states)
        except KeyError as error:
            raise UgiChemistryFlowError(
                "condition contains an origin absent from the role-local policy"
            ) from error
        if any(role == "adapter_unspecified" for role in role_names):
            raise UgiChemistryFlowError(
                "condition contains an adapter-unspecified role during Ugi decoding"
            )
        for role in sorted(set(role_names)):
            local_chemistry_support.component_support_bounds(program_id, role)
    else:
        role_names = tuple("" for _ in range(node_count))
    atom_choice_mode, bond_choice_mode, decoration_choice_mode = _terminal_choice_modes(mode)
    atom_temperatures, resolved_bond_temperature, resolved_decoration_temperature = (
        _terminal_channel_temperatures(
            condition,
            temperature=temperature,
            atom_temperature=atom_temperature,
            bond_temperature=bond_temperature,
            decoration_temperature=decoration_temperature,
            atom_temperatures_by_origin=atom_temperatures_by_origin,
        )
    )
    parents = condition.parents
    degrees = np.zeros(node_count, dtype=np.int64)
    for child in range(1, node_count):
        degrees[child] += 1
        degrees[int(parents[child])] += 1
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        degrees[int(left)] += 1
        degrees[int(right)] += 1

    fixed_extra = np.zeros(node_count, dtype=np.int64)
    for child in range(1, node_count):
        if condition.fixed_parent_bond_mask[child]:
            units = int(_BOND_VALENCE_UNITS[int(condition.fixed_parent_bond_states[child])])
            extra = units - 2
            fixed_extra[child] += extra
            fixed_extra[int(parents[child])] += extra
    for slot, (left, right) in enumerate(
        zip(condition.closure_left, condition.closure_right, strict=True)
    ):
        if condition.fixed_closure_bond_mask[slot]:
            units = int(_BOND_VALENCE_UNITS[int(condition.fixed_closure_bond_states[slot])])
            extra = units - 2
            fixed_extra[int(left)] += extra
            fixed_extra[int(right)] += extra
    minimum_units = 2 * degrees + fixed_extra

    cycle_edges: set[tuple[int, int]] = set()
    cycle_nodes_by_closure: list[set[int]] = []
    cycle_paths_by_closure: list[tuple[int, ...]] = []
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        path_edges = _tree_path_edges(parents, int(left), int(right))
        path_edges.add(tuple(sorted((int(left), int(right)))))
        cycle_edges.update(path_edges)
        cycle_nodes_by_closure.append({node for edge in path_edges for node in edge})
        cycle_paths_by_closure.append(tree_path_indices(parents, int(left), int(right)))
    nodes_in_cycles = {node for group in cycle_nodes_by_closure for node in group}

    node_logits = terminal["nodes"][batch_index, :node_count].detach().cpu()
    atom_states = np.zeros(node_count, dtype=np.int64)
    raw_states = np.zeros(node_count, dtype=np.int64)
    assigned = np.zeros(node_count, dtype=np.bool_)
    neighbors: list[set[int]] = [set() for _ in range(node_count)]
    for child in range(1, node_count):
        parent = int(parents[child])
        neighbors[child].add(parent)
        neighbors[parent].add(child)
    for left, right in zip(condition.closure_left, condition.closure_right, strict=True):
        neighbors[int(left)].add(int(right))
        neighbors[int(right)].add(int(left))
    for node in range(node_count):
        if condition.fixed_atom_mask[node]:
            state_index = int(condition.fixed_atom_states[node])
            atom_states[node] = raw_states[node] = state_index
            assigned[node] = True

    role_nodes = {
        role: tuple(index for index, observed in enumerate(role_names) if observed == role)
        for role in sorted(set(role_names))
    }

    def atom_is_role_locally_allowed(node: int, state: AtomState) -> bool:
        if local_chemistry_support is None:
            return True
        assert program_id is not None
        role = role_names[node]
        bounds = local_chemistry_support.component_support_bounds(program_id, role)
        nodes = role_nodes[role]
        if local_chemistry_constraint_scope == "role_edges_cycles_bounds":
            assigned_symbols = [
                atom_vocabulary[int(atom_states[index])].symbol
                for index in nodes
                if assigned[index] and index != node
            ]
            remaining_after = sum(not assigned[index] and index != node for index in nodes)
            carbon = assigned_symbols.count("C") + int(state.symbol == "C")
            hetero = (
                len(assigned_symbols)
                - assigned_symbols.count("C")
                + int(state.symbol != "C")
            )
            if (
                carbon > bounds.carbon_atoms_max
                or carbon + remaining_after < bounds.carbon_atoms_min
                or hetero > bounds.heteroatoms_max
                or hetero + remaining_after < bounds.heteroatoms_min
            ):
                return False
        for neighbor in neighbors[node]:
            if not assigned[neighbor] or neighbor == node:
                continue
            neighbor_state = atom_vocabulary[int(atom_states[neighbor])]
            if not local_chemistry_support.allows_role_edge_for_any_bond(
                program_id,
                role,
                state.symbol,
                role_names[neighbor],
                neighbor_state.symbol,
            ):
                return False
        if (
            local_chemistry_constraint_scope == "role_edges_cycles_bounds"
            and local_chemistry_support.enforces_role_cycles
        ):
            for cycle in cycle_paths_by_closure:
                if node not in cycle:
                    continue
                others = tuple(index for index in cycle if index != node)
                if not all(assigned[index] for index in others):
                    continue
                signature = (
                    (role, state.symbol),
                    *(
                        (
                            role_names[index],
                            atom_vocabulary[int(atom_states[index])].symbol,
                        )
                        for index in others
                    ),
                )
                if not local_chemistry_support.allows_role_cycle(program_id, signature):
                    return False
        return True

    def atom_is_allowed(node: int, state: AtomState) -> bool:
        if state.aromatic and node not in nodes_in_cycles:
            return False
        if not atom_is_role_locally_allowed(node, state):
            return False
        if not forbid_oxygen_oxygen_bonds or state.symbol != "O":
            return True
        return not any(
            assigned[neighbor] and atom_vocabulary[int(atom_states[neighbor])].symbol == "O"
            for neighbor in neighbors[node]
        )

    for node in range(node_count):
        if condition.fixed_atom_mask[node]:
            continue
        valence_valid = torch.tensor(
            [
                _atom_valence_units(state) >= minimum_units[node]
                for state in atom_vocabulary
            ],
            dtype=torch.bool,
        )
        valid = torch.tensor(
            [
                bool(valence_valid[index]) and atom_is_allowed(node, state)
                for index, state in enumerate(atom_vocabulary)
            ],
            dtype=torch.bool,
        )
        if not bool(valid.any()):
            code = (
                "role_local_atom_state_unavailable"
                if bool(valence_valid.any()) and local_chemistry_support is not None
                else "no_valence_feasible_atom_state"
            )
            raise UgiTerminalDecodeError(
                code,
                "atom_state",
                node=node,
                origin=role_names[node] if local_chemistry_support is not None else "unconditioned",
                minimum_valence_units=int(minimum_units[node]),
            )
        raw_states[node] = _masked_terminal_choice(
            node_logits[node],
            valid,
            mode=atom_choice_mode,
            generator=generator,
            temperature=float(atom_temperatures[node]),
        )
        atom_states[node] = raw_states[node]
        assigned[node] = True

    # Aromaticity is a cycle-level state.  An isolated aromatic atom is never
    # emitted merely because its local logit is high.
    aromatic_cycle_nodes: set[int] = set()
    for group in cycle_nodes_by_closure:
        if all(atom_vocabulary[int(raw_states[node])].aromatic for node in group):
            aromatic_cycle_nodes.update(group)
    for node in range(node_count):
        if condition.fixed_atom_mask[node] or not atom_vocabulary[int(atom_states[node])].aromatic:
            continue
        if node not in aromatic_cycle_nodes:
            valid = torch.tensor(
                [
                    not state.aromatic
                    and _atom_valence_units(state) >= minimum_units[node]
                    and atom_is_allowed(node, state)
                    for state in atom_vocabulary
                ],
                dtype=torch.bool,
            )
            if not bool(valid.any()):
                raise UgiTerminalDecodeError(
                    "nonaromatic_state_unavailable",
                    "aromatic_cycle_consistency",
                    node=node,
                    origin=(
                        role_names[node]
                        if local_chemistry_support is not None
                        else "unconditioned"
                    ),
                )
            atom_states[node] = _masked_terminal_choice(
                node_logits[node],
                valid,
                mode=atom_choice_mode,
                generator=generator,
                temperature=float(atom_temperatures[node]),
            )

    capacities = np.asarray(
        [_atom_valence_units(atom_vocabulary[int(state)]) for state in atom_states],
        dtype=np.int64,
    )
    used_units = minimum_units.copy()
    parent_bonds = np.zeros(node_count, dtype=np.int64)
    parent_bonds[0] = ROOT_BOND_TARGET
    parent_logits = terminal["parent_bonds"][batch_index, :node_count].detach().cpu()
    closure_logits = terminal["closure_bonds"][batch_index, :closure_count].detach().cpu()

    def choose_bond(logits: Any, left: int, right: int, *, cycle_edge: bool) -> int:
        spare = min(capacities[left] - used_units[left], capacities[right] - used_units[right])
        valence_valid = torch.tensor(_BOND_VALENCE_UNITS <= 2 + max(0, int(spare)))
        valid = valence_valid.clone()
        left_aromatic = atom_vocabulary[int(atom_states[left])].aromatic
        right_aromatic = atom_vocabulary[int(atom_states[right])].aromatic
        if cycle_edge and left_aromatic and right_aromatic:
            valid[:] = False
            valid[3] = bool(spare >= 1)
        else:
            valid[3] = False
        valence_and_aromatic_valid = valid.clone()
        if local_chemistry_support is not None:
            assert program_id is not None
            left_symbol = atom_vocabulary[int(atom_states[left])].symbol
            right_symbol = atom_vocabulary[int(atom_states[right])].symbol
            for bond in range(len(valid)):
                if valid[bond] and not local_chemistry_support.allows_role_edge(
                    program_id,
                    role_names[left],
                    left_symbol,
                    bond,
                    role_names[right],
                    right_symbol,
                ):
                    valid[bond] = False
        if not bool(valid.any()):
            code = (
                "role_local_bond_state_unavailable"
                if bool(valence_and_aromatic_valid.any())
                and local_chemistry_support is not None
                else "no_valence_feasible_bond_state"
            )
            raise UgiTerminalDecodeError(
                code,
                "bond_state",
                left=left,
                right=right,
                left_origin=(
                    role_names[left] if local_chemistry_support is not None else "unconditioned"
                ),
                right_origin=(
                    role_names[right]
                    if local_chemistry_support is not None
                    else "unconditioned"
                ),
                cycle_edge=cycle_edge,
            )
        selected = _masked_terminal_choice(
            logits,
            valid,
            mode=bond_choice_mode,
            generator=generator,
            temperature=resolved_bond_temperature,
        )
        extra = int(_BOND_VALENCE_UNITS[selected]) - 2
        used_units[left] += extra
        used_units[right] += extra
        return selected

    for child in range(1, node_count):
        parent = int(parents[child])
        if condition.fixed_parent_bond_mask[child]:
            parent_bonds[child] = int(condition.fixed_parent_bond_states[child])
        else:
            parent_bonds[child] = choose_bond(
                parent_logits[child],
                child,
                parent,
                cycle_edge=tuple(sorted((child, parent))) in cycle_edges,
            )
    closure_bonds = np.zeros(closure_count, dtype=np.int64)
    for slot, (left, right) in enumerate(
        zip(condition.closure_left, condition.closure_right, strict=True)
    ):
        if condition.fixed_closure_bond_mask[slot]:
            closure_bonds[slot] = int(condition.fixed_closure_bond_states[slot])
        else:
            closure_bonds[slot] = choose_bond(
                closure_logits[slot],
                int(left),
                int(right),
                cycle_edge=True,
            )

    if maximum_decorations == 1:
        anchor_logits = terminal["decoration_anchor"][batch_index].detach().cpu()
        if decoration_choice_mode == "argmax" and local_chemistry_support is None:
            decoration_anchor = int(anchor_logits.argmax())
            if decoration_anchor:
                anchor = decoration_anchor - 1
                if capacities[anchor] - used_units[anchor] < 4:
                    decoration_anchor = 0
        else:
            valid_anchors = torch.ones_like(anchor_logits, dtype=torch.bool)
            for encoded_anchor in range(1, anchor_logits.numel()):
                anchor = encoded_anchor - 1
                allowed = bool(capacities[anchor] - used_units[anchor] >= 4)
                if allowed and local_chemistry_support is not None:
                    assert program_id is not None
                    allowed = local_chemistry_support.allows_role_edge(
                        program_id,
                        role_names[anchor],
                        atom_vocabulary[int(atom_states[anchor])].symbol,
                        1,
                        role_names[anchor],
                        "O",
                    )
                valid_anchors[encoded_anchor] = allowed
            decoration_anchor = _masked_terminal_choice(
                anchor_logits,
                valid_anchors,
                mode=decoration_choice_mode,
                generator=generator,
                temperature=resolved_decoration_temperature,
            )
        return UgiChemistrySample(
            atom_states=atom_states,
            parent_bond_states=parent_bonds,
            closure_bond_states=closure_bonds,
            decoration_anchor=decoration_anchor,
        )

    decoration_anchors = np.zeros(maximum_decorations, dtype=np.int64)
    decoration_atoms = np.zeros(maximum_decorations, dtype=np.int64)
    decoration_bonds = np.zeros(maximum_decorations, dtype=np.int64)
    for slot in range(maximum_decorations):
        anchor_logits = terminal["decoration_anchors"][batch_index, slot].detach().cpu()
        atom_logits = terminal["decoration_atoms"][batch_index, slot].detach().cpu()
        bond_logits = terminal["decoration_bonds"][batch_index, slot].detach().cpu()
        remaining_anchors = torch.ones_like(anchor_logits, dtype=torch.bool)
        emitted = False
        while bool(remaining_anchors.any()):
            encoded_anchor = _masked_terminal_choice(
                anchor_logits,
                remaining_anchors,
                mode=decoration_choice_mode,
                generator=generator,
                temperature=resolved_decoration_temperature,
            )
            if encoded_anchor == 0:
                break
            anchor = int(encoded_anchor) - 1
            remaining_anchors[encoded_anchor] = False
            if anchor >= node_count or condition.fixed_atom_mask[anchor]:
                continue
            feasible_pairs: list[tuple[int, int]] = []
            pair_scores: list[float] = []
            for atom_index, state in enumerate(atom_vocabulary):
                if state.aromatic:
                    continue
                if (
                    forbid_oxygen_oxygen_bonds
                    and state.symbol == "O"
                    and atom_vocabulary[int(atom_states[anchor])].symbol == "O"
                ):
                    continue
                for bond_index, units in enumerate(_BOND_VALENCE_UNITS):
                    if bond_index == 3:
                        continue
                    locally_allowed = True
                    if local_chemistry_support is not None:
                        assert program_id is not None
                        locally_allowed = local_chemistry_support.allows_role_edge(
                            program_id,
                            role_names[anchor],
                            atom_vocabulary[int(atom_states[anchor])].symbol,
                            bond_index,
                            role_names[anchor],
                            state.symbol,
                        )
                    if (
                        locally_allowed
                        and int(units) <= capacities[anchor] - used_units[anchor]
                        and int(units) <= _atom_valence_units(state)
                    ):
                        feasible_pairs.append((atom_index, bond_index))
                        pair_scores.append(float(atom_logits[atom_index] + bond_logits[bond_index]))
            if not feasible_pairs:
                continue
            scores = torch.as_tensor(pair_scores, dtype=atom_logits.dtype)
            pair_index = _masked_terminal_choice(
                scores,
                torch.ones_like(scores, dtype=torch.bool),
                mode=decoration_choice_mode,
                generator=generator,
                temperature=resolved_decoration_temperature,
            )
            atom_index, bond_index = feasible_pairs[pair_index]
            decoration_anchors[slot] = encoded_anchor
            decoration_atoms[slot] = atom_index
            decoration_bonds[slot] = bond_index
            used_units[anchor] += int(_BOND_VALENCE_UNITS[bond_index])
            emitted = True
            break
        if not emitted:
            continue
    return UgiChemistrySample(
        atom_states=atom_states,
        parent_bond_states=parent_bonds,
        closure_bond_states=closure_bonds,
        decoration_anchor=0,
        decoration_anchors=decoration_anchors,
        decoration_atom_states=decoration_atoms,
        decoration_bond_states=decoration_bonds,
    )


def sample_ugi_chemistry(
    model: Any,
    conditions: Sequence[ChemistryTopologyCondition],
    source_marginals: dict[str, np.ndarray],
    *,
    atom_classes: int,
    sample_steps: int,
    batch_size: int,
    seed: int,
    device: str,
    atom_vocabulary: tuple[AtomState, ...] | None = None,
    allow_terminal_failures: bool = False,
) -> tuple[list[UgiChemistrySample | None], dict[str, Any]]:
    """Sample chemistry on generated sparse topologies without structural repair."""

    if torch is None:
        raise UgiChemistryFlowError("Ugi chemistry sampling requires torch")
    if not conditions or sample_steps < 2 or batch_size < 1:
        raise UgiChemistryFlowError("invalid Ugi chemistry sampling request")
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device).manual_seed(seed)
    sources = {
        key: torch.as_tensor(value, dtype=torch.float32, device=resolved_device)
        for key, value in source_marginals.items()
    }
    maximum_nodes = max(condition.node_count for condition in conditions)
    maximum_closures = max(condition.closure_count for condition in conditions)
    output: list[UgiChemistrySample | None] = []
    terminal_failures = 0
    start = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(conditions), batch_size):
            local = tuple(conditions[offset : offset + batch_size])
            topology = {
                key: value.to(resolved_device)
                for key, value in collate_ugi_chemistry_conditions(
                    local,
                    maximum_nodes=maximum_nodes,
                    maximum_closures=maximum_closures,
                ).items()
            }
            topology["maximum_decorations"] = model.maximum_decorations
            state = _initial_chemistry_state(topology, sources, generator)
            for step in range(sample_steps):
                t_value = step / sample_steps
                t = torch.full(
                    (len(local),),
                    t_value,
                    dtype=torch.float32,
                    device=resolved_device,
                )
                predictions = model(
                    nodes=state["nodes"],
                    parent_bonds=state["parent_bonds"],
                    closure_bonds=state["closure_bonds"],
                    t=t,
                    topology=topology,
                )
                state = rstar_ugi_chemistry_step(
                    state,
                    predictions,
                    topology,
                    sources,
                    t=t_value,
                    dt=1.0 / sample_steps,
                    generator=generator,
                )
            terminal = model(
                nodes=state["nodes"],
                parent_bonds=state["parent_bonds"],
                closure_bonds=state["closure_bonds"],
                t=torch.ones(len(local), dtype=torch.float32, device=resolved_device),
                topology=topology,
            )
            if atom_vocabulary is not None:
                for batch_index, condition in enumerate(local):
                    try:
                        decoded = valence_constrained_terminal_sample(
                            condition,
                            terminal,
                            batch_index,
                            atom_vocabulary,
                            model.maximum_decorations,
                        )
                    except UgiChemistryFlowError:
                        if not allow_terminal_failures:
                            raise
                        decoded = None
                        terminal_failures += 1
                    output.append(decoded)
                continue
            node_argmax = terminal["nodes"].argmax(dim=-1)
            parent_argmax = terminal["parent_bonds"].argmax(dim=-1)
            closure_argmax = terminal["closure_bonds"].argmax(dim=-1)
            state["nodes"][topology["atom_variable_mask"]] = node_argmax[
                topology["atom_variable_mask"]
            ]
            state["parent_bonds"][topology["parent_bond_variable_mask"]] = parent_argmax[
                topology["parent_bond_variable_mask"]
            ]
            state["closure_bonds"][topology["closure_bond_variable_mask"]] = closure_argmax[
                topology["closure_bond_variable_mask"]
            ]
            if model.maximum_decorations == 1:
                state["decoration_anchor"] = terminal["decoration_anchor"].argmax(dim=-1)
            else:
                state["decoration_anchors"] = terminal["decoration_anchors"].argmax(dim=-1)
                state["decoration_atoms"] = terminal["decoration_atoms"].argmax(dim=-1)
                state["decoration_bonds"] = terminal["decoration_bonds"].argmax(dim=-1)
            for batch_index, condition in enumerate(local):
                output.append(
                    UgiChemistrySample(
                        atom_states=state["nodes"][batch_index, : condition.node_count]
                        .cpu()
                        .numpy()
                        .astype(np.int64),
                        parent_bond_states=state["parent_bonds"][
                            batch_index, : condition.node_count
                        ]
                        .cpu()
                        .numpy()
                        .astype(np.int64),
                        closure_bond_states=state["closure_bonds"][
                            batch_index, : condition.closure_count
                        ]
                        .cpu()
                        .numpy()
                        .astype(np.int64),
                        decoration_anchor=(
                            int(state["decoration_anchor"][batch_index])
                            if model.maximum_decorations == 1
                            else 0
                        ),
                        decoration_anchors=(
                            None
                            if model.maximum_decorations == 1
                            else state["decoration_anchors"][batch_index]
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        ),
                        decoration_atom_states=(
                            None
                            if model.maximum_decorations == 1
                            else state["decoration_atoms"][batch_index]
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        ),
                        decoration_bond_states=(
                            None
                            if model.maximum_decorations == 1
                            else state["decoration_bonds"][batch_index]
                            .cpu()
                            .numpy()
                            .astype(np.int64)
                        ),
                    )
                )
    return output, {
        "samples": len(output),
        "sample_steps": sample_steps,
        "seconds": time.perf_counter() - start,
        "terminal_decoder": (
            "valence_constrained_categorical_argmax_without_structural_repair"
            if atom_vocabulary is not None
            else "categorical_argmax_without_structural_repair"
        ),
        "valence_constrained_terminal_support": atom_vocabulary is not None,
        "terminal_support_failures": terminal_failures,
        "atom_classes": atom_classes,
    }


def chemistry_sample_to_molecule(
    condition: ChemistryTopologyCondition,
    sample: UgiChemistrySample,
    atom_vocabulary: tuple[AtomState, ...],
) -> Chem.Mol:
    """Construct one complete molecule; sanitization failure is not repaired."""

    expanded_decorations = sample.decoration_anchors is not None
    if (
        sample.atom_states.shape != (condition.node_count,)
        or sample.parent_bond_states.shape != (condition.node_count,)
        or sample.closure_bond_states.shape != (condition.closure_count,)
        or not 0 <= sample.decoration_anchor <= condition.node_count
    ):
        raise UgiChemistryFlowError("generated chemistry does not match its topology")
    if expanded_decorations and (
        sample.decoration_atom_states is None
        or sample.decoration_bond_states is None
        or sample.decoration_anchors.ndim != 1
        or sample.decoration_atom_states.shape != sample.decoration_anchors.shape
        or sample.decoration_bond_states.shape != sample.decoration_anchors.shape
        or np.any(sample.decoration_anchors < 0)
        or np.any(sample.decoration_anchors > condition.node_count)
        or np.any(sample.decoration_atom_states < 0)
        or np.any(sample.decoration_atom_states >= len(atom_vocabulary))
        or np.any(sample.decoration_bond_states < 0)
        or np.any(sample.decoration_bond_states >= len(_INDEX_TO_BOND_TYPE))
    ):
        raise UgiChemistryFlowError("generated sparse decorations are invalid")
    parent_bonds = sample.parent_bond_states.copy()
    parent_bonds[0] = 0
    record = V5SparseGraphRecord(
        structure_id=condition.structure_id,
        canonical_smiles="",
        node_states=sample.atom_states.copy(),
        offspring=condition.offspring.copy(),
        parent_bonds=parent_bonds,
        closure_left=condition.closure_left.copy(),
        closure_right=condition.closure_right.copy(),
        closure_bonds=sample.closure_bond_states.copy(),
        tree_traversal=condition.tree_traversal,
    )
    molecule = v5_graph_to_molecule(record, atom_vocabulary)
    if expanded_decorations:
        editable = Chem.RWMol(molecule)
        for anchor, atom_index, bond_index in zip(
            sample.decoration_anchors,
            sample.decoration_atom_states,
            sample.decoration_bond_states,
            strict=True,
        ):
            if int(anchor) == 0:
                continue
            state = atom_vocabulary[int(atom_index)]
            atom = Chem.Atom(state.symbol)
            atom.SetFormalCharge(state.formal_charge)
            atom.SetIsAromatic(state.aromatic)
            if state.explicit_hydrogens:
                atom.SetNumExplicitHs(state.explicit_hydrogens)
                atom.SetNoImplicit(True)
            decoration_index = editable.AddAtom(atom)
            editable.AddBond(
                int(anchor) - 1,
                decoration_index,
                _INDEX_TO_BOND_TYPE[int(bond_index)],
            )
        molecule = editable.GetMol()
        Chem.SanitizeMol(molecule)
    elif sample.decoration_anchor:
        editable = Chem.RWMol(molecule)
        oxygen_index = editable.AddAtom(Chem.Atom("O"))
        editable.AddBond(sample.decoration_anchor - 1, oxygen_index, Chem.BondType.DOUBLE)
        molecule = editable.GetMol()
        Chem.SanitizeMol(molecule)
    return molecule


def chemistry_sample_statistics(
    conditions: Sequence[ChemistryTopologyCondition],
    samples: Sequence[UgiChemistrySample | None],
    atom_vocabulary: tuple[AtomState, ...],
) -> dict[str, Any]:
    """Report molecule validity without silently repairing invalid samples."""

    if len(conditions) != len(samples):
        raise UgiChemistryFlowError("chemistry samples and topologies are misaligned")
    valid_smiles = []
    errors: dict[str, int] = {}
    aromatic = 0
    decorated = 0
    for condition, sample in zip(conditions, samples, strict=True):
        if sample is None:
            errors["TerminalSupportFailure"] = errors.get("TerminalSupportFailure", 0) + 1
            continue
        decorated += int(
            np.count_nonzero(sample.decoration_anchors)
            if sample.decoration_anchors is not None
            else sample.decoration_anchor > 0
        )
        try:
            molecule = chemistry_sample_to_molecule(condition, sample, atom_vocabulary)
            valid_smiles.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))
            aromatic += int(any(atom.GetIsAromatic() for atom in molecule.GetAtoms()))
        except (ValueError, RuntimeError, UgiChemistryFlowError) as exc:
            label = type(exc).__name__
            errors[label] = errors.get(label, 0) + 1
    count = len(samples)
    return {
        "samples": count,
        "valid_molecules": len(valid_smiles),
        "valid_fraction": len(valid_smiles) / count,
        "unique_valid_molecules": len(set(valid_smiles)),
        "mean_decorations": decorated / count,
        "decoration_fraction": sum(
            int(
                np.count_nonzero(sample.decoration_anchors) > 0
                if sample is not None and sample.decoration_anchors is not None
                else sample is not None and sample.decoration_anchor > 0
            )
            for sample in samples
        )
        / count,
        "aromatic_valid_fraction": aromatic / max(len(valid_smiles), 1),
        "failure_types": dict(sorted(errors.items())),
        "valid_smiles": valid_smiles,
    }
