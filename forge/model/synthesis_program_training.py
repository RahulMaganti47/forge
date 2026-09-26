"""Reusable training primitives for the shared synthesis-program sparse flow."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from forge.model.reaction_program_conditioning import ReactionProgramVocabulary
from forge.model.reaction_program_flow import (
    ReactionProgramSparseFlow,
    collate_synthesis_program_records,
    decode_synthesis_program_argmax,
    noise_synthesis_program_batch,
    synthesis_program_flow_loss,
)
from forge.model.synthesis_program_graph import SynthesisProgramGraphRecord

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]


class SynthesisProgramTrainingPrimitiveError(ValueError):
    """A shared training batch violates its conditioning or fixed-state contract."""


def build_synthesis_program_flow(
    *,
    vocabulary: ReactionProgramVocabulary,
    node_classes: int,
    model_config: Mapping[str, Any],
    device: Any,
) -> Any:
    """Construct the one shared architecture from an explicit model contract."""

    architecture = str(model_config.get("architecture", "sparse_mpnn"))
    if architecture == "reaction_program_graph_transformer":
        from forge.model.reaction_program_transformer import ReactionProgramGraphTransformer

        return ReactionProgramGraphTransformer(
            vocabulary=vocabulary,
            node_classes=node_classes,
            hidden_dim=int(model_config["hidden_dim"]),
            layers=int(model_config["layers"]),
            heads=int(model_config["attention_heads"]),
            expert_count=int(model_config["expert_count"]),
            adapter_dim=int(model_config["adapter_dim"]),
            maximum_closures=int(model_config["maximum_closures"]),
            maximum_heavy_atoms=int(model_config["maximum_heavy_atoms"]),
            dropout=float(model_config["dropout"]),
            bond_classes=int(model_config["bond_classes"]),
            layerwise_program_cross_attention=bool(
                model_config.get("layerwise_program_cross_attention", True)
            ),
            routed_adapters=bool(model_config.get("routed_adapters", True)),
            role_isolated_attention=bool(model_config.get("role_isolated_attention", False)),
            role_specific_parameters=bool(model_config.get("role_specific_parameters", False)),
            repeat_group_conditioning=bool(model_config.get("repeat_group_conditioning", False)),
            role_morphology_conditioning=bool(
                model_config.get("role_morphology_conditioning", False)
            ),
            specialist_adapter_dim=int(model_config.get("specialist_adapter_dim", 0)),
            maximum_children=int(model_config.get("maximum_children", 0)),
            program_routed_output_heads=bool(
                model_config.get("program_routed_output_heads", False)
            ),
        ).to(device)
    if architecture not in {"sparse_mpnn", "reaction_program_sparse_whole_lipid_flow"}:
        raise SynthesisProgramTrainingPrimitiveError(
            f"unsupported synthesis-program architecture: {architecture!r}"
        )
    return ReactionProgramSparseFlow(
        vocabulary=vocabulary,
        node_classes=node_classes,
        hidden_dim=int(model_config["hidden_dim"]),
        layers=int(model_config["layers"]),
        maximum_closures=int(model_config["maximum_closures"]),
        maximum_heavy_atoms=int(model_config["maximum_heavy_atoms"]),
        dropout=float(model_config["dropout"]),
        bond_classes=int(model_config["bond_classes"]),
    ).to(device)


def move_tensors(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    """Move tensor values while leaving any future metadata values unchanged."""

    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()
    }


def collate_synthesis_program_training_batch(
    records: Sequence[SynthesisProgramGraphRecord],
    *,
    maximum_closures: int,
    conditioning: str,
    vocabulary: ReactionProgramVocabulary,
    program_id_mapping: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Collate one batch and apply an explicit matched semantic-control transform."""

    if conditioning == "program":
        batch = collate_synthesis_program_records(
            records, maximum_closures=maximum_closures, conditioning_mode="program"
        )
        batch["source_program_states"] = batch["program_states"].clone()
        return batch
    if conditioning == "null_all_program_coordinates":
        if program_id_mapping is not None:
            raise SynthesisProgramTrainingPrimitiveError(
                "null conditioning cannot also map program identifiers"
            )
        batch = collate_synthesis_program_records(
            records, maximum_closures=maximum_closures, conditioning_mode="program"
        )
        batch["source_program_states"] = batch["program_states"].clone()
        batch["program_states"].zero_()
        batch["role_states"].zero_()
        batch["core_position_states"].zero_()
        batch["program_depths"].zero_()
        batch["component_instance_states"].zero_()
        batch["component_position_states"].zero_()
        batch["repeat_group_states"].zero_()
        batch["role_morphology_states"].zero_()
        return batch
    if conditioning != "cyclic_program_id_only_roles_core_and_depth_retained":
        raise SynthesisProgramTrainingPrimitiveError(
            f"unsupported synthesis-program conditioning: {conditioning!r}"
        )
    if program_id_mapping is None or set(program_id_mapping) != set(vocabulary.program_states[1:]):
        raise SynthesisProgramTrainingPrimitiveError(
            "cyclic control requires one mapped identifier per program"
        )
    if set(program_id_mapping.values()) != set(vocabulary.program_states[1:]):
        raise SynthesisProgramTrainingPrimitiveError(
            "cyclic control mapping must be a permutation of program identifiers"
        )
    batch = collate_synthesis_program_records(
        records, maximum_closures=maximum_closures, conditioning_mode="program"
    )
    batch["source_program_states"] = batch["program_states"].clone()
    mapped = batch["program_states"].clone()
    for index, record in enumerate(records):
        mapped[index] = vocabulary.program_to_index[program_id_mapping[record.program_id]]
    batch["program_states"] = mapped
    return batch


def synthesis_program_forward(
    model: Any,
    clean: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    t: Any,
    generator: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Noise learnable states and run the shared sparse flow once."""

    noisy = noise_synthesis_program_batch(clean, node_marginal, bond_marginal, t, generator)
    predictions = _synthesis_program_predict(model, clean, noisy, t)
    return predictions, noisy


def _synthesis_program_predict(
    model: Any, clean: Mapping[str, Any], state: Mapping[str, Any], t: Any
) -> dict[str, Any]:
    return model(
        nodes=state["nodes"],
        parents=state["parents"],
        parent_bonds=state["parent_bonds"],
        closure_left=state["closure_left"],
        closure_right=state["closure_right"],
        closure_bonds=state["closure_bonds"],
        t=t,
        node_mask=clean["node_mask"],
        child_mask=clean["child_mask"],
        closure_mask=clean["closure_mask"],
        program_states=clean["program_states"],
        role_states=clean["role_states"],
        core_position_states=clean["core_position_states"],
        program_depths=clean["program_depths"],
        adapter_mask=clean["adapter_mask"],
        repeat_group_states=clean["repeat_group_states"],
        component_position_states=clean["component_position_states"],
        component_instance_states=clean["component_instance_states"],
        role_morphology_states=clean["role_morphology_states"],
    )


def synthesis_program_topology_conditioned_forward(
    model: Any,
    clean: Mapping[str, Any],
    noisy: Mapping[str, Any],
    t: Any,
) -> dict[str, Any]:
    """Predict chemistry from the same corruption after replacing topology with its target."""

    conditioned = dict(noisy)
    for field in ("parents", "closure_left", "closure_right"):
        conditioned[field] = clean[field]
    return _synthesis_program_predict(model, clean, conditioned, t)


def synthesis_program_paired_topology_forward(
    model: Any,
    clean: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    t: Any,
    generator: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run noisy- and target-topology predictions in one accelerator-sized model call.

    The two halves contain the same examples and corruption.  Only the topology fields differ.
    This preserves the topology-conditioned objective while replacing two small Transformer calls
    with one larger call that uses accelerator matrix units more effectively.
    """

    if torch is None:
        raise SynthesisProgramTrainingPrimitiveError("paired topology forward requires torch")
    noisy = noise_synthesis_program_batch(clean, node_marginal, bond_marginal, t, generator)
    batch_size = int(t.shape[0])
    paired_clean = {
        key: (
            torch.cat((value, value), dim=0)
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == batch_size
            else value
        )
        for key, value in clean.items()
    }
    conditioned = dict(noisy)
    for field in ("parents", "closure_left", "closure_right"):
        conditioned[field] = clean[field]
    paired_state = {
        key: torch.cat((value, conditioned[key]), dim=0) for key, value in noisy.items()
    }
    paired_predictions = _synthesis_program_predict(
        model,
        paired_clean,
        paired_state,
        torch.cat((t, t), dim=0),
    )
    predictions: dict[str, Any] = {}
    topology_predictions: dict[str, Any] = {}
    for key, value in paired_predictions.items():
        if not torch.is_tensor(value) or value.ndim == 0 or int(value.shape[0]) != 2 * batch_size:
            raise SynthesisProgramTrainingPrimitiveError(
                f"paired topology output is not batch-aligned: {key}"
            )
        predictions[key] = value[:batch_size]
        topology_predictions[key] = value[batch_size:]
    return predictions, noisy, topology_predictions


def synthesis_program_forward_loss(
    model: Any,
    clean: Mapping[str, Any],
    node_marginal: Any,
    bond_marginal: Any,
    t: Any,
    generator: Any,
) -> tuple[Any, dict[str, float], dict[str, Any]]:
    """Noise learnable states, run the model and compute the masked objective."""

    predictions, noisy = synthesis_program_forward(
        model, clean, node_marginal, bond_marginal, t, generator
    )
    loss, metrics = synthesis_program_flow_loss(predictions, clean)
    return loss, metrics, noisy


def synthesis_program_reconstruction_metrics(
    predictions: Mapping[str, Any], clean: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure exact clean-tensor reconstruction after one fixed noising draw."""

    terminal = decode_synthesis_program_argmax(predictions, clean)
    fields = {
        "nodes": "node_mask",
        "parents": "child_mask",
        "parent_bonds": "child_mask",
        "closure_left": "closure_mask",
        "closure_right": "closure_mask",
        "closure_bonds": "closure_mask",
    }
    exact = terminal["nodes"].new_ones(terminal["nodes"].shape[0], dtype=bool)
    field_correct: dict[str, int] = {}
    field_total: dict[str, int] = {}
    for field, mask_name in fields.items():
        mask = clean[mask_name]
        correct = terminal[field] == clean[field]
        field_correct[field] = int(correct[mask].sum())
        field_total[field] = int(mask.sum())
        exact &= (correct | ~mask).all(dim=1)
    return {
        "records": int(exact.numel()),
        "exact_tensor_records": int(exact.sum()),
        "exact_tensor_fraction": float(exact.float().mean()),
        "field_correct": field_correct,
        "field_total": field_total,
        "fixed_states_exact": synthesis_program_fixed_state_exact(terminal, clean),
    }


def synthesis_program_fixed_state_exact(state: Mapping[str, Any], clean: Mapping[str, Any]) -> bool:
    """Check every adapter-fixed state after noising or decoding."""

    return bool(synthesis_program_fixed_state_exact_tensor(state, clean))


def synthesis_program_fixed_state_exact_tensor(
    state: Mapping[str, Any], clean: Mapping[str, Any]
) -> Any:
    """Return the fixed-state invariant as a scalar tensor without host synchronization."""

    checks = [
        (state[field][clean[mask]] == clean[field][clean[mask]]).all()
        for field, mask in {
            "nodes": "fixed_atom_mask",
            "parents": "fixed_parent_mask",
            "parent_bonds": "fixed_parent_bond_mask",
            "closure_left": "fixed_closure_endpoint_mask",
            "closure_right": "fixed_closure_endpoint_mask",
            "closure_bonds": "fixed_closure_bond_mask",
        }.items()
    ]
    exact = checks[0]
    for check in checks[1:]:
        exact = exact & check
    return exact


__all__ = [
    "SynthesisProgramTrainingPrimitiveError",
    "build_synthesis_program_flow",
    "collate_synthesis_program_training_batch",
    "move_tensors",
    "synthesis_program_fixed_state_exact",
    "synthesis_program_fixed_state_exact_tensor",
    "synthesis_program_forward",
    "synthesis_program_paired_topology_forward",
    "synthesis_program_forward_loss",
    "synthesis_program_reconstruction_metrics",
    "synthesis_program_topology_conditioned_forward",
]
