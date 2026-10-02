from __future__ import annotations

import math

import pytest

from forge.model.conditioning.reaction_program import ReactionProgramVocabulary
from forge.model.networks.transformer import ReactionProgramGraphTransformer
from forge.model.objectives.pcgrad import balanced_pcgrad_backward
from forge.model.objectives.transformer import (
    _repeat_component_consistency,
    per_program_transformer_losses,
    reaction_program_transformer_loss,
    synthesis_program_offspring_targets,
)
from forge.model.training import (
    synthesis_program_forward,
    synthesis_program_paired_topology_forward,
    synthesis_program_topology_conditioned_forward,
)

torch = pytest.importorskip("torch")


def _dense_repeat_consistency(
    predictions: dict[str, torch.Tensor], clean: dict[str, torch.Tensor]
) -> torch.Tensor:
    groups = clean["repeat_group_states"]
    positions = clean["component_position_states"]
    instances = clean["component_instance_states"]
    active = (groups > 0) & clean["node_mask"] & (clean["core_position_states"] == 1)
    pairs = (
        active[:, :, None]
        & active[:, None, :]
        & (groups[:, :, None] == groups[:, None, :])
        & (positions[:, :, None] == positions[:, None, :])
        & (instances[:, :, None] != instances[:, None, :])
    )
    pairs &= torch.triu(torch.ones_like(pairs[0]), diagonal=1)[None]
    node_probabilities = torch.softmax(predictions["nodes"], dim=-1)
    node_distances = (
        (node_probabilities[:, :, None] - node_probabilities[:, None]).square().mean(dim=-1)
    )
    node_loss = (node_distances * pairs).sum() / pairs.sum().clamp(min=1)
    bond_pairs = pairs & clean["child_mask"][:, :, None] & clean["child_mask"][:, None, :]
    bond_probabilities = torch.softmax(predictions["parent_bonds"], dim=-1)
    bond_distances = (
        (bond_probabilities[:, :, None] - bond_probabilities[:, None]).square().mean(dim=-1)
    )
    return node_loss + (bond_distances * bond_pairs).sum() / bond_pairs.sum().clamp(min=1)


def _batch() -> dict[str, torch.Tensor]:
    batch, nodes, closures = 3, 6, 1
    node_mask = torch.ones((batch, nodes), dtype=torch.bool)
    child_mask = node_mask.clone()
    child_mask[:, 0] = False
    closure_mask = torch.zeros((batch, closures), dtype=torch.bool)
    program_states = torch.tensor([1, 2, 3])
    return {
        "nodes": torch.tensor([[1, 2, 3, 1, 2, 3]] * batch),
        "parents": torch.tensor([[0, 0, 1, 1, 2, 3]] * batch),
        "parent_bonds": torch.tensor([[0, 1, 1, 2, 1, 1]] * batch),
        "closure_left": torch.zeros((batch, closures), dtype=torch.long),
        "closure_right": torch.zeros((batch, closures), dtype=torch.long),
        "closure_bonds": torch.zeros((batch, closures), dtype=torch.long),
        "t": torch.tensor([0.2, 0.5, 0.8]),
        "node_mask": node_mask,
        "child_mask": child_mask,
        "closure_mask": closure_mask,
        "program_states": program_states,
        "source_program_states": program_states.clone(),
        "role_states": torch.tensor([[1, 1, 2, 2, 3, 3], [1, 2, 2, 3, 3, 1], [2, 2, 1, 1, 3, 3]]),
        "core_position_states": torch.tensor(
            [[2, 1, 1, 1, 1, 1], [3, 1, 1, 1, 1, 1], [4, 1, 1, 1, 1, 1]]
        ),
        "program_depths": torch.tensor([1, 2, 3]),
        "component_instance_states": torch.tensor([[1, 1, 2, 2, 3, 3]] * batch),
        "component_position_states": torch.tensor([[1, 2, 1, 2, 1, 2]] * batch),
        "repeat_group_states": torch.tensor([[2, 2, 2, 2, 0, 0]] * batch),
        "role_morphology_states": torch.tensor(
            [[[3, 1, 1, 2], [3, 1, 1, 2], [3, 2, 1, 2], [3, 2, 1, 2], [3, 1, 2, 2], [3, 1, 2, 2]]]
            * batch
        ),
        "adapter_mask": node_mask.clone(),
        "atom_variable_mask": node_mask.clone(),
        "parent_variable_mask": child_mask.clone(),
        "parent_bond_variable_mask": child_mask.clone(),
        "closure_endpoint_variable_mask": closure_mask.clone(),
        "closure_bond_variable_mask": closure_mask.clone(),
    }


def _model(**overrides: object) -> ReactionProgramGraphTransformer:
    vocabulary = ReactionProgramVocabulary(
        program_states=("unconditioned", "ugi", "bl", "lx"),
        role_states=("unassigned", "head", "tail_a", "tail_b"),
        core_position_states=("unconditioned", "exterior", "ugi:c1", "bl:c1", "lx:c1"),
        maximum_steps=4,
    )
    arguments = {
        "vocabulary": vocabulary,
        "node_classes": 5,
        "hidden_dim": 32,
        "layers": 3,
        "heads": 4,
        "expert_count": 3,
        "adapter_dim": 8,
        "maximum_closures": 2,
        "maximum_heavy_atoms": 16,
        "dropout": 0.0,
        "bond_classes": 4,
        **overrides,
    }
    return ReactionProgramGraphTransformer(**arguments)


def test_transformer_cross_attends_in_every_layer_and_balances_family_gradients() -> None:
    torch.manual_seed(17)
    model = _model()
    batch = _batch()
    calls = [0, 0, 0]
    hooks = [
        block.cross_attention.register_forward_hook(
            lambda _module, _inputs, _output, index=index: calls.__setitem__(
                index, calls[index] + 1
            )
        )
        for index, block in enumerate(model.blocks)
    ]
    predictions = model(
        **{
            key: value
            for key, value in batch.items()
            if key
            not in {
                "source_program_states",
                "atom_variable_mask",
                "parent_variable_mask",
                "parent_bond_variable_mask",
                "closure_endpoint_variable_mask",
                "closure_bond_variable_mask",
            }
        }
    )
    for hook in hooks:
        hook.remove()

    assert calls == [1, 1, 1]
    assert predictions["nodes"].shape == (3, 6, 5)
    assert predictions["role_states"].shape == (3, 6, 4)
    assert predictions["core_position_states"].shape == (3, 6, 5)
    assert predictions["expert_weights"].shape == (3, 3, 3)
    assert torch.allclose(predictions["expert_weights"].sum(dim=-1), torch.ones((3, 3)))

    family_losses, metrics = per_program_transformer_losses(
        predictions, batch, role_weight=0.25, core_weight=0.25
    )
    diagnostic = balanced_pcgrad_backward(family_losses, model)
    assert sorted(family_losses) == [1, 2, 3]
    assert all(torch.isfinite(loss) for loss in family_losses.values())
    assert all(value > 0.0 for value in diagnostic["raw_gradient_norms"])
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert "program_1_role_consistency_ce" in metrics


def test_program_routed_terminal_and_closure_heads_are_trained_inside_shared_model() -> None:
    torch.manual_seed(18)
    model = _model(program_routed_output_heads=True).eval()
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    with torch.no_grad():
        output = model(**model_inputs)
    assert output["terminal_chemistry_expert_weights"].shape == (3, 3)
    assert output["closure_output_expert_weights"].shape == (3, 3)
    assert torch.allclose(output["terminal_chemistry_expert_weights"].sum(dim=-1), torch.ones(3))


def test_topology_conditioned_second_pass_directly_supervises_chemistry() -> None:
    torch.manual_seed(181)
    model = _model().eval()
    clean = _batch()
    model_inputs = {
        key: value
        for key, value in clean.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    with torch.no_grad():
        predictions = model(**model_inputs)
    exact = {key: value.clone() for key, value in predictions.items()}
    for field in ("nodes", "parent_bonds", "closure_bonds"):
        exact[field].fill_(-20.0)
        exact[field].scatter_(-1, clean[field].unsqueeze(-1), 20.0)
    _, metrics = reaction_program_transformer_loss(
        predictions,
        clean,
        role_weight=0.0,
        core_weight=0.0,
        chemistry_loss_balancing="equal_present_role_mass",
        topology_conditioned_predictions=exact,
        topology_conditioned_chemistry_weight=1.0,
    )
    assert metrics["topology_conditioned_chemistry_ce"] == pytest.approx(0.0, abs=1e-6)


def test_transformer_predicts_and_learns_exact_exterior_offspring_counts() -> None:
    torch.manual_seed(19)
    model = _model(maximum_children=3).eval()
    clean = _batch()
    clean["core_position_states"] = torch.ones_like(clean["core_position_states"])
    clean["component_instance_states"] = torch.ones_like(clean["component_instance_states"])
    clean["role_states"] = torch.ones_like(clean["role_states"])
    clean["role_morphology_states"] = torch.tensor(
        [[[7, 2, 1, 2]] * clean["nodes"].shape[1]] * clean["nodes"].shape[0]
    )
    model_inputs = {
        key: value
        for key, value in clean.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    with torch.no_grad():
        predictions = model(**model_inputs)
    targets, mask = synthesis_program_offspring_targets(clean, maximum_children=3)

    assert predictions["offspring"].shape == (3, 6, 4)
    assert torch.equal(targets[0], torch.tensor([1, 2, 1, 1, 0, 0]))
    assert torch.all(mask)

    exact_logits = torch.full_like(predictions["offspring"], -20.0)
    exact_logits.scatter_(2, targets[:, :, None], 20.0)
    exact = {**predictions, "offspring": exact_logits}
    uniform = {**predictions, "offspring": torch.zeros_like(exact_logits)}
    _, exact_metrics = reaction_program_transformer_loss(
        exact,
        clean,
        role_weight=0.0,
        core_weight=0.0,
        offspring_weight=1.0,
        junction_consistency_weight=1.0,
    )
    _, uniform_metrics = reaction_program_transformer_loss(
        uniform,
        clean,
        role_weight=0.0,
        core_weight=0.0,
        offspring_weight=1.0,
        junction_consistency_weight=1.0,
    )
    assert exact_metrics["offspring_ce"] == pytest.approx(0.0, abs=1e-6)
    assert exact_metrics["junction_budget_consistency"] == pytest.approx(0.0, abs=1e-6)
    assert uniform_metrics["offspring_ce"] > exact_metrics["offspring_ce"]
    assert (
        uniform_metrics["junction_budget_consistency"]
        > exact_metrics["junction_budget_consistency"]
    )


def test_transformer_is_deterministic_and_program_conditioning_changes_output() -> None:
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    torch.manual_seed(23)
    first = _model().eval()
    torch.manual_seed(23)
    second = _model().eval()
    with torch.no_grad():
        first_output = first(**model_inputs)["nodes"]
        second_output = second(**model_inputs)["nodes"]
        changed = dict(model_inputs)
        changed["program_states"] = torch.tensor([2, 3, 1])
        changed_output = first(**changed)["nodes"]
    assert torch.equal(first_output, second_output)
    assert not torch.equal(first_output, changed_output)


def test_scaled_dot_product_attention_matches_materialized_reference() -> None:
    torch.manual_seed(29)
    attention = _model(layers=1).blocks[0].self_attention.eval()
    query = torch.randn((2, 6, 32))
    memory = torch.randn((2, 7, 32))
    query_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    memory_mask = torch.tensor([[True] * 7, [True] * 5 + [False] * 2])
    bias = torch.randn((2, 4, 6, 7)) * 0.1
    batch, queries, hidden = query.shape
    keys = memory.shape[1]
    q = attention.query(query).reshape(batch, queries, attention.heads, attention.head_dim)
    k = attention.key(memory).reshape(batch, keys, attention.heads, attention.head_dim)
    v = attention.value(memory).reshape(batch, keys, attention.heads, attention.head_dim)
    q, k, v = (value.transpose(1, 2) for value in (q, k, v))
    scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / math.sqrt(attention.head_dim)
    scores = (scores + bias).masked_fill(~memory_mask[:, None, None], -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    reference = torch.einsum("bhqk,bhkd->bhqd", probabilities, v)
    reference = attention.output(reference.transpose(1, 2).reshape(batch, queries, hidden))
    reference *= query_mask[:, :, None]

    observed = attention(
        query,
        memory,
        query_mask=query_mask,
        memory_mask=memory_mask,
        attention_bias=bias,
    )

    assert torch.allclose(observed, reference, rtol=1e-5, atol=1e-6)


def test_program_memory_preserves_node_alignment() -> None:
    torch.manual_seed(31)
    model = _model().eval()
    batch = _batch()
    with torch.no_grad():
        tokens, mask, _ = model.program_encoder(
            program_states=batch["program_states"],
            role_states=torch.ones_like(batch["role_states"]),
            core_position_states=torch.ones_like(batch["core_position_states"]),
            program_depths=batch["program_depths"],
            adapter_mask=batch["adapter_mask"],
        )
    assert torch.all(mask)
    # Program and depth occupy slots 0/1. Identical role/core states at node slots still differ
    # because the memory retains the structural slot to which each semantic coordinate applies.
    assert not torch.equal(tokens[:, 2], tokens[:, 3])


def test_repeat_group_conditioning_uses_position_but_not_component_identity() -> None:
    torch.manual_seed(33)
    model = _model(repeat_group_conditioning=True).eval()
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    changed_instances = dict(model_inputs)
    changed_instances["component_instance_states"] = torch.flip(
        model_inputs["component_instance_states"], dims=(1,)
    )
    removed_repeats = dict(model_inputs)
    removed_repeats["repeat_group_states"] = torch.zeros_like(model_inputs["repeat_group_states"])
    with torch.no_grad():
        reference = model(**model_inputs)["nodes"]
        instance_changed = model(**changed_instances)["nodes"]
        repeat_removed = model(**removed_repeats)["nodes"]
    assert torch.equal(reference, instance_changed)
    assert not torch.equal(reference, repeat_removed)


def test_role_local_morphology_conditioning_changes_output_without_component_identity() -> None:
    torch.manual_seed(34)
    model = _model(role_morphology_conditioning=True).eval()
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    changed = dict(model_inputs)
    changed["role_morphology_states"] = model_inputs["role_morphology_states"].clone()
    changed["role_morphology_states"][:, :2, 1] += 1
    with torch.no_grad():
        reference = model(**model_inputs)["nodes"]
        intervention = model(**changed)["nodes"]
    assert not torch.equal(reference, intervention)


def test_role_local_morphology_conditioning_requires_explicit_states() -> None:
    model = _model(role_morphology_conditioning=True).eval()
    batch = _batch()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
            "role_morphology_states",
        }
    }
    with pytest.raises(Exception, match="role-local morphology token shapes"):
        model(**model_inputs)


def test_repeat_consistency_loss_aligns_matched_exterior_positions() -> None:
    torch.manual_seed(35)
    batch = _batch()
    model = _model().eval()
    model_inputs = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    with torch.no_grad():
        predictions = model(**model_inputs)
    aligned = {key: value.clone() for key, value in predictions.items()}
    aligned["nodes"][:, 2] = aligned["nodes"][:, 0]
    aligned["nodes"][:, 3] = aligned["nodes"][:, 1]
    aligned["parent_bonds"][:, 2] = aligned["parent_bonds"][:, 0]
    aligned["parent_bonds"][:, 3] = aligned["parent_bonds"][:, 1]
    _, aligned_metrics = reaction_program_transformer_loss(
        aligned,
        batch,
        role_weight=0.0,
        core_weight=0.0,
        repeat_consistency_weight=1.0,
    )
    misaligned = {key: value.clone() for key, value in aligned.items()}
    misaligned["nodes"][:, 3].zero_()
    misaligned["nodes"][:, 3, 0] = 20.0
    _, misaligned_metrics = reaction_program_transformer_loss(
        misaligned,
        batch,
        role_weight=0.0,
        core_weight=0.0,
        repeat_consistency_weight=1.0,
    )
    assert aligned_metrics["repeat_consistency_mse"] == pytest.approx(0.0)
    assert aligned_metrics["repeat_consistency_pairs"] > 0
    assert misaligned_metrics["repeat_consistency_mse"] > 0.0


def test_sparse_repeat_consistency_matches_dense_value_and_gradients() -> None:
    torch.manual_seed(36)
    batch = _batch()
    sparse_predictions = {
        "nodes": torch.randn((3, 6, 5), requires_grad=True),
        "parent_bonds": torch.randn((3, 6, 4), requires_grad=True),
    }
    dense_predictions = {
        key: value.detach().clone().requires_grad_() for key, value in sparse_predictions.items()
    }

    sparse_loss, sparse_pairs = _repeat_component_consistency(sparse_predictions, batch)
    dense_loss = _dense_repeat_consistency(dense_predictions, batch)
    sparse_loss.backward()
    dense_loss.backward()

    assert sparse_pairs > 0
    assert float(sparse_loss.detach()) == pytest.approx(float(dense_loss.detach()), abs=1e-7)
    for key in sparse_predictions:
        assert torch.allclose(
            sparse_predictions[key].grad,
            dense_predictions[key].grad,
            rtol=1e-5,
            atol=1e-7,
        )


def test_factorized_control_prevents_cross_role_identity_messages() -> None:
    torch.manual_seed(37)
    model = _model(role_isolated_attention=True).eval()
    batch = _batch()
    model_inputs = {
        key: value.clone()
        for key, value in batch.items()
        if key
        not in {
            "source_program_states",
            "atom_variable_mask",
            "parent_variable_mask",
            "parent_bond_variable_mask",
            "closure_endpoint_variable_mask",
            "closure_bond_variable_mask",
        }
    }
    changed = {key: value.clone() for key, value in model_inputs.items()}
    target_role = 1
    other_roles = changed["role_states"] != target_role
    changed["nodes"][other_roles] = (changed["nodes"][other_roles] + 1) % 5
    changed["core_position_states"][other_roles] = 1
    with torch.no_grad():
        reference = model(**model_inputs)
        intervention = model(**changed)
    target_nodes = model_inputs["role_states"] == target_role
    assert torch.equal(reference["nodes"][target_nodes], intervention["nodes"][target_nodes])
    assert torch.equal(
        reference["role_states"][target_nodes], intervention["role_states"][target_nodes]
    )
    assert torch.equal(
        reference["core_position_states"][target_nodes],
        intervention["core_position_states"][target_nodes],
    )


def test_family_balancing_does_not_amplify_a_converged_zero_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = torch.nn.ParameterList([parameter])
    diagnostic = balanced_pcgrad_backward({1: parameter.square(), 2: parameter.sum() * 0.0}, model)
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(1.0)
    assert diagnostic["raw_gradient_norms"] == pytest.approx([2.0, 0.0])
    assert diagnostic["family_weighting"] == "equal_loss_mass_without_norm_amplification"


def test_family_balancing_preserves_unused_parameter_semantics_and_accumulates() -> None:
    shared = torch.nn.Parameter(torch.tensor(0.0))
    first_only = torch.nn.Parameter(torch.tensor(0.0))
    model = torch.nn.ParameterList([shared, first_only])
    losses = {1: shared + first_only, 2: -shared}

    diagnostic = balanced_pcgrad_backward(losses, model, scale=0.5, materialize_diagnostics=False)
    assert shared.grad is not None
    assert first_only.grad is not None
    assert shared.grad.item() == pytest.approx(-0.125)
    assert first_only.grad.item() == pytest.approx(0.25)
    assert diagnostic["projected_conflicts"].item() == 2

    losses = {1: shared + first_only, 2: -shared}
    balanced_pcgrad_backward(losses, model, scale=0.5, materialize_diagnostics=False)
    assert shared.grad.item() == pytest.approx(-0.25)
    assert first_only.grad.item() == pytest.approx(0.5)


def test_batched_vjp_pcgrad_matches_sequential_shared_parameter_gradients() -> None:
    torch.manual_seed(203)
    sequential = torch.nn.Linear(4, 3)
    batched = torch.nn.Linear(4, 3)
    batched.load_state_dict(sequential.state_dict())
    inputs = torch.randn((9, 4))

    def losses(model: torch.nn.Module) -> dict[int, torch.Tensor]:
        output = model(inputs)
        return {
            family: output[start : start + 3].square().mean()
            for family, start in ((1, 0), (2, 3), (3, 6))
        }

    sequential_diagnostic = balanced_pcgrad_backward(
        losses(sequential), sequential, backend="sequential"
    )
    batched_diagnostic = balanced_pcgrad_backward(losses(batched), batched, backend="batched_vjp")

    assert sequential_diagnostic["projected_conflicts"] == batched_diagnostic["projected_conflicts"]
    assert sequential_diagnostic["raw_gradient_norms"] == pytest.approx(
        batched_diagnostic["raw_gradient_norms"], abs=1e-7
    )
    for left, right in zip(sequential.parameters(), batched.parameters(), strict=True):
        assert torch.equal(left.grad, right.grad)


def test_paired_topology_forward_matches_two_calls_without_dropout() -> None:
    class EchoModel(torch.nn.Module):
        def forward(self, **values: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                field: values[field].to(torch.float32).unsqueeze(-1)
                for field in (
                    "nodes",
                    "parents",
                    "parent_bonds",
                    "closure_left",
                    "closure_right",
                    "closure_bonds",
                )
            }

    clean = _batch()
    node_marginal = torch.full((5,), 0.2)
    bond_marginal = torch.full((4,), 0.25)
    t = torch.tensor([0.2, 0.5, 0.8])
    sequential_generator = torch.Generator().manual_seed(211)
    paired_generator = torch.Generator().manual_seed(211)

    predictions, noisy = synthesis_program_forward(
        EchoModel(), clean, node_marginal, bond_marginal, t, sequential_generator
    )
    topology = synthesis_program_topology_conditioned_forward(EchoModel(), clean, noisy, t)
    paired_predictions, paired_noisy, paired_topology = synthesis_program_paired_topology_forward(
        EchoModel(), clean, node_marginal, bond_marginal, t, paired_generator
    )

    assert all(torch.equal(noisy[key], paired_noisy[key]) for key in noisy)
    assert all(torch.equal(predictions[key], paired_predictions[key]) for key in predictions)
    assert all(torch.equal(topology[key], paired_topology[key]) for key in topology)
