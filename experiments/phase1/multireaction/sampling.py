"""Checkpoint-driven bounded sampling for the multi-reaction sparse flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.assembly import RegistryRepeatedReactionProgram
from forge.core.hashing import pin_record, resolve_pin, sha256_file
from forge.core.io import atomic_write, pretty_json_bytes, read_csv_rows, read_json_object
from forge.corpus.reaction_program_records import repeat_component_smiles
from forge.corpus.reaction_program_training import load_reaction_program_training_corpus
from forge.model.reaction_program_evaluation import evaluate_reaction_program_samples
from forge.model.reaction_program_sampling import (
    load_reaction_program_checkpoint,
    sample_factorized_program_layouts,
    sample_reaction_program_products,
    sample_training_semantic_layouts,
)

CONFIG_SCHEMA = "forge.multireaction_sampling_config.v2"
RESULT_SCHEMA = "forge.multireaction_sampling_result.v2"
SAMPLES_SCHEMA = "forge.multireaction_samples.v2"


class MultiReactionSamplingError(ValueError):
    """The checkpoint-driven sampling smoke violates its pinned contract."""


def run_multireaction_sampling(
    config_path: Path,
    repo: Path,
    checkpoint_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Load a checkpoint, sample coarse programs and audit exact L1 decompositions."""

    config = read_json_object(
        config_path,
        error=MultiReactionSamplingError,
        label="multi-reaction sampling config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise MultiReactionSamplingError("unsupported multi-reaction sampling config")
    run_kind = config.get("run_kind")
    if run_kind not in {"diagnostic_smoke", "overfit_gate", "production_qualification"}:
        raise MultiReactionSamplingError("multi-reaction sampling run_kind is invalid")
    input_paths = {
        label: resolve_pin(record, repo, label=label) for label, record in config["inputs"].items()
    }
    device = str(config["execution"]["device"])
    model, vocabulary, atom_vocabulary, node_marginal, bond_marginal, checkpoint = (
        load_reaction_program_checkpoint(checkpoint_path, device=device)
    )
    expected_inputs = {label: pin_record(path, repo) for label, path in sorted(input_paths.items())}
    training_inputs = {
        label: record
        for label, record in expected_inputs.items()
        if label != "qualified_reaction_families"
    }
    if checkpoint.get("inputs") != training_inputs:
        raise MultiReactionSamplingError("checkpoint and sampling inputs do not agree")
    corpus = load_reaction_program_training_corpus(
        program_config_path=input_paths["program_config"],
        atlas_path=input_paths["atlas"],
        semantic_atoms_path=input_paths["semantic_atoms"],
        splits_path=input_paths["splits"],
        declared_elements=set(config["model_contract"]["declared_elements"]),
    )
    if corpus.vocabulary != vocabulary or corpus.atom_vocabulary != atom_vocabulary:
        raise MultiReactionSamplingError("checkpoint vocabulary differs from the sampling corpus")
    registry = input_paths["qualified_reaction_families"]
    adapters = {
        spec.program_id: RegistryRepeatedReactionProgram.from_registry(
            registry,
            spec,
            expected_sha256=str(config["inputs"]["qualified_reaction_families"]["sha256"]),
        )
        for spec in corpus.specifications
    }
    layout_source = config["sampling"].get(
        "layout_source", "factorized_training_fold_count_marginals"
    )
    if layout_source == "factorized_training_fold_count_marginals":
        layouts = sample_factorized_program_layouts(
            corpus.records_by_fold["train"],
            corpus.weights_by_fold["train"],
            corpus.specifications,
            corpus.vocabulary,
            sample_count=int(config["sampling"]["samples"]),
            seed=int(config["seed"]),
        )
    elif layout_source == "checkpoint_training_semantics":
        selected_ids = checkpoint.get("training_record_ids")
        if not isinstance(selected_ids, list) or not selected_ids:
            raise MultiReactionSamplingError("checkpoint has no overfit training selection")
        selected = tuple(
            record
            for record in corpus.records_by_fold["train"]
            if record.graph.structure_id in set(selected_ids)
        )
        if {record.graph.structure_id for record in selected} != set(selected_ids):
            raise MultiReactionSamplingError("checkpoint training selection changed")
        layouts = sample_training_semantic_layouts(
            selected,
            corpus.vocabulary,
            sample_count=int(config["sampling"]["samples"]),
            seed=int(config["seed"]),
        )
    else:
        raise MultiReactionSamplingError(f"unsupported layout source: {layout_source!r}")
    rows, metrics = sample_reaction_program_products(
        model,
        layouts,
        atom_vocabulary,
        node_marginal,
        bond_marginal,
        adapters=adapters,
        sample_steps=int(config["sampling"]["steps"]),
        batch_size=int(config["sampling"]["batch_size"]),
        seed=int(config["seed"]) + 1,
        device=device,
    )
    split_rows = read_csv_rows(
        input_paths["splits"],
        error=MultiReactionSamplingError,
        label="reaction-program splits",
        required_fields=("record_id", "product_fold"),
    )
    training_ids = {row["record_id"] for row in split_rows if row["product_fold"] == "train"}
    atlas_rows = read_csv_rows(
        input_paths["atlas"],
        error=MultiReactionSamplingError,
        label="reaction-program atlas",
        required_fields=(
            "record_id",
            "program_id",
            "terminal_head_smiles",
            "repeat_component_smiles",
        ),
    )
    training_products: dict[str, set[str]] = {
        program_id: set() for program_id in vocabulary.program_states[1:]
    }
    training_components: dict[str, dict[str, set[str]]] = {
        program_id: {"terminal_head": set(), "repeat_component": set()}
        for program_id in vocabulary.program_states[1:]
    }
    for row in atlas_rows:
        if row["record_id"] in training_ids:
            program_id = row["program_id"]
            training_components[program_id]["terminal_head"].add(row["terminal_head_smiles"])
            training_components[program_id]["repeat_component"].update(repeat_component_smiles(row))
    for record in corpus.records_by_fold["train"]:
        training_products[record.program_id].add(record.graph.canonical_smiles)
    evaluation = evaluate_reaction_program_samples(
        rows,
        training_products=training_products,
        training_components=training_components,
    )
    samples_document = {
        "schema_version": SAMPLES_SCHEMA,
        "seed": int(config["seed"]),
        "layout_source": layout_source,
        "component_identifiers_used": False,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write(output_dir / "samples.json", pretty_json_bytes(samples_document))
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": f"{run_kind}_complete",
        "run_kind": run_kind,
        "seed": int(config["seed"]),
        "config": pin_record(config_path, repo),
        "inputs": expected_inputs,
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": str(sha256_file(checkpoint_path)),
            "model_state_sha256": checkpoint["model_state_sha256"],
        },
        "metrics": {**metrics, "evaluation": evaluation},
        "layout_source": layout_source,
        "component_identifiers_used": False,
        "claim_boundary": str(config["claim_boundary"]),
    }
    atomic_write(output_dir / "result.json", pretty_json_bytes(result))
    return result


__all__ = ["MultiReactionSamplingError", "run_multireaction_sampling"]
