"""Compose generated Ugi morphology, sparse closures, and chemistry."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, rdFingerprintGenerator

from experiments.phase1.product_l1.training.ugi_training_cache import load_ugi_training_cache
from forge.core.io import write_json as _atomic_json
from forge.corpus.ugi_chemistry_corpus import (
    load_expanded_ugi_chemistry_corpus,
    load_ugi_chemistry_corpus,
)
from forge.corpus.ugi_morphology_corpus import (
    load_expanded_ugi_morphology_corpus,
    load_ugi_morphology_corpus,
    sample_family_balanced_records,
)
from forge.model.defog_feasibility import sha256_file
from forge.model.ugi_chemistry_flow import (
    UgiChemistryFlow,
    chemistry_sample_statistics,
    chemistry_sample_to_molecule,
    sample_ugi_chemistry,
)
from forge.model.ugi_chemistry_interface import assemble_ugi_chemistry_topology_condition
from forge.model.ugi_closure_placement import UgiSparseClosureScorer, sample_sparse_closures
from forge.model.ugi_morphology_flow import UgiMorphologyFlow, sample_ugi_morphologies
from forge.model.ugi_morphology_program import (
    UgiMorphologyProgram,
    component_weighted_program_pool,
    sample_component_weighted_programs,
)
from forge.potency.annotations import ROLE_NAMES

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    torch = None


class UgiEndToEndSamplingError(RuntimeError):
    """Raised when checkpoint composition violates the generated-data contract."""


def _load_checkpoint(path: Path, schema: str | tuple[str, ...]) -> dict[str, Any]:
    if torch is None or not path.is_file():
        raise UgiEndToEndSamplingError(f"missing checkpoint: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    supported = (schema,) if isinstance(schema, str) else schema
    if value.get("schema_version") not in supported:
        raise UgiEndToEndSamplingError(f"unexpected checkpoint schema: {path}")
    return value


def _load_original_morphology(repo: Path) -> Any:
    assignments = repo / "data/splits/phase1/ugi_l1_assignments.csv.gz"
    semantic_products = repo / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_products.csv.gz"
    semantic_atoms = repo / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_atoms.csv.gz"
    vocabulary = repo / "results/phase1/product_v3_atom_vocabulary.json"
    return load_ugi_morphology_corpus(
        assignments,
        semantic_products,
        semantic_atoms,
        vocabulary,
    )


def _load_original_chemistry(repo: Path) -> Any:
    return load_ugi_chemistry_corpus(
        repo / "data/splits/phase1/ugi_l1_assignments.csv.gz",
        repo / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_products.csv.gz",
        repo / "results/phase1/ugi_l1_semantics/ugi_l1_semantic_atoms.csv.gz",
        repo / "results/phase1/product_v3_atom_vocabulary.json",
    )


def _load_expanded_morphology(repo: Path) -> Any:
    return load_expanded_ugi_morphology_corpus(
        repo / "results/phase1/ugi_expanded_exemplars/component_exemplar_ledger.csv.gz",
        repo / "results/phase1/ugi_expanded_exemplars/semantic_products.csv.gz",
        repo / "results/phase1/ugi_expanded_exemplars/semantic_atoms.csv.gz",
        repo / "results/phase1/product_v3_atom_vocabulary.json",
    )


def _load_expanded_chemistry(repo: Path) -> Any:
    return load_expanded_ugi_chemistry_corpus(
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/assignments.csv.gz",
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_products.csv.gz",
        repo / "results/phase1/ugi_expanded_chemistry_exemplars/semantic_atoms.csv.gz",
        repo / "results/phase1/product_v3_atom_vocabulary.json",
    )


def _load_program_probe(path: Path) -> tuple[UgiMorphologyProgram, ...]:
    value = json.loads(path.read_text())
    programs = []
    for row in value["samples"]:
        program = row["program"]
        programs.append(
            UgiMorphologyProgram(
                node_counts=tuple(program["node_counts"]),
                junction_budgets=tuple(program["junction_budgets"]),
                cycle_ranks=tuple(program["cycle_ranks"]),
                attachment_counts=tuple(program["attachment_counts"]),
            )
        )
    if not programs:
        raise UgiEndToEndSamplingError("fixed morphology-program probe is empty")
    return tuple(programs)


def _morphology_model(checkpoint: dict[str, Any]) -> Any:
    config = dict(checkpoint["model_config"])
    config.pop("source_probability_floor", None)
    model = UgiMorphologyFlow(**config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _closure_model(checkpoint: dict[str, Any]) -> Any:
    model = UgiSparseClosureScorer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def _chemistry_model(checkpoint: dict[str, Any], atom_classes: int) -> Any:
    config = dict(checkpoint["model_config"])
    source_probability_floor = config.pop("source_probability_floor", None)
    if source_probability_floor is None:
        raise UgiEndToEndSamplingError("chemistry checkpoint lacks source support")
    model = UgiChemistryFlow(atom_classes=atom_classes, **config)
    model.load_state_dict(checkpoint["model_state"])
    return model


def _descriptor_summary(molecules: list[Chem.Mol]) -> dict[str, dict[str, float]]:
    functions = {
        "heavy_atoms": lambda molecule: molecule.GetNumHeavyAtoms(),
        "molecular_weight": Descriptors.MolWt,
        "logp": Crippen.MolLogP,
        "rings": Lipinski.RingCount,
        "rotatable_bonds": Lipinski.NumRotatableBonds,
        "heteroatoms": Lipinski.NumHeteroatoms,
    }
    output = {}
    for label, function in functions.items():
        values = np.asarray([function(molecule) for molecule in molecules], dtype=np.float64)
        output[label] = {
            "q05": float(np.quantile(values, 0.05)),
            "median": float(np.median(values)),
            "q95": float(np.quantile(values, 0.95)),
        }
    return output


def _reference_comparison(
    generated: list[Chem.Mol],
    chemistry_corpus: Any,
) -> dict[str, Any]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    generated_smiles = [
        Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) for molecule in generated
    ]
    output: dict[str, Any] = {"generated_descriptors": _descriptor_summary(generated)}
    for label, folds in (
        ("strict_train", ("train",)),
        ("all_frozen_ugi", ("train", "calibration", "heldout")),
    ):
        reference_smiles = [
            row["canonical_product_smiles"]
            for fold in folds
            for row in chemistry_corpus.assignments_by_fold[fold]
        ]
        reference_molecules = [Chem.MolFromSmiles(smiles) for smiles in reference_smiles]
        if any(molecule is None for molecule in reference_molecules):
            raise UgiEndToEndSamplingError("frozen Ugi reference contains an invalid molecule")
        canonical_reference = {
            Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            for molecule in reference_molecules
        }
        reference_fingerprints = [
            generator.GetFingerprint(molecule) for molecule in reference_molecules
        ]
        nearest = [
            max(
                DataStructs.BulkTanimotoSimilarity(
                    generator.GetFingerprint(molecule),
                    reference_fingerprints,
                )
            )
            for molecule in generated
        ]
        output[label] = {
            "reference_molecules": len(reference_molecules),
            "exact_matches": sum(smiles in canonical_reference for smiles in generated_smiles),
            "nearest_morgan_tanimoto": {
                "minimum": float(min(nearest)),
                "median": float(np.median(nearest)),
                "maximum": float(max(nearest)),
            },
            "reference_descriptors": _descriptor_summary(reference_molecules),
        }
    return output


def sample_ugi_end_to_end(
    repo: Path,
    output_dir: Path,
    *,
    morphology_checkpoint_path: Path,
    closure_checkpoint_path: Path,
    chemistry_checkpoint_path: Path,
    sample_count: int,
    morphology_steps: int,
    chemistry_steps: int,
    batch_size: int,
    seed: int,
    overwrite: bool,
    program_probe_path: Path | None = None,
) -> dict[str, Any]:
    """Generate complete molecules without target topology or chemistry inputs."""

    if torch is None or sample_count < 1 or morphology_steps < 2 or chemistry_steps < 2:
        raise UgiEndToEndSamplingError("invalid end-to-end sampling request")
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise UgiEndToEndSamplingError(f"output directory is nonempty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    morphology_checkpoint = _load_checkpoint(
        morphology_checkpoint_path,
        (
            "phase1_ugi_morphology_checkpoint.v1",
            "phase1_ugi_morphology_checkpoint.v2",
        ),
    )
    closure_checkpoint = _load_checkpoint(
        closure_checkpoint_path,
        (
            "phase1_ugi_sparse_closure_checkpoint.v1",
            "phase1_ugi_sparse_closure_checkpoint.v2",
        ),
    )
    chemistry_checkpoint = _load_checkpoint(
        chemistry_checkpoint_path,
        (
            "phase1_ugi_chemistry_checkpoint.v1",
            "phase1_ugi_chemistry_checkpoint.v2",
        ),
    )
    morphology_corpus = (
        _load_expanded_morphology(repo)
        if morphology_checkpoint["schema_version"] == "phase1_ugi_morphology_checkpoint.v2"
        else _load_original_morphology(repo)
    )
    if "prepared_cache" in chemistry_checkpoint["inputs"]:
        cache_path = Path(chemistry_checkpoint["inputs"]["prepared_cache"]["path"])
        if not cache_path.is_file() and str(cache_path).startswith("/root/forge_repo/"):
            cache_path = repo / cache_path.relative_to("/root/forge_repo")
        chemistry_corpus, _ = load_ugi_training_cache(cache_path)
    else:
        chemistry_corpus = (
            _load_expanded_chemistry(repo)
            if chemistry_checkpoint["schema_version"] == "phase1_ugi_chemistry_checkpoint.v2"
            else _load_original_chemistry(repo)
        )
    morphology_model = _morphology_model(morphology_checkpoint)
    closure_model = _closure_model(closure_checkpoint)
    chemistry_model = _chemistry_model(
        chemistry_checkpoint,
        len(chemistry_corpus.atom_vocabulary),
    )
    rng = np.random.default_rng(seed)
    if program_probe_path is not None:
        programs = _load_program_probe(program_probe_path)
        if len(programs) != sample_count:
            raise UgiEndToEndSamplingError(
                "fixed morphology-program probe count disagrees with sample count"
            )
        morphology_program_source = "fixed_source_branch_stratified_probe"
    elif morphology_checkpoint["schema_version"] == "phase1_ugi_morphology_checkpoint.v2":
        programs = tuple(
            record.program
            for record in sample_family_balanced_records(
                morphology_corpus,
                fold="train",
                count=sample_count,
                rng=rng,
                id_prefix="end-to-end",
            )
        )
        morphology_program_source = "expanded_train_family_balanced"
    else:
        pools = component_weighted_program_pool(morphology_corpus.records_by_fold["train"])
        programs = sample_component_weighted_programs(
            pools,
            count=sample_count,
            rng=rng,
        )
        morphology_program_source = "original_train_component_weighted"
    start = time.perf_counter()
    morphology_samples, morphology_sampling = sample_ugi_morphologies(
        morphology_model,
        programs,
        np.asarray(morphology_checkpoint["source_marginals"], dtype=np.float64),
        sample_steps=morphology_steps,
        batch_size=batch_size,
        seed=seed + 1,
        device="cpu",
    )
    closure_generator = torch.Generator().manual_seed(seed + 2)
    conditions = []
    topology_rows = []
    for sample_index, sample in enumerate(morphology_samples):
        offspring_by_role = {
            role: sample.offspring[role_index] for role_index, role in enumerate(ROLE_NAMES)
        }
        left_by_role = {}
        right_by_role = {}
        for role_index, role in enumerate(ROLE_NAMES):
            left, right = sample_sparse_closures(
                closure_model,
                sample.offspring[role_index],
                role_index=role_index,
                cycle_rank=sample.program.cycle_ranks[role_index],
                attachment_count=sample.program.attachment_counts[role_index],
                generator=closure_generator,
                allowed_ring_sizes=closure_checkpoint["allowed_ring_sizes"],
                maximum_heavy_degree=int(closure_checkpoint["maximum_heavy_degree"]),
                device="cpu",
            )
            left_by_role[role] = left
            right_by_role[role] = right
        condition = assemble_ugi_chemistry_topology_condition(
            structure_id=f"generated_{sample_index:04d}",
            offspring_by_role=offspring_by_role,
            attachment_counts_by_role={
                role: sample.program.attachment_counts[role_index]
                for role_index, role in enumerate(ROLE_NAMES)
            },
            closure_left_by_role=left_by_role,
            closure_right_by_role=right_by_role,
            schema=chemistry_corpus.core_schema,
        )
        conditions.append(condition)
        topology_rows.append(
            {
                "structure_id": condition.structure_id,
                "program": {
                    "node_counts": sample.program.node_counts,
                    "junction_budgets": sample.program.junction_budgets,
                    "cycle_ranks": sample.program.cycle_ranks,
                    "attachment_counts": sample.program.attachment_counts,
                },
                "offspring_by_role": {
                    role: offspring_by_role[role].tolist() for role in ROLE_NAMES
                },
                "closure_pairs_by_role": {
                    role: [
                        list(edge)
                        for edge in zip(
                            left_by_role[role].tolist(),
                            right_by_role[role].tolist(),
                            strict=True,
                        )
                    ]
                    for role in ROLE_NAMES
                },
            }
        )
    chemistry_samples, chemistry_sampling = sample_ugi_chemistry(
        chemistry_model,
        tuple(conditions),
        {
            key: np.asarray(value, dtype=np.float64)
            for key, value in chemistry_checkpoint["source_marginals"].items()
        },
        atom_classes=len(chemistry_corpus.atom_vocabulary),
        sample_steps=chemistry_steps,
        batch_size=batch_size,
        seed=seed + 3,
        device="cpu",
        atom_vocabulary=chemistry_corpus.atom_vocabulary,
        allow_terminal_failures=True,
    )
    generated_rows = []
    molecules = []
    legends = []
    for topology, condition, sample in zip(
        topology_rows,
        conditions,
        chemistry_samples,
        strict=True,
    ):
        if sample is None:
            generated_rows.append(
                {
                    **topology,
                    "valid": False,
                    "smiles": None,
                    "failure_type": "TerminalSupportFailure",
                }
            )
            continue
        try:
            molecule = chemistry_sample_to_molecule(
                condition,
                sample,
                chemistry_corpus.atom_vocabulary,
            )
            smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
            valid = True
            molecules.append(molecule)
            legends.append(condition.structure_id)
        except (ValueError, RuntimeError):
            smiles = None
            valid = False
        generated_rows.append(
            {
                **topology,
                "valid": valid,
                "smiles": smiles,
                "failure_type": None if valid else "MoleculeSanitizationFailure",
            }
        )
    statistics = chemistry_sample_statistics(
        tuple(conditions),
        chemistry_samples,
        chemistry_corpus.atom_vocabulary,
    )
    valid_smiles = [row["smiles"] for row in generated_rows if row["smiles"] is not None]
    statistics["unique_valid_fraction"] = (
        len(set(valid_smiles)) / len(valid_smiles) if valid_smiles else 0.0
    )
    reference_comparison = _reference_comparison(molecules, chemistry_corpus) if molecules else None
    image_path = output_dir / "samples.png"
    if molecules:
        image = Draw.MolsToGridImage(
            molecules,
            molsPerRow=3,
            subImgSize=(500, 300),
            legends=legends,
            useSVG=False,
        )
        image.save(image_path)
    result = {
        "schema_version": "phase1_ugi_end_to_end_sampling_result.v1",
        "status": "complete",
        "seed": seed,
        "checkpoints": {
            "morphology": {
                "path": str(morphology_checkpoint_path),
                "sha256": sha256_file(morphology_checkpoint_path),
            },
            "closure": {
                "path": str(closure_checkpoint_path),
                "sha256": sha256_file(closure_checkpoint_path),
            },
            "chemistry": {
                "path": str(chemistry_checkpoint_path),
                "sha256": sha256_file(chemistry_checkpoint_path),
            },
        },
        "sampling": {
            "sample_count": sample_count,
            "morphology": morphology_sampling,
            "chemistry": chemistry_sampling,
            "total_seconds": time.perf_counter() - start,
        },
        "boundary": {
            "topology": "generated",
            "closure_endpoints": "generated_sparse_feasible_pairs",
            "atom_and_bond_chemistry": "generated_discrete_flow",
            "component_catalog_ids_in_model_state": False,
            "route_oracle_or_guidance": False,
            "training_scope": "bounded_smoke_checkpoints_not_production_claim",
            "morphology_program_source": morphology_program_source,
        },
        "statistics": statistics,
        "reference_comparison": reference_comparison,
        "samples": generated_rows,
        "render": str(image_path) if image_path.is_file() else None,
    }
    _atomic_json(output_dir / "result.json", result)
    return result
