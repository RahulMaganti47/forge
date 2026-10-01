"""Neural execution for the reviewer wrapper; imported only after input verification."""

from __future__ import annotations

import json
import tarfile
import time
from pathlib import Path
from typing import Any

import torch
from rdkit import rdBase

from forge.assembly import RegistryRepeatedReactionProgram, Ugi3AssemblyAdapter
from forge.corpus.reaction_program_training import load_reaction_program_specifications
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.reaction_program_evaluation import adjudicate_reaction_program_rows
from forge.model.synthesis_program_layout import SynthesisProgramLayoutPrior
from forge.model.synthesis_program_sampling import sample_synthesis_program_products
from forge.release.generate import ARM, DECODER, FLOW_STEPS, STEP
from forge.workflows.production_evaluation import (
    _load_checkpoint,
    _select_evaluation_snapshots,
    _validate_archive_members,
)
from forge.workflows.production_randomness import production_seed


def generate(
    report: dict[str, Any],
    *,
    program_id: str,
    count: int,
    seed: int,
    device: str,
    batch_size: int,
    threads: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reuse frozen loader, count prior, sampler and exact-L1 verifier without evaluation jobs."""
    if not report["ready"]:
        raise ValueError("generation requires verified inputs")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; select --device cpu")
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic()
    pins = report["inputs"]
    paths = {name: Path(row["path"]) for name, row in pins.items()}
    training = json.loads(paths["training_result"].read_text())
    design = json.loads(paths["production_design"].read_text())
    if (
        training.get("schema_version") != "forge.synthesis_program_production_training_result.v1"
        or training.get("status") != "pass"
        or training.get("replicate") != report["replicate"]
        or training.get("seed") != report["training_seed"]
    ):
        raise ValueError("training record does not match the selected paper replicate")
    arm = design["training"]["arms"][ARM]
    if float(arm["program_mass"].get(program_id, 0)) <= 0:
        raise ValueError("reaction program is not supported by the selected arm")
    (snapshot,) = _select_evaluation_snapshots(
        training["arms"][ARM]["checkpoints"], [STEP], arm_id=ARM
    )
    specs = {
        spec.program_id: spec
        for spec in load_reaction_program_specifications(paths["program_config"])
    }
    adapters: dict[str, Any] = {
        spec.program_id: RegistryRepeatedReactionProgram.from_registry(
            paths["qualified_reaction_families"],
            spec,
            expected_sha256=pins["qualified_reaction_families"]["expected_sha256"],
        )
        for spec in specs.values()
    }
    adapters["ugi_3cr_agile"] = Ugi3AssemblyAdapter.from_registry(
        paths["qualified_ugi_reactions"],
        expected_sha256=pins["qualified_ugi_reactions"]["expected_sha256"],
    )
    policy = ReactionCoreSaturationPolicy.from_qualified_registry(
        paths["qualified_ugi_reactions"],
        reaction_id="ugi_3cr_agile",
        expected_sha256=pins["qualified_ugi_reactions"]["expected_sha256"],
    )
    layout_seed = production_seed(seed, "reviewer_demo", program_id, "layout")
    flow_seed = production_seed(seed, "reviewer_demo", program_id, "flow")
    member = f"{ARM}/{snapshot['filename']}"
    with SynthesisProgramProductionCache(paths["production_cache"]) as cache:
        with tarfile.open(paths["checkpoint"], mode="r") as archive:
            _validate_archive_members(archive, training)
            model, package = _load_checkpoint(
                archive,
                member_name=member,
                expected_sha256=snapshot["sha256"],
                design_sha256=pins["production_design"]["expected_sha256"],
                cache_sha256=pins["production_cache"]["expected_sha256"],
                device=torch.device(device),
                cache=cache,
            )
        if (
            package.get("arm_id") != ARM
            or package.get("step") != STEP
            or package.get("seed") != report["training_seed"]
            or package.get("conditioning") != "program"
        ):
            raise ValueError("loaded checkpoint is not the selected paper conditioned model")
        prior = SynthesisProgramLayoutPrior(cache)
        layouts = prior.sample(
            program_id,
            sample_count=count,
            seed=layout_seed,
            role_morphology_conditioning=bool(
                package["model_config"].get("role_morphology_conditioning", False)
            ),
        )
        with torch.inference_mode():
            rows, sampling = sample_synthesis_program_products(
                model,
                layouts,
                cache.atom_vocabulary,
                package["node_marginal"].detach().cpu().numpy(),
                package["bond_marginal"].detach().cpu().numpy(),
                samples_per_program=1,
                sample_steps=FLOW_STEPS,
                batch_size=batch_size,
                seed=flow_seed,
                device=device,
                conditioning_mode="program",
                terminal_decode_policy=DECODER,
                reaction_core_saturation_policy=policy,
            )
        adjudicate_reaction_program_rows(
            rows, adapters=adapters, repeated_program_specs=specs, ugi_program_id="ugi_3cr_agile"
        )
    return rows, {
        "elapsed_seconds": time.monotonic() - started,
        "torch": torch.__version__,
        "rdkit": rdBase.rdkitVersion,
        "deterministic_algorithms": True,
        "tf32": False,
        "layout_seed": layout_seed,
        "flow_seed": flow_seed,
        "sampling": sampling,
        "checkpoint_member": member,
        "checkpoint_member_sha256": snapshot["sha256"],
    }
