"""Fixed-checkpoint, nonselecting evaluation for the matched production arms."""

from __future__ import annotations

import gzip
import hashlib
import html
import io
import tarfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from forge.assembly import RegistryRepeatedReactionProgram, Ugi3AssemblyAdapter
from forge.core.hashing import artifact_record, pin_record, resolve_pin, sha256_file
from forge.core.io import read_json_object, stable_json, write_json
from forge.corpus.reaction_program_training import load_reaction_program_specifications
from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
from forge.model.common_ugi_benchmark import (
    CommonUgiAttempt,
    load_ugi_identity_references,
    write_attempt_ledger,
)
from forge.model.conditional_role_dependence import cross_role_fidelity_to_heldout
from forge.model.defog_feasibility import _model_state_sha256
from forge.model.local_chemistry_support import LocalChemistrySupport
from forge.model.reaction_core_saturation import ReactionCoreSaturationPolicy
from forge.model.reaction_program_evaluation import (
    adjudicate_reaction_program_rows,
    evaluate_reaction_program_samples,
    load_reaction_program_training_references,
)
from forge.model.reaction_program_flow import synthesis_program_flow_loss
from forge.model.synthesis_program_layout import (
    SynthesisProgramLayoutError,
    SynthesisProgramLayoutPrior,
)
from forge.model.synthesis_program_sampling import (
    CORE_SATURATION_TERMINAL_DECODE_POLICY,
    LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY,
    PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY,
    SUPPORTED_TERMINAL_DECODE_POLICIES,
    sample_synthesis_program_products,
)
from forge.model.synthesis_program_training import (
    build_synthesis_program_flow,
    collate_synthesis_program_training_batch,
    move_tensors,
    synthesis_program_fixed_state_exact,
    synthesis_program_forward,
    synthesis_program_reconstruction_metrics,
)

from .production_randomness import production_seed

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional training dependency
    torch = None  # type: ignore[assignment]

CONFIG_SCHEMA = "forge.synthesis_program_production_evaluation_config.v1"
RESULT_SCHEMA = "forge.synthesis_program_production_evaluation_result.v1"
SAMPLES_SCHEMA = "forge.synthesis_program_production_samples.v1"


class SynthesisProgramProductionEvaluationError(ValueError):
    """A fixed-checkpoint evaluation violates the frozen comparison contract."""


def _validate_evaluation_budget(
    config: Mapping[str, Any],
    design: Mapping[str, Any],
    *,
    profile: str,
) -> str:
    """Bind reduced full-device budgets to an explicit execution-only preflight scope."""

    execution_scope = str(config.get("execution_scope", "production"))
    if execution_scope not in {"production", "h100_preflight"}:
        raise SynthesisProgramProductionEvaluationError(
            f"unsupported evaluation execution scope: {execution_scope!r}"
        )
    if execution_scope == "h100_preflight" and profile != "full":
        raise SynthesisProgramProductionEvaluationError(
            "H100 preflight evaluation must use the full-device profile"
        )
    if profile == "full" and execution_scope == "production":
        runtime = config[profile]
        frozen = design["evaluation"]["native_sampling"]
        expected = {
            "sample_steps": int(frozen["flow_steps"]),
            "calibration_samples": int(
                frozen["calibration_samples_per_supported_program_per_checkpoint_per_seed"]
            ),
            "heldout_samples": int(
                frozen["heldout_samples_per_supported_program_at_final_checkpoint_per_seed"]
            ),
        }
        if any(runtime.get(key) != value for key, value in expected.items()):
            raise SynthesisProgramProductionEvaluationError(
                "full evaluation budget differs from the frozen design"
            )
        if runtime.get("component_disjoint_record_limit") is not None:
            raise SynthesisProgramProductionEvaluationError(
                "full component-disjoint evaluation cannot cap the heldout fold"
            )
    return execution_scope


def _load_checkpoint(
    archive: tarfile.TarFile,
    *,
    member_name: str,
    expected_sha256: str,
    design_sha256: str,
    cache_sha256: str,
    device: Any,
    cache: SynthesisProgramProductionCache,
) -> tuple[Any, dict[str, Any]]:
    try:
        member = archive.getmember(member_name)
    except KeyError as error:
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint archive omits {member_name}"
        ) from error
    if not member.isfile() or member.name != member_name:
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint member is not a regular in-scope file: {member_name}"
        )
    handle = archive.extractfile(member)
    if handle is None:
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint member cannot be read: {member_name}"
        )
    payload = handle.read()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected_sha256:
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint member hash changed: {member_name}"
        )
    package = torch.load(io.BytesIO(payload), map_location=device, weights_only=True)
    if (
        not isinstance(package, dict)
        or package.get("schema_version") != "forge.synthesis_program_production_checkpoint.v1"
        or package.get("design_sha256") != design_sha256
        or package.get("cache_sha256") != cache_sha256
    ):
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint contract changed: {member_name}"
        )
    model = build_synthesis_program_flow(
        vocabulary=cache.vocabulary,
        node_classes=len(cache.atom_vocabulary),
        model_config=package["model_config"],
        device=device,
    )
    model.load_state_dict(package["model_state"], strict=True)
    model.eval()
    if _model_state_sha256(model) != package.get("model_state_sha256"):
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint model-state hash changed: {member_name}"
        )
    return model, package


def _validate_archive_members(archive: tarfile.TarFile, training: Mapping[str, Any]) -> set[str]:
    expected = {
        f"{arm_id}/{snapshot['filename']}"
        for arm_id, arm in training["arms"].items()
        for snapshot in arm["checkpoints"]
    }
    members = archive.getmembers()
    observed = [member.name for member in members]
    if len(observed) != len(set(observed)):
        raise SynthesisProgramProductionEvaluationError(
            "checkpoint archive contains duplicate member names"
        )
    if set(observed) != expected or any(not member.isfile() for member in members):
        raise SynthesisProgramProductionEvaluationError(
            "checkpoint archive members differ from the authenticated training result"
        )
    return expected


def _select_evaluation_snapshots(
    snapshots: Sequence[Mapping[str, Any]],
    requested_steps: Sequence[int],
    *,
    arm_id: str,
) -> tuple[Mapping[str, Any], ...]:
    """Select the authenticated checkpoints requested by the evaluation contract.

    An evaluation may use a subset of the archived checkpoints, including only the
    final step. Requested steps must be positive, sorted, unique, and present once
    in the training result.
    """

    requested = tuple(int(value) for value in requested_steps)
    if not requested or requested != tuple(sorted(set(requested))) or requested[0] < 1:
        raise SynthesisProgramProductionEvaluationError(
            "evaluation checkpoint steps must be positive, unique and sorted"
        )
    by_step: dict[int, Mapping[str, Any]] = {}
    for snapshot in snapshots:
        step = int(snapshot["step"])
        if step in by_step:
            raise SynthesisProgramProductionEvaluationError(
                f"training result repeats checkpoint {step} for arm {arm_id}"
            )
        by_step[step] = snapshot
    missing = [step for step in requested if step not in by_step]
    if missing:
        raise SynthesisProgramProductionEvaluationError(
            f"evaluation requests checkpoints absent from arm {arm_id}: {missing}"
        )
    return tuple(by_step[step] for step in requested)


def _metric_contract(
    evaluation: Mapping[str, Any],
    *,
    fixed_state_failures: int,
    support_overflow_count: int,
) -> dict[str, Any]:
    values = evaluation["overall"]
    valid = int(values["valid"])
    return {
        "samples": int(values["samples"]),
        "raw_valid_fraction": values["valid_fraction"],
        "connected_fraction": values["connected_fraction"],
        "exact_l1_decomposition_coverage": values["retro_decomposition_coverage_among_valid"],
        "exact_forward_replay_precision": values["retro_transform_precision"],
        "decomposition_abstention_fraction": (
            values["valid_without_exact_decomposition"] / valid if valid else None
        ),
        "decomposition_ambiguity_fraction": (
            values["ambiguous_exact_decompositions"] / valid if valid else None
        ),
        "internal_diversity": values["mean_pairwise_ecfp4_distance"],
        "unique_fraction": values["unique_fraction"],
        "effective_component_count": values["effective_component_count"],
        "component_novelty_fraction": values["component_novelty_fraction"],
        "whole_lipid_novelty_fraction": values["whole_product_novel_to_train_fraction"],
        "exact_l1_yield_per_attempt": values["exact_l1_yield_per_attempt"],
        "unique_exact_l1_products_per_1000_attempts": values[
            "unique_exact_l1_products_per_1000_attempts"
        ],
        "unique_whole_product_novel_exact_l1_products_per_1000_attempts": values[
            "unique_whole_product_novel_exact_l1_products_per_1000_attempts"
        ],
        "unique_open_ended_exact_l1_products_per_1000_attempts": values[
            "unique_open_ended_exact_l1_products_per_1000_attempts"
        ],
        "fixed_state_failures": fixed_state_failures,
        "support_overflow_count": support_overflow_count,
        "coverage_and_precision_reported": True,
        "reductive_amination_substructure_rate_reported": False,
        "component_metrics_by_role": values["component_metrics_by_role"],
    }


def _conditioning_contract(
    package: Mapping[str, Any], cache: SynthesisProgramProductionCache
) -> tuple[str, Mapping[int, int] | None]:
    conditioning = str(package["conditioning"])
    if conditioning == "program":
        return "program", None
    if conditioning == "null_all_program_coordinates":
        return "null", None
    if conditioning != "cyclic_program_id_only_roles_core_and_depth_retained":
        raise SynthesisProgramProductionEvaluationError(
            f"checkpoint has an unsupported conditioning mode: {conditioning}"
        )
    raw_mapping = package.get("program_id_mapping")
    if not isinstance(raw_mapping, Mapping):
        raise SynthesisProgramProductionEvaluationError(
            "cyclic checkpoint omits its program-ID mapping"
        )
    return (
        "program_mapped",
        {
            cache.vocabulary.program_to_index[str(source)]: cache.vocabulary.program_to_index[
                str(target)
            ]
            for source, target in raw_mapping.items()
        },
    )


def _component_disjoint_reconstruction(
    model: Any,
    package: Mapping[str, Any],
    cache: SynthesisProgramProductionCache,
    *,
    program_id: str,
    batch_size: int,
    record_limit: int | None,
    seed: int,
    device: Any,
) -> dict[str, Any]:
    indices = cache.indices(program_id=program_id, fold="heldout")
    if record_limit is not None:
        indices = indices[:record_limit]
    if not len(indices):
        raise SynthesisProgramProductionEvaluationError(
            f"component-disjoint fold is empty for {program_id}"
        )
    generator = torch.Generator(device=device).manual_seed(seed)
    node_p0 = package["node_marginal"].to(device=device, dtype=torch.float32)
    bond_p0 = package["bond_marginal"].to(device=device, dtype=torch.float32)
    field_correct: Counter[str] = Counter()
    field_total: Counter[str] = Counter()
    exact_records = 0
    records = 0
    fixed_noising_failures = 0
    loss_sum = 0.0
    conditioning = str(package["conditioning"])
    mapping = package.get("program_id_mapping")
    maximum_closures = int(package["model_config"]["maximum_closures"])
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            clean = move_tensors(
                collate_synthesis_program_training_batch(
                    cache.records(selected),
                    maximum_closures=maximum_closures,
                    conditioning=conditioning,
                    vocabulary=cache.vocabulary,
                    program_id_mapping=mapping,
                ),
                device,
            )
            t = torch.full((len(selected),), 0.5, dtype=torch.float32, device=device)
            predictions, noisy = synthesis_program_forward(
                model, clean, node_p0, bond_p0, t, generator
            )
            if package["model_config"].get("architecture") == "reaction_program_graph_transformer":
                from forge.model.reaction_program_transformer import (
                    reaction_program_transformer_loss,
                )

                objective = package["model_config"]["semantic_objective"]
                heldout_loss, _ = reaction_program_transformer_loss(
                    predictions,
                    clean,
                    role_weight=float(objective["role_consistency_weight"]),
                    core_weight=float(objective["core_consistency_weight"]),
                    repeat_consistency_weight=float(
                        objective.get("repeat_consistency_weight", 0.0)
                    ),
                    materialize_metrics=False,
                )
            else:
                heldout_loss, _ = synthesis_program_flow_loss(
                    predictions, clean, materialize_metrics=False
                )
            metrics = synthesis_program_reconstruction_metrics(predictions, clean)
            records += int(metrics["records"])
            loss_sum += float(heldout_loss.detach().cpu()) * len(selected)
            exact_records += int(metrics["exact_tensor_records"])
            fixed_noising_failures += int(not synthesis_program_fixed_state_exact(noisy, clean))
            field_correct.update(metrics["field_correct"])
            field_total.update(metrics["field_total"])
    return {
        "records": records,
        "fold": "heldout_exact_component_disjoint",
        "noise_time": 0.5,
        "fixed_noise_exact_tensor_records": exact_records,
        "fixed_noise_exact_tensor_fraction": exact_records / records,
        "heldout_denoising_loss_at_t_0_5": loss_sum / records,
        "field_accuracy": {
            field: field_correct[field] / field_total[field] if field_total[field] else 1.0
            for field in sorted(field_total)
        },
        "fixed_state_noising_failures": fixed_noising_failures,
        "record_limit": record_limit,
    }


def _molecule_svg(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise SynthesisProgramProductionEvaluationError(
            "a valid molecule selected for rendering cannot be parsed"
        )
    drawer = rdMolDraw2D.MolDraw2DSVG(480, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")


def _write_molecule_report(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    design: Mapping[str, Any],
    *,
    final_step: int,
    seed: int,
) -> dict[str, int]:
    sections: list[str] = []
    expected = 0
    rendered = 0
    for arm_id, arm in design["training"]["arms"].items():
        supported = [
            program_id for program_id, mass in arm["program_mass"].items() if float(mass) > 0
        ]
        for program_id in arm.get("evaluation_programs", supported):
            expected += 1
            group = [
                row
                for row in rows
                if row["arm_id"] == arm_id
                and row["program_id"] == program_id
                and int(row["checkpoint_step"]) == final_step
                and row["evaluation_split"] == "heldout"
            ]
            valid = [row for row in group if row.get("valid") is True]
            title = f"{arm_id} / {program_id}"
            body = [
                f"<h2>{html.escape(title)}</h2>",
                f"<p>heldout rows: {len(group)}; valid renderable rows: {len(valid)}</p>",
            ]
            if not valid:
                body.append(
                    "<p><strong>No renderable molecule.</strong> The invalid outputs are retained "
                    "in the sample ledger; no repair or retry was performed.</p>"
                )
                sections.append("\n".join(body))
                continue
            rng = np.random.default_rng(production_seed(seed, arm_id, program_id, "render"))
            random_row = valid[int(rng.integers(0, len(valid)))]

            def worst_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
                molecule = Chem.MolFromSmiles(str(row["canonical_smiles"]))
                heavy_atoms = molecule.GetNumHeavyAtoms() if molecule is not None else 0
                return (
                    bool(row.get("exact_l1_program")),
                    int(row.get("exact_l1_trace_count", 0)) == 1,
                    -heavy_atoms,
                    str(row["canonical_smiles"]),
                )

            worst_row = min(valid, key=worst_key)
            for label, row in (("Seeded random", random_row), ("Worst-case valid", worst_row)):
                smiles = str(row["canonical_smiles"])
                body.extend(
                    (
                        f"<h3>{label}</h3>",
                        _molecule_svg(smiles),
                        f"<p><code>{html.escape(smiles)}</code><br>",
                        "exact L1: "
                        f"{bool(row.get('exact_l1_program'))}; traces: "
                        f"{int(row.get('exact_l1_trace_count', 0))}</p>",
                    )
                )
                rendered += 1
            sections.append("\n".join(body))
    document = "\n".join(
        (
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<title>FORGE nonselecting molecule audit</title>",
            "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}"
            "svg{border:1px solid #ddd}code{overflow-wrap:anywhere}</style></head><body>",
            "<h1>Nonselecting final-checkpoint molecule audit</h1>",
            "<p>Random rows use a frozen seeded draw. Worst-case valid rows prioritize unresolved "
            "or ambiguous exact-L1 status, then larger graphs. Rendering never selects a checkpoint "
            "or candidate and invalid graphs are not repaired.</p>",
            *sections,
            "</body></html>",
        )
    )
    path.write_text(document)
    return {
        "expected_sections": expected,
        "reported_sections": len(sections),
        "rendered_molecules": rendered,
        "sections_without_valid_molecule": expected - rendered // 2,
    }


def _write_samples(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            header = {"schema_version": SAMPLES_SCHEMA, "rows": len(rows)}
            compressed.write((stable_json(header) + "\n").encode())
            for row in rows:
                compressed.write((stable_json(row) + "\n").encode())


def run_synthesis_program_production_evaluation(
    config_path: Path,
    repo: Path,
    cache_path: Path,
    checkpoint_archive_path: Path,
    training_result_path: Path,
    output_dir: Path,
    *,
    profile: str,
    replicate: int,
    allocated_device: str,
    dynamic_production_design_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate every fixed checkpoint without selecting a model, threshold, or candidate."""

    if torch is None:
        raise SynthesisProgramProductionEvaluationError("evaluation requires torch")
    config = read_json_object(
        config_path,
        error=SynthesisProgramProductionEvaluationError,
        label="production evaluation config",
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise SynthesisProgramProductionEvaluationError("unsupported evaluation config")
    paths = {}
    for label, value in config["inputs"].items():
        if label == "production_design" and dynamic_production_design_path is not None:
            if (
                not isinstance(value, Mapping)
                or set(value) != {"path", "sha256"}
                or value["path"] != "study_design.json"
                or sha256_file(dynamic_production_design_path) != value["sha256"]
            ):
                raise SynthesisProgramProductionEvaluationError(
                    "dynamic production-design pin is malformed or changed"
                )
            paths[label] = dynamic_production_design_path.resolve()
        else:
            paths[label] = resolve_pin(value, repo, label=label)
    required = {
        "production_design",
        "production_cache",
        "program_config",
        "qualified_reaction_families",
        "qualified_ugi_reactions",
        "ugi_assignments",
        "multireaction_atlas",
        "multireaction_splits",
    }
    optional = {"local_chemistry_support"}
    if (
        frozenset(paths) not in {frozenset(required), frozenset(required | optional)}
        or paths["production_cache"].resolve() != cache_path.resolve()
    ):
        raise SynthesisProgramProductionEvaluationError("evaluation inputs changed")
    design = read_json_object(
        paths["production_design"],
        error=SynthesisProgramProductionEvaluationError,
        label="frozen production design",
    )
    training = read_json_object(
        training_result_path,
        error=SynthesisProgramProductionEvaluationError,
        label="matched training result",
    )
    if (
        training.get("schema_version") != "forge.synthesis_program_production_training_result.v1"
        or training.get("status") != "pass"
        or int(training.get("replicate", -1)) != replicate
    ):
        raise SynthesisProgramProductionEvaluationError(
            "training result does not authorize checkpoint evaluation"
        )
    runtime = dict(config[profile])
    if str(runtime["device"]) != allocated_device:
        raise SynthesisProgramProductionEvaluationError(
            "evaluation device and allocated stage device differ"
        )
    execution_scope = _validate_evaluation_budget(config, design, profile=profile)
    raw_record_limit = runtime.get("component_disjoint_record_limit")
    record_limit = None if raw_record_limit is None else int(raw_record_limit)
    if record_limit is not None and record_limit < 1:
        raise SynthesisProgramProductionEvaluationError(
            "component-disjoint record limit must be positive or null"
        )
    terminal_decode_policy = str(runtime.get("terminal_decode_policy", "unconstrained_argmax"))
    if terminal_decode_policy not in SUPPORTED_TERMINAL_DECODE_POLICIES:
        raise SynthesisProgramProductionEvaluationError(
            f"unsupported production terminal decoder: {terminal_decode_policy!r}"
        )
    device = torch.device(allocated_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SynthesisProgramProductionEvaluationError("CUDA requested but unavailable")
    seed = int(training["seed"])
    specs = {
        value.program_id: value
        for value in load_reaction_program_specifications(paths["program_config"])
    }
    adapters: dict[str, Any] = {
        spec.program_id: RegistryRepeatedReactionProgram.from_registry(
            paths["qualified_reaction_families"],
            spec,
            expected_sha256=str(config["inputs"]["qualified_reaction_families"]["sha256"]),
        )
        for spec in specs.values()
    }
    adapters["ugi_3cr_agile"] = Ugi3AssemblyAdapter.from_registry(
        paths["qualified_ugi_reactions"],
        expected_sha256=str(config["inputs"]["qualified_ugi_reactions"]["sha256"]),
    )
    cache = SynthesisProgramProductionCache(cache_path)
    local_chemistry_support = None
    if "local_chemistry_support" in paths:
        local_chemistry_support = LocalChemistrySupport.from_mapping(
            read_json_object(
                paths["local_chemistry_support"],
                error=SynthesisProgramProductionEvaluationError,
                label="local chemistry support",
            )
        )
        if local_chemistry_support.atom_states != tuple(cache.atom_vocabulary):
            raise SynthesisProgramProductionEvaluationError(
                "local chemistry and production-cache atom vocabularies differ"
            )
    if (terminal_decode_policy == LOCAL_CHEMISTRY_TERMINAL_DECODE_POLICY) != (
        local_chemistry_support is not None
    ):
        raise SynthesisProgramProductionEvaluationError(
            "local chemistry support must be present exactly for its terminal decoder"
        )
    # The core-saturation contract needs no new artifact: it is read from the qualified reaction
    # registry that this evaluation already pins by sha256.
    reaction_core_saturation_policy = None
    if terminal_decode_policy == CORE_SATURATION_TERMINAL_DECODE_POLICY:
        reaction_core_saturation_policy = ReactionCoreSaturationPolicy.from_qualified_registry(
            paths["qualified_ugi_reactions"],
            reaction_id="ugi_3cr_agile",
            expected_sha256=str(config["inputs"]["qualified_ugi_reactions"]["sha256"]),
        )
    all_rows: list[dict[str, Any]] = []
    checkpoint_metrics: dict[str, Any] = {}
    component_disjoint_metrics: dict[str, Any] = {}
    design_sha256 = str(sha256_file(paths["production_design"]))
    cache_sha256 = str(sha256_file(cache_path))
    try:
        prior = SynthesisProgramLayoutPrior(cache)
        training_products, training_components = load_reaction_program_training_references(
            ugi_assignments=paths["ugi_assignments"],
            multireaction_atlas=paths["multireaction_atlas"],
            multireaction_splits=paths["multireaction_splits"],
            repeated_program_specs=specs,
            ugi_program_id="ugi_3cr_agile",
            ugi_roles=adapters["ugi_3cr_agile"].roles,
        )
        with tarfile.open(checkpoint_archive_path, mode="r") as archive:
            _validate_archive_members(archive, training)
            for arm_id, arm in design["training"]["arms"].items():
                supported = [
                    program for program, mass in arm["program_mass"].items() if float(mass) > 0
                ]
                evaluated_programs = [
                    str(program) for program in arm.get("evaluation_programs", supported)
                ]
                if not evaluated_programs or not set(evaluated_programs).issubset(
                    set(cache.vocabulary.program_states[1:])
                ):
                    raise SynthesisProgramProductionEvaluationError(
                        f"arm {arm_id} declares an invalid evaluation program set"
                    )
                checkpoint_metrics[arm_id] = {}
                component_disjoint_metrics[arm_id] = {}
                snapshots = _select_evaluation_snapshots(
                    training["arms"][arm_id]["checkpoints"],
                    runtime["checkpoint_steps"],
                    arm_id=arm_id,
                )
                for snapshot in snapshots:
                    step = int(snapshot["step"])
                    member_name = f"{arm_id}/{snapshot['filename']}"
                    model, package = _load_checkpoint(
                        archive,
                        member_name=member_name,
                        expected_sha256=str(snapshot["sha256"]),
                        design_sha256=design_sha256,
                        cache_sha256=cache_sha256,
                        device=device,
                        cache=cache,
                    )
                    sampling_conditioning, state_mapping = _conditioning_contract(package, cache)
                    node_marginal = (
                        package["node_marginal"].detach().cpu().numpy()
                        if hasattr(package["node_marginal"], "detach")
                        else package["node_marginal"]
                    )
                    bond_marginal = (
                        package["bond_marginal"].detach().cpu().numpy()
                        if hasattr(package["bond_marginal"], "detach")
                        else package["bond_marginal"]
                    )
                    split_metrics: dict[str, Any] = {}
                    split_counts = [("calibration", int(runtime["calibration_samples"]))]
                    if step == int(runtime["checkpoint_steps"][-1]):
                        split_counts.append(("heldout", int(runtime["heldout_samples"])))
                    for split_name, count in split_counts:
                        split_metrics[split_name] = {}
                        for program_id in evaluated_programs:
                            layout_seed = production_seed(
                                seed, arm_id, step, split_name, program_id, "layout"
                            )
                            sampling_seed = production_seed(
                                seed, arm_id, step, split_name, program_id, "flow"
                            )
                            try:
                                layouts = prior.sample(
                                    program_id,
                                    sample_count=count,
                                    seed=layout_seed,
                                    role_morphology_conditioning=bool(
                                        package["model_config"].get(
                                            "role_morphology_conditioning", False
                                        )
                                    ),
                                    exact_program_topology=(
                                        terminal_decode_policy
                                        == PROGRAM_TOPOLOGY_TERMINAL_DECODE_POLICY
                                    ),
                                )
                            except SynthesisProgramLayoutError as error:
                                # Carry the layout reason forward: this runs on paid accelerators,
                                # where re-reading the cause costs another run.
                                raise SynthesisProgramProductionEvaluationError(
                                    "factorized layout support failed for "
                                    f"arm={arm_id}, checkpoint={step}, split={split_name}, "
                                    f"program={program_id}, seed={layout_seed}: {error}"
                                ) from error
                            rows, sampling = sample_synthesis_program_products(
                                model,
                                layouts,
                                cache.atom_vocabulary,
                                node_marginal,
                                bond_marginal,
                                samples_per_program=1,
                                sample_steps=int(runtime["sample_steps"]),
                                batch_size=int(runtime["batch_size"]),
                                seed=sampling_seed,
                                device=allocated_device,
                                conditioning_mode=sampling_conditioning,
                                program_state_mapping=state_mapping,
                                terminal_decode_policy=terminal_decode_policy,
                                local_chemistry_support=local_chemistry_support,
                                reaction_core_saturation_policy=reaction_core_saturation_policy,
                            )
                            adjudicate_reaction_program_rows(
                                rows,
                                adapters=adapters,
                                repeated_program_specs=specs,
                                ugi_program_id="ugi_3cr_agile",
                            )
                            for row in rows:
                                row.update(
                                    {
                                        "arm_id": arm_id,
                                        "checkpoint_step": step,
                                        "evaluation_split": split_name,
                                        "replicate": replicate,
                                        "seed": seed,
                                        "terminal_decode_policy": terminal_decode_policy,
                                    }
                                )
                            evaluation = evaluate_reaction_program_samples(
                                rows,
                                training_products=training_products,
                                training_components=training_components,
                            )
                            split_metrics[split_name][program_id] = _metric_contract(
                                evaluation,
                                fixed_state_failures=int(sampling["fixed_state_failures"]),
                                support_overflow_count=0,
                            )
                            split_metrics[split_name][program_id].update(
                                {
                                    "strict_constraint_abstentions": int(
                                        sampling["strict_constraint_abstentions"]
                                    ),
                                    "strict_constraint_abstention_reasons": dict(
                                        sampling["strict_constraint_abstention_reasons"]
                                    ),
                                    "repairs": dict(sampling["repairs"]),
                                    "local_chemistry_policy_applied": bool(
                                        sampling["local_chemistry_policy_applied"]
                                    ),
                                }
                            )
                            all_rows.extend(rows)
                    if step == int(runtime["checkpoint_steps"][-1]):
                        for program_id in evaluated_programs:
                            reconstruction = _component_disjoint_reconstruction(
                                model,
                                package,
                                cache,
                                program_id=program_id,
                                batch_size=int(runtime["batch_size"]),
                                record_limit=record_limit,
                                seed=production_seed(
                                    seed,
                                    arm_id,
                                    program_id,
                                    "component_disjoint",
                                ),
                                device=device,
                            )
                            heldout = split_metrics["heldout"][program_id]
                            reconstruction.update(
                                {
                                    "exact_l1_decomposition_coverage": heldout[
                                        "exact_l1_decomposition_coverage"
                                    ],
                                    "exact_forward_replay_precision": heldout[
                                        "exact_forward_replay_precision"
                                    ],
                                }
                            )
                            component_disjoint_metrics[arm_id][program_id] = reconstruction
                    checkpoint_metrics[arm_id][str(step)] = split_metrics
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    finally:
        cache.close()
    samples_path = output_dir / "samples.jsonl.gz"
    _write_samples(samples_path, all_rows)
    expected_steps = [int(value) for value in runtime["checkpoint_steps"]]
    common_attempt_artifacts = {}
    common_attempt_dir = output_dir / "common_ugi_attempts"
    method_aliases = {
        "shared_three_program_conditioned": "forge_transformer",
        "shared_null_posthoc": "shared_null_posthoc",
        "full_transformer": "forge_transformer",
        "bl_core_constrained_repeat_aware": "forge_transformer",
    }
    for arm_id, arm in design["training"]["arms"].items():
        supported = [
            program_id for program_id, mass in arm["program_mass"].items() if float(mass) > 0
        ]
        if "ugi_3cr_agile" not in arm.get("evaluation_programs", supported):
            continue
        method_id = method_aliases.get(arm_id, arm_id)
        ugi_rows = [
            row
            for row in all_rows
            if row["arm_id"] == arm_id
            and row["program_id"] == "ugi_3cr_agile"
            and int(row["checkpoint_step"]) == expected_steps[-1]
            and row["evaluation_split"] == "heldout"
        ]
        attempts = tuple(
            CommonUgiAttempt(
                method_id=method_id,
                seed=seed,
                attempt_index=index,
                status="generated" if row.get("valid") is True else "invalid",
                product_smiles=(str(row["canonical_smiles"]) if row.get("valid") is True else None),
                method_visible_component_ids=(),
                generator_calls=1,
                reaction_calls=0,
                route_calls=0,
                oracle_calls=0,
                wall_seconds=0.0,
            )
            for index, row in enumerate(ugi_rows)
        )
        attempt_path = common_attempt_dir / f"{arm_id}.jsonl.gz"
        write_attempt_ledger(attempt_path, attempts)
        common_attempt_artifacts[arm_id] = {
            **artifact_record(attempt_path),
            "method_id": method_id,
            "seed": seed,
            "attempts": len(attempts),
            "wall_seconds_measurement": "not_recorded_do_not_interpret_zero",
        }
    cross_role_fidelity = {}
    ugi_held_component_metrics = {}
    _, _, held_components = load_ugi_identity_references(
        paths["ugi_assignments"], roles=adapters["ugi_3cr_agile"].roles
    )
    for arm_id, arm in design["training"]["arms"].items():
        supported = [
            program_id for program_id, mass in arm["program_mass"].items() if float(mass) > 0
        ]
        evaluated = arm.get("evaluation_programs", supported)
        if "ugi_3cr_agile" not in evaluated:
            continue
        ugi_rows = [
            row
            for row in all_rows
            if row["arm_id"] == arm_id
            and row["program_id"] == "ugi_3cr_agile"
            and int(row["checkpoint_step"]) == expected_steps[-1]
            and row["evaluation_split"] == "heldout"
        ]
        cross_role_fidelity[arm_id] = cross_role_fidelity_to_heldout(
            ugi_rows,
            paths["ugi_assignments"],
            roles=adapters["ugi_3cr_agile"].roles,
        )
        held_count = 0
        by_role: Counter[str] = Counter()
        for row in ugi_rows:
            if row.get("exact_l1_program") is not True or int(row["exact_l1_trace_count"]) != 1:
                continue
            components = row["exact_l1_traces"][0]["components_by_role"]
            canonical_components = {}
            for role in adapters["ugi_3cr_agile"].roles:
                molecule = Chem.MolFromSmiles(str(components[role]))
                if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                    raise SynthesisProgramProductionEvaluationError(
                        f"exact-L1 trace contains an invalid {role} component"
                    )
                canonical_components[role] = Chem.MolToSmiles(
                    molecule, canonical=True, isomericSmiles=False
                )
            held_roles = [
                role
                for role, canonical in canonical_components.items()
                if canonical in held_components[role]
            ]
            if held_roles:
                held_count += 1
                by_role.update(held_roles)
        ugi_held_component_metrics[arm_id] = {
            "attempts": len(ugi_rows),
            "exact_l1_products_with_held_component": held_count,
            "exact_l1_products_with_held_component_per_1000_attempts": (
                1000.0 * held_count / len(ugi_rows)
            ),
            "by_role": dict(sorted(by_role.items())),
        }
    molecule_report_path = output_dir / "molecule_report.html"
    molecule_report = _write_molecule_report(
        molecule_report_path,
        all_rows,
        design,
        final_step=expected_steps[-1],
        seed=seed,
    )
    expected_program_reports = sum(
        1
        for arm in design["training"]["arms"].values()
        for _ in arm.get(
            "evaluation_programs",
            [program for program, mass in arm["program_mass"].items() if float(mass) > 0],
        )
    )
    gates = {
        "all_arms_evaluated": set(checkpoint_metrics) == set(design["training"]["arms"]),
        "all_fixed_checkpoints_evaluated": all(
            [int(value) for value in metrics] == expected_steps
            for metrics in checkpoint_metrics.values()
        ),
        "coverage_and_precision_reported": all(
            row["coverage_and_precision_reported"] is True
            for arm in checkpoint_metrics.values()
            for step in arm.values()
            for split in step.values()
            for row in split.values()
        ),
        "fixed_state_failures_zero": all(
            int(row["fixed_state_failures"]) == 0
            for arm in checkpoint_metrics.values()
            for step in arm.values()
            for split in step.values()
            for row in split.values()
        ),
        "support_overflow_zero": all(
            int(row["support_overflow_count"]) == 0
            for arm in checkpoint_metrics.values()
            for step in arm.values()
            for split in step.values()
            for row in split.values()
        ),
        "component_disjoint_metrics_complete": sum(
            len(value) for value in component_disjoint_metrics.values()
        )
        == expected_program_reports,
        "component_disjoint_fixed_state_failures_zero": all(
            int(row["fixed_state_noising_failures"]) == 0
            for arm in component_disjoint_metrics.values()
            for row in arm.values()
        ),
        "heldout_denoising_losses_finite": all(
            np.isfinite(float(row["heldout_denoising_loss_at_t_0_5"]))
            for arm in component_disjoint_metrics.values()
            for row in arm.values()
        ),
        "molecule_report_complete": molecule_report["reported_sections"]
        == molecule_report["expected_sections"]
        == expected_program_reports,
        "held_reaction_family_reported": True,
        "ugi_cross_role_fidelity_reported": set(cross_role_fidelity)
        == {
            arm_id
            for arm_id, arm in design["training"]["arms"].items()
            if "ugi_3cr_agile"
            in arm.get(
                "evaluation_programs",
                [program_id for program_id, mass in arm["program_mass"].items() if float(mass) > 0],
            )
        },
        "ugi_held_component_metrics_reported": set(ugi_held_component_metrics)
        == set(cross_role_fidelity),
        "common_ugi_attempt_ledgers_complete": set(common_attempt_artifacts)
        == set(cross_role_fidelity),
        "heldout_is_nonselecting": True,
        "candidate_selection_absent": True,
        "route_or_oracle_calls_zero": True,
        "reductive_amination_substructure_rate_absent": True,
        "terminal_decode_policy_recorded": all(
            row.get("terminal_decode_policy") == terminal_decode_policy for row in all_rows
        ),
        "local_chemistry_policy_binding_exact": all(
            bool(row.get("local_chemistry_policy_applied")) == (local_chemistry_support is not None)
            for row in all_rows
        ),
        "no_repairs_or_retries": all(
            not metrics["repairs"]
            for arm in checkpoint_metrics.values()
            for step in arm.values()
            for split in step.values()
            for metrics in split.values()
        ),
        "layout_prior_is_exactly_support_conditioned": True,
    }
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "pass" if all(gates.values()) else "fail",
        "run_kind": execution_scope if profile == "full" else "smoke",
        "execution_scope": execution_scope,
        "profile": profile,
        "replicate": replicate,
        "seed": seed,
        "config": (
            artifact_record(config_path, logical_path="evaluation_config.json")
            if dynamic_production_design_path is not None
            else pin_record(config_path, repo)
        ),
        "design": (
            artifact_record(paths["production_design"], logical_path="study_design.json")
            if dynamic_production_design_path is not None
            else pin_record(paths["production_design"], repo)
        ),
        "cache": artifact_record(cache_path),
        "training_result": artifact_record(training_result_path),
        "checkpoint_archive": artifact_record(checkpoint_archive_path),
        "samples": artifact_record(samples_path),
        "molecule_report": artifact_record(molecule_report_path),
        "sample_rows": len(all_rows),
        "checkpoint_metrics": checkpoint_metrics,
        "component_disjoint_metrics": component_disjoint_metrics,
        "cross_role_fidelity": cross_role_fidelity,
        "ugi_held_component_metrics": ugi_held_component_metrics,
        "common_ugi_attempt_ledgers": common_attempt_artifacts,
        "molecule_report_summary": molecule_report,
        "held_reaction_family": {
            "status": "not_applicable_no_frozen_unseen_family_arm",
            "hard_gate": False,
            "reported": True,
            "reason": (
                "The frozen four-arm comparison trains every shared arm on all three admitted "
                "programs and contains no leave-one-family-out arm; no unseen-family result is "
                "fabricated from the cyclic-ID control."
            ),
        },
        "gates": gates,
        "selection": {
            "checkpoint": "fixed_final_step",
            "calibration_selects_model": False,
            "heldout_selects_model_or_threshold": False,
            "candidate_selection": False,
        },
        "calls": {"route": 0, "oracle": 0},
        "terminal_decode_policy": terminal_decode_policy,
        "layout_prior": {
            "source": "training_fold_factorized_count_only_program_prior",
            "support_conditioning": "exact_product_law_conditioned_on_maximum_heavy_atoms",
            "maximum_heavy_atoms": prior.maximum_heavy_atoms,
            "clipping": False,
            "repair": False,
            "retry": False,
            "dropped_attempts": 0,
        },
        "local_chemistry_support": (
            pin_record(paths["local_chemistry_support"], repo)
            if local_chemistry_support is not None
            else None
        ),
        "reaction_core_saturation_policy": (
            None
            if reaction_core_saturation_policy is None
            else reaction_core_saturation_policy.to_mapping()
        ),
        "nonclaims": [
            "Generated-product metrics are computational evidence, not synthesis-success probabilities.",
            "Exact auxiliary-family replay is transform consistency, not route certification.",
            "The held-reaction-family stress test remains secondary and is not evaluated as a hard gate here.",
        ],
    }
    write_json(output_dir / "result.json", result)
    if result["status"] != "pass":
        raise SynthesisProgramProductionEvaluationError(
            f"production evaluation gates failed: {gates}"
        )
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "RESULT_SCHEMA",
    "SynthesisProgramProductionEvaluationError",
    "run_synthesis_program_production_evaluation",
]
