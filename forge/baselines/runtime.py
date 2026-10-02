"""Execute a frozen external Ugi request in the method's native environment.

Use ``forge baseline native`` with the pinned upstream checkout. Third-party
packages load only inside their adapter. Compatibility edits apply to a copy,
leaving the original checkout available for commit verification.
"""

from __future__ import annotations

import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.baselines.contract import load_external_baseline_manifest, verify_external_checkout
from forge.baselines.requests import REQUEST_SCHEMA, NativeBaselinePortError
from forge.core.hashing import artifact_record, sha256_file
from forge.core.io import atomic_write, read_csv, read_json_object, write_csv, write_json


def _verify_request(request_path: Path) -> dict[str, Any]:
    request = read_json_object(
        request_path, error=NativeBaselinePortError, label="external native request"
    )
    required = {
        "schema_version",
        "method",
        "checkout",
        "profile",
        "seed",
        "requested_attempts",
        "parameters",
        "inputs",
        "common_export_result",
        "common_export_sampling_measure",
        "output_contract",
        "call_budget",
        "nonclaims",
    }
    if set(request) != required or request["schema_version"] != REQUEST_SCHEMA:
        raise NativeBaselinePortError("external native request schema changed")
    base = request_path.parent
    inputs = request["inputs"]
    if not isinstance(inputs, Mapping) or not inputs:
        raise NativeBaselinePortError("external native request has no inputs")
    for filename, record in inputs.items():
        path = base / "inputs" / str(filename)
        if (
            not path.is_file()
            or not isinstance(record, Mapping)
            or record.get("logical_path") != f"inputs/{filename}"
            or record.get("sha256") != str(sha256_file(path))
            or record.get("bytes") != path.stat().st_size
        ):
            raise NativeBaselinePortError(f"native request input changed: {filename}")
    return request


def _replace_once(path: Path, before: str, after: str) -> None:
    text = path.read_text()
    if text.count(before) != 1:
        raise NativeBaselinePortError(f"reviewed upstream edit no longer applies once: {path}")
    atomic_write(path, text.replace(before, after).encode())


def _replace_exact_count(path: Path, before: str, after: str, *, expected: int) -> None:
    text = path.read_text()
    observed = text.count(before)
    if observed != expected:
        raise NativeBaselinePortError(
            f"reviewed upstream edit count changed for {path}: expected {expected}, found {observed}"
        )
    atomic_write(path, text.replace(before, after).encode())


def _run_checked(argv: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None) -> None:
    try:
        subprocess.run(list(argv), cwd=cwd, env=dict(env) if env else None, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise NativeBaselinePortError(f"native command failed: {list(argv)}") from error


def _genmol_sample_row(attempt_index: int, decoded_smiles: str | None) -> dict[str, Any]:
    """Map one strict SAFE decode to the native attempt contract.

    SAFE can decode a token sequence to the empty string without raising.  An empty string is not a
    generated molecule, so its attempt must remain in the denominator as invalid.
    """

    generated = isinstance(decoded_smiles, str) and bool(decoded_smiles)
    return {
        "attempt_index": attempt_index,
        "status": "generated" if generated else "invalid",
        "product_smiles": decoded_smiles if generated else "",
    }


def _stage_checkout(checkout: Path, output_dir: Path) -> Path:
    staged = output_dir / "upstream"
    shutil.copytree(checkout, staged, ignore=shutil.ignore_patterns(".git", "__pycache__"))
    return staged


def _run_rgfn(
    request: Mapping[str, Any], request_dir: Path, checkout: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    try:
        import gin
        import pandas as pd
        import torch
        from rgfn.api.proxy_base import ProxyBase, ProxyOutput
        from rgfn.gfns.reaction_gfn.api.reaction_api import (
            ReactionActionC,
            ReactionStateTerminal,
        )
        from rgfn.shared.samplers.random_sampler import RandomSampler
        from rgfn.trainer.logger.logger_base import LoggerBase
        from rgfn.trainer.trainer import Trainer
        from rgfn.utils.helpers import seed_everything
    except ImportError as error:
        raise NativeBaselinePortError(
            "RGFN native dependencies are unavailable; run from the pinned RGFN environment"
        ) from error

    class ConstantReactionProxy(ProxyBase):
        def __init__(self):
            self.device = "cpu"

        @property
        def is_non_negative(self) -> bool:
            return True

        @property
        def higher_is_better(self) -> bool:
            return True

        def compute_proxy_output(self, states):
            return ProxyOutput(
                value=torch.ones(len(states), dtype=torch.float32, device=self.device)
            )

    class NoOpLogger(LoggerBase):
        """Complete upstream logger interface without network or experiment selection."""

        def __init__(self):
            super().__init__(output_dir / "work" / "logs")

        def log_metrics(self, metrics, prefix):
            return None

        def log_code(self, source_path):
            return None

        def log_to_file(self, content, name, type="txt"):
            return None

        def log_config(self, config):
            return None

        def log_files(self, file_paths):
            return None

        def close(self):
            return None

        def restart(self):
            return None

    gin.external_configurable(ConstantReactionProxy, module="forge_native")
    gin.external_configurable(NoOpLogger, module="forge_native")
    work = output_dir / "work"
    work.mkdir()
    components = read_csv(request_dir / "inputs" / "train_components.csv.gz")
    pd.DataFrame(
        {"SMILES": sorted({row["canonical_component_smiles"] for row in components})}
    ).to_csv(work / "fragments.csv", index=False)
    reaction_export = read_json_object(
        request_dir / "inputs" / "ugi_reaction.json",
        error=NativeBaselinePortError,
        label="qualified Ugi reaction export",
    )
    reaction = reaction_export.get("reaction")
    if not isinstance(reaction, Mapping) or not isinstance(
        reaction.get("atom_mapped_reaction_smarts"), str
    ):
        raise NativeBaselinePortError("qualified Ugi reaction export changed")
    with pd.ExcelWriter(work / "chemistry.xlsx") as writer:
        pd.DataFrame({"Reaction": [reaction["atom_mapped_reaction_smarts"]]}).to_excel(
            writer, sheet_name="Reactions_NoDocking", index=False
        )
    parameters = request["parameters"]
    run_dir = work / "run"
    config = work / "forge_rgfn.gin"
    config_text = f"""
import gin_config
import rgfn
include 'configs/rgfn_base.gin'
include 'configs/envs/reaction.gin'

proxy/gin.singleton.constructor = @forge_native.ConstantReactionProxy
train_proxy = @proxy/gin.singleton()
valid_proxy = %train_proxy
include 'configs/rewards/linear.gin'

hidden_dim = 64
include 'configs/policies/action_embeddings/one_hot.gin'
include 'configs/policies/reaction.gin'
include 'configs/policies/exploration/uniform.gin'
include 'configs/objectives/trajectory_balance.gin'

ReactionDataFactory.reaction_path = {str(work / "chemistry.xlsx")!r}
ReactionDataFactory.fragment_path = {str(work / "fragments.csv")!r}
ReactionDataFactory.docking = False
ReactionEnv.max_num_reactions = 1
Reward.reward_boosting = 'linear'
Reward.min_reward = 1.0
Reward.beta = 1.0

Trainer.run_dir = {str(run_dir)!r}
Trainer.logger = @forge_native.NoOpLogger()
Trainer.train_forward_sampler = %train_forward_sampler
Trainer.train_backward_sampler = None
Trainer.train_replay_buffer = None
Trainer.train_forward_n_trajectories = {int(parameters["trajectories_per_iteration"])}
Trainer.train_backward_n_trajectories = 0
Trainer.train_replay_n_trajectories = 0
Trainer.train_batch_size = {int(parameters["batch_size"])}
Trainer.train_metrics = []
Trainer.train_artifacts = []
Trainer.valid_sampler = None
Trainer.valid_metrics = []
Trainer.valid_artifacts = []
Trainer.objective = %objective
Trainer.optimizer = @TrajectoryBalanceOptimizer()
Trainer.lr_scheduler = None
Trainer.n_iterations = {int(parameters["training_iterations"])}
Trainer.checkpoint_mode = 'last'
Trainer.best_metric = 'loss'
Trainer.device = 'cuda'
""".strip()
    atomic_write(config, f"{config_text}\n".encode())
    seed = int(request["seed"])
    seed_everything(seed)
    torch.use_deterministic_algorithms(True)
    old_cwd = Path.cwd()
    try:
        os.chdir(checkout)
        gin.clear_config()
        gin.parse_config_file(str(config))
        trainer = Trainer()
        trainer.train()
        trainer.objective.forward_policy.eval()
        sampler = RandomSampler(
            policy=trainer.objective.forward_policy,
            env=trainer.train_forward_sampler.env,
            reward=None,
        )
        rows: list[dict[str, Any]] = []
        reaction_calls = 0
        next_index = 0
        for trajectories in sampler.get_trajectories_iterator(
            int(request["requested_attempts"]), int(parameters["batch_size"])
        ):
            actions_by_trajectory = trajectories._actions_list
            for state, actions in zip(trajectories.get_last_states_flat(), actions_by_trajectory):
                reaction_calls += sum(isinstance(action, ReactionActionC) for action in actions)
                complete = isinstance(state, ReactionStateTerminal) and state.num_reactions == 1
                rows.append(
                    {
                        "attempt_index": next_index,
                        "status": "generated" if complete else "failed",
                        "product_smiles": state.molecule.smiles if complete else "",
                    }
                )
                next_index += 1
        trainer.close()
    finally:
        os.chdir(old_cwd)
        gin.clear_config()
    checkpoint = output_dir / "checkpoint.pt"
    shutil.copyfile(run_dir / "train" / "checkpoints" / "last_gfn.pt", checkpoint)
    return (
        rows,
        reaction_calls,
        {
            "adapter": "rgfn_exact_ugi_three_reactant_constant_reward_v1",
            "checkpoint": artifact_record(checkpoint),
            "visible_components": len(components),
            "compatibility_edits": [
                "supply the complete no-op logger interface omitted by the pinned DummyLogger"
            ],
        },
    )


def _decode_defog_graph(atom_types: Any, edge_types: Any) -> str | None:
    try:
        import torch
        from rdkit import Chem
    except ImportError as error:
        raise NativeBaselinePortError("DeFoG decode requires torch and RDKit") from error
    atom_decoder = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
    bond_decoder = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    molecule = Chem.RWMol()
    try:
        for atom in atom_types:
            molecule.AddAtom(Chem.Atom(atom_decoder[int(atom.item())]))
        upper = torch.triu(edge_types)
        for start, end in torch.nonzero(upper):
            start_i, end_i = int(start.item()), int(end.item())
            bond = int(upper[start_i, end_i].item())
            if start_i != end_i and bond in bond_decoder:
                molecule.AddBond(start_i, end_i, bond_decoder[bond])
        built = molecule.GetMol()
        Chem.SanitizeMol(built)
        return Chem.MolToSmiles(built, canonical=True, isomericSmiles=False)
    except (IndexError, RuntimeError, ValueError):
        return None


def _run_defog(
    request: Mapping[str, Any], request_dir: Path, checkout: Path, output_dir: Path
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    staged = _stage_checkout(checkout, output_dir)
    main_path = staged / "src" / "main.py"
    dataset_path = staged / "src" / "datasets" / "guacamol_dataset.py"
    model_path = staged / "src" / "graph_discrete_flow_model.py"
    # graph-tool is imported unconditionally by upstream but is used only by the non-molecular
    # Spectre branches. The retained GuacaMol branch never references it, so remove the unused
    # import rather than adding an unrelated graph-analysis runtime to the molecular baseline.
    _replace_once(main_path, "import graph_tool\n", "")
    _replace_once(
        main_path,
        "dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)",
        "dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg, recompute_statistics=True)",
    )
    _replace_once(
        main_path,
        "        logger=[],\n    )",
        "        logger=[],\n        deterministic=True,\n    )",
    )
    _replace_once(
        main_path,
        "        max_epochs=cfg.train.n_epochs,",
        "        max_epochs=cfg.train.n_epochs,\n"
        '        max_steps=cfg.train.get("max_steps", -1),\n'
        '        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),',
    )
    _replace_once(
        main_path,
        '        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),',
        '        limit_train_batches=cfg.train.get("limit_train_batches", 1.0),\n'
        '        limit_val_batches=cfg.train.get("limit_val_batches", 1.0),\n'
        '        limit_test_batches=cfg.train.get("limit_test_batches", 1.0),',
    )
    _replace_once(
        main_path,
        "            save_top_k=-1,",
        "            save_top_k=-1,\n            save_last=True,",
    )
    _replace_once(
        main_path,
        "    dataset_infos.compute_reference_metrics(\n"
        "        datamodule=datamodule,\n"
        "        sampling_metrics=sampling_metrics,\n"
        "    )",
        # FCD is explicitly disabled and the common FORGE assessor computes every retained
        # paper metric.  Upstream otherwise performs multiple non-training RDKit passes over
        # all 66,464 training graphs before fitting, including partial-charge repair that is
        # outside the raw-generation contract.  Empty reference mappings preserve the runtime
        # interface; no training loss, graph decoder, checkpoint, or generated attempt changes.
        '    dataset_infos.ref_metrics = {"val": {}, "test": {}}',
    )
    _replace_once(
        dataset_path,
        "            valencies = datamodule.valency_count()",
        "            self.complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)\n"
        "            valencies = datamodule.valency_count(self.max_n_nodes)",
    )
    _replace_once(
        dataset_path,
        "self.data, self.slices = torch.load(self.processed_paths[self.file_idx])",
        "self.data, self.slices = torch.load("
        "self.processed_paths[self.file_idx], weights_only=False)",
    )
    _replace_once(
        model_path,
        "            bs = 2 * self.cfg.train.batch_size",
        "            bs = self.cfg.general.sampling_batch_size",
    )
    _replace_once(
        model_path,
        "            samples, labels = self.sample(\n"
        "                is_test=True,\n"
        "                save_samples=self.cfg.general.save_samples,\n"
        "                save_visualization=True,\n"
        "            )\n"
        "            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)\n"
        "\n"
        "            # Store results\n"
        "            filename = os.path.join(\n"
        "                os.getcwd(),\n"
        '                f"test_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",\n'
        "            )\n"
        '            with open(filename, "w") as file:\n'
        "                for key, value in to_log.items():\n"
        '                    file.write(f"{key}: {value}\\n")',
        "            self.sample(\n"
        "                is_test=True,\n"
        "                save_samples=self.cfg.general.save_samples,\n"
        "                save_visualization=True,\n"
        "            )",
    )
    raw = staged / "data" / "forge_ugi" / "raw"
    raw.mkdir(parents=True)
    for source, destination in (
        ("train.smi", "guacamol_v1_train.smiles"),
        ("calibration.smi", "guacamol_v1_valid.smiles"),
        ("heldout.smi", "guacamol_v1_test.smiles"),
    ):
        shutil.copyfile(request_dir / "inputs" / source, raw / destination)
    parameters = request["parameters"]
    seed = int(request["seed"])
    work = output_dir / "work"
    train_run = work / "train"
    common = [
        sys.executable,
        "src/main.py",
        "+experiment=guacamol",
        "dataset=guacamol",
        "dataset.datadir=data/forge_ugi",
        "dataset.filter=false",
        "dataset.compute_fcd=false",
        "general.name=forge_ugi",
        "general.wandb=disabled",
        "general.gpus=1",
        "general.sample_every_val=1",
        "general.check_val_every_n_epochs=1",
        "general.samples_to_generate=4",
        "general.samples_to_save=0",
        "general.chains_to_save=0",
        f"train.seed={seed}",
        f"train.batch_size={int(parameters['batch_size'])}",
        "train.num_workers=0",
        f"train.n_epochs={int(parameters['training_epochs'])}",
        f"+train.max_steps={int(parameters['training_steps'])}",
        f"+train.limit_val_batches={int(parameters['validation_batches'])}",
        f"+train.limit_test_batches={int(parameters['test_batches'])}",
        f"sample.sample_steps={int(parameters['sampling_steps'])}",
        "sample.time_distortion=identity",
        "sample.eta=0.0",
        "sample.omega=0.0",
        f"+general.sampling_batch_size={int(parameters['sampling_batch_size'])}",
        f"hydra.run.dir={train_run}",
    ]
    if "training_batches" in parameters:
        common.insert(
            -1,
            f"+train.limit_train_batches={int(parameters['training_batches'])}",
        )
    defog_env = dict(os.environ)
    # PyTorch >=2.6 defaults unspecified checkpoint loads to ``weights_only=True`` while the
    # pinned Lightning release serializes its OmegaConf training state.  The only checkpoint
    # loaded here was created moments earlier by this exact request and is hash-retained below.
    defog_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    defog_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(staged),
            str(staged / "src"),
            *([value] if (value := defog_env.get("PYTHONPATH")) else []),
        ]
    )
    _run_checked(common, cwd=staged, env=defog_env)
    checkpoints = sorted((train_run / "checkpoints" / "forge_ugi").glob("*.ckpt"))
    if not checkpoints:
        raise NativeBaselinePortError("DeFoG fixed-final checkpoint was not written")
    checkpoint = output_dir / "checkpoint.ckpt"
    shutil.copyfile(checkpoints[-1], checkpoint)
    test_run = work / "test"
    test = [
        *(argument for argument in common[:-1] if not argument.startswith("train.n_epochs=")),
        "train.n_epochs=1",
        f"general.test_only={checkpoint}",
        f"general.final_model_samples_to_generate={int(request['requested_attempts'])}",
        "general.final_model_samples_to_save=0",
        "general.final_model_chains_to_save=0",
        "general.num_sample_fold=1",
        "general.save_samples=true",
        f"hydra.run.dir={test_run}",
    ]
    _run_checked(test, cwd=staged, env=defog_env)
    pickles = sorted(test_run.glob("generated_samples_rank*.pkl"))
    if len(pickles) != 1:
        raise NativeBaselinePortError("DeFoG did not write exactly one raw generated-graph ledger")
    with pickles[0].open("rb") as handle:
        generated = pickle.load(handle)
    if not isinstance(generated, list) or len(generated) != int(request["requested_attempts"]):
        raise NativeBaselinePortError("DeFoG raw generation count changed")
    rows = []
    for index, graph in enumerate(generated):
        smiles = (
            _decode_defog_graph(graph[0], graph[1])
            if isinstance(graph, (list, tuple)) and len(graph) == 2
            else None
        )
        rows.append(
            {
                "attempt_index": index,
                "status": "generated" if smiles is not None else "invalid",
                "product_smiles": smiles or "",
            }
        )
    return (
        rows,
        0,
        {
            "adapter": "defog_custom_ugi_dataset_direct_decode_v1",
            "checkpoint": artifact_record(checkpoint),
            "compatibility_edits": [
                "remove unused graph-tool import from the molecular-only execution path",
                "recompute custom dataset statistics and 194-atom support",
                "load the same-run generated PyG dataset under modern Torch trusted-data semantics",
                "load only the trusted same-run Lightning checkpoint with full training state",
                "elide non-training upstream reference and sample reports because common assessment is external",
                "bound the fixed-final optimizer-step budget and smoke-only training batches",
                "retain one no-op test batch only to trigger native raw-graph sampling",
                "use the request-pinned raw sampling batch size",
                "enable deterministic Lightning execution",
                "decode raw graphs without partial-charge repair",
            ],
        },
    )


def _directory_digest(root: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(str(sha256_file(path))))
    return digest.hexdigest()


def _run_genmol(
    request: Mapping[str, Any],
    request_dir: Path,
    checkout: Path,
    output_dir: Path,
    *,
    tokenizer_snapshot: Path | None,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    staged = _stage_checkout(checkout, output_dir)
    parameters = request["parameters"]
    revision = str(parameters["safe_tokenizer_revision"])
    if tokenizer_snapshot is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise NativeBaselinePortError(
                "GenMol requires huggingface_hub or --tokenizer-snapshot for the pinned tokenizer"
            ) from error
        tokenizer_snapshot = Path(
            snapshot_download(repo_id="datamol-io/safe-gpt", revision=revision)
        )
    if not tokenizer_snapshot.is_dir():
        raise NativeBaselinePortError(
            f"pinned SAFE tokenizer snapshot is missing: {tokenizer_snapshot}"
        )
    tokenizer_commit = tokenizer_snapshot.name
    if len(tokenizer_commit) == 40 and tokenizer_commit != revision:
        raise NativeBaselinePortError(
            f"SAFE tokenizer revision changed: expected {revision}, found {tokenizer_commit}"
        )
    data_module = staged / "src" / "genmol" / "utils" / "utils_data.py"
    _replace_once(
        data_module,
        "SAFETokenizer.from_pretrained('datamol-io/safe-gpt').get_pretrained()",
        f"SAFETokenizer.from_pretrained({str(tokenizer_snapshot)!r}).get_pretrained()",
    )
    _replace_exact_count(
        data_module,
        "persistent_workers=True)",
        "persistent_workers=config.loader.num_workers > 0)",
        expected=2,
    )
    try:
        import safe as sf
        import torch
        from rdkit import Chem
        from safe.tokenizer import SAFETokenizer
    except ImportError as error:
        raise NativeBaselinePortError(
            "GenMol native dependencies are unavailable; run from the pinned GenMol environment"
        ) from error
    train_smiles = [
        line.strip()
        for line in (request_dir / "inputs" / "train.smi").read_text().splitlines()
        if line.strip()
    ]
    converter = sf.SAFEConverter(ignore_stereo=True)
    safe_rows = []
    for smiles in train_smiles:
        try:
            encoded = converter.encoder(smiles, allow_empty=False)
        except Exception as error:
            raise NativeBaselinePortError(
                f"SAFE preprocessing failed for training graph: {smiles}"
            ) from error
        if not encoded:
            raise NativeBaselinePortError("SAFE preprocessing emitted an empty training record")
        safe_rows.append(encoded)
    work = output_dir / "work"
    work.mkdir()
    safe_path = work / "train.safe"
    atomic_write(safe_path, "".join(f"{row}\n" for row in safe_rows).encode())
    tokenizer = SAFETokenizer.from_pretrained(str(tokenizer_snapshot)).get_pretrained()
    lengths = [len(tokenizer(row)["input_ids"]) for row in safe_rows]
    max_position = max(256, max(lengths))
    if max_position > 2048:
        raise NativeBaselinePortError(
            f"SAFE serialization exceeds reviewed positional support: {max_position} > 2048"
        )
    (staged / "data").mkdir(exist_ok=True)
    with (staged / "data" / "len.pk").open("wb") as handle:
        pickle.dump(lengths, handle)
    checkpoint_dir = work / "checkpoints"
    train_run = work / "train"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(staged / "src"), value] if (value := env.get("PYTHONPATH")) else [str(staged / "src")]
    )
    command = [
        sys.executable,
        "scripts/train.py",
        f"data={safe_path}",
        f"trainer.max_steps={int(parameters['training_steps'])}",
        "trainer.accelerator=cuda",
        "trainer.devices=1",
        "trainer.num_nodes=1",
        "trainer.precision=32-true",
        "+trainer.deterministic=true",
        f"loader.global_batch_size={int(parameters['global_batch_size'])}",
        f"loader.batch_size={int(parameters['global_batch_size'])}",
        "loader.num_workers=0",
        "loader.pin_memory=false",
        f"model.max_position_embeddings={max_position}",
        f"callback.dirpath={checkpoint_dir}",
        f"callback.every_n_train_steps={int(parameters['training_steps'])}",
        f"hydra.run.dir={train_run}",
        "wandb.name=null",
    ]
    _run_checked(command, cwd=staged, env=env)
    checkpoints = sorted(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        raise NativeBaselinePortError("GenMol fixed-final checkpoint was not written")
    checkpoint = output_dir / "checkpoint.ckpt"
    shutil.copyfile(checkpoints[-1], checkpoint)
    old_path = list(sys.path)
    old_cwd = Path.cwd()
    try:
        sys.path.insert(0, str(staged / "src"))
        os.chdir(staged)
        from genmol.sampler import Sampler

        random.seed(int(request["seed"]))
        torch.manual_seed(int(request["seed"]))
        torch.cuda.manual_seed_all(int(request["seed"]))
        torch.use_deterministic_algorithms(True)
        sampler = Sampler(str(checkpoint))
        sampler.model.to("cuda")
        sampler.mdlm.to_device(sampler.model.device)
        rows = []
        batch_size = int(parameters["sampling_batch_size"])
        attempts = int(request["requested_attempts"])
        for start in range(0, attempts, batch_size):
            count = min(batch_size, attempts - start)
            x = torch.hstack(
                [
                    torch.full((1, 1), sampler.model.bos_index),
                    torch.full((1, 1), sampler.model.eos_index),
                ]
            )
            x = sampler._insert_mask(x, count, min_add_len=1).to(sampler.model.device)
            attention_mask = x != sampler.pad_index
            steps = max(sampler.mdlm.get_num_steps_confidence(x), 2)
            for step in range(steps):
                logits = sampler.model(x, attention_mask)
                x = sampler.mdlm.step_confidence(logits, x, step, steps, 0.8, 0.5)
            raw_safe = sampler.model.tokenizer.batch_decode(x, skip_special_tokens=True)
            for offset, encoded in enumerate(raw_safe):
                decoded_smiles: str | None = None
                try:
                    decoded = sf.decode(encoded, canonical=True, ignore_errors=False)
                    molecule = Chem.MolFromSmiles(decoded)
                    if molecule is not None:
                        Chem.SanitizeMol(molecule)
                        decoded_smiles = Chem.MolToSmiles(
                            molecule, canonical=True, isomericSmiles=False
                        )
                except Exception:
                    decoded_smiles = None
                rows.append(_genmol_sample_row(start + offset, decoded_smiles))
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
    return (
        rows,
        0,
        {
            "adapter": "genmol_train_from_scratch_strict_safe_decode_v1",
            "checkpoint": artifact_record(checkpoint),
            "tokenizer": {
                "repository": "datamol-io/safe-gpt",
                "revision": revision,
                "tree_sha256": _directory_digest(tokenizer_snapshot),
            },
            "training_records": len(safe_rows),
            "max_position_embeddings": max_position,
            "pretrained_weights_used": False,
            "safe_repair_used": False,
        },
    )


def run_native_baseline(
    request_path: Path,
    checkout: Path,
    output_dir: Path,
    *,
    manifest_path: Path,
    tokenizer_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Run a prepared request and emit the strict canonical native receipt."""

    request = _verify_request(request_path)
    method_id = str(request["method"]["method_id"])
    methods = load_external_baseline_manifest(manifest_path)
    method = methods.get(method_id)
    if method is None or method["integration_status"] != "native_port_ready":
        raise NativeBaselinePortError(
            f"method is no longer admitted for native execution: {method_id}"
        )
    checkout_receipt = verify_external_checkout(method, checkout)
    if checkout_receipt != request["checkout"]:
        raise NativeBaselinePortError("native checkout no longer matches the prepared request")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise NativeBaselinePortError(f"native output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if method_id == "rgfn":
        rows, reaction_calls, audit = _run_rgfn(request, request_path.parent, checkout, output_dir)
    elif method_id == "defog_unconditional":
        rows, reaction_calls, audit = _run_defog(request, request_path.parent, checkout, output_dir)
    elif method_id == "genmol_safe":
        rows, reaction_calls, audit = _run_genmol(
            request,
            request_path.parent,
            checkout,
            output_dir,
            tokenizer_snapshot=tokenizer_snapshot,
        )
    else:
        raise NativeBaselinePortError(f"no reviewed native runtime exists for {method_id}")
    attempts = int(request["requested_attempts"])
    if len(rows) != attempts or [row["attempt_index"] for row in rows] != list(range(attempts)):
        raise NativeBaselinePortError("native runtime did not preserve every requested attempt")
    samples_path = output_dir / "samples.csv"
    write_csv(samples_path, rows, ["attempt_index", "status", "product_smiles"])
    wall_seconds = time.perf_counter() - started
    receipt = {
        "schema_version": "forge.external_ugi_native_run_receipt.v1",
        "method_id": method_id,
        "upstream_commit": method["commit"],
        "seed": int(request["seed"]),
        "requested_attempts": attempts,
        "repairs_or_retries": False,
        "generator_calls": attempts,
        "reaction_calls": reaction_calls,
        "route_calls": 0,
        "oracle_calls": 0,
        "wall_seconds": wall_seconds,
        "samples_sha256": str(sha256_file(samples_path)),
    }
    write_json(output_dir / "receipt.json", receipt)
    runtime_audit = {
        "schema_version": "forge.external_ugi_native_runtime_audit.v1",
        "status": "pass",
        "method_id": method_id,
        "request": artifact_record(request_path),
        "samples": artifact_record(samples_path),
        "receipt": artifact_record(output_dir / "receipt.json"),
        "checkout": checkout_receipt,
        "native_adapter": audit,
        "gates": {
            "attempt_denominator_preserved": len(rows) == attempts,
            "repairs_or_retries_absent": True,
            "route_and_oracle_calls_zero": True,
            "candidate_selection_absent": True,
        },
    }
    write_json(output_dir / "result.json", runtime_audit)
    # The immutable request, pinned source adapter, final checkpoint and raw attempt ledger are the
    # reproducibility contract.  Native work directories contain staged checkout copies, processed
    # dataset caches and intermediate logs that are both redundant and expensive to move off Modal.
    # Remove only these scratch trees after every retained artifact has been hash-recorded.
    for scratch_name in ("work", "upstream"):
        shutil.rmtree(output_dir / scratch_name, ignore_errors=True)
    return runtime_audit


__all__ = ["run_native_baseline"]
