"""Verify the review release and recompute main-table aggregates offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: str):
    return json.loads((ROOT / path).read_text())


def verified_inputs() -> dict[str, str]:
    pins = {}
    for entry in read("MANIFEST.json")["files"]:
        path = ROOT / entry["path"]
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"Missing or changed release input: {entry['path']}")
        pins[entry["path"]] = entry["sha256"]
    return pins


def aggregate(values: list[float]) -> dict:
    if len(values) != 3:
        raise ValueError("Each comparison must retain three independent training seeds")
    return {"seeds": values, "mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def tables() -> dict:
    index = read("evidence/index.json")
    table1 = {}
    for arm, paths in index["table1"].items():
        rows = defaultdict(list)
        for i, path in enumerate(paths):
            doc = read(path)
            if doc["seed"] != 20260825 + i or doc["status"] != "pass":
                raise ValueError(f"Unexpected seed or result status in {path}")
            metrics = next(iter(doc["checkpoint_metrics"].values()))["9143"]["heldout"]
            for family, values in metrics.items():
                if values["samples"] != 3072:
                    raise ValueError("Changed generation-attempt denominator")
                rows[family].append(100 * values["exact_l1_yield_per_attempt"])
        table1[arm] = {family: aggregate(values) for family, values in rows.items()}
    rows = defaultdict(list)
    for path in index["table2"]:
        assessment = read(path)["common_assessment"]
        if assessment["attempts"] != 3072:
            raise ValueError("Changed common-assessment denominator")
        rows[assessment["method_id"]].append(assessment)
    table2 = {}
    for method, seeds in rows.items():
        seeds.sort(key=lambda row: row["seed"])
        if [row["seed"] for row in seeds] != [20260825, 20260826, 20260827]:
            raise ValueError(f"Missing or repeated seeds for {method}")
        table2[method] = {
            label: aggregate([row["metrics"][key] for row in seeds])
            for label, key in {
                "distinct_exact_l1_per_1000": "unique_exact_l1_products_per_1000_attempts",
                "training_catalogue_component_novel_per_1000": "unique_open_ended_exact_l1_products_per_1000_attempts",
            }.items()
        }
    return {"table1_exact_l1_percent": table1, "table2": table2}


def restore() -> list[str]:
    restored = []
    for item in read("artifacts/index.json"):
        path = ROOT / item["path"]
        if path.exists():
            if digest(path) != item["sha256"]:
                raise ValueError(f"Existing artifact differs: {item['path']}")
        else:
            payload = b"".join((ROOT / part).read_bytes() for part in item["parts"])
            if (
                len(payload) != item["bytes"]
                or hashlib.sha256(payload).hexdigest() != item["sha256"]
            ):
                raise ValueError(f"Corrupt artifact shards: {item['path']}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        restored.append(item["path"])
    return restored


def chemistry() -> dict:
    from forge.assembly.ugi3 import Ugi3AssemblyAdapter

    registry = ROOT / "data/vendor/qualified_reactions_v1.json"
    source = read("evidence/atlas_l1.json")
    adapter = Ugi3AssemblyAdapter.from_registry(
        registry, expected_sha256=source["inputs"]["data/vendor/qualified_reactions_v1.json"]
    )
    verified = []
    for row in source["rows"]:
        product = row["canonical_constitutional_smiles"]
        candidates = [
            c
            for c in adapter.transform_consistent_decomposition_candidates(product)
            if c.registry_handle_qualified
        ]
        if len(candidates) != 1:
            raise ValueError("Atlas decomposition is no longer unique")
        if not adapter.check_forward(row["components_by_role"], product).exact:
            raise ValueError("Atlas forward replay failed")
        verified.append(row["display_order"])
    return {"exact_replay_verified_rows": verified, "scope": "L1 structural consistency only"}


def checkpoints() -> dict:
    import torch

    from forge.corpus.synthesis_program_production_cache import SynthesisProgramProductionCache
    from forge.model.defog_feasibility import _model_state_sha256
    from forge.model.synthesis_program_training import build_synthesis_program_flow

    restore()
    torch.set_num_threads(2)
    cache_path = ROOT / "restored/cache.npz"
    cache = SynthesisProgramProductionCache(cache_path)
    results = []
    for seed in range(3):
        path = ROOT / f"restored/checkpoints/seed{seed}.pt"
        package = torch.load(path, map_location="cpu", weights_only=True)
        if package["cache_sha256"] != digest(cache_path):
            raise ValueError("Checkpoint/cache identity mismatch")
        design_path = ROOT / f"evidence/training/seed{seed}_production_design.json"
        if package["design_sha256"] != digest(design_path):
            raise ValueError("Checkpoint/design identity mismatch")
        model = build_synthesis_program_flow(
            vocabulary=cache.vocabulary,
            node_classes=len(cache.atom_vocabulary),
            model_config=package["model_config"],
            device=torch.device("cpu"),
        )
        model.load_state_dict(package["model_state"], strict=True)
        if _model_state_sha256(model) != package["model_state_sha256"]:
            raise ValueError("Model-state identity mismatch")
        results.append({"seed_label": seed, "sha256": digest(path), "strict_load": True})
    return {
        "checkpoints": results,
        "cache_fold_counts": cache.fold_counts(),
        "scope": "CPU loading and state identity, not rerun generation or training",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--chemistry", action="store_true")
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    report = {"input_sha256": verified_inputs(), "aggregates": tables()}
    if args.restore:
        report["restored"] = restore()
    if args.chemistry:
        report["chemistry"] = chemistry()
    if args.checkpoints:
        report["checkpoint_check"] = checkpoints()
    output = ROOT / "outputs/reproduction.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    forge = report["aggregates"]["table2"]["forge_transformer"]
    for name, value in forge.items():
        print(f"FORGE {name}: {value['mean']:.1f} +/- {value['sample_sd']:.1f}")
    print("Release checks passed; report: outputs/reproduction.json")


if __name__ == "__main__":
    main()
