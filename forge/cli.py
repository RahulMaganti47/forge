"""Small command surface for the pinned submission; expensive work is always explicit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "generate":
        from .commands.generate import main as generate

        return generate(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    artifacts = commands.add_parser(
        "artifacts", help="fetch, restore or verify immutable artifacts"
    )
    artifacts.add_argument("action", choices=("fetch", "install", "restore", "verify"))
    artifacts.add_argument(
        "--group",
        choices=(
            "paper-model-v1",
            "submission19337-evidence-v1",
            "submission19337-ablations-v1",
            "hela-oracle-v1",
        ),
        required=True,
    )
    artifacts.add_argument("--bundle", type=Path)
    artifacts.add_argument("--profile", default="kosha-labs")
    artifacts.add_argument("--environment", default="main")
    artifacts.add_argument("--backend", choices=("github", "modal"), default="github")
    replay = commands.add_parser("reproduce", help="reaggregate frozen tables; no training")
    replay.add_argument(
        "--target", default="all", choices=["all", *(f"table-{i}" for i in range(1, 12))]
    )
    prepare = commands.add_parser("prepare", help="rebuild the full numeric training cache")
    prepare.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multireaction/shared_mixed_production_cache_v1.json"),
    )
    train = commands.add_parser("train", help="run frozen mechanism training and evaluation")
    train.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/multireaction/shared_bias_parallel_shared_bias_program_role_source_v2.json"
        ),
    )
    train.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    train.add_argument("--replicate", type=int, choices=(0, 1, 2), default=0)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser(
        "evaluate", help="evaluate released or freshly trained checkpoints"
    )
    evaluate.add_argument("--replicate", type=int, choices=(0, 1, 2), default=0)
    evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluate.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--checkpoint", type=Path)
    evaluate.add_argument("--training-result", type=Path)
    evaluate.add_argument("--study-design", type=Path)
    assess = commands.add_parser(
        "assess", help="apply the common Ugi verifier to an attempt ledger"
    )
    assess.add_argument("--attempts", type=Path, required=True)
    baseline = commands.add_parser("baseline", help="execute a native port or the finite catalogue")
    baseline.add_argument("method", choices=("catalogue", "selector", "native"))
    baseline.add_argument(
        "--config",
        type=Path,
        default=Path("configs/multireaction/finite_component_catalogue_baseline_v1.json"),
    )
    baseline.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    baseline.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    baseline.add_argument("--replicate", type=int, choices=(0, 1, 2), default=0)
    baseline.add_argument("--request", type=Path)
    baseline.add_argument("--checkout", type=Path)
    baseline.add_argument("--tokenizer-snapshot", type=Path)
    for command in (artifacts, replay, prepare, train, evaluate, assess, baseline):
        command.add_argument("--root", type=Path, default=Path.cwd(), help="release checkout root")
    for command in (replay, prepare, train, evaluate, assess, baseline):
        command.add_argument("--output", type=Path, required=True)
    commands.add_parser("generate", help="bounded generation from a real paper checkpoint")
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        if not (root / "manifests/paper-model-v1.json").is_file():
            raise ValueError("--root must name this submission's release checkout")
        from .commands import artifacts as store

        if args.command == "artifacts":
            manifest = root / f"manifests/{args.group}.json"
            if args.action == "verify":
                result = store.verify(root, manifest)
            elif args.action == "install":
                result = store.install(root, manifest)
            elif args.action == "restore":
                if args.bundle is None:
                    raise ValueError("restore requires --bundle")
                result = store.restore(root, manifest, args.bundle)
            else:
                result = store.fetch(
                    root,
                    manifest,
                    profile=args.profile,
                    environment=args.environment,
                    backend=args.backend,
                    downloads=args.bundle,
                )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0 if result["ready"] else 2
        output = args.output.resolve()
        if output.exists() and not (args.command == "train" and args.resume):
            raise ValueError(f"output already exists: {output}")
        if args.command == "reproduce":
            from .commands.reproduce import reproduce

            reproduce(root, output, args.target)
        elif args.command == "prepare":
            from forge.corpus.synthesis_program_production_cache import (
                build_synthesis_program_production_cache,
            )

            output.mkdir(parents=True)
            build_synthesis_program_production_cache(
                root / args.config, root, output / "cache.npz", output / "result.json"
            )
        elif args.command == "train":
            from forge.experiments.study import run_transformer_mechanism_study

            run_transformer_mechanism_study(
                root / args.config,
                root,
                output,
                work_dir=output / "work",
                profile="full" if args.profile == "paper" else "smoke",
                replicate=args.replicate,
                allocated_device=args.device,
                resume=args.resume,
            )
        elif args.command == "evaluate":
            from .commands.evaluation import evaluate

            evaluate(
                root,
                output,
                replicate=args.replicate,
                device=args.device,
                profile=args.profile,
                config=args.config,
                checkpoint=args.checkpoint,
                training_result=args.training_result,
                study_design=args.study_design,
            )
        elif args.command == "assess":
            from .commands.assessment import assess

            assess(root, args.attempts, output)
        elif args.command == "baseline":
            if args.method == "selector":
                from forge.experiments.benchmarks import (
                    run_learned_inventory_selector_study,
                )

                run_learned_inventory_selector_study(
                    root / "configs/multireaction/learned_inventory_selector_v1.json",
                    root,
                    output,
                    profile="full" if args.profile == "paper" else "smoke",
                    replicate=args.replicate,
                    allocated_device=args.device,
                )
            elif args.method == "catalogue":
                from forge.experiments.catalogue import (
                    run_finite_component_catalogue_baseline,
                )

                run_finite_component_catalogue_baseline(
                    root / args.config,
                    root,
                    output,
                    profile="full" if args.profile == "paper" else "smoke",
                    replicate=args.replicate,
                )
            else:
                if args.request is None or args.checkout is None:
                    raise ValueError("native execution requires --request and --checkout")
                from forge.baselines.runtime import run_native_baseline

                run_native_baseline(
                    args.request,
                    args.checkout,
                    output,
                    manifest_path=root / "configs/baselines/external_ugi_v1.json",
                    tokenizer_snapshot=args.tokenizer_snapshot,
                )
        print(f"Saved {args.command} outputs to {output}")
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"forge: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
