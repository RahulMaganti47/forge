"""Verify downloaded bundles, replay the numerical tables, and repeat two CPU Ugi attempts."""

import argparse
from pathlib import Path

from forge.commands.qualification import qualify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = qualify(args.root, args.output)
    print(f"Reproduction check {result['status']}; receipt: {args.output / 'receipt.json'}")


if __name__ == "__main__":
    main()
