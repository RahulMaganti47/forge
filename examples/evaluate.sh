#!/usr/bin/env bash
set -euo pipefail

for replicate in 0 1 2; do
  uv run forge evaluate --profile paper --device cuda --replicate "$replicate" \
    --output "results/evaluated-seed${replicate}"
done
