#!/usr/bin/env bash
set -euo pipefail

for replicate in 0 1 2; do
  uv run forge baseline catalogue --profile paper --replicate "$replicate" \
    --output "results/catalogue-seed${replicate}"
  uv run forge baseline selector --profile paper --device cuda --replicate "$replicate" \
    --output "results/selector-seed${replicate}"
done
