#!/usr/bin/env bash
set -euo pipefail

uv run forge generate --replicate 0 --family ugi --count 2 --seed 42 \
  --device cpu --output results/generated-ugi
