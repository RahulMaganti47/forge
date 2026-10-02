#!/usr/bin/env bash
set -euo pipefail

for replicate in 0 1 2; do
  config=configs/multireaction/shared_bias_program_role_core_saturation_seeds12_v2.json
  if [ "$replicate" -eq 0 ]; then
    config=configs/multireaction/shared_bias_parallel_shared_bias_program_role_source_v2.json
  fi
  output="results/retrained-seed${replicate}"
  uv run forge train --profile paper --device cuda --replicate "$replicate" \
    --config "$config" --output "$output"
  uv run forge evaluate --profile paper --device cuda --replicate "$replicate" \
    --config "$output/evaluation_config.json" \
    --checkpoint "$output/training/checkpoints.tar" \
    --training-result "$output/training/result.json" \
    --study-design "$output/study_design.json" \
    --output "results/reassessed-seed${replicate}"
done
