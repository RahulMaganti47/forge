# FORGE

[Experiments](examples/README.md)

Reaction-guided generative design of ionizable lipids, using AGILE-type Ugi 3CR,
repeated aza-Michael addition, and repeated reductive amination.

## Installation

Use Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen --extra dev --extra modal
```

## Usage

Fetch the paper checkpoints and evaluation inputs, then generate molecules or reproduce tables:

```bash
uv run forge artifacts fetch --group paper-model-v1
uv run forge artifacts fetch --group submission19337-evidence-v1
uv run forge generate --replicate 0 --family ugi --count 2 --seed 42 \
  --device cpu --output results/demo
uv run forge reproduce --target all --output results/tables
```

Artifacts are stored in the `forge-paper-artifacts` Modal volume in `kosha-labs/main`.
Teammates need workspace membership and an authenticated Modal profile. See
[artifact access](examples/README.md#checkpoints-and-data) for offline restoration and optional weights.

Generation retains every attempt, including failures. Table reproduction aggregates saved evidence.
The [experiment guide](examples/README.md) covers training, evaluation, baselines, and
[reproduction limitations](examples/README.md#limitations).

## Structure

```text
forge/          # Library: models, flow, chemistry, data, and evaluation
examples/       # Paper experiments, table aggregation, and reproduction checks
tests/          # Local correctness checks
configs/        # Frozen experiment settings
manifests/      # Checkpoint and data identities
provenance/     # Source and result records
```

The primary model is [model/networks/transformer.py](forge/model/networks/transformer.py), with
[semantic losses](forge/model/objectives/transformer.py) and [PCGrad](forge/model/objectives/pcgrad.py).

## Local checks

```bash
uv run pytest
uv run python examples/check_reproduction.py --output results/reproduction-check
```

The reproduction check verifies both artifact bundles, matches 89 numerical table rows, and repeats
two CPU Ugi attempts. Full GPU retraining remains unverified; unavailable historical inputs are
listed in the experiment guide.
