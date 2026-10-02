# FORGE

[Experiments](examples/README.md)

Reaction-guided generative design of ionizable lipids, using Ugi 3CR,
repeated aza-Michael addition, and repeated reductive amination.

![FORGE generation and exact assembly verification](assets/forge_overview.png)

![Mouse bioluminescence imaging and reported 4-hour ROI signals](assets/forge_in_vivo.png)

## Installation

Use Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
git lfs install
git lfs pull
uv sync --frozen --extra dev
source .venv/bin/activate
```

## Usage

```bash
forge artifacts install --group paper-model-v1
forge artifacts install --group evidence-v1
forge reproduce --target all --output results/tables
```

Restore the checkpoint and cache bundles supplied separately:

```bash
forge artifacts restore --group paper-model-v1 --bundle /path/to/paper-model-v1
forge artifacts restore --group evidence-v1 --bundle /path/to/evidence-v1
forge generate --replicate 0 --family ugi --count 2 --seed 42 \
  --device cpu --output results/demo
```

## Structure

```text
forge/          # Library: models, flow, chemistry, data, and evaluation
examples/       # Runnable paper experiment recipes
tests/          # Local correctness checks
configs/        # Frozen experiment settings
manifests/      # Checkpoint and data identities
data/           # Processed datasets, splits and saved experiment results
```

The primary model is [model/networks/transformer.py](forge/model/networks/transformer.py), with
[semantic losses](forge/model/objectives/transformer.py) and [PCGrad](forge/model/objectives/pcgrad.py).
