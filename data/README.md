# Experiment data

Processed inputs and saved results for the three-family experiments are included in this checkout.
Install [Git LFS](https://git-lfs.com/) and run `git lfs pull` to retrieve the larger files.

| Directory | Contents |
|---|---|
| `training/` | Ugi and multi-reaction datasets, atom annotations, designs and seed-0/1/2 training records |
| `splits/` | R0 fold assignments; reaction-specific splits are alongside their training datasets |
| `vendor/` | LNPDB records and qualified chemistry registries |
| `baselines/` | Native-baseline requests, train/calibration/held-out inputs and component catalogues |
| `evaluation/` | Per-seed summaries, complete attempt ledgers, controls, ablations and final adjudications |
| `hela/` | Frozen HeLa predictor, curated assay data, splits and calibration records |
| `imaging/` | The four reported 4-hour ROI values used in Figure 2 |

`table_reference.json` contains the expected 89 numerical table rows. Data source information is in
[SOURCES.md](SOURCES.md).

## Install the included data

```bash
forge artifacts install --group paper-model-v1
forge artifacts install --group evidence-v1
forge reproduce --target all --output results/tables
```

## Checkpoints and caches

Checkpoint and cache bundles are supplied separately. Restore them from local directories:

```bash
forge artifacts restore --group paper-model-v1 --bundle /path/to/paper-model-v1
forge artifacts restore --group evidence-v1 --bundle /path/to/evidence-v1
```

| Bundle | Contents |
|---|---|
| `paper-model-v1` | Three checkpoints and the primary training cache |
| `evidence-v1` | Supporting cache and chemistry source PDFs |
| `ablations-v1` | Optional architecture-study weights for all three seeds |

For the architecture study:

```bash
forge artifacts restore --group ablations-v1 --bundle /path/to/ablations-v1
```

## HeLa predictor

The original checkpoint and matching inputs are included in `hela/`. Restore them with:

```bash
forge artifacts install --group hela-oracle-v1
```
