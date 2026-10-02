# Paper experiments

Run these examples from the repository root after [installation](../README.md#installation).
Each command writes to a new directory under `results/`.

| Example | Runs |
|---|---|
| [reproduce.py](reproduce.py) | All numerical tables and a repeated CPU generation check |
| [generate.sh](generate.sh) | Two Ugi attempts from the seed-0 checkpoint |
| [train.sh](train.sh) | Primary-model training and evaluation for all three replicates |
| [evaluate.sh](evaluate.sh) | Evaluation of the supplied checkpoints for all three replicates |
| [baselines.sh](baselines.sh) | Catalogue and learned-selector baselines for all three replicates |

## Data and checkpoints

The processed datasets, splits and saved results are included in [data/](../data/README.md).
Run `git lfs pull`, then fetch checkpoints and caches with an authenticated GitHub CLI:

```bash
uv run forge artifacts fetch --group paper-model-v1
uv run forge artifacts fetch --group submission19337-evidence-v1
```

Fetch combines the included data with the [direct downloads](../data/README.md#download-checkpoints-and-caches)
and verifies all hashes. Readers need access to this private repository; no Modal account is needed.

Architecture-study weights are optional:

```bash
uv run forge artifacts fetch --group submission19337-ablations-v1
```

For table replay without checkpoint downloads, use `forge artifacts install --group <group>` for
both required groups.

## Reproduce tables and generate molecules

```bash
uv run python examples/reproduce.py --output results/reproduction-check
bash examples/generate.sh
```

The reproduction check verifies the bundles, matches all 89 numerical table rows, and repeats two
CPU Ugi attempts. Its `receipt.json` records commands, input hashes, environment and output hashes.
Generation retains every attempt, including invalid molecules. Use `forge generate --help` to select
another family, replicate or sampling budget.

To reproduce an individual table, use `forge reproduce --target table-N --output results/table-N`.
The expected numerical rows are in `data/table_reference.json`.

| Tables | Config under `configs/reproduction/` |
|---|---|
| 1, 3–4: production and controls | `gem_table1_core_saturation_complete_v1.json`, `gem_table1_core_saturation_final_v1.json` |
| 2, 7–8: common Ugi benchmark | `common_ugi.json` |
| 5: decoder/source ablation | `gem_table5_decoder_source_ablation_v1.json` |
| 6: architecture ablation | `gem_table8_architecture_ablations_v1.json` |
| 9: catalogue comparison | `gem_table9_catalogue_comparison_v1.json` |
| 10: structural realism | `gem_table7_lipid_realism_v1.json` |
| 11: HeLa diagnostic | Final adjudication in the evidence bundle |

Table reproduction aggregates saved evidence. It does not rerun Table 5's decoder-intervention
sweep. Table 11 supports summary replay only; four upstream study records are unavailable.
The recovered HeLa predictor and inputs are in [data/hela](../data/hela).

## Train and evaluate

Both full-run examples require CUDA:

```bash
bash examples/train.sh
bash examples/evaluate.sh
```

`train.sh` evaluates each newly trained checkpoint using that run's config and companion records.
`evaluate.sh` evaluates the supplied checkpoints. Replicates 0, 1 and 2 use training seeds 20260825,
20260826 and 20260827. Primary training uses 9,143 steps and 126 effective examples per step; final
evaluation requests 3,072 attempts per program and replicate.

For a two-step CPU smoke run:

```bash
uv run forge train --profile smoke --output results/train-smoke
```

To rebuild the complete training cache, use `forge prepare --output results/cache-rebuild`.
`forge train --resume` resumes an existing run. Custom evaluation requires `--config`, `--checkpoint`,
`--training-result` and `--study-design` together, as shown in `train.sh`.

The same training command accepts these configs under `configs/multireaction/`:

| Experiment | Config | Replicates |
|---|---|---|
| Shared-null control | `shared_bias_shared_null_core_saturation_v1.json` | 0, 1, 2 |
| Cyclic-ID control | `shared_bias_cyclic_core_saturation_v1.json` | 0, 1, 2 |
| Global-source control | `shared_bias_parallel_shared_bias_global_source_control_v2.json` | 0 |
| Architecture/FACT study | `transformer_mechanism_study_v1.json` | 0, 1, 2 |

On Modal, launch long jobs detached and persist checkpoints on a volume. Record call IDs and
source/config/input hashes before monitoring.

## Baselines

```bash
bash examples/baselines.sh
uv run forge baseline native \
  --request runs/phase1-external-ugi-native-requests-v3/rgfn/seed0/request.json \
  --checkout /path/to/clean/RGFN --output results/rgfn-seed0
```

The learned-selector baseline requires CUDA. Native RGFN, DeFoG and GenMol/SAFE run in pinned
upstream checkouts and separate environments. Use the corresponding method/seed request;
GenMol also accepts `--tokenizer-snapshot`. See [external_ugi_v1.json](../configs/baselines/external_ugi_v1.json).

Apply the common verifier to a baseline/evaluation attempt ledger with
`forge assess --attempts /path/to/attempts.jsonl.gz --output results/common-assessment`.
The raw `generate/attempts.jsonl` has a different schema. Preserve invalid attempts and parser failures.

Training/evaluation implementations are in [forge/experiments](../forge/experiments), baseline
implementations in [forge/baselines](../forge/baselines), and numerical table aggregation in
[forge/reporting](../forge/reporting). Structural-realism reassessment uses
`forge.experiments.realism.run_lipid_realism_assessment` and
`forge.experiments.realism_aggregation.aggregate_lipid_realism` with
`configs/multireaction/common_lipid_realism_v1.json` and complete method-seed ledgers.
