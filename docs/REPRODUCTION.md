# Reproducing submission 19337

The reference is the 35-page PDF committed as `paper/submission.pdf`. The matching manuscript is
`paper/source/main.tex`. Neither is rewritten by the reproduction commands. The submitted methods
use three reaction programs; the later 22-family work is excluded.

## Environment

Use Python 3.11 and the committed `uv.lock`:

```bash
uv sync --frozen --extra dev --extra modal --extra figures
uv run pytest
```

The release qualification environment is recorded in `provenance/qualification/`. This is a newly
qualified environment, not a recovered historical training environment. Linux/CUDA production
training requires the declared CUDA device. CPU generation is deterministic in the qualified
environment; bitwise equivalence between devices or RDKit versions is not promised.

## Choose the level of reproduction

| Task | Command | Meaning |
|---|---|---|
| Verify downloaded artifacts | `forge artifacts verify --group paper-model-v1` | Byte identity only |
| Reaggregate all numerical tables | `forge reproduce --target all --output results/tables` | Frozen evidence replay |
| Generate with a real checkpoint | `forge generate --replicate 0 --family ugi --count 4 --seed 42 --output results/demo` | New bounded generation |
| Rebuild the training cache | `forge prepare --output results/cache-rebuild` | Prepare all admitted records |
| Two-step training and evaluation | `forge train --profile smoke --output results/train-smoke` | Training and evaluation smoke test |
| Resume that run | `forge train --profile smoke --resume --output results/train-smoke` | Resume with authenticated restart state |
| Verify interruption and resume | `python scripts/qualify_training_resume.py --reference results/train-smoke/training/result.json --output results/interrupted-training` | Interrupt after durable step 1; resume to step 2 and compare complete arm records |
| Repeat the checkpoint qualification | `python scripts/qualify_checkpoints.py --output results/checkpoint-qualification` | All 3 seeds × 3 families, two attempts twice |
| Compile the exact source | `forge paper --rebuild-figures --output results/manuscript` | Build 35 pages and compare extracted text |

Prefix commands with `uv run` from the checkout. Commands and qualification scripts accept `--root /path/to/checkout`. Output directories must be
new, except when explicitly resuming training. Results and large data remain outside Git.

`generate` supports `ugi`, `aza-michael`, and `reductive-amination`. It authenticates the complete
checkpoint archive, its recorded member digest, design and cache before strict state-dictionary
loading. It uses step 9143, 32 flow steps, and the frozen core-saturation decoder. Count means
**attempts**, including invalid or unresolved outputs. There is no retry or top-up. Demo seed 42
is not the paper's held-out evaluation seed schedule.

## Full training and evaluation

These commands are computationally expensive and were **not** executed as part of this release.
Run on the declared hardware only after allocating the desired compute:

```bash
# Primary seed 0: 9,143 optimizer steps, 126 effective examples/step.
forge train --profile paper --device cuda --replicate 0 \
  --config configs/multireaction/shared_bias_parallel_shared_bias_program_role_source_v2.json \
  --output results/retrained-seed0

# Use this frozen contract for primary seeds 1 and 2.
forge train --profile paper --device cuda --replicate 1 \
  --config configs/multireaction/shared_bias_program_role_core_saturation_seeds12_v2.json \
  --output results/retrained-seed1

# Evaluate the authenticated released checkpoint with the complete frozen schedule.
forge evaluate --replicate 0 --device cuda --output results/evaluated-seed0

# The eight-arm architecture/FACT study has its own 1,700-step contract.
forge train --profile paper --device cuda --replicate 0 \
  --config configs/multireaction/transformer_mechanism_study_v1.json \
  --output results/architecture-seed0
```

Use replicates 0, 1 and 2 for the complete study. Report every program and training seed separately;
the three training seeds, not individual generated molecules, are the independent units.
The architecture study's full Transformer is not the separately trained production model.
The independently trained shared-null and cyclic controls use the same `train` command with
`configs/multireaction/shared_bias_shared_null_core_saturation_v1.json` and
`configs/multireaction/shared_bias_cyclic_core_saturation_v1.json`, respectively.

Training defaults to a two-step CPU smoke run. Full runs preserve the complete declared molecular
size support, weighted source measure, frozen splits, optimizer schedule and all failure records.
Checkpoint/restart states persist under the output directory. To evaluate freshly trained weights,
call `forge.workflows.production_evaluation.run_synthesis_program_production_evaluation` with the
run's `training/checkpoints.tar`, `training/result.json`, and `study_design.json` via
`dynamic_production_design_path`; the public `generate` command accepts only
the original paper weights.

For Modal, place the run directory on a persistent volume, launch in detached mode, and retain the
application/function identifier, request/configuration/source hashes and input pins before monitoring.
GPU provisioning and retries require separate execution authorization.

## Baselines and common verification

```bash
forge baseline catalogue --profile smoke --output results/catalogue-smoke
forge baseline selector --profile smoke --device cpu --output results/selector-smoke
forge assess --attempts path/to/attempts.jsonl.gz --output results/common-assessment
```

The finite catalogue and learned selector use training components only. `--profile paper` selects
the frozen full budget. For the selector, also specify `--device cuda`.

Native RGFN, DeFoG and GenMol/SAFE adapters are in `forge.workflows.native_baseline_runtime`.
Their repository URLs, exact commits, licenses and admitted/excluded status are in
`configs/baselines/external_ugi_v1.json`. Do not install all upstream environments into the FORGE
environment. Check out the specified clean commit, use that method's native environment, install this
release without replacing upstream dependencies, and pass the frozen request plus its `inputs/`:

```bash
forge baseline native \
  --request runs/phase1-external-ugi-native-requests-v3/rgfn/seed0/request.json \
  --checkout /path/to/clean/RGFN --output results/rgfn-seed0
```

Use the corresponding `defog_unconditional` or `genmol_safe` request for those methods. GenMol also
accepts `--tokenizer-snapshot`; the adapter verifies its identity. Native training/weights are not
requalified by the small FORGE smoke run. Preserve zero-output methods and parser failures in the
attempt denominator. The common assessment computes training-catalogue component novelty, distinct
exact L1, held-component recovery, decomposition coverage, precision and ambiguity separately.

## Manuscript coverage

To recompute structural-realism diagnostics from attempt ledgers, use
`forge.workflows.lipid_realism_assessment.run_lipid_realism_assessment` with
`configs/multireaction/common_lipid_realism_v1.json`, the checkout root, ledger path, new output
directory, and the recorded `method_id`, `seed`, and `expected_attempts=3072`. Combine the complete
three-seed set using `forge.workflows.lipid_realism_aggregation.aggregate_lipid_realism`.

| Paper item | Evidence and implementation |
|---|---|
| Table 1 | Nine matched conditioned/null/cyclic evaluation records; `reporting/gem_table1.py` |
| Tables 2, 7, 8 | 27 common Ugi assessments; `release/reproduce.py` |
| Tables 3, 4 | Matched production metrics and exact per-seed numerators; `reporting/gem_table4.py`, `gem_table6.py` |
| Table 5 | Frozen decoder/source intervention rows; `reporting/gem_table5.py` |
| Table 6 | 24 architecture-study rows and authenticated source evaluations; `reporting/gem_table8.py` |
| Table 9 | Three-program finite-catalogue comparison; `reporting/gem_table9.py` |
| Table 10 | Frozen method-blind structural-realism aggregate; `model/common_lipid_realism.py` and `release/reproduce.py` |
| Table 11 | Frozen negative HeLa adjudication; underlying analysis code in `diagnostics/`; upstream records missing |
| Table 12 | Chemistry definitions in the pinned registries and program configuration; descriptive table in source |
| Table 13 | Eight fixed source structures, exact forward checks and atom-order invariance; `release/atlas.py` |
| Figures 1, 3 | Editable TikZ source under `paper/source/figures/` |
| Figure 2 | Exact submitted composite PDF, original imaging asset and ROI CSV/plot source; composite authoring source missing |

The table test compares all 89 generated numerical rows with the exact manuscript. The novelty column
uses `unique_open_ended_exact_l1_products_per_1000_attempts`, giving FORGE **839.2 ± 26.5**.
It must not be replaced with distinct-L1 yield (**862.6 ± 19.2**) or method-visible novelty.

Rebuilding a plot or table is distinct from repeating the underlying experiment. Experimental imaging
assets are supplied observations; this code cannot recreate laboratory measurements or invent their
missing replicate/uncertainty metadata. No prospective experimental work is part of this release.
