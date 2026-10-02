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

Prefix commands with `uv run` from the checkout. Commands and qualification scripts accept
`--root /path/to/checkout`. Output directories must be new, except when explicitly resuming training. Results and large data remain outside Git.

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
Checkpoint/restart states persist under the output directory. Evaluate a fresh run using its own
generated config and companions; all four custom input arguments are required together:

```bash
# After: forge train --profile smoke --output results/train-smoke
forge evaluate --profile smoke --device cpu --replicate 0 \
  --config results/train-smoke/evaluation_config.json \
  --checkpoint results/train-smoke/training/checkpoints.tar \
  --training-result results/train-smoke/training/result.json \
  --study-design results/train-smoke/study_design.json \
  --output results/reassessed-training
```

For a production run, use that run's generated config, `--profile paper --device cuda`, and the
matching replicate. Relative custom input paths resolve against `--root`. The evaluator verifies the
archive digest, design and cache pins, member/state identities, checkpoint schedule, and device.
Failures inside the evaluator are kept under `<output>.failed/` with `FAILED.json`; they are never
published as successful outputs. Incomplete or mismatched inputs are rejected before evaluation. The `generate` command continues to authenticate only the original paper weights.

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

## Paper-to-command map

To recompute structural-realism diagnostics from attempt ledgers, use
`forge.workflows.lipid_realism_assessment.run_lipid_realism_assessment` with
`configs/multireaction/common_lipid_realism_v1.json`, the checkout root, ledger path, new output
directory, and the recorded `method_id`, `seed`, and `expected_attempts=3072`. Combine the complete
three-seed set using `forge.workflows.lipid_realism_aggregation.aggregate_lipid_realism`.

Run `forge reproduce --target table-N --output results/table-N` for any numerical table
N=1–11. Replace N with its number. Each output contains `table-N.tex`, a numeric/provenance JSON,
and `receipt.json`. `--target all` produces all 11 tables in one directory. The following map uses
paths relative to `configs/`; every replay needs the two required bundles, never the optional
architecture checkpoint bundle. Expected row counts, values and receipts are in [results](RESULTS.md).

| Paper item | Exact replay target / output | Frozen contract and implementation | Budget / source |
|---|---|---|---|
| Table 1: conditioned/null/cyclic assembly | `table-1` → `table-1.tex` | `reproduction/gem_table1_core_saturation_complete_v1.json`; `reporting/gem_table1.py` | Three training seeds; 3,072 held-out attempts/program/seed |
| Table 2: common Ugi benchmark | `table-2` → `table-2.tex` | `reproduction/common_ugi.json`; `release/reproduce.py` | Nine methods × three seeds; 3,072 attempts/method/seed |
| Table 3: production comparison | `table-3` → `table-3.tex` | Table-1 final/control evidence; `reporting/gem_table4.py` | Three programs; matched conditioned/null/cyclic arms |
| Table 4: final exact-L1 counts | `table-4` → `table-4.tex` | `reproduction/gem_table1_core_saturation_final_v1.json`; `reporting/gem_table6.py` | Exact numerators; all 3,072 attempts in each of nine cells |
| Table 5: decoder/source ablation | `table-5` → `table-5.tex` | `reproduction/gem_table5_decoder_source_ablation_v1.json`; `reporting/gem_table5.py` | Seed 20260825; checkpoint 9,143; 3,072 Ugi attempts/row |
| Table 6: architecture ablation | `table-6` → `table-6.tex` | `reproduction/gem_table8_architecture_ablations_v1.json`; `reporting/gem_table8.py` | Eight arms × three seeds; 1,700 training steps; 3,072 attempts/program/seed |
| Table 7: baseline seed rows | `table-7` → `table-7.tex` | `reproduction/common_ugi.json`; `release/reproduce.py` | All 27 method-seed records; no seed selection |
| Table 8: decomposition outcomes | `table-8` → `table-8.tex` | Same common Ugi records; `release/reproduce.py` | Coverage, precision, no decomposition, and ambiguity reported separately |
| Table 9: finite catalogue comparison | `table-9` → `table-9.tex` | `reproduction/gem_table9_catalogue_comparison_v1.json`; `reporting/gem_table9.py` | Three programs × three seeds; 3,072 attempts/program/seed |
| Table 10: structural realism | `table-10` → `table-10.tex` | `reproduction/gem_table7_lipid_realism_v1.json`; `release/reproduce.py` | Frozen method-blind aggregate; three-seed mean/sample SD |
| Table 11: HeLa diagnostic | `table-11` → `table-11.tex` | Frozen final adjudication; `diagnostics/hela/` and `release/reproduce.py` | Two 3,072-attempt arms; upstream records/oracle inputs missing |
| Table 12: chemistry background | `forge paper --output results/manuscript` → `main.pdf` | Pinned registries and `multireaction/lnpdb_reaction_programs_v1.json` | Descriptive manuscript table; no new experimental execution |
| Table 13: fixed structure atlas | `forge paper --rebuild-figures --output results/manuscript` → `regenerated-atlas/` | Preserved atlas receipt; `release/atlas.py` | Eight fixed structures; unique forward and atom-order checks |
| Figure 1: method overview | Same manuscript rebuild → `figures/forge_overview.pdf` | `paper/source/figures/forge_overview.tex` | Rendering of the existing scientific diagram |
| Figure 2: imaging | Same manuscript rebuild preserves `figures/forge_in_vivo_v2.pdf` | Exact composite; ROI CSV and standalone plot source | Measurements are supplied observations; final composition script missing |
| Figure 3: reaction schemes | Same manuscript rebuild → `figures/reaction_schemes.pdf` | `paper/source/figures/reaction_schemes.tex` | Registry-backed chemistry illustrated in the submitted source |

### Commands for the underlying experiments

The commands above replay or render the recorded paper outputs. New inference, retraining, and
reassessment are separate operations:

| Experiment / paper items | Command | Config, inputs, and expected outputs |
|---|---|---|
| Original final checkpoints; Tables 1–4, 9 | `forge evaluate --replicate 0 --device cuda --output results/seed0-evaluation` | `shared_bias_parallel_program_role_seed0_core_saturation_v2.json`; `result.json`, `samples.jsonl.gz`, common attempt ledgers, molecule report |
| Primary seed 0 training | `forge train --profile paper --device cuda --replicate 0 --config configs/multireaction/shared_bias_parallel_shared_bias_program_role_source_v2.json --output results/primary-seed0` | 9,143 optimizer steps, 126 effective examples/step; `training/checkpoints.tar`, training result, design and evaluation config |
| Primary seeds 1/2 training | `forge train --profile paper --device cuda --replicate 1 --config configs/multireaction/shared_bias_program_role_core_saturation_seeds12_v2.json --output results/primary-seed1` | Repeat with replicate 2 and a new output directory; preserve the separately frozen seed-1/2 contract |
| Shared-null control | `forge train --profile paper --device cuda --replicate 0 --config configs/multireaction/shared_bias_shared_null_core_saturation_v1.json --output results/null-seed0` | Repeat across all three replicates; training/checkpoint/evaluation companions |
| Cyclic-ID control | `forge train --profile paper --device cuda --replicate 0 --config configs/multireaction/shared_bias_cyclic_core_saturation_v1.json --output results/cyclic-seed0` | Repeat across all three replicates; frozen cyclic mapping |
| Global-source control; Table 5 | `forge train --profile paper --device cuda --replicate 0 --config configs/multireaction/shared_bias_parallel_shared_bias_global_source_control_v2.json --output results/global-seed0` | Global-source contract; Table 5's historical rows remain replayed under their original identities |
| Architecture/FACT study; Table 6 | `forge train --profile paper --device cuda --replicate 0 --config configs/multireaction/transformer_mechanism_study_v1.json --output results/architecture-seed0` | Eight arms; 1,700 steps, 128 effective examples/step; repeat all three replicates |
| Finite catalogue; Tables 2, 7–9 | `forge baseline catalogue --profile paper --replicate 0 --output results/catalogue-seed0` | `finite_component_catalogue_baseline_v1.json`; all attempt ledgers and assessment results |
| Learned selector; Tables 2, 7–8 | `forge baseline selector --profile paper --device cuda --replicate 0 --output results/selector-seed0` | `learned_inventory_selector_v1.json`; model and complete method-neutral attempt ledger |
| Native baselines; Tables 2, 7–8 | `forge baseline native --request runs/phase1-external-ugi-native-requests-v3/rgfn/seed0/request.json --checkout /path/to/clean/RGFN --output results/rgfn-seed0` | Pinned upstream checkout and all request inputs; native result and every attempted product. Repeat corresponding method/seed requests |
| Common Ugi reassessment | `forge assess --attempts path/to/attempts.jsonl.gz --output results/common-assessment` | Canonical method-neutral ledger; `assessed_attempts.jsonl.gz`, `result.json` |

For the final production checkpoints, evaluation uses 32 flow steps, the frozen decoder, 512
calibration attempts at evaluated checkpoints, and 3,072 held-out attempts/program at the final step.
The generation example uses seed 42; it is not a paper-evaluation seed. Raw `generate` rows and
canonical benchmark ledgers have different schemas: use `assess` on a method-neutral ledger from
the baseline/evaluation workflows, not directly on `generate/attempts.jsonl`.

Table 5's exact historical decoder-intervention sweep and Table 11's upstream diagnostic cannot
be claimed as freshly replicated by table replay. Table 5's contracts and source evaluations remain
authenticated in its config; Table 11's missing records are listed in [limitations](LIMITATIONS.md).
For recomputation of Table 10, use the structural-realism functions described above with each
complete method-seed ledger; the common config controls that assessment. Full native and GPU runs
were not launched for this release.

### Environment and time

The committed qualification environment is Python 3.11 on macOS ARM64 with CPU float32 and two
Torch threads for the generation example. The package is also checked on Ubuntu in CI without
private artifacts. Original production runs used the declared H100/CUDA contracts. Those are
requirements of the frozen runs, not an inference that any CUDA device is historically equivalent.

The earlier seed-0 Ugi check measured 14.0 and 14.8 seconds for two attempts in its recorded CPU
environment ([receipt](../provenance/qualification/checkpoints.json)). Full training, baseline,
cache-preparation, table-replay and manuscript-build runtimes are not established by those numbers.
Where no timing is recorded, runtime is **not measured**. The release-check script records new
wall times for each command and the environment used.

### Manuscript tools

For compilation, install a TeX distribution providing `latexmk`, `pdflatex`, BibTeX, TikZ/PGFPlots,
chemfig, tcolorbox, and the standard packages referenced by the retained source. The ICLR style
files are included. Install Python rendering dependencies with:

```bash
uv sync --frozen --extra dev --extra figures
uv run forge paper --rebuild-figures --output results/manuscript
```

CairoSVG requires its native Cairo library; on systems where it is not supplied, install Cairo
through your system package manager. Poppler's `pdftoppm` is used only to regenerate the README
preview, not for training or the basic manuscript command. Missing tools are environment failures,
not scientific results. A successful manuscript build has 35 pages and matching extracted text,
retains Figure 2, and saves `build_receipt.json`. It is not expected to have the same PDF binary hash
because TeX metadata can differ.

### Bounded release qualification

```bash
uv run python scripts/qualify_release.py --output results/release-check
```

Restore both required bundles before running. The command neither downloads data nor starts
training. It verifies both bundles, replays Tables 1–11, checks all 89 rows, and generates two
seed-0 Ugi attempts twice on CPU using seed 42, batch size 2 and two Torch threads. Invalid attempts
remain in the denominator. Compare scientific attempt rows, not elapsed time or temporary paths.
The output contains logs, tables, both complete generation outputs and `receipt.json`. A failed
execution writes a failed receipt and `FAILED.json`, never a passing qualification. Missing or
mismatched bundles are rejected before an output directory is created. Source and config changes
during the check are rejected.

The table test compares all 89 generated numerical rows with the exact manuscript. The novelty column
uses `unique_open_ended_exact_l1_products_per_1000_attempts`, giving FORGE **839.2 ± 26.5**.
It must not be replaced with distinct-L1 yield (**862.6 ± 19.2**) or method-visible novelty.

Rebuilding a plot or table is distinct from repeating the underlying experiment. Experimental imaging
assets are supplied observations; this code cannot recreate laboratory measurements or invent their
missing replicate/uncertainty metadata. No prospective experimental work is part of this release.
