# FORGE: anonymous review repository

Research artifacts for the three-family ICLR manuscript: AGILE-type Ugi 3-CR,
repeated aza-Michael addition and repeated reductive amination.

This release contains historical model, sampling, training and evaluation code;
the three final model checkpoints; the packed training cache; experiment
specifications; frozen result summaries for the main comparison tables; and
eight lipid examples with exact L1 building-block replay checks.

## Reproduce the reported aggregates

From this directory, using Python 3.10 or later:

```bash
python3 scripts/reproduce.py
```

This needs only the Python standard library. It verifies every manifest hash,
preserves all generation-attempt denominators, and recomputes three-seed means
and sample standard deviations from the retained evaluation summaries.
The report is written to `outputs/reproduction.json`.

The corrected Ugi results are **862.6 ± 19.2 distinct exact-L1 products** and
**839.2 ± 26.5 training-catalogue component-novel products**, per 1,000 attempts.
The component-novel calculation uses the shared training catalogue throughout.

## Check the chemistry and model artifacts

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/reproduce.py --chemistry --checkpoints
PYTHONPATH=. python -m pytest -q tests
```

Checkpoint and cache files are split into parts smaller than 8 MB for anonymous
hosting. The script reconstructs them under `restored/`, checks their complete
SHA-256 hashes, loads each model strictly on CPU, and verifies its state hash.
Use `python scripts/reproduce.py --restore` to reconstruct files without PyTorch.
The chemistry check replays all eight displayed lipids using the pinned registry.

## Contents

| Location | Contents |
| --- | --- |
| `forge/model/` | Historical graph transformer, flow, training and sampling implementation |
| `forge/assembly/` | Registry-backed final-assembly adapters |
| `experiments/phase1/multireaction/` | Historical training/evaluation functions and retained run specifications |
| `evidence/table1/` | Conditioned, shared-null and cyclic-control evaluation summaries for three seeds |
| `evidence/table2/` | Common Ugi assessments for nine methods and three seeds |
| `evidence/training/` | Training receipts and model designs |
| `evidence/atlas_l1.json` | Product and building-block identities and recorded replay results |
| `artifacts/` | Final checkpoints at step 9143 and the packed training cache |
| `MANIFEST.json` | Release-file hashes and original-source hashes |

## Reproduction limits

The code was extracted from the retained snapshot that matched the complete
seed-0 evaluation-source fingerprint. Its relevant file hashes are recorded in
`evidence/source_status.json`. Exact evaluation-source matches for seeds 1 and 2
and a complete historical training-source/environment manifest were not
recovered. Loading their checkpoints with the retained code does not establish
that this exact source produced those runs.

The packed cache includes training, calibration and held-out records with their
fold assignments. The complete upstream corpus-construction inputs, intermediate
checkpoints, all per-attempt evaluation ledgers and external baseline environments
are not included. Historical experiment specifications reference additional
hash-pinned inputs; they are documentary specifications, not standalone rerun
commands. The commands above check retained summaries, chemistry and model
artifacts. They do not rerun training or the full production comparison.

Some unchanged historical modules and receipts contain route-related interfaces
or diagnostics. They are retained only where required by source dependencies or
immutable records. The three-family paper evaluates generation and final-assembly
consistency; this release makes no precursor-route, supplier-availability, complete
dossier or experimental synthesis claim.

This is a fresh anonymous snapshot with no original Git history. It contains no
manuscript PDF, experimental images or author metadata. The package requirements
describe the review checks; they are not a recovered training-environment lockfile.
