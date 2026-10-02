# FORGE: Reaction-Guided Generative Design of Ionizable Lipids

Implementation for the **35-page anonymous ICLR 2027 submission 19337**.
[Reference paper](paper/submission.pdf) · [Reproduction guide](docs/REPRODUCTION.md) ·
[Expected results](docs/RESULTS.md) · [Artifact access](docs/ARTIFACTS.md) ·
[Known limitations](docs/LIMITATIONS.md)

![FORGE generation and exact assembly verification](docs/assets/forge-overview.png)

FORGE generates whole-molecule graphs conditioned on three reaction programs: AGILE-type Ugi 3CR,
repeated aza-Michael addition, and repeated reductive amination. Exact L1 verification measures
transform consistency. The supplied paper and its matching source are preserved byte-for-byte;
their identities are recorded in [paper/README.md](paper/README.md).

## Quick start

Use Python 3.11 and [uv](https://docs.astral.sh/uv/). Run from the release checkout:

```bash
uv sync --frozen --extra dev --extra modal
uv run forge artifacts fetch --group paper-model-v1
uv run forge artifacts fetch --group submission19337-evidence-v1
uv run forge reproduce --target all --output results/reproduced-tables
uv run forge generate --replicate 0 --family ugi --count 2 --seed 42 \
  --device cpu --output results/demo
```

The Modal volume is `forge-paper-artifacts` in `kosha-labs/main`. Teammates need workspace membership
and their own authenticated Modal profile. [Offline restoration](docs/ARTIFACTS.md#offline-or-alternate-transfer)
also verifies the same hashes. The two required bundles are approximately 593 MB together; original
architecture-study checkpoints are a separate, optional 4.01 GB bundle.

`reproduce` reaggregates saved results. `generate` uses the original paper weights and saves every
attempt, including failures. Read [the paper-to-command map](docs/REPRODUCTION.md#paper-to-command-map)
for training, controls, baselines, figures, seeds, and expected outputs.

## Check the release

After restoring both required bundles:

```bash
uv run python scripts/qualify_release.py --output results/release-check
```

This checks artifacts, verifies all 89 numerical table rows, and repeats two CPU Ugi attempts.
It records commands, timings, environment, input hashes, and outputs in `receipt.json`.
It performs no training. Full GPU reproduction and the missing historical records remain explicitly
listed in [limitations](docs/LIMITATIONS.md).

## Code and data

- `src/forge/`: scientific implementation and CLI; see [code boundaries](docs/ARCHITECTURE.md).
- `configs/`: pinned experiment and chemistry contracts.
- `manifests/`: immutable identities for data, weights, and evidence stored in Modal.
- `paper/`: reference PDF, matching LaTeX, bibliography, and figure assets.
- `provenance/`: source extraction and executed qualification receipts.
- `tests/`: numerical, chemistry, artifact, and workflow checks.

## Citation and licenses

Reference this package as **FORGE: Reaction-Guided Generative Design of Ionizable Lipids,
submission 19337**, together with the Git commit used. Final author metadata and an archival
camera-ready citation have not been supplied. This release does not assert acceptance.

The implementation retains its [MIT license](LICENSE). Third-party projects, source documents, and
data have separate terms recorded in [third-party attribution](docs/THIRD_PARTY.md).
The later 22-family study belongs to the development repository and is outside this release.
