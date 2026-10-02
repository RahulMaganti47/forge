# Experiment data

Processed inputs and saved results for the three-family submission are included in this checkout.
Install [Git LFS](https://git-lfs.com/) and run `git lfs pull` to retrieve the larger files.

| Directory | Contents |
|---|---|
| `training/` | Ugi and multi-reaction datasets, atom annotations, designs and seed-0/1/2 training records |
| `splits/` | R0 fold assignments; reaction-specific splits are alongside their training datasets |
| `vendor/` | LNPDB records and qualified chemistry registries |
| `baselines/` | Native-baseline requests, train/calibration/held-out inputs and component catalogues |
| `evaluation/` | Per-seed summaries, complete attempt ledgers, controls, ablations and final adjudications |
| `imaging/` | The four reported 4-hour ROI values used in Figure 2 |

`table_reference.json` contains the expected 89 numerical table rows. Data source information is in
[SOURCES.md](SOURCES.md).

## Install the included data

```bash
uv run forge artifacts install --group paper-model-v1
uv run forge artifacts install --group submission19337-evidence-v1
uv run forge reproduce --target all --output results/tables
```

Installation verifies the included files and copies them to the input paths used by the frozen
configs. It refuses to replace a different local file. The manifests map each `checkout_path` to its
original `repo_path`; contents and hashes are unchanged.

## Download checkpoints and caches

Install the [GitHub CLI](https://cli.github.com/) and authenticate with `gh auth login`.
Readers need access to this private repository.

```bash
uv run forge artifacts fetch --group paper-model-v1
uv run forge artifacts fetch --group submission19337-evidence-v1
```

These commands combine the included data with the downloads below, verify both archive and file
hashes, and restore the complete bundles. They require no Modal account.

| Download | Contents | Size |
|---|---|---|
| [Primary model](https://github.com/RahulMaganti47/forge-iclr-review/releases/download/submission19337-artifacts-v1/paper-model-v1.tar.gz) | Three checkpoints and the primary training cache | 324 MB |
| [Supporting inputs](https://github.com/RahulMaganti47/forge-iclr-review/releases/download/submission19337-artifacts-v1/submission19337-evidence-v1.tar.gz) | Supporting cache and two chemistry source PDFs | 23 MB |

Optional original architecture-study weights are available for
[seed 0](https://github.com/RahulMaganti47/forge-iclr-review/releases/download/submission19337-artifacts-v1/architecture-seed0.tar.gz),
[seed 1](https://github.com/RahulMaganti47/forge-iclr-review/releases/download/submission19337-artifacts-v1/architecture-seed1.tar.gz) and
[seed 2](https://github.com/RahulMaganti47/forge-iclr-review/releases/download/submission19337-artifacts-v1/architecture-seed2.tar.gz)
(1.23 GB each). Fetch them with `forge artifacts fetch --group submission19337-ablations-v1`.

For offline installation, save the downloaded tarballs in a directory and run
`forge artifacts fetch --group <group> --bundle /path/to/downloads`. This uses the same hash checks
without contacting GitHub.

The original complete bundles remain on the `forge-paper-artifacts` Modal volume in
`kosha-labs/main`. Use `forge artifacts fetch --group <group> --backend modal` with a workspace
profile (`--profile <name>`) if needed.

Table 11 supports summary replay only; its upstream HeLa inputs are unavailable. The imaging CSV
contains individual reported values, with no replicate or uncertainty metadata.
