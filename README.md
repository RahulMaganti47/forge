# FORGE — submission 19337

Reproduction code for the **35-page, three-family submission** in [`paper/submission.pdf`](paper/submission.pdf).
The reference PDF SHA-256 is `7ea7cf2e51922e5e58cd6659dc183409e465ede2eb1586fc8d0ec39935c4fd53`.
The three reaction programs are AGILE-type Ugi 3CR, repeated aza-Michael addition, and repeated
reductive amination. Exact L1 verification measures transform consistency.

## Install and reproduce

```bash
uv sync --frozen --extra dev --extra modal
uv run forge artifacts fetch --group paper-model-v1
uv run forge artifacts fetch --group submission19337-evidence-v1
uv run forge reproduce --target all --output results/reproduced-tables
uv run forge generate --replicate 0 --family ugi --count 4 --seed 42 --output results/demo
```

Artifacts are stored in the `forge-paper-artifacts` Modal volume in workspace `kosha-labs`,
environment `main`. Your account needs membership in that workspace. All downloads are verified
against committed SHA-256 manifests. Local bundles can be restored without Modal credentials.
See [artifact access](docs/ARTIFACTS.md) and the [reproduction guide](docs/REPRODUCTION.md).

`reproduce` reaggregates frozen results. `generate` uses actual paper weights and retains all
attempts, including failures. Neither command claims a new full training replication.
Known historical evidence gaps are recorded in [limitations](docs/LIMITATIONS.md).

## Layout

- `src/forge/`: model, flow, registry-backed chemistry, preparation, evaluation, and baseline ports.
- `configs/`: pinned contracts for the submitted experiments.
- `manifests/`: immutable artifact identities and storage locations.
- `paper/`: exact submission and matching LaTeX, bibliography, figures, and source receipts.
- `provenance/`: source extraction and executed qualification records.
- `tests/`: numerical, chemistry, artifact, and command checks.

The original development repository and the later 22-family study are outside this release.
