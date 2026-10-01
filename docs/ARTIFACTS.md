# Artifact access and ownership

The Git repository contains code, configurations, the exact submission, matching manuscript sources,
tests and manifests. Large immutable model and evidence payloads live in Modal:

| Bundle | Contents | Approximate size |
|---|---|---|
| `/paper-model-v1` | Three primary checkpoint archives, training/design companions, cache, registries and frozen summaries | 363 MB upstream bundle |
| `/submission19337-evidence-v1` | Additional table evidence, source attempt ledgers, cache preparation inputs and native-baseline request inputs | 230 MB |
| `/submission19337-ablations-v1` | Three original eight-arm architecture-study checkpoint archives | 4.01 GB, optional |

Workspace: `kosha-labs`. Environment: `main`. Volume: `forge-paper-artifacts`.
The committed model manifest selects the relevant files from the original bundle; the original
manifest and its hash are retained in `provenance/`. Existing payload bytes are unchanged.

## Teammate setup

1. Obtain access to the GitHub repository and membership in the `kosha-labs` Modal workspace.
2. Authenticate Modal on your machine (`modal token new`) and select the appropriate local profile.
3. Fetch both default bundles and verify them:

```bash
forge artifacts fetch --group paper-model-v1 --profile kosha-labs
forge artifacts fetch --group submission19337-evidence-v1 --profile kosha-labs
forge artifacts verify --group paper-model-v1
forge artifacts verify --group submission19337-evidence-v1

# Optional: original architecture/FACT weights.
forge artifacts fetch --group submission19337-ablations-v1 --profile kosha-labs
```

Membership is account-specific; possession of the Git URL alone does not grant Modal access.
No new invitations or changes to workspace permissions are made by the repository commands.
If your local profile has another name, supply it with `--profile`.

## Offline or alternate transfer

```bash
forge artifacts restore --group paper-model-v1 --bundle /downloads/paper-model-v1
forge artifacts restore --group submission19337-evidence-v1 \
  --bundle /downloads/submission19337-evidence-v1
```

The same payload may be transferred by another approved channel: its hash, byte count and destination
are the contract. Restore validates every selected payload before writing, rejects traversal and
symlinks, refuses to overwrite different local files, and is idempotent for matching files.
Failed downloads do not produce a successful verification record.

Treat published bundle prefixes as immutable. Publish corrections under a new version, update the
manifest and document why the identity changed. Never replace a missing historical file with a
plausible substitute. The original archives preserve intermediate checkpoints as well as the final
step; readers select the exact authenticated member without unsafe archive extraction.
