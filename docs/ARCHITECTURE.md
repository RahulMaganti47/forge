# Code boundaries

`forge.release` provides the CLI, artifact access, manuscript build, and table replay.

`forge.assembly` and `forge.chemistry` read chemistry from authenticated registries. `forge.corpus`
builds weighted, split-aware program records and the deterministic numeric NPZ cache. `forge.flow`
implements the discrete sampler. `forge.model` implements sparse whole-molecule state, program
conditioning, routed Transformer heads, source marginals, masked losses, sampling and exact-L1
assessment. `forge.workflows` joins those components under frozen training/evaluation and baseline
contracts. `forge.reporting` authenticates the evidence before aggregating manuscript rows.

`forge.diagnostics` retains the historical HeLa analysis dependencies required by Appendix B.
Missing upstream artifacts are listed in [limitations](LIMITATIONS.md).

No runtime import reaches into the development checkout, a global experiment catalogue, or a
22-family implementation. Source provenance records the extraction and the unused definitions
removed from shared historical utility modules. Historical numerical functions retain their logic;
release edits change imports, formatting, comments, docstrings, and workflow entry points.
The custom evaluation CLI authenticates a complete config/archive/training-result/design set and
calls the same evaluator as released checkpoints. Generated native-baseline commands use the release
CLI. The bounded reader check shares the existing manuscript-row comparison with the tests; it
does not change training, sampling, or numerical aggregation.

The checkpoint loader adds subset selection at the orchestration boundary: a final-step
evaluation may select step 9143 from an archive that also retains steps 100, 500, 1700 and 4500.
It rejects missing or duplicate steps and verifies every archive member against the training receipt.
It does not pick a checkpoint by performance.

Every numeric artifact records its configuration, seed and input hashes. Expensive functions remain
separate from artifact fetching and evidence replay. Outputs preserve all attempts and conditional
denominators; errors fail explicitly instead of repairing data or weakening acceptance gates.
