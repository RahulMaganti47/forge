# Code boundaries

`forge.release` owns the portable command surface, artifact access and submitted-paper reporting.
It does not own chemical transforms or the numerical training kernels.

`forge.assembly` and `forge.chemistry` read chemistry from authenticated registries. `forge.corpus`
builds weighted, split-aware program records and the deterministic numeric NPZ cache. `forge.flow`
implements the discrete sampler. `forge.model` implements sparse whole-molecule state, program
conditioning, routed Transformer heads, source marginals, masked losses, sampling and exact-L1
assessment. `forge.workflows` joins those components under frozen training/evaluation and baseline
contracts. `forge.reporting` authenticates the evidence before aggregating manuscript rows.

`forge.diagnostics` retains the narrow historical HeLa analysis dependencies required by Appendix B.
It is not a generic biological-optimization entry point. Its missing upstream artifacts are explicit.

No runtime import reaches into the development checkout, a global experiment catalogue, or a
22-family implementation. Source provenance records the extraction and the unused definitions
removed from shared historical utility modules. Numerical functions retained from the historical
core preserve their bodies apart from formatting/import relocation.

The checkpoint-subset loader is the deliberate exception at the orchestration boundary: a final-step
evaluation may select step 9143 from an archive that also retains steps 100, 500, 1700 and 4500.
It rejects missing or duplicate steps and verifies every archive member against the training receipt.
It does not pick a checkpoint by performance.

Every numeric artifact records its configuration, seed and input hashes. Expensive functions remain
separate from artifact fetching and evidence replay. Outputs preserve all attempts and conditional
denominators; errors fail explicitly instead of repairing data or weakening acceptance gates.
