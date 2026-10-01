# Known limits of reproduction

The release supports frozen-table replay, real-checkpoint generation, complete cache preparation,
training/resume smoke checks and manuscript compilation. These checks do not establish an independent
full GPU retraining replication. See the committed qualification receipts for exact executed scope.

## Missing historical records

The final HeLa adjudication supports Table 11's displayed summary. Four of its authenticated upstream
records have not been recovered:

| Original result directory under `results/phase1/` | Expected `result.json` SHA-256 |
|---|---|
| `ugi_morphology_high_potency_challenger_v1_retry1` | `91839b27d54bc73c55dd471eb52e125e601d4b1a2219e8fe5548cfb305df5ed8` |
| `ugi_morphology_high_potency_proposal_v1` | `d8117d11e06a592bc6e2a54b2b0d549f610228dc4595c97aa918e68d2f0e71bb` |
| `ugi_high_potency_challenger_terminal_generation_v1` | `e47ccb2c74f062bf3fa37f38d25df17deefd633ce38bd1b8e05207258a4cb606` |
| `ugi_high_potency_challenger_continuous_ranking_v1` | `dc19f876235cd68c9ae36936ea53dddf0f31dc04451218c47e67dcccf01e3471` |

The source analysis, proposal, matched generation, ranking and adjudication modules are included in
`src/forge/diagnostics/`, with their scientific dependencies. Their input records and original oracle
environment still need recovery before claiming end-to-end HeLa replication. A new run is a new
experiment, not recovery of these missing records. The reported negative result remains unchanged.

The final Figure 2 composite is available exactly as submitted. The supplied mouse image, ROI values
and editable standalone ROI plot are retained, but the exact final composite's authoring script was
not recovered. The `paper` command preserves that composite. It does not claim to regenerate it.

## Historical source and environment identity

The core extraction is anchored to development snapshot
`ac9ef87be4c8e5c14c477bc194a333344e621d3e`, associated with the recovered seed-0 evaluation source.
Reporting and narrow diagnostic support also require later files; their original and release hashes
are listed in `provenance/source_inventory.json`. Relocation, unreachable helper removal and the
authenticated checkpoint-subset loader are explicit release changes.

The historical seed-1/2 evaluation fingerprint
`8050c5ac500fd64bf6e3252cc537d3fb16f9d6ef8d06044fdb272618d4591e03` does not yet have a completely
recovered source tree. A complete original training environment manifest is also missing. Finding
checkpoints does not resolve either issue. `uv.lock` records the release environment used now.
Historical config/source pins inside results remain intact, including unresolved older identities;
they are not rewritten to point at the current release and called historical matches.

## Scientific interpretation

- Exact L1 replay is transform consistency, not synthesis success, procurement or biological efficacy.
- Component novelty is relative to training catalogues and the declared verifier. It is not strong
  catalogue escape; all admissible decompositions were not exhaustively searched for that claim.
- Structural zeros, undefined metrics, invalid attempts and negative diagnostics are retained.
- The imaging evidence contains individual reported ROI values, without supplied replicate counts
  or uncertainty. This release preserves the submitted wording; claims about potency still require
  review against those records before a camera-ready revision.
- Linux/CUDA and native external-baseline production runs need their own qualification. Local CPU
  smoke results and newly locked dependencies cannot certify historical GPU bitwise equivalence.
