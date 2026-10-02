# Expected results and reproduction status

The reference is the exact 35-page submission, not a newly revised manuscript. Reported means and
sample standard deviations use independent training seeds 20260825, 20260826, and 20260827.
A generated molecule is not an independent training replicate. N/E and structural zeros are retained.

## Numerical table replay

`forge reproduce --target all --output results/tables` writes 11 tables. The manuscript comparison
checks **89 numerical rows**, with no numerical tolerance or selection of convenient rows.
The following are expected row counts and exact output digests, independently recorded by the
[reader workflow receipt](../provenance/qualification/reader_workflow.json).

| Table | Numerical rows | Expected `table-N.tex` SHA-256 |
|---|---:|---|
| 1 | 3 | `7429d99629ca33db1310bb308379655d3f7514ff00406bd3053647f102c7c6e4` |
| 2 | 9 | `a7ab72a8e90c2dde35d0d73a3453e0afae7bc50d197915169badf370a129b3af` |
| 3 | 6 | `b2ce2a15fdc135cc30194f6f3d473e97e9770e2c607371874aa4e31cd54e074e` |
| 4 | 9 | `b21009d1c1c893705d6b615d25b977ad6ac4a88450cb40a7d351642fa95c8812` |
| 5 | 3 | `8e9c4e90ba722e51a216a0cb065143cf0e7ea29230713d6e8a59f5a36b9a1920` |
| 6 | 8 | `9dd7802bc9b9eb03ef88c00bc14a7073086ddb6a4693d01008ded81378278ba1` |
| 7 | 27 | `371dda8f4ef614a05793dec3b86769d83a91fc0b9f19efbe6b214bfd6b0a34b1` |
| 8 | 9 | `5e0aefd7cedd536d33bcfe16fe009db8c2e797383b4d1e7d6c457889c6e60478` |
| 9 | 7 | `f82801f868f60432d776e4d1ea2e34fafb2dcb9d541c63a499908ffcff8f2374` |
| 10 | 6 | `97bdf3e21f8264dcb29342bafb1287edc1a222b14210eaaa58d7a8eae75a978f` |
| 11 | 2 | `c7dc09b231e82a1333d8e868d075323ea29c61316d03c4a8fcd33d8623059a3c` |

Tables 12 and 13 are descriptive chemistry and fixed-structure atlas tables; they are covered by
manuscript compilation and atlas checks rather than numerical table replay. The output `.json` files
retain per-seed values or input identities. Follow the [paper-to-command map](REPRODUCTION.md#paper-to-command-map)
for each result's configuration, budget, and underlying implementation.

## Representative paper values

Table 2 reports FORGE's common Ugi benchmark per **1,000 requested attempts**:

| Metric | Reported mean ± sample SD |
|---|---:|
| Valid products | 965.0 ± 21.9 |
| Exact-L1 products | 963.9 ± 21.4 |
| Distinct exact-L1 products | 862.6 ± 19.2 |
| Training-catalogue component-novel exact-L1 products | 839.2 ± 26.5 |
| Designated held-component exact-L1 products | 1.4 ± 1.7 |
| Mean pairwise ECFP4 distance | 0.643 ± 0.022 |

Component novelty and distinct-L1 yield are different columns. Broader component novelty does not
establish strong recovery of designated held components. Exact L1 is transform consistency, not
synthesis success or biological efficacy.

Table 11 preserves the reported negative HeLa diagnostic: support-enriched and nested potency arms
both attempted 3,072 outputs. Their respective valid counts were 2,955 and 2,962, while their
interpolative-applicability counts were 90 and 82. Replaying those rows does not recover the four
missing upstream records or the dependent historical oracle environment.

## Executed checks

| Workflow | Verified scope | Evidence |
|---|---|---|
| Artifact access | Selected model files and all 141 additional evidence files hash-verified; three optional original ablation archives independently verified | [Release record](../provenance/qualification/release.json), [evidence download](../provenance/qualification/evidence_download.json), [ablation download](../provenance/qualification/ablations_download.json) |
| Fresh locked installation | 75 tests passed, including artifact checks; all 89 rows match; two seed-0 CPU Ugi attempts repeated identically | [Delivery record](../provenance/qualification/paper_workflow.json), [fresh reader receipt](../provenance/qualification/fresh_reader_workflow.json) |
| Released checkpoint coverage | All three seeds × three programs, two attempts/cell, repeated on CPU | [Checkpoint receipt](../provenance/qualification/checkpoints.json) |
| Preparation | Complete cache rebuild matches original digest | [Cache receipt](../provenance/qualification/cache_rebuild.json) |
| Training/restart | Two CPU optimizer steps; interruption after durable step 1 and resume to step 2 matched complete arm records | [Training](../provenance/qualification/training_smoke.json), [interrupted resume](../provenance/qualification/interrupted_training_resume.json) |
| Custom evaluation CLI | Authenticated two-step smoke checkpoint reevaluated; all existing evaluation gates passed, 18 sample rows retained | [Evaluation receipt](../provenance/qualification/custom_checkpoint_evaluation.json) |
| Selector/common assessment | Three-step CPU selector, four attempts, complete common reassessment | [Selector](../provenance/qualification/selector_smoke.json), [reassessment](../provenance/qualification/selector_reassessment.json) |
| Manuscript | 35 pages and matching extracted text; eight fixed structures reverified; Figure 2 preserved | [Build receipt](../provenance/qualification/manuscript_build.json) |

The bounded reader check measured 33.7 seconds for replay and two generation commands on the recorded
macOS ARM64 CPU environment. Its table command took 0.11 seconds; the generation commands took
16.83 and 16.73 seconds each. Artifact verification occurs before that timed interval. These figures
are observations for this environment, not runtime guarantees. Full GPU training/evaluation and
native external-baseline production times remain **not measured for this release**.

The final fresh-checkout reader run took 70.2 seconds while tests and custom checkpoint evaluation
ran concurrently on the same CPU. It passed the same checks using source identity recorded in
[fresh_reader_workflow.json](../provenance/qualification/fresh_reader_workflow.json). This concurrent
run is a functional check, not a performance comparison. The custom evaluation result matched the
entire earlier smoke evaluation record. The [delivery record](../provenance/qualification/paper_workflow.json)
records the tested source and confirms that frozen configurations and manuscript bytes were unchanged.

## What remains unverified

Full GPU retraining and independent native-baseline production were not executed. Four HeLa upstream
result records, dependent oracle inputs/environment, the complete historical seed-1/2 source identity,
and the original training-environment record remain incomplete. The final Figure 2 composition source
and imaging replication/uncertainty metadata are missing. See [limitations](LIMITATIONS.md).

Latest passing Linux checks do not erase the earlier optional batched-VJP PCGrad bitwise-equivalence
failure. Its strict test and [failure receipt](../provenance/qualification/linux_pcgrad_equivalence.json)
are retained; the submitted training configs use the sequential backend.
