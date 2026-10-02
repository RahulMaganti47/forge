# Data sources and external methods

Upstream projects, datasets, publications, and tokenizer assets retain their own terms.

## External methods

The following identities and license labels are recorded in the frozen
[`external_ugi_v1.json`](../configs/baselines/external_ugi_v1.json) manifest. These are the manifest's
recorded labels, not a new audit of upstream terms. External projects run in separate environments;
FORGE's adapters preserve the specified commit and apply compatibility edits to a copy.

| Project | Pinned commit | Recorded license | Use in this package |
|---|---|---|---|
| [Ou et al. Synthesis-DAG + Chemformer](https://github.com/john-bradshaw/synthesis-dags) | `9fb6c710598c4e250c6c49d15910644a3cf80129` | GPL-3.0 | `excluded_no_author_implementation` |
| [SynFlowNet](https://github.com/mirunacrt/synflownet) | `574f1e148f42e0c79877318fa9d84d2552cf5025` | MIT | `excluded_incompatible_reaction_arity` |
| [RGFN](https://github.com/koziarskilab/RGFN) | `6ce59169f855ed18f34ba4e8279de93bee306e4f` | MIT | `native_port_ready` |
| [SynCoGen](https://github.com/andreirekesh/SynCoGen) | `6b38eec26ccc687b32808c37aefdf75a4a30f1da` | unlicensed_repository | `excluded_unlicensed` |
| [DeFoG unconditional](https://github.com/manuelmlmadeira/DeFoG) | `365bda9affadd5c2307014a0532ddaa244399441` | MIT | `native_port_ready` |
| [GenMol/SAFE](https://github.com/NVIDIA-BioNeMo/genmol) | `add09fc83b7255bd09c797e527c0f4b51f5fb7c1` | Apache-2.0-code | `native_port_ready` |

GenMol code and pretrained-weight licensing are separate. The recorded FORGE comparison trains
from scratch and uses the pinned SAFE tokenizer; it does not admit external pretrained model weights.
SynCoGen remains excluded with unresolved licensing. Excluded methods are cited context, not
completed experimental runs. See the manifest for the exact exclusion reasons and native environments.

## Data and source materials

| Material | Identity/access record | Terms recorded here |
|---|---|---|
| AGILE assay table and Ugi source data | Model manifest and preserved training/evidence receipts | Upstream dataset terms are not recorded as an independent license in these bundles. |
| LNPDB records and derived splits | Model/evidence manifests and input hashes in result receipts | Upstream dataset terms need separate verification before public redistribution. |
| Qualified chemistry registries and derived caches | Model manifest, registry source citations, preparation receipts | FORGE-derived records retain source provenance; source provenance does not replace source terms. |
| Primary papers and supplementary documents | Evidence manifest paths, original filenames, and SHA-256 digests | Publisher licenses and source notices govern these documents; no blanket MIT license is asserted. |

The exact payload inventory is in the three committed manifests. Chemistry source citations remain
in the authenticated registries; do not replace those records with
retyped reaction definitions. This document records unresolved metadata rather than assuming that
an accessible or published file is unrestricted. A future public release must establish the terms
for the actual files it distributes.

## Evaluation records

Table 11 supports summary replay only; its upstream HeLa inputs are unavailable.
`imaging/reported_roi_values.csv` contains the four reported 4-hour values from Figure 2;
replicate counts and uncertainty metadata are unavailable.
