FORGE ICLR 2027: COMPLETE workshop-framed Overleaf project

2026-09-26 anonymous reproducibility repository
- Added the supplied anonymous repository URL to the reproducibility statement.
  https://anonymous.4open.science/r/forge-iclr-review-FBF8/
- Public repository file listing and the PDF hyperlink target were verified.
- The current PDF has 35 pages; the link appears on page 10.
- Existing manuscript text and scientific results are unchanged.

2026-09-26 typography preference: italicize in vivo everywhere
- All 12 occurrences in main.pdf use italic fonts, including the abstract,
  Figure 2 caption, appendix heading and four reference titles. Keep this styling
  in future edits. Bibliography formatting is stored in references.bib.
- The PDF font spans were checked and pages 1, 8, 10, 11 and 25 were visually
  inspected. The document remains 34 pages.

2026-09-26 latest prose edits
- Added the approved closing discussion sentence about reaction-guided de novo
  lipid design and exact assembly verification.
- Restored "component-disjoint" in Table 2's caption.
- Only these two manuscript edits were made. Numerical results and the abstract
  are unchanged. main.pdf remains 34 pages; pages 7-8 were visually checked.

2026-09-26 current correction: training-catalogue component novelty
- Tables 2 and 7 now consistently use the shared training-component catalogue.
  The FORGE component-novel yield is 839.2 +/- 26.5 per 1,000 attempts; seed
  values are 852.9, 856.1 and 808.6. Distinct exact-L1 yield remains 862.6 +/- 19.2.
- Corrected the same component-novel metric for DeFoG, shared-null and both FACT
  baselines. All values were derived from hash-verified existing assessments.
- Updated the main and appendix prose and Table 2's metric definition. Table 9
  already uses the correct metric and is unchanged. Raw results are preserved.
- component_novel_correction_receipt.json records the 27 source hashes, both
  metric definitions, seed values, mean/sample SD and verification results.
- main.pdf remains 34 pages; changed pages 7, 22 and 23 were visually inspected.

2026-09-26 prior scope revision: generation and exact L1 assembly
- Removed precursor-route construction and dossier claims from the abstract,
  introduction, workflow, Figure 1, inference algorithm and appendices.
- The three-family study now explicitly evaluates final-assembly consistency and
  component novelty; precursor preparation and supplier availability are outside
  the evaluated scope.
- Retained all 12 result tables, all 194 retained result macros and the eight
  lipid identities and L1 building-block drawings unchanged. Unused route-result
  macros were removed without altering archived study outputs.
- Shortened the computational discussion to avoid a nearly empty spillover page.
- main.pdf has 34 pages; the main text ends on page 8 and the atlas is on pages
  33-34. The build has no unresolved references/citations or overfull boxes.
- manuscript_scope_revision.json records input/output hashes and checks.

2026-09-26 current display: exact L1 building blocks
- Both columns now use matching black skeletal vector drawings, with the same
  bond stroke and atom-label style. Product molecular identities, canonical
  SMILES and display order are unchanged.
- The second atlas column now shows the amine, aldehyde and isocyanide building
  blocks for each of the eight lipids, replacing the decorated 3D views.
- Each lipid has one unique decomposition passing the registry handle policy;
  forward replay yields exactly the displayed product. Atom-order variants
  reproduce the same component identities. The focused assembly checks pass.
- Drawings are vector PDFs. Their exact component SMILES, handle assessments,
  forward-check results and source hashes are recorded in
  figures/lipid_l1_building_blocks/l1_building_blocks_receipt.json.
- Regenerate the panels from the FORGE repository root with:
  PYTHONPATH=. .venv/bin/python iclr_final_3/figures/lipid_l1_building_blocks/render.py
- main.pdf is the 34-page revision; Table 13 is on pages 33-34. The original
  extracted product and 3D assets remain in the package but are not used by
  main.tex; both displayed columns use figures/lipid_l1_building_blocks/.
- This verifies computational final-assembly consistency only; it does not
  establish complete upstream routes, availability or experimental synthesis.

2026-09-26 initial lipid transfer (superseded display description)
- Replaced the previous two-example precursor figure and 12-row atlas with the
  eight lipid examples in Appendix F, Table 14 (pages 34-35) of the supplied
  26_FORGE_Reaction_Guided_Gener (1).pdf.
- The original embedded 2D structures and decorated 3D views were extracted
  without redrawing or resizing; their canonical SMILES and order are preserved.
- At initial transfer, the atlas was Table 13, Appendix D.2, on pages 36-37
  of the 37-page manuscript. See the current scope note for revised pagination.
- lipid_atlas_transfer_receipt.json records the source PDF hash, image hashes,
  exact SMILES, source locations, and evidence status. All eight SMILES parse
  and match RDKit's canonical strings. At initial transfer, generation, exact-L1
  replay and conformer optimization were not rerun. Exact-L1 replay was verified
  for the current building-block display as recorded above.
- Quantitative results are unchanged. These are illustrative structures, not a
  prospective candidate panel. The superseded, unused figure files remain in
  figures/ for preservation. The current manuscript uses newly rendered products
  and precursors in figures/lipid_l1_building_blocks/.

This is the complete project, including every required figure and style file.
Upload the ZIP as a new Overleaf project, or upload all folder contents into the
root of your existing project. Set main.tex as the main document and select
pdfLaTeX.

Figure 1 references figures/forge_overview.pdf, updated from its editable TikZ
source to show generation and exact assembly verification. Figure 2 references figures/forge_in_vivo_v2.pdf, the complete revised
mouse image and log-scale RLU chart. These are two different verified images.
All other figure assets are included. The editable chart source and original
mouse image are also retained in figures/.

Scientific identity
- Problem: design ionizable lipids beyond a fixed precursor inventory while
  retaining a verifiable final assembly reaction.
- Primary output: a complete molecular graph conditioned on a reaction program.
- Claim tested: conditioning changes verified exact-assembly yield and supports
  component identities outside training catalogues under the evaluated programs.
- Setting: Ugi, repeated aza-Michael addition and repeated reductive amination;
  matched shared-null, cyclic-label, finite-inventory and external comparators.
- Declared representation support: up to 194 atoms and three closure bonds,
  with registered chemistry and program-compatible coordinates. The model does
  not claim universal chemistry or zero-shot support for arbitrary reactions.

Editorial integration
- The supplied ICLR project is the technical base so its newer model mathematics,
  numerical results, appendices, references and exact figure assets are retained.
- The workshop draft's problem-first abstract and introduction, concise FORGE
  framing, and discussion structure were restored with updated results and limits.
- Older workshop yields (78.7%, 77.5%, 31.3%) were not carried forward. The newer
  ICLR source reports 96.4%, 72.2%, and 52.3% for the three conditioned programs.
- The introduction now distinguishes fixed-inventory reaction generators from
  SynCoGen's joint building-block/reaction generation.
- Figure 2 removes the lipid structures, crops the supplied image around both
  mice without changing source pixels, and shows the four reported values as
  vertical bars on a log RLU axis from 10^4 to 10^8.

Scientific boundaries
- Exact forward replay verifies the specified final reaction, not a complete
  experimental synthesis route.
- Component-level novelty does not establish strong product-level catalogue
  escape or recovery of designated held-out components.
- The ideal masked-flow theorem does not certify finite training or the numerical
  sampler. The full mathematical formulation remains in main.tex.
- Mouse ROI values are single reported values. The RLU designation was supplied
  by the author on 2026-09-25; it is not metadata in the unchanged source CSV.
  Replicate counts and error estimates were not supplied. The bars start at the
  displayed 10^4 axis floor.
- Experimental tables and macros were retained from the supplied newer ICLR
  manuscript. Raw run artifacts were not included in either archive, so those
  numbers were not independently recomputed here.

Source archive SHA-256
FORGE__ICLR_.zip:
  ee8663693acfee829c0b9847de058a9d08c8c79efc5b0e589968f1c74f68214e
FORGE__NeurIPS_Workshop_2026_ (4).zip:
  f25c8e6195de67655daf3db2777d0f8a2c6d8988b27ddcef1febd2be9c6235a0
Unchanged reported_roi_values.csv:
  ec4992e8068eccec16eb885dc62abd6a9cc9f17e1ad01071552f023fd3d69e11
Unchanged mouse_imaging_supplied.pdf:
  c3dd25bfd2d9cb36fa3c1a155835babce5634b185b3d51955e627e02a5593594

Original-archive check: latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
compiled a 40-page manuscript with pdfLaTeX (TeX Live 2024). Figure 2 and the
abstract were visually inspected. The chart can be rebuilt from the project
root with:
  pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory=figures figures/reported_roi_plot.tex

Original-archive verification: all required assets are included. A clean build
with latexmk and pdfLaTeX produced 40 pages, with Figure 1 on page 4 and Figure 2
on page 9. Both figure pages were visually inspected and are distinct.
