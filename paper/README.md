# Authoritative manuscript

`submission.pdf` is the exact 35-page submission supplied for this release:
`7ea7cf2e51922e5e58cd6659dc183409e465ede2eb1586fc8d0ec39935c4fd53`.

`source/main.tex` matches that submission:
`36aba6ea556cc6b62a436d3b2df48d79e39813773f549592df053a163f78fafa`.
The bibliography, local style files, required figures, editable figure sources and correction receipts
are included. The matching source was recovered from the development folder named `iclr_final_3`;
identity is established by the PDF/source hashes and compiled text, not that folder name.

`source/README.txt` is an inherited archive note describing earlier assembly and editorial work. Its
earlier page counts and verification status are historical. Current release checks are recorded in
`../provenance/qualification/manuscript_build.json` and `../docs/REPRODUCTION.md`.

Run `forge paper --rebuild-figures --output results/manuscript` from the checkout. The command compiles
a separate copy, checks the page count and extracted text, and leaves this source unchanged. It
rebuilds the recoverable TikZ/ROI and molecular-depiction sources; the exact Figure 2 composite is
preserved because its authoring script is missing. No scientific wording or data is revised here.
