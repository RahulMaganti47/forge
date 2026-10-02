"""Build the matching manuscript and recoverable figure sources in a separate directory."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file
from forge.core.io import write_json


def build(root: Path, output: Path, *, figures: bool = False) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    source = root / "paper/source"
    shutil.copytree(source, output)
    inputs = {
        str(p.relative_to(root)): str(sha256_file(p))
        for p in sorted(source.rglob("*"))
        if p.is_file()
    }
    with (output / "build.log").open("w") as log:
        if figures:
            for name in ("forge_overview", "reaction_schemes", "reported_roi_plot"):
                subprocess.run(
                    [
                        "pdflatex",
                        "-interaction=nonstopmode",
                        "-halt-on-error",
                        "-output-directory=figures",
                        f"figures/{name}.tex",
                    ],
                    cwd=output,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            from .atlas import render

            render(root, output / "regenerated-atlas")
        subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=output,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    from pypdf import PdfReader

    reference = PdfReader(root / "paper/submission.pdf")
    rebuilt = PdfReader(output / "main.pdf")
    text_equal = [page.extract_text() for page in reference.pages] == [
        page.extract_text() for page in rebuilt.pages
    ]
    result = {
        "schema_version": "forge.release.manuscript_build.v1",
        "inputs": inputs,
        "reference_pdf_sha256": str(sha256_file(root / "paper/submission.pdf")),
        "rebuilt_pdf_sha256": str(sha256_file(output / "main.pdf")),
        "pages": len(rebuilt.pages),
        "reference_pages": len(reference.pages),
        "extracted_text_matches_reference": text_equal,
        "figure_sources_rebuilt": figures,
        "figure_2_composite": "preserved exact submitted PDF; composition source not recovered",
        "status": "pass" if text_equal and len(rebuilt.pages) == 35 else "mismatch",
    }
    write_json(output / "build_receipt.json", result)
    if result["status"] != "pass":
        raise ValueError(
            "rebuilt manuscript does not match reference text/page count; inspect build_receipt.json"
        )
    return result
