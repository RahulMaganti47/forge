"""Draw eight source lipids and verified L1 precursors in one skeletal vector style.

Run from the FORGE repository root:
PYTHONPATH=. .venv/bin/python paper/iclr_final_3/figures/lipid_l1_building_blocks/render.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from forge.assembly.ugi3 import Ugi3AssemblyAdapter

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
REPO = PROJECT.parents[1]
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def molecule_drawing(
    smiles: str, width: int, height: int, *, fit_labels: bool = False
) -> ET.Element:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecular structure")
    original = Chem.MolToSmiles(mol)
    rdDepictor.Compute2DCoords(mol, canonOrient=True, clearConfs=True)
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    options = drawer.drawOptions()
    options.padding = 0.045
    options.bondLineWidth = 2.4
    if fit_labels:
        options.minFontSize = 26
        options.maxFontSize = 26
    else:
        options.fixedFontSize = 35
    options.multipleBondOffset = 0.14
    options.useBWAtomPalette()
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    if Chem.MolToSmiles(mol) != original:
        raise ValueError("Depiction changed molecular identity")
    return ET.fromstring(drawer.GetDrawingText())


def panel(components: dict[str, str]) -> ET.Element:
    root = ET.Element(f"{{{SVG}}}svg", width="1000", height="620", viewBox="0 0 1000 620")
    ET.SubElement(root, f"{{{SVG}}}rect", width="100%", height="100%", fill="white")
    labels = (
        ("amine_head", "Amine"),
        ("oxoester_aldehyde_body_tail", "Aldehyde"),
        ("isocyanide_tail", "Isocyanide"),
    )
    for index, (role, title) in enumerate(labels):
        top = 205 * index
        label = ET.SubElement(
            root,
            f"{{{SVG}}}text",
            x="20",
            y=str(top + 35),
            fill="#303030",
            attrib={"font-family": "Times New Roman, serif", "font-size": "38"},
        )
        label.text = title
        drawing = molecule_drawing(components[role], 970, 160)
        drawing.set("x", "15")
        drawing.set("y", str(top + 42))
        root.append(drawing)
    return root


def main() -> None:
    source = PROJECT / "lipid_atlas_transfer_receipt.json"
    pin_path = REPO / "configs/corpus/m0_04_source_platform_registry_audit.json"
    registry = REPO / "data/vendor/qualified_reactions_v1.json"
    expected = json.loads(pin_path.read_text())["expected_inputs"][registry.name]
    adapter = Ugi3AssemblyAdapter.from_registry(registry, expected_sha256=expected)
    source_data = json.loads(source.read_text())
    rows = source_data["rows"]
    if [row["display_order"] for row in rows] != list(range(1, 9)):
        raise ValueError("The eight source lipid identities or display order changed")
    verified = []
    for row in rows:
        product = row["canonical_constitutional_smiles"]
        candidates = adapter.transform_consistent_decomposition_candidates(product)
        admitted = [candidate for candidate in candidates if candidate.registry_handle_qualified]
        if len(admitted) != 1:
            raise ValueError(
                f"Row {row['display_order']} has {len(admitted)} qualified decompositions"
            )
        candidate = admitted[0]
        components = dict(candidate.components)
        check = adapter.check_forward(components, product)
        products = adapter.forward_products(components)
        if not check.exact or check.saturated or products.products != (product,):
            raise ValueError(f"Row {row['display_order']} fails unique exact forward replay")
        molecule = Chem.MolFromSmiles(product)
        reversed_atoms = Chem.RenumberAtoms(molecule, list(reversed(range(molecule.GetNumAtoms()))))
        variants = (
            Chem.MolToSmiles(molecule, rootedAtAtom=1),
            Chem.MolToSmiles(reversed_atoms, canonical=False),
        )
        for variant in variants:
            if [trace.as_mapping() for trace in adapter.decompose(variant)] != [components]:
                raise ValueError("Decomposition depends on atom ordering")
        verified.append(
            {
                "display_order": row["display_order"],
                "canonical_constitutional_smiles": product,
                "components_by_role": components,
                "handle_assessments": [item.to_mapping() for item in candidate.handle_assessments],
                "unique_qualified_decompositions": len(admitted),
                "forward_check": asdict(check),
                "unique_forward_products": list(products.products),
                "atom_order_invariance_passed": True,
                "evidence_basis": "computed_transform_consistency",
                "disposition": "admit_transform_consistency",
            }
        )
    outputs = {}
    for row in verified:
        drawings = {
            "precursors": panel(row["components_by_role"]),
            # Match the column-width ratio for equal printed line weights. Scale
            # product labels to fit crowded regions of the larger molecular graph.
            "product": molecule_drawing(
                row["canonical_constitutional_smiles"], 1275, 400, fit_labels=True
            ),
        }
        for kind, drawing in drawings.items():
            name = f"lipid_{row['display_order']:02d}_{kind}"
            svg = ROOT / f"{name}.svg"
            svg.write_bytes(ET.tostring(drawing, encoding="utf-8", xml_declaration=True))
            pdf = svg.with_suffix(".pdf")
            subprocess.run(
                ["rsvg-convert", "--format=pdf", "--output", str(pdf), str(svg)], check=True
            )
            outputs.update({str(path.relative_to(PROJECT)): sha(path) for path in (svg, pdf)})
    source_files = [
        "forge/assembly/ugi3.py",
        "forge/assembly/program.py",
        "forge/corpus/ugi_held_component_gate.py",
        "forge/corpus/r1_prime_audit.py",
        "forge/chemistry/reactive_sites.py",
    ]
    receipt = {
        "scope": "Matched skeletal vector drawings of eight fixed source lipids and verified L1 building blocks; no reselection or experimental claim.",
        "inputs": {
            "source_pdf_sha256": source_data["source_pdf"]["sha256"],
            "source_rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True).encode()
            ).hexdigest(),
            str(registry.relative_to(REPO)): sha(registry),
            str(pin_path.relative_to(REPO)): sha(pin_path),
            **{path: sha(REPO / path) for path in source_files},
        },
        "renderer_sha256": sha(Path(__file__)),
        "reaction_id": adapter.reaction_id,
        "rdkit_version": rdBase.rdkitVersion,
        "svg_converter": subprocess.check_output(["rsvg-convert", "--version"], text=True).strip(),
        "seed": 0,
        "randomness_used": False,
        "drawing_style": {
            "atom_palette": "black and white",
            "bond_line_width": 2.4,
            "precursor_font_size": 35,
            "product_font_size_bounds": [26, 26],
            "product_canvas": [1275, 400],
            "precursor_panel_canvas": [1000, 620],
            "product_to_precursor_column_width_ratio": 1.275,
            "all_depicted_molecular_identities_preserved": True,
        },
        "rows": verified,
        "outputs": outputs,
    }
    (ROOT / "l1_building_blocks_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Verified and rendered {len(verified)} unique, handle-qualified L1 decompositions.")


if __name__ == "__main__":
    main()
