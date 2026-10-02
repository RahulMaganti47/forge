"""Reaggregate manuscript tables from authenticated, frozen evaluation records.

Common Ugi metrics use training-catalogue component novelty, distinct from
method-visible novelty and distinct-L1 yield. No training or generation runs here.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from forge.core.hashing import resolve_pin, sha256_file
from forge.core.io import write_json

METHODS = {
    "rgfn": "RGFN",
    "defog_unconditional": "DeFoG unconditional",
    "genmol_safe": "GenMol/SAFE",
    "learned_inventory_selector": "Learned inventory selector",
    "finite_catalogue_oracle": "Finite catalogue oracle",
    "shared_null_posthoc": "Shared-null + post-hoc",
    "fact_matched": "FACT-matched",
    "fact_generous": "FACT-generous",
    "forge_transformer": "FORGE",
}
COMMON_METRICS = (
    "valid_products_per_1000_attempts",
    "exact_l1_products_per_1000_attempts",
    "unique_exact_l1_products_per_1000_attempts",
    "unique_open_ended_exact_l1_products_per_1000_attempts",
    "held_component_exact_l1_products_per_1000_attempts",
    "mean_pairwise_ecfp4_distance",
)


def summary(values: list[float | None], *, digits: int = 1) -> str:
    if all(value is None for value in values):
        return "N/E"
    if len(values) != 3 or any(value is None or not math.isfinite(value) for value in values):
        raise ValueError("expected three finite seed metrics, or three undefined metrics")
    return f"${statistics.mean(values):.{digits}f}\\pm{statistics.stdev(values):.{digits}f}$"


def _load(root: Path, pin: dict[str, Any]) -> dict[str, Any]:
    path = resolve_pin({key: pin[key] for key in ("path", "sha256")}, root, label="paper evidence")
    return json.loads(path.read_text())


def common_tables(root: Path) -> tuple[dict[int, list[str]], dict[str, Any]]:
    config = json.loads((root / "configs/reproduction/common_ugi.json").read_text())
    pins = config["common_assessments"]
    if set(pins) != set(METHODS):
        raise ValueError("common-Ugi method set changed")
    rows: dict[int, list[str]] = {2: [], 7: [], 8: []}
    evidence = {}
    for method, name in METHODS.items():
        records = [_load(root, pin) for pin in pins[method]]
        records.sort(key=lambda record: record["seed"])
        if [record["seed"] for record in records] != [20260825, 20260826, 20260827]:
            raise ValueError(f"seed vector changed: {method}")
        for record in records:
            assessment = record["common_assessment"]
            if (
                record["status"] != "pass"
                or assessment["attempts"] != 3072
                or assessment["method_id"]
                != ("shared_three_program_null" if method == "shared_null_posthoc" else method)
                or record["candidate_selection"] is not False
                or not all(record["gates"].values())
            ):
                raise ValueError(f"invalid frozen assessment: {method}")
        metrics = [record["common_assessment"]["metrics"] for record in records]
        evidence[method] = {"inputs": pins[method], "per_seed_metrics": metrics}
        cells = [
            summary([metric[key] for metric in metrics], digits=3 if i == 5 else 1)
            for i, key in enumerate(COMMON_METRICS)
        ]
        if method in {"rgfn", "finite_catalogue_oracle", "learned_inventory_selector"}:
            cells[3] = cells[3][:-1] + r"^{\dagger}$"
        rows[2].append(" & ".join([name, *cells]) + r" \\")
        for record, metric in zip(records, metrics, strict=True):
            cells = [
                "N/E" if metric[key] is None else f"{metric[key]:.{3 if i == 5 else 1}f}"
                for i, key in enumerate(COMMON_METRICS)
            ]
            rows[7].append(" & ".join([name, str(record["seed"]), *cells]) + r" \\")

        def fraction(metric: dict[str, Any], numerator: str, denominator: str) -> float | None:
            return 100 * metric[numerator] / metric[denominator] if metric[denominator] else None

        decomposition = [
            [metric["valid"] for metric in metrics],
            [
                (
                    100 * metric["retro_decomposition_coverage_among_valid"]
                    if metric["retro_decomposition_coverage_among_valid"] is not None
                    else None
                )
                for metric in metrics
            ],
            [
                (
                    100 * metric["retro_transform_precision"]
                    if metric["retro_transform_precision"] is not None
                    else None
                )
                for metric in metrics
            ],
            [fraction(metric, "valid_without_exact_decomposition", "valid") for metric in metrics],
            [fraction(metric, "ambiguous_exact_decompositions", "valid") for metric in metrics],
        ]
        rows[8].append(" & ".join([name, *(summary(vector) for vector in decomposition)]) + r" \\")
    return rows, evidence


def realism(root: Path) -> tuple[list[str], dict[str, Any]]:
    config = json.loads(
        (root / "configs/reproduction/gem_table7_lipid_realism_v1.json").read_text()
    )
    result = _load(root, config["aggregate"])
    if result["status"] != "pass" or result["candidate_selection"] is not False:
        raise ValueError("invalid realism aggregate")

    def cell(metrics: dict[str, Any], key: str, scale: float, digits: int) -> str:
        value = metrics[key]
        if value["status"] == "not_estimable":
            return "N/E"
        if value["status"] != "estimated":
            raise ValueError(f"invalid realism status: {key}")
        return f"${scale * value['mean']:.{digits}f}\\pm{scale * value['sample_standard_deviation']:.{digits}f}$"

    rows = []
    for method in config["method_order"]:
        metrics = result["methods"][method]["metrics"]
        cells = [METHODS[method], cell(metrics, "connected_fraction_per_attempt", 1000, 1)]
        for space in ("fingerprint", "descriptor"):
            cells.append(
                cell(metrics, f"{space}_manifold_precision_per_attempt", 1000, 1)
                + "/"
                + cell(metrics, f"{space}_manifold_coverage", 100, 2)
            )
        cells += [
            cell(metrics, "grouped_c2st_auc", 1, 3),
            cell(metrics, "effective_molecule_count", 1, 1),
            cell(metrics, "internal_diversity", 1, 3),
        ]
        rows.append(" & ".join(cells) + r" \\")
    return rows, {"input": config["aggregate"], "aggregate": result}


def hela(root: Path) -> tuple[list[str], dict[str, Any]]:
    pin = {
        "path": "results/phase1/ugi_high_potency_challenger_adjudication_v1/result.json",
        "sha256": "a508b45d455b7b164e54a8345b0dff1f8e4de928eec5e2f631e145dc0d981cc3",
    }
    record = _load(root, pin)
    gate = record["fresh_matched_terminal_gate"]
    lane = gate["authorized_tail_lane"]
    rows = []
    for name, key, prefix in (
        ("Support-enriched", "support_enriched", "support"),
        ("Nested potency proposal", "nested_potency", "potency"),
    ):
        cells = [
            name,
            gate["generator_calls_per_arm"],
            gate["valid_exact_l1"][key],
            lane[f"{prefix}_eligible_before_budget"],
            gate["equal_oracle_budget"],
            lane[f"{prefix}_unique_conservative_high"],
        ]
        rows.append(" & ".join(map(str, cells)) + r" \\")
    return rows, {
        "input": pin,
        "adjudication": record,
        "scope": "summary replay; upstream four records remain unavailable",
    }


def reproduce(root: Path, output: Path, target: str = "all") -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    output.mkdir(parents=True)
    from forge.reporting.gem_table1 import render_gem_table1_final_evidence
    from forge.reporting.gem_table4 import render_gem_table4_production_comparison
    from forge.reporting.gem_table5 import render_gem_table5_decoder_source_ablation
    from forge.reporting.gem_table6 import render_gem_table6_exact_l1_counts
    from forge.reporting.gem_table8 import render_gem_table8_architecture_ablations
    from forge.reporting.gem_table9 import render_gem_table9_catalogue_comparison

    renderers = {
        1: (render_gem_table1_final_evidence, "gem_table1_core_saturation_complete_v1"),
        3: (render_gem_table4_production_comparison, "gem_table1_core_saturation_complete_v1"),
        4: (render_gem_table6_exact_l1_counts, "gem_table1_core_saturation_final_v1"),
        5: (render_gem_table5_decoder_source_ablation, "gem_table5_decoder_source_ablation_v1"),
        6: (render_gem_table8_architecture_ablations, "gem_table8_architecture_ablations_v1"),
        9: (render_gem_table9_catalogue_comparison, "gem_table9_catalogue_comparison_v1"),
    }
    targets = range(1, 12) if target == "all" else [int(target.removeprefix("table-"))]
    evidence = {}
    try:
        for number in targets:
            rows_path = output / f"table-{number}.tex"
            if number in renderers:
                renderer, config = renderers[number]
                evidence[str(number)] = renderer(
                    root / f"configs/reproduction/{config}.json",
                    root,
                    output if number == 1 else rows_path,
                    result_path=output / f"table-{number}.json",
                )
                if number == 1:
                    rows_path.write_bytes((output / "shared_program_figure_rows.tex").read_bytes())
            else:
                if number in (2, 7, 8):
                    common, inputs = common_tables(root)
                    rows = common[number]
                elif number == 10:
                    rows, inputs = realism(root)
                elif number == 11:
                    rows, inputs = hela(root)
                else:
                    raise ValueError(f"unsupported numerical table: {number}")
                rows_path.write_text("\n".join(rows) + "\n")
                evidence[str(number)] = inputs
                write_json(output / f"table-{number}.json", inputs)
        receipt = {
            "schema_version": "forge.release.table_replay.v1",
            "status": "pass",
            "scope": "reaggregation of frozen evidence, not model retraining",
            "reference_pdf_sha256": str(sha256_file(root / "paper/submission.pdf")),
            "outputs": {p.name: str(sha256_file(p)) for p in sorted(output.iterdir())},
        }
        write_json(output / "receipt.json", receipt)
        return receipt
    except Exception as error:
        write_json(output / "FAILED.json", {"status": "failed", "error": str(error)})
        raise
