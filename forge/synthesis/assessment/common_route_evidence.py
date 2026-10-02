"""Apply one frozen exact-evidence route index to common Ugi benchmark outputs.

This is intentionally a lookup-only assessment.  It does not call a planner after seeing a method's
outputs and cannot promote similarity, reaction-family projection, or public-stock search into route
evidence.  Components absent from the frozen index abstain.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.core.io import iter_jsonl


class CommonRouteEvidenceError(ValueError):
    """The common assessed rows or frozen route evidence are malformed."""


def load_frozen_component_evidence(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """Load exact role/constitution evidence from the frozen bounded-cascade ledger."""

    evidence: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in iter_jsonl(path):
        if not isinstance(raw, Mapping):
            raise CommonRouteEvidenceError("component evidence row must be an object")
        role = raw.get("role")
        smiles = raw.get("canonical_smiles")
        exact = raw.get("exact_evidence")
        state = raw.get("final_component_state")
        if (
            not isinstance(role, str)
            or not role
            or not isinstance(smiles, str)
            or not smiles
            or not isinstance(exact, Mapping)
            or state not in {"complete", "unresolved", "search_censored"}
        ):
            raise CommonRouteEvidenceError("component evidence schema changed")
        key = (role, smiles)
        if key in evidence:
            raise CommonRouteEvidenceError(f"duplicate component evidence: {key}")
        strict = exact.get("strict_complete")
        if not isinstance(strict, bool) or (state == "complete") != strict:
            raise CommonRouteEvidenceError("strict route closure and final state disagree")
        verified_upstream = exact.get("verified_upstream", strict)
        terminal_evidence = exact.get(
            "terminal_evidence", int(exact.get("current_terminal_leaf_count", 0)) > 0
        )
        explicit_abstention = exact.get("explicit_abstention", not strict)
        if not all(
            isinstance(value, bool)
            for value in (verified_upstream, terminal_evidence, explicit_abstention)
        ):
            raise CommonRouteEvidenceError("route-evidence disposition flags must be boolean")
        if strict != (verified_upstream and terminal_evidence) or explicit_abstention == strict:
            raise CommonRouteEvidenceError("route-evidence disposition flags are inconsistent")
        component_value = exact.get("component_value")
        evidence[key] = {
            "assessment_outcome": exact.get("assessment_outcome"),
            "strict_complete": strict,
            "verified_upstream": verified_upstream,
            "terminal_evidence": terminal_evidence,
            "explicit_abstention": explicit_abstention,
            "current_terminal_leaf_count": int(exact.get("current_terminal_leaf_count", 0)),
            "forward_consistency": (
                component_value.get("forward_consistency")
                if isinstance(component_value, Mapping)
                else None
            ),
            "final_component_state": state,
        }
    if not evidence:
        raise CommonRouteEvidenceError("frozen component evidence is empty")
    return evidence


def assess_common_route_evidence(
    rows: Sequence[Mapping[str, Any]],
    *,
    component_evidence: Mapping[tuple[str, str], Mapping[str, Any]],
    evidence_scope: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assess unique exact-L1 traces against the same frozen L2/L3 evidence index."""

    if not rows:
        raise CommonRouteEvidenceError("route assessment requires common assessed attempts")
    assessed: list[dict[str, Any]] = []
    dispositions: Counter[str] = Counter()
    for raw in rows:
        row = dict(raw)
        route = {
            "eligible_unique_exact_l1": False,
            "evidence_index_covered": False,
            "verified_upstream": False,
            "terminal_evidence": False,
            "complete_dossier": False,
            "abstained": False,
            "component_assessments": [],
            "disposition": "not_unique_exact_l1",
        }
        if row.get("exact_l1_program") is True and int(row.get("exact_l1_trace_count", 0)) == 1:
            route["eligible_unique_exact_l1"] = True
            trace = row.get("exact_l1_traces")
            if not isinstance(trace, list) or not isinstance(trace[0], Mapping):
                raise CommonRouteEvidenceError("exact-L1 trace payload is malformed")
            components = trace[0].get("components_by_role")
            if not isinstance(components, Mapping) or not components:
                raise CommonRouteEvidenceError("exact-L1 component mapping is malformed")
            component_rows = []
            all_covered = True
            all_verified_upstream = True
            all_terminal = True
            all_complete = True
            any_abstention = False
            for role, value in sorted(components.items()):
                values = value if isinstance(value, list) else [value]
                for smiles in values:
                    key = (str(role), str(smiles))
                    evidence = component_evidence.get(key)
                    covered = evidence is not None
                    complete = bool(evidence and evidence.get("strict_complete") is True)
                    upstream = bool(evidence and evidence.get("verified_upstream") is True)
                    terminal = bool(evidence and evidence.get("terminal_evidence") is True)
                    abstained = bool(
                        evidence is None or evidence.get("explicit_abstention") is True
                    )
                    component_rows.append(
                        {
                            "role": key[0],
                            "canonical_smiles": key[1],
                            "covered": covered,
                            "verified_upstream": upstream,
                            "strict_complete": complete,
                            "terminal_evidence": terminal,
                            "abstained": abstained,
                            "final_component_state": (
                                evidence.get("final_component_state")
                                if evidence
                                else "absent_from_evidence_union"
                            ),
                            "assessment_outcome": (
                                evidence.get("assessment_outcome") if evidence else "not_in_index"
                            ),
                        }
                    )
                    all_covered &= covered
                    all_verified_upstream &= upstream
                    all_terminal &= terminal
                    all_complete &= complete
                    any_abstention |= abstained
            route["component_assessments"] = component_rows
            route["evidence_index_covered"] = all_covered
            route["verified_upstream"] = all_covered and all_verified_upstream
            route["terminal_evidence"] = all_covered and all_terminal
            route["complete_dossier"] = all_covered and all_complete
            route["abstained"] = any_abstention
            route["disposition"] = (
                "complete_dossier"
                if route["complete_dossier"]
                else (
                    "explicit_evidence_abstention"
                    if all_covered
                    else "abstain_component_absent_from_frozen_index"
                )
            )
        dispositions[str(route["disposition"])] += 1
        row["common_route_evidence"] = route
        assessed.append(row)
    eligible = sum(
        bool(row["common_route_evidence"]["eligible_unique_exact_l1"]) for row in assessed
    )
    covered_count = sum(
        bool(row["common_route_evidence"]["evidence_index_covered"]) for row in assessed
    )
    upstream_count = sum(
        bool(row["common_route_evidence"]["verified_upstream"]) for row in assessed
    )
    terminal_count = sum(
        bool(row["common_route_evidence"]["terminal_evidence"]) for row in assessed
    )
    complete_count = sum(bool(row["common_route_evidence"]["complete_dossier"]) for row in assessed)
    abstained = sum(bool(row["common_route_evidence"]["abstained"]) for row in assessed)
    result = {
        "schema_version": "forge.common_ugi_route_evidence_assessment.v1",
        "attempts": len(assessed),
        "eligible_unique_exact_l1": eligible,
        "evidence_index_covered": covered_count,
        "evidence_index_coverage_among_eligible": (covered_count / eligible if eligible else None),
        "verified_upstream": upstream_count,
        "terminal_evidence": terminal_count,
        "complete_dossier": complete_count,
        "complete_dossier_fraction_among_eligible": (
            complete_count / eligible if eligible else None
        ),
        "abstentions": abstained,
        "dispositions": dict(sorted(dispositions.items())),
        "route_or_oracle_calls": 0,
        "assessment_mode": (
            "exact_role_constitution_lookup_in_frozen_method_blind_union"
            if evidence_scope is not None
            and evidence_scope.get("method_blind_cross_method_union") is True
            else "exact_role_constitution_lookup_in_frozen_bounded_cascade"
        ),
        "evidence_scope": (
            dict(evidence_scope)
            if evidence_scope is not None
            else {
                "population": "bounded_forge_ugi_candidate_cascade",
                "method_blind_cross_method_union": False,
                "synthesis_success_comparison_authorized": False,
            }
        ),
        "nonclaims": [
            "An index miss is an abstention, not proof of unsynthesizability.",
            "Exact L1 replay is not a synthesis-success probability.",
            "This lookup-only comparison does not rerun or retune a planner per method.",
            "This bounded index cannot support a cross-method synthesis-success ranking.",
        ],
    }
    return assessed, result


__all__ = [
    "CommonRouteEvidenceError",
    "assess_common_route_evidence",
    "load_frozen_component_evidence",
]
