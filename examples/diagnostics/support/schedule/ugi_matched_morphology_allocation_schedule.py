"""Freeze the matched three-arm morphology-allocation generation schedule.

The schedule is the experimental design, not a terminal outcome.  A single
predeclared vector of uniforms is inverse-CDF transformed through the broad,
promoted-support and nested-potency morphology distributions.  This gives the
three arms common random numbers without pretending that they select the same
morphology program.
"""

from __future__ import annotations

CONFIG_SCHEMA_VERSION = "phase1_ugi_matched_morphology_allocation_schedule_config.v1"
RESULT_SCHEMA_VERSION = "phase1_ugi_matched_morphology_allocation_schedule.v1"
SCHEDULE_SCHEMA_VERSION = "forge.ugi_matched_morphology_allocation_schedule.v1"
ARM_IDS = ("broad_prior", "support_enriched", "nested_potency")
EXPECTED_SCOPE = {
    "read_only_distribution_sampling": True,
    "complete_qualified_morphology_support": True,
    "common_uniforms_across_arms": True,
    "terminal_outcomes_consumed": False,
    "generator_calls": 0,
    "oracle_calls": 0,
    "route_calls": 0,
    "synthesis_calls": 0,
    "candidate_selection": False,
    "sealed_holdout_access": False,
}
EXPECTED_INPUTS = {
    "potency_proposal_ledger",
    "potency_proposal_result",
    "promoted_proposal_ledger",
    "promoted_proposal_result",
    "runner",
    "source",
    "tests",
}


__all__ = ["ARM_IDS"]
