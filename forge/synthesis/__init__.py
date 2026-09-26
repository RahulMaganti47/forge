"""Synthesis routing: from a target molecule to an auditable path to terminal materials.

Split by the question each group answers rather than by chronology:

    engine/      the retrosynthesis models and planner that propose routes
    sources/     where route knowledge comes from, and what each source supports
    evidence/    whether a proposed transformation is actually supported
    terminals/   the recursive closure toward purchasable materials
    assessment/  reading the state of a route without closing it
    value/       structured, evidence-preserving synthesis values

The ordering is a dependency gradient: a proposal from `engine` is a hypothesis until `evidence`
admits it and `terminals` closes it. Nothing in this package may treat route likelihood as a
synthesis-success probability.
"""
