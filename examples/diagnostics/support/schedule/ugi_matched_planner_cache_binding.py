"""Lazy planner-cache binding for matched diagnostic route assessments.

The hash-pinned matched-budget runner derives arm clone identifiers internally
and exposes them only through ``MatchedAssessmentContext`` callbacks.  This
module therefore seals both empty overlay roots before generation, then binds
each arm lazily to the clone identifier supplied by its first callback.  It
does not reproduce the runner's schedule or clone-ID hash formulas.

This layer owns cache provenance only.  It does not run a generator or route
source, define a synthesis scalar, guide sampling, invoke biology, or select a
candidate.
"""

from __future__ import annotations

import re

MATCHED_PLANNER_CACHE_PREFLIGHT_SCHEMA_VERSION = "forge.matched_planner_cache_binding_preflight.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
