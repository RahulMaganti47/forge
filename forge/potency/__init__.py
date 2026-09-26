"""Potency prediction: how well a lipid is likely to work, and whether we may say so.

    oracle/         fitting and freezing the predictors
    applicability/  whether a candidate is inside the calibrated domain

Applicability gates live beside the predictors: a prediction outside the calibrated domain
abstains rather than scoring low. Phase-specific diagnostics, proposals, and rankings live under
``experiments/`` because they are workflows, not reusable endpoint APIs.
"""

from forge.potency.endpoint import (
    Endpoint,
    EndpointError,
    EndpointSpecification,
    Readout,
    endpoint_ids,
    load_endpoint,
)

__all__ = [
    "Endpoint",
    "EndpointError",
    "EndpointSpecification",
    "Readout",
    "endpoint_ids",
    "load_endpoint",
]
