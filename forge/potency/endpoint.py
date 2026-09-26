"""Endpoint-generic interfaces for FORGE biological validation.

M0-08 defines the decision boundary and required evidence without selecting an
endpoint in code. Downstream callers must request an endpoint explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal

EndpointId = Literal[
    "im_functional_editing",
    "im_vaccination",
    "liver_functional_editing",
]
ReadoutStage = Literal["bridge", "in_vivo", "safety"]
ReadoutPriority = Literal["required", "recommended"]


class EndpointError(ValueError):
    """Raised when an endpoint specification or request is invalid."""


@dataclass(frozen=True)
class Readout:
    """One declared assay or outcome in an endpoint evidence package."""

    readout_id: str
    stage: ReadoutStage
    priority: ReadoutPriority
    description: str
    prospective_only: bool = True

    def validate(self) -> None:
        if not self.readout_id or not self.readout_id.replace("_", "").isalnum():
            raise EndpointError(f"invalid readout identifier: {self.readout_id!r}")
        if not self.description.strip():
            raise EndpointError(f"readout {self.readout_id!r} needs a description")


@dataclass(frozen=True)
class EndpointSpecification:
    """Frozen biological evidence contract for one endpoint option."""

    endpoint_id: EndpointId
    display_name: str
    administration_route: str
    bridge_rationale: str
    readouts: tuple[Readout, ...]
    success_definition: tuple[str, ...]
    disallowed_inferences: tuple[str, ...]

    def validate(self) -> None:
        if not self.display_name.strip() or not self.administration_route.strip():
            raise EndpointError(f"endpoint {self.endpoint_id!r} lacks identifying metadata")
        if not self.bridge_rationale.strip():
            raise EndpointError(f"endpoint {self.endpoint_id!r} lacks a bridge rationale")
        if not self.readouts:
            raise EndpointError(f"endpoint {self.endpoint_id!r} has no readouts")
        identifiers = [readout.readout_id for readout in self.readouts]
        if len(identifiers) != len(set(identifiers)):
            raise EndpointError(f"endpoint {self.endpoint_id!r} contains duplicate readout ids")
        for readout in self.readouts:
            readout.validate()
        required_stages = {
            readout.stage for readout in self.readouts if readout.priority == "required"
        }
        if required_stages != {"bridge", "in_vivo", "safety"}:
            raise EndpointError(
                f"endpoint {self.endpoint_id!r} must require bridge, in-vivo, and safety evidence"
            )
        if not self.success_definition:
            raise EndpointError(f"endpoint {self.endpoint_id!r} lacks a success definition")
        if not self.disallowed_inferences:
            raise EndpointError(f"endpoint {self.endpoint_id!r} lacks inference limits")


class Endpoint(ABC):
    """Interface implemented by an endpoint-specific M0 evidence stub."""

    @property
    @abstractmethod
    def specification(self) -> EndpointSpecification:
        """Return the validated endpoint evidence contract."""

    def required_readout_ids(self) -> tuple[str, ...]:
        """Return required readouts without inventing prospective measurements."""

        specification = self.specification
        specification.validate()
        return tuple(
            readout.readout_id
            for readout in specification.readouts
            if readout.priority == "required"
        )

    def missing_required_readouts(self, observed: Mapping[str, Any]) -> tuple[str, ...]:
        """Report absent readout keys without judging scientific success."""

        return tuple(
            readout_id for readout_id in self.required_readout_ids() if readout_id not in observed
        )


_ENDPOINT_IMPLEMENTATIONS: dict[EndpointId, tuple[str, str]] = {
    "im_functional_editing": (
        "forge.potency.endpoints.muscle",
        "IntramuscularFunctionalEditingEndpoint",
    ),
    "liver_functional_editing": (
        "forge.potency.endpoints.liver",
        "LiverFunctionalEditingEndpoint",
    ),
    "im_vaccination": (
        "forge.potency.endpoints.vaccine",
        "IntramuscularVaccinationEndpoint",
    ),
}


def endpoint_ids() -> tuple[EndpointId, ...]:
    """List supported endpoint stubs. The order does not imply a default."""

    return tuple(sorted(_ENDPOINT_IMPLEMENTATIONS))


def load_endpoint(endpoint_id: EndpointId) -> Endpoint:
    """Load an endpoint explicitly. No implicit default is permitted."""

    try:
        module_name, class_name = _ENDPOINT_IMPLEMENTATIONS[endpoint_id]
    except KeyError as exc:
        raise EndpointError(
            f"unknown endpoint {endpoint_id!r}; choose one of {endpoint_ids()}"
        ) from exc
    module = import_module(module_name)
    endpoint_class = getattr(module, class_name)
    endpoint = endpoint_class()
    if not isinstance(endpoint, Endpoint):
        raise EndpointError(f"{module_name}.{class_name} does not implement Endpoint")
    endpoint.specification.validate()
    return endpoint
