"""M0-08 intramuscular functional-editing endpoint stub."""

from __future__ import annotations

from forge.potency.endpoint import Endpoint, EndpointSpecification, Readout


class IntramuscularFunctionalEditingEndpoint(Endpoint):
    """Evidence contract for intramuscular delivery followed by editing."""

    @property
    def specification(self) -> EndpointSpecification:
        return EndpointSpecification(
            endpoint_id="im_functional_editing",
            display_name="Intramuscular reporter delivery and functional editing",
            administration_route="intramuscular",
            bridge_rationale=(
                "AGILE directly evaluated HeLa-to-intramuscular reporter translation in a "
                "selected lipid panel. This is stronger endpoint alignment than is currently "
                "available for intravenous liver delivery, but it must still be tested "
                "prospectively for generated components."
            ),
            readouts=(
                Readout(
                    readout_id="muscle_relevant_expression",
                    stage="bridge",
                    priority="required",
                    description=(
                        "Reporter expression in a muscle-relevant system under the intended "
                        "formulation."
                    ),
                ),
                Readout(
                    readout_id="formulation_quality",
                    stage="bridge",
                    priority="required",
                    description=(
                        "Prospectively measured particle size, polydispersity, encapsulation, "
                        "and formulation robustness under a predeclared protocol."
                    ),
                ),
                Readout(
                    readout_id="intramuscular_reporter_delivery",
                    stage="in_vivo",
                    priority="required",
                    description=(
                        "Reporter expression at the injection site and major-organ distribution "
                        "for the selected panel."
                    ),
                ),
                Readout(
                    readout_id="functional_muscle_editing",
                    stage="in_vivo",
                    priority="required",
                    description=(
                        "Cre reporter conversion or sequence-verified reporter-locus or "
                        "endogenous editing in muscle for at least one lead."
                    ),
                ),
                Readout(
                    readout_id="benchmark_comparison",
                    stage="in_vivo",
                    priority="required",
                    description="Comparison with a declared industry or laboratory benchmark LNP.",
                ),
                Readout(
                    readout_id="dose_response",
                    stage="in_vivo",
                    priority="recommended",
                    description="Reporter delivery or editing across a predeclared dose range.",
                ),
                Readout(
                    readout_id="basic_tolerability",
                    stage="safety",
                    priority="required",
                    description=(
                        "Basic local and systemic tolerability appropriate to the selected dose "
                        "and schedule."
                    ),
                ),
            ),
            success_definition=(
                "At least two structurally distinct FORGE lipids deliver reporter mRNA in vivo.",
                "At least one lead demonstrates functional editing in muscle.",
                "Benchmark comparison, distribution, and basic tolerability support the claim.",
            ),
            disallowed_inferences=(
                "The reported HeLa-to-IM correlation is not a universal transfer guarantee.",
                "Reporter expression alone establishes delivery, not genome editing.",
                "Cre reporter conversion does not establish endogenous or disease correction.",
                "Predicted particle properties cannot replace prospective formulation measurement.",
            ),
        )
