"""M0-08 liver functional-editing endpoint stub."""

from __future__ import annotations

from forge.potency.endpoint import Endpoint, EndpointSpecification, Readout


class LiverFunctionalEditingEndpoint(Endpoint):
    """Evidence contract for intravenous functional editing in liver."""

    @property
    def specification(self) -> EndpointSpecification:
        return EndpointSpecification(
            endpoint_id="liver_functional_editing",
            display_name="Intravenous functional liver editing",
            administration_route="intravenous",
            bridge_rationale=(
                "AGILE HeLa and RAW 264.7 measurements are general-transfection evidence, "
                "not hepatocyte or liver-editing labels. A hepatocyte-relevant bridge is "
                "required before in vivo advancement."
            ),
            readouts=(
                Readout(
                    readout_id="hepatocyte_expression_or_editing",
                    stage="bridge",
                    priority="required",
                    description=(
                        "Expression or sequence-verified editing in a hepatocyte-relevant "
                        "system under the intended formulation."
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
                    readout_id="sequence_verified_liver_editing",
                    stage="in_vivo",
                    priority="required",
                    description=(
                        "Sequence-verified editing at a reporter locus or endogenous target in "
                        "liver after intravenous dosing."
                    ),
                ),
                Readout(
                    readout_id="reporter_delivery_and_biodistribution",
                    stage="in_vivo",
                    priority="required",
                    description=(
                        "Reporter delivery to liver and major-organ distribution for the selected "
                        "panel under a predeclared dose and schedule."
                    ),
                ),
                Readout(
                    readout_id="dose_response",
                    stage="in_vivo",
                    priority="recommended",
                    description="Editing and expression measured across a predeclared dose range.",
                ),
                Readout(
                    readout_id="benchmark_comparison",
                    stage="in_vivo",
                    priority="required",
                    description="Comparison with a declared industry or laboratory benchmark LNP.",
                ),
                Readout(
                    readout_id="basic_tolerability",
                    stage="safety",
                    priority="required",
                    description=(
                        "Basic tolerability appropriate to the selected dose and schedule, with "
                        "expanded clinical chemistry or histopathology added when the claim requires it."
                    ),
                ),
                Readout(
                    readout_id="disease_relevant_editing",
                    stage="in_vivo",
                    priority="recommended",
                    description=("Endogenous or disease-relevant editing beyond a reporter locus."),
                ),
            ),
            success_definition=(
                "At least two structurally distinct FORGE lipids deliver the reporter in vivo.",
                "At least one lead achieves sequence-verified editing.",
                "Benchmark comparison, biodistribution, and basic tolerability support the claim.",
            ),
            disallowed_inferences=(
                "HeLa or RAW 264.7 potency alone does not establish liver editing.",
                "Predicted apparent pKa or particle properties cannot replace prospective measurement.",
                "Reporter expression alone establishes delivery, not sequence-verified editing.",
                "Reporter-locus editing does not establish disease correction.",
            ),
        )
