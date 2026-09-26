"""M0-08 intramuscular-vaccination endpoint stub."""

from __future__ import annotations

from forge.potency.endpoint import Endpoint, EndpointSpecification, Readout


class IntramuscularVaccinationEndpoint(Endpoint):
    """Evidence contract for intramuscular mRNA vaccination."""

    @property
    def specification(self) -> EndpointSpecification:
        return EndpointSpecification(
            endpoint_id="im_vaccination",
            display_name="Intramuscular mRNA vaccination",
            administration_route="intramuscular",
            bridge_rationale=(
                "AGILE, JC_2023, and Miao 2019 connect HeLa screening to intramuscular "
                "expression or vaccine studies. This supports delivery transfer, but HeLa "
                "and RAW 264.7 do not directly predict antigen-specific immunity."
            ),
            readouts=(
                Readout(
                    readout_id="endpoint_relevant_expression",
                    stage="bridge",
                    priority="required",
                    description=(
                        "Expression in at least one predeclared muscle-relevant or antigen-presenting "
                        "cell system under the intended formulation."
                    ),
                ),
                Readout(
                    readout_id="innate_activation_profile",
                    stage="bridge",
                    priority="recommended",
                    description=(
                        "Innate activation and cytokine measurements that distinguish delivery "
                        "from excessive or chemistry-specific adjuvant activity."
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
                    readout_id="intramuscular_expression_and_localization",
                    stage="in_vivo",
                    priority="required",
                    description=(
                        "Expression at the injection site with major-organ biodistribution and "
                        "off-target expression."
                    ),
                ),
                Readout(
                    readout_id="antigen_specific_response",
                    stage="in_vivo",
                    priority="required",
                    description=(
                        "At least one predeclared antigen-specific humoral or cellular response "
                        "appropriate to the vaccine goal."
                    ),
                ),
                Readout(
                    readout_id="second_adaptive_immunity_arm",
                    stage="in_vivo",
                    priority="recommended",
                    description=(
                        "A second antigen-specific adaptive-immunity arm complementary to the "
                        "required response."
                    ),
                ),
                Readout(
                    readout_id="functional_efficacy",
                    stage="in_vivo",
                    priority="recommended",
                    description=(
                        "Challenge protection or therapeutic efficacy in the laboratory's "
                        "established vaccine model."
                    ),
                ),
                Readout(
                    readout_id="basic_tolerability",
                    stage="safety",
                    priority="required",
                    description=(
                        "Basic local and systemic tolerability appropriate to the selected dose and "
                        "schedule, with expanded toxicology added when the claim requires it."
                    ),
                ),
            ),
            success_definition=(
                "At least two structurally distinct FORGE lipids support functional IM expression.",
                "At least one predeclared antigen-specific response meets the benchmark.",
                "Basic tolerability remains acceptable under predeclared advancement rules.",
                "A second immune arm, durability, and functional efficacy strengthen the claim.",
            ),
            disallowed_inferences=(
                "HeLa or RAW 264.7 potency alone does not establish vaccine immunogenicity.",
                "Innate activation alone is not antigen-specific vaccine efficacy.",
                "IM reporter expression alone establishes delivery, not vaccination.",
                "Predicted particle properties cannot replace prospective formulation measurement.",
            ),
        )
