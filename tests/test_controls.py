import json
from pathlib import Path

from forge.workflows.mechanism_study import _study_arms

ROOT = Path(__file__).resolve().parents[1]
PROGRAMS = (
    "ugi_3cr_agile",
    "bl_2023_repeated_aza_michael",
    "lx_2024_repeated_reductive_amination",
)


def test_frozen_shared_null_and_cyclic_configs_select_separate_controls() -> None:
    for name, expected, conditioning in (
        (
            "shared_bias_shared_null_core_saturation_v1",
            "shared_three_program_null",
            "null_all_program_coordinates",
        ),
        (
            "shared_bias_cyclic_core_saturation_v1",
            "shared_three_program_program_id_cyclic",
            "cyclic_program_id_only_roles_core_and_depth_retained",
        ),
    ):
        config = json.loads((ROOT / f"configs/multireaction/{name}.json").read_text())
        arms = _study_arms(config, PROGRAMS)
        assert list(arms) == [expected]
        arm = arms[expected]
        assert arm["conditioning"] == conditioning
        assert arm["source_marginal_mode"] == "program_role_full_support"
        assert arm["model_overrides"]["program_routed_output_heads"] is True
        assert arm["candidate_source"] is False
        if "program_id_mapping" in arm:
            mapping = arm["program_id_mapping"]
            assert set(mapping) == set(mapping.values()) == set(PROGRAMS)
            assert all(key != value for key, value in mapping.items())
