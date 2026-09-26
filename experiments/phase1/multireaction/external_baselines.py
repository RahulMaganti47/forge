"""Common data handoff for pinned third-party baseline ports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from forge.core.hashing import pin_record, resolve_pin
from forge.core.io import read_json_object, write_json

from .external_ugi_contract import export_common_ugi_inputs, load_external_baseline_manifest

CONFIG_SCHEMA = "forge.external_ugi_export_config.v1"


class ExternalBaselineStudyError(ValueError):
    """The external baseline handoff changed or is incomplete."""


def run_external_ugi_export(config_path: Path, repo: Path, output_dir: Path) -> dict[str, Any]:
    config = read_json_object(
        config_path, error=ExternalBaselineStudyError, label="external Ugi export config"
    )
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ExternalBaselineStudyError("unsupported external Ugi export config")
    inputs = config.get("inputs")
    required = {"manifest", "ugi_assignments", "qualified_ugi_reactions"}
    if not isinstance(inputs, dict) or set(inputs) != required:
        raise ExternalBaselineStudyError("external Ugi export inputs changed")
    paths = {key: resolve_pin(value, repo, label=key) for key, value in inputs.items()}
    methods = load_external_baseline_manifest(paths["manifest"])
    result = export_common_ugi_inputs(
        paths["ugi_assignments"],
        paths["qualified_ugi_reactions"],
        output_dir,
        expected_reaction_registry_sha256=str(inputs["qualified_ugi_reactions"]["sha256"]),
    )
    result.update(
        {
            "methods": {
                method_id: {
                    "commit": method["commit"],
                    "integration_status": method["integration_status"],
                    "license_status": method["license_status"],
                }
                for method_id, method in sorted(methods.items())
            },
            "config": pin_record(config_path, repo),
            "manifest": pin_record(paths["manifest"], repo),
            "ready_native_ports": sorted(
                method_id
                for method_id, method in methods.items()
                if method["integration_status"] == "native_port_ready"
            ),
            "candidate_selection": False,
        }
    )
    # A handoff is complete even when native ports remain blocked. It is never mislabeled as a
    # completed external model run.
    write_json(output_dir / "result.json", result)
    return result


__all__ = ["ExternalBaselineStudyError", "run_external_ugi_export"]
