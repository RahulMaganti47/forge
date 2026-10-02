"""Fail-closed HeLa potency value for a bounded Ugi diagnostic pilot.

This module does not authorize execution.  It authenticates the selected
M0-07 HeLa ensemble, regenerates the two prespecified conditional-calibration
scales, enforces the version-3 four-view applicability policy, and maps only
the above-median portion of an eligible lower-confidence-bound score to
``[0, 1]`` through a calibration-only empirical CDF.  Every other terminal
receives an exactly neutral potential.

The generator and oracle have different qualified software environments.  A
persistent, hash-pinned worker process therefore owns oracle inference; the
current generator runtime owns applicability and SMC orchestration.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

import numpy as np

from examples.diagnostics.support.adapters.terminal_support import (
    native_completion_record_from_locked_terminal,
)
from forge.core.hashing import sha256_file
from forge.core.hashing import sha256_json as _sha256_payload
from forge.core.io import stable_json as _stable_json
from forge.potency.applicability import ugi_distributional_applicability as v1
from forge.potency.applicability import ugi_distributional_applicability_v2 as v2
from forge.potency.applicability.ugi_interpolative_conformal import (
    _SCHEME_ROLES,
    _calibration_bins,
    _selected_calibration_ensembles,
)
from forge.potency.oracle.oracle_classical import conformal_radius
from forge.synthesis.matched import LockedMatchedTerminal

CONFIG_SCHEMA_VERSION = "phase1_ugi_hela_potency_diagnostic_authorization_config.v1"
POLICY_ID = "phase1_ugi_hela_potency_diagnostic_lcb90_ecdf.v1"
WORKER_REQUEST_SCHEMA_VERSION = "forge.ugi_hela_oracle_worker_request.v1"
WORKER_RESPONSE_SCHEMA_VERSION = "forge.ugi_hela_oracle_worker_response.v1"
WORKER_SCHEMA_VERSION = "forge.ugi_hela_oracle_worker.v1"
EXPECTED_SELECTED_CANDIDATE = (
    "supervised_graph::ugi_component_role_aware_dmpnn::neural_3seed_ensemble"
)
EXPECTED_INPUTS = {
    "applicability_v3_result",
    "conformal_v1_result",
    "curated_agile",
    "oracle_campaign_selection",
    "oracle_graph_matrix_result",
    "oracle_production_checkpoint",
    "oracle_production_config",
    "oracle_production_result",
    "oracle_split_assignments",
    "oracle_worker",
}
ROLE_MAP = {
    "amine": "amine_head",
    "aldehyde": "oxoester_aldehyde_body_tail",
    "isocyanide": "isocyanide_tail",
}


class UgiHeLaPotencyDiagnosticError(RuntimeError):
    """Raised when the bounded potency diagnostic cannot be reproduced exactly."""


def _load(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise UgiHeLaPotencyDiagnosticError(f"invalid {label}: {path}") from error
    if not isinstance(value, dict):
        raise UgiHeLaPotencyDiagnosticError(f"{label} must contain a JSON object")
    return value


def _pin(repo: Path, record: Any, *, label: str) -> Path:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise UgiHeLaPotencyDiagnosticError(f"{label} pin is malformed")
    path = (repo / str(record["path"])).resolve()
    try:
        path.relative_to(repo)
    except ValueError as error:
        raise UgiHeLaPotencyDiagnosticError(f"{label} path escapes repository") from error
    if not path.is_file() or path.is_symlink() or sha256_file(path) != record["sha256"]:
        raise UgiHeLaPotencyDiagnosticError(f"{label} hash changed")
    return path


def _finite(value: Any, *, label: str) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError) as error:
        raise UgiHeLaPotencyDiagnosticError(f"{label} is not numeric") from error
    if not math.isfinite(output):
        raise UgiHeLaPotencyDiagnosticError(f"{label} must be finite")
    return output


@dataclass(frozen=True)
class CalibrationScale:
    """One prespecified exact-new-component calibration scale."""

    pattern_id: str
    unseen_roles: tuple[str, ...]
    scheme: str
    eligible_folds: tuple[int, ...]
    fold_q90: tuple[tuple[int, float], ...]
    max_q90: float
    sorted_lcb90: tuple[float, ...]
    values_sha256: str

    def score(self, ensemble_mean: float) -> tuple[float, float, float]:
        """Return LCB90, right-continuous empirical CDF and positive-only utility."""

        lcb90 = _finite(ensemble_mean, label="ensemble mean") - self.max_q90
        cdf = bisect_right(self.sorted_lcb90, lcb90) / len(self.sorted_lcb90)
        potential = max(0.0, 2.0 * cdf - 1.0)
        if not 0.0 <= potential <= 1.0:
            raise UgiHeLaPotencyDiagnosticError("empirical-CDF potential escaped [0, 1]")
        return lcb90, cdf, potential


@dataclass(frozen=True)
class HeLaPotencyEvaluation:
    """One terminal-level potency assessment or exact neutral abstention."""

    terminal_id: str
    terminal_sha256: str
    action: str
    reason: str
    potential: float
    exact_forward_verified: bool
    exact_measured_combination: bool | None
    unseen_roles: tuple[str, ...]
    pattern_id: str | None
    distribution_bins: tuple[tuple[str, str], ...]
    ensemble_mean: float | None
    ensemble_standard_deviation: float | None
    conformal_q90: float | None
    lcb90: float | None
    calibration_ecdf: float | None
    worker_receipt_sha256: str | None
    policy_id: str = POLICY_ID
    synthesis_calls: int = 0
    proposal_calls: int = 0

    def __post_init__(self) -> None:
        if self.action not in {"potency_guidance", "abstain"}:
            raise UgiHeLaPotencyDiagnosticError("unsupported potency action")
        if not 0.0 <= self.potential <= 1.0 or not math.isfinite(self.potential):
            raise UgiHeLaPotencyDiagnosticError("potency potential must be finite in [0, 1]")
        if self.action == "abstain" and self.potential != 0.0:
            raise UgiHeLaPotencyDiagnosticError("every abstention must be exactly neutral")
        if self.action == "potency_guidance" and any(
            value is None
            for value in (
                self.pattern_id,
                self.ensemble_mean,
                self.ensemble_standard_deviation,
                self.conformal_q90,
                self.lcb90,
                self.calibration_ecdf,
                self.worker_receipt_sha256,
            )
        ):
            raise UgiHeLaPotencyDiagnosticError("active potency guidance lacks evidence")
        if self.synthesis_calls != 0 or self.proposal_calls != 0:
            raise UgiHeLaPotencyDiagnosticError("potency-only diagnostic invoked another guide")

    @property
    def receipt_sha256(self) -> str:
        return _sha256_payload(asdict(self))


class HeLaBatchPredictor(Protocol):
    """Narrow dependency boundary for the frozen oracle process."""

    def predict(self, candidates: Sequence[Mapping[str, str]]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class _CandidateContext:
    terminal: LockedMatchedTerminal
    candidate: dict[str, str]
    canonical: dict[str, str]
    unseen_roles: tuple[str, ...]
    exact_measured_combination: bool
    view_bins: tuple[tuple[str, str], ...]
    overall_bin: str
    pattern_id: str | None


class HeLaPotencyDiagnosticPolicy:
    """Authenticated applicability, calibration and score policy."""

    def __init__(self, repo: Path, config_path: Path) -> None:
        self.repo = repo.resolve()
        self.config_path = config_path.resolve()
        config = _load(self.config_path, label="HeLa diagnostic authorization")
        if (
            config.get("schema_version") != CONFIG_SCHEMA_VERSION
            or config.get("status") != "frozen_nonexecuting_diagnostic_policy"
        ):
            raise UgiHeLaPotencyDiagnosticError("unsupported HeLa diagnostic policy")
        scope = config.get("scope")
        required_scope = {
            "diagnostic_only": True,
            "endpoint": "expt_Hela",
            "oracle_candidate_id": EXPECTED_SELECTED_CANDIDATE,
            "synthesis_guidance": False,
            "proposal_guidance": False,
            "candidate_selection": False,
            "prospective_candidate_lock": False,
            "outer_test_used_for_transform": False,
            "generated_data_used_for_transform": False,
            "nonzero_execution_authorized": False,
        }
        if not isinstance(scope, Mapping) or any(
            scope.get(key) != value for key, value in required_scope.items()
        ):
            raise UgiHeLaPotencyDiagnosticError("HeLa diagnostic scope changed")
        inputs = config.get("inputs")
        if not isinstance(inputs, Mapping) or set(inputs) != EXPECTED_INPUTS:
            raise UgiHeLaPotencyDiagnosticError("HeLa diagnostic input set changed")
        self.paths = {name: _pin(self.repo, record, label=name) for name, record in inputs.items()}
        self._validate_upstream_statuses()
        self.thresholds = self._load_thresholds()
        self.curated_rows = self._load_curated_rows()
        self.references = v2._references(self.curated_rows)
        self.component_sets = {
            role: frozenset(v1._canonical(row[field]) for row in self.curated_rows)
            for role, field in v1.ROLE_FIELDS.items()
        }
        self.measured_triples = frozenset(
            "\x1f".join(v1._canonical(row[v1.ROLE_FIELDS[role]]) for role in ROLE_MAP)
            for row in self.curated_rows
        )
        self.scales = self._regenerate_scales(config)
        self.worker = self._worker_contract(config)
        self.policy_sha256 = sha256_file(self.config_path)

    def _validate_upstream_statuses(self) -> None:
        production = _load(self.paths["oracle_production_result"], label="production oracle")
        selected = production.get("selected_model")
        candidate_id = (
            "::".join(str(selected.get(key)) for key in ("lane", "representation", "model"))
            if isinstance(selected, Mapping)
            else ""
        )
        checkpoint = production.get("checkpoint")
        configuration = production.get("configuration")
        if (
            production.get("status") != "production_oracle_checkpoint_frozen"
            or candidate_id != EXPECTED_SELECTED_CANDIDATE
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("sha256") != sha256_file(self.paths["oracle_production_checkpoint"])
            or not isinstance(configuration, Mapping)
            or configuration.get("sha256") != sha256_file(self.paths["oracle_production_config"])
        ):
            raise UgiHeLaPotencyDiagnosticError("production oracle identity changed")
        campaign = _load(self.paths["oracle_campaign_selection"], label="campaign selection")
        if campaign.get("status") not in {
            "oracle_campaign_model_selected_guidance_abstained",
            "oracle_campaign_selection_complete_guidance_abstained",
            "hela_campaign_oracle_selected_but_guidance_still_abstained",
        }:
            raise UgiHeLaPotencyDiagnosticError("historical campaign abstention changed")
        applicability = _load(
            self.paths["applicability_v3_result"], label="applicability v3 result"
        )
        if applicability.get("status") != (
            "complete_component_shift_calibrated_audit_guidance_still_abstained"
        ):
            raise UgiHeLaPotencyDiagnosticError("applicability v3 status changed")
        conformal = _load(self.paths["conformal_v1_result"], label="conformal v1 result")
        if (
            conformal.get("status") != "complete_interpolative_conditional_calibration_diagnostic"
            or conformal.get("adjudication", {}).get("biological_guidance_authorized") is not False
        ):
            raise UgiHeLaPotencyDiagnosticError("historical conformal adjudication changed")

    def _load_thresholds(self) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
        value = _load(self.paths["applicability_v3_result"], label="applicability v3 result")
        thresholds = value.get("thresholds")
        if not isinstance(thresholds, Mapping) or set(thresholds) != set(v1.VIEWS):
            raise UgiHeLaPotencyDiagnosticError("applicability thresholds are incomplete")
        return thresholds  # type: ignore[return-value]

    def _load_curated_rows(self) -> list[dict[str, str]]:
        rows = v1._read_csv(self.paths["curated_agile"])
        output = [{**row, "product_smiles": row["model_smiles"]} for row in rows]
        if len(output) != 1100 or len({row["label"] for row in output}) != len(output):
            raise UgiHeLaPotencyDiagnosticError("curated AGILE population changed")
        return output

    def _regenerate_scales(self, config: Mapping[str, Any]) -> dict[str, CalibrationScale]:
        specifications = config.get("eligible_patterns")
        if not isinstance(specifications, Mapping) or set(specifications) != {
            "amine_only",
            "aldehyde_isocyanide_pair",
        }:
            raise UgiHeLaPotencyDiagnosticError("eligible potency patterns changed")
        graph_result = _load(self.paths["oracle_graph_matrix_result"], label="graph matrix")
        ensembles = _selected_calibration_ensembles(
            self.repo,
            graph_result,
            endpoint="expt_Hela",
            representation=v1.SELECTED_REPRESENTATION,
        )
        assignments = v1._split_index(v1._read_csv(self.paths["oracle_split_assignments"]))
        curated = {row["label"]: row for row in self.curated_rows}
        output: dict[str, CalibrationScale] = {}
        for pattern_id, specification in specifications.items():
            if not isinstance(specification, Mapping):
                raise UgiHeLaPotencyDiagnosticError("calibration specification is malformed")
            scheme = str(specification.get("scheme"))
            unseen_roles = tuple(str(value) for value in specification.get("unseen_roles", []))
            eligible_folds = tuple(int(value) for value in specification.get("eligible_folds", []))
            if (
                scheme not in _SCHEME_ROLES
                or unseen_roles != tuple(_SCHEME_ROLES[scheme])
                or not eligible_folds
            ):
                raise UgiHeLaPotencyDiagnosticError("pattern and held-role scheme disagree")
            fold_q90: list[tuple[int, float]] = []
            calibration_predictions: list[float] = []
            fold_counts: dict[int, int] = {}
            for fold in eligible_folds:
                ensemble = ensembles.get((scheme, fold))
                if ensemble is None:
                    raise UgiHeLaPotencyDiagnosticError("eligible calibration fold is missing")
                training = v1._training_rows(curated, assignments, scheme, fold, "train")
                calibration_rows = [curated[str(label)] for label in ensemble["labels"]]
                bins = _calibration_bins(
                    calibration_rows,
                    training,
                    unseen_roles,
                    self.thresholds,
                )
                residuals = []
                for bin_name, truth, prediction in zip(
                    bins,
                    ensemble["truth"],
                    ensemble["prediction"],
                    strict=True,
                ):
                    if bin_name != "interpolative":
                        continue
                    residuals.append(abs(float(truth) - float(prediction)))
                    calibration_predictions.append(float(prediction))
                if len(residuals) < 20:
                    raise UgiHeLaPotencyDiagnosticError("eligible fold has fewer than 20 rows")
                fold_counts[fold] = len(residuals)
                fold_q90.append((fold, conformal_radius(np.asarray(residuals), 0.9)))
            max_q90 = max(value for _, value in fold_q90)
            sorted_lcb90 = tuple(sorted(value - max_q90 for value in calibration_predictions))
            values_sha256 = _sha256_payload(list(sorted_lcb90))
            expected = specification.get("expected")
            observed = {
                "fold_interpolative_rows": {str(key): value for key, value in fold_counts.items()},
                "fold_q90": {str(key): value for key, value in fold_q90},
                "max_q90": max_q90,
                "calibration_records": len(sorted_lcb90),
                "minimum_lcb90": min(sorted_lcb90),
                "maximum_lcb90": max(sorted_lcb90),
                "sorted_lcb90_sha256": values_sha256,
            }
            if expected != observed:
                raise UgiHeLaPotencyDiagnosticError(
                    f"regenerated calibration differs for {pattern_id}"
                )
            output[str(pattern_id)] = CalibrationScale(
                pattern_id=str(pattern_id),
                unseen_roles=unseen_roles,
                scheme=scheme,
                eligible_folds=eligible_folds,
                fold_q90=tuple(fold_q90),
                max_q90=max_q90,
                sorted_lcb90=sorted_lcb90,
                values_sha256=values_sha256,
            )
        return output

    def _worker_contract(self, config: Mapping[str, Any]) -> dict[str, Any]:
        worker = config.get("oracle_worker")
        expected = {
            "python": ".venv-oracle-2025/bin/python",
            "endpoint": "expt_Hela",
            "persistent_json_lines": True,
            "expected_software": {
                "rdkit": "2025.09.6",
                "scikit_learn": "1.8.0",
                "torch": "2.11.0",
            },
        }
        if not isinstance(worker, Mapping) or any(worker.get(k) != v for k, v in expected.items()):
            raise UgiHeLaPotencyDiagnosticError("oracle worker contract changed")
        executable = (self.repo / expected["python"]).resolve()
        if not executable.is_file():
            raise UgiHeLaPotencyDiagnosticError("frozen oracle Python executable is unavailable")
        return {**expected, "executable": str(executable)}

    def _abstain(
        self,
        terminal: LockedMatchedTerminal,
        *,
        reason: str,
        exact_forward_verified: bool,
        exact_measured_combination: bool | None = None,
        unseen_roles: tuple[str, ...] = (),
        pattern_id: str | None = None,
        view_bins: tuple[tuple[str, str], ...] = (),
    ) -> HeLaPotencyEvaluation:
        return HeLaPotencyEvaluation(
            terminal_id=terminal.terminal_id,
            terminal_sha256=terminal.terminal_sha256,
            action="abstain",
            reason=reason,
            potential=0.0,
            exact_forward_verified=exact_forward_verified,
            exact_measured_combination=exact_measured_combination,
            unseen_roles=unseen_roles,
            pattern_id=pattern_id,
            distribution_bins=view_bins,
            ensemble_mean=None,
            ensemble_standard_deviation=None,
            conformal_q90=None,
            lcb90=None,
            calibration_ecdf=None,
            worker_receipt_sha256=None,
        )

    def _candidate_context(
        self, terminal: LockedMatchedTerminal
    ) -> _CandidateContext | HeLaPotencyEvaluation:
        if not isinstance(terminal, LockedMatchedTerminal) or not terminal.terminal_locked:
            raise UgiHeLaPotencyDiagnosticError("potency evaluation requires a locked terminal")
        if not terminal.terminal_valid:
            return self._abstain(
                terminal,
                reason="invalid_terminal",
                exact_forward_verified=False,
            )
        if not terminal.exact_l1:
            return self._abstain(
                terminal,
                reason="nonexact_l1",
                exact_forward_verified=False,
            )
        native = native_completion_record_from_locked_terminal(terminal)
        components = native.get("component_smiles_by_role")
        if (
            native.get("valid") is not True
            or native.get("component_reconstruction_valid") is not True
            or not isinstance(components, Mapping)
        ):
            raise UgiHeLaPotencyDiagnosticError("exact terminal trace lacks exact components")
        candidate = {
            "label": terminal.terminal_sha256,
            "product_smiles": str(native["smiles"]),
            **{f"{role}_smiles": str(components[ROLE_MAP[role]]) for role in ROLE_MAP},
        }
        classification = self.classify_candidate_mapping(candidate)
        return _CandidateContext(
            terminal=terminal,
            candidate=candidate,
            canonical=classification["canonical"],
            unseen_roles=classification["unseen_roles"],
            exact_measured_combination=classification["exact_measured_combination"],
            view_bins=classification["view_bins"],
            overall_bin=classification["overall_bin"],
            pattern_id=classification["pattern_id"],
        )

    def classify_candidate_mapping(self, candidate: Mapping[str, str]) -> dict[str, Any]:
        """Classify an already exact-L1 candidate under the frozen structural policy."""

        required = {
            "label",
            "product_smiles",
            "amine_smiles",
            "aldehyde_smiles",
            "isocyanide_smiles",
        }
        if set(candidate) != required:
            raise UgiHeLaPotencyDiagnosticError("candidate applicability fields changed")
        canonical = {
            "product": v1._canonical(candidate["product_smiles"]),
            **{role: v1._canonical(candidate[f"{role}_smiles"]) for role in ROLE_MAP},
        }
        unseen_roles = tuple(
            role for role in ROLE_MAP if canonical[role] not in self.component_sets[role]
        )
        triple = "\x1f".join(canonical[role] for role in ROLE_MAP)
        exact_measured = triple in self.measured_triples
        distances = v1._distance_fields(self.references, candidate, generated=True)
        bins, overall = v1._bins(distances, self.thresholds)
        view_bins = tuple((view, bins[view]) for view in v1.VIEWS)
        pattern_by_roles = {scale.unseen_roles: name for name, scale in self.scales.items()}
        pattern_id = pattern_by_roles.get(unseen_roles)
        return {
            "canonical": canonical,
            "unseen_roles": unseen_roles,
            "exact_measured_combination": exact_measured,
            "view_bins": view_bins,
            "overall_bin": overall,
            "pattern_id": pattern_id,
        }

    def evaluate_batch(
        self,
        terminals: Sequence[LockedMatchedTerminal],
        predictor: HeLaBatchPredictor,
    ) -> tuple[HeLaPotencyEvaluation, ...]:
        """Evaluate a scheduled terminal batch without silently dropping attempts."""

        contexts: list[_CandidateContext | None] = []
        output: list[HeLaPotencyEvaluation | None] = []
        eligible: list[_CandidateContext] = []
        for terminal in terminals:
            context = self._candidate_context(terminal)
            if isinstance(context, HeLaPotencyEvaluation):
                contexts.append(None)
                output.append(context)
                continue
            contexts.append(context)
            if context.exact_measured_combination:
                output.append(
                    self._abstain(
                        terminal,
                        reason="exact_measured_combination_neutral",
                        exact_forward_verified=True,
                        exact_measured_combination=True,
                        unseen_roles=context.unseen_roles,
                        view_bins=context.view_bins,
                    )
                )
            elif context.overall_bin != "interpolative" or any(
                value != "interpolative" for _, value in context.view_bins
            ):
                output.append(
                    self._abstain(
                        terminal,
                        reason=f"distribution_{context.overall_bin}",
                        exact_forward_verified=True,
                        exact_measured_combination=False,
                        unseen_roles=context.unseen_roles,
                        pattern_id=context.pattern_id,
                        view_bins=context.view_bins,
                    )
                )
            elif context.pattern_id is None:
                output.append(
                    self._abstain(
                        terminal,
                        reason="ineligible_exact_novelty_pattern",
                        exact_forward_verified=True,
                        exact_measured_combination=False,
                        unseen_roles=context.unseen_roles,
                        view_bins=context.view_bins,
                    )
                )
            else:
                output.append(None)
                eligible.append(context)
        if not eligible:
            return tuple(value for value in output if value is not None)

        response = predictor.predict([context.candidate for context in eligible])
        if (
            response.get("schema_version") != WORKER_RESPONSE_SCHEMA_VERSION
            or response.get("status") != "complete"
        ):
            raise UgiHeLaPotencyDiagnosticError("oracle worker prediction failed")
        classifications = response.get("classifications")
        prediction = response.get("prediction")
        worker_receipt = response.get("receipt_sha256")
        if (
            not isinstance(classifications, list)
            or len(classifications) != len(eligible)
            or not isinstance(prediction, Mapping)
            or prediction.get("endpoint") != "expt_Hela"
            or prediction.get("records") != len(eligible)
            or not isinstance(worker_receipt, str)
            or len(worker_receipt) != 64
        ):
            raise UgiHeLaPotencyDiagnosticError("oracle worker response is malformed")
        means = prediction.get("ensemble_mean")
        deviations = prediction.get("ensemble_standard_deviation")
        if (
            not isinstance(means, list)
            or not isinstance(deviations, list)
            or len(means) != len(eligible)
            or len(deviations) != len(eligible)
        ):
            raise UgiHeLaPotencyDiagnosticError("oracle worker predictions are misaligned")

        scored: dict[str, HeLaPotencyEvaluation] = {}
        for context, classification, mean_value, deviation_value in zip(
            eligible,
            classifications,
            means,
            deviations,
            strict=True,
        ):
            if not isinstance(classification, Mapping):
                raise UgiHeLaPotencyDiagnosticError("oracle classification is malformed")
            worker_canonical = classification.get("canonical")
            if (
                classification.get("exact_forward_verified") is not True
                or tuple(classification.get("unseen_component_roles", [])) != context.unseen_roles
                or classification.get("combination_seen_in_measured_training")
                is not context.exact_measured_combination
                or worker_canonical != context.canonical
            ):
                raise UgiHeLaPotencyDiagnosticError(
                    "oracle-runtime identity disagrees with the generator runtime"
                )
            mean = _finite(mean_value, label="ensemble mean")
            deviation = _finite(deviation_value, label="ensemble standard deviation")
            if deviation < 0.0:
                raise UgiHeLaPotencyDiagnosticError("ensemble deviation is negative")
            scale = self.scales[str(context.pattern_id)]
            lcb90, cdf, potential = scale.score(mean)
            evaluation = HeLaPotencyEvaluation(
                terminal_id=context.terminal.terminal_id,
                terminal_sha256=context.terminal.terminal_sha256,
                action="potency_guidance",
                reason="eligible_interpolative_exact_new_pattern",
                potential=potential,
                exact_forward_verified=True,
                exact_measured_combination=False,
                unseen_roles=context.unseen_roles,
                pattern_id=context.pattern_id,
                distribution_bins=context.view_bins,
                ensemble_mean=mean,
                ensemble_standard_deviation=deviation,
                conformal_q90=scale.max_q90,
                lcb90=lcb90,
                calibration_ecdf=cdf,
                worker_receipt_sha256=worker_receipt,
            )
            scored[context.terminal.terminal_sha256] = evaluation
        final = []
        for context, value in zip(contexts, output, strict=True):
            if value is not None:
                final.append(value)
            elif context is not None:
                final.append(scored[context.terminal.terminal_sha256])
            else:  # pragma: no cover - impossible by construction.
                raise UgiHeLaPotencyDiagnosticError("evaluation alignment failed")
        return tuple(final)


class FrozenHeLaOracleWorker:
    """Persistent client for the separately qualified oracle environment."""

    def __init__(self, policy: HeLaPotencyDiagnosticPolicy) -> None:
        self.policy = policy
        worker_path = policy.paths["oracle_worker"]
        result_path = policy.paths["oracle_production_result"]
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        source = str(policy.repo)
        environment["PYTHONPATH"] = source if not existing else os.pathsep.join((source, existing))
        self.process = subprocess.Popen(  # noqa: S603 - executable and script are hash-pinned.
            [
                str(policy.worker["executable"]),
                str(worker_path),
                "--repo",
                str(policy.repo),
                "--production-result",
                str(result_path),
                "--production-result-sha256",
                sha256_file(result_path),
            ],
            cwd=policy.repo,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.stdin: TextIO = self._require_stream(self.process.stdin, "stdin")
        self.stdout: TextIO = self._require_stream(self.process.stdout, "stdout")
        ready = self._read_response()
        selected = ready.get("selected_model")
        candidate_id = (
            "::".join(str(selected.get(key)) for key in ("lane", "representation", "model"))
            if isinstance(selected, Mapping)
            else ""
        )
        if (
            ready.get("schema_version") != WORKER_SCHEMA_VERSION
            or ready.get("status") != "ready"
            or ready.get("endpoint") != "expt_Hela"
            or candidate_id != EXPECTED_SELECTED_CANDIDATE
            or ready.get("software") != policy.worker["expected_software"]
            or ready.get("production_result_sha256") != sha256_file(result_path)
        ):
            self.close(force=True)
            raise UgiHeLaPotencyDiagnosticError("oracle worker readiness receipt changed")
        self.ready_receipt_sha256 = str(ready["receipt_sha256"])

    @staticmethod
    def _require_stream(value: TextIO | None, label: str) -> TextIO:
        if value is None:
            raise UgiHeLaPotencyDiagnosticError(f"oracle worker {label} is unavailable")
        return value

    def _read_response(self) -> dict[str, Any]:
        line = self.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr is not None else ""
            raise UgiHeLaPotencyDiagnosticError(
                f"oracle worker exited before responding: {stderr[-2000:]}"
            )
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise UgiHeLaPotencyDiagnosticError("oracle worker emitted invalid JSON") from error
        if not isinstance(value, dict) or value.get("status") == "error":
            raise UgiHeLaPotencyDiagnosticError(f"oracle worker failed: {value}")
        return value

    def predict(self, candidates: Sequence[Mapping[str, str]]) -> Mapping[str, Any]:
        request = {
            "schema_version": WORKER_REQUEST_SCHEMA_VERSION,
            "action": "predict",
            "endpoint": "expt_Hela",
            "candidates": [dict(candidate) for candidate in candidates],
        }
        request_sha256 = _sha256_payload(request)
        self.stdin.write(_stable_json(request) + "\n")
        self.stdin.flush()
        response = self._read_response()
        if response.get("request_sha256") != request_sha256 or response.get(
            "receipt_sha256"
        ) != _sha256_payload(
            {key: value for key, value in response.items() if key != "receipt_sha256"}
        ):
            raise UgiHeLaPotencyDiagnosticError("oracle worker response receipt changed")
        return response

    def close(self, *, force: bool = False) -> None:
        if self.process.poll() is not None:
            return
        if not force:
            request = {
                "schema_version": WORKER_REQUEST_SCHEMA_VERSION,
                "action": "shutdown",
            }
            try:
                self.stdin.write(_stable_json(request) + "\n")
                self.stdin.flush()
                self._read_response()
            except (BrokenPipeError, UgiHeLaPotencyDiagnosticError):
                force = True
        if force and self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

    def __enter__(self) -> FrozenHeLaOracleWorker:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


__all__ = [
    "CalibrationScale",
    "FrozenHeLaOracleWorker",
    "HeLaBatchPredictor",
    "HeLaPotencyDiagnosticPolicy",
    "HeLaPotencyEvaluation",
    "POLICY_ID",
    "UgiHeLaPotencyDiagnosticError",
]
