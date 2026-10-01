"""Run the leak-aware classical lane of the M0-07 AGILE oracle matrix.

This module evaluates count-based Morgan fingerprints, structure-computable
RDKit descriptors, and their concatenation with fixed classical regressors.
All preprocessing is fitted on training rows only. Calibration rows are used
only after model fitting to construct split-conformal intervals.

The lane is an M0 audit. It does not freeze a biological oracle.
"""

from __future__ import annotations

import math

import numpy as np

CONFIG_SCHEMA_VERSION = "m0_07_oracle_classical_config.v1"
RESULT_SCHEMA_VERSION = "m0_07_oracle_classical.v1"
STAGES = ("train", "calibration", "test")
SERIALIZED_FLOAT_SIGNIFICANT_DIGITS = 12

METRIC_FIELDS = (
    "representation",
    "model",
    "endpoint",
    "scheme",
    "fold",
    "fit_seed",
    "train_rows",
    "calibration_rows",
    "test_rows",
    "calibration_r2",
    "calibration_rmse",
    "calibration_mae",
    "calibration_pearson_r",
    "calibration_spearman_rho",
    "test_r2",
    "test_rmse",
    "test_mae",
    "test_pearson_r",
    "test_spearman_rho",
    "conformal_q80",
    "test_coverage80",
    "test_mean_interval_width80",
    "conformal_q90",
    "test_coverage90",
    "test_mean_interval_width90",
    "conformal_q95",
    "test_coverage95",
    "test_mean_interval_width95",
    "fit_warning_count",
    "fit_warning_categories",
)

PREDICTION_FIELDS = (
    "representation",
    "model",
    "endpoint",
    "scheme",
    "fold",
    "label",
    "y_true",
    "y_pred",
    "absolute_error",
)

APPLICABILITY_FIELDS = (
    "source_row_index",
    "canonical_model_smiles",
    "nearest_curated_label",
    "max_binary_morgan_tanimoto",
)


class OracleClassicalError(ValueError):
    """Raised when the classical oracle lane violates its frozen contract."""


def conformal_radius(residuals: np.ndarray, coverage: float) -> float:
    """Return the finite-sample split-conformal absolute-residual radius."""

    values = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise OracleClassicalError("conformal calibration residuals must be finite and nonempty")
    if not 0.0 < coverage < 1.0:
        raise OracleClassicalError(f"invalid conformal coverage: {coverage}")
    rank = min(values.size, math.ceil((values.size + 1) * coverage))
    return float(np.partition(values, rank - 1)[rank - 1])
