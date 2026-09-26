"""Errors raised at experiment-system boundaries."""

from __future__ import annotations


class ExperimentError(RuntimeError):
    """Base class for an experiment that cannot be planned, run, or verified."""


class SpecError(ExperimentError, ValueError):
    """An experiment specification is malformed or internally inconsistent."""


class RegistryError(ExperimentError, LookupError):
    """A stage implementation is missing or registered more than once."""


class StageError(ExperimentError):
    """A stage violated its declared inputs, outputs, or execution contract."""


class RunExistsError(ExperimentError):
    """A completed run exists and the caller did not request verified resume."""


class VerificationError(ExperimentError):
    """A completed run no longer matches its recorded hashes or schemas."""


class BackendError(ExperimentError):
    """An execution backend could not prepare, execute, or collect a stage."""


__all__ = [
    "BackendError",
    "ExperimentError",
    "RegistryError",
    "RunExistsError",
    "SpecError",
    "StageError",
    "VerificationError",
]
