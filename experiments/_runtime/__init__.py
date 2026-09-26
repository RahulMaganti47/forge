"""Typed, content-addressed experiment execution.

Outside `forge` because it is orchestration rather than science: typed contracts, keyed seeds,
atomic stage commits, verified resume, and execution backends. FORGE-specific adapters live with
their experiment applications and are loaded explicitly through ``experiments.catalog``.

"""

# Importing the runtime registers infrastructure-only stages. Scientific stage adapters are loaded
# explicitly by the experiment catalog; JSON specifications never import arbitrary Python modules.
import experiments._runtime.builtin_stages as _builtin_stages  # noqa: F401
from experiments._runtime.backends import ExecutionBackend, LocalBackend, ModalRuntimeBackend
from experiments._runtime.doctor import diagnose_experiment
from experiments._runtime.errors import (
    BackendError,
    ExperimentError,
    RegistryError,
    RunExistsError,
    SpecError,
    StageError,
    VerificationError,
)
from experiments._runtime.registry import StageRegistry, registry, stage
from experiments._runtime.runner import (
    ExperimentRunner,
    ExperimentRunResult,
    RunPlan,
    verify_run_directory,
)
from experiments._runtime.seed import SeedPlan
from experiments._runtime.spec import (
    DeterminismSpec,
    ExperimentSpec,
    OutputSpec,
    ResourceSpec,
    StageSpec,
)
from experiments._runtime.stage import ProducedArtifact, RunContext, StageResult

__all__ = [
    "BackendError",
    "DeterminismSpec",
    "ExecutionBackend",
    "ExperimentError",
    "ExperimentRunResult",
    "ExperimentRunner",
    "ExperimentSpec",
    "LocalBackend",
    "ModalRuntimeBackend",
    "OutputSpec",
    "ProducedArtifact",
    "RegistryError",
    "ResourceSpec",
    "RunContext",
    "RunExistsError",
    "RunPlan",
    "SeedPlan",
    "SpecError",
    "StageError",
    "StageRegistry",
    "StageResult",
    "StageSpec",
    "VerificationError",
    "diagnose_experiment",
    "registry",
    "stage",
    "verify_run_directory",
]
