"""Content-addressed, fail-closed execution of experiment DAGs."""

from __future__ import annotations

import datetime as dt
import os
import shutil
import tempfile
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiments._runtime.backends.base import ExecutionBackend
from experiments._runtime.backends.local import LocalBackend
from experiments._runtime.environment import EnvironmentRecord
from experiments._runtime.errors import (
    RunExistsError,
    StageError,
    VerificationError,
)
from experiments._runtime.registry import StageRegistry
from experiments._runtime.registry import registry as global_registry
from experiments._runtime.seed import SeedPlan
from experiments._runtime.source import source_fingerprint
from experiments._runtime.spec import ExperimentSpec, ResourceSpec, StageSpec
from experiments._runtime.stage import (
    DependencyArtifact,
    ProducedArtifact,
    RunContext,
    StageResult,
)
from forge.core.hashing import sha256_file, sha256_json
from forge.core.io import read_json_object, write_json

STAGE_MANIFEST_SCHEMA = "forge.stage_manifest.v1"
RUN_MANIFEST_SCHEMA = "forge.run_manifest.v1"


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _contained(path: Path, parent: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(parent.resolve())
    except ValueError as error:
        raise StageError(f"{label} escapes its declared directory: {resolved}") from error
    return resolved


@dataclass(frozen=True)
class PlannedStage:
    stage_id: str
    implementation: str
    resources: ResourceSpec
    external_inputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    outputs: tuple[str, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "dependencies": list(self.dependencies),
            "external_inputs": list(self.external_inputs),
            "implementation": self.implementation,
            "outputs": list(self.outputs),
            "resources": self.resources.to_mapping(),
            "stage_id": self.stage_id,
        }


@dataclass(frozen=True)
class RunPlan:
    experiment_id: str
    run_id: str
    profile: str
    replicate: int
    master_seed: int
    root_seed: int
    backend: str
    run_dir: Path
    source_sha256: str
    spec_sha256: str
    environment: EnvironmentRecord
    stages: tuple[PlannedStage, ...]

    def to_mapping(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "environment": self.environment.to_mapping(),
            "experiment_id": self.experiment_id,
            "profile": self.profile,
            "replicate": self.replicate,
            "master_seed": self.master_seed,
            "root_seed": self.root_seed,
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "source_sha256": self.source_sha256,
            "spec_sha256": self.spec_sha256,
            "stages": [stage.to_mapping() for stage in self.stages],
        }


@dataclass(frozen=True)
class ExperimentRunResult:
    plan: RunPlan
    stage_manifests: Mapping[str, Mapping[str, Any]]


class ExperimentRunner:
    """Plan, execute, resume, and verify one immutable experiment specification."""

    def __init__(
        self,
        repo: Path,
        *,
        runs_root: Path | None = None,
        registry: StageRegistry | None = None,
        backend: ExecutionBackend | None = None,
        source_paths: tuple[str, ...] | None = None,
    ) -> None:
        self.repo = repo.resolve()
        self.runs_root = (runs_root or (self.repo / "runs")).resolve()
        self.registry = registry or global_registry
        self.backend = backend or LocalBackend()
        self.source_paths = source_paths
        _contained(self.runs_root, self.repo, label="runs root")

    def _resources(self, stage: StageSpec, device: str | None) -> ResourceSpec:
        return stage.resources if device is None else stage.resources.with_device(device)

    def _resolve_external_inputs(
        self, spec: ExperimentSpec
    ) -> tuple[dict[str, Path], dict[str, dict[str, Path]]]:
        configs: dict[str, Path] = {}
        inputs: dict[str, dict[str, Path]] = {}
        for stage in spec.stages:
            try:
                configs[stage.stage_id] = stage.config.resolve(self.repo)
                inputs[stage.stage_id] = {
                    label: pin.resolve(self.repo) for label, pin in stage.inputs.items()
                }
            except (OSError, ValueError) as error:
                raise StageError(
                    f"stage {stage.stage_id!r} external input verification failed: {error}"
                ) from error
        return configs, inputs

    def plan(
        self,
        spec_path: Path,
        *,
        profile: str,
        device: str | None = None,
        replicate: int = 0,
    ) -> RunPlan:
        spec_path = spec_path.resolve()
        _contained(spec_path, self.repo, label="experiment spec")
        spec = ExperimentSpec.load(spec_path)
        if profile not in spec.profiles:
            raise StageError(
                f"profile {profile!r} is not declared by {spec.experiment_id!r}; "
                f"available: {list(spec.profiles)}"
            )
        replicate_count = spec.replicates[profile]
        if isinstance(replicate, bool) or not 0 <= replicate < replicate_count:
            raise StageError(
                f"replicate {replicate!r} is outside [0, {replicate_count}) for profile {profile!r}"
            )
        root_seed = SeedPlan(
            root_seed=spec.root_seed,
            namespace=f"{spec.experiment_id}/{profile}/replicates",
        ).derive("replicate", replicate)
        self._resolve_external_inputs(spec)
        for stage in spec.stages:
            self.registry.resolve(stage.implementation)

        source_sha256 = (
            source_fingerprint(self.repo, self.source_paths)
            if self.source_paths
            else source_fingerprint(self.repo)
        )
        spec_sha256 = str(sha256_file(spec_path))
        environment = EnvironmentRecord.capture(self.repo)
        resources = {
            stage.stage_id: self._resources(stage, device).to_mapping() for stage in spec.stages
        }
        run_id = str(
            sha256_json(
                {
                    "backend": self.backend.name,
                    "environment_sha256": environment.fingerprint(),
                    "profile": profile,
                    "replicate": replicate,
                    "root_seed": root_seed,
                    "resources": resources,
                    "source_sha256": source_sha256,
                    "spec": spec.to_mapping(),
                    "spec_sha256": spec_sha256,
                }
            )
        )
        run_dir = self.runs_root / spec.experiment_id / run_id
        stages = tuple(
            PlannedStage(
                stage_id=stage.stage_id,
                implementation=stage.implementation,
                resources=self._resources(stage, device),
                external_inputs=tuple(sorted(stage.inputs)),
                dependencies=stage.needs,
                outputs=tuple(sorted(stage.outputs)),
            )
            for stage in spec.topological_stages()
        )
        return RunPlan(
            experiment_id=spec.experiment_id,
            run_id=run_id,
            profile=profile,
            replicate=replicate,
            master_seed=spec.root_seed,
            root_seed=root_seed,
            backend=self.backend.name,
            run_dir=run_dir,
            source_sha256=source_sha256,
            spec_sha256=spec_sha256,
            environment=environment,
            stages=stages,
        )

    def _stage_fingerprint(
        self,
        *,
        stage: StageSpec,
        resources: ResourceSpec,
        plan: RunPlan,
        dependencies: Mapping[str, Mapping[str, Any]],
    ) -> str:
        dependency_contract = {
            stage_id: {
                "fingerprint": manifest["fingerprint"],
                "artifacts": {
                    label: {
                        "schema_version": record["schema_version"],
                        "sha256": record["sha256"],
                    }
                    for label, record in sorted(manifest["artifacts"].items())
                },
            }
            for stage_id, manifest in sorted(dependencies.items())
        }
        return str(
            sha256_json(
                {
                    "backend": plan.backend,
                    "dependencies": dependency_contract,
                    "environment_sha256": plan.environment.fingerprint(),
                    "profile": plan.profile,
                    "replicate": plan.replicate,
                    "root_seed": plan.root_seed,
                    "resources": resources.to_mapping(),
                    "source_sha256": plan.source_sha256,
                    "stage": stage.to_mapping(),
                }
            )
        )

    def _dependency_artifacts(
        self,
        manifests: Mapping[str, Mapping[str, Any]],
        stage_dirs: Mapping[str, Path],
    ) -> dict[str, dict[str, DependencyArtifact]]:
        dependencies: dict[str, dict[str, DependencyArtifact]] = {}
        for stage_id, manifest in manifests.items():
            records: dict[str, DependencyArtifact] = {}
            for label, artifact in manifest["artifacts"].items():
                path = _contained(
                    stage_dirs[stage_id] / artifact["path"],
                    stage_dirs[stage_id],
                    label=f"dependency {stage_id}.{label}",
                )
                records[label] = DependencyArtifact(
                    stage_id=stage_id,
                    label=label,
                    path=path,
                    sha256=str(artifact["sha256"]),
                    schema_version=str(artifact["schema_version"]),
                    rows=artifact.get("rows"),
                )
            dependencies[stage_id] = records
        return dependencies

    def _artifact_records(
        self,
        stage: StageSpec,
        result: StageResult,
        output_dir: Path,
    ) -> dict[str, dict[str, Any]]:
        produced = {artifact.label: artifact for artifact in result.artifacts}
        if set(produced) != set(stage.outputs):
            missing = sorted(set(stage.outputs) - set(produced))
            unexpected = sorted(set(produced) - set(stage.outputs))
            raise StageError(
                f"stage {stage.stage_id!r} output labels differ from its spec; "
                f"missing={missing}, unexpected={unexpected}"
            )
        records: dict[str, dict[str, Any]] = {}
        for label, expected in sorted(stage.outputs.items()):
            artifact: ProducedArtifact = produced[label]
            if artifact.relative_path != expected.path:
                raise StageError(
                    f"stage {stage.stage_id!r} artifact {label!r} wrote "
                    f"{artifact.relative_path!r}; expected {expected.path!r}"
                )
            if artifact.schema_version != expected.schema_version:
                raise StageError(
                    f"stage {stage.stage_id!r} artifact {label!r} schema is "
                    f"{artifact.schema_version!r}; expected {expected.schema_version!r}"
                )
            path = _contained(
                output_dir / expected.path,
                output_dir,
                label=f"output {stage.stage_id}.{label}",
            )
            if path.is_symlink() or not path.is_file():
                raise StageError(
                    f"stage {stage.stage_id!r} did not produce a regular file for {label!r}: {path}"
                )
            record: dict[str, Any] = {
                "bytes": path.stat().st_size,
                "path": f"artifacts/{expected.path}",
                "schema_version": expected.schema_version,
                "sha256": str(sha256_file(path)),
            }
            if artifact.rows is not None:
                record["rows"] = artifact.rows
            records[label] = record
        return records

    def _write_failure(
        self,
        plan: RunPlan,
        stage: StageSpec,
        stage_fingerprint: str,
        error: BaseException,
    ) -> None:
        failure_dir = plan.run_dir / "failures" / stage.stage_id
        failure_dir.mkdir(parents=True, exist_ok=True)
        identifier = _utc_now().replace(":", "-")
        write_json(
            failure_dir / f"{identifier}.json",
            {
                "error": {"message": str(error), "type": type(error).__name__},
                "experiment_id": plan.experiment_id,
                "fingerprint": stage_fingerprint,
                "run_id": plan.run_id,
                "schema_version": "forge.stage_failure.v1",
                "stage_id": stage.stage_id,
                "status": "failed",
                "traceback": traceback.format_exc(),
                "utc": _utc_now(),
            },
        )

    def _load_and_verify_stage(
        self,
        stage: StageSpec,
        stage_dir: Path,
        expected_fingerprint: str,
    ) -> dict[str, Any]:
        manifest_path = stage_dir / "manifest.json"
        manifest = read_json_object(
            manifest_path,
            error=VerificationError,
            label=f"stage {stage.stage_id} manifest",
        )
        if manifest.get("schema_version") != STAGE_MANIFEST_SCHEMA:
            raise VerificationError(f"unsupported stage manifest: {manifest_path}")
        if manifest.get("status") != "complete":
            raise VerificationError(f"stage manifest is not complete: {manifest_path}")
        if manifest.get("stage_id") != stage.stage_id:
            raise VerificationError(f"stage id mismatch in {manifest_path}")
        if manifest.get("fingerprint") != expected_fingerprint:
            raise VerificationError(
                f"stage fingerprint mismatch for {stage.stage_id}: "
                f"expected {expected_fingerprint}, found {manifest.get('fingerprint')}"
            )
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(stage.outputs):
            raise VerificationError(f"stage artifact labels changed: {stage.stage_id}")
        for label, expected in stage.outputs.items():
            record = artifacts[label]
            if not isinstance(record, dict):
                raise VerificationError(f"malformed artifact record: {stage.stage_id}.{label}")
            if record.get("schema_version") != expected.schema_version:
                raise VerificationError(f"artifact schema changed: {stage.stage_id}.{label}")
            path = _contained(
                stage_dir / str(record.get("path", "")),
                stage_dir,
                label=f"artifact {stage.stage_id}.{label}",
            )
            if path.is_symlink() or not path.is_file():
                raise VerificationError(f"artifact is missing: {path}")
            observed = str(sha256_file(path))
            if observed != record.get("sha256"):
                raise VerificationError(
                    f"artifact changed: {stage.stage_id}.{label}; "
                    f"expected {record.get('sha256')}, found {observed}"
                )
            if path.stat().st_size != record.get("bytes"):
                raise VerificationError(f"artifact byte count changed: {stage.stage_id}.{label}")
        return manifest

    def run(
        self,
        spec_path: Path,
        *,
        profile: str,
        device: str | None = None,
        resume: bool = False,
        replicate: int = 0,
    ) -> ExperimentRunResult:
        spec = ExperimentSpec.load(spec_path.resolve())
        plan = self.plan(spec_path, profile=profile, device=device, replicate=replicate)
        config_paths, input_paths = self._resolve_external_inputs(spec)
        plan.run_dir.mkdir(parents=True, exist_ok=True)

        manifests: dict[str, dict[str, Any]] = {}
        stage_dirs: dict[str, Path] = {}
        for stage in spec.topological_stages():
            resources = self._resources(stage, device)
            dependency_manifests = {stage_id: manifests[stage_id] for stage_id in stage.needs}
            dependency_dirs = {stage_id: stage_dirs[stage_id] for stage_id in stage.needs}
            fingerprint = self._stage_fingerprint(
                stage=stage,
                resources=resources,
                plan=plan,
                dependencies=dependency_manifests,
            )
            final_dir = plan.run_dir / "stages" / stage.stage_id
            partial_dir = final_dir.parent / f".{stage.stage_id}.partial"
            if final_dir.exists():
                if not resume:
                    raise RunExistsError(
                        f"completed stage exists at {final_dir}; pass --resume to verify and reuse it"
                    )
                manifest = self._load_and_verify_stage(stage, final_dir, fingerprint)
                manifests[stage.stage_id] = manifest
                stage_dirs[stage.stage_id] = final_dir
                continue

            final_dir.parent.mkdir(parents=True, exist_ok=True)
            resuming_partial = partial_dir.exists()
            if resuming_partial:
                if not resume:
                    raise RunExistsError(
                        f"incomplete stage exists at {partial_dir}; pass --resume to continue it"
                    )
                partial = read_json_object(
                    partial_dir / "partial.json",
                    error=VerificationError,
                    label=f"stage {stage.stage_id} partial receipt",
                )
                if (
                    partial.get("schema_version") != "forge.partial_stage.v1"
                    or partial.get("stage_id") != stage.stage_id
                    or partial.get("run_id") != plan.run_id
                    or partial.get("fingerprint") != fingerprint
                ):
                    raise VerificationError(
                        f"incomplete stage does not match the planned run: {partial_dir}"
                    )
                temporary = partial_dir
            else:
                partial_dir.mkdir()
                temporary = partial_dir
                write_json(
                    temporary / "partial.json",
                    {
                        "experiment_id": spec.experiment_id,
                        "fingerprint": fingerprint,
                        "run_id": plan.run_id,
                        "schema_version": "forge.partial_stage.v1",
                        "stage_id": stage.stage_id,
                        "status": "incomplete",
                    },
                )
            output_dir = temporary / "artifacts"
            output_dir.mkdir(exist_ok=resuming_partial)
            work_dir = temporary / "work"
            work_dir.mkdir(exist_ok=resuming_partial)
            seed_plan = SeedPlan(
                root_seed=plan.root_seed,
                namespace=f"{spec.experiment_id}/{plan.profile}/{stage.determinism.stream}",
            )
            context = RunContext(
                repo=self.repo,
                experiment_id=spec.experiment_id,
                run_id=plan.run_id,
                profile=plan.profile,
                replicate=plan.replicate,
                backend=plan.backend,
                stage=stage,
                resources=resources,
                work_dir=work_dir,
                output_dir=output_dir,
                config_path=config_paths[stage.stage_id],
                inputs=input_paths[stage.stage_id],
                dependencies=self._dependency_artifacts(
                    dependency_manifests,
                    dependency_dirs,
                ),
                seed_plan=seed_plan,
                resume=resuming_partial,
            )
            try:
                function = self.registry.resolve(stage.implementation)
                result = self.backend.execute(function, context)
                artifact_records = self._artifact_records(stage, result, output_dir)
                manifest = {
                    "artifacts": artifact_records,
                    "backend": plan.backend,
                    "completed_utc": _utc_now(),
                    "config": stage.config.to_mapping(),
                    "dependencies": {
                        stage_id: {"fingerprint": dependency_manifests[stage_id]["fingerprint"]}
                        for stage_id in stage.needs
                    },
                    "determinism": stage.determinism.to_mapping(),
                    "environment": plan.environment.to_mapping(),
                    "experiment_id": spec.experiment_id,
                    "external_inputs": {
                        label: pin.to_mapping() for label, pin in sorted(stage.inputs.items())
                    },
                    "fingerprint": fingerprint,
                    "implementation": stage.implementation,
                    "metrics": dict(result.metrics),
                    "profile": plan.profile,
                    "randomness": {
                        "derived_seeds": dict(context.derived_seeds),
                        "master_seed": plan.master_seed,
                        "replicate": plan.replicate,
                        "root_seed": plan.root_seed,
                        "stream": stage.determinism.stream,
                    },
                    "resources": resources.to_mapping(),
                    "run_id": plan.run_id,
                    "schema_version": STAGE_MANIFEST_SCHEMA,
                    "source_sha256": plan.source_sha256,
                    "stage_id": stage.stage_id,
                    "status": "complete",
                    "summary": dict(result.summary),
                }
                write_json(temporary / "manifest.json", manifest)
                (temporary / "partial.json").unlink()
                os.rename(temporary, final_dir)
                # Scratch state is deliberately retained until after the atomic commit so a
                # failure anywhere in output publication remains resumable.  It is not part of
                # the immutable run contract and can be discarded after the stage is visible.
                shutil.rmtree(final_dir / "work", ignore_errors=True)
            except BaseException as error:
                self._write_failure(plan, stage, fingerprint, error)
                raise

            manifests[stage.stage_id] = manifest
            stage_dirs[stage.stage_id] = final_dir

        run_manifest = {
            "backend": plan.backend,
            "completed_utc": _utc_now(),
            "experiment_id": spec.experiment_id,
            "nonclaims": list(spec.nonclaims),
            "profile": plan.profile,
            "replicate": plan.replicate,
            "randomness": {
                "master_seed": plan.master_seed,
                "root_seed": plan.root_seed,
            },
            "run_id": plan.run_id,
            "schema_version": RUN_MANIFEST_SCHEMA,
            "source_sha256": plan.source_sha256,
            "spec_sha256": plan.spec_sha256,
            "stages": {
                stage_id: {"fingerprint": manifest["fingerprint"], "status": "complete"}
                for stage_id, manifest in manifests.items()
            },
            "status": "complete",
        }
        write_json(plan.run_dir / "run.json", run_manifest)
        return ExperimentRunResult(plan=plan, stage_manifests=manifests)

    def verify(
        self,
        spec_path: Path,
        *,
        profile: str,
        device: str | None = None,
        replicate: int = 0,
    ) -> ExperimentRunResult:
        spec = ExperimentSpec.load(spec_path.resolve())
        plan = self.plan(spec_path, profile=profile, device=device, replicate=replicate)
        manifests: dict[str, dict[str, Any]] = {}
        stage_dirs: dict[str, Path] = {}
        for stage in spec.topological_stages():
            dependencies = {stage_id: manifests[stage_id] for stage_id in stage.needs}
            fingerprint = self._stage_fingerprint(
                stage=stage,
                resources=self._resources(stage, device),
                plan=plan,
                dependencies=dependencies,
            )
            stage_dir = plan.run_dir / "stages" / stage.stage_id
            manifests[stage.stage_id] = self._load_and_verify_stage(stage, stage_dir, fingerprint)
            stage_dirs[stage.stage_id] = stage_dir
        run_manifest = read_json_object(
            plan.run_dir / "run.json",
            error=VerificationError,
            label="experiment run manifest",
        )
        if run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
            raise VerificationError("unsupported experiment run manifest")
        if run_manifest.get("run_id") != plan.run_id or run_manifest.get("status") != "complete":
            raise VerificationError("experiment run manifest does not match the planned run")
        return ExperimentRunResult(plan=plan, stage_manifests=manifests)

    def reproduce(
        self,
        spec_path: Path,
        *,
        profile: str,
        device: str | None = None,
        replicate: int = 0,
    ) -> dict[str, Any]:
        """Execute twice from empty roots and require byte identity for strict stages."""

        spec = ExperimentSpec.load(spec_path.resolve())
        scratch_parent = self.runs_root / ".reproduction"
        scratch_parent.mkdir(parents=True, exist_ok=True)
        first_root = Path(tempfile.mkdtemp(prefix="first-", dir=scratch_parent))
        second_root = Path(tempfile.mkdtemp(prefix="second-", dir=scratch_parent))
        try:
            first = ExperimentRunner(
                self.repo,
                runs_root=first_root,
                registry=self.registry,
                backend=self.backend,
                source_paths=self.source_paths,
            ).run(
                spec_path,
                profile=profile,
                device=device,
                replicate=replicate,
            )
            second = ExperimentRunner(
                self.repo,
                runs_root=second_root,
                registry=self.registry,
                backend=self.backend,
                source_paths=self.source_paths,
            ).run(
                spec_path,
                profile=profile,
                device=device,
                replicate=replicate,
            )
            comparisons: dict[str, Any] = {}
            for stage in spec.topological_stages():
                first_artifacts = first.stage_manifests[stage.stage_id]["artifacts"]
                second_artifacts = second.stage_manifests[stage.stage_id]["artifacts"]
                first_hashes = {
                    label: record["sha256"] for label, record in sorted(first_artifacts.items())
                }
                second_hashes = {
                    label: record["sha256"] for label, record in sorted(second_artifacts.items())
                }
                identical = first_hashes == second_hashes
                if stage.determinism.mode == "strict" and not identical:
                    raise VerificationError(
                        f"strict stage is not byte-reproducible: {stage.stage_id}; "
                        f"first={first_hashes}, second={second_hashes}"
                    )
                comparisons[stage.stage_id] = {
                    "artifact_sha256": first_hashes,
                    "byte_identical": identical,
                    "mode": stage.determinism.mode,
                    "requirement": (
                        "byte_identity"
                        if stage.determinism.mode == "strict"
                        else "replicated_statistics"
                    ),
                }
            return {
                "backend": self.backend.name,
                "experiment_id": spec.experiment_id,
                "profile": profile,
                "replicate": replicate,
                "run_id": first.plan.run_id,
                "schema_version": "forge.reproduction_check.v1",
                "stages": comparisons,
                "status": "reproduced",
            }
        finally:
            shutil.rmtree(first_root, ignore_errors=True)
            shutil.rmtree(second_root, ignore_errors=True)


def verify_run_directory(run_dir: Path) -> dict[str, Any]:
    """Verify a downloaded run from its self-contained manifests.

    This check is environment-independent, so a Modal run can be verified on a CPU workstation.
    It proves that every downloaded artifact still matches the digest committed by the remote
    runner.  Re-executing the spec remains the stronger reproducibility test.
    """

    resolved_run = run_dir.resolve()
    run_manifest = read_json_object(
        resolved_run / "run.json",
        error=VerificationError,
        label="experiment run manifest",
    )
    if run_manifest.get("schema_version") != RUN_MANIFEST_SCHEMA:
        raise VerificationError("unsupported experiment run manifest")
    if run_manifest.get("status") != "complete":
        raise VerificationError("experiment run is not complete")
    run_id = run_manifest.get("run_id")
    if not isinstance(run_id, str) or resolved_run.name != run_id:
        raise VerificationError("run directory name and manifest run id differ")
    declared_stages = run_manifest.get("stages")
    if not isinstance(declared_stages, dict) or not declared_stages:
        raise VerificationError("experiment run has no declared stages")

    verified_artifacts = 0
    for stage_id, run_record in sorted(declared_stages.items()):
        if not isinstance(stage_id, str) or not isinstance(run_record, dict):
            raise VerificationError("experiment run contains a malformed stage record")
        manifest_path = resolved_run / "stages" / stage_id / "manifest.json"
        manifest = read_json_object(
            manifest_path,
            error=VerificationError,
            label=f"stage {stage_id} manifest",
        )
        if (
            manifest.get("schema_version") != STAGE_MANIFEST_SCHEMA
            or manifest.get("status") != "complete"
            or manifest.get("stage_id") != stage_id
            or manifest.get("run_id") != run_id
            or manifest.get("fingerprint") != run_record.get("fingerprint")
        ):
            raise VerificationError(f"stage manifest does not match run receipt: {stage_id}")
        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict):
            raise VerificationError(f"stage has no artifact mapping: {stage_id}")
        for label, record in sorted(artifacts.items()):
            if not isinstance(record, dict):
                raise VerificationError(f"malformed artifact record: {stage_id}.{label}")
            path = _contained(
                resolved_run / "stages" / stage_id / str(record.get("path", "")),
                resolved_run / "stages" / stage_id,
                label=f"artifact {stage_id}.{label}",
            )
            if path.is_symlink() or not path.is_file():
                raise VerificationError(f"artifact is missing: {path}")
            observed = str(sha256_file(path))
            if observed != record.get("sha256") or path.stat().st_size != record.get("bytes"):
                raise VerificationError(f"artifact changed: {stage_id}.{label}")
            verified_artifacts += 1
    return {
        "artifacts": verified_artifacts,
        "backend": run_manifest.get("backend"),
        "experiment_id": run_manifest.get("experiment_id"),
        "run_id": run_id,
        "schema_version": "forge.run_verification.v1",
        "stages": len(declared_stages),
        "status": "verified",
    }


__all__ = [
    "ExperimentRunResult",
    "ExperimentRunner",
    "PlannedStage",
    "RUN_MANIFEST_SCHEMA",
    "RunPlan",
    "STAGE_MANIFEST_SCHEMA",
    "verify_run_directory",
]
