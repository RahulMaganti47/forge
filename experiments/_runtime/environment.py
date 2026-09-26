"""Captured runtime identity used in run fingerprints and receipts."""

from __future__ import annotations

import importlib.metadata
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.core.hashing import sha256_file, sha256_json

_PACKAGES = (
    "forge",
    "numpy",
    "pandas",
    "rdkit",
    "scikit-learn",
    "torch",
    "torch-geometric",
    "xgboost",
)


def _versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _accelerators() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"torch_available": False}
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    record: dict[str, Any] = {
        "torch_available": True,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    if cuda_available:
        record["cuda_runtime"] = torch.version.cuda
        record["cuda_device_count"] = int(torch.cuda.device_count())
        record["cuda_devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        record["cudnn_version"] = torch.backends.cudnn.version()
    return record


@dataclass(frozen=True)
class EnvironmentRecord:
    """The software and hardware facts that can alter a numeric run."""

    python: str
    implementation: str
    executable: str
    platform: str
    machine: str
    processor: str
    packages: dict[str, str]
    accelerators: dict[str, Any]
    lock_path: str
    lock_sha256: str

    @classmethod
    def capture(cls, repo: Path) -> EnvironmentRecord:
        lock = repo / "uv.lock"
        if not lock.is_file():
            lock = repo / "pyproject.toml"
        if not lock.is_file():
            raise FileNotFoundError("neither uv.lock nor pyproject.toml exists at repository root")
        return cls(
            python=platform.python_version(),
            implementation=platform.python_implementation(),
            executable=sys.executable,
            platform=platform.platform(),
            machine=platform.machine(),
            processor=platform.processor(),
            packages=_versions(),
            accelerators=_accelerators(),
            lock_path=str(lock.relative_to(repo)),
            lock_sha256=str(sha256_file(lock)),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "accelerators": self.accelerators,
            "executable": self.executable,
            "implementation": self.implementation,
            "lock": {"path": self.lock_path, "sha256": self.lock_sha256},
            "machine": self.machine,
            "packages": self.packages,
            "platform": self.platform,
            "processor": self.processor,
            "python": self.python,
        }

    def fingerprint(self) -> str:
        return str(sha256_json(self.to_mapping()))


__all__ = ["EnvironmentRecord"]
