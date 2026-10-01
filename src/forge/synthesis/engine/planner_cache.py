"""Deterministic content-addressed cache for complete route assessments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem, rdBase

from forge.synthesis.engine.planner import (
    PlannerBudgetLedger,
    PlannerBudgetLimits,
    RouteTarget,
    SynthesisAssessment,
)

PLANNER_CACHE_KEY_SCHEMA_VERSION = "forge.planner_cache_key.v1"
PLANNER_CACHE_ENTRY_SCHEMA_VERSION = "forge.planner_cache_entry.v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PlannerCacheError(RuntimeError):
    """Raised when a content-addressed cache record violates its contract."""


def _stable_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _sha256_payload(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode()).hexdigest()


def _require_nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlannerCacheError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise PlannerCacheError(f"{label} must contain 64 lowercase hexadecimal characters")
    return value


def _canonical_constitution(smiles: str, *, label: str) -> str:
    _require_nonempty(smiles, label=label)
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise PlannerCacheError(f"{label} is invalid SMILES")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


@dataclass(frozen=True)
class PlannerCacheContext:
    """Frozen planner, evidence, software and identity state used in a key."""

    planner_id: str
    planner_sha256: str
    search_policy_sha256: str
    value_policy_sha256: str
    l1_reaction_sha256: str
    upstream_reaction_registry_sha256: str
    variant_registry_sha256: str
    verifier_sha256: str
    l3_snapshot_sha256: str
    l3_region: str
    l3_accessed_at_utc: str
    l3_expires_at_utc: str
    software_versions: tuple[tuple[str, str], ...]
    identity_policy: str
    stereochemistry_policy: str
    budget_limits: PlannerBudgetLimits

    def __post_init__(self) -> None:
        _require_nonempty(self.planner_id, label="planner_id")
        for field_name in (
            "planner_sha256",
            "search_policy_sha256",
            "value_policy_sha256",
            "l1_reaction_sha256",
            "upstream_reaction_registry_sha256",
            "variant_registry_sha256",
            "verifier_sha256",
            "l3_snapshot_sha256",
        ):
            _require_sha256(getattr(self, field_name), label=field_name)
        for field_name in (
            "l3_region",
            "l3_accessed_at_utc",
            "l3_expires_at_utc",
            "identity_policy",
            "stereochemistry_policy",
        ):
            _require_nonempty(getattr(self, field_name), label=field_name)
        if not self.software_versions:
            raise PlannerCacheError("software_versions must not be empty")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(value, str) and value for value in item)
            for item in self.software_versions
        ):
            raise PlannerCacheError("software_versions must contain nonempty name/version pairs")
        names = [name for name, _ in self.software_versions]
        if names != sorted(names) or len(names) != len(set(names)):
            raise PlannerCacheError("software_versions must be unique and sorted by name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_id": self.planner_id,
            "planner_sha256": self.planner_sha256,
            "search_policy_sha256": self.search_policy_sha256,
            "value_policy_sha256": self.value_policy_sha256,
            "l1_reaction_sha256": self.l1_reaction_sha256,
            "upstream_reaction_registry_sha256": self.upstream_reaction_registry_sha256,
            "variant_registry_sha256": self.variant_registry_sha256,
            "verifier_sha256": self.verifier_sha256,
            "l3_snapshot_sha256": self.l3_snapshot_sha256,
            "l3_region": self.l3_region,
            "l3_accessed_at_utc": self.l3_accessed_at_utc,
            "l3_expires_at_utc": self.l3_expires_at_utc,
            "software_versions": [list(item) for item in self.software_versions],
            "identity_policy": self.identity_policy,
            "stereochemistry_policy": self.stereochemistry_policy,
            "budget_limits": self.budget_limits.to_dict(),
        }


@dataclass(frozen=True)
class PlannerCacheKey:
    """Canonical serialized cache key and its SHA-256 content address."""

    digest: str
    payload_json: str

    def __post_init__(self) -> None:
        _require_sha256(self.digest, label="cache-key digest")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise PlannerCacheError("cache-key payload_json is invalid") from exc
        if _stable_json(payload) != self.payload_json:
            raise PlannerCacheError("cache-key payload_json is not canonical")
        if _sha256_payload(payload) != self.digest:
            raise PlannerCacheError("cache-key digest does not match payload")

    @property
    def payload(self) -> dict[str, Any]:
        value = json.loads(self.payload_json)
        if not isinstance(value, dict):  # pragma: no cover - constructed by build
            raise PlannerCacheError("cache-key payload must be an object")
        return value

    @classmethod
    def build(
        cls,
        target: RouteTarget,
        context: PlannerCacheContext,
        budget: PlannerBudgetLedger,
    ) -> PlannerCacheKey:
        canonical_target = {
            "role": target.role,
            "canonical_smiles": _canonical_constitution(
                target.canonical_smiles,
                label="route target",
            ),
            "product_context_smiles": [
                _canonical_constitution(value, label="product context")
                for value in target.product_context_smiles
            ],
        }
        payload = {
            "schema_version": PLANNER_CACHE_KEY_SCHEMA_VERSION,
            "target": canonical_target,
            "context": context.to_dict(),
            "resource_usage_before_call": budget.cache_key_usage(),
        }
        payload_json = _stable_json(payload)
        return cls(
            digest=hashlib.sha256(payload_json.encode()).hexdigest(),
            payload_json=payload_json,
        )


class FilePlannerCache:
    """Atomic deterministic file cache keyed by the complete planner context."""

    def __init__(self, root: Path):
        self.root = root

    def path_for(self, key: PlannerCacheKey) -> Path:
        return self.root / key.digest[:2] / f"{key.digest}.json"

    def _entry(self, key: PlannerCacheKey, assessment: SynthesisAssessment) -> dict[str, Any]:
        assessment_value = assessment.to_dict()
        return {
            "schema_version": PLANNER_CACHE_ENTRY_SCHEMA_VERSION,
            "key": {
                "digest": key.digest,
                "payload": key.payload,
            },
            "assessment_sha256": _sha256_payload(assessment_value),
            "assessment": assessment_value,
        }

    def get(self, key: PlannerCacheKey) -> SynthesisAssessment | None:
        path = self.path_for(key)
        try:
            raw = path.read_text()
        except FileNotFoundError:
            return None
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PlannerCacheError(f"cache entry is invalid JSON: {path}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != (
            PLANNER_CACHE_ENTRY_SCHEMA_VERSION
        ):
            raise PlannerCacheError(f"cache entry has an unsupported schema: {path}")
        expected_key = {"digest": key.digest, "payload": key.payload}
        if value.get("key") != expected_key:
            raise PlannerCacheError(f"cache entry key mismatch: {path}")
        assessment_value = value.get("assessment")
        if value.get("assessment_sha256") != _sha256_payload(assessment_value):
            raise PlannerCacheError(f"cache assessment checksum mismatch: {path}")
        assessment = SynthesisAssessment.from_dict(assessment_value)
        canonical_bytes = (_stable_json(value) + "\n").encode()
        if raw.encode() != canonical_bytes:
            raise PlannerCacheError(f"cache entry is not canonical JSON: {path}")
        return assessment

    def put(self, key: PlannerCacheKey, assessment: SynthesisAssessment) -> None:
        path = self.path_for(key)
        existing = self.get(key)
        if existing is not None:
            if existing != assessment:
                raise PlannerCacheError("cache key already stores a different assessment")
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = self._entry(key, assessment)
        payload = (_stable_json(entry) + "\n").encode()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                existing = self.get(key)
                if existing != assessment:
                    raise PlannerCacheError("concurrent cache write stored different assessment")
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:  # pragma: no cover - defensive cleanup
                pass
