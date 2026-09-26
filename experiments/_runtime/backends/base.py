"""Execution backend protocol."""

from __future__ import annotations

from typing import Protocol

from experiments._runtime.stage import RunContext, StageCallable, StageResult


class ExecutionBackend(Protocol):
    """Run one already-prepared stage and return its declared receipt."""

    name: str

    def execute(self, function: StageCallable, context: RunContext) -> StageResult: ...


__all__ = ["ExecutionBackend"]
