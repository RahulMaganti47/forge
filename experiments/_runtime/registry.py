"""Explicit experiment-stage registration; specifications never import arbitrary code."""

from __future__ import annotations

from collections.abc import Callable

from experiments._runtime.errors import RegistryError
from experiments._runtime.stage import StageCallable


class StageRegistry:
    """A small auditable mapping from stable implementation ids to callables."""

    def __init__(self) -> None:
        self._stages: dict[str, StageCallable] = {}

    def register(self, implementation_id: str, function: StageCallable) -> StageCallable:
        if implementation_id in self._stages:
            raise RegistryError(f"stage implementation already registered: {implementation_id}")
        self._stages[implementation_id] = function
        return function

    def decorator(self, implementation_id: str) -> Callable[[StageCallable], StageCallable]:
        def register(function: StageCallable) -> StageCallable:
            return self.register(implementation_id, function)

        return register

    def resolve(self, implementation_id: str) -> StageCallable:
        try:
            return self._stages[implementation_id]
        except KeyError as error:
            known = ", ".join(sorted(self._stages)) or "none"
            raise RegistryError(
                f"unknown stage implementation {implementation_id!r}; registered: {known}"
            ) from error

    def identifiers(self) -> tuple[str, ...]:
        return tuple(sorted(self._stages))


registry = StageRegistry()
stage = registry.decorator


__all__ = ["StageRegistry", "registry", "stage"]
