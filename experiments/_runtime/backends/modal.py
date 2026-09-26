"""In-container execution identity for the generic Modal launcher."""

from experiments._runtime.backends.local import LocalBackend


class ModalRuntimeBackend(LocalBackend):
    """Execute a stage inside Modal while preserving the ordinary stage callable.

    Resource allocation and transport happen in ``experiments._runtime.modal_app``.  Once the
    container starts, seeding and deterministic-kernel handling are identical to local execution.
    """

    name = "modal"


__all__ = ["ModalRuntimeBackend"]
