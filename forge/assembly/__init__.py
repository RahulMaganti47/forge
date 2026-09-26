"""Reaction-assembly interfaces shared by generation, routing, and audits.

Assembly adapters load transforms and role policy from the vendored registry.  Callers never embed
reaction SMARTS or reinterpret a registry hit as route certification.
"""

from forge.assembly.api import (
    AssemblyAdapter,
    ForwardAssemblyCheck,
    ForwardAssemblyProducts,
    ReactionProgramAdapter,
    ReactionProgramAtomOrigins,
    ReactionProgramCheck,
    ReactionProgramSpec,
    ReactionProgramTrace,
)
from forge.assembly.program import (
    ReactionProgramError,
    RegistryRepeatedReactionProgram,
    repair_template_hydrogens,
)
from forge.assembly.registry import (
    CompiledRegistryReaction,
    ReactionRegistryEntry,
    ReactionRegistryError,
    ReactionRolePolicy,
    load_compiled_registry_reaction,
)
from forge.assembly.ugi3 import (
    Ugi3AssemblyAdapter,
    Ugi3AssemblyError,
    Ugi3DecompositionTrace,
    Ugi3RoleHandleAssessment,
    Ugi3TransformConsistentCandidate,
)

__all__ = [
    "AssemblyAdapter",
    "CompiledRegistryReaction",
    "ForwardAssemblyCheck",
    "ForwardAssemblyProducts",
    "ReactionProgramAdapter",
    "ReactionProgramAtomOrigins",
    "ReactionProgramCheck",
    "ReactionProgramError",
    "ReactionProgramSpec",
    "ReactionProgramTrace",
    "ReactionRegistryEntry",
    "ReactionRegistryError",
    "ReactionRolePolicy",
    "RegistryRepeatedReactionProgram",
    "Ugi3AssemblyAdapter",
    "Ugi3AssemblyError",
    "Ugi3DecompositionTrace",
    "Ugi3RoleHandleAssessment",
    "Ugi3TransformConsistentCandidate",
    "repair_template_hydrogens",
    "load_compiled_registry_reaction",
]
