"""Exact execution of repeatable reaction programs from a qualified registry.

This module owns chemistry execution, not evidence admission.  A forward round trip establishes
transform consistency only; the corpus builder separately decides whether a source supports the
identity and execution claim for a record.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase

from forge.assembly.api import (
    ForwardAssemblyProducts,
    ReactionProgramAtomOrigins,
    ReactionProgramCheck,
    ReactionProgramSpec,
    ReactionProgramTrace,
)
from forge.assembly.registry import (
    CompiledRegistryReaction,
    ReactionRolePolicy,
    load_compiled_registry_reaction,
)
from forge.chemistry.reactive_sites import RAW_SUBSTRUCTURE_MATCHES
from forge.core.hashing import sha256_file


class ReactionProgramError(ValueError):
    """A reaction program cannot be loaded or replayed exactly."""


def _repair_molecule(fragment: Chem.Mol) -> Chem.Mol | None:
    """Return a sanitized copy while retaining temporary atom properties and isotopes."""

    molecule = Chem.Mol(fragment)
    for _ in range(4):
        try:
            with rdBase.BlockLogs():
                Chem.SanitizeMol(molecule)
            return molecule
        except Chem.AtomValenceException as exc:
            cause = getattr(exc, "cause", None)
            atom_index = cause.GetAtomIdx() if cause is not None else None
            if atom_index is None:
                candidates = [
                    atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetNumExplicitHs()
                ]
                if not candidates:
                    return None
                atom_index = candidates[0]
            atom = molecule.GetAtomWithIdx(int(atom_index))
            if atom.GetNumExplicitHs() == 0 and not atom.GetNoImplicit():
                return None
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)
        except Exception:
            return None
    return None


def repair_template_hydrogens(fragment: Chem.Mol) -> tuple[str, Chem.Mol] | None:
    """Sanitize a reverse-template fragment after removing impossible explicit-H constraints.

    RDKit writes a disjunctive ``H1,H2`` reactant query back as a fixed explicit-H product atom
    when a SMARTS transform is inverted.  That makes valid secondary amines appear valence-four.
    The repair only clears template-imposed explicit hydrogens on atoms implicated in a valence
    failure; ordinary invalid fragments remain rejected.
    """

    molecule = _repair_molecule(fragment)
    if molecule is None:
        return None
    if len(Chem.GetMolFrags(molecule)) != 1:
        return None
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
    with rdBase.BlockLogs():
        parsed = Chem.MolFromSmiles(smiles)
    return (smiles, parsed) if parsed is not None else None


def _canonical(smiles: str) -> tuple[str, Chem.Mol]:
    with rdBase.BlockLogs():
        molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise ReactionProgramError(f"invalid connected molecular graph: {smiles!r}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False), molecule


def _role_match_count(molecule: Chem.Mol, role: ReactionRolePolicy, handle: Chem.Mol) -> int:
    if role.site_multiplicity_semantics != RAW_SUBSTRUCTURE_MATCHES:
        from forge.chemistry.reactive_sites import audit_reactive_site_multiplicity

        return audit_reactive_site_multiplicity(molecule, handle).count(
            role.site_multiplicity_semantics
        )
    return len(molecule.GetSubstructMatches(handle, uniquify=True))


def _role_accepts(
    molecule: Chem.Mol,
    role: ReactionRolePolicy,
    handle: Chem.Mol,
    forbidden: Sequence[Chem.Mol],
    *,
    terminal_policy: bool,
) -> bool:
    multiplicity = _role_match_count(molecule, role, handle)
    if terminal_policy:
        accepted = multiplicity in role.allowed_site_multiplicity
    else:
        # An accumulator is an intentional reaction intermediate.  A polyamine may retain more
        # sites than the registry's single-step enumeration policy admits, but it must expose at
        # least one valid next handle and may never violate forbidden-pattern policy.
        accepted = multiplicity >= 1
    return accepted and not any(molecule.HasSubstructMatch(pattern) for pattern in forbidden)


@dataclass(frozen=True)
class _ForwardLayer:
    products: tuple[str, ...]
    saturated: bool


@dataclass(frozen=True)
class RegistryRepeatedReactionProgram:
    """A deterministic repeated two-reactant program backed by one registry transform."""

    spec: ReactionProgramSpec
    _reaction: CompiledRegistryReaction
    registry_path: Path
    registry_sha256: str

    @classmethod
    def from_registry(
        cls,
        registry_path: Path,
        spec: ReactionProgramSpec,
        *,
        expected_sha256: str | None = None,
    ) -> RegistryRepeatedReactionProgram:
        resolved = registry_path.resolve()
        observed = str(sha256_file(resolved))
        if expected_sha256 is not None and observed != expected_sha256:
            raise ReactionProgramError(
                f"reaction registry changed: expected {expected_sha256}, found {observed}"
            )
        try:
            reaction = load_compiled_registry_reaction(resolved, reaction_id=spec.reaction_id)
        except ValueError as exc:
            raise ReactionProgramError(str(exc)) from exc
        roles = tuple(role.name for role in reaction.definition.reactant_roles)
        if roles != (spec.accumulator_role, spec.repeat_role):
            raise ReactionProgramError(
                f"program roles {spec.accumulator_role, spec.repeat_role} differ from {roles}"
            )
        if reaction.forward.GetNumReactantTemplates() != 2:
            raise ReactionProgramError("repeated programs currently require a two-reactant step")
        return cls(
            spec=spec,
            _reaction=reaction,
            registry_path=resolved,
            registry_sha256=observed,
        )

    def _single_forward_layer(
        self,
        accumulator_smiles: str,
        repeated: Chem.Mol,
        *,
        maximum_outcomes: int,
    ) -> _ForwardLayer:
        """Apply the frozen transform once, to one accumulator and one already-parsed repeat."""

        _, accumulator = _canonical(accumulator_smiles)
        with rdBase.BlockLogs():
            outcomes = self._reaction.forward.RunReactants(
                (accumulator, repeated), maxProducts=maximum_outcomes
            )
        products: set[str] = set()
        for outcome in outcomes:
            if len(outcome) != 1:
                continue
            repaired = repair_template_hydrogens(outcome[0])
            if repaired is not None:
                products.add(repaired[0])
        return _ForwardLayer(tuple(sorted(products)), len(outcomes) >= maximum_outcomes)

    def _forward_layer(
        self,
        accumulator_smiles: Sequence[str],
        repeated_smiles: str,
        *,
        maximum_outcomes: int,
        memo: dict[tuple[str, str, int], _ForwardLayer] | None = None,
    ) -> _ForwardLayer:
        """Union one repeat step over the given accumulators.

        ``memo`` is an optional caller-owned reuse table for the single-accumulator step.  That
        step is a pure function of the accumulator SMILES, the repeat SMILES and the outcome
        bound, given this program's frozen registry transform, so reusing it cannot change any
        product, saturation flag or raised error: an accumulator or repeat that fails to parse
        raises on its first, uncached occurrence and never reaches the table.  The reverse search
        in :meth:`decompose` revisits the same one-step assemblies several times per product --
        once while testing a candidate disconnection and again inside every forward replay that
        shares its prefix -- and that repetition is what the table removes.
        """

        if isinstance(maximum_outcomes, bool) or maximum_outcomes < 2:
            raise ReactionProgramError("maximum_outcomes must be an integer of at least two")
        _, repeated = _canonical(repeated_smiles)
        products: set[str] = set()
        saturated = False
        for accumulator_smiles_value in accumulator_smiles:
            key = (accumulator_smiles_value, repeated_smiles, maximum_outcomes)
            layer = None if memo is None else memo.get(key)
            if layer is None:
                layer = self._single_forward_layer(
                    accumulator_smiles_value, repeated, maximum_outcomes=maximum_outcomes
                )
                if memo is not None:
                    memo[key] = layer
            products.update(layer.products)
            saturated = saturated or layer.saturated
        return _ForwardLayer(tuple(sorted(products)), saturated)

    def check_forward(
        self,
        terminal_head_smiles: str,
        repeated_component_smiles: Sequence[str],
        product_smiles: str,
        *,
        maximum_outcomes: int = 512,
        memo: dict[tuple[str, str, int], _ForwardLayer] | None = None,
    ) -> ReactionProgramCheck:
        target, _ = _canonical(product_smiles)
        products = self.forward_products(
            terminal_head_smiles,
            repeated_component_smiles,
            maximum_outcomes=maximum_outcomes,
            memo=memo,
        )
        steps = len(repeated_component_smiles)
        return ReactionProgramCheck(
            program_id=self.spec.program_id,
            reaction_id=self.spec.reaction_id,
            exact=target in products.products,
            saturated=products.saturated,
            steps=steps,
            enumerated_outcomes_by_step=products.enumerated_outcomes_by_step,
        )

    def forward_products(
        self,
        terminal_head_smiles: str,
        repeated_component_smiles: Sequence[str],
        *,
        maximum_outcomes: int = 512,
        memo: dict[tuple[str, str, int], _ForwardLayer] | None = None,
    ) -> ForwardAssemblyProducts:
        """Execute a complete repeated program and return every unique final product."""

        steps = len(repeated_component_smiles)
        if steps < self.spec.minimum_steps or steps > self.spec.maximum_steps:
            raise ReactionProgramError(
                f"program {self.spec.program_id} received {steps} steps outside "
                f"[{self.spec.minimum_steps}, {self.spec.maximum_steps}]"
            )
        current: tuple[str, ...] = (_canonical(terminal_head_smiles)[0],)
        outcome_counts: list[int] = []
        saturated = False
        for repeated in repeated_component_smiles:
            layer = self._forward_layer(
                current, repeated, maximum_outcomes=maximum_outcomes, memo=memo
            )
            current = layer.products
            outcome_counts.append(len(current))
            saturated = saturated or layer.saturated
            if not current:
                break
        if len(outcome_counts) < steps:
            outcome_counts.extend(0 for _ in range(steps - len(outcome_counts)))
        return ForwardAssemblyProducts(
            assembly_id=self.spec.program_id,
            reaction_id=self.spec.reaction_id,
            roles=(self.spec.accumulator_role, self.spec.repeat_role),
            products=tuple(sorted(set(current))),
            saturated=saturated,
            enumerated_outcomes_by_step=tuple(outcome_counts),
        )

    def forward_traces(
        self,
        terminal_head_smiles: str,
        repeated_component_smiles: Sequence[str],
        *,
        maximum_outcomes: int = 512,
        maximum_states: int = 4096,
    ) -> tuple[ReactionProgramTrace, ...]:
        """Enumerate deterministic forward lineages without an exponential reverse search.

        Repeated-library enumeration starts from known components, so forward dynamic programming
        is both cheaper and more direct than generating a product and rediscovering its inputs by
        retrosynthesis.  One lexicographically stable lineage is retained per constitutional state;
        ``atom_origins`` still rejects any semantic ambiguity along that exact lineage.
        """

        steps = len(repeated_component_smiles)
        if steps < self.spec.minimum_steps or steps > self.spec.maximum_steps:
            raise ReactionProgramError(
                f"program {self.spec.program_id} received {steps} steps outside "
                f"[{self.spec.minimum_steps}, {self.spec.maximum_steps}]"
            )
        if isinstance(maximum_states, bool) or maximum_states < 1:
            raise ReactionProgramError("maximum_states must be a positive integer")
        head = _canonical(terminal_head_smiles)[0]
        current: dict[str, tuple[str, ...]] = {head: ()}
        ordered_repeats: list[str] = []
        memo: dict[tuple[str, str, int], _ForwardLayer] = {}
        for repeated_value in repeated_component_smiles:
            repeated = _canonical(repeated_value)[0]
            ordered_repeats.append(repeated)
            following: dict[str, tuple[str, ...]] = {}
            for accumulator, lineage in sorted(current.items()):
                layer = self._forward_layer(
                    (accumulator,), repeated, maximum_outcomes=maximum_outcomes, memo=memo
                )
                if layer.saturated:
                    raise ReactionProgramError(
                        f"program {self.spec.program_id} forward step reached "
                        f"maximum_outcomes={maximum_outcomes}"
                    )
                for product in layer.products:
                    candidate = (*lineage, product)
                    incumbent = following.get(product)
                    if incumbent is None or candidate < incumbent:
                        following[product] = candidate
            if not following:
                return ()
            if len(following) > maximum_states:
                raise ReactionProgramError(
                    f"program {self.spec.program_id} exceeded maximum_states={maximum_states}"
                )
            current = following
        return tuple(
            ReactionProgramTrace(
                program_id=self.spec.program_id,
                reaction_id=self.spec.reaction_id,
                terminal_head_smiles=head,
                repeated_component_smiles=tuple(ordered_repeats),
                intermediate_product_smiles=lineage,
            )
            for _, lineage in sorted(current.items())
        )

    def decompose(
        self,
        product_smiles: str,
        *,
        terminal_head_smiles: str | None = None,
        expected_repeat_smiles: str | None = None,
        maximum_outcomes: int = 512,
        maximum_states: int = 4096,
    ) -> tuple[ReactionProgramTrace, ...]:
        if isinstance(maximum_states, bool) or maximum_states < 1:
            raise ReactionProgramError("maximum_states must be a positive integer")
        target, _ = _canonical(product_smiles)
        terminal = _canonical(terminal_head_smiles)[0] if terminal_head_smiles is not None else None
        expected_repeat = _canonical(expected_repeat_smiles)[0] if expected_repeat_smiles else None
        roles = self._reaction.definition.reactant_roles
        queue: deque[tuple[str, tuple[str, ...], tuple[str, ...]]] = deque(
            [(target, (), (target,))]
        )
        visited: set[tuple[str, int]] = {(target, 0)}
        traces: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()
        explored = 0
        # One reuse table per decomposition.  The reverse search rebuilds the same single-step
        # assemblies repeatedly -- every accepted candidate is re-assembled once to confirm it,
        # and then again as the first step of each forward replay that shares its prefix -- and
        # the deepest products are exactly where that repetition compounds.  The table is
        # discarded with the call, so nothing accumulates across a ledger.
        memo: dict[tuple[str, str, int], _ForwardLayer] = {}
        while queue:
            current_smiles, removed_repeats, reverse_lineage = queue.popleft()
            explored += 1
            if explored > maximum_states:
                raise ReactionProgramError(
                    f"program {self.spec.program_id} exceeded maximum_states={maximum_states}"
                )
            _, current = _canonical(current_smiles)
            with rdBase.BlockLogs():
                outcomes = self._reaction.reverse.RunReactants(
                    (current,), maxProducts=maximum_outcomes
                )
            if len(outcomes) >= maximum_outcomes:
                raise ReactionProgramError(
                    f"program {self.spec.program_id} reverse step reached "
                    f"maximum_outcomes={maximum_outcomes}"
                )
            candidates: set[tuple[str, str]] = set()
            for outcome in outcomes:
                if len(outcome) != 2:
                    continue
                accumulator = repair_template_hydrogens(outcome[0])
                repeated = repair_template_hydrogens(outcome[1])
                if accumulator is None or repeated is None:
                    continue
                accumulator_smiles, accumulator_molecule = accumulator
                repeated_smiles, repeated_molecule = repeated
                if not _role_accepts(
                    accumulator_molecule,
                    roles[0],
                    self._reaction.handles[0],
                    self._reaction.forbidden[0],
                    terminal_policy=False,
                ):
                    continue
                if not _role_accepts(
                    repeated_molecule,
                    roles[1],
                    self._reaction.handles[1],
                    self._reaction.forbidden[1],
                    terminal_policy=True,
                ):
                    continue
                if expected_repeat is not None and repeated_smiles != expected_repeat:
                    continue
                rebuilt = self._forward_layer(
                    (accumulator_smiles,),
                    repeated_smiles,
                    maximum_outcomes=maximum_outcomes,
                    memo=memo,
                )
                if current_smiles not in rebuilt.products or rebuilt.saturated:
                    continue
                candidates.add((accumulator_smiles, repeated_smiles))
            for accumulator_smiles, repeated_smiles in sorted(candidates):
                removed = (*removed_repeats, repeated_smiles)
                lineage = (*reverse_lineage, accumulator_smiles)
                steps = len(removed)
                terminal_candidate = (
                    accumulator_smiles == terminal
                    if terminal is not None
                    # Open decomposition enumerates every exact depth.  Repeated-program terminal
                    # heads can intentionally retain multiple handles, so the single-step
                    # registry multiplicity policy cannot identify the deepest valid stop.
                    else True
                )
                if terminal_candidate and self.spec.minimum_steps <= steps:
                    ordered_repeats = tuple(reversed(removed))
                    intermediates = tuple(reversed(lineage[:-1]))
                    check = self.check_forward(
                        accumulator_smiles,
                        ordered_repeats,
                        target,
                        maximum_outcomes=maximum_outcomes,
                        memo=memo,
                    )
                    if check.exact and not check.saturated:
                        traces.add((accumulator_smiles, ordered_repeats, intermediates))
                if steps >= self.spec.maximum_steps or (
                    terminal is not None and accumulator_smiles == terminal
                ):
                    continue
                state = (accumulator_smiles, steps)
                if state not in visited:
                    visited.add(state)
                    queue.append((accumulator_smiles, removed, lineage))
        return tuple(
            ReactionProgramTrace(
                program_id=self.spec.program_id,
                reaction_id=self.spec.reaction_id,
                terminal_head_smiles=head,
                repeated_component_smiles=repeats,
                intermediate_product_smiles=intermediates,
            )
            for head, repeats, intermediates in sorted(traces)
        )

    @staticmethod
    def _tagged_molecule(smiles: str, isotope: int) -> Chem.Mol:
        _, molecule = _canonical(smiles)
        if any(atom.GetIsotope() for atom in molecule.GetAtoms()):
            raise ReactionProgramError("atom-origin tracing does not accept isotope-labelled input")
        for atom in molecule.GetAtoms():
            atom.SetIsotope(isotope)
        return molecule

    @staticmethod
    def _clear_origin_tags(molecule: Chem.Mol) -> Chem.Mol:
        cleared = Chem.Mol(molecule)
        for atom in cleared.GetAtoms():
            atom.SetIsotope(0)
        return cleared

    @staticmethod
    def _canonical_semantic_states(
        molecule: Chem.Mol,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
        labels_by_isotope = {1001: "accumulator", 1002: "repeat"}
        origin_labels = []
        core_labels = []
        for atom in molecule.GetAtoms():
            isotope = atom.GetIsotope()
            if isotope not in labels_by_isotope:
                raise ReactionProgramError(
                    "forward program produced an atom with no precursor-role origin"
                )
            origin_labels.append(labels_by_isotope[isotope])
            core_labels.append(
                atom.GetProp("_forge_core_position") if atom.HasProp("_forge_core_position") else ""
            )
        cleared = RegistryRepeatedReactionProgram._clear_origin_tags(molecule)
        smiles = Chem.MolToSmiles(cleared, canonical=True, isomericSmiles=False)
        canonical = Chem.MolFromSmiles(smiles)
        if canonical is None:
            raise ReactionProgramError("canonical atom-origin product did not parse")
        matches = cleared.GetSubstructMatches(canonical, uniquify=False, maxMatches=4096)
        if not matches:
            raise ReactionProgramError("canonical atom-origin product did not map to its source")
        assignments = {
            (
                tuple(origin_labels[index] for index in match),
                tuple(core_labels[index] for index in match),
            )
            for match in matches
        }
        if len(assignments) != 1:
            raise ReactionProgramError(
                "precursor-role or core-position states are ambiguous under graph symmetry"
            )
        origins, core_positions = next(iter(assignments))
        return smiles, origins, core_positions

    @staticmethod
    def _record_product_template_positions(molecule: Chem.Mol) -> None:
        """Accumulate registry product-template map positions on a forward outcome."""

        for atom in molecule.GetAtoms():
            if atom.HasProp("old_mapno"):
                atom.SetProp("_forge_core_position", f"map_{atom.GetIntProp('old_mapno')}")

    def atom_origins(self, trace: ReactionProgramTrace) -> ReactionProgramAtomOrigins:
        """Replay a trace with temporary isotopes and recover role origin for every product atom."""

        if trace.program_id != self.spec.program_id or trace.reaction_id != self.spec.reaction_id:
            raise ReactionProgramError("trace belongs to a different reaction program")
        if trace.step_count < self.spec.minimum_steps or trace.step_count > self.spec.maximum_steps:
            raise ReactionProgramError("trace step count lies outside this program")
        current = self._tagged_molecule(trace.terminal_head_smiles, 1001)
        final_smiles = ""
        final_origins: tuple[str, ...] = ()
        final_core_positions: tuple[str, ...] = ()
        for repeated_smiles, expected_product in zip(
            trace.repeated_component_smiles,
            trace.intermediate_product_smiles,
            strict=True,
        ):
            repeated = self._tagged_molecule(repeated_smiles, 1002)
            with rdBase.BlockLogs():
                outcomes = self._reaction.forward.RunReactants((current, repeated), maxProducts=512)
            candidates: list[tuple[Chem.Mol, str, tuple[str, ...], tuple[str, ...]]] = []
            expected, _ = _canonical(expected_product)
            for raw_outcome in outcomes:
                if len(raw_outcome) != 1:
                    continue
                product = _repair_molecule(raw_outcome[0])
                if product is None:
                    continue
                self._record_product_template_positions(product)
                smiles, origins, core_positions = self._canonical_semantic_states(product)
                if smiles == expected:
                    candidates.append((product, smiles, origins, core_positions))
            if not candidates:
                raise ReactionProgramError("atom-origin replay did not reconstruct an exact step")
            semantic_assignments = {
                (origins, core_positions) for _, _, origins, core_positions in candidates
            }
            if len(semantic_assignments) != 1:
                raise ReactionProgramError(
                    "exact step has ambiguous precursor-role or core-position states"
                )
            candidates.sort(key=lambda value: Chem.MolToSmiles(value[0], canonical=False))
            current, final_smiles, final_origins, final_core_positions = candidates[0]
        return ReactionProgramAtomOrigins(
            program_id=self.spec.program_id,
            canonical_product_smiles=final_smiles,
            atom_origins=final_origins,
            core_positions=final_core_positions,
            step_count=trace.step_count,
        )


__all__ = [
    "ReactionProgramError",
    "RegistryRepeatedReactionProgram",
    "repair_template_hydrogens",
]
