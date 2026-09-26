"""The single RDKit boundary for parsing and canonical identity.

Before this module the chemistry primitives lived in `bio/ugi_distributional_applicability.py` and
were reached by importing its private `_canonical` and `_molecule` -- one consumer even aliases that
module to the name `chemistry` to make the call sites read sensibly. Seventeen modules depend on it.

Three canonical forms exist here because the codebase genuinely needs three, and conflating any of
them would be a correctness change rather than a cleanup:

  `canonical_constitution`            stereo-free; accepts a disconnected input (a salt, `CCN.Cl`)
  `canonical_connected_constitution`  stereo-free; rejects one, since a lipid product is one molecule
  `canonical_isomeric`                keeps stereochemistry, for procurement and component work

An audit executing every canonicalizer in the package against a battery covering stereochemistry,
salts, charge, isotopes and aromatic form found **zero divergences of molecular identity**: the
stereo-free ones all agree. They differ only in what input they admit. A separate survey of
`MolToSmiles` call sites found 50 explicitly stereo-free and 25 explicitly stereo-preserving, which
is why the third form exists rather than being folded into the first.

**Stereo-free is not incidental.** `isomericSmiles=False` discards E/Z, chirality and isotope
labels, which is what makes two source strings collapse to one constitutional graph. Turning it on
would silently change what counts as the same molecule everywhere.

**Caching covers strings, never molecules.** `Chem.Mol` is mutable, so handing the same cached
object to two callers would let one mutate what the other is reading -- a bug that would surface as
irreproducible chemistry rather than an exception. Only the string-to-string functions are memoized.

**The memoization is not a measured speedup, and should not be described as one.** The restructuring
plan assumed that 274 parse sites with no caching meant repeated canonicalization was a bottleneck.
Measured on 12,000 canonicalizations of 1,500 real lipid SMILES from `results/m0_03`, with each
structure seen eight times, the cache took 10,500 hits and the run was **1.0x** -- indistinguishable
from uncached. RDKit canonicalizes a 93-character lipid in microseconds. The cache is kept because
it is bounded and free, not because it earns anything; anyone hunting a slow pipeline should profile
rather than assume this is the cost.
"""

from __future__ import annotations

from functools import lru_cache

from rdkit import Chem, RDLogger

from forge.core.types import CanonicalSmiles, Smiles

# RDKit logs parse failures to stderr by default. Failures here are raised, not printed, and the
# chatter otherwise buries real output in any run that probes candidate structures.
RDLogger.DisableLog("rdApp.error")  # type: ignore[attr-defined]  # rdkit ships no stubs

CACHE_SIZE = 100_000


class ChemError(ValueError):
    """A SMILES string cannot be parsed, or violates a declared structural requirement."""


_Error = type[Exception] | None


def _translate(exc: ChemError, error: _Error) -> Exception:
    """Re-raise a ChemError as the caller's own exception type, preserving the cause.

    Every local helper this boundary replaces wraps failures in its module's own error class, and
    callers -- including tests -- catch that type, so without this not one of them is a drop-in
    replacement. Same convention as `core.io.read_json_object`, so one rule covers both. The
    original is kept as `__cause__`: a bare "invalid structure" with no underlying error is a dead
    end when the real problem is one malformed row in a large ledger.
    """
    return exc if error is None else error(str(exc))


def parse_smiles(smiles: str, *, error: type[Exception] | None = None) -> Chem.Mol:
    """Parse SMILES into a molecule, raising rather than returning None.

    Deliberately not cached: `Chem.Mol` is mutable, and a shared instance would let one caller's
    edit leak into another's read. Callers that only need identity should use the canonical
    functions below, which are cached.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        exc = ChemError(f"invalid molecular graph: {smiles!r}")
        raise _translate(exc, error) from exc
    return molecule


@lru_cache(maxsize=CACHE_SIZE)
def _canonical_constitution(smiles: str) -> CanonicalSmiles:
    return CanonicalSmiles(
        Chem.MolToSmiles(parse_smiles(smiles), canonical=True, isomericSmiles=False)
    )


def canonical_constitution(smiles: str, *, error: type[Exception] | None = None) -> CanonicalSmiles:
    """Canonical, stereo-free SMILES. Disconnected input is preserved as-is.

    Matches the behaviour of the six existing canonicalizers that accept salts, including
    `bio.ugi_distributional_applicability._canonical`, which most of the package routes through.
    """
    try:
        return _canonical_constitution(smiles)
    except ChemError as exc:
        raise _translate(exc, error) from exc


@lru_cache(maxsize=CACHE_SIZE)
def _canonical_connected_constitution(smiles: str) -> CanonicalSmiles:
    molecule = parse_smiles(smiles)
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ChemError(f"expected a single connected molecule: {smiles!r}")
    return CanonicalSmiles(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False))


def canonical_connected_constitution(
    smiles: str, *, error: type[Exception] | None = None
) -> CanonicalSmiles:
    """Canonical, stereo-free SMILES for a single connected molecule.

    Raises on a disconnected input rather than silently accepting a salt or a mixture. A Ugi
    product that arrives in two pieces is not a product, so admitting one would let a malformed
    structure travel downstream looking well formed.
    """
    try:
        return _canonical_connected_constitution(smiles)
    except ChemError as exc:
        raise _translate(exc, error) from exc


@lru_cache(maxsize=CACHE_SIZE)
def _canonical_isomeric(smiles: str) -> CanonicalSmiles:
    return CanonicalSmiles(
        Chem.MolToSmiles(parse_smiles(smiles), canonical=True, isomericSmiles=True)
    )


def canonical_isomeric(smiles: str, *, error: type[Exception] | None = None) -> CanonicalSmiles:
    """Canonical SMILES that **keeps** stereochemistry.

    A third form, not a variant of the other two. A survey of `MolToSmiles` across the free modules
    found 50 explicitly stereo-free sites and 25 explicitly stereo-preserving ones, the latter in
    procurement and component-projection contexts where a cis and a trans isomer are different
    things you would order from different suppliers.

    Keep the distinction deliberate. The model-facing identity is constitutional and stereo-free
    (AGENTS.md), so this must never be used to decide whether two generated products are the same
    molecule -- only where the physical isomer is what matters.
    """
    try:
        return _canonical_isomeric(smiles)
    except ChemError as exc:
        raise _translate(exc, error) from exc


def is_valid(smiles: str) -> bool:
    """Whether RDKit can parse this SMILES at all."""
    return Chem.MolFromSmiles(smiles) is not None


def same_constitution(left: str, right: str) -> bool:
    """Whether two SMILES denote the same constitutional graph.

    The comparison every module was open-coding. Stereo-free, so a cis and a trans form of the
    same skeleton compare equal -- which is the model-facing identity this project declares.
    """
    return canonical_constitution(left) == canonical_constitution(right)


def cache_stats() -> dict[str, dict[str, int]]:
    """Hit and miss counts, so a slow pipeline can be shown to be re-parsing rather than guessed at."""
    return {
        name: {
            "hits": info.hits,
            "misses": info.misses,
            "size": info.currsize,
        }
        for name, info in (
            ("canonical_constitution", _canonical_constitution.cache_info()),
            ("canonical_connected_constitution", _canonical_connected_constitution.cache_info()),
            ("canonical_isomeric", _canonical_isomeric.cache_info()),
        )
    }


def clear_caches() -> None:
    """Drop memoized results. For benchmarks and for tests that measure parse counts."""
    _canonical_constitution.cache_clear()
    _canonical_connected_constitution.cache_clear()
    _canonical_isomeric.cache_clear()


__all__ = [
    "CACHE_SIZE",
    "ChemError",
    "CanonicalSmiles",
    "Smiles",
    "cache_stats",
    "canonical_connected_constitution",
    "canonical_constitution",
    "canonical_isomeric",
    "clear_caches",
    "is_valid",
    "parse_smiles",
    "same_constitution",
]
