"""Lipid-native topology context that does not depend on fragment identities."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
from rdkit import Chem

from forge.model.defog_feasibility import FeasibilityError

LIPID_REGION_NAMES = ("head", "interface", "tail")
HEAD_REGION = 0
INTERFACE_REGION = 1
TAIL_REGION = 2


def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    if atom.GetSymbol() != "C":
        return False
    return any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetSymbol() in {"O", "S"}
        for bond in atom.GetBonds()
    )


def _is_amide_like_nitrogen(atom: Chem.Atom) -> bool:
    if atom.GetSymbol() != "N":
        return False
    return any(_is_carbonyl_carbon(neighbor) for neighbor in atom.GetNeighbors())


def _adjacency(molecule: Chem.Mol) -> list[list[int]]:
    """Materialize the neighbor lists once instead of per breadth-first step.

    ``GetAtomWithIdx(i).GetNeighbors()`` builds a fresh atom sequence on every visit, which is the
    dominant cost of a Python traversal.  One pass over the bonds yields the same adjacency.
    """

    adjacency: list[list[int]] = [[] for _ in range(molecule.GetNumAtoms())]
    for bond in molecule.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        adjacency[begin].append(end)
        adjacency[end].append(begin)
    return adjacency


def _breadth_first(adjacency: list[list[int]], start: int) -> list[int]:
    distances = [-1] * len(adjacency)
    distances[start] = 0
    queue = deque([start])
    while queue:
        atom_index = queue.popleft()
        next_distance = distances[atom_index] + 1
        for neighbor_index in adjacency[atom_index]:
            if distances[neighbor_index] < 0:
                distances[neighbor_index] = next_distance
                queue.append(neighbor_index)
    if any(distance < 0 for distance in distances):
        raise FeasibilityError("lipid-native root selection requires one connected molecule")
    return distances


def _distances_from(molecule: Chem.Mol, start: int) -> list[int]:
    adjacency = _adjacency(molecule)
    if not 0 <= start < len(adjacency):
        raise IndexError("list assignment index out of range")
    return _breadth_first(adjacency, start)


def _local_hetero_count(molecule: Chem.Mol, start: int, radius: int = 2) -> int:
    distances = _distances_from(molecule, start)
    return sum(
        distance <= radius and molecule.GetAtomWithIdx(index).GetSymbol() not in {"C", "H", "F"}
        for index, distance in enumerate(distances)
    )


def _highest(candidates: list[int], coordinate: Callable[[int], int]) -> list[int]:
    """Keep every candidate attaining the maximum of one tie-break coordinate, in order."""

    scores = [coordinate(index) for index in candidates]
    best = max(scores)
    return [index for index, score in zip(candidates, scores, strict=True) if score == best]


def select_lipid_polar_root(molecule: Chem.Mol) -> int:
    """Select a deterministic polar head anchor without using a component catalog.

    The priority favors ionizable, non-amide nitrogens and phosphorus atoms,
    then local heteroatom context and graph centrality. Canonical ranks resolve
    symmetry and make the result independent of input atom order.
    """

    if molecule.GetNumAtoms() == 0:
        raise FeasibilityError("cannot select a root for an empty molecule")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise FeasibilityError("lipid-native root selection requires one connected molecule")
    atoms = list(molecule.GetAtoms())
    # One carbonyl pass per molecule instead of one per neighbor lookup; the amide-like nitrogen
    # rule is exactly `_is_amide_like_nitrogen` with this table substituted for its inner call.
    carbonyl = [_is_carbonyl_carbon(atom) for atom in atoms]
    primary_roles: list[int] = []
    for atom in atoms:
        symbol = atom.GetSymbol()
        charge = atom.GetFormalCharge()
        non_amide_nitrogen = symbol == "N" and not any(
            carbonyl[neighbor.GetIdx()] for neighbor in atom.GetNeighbors()
        )
        primary_roles.append(
            5
            if non_amide_nitrogen and charge > 0
            else (
                4
                if non_amide_nitrogen
                else (
                    3 if symbol == "P" else 2 if symbol == "N" else 1 if symbol in {"O", "S"} else 0
                )
            )
        )
    best_role = max(primary_roles)
    candidates = [
        index for index, primary_role in enumerate(primary_roles) if primary_role == best_role
    ]
    # The tie-break is a lexicographic maximum, so resolve it one coordinate at a time and stop as
    # soon as a single candidate survives.  This is the same argmax -- ties keep their original
    # order at every stage and the final canonical rank is unique -- but the two expensive
    # coordinates, whole-graph eccentricity and the canonical ranking, are usually never reached.
    hetero = [atom.GetSymbol() not in {"C", "H", "F"} for atom in atoms]

    def local_hetero_count(index: int) -> int:
        # Exactly `_local_hetero_count` at radius two: the closed two-hop neighborhood of a
        # connected molecule is the set of atoms at graph distance at most two.
        neighborhood = {index}
        for neighbor in atoms[index].GetNeighbors():
            neighborhood.add(neighbor.GetIdx())
            neighborhood.update(second.GetIdx() for second in neighbor.GetNeighbors())
        return sum(hetero[member] for member in neighborhood)

    remaining = candidates
    for coordinate in (
        local_hetero_count,
        lambda index: int(atoms[index].GetFormalCharge() != 0),
        lambda index: atoms[index].GetDegree(),
    ):
        if len(remaining) == 1:
            return remaining[0]
        remaining = _highest(remaining, coordinate)
    if len(remaining) == 1:
        return remaining[0]
    adjacency = _adjacency(molecule)
    remaining = _highest(remaining, lambda index: -max(_breadth_first(adjacency, index)))
    if len(remaining) == 1:
        return remaining[0]
    ranks = list(Chem.CanonicalRankAtoms(molecule, breakTies=True))
    return max(remaining, key=lambda index: -ranks[index])


def rooted_distances(molecule: Chem.Mol, root: int) -> tuple[int, ...]:
    """Return exact graph distances from a validated root."""

    if not 0 <= root < molecule.GetNumAtoms():
        raise FeasibilityError("root index lies outside the molecule")
    return tuple(_distances_from(molecule, root))


def assign_lipid_regions(
    molecule: Chem.Mol, root: int, *, distances: tuple[int, ...] | None = None
) -> np.ndarray:
    """Assign generic structural regions without component or fragment identities.

    The frozen rule was calibrated against atom-mapped Ugi products. It treats
    the polar-root neighborhood and its small rings as head-like, heteroatoms
    and carbonyl centers as interfacial, and remote carbon-rich atoms as tails.

    ``distances`` is an optional pass-through for a caller that has already computed
    ``rooted_distances(molecule, root)``.  It is validated against the same contract rather than
    trusted, so supplying it can only save a repeated traversal.
    """

    if distances is None:
        distances = rooted_distances(molecule, root)
    elif len(distances) != molecule.GetNumAtoms() or distances[root] != 0:
        raise FeasibilityError("supplied rooted distances do not describe this molecule and root")
    head_ring_atoms: set[int] = set()
    for ring in molecule.GetRingInfo().AtomRings():
        if any(distances[index] <= 3 for index in ring):
            head_ring_atoms.update(ring)
    regions = np.full(molecule.GetNumAtoms(), TAIL_REGION, dtype=np.int64)
    atoms = list(molecule.GetAtoms())
    # As in `select_lipid_polar_root`, one carbonyl pass replaces the repeated neighbor scan that
    # `_is_amide_like_nitrogen` would otherwise run for every nitrogen.
    carbonyl = [_is_carbonyl_carbon(atom) for atom in atoms]
    for atom in atoms:
        index = atom.GetIdx()
        symbol = atom.GetSymbol()
        hard_linker = (
            symbol in {"O", "S", "P"}
            or carbonyl[index]
            or (
                symbol == "N"
                and any(carbonyl[neighbor.GetIdx()] for neighbor in atom.GetNeighbors())
            )
        )
        if (distances[index] <= 3 or index in head_ring_atoms) and not hard_linker:
            regions[index] = HEAD_REGION
        elif hard_linker or (4 <= distances[index] <= 7 and atom.GetDegree() >= 3):
            regions[index] = INTERFACE_REGION
    return regions


def tree_pair_ring_sizes(parents: np.ndarray) -> np.ndarray:
    """Return the ring size formed by adding every possible tree closure edge."""

    if parents.ndim != 1 or len(parents) == 0 or int(parents[0]) != 0:
        raise FeasibilityError("tree parents must be one-dimensional and root at zero")
    node_count = len(parents)
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for child in range(1, node_count):
        parent = int(parents[child])
        if not 0 <= parent < child:
            raise FeasibilityError("tree parent must precede its child")
        adjacency[child].append(parent)
        adjacency[parent].append(child)
    sizes = np.zeros((node_count, node_count), dtype=np.int64)
    for start in range(node_count):
        distances = [-1] * node_count
        distances[start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        sizes[start] = np.asarray(distances, dtype=np.int64) + 1
    return sizes
