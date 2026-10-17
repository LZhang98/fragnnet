"""Natural isotope fine structure at exact masses for MASSter ion compositions."""

from collections import defaultdict
from functools import cache, lru_cache

from fragnnet.utils.formula_utils import PERIODIC_TABLE

# Integer mass keys merge indistinguishable atom permutations without float-order errors.
# Rounding an atomic mass difference to 1e-8 Da is far below prediction mass precision.
MASS_SCALE = 100_000_000


@cache
def atomic_isotopes(element: str) -> tuple[tuple[int, int, float], ...]:
    """Return nominal offsets, exact mass increments (1e-8 Da), and natural abundances."""
    atomic_number = PERIODIC_TABLE.GetAtomicNumber(element)
    base = PERIODIC_TABLE.GetMostCommonIsotope(atomic_number)
    base_mass = PERIODIC_TABLE.GetMassForIsotope(atomic_number, base)
    rows = []
    for isotope in range(1, 300):
        abundance = PERIODIC_TABLE.GetAbundanceForIsotope(atomic_number, isotope) / 100
        if abundance:
            mass = PERIODIC_TABLE.GetMassForIsotope(atomic_number, isotope)
            rows.append((isotope - base, round((mass - base_mass) * MASS_SCALE), abundance))
    if not rows:
        return ((0, 0, 1.0),)
    total = sum(row[2] for row in rows)
    return tuple((offset, mass, probability / total) for offset, mass, probability in rows)


@lru_cache(maxsize=4096)
def exact_isotope_pattern(
    composition: tuple[tuple[str, int], ...], max_order: int, min_ratio: float
) -> tuple[tuple[int, float, float], ...]:
    """Return retained (nominal offset, exact mass shift, normalized share) fine structure.

    Convolve natural atom distributions, retaining only intermediate states that can
    reach the final nominal window [-max_order, max_order]. Bounds use every remaining
    atom, so positive/negative offsets can cancel across elements. Probability pruning
    occurs only after convolution and window normalization, not during intermediate steps.
    """
    if max_order < 1 or not 0 <= min_ratio < 1:
        raise ValueError("Invalid isotope order or minimum share")
    atoms = []
    for element, count in composition:
        if count < 0:
            raise ValueError("Ion composition has a negative atom count")
        distribution = atomic_isotopes(element)
        if len(distribution) > 1:
            atoms.extend([distribution] * count)
    remaining_low = sum(min(row[0] for row in atom) for atom in atoms)
    remaining_high = sum(max(row[0] for row in atom) for atom in atoms)
    states = {(0, 0): 1.0}
    for atom in atoms:
        remaining_low -= min(row[0] for row in atom)
        remaining_high -= max(row[0] for row in atom)
        combined = defaultdict(float)
        for (offset, mass), probability in states.items():
            for delta, increment, abundance in atom:
                next_offset = offset + delta
                if -max_order - remaining_high <= next_offset <= max_order - remaining_low:
                    combined[next_offset, mass + increment] += probability * abundance
        states = combined
    total = sum(states.values())
    kept = [
        (offset, mass / MASS_SCALE, probability / total)
        for (offset, mass), probability in sorted(states.items())
        if probability / total >= min_ratio and probability > 0
    ]
    if not kept:
        raise ValueError("No isotope fine-structure rows survive; reduce isotope_min_ratio")
    retained = sum(row[2] for row in kept)
    return tuple((offset, mass, probability / retained) for offset, mass, probability in kept)
