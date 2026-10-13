"""MASSter's nominal-mass natural isotope patterns: per-element distributions, a composition's
pattern cropped to +/-``max_order``, and power tables by atom count (moved from
``utils/formula_utils.py``).
"""

from functools import cache, lru_cache

import numpy as np

from fragnnet.utils.formula_utils import PERIODIC_TABLE


@cache
def nominal_isotope_distribution(element: str) -> tuple[int, tuple[float, ...]]:
    """Return the lightest natural isotope's offset and its dense probability polynomial.

    Offsets are relative to the most abundant isotope. Polynomial indices are relative to
    the lightest isotope, so convolution indices are nonnegative even for Se, B and metals.
    """
    atomic_number = PERIODIC_TABLE.GetAtomicNumber(element)
    base = PERIODIC_TABLE.GetMostCommonIsotope(atomic_number)
    abundances = np.array(
        [PERIODIC_TABLE.GetAbundanceForIsotope(atomic_number, mass) for mass in range(300)]
    )
    present = np.flatnonzero(abundances)
    if not present.size:
        return 0, (1.0,)
    low, high = int(present[0]), int(present[-1])
    return low - base, tuple(abundances[low : high + 1] / abundances.sum())


@lru_cache(maxsize=65536)
def nominal_isotope_pattern(
    composition: tuple[tuple[str, int], ...], max_order: int
) -> tuple[float, ...]:
    """Convolve a composition's natural distribution, then crop to ±``max_order``.

    A lightest-isotope origin makes every polynomial index nonnegative. Only coefficients
    above the final upper index can be discarded during convolution: they cannot return.
    The returned probabilities are not renormalized after cropping.
    """
    distributions = [
        (count, nominal_isotope_distribution(element)) for element, count in composition
    ]
    origin = sum(count * low for count, (low, _) in distributions)
    upper = max_order - origin
    pattern = np.ones(1)
    for count, (_, single) in distributions:
        for _ in range(count):
            pattern = np.convolve(pattern, single)[: upper + 1]
    offsets = np.arange(-max_order, max_order + 1) - origin
    result = np.zeros(2 * max_order + 1)
    valid = (offsets >= 0) & (offsets < len(pattern))
    result[valid] = pattern[offsets[valid]]
    return tuple(result)


def nominal_isotope_power_table(element: str, max_count: int, max_order: int) -> np.ndarray:
    """Return natural patterns for 0..``max_count`` atoms, cropped after convolution.

    Column ``j`` represents offset ``j - max_order`` from the most abundant isotope.
    Intermediate support includes offsets that can cancel later; cropping does not
    renormalize the retained probabilities. Shape: (max_count + 1, 2 * max_order + 1).
    """
    low, single = nominal_isotope_distribution(element)
    # Every row up to max_count must remain recoverable, including negative-offset elements.
    upper = max_order - max_count * low
    table = np.zeros((max_count + 1, 2 * max_order + 1))
    pattern = np.ones(1)
    for count in range(max_count + 1):
        offsets = np.arange(-max_order, max_order + 1) - count * low
        valid = (offsets >= 0) & (offsets < len(pattern))
        table[count, valid] = pattern[offsets[valid]]
        if count < max_count:
            pattern = np.convolve(pattern, single)[: upper + 1]
    return table
