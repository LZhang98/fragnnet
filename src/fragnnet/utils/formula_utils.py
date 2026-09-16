import re
from functools import lru_cache
from typing import Literal

import rdkit.Chem as Chem
from pyteomics.mass import Composition

from fragnnet.utils.misc_utils import none_or_nan

PERIODIC_TABLE = Chem.GetPeriodicTable()
ELECTRON_MASS = 0.00054858

H_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("H")
NA_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Na")
N_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("N")
O_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("O")
CL_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Cl")
K_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("K")
I_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("I")
C_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("C")

PREC_TYPE_TO_MASS_DIFF = {
    "[M+H]+": H_MASS - ELECTRON_MASS,
    "[M+Na]+": NA_MASS - ELECTRON_MASS,
    "[M+H+2i]+": H_MASS + 2 * I_MASS - ELECTRON_MASS,
    "[M+K]+": K_MASS - ELECTRON_MASS,
    "[M+NH4]+": N_MASS + 4 * H_MASS - ELECTRON_MASS,
    "[M-H2O+H]+": -H_MASS - O_MASS - ELECTRON_MASS,
    "[M-2H2O+H]+": -3 * H_MASS - 2 * O_MASS - ELECTRON_MASS,
    "[M+H-NH3]+": -2 * H_MASS - N_MASS - ELECTRON_MASS,
    "[M]+": -ELECTRON_MASS,
    "": 0.0,
    # negative stuff
    "[M-H]-": -H_MASS + ELECTRON_MASS,
    "[M-H-CO2]-": -H_MASS - C_MASS - 2 * O_MASS + ELECTRON_MASS,
    "[M+Cl]-": CL_MASS + ELECTRON_MASS,
    "[M-H-H2O]-": -3 * H_MASS - O_MASS + ELECTRON_MASS,
    "[M-H+2i]-": -H_MASS + 2 * I_MASS + ELECTRON_MASS,
    "[M+HCOOH-H]-": C_MASS + 1 * H_MASS + 2 * O_MASS + ELECTRON_MASS,
    "[M+CH3COOH-H]-": 2 * C_MASS + 3 * H_MASS + 2 * O_MASS + ELECTRON_MASS,
    "[M+CH3COO]-": 2 * C_MASS + 3 * H_MASS + 2 * O_MASS + ELECTRON_MASS,
}
# Charge migration fragmentation (CMF): the product ion does NOT carry the precursor's charge
# carrier, as opposed to charge retention fragmentation (CRF), where it does. For sodiated
# precursors this is the established definition — CMF yields nonsodiated fragment ions — and
# Xue et al., J. Am. Soc. Mass Spectrom. 2021, 32, 226 (doi:10.1021/jasms.0c00240) found such
# ions to be a substantial share of sodiated library spectra.
#
# What the fragment is left holding depends on which carrier departed:
#   - a discrete adduct (Na+, K+, NH4+, Cl-, carboxylate) leaves and the fragment appears with
#     the conventional carrier instead: [F+H]+ in positive mode, [F-H]- in negative mode.
#     Chloride and ammonium adducts do this by losing HCl / NH3, hence the H-1 / H1 delta.
#   - a proton leaves nothing behind, so the fragment carries the charge itself: [F]+ or [F]-,
#     an empty delta with only the electron in the mass. This covers [M+H]+ and [M-H]-.
#
# In-source loss precursors ([M-H2O+H]+, [M-2H2O+H]+, [M+H-NH3]+, [M-H-H2O]-, [M-H-CO2]-) are
# deliberately absent: their charge is already intrinsic to the loss cation/anion, so there is no
# carrier left to migrate away and no CMF channel to enumerate. A precursor missing here simply
# emits CRF rows only, as [M]+ and [M-] do.
#
# [M+H]+ and [M-H]- are the opposite case and are kept: their CMF channel is real and common
# (tropylium, acylium, iminium), it is just that the CRF hydrogen ladder already enumerates that
# same ion at delta_h = -1 / +1. ``cmf_duplicate_crf_h_shift`` suppresses the duplicate row when
# the ladder reaches it, and lets it through when max_h_transfer is too small to.
#
# Do not give a precursor its CRF value here: CMF(F) == CRF(F) makes every CMF row an exact m/z
# duplicate of its own CRF row, adding no candidate at all. See ``cmf_duplicate_crf_h_shift`` for
# the weaker form of the same trap, where CMF coincides with a CRF hydrogen shift.
PREC_TYPE_TO_CMF_MASS_DIFF = {
    "[M+H]+": -ELECTRON_MASS,  # direct cation: observed ≈ neutral mass
    "[M+Na]+": H_MASS - ELECTRON_MASS,
    "[M+H+2i]+": H_MASS - ELECTRON_MASS,
    "[M+K]+": H_MASS - ELECTRON_MASS,
    "[M+NH4]+": H_MASS - ELECTRON_MASS,
    # negative stuff
    "[M-H]-": ELECTRON_MASS,  # direct anion: observed ≈ neutral mass
    "[M+Cl]-": -H_MASS + ELECTRON_MASS,  # loses HCl, so fragments are [frag-H]-
    "[M-H+2i]-": -H_MASS + ELECTRON_MASS,
    "[M+CH3COOH-H]-": -H_MASS + ELECTRON_MASS,
    "[M+CH3COO]-": -H_MASS + ELECTRON_MASS,
    "[M+HCOOH-H]-": -H_MASS + ELECTRON_MASS,
}

PREC_TYPE_TO_CMF_FORMULA_DIFF = {
    "[M+H]+": "",
    "[M+Na]+": "H1",
    "[M+H+2i]+": "H1",
    "[M+K]+": "H1",
    "[M+NH4]+": "H1",
    "[M-H]-": "",
    "[M+Cl]-": "H-1",
    "[M-H+2i]-": "H-1",
    "[M+CH3COOH-H]-": "H-1",
    "[M+CH3COO]-": "H-1",
    "[M+HCOOH-H]-": "H-1",
}

PREC_TYPE_TO_FORMULA_DIFF = {
    "[M+H]+": "H1",
    "[M+Na]+": "Na1",
    "[M+H+2i]+": "H1I2",
    "[M+K]+": "K1",
    "[M+NH4]+": "N1H4",
    "[M-H2O+H]+": "O-1H-1",
    "[M-2H2O+H]+": "O-2H-3",
    "[M+H-NH3]+": "H-2N-1",
    "[M]+": "",
    # negative stuff
    "[M-H]-": "H-1",
    "[M-H-CO2]-": "H-1C-1O-2",
    "[M+Cl]-": "Cl1",
    "[M-H-H2O]-": "H-3O-1",
    "[M-H+2i]-": "H-1I2",
    "[M+HCOOH-H]-": "C1H1O2",
    "[M+CH3COOH-H]-": "C2H3O2",
    "[M+CH3COO]-": "C2H3O2",
}

PREC_TYPE_TO_COMP_DIFF = {k: Composition(v) for k, v in PREC_TYPE_TO_FORMULA_DIFF.items()}
PREC_TYPE_TO_CMF_COMP_DIFF = {k: Composition(v) for k, v in PREC_TYPE_TO_CMF_FORMULA_DIFF.items()}

# A precursor delta can only make a fragment's ion formula impossible through the elements it
# removes. Loss-type precursors such as [M-H-CO2]- subtract their neutral loss from every
# enumerated fragment, but that loss left the precursor once, from one site: a fragment that
# never carried the group cannot lose it. Tracking these few elements is enough to reject those
# channels without carrying a full composition per formula.
PREC_DELTA_REMOVED_ELEMENTS = tuple(
    sorted(
        {
            element
            for delta in (*PREC_TYPE_TO_COMP_DIFF.values(), *PREC_TYPE_TO_CMF_COMP_DIFF.values())
            for element, count in delta.items()
            if count < 0
        }
    )
)

NEUTRON_MASS = 1.008665
C13_SPACING = 1.003355  # ¹³C – ¹²C monoisotopic mass difference (Da); true M+1 peak spacing


def MASS(element: str | int) -> float:
    return PERIODIC_TABLE.GetMostCommonIsotopeMass(element)


# Layout of a tracked-count row: one column per removable element, then the total atom count.
TOTAL_ATOM_COLUMN = len(PREC_DELTA_REMOVED_ELEMENTS)


@lru_cache(maxsize=1 << 16)
def tracked_element_counts(formula: str) -> tuple[int, ...]:
    """Count the ``PREC_DELTA_REMOVED_ELEMENTS`` of a neutral formula, plus its atom total.

    The total is what rejects an ion that has been emptied out entirely, which a per-element
    check cannot see. Fragment formulas repeat heavily across a dataset, so this is cached;
    the NULL formula ``""`` counts as all zeros.
    """
    counts = parse_formula(formula) if formula else {}
    tracked = tuple(counts.get(element, 0) for element in PREC_DELTA_REMOVED_ELEMENTS)
    return (*tracked, sum(counts.values()))


def prec_type_removed_counts(prec_type: str, *, cmf: bool = False) -> dict[int, int]:
    """Map tracked-count columns to the minimum a formula needs to survive this delta.

    Includes ``TOTAL_ATOM_COLUMN``, whose minimum keeps at least one atom in the ion. Returns
    an empty mapping when nothing can be violated, so the validity check can be skipped.
    """
    delta = PREC_TYPE_TO_CMF_COMP_DIFF[prec_type] if cmf else PREC_TYPE_TO_COMP_DIFF[prec_type]
    minimums = {
        position: -count
        for position, element in enumerate(PREC_DELTA_REMOVED_ELEMENTS)
        if (count := delta.get(element, 0)) < 0
    }
    net_atoms = sum(delta.values())
    if net_atoms < 0:
        minimums[TOTAL_ATOM_COLUMN] = 1 - net_atoms
    return minimums


def cmf_duplicate_crf_h_shift(
    prec_type: str,
    max_h_transfer: int,
    *,
    tolerance: float = 1e-8,
) -> int | None:
    """Return the CRF hydrogen shift that already emits this adduct's CMF ion.

    CMF and CRF describe the same ion whenever their mass and formula deltas differ by a
    whole number of hydrogens that hydrogen transfer also reaches. For ``[M+H]+`` the CMF
    state is the CRF state at ``delta_h = -1``, so emitting both puts two formula slots on
    one m/z. Adducts whose carrier swap is not a whole number of hydrogens, such as
    ``[M+Na]+`` and ``[M+Cl]-``, stay distinct.

    Args:
        prec_type: Precursor type string, e.g. ``"[M+H]+"``.
        max_h_transfer: Hydrogen-transfer limit of the enumerated CRF channels. A shift
            outside this range is not enumerated, so CMF still contributes a new ion.
        tolerance: Absolute tolerance on the integer hydrogen shift, in hydrogen masses.

    Returns:
        The duplicated CRF hydrogen shift, or ``None`` when CMF contributes a distinct ion.
    """
    if prec_type not in PREC_TYPE_TO_CMF_MASS_DIFF:
        return None
    shift = (PREC_TYPE_TO_CMF_MASS_DIFF[prec_type] - PREC_TYPE_TO_MASS_DIFF[prec_type]) / H_MASS
    h_shift = round(shift)
    if abs(shift - h_shift) > tolerance or not -max_h_transfer <= h_shift <= max_h_transfer:
        return None
    # Equal masses are not enough: the two states must also be the same formula.
    crf_counts = PREC_TYPE_TO_COMP_DIFF[prec_type]
    cmf_counts = PREC_TYPE_TO_CMF_COMP_DIFF[prec_type]
    for element in set(crf_counts) | set(cmf_counts) | {"H"}:
        expected = crf_counts.get(element, 0) + (h_shift if element == "H" else 0)
        if cmf_counts.get(element, 0) != expected:
            return None
    return h_shift


def get_peaks_for_formula(
    element_counts: dict[str, int],
    threshold: float = 0.001,
    peaks_for_element_cache: dict = None,  # kept for API compatibility, unused
) -> tuple[list[float], list[float]]:
    """Return isotope peak masses and relative intensities for a molecular formula.

    Uses pyopenms ``CoarseIsotopePatternGenerator`` with IUPAC 2016 natural
    abundances.  Replaces the previous multinomial convolution implementation,
    which was 100–1000× slower and used an older RDKit abundance table.

    Args:
        element_counts: Element counts dict, e.g. ``{"C": 6, "H": 12, "O": 6}``.
        threshold: Minimum relative intensity to include.  Defaults to 0.001.
        peaks_for_element_cache: Ignored.  Kept for API compatibility.

    Returns:
        Tuple of ``(masses, probs)`` where each is a list of floats.  Masses
        are monoisotopic-based exact masses in Da; probs are renormalised
        relative intensities summing to ≤ 1 (peaks below threshold excluded).

    Raises:
        ValueError: If ``element_counts`` is empty or contains no positive counts.
    """
    from pyopenms import CoarseIsotopePatternGenerator, EmpiricalFormula

    formula_str = "".join(f"{el}{n}" for el, n in element_counts.items() if n > 0)
    if not formula_str:
        raise ValueError(f"element_counts has no positive counts: {element_counts}")
    emp = EmpiricalFormula(formula_str)
    gen = CoarseIsotopePatternGenerator(5)
    iso = gen.run(emp)
    iso.renormalize()
    container = iso.getContainer()
    keep = [(p.getMZ(), p.getIntensity()) for p in container if p.getIntensity() >= threshold]
    if not keep:
        return [container[0].getMZ()], [1.0]
    return [m for m, p in keep], [p for m, p in keep]


def formula_to_peak_mzs(
    formula,
    prec_type: Literal["[M+H]+", "[M-H]-", ""],
    isotopes=True,
    return_map=False,
    return_probs=False,
    peaks_for_element_cache: dict | None = None,
):
    """_summary_

    Args:
        formula (_type_): _description_
        prec_type (_type_): _description_
        isotopes (bool, optional): _description_. Defaults to True.
        return_map (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    assert not (return_map and return_probs)
    element_counts = parse_formula(formula)
    mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
    if not isotopes:
        peak_mz = 0.0
        for k, v in element_counts.items():
            peak_mz += v * PERIODIC_TABLE.GetMostCommonIsotopeMass(k)
        peak_mz += mass_diff
        peak_mzs = [peak_mz]
        peak_probs = [1.0]
    else:
        peak_mzs, peak_probs = get_peaks_for_formula(
            element_counts, peaks_for_element_cache=peaks_for_element_cache
        )
        peak_mzs = [peak_mz + mass_diff for peak_mz in peak_mzs]
    if return_map:
        return dict.fromkeys(peak_mzs, formula)
    elif return_probs:
        return peak_mzs, peak_probs
    else:
        return peak_mzs


def get_formulae_hill_notation(element_counts: dict[str, int]) -> str:
    """return formulae in string following hill notation

    Args:
        element_counts (Dict[str,int]): element configs eg:  {"C":6,"H":7,"O":6,"N":0}

    Returns:
        str: formulae string eg: BrClH2Si
    """
    formulae_string = ""
    sorted_keys = list(element_counts.keys())
    sorted_keys.sort()
    for element in sorted_keys:
        formulae_string += element
        if element_counts[element] > 1:
            formulae_string += str(element_counts[element])
    return formulae_string


def parse_formula(formula: str) -> dict[str, int]:
    """
        Return a Dict of count of each elemnt in the forumla
        NOTE: THIS DOES NOT HANDLE Condensed formulas eg. CH3CH2OH
    Args:
        formula (str): chemical forumla eg.CH4 or chemical forumla with adducts eg. CH4+H

    Raises:
        ValueError: _description_

    Returns:
        Dict[str,int]: count per element eg {"C":1, "H":4}
    """

    assert not none_or_nan(formula)
    cur_element = None
    cur_count = 1
    element_counts = {}
    if "-" in formula:
        formula = formula[: formula.index("-")]
    if "+" in formula:
        formula = formula[: formula.index("+")]
    # Use finditer to strictly parse tokens and detect unexpected characters
    pattern = re.finditer(r"[A-Z][a-z]?|\d+", formula)
    pos = 0
    for m in pattern:
        # if there's any gap between last position and this match, it's invalid
        if m.start() != pos:
            offending = formula[pos : m.start()]
            raise ValueError(
                f"Invalid substring '{offending}' at position {pos} in formula '{formula}'"
            )
        token = m.group(0)
        if token.isalpha():
            if cur_element is not None:
                assert cur_element not in element_counts
                element_counts[cur_element] = cur_count
            cur_element = token
            cur_count = 1
        elif token.isdigit():
            cur_count = int(token)
        pos = m.end()
    # any trailing characters after last token are invalid
    if pos != len(formula):
        offending = formula[pos:]
        raise ValueError(
            f"Invalid trailing substring '{offending}' at position {pos} in formula '{formula}'"
        )
    if cur_element is not None:
        assert cur_element not in element_counts
        element_counts[cur_element] = cur_count
    return element_counts


def isotope_offsets_from_formula(
    element_counts: dict[str, int],
    max_isotope: int = 4,
    threshold: float = 0.001,
) -> list[tuple[float, float]]:
    """Return isotopologue mass offsets and relative intensities for a molecular formula.

    Uses pyopenms ``CoarseIsotopePatternGenerator`` to compute the exact mass
    offset of each M+k peak from the monoisotopic M+0 peak.  These offsets
    differ from ``k * NEUTRON_MASS`` for heavy-isotope elements such as Cl
    and Br where even-mass isotopes dominate.

    Args:
        element_counts: Element counts dict as returned by ``parse_formula``.
        max_isotope: Maximum number of isotope peaks beyond M+0 to return.
            Defaults to 4.
        threshold: Minimum relative intensity to include.  Peaks below this
            threshold are discarded.  Defaults to 0.001.

    Returns:
        List of ``(offset_da, relative_intensity)`` tuples sorted by offset,
        excluding M+0 (offset ≈ 0).  Empty list if no peaks pass the threshold.

    Example:
        >>> ec = parse_formula("C10H8Br2")
        >>> offsets = isotope_offsets_from_formula(ec)
        >>> # M+2 ~0.46 dominates (81Br), not M+1 ~0.03
        >>> offsets[0][0]  # offset_da ≈ 1.0034
        1.003...
    """
    from pyopenms import CoarseIsotopePatternGenerator, EmpiricalFormula

    formula_str = "".join(f"{el}{n}" for el, n in element_counts.items() if n > 0)
    if not formula_str:
        return []
    emp = EmpiricalFormula(formula_str)
    gen = CoarseIsotopePatternGenerator(max_isotope + 1)
    iso = gen.run(emp)
    iso.renormalize()
    container = iso.getContainer()

    peaks = sorted((p.getMZ(), p.getIntensity()) for p in container)
    if not peaks:
        return []

    mono_mass = peaks[0][0]
    mono_intensity = peaks[0][1]
    return [
        (mass - mono_mass, intensity / mono_intensity)
        for mass, intensity in peaks[1:]
        if (intensity / mono_intensity) >= threshold
    ]


def get_elements_set(formula: str) -> set[str]:
    """
    method to get set of elements in a formula
    Args:
        formula (str): chemical formula eg.CH4 or chemical formula with adducts eg. CH4+H
    """
    return set(list(parse_formula(formula).keys()))
