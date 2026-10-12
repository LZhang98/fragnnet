import re
from functools import lru_cache

import numpy as np
import rdkit.Chem as Chem
import torch as th
from pyteomics.mass import Composition

from fragnnet.utils.misc_utils import none_or_nan

PERIODIC_TABLE = Chem.GetPeriodicTable()
ELECTRON_MASS = 0.00054858

H_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("H")
NA_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Na")
N_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("N")
O_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("O")
CL_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Cl")
BR_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("Br")
K_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("K")
C_MASS = PERIODIC_TABLE.GetMostCommonIsotopeMass("C")

# Fragment-DAG size limits, shared by the DAG builders, the DAG format and core preprocessing.
# The Cython builders (``compute_frags.pyx``, ``multi_cut_bfs.pyx``) need compile-time sizes and
# keep literal copies; ``tests/test_reorg_size_constants.py`` asserts they match.
MAX_NUM_NODES = 128
MAX_NUM_EDGES = 4 * MAX_NUM_NODES
MASK_SIZE = 128
assert MASK_SIZE >= MAX_NUM_NODES, "MASK_SIZE should be larger than MAX_NUM_NODES"
assert MASK_SIZE % 64 == 0, "MASK_SIZE should be a multiple of 64"

# full list of common elements
# NOTE: this maybe a bad idea to not cover all the element
# we should handle all the elemnet
ELEMENT_TO_VE = {
    "C": 4,
    "O": 2,
    "N": 3,
    "P": 3,  # up to 5
    "S": 2,  # up to 6
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
    "Se": 2,  # up to 6, same as S
    "Si": 4,
    "As": 3,  # up to 5
    "B": 3,
    # Tier 1 metals — covalent organometallics only (C–metal single bonds)
    "Hg": 2,  # methylmercury, ethylmercury
    "Sn": 4,  # organotin (tributyltin, trimethyltin)
    "Pb": 4,  # organolead (tetraethyllead)
}
HEAVY_ELEMENTS = list(ELEMENT_TO_VE.keys())
NUM_HEAVY_ELEMENTS = len(HEAVY_ELEMENTS)
ELEMENTS = HEAVY_ELEMENTS + ["H"]
NUM_ELEMENTS = len(ELEMENTS)

CANONICAL_ELEMENT_ORDER = ["C", "H"] + sorted([elem for elem in HEAVY_ELEMENTS if elem != "C"])
CANONICAL_H_IDX = CANONICAL_ELEMENT_ORDER.index("H")

ELEMENT_TO_IDX = {elem: idx for idx, elem in enumerate(ELEMENT_TO_VE.keys())}
ELEMENT_TO_IDX["H"] = len(ELEMENT_TO_VE.keys())

IDX_TO_ELEMENT = {idx: elem for elem, idx in ELEMENT_TO_IDX.items()}

# NIST ``+Ni`` precursor labels -- ``[M+H+2i]+``, ``[M+H+4i]+``, ``[M-H+2i]-``, ``[M+Na+2i]+``
# and friends -- are NOT SUPPORTED and are deliberately absent from every table below.
#
# The trailing ``+Ni`` is NIST notation for "the M+N isotopologue was isolated as the precursor":
# the quadrupole selected the M+2 (or M+4) peak of the natural isotope envelope instead of the
# monoisotopic one. It does not name an added adduct atom. These entries used to be present and
# read the ``i`` as IODINE (``"H1I2"``, ``H_MASS + 2 * I_MASS``), putting the precursor ~253.8 Da
# off; the ``prec_types`` filter in every shipped config kept them out of training, so the error
# stayed latent.
#
# They cannot be re-added as a fixed delta, because the shift is formula dependent: +2i is
# +1.99705 Da when the carrier is 37Cl, +1.99795 for 81Br, +1.99812 for 41K, +2.00671 for two
# 13C. Real support would also have to shift every halogen-retaining *fragment*, since the heavy
# atom is carried through fragmentation -- these are not [M+H]+ spectra with an offset precursor.
# NIST20 holds ~27k ``[M+H+2i]+`` spectra plus smaller ``+4i`` and other ``+Ni`` variants.
#
# Absence from these tables is what enforces it: ``base_dataset.py`` asserts every configured
# prec_type is a key of ``PREC_TYPE_TO_MASS_DIFF``, ``clean_spectra_utils._subformula_masses``
# raises on a missing key, and the ``spec_df[...isin(PREC_TYPE_TO_MASS_DIFF)]`` filters across
# scripts/eval drop them from analyses.
PREC_TYPE_TO_MASS_DIFF = {
    "[M+H]+": H_MASS - ELECTRON_MASS,
    "[M+Na]+": NA_MASS - ELECTRON_MASS,
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
    "[M+Br]-": BR_MASS + ELECTRON_MASS,
    "[M-H-H2O]-": -3 * H_MASS - O_MASS + ELECTRON_MASS,
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
#   - a discrete adduct (Na+, K+, NH4+, Cl-, Br-, carboxylate) leaves and the fragment appears
#     with the conventional carrier instead: [F+H]+ in positive mode, [F-H]- in negative mode.
#     Halide and ammonium adducts do this by losing HCl / HBr / NH3, hence the H-1 / H1 delta.
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
    "[M+K]+": H_MASS - ELECTRON_MASS,
    "[M+NH4]+": H_MASS - ELECTRON_MASS,
    # negative stuff
    "[M-H]-": ELECTRON_MASS,  # direct anion: observed ≈ neutral mass
    "[M+Cl]-": -H_MASS + ELECTRON_MASS,  # loses HCl, so fragments are [frag-H]-
    "[M+Br]-": -H_MASS + ELECTRON_MASS,  # loses HBr, so fragments are [frag-H]-
    "[M+CH3COOH-H]-": -H_MASS + ELECTRON_MASS,
    "[M+CH3COO]-": -H_MASS + ELECTRON_MASS,
    "[M+HCOOH-H]-": -H_MASS + ELECTRON_MASS,
}

# Element-count counterparts to ``PREC_TYPE_TO_CMF_MASS_DIFF``. CMF represents
# direct charge retention: protonated positive products gain H, deprotonated
# negative products lose H, while [M+H]+/[M-H]- direct-charge products retain
# the component's neutral elemental formula.
PREC_TYPE_TO_CMF_FORMULA_DIFF = {
    "[M+H]+": "",
    "[M+Na]+": "H1",
    "[M+K]+": "H1",
    "[M+NH4]+": "H1",
    "[M-H]-": "",
    "[M+Cl]-": "H-1",
    "[M+Br]-": "H-1",
    "[M+CH3COOH-H]-": "H-1",
    "[M+CH3COO]-": "H-1",
    "[M+HCOOH-H]-": "H-1",
}

PREC_TYPE_TO_FORMULA_DIFF = {
    "[M+H]+": "H1",
    "[M+Na]+": "Na1",
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
    "[M+Br]-": "Br1",
    "[M-H-H2O]-": "H-3O-1",
    "[M+HCOOH-H]-": "C1H1O2",
    "[M+CH3COOH-H]-": "C2H3O2",
    "[M+CH3COO]-": "C2H3O2",
}

# `formula=` throughout: a positional string is read by pyteomics as a PEPTIDE SEQUENCE.
# Every value above happens to carry a digit (or be empty), which is what keeps the
# positional form correct today - adding a plain "Na" or "H" would silently yield sodium's
# or histidine's residue composition instead. Pinning the keyword removes that trap.
PREC_TYPE_TO_COMP_DIFF = {k: Composition(formula=v) for k, v in PREC_TYPE_TO_FORMULA_DIFF.items()}
PREC_TYPE_TO_CMF_COMP_DIFF = {
    k: Composition(formula=v) for k, v in PREC_TYPE_TO_CMF_FORMULA_DIFF.items()
}


NEUTRON_MASS = 1.008665
C13_SPACING = 1.003355  # ¹³C – ¹²C monoisotopic mass difference (Da); true M+1 peak spacing


def MASS(element: str | int) -> float:
    return PERIODIC_TABLE.GetMostCommonIsotopeMass(element)


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


@lru_cache(maxsize=64)
def formula_delta_to_signed_str(delta_formula: str) -> str:
    """Render a precursor delta formula in signed notation for annotation.

    Cached: the export calls this once per formula row, but the deltas come from the two
    fixed precursor tables, so there are only a handful of distinct inputs.

    ``"H1" -> "+H"``, ``"H-1" -> "-H"``, ``"H-1C-1O-2" -> "-CHO2"``. Atoms removed and atoms
    added are grouped under one sign each, losses first, so a delta that does both reads the
    way an adduct name is written: ``"H-1Na1"`` renders as ``"-H+Na"``. No shipped precursor
    delta currently mixes signs, but the rendering handles it. Element order and the dropped
    ``1`` suffix follow :func:`get_formulae_hill_notation`, so a fragment annotation reads as
    ``"C6H4O2 -H"`` against a neutral formula string.

    Args:
        delta_formula: Delta in canonical form, e.g. a ``PREC_TYPE_TO_FORMULA_DIFF`` value.

    Returns:
        Signed string, or ``""`` for a zero delta (a direct cation/anion adds nothing).
    """
    counts = {el: n for el, n in Composition(formula=delta_formula).items() if n != 0}
    rendered = ""
    for sign, keep in (("-", False), ("+", True)):
        group = {el: abs(n) for el, n in counts.items() if (n > 0) is keep}
        if group:
            rendered += sign + get_formulae_hill_notation(group)
    return rendered


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
    return set(parse_formula(formula).keys())


def formula_arr_to_str(
    formula_arr: np.ndarray, get_h_template: bool = False
) -> str | tuple[str, str]:
    """
    Convert a formula array (element counts) to a string in canonical element order.
    Optionally returns a template for hydrogen count substitution.

    Args:
        formula_arr (np.ndarray): Array of element counts (including hydrogen).
        get_h_template (bool, optional): If True, return template string for hydrogen count. Defaults to False.

    Returns:
        str | tuple[str, str]:
            - If get_h_template is False: Formula string (e.g., 'C6H12O6').
            - If get_h_template is True: Tuple of (template string with 'H{:d}' placeholder, string without hydrogen).
    """
    elem_d = {}
    for idx, count in enumerate(formula_arr):
        elem = IDX_TO_ELEMENT[idx]
        elem_d[elem] = count

    if not get_h_template:
        formula_str = ""
        for elem in CANONICAL_ELEMENT_ORDER:
            if elem in elem_d:
                count = elem_d[elem]
                if count > 0:
                    formula_str += elem
                if count > 1:
                    formula_str += str(count)
        return formula_str
    else:
        formula_str = ""
        formula_str_no_h = ""
        for elem in CANONICAL_ELEMENT_ORDER:
            if elem in elem_d:
                if elem == "H":
                    formula_str += "H{:d}"
                else:
                    count = elem_d[elem]
                    if count > 0:
                        formula_str += elem
                        formula_str_no_h += elem
                    if count > 1:
                        formula_str += str(count)
                        formula_str_no_h += str(count)
        return formula_str, formula_str_no_h


def th_long_to_mask(long: th.Tensor) -> th.Tensor:
    """convert long to mask, this can be more then one 64 bit longs in given row

    Args:
        long (_type_): N x MASK_SIZE//64 th.Tensor of int
    Returns:
        _type_: _description_
    """
    # long is N x MASK_SIZE//64
    num_dims = long.shape[1]
    long = long.reshape(long.shape[0], num_dims, 1)
    mask = 2 ** th.arange(64 - 1, -1, -1, device=long.device)
    mask = mask.reshape(1, 1, 64)
    return long.bitwise_and(mask).ne(0).reshape(long.shape[0], -1)


def th_mask_to_long(mask: th.Tensor) -> th.Tensor:
    """convert torch binary mask to 64 bits ints, MASK_SIZE//64 number of int are returned per row
    Args:
        long (_type_): N x MASK_SIZE//64 th.Tensor of int
    Returns:
        _type_: _description_
    """
    # mask is N x MASK_SIZE
    num_dims = mask.shape[1] // 64
    mask = mask.reshape(mask.shape[0], num_dims, 64)
    long = 2 ** th.arange(64 - 1, -1, -1, device=mask.device).expand(1, num_dims, 64)
    return th.sum(long * mask, dim=2).long()
