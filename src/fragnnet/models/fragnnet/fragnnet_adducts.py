"""FraGNNet's precursor-type (adduct) handling of formula rows: the elements a precursor delta
can remove, tracked-count rows that reject impossible ion formulas, and the precursor delta of
every formula row of a batch.
"""

from collections.abc import Sequence
from functools import lru_cache

import numpy as np

from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_CMF_COMP_DIFF,
    PREC_TYPE_TO_CMF_FORMULA_DIFF,
    PREC_TYPE_TO_COMP_DIFF,
    PREC_TYPE_TO_FORMULA_DIFF,
    cmf_duplicate_crf_h_shift,
    parse_formula,
)

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


def formula_row_prec_deltas(
    prec_type_strs: Sequence[str],
    formula_cumsizes: Sequence[int],
    *,
    include_cmf: bool,
    max_h_transfer: int,
) -> np.ndarray:
    """Precursor delta formula for every formula row of a batch.

    The dataset lays each sample's formula rows out as ``NULL, CRF[1:F]``, and appends
    ``CMF[1:F]`` only when the CMF channel is actually enumerated — that is, when
    ``include_cmf`` is set, the adduct has a CMF delta, and
    :func:`cmf_duplicate_crf_h_shift` does not already emit that ion as a CRF hydrogen
    shift. This reproduces that split so callers do not each re-derive it.

    Args:
        prec_type_strs: Precursor type per sample, length ``B``.
        formula_cumsizes: Formula-row offsets into the batch-concatenated arrays,
            length ``B + 1``.
        include_cmf: The ``frag_params["include_cmf"]`` setting the batch was built with.
        max_h_transfer: Hydrogen-transfer limit of the enumerated CRF channels.

    Returns:
        Object array of delta formula strings, shape ``[formula_cumsizes[-1]]``. The NULL
        row of each sample is ``""``: it stands for no fragment, so no adduct applies.
    """
    offsets = [int(o) for o in formula_cumsizes]
    if len(offsets) != len(prec_type_strs) + 1:
        raise ValueError(
            f"formula_cumsizes has {len(offsets)} entries for {len(prec_type_strs)} samples; "
            "expected batch_size + 1"
        )
    deltas = np.empty(offsets[-1], dtype=object)
    for i, prec_type in enumerate(prec_type_strs):
        start, end = offsets[i], offsets[i + 1]
        deltas[start:end] = PREC_TYPE_TO_FORMULA_DIFF[prec_type]
        deltas[start] = ""  # NULL row
        emits_cmf = (
            include_cmf
            and prec_type in PREC_TYPE_TO_CMF_FORMULA_DIFF
            and cmf_duplicate_crf_h_shift(prec_type, max_h_transfer) is None
        )
        if emits_cmf:
            # Layout: NULL, CRF[1:F], CMF[1:F] -> the CMF half starts after NULL + F rows.
            cmf_start = start + (end - start + 1) // 2
            deltas[cmf_start:end] = PREC_TYPE_TO_CMF_FORMULA_DIFF[prec_type]
    return deltas
