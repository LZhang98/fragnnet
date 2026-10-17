"""Peak lookup and halogen labels shared by the isotope probe scripts."""

from __future__ import annotations

import numpy as np


def find_peak(mzs: np.ndarray, target: float, tol: float) -> int | None:
    """Index of the peak nearest ``target`` within ``tol``, or None."""
    i = np.searchsorted(mzs, target)
    hits = [j for j in (i - 1, i) if 0 <= j < len(mzs) and abs(mzs[j] - target) <= tol]
    return min(hits, key=lambda j: abs(mzs[j] - target)) if hits else None


def halogen_class(formula: str) -> str:
    """Label a formula by its Br/Cl content: "both", "Br", "Cl" or "none"."""
    has_br, has_cl = "Br" in formula, "Cl" in formula
    return "both" if has_br and has_cl else "Br" if has_br else "Cl" if has_cl else "none"
