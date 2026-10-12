"""Tests for per-instrument (inst_tol) tolerance in DAG peak-coverage stats.

Exercises the ``inst_type`` / ``inst_type_tol`` path of ``compute_frag_peak_stats`` that
``preproc_scripts/03_prepare_dag_feats.py`` now wires in to emit ``*_inst_tol_h*`` columns.
"""

import torch as th

from fragnnet.models.fragnnet.frag_utils import compute_frag_peak_stats
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF

# Matches INST_TYPE_LOSS_TOL in 03_prepare_dag_feats.py / the training configs.
INST_TYPE_LOSS_TOL = {
    "ORB": {"rel": 1.0e-5, "min_mz": 200.0},  # eff tol at m/z<200 == 1e-5 * 200 = 0.002 Da
    "QTOF": {"rel": 2.0e-5, "min_mz": 750.0},  # eff tol at m/z<750 == 2e-5 * 750 = 0.015 Da
}

# One neutral fragment of mass 100 Da; single non-isotope peak.
_FORMULA_PEAK_MZS = th.tensor([[100.0]])
_FORMULA_PEAK_PROBS = th.tensor([[1.0]])
_IDX_BY_H_DELTA = [{0}]
_PREC_TYPE = "[M+H]+"
_ION_MZ = 100.0 + PREC_TYPE_TO_MASS_DIFF[_PREC_TYPE]  # theoretical [M+H]+ fragment ion m/z (~101)


def _recall(inst_type: str, delta: float) -> float:
    """Recall for two peaks: one on the ion, one offset by ``delta`` Da, under inst_tol."""
    peaks = [(_ION_MZ, 1.0), (_ION_MZ + delta, 1.0)]
    result = compute_frag_peak_stats(
        peaks,
        _FORMULA_PEAK_MZS,
        _FORMULA_PEAK_PROBS,
        _IDX_BY_H_DELTA,
        prec_mz=200.0,
        allowed_h_transfer=1,
        tolerance=10.0,
        prec_type=_PREC_TYPE,
        is_ppm=True,
        inst_type=inst_type,
        inst_type_tol=INST_TYPE_LOSS_TOL,
    )
    return float(result.iloc[0])  # recall is the first element of the returned Series


def test_ft_tolerance_rejects_peak_qtof_accepts():
    """A peak 0.008 Da off is outside FT's 0.002 Da window but inside QTOF's 0.015 Da window."""
    # Only the on-ion peak matches under FT -> recall 0.5.
    assert _recall("ORB", 0.008) == 0.5
    # Both peaks match under QTOF -> recall 1.0.
    assert _recall("QTOF", 0.008) == 1.0


def test_unlisted_inst_type_falls_back_to_10ppm():
    """An instrument absent from the dict uses the 10 ppm base tolerance (~0.001 Da at m/z 101)."""
    # 0.008 Da >> 10 ppm at m/z ~101, so the offset peak is rejected -> recall 0.5.
    assert _recall("IT", 0.008) == 0.5


def test_both_peaks_within_all_windows():
    """A tiny 0.0005 Da offset is inside every window; both peaks match for all inst types."""
    for inst_type in ("ORB", "QTOF", "IT"):
        assert _recall(inst_type, 0.0005) == 1.0
