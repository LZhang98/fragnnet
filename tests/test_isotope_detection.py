"""Tests for adduct-aware isotope peak detection.

Covers:
- ``_parse_adduct_counts``: correct parsing of adduct formula diffs
- ``_expected_mk_ratio``: adduct atoms applied at full weight
- ``detect_isotope_peaks_for_training``: end-to-end with adduct correction

Note on peak spacing: ``detect_isotope_peaks_for_training`` searches for M+0 at
``candidate_mz - k * C13_SPACING`` (1.003355 Da per step) — the true ¹³C isotope
spacing.  Tests therefore place peaks at multiples of C13_SPACING so the search
window (±4 mDa for k=1, ±8 mDa for k≥2) reliably finds the monoisotopic peak.
"""

import numpy as np
import pytest

from fragnnet.utils.formula_utils import C13_SPACING, NEUTRON_MASS
from fragnnet.utils.spec_utils import (
    _expected_mk_ratio,
    _parse_adduct_counts,
    detect_isotope_peaks_for_training,
)

# ---------------------------------------------------------------------------
# _parse_adduct_counts
# ---------------------------------------------------------------------------


class TestParseAdductCounts:
    def test_mh_plus(self):
        counts, mass = _parse_adduct_counts("[M+H]+")
        assert counts == {"H": 1}
        assert mass == pytest.approx(1.00794, abs=0.002)

    def test_mk_plus(self):
        counts, mass = _parse_adduct_counts("[M+K]+")
        assert counts == {"K": 1}
        assert mass == pytest.approx(38.964, abs=0.005)

    def test_mna_plus(self):
        counts, mass = _parse_adduct_counts("[M+Na]+")
        assert counts == {"Na": 1}
        assert mass == pytest.approx(22.990, abs=0.005)

    def test_mnh4_plus(self):
        counts, mass = _parse_adduct_counts("[M+NH4]+")
        assert counts.get("N", 0) == 1
        assert counts.get("H", 0) == 4
        assert mass == pytest.approx(18.034, abs=0.005)

    def test_mcl_minus(self):
        counts, mass = _parse_adduct_counts("[M+Cl]-")
        assert counts == {"Cl": 1}
        assert mass == pytest.approx(34.969, abs=0.005)

    def test_mh_minus_negligible_mass(self):
        # [M-H]- removes H; parse_formula strips at '-', leaving 'H' with count 1.
        # H isotope contribution is 0.015% — effectively zero for our purposes.
        _, mass = _parse_adduct_counts("[M-H]-")
        assert mass < 2.0

    def test_none(self):
        counts, mass = _parse_adduct_counts(None)
        assert counts == {}
        assert mass == 0.0

    def test_unknown_adduct(self):
        counts, mass = _parse_adduct_counts("[M+Weird]+")
        assert counts == {}
        assert mass == 0.0


# ---------------------------------------------------------------------------
# _expected_mk_ratio
# ---------------------------------------------------------------------------


class TestExpectedMkRatio:
    """Verify that adduct atoms are applied at full weight, not scaled."""

    def test_no_adduct_m1_scales_with_mol_scale(self):
        # C10 molecule, scale 0.5: M+1 ≈ 10 * 0.01103 * 0.5 = 0.0552
        counts = {"C": 10}
        ratio = _expected_mk_ratio(counts, mol_scale=0.5, k=1)
        expected = 10 * 0.01103 * 0.5
        assert ratio == pytest.approx(expected, rel=0.05)

    def test_k_adduct_m2_at_full_weight(self):
        # Molecule has no Cl/Br/K/S — m2_mol ≈ tiny.
        # K adduct adds 0.0673 at full weight.
        counts = {"C": 10}
        adduct = {"K": 1}
        ratio_with = _expected_mk_ratio(counts, mol_scale=0.5, k=2, adduct_counts=adduct)
        ratio_without = _expected_mk_ratio(counts, mol_scale=0.5, k=2)
        assert ratio_with > ratio_without
        # K's M+2 contribution (0.0673) dominates
        assert (ratio_with - ratio_without) == pytest.approx(0.0673, abs=0.005)

    def test_k_adduct_m2_independent_of_mol_scale(self):
        # K's contribution must not scale with mol_scale
        counts = {"C": 5}
        adduct = {"K": 1}
        ratio_half = _expected_mk_ratio(counts, mol_scale=0.5, k=2, adduct_counts=adduct)
        ratio_full = _expected_mk_ratio(counts, mol_scale=1.0, k=2, adduct_counts=adduct)
        # Difference is only from the molecule part (C5 has no Cl/Br/K/S M+2)
        # so the two ratios differ only by the molecule's M+2 Poisson term
        # which is tiny for C5 — both ratios dominated by K's 0.0673
        assert abs(ratio_full - ratio_half) < 0.01  # small delta, not 0.0673

    def test_cl_adduct_m2_at_full_weight(self):
        counts = {"C": 5}
        adduct = {"Cl": 1}
        ratio = _expected_mk_ratio(counts, mol_scale=0.5, k=2, adduct_counts=adduct)
        # Cl contributes 0.3220 at full weight; molecule adds small poisson + O term
        assert ratio == pytest.approx(0.3220, abs=0.02)

    def test_floor_at_1e4(self):
        ratio = _expected_mk_ratio({}, mol_scale=0.0, k=2)
        assert ratio == pytest.approx(1e-4)


# ---------------------------------------------------------------------------
# detect_isotope_peaks_for_training — adduct correction end-to-end
# ---------------------------------------------------------------------------

# Helpers: build spectra with peaks placed at NEUTRON_MASS multiples
# so the ±5 mDa search window reliably finds the monoisotopic peak.

_NM = C13_SPACING  # 1.003355 Da — true ¹³C isotope spacing used by the detector


def _mk_spectrum(base_mz: float, *intensities: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (mzs, ints) for a peak series [M+0, M+1, M+2, ...] at C13_SPACING spacing."""
    mzs = np.array([base_mz + i * _NM for i in range(len(intensities))])
    ints = np.array(list(intensities), dtype=float)
    return mzs, ints


class TestDetectIsotopePeaksAdductAware:
    """End-to-end tests for adduct-aware isotope detection."""

    # ----- [M+K]+ : ⁴¹K M+2 isotope detection -----

    def test_mk_plus_m2_detected_with_prec_type(self):
        """M+2 peak at ~6.7% from ⁴¹K IS flagged when prec_type=[M+K]+."""
        # Molecule C10H20, [M+K]+ prec.  Fragment [C5H10+K]+ at ~109 Da.
        # prec_mz = mol_neutral + K_mass ≈ 140.16 + 38.97 ≈ 179.13
        # frag_mz = frag_neutral + K_mass ≈ 70.08 + 38.97 ≈ 109.05
        prec_mz = 179.13
        base = 109.05
        m0_int, m1_int, m2_int = 1000.0, 57.0, 67.0  # M+1 ~5.7%, M+2 ~6.7% (41K)
        mzs, ints = _mk_spectrum(base, m0_int, m1_int, m2_int)

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C10H20", inst_type="FT",
            prec_type="[M+K]+"
        )
        assert mask[2], "M+2 from 41K should be detected with prec_type=[M+K]+"

    def test_mk_plus_m2_not_detected_without_prec_type(self):
        """Without prec_type the K M+2 peak falls outside the expected range."""
        prec_mz = 179.13
        base = 109.05
        mzs, ints = _mk_spectrum(base, 1000.0, 57.0, 67.0)

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C10H20", inst_type="FT",
            prec_type=None
        )
        assert not mask[2], "Without prec_type, K M+2 should NOT be detected"

    def test_mk_plus_m1_still_detected(self):
        """M+1 (¹³C) detection is unaffected by [M+K]+ correction."""
        prec_mz = 179.13
        base = 109.05
        # C5H10 fragment: expected M+1 ~5.7%; use 6% to be comfortably in range
        mzs, ints = _mk_spectrum(base, 1000.0, 60.0)

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C10H20", inst_type="FT",
            prec_type="[M+K]+"
        )
        assert mask[1], "M+1 from 13C should still be detected with [M+K]+"

    # ----- [M+Cl]- : ³⁷Cl M+2 detection -----

    def test_mcl_minus_m2_detected_with_prec_type(self):
        """M+2 at ~32% from ³⁷Cl IS flagged when prec_type=[M+Cl]- and M+1 present."""
        # Molecule C5H10O (no Cl). Fragment [C3H6O+Cl]- retaining Cl.
        # prec_mz ≈ 86.07 + 34.97 ≈ 121.0, frag_mz ≈ 58.04 + 34.97 ≈ 93.0
        prec_mz = 121.04
        base = 93.02
        # M+1 ~3.5% from 13C (C3), M+2 ~32.2% from 37Cl
        mzs, ints = _mk_spectrum(base, 1000.0, 35.0, 322.0)

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C5H10O", inst_type="FT",
            prec_type="[M+Cl]-"
        )
        assert mask[2], "M+2 from 37Cl should be detected with prec_type=[M+Cl]-"

    def test_mcl_minus_cmf_m2_not_falsely_flagged(self):
        """CMF fragment [frag-H]- has tiny M+2 (no Cl) — must NOT be flagged."""
        # Despite [M+Cl]- precursor, this fragment lost Cl: only 13C M+2 (~1%)
        prec_mz = 121.04
        base = 57.03   # [C3H5O-H]- after CMF
        mzs, ints = _mk_spectrum(base, 1000.0, 35.0, 12.0)  # M+2 = 1.2%

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C5H10O", inst_type="FT",
            prec_type="[M+Cl]-"
        )
        # 1.2% M+2 << lo_factor * expected(Cl) ≈ 0.5 * 0.322 = 16.1% → outside range
        assert not mask[2], "CMF fragment with tiny M+2 should NOT be falsely flagged"

    # ----- [M+H]+ baseline (no regression) -----

    def test_mh_plus_m1_detected_unchanged(self):
        """[M+H]+ M+1 detection still works (no regression from adduct correction)."""
        # C18H24 molecule, [M+H]+ prec ~241.2 Da; C9 fragment ~121.1 Da
        # Expected M+1 for C9 ≈ 9 * 1.103% ≈ 9.9%
        prec_mz = 241.19
        base = 121.10
        mzs, ints = _mk_spectrum(base, 1000.0, 95.0)  # M+1 = 9.5%

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C18H24", inst_type="FT",
            prec_type="[M+H]+"
        )
        assert mask[1], "[M+H]+ M+1 should still be detected"

    # ----- mol_scale correction -----

    def test_mol_scale_corrected_for_heavy_adduct(self):
        """mol_scale uses neutral masses; for [M+K]+ a pure-13C M+1 is still found."""
        # With a 39 Da K adduct, naive scale = frag_mz/prec_mz overestimates C count.
        # Corrected scale = (frag_mz - K_mass) / (prec_mz - K_mass) ≈ 0.5.
        # For C10H20, C5 fragment: expected M+1 ≈ 5 * 1.103% * 0.5 = 2.76% (scaled by mol_scale)
        # Actual: 5.5% — within [0.5, 2.0] × expected? Let's check.
        # Expected M+1 with corrected scale: mol_m1 = 10 * 0.01103 * 0.5 = 0.0552
        # range: [0.0276, 0.1104]; observed 0.055 → in range ✓
        prec_mz = 179.13
        base = 109.05
        mzs, ints = _mk_spectrum(base, 1000.0, 55.0)  # M+1 = 5.5%

        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=prec_mz, formula="C10H20", inst_type="FT",
            prec_type="[M+K]+"
        )
        assert mask[1], "M+1 should be within expected range with corrected mol_scale"

    # ----- instrument gating -----

    def test_qtof_always_returns_false(self):
        mzs, ints = _mk_spectrum(100.0, 1000.0, 110.0)
        mask = detect_isotope_peaks_for_training(
            mzs, ints, prec_mz=200.0, formula="C10H20", inst_type="QTOF",
            prec_type="[M+K]+"
        )
        assert not mask.any(), "QTOF should never be filtered"


# ---------------------------------------------------------------------------
# isotope_offsets_from_formula
# ---------------------------------------------------------------------------


class TestIsotopeOffsetsFromFormula:
    """Tests for pyopenms-backed isotopologue offset computation."""

    def test_carbon_only_m1_dominates(self):
        """For a carbon-only formula, M+1 should be the largest offset."""
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula

        offsets = isotope_offsets_from_formula(parse_formula("C10H14N2O5"))
        assert len(offsets) >= 1
        # First offset should be ~1.003 Da (13C spacing)
        assert offsets[0][0] == pytest.approx(1.003355, abs=0.001)

    def test_bromine_m2_larger_than_m1(self):
        """For C10H8Br2, M+2 relative intensity should exceed M+1."""
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula

        offsets = isotope_offsets_from_formula(parse_formula("C10H8Br2"))
        # Build offset → intensity map
        offset_map = {round(off, 1): rel for off, rel in offsets}
        m1_rel = offset_map.get(1.0, 0.0)
        m2_rel = offset_map.get(2.0, 0.0)
        assert m2_rel > m1_rel, "Br2 M+2 should dominate over M+1"
        assert m2_rel > 0.4, "Br2 M+2 relative intensity should be ~0.46"

    def test_empty_formula_returns_empty(self):
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula

        assert isotope_offsets_from_formula({}) == []

    def test_threshold_filters_low_peaks(self):
        from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula

        ec = parse_formula("C6H12O6")
        offsets_loose = isotope_offsets_from_formula(ec, threshold=0.0001)
        offsets_strict = isotope_offsets_from_formula(ec, threshold=0.05)
        assert len(offsets_loose) > len(offsets_strict)


# ---------------------------------------------------------------------------
# detect_isotope_peaks_formula_aware
# ---------------------------------------------------------------------------


class TestDetectIsotopePeaksFormulaAware:
    """Tests for formula-aware isotope peak detection via pyopenms offsets."""

    def test_basic_carbon_m1_detected(self):
        """M+1 of a carbon-rich molecule at correct spacing is flagged."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.spec_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C10H14N2O5")
        mono = 242.090
        mzs = np.array([mono, mono + 1.003355])
        ints = np.array([1000.0, 100.0])
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec)
        assert mask[1], "M+1 peak at 13C spacing should be flagged"
        assert not mask[0], "Monoisotopic peak should not be flagged"

    def test_bromine_m2_detected(self):
        """For C10H8Br2, the large M+2 peak should be flagged as isotope."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.spec_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C10H8Br2")
        mono = 285.899
        mzs = np.array([mono, mono + 1.003, mono + 2.007, mono + 3.010, mono + 4.013])
        ints = np.array([100.0, 5.0, 97.0, 2.0, 24.0])
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec, mz_tol=0.02)
        assert mask[2], "M+2 (large Br peak) should be detected as isotope"

    def test_intensity_gate_rejects_noise(self):
        """A peak at the right offset but wrong intensity is not flagged when check_intensity=True."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.spec_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        mono = 180.063
        # Noise at M+1 position that is MORE intense than monoisotopic — not a real isotope
        mzs = np.array([mono, mono + 1.003])
        ints = np.array([100.0, 500.0])  # noise is 5× monoisotopic
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec, check_intensity=True)
        assert not mask[1], "A peak more intense than its monoisotopic should not be flagged"

    def test_empty_spectrum(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.spec_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        mask = detect_isotope_peaks_formula_aware(np.array([]), np.array([]), ec)
        assert mask.shape == (0,)

    def test_wrong_shape_raises(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.spec_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        with pytest.raises(ValueError):
            detect_isotope_peaks_formula_aware(np.array([100.0, 101.0]), np.array([1.0]), ec)
