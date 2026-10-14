"""Tests for isotope peak detection.

Covers:
- ``_parse_adduct_counts``: correct parsing of adduct formula diffs
- ``_expected_mk_ratio``: adduct atoms applied at full weight
- ``detect_isotope_peaks_for_training``: co-isolation R_k based detection

Design of ``detect_isotope_peaks_for_training`` spectra
-------------------------------------------------------
The function estimates R_k from precursor M+k peaks *above* prec_mz.  Test
spectra are therefore built with two series:
  1. Fragment peaks: [M+0, M+1, ...] at ``frag_base + i * C13_SPACING``
  2. Precursor peaks: [M+0, M+1, ...] at ``prec_mz + i * C13_SPACING``

The tight FT tolerance (4 mDa for k=1, 8 mDa for k≥2) means peaks must be
placed at exact C13_SPACING multiples so the search window reliably finds them.
"""

import numpy as np
import pytest

from fragnnet.utils import isotope_cleanup_utils
from fragnnet.utils.formula_utils import C13_SPACING, parse_formula
from fragnnet.utils.isotope_cleanup_utils import (
    _expected_mk_ratio,
    _parse_adduct_counts,
    dag_formula_source,
    detect_isotope_peaks_for_training,
    detect_isotope_peaks_for_training_cleanup,
    detect_isotope_peaks_formula_aware,
    estimate_coisolation_fraction_from_precursor,
    subformula_formula_source,
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
# detect_isotope_peaks_for_training — co-isolation R_k based detection
# ---------------------------------------------------------------------------

_NM = C13_SPACING  # 1.003355 Da


def _build_spectrum(
    frag_base: float,
    frag_ints: tuple[float, ...],
    prec_mz: float,
    prec_ints: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build a spectrum with a fragment peak series and precursor isotope peaks.

    Fragment peaks are placed at ``frag_base + i * C13_SPACING``.
    Precursor peaks are placed at ``prec_mz + i * C13_SPACING`` (M+0 at prec_mz,
    M+1 at prec_mz + C13_SPACING, etc.).  The precursor M+k peaks above prec_mz
    serve as the reference from which R_k is estimated.

    Args:
        frag_base: m/z of the monoisotopic fragment peak (M+0).
        frag_ints: Intensities for fragment M+0, M+1, M+2, ...
        prec_mz: Precursor m/z (M+0 position).
        prec_ints: Intensities for precursor M+0, M+1, M+2, ...

    Returns:
        Tuple of (mzs, ints) arrays for the combined spectrum.
    """
    frag_mzs = np.array([frag_base + i * _NM for i in range(len(frag_ints))])
    prec_mzs = np.array([prec_mz + i * _NM for i in range(len(prec_ints))])
    mzs = np.concatenate([frag_mzs, prec_mzs])
    ints = np.concatenate([np.array(frag_ints, dtype=float), np.array(prec_ints, dtype=float)])
    return mzs, ints


class TestDetectIsotopePeaksForTraining:
    """Tests for co-isolation-ratio-based isotope leak detection."""

    # ----- M+1 detection -----

    def test_m1_detected_when_ratio_matches_prec(self):
        """Fragment M+1 is flagged when its ratio matches the precursor R_1."""
        # prec_mz=179.13, frag at 109.05.
        # R_1 = 57/1000 = 0.057; fragment M+1 intensity = 57 → ratio matches.
        prec_mz = 179.13
        mzs, ints = _build_spectrum(
            frag_base=109.05,
            frag_ints=(1000.0, 57.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 57.0),  # R_1 = 0.057
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert mask[1], "Fragment M+1 with ratio matching prec R_1 should be detected"

    def test_m1_not_detected_when_no_above_prec_peaks(self):
        """Without any above-prec peaks, R_k cannot be estimated → nothing flagged."""
        prec_mz = 179.13
        # Fragment only — no precursor peaks in spectrum
        mzs = np.array([109.05, 109.05 + _NM])
        ints = np.array([1000.0, 57.0])
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert not mask.any(), "Without above-prec peaks, R_k is unknown → all-False"

    def test_m1_not_detected_when_ratio_far_from_prec(self):
        """Fragment M+1 whose ratio differs greatly from prec R_1 is not flagged."""
        prec_mz = 179.13
        # R_1 = 30/1000 = 0.03; fragment M+1 = 30% → obs/R_k = 10.0 => ratio_hi=10.0, we will push it to 20.0
        mzs, ints = _build_spectrum(
            frag_base=109.05,
            frag_ints=(1000.0, 600.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 30.0),  # R_1 = 0.03
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert not mask[1], "Fragment M+1 with obs/R_k=20 >> ratio_hi=10.0 should not be flagged"

    # ----- M+2 detection with Cl precursor -----

    def test_cl_m2_detected_for_fragment_retaining_cl(self):
        """Fragment that retained Cl has M+2 ≈ 32.2%, matching prec R_2 → detected."""
        # [M+Cl]- precursor.  Precursor R_2 = 322/1000 = 0.322 (37Cl dominates).
        # Fragment [frag+Cl]- retains Cl → its M+2 ≈ 32.2% of M+0.
        prec_mz = 121.04
        frag_base = 93.02
        mzs, ints = _build_spectrum(
            frag_base=frag_base,
            frag_ints=(1000.0, 35.0, 322.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 35.0, 322.0),  # R_1=0.035, R_2=0.322
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert mask[2], "Cl-retaining fragment M+2 matching prec R_2 should be detected"

    def test_cmf_fragment_not_falsely_flagged(self):
        """CMF fragment [frag-H]- lost Cl: M+2 ≈ 1.2%, far below prec R_2 = 32.2%."""
        # Even though the precursor was [M+Cl]-, this fragment lost Cl via CMF.
        # Its M+2 comes only from 13C (C3 → ~1.2%).
        # prec R_2 = 0.322; fragment ratio = 0.014 -> 0.014 / 0.322 = 0.043 < ratio_lo=0.05. We reduce it to 14.0 (0.014 / 0.322 = 0.043)
        # 14.0 also fails the false-positive M+1 link check (14.0 / 35.0 = 0.40 > 10.0 * 0.035 = 0.35).
        prec_mz = 121.04
        frag_base = 57.03
        mzs, ints = _build_spectrum(
            frag_base=frag_base,
            frag_ints=(1000.0, 35.0, 14.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 35.0, 322.0),  # R_2 = 0.322
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert not mask[2], "CMF fragment with tiny M+2 should NOT be flagged (ratio << R_2)"

    # ----- [M+H]+ baseline -----

    def test_mh_plus_m1_detected(self):
        """[M+H]+ M+1 detection works via observed prec R_1."""
        # C9 fragment ~121.1 Da; prec ~241.2 Da.  R_1 = 95/1000 = 0.095.
        # Fragment M+1 = 9.5% → ratio matches R_1.
        prec_mz = 241.19
        mzs, ints = _build_spectrum(
            frag_base=121.10,
            frag_ints=(1000.0, 95.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 95.0),  # R_1 = 0.095
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert mask[1], "[M+H]+ M+1 should be detected"

    # ----- instrument gating -----

    def test_qtof_always_returns_false(self):
        prec_mz = 200.0
        mzs, ints = _build_spectrum(
            frag_base=100.0,
            frag_ints=(1000.0, 110.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 110.0),
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="QTOF")
        assert not mask.any(), "QTOF should never be filtered"


# ---------------------------------------------------------------------------
# detect_isotope_peaks_for_training — aggressive mode (cross-pair consistency)
# ---------------------------------------------------------------------------


class TestDetectIsotopePeaksAggressiveMode:
    """Tests for aggressive=True: R_k estimated from cross-pair consistency.

    The key identity: if co-isolation is present with ratio R, then for any two
    fragment pairs (M_i, M+1_i) and (M_j, M+1_j):

        I(M+1_i) / I(M+1_j) = I(M_i) / I(M_j)   (= R is the same for all i)

    This cross-pair consistency (low CV of r_i = I(M+1_i)/I(M_i)) allows R to
    be estimated from the spectrum itself, without requiring above-prec peaks.
    """

    def _make_consistent_spectrum(
        self,
        R: float,
        prec_mz: float,
        frag_bases: tuple[float, ...],
        m0_ints: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a spectrum with N fragments each having M+1 = R * M+0.

        No precursor M+k peaks above prec_mz are included, so the default mode
        would return all-False.  Aggressive mode should recover R from consistency.
        """
        mz_list, int_list = [], []
        for base, m0 in zip(frag_bases, m0_ints):
            mz_list += [base, base + _NM]
            int_list += [m0, R * m0]
        return np.array(mz_list), np.array(int_list)

    def test_consistent_ratio_detected_in_aggressive_mode(self):
        """Three fragments all with R=0.06 → aggressive mode detects all M+1 peaks."""
        R = 0.06
        prec_mz = 400.0
        mzs, ints = self._make_consistent_spectrum(
            R,
            prec_mz,
            frag_bases=(80.0, 120.0, 160.0),
            m0_ints=(2000.0, 1500.0, 800.0),
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            aggressive=True,
            aggressive_min_pairs=3,
            aggressive_cv=0.3,
        )
        # Indices 1, 3, 5 are the M+1 peaks (every second element)
        m1_indices = [1, 3, 5]
        assert all(mask[i] for i in m1_indices), (
            "All M+1 peaks should be detected in aggressive mode"
        )

    def test_aggressive_not_triggered_when_above_prec_available(self):
        """When above-prec peaks exist, aggressive fallback is not needed."""
        prec_mz = 300.0
        # Build with above-prec peaks — primary path should handle this (R_1 = 0.07)
        mzs, ints = _build_spectrum(
            frag_base=100.0,
            frag_ints=(1000.0, 70.0),
            prec_mz=prec_mz,
            prec_ints=(2000.0, 140.0),
        )
        mask_default = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            aggressive=False,
        )
        mask_aggressive = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            aggressive=True,
        )
        # Both modes should give the same result when above-prec peaks are present
        np.testing.assert_array_equal(mask_default, mask_aggressive)

    def test_inconsistent_ratios_not_flagged_in_aggressive_mode(self):
        """Fragments with scattered r_i (high CV) → aggressive mode returns all-False."""
        prec_mz = 400.0
        # Three fragments with very different M+1/M+0 ratios → not co-isolation
        mzs = np.array(
            [
                80.0,
                80.0 + _NM,  # r = 0.02
                120.0,
                120.0 + _NM,  # r = 0.20
                160.0,
                160.0 + _NM,  # r = 0.50
            ]
        )
        ints = np.array([2000.0, 40.0, 1500.0, 300.0, 800.0, 400.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            aggressive=True,
            aggressive_min_pairs=3,
            aggressive_cv=0.3,
        )
        assert not mask.any(), "Inconsistent ratios (high CV) should not be flagged"

    def test_aggressive_requires_min_pairs(self):
        """Fewer pairs than min_pairs → aggressive mode returns all-False."""
        R = 0.06
        prec_mz = 400.0
        # Only 2 pairs, but min_pairs=3
        mzs, ints = self._make_consistent_spectrum(
            R,
            prec_mz,
            frag_bases=(80.0, 120.0),
            m0_ints=(2000.0, 1500.0),
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            aggressive=True,
            aggressive_min_pairs=3,
        )
        assert not mask.any(), "Fewer than min_pairs should not trigger aggressive detection"


# ---------------------------------------------------------------------------
# detect_isotope_peaks_for_training — formula-aware co-isolation (corrected physics)
# ---------------------------------------------------------------------------


class TestDetectIsotopePeaksFormulaMode:
    """Tests for formula-aware co-isolation detection.

    Physics: the isolation window co-selects fraction f of the M+k precursor
    isotopologue.  Every fragment gains a satellite at +k×C13 with intensity

        I(M+k_frag) = f × R_exp_frag_i × I(M+0_frag)

    At the precursor level:  R_k / R_exp_prec = f  (always ≤ 1.0).
    The old ``min_leak_factor ≥ 1.5`` was impossible; the correct gate is
    ``f ≥ min_coiso_fraction`` (default 0.05).
    """

    @staticmethod
    def _build_coiso_spectrum(
        formula: str,
        prec_type: str,
        prec_mz: float,
        f: float,
        frag_bases: tuple[float, ...],
        m0_ints: tuple[float, ...],
        with_prec_residuals: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build a spectrum with physics-correct co-isolation intensities."""
        from fragnnet.utils.formula_utils import parse_formula

        mol_counts = parse_formula(formula)
        adduct_counts, adduct_mass = _parse_adduct_counts(prec_type)
        prec_neutral = prec_mz - adduct_mass

        r_exp_prec1 = _expected_mk_ratio(
            mol_counts, mol_scale=1.0, k=1, adduct_counts=adduct_counts
        )

        mz_list: list[float] = []
        int_list: list[float] = []

        for frag_mz, m0_int in zip(frag_bases, m0_ints):
            mol_scale = max(0.0, min(1.0, (frag_mz - adduct_mass) / prec_neutral))
            r_exp_frag = _expected_mk_ratio(
                mol_counts, mol_scale=mol_scale, k=1, adduct_counts=adduct_counts
            )
            mz_list += [frag_mz, frag_mz + _NM]
            int_list += [m0_int, f * r_exp_frag * m0_int]

        if with_prec_residuals:
            mz_list += [prec_mz, prec_mz + _NM]
            int_list += [1000.0, f * r_exp_prec1 * 1000.0]

        return np.array(mz_list), np.array(int_list)

    def test_genuine_coiso_detected_primary_mode(self):
        """f=0.3 co-isolation detected via precursor residuals in formula mode."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
        )
        # Index 1 is the fragment M+1 peak
        assert mask[1], "Fragment M+1 from f=0.3 co-isolation should be detected"
        assert not mask[0], "Fragment M+0 should not be flagged"

    def test_precursor_f_estimator_recovers_coiso_fraction(self):
        """Precursor residuals provide f that can be reused at group level."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,
        )
        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
        )
        assert f_by_k[1] == pytest.approx(0.3, rel=1e-6)

    def test_precursor_f_estimator_recovers_m1_and_m2_when_envelope_matches(self):
        """Consistent above-precursor M2/M1 residuals recover both f values."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        f = 0.3
        mol_counts = parse_formula(formula)
        adduct_counts, _ = _parse_adduct_counts(prec_type)
        r1 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=1, adduct_counts=adduct_counts)
        r2 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=2, adduct_counts=adduct_counts)
        mzs = np.array([prec_mz, prec_mz + _NM, prec_mz + 2 * _NM])
        ints = np.array([1000.0, f * r1 * 1000.0, f * r2 * 1000.0])

        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            formula=formula,
            prec_type=prec_type,
            max_isotope=2,
        )

        assert f_by_k[1] == pytest.approx(f, rel=1e-6)
        assert f_by_k[2] == pytest.approx(f, rel=1e-6)

    def test_precursor_f_estimator_rejects_bad_m2_m1_envelope(self):
        """A stray M+2 residual is rejected by the explicit M2/M1 envelope check."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        mol_counts = parse_formula(formula)
        adduct_counts, _ = _parse_adduct_counts(prec_type)
        r1 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=1, adduct_counts=adduct_counts)
        r2 = _expected_mk_ratio(mol_counts, mol_scale=1.0, k=2, adduct_counts=adduct_counts)
        mzs = np.array([prec_mz, prec_mz + _NM, prec_mz + 2 * _NM])
        # Both individual f values pass min/max gates, and f2/f1=8 would pass the
        # broad fragment ratio_hi=10 gate.  The tighter precursor M2/M1 envelope
        # check rejects the pair as incoherent above-precursor evidence.
        ints = np.array([1000.0, 0.05 * r1 * 1000.0, 0.4 * r2 * 1000.0])

        f_by_k = estimate_coisolation_fraction_from_precursor(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            formula=formula,
            prec_type=prec_type,
            max_isotope=2,
            ratio_hi=10.0,
            precursor_envelope_hi=5.0,
        )

        assert f_by_k == {}

    def test_group_level_f_detects_spectrum_without_precursor_residuals(self):
        """Borrowed group f fills the gap when this spectrum has no precursor f."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=False,
        )
        mask_without_group_f = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            aggressive=False,
        )
        mask_with_group_f = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            aggressive=False,
            coiso_fraction_by_k={1: 0.3},
        )
        assert not mask_without_group_f.any(), "No local f and no fallback should not fire"
        assert mask_with_group_f[1], "Group-level f should recover the M+1 coisolation peak"

    def test_coiso_below_min_fraction_not_detected(self):
        """f=0.01 (1%) is below the default min_coiso_fraction=0.05 → all-False."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.01,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
        )
        assert not mask.any(), "1% co-isolation below min_coiso_fraction=0.05 should not fire"

    def test_genuine_coiso_detected_aggressive_formula(self):
        """f=0.3, no above-prec peaks: aggressive formula mode detects from fragment pairs."""
        mzs, ints = self._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=400.0,
            f=0.3,
            frag_bases=(80.0, 120.0, 160.0),
            m0_ints=(2000.0, 1500.0, 800.0),
            with_prec_residuals=False,
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=400.0,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            aggressive=True,
            aggressive_min_pairs=3,
        )
        m1_indices = [1, 3, 5]
        assert all(mask[i] for i in m1_indices), (
            "All M+1 co-isolation peaks should be detected in aggressive formula mode"
        )

    def test_k1_kept_when_k2_absent(self):
        """k=2 below noise for small molecule: k=1 is still kept (not discarded)."""
        # Small molecule: M+2 residual ≈ f × R_exp_prec[2] × 1000 ≈ very small → absent
        mzs, ints = self._build_coiso_spectrum(
            formula="C5H8O2",
            prec_type="[M+H]+",
            prec_mz=101.060,
            f=0.5,
            frag_bases=(57.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=True,  # only k=1 residual; k=2 too tiny to include
        )
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=101.060,
            inst_type="ORB",
            formula="C5H8O2",
            prec_type="[M+H]+",
            max_isotope=2,
        )
        assert mask[1], "Fragment M+1 should be detected even without k=2 residual peak"

    def test_ratio_lo_20pct_detected(self):
        """Fragment M+1 at 25% of expected R_1 is caught with ratio_lo=0.2."""
        # prec R_1 = 100/1000 = 0.10; fragment M+1 = 2.5% → obs/R_k = 0.25.
        # Below old ratio_lo=0.5 but above new ratio_lo=0.2.
        prec_mz = 300.0
        mzs, ints = _build_spectrum(
            frag_base=150.0,
            frag_ints=(1000.0, 25.0),
            prec_mz=prec_mz,
            prec_ints=(1000.0, 100.0),
        )
        mask = detect_isotope_peaks_for_training(mzs, ints, prec_mz=prec_mz, inst_type="ORB")
        assert mask[1], "Fragment M+1 at 25% of R_k should be flagged with ratio_lo=0.2"

    def test_two_pair_aggressive_detected(self):
        """Two consistent pairs (aggressive_min_pairs=2) are enough to detect co-isolation."""
        R = 0.07
        prec_mz = 350.0
        # Only 2 fragment pairs; old default (min_pairs=3) would miss this.
        mzs = np.array(
            [
                80.0,
                80.0 + _NM,
                130.0,
                130.0 + _NM,
            ]
        )
        ints = np.array([2000.0, R * 2000.0, 1500.0, R * 1500.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            aggressive=True,
        )
        assert mask[1] and mask[3], "Two consistent pairs should be enough for aggressive detection"

    def test_single_pair_formula_fallback(self):
        """Single fragment pair with large f detected via formula single-pair fallback."""
        # f=0.4 >> aggressive_single_pair_min_f=0.15; only 1 pair visible.
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import _parse_adduct_counts

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        mol_counts = parse_formula(formula)
        adduct_counts, adduct_mass = _parse_adduct_counts(prec_type)
        prec_neutral = prec_mz - adduct_mass
        f = 0.4
        frag_mz = 120.0
        mol_scale = max(0.0, min(1.0, (frag_mz - adduct_mass) / prec_neutral))
        r_exp_frag = _expected_mk_ratio(mol_counts, mol_scale, k=1, adduct_counts=adduct_counts)

        # Only one fragment pair, no above-prec peaks.
        mzs = np.array([frag_mz, frag_mz + _NM])
        ints = np.array([2000.0, f * r_exp_frag * 2000.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            formula=formula,
            prec_type=prec_type,
            aggressive=True,
        )
        assert mask[1], "Single-pair formula fallback should detect large-f co-isolation"

    def test_single_pair_formula_fallback_small_f_rejected(self):
        """Single fragment pair with f=0.01 (below threshold) is not flagged."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import _parse_adduct_counts

        formula = "C10H14N2O5"
        prec_type = "[M+H]+"
        prec_mz = 243.098
        mol_counts = parse_formula(formula)
        adduct_counts, adduct_mass = _parse_adduct_counts(prec_type)
        prec_neutral = prec_mz - adduct_mass
        f = 0.01  # below aggressive_single_pair_min_f=0.05
        frag_mz = 120.0
        mol_scale = max(0.0, min(1.0, (frag_mz - adduct_mass) / prec_neutral))
        r_exp_frag = _expected_mk_ratio(mol_counts, mol_scale, k=1, adduct_counts=adduct_counts)

        mzs = np.array([frag_mz, frag_mz + _NM])
        ints = np.array([2000.0, f * r_exp_frag * 2000.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            formula=formula,
            prec_type=prec_type,
            aggressive=True,
        )
        assert not mask.any(), "Single-pair with f below threshold should not be flagged"

    def test_single_pair_fallback_uses_denominator_corrected_k1_math(self):
        """For f≈1, M+1 precursor neutral-loss contribution must not under-estimate f."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H22"
        frag_formula = "C5H11"
        prec_mz = 143.179
        frag_mz = 71.086
        f = 1.0

        r_prec = _expected_mk_ratio(parse_formula(formula), 1.0, k=1)
        r_frag = _expected_mk_ratio(parse_formula(frag_formula), 1.0, k=1)
        obs = (f * r_frag) / (1.0 + f * (r_prec - r_frag))

        mzs = np.array([frag_mz, frag_mz + _NM])
        ints = np.array([2000.0, obs * 2000.0])
        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="ORB",
            formula=formula,
            prec_type="[M+H]+",
            aggressive=True,
            aggressive_single_pair_min_f=0.99,
            formula_source=dag_formula_source(
                np.array([frag_mz]), np.array([frag_formula], dtype=object)
            ),
        )
        assert mask[1], "Corrected k=1 fallback should recover f≈1 and flag the isotope peak"

    def test_dag_mono_protection_keeps_plausible_monoisotopic_peak(self):
        """A shifted peak can be protected when the DAG also explains it as monoisotopic."""
        from fragnnet.utils.formula_utils import parse_formula

        formula = "C10H22"
        frag_formula = "C5H11"
        competing_formula = "C4H9N"
        prec_mz = 143.179
        frag_mz = 71.086
        isotope_mz = frag_mz + _NM

        r_prec = _expected_mk_ratio(parse_formula(formula), 1.0, k=1)
        r_frag = _expected_mk_ratio(parse_formula(frag_formula), 1.0, k=1)
        obs = r_frag / (1.0 + (r_prec - r_frag))

        mzs = np.array([frag_mz, isotope_mz])
        ints = np.array([2000.0, obs * 2000.0])
        kwargs = {
            "mzs": mzs,
            "ints": ints,
            "prec_mz": prec_mz,
            "inst_type": "ORB",
            "formula": formula,
            "prec_type": "[M+H]+",
            "aggressive": True,
            "aggressive_single_pair_min_f": 0.99,
            "formula_source": dag_formula_source(
                np.array([frag_mz, isotope_mz]),
                np.array([frag_formula, competing_formula], dtype=object),
            ),
        }
        mask_unprotected = detect_isotope_peaks_for_training(**kwargs)
        mask_protected = detect_isotope_peaks_for_training(**kwargs, protect_known_mono=True)

        assert mask_unprotected[1]
        assert not mask_protected[1]


# ---------------------------------------------------------------------------
# detect_isotope_peaks_formula_aware
# ---------------------------------------------------------------------------


class TestIsotopeOffsetsFromFormula:
    """Tests for pyopenms-backed isotopologue offset computation."""

    def test_carbon_only_m1_dominates(self):
        """For a carbon-only formula, M+1 should be the largest offset."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import isotope_offsets_from_formula

        offsets = isotope_offsets_from_formula(parse_formula("C10H14N2O5"))
        assert len(offsets) >= 1
        # First offset should be ~1.003 Da (13C spacing)
        assert offsets[0][0] == pytest.approx(1.003355, abs=0.001)

    def test_bromine_m2_larger_than_m1(self):
        """For C10H8Br2, M+2 relative intensity should exceed M+1."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import isotope_offsets_from_formula

        offsets = isotope_offsets_from_formula(parse_formula("C10H8Br2"))
        # Build offset → intensity map
        offset_map = {round(off, 1): rel for off, rel in offsets}
        m1_rel = offset_map.get(1.0, 0.0)
        m2_rel = offset_map.get(2.0, 0.0)
        assert m2_rel > m1_rel, "Br2 M+2 should dominate over M+1"
        assert m2_rel > 0.4, "Br2 M+2 relative intensity should be ~0.46"

    def test_empty_formula_returns_empty(self):
        from fragnnet.utils.isotope_cleanup_utils import isotope_offsets_from_formula

        assert isotope_offsets_from_formula({}) == []

    def test_threshold_filters_low_peaks(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import isotope_offsets_from_formula

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
        from fragnnet.utils.isotope_cleanup_utils import detect_isotope_peaks_formula_aware

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
        from fragnnet.utils.isotope_cleanup_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C10H8Br2")
        mono = 285.899
        mzs = np.array([mono, mono + 1.003, mono + 2.007, mono + 3.010, mono + 4.013])
        ints = np.array([100.0, 5.0, 97.0, 2.0, 24.0])
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec, mz_tol=0.02)
        assert mask[2], "M+2 (large Br peak) should be detected as isotope"

    def test_intensity_gate_rejects_noise(self):
        """A peak at the right offset but wrong intensity is not flagged when check_intensity=True."""
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        mono = 180.063
        # Noise at M+1 position that is MORE intense than monoisotopic — not a real isotope
        mzs = np.array([mono, mono + 1.003])
        ints = np.array([100.0, 500.0])  # noise is 5× monoisotopic
        mask = detect_isotope_peaks_formula_aware(mzs, ints, ec, check_intensity=True)
        assert not mask[1], "A peak more intense than its monoisotopic should not be flagged"

    def test_empty_spectrum(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        mask = detect_isotope_peaks_formula_aware(np.array([]), np.array([]), ec)
        assert mask.shape == (0,)

    def test_wrong_shape_raises(self):
        from fragnnet.utils.formula_utils import parse_formula
        from fragnnet.utils.isotope_cleanup_utils import detect_isotope_peaks_formula_aware

        ec = parse_formula("C6H12O6")
        with pytest.raises(ValueError):
            detect_isotope_peaks_formula_aware(np.array([100.0, 101.0]), np.array([1.0]), ec)


class TestTrainingIsotopeCleanup:
    """Tests for monoisotopic target cleanup masks."""

    def test_regular_isotope_peak_removed(self):
        mono = 120.0
        mzs = np.array([mono, mono + C13_SPACING, 150.0])
        ints = np.array([1000.0, 80.0, 500.0])
        mask = detect_isotope_peaks_for_training_cleanup(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            remove_regular_isotopes=True,
            remove_coisolation=False,
        )
        assert mask.tolist() == [False, True, False]

    def test_group_level_coiso_peak_removed_without_regular_path(self):
        mzs, ints = TestDetectIsotopePeaksFormulaMode._build_coiso_spectrum(
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            prec_mz=243.098,
            f=0.3,
            frag_bases=(120.0,),
            m0_ints=(2000.0,),
            with_prec_residuals=False,
        )
        mask = detect_isotope_peaks_for_training_cleanup(
            mzs,
            ints,
            prec_mz=243.098,
            inst_type="ORB",
            formula="C10H14N2O5",
            prec_type="[M+H]+",
            remove_regular_isotopes=False,
            remove_coisolation=True,
            coiso_fraction_by_k={1: 0.3},
        )
        assert mask.tolist() == [False, True]


# ---------------------------------------------------------------------------
# formula sources: DAG vs subformula
# ---------------------------------------------------------------------------


class TestFormulaSources:
    """Where the fragment composition comes from, and what happens when it is ambiguous."""

    FORMULA = "C10H22"
    FRAG_FORMULA = "C5H11"
    PREC_MZ = 143.179
    FRAG_MZ = 71.086

    def _spectrum(self):
        """One fragment and its M+1 satellite at exactly the f=1 co-isolation ratio."""
        from fragnnet.utils.formula_utils import parse_formula

        r_prec = _expected_mk_ratio(parse_formula(self.FORMULA), 1.0, k=1)
        r_frag = _expected_mk_ratio(parse_formula(self.FRAG_FORMULA), 1.0, k=1)
        obs = r_frag / (1.0 + (r_prec - r_frag))
        mzs = np.array([self.FRAG_MZ, self.FRAG_MZ + _NM])
        ints = np.array([2000.0, obs * 2000.0])
        return mzs, ints

    def _detect(self, **kwargs):
        mzs, ints = self._spectrum()
        return detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=self.PREC_MZ,
            inst_type="ORB",
            formula=self.FORMULA,
            prec_type="[M+H]+",
            aggressive=True,
            aggressive_single_pair_min_f=0.99,
            **kwargs,
        )

    def test_subformula_source_returns_the_fragment_composition(self):
        source = subformula_formula_source(self.FORMULA, "[M+H]+", 0.004)

        assert {"C": 5, "H": 11} in source(self.FRAG_MZ)

    def test_subformula_source_is_empty_off_the_grid(self):
        source = subformula_formula_source(self.FORMULA, "[M+H]+", 0.004)

        assert source(self.FRAG_MZ + 0.5) == []

    def test_subformula_source_validates_eagerly(self):
        with pytest.raises(ValueError, match="Unsupported precursor type"):
            subformula_formula_source(self.FORMULA, "[M+Xx]+", 0.004)

    def test_subformula_source_matches_the_dag_source_on_an_unambiguous_fragment(self):
        """The DAG is one way to name C5H11; decomposing the m/z is another."""
        dag_mask = self._detect(
            formula_source=dag_formula_source(
                np.array([self.FRAG_MZ]), np.array([self.FRAG_FORMULA], dtype=object)
            )
        )
        subformula_mask = self._detect(
            formula_source=subformula_formula_source(self.FORMULA, "[M+H]+", 0.004)
        )

        assert dag_mask.tolist() == [False, True]
        assert subformula_mask.tolist() == dag_mask.tolist()

    def test_candidate_rule_decides_when_candidates_disagree(self):
        """Several compositions make the expected ratio an interval, not a point.

        A synthetic three-candidate source isolates the rule: the middle one is the true
        fragment, so the median behaves exactly as the DAG source does, while the outliers
        separate ``any`` from ``all``. The shipped ratio window is 0.05x-10x, wide enough to
        swallow a 9x disagreement, so the window is tightened here to make the rule visible -
        which is itself the point: the rule only bites once the window is tight.
        """
        ambiguous = lambda _mz: [  # noqa: E731
            {"C": 1, "H": 3, "N": 3, "O": 1},
            {"C": 5, "H": 11},
            {"C": 9, "H": 11},
        ]
        window = {"ratio_lo": 0.5, "ratio_hi": 2.0}

        assert self._detect(formula_source=ambiguous, candidate_rule="any", **window)[1]
        assert self._detect(formula_source=ambiguous, candidate_rule="median", **window)[1]
        assert not self._detect(formula_source=ambiguous, candidate_rule="all", **window)[1]

    def test_protect_known_mono_needs_a_source(self):
        """Protection is a statement about the source, so without one it cannot fire."""
        assert self._detect(formula_source=None, protect_known_mono=True)[1]

    def test_empty_source_falls_back_to_the_mz_proxy(self):
        """No candidate is not an error; it is the pre-existing m/z-scaled behavior."""
        empty = lambda _mz: []  # noqa: E731

        assert self._detect(formula_source=empty).tolist() == self._detect().tolist()


class TestRatioGate:
    def test_one_candidate_makes_every_rule_agree(self):
        expected = np.array([0.05])

        for rule in ("median", "all", "any"):
            assert isotope_cleanup_utils._ratio_gate(0.06, expected, 0.5, 2.0, rule)
            assert not isotope_cleanup_utils._ratio_gate(0.5, expected, 0.5, 2.0, rule)

    def test_rules_differ_once_candidates_disagree(self):
        expected = np.array([0.05, 0.08])

        # 0.026 is inside 0.05's window [0.025, 0.1] and outside 0.08's [0.04, 0.16].
        assert isotope_cleanup_utils._ratio_gate(0.026, expected, 0.5, 2.0, "any")
        assert not isotope_cleanup_utils._ratio_gate(0.026, expected, 0.5, 2.0, "all")
        # The median of two is their mean, 0.065, whose window starts at 0.0325.
        assert not isotope_cleanup_utils._ratio_gate(0.026, expected, 0.5, 2.0, "median")
        # 0.09 is inside both windows, so even the intersection accepts it.
        assert isotope_cleanup_utils._ratio_gate(0.09, expected, 0.5, 2.0, "all")

    def test_unknown_rule_raises(self):
        with pytest.raises(ValueError, match="unknown candidate_rule"):
            isotope_cleanup_utils._ratio_gate(0.1, np.array([0.1]), 0.5, 2.0, "mean")


# ---------------------------------------------------------------------------
# Regular-isotope branch: characterization of the 2026-09-20 audit findings
# ---------------------------------------------------------------------------


class TestRegularIsotopeBranchKnownDefects:
    """Pin the audited behaviour of `detect_isotope_peaks_formula_aware` and its caller.

    These tests assert what the code does today, not what it should do. Each one names the
    defect it records, so a fix inverts the assertion rather than discovering the behaviour
    again. See `git show e65339ed:docs/plans/subformula_isotope_detection_plan.md` and
    ``docs/reference/detect_isotope_peaks_tuning.md``. The whole branch is off in every
    template, which is why these are recorded rather than fixed.
    """

    CL37 = 1.99705  # true 37Cl - 35Cl spacing
    BR81 = 1.99795  # true 81Br - 79Br spacing
    COARSE_M2 = 2.00671  # what the coarse generator reports for every formula

    def test_coarse_generator_gives_every_formula_the_same_m2_offset(self):
        """DEFECT 1: the M+2 offset ignores which heavy isotope carries it."""
        from fragnnet.utils.isotope_cleanup_utils import isotope_offsets_from_formula

        for formula in ("C10H8Br2", "C6H5Cl", "C20H30N2O3"):
            offsets = dict(isotope_offsets_from_formula(parse_formula(formula), max_isotope=2))
            m2_offset = next(off for off in offsets if off > 1.5)

            assert m2_offset == pytest.approx(self.COARSE_M2, abs=1e-5)
        # The real spacings it should have used differ by ~9 mDa, far outside FT's 4 mDa.
        assert pytest.approx(0.00876, abs=1e-5) == self.COARSE_M2 - self.BR81
        assert pytest.approx(0.00966, abs=1e-5) == self.COARSE_M2 - self.CL37

    def test_real_bromine_m2_is_unmatchable_at_the_ft_window(self):
        """DEFECT 1: at the tolerance FT actually uses, the biggest isotope peak is missed."""
        element_counts = parse_formula("C10H8Br2")
        m0 = 285.899
        mzs = np.array([m0, m0 + _NM, m0 + self.BR81])
        ints = np.array([100.0, 11.0, 195.0])

        ft_mask = detect_isotope_peaks_formula_aware(mzs, ints, element_counts, mz_tol=0.004)
        loose_mask = detect_isotope_peaks_formula_aware(mzs, ints, element_counts, mz_tol=0.02)

        assert not ft_mask[2], "M+2 unexpectedly matched at 4 mDa - defect 1 may be fixed"
        assert loose_mask[2], "the docstring example only holds at the loose default"
        assert ft_mask[1] and loose_mask[1], "the 13C M+1 is matched either way"

    def test_a_ladder_collapses_because_a_flagged_peak_can_still_be_a_parent(self):
        """DEFECT 3: no cascade guard, so everything above the first peak goes."""
        m0 = 285.899
        mzs = m0 + np.arange(5) * _NM
        ints = np.array([100.0, 80.0, 60.0, 40.0, 20.0])

        mask = detect_isotope_peaks_formula_aware(
            mzs, ints, parse_formula("C20H30N2O3"), mz_tol=0.004
        )

        assert mask.tolist() == [False, True, True, True, True]

    def test_the_monoisotopic_parent_is_the_most_intense_peak_not_the_nearest(self):
        """DEFECT 5: picking the strongest parent lowers the observed ratio, easing the gate."""
        m0 = 285.899
        # Two candidate parents inside the window; the nearer one is far weaker.
        mzs = np.array([m0 - 0.003, m0, m0 + _NM])
        ints = np.array([1000.0, 1.0, 20.0])

        mask = detect_isotope_peaks_formula_aware(
            mzs,
            ints,
            parse_formula("C20H30N2O3"),
            mz_tol=0.004,
            intensity_ratio_lo=0.5,
            intensity_ratio_hi=2.0,
        )

        # Against the nearest parent (1.0) the ratio would be 20; against the strongest
        # (1000.0) it is 0.02. Expected M+1 is 0.228, so only the second passes [0.5x, 2x].
        assert not mask[2]

    def test_the_cleanup_caller_disables_the_intensity_ratio_gate(self):
        """DEFECT 5: an M+1 at 37x its expected ratio is still removed as an isotope.

        The monotonicity check still applies, so the observed ratio has to stay under 1; the
        ratio gate is what would have caught a ratio that is merely far too large for the
        formula, and the cleanup passes [0.0, inf] for it.
        """
        element_counts = parse_formula("C2H6O")  # expected M+1 ~= 0.024
        m0 = 46.0419
        mzs = np.array([m0, m0 + _NM])
        ints = np.array([1000.0, 900.0])  # ratio 0.9, i.e. ~37x the expected 0.024

        gated = detect_isotope_peaks_formula_aware(
            mzs, ints, element_counts, mz_tol=0.004, check_intensity=True
        )
        as_cleanup_calls_it = detect_isotope_peaks_formula_aware(
            mzs,
            ints,
            element_counts,
            mz_tol=0.004,
            check_intensity=True,
            intensity_ratio_lo=0.0,
            intensity_ratio_hi=np.inf,
        )

        assert not gated[1]
        assert as_cleanup_calls_it[1]

    @pytest.mark.parametrize(
        ("inst_type", "expected_removed"),
        [("ORB", 0), ("QTOF", 3), ("IT", 3)],
    )
    def test_the_regular_branch_is_not_instrument_gated(self, inst_type, expected_removed):
        """DEFECT 4: `_INST_MZ_TOL`'s None means "never filter", but becomes 0.02 Da here.

        An H-transfer ladder is what the 4 mDa FT window exists to protect, and it is exactly
        what the non-FT fallback deletes.
        """
        m0 = 285.899
        mzs = m0 + np.arange(4) * 1.007825  # +H steps, 4.47 mDa off the 13C spacing
        ints = np.array([100.0, 80.0, 60.0, 40.0])

        mask = detect_isotope_peaks_for_training_cleanup(
            mzs,
            ints,
            prec_mz=400.0,
            inst_type=inst_type,
            formula="C20H30N2O3",
            prec_type="[M+H]+",
            remove_regular_isotopes=True,
            remove_coisolation=False,
        )

        assert int(mask.sum()) == expected_removed


class TestDagFormulaSourceEdges:
    def test_missing_arrays_give_no_candidates(self):
        assert dag_formula_source(None, None)(100.0) == []

    def test_mismatched_array_lengths_give_no_candidates(self):
        source = dag_formula_source(np.array([100.0, 200.0]), np.array(["CH4"], dtype=object))

        assert source(100.0) == []

    def test_an_empty_formula_string_is_not_a_candidate(self):
        source = dag_formula_source(np.array([100.0]), np.array([""], dtype=object))

        assert source(100.0) == []

    def test_an_unparseable_formula_is_dropped_rather_than_raised(self):
        source = dag_formula_source(np.array([100.0]), np.array(["not-a-formula"], dtype=object))

        assert source(100.0) == []

    def test_the_closest_node_wins_inside_the_window(self):
        source = dag_formula_source(
            np.array([100.000, 100.010]), np.array(["CH4", "C2H6"], dtype=object), 0.015
        )

        assert source(100.009) == [parse_formula("C2H6")]


class TestFragmentExpectedRatios:
    def test_one_ratio_per_candidate(self):
        ratios = isotope_cleanup_utils._fragment_expected_ratios(
            100.0, 1, parse_formula("C10H22"), {}, 1.00728, 142.17,
            formula_source=lambda _mz: [{"C": 5, "H": 11}, {"C": 9, "H": 11}],
        )

        assert ratios.shape == (2,)
        assert ratios[0] == pytest.approx(5 * 0.01103 + 11 * 0.000149, rel=1e-6)
        assert ratios[1] > ratios[0]

    def test_an_empty_source_falls_back_to_the_scaled_precursor(self):
        proxy = isotope_cleanup_utils._fragment_expected_ratios(
            100.0, 1, parse_formula("C10H22"), {}, 1.00728, 142.17, formula_source=None
        )
        empty = isotope_cleanup_utils._fragment_expected_ratios(
            100.0, 1, parse_formula("C10H22"), {}, 1.00728, 142.17,
            formula_source=lambda _mz: [],
        )

        assert empty.shape == (1,)
        assert empty == pytest.approx(proxy)

    def test_a_zero_precursor_mass_does_not_divide(self):
        """The caller guards this, but the fallback must not depend on that guard."""
        ratios = isotope_cleanup_utils._fragment_expected_ratios(
            100.0, 1, parse_formula("C10H22"), {}, 0.0, 0.0, formula_source=None
        )

        assert np.isfinite(ratios).all()

    def test_an_adduct_carrier_reaches_the_m2_ratio(self):
        """A retained Cl is all-or-nothing, and the ion composition is what carries it."""
        with_cl = isotope_cleanup_utils._fragment_expected_ratios(
            100.0, 2, parse_formula("C5H12O"), {}, 0.0, 88.09,
            formula_source=lambda _mz: [{"C": 2, "H": 5, "O": 1, "Cl": 1}],
        )
        without_cl = isotope_cleanup_utils._fragment_expected_ratios(
            100.0, 2, parse_formula("C5H12O"), {}, 0.0, 88.09,
            formula_source=lambda _mz: [{"C": 2, "H": 5, "O": 1}],
        )

        assert with_cl[0] == pytest.approx(without_cl[0] + 0.3220, abs=1e-6)

    def test_an_even_candidate_count_tests_a_ratio_no_composition_predicts(self):
        """`median` of two is their mean, so with an even count it tests a fictional fragment.

        Documented rather than fixed: it is the behaviour the DAG-matching arms were measured
        with, and it only differs from picking a side when the candidates disagree sharply.
        """
        expected = np.array([0.02, 0.10])

        # 0.058 matches neither candidate under a tight window, but sits on their mean.
        assert isotope_cleanup_utils._ratio_gate(0.058, expected, 0.9, 1.1, "median")
        assert not isotope_cleanup_utils._ratio_gate(0.058, expected, 0.9, 1.1, "any")
        assert isotope_cleanup_utils._ratio_point(expected) == pytest.approx(0.06)
