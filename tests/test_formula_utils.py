"""Unit tests for fragnnet.utils.formula_utils module.

Tests for formula parsing, isotopic peak generation, and mass calculations.
"""

import numpy as np
import pytest
import rdkit.Chem as Chem

from fragnnet.utils import formula_utils as fu


class TestParseFormula:
    """Parsing and element utilities."""

    def test_parse_formula_simple(self):
        result = fu.parse_formula("C6H12O6")
        assert result == {"C": 6, "H": 12, "O": 6}

    def test_parse_formula_invalid_trailing(self):
        with pytest.raises(ValueError):
            fu.parse_formula("CH4*")

    def test_get_elements_set(self):
        elems = fu.get_elements_set("C2H6O")
        assert elems == {"C", "H", "O"}


class TestMassAndIsotopes:
    """Mass table and isotope-related helpers."""

    def test_mass_matches_periodic_table(self):
        expected = Chem.GetPeriodicTable().GetMostCommonIsotopeMass("H")
        assert fu.MASS("H") == pytest.approx(expected, rel=1e-8, abs=1e-8)

    def test_get_isotopes_nonexistent(self):
        assert fu.get_isotopes(999) == []

    def test_get_isotopes_threshold(self):
        isotopes = fu.get_isotopes(1, pct_thold=0.01)
        assert isotopes[0][0] == 1  # most abundant hydrogen isotope
        assert isotopes[0][1] > 0.9

    def test_enumerate_multinomial_prob_sums_to_one(self):
        probs = fu.enumerate_multinomial(2, [0.5, 0.5], cutoff_thold=0.0)
        total = sum(p for p, _ in probs)
        assert total == pytest.approx(1.0, rel=1e-7, abs=1e-7)


class TestPeaks:
    """Peak generation helpers."""

    def test_get_peaks_for_element(self):
        peaks = fu.get_peaks_for_element("H", 2)
        assert len(peaks) > 0
        probs = sum(p for p, _ in peaks)
        assert probs == pytest.approx(1.0, rel=5e-4, abs=5e-4)

    def test_get_peaks_for_formula(self):
        masses, probs = fu.get_peaks_for_formula({"H": 2, "O": 1}, threshold=0.0)
        assert len(masses) == len(probs) > 0
        assert sum(probs) == pytest.approx(1.0, rel=1e-3, abs=1e-3)

    def test_formula_to_peak_mzs_no_isotopes(self):
        peak_mzs = fu.formula_to_peak_mzs("H2O", prec_type="[M+H]+", isotopes=False)
        assert len(peak_mzs) == 1
        expected = (
            2 * Chem.GetPeriodicTable().GetMostCommonIsotopeMass("H")
            + Chem.GetPeriodicTable().GetMostCommonIsotopeMass("O")
            + fu.PREC_TYPE_TO_MASS_DIFF["[M+H]+"]
        )
        assert peak_mzs[0] == pytest.approx(expected, rel=1e-6, abs=1e-6)

    def test_formula_to_peak_mzs_with_probs(self):
        peak_mzs, peak_probs = fu.formula_to_peak_mzs(
            "H2",
            prec_type="[M+H]+",
            isotopes=True,
            return_probs=True,
        )
        assert len(peak_mzs) == len(peak_probs) > 0
        assert sum(peak_probs) == pytest.approx(1.0, rel=5e-4, abs=5e-4)


class TestHillNotation:
    """Hill notation helper."""

    def test_get_formulae_hill_notation(self):
        result = fu.get_formulae_hill_notation({"O": 1, "H": 2})
        assert result == "H2O"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
