"""Unit tests for data_utils.py utility functions.

Covers small deterministic helpers: normalization, tokenization, charge parsing, energy conversions.
"""

import numpy as np
import pytest

from fragnnet.utils import data_utils as du


class TestNormalizeInts:
    """Tests for normalize_ints."""

    def test_normalize_basic(self):
        vals = [1.0, 2.0, 3.0]
        norm = du.normalize_ints(vals)
        assert pytest.approx(sum(norm), rel=1e-9) == 1.0
        assert norm == [pytest.approx(1 / 6), pytest.approx(2 / 6), pytest.approx(3 / 6)]


class TestSplitSmiles:
    """Tests for split_smiles tokenizer."""

    def test_simple_tokens(self):
        tokens = du.split_smiles("CCO")
        assert tokens == ["C", "C", "O"]

    def test_bracket_and_ring(self):
        smiles = "C1=CC=CC=C1"  # benzene
        tokens = du.split_smiles(smiles)
        assert "1" in tokens
        assert tokens[0] == "C"


class TestListReplace:
    """Tests for list_replace mapping."""

    def test_mapping(self):
        vals = [0, 1, 2]
        mapping = {0: "a", 1: "b", 2: "c"}
        assert du.list_replace(vals, mapping) == ["a", "b", "c"]


class TestParseAceNce:
    """Tests for collision energy string parsing."""

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("10", 10.0),
            ("10 eV", 10.0),
            ("NCE=30% 15eV", 15.0),
            ("25HCD", 25.0),
            ("CE 20", 20.0),
            ("bad", np.nan),
        ],
    )
    def test_parse_ace_str(self, input_str, expected):
        result = du.parse_ace_str(input_str)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "input_str,expected",
        [
            ("NCE=30% 15eV", 30.0),
            ("NCE=25%", 25.0),
            ("40% resonant relative/normalized", 40.0),
            ("50(NCE)", 50.0),
            ("HCD (NCE 35%)", 35.0),
            ("bad", np.nan),
        ],
    )
    def test_parse_nce_str(self, input_str, expected):
        result = du.parse_nce_str(input_str)
        if np.isnan(expected):
            assert np.isnan(result)
        else:
            assert result == pytest.approx(expected)


class TestChargeAndEnergy:
    """Tests for charge parsing and energy conversions."""

    def test_get_charge(self):
        assert du.get_charge("[M+H]+") == 1
        assert du.get_charge("[M+H]2+") == 2
        assert du.get_charge("[M-H]-") == -1
        assert du.get_charge("EI") == 1

    def test_nce_ace_roundtrip(self):
        nce = 30.0
        charge = 2  # factor 0.9
        prec_mz = 500.0
        ace = du.nce_to_ace_helper(nce, charge, prec_mz)
        assert ace == pytest.approx(27.0)
        back_nce = du.ace_to_nce_helper(ace, charge, prec_mz)
        assert back_nce == pytest.approx(nce)


class TestNormalizationHelpers:
    """Tests for min-max and z-score normalization helpers."""

    def test_min_max_roundtrip(self):
        val = 5.0
        mn, mx = 0.0, 10.0
        norm = du.min_max_normalize(val, mn, mx)
        assert norm == pytest.approx(0.5, abs=1e-6)
        denorm = du.min_max_denormalize(norm, mn, mx)
        assert denorm == pytest.approx(val)

    def test_zscore_roundtrip(self):
        val = 5.0
        mean, std = 2.0, 3.0
        norm = du.zscore_normalize(val, mean, std)
        assert norm == pytest.approx(1.0, rel=1e-7)
        denorm = du.zscore_denormalize(norm, mean, std)
        assert denorm == pytest.approx(val)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
