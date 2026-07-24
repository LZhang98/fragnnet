"""Focused tests for CE-sibling detection and plot metadata."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_SCRIPT_DP = Path(__file__).parents[1] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_SCRIPT_DP))

from oracle_ce_sibling_trend_audit import add_ce_columns, add_sibling_features
from plot_ce_sibling_spectrum_groups import format_group_values, label_for_row


def test_sibling_detection_does_not_compare_ace_with_nce() -> None:
    """ACE fallback rows must not be treated as siblings of an NCE series."""
    df = pd.DataFrame(
        {
            "mol_id": [1, 1, 1],
            "prec_type": ["[M+H]+"] * 3,
            "inst_type": ["FT"] * 3,
            "frag_mode": ["HCD"] * 3,
            "ace": [10.0, 20.0, 30.0],
            "nce": [20.0, 40.0, np.nan],
            "metric": [0.2, 0.9, 1.0],
        }
    )
    df = add_ce_columns(df)

    scored = add_sibling_features(
        df,
        ["mol_id", "prec_type", "inst_type", "frag_mode"],
        "metric",
    )

    assert scored.loc[0, "best_higher_ce"] == 0.9
    assert np.isnan(scored.loc[2, "best_lower_ce"])
    assert np.isnan(scored.loc[2, "best_higher_ce"])


def test_plot_metadata_uses_recorded_pmz_and_group_values() -> None:
    row = pd.Series(
        {
            "spec_id": 7,
            "dset": "test",
            "ce_key": "nce=20.0",
            "prec_mz": 101.12345,
            "calc_prec_mz": 101.12001,
            "top_peak_mz": 80.0,
            "metric": 0.5,
        }
    )

    label, _color = label_for_row(row, "metric")

    assert "pmz=101.1235" in label
    assert format_group_values(pd.Series([101.2, np.nan, 100.1, 101.2])) == "100.1000, 101.2000"
