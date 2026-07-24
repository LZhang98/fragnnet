"""Focused tests for CE-sibling detection and plot metadata."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

EVAL_SCRIPT_DP = Path(__file__).parents[1] / "scripts" / "eval"
sys.path.insert(0, str(EVAL_SCRIPT_DP))

from oracle_ce_sibling_trend_audit import add_ce_columns, add_sibling_features  # noqa: E402
from plot_ce_sibling_spectrum_groups import (  # noqa: E402
    add_peak_tracks,
    format_group_values,
    keep_tracks_with_unexplainable_peaks,
    label_for_row,
)


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


def test_add_peak_tracks_labels_only_cross_ce_matches() -> None:
    group = pd.DataFrame(
        {
            "spec_id": [1, 2],
            "inst_type": ["FT", "FT"],
            "ce_key": ["nce=10.0", "nce=20.0"],
            "ce_value": [10.0, 20.0],
            "peaks": [
                [(100.0000, 0.2), (150.0, 0.8)],
                [(100.0015, 0.6), (170.0, 0.4)],
            ],
        }
    )

    tracked, track_df = add_peak_tracks(group)

    assert tracked.iloc[0]["peak_track_ids"].tolist() == [1, 0]
    assert tracked.iloc[1]["peak_track_ids"].tolist() == [1, 0]
    assert track_df["track_id"].unique().tolist() == [1]
    assert track_df["n_ce"].unique().tolist() == [2]
    assert set(track_df["spec_id"]) == {1, 2}


def test_add_peak_tracks_never_merges_peaks_from_one_spectrum() -> None:
    group = pd.DataFrame(
        {
            "spec_id": [1, 2],
            "inst_type": ["QTOF", "QTOF"],
            "ce_key": ["ace=10.0", "ace=20.0"],
            "ce_value": [10.0, 20.0],
            "peaks": [
                [(100.000, 0.2), (100.010, 0.3)],
                [(100.005, 0.6)],
            ],
        }
    )

    tracked, _track_df = add_peak_tracks(group)

    first_ids = tracked.iloc[0]["peak_track_ids"]
    assert sum(first_ids > 0) == 1


def test_add_peak_tracks_uses_absolute_0p01_da_tolerance() -> None:
    group = pd.DataFrame(
        {
            "spec_id": [1, 2, 3],
            "inst_type": ["FT"] * 3,
            "ce_key": ["nce=10.0", "nce=20.0", "nce=30.0"],
            "ce_value": [10.0, 20.0, 30.0],
            "peaks": [[(1000.0, 0.2)], [(1000.009, 0.4)], [(1000.02, 0.6)]],
        }
    )

    tracked, track_df = add_peak_tracks(group)

    assert tracked.iloc[0]["peak_track_ids"].tolist() == [1]
    assert tracked.iloc[1]["peak_track_ids"].tolist() == [1]
    assert tracked.iloc[2]["peak_track_ids"].tolist() == [0]
    assert len(track_df) == 2


def test_track_label_requires_an_unexplainable_member() -> None:
    group = pd.DataFrame(
        {
            "peak_track_ids": [
                np.array([1, 2, 0]),
                np.array([1, 2]),
            ],
            "match_mask": [
                np.array([True, True, False]),
                np.array([False, True]),
            ],
        }
    )

    filtered = keep_tracks_with_unexplainable_peaks(group)

    assert filtered.iloc[0]["peak_track_ids"].tolist() == [1, 0, 0]
    assert filtered.iloc[1]["peak_track_ids"].tolist() == [1, 0]
