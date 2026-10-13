"""Near-duplicate scoring in preproc_scripts/spectraverse/dedup_train_split.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def dedup():
    path = Path(__file__).parent.parent / "preproc_scripts" / "spectraverse" / "dedup_train_split.py"
    spec = importlib.util.spec_from_file_location("dedup_train_split", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def peaks():
    return [(121.06479, 0.31), (149.05971, 1.0), (195.06519, 0.12), (338.13412, 0.45)]


def test_identical_spectra_score_one_with_zero_mz_difference(dedup, peaks):
    score, dmz = dedup.tolerance_cosine(peaks, list(peaks), ppm=5.0, min_mz=200.0)

    assert score == pytest.approx(1.0)
    assert dmz == pytest.approx(0.0)


def test_three_decimal_copy_still_matches(dedup, peaks):
    # A library that stores 3 decimals: rounding moves each m/z by at most 0.0005 Da, inside the
    # 1 mDa floor that min_mz=200 gives at 5 ppm.
    rounded = [(round(mz, 3), it) for mz, it in peaks]

    score, dmz = dedup.tolerance_cosine(peaks, rounded, ppm=5.0, min_mz=200.0)

    assert score == pytest.approx(1.0)
    assert 0.0 < dmz <= 0.0005


def test_peaks_shifted_beyond_tolerance_do_not_match(dedup, peaks):
    # 20 ppm off at every peak: an independent acquisition with a calibration offset.
    shifted = [(mz * (1 + 20e-6), it) for mz, it in peaks]

    score, dmz = dedup.tolerance_cosine(peaks, shifted, ppm=5.0, min_mz=200.0)

    assert score == pytest.approx(0.0)
    assert np.isnan(dmz)


def test_binned_prefilter_survives_bin_edge(dedup):
    # 100.0049 and 100.0051 sit on either side of a 0.01 Da bin edge on the first grid only.
    a, b = [(100.0049, 1.0)], [(100.0051, 1.0)]

    sim = dedup.group_cosine([a, b], bin_da=0.01)

    assert sim.shape == (2, 2)
    assert sim[0, 1] == pytest.approx(1.0)


def _task(spec_ids, peak_lists, insts=None, ces=None):
    n = len(spec_ids)
    insts = insts or ["ORB"] * n
    ces = ces or ["20.0"] * n
    npks = [len(p) for p in peak_lists]
    return (spec_ids, peak_lists, insts, ces, npks, 0.01, 0.95, 5.0, 200.0, 0.99)


def test_group_edges_keeps_only_pairs_above_threshold(dedup, peaks):
    different = [(80.0495, 1.0), (110.0600, 0.4)]

    edges = dedup._group_edges(_task([10, 11, 12], [peaks, list(peaks), different]))

    assert [(a, b, kind) for a, b, _, _, kind, _ in edges] == [(10, 11, "same")]
    assert edges[0][2] == pytest.approx(1.0)


def test_same_mz_different_intensities_is_not_a_duplicate(dedup):
    # Two collision energies of one compound in a library that reports one m/z list: identical
    # m/z, different intensities (cosine ~0.96). They are different spectra and must stay apart.
    low_ce = [(201.016, 0.186), (325.142, 0.909), (351.085, 0.299), (474.349, 1.0), (475.352, 0.577)]
    high_ce = [(201.016, 0.270), (325.142, 1.0), (351.085, 0.342), (474.349, 0.606), (475.352, 0.562)]

    assert dedup._group_edges(_task([1, 2], [low_ce, high_ce])) == []


@pytest.mark.parametrize(
    ("a", "b", "kind"),
    [
        (("ORB", "10.0"), ("ORB", "10.0"), "same"),
        (("NA", ""), ("NA", ""), "same"),
        (("ORB", "10.0"), ("QTOF", "10.0"), None),  # known instruments differ
        (("ORB", "10.0"), ("ORB", "20.0"), None),  # known CEs differ
        (("NA", "10.0"), ("QTOF", "10.0"), "wildcard"),  # missing instrument matches any
        (("ORB", ""), ("ORB", "20.0"), "wildcard"),  # missing CE matches any
        (("NA", "10.0"), ("ORB", ""), "wildcard"),
    ],
)
def test_pair_kind(dedup, a, b, kind):
    assert dedup.pair_kind(*a, *b) == kind


def test_wildcard_pair_drops_the_spectrum_with_missing_values(dedup, peaks):
    # Spectrum 2 is the CE-missing copy of spectrum 1; spectrum 3 is identical but at another
    # known CE, so it pairs with neither 1 (different CE) nor merges with it through 2.
    edges = dedup._group_edges(
        _task([1, 2, 3], [peaks, list(peaks), list(peaks)], ces=["10.0", "", "20.0"])
    )

    assert sorted((a, b, kind, loser) for a, b, _, _, kind, loser in edges) == [
        (1, 2, "wildcard", 2),
        (2, 3, "wildcard", 2),
    ]


def test_wildcard_tie_drops_the_spectrum_with_fewer_peaks(dedup, peaks):
    # One missing value each (instrument vs CE); spectrum 2 carries one extra tiny peak, so the
    # pair still scores > 0.99 and spectrum 1, with fewer peaks, loses.
    longer = peaks + [(400.12345, 0.001)]
    edges = dedup._group_edges(_task([1, 2], [peaks, longer], insts=["NA", "ORB"], ces=["10.0", ""]))

    assert [(kind, loser) for _, _, _, _, kind, loser in edges] == [("wildcard", 1)]


def test_ce_signature_reads_primary_and_extra_readings(dedup):
    spec = pd.DataFrame({
        "ace": [10.0, np.nan, 20.0],
        "ace_extra_1": [np.nan, np.nan, 40.0],
        "ace_extra_2": [np.nan, np.nan, np.nan],
    })

    assert dedup.ce_signature(spec).tolist() == ["10.0", "", "20.0|40.0"]
