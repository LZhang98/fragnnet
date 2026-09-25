"""`min_peak_int` must filter peaks even when there is no peak -> formula map.

`batch_filter_by_peak_intensity` used to gate the peak filter itself on
`frag_formula_peak_idxs`, which only DAG models (FraGNNet) provide. For MASSter the
threshold was a silent no-op and exports kept every candidate. The formula cascade still
needs the map; the peak axis does not.
"""

import pytest
import torch as th

from fragnnet.utils.script_utils import PredArrays, batch_filter_by_peak_intensity, get_split_fn

# Three spectra on the exporter's 0-1000 scale. Spectrum 2 has no peak above 1.0, so a
# threshold of 1.0 empties it.
PRED_BATCH_IDXS = th.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2])
PRED_INTS = th.tensor([1000.0, 0.5, 1.0, 12.0, 1000.0, 3.0, 0.2, 0.9, 1.0])
PRED_MZS = th.tensor([100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 300.0, 301.0])


def _peak_only_arrays(**formula_axis):
    return PredArrays(
        pred_batch_idxs=PRED_BATCH_IDXS,
        pred_ints=PRED_INTS,
        pred_mzs=PRED_MZS,
        **formula_axis,
    )


def _filter(arrays, min_peak_int, filter_stats=None):
    return batch_filter_by_peak_intensity(
        arrays,
        min_peak_int,
        frag_formula_cumsizes=th.tensor([0, 2, 4, 6]),
        frag_num_nodes=None,
        filter_stats=filter_stats,
    )


def test_peaks_are_filtered_without_a_formula_map():
    out = _filter(_peak_only_arrays(), min_peak_int=1.0)
    # Strictly above the threshold survives; 1.0 itself does not.
    assert th.equal(out.pred_ints, th.tensor([1000.0, 12.0, 1000.0, 3.0]))
    assert th.equal(out.pred_mzs, th.tensor([100.0, 103.0, 200.0, 201.0]))
    assert th.equal(out.pred_batch_idxs, th.tensor([0, 0, 1, 1]))
    assert out.frag_formula_peak_idxs is None


def test_zero_threshold_keeps_every_peak():
    out = _filter(_peak_only_arrays(), min_peak_int=0.0)
    assert th.equal(out.pred_ints, PRED_INTS)
    assert th.equal(out.pred_mzs, PRED_MZS)
    assert th.equal(out.pred_batch_idxs, PRED_BATCH_IDXS)


def test_filter_stats_count_peak_only_filtering():
    stats = {}
    _filter(_peak_only_arrays(), min_peak_int=1.0, filter_stats=stats)
    assert stats == {
        "n_peaks_before": 9,
        "n_peaks_after": 4,
        "n_inst_filtered": 3,
        "n_inst_total": 3,
    }


def test_emptied_spectrum_keeps_its_slot_in_the_split():
    """A spectrum with no surviving peak must still get a (zero-length) row downstream."""
    out = _filter(_peak_only_arrays(), min_peak_int=1.0)
    split = get_split_fn(out.pred_batch_idxs, batch_size=3)(out.pred_ints)
    assert [len(s) for s in split] == [2, 2, 0]


def test_formula_axis_without_a_map_passes_through_and_is_rebased():
    """No peak -> formula map means no cascade: every formula row survives, rebased."""
    formula_batch_idxs = th.tensor([0, 0, 1, 1, 2, 2])
    formula_formula_idxs = th.arange(6)  # batch-global
    formula_probs = th.tensor([0.6, 0.4, 0.7, 0.3, 0.5, 0.5])
    out = _filter(
        _peak_only_arrays(
            pred_formula_batch_idxs=formula_batch_idxs,
            pred_formula_formula_idxs=formula_formula_idxs,
            pred_formula_probs=formula_probs,
        ),
        min_peak_int=1.0,
    )
    assert th.equal(out.pred_ints, th.tensor([1000.0, 12.0, 1000.0, 3.0]))
    assert th.equal(out.pred_formula_batch_idxs, formula_batch_idxs)
    assert th.equal(out.pred_formula_formula_idxs, th.tensor([0, 1, 0, 1, 0, 1]))
    assert th.allclose(out.pred_formula_probs, formula_probs)


@pytest.mark.parametrize("min_peak_int", [0.0, 1.0])
def test_formula_map_path_filters_peaks_identically(min_peak_int):
    """With a map, the peak axis must come out exactly as in the map-free path."""
    peak_map = th.tensor([0, 1, 1, 0, 0, 1, 1, 0, 1])  # local formula per peak
    with_map = _filter(
        _peak_only_arrays(
            frag_formula_peak_idxs=peak_map,
            pred_formula_batch_idxs=th.tensor([0, 0, 1, 1, 2, 2]),
            pred_formula_formula_idxs=th.arange(6),
        ),
        min_peak_int=min_peak_int,
    )
    without_map = _filter(_peak_only_arrays(), min_peak_int=min_peak_int)
    assert th.equal(with_map.pred_ints, without_map.pred_ints)
    assert th.equal(with_map.pred_mzs, without_map.pred_mzs)
    assert th.equal(with_map.pred_batch_idxs, without_map.pred_batch_idxs)
    assert th.equal(with_map.frag_formula_peak_idxs, peak_map[min_peak_int < PRED_INTS])
