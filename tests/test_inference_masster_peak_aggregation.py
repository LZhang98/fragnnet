"""MASSter exports one row per peak, not one per candidate fragment.

MASSter's `inference_step` returns a row for every candidate, and the connected fragments
that share a formula sit at one m/z. Every score sums them before matching
(`round_aggregate_peaks`, 4 decimals), so the exporter must do the same: otherwise peak counts
are candidate counts, `min_peak_int` filters candidates (dropping a peak whose candidates are
each below the threshold but sum above it), and the base-peak rescale is taken over candidates.
"""

import numpy as np
import pytest
import torch as th

from fragnnet.pl_model import MassterPL
from fragnnet.utils import script_utils
from fragnnet.utils.spec_utils import aggregate_peaks_exact_mz, round_aggregate_peaks


class _MassterStub(MassterPL):
    """MassterPL-typed stub replaying one canned `inference_step` output."""

    def __init__(self, batch_output: dict):
        self.training = False
        self._batch_output = batch_output
        self._hparams = {
            "auxiliary_scores": [],
            "eval_mz_bin_res": [],
            "eval_hun_sqrt": False,
            "eval_bin_sqrt": False,
            "sum_ints": True,
        }

    @property
    def hparams(self):
        return _AttrDict(self._hparams)

    def eval(self):
        return self

    def inference_step(self, batch_input, *, metrics=None):  # noqa: ARG002
        return self._batch_output


class _AttrDict(dict):
    def __getattr__(self, key):
        return self[key]


def _export(pred_mzs, pred_ints, pred_batch_idxs, min_peak_int=0.0):
    """Run the tabular exporter on one canned batch; return per-spectrum (mzs, ints)."""
    model = _MassterStub(
        {"pred_mzs": pred_mzs, "pred_ints": pred_ints, "pred_batch_idxs": pred_batch_idxs}
    )
    batch_size = int(pred_batch_idxs.max()) + 1
    batch_input = {
        "spec_unique_id": th.arange(batch_size),
        "mol_id": [f"m{i}" for i in range(batch_size)],
        "batch_size": th.tensor(batch_size),
    }
    output_set = {"pred_mzs", "pred_ints"}
    batch_idx_vars, node_idx_vars, formula_idx_vars, _ = script_utils.get_idx_vars(
        output_set, model, nb_iso=False
    )
    columns = script_utils.get_tabular_columns(
        model.hparams, return_formula_str=False, return_cc_hash=False, output_subset=output_set
    )
    rows = script_utils.process_batch_to_tabular_row(
        batch_input=batch_input,
        model=model,
        device=th.device("cpu"),
        evaluate=False,
        eval_score_types=[],
        BATCH_IDX_VARS=batch_idx_vars,
        NODE_IDX_VARS=node_idx_vars,
        FORMULA_IDX_VARS=formula_idx_vars,
        min_peak_int=min_peak_int,
        rebase_indices=True,
        return_formula_str=False,
        return_cc_hash=False,
        output_set=output_set,
    )
    table = dict(zip(columns, rows))
    return [
        (np.asarray(m, dtype=np.float64), np.asarray(i, dtype=np.float64))
        for m, i in zip(table["pred_mzs"], table["pred_ints"])
    ]


@pytest.fixture
def candidates():
    """Spectrum 0: three candidates at m/z 150 (0.1 each) beside 100 (0.3) and 200 (0.4).
    Spectrum 1: a dominant 300 and two candidates at 310 of 0.05 each."""
    return (
        th.tensor([100.0, 150.0, 150.0, 150.0, 200.0, 300.0, 310.0, 310.0], dtype=th.float64),
        th.tensor([0.3, 0.1, 0.1, 0.1, 0.4, 0.9, 0.05, 0.05]),
        th.tensor([0, 0, 0, 0, 0, 1, 1, 1]),
    )


def test_candidates_at_one_mz_become_one_peak(candidates):
    (mzs0, ints0), (mzs1, ints1) = _export(*candidates)
    assert mzs0.tolist() == [100.0, 150.0, 200.0]
    # Summed before the base-peak rescale: 0.3 / 0.3 / 0.4 of base 0.4.
    np.testing.assert_allclose(ints0, [750.0, 750.0, 1000.0], rtol=1e-5)
    assert mzs1.tolist() == [300.0, 310.0]
    np.testing.assert_allclose(ints1, [1000.0, 1000.0 / 9.0], rtol=1e-5)


def test_min_peak_int_filters_summed_peaks(candidates):
    # Each m/z 310 candidate is 55.6 on the 0-1000 scale, their peak 111.1: a 100 cut keeps it.
    (_, ints0), (mzs1, ints1) = _export(*candidates, min_peak_int=100.0)
    assert mzs1.tolist() == [300.0, 310.0]
    np.testing.assert_allclose(ints1, [1000.0, 1000.0 / 9.0], rtol=1e-5)
    # A 200 cut drops it, as a peak.
    (_, _), (mzs1, _) = _export(*candidates, min_peak_int=200.0)
    assert mzs1.tolist() == [300.0]
    assert len(ints0) == 3


def test_mzs_equal_to_four_decimals_merge_like_the_scores():
    # One peak at the 4-decimal key; its m/z is the intensity-weighted mean, not the key.
    pred_mzs = th.tensor([150.00001, 150.0, 250.0], dtype=th.float64)
    pred_ints = th.tensor([0.25, 0.25, 0.5])
    ((mzs, ints),) = _export(pred_mzs, pred_ints, th.tensor([0, 0, 0]))
    np.testing.assert_allclose(mzs, [150.000005, 250.0], rtol=0, atol=1e-9)
    np.testing.assert_allclose(ints, [1000.0, 1000.0], rtol=1e-5)


def test_exact_mz_is_kept_so_rebinning_matches():
    # 150.00996 rounds to the 4-decimal key 150.0100, the next 0.01 Da bin. The export keeps
    # the exact value, so a rebinned cosine sees the bin the model's cos_sim_0.01 did.
    pred_mzs = th.tensor([150.00996, 150.00996, 300.0], dtype=th.float64)
    pred_ints = th.tensor([0.25, 0.25, 0.5])
    ((mzs, ints),) = _export(pred_mzs, pred_ints, th.tensor([0, 0, 0]))
    assert mzs.tolist() == [150.00996, 300.0]
    np.testing.assert_allclose(ints, [1000.0, 1000.0], rtol=1e-5)


def test_aggregate_peaks_exact_mz_matches_round_aggregate_peaks():
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 500, 40).round(5)
    mzs = th.as_tensor(np.concatenate([base, base[:15], base[:5] + 2e-6]))
    ints = th.as_tensor(rng.uniform(0, 1, len(mzs)))
    batch = th.as_tensor(rng.integers(0, 3, len(mzs)))
    for agg in ("sum", "amax"):
        e_mzs, e_ints, e_batch = aggregate_peaks_exact_mz(mzs, ints, batch, agg=agg)
        r_mzs, r_ints, r_batch = round_aggregate_peaks(mzs, ints, batch, agg=agg)
        assert th.equal(e_batch, r_batch)
        assert th.allclose(e_ints, r_ints)
        assert th.equal(th.round(e_mzs, decimals=4), r_mzs)


def test_aggregate_peaks_exact_mz_zero_weight_group_uses_first_member():
    mzs = th.tensor([100.00001, 100.00002], dtype=th.float64)
    out_mzs, out_ints, _ = aggregate_peaks_exact_mz(mzs, th.zeros(2, dtype=th.float64),
                                                    th.zeros(2, dtype=th.long))
    assert out_mzs.tolist() == [100.00001]
    assert out_ints.tolist() == [0.0]
