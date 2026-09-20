"""Configured per-spectrum metrics must work in labeled inference and testing."""

from types import SimpleNamespace

import pytest
import torch as th

from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.runner import load_config


@pytest.mark.parametrize("score", ["ndcg", "true_oos_prob", "true_oos_e"])
@pytest.mark.parametrize("split", ["test", "predict"])
def test_indexed_metrics_match_validation(score, split):
    config = load_config("config", None, None)
    config.update(auxiliary_scores=[score], sparse_cosine_similarity=True)
    pl = SimpleNamespace(
        hparams=SimpleNamespace(**config), metric_names=set(),
        tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
    )
    SpectrumPL._setup_metric_fns(pl)
    inputs = {
        "true_mzs": th.tensor([100.0, 120.0, 100.0, 120.0]),
        "true_ints": th.tensor([0.8, 0.2, 0.3, 0.7]),
        "true_batch_idxs": th.tensor([0, 0, 1, 1]),
        "pred_mzs": th.tensor([100.0, 140.0, 100.0, 120.0]),
        "pred_ints": th.tensor([0.6, 0.4, 0.3, 0.7]),
        "pred_batch_idxs": th.tensor([0, 0, 1, 1]),
        "true_prec_mzs": th.tensor([200.0, 200.0]),
    }
    expected = pl.metric_fn(**inputs, split="val")
    actual = pl.metric_fn(**inputs, split=split)
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        assert value.shape == (2,)
        assert th.isfinite(value).all()
        th.testing.assert_close(value, expected[key])
    if score == "true_oos_prob":
        th.testing.assert_close(actual[score], th.tensor([0.2, 0.0]))


def _binned_metric_pl(**config_overrides):
    """Minimal SpectrumPL stand-in wired for the binned auxiliary metrics."""
    config = load_config("config", None, None)
    config.update(
        auxiliary_scores=["cos_sim"],
        eval_mz_bin_res=[0.01],
        sparse_cosine_similarity=True,
        **config_overrides,
    )
    pl = SimpleNamespace(
        hparams=SimpleNamespace(**config),
        metric_names=set(),
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )
    SpectrumPL._setup_metric_fns(pl)
    return pl


def test_binned_metrics_drop_out_of_range_pred_peaks():
    """A predicted peak at or above mz_max is dropped, not fatal.

    Predicted m/z are never range-filtered upstream (``preproc_spec``'s eval/pred branch only
    touches intensities), so a candidate above ``mz_max`` used to trip ``batched_bin_func``'s
    range assert and kill the run. Regression for that crash: the surviving peaks must score
    exactly as they would have had the out-of-range peak never been predicted.
    """
    pl = _binned_metric_pl(mz_max=1500.0)
    shared = {
        "true_mzs": th.tensor([100.0, 120.0, 100.0, 120.0]),
        "true_ints": th.tensor([0.8, 0.2, 0.3, 0.7]),
        "true_batch_idxs": th.tensor([0, 0, 1, 1]),
        "true_prec_mzs": th.tensor([1501.7432, 200.0]),
        "split": "train",
    }
    with_out_of_range = pl.metric_fn(
        pred_mzs=th.tensor([100.0, 1501.7432, 100.0, 120.0]),
        pred_ints=th.tensor([0.6, 0.4, 0.3, 0.7]),
        pred_batch_idxs=th.tensor([0, 0, 1, 1]),
        **shared,
    )
    without = pl.metric_fn(
        pred_mzs=th.tensor([100.0, 100.0, 120.0]),
        pred_ints=th.tensor([0.6, 0.3, 0.7]),
        pred_batch_idxs=th.tensor([0, 1, 1]),
        **shared,
    )
    assert th.isfinite(with_out_of_range["cos_sim_0.01"]).all()
    th.testing.assert_close(with_out_of_range["cos_sim_0.01"], without["cos_sim_0.01"])


def test_binned_metrics_still_reject_out_of_range_true_peaks():
    """The range guard stays live for target peaks, which ARE filtered upstream.

    Only the predicted side is unfiltered by construction; an out-of-range true peak means the
    target pipeline is broken and must keep failing loudly.
    """
    pl = _binned_metric_pl(mz_max=1500.0)
    with pytest.raises(AssertionError):
        pl.metric_fn(
            true_mzs=th.tensor([100.0, 1501.7432]),
            true_ints=th.tensor([0.8, 0.2]),
            true_batch_idxs=th.tensor([0, 0]),
            pred_mzs=th.tensor([100.0, 120.0]),
            pred_ints=th.tensor([0.6, 0.4]),
            pred_batch_idxs=th.tensor([0, 0]),
            true_prec_mzs=th.tensor([200.0]),
            split="train",
        )
