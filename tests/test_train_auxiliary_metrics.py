"""Tests for the ``compute_train_auxiliary_metrics`` gate on train-split metrics.

Auxiliary scores (``cos_sim``, ``opt_cos_sim``, ...) are stripped from the train split by
default, because materializing the eval tensors on every train batch is expensive. The
``compute_train_auxiliary_metrics`` flag opts back in, which is what the overfit/generalize
diagnostic needs: it compares ``train_cos_sim`` against the per-subset oracle
``train_opt_cos_sim``, and both are auxiliary metrics.
"""

from types import SimpleNamespace

import pytest

from fragnnet.pl_model.spectrum_pl import SpectrumPL


@pytest.fixture
def metric_stub():
    """Minimal stub exposing the attributes ``_get_active_metric_names`` reads."""
    return SimpleNamespace(
        metric_names={"loss", "cos_sim", "opt_cos_sim"},
        auxiliary_metric_names={"cos_sim", "opt_cos_sim"},
        hparams=SimpleNamespace(compute_train_auxiliary_metrics=False),
    )


def test_train_strips_auxiliary_metrics_by_default(metric_stub):
    """Default (False) keeps the pre-existing behavior: no aux metrics on train."""
    metric_stub.hparams.compute_train_auxiliary_metrics = False
    assert SpectrumPL._get_active_metric_names(metric_stub, "train") == {"loss"}


def test_train_keeps_auxiliary_metrics_when_enabled(metric_stub):
    """Enabling the flag is what makes train_cos_sim / train_opt_cos_sim get logged."""
    metric_stub.hparams.compute_train_auxiliary_metrics = True
    assert SpectrumPL._get_active_metric_names(metric_stub, "train") == {
        "loss",
        "cos_sim",
        "opt_cos_sim",
    }


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("split", ["val", "test"])
def test_non_train_splits_unaffected_by_flag(metric_stub, flag, split):
    """The flag must only ever affect the train split."""
    metric_stub.hparams.compute_train_auxiliary_metrics = flag
    assert SpectrumPL._get_active_metric_names(metric_stub, split) == {
        "loss",
        "cos_sim",
        "opt_cos_sim",
    }


def test_no_auxiliary_metrics_configured_is_a_noop(metric_stub):
    """With no aux metrics registered, the flag changes nothing on either split."""
    metric_stub.metric_names = {"loss"}
    metric_stub.auxiliary_metric_names = set()
    for flag in (False, True):
        metric_stub.hparams.compute_train_auxiliary_metrics = flag
        assert SpectrumPL._get_active_metric_names(metric_stub, "train") == {"loss"}
        assert SpectrumPL._get_active_metric_names(metric_stub, "val") == {"loss"}
