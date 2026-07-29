import torch as th

from fragnnet.model.fragnnet_model import FraGNNetModel


def _make_model_stub(
    *,
    p: float,
    channel_p: float = 0.0,
    frag_node_feats: list[str] | None = None,
    training: bool = True,
) -> FraGNNetModel:
    model = FraGNNetModel.__new__(FraGNNetModel)
    th.nn.Module.__init__(model)
    model.cc_feature_dropout = p
    model.cc_channel_dropout = channel_p
    model.frag_node_feats = frag_node_feats or ["cc"]
    if training:
        model.train()
    else:
        model.eval()
    return model


def test_cc_feature_dropout_zero_never_changes_tensor():
    model = _make_model_stub(p=0.0, training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_feature_dropout_one_zeros_cc_group_in_training():
    model = _make_model_stub(p=1.0, training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is not x
    assert th.equal(out, th.zeros_like(x))


def test_cc_feature_dropout_eval_mode_does_not_drop():
    model = _make_model_stub(p=1.0, training=False)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_feature_dropout_noop_when_cc_feature_absent():
    model = _make_model_stub(p=1.0, frag_node_feats=["boundary_cc"], training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_feature_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_channel_dropout_zero_never_changes_tensor():
    model = _make_model_stub(p=0.0, channel_p=0.0, training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_channel_dropout(x)

    assert out is x
    assert th.equal(out, x)


def test_cc_channel_dropout_shares_mask_and_uses_inverted_scaling():
    model = _make_model_stub(p=0.0, channel_p=0.5, training=True)
    x = th.ones(4, 128)
    th.manual_seed(0)

    out = model._maybe_apply_cc_channel_dropout(x)

    assert th.equal(out[0], out[1])
    assert th.equal(out[0], out[2])
    assert th.equal(out[0], out[3])
    assert th.all((out == 0.0) | (out == 2.0))
    assert th.any(out == 0.0)
    assert th.any(out == 2.0)


def test_cc_channel_dropout_one_zeros_all_channels():
    model = _make_model_stub(p=0.0, channel_p=1.0, training=True)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_channel_dropout(x)

    assert th.equal(out, th.zeros_like(x))


def test_cc_channel_dropout_eval_mode_does_not_drop():
    model = _make_model_stub(p=0.0, channel_p=1.0, training=False)
    x = th.randn(4, 3)

    out = model._maybe_apply_cc_channel_dropout(x)

    assert out is x
    assert th.equal(out, x)
