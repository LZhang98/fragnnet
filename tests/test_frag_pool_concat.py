"""Tests for frag_pool_combine='concat': DeepSets-style set-context injection into the readout.

The 'subtract'/'add' combine modes fold the pooled fragment-set vector into each node's embedding
with a fixed elementwise shift, so the per-node readout logit is still an (order-preserving)
function of the node's own features plus a single shared offset. 'concat' instead appends the
pooled set-context to each node's readout input, letting the readout MLP learn an arbitrary
node x set interaction -- a strictly more expressive coupling that also unlocks multi-stat pools
(which expand feature width and are therefore rejected by subtract/add).

Invariants under test:
- concat widens the readout input by exactly the pool output width (1x hidden, or 3x for
  multi-stat pools), and unlike subtract it is compatible with width-expanding pools;
- the appended set-context actually feeds the logits (zeroing its input columns changes output);
- a known-molecule forward is finite and normalized;
- concat is only valid with mlp_output_format='formula';
- the subtract/add width-preservation guard is unchanged.
"""

from __future__ import annotations

import pytest
import torch as th

from tests.test_cc_atom_dropout import _forward_kwargs
from tests.test_ce_multi_location import _build

# _build uses frag_hidden_size=32 and frag_embed_combine='cat' (=> base readout input = 2*hidden).
HIDDEN = 32


class TestConcatWiring:
    def test_concat_widens_readout_by_pool_width(self):
        """max pool preserves width, so concat adds exactly one hidden-size block."""
        base = _build(frag_pool_combine="none")
        concat = _build(frag_pool_combine="concat")
        assert concat.frag_pool_out_size == HIDDEN
        assert (
            concat.formula_module.input_layer.in_features
            - base.formula_module.input_layer.in_features
            == HIDDEN
        )

    def test_concat_multistat_pool_triples_width(self):
        """Multi-stat pools expand width 3x; concat accepts them (subtract/add do not)."""
        concat = _build(frag_pool_combine="concat", frag_pool_type="mean_max_sum")
        base = _build(frag_pool_combine="none", frag_pool_type="mean_max_sum")
        assert concat.frag_pool_out_size == 3 * HIDDEN
        assert (
            concat.formula_module.input_layer.in_features
            - base.formula_module.input_layer.in_features
            == 3 * HIDDEN
        )


class TestConcatValidation:
    def test_concat_rejects_node_formula(self):
        with pytest.raises(ValueError, match="frag_pool_combine='concat' is only supported"):
            _build(frag_pool_combine="concat", mlp_output_format="node_formula")

    def test_subtract_still_forbids_expanding_pool(self):
        """Regression: the width-preservation guard for subtract/add is untouched."""
        with pytest.raises(ValueError, match="frag_pool_combine must be 'none'"):
            _build(frag_pool_combine="subtract", frag_pool_type="mean_max_sum")


class TestConcatBehavior:
    def test_concat_forward_finite_and_valid(self):
        """Known molecule: concat forward is finite and yields valid (subset) probabilities.

        pred_logprobs is indexed at the observed peak positions only -- a subset of the full
        per-spectrum distribution -- so its mass is in (0, 1], not exactly 1.
        """
        model = _build(frag_pool_combine="concat")
        model.eval()
        with th.no_grad():
            out = model(**_forward_kwargs())
        probs = out["pred_logprobs"].exp()
        assert th.isfinite(out["pred_logprobs"]).all()
        assert (probs >= 0).all() and (probs <= 1).all()
        assert 0.0 < probs.sum().item() <= 1.0 + 1e-4

    def test_pool_context_feeds_readout(self):
        """The appended set-context is live: zeroing its input columns changes the prediction.

        With no mlp-injected conditioning (the _build default), frag_embed = cat([gnn, node, pool]),
        so the pooled set-context occupies the final frag_pool_out_size input columns.
        """
        model = _build(frag_pool_combine="concat")
        model.eval()
        kwargs = _forward_kwargs()
        # Lock the assumed column layout: base (2*hidden) + pooled set-context.
        assert model.formula_module.input_layer.in_features == 2 * HIDDEN + model.frag_pool_out_size
        with th.no_grad():
            out_full = model(**kwargs)["pred_logprobs"]
            model.formula_module.input_layer.weight[:, -model.frag_pool_out_size :] = 0.0
            out_no_context = model(**kwargs)["pred_logprobs"]
        assert not th.allclose(out_full, out_no_context, atol=1e-5)
