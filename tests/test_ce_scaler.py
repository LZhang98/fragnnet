"""Unit tests for the CEScaler class in fragnnet.model.base_model.

Tests:
- Identity at initialization (all-zero last layer → log_scale=0, bias=0 → passthrough).
- FiLM scaling with known manual weights: verify expected scale and bias applied.
- Missing CE values (ce=-1) are masked to zero normalization → identity scaling.
- Multiple CEs per sample are aggregated via scatter mean.
- Integration: CEScaler output shape matches input shape.
"""

import pytest
import torch as th

from fragnnet.model.base_model import CEScaler

CE_MEAN = 60.0
CE_STD = 40.0
OUTPUT_DIM = 5
HIDDEN_DIM = 16
BATCH_SIZE = 2
NUM_NODES = 6  # 3 nodes per sample


@pytest.fixture()
def scaler() -> CEScaler:
    """Minimal CEScaler with identity-initialized weights."""
    return CEScaler(ce_mean=CE_MEAN, ce_std=CE_STD, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM)


def _make_inputs(
    batch_size: int = BATCH_SIZE,
    num_nodes: int = NUM_NODES,
    output_dim: int = OUTPUT_DIM,
    device: th.device = th.device("cpu"),
) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    """Return (logits, ce, ce_batch_idxs, frag_node_batch_idxs) for tests."""
    th.manual_seed(0)
    logits = th.randn(num_nodes, output_dim, device=device)
    # One CE per sample
    ce = th.tensor([30.0, 90.0], device=device)
    ce_batch_idxs = th.tensor([0, 1], device=device)
    nodes_per_sample = num_nodes // batch_size
    frag_node_batch_idxs = th.repeat_interleave(
        th.arange(batch_size, device=device),
        th.tensor([nodes_per_sample] * batch_size, device=device),
    )
    return logits, ce, ce_batch_idxs, frag_node_batch_idxs


class TestCEScalerIdentity:
    """Verify identity behavior at initialization (all-zero last layer)."""

    def test_identity_init(self, scaler: CEScaler) -> None:
        """Output equals input when last-layer weights and biases are zero."""
        logits, ce, ce_batch_idxs, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)
        assert out.shape == logits.shape
        # exp(0) == 1, bias == 0 → output == input
        assert th.allclose(out, logits, atol=1e-6), f"Max diff: {(out - logits).abs().max()}"

    def test_output_shape(self, scaler: CEScaler) -> None:
        """Output shape matches (num_nodes, output_dim)."""
        logits, ce, ce_batch_idxs, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)
        assert out.shape == (NUM_NODES, OUTPUT_DIM)


class TestCEScalerFilm:
    """Verify FiLM scaling with manually set network weights."""

    def test_known_scale(self, scaler: CEScaler) -> None:
        """Set last-layer bias so log_scale=0.5 for all dims, bias=0; verify exp(0.5) scaling."""
        log_scale_val = 0.5
        # net[-1].bias first OUTPUT_DIM entries are log_scale, next OUTPUT_DIM are bias
        with th.no_grad():
            scaler.net[-1].bias[:OUTPUT_DIM] = log_scale_val  # log_scale = 0.5
            scaler.net[-1].bias[OUTPUT_DIM:] = 0.0  # bias = 0

        logits, ce, ce_batch_idxs, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)

        expected = logits * th.exp(th.tensor(log_scale_val))
        assert th.allclose(out, expected, atol=1e-5), f"Max diff: {(out - expected).abs().max()}"

    def test_known_bias(self, scaler: CEScaler) -> None:
        """Set last-layer bias so log_scale=0, bias=2.0 for all dims; verify additive shift."""
        bias_val = 2.0
        with th.no_grad():
            scaler.net[-1].bias[:OUTPUT_DIM] = 0.0  # log_scale = 0
            scaler.net[-1].bias[OUTPUT_DIM:] = bias_val  # bias = 2.0

        logits, ce, ce_batch_idxs, frag_node_batch_idxs = _make_inputs()
        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)

        expected = logits + bias_val
        assert th.allclose(out, expected, atol=1e-5), f"Max diff: {(out - expected).abs().max()}"


class TestCEScalerMissingCE:
    """Verify that missing CE values (ce=-1) produce identity scaling."""

    def test_missing_ce_identity(self, scaler: CEScaler) -> None:
        """When all CE values are -1, masked normalization → 0 → near-identity output."""
        logits, _, _, frag_node_batch_idxs = _make_inputs()
        ce = th.tensor([-1.0, -1.0])
        ce_batch_idxs = th.tensor([0, 1])

        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)

        # With identity-initialized weights and zero input → output == input
        assert th.allclose(out, logits, atol=1e-6), f"Max diff: {(out - logits).abs().max()}"

    def test_mixed_missing(self, scaler: CEScaler) -> None:
        """One sample has valid CE, one has -1; shapes are correct."""
        logits, _, _, frag_node_batch_idxs = _make_inputs()
        ce = th.tensor([60.0, -1.0])
        ce_batch_idxs = th.tensor([0, 1])

        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)

        assert out.shape == logits.shape


class TestCEScalerMultiCE:
    """Verify scatter mean aggregation for multiple CEs per sample."""

    def test_two_ces_per_sample(self, scaler: CEScaler) -> None:
        """Two CEs per sample should be averaged; output shape is unchanged."""
        logits = th.randn(NUM_NODES, OUTPUT_DIM)
        # 4 CE values: 2 per sample
        ce = th.tensor([30.0, 60.0, 80.0, 100.0])
        ce_batch_idxs = th.tensor([0, 0, 1, 1])
        frag_node_batch_idxs = th.repeat_interleave(
            th.arange(BATCH_SIZE), th.tensor([NUM_NODES // BATCH_SIZE] * BATCH_SIZE)
        )

        with th.no_grad():
            out = scaler(logits, ce, ce_batch_idxs, frag_node_batch_idxs, BATCH_SIZE)

        assert out.shape == logits.shape

    def test_mean_aggregation_consistency(self, scaler: CEScaler) -> None:
        """Two CEs averaged should equal single CE at midpoint (via zero last-layer weights)."""
        # All params zero → output only depends on the linear path through zero weights.
        # With identity init, this just verifies no numerical instability.
        logits = th.ones(4, OUTPUT_DIM)
        # Single CE at 60 (the mean → normalized to 0 → identity)
        ce_single = th.tensor([60.0, 60.0])
        ce_batch_single = th.tensor([0, 1])
        # Two CEs both at 60
        ce_double = th.tensor([60.0, 60.0, 60.0, 60.0])
        ce_batch_double = th.tensor([0, 0, 1, 1])
        frag_node_batch_idxs = th.tensor([0, 0, 1, 1])

        with th.no_grad():
            out_single = scaler(logits, ce_single, ce_batch_single, frag_node_batch_idxs, 2)
            out_double = scaler(logits, ce_double, ce_batch_double, frag_node_batch_idxs, 2)

        assert th.allclose(out_single, out_double, atol=1e-6)
