"""Unit tests for the CE × depth prior (§3c, depth_ce_bias).

Tests cover:
- Identity at init: zero-init layer produces zero bias → logits unchanged.
- Output shape: broadcast from (n_nodes,) to (n_nodes, output_dim) is correct.
- Known weights: set bias for a specific depth and verify the logit shift.
- CE differentiation: different CE inputs → different per-node biases at the same depth.
- Depth differentiation: same CE but different depth indices → different biases.
- ce_embed_dim: CEModel sets this correctly for mol, mlp, and dual-CE configurations.
"""

import torch
import torch.nn as nn

from fragnnet.layers.metadata_embedding import CEModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CE_DIM = 32  # single NCE embed size
NUM_DEPTH = 3  # depth 0..3 → 4 buckets
OUTPUT_DIM = 9  # 2*num_hs+1 (num_hs=4)
BATCH_SIZE = 3
NUM_NODES = 9  # 3 nodes per sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_layer(ce_dim: int = CE_DIM, num_depth: int = NUM_DEPTH) -> nn.Linear:
    """Return a zero-initialised depth_ce_bias_layer."""
    layer = nn.Linear(ce_dim, num_depth + 1)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _apply_depth_ce_bias(
    logits: torch.Tensor,  # (n_nodes, output_dim)
    layer: nn.Linear,
    ce_embed: torch.Tensor,  # (batch, ce_dim)
    depth_idx: torch.Tensor,  # (n_nodes,) int in [0, num_depth]
    frag_node_batch_idxs: torch.Tensor,  # (n_nodes,) int
) -> torch.Tensor:
    """Mirror the forward-pass application logic from fragnnet_model.py."""
    bias_per_depth = layer(ce_embed)  # (batch, num_depth+1)
    per_node_bias = bias_per_depth[frag_node_batch_idxs, depth_idx]  # (n_nodes,)
    return logits + per_node_bias.unsqueeze(-1)


# ---------------------------------------------------------------------------
# Identity at initialisation
# ---------------------------------------------------------------------------


class TestDepthCEBiasIdentityAtInit:
    def test_logits_unchanged(self) -> None:
        """Zero-init layer → all biases are zero → logits must be exactly unchanged."""
        layer = _make_layer()
        torch.manual_seed(0)
        logits = torch.randn(NUM_NODES, OUTPUT_DIM)
        ce_embed = torch.randn(BATCH_SIZE, CE_DIM)
        depth_idx = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2])
        node_batch_idxs = torch.repeat_interleave(torch.arange(BATCH_SIZE), NUM_NODES // BATCH_SIZE)

        with torch.no_grad():
            out = _apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs)

        assert torch.allclose(out, logits, atol=1e-6), f"Max diff: {(out - logits).abs().max()}"

    def test_layer_weights_are_zero(self) -> None:
        """Verify that zero-init creates exactly zero weight and bias tensors."""
        layer = _make_layer()
        assert torch.all(layer.weight == 0.0)
        assert torch.all(layer.bias == 0.0)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestDepthCEBiasShape:
    def test_output_shape_matches_logits(self) -> None:
        """Result has the same shape as the input logits."""
        layer = _make_layer()
        logits = torch.randn(NUM_NODES, OUTPUT_DIM)
        ce_embed = torch.randn(BATCH_SIZE, CE_DIM)
        depth_idx = torch.zeros(NUM_NODES, dtype=torch.long)
        node_batch_idxs = torch.repeat_interleave(torch.arange(BATCH_SIZE), NUM_NODES // BATCH_SIZE)

        out = _apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs)
        assert out.shape == logits.shape

    def test_bias_per_depth_shape(self) -> None:
        """Layer output is (batch, num_depth+1)."""
        layer = _make_layer(ce_dim=CE_DIM, num_depth=NUM_DEPTH)
        ce_embed = torch.randn(BATCH_SIZE, CE_DIM)
        bias = layer(ce_embed)
        assert bias.shape == (BATCH_SIZE, NUM_DEPTH + 1)


# ---------------------------------------------------------------------------
# Known-weight tests
# ---------------------------------------------------------------------------


class TestDepthCEBiasKnownWeights:
    def test_known_depth1_bias(self) -> None:
        """Set bias so that depth-1 nodes get +0.5 additively; verify shift."""
        layer = _make_layer()
        with torch.no_grad():
            # bias[1] = 0.5 for all batches → depth-1 nodes get +0.5 across all output cols
            layer.bias[1] = 0.5

        logits = torch.ones(6, OUTPUT_DIM)
        # 2 samples, 3 nodes each; depth layout: [0, 1, 2, 0, 1, 2]
        depth_idx = torch.tensor([0, 1, 2, 0, 1, 2])
        node_batch_idxs = torch.tensor([0, 0, 0, 1, 1, 1])
        ce_embed = torch.zeros(2, CE_DIM)  # zero CE → only bias matters

        with torch.no_grad():
            out = _apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs)

        # Depth-1 nodes (indices 1 and 4) should be shifted by +0.5
        expected = logits.clone()
        expected[[1, 4]] += 0.5
        assert torch.allclose(out, expected, atol=1e-6), f"Max diff: {(out - expected).abs().max()}"

    def test_negative_bias_suppresses_logits(self) -> None:
        """Negative bias at depth-3 suppresses those nodes' logits."""
        layer = _make_layer()
        with torch.no_grad():
            layer.bias[3] = -2.0

        logits = torch.zeros(4, OUTPUT_DIM)
        depth_idx = torch.tensor([0, 1, 2, 3])
        node_batch_idxs = torch.zeros(4, dtype=torch.long)
        ce_embed = torch.zeros(1, CE_DIM)

        with torch.no_grad():
            out = _apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs)

        # Only depth-3 node (index 3) should be shifted
        assert torch.allclose(out[:3], torch.zeros(3, OUTPUT_DIM), atol=1e-6)
        assert torch.allclose(out[3], torch.full((OUTPUT_DIM,), -2.0), atol=1e-6)


# ---------------------------------------------------------------------------
# CE and depth differentiation
# ---------------------------------------------------------------------------


class TestDepthCEBiasDifferentiation:
    def test_different_ce_same_depth_gives_different_bias(self) -> None:
        """After one SGD step, different CE values produce different per-node biases."""
        layer = _make_layer(ce_dim=4, num_depth=2)

        # Create two samples at depth=1 with different CE embeddings
        ce_embed = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        depth_idx = torch.tensor([1, 1])  # both nodes at depth 1
        node_batch_idxs = torch.tensor([0, 1])
        logits = torch.ones(2, 5)
        target = torch.tensor([[2.0] * 5, [0.5] * 5])

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.5)
        loss = (
            (_apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs) - target)
            .pow(2)
            .mean()
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            out = _apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs)

        # After training the two CE inputs must produce different outputs
        assert not torch.allclose(
            out[0], out[1], atol=1e-4
        ), "Samples with different CE embeddings should produce different biases after training"

    def test_different_depth_same_ce_gives_different_bias(self) -> None:
        """After one SGD step, nodes at different depths get different biases."""
        layer = _make_layer(ce_dim=4, num_depth=3)

        ce_embed = torch.zeros(1, 4)  # single sample
        depth_idx = torch.tensor([0, 3])  # depth 0 vs depth 3
        node_batch_idxs = torch.zeros(2, dtype=torch.long)
        logits = torch.ones(2, 5)
        target = torch.tensor([[3.0] * 5, [-1.0] * 5])

        optimizer = torch.optim.SGD(layer.parameters(), lr=0.5)
        loss = (
            (_apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs) - target)
            .pow(2)
            .mean()
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            out = _apply_depth_ce_bias(logits, layer, ce_embed, depth_idx, node_batch_idxs)

        assert not torch.allclose(
            out[0], out[1], atol=1e-4
        ), "Nodes at different depths should produce different biases after training"


# ---------------------------------------------------------------------------
# CEModel.ce_embed_dim correctness
# ---------------------------------------------------------------------------


class _ConcreteCEModel(nn.Module, CEModel):
    """Minimal concrete CEModel subclass for testing _setup_ce."""

    def _ce_location_check(self) -> None:
        assert self.ce_insert_location in {"none", "mol", "mlp"}


def _make_ce_model(
    location: str = "mlp",
    use_nce: bool = True,
    use_ace: bool = False,
    insert_size: int = CE_DIM,
) -> _ConcreteCEModel:
    m = _ConcreteCEModel()
    m._ce_init(
        int_embedder="abs-sines",
        ce_insert_location=location,
        ce_insert_type="embed_clamped",
        ce_insert_merge=True,
        ce_insert_size=insert_size,
        nce_max=200.0,
        nce_mean=60.0,
        nce_std=40.0,
        use_nce=use_nce,
        use_ace=use_ace,
        ace_max=100.0,
        ace_mean=25.0,
        ace_std=15.0,
    )
    return m


class TestCEEmbedDim:
    def test_nce_only_mlp(self) -> None:
        m = _make_ce_model(location="mlp", use_nce=True, use_ace=False, insert_size=32)
        assert m.ce_embed_dim == 32
        assert m.ce_mlp_input_dim == 32
        assert m.ce_mol_input_dim == 0

    def test_nce_only_mol(self) -> None:
        m = _make_ce_model(location="mol", use_nce=True, use_ace=False, insert_size=32)
        assert m.ce_embed_dim == 32
        assert m.ce_mol_input_dim == 32
        assert m.ce_mlp_input_dim == 0

    def test_dual_ce_mlp(self) -> None:
        """Dual CE: NCE(32) + ACE(32) → ce_embed_dim = 64."""
        m = _make_ce_model(location="mlp", use_nce=True, use_ace=True, insert_size=32)
        assert m.ce_embed_dim == 64
        assert m.ce_mlp_input_dim == 64
        assert m.ce_mol_input_dim == 0

    def test_no_ce(self) -> None:
        """No CE active → ce_embed_dim = 0."""
        m = _make_ce_model(location="none", use_nce=False, use_ace=False)
        assert m.ce_embed_dim == 0
        assert m.ce_mlp_input_dim == 0
        assert m.ce_mol_input_dim == 0

    def test_ace_only(self) -> None:
        m = _make_ce_model(location="mlp", use_nce=False, use_ace=True, insert_size=16)
        assert m.ce_embed_dim == 16
