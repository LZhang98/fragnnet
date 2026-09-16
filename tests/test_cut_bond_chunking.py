"""Tests for the chunked cut-bond encode/pool path shared by boundary_pair and cut_bond.

Nothing bounds how many (fragment, cut bond) pairs a batch holds: batches are packed to a
frag-node budget, and cut bonds are not frag nodes. Encoding them in one shot materializes
several (num_pairs, mol_hidden_size) tensors at once, which OOMs at inference on real DAGs.
`FraGNNetModel._encode_pool_cut_bonds` accumulates into the pooled result in row chunks
instead. That is only a memory optimization, so the contract under test is that it changes
nothing: same numbers as the single-shot path, and the pool types that do NOT decompose over
chunks keep going through `mol_pool` untouched.
"""

from __future__ import annotations

import pytest
import torch as th
from test_ce_multi_location import _build as _build_fragnnet
from test_cut_bond import _forward_kwargs_with_cut_bonds

from fragnnet.model import fragnnet_model

BASE_FEATS = ["cc", "base_formula", "depth", "h_formulae_idx"]
NUM_ATOMS = 40
NUM_FRAGS = 7
NUM_PAIRS = 64


def _model(feat: str, mol_pool_type: str = "mean"):
    return _build_fragnnet(
        frag_node_feats=[*BASE_FEATS, feat],
        frag_edge_feats=[],
        mol_pool_type=mol_pool_type,
        cut_bond_hidden_dim=32,
    ).eval()


def _encoder(model, feat: str):
    if feat == "cut_bond":
        return model.cut_bond_encoder
    return lambda a, b: model.boundary_pair_proj(th.cat([a, b], dim=-1))


def _pair_inputs(model):
    """Synthetic atom embeddings and cut-bond indices, with every fragment row hit at least once."""
    g = th.Generator().manual_seed(0)
    atom_embed = th.randn(NUM_ATOMS, model.mol_embedder.hidden_size, generator=g)
    # Cover every fragment row so the mean-pool divisor is exercised, then fill at random.
    frag_idxs = th.cat(
        [
            th.arange(NUM_FRAGS),
            th.randint(0, NUM_FRAGS, (NUM_PAIRS - NUM_FRAGS,), generator=g),
        ]
    )
    in_idxs = th.randint(0, NUM_ATOMS, (NUM_PAIRS,), generator=g)
    out_idxs = th.randint(0, NUM_ATOMS, (NUM_PAIRS,), generator=g)
    return atom_embed, frag_idxs, in_idxs, out_idxs


@pytest.mark.parametrize("feat", ["boundary_pair", "cut_bond"])
@pytest.mark.parametrize("mol_pool_type", ["mean", "sum"])
def test_chunked_matches_single_shot(monkeypatch, feat, mol_pool_type):
    """The chunked accumulation is exact, not an approximation, for the decomposing pools."""
    model = _model(feat, mol_pool_type)
    encode = _encoder(model, feat)
    atom_embed, frag_idxs, in_idxs, out_idxs = _pair_inputs(model)
    args = (encode, atom_embed, frag_idxs, in_idxs, out_idxs, NUM_FRAGS)

    with th.no_grad():
        single = model._encode_pool_cut_bonds(*args)
        # One gathered row per chunk: the most fragmented schedule the helper can take.
        monkeypatch.setattr(fragnnet_model, "CUT_BOND_CHUNK_ELEMS", model.mol_embedder.hidden_size)
        chunked = model._encode_pool_cut_bonds(*args)

    assert chunked.shape == (NUM_FRAGS, model.mol_embedder.hidden_size)
    assert th.allclose(single, chunked, atol=1e-6)


@pytest.mark.parametrize("feat", ["boundary_pair", "cut_bond"])
def test_non_decomposing_pool_still_goes_through_mol_pool(monkeypatch, feat):
    """`max` does not compose as a sum accumulation, so it must keep the single-shot path."""
    model = _model(feat, mol_pool_type="max")
    encode = _encoder(model, feat)
    atom_embed, frag_idxs, in_idxs, out_idxs = _pair_inputs(model)

    with th.no_grad():
        expected = model.mol_pool(
            encode(
                th.index_select(atom_embed, 0, in_idxs),
                th.index_select(atom_embed, 0, out_idxs),
            ),
            frag_idxs,
            dim_size=NUM_FRAGS,
        )
        monkeypatch.setattr(fragnnet_model, "CUT_BOND_CHUNK_ELEMS", model.mol_embedder.hidden_size)
        got = model._encode_pool_cut_bonds(
            encode, atom_embed, frag_idxs, in_idxs, out_idxs, NUM_FRAGS
        )

    assert th.allclose(expected, got, atol=1e-6)


@pytest.mark.parametrize("feat", ["boundary_pair", "cut_bond"])
def test_training_keeps_single_shot_path_and_stays_differentiable(monkeypatch, feat):
    """Chunking cannot help under autograd (every chunk's input stays saved), so skip it."""
    model = _model(feat)
    encode = _encoder(model, feat)
    atom_embed, frag_idxs, in_idxs, out_idxs = _pair_inputs(model)
    atom_embed.requires_grad_(True)
    monkeypatch.setattr(fragnnet_model, "CUT_BOND_CHUNK_ELEMS", model.mol_embedder.hidden_size)

    pooled = model._encode_pool_cut_bonds(
        encode, atom_embed, frag_idxs, in_idxs, out_idxs, NUM_FRAGS
    )
    pooled.sum().backward()

    assert atom_embed.grad is not None
    assert th.isfinite(atom_embed.grad).all()


@pytest.mark.parametrize("feat", ["boundary_pair", "cut_bond"])
def test_forward_predictions_unchanged_by_chunking(monkeypatch, feat):
    """End-to-end: chunking is invisible in the model output on a real molecule's DAG."""
    model = _model(feat)
    kwargs = _forward_kwargs_with_cut_bonds()
    assert kwargs["boundary_pair_frag_idxs"].numel() > 1, "test would not exercise chunking"

    with th.no_grad():
        single = model(**kwargs)["pred_logprobs"]
        monkeypatch.setattr(fragnnet_model, "CUT_BOND_CHUNK_ELEMS", model.mol_embedder.hidden_size)
        chunked = model(**kwargs)["pred_logprobs"]

    assert th.allclose(single, chunked, atol=1e-6)
