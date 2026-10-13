"""The frozen-token allocation probe.

Script: ``scripts/debug/masster_memorization/debug_masster_local_context_allocation.py``.

Covers the pure helpers (segment ops, the context penalty, the compact-bin production cosine,
manifests, config merging, calibration), the two local scorers (zero start, retained/lost
orientation, radius-2 locality, batch isolation) and the capture/rebuild of a real tiny
MASSter forward, which the extraction relies on to align cached tokens with the model output.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th

import scripts.debug.masster_memorization.debug_masster_local_context_allocation as PROBE
from fragnnet.loss.cross_entropy import get_sparse_cross_entropy_fn
from fragnnet.utils.spec_similarity import cos_sim_helper
from tests.test_masster_model import _build_batch, _build_model

# 1. Segment helpers and the penalty.


def test_segment_logsumexp_matches_per_segment() -> None:
    values = th.tensor([0.5, -1.0, 2.0, 3.0, -th.inf, 1.0])
    segments = th.tensor([0, 0, 1, 1, 2, 2])
    out = PROBE.segment_logsumexp(values, segments, 4)
    for k in range(3):
        assert out[k] == pytest.approx(float(th.logsumexp(values[segments == k], 0)))
    assert out[3] == -th.inf


def test_context_penalty_offset_invariant_and_zero_for_constant() -> None:
    context = th.tensor([1.0, 3.0, -2.0, 0.0, 0.0])
    segments = th.tensor([0, 0, 1, 1, 1])
    penalty = PROBE.context_penalty(context, segments, 2)
    # Spectrum 0: mean 2, deviations +-1 -> 1. Spectrum 1: the variance of (-2, 0, 0).
    assert penalty[0].item() == pytest.approx(1.0)
    assert penalty[1].item() == pytest.approx(float(np.var([-2.0, 0.0, 0.0])))
    offsets = th.tensor([5.0, 5.0, -7.0, -7.0, -7.0])
    assert th.allclose(PROBE.context_penalty(context + offsets, segments, 2), penalty)
    assert th.allclose(PROBE.context_penalty(th.full((5,), 4.0), segments, 2), th.zeros(2))


# 2. The compact-bin production cosine.


def _aggregate(spec: np.ndarray, bins: np.ndarray, vals: np.ndarray, num_bins: int) -> tuple:
    frame = pd.DataFrame({"s": spec, "b": bins, "v": vals}).groupby(["s", "b"]).v.sum()
    idx = np.array([s * num_bins + b for s, b in frame.index])
    return th.tensor(idx), th.tensor(frame.to_numpy()), th.tensor([s for s, _ in frame.index])


def test_binned_cosine_matches_cos_sim_helper() -> None:
    rng = np.random.default_rng(3)
    num_bins = 1000
    pred_spec, pred_bin, pred_p = (
        np.array([0, 0, 0, 1, 1]),
        np.array([10, 10, 25, 4, 9]),
        rng.random(5),
    )
    true_spec, true_bin, true_p = (
        np.array([0, 0, 1, 1, 1]),
        np.array([10, 40, 4, 9, 11]),
        rng.random(5),
    )
    pred_c, true_c, bin_spec, _ = PROBE._compact_bins(
        th.tensor(pred_spec * num_bins + pred_bin),
        th.tensor(true_spec * num_bins + true_bin),
        num_bins,
    )
    tbin = th.zeros(bin_spec.numel(), dtype=th.float64).index_add(0, true_c, th.tensor(true_p))
    ours = PROBE.binned_cosine(th.tensor(np.log(pred_p)), pred_c, tbin, bin_spec, 2)
    reference = cos_sim_helper(
        *_aggregate(true_spec, true_bin, true_p, num_bins),
        *_aggregate(pred_spec, pred_bin, pred_p, num_bins),
    )
    assert th.allclose(ours, reference.to(ours.dtype), atol=1e-12)


def test_binned_cosine_drops_out_of_range_channels_and_handles_empty() -> None:
    tbin, bin_spec = th.tensor([1.0, 0.0]), th.tensor([0, 0])
    # One in-range channel on bin 0; the dropped (-1) channel carries most of the mass.
    cos = PROBE.binned_cosine(th.log(th.tensor([0.1, 0.9])), th.tensor([0, -1]), tbin, bin_spec, 1)
    assert cos.item() == pytest.approx(1.0)
    empty = PROBE.binned_cosine(th.log(th.tensor([1.0])), th.tensor([-1]), tbin, bin_spec, 1)
    assert empty.item() == 0.0


# 3. Manifests, ranges, bins and config merging.


def _write_split(dp: Path, name: str, mols: list[int], per_mol: int = 2) -> None:
    rows = [(1000 * m + i, m, m) for m in mols for i in range(per_mol)]
    pd.DataFrame(rows, columns=["spec_id", "mol_id", "group_id"]).to_csv(
        dp / f"{name}_ids.csv", index=False
    )


def test_build_manifests_splits_and_disjointness(tmp_path: Path) -> None:
    local, canon = tmp_path / "local", tmp_path / "canon"
    local.mkdir()
    canon.mkdir()
    _write_split(canon, "train", [1, 2, 3, 4])
    _write_split(canon, "val", [5, 6, 7])
    _write_split(canon, "test", [8])
    _write_split(local, "train", [1, 2])
    _write_split(local, "val", [5])
    manifests = PROBE.build_manifests(local, canon)
    assert set(manifests["local"]["fit"].mol_id) == {1, 2}
    assert set(manifests["local"]["tune"].mol_id) == {3, 4}
    assert set(manifests["full"]["tune"].mol_id) == {6, 7}
    assert set(manifests["local"]["confirm"].mol_id) == {5}
    assert len(manifests["local"]["tune"]) == 4  # every CE row of a molecule kept together
    _write_split(local, "val", [5, 3])  # a canonical train molecule in local val
    with pytest.raises(ValueError, match="local val"):
        PROBE.build_manifests(local, canon)


def test_ranges_and_nce_bins() -> None:
    assert PROBE._ranges(np.array([5, 0, 9]), np.array([2, 0, 3])).tolist() == [5, 6, 9, 10, 11]
    bins = PROBE.nce_bins(np.array([10.0, 20.0, 20.5, 40.0, 60.0, 61.0]))
    assert bins.tolist() == [0, 0, 1, 1, 2, 3]


def test_merge_config_fills_only_missing_keys_including_nested() -> None:
    base = {"a": 1, "new_key": 0.0, "spec_params": {"x": 1, "fill_missing_ace": False}}
    own = {"a": 2, "spec_params": {"x": 5}}
    merged, filled = PROBE.merge_config(base, own)
    assert merged == {"a": 2, "new_key": 0.0, "spec_params": {"x": 5, "fill_missing_ace": False}}
    assert filled == {"new_key": 0.0, "spec_params.fill_missing_ace": False}


# 4. Local scorers.


def _local_batch(graphs: list[tuple], cuts: list[list]) -> PROBE.Batch:
    """A batch holding only what the local scorers read.

    Args:
        graphs: Per spectrum, atom features ``(n, len(ATOM_COLUMNS))`` and a bond list.
        cuts: Per spectrum, fragments as lists of ``(bond index, inside atom, outside atom)``;
            an empty list is the intact precursor.
    """
    xs, edges, attrs, frag_spec, prec, b_frag, b_atoms, b_edge = ([] for _ in range(8))
    atom_base = bond_base = frag = 0
    for spec, ((x, bonds), fragments) in enumerate(zip(graphs, cuts, strict=True)):
        xs.append(x)
        for u, v in bonds:
            edges.append((u + atom_base, v + atom_base))
            attrs.append(np.eye(len(PROBE.BOND_COLUMNS))[0])
        for fragment in fragments:
            frag_spec.append(spec)
            prec.append(len(fragment) == 0)
            for bond, inside, outside in fragment:
                b_frag.append(frag)
                b_atoms.append((inside + atom_base, outside + atom_base))
                b_edge.append(bond + bond_base)
            frag += 1
        atom_base += x.shape[0]
        bond_base += len(bonds)
    index = th.tensor(edges, dtype=th.long).T
    attr = th.tensor(np.array(attrs), dtype=th.float32)
    graph = {
        "x": th.tensor(np.concatenate(xs), dtype=th.float32),
        "edge_index": th.cat([index, index.flip(0)], dim=1),
        "edge_attr": th.cat([attr, attr]),
        "bond_attr": attr,
    }
    size = len(graphs)
    fields = dict.fromkeys(
        (
            "tokens",
            "s_ckpt",
            "chan_frag",
            "chan_lp",
            "chan_mz",
            "chan_cbin",
            "tbin",
            "tbin_np",
            "bin_spec",
            "true_mz",
            "true_lp",
            "true_spec",
            "tol",
            "min_mz",
            "rows",
        )
    )
    return PROBE.Batch(
        size=size,
        frag_spec=th.tensor(frag_spec),
        cut=th.tensor([0 if p else 1 for p in prec]),
        prec=th.tensor(prec),
        acq=th.tensor([[0.3, 1.0, 0.0]] * size, dtype=th.float32),
        nce_bin=th.zeros(size, dtype=th.long),
        graph=graph,
        bnd_frag=th.tensor(b_frag, dtype=th.long),
        bnd_atoms=th.tensor(b_atoms, dtype=th.long).T,
        bnd_edge=th.tensor(b_edge, dtype=th.long),
        **fields,
    )


def _chain(n: int, seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    x = np.random.default_rng(seed).random((n, len(PROBE.ATOM_COLUMNS)))
    return x, [(i, i + 1) for i in range(n - 1)]


def _randomize(module: th.nn.Module) -> None:
    th.manual_seed(11)
    with th.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.5)


def test_local_scorers_start_at_zero() -> None:
    b = _local_batch([_chain(5)], [[[], [(1, 1, 2)], [(1, 2, 1)]]])
    assert th.equal(PROBE.LocalCutScorer()(b), th.zeros(3))
    assert th.equal(PROBE.CutCountScorer()(b), th.zeros(3))


def test_local_scorer_orientation_and_radius_two_locality() -> None:
    chain = _chain(6)
    # The same six atoms plus one more atom bonded to atom 5: three bonds from atom 3 and
    # four from atom 2, so outside the radius-2 environment of bond 2 = (2, 3).
    extra = np.random.default_rng(9).random((1, len(PROBE.ATOM_COLUMNS)))
    longer = (np.vstack([chain[0], extra]), [*chain[1], (5, 6)])
    fragments = [[], [(2, 2, 3)], [(2, 3, 2)], [(4, 4, 5)]]
    scorer = PROBE.LocalCutScorer()
    _randomize(scorer)
    scores = scorer(_local_batch([chain, longer], [fragments, fragments]))
    short, long_ = scores[:4], scores[4:]
    assert short[1].item() == pytest.approx(long_[1].item(), abs=1e-5)  # remote change ignored
    assert abs(short[1].item() - short[2].item()) > 1e-4  # retained/lost orientation matters
    assert abs(short[3].item() - long_[3].item()) > 1e-4  # a change within radius 2 is seen
    assert short[0].item() == pytest.approx(long_[0].item())  # acquisition-only survival


def test_local_scorer_batch_isolation_and_cut_sum() -> None:
    fragments = [[], [(1, 1, 2)], [(0, 0, 1), (2, 2, 3)]]
    scorer = PROBE.LocalCutScorer()
    _randomize(scorer)
    alone = scorer(_local_batch([_chain(4)], [fragments]))
    together = scorer(_local_batch([_chain(7, seed=4), _chain(4)], [[[], [(3, 3, 4)]], fragments]))
    assert th.allclose(alone, together[2:], atol=1e-6)
    # A two-cut fragment sums the scores of its two oriented boundary bonds.
    parts = scorer(_local_batch([_chain(4)], [[[(0, 0, 1)], [(2, 2, 3)]]]))
    assert alone[2].item() == pytest.approx(float(parts.sum()), rel=1e-6)


# 5. Arms, loss and calibration.


def test_arm_starts_at_checkpoint_head_and_penalty_reaches_only_the_head() -> None:
    weight, bias = th.randn(1, 8), th.randn(1)
    b = _local_batch([_chain(5)], [[[], [(1, 1, 2)], [(3, 3, 4)]]])
    b.tokens = th.randn(3, 8)
    model = PROBE.ArmModel(PROBE.ArmSpec("R1", True, "gine", "weak", "joint"), weight, bias)
    s, _c, local = model(b)
    assert th.allclose(s, b.tokens @ weight[0] + bias)
    assert th.equal(local, th.zeros(3))
    _randomize(model.local)
    _s, c, _l = model(b)
    PROBE.context_penalty(c, b.frag_spec, b.size).sum().backward()
    assert model.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.local.parameters())


def _toy_batch() -> PROBE.Batch:
    return PROBE.Batch(
        size=2,
        frag_spec=th.tensor([0, 0, 1, 1]),
        tokens=None,
        s_ckpt=None,
        cut=th.tensor([0, 1, 0, 1]),
        prec=th.tensor([True, False, True, False]),
        chan_frag=th.tensor([0, 1, 1, 2, 3]),
        chan_lp=th.log(th.tensor([1.0, 0.5, 0.5, 1.0, 1.0])),
        chan_mz=th.tensor([300.0, 150.0, 151.0, 250.0, 120.0], dtype=th.float64),
        chan_cbin=th.tensor([0, 1, 2, 3, 4]),
        tbin=th.tensor([0.6, 0.4, 0.0, 0.7, 0.3]),
        tbin_np=th.tensor([0.0, 0.4, 0.0, 0.0, 0.3]),
        bin_spec=th.tensor([0, 0, 0, 1, 1]),
        true_mz=th.tensor([300.0, 150.0, 250.0, 120.0]),
        true_lp=th.log(th.tensor([0.6, 0.4, 0.7, 0.3])),
        true_spec=th.tensor([0, 0, 1, 1]),
        tol=th.tensor([1e-5, 1e-5]),
        min_mz=th.tensor([200.0, 200.0]),
        acq=None,
        nce_bin=th.tensor([0, 2]),
        graph=None,
        bnd_frag=None,
        bnd_atoms=None,
        bnd_edge=None,
        rows=None,
    )


def test_ce_cosine_invariant_to_score_offsets_and_calibration_identity() -> None:
    sce_fn = get_sparse_cross_entropy_fn(
        dist="gaussian",
        vectorized=True,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        oos_tolerance_multiple=1.0,
        gaussian_renormalize=True,
        pm_tolerance_multiple=3.0,
        loss_batch_size=64,
    )
    b = _toy_batch()
    scores = th.tensor([0.3, -0.2, 1.0, 0.1])
    shifted = scores + th.tensor([4.0, 4.0, -3.0, -3.0])
    assert th.allclose(PROBE.spectral_ce(scores, b, sce_fn), PROBE.spectral_ce(shifted, b, sce_fn))
    assert th.allclose(PROBE.spectrum_cosine(scores, b), PROBE.spectrum_cosine(shifted, b))
    # With the precursor removed, spectrum 0 keeps its fragment's two channels (bins 1, 2);
    # only bin 1 has nonprecursor target, so the cosine is 0.5 / sqrt(0.5^2 + 0.5^2).
    np_cos = PROBE.spectrum_cosine(scores, b, nonprecursor=True)
    assert np_cos[0].item() == pytest.approx(1.0 / np.sqrt(2.0), rel=1e-5)
    for family in ("gamma", "nce", "stack"):
        params = PROBE.initial_params(family, th.device("cpu")).float()
        assert th.allclose(PROBE.calibrate_scores(family, params, scores, b, th.ones(4)), scores)


# 6. Capture and rebuild of a real tiny MASSter forward.


def test_forward_capture_rebuilds_the_model_prediction() -> None:
    batch = _build_batch(("CCOC", "CCO"))
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        include_cmf=False,
        max_masks=4,
        max_enumerated_masks=8,
    )
    model.eval()
    capture = PROBE.ForwardCapture(model)
    with th.no_grad():
        out = model(
            mol_pyg=batch, spec_prec_type=th.tensor([0, 0]), spec_nce=th.tensor([10.0, 40.0])
        )
    got = capture.take()
    rebuilt = PROBE.rebuild_prediction(got, 2, out["pred_mzs"].dtype)
    assert th.equal(rebuilt["path_mz"], out["pred_mzs"])
    assert th.equal(rebuilt["v_spec"][rebuilt["path_frag"]], out["pred_batch_idxs"])
    assert th.allclose(rebuilt["path_lp"], out["pred_logprobs"], atol=1e-6)
    assert int(rebuilt["valid"].sum()) == out["ion_h_fragment_batch_idxs"].numel()
    head = model.intensity_head.mask_residual_head
    assert th.allclose(got["tokens"] @ head.weight[0] + head.bias[0], got["residuals"], atol=1e-6)
    capture.close()
    assert "_fragment_masses_and_valid_channels" not in vars(model)
    capture.reset()
    with pytest.raises(RuntimeError, match="fired 0 times"):
        capture.take()


def test_arms_of_one_structure_share_local_initialization() -> None:
    weight, bias = th.randn(1, 8), th.randn(1)
    locals_ = {}
    for spec in PROBE.FITTED_ARMS:
        th.manual_seed(0)
        model = PROBE.ArmModel(spec, weight, bias)
        if model.local is not None:
            locals_[spec.name] = model.local.state_dict()
        assert (model.head is None) == (not spec.context)
    for name in ("R1", "R2", "L"):
        for key, value in locals_["U"].items():
            assert th.equal(value, locals_[name][key]), (name, key)
