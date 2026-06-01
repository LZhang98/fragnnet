from types import SimpleNamespace

import torch as th

from fragnnet.pl_model.spectrum_mol_clip_pl import SpectrumMolClipPL


def test_retrieval_metrics_perfect_alignment() -> None:
    embed = th.eye(4)

    metrics = SpectrumMolClipPL._retrieval_metrics(embed, embed, "s2m", top_k=5)

    assert th.isclose(metrics["s2m_top1"], th.tensor(1.0))
    assert th.isclose(metrics["s2m_top4"], th.tensor(1.0))
    assert th.isclose(metrics["s2m_mrr"], th.tensor(1.0))


def test_retrieval_metrics_detects_bad_ranking() -> None:
    query = th.eye(3)
    target = th.roll(th.eye(3), shifts=1, dims=0)

    metrics = SpectrumMolClipPL._retrieval_metrics(query, target, "s2m", top_k=2)

    assert metrics["s2m_top1"] < 1.0
    assert metrics["s2m_top2"] <= 1.0
    assert metrics["s2m_mrr"] < 1.0


def test_retrieval_metrics_exact_ranks() -> None:
    # query[0] = [1,0,0], target[0] = [0.6,0.8,0] → diag_sim=0.6
    #   target[1]=[0.8,0.6,0] scores 0.8 > 0.6 → rank[0]=2
    # query[1] = [0,1,0], target[1] = [0.8,0.6,0] → diag_sim=0.6
    #   target[0]=[0.6,0.8,0] scores 0.8 > 0.6 → rank[1]=2
    # query[2] = [0,0,1], target[2] = [0,0,1] → diag_sim=1.0, no target scores higher → rank[2]=1
    query = th.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    target = th.tensor([[0.6, 0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])

    metrics = SpectrumMolClipPL._retrieval_metrics(query, target, "s2m", top_k=2)

    assert th.isclose(metrics["s2m_top1"], th.tensor(1.0 / 3.0), atol=1e-6)
    assert th.isclose(metrics["s2m_top2"], th.tensor(1.0), atol=1e-6)
    assert th.isclose(metrics["s2m_mrr"], th.tensor((0.5 + 0.5 + 1.0) / 3.0), atol=1e-6)


def test_retrieval_metrics_chunk_matches_full() -> None:
    # chunked computation must give the same result as a single pass
    th.manual_seed(0)
    embed = th.randn(16, 8)
    embed = th.nn.functional.normalize(embed, dim=-1)

    metrics_full = SpectrumMolClipPL._retrieval_metrics(embed, embed, "p", top_k=5, chunk_size=16)
    metrics_chunked = SpectrumMolClipPL._retrieval_metrics(embed, embed, "p", top_k=5, chunk_size=3)

    for key in metrics_full:
        assert th.isclose(metrics_full[key], metrics_chunked[key], atol=1e-6), key


def test_retrieval_metrics_topk_capped_at_n() -> None:
    # top_k > n should be silently capped; key name uses the capped value
    embed = th.eye(3)
    metrics = SpectrumMolClipPL._retrieval_metrics(embed, embed, "s2m", top_k=10)
    assert "s2m_top3" in metrics
    assert th.isclose(metrics["s2m_top3"], th.tensor(1.0))


def test_retrieval_metrics_single_item() -> None:
    embed = th.tensor([[0.6, 0.8]])
    metrics = SpectrumMolClipPL._retrieval_metrics(embed, embed, "s2m", top_k=5)
    assert th.isclose(metrics["s2m_top1"], th.tensor(1.0))
    assert th.isclose(metrics["s2m_mrr"], th.tensor(1.0))


def test_build_peak_features_normalizes_mz_and_intensity() -> None:
    module = SimpleNamespace(
        clip_intensity_transform="sqrt",
        hparams=SimpleNamespace(mz_max=1000.0),
    )
    # spec 0: prec_mz=300, peaks at mz=[100,200]; spec 1: prec_mz=100, peak at mz=[50]
    batch = {
        "spec_mzs": th.tensor([100.0, 200.0, 50.0]),
        "spec_ints": th.tensor([1.0, 4.0, 9.0]),
        "spec_batch_idxs": th.tensor([0, 0, 1]),
        "batch_size": th.tensor(2),
        "spec_prec_mz": th.tensor([300.0, 100.0]),
    }

    features = SpectrumMolClipPL._build_peak_features(module, batch)

    # columns: [mzs_norm, ints_norm, neutral_loss_norm, prec_norm]
    # mzs_norm = mz / 1000; ints_norm = sqrt(int) / max_sqrt_int per spec
    # neutral_loss_norm = (prec_mz - mz) / 1000; prec_norm = prec_mz / 1000
    expected = th.tensor(
        [
            [0.10, 0.5, 0.20, 0.30],
            [0.20, 1.0, 0.10, 0.30],
            [0.05, 1.0, 0.05, 0.10],
        ]
    )
    assert th.allclose(features, expected, atol=1e-6)
