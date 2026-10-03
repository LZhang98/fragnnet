import numpy as np
import pytest

from fragnnet.utils.retrieval_utils import (
    entropy_similarity_inst_tol,
    expected_hits,
    molecule_bootstrap_ci,
    mspred_bin,
    mspred_bootstrap_ci,
    mspred_cosine,
    mspred_entropy,
    mspred_prediction,
    mspred_target,
    prepare_sqrt_peaks,
    query_log_likelihood,
)


@pytest.fixture
def spectrum():
    mz = np.array([91.0542, 119.0491, 147.0441, 175.0390])
    inten = np.array([400.0, 1000.0, 90.0, 10.0])
    return mz, inten


def _dense_entropy_sim(p, q):
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    h = lambda x: -(x[x > 0] * np.log(x[x > 0])).sum()  # noqa: E731
    return 1 - (2 * h(m) - h(p) - h(q)) / np.log(4)


# ms-pred replicas ------------------------------------------------------------------------------


def test_mspred_bin_index_and_pooling():
    # floor(mz * 149999 / 1500) + 1 (bin width 0.0100007 Da); 100.0 and 100.0005 share a bin
    idx, vals = mspred_bin(np.array([100.0, 100.0005, 250.5]), np.array([1.0, 3.0, 2.0]), "max")
    assert idx.tolist() == [
        int(np.floor(100.0 * 149999 / 1500)) + 1,
        int(np.floor(250.5 * 149999 / 1500)) + 1,
    ]
    assert vals.tolist() == [3.0, 2.0]
    _, vals_add = mspred_bin(np.array([100.0, 100.0005, 250.5]), np.array([1.0, 3.0, 2.0]), "add")
    assert vals_add.tolist() == [4.0, 2.0]
    with pytest.raises(ValueError):
        mspred_bin(np.array([1.0]), np.array([1.0]), "mean")


def test_mspred_target_drops_above_parent_and_sqrt_normalizes(spectrum):
    mz, inten = spectrum
    idx, vals = mspred_target(np.append(mz, 300.0), np.append(inten, 500.0), parent_mass=176.0)
    assert len(idx) == 4  # 300 > 176 + 1 removed
    assert vals.max() == pytest.approx(1.0)
    assert vals.min() == pytest.approx(np.sqrt(10.0 / 1000.0))


def test_mspred_cosine_and_entropy_identical_disjoint_and_dense(spectrum):
    mz, inten = spectrum
    tgt = mspred_target(mz, inten, parent_mass=500.0)
    same = mspred_prediction(mz, inten / inten.max(), sqrt_input=False)
    assert mspred_cosine(tgt, same) == pytest.approx(1.0)
    assert mspred_entropy(tgt, same) == pytest.approx(1.0)
    far = mspred_prediction(mz + 5.0, inten, sqrt_input=False)
    assert mspred_cosine(tgt, far) == pytest.approx(0.0)
    assert mspred_entropy(tgt, far) == pytest.approx(0.0, abs=1e-12)
    # different intensities on the same bins: matches the dense formulas
    other = np.array([100.0, 300.0, 600.0, 50.0])
    pred = mspred_prediction(mz, other, sqrt_input=False)
    t, p = np.sqrt(inten / inten.max()), np.sqrt(other)
    assert mspred_cosine(tgt, pred) == pytest.approx(t @ p / np.linalg.norm(t) / np.linalg.norm(p))
    assert mspred_entropy(tgt, pred) == pytest.approx(_dense_entropy_sim(p, t))
    assert mspred_cosine(tgt, (np.zeros(0, int), np.zeros(0))) == 0.0


def test_mspred_prediction_sqrt_input_and_filter(spectrum):
    mz, inten = spectrum
    lin = inten / inten.max()
    a = mspred_prediction(mz, np.sqrt(lin), sqrt_input=True)
    b = mspred_prediction(mz, lin, sqrt_input=False)
    assert np.allclose(a[1], b[1])
    f = mspred_prediction(mz, lin, sqrt_input=False, min_rel_intensity=0.05)
    assert len(f[0]) == 3  # 10/1000 < 5% removed, 90/1000 kept


# instrument-tolerance entropy -------------------------------------------------------------------


def test_entropy_inst_tol_identity_tolerance_and_dense(spectrum):
    mz, inten = spectrum
    tgt = prepare_sqrt_peaks(mz, inten)
    preds = [
        prepare_sqrt_peaks(mz, inten),  # identical
        prepare_sqrt_peaks(mz + 0.0015, inten),  # 1.5 mDa: inside the 2 mDa floor (10 ppm @ 200)
        prepare_sqrt_peaks(mz + 0.0030, inten),  # 3 mDa: outside
        prepare_sqrt_peaks(mz, np.array([100.0, 300.0, 600.0, 50.0])),
    ]
    s = entropy_similarity_inst_tol(tgt, preds, rel=1e-5, min_mz=200.0)
    assert s[0] == pytest.approx(1.0)
    assert s[1] == pytest.approx(1.0)
    assert s[2] == pytest.approx(0.0, abs=1e-12)
    assert s[3] == pytest.approx(
        _dense_entropy_sim(np.sqrt([100.0, 300.0, 600.0, 50.0]), np.sqrt(inten))
    )


def test_entropy_inst_tol_one_to_one_and_chunk_invariance(spectrum):
    mz, inten = spectrum
    tgt = prepare_sqrt_peaks(mz, inten)
    # two predicted peaks within tolerance of one target peak: only one can match
    split = prepare_sqrt_peaks(np.array([119.0491, 119.0495]), np.array([1.0, 1.0]))
    s_split = entropy_similarity_inst_tol(tgt, [split], rel=1e-5, min_mz=200.0)[0]
    single = prepare_sqrt_peaks(np.array([119.0491]), np.array([1.0]))
    s_single = entropy_similarity_inst_tol(tgt, [single], rel=1e-5, min_mz=200.0)[0]
    assert 0.0 < s_split < s_single
    preds = [prepare_sqrt_peaks(mz * (1 + 1e-6 * k), inten * (1 + k)) for k in range(7)]
    a = entropy_similarity_inst_tol(tgt, preds, rel=1e-5, min_mz=200.0, chunk_size=2)
    b = entropy_similarity_inst_tol(tgt, preds, rel=1e-5, min_mz=200.0, chunk_size=32)
    assert np.allclose(a, b)


def test_prepare_sqrt_peaks_filters():
    m, p = prepare_sqrt_peaks(
        np.array([100.0, 100.00001, 200.0, 400.0]),
        np.array([1.0, 1.0, 98.0, 50.0]),
        max_mz=300.0,
        min_rel_intensity=0.05,
    )
    assert m.tolist() == [200.0]  # 100 merged (2 < 5% of 98) and removed; 400 above max_mz
    assert float(p.sum()) == pytest.approx(1.0)


# log-likelihood --------------------------------------------------------------------------------


def test_query_log_likelihood_window_and_floor():
    q_mz, q_p = np.array([100.0, 200.0]), np.array([0.5, 0.5])
    p_mz, p_p = np.array([100.0005, 100.0015, 300.0]), np.array([0.3, 0.2, 0.5])
    # window at 100 is +-2 mDa (10 ppm of the 200 floor): both nearby peaks count; 200 gets eps
    ll = query_log_likelihood(q_mz, q_p, p_mz, p_p, rel=1e-5, min_mz=200.0, eps=1e-5)
    assert ll == pytest.approx(0.5 * np.log(0.5) + 0.5 * np.log(1e-5))


# ranking and intervals -------------------------------------------------------------------------


def test_expected_hits_ties_multiple_truths_and_nan():
    assert expected_hits(np.array([0.9, 0.5, 0.1]), np.array([True, False, False])) == [
        1.0,
        1.0,
        1.0,
    ]
    assert expected_hits(np.array([0.5, 0.5, 0.9]), np.array([True, False, False]), ks=(1, 2)) == [
        0.0,
        0.5,
    ]
    # truth tied with 3 decoys at the top: 1/4 for hit@1
    assert expected_hits(np.full(4, 0.7), np.array([True, False, False, False]), ks=(1,))[
        0
    ] == pytest.approx(0.25)
    # two tied truths among 4 tied candidates: P(a truth first) = 1/2
    assert expected_hits(np.full(4, 0.7), np.array([True, True, False, False]), ks=(1,))[
        0
    ] == pytest.approx(0.5)
    # NaN truth ranks last
    assert expected_hits(np.array([np.nan, 0.1]), np.array([True, False]), ks=(1,))[0] == 0.0
    with pytest.raises(ValueError):
        expected_hits(np.array([0.1]), np.array([False]))


def test_molecule_bootstrap_ci_clusters_and_mspred_ci():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(200), 5)  # 5 identical spectra per molecule
    v = np.repeat(rng.integers(0, 2, 200), 5).astype(float)
    mean, lo, hi = molecule_bootstrap_ci(v, groups)
    _, mlo, mhi = mspred_bootstrap_ci(v, level=0.95, n_resamples=2000)
    assert lo < mean < hi
    # correlated spectra: the molecule interval is wider than the i.i.d. one
    assert (hi - lo) > (mhi - mlo)
    assert molecule_bootstrap_ci(v, groups) == molecule_bootstrap_ci(v, groups)
