"""Tests for the dropout-view disagreement probe's peak-group comparisons."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch as th

from fragnnet.utils.spec_utils import batched_bin_func, cos_sim_helper

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "debug" / "debug_masster_dropout_view_disagreement.py"
)
SPEC = importlib.util.spec_from_file_location(
    "debug_masster_dropout_view_disagreement", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def pathways(batch: list[int], mzs: list[float], probabilities: list[float]) -> tuple:
    """(spectrum, m/z, log probability) rows as the probe passes them around."""
    return (
        th.tensor(batch, dtype=th.long),
        th.tensor(mzs, dtype=th.float64),
        th.tensor(probabilities, dtype=th.float64).log(),
    )


@pytest.fixture
def two_spectra() -> tuple:
    """Spectrum 0 splits a peak over two isomers; spectrum 1 has one dominant peak."""
    return pathways(
        [0, 0, 0, 1, 1], [100.05, 100.05, 150.10, 80.02, 95.07], [0.3, 0.2, 0.5, 0.9, 0.1]
    )


def test_identical_predictions_have_zero_distance(two_spectra: tuple) -> None:
    distances = PROBE.pair_distances(two_spectra, two_spectra, 2, binned=False)
    assert distances["tv"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert distances["symkl"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert distances["js"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert distances["cos"] == pytest.approx([1.0, 1.0])
    assert distances["top1_same"].tolist() == [True, True]
    assert distances["churn"] == pytest.approx([0.0, 0.0])


def test_isomers_at_one_mass_are_one_peak(two_spectra: tuple) -> None:
    # The same spectrum 0 with its 0.5 at 100.05 carried by one isomer instead of two.
    merged = pathways([0, 0, 1, 1], [100.05, 150.10, 80.02, 95.07], [0.5, 0.5, 0.9, 0.1])
    distances = PROBE.pair_distances(two_spectra, merged, 2, binned=False)
    assert distances["tv"] == pytest.approx([0.0, 0.0], abs=1e-12)
    assert distances["symkl"] == pytest.approx([0.0, 0.0], abs=1e-12)


def test_known_values_for_two_separated_peaks() -> None:
    first = pathways([0, 0], [100.0, 200.0], [0.75, 0.25])
    second = pathways([0, 0], [100.0, 200.0], [0.25, 0.75])
    distances = PROBE.pair_distances(first, second, 1, binned=False)
    assert distances["tv"] == pytest.approx([0.5])
    # 0.5 * sum (p - q)(log p - log q) = 0.5 * (0.5 log 3 + 0.5 log 3).
    assert distances["symkl"] == pytest.approx([0.5 * math.log(3.0)])
    assert distances["cos"] == pytest.approx([0.375 / 0.625])
    assert distances["top1_same"].tolist() == [False]


def test_one_sided_group_is_churn_and_infinite_kl() -> None:
    first = pathways([0, 0], [100.0, 200.0], [0.9, 0.1])
    second = pathways([0], [100.0], [1.0])
    distances = PROBE.pair_distances(first, second, 1, binned=False)
    assert distances["tv"] == pytest.approx([0.1])
    assert distances["churn"] == pytest.approx([0.05])
    assert math.isinf(distances["symkl"][0])
    assert 0.0 < distances["js"][0] < math.log(2.0)


def test_binned_cosine_reproduces_the_logged_metric() -> None:
    # Spectrum 0: a target recorded at exactly 141.01 shares the logged metric's bin with a
    # 141.0102 prediction (float64 floor(mz / 0.01) would split them). Spectrum 1: a target at
    # 110.99 and a prediction at 110.9899 straddle an edge, so the metric scores them apart.
    predictions = pathways(
        [0, 0, 1, 1], [141.0102, 158.0367, 110.9899, 82.995], [0.8, 0.2, 0.9, 0.1]
    )
    target = (
        th.tensor([0, 0, 1, 1]),
        th.tensor([141.01, 158.0365, 110.99, 82.9951], dtype=th.float32),
        th.tensor([0.8, 0.2, 0.9, 0.1], dtype=th.float64).log(),
    )
    ours = PROBE.pair_distances(predictions, target, 2, binned=True)["cos"]

    binned = []
    for batch, mzs, log_p in (predictions, target):
        binned.append(
            batched_bin_func(
                mzs,
                log_p.exp().to(mzs.dtype),
                batch,
                mz_max=PROBE.MZ_MAX,
                mz_bin_res=PROBE.BIN_DA,
                agg="sum",
                sparse=True,
            )
        )
    logged = cos_sim_helper(*binned[1], *binned[0])
    assert ours == pytest.approx(logged.to(th.float64).numpy(), abs=1e-6)
    assert ours[0] == pytest.approx(1.0, abs=1e-3)
    assert ours[1] < 0.2


def test_flattening_and_mixture() -> None:
    prediction = pathways([0, 0, 0], [100.0, 100.0, 200.0], [0.4, 0.4, 0.2])
    identity = PROBE.flattened(prediction, 1.0, 1)
    assert identity[2].exp().tolist() == pytest.approx([0.8, 0.2])
    halved = PROBE.flattened(prediction, 0.5, 1)
    expected = [math.sqrt(0.8), math.sqrt(0.2)]
    assert halved[2].exp().tolist() == pytest.approx([value / sum(expected) for value in expected])
    averaged = PROBE.mixture(prediction, prediction)
    assert PROBE.pair_distances(averaged, prediction, 1, binned=False)["tv"] == pytest.approx([0.0])


def test_support_mismatch_ignores_order_but_not_multiplicity() -> None:
    reference = pathways([0, 0, 0, 1], [100.0, 100.0, 120.0, 90.0], [0.25, 0.25, 0.5, 1.0])
    permuted = pathways([1, 0, 0, 0], [90.0, 120.0, 100.0, 100.0], [1.0, 0.5, 0.25, 0.25])
    swapped = pathways([0, 0, 0, 1], [100.0, 120.0, 120.0, 90.0], [0.25, 0.25, 0.5, 1.0])
    assert PROBE.support_mismatch(permuted, reference, 2).tolist() == [False, False]
    assert PROBE.support_mismatch(swapped, reference, 2).tolist() == [True, False]


def test_mc_head_sizes_are_powers_of_two_up_to_the_view_count() -> None:
    assert PROBE.mc_head_sizes(2) == [1, 2]
    assert PROBE.mc_head_sizes(8) == [1, 2, 4, 8]
    for bad in (1, 6, 32):
        with pytest.raises(ValueError, match="power of two"):
            PROBE.mc_head_sizes(bad)


def test_mc_limit_weights_recover_the_intercept_of_a_one_over_n_curve() -> None:
    sizes = [1, 2, 4, 8]
    weights = PROBE.mc_limit_weights(sizes)
    assert weights.sum() == pytest.approx(1.0)
    # y_n = S_inf - k / n exactly: the weighted sum returns S_inf for any k.
    for limit, slope in ((0.01, 0.015), (-0.003, 0.0)):
        curve = [limit - slope / n for n in sizes]
        assert float(weights @ curve) == pytest.approx(limit, abs=1e-12)
    # Two sizes reduce to the two-point extrapolation 2 y(2) - y(1).
    assert PROBE.mc_limit_weights([1, 2]).tolist() == pytest.approx([-1.0, 2.0])
