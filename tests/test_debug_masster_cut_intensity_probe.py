"""Scientific contracts for the frozen cut-to-intensity screen."""

import numpy as np
import pandas as pd
import pytest
import torch as th

from fragnnet.utils.spec_similarity import batch_cos_hun_helper, calculate_match_mzs
from fragnnet.utils.spec_utils import round_aggregate_peaks
from scripts.debug.masster_memorization.debug_masster_cut_intensity_probe import (
    bond_features,
    bootstrap_delta,
    choose_arm,
    fit_probe,
    fragment_corrections,
    metric_cache,
    report,
    score_channels,
    soft_cut_label,
    split_identities,
)


def config():
    return {
        "seed": 17,
        "probe_kinds": ["encoder", "handmade"],
        "probe_strengths": [0.0, 0.1],
        "proposal_strengths": [0.0, 0.5],
        "cut_count_strengths": [0.0, -0.1, 0.1],
        "temperature_strengths": [0.0, 0.25],
        "gbm_iterations": 3,
        "gbm_leaf_nodes": 3,
        "gbm_learning_rate": 0.1,
        "probability_clip": 0.0001,
    }


def test_ambiguous_label_ignores_model_shares_and_duplicate_channels():
    endpoints = np.array([[0, 1], [1, 2]])
    structures = [((0,), 0, 0, 0.99), ((0, 1), 0, 0, 0.01)]
    np.testing.assert_array_equal(soft_cut_label(structures, endpoints, 3), [0.5, 0.5])
    altered = [((0,), 1, 0, 0.0), ((0,), 0, 1, 0.0), ((0, 1), 0, 0, 1.0)]
    np.testing.assert_array_equal(soft_cut_label(altered, endpoints, 3), [0.5, 0.5])
    np.testing.assert_array_equal(soft_cut_label([((0, 1, 2), 0, 0, 1.0)], endpoints, 3), [0, 0])
    with pytest.raises(ValueError, match="invalid"):
        soft_cut_label([((4,), 0, 0, 1)], endpoints, 3)


def test_split_keeps_identity_aliases_together_and_is_deterministic():
    mols = np.array([1, 2, 3, 4, 5])
    identities = {1: "A", 2: "A", 3: "B", 4: "C", 5: "D"}
    first, second = split_identities(mols, identities, 0.5, 19)
    assert (1 in first) == (2 in first)
    assert first.isdisjoint(second) and first | second == set(mols)
    assert {identities[m] for m in first}.isdisjoint({identities[m] for m in second})
    assert (first, second) == split_identities(mols[::-1], identities, 0.5, 19)


def production_cosine(pm, pi, tm, ti):
    pm, pi, _ = round_aggregate_peaks(
        th.tensor(pm, dtype=th.float64),
        th.tensor(pi, dtype=th.float64),
        th.zeros(len(pm), dtype=th.long),
        agg="sum",
    )
    tm, ti, _ = round_aggregate_peaks(
        th.tensor(tm, dtype=th.float64),
        th.tensor(ti, dtype=th.float64),
        th.zeros(len(tm), dtype=th.long),
        agg="sum",
    )
    tol = 1e-5 * tm.clamp_min(200)
    mask = calculate_match_mzs(
        tm[None], pm[None], tolerance=0.0, relative=False, tol_per_true=tol[None]
    )
    return float(
        batch_cos_hun_helper(
            ti[None], pi[None], mask, mask.any(-1), mask.any(-2), False, None, None
        )[0]
    )


@pytest.mark.parametrize(
    "pm,pi,tm,ti",
    [
        # Two channels sum to the winning peak; an uncovered measured peak stays in the norm.
        ([100.0, 100.00001, 150.0], [0.3, 0.3, 0.4], [100.0, 150.0, 220.0], [0.5, 0.2, 0.3]),
        # One predicted peak matches two targets; intensity may be assigned only once.
        ([100.0009, 200.0], [0.6, 0.4], [100.0, 100.0018, 200.0], [0.2, 0.5, 0.3]),
        # One true peak can match multiple predicted groups; choose the maximum product.
        ([100.0, 100.0015, 150.0], [0.1, 0.7, 0.2], [100.0, 150.0], [0.8, 0.2]),
        ([99.0, 140.0], [0.5, 0.5], [100.0, 150.0], [0.8, 0.2]),
    ],
)
def test_cosine_matches_production_with_unmatched_and_ambiguous_masses(pm, pi, tm, ti):
    cache = metric_cache(np.array(pm), np.array(tm), np.array(ti), 1e-5, 200.0)
    actual, _, _ = score_channels(np.log(pi), cache)
    assert actual == pytest.approx(production_cosine(pm, pi, tm, ti), abs=1e-12)


def test_base_hit_sums_channels_and_does_not_choose_one_latent_fragment():
    cache = metric_cache(
        np.array([100.0, 100.00001, 150.0]),
        np.array([100.0, 150.0]),
        np.array([0.8, 0.2]),
        1e-5,
        200.0,
    )
    _, hit, rank = score_channels(np.log([0.3, 0.3, 0.4]), cache)
    assert hit == 1 and rank == 1
    empty = metric_cache(np.array([]), np.array([100.0]), np.array([1.0]), 1e-5, 200.0)
    cosine, hit, rank = score_channels(np.array([]), empty)
    assert cosine == 0 and hit == 0 and np.isnan(rank)


def test_boundary_sum_zero_strength_and_within_fragment_conditionals():
    inside = np.array([[1, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool)
    endpoints = np.array([[0, 1], [1, 2]])
    corrections = fragment_corrections(
        inside,
        endpoints,
        {"encoder": np.array([2.0, -3.0]), "handmade": np.array([1.0, 1.0])},
        np.array([-1.0, -2.0, -3.0]),
        np.array([-0.4, -1.4, -2.4]),
        12,
        config(),
    )
    np.testing.assert_allclose(corrections["encoder:0.1"], [0.0, -0.3, -0.1])
    np.testing.assert_allclose(corrections["cut_count:0.1"], [0.0, 0.1, 0.2])
    for tag in corrections:
        if tag == "baseline" or tag.endswith(":0.0"):
            np.testing.assert_array_equal(corrections[tag], np.zeros(3))
    lp = np.log([0.2, 0.1, 0.3, 0.4])
    channel_frag = np.array([0, 0, 1, 2])
    adjusted = lp + corrections["encoder:0.1"][channel_frag]
    assert np.exp(adjusted[0] - adjusted[1]) == pytest.approx(2.0)
    assert len(adjusted) == len(lp)  # Every channel survives, including both shared-fragment rows.


def test_probe_features_are_endpoint_symmetric_and_use_no_spectrum_targets():
    pca = {
        "left": np.array([[1.0, 2.0]]),
        "right": np.array([[3.0, 4.0]]),
        "bond_attr": np.array([[0.5]]),
    }
    x = bond_features(pca, pd.DataFrame(), 40.0, "Lumos", "encoder")
    np.testing.assert_array_equal(x, [[4.0, 6.0, 2.0, 2.0, 0.5, 40.0, 1.0]])
    swapped = {**pca, "left": pca["right"], "right": pca["left"]}
    np.testing.assert_array_equal(x, bond_features(swapped, pd.DataFrame(), 40, "ORB_F", "encoder"))


def test_fractional_label_fit_and_calibration_are_finite_and_reproducible():
    x = np.arange(80, dtype=np.float32)[:, None]
    y = np.where(x[:, 0] < 40, 0.1, 0.8)
    weight = np.ones(80)
    first, calibration = fit_probe(x, y, weight, x[::2], y[::2], weight[::2], config())
    second, other = fit_probe(x, y, weight, x[::2], y[::2], weight[::2], config())
    np.testing.assert_array_equal(first.predict(x), second.predict(x))
    np.testing.assert_array_equal(calibration.coef_, other.coef_)
    assert np.isfinite(first.predict(x)).all() and np.isfinite(calibration.coef_).all()


def test_tuning_choice_ignores_confirmation_and_ties_prefer_zero():
    tune = pd.DataFrame({"cos|encoder:0.0": [0.5, 0.7], "cos|encoder:0.1": [0.5, 0.7]})
    assert choose_arm(tune, ["encoder:0.1", "encoder:0.0"]) == "encoder:0.0"
    tune["cos|encoder:0.1"] += 0.01
    assert choose_arm(tune, ["encoder:0.0", "encoder:0.1"]) == "encoder:0.1"


def test_bootstrap_keeps_molecules_and_reports_both_weightings():
    frame = pd.DataFrame(
        {"mol_id": [1, 1, 1, 2], "cos|baseline": [0.5] * 4, "cos|probe": [0.6, 0.6, 0.6, 0.3]}
    )
    result = bootstrap_delta(frame, "probe", "baseline", 1000, 3)
    assert result["mean"] == pytest.approx(0.025)
    assert result["molecule_mean"] == pytest.approx(-0.05)
    assert result["low"] == pytest.approx(-0.2)
    assert result["high"] == pytest.approx(0.1)
    assert result["molecules"] == 2 and result["spectra"] == 4


def test_report_locks_on_tuning_only_and_rejects_partial_evaluation(tmp_path):
    import json

    cfg = {**config(), "bootstrap_repetitions": 100, "screen_delta": 0.005}
    frame = pd.DataFrame(
        {
            "mol_id": np.arange(8),
            "partition": ["tune"] * 4 + ["confirm"] * 4,
            "novelty": [0.3] * 8,
            "independent_floor": [False, False, False, False, True, True, False, False],
        },
        index=np.arange(100, 108),
    )
    arms = ["baseline"]
    arms += [
        f"{kind}:{alpha}"
        for kind in (
            "encoder",
            "handmade",
            "proposal",
            "cut_count",
            "temperature",
            "encoder_shuffle",
            "handmade_shuffle",
        )
        for alpha in (0.0, 0.1)
    ]
    for arm in arms:
        frame[f"cos|{arm}"] = 0.8
        frame[f"hit|{arm}"] = 1.0
    frame.loc[100:103, "cos|encoder:0.1"] = 0.81
    frame.loc[104:107, "cos|encoder:0.1"] = 0.79
    # A spectacular confirmation effect must never select the weaker tuning arm.
    frame.loc[104:107, "cos|handmade:0.1"] = 0.99
    frame.to_pickle(tmp_path / "scores.pkl")
    frame[["mol_id", "partition"]].to_pickle(tmp_path / "validation_manifest.pkl")
    report(cfg, tmp_path)
    first = json.loads((tmp_path / "result.json").read_text())
    assert first["selected_probe"] == "encoder:0.1"
    assert not first["screen_pass"]
    assert first["comparisons"]["confirmation"]["vs_baseline"]["mean"] == pytest.approx(-0.01)
    frame.loc[104:107, "cos|encoder:0.1"] = 0.82
    frame.to_pickle(tmp_path / "scores.pkl")
    report(cfg, tmp_path)
    second = json.loads((tmp_path / "result.json").read_text())
    assert second["selected_probe"] == first["selected_probe"]
    assert second["screen_pass"]
    frame.iloc[:-1].to_pickle(tmp_path / "scores.pkl")
    with pytest.raises(ValueError, match="complete validation"):
        report(cfg, tmp_path)
