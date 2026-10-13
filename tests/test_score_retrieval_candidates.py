import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
CLI = REPO / "scripts/eval/score_retrieval_candidates.py"


@pytest.fixture
def retrieval_inputs(tmp_path):
    """One ORB query (two peaks) and two candidates in a MASSter-style export."""
    pd.DataFrame(
        {
            "spec_id": [7],
            "peaks": [[[100.0, 9.0], [200.0, 1.0]]],
            "prec_mz": [250.0],
            "inst_type": ["ORB"],
        }
    ).to_pickle(tmp_path / "query.pkl")
    pd.DataFrame(
        {"cand_uid": [0, 1], "query_spec_id": [7, 7], "is_truth": [True, False]}
    ).to_csv(tmp_path / "candidates.csv", index=False)
    # candidate 0 predicts both query peaks (200.003 is outside 10 ppm, inside 20 ppm);
    # candidate 1 predicts one unrelated peak
    mzs = np.array([100.0, 200.003, 150.0])
    ints = np.array([0.8, 0.2, 1.0])
    with h5py.File(tmp_path / "pred.hdf5", "w") as f:
        f["spec_id"] = np.array([0, 1])
        f["start_points"] = np.array([0, 2])
        f["end_points"] = np.array([2, 3])
        f["pred_mzs"] = mzs
        f["pred_ints"] = ints
        f["cos_sim_0.01"] = np.array([0.9, 0.1])
    return tmp_path


def _score(d, *extra):
    out = d / "scores.pkl"
    subprocess.run(
        [sys.executable, str(CLI), "--candidates", str(d / "candidates.csv"),
         "--query_spec_df", str(d / "query.pkl"), "--pred_format", "masster",
         "--pred_fp", str(d / "pred.hdf5"), "--out", str(out), "--device", "cpu", *extra],
        check=True, cwd=REPO,
    )  # fmt: skip
    return pd.read_pickle(out).set_index("cand_uid")


def test_loglik_cal_matches_hand_computation(retrieval_inputs):
    s = _score(retrieval_inputs, "--scores", "loglik", "loglik_cal",
               "--inmodel_scores", "cos_sim_0.01")  # fmt: skip
    # weights y^0.5 renormalized: [3, 1] / 4; prediction p^0.75 renormalized
    w = np.array([0.75, 0.25])
    p = np.array([0.8, 0.2]) ** 0.75
    p = p / p.sum()
    # tolerance 2 x 10 ppm x max(mz, 200): 200.003 now matches 200.0
    assert s.loc[0, "loglik_cal"] == pytest.approx((w * np.log(p)).sum())
    # candidate 1 matches nothing: every peak takes the 1e-4 floor
    assert s.loc[1, "loglik_cal"] == pytest.approx(np.log(1e-4))
    # the frozen rule at 10 ppm misses 200.003 and floors it at 1e-5
    assert s.loc[0, "loglik"] == pytest.approx(0.9 * np.log(0.8) + 0.1 * np.log(1e-5))
    assert s.loc[0, "cos_sim_0.01"] == pytest.approx(0.9)
    assert s.loc[1, "cos_sim_0.01"] == pytest.approx(0.1)


def test_loglik_cal_reduces_to_loglik_at_identity_settings(retrieval_inputs):
    s = _score(retrieval_inputs, "--scores", "loglik", "loglik_cal", "--cal_gamma", "1",
               "--cal_tol_mult", "1", "--cal_eps", "1e-5", "--cal_alpha", "1")  # fmt: skip
    np.testing.assert_allclose(s["loglik_cal"], s["loglik"])
