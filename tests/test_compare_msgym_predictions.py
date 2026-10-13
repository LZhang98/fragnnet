import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval" / "compare_msgym_predictions.py"
SPEC = importlib.util.spec_from_file_location("compare_msgym_predictions", SCRIPT_PATH)
CMP = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(CMP)


@pytest.fixture
def export_h5(tmp_path):
    """Two spectra in the run_save_inference_eval.py layout, base peak 1000."""
    fp = tmp_path / "predictions_test.hdf5"
    with h5py.File(fp, "w") as f:
        f["spec_id"] = np.array([7, 9])
        f["start_points"] = np.array([0, 3])
        f["end_points"] = np.array([3, 5])
        f["pred_mzs"] = np.array([100.0, 101.0, 102.0, 200.0, 201.0])
        f["pred_ints"] = np.array([1000.0, 1.0, 0.5, 1000.0, 2.0], dtype=np.float32)
        f["cos_sim_0.01"] = np.array([0.8, 0.6])
    return fp


def test_export_peaks_unfiltered_and_logged(export_h5):
    peaks, logged = CMP.load_export_peaks(str(export_h5), min_peak_int=0.0)
    assert peaks[7].shape == (3, 2)
    assert peaks[9][:, 0].tolist() == [200.0, 201.0]
    assert logged == {"cos_sim_0.01": {7: pytest.approx(0.8), 9: pytest.approx(0.6)}}


def test_export_peak_filter_is_strict_like_the_exporter(export_h5):
    peaks, _ = CMP.load_export_peaks(str(export_h5), min_peak_int=1.0)
    # 1.0 itself is dropped, as batch_filter_by_peak_intensity drops it.
    assert peaks[7][:, 0].tolist() == [100.0]
    assert peaks[9][:, 0].tolist() == [200.0, 201.0]


def test_glacier_peaks_are_squared_and_keyed_by_spec_id(tmp_path):
    fp = tmp_path / "preds.hdf5"
    with h5py.File(fp, "w") as f:
        g = f.create_group("pred_MassSpecGymID0000001/ikey AAA/collision 30")
        g["f"] = np.array([[150.0, 0.5], [151.0, 0.0], [152.0, 0.25]], dtype=np.float32)
        f.create_group("__predspec_manifest__")
    peaks = CMP.load_glacier_peaks(str(fp), {"MassSpecGymID0000001": 42})
    assert list(peaks) == [42]
    np.testing.assert_allclose(peaks[42], [[150.0, 0.25], [152.0, 0.0625]])


def test_glacier_peaks_reject_multiple_collision_energies(tmp_path):
    fp = tmp_path / "preds.hdf5"
    with h5py.File(fp, "w") as f:
        for ce in (20, 40):
            f.create_group(f"pred_X/ikey AAA/collision {ce}")["f"] = np.ones((1, 2), np.float32)
    with pytest.raises(ValueError, match="one collision energy"):
        CMP.load_glacier_peaks(str(fp), {"X": 1})


def _scores(offset: float) -> pd.DataFrame:
    base = np.array([0.5, 0.6, 0.7, 0.8])
    return pd.DataFrame(
        {"spec_id": [1, 2, 3, 4], "mol_id": [10, 10, 11, 12], **dict.fromkeys(CMP.METRICS, base + offset)}
    )


def test_paired_diff_ci_of_a_constant_difference_is_that_constant():
    rng = np.random.default_rng(0)
    mean, lo, hi = CMP.paired_diff_ci(pd.Series([0.1] * 6), pd.Series([1, 1, 2, 3, 3, 3]), rng)
    assert (mean, lo, hi) == pytest.approx((0.1, 0.1, 0.1))


def test_build_table_means_and_differences():
    table = CMP.build_table({"ours": _scores(0.05), "GLACIER": _scores(0.0)}, reference="GLACIER")
    assert table["metric"].tolist() == CMP.METRICS
    row = table.set_index("metric").loc["cos_sim_0.01"]
    assert row["ours"] == pytest.approx(0.70)
    assert row["GLACIER"] == pytest.approx(0.65)
    assert row["ours - GLACIER"] == pytest.approx(0.05)
    assert row["ours - GLACIER ci95_lo"] == pytest.approx(0.05)
    assert row["n"] == 4
    assert "GLACIER - GLACIER" not in table.columns


def test_build_table_rejects_mismatched_spectra():
    other = _scores(0.0).iloc[:3]
    with pytest.raises(ValueError, match="different spectra"):
        CMP.build_table({"ours": other, "GLACIER": _scores(0.0)}, reference="GLACIER")
