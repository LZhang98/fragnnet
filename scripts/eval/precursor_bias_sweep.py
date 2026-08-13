"""Evaluate a post-hoc precursor log-intensity correction at low collision energy.

The scalar correction is selected on a molecule-disjoint calibration half of a
validation prediction export and reported on the other half. Only FT spectra in
the requested NCE interval participate. A positive bias subtracts from every
predicted peak matching the precursor mass before intensities are renormalized.

Example:
    python scripts/eval/precursor_bias_sweep.py \
        --pred_fp output/inference_eval/9t63zils/predictions_val.hdf5 \
        --spec_df_fp data/proc/nist20/spec_df.pkl \
        --out_fp output/diagnostics/precursor_bias_9t63zils_val.csv
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FT_REL_TOL = 1.0e-5
FT_MIN_MZ = 200.0


@dataclass(frozen=True)
class Spectrum:
    """One filtered observed/predicted spectrum used by the sweep."""

    mol_id: int
    true_mzs: np.ndarray
    true_ints: np.ndarray
    pred_mzs: np.ndarray
    pred_ints: np.ndarray
    prec_mz: float


def _matches(mzs: np.ndarray, mass: float) -> np.ndarray:
    """Return the FT instrument-tolerance match mask for one reference mass."""
    return np.abs(mzs - mass) <= FT_REL_TOL * max(mass, FT_MIN_MZ)


def _cosine_hungarian(
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    pred_mzs: np.ndarray,
    pred_ints: np.ndarray,
) -> float:
    """Compute the repository's one-to-one tolerance-matched cosine score."""
    true_norm = np.linalg.norm(true_ints)
    pred_norm = np.linalg.norm(pred_ints)
    if true_norm <= 0.0 or pred_norm <= 0.0:
        return 0.0
    true_scaled = true_ints / true_norm
    pred_scaled = pred_ints / pred_norm
    tolerance = FT_REL_TOL * np.maximum(true_mzs, FT_MIN_MZ)
    match = np.abs(true_mzs[:, None] - pred_mzs[None, :]) < tolerance[:, None]
    if not match.any():
        return 0.0
    score = match * (true_scaled[:, None] * pred_scaled[None, :])
    rows, cols = linear_sum_assignment(score, maximize=True)
    return float(score[rows, cols].sum())


def evaluate_spectrum(spectrum: Spectrum, bias: float) -> dict[str, float | bool]:
    """Apply one precursor bias and calculate the requested low-CE metrics."""
    adjusted = spectrum.pred_ints.copy()
    pred_prec_mask = _matches(spectrum.pred_mzs, spectrum.prec_mz)
    adjusted[pred_prec_mask] *= np.exp(-bias)

    true_base_idx = int(np.argmax(spectrum.true_ints))
    pred_base_idx = int(np.argmax(adjusted))
    true_base_mz = float(spectrum.true_mzs[true_base_idx])
    pred_base_mz = float(spectrum.pred_mzs[pred_base_idx])
    true_base_is_prec = bool(_matches(np.array([true_base_mz]), spectrum.prec_mz)[0])
    pred_base_is_prec = bool(pred_prec_mask[pred_base_idx])
    base_hit = bool(_matches(np.array([pred_base_mz]), true_base_mz)[0])

    true_mean_mz = float(np.average(spectrum.true_mzs, weights=spectrum.true_ints))
    pred_mean_mz = float(np.average(spectrum.pred_mzs, weights=adjusted))
    return {
        "true_base_is_prec": true_base_is_prec,
        "pred_base_is_prec": pred_base_is_prec,
        "base_hit": base_hit,
        "mean_mz_error": pred_mean_mz - true_mean_mz,
        "cos_hun": _cosine_hungarian(
            spectrum.true_mzs,
            spectrum.true_ints,
            spectrum.pred_mzs,
            adjusted,
        ),
    }


def _split_for_molecule(mol_id: int, seed: int) -> str:
    """Assign all spectra of a molecule to a deterministic split."""
    digest = hashlib.blake2b(f"{seed}:{mol_id}".encode(), digest_size=8).digest()
    return "calibration" if int.from_bytes(digest, "little") % 2 == 0 else "evaluation"


def load_spectra(
    pred_fp: Path,
    spec_df_fp: Path,
    nce_min: float,
    nce_max: float,
) -> tuple[list[Spectrum], np.ndarray]:
    """Load FT spectra in the selected NCE interval and reproduce eval filtering."""
    with h5py.File(pred_fp, "r") as h5:
        meta = pd.DataFrame(
            {
                "spec_id": h5["spec_id"][:],
                "mol_id": h5["mol_id"][:],
                "start": h5["start_points"][:],
                "end": h5["end_points"][:],
                "stored_cos": h5["cos_hun_inst_tol"][:],
            }
        )
        pred_mzs = h5["pred_mzs"][:]
        pred_ints = h5["pred_ints"][:]

    spec_df = pd.read_pickle(spec_df_fp).set_index("spec_id")
    gt = spec_df.loc[
        meta["spec_id"].to_numpy(),
        ["peaks", "prec_mz", "inst_type", "nce"],
    ].reset_index(drop=True)
    selected = (
        gt["inst_type"].eq("FT")
        & gt["nce"].ge(nce_min)
        & gt["nce"].lt(nce_max)
    ).to_numpy()

    spectra: list[Spectrum] = []
    stored_scores: list[float] = []
    for row_idx in np.flatnonzero(selected):
        peaks = np.asarray(gt.iloc[row_idx]["peaks"], dtype=float)
        prec_mz = float(gt.iloc[row_idx]["prec_mz"])
        keep = peaks[:, 0] <= prec_mz + FT_REL_TOL * max(prec_mz, FT_MIN_MZ)
        peaks = peaks[keep]
        start = int(meta.iloc[row_idx]["start"])
        end = int(meta.iloc[row_idx]["end"])
        if len(peaks) == 0 or start == end:
            continue
        spectra.append(
            Spectrum(
                mol_id=int(meta.iloc[row_idx]["mol_id"]),
                true_mzs=peaks[:, 0],
                true_ints=peaks[:, 1],
                pred_mzs=pred_mzs[start:end].astype(float),
                pred_ints=pred_ints[start:end].astype(float),
                prec_mz=prec_mz,
            )
        )
        stored_scores.append(float(meta.iloc[row_idx]["stored_cos"]))
    return spectra, np.asarray(stored_scores)


def summarize(rows: pd.DataFrame) -> dict[str, float]:
    """Aggregate the experiment's primary and safety metrics."""
    true_prec = rows["true_base_is_prec"].to_numpy(dtype=bool)
    return {
        "n": float(len(rows)),
        "precursor_winner_fraction": float(true_prec.mean()),
        "predicted_precursor_fraction": float(rows["pred_base_is_prec"].mean()),
        "precursor_calibration_gap": float(
            rows["pred_base_is_prec"].mean() - true_prec.mean()
        ),
        "precursor_winner_hit": float(rows.loc[true_prec, "base_hit"].mean()),
        "nonprecursor_winner_hit": float(rows.loc[~true_prec, "base_hit"].mean()),
        "base_hit": float(rows["base_hit"].mean()),
        "mean_mz_error": float(rows["mean_mz_error"].mean()),
        "cos_hun": float(rows["cos_hun"].mean()),
    }


def run_sweep(spectra: list[Spectrum], biases: np.ndarray, seed: int) -> pd.DataFrame:
    """Evaluate every bias on deterministic molecule-disjoint halves."""
    split = np.asarray([_split_for_molecule(spectrum.mol_id, seed) for spectrum in spectra])
    output = []
    for bias in biases:
        metrics = pd.DataFrame([evaluate_spectrum(spectrum, float(bias)) for spectrum in spectra])
        for split_name in ("calibration", "evaluation"):
            summary = summarize(metrics.loc[split == split_name])
            output.append({"bias": float(bias), "split": split_name, **summary})
        log.info("Evaluated precursor bias %.3f", bias)
    return pd.DataFrame(output)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred_fp", type=Path, required=True)
    parser.add_argument("--spec_df_fp", type=Path, required=True)
    parser.add_argument("--out_fp", type=Path, required=True)
    parser.add_argument("--nce_min", type=float, default=0.0)
    parser.add_argument("--nce_max", type=float, default=20.0)
    parser.add_argument("--bias_min", type=float, default=0.0)
    parser.add_argument("--bias_max", type=float, default=2.0)
    parser.add_argument("--bias_steps", type=int, default=41)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    """Run the sweep, select by calibration cosine, and write all results."""
    args = parse_args()
    spectra, stored_scores = load_spectra(
        args.pred_fp,
        args.spec_df_fp,
        args.nce_min,
        args.nce_max,
    )
    if not spectra:
        raise ValueError("No eligible spectra found")
    biases = np.linspace(args.bias_min, args.bias_max, args.bias_steps)
    results = run_sweep(spectra, biases, args.seed)
    zero = results[np.isclose(results["bias"], 0.0)]
    reconstructed = float(
        np.average(zero["cos_hun"], weights=zero["n"])
    )
    stored = float(stored_scores.mean())
    log.info(
        "Zero-bias reconstructed cosine %.6f; stored pre-export cosine %.6f; delta %.6f",
        reconstructed,
        stored,
        reconstructed - stored,
    )
    calibration = results[results["split"] == "calibration"]
    selected_bias = float(calibration.loc[calibration["cos_hun"].idxmax(), "bias"])
    results["selected_by_calibration_cosine"] = np.isclose(results["bias"], selected_bias)
    results["zero_bias_cosine_delta_from_stored"] = reconstructed - stored
    args.out_fp.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_fp, index=False)
    selected = results[
        results["selected_by_calibration_cosine"] & results["split"].eq("evaluation")
    ].iloc[0]
    log.info("Selected bias %.3f; evaluation metrics:\n%s", selected_bias, selected.to_string())
    log.info("Saved %s", args.out_fp)


if __name__ == "__main__":
    main()
