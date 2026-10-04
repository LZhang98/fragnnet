"""Count distinct fragment contributors per EI peak from an annotated MASSter checkpoint.

The output counts fragment IDs once per integer m/z, including all H and isotope rows.
It also records each H shift's share of the predicted spectrum. No target is used in the
model forward pass; measured peaks are read only for the target-peak summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Subset

from fragnnet.runner import load_config
from fragnnet.utils.script_utils import init_model_from_ckpt
from fragnnet.utils.setup_utils import get_dataset_cls


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def _stein_similarity(
    pred_mzs: np.ndarray,
    pred_intensities: np.ndarray,
    target_mzs: np.ndarray,
    target_intensities: np.ndarray,
) -> float:
    """Compute RASSP's Stein weighted cosine: mass**3 * intensity**0.6."""
    pred = pred_mzs.astype(np.float64) ** 3 * pred_intensities.astype(np.float64) ** 0.6
    target = target_mzs.astype(np.float64) ** 3 * target_intensities.astype(np.float64) ** 0.6
    denominator = np.linalg.norm(pred) * np.linalg.norm(target)
    if denominator == 0:
        return 0.0
    target_by_mz = dict(zip(target_mzs, target))
    matched = np.array([target_by_mz.get(mz, 0.0) for mz in pred_mzs])
    return float(np.dot(pred, matched) / denominator)


def _spectrum_rows(
    spec_id: int,
    mzs: np.ndarray,
    intensities: np.ndarray,
    fragments: np.ndarray,
    shifts: np.ndarray,
    target_mzs: np.ndarray,
    target_intensities: np.ndarray,
    include_peak_rows: bool = True,
) -> tuple[list[dict], dict]:
    if not (len(mzs) == len(intensities) == len(fragments) == len(shifts)):
        raise ValueError("prediction rows and annotation must align")
    if len(mzs) == 0 or not np.allclose(mzs, np.rint(mzs)):
        raise ValueError("EI prediction requires nonempty integer m/z rows")
    intensity_sum = float(intensities.sum())
    if not np.isfinite(intensity_sum) or intensity_sum < 0:
        raise ValueError(f"invalid predicted intensity sum for spectrum {spec_id}: {intensity_sum}")
    if intensity_sum == 0:
        info = {"spec_id": spec_id, "n_pred_peaks": 0, "pred_intensity_sum": 0.0}
        info.update(dict.fromkeys(("cosine_full", "cosine_no_h4", "cosine_no_h34", "stein_full"), 0.0))
        info.update({f"h_{shift:+d}": 0.0 for shift in range(-4, 5)})
        return [], info
    intensities = intensities / intensity_sum
    target_mzs = np.rint(target_mzs).astype(int)
    target_weights = target_intensities / target_intensities.sum()
    target_by_mz = dict(zip(target_mzs, target_weights))
    unique_mzs, peak_index = np.unique(mzs.astype(int), return_inverse=True)
    peak_weights = np.bincount(peak_index, weights=intensities)
    peak_rows = []
    base = float(peak_weights.max())
    if include_peak_rows:
        for peak, mz in enumerate(unique_mzs):
            in_peak = peak_index == peak
            frag, inverse = np.unique(fragments[in_peak], return_inverse=True)
            frag_weights = np.bincount(inverse, weights=intensities[in_peak])
            shares = frag_weights / peak_weights[peak]
            peak_rows.append(
                {
                    "spec_id": spec_id,
                    "mz": int(mz),
                    "pred_share": float(peak_weights[peak]),
                    "relative_to_base": float(peak_weights[peak] / base),
                    "target_share": float(target_by_mz.get(int(mz), 0.0)),
                    "n_fragments": len(frag),
                    "n_fragments_1pct": int(np.sum(shares >= 0.01)),
                    "n_fragments_10pct": int(np.sum(shares >= 0.1)),
                    "effective_fragments": float(1.0 / np.sum(shares**2)),
                    "top_fragment_share": float(shares.max()),
                    "n_pathway_rows": int(np.sum(in_peak)),
                }
            )
    shift_weights = np.bincount(shifts + 4, weights=intensities, minlength=9)
    matched_target = np.array([target_by_mz.get(int(mz), 0.0) for mz in unique_mzs])
    target_norm = np.linalg.norm(target_weights)

    def cosine_without(keep: np.ndarray) -> float:
        peak_mass = np.bincount(peak_index[keep], weights=intensities[keep], minlength=len(unique_mzs))
        denominator = np.linalg.norm(peak_mass) * target_norm
        return float(np.dot(peak_mass, matched_target) / denominator) if denominator else 0.0

    all_rows = np.ones(len(shifts), dtype=bool)
    no_h4 = np.abs(shifts) < 4
    no_outer = np.abs(shifts) < 3
    info = {
        "spec_id": spec_id,
        "n_pred_peaks": len(unique_mzs),
        "pred_intensity_sum": intensity_sum,
        "cosine_full": cosine_without(all_rows),
        "stein_full": _stein_similarity(
            unique_mzs, peak_weights, target_mzs, target_weights
        ),
        "cosine_no_h4": cosine_without(no_h4),
        "cosine_no_h34": cosine_without(no_outer),
    }
    info.update({f"h_{shift:+d}": float(shift_weights[shift + 4]) for shift in range(-4, 5)})
    return peak_rows, info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-molecules", type=int, default=512)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--all", action="store_true", help="use every spectrum in the split")
    parser.add_argument("--scores-only", action="store_true", help="skip peak-level output")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    config = load_config("config/template.yml", "config/masster/ei/e1_nominal_natural_isotopes.yml", [])
    config["num_workers"] = 0
    config["eval_batch_size"] = args.batch_size
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    # Restrict the source tables before dataset preprocessing, which otherwise builds all
    # 51,700 validation graphs even for a small diagnostic sample.
    all_specs = pd.read_pickle(config["spec_fp"])
    split_ids = pd.read_csv(Path(config["split_dp"]) / f"{args.split}_ids.csv")["spec_id"]
    selected = all_specs[all_specs.spec_id.isin(split_ids)]
    if not args.all:
        selected = selected.drop_duplicates("mol_id").sample(
            args.n_molecules, random_state=args.seed
        )
    selected = selected.sort_values("spec_id")
    all_molecules = pd.read_pickle(config["mol_fp"])
    selected_molecules = all_molecules[all_molecules.mol_id.isin(selected.mol_id)]
    subset_dir = args.output / "input_subset"
    subset_dir.mkdir()
    selected.to_pickle(subset_dir / "spec_df.pkl")
    selected_molecules.to_pickle(subset_dir / "mol_df.pkl")
    del all_specs, all_molecules
    config["spec_fp"] = str(subset_dir / "spec_df.pkl")
    config["mol_fp"] = str(subset_dir / "mol_df.pkl")
    model = init_model_from_ckpt(str(args.checkpoint), config, device)
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **config)
    specs = dataset.spec_df.reset_index(drop=True)
    if set(specs.spec_id) != set(selected.spec_id):
        raise RuntimeError("sampled EI spectrum IDs changed in dataset preprocessing")
    positions = np.arange(len(specs))
    loader = DataLoader(
        Subset(dataset, positions.tolist()), batch_size=args.batch_size,
        shuffle=False, collate_fn=dataset.get_collate_fn(), num_workers=0,
    )
    peaks, spectra = [], []
    with th.inference_mode():
        for batch_number, batch in enumerate(loader):
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            result = model.inference_step(
                {**batch, "return_explainability_tensors": True}, metrics=None
            )
            true_mzs, true_ints, true_batches = model.preproc_spec(
                batch["spec_mzs"], batch["spec_ints"], batch["spec_batch_idxs"],
                train=False, pred=False, batch_size=int(batch["batch_size"]),
            )
            arrays = {
                name: result[name].detach().cpu().numpy()
                for name in (
                    "pred_mzs", "pred_ints", "pred_batch_idxs",
                    "pred_pathway_fragment_idxs", "pred_pathway_h_shifts",
                )
            }
            target = [x.cpu().numpy() for x in (true_mzs, true_ints, true_batches)]
            batch_size = int(batch["batch_size"])
            for view in range(batch_size):
                spec_id = int(specs.iloc[positions[batch_number * args.batch_size + view]].spec_id)
                pred_mask = arrays["pred_batch_idxs"] == view
                true_mask = target[2] == view
                rows, info = _spectrum_rows(
                    spec_id,
                    arrays["pred_mzs"][pred_mask],
                    arrays["pred_ints"][pred_mask],
                    arrays["pred_pathway_fragment_idxs"][pred_mask],
                    arrays["pred_pathway_h_shifts"][pred_mask],
                    target[0][true_mask], target[1][true_mask],
                    include_peak_rows=not args.scores_only,
                )
                if not args.scores_only:
                    peaks.extend(rows)
                spectra.append(info)
            if (batch_number + 1) % 16 == 0:
                print(f"completed {len(spectra)} spectra", flush=True)
    spec_frame = pd.DataFrame(spectra)
    spec_frame.to_csv(args.output / "spectrum_rows.csv", index=False)
    summary = {"spectra": len(spec_frame)}
    summary["zero_intensity_spectra"] = int((spec_frame.pred_intensity_sum == 0).sum())
    summary["nonunit_intensity_spectra"] = int(
        (~np.isclose(spec_frame.pred_intensity_sum, 1.0, atol=1e-5)).sum()
    )
    if not args.scores_only:
        peak_frame = pd.DataFrame(peaks)
        peak_frame.to_csv(args.output / "peak_rows.csv.gz", index=False, compression="gzip")
        summary["peaks"] = len(peak_frame)
        for cut in (0.0, 0.001, 0.01):
            kept = peak_frame[peak_frame.relative_to_base >= cut]
            summary[f"pred_peaks_ge_{cut}"] = len(kept)
            summary[f"fragments_mean_ge_{cut}"] = float(kept.n_fragments.mean())
            summary[f"fragments_median_ge_{cut}"] = float(kept.n_fragments.median())
            summary[f"fragments_p90_ge_{cut}"] = float(kept.n_fragments.quantile(0.9))
            summary[f"fragments_1pct_mean_ge_{cut}"] = float(kept.n_fragments_1pct.mean())
            summary[f"effective_fragments_mean_ge_{cut}"] = float(kept.effective_fragments.mean())
        target = peak_frame[peak_frame.target_share > 0]
        summary["target_matched_peaks"] = len(target)
        summary["target_weighted_fragments_per_peak"] = float(
            np.average(target.n_fragments, weights=target.target_share)
        )
        summary["pred_weighted_fragments_per_peak"] = float(
            np.average(peak_frame.n_fragments, weights=peak_frame.pred_share)
        )
    summary["mean_h_shift_shares"] = {
        str(shift): float(spec_frame[f"h_{shift:+d}"].mean()) for shift in range(-4, 5)
    }
    summary["mean_outer_h_share"] = float(
        spec_frame[["h_-4", "h_-3", "h_+3", "h_+4"]].sum(axis=1).mean()
    )
    for column in ("cosine_full", "cosine_no_h4", "cosine_no_h34", "stein_full"):
        summary[f"mean_{column}"] = float(spec_frame[column].mean())
    summary["fraction_h4_helps_cosine"] = float(
        (spec_frame.cosine_full > spec_frame.cosine_no_h4).mean()
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_sha256": _file_sha256(args.checkpoint),
                "source_config": "config/masster/ei/e1_nominal_natural_isotopes.yml",
                "n_molecules": int(selected.mol_id.nunique()),
                "seed": args.seed,
                "split": args.split,
                "all_spectra": args.all,
                "scores_only": args.scores_only,
                "device": str(device),
                "sample_spec_ids": [int(specs.iloc[position].spec_id) for position in positions],
            },
            indent=2,
        )
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
