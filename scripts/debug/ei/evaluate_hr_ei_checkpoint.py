"""Compare an EI checkpoint's exact-mass channels with audited HR-EI spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from fragnnet.config import load_config
from fragnnet.inference.checkpoint import init_model_from_ckpt
from fragnnet.utils.migrations import get_pl_hparams
from fragnnet.utils.misc_utils import deep_update
from fragnnet.utils.mol_utils import compute_mol_df_props
from fragnnet.utils.setup_utils import get_dataset_cls
from scripts.debug.ei.audit_hr_ei_raw import read_records

SOURCES = {
    "hrei_msdb": "hrei_msdb/all_spectra.msp",
    "recetox_exposome": "recetox_exposome/recetox_exposome_gc-ei-ms_20201028.msp",
    "recetox_metabolome": "recetox_metabolome/rcx_gc-orbitrap_metabolites_20210817.msp",
}


def aggregate(mzs: np.ndarray, intensities: np.ndarray, decimals: int = 4) -> tuple:
    """Sum intensities at equal rounded masses, preserving the intensity-weighted mass."""
    _, index = np.unique(np.round(mzs, decimals), return_inverse=True)
    weights = np.bincount(index, weights=intensities)
    masses = np.bincount(index, weights=mzs * intensities) / np.maximum(weights, 1e-300)
    keep = weights > 0
    return masses[keep], weights[keep]


def score(pred_mz, pred_int, true_mz, true_int, ppm: float, floor: float) -> dict:
    """Compute one-to-one maximum-product cosine and intensity-weighted support recall.

    A pair matches when its mass error is at most max(ppm * target m/z / 1e6, floor).
    Cosine uses linear intensities and full-spectrum L2 norms, including unmatched peaks.
    """
    tolerance = np.maximum(true_mz * ppm * 1e-6, floor)
    # Weighted aggregation can leave an integer at e.g. 69.00000000000001.
    mask = abs(true_mz[:, None] - pred_mz[None, :]) <= tolerance[:, None] + 1e-10
    products = true_int[:, None] * pred_int[None, :] * mask
    # Restrict assignment to potentially matching rows/columns without changing full norms.
    subset = products[mask.any(axis=1)][:, mask.any(axis=0)]
    numerator = 0.0
    if subset.size:
        a, b = linear_sum_assignment(subset, maximize=True)
        numerator = subset[a, b].sum()
    denom = np.linalg.norm(true_int) * np.linalg.norm(pred_int)
    return {
        "cosine": float(numerator / denom) if denom else 0.0,
        "target_intensity_recall": float(true_int[mask.any(axis=1)].sum() / true_int.sum()),
        "target_l2_coverage": float(
            np.square(true_int[mask.any(axis=1)]).sum() / np.square(true_int).sum()
        ),
    }


def main() -> None:
    """Prepare isolated inputs, run frozen weights, and save per-spectrum matched scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saved_ckpts/e1_nominal_natural_isotopes_jeljdikj/ckpt/model-epoch=037.ckpt"),
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("output/hr_ei_raw_audit_2026_10_17/verified/records.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--isotope-envelope", choices=("none", "natural"), default="none")
    parser.add_argument("--rescore-from", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if args.rescore_from is not None:
        rescore_cached(args.rescore_from, args.output)
        return
    random.seed(20261017)
    np.random.seed(20261017)  # noqa: NPY002 - seed legacy library consumers too
    torch.manual_seed(20261017)
    torch.set_num_threads(4)
    RDLogger.DisableLog("rdApp.*")
    started = time.time()

    # 1. Load the trained configuration; normalize retired metadata labels before migration.
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    old = ckpt["hyper_parameters"]["spec_params"]["merge_tolerance_by_inst"]
    ckpt["hyper_parameters"]["spec_params"]["merge_tolerance_by_inst"] = {
        {"OB": "ORB", "OB_F": "ORB_F"}.get(k, k): v for k, v in old.items()
    }
    ce_keys = ckpt["hyper_parameters"]["ce_context_ce_key_by_inst"]
    ckpt["hyper_parameters"]["ce_context_ce_key_by_inst"] = {
        {"OB": "ORB", "OB_F": "ORB_F"}.get(k, k): v for k, v in ce_keys.items()
    }
    saved_config = get_pl_hparams(ckpt)
    template = load_config("config/template.yml", None, [], model_type="masster")
    config = deep_update(template, saved_config)
    del ckpt
    config.update(
        masster_mass_mode="exact",
        masster_isotope_envelope=args.isotope_envelope,
        masster_candidate_backend="cpu",
        compile=False,
        num_workers=0,
        pl_enable_progress_bar=False,
    )
    # The actual targets are scored externally, with no isotope removal or target transform.
    config["spec_params"].update(preprocess=False, merge=False)
    config["mol_params"]["preprocess"] = True
    config["mol_fp"] = str(args.output / "mol_df.pkl")
    config["spec_fp"] = str(args.output / "spec_df.pkl")
    (args.output / "resolved_config.json").write_text(json.dumps(config, indent=2, default=str))

    # 2. Include all mapping-consistent records, regardless of nominal-training overlap.
    audit = pd.read_csv(args.audit)
    selected = audit[audit.mapping_consistent].copy()
    if args.limit:
        selected = selected.groupby("source", group_keys=False).head(args.limit)
    selected = selected.reset_index(drop=True)
    selected["spec_id"] = np.arange(len(selected))
    selected["mol_id"] = pd.factorize(selected.smiles)[0]
    selected.to_csv(args.output / "selected_records.csv", index=False)
    mols = selected[["mol_id", "smiles"]].drop_duplicates("mol_id").copy()
    mols["mol"] = mols.smiles.map(Chem.MolFromSmiles)
    compute_mol_df_props(mols, scaffold=False, inchi=False, use_tqdm=False)
    mols.to_pickle(args.output / "mol_df.pkl")
    raw = {
        source: read_records(Path("data/raw/hr_ei_ms") / path) for source, path in SOURCES.items()
    }
    spec_rows, targets = [], {}
    for row in selected.itertuples():
        peaks = np.array(raw[row.source][row.record][1])
        targets[row.spec_id] = peaks
        spec_rows.append(
            dict(  # noqa: C408 - parallel named fields make the input schema visible
                spec_id=row.spec_id,
                mol_id=row.mol_id,
                group_id=row.spec_id,
                dset="hr_ei",
                dset_spec_id=str(row.spec_id),
                prec_type="[M]+",
                inst_type="Other",
                frag_mode="EI",
                spec_type="MS1",
                ion_mode="P",
                prec_mz=float(mols.set_index("mol_id").loc[row.mol_id, "exact_mw"]),
                nce=np.nan,
                ace=np.nan,
                peaks=peaks.tolist(),
            )
        )
    pd.DataFrame(spec_rows).to_pickle(args.output / "spec_df.pkl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        config["masster_candidate_backend"] = "cuda"
    model = init_model_from_ckpt(str(args.checkpoint), config, device)
    dataset = get_dataset_cls(config["model_type"])(split="predict_only", **config)
    if set(dataset.spec_df.spec_id) != set(selected.spec_id):
        raise RuntimeError("Dataset filtered audit records; investigate before scoring")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
    )

    # 3. Score HR tolerances and a rounded-mass control using the same intensities.
    scores, predictions = [], {}
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            batch = model.transfer_batch_to_device(batch, device, 0)
            result = model.inference_step(batch, metrics=None)
            arrays = {
                k: result[k].cpu().numpy() for k in ["pred_mzs", "pred_ints", "pred_batch_idxs"]
            }
            for local in range(int(batch["batch_size"])):
                spec_id = int(dataset.spec_df.iloc[batch_index * args.batch_size + local].spec_id)
                mask = arrays["pred_batch_idxs"] == local
                mz, weights = aggregate(
                    arrays["pred_mzs"][mask].astype(float), arrays["pred_ints"][mask].astype(float)
                )
                target = targets[spec_id]
                # Respect the acquired mass window without target-peak-dependent filtering.
                source = selected.iloc[spec_id].source
                lower = 40.0 if source == "hrei_msdb" else 0.0
                upper = 500.0 if source == "hrei_msdb" else 1500.0
                keep = (mz >= lower) & (mz <= upper)
                mz, weights = mz[keep], weights[keep]
                tm, ti = aggregate(target[:, 0], target[:, 1])
                predictions[f"{spec_id}_mz"] = mz
                predictions[f"{spec_id}_int"] = weights
                info = {
                    "spec_id": spec_id,
                    "source": source,
                    "name": selected.iloc[spec_id]["name"],
                    "pred_peaks": len(mz),
                }
                for label, ppm, floor in [
                    ("5ppm", 5, 0.0005),
                    ("10ppm", 10, 0.001),
                    ("20ppm", 20, 0.002),
                ]:
                    info.update(
                        {
                            f"{label}_{k}": v
                            for k, v in score(mz, weights, tm, ti, ppm, floor).items()
                        }
                    )
                pm, pi = aggregate(np.rint(mz), weights, decimals=0)
                nm, ni = aggregate(np.rint(tm), ti, decimals=0)
                info["nominal_cosine"] = score(pm, pi, nm, ni, 0, 0)["cosine"]
                scores.append(info)
            print(
                f"{len(scores)}/{len(selected)} spectra; {time.time() - started:.1f}s", flush=True
            )
    frame = pd.DataFrame(scores).merge(selected[["spec_id", "nominal_train_overlap"]], on="spec_id")
    frame.to_csv(args.output / "scores.csv", index=False)
    frame.groupby("source").mean(numeric_only=True).to_csv(args.output / "source_means.csv")
    (args.output / "cache").mkdir()
    np.savez_compressed(args.output / "cache/exact_predictions.npz", **predictions)
    provenance = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "audit_sha256": hashlib.sha256(args.audit.read_bytes()).hexdigest(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "seed": 20261017,
        "isotope_envelope": args.isotope_envelope,
        "prediction_aggregation_decimals": 4,
        "target_intensities": "linear raw",
        "mass_windows": {"hrei_msdb": [40, 500], "others": [0, 1500]},
    }
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        frame.groupby("source")[["5ppm_cosine", "10ppm_cosine", "20ppm_cosine", "nominal_cosine"]]
        .mean()
        .to_string()
    )


def rescore_cached(source: Path, output: Path) -> None:
    """Rescore stored predictions into a fresh directory without repeating inference."""
    selected = pd.read_csv(source / "selected_records.csv").set_index("spec_id")
    frame = pd.read_csv(source / "scores.csv")
    cache = np.load(source / "cache/exact_predictions.npz")
    raw = {name: read_records(Path("data/raw/hr_ei_ms") / path) for name, path in SOURCES.items()}
    for index, row in frame.iterrows():
        spec_id = int(row.spec_id)
        record = selected.loc[spec_id]
        target = np.array(raw[record.source][int(record.record)][1])
        tm, ti = aggregate(target[:, 0], target[:, 1])
        mz, weights = cache[f"{spec_id}_mz"], cache[f"{spec_id}_int"]
        for label, ppm, floor in [("5ppm", 5, 0.0005), ("10ppm", 10, 0.001), ("20ppm", 20, 0.002)]:
            for key, value in score(mz, weights, tm, ti, ppm, floor).items():
                frame.loc[index, f"{label}_{key}"] = value
        pm, pi = aggregate(np.rint(mz), weights, decimals=0)
        nm, ni = aggregate(np.rint(tm), ti, decimals=0)
        frame.loc[index, "nominal_cosine"] = score(pm, pi, nm, ni, 0, 0)["cosine"]
    frame.to_csv(output / "scores.csv", index=False)
    frame.groupby("source").mean(numeric_only=True).to_csv(output / "source_means.csv")
    for name in ["selected_records.csv", "resolved_config.json"]:
        shutil.copyfile(source / name, output / name)
    (output / "cache").mkdir()
    shutil.copyfile(source / "cache/exact_predictions.npz", output / "cache/exact_predictions.npz")
    provenance = json.loads((source / "provenance.json").read_text())
    provenance.update(
        rescored_from=str(source),
        numerical_mass_comparison_epsilon_da=1e-10,
        scoring_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        frame.groupby("source")[["5ppm_cosine", "10ppm_cosine", "20ppm_cosine", "nominal_cosine"]]
        .mean()
        .to_string()
    )


if __name__ == "__main__":
    main()
