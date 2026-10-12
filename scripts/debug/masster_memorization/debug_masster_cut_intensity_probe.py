"""Fit train-only cut probes and screen their frozen fragment-intensity corrections.

Run from the repository root with the FRAGNNET-GPU interpreter:
    python scripts/debug/masster_memorization/debug_masster_cut_intensity_probe.py --stage fit
    python scripts/debug/masster_memorization/debug_masster_cut_intensity_probe.py --stage score --device cpu
    python scripts/debug/masster_memorization/debug_masster_cut_intensity_probe.py --stage report

The analysis YAML is the required schema. Fit/calibration use disjoint training identities;
tuning/confirmation use disjoint validation identities. Candidate support and ion/H conditionals
are held fixed. No production model, dataset or checkpoint is modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch as th
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Subset

from fragnnet.inference.checkpoint import apply_inference_config, init_model_from_ckpt
from fragnnet.runner import load_config
from fragnnet.utils.setup_utils import get_dataset_cls
from fragnnet.utils.spec_similarity import batch_cos_hun_helper, calculate_match_mzs
from fragnnet.utils.spec_utils import round_aggregate_peaks

LOGGER = logging.getLogger(__name__)
HAND_COLUMNS = (
    "bond_order",
    "aromatic",
    "ring",
    "ring_size",
    "conjugated",
    "deg_lo",
    "deg_hi",
    "nh_lo",
    "nh_hi",
    "heteroatom_heteroatom",
    "alpha_to_co",
    "benzylic",
    "allylic",
    "cx_amide",
    "cx_ester",
    "cx_ether",
    "cx_amine",
    "small_side_frac",
    "gasteiger_diff",
    "ep_CC",
    "ep_CN",
    "ep_CO",
    "ep_CS",
    "ep_CX",
    "ep_NO",
    "ep_other",
)


def identity(path: str | Path) -> dict:
    """Hash an input without materializing its contents in memory."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def split_identities(
    mol_ids: np.ndarray, identities: dict[int, str], fraction: float, seed: int
) -> tuple[set[int], set[int]]:
    """Split full molecular identities, keeping aliases in the same partition."""
    unique = np.array(sorted({identities[int(m)] for m in mol_ids}))
    if not 0 < fraction < 1 or len(unique) < 2:
        raise ValueError("identity split needs at least two identities and fraction in (0, 1)")
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    count = max(1, min(len(unique) - 1, int(len(unique) * fraction)))
    chosen = set(unique[:count])
    first = {int(m) for m in mol_ids if identities[int(m)] in chosen}
    second = set(map(int, mol_ids)) - first
    return first, second


def soft_cut_label(structures: list, endpoints: np.ndarray, n_atoms: int) -> np.ndarray:
    """Average boundary-cut indicators over distinct compatible atom sets.

    Predicted shares and repeated H/ion channels never weight the weak label. Endpoints have
    shape (2, B); the return value has shape (B,), with values in [0, 1].
    """
    atom_sets = sorted({tuple(sorted(s[0])) for s in structures})
    if not atom_sets:
        raise ValueError("a compatible group needs at least one atom set")
    inside = np.zeros((len(atom_sets), n_atoms), dtype=bool)
    for i, atoms in enumerate(atom_sets):
        if not atoms or min(atoms) < 0 or max(atoms) >= n_atoms or len(set(atoms)) != len(atoms):
            raise ValueError("invalid compatible atom set")
        inside[i, list(atoms)] = True
    return (inside[:, endpoints[0]] != inside[:, endpoints[1]]).mean(axis=0)


def bond_features(pca: dict, hand: pd.DataFrame, nce: float, inst: str, kind: str) -> np.ndarray:
    """Build symmetric frozen features without measured-spectrum inputs."""
    if kind == "encoder":
        left, right = pca["left"], pca["right"]
        features = np.concatenate([left + right, np.abs(left - right), pca["bond_attr"]], axis=1)
    elif kind == "handmade":
        values = hand[list(HAND_COLUMNS)].to_numpy(dtype=np.float32)
        missing = ~np.isfinite(values[:, [17, 18]])
        features = np.concatenate([np.nan_to_num(values), missing.astype(np.float32)], axis=1)
    else:
        raise ValueError(kind)
    context = np.broadcast_to([nce, float(inst in ("Lumos", "ORB_F"))], (len(features), 2))
    return np.concatenate([features, context], axis=1).astype(np.float32)


def fit_probe(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    calibration_weight: np.ndarray,
    cfg: dict,
) -> tuple[HistGradientBoostingRegressor, LogisticRegression]:
    """Regress fractional weak labels and calibrate log odds on separate molecules."""
    probe = HistGradientBoostingRegressor(
        max_iter=cfg["gbm_iterations"],
        max_leaf_nodes=cfg["gbm_leaf_nodes"],
        learning_rate=cfg["gbm_learning_rate"],
        random_state=cfg["seed"],
        early_stopping=False,
    )
    probe.fit(x, y, sample_weight=weight)
    clip = cfg["probability_clip"]
    z = logit(np.clip(probe.predict(calibration_x), clip, 1 - clip))[:, None]
    # Duplicating a fractional target into weighted Bernoulli outcomes fits its cross entropy.
    calibrator = LogisticRegression(C=100.0, max_iter=200, random_state=cfg["seed"])
    calibrator.fit(
        np.concatenate([z, z]),
        np.concatenate([np.ones(len(z)), np.zeros(len(z))]),
        sample_weight=np.concatenate(
            [
                calibration_weight * calibration_y,
                calibration_weight * (1 - calibration_y),
            ]
        ),
    )
    return probe, calibrator


def selected_spectra(summary: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Cap conditions per molecule with target-independent, deterministic sampling."""
    rng = np.random.default_rng(seed)
    ids = []
    for _, group in summary.sort_index().groupby("mol_id", sort=True):
        values = group.index.to_numpy()
        ids.extend(rng.choice(values, min(cap, len(values)), replace=False))
    return summary.loc[sorted(ids)]


def training_arrays(
    summary: pd.DataFrame, labels: dict[int, np.ndarray], pca: dict, hand: dict, kind: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weight each molecule equally, then each spectrum and each bond equally."""
    counts = summary.groupby("mol_id").size()
    xs, ys, weights = [], [], []
    for sid, row in summary.iterrows():
        y = labels[int(sid)]
        xs.append(
            bond_features(pca[int(row.mol_id)], hand[int(row.mol_id)], row.nce, row.inst, kind)
        )
        ys.append(y)
        weights.append(np.full(len(y), 1 / (counts.loc[row.mol_id] * len(y))))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def fit(cfg: dict, output: Path) -> None:
    """Fit probes, predict all validation bonds and save the split/score manifests."""
    # 1. Audit identities and construct ambiguity-preserving training-only weak labels.
    root = Path(cfg["artifact_root"])
    probe_root = root / "enc_probe"
    molecules = pd.read_pickle(cfg["molecules"]).set_index("mol_id")
    identities = molecules.inchikey_s.to_dict()
    train = pd.read_pickle(root / "train_summary.pkl")
    val = pd.read_pickle(root / "val_summary.pkl")
    train_ids = {identities[int(m)] for m in train.mol_id.unique()}
    val_ids = {identities[int(m)] for m in val.mol_id.unique()}
    if train_ids & val_ids:
        raise ValueError("train/validation molecular identity overlap")
    pca_train = pd.read_pickle(probe_root / "mol_bondfeat_pca_train4k.pkl")
    pca_val = pd.read_pickle(probe_root / "mol_bondfeat_pca_val.pkl")
    hand = pd.read_pickle(probe_root / "mol_handfeat.pkl")
    peaks = pd.read_pickle(cfg["training_peaks"])
    bases = peaks.loc[peaks.tgt_rank == 1].set_index("spec_id")
    if not bases.index.is_unique or not bases.index.isin(train.index).all():
        raise ValueError("weak labels must be unique training-spectrum base groups")
    labels, ambiguous, omitted = {}, 0, 0
    for sid, row in bases.iterrows():
        mid = int(train.loc[sid, "mol_id"])
        endpoints = pca_train[mid]["endpoints"]
        mol = molecules.loc[mid, "mol"]
        expected = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]).T
        if not np.array_equal(endpoints, expected):
            raise ValueError(f"molecule {mid}: cached bond order differs from RDKit")
        y = soft_cut_label(row.structures, endpoints, mol.GetNumAtoms())
        if not np.any(y):
            omitted += 1
            continue
        labels[int(sid)] = y.astype(np.float32)
        ambiguous += int(len({s[0] for s in row.structures}) > 1)
    eligible = train.loc[sorted(labels)]
    calibration_mols, fit_mols = split_identities(
        eligible.mol_id.unique(), identities, cfg["calibration_fraction"], cfg["seed"]
    )
    tune_mols, confirm_mols = split_identities(
        val.mol_id.unique(), identities, cfg["tuning_fraction"], cfg["seed"] + 1
    )
    cap = cfg["max_fit_spectra_per_molecule"]
    fit_summary = selected_spectra(eligible[eligible.mol_id.isin(fit_mols)], cap, cfg["seed"])
    cal_summary = selected_spectra(
        eligible[eligible.mol_id.isin(calibration_mols)], cap, cfg["seed"] + 2
    )
    audit = {
        "fit_molecules": sorted(fit_mols),
        "calibration_molecules": sorted(calibration_mols),
        "tuning_molecules": sorted(tune_mols),
        "confirmation_molecules": sorted(confirm_mols),
        "fit_spec_ids": fit_summary.index.tolist(),
        "calibration_spec_ids": cal_summary.index.tolist(),
        "weak_label_spectra": len(labels),
        "ambiguous_label_spectra": ambiguous,
        "no_cut_base_groups_omitted": omitted,
        "identity_overlap": 0,
    }
    (output / "split_manifest.json").write_text(json.dumps(audit, indent=2))
    val = val[["mol_id", "nce", "inst", "novelty"]].copy()
    val["partition"] = np.where(val.mol_id.isin(tune_mols), "tune", "confirm")
    selector = pd.read_pickle(root / "i3/scores.pkl")
    val["independent_floor"] = selector.cos_it_fa1_63.reindex(val.index) < 0.05
    if selector.cos_it_fa1_63.reindex(val.index).isna().any():
        raise ValueError("missing independent floor selector")
    val.to_pickle(output / "validation_manifest.pkl")
    LOGGER.info(
        "Weak labels: %d spectra, %d ambiguous; fit/calibration %d/%d molecules",
        len(labels),
        ambiguous,
        len(fit_mols),
        len(calibration_mols),
    )

    # 2. Fit fixed probes and calibration; score validation without using its labels.
    score_maps = {}
    for kind in cfg["probe_kinds"]:
        x, y, w = training_arrays(fit_summary, labels, pca_train, hand, kind)
        cx, cy, cw = training_arrays(cal_summary, labels, pca_train, hand, kind)
        LOGGER.info("Fitting %s on %d bonds, calibrating on %d", kind, len(y), len(cy))
        probe, calibrator = fit_probe(x, y, w, cx, cy, cw, cfg)
        del x, y, w, cx, cy, cw
        joblib.dump((probe, calibrator), output / f"probe_{kind}.joblib")
        LOGGER.info(
            "%s calibration slope %.4f, intercept %.4f",
            kind,
            calibrator.coef_[0, 0],
            calibrator.intercept_[0],
        )
        predictions = {}
        # Bound temporary feature storage to approximately 50,000 bonds.
        chunk_x, chunk_ids, chunk_sizes = [], [], []
        for sid, row in val.iterrows():
            features = bond_features(
                pca_val[int(row.mol_id)], hand[int(row.mol_id)], row.nce, row.inst, kind
            )
            chunk_x.append(features)
            chunk_ids.append(int(sid))
            chunk_sizes.append(len(features))
            if sum(chunk_sizes) >= 50000 or sid == val.index[-1]:
                raw = probe.predict(np.concatenate(chunk_x))
                clip = cfg["probability_clip"]
                z = logit(np.clip(raw, clip, 1 - clip))[:, None]
                calibrated = calibrator.decision_function(z)
                for spec, values in zip(
                    chunk_ids, np.split(calibrated, np.cumsum(chunk_sizes)[:-1])
                ):
                    predictions[spec] = values.astype(np.float32)
                chunk_x, chunk_ids, chunk_sizes = [], [], []
        score_maps[kind] = predictions
        LOGGER.info("%s validation scores: %d spectra", kind, len(predictions))
    pd.to_pickle(score_maps, output / "bond_log_odds.pkl")


def metric_cache(
    channel_mzs: np.ndarray, true_mzs: np.ndarray, true_ints: np.ndarray, rel: float, min_mz: float
) -> dict:
    """Prepare sum-rounded peaks and an intensity-independent tolerance matching mask."""
    pm, inverse = np.unique(
        th.round(th.as_tensor(channel_mzs), decimals=4).numpy(), return_inverse=True
    )
    tm, ti_idx = np.unique(
        th.round(th.as_tensor(true_mzs), decimals=4).numpy(), return_inverse=True
    )
    ti = np.bincount(ti_idx, weights=true_ints)
    tolerance = rel * np.maximum(tm, min_mz)
    mask = np.abs(tm[:, None] - pm[None, :]) <= tolerance[:, None]
    true_rows = np.flatnonzero(mask.any(axis=1))
    pred_cols = np.flatnonzero(mask.any(axis=0))
    submask = mask[np.ix_(true_rows, pred_cols)]
    # If no predicted peak can serve two targets, row maxima already solve the assignment.
    independent = not np.any(submask.sum(axis=0) > 1)
    base = int(np.argmax(ti))
    return {
        "inverse": inverse,
        "pm": pm,
        "ti": ti,
        "target_norm": np.linalg.norm(ti),
        "rows": true_rows,
        "cols": pred_cols,
        "mask": submask,
        "independent": independent,
        "base_mask": mask[base],
    }


def score_channels(log_prob: np.ndarray, cache: dict) -> tuple[float, float, float]:
    """Return production-equivalent cosine, base hit and best compatible mass-group rank.

    All unmatched target and predicted peaks contribute to their L2 norms. Ambiguous masses
    use one-to-one maximum-product assignment, rather than duplicating target intensity.
    """
    if not len(log_prob):
        return 0.0, 0.0, float("nan")
    pi = np.bincount(
        cache["inverse"], weights=np.exp(log_prob - log_prob.max()), minlength=len(cache["pm"])
    )
    ti = cache["ti"]
    rows, cols, mask = cache["rows"], cache["cols"], cache["mask"]
    numerator = 0.0
    if len(rows) and len(cols):
        cost = ti[rows, None] * pi[cols][None, :] * mask
        if cache["independent"]:
            numerator = cost.max(axis=1).sum()
        else:
            i, j = linear_sum_assignment(cost, maximize=True)
            numerator = cost[i, j].sum()
    denominator = cache["target_norm"] * np.linalg.norm(pi)
    cosine = float(numerator / denominator) if denominator > 0 else 0.0
    hit = float(cache["base_mask"][int(np.argmax(pi))])
    matches = np.flatnonzero(cache["base_mask"])
    rank = float(1 + (pi > pi[matches].max()).sum()) if len(matches) else float("nan")
    return cosine, hit, rank


def fragment_corrections(
    inside: np.ndarray,
    endpoints: np.ndarray,
    scores: dict[str, np.ndarray],
    proposal: np.ndarray,
    final: np.ndarray,
    spec_id: int,
    cfg: dict,
) -> dict[str, np.ndarray]:
    """Sum per-bond log odds over each fragment's own boundary, retaining every fragment."""
    boundary = inside[:, endpoints[0]] != inside[:, endpoints[1]]
    corrections = {"baseline": np.zeros(len(inside))}
    permutation = np.random.default_rng([cfg["seed"], spec_id]).permutation(endpoints.shape[1])
    for kind in cfg["probe_kinds"]:
        values = scores[kind]
        if len(values) != endpoints.shape[1] or not np.isfinite(values).all():
            raise ValueError("per-bond probe scores must align with all molecule bonds")
        for tag, vector in ((kind, values), (f"{kind}_shuffle", values[permutation])):
            summed = boundary @ vector.astype(np.float64)
            for alpha in cfg["probe_strengths"]:
                corrections[f"{tag}:{alpha}"] = alpha * summed
    for tag, values, strengths in (
        ("proposal", proposal, cfg["proposal_strengths"]),
        ("cut_count", boundary.sum(axis=1), cfg["cut_count_strengths"]),
        ("temperature", final, cfg["temperature_strengths"]),
    ):
        for alpha in strengths:
            corrections[f"{tag}:{alpha}"] = alpha * values
    return corrections


def score(cfg: dict, output: Path, device_name: str, limit: int) -> None:
    """Run fresh annotated inference and score all fixed corrections on that same output."""
    # 1. Diagnostic instrument-label copy, strict checkpoint load and selected dataset.
    started_utc = pd.Timestamp.now(tz="UTC").isoformat()
    device = th.device(device_name)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu")
    root = Path(cfg["artifact_root"])
    copied_fp = output / "spec_val.pkl"
    specs = pd.read_pickle(root / "subset_spec_df_val_all.pkl")
    specs["inst_type"] = specs.inst_type.replace({"OB": "ORB", "OB_F": "ORB_F"})
    specs.to_pickle(copied_fp)
    config = load_config("config", cfg["custom_fp"], [f"spec_fp={copied_fp}"])
    config = apply_inference_config(
        config, disable_preproc=True, num_workers=0, eval_batch_size=cfg["batch_size"]
    )
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(cfg["checkpoint"], config, device)
    dataset = get_dataset_cls(config["model_type"])(split="val", **config)
    specs = dataset.spec_df.reset_index(drop=True)
    n = min(limit, len(dataset)) if limit else len(dataset)
    loader = DataLoader(
        Subset(dataset, list(range(n))),
        batch_size=cfg["batch_size"],
        collate_fn=dataset.get_collate_fn(),
        num_workers=0,
        shuffle=False,
    )
    score_maps = pd.read_pickle(output / "bond_log_odds.pkl")
    cached_bonds = pd.read_pickle(root / "enc_probe/mol_bondfeat_pca_val.pkl")
    manifest = pd.read_pickle(output / "validation_manifest.pkl")
    inst_types = config["spec_params"]["inst_types"]
    rows, cursor, reference_max = [], 0, 0.0
    started = time.monotonic()

    # 2. Emit all channels once; target arrays are used only by metrics, never corrections.
    with th.inference_mode():
        for batch in loader:
            size = int(batch["batch_size"])
            batch_specs = specs.iloc[cursor : cursor + size]
            if not np.array_equal(batch["spec_unique_id"].numpy(), batch_specs.spec_id.to_numpy()):
                raise RuntimeError("dataset spectrum order mismatch")
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            kwargs = {
                k: batch[k]
                for k in (
                    "mol_pyg",
                    "spec_prec_type",
                    "spec_inst_type",
                    "spec_frag_mode",
                    "spec_nce",
                    "spec_nce_batch_idxs",
                    "spec_ace",
                    "spec_ace_batch_idxs",
                    "spec_nce_weights",
                    "spec_ace_weights",
                )
                if k in batch
            }
            out = model.model(**kwargs, return_annotation=True)
            if "pred_pathway_scores" in out:  # merged rows: use the per-pathway rows
                out = {
                    **out,
                    "pred_mzs": out["pred_pathway_mzs"],
                    "pred_ints": out["pred_pathway_scores"],
                    "pred_logprobs": out["pred_pathway_scores"].log(),
                    "pred_batch_idxs": out["pred_pathway_batch_idxs"],
                }
            arrays = {
                k: out[k].cpu().numpy()
                for k in (
                    "ion_h_fragment_batch_idxs",
                    "ion_h_fragment_proposal_logprobs",
                    "cross_ev_fragment_logprobs",
                    "pred_batch_idxs",
                    "pred_mzs",
                    "pred_logprobs",
                    "pred_pathway_fragment_idxs",
                    "fragment_atom_indices",
                    "fragment_atom_fragment_idxs",
                )
            }
            mol_pyg = batch["mol_pyg"]
            ptr = mol_pyg.ptr.cpu().numpy()
            bond_index = mol_pyg.chemical_bond_index.cpu().numpy()
            bond_batch = mol_pyg.batch[mol_pyg.chemical_bond_index[0]].cpu().numpy()
            tm, ti, tb = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                batch_size=size,
            )
            tm, ti, tb = tm.double().cpu().numpy(), ti.double().cpu().numpy(), tb.cpu().numpy()
            frag_batch = arrays["ion_h_fragment_batch_idxs"]
            fptr = np.searchsorted(frag_batch, np.arange(size + 1))
            cptr = np.searchsorted(arrays["pred_batch_idxs"], np.arange(size + 1))
            bptr = np.searchsorted(bond_batch, np.arange(size + 1))
            atom_order = np.argsort(arrays["fragment_atom_fragment_idxs"], kind="stable")
            atom_frags = arrays["fragment_atom_fragment_idxs"][atom_order]
            member_atoms = arrays["fragment_atom_indices"][atom_order]
            aptr = np.searchsorted(atom_frags, fptr)
            for view, row in enumerate(batch_specs.itertuples()):
                sid, mid = int(row.spec_id), int(row.mol_id)
                f0, f1 = fptr[view : view + 2]
                c0, c1 = cptr[view : view + 2]
                a0, a1 = ptr[view : view + 2]
                b0, b1 = bptr[view : view + 2]
                mem0, mem1 = aptr[view : view + 2]
                inside = np.zeros((f1 - f0, a1 - a0), dtype=bool)
                inside[atom_frags[mem0:mem1] - f0, member_atoms[mem0:mem1] - a0] = True
                endpoints = bond_index[:, b0:b1] - a0
                if not np.array_equal(endpoints, cached_bonds[mid]["endpoints"]):
                    raise ValueError(f"molecule {mid}: inference and cached bond orders differ")
                if not inside.any(axis=1).all():
                    raise ValueError(f"spectrum {sid}: emitted fragment has no atom membership")
                final = arrays["cross_ev_fragment_logprobs"][f0:f1].astype(np.float64)
                proposal = arrays["ion_h_fragment_proposal_logprobs"][f0:f1].astype(np.float64)
                scores = {k: score_maps[k][sid] for k in cfg["probe_kinds"]}
                corrections = fragment_corrections(
                    inside, endpoints, scores, proposal, final, sid, cfg
                )
                channel_frag = arrays["pred_pathway_fragment_idxs"][c0:c1] - f0
                mz = arrays["pred_mzs"][c0:c1].astype(np.float64)
                lp = arrays["pred_logprobs"][c0:c1].astype(np.float64)
                keep = mz < config["mz_max"]
                mz, lp, channel_frag = mz[keep], lp[keep], channel_frag[keep]
                tolerance = config["inst_type_loss_tol"][
                    inst_types[int(batch["spec_inst_type"][view])]
                ]
                cache = metric_cache(
                    mz, tm[tb == view], ti[tb == view], tolerance["rel"], tolerance["min_mz"]
                )
                record = {
                    "spec_id": sid,
                    "mol_id": mid,
                    **manifest.loc[sid].drop("mol_id").to_dict(),
                }
                for tag, correction in corrections.items():
                    cosine, hit, rank = score_channels(lp + correction[channel_frag], cache)
                    record[f"cos|{tag}"], record[f"hit|{tag}"], record[f"rank|{tag}"] = (
                        cosine,
                        hit,
                        rank,
                    )
                # Fresh direct metric check on the first 32 spectra against production helpers.
                if cursor + view < 32:
                    pt, pp, _ = round_aggregate_peaks(
                        th.from_numpy(mz),
                        th.from_numpy(np.exp(lp)),
                        th.zeros(len(mz), dtype=th.long),
                        agg="sum",
                    )
                    tt, tp, _ = round_aggregate_peaks(
                        th.from_numpy(tm[tb == view]),
                        th.from_numpy(ti[tb == view]),
                        th.zeros(sum(tb == view), dtype=th.long),
                        agg="sum",
                    )
                    tol = tolerance["rel"] * tt.clamp_min(tolerance["min_mz"])
                    mask = calculate_match_mzs(
                        tt[None], pt[None], tolerance=0.0, relative=False, tol_per_true=tol[None]
                    )
                    direct = float(
                        batch_cos_hun_helper(
                            tp[None], pp[None], mask, mask.any(-1), mask.any(-2), False, None, None
                        )[0]
                    )
                    error = abs(direct - record["cos|baseline"])
                    reference_max = max(reference_max, error)
                    if error > 1e-7:
                        raise ValueError(f"metric mismatch on spectrum {sid}: {error}")
                rows.append(record)
            cursor += size
            if cursor % (cfg["batch_size"] * 10) == 0 or cursor == n:
                LOGGER.info("Scored %d/%d spectra in %.1f s", cursor, n, time.monotonic() - started)
    # 3. Save complete matched rows; partial smoke runs cannot be reported as a full screen.
    pd.DataFrame(rows).set_index("spec_id").to_pickle(
        output / ("smoke_scores.pkl" if limit else "scores.pkl")
    )
    (output / ("smoke_audit.json" if limit else "score_audit.json")).write_text(
        json.dumps(
            {
                "spectra": n,
                "elapsed_seconds": time.monotonic() - started,
                "device": device_name,
                "candidate_backend": config["masster_candidate_backend"],
                "max_production_metric_error": reference_max,
                "resolved_config": config,
                "source": identity(__file__),
                "started_utc": started_utc,
                "completed_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "inputs": [
                    identity(p)
                    for p in (
                        cfg["checkpoint"],
                        copied_fp,
                        config["mol_fp"],
                        output / "bond_log_odds.pkl",
                        output / "validation_manifest.pkl",
                    )
                ],
            },
            indent=2,
            default=str,
        )
    )


def choose_arm(frame: pd.DataFrame, arms: list[str]) -> str:
    """Select by tuning cosine only; ties prefer no correction, then smaller magnitude."""

    def key(tag: str) -> tuple:
        strength = float(tag.split(":")[-1]) if ":" in tag else 0.0
        return -frame[f"cos|{tag}"].mean(), abs(strength), tag

    return min(arms, key=key)


def bootstrap_delta(
    frame: pd.DataFrame, first: str, second: str, repetitions: int, seed: int
) -> dict:
    """Bootstrap molecule clusters for a spectrum-weighted paired cosine difference."""
    values = frame[f"cos|{first}"] - frame[f"cos|{second}"]
    grouped = pd.DataFrame({"mol_id": frame.mol_id, "delta": values}).groupby("mol_id").delta
    stats = grouped.agg(["sum", "count", "mean"])
    if not len(stats):
        return {"spectra": 0, "molecules": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(stats), size=(repetitions, len(stats)))
    boot = stats["sum"].to_numpy()[draws].sum(axis=1) / stats["count"].to_numpy()[draws].sum(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "spectra": len(frame),
        "molecules": len(stats),
        "mean": float(values.mean()),
        "molecule_mean": float(stats["mean"].mean()),
        "low": float(low),
        "high": float(high),
    }


def report(cfg: dict, output: Path) -> None:
    """Lock tuning choices before reading their correction-held-out confirmation effect."""
    scores = pd.read_pickle(output / "scores.pkl")
    manifest = pd.read_pickle(output / "validation_manifest.pkl")
    if set(scores.index) != set(manifest.index) or not scores.index.is_unique:
        raise ValueError("screen report requires exactly the complete validation manifest")
    if set(scores.loc[scores.partition == "tune", "mol_id"]) & set(
        scores.loc[scores.partition == "confirm", "mol_id"]
    ):
        raise ValueError("tuning/confirmation molecule overlap")
    arms = [column[4:] for column in scores if column.startswith("cos|")]
    tune = scores[scores.partition == "tune"]
    selected = choose_arm(tune, [a for a in arms if a.split(":")[0] in cfg["probe_kinds"]])
    control_choices = {
        kind: choose_arm(tune, [a for a in arms if a.split(":")[0] == kind])
        for kind in (
            "proposal",
            "cut_count",
            "temperature",
            *[f"{k}_shuffle" for k in cfg["probe_kinds"]],
        )
    }
    strongest = choose_arm(tune, ["baseline", *control_choices.values()])
    lock = {
        "selected_probe": selected,
        "strongest_control": strongest,
        "control_choices": control_choices,
        "selection_metric": "tuning mean instrument cosine",
    }
    (output / "locked_choices.json").write_text(json.dumps(lock, indent=2))
    confirm = scores[scores.partition == "confirm"]
    comparisons = {}
    for name, subset in (
        ("confirmation", confirm),
        ("unfamiliar", confirm[confirm.novelty <= 0.4]),
        ("independent_floor", confirm[confirm.independent_floor]),
    ):
        comparisons[name] = {
            "baseline": float(subset["cos|baseline"].mean()),
            "corrected": float(subset[f"cos|{selected}"].mean()),
            "vs_baseline": bootstrap_delta(
                subset, selected, "baseline", cfg["bootstrap_repetitions"], cfg["seed"]
            ),
            "vs_control": bootstrap_delta(
                subset, selected, strongest, cfg["bootstrap_repetitions"], cfg["seed"]
            ),
            "base_hit_baseline": float(subset["hit|baseline"].mean()),
            "base_hit_corrected": float(subset[f"hit|{selected}"].mean()),
        }
    main = comparisons["confirmation"]
    delta, excess = main["vs_baseline"], main["vs_control"]
    passed = delta["mean"] >= cfg["screen_delta"] and delta["low"] > 0 and excess["low"] > 0
    for population in ("unfamiliar", "independent_floor"):
        result = comparisons[population]["vs_baseline"]
        if result["high"] is not None and result["high"] < 0:
            passed = False
    result = {**lock, "screen_pass": bool(passed), "comparisons": comparisons}
    (output / "result.json").write_text(json.dumps(result, indent=2))
    sweep = []
    for arm in arms:
        sweep.append(
            {
                "arm": arm,
                "tune_cosine": tune[f"cos|{arm}"].mean(),
                "confirm_cosine": confirm[f"cos|{arm}"].mean(),
                "confirm_base_hit": confirm[f"hit|{arm}"].mean(),
            }
        )
    pd.DataFrame(sweep).to_csv(output / "sweep.csv", index=False)
    LOGGER.info(
        "Locked probe %s; strongest control %s; screen pass %s", selected, strongest, passed
    )
    LOGGER.info("Confirmation %s", json.dumps(main))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/analysis/masster_cut_intensity_probe.yaml")
    parser.add_argument("--stage", choices=("fit", "score", "report"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--limit", type=int, default=0, help="score smoke only; full report refuses it"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = yaml.safe_load(Path(args.config).read_text())
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])  # noqa: NPY002 - seed dependencies using the legacy RNG
    th.manual_seed(cfg["seed"])
    th.set_num_threads(cfg["threads"])
    output = Path(cfg["output"])
    output.mkdir(parents=True, exist_ok=True)
    source = identity(__file__)
    if args.stage == "fit":
        root = Path(cfg["artifact_root"])
        paths = [
            args.config,
            cfg["checkpoint"],
            cfg["custom_fp"],
            cfg["training_peaks"],
            cfg["molecules"],
            root / "train_summary.pkl",
            root / "val_summary.pkl",
            root / "enc_probe/mol_bondfeat_pca_train4k.pkl",
            root / "enc_probe/mol_bondfeat_pca_val.pkl",
            root / "enc_probe/mol_handfeat.pkl",
            root / "i3/scores.pkl",
        ]
        provenance = {
            "config": cfg,
            "source": source,
            "inputs": [identity(p) for p in paths],
            "started_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "git_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_diff_sha256": hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"])
            ).hexdigest(),
            "torch": th.__version__,
            "numpy": np.__version__,
        }
        (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
        fit(cfg, output)
    elif args.stage == "score":
        score(cfg, output, args.device, args.limit)
    else:
        report(cfg, output)


if __name__ == "__main__":
    main()
