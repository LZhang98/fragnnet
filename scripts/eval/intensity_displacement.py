"""Phase 0 of the intensity-displacement plan: where does misplaced predicted intensity go?

See ``docs/plans/masshatter/masshatter_intensity_displacement_plan.md`` §4. Works offline on
MASShatter inference exports (``run_save_inference_eval.py`` HDF5, full candidate set) and dumped
eval targets (pickles with ``spec_id``, ``mol_id``, ``nce``, ``prec_mz``, ``peaks``). All quantities
live in the candidate space of the ranking decomposition: candidates sharing an exact m/z (5
decimals) are merged, and each true peak maps to its nearest candidate within tolerance (10 ppm,
200 Da floor). Candidates are the union over the runs being compared.

Stages:

``bias``
    Two runs A and B on the same spectra. With ``p_k`` the L2-normalized prediction of run k and
    ``t = T_covered / |T_all|``, the cosine gap is ``1 - cos_k = |e_k|^2 / 2 + (1 - |t|^2) / 2``
    with ``e_k = p_k - t``. The first term is displacement, the second coverage. With
    ``e_bar = (e_A + e_B) / 2`` and ``d = (e_A - e_B) / 2``, the mean single-run displacement
    splits exactly as ``|e_bar|^2 + |d|^2``. If ``e_k = b + v_k`` with independent zero-mean
    ``v_k`` of variance ``s^2``, then ``E|e_bar|^2 = |b|^2 + s^2 / 2`` and ``E|d|^2 = s^2 / 2``,
    so the shared (bias) part is ``|e_bar|^2 - |d|^2`` and the run-specific part ``2 |d|^2``.
    All three are reported in cosine units (halved). A replicate floor comes from spectra of the
    same molecule at the same NCE: the target noise ``tau^2 = |t_a - t_b|^2 / 2`` is part of every
    run's error and is counted as shared, so ``shared - tau^2 / 2`` (cosine units) is the
    reducible shared part.

``transport``
    For a predictor ``q`` (unit L1 over candidates) and the covered target ``y`` (unit L1),
    excess ``(q - y)+`` must travel to deficit ``(y - q)+``. An optimal transport plan with a
    class-ordered ground cost assigns every unit of displaced mass to a (donor, recipient) pair,
    classified by the two candidates' ion formulas (closest subformula of the precursor ion):

    - ``prec``: either end is the precursor;
    - ``ambig``: either end has a second subformula within the exact-mass tolerance, or none;
    - ``hshift``: same heavy atoms, different H;
    - ``nl1``: heavy atoms nested, and the heavy-atom difference is one common neutral loss
      (H count within 2 of that loss's);
    - ``nested``: heavy atoms nested, any other difference;
    - ``sibling``: neither heavy-atom set contains the other.

    Controls: the same transport with a class-agnostic m/z-distance cost; a chance predictor that
    permutes ``q`` over all candidates of the spectrum; and a support-preserving null that
    permutes ``q`` within the target-supported candidates and within the rest separately. Real
    peaks are themselves linked by common losses, so only the support-preserving null separates
    "moved to a neighbour" from "wrong order among real peaks". Predictors: A, B, their mean (the
    shared component), and A against B (where the runs disagree). Base-peak misses compare the
    predicted top's relation to the true base peak with a random candidate, an
    intensity-weighted random other true peak, and the true second peak.

``precursor``
    At low NCE, the precursor against the strongest covered non-precursor target peak. Records
    the observed winner and the model's implied winner; ``summary`` fits a formula-only learner
    (neutral-loss composition, NCE, precursor composition) on train pairs and scores it on val.

``summary``
    Reads the per-spectrum tables and writes the gate tables.

Usage (from the repository root)::

    python scripts/eval/intensity_displacement.py bias --targets_fp T.pkl \
        --pred_hdf5 A.hdf5 B.hdf5 --mol_df_fp data/proc/nist20/mol_df.pkl --out_fp out.pkl
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import linear_sum_assignment

LOG = logging.getLogger("intensity_displacement")

EL = ["C", "H", "N", "O", "S", "Cl", "F", "Br", "P", "I", "Si", "Se"]
H_IDX = EL.index("H")
HEAVY = [i for i in range(len(EL)) if i != H_IDX]
MASS = {"C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956, "S": 31.97207100,
        "Cl": 34.96885268, "F": 18.99840322, "Br": 78.9183371, "P": 30.97376163, "I": 126.904473,
        "Si": 27.9769265325, "Se": 79.9165213}
MV = np.array([MASS[e] for e in EL])
M_E = 0.00054857990946
TOL_PPM, TOL_FLOOR = 10e-6, 200.0
# Candidate m/z are theoretical ion masses rounded to 5 decimals, so their formulas are assigned
# at an exact-mass tolerance; the 10 ppm instrument tolerance applies only to target peaks.
ASSIGN_TOL_DA = 5e-5
# Transport resolution: each side is quantized into this many equal mass units.
TRANSPORT_UNITS = 500

# Common neutral losses (heavy atoms define the class; H may differ by up to 2 from the listed
# count). Seeded from the dominant-peak decomposition's failing-cohort list plus the usual
# small losses.
NL_VOCAB = ["H2O", "NH3", "CO", "CO2", "CH2O2", "C2H2O", "CH4O", "CH2O", "C2H4", "C2H2", "CH4",
            "HCN", "HCl", "HF", "HBr", "HI", "H2S", "SO2", "SO3", "NO", "NO2", "CH3NO", "CH5N",
            "C2H4O2", "C3H9N", "C2H7N", "C4H8", "C3H6", "HNCO", "H3PO4", "HPO3", "C6H10O5",
            "C6H8O6", "C5H8O4", "C6H10O4", "CH3", "C2H6"]
CLASSES = ["prec", "ambig", "hshift", "nl1", "nested", "sibling"]
C_PREC, C_AMBIG, C_HSHIFT, C_NL1, C_NESTED, C_SIBLING = range(6)
# Class-ordered ground cost; a small m/z term breaks ties inside a class.
CLASS_COST = np.array([2.0, 2.5, 1.0, 1.0, 3.0, 4.0])
MZ_TIE = 1e-4
EXCESS_KEEP = 0.99
MAX_ENTRIES = 150


def tol(mz: np.ndarray | float) -> np.ndarray:
    """Instrument tolerance in Da for FT: 10 ppm with a 200 Da floor."""
    return TOL_PPM * np.maximum(mz, TOL_FLOOR)


def formula_vec(formula: str) -> np.ndarray:
    """Element-count vector (length ``len(EL)``) of a formula string such as ``C6H10O5``."""
    v = np.zeros(len(EL), np.int64)
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        v[EL.index(el)] += int(n or 1)
    return v


NL_HEAVY = np.array([formula_vec(f)[HEAVY] for f in NL_VOCAB])
NL_H = np.array([formula_vec(f)[H_IDX] for f in NL_VOCAB])


def assign_formulas(
    prec_ion: np.ndarray, mzs: np.ndarray, tol_da: float = ASSIGN_TOL_DA,
) -> tuple[np.ndarray, np.ndarray]:
    """Map m/z values to the closest ion formula that is a subformula of the precursor ion.

    Heavy-atom subformulas are enumerated and sorted by mass. For each H count the needed heavy
    mass is found by binary search, and the (heavy, H) pair with the smallest error wins. A
    formula is ambiguous when a second (heavy, H) combination also lies within ``tol_da``.

    Args:
        prec_ion: Element counts of the precursor ion, shape ``(len(EL),)``.
        mzs: Candidate m/z values, shape ``(n,)``.
        tol_da: Assignment tolerance in Da.

    Returns:
        Ion formula vectors ``(n, len(EL))`` and a flag ``(n,)`` that is true when the
        assignment is ambiguous or no subformula lies within tolerance.
    """
    mzs = np.asarray(mzs, float)
    heavy = [i for i in HEAVY if prec_ion[i] > 0]
    grid = np.array(list(itertools.product(*[range(prec_ion[i] + 1) for i in heavy])), np.int64)
    grid = grid.reshape(-1, len(heavy))
    hm = grid @ MV[heavy]
    order = np.argsort(hm)
    hm, grid = hm[order], grid[order]
    hs = np.arange(prec_ion[H_IDX] + 1)
    need = mzs[:, None] + M_E - hs[None, :] * MASS["H"]
    j = np.clip(np.searchsorted(hm, need), 1, len(hm) - 1)
    lo, hi = np.abs(need - hm[j - 1]), np.abs(need - hm[j])
    near = np.where(lo <= hi, j - 1, j)
    err = np.minimum(lo, hi)
    best_h = np.argmin(err, axis=1)
    rows = np.arange(len(mzs))
    out = np.zeros((len(mzs), len(EL)), np.int64)
    out[:, heavy] = grid[near[rows, best_h]]
    out[:, H_IDX] = hs[best_h]
    # Count (heavy, H) combinations within tolerance: both neighbours for every H count.
    t = np.full((len(mzs), 1), tol_da)
    n_within = (lo <= t).sum(1) + (hi <= t).sum(1)
    unassigned = err[rows, best_h] > t[:, 0]
    return out, (n_within >= 2) | unassigned


def pair_classes(
    v_don: np.ndarray, v_rec: np.ndarray, prec_don: np.ndarray, prec_rec: np.ndarray,
    amb_don: np.ndarray, amb_rec: np.ndarray,
) -> np.ndarray:
    """Displacement class of every (donor, recipient) pair.

    Args:
        v_don: Donor ion formulas ``(n, len(EL))``.
        v_rec: Recipient ion formulas ``(m, len(EL))``.
        prec_don, prec_rec: Whether each candidate is the precursor, ``(n,)`` / ``(m,)``.
        amb_don, amb_rec: Whether each assignment is ambiguous, ``(n,)`` / ``(m,)``.

    Returns:
        Class indices ``(n, m)`` into ``CLASSES``, applied in the order prec, ambig, hshift,
        nl1, nested, sibling.
    """
    hd, hr = v_don[:, None, HEAVY], v_rec[None, :, HEAVY]
    diff_heavy = hd - hr
    diff_h = v_don[:, None, H_IDX] - v_rec[None, :, H_IDX]
    same = (diff_heavy == 0).all(-1)
    don_contains = (diff_heavy >= 0).all(-1)
    rec_contains = (diff_heavy <= 0).all(-1)
    nested = (don_contains | rec_contains) & ~same
    # One-loss: |heavy difference| equals a vocabulary loss's heavy atoms, H within 2.
    sign = np.where(don_contains, 1, -1)[..., None]
    loss_heavy = diff_heavy * sign
    loss_h = diff_h * sign[..., 0]
    match = (loss_heavy[:, :, None, :] == NL_HEAVY[None, None]).all(-1)
    match &= np.abs(loss_h[:, :, None] - NL_H[None, None]) <= 2
    nl1 = nested & match.any(-1)
    cls = np.full(same.shape, C_SIBLING, np.int64)
    cls[nested] = C_NESTED
    cls[nl1] = C_NL1
    cls[same] = C_HSHIFT
    cls[amb_don[:, None] | amb_rec[None, :]] = C_AMBIG
    cls[prec_don[:, None] | prec_rec[None, :]] = C_PREC
    return cls


def _quantize(v: np.ndarray, units: int) -> np.ndarray:
    """Largest-remainder rounding of ``v`` into ``units`` integer units (sums exactly)."""
    f = v / v.sum() * units
    b = np.floor(f).astype(np.int64)
    b[np.argsort(-(f - b))[: units - int(b.sum())]] += 1
    return b


def transport(x: np.ndarray, y: np.ndarray, cost: np.ndarray, units: int = TRANSPORT_UNITS) -> np.ndarray:
    """Optimal transport of excess ``x`` to deficit ``y`` on a quantized mass grid.

    Each side is rounded into ``units`` equal units (largest remainder), and the unit-level
    problem is an exact assignment (``linear_sum_assignment``). Each entry is off by less than
    one unit, i.e. ``min(sum(x), sum(y)) / units``. Against HiGHS on the same problem, class
    shares agreed to 0.2% at 1,000 units and about 1% at 400, at 1/300 of the time.

    Args:
        x: Excess masses ``(n,)``.
        y: Deficit masses ``(m,)``.
        cost: Ground cost ``(n, m)``.
        units: Quantization resolution.

    Returns:
        Flow ``(n, m)`` moving ``min(sum(x), sum(y))`` in total.
    """
    ri = np.repeat(np.arange(len(x)), _quantize(x, units))
    cj = np.repeat(np.arange(len(y)), _quantize(y, units))
    r, c = linear_sum_assignment(cost[ri][:, cj])
    flow = np.zeros_like(cost)
    np.add.at(flow, (ri[r], cj[c]), min(x.sum(), y.sum()) / units)
    return flow


def _keep(v: np.ndarray) -> np.ndarray:
    """Indices of the largest entries of ``v`` holding ``EXCESS_KEEP`` of its mass."""
    idx = np.flatnonzero(v > 0)
    idx = idx[np.argsort(-v[idx])]
    csum = np.cumsum(v[idx])
    k = int(np.searchsorted(csum, EXCESS_KEEP * csum[-1])) + 1 if len(idx) else 0
    return idx[: min(k, MAX_ENTRIES)]


def displacement_classes(
    q: np.ndarray, y: np.ndarray, cm: np.ndarray, info: dict, use_mz_cost: bool = False,
) -> dict[str, float]:
    """Mass per displacement class for predictor ``q`` against target ``y`` (both unit L1).

    Args:
        q: Predicted mass per candidate ``(n_cand,)``.
        y: Target mass per candidate ``(n_cand,)``.
        cm: Candidate m/z ``(n_cand,)``.
        info: Per-candidate ``v`` (formulas), ``prec`` and ``amb`` flags.
        use_mz_cost: Use the class-agnostic ``|delta m/z|`` cost instead of the class order.

    Returns:
        ``D`` (total displaced mass, the TV distance), one entry per class holding the mass it
        carries, ``unres`` (pruned mass), and ``down``/``up`` for nested pairs: mass whose donor is
        heavier (stuck upstream) or lighter than its recipient.
    """
    ex, de = np.maximum(q - y, 0.0), np.maximum(y - q, 0.0)
    out = {"D": float(ex.sum())}
    i, j = _keep(ex), _keep(de)
    zero = dict.fromkeys(CLASSES, 0.0) | {"unres": out["D"], "down": 0.0, "up": 0.0}
    if len(i) == 0 or len(j) == 0:
        return out | zero
    cls = pair_classes(info["v"][i], info["v"][j], info["prec"][i], info["prec"][j],
                       info["amb"][i], info["amb"][j])
    dmz = np.abs(cm[i][:, None] - cm[j][None, :])
    cost = dmz / 1000.0 if use_mz_cost else CLASS_COST[cls] + MZ_TIE * dmz
    flow = transport(ex[i], de[j], cost)
    for k, name in enumerate(CLASSES):
        out[name] = float(flow[cls == k].sum())
    out["unres"] = out["D"] - float(flow.sum())
    heavier = cm[i][:, None] > cm[j][None, :]
    nested_like = np.isin(cls, [C_PREC, C_NL1, C_NESTED])
    out["down"] = float(flow[nested_like & heavier].sum())
    out["up"] = float(flow[nested_like & ~heavier].sum())
    return out


def _load_run(fp: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    with h5py.File(fp, "r") as f:
        idx = pd.DataFrame({"s": f["start_points"][:], "e": f["end_points"][:]}, index=f["spec_id"][:])
        return idx, f["pred_mzs"][:], f["pred_ints"][:]


def candidate_space(tm: np.ndarray, ti: np.ndarray, preds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Union candidate space for several runs' predictions of one spectrum.

    Args:
        tm, ti: Target peak m/z and intensity.
        preds: Per run ``(mzs, ints)`` over its own candidates.

    Returns:
        ``cm`` union m/z, ``P`` list of per-run unit-L1 vectors over ``cm``, ``T`` target
        intensity mapped to candidates, ``t_all`` the L2 norm of every target peak, and ``near``/
        ``covered`` for the peak mapping.
    """
    cm, inv = np.unique(np.round(np.concatenate([m for m, _ in preds]), 5), return_inverse=True)
    P, s = [], 0
    for m, it in preds:
        p = np.bincount(inv[s: s + len(m)], weights=it, minlength=len(cm))
        P.append(p / p.sum())
        s += len(m)
    j = np.clip(np.searchsorted(cm, tm), 1, len(cm) - 1)
    near = np.where(np.abs(tm - cm[j - 1]) <= np.abs(tm - cm[j]), j - 1, j)
    covered = np.abs(cm[near] - tm) <= tol(tm)
    T = np.bincount(near[covered], weights=ti[covered], minlength=len(cm))
    return {"cm": cm, "P": P, "T": T, "t_all": float(np.linalg.norm(ti)), "near": near, "covered": covered}


def bias_record(sp: dict) -> dict[str, float]:
    """Shared and run-specific displacement for two runs, in cosine units."""
    t = sp["T"] / sp["t_all"]
    e = [p / np.linalg.norm(p) - t for p in sp["P"]]
    e_bar, d = (e[0] + e[1]) / 2.0, (e[0] - e[1]) / 2.0
    return {
        "cos_a": float((sp["P"][0] / np.linalg.norm(sp["P"][0])) @ t),
        "cos_b": float((sp["P"][1] / np.linalg.norm(sp["P"][1])) @ t),
        "disp_a": float(e[0] @ e[0]) / 2.0, "disp_b": float(e[1] @ e[1]) / 2.0,
        "e_bar2": float(e_bar @ e_bar) / 2.0, "d2": float(d @ d) / 2.0,
        "coverage": (1.0 - float(t @ t)) / 2.0,
    }


def _prec_ion(formula: str) -> np.ndarray:
    return formula_vec(formula) + np.eye(len(EL), dtype=np.int64)[H_IDX]


def _spectrum_inputs(tg: pd.DataFrame, sid: int, runs: list) -> tuple:
    pk = np.asarray(tg.at[sid, "peaks"], float).reshape(-1, 2)
    preds = [(m[idx.at[sid, "s"]: idx.at[sid, "e"]], it[idx.at[sid, "s"]: idx.at[sid, "e"]].astype(float))
             for idx, m, it in runs]
    return pk[:, 0], pk[:, 1], preds


def _info(cm: np.ndarray, prec_ion: np.ndarray, prec_mz: float) -> dict:
    v, amb = assign_formulas(prec_ion, cm)
    return {"v": v, "amb": amb, "prec": np.abs(cm - prec_mz) <= tol(prec_mz)}


def _chunk_bias(sids, tg, runs):
    out = []
    for sid in sids:
        tm, ti, preds = _spectrum_inputs(tg, sid, runs)
        out.append(bias_record(candidate_space(tm, ti, preds)) | {"spec_id": sid})
    return out


def _chunk_transport(sids, tg, runs, formulas, seed):
    rng = np.random.default_rng(seed)
    out = []
    for sid in sids:
        tm, ti, preds = _spectrum_inputs(tg, sid, runs)
        sp = candidate_space(tm, ti, preds)
        if sp["T"].sum() <= 0:
            continue
        y = sp["T"] / sp["T"].sum()
        info = _info(sp["cm"], _prec_ion(formulas[tg.at[sid, "mol_id"]]), float(tg.at[sid, "prec_mz"]))
        qs = {"a": sp["P"][0]}
        if len(sp["P"]) > 1:
            qs |= {"b": sp["P"][1], "mean": (sp["P"][0] + sp["P"][1]) / 2.0}
        qs["chance"] = rng.permutation(sp["P"][0])
        # Stronger null: keep how much mass sits on target-supported candidates, shuffle the
        # order within the supported set and within the rest. Real peaks are themselves linked
        # by common losses, so this separates "wrong neighbour" from "wrong order".
        qs["chance_support"] = sp["P"][0].copy()
        for grp in (np.flatnonzero(sp["T"] > 0), np.flatnonzero(sp["T"] <= 0)):
            qs["chance_support"][grp] = rng.permutation(qs["chance_support"][grp])
        rec = {"spec_id": sid}
        for name, q in qs.items():
            for k, v in displacement_classes(q, y, sp["cm"], info).items():
                rec[f"{name}_{k}"] = v
        for name in ("a", "chance"):
            for k, v in displacement_classes(qs[name], y, sp["cm"], info, use_mz_cost=True).items():
                rec[f"{name}_mz_{k}"] = v
        if len(sp["P"]) > 1:
            # Where the runs disagree: A's excess over B travels to B's excess over A.
            for k, v in displacement_classes(sp["P"][0], sp["P"][1], sp["cm"], info).items():
                rec[f"ab_{k}"] = v
        # Base-peak miss anatomy (run A): relation of the predicted top to the true base peak.
        b = int(np.argmax(ti))
        top = int(np.argmax(sp["P"][0]))
        rec["base_covered"] = bool(sp["covered"][b])
        rec["base_hit"] = bool(abs(sp["cm"][top] - tm[b]) <= tol(tm[b]))
        if rec["base_covered"] and not rec["base_hit"]:
            nb = int(sp["near"][b])
            true = np.flatnonzero(sp["T"] > 0)
            others = true[true != nb]
            refs = {"miss": top, "miss_chance": int(rng.choice(np.delete(np.arange(len(sp["cm"])), nb)))}
            if len(others):
                # Support-preserving references: an intensity-weighted random other true peak,
                # and the true second-highest peak.
                refs["miss_wrand"] = int(rng.choice(others, p=sp["T"][others] / sp["T"][others].sum()))
                refs["miss_second"] = int(others[np.argmax(sp["T"][others])])
            for tag, k in refs.items():
                c = pair_classes(info["v"][[k]], info["v"][[nb]], info["prec"][[k]], info["prec"][[nb]],
                                 info["amb"][[k]], info["amb"][[nb]])[0, 0]
                rec[f"{tag}_class"] = CLASSES[c]
            rec["miss_top_is_true"] = bool(sp["T"][top] > 0)
            rec["miss_top_true_rank"] = int((sp["T"] > sp["T"][top]).sum()) + 1 if sp["T"][top] > 0 else -1
            rec["miss_loss"] = _loss_str(info["v"][top], info["v"][nb])
        out.append(rec)
    return out


def _loss_str(va: np.ndarray, vb: np.ndarray) -> str:
    """Signed formula difference ``va - vb`` as a string (positive: ``va`` heavier)."""
    d = va - vb
    sgn = "+" if d @ MV >= 0 else "-"
    d = np.abs(d) if sgn == "+" else -d
    if (d < 0).any():
        return "mixed"
    return sgn + "".join(f"{e}{n if n > 1 else ''}" for e, n in zip(EL, d) if n > 0)


def _chunk_replicates(pairs, tg, runs):
    out = []
    for sa, sb in pairs:
        tma, tia, preds = _spectrum_inputs(tg, sa, runs)
        tmb, tib, _ = _spectrum_inputs(tg, sb, runs)
        sp = candidate_space(tma, tia, preds)
        cm = sp["cm"]
        j = np.clip(np.searchsorted(cm, tmb), 1, len(cm) - 1)
        near = np.where(np.abs(tmb - cm[j - 1]) <= np.abs(tmb - cm[j]), j - 1, j)
        cov = np.abs(cm[near] - tmb) <= tol(tmb)
        tb = np.bincount(near[cov], weights=tib[cov], minlength=len(cm)) / np.linalg.norm(tib)
        ta = sp["T"] / sp["t_all"]
        out.append({"spec_id": sa, "spec_id_b": sb, "tau2": float((ta - tb) @ (ta - tb)) / 2.0})
    return out


def _chunk_precursor(sids, tg, runs, formulas, nce_max):
    out = []
    for sid in sids:
        nce = _nce(tg.at[sid, "nce"])
        if nce > nce_max:
            continue
        tm, ti, preds = _spectrum_inputs(tg, sid, runs)
        sp = candidate_space(tm, ti, preds)
        prec_mz = float(tg.at[sid, "prec_mz"])
        pi = np.flatnonzero(np.abs(sp["cm"] - prec_mz) <= tol(prec_mz))
        if len(pi) == 0:
            continue
        pi = int(pi[np.argmax(sp["T"][pi])]) if sp["T"][pi].max() > 0 else int(pi[0])
        tn = sp["T"].copy()
        tn[sp["cm"] >= prec_mz - tol(prec_mz)] = -1.0
        c = int(np.argmax(tn))
        if tn[c] <= 0:
            continue
        prec_ion = _prec_ion(formulas[tg.at[sid, "mol_id"]])
        vc = assign_formulas(prec_ion, sp["cm"][[c]])[0][0]
        rec = {"spec_id": sid, "mol_id": tg.at[sid, "mol_id"], "nce": nce, "prec_mz": prec_mz,
               "y_prec_wins": bool(sp["T"][pi] > sp["T"][c]),
               "t_logratio": float(np.log((sp["T"][pi] + 1e-6) / sp["T"][c])),
               "contender_mz": float(sp["cm"][c])}
        for k, p in enumerate(sp["P"]):
            rec[f"m{k}_logratio"] = float(np.log((p[pi] + 1e-12) / (p[c] + 1e-12)))
        loss = prec_ion - vc
        for e, n in zip(EL, loss):
            rec[f"nl_{e}"] = int(n)
        for e, n in zip(EL, prec_ion):
            rec[f"p_{e}"] = int(n)
        out.append(rec)
    return out


def _nce(x) -> float:
    return float(np.mean(x)) if hasattr(x, "__len__") else float(x)


def _load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, list, pd.Index]:
    tg = pd.read_pickle(args.targets_fp).set_index("spec_id")
    runs = [_load_run(fp) for fp in args.pred_hdf5]
    ids = tg.index
    for idx, _, _ in runs:
        ids = ids.intersection(idx.index)
    ids = ids.sort_values()
    if args.limit:
        ids = pd.Index(np.random.default_rng(0).choice(ids.to_numpy(), args.limit, replace=False)).sort_values()
    LOG.info("%d spectra, %d runs", len(ids), len(runs))
    return tg, runs, ids


def _parallel(fn, items, n_jobs, *extra) -> list[dict]:
    chunks = np.array_split(np.asarray(items, dtype=object if np.ndim(items) > 1 else None), n_jobs * 8)
    parts = Parallel(n_jobs=n_jobs)(delayed(fn)(c, *extra) for c in chunks if len(c))
    return [r for p in parts for r in p]


def _add_meta(df: pd.DataFrame, tg: pd.DataFrame) -> pd.DataFrame:
    df["mol_id"] = tg.loc[df.index, "mol_id"].to_numpy()
    df["nce"] = tg.loc[df.index, "nce"].map(_nce).to_numpy()
    return df


def run_stage(args: argparse.Namespace) -> None:
    """Run one per-spectrum stage and pickle its table."""
    tg, runs, ids = _load_inputs(args)
    formulas = pd.read_pickle(args.mol_df_fp).set_index("mol_id")["formula"]
    if args.stage == "bias":
        if len(runs) != 2:
            raise ValueError("bias needs exactly two --pred_hdf5 files")
        df = pd.DataFrame(_parallel(_chunk_bias, ids, args.n_jobs, tg, runs)).set_index("spec_id")
        df = _add_meta(df, tg)
        # Replicate floor: pairs of spectra of one molecule at the same NCE.
        sub = tg.loc[ids, ["mol_id"]].assign(nce=tg.loc[ids, "nce"].map(_nce))
        pairs = [tuple(g.index[:2]) for _, g in sub.groupby(["mol_id", "nce"]) if len(g) >= 2]
        rep = pd.DataFrame(_parallel(_chunk_replicates, pairs, args.n_jobs, tg, runs))
        rep.to_pickle(str(Path(args.out_fp).with_suffix("")) + "_replicates.pkl")
        LOG.info("%d replicate pairs", len(rep))
    elif args.stage == "transport":
        parts = Parallel(n_jobs=args.n_jobs)(
            delayed(_chunk_transport)(c, tg, runs, formulas, s)
            for s, c in enumerate(np.array_split(ids.to_numpy(), args.n_jobs * 8)) if len(c))
        df = _add_meta(pd.DataFrame([r for p in parts for r in p]).set_index("spec_id"), tg)
    elif args.stage == "precursor":
        df = pd.DataFrame(_parallel(_chunk_precursor, ids, args.n_jobs, tg, runs, formulas, args.nce_max))
        df = df.set_index("spec_id")
    else:
        raise ValueError(args.stage)
    df.to_pickle(args.out_fp)
    LOG.info("wrote %s (%d rows)", args.out_fp, len(df))


NCE_EDGES = [-1, 20, 30, 40, 60, 1e9]
NCE_LABELS = ["<=20", "20-30", "30-40", "40-60", ">60"]
NOVELTY_EDGES = [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001]


def summarize_bias(fp: str, novelty: pd.Series | None) -> dict:
    """Shared and run-specific displacement, overall, by NCE and by novelty, plus the floor."""
    d = pd.read_pickle(fp)
    d["disp"] = (d.disp_a + d.disp_b) / 2.0
    d["shared"] = d.e_bar2 - d.d2
    d["runspec"] = 2.0 * d.d2
    rep = pd.read_pickle(str(Path(fp).with_suffix("")) + "_replicates.pkl")
    sub = d.loc[rep.spec_id]
    floor = float((rep.tau2 / 2.0).mean())
    out = {
        "n": len(d), "cos_a": d.cos_a.mean(), "cos_b": d.cos_b.mean(), "coverage": d.coverage.mean(),
        "disp": d.disp.mean(), "shared": d.shared.mean(), "runspec": d.runspec.mean(),
        "replicate_pairs": len(rep), "replicate_disp": sub.disp.mean(),
        "replicate_shared": sub.shared.mean(), "target_noise_floor": floor,
    }
    out["shared_frac"] = out["shared"] / out["disp"]
    out["reducible_shared_frac"] = (out["replicate_shared"] - floor) / out["replicate_disp"]
    cols = ["disp", "shared", "runspec"]
    by_nce = d.groupby(pd.cut(d.nce, NCE_EDGES, labels=NCE_LABELS), observed=True)[cols].mean()
    out["by_nce"] = (by_nce.assign(shared_frac=by_nce.shared / by_nce.disp)).round(4).to_dict("index")
    if novelty is not None:
        nov = pd.cut(d.mol_id.map(novelty), NOVELTY_EDGES)
        by_nov = d.groupby(nov, observed=True)[cols].mean()
        out["by_novelty"] = {str(k): v for k, v in
                             by_nov.assign(shared_frac=by_nov.shared / by_nov.disp).round(4).to_dict("index").items()}
    return out


def _class_shares(g: pd.DataFrame, pred: str) -> dict[str, float]:
    return {c: float(g[f"{pred}_{c}"].sum() / g[f"{pred}_D"].sum()) for c in CLASSES + ["unres"]}


def summarize_transport(fp: str, novelty: pd.Series | None) -> dict:
    """Class shares of displaced mass, enrichment over chance, strata and base-peak misses."""
    d = pd.read_pickle(fp)
    preds = [p for p in ("a", "b", "mean", "ab", "chance", "chance_support", "a_mz", "chance_mz")
             if f"{p}_D" in d]
    out = {"n": len(d), "D": {p: float(d[f"{p}_D"].mean()) for p in preds},
           "shares": {p: _class_shares(d, p) for p in preds}}
    ch, ch_mz = out["shares"]["chance"], out["shares"]["chance_mz"]
    out["enrichment"] = {p: {c: out["shares"][p][c] / ch[c] for c in CLASSES if ch[c] > 0}
                         for p in preds if p in ("a", "b", "mean", "ab")}
    out["enrichment"]["a_mz"] = {c: out["shares"]["a_mz"][c] / ch_mz[c] for c in CLASSES if ch_mz[c] > 0}
    if "chance_support" in out["shares"]:
        cs = out["shares"]["chance_support"]
        out["enrichment_vs_support_null"] = {
            p: {c: out["shares"][p][c] / cs[c] for c in CLASSES if cs[c] > 0}
            for p in ("a", "b", "mean") if p in out["shares"]}
    nested_like = d.a_down.sum() + d.a_up.sum()
    out["down_share"] = {p: float(d[f"{p}_down"].sum() / (d[f"{p}_down"].sum() + d[f"{p}_up"].sum()))
                         for p in ("a", "mean", "chance", "chance_support") if nested_like > 0 and f"{p}_D" in d}
    rep = "mean" if "mean_D" in d else "a"
    strata = {"nce": pd.cut(d.nce, NCE_EDGES, labels=NCE_LABELS)}
    if novelty is not None and d.mol_id.isin(novelty.index).any():
        strata["novelty"] = pd.cut(d.mol_id.map(novelty), NOVELTY_EDGES)
    for name, key in strata.items():
        out[f"by_{name}"] = {}
        for k, g in d.groupby(key, observed=True):
            s, c = _class_shares(g, rep), _class_shares(g, "chance")
            m = g[g.base_covered & ~g.base_hit]
            out[f"by_{name}"][str(k)] = {
                "n": len(g), "D": float(g[f"{rep}_D"].mean()),
                **{cl: s[cl] for cl in CLASSES}, **{f"{cl}_x": s[cl] / c[cl] for cl in CLASSES if c[cl] > 0},
                "down": float(g[f"{rep}_down"].sum() / (g[f"{rep}_down"].sum() + g[f"{rep}_up"].sum())),
                **({"down_support_null": float(g.chance_support_down.sum()
                                               / (g.chance_support_down.sum() + g.chance_support_up.sum()))}
                   if "chance_support_down" in g else {}),
                "base_miss": len(m) / len(g),
                **{f"miss_{cl}": float((m.miss_class == cl).mean()) for cl in ("prec", "nl1", "hshift")},
            }
    m = d[d.base_covered & ~d.base_hit]
    out["base_miss"] = {"n": len(m), "frac": len(m) / len(d),
                        "model_top": m.miss_class.value_counts(normalize=True).to_dict(),
                        "random_candidate": m.miss_chance_class.value_counts(normalize=True).to_dict(),
                        **{ref: m[f"miss_{ref}_class"].value_counts(normalize=True).to_dict()
                           for ref in ("wrand", "second") if f"miss_{ref}_class" in m},
                        "top_losses": m.miss_loss.value_counts().head(20).to_dict()}
    return out


def _precursor_features(d: pd.DataFrame) -> pd.DataFrame:
    """Formula-only features of a precursor/contender pair: loss and precursor composition, NCE."""
    nl = d[[f"nl_{e}" for e in EL]].to_numpy()
    pr = d[[f"p_{e}" for e in EL]].to_numpy()
    x = pd.DataFrame(np.hstack([nl, pr]), index=d.index, columns=[f"nl_{e}" for e in EL] + [f"p_{e}" for e in EL])
    x["loss_mass"] = nl @ MV
    x["loss_frac"] = x.loss_mass / (pr @ MV)
    x["nce"] = d.nce
    x["prec_mz"] = d.prec_mz
    return x


def summarize_precursor(train_fp: str, eval_fps: list[str]) -> dict:
    """Can formula-level inputs call precursor versus contender better than the model?

    A gradient-boosted classifier on train pairs predicts whether the precursor beats the
    strongest covered non-precursor peak. It is compared on held-out molecules with the model's
    implied winner and with the per-NCE-bin population rate.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    tr = pd.read_pickle(train_fp)
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)
    clf.fit(_precursor_features(tr), tr.y_prec_wins)
    bins = pd.cut(tr.nce, [-1, 10, 15, 20, 25, 30])
    rate = tr.groupby(bins, observed=True).y_prec_wins.mean()
    out = {"train_pairs": len(tr), "train_model_acc": float(((tr.m0_logratio > 0) == tr.y_prec_wins).mean())}
    for fp in eval_fps:
        ev = pd.read_pickle(fp)
        y = ev.y_prec_wins.to_numpy()
        p = clf.predict_proba(_precursor_features(ev))[:, 1]
        prior = pd.cut(ev.nce, [-1, 10, 15, 20, 25, 30]).map(rate).astype(float).to_numpy()
        r = {"pairs": len(ev), "prec_wins_rate": float(y.mean()),
             "formula_acc": float(((p > 0.5) == y).mean()), "formula_auc": float(roc_auc_score(y, p)),
             "prior_acc": float(((prior > 0.5) == y).mean())}
        for k in (0, 1):
            col = f"m{k}_logratio"
            if col in ev:
                r[f"model{k}_acc"] = float(((ev[col] > 0) == y).mean())
                r[f"model{k}_auc"] = float(roc_auc_score(y, ev[col]))
                for tag, mask in (("prec_wins", y), ("frag_wins", ~y)):
                    r[f"model{k}_acc_{tag}"] = float(((ev[col][mask] > 0) == y[mask]).mean())
        for tag, mask in (("prec_wins", y), ("frag_wins", ~y)):
            r[f"formula_acc_{tag}"] = float(((p[mask] > 0.5) == y[mask]).mean())
        out[Path(fp).stem] = r
    out["stack"] = _stack_precursor(clf, [pd.read_pickle(fp) for fp in eval_fps], [Path(fp).stem for fp in eval_fps])
    return out


def _stack_precursor(clf, evals: list[pd.DataFrame], names: list[str]) -> dict:
    """Is the formula learner complementary to the model on precursor-vs-contender?

    A logistic stack of the model's log-ratio and the learner's logit is fitted on half of the
    first eval split's molecules and scored on the other half and on every other split. The
    model-only baseline is the same logistic fit on the model's log-ratio alone.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    def feats(d: pd.DataFrame) -> np.ndarray:
        p = np.clip(clf.predict_proba(_precursor_features(d))[:, 1], 1e-4, 1 - 1e-4)
        return np.column_stack([d.m0_logratio.clip(-30, 30), np.log(p / (1 - p))])

    first = evals[0]
    mols = first.mol_id.unique()
    half = set(np.random.default_rng(0).choice(mols, len(mols) // 2, replace=False))
    fit = first[first.mol_id.isin(half)]
    both = LogisticRegression().fit(feats(fit), fit.y_prec_wins)
    model = LogisticRegression().fit(feats(fit)[:, :1], fit.y_prec_wins)
    out = {}
    held = [(f"{names[0]}_other_half", first[~first.mol_id.isin(half)])] + list(zip(names[1:], evals[1:]))
    for name, d in held:
        y = d.y_prec_wins.to_numpy()
        s_both, s_model = both.predict_proba(feats(d))[:, 1], model.predict_proba(feats(d)[:, :1])[:, 1]
        out[name] = {
            "model_auc": float(roc_auc_score(y, d.m0_logratio)), "stack_auc": float(roc_auc_score(y, s_both)),
            "model_acc": float(((s_model > 0.5) == y).mean()), "stack_acc": float(((s_both > 0.5) == y).mean()),
            "model_acc_frag_wins": float(((s_model[~y] > 0.5) == y[~y]).mean()),
            "stack_acc_frag_wins": float(((s_both[~y] > 0.5) == y[~y]).mean()),
        }
    return out


def run_summary(args: argparse.Namespace) -> None:
    """Write the Phase 0 gate tables from the per-spectrum pickles in ``--in_dp``."""
    import json

    in_dp = Path(args.in_dp)
    novelty = None
    if args.novelty_fp:
        novelty = pd.read_pickle(args.novelty_fp).groupby("mol_id").nn_sim.first()
    out = {}
    for fp in sorted(in_dp.glob("bias_*.pkl")):
        if not fp.stem.endswith("_replicates"):
            out[fp.stem] = summarize_bias(str(fp), novelty)
    for fp in sorted(in_dp.glob("transport_*.pkl")):
        out[fp.stem] = summarize_transport(str(fp), novelty)
    train = sorted(in_dp.glob("precursor_*train*.pkl"))
    if train:
        # Val first: the stack is fitted on half of the first split's molecules.
        evals = sorted((str(f) for f in in_dp.glob("precursor_*.pkl") if "train" not in f.stem),
                       key=lambda f: ("_val" not in f, f))
        out["precursor"] = summarize_precursor(str(train[0]), evals)
    with open(in_dp / "summary.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    LOG.info("wrote %s", in_dp / "summary.json")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="stage", required=True)
    s = sub.add_parser("summary")
    s.add_argument("--in_dp", required=True)
    s.add_argument("--novelty_fp", default=None, help="Pickle with mol_id and nn_sim (val molecules).")
    for name in ("bias", "transport", "precursor"):
        s = sub.add_parser(name)
        _stage_args(s)
    return p.parse_args()


def _stage_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--targets_fp", required=True)
    p.add_argument("--pred_hdf5", nargs="+", required=True)
    p.add_argument("--mol_df_fp", default="data/proc/nist20/mol_df.pkl")
    p.add_argument("--out_fp", required=True)
    p.add_argument("--n_jobs", type=int, default=10)
    p.add_argument("--limit", type=int, default=0, help="Random subsample of spectra (0 = all).")
    p.add_argument("--nce_max", type=float, default=30.0, help="precursor stage: NCE cut-off.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    cli = get_args()
    if cli.stage == "summary":
        run_summary(cli)
    else:
        run_stage(cli)
