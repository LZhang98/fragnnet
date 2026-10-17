"""Can an identity-blind, bond-local learner order a novel molecule's bonds better than MASSter?

Gate for arm 2A of ``docs/plans/masster/masster_intensity_displacement_plan.md``. The
bond-type check found 59% of the val-train bond-level gap between bond types inside a
molecule, 26% within a type and 16% at the precursor. 2A adds a local lability prior to the
fragment score; this script asks, offline, whether such a prior carries information the model
lacks on molecules it has not seen.

Input is the output of ``bond_type_ce_allocation.py --bond_records_fp ... --fragment_level``:
per spectrum, every candidate bond's model mass ``p`` and target mass under the model-free
equal split (``t_eq``, the training target) and the responsibility split (``t_resp``), plus
the retained fragments' cut bonds and channels with their m/z groups and model mass.

Learners see only local chemistry around each bond (bond order, ring context, endpoint atoms,
element counts within two bonds, the formula of the piece a non-ring cut releases), molecule
size, NCE and the candidate count; never the molecule's identity or the model's output. They
are fitted on train molecules with 10% of them held out for early stopping:

- ``gbdt``: HistGradientBoosting, Poisson loss on the bond's share (log link = log-share score);
- ``mlp``: 3-layer MLP trained with the model's own listwise cross-entropy per spectrum;
- ``gbdt_type``: GBDT on bond type, NCE and candidate count only (the ICEBERG-level response).

Primary read, peak level (the form 2A would take). Each fragment is re-scored as
``a log w_f + c mean_b log p_b + sum_k b_k mean_b S_k(b)`` over its cut bonds (the precursor
uses its own row), renormalized per spectrum, split over ion/H channels as the model does and
summed into m/z groups. Weights are fitted by group-level cross-entropy on 40% of the other
split's molecules (val and test share no molecules) and scored by group cosine on every
spectrum. Controls: ``model_T`` (temperature
only, which includes the free ``P^0.75``) and ``stack_logp`` (the model's own bond-share
aggregate, no learner). Gains carry molecule-clustered standard errors.

Secondary read, bond level: pairwise ordering accuracy (cross-type, same-type,
precursor-vs-bond) and bond/type/within TV, with the same cross-fit. Isomeric fragments share
an m/z, so the target's bond attribution is a convention there (``t_eq`` gives 0.25 bond TV on
TRAIN against 0.07 for ``t_resp``); these tables are descriptive, not the gate.

Pre-registered gate (plan §5, 1.4b; moved from bond level to peak level after the train-split
isomer check, before any val/test number was seen): PASS if ``stack_all`` beats both
``model_T`` and ``stack_logp`` by at least +0.005 group cosine on BOTH val and test, with the
gain on novel molecules (nn_sim < 0.5) no smaller than on familiar ones. FAIL if the gain over
``model_T`` is below +0.002 on either split.

Usage (repository root):

    python scripts/debug/masster_memorization/bond_local_learner.py \
        --train_fp .../bond_local/bonds_train.pkl --eval_fp .../bond_local/bonds_valtest.pkl \
        --train_mols_fp output/cross_eval_mh_ft_clean_2026_09_23/targets/targets_filtered_train.pkl \
        --out_dp .../bond_local
"""

from __future__ import annotations

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

LOG = logging.getLogger("bond_local_learner")
ORDER_CHAR = {1.0: "-", 2.0: "=", 3.0: "#", 1.5: ":"}
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
HALOGENS = {"F", "Cl", "Br", "I"}
PRECURSOR = "precursor"
NCE_EDGES = [-1, 20, 40, 60, 1e9]
NCE_LABELS = ["<=20", "20-40", "40-60", ">60"]
NOV_EDGES = [0, 0.4, 0.5, 0.6, 0.8, 1.0001]
PAIR_CLASSES = ("cross_type", "same_type", "prec_vs_bond")
EPS = 1e-12


def _atom_features(atom: Chem.Atom, exclude: int) -> dict[str, float]:
    """Endpoint descriptors; ``exclude`` is the other endpoint of the bond."""
    nbrs = [n for n in atom.GetNeighbors() if n.GetIdx() != exclude]
    has_do = any(
        b.GetBondTypeAsDouble() == 2.0 and b.GetOtherAtom(atom).GetSymbol() == "O"
        for b in atom.GetBonds() if b.GetOtherAtomIdx(atom.GetIdx()) != exclude
    )
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization()
    return {
        **{f"is_{e}": float(sym == e) for e in ("C", "N", "O", "S", "P")},
        "is_hal": float(sym in HALOGENS),
        "deg": float(atom.GetDegree()), "nH": float(atom.GetTotalNumHs()),
        "charge": float(atom.GetFormalCharge()), "aro": float(atom.GetIsAromatic()),
        "sp": float(hyb == Chem.HybridizationType.SP), "sp2": float(hyb == Chem.HybridizationType.SP2),
        "sp3": float(hyb == Chem.HybridizationType.SP3), "ring": float(atom.IsInRing()),
        "n_het_nbr": float(sum(n.GetSymbol() not in ("C", "H") for n in nbrs)),
        "has_dO": float(has_do), "aro_nbr": float(any(n.GetIsAromatic() for n in nbrs)),
    }


def _side(mol: Chem.Mol, start: int, blocked: int) -> list[int]:
    """Atoms reachable from ``start`` without crossing to ``blocked`` through the cut bond."""
    seen, stack = {start}, [start]
    while stack:
        a = stack.pop()
        for n in mol.GetAtomWithIdx(a).GetNeighbors():
            j = n.GetIdx()
            if j not in seen and not (a == start and j == blocked):
                seen.add(j)
                stack.append(j)
    return sorted(seen)


def bond_label(bond: Chem.Bond) -> str:
    """Element pair and order, as in ``bond_type_ce_allocation.bond_labels``."""
    ea, eb = sorted((bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()))
    return f"{ea}{ORDER_CHAR.get(bond.GetBondTypeAsDouble(), '?')}{eb}"


def bond_features(smiles: str) -> tuple[pd.DataFrame, list[str]]:
    """Local descriptors per RDKit bond index, plus a precursor row at index ``-1``.

    Returns:
        The feature table (index = bond index) and the type label per row, same order.
    """
    mol = Chem.MolFromSmiles(smiles)
    dist = Chem.GetDistanceMatrix(mol)
    ri = mol.GetRingInfo()
    n_heavy = mol.GetNumAtoms()
    masses = np.array([a.GetMass() + a.GetTotalNumHs() * 1.008 for a in mol.GetAtoms()])
    mw = float(masses.sum())
    rows, labels = {}, []
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtom(), bond.GetEndAtom()
        fa, fb = _atom_features(a, b.GetIdx()), _atom_features(b, a.GetIdx())
        # Canonical endpoint order, so the features do not depend on RDKit's begin/end atom.
        hi, lo = (fa, fb) if tuple(fa.values()) <= tuple(fb.values()) else (fb, fa)
        rings = [r for r in ri.BondRings() if bond.GetIdx() in r]
        feat = {
            "is_precursor": 0.0, "order": bond.GetBondTypeAsDouble(), "aromatic": float(bond.GetIsAromatic()),
            "conj": float(bond.GetIsConjugated()), "in_ring": float(bond.IsInRing()),
            "ring_min": float(min((len(r) for r in rings), default=0)), "n_rings": float(len(rings)),
            **{f"hi_{k}": v for k, v in hi.items()}, **{f"lo_{k}": v for k, v in lo.items()},
        }
        d = np.minimum(dist[a.GetIdx()], dist[b.GetIdx()])
        for r in (1, 2):
            env = np.flatnonzero(d == r)
            syms = [mol.GetAtomWithIdx(int(i)).GetSymbol() for i in env]
            for e in ELEMENTS:
                feat[f"d{r}_{e}"] = float(syms.count(e))
            feat[f"d{r}_aro"] = float(sum(mol.GetAtomWithIdx(int(i)).GetIsAromatic() for i in env))
        if not bond.IsInRing():
            side_a = _side(mol, a.GetIdx(), b.GetIdx())
            side_b = [i for i in range(n_heavy) if i not in set(side_a)]
            small = side_a if masses[side_a].sum() <= masses[side_b].sum() else side_b
            syms = [mol.GetAtomWithIdx(i).GetSymbol() for i in small]
            feat.update({
                "small_heavy": float(len(small)), "large_heavy": float(n_heavy - len(small)),
                "small_frac": float(masses[small].sum() / mw),
                "small_H": float(sum(mol.GetAtomWithIdx(i).GetTotalNumHs() for i in small)),
                **{f"small_{e}": float(syms.count(e)) for e in ("C", "N", "O", "S", "P")},
                "small_hal": float(sum(s in HALOGENS for s in syms)),
            })
        rows[bond.GetIdx()] = feat
        labels.append(bond_label(bond))
    rows[-1] = {"is_precursor": 1.0}
    labels.append(PRECURSOR)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["n_heavy"] = float(n_heavy)
    df["mw"] = mw
    return df, labels


def build_table(recs: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Long table (one row per spectrum-bond, contiguous per spectrum) with features.

    Returns:
        The table, per-row spectrum index into ``recs``, and the type label per row.
    """
    cache: dict[str, tuple[pd.DataFrame, dict[int, str]]] = {}
    parts, labels = [], []
    uniq = recs.smiles.unique()
    with ProcessPoolExecutor(max_workers=8) as pool:
        for smi, (df, lab) in zip(uniq, pool.map(bond_features, uniq, chunksize=64)):
            cache[smi] = (df, dict(zip(df.index, lab)))
    for i, r in enumerate(recs.itertuples()):
        df, lab = cache[r.smiles]
        part = df.loc[r.bond_idx].reset_index(drop=True)
        part["nce"] = r.nce
        part["n_cand"] = float(len(r.bond_idx))
        part["spec"] = i
        part["p"] = r.p
        part["t_eq"] = r.t_eq
        part["t_resp"] = r.t_resp
        parts.append(part)
        labels += [lab[int(b)] for b in r.bond_idx]
    tab = pd.concat(parts, ignore_index=True)
    return tab, tab.spec.to_numpy(), labels


def novelty(smiles: pd.Series, train_mols_fp: str) -> pd.Series:
    """Max Tanimoto (Morgan r=2, 2048 bits) of each SMILES to the train molecules."""
    train_ids = pd.read_pickle(train_mols_fp).mol_id.unique()
    mol = pd.read_pickle("data/proc/nist20/mol_df.pkl").set_index("mol_id")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    tr = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in mol.loc[train_ids, "smiles"]]
    uniq = smiles.unique()
    nn = {s: max(DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(Chem.MolFromSmiles(s)), tr)) for s in uniq}
    return smiles.map(nn)


# ---------------------------------------------------------------------------------------------
# Per-spectrum softmax utilities (rows contiguous per spectrum).


def _starts(spec: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.r_[True, spec[1:] != spec[:-1]])


def seg_log_softmax(s: np.ndarray, starts: np.ndarray, spec_pos: np.ndarray) -> np.ndarray:
    m = np.maximum.reduceat(s, starts)
    z = s - m[spec_pos]
    lse = np.log(np.add.reduceat(np.exp(z), starts)) + m
    return s - lse[spec_pos]


def fit_stack(cols: list[np.ndarray], t: np.ndarray, starts: np.ndarray, spec_pos: np.ndarray) -> np.ndarray:
    """Weights ``w`` minimizing the mean listwise CE of ``softmax(sum_k w_k cols_k)`` against ``t``."""
    S = np.stack(cols, 1)
    n = len(starts)

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        lq = seg_log_softmax(S @ w, starts, spec_pos)
        q = np.exp(lq)
        grad = (S * (q - t)[:, None]).sum(0) / n
        return float(-(t * lq).sum() / n), grad

    return minimize(f, np.ones(S.shape[1]) / S.shape[1], jac=True, method="L-BFGS-B").x


# ---------------------------------------------------------------------------------------------
# Learners.


def fit_gbdt(X: pd.DataFrame, y: np.ndarray, Xv: pd.DataFrame, yv: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    m = HistGradientBoostingRegressor(
        loss="poisson", learning_rate=0.05, max_iter=1000, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, early_stopping=True, n_iter_no_change=30, random_state=seed,
    )
    m.fit(X, y, X_val=Xv, y_val=yv)
    LOG.info("gbdt: %d iterations", m.n_iter_)
    return m


class _MLP(th.nn.Module):
    def __init__(self, d: int, h: int = 256) -> None:
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.Linear(d, h), th.nn.SiLU(), th.nn.Dropout(0.1),
            th.nn.Linear(h, h), th.nn.SiLU(), th.nn.Dropout(0.1),
            th.nn.Linear(h, h), th.nn.SiLU(), th.nn.Linear(h, 1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x).squeeze(-1)


def _listwise_ce(s: th.Tensor, t: th.Tensor, seg: th.Tensor, n: int) -> th.Tensor:
    m = th.full((n,), -th.inf, device=s.device).scatter_reduce(0, seg, s, "amax", include_self=True)
    z = (s - m[seg]).exp()
    lse = th.zeros(n, device=s.device).index_add_(0, seg, z).log() + m
    return -(t * (s - lse[seg])).sum() / n


class MLPLearner:
    """Listwise MLP on standardized features (NaN -> 0 after scaling)."""

    def __init__(self, cols: list[str], seed: int) -> None:
        self.cols, self.seed = cols, seed
        self.dev = th.device("cuda" if th.cuda.is_available() else "cpu")

    def _x(self, df: pd.DataFrame) -> th.Tensor:
        x = (df[self.cols].to_numpy(np.float64) - self.mu) / self.sd
        return th.tensor(np.nan_to_num(x, nan=0.0), dtype=th.float32, device=self.dev)

    def fit(self, tr: pd.DataFrame, va: pd.DataFrame, epochs: int = 40, spectra_per_batch: int = 1024) -> MLPLearner:
        th.manual_seed(self.seed)
        x = tr[self.cols].to_numpy(np.float64)
        self.mu, self.sd = np.nanmean(x, 0), np.nanstd(x, 0) + 1e-6
        net = _MLP(len(self.cols)).to(self.dev)
        opt = th.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        X, T = self._x(tr), th.tensor(tr.t_eq.to_numpy(), dtype=th.float32, device=self.dev)
        spec = tr.spec.to_numpy()
        Xv, Tv = self._x(va), th.tensor(va.t_eq.to_numpy(), dtype=th.float32, device=self.dev)
        segv = th.tensor(pd.factorize(va.spec)[0], device=self.dev)
        nv = int(segv.max()) + 1
        uniq = np.unique(spec)
        rng = np.random.default_rng(self.seed)
        best, best_state, bad = np.inf, None, 0
        for ep in range(epochs):
            net.train()
            order = rng.permutation(uniq)
            for c in range(0, len(order), spectra_per_batch):
                rows = np.flatnonzero(np.isin(spec, order[c:c + spectra_per_batch]))
                seg_np, _ = pd.factorize(spec[rows])
                seg = th.tensor(seg_np, device=self.dev)
                loss = _listwise_ce(net(X[rows]), T[rows], seg, int(seg_np.max()) + 1)
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with th.no_grad():
                vl = float(_listwise_ce(net(Xv), Tv, segv, nv))
            LOG.info("mlp epoch %d: held-out train CE %.4f", ep, vl)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 5:
                    break
        net.load_state_dict(best_state)
        self.net = net.eval()
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        with th.no_grad():
            return self.net(self._x(df)).cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------------------------
# Metrics.


def spectrum_metrics(
    scores: dict[str, np.ndarray], t: np.ndarray, labels: np.ndarray, starts: np.ndarray,
) -> pd.DataFrame:
    """Per spectrum and predictor: weighted pair-order sums by class, TV decomposition, top hit.

    ``scores`` holds per-row log-probabilities (already normalized per spectrum).
    """
    ends = np.r_[starts[1:], len(t)]
    out = []
    is_prec = labels == PRECURSOR
    for si, (a, b) in enumerate(zip(starts, ends)):
        tt, lab, pr = t[a:b], labels[a:b], is_prec[a:b]
        dt = tt[:, None] - tt[None, :]
        w = np.triu(np.abs(dt), 1)
        same = lab[:, None] == lab[None, :]
        anyp = pr[:, None] | pr[None, :]
        cls = {"cross_type": w * (~same & ~anyp), "same_type": w * (same & ~anyp), "prec_vs_bond": w * anyp}
        _, tinv = np.unique(lab, return_inverse=True)
        tk = np.bincount(tinv, weights=tt)
        rec: dict[str, float] = {"spec": si}
        for c, wc in cls.items():
            rec[f"w_{c}"] = wc.sum()
        for name, lq in scores.items():
            s = lq[a:b]
            ds = s[:, None] - s[None, :]
            corr = np.where(ds == 0, 0.5, (np.sign(ds) == np.sign(dt)).astype(float))
            for c, wc in cls.items():
                rec[f"{name}__c_{c}"] = (wc * corr).sum()
            q = np.exp(s)
            tv_bond = 0.5 * np.abs(q - tt).sum()
            tv_type = 0.5 * np.abs(np.bincount(tinv, weights=q) - tk).sum()
            rec[f"{name}__tv_bond"] = tv_bond
            rec[f"{name}__tv_type"] = tv_type
            rec[f"{name}__tv_within"] = tv_bond - tv_type
            rec[f"{name}__top_hit"] = float(np.argmax(s) == np.argmax(tt))
        out.append(rec)
    return pd.DataFrame(out)


def aggregate(m: pd.DataFrame, names: list[str]) -> dict[str, dict[str, float]]:
    res = {}
    for name in names:
        r = {f"acc_{c}": m[f"{name}__c_{c}"].sum() / m[f"w_{c}"].sum() for c in PAIR_CLASSES}
        for k in ("tv_bond", "tv_type", "tv_within", "top_hit"):
            r[k] = m[f"{name}__{k}"].mean()
        res[name] = {k: round(float(v), 4) for k, v in r.items()}
    res["_n"] = int(len(m))
    return res


def stratified(m: pd.DataFrame, names: list[str], col: str) -> dict[str, dict]:
    return {str(k): aggregate(g, names) for k, g in m.groupby(col, observed=True)}


# ---------------------------------------------------------------------------------------------
# Peak level: re-weight the model's fragments with bond scores, score at m/z-group level.


class PeakSplit:
    """Flat fragment/channel/group arrays of one split, for peak-level stacking.

    Fragment ``f`` of spectrum ``i`` gets features ``X[f] = [log w_f, mean_{b in cut(f)} S_b ...]``
    for each bond score ``S`` (per bond-table row); the intact precursor uses its own row. A
    stack scores ``s_f = X[f] @ theta``, renormalizes over the spectrum's fragments, keeps the
    model's ion/H split inside each fragment, and sums channels into their m/z groups.
    """

    def __init__(self, recs: pd.DataFrame, row_offset: np.ndarray) -> None:
        n_f = np.array([len(p) - 1 for p in recs.frag_bond_ptr])
        n_c = np.array([len(w) for w in recs.chan_w])
        n_g = np.array([len(g) for g in recs.group_t])
        f_off, c_off, g_off = (np.r_[0, np.cumsum(n)[:-1]] for n in (n_f, n_c, n_g))
        self.n_spec = len(recs)
        self.frag_spec = np.repeat(np.arange(self.n_spec), n_f)
        self.frag_starts = f_off
        self.group_spec = np.repeat(np.arange(self.n_spec), n_g)
        self.chan_frag = np.empty(n_c.sum(), dtype=np.int32)
        self.chan_group = np.empty(n_c.sum(), dtype=np.int32)
        chan_w = np.empty(n_c.sum(), dtype=np.float64)
        members, member_n, is_prec = [], [], []
        for i, r in enumerate(recs.itertuples()):
            c = slice(c_off[i], c_off[i] + n_c[i])
            self.chan_frag[c] = r.chan_frag.astype(np.int32) + f_off[i]
            self.chan_group[c] = r.chan_group.astype(np.int32) + g_off[i]
            chan_w[c] = r.chan_w
            lookup = np.full(int(r.bond_idx.max()) + 2, -1, dtype=np.int64)
            lookup[r.bond_idx] = np.arange(len(r.bond_idx)) + row_offset[i]  # -1 -> last slot
            prec = lookup[-1]
            if prec < 0:
                raise ValueError("spectrum without a precursor row")
            flat = lookup[r.frag_bond_flat.astype(np.int64)]
            if (flat < 0).any():
                raise ValueError("fragment cut bond missing from the bond table")
            # Members per fragment: its cut bonds' rows, or the precursor row if it has none.
            lens = np.diff(r.frag_bond_ptr)
            n_m = np.maximum(lens, 1)
            out_start = np.r_[0, np.cumsum(n_m)[:-1]]
            out = np.full(n_m.sum(), prec, dtype=np.int64)
            owner = np.repeat(np.arange(len(lens)), lens)
            out[out_start[owner] + np.arange(len(flat)) - r.frag_bond_ptr[owner]] = flat
            members.append(out)
            member_n.append(n_m)
            is_prec.append(lens == 0)
        self.members = np.concatenate(members)
        self.member_n = np.concatenate(member_n)
        self.is_prec = np.concatenate(is_prec).astype(np.float64)
        self.nce_bin = np.digitize(recs.nce.to_numpy(), NCE_EDGES[1:-1], right=True)[self.frag_spec]
        self.member_starts = np.r_[0, np.cumsum(self.member_n)[:-1]]
        w_f = np.bincount(self.chan_frag, weights=chan_w, minlength=len(self.frag_spec))
        self.log_wf = np.log(w_f + EPS)
        self.chan_r = chan_w / np.maximum(w_f[self.chan_frag], EPS)
        group_t = np.concatenate(recs.group_t.tolist()).astype(np.float64)
        cov = np.bincount(self.group_spec, weights=group_t, minlength=self.n_spec)
        self.group_t = group_t
        self.group_that = group_t / cov[self.group_spec]
        self.t_norm = np.sqrt(np.bincount(self.group_spec, weights=group_t**2, minlength=self.n_spec)
                              + recs.unc_t2.to_numpy())

    def column(self, base: dict[str, np.ndarray], name: str) -> np.ndarray:
        """A stack column. ``logw@k`` / ``prec@k``: log w or the precursor flag in NCE bin ``k``;
        ``<col>|prec`` / ``<col>|bond``: a column restricted to the precursor / cut fragments."""
        if name == "prec":
            return self.is_prec
        if "@" in name:
            col, k = name.split("@")
            return (self.is_prec if col == "prec" else base[col]) * (self.nce_bin == int(k))
        if "|" in name:
            col, part = name.split("|")
            return base[col] * (self.is_prec if part == "prec" else 1.0 - self.is_prec)
        return base[name]

    def bond_mean(self, score: np.ndarray) -> np.ndarray:
        """Mean bond score over each fragment's cut bonds (precursor: its own row)."""
        return np.add.reduceat(score[self.members], self.member_starts) / self.member_n

    def forward(self, X: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = X @ theta
        q = np.exp(seg_log_softmax(s, self.frag_starts, self.frag_spec))
        P = np.bincount(self.chan_group, weights=q[self.chan_frag] * self.chan_r, minlength=len(self.group_t))
        return q, P

    def ce_and_grad(self, X: np.ndarray, theta: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean over spectra of the covered-target group cross-entropy, and its gradient."""
        q, P = self.forward(X, theta)
        ce = -(self.group_that * np.log(P + EPS)).sum() / self.n_spec
        u = self.group_that[self.chan_group] * self.chan_r / np.maximum(P[self.chan_group], EPS)
        U = np.bincount(self.chan_frag, weights=u, minlength=len(q))
        return float(ce), (X * (q * (1.0 - U))[:, None]).sum(0) / self.n_spec

    def fit(self, X: np.ndarray, theta0: np.ndarray) -> np.ndarray:
        res = minimize(lambda th_: self.ce_and_grad(X, th_), theta0, jac=True, method="L-BFGS-B")
        return res.x

    def metrics(self, X: np.ndarray, theta: np.ndarray) -> pd.DataFrame:
        """Per-spectrum group cosine (uncovered target in the norm), covered TV and CE."""
        _, P = self.forward(X, theta)
        n = self.n_spec
        dot = np.bincount(self.group_spec, weights=P * self.group_t, minlength=n)
        pn = np.sqrt(np.bincount(self.group_spec, weights=P**2, minlength=n))
        tv = 0.5 * np.bincount(self.group_spec, weights=np.abs(P - self.group_that), minlength=n)
        ce = -np.bincount(self.group_spec, weights=self.group_that * np.log(P + EPS), minlength=n)
        return pd.DataFrame({"cos": dot / (pn * self.t_norm), "tv": tv, "ce": ce})


def clustered_diff(a: np.ndarray, b: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """Mean of ``a - b`` and its molecule-clustered standard error."""
    d = a - b
    resid = pd.Series(d - d.mean()).groupby(groups).sum().to_numpy()
    return float(d.mean()), float(np.sqrt((resid**2).sum()) / len(d))


# ---------------------------------------------------------------------------------------------


_NCE_T = [f"logw@{k}" for k in range(4)] + [f"prec@{k}" for k in range(4)]
PEAK_STACKS = {
    "model_T": ["logw"],
    "stack_logp": ["logw", "B_model"],
    "stack_gbdt_type": ["logw", "B_model", "B_gbdt_type"],
    "stack_gbdt": ["logw", "B_model", "B_gbdt"],
    "stack_mlp": ["logw", "B_model", "B_mlp"],
    "stack_all": ["logw", "B_model", "B_gbdt", "B_mlp"],
    # Controls added after the first read: NCE-dependent temperature plus a per-NCE precursor
    # offset (no learner), and the learner restricted to the precursor or to cut fragments.
    "ctrl_nce": _NCE_T,
    "stack_gbdt_prec_only": ["logw", "B_model", "B_gbdt|prec"],
    "stack_gbdt_bond_only": ["logw", "B_model", "B_gbdt|bond"],
    "ctrl_nce_gbdt_bond": _NCE_T + ["B_model", "B_gbdt|bond"],
    "ctrl_nce_gbdt": _NCE_T + ["B_model", "B_gbdt|bond", "B_gbdt|prec"],
}


def peak_summary(m: pd.DataFrame, names: list[str], groups: np.ndarray) -> dict:
    """Mean cos/TV/CE per predictor, and cosine gains vs ``model_T`` and ``stack_logp`` (clustered SE)."""
    out: dict = {"_n": int(len(m)), "_mols": int(pd.Series(groups).nunique())}
    for name in names:
        out[name] = {k: round(float(m[f"{name}__{k}"].mean()), 4) for k in ("cos", "tv", "ce")}
    for base in ("model_T", "stack_logp", "ctrl_nce"):
        for name in names:
            if name in (base, "model_raw") or (base == "ctrl_nce" and not name.startswith("ctrl_nce")):
                continue
            d, se = clustered_diff(m[f"{name}__cos"].to_numpy(), m[f"{base}__cos"].to_numpy(), groups)
            out[f"dcos_{name}_vs_{base}"] = {"mean": round(d, 4), "se": round(se, 4)}
    return out


def main() -> None:
    args = get_args()
    out_dp = Path(args.out_dp)
    tr_recs = pd.read_pickle(args.train_fp)
    tr_recs = tr_recs[tr_recs.split == "train"].reset_index(drop=True)
    LOG.info("train %d spectra / %d molecules", len(tr_recs), tr_recs.smiles.nunique())
    tab_tr, _, lab_tr = build_table(tr_recs)
    feat_cols = [c for c in tab_tr.columns if c not in ("spec", "p", "t_eq", "t_resp")]
    type_vocab = sorted(set(lab_tr))
    tab_tr["type_id"] = pd.Categorical(lab_tr, categories=type_vocab).codes.astype(float)
    # Hold out 10% of train molecules for early stopping.
    rng = np.random.default_rng(args.seed)
    mols = tr_recs.smiles.unique()
    held = set(rng.choice(mols, size=len(mols) // 10, replace=False))
    is_held = tr_recs.smiles.isin(held).to_numpy()[tab_tr.spec.to_numpy()]
    fit_tab, hold_tab = tab_tr[~is_held], tab_tr[is_held]
    type_cols = ["type_id", "is_precursor", "nce", "n_cand"]

    gbdt = fit_gbdt(fit_tab[feat_cols], fit_tab.t_eq.to_numpy(), hold_tab[feat_cols], hold_tab.t_eq.to_numpy(), args.seed)
    gbdt_type = fit_gbdt(fit_tab[type_cols], fit_tab.t_eq.to_numpy(), hold_tab[type_cols], hold_tab.t_eq.to_numpy(), args.seed)
    mlp = MLPLearner(feat_cols, args.seed).fit(fit_tab, hold_tab)
    del tab_tr, fit_tab, hold_tab

    # Score val and test at bond level, and build the peak-level arrays.
    ev_recs = pd.read_pickle(args.eval_fp)
    splits: dict[str, dict] = {}
    for split in args.eval_splits:
        recs = ev_recs[ev_recs.split == split].reset_index(drop=True)
        tab, spec, lab = build_table(recs)
        tab["type_id"] = pd.Categorical(lab, categories=type_vocab).codes.astype(float)
        tab.loc[tab.type_id < 0, "type_id"] = np.nan
        starts = _starts(spec)
        raw = {
            "model": np.log(tab.p.to_numpy() + EPS),
            "gbdt": np.log(gbdt.predict(tab[feat_cols]) + EPS),
            "gbdt_type": np.log(gbdt_type.predict(tab[type_cols]) + EPS),
            "mlp": mlp.score(tab),
        }
        peak = PeakSplit(recs, starts)
        X = {"logw": peak.log_wf, **{f"B_{k}": peak.bond_mean(v) for k, v in raw.items()}}
        # Stack weights (1-4 numbers) are fitted on 40% of this split's molecules, to bound cost.
        fit_mols = set(rng.choice(recs.smiles.unique(), size=int(0.4 * recs.smiles.nunique()), replace=False))
        fit_idx = np.flatnonzero(recs.smiles.isin(fit_mols).to_numpy())
        peak_fit = PeakSplit(recs.iloc[fit_idx].reset_index(drop=True), starts[fit_idx])
        X_fit = {"logw": peak_fit.log_wf, **{f"B_{k}": peak_fit.bond_mean(v) for k, v in raw.items()}}
        recs = recs.drop(columns=[c for c in recs.columns if c.startswith(("chan_", "frag_", "group_"))])
        recs["nn_sim"] = novelty(recs.smiles, args.train_mols_fp)
        splits[split] = {"recs": recs, "tab": tab[["t_eq", "t_resp"]], "spec": spec, "labels": np.asarray(lab),
                         "starts": starts, "raw": raw, "peak": peak, "X": X, "peak_fit": peak_fit, "X_fit": X_fit}
        LOG.info("%s: %d spectra, %d bond rows, %d fragments, %d channels", split, len(recs), len(tab),
                 len(peak.frag_spec), len(peak.chan_frag))
        del tab
    del ev_recs

    summary: dict[str, dict] = {}
    learners = ["gbdt", "mlp", "gbdt_type"]
    for split, d in splits.items():
        other = splits[[s for s in splits if s != split][0]] if len(splits) > 1 else d
        recs = d["recs"]
        strata = {
            "nce_bin": pd.cut(recs.nce.to_numpy(), NCE_EDGES, labels=NCE_LABELS),
            "nov_bin": pd.cut(recs.nn_sim.to_numpy(), NOV_EDGES),
            "novel": np.where(recs.nn_sim.to_numpy() < 0.5, "novel<0.5", "familiar>=0.5"),
        }
        res: dict = {}

        # Primary: peak level. Weights fitted on the other split (cross-fit).
        pm = pd.DataFrame(index=recs.index)
        theta: dict[str, list[float]] = {}
        for name, cols in {"model_raw": ["logw"], **PEAK_STACKS}.items():
            pf = other["peak_fit"]
            Xo = np.stack([pf.column(other["X_fit"], c) for c in cols], 1)
            theta0 = np.array([1.0 if c.startswith("logw") else 0.0 for c in cols])
            th_ = theta0 if name == "model_raw" else pf.fit(Xo, theta0)
            del Xo
            theta[name] = [round(float(x), 4) for x in th_]
            mm = d["peak"].metrics(np.stack([d["peak"].column(d["X"], c) for c in cols], 1), th_)
            for k in mm.columns:
                pm[f"{name}__{k}"] = mm[k].to_numpy()
            LOG.info("%s %s theta=%s cos=%.4f", split, name, theta[name], mm.cos.mean())
        names = list(theta)
        mols_arr = recs.smiles.to_numpy()
        for k, v in strata.items():
            pm[k] = v
        pm["smiles"] = mols_arr
        pm.to_pickle(out_dp / f"peak_metrics_{split}.pkl")
        res["peak"] = {
            "theta": theta, "columns": PEAK_STACKS, "overall": peak_summary(pm, names, mols_arr),
            **{f"by_{k}": {str(g): peak_summary(pm[idx], names, mols_arr[idx])
                           for g in pd.unique(pm[k]) if (idx := (pm[k] == g).to_numpy()).sum() > 50}
               for k in strata},
        }

        # Secondary: bond level. Isomeric fragments share an m/z, so bond attribution of the
        # target is not identifiable; t_eq and t_resp are two conventions, not ground truth.
        predictors: dict[str, np.ndarray] = {}
        weights: dict[str, list[float]] = {}
        specs = {"model_T": ["model"], **{f"{k}_T": [k] for k in learners},
                 **{f"stack_{k}": ["model", k] for k in learners}, "stack_all": ["model", "gbdt", "mlp"]}
        for name, cols in specs.items():
            w = fit_stack([other["raw"][c] for c in cols], other["tab"].t_eq.to_numpy(), other["starts"], other["spec"])
            weights[name] = [round(float(x), 4) for x in w]
            s = sum(wi * d["raw"][c] for wi, c in zip(w, cols))
            predictors[name] = seg_log_softmax(s, d["starts"], d["spec"])
        predictors["model_raw"] = seg_log_softmax(d["raw"]["model"], d["starts"], d["spec"])
        bnames = list(predictors)
        res["bond"] = {"weights": weights}
        for target_col in ("t_eq", "t_resp"):
            m = spectrum_metrics(predictors, d["tab"][target_col].to_numpy(), d["labels"], d["starts"])
            for k, v in strata.items():
                m[k] = v
            m.to_pickle(out_dp / f"bond_metrics_{split}_{target_col}.pkl")
            res["bond"][target_col] = {
                "overall": aggregate(m, bnames), "by_nce": stratified(m, bnames, "nce_bin"),
                "novel_split": stratified(m, bnames, "novel"),
            }
        summary[split] = res
    summary["_meta"] = {
        "train_spectra": int(len(tr_recs)), "train_molecules": int(len(mols)), "held_molecules": len(held),
        "gbdt_iters": int(gbdt.n_iter_), "gbdt_type_iters": int(gbdt_type.n_iter_), "features": feat_cols,
    }
    with open(out_dp / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    for split in splits:
        o = summary[split]["peak"]["overall"]
        LOG.info("%s peak level:\n%s", split, json.dumps(o, indent=1))


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train_fp", required=True)
    p.add_argument("--eval_fp", required=True)
    p.add_argument("--eval_splits", nargs="+", default=["val", "test"])
    p.add_argument("--train_mols_fp", required=True, help="Pickle with the train split's mol_id column.")
    p.add_argument("--out_dp", required=True)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
