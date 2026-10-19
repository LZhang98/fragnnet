"""Model-free headroom gate for a cross-adduct loss.

Question: is the [M+H]+ vs [M-H]- difference MOLECULE-SPECIFIC (headroom for a
per-molecule cross-adduct loss) or a GLOBAL pattern in neutral-loss space that an
adduct embedding already captures (no headroom)?

Compares, on held-out molecules:
  base   cos(+H, -H) in neutral-loss space, no correction
  global cos(+H * exp(w_NL), -H) with w fitted on DISJOINT training molecules
  floor  cos(replicate +H, replicate +H) same molecule/NCE -> measurement noise
"""
from collections import defaultdict

import numpy as np
import pandas as pd

PROTON = 1.007276
BIN = 0.01
TOPN = 40

d = pd.read_pickle('data/proc/nist20/spec_df.pkl')
d = d[d.prec_type.isin(['[M+H]+', '[M-H]-']) & d.inst_type.eq('FT')
      & d.frag_mode.eq('HCD') & d.nce.notna()]

def nl_vector(peaks, prec_mz, adduct):
    """Peaks -> {neutral-loss bin: relative intensity}."""
    if not peaks:
        return {}
    a = np.array(peaks, float)
    a = a[np.argsort(-a[:, 1])][:TOPN]
    sign = -1.0 if adduct == '[M+H]+' else +1.0
    neutral_prec = prec_mz + sign * PROTON
    neutral_frag = a[:, 0] + sign * PROTON
    nl = neutral_prec - neutral_frag
    keep = nl > -0.5
    nl, inten = nl[keep], a[keep, 1]
    if inten.sum() <= 0:
        return {}
    inten = inten / inten.sum()
    out = defaultdict(float)
    for x, y in zip(np.round(nl / BIN).astype(np.int64), inten):
        out[int(x)] += float(y)
    return dict(out)

def cos(u, v):
    keys = set(u) | set(v)
    if not keys:
        return np.nan
    a = np.array([u.get(k, 0.0) for k in keys])
    b = np.array([v.get(k, 0.0) for k in keys])
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / n) if n > 0 else np.nan

# 1. Build matched +H / -H pairs at identical (molecule, NCE).
pairs = {}
reps = defaultdict(list)
for r in d.itertuples():
    k = (r.mol_id, r.nce)
    reps[(r.mol_id, r.nce, r.prec_type)].append((r.peaks, r.prec_mz))
    pairs.setdefault(k, {}).setdefault(r.prec_type, (r.peaks, r.prec_mz))
pairs = {k: v for k, v in pairs.items() if len(v) == 2}
print(f"matched (molecule, NCE) pairs with both polarities: {len(pairs):,}")

mols = sorted({k[0] for k in pairs})
rng = np.random.default_rng(0)
rng.shuffle(mols)
cut = len(mols) // 2
fit_mols, test_mols = set(mols[:cut]), set(mols[cut:])
print(f"molecules: {len(fit_mols):,} fit / {len(test_mols):,} held out (disjoint)")

# 2. Fit a GLOBAL neutral-loss polarity reweighting on the fit molecules only.
num = defaultdict(float)
den = defaultdict(float)
EPS = 1e-4
for (mol, nce), v in pairs.items():
    if mol not in fit_mols:
        continue
    p = nl_vector(*v['[M+H]+'], '[M+H]+')
    m = nl_vector(*v['[M-H]-'], '[M-H]-')
    for b in set(p) | set(m):
        num[b] += np.log((m.get(b, 0.0) + EPS) / (p.get(b, 0.0) + EPS))
        den[b] += 1.0
w = {b: num[b] / den[b] for b in num if den[b] >= 5}
print(f"global reweighting fitted on {len(w):,} recurring neutral-loss bins")

# 3. Score held-out molecules.
base, glob = [], []
for (mol, nce), v in pairs.items():
    if mol not in test_mols:
        continue
    p = nl_vector(*v['[M+H]+'], '[M+H]+')
    m = nl_vector(*v['[M-H]-'], '[M-H]-')
    if not p or not m:
        continue
    base.append(cos(p, m))
    q = {b: i * np.exp(np.clip(w.get(b, 0.0), -4, 4)) for b, i in p.items()}
    s = sum(q.values())
    glob.append(cos({b: i / s for b, i in q.items()}, m) if s > 0 else np.nan)

# 4. Replicate floor: two spectra, same molecule/NCE/adduct.
floor = []
for (mol, nce, ad), lst in reps.items():
    if mol in test_mols and len(lst) >= 2:
        a = nl_vector(*lst[0], ad)
        b = nl_vector(*lst[1], ad)
        if a and b:
            floor.append(cos(a, b))

base = np.array(base, float)
glob = np.array(glob, float)
floor = np.array(floor, float)
print(f"\nheld-out pairs scored: {np.isfinite(base).sum():,}   replicate pairs: {len(floor):,}")
print(f"  replicate floor (same molecule/adduct/NCE) : {np.nanmean(floor):.3f}")
print(f"  base    cos(+H, -H), no correction          : {np.nanmean(base):.3f}")
print(f"  global  cos(+H * exp(w_NL), -H)             : {np.nanmean(glob):.3f}")
gap = np.nanmean(floor) - np.nanmean(base)
closed = (np.nanmean(glob) - np.nanmean(base)) / gap if gap > 0 else np.nan
print(f"\n  gap to replicate floor                     : {gap:.3f}")
print(f"  fraction closed by the GLOBAL pattern       : {closed:.1%}")
print(f"  -> remaining, molecule-specific             : {1-closed:.1%}")
