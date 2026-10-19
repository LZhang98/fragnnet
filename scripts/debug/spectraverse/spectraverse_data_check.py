"""Check local Spectraverse data against the v1.0.2 release and its own dedup rule.

1. ``md5`` of the raw MGF against the Zenodo v1.0.2 record.
2. ``spec_df.pkl`` against the MGF headers, record by record (``dset_spec_id`` is the 1-based
   MGF record index): precursor m/z, adduct and peak count.
3. ``<split-dp>/cv{k}`` against the MGF ``FOLD`` field, and 2D InChIKeys spanning folds.
4. Byte-identical twins: spectra of the same molecule with identical peak lists, and peak lists
   shared by different molecules.
5. Near-duplicates under the curation paper's rule (same InChIKey first block and polarity,
   cosine > 0.99), on sampled groups: greedy cosine at 0.1 Da (matchms ``CosineGreedy``
   defaults), binned cosine at 0.01 Da, and byte-identical.
6. Optional (``--spectral-entropy-dir``): replay of the upstream dedup score,
   ``spectral_entropy.similarity(method="dot_product", need_clean_spectra=False, ms2_ppm=5)``
   from ``spectraverse-analysis/spectraverse/step3-4_uniq-cos-calc.py``, on sampled twin pairs.
   SpectralEntropy is not on PyPI; pass a checkout of github.com/YuanyueLi/SpectralEntropy.

See docs/notes/spectraverse_curation_paper.md and
docs/findings/masster/spectraverse_masster_findings.md.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ZENODO_V102_MD5 = "50803b370a118299bfef3e9485ef6895"  # zenodo.org/api/records/19927403
HEADER_KEYS = {
    "SOURCE": "source",
    "ADDUCT": "adduct",
    "PRECURSOR_MZ": "prec_mz",
    "NUM_PEAKS": "num_peaks",
    "INCHIKEY": "inchikey",
    "IONMODE": "ion_mode",
    "FOLD": "fold",
}


def read_mgf_headers(mgf_fp: str) -> pd.DataFrame:
    """Return the ``HEADER_KEYS`` fields of every MGF record, in file order."""
    rows, cur = [], {}
    with open(mgf_fp) as f:
        for line in f:
            key, _, value = line.rstrip("\n").partition("=")
            if key in HEADER_KEYS:
                cur[HEADER_KEYS[key]] = value
            elif line.startswith("END IONS"):
                rows.append(cur)
                cur = {}
    headers = pd.DataFrame(rows)
    headers["prec_mz"] = headers["prec_mz"].astype(float)
    headers["num_peaks"] = headers["num_peaks"].astype(int)
    headers["fold"] = headers["fold"].astype(int)
    return headers


def greedy_cosine(a: list, b: list, tol: float = 0.1) -> float:
    """Greedy peak-matching cosine (matchms ``CosineGreedy``, intensity and m/z power 1, 0)."""
    am, ai = np.array([p[0] for p in a]), np.array([p[1] for p in a])
    bm, bi = np.array([p[0] for p in b]), np.array([p[1] for p in b])
    i, j = np.nonzero(np.abs(am[:, None] - bm[None, :]) <= tol)
    scores = ai[i] * bi[j]
    used_a, used_b, total = set(), set(), 0.0
    for k in np.argsort(-scores, kind="stable"):
        if i[k] in used_a or j[k] in used_b:
            continue
        used_a.add(i[k])
        used_b.add(j[k])
        total += scores[k]
    return total / (np.linalg.norm(ai) * np.linalg.norm(bi))


def binned_cosine(a: list, b: list, width: float = 0.01) -> float:
    """Cosine of intensity vectors binned at ``width`` Da (summed within a bin)."""
    va, vb = {}, {}
    for peaks, vec in ((a, va), (b, vb)):
        for mz, intensity in peaks:
            key = int(round(mz / width))
            vec[key] = vec.get(key, 0.0) + intensity
    num = sum(va[k] * vb[k] for k in va.keys() & vb.keys())
    return num / (np.linalg.norm(list(va.values())) * np.linalg.norm(list(vb.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mgf-fp", default="data/raw/spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf"
    )
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument("--split-dp", default="data/split/spectraverse_benchmark")
    parser.add_argument("--n-groups", type=int, default=1500, help="groups sampled in step 5")
    parser.add_argument("--max-group-size", type=int, default=60)
    parser.add_argument("--n-twin-pairs", type=int, default=400, help="pairs replayed in step 6")
    parser.add_argument("--spectral-entropy-dir", default=None)
    parser.add_argument("--pairs-out", default=None, help="optional CSV of step-5 pair scores")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # 1. Checksum.
    md5 = hashlib.md5()
    with open(args.mgf_fp, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    print(f"[1] md5 {md5.hexdigest()}; Zenodo v1.0.2: {md5.hexdigest() == ZENODO_V102_MD5}")

    # 2. spec_df against the MGF, record by record.
    headers = read_mgf_headers(args.mgf_fp)
    spec = pd.read_pickle(args.spec_fp)
    spec = spec.assign(rec=spec["dset_spec_id"].astype(int) - 1).sort_values("rec")
    spec = spec.reset_index(drop=True)
    contiguous = (
        len(spec) == len(headers) and (spec["rec"].to_numpy() == np.arange(len(spec))).all()
    )
    print(f"[2] records: MGF {len(headers):,}, spec_df {len(spec):,}, contiguous ids {contiguous}")
    if not contiguous:
        raise ValueError("spec_df dset_spec_id does not index the MGF records 1..N")
    agree = {
        "prec_mz": np.isclose(spec["prec_mz"].astype(float), headers["prec_mz"], atol=1e-4).mean(),
        "adduct": (spec["prec_type"].to_numpy() == headers["adduct"].to_numpy()).mean(),
        "peak count": (spec["peaks"].map(len).to_numpy() == headers["num_peaks"].to_numpy()).mean(),
    }
    print("    agree: " + ", ".join(f"{k} {v:.6f}" for k, v in agree.items()))
    spec["source"] = headers["source"].to_numpy()
    spec["ik14"] = headers["inchikey"].str[:14].to_numpy()
    spec["fold"] = headers["fold"].to_numpy()

    # 3. Our cv splits against FOLD.
    fold_of = dict(zip(spec["spec_id"], spec["fold"]))
    for cv_dp in sorted(Path(args.split_dp).glob("cv*"), key=lambda p: int(p.name[2:])):
        got = {
            sp: sorted(pd.read_csv(cv_dp / f"{sp}_ids.csv")["spec_id"].map(fold_of).unique())
            for sp in ("train", "val", "test")
        }
        print(
            f"[3] {cv_dp.name}: test FOLD {got['test']}, val FOLD {got['val']}, "
            f"train FOLD {got['train']}"
        )
    spanning = (spec.groupby("ik14")["fold"].nunique() > 1).sum()
    print(f"    2D InChIKeys spanning >1 FOLD: {spanning} of {spec['ik14'].nunique():,}")

    # 4. Byte-identical twins.
    spec["key"] = spec["peaks"].map(lambda p: hash(tuple(p)))
    twins = spec[spec.duplicated(["mol_id", "key"], keep=False)]
    twin_sets = list(twins.groupby(["mol_id", "key"]).groups.values())
    desc = twins.groupby(["mol_id", "key"]).agg(
        adducts=("prec_type", "nunique"), ace=("ace", lambda x: x.nunique(dropna=False))
    )
    shared = spec[spec.duplicated("key", keep=False)].groupby("key")["mol_id"].nunique()
    print(
        f"[4] spectra with a byte-identical twin (same molecule): {len(twins):,} "
        f"({len(twins) / len(spec):.2%}) in {len(twin_sets):,} sets; one adduct "
        f"{(desc.adducts == 1).mean():.3f}; disagree on ACE {(desc.ace > 1).mean():.3f}"
    )
    print(f"    peak lists shared by >1 molecule: {(shared > 1).sum():,}")

    # 5. Near-duplicates under the paper's rule, on sampled (ik14, polarity) groups.
    rng = np.random.default_rng(args.seed)
    groups = spec.groupby(["ik14", "ion_mode"]).indices
    keys = [k for k, v in groups.items() if len(v) >= 2]
    rows = []
    for g in rng.choice(len(keys), min(args.n_groups, len(keys)), replace=False):
        idx = groups[keys[g]]
        if len(idx) > args.max_group_size:
            idx = rng.choice(idx, args.max_group_size, replace=False)
        sub = spec.iloc[idx]
        peaks, sources = sub["peaks"].tolist(), sub["source"].tolist()
        for a in range(len(peaks)):
            for b in range(a + 1, len(peaks)):
                if len(peaks[a]) * len(peaks[b]) > 400_000:
                    continue
                rows.append(
                    {
                        "identical": peaks[a] == peaks[b],
                        "greedy_01": greedy_cosine(peaks[a], peaks[b]),
                        "binned_001": binned_cosine(peaks[a], peaks[b]),
                        "same_source": sources[a] == sources[b],
                        "source_pair": "|".join(sorted([sources[a], sources[b]])),
                    }
                )
    pairs = pd.DataFrame(rows)
    print(
        f"[5] {len(pairs):,} pairs from {min(args.n_groups, len(keys))} groups: "
        f"identical {pairs.identical.mean():.4f}; > 0.99 greedy 0.1 Da "
        f"{(pairs.greedy_01 > 0.99).mean():.4f}, binned 0.01 Da {(pairs.binned_001 > 0.99).mean():.4f}"
    )
    high = pairs[pairs.greedy_01 > 0.99]
    print(
        f"    > 0.99 pairs from the same source: {high.same_source.mean():.3f}; top source pairs:"
    )
    print(high.source_pair.value_counts().head(6).to_string())
    if args.pairs_out:
        pairs.to_csv(args.pairs_out, index=False)

    # 6. Optional replay of the upstream dedup score on twin pairs.
    if args.spectral_entropy_dir is None:
        return
    sys.path.insert(0, args.spectral_entropy_dir)
    import spectral_entropy

    def upstream_input(peaks):
        arr = np.array(peaks, dtype=np.float32)
        arr[:, 1] /= arr[:, 1].max()  # matchms normalize_intensities
        return np.vstack([arr, [0, 0]]) if len(arr) == 1 else arr  # upstream single-peak padding

    scores = []
    for i in rng.choice(len(twin_sets), min(args.n_twin_pairs, len(twin_sets)), replace=False):
        a, b = list(twin_sets[i])[:2]
        scores.append(
            spectral_entropy.similarity(
                upstream_input(spec.loc[a, "peaks"]),
                upstream_input(spec.loc[b, "peaks"]),
                method="dot_product",
                need_clean_spectra=False,
                ms2_ppm=5,
            )
        )
    scores = np.array(scores)
    print(
        f"[6] upstream score on {len(scores)} twin pairs: > 0.99 {(scores > 0.99).mean():.3f}, "
        f"median {np.median(scores):.4f}, min {scores.min():.4f}"
    )


if __name__ == "__main__":
    main()
