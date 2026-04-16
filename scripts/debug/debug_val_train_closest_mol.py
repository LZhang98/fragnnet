"""Debug script: for each val-set (spectrum, mol) pair, find its closest
training-set molecule that shares the same molecular formula (by ECFP
Tanimoto distance), then compare that val spectrum to the train spectrum
with the same adduct + instrument type and closest CE.

One output row per val spectrum.  The closest-train-mol lookup is cached
per val mol_id so ECFP search is done only once per unique val molecule.

Usage:
    python scripts/debug/debug_val_train_closest_mol.py \
        --spec_fp data/proc/nist20/spec_df.pkl \
        --mol_fp  data/proc/nist20/mol_df.pkl \
        --split_dp data/split/nist20_inchikey \
        --max_val_specs 2000 \
        --output_fp /tmp/debug_closest_mol.pkl

Optional flags:
    --adducts "[M+H]+"         # comma-separated; default = all
    --inst_types FT            # comma-separated; default = all
    --morgan_radius 3
    --morgan_nbits 2048
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fingerprint helpers
# ---------------------------------------------------------------------------

def smiles_to_ecfp(smiles: str, radius: int = 3, nbits: int = 2048) -> np.ndarray | None:
    """Convert SMILES to ECFP bit-vector as a numpy array.

    Returns None if SMILES is invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    arr = np.zeros(nbits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_matrix(query: np.ndarray, db: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity between a single query (1-D) and a DB matrix (N × bits).

    Returns 1-D array of shape (N,) with similarities in [0, 1].
    """
    # intersection = dot product of bit vectors
    inter = db @ query  # (N,)
    # union = |query| + |db_i| - inter
    q_sum = query.sum()
    db_sums = db.sum(axis=1)  # (N,)
    union = q_sum + db_sums - inter
    # avoid division by zero (both empty fingerprints)
    sim = np.where(union > 0, inter / union, 0.0)
    return sim.astype(np.float32)


# ---------------------------------------------------------------------------
# Spectrum cosine similarity (simple, numpy-based)
# ---------------------------------------------------------------------------

def peaks_to_array(peaks, mz_max: float = 1500.0, bin_res: float = 0.1) -> np.ndarray:
    """Convert a list of (mz, intensity) pairs to a binned intensity array."""
    n_bins = int(round(mz_max / bin_res))
    arr = np.zeros(n_bins, dtype=np.float32)
    if peaks is None:
        return arr
    for mz, intensity in peaks:
        idx = int(round(mz / bin_res))
        if 0 <= idx < n_bins:
            arr[idx] += intensity
    norm = np.linalg.norm(arr)
    if norm > 0:
        arr /= norm
    return arr


def cosine_sim(peaks_a, peaks_b, mz_max: float = 1500.0, bin_res: float = 0.1) -> float:
    """Compute cosine similarity between two spectra (list of (mz, intensity) pairs)."""
    a = peaks_to_array(peaks_a, mz_max, bin_res)
    b = peaks_to_array(peaks_b, mz_max, bin_res)
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# CE matching: pick the training spectrum whose CE is closest to the val CE
# ---------------------------------------------------------------------------

def get_ce(row: pd.Series) -> float | None:
    """Extract a single float CE value from a spectrum row.

    Prefers nce if available, else ace.  Returns None if both are missing.
    """
    for col in ("nce", "ace"):
        if col in row.index:
            v = row[col]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return float(v)
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    p.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    p.add_argument("--split_dp", default="data/split/nist20_inchikey")
    p.add_argument("--max_val_specs", type=int, default=2000,
                   help="Cap on number of val spectra to process (0 = no cap)")
    p.add_argument("--adducts", default="",
                   help="Comma-separated adduct filter, e.g. '[M+H]+'.  Empty = all.")
    p.add_argument("--inst_types", default="",
                   help="Comma-separated instrument filter, e.g. 'FT'.  Empty = all.")
    p.add_argument("--morgan_radius", type=int, default=3)
    p.add_argument("--morgan_nbits", type=int, default=2048)
    p.add_argument("--output_fp", default="/tmp/debug_closest_mol.pkl")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    adduct_filter = [a.strip() for a in args.adducts.split(",") if a.strip()]
    inst_filter = [i.strip() for i in args.inst_types.split(",") if i.strip()]

    # ------------------------------------------------------------------
    # 1. Load dataframes
    # ------------------------------------------------------------------
    logger.info("Loading spec_df from %s", args.spec_fp)
    spec_df: pd.DataFrame = pd.read_pickle(args.spec_fp)
    logger.info("Loading mol_df from %s", args.mol_fp)
    mol_df: pd.DataFrame = pd.read_pickle(args.mol_fp)

    logger.info("spec_df shape: %s   mol_df shape: %s", spec_df.shape, mol_df.shape)

    # ------------------------------------------------------------------
    # 2. Load splits
    # ------------------------------------------------------------------
    def load_split(name: str) -> pd.DataFrame:
        fp = os.path.join(args.split_dp, f"{name}_ids.csv")
        assert os.path.isfile(fp), f"Split file not found: {fp}"
        return pd.read_csv(fp)

    train_split = load_split("train")
    val_split = load_split("val")

    train_spec_ids = set(train_split["spec_id"].unique())
    train_mol_ids = set(train_split["mol_id"].unique())
    val_spec_ids = set(val_split["spec_id"].unique())
    val_mol_ids = set(val_split["mol_id"].unique())
    logger.info(
        "Train: %d spectra, %d mols   Val: %d spectra, %d mols",
        len(train_spec_ids), len(train_mol_ids),
        len(val_spec_ids), len(val_mol_ids),
    )

    # ------------------------------------------------------------------
    # 3. Build val spectra table (apply adduct / inst filters here)
    # ------------------------------------------------------------------
    val_spec_df = spec_df[spec_df["spec_id"].isin(val_spec_ids)].copy()
    if adduct_filter:
        val_spec_df = val_spec_df[val_spec_df["prec_type"].isin(adduct_filter)]
        logger.info("Val spectra after adduct filter: %d", len(val_spec_df))
    if inst_filter:
        val_spec_df = val_spec_df[val_spec_df["inst_type"].isin(inst_filter)]
        logger.info("Val spectra after inst filter: %d", len(val_spec_df))

    if args.max_val_specs and args.max_val_specs > 0:
        val_spec_df = val_spec_df.sample(
            n=min(args.max_val_specs, len(val_spec_df)), random_state=42
        )
    logger.info("Processing %d val spectra …", len(val_spec_df))

    # ------------------------------------------------------------------
    # 4. Build training molecule set (mol_id → formula, fingerprint)
    # ------------------------------------------------------------------
    mol_df = mol_df.set_index("mol_id")

    logger.info("Computing ECFP for training molecules …")
    train_mol_df = mol_df[mol_df.index.isin(train_mol_ids)].copy()

    fp_list: list[np.ndarray | None] = []
    for smiles in train_mol_df["smiles"]:
        fp_list.append(smiles_to_ecfp(smiles, args.morgan_radius, args.morgan_nbits))
    train_mol_df["ecfp"] = fp_list
    train_mol_df = train_mol_df[train_mol_df["ecfp"].notna()].copy()
    logger.info("Train mols with valid ECFP: %d", len(train_mol_df))

    # Group training mols by formula for fast same-formula lookup
    train_by_formula: dict[str, pd.DataFrame] = {  # noqa: C417
        f: g for f, g in train_mol_df.groupby("formula")
    }

    # Index training spectra by mol_id (no adduct/inst filter — we match on
    # the val spectrum's own adduct + inst_type below)
    train_spec_df = spec_df[spec_df["spec_id"].isin(train_spec_ids)]
    train_specs_by_mol: dict[str, pd.DataFrame] = {  # noqa: C417
        mid: g for mid, g in train_spec_df.groupby("mol_id")
    }

    # ------------------------------------------------------------------
    # 5. Main loop: one row per val spectrum
    # ------------------------------------------------------------------
    # Cache closest-train-mol lookup per val mol_id (ECFP search is expensive)
    mol_match_cache: dict[str, tuple[str, str, float] | None] = {}
    # cache value: (train_mol_id, train_smiles, tanimoto) or None

    rows = []
    n_no_formula_match = 0
    n_no_spec_match = 0

    for _, v_spec in val_spec_df.iterrows():
        val_mol_id: str = v_spec["mol_id"]
        val_spec_id: str = v_spec["spec_id"]
        adduct: str = v_spec["prec_type"]
        inst: str = v_spec["inst_type"]
        v_ce = get_ce(v_spec)

        # --- 5a. Mol metadata for this val spectrum ---------------------
        if val_mol_id not in mol_df.index:
            continue
        val_mol_row = mol_df.loc[val_mol_id]
        formula: str = val_mol_row["formula"]
        val_smiles: str = val_mol_row["smiles"]

        # --- 5b. Find closest train mol (cached per val mol_id) --------
        if val_mol_id not in mol_match_cache:
            val_fp = smiles_to_ecfp(val_smiles, args.morgan_radius, args.morgan_nbits)
            if val_fp is None or formula not in train_by_formula:
                mol_match_cache[val_mol_id] = None
            else:
                cand = train_by_formula[formula]
                cand_fps = np.stack(cand["ecfp"].tolist(), axis=0)
                sims = tanimoto_matrix(val_fp, cand_fps)
                best_idx = int(np.argmax(sims))
                mol_match_cache[val_mol_id] = (
                    cand.index[best_idx],
                    cand.iloc[best_idx]["smiles"],
                    float(sims[best_idx]),
                )

        match = mol_match_cache[val_mol_id]
        if match is None:
            n_no_formula_match += 1
            continue

        best_train_mol_id, best_train_smiles, tanimoto = match

        # --- 5c. Find train spectrum: same adduct + inst, closest CE ---
        train_specs = train_specs_by_mol.get(best_train_mol_id)
        if train_specs is None or len(train_specs) == 0:
            n_no_spec_match += 1
            rows.append({
                "val_spec_id": val_spec_id,
                "val_mol_id": val_mol_id,
                "val_smiles": val_smiles,
                "formula": formula,
                "adduct": adduct,
                "inst_type": inst,
                "val_ce": v_ce,
                "train_mol_id": best_train_mol_id,
                "train_smiles": best_train_smiles,
                "tanimoto": tanimoto,
                "train_spec_id": None,
                "train_ce": None,
                "ce_diff": None,
                "cosine_sim": None,
            })
            continue

        t_subs = train_specs[
            (train_specs["prec_type"] == adduct) & (train_specs["inst_type"] == inst)
        ]
        if len(t_subs) == 0:
            n_no_spec_match += 1
            rows.append({
                "val_spec_id": val_spec_id,
                "val_mol_id": val_mol_id,
                "val_smiles": val_smiles,
                "formula": formula,
                "adduct": adduct,
                "inst_type": inst,
                "val_ce": v_ce,
                "train_mol_id": best_train_mol_id,
                "train_smiles": best_train_smiles,
                "tanimoto": tanimoto,
                "train_spec_id": None,
                "train_ce": None,
                "ce_diff": None,
                "cosine_sim": None,
            })
            continue

        # Pick train spectrum with closest CE to this val spectrum
        t_ces = [get_ce(r) for _, r in t_subs.iterrows()]
        if v_ce is not None and any(tc is not None for tc in t_ces):
            diffs = [abs(v_ce - tc) if tc is not None else float("inf") for tc in t_ces]
            best_t_idx = int(np.argmin(diffs))
            ce_diff: float | None = diffs[best_t_idx] if diffs[best_t_idx] != float("inf") else None
        else:
            best_t_idx = 0
            ce_diff = None

        t_row = t_subs.iloc[best_t_idx]
        t_ce = t_ces[best_t_idx]
        cos = cosine_sim(v_spec["peaks"], t_row["peaks"])

        rows.append({
            "val_spec_id": val_spec_id,
            "val_mol_id": val_mol_id,
            "val_smiles": val_smiles,
            "formula": formula,
            "adduct": adduct,
            "inst_type": inst,
            "val_ce": v_ce,
            "train_mol_id": best_train_mol_id,
            "train_smiles": best_train_smiles,
            "tanimoto": tanimoto,
            "train_spec_id": t_row["spec_id"],
            "train_ce": t_ce,
            "ce_diff": ce_diff,
            "cosine_sim": cos,
        })

    # ------------------------------------------------------------------
    # 6. Assemble and save results
    # ------------------------------------------------------------------
    result_df = pd.DataFrame(rows)

    out_fp = Path(args.output_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_pickle(str(out_fp))
    logger.info("Saved results to %s", out_fp)

    # ------------------------------------------------------------------
    # 7. Print stats
    # ------------------------------------------------------------------
    sep = "=" * 70
    n_total = len(val_spec_df)
    n_rows = len(result_df)
    matched = result_df[result_df["cosine_sim"].notna()] if n_rows > 0 else pd.DataFrame()
    n_matched = len(matched)

    def _pct(n: int, total: int) -> str:
        return f"{n} / {total}  ({100 * n / total:.1f}%)" if total else str(n)

    print(f"\n{sep}")
    print("COVERAGE")
    print(sep)
    print(f"  Val spectra processed       : {n_total}")
    print(f"  No same-formula train mol   : {_pct(n_no_formula_match, n_total)}")
    print(f"  No shared adduct/inst specs : {_pct(n_no_spec_match, n_total)}")
    print(f"  Matched pairs               : {_pct(n_matched, n_total)}")

    if n_matched > 0:
        def _fmt_series(s: pd.Series) -> str:
            return (
                f"mean={s.mean():.4f}  median={s.median():.4f}"
                f"  p10={s.quantile(0.1):.4f}  p25={s.quantile(0.25):.4f}"
                f"  p75={s.quantile(0.75):.4f}  p90={s.quantile(0.9):.4f}"
                f"  min={s.min():.4f}  max={s.max():.4f}"
            )

        print(f"\n{sep}")
        print("TANIMOTO SIMILARITY  (val mol vs closest same-formula train mol)")
        print(sep)
        print(f"  {_fmt_series(result_df['tanimoto'])}")
        # Histogram buckets
        bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, float("inf")]
        labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
                  "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", "=1.0"]
        tan_valid = result_df["tanimoto"].dropna()
        counts, _ = np.histogram(tan_valid, bins=bins)
        for label, cnt in zip(labels, counts):
            bar = "#" * int(50 * cnt / len(tan_valid)) if len(tan_valid) else ""
            print(f"  {label:>10s}  {cnt:5d}  {bar}")

        print(f"\n{sep}")
        print("COSINE SIMILARITY  (val vs closest same-formula train spectrum)")
        print(sep)
        print(f"  {_fmt_series(matched['cosine_sim'])}")
        counts, _ = np.histogram(matched["cosine_sim"], bins=bins)
        for label, cnt in zip(labels, counts):
            bar = "#" * int(50 * cnt / n_matched)
            print(f"  {label:>10s}  {cnt:5d}  {bar}")

        print(f"\n{sep}")
        print("CE DIFFERENCE  (val CE minus train CE, absolute)")
        print(sep)
        ce_diff = matched["ce_diff"].dropna()
        if len(ce_diff) > 0:
            print(f"  n with CE info : {len(ce_diff)} / {n_matched}")
            print(f"  {_fmt_series(ce_diff)}")
        else:
            print("  (no CE info available in matched pairs)")

        if "adduct" in matched.columns:
            print(f"\n{sep}")
            print("COSINE SIM  by adduct")
            print(sep)
            for adduct, grp in matched.groupby("adduct"):
                print(f"  {adduct:>12s}  n={len(grp):5d}  {_fmt_series(grp['cosine_sim'])}")

        if "inst_type" in matched.columns:
            print(f"\n{sep}")
            print("COSINE SIM  by instrument type")
            print(sep)
            for inst, grp in matched.groupby("inst_type"):
                print(f"  {inst:>8s}  n={len(grp):5d}  {_fmt_series(grp['cosine_sim'])}")

        # Tanimoto-binned cosine: how does spectral similarity vary with structural similarity?
        print(f"\n{sep}")
        print("COSINE SIM  by Tanimoto bin  (structural → spectral relationship)")
        print(sep)
        tan_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 1.0, float("inf")]
        tan_labels = ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.9", "0.9-1.0", "=1.0"]
        matched_copy = matched.copy()
        matched_copy["tan_bin"] = pd.cut(
            matched_copy["tanimoto"], bins=tan_bins, labels=tan_labels, right=False
        )
        for label, grp in matched_copy.groupby("tan_bin", observed=True):
            if len(grp) == 0:
                continue
            print(f"  {str(label):>10s}  n={len(grp):5d}  cos: {_fmt_series(grp['cosine_sim'])}")

    print(f"\n{sep}")
    print(f"Results saved to: {out_fp}")
    print(sep)


if __name__ == "__main__":
    main()
