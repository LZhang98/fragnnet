"""Find systematically missing peaks per ClassyFire subclass.

For each subclass, identifies true spectrum peaks that are consistently:
  (A) absent from the BFS fragmentation DAG (BFS cannot generate this formula)
  (B) in the DAG but under-predicted by the model (predicted intensity < 0.05)

When --frag_dp is omitted, category A falls back to checking the model's
predicted frag_formula_peak_mzs — which only contains peaks the model assigned
non-zero weight to, so "absent" means "model never predicts it" rather than
"BFS can't generate it". Always pass --frag_dp for accurate DAG coverage.

Usage:
    python scripts/eval/find_missing_peaks.py \
        --pred_dir data/predicted \
        --pred_prefix nps_inchikey_nist23_nist23 \
        --spec_fp data/proc/nist23/spec_df.pkl \
        --cf_fp data/classyfire/nist23_mol_classyfire.pkl \
        --frag_dp data/frag/nist23_ma_mi_d3_h4_isoFalse.tar \
        --n_cvs 9 \
        --inst_type FT \
        --prec_type "[M+H]+" \
        --min_intensity 0.05 \
        --min_spectra 20 \
        --top_n 10
"""

import argparse
import bz2
import os
import pickle
import tarfile
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

TOL = 0.02  # Da matching tolerance

# Approximate mass shift from neutral formula mass to observed m/z.
# These match fragnnet.utils.formula_utils.PREC_TYPE_TO_MASS_DIFF.
_H = 1.0078250319
_e = 0.0005485799
_PREC_MASS_DIFF: dict[str, float] = {
    "[M+H]+": _H - _e,
    "[M+Na]+": 22.9897692 - _e,
    "[M+K]+": 38.9637068 - _e,
    "[M+NH4]+": 14.0030740 + 4 * _H - _e,
    "[M-H]-": -(_H - _e),
    "[M+Cl]-": 34.9688853 + _e,
    "[M]+": -_e,
    "[M-]+": _e,
}


class DagLoader:
    """Lazy-loading cache for fragment DAG formula peak m/z values.

    Supports a flat directory of ``{mol_id}.pkl[.bz2]`` files or a ``.tar``
    archive whose members follow the same naming convention.
    """

    def __init__(self, frag_dp: Path) -> None:
        self._dp = frag_dp
        self._is_tar = str(frag_dp).endswith(".tar")
        self._neutral_cache: dict[int, np.ndarray | None] = {}
        if self._is_tar:
            print(f"Indexing DAG tar: {frag_dp} …", flush=True)
            self._tar = tarfile.open(frag_dp)
            self._tar_index: dict[int, tarfile.TarInfo] = {}
            for m in self._tar.getmembers():
                base = os.path.basename(m.name)
                try:
                    if base.endswith(".pkl.bz2"):
                        mid = int(base[:-8])
                        self._tar_index[mid] = m
                    elif base.endswith(".pkl"):
                        mid = int(base[:-4])
                        if mid not in self._tar_index:
                            self._tar_index[mid] = m
                except ValueError:
                    pass
            print(f"  Indexed {len(self._tar_index):,} DAG entries.", flush=True)
        else:
            self._tar = None

    def get(self, mol_id: int, prec_type: str) -> np.ndarray | None:
        """Return all DAG formula peak m/z values in observed m/z space."""
        if mol_id not in self._neutral_cache:
            self._neutral_cache[mol_id] = self._load_neutral(mol_id)
        neutral = self._neutral_cache[mol_id]
        if neutral is None:
            return None
        mass_diff = _PREC_MASS_DIFF.get(prec_type, _H - _e)
        return neutral + mass_diff

    def _load_neutral(self, mol_id: int) -> np.ndarray | None:
        """Load the flat array of non-zero neutral formula_peak_mzs values."""
        try:
            if self._tar is not None:
                member = self._tar_index.get(mol_id)
                if member is None:
                    return None
                raw = self._tar.extractfile(member).read()
                data = bz2.decompress(raw) if member.name.endswith(".bz2") else raw
                dag = pickle.loads(data)
            else:
                for suffix, opener in [(".pkl.bz2", bz2.open), (".pkl", open)]:
                    fp = self._dp / f"{mol_id}{suffix}"
                    if fp.exists():
                        with opener(fp, "rb") as f:
                            dag = pickle.load(f)
                        break
                else:
                    return None
            mzs = np.asarray(dag["formula_peak_mzs"], dtype=float).ravel()
            return mzs[mzs > 0]
        except Exception:
            return None

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()


def mz_to_bin(mz: float, bin_size: float = 0.1) -> float:
    return round(mz / bin_size) * bin_size


def iter_cv(pred_dir: Path, pred_prefix: str, cv_idx: int):
    """Load one CV fold from zip and yield the dataframe."""
    zip_path = pred_dir / f"{pred_prefix}_cv{cv_idx}.zip"
    if not zip_path.exists():
        return None
    with zipfile.ZipFile(zip_path) as z:
        pkl_name = [n for n in z.namelist() if n.endswith(".pkl")][0]
        with z.open(pkl_name) as f:
            return pickle.load(f)


def accumulate(
    acc: dict,
    n_spec: dict,
    row_iter,
    spec_lookup: dict,
    cf_lookup: dict,
    group_keys: list,
    inst_type: str | None,
    superklass: str | None,
    prec_type: str | None,
    min_intensity: float,
    n_rows: int,
    mz_bin_size: float = 0.1,
    dag_loader: "DagLoader | None" = None,
) -> None:
    """Stream rows and accumulate missing peak counts directly — no intermediate lists."""
    for row in tqdm(row_iter, total=n_rows, leave=False, unit="spec"):
        spec_id = row.spec_id
        mol_id = row.mol_id

        meta = spec_lookup.get(spec_id)
        if meta is None:
            continue
        if inst_type and meta["inst_type"] != inst_type:
            continue
        if prec_type and meta["prec_type"] != prec_type:
            continue

        cf = cf_lookup.get(mol_id, {})
        if superklass and cf.get("superklass") != superklass:
            continue

        true_peaks = meta["peaks"]
        if not true_peaks:
            continue

        # Build group key
        key_parts = []
        for k in group_keys:
            if k == "subklass":
                key_parts.append(cf.get("subklass", "unknown"))
            elif k == "prec_type":
                key_parts.append(meta["prec_type"])
        group_key = tuple(key_parts) if len(key_parts) > 1 else key_parts[0]

        n_spec[group_key] += 1

        # Vectorised peak classification
        true_mzs = np.array([p[0] for p in true_peaks])
        true_ints = np.array([p[1] for p in true_peaks], dtype=float)
        true_ints /= true_ints.max()

        # dag_mzs: observed m/z values the BFS DAG can explain.
        # Prefer raw DAG files (true BFS coverage). Fall back to the model's
        # frag_formula_peak_mzs, which only contains peaks with non-zero
        # predicted intensity — so "absent" becomes "model never predicts it".
        if dag_loader is not None:
            dag_mzs = dag_loader.get(mol_id, meta["prec_type"])
            if dag_mzs is None:
                continue  # mol not in tar — skip rather than misclassify
        else:
            raw = row.frag_formula_peak_mzs
            dag_mzs = np.array(raw) if isinstance(raw, (list, np.ndarray)) and len(raw) > 0 else None

        pred_mzs = row.pred_mzs
        pred_ints = row.pred_ints
        if isinstance(pred_mzs, (list, np.ndarray)) and len(pred_mzs) > 0:
            pred_mzs = np.array(pred_mzs)
            pred_ints = np.array(pred_ints, dtype=float)
            pred_ints = pred_ints / pred_ints.max()
        else:
            pred_mzs = pred_ints = None

        for mz, intensity in zip(true_mzs, true_ints):
            if intensity < min_intensity:
                continue

            in_dag = dag_mzs is not None and np.any(np.abs(dag_mzs - mz) < TOL)

            if pred_mzs is not None:
                matches = np.abs(pred_mzs - mz) < TOL
                pred_int_at_mz = float(pred_ints[matches].max()) if matches.any() else 0.0
            else:
                pred_int_at_mz = 0.0

            if not in_dag:
                cat = "A"  # dag missing
            elif pred_int_at_mz < 0.05:
                cat = "B"  # underpredicted
            else:
                continue  # ok — skip to save memory

            mz_bin = mz_to_bin(mz, mz_bin_size)
            entry = acc[group_key][mz_bin]
            entry[cat + "_n"] = entry.get(cat + "_n", 0) + 1
            entry[cat + "_int"] = entry.get(cat + "_int", 0.0) + intensity
            entry[cat + "_pred"] = entry.get(cat + "_pred", 0.0) + pred_int_at_mz


def build_top_df(acc_group: dict, cat: str, top_n: int) -> pd.DataFrame:
    rows = []
    for mz_bin, entry in acc_group.items():
        n = entry.get(cat + "_n", 0)
        if n == 0:
            continue
        rows.append({
            "mz_bin": mz_bin,
            "n_spectra": n,
            "mean_true_intensity": entry[cat + "_int"] / n,
            "mean_pred_intensity": entry[cat + "_pred"] / n,
        })
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("n_spectra", ascending=False)
        .head(top_n)
        .set_index("mz_bin")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, default=Path("data/predicted"))
    parser.add_argument("--pred_prefix", type=str, default="nps_inchikey_nist23_nist23")
    parser.add_argument("--spec_fp", type=Path, default=Path("data/proc/nist23/spec_df.pkl"))
    parser.add_argument("--cf_fp", type=Path, default=Path("data/classyfire/nist23_mol_classyfire.pkl"))
    parser.add_argument(
        "--frag_dp", type=Path, default=None,
        help="Path to a directory of {mol_id}.pkl.bz2 DAG files, or a .tar archive "
             "of the same. When provided, category A uses true BFS coverage instead "
             "of the model's filtered prediction output.",
    )
    parser.add_argument("--n_cvs", type=int, default=9)
    parser.add_argument("--inst_type", type=str, default="OB")
    parser.add_argument("--min_intensity", type=float, default=0.05,
                        help="Min normalised true peak intensity to consider (0–1)")
    parser.add_argument("--min_spectra", type=int, default=20,
                        help="Min spectra in group to report")
    parser.add_argument("--top_n", type=int, default=10,
                        help="Top N missing peaks to show per group")
    parser.add_argument("--superklass", type=str, default=None,
                        help="Filter to this superclass (e.g. 'benzenoids')")
    parser.add_argument("--prec_type", type=str, default=None,
                        help="Filter to one adduct (e.g. '[M+H]+')")
    parser.add_argument("--by_adduct", action="store_true",
                        help="Group by (subklass, prec_type) instead of subklass alone")
    parser.add_argument("--mz_bin_size", type=float, default=0.1,
                        help="m/z bin width in Da (default 0.1)")
    args = parser.parse_args()

    # ── Load lookup tables (small, kept in memory) ────────────────────────────
    print("Loading metadata...")
    with open(args.spec_fp, "rb") as f:
        spec_df = pickle.load(f)
    with open(args.cf_fp, "rb") as f:
        cf_df = pickle.load(f)

    spec_lookup = {
        row.spec_id: {"inst_type": row.inst_type, "prec_type": row.prec_type, "peaks": row.peaks}
        for row in spec_df[["spec_id", "inst_type", "prec_type", "peaks"]].itertuples(index=False)
    }
    cf_lookup = {
        row.mol_id: {"superklass": row.superklass, "klass": row.klass, "subklass": row.subklass}
        for row in cf_df[["mol_id", "superklass", "klass", "subklass"]].itertuples(index=False)
    }
    del spec_df, cf_df

    group_keys = ["subklass", "prec_type"] if args.by_adduct else ["subklass"]

    # acc[group_key][mz_bin] = {A_n, A_int, A_pred, B_n, B_int, B_pred}
    acc: dict = defaultdict(lambda: defaultdict(dict))
    n_spec: dict = defaultdict(int)

    dag_loader = DagLoader(args.frag_dp) if args.frag_dp is not None else None
    if dag_loader is None:
        print(
            "WARNING: --frag_dp not set. Category A uses model prediction output "
            "(frag_formula_peak_mzs), not raw BFS DAG coverage. "
            "Pass --frag_dp for accurate results.",
            flush=True,
        )

    # ── Stream one CV fold at a time ─────────────────────────────────────────
    cv_indices = [i for i in range(args.n_cvs)
                  if (args.pred_dir / f"{args.pred_prefix}_cv{i}.zip").exists()]

    try:
        for cv_idx in tqdm(cv_indices, desc="CV folds", unit="fold"):
            cv_df = iter_cv(args.pred_dir, args.pred_prefix, cv_idx)
            if cv_df is None:
                continue
            accumulate(
                acc, n_spec,
                cv_df.itertuples(index=False),
                spec_lookup, cf_lookup,
                group_keys,
                args.inst_type, args.superklass, args.prec_type,
                args.min_intensity,
                n_rows=len(cv_df),
                mz_bin_size=args.mz_bin_size,
                dag_loader=dag_loader,
            )
            del cv_df  # release memory immediately
    finally:
        if dag_loader is not None:
            dag_loader.close()

    # ── Build results ─────────────────────────────────────────────────────────
    all_results = []
    for group_key, mz_dict in acc.items():
        n = n_spec[group_key]
        if n < args.min_spectra:
            continue

        total_a = sum(e.get("A_n", 0) for e in mz_dict.values())
        total_b = sum(e.get("B_n", 0) for e in mz_dict.values())
        total = total_a + total_b
        dag_miss_rate = total_a / (total + 1e-9)

        subklass = group_key[0] if args.by_adduct else group_key
        prec_type = group_key[1] if args.by_adduct else ""

        all_results.append({
            "subklass": subklass,
            "prec_type": prec_type,
            "n_spectra": n,
            "dag_miss_rate": dag_miss_rate,
            "dag_missing_peaks": build_top_df(mz_dict, "A", args.top_n),
            "underpred_peaks": build_top_df(mz_dict, "B", args.top_n),
        })

    all_results.sort(key=lambda x: -x["dag_miss_rate"])

    # ── Print results ─────────────────────────────────────────────────────────
    adduct_label = (f", prec_type={args.prec_type}" if args.prec_type
                    else (", by_adduct" if args.by_adduct else ""))
    print(f"\n{'=' * 70}")
    print(f"MISSING PEAK ANALYSIS  (inst={args.inst_type}{adduct_label}, min_int={args.min_intensity})")
    print(f"{'=' * 70}")

    for r in all_results:
        adduct_str = f"  adduct={r['prec_type']}" if args.by_adduct else ""
        print(f"\n── {r['subklass']}{adduct_str}"
              f"  (n={r['n_spectra']}, DAG miss rate={r['dag_miss_rate']:.1%}) ──")

        if not r["dag_missing_peaks"].empty:
            print("  [A] Absent from DAG entirely:")
            print("      " + r["dag_missing_peaks"].to_string().replace("\n", "\n      "))

        if not r["underpred_peaks"].empty:
            print("  [B] In DAG but under-predicted:")
            print("      " + r["underpred_peaks"].to_string().replace("\n", "\n      "))

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: DAG miss rate by subclass" + (" × adduct" if args.by_adduct else ""))
    print(f"{'=' * 70}")
    summary = {"subklass": [], "n_spectra": [], "dag_miss_rate": []}
    if args.by_adduct:
        summary["prec_type"] = []
    for r in all_results:
        summary["subklass"].append(r["subklass"])
        summary["n_spectra"].append(r["n_spectra"])
        summary["dag_miss_rate"].append(round(r["dag_miss_rate"], 4))
        if args.by_adduct:
            summary["prec_type"].append(r["prec_type"])
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
