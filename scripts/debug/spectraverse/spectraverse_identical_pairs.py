"""List every pair of Spectraverse spectra with identical peak lists.

Two spectra are identical when their full peak lists are equal: every m/z and intensity, in
order. ``--decimals`` rounds m/z and intensity first, to also catch copies that differ only in
float formatting. A pair is reported only when both spectra share the molecule (InChIKey first
block; Spectraverse structures carry no stereochemistry) and the adduct, and their instrument type
and CE readings (all of ``COLLISION_ENERGY_1..3``) match. A missing instrument or CE matches
anything. a-b and b-a are the same pair and appear once, with ``id_a`` the earlier record. By
default (``--pairs reference``) each spectrum appears once as ``id_b``, with its earliest identical
partner, so a group of k identical spectra gives k-1 pairs; ``--pairs all`` lists every pair.

Input, one of:

- ``--mgf`` (recommended for sharing): a Zenodo release MGF, e.g.
  ``data/raw/spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf``. Ids are the release
  ``TITLE`` values and all metadata comes from the release headers.
- default: our processed ``spec_df.pkl`` (ids are ``spec_id``; record ``spec_id + 1`` of the
  v1.0.2 MGF).

Outputs (default under ``output/analysis/spectraverse_identical_pairs/<tag>/``):

- ``identical_pairs.csv``: one row per pair, ids and the metadata of both spectra.
- ``identical_pairs.txt``: one block per pair, both headers and the two peak lists side by side.

    python scripts/debug/spectraverse/spectraverse_identical_pairs.py \\
        --mgf data/raw/spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

META = ["source", "molecule", "adduct", "instrument", "ce", "prec_mz", "n_peaks"]
MGF_KEYS = {
    "TITLE": "id",
    "SOURCE": "source",
    "INCHIKEY": "inchikey",
    "ADDUCT": "adduct",
    "INSTRUMENT_TYPE": "instrument",
    "COLLISION_ENERGY_1": "ce1",
    "COLLISION_ENERGY_2": "ce2",
    "COLLISION_ENERGY_3": "ce3",
    "PRECURSOR_MZ": "prec_mz",
}


def load_mgf(mgf_fp: str) -> pd.DataFrame:
    """Release MGF: one row per record, indexed by TITLE, with peaks as (m/z, intensity) tuples."""
    rows, cur, peaks = [], {}, []
    with open(mgf_fp) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                cur, peaks = {}, []
            elif line == "END IONS":
                cur["peaks"] = peaks
                rows.append(cur)
            elif line and line[0].isdigit():
                mz, it = line.split()[:2]
                peaks.append((float(mz), float(it)))
            elif "=" in line:
                key, _, value = line.partition("=")
                if key in MGF_KEYS:
                    cur[MGF_KEYS[key]] = value
    df = pd.DataFrame(rows).set_index("id")
    df["molecule"] = df["inchikey"].str[:14]
    ces = df[["ce1", "ce2", "ce3"]].apply(pd.to_numeric, errors="coerce")
    df["ce"] = ces.apply(
        lambda r: "|".join(f"{v:.1f}" for v in r if pd.notna(v)) or np.nan, axis=1
    )
    df["instrument"] = df["instrument"].replace("nan", np.nan)
    if "source" not in df.columns:  # v1.0.0 has no SOURCE field
        df["source"] = np.nan
    return df


def load_spec_df(spec_fp: str, mol_fp: str, source_fp: str) -> pd.DataFrame:
    """Our processed spec_df, indexed by spec_id, with the same columns as ``load_mgf``."""
    spec = pd.read_pickle(spec_fp)
    spec["spec_id"] = spec["spec_id"].astype(np.int64)
    spec = spec.set_index("spec_id")
    ik = pd.read_pickle(mol_fp).set_index("mol_id")["inchikey_s"]
    spec["molecule"] = spec["mol_id"].map(ik).str[:14]
    spec["source"] = pd.read_csv(source_fp).set_index("spec_id")["source"].loc[spec.index]
    ces = spec[["ace", "ace_extra_1", "ace_extra_2"]]
    spec["ce"] = ces.apply(
        lambda r: "|".join(f"{v:.1f}" for v in r if pd.notna(v) and v != -1) or np.nan, axis=1
    )
    spec["inst_type"] = spec["inst_type"].replace("NA", np.nan)
    return spec.rename(columns={"prec_type": "adduct", "inst_type": "instrument"})


def _key(peaks: list, decimals: int | None) -> tuple:
    if decimals is None:
        return tuple(peaks)
    return tuple((round(mz, decimals), round(it, decimals)) for mz, it in peaks)


def _same(a, b) -> bool:
    """Equality that treats two missing values as equal."""
    return bool((pd.isna(a) and pd.isna(b)) or a == b)


def _compatible(a, b) -> bool:
    """Equal, or missing on either side."""
    return bool(pd.isna(a) or pd.isna(b) or a == b)


def _block(i: int, id_a, id_b, a: pd.Series, b: pd.Series) -> str:
    """Text block: both headers, then the peak lists side by side with a match column."""
    lines = [f"=== pair {i}: {id_a} vs {id_b}"]
    for col in META:
        flag = "" if _same(a[col], b[col]) else "   <-- differs"
        lines.append(f"  {col:10s} {str(a[col]):>40s} | {str(b[col]):<40s}{flag}")
    lines.append(f"  {'m/z':>14s} {'int':>10s} | {'m/z':>14s} {'int':>10s}  same")
    for k in range(max(len(a.peaks), len(b.peaks))):
        pa = a.peaks[k] if k < len(a.peaks) else None
        pb = b.peaks[k] if k < len(b.peaks) else None
        left = f"{pa[0]:14.6f} {pa[1]:10.6f}" if pa else " " * 25
        right = f"{pb[0]:14.6f} {pb[1]:10.6f}" if pb else " " * 25
        lines.append(f"  {left} | {right}  {'yes' if pa == pb else 'NO'}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgf", default=None, help="release MGF; overrides the spec_df inputs")
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument("--mol-fp", default="data/proc/spectraverse/mol_df.pkl")
    parser.add_argument("--source-fp", default="data/proc/spectraverse/spec_source.csv")
    parser.add_argument("--out-dp", default=None,
                        help="default: output/analysis/spectraverse_identical_pairs/<mgf stem or spec_df>")
    parser.add_argument("--decimals", type=int, default=None,
                        help="round m/z and intensity before comparing (default: exact)")
    parser.add_argument(
        "--pairs", choices=["reference", "all"], default="reference",
        help="reference: each spectrum once, with its earliest identical partner (k-1 pairs for a "
             "group of k); all: every unordered pair (k(k-1)/2)",
    )
    parser.add_argument("--text-max-pairs", type=int, default=None,
                        help="cap the number of pairs written to the text file")
    args = parser.parse_args()

    # 1. Load spectra.
    if args.mgf:
        spec = load_mgf(args.mgf)
        tag = Path(args.mgf).stem
    else:
        spec = load_spec_df(args.spec_fp, args.mol_fp, args.source_fp)
        tag = "spec_df"
    spec["n_peaks"] = spec["peaks"].map(len)
    if spec.index.duplicated().any():
        raise ValueError("duplicate spectrum ids in the input")

    # 2. Group identical peak lists and expand each set into pairs (ids in file order).
    order = pd.Series(np.arange(len(spec)), index=spec.index)
    key = spec["peaks"].map(lambda p: _key(p, args.decimals))
    rows = []
    for ids in spec.groupby(key).groups.values():
        if len(ids) < 2:
            continue
        for a, b in combinations(sorted(ids, key=order.get), 2):
            if (spec.at[a, "molecule"], spec.at[a, "adduct"]) != (
                spec.at[b, "molecule"], spec.at[b, "adduct"]
            ):
                continue
            if not (_compatible(spec.at[a, "instrument"], spec.at[b, "instrument"])
                    and _compatible(spec.at[a, "ce"], spec.at[b, "ce"])):
                continue
            rows.append((a, b))
    pairs = pd.DataFrame(rows, columns=["id_a", "id_b"])
    pairs = pairs.sort_values(
        ["id_a", "id_b"], key=lambda col: col.map(order), ignore_index=True
    )
    if args.pairs == "reference":
        # Rows are in file order with id_a earlier than id_b, so the first row naming a spectrum
        # as id_b holds its earliest compatible partner. Each spectrum then appears once as id_b.
        pairs = pairs.drop_duplicates("id_b", keep="first").reset_index(drop=True)
    for col in META:
        pairs[f"{col}_a"] = spec.loc[pairs["id_a"], col].to_numpy()
        pairs[f"{col}_b"] = spec.loc[pairs["id_b"], col].to_numpy()
    for col in ("molecule", "adduct", "instrument", "ce"):
        pairs[f"same_{col}"] = [_same(x, y) for x, y in zip(pairs[f"{col}_a"], pairs[f"{col}_b"])]

    # 3. Write the CSVs and the side-by-side text file.
    out = Path(args.out_dp or f"output/analysis/spectraverse_identical_pairs/{tag}")
    out.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out / "identical_pairs.csv", index=False)
    text_pairs = pairs if args.text_max_pairs is None else pairs.head(args.text_max_pairs)
    with open(out / "identical_pairs.txt", "w") as f:
        f.write(f"# {len(pairs):,} identical pairs from {tag} (decimals={args.decimals}, "
                f"pairs={args.pairs}); "
                f"{len(text_pairs):,} written below\n\n")
        for i, (a, b) in enumerate(zip(text_pairs["id_a"], text_pairs["id_b"]), 1):
            f.write(_block(i, a, b, spec.loc[a], spec.loc[b]))
            f.write("\n")

    # 4. Summary.
    print(f"{tag}: {len(spec):,} spectra; {len(pairs):,} identical pairs from "
          f"{pairs[['id_a', 'id_b']].stack().nunique():,} spectra")
    for col in ("instrument", "ce"):
        print(f"  differ in {col:10s} (missing on one side counts as differ): "
              f"{int((~pairs[f'same_{col}']).sum()):,}")
    print(f"wrote {out}/identical_pairs.csv and identical_pairs.txt")


if __name__ == "__main__":
    main()
