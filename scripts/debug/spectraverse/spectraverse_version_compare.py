"""Compare Spectraverse releases: size, sources, surviving exact duplicates, and what changed.

Reads the Zenodo MGFs (``data/raw/spectraverse/spectraverse-<v>/spectraverse-<v>.mgf``) directly.

1. Per release: spectra, 2D InChIKeys, sources, and byte-identical twins (same InChIKey first block
   and ion mode, identical peak list). The curation paper removes cosine > 0.99 near-duplicates, so
   a release whose dedup ran should hold almost no exact twins.
2. Group-order test (per release): the v1.0.2-era ``step3-5_uniq-select.py`` re-sorts rows by
   (instrument, source) before reading cosine-matrix positions. A group whose members share one
   source and one instrument cannot be reordered, so the index mismatch cannot hit it. If twins
   survive only in reorderable groups of size >= 3, the mismatch is the likely cause. v1.0.0 has
   no SOURCE field, so there the test uses instrument only.
3. Release diff by content: a spectrum is (InChIKey first block, adduct, precursor m/z to 4
   decimals, peak list rounded to 4 / 6 decimals).
4. Optional ``--processed-csv``: which release the old processed table (``data/df/spectraverse_df.csv``)
   was built from, by the same content key.

    python scripts/debug/spectraverse/spectraverse_version_compare.py --versions 1.0.0 1.0.1 1.0.2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

KEYS = {
    "TITLE": "title",
    "SOURCE": "source",
    "INCHIKEY": "inchikey",
    "ADDUCT": "adduct",
    "IONMODE": "ion_mode",
    "PRECURSOR_MZ": "prec_mz",
    "INSTRUMENT_TYPE": "inst",
    "COLLISION_ENERGY_1": "ce1",
    "FOLD": "fold",
}


def _peak_key(lines: list[str]) -> int:
    """Hash of the peak list with m/z rounded to 4 and intensity to 6 decimals."""
    peaks = []
    for line in lines:
        mz, it = line.split()[:2]
        peaks.append((round(float(mz), 4), round(float(it), 6)))
    return hash(tuple(peaks))


def read_mgf(mgf_fp: Path) -> pd.DataFrame:
    """Header fields plus ``raw_key`` (hash of the exact peak text) and ``peak_key`` (rounded)."""
    rows, cur, peaks = [], {}, []
    with open(mgf_fp) as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "BEGIN IONS":
                cur, peaks = {}, []
            elif line == "END IONS":
                cur["n_peaks"] = len(peaks)
                cur["raw_key"] = hash("\n".join(peaks))
                cur["peak_key"] = _peak_key(peaks)
                rows.append(cur)
            elif "=" in line and not line[0].isdigit():
                key, _, value = line.partition("=")
                if key in KEYS:
                    cur[KEYS[key]] = value
            elif line:
                peaks.append(line)
    df = pd.DataFrame(rows)
    for col in ("source", "fold", "inst"):  # v1.0.0 has no SOURCE or FOLD field
        if col not in df.columns:
            df[col] = pd.NA
    df["source"] = df["source"].fillna("unknown")
    df["ik14"] = df["inchikey"].str[:14]
    df["ion_mode"] = df["ion_mode"].str.lower()
    df["prec_mz"] = pd.to_numeric(df["prec_mz"], errors="coerce")
    df["content"] = list(zip(df["ik14"], df["adduct"], df["prec_mz"].round(4), df["peak_key"]))
    return df


def twin_stats(df: pd.DataFrame) -> dict:
    """Byte-identical twins and the group-order test."""
    twin = df.duplicated(["ik14", "ion_mode", "raw_key"], keep=False)
    groups = df.groupby(["ik14", "ion_mode"])
    g = groups.agg(n=("raw_key", "size"), n_src=("source", "nunique"), n_inst=("inst", "nunique"))
    g["has_twin"] = twin.groupby([df["ik14"], df["ion_mode"]]).any()
    g = g[g["n"] >= 2]
    g["fixed_order"] = (g["n_src"] == 1) & (g["n_inst"] == 1)
    big = g[g["n"] >= 3]
    return {
        "spectra": len(df),
        "ik14": df["ik14"].nunique(),
        "spectra_with_exact_twin": int(twin.sum()),
        "share_with_exact_twin": round(float(twin.mean()), 4),
        "groups>=3_with_twin_reorderable": round(float(big.loc[~big.fixed_order, "has_twin"].mean()), 4),
        "groups>=3_with_twin_order_fixed": round(float(big.loc[big.fixed_order, "has_twin"].mean()), 4),
        "groups>=3_reorderable / order_fixed": f"{(~big.fixed_order).sum():,} / {big.fixed_order.sum():,}",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dp", default="data/raw/spectraverse")
    parser.add_argument("--versions", nargs="+", default=["1.0.0", "1.0.1", "1.0.2"])
    parser.add_argument("--processed-csv", default="data/df/spectraverse_df.csv")
    args = parser.parse_args()

    # 1-2. Per release.
    data = {}
    for v in args.versions:
        df = read_mgf(Path(args.raw_dp) / f"spectraverse-{v}" / f"spectraverse-{v}.mgf")
        data[v] = df
        print(f"== v{v}")
        for k, val in twin_stats(df).items():
            print(f"  {k}: {val:,}" if isinstance(val, int) else f"  {k}: {val}")
        print("  sources:", df["source"].value_counts().head(8).to_dict())
        print("  has FOLD field:", round(float(df["fold"].notna().mean()), 3))

    # 3. Pairwise content diff between consecutive releases.
    for a, b in zip(args.versions, args.versions[1:]):
        ca, cb = set(data[a]["content"]), set(data[b]["content"])
        print(f"== v{a} -> v{b}: unique content {len(ca):,} -> {len(cb):,}; "
              f"kept {len(ca & cb):,}, removed {len(ca - cb):,}, added {len(cb - ca):,}")
        added = data[b][data[b]["content"].isin(cb - ca)]
        removed = data[a][data[a]["content"].isin(ca - cb)]
        print("  added by source:", added["source"].value_counts().head(8).to_dict())
        print("  removed by source:", removed["source"].value_counts().head(8).to_dict())

    # 4. Which release the old processed table matches.
    if args.processed_csv and Path(args.processed_csv).exists():
        old = pd.read_csv(args.processed_csv, usecols=["inchi_key", "prec_type", "prec_mz", "peaks"],
                          low_memory=False)
        old_content = set(zip(
            old["inchi_key"].astype(str).str[:14], old["prec_type"],
            pd.to_numeric(old["prec_mz"], errors="coerce").round(4),
            old["peaks"].astype(str).map(lambda s: _peak_key([x for x in s.split("\n") if x.strip()])),
        ))
        print(f"== processed table {args.processed_csv}: {len(old):,} rows, {len(old_content):,} unique")
        for v, df in data.items():
            c = set(df["content"])
            print(f"  share found in v{v}: {len(old_content & c) / len(old_content):.4f}; "
                  f"v{v} share found in table: {len(old_content & c) / len(c):.4f}")


if __name__ == "__main__":
    main()
