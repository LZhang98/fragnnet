"""Build Spectraverse fold splits whose TRAIN part drops spectra from named sources.

``spec_df.pkl`` does not keep the raw ``SOURCE`` field, so it is read back from the raw MGF:
record ``i`` (1-based) of ``spectraverse-1.0.2.mgf`` is ``dset_spec_id == i``. The mapping is
written to ``--source-fp`` for per-source evaluation. Val and test ids are copied unchanged, so
held-out scores stay comparable with the source split.

    python preproc_scripts/spectraverse/source_filter_split.py --folds 0 1 --drop-sources MS-DIAL
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_mgf_sources(mgf_fp: str) -> list[str]:
    """Return the ``SOURCE`` value of every record, in file order."""
    sources = []
    current = None
    with open(mgf_fp) as f:
        for line in f:
            if line.startswith("SOURCE="):
                current = line[len("SOURCE=") :].rstrip("\n")
            elif line.startswith("END IONS"):
                if current is None:
                    raise ValueError(f"record {len(sources) + 1} has no SOURCE")
                sources.append(current)
                current = None
    return sources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-fp", default="data/proc/spectraverse/spec_df.pkl")
    parser.add_argument(
        "--mgf-fp", default="data/raw/spectraverse/spectraverse-1.0.2/spectraverse-1.0.2.mgf"
    )
    parser.add_argument("--source-fp", default="data/proc/spectraverse/spec_source.csv")
    parser.add_argument("--src-split-dp", default="data/split/spectraverse_benchmark")
    parser.add_argument("--dst-split-dp", default="data/split/spectraverse_benchmark_nomsdial")
    parser.add_argument("--drop-sources", nargs="+", required=True)
    parser.add_argument("--folds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    # 1. Map spec_id to raw SOURCE through dset_spec_id (the 1-based MGF record index).
    spec = pd.read_pickle(args.spec_fp)[["spec_id", "dset_spec_id", "prec_mz"]]
    sources = read_mgf_sources(args.mgf_fp)
    if len(sources) != len(spec):
        raise ValueError(f"MGF has {len(sources)} records, spec_df has {len(spec)}")
    record = spec["dset_spec_id"].astype(int).to_numpy()
    source = pd.DataFrame({"spec_id": spec["spec_id"], "source": np.array(sources)[record - 1]})
    unknown = sorted(set(args.drop_sources).difference(source["source"]))
    if unknown:
        raise ValueError(f"--drop-sources not in the MGF: {unknown}")
    source.to_csv(args.source_fp, index=False)

    # 2. Drop the named sources from train; copy val/test.
    drop = set(source.spec_id[source.source.isin(args.drop_sources)])
    stats = {"drop_sources": args.drop_sources, "n_spectra": len(source),
             "n_dropped_all_splits": len(drop), "folds": {}}
    for fold in args.folds:
        src = Path(args.src_split_dp) / f"cv{fold}"
        dst = Path(args.dst_split_dp) / f"cv{fold}"
        dst.mkdir(parents=True, exist_ok=True)
        train = pd.read_csv(src / "train_ids.csv")
        kept = train[~train.spec_id.isin(drop)]
        kept.to_csv(dst / "train_ids.csv", index=False)
        for name in ("val_ids.csv", "test_ids.csv"):
            shutil.copyfile(src / name, dst / name)
        stats["folds"][f"cv{fold}"] = {
            "train_before": len(train), "train_after": len(kept),
            "train_dropped": len(train) - len(kept),
            "train_mols_before": int(train.mol_id.nunique()),
            "train_mols_after": int(kept.mol_id.nunique()),
            "train_groups_after": int(kept.group_id.nunique()),
        }
    (Path(args.dst_split_dp) / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
