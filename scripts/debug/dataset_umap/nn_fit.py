"""Where a structure-only set (HMDB, NP Atlas: ``--dataset``) sits relative to the MS/MS training datasets.

For a seeded sample of the set's molecules (``collect_mols.py`` output; HMDB is the PubChem copy of
the HMDB deposit), the nearest molecule in each MS/MS dataset (Morgan r=2, 2048 bits, Tanimoto),
and in NIST20 ∪ NIST23 HR and in the non-NIST datasets (max over their members). Molecules with a
long aliphatic chain (eight consecutive acyclic CH2) are reported separately: they are mostly the
enumerated lipids that dominate HMDB's predicted entries, a status the PubChem copy does not carry.
Writes ``<slug>_nn.parquet`` (read by ``project.py`` and, for HMDB, ``combo_coverage.py``) and
``<slug>_fit.json``.

Usage (from the repository root, ``umap-viz`` env):
    python -I scripts/debug/dataset_umap/nn_fit.py \
        --mols_fp output/dataset_umap_hmdb_2026_10_17/mols.parquet \
        --out_dp output/dataset_umap_hmdb_2026_10_17 --n 10000 --workers 6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MSMS_REFS, NIST, nearest, similarity_summary, slug  # noqa: E402

LONG_CHAIN = Chem.MolFromSmarts("[CH2;!R]" * 8)


def long_chain(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.HasSubstructMatch(LONG_CHAIN)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols_fp", type=Path, required=True)
    ap.add_argument("--out_dp", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dataset", default="HMDB", help="Dataset name in mols.parquet; outputs <slug>_nn.parquet")
    args = ap.parse_args()

    mols = pd.read_parquet(args.mols_fp)
    name = slug(args.dataset)
    full = mols[mols["dataset"] == args.dataset]
    sample = full.sample(n=min(args.n, len(full)), random_state=args.seed).reset_index(drop=True)
    sample["long_chain"] = [long_chain(s) for s in sample["smiles"]]
    full_long = float(np.mean([long_chain(s) for s in full["smiles"].sample(min(20000, len(full)), random_state=1)]))
    for ref in MSMS_REFS:
        ref_smiles = mols.loc[mols["dataset"] == ref, "smiles"].tolist()
        sample[f"nn_{ref}"] = nearest(sample["smiles"].tolist(), ref_smiles, args.workers)
        print(f"done {ref}", flush=True)
    sample["nn_NIST20∪23"] = sample[[f"nn_{r}" for r in NIST]].max(axis=1)
    sample["nn_non-NIST"] = sample[[f"nn_{r}" for r in MSMS_REFS if r not in NIST]].max(axis=1)
    sample["nn_all"] = sample[["nn_NIST20∪23", "nn_non-NIST"]].max(axis=1)
    sample.to_parquet(args.out_dp / f"{name}_nn.parquet", index=False)

    cols = [f"nn_{r}" for r in MSMS_REFS] + ["nn_NIST20∪23", "nn_non-NIST", "nn_all"]
    report = {"sample": len(sample), "long_chain_frac_sample": round(float(sample["long_chain"].mean()), 3),
              "long_chain_frac_full_estimate": round(full_long, 3)}
    for group, sub in [("all", sample), ("long_chain", sample[sample["long_chain"]]),
                       ("other", sample[~sample["long_chain"]])]:
        report[group] = {"n": len(sub), **{c[3:]: similarity_summary(sub[c]) for c in cols}}
    (args.out_dp / f"{name}_fit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for group in ("all", "long_chain", "other"):
        table = pd.DataFrame(report[group]).drop(columns="n").T
        print(f"\n{group} (n={report[group]['n']})\n{table.to_string()}")
    print({k: report[k] for k in ("long_chain_frac_sample", "long_chain_frac_full_estimate")})


if __name__ == "__main__":
    main()
