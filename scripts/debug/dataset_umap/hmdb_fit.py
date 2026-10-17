"""Where HMDB sits relative to the MS/MS training datasets.

For a seeded sample of HMDB molecules (PubChem copy of the HMDB deposit; see ``collect_mols.py``),
the nearest molecule in each dataset (Morgan r=2, 2048 bits, Tanimoto), and in NIST20 ∪ NIST23 HR
and in the non-NIST datasets (max over their members). Molecules with a long aliphatic chain
(eight consecutive acyclic CH2) are reported separately: they are mostly the enumerated lipids
that dominate HMDB's predicted entries, a status the PubChem copy does not carry.

Usage (from the repository root, ``umap-viz`` env):
    python -I scripts/debug/dataset_umap/hmdb_fit.py \
        --mols_fp output/dataset_umap_hmdb_2026_10_17/mols.parquet \
        --out_dp output/dataset_umap_hmdb_2026_10_17 --n 10000 --workers 6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_vs_nist import nearest  # noqa: E402

RDLogger.DisableLog("rdApp.*")

NIST = ["NIST20 HR", "NIST23 HR"]
REFS = ["NIST20 HR", "NIST23 HR", "MassSpecGym", "Spectraverse", "MSnLib", "Enveda-180"]
LONG_CHAIN = Chem.MolFromSmarts("[CH2;!R]" * 8)


def long_chain(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None and mol.HasSubstructMatch(LONG_CHAIN)


def summary(nn: pd.Series) -> dict:
    return {"median": round(float(nn.median()), 3), "frac_ge_0.7": round(float((nn >= 0.7).mean()), 3),
            "frac_lt_0.4": round(float((nn < 0.4).mean()), 3), "frac_identical": round(float((nn >= 0.999).mean()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mols_fp", type=Path, required=True)
    ap.add_argument("--out_dp", type=Path, required=True)
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    mols = pd.read_parquet(args.mols_fp)
    hmdb_all = mols[mols["dataset"] == "HMDB"]
    hmdb = hmdb_all.sample(n=min(args.n, len(hmdb_all)), random_state=args.seed).reset_index(drop=True)
    hmdb["long_chain"] = [long_chain(s) for s in hmdb["smiles"]]
    full_long = float(np.mean([long_chain(s) for s in hmdb_all["smiles"].sample(20000, random_state=1)]))
    for ref in REFS:
        ref_smiles = mols.loc[mols["dataset"] == ref, "smiles"].tolist()
        hmdb[f"nn_{ref}"] = nearest(hmdb["smiles"].tolist(), ref_smiles, args.workers)
        print(f"done {ref}", flush=True)
    hmdb["nn_NIST20∪23"] = hmdb[[f"nn_{r}" for r in NIST]].max(axis=1)
    hmdb["nn_non-NIST"] = hmdb[[f"nn_{r}" for r in REFS if r not in NIST]].max(axis=1)
    hmdb["nn_all"] = hmdb[["nn_NIST20∪23", "nn_non-NIST"]].max(axis=1)
    hmdb.to_parquet(args.out_dp / "hmdb_nn.parquet", index=False)

    cols = [f"nn_{r}" for r in REFS] + ["nn_NIST20∪23", "nn_non-NIST", "nn_all"]
    report = {"sample": len(hmdb), "long_chain_frac_sample": round(float(hmdb["long_chain"].mean()), 3),
              "long_chain_frac_full_estimate": round(full_long, 3)}
    for group, sub in [("all", hmdb), ("long_chain", hmdb[hmdb["long_chain"]]), ("other", hmdb[~hmdb["long_chain"]])]:
        report[group] = {"n": len(sub), **{c[3:]: summary(sub[c]) for c in cols}}
    (args.out_dp / "hmdb_fit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    for group in ("all", "long_chain", "other"):
        table = pd.DataFrame(report[group]).drop(columns="n").T
        print(f"\n{group} (n={report[group]['n']})\n{table.to_string()}")
    print({k: report[k] for k in ("long_chain_frac_sample", "long_chain_frac_full_estimate")})


if __name__ == "__main__":
    main()
