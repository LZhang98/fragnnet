"""Count measured same-group element-swap pairs (Br->Cl, I->Br, Se->S, As->P, Si->C, ...).

For every molecule carrying the heavier/rarer element, replace ONE atom of it by its cousin
(and separately ALL atoms), canonicalize to the 2D InChIKey block and look the product up among
measured molecules. Reports pairs by split and how many share adduct+instrument(+frag mode) and
the full acquisition key (incl. raw CE).

Usage (from the repository root, where data/ lives):
    python scripts/debug/masster_memorization/element_swap_census.py --template_dp config \
        --config config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
        > output/element_swap_census/nist20_i1.json
"""

import argparse
import json
import sys
from collections import defaultdict

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

from fragnnet.config import load_config  # noqa: E402
from fragnnet.dataset.base_dataset import BaseDataset  # noqa: E402

# Raw CE columns that define the recorded CE signature.
RAW_CE_COLS = ("ce_type", "nce", "nce_extra_1", "nce_extra_2", "ace", "ace_extra_1", "ace_extra_2")


def _ce_token(value):
    """Format one raw CE field, keeping missingness explicit."""
    if pd.isna(value) or value == -1:
        return "NA"
    return repr(float(value))


def acquisition_keys(spec_df, raw_ce_df):
    """Per-spectrum ``adduct|instrument|frag_mode|ce_type|nce=...|ace=...`` from raw metadata."""
    raw = raw_ce_df.loc[spec_df["spec_id"].to_numpy()]
    keys = []
    for meta, ce in zip(
        spec_df[["prec_type", "inst_type", "frag_mode"]].itertuples(index=False, name=None),
        raw[list(RAW_CE_COLS)].itertuples(index=False, name=None),
    ):
        ce_type = "NA" if pd.isna(ce[0]) else str(ce[0])
        nce = ",".join(_ce_token(v) for v in ce[1:4])
        ace = ",".join(_ce_token(v) for v in ce[4:7])
        meta_str = "|".join("NA" if pd.isna(v) else str(v) for v in meta)
        keys.append(f"{meta_str}|{ce_type}|nce={nce}|ace={ace}")
    return keys


SWAPS = [
    ("Br", "Cl"),
    ("I", "Br"),
    ("I", "Cl"),
    ("Se", "S"),
    ("As", "P"),
    ("Si", "C"),
    ("Te", "Se"),
]


def key2d(mol):
    try:
        ik = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return ik.split("-")[0] if ik else None


def swapped(mol, src, dst):
    """Yield (mode, key) for each single-site swap and for the all-site swap."""
    dst_z = Chem.GetPeriodicTable().GetAtomicNumber(dst)
    idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == src]
    out = []
    for sites, mode in [([i], "single") for i in idxs] + ([(idxs, "all")] if len(idxs) > 1 else []):
        rw = Chem.RWMol(mol)
        for i in sites:
            rw.GetAtomWithIdx(i).SetAtomicNum(dst_z)
            rw.GetAtomWithIdx(i).SetNoImplicit(False)
        try:
            m = rw.GetMol()
            Chem.SanitizeMol(m)
        except Exception:
            continue
        k = key2d(m)
        if k:
            out.append((mode, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_dp", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--pairs_csv", help="Optional path for the per-pair table")
    args = ap.parse_args()
    cfg = load_config(args.template_dp, args.config, None)
    raw = pd.read_pickle(cfg["spec_fp"])
    raw_ce = raw.set_index("spec_id")[list(RAW_CE_COLS)]
    split_of, conds, smiles_of = {}, defaultdict(set), {}
    for split in ("train", "val", "test"):
        spec_df, mol_df, *_ = BaseDataset._setup_dfs(
            spec_fp_or_df=raw.copy(),
            mol_fp_or_df=cfg["mol_fp"],
            split_dp=cfg["split_dp"],
            splits=[split],
            subsample_params=cfg["subsample_params"],
            spec_params=cfg["spec_params"],
        )
        acq = acquisition_keys(spec_df, raw_ce)
        smi = mol_df.loc[spec_df["mol_id"].to_numpy(), "smiles"].to_numpy()
        cols = [c for c in ("prec_type", "inst_type", "frag_mode") if c in spec_df.columns]
        coarse = spec_df[cols].astype(str).agg("|".join, axis=1).to_numpy()
        for s, a, c in zip(smi, acq, coarse):
            smiles_of.setdefault(s, None)
            conds[s].add(("acq", a))
            conds[s].add(("coarse", c))
            split_of[s] = split
    # Molecule identity = 2D InChIKey block.
    by_key, mol_split, mol_conds, mols = {}, {}, defaultdict(set), {}
    for s in smiles_of:
        m = Chem.MolFromSmiles(s)
        k = key2d(m) if m is not None else None
        if not k:
            continue
        by_key.setdefault(k, s)
        mols[k] = m
        mol_split[k] = split_of[s]
        mol_conds[k] |= conds[s]
    res, all_pairs = {}, []
    for src, dst in SWAPS:
        carriers = [k for k, m in mols.items() if any(a.GetSymbol() == src for a in m.GetAtoms())]
        rows = []
        for k in carriers:
            for mode, pk in swapped(mols[k], src, dst):
                if pk in mol_split and pk != k:
                    shared = mol_conds[k] & mol_conds[pk]
                    rows.append(
                        {
                            "src_key": k,
                            "dst_key": pk,
                            "mode": mode,
                            "src_split": mol_split[k],
                            "dst_split": mol_split[pk],
                            "share_coarse": any(t == "coarse" for t, _ in shared),
                            "share_acq": any(t == "acq" for t, _ in shared),
                            "src_smiles": by_key[k],
                            "dst_smiles": by_key[pk],
                        }
                    )
        df = pd.DataFrame(rows).drop_duplicates(["src_key", "dst_key"]) if rows else pd.DataFrame()
        all_pairs.append(df.assign(swap=f"{src}->{dst}"))
        carrier_split = pd.Series([mol_split[k] for k in carriers]).value_counts().to_dict()
        r = {"carriers_by_split": carrier_split, "pairs": len(df)}
        if len(df):
            tt = df[(df.src_split == "train") & (df.dst_split == "train")]
            r.update(
                {
                    "train_train_pairs": len(tt),
                    "train_train_share_coarse": int(tt.share_coarse.sum()),
                    "train_train_share_acq": int(tt.share_acq.sum()),
                    "train_train_distinct_src": int(tt.src_key.nunique()),
                    "heldout_src_with_train_partner": int(
                        df[(df.src_split != "train") & (df.dst_split == "train")].src_key.nunique()
                    ),
                    "pairs_by_split": df.groupby(["src_split", "dst_split"])
                    .size()
                    .rename(lambda x: x)
                    .to_dict(),
                    "examples": df.head(3)[
                        ["src_smiles", "dst_smiles", "share_acq"]
                    ].values.tolist(),
                }
            )
            r["pairs_by_split"] = {f"{a}->{b}": int(v) for (a, b), v in r["pairs_by_split"].items()}
        res[f"{src}->{dst}"] = r
    if args.pairs_csv:
        pd.concat(all_pairs).to_csv(args.pairs_csv, index=False)
    json.dump(res, sys.stdout, indent=1, default=str)


if __name__ == "__main__":
    main()
