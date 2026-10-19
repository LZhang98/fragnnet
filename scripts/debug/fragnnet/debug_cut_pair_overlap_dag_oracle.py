from __future__ import annotations

import os
import sys

import h5py
import pandas as pd
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "oracle")
)

from dag_oracle_decomposition import load_test_spec_df
from dag_oracle_scores import score_dag

CACHE_FP = "/tmp/cut-pair-review-BJ5z6X/cache_500mol.pt"
PRED_FP = "output/inference_eval/9t63zils/predictions_val.hdf5"
FRAG_DP = "data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags"
OUT_FP = "/tmp/cut-pair-review-BJ5z6X/dag_oracle_overlap_500mol_val.csv"


def main() -> None:
    cache = torch.load(CACHE_FP, weights_only=False)
    spec_ptr = cache["spec_group_ptr"].numpy()
    usable_ids = {
        int(spec_id)
        for i, spec_id in enumerate(cache["spec_ids"].tolist())
        if spec_ptr[i + 1] > spec_ptr[i]
    }
    with h5py.File(PRED_FP, "r") as pred:
        pred_scores = pd.DataFrame(
            {
                "spec_id": pred["spec_id"][:].astype(int),
                "cos_actual": pred["cos_hun_inst_tol"][:],
            }
        )
    spec_df = load_test_spec_df(
        "data/proc/nist20/spec_df.pkl",
        "data/split/nist20_ma_mi_inchikey",
        "val",
        ["nist20_hr"],
    )
    spec_df = spec_df[spec_df["spec_id"].isin(usable_ids)]
    oracle = score_dag(spec_df, FRAG_DP, n_jobs=8)
    oracle_col = "cos_hun_inst_tol_opt__1_above_prec"
    out = oracle[["spec_id", "mol_id", oracle_col]].rename(columns={oracle_col: "cos_oracle"})
    out = out.merge(pred_scores, on="spec_id", how="inner", validate="one_to_one")
    out["intensity_headroom"] = (out["cos_oracle"] - out["cos_actual"]).clip(lower=0.0)
    out.to_csv(OUT_FP, index=False)
    print(
        {
            "spectra": len(out),
            "molecules": out["mol_id"].nunique(),
            "mean_actual": out["cos_actual"].mean(),
            "mean_oracle": out["cos_oracle"].mean(),
            "mean_intensity_headroom": out["intensity_headroom"].mean(),
            "output": OUT_FP,
        }
    )


if __name__ == "__main__":
    main()
