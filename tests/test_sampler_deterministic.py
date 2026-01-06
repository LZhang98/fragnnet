import os
import sys
# ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

import pandas as pd
import torch as th
import random

from src.fragnnet.dataset.data_sampler import (
    FormulaHardNegativeSampler,
    FormulaHardNegativeGroupedSampler,
)

def make_dfs():
    # small example dataset
    spec_df = pd.DataFrame({
        "spec_id": list(range(6)),
        "mol_id": [1, 1, 2, 2, 3, 1],
    })

    mol_df = pd.DataFrame(
        {
            "mol_id": [1, 2, 3],
            "formula": ["C1", "C1", "C2"],
            "inchikey": ["A", "B", "C"],
        }
    )
    return spec_df, mol_df


def test_formula_hard_negative_sampler():
    spec_df, mol_df = make_dfs()

    g1 = th.Generator()
    g1.manual_seed(123)
    g2 = th.Generator()
    g2.manual_seed(123)

    s1 = FormulaHardNegativeSampler(
        spec_df=spec_df,
        mol_df=mol_df,
        num_hard_negatives=1,
        num_random_negatives=1,
        shuffle_groups=True,
        generator=g1,
    )

    s2 = FormulaHardNegativeSampler(
        spec_df=spec_df,
        mol_df=mol_df,
        num_hard_negatives=1,
        num_random_negatives=1,
        shuffle_groups=True,
        generator=g2,
    )

    out1 = list(s1)
    out2 = list(s2)

    assert out1 == out2, f"FormulaHardNegativeSampler not deterministic: {out1} != {out2}"
    print("FormulaHardNegativeSampler determinism: OK")


def test_formula_grouped_sampler():
    spec_df, mol_df = make_dfs()

    g1 = th.Generator()
    g1.manual_seed(999)
    g2 = th.Generator()
    g2.manual_seed(999)

    gs1 = FormulaHardNegativeGroupedSampler(
        spec_df=spec_df,
        mol_df=mol_df,
        min_specs_per_formula=2,
        shuffle=True,
        generator=g1,
        data_source=None,
    )

    gs2 = FormulaHardNegativeGroupedSampler(
        spec_df=spec_df,
        mol_df=mol_df,
        min_specs_per_formula=2,
        shuffle=True,
        generator=g2,
        data_source=None,
    )

    out1 = list(gs1)
    out2 = list(gs2)

    assert out1 == out2, f"FormulaHardNegativeGroupedSampler not deterministic: {out1} != {out2}"
    print("FormulaHardNegativeGroupedSampler determinism: OK")


if __name__ == "__main__":
    test_formula_hard_negative_sampler()
    test_formula_grouped_sampler()
    print("All sampler determinism tests passed.")
