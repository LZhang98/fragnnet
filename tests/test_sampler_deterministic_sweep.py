import os
import sys
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
    spec_df = pd.DataFrame({
        "spec_id": list(range(12)),
        "mol_id": [1, 1, 2, 2, 3, 1, 4, 4, 4, 5, 5, 6],
    })

    mol_df = pd.DataFrame(
        {
            "mol_id": [1, 2, 3, 4, 5, 6],
            "formula": ["C1", "C1", "C2", "C2", "C3", "C4"],
            "inchikey": ["A", "B", "C", "D", "E", "F"],
        }
    )
    return spec_df, mol_df


def check_seed(seed):
    spec_df, mol_df = make_dfs()

    g1 = th.Generator()
    g1.manual_seed(seed)
    g2 = th.Generator()
    g2.manual_seed(seed)

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

    if out1 != out2:
        print(f"Seed {seed}: FormulaHardNegativeSampler mismatch")
        return False

    # grouped sampler
    gg1 = th.Generator()
    gg1.manual_seed(seed)
    gg2 = th.Generator()
    gg2.manual_seed(seed)

    gs1 = FormulaHardNegativeGroupedSampler(
        spec_df=spec_df,
        mol_df=mol_df,
        min_specs_per_formula=2,
        shuffle=True,
        generator=gg1,
        data_source=None,
    )
    gs2 = FormulaHardNegativeGroupedSampler(
        spec_df=spec_df,
        mol_df=mol_df,
        min_specs_per_formula=2,
        shuffle=True,
        generator=gg2,
        data_source=None,
    )

    outg1 = list(gs1)
    outg2 = list(gs2)

    if outg1 != outg2:
        print(f"Seed {seed}: FormulaHardNegativeGroupedSampler mismatch")
        return False

    print(f"Seed {seed}: OK")
    return True


if __name__ == '__main__':
    seeds = [0, 1, 42, 123, 999, 2026]
    all_ok = True
    for s in seeds:
        ok = check_seed(s)
        all_ok = all_ok and ok
    if not all_ok:
        raise SystemExit(1)
    print("Sweep passed for all seeds")
