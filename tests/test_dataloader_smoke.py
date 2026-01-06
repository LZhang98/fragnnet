import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

import torch as th
from types import SimpleNamespace
from torch.utils.data import DataLoader
import pandas as pd

from src.fragnnet.dataset.data_sampler import (
    SpecMolFragDynamicBatchSampler,
    FormulaHardNegativeGroupedSampler,
    GroupSampler,
)


class FakeFrag:
    def __init__(self, nodes, edges):
        self.num_nodes = nodes
        self.num_edges = edges


class FakeDataset:
    def __init__(self, sizes):
        self.sizes = sizes

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, idx):
        nodes, edges = self.sizes[idx]
        return {"frag_pyg": SimpleNamespace(num_nodes=nodes, num_edges=edges)}


def test_dynamic_batch_sampler():
    sizes = [(1, 2), (3, 4), (2, 1), (8, 2), (5, 6), (2, 2), (4, 1), (6, 6)]
    ds = FakeDataset(sizes)
    sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by='frag_node')
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=lambda x: x)
    print("DynamicBatchSampler batches:")
    for i, batch in enumerate(dl):
        print(i, batch)
    print("Dynamic batch sampler iteration OK")


def test_formula_grouped_sampler():
    # create a small spec_df and mol_df matching expected format
    spec_df = pd.DataFrame({
        "spec_id": list(range(8)),
        "mol_id": [1, 1, 2, 2, 3, 3, 4, 4],
    })
    mol_df = pd.DataFrame({
        "mol_id": [1, 2, 3, 4],
        "formula": ["C1", "C1", "C2", "C2"],
        "inchikey": ["A", "B", "C", "D"],
    })

    # Data source length should align with spec_df if used
    sizes = [(2, 2)] * len(spec_df)
    ds = FakeDataset(sizes)

    sampler = FormulaHardNegativeGroupedSampler(spec_df=spec_df, mol_df=mol_df, min_specs_per_formula=2, shuffle=True, generator=th.Generator(), data_source=ds)

    # iterate sampler
    print("FormulaHardNegativeGroupedSampler batches:")
    for i, batch in enumerate(sampler):
        print(i, batch)
    print("Formula grouped sampler iteration OK")


if __name__ == '__main__':
    test_dynamic_batch_sampler()
    test_formula_grouped_sampler()
    print("Dataloader smoke tests passed")
