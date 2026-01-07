import sys
import os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, repo_root)

import importlib.util
import pathlib
import torch as th
from types import SimpleNamespace
from torch.utils.data import DataLoader
import pandas as pd

# Load data_sampler directly to avoid package-level import side-effects
repo_root = pathlib.Path(__file__).resolve().parents[1]
data_sampler_path = repo_root / "src" / "fragnnet" / "dataset" / "data_sampler.py"
spec = importlib.util.spec_from_file_location("data_sampler", str(data_sampler_path))
data_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_sampler)
SpecMolFragDynamicBatchSampler = data_sampler.SpecMolFragDynamicBatchSampler
DualGroupDynamicBatchSampler = data_sampler.DualGroupDynamicBatchSampler
GroupSampler = data_sampler.GroupSampler


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


def test_dual_group_sampler_smoke():
    # small dataset where each sample contains adduct/mol_id/formula
    samples = [
        {"frag_pyg": SimpleNamespace(num_nodes=3, num_edges=2), "adduct": "A", "mol_id": 1, "formula": "F1"},
        {"frag_pyg": SimpleNamespace(num_nodes=2, num_edges=1), "adduct": "A", "mol_id": 1, "formula": "F2"},
        {"frag_pyg": SimpleNamespace(num_nodes=4, num_edges=3), "adduct": "A", "mol_id": 2, "formula": "F1"},
        {"frag_pyg": SimpleNamespace(num_nodes=1, num_edges=1), "adduct": "B", "mol_id": 3, "formula": "F3"},
        {"frag_pyg": SimpleNamespace(num_nodes=5, num_edges=4), "adduct": "B", "mol_id": 3, "formula": "F3"},
        {"frag_pyg": SimpleNamespace(num_nodes=2, num_edges=1), "adduct": "C", "mol_id": 4, "formula": "F4"},
    ]

    class DS:
        def __init__(self, samples):
            self._s = samples

        def __len__(self):
            return len(self._s)

        def __getitem__(self, idx):
            return self._s[idx]

    ds = DS(samples)

    gen = th.Generator()
    gen.manual_seed(123)

    sampler = DualGroupDynamicBatchSampler(ds, max_num=6, limited_by="frag_node", skip_too_big=True, generator=gen)
    batches = list(iter(sampler))
    assert len(batches) > 0
    included = set(i for b in batches for i in b)
    assert included == set(range(len(ds)))
    for b in batches:
        total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
        assert total_nodes <= 6


# Tests are collected by pytest; no direct execution block required.
    test_dynamic_batch_sampler()
