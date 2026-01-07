import pandas as pd
import torch as th

from fragnnet.dataset.data_sampler import FormulaHardNegativeDynamicBatchSampler


class MockFragPyG:
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges


class MockDataset:
    def __init__(self, data_list, spec_df, mol_df):
        self.data_list = data_list
        self.spec_df = spec_df
        self.mol_df = mol_df

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def make_dataset_with_large_sample():
    # Create 4 samples where index 1 is extremely large (edges)
    data_list = [
        {"frag_pyg": MockFragPyG(10, 1000)},
        {"frag_pyg": MockFragPyG(120, 60000)},  # TOO LARGE
        {"frag_pyg": MockFragPyG(12, 1200)},
        {"frag_pyg": MockFragPyG(15, 1500)},
    ]

    # spec_df rows correspond to dataset indices and assign mol_ids
    spec_df = pd.DataFrame({"spec_id": list(range(4)), "mol_id": [1, 2, 1, 3]})

    # mol_df maps mol_id to formula and inchikey
    mol_df = pd.DataFrame({
        "mol_id": [1, 2, 3],
        "formula": ["C1", "C1", "C3"],  # note mol_id 2 shares formula C1
        "inchikey": ["A", "B", "C"],
    })

    ds = MockDataset(data_list, spec_df, mol_df)
    return ds


def test_formula_hard_negative_dynamic_sampler_skip_too_big_true():
    ds = make_dataset_with_large_sample()

    sampler = FormulaHardNegativeDynamicBatchSampler(
        data_source=ds,
        spec_df=ds.spec_df,
        mol_df=ds.mol_df,
        max_num=50000,
        limited_by="frag_edge",
        skip_too_big=True,
        min_specs_per_formula=2,
        shuffle=False,
    )

    batches = list(sampler)
    all_indices = [idx for batch in batches for idx in batch]

    # The oversized sample (index 1) should be skipped
    assert 1 not in all_indices


def test_formula_hard_negative_dynamic_sampler_skip_too_big_false():
    ds = make_dataset_with_large_sample()

    sampler = FormulaHardNegativeDynamicBatchSampler(
        data_source=ds,
        spec_df=ds.spec_df,
        mol_df=ds.mol_df,
        max_num=50000,
        limited_by="frag_edge",
        skip_too_big=False,
        min_specs_per_formula=2,
        shuffle=False,
    )

    batches = list(sampler)
    all_indices = [idx for batch in batches for idx in batch]

    # The oversized sample (index 1) should be present when skip_too_big=False
    assert 1 in all_indices
