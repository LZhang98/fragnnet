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


def make_grouped_dataset():
    # Create 6 samples with formulas: C1 x3, C2 x2, C3 x1
    data_list = [
        {"frag_pyg": MockFragPyG(10, 1000)},
        {"frag_pyg": MockFragPyG(12, 1100)},
        {"frag_pyg": MockFragPyG(11, 1050)},
        {"frag_pyg": MockFragPyG(13, 1200)},
        {"frag_pyg": MockFragPyG(9, 900)},
        {"frag_pyg": MockFragPyG(8, 800)},
    ]

    # Map spec indices to mol_ids such that formulas are distributed
    # spec 0,1,2 -> formula C1 (mol_id 1,2,1 for variety)
    # spec 3,4 -> formula C2
    # spec 5 -> formula C3 (single)
    spec_df = pd.DataFrame({
        "spec_id": list(range(6)),
        "mol_id": [1, 2, 1, 3, 4, 5],
    })

    mol_df = pd.DataFrame({
        "mol_id": [1, 2, 3, 4, 5],
        "formula": ["C1", "C1", "C2", "C2", "C3"],
        "inchikey": ["A", "B", "D", "E", "F"],
    })

    return MockDataset(data_list, spec_df, mol_df)


def test_formula_groups_are_homogeneous_and_min_specs_enforced():
    ds = make_grouped_dataset()

    sampler = FormulaHardNegativeDynamicBatchSampler(
        data_source=ds,
        spec_df=ds.spec_df,
        mol_df=ds.mol_df,
        max_num=100000,  # large so no splitting by edges
        limited_by="frag_edge",
        skip_too_big=False,
        min_specs_per_formula=2,
        shuffle=False,
    )

    batches = list(sampler)

    # Build mapping from spec idx to formula
    idx_to_formula = {}
    for i, row in ds.spec_df.iterrows():
        mol_id = row["mol_id"]
        formula = ds.mol_df.loc[ds.mol_df["mol_id"] == mol_id, "formula"].iloc[0]
        idx_to_formula[i] = formula

    # Check that every batch contains only a single formula
    for batch in batches:
        formulas_in_batch = {idx_to_formula[idx] for idx in batch}
        assert len(formulas_in_batch) == 1, f"Batch {batch} contains multiple formulas: {formulas_in_batch}"

    # Ensure formulas with fewer than min_specs_per_formula (C3) are skipped
    all_indices = [idx for batch in batches for idx in batch]
    assert 5 not in all_indices, "Spec with formula C3 (single) should be skipped due to min_specs_per_formula=2"
