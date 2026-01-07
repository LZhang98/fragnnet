import importlib.util
import pathlib
import torch as th

# Load data_sampler directly from source file to avoid package-level imports
data_sampler_path = pathlib.Path(__file__).resolve().parents[1] / "src" / "fragnnet" / "dataset" / "data_sampler.py"
spec = importlib.util.spec_from_file_location("data_sampler", str(data_sampler_path))
data_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_sampler)
DualGroupDynamicBatchSampler = data_sampler.DualGroupDynamicBatchSampler


class DummyFrag:
    def __init__(self, nodes, edges):
        self.num_nodes = nodes
        self.num_edges = edges


class DummyDataset:
    def __init__(self, samples):
        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        return self._samples[idx]


def test_dual_group_dynamic_batch_sampler_nodes_limit():
    # Build small controlled dataset of 6 samples
    samples = [
        {"frag_pyg": DummyFrag(3, 2), "adduct": "A", "mol_id": 1, "formula": "X"},
        {"frag_pyg": DummyFrag(2, 1), "adduct": "A", "mol_id": 1, "formula": "Y"},
        {"frag_pyg": DummyFrag(4, 3), "adduct": "A", "mol_id": 2, "formula": "X"},
        {"frag_pyg": DummyFrag(1, 1), "adduct": "B", "mol_id": 3, "formula": "Z"},
        {"frag_pyg": DummyFrag(5, 4), "adduct": "B", "mol_id": 3, "formula": "Z"},
        {"frag_pyg": DummyFrag(2, 1), "adduct": "C", "mol_id": 4, "formula": "W"},
    ]

    ds = DummyDataset(samples)

    gen = th.Generator()
    gen.manual_seed(42)

    sampler = DualGroupDynamicBatchSampler(
        ds,
        max_num=6,
        limited_by="frag_node",
        skip_too_big=True,
        generator=gen,
    )

    batches = list(iter(sampler))

    # Expect at least one batch and that all indices are included
    assert len(batches) > 0
    included = set(i for b in batches for i in b)
    assert included == set(range(len(ds)))

    # Each batch must respect node budget
    for b in batches:
        total_nodes = sum(samples[i]["frag_pyg"].num_nodes for i in b)
        assert total_nodes <= 6
