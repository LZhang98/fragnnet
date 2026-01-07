import importlib.util
import pathlib
from types import SimpleNamespace
import torch as th

# Load data_sampler directly to avoid package import side-effects
repo_root = pathlib.Path(__file__).resolve().parents[1]
data_sampler_path = repo_root / "src" / "fragnnet" / "dataset" / "data_sampler.py"
import importlib.util
import pathlib
from types import SimpleNamespace
import torch as th
import pytest

# Load data_sampler directly to avoid package import side-effects
repo_root = pathlib.Path(__file__).resolve().parents[1]
data_sampler_path = repo_root / "src" / "fragnnet" / "dataset" / "data_sampler.py"
spec = importlib.util.spec_from_file_location("data_sampler", str(data_sampler_path))
data_sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_sampler)
SpecMolFragDynamicBatchSampler = data_sampler.SpecMolFragDynamicBatchSampler


class MockFragPyG:
    def __init__(self, num_nodes, num_edges):
        self.num_nodes = num_nodes
        self.num_edges = num_edges


class MockDataset:
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


def create_mock_dataset_for_dynamic_sampler():
    # Create 12 samples with varying frag sizes
    data_list = [
        {"idx": 0, "frag_pyg": MockFragPyG(100, 12000)},
        {"idx": 1, "frag_pyg": MockFragPyG(90, 11000)},
        {"idx": 2, "frag_pyg": MockFragPyG(110, 13000)},
        {"idx": 3, "frag_pyg": MockFragPyG(95, 10000)},
        {"idx": 4, "frag_pyg": MockFragPyG(105, 14000)},
        {"idx": 5, "frag_pyg": MockFragPyG(88, 9000)},
        {"idx": 6, "frag_pyg": MockFragPyG(120, 18000)},
        {"idx": 7, "frag_pyg": MockFragPyG(100, 15000)},
        {"idx": 8, "frag_pyg": MockFragPyG(85, 8000)},
        {"idx": 9, "frag_pyg": MockFragPyG(115, 16000)},
        {"idx": 10, "frag_pyg": MockFragPyG(92, 11000)},
        {"idx": 11, "frag_pyg": MockFragPyG(110, 13000)},
    ]
    return MockDataset(data_list)


@pytest.fixture
def mock_dataset():
    return create_mock_dataset_for_dynamic_sampler()


@pytest.fixture
def large_dataset():
    data_list = [
        {"idx": 0, "frag_pyg": MockFragPyG(100, 10000)},
        {"idx": 1, "frag_pyg": MockFragPyG(200, 60000)},    # TOO LARGE (> 50k)
        {"idx": 2, "frag_pyg": MockFragPyG(100, 10000)},
    ]
    return MockDataset(data_list)


def test_dynamic_batch_sampler_basic(mock_dataset):
    dataset = mock_dataset

    sampler = SpecMolFragDynamicBatchSampler(
        data_source=dataset,
        max_num=50000,
        limited_by='frag_edge',
        skip_too_big=False,
        sampler=None,
    )

    batches = list(sampler)

    # Verify batches respect max_num constraint
    batch_edges = [sum(dataset[idx]['frag_pyg'].num_edges for idx in batch) for batch in batches]
    assert all(e <= 50000 for e in batch_edges), "Some batches exceed max_num limit"

    # Verify no samples are duplicated or missing
    all_indices = [idx for batch in batches for idx in batch]
    expected_indices = set(range(len(dataset)))
    actual_indices = set(all_indices)
    assert actual_indices == expected_indices, f"Missing or extra indices: {actual_indices ^ expected_indices}"


def test_dynamic_batch_sampler_epoch_aware(mock_dataset):
    from torch.utils.data import RandomSampler

    dataset = mock_dataset

    generator = th.Generator()
    base_sampler = RandomSampler(dataset, generator=generator)

    sampler = SpecMolFragDynamicBatchSampler(
        data_source=dataset,
        max_num=50000,
        limited_by='frag_edge',
        skip_too_big=False,
        sampler=base_sampler,
    )

    epoch0_batches = list(sampler)
    sampler.set_epoch(1)
    epoch1_batches = list(sampler)

    assert len(epoch0_batches) > 0
    assert len(epoch1_batches) > 0

    # Setting same epoch should be deterministic
    sampler.set_epoch(0)
    epoch0_again = list(sampler)
    assert epoch0_batches == epoch0_again


def test_dynamic_batch_sampler_no_base_sampler(mock_dataset):
    dataset = mock_dataset

    sampler = SpecMolFragDynamicBatchSampler(
        data_source=dataset,
        max_num=50000,
        limited_by='frag_edge',
        skip_too_big=False,
        sampler=None,
    )

    batches1 = list(sampler)
    batches2 = list(sampler)
    assert batches1 == batches2


def test_dynamic_batch_sampler_skip_too_big(large_dataset):
    dataset = large_dataset

    sampler = SpecMolFragDynamicBatchSampler(
        data_source=dataset,
        max_num=50000,
        limited_by='frag_edge',
        skip_too_big=True,
    )
    batches = list(sampler)
    all_indices = [idx for batch in batches for idx in batch]
    assert 1 not in all_indices

    sampler = SpecMolFragDynamicBatchSampler(
        data_source=dataset,
        max_num=50000,
        limited_by='frag_edge',
        skip_too_big=False,
    )
    batches = list(sampler)
    all_indices = [idx for batch in batches for idx in batch]
    assert 1 in all_indices


def test_specmolfrag_dynamic_batch_nodes_limit_skip_true():
    # small helper DS using SimpleNamespace for frag_pyg
    class DS2:
        def __init__(self, sizes):
            self.sizes = sizes

        def __len__(self):
            return len(self.sizes)

        def __getitem__(self, idx):
            n, e = self.sizes[idx]
            return {"frag_pyg": SimpleNamespace(num_nodes=n, num_edges=e)}

    sizes = [(3, 1), (4, 2), (12, 4), (2, 1)]  # index 2 is too large for max_num=10
    ds = DS2(sizes)
    sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by='frag_node', skip_too_big=True)
    batches = list(iter(sampler))

    included = set(i for b in batches for i in b)
    assert included == {0, 1, 3}

    for b in batches:
        total_nodes = sum(sizes[i][0] for i in b)
        assert total_nodes <= 10


def test_specmolfrag_dynamic_batch_nodes_limit_skip_false_all_included_and_overflow():
    class DS2:
        def __init__(self, sizes):
            self.sizes = sizes

        def __len__(self):
            return len(self.sizes)

        def __getitem__(self, idx):
            n, e = self.sizes[idx]
            return {"frag_pyg": SimpleNamespace(num_nodes=n, num_edges=e)}

    sizes = [(3, 1), (4, 2), (12, 4), (2, 1)]
    ds = DS2(sizes)
    sampler = SpecMolFragDynamicBatchSampler(data_source=ds, max_num=10, limited_by='frag_node', skip_too_big=False)
    batches = list(iter(sampler))

    included = set(i for b in batches for i in b)
    assert included == set(range(len(ds)))

    # When skip_too_big is False, the large sample should be included and cause at least one batch to exceed max
    assert any(sum(sizes[i][0] for i in b) > 10 for b in batches)
