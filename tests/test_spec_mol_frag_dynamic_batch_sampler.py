"""
Test SpecMolFragDynamicBatchSampler epoch-aware reset functionality.

Validates:
- Batch computation respects max_num constraint
- set_epoch() triggers re-batching with new randomization
- Same epoch yields deterministic batches
- Different epochs may yield different batches (with shuffled base sampler)
"""

import torch as th
import pytest


def create_mock_dataset_for_dynamic_sampler():
	"""Create mock dataset with frag_pyg for testing SpecMolFragDynamicBatchSampler."""
	
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
	
	# Create 12 samples with varying frag sizes
	# Total ~180k edges, so batching with 50k limit will create multiple batches
	data_list = [
		{"idx": 0, "frag_pyg": MockFragPyG(100, 12000)},    # 12k
		{"idx": 1, "frag_pyg": MockFragPyG(90, 11000)},     # 11k
		{"idx": 2, "frag_pyg": MockFragPyG(110, 13000)},    # 13k
		{"idx": 3, "frag_pyg": MockFragPyG(95, 10000)},     # 10k
		{"idx": 4, "frag_pyg": MockFragPyG(105, 14000)},    # 14k
		{"idx": 5, "frag_pyg": MockFragPyG(88, 9000)},      # 9k
		{"idx": 6, "frag_pyg": MockFragPyG(120, 18000)},    # 18k
		{"idx": 7, "frag_pyg": MockFragPyG(100, 15000)},    # 15k
		{"idx": 8, "frag_pyg": MockFragPyG(85, 8000)},      # 8k
		{"idx": 9, "frag_pyg": MockFragPyG(115, 16000)},    # 16k
		{"idx": 10, "frag_pyg": MockFragPyG(92, 11000)},    # 11k
		{"idx": 11, "frag_pyg": MockFragPyG(110, 13000)},   # 13k
	]
	
	return MockDataset(data_list)


@pytest.fixture
def mock_dataset():
	return create_mock_dataset_for_dynamic_sampler()


@pytest.fixture
def large_dataset():
	# Create dataset with one large sample used in skip_too_big test
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

	data_list = [
		{"idx": 0, "frag_pyg": MockFragPyG(100, 10000)},
		{"idx": 1, "frag_pyg": MockFragPyG(200, 60000)},    # TOO LARGE (> 50k)
		{"idx": 2, "frag_pyg": MockFragPyG(100, 10000)},
	]
	return MockDataset(data_list)


def test_dynamic_batch_sampler_basic(mock_dataset):
	"""Test basic SpecMolFragDynamicBatchSampler functionality."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler

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
	"""Test epoch-aware reset functionality."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler
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
	"""Test SpecMolFragDynamicBatchSampler without base sampler (sequential)."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler

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
	"""Test skip_too_big flag behavior."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler

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


# Pytest will discover and run the test_* functions above.
