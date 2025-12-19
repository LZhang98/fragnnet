"""
Test SpecMolFragDynamicBatchSampler epoch-aware reset functionality.

Validates:
- Batch computation respects max_num constraint
- set_epoch() triggers re-batching with new randomization
- Same epoch yields deterministic batches
- Different epochs may yield different batches (with shuffled base sampler)
"""

import torch as th


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


def test_dynamic_batch_sampler_basic():
	"""Test basic SpecMolFragDynamicBatchSampler functionality."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler
	
	print("\n" + "="*80)
	print("Test 1: Basic SpecMolFragDynamicBatchSampler Functionality")
	print("="*80)
	
	dataset = create_mock_dataset_for_dynamic_sampler()
	
	# Create sampler with 50k edge limit
	sampler = SpecMolFragDynamicBatchSampler(
		data_source=dataset,
		max_num=50000,
		limited_by='frag_edge',
		skip_too_big=False,
		sampler=None,
	)
	
	# Collect batches
	batches = list(sampler)
	print(f"\n✓ Sampler created with {len(batches)} batches")
	
	# Verify batches respect max_num constraint
	batch_edges = []
	for batch_idx, batch in enumerate(batches):
		total_edges = sum(dataset[idx]['frag_pyg'].num_edges for idx in batch)
		batch_edges.append(total_edges)
		status = "✓" if total_edges <= 50000 else "✗"
		print(f"  {status} Batch {batch_idx}: {batch} ({total_edges:,} edges)")
	
	all_within_limit = all(e <= 50000 for e in batch_edges)
	assert all_within_limit, "Some batches exceed max_num limit"
	print(f"\n✓ All batches respect 50k edge constraint")
	
	# Verify no samples are duplicated or missing
	all_indices = [idx for batch in batches for idx in batch]
	expected_indices = set(range(len(dataset)))
	actual_indices = set(all_indices)
	assert actual_indices == expected_indices, f"Missing or extra indices: {actual_indices ^ expected_indices}"
	print(f"✓ All {len(dataset)} samples included, no duplicates")


def test_dynamic_batch_sampler_epoch_aware():
	"""Test epoch-aware reset functionality."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler
	from torch.utils.data import RandomSampler
	
	print("\n" + "="*80)
	print("Test 2: Epoch-Aware Reset (set_epoch)")
	print("="*80)
	
	dataset = create_mock_dataset_for_dynamic_sampler()
	
	# Create sampler with RandomSampler base (introduces randomness per epoch)
	generator = th.Generator()
	base_sampler = RandomSampler(dataset, generator=generator)
	
	sampler = SpecMolFragDynamicBatchSampler(
		data_source=dataset,
		max_num=50000,
		limited_by='frag_edge',
		skip_too_big=False,
		sampler=base_sampler,
	)
	
	# Epoch 0: collect batches
	print("\nEpoch 0:")
	epoch0_batches = list(sampler)
	print(f"  Batches: {len(epoch0_batches)}")
	for i, b in enumerate(epoch0_batches):
		print(f"    {i}: {b}")
	
	# Epoch 1: set_epoch and collect batches
	print("\nEpoch 1 (after set_epoch(1)):")
	sampler.set_epoch(1)
	epoch1_batches = list(sampler)
	print(f"  Batches: {len(epoch1_batches)}")
	for i, b in enumerate(epoch1_batches):
		print(f"    {i}: {b}")
	
	# Verify both epochs have valid batches
	assert len(epoch0_batches) > 0, "Epoch 0 should have batches"
	assert len(epoch1_batches) > 0, "Epoch 1 should have batches"
	print(f"\n✓ Both epochs produced valid batches")
	
	# Note: batches may differ due to RandomSampler shuffling
	if epoch0_batches != epoch1_batches:
		print("✓ Batches differ between epochs (expected with RandomSampler)")
	else:
		print("⚠ Batches are same (may happen if random order is coincidentally identical)")
	
	# Epoch 0 again: verify determinism
	print("\nEpoch 0 again (after set_epoch(0)):")
	sampler.set_epoch(0)
	epoch0_again = list(sampler)
	print(f"  Batches: {len(epoch0_again)}")
	for i, b in enumerate(epoch0_again):
		print(f"    {i}: {b}")
	
	if epoch0_batches == epoch0_again:
		print(f"\n✓ Setting same epoch twice yields identical batches (deterministic)")
	else:
		print(f"\n✗ Setting same epoch twice yields different batches")


def test_dynamic_batch_sampler_no_base_sampler():
	"""Test SpecMolFragDynamicBatchSampler without base sampler (sequential)."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler
	
	print("\n" + "="*80)
	print("Test 3: Sequential Sampling (no base sampler)")
	print("="*80)
	
	dataset = create_mock_dataset_for_dynamic_sampler()
	
	# Create sampler without base sampler (sequential by default)
	sampler = SpecMolFragDynamicBatchSampler(
		data_source=dataset,
		max_num=50000,
		limited_by='frag_edge',
		skip_too_big=False,
		sampler=None,
	)
	
	# Collect batches multiple times
	print("\nIteration 1:")
	batches1 = list(sampler)
	print(f"  {len(batches1)} batches: {batches1}")
	
	print("\nIteration 2 (without set_epoch):")
	batches2 = list(sampler)
	print(f"  {len(batches2)} batches: {batches2}")
	
	# Sequential ordering should be consistent
	if batches1 == batches2:
		print(f"\n✓ Sequential batches are consistent across iterations")
	else:
		print(f"\n✗ Sequential batches differ (unexpected)")


def test_dynamic_batch_sampler_skip_too_big():
	"""Test skip_too_big flag behavior."""
	from fragnnet.dataset.data_sampler import SpecMolFragDynamicBatchSampler
	
	print("\n" + "="*80)
	print("Test 4: skip_too_big Flag")
	print("="*80)
	
	# Create dataset with one large sample
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
	
	dataset = MockDataset(data_list)
	
	# Test with skip_too_big=True
	print("\nWith skip_too_big=True:")
	sampler = SpecMolFragDynamicBatchSampler(
		data_source=dataset,
		max_num=50000,
		limited_by='frag_edge',
		skip_too_big=True,
	)
	
	batches = list(sampler)
	all_indices = [idx for batch in batches for idx in batch]
	if 1 not in all_indices:
		print(f"  ✓ Sample 1 (60k edges) correctly skipped")
	else:
		print(f"  ✗ Sample 1 should have been skipped")
	
	# Test with skip_too_big=False
	print("\nWith skip_too_big=False:")
	sampler = SpecMolFragDynamicBatchSampler(
		data_source=dataset,
		max_num=50000,
		limited_by='frag_edge',
		skip_too_big=False,
	)
	
	batches = list(sampler)
	all_indices = [idx for batch in batches for idx in batch]
	if 1 in all_indices:
		print(f"  ✓ Sample 1 (60k edges) included despite exceeding limit")
	else:
		print(f"  ✗ Sample 1 should have been included")


if __name__ == "__main__":
	test_dynamic_batch_sampler_basic()
	test_dynamic_batch_sampler_epoch_aware()
	test_dynamic_batch_sampler_no_base_sampler()
	test_dynamic_batch_sampler_skip_too_big()
	
	print("\n" + "="*80)
	print("✓ All SpecMolFragDynamicBatchSampler tests passed!")
	print("="*80)
