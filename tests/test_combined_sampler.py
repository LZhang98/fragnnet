"""
Test script for FormulaHardNegativeDynamicBatchSampler.

Validates the combined sampler behavior:
- Layer 1: Formula grouping (hard negatives)
- Layer 2: Dynamic batching (OOM prevention)
"""

import pandas as pd
import numpy as np
import torch as th
from collections import defaultdict


def create_mock_dataset_with_frags():
	"""Create mock dataset with frag_pyg data for testing."""
	
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
	
	# Create mock data matching the example from the plan
	data_list = [
		# Formula C6H12O6 (4 isomers with varying frag sizes)
		{"spec_idx": 0, "mol_id": "mol_A", "frag_pyg": MockFragPyG(100, 8000)},   # glucose
		{"spec_idx": 1, "mol_id": "mol_B", "frag_pyg": MockFragPyG(90, 7500)},    # fructose
		{"spec_idx": 2, "mol_id": "mol_C", "frag_pyg": MockFragPyG(500, 45000)},  # galactose (LARGE)
		{"spec_idx": 3, "mol_id": "mol_D", "frag_pyg": MockFragPyG(110, 9000)},   # mannose
		
		# Formula C10H8 (2 isomers)
		{"spec_idx": 4, "mol_id": "mol_E", "frag_pyg": MockFragPyG(150, 12000)},
		{"spec_idx": 5, "mol_id": "mol_F", "frag_pyg": MockFragPyG(140, 11000)},
		
		# Formula C5H10O5 (2 isomers)
		{"spec_idx": 6, "mol_id": "mol_G", "frag_pyg": MockFragPyG(80, 6000)},
		{"spec_idx": 7, "mol_id": "mol_H", "frag_pyg": MockFragPyG(70, 5000)},
		
		# Single formula (should be filtered out by min_specs_per_formula=2)
		{"spec_idx": 8, "mol_id": "mol_I", "frag_pyg": MockFragPyG(50, 3000)},  # C3H6O
		
		# Another large formula group to test splitting
		{"spec_idx": 9, "mol_id": "mol_J", "frag_pyg": MockFragPyG(200, 20000)},   # C8H10N4O2
		{"spec_idx": 10, "mol_id": "mol_K", "frag_pyg": MockFragPyG(210, 22000)},  # C8H10N4O2
		{"spec_idx": 11, "mol_id": "mol_L", "frag_pyg": MockFragPyG(190, 18000)},  # C8H10N4O2
	]
	
	return MockDataset(data_list)


def create_mock_dataframes():
	"""Create mock spec_df and mol_df for testing."""
	
	# Important: spec_df index must match dataset index
	spec_data = {
		"spec_id": [f"spec_{i}" for i in range(12)],
		"mol_id": [f"mol_{c}" for c in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]],
		"dataset": ["train"] * 12,
	}
	spec_df = pd.DataFrame(spec_data, index=list(range(12)))
	
	mol_data = {
		"mol_id": [f"mol_{c}" for c in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]],
		"formula": [
			"C6H12O6",   # A - glucose
			"C6H12O6",   # B - fructose
			"C6H12O6",   # C - galactose
			"C6H12O6",   # D - mannose
			"C10H8",     # E
			"C10H8",     # F
			"C5H10O5",   # G
			"C5H10O5",   # H
			"C3H6O",     # I (single, will be filtered)
			"C8H10N4O2", # J
			"C8H10N4O2", # K
			"C8H10N4O2", # L
		],
		"inchikey": [
			"GLUCOSE-INCHIKEY",
			"FRUCTOSE-INCHIKEY",
			"GALACTOSE-INCHIKEY",
			"MANNOSE-INCHIKEY",
			"NAPHTHALENE-1-INCHIKEY",
			"NAPHTHALENE-2-INCHIKEY",
			"RIBOSE-1-INCHIKEY",
			"RIBOSE-2-INCHIKEY",
			"ACETONE-INCHIKEY",
			"CAFFEINE-1-INCHIKEY",
			"CAFFEINE-2-INCHIKEY",
			"CAFFEINE-3-INCHIKEY",
		],
	}
	mol_df = pd.DataFrame(mol_data)
	mol_df.set_index("mol_id", inplace=True)
	
	return spec_df, mol_df


def test_combined_sampler():
	"""Test FormulaHardNegativeDynamicBatchSampler behavior."""
	
	print("=" * 80)
	print("Testing FormulaHardNegativeDynamicBatchSampler")
	print("=" * 80)
	
	# Setup
	dataset = create_mock_dataset_with_frags()
	spec_df, mol_df = create_mock_dataframes()
	
	print(f"\nDataset size: {len(dataset)}")
	print(f"Spec DataFrame: {len(spec_df)} rows")
	print(f"Mol DataFrame: {len(mol_df)} rows")
	
	# Import sampler from data_sampler (consolidated location)
	from fragnnet.dataset.data_sampler import FormulaHardNegativeDynamicBatchSampler
	
	# Create sampler with max_num=50000 edges (from the plan example)
	print("\n" + "-" * 80)
	print("Creating sampler with max_num=50000 edges")
	print("-" * 80)
	
	sampler = FormulaHardNegativeDynamicBatchSampler(
		data_source=dataset,
		spec_df=spec_df,
		mol_df=mol_df,
		max_num=50000,
		limited_by='frag_edge',
		skip_too_big=False,
		min_specs_per_formula=2,
		shuffle=False,  # Disable shuffle for predictable testing
	)
	
	print(f"\nSampler created with {len(sampler)} batches")
	
	# Collect and analyze batches
	batches = list(sampler)
	print(f"\nGenerated {len(batches)} batches")
	
	# Analyze each batch
	print("\n" + "=" * 80)
	print("Batch Analysis")
	print("=" * 80)
	
	formula_map = mol_df['formula'].to_dict()
	
	for batch_idx, batch_indices in enumerate(batches):
		print(f"\nBatch {batch_idx}:")
		print(f"  Indices: {batch_indices}")
		print(f"  Size: {len(batch_indices)} specs")
		
		# Get formulas for all specs in batch
		batch_mol_ids = [spec_df.iloc[idx]['mol_id'] for idx in batch_indices]
		batch_formulas = [formula_map[mol_id] for mol_id in batch_mol_ids]
		
		# Check if all formulas are the same (hard negative property)
		unique_formulas = set(batch_formulas)
		if len(unique_formulas) == 1:
			print(f"  ✓ All specs have formula: {list(unique_formulas)[0]} (HARD NEGATIVES)")
		else:
			print(f"  ✗ WARNING: Mixed formulas in batch: {unique_formulas}")
		
		# Calculate total frag edges
		total_edges = sum(dataset[idx]['frag_pyg'].num_edges for idx in batch_indices)
		print(f"  Total frag edges: {total_edges:,}")
		
		if total_edges > 50000:
			print(f"  ⚠ WARNING: Batch exceeds max_num (50000 edges)")
		else:
			print(f"  ✓ Within max_num constraint (≤50000 edges)")
		
		# Show individual spec details
		for idx in batch_indices:
			mol_id = spec_df.iloc[idx]['mol_id']
			formula = formula_map[mol_id]
			edges = dataset[idx]['frag_pyg'].num_edges
			print(f"    - spec_idx={idx}, mol_id={mol_id}, formula={formula}, edges={edges:,}")
	
	# Validate expected behavior from plan
	print("\n" + "=" * 80)
	print("Validation Against Expected Behavior")
	print("=" * 80)
	
	# Expected from plan:
	# Batch 0: [0, 1]    - C6H12O6 isomers, 15.5k edges
	# Batch 1: [2]       - C6H12O6 isomer (too large), 45k edges
	# Batch 2: [3]       - C6H12O6 isomer (leftover), 9k edges
	# Batch 3: [4, 5]    - C10H8 isomers, 23k edges
	# Batch 4: [6, 7]    - C5H10O5 isomers, 11k edges
	# Batch 5+: C8H10N4O2 group (might be split or together)
	
	print("\nExpected behavior checks:")
	
	# Check 1: All batches respect formula grouping
	all_same_formula = True
	for batch_idx, batch_indices in enumerate(batches):
		batch_mol_ids = [spec_df.iloc[idx]['mol_id'] for idx in batch_indices]
		batch_formulas = [formula_map[mol_id] for mol_id in batch_mol_ids]
		if len(set(batch_formulas)) > 1:
			all_same_formula = False
			print(f"  ✗ Batch {batch_idx} has mixed formulas")
	
	if all_same_formula:
		print("  ✓ CHECK 1 PASSED: All batches contain only hard negatives (same formula)")
	else:
		print("  ✗ CHECK 1 FAILED: Some batches have mixed formulas")
	
	# Check 2: All batches respect max_num constraint (or are single large specs)
	all_within_limit = True
	for batch_idx, batch_indices in enumerate(batches):
		total_edges = sum(dataset[idx]['frag_pyg'].num_edges for idx in batch_indices)
		if total_edges > 50000 and len(batch_indices) > 1:
			all_within_limit = False
			print(f"  ✗ Batch {batch_idx} exceeds max_num with multiple specs")
	
	if all_within_limit:
		print("  ✓ CHECK 2 PASSED: All batches respect max_num constraint")
	else:
		print("  ✗ CHECK 2 FAILED: Some batches exceed max_num")
	
	# Check 3: spec_idx=8 (C3H6O, single formula) should be filtered out
	all_indices = [idx for batch in batches for idx in batch]
	if 8 not in all_indices:
		print("  ✓ CHECK 3 PASSED: Single-formula spec (idx=8) correctly filtered out")
	else:
		print("  ✗ CHECK 3 FAILED: Single-formula spec (idx=8) should be filtered")
	
	# Check 4: Large formula group (C6H12O6) should be split appropriately
	c6h12o6_batches = []
	for batch_idx, batch_indices in enumerate(batches):
		batch_mol_ids = [spec_df.iloc[idx]['mol_id'] for idx in batch_indices]
		batch_formulas = [formula_map[mol_id] for mol_id in batch_mol_ids]
		if "C6H12O6" in batch_formulas:
			c6h12o6_batches.append((batch_idx, batch_indices))
	
	print(f"  C6H12O6 formula split across {len(c6h12o6_batches)} batches:")
	for batch_idx, batch_indices in c6h12o6_batches:
		total_edges = sum(dataset[idx]['frag_pyg'].num_edges for idx in batch_indices)
		print(f"    Batch {batch_idx}: {batch_indices} ({total_edges:,} edges)")
	
	if len(c6h12o6_batches) >= 2:
		print("  ✓ CHECK 4 PASSED: Large formula group correctly split across multiple batches")
	else:
		print("  ⚠ CHECK 4: Large formula group behavior may vary")
	
	print("\n" + "=" * 80)
	print("Test Complete!")
	print("=" * 80)

