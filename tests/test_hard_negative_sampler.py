"""
Simple test and validation script for hard negative samplers.

Run this to verify the samplers work correctly with your dataset.
"""

import pandas as pd
import numpy as np
from fragnnet.dataset.data_sampler import (
	FormulaHardNegativeSampler,
	FormulaHardNegativeGroupedSampler,
	compute_sampler_stats,
)


def test_samplers_with_demo_data():
	"""Test samplers with synthetic demo data."""
	
	print("=" * 60)
	print("Testing Hard Negative Samplers")
	print("=" * 60)
	
	# Create demo spec_df
	spec_data = {
		"spec_id": [f"spec_{i}" for i in range(20)],
		"mol_id": [f"mol_{i % 5}" for i in range(20)],  # 5 unique molecules
		"dataset": ["train"] * 20,
	}
	spec_df = pd.DataFrame(spec_data)
	
	# Create demo mol_df with formulas and inchikeys
	mol_data = {
		"mol_id": [f"mol_{i}" for i in range(5)],
		"formula": [
			"C6H12O6",  # glucose
			"C6H12O6",  # fructose - same formula, different structure
			"C12H22O11",  # sucrose
			"C6H12O6",  # another C6H12O6 isomer
			"C2H6O",  # ethanol
		],
		"inchikey": [
			"WQZGKKKJIJFFOK-VFUOTQFVSA-N",
			"JLGKEXUKUSHXEM-VFUOTQFVSA-N",
			"CXAZRCDVEGUEHA-KBQPJGBRSA-N",
			"QZWZWRQVJJWEOV-VFUOTQFVSA-N",
			"LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
		],
	}
	mol_df = pd.DataFrame(mol_data)
	mol_df.set_index("mol_id", inplace=True)
	
	print("\n--- Dataset Info ---")
	print(f"Spectra: {len(spec_df)}")
	print(f"Molecules: {len(mol_df)}")
	print("\nSpectra DataFrame:")
	print(spec_df.head(10))
	print("\nMolecule DataFrame:")
	print(mol_df)
	
	# Test 1: FormulaHardNegativeSampler
	print("\n" + "=" * 60)
	print("Test 1: FormulaHardNegativeSampler (Triplet Mode)")
	print("=" * 60)
	
	sampler = FormulaHardNegativeSampler(
		spec_df=spec_df,
		mol_df=mol_df,
		num_hard_negatives=2,
		num_random_negatives=1,
		shuffle_groups=False,
		seed=42,
	)
	
	print(f"Total triplet batches: {len(sampler)}")
	print("\nFirst 5 triplet batches:")
	for i, batch in enumerate(sampler):
		if i >= 5:
			break
		print(f"  Batch {i}: {batch}")
	
	# Test 2: FormulaHardNegativeGroupedSampler
	print("\n" + "=" * 60)
	print("Test 2: FormulaHardNegativeGroupedSampler (Grouped Mode)")
	print("=" * 60)
	
	grouped_sampler = FormulaHardNegativeGroupedSampler(
		spec_df=spec_df,
		mol_df=mol_df,
		min_specs_per_formula=2,
		shuffle=False,
		seed=42,
	)
	
	print(f"Total formula groups: {len(grouped_sampler)}")
	print("\nAll formula groups:")
	for i, group in enumerate(grouped_sampler):
		print(f"  Group {i} (size {len(group)}): {group}")
	
	# Test 3: Statistics
	print("\n" + "=" * 60)
	print("Test 3: Sampler Statistics")
	print("=" * 60)
	
	stats = compute_sampler_stats(spec_df, mol_df)
	print("\nSampler Statistics:")
	for key, value in stats.items():
		print(f"  {key}: {value}")
	
	print("\n" + "=" * 60)
	print("All tests passed!")
	print("=" * 60)


def test_with_real_dataset(spec_df_path: str, mol_df_path: str):
	"""Test samplers with real dataset files."""
	
	print("\n" + "=" * 60)
	print("Testing with Real Dataset")
	print("=" * 60)
	
	# Load datasets
	print(f"Loading spec_df from {spec_df_path}...")
	spec_df = pd.read_csv(spec_df_path)
	
	print(f"Loading mol_df from {mol_df_path}...")
	mol_df = pd.read_csv(mol_df_path)
	if "mol_id" in mol_df.columns:
		mol_df.set_index("mol_id", inplace=True)
	
	print(f"\nLoaded {len(spec_df)} spectra and {len(mol_df)} molecules")
	
	# Compute stats
	stats = compute_sampler_stats(spec_df, mol_df)
	print("\nDataset Statistics:")
	for key, value in stats.items():
		print(f"  {key}: {value}")
	
	# Create samplers
	print("\nCreating FormulaHardNegativeSampler...")
	sampler = FormulaHardNegativeSampler(
		spec_df=spec_df,
		mol_df=mol_df,
		num_hard_negatives=3,
		num_random_negatives=2,
		shuffle_groups=True,
		seed=42,
	)
	print(f"Created {len(sampler)} triplet batches")
	
	print("\nCreating FormulaHardNegativeGroupedSampler...")
	grouped_sampler = FormulaHardNegativeGroupedSampler(
		spec_df=spec_df,
		mol_df=mol_df,
		min_specs_per_formula=3,
		shuffle=True,
		seed=42,
	)
	print(f"Created {len(grouped_sampler)} formula groups")
	
	# Show some examples
	print("\nFirst 3 triplet batches:")
	for i, batch in enumerate(sampler):
		if i >= 3:
			break
		print(f"  Batch {i}: {batch}")
	
	print("\nFirst 3 formula groups:")
	for i, group in enumerate(grouped_sampler):
		if i >= 3:
			break
		print(f"  Group {i} (size {len(group)}): {group[:5]}{'...' if len(group) > 5 else ''}")


if __name__ == "__main__":
	import sys
	
	# Test with demo data
	test_samplers_with_demo_data()
	
	# Optionally test with real data
	if len(sys.argv) > 2:
		spec_df_path = sys.argv[1]
		mol_df_path = sys.argv[2]
		test_with_real_dataset(spec_df_path, mol_df_path)
	else:
		print("\n(To test with real data, run: python test_hard_negative_sampler.py <spec_df_path> <mol_df_path>)")
