
"""
Data samplers for spectrum-molecule-fragment datasets.

Includes:
- SpecMolFragDynamicBatchSampler: Dynamic batching to avoid OOM
- GroupSampler: Sample K specs per group_id
- get_group_sampler: WeightedRandomSampler factory
"""

from typing import Iterator, List, Optional
import torch as th
from torch.utils.data import Sampler, BatchSampler, WeightedRandomSampler
from tqdm import tqdm

import numpy as np
import random
import pandas as pd
import torch as th
from typing import Dict, List, Optional, Tuple, Iterator
from torch.utils.data import Sampler, DataLoader, BatchSampler
from collections import defaultdict
from tqdm import tqdm

#####################################################
# Batch Samplers
#####################################################

class SpecMolFragDynamicBatchSampler(BatchSampler):
	"""Dynamically adds samples to a mini-batch up to a maximum size either based on number of nodes on frag DAG or number of edges on frag DAG.
	 This is used to avoid CUDA OOM errors, implmentation is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
	 This should have the same random sampling behavior as RandomSampler
	"""
	def __init__(self, data_source, max_num: int, limited_by: str = 'frag_edge',
			  	 	skip_too_big: bool = False, num_samples: Optional[int] = None, 
					return_batch_at: int = 0, sampler: Optional[Sampler] = None) -> None:
		"""Initialize dynamic batch sampler.

		Args:
			data_source: Dataset with frag_pyg data
			max_num (int): Maximum number of nodes or edges per batch
			limited_by (str): 'frag_node' or 'frag_edge' (default: 'frag_edge')
			skip_too_big (bool): Skip samples larger than max_num (default: False)
			num_samples (Optional[int]): Number of samples to draw (default: all)
			return_batch_at (int): Return batch every N samples for gradient accumulation
			sampler (Optional[Sampler]): Base sampler (e.g., RandomSampler, GroupSampler)

		Raises:
			ValueError: If max_num <= 0 or limited_by not in ['frag_node', 'frag_edge']
		"""
		if not isinstance(max_num, int) or max_num <= 0:
			raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
		if limited_by not in ['frag_node', 'frag_edge']:
			raise ValueError(f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}').")

		if num_samples is None:
			num_samples = len(data_source)

		self.data_source = data_source
		self._max_num = max_num
		self._limited_by = limited_by
		self._skip_too_big = skip_too_big
		self._max_sampling_step = num_samples
		self._return_batch_at = return_batch_at
		self._batches = []
		self._data_meta = []
		self.sampler = sampler
		self._epoch = 0
		self._pre_load_batches()
		self._pre_compute_batches()

	def _pre_load_batches(self):
		"""Pre-load and cache frag graph metadata (nodes, edges) for each sample.
		
		Optimizations:
		- Only load metadata once (no reloading across epochs)
		- Use compact tuple storage instead of dict access
		- Disable tqdm for faster loading
		"""
		# get data meta once and cache them
		assert len(self._data_meta) == 0, len(self._data_meta)
		expected_total = 0
		warning_msg = ""
		
		print(f"[SpecMolFragDynamicBatchSampler] Pre-loading metadata for {len(self.data_source)} samples...")
		
		for dataset_idx in range(len(self.data_source)):
			data = self.data_source[dataset_idx]
			# Cache as compact tuple (num_nodes, num_edges)
			self._data_meta.append((data['frag_pyg'].num_nodes, data['frag_pyg'].num_edges))
			
			# Quick lookup using limited_by index (0=nodes, 1=edges)
			limited_by_idx = 0 if self._limited_by == 'frag_node' else 1
			n = self._data_meta[dataset_idx][limited_by_idx]
			
			if not (n > self._max_num and self._skip_too_big):
				expected_total += 1
			else:
				warning_msg += f"Size of data sample at index {dataset_idx} is larger than {self._max_num} {self._limited_by}s (got {n}). "
				if self._skip_too_big:
					warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR.\n"
				else:
					warning_msg += "Attempting to fit into batch, may cause CUDA OOM ERROR.\n"
		
		if warning_msg:
			print(f"[SpecMolFragDynamicBatchSampler] {warning_msg}")
		print(f"[SpecMolFragDynamicBatchSampler] Metadata loaded. Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big={self._skip_too_big}")

	def set_epoch(self, epoch: int) -> None:
		"""Set epoch number to trigger batch recomputation.
		
		Args:
			epoch (int): Current epoch number
		"""
		self._epoch = epoch
		# Let the underlying sampler manage its own randomness per epoch
		if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
			self.sampler.set_epoch(epoch)
		self._pre_compute_batches()

	def _pre_compute_batches(self):
		"""Pre-compute batches respecting max_num constraint.
		
		Optimizations:
		- Cache limited_by_idx (0 or 1) to avoid string comparison per sample
		- Use direct tuple indexing instead of dictionary
		- Skip empty batches
		"""
		self._batches = []

		if self.sampler is not None:
			indices = th.tensor([idx for idx in self.sampler])
		else:
			indices = th.arange(len(self.data_source), dtype=th.long)

		# limited index to _max_sampling_step
		indices = indices[:self._max_sampling_step]

		# Cache this to avoid string comparison per iteration
		limited_by_idx = 0 if self._limited_by == 'frag_node' else 1

		num_processed = 0
		batch = []
		batch_n = 0
		batch_filled = False

		# Fill batch
		for idx in indices:
			idx_item = idx.item()
			# Direct tuple indexing (faster than dict access)
			n = self._data_meta[idx_item][limited_by_idx]
			
			if n > self._max_num and self._skip_too_big:
				continue 
			# check batch_filled condition
			if batch_n + n > self._max_num:
				# no more budget left, mini-batch filled
				batch_filled = True	
			# check we need return at this point for ga
			if self._return_batch_at > 0 \
				and num_processed > 0 \
				and num_processed % self._return_batch_at == 0:
				# Mini-batch filled
				batch_filled = True
			if batch_filled:
				if batch:  # Only append non-empty batches
					self._batches.append(batch)
				batch_n = 0
				batch = []
				batch_filled = False
			# Add sample to current batch
			batch.append(idx_item)
			num_processed += 1
			batch_n += n
			
		if batch:  # Don't forget final batch
			self._batches.append(batch)
		print(f"[SpecMolFragDynamicBatchSampler] Batch indices computed, Expecting {len(self._batches)} mini-batches for next epoch")

	def __iter__(self) -> Iterator[List[int]]:
		"""Iterate over pre-computed batches.
		
		Yields:
			List[int]: Batch of sample indices
		"""
		for batch in self._batches:
			yield batch

	def __len__(self) -> int:
		"""Return number of batches.
		
		Note: __len__() is expected in calculations involving the length of a DataLoader.
		ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		
		Returns:
			int: Number of batches
		"""
		return len(self._batches)


class GroupSampler(Sampler):
	"""Sample K spectra per group_id to keep experimental groups together.
	
	Use case: When spectra are grouped by experiment/acquisition (group_id),
	this sampler ensures that K random spectra from each group are sampled
	in each epoch.
	"""

	def __init__(self, data_source, sample_k: Optional[int] = None, generator: Optional[th.Generator] = None) -> None:
		"""Initialize group sampler.

		Args:
			data_source: Dataset with group_id in each sample
			sample_k (Optional[int]): Specs to sample per group (default: 3)
			generator (Optional[th.Generator]): PyTorch generator for reproducibility
		"""
		self.data_source = data_source
		self.num_samples = None
		self._data_meta_d = {}
		self.sample_k = 3 if sample_k is None else sample_k
		self.generator = generator
		self._epoch = 0
		self._pre_compute_meta()
		self._pre_compute_batches()

	def _pre_compute_meta(self):
		"""Build group_id -> [dataset_indices] mapping."""
		for dataset_idx in range(len(self.data_source)):
			data = self.data_source[dataset_idx]
			group_id = data['group_id'].item()
			if group_id not in self._data_meta_d:
				self._data_meta_d[group_id] = []
			self._data_meta_d[group_id].append(dataset_idx)
		
		for group_id in self._data_meta_d:
			self._data_meta_d[group_id] = th.tensor(self._data_meta_d[group_id])

	def _pre_compute_batches(self):
		"""Pre-compute sampled indices (one sample per group) with epoch-aware randomization."""
		if self.generator is None:
			generator = th.Generator()
		else:
			generator = self.generator

		sampled_indices = []
		for group_id in self._data_meta_d:
			group_indices = self._data_meta_d[group_id]
			sampled_group_indices = group_indices[th.randperm(min(len(group_indices), self.sample_k), generator=generator)]
			sampled_indices.append(sampled_group_indices)
		sampled_indices = th.cat(sampled_indices)
		self.sampled_indices = th.randperm(len(sampled_indices), generator=generator)
		self.num_samples = len(sampled_indices)

	def set_epoch(self, epoch: int) -> None:
		"""Set epoch number to trigger re-sampling.
		
		Args:
			epoch (int): Current epoch number
		"""
		self._epoch = epoch
		self._pre_compute_batches()

	def __iter__(self) -> Iterator[int]:
		"""Iterate over sampled indices.
		
		Yields:
			int: Sample index
		"""
		for i in range(self.num_samples):
			yield self.sampled_indices[i].item()

	def __len__(self) -> int:
		"""Return number of sampled indices.
		
		Note: __len__() is not strictly required by DataLoader but is expected
		in calculations involving the length of a DataLoader.
		ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
		
		Returns:
			int: Number of sampled indices
		"""
		assert self.num_samples is not None
		return self.num_samples 
	
def get_group_sampler(ds, sampler_type: str, avg_per_group: int, generator: th.Generator) -> WeightedRandomSampler:
	"""Create a WeightedRandomSampler based on group/molecule statistics.
	
	Balances sampling to ensure certain groups/molecules are sampled proportionally.
	
	Args:
		ds: Dataset with get_group_mol_stats() method
		sampler_type (str): Weighting strategy:
			- 'group': Weight by 1/specs_per_group
			- 'mol': Weight by 1/specs_per_mol
			- 'group_mol': Weight by 1/(specs_per_group * group_per_mol)
		avg_per_group (int): Average samples per group to target
		generator (th.Generator): PyTorch generator for reproducibility

	Returns:
		WeightedRandomSampler: Configured sampler, or None if invalid sampler_type

	Example:
		```python
		sampler = get_group_sampler(ds, "group", avg_per_group=3, generator=generator)
		loader = DataLoader(ds, batch_sampler=sampler)
		```
	"""
	group_ids, _, spec_per_group, spec_per_mol, group_per_mol = ds.get_group_mol_stats()

	if sampler_type == "group":
		sample_weights = 1.0 / spec_per_group
	elif sampler_type == "mol":
		sample_weights = 1.0 / spec_per_mol
	elif sampler_type == "group_mol":
		sample_weights = 1.0 / (spec_per_group * group_per_mol)
	else:
		return None
	
	# Get unique groups and their counts
	_, counts = th.unique(group_ids, return_counts=True)
	# For each group, we only want to sample up to the number of available samples
	total_samples = sum(min(count.item(), avg_per_group) for count in counts)
	# Make sure we don't exceed the total dataset size
	num_samples = min(total_samples, len(ds))
 
	sampler = WeightedRandomSampler(
		sample_weights,
		num_samples=num_samples,
		replacement=False,
		generator=generator)
	return sampler

"""
Formula-based hard negative sampler for contrastive spectrum-molecule learning.

Mines spectra with the same molecular formula but different structures/connectivity
as hard negatives for use in triplet, pairwise, or contrastive loss functions.
"""

class FormulaHardNegativeSampler(Sampler):
	"""
	Custom sampler that groups spectra by molecular formula and creates triplet-like
	batches with anchor + hard negatives (same formula, different structure) +
	optional random negatives.
	
	Hard negatives are identified by:
	- Same molecular formula
	- Different InChIKey (different structure/connectivity/stereochemistry)
	
	Args:
		spec_df: Spectrum DataFrame (must have: spec_id, mol_id, group_id or similar)
		mol_df: Molecule DataFrame (must have: mol_id as index, formula, inchikey [optional])
		num_hard_negatives: Number of hard negatives per anchor (default: 2)
		num_random_negatives: Number of random negatives per anchor (default: 1)
		shuffle_groups: Whether to shuffle formula groups each epoch (default: True)
		seed: Random seed for reproducibility (default: 42)
		min_specs_per_formula: Minimum spectra per formula to include (default: 2)
	"""
	
	def __init__(
		self,
		spec_df: pd.DataFrame,
		mol_df: pd.DataFrame,
		num_hard_negatives: int = 2,
		num_random_negatives: int = 1,
		shuffle_groups: bool = True,
		seed: int = 42,
		min_specs_per_formula: int = 2,
		generator: Optional[th.Generator] = None,
	):
		self.spec_df = spec_df.reset_index(drop=True)
		# Preserve mol_df indexing and columns (do NOT drop mol_id)
		self.mol_df = mol_df.copy()
		self.num_hard_negatives = num_hard_negatives
		self.num_random_negatives = num_random_negatives
		self.shuffle_groups = shuffle_groups
		self.seed = seed
		self.min_specs_per_formula = min_specs_per_formula
		self.shuffle_groups = shuffle_groups
		# Prefer torch Generator when provided for deterministic behavior
		self.generator = generator
		# Python-level RNG for deterministic fallback when `generator` is not provided
		if self.generator is None:
			self._py_random = random.Random(self.seed)
		else:
			self._py_random = None
		
		# Build formula -> [spec_ids] index
		self._build_formula_index()
		# Build InChIKey index (for distinguishing isomers)
		self._build_inchikey_index()
		# Create triplet batch indices
		self._create_triplet_batches()
		
	def _build_formula_index(self):
		"""Build mapping from formula to spectrum indices."""
		self.formula_to_spec_ids = defaultdict(list)
		self.spec_id_to_formula = {}
		
		for spec_idx, row in self.spec_df.iterrows():
			mol_id = row["mol_id"]
			spec_id = row.get("spec_id", None)
			
			# Handle both index and value
			if isinstance(self.mol_df.index.name, str) and self.mol_df.index.name == "mol_id":
				if mol_id in self.mol_df.index:
					mol_row = self.mol_df.loc[mol_id]
					formula = mol_row.get("formula", None)
					if formula is not None and pd.notna(formula):
						self.formula_to_spec_ids[formula].append(spec_idx)
						self.spec_id_to_formula[spec_idx] = formula
			else:
				# Try finding mol_id in the dataframe
				mol_mask = self.mol_df["mol_id"] == mol_id if "mol_id" in self.mol_df.columns else pd.Series([False] * len(self.mol_df))
				if mol_mask.any():
					mol_row = self.mol_df[mol_mask].iloc[0]
					formula = mol_row.get("formula", None)
					if formula is not None and pd.notna(formula):
						self.formula_to_spec_ids[formula].append(spec_idx)
						self.spec_id_to_formula[spec_idx] = formula
	
	def _build_inchikey_index(self):
		"""Build mapping from mol_id to InChIKey for isomer detection."""
		self.mol_id_to_inchikey = {}
		
		if "inchikey" in self.mol_df.columns:
			# Handle index-based lookup
			if isinstance(self.mol_df.index.name, str) and self.mol_df.index.name == "mol_id":
				for mol_id in self.mol_df.index:
					inchikey = self.mol_df.loc[mol_id, "inchikey"]
					if pd.notna(inchikey):
						self.mol_id_to_inchikey[mol_id] = inchikey
			else:
				# Handle value-based lookup
				if "mol_id" in self.mol_df.columns:
					for _, row in self.mol_df.iterrows():
						mol_id = row["mol_id"]
						inchikey = row["inchikey"]
						if pd.notna(inchikey):
							self.mol_id_to_inchikey[mol_id] = inchikey
	
	def _get_mol_row(self, mol_id):
		"""Safely get molecule row by mol_id."""
		if isinstance(self.mol_df.index.name, str) and self.mol_df.index.name == "mol_id":
			if mol_id in self.mol_df.index:
				return self.mol_df.loc[mol_id]
		else:
			if "mol_id" in self.mol_df.columns:
				mask = self.mol_df["mol_id"] == mol_id
				if mask.any():
					return self.mol_df[mask].iloc[0]
		return None
	
	def _create_triplet_batches(self):
		"""
		Create anchor + hard negative + random negative triplet batches.
		Only include formulas with multiple spectra (candidates for hard negatives).
		"""
		self.triplet_batches = []
		
		for formula, spec_indices in self.formula_to_spec_ids.items():
			# Skip formulas with too few spectra
			if len(spec_indices) < self.min_specs_per_formula:
				continue
			
			# For each spec with this formula, pick hard negatives (different structure)
			for anchor_idx in spec_indices:
				anchor_mol_id = self.spec_df.loc[anchor_idx, "mol_id"]
				anchor_inchikey = self.mol_id_to_inchikey.get(anchor_mol_id, None)
				
				# Find hard negatives: same formula, different InChIKey/structure
				hard_negative_candidates = [
					idx for idx in spec_indices
					if idx != anchor_idx
					and self._is_hard_negative(idx, anchor_mol_id, anchor_inchikey)
				]
				
				# If not enough hard negatives by structure, use all other specs
				if len(hard_negative_candidates) < self.num_hard_negatives:
					hard_negative_candidates = [
						idx for idx in spec_indices if idx != anchor_idx
					]
				
				# Sample hard negatives
				if len(hard_negative_candidates) > 0:
					num_to_sample = min(self.num_hard_negatives, len(hard_negative_candidates))
					if self.generator is not None:
						perm = th.randperm(len(hard_negative_candidates), generator=self.generator)[:num_to_sample].tolist()
						hard_negatives = [hard_negative_candidates[i] for i in perm]
					else:
						# use local Python RNG if available for deterministic fallback
						if self._py_random is not None:
							hard_negatives = self._py_random.sample(hard_negative_candidates, num_to_sample)
						else:
							hard_negatives = random.sample(hard_negative_candidates, num_to_sample)
					
					# Sample random negatives from entire dataset
					all_other_indices = [
						idx for idx in range(len(self.spec_df)) if idx != anchor_idx
					]
					num_random = min(self.num_random_negatives, len(all_other_indices))
					if self.generator is not None:
						perm = th.randperm(len(all_other_indices), generator=self.generator)[:num_random].tolist()
						random_negatives = [all_other_indices[i] for i in perm]
					else:
						# use local Python RNG if available for deterministic fallback
						if self._py_random is not None:
							random_negatives = self._py_random.sample(all_other_indices, num_random)
						else:
							random_negatives = random.sample(all_other_indices, num_random)
					
					# Create triplet: [anchor, hard_neg_1, ..., random_neg_1, ...]
					batch = [anchor_idx] + hard_negatives + random_negatives
					self.triplet_batches.append(batch)
		
		if self.shuffle_groups:
			if self.generator is not None:
				idx = th.randperm(len(self.triplet_batches), generator=self.generator).tolist()
				self.triplet_batches = [self.triplet_batches[i] for i in idx]
			else:
				if self._py_random is not None:
					self._py_random.shuffle(self.triplet_batches)
				else:
					random.shuffle(self.triplet_batches)
	
	def _is_hard_negative(
		self, candidate_idx: int, anchor_mol_id, anchor_inchikey: Optional[str]
	) -> bool:
		"""Check if candidate is a hard negative (same formula, different structure)."""
		candidate_mol_id = self.spec_df.loc[candidate_idx, "mol_id"]
		candidate_inchikey = self.mol_id_to_inchikey.get(candidate_mol_id, None)
		
		# If InChIKeys available, use them; otherwise assume different if different mol_id
		if anchor_inchikey is not None and candidate_inchikey is not None:
			return anchor_inchikey != candidate_inchikey
		else:
			# Fallback: different molecule = hard negative
			return anchor_mol_id != candidate_mol_id
	
	def __iter__(self):
		"""Yield triplet batch indices."""
		for batch_indices in self.triplet_batches:
			yield batch_indices
	
	def __len__(self):
		"""Number of triplet batches."""
		return len(self.triplet_batches)


class FormulaHardNegativeGroupedSampler(Sampler):
	"""
	Alternative sampler that returns batches organized by formula.
	All spectra with the same formula are grouped together, facilitating
	all-pairs hard negative comparisons within a formula group.

	Useful for pairwise loss functions (e.g., pairwise sparse_cross_entropy_vec).

	Args:
		spec_df: Spectrum DataFrame
		mol_df: Molecule DataFrame
		min_specs_per_formula: Minimum spectra per formula to include (default: 2)
		shuffle: Whether to shuffle groups (default: True)
		generator: Optional[th.Generator] for reproducibility
	"""

	def __init__(
		self,
		spec_df: pd.DataFrame,
		mol_df: pd.DataFrame,
		data_source=None,
		min_specs_per_formula: int = 2,
		shuffle: bool = True,
		seed: int = 42,
		generator: Optional[th.Generator] = None,
	):
		self.data_source = data_source
		self.spec_df = spec_df.reset_index(drop=True)
		# Preserve mol_df indexing and columns (do NOT drop mol_id)
		self.mol_df = mol_df.copy()
		self.min_specs_per_formula = min_specs_per_formula
		self.shuffle = shuffle
		self.generator = generator
		self.seed = seed
		# Use `generator` (torch.Generator) when provided for deterministic sampling.
		# If None, fall back to a local Python `random.Random` instance.
		if self.generator is None:
			self._py_random = random.Random(self.seed)
		else:
			self._py_random = None
		self._epoch = 0

		# Build formula index
		self._build_formula_index()
		# Create batches grouped by formula
		self._create_formula_grouped_batches()

	def set_epoch(self, epoch: int) -> None:
		"""Set epoch to trigger re-grouping.

		Args:
			epoch (int): Current epoch number
		"""
		self._epoch = epoch
		# If a generator is provided, reseed it for deterministic shuffling
		if self.generator is not None:
			base_seed = int(self.generator.initial_seed())
			self.generator.manual_seed(base_seed + epoch)
		self._create_formula_grouped_batches()
	
	def _build_formula_index(self):
		"""Build mapping from formula to indices.

		If a dataset (`data_source`) is provided, build groups using dataset indices
		to ensure alignment with dynamic batching. Otherwise, fall back to using
		`spec_df` row indices.
		"""
		self.formula_to_spec_ids = defaultdict(list)
		
		# Helper to fetch formula from mol_df given a mol_id
		def get_formula(mol_id):
			if isinstance(self.mol_df.index.name, str) and self.mol_df.index.name == "mol_id":
				if mol_id in self.mol_df.index:
					mol_row = self.mol_df.loc[mol_id]
					return mol_row.get("formula", None)
			else:
				if "mol_id" in self.mol_df.columns:
					mask = self.mol_df["mol_id"] == mol_id
					if mask.any():
						mol_row = self.mol_df[mask].iloc[0]
						return mol_row.get("formula", None)
			return None

		if self.data_source is not None:
			# Build groups from dataset indices using dataset.spec_df if available
			spec_df_src = getattr(self.data_source, 'spec_df', None)
			if spec_df_src is None:
				# Fall back to provided spec_df
				spec_df_src = self.spec_df
			for ds_idx in range(len(self.data_source)):
				if ds_idx >= len(spec_df_src):
					break
				mol_id = spec_df_src.iloc[ds_idx]["mol_id"]
				formula = get_formula(mol_id)
				if formula is not None and pd.notna(formula):
					self.formula_to_spec_ids[formula].append(ds_idx)
		else:
			# Fall back to spec_df-based grouping (uses spec_df row indices)
			for spec_idx, row in self.spec_df.iterrows():
				mol_id = row["mol_id"]
				formula = get_formula(mol_id)
				if formula is not None and pd.notna(formula):
					self.formula_to_spec_ids[formula].append(spec_idx)
	
	def _create_formula_grouped_batches(self):
		"""Create batches where all specs in a batch share the same formula."""
		self.formula_batches = []

		for formula, spec_indices in self.formula_to_spec_ids.items():
			# Only include formulas with enough spectra for hard negatives
			if len(spec_indices) >= self.min_specs_per_formula:
				self.formula_batches.append(spec_indices)

		if self.shuffle:
			if self.generator is not None:
				idx = th.randperm(len(self.formula_batches), generator=self.generator)
				self.formula_batches = [self.formula_batches[i] for i in idx.tolist()]
			else:
				if self._py_random is not None:
					self._py_random.shuffle(self.formula_batches)
				else:
					random.shuffle(self.formula_batches)
	
	def __iter__(self):
		"""Yield formula-grouped batch indices."""
		for batch_indices in self.formula_batches:
			# Optionally shuffle within each batch
			batch = list(batch_indices)
			if self.shuffle:
				if self.generator is not None:
					perm = th.randperm(len(batch), generator=self.generator).tolist()
					batch = [batch[i] for i in perm]
				else:
					if self._py_random is not None:
						self._py_random.shuffle(batch)
					else:
						random.shuffle(batch)
			yield batch
	
	def __len__(self):
		"""Number of formula groups."""
		return len(self.formula_batches)


class FormulaHardNegativeCollator:
	"""
	Collator for FormulaHardNegativeSampler that organizes batch data
	with explicit anchor/hard-negative/random-negative structure.
	
	Assumes dataset returns dicts with various keys (e.g., spec_peaks, mol_graph, etc.).
	
	Returns a dict with keys organized by role:
	- anchor_{key}: anchor data
	- hard_neg_{i}_{key}: hard negative i data
	- random_neg_{i}_{key}: random negative i data
	- roles: [0 for anchor, 1 for hard negatives, 2 for random negatives]
	- batch_indices: original indices from sampler
	"""
	
	def __init__(self, dataset, num_hard_negatives: int, num_random_negatives: int):
		self.dataset = dataset
		self.num_hard_negatives = num_hard_negatives
		self.num_random_negatives = num_random_negatives
	
	def __call__(self, batch_indices: List[int]) -> Dict:
		"""
		Collate a triplet batch into organized structures.
		
		Args:
			batch_indices: [anchor_idx, hard_neg_1, ..., random_neg_1, ...]
		
		Returns:
			Organized dict with anchor/hard-neg/random-neg data
		"""
		# Get individual items
		items = [self.dataset[idx] for idx in batch_indices]
		
		# Determine number of hard and random negatives actually in batch
		num_hard = min(self.num_hard_negatives, len(batch_indices) - 1)
		num_random = len(batch_indices) - 1 - num_hard
		
		# Organize into roles
		anchor_item = items[0]
		hard_neg_items = items[1 : 1 + num_hard]
		random_neg_items = items[1 + num_hard :]
		
		# Create collated dict
		collated = {"batch_indices": batch_indices}
		collated["roles"] = (
			[0]  # anchor
			+ [1] * len(hard_neg_items)  # hard negatives
			+ [2] * len(random_neg_items)  # random negatives
		)
		collated["num_hard_negatives"] = len(hard_neg_items)
		collated["num_random_negatives"] = len(random_neg_items)
		
		# Collate anchor
		for key, value in anchor_item.items():
			collated[f"anchor_{key}"] = value
		
		# Collate hard negatives
		for i, item in enumerate(hard_neg_items):
			for key, value in item.items():
				collated[f"hard_neg_{i}_{key}"] = value
		
		# Collate random negatives
		for i, item in enumerate(random_neg_items):
			for key, value in item.items():
				collated[f"random_neg_{i}_{key}"] = value
		
		return collated


def create_formula_hard_negative_loader(
	dataset,
	spec_df: pd.DataFrame,
	mol_df: pd.DataFrame,
	batch_size: int = 16,
	num_hard_negatives: int = 2,
	num_random_negatives: int = 1,
	num_workers: int = 0,
	shuffle_epoch: bool = True,
	seed: int = 42,
	**sampler_kwargs,
):
	"""
	Convenience function to create a DataLoader with formula-based hard negative sampling.
	
	Args:
		dataset: PyTorch Dataset
		spec_df: Spectrum DataFrame
		mol_df: Molecule DataFrame
		batch_size: Triplets per batch (each triplet is anchor + negatives)
		num_hard_negatives: Hard negatives per triplet
		num_random_negatives: Random negatives per triplet
		num_workers: DataLoader workers
		shuffle_epoch: Reshuffle groups each epoch
		seed: Random seed
		**sampler_kwargs: Additional args for FormulaHardNegativeSampler
	
	Returns:
		torch.utils.data.DataLoader
	"""
	
	sampler = FormulaHardNegativeSampler(
		spec_df=spec_df,
		mol_df=mol_df,
		num_hard_negatives=num_hard_negatives,
		num_random_negatives=num_random_negatives,
		shuffle_groups=shuffle_epoch,
		seed=seed,
		**sampler_kwargs,
	)
	
	collator = FormulaHardNegativeCollator(
		dataset=dataset,
		num_hard_negatives=num_hard_negatives,
		num_random_negatives=num_random_negatives,
	)
	
	loader = DataLoader(
		dataset=dataset,
		batch_sampler=sampler,
		collate_fn=collator,
		num_workers=num_workers,
	)
	
	return loader


def create_formula_grouped_loader(
	dataset,
	spec_df: pd.DataFrame,
	mol_df: pd.DataFrame,
	batch_size: int = 16,
	num_workers: int = 0,
	shuffle_epoch: bool = True,
	seed: int = 42,
	min_specs_per_formula: int = 2,
):
	"""
	Convenience function to create a DataLoader with formula-grouped sampling.
	
	Useful for all-pairs hard negative comparisons (pairwise loss).
	
	Args:
		dataset: PyTorch Dataset
		spec_df: Spectrum DataFrame
		mol_df: Molecule DataFrame
		batch_size: Specs per batch
		num_workers: DataLoader workers
		shuffle_epoch: Reshuffle groups each epoch
		seed: Random seed
		min_specs_per_formula: Minimum specs per formula group
	
	Returns:
		torch.utils.data.DataLoader
	"""
	
	sampler = FormulaHardNegativeGroupedSampler(
		data_source=dataset,
		spec_df=spec_df,
		mol_df=mol_df,
		min_specs_per_formula=min_specs_per_formula,
		shuffle=shuffle_epoch,
		seed=seed,
	)
	
	loader = DataLoader(
		dataset=dataset,
		batch_sampler=sampler,
		num_workers=num_workers,
		collate_fn=None,  # Use default collate_fn
	)
	
	return loader


def compute_sampler_stats(spec_df: pd.DataFrame, mol_df: pd.DataFrame) -> Dict:
	"""
	Compute statistics about hard negative potential in the dataset.
	
	Returns dict with:
	- num_formulas: Number of unique formulas
	- num_formulas_with_multiple: Formulas with 2+ spectra
	- mean_specs_per_formula: Average spectra per formula
	- max_specs_per_formula: Max spectra in one formula
	- estimated_triplets: Approximate number of hard negative triplets
	"""
	formula_to_spec_ids = defaultdict(list)
	
	for _, row in spec_df.iterrows():
		mol_id = row["mol_id"]
		
		if isinstance(mol_df.index.name, str) and mol_df.index.name == "mol_id":
			if mol_id in mol_df.index:
				formula = mol_df.loc[mol_id, "formula"]
				if pd.notna(formula):
					formula_to_spec_ids[formula].append(mol_id)
		else:
			if "mol_id" in mol_df.columns and "formula" in mol_df.columns:
				mask = mol_df["mol_id"] == mol_id
				if mask.any():
					formula = mol_df[mask].iloc[0]["formula"]
					if pd.notna(formula):
						formula_to_spec_ids[formula].append(mol_id)
	
	num_formulas = len(formula_to_spec_ids)
	formula_counts = [len(v) for v in formula_to_spec_ids.values()]
	formulas_with_multiple = sum(1 for c in formula_counts if c >= 2)
	mean_specs = np.mean(formula_counts) if formula_counts else 0
	max_specs = max(formula_counts) if formula_counts else 0
	
	# Rough estimate: each spec with formula can be anchor
	# Expected triplets: (specs_per_formula - 1) * num_formulas_with_multiple
	estimated_triplets = sum(max(0, count - 1) for count in formula_counts)
	
	return {
		"num_formulas": num_formulas,
		"num_formulas_with_multiple_specs": formulas_with_multiple,
		"mean_specs_per_formula": mean_specs,
		"max_specs_per_formula": max_specs,
		"estimated_hard_negative_triplets": estimated_triplets,
	}


class FormulaHardNegativeDynamicBatchSampler(BatchSampler):
	"""
	Combined sampler: Formula-grouped hard negatives + Dynamic batching.
	
	Two-layer architecture:
		Layer 1: FormulaHardNegativeGroupedSampler - Groups specs by formula
		Layer 2: Dynamic batching - Respects frag_node/frag_edge limits
	
	This ensures:
	- All specs in a batch share the same formula (hard negatives)
	- No batch exceeds max_num constraint (OOM prevention)
	- Formula groups are split across batches if needed
	
	Args:
		data_source: Dataset with frag_pyg data
		spec_df: Spectrum DataFrame
		mol_df: Molecule DataFrame
		max_num (int): Maximum number of nodes or edges per batch
		limited_by (str): 'frag_node' or 'frag_edge' (default: 'frag_edge')
		skip_too_big (bool): Skip samples larger than max_num (default: False)
		min_specs_per_formula (int): Minimum spectra per formula to include (default: 2)
		shuffle (bool): Shuffle formula groups (default: True)
		seed (int): Random seed (default: 42)
	
	Example:
		```python
		sampler = FormulaHardNegativeDynamicBatchSampler(
			data_source=train_ds,
			spec_df=spec_df,
			mol_df=mol_df,
			max_num=50000,
			limited_by='frag_edge',
			min_specs_per_formula=2
		)
		loader = DataLoader(train_ds, batch_sampler=sampler)
		```
	"""
	
	def __init__(
		self,
		data_source,
		spec_df: pd.DataFrame,
		mol_df: pd.DataFrame,
		max_num: int,
		limited_by: str = 'frag_edge',
		skip_too_big: bool = False,
		min_specs_per_formula: int = 2,
		shuffle: bool = True,
		generator: Optional[th.Generator] = None,
	):
		if not isinstance(max_num, int) or max_num <= 0:
			raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
		if limited_by not in ['frag_node', 'frag_edge']:
			raise ValueError(f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}').")
		
		self.data_source = data_source
		self.spec_df = spec_df
		self.mol_df = mol_df
		self._max_num = max_num
		self._limited_by = limited_by
		self._skip_too_big = skip_too_big
		self.min_specs_per_formula = min_specs_per_formula
		self.shuffle = shuffle
		self.generator = generator
		
		self._batches = []
		self._data_meta = []
		self._epoch = 0
		
		# Layer 1: Create formula-grouped sampler
		self.formula_sampler = FormulaHardNegativeGroupedSampler(
			data_source=data_source,
			spec_df=spec_df,
			mol_df=mol_df,
			min_specs_per_formula=min_specs_per_formula,
			shuffle=shuffle,
			generator=generator,
		)
		
		# Pre-load metadata
		self._pre_load_metadata()
		# Layer 2: Create dynamic batches from formula groups
		self._pre_compute_dynamic_batches()

	def set_epoch(self, epoch: int) -> None:
		"""Set epoch to trigger re-grouping and re-batching.
		
		Args:
			epoch (int): Current epoch number
		"""
		self._epoch = epoch
		self.formula_sampler.set_epoch(epoch)
		self._pre_compute_dynamic_batches()
	
	def _pre_load_metadata(self):
		"""Pre-load and cache frag graph metadata (nodes, edges) for each sample."""
		assert len(self._data_meta) == 0, len(self._data_meta)
		expected_total = 0
		warning_msg = ""
		
		for dataset_idx in tqdm(range(len(self.data_source)), desc="FormulaHardNegativeDynamicBatchSampler:pre_load_metadata"):
			data = self.data_source[dataset_idx]
			self._data_meta.append((data['frag_pyg'].num_nodes, data['frag_pyg'].num_edges))
			n = self._data_meta[dataset_idx][0] if self._limited_by == 'frag_node' else self._data_meta[dataset_idx][1]
			if not (n > self._max_num and self._skip_too_big):
				expected_total += 1
			else:
				warning_msg += f"Size of data sample at index {dataset_idx} is larger than " +\
					f"{self._max_num} {self._limited_by}s (Got {n} {self._limited_by}s). " +\
					"This sample can not fit into batch. "
				if self._skip_too_big:
					warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR\n"
				else:
					warning_msg += "Attempting to fit this in batch, this may cause CUDA OOM ERROR\n"
		
		if warning_msg != "":
			print("[FormulaHardNegativeDynamicBatchSampler]", warning_msg)
		print(f"[FormulaHardNegativeDynamicBatchSampler] Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big:{self._skip_too_big}")
	
	def _pre_compute_dynamic_batches(self):
		"""
		Pre-compute batches by:
		1. Iterating through formula groups from Layer 1
		2. For each group, create dynamic batches respecting max_num constraint
		3. Split large formula groups across multiple batches if needed
		"""
		# Clear previous batches for fresh recomputation
		self._batches = []
		total_formula_groups = 0
		total_batches = 0
		
		# Iterate through formula groups from Layer 1
		for formula_group_indices in self.formula_sampler:
			total_formula_groups += 1
			
			# Create dynamic batches for this formula group
			batch = []
			batch_n = 0
			
			for idx in formula_group_indices:
				# Size of sample
				n = self._data_meta[idx][0] if self._limited_by == 'frag_node' else self._data_meta[idx][1]
				
				if n > self._max_num and self._skip_too_big:
					continue
				
				# Check if adding this sample would exceed max_num
				if batch_n + n > self._max_num and len(batch) > 0:
					# Batch filled, yield current batch
					self._batches.append(batch)
					total_batches += 1
					batch = []
					batch_n = 0
				
				# Add sample to current batch
				batch.append(idx)
				batch_n += n
			
			# Add remaining samples in formula group
			if len(batch) > 0:
				self._batches.append(batch)
				total_batches += 1
		
		print(f"[FormulaHardNegativeDynamicBatchSampler] Created {total_batches} dynamic batches from {total_formula_groups} formula groups")
		
		# Report statistics
		batch_sizes = [len(b) for b in self._batches]
		if batch_sizes:
			print(f"[FormulaHardNegativeDynamicBatchSampler] Batch size stats: min={min(batch_sizes)}, max={max(batch_sizes)}, mean={sum(batch_sizes)/len(batch_sizes):.2f}")
	
	def __iter__(self) -> Iterator[List[int]]:
		"""Iterate over pre-computed dynamic batches.
		
		Yields:
			List[int]: Batch of sample indices (all with same formula when possible)
		"""
		for batch in self._batches:
			yield batch
	
	def __len__(self) -> int:
		"""Return number of batches.
		
		Returns:
			int: Number of dynamic batches
		"""
		return len(self._batches)


def create_formula_hard_negative_dynamic_loader(
	dataset,
	spec_df: pd.DataFrame,
	mol_df: pd.DataFrame,
	max_num: int = 50000,
	limited_by: str = 'frag_edge',
	batch_size: int = 16,
	skip_too_big: bool = False,
	min_specs_per_formula: int = 2,
	shuffle: bool = True,
	generator: Optional[th.Generator] = None,
	num_workers: int = 0,
):
	"""
	Create a DataLoader with combined formula-based hard negatives + dynamic batching.
	
	This is the recommended way to use FormulaHardNegativeDynamicBatchSampler.
	
	Args:
		dataset: PyTorch Dataset (must have frag_pyg data)
		spec_df: Spectrum DataFrame
		mol_df: Molecule DataFrame
		max_num: Maximum nodes/edges per batch (default: 50000)
		limited_by: 'frag_node' or 'frag_edge' (default: 'frag_edge')
		batch_size: Ignored (batch size determined dynamically)
		skip_too_big: Skip samples larger than max_num (default: False)
		min_specs_per_formula: Minimum specs per formula (default: 2)
		shuffle: Shuffle formula groups (default: True)
		seed: Random seed (default: 42)
		num_workers: DataLoader workers (default: 0)
	
	Returns:
		DataLoader with combined sampler
	
	Example:
		```python
		train_loader = create_formula_hard_negative_dynamic_loader(
			dataset=train_ds,
			spec_df=spec_df,
			mol_df=mol_df,
			max_num=50000,
			limited_by='frag_edge',
			min_specs_per_formula=2
		)
		
		for batch in train_loader:
			# All specs in batch have same formula (hard negatives)
			# Batch respects frag_edge <= 50000 constraint
			loss = compute_pairwise_loss(batch)
			loss.backward()
		```
	"""
	sampler = FormulaHardNegativeDynamicBatchSampler(
		data_source=dataset,
		spec_df=spec_df,
		mol_df=mol_df,
		max_num=max_num,
		limited_by=limited_by,
		skip_too_big=skip_too_big,
		min_specs_per_formula=min_specs_per_formula,
		shuffle=shuffle,
		generator=generator,
	)
	
	loader = DataLoader(
		dataset=dataset,
		batch_sampler=sampler,
		num_workers=num_workers,
		collate_fn=dataset.get_collate_fn() if hasattr(dataset, 'get_collate_fn') else None,
	)
	
	return loader
