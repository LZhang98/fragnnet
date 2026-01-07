
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
		# If underlying sampler exposes a torch.Generator, capture its initial seed
		# so we can reseed it in set_epoch() for deterministic epoch-aware behavior.
		self._base_sampler_generator = None
		self._base_sampler_seed = None
		if self.sampler is not None and hasattr(self.sampler, 'generator'):
			gen = getattr(self.sampler, 'generator')
			if isinstance(gen, th.Generator):
				self._base_sampler_generator = gen
				try:
					self._base_sampler_seed = int(self._base_sampler_generator.initial_seed())
				except Exception:
					self._base_sampler_seed = 0
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
		# If underlying sampler supports set_epoch, delegate to it
		if self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
			self.sampler.set_epoch(epoch)
		# If underlying sampler exposes a torch.Generator but lacks set_epoch,
		# reseed it deterministically based on the captured base seed.
		elif self._base_sampler_generator is not None:
			seed = self._base_sampler_seed if self._base_sampler_seed is not None else 0
			self._base_sampler_generator.manual_seed(seed + int(epoch))
		# Recompute batches after epoch change
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

class DualGroupDynamicBatchSampler(BatchSampler):
	"""Build mini-batches that include samples grouped by two keys:
	1) adduct + mol_id
	2) adduct + formula

	Each produced batch will try to include at least one sample of each
	type (when available) and will respect maximum fragment node/edge
	constraints (like SpecMolFragDynamicBatchSampler).
	"""

	def __init__(self, data_source, max_num: int, limited_by: str = 'frag_edge',
			 skip_too_big: bool = False, num_samples: Optional[int] = None,
			 return_batch_at: int = 0, sample_k1: int = 1, sample_k2: int = 1,
			 generator: Optional[th.Generator] = None) -> None:
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
		self.sample_k1 = int(sample_k1)
		self.sample_k2 = int(sample_k2)
		self.generator = generator
		self._epoch = 0
		self._data_meta = []  # tuples (num_nodes, num_edges)
		self._group1 = {}  # adduct+mol_id -> tensor(indices)
		self._group2 = {}  # adduct+formula -> tensor(indices)
		self._batches = []

		self._pre_load_metadata_and_groups()
		self._pre_compute_batches()

	def _pre_load_metadata_and_groups(self):
		"""Load frag metadata and build group mappings.
		Assumes each dataset sample contains keys: 'adduct', 'mol_id', 'formula'.
		"""
		assert len(self._data_meta) == 0, len(self._data_meta)
		for dataset_idx in range(len(self.data_source)):
			data = self.data_source[dataset_idx]
			# Expecting 'frag_pyg' present like other samplers
			n_nodes = data['frag_pyg'].num_nodes
			n_edges = data['frag_pyg'].num_edges
			self._data_meta.append((n_nodes, n_edges))

			# Build group keys with safe fallbacks
			adduct = data.get('adduct', None)
			mol_id = data.get('mol_id', None)
			formula = data.get('formula', None)

			key1 = None
			key2 = None
			if adduct is not None and mol_id is not None:
				key1 = f"{adduct}::{mol_id}"
			if adduct is not None and formula is not None:
				key2 = f"{adduct}::{formula}"

			if key1 is not None:
				self._group1.setdefault(key1, []).append(dataset_idx)
			if key2 is not None:
				self._group2.setdefault(key2, []).append(dataset_idx)

		# convert lists to tensors for faster ops
		for k in list(self._group1.keys()):
			self._group1[k] = th.tensor(self._group1[k], dtype=th.long)
		for k in list(self._group2.keys()):
			self._group2[k] = th.tensor(self._group2[k], dtype=th.long)

		limited_by_idx = 0 if self._limited_by == 'frag_node' else 1
		# Optional report
		valid_count = 0
		for i, t in enumerate(self._data_meta):
			n = t[limited_by_idx]
			if not (n > self._max_num and self._skip_too_big):
				valid_count += 1
		print(f"[DualGroupDynamicBatchSampler] Loaded metadata for {len(self._data_meta)} samples; {valid_count} usable with skip_too_big={self._skip_too_big}")

	def set_epoch(self, epoch: int) -> None:
		self._epoch = epoch
		# reseed generator if provided
		if self.generator is not None and hasattr(self.generator, 'manual_seed'):
			try:
				self.generator.manual_seed(int(epoch))
			except Exception:
				pass
		self._pre_compute_batches()

	def _sample_from_group_dict(self, group_dict, used_mask, budget_left, take_k):
		"""Try to take up to `take_k` samples from group_dict (which maps key->tensor).
		Returns list of taken indices and updated used_mask.
		"""
		taken = []
		if not group_dict:
			return taken, used_mask
		keys = list(group_dict.keys())
		# randomize key order
		if self.generator is None:
			rand.shuffle(keys)
		else:
			perm = th.randperm(len(keys), generator=self.generator)
			keys = [keys[i] for i in perm.tolist()]
		for key in keys:
			if take_k <= 0:
				break
			candidates = group_dict[key]
			for idx in candidates.tolist():
				if used_mask.get(idx, False):
					continue
				# check size
				n = self._data_meta[idx][0 if self._limited_by == 'frag_node' else 1]
				if n > budget_left and self._skip_too_big:
					continue
				# take it
				used_mask[idx] = True
				taken.append(idx)
				take_k -= 1
				break
		return taken, used_mask

	def _pre_compute_batches(self):
		self._batches = []
		limited_by_idx = 0 if self._limited_by == 'frag_node' else 1

		# Flatten available indices and limit to max_sampling_step
		all_indices = list(range(len(self.data_source)))[: self._max_sampling_step]
		available = set(all_indices)
		used_mask = {}

		# Copy group dicts so we can remove emptied keys
		group1 = {k: v.clone() for k, v in self._group1.items()}
		group2 = {k: v.clone() for k, v in self._group2.items()}

		while available:
			batch = []
			batch_n = 0
			# Ensure at least one sample from each group type when possible
			need1 = any(k for k in group1)
			need2 = any(k for k in group2)
			# First try to take one from group1 and one from group2
			if need1:
				taken1, used_mask = self._sample_from_group_dict(group1, used_mask, self._max_num - batch_n, self.sample_k1)
				for idx in taken1:
					batch.append(idx)
					batch_n += self._data_meta[idx][limited_by_idx]
			if need2:
				taken2, used_mask = self._sample_from_group_dict(group2, used_mask, self._max_num - batch_n, self.sample_k2)
				for idx in taken2:
					if idx not in batch:
						batch.append(idx)
						batch_n += self._data_meta[idx][limited_by_idx]

			# Fill remaining budget with any available samples (preferring not-yet-used group members)
			if batch_n < self._max_num:
				# iterate available indices in random order
				cand_list = list(available - set(k for k,v in used_mask.items() if v))
				if self.generator is None:
					rand.shuffle(cand_list)
				else:
					perm = th.randperm(len(cand_list), generator=self.generator)
					cand_list = [cand_list[i] for i in perm.tolist()]
				for idx in cand_list:
					# if already chosen skip
					if used_mask.get(idx, False):
						continue
					n = self._data_meta[idx][limited_by_idx]
					if n > self._max_num and self._skip_too_big:
						continue
					if batch_n + n > self._max_num:
						continue
					batch.append(idx)
					batch_n += n
					used_mask[idx] = True
					if batch_n >= self._max_num:
						break

			if not batch:
				# nothing could be placed (maybe all remaining too large)
				break

			# mark used and remove from available
			for idx in batch:
				available.discard(idx)

			self._batches.append(batch)

		print(f"[DualGroupDynamicBatchSampler] Prepared {len(self._batches)} batches respecting max {self._limited_by}={self._max_num}")

	def __iter__(self) -> Iterator[List[int]]:
		for b in self._batches:
			yield b

	def __len__(self) -> int:
		return len(self._batches)
	
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

