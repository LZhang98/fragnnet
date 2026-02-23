"""
Data samplers for spectrum-molecule-fragment datasets.

Includes:
- SpecMolFragDynamicBatchSampler: Dynamic batching to avoid OOM
- GroupSampler: Sample K specs per group_id
- get_group_sampler: WeightedRandomSampler factory
"""

import logging
from collections.abc import Iterator

import torch as th
from torch.utils.data import BatchSampler, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)

#####################################################
# Batch Samplers
#####################################################


class SpecMolFragDynamicBatchSampler(BatchSampler):
    """Dynamically adds samples to a mini-batch up to a maximum size either based on number of nodes on frag DAG or number of edges on frag DAG.
    This is used to avoid CUDA OOM errors, implmentation is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
    This should have the same random sampling behavior as RandomSampler
    """

    def __init__(
        self,
        data_source,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples: int | None = None,
        return_batch_at: int = 0,
        sampler: Sampler | None = None,
    ) -> None:
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
        if limited_by not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}')."
            )

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
        if self.sampler is not None and hasattr(self.sampler, "generator"):
            gen = self.sampler.generator
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

        logger.info(
            f"[SpecMolFragDynamicBatchSampler] Pre-loading metadata for {len(self.data_source)} samples..."
        )

        for dataset_idx in range(len(self.data_source)):
            if hasattr(self.data_source, "precom_meta_info"):
                self._data_meta.append(self.data_source.precom_meta_info[dataset_idx])
            else:
                data = self.data_source[dataset_idx]
                # Cache as compact tuple (num_nodes, num_edges)
                self._data_meta.append((data["frag_pyg"].num_nodes, data["frag_pyg"].num_edges))

            # Quick lookup using limited_by index (0=nodes, 1=edges)
            limited_by_idx = 0 if self._limited_by == "frag_node" else 1
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
            logger.warning(f"[SpecMolFragDynamicBatchSampler] {warning_msg}")
        logger.info(
            f"[SpecMolFragDynamicBatchSampler] Metadata loaded. Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big={self._skip_too_big}"
        )

    def set_epoch(self, epoch: int) -> None:
        """Set epoch number to trigger batch recomputation.

        Args:
            epoch (int): Current epoch number
        """
        self._epoch = epoch
        # If underlying sampler supports set_epoch, delegate to it
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
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
        indices = indices[: self._max_sampling_step]

        # Cache this to avoid string comparison per iteration
        limited_by_idx = 0 if self._limited_by == "frag_node" else 1

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
            if (
                self._return_batch_at > 0
                and num_processed > 0
                and num_processed % self._return_batch_at == 0
            ):
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
        logger.info(
            f"[SpecMolFragDynamicBatchSampler] Batch indices computed, Expecting {len(self._batches)} mini-batches for next epoch"
        )

    def __iter__(self) -> Iterator[list[int]]:
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

    def __init__(
        self, data_source, sample_k: int | None = None, generator: th.Generator | None = None
    ) -> None:
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
            group_id = data["group_id"].item()
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
            # shuffle group indices and take up to sample_k
            if len(group_indices) == 0:
                continue
            take = min(len(group_indices), self.sample_k)
            if len(group_indices) == 1:
                sampled_group_indices = group_indices[:take]
            else:
                perm = th.randperm(len(group_indices), generator=generator)
                sampled_group_indices = group_indices[perm[:take]]
            sampled_indices.append(sampled_group_indices)

        if sampled_indices:
            sampled_indices = th.cat(sampled_indices)
            # shuffle across groups for variety: permute actual dataset indices
            if len(sampled_indices) > 1:
                perm_all = th.randperm(len(sampled_indices), generator=generator)
                self.sampled_indices = sampled_indices[perm_all]
            else:
                self.sampled_indices = sampled_indices
            self.num_samples = len(sampled_indices)
        else:
            self.sampled_indices = th.tensor([], dtype=th.long)
            self.num_samples = 0

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


class FormulaPrecTypeGroupSampler(Sampler):
    """Sample indices grouped by composite key `prec_type::formula`.

    This sampler yields up to `sample_k` indices per formula+prec_type
    cluster each epoch. It is compatible with `SpecMolFragDynamicBatchSampler`
    via the `sampler` argument and exposes an optional `generator` for
    deterministic behavior.

    Args:
        data_source: dataset providing __len__ and __getitem__ returning dicts
            with keys for formula and adduct (defaults: 'formula', 'adduct').
        sample_k (int): max samples to take from each composite group.
        key_formula (str): key name to read formula from sample dict.
        key_adduct (str): key name to read adduct from sample dict.
        generator (Optional[th.Generator]): optional torch Generator for determinism.
        num_samples (Optional[int]): limit total number of sampled indices.
    """

    def __init__(
        self,
        data_source,
        sample_k: int = 1,
        key_formula: str = "formula",
        key_adduct: str = "prec_type",
        generator: th.Generator | None = None,
        num_samples: int | None = None,
    ) -> None:
        self.data_source = data_source
        self.sample_k = int(sample_k)
        self.key_formula = key_formula
        self.key_adduct = key_adduct
        self.generator = generator
        self._epoch = 0
        if num_samples is None:
            num_samples = len(data_source)
        self.num_samples = int(num_samples)

        # capture base seed if generator provided
        self._base_seed = None
        if self.generator is not None and hasattr(self.generator, "initial_seed"):
            try:
                self._base_seed = int(self.generator.initial_seed())
            except Exception:
                self._base_seed = 0

        # group -> tensor(indices)
        self._groups = {}
        self._pre_compute_meta()
        self._pre_compute_indices()

    def _pre_compute_meta(self):
        """Build composite key -> list(indices) mapping.
        Raises KeyError if required keys are missing.
        """
        for dataset_idx in range(len(self.data_source)):
            data = self.data_source[dataset_idx]
            if self.key_formula not in data or self.key_adduct not in data:
                raise KeyError(
                    f"Sample {dataset_idx} missing '{self.key_formula}' or '{self.key_adduct}' key."
                )
            # normalize to str for stable keys
            formula = str(data[self.key_formula])
            adduct = str(data[self.key_adduct])
            key = f"{adduct}::{formula}"
            if key not in self._groups:
                self._groups[key] = []
            self._groups[key].append(dataset_idx)

        # convert to tensors
        for k in list(self._groups.keys()):
            self._groups[k] = th.tensor(self._groups[k], dtype=th.long)

    def _pre_compute_indices(self):
        """Compute epoch-local sampled indices based on `sample_k` and `generator`.
        Stores results in `self.sampled_indices` and updates `self.num_samples`.
        """
        if self.generator is None:
            gen = th.Generator()
        else:
            gen = self.generator

        sampled_parts = []
        for k in self._groups:
            indices = self._groups[k]
            if len(indices) == 0:
                continue
            take = min(len(indices), self.sample_k)
            if len(indices) == 1:
                part = indices[:take]
            else:
                perm = th.randperm(len(indices), generator=gen)
                part = indices[perm][:take]
            sampled_parts.append(part)

        if sampled_parts:
            self.sampled_indices = th.cat(sampled_parts)
            # shuffle across groups for variety
            if len(self.sampled_indices) > 1:
                self.sampled_indices = self.sampled_indices[
                    th.randperm(len(self.sampled_indices), generator=gen)
                ]
            self.num_samples = len(self.sampled_indices)
        else:
            self.sampled_indices = th.tensor([], dtype=th.long)
            self.num_samples = 0

    def set_epoch(self, epoch: int) -> None:
        """Set epoch, reseed generator deterministically and recompute indices."""
        self._epoch = int(epoch)
        if self.generator is not None and hasattr(self.generator, "manual_seed"):
            try:
                base = self._base_seed if self._base_seed is not None else 0
                self.generator.manual_seed(base + int(epoch))
            except Exception:
                pass
        self._pre_compute_indices()

    def __iter__(self) -> Iterator[int]:
        for i in range(self.num_samples):
            yield int(self.sampled_indices[i].item())

    def __len__(self) -> int:
        return self.num_samples


class DualGroupDynamicBatchSampler(BatchSampler):
    """Build mini-batches that include samples grouped by two keys:
    1) adduct + mol_id
    2) adduct + formula

    Each produced batch will try to include at least one sample of each
    type (when available) and will respect maximum fragment node/edge
    constraints (like SpecMolFragDynamicBatchSampler).
    """

    def __init__(
        self,
        data_source,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples: int | None = None,
        return_batch_at: int = 0,
        sample_k1: int = 1,
        sample_k2: int = 1,
        generator: th.Generator | None = None,
    ) -> None:
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
        if limited_by not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}')."
            )

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
            n_nodes = data["frag_pyg"].num_nodes
            n_edges = data["frag_pyg"].num_edges
            self._data_meta.append((n_nodes, n_edges))

            # Validate required keys (each item must have both key pairs: adduct+mol_id and adduct+formula)
            for req_key in ("adduct", "mol_id", "formula"):
                if req_key not in data:
                    raise KeyError(
                        f"Sample at index {dataset_idx} missing required key '{req_key}'."
                    )
                if data[req_key] is None:
                    raise ValueError(
                        f"Sample at index {dataset_idx} has None value for required key '{req_key}'."
                    )

            # Build group keys (keys are required to be present)
            adduct = data["adduct"]
            mol_id = data["mol_id"]
            formula = data["formula"]

            # Add to group1 (adduct+mol_id) — mol_id is required to be present
            key1 = f"{adduct}::{mol_id}"
            self._group1.setdefault(key1, []).append(dataset_idx)

            key2 = f"{adduct}::{formula}"
            self._group2.setdefault(key2, []).append(dataset_idx)

            # Drawing from the same pool: avoid building formula-based group2 to prevent overlap
            # (samples remain uniquely identified by adduct+mol_id)

        # Remove overlapping samples: keep items in group1; drop them from group2
        group1_indices = set()
        for indices in self._group1.values():
            group1_indices.update(indices)
        # Prune group2 entries that overlap with group1
        for k in list(self._group2.keys()):
            indices = self._group2[k]
            non_overlap = [idx for idx in indices if idx not in group1_indices]
            if non_overlap:
                self._group2[k] = non_overlap
            else:
                # Remove empty groups
                del self._group2[k]

        # convert lists to tensors for faster ops
        for k in list(self._group1.keys()):
            self._group1[k] = th.tensor(self._group1[k], dtype=th.long)
        for k in list(self._group2.keys()):
            self._group2[k] = th.tensor(self._group2[k], dtype=th.long)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        # reseed generator if provided
        if self.generator is not None and hasattr(self.generator, "manual_seed"):
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
            return taken, used_mask, budget_left
        keys = list(group_dict.keys())
        # randomize key order using PyTorch Generator for consistency
        if self.generator is not None:
            perm = th.randperm(len(keys), generator=self.generator)
            keys = [keys[i] for i in perm.tolist()]
        # If no generator, keys remain in original order (deterministic)

        limited_by_idx = 0 if self._limited_by == "frag_node" else 1
        for key in keys:
            if take_k <= 0:
                break
            candidates = group_dict[key]
            # Try to take one sample from this group
            for idx in candidates.tolist():
                if used_mask.get(idx, False):
                    continue
                # check size against remaining budget
                n = self._data_meta[idx][limited_by_idx]
                if n > budget_left and self._skip_too_big:
                    continue
                if n > budget_left:
                    # Won't fit in budget
                    continue
                # take it
                used_mask[idx] = True
                taken.append(idx)
                budget_left -= n
                take_k -= 1
                break
        return taken, used_mask, budget_left

    def _pre_compute_batches(self):
        self._batches = []

        # Flatten available indices and limit to max_sampling_step
        all_indices = list(range(len(self.data_source)))[: self._max_sampling_step]
        available = set(all_indices)
        used_mask = {}

        # Cache limited_by index for budget checks
        limited_by_idx = 0 if self._limited_by == "frag_node" else 1

        # Track which group keys still have available samples
        active_group1_keys = set(self._group1.keys())
        active_group2_keys = set(self._group2.keys())

        while available:
            batch = []
            batch_n = 0
            budget_left = self._max_num

            # Initial sample budget for this batch
            if self._return_batch_at > 0:
                samples_to_take = self._return_batch_at
            else:
                # Effectively unlimited if return_batch_at is 0
                samples_to_take = len(available)

            # First try to take sample_k1 samples from group1
            if active_group1_keys and samples_to_take > 0:
                k1 = min(self.sample_k1, samples_to_take)
                group1_active = {k: self._group1[k] for k in active_group1_keys}
                taken1, used_mask, budget_left = self._sample_from_group_dict(
                    group1_active, used_mask, budget_left, k1
                )
                batch.extend(taken1)
                samples_to_take -= len(taken1)
                # Update batch_n from budget_left
                batch_n = self._max_num - budget_left
                # Remove exhausted keys
                for key in list(active_group1_keys):
                    if all(used_mask.get(idx, False) for idx in self._group1[key].tolist()):
                        active_group1_keys.discard(key)

            # Then try to take sample_k2 samples from group2
            if active_group2_keys and samples_to_take > 0:
                k2 = min(self.sample_k2, samples_to_take)
                group2_active = {k: self._group2[k] for k in active_group2_keys}
                taken2, used_mask, budget_left = self._sample_from_group_dict(
                    group2_active, used_mask, budget_left, k2
                )
                batch.extend(taken2)
                samples_to_take -= len(taken2)
                batch_n = self._max_num - budget_left
                # Remove exhausted keys
                for key in list(active_group2_keys):
                    if all(used_mask.get(idx, False) for idx in self._group2[key].tolist()):
                        active_group2_keys.discard(key)

            # Fill remaining budget with any available samples
            if budget_left > 0 and samples_to_take > 0:
                cand_list = list(available - set(k for k, v in used_mask.items() if v))
                if cand_list:
                    # Use PyTorch Generator for consistency
                    if self.generator is not None:
                        perm = th.randperm(len(cand_list), generator=self.generator)
                        cand_list = [cand_list[i] for i in perm.tolist()]
                    # If no generator, keep original order

                    for idx in cand_list:
                        if used_mask.get(idx, False):
                            continue
                        if samples_to_take <= 0:
                            break
                        n = self._data_meta[idx][limited_by_idx]
                        if n > self._max_num and self._skip_too_big:
                            continue
                        # Check budget BEFORE adding
                        if n > budget_left:
                            continue
                        batch.append(idx)
                        batch_n += n
                        samples_to_take -= 1
                        used_mask[idx] = True
                        budget_left -= n

            if not batch:
                # Nothing could be placed (maybe all remaining too large or already used)
                break

            # Mark used and remove from available
            for idx in batch:
                available.discard(idx)

            self._batches.append(batch)

        logger.info(
            f"[DualGroupDynamicBatchSampler] Prepared {len(self._batches)} batches respecting max {self._limited_by}={self._max_num}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        for b in self._batches:
            yield b

    def __len__(self) -> int:
        return len(self._batches)


def get_group_sampler(
    ds, sampler_type: str, avg_per_group: int, generator: th.Generator
) -> WeightedRandomSampler | None:
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
        WeightedRandomSampler | None: Configured sampler, or None if invalid sampler_type

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
        sample_weights, num_samples=num_samples, replacement=False, generator=generator
    )
    return sampler
