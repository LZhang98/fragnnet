"""FraGNNet's batch samplers, which budget batches by fragment-DAG nodes or edges.

Moved out of core ``data_sampler`` in the reorganization (docs/plans/repo_reorg_plan.md, Phase 4):
- validate_dag_size_curriculum: check the DAG-size curriculum settings
- SpecMolFragDynamicBatchSampler: dynamic batching to avoid OOM, with the DAG-size curriculum
- CEContextDynamicBatchSampler: full-coverage bounded adjacent-CE batching by DAG edges
- DualGroupDynamicBatchSampler: batches mixing adduct+molecule and adduct+formula groups
"""

import logging
import math
from collections.abc import Iterator

import numpy as np
import torch as th
from torch.utils.data import BatchSampler, Dataset, Sampler
from tqdm import tqdm

from fragnnet.dataset.data_sampler import _CEContextBatchSamplerBase

logger = logging.getLogger(__name__)


def validate_dag_size_curriculum(start_fraction: float, curriculum_epochs: int) -> None:
    """Validate DAG-size curriculum schedule values."""
    if (
        isinstance(start_fraction, bool)
        or not isinstance(start_fraction, (int, float))
        or not 0 < start_fraction <= 1
    ):
        raise ValueError("dag_size_curriculum_start_fraction must be in (0, 1]")
    if (
        isinstance(curriculum_epochs, bool)
        or not isinstance(curriculum_epochs, int)
        or curriculum_epochs <= 0
    ):
        raise ValueError("dag_size_curriculum_epochs must be a positive integer")


class SpecMolFragDynamicBatchSampler(BatchSampler):
    """Dynamically adds samples to a mini-batch up to a maximum size either based on number of nodes on frag DAG or number of edges on frag DAG.
    This is used to avoid CUDA OOM errors, implmentation is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
    This should have the same random sampling behavior as RandomSampler
    """

    def __init__(
        self,
        data_source: Dataset,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples: int | None = None,
        return_batch_at: int = 0,
        sampler: Sampler | None = None,
        dag_size_curriculum: bool = False,
        dag_size_curriculum_start_fraction: float = 0.25,
        dag_size_curriculum_epochs: int = 10,
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
            dag_size_curriculum: Progressively admit samples by fragment DAG node count.
            dag_size_curriculum_start_fraction: Fraction of the smallest DAG samples
                eligible at epoch zero.
            dag_size_curriculum_epochs: Epoch at which all DAG sizes become eligible.

        Raises:
            ValueError: If max_num <= 0 or limited_by not in ['frag_node', 'frag_edge']
        """
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
        if limited_by not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"`limited_by` choice should be either 'frag_node' or 'frag_edge' (got '{limited_by}')."
            )
        if dag_size_curriculum:
            validate_dag_size_curriculum(
                dag_size_curriculum_start_fraction,
                dag_size_curriculum_epochs,
            )

        if num_samples is None:
            num_samples = len(data_source)

        self.data_source = data_source
        self._max_num = max_num
        self._limited_by = limited_by
        self._skip_too_big = skip_too_big
        self._max_sampling_step = num_samples
        self._return_batch_at = return_batch_at
        self._dag_size_curriculum = dag_size_curriculum
        self._dag_size_curriculum_start_fraction = dag_size_curriculum_start_fraction
        self._dag_size_curriculum_epochs = dag_size_curriculum_epochs
        self._batches = []
        self._data_meta = []  # becomes np.ndarray (N, 2) after _pre_load_batches
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

    def _curriculum_node_limit(self) -> int | None:
        """Return the largest eligible DAG node count for the current epoch."""
        if not self._dag_size_curriculum or len(self._data_meta) == 0:
            return None

        progress = min(self._epoch / self._dag_size_curriculum_epochs, 1.0)
        fraction = (
            self._dag_size_curriculum_start_fraction
            + (1.0 - self._dag_size_curriculum_start_fraction) * progress
        )
        node_counts = self._data_meta[:, 0]
        target_count = min(len(node_counts), max(1, math.ceil(len(node_counts) * fraction)))
        return int(np.partition(node_counts, target_count - 1)[target_count - 1])

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

        for dataset_idx in tqdm(range(len(self.data_source))):
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
        # Convert to numpy array for O(1) vectorised column reads in _pre_compute_batches
        self._data_meta = np.asarray(self._data_meta, dtype=np.int32).reshape(-1, 2)

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
        - Vectorised size pre-fetch via numpy to avoid per-element .item() and tuple indexing
        - Use zip(indices_list, sizes_list) to eliminate all per-element Python boxing
        - Skip empty batches
        """
        self._batches = []

        if self.sampler is not None:
            indices = th.as_tensor(list(self.sampler), dtype=th.long)
        else:
            indices = th.arange(len(self.data_source), dtype=th.long)

        # limited index to _max_sampling_step
        indices = indices[: self._max_sampling_step]

        curriculum_node_limit = self._curriculum_node_limit()
        if curriculum_node_limit is not None:
            sampled_count = len(indices)
            eligible = self._data_meta[indices.numpy(), 0] <= curriculum_node_limit
            indices = indices[th.from_numpy(eligible)]
            logger.info(
                "[SpecMolFragDynamicBatchSampler] DAG curriculum epoch %d: "
                "%d/%d sampled items eligible at <= %d fragment nodes",
                self._epoch,
                len(indices),
                sampled_count,
                curriculum_node_limit,
            )

        # Cache this to avoid string comparison per iteration
        limited_by_idx = 0 if self._limited_by == "frag_node" else 1

        # Pre-fetch all sizes in one vectorised numpy read, then convert to plain Python
        # lists so the inner loop pays no per-element boxing or array-indexing overhead.
        indices_np = indices.numpy()
        sizes_list: list[int] = self._data_meta[indices_np, limited_by_idx].tolist()
        indices_list: list[int] = indices_np.tolist()

        num_processed = 0
        batch = []
        batch_n = 0
        batch_filled = False

        # Fill batch — plain zip over pre-computed Python lists, no .item() calls
        for idx_item, n in zip(indices_list, sizes_list):
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
        yield from self._batches

    def __len__(self) -> int:
        """Return number of batches.

        Note: __len__() is expected in calculations involving the length of a DataLoader.
        ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler

        Returns:
            int: Number of batches
        """
        return len(self._batches)


class CEContextDynamicBatchSampler(_CEContextBatchSamplerBase):
    """Pack adjacent-CE spectra using fragment-DAG edge budgets."""

    def __init__(
        self,
        data_source: Dataset,
        max_num: int,
        ce_key_by_inst: dict[str, str],
        limited_by: str = "frag_edge",
        spectra_per_context: int = 5,
        return_batch_at: int = 0,
        seed: int = 0,
        skip_too_big: bool = True,
    ) -> None:
        if limited_by != "frag_edge":
            raise ValueError(
                "CE context batching requires limited_by='frag_edge' " f"(got '{limited_by}')."
            )
        super().__init__(
            data_source=data_source,
            max_num=max_num,
            ce_key_by_inst=ce_key_by_inst,
            spectra_per_context=spectra_per_context,
            return_batch_at=return_batch_at,
            seed=seed,
            skip_too_big=skip_too_big,
        )

    def _load_size_metadata(self) -> np.ndarray:
        """Load per-spectrum fragment-DAG node/edge counts once."""
        metadata = []
        for idx in range(len(self.data_source)):
            if hasattr(self.data_source, "precom_meta_info"):
                metadata.append(self.data_source.precom_meta_info[idx])
            else:
                data = self.data_source[idx]
                metadata.append((data["frag_pyg"].num_nodes, data["frag_pyg"].num_edges))
        return np.asarray(metadata, dtype=np.int64).reshape(-1, 2)


class DualGroupDynamicBatchSampler(BatchSampler):
    """Build mini-batches that include samples grouped by two keys:
    1) adduct + mol_id
    2) adduct + formula

    Each produced batch will try to include at least one sample of each
    type (when available) and will generally respect maximum fragment
    node/edge constraints (like SpecMolFragDynamicBatchSampler).
    When `skip_too_big=False`, an oversized sample may be emitted as a
    singleton batch to avoid dropping data.
    """

    def __init__(
        self,
        data_source: Dataset,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples: int | None = None,
        return_batch_at: int = 0,
        sample_k1: int = 1,
        sample_k2: int = 1,
        sample_k1_per_key: int = 1,
        sample_k2_per_key: int = 1,
        generator: th.Generator | None = None,
        key_adduct: str = "prec_type",
        key_mol_id: str = "mol_id",
        key_formula: str = "formula",
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
        self.sample_k1_per_key = int(sample_k1_per_key)
        self.sample_k2_per_key = int(sample_k2_per_key)
        self.generator = generator
        self._epoch = 0
        self._data_meta = []  # becomes np.ndarray (N, 2) after _pre_load_metadata_and_groups
        self._group1 = {}  # adduct+mol_id -> list(indices)
        self._group2 = {}  # adduct+formula -> list(indices)
        self._idx_to_group1_key: list[str] = []  # per-sample inverse map for O(1) count updates
        self._idx_to_group2_key: list[str] = []  # per-sample inverse map for O(1) count updates
        self._batches = []
        self.key_adduct = key_adduct
        self.key_mol_id = key_mol_id
        self.key_formula = key_formula

        self._pre_load_metadata_and_groups()
        self._pre_compute_batches()

    def _pre_load_metadata_and_groups(self):
        """Load frag metadata and build group mappings.

        Uses precom_meta_info and precom_group_info caches on the dataset when available,
        avoiding a full frag_pyg load for each sample (which would be expensive for
        HDF5-backed datasets). Falls back to data_source[idx] otherwise.

        Also builds per-sample inverse maps (_idx_to_group1_key, _idx_to_group2_key) for
        O(1) available-count updates in _pre_compute_batches, replacing the original
        O(G*S) `any(idx in available ...)` scan that ran over all active keys per batch.
        """
        assert len(self._data_meta) == 0, len(self._data_meta)

        N = len(self.data_source)
        self._idx_to_group1_key = [""] * N
        self._idx_to_group2_key = [""] * N

        has_meta_cache = hasattr(self.data_source, "precom_meta_info")
        has_group_cache = hasattr(self.data_source, "precom_group_info")

        for dataset_idx in range(N):
            # --- node/edge counts ---
            if has_meta_cache:
                n_nodes, n_edges = self.data_source.precom_meta_info[dataset_idx]
            else:
                data = self.data_source[dataset_idx]
                n_nodes = data["frag_pyg"].num_nodes
                n_edges = data["frag_pyg"].num_edges
            self._data_meta.append((n_nodes, n_edges))

            # --- group keys (adduct, mol_id, formula) ---
            if has_group_cache:
                adduct, mol_id, formula = self.data_source.precom_group_info[dataset_idx]
            else:
                if not has_meta_cache:
                    # data already loaded above
                    pass
                else:
                    data = self.data_source[dataset_idx]
                for req_key in (self.key_adduct, self.key_mol_id, self.key_formula):
                    if req_key not in data:
                        raise KeyError(
                            f"Sample at index {dataset_idx} missing required key '{req_key}'."
                        )
                    if data[req_key] is None:
                        raise ValueError(
                            f"Sample at index {dataset_idx} has None value for required key '{req_key}'."
                        )
                adduct = data[self.key_adduct]
                mol_id = data[self.key_mol_id]
                formula = data[self.key_formula]

            key1 = f"{adduct}::{mol_id}"
            self._group1.setdefault(key1, []).append(dataset_idx)
            self._idx_to_group1_key[dataset_idx] = key1

            key2 = f"{adduct}::{formula}"
            self._group2.setdefault(key2, []).append(dataset_idx)
            self._idx_to_group2_key[dataset_idx] = key2

        # Within-batch deduplication is handled by taken_mask in _sample_from_group_dict,
        # so group2 intentionally retains all indices (including those also in group1).
        # This allows isomer contrast: group1 picks a sample first, taken_mask prevents
        # group2 from re-picking the same index in the same batch.

        # Keep groups as plain Python lists — no tensor conversion needed since we
        # access them by iteration, not by fancy indexing.

        # Convert group candidate lists to numpy arrays so callers/tests can
        # call `.tolist()` and iteration yields integer indices usable as
        # numpy array indices. Numpy arrays keep iteration semantics while
        # avoiding PyTorch-tensor indexing issues with numpy masks.
        for k in list(self._group1.keys()):
            self._group1[k] = np.array(self._group1[k], dtype=np.int64)
        for k in list(self._group2.keys()):
            self._group2[k] = np.array(self._group2[k], dtype=np.int64)

        # Convert metadata to numpy array for vectorised column reads
        self._data_meta = np.array(self._data_meta, dtype=np.int32)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        # NOTE: unlike FormulaPrecTypeGroupSampler this reseeds from the epoch alone, so the
        # run seed does not reach it. Changing that would move every existing run's batch
        # order, so it is left as is.
        if self.generator is not None:
            self.generator.manual_seed(int(epoch))
        self._pre_compute_batches()

    def _sample_from_group_dict(
        self,
        group_dict,
        active_keys,
        taken_mask: np.ndarray,
        budget_left: int,
        take_k: int,
        limited_by_idx: int,
        per_key: int = 1,
    ):
        """Try to take up to `take_k` samples from group_dict (which maps key->list).

        Iterates over up to `take_k` group keys and takes up to `per_key` samples from
        each key. Using `per_key > 1` places multiple spectra of the same molecule (or
        same formula) in a single batch, which provides meaningful signal for pairwise
        similarity losses.

        Modifies `taken_mask` in-place by marking taken indices as True.

        Args:
            group_dict: Mapping of key -> list of dataset indices.
            active_keys: Set of keys in group_dict that still have available samples.
            taken_mask: Boolean numpy array of shape (N,); True = already assigned.
            budget_left: Remaining node/edge budget for the current batch.
            take_k: Maximum number of keys to visit (total samples taken ≤ take_k * per_key).
            limited_by_idx: 0 for frag_node, 1 for frag_edge.
            per_key: Maximum number of samples to take from each key. Defaults to 1.

        Returns:
            Tuple of (taken indices list, updated budget_left).
        """
        taken = []
        if not active_keys:
            return taken, budget_left
        keys = list(active_keys)
        # randomize key order using PyTorch Generator for consistency
        if self.generator is not None:
            perm = th.randperm(len(keys), generator=self.generator)
            keys = [keys[i] for i in perm.tolist()]
        # If no generator, keys remain in original order (deterministic)

        for key in keys:
            if take_k <= 0:
                break
            candidates = group_dict[key]  # plain Python list, no .tolist() needed
            count_from_key = 0
            for idx in candidates:
                if count_from_key >= per_key:
                    break
                if taken_mask[idx]:  # faster than `idx not in available` (set lookup)
                    continue
                # check size against remaining budget
                n = int(self._data_meta[idx, limited_by_idx])
                if n > budget_left and self._skip_too_big:
                    continue
                # Match SpecMolFragDynamicBatchSampler behavior: when skip_too_big=False,
                # allow an oversized singleton batch instead of silently dropping the sample.
                if n > budget_left and not (n > self._max_num and budget_left == self._max_num):
                    continue
                # take it — mark in taken_mask immediately so fill phase sees it
                taken_mask[idx] = True
                taken.append(idx)
                budget_left = max(0, budget_left - n)
                count_from_key += 1
            if count_from_key > 0:
                take_k -= 1
        return taken, budget_left

    def _pre_compute_batches(self):
        """Pre-compute batches for one epoch.

        Key optimisations vs. original:
        - Boolean numpy mask replaces Python set — O(1) membership test and mark.
        - Per-group available counts (group1_avail_count, group2_avail_count) replace
          the O(G*S) `any(idx in available for idx in group)` exhausted-key scan:
          counts are decremented O(1) per taken sample via the inverse maps built in
          _pre_load_metadata_and_groups.
        - Group candidate lists are plain Python lists (no .tolist() per call).
        - Fill phase uses np.where(~taken_mask) — vectorised C scan replaces
          list(python_set) materialisation every batch.
        """
        self._batches = []
        n = min(len(self.data_source), self._max_sampling_step)
        limited_by_idx = 0 if self._limited_by == "frag_node" else 1

        # Boolean mask over full dataset; pre-mark out-of-range indices as taken
        taken_mask = np.zeros(len(self.data_source), dtype=bool)
        if n < len(self.data_source):
            taken_mask[n:] = True
        remaining = n

        # Per-group available counts — O(1) exhausted-key detection
        if n == len(self.data_source):
            group1_avail_count = {k: len(v) for k, v in self._group1.items()}
            group2_avail_count = {k: len(v) for k, v in self._group2.items()}
        else:
            group1_avail_count = {k: sum(1 for i in v if i < n) for k, v in self._group1.items()}
            group2_avail_count = {k: sum(1 for i in v if i < n) for k, v in self._group2.items()}
        active_group1_keys = {k for k, c in group1_avail_count.items() if c > 0}
        active_group2_keys = {k for k, c in group2_avail_count.items() if c > 0}

        while remaining > 0:
            batch = []
            budget_left = self._max_num
            samples_to_take = self._return_batch_at if self._return_batch_at > 0 else remaining

            # Group1 phase
            if active_group1_keys and samples_to_take > 0:
                k1 = min(self.sample_k1, samples_to_take)
                taken1, budget_left = self._sample_from_group_dict(
                    self._group1,
                    active_group1_keys,
                    taken_mask,
                    budget_left,
                    k1,
                    limited_by_idx=limited_by_idx,
                    per_key=self.sample_k1_per_key,
                )
                batch.extend(taken1)
                samples_to_take -= len(taken1)
                remaining -= len(taken1)
                # O(taken1) count updates — replaces O(G*S) cleanup scan
                for idx in taken1:
                    k = self._idx_to_group1_key[idx]
                    group1_avail_count[k] -= 1
                    if group1_avail_count[k] == 0:
                        active_group1_keys.discard(k)
                    k2 = self._idx_to_group2_key[idx]
                    group2_avail_count[k2] -= 1
                    if group2_avail_count[k2] == 0:
                        active_group2_keys.discard(k2)

            # Group2 phase
            if active_group2_keys and samples_to_take > 0:
                k2 = min(self.sample_k2, samples_to_take)
                taken2, budget_left = self._sample_from_group_dict(
                    self._group2,
                    active_group2_keys,
                    taken_mask,
                    budget_left,
                    k2,
                    limited_by_idx=limited_by_idx,
                    per_key=self.sample_k2_per_key,
                )
                batch.extend(taken2)
                samples_to_take -= len(taken2)
                remaining -= len(taken2)
                for idx in taken2:
                    k1 = self._idx_to_group1_key[idx]
                    group1_avail_count[k1] -= 1
                    if group1_avail_count[k1] == 0:
                        active_group1_keys.discard(k1)
                    k2 = self._idx_to_group2_key[idx]
                    group2_avail_count[k2] -= 1
                    if group2_avail_count[k2] == 0:
                        active_group2_keys.discard(k2)

            # Fill phase: np.where vectorised scan replaces list(python_set) per batch
            if budget_left > 0 and samples_to_take > 0:
                avail_indices = np.where(~taken_mask[:n])[0]
                if len(avail_indices) > 0:
                    if self.generator is not None:
                        perm = th.randperm(len(avail_indices), generator=self.generator).numpy()
                        avail_indices = avail_indices[perm]
                    for idx in avail_indices.tolist():
                        if taken_mask[idx]:
                            continue
                        if samples_to_take <= 0:
                            break
                        n_val = int(self._data_meta[idx, limited_by_idx])
                        if n_val > self._max_num and self._skip_too_big:
                            continue
                        if n_val > budget_left and not (n_val > self._max_num and len(batch) == 0):
                            continue
                        batch.append(idx)
                        samples_to_take -= 1
                        remaining -= 1
                        taken_mask[idx] = True
                        k1 = self._idx_to_group1_key[idx]
                        group1_avail_count[k1] -= 1
                        if group1_avail_count[k1] == 0:
                            active_group1_keys.discard(k1)
                        k2 = self._idx_to_group2_key[idx]
                        group2_avail_count[k2] -= 1
                        if group2_avail_count[k2] == 0:
                            active_group2_keys.discard(k2)
                        budget_left = max(0, budget_left - n_val)

            if not batch:
                # Nothing could be placed (maybe all remaining too large)
                break

            self._batches.append(batch)

        logger.info(
            f"[DualGroupDynamicBatchSampler] Prepared {len(self._batches)} batches respecting max {self._limited_by}={self._max_num}"
        )

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)
