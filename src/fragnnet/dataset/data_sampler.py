"""
Data samplers shared by every model.

Includes:
- BondCountCurriculumSampler: Bond-count curriculum over fixed-size batches
- _CEContextBatchSamplerBase: Adjacent-CE window batching; the models' subclasses set the budget
- GroupSampler: Sample K specs per group_id
- FormulaPrecTypeGroupSampler: Sample K specs per formula and precursor type
- get_group_sampler: WeightedRandomSampler factory

FraGNNet's fragment-DAG samplers are in ``fragnnet_data_sampler`` and MASSter's in
``masster_data_sampler``.
"""

import logging
import math
import os
from collections import deque
from collections.abc import Iterator

import numpy as np
import torch as th
from torch.utils.data import BatchSampler, Dataset, Sampler, WeightedRandomSampler

logger = logging.getLogger(__name__)


def validate_bond_size_curriculum(start_fraction: float, curriculum_epochs: int) -> None:
    """Validate precursor-bond-count curriculum schedule values."""
    if (
        isinstance(start_fraction, bool)
        or not isinstance(start_fraction, (int, float))
        or not 0 < start_fraction <= 1
    ):
        raise ValueError("bond_size_curriculum_start_fraction must be in (0, 1]")
    if (
        isinstance(curriculum_epochs, bool)
        or not isinstance(curriculum_epochs, int)
        or curriculum_epochs <= 0
    ):
        raise ValueError("bond_size_curriculum_epochs must be a positive integer")


class BondCountCurriculumSampler(Sampler[int]):
    """Randomly sample the smallest precursor molecules first by heavy-bond count."""

    def __init__(
        self,
        data_source: Dataset,
        start_fraction: float = 0.25,
        curriculum_epochs: int = 20,
        generator: th.Generator | None = None,
    ) -> None:
        validate_bond_size_curriculum(start_fraction, curriculum_epochs)
        if not hasattr(data_source, "spec_df") or not hasattr(data_source, "mol_df"):
            raise TypeError("BondCountCurriculumSampler requires spec_df and mol_df on the dataset")

        mol_ids = data_source.spec_df["mol_id"].to_numpy()
        bond_counts = data_source.mol_df.loc[mol_ids, "num_bonds"].to_numpy()
        if not np.isfinite(bond_counts).all():
            raise ValueError("bond-size curriculum requires finite mol_df.num_bonds values")

        self.data_source = data_source
        self.bond_counts = np.asarray(bond_counts, dtype=np.int32)
        self.start_fraction = start_fraction
        self.curriculum_epochs = curriculum_epochs
        self.generator = generator
        self._epoch = 0

    def _bond_count_limit(self) -> int | None:
        if len(self.bond_counts) == 0:
            return None
        progress = min(self._epoch / self.curriculum_epochs, 1.0)
        fraction = self.start_fraction + (1.0 - self.start_fraction) * progress
        target_count = min(
            len(self.bond_counts),
            max(1, math.ceil(len(self.bond_counts) * fraction)),
        )
        return int(np.partition(self.bond_counts, target_count - 1)[target_count - 1])

    def _eligible_indices(self) -> np.ndarray:
        bond_count_limit = self._bond_count_limit()
        if bond_count_limit is None:
            return np.empty(0, dtype=np.int64)
        return np.flatnonzero(self.bond_counts <= bond_count_limit)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self) -> Iterator[int]:
        eligible_indices = self._eligible_indices()
        if len(eligible_indices) == 0:
            return iter(())
        order = th.randperm(len(eligible_indices), generator=self.generator).tolist()
        return iter(eligible_indices[order].tolist())

    def __len__(self) -> int:
        return len(self._eligible_indices())


#####################################################
# Batch Samplers
#####################################################


class _CEContextBatchSamplerBase(BatchSampler):
    """Shared adjacent-CE grouping and bounded packing behavior."""

    def __init__(
        self,
        data_source: Dataset,
        max_num: int,
        ce_key_by_inst: dict[str, str],
        spectra_per_context: int = 5,
        return_batch_at: int = 0,
        seed: int = 0,
        skip_too_big: bool = True,
    ) -> None:
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value (got {max_num}).")
        if not isinstance(spectra_per_context, int) or spectra_per_context < 2:
            raise ValueError("spectra_per_context must be an integer >= 2")
        if not isinstance(return_batch_at, int) or return_batch_at < 0:
            raise ValueError("return_batch_at must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(skip_too_big, bool):
            raise ValueError("skip_too_big must be boolean")
        if not hasattr(data_source, "spec_df") or not hasattr(data_source, "ce_key"):
            raise ValueError("CE context batching requires a dataset with spec_df and ce_key")
        if data_source.ce_key is None:
            raise ValueError("CE context batching requires an enabled collision-energy feature")
        if not isinstance(ce_key_by_inst, dict) or not ce_key_by_inst:
            raise ValueError("ce_key_by_inst must be a non-empty instrument-to-CE mapping")
        if any(
            not isinstance(inst_type, str) or ce_key not in {"nce", "ace"}
            for inst_type, ce_key in ce_key_by_inst.items()
        ):
            raise ValueError("ce_key_by_inst values must be 'nce' or 'ace'")

        if th.distributed.is_available() and th.distributed.is_initialized():
            world_size = th.distributed.get_world_size()
        else:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != 1:
            raise ValueError("CE context batching v1 supports single-GPU training only")

        self.data_source = data_source
        self._max_num = max_num
        self._ce_key_by_inst = ce_key_by_inst
        self._spectra_per_context = spectra_per_context
        self._return_batch_at = return_batch_at
        self._seed = seed
        self._skip_too_big = skip_too_big
        self._epoch = 0
        self._data_meta = self._load_size_metadata()
        self._contexts, self._fallback_indices, self._coverage_stats = self._load_context_metadata()
        self.stats: dict[str, int | float] = {}
        self._batches: list[list[int]] = []
        self._pre_compute_batches()

    @staticmethod
    def _ce_value(raw_ce) -> float | None:
        """Return the mean valid CE used for ordering, or ``None`` if missing."""
        values = np.asarray(raw_ce, dtype=np.float64).reshape(-1)
        values = values[np.isfinite(values) & (values >= 0)]
        if len(values) == 0:
            return None
        return round(float(values.mean()), 6)

    def _load_size_metadata(self) -> np.ndarray:
        raise NotImplementedError

    def _load_context_metadata(
        self,
    ) -> tuple[
        list[tuple[object, tuple[tuple[float, tuple[int, ...]], ...]]],
        list[tuple[object, int]],
        dict[str, int],
    ]:
        """Build pair-capable contexts and primary-loss-only fallback spectra."""
        spec_df = self.data_source.spec_df
        missing_columns = {"group_id", "inst_type"}.difference(spec_df.columns)
        if missing_columns:
            raise ValueError(
                "CE context batching requires spec_df columns " f"{sorted(missing_columns)}"
            )
        inst_types = set(spec_df["inst_type"].astype(str))
        unmapped_inst_types = inst_types.difference(self._ce_key_by_inst)
        if unmapped_inst_types:
            raise ValueError(
                "ce_key_by_inst has no CE key for instrument types "
                f"{sorted(unmapped_inst_types)}"
            )
        missing_ce_columns = {
            self._ce_key_by_inst[inst_type] for inst_type in inst_types
        }.difference(spec_df.columns)
        if missing_ce_columns:
            raise ValueError(
                "CE context batching requires configured CE columns "
                f"{sorted(missing_ce_columns)}"
            )
        if hasattr(self.data_source, "get_frag_mode_stats"):
            frag_modes = self.data_source.get_frag_mode_stats()
        else:
            frag_modes = ["unknown"] * len(spec_df)

        modes_by_group: dict[object, set[str]] = {}
        inst_types_by_group: dict[object, set[str]] = {}
        for idx in range(len(spec_df)):
            group_id = spec_df.iloc[idx]["group_id"]
            modes_by_group.setdefault(group_id, set()).add(str(frag_modes[idx]))
            inst_types_by_group.setdefault(group_id, set()).add(str(spec_df.iloc[idx]["inst_type"]))
        mixed_inst_groups = {
            group_id
            for group_id, group_inst_types in inst_types_by_group.items()
            if len(group_inst_types) > 1
        }
        if mixed_inst_groups:
            raise ValueError(
                "CE context batching requires one instrument type per group_id; "
                f"found mixed instruments in {len(mixed_inst_groups)} groups"
            )
        mixed_groups = {group_id for group_id, modes in modes_by_group.items() if len(modes) > 1}

        contexts: dict[object, dict[float, list[int]]] = {}
        fallback_indices: list[tuple[object, int]] = []
        mixed_mode_spectra = 0
        missing_ce_spectra = 0
        for idx in range(len(spec_df)):
            row = spec_df.iloc[idx]
            group_id = row["group_id"]
            if group_id in mixed_groups:
                mixed_mode_spectra += 1
                fallback_indices.append((group_id, idx))
                continue
            inst_type = str(row["inst_type"])
            ce_key = self._ce_key_by_inst[inst_type]
            ce_value = self._ce_value(row[ce_key])
            if ce_value is None:
                missing_ce_spectra += 1
                fallback_indices.append((group_id, idx))
                continue
            contexts.setdefault(group_id, {}).setdefault(ce_value, []).append(idx)

        eligible_contexts = []
        single_level_spectra = 0
        for group_id, levels in contexts.items():
            if len(levels) < 2:
                single_level_spectra += sum(len(indices) for indices in levels.values())
                fallback_indices.extend(
                    (group_id, idx) for indices in levels.values() for idx in indices
                )
                continue
            eligible_contexts.append(
                (
                    group_id,
                    tuple((ce, tuple(sorted(indices))) for ce, indices in sorted(levels.items())),
                )
            )

        coverage_stats = {
            "total_spectra": len(spec_df),
            "mixed_mode_spectra": mixed_mode_spectra,
            "missing_ce_spectra": missing_ce_spectra,
            "single_level_spectra": single_level_spectra,
            "eligible_spectra": sum(
                len(indices) for _, context in eligible_contexts for _, indices in context
            ),
        }
        return eligible_contexts, fallback_indices, coverage_stats

    def _context_atoms(
        self,
        levels: tuple[tuple[float, tuple[int, ...]], ...],
        context_idx: int,
    ) -> tuple[list[list[int]], int]:
        """Partition every context spectrum into bounded consecutive atoms."""
        rotation = self._epoch + context_idx
        primary = []
        replicate_singletons = []
        for _, indices in levels:
            primary_pos = rotation % len(indices)
            primary.append(indices[primary_pos])
            replicate_singletons.extend(
                idx for pos, idx in enumerate(indices) if pos != primary_pos
            )

        # Rotate chunk boundaries without wrapping the ordered CE axis. A short
        # prefix is allowed; it still receives the ordinary spectrum loss.
        boundary_offset = (
            rotation % self._spectra_per_context if len(primary) > self._spectra_per_context else 0
        )
        level_chunks = []
        if boundary_offset:
            level_chunks.append(primary[:boundary_offset])
        level_chunks.extend(
            primary[start : start + self._spectra_per_context]
            for start in range(boundary_offset, len(primary), self._spectra_per_context)
        )

        atoms = []
        oversized_spectra = 0
        for chunk in [*level_chunks, *[[idx] for idx in replicate_singletons]]:
            atom = []
            atom_edges = 0
            for idx in chunk:
                edges = int(self._data_meta[idx, 1])
                if edges > self._max_num:
                    oversized_spectra += 1
                    if atom:
                        atoms.append(atom)
                        atom = []
                        atom_edges = 0
                    if not self._skip_too_big:
                        atoms.append([idx])
                    continue
                if atom and atom_edges + edges > self._max_num:
                    atoms.append(atom)
                    atom = []
                    atom_edges = 0
                atom.append(idx)
                atom_edges += edges
            if atom:
                atoms.append(atom)
        return atoms, oversized_spectra

    def _pre_compute_batches(self) -> None:
        """Partition all spectra and greedily pack atomic context chunks."""
        generator = th.Generator().manual_seed(self._seed + self._epoch)
        order = (
            th.randperm(len(self._contexts), generator=generator).tolist() if self._contexts else []
        )

        atoms: list[tuple[object, list[int]]] = []
        oversized_spectra = 0
        for context_idx in order:
            group_id, levels = self._contexts[context_idx]
            context_atoms, context_oversized = self._context_atoms(levels, context_idx)
            atoms.extend((group_id, atom) for atom in context_atoms)
            oversized_spectra += context_oversized
        fallback_order = (
            th.randperm(len(self._fallback_indices), generator=generator).tolist()
            if self._fallback_indices
            else []
        )
        for fallback_idx in fallback_order:
            group_id, idx = self._fallback_indices[fallback_idx]
            if int(self._data_meta[idx, 1]) > self._max_num:
                oversized_spectra += 1
                if not self._skip_too_big:
                    atoms.append((group_id, [idx]))
            else:
                atoms.append((group_id, [idx]))

        # Contexts can contribute several chunks. Merge the contexts' chunk sequences in a
        # uniformly random order, each keeping its own chunk order, so every context's chunks
        # spread over the epoch. A round-robin merge left the context with the most chunks alone
        # at the end of every epoch, one spectrum per batch, and those batches set the BatchNorm
        # running statistics that evaluation uses.
        atoms_by_group: dict[object, deque[list[int]]] = {}
        for group_id, atom in atoms:
            atoms_by_group.setdefault(group_id, deque()).append(atom)
        slots = [group_id for group_id, group_atoms in atoms_by_group.items() for _ in group_atoms]
        atoms = [
            (slots[slot], atoms_by_group[slots[slot]].popleft())
            for slot in th.randperm(len(slots), generator=generator).tolist()
        ]

        # Pack greedily, at most one chunk per context in a batch. A chunk whose context is
        # already in the open batch waits and is offered first to the next batch; closing the
        # batch on the repeat instead would emit near-empty batches.
        batches: list[list[int]] = []
        pending = deque(atoms)
        while pending:
            batch: list[int] = []
            batch_size = 0
            batch_groups: set[object] = set()
            waiting: list[tuple[object, list[int]]] = []
            while pending:
                group_id, atom = pending[0]
                if group_id in batch_groups:
                    waiting.append(pending.popleft())
                    continue
                atom_size = int(self._data_meta[atom, 1].sum())
                exceeds_edges = batch and batch_size + atom_size > self._max_num
                exceeds_spectra = (
                    batch
                    and self._return_batch_at > 0
                    and len(batch) + len(atom) > self._return_batch_at
                )
                if exceeds_edges or exceeds_spectra:
                    break
                pending.popleft()
                batch.extend(atom)
                batch_size += atom_size
                batch_groups.add(group_id)
            batches.append(batch)
            pending.extendleft(reversed(waiting))

        selected_spectra = sum(len(atom) for _, atom in atoms)
        self._batches = batches
        self.stats = {
            **self._coverage_stats,
            "eligible_contexts": len(self._contexts),
            "emitted_chunks": len(atoms),
            "oversized_spectra": oversized_spectra,
            "selected_spectra": selected_spectra,
            "selected_spectra_fraction": selected_spectra
            / max(1, self._coverage_stats["total_spectra"]),
            "batches": len(batches),
            "sampler_seed": self._seed,
        }
        logger.info("[%s] %s", type(self).__name__, self.stats)

    def set_epoch(self, epoch: int) -> None:
        """Rotate CE chunk boundaries and primary replicate choice."""
        self._epoch = int(epoch)
        self._pre_compute_batches()

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


class GroupSampler(Sampler):
    """Sample K spectra per group_id to keep experimental groups together.

    Use case: When spectra are grouped by experiment/acquisition (group_id),
    this sampler ensures that K random spectra from each group are sampled
    in each epoch.
    """

    def __init__(
        self,
        data_source: Dataset,
        sample_k: int | None = None,
        generator: th.Generator | None = None,
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
        data_source: Dataset,
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
        if self.generator is not None:
            base = self._base_seed if self._base_seed is not None else 0
            self.generator.manual_seed(base + int(epoch))
        self._pre_compute_indices()

    def __iter__(self) -> Iterator[int]:
        for i in range(self.num_samples):
            yield int(self.sampled_indices[i].item())

    def __len__(self) -> int:
        return self.num_samples


def get_group_sampler(
    ds: Dataset, sampler_type: str, avg_per_group: int, generator: th.Generator
) -> WeightedRandomSampler | None:
    """Create a WeightedRandomSampler based on group/molecule or adduct+instrument statistics.

    Balances sampling to ensure certain groups/molecules are sampled proportionally.

    Args:
        ds: Dataset with get_group_mol_stats() and get_adduct_inst_type_stats() methods.
        sampler_type (str): Weighting strategy:
            - 'group': Weight by 1/specs_per_group (replacement=False)
            - 'mol': Weight by 1/specs_per_mol (replacement=False)
            - 'group_mol': Weight by 1/(specs_per_group * group_per_mol) (replacement=False)
            - 'adduct_inst_type': Weight by 1/count(prec_type, inst_type) (replacement=True).
              Oversample minority (adduct, instrument) combinations. Epoch size = len(ds).
            - 'adduct_inst_type_group': Weight by 1/(count(prec_type, inst_type) *
              specs_per_group) (replacement=True). Combines adduct+instrument balancing
              with group-level balancing. Epoch size = len(ds).
            - 'adduct_inst_type_frag_mode': Weight by 1/count(prec_type, inst_type, frag_mode)
              (replacement=True). Extends adduct+instrument balancing to also upweight minority
              fragmentation modes (e.g. CID vs HCD). Epoch size = len(ds).
            - 'adduct_inst_type_frag_mode_group': Weight by 1/(count(prec_type, inst_type,
              frag_mode) * specs_per_group) (replacement=True). Combines frag-mode balancing
              with group-level balancing. Epoch size = len(ds).
        avg_per_group (int): Average samples per group to target (used by group/mol types).
        generator (th.Generator): PyTorch generator for reproducibility.

    Returns:
        WeightedRandomSampler | None: Configured sampler, or None if invalid sampler_type.

    Example:
        ```python
        sampler = get_group_sampler(ds, "adduct_inst_type", avg_per_group=1, generator=generator)
        loader = DataLoader(ds, batch_sampler=sampler)
        ```
    """
    if sampler_type in (
        "adduct_inst_type",
        "adduct_inst_type_group",
        "adduct_inst_type_frag_mode",
        "adduct_inst_type_frag_mode_group",
    ):
        prec_types, inst_types = ds.get_adduct_inst_type_stats()
        use_frag_mode = sampler_type in (
            "adduct_inst_type_frag_mode",
            "adduct_inst_type_frag_mode_group",
        )
        if use_frag_mode:
            frag_modes = ds.get_frag_mode_stats()
        combo_counts: dict[str, int] = {}
        for i, (pt, it) in enumerate(zip(prec_types, inst_types)):
            key = f"{pt}::{it}::{frag_modes[i]}" if use_frag_mode else f"{pt}::{it}"
            combo_counts[key] = combo_counts.get(key, 0) + 1
        if use_frag_mode:
            sample_weights = th.tensor(
                [
                    1.0 / combo_counts[f"{pt}::{it}::{fm}"]
                    for pt, it, fm in zip(prec_types, inst_types, frag_modes)
                ],
                dtype=th.float32,
            )
        else:
            sample_weights = th.tensor(
                [1.0 / combo_counts[f"{pt}::{it}"] for pt, it in zip(prec_types, inst_types)],
                dtype=th.float32,
            )
        if sampler_type in ("adduct_inst_type_group", "adduct_inst_type_frag_mode_group"):
            _, _, spec_per_group, _, _ = ds.get_group_mol_stats()
            sample_weights = sample_weights / spec_per_group.float()
        num_samples = len(ds)
        return WeightedRandomSampler(
            sample_weights, num_samples=num_samples, replacement=True, generator=generator
        )

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
