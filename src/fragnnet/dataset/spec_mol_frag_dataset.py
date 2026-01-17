import sys
from collections.abc import Iterator

import numpy as np
import torch as th
import torch.nn.functional as F
import torch_geometric as pyg
from torch.utils.data.sampler import BatchSampler, Sampler, WeightedRandomSampler
from torch_geometric.data import Batch
from tqdm import tqdm

from fragnnet.dataset import BaseDataset, SpecMolDataset
from fragnnet.utils.feat_utils import batch_mols_frags, get_frag_graph
from fragnnet.utils.formula_utils import (
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
)
from fragnnet.utils.frag_utils import NODE_FEAT_TO_IDX, load_frag_d
from fragnnet.utils.misc_utils import get_pyg_memory_usage


class SpecMolFragDataset(SpecMolDataset):
    def __init__(
        self,
        spec_fp: str,
        mol_fp: str,
        split_dp: str,
        split: str,
        subsample_params: dict,
        spec_params: dict,
        mol_params: dict,
        frag_dp: str,
        frag_params: dict,
        spec_pp_sd: dict | None = None,  # preprocessed spec data
        mol_pp_sd: dict | None = None,  # preprocessed mol data
        frag_pl_sd: dict | None = None,  # preloaded frag data
        frag_pp_sd: dict | None = None,  # preprocessed frag data
        **kwargs,
    ) -> None:
        """
        Initialize spectrum-molecule-fragment dataset with fragmentation graphs.

        Args:
            spec_fp: File path to the spectral data pickle file.
            mol_fp: File path to the molecular data pickle file.
            split_dp: Directory path containing split CSV files.
            split: Name of the data split (e.g., 'train', 'val', 'test').
            subsample_params: Dictionary of subsampling parameters.
            spec_params: Dictionary of spectral processing parameters.
            mol_params: Dictionary of molecular processing parameters.
            frag_dp: Directory path containing fragmentation DAG data.
            frag_params: Dictionary of fragmentation processing parameters.
            spec_pp_sd: Optional shared dictionary for preprocessed spectral data.
            mol_pp_sd: Optional shared dictionary for preprocessed molecular data.
            frag_pl_sd: Optional shared dictionary for preloaded fragmentation data.
            frag_pp_sd: Optional shared dictionary for preprocessed fragmentation data.
            **kwargs: Additional keyword arguments.
        """

        BaseDataset.__init__(self)
        self._base_init(
            spec_fp=spec_fp,
            mol_fp=mol_fp,
            split_dp=split_dp,
            split=split,
            subsample_params=subsample_params,
            spec_params=spec_params,
        )
        self.mol_params = mol_params
        self.frag_dp = frag_dp
        self.frag_params = frag_params

        if spec_pp_sd is None:
            spec_pp_sd = {}
        if mol_pp_sd is None:
            mol_pp_sd = {}
        if frag_pl_sd is None:
            frag_pl_sd = {}
        if frag_pp_sd is None:
            frag_pp_sd = {}

        self._preprocess_spec(spec_pp_sd)
        self._preprocess_mol(mol_pp_sd)
        self._preprocess_frag(frag_pl_sd, frag_pp_sd)

    @staticmethod
    def get_data_dict_types() -> list[str]:
        """
        Return list of data dictionary types used for caching.

        Returns:
            List containing data dictionary type identifiers:
                - 'spec_pp_sd': Spectral Preprocessed Shared Dictionary
                - 'mol_pp_sd': Molecular Preprocessed Shared Dictionary
                - 'frag_pl_sd': Fragment Preloaded Shared Dictionary
                - 'frag_pp_sd': Fragment Preprocessed Shared Dictionary
        """
        return ["spec_pp_sd", "mol_pp_sd", "frag_pl_sd", "frag_pp_sd"]

    def _preprocess_frag(self, frag_pl_sd: dict, frag_pp_sd: dict) -> None:
        """
        Preload and pre-process fragmentation DAG data into shared dictionaries.

        Args:
            frag_pl_sd: Shared dictionary to store preloaded fragmentation entries.
            frag_pp_sd: Shared dictionary to store preprocessed fragmentation data.
        """

        # preload frag dags
        if self.frag_params["preload"]:
            self.frag_entries = frag_pl_sd
            total_frag_entry_size = 0
            for mol_id in tqdm(
                self.mol_df["mol_id"].values, desc="> preload frag", total=len(self.mol_df)
            ):
                frag_entry = load_frag_d(mol_id, self.frag_dp, self.frag_params["compressed"])
                total_frag_entry_size += get_pyg_memory_usage(frag_entry["dag"])
                self.frag_entries[mol_id] = frag_entry
            print(f"> total_frag_entry_size: {total_frag_entry_size / 1e6:.2f} MB")

        # preprocess frag dags
        if self.frag_params["preprocess"]:
            assert self.frag_params["preload"]
            self.frag_data = frag_pp_sd
            total_frag_data_size = 0
            for k in tqdm(
                list(self.frag_entries.keys()),
                desc="> preprocess frag",
                total=len(self.frag_entries),
            ):
                frag_data = self._process_neutral_frag(self.frag_entries.pop(k))
                total_frag_data_size += get_pyg_memory_usage(frag_data["frag_pyg"])
                self.frag_data[k] = frag_data
            print(f"> total_frag_data_size: {total_frag_data_size / 1e6:.2f} MB")
            # remove them from the entries
            self.frag_entries = {}

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single spectrum-molecule-fragment sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Dictionary containing spectral, molecular, and fragmentation data.
        """
        spec_entry = self.spec_df.iloc[idx]
        mol_id = spec_entry["mol_id"]
        mol_entry = self.mol_df.loc[mol_id]

        # spectra data
        if self.spec_params["preprocess"]:
            # we don't need to deep copy here because spec data won't be modified
            spec_data = self.spec_datas[idx]
        else:
            spec_data = self._process_spec(spec_entry)

        # molecule data
        if self.mol_params["preprocess"]:
            # we don't need to deep copy here because mol data won't be modified
            mol_data = self.mol_datas[mol_id]
        else:
            mol_data = self._process_mol(mol_entry)

        # frag data
        if self.frag_params["preprocess"]:
            frag_data = self.frag_data[mol_id].copy()

            # formula_peak_mzs = frag_entry["formula_peak_mzs"]
            # formula_peak_mzs = formula_peak_mzs[:,:self.frag_params["num_isotopes"]]
            # in this case we have not applied precursor type changes yet
            # frag_formula_peak_mzs  is in neutral form
            neutral_peaks = frag_data["frag_formula_peak_mzs"]
            frag_data = self._apply_fragment_mz_change(
                frag_data, spec_entry["prec_type"], neutral_peaks, clone_pyg=True
            )

        elif self.frag_params["preload"]:
            frag_entry = self.frag_entries[mol_id]
            # frag_entry['frag_pyg'] = frag_entry['frag_pyg'].clone() # clone to avoid modifying the original one
            frag_data = self._process_neutral_frag(frag_entry)
            frag_data = self._apply_fragment_mz_change(
                frag_data,
                spec_entry["prec_type"],
                frag_data["frag_formula_peak_mzs"],
                clone_pyg=True,
            )
        else:
            frag_entry = self._load_frag_entry(mol_id)
            frag_data = self._process_neutral_frag(frag_entry)
            frag_data = self._apply_fragment_mz_change(
                frag_data,
                spec_entry["prec_type"],
                frag_data["frag_formula_peak_mzs"],
                clone_pyg=False,
            )

        data = {**spec_data, **mol_data, **frag_data}
        return data

    def _load_frag_entry(self, mol_id: str) -> dict:
        """
        Load fragmentation entry from precomputed DAG on disk.

        Args:
            mol_id: Molecule identifier.

        Returns:
            Dictionary containing DAG and fragmentation data.
        """

        frag_entry = load_frag_d(mol_id, self.frag_dp, self.frag_params["compressed"])
        return frag_entry

    def _process_neutral_frag(self, frag_entry: dict) -> dict:
        """
        Process the fragmentation dictionary to generate neutral fragment data.

        This method processes the fragmentation graph (frag DAG) and associated data,
        such as formula peak probabilities, formula strings, and m/z values.
        Args:
            frag_entry (dict): Fragmentation entry containing the DAG and related data.

        Returns:
            dict: A dictionary containing processed fragmentation data, including:
                  - 'frag_pyg': PyG graph of the fragmentation DAG.
                  - 'frag_formula_peak_probs': Normalized formula peak probabilities.
                  - 'frag_formula_str': Array of formula strings.
                  - 'frag_formula_peak_mzs': Adjusted m/z values for the fragmentation peaks.
        """
        frag_data = {}
        # case use frag dags as second gnn
        if self.frag_params["pyg"]:
            frag_pyg = frag_entry["dag"]
            frag_pyg = get_frag_graph(
                frag_pyg,
                self.frag_params["pyg_node_feats"],
                self.frag_params["pyg_edge_feats"],
                self.frag_params["pyg_edges"],
                self.frag_params["pyg_bigraph"],
            )
            frag_data["frag_pyg"] = frag_pyg

        if self.frag_params["formula_peak_probs"]:
            formula_peak_probs = frag_entry["formula_peak_probs"]
            formula_peak_probs = F.normalize(
                formula_peak_probs[:, : self.frag_params["num_isotopes"]], dim=1, p=1
            )
            frag_data["frag_formula_peak_probs"] = formula_peak_probs
        if self.frag_params["formula_str"]:
            # import pdb; pdb.set_trace()
            formula_str = list(frag_entry["idx_to_formula"].values())
            assert formula_str[0] == "", formula_str[0]
            formula_str = np.array(formula_str)
            frag_data["frag_formula_str"] = formula_str

        if self.frag_params["formula_peak_mzs"]:
            formula_peak_mzs = frag_entry["formula_peak_mzs"]
            formula_peak_mzs = formula_peak_mzs[:, : self.frag_params["num_isotopes"]]
            frag_data["frag_formula_peak_mzs"] = formula_peak_mzs

        return frag_data

    def _apply_fragment_mz_change(
        self, frag_data: dict, prec_type: str, neutral_peaks: th.Tensor, clone_pyg: bool = False
    ) -> dict:
        """
        Apply precursor mass difference for Charge Remote Fragmentation (CRF) and
        Charge Migration Fragmentation (CMF) to frag_data.

        This method updates 'frag_formula_peak_mzs', 'frag_formula_peak_probs',
        'frag_formula_str', and 'frag_pyg' in frag_data.

        Args:
            frag_data (dict): The dictionary containing fragmentation data.
            prec_type (str): The precursor type string.
            neutral_peaks (th.Tensor): The neutral formula peak m/z values.
            clone_pyg (bool): Whether to clone the PyG object before modification.
                              Set to True if frag_data['frag_pyg'] is a reference to cached data.

        Returns:
            dict: The updated frag_data dictionary.
        """
        prec_type_mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
        crf_peaks = neutral_peaks + prec_type_mass_diff

        # Charge Migration Fragmentation stuff
        if self.frag_params["include_cmf"] and prec_type not in ["[M+H]+", "[M-H]-", "[M]+", "M-"]:
            # Check if cmf_h_formulae_idx exists in pyg node features:
            assert "cmf_h_formulae_idx" in self.frag_params["pyg_node_feats"], (
                "cmf_h_formulae_idx not in pyg_node_feats"
            )
            cmf_type_mass_diff = PREC_TYPE_TO_CMF_MASS_DIFF[prec_type]
            num_formulae = neutral_peaks.shape[0]
            cmf_peaks = neutral_peaks[1:] + cmf_type_mass_diff  # first peak is for NULL

            # Double all formula-related arrays
            if self.frag_params["formula_peak_mzs"]:
                frag_data["frag_formula_peak_mzs"] = th.cat([crf_peaks, cmf_peaks], dim=0)
            if self.frag_params["formula_peak_probs"]:
                frag_data["frag_formula_peak_probs"] = th.cat(
                    [
                        frag_data["frag_formula_peak_probs"],
                        frag_data["frag_formula_peak_probs"][1:],
                    ],
                    dim=0,
                )
            if "frag_formula_str" in frag_data:
                frag_data["frag_formula_str"] = np.concatenate(
                    [frag_data["frag_formula_str"], frag_data["frag_formula_str"][1:]], axis=0
                )

            # update pyg
            if clone_pyg:
                frag_pyg = frag_data["frag_pyg"].clone()
            else:
                frag_pyg = frag_data["frag_pyg"]

            frag_node_feat_idxs = frag_pyg.node_feat_idxs[0]
            crf_node_feat_idx = NODE_FEAT_TO_IDX["h_formulae_idx"]
            cmf_node_feat_idx = NODE_FEAT_TO_IDX["cmf_h_formulae_idx"]
            crf_formula_matrix = frag_pyg.x[
                :,
                frag_node_feat_idxs[crf_node_feat_idx] : frag_node_feat_idxs[crf_node_feat_idx + 1],
            ]

            # update cmf formula matrix if crf_formula_matrix value is not zero
            # if crf_formula_matrix is zero, that means h_transfer is not possible, then cmf formula matrix should be zero
            # cmf formula matrix is shifted by num_formulae-1, as forumlea[0] is null formula
            frag_pyg.x[
                :,
                frag_node_feat_idxs[cmf_node_feat_idx] : frag_node_feat_idxs[cmf_node_feat_idx + 1],
            ] = th.where(
                crf_formula_matrix != 0, crf_formula_matrix + num_formulae - 1, crf_formula_matrix
            )
            frag_data["frag_pyg"] = frag_pyg
        else:
            frag_data["frag_formula_peak_mzs"] = crf_peaks
        return frag_data

    @staticmethod
    def get_collate_fn():
        """
        Get the collate function for batching dataset samples.

        Returns:
            Collate function to use with DataLoader.
        """
        return SpecMolFragDataset.collate_fn

    @staticmethod
    def _special_collate(keys: list[str], collate_data: dict) -> None:
        """
        Handle special collation for fragmentation graphs and molecular data.

        Mutates keys and collate_data in-place to batch fragment DAGs with molecules.

        Args:
            keys: List of data keys being collated (modified in-place).
            collate_data: Dictionary containing lists of values to collate (modified in-place).
        """

        if "frag_pyg" in keys:
            assert "mol_pyg" in keys
            assert "frag_formula_peak_mzs" in keys
            assert "frag_formula_peak_probs" in keys
            # process
            batch_mol_frag_data = batch_mols_frags(
                collate_data["mol_pyg"],
                collate_data["frag_pyg"],
                collate_data["frag_formula_peak_mzs"],
                collate_data["frag_formula_peak_probs"],
            )
            for k, v in batch_mol_frag_data.items():
                collate_data[k] = v
            # remove from list
            keys.remove("frag_pyg")
            keys.remove("mol_pyg")
            keys.remove("frag_formula_peak_mzs")
            keys.remove("frag_formula_peak_probs")
        SpecMolDataset._special_collate(keys, collate_data)

    @staticmethod
    def collate_fn(data_list: list[dict]) -> dict:
        """
        Collate a list of spectrum-molecule-fragment data samples into a batched dictionary.

        Args:
            data_list: List of data dictionaries from dataset samples.

        Returns:
            Batched data dictionary ready for model input.
        """

        # prevent edge case causing crash
        if len(data_list) == 0:
            return {"batch_size": th.tensor(0)}
        batch_size, keys, collate_data = SpecMolFragDataset._setup_collate(data_list)
        # special handling
        SpecMolFragDataset._special_collate(keys, collate_data)
        SpecMolFragDataset._standard_collate(batch_size, keys, collate_data)
        return collate_data


def get_batch_memory(batch: dict) -> tuple[dict, int]:
    """
    Calculate memory usage of a batched data dictionary.

    Args:
        batch: Batched data dictionary.

    Returns:
        Tuple containing:
            - Dictionary mapping keys to their memory usage in bytes
            - Total memory usage in bytes

    Raises:
        ValueError: If an unsupported data type is encountered.
    """

    batch_mem_d = {}
    for k, v in batch.items():
        if isinstance(v, th.Tensor):
            batch_mem_d[k] = v.element_size() * v.nelement()
        elif isinstance(v, Batch):
            batch_mem_d[k] = pyg.profile.get_data_size(v)
        elif isinstance(v, list):
            batch_mem_d[k] = sum(sys.getsizeof(x) for x in v)
        else:
            raise ValueError(f"Unsupported type: {type(v)}")
    batch_mem_total = sum(batch_mem_d.values())
    return batch_mem_d, batch_mem_total


def find_largest_batch(dl) -> tuple[dict, int]:
    """
    Find the batch with largest memory footprint in a DataLoader.

    Args:
        dl: PyTorch DataLoader to analyze.

    Returns:
        Tuple containing:
            - Memory dictionary of the largest batch
            - Total memory of the largest batch in bytes
    """

    batch_mem_ds, batch_mem_totals = [], []
    for batch in iter(dl):
        batch_mem_d, batch_mem_total = get_batch_memory(batch)
        batch_mem_ds.append(batch_mem_d)
        batch_mem_totals.append(batch_mem_total)
    argmax_idx = np.argmax(batch_mem_totals)
    return batch_mem_ds[argmax_idx], batch_mem_totals[argmax_idx]


class SpecMolFragDynamicBatchSampler(BatchSampler):
    """Dynamically adds samples to a mini-batch up to a maximum size  either based on number of nodes on frag DAG or number of edges on frag DAG.
    This is used to avoid CUDA OOM errors, implmentaion is inspired by PyG DynamicBatchSampler, and this should be used to replace default BatchSampler
    This should have the same random sampling beheivor as RandomSampler
    """

    def __init__(
        self,
        data_source: SpecMolFragDataset,
        max_num: int,
        limited_by: str = "frag_edge",
        skip_too_big: bool = False,
        num_samples=None,
        return_batch_at=0,
        sampler=None,
    ) -> None:
        """
        Initialize dynamic batch sampler for fragmentation datasets.

        Args:
            data_source: Dataset to sample from.
            max_num: Maximum number of nodes/edges per batch.
            limited_by: Limit batches by 'frag_node' or 'frag_edge'.
            skip_too_big: Whether to skip samples larger than max_num.
            num_samples: Number of samples to draw (None = all samples).
            return_batch_at: Return batch at this interval for gradient accumulation.
            sampler: Optional sampler for custom sampling order.

        Raises:
            ValueError: If max_num is not positive or limited_by is invalid.
        """
        if not isinstance(max_num, int) or max_num <= 0:
            raise ValueError("`dag_node` should be a positive integer value (got {max_num}).")
        if limited_by not in ["frag_node", "frag_edge"]:
            raise ValueError(
                "`limited_by` choice should be either "
                f"'frag_node' or 'frag_edge' (got '{limited_by}')."
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
        self._pre_load_batches()
        self._pre_compute_batches()

    def _pre_load_batches(self) -> None:
        """
        Pre-load batch metadata (node and edge counts) from all samples.

        Caches the number of nodes and edges for each sample to enable
        efficient dynamic batch construction.
        """

        # get data meta once and cache them
        assert len(self._data_meta) == 0, len(self._data_meta)
        expected_total = 0
        warning_msg = ""
        for dataset_idx in tqdm(
            range(len(self.data_source)), desc="SpecMolFragDynamicBatchSampler:pre_load_batches"
        ):
            data = self.data_source[dataset_idx]
            self._data_meta.append((data["frag_pyg"].num_nodes, data["frag_pyg"].num_edges))
            n = (
                self._data_meta[dataset_idx][0]
                if self._limited_by == "frag_node"
                else self._data_meta[dataset_idx][1]
            )
            if not (n > self._max_num and self._skip_too_big):
                expected_total += 1
            else:
                warning_msg += (
                    "Size of data sample at index "
                    + f"{dataset_idx} is larger than "
                    + f"{self._max_num} {self._limited_by}s "
                    + f"Got {n} {self._limited_by}s."
                    + "This sample can not fit into batch. "
                )
                if self._skip_too_big:
                    warning_msg += "Sampler will skip this to prevent CUDA OOM ERROR \n"
                else:
                    warning_msg += (
                        "Attempting to fit this in to batch, this may cause CUDA OOM ERROR  \n"
                    )
                # warnings.warn(warning_msg)
                # print(warning_msg)
        if warning_msg != "":
            print("[specmolfrag_dynamic_batch_sampler]", warning_msg)
        print(
            f"[SpecMolFragDynamicBatchSampler] Expecting {expected_total}/{len(self.data_source)} samples with skip_too_big:{self._skip_too_big}"
        )

    def _pre_compute_batches(self) -> None:
        """
        Pre-compute batch indices based on size constraints.

        Groups samples into batches such that the total number of nodes/edges
        per batch does not exceed max_num.
        """

        self._batches = []

        if self.sampler is not None:
            indices = th.tensor([idx for idx in self.sampler])
        else:
            indices = th.arange(len(self.data_source), dtype=th.long)

        # limited index to _max_sampling_step
        indices = indices[: self._max_sampling_step]

        num_processed = 0
        batch = []
        batch_n = 0
        batch_filled = False

        # Fill batch
        for idx in indices:
            # Size of sample
            n = (
                self._data_meta[idx.item()][0]
                if self._limited_by == "frag_node"
                else self._data_meta[idx.item()][1]
            )
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
                self._batches.append(batch)
                batch_n = 0
                batch = []
                batch_filled = False
            # Add sample to current batch
            batch.append(idx.item())
            num_processed += 1
            batch_n += n

        if len(batch) > 0:
            self._batches.append(batch)
        print(
            f"[SpecMolFragDynamicBatchSampler] Batch indices computed, Expecting {len(self._batches)} mini-batch for next epoch"
        )

    def __iter__(self) -> Iterator[list[int]]:
        """we use a pre computed batche list, this way we could have a correct batch size
            PL uses last batch in progress to toggle val step in training

        Yields:
            Iterator[List[int]]: _description_
        """
        for batch in self._batches:
            yield batch

    def __len__(self) -> int:
        """Note The __len__() method isn't strictly required by DataLoader, but is expected in any calculation involving the length of a DataLoader.
            ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
        Returns:
            int: length of datasource
        """
        return len(self._batches)


class GroupSampler(Sampler):
    def __init__(self, data_source: SpecMolFragDataset, sample_k=None, generator=None) -> None:
        """
        Initialize group-based sampler for balanced sampling across spectrum groups.

        Args:
            data_source: Dataset to sample from.
            sample_k: Number of samples to draw per group (default: 3).
            generator: Optional random number generator for reproducibility.
        """
        self.data_source = data_source
        self.num_samples = None
        self._data_meta_d = {}
        self.sample_k = 3 if sample_k is None else sample_k
        self.generator = generator
        self._pre_compute_meta()
        self._pre_compute_batches()

    def _pre_compute_meta(self) -> None:
        """
        Pre-compute metadata mapping group IDs to sample indices.

        Builds a dictionary mapping each group_id to the list of dataset
        indices belonging to that group.
        """

        for dataset_idx in range(len(self.data_source)):
            data = self.data_source[dataset_idx]
            group_id = data["group_id"].item()
            if group_id not in self._data_meta_d:
                self._data_meta_d[group_id] = []
            self._data_meta_d[group_id].append(dataset_idx)

        for group_id in self._data_meta_d:
            self._data_meta_d[group_id] = th.tensor(self._data_meta_d[group_id])

    def _pre_compute_batches(self) -> None:
        """
        Pre-compute sampled indices for the epoch.

        Samples up to sample_k indices from each group and shuffles them
        to create a balanced sampling distribution.
        """

        if self.generator is None:
            seed = int(th.empty((), dtype=th.int64).random_().item())
            generator = th.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        sampled_indices = []
        for group_id in self._data_meta_d:
            group_indices = self._data_meta_d[group_id]
            sampled_group_indices = group_indices[
                th.randperm(min(len(group_indices), self.sample_k), generator=generator)
            ]
            sampled_indices.append(sampled_group_indices)
        sampled_indices = th.cat(sampled_indices)
        self.sampled_indices = th.randperm(len(sampled_indices), generator=generator)
        self.num_samples = len(sampled_indices)

    def __iter__(self) -> Iterator[int]:
        """
        Yields:
            Iterator[List[int]]: _description_
        """

        for i in range(self.num_samples):
            yield self.sampled_indices[i].item()

    def __len__(self) -> int:
        """Note The __len__() method isn't strictly required by DataLoader
            ref: https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler
        Returns:
            int: length of datasource
        """
        assert self.num_samples is not None
        return self.num_samples


def get_group_sampler(
    ds: SpecMolFragDataset, sampler_type: str, avg_per_group: int, generator
) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced group/molecule sampling.

    Args:
        ds: Dataset to sample from.
        sampler_type: Weighting strategy ('group', 'mol', or 'group_mol').
        avg_per_group: Average number of samples to draw per group.
        generator: Random number generator for reproducibility.

    Returns:
        WeightedRandomSampler configured with appropriate sample weights,
        or None if sampler_type is invalid.
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

    # spec_per_group_2 = th.mean(th.unique(group_ids,return_counts=True)[1].float(),dim=0)
    # num_samples = th.ceil(len(ds) / spec_per_group_2 * avg_per_group).long()
    # num_samples = min(num_samples.item(), len(ds))

    # Get unique groups and their counts
    _, counts = th.unique(group_ids, return_counts=True)
    # For each group, we only  want to sample up to the number of available samples
    total_samples = sum(min(count.item(), avg_per_group) for count in counts)
    # Make sure we don't exceed the total dataset size
    num_samples = min(total_samples, len(ds))

    sampler = WeightedRandomSampler(
        sample_weights, num_samples=num_samples, replacement=False, generator=generator
    )
    return sampler
