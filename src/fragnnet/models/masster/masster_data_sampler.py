"""MASSter's batch sampler: adjacent-CE batching budgeted by molecular-graph edges.

Moved out of core ``data_sampler`` in the reorganization (docs/plans/repo_reorg_plan.md, Phase 4).
"""

import numpy as np
from torch.utils.data import Dataset

from fragnnet.dataset.data_sampler import _CEContextBatchSamplerBase


class MassterCEContextDynamicBatchSampler(_CEContextBatchSamplerBase):
    """Pack adjacent-CE MASSter spectra using molecular graph edges.

    This sampler never reads fragment-DAG graphs or precomputed DAG size metadata.
    """

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
        """Load molecular graph node/edge counts once per molecule."""
        metadata = []
        mol_sizes = {}
        for idx in range(len(self.data_source)):
            mol_id = self.data_source.spec_df.iloc[idx]["mol_id"]
            if mol_id not in mol_sizes:
                if hasattr(self.data_source, "mol_datas"):
                    graph = self.data_source.mol_datas[mol_id]["mol_pyg"]
                else:
                    graph = self.data_source[idx]["mol_pyg"]
                mol_sizes[mol_id] = (graph.num_nodes, graph.num_edges)
            metadata.append(mol_sizes[mol_id])
        return np.asarray(metadata, dtype=np.int64).reshape(-1, 2)
