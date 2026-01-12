"""
base_model.py

Base classes for embedding collision energy (CE), precursor type(Prec), and instrument type (Inst)
for FragGNN models. Intended to be used as base classes for more complex model architectures.

- CEModel: Handles collision energy embedding and transformation.
- PrecModel: Handles precursor type embedding.
- InstModel: Handles instrument type embedding.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.utils.misc_utils import scatter_reduce
from fragnnet.model.form_embedder import get_embedder
from fragnnet.utils.spec_utils import transform_ce

class CEModel:
    """Class for handling collision energy (CE) embedding."""

    def _ce_init(
        self,
        int_embedder,
        ce_insert_location: str,
        ce_insert_type: str,
        ce_insert_merge: bool,
        ce_insert_size: int,
        ce_mean: float,
        ce_std: float,
        ce_max: float,
    ):
        """
        Initialize CE embedding configuration and embedding layers.

        Args:
            int_embedder: Embedder for integer features.
            ce_insert_location: Where to insert CE embedding ("none", "mol", "frag", "mlp").
            ce_insert_type: Type of CE embedding ("id", "lin", "embed", "bin").
            ce_insert_merge: Whether to merge CE embeddings across batch.
            ce_insert_size: Output embedding size.
            ce_mean: Mean for CE normalization.
            ce_std: Std for CE normalization.
            ce_max: Max value for CE binning/embedding.
        """
        assert ce_insert_type in ["id","lin","embed","embed_clamped","bin"]
        assert ce_insert_location in ["none","mol","frag","mlp"]
        self.ce_insert_type = ce_insert_type
        self.ce_insert_location = ce_insert_location
        self.ce_insert_merge = ce_insert_merge
        self.ce_insert_size = ce_insert_size
        self.int_embedder = int_embedder
        self.ce_max = ce_max
        self.ce_mean = ce_mean
        self.ce_std = ce_std
        self._ce_location_check()
        self._setup_ce()

    def _ce_location_check(self):
        raise NotImplementedError

    def _setup_ce(self):
        """Setup CE embedding and transformation logic based on type and location."""
        # Setup embedding type
        if self.ce_insert_type == "id":
            def ce_transform(ce):
                ce = transform_ce(ce, self.ce_mean, self.ce_std)
                ce = ce.reshape(-1,1)
                ce = th.repeat_interleave(ce, self.ce_insert_size, dim=1)
                return ce
            ce_embedder = nn.Identity()
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "lin":
            def ce_transform(ce):
                ce = transform_ce(ce, self.ce_mean, self.ce_std)
                ce = ce.reshape(-1,1)
                return ce
            ce_embedder = nn.Linear(1,self.ce_insert_size)
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "embed":
            def ce_transform(ce):
                ce = th.clamp(ce, min=-1)
                ce = th.round(ce, decimals=0).long()
                ce = ce.reshape(-1,1)
                return ce
            embedder = get_embedder(self.int_embedder, max_count_int=int(self.ce_max)+1)  # +1 for the shifted range
            ce_embedder = nn.Sequential(
                embedder,
                nn.Linear(embedder.num_dim,self.ce_insert_size)
            )   
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "embed_clamped":
            def ce_transform(ce):
                ce = th.clamp(ce, min=-1, max=int(self.ce_max)-1)
                ce = th.round(ce, decimals=0).long()
                ce = ce.reshape(-1,1)
                return ce
            embedder = get_embedder(self.int_embedder, max_count_int=int(self.ce_max))
            ce_embedder = nn.Sequential(
                embedder,
                nn.Linear(embedder.num_dim,self.ce_insert_size)
            )   
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "bin":
            def ce_transform(ce):
                # Handle -1 (missing/unknown) by giving it its own bin (index 0)
                # Other values are binned normally then shifted up by 1
                ce = th.clamp(ce, min=-1, max=int(self.ce_max)-10)              
                # Create mask for -1 values
                missing_mask = (ce == -1)
                # For non-missing values: bin them and shift up by 1
                ce_binned = th.round(ce, decimals=-1).long() // 10
                ce_binned = th.where(missing_mask, th.tensor(0, device=ce.device, dtype=ce_binned.dtype), ce_binned + 1)
                # One-hot encode with -1 getting bin 0, others getting bins 1+
                ce = F.one_hot(ce_binned, num_classes=(int(self.ce_max)//10)+1).float()  # +1 for the -1 bin
                return ce
            ce_embedder = nn.Linear((int(self.ce_max)//10)+1,self.ce_insert_size)  # +1 for the -1 bin
            ce_input_dim = self.ce_insert_size

        # Setup location
        if self.ce_insert_location == "mol":
            ce_mol_input_dim = ce_input_dim
            ce_mlp_input_dim = 0
        elif self.ce_insert_location == "mlp":
            ce_mol_input_dim = 0
            ce_mlp_input_dim = ce_input_dim
        else:
            assert self.ce_insert_location == "none"
            ce_mol_input_dim = 0
            ce_mlp_input_dim = 0
        self.ce_transform = ce_transform
        self.ce_embedder = ce_embedder
        self.ce_mol_input_dim = ce_mol_input_dim
        self.ce_mlp_input_dim = ce_mlp_input_dim

    def embed_ce(self, ce, ce_batch_idxs, batch_size):
        """
        Embed the collision energy values.

        Args:
            ce: Tensor of collision energies.
            ce_batch_idxs: Batch indices for each CE value.
            batch_size: Total batch size.

        Returns:
            Embedded CE tensor or None.
        """
        if self.ce_insert_location != "none":
            ce_embed = self.ce_transform(ce)
            ce_embed = self.ce_embedder(ce_embed)
            # possibly merge the embeddings
            if self.ce_insert_merge:
                ce_embed = scatter_reduce(
                    src=ce_embed,
                    index=ce_batch_idxs.unsqueeze(1).expand_as(ce_embed),
                    reduce="mean",
                    dim_size=batch_size,
                    include_self=False
                )
        else:
            ce_embed = None
        return ce_embed

class PrecModel:
    """Class for handling precursor type embedding."""

    def _prec_init(
        self,
        prec_insert_location: str,
        prec_insert_size: int,
        prec_num_types: int
    ):
        """
        Initialize precursor embedding configuration and embedding layers.

        Args:
            prec_insert_location: Where to insert precursor embedding ("none", "mol", "mlp").
            prec_insert_size: Output embedding size.
            prec_num_types: Number of precursor types.
        """
        self.prec_insert_location = prec_insert_location
        self.prec_embedder = nn.Embedding(prec_num_types+1, prec_insert_size)   
        prec_dim = prec_insert_size

        self._prec_location_check()

        if self.prec_insert_location == "mol":
            prec_mol_input_dim = prec_dim
            prec_mlp_input_dim = 0
        elif self.prec_insert_location == "mlp":
            prec_mol_input_dim = 0
            prec_mlp_input_dim = prec_dim
        else:
            assert self.prec_insert_location == "none"
            prec_mol_input_dim = 0
            prec_mlp_input_dim = 0
        self.prec_mol_input_dim = prec_mol_input_dim
        self.prec_mlp_input_dim = prec_mlp_input_dim

    def _prec_location_check(self):
        raise NotImplementedError

    def embed_prec(self, prec_type):
        """
        Embed the precursor type.

        Args:
            prec_type: Tensor of precursor type indices.

        Returns:
            Embedded precursor tensor or None.
        """
        if self.prec_insert_location != "none":
            if prec_type is None:
                raise ValueError(f"prec_type is None but prec_insert_location is {self.prec_insert_location}")
            prec_embed = self.prec_embedder(prec_type)
        else:
            prec_embed = None
        return prec_embed

class InstModel:
    """Class for handling instrument type embedding."""

    def _inst_init(
        self,
        inst_insert_location: str,
        inst_insert_size: int,
        inst_num_types: int
    ):
        """
        Initialize instrument embedding configuration and embedding layers.

        Args:
            inst_insert_location: Where to insert instrument embedding ("none", "mol", "mlp").
            inst_insert_size: Output embedding size.
            inst_num_types: Number of instrument types.
        """
        self.inst_insert_location = inst_insert_location
        self.inst_embedder = nn.Embedding(inst_num_types + 1, inst_insert_size) 
        inst_dim = inst_insert_size

        self._inst_location_check()

        if self.inst_insert_location == "mol":
            inst_mol_input_dim = inst_dim
            inst_mlp_input_dim = 0
        elif self.inst_insert_location == "mlp":
            inst_mol_input_dim = 0
            inst_mlp_input_dim = inst_dim
        else:
            assert self.inst_insert_location == "none"
            inst_mol_input_dim = 0
            inst_mlp_input_dim = 0
        self.inst_mol_input_dim = inst_mol_input_dim
        self.inst_mlp_input_dim = inst_mlp_input_dim

    def _inst_location_check(self):
        raise NotImplementedError(".")

    def embed_inst(self, inst_type):
        """
        Embed the instrument type.

        Args:
            inst_type: Tensor of instrument type indices.

        Returns:
            Embedded instrument tensor or None.
        """
        if self.inst_insert_location != "none":
            if inst_type is None:
                raise ValueError(f"inst_type is None but inst_insert_location is {self.inst_insert_location}")
            inst_embed = self.inst_embedder(inst_type)
        else:
            inst_embed = None
        return inst_embed
