"""
base_model.py

Base classes for embedding collision energy (CE), precursor type(Prec), instrument type (Inst),
and fragmentation mode (FragMode) for FragGNN models. Intended to be used as base classes for
more complex model architectures.

- CEModel: Handles collision energy embedding and transformation.
- PrecModel: Handles precursor type embedding.
- InstModel: Handles instrument type embedding.
- FragModeModel: Handles fragmentation mode embedding (HCD/CID/unknown).
- FragModeScaler: Mode-conditioned affine output scaling layer.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.model.form_embedder import get_embedder
from fragnnet.utils.misc_utils import scatter_reduce
from fragnnet.utils.spec_utils import transform_ce


class CEModel:
    """Class for handling collision energy (CE) embedding."""

    def _ce_init(
        self,
        int_embedder: str,
        ce_insert_location: str,
        ce_insert_type: str,
        ce_insert_merge: bool,
        ce_insert_size: int,
        ce_mean: float,
        ce_std: float,
        ce_max: float,
    ) -> None:
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
        assert ce_insert_type in ["id", "lin", "embed", "embed_clamped", "bin"]
        assert ce_insert_location in ["none", "mol", "frag", "mlp"]
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

    def _ce_location_check(self) -> None:
        raise NotImplementedError("Subclasses must implement _ce_location_check")

    def _setup_ce(self) -> None:
        """Setup CE embedding and transformation logic based on type and location."""
        # Setup embedding type
        if self.ce_insert_type == "id":

            def ce_transform(ce: th.Tensor) -> th.Tensor:
                ce = transform_ce(ce, self.ce_mean, self.ce_std)
                ce = ce.reshape(-1, 1)
                ce = th.repeat_interleave(ce, self.ce_insert_size, dim=1)
                return ce

            ce_embedder = nn.Identity()
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "lin":

            def ce_transform(ce: th.Tensor) -> th.Tensor:
                ce = transform_ce(ce, self.ce_mean, self.ce_std)
                ce = ce.reshape(-1, 1)
                return ce

            ce_embedder = nn.Linear(1, self.ce_insert_size)
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "embed":

            def ce_transform(ce: th.Tensor) -> th.Tensor:
                ce = th.clamp(ce, min=-1)
                ce = th.round(ce, decimals=0).long()
                ce = ce.reshape(-1, 1)
                return ce

            embedder = get_embedder(
                self.int_embedder, max_count_int=int(self.ce_max) + 1
            )  # +1 for the shifted range
            ce_embedder = nn.Sequential(embedder, nn.Linear(embedder.num_dim, self.ce_insert_size))
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "embed_clamped":

            def ce_transform(ce: th.Tensor) -> th.Tensor:
                ce = th.clamp(ce, min=-1, max=int(self.ce_max) - 1)
                ce = th.round(ce, decimals=0).long()
                ce = ce.reshape(-1, 1)
                return ce

            embedder = get_embedder(self.int_embedder, max_count_int=int(self.ce_max))
            ce_embedder = nn.Sequential(embedder, nn.Linear(embedder.num_dim, self.ce_insert_size))
            ce_input_dim = self.ce_insert_size
        elif self.ce_insert_type == "bin":

            def ce_transform(ce: th.Tensor) -> th.Tensor:
                # Handle -1 (missing/unknown) by giving it its own bin (index 0)
                # Other values are binned normally then shifted up by 1
                ce = th.clamp(ce, min=-1, max=int(self.ce_max) - 10)
                # Create mask for -1 values
                missing_mask = ce == -1
                # For non-missing values: bin them and shift up by 1
                ce_binned = th.round(ce, decimals=-1).long() // 10
                ce_binned = th.where(
                    missing_mask,
                    th.zeros_like(ce_binned),
                    ce_binned + 1,
                )
                # One-hot encode with -1 getting bin 0, others getting bins 1+
                ce = F.one_hot(
                    ce_binned, num_classes=(int(self.ce_max) // 10) + 1
                ).float()  # +1 for the -1 bin
                return ce

            ce_embedder = nn.Linear(
                (int(self.ce_max) // 10) + 1, self.ce_insert_size
            )  # +1 for the -1 bin
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

    def embed_ce(
        self, ce: th.Tensor, ce_batch_idxs: th.Tensor, batch_size: int
    ) -> th.Tensor | None:
        """
        Embed the collision energy values.

        Args:
            ce: Tensor of collision energies.
            ce_batch_idxs: Batch indices for each CE value.
            batch_size: Total batch size.

        Returns:
            Embedded CE tensor or None.
        """
        ce_embed = None
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
                    include_self=False,
                )
        return ce_embed


class PrecModel:
    """Class for handling precursor type embedding."""

    def _prec_init(
        self, prec_insert_location: str, prec_insert_size: int, prec_num_types: int
    ) -> None:
        """
        Initialize precursor embedding configuration and embedding layers.

        Args:
            prec_insert_location: Where to insert precursor embedding ("none", "mol", "mlp").
            prec_insert_size: Output embedding size.
            prec_num_types: Number of precursor types.
        """
        self.prec_insert_location = prec_insert_location
        self.prec_embedder = nn.Embedding(prec_num_types + 1, prec_insert_size)
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

    def _prec_location_check(self) -> None:
        raise NotImplementedError("Subclasses must implement _prec_location_check")

    def embed_prec(self, prec_type: th.Tensor | None) -> th.Tensor | None:
        """
        Embed the precursor type.

        Args:
            prec_type: Tensor of precursor type indices.

        Returns:
            Embedded precursor tensor or None.
        """
        prec_embed = None
        if self.prec_insert_location != "none":
            if prec_type is None:
                raise ValueError(
                    f"prec_type is None but prec_insert_location is {self.prec_insert_location}"
                )
            prec_embed = self.prec_embedder(prec_type)
        return prec_embed


class InstModel:
    """Class for handling instrument type embedding."""

    def _inst_init(
        self, inst_insert_location: str, inst_insert_size: int, inst_num_types: int
    ) -> None:
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

    def _inst_location_check(self) -> None:
        raise NotImplementedError("Subclasses must implement _inst_location_check")

    def embed_inst(self, inst_type: th.Tensor | None) -> th.Tensor | None:
        """
        Embed the instrument type.

        Args:
            inst_type: Tensor of instrument type indices.

        Returns:
            Embedded instrument tensor or None.
        """
        inst_embed = None
        if self.inst_insert_location != "none":
            if inst_type is None:
                raise ValueError(
                    f"inst_type is None but inst_insert_location is {self.inst_insert_location}"
                )
            inst_embed = self.inst_embedder(inst_type)
        return inst_embed


class FragModeModel:
    """Class for handling fragmentation mode embedding (e.g. HCD, CID, unknown)."""

    def _frag_mode_init(
        self,
        frag_mode_insert_location: str,
        frag_mode_insert_size: int,
        frag_mode_num_types: int,
    ) -> None:
        """
        Initialize fragmentation mode embedding configuration and embedding layers.

        Args:
            frag_mode_insert_location: Where to insert frag-mode embedding ("none", "mol", "mlp").
            frag_mode_insert_size: Output embedding size.
            frag_mode_num_types: Number of known fragmentation mode types (unknown handled as +1).
        """
        self.frag_mode_insert_location = frag_mode_insert_location
        # +1 for the unknown/fallback token at index frag_mode_num_types
        self.frag_mode_embedder = nn.Embedding(frag_mode_num_types + 1, frag_mode_insert_size)
        frag_mode_dim = frag_mode_insert_size

        self._frag_mode_location_check()

        if self.frag_mode_insert_location == "mol":
            frag_mode_mol_input_dim = frag_mode_dim
            frag_mode_mlp_input_dim = 0
        elif self.frag_mode_insert_location == "mlp":
            frag_mode_mol_input_dim = 0
            frag_mode_mlp_input_dim = frag_mode_dim
        else:
            assert self.frag_mode_insert_location == "none"
            frag_mode_mol_input_dim = 0
            frag_mode_mlp_input_dim = 0
        self.frag_mode_mol_input_dim = frag_mode_mol_input_dim
        self.frag_mode_mlp_input_dim = frag_mode_mlp_input_dim

    def _frag_mode_location_check(self) -> None:
        raise NotImplementedError("Subclasses must implement _frag_mode_location_check")

    def embed_frag_mode(self, frag_mode: th.Tensor | None) -> th.Tensor | None:
        """
        Embed the fragmentation mode.

        Args:
            frag_mode: Tensor of fragmentation mode indices (HCD=0, CID=1, unknown=num_types).

        Returns:
            Embedded frag-mode tensor or None.
        """
        frag_mode_embed = None
        if self.frag_mode_insert_location != "none":
            if frag_mode is None:
                raise ValueError(
                    f"frag_mode is None but frag_mode_insert_location is "
                    f"{self.frag_mode_insert_location}"
                )
            frag_mode_embed = self.frag_mode_embedder(frag_mode)
        return frag_mode_embed


class CEScaler(nn.Module):
    """CE-conditioned FiLM output scaling layer.

    Treats collision energy like a temperature: predicts a per-H-shift (log_scale, bias)
    from normalized CE via a small MLP, then applies:
        ``logits_out = logits * exp(log_scale) + bias``
    before the scatter softmax. Higher CE can increase certain H-shift dimensions
    (e.g. Δ-2) while suppressing others, capturing the physics of CE-dependent
    fragment appearance / disappearance.

    Initialized to zero so the layer starts as an identity transform.

    Attributes:
        ce_mean: Mean used for CE z-score normalization.
        ce_std: Std used for CE z-score normalization.
        output_dim: Number of H-shift logit dimensions (2*num_hs+1, or doubled for CMF).
        net: MLP mapping normalized CE scalar → (log_scale, bias) of shape (output_dim*2,).
    """

    def __init__(
        self,
        ce_mean: float,
        ce_std: float,
        output_dim: int,
        hidden_dim: int = 64,
    ) -> None:
        """Initialize CEScaler.

        Args:
            ce_mean: Mean CE for z-score normalization.
            ce_std: Std CE for z-score normalization.
            output_dim: Logit output dimension (mlp_output_dim).
            hidden_dim: Hidden dim of the CE→FiLM MLP. Defaults to 64.
        """
        super().__init__()
        self.ce_mean = ce_mean
        self.ce_std = ce_std
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim * 2),
        )
        # Init to zero → identity transform at the start of training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        logits: th.Tensor,
        ce: th.Tensor,
        ce_batch_idxs: th.Tensor,
        frag_node_batch_idxs: th.Tensor,
        batch_size: int,
    ) -> th.Tensor:
        """Apply CE-conditioned affine scaling to intensity logits.

        Args:
            logits: Raw MLP logits of shape (num_nodes, output_dim).
            ce: Per-CE values of shape (num_ces,); -1 means missing.
            ce_batch_idxs: Batch index per CE of shape (num_ces,).
            frag_node_batch_idxs: Batch index per node of shape (num_nodes,).
            batch_size: Number of samples in the batch.

        Returns:
            Scaled logits of shape (num_nodes, output_dim).
        """
        # Normalize CE; treat missing values (-1) as 0 so identity scaling is preserved
        mask = ce >= 0
        norm_ce = th.where(mask, (ce - self.ce_mean) / self.ce_std, th.zeros_like(ce))
        norm_ce = norm_ce.reshape(-1, 1)
        # MLP: (num_ces, 1) → (num_ces, output_dim * 2)
        params = self.net(norm_ce)
        # Aggregate multiple CEs per sample via mean
        params_per_sample = scatter_reduce(
            src=params,
            index=ce_batch_idxs.unsqueeze(1).expand_as(params),
            reduce="mean",
            dim_size=batch_size,
            include_self=False,
        )  # (batch_size, output_dim * 2)
        params_per_node = params_per_sample[frag_node_batch_idxs]  # (num_nodes, output_dim * 2)
        log_scale = params_per_node[:, : self.output_dim]
        bias = params_per_node[:, self.output_dim :]
        return logits * th.exp(log_scale) + bias


class FragModeScaler(nn.Module):
    """Mode-conditioned affine output scaling layer.

    Applies a per-fragmentation-mode learned scale and bias to the raw logit output of the
    intensity head, allowing the model to shift its predicted intensity distribution based
    on fragmentation physics (HCD vs CID vs unknown).

    The transform is: ``logits_out = logits * exp(log_scale) + bias``

    Both ``log_scale`` and ``bias`` are initialised to zero so the layer starts as the
    identity transform and can be trained from scratch or fine-tuned.

    Attributes:
        params: Embedding of shape (num_frag_modes + 1, 2) storing (log_scale, bias)
            for each mode.  Index ``num_frag_modes`` is the fallback unknown token.
    """

    def __init__(self, num_frag_modes: int, output_dim: int) -> None:
        """
        Initialize the FragModeScaler.

        Args:
            num_frag_modes: Number of known fragmentation modes (unknown handled as +1).
            output_dim: Dimension of the logit output to scale (mlp_output_dim).
        """
        super().__init__()
        self.output_dim = output_dim
        # shape: (num_frag_modes + 1, output_dim * 2) — first half is log_scale, second is bias
        # Initialise to zero so the layer starts as an identity transform.
        self.params = nn.Embedding(num_frag_modes + 1, output_dim * 2)
        nn.init.zeros_(self.params.weight)

    def forward(
        self,
        logits: th.Tensor,
        frag_mode: th.Tensor,
        node_batch_idxs: th.Tensor,
    ) -> th.Tensor:
        """
        Apply per-mode affine scaling to the intensity logits.

        Args:
            logits: Raw MLP logits of shape (num_nodes, output_dim).
            frag_mode: Per-sample mode indices of shape (batch_size,).
            node_batch_idxs: Batch index for each node of shape (num_nodes,).

        Returns:
            Scaled logits of shape (num_nodes, output_dim).
        """
        # mode_params: (batch_size, output_dim * 2)
        mode_params = self.params(frag_mode)
        # expand to per-node: (num_nodes, output_dim * 2)
        mode_params_per_node = mode_params[node_batch_idxs]
        log_scale = mode_params_per_node[:, : self.output_dim]
        bias = mode_params_per_node[:, self.output_dim :]
        return logits * th.exp(log_scale) + bias
