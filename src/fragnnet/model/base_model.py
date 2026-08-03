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
        use_nce: bool,
        nce_mean: float,
        nce_std: float,
        nce_max: float,
        use_ace: bool,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
    ) -> None:
        """
        Initialize CE embedding configuration and embedding layers.

        Args:
            int_embedder: Embedder for integer features.
            ce_insert_location: Where to insert CE embedding ("none", "mol", "frag", "mlp").
            ce_insert_type: Type of CE embedding ("id", "lin", "embed", "bin").
            ce_insert_merge: Whether to merge CE embeddings across batch.
            ce_insert_size: Output embedding size.
            nce_mean: Mean for NCE normalization.
            nce_std: Std for NCE normalization.
            nce_max: Max NCE value for binning/embedding.
            use_nce: Embed NCE signal. Default True.
            use_ace: Embed ACE signal. Default False. When both use_nce and use_ace
                are True the two embeddings are concatenated, doubling the CE dim.
            ace_max: Max ACE value (eV) for binning/embedding. Default 100.0.
            ace_mean: Mean ACE value (eV) for normalization (lin/id types). Default 20.0.
            ace_std: Std of ACE for normalization (lin/id types). Default 15.0.
        """
        assert ce_insert_type in ["id", "lin", "embed", "embed_clamped", "bin"]
        # ce_insert_location may name multiple insertion points, comma-separated (e.g.
        # "mol,mlp"): inject CE into the molecule encoder input AND the per-fragment scoring
        # MLP. "mol" alone (mol-encoder only) or "mlp" alone (per-fragment only) are the
        # single-location cases; "none" disables CE. Parsed into a set so downstream checks
        # are membership tests instead of exact string matches.
        if ce_insert_location == "none":
            self.ce_insert_locations: set[str] = set()
        else:
            self.ce_insert_locations = {loc.strip() for loc in ce_insert_location.split(",")}
            invalid = self.ce_insert_locations - {"mol", "frag", "mlp"}
            if invalid:
                raise ValueError(
                    f"Invalid ce_insert_location(s) {invalid}; allowed: mol, frag, mlp (or 'none')"
                )
        if self.ce_insert_locations and not use_nce and not use_ace:
            raise ValueError("At least one of use_nce or use_ace must be True")
        self.ce_insert_type = ce_insert_type
        self.ce_insert_location = ce_insert_location
        self.ce_insert_merge = ce_insert_merge
        self.ce_insert_size = ce_insert_size
        self.int_embedder = int_embedder
        self.nce_max = nce_max
        self.nce_mean = nce_mean
        self.nce_std = nce_std
        self.use_nce = use_nce
        self.use_ace = use_ace
        self.ace_max = ace_max
        self.ace_mean = ace_mean
        self.ace_std = ace_std
        self._ce_location_check()
        self._setup_ce()

    def _ce_location_check(self) -> None:
        raise NotImplementedError("Subclasses must implement _ce_location_check")

    def _build_ce_components(self, max_val: float, mean_val: float, std_val: float) -> tuple:
        """Build transform function and embedder module for a given CE scale.

        Args:
            max_val: Maximum CE value for clamping/binning.
            mean_val: Mean for normalization (lin/id types).
            std_val: Std for normalization (lin/id types).

        Returns:
            Tuple of (transform_fn, embedder_module, output_dim).
        """
        insert_size = self.ce_insert_size
        if self.ce_insert_type == "id":

            def transform(ce: th.Tensor) -> th.Tensor:
                ce = transform_ce(ce, mean_val, std_val)
                ce = ce.reshape(-1, 1)
                return th.repeat_interleave(ce, insert_size, dim=1)

            embedder = nn.Identity()
        elif self.ce_insert_type == "lin":

            def transform(ce: th.Tensor) -> th.Tensor:
                ce = transform_ce(ce, mean_val, std_val)
                return ce.reshape(-1, 1)

            embedder = nn.Linear(1, insert_size)
        elif self.ce_insert_type == "embed":

            def transform(ce: th.Tensor) -> th.Tensor:
                ce = th.clamp(ce, min=-1, max=int(max_val))
                ce = th.round(ce, decimals=0).long()
                return ce.reshape(-1, 1)

            raw_emb = get_embedder(self.int_embedder, max_count_int=int(max_val) + 1)
            embedder = nn.Sequential(raw_emb, nn.Linear(raw_emb.num_dim, insert_size))
        elif self.ce_insert_type == "embed_clamped":

            def transform(ce: th.Tensor) -> th.Tensor:
                ce = th.clamp(ce, min=-1, max=int(max_val) - 1)
                ce = th.round(ce, decimals=0).long()
                return ce.reshape(-1, 1)

            raw_emb = get_embedder(self.int_embedder, max_count_int=int(max_val))
            embedder = nn.Sequential(raw_emb, nn.Linear(raw_emb.num_dim, insert_size))
        elif self.ce_insert_type == "bin":

            def transform(ce: th.Tensor) -> th.Tensor:
                ce = th.clamp(ce, min=-1, max=int(max_val) - 10)
                missing_mask = ce == -1
                ce_binned = th.round(ce, decimals=-1).long() // 10
                ce_binned = th.where(missing_mask, th.zeros_like(ce_binned), ce_binned + 1)
                return F.one_hot(ce_binned, num_classes=(int(max_val) // 10) + 1).float()

            embedder = nn.Linear((int(max_val) // 10) + 1, insert_size)
        return transform, embedder, insert_size

    def _setup_ce(self) -> None:
        """Setup CE embedding and transformation logic based on type and location."""
        effective_dim = 0
        if self.use_nce:
            nce_transform, nce_embedder, ce_input_dim = self._build_ce_components(
                self.nce_max, self.nce_mean, self.nce_std
            )
            self.ce_transform = nce_transform
            self.ce_embedder = nce_embedder
            effective_dim += ce_input_dim
        if self.use_ace:
            ace_transform, ace_embedder, ace_input_dim = self._build_ce_components(
                self.ace_max, self.ace_mean, self.ace_std
            )
            self.ace_transform = ace_transform
            self.ce_embedder_ace = ace_embedder
            effective_dim += ace_input_dim

        self.ce_embed_dim = effective_dim

        # Independent dims: with "mol,mlp" both are nonzero, so CE widens the mol-encoder
        # input AND the per-fragment MLP input.
        self.ce_mol_input_dim = effective_dim if "mol" in self.ce_insert_locations else 0
        self.ce_mlp_input_dim = effective_dim if "mlp" in self.ce_insert_locations else 0

    def _embed_single_ce(
        self,
        ce: th.Tensor,
        ce_batch_idxs: th.Tensor,
        batch_size: int,
        transform,
        embedder: nn.Module,
    ) -> th.Tensor:
        """Embed a single CE tensor using the given transform and embedder.

        Args:
            ce: CE values tensor.
            ce_batch_idxs: Batch indices for each CE value.
            batch_size: Total batch size.
            transform: Transform function to apply before embedding.
            embedder: Embedding module.

        Returns:
            Embedded CE tensor of shape (batch_size, ce_insert_size).
        """
        ce_embed = transform(ce)
        ce_embed = embedder(ce_embed)
        if self.ce_insert_merge:
            ce_embed = scatter_reduce(
                src=ce_embed,
                index=ce_batch_idxs.unsqueeze(1).expand_as(ce_embed),
                reduce="mean",
                dim_size=batch_size,
                include_self=False,
            )
        return ce_embed

    def embed_ce(
        self,
        ce: th.Tensor,
        ce_batch_idxs: th.Tensor,
        batch_size: int,
        ace: th.Tensor | None = None,
        ace_batch_idxs: th.Tensor | None = None,
    ) -> th.Tensor | None:
        """Embed the collision energy values.

        Embeds each active CE signal (NCE, ACE) independently and concatenates them.
        When both use_nce and use_ace are True the output dimension doubles.

        Args:
            ce: NCE tensor of collision energies. Required when use_nce is True.
            ce_batch_idxs: Batch indices for each NCE value.
            batch_size: Total batch size.
            ace: ACE tensor (eV). Required when use_ace is True.
            ace_batch_idxs: Batch indices for each ACE value. Required when use_ace is True.

        Returns:
            Embedded CE tensor or None if no insertion location is set ("none").
        """
        if not self.ce_insert_locations:
            return None
        embeds = []
        if self.use_nce:
            embeds.append(
                self._embed_single_ce(
                    ce, ce_batch_idxs, batch_size, self.ce_transform, self.ce_embedder
                )
            )
        if self.use_ace:
            assert ace is not None and ace_batch_idxs is not None, (
                "ace and ace_batch_idxs required when use_ace is True"
            )
            embeds.append(
                self._embed_single_ce(
                    ace, ace_batch_idxs, batch_size, self.ace_transform, self.ce_embedder_ace
                )
            )
        if len(embeds) == 1:
            return embeds[0]
        return th.cat(embeds, dim=1)


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

    Each active CE signal (NCE and/or ACE) contributes a ``(mean, std, valid)`` triple per
    sample, where ``std`` is zero for single-CE spectra and positive for ramped/stepped ones.
    ``valid`` is the mean valid bit across CE entries. Missing signals (valid=0) produce
    all-zero features. Initialized to zero so the layer starts as an identity transform.

    Attributes:
        use_nce: Whether NCE signal is active.
        use_ace: Whether ACE signal is active.
        output_dim: Number of H-shift logit dimensions (2*num_hs+1, or doubled for CMF).
        net: MLP mapping normalized CE vector → (log_scale, bias) of shape (output_dim*2,).
    """

    def __init__(
        self,
        nce_mean: float,
        nce_std: float,
        output_dim: int,
        hidden_dim: int,
        use_nce: bool,
        use_ace: bool,
        ace_mean: float,
        ace_std: float,
    ) -> None:
        """Initialize CEScaler.

        Args:
            nce_mean: Mean NCE (%) for z-score normalization.
            nce_std: Std NCE (%) for z-score normalization.
            output_dim: Logit output dimension (mlp_output_dim).
            hidden_dim: Hidden dim of the CE→FiLM MLP. Defaults to 64.
            use_nce: Whether to include NCE as a scaler input. Defaults to True.
            use_ace: Whether to include ACE as a scaler input. Defaults to False.
            ace_mean: Mean ACE (eV) for z-score normalization.
            ace_std: Std ACE (eV) for z-score normalization.
        """
        super().__init__()
        if not use_nce and not use_ace:
            raise ValueError("At least one of use_nce or use_ace must be True")
        self.use_nce = use_nce
        self.use_ace = use_ace
        self.nce_mean = nce_mean
        self.nce_std = nce_std
        self.ace_mean = ace_mean
        self.ace_std = ace_std
        self.output_dim = output_dim
        # Each active signal contributes (mean, std, valid) per sample, so 3 dims per signal.
        # std=0 → single-CE; std>0 → ramped or stepped spectrum.
        input_dim = 3 * (int(use_nce) + int(use_ace))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim * 2),
        )
        # Init to zero → identity transform at the start of training
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        logits: th.Tensor,
        frag_node_batch_idxs: th.Tensor,
        nce_stats: th.Tensor | None = None,
        ace_stats: th.Tensor | None = None,
    ) -> th.Tensor:
        """Apply CE-conditioned affine scaling to intensity logits.

        Args:
            logits: Raw MLP logits of shape (num_nodes, output_dim).
            frag_node_batch_idxs: Batch index per node of shape (num_nodes,).
            nce_stats: Pre-reduced NCE stats of shape (batch_size, 3): columns are
                (mean_raw, std_raw, valid_flag). Required when use_nce is True.
            ace_stats: Pre-reduced ACE stats of shape (batch_size, 3): columns are
                (mean_raw, std_raw, valid_flag). Required when use_ace is True.

        Returns:
            Scaled logits of shape (num_nodes, output_dim).
        """
        batch_feats = []
        if self.use_nce:
            assert nce_stats is not None, "nce_stats required when use_nce is True"
            mean, std, valid = nce_stats.to(logits.dtype).unbind(dim=1)
            norm_mean = valid * (mean - self.nce_mean) / self.nce_std
            norm_std = std / self.nce_std
            batch_feats.append(th.stack([norm_mean, norm_std, valid], dim=1))
        if self.use_ace:
            assert ace_stats is not None, "ace_stats required when use_ace is True"
            mean, std, valid = ace_stats.to(logits.dtype).unbind(dim=1)
            norm_mean = valid * (mean - self.ace_mean) / self.ace_std
            norm_std = std / self.ace_std
            batch_feats.append(th.stack([norm_mean, norm_std, valid], dim=1))
        # (batch_size, 3*n_signals) → MLP → (batch_size, output_dim * 2)
        params = self.net(th.cat(batch_feats, dim=1))
        params_per_node = params[frag_node_batch_idxs]  # (num_nodes, output_dim * 2)
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


class AdductScaler(nn.Module):
    """Adduct-type-conditioned FiLM output scaling layer.

    Applies a per-adduct-type learned scale and bias to the raw logit output of the
    intensity head, allowing the model to correct systematic intensity shifts for
    rare adducts (``[M+Na]+``, ``[M+Cl]-``, ``[M-H]-``) relative to ``[M+H]+``.

    The transform is: ``logits_out = logits * exp(log_scale) + bias``

    Both ``log_scale`` and ``bias`` are initialised to zero so the layer starts as the
    identity transform.

    Attributes:
        params: Embedding of shape ``(num_adduct_types + 1, output_dim * 2)`` storing
            ``(log_scale, bias)`` for each adduct.  Index ``num_adduct_types`` is the
            fallback unknown token.
    """

    def __init__(self, num_adduct_types: int, output_dim: int) -> None:
        """Initialize the AdductScaler.

        Args:
            num_adduct_types: Number of known adduct/precursor types (unknown handled as +1).
            output_dim: Dimension of the logit output to scale (mlp_output_dim).
        """
        super().__init__()
        self.output_dim = output_dim
        self.params = nn.Embedding(num_adduct_types + 1, output_dim * 2)
        nn.init.zeros_(self.params.weight)

    def forward(
        self,
        logits: th.Tensor,
        prec_type: th.Tensor,
        node_batch_idxs: th.Tensor,
    ) -> th.Tensor:
        """Apply per-adduct affine scaling to the intensity logits.

        Args:
            logits: Raw MLP logits of shape (num_nodes, output_dim).
            prec_type: Per-sample adduct indices of shape (batch_size,).
            node_batch_idxs: Batch index for each node of shape (num_nodes,).

        Returns:
            Scaled logits of shape (num_nodes, output_dim).
        """
        adduct_params = self.params(prec_type)
        adduct_params_per_node = adduct_params[node_batch_idxs]
        log_scale = adduct_params_per_node[:, : self.output_dim]
        bias = adduct_params_per_node[:, self.output_dim :]
        return logits * th.exp(log_scale) + bias


class StratumScaler(nn.Module):
    """Combined acquisition-stratum FiLM output scaling layer.

    Conditions on instrument type, fragmentation mode, and adduct type simultaneously.
    Each categorical axis has its own small embedding table; the three embeddings are
    summed and projected to per-H-shift ``(log_scale, bias)`` pairs.

    The transform is: ``logits_out = logits * exp(log_scale) + bias``

    The projection is initialised to zero so the layer starts as the identity transform.

    Attributes:
        inst_emb: Embedding for instrument type, shape ``(num_inst_types + 1, embed_dim)``.
        frag_mode_emb: Embedding for fragmentation mode, shape ``(num_frag_modes + 1, embed_dim)``.
        prec_emb: Embedding for adduct/precursor type, shape ``(num_adduct_types + 1, embed_dim)``.
        proj: Linear projection from ``embed_dim`` to ``output_dim * 2``.
    """

    def __init__(
        self,
        num_inst_types: int,
        num_frag_modes: int,
        num_adduct_types: int,
        embed_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize the StratumScaler.

        Args:
            num_inst_types: Number of known instrument types (unknown handled as +1).
            num_frag_modes: Number of known fragmentation modes (unknown handled as +1).
            num_adduct_types: Number of known adduct/precursor types (unknown handled as +1).
            embed_dim: Dimension of each categorical embedding before projection.
            output_dim: Dimension of the logit output to scale (mlp_output_dim).
        """
        super().__init__()
        self.output_dim = output_dim
        self.inst_emb = nn.Embedding(num_inst_types + 1, embed_dim)
        self.frag_mode_emb = nn.Embedding(num_frag_modes + 1, embed_dim)
        self.prec_emb = nn.Embedding(num_adduct_types + 1, embed_dim)
        self.proj = nn.Linear(embed_dim, output_dim * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        logits: th.Tensor,
        inst_type: th.Tensor,
        frag_mode: th.Tensor,
        prec_type: th.Tensor,
        node_batch_idxs: th.Tensor,
    ) -> th.Tensor:
        """Apply per-stratum affine scaling to the intensity logits.

        Args:
            logits: Raw MLP logits of shape (num_nodes, output_dim).
            inst_type: Per-sample instrument indices of shape (batch_size,).
            frag_mode: Per-sample fragmentation mode indices of shape (batch_size,).
            prec_type: Per-sample adduct indices of shape (batch_size,).
            node_batch_idxs: Batch index for each node of shape (num_nodes,).

        Returns:
            Scaled logits of shape (num_nodes, output_dim).
        """
        stratum = (
            self.inst_emb(inst_type)
            + self.frag_mode_emb(frag_mode)
            + self.prec_emb(prec_type)
        )  # (batch_size, embed_dim)
        params = self.proj(stratum)  # (batch_size, output_dim * 2)
        params_per_node = params[node_batch_idxs]
        log_scale = params_per_node[:, : self.output_dim]
        bias = params_per_node[:, self.output_dim :]
        return logits * th.exp(log_scale) + bias


class ContextEmbedding(nn.Module):
    """Combine acquisition-metadata embeddings into one learned context (stratum) vector.

    FraGNNet conditions on CE, adduct, instrument, and fragmentation mode through four separate
    ``*_insert`` embeddings that are concatenated into node features, plus a chain of
    rescale-only output-FiLM scalers (see ``docs/plans/conditioning_refactor_plan.md``). This module
    consolidates the metadata into a single vector, projected through an MLP so cross-metadata
    interactions (e.g. CE x mode, adduct x CE) are modeled in a shared low-dimensional space
    *before* injection -- unlike plain concatenation, which defers all interaction to the
    downstream network and produces a variable-width block. The one context vector is meant to
    be injected consistently at the mol encoder, the per-fragment nodeMLP, and the competition
    module, replacing the four separate insert paths.

    This is a self-contained module: it does NOT wire itself into the model. The forward model
    keeps computing the individual ``ce_embed``/``prec_embed``/``inst_embed``/``frag_mode_embed``
    and passes whichever are active (in a fixed order matching ``input_dim``) to this module.

    Attributes:
        input_dim: Sum of the active metadata embedding dims (fixed at construction).
        out_dim: Dimension of the produced context vector.
        net: MLP mapping the concatenated metadata embeddings to the context vector.
    """

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize ContextEmbedding.

        Args:
            input_dim: Total width of the concatenated active metadata embeddings. Must equal
                the summed dim of the tensors passed to ``forward``.
            hidden_dim: Hidden width of the combining MLP.
            out_dim: Dimension of the output context vector.
        """
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"ContextEmbedding input_dim must be positive, got {input_dim}")
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, embeds: list[th.Tensor]) -> th.Tensor:
        """Combine active metadata embeddings into a context vector.

        Args:
            embeds: Active per-spectrum metadata embeddings, each of shape (batch, d_k), in a
                fixed order; the summed ``d_k`` must equal ``input_dim``.

        Returns:
            Context vector of shape (batch, out_dim).

        Raises:
            ValueError: If ``embeds`` is empty or its concatenated width != ``input_dim``.
        """
        if not embeds:
            raise ValueError("ContextEmbedding.forward requires at least one metadata embedding")
        x = th.cat(embeds, dim=-1)
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"ContextEmbedding got concatenated width {x.shape[-1]}, expected {self.input_dim}"
            )
        return self.net(x)


class CutBondEncoder(nn.Module):
    """Encode a fragment's cut bonds from the (inside, outside) atom embeddings of each bond.

    Replaces the ``boundary_pair`` feature, which encodes each cut bond as
    ``Linear(cat(in, out)) == W_in @ in + W_out @ out + b`` and then mean-pools over the
    fragment's bonds. That construction is degenerate in two ways:

    1. The projection is additive, so a bond is only ever the *sum of two independent atom
       terms* -- "C-N bond" cannot differ from "C contribution + N contribution". Bond identity
       is inherently an interaction, so no linear map on a concatenation can express it. This
       holds even for a fragment with a single cut bond.
    2. Linear commutes through mean/sum pooling, so for >= 2 cut bonds the pooled result equals
       ``Linear(mean(inside_atoms), mean(outside_atoms))`` -- which atom was bonded to which is
       discarded exactly. Cutting a1-b1 + a2-b2 gives a bit-identical embedding to cutting
       a1-b2 + a2-b1. That is precisely the multi-cut ring-cleavage regime.

    This encoder fixes both: an explicit elementwise product ``in * out`` gives each bond a
    multiplicative signature, and the hidden nonlinearity stops the projection commuting through
    the pool, so the pairing survives. The (in, out) order is kept asymmetric because it is
    chemically meaningful -- inside atoms are retained by the fragment, outside atoms are lost.

    Pooling is left to the caller (the model reuses ``mol_pool``, as ``boundary_pair`` does) so
    swapping the two features is a clean one-variable comparison at identical width.

    Attributes:
        atom_dim: Width of a single atom embedding from the mol encoder.
        out_dim: Width of the per-bond encoding (matched to boundary_pair for comparability).
        net: MLP mapping [in, out, in * out] to the per-bond encoding.

    Example:
        >>> enc = CutBondEncoder(atom_dim=384, hidden_dim=256, out_dim=384)
        >>> bonds = enc(in_embed, out_embed)  # (num_cut_bonds, 384)
        >>> frag = mol_pool(bonds, bond_frag_idxs, dim_size=num_frags)
    """

    def __init__(self, atom_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize CutBondEncoder.

        Args:
            atom_dim: Width of one atom embedding from the mol encoder.
            hidden_dim: Hidden width of the bond MLP.
            out_dim: Width of the per-bond output encoding.

        Raises:
            ValueError: If any dimension is not positive.
        """
        super().__init__()
        for name, dim in (("atom_dim", atom_dim), ("hidden_dim", hidden_dim), ("out_dim", out_dim)):
            if dim <= 0:
                raise ValueError(f"CutBondEncoder {name} must be positive, got {dim}")
        self.atom_dim = atom_dim
        self.out_dim = out_dim
        # 3 * atom_dim: [in, out, in * out] -- the product is what makes the bond more than the
        # sum of its atoms; the hidden nonlinearity is what stops pooling collapsing the pairing.
        self.net = nn.Sequential(
            nn.Linear(3 * atom_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, in_embed: th.Tensor, out_embed: th.Tensor) -> th.Tensor:
        """Encode each cut bond from its inside/outside atom embeddings.

        Args:
            in_embed: Inside-atom (retained) embeddings of shape (num_cut_bonds, atom_dim).
            out_embed: Outside-atom (lost) embeddings of shape (num_cut_bonds, atom_dim).

        Returns:
            Per-bond encoding of shape (num_cut_bonds, out_dim). Pool over the fragment's bonds
            to obtain a per-fragment feature.

        Raises:
            ValueError: If the two inputs disagree in shape or are not atom_dim wide.
        """
        if in_embed.shape != out_embed.shape:
            raise ValueError(
                f"in_embed {tuple(in_embed.shape)} and out_embed {tuple(out_embed.shape)} "
                "must have the same shape"
            )
        if in_embed.shape[-1] != self.atom_dim:
            raise ValueError(
                f"CutBondEncoder got atom width {in_embed.shape[-1]}, expected {self.atom_dim}"
            )
        return self.net(th.cat([in_embed, out_embed, in_embed * out_embed], dim=-1))
