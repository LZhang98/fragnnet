"""FraGNNet's output conditioning: affine scalers and the context embedding.

Only FraGNNet uses these (moved from ``model/base_model.py`` in reorganization Phase 8 step 3).

- CEScaler: CE-conditioned FiLM scaling of the per-H-shift logits.
- FragModeScaler: fragmentation-mode-conditioned FiLM output scaling.
- AdductScaler: adduct-type-conditioned FiLM output scaling.
- StratumScaler: FiLM output scaling conditioned on instrument, fragmentation mode and adduct.
- ContextEmbedding: combines the acquisition-metadata embeddings into one context vector.
"""

import torch as th
import torch.nn as nn


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
