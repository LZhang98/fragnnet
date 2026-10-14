"""Metadata embedding mixins: collision energy (CE), precursor type (Prec), instrument type
(Inst) and fragmentation mode (FragMode), shared by the models that condition on them (was
``model/base_model.py``).

- CEModel: Handles collision energy embedding and transformation.
- QETEnergyEncoder: Smooth CE embedding on the breakdown-curve axis 1 / (CE + offset).
- PrecModel: Handles precursor type embedding.
- InstModel: Handles instrument type embedding.
- FragModeModel: Handles fragmentation mode embedding (HCD/CID/unknown).

FraGNNet's output scalers and context embedding are in models/fragnnet/fragnnet_conditioning.py.
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.layers.int_featurizer import get_embedder
from fragnnet.utils.scatter_utils import scatter_reduce
from fragnnet.utils.spec_utils import transform_ce, weighted_mean_by_batch

# Offset of the breakdown-curve energy axis 1 / (CE + offset). Fitting the QET/Arrhenius form
# softmax(a_i - e_i * beta(NCE)) to NIST20 [M+H]+/FT CE series put the shared curve at
# 1 / (NCE + n0) with n0 near 200 (docs/findings/breakdown_curve_form_fit_2026_09_26.md). It is a
# shape parameter, not a calibrated internal energy, and it was measured on NCE only.
QET_ENERGY_OFFSET = 200.0
# The ACE (eV) path is linear. The same form fitted to NIST20 [M+H]+/QTOF eV series (1-50 eV) fits
# best with no offset, and a held-out interior energy is predicted 0.0009 better (paired SE 0.0001)
# than with eV converted to NCE and put on 1 / (NCE + 200)
# (docs/findings/masster/masster_qet_projection_candidate_union_2026_09_26.md).
QET_ACE_OFFSET = None


class QETEnergyEncoder(nn.Module):
    """Embed collision energies on the breakdown-curve axis, with a missing-energy flag.

    Each reading becomes two features: z, a standardized 1 / (CE + offset), and a flag that is 1
    for a missing CE (any value below 0), whose z is set to 0. A linear layer maps them to
    ``out_dim`` (``ce_insert_type: qet``), or with ``hidden_dim`` a small MLP,
    Linear(2, hidden_dim) -> SiLU -> Linear(hidden_dim, out_dim) (``qet_mlp``). z is scaled so
    that z ~ (CE - mean) / std near ``mean``: it rises with CE, is smooth everywhere, and
    extrapolates past the training range instead of saturating. With ``offset=None`` the axis is
    linear, z = (CE - mean) / std, as the ACE (eV) path uses.

    Integer-table CE embeddings (``ce_insert_type: embed``) give neighbouring energies unrelated
    rows, so an energy unseen in training gets arbitrary features, and the default abs-sines table
    aliases CE x with 2**k - x. Among the encodings compared on held-out NCE values in
    docs/findings/masster/masster_qet_projection_candidate_union_2026_09_26.md, this axis
    generalized best.

    Args:
        mean: CE the standardization centres on (the configured nce_mean or ace_mean).
        std: CE spread setting the scale of z near ``mean``.
        out_dim: Width of the embedding.
        offset: Offset of the 1 / (CE + offset) axis, or None for a linear axis.
        hidden_dim: Hidden width of the MLP, or None for the single linear layer.

    Raises:
        ValueError: If ``std`` is not positive, ``mean + offset`` is not positive, or ``out_dim``
            or ``hidden_dim`` is not positive.
    """

    def __init__(
        self,
        mean: float,
        std: float,
        out_dim: int,
        offset: float | None = QET_ENERGY_OFFSET,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if std <= 0.0 or out_dim <= 0 or (offset is not None and mean + offset <= 0.0):
            raise ValueError("QETEnergyEncoder needs std > 0, mean + offset > 0 and out_dim > 0")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("QETEnergyEncoder needs hidden_dim > 0")
        self.offset = None if offset is None else float(offset)
        self.mean = float(mean)
        self.std = float(std)
        if offset is not None:
            # z = (1 / (mean + offset) - 1 / (CE + offset)) * (mean + offset)**2 / std, whose
            # slope at CE = mean is 1 / std.
            self.center = 1.0 / (mean + offset)
            self.scale = std / (mean + offset) ** 2
        # The first layer keeps the name "linear" in both forms, so "qet" checkpoints still load.
        self.linear = nn.Linear(2, out_dim if hidden_dim is None else hidden_dim)
        self.output = None if hidden_dim is None else nn.Linear(hidden_dim, out_dim)

    def forward(self, ce: th.Tensor) -> th.Tensor:
        """Embed collision-energy readings.

        Args:
            ce: Collision energies of any shape; values below 0 mark a missing CE.

        Returns:
            Embeddings, shape ``(ce.numel(), out_dim)``.
        """
        ce = ce.reshape(-1).to(self.linear.weight.dtype)
        missing = ce < 0
        if self.offset is None:
            z = (ce - self.mean) / self.std
        else:
            z = (self.center - 1.0 / (ce.clamp_min(0.0) + self.offset)) / self.scale
        z = th.where(missing, th.zeros_like(z), z)
        hidden = self.linear(th.stack((z, missing.to(z.dtype)), dim=1))
        return hidden if self.output is None else self.output(F.silu(hidden))


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
            ce_insert_type: Type of CE embedding ("id", "lin", "embed", "embed_clamped", "bin",
                "qet", "qet_mlp"). "embed" and "embed_clamped" look the rounded CE up in the
                ``int_embedder`` table; "qet" embeds the raw CE with ``QETEnergyEncoder``,
                smooth in energy and with a learned missing-CE flag, and "qet_mlp" does the same
                through a hidden layer of width ``ce_insert_size``.
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
        assert ce_insert_type in ["id", "lin", "embed", "embed_clamped", "bin", "qet", "qet_mlp"]
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

    def _build_ce_components(
        self,
        max_val: float,
        mean_val: float,
        std_val: float,
        qet_offset: float | None = QET_ENERGY_OFFSET,
    ) -> tuple:
        """Build transform function and embedder module for a given CE scale.

        Args:
            max_val: Maximum CE value for clamping/binning.
            mean_val: Mean for normalization (lin/id/qet/qet_mlp types).
            std_val: Std for normalization (lin/id/qet/qet_mlp types).
            qet_offset: Offset of the "qet"/"qet_mlp" axis 1 / (CE + offset); None makes it
                linear, as the ACE path uses.

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
        elif self.ce_insert_type in ("qet", "qet_mlp"):

            def transform(ce: th.Tensor) -> th.Tensor:
                # Raw energies: no rounding, no clamp at max_val; below 0 stays missing.
                return ce.reshape(-1)

            hidden_dim = insert_size if self.ce_insert_type == "qet_mlp" else None
            embedder = QETEnergyEncoder(
                mean_val, std_val, insert_size, offset=qet_offset, hidden_dim=hidden_dim
            )
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
                self.ace_max, self.ace_mean, self.ace_std, qet_offset=QET_ACE_OFFSET
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
        weights: th.Tensor | None = None,
    ) -> th.Tensor:
        """Embed a single CE tensor using the given transform and embedder.

        Args:
            ce: CE values tensor.
            ce_batch_idxs: Batch indices for each CE value.
            batch_size: Total batch size.
            transform: Transform function to apply before embedding.
            embedder: Embedding module.
            weights: Optional per-reading weights for merged CE context.

        Returns:
            Embedded CE tensor of shape (batch_size, ce_insert_size).
        """
        ce_embed = transform(ce)
        ce_embed = embedder(ce_embed)
        if weights is not None:
            if not self.ce_insert_merge:
                raise ValueError("Weighted CE context requires ce_insert_merge=True")
            return weighted_mean_by_batch(ce_embed, weights, ce_batch_idxs, batch_size)
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
        ce_weights: th.Tensor | None = None,
        ace_weights: th.Tensor | None = None,
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
            ce_weights: Optional NCE reading weights.
            ace_weights: Optional ACE reading weights.

        Returns:
            Embedded CE tensor or None if no insertion location is set ("none").
        """
        if not self.ce_insert_locations:
            return None
        embeds = []
        if self.use_nce:
            embeds.append(
                self._embed_single_ce(
                    ce, ce_batch_idxs, batch_size, self.ce_transform, self.ce_embedder, ce_weights
                )
            )
        if self.use_ace:
            assert ace is not None and ace_batch_idxs is not None, (
                "ace and ace_batch_idxs required when use_ace is True"
            )
            embeds.append(
                self._embed_single_ce(
                    ace, ace_batch_idxs, batch_size, self.ace_transform, self.ce_embedder_ace,
                    ace_weights,
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
