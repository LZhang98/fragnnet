"""MASShatter: one-pass bond-set fragmentation spectrum prediction.

Unlike the DAG-based FraGNNet path, MASShatter never enumerates fragments ahead of
time. A single GNN pass over the intact precursor produces one or more Bernoulli
break-logit fields over the chemical bonds. Multiple fields are marginalized through
normalized molecule-level mixture weights, preserving a distribution over all
``2 ** num_bonds`` cut masks while allowing correlated cleavage regimes. Only a
probability-ranked prefix is materialized.
Each retained mask is cut into connected components, and every component is expanded
into the ion channels it could plausibly produce (charge retention vs charge
migration, plus +/- hydrogen transfers). The resulting sparse prediction is either a
legacy normalized log-probability spectrum or an independent nonnegative intensity spectrum.

The optional unique-fragment path treats masks as latent events: enumeration
continues until a target number of distinct atom sets is reached, charge retention is
normalized inside each mask, and duplicate component pathways are marginalized with
``logsumexp`` before intensity scoring.

The conditional probability of an enumerated pathway therefore factorizes as::

    P(mask | enumerated) * P(component keeps the charge | mask)
    * P(ion mode, H shift | component)

The final spectrum is normalized over the candidates generated for that precursor.
Unmatched observed peaks are candidate-coverage diagnostics, not a predicted class.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterCandidateBatch,
    MassShatterStaticGraph,
    build_mass_shatter_candidates,
)
from fragnnet.frag.mass_shatter_templates import (
    MASS_SHATTER_NUM_RULES,
    MassShatterTemplate,
    filter_mass_shatter_templates,
)
from fragnnet.model.bond_fragment_heads import (
    AtomChargeAffinityHead,
    BondFocusedBreakHead,
    CandidateIntensityTransformerHead,
    EndpointBondBreakHead,
    IonModeProposalHead,
    _segment_logsumexp,
    _validate_tensor_condition,
    component_charge_log_probabilities,
    latent_pathway_charge_log_probabilities,
    masked_ion_hydrogen_log_probabilities,
    pool_boundary_bond_embeddings,
    pool_component_and_complement_embeddings,
)
from fragnnet.model.form_embedder import get_embedder
from fragnnet.model.nn_blocks import GNN
from fragnnet.utils.formula_utils import (
    ELECTRON_MASS,
    H_MASS,
    PERIODIC_TABLE,
    PREC_TYPE_TO_CMF_FORMULA_DIFF,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_FORMULA_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    parse_formula,
)


def _mean_values_by_batch(
    values: th.Tensor,
    batch_indices: th.Tensor | None,
    batch_size: int,
) -> th.Tensor:
    """Average per-spectrum collision-energy values into one value per spectrum.

    A merged-CE spectrum carries several CE readings, packed as a ragged vector plus
    the spectrum index each reading belongs to. The context encoder needs a single
    scalar per spectrum, so the readings are mean-pooled by spectrum.

    Args:
        values: CE readings of shape ``(num_values,)`` (or ``(batch_size,)`` when
            ``batch_indices`` is None).
        batch_indices: Spectrum index per reading, shape ``(num_values,)``. None means
            the values are already one-per-spectrum and are returned unchanged.
        batch_size: Number of spectra in the batch.

    Returns:
        Mean CE value per spectrum, shape ``(batch_size,)``.

    Raises:
        ValueError: If the shapes/dtypes disagree, or a spectrum has no CE value.
    """
    values = values.reshape(-1)
    if batch_indices is None:
        if values.shape[0] != batch_size:
            raise ValueError("one CE value per spectrum is required without batch indices")
        return values
    batch_indices = batch_indices.reshape(-1)
    if batch_indices.shape != values.shape or batch_indices.dtype != th.long:
        raise ValueError("CE batch indices must be a matching torch.long vector")
    sums = values.new_zeros(batch_size).index_add(0, batch_indices, values)
    counts = values.new_zeros(batch_size).index_add(
        0,
        batch_indices,
        th.ones_like(values),
    )
    _validate_tensor_condition(
        (counts > 0).all(),
        "every spectrum must have at least one CE value",
    )
    return sums / counts


def _relative_fragment_keep_mask(
    fragment_log_probabilities: th.Tensor,
    fragment_batch_indices: th.Tensor,
    batch_size: int,
    min_relative_probability: float,
) -> th.Tensor:
    """Select fragments whose proposal mass is within a ratio of the MAP fragment."""
    if fragment_log_probabilities.shape != fragment_batch_indices.shape:
        raise ValueError("fragment probabilities and batch indices must align")
    if fragment_batch_indices.dtype != th.long:
        raise ValueError("fragment batch indices must have dtype torch.long")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(min_relative_probability) or not 0.0 <= min_relative_probability <= 1.0:
        raise ValueError("min_relative_probability must be finite and in [0, 1]")
    fragment_counts = th.bincount(fragment_batch_indices, minlength=batch_size)
    _validate_tensor_condition(
        (fragment_counts > 0).all(),
        "every molecule must contain at least one fragment",
    )
    if min_relative_probability == 0.0:
        return th.ones_like(fragment_log_probabilities, dtype=th.bool)
    map_log_probabilities = (
        fragment_log_probabilities.detach()
        .new_full(
            (batch_size,),
            -th.inf,
        )
        .scatter_reduce(
            0,
            fragment_batch_indices,
            fragment_log_probabilities.detach(),
            reduce="amax",
            include_self=True,
        )
    )
    return fragment_log_probabilities.detach() >= (
        map_log_probabilities[fragment_batch_indices] + math.log(min_relative_probability)
    )


def _relative_channel_keep_mask(
    logits: th.Tensor,
    valid_mask: th.Tensor,
    min_relative_probability: float,
    *,
    binary_logits: bool = False,
) -> th.Tensor:
    """Apply a per-fragment MAP-relative probability floor over valid ion channels."""
    if logits.shape != valid_mask.shape or valid_mask.dtype != th.bool or logits.ndim < 2:
        raise ValueError("logits and valid_mask must have matching rank-2-or-higher shapes")
    if not math.isfinite(min_relative_probability) or not 0.0 <= min_relative_probability <= 1.0:
        raise ValueError("min_relative_probability must be finite and in [0, 1]")
    if not isinstance(binary_logits, bool):
        raise ValueError("binary_logits must be boolean")
    flat_valid = valid_mask.flatten(start_dim=1)
    _validate_tensor_condition(
        flat_valid.any(dim=1).all(),
        "every fragment must have at least one valid channel",
    )
    if min_relative_probability == 0.0:
        return valid_mask.clone()
    log_scores = F.logsigmoid(logits) if binary_logits else logits
    flat_logits = log_scores.detach().flatten(start_dim=1)
    map_logits = flat_logits.masked_fill(~flat_valid, -th.inf).max(dim=1).values
    keep = flat_valid & (
        flat_logits >= map_logits.unsqueeze(1) + math.log(min_relative_probability)
    )
    return keep.reshape_as(valid_mask)


def _absolute_binary_channel_keep_mask(
    logits: th.Tensor,
    valid_mask: th.Tensor,
    fragment_batch_indices: th.Tensor,
    batch_size: int,
    min_probability: float,
) -> th.Tensor:
    """Apply an absolute Bernoulli support floor with one fallback per spectrum."""
    if logits.shape != valid_mask.shape or valid_mask.dtype != th.bool or logits.ndim != 3:
        raise ValueError("logits and valid_mask must be matching rank-3 tensors")
    if fragment_batch_indices.shape != (logits.shape[0],):
        raise ValueError("fragment_batch_indices must contain one index per fragment")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(min_probability) or not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be finite and in [0, 1]")
    flat_valid = valid_mask.flatten()
    expanded_batch_indices = fragment_batch_indices[:, None, None].expand_as(valid_mask)
    flat_batch_indices = expanded_batch_indices.flatten()
    _validate_tensor_condition(
        th.bincount(flat_batch_indices[flat_valid], minlength=batch_size).gt(0).all(),
        "every spectrum must have at least one valid channel",
    )
    if min_probability == 0.0:
        return valid_mask.clone()

    flat_log_probabilities = F.logsigmoid(logits.detach()).flatten()
    keep = flat_valid & (flat_log_probabilities >= math.log(min_probability))
    kept_per_spectrum = th.bincount(flat_batch_indices[keep], minlength=batch_size)
    if kept_per_spectrum.gt(0).all():
        return keep.reshape_as(valid_mask)

    for batch_index in th.where(kept_per_spectrum == 0)[0]:
        batch_valid_indices = th.where(flat_valid & (flat_batch_indices == batch_index))[0]
        fallback_index = batch_valid_indices[
            flat_log_probabilities[batch_valid_indices].argmax()
        ]
        keep[fallback_index] = True
    return keep.reshape_as(valid_mask)


def _ion_mode_logits_from_ion_h_support(
    ion_h_support_logits: th.Tensor,
    valid_channels: th.Tensor,
    *,
    binary_logits: bool,
) -> th.Tensor:
    """Collapse ion/H support scores to ion-mode scores with matching semantics.

    Categorical channel logits are marginalized from their masked joint distribution.
    For independent Bernoulli channels, a mode is supported when any valid H channel
    is supported, so its probability is ``1 - product_h(1 - p_h)``.
    """
    if (
        ion_h_support_logits.ndim != 3
        or valid_channels.shape != ion_h_support_logits.shape
        or valid_channels.dtype != th.bool
    ):
        raise ValueError("ion/H support logits and validity must be aligned rank-3 tensors")
    if not isinstance(binary_logits, bool):
        raise ValueError("binary_logits must be boolean")
    _validate_tensor_condition(
        valid_channels.flatten(start_dim=1).any(dim=1).all(),
        "every fragment must have at least one valid ion/H channel",
    )
    if not binary_logits:
        channel_log_probabilities = masked_ion_hydrogen_log_probabilities(
            ion_h_support_logits,
            valid_channels,
        )
        return th.logsumexp(channel_log_probabilities, dim=2)

    valid_logits = ion_h_support_logits.masked_fill(~valid_channels, -th.inf)
    mode_log_no_support = F.logsigmoid(-valid_logits).sum(dim=2)
    mode_log_support = th.where(
        mode_log_no_support < -math.log(2.0),
        th.log1p(-mode_log_no_support.exp()),
        th.log(-th.expm1(mode_log_no_support)),
    )
    return mode_log_support - mode_log_no_support


def _fixed_top_k_channel_keep_mask(
    logits: th.Tensor,
    valid_mask: th.Tensor,
    top_k: int,
) -> th.Tensor:
    """Keep the stable top-k valid ion channels independently per fragment."""
    if logits.shape != valid_mask.shape or valid_mask.dtype != th.bool or logits.ndim < 2:
        raise ValueError("logits and valid_mask must have matching rank-2-or-higher shapes")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    flat_valid = valid_mask.flatten(start_dim=1)
    _validate_tensor_condition(
        flat_valid.any(dim=1).all(),
        "every fragment must have at least one valid channel",
    )
    flat_logits = logits.detach().flatten(start_dim=1).masked_fill(~flat_valid, -th.inf)
    stable_order = th.sort(flat_logits, dim=1, descending=True, stable=True).indices
    keep_count = flat_valid.sum(dim=1).clamp(max=top_k)
    ranked_keep = th.arange(flat_valid.shape[1], device=logits.device).unsqueeze(0) < (
        keep_count.unsqueeze(1)
    )
    keep = th.zeros_like(flat_valid).scatter(1, stable_order, ranked_keep)
    return (keep & flat_valid).reshape_as(valid_mask)


def _relative_joint_channel_keep_mask(
    channel_log_weights: th.Tensor,
    valid_mask: th.Tensor,
    fragment_batch_indices: th.Tensor,
    batch_size: int,
    min_relative_weight: float,
) -> th.Tensor:
    """Apply a spectrum-relative floor to joint fragment/ion/H weights.

    At least the highest-weight valid channel of every fragment is retained because
    downstream conditional channel normalization requires nonempty fragment support.
    """
    if channel_log_weights.shape != valid_mask.shape or channel_log_weights.ndim != 3:
        raise ValueError("channel weights and valid_mask must be aligned rank-3 tensors")
    if valid_mask.dtype != th.bool:
        raise ValueError("valid_mask must be boolean")
    if fragment_batch_indices.shape != (channel_log_weights.shape[0],):
        raise ValueError("fragment_batch_indices must contain one index per fragment")
    if fragment_batch_indices.dtype != th.long:
        raise ValueError("fragment_batch_indices must have dtype torch.long")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not math.isfinite(min_relative_weight) or not 0.0 <= min_relative_weight <= 1.0:
        raise ValueError("min_relative_weight must be finite and in [0, 1]")
    flat_valid = valid_mask.flatten(start_dim=1)
    _validate_tensor_condition(
        flat_valid.any(dim=1).all(),
        "every fragment must have at least one valid channel",
    )
    if min_relative_weight == 0.0:
        return valid_mask.clone()

    flat_log_weights = channel_log_weights.detach().flatten(start_dim=1)
    fragment_maxima = flat_log_weights.masked_fill(~flat_valid, -th.inf).max(dim=1).values
    spectrum_maxima = fragment_maxima.new_full((batch_size,), -th.inf).scatter_reduce(
        0,
        fragment_batch_indices,
        fragment_maxima,
        reduce="amax",
        include_self=True,
    )
    keep = flat_valid & (
        flat_log_weights
        >= spectrum_maxima[fragment_batch_indices].unsqueeze(1) + math.log(min_relative_weight)
    )
    fragment_fallback = _fixed_top_k_channel_keep_mask(
        channel_log_weights,
        valid_mask,
        top_k=1,
    ).flatten(start_dim=1)
    return (keep | fragment_fallback).reshape_as(valid_mask)


def _parse_formula_delta(formula: str) -> dict[str, int]:
    """Parse the signed element-count notation used by the adduct tables."""
    if not formula:
        return {}
    counts: dict[str, int] = {}
    position = 0
    for match in re.finditer(r"([A-Z][a-z]?)(-?\d*)", formula):
        if match.start() != position:
            raise ValueError(f"invalid formula delta: {formula}")
        count_text = match.group(2)
        count = 1 if not count_text else int(count_text)
        counts[match.group(1)] = counts.get(match.group(1), 0) + count
        position = match.end()
    if position != len(formula):
        raise ValueError(f"invalid formula delta: {formula}")
    return counts


def _cmf_duplicate_crf_shift(
    precursor_type: str,
    max_h_transfer: int,
    *,
    tolerance: float = 1e-8,
) -> int | None:
    """Return the reachable CRF H shift duplicated by CMF, if one exists."""
    if precursor_type not in PREC_TYPE_TO_CMF_MASS_DIFF:
        return None
    shift = (
        PREC_TYPE_TO_CMF_MASS_DIFF[precursor_type] - PREC_TYPE_TO_MASS_DIFF[precursor_type]
    ) / H_MASS
    rounded_shift = round(shift)
    if abs(shift - rounded_shift) > tolerance or not (
        -max_h_transfer <= rounded_shift <= max_h_transfer
    ):
        return None
    crf_counts = _parse_formula_delta(PREC_TYPE_TO_FORMULA_DIFF[precursor_type])
    cmf_counts = _parse_formula_delta(PREC_TYPE_TO_CMF_FORMULA_DIFF[precursor_type])
    for element in set(crf_counts) | set(cmf_counts) | {"H"}:
        expected = crf_counts.get(element, 0) + (rounded_shift if element == "H" else 0)
        if cmf_counts.get(element, 0) != expected:
            return None
    return rounded_shift


class BernoulliMixtureWeightHead(nn.Module):
    """Predict molecule-level log weights for latent Bernoulli proposal modes."""

    def __init__(
        self,
        atom_embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_mixtures: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(atom_embedding_dim, hidden_dim, num_mixtures) <= 0:
            raise ValueError("mixture-head dimensions must be positive")
        if context_dim < 0:
            raise ValueError("context_dim must be nonnegative")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        self.atom_embedding_dim = atom_embedding_dim
        self.context_dim = context_dim
        self.num_mixtures = num_mixtures
        self.head = nn.Sequential(
            nn.Linear(atom_embedding_dim + context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_mixtures),
        )

    def forward(
        self,
        atom_embeddings: th.Tensor,
        atom_batch_indices: th.Tensor,
        context: th.Tensor,
        *,
        batch_size: int,
    ) -> th.Tensor:
        """Return normalized log mixture weights with shape ``(batch_size, M)``."""
        if atom_embeddings.ndim != 2 or atom_embeddings.shape[1] != self.atom_embedding_dim:
            raise ValueError(
                f"atom_embeddings must have shape (num_atoms, {self.atom_embedding_dim})"
            )
        if atom_batch_indices.shape != (atom_embeddings.shape[0],):
            raise ValueError("atom_batch_indices must contain one index per atom")
        if atom_batch_indices.dtype != th.long:
            raise ValueError("atom_batch_indices must have dtype torch.long")
        atom_sums = atom_embeddings.new_zeros((batch_size, self.atom_embedding_dim)).index_add(
            0, atom_batch_indices, atom_embeddings
        )
        atom_counts = th.bincount(atom_batch_indices, minlength=batch_size).to(atom_embeddings)
        _validate_tensor_condition(
            (atom_counts > 0).all(),
            "every molecule must contain at least one atom",
        )
        molecule_embeddings = atom_sums / atom_counts.unsqueeze(1)
        if context.shape != (batch_size, self.context_dim):
            raise ValueError(f"context must have shape ({batch_size}, {self.context_dim})")
        return F.log_softmax(
            self.head(th.cat([molecule_embeddings, context], dim=1)),
            dim=1,
        )


class SmartsMechanismHead(nn.Module):
    """Choose ordinary versus SMARTS assembly within one retained cut mask."""

    def __init__(
        self,
        bond_embedding_dim: int,
        context_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        num_rule_indices = 1 + MASS_SHATTER_NUM_RULES
        self.rule_embedding = nn.Embedding(num_rule_indices, context_dim)
        self.scorer = nn.Sequential(
            nn.Linear(bond_embedding_dim + 2 * context_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        boundary_embeddings: th.Tensor,
        rule_indices: th.Tensor,
        context: th.Tensor,
    ) -> th.Tensor:
        """Return one unnormalized mechanism logit per mechanism."""
        return self.scorer(
            th.cat(
                (boundary_embeddings, self.rule_embedding(rule_indices), context),
                dim=1,
            )
        ).squeeze(1)


class MassShatterContextEncoder(nn.Module):
    """Encode acquisition metadata once per spectrum.

    Categorical metadata (precursor adduct, instrument, fragmentation mode) is
    embedded; collision energies are either standardized scalars or learned vectors;
    and the concatenation is projected to one conditioning vector reused by every head.
    Instrument and fragmentation-mode embeddings are only created when the dataset
    declares those vocabularies; CE inputs are only consumed when enabled.

    Attributes:
        use_nce: Whether normalized collision energy is part of the context.
        use_ace: Whether absolute collision energy is part of the context.
        precursor_embedding: Adduct-type embedding table.
        instrument_embedding: Instrument-type embedding table, or None if unused.
        fragmentation_mode_embedding: Fragmentation-mode embedding table (one extra
            row reserved for the unknown/missing mode), or None if unused.
        projector: MLP mapping the concatenated features to ``context_dim``.
    """

    def __init__(
        self,
        *,
        num_precursor_types: int,
        num_instrument_types: int,
        num_fragmentation_modes: int,
        use_nce: bool,
        use_ace: bool,
        int_embedder: str,
        ce_insert_type: str,
        ce_insert_size: int,
        metadata_embedding_dim: int,
        context_dim: int,
        nce_max: float,
        nce_mean: float,
        nce_std: float,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
    ) -> None:
        """Initialize the context encoder.

        Args:
            num_precursor_types: Size of the adduct vocabulary.
            num_instrument_types: Size of the instrument vocabulary; 0 disables the
                instrument embedding.
            num_fragmentation_modes: Size of the fragmentation-mode vocabulary; 0
                disables the embedding. One extra row is allocated for the
                unknown/missing mode.
            use_nce: Feed normalized collision energy into the context.
            use_ace: Feed absolute collision energy into the context.
            int_embedder: Integer featurizer used for learned CE embeddings.
            ce_insert_type: ``embed`` or ``embed_clamped`` enables learned CE
                embeddings; other modes retain the standardized scalar input.
            ce_insert_size: Width of each learned CE embedding.
            metadata_embedding_dim: Width of each categorical embedding.
            context_dim: Width of the emitted conditioning vector.
            nce_max: Largest NCE represented by the learned embedding.
            nce_mean: Mean used to standardize NCE.
            nce_std: Standard deviation used to standardize NCE.
            ace_max: Largest ACE represented by the learned embedding.
            ace_mean: Mean used to standardize ACE.
            ace_std: Standard deviation used to standardize ACE.

        Raises:
            ValueError: If a vocabulary size, dimension, or CE scale is invalid.
        """
        super().__init__()
        if num_precursor_types <= 0:
            raise ValueError("num_precursor_types must be positive")
        if metadata_embedding_dim <= 0 or context_dim <= 0:
            raise ValueError("context dimensions must be positive")
        if ce_insert_type not in {"id", "lin", "embed", "embed_clamped", "bin"}:
            raise ValueError(f"unknown CE insertion type: {ce_insert_type}")
        if ce_insert_size <= 0:
            raise ValueError("ce_insert_size must be positive")
        if nce_std <= 0.0 or ace_std <= 0.0:
            raise ValueError("CE standard deviations must be positive")
        if nce_max <= 0.0 or ace_max <= 0.0:
            raise ValueError("CE maxima must be positive")

        self.use_nce = use_nce
        self.use_ace = use_ace
        self.ce_insert_type = ce_insert_type
        self.ce_insert_size = ce_insert_size
        self.nce_max = nce_max
        self.nce_mean = nce_mean
        self.nce_std = nce_std
        self.ace_max = ace_max
        self.ace_mean = ace_mean
        self.ace_std = ace_std
        self.use_ce_embedding = ce_insert_type in {"embed", "embed_clamped"}
        self.nce_embedding = None
        self.ace_embedding = None
        if self.use_ce_embedding:
            nce_count = int(nce_max) + int(ce_insert_type == "embed")
            ace_count = int(ace_max) + int(ce_insert_type == "embed")
            if use_nce:
                nce_embedder = get_embedder(int_embedder, max_count_int=nce_count)
                self.nce_embedding = nn.Sequential(
                    nce_embedder,
                    nn.Linear(nce_embedder.num_dim, ce_insert_size),
                )
            if use_ace:
                ace_embedder = get_embedder(int_embedder, max_count_int=ace_count)
                self.ace_embedding = nn.Sequential(
                    ace_embedder,
                    nn.Linear(ace_embedder.num_dim, ce_insert_size),
                )
        self.precursor_embedding = nn.Embedding(
            num_precursor_types,
            metadata_embedding_dim,
        )
        self.instrument_embedding = (
            nn.Embedding(num_instrument_types, metadata_embedding_dim)
            if num_instrument_types > 0
            else None
        )
        self.fragmentation_mode_embedding = (
            nn.Embedding(num_fragmentation_modes + 1, metadata_embedding_dim)
            if num_fragmentation_modes > 0
            else None
        )
        # Feature width: adduct embedding, one scalar or learned vector per enabled CE
        # channel, plus each optional categorical embedding that was actually created.
        input_dim = metadata_embedding_dim
        ce_feature_dim = ce_insert_size if self.use_ce_embedding else 1
        input_dim += ce_feature_dim * (int(use_nce) + int(use_ace))
        input_dim += metadata_embedding_dim * int(self.instrument_embedding is not None)
        input_dim += metadata_embedding_dim * int(self.fragmentation_mode_embedding is not None)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

    def _embed_ce_values(
        self,
        values: th.Tensor,
        batch_indices: th.Tensor | None,
        batch_size: int,
        max_value: float,
        embedding: nn.Module,
    ) -> th.Tensor:
        """Embed each CE reading, then mean-pool readings by spectrum."""
        values = values.reshape(-1)
        if batch_indices is None:
            if values.shape[0] != batch_size:
                raise ValueError("one CE value per spectrum is required without batch indices")
            batch_indices = th.arange(batch_size, device=values.device)
        else:
            batch_indices = batch_indices.reshape(-1)
            if batch_indices.shape != values.shape or batch_indices.dtype != th.long:
                raise ValueError("CE batch indices must be a matching torch.long vector")

        upper_bound = int(max_value) - int(self.ce_insert_type == "embed_clamped")
        embedding_indices = values.clamp(min=-1, max=upper_bound).round().long().unsqueeze(1)
        embedded_values = embedding(embedding_indices)
        sums = embedded_values.new_zeros(batch_size, self.ce_insert_size).index_add(
            0,
            batch_indices,
            embedded_values,
        )
        counts = embedded_values.new_zeros(batch_size).index_add(
            0,
            batch_indices,
            th.ones_like(values, dtype=embedded_values.dtype),
        )
        _validate_tensor_condition(
            (counts > 0).all(),
            "every spectrum must have at least one CE value",
        )
        return sums / counts.unsqueeze(1)

    def forward(
        self,
        *,
        batch_size: int,
        precursor_types: th.Tensor,
        instrument_types: th.Tensor | None,
        fragmentation_modes: th.Tensor | None,
        nce: th.Tensor | None,
        nce_batch_indices: th.Tensor | None,
        ace: th.Tensor | None,
        ace_batch_indices: th.Tensor | None,
        include_collision_energy: bool = True,
    ) -> th.Tensor:
        """Return one conditioning vector per spectrum.

        Args:
            batch_size: Number of spectra in the batch.
            precursor_types: Adduct index per spectrum, shape ``(batch_size,)``.
            instrument_types: Instrument index per spectrum, shape ``(batch_size,)``.
                Required only when the instrument embedding exists.
            fragmentation_modes: Fragmentation-mode index per spectrum, shape
                ``(batch_size,)``. Required only when that embedding exists.
            nce: Normalized collision energies, ragged over spectra.
            nce_batch_indices: Spectrum index per NCE value, or None when there is
                exactly one value per spectrum.
            ace: Absolute collision energies, ragged over spectra.
            ace_batch_indices: Spectrum index per ACE value, or None when there is
                exactly one value per spectrum.

        Returns:
            Context vectors of shape ``(batch_size, context_dim)``.

        Raises:
            ValueError: If a field required by the configured context is missing, or
                ``precursor_types`` does not hold one index per spectrum.
        """
        precursor_types = precursor_types.reshape(-1)
        if precursor_types.shape[0] != batch_size:
            raise ValueError("precursor_types must contain one index per spectrum")
        features = [self.precursor_embedding(precursor_types)]

        if self.instrument_embedding is not None:
            if instrument_types is None:
                raise ValueError("instrument_types are required by the configured context")
            features.append(self.instrument_embedding(instrument_types.reshape(-1)))
        if self.fragmentation_mode_embedding is not None:
            if fragmentation_modes is None:
                raise ValueError("fragmentation_modes are required by the configured context")
            features.append(self.fragmentation_mode_embedding(fragmentation_modes.reshape(-1)))
        if self.use_nce:
            if not include_collision_energy:
                ce_width = self.ce_insert_size if self.use_ce_embedding else 1
                features.append(features[0].new_zeros((batch_size, ce_width)))
            elif nce is None:
                raise ValueError("NCE values are required by the configured context")
            elif self.nce_embedding is None:
                mean_nce = _mean_values_by_batch(nce, nce_batch_indices, batch_size)
                features.append(((mean_nce - self.nce_mean) / self.nce_std).unsqueeze(1))
            else:
                features.append(
                    self._embed_ce_values(
                        nce,
                        nce_batch_indices,
                        batch_size,
                        self.nce_max,
                        self.nce_embedding,
                    )
                )
        if self.use_ace:
            if not include_collision_energy:
                ce_width = self.ce_insert_size if self.use_ce_embedding else 1
                features.append(features[0].new_zeros((batch_size, ce_width)))
            elif ace is None:
                raise ValueError("ACE values are required by the configured context")
            elif self.ace_embedding is None:
                mean_ace = _mean_values_by_batch(ace, ace_batch_indices, batch_size)
                features.append(((mean_ace - self.ace_mean) / self.ace_std).unsqueeze(1))
            else:
                features.append(
                    self._embed_ce_values(
                        ace,
                        ace_batch_indices,
                        batch_size,
                        self.ace_max,
                        self.ace_embedding,
                    )
                )
        return self.projector(th.cat(features, dim=1))


class MassShatterModel(nn.Module):
    """Predict sparse MS/MS spectra from globally ranked precursor bond masks.

    The forward pass runs four stages (see the numbered comments in :meth:`forward`):

    1. Encode the intact precursor with one GNN pass and score every chemical bond
       with an independent break logit.
    2. Rank whole cut masks by their Bernoulli probability and cut each into connected
       components. Optionally continue until ``max_masks`` unique atom sets are found.
    3. Score which component keeps the charge, then which ion mode / hydrogen-transfer
       channel that component emits; a transformer over a molecule's candidate set
       supplies residual corrections so components compete with one another.
    4. Flatten every chemically valid channel into the sparse prediction contract.

    Attributes:
        max_masks: Number of cut masks materialized per precursor.
        unique_fragment_mode: Whether ``max_masks`` is a unique atom-set target.
        max_enumerated_masks: Safety cap for unique-fragment mask enumeration.
        mask_selection_threads: Worker threads used to select cut masks on CPU.
        min_mask_relative_probability: Masks below this fraction of the MAP mask
            probability are dropped.
        min_fragment_relative_probability: Unique fragments below this fraction of
            the MAP marginalized fragment probability are dropped.
        max_h_transfer: Largest hydrogen transfer (in either direction) per fragment.
        include_cmf: Whether the charge-migration ion channel is emitted.
        num_ion_modes: 2 with CMF enabled (CRF + CMF), otherwise 1.
        num_h_shifts: ``2 * max_h_transfer + 1`` hydrogen-transfer channels.
    """

    def __init__(
        self,
        *,
        node_feature_dim: int,
        edge_feature_dim: int,
        precursor_hidden_dim: int,
        precursor_num_layers: int,
        precursor_num_heads: int,
        precursor_gnn_type: str,
        precursor_normalization: str,
        precursor_dropout: float,
        context_dim: int,
        metadata_embedding_dim: int,
        head_hidden_dim: int,
        head_dropout: float,
        bond_num_layers: int,
        proposal_num_mixtures: int,
        intensity_transformer_dim: int,
        intensity_transformer_num_layers: int,
        intensity_transformer_num_heads: int,
        intensity_transformer_feedforward_dim: int,
        intensity_transformer_num_inducing_points: int,
        intensity_atom_decoder_num_layers: int,
        intensity_transformer_dropout: float,
        intensity_output: str,
        use_mask_residuals: bool,
        use_base_probability_feature: bool,
        return_base_spectrum: bool,
        detach_base_ion_h: bool,
        proposal_selection_only: bool,
        separate_proposal_encoder: bool,
        share_proposal_across_ev: bool,
        separate_ion_mode_proposal_head: bool,
        ion_mode_proposal_binary_logits: bool,
        min_ion_mode_relative_probability: float,
        separate_ion_h_support_head: bool,
        ion_h_support_binary_logits: bool,
        gate_independent_intensity_with_support: bool,
        min_ion_h_relative_probability: float,
        min_ion_h_absolute_probability: float,
        apply_channel_selection_in_training: bool,
        precursor_types: Sequence[str],
        instrument_types: Sequence[str],
        fragmentation_modes: Sequence[str],
        int_embedder: str,
        ce_insert_type: str,
        ce_insert_size: int,
        use_nce: bool,
        use_ace: bool,
        nce_max: float,
        nce_mean: float,
        nce_std: float,
        ace_max: float,
        ace_mean: float,
        ace_std: float,
        max_masks: int,
        unique_fragment_mode: bool,
        max_enumerated_masks: int,
        mask_selection_threads: int = 1,
        min_mask_relative_probability: float,
        min_fragment_relative_probability: float,
        include_single_bond_cuts: bool,
        max_h_transfer: int,
        include_cmf: bool,
        charge_pooling: str,
        use_smarts_templates: bool,
    ) -> None:
        """Initialize the MASShatter model.

        Args:
            node_feature_dim: Width of the precursor atom features.
            edge_feature_dim: Width of the precursor bond features.
            precursor_hidden_dim: Hidden width of the precursor GNN; also the width of
                the pooled fragment/complement/boundary embeddings.
            precursor_num_layers: Number of message-passing layers.
            precursor_num_heads: Number of molecular-attention heads.
            precursor_gnn_type: GNN cell name understood by :class:`GNN`.
            precursor_normalization: Normalization used inside the GNN.
            precursor_dropout: Dropout inside the GNN.
            context_dim: Width of the per-spectrum conditioning vector.
            metadata_embedding_dim: Width of each metadata embedding.
            head_hidden_dim: Hidden width of the bond and charge heads.
            head_dropout: Dropout inside the bond and charge heads.
            bond_num_layers: Bond-refinement layers; 0 selects the cheap
                :class:`EndpointBondBreakHead`, otherwise
                :class:`BondFocusedBreakHead`.
            proposal_num_mixtures: Number of latent normalized Bernoulli mask models.
                One recovers the original independent proposal exactly.
            intensity_transformer_dim: Model width of the candidate transformer.
            intensity_transformer_num_layers: Depth of the candidate transformer.
            intensity_transformer_num_heads: Attention heads per layer.
            intensity_transformer_feedforward_dim: Feed-forward width per layer.
            intensity_transformer_num_inducing_points: Latent tokens used to keep
                attention linear in the number of candidates.
            intensity_atom_decoder_num_layers: Masked fragment-to-atom cross-attention
                layers applied before candidate-set interaction. Zero disables them.
            intensity_transformer_dropout: Dropout inside the candidate transformer.
            intensity_output: ``"probability"`` for the normalized legacy spectrum or
                ``"independent"`` for normal-space nonnegative pathway intensities.
            use_mask_residuals: Whether transformer residuals may redistribute retained
                probability between selected masks.
            use_base_probability_feature: Whether the candidate transformer receives
                the pre-residual charge and mask/fragment log probabilities as inputs.
            return_base_spectrum: Whether to return pre-residual spectrum log
                probabilities for the base-only CE diagnostic or auxiliary loss.
            detach_base_ion_h: Whether the pre-residual spectrum treats the shared
                ion/H probabilities as constants, preventing base-only CE gradients
                from entering the candidate intensity transformer through that head.
            proposal_selection_only: Whether proposal probabilities only choose the
                retained unique-fragment set. The final retained shape then comes from
                the intensity transformer alone.
            separate_proposal_encoder: Whether bond/charge proposal scoring uses an
                independent molecular GNN trained by proposal supervision.
            share_proposal_across_ev: Run the proposal encoder and heads once for all
                spectra in the same acquisition group, without collision-energy input.
            ion_mode_proposal_binary_logits: Whether the separate ion-mode proposal
                head emits independent Bernoulli logits.
            ion_h_support_binary_logits: Whether the ion/H support source emits
                independent Bernoulli logits instead of categorical logits.
            gate_independent_intensity_with_support: Whether independent-output
                intensities are scaled by the separate ion/H support head's sigmoid
                probability, in both training and evaluation. Requires independent
                intensity output, a separate binary ion/H support head, and is
                independent of ``min_ion_h_absolute_probability``.
            min_ion_h_absolute_probability: Evaluation-only absolute Bernoulli
                support floor. Zero disables it.
            precursor_types: Supported adduct strings; sorted and used as the label
                vocabulary, so their order must match the dataset encoding.
            instrument_types: Instrument vocabulary; empty disables the embedding.
            fragmentation_modes: Fragmentation-mode vocabulary; empty disables it.
            int_embedder: Integer featurizer used by learned CE embeddings.
            ce_insert_type: CE representation type; ``embed`` enables learned vectors.
            ce_insert_size: Width of each learned CE embedding.
            use_nce: Condition on normalized collision energy.
            use_ace: Condition on absolute collision energy.
            nce_max: Largest NCE represented by the learned embedding.
            nce_mean: Mean used to standardize NCE.
            nce_std: Standard deviation used to standardize NCE.
            ace_max: Largest ACE represented by the learned embedding.
            ace_mean: Mean used to standardize ACE.
            ace_std: Standard deviation used to standardize ACE.
            max_masks: Cut masks kept per precursor.
            unique_fragment_mode: Treat ``max_masks`` as a unique-fragment target and
                marginalize duplicate component pathways before intensity scoring.
            mask_selection_threads: Worker threads used to select Bernoulli-mixture cut
                masks. The selection kernel releases the GIL, so values above one give
                real parallelism and leave the selected masks unchanged. Defaults to 1.
            max_enumerated_masks: Safety cap on latent masks considered while filling
                the unique-fragment target.
            min_mask_relative_probability: MAP-relative probability floor below which
                a ranked mask is dropped before candidate assembly.
            min_fragment_relative_probability: MAP-relative probability floor below
                which a marginalized unique fragment is dropped before intensity
                scoring. Zero disables fragment pruning.
            include_single_bond_cuts: Enumerate single-bond cuts in the ranked masks.
            max_h_transfer: Largest hydrogen transfer per fragment, in either
                direction.
            include_cmf: Emit the charge-migration ion channel alongside CRF.
            charge_pooling: ``logsumexp`` treats every atom as a potential charge site
                (favoring large components); ``logmeanexp`` removes that size prior.
            use_smarts_templates: Add matched SMARTS rearrangements as alternative
                assembly mechanisms within their corresponding cut masks.

        Raises:
            ValueError: If any hyperparameter is out of range or an adduct in
                ``precursor_types`` has no tabulated mass difference.
        """
        super().__init__()
        if max_masks <= 0 or max_h_transfer < 0:
            raise ValueError("max_masks must be positive and max_h_transfer nonnegative")
        if max_enumerated_masks <= 0:
            raise ValueError("max_enumerated_masks must be positive")
        if mask_selection_threads < 1:
            raise ValueError("mask_selection_threads must be at least 1")
        if unique_fragment_mode and max_enumerated_masks < max_masks:
            raise ValueError(
                "max_enumerated_masks must be at least max_masks in unique-fragment mode"
            )
        if proposal_selection_only and not unique_fragment_mode:
            raise ValueError("proposal_selection_only requires unique_fragment_mode")
        if proposal_selection_only and use_base_probability_feature:
            raise ValueError(
                "proposal_selection_only cannot expose proposal probabilities to the "
                "intensity transformer"
            )
        if proposal_selection_only and not use_mask_residuals:
            raise ValueError("proposal_selection_only requires intensity residuals")
        if intensity_output not in {"probability", "independent"}:
            raise ValueError(f"unknown MASShatter intensity output: {intensity_output}")
        if intensity_output == "independent" and not (
            unique_fragment_mode and proposal_selection_only
        ):
            raise ValueError(
                "independent intensity output requires unique-fragment mode and "
                "proposal-selection-only mode"
            )
        if separate_proposal_encoder and not proposal_selection_only:
            raise ValueError("separate_proposal_encoder requires proposal_selection_only")
        if (separate_ion_mode_proposal_head or separate_ion_h_support_head) and not (
            unique_fragment_mode and proposal_selection_only
        ):
            raise ValueError(
                "ion-mode and ion/H support heads require unique-fragment "
                "proposal-selection-only mode"
            )
        for name, flag_value in (
            ("ion_mode_proposal_binary_logits", ion_mode_proposal_binary_logits),
            ("ion_h_support_binary_logits", ion_h_support_binary_logits),
            ("gate_independent_intensity_with_support", gate_independent_intensity_with_support),
        ):
            if not isinstance(flag_value, bool):
                raise ValueError(f"{name} must be boolean")
        if gate_independent_intensity_with_support and not (
            intensity_output == "independent" and separate_ion_h_support_head
        ):
            raise ValueError(
                "gating independent intensity by support requires independent intensity "
                "output and a separate ion/H support head"
            )
        if gate_independent_intensity_with_support and not ion_h_support_binary_logits:
            raise ValueError("independent intensity support gating requires binary support logits")
        if min_ion_h_absolute_probability > 0.0 and not (
            intensity_output == "independent"
            and separate_ion_h_support_head
            and ion_h_support_binary_logits
        ):
            raise ValueError(
                "an absolute ion/H support floor requires independent intensity and a "
                "separate binary support head"
            )
        if (
            not math.isfinite(min_mask_relative_probability)
            or not 0.0 <= min_mask_relative_probability <= 1.0
        ):
            raise ValueError(
                "min_mask_relative_probability must be finite and in [0, 1], "
                f"got {min_mask_relative_probability}"
            )
        if (
            not math.isfinite(min_fragment_relative_probability)
            or not 0.0 <= min_fragment_relative_probability <= 1.0
        ):
            raise ValueError(
                "min_fragment_relative_probability must be finite and in [0, 1], "
                f"got {min_fragment_relative_probability}"
            )
        if min_fragment_relative_probability > 0.0 and not unique_fragment_mode:
            raise ValueError("min_fragment_relative_probability requires unique_fragment_mode")
        for name, value in (
            ("min_ion_mode_relative_probability", min_ion_mode_relative_probability),
            ("min_ion_h_relative_probability", min_ion_h_relative_probability),
            ("min_ion_h_absolute_probability", min_ion_h_absolute_probability),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if bond_num_layers < 0:
            raise ValueError("bond_num_layers must be nonnegative")
        if proposal_num_mixtures <= 0:
            raise ValueError("proposal_num_mixtures must be positive")
        if charge_pooling not in {"logsumexp", "logmeanexp"}:
            raise ValueError(f"unknown charge pooling mode: {charge_pooling}")

        # Sorting fixes the adduct index order that ``spec_prec_type`` refers to and
        # that the registered chemistry buffers are laid out along.
        self.precursor_types = tuple(sorted(precursor_types))
        if not self.precursor_types:
            raise ValueError("precursor_types must not be empty")
        self.fragmentation_modes = tuple(sorted(fragmentation_modes))
        missing_precursors = set(self.precursor_types) - PREC_TYPE_TO_MASS_DIFF.keys()
        if missing_precursors:
            raise ValueError(f"unsupported precursor types: {sorted(missing_precursors)}")
        non_degenerate_cmf = [
            precursor_type
            for precursor_type in self.precursor_types
            if precursor_type in PREC_TYPE_TO_CMF_MASS_DIFF
            and _cmf_duplicate_crf_shift(precursor_type, max_h_transfer) is None
        ]
        if separate_ion_mode_proposal_head and (not include_cmf or not non_degenerate_cmf):
            raise ValueError(
                "ion-mode proposal head requires a configured non-degenerate CMF adduct"
            )
        self.max_masks = max_masks
        self.unique_fragment_mode = unique_fragment_mode
        self.max_enumerated_masks = max_enumerated_masks
        self.mask_selection_threads = mask_selection_threads
        self.min_mask_relative_probability = min_mask_relative_probability
        self.min_fragment_relative_probability = min_fragment_relative_probability
        self.include_single_bond_cuts = include_single_bond_cuts
        self.max_h_transfer = max_h_transfer
        self.include_cmf = include_cmf
        self.charge_pooling = charge_pooling
        self.use_smarts_templates = use_smarts_templates
        self.use_mask_residuals = use_mask_residuals
        self.intensity_output = intensity_output
        self.return_base_spectrum = return_base_spectrum
        self.detach_base_ion_h = detach_base_ion_h
        self.proposal_selection_only = proposal_selection_only
        self.separate_proposal_encoder = separate_proposal_encoder
        self.share_proposal_across_ev = share_proposal_across_ev
        self.ion_mode_proposal_binary_logits = ion_mode_proposal_binary_logits
        self.min_ion_mode_relative_probability = min_ion_mode_relative_probability
        self.ion_h_support_binary_logits = ion_h_support_binary_logits
        self.gate_independent_intensity_with_support = gate_independent_intensity_with_support
        self.min_ion_h_relative_probability = min_ion_h_relative_probability
        self.min_ion_h_absolute_probability = min_ion_h_absolute_probability
        self.apply_channel_selection_in_training = apply_channel_selection_in_training
        self.evaluation_ion_h_top_k = 0
        self.evaluation_joint_channel_relative_threshold = 0.0
        self.proposal_num_mixtures = proposal_num_mixtures
        # Every component fans out into num_ion_modes * num_h_shifts candidate ions:
        # charge retention (CRF) and optionally charge migration (CMF), crossed with
        # hydrogen transfers from -max_h_transfer to +max_h_transfer. Keep the CMF
        # column when enabled even if every configured adduct marks it invalid: this
        # fixed width preserves intensity-head and checkpoint shapes across adduct sets.
        self.num_ion_modes = 2 if include_cmf else 1
        self.num_h_shifts = 2 * max_h_transfer + 1

        self.precursor_encoder = GNN(
            node_feats_size=node_feature_dim,
            edge_feats_size=edge_feature_dim,
            hidden_size=precursor_hidden_dim,
            num_layers=precursor_num_layers,
            gnn_type=precursor_gnn_type,
            normalization=precursor_normalization,
            dropout=precursor_dropout,
            num_heads=precursor_num_heads,
        )
        self.proposal_encoder = (
            GNN(
                node_feats_size=node_feature_dim,
                edge_feats_size=edge_feature_dim,
                hidden_size=precursor_hidden_dim,
                num_layers=precursor_num_layers,
                gnn_type=precursor_gnn_type,
                normalization=precursor_normalization,
                dropout=precursor_dropout,
                num_heads=precursor_num_heads,
            )
            if separate_proposal_encoder
            else None
        )
        self.context_encoder = MassShatterContextEncoder(
            num_precursor_types=len(self.precursor_types),
            num_instrument_types=len(instrument_types),
            num_fragmentation_modes=len(self.fragmentation_modes),
            use_nce=use_nce,
            use_ace=use_ace,
            int_embedder=int_embedder,
            ce_insert_type=ce_insert_type,
            ce_insert_size=ce_insert_size,
            metadata_embedding_dim=metadata_embedding_dim,
            context_dim=context_dim,
            nce_max=nce_max,
            nce_mean=nce_mean,
            nce_std=nce_std,
            ace_max=ace_max,
            ace_mean=ace_mean,
            ace_std=ace_std,
        )
        # Bond scoring either reads the two endpoint embeddings directly, or refines a
        # dedicated bond state through bond<->atom message passing first.
        if bond_num_layers == 0:
            self.bond_break_head = EndpointBondBreakHead(
                precursor_hidden_dim,
                edge_feature_dim,
                context_dim,
                head_hidden_dim,
                head_dropout,
                num_mixtures=proposal_num_mixtures,
            )
        else:
            self.bond_break_head = BondFocusedBreakHead(
                atom_embedding_dim=precursor_hidden_dim,
                bond_feature_dim=edge_feature_dim,
                context_dim=context_dim,
                bond_state_dim=precursor_hidden_dim,
                head_hidden_dim=head_hidden_dim,
                num_layers=bond_num_layers,
                dropout=head_dropout,
                num_mixtures=proposal_num_mixtures,
            )
        self.proposal_mixture_head = (
            BernoulliMixtureWeightHead(
                atom_embedding_dim=precursor_hidden_dim,
                context_dim=context_dim,
                hidden_dim=head_hidden_dim,
                num_mixtures=proposal_num_mixtures,
                dropout=head_dropout,
            )
            if proposal_num_mixtures > 1
            else None
        )
        self.charge_head = AtomChargeAffinityHead(
            precursor_hidden_dim,
            context_dim,
            head_hidden_dim,
            head_dropout,
        )
        self.ion_mode_proposal_head: IonModeProposalHead | None = None
        if separate_ion_mode_proposal_head:
            with th.random.fork_rng(devices=[]):
                self.ion_mode_proposal_head = IonModeProposalHead(
                    precursor_hidden_dim,
                    context_dim,
                    head_hidden_dim,
                    self.num_ion_modes,
                    head_dropout,
                )
        self.mechanism_head = (
            SmartsMechanismHead(
                precursor_hidden_dim,
                context_dim,
                head_hidden_dim,
                head_dropout,
            )
            if use_smarts_templates
            else None
        )
        self.intensity_head = CandidateIntensityTransformerHead(
            fragment_embedding_dim=precursor_hidden_dim,
            context_dim=context_dim,
            model_dim=intensity_transformer_dim,
            num_layers=intensity_transformer_num_layers,
            num_heads=intensity_transformer_num_heads,
            feedforward_dim=intensity_transformer_feedforward_dim,
            num_inducing_points=intensity_transformer_num_inducing_points,
            atom_decoder_num_layers=intensity_atom_decoder_num_layers,
            num_ion_modes=self.num_ion_modes,
            num_h_shifts=self.num_h_shifts,
            dropout=intensity_transformer_dropout,
            use_base_probability_feature=use_base_probability_feature,
            separate_ion_h_support_head=separate_ion_h_support_head,
            predict_oos=False,
            initial_oos_probability=0.1,
        )
        self._setup_chemistry_buffers()

    def set_evaluation_ion_h_top_k(self, top_k: int) -> None:
        """Set evaluation-only fixed-k ion/H selection; zero disables it."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("evaluation ion/H top_k must be a nonnegative integer")
        self.evaluation_ion_h_top_k = top_k

    def set_evaluation_joint_channel_relative_threshold(self, threshold: float) -> None:
        """Set an evaluation-only spectrum-relative joint channel floor."""
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
        ):
            raise ValueError("evaluation joint channel threshold must be finite and in [0, 1]")
        self.evaluation_joint_channel_relative_threshold = float(threshold)

    def _setup_chemistry_buffers(self) -> None:
        """Precompute per-adduct mass and validity deltas.

        Turning an adduct string such as ``[M-H2O+H]+`` into an ion requires adding a
        mass offset and adding/removing element counts. Doing that per batch would mean
        re-parsing formulas on the hot path, so both are baked into buffers indexed by
        adduct. Element counts are only tracked for elements an adduct can *remove*
        (plus hydrogen, which H-transfer also consumes), because those are the only
        counts that can go negative and thereby invalidate a channel.
        """

        def parse_diff(formula: str) -> dict[str, int]:
            # Preserve the established validity buffers and checkpoint shapes. The
            # signed parser is used only by CMF degeneracy detection above.
            return parse_formula(formula) if formula else {}

        required_elements = {1}
        parsed_crf: list[dict[str, int]] = []
        parsed_cmf: list[dict[str, int]] = []
        for precursor_type in self.precursor_types:
            crf_counts = parse_diff(PREC_TYPE_TO_FORMULA_DIFF[precursor_type])
            cmf_counts = parse_diff(PREC_TYPE_TO_CMF_FORMULA_DIFF.get(precursor_type, ""))
            parsed_crf.append(crf_counts)
            parsed_cmf.append(cmf_counts)
            for element, count in {**crf_counts, **cmf_counts}.items():
                if count < 0:
                    required_elements.add(PERIODIC_TABLE.GetAtomicNumber(element))
        required_atomic_numbers = sorted(required_elements)
        self.required_atomic_number_values = tuple(required_atomic_numbers)
        # H is tracked in its own column because hydrogen transfer shifts it per channel.
        self.hydrogen_count_column = required_atomic_numbers.index(1)
        atomic_number_to_column = {
            atomic_number: column for column, atomic_number in enumerate(required_atomic_numbers)
        }

        def formula_deltas(parsed: list[dict[str, int]]) -> th.Tensor:
            """Lay element counts out as ``(num_precursor_types, num_tracked_elements)``."""
            deltas = th.zeros(
                (len(self.precursor_types), len(required_atomic_numbers)),
                dtype=th.long,
            )
            for precursor_index, counts in enumerate(parsed):
                for element, count in counts.items():
                    atomic_number = PERIODIC_TABLE.GetAtomicNumber(element)
                    if atomic_number in atomic_number_to_column:
                        deltas[precursor_index, atomic_number_to_column[atomic_number]] = count
            return deltas

        self.register_buffer(
            "required_atomic_numbers",
            th.tensor(required_atomic_numbers, dtype=th.long),
        )
        self.register_buffer("crf_formula_deltas", formula_deltas(parsed_crf))
        self.register_buffer("cmf_formula_deltas", formula_deltas(parsed_cmf))
        self.register_buffer(
            "crf_mass_deltas",
            th.tensor(
                [PREC_TYPE_TO_MASS_DIFF[p] for p in self.precursor_types],
                dtype=th.float64,
            ),
        )
        self.register_buffer(
            "cmf_mass_deltas",
            th.tensor(
                [PREC_TYPE_TO_CMF_MASS_DIFF.get(p, 0.0) for p in self.precursor_types],
                dtype=th.float64,
            ),
        )
        duplicate_crf_shifts = tuple(
            _cmf_duplicate_crf_shift(precursor_type, self.max_h_transfer)
            for precursor_type in self.precursor_types
        )
        self.cmf_duplicate_crf_shifts = duplicate_crf_shifts
        # CMF is invalid when absent or chemically equivalent to a reachable CRF state.
        self.register_buffer(
            "cmf_precursor_valid",
            th.tensor(
                [
                    precursor_type in PREC_TYPE_TO_CMF_MASS_DIFF and duplicate_shift is None
                    for precursor_type, duplicate_shift in zip(
                        self.precursor_types,
                        duplicate_crf_shifts,
                    )
                ],
                dtype=th.bool,
            ),
        )

    def _fragment_masses_and_valid_channels(
        self,
        mol_pyg: Batch,
        atom_indices: th.Tensor,
        atom_fragment_indices: th.Tensor,
        fragment_batch_indices: th.Tensor,
        num_fragments: int,
        precursor_type_indices: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Translate fragment atom sets into monoisotopic ion channels.

        Args:
            mol_pyg: Batched precursor graphs carrying the per-atom chemistry fields.
            atom_indices: Atom index for every sparse fragment membership.
            atom_fragment_indices: Fragment index for every membership.
            fragment_batch_indices: Spectrum index for every fragment.
            num_fragments: Number of fragment atom sets.
            precursor_type_indices: Adduct index per spectrum, shape ``(batch_size,)``.

        Returns:
            Tuple of ``(ion_mzs, valid_channels)``, both shaped
            ``(num_fragments, num_ion_modes, num_h_shifts)``. ``valid_channels`` is
            False wherever a channel would need more atoms of some element than the
            component actually contains (or a CMF channel the adduct does not define).
        """
        atom_masses = mol_pyg.atom_monoisotopic_masses
        atom_hydrogens = mol_pyg.atom_hydrogen_counts
        atom_charges = mol_pyg.atom_formal_charges
        atom_numbers = mol_pyg.atom_atomic_numbers

        # Heavy-atom mass plus its implicit/explicit hydrogens, corrected for any formal
        # charge already present in the precursor; summing over a component's atoms
        # gives the neutral mass of that fragment.
        per_atom_neutral_mass = (
            atom_masses
            + atom_hydrogens.to(atom_masses) * H_MASS
            - atom_charges.to(atom_masses) * ELECTRON_MASS
        )
        component_masses = atom_masses.new_zeros(num_fragments).index_add(
            0,
            atom_fragment_indices,
            per_atom_neutral_mass[atom_indices],
        )

        component_counts = th.zeros(
            (num_fragments, self.required_atomic_numbers.numel()),
            dtype=th.long,
            device=atom_indices.device,
        )
        # Count the tracked elements per component. Hydrogen is special: explicit H
        # atoms in the graph and the implicit H count of every heavy atom both count.
        membership_atomic_numbers = atom_numbers[atom_indices]
        membership_hydrogens = atom_hydrogens[atom_indices]
        for column, atomic_number in enumerate(self.required_atomic_number_values):
            values = (membership_atomic_numbers == atomic_number).long()
            if atomic_number == 1:
                values = values + membership_hydrogens
            component_counts[:, column].index_add_(0, atom_fragment_indices, values)

        component_precursor_types = precursor_type_indices[fragment_batch_indices]
        h_shifts = th.arange(
            -self.max_h_transfer,
            self.max_h_transfer + 1,
            device=component_masses.device,
        )
        # CRF channel: the fragment keeps the adduct, so apply the adduct's element and
        # mass deltas, then sweep the hydrogen-transfer offsets. A channel is invalid
        # once any tracked element count would go negative.
        crf_counts = (
            component_counts[:, None, :]
            + self.crf_formula_deltas[component_precursor_types, None, :]
        )
        crf_counts = crf_counts.expand(-1, self.num_h_shifts, -1).clone()
        crf_counts[:, :, self.hydrogen_count_column] += h_shifts
        crf_valid = (crf_counts >= 0).all(dim=2)
        crf_mzs = (
            component_masses[:, None]
            + self.crf_mass_deltas[component_precursor_types, None]
            + h_shifts.to(component_masses) * H_MASS
        )

        mz_channels = [crf_mzs]
        valid_channels = [crf_valid]
        if self.include_cmf:
            # CMF channel: the charge stays on this fragment while the proton/adduct
            # leaves. It is defined only for the zero-shift column (index
            # ``max_h_transfer``), so the remaining H columns stay invalid and its m/z
            # is broadcast purely to keep the channel tensors rectangular.
            cmf_counts = component_counts + self.cmf_formula_deltas[component_precursor_types]
            cmf_valid_base = (cmf_counts >= 0).all(dim=1)
            cmf_valid_base &= self.cmf_precursor_valid[component_precursor_types]
            cmf_valid = th.zeros_like(crf_valid)
            cmf_valid[:, self.max_h_transfer] = cmf_valid_base
            cmf_mzs = (
                component_masses[:, None] + self.cmf_mass_deltas[component_precursor_types, None]
            ).expand(-1, self.num_h_shifts)
            mz_channels.append(cmf_mzs)
            valid_channels.append(cmf_valid)
        return th.stack(mz_channels, dim=1), th.stack(valid_channels, dim=1)

    def _eligible_smarts_templates(
        self,
        templates_by_molecule: tuple[tuple[MassShatterTemplate, ...], ...],
        precursor_type_indices: th.Tensor,
        fragmentation_mode_indices: th.Tensor | None,
    ) -> tuple[tuple[MassShatterTemplate, ...], ...]:
        """Apply literature-supported adduct and fragmentation-mode guards."""
        precursor_indices = precursor_type_indices.detach().cpu().reshape(-1).tolist()
        if len(templates_by_molecule) != len(precursor_indices):
            raise ValueError("SMARTS templates must contain one sequence per precursor")

        fragmentation_modes: list[str | None] = [None] * len(precursor_indices)
        if fragmentation_mode_indices is not None and self.fragmentation_modes:
            mode_indices = fragmentation_mode_indices.detach().cpu().reshape(-1).tolist()
            if len(mode_indices) != len(precursor_indices):
                raise ValueError("fragmentation modes must contain one index per precursor")
            fragmentation_modes = [
                (
                    self.fragmentation_modes[index]
                    if 0 <= index < len(self.fragmentation_modes)
                    else None
                )
                for index in mode_indices
            ]

        return tuple(
            filter_mass_shatter_templates(
                templates,
                precursor_type=self.precursor_types[precursor_index],
                fragmentation_mode=fragmentation_mode,
            )
            for templates, precursor_index, fragmentation_mode in zip(
                templates_by_molecule,
                precursor_indices,
                fragmentation_modes,
            )
        )

    @staticmethod
    def _unique_component_memberships(
        candidates: MassShatterCandidateBatch,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Return canonical sparse memberships for one token per atom subset.

        The compiled candidate builder records the first component pathway that
        produced each atom set. Membership and boundary tensors from that
        representative are canonical: spectator cuts cannot alter either the atoms
        retained by the fragment or the precursor bonds crossing its boundary.
        """
        representatives = candidates.unique_representative_component_indices
        num_unique_components = candidates.num_unique_components
        if num_unique_components == 0:
            raise ValueError("candidate generation produced no unique components")

        representative_to_unique = representatives.new_full(
            (candidates.num_components,),
            -1,
        )
        representative_to_unique[representatives] = th.arange(
            num_unique_components,
            dtype=th.long,
            device=representatives.device,
        )

        atom_unique_indices = representative_to_unique[candidates.atom_component_indices]
        atom_keep = atom_unique_indices >= 0
        unique_atom_indices = candidates.atom_indices[atom_keep]
        atom_unique_indices = atom_unique_indices[atom_keep]

        boundary_unique_indices = representative_to_unique[candidates.boundary_component_indices]
        boundary_keep = boundary_unique_indices >= 0
        unique_boundary_bond_indices = candidates.boundary_bond_indices[boundary_keep]
        boundary_unique_indices = boundary_unique_indices[boundary_keep]
        boundary_counts = th.bincount(
            boundary_unique_indices,
            minlength=num_unique_components,
        )
        unique_batch_indices = candidates.component_batch_indices[representatives]
        return (
            unique_batch_indices,
            unique_atom_indices,
            atom_unique_indices,
            unique_boundary_bond_indices,
            boundary_unique_indices,
            boundary_counts,
        )

    @staticmethod
    def _shared_proposal_layout(
        mol_pyg: Batch,
        group_ids: th.Tensor,
    ) -> dict[str, th.Tensor]:
        """Compact repeated precursor graphs and map proposal outputs back."""
        batch_size = int(mol_pyg.ptr.numel() - 1)
        group_ids = group_ids.reshape(-1)
        if group_ids.shape[0] != batch_size:
            raise ValueError("proposal_group_ids must contain one id per spectrum")

        # 1. Build stable first-occurrence group indices.
        _, sorted_inverse = th.unique(group_ids, sorted=True, return_inverse=True)
        num_shared = int(sorted_inverse.max().item()) + 1
        spectrum_indices = th.arange(batch_size, device=group_ids.device)
        first_by_sorted_group = spectrum_indices.new_full((num_shared,), batch_size).scatter_reduce(
            0,
            sorted_inverse,
            spectrum_indices,
            reduce="amin",
            include_self=True,
        )
        sorted_group_order = th.argsort(first_by_sorted_group)
        sorted_to_stable = th.empty_like(sorted_group_order)
        sorted_to_stable[sorted_group_order] = th.arange(num_shared, device=group_ids.device)
        spectrum_to_shared = sorted_to_stable[sorted_inverse]
        representatives = first_by_sorted_group[sorted_group_order]

        # 2. Compact representative atom graphs.
        atom_counts = mol_pyg.ptr[1:] - mol_pyg.ptr[:-1]
        shared_atom_counts = atom_counts[representatives]
        _validate_tensor_condition(
            (atom_counts == shared_atom_counts[spectrum_to_shared]).all(),
            "all spectra in a proposal group must have identical atom counts",
        )
        shared_atom_ptr = th.cat((shared_atom_counts.new_zeros(1), shared_atom_counts.cumsum(0)))
        atom_indices = th.arange(mol_pyg.x.shape[0], device=mol_pyg.x.device)
        atom_local_indices = atom_indices - mol_pyg.ptr[mol_pyg.batch]
        atom_to_shared = shared_atom_ptr[spectrum_to_shared[mol_pyg.batch]] + atom_local_indices
        representative_atom_mask = th.isin(mol_pyg.batch, representatives)
        representative_atom_indices = atom_indices[representative_atom_mask]
        old_to_compact_atom = atom_indices.new_full((atom_indices.numel(),), -1)
        old_to_compact_atom[representative_atom_indices] = th.arange(
            representative_atom_indices.numel(), device=atom_indices.device
        )
        representative_edge_mask = representative_atom_mask[mol_pyg.edge_index[0]]
        shared_edge_index = old_to_compact_atom[mol_pyg.edge_index[:, representative_edge_mask]]
        shared_atom_batch = th.repeat_interleave(
            th.arange(num_shared, device=group_ids.device),
            shared_atom_counts,
        )

        # 3. Compact the single-entry chemical-bond view and build its fan-out map.
        bond_batch = mol_pyg.batch[mol_pyg.chemical_bond_index[0]]
        bond_counts = th.bincount(bond_batch, minlength=batch_size)
        shared_bond_counts = bond_counts[representatives]
        _validate_tensor_condition(
            (bond_counts == shared_bond_counts[spectrum_to_shared]).all(),
            "all spectra in a proposal group must have identical bond counts",
        )
        shared_bond_ptr = th.cat((shared_bond_counts.new_zeros(1), shared_bond_counts.cumsum(0)))
        bond_indices = th.arange(bond_batch.numel(), device=bond_batch.device)
        bond_ptr = th.cat((bond_counts.new_zeros(1), bond_counts.cumsum(0)))
        bond_local_indices = bond_indices - bond_ptr[bond_batch]
        bond_to_shared = shared_bond_ptr[spectrum_to_shared[bond_batch]] + bond_local_indices
        representative_bond_mask = th.isin(bond_batch, representatives)
        shared_chemical_bond_index = old_to_compact_atom[
            mol_pyg.chemical_bond_index[:, representative_bond_mask]
        ]
        shared_bond_batch = th.repeat_interleave(
            th.arange(num_shared, device=group_ids.device),
            shared_bond_counts,
        )

        # Sharing is only valid for repeated views of the exact same precursor graph.
        _validate_tensor_condition(
            (mol_pyg.x == mol_pyg.x[representative_atom_indices][atom_to_shared]).all(),
            "all spectra in a proposal group must have identical atom features",
        )
        _validate_tensor_condition(
            (
                mol_pyg.chemical_bond_attr
                == mol_pyg.chemical_bond_attr[representative_bond_mask][bond_to_shared]
            ).all(),
            "all spectra in a proposal group must have identical bond features",
        )
        local_bond_index = mol_pyg.chemical_bond_index - mol_pyg.ptr[bond_batch].unsqueeze(0)
        shared_local_bond_index = shared_chemical_bond_index - shared_atom_ptr[
            shared_bond_batch
        ].unsqueeze(0)
        _validate_tensor_condition(
            (local_bond_index == shared_local_bond_index[:, bond_to_shared]).all(),
            "all spectra in a proposal group must have identical bond topology",
        )

        return {
            "representatives": representatives,
            "spectrum_to_shared": spectrum_to_shared,
            "representative_atom_indices": representative_atom_indices,
            "atom_to_shared": atom_to_shared,
            "shared_atom_batch": shared_atom_batch,
            "representative_edge_mask": representative_edge_mask,
            "shared_edge_index": shared_edge_index,
            "representative_bond_mask": representative_bond_mask,
            "bond_to_shared": bond_to_shared,
            "shared_chemical_bond_index": shared_chemical_bond_index,
            "shared_bond_batch": shared_bond_batch,
        }

    def forward(
        self,
        *,
        mol_pyg: Batch,
        spec_prec_type: th.Tensor,
        spec_inst_type: th.Tensor | None = None,
        spec_frag_mode: th.Tensor | None = None,
        spec_nce: th.Tensor | None = None,
        spec_nce_batch_idxs: th.Tensor | None = None,
        spec_ace: th.Tensor | None = None,
        spec_ace_batch_idxs: th.Tensor | None = None,
        static_graph: MassShatterStaticGraph | None = None,
        proposal_group_ids: th.Tensor | None = None,
    ) -> dict[str, th.Tensor]:
        """Generate a sparse monoisotopic spectrum for every precursor.

        Args:
            mol_pyg: Batched precursor graphs. Beyond ``x``/``edge_index``/``edge_attr``
                these must carry the MASShatter chemistry fields listed in
                ``required_graph_fields`` (chemical-bond view of the graph plus per-atom
                masses, hydrogen counts, formal charges, and atomic numbers).
            spec_prec_type: Adduct index per spectrum, shape ``(batch_size,)``.
            spec_inst_type: Instrument index per spectrum, if the context uses it.
            spec_frag_mode: Fragmentation-mode index per spectrum, if the context uses it.
            spec_nce: Normalized collision energies, ragged over spectra.
            spec_nce_batch_idxs: Spectrum index per NCE value.
            spec_ace: Absolute collision energies, ragged over spectra.
            spec_ace_batch_idxs: Spectrum index per ACE value.
            static_graph: CPU copy of the graph indices kept across Lightning's device
                transfer, so mask selection only has to move the bond logits back.

        Returns:
            Dictionary with the sparse prediction contract plus diagnostics:
            - "pred_mzs"/"pred_batch_idxs" and exactly one of "pred_logprobs" or
              "pred_intensities": one entry per valid (component, ion mode, H shift)
              pathway, shape ``(num_pathways,)``.
            - "bond_break_logits"/"atom_charge_logits": raw head outputs.
            - remaining keys: per-spectrum counters and candidate-generation timings.

        Raises:
            ValueError: If ``mol_pyg`` lacks the MASShatter chemistry fields.
        """
        required_graph_fields = (
            "chemical_bond_index",
            "chemical_bond_attr",
            "atom_atomic_numbers",
            "atom_hydrogen_counts",
            "atom_formal_charges",
            "atom_monoisotopic_masses",
            "batch",
            "ptr",
        )
        missing = [field for field in required_graph_fields if not hasattr(mol_pyg, field)]
        if missing:
            raise ValueError(
                "mol_pyg is missing MASShatter chemistry fields; regenerate molecular "
                f"preprocessing caches: {missing}"
            )
        batch_size = int(mol_pyg.ptr.numel() - 1)
        precursor_type_indices = spec_prec_type.reshape(-1)
        context = self.context_encoder(
            batch_size=batch_size,
            precursor_types=precursor_type_indices,
            instrument_types=spec_inst_type,
            fragmentation_modes=spec_frag_mode,
            nce=spec_nce,
            nce_batch_indices=spec_nce_batch_idxs,
            ace=spec_ace,
            ace_batch_indices=spec_ace_batch_idxs,
        )

        # 1. Encode each intact precursor once and score every chemical bond.
        layout = None
        if self.share_proposal_across_ev:
            if proposal_group_ids is None:
                raise ValueError("share_proposal_across_ev requires proposal_group_ids")
            layout = self._shared_proposal_layout(mol_pyg, proposal_group_ids)
            shared_atom_indices = layout["representative_atom_indices"]
            shared_atom_embeddings = self.precursor_encoder(
                mol_pyg.x[shared_atom_indices],
                layout["shared_atom_batch"],
                layout["shared_edge_index"],
                mol_pyg.edge_attr[layout["representative_edge_mask"]],
            )
            atom_embeddings = shared_atom_embeddings[layout["atom_to_shared"]]
        else:
            atom_embeddings = self.precursor_encoder(
                mol_pyg.x,
                mol_pyg.batch,
                mol_pyg.edge_index,
                mol_pyg.edge_attr,
            )
        # ``chemical_bond_index`` is the undirected single-entry-per-bond view of the
        # graph (``edge_index`` is directed), so either endpoint identifies the molecule.
        bond_batch_indices = mol_pyg.batch[mol_pyg.chemical_bond_index[0]]
        if self.share_proposal_across_ev:
            assert layout is not None
            representatives = layout["representatives"]
            spectrum_to_shared = layout["spectrum_to_shared"]
            _validate_tensor_condition(
                (
                    precursor_type_indices
                    == precursor_type_indices[representatives][spectrum_to_shared]
                ).all(),
                "all spectra in a proposal group must have the same precursor type",
            )
            if spec_inst_type is not None:
                _validate_tensor_condition(
                    (
                        spec_inst_type.reshape(-1)
                        == spec_inst_type.reshape(-1)[representatives][spectrum_to_shared]
                    ).all(),
                    "all spectra in a proposal group must have the same instrument type",
                )
            if spec_frag_mode is not None:
                _validate_tensor_condition(
                    (
                        spec_frag_mode.reshape(-1)
                        == spec_frag_mode.reshape(-1)[representatives][spectrum_to_shared]
                    ).all(),
                    "all spectra in a proposal group must have the same fragmentation mode",
                )
            shared_context = self.context_encoder(
                batch_size=representatives.numel(),
                precursor_types=precursor_type_indices[representatives],
                instrument_types=(
                    None if spec_inst_type is None else spec_inst_type[representatives]
                ),
                fragmentation_modes=(
                    None if spec_frag_mode is None else spec_frag_mode[representatives]
                ),
                nce=None,
                nce_batch_indices=None,
                ace=None,
                ace_batch_indices=None,
                include_collision_energy=False,
            )
            shared_atom_indices = layout["representative_atom_indices"]
            if self.proposal_encoder is None:
                shared_proposal_atom_embeddings = atom_embeddings[shared_atom_indices]
                if self.proposal_selection_only:
                    shared_proposal_atom_embeddings = shared_proposal_atom_embeddings.detach()
            else:
                shared_proposal_atom_embeddings = self.proposal_encoder(
                    mol_pyg.x[shared_atom_indices],
                    layout["shared_atom_batch"],
                    layout["shared_edge_index"],
                    mol_pyg.edge_attr[layout["representative_edge_mask"]],
                )
            if self.ion_mode_proposal_head is not None:
                proposal_atom_embeddings = shared_proposal_atom_embeddings[layout["atom_to_shared"]]
                proposal_context = context.detach() if self.proposal_selection_only else context
            shared_proposal_context = (
                shared_context.detach() if self.proposal_selection_only else shared_context
            )
            shared_bond_logits = self.bond_break_head(
                shared_proposal_atom_embeddings,
                layout["shared_chemical_bond_index"],
                mol_pyg.chemical_bond_attr[layout["representative_bond_mask"]],
                context=shared_proposal_context,
                bond_batch_indices=layout["shared_bond_batch"],
            )
            bond_logits = shared_bond_logits[layout["bond_to_shared"]]
            shared_atom_charge_logits = self.charge_head(
                shared_proposal_atom_embeddings,
                context=shared_proposal_context,
                atom_batch_indices=layout["shared_atom_batch"],
            )
            atom_charge_logits = shared_atom_charge_logits[layout["atom_to_shared"]]
            if self.proposal_mixture_head is None:
                mixture_log_weights = None
            else:
                shared_mixture_log_weights = self.proposal_mixture_head(
                    shared_proposal_atom_embeddings,
                    layout["shared_atom_batch"],
                    shared_proposal_context,
                    batch_size=representatives.numel(),
                )
                mixture_log_weights = shared_mixture_log_weights[spectrum_to_shared]
        else:
            # V1/V2 detach proposal inputs from the intensity encoder. V3 instead learns
            # an independent molecular representation from proposal supervision.
            if self.proposal_encoder is None:
                proposal_atom_embeddings = (
                    atom_embeddings.detach() if self.proposal_selection_only else atom_embeddings
                )
            else:
                proposal_atom_embeddings = self.proposal_encoder(
                    mol_pyg.x,
                    mol_pyg.batch,
                    mol_pyg.edge_index,
                    mol_pyg.edge_attr,
                )
            proposal_context = context.detach() if self.proposal_selection_only else context
            bond_logits = self.bond_break_head(
                proposal_atom_embeddings,
                mol_pyg.chemical_bond_index,
                mol_pyg.chemical_bond_attr,
                context=proposal_context,
                bond_batch_indices=bond_batch_indices,
            )
            mixture_log_weights = (
                self.proposal_mixture_head(
                    proposal_atom_embeddings,
                    mol_pyg.batch,
                    proposal_context,
                    batch_size=batch_size,
                )
                if self.proposal_mixture_head is not None
                else None
            )
            atom_charge_logits = self.charge_head(
                proposal_atom_embeddings,
                context=proposal_context,
                atom_batch_indices=mol_pyg.batch,
            )

        # 2. Select global masks and construct sparse component pathways.
        # Mask identities are picked from detached logits on CPU; the returned log
        # probabilities are recomputed from the live logits, so gradients still reach
        # every cut/intact bond decision. Exploration masks are training-only.
        smarts_templates: tuple[tuple[MassShatterTemplate, ...], ...] = tuple(
            () for _ in range(batch_size)
        )
        if (
            self.use_smarts_templates
            and static_graph is not None
            and static_graph.smarts_templates is not None
        ):
            smarts_templates = self._eligible_smarts_templates(
                static_graph.smarts_templates,
                precursor_type_indices,
                spec_frag_mode,
            )
        candidates = build_mass_shatter_candidates(
            bond_logits,
            mol_pyg.chemical_bond_index,
            mol_pyg.batch,
            mol_pyg.ptr,
            max_masks=self.max_masks,
            min_mask_relative_probability=self.min_mask_relative_probability,
            include_single_bond_cuts=self.include_single_bond_cuts,
            max_unique_components=(self.max_masks if self.unique_fragment_mode else None),
            max_enumerated_masks=(self.max_enumerated_masks if self.unique_fragment_mode else None),
            mask_selection_threads=self.mask_selection_threads,
            static_graph=static_graph,
            smarts_templates=smarts_templates,
            mixture_log_weights=mixture_log_weights,
            candidate_group_ids=(layout["spectrum_to_shared"] if layout is not None else None),
        )

        # 3. Score charge retention. In unique-fragment mode, masks remain latent
        # pathways and are marginalized before the expensive candidate transformer.
        num_masks = candidates.mask_log_probabilities.shape[0]
        chemical_bond_embeddings = (
            atom_embeddings[mol_pyg.chemical_bond_index[0]]
            + atom_embeddings[mol_pyg.chemical_bond_index[1]]
        ) / 2.0
        if self.mechanism_head is None:
            mechanism_log_probabilities = bond_logits.new_zeros(candidates.num_mechanisms)
        else:
            mechanism_boundary_embeddings = pool_boundary_bond_embeddings(
                chemical_bond_embeddings,
                candidates.mechanism_boundary_bond_indices,
                candidates.boundary_mechanism_indices,
                num_components=candidates.num_mechanisms,
            )
            mechanism_batch_indices = candidates.mask_batch_indices[
                candidates.mechanism_mask_indices
            ]
            mechanism_logits = self.mechanism_head(
                mechanism_boundary_embeddings,
                candidates.mechanism_rule_indices,
                context[mechanism_batch_indices],
            )
            mechanism_log_normalizers = _segment_logsumexp(
                mechanism_logits,
                candidates.mechanism_mask_indices,
                num_masks,
            )
            mechanism_log_probabilities = (
                mechanism_logits - mechanism_log_normalizers[candidates.mechanism_mask_indices]
            )
        unique_fragment_effective_count = bond_logits.new_zeros(batch_size)
        unique_fragment_top1_mass_fraction = bond_logits.new_zeros(batch_size)
        duplicate_pathway_probability_fraction = bond_logits.new_zeros(batch_size)
        fragment_count_before_cutoff = bond_logits.new_zeros(batch_size)
        fragment_count_after_cutoff = bond_logits.new_zeros(batch_size)
        retained_fragment_probability = bond_logits.new_zeros(batch_size)
        fragment_cutoff_pruned_fraction = bond_logits.new_zeros(batch_size)
        learned_ion_mode_proposal_logits: th.Tensor | None = None
        if self.unique_fragment_mode:
            (
                fragment_batch_indices,
                fragment_atom_indices,
                atom_fragment_indices,
                fragment_boundary_bond_indices,
                boundary_fragment_indices,
                fragment_cut_counts,
            ) = self._unique_component_memberships(candidates)
            num_fragments = candidates.num_unique_components
            charge_log_probabilities = latent_pathway_charge_log_probabilities(
                atom_charge_logits,
                fragment_atom_indices,
                atom_fragment_indices,
                candidates.component_unique_indices,
                candidates.component_mechanism_indices,
                num_unique_components=num_fragments,
                num_masks=candidates.num_mechanisms,
                pooling=self.charge_pooling,
            )
            # 3a. Sum every charged pathway that produces the same precursor atom set.
            component_log_probabilities = (
                candidates.mask_log_probabilities[candidates.component_mask_indices]
                + mechanism_log_probabilities[candidates.component_mechanism_indices]
                + charge_log_probabilities
            )
            fragment_log_probabilities = _segment_logsumexp(
                component_log_probabilities,
                candidates.component_unique_indices,
                num_fragments,
            )
            proposal_log_normalizers = _segment_logsumexp(
                fragment_log_probabilities,
                fragment_batch_indices,
                batch_size,
            )
            base_fragment_log_probabilities = (
                fragment_log_probabilities - proposal_log_normalizers[fragment_batch_indices]
            )
            best_pathway_log_probabilities = component_log_probabilities.new_full(
                (num_fragments,),
                -th.inf,
            ).scatter_reduce(
                0,
                candidates.component_unique_indices,
                component_log_probabilities,
                reduce="amax",
                include_self=True,
            )
            retained_fragment_log_probability = _segment_logsumexp(
                fragment_log_probabilities,
                fragment_batch_indices,
                batch_size,
            )
            best_pathway_log_probability = _segment_logsumexp(
                best_pathway_log_probabilities,
                fragment_batch_indices,
                batch_size,
            )
            duplicate_pathway_probability_fraction = (
                -th.expm1(best_pathway_log_probability - retained_fragment_log_probability)
            ).clamp(min=0.0, max=1.0)

            # Prune low-probability unique atom sets only after all duplicate
            # pathways have been marginalized. The hard decision is detached, while
            # probabilities for retained fragments stay live for proposal training.
            fragment_keep = _relative_fragment_keep_mask(
                fragment_log_probabilities,
                fragment_batch_indices,
                batch_size,
                self.min_fragment_relative_probability,
            )
            fragment_count_before_cutoff = th.bincount(
                fragment_batch_indices,
                minlength=batch_size,
            ).to(bond_logits)
            fragment_count_after_cutoff = th.bincount(
                fragment_batch_indices[fragment_keep],
                minlength=batch_size,
            ).to(bond_logits)
            retained_fragment_probability = bond_logits.new_zeros(batch_size).index_add(
                0,
                fragment_batch_indices,
                base_fragment_log_probabilities.exp() * fragment_keep.to(bond_logits),
            )
            fragment_cutoff_pruned_fraction = (
                1.0 - fragment_count_after_cutoff / fragment_count_before_cutoff
            )

            old_to_new_fragment = fragment_keep.to(th.long).cumsum(0) - 1
            old_to_new_fragment[~fragment_keep] = -1
            atom_keep = fragment_keep[atom_fragment_indices]
            fragment_atom_indices = fragment_atom_indices[atom_keep]
            atom_fragment_indices = old_to_new_fragment[atom_fragment_indices[atom_keep]]
            boundary_keep = fragment_keep[boundary_fragment_indices]
            fragment_boundary_bond_indices = fragment_boundary_bond_indices[boundary_keep]
            boundary_fragment_indices = old_to_new_fragment[
                boundary_fragment_indices[boundary_keep]
            ]
            fragment_batch_indices = fragment_batch_indices[fragment_keep]
            fragment_cut_counts = fragment_cut_counts[fragment_keep]
            fragment_log_probabilities = fragment_log_probabilities[fragment_keep]
            num_fragments = int(fragment_log_probabilities.shape[0])
            retained_proposal_log_normalizers = _segment_logsumexp(
                fragment_log_probabilities,
                fragment_batch_indices,
                batch_size,
            )
            base_fragment_log_probabilities = (
                fragment_log_probabilities
                - retained_proposal_log_normalizers[fragment_batch_indices]
            )
            fragment_embeddings, complement_embeddings = pool_component_and_complement_embeddings(
                atom_embeddings,
                fragment_atom_indices,
                atom_fragment_indices,
                fragment_batch_indices,
                mol_pyg.batch,
                num_components=num_fragments,
                num_molecules=batch_size,
            )
            boundary_embeddings = pool_boundary_bond_embeddings(
                chemical_bond_embeddings,
                fragment_boundary_bond_indices,
                boundary_fragment_indices,
                num_components=num_fragments,
            )
            fragment_sizes = th.bincount(
                atom_fragment_indices,
                minlength=num_fragments,
            )
            if self.ion_mode_proposal_head is not None:
                (
                    proposal_fragment_embeddings,
                    proposal_complement_embeddings,
                ) = pool_component_and_complement_embeddings(
                    proposal_atom_embeddings,
                    fragment_atom_indices,
                    atom_fragment_indices,
                    fragment_batch_indices,
                    mol_pyg.batch,
                    num_components=num_fragments,
                    num_molecules=batch_size,
                )
                proposal_chemical_bond_embeddings = (
                    proposal_atom_embeddings[mol_pyg.chemical_bond_index[0]]
                    + proposal_atom_embeddings[mol_pyg.chemical_bond_index[1]]
                ) / 2.0
                proposal_boundary_embeddings = pool_boundary_bond_embeddings(
                    proposal_chemical_bond_embeddings,
                    fragment_boundary_bond_indices,
                    boundary_fragment_indices,
                    num_components=num_fragments,
                )
                learned_ion_mode_proposal_logits = self.ion_mode_proposal_head(
                    proposal_fragment_embeddings,
                    proposal_complement_embeddings,
                    proposal_boundary_embeddings,
                    fragment_cut_counts,
                    fragment_sizes,
                    proposal_context[fragment_batch_indices],
                )
            fragment_indices = th.arange(
                num_fragments,
                dtype=th.long,
                device=fragment_log_probabilities.device,
            )
            intensity_outputs = self.intensity_head(
                fragment_embeddings,
                complement_embeddings,
                boundary_embeddings,
                fragment_cut_counts,
                fragment_sizes,
                th.zeros_like(fragment_log_probabilities),
                fragment_log_probabilities,
                fragment_batch_indices,
                fragment_indices,
                batch_size=batch_size,
                num_masks=num_fragments,
                context=context[fragment_batch_indices],
                atom_embeddings=atom_embeddings,
                atom_indices=fragment_atom_indices,
                atom_fragment_indices=atom_fragment_indices,
                return_support_logits=(self.intensity_head.ion_h_support_head is not None),
            )
            _, fragment_residuals, ion_h_logits = intensity_outputs[:3]
            ion_h_support_logits = (
                intensity_outputs[3]
                if self.intensity_head.ion_h_support_head is not None
                else ion_h_logits
            )
            if not self.use_mask_residuals:
                fragment_residuals = th.zeros_like(fragment_residuals)
            corrected_fragment_logits = (
                fragment_residuals
                if self.proposal_selection_only
                else fragment_log_probabilities + fragment_residuals
            )
            fragment_log_normalizers = _segment_logsumexp(
                corrected_fragment_logits,
                fragment_batch_indices,
                batch_size,
            )
            fragment_log_probabilities = (
                corrected_fragment_logits - fragment_log_normalizers[fragment_batch_indices]
            )
            normalized_fragment_log_probabilities = fragment_log_probabilities
            normalized_fragment_probabilities = normalized_fragment_log_probabilities.exp()
            fragment_entropy_terms = (
                -normalized_fragment_probabilities * normalized_fragment_log_probabilities
            )
            unique_fragment_entropy = fragment_entropy_terms.new_zeros(batch_size).index_add(
                0,
                fragment_batch_indices,
                fragment_entropy_terms,
            )
            unique_fragment_effective_count = unique_fragment_entropy.exp()
            unique_fragment_top1_mass_fraction = (
                normalized_fragment_log_probabilities.new_full(
                    (batch_size,),
                    -th.inf,
                )
                .scatter_reduce(
                    0,
                    fragment_batch_indices,
                    normalized_fragment_log_probabilities,
                    reduce="amax",
                    include_self=True,
                )
                .exp()
            )
            ion_mzs, valid_channels = self._fragment_masses_and_valid_channels(
                mol_pyg,
                fragment_atom_indices,
                atom_fragment_indices,
                fragment_batch_indices,
                num_fragments,
                precursor_type_indices,
            )
        else:
            # 3b. Legacy pathway-token model: contextualize every component occurrence.
            charge_log_probabilities = component_charge_log_probabilities(
                atom_charge_logits,
                candidates.atom_indices,
                candidates.atom_component_indices,
                candidates.component_mechanism_indices,
                pooling=self.charge_pooling,
            )
            num_fragments = candidates.num_components
            fragment_batch_indices = candidates.component_batch_indices
            fragment_cut_counts = candidates.component_cut_counts
            fragment_embeddings, complement_embeddings = pool_component_and_complement_embeddings(
                atom_embeddings,
                candidates.atom_indices,
                candidates.atom_component_indices,
                fragment_batch_indices,
                mol_pyg.batch,
                num_components=num_fragments,
                num_molecules=batch_size,
            )
            boundary_embeddings = pool_boundary_bond_embeddings(
                chemical_bond_embeddings,
                candidates.boundary_bond_indices,
                candidates.boundary_component_indices,
                num_components=num_fragments,
            )
            fragment_sizes = th.bincount(
                candidates.atom_component_indices,
                minlength=num_fragments,
            )
            original_component_mask_log_probabilities = candidates.mask_log_probabilities[
                candidates.component_mask_indices
            ]
            component_mechanism_log_probabilities = mechanism_log_probabilities[
                candidates.component_mechanism_indices
            ]
            base_fragment_log_probabilities = (
                original_component_mask_log_probabilities
                + component_mechanism_log_probabilities
                + charge_log_probabilities
            )
            intensity_outputs = self.intensity_head(
                fragment_embeddings,
                complement_embeddings,
                boundary_embeddings,
                fragment_cut_counts,
                fragment_sizes,
                charge_log_probabilities + component_mechanism_log_probabilities,
                original_component_mask_log_probabilities,
                fragment_batch_indices,
                candidates.component_mask_indices,
                batch_size=batch_size,
                num_masks=num_masks,
                context=context[fragment_batch_indices],
                atom_embeddings=atom_embeddings,
                atom_indices=candidates.atom_indices,
                atom_fragment_indices=candidates.atom_component_indices,
                return_support_logits=(self.intensity_head.ion_h_support_head is not None),
            )
            component_residuals, mask_residuals, ion_h_logits = intensity_outputs[:3]
            ion_h_support_logits = (
                intensity_outputs[3]
                if self.intensity_head.ion_h_support_head is not None
                else ion_h_logits
            )
            if not self.use_mask_residuals:
                mask_residuals = th.zeros_like(mask_residuals)

            mask_logits = candidates.mask_log_probabilities + mask_residuals
            retained_log_probabilities = _segment_logsumexp(
                candidates.mask_log_probabilities,
                candidates.mask_batch_indices,
                batch_size,
            )
            base_fragment_log_probabilities = (
                base_fragment_log_probabilities - retained_log_probabilities[fragment_batch_indices]
            )
            mask_log_normalizers = _segment_logsumexp(
                mask_logits,
                candidates.mask_batch_indices,
                batch_size,
            )
            mask_log_probabilities = (
                mask_logits - mask_log_normalizers[candidates.mask_batch_indices]
            )
            component_charge_logits = charge_log_probabilities + component_residuals
            component_log_normalizers = _segment_logsumexp(
                component_charge_logits,
                candidates.component_mechanism_indices,
                candidates.num_mechanisms,
            )
            charge_log_probabilities = (
                component_charge_logits
                - component_log_normalizers[candidates.component_mechanism_indices]
            )
            fragment_log_probabilities = (
                mask_log_probabilities[candidates.component_mask_indices]
                + component_mechanism_log_probabilities
                + charge_log_probabilities
            )
            ion_mzs, valid_channels = self._fragment_masses_and_valid_channels(
                mol_pyg,
                candidates.atom_indices,
                candidates.atom_component_indices,
                fragment_batch_indices,
                num_fragments,
                precursor_type_indices,
            )

        # 4. Separate full chemical support from selected spectral output.
        dense_intensity_log_probabilities = masked_ion_hydrogen_log_probabilities(
            ion_h_logits,
            valid_channels,
        )
        if self.intensity_output == "independent":
            dense_channel_weights = F.softplus(
                corrected_fragment_logits[:, None, None] + ion_h_logits
            )
            if self.gate_independent_intensity_with_support:
                if not self.ion_h_support_binary_logits:
                    raise ValueError(
                        "independent intensity support gating requires binary support logits"
                    )
                dense_channel_weights = dense_channel_weights * ion_h_support_logits.sigmoid()
            dense_channel_log_weights = th.log(
                dense_channel_weights.clamp_min(th.finfo(dense_channel_weights.dtype).tiny)
            ).masked_fill(~valid_channels, -th.inf)
        else:
            dense_channel_log_weights = (
                fragment_log_probabilities[:, None, None] + dense_intensity_log_probabilities
            )
        valid_ion_modes = valid_channels.any(dim=2)
        if learned_ion_mode_proposal_logits is None:
            ion_mode_proposal_logits = _ion_mode_logits_from_ion_h_support(
                ion_h_support_logits,
                valid_channels,
                binary_logits=self.ion_h_support_binary_logits,
            )
            ion_mode_proposal_binary_logits = self.ion_h_support_binary_logits
        else:
            ion_mode_proposal_logits = learned_ion_mode_proposal_logits
            ion_mode_proposal_binary_logits = self.ion_mode_proposal_binary_logits
        apply_selection = not self.training or self.apply_channel_selection_in_training
        mode_keep = (
            _relative_channel_keep_mask(
                ion_mode_proposal_logits,
                valid_ion_modes,
                self.min_ion_mode_relative_probability,
                binary_logits=ion_mode_proposal_binary_logits,
            )
            if apply_selection
            else valid_ion_modes
        )
        mode_selected_channels = valid_channels & mode_keep.unsqueeze(2)
        if not self.training and self.evaluation_ion_h_top_k > 0:
            spectral_output_channels = _fixed_top_k_channel_keep_mask(
                ion_h_support_logits,
                mode_selected_channels,
                self.evaluation_ion_h_top_k,
            )
        elif apply_selection:
            spectral_output_channels = _relative_channel_keep_mask(
                ion_h_support_logits,
                mode_selected_channels,
                self.min_ion_h_relative_probability,
                binary_logits=self.ion_h_support_binary_logits,
            )
        else:
            spectral_output_channels = valid_channels
        if not self.training and self.evaluation_joint_channel_relative_threshold > 0.0:
            spectral_output_channels &= _relative_joint_channel_keep_mask(
                dense_channel_log_weights,
                spectral_output_channels,
                fragment_batch_indices,
                batch_size,
                self.evaluation_joint_channel_relative_threshold,
            )
        if not self.training and self.min_ion_h_absolute_probability > 0.0:
            spectral_output_channels &= _absolute_binary_channel_keep_mask(
                ion_h_support_logits,
                spectral_output_channels,
                fragment_batch_indices,
                batch_size,
                self.min_ion_h_absolute_probability,
            )
        probability_output_channels = (
            valid_channels if self.intensity_output == "independent" else spectral_output_channels
        )
        ion_h_log_probabilities = masked_ion_hydrogen_log_probabilities(
            ion_h_logits,
            probability_output_channels,
        )
        pathway_log_probabilities = (
            fragment_log_probabilities[:, None, None] + ion_h_log_probabilities
        )
        base_ion_h_log_probabilities = (
            ion_h_log_probabilities.detach() if self.detach_base_ion_h else ion_h_log_probabilities
        )
        base_pathway_log_probabilities = (
            base_fragment_log_probabilities[:, None, None] + base_ion_h_log_probabilities
        )
        full_flat_valid = valid_channels.flatten()
        selected_flat_valid = spectral_output_channels.flatten()
        full_candidate_mzs = ion_mzs.flatten()[full_flat_valid]
        pred_mzs = ion_mzs.flatten()[selected_flat_valid]
        pred_logprobs = pathway_log_probabilities.flatten()[selected_flat_valid]
        pred_intensities = (
            dense_channel_weights.flatten()[selected_flat_valid]
            if self.intensity_output == "independent"
            else None
        )
        expanded_batch_indices = fragment_batch_indices[:, None, None].expand_as(valid_channels)
        full_candidate_batch_idxs = expanded_batch_indices.flatten()[full_flat_valid]
        pred_batch_idxs = expanded_batch_indices.flatten()[selected_flat_valid]
        expanded_cut_counts = fragment_cut_counts[:, None, None].expand_as(valid_channels)
        full_candidate_cut_counts = expanded_cut_counts.flatten()[full_flat_valid]
        fragment_channel_indices = (
            th.arange(num_fragments, device=fragment_batch_indices.device)
            .reshape(-1, 1, 1)
            .expand_as(valid_channels)
        )
        full_candidate_fragment_idxs = fragment_channel_indices.flatten()[full_flat_valid]

        fragment_counts = th.bincount(fragment_batch_indices, minlength=batch_size).to(bond_logits)
        valid_ion_h_count = th.bincount(
            full_candidate_batch_idxs,
            minlength=batch_size,
        ).to(bond_logits)
        retained_ion_h_count = th.bincount(
            pred_batch_idxs,
            minlength=batch_size,
        ).to(bond_logits)
        retained_intensity_mass = (
            dense_intensity_log_probabilities.exp() * spectral_output_channels
        ).sum(dim=(1, 2))
        retained_ion_h_probability = (
            bond_logits.new_zeros(batch_size).index_add(
                0,
                fragment_batch_indices,
                retained_intensity_mass.to(bond_logits),
            )
            / fragment_counts
        )
        dense_joint_weights = dense_channel_log_weights.exp()
        joint_intensity_by_fragment = dense_joint_weights.sum(dim=(1, 2))
        retained_joint_intensity_by_fragment = (dense_joint_weights * spectral_output_channels).sum(
            dim=(1, 2)
        )
        total_joint_intensity = bond_logits.new_zeros(batch_size).index_add(
            0,
            fragment_batch_indices,
            joint_intensity_by_fragment.to(bond_logits),
        )
        retained_joint_intensity = bond_logits.new_zeros(batch_size).index_add(
            0,
            fragment_batch_indices,
            retained_joint_intensity_by_fragment.to(bond_logits),
        )
        retained_joint_intensity_fraction = retained_joint_intensity / total_joint_intensity
        mode_log_scores = (
            F.logsigmoid(ion_mode_proposal_logits)
            if ion_mode_proposal_binary_logits
            else ion_mode_proposal_logits
        )
        mode_log_probabilities = th.log_softmax(
            mode_log_scores.masked_fill(~valid_ion_modes, -th.inf),
            dim=1,
        )
        retained_mode_mass = (mode_log_probabilities.exp() * mode_keep).sum(dim=1)
        retained_ion_mode_probability = (
            bond_logits.new_zeros(batch_size).index_add(
                0,
                fragment_batch_indices,
                retained_mode_mass.to(bond_logits),
            )
            / fragment_counts
        )
        valid_ion_mode_count = bond_logits.new_zeros(batch_size).index_add(
            0,
            fragment_batch_indices,
            valid_ion_modes.sum(dim=1).to(bond_logits),
        )
        retained_ion_mode_count = bond_logits.new_zeros(batch_size).index_add(
            0,
            fragment_batch_indices,
            mode_keep.sum(dim=1).to(bond_logits),
        )
        selected_mask_probabilities = candidates.mask_log_probabilities.exp()
        retained_mask_probability = selected_mask_probabilities.new_zeros(batch_size).index_add(
            0,
            candidates.mask_batch_indices,
            selected_mask_probabilities,
        )
        selected_mask_count = th.bincount(
            candidates.mask_batch_indices,
            minlength=batch_size,
        )
        component_pathway_count = th.bincount(
            candidates.component_batch_indices,
            minlength=batch_size,
        )
        selected_mask_count_float = selected_mask_count.to(bond_logits)
        component_pathway_count_float = component_pathway_count.to(bond_logits)
        unique_component_count_float = candidates.unique_component_count.to(bond_logits)
        productive_mask_indices = th.unique(
            candidates.component_mask_indices[candidates.unique_representative_component_indices]
        )
        productive_mask_count = bond_logits.new_zeros(batch_size).index_add(
            0,
            candidates.mask_batch_indices[productive_mask_indices],
            th.ones_like(productive_mask_indices, dtype=bond_logits.dtype),
        )
        mean_mask_cut_count = (
            bond_logits.new_zeros(batch_size).index_add(
                0,
                candidates.mask_batch_indices,
                candidates.mask_cut_counts.to(bond_logits),
            )
            / selected_mask_count_float
        )
        mask_cut_fractions = {}
        cut_count_buckets = {
            "0": candidates.mask_cut_counts == 0,
            "1": candidates.mask_cut_counts == 1,
            "2": candidates.mask_cut_counts == 2,
            "3": candidates.mask_cut_counts == 3,
            "4": candidates.mask_cut_counts == 4,
            "5": candidates.mask_cut_counts == 5,
            "6plus": candidates.mask_cut_counts >= 6,
            "4plus": candidates.mask_cut_counts >= 4,
        }
        for bucket_name, bucket_mask in cut_count_buckets.items():
            bucket_count = bond_logits.new_zeros(batch_size).index_add(
                0,
                candidates.mask_batch_indices,
                bucket_mask.to(bond_logits),
            )
            mask_cut_fractions[f"mask_cut_fraction_{bucket_name}"] = (
                bucket_count / selected_mask_count_float
            )
        candidate_timing_values = {
            "candidate_input_transfer_batch_ms": candidates.timings.input_transfer_ms,
            "candidate_mask_selection_batch_ms": candidates.timings.mask_selection_ms,
            "candidate_topology_batch_ms": candidates.timings.topology_ms,
            "candidate_assembly_batch_ms": candidates.timings.assembly_ms,
            "candidate_output_transfer_batch_ms": candidates.timings.output_transfer_ms,
            "candidate_total_batch_ms": candidates.timings.total_ms,
            "candidate_total_molecule_ms": candidates.timings.total_ms / batch_size,
        }
        # Timings are per batch, but Lightning logs per-spectrum tensors, so broadcast
        # each scalar across the batch dimension.
        candidate_timing_tensors = {
            name: bond_logits.new_full((batch_size,), value)
            for name, value in candidate_timing_values.items()
        }
        logged_mixture_log_weights = (
            mixture_log_weights
            if mixture_log_weights is not None
            else bond_logits.new_zeros((batch_size, 1))
        )
        mixture_weights = logged_mixture_log_weights.exp()
        proposal_mixture_entropy = -(mixture_weights * logged_mixture_log_weights).sum(dim=1)
        output = {
            "pred_mzs": pred_mzs,
            "pred_batch_idxs": pred_batch_idxs,
            "candidate_mzs_by_cut": full_candidate_mzs,
            "candidate_batch_idxs_by_cut": full_candidate_batch_idxs,
            "candidate_cut_counts": full_candidate_cut_counts,
            "full_candidate_fragment_idxs": full_candidate_fragment_idxs,
            "ion_h_dense_mzs": ion_mzs,
            "ion_h_dense_log_weights": dense_channel_log_weights,
            "ion_h_chemical_valid_channels": valid_channels,
            "ion_h_support_logits": ion_h_support_logits,
            "ion_mode_proposal_logits": ion_mode_proposal_logits,
            "ion_h_fragment_batch_idxs": fragment_batch_indices,
            "valid_ion_h_count": valid_ion_h_count,
            "retained_ion_h_count": retained_ion_h_count,
            "retained_ion_h_probability": retained_ion_h_probability,
            "retained_joint_channel_intensity_fraction": retained_joint_intensity_fraction,
            "ion_h_cutoff_pruned_fraction": 1.0 - retained_ion_h_count / valid_ion_h_count,
            "valid_ion_mode_count": valid_ion_mode_count,
            "retained_ion_mode_count": retained_ion_mode_count,
            "retained_ion_mode_probability": retained_ion_mode_probability,
            "ion_mode_cutoff_pruned_fraction": (
                1.0 - retained_ion_mode_count / valid_ion_mode_count
            ),
            "bond_break_logits": bond_logits,
            "proposal_mixture_log_weights": logged_mixture_log_weights,
            "proposal_mixture_entropy": proposal_mixture_entropy,
            "proposal_mixture_effective_count": proposal_mixture_entropy.exp(),
            "proposal_mixture_top1_weight": mixture_weights.max(dim=1).values,
            "atom_charge_logits": atom_charge_logits,
            "retained_mask_probability": retained_mask_probability,
            "selected_mask_count": selected_mask_count,
            "component_pathway_count": component_pathway_count,
            "mean_mask_cut_count": mean_mask_cut_count,
            "proposal_single_cut_fraction": candidates.proposal_single_cut_fraction,
            "unique_component_count": candidates.unique_component_count,
            "fragment_count_before_cutoff": fragment_count_before_cutoff,
            "fragment_count_after_cutoff": fragment_count_after_cutoff,
            "retained_fragment_probability": retained_fragment_probability,
            "fragment_cutoff_pruned_fraction": fragment_cutoff_pruned_fraction,
            "components_per_mask": component_pathway_count_float / selected_mask_count_float,
            "unique_components_per_mask": (
                unique_component_count_float / selected_mask_count_float
            ),
            "latent_pathways_per_unique_fragment": (
                component_pathway_count_float / unique_component_count_float
            ),
            "unique_fragment_target_fill_fraction": (
                unique_component_count_float / self.max_masks
            ).clamp(max=1.0)
            * float(self.unique_fragment_mode),
            "unique_fragment_target_reached": (
                (unique_component_count_float >= self.max_masks).to(bond_logits)
                * float(self.unique_fragment_mode)
            ),
            "unique_fragment_effective_count": unique_fragment_effective_count,
            "unique_fragment_top1_mass_fraction": (unique_fragment_top1_mass_fraction),
            "duplicate_pathway_probability_fraction": (duplicate_pathway_probability_fraction),
            "productive_mask_fraction": (productive_mask_count / selected_mask_count_float),
            "duplicate_component_pathway_fraction": (
                1.0 - unique_component_count_float / component_pathway_count_float
            ),
            **mask_cut_fractions,
            **candidate_timing_tensors,
        }
        if pred_intensities is None:
            output["pred_logprobs"] = pred_logprobs
        else:
            output["pred_intensities"] = pred_intensities
        if self.return_base_spectrum:
            output["base_pred_logprobs"] = base_pathway_log_probabilities.flatten()[
                selected_flat_valid
            ]
        if self.unique_fragment_mode:
            output.update(
                {
                    "cross_ev_candidate_mzs": full_candidate_mzs,
                    "cross_ev_candidate_fragment_idxs": full_candidate_fragment_idxs,
                    "cross_ev_fragment_logprobs": fragment_log_probabilities,
                    "cross_ev_fragment_batch_idxs": fragment_batch_indices,
                }
            )
        if self.proposal_selection_only:
            output.update(
                {
                    "proposal_candidate_mzs": full_candidate_mzs,
                    "proposal_candidate_fragment_idxs": full_candidate_fragment_idxs,
                    "proposal_fragment_logprobs": base_fragment_log_probabilities,
                    "proposal_fragment_batch_idxs": fragment_batch_indices,
                }
            )
        return output
