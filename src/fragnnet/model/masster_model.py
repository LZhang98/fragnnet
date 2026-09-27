"""MASSter: one-pass bond-set fragmentation spectrum prediction.

Unlike the DAG-based FraGNNet path, MASSter never enumerates fragments ahead of
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

The optional ``connected_fragment_beam`` proposal instead grows connected atom sets by beam search.
It normalizes exact boundary-mask scores over retained fragments, without latent
pathway marginalization or charge allocation; chemistry and intensity scoring are shared.

The final spectrum is normalized over the candidates generated for that precursor.
Unmatched observed peaks are candidate-coverage diagnostics, not a predicted class.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from fragnnet.frag.masster_candidates import (
    MassterCandidateBatch,
    MassterStaticGraph,
    build_masster_candidates,
)
from fragnnet.frag.masster_fast_gnn import build_fast_gnn_candidates
from fragnnet.frag.masster_templates import (
    MASSTER_NUM_RULES,
    MassterTemplate,
    filter_masster_templates,
)
from fragnnet.model.base_model import (
    QET_ACE_OFFSET,
    CutBondEncoder,
    QETEnergyEncoder,
    encode_cut_bond_pairs,
)
from fragnnet.model.bond_fragment_heads import (
    AtomChargeAffinityHead,
    BondFocusedBreakHead,
    CandidateIntensityTransformerHead,
    EndpointBondBreakHead,
    IonModeProposalHead,
    _segment_logsumexp,
    _validate_tensor_condition,
    boundary_bond_endpoint_orientation,
    component_charge_log_probabilities,
    latent_pathway_charge_log_probabilities,
    masked_ion_hydrogen_log_probabilities,
    pool_boundary_bond_embeddings,
    pool_component_and_complement_embeddings,
)
from fragnnet.model.candidate_features import (
    CandidateFeatureInputs,
    build_candidate_features,
    candidate_feature_required_graph_fields,
    candidate_feature_width,
    component_neutral_masses,
    resolve_candidate_features,
)
from fragnnet.model.form_embedder import get_embedder
from fragnnet.model.nn_blocks import GNN
from fragnnet.utils.formula_utils import (
    H_MASS,
    PERIODIC_TABLE,
    PREC_TYPE_TO_CMF_COMP_DIFF,
    PREC_TYPE_TO_CMF_MASS_DIFF,
    PREC_TYPE_TO_COMP_DIFF,
    PREC_TYPE_TO_MASS_DIFF,
    cmf_duplicate_crf_h_shift,
)
from fragnnet.utils.frag_utils import CANONICAL_ELEMENT_ORDER
from fragnnet.utils.spec_utils import weighted_mean_by_batch


def _mean_values_by_batch(
    values: th.Tensor,
    batch_indices: th.Tensor | None,
    batch_size: int,
    weights: th.Tensor | None = None,
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
        weights: Optional reading weights for merged context.

    Returns:
        Mean CE value per spectrum, shape ``(batch_size,)``.

    Raises:
        ValueError: If the shapes/dtypes disagree, or a spectrum has no CE value.
    """
    values = values.reshape(-1)
    if weights is not None:
        return weighted_mean_by_batch(values.unsqueeze(1), weights, batch_indices, batch_size)[:, 0]
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


def _fragment_release_log_probabilities(
    bond_logits: th.Tensor,
    chemical_bond_index: th.Tensor,
    num_atoms: int,
    fragment_atom_indices: th.Tensor,
    atom_fragment_indices: th.Tensor,
    fragment_boundary_bond_indices: th.Tensor,
    boundary_fragment_indices: th.Tensor,
    num_fragments: int,
) -> th.Tensor:
    """Return the closed-form probability that each fragment is released intact.

    Under the independent Bernoulli bond model, atom set ``F`` is released with its
    interior intact when every bond crossing its boundary ``dF`` is cut and every bond
    with both endpoints in ``F`` is not::

        log P_release(F) = sum_{b in dF} log sigmoid(l_b) + sum_{b in int F} log sigmoid(-l_b)

    Bonds elsewhere in the precursor do not affect the event, so no mask enumeration is
    involved. The interior sum needs no per-fragment bond scan: adding the intact log
    probability of every bond incident to an atom of ``F`` counts each interior bond twice
    and each boundary bond once, so
    ``sum_{int F} = (sum_{a in F} S_a - sum_{b in dF} log sigmoid(-l_b)) / 2`` with
    ``S_a = sum_{b incident to a} log sigmoid(-l_b)``.

    For an acyclic fragment this is the exact atom-set marginal. For a fragment containing a
    ring it is a lower bound, because one ring bond inside ``F`` can break without changing
    the atom set.

    Args:
        bond_logits: Cut logit per chemical bond, shape ``(num_bonds,)``.
        chemical_bond_index: Global endpoint atoms per chemical bond, shape ``(2, num_bonds)``.
        num_atoms: Number of atoms in the batch.
        fragment_atom_indices: Global atom index of every fragment membership, shape ``(M,)``.
        atom_fragment_indices: Fragment index of every membership, shape ``(M,)``.
        fragment_boundary_bond_indices: Global bond index of every boundary membership,
            shape ``(D,)``.
        boundary_fragment_indices: Fragment index of every boundary membership, shape ``(D,)``.
        num_fragments: Number of fragments.

    Returns:
        ``log P_release`` per fragment in float32, shape ``(num_fragments,)``.
    """
    if bond_logits.ndim != 1 or chemical_bond_index.shape != (2, bond_logits.shape[0]):
        raise ValueError("release probabilities need one logit per chemical bond")
    logits = bond_logits.float()
    cut_log_probabilities = F.logsigmoid(logits)
    intact_log_probabilities = F.logsigmoid(-logits)
    atom_intact_sums = (
        logits.new_zeros(num_atoms)
        .index_add(0, chemical_bond_index[0], intact_log_probabilities)
        .index_add(0, chemical_bond_index[1], intact_log_probabilities)
    )
    incident_intact = logits.new_zeros(num_fragments).index_add(
        0, atom_fragment_indices, atom_intact_sums[fragment_atom_indices]
    )
    boundary_cut = logits.new_zeros(num_fragments).index_add(
        0, boundary_fragment_indices, cut_log_probabilities[fragment_boundary_bond_indices]
    )
    boundary_intact = logits.new_zeros(num_fragments).index_add(
        0, boundary_fragment_indices, intact_log_probabilities[fragment_boundary_bond_indices]
    )
    return boundary_cut + 0.5 * (incident_intact - boundary_intact)


def _top_k_fragment_keep_mask(
    scores: th.Tensor,
    fragment_batch_indices: th.Tensor,
    forced: th.Tensor,
    batch_size: int,
    k: int,
) -> th.Tensor:
    """Keep every forced fragment plus the highest-scoring others, up to ``k`` per molecule.

    A molecule with at least ``k`` forced fragments keeps exactly its forced set. Ties are
    broken by pool order, so the selection is deterministic.

    Args:
        scores: Ranking score per fragment; ``-inf`` ranks last. Shape ``(num_fragments,)``.
        fragment_batch_indices: Molecule index per fragment, shape ``(num_fragments,)``.
        forced: Fragments kept regardless of score, shape ``(num_fragments,)``.
        batch_size: Number of molecules.
        k: Fragment budget per molecule.

    Returns:
        Boolean keep mask, shape ``(num_fragments,)``.
    """
    if scores.shape != fragment_batch_indices.shape or forced.shape != scores.shape:
        raise ValueError("scores, batch indices and forced flags must align")
    if forced.dtype != th.bool or fragment_batch_indices.dtype != th.long:
        raise ValueError("forced must be boolean and batch indices torch.long")
    if k < 1:
        raise ValueError(f"k must be positive, got {k}")
    # 1. Forced fragments sort first, then descending score, then pool order, per molecule.
    ranked_scores = scores.detach().float().masked_fill(forced, th.inf)
    order = th.argsort(ranked_scores, descending=True, stable=True)
    order = order[th.argsort(fragment_batch_indices[order], stable=True)]
    # 2. Rank within each molecule and keep max(k, forced count) of them.
    counts = th.bincount(fragment_batch_indices, minlength=batch_size)
    starts = th.cumsum(counts, dim=0) - counts
    sorted_batch = fragment_batch_indices[order]
    rank = th.arange(order.numel(), device=order.device) - starts[sorted_batch]
    forced_counts = th.bincount(fragment_batch_indices[forced], minlength=batch_size)
    keep_sorted = rank < th.clamp(forced_counts, min=k)[sorted_batch]
    keep = th.zeros_like(forced)
    keep[order] = keep_sorted
    return keep


def _missing_valid_fallback_mask(
    scores: th.Tensor,
    valid: th.Tensor,
    keep: th.Tensor,
    fragment_batch_indices: th.Tensor,
    batch_size: int,
) -> th.Tensor:
    """Return one best valid fragment for each molecule whose kept set has none.

    A selection must not leave a molecule without a fragment that can form an ion. Exactly one
    fragment is added, the first in pool order among the highest-scoring valid ones: matching
    the maximum score instead would re-admit a whole group of exactly tied fragments, which
    closed-form release scores produce on every symmetric substructure.

    Args:
        scores: Ranking score per fragment, shape ``(num_fragments,)``.
        valid: Fragments with at least one chemically valid channel, shape ``(num_fragments,)``.
        keep: Current keep mask, shape ``(num_fragments,)``.
        fragment_batch_indices: Molecule index per fragment, shape ``(num_fragments,)``.
        batch_size: Number of molecules.

    Returns:
        Boolean mask with at most one True per molecule, shape ``(num_fragments,)``.
    """
    num_fragments = scores.shape[0]
    kept_valid_counts = th.bincount(fragment_batch_indices[keep & valid], minlength=batch_size)
    masked_scores = scores.detach().float().masked_fill(~valid, -th.inf)
    best_scores = masked_scores.new_full((batch_size,), -th.inf).scatter_reduce(
        0, fragment_batch_indices, masked_scores, reduce="amax", include_self=True
    )
    candidates = (
        valid
        & (masked_scores == best_scores[fragment_batch_indices])
        & (kept_valid_counts == 0)[fragment_batch_indices]
    )
    fragment_indices = th.arange(num_fragments, device=scores.device)
    first_candidate = fragment_indices.new_full((batch_size,), num_fragments).scatter_reduce(
        0,
        fragment_batch_indices[candidates],
        fragment_indices[candidates],
        reduce="amin",
        include_self=True,
    )
    fallback = th.zeros_like(valid)
    fallback[first_candidate[first_candidate < num_fragments]] = True
    return fallback


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
    available_per_spectrum = th.bincount(flat_batch_indices[flat_valid], minlength=batch_size)
    if min_probability == 0.0:
        return valid_mask.clone()

    flat_log_probabilities = F.logsigmoid(logits.detach()).flatten()
    keep = flat_valid & (flat_log_probabilities >= math.log(min_probability))
    kept_per_spectrum = th.bincount(flat_batch_indices[keep], minlength=batch_size)
    if kept_per_spectrum.gt(0).all():
        return keep.reshape_as(valid_mask)

    for batch_index in th.where((kept_per_spectrum == 0) & (available_per_spectrum > 0))[0]:
        batch_valid_indices = th.where(flat_valid & (flat_batch_indices == batch_index))[0]
        fallback_index = batch_valid_indices[flat_log_probabilities[batch_valid_indices].argmax()]
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


# How far a fragment may be pushed up the hydrogen-transfer ladder.
#   "none"       - only the element counts must stay nonnegative. Hydrogen may be gained with
#                  no source, which is how MASSter behaved before this option existed: the
#                  intact molecule reached prec_mz + max_h_transfer * H. Those channels are
#                  chemically unreachable but act as negatives in the candidate softmax, and
#                  they currently absorb M+1 isotope peaks, so removing them is a measured
#                  trade rather than a pure win.
#   "complement" - a gain may not exceed the hydrogens held by the complementary neutral
#                  fragment; the intact molecule can then only lose hydrogens. Matches the
#                  `num_hs` term of FraGNNet's cap.
#   "valence"    - additionally caps the gain at the fragment's free valence slots plus one
#                  per severed bond, i.e. FraGNNet's `compute_cc_h_cap`. Full parity.
H_TRANSFER_BOUNDS = ("none", "complement", "valence")


def _cmf_duplicate_crf_shift(
    precursor_type: str,
    max_h_transfer: int,
    *,
    tolerance: float = 1e-8,
) -> int | None:
    """Return the reachable CRF H shift duplicated by CMF, if one exists."""
    return cmf_duplicate_crf_h_shift(
        precursor_type,
        max_h_transfer,
        tolerance=tolerance,
    )


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
        num_rule_indices = 1 + MASSTER_NUM_RULES
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


class MassterContextEncoder(nn.Module):
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
        normalization: str = "none",
        dropout: float = 0.0,
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
            ce_insert_type: ``embed``, ``embed_clamped`` or ``qet`` enables learned CE
                embeddings; other modes retain the standardized scalar input. ``embed``
                and ``embed_clamped`` look the rounded CE up in the ``int_embedder``
                table; ``qet`` embeds the raw CE with ``QETEnergyEncoder``, smooth in
                energy and with a learned missing-CE flag.
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
        if ce_insert_type not in {"id", "lin", "embed", "embed_clamped", "bin", "qet"}:
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
        self.use_ce_embedding = ce_insert_type in {"embed", "embed_clamped", "qet"}
        self.nce_embedding = None
        self.ace_embedding = None
        if ce_insert_type == "qet":
            if use_nce:
                self.nce_embedding = QETEnergyEncoder(nce_mean, nce_std, ce_insert_size)
            if use_ace:
                # eV is linear; see QET_ACE_OFFSET.
                self.ace_embedding = QETEnergyEncoder(
                    ace_mean, ace_std, ce_insert_size, offset=QET_ACE_OFFSET
                )
        elif self.use_ce_embedding:
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
        if normalization not in ("none", "layer"):
            raise ValueError(
                f"context normalization must be 'none' or 'layer', got {normalization!r}"
            )
        if not 0.0 <= dropout <= 1.0:
            raise ValueError(f"context dropout must be in [0, 1], got {dropout}")
        # The context vector conditions the GNN and every head, but its input scale
        # varies with how many CE readings a merged acquisition group contributed
        # (median 4, max 86), so the projector is optionally normalized.
        projector_layers: list[nn.Module] = [
            nn.Linear(input_dim, context_dim),
            nn.SiLU(),
        ]
        if dropout > 0.0:
            projector_layers.append(nn.Dropout(dropout))
        projector_layers.append(nn.Linear(context_dim, context_dim))
        if normalization == "layer":
            projector_layers.append(nn.LayerNorm(context_dim))
        self.projector = nn.Sequential(*projector_layers)
        # Width of the collision-energy slice of the concatenated features, before the
        # projector mixes it with adduct/instrument/mode. Conditioning the candidate
        # attention on THIS rather than on the projected context keeps the experiment about
        # collision energy instead of about metadata in general.
        ce_width = self.ce_insert_size if self.use_ce_embedding else 1
        self.ce_feature_dim = ce_width * (int(self.use_nce) + int(self.use_ace))

    def _embed_ce_values(
        self,
        values: th.Tensor,
        batch_indices: th.Tensor | None,
        batch_size: int,
        max_value: float,
        embedding: nn.Module,
        weights: th.Tensor | None = None,
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

        if self.ce_insert_type == "qet":
            # Raw energies: no rounding, no clamp at max_value; below 0 is flagged as missing.
            embedded_values = embedding(values)
        else:
            upper_bound = int(max_value) - int(self.ce_insert_type == "embed_clamped")
            embedding_indices = values.clamp(min=-1, max=upper_bound).round().long().unsqueeze(1)
            embedded_values = embedding(embedding_indices)
        if weights is not None:
            return weighted_mean_by_batch(embedded_values, weights, batch_indices, batch_size)
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
        nce_weights: th.Tensor | None = None,
        ace_weights: th.Tensor | None = None,
        return_ce_features: bool = False,
    ) -> th.Tensor | tuple[th.Tensor, th.Tensor]:
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
            nce_weights: Optional per-reading NCE weights.
            ace_weights: Optional per-reading ACE weights.

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
        ce_feature_start = None

        if self.instrument_embedding is not None:
            if instrument_types is None:
                raise ValueError("instrument_types are required by the configured context")
            features.append(self.instrument_embedding(instrument_types.reshape(-1)))
        if self.fragmentation_mode_embedding is not None:
            if fragmentation_modes is None:
                raise ValueError("fragmentation_modes are required by the configured context")
            features.append(self.fragmentation_mode_embedding(fragmentation_modes.reshape(-1)))
        if self.use_nce or self.use_ace:
            ce_feature_start = len(features)
        if self.use_nce:
            if not include_collision_energy:
                ce_width = self.ce_insert_size if self.use_ce_embedding else 1
                features.append(features[0].new_zeros((batch_size, ce_width)))
            elif nce is None:
                raise ValueError("NCE values are required by the configured context")
            elif self.nce_embedding is None:
                mean_nce = _mean_values_by_batch(nce, nce_batch_indices, batch_size, nce_weights)
                features.append(((mean_nce - self.nce_mean) / self.nce_std).unsqueeze(1))
            else:
                features.append(
                    self._embed_ce_values(
                        nce,
                        nce_batch_indices,
                        batch_size,
                        self.nce_max,
                        self.nce_embedding,
                        nce_weights,
                    )
                )
        if self.use_ace:
            if not include_collision_energy:
                ce_width = self.ce_insert_size if self.use_ce_embedding else 1
                features.append(features[0].new_zeros((batch_size, ce_width)))
            elif ace is None:
                raise ValueError("ACE values are required by the configured context")
            elif self.ace_embedding is None:
                mean_ace = _mean_values_by_batch(ace, ace_batch_indices, batch_size, ace_weights)
                features.append(((mean_ace - self.ace_mean) / self.ace_std).unsqueeze(1))
            else:
                features.append(
                    self._embed_ce_values(
                        ace,
                        ace_batch_indices,
                        batch_size,
                        self.ace_max,
                        self.ace_embedding,
                        ace_weights,
                    )
                )
        context = self.projector(th.cat(features, dim=1))
        if not return_ce_features:
            return context
        if ce_feature_start is None:
            raise ValueError("CE features were requested but the context carries no CE input")
        return context, th.cat(features[ce_feature_start:], dim=1)


class MassterModel(nn.Module):
    """Predict sparse MS/MS spectra from globally ranked precursor bond masks.

    The default cut-mask forward pass runs four stages (see :meth:`forward`):

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
        mask_selection_threads: Worker threads for CPU mask selection and topology.
        candidate_backend: Explicit CPU or CUDA candidate construction.
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
        context_normalization: str = "none",
        context_dropout: float = 0.0,
        precursor_dropout: float,
        precursor_max_degree: int | None = None,
        precursor_max_spatial_distance: int | None = None,
        precursor_multi_hop_max_distance: int | None = None,
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
        intensity_transformer_attention: Literal["induced", "full"],
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
        candidate_backend: Literal["cpu", "cuda"] = "cpu",
        proposal_method: Literal["cut_masks", "connected_fragment_beam"] = "cut_masks",
        fast_gnn_beam_width: int = 256,
        fast_gnn_max_boundary_cuts: int = 3,
        min_mask_relative_probability: float,
        min_fragment_relative_probability: float,
        fragment_selection: Literal["mask_order", "marginal"] = "mask_order",
        selected_fragment_target: int | None = None,
        include_single_bond_cuts: bool,
        max_h_transfer: int,
        h_transfer_bound: str = "none",
        include_cmf: bool,
        charge_pooling: str,
        use_smarts_templates: bool,
        context_in_gnn: bool = False,
        boundary_bond_encoder: bool = False,
        intensity_norm_placement: str = "post",
        intensity_qk_norm: bool = False,
        intensity_mass_pair_bias: bool = False,
        intensity_ce_pair_bias: bool = False,
        intensity_ce_film: bool = False,
        bond_endpoint_aggregation: str = "mean",
        gnn_residual_activation: str = "relu",
        intensity_escape: bool = False,
        candidate_features: Sequence[str] = ("c_cut_count", "c_size"),
        candidate_atom_dropout: float = 0.0,
        candidate_boundary_dropout: float = 0.0,
        candidate_boundary_block: bool = True,
        boundary_local_layers: int = 0,
        boundary_encoder_dropout: float = 0.0,
        intensity_input_dropout: float = 0.0,
        memorization_sink_dim: int = 0,
        memorization_sink_num_molecules: int = 0,
        global_message_substitution: float = 0.0,
        global_message_noise_std: float = 0.0,
        global_message_swap: float = 0.0,
        global_message_swap_unit: str = "atom",
        light_side_token: bool = False,
    ) -> None:
        """Initialize the MASSter model.

        Args:
            node_feature_dim: Width of the precursor atom features.
            edge_feature_dim: Width of the precursor bond features.
            precursor_hidden_dim: Hidden width of the precursor GNN; also the width of
                the pooled fragment/complement/boundary embeddings.
            precursor_num_layers: Number of message-passing layers.
            precursor_num_heads: Number of molecular-attention heads.
            precursor_max_degree: Largest degree the Graphormer degree embedding
                distinguishes; ``None`` keeps the encoder default.
            precursor_max_spatial_distance: Largest shortest-path distance the Graphormer
                spatial embedding distinguishes; ``None`` keeps the encoder default.
            precursor_multi_hop_max_distance: Number of leading bonds encoded along each
                shortest path by the Graphormer edge bias; ``None`` keeps the encoder
                default. Ignored by every non-Graphormer ``precursor_gnn_type``.
            precursor_gnn_type: GNN cell name understood by :class:`GNN`.
            precursor_normalization: Normalization used inside the GNN.
            context_normalization: ``layer`` appends a LayerNorm to the metadata
                context projector; ``none`` keeps the original unnormalized MLP.
            context_dropout: Dropout inside the context projector.
            precursor_dropout: Dropout inside the GNN.
            context_dim: Width of the per-spectrum conditioning vector.
            context_in_gnn: Append context to each atom's input in the precursor and
                optional proposal GNN. Disabled preserves legacy parameter shapes.
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
            intensity_transformer_attention: ``induced`` routes candidate exchange
                through a latent set and stays linear in the candidate count; ``full``
                uses direct candidate-to-candidate attention, which is quadratic but
                is not capped at the inducing-point rank. ``full`` ignores
                ``intensity_transformer_num_inducing_points``.
            intensity_transformer_num_inducing_points: Latent tokens used to keep
                attention linear in the number of candidates.
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
            ce_insert_type: CE representation type; ``embed``, ``embed_clamped`` and
                ``qet`` enable learned vectors (see ``MassterContextEncoder``).
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
                masks and build topology on CPU. Numeric kernels release the GIL.
                Ignored by the CUDA candidate backend. Defaults to 1.
            proposal_method: ``cut_masks`` retains latent-pathway marginalization;
                ``connected_fragment_beam`` searches connected atom sets and normalizes exact boundary
                mask scores over fragments, without a charge-allocation factor.
            fast_gnn_beam_width: States retained per connected-set growth round.
            fast_gnn_max_boundary_cuts: Maximum boundary size for proper fragments.
            candidate_backend: Exact cut-mask construction on CPU or opt-in CUDA.
                Ignored by fast_gnn, which runs PyTorch search on the model device.
                CUDA requires FP32, one mixture, unique mode, zero mask floors and
                SMARTS disabled. The model can be constructed before device transfer.
            max_enumerated_masks: Safety cap on latent masks considered while filling
                the unique-fragment target.
            min_mask_relative_probability: MAP-relative probability floor below which
                a ranked mask is dropped before candidate assembly.
            min_fragment_relative_probability: MAP-relative probability floor below
                which a marginalized unique fragment is dropped before intensity
                scoring. Zero disables fragment pruning.
            fragment_selection: Which unique fragments of the enumerated pool reach the
                intensity head. ``mask_order`` keeps the whole pool the enumeration
                produced. ``marginal`` keeps the forced precursor and single-cut
                fragments plus the highest closed-form release probabilities up to
                ``selected_fragment_target``, and trains the proposal on the whole pool.
            selected_fragment_target: Fragments kept per precursor in ``marginal`` mode;
                must be ``None`` in ``mask_order`` mode.
            include_single_bond_cuts: Enumerate single-bond cuts in the ranked masks.
            max_h_transfer: Largest hydrogen transfer per fragment, in either
                direction.
            include_cmf: Emit the charge-migration ion channel alongside CRF.
            charge_pooling: ``logsumexp`` treats every atom as a potential charge site
                (favoring large components); ``logmeanexp`` removes that size prior.
            use_smarts_templates: Add matched SMARTS rearrangements as alternative
                assembly mechanisms within their corresponding cut masks.
            boundary_bond_encoder: Encode each cut bond with :class:`CutBondEncoder` from its
                (retained, lost) atom embeddings instead of averaging the two endpoints. The
                average is degenerate: it collapses to half the mean inside plus half the mean
                outside atom, so which atom was bonded to which is discarded and the retained
                and lost sides are indistinguishable.
            intensity_escape: Give the intensity output an escape coordinate for observed
                intensity no candidate can reach. The head emits one logit per spectrum; the
                model converts it to a mass in ``pred_intensities`` units and returns it as
                ``pred_escape_intensities`` for ``sparse_peak_list_distance``.
            candidate_features: Names of the descriptor columns appended to each candidate's
                intensity token, resolved against
                :data:`~fragnnet.model.candidate_features.CANDIDATE_FEATURE_SPECS`, which
                documents what each one measures. The list is a set: columns are concatenated
                in that module's canonical order, not the order given here. The default is the
                pair of columns that used to be unconditional.
            candidate_atom_dropout: Training-only probability of dropping each atom from a
                candidate's membership set before the component pool, so the pooled fragment
                and complement embeddings see a random subset. Regularizes against
                memorizing exact atom sets; every other candidate input keeps the true
                membership.
            candidate_boundary_dropout: Training-only probability of dropping a candidate's
                whole boundary block from the intensity token. A dropped block is zeroed and
                kept blocks are scaled by ``1 / (1 - p)`` (inverted dropout), so the expected
                input to the token projection is unchanged and evaluation uses the block
                as-is. Aimed at the boundary block because an inference-time corruption probe
                located most of the train-only intensity allocation there. The proposal branch
                and the candidate descriptors are untouched.
            candidate_boundary_block: False feeds a zero boundary block to the intensity head in
                training and evaluation alike, which removes the block. A rerouting test for the
                memorization-route plan, so it rejects the options that act on the block.
            boundary_local_layers: 0 builds the boundary block from the shared encoder's atom
                embeddings. ``k > 0`` builds it from a separate ``k``-layer plain GINE over the
                raw atom features, with no global attention and no context, so the cut term sees
                only the cut site's ``k``-hop chemistry and cannot name the molecule. Requires
                ``boundary_bond_encoder``.
            boundary_encoder_dropout: Training-only dropout on the ``CutBondEncoder`` hidden
                layer. Requires ``boundary_bond_encoder``.
            intensity_input_dropout: Training-only dropout on the projected candidate token.
            memorization_sink_dim: Width of the per-molecule memorization sink; 0 disables it.
                A zero-initialized row per molecule adds ``row · phi(token)`` to each candidate's
                residual when ``forward`` receives ``memorization_sink_indices`` (training only).
                Without indices the model is exactly the shared model. Requires
                ``unique_fragment_mode``.
            memorization_sink_num_molecules: Rows in the sink table, indexed by ``mol_id``; must
                exceed the largest ``mol_id`` passed. Set together with ``memorization_sink_dim``.
            global_message_substitution: Training-only probability, per atom and layer, that the
                shared encoder's global-attention message is replaced by the batch mean. Only the
                shared precursor encoder gets it. Requires ``precursor_gnn_type: GINE_GSA``.
            global_message_noise_std: Training-only Gaussian noise on the shared encoder's
                global-attention message, in units of each feature's batch standard deviation.
                Requires ``precursor_gnn_type: GINE_GSA``.
            global_message_swap: Training-only probability, per layer, that the shared encoder's
                global-attention message is replaced by another molecule's real message; the
                label is not mixed. Requires ``precursor_gnn_type: GINE_GSA``.
            global_message_swap_unit: What one swap draw replaces: ``atom``, ``molecule`` or
                ``feature``; see ``GINE_GSA``.
            light_side_token: Represent each candidate by the lighter side of its cut, in
                training and evaluation alike. The fragment slot of the token takes the lighter
                side's pool (the fragment itself, or the neutral loss of a large fragment), and
                the complement slot is zeroed, so no pool of a near-whole molecule reaches the
                head. The heavier side enters through the ``c_heavy_side`` descriptor alone.
                Requires ``unique_fragment_mode`` and ``c_heavy_side`` in
                ``candidate_features``.

        Raises:
            ValueError: If any hyperparameter is out of range or an adduct in
                ``precursor_types`` has no tabulated mass difference.
        """
        super().__init__()
        if max_masks <= 0 or max_h_transfer < 0:
            raise ValueError("max_masks must be positive and max_h_transfer nonnegative")
        if h_transfer_bound not in H_TRANSFER_BOUNDS:
            raise ValueError(
                f"h_transfer_bound must be one of {list(H_TRANSFER_BOUNDS)}, "
                f"got {h_transfer_bound!r}"
            )
        if max_enumerated_masks <= 0:
            raise ValueError("max_enumerated_masks must be positive")
        if mask_selection_threads < 1:
            raise ValueError("mask_selection_threads must be at least 1")
        if (
            proposal_method == "cut_masks"
            and unique_fragment_mode
            and max_enumerated_masks < max_masks
        ):
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
            raise ValueError(f"unknown MASSter intensity output: {intensity_output}")
        if intensity_escape and intensity_output != "independent":
            raise ValueError(
                "intensity_escape requires the independent intensity output; the escape is "
                "only read by the peak-list losses"
            )
        if intensity_output == "independent" and not (
            unique_fragment_mode and proposal_selection_only
        ):
            raise ValueError(
                "independent intensity output requires unique-fragment mode and "
                "proposal-selection-only mode"
            )
        if intensity_transformer_attention == "full" and not unique_fragment_mode:
            # Unique-fragment mode bounds a molecule's candidate count by max_masks.
            # The legacy pathway-token mode instead emits one token per component
            # occurrence, where quadratic candidate attention is not affordable.
            raise ValueError(
                "full candidate attention requires unique_fragment_mode, which bounds "
                "the quadratic candidate count by max_masks"
            )
        if separate_proposal_encoder and not proposal_selection_only:
            raise ValueError("separate_proposal_encoder requires proposal_selection_only")
        if not 0.0 <= candidate_atom_dropout < 1.0:
            raise ValueError(
                f"candidate_atom_dropout must be in [0, 1), got {candidate_atom_dropout}"
            )
        if not 0.0 <= candidate_boundary_dropout < 1.0:
            raise ValueError(
                f"candidate_boundary_dropout must be in [0, 1), got {candidate_boundary_dropout}"
            )
        if not candidate_boundary_block and (
            boundary_bond_encoder or boundary_local_layers > 0 or candidate_boundary_dropout > 0.0
        ):
            raise ValueError(
                "candidate_boundary_block: False removes the boundary block; turn off "
                "boundary_bond_encoder, boundary_local_layers and candidate_boundary_dropout"
            )
        if boundary_local_layers < 0:
            raise ValueError(
                f"boundary_local_layers must be nonnegative, got {boundary_local_layers}"
            )
        if (boundary_local_layers > 0 or boundary_encoder_dropout > 0.0) and not (
            boundary_bond_encoder
        ):
            raise ValueError(
                "boundary_local_layers and boundary_encoder_dropout require boundary_bond_encoder"
            )
        if memorization_sink_dim < 0 or memorization_sink_num_molecules < 0:
            raise ValueError("memorization sink dimensions must be nonnegative")
        if (memorization_sink_dim > 0) != (memorization_sink_num_molecules > 0):
            raise ValueError(
                "memorization_sink_dim and memorization_sink_num_molecules are set together"
            )
        if memorization_sink_dim > 0 and not unique_fragment_mode:
            raise ValueError("the memorization sink requires unique_fragment_mode")
        if light_side_token and not unique_fragment_mode:
            raise ValueError("light_side_token requires unique_fragment_mode")
        if light_side_token and "c_heavy_side" not in candidate_features:
            raise ValueError(
                "light_side_token requires c_heavy_side in candidate_features: the head must "
                "know which side it sees and keep a summary of the side it no longer pools"
            )
        if (
            global_message_substitution > 0.0
            or global_message_noise_std > 0.0
            or global_message_swap > 0.0
        ) and precursor_gnn_type != "GINE_GSA":
            raise ValueError(
                "global_message_substitution, global_message_noise_std and global_message_swap "
                f"require precursor_gnn_type GINE_GSA, got {precursor_gnn_type}"
            )
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
        if fragment_selection not in {"mask_order", "marginal"}:
            raise ValueError(f"unknown fragment_selection: {fragment_selection!r}")
        if fragment_selection == "mask_order" and selected_fragment_target is not None:
            raise ValueError(
                "selected_fragment_target is only read by fragment_selection='marginal'"
            )
        if fragment_selection == "marginal":
            if (
                isinstance(selected_fragment_target, bool)
                or not isinstance(selected_fragment_target, int)
                or not 1 <= selected_fragment_target <= max_masks
            ):
                raise ValueError(
                    "marginal fragment selection needs an integer selected_fragment_target in "
                    f"[1, unique_fragment_target={max_masks}], got {selected_fragment_target!r}"
                )
            # The release score is closed-form for one plain Bernoulli over cut masks, and the
            # selection only decides membership, so the proposal must not also shape intensity.
            if (
                not unique_fragment_mode
                or not proposal_selection_only
                or proposal_num_mixtures != 1
                or proposal_method != "cut_masks"
                or use_smarts_templates
                or min_fragment_relative_probability != 0.0
                or share_proposal_across_ev
            ):
                raise ValueError(
                    "marginal fragment selection requires unique mode, proposal_selection_only, "
                    "one mixture, cut_masks, SMARTS off, a zero fragment probability floor and "
                    "share_proposal_across_ev off"
                )
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
        if candidate_backend not in ("cpu", "cuda"):
            raise ValueError(f"unknown candidate_backend: {candidate_backend!r}; use cpu or cuda")
        if proposal_method not in ("cut_masks", "connected_fragment_beam"):
            raise ValueError(f"unknown proposal_method: {proposal_method!r}")
        if any(
            isinstance(v, bool) or not isinstance(v, int)
            for v in (fast_gnn_beam_width, fast_gnn_max_boundary_cuts)
        ):
            raise ValueError("fast_gnn beam width and boundary cuts must be integers")
        if fast_gnn_beam_width < 1 or fast_gnn_max_boundary_cuts < 0:
            raise ValueError("fast_gnn beam width must be positive and boundary cuts nonnegative")
        if proposal_method == "connected_fragment_beam" and (
            not unique_fragment_mode
            or proposal_num_mixtures != 1
            or use_smarts_templates
            or share_proposal_across_ev
            or min_mask_relative_probability != 0.0
        ):
            raise ValueError(
                "fast_gnn requires unique mode, one mixture, SMARTS off, "
                "share_proposal_across_ev off and zero mask probability floor"
            )
        if (
            proposal_method == "cut_masks"
            and candidate_backend == "cuda"
            and (
                proposal_num_mixtures != 1
                or not unique_fragment_mode
                or min_mask_relative_probability != 0.0
                or use_smarts_templates
            )
        ):
            raise ValueError(
                "CUDA candidates require one mixture, unique mode, zero mask probability "
                "floors and SMARTS disabled; select candidate_backend='cpu' for other modes"
            )
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
        self.candidate_backend = candidate_backend
        self.proposal_method = proposal_method
        self.fast_gnn_beam_width = fast_gnn_beam_width
        self.fast_gnn_max_boundary_cuts = fast_gnn_max_boundary_cuts
        self.min_mask_relative_probability = min_mask_relative_probability
        self.min_fragment_relative_probability = min_fragment_relative_probability
        self.fragment_selection = fragment_selection
        self.selected_fragment_target = selected_fragment_target
        self.include_single_bond_cuts = include_single_bond_cuts
        self.max_h_transfer = max_h_transfer
        self.h_transfer_bound = h_transfer_bound
        self.include_cmf = include_cmf
        self.charge_pooling = charge_pooling
        self.use_smarts_templates = use_smarts_templates
        self.use_mask_residuals = use_mask_residuals
        self.intensity_output = intensity_output
        self.return_base_spectrum = return_base_spectrum
        self.detach_base_ion_h = detach_base_ion_h
        self.proposal_selection_only = proposal_selection_only
        self.proposal_input_gradients_enabled = False
        self.separate_proposal_encoder = separate_proposal_encoder
        self.share_proposal_across_ev = share_proposal_across_ev
        if context_in_gnn and share_proposal_across_ev:
            raise ValueError("context_in_gnn is incompatible with share_proposal_across_ev")
        self.context_in_gnn = context_in_gnn
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

        # Graphormer structural-encoding sizes; omitted keys fall through to the encoder
        # defaults, and non-Graphormer gnn_types ignore them.
        graphormer_structural_kwargs = {
            "max_degree": precursor_max_degree,
            "max_spatial_distance": precursor_max_spatial_distance,
            "multi_hop_max_distance": precursor_multi_hop_max_distance,
        }
        self.precursor_encoder = GNN(
            node_feats_size=node_feature_dim + (context_dim if context_in_gnn else 0),
            edge_feats_size=edge_feature_dim,
            hidden_size=precursor_hidden_dim,
            num_layers=precursor_num_layers,
            gnn_type=precursor_gnn_type,
            normalization=precursor_normalization,
            dropout=precursor_dropout,
            num_heads=precursor_num_heads,
            residual_activation=gnn_residual_activation,
            global_message_substitution=global_message_substitution,
            global_message_noise_std=global_message_noise_std,
            global_message_swap=global_message_swap,
            global_message_swap_unit=global_message_swap_unit,
            **graphormer_structural_kwargs,
        )
        self.proposal_encoder = (
            GNN(
                node_feats_size=node_feature_dim + (context_dim if context_in_gnn else 0),
                edge_feats_size=edge_feature_dim,
                hidden_size=precursor_hidden_dim,
                num_layers=precursor_num_layers,
                gnn_type=precursor_gnn_type,
                normalization=precursor_normalization,
                dropout=precursor_dropout,
                num_heads=precursor_num_heads,
                residual_activation=gnn_residual_activation,
                **graphormer_structural_kwargs,
            )
            if separate_proposal_encoder
            else None
        )
        self.context_encoder = MassterContextEncoder(
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
            normalization=context_normalization,
            dropout=context_dropout,
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
                endpoint_aggregation=bond_endpoint_aggregation,
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
        self.intensity_mass_pair_bias = intensity_mass_pair_bias
        self.intensity_ce_conditioned = intensity_ce_pair_bias or intensity_ce_film
        self.intensity_escape = intensity_escape
        self.candidate_feature_specs = resolve_candidate_features(candidate_features)
        self.candidate_feature_dim = candidate_feature_width(
            self.candidate_feature_specs,
            edge_feature_dim,
        )
        self.candidate_atom_dropout = float(candidate_atom_dropout)
        self.candidate_boundary_dropout = float(candidate_boundary_dropout)
        self.candidate_boundary_block = candidate_boundary_block
        self.light_side_token = light_side_token
        self.boundary_bond_encoder = (
            CutBondEncoder(
                atom_dim=precursor_hidden_dim,
                hidden_dim=head_hidden_dim,
                out_dim=precursor_hidden_dim,
                dropout=boundary_encoder_dropout,
            )
            if boundary_bond_encoder
            else None
        )
        # The optional modules below use a forked RNG, so enabling one leaves every other
        # parameter's initialization exactly as in the configuration without it.
        self.boundary_local_encoder: GNN | None = None
        if boundary_local_layers > 0:
            with th.random.fork_rng(devices=[]):
                self.boundary_local_encoder = GNN(
                    node_feats_size=node_feature_dim,
                    edge_feats_size=edge_feature_dim,
                    hidden_size=precursor_hidden_dim,
                    num_layers=boundary_local_layers,
                    gnn_type="GINE",
                    normalization=precursor_normalization,
                    dropout=precursor_dropout,
                    residual_activation=gnn_residual_activation,
                )
        self.memorization_sink: nn.Embedding | None = None
        if memorization_sink_dim > 0:
            with th.random.fork_rng(devices=[]):
                self.memorization_sink = nn.Embedding(
                    memorization_sink_num_molecules,
                    memorization_sink_dim,
                )
            # Zero rows: an unseen molecule, and every molecule at the start, add nothing.
            nn.init.zeros_(self.memorization_sink.weight)
        self.charge_head = AtomChargeAffinityHead(
            precursor_hidden_dim,
            context_dim,
            head_hidden_dim,
            head_dropout,
        )
        if proposal_method == "connected_fragment_beam":
            # Retain checkpoint tensor names/shapes, but do not train an unused head.
            self.charge_head.requires_grad_(False)
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
            attention_type=intensity_transformer_attention,
            num_ion_modes=self.num_ion_modes,
            num_h_shifts=self.num_h_shifts,
            dropout=intensity_transformer_dropout,
            use_base_probability_feature=use_base_probability_feature,
            separate_ion_h_support_head=separate_ion_h_support_head,
            predict_oos=intensity_escape,
            initial_oos_probability=0.1,
            candidate_feature_dim=self.candidate_feature_dim,
            unique_fragment_tokens=unique_fragment_mode,
            norm_placement=intensity_norm_placement,
            qk_norm=intensity_qk_norm,
            mass_pair_bias=intensity_mass_pair_bias,
            ce_pair_bias=intensity_ce_pair_bias,
            ce_film=intensity_ce_film,
            ce_feature_dim=self.context_encoder.ce_feature_dim,
            input_dropout=intensity_input_dropout,
            sink_dim=memorization_sink_dim,
        )
        self._setup_chemistry_buffers()
        self._register_load_state_dict_pre_hook(self._restore_chemistry_buffers)

    def _restore_chemistry_buffers(
        self,
        state_dict: dict[str, th.Tensor],
        prefix: str,
        *args: object,
    ) -> None:
        """Migrate derived legacy counts without restoring unsigned validity rules."""
        for name in (
            "required_atomic_numbers",
            "crf_formula_deltas",
            "cmf_formula_deltas",
            "composition_element_columns",
        ):
            state_dict[prefix + name] = self.get_buffer(name)

    def set_evaluation_ion_h_top_k(self, top_k: int) -> None:
        """Set evaluation-only fixed-k ion/H selection; zero disables it."""
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("evaluation ion/H top_k must be a nonnegative integer")
        self.evaluation_ion_h_top_k = top_k

    def set_proposal_input_gradients_enabled(self, enabled: bool) -> None:
        """Control whether proposal supervision updates its shared input encoders."""
        if not isinstance(enabled, bool):
            raise ValueError("proposal input-gradient state must be boolean")
        self.proposal_input_gradients_enabled = enabled

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

        required_elements = {1}
        parsed_crf: list[dict[str, int]] = []
        parsed_cmf: list[dict[str, int]] = []
        for precursor_type in self.precursor_types:
            crf_counts = PREC_TYPE_TO_COMP_DIFF[precursor_type]
            cmf_counts = PREC_TYPE_TO_CMF_COMP_DIFF.get(precursor_type, {})
            parsed_crf.append(crf_counts)
            parsed_cmf.append(cmf_counts)
            for counts in (crf_counts, cmf_counts):
                for element, count in counts.items():
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
        # Composition features count the full element vocabulary, not the validity-only set
        # above: that set holds just the elements an adduct can remove, so on the common
        # adducts it is hydrogen alone. Unknown elements share the trailing column, the same
        # bucket ``a_onehot`` gives them.
        composition_columns = th.full(
            (PERIODIC_TABLE.GetAtomicNumber("Og") + 1,),
            len(CANONICAL_ELEMENT_ORDER),
            dtype=th.long,
        )
        for column, element in enumerate(CANONICAL_ELEMENT_ORDER):
            composition_columns[PERIODIC_TABLE.GetAtomicNumber(element)] = column
        self.register_buffer("composition_element_columns", composition_columns)
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

    def _transferable_hydrogen_shifts(
        self,
        mol_pyg: Batch,
        atom_indices: th.Tensor,
        atom_fragment_indices: th.Tensor,
        fragment_batch_indices: th.Tensor,
        component_counts: th.Tensor,
        h_shifts: th.Tensor,
        fragment_cut_counts: th.Tensor | None,
        atom_numbers: th.Tensor,
        atom_hydrogens: th.Tensor,
        batch_size: int,
    ) -> th.Tensor:
        """Return which hydrogen shifts a source of hydrogen can actually supply.

        Hydrogen transfer moves hydrogens from the complementary neutral fragment, so a gain
        needs both a donor and somewhere to put the hydrogen. ``h_transfer_bound`` selects how
        much of that is enforced; see ``H_TRANSFER_BOUNDS``. Losses are bounded elsewhere, by
        the element counts staying nonnegative.

        Args:
            mol_pyg: Batched precursor graphs.
            atom_indices: Atom index for every sparse fragment membership.
            atom_fragment_indices: Fragment index for every membership.
            fragment_batch_indices: Spectrum index for every fragment.
            component_counts: Tracked element counts per fragment.
            h_shifts: The ladder, from ``-max_h_transfer`` to ``+max_h_transfer``.
            fragment_cut_counts: Severed bonds per fragment; required by ``valence``.
            atom_numbers: Atomic number per atom.
            atom_hydrogens: Hydrogen count per atom.
            batch_size: Number of spectra in the batch.

        Returns:
            Bool tensor of shape ``(num_fragments, num_h_shifts)``.
        """
        num_fragments = component_counts.shape[0]
        if self.h_transfer_bound == "none":
            return h_shifts.new_ones((num_fragments, h_shifts.numel()), dtype=th.bool)

        # The complement's hydrogens: everything the molecule has that this fragment does not.
        per_atom_hydrogens = (atom_numbers == 1).long() + atom_hydrogens.long()
        molecule_hydrogens = per_atom_hydrogens.new_zeros(batch_size).index_add(
            0, mol_pyg.batch, per_atom_hydrogens
        )
        max_gain = (
            molecule_hydrogens[fragment_batch_indices]
            - component_counts[:, self.hydrogen_count_column]
        )

        if self.h_transfer_bound == "valence":
            if fragment_cut_counts is None:
                raise ValueError(
                    "h_transfer_bound='valence' needs fragment_cut_counts to count the "
                    "valence slots freed by each severed bond"
                )
            if not hasattr(mol_pyg, "atom_free_valence_slots"):
                raise ValueError(
                    "h_transfer_bound='valence' needs `atom_free_valence_slots` on the "
                    "molecule graph; rebuild the graphs with the current featurizer"
                )
            # A fragment can hold its current hydrogens plus one per unsaturation it could
            # open, plus one per bond severed from the complement. FraGNNet computes the same
            # quantity as `sum(ve) - sum(retained single bonds)` in `compute_cc_h_cap`.
            slots = mol_pyg.atom_free_valence_slots.long()[atom_indices]
            fragment_slots = slots.new_zeros(num_fragments).index_add(
                0, atom_fragment_indices, slots
            )
            valence_gain = fragment_slots + fragment_cut_counts.to(fragment_slots)
            max_gain = th.minimum(max_gain, valence_gain)

        return h_shifts[None, :] <= max_gain.clamp_min(0)[:, None]

    def _fragment_masses_and_valid_channels(
        self,
        mol_pyg: Batch,
        atom_indices: th.Tensor,
        atom_fragment_indices: th.Tensor,
        fragment_batch_indices: th.Tensor,
        num_fragments: int,
        precursor_type_indices: th.Tensor,
        fragment_cut_counts: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Translate fragment atom sets into monoisotopic ion channels.

        Args:
            mol_pyg: Batched precursor graphs carrying the per-atom chemistry fields.
            atom_indices: Atom index for every sparse fragment membership.
            atom_fragment_indices: Fragment index for every membership.
            fragment_batch_indices: Spectrum index for every fragment.
            num_fragments: Number of fragment atom sets.
            precursor_type_indices: Adduct index per spectrum, shape ``(batch_size,)``.
            fragment_cut_counts: Severed bonds per fragment, required by the ``valence``
                hydrogen-transfer bound and ignored by the others.

        Returns:
            Tuple of ``(ion_mzs, valid_channels)``, both shaped
            ``(num_fragments, num_ion_modes, num_h_shifts)``. ``valid_channels`` is
            False wherever a channel would need more atoms of some element than the
            component actually contains, wherever the hydrogen transfer would take more
            hydrogens than the complementary neutral fragment has to give, or for a CMF
            channel the adduct does not define.
        """
        atom_hydrogens = mol_pyg.atom_hydrogen_counts
        atom_numbers = mol_pyg.atom_atomic_numbers

        # CRF/CMF deltas below apply the output ion's electron correction once,
        # irrespective of precursor atom formal charges.
        component_masses = component_neutral_masses(
            mol_pyg,
            atom_indices,
            atom_fragment_indices,
            num_fragments,
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
        transferable = self._transferable_hydrogen_shifts(
            mol_pyg,
            atom_indices,
            atom_fragment_indices,
            fragment_batch_indices,
            component_counts,
            h_shifts,
            fragment_cut_counts,
            atom_numbers,
            atom_hydrogens,
            precursor_type_indices.shape[0],
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
        crf_valid = (crf_counts >= 0).all(dim=2) & transferable
        crf_mzs = (
            component_masses[:, None]
            + self.crf_mass_deltas[component_precursor_types, None]
            + h_shifts.to(component_masses) * H_MASS
        )

        mz_channels = [crf_mzs]
        valid_channels = [crf_valid]
        if self.include_cmf:
            # CMF channel: the charge stays on this fragment while the proton/adduct
            # leaves. Non-degenerate adducts use the same full H-transfer ladder as
            # FraGNNet. Direct-charge adducts remain entirely invalid here when their
            # CMF base state duplicates a reachable CRF state.
            cmf_counts = (
                component_counts[:, None, :]
                + self.cmf_formula_deltas[component_precursor_types, None, :]
            )
            cmf_counts = cmf_counts.expand(-1, self.num_h_shifts, -1).clone()
            cmf_counts[:, :, self.hydrogen_count_column] += h_shifts
            cmf_valid = (cmf_counts >= 0).all(dim=2) & transferable
            cmf_valid &= self.cmf_precursor_valid[component_precursor_types, None]
            cmf_mzs = (
                component_masses[:, None]
                + self.cmf_mass_deltas[component_precursor_types, None]
                + h_shifts.to(component_masses) * H_MASS
            )
            mz_channels.append(cmf_mzs)
            valid_channels.append(cmf_valid)
        ion_mzs = th.stack(mz_channels, dim=1)
        return ion_mzs, th.stack(valid_channels, dim=1) & (ion_mzs > 0.0)

    def _boundary_embeddings(
        self,
        mol_pyg: Batch,
        atom_embeddings: th.Tensor,
        chemical_bond_embeddings: th.Tensor,
        boundary_bond_indices: th.Tensor,
        boundary_component_indices: th.Tensor,
        atom_indices: th.Tensor,
        atom_component_indices: th.Tensor,
        num_components: int,
    ) -> th.Tensor:
        """Pool each component's severed bonds into one embedding.

        The default averages the two endpoint atom embeddings, which is degenerate under the
        subsequent mean pool. With ``boundary_bond_encoder`` the bond is encoded from its
        oriented (retained, lost) endpoints instead, so the pairing survives pooling.

        Args:
            mol_pyg: Batched precursor graphs.
            atom_embeddings: Per-atom embeddings, shape ``(num_atoms, hidden)``.
            chemical_bond_embeddings: Endpoint-averaged bond embeddings, one per chemical bond.
            boundary_bond_indices: Bond index of each (bond, component) boundary pair.
            boundary_component_indices: Component index of each boundary pair.
            atom_indices: Atom index of each component-membership entry.
            atom_component_indices: Component index of each membership entry.
            num_components: Number of components.

        Returns:
            Boundary embedding per component, shape ``(num_components, hidden)``.
        """
        if self.boundary_bond_encoder is None:
            return pool_boundary_bond_embeddings(
                chemical_bond_embeddings,
                boundary_bond_indices,
                boundary_component_indices,
                num_components=num_components,
            )
        inside_atoms, outside_atoms = boundary_bond_endpoint_orientation(
            mol_pyg.chemical_bond_index,
            boundary_bond_indices,
            boundary_component_indices,
            atom_indices,
            atom_component_indices,
            num_atoms=atom_embeddings.shape[0],
        )
        bond_embeddings = encode_cut_bond_pairs(
            self.boundary_bond_encoder,
            atom_embeddings,
            inside_atoms,
            outside_atoms,
        )
        return pool_boundary_bond_embeddings(
            bond_embeddings,
            th.arange(bond_embeddings.shape[0], device=bond_embeddings.device),
            boundary_component_indices,
            num_components=num_components,
        )

    def _candidate_atom_dropout(
        self,
        atom_indices: th.Tensor,
        atom_component_indices: th.Tensor,
        num_components: int,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Drop a random share of each candidate's atoms before the component pool.

        Training-only regularization against memorizing exact atom sets: the pooled fragment
        embedding then sees a random subset of the candidate's atoms. The complement is derived
        by subtracting the candidate from its precursor, so the dropped atoms move to the
        complement side and both pools remain means over a well-formed atom set. Keeping the
        complement exact instead would leave the true candidate sum recoverable as
        ``precursor - complement``, which is the memorization route this targets. Every other
        candidate input -- sizes, masses, charges, compositions, cut counts and the boundary
        embedding -- keeps the true membership, so the intervention is confined to the pooled
        component embeddings.

        Args:
            atom_indices: Atom index of each component-membership entry.
            atom_component_indices: Component index of each membership entry.
            num_components: Number of components.

        Returns:
            The retained ``(atom_indices, atom_component_indices)`` pair, unchanged outside
            training. A component that loses every atom keeps all of them, since an empty pool
            has no mean.
        """
        if not self.training or self.candidate_atom_dropout <= 0.0:
            return atom_indices, atom_component_indices
        keep = (
            th.rand(atom_component_indices.shape, device=atom_component_indices.device)
            >= self.candidate_atom_dropout
        )
        kept_component_indices = atom_component_indices[keep]
        kept_sizes = th.zeros(
            num_components,
            dtype=th.long,
            device=atom_component_indices.device,
        ).index_add(0, kept_component_indices, th.ones_like(kept_component_indices))
        keep |= (kept_sizes == 0)[atom_component_indices]
        return atom_indices[keep], atom_component_indices[keep]

    def _candidate_boundary_dropout(self, boundary_embeddings: th.Tensor) -> th.Tensor:
        """Drop each candidate's whole boundary block with the configured probability.

        Inverted dropout at the block level: each row is zeroed with probability ``p`` and
        the surviving rows are scaled by ``1 / (1 - p)``, so ``E[output] = input`` and the
        token projection, which starts with a linear layer, sees the same expected input in
        training and evaluation. Training-only; the default ``p = 0`` returns the input
        without drawing random numbers, so existing configurations are bit-identical.

        Args:
            boundary_embeddings: Boundary block per candidate, shape
                ``(num_components, hidden)``.

        Returns:
            Tensor of the same shape: the input outside training or at ``p = 0``, otherwise
            the input with dropped rows zeroed and kept rows rescaled.
        """
        if not self.training or self.candidate_boundary_dropout <= 0.0:
            return boundary_embeddings
        keep_scale = F.dropout(
            boundary_embeddings.new_ones((boundary_embeddings.shape[0], 1)),
            p=self.candidate_boundary_dropout,
            training=True,
        )
        return boundary_embeddings * keep_scale

    def _eligible_smarts_templates(
        self,
        templates_by_molecule: tuple[tuple[MassterTemplate, ...], ...],
        precursor_type_indices: th.Tensor,
        fragmentation_mode_indices: th.Tensor | None,
    ) -> tuple[tuple[MassterTemplate, ...], ...]:
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
            filter_masster_templates(
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
        candidates: MassterCandidateBatch,
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
        spec_nce_weights: th.Tensor | None = None,
        spec_ace_weights: th.Tensor | None = None,
        static_graph: MassterStaticGraph | None = None,
        proposal_group_ids: th.Tensor | None = None,
        memorization_sink_indices: th.Tensor | None = None,
        return_annotation: bool = False,
    ) -> dict[str, th.Tensor]:
        """Generate a sparse monoisotopic spectrum for every precursor.

        Args:
            mol_pyg: Batched precursor graphs. Beyond ``x``/``edge_index``/``edge_attr``
                these must carry the MASSter chemistry fields listed in
                ``required_graph_fields`` (chemical-bond view of the graph plus per-atom
                masses, hydrogen counts, formal charges, and atomic numbers).
            spec_prec_type: Adduct index per spectrum, shape ``(batch_size,)``.
            spec_inst_type: Instrument index per spectrum, if the context uses it.
            spec_frag_mode: Fragmentation-mode index per spectrum, if the context uses it.
            spec_nce: Normalized collision energies, ragged over spectra.
            spec_nce_batch_idxs: Spectrum index per NCE value.
            spec_ace: Absolute collision energies, ragged over spectra.
            spec_ace_batch_idxs: Spectrum index per ACE value.
            spec_nce_weights: Optional per-reading NCE weights.
            spec_ace_weights: Optional per-reading ACE weights.
            static_graph: CPU copy of the graph indices kept across Lightning's device
                transfer, so mask selection only has to move the bond logits back.
                Must include matched ``smarts_templates`` when SMARTS is enabled;
                an empty tuple per molecule explicitly represents no matches.
            proposal_group_ids: Spectrum group ids; required by ``share_proposal_across_ev``.
            memorization_sink_indices: Sink row (``mol_id``) per spectrum, shape
                ``(batch_size,)``. Only a model with a memorization sink accepts it; ``None``
                runs the shared model, which is what evaluation always does.
            return_annotation: Also return what each output pathway is, for the per-peak
                annotation export. Inference only; nothing in training reads it.

        Returns:
            Dictionary with the sparse prediction contract plus diagnostics:
            - "pred_mzs"/"pred_batch_idxs" and exactly one of "pred_logprobs" or
              "pred_intensities": one entry per valid (component, ion mode, H shift)
              pathway, shape ``(num_pathways,)``.
            - "bond_break_logits"/"atom_charge_logits": raw head outputs.
            - with ``return_annotation``: "pred_pathway_fragment_idxs",
              "pred_pathway_ion_mode_idxs" (0 CRF, 1 CMF) and "pred_pathway_h_shifts",
              aligned with "pred_mzs"; and "fragment_atom_indices" /
              "fragment_atom_fragment_idxs", one entry per (fragment, atom) membership, with
              fragments indexed like "ion_h_fragment_batch_idxs".
            - remaining keys: per-spectrum counters and candidate-generation timings.

        Raises:
            ValueError: If ``mol_pyg`` lacks the MASSter chemistry fields or
                SMARTS is enabled without matched templates.
        """
        required_graph_fields = (
            "chemical_bond_index",
            "chemical_bond_attr",
            "atom_atomic_numbers",
            "atom_hydrogen_counts",
            "atom_monoisotopic_masses",
            "batch",
            "ptr",
        )
        required_graph_fields = (
            *required_graph_fields,
            *(
                field
                for field in candidate_feature_required_graph_fields(self.candidate_feature_specs)
                if field not in required_graph_fields
            ),
        )
        missing = [field for field in required_graph_fields if not hasattr(mol_pyg, field)]
        if missing:
            raise ValueError(
                "mol_pyg is missing MASSter chemistry fields; regenerate molecular "
                f"preprocessing caches: {missing}"
            )
        batch_size = int(mol_pyg.ptr.numel() - 1)
        precursor_type_indices = spec_prec_type.reshape(-1)
        context_outputs = self.context_encoder(
            batch_size=batch_size,
            precursor_types=precursor_type_indices,
            instrument_types=spec_inst_type,
            fragmentation_modes=spec_frag_mode,
            nce=spec_nce,
            nce_batch_indices=spec_nce_batch_idxs,
            ace=spec_ace,
            ace_batch_indices=spec_ace_batch_idxs,
            nce_weights=spec_nce_weights,
            ace_weights=spec_ace_weights,
            return_ce_features=self.intensity_ce_conditioned,
        )
        if self.intensity_ce_conditioned:
            context, spectrum_ce_features = context_outputs
        else:
            context, spectrum_ce_features = context_outputs, None

        # 1. Encode each intact precursor once and score every chemical bond.
        atom_inputs = (
            th.cat((mol_pyg.x, context[mol_pyg.batch]), dim=1) if self.context_in_gnn else mol_pyg.x
        )
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
                atom_inputs,
                mol_pyg.batch,
                mol_pyg.edge_index,
                mol_pyg.edge_attr,
            )
        # The boundary block reads the shared embeddings, or a local cut-site encoder that sees
        # only raw atom features and k hops, so the cut term cannot name the molecule.
        boundary_atom_embeddings = (
            atom_embeddings
            if self.boundary_local_encoder is None
            else self.boundary_local_encoder(
                mol_pyg.x,
                mol_pyg.batch,
                mol_pyg.edge_index,
                mol_pyg.edge_attr,
            )
        )
        sink_rows = None
        if memorization_sink_indices is not None:
            if self.memorization_sink is None:
                raise ValueError("memorization_sink_indices given to a model without a sink")
            memorization_sink_indices = memorization_sink_indices.reshape(-1)
            if memorization_sink_indices.shape != (batch_size,):
                raise ValueError("memorization_sink_indices must hold one index per spectrum")
            _validate_tensor_condition(
                (
                    (memorization_sink_indices >= 0)
                    & (memorization_sink_indices < self.memorization_sink.num_embeddings)
                ).all(),
                "memorization_sink_indices must lie in [0, memorization_sink_num_molecules)",
            )
            sink_rows = self.memorization_sink(memorization_sink_indices)
        # ``chemical_bond_index`` is the undirected single-entry-per-bond view of the
        # graph (``edge_index`` is directed), so either endpoint identifies the molecule.
        bond_batch_indices = mol_pyg.batch[mol_pyg.chemical_bond_index[0]]
        detach_proposal_inputs = (
            self.proposal_selection_only and not self.proposal_input_gradients_enabled
        )
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
                if detach_proposal_inputs:
                    shared_proposal_atom_embeddings = shared_proposal_atom_embeddings.detach()
            else:
                shared_proposal_atom_embeddings = self.proposal_encoder(
                    mol_pyg.x[shared_atom_indices],
                    layout["shared_atom_batch"],
                    layout["shared_edge_index"],
                    mol_pyg.edge_attr[layout["representative_edge_mask"]],
                )
            proposal_atom_embeddings = shared_proposal_atom_embeddings[layout["atom_to_shared"]]
            if self.ion_mode_proposal_head is not None:
                proposal_context = context.detach() if detach_proposal_inputs else context
            shared_proposal_context = (
                shared_context.detach() if detach_proposal_inputs else shared_context
            )
            mechanism_context = shared_proposal_context[spectrum_to_shared]
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
                    atom_embeddings.detach() if detach_proposal_inputs else atom_embeddings
                )
            else:
                # Keep proposal-only supervision from entering the shared context
                # encoder when the existing proposal input routing is detached.
                proposal_atom_inputs = (
                    th.cat(
                        (
                            mol_pyg.x,
                            (context.detach() if detach_proposal_inputs else context)[
                                mol_pyg.batch
                            ],
                        ),
                        dim=1,
                    )
                    if self.context_in_gnn
                    else mol_pyg.x
                )
                proposal_atom_embeddings = self.proposal_encoder(
                    proposal_atom_inputs,
                    mol_pyg.batch,
                    mol_pyg.edge_index,
                    mol_pyg.edge_attr,
                )
            proposal_context = context.detach() if detach_proposal_inputs else context
            mechanism_context = proposal_context
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
            if self.proposal_method == "connected_fragment_beam":
                atom_charge_logits = bond_logits.new_zeros(mol_pyg.batch.numel())
            else:
                atom_charge_logits = self.charge_head(
                    proposal_atom_embeddings,
                    context=proposal_context,
                    atom_batch_indices=mol_pyg.batch,
                )

        # 2. Select cut-mask pathways or direct fragments with detached search scores.
        # Recompute probabilities from live logits for proposal supervision.
        smarts_templates: tuple[tuple[MassterTemplate, ...], ...] = tuple(
            () for _ in range(batch_size)
        )
        if self.use_smarts_templates:
            if static_graph is None or static_graph.smarts_templates is None:
                raise ValueError(
                    "SMARTS-enabled models require static_graph.smarts_templates; "
                    "use MassterPL with mol_smiles or supply matched templates"
                )
            smarts_templates = self._eligible_smarts_templates(
                static_graph.smarts_templates,
                precursor_type_indices,
                spec_frag_mode,
            )
        if self.proposal_method == "connected_fragment_beam":
            candidates = build_fast_gnn_candidates(
                bond_logits,
                mol_pyg.chemical_bond_index,
                mol_pyg.batch,
                mol_pyg.ptr,
                max_fragments=self.max_masks,
                beam_width=self.fast_gnn_beam_width,
                max_boundary_cuts=self.fast_gnn_max_boundary_cuts,
            )
        else:
            candidates = build_masster_candidates(
                bond_logits,
                mol_pyg.chemical_bond_index,
                mol_pyg.batch,
                mol_pyg.ptr,
                max_masks=self.max_masks,
                min_mask_relative_probability=self.min_mask_relative_probability,
                include_single_bond_cuts=self.include_single_bond_cuts,
                max_unique_components=(self.max_masks if self.unique_fragment_mode else None),
                max_enumerated_masks=(
                    self.max_enumerated_masks if self.unique_fragment_mode else None
                ),
                mask_selection_threads=self.mask_selection_threads,
                candidate_backend=self.candidate_backend,
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
            proposal_bond_embeddings = (
                proposal_atom_embeddings[mol_pyg.chemical_bond_index[0]]
                + proposal_atom_embeddings[mol_pyg.chemical_bond_index[1]]
            ) / 2.0
            mechanism_boundary_embeddings = pool_boundary_bond_embeddings(
                proposal_bond_embeddings,
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
                mechanism_context[mechanism_batch_indices],
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
        # Marginal fragment selection replaces the proposal outputs with whole-pool ones.
        pool_proposal_outputs: dict[str, th.Tensor] = {}
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
            if self.proposal_method == "connected_fragment_beam":
                # One unique fragment per adapter row: no latent charge competition.
                component_log_probabilities = candidates.mask_log_probabilities
                fragment_log_probabilities = component_log_probabilities
            else:
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
            ion_mzs, valid_channels = self._fragment_masses_and_valid_channels(
                mol_pyg,
                fragment_atom_indices,
                atom_fragment_indices,
                fragment_batch_indices,
                num_fragments,
                precursor_type_indices,
                # Bonds severed at this fragment's own boundary, which is what frees a
                # valence slot. Intra-fragment breaks are excluded, matching the retained-bond
                # count FraGNNet's `update_bonds` computes.
                fragment_cut_counts,
            )
            valid_fragments = valid_channels.flatten(1).any(dim=1)
            if self.fragment_selection == "marginal":
                # Rank the pool by each fragment's own release probability, not by the mask
                # that first produced it, and train the proposal on the whole pool.
                selection_scores = _fragment_release_log_probabilities(
                    bond_logits,
                    mol_pyg.chemical_bond_index,
                    mol_pyg.batch.numel(),
                    fragment_atom_indices,
                    atom_fragment_indices,
                    fragment_boundary_bond_indices,
                    boundary_fragment_indices,
                    num_fragments,
                )
                # Mask order places the intact and single-cut masks first; keep them too.
                forced_fragments = fragment_cut_counts <= int(self.include_single_bond_cuts)
                fragment_keep = _top_k_fragment_keep_mask(
                    selection_scores.masked_fill(~valid_fragments, -th.inf),
                    fragment_batch_indices,
                    forced_fragments,
                    batch_size,
                    self.selected_fragment_target,
                )
                pool_flat_valid = valid_channels.flatten()
                pool_candidate_fragment_idxs = (
                    th.arange(num_fragments, device=fragment_batch_indices.device)
                    .reshape(-1, 1, 1)
                    .expand_as(valid_channels)
                    .flatten()[pool_flat_valid]
                )
                pool_candidate_mzs = ion_mzs.flatten()[pool_flat_valid]
                pool_release_log_normalizers = _segment_logsumexp(
                    selection_scores,
                    fragment_batch_indices,
                    batch_size,
                )
                pool_proposal_outputs = {
                    "proposal_candidate_mzs": pool_candidate_mzs,
                    "proposal_candidate_fragment_idxs": pool_candidate_fragment_idxs,
                    "proposal_fragment_logprobs": (
                        selection_scores - pool_release_log_normalizers[fragment_batch_indices]
                    ),
                    "proposal_fragment_batch_idxs": fragment_batch_indices,
                    "pool_candidate_mzs": pool_candidate_mzs,
                    "pool_candidate_batch_idxs": fragment_batch_indices[
                        pool_candidate_fragment_idxs
                    ],
                }
            else:
                fragment_keep = _relative_fragment_keep_mask(
                    fragment_log_probabilities,
                    fragment_batch_indices,
                    batch_size,
                    self.min_fragment_relative_probability,
                )
            # A probability cutoff must not discard the last realizable fragment.
            if self.fragment_selection == "marginal":
                # Release scores tie exactly on symmetric substructures, so the fallback adds
                # a single fragment, and only where the budget kept no realizable one.
                fragment_keep |= _missing_valid_fallback_mask(
                    selection_scores,
                    valid_fragments,
                    fragment_keep,
                    fragment_batch_indices,
                    batch_size,
                )
            else:
                valid_scores = fragment_log_probabilities.detach().masked_fill(
                    ~valid_fragments, -th.inf
                )
                best_valid_scores = valid_scores.new_full((batch_size,), -th.inf).scatter_reduce(
                    0,
                    fragment_batch_indices,
                    valid_scores,
                    reduce="amax",
                    include_self=True,
                )
                fragment_keep |= valid_fragments & (
                    valid_scores == best_valid_scores[fragment_batch_indices]
                )
            ion_mzs = ion_mzs[fragment_keep]
            valid_channels = valid_channels[fragment_keep]
            fragment_count_before_cutoff = th.bincount(
                fragment_batch_indices,
                minlength=batch_size,
            ).to(bond_logits)
            fragment_count_after_cutoff = th.bincount(
                fragment_batch_indices[fragment_keep],
                minlength=batch_size,
            ).to(bond_logits)
            # Autocast keeps exp() in float32 while bond_logits stays bfloat16, so the
            # accumulator is built in the contribution dtype and cast back afterwards.
            retained_fragment_contributions = base_fragment_log_probabilities.exp() * (
                fragment_keep.to(base_fragment_log_probabilities.dtype)
            )
            retained_fragment_probability = (
                retained_fragment_contributions.new_zeros(batch_size)
                .index_add(0, fragment_batch_indices, retained_fragment_contributions)
                .to(bond_logits)
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
            pooled_atom_indices, pooled_atom_fragment_indices = self._candidate_atom_dropout(
                fragment_atom_indices,
                atom_fragment_indices,
                num_fragments,
            )
            fragment_embeddings, complement_embeddings = pool_component_and_complement_embeddings(
                atom_embeddings,
                pooled_atom_indices,
                pooled_atom_fragment_indices,
                fragment_batch_indices,
                mol_pyg.batch,
                num_components=num_fragments,
                num_molecules=batch_size,
            )
            boundary_embeddings = (
                self._boundary_embeddings(
                    mol_pyg,
                    boundary_atom_embeddings,
                    chemical_bond_embeddings,
                    fragment_boundary_bond_indices,
                    boundary_fragment_indices,
                    fragment_atom_indices,
                    atom_fragment_indices,
                    num_fragments,
                )
                if self.candidate_boundary_block
                else fragment_embeddings.new_zeros(fragment_embeddings.shape)
            )
            # The TRUE atom membership, never the subset `_candidate_atom_dropout` returned
            # above: that dropout perturbs the pooled embeddings alone, and every descriptor
            # here keeps the real atom set.
            candidate_inputs = CandidateFeatureInputs(
                mol_pyg=mol_pyg,
                atom_indices=fragment_atom_indices,
                atom_component_indices=atom_fragment_indices,
                boundary_bond_indices=fragment_boundary_bond_indices,
                boundary_component_indices=boundary_fragment_indices,
                component_batch_indices=fragment_batch_indices,
                cut_counts=fragment_cut_counts,
                composition_element_columns=self.composition_element_columns,
                num_components=num_fragments,
                batch_size=batch_size,
                dtype=fragment_embeddings.dtype,
            )
            fragment_candidate_features = build_candidate_features(
                self.candidate_feature_specs,
                candidate_inputs,
            )
            if self.light_side_token:
                # Each cut enters by its lighter side: the heavier side's pool averages most of
                # the molecule, which names it. The precursor's lighter side is its empty
                # complement, which pools to zeros.
                fragment_embeddings = th.where(
                    candidate_inputs.fragment_is_lighter_side.unsqueeze(1),
                    fragment_embeddings,
                    complement_embeddings,
                )
                complement_embeddings = th.zeros_like(complement_embeddings)
            fragment_sizes = candidate_inputs.component_sizes
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
            # Each unique fragment is its own mask here, so the head's per-mask
            # component count is one everywhere and its component-level residual is
            # discarded in favor of the mask-pooled one.
            intensity_outputs = self.intensity_head(
                fragment_embeddings,
                complement_embeddings,
                self._candidate_boundary_dropout(boundary_embeddings),
                th.zeros_like(fragment_log_probabilities),
                fragment_log_probabilities,
                fragment_batch_indices,
                fragment_indices,
                batch_size=batch_size,
                num_masks=num_fragments,
                context=context[fragment_batch_indices],
                return_support_logits=(self.intensity_head.ion_h_support_head is not None),
                component_neutral_masses=(
                    candidate_inputs.component_neutral_masses
                    if self.intensity_mass_pair_bias
                    else None
                ),
                ce_features=spectrum_ce_features,
                candidate_features=fragment_candidate_features,
                sink_vectors=None if sink_rows is None else sink_rows[fragment_batch_indices],
            )
            _, fragment_residuals, ion_h_logits = intensity_outputs[:3]
            escape_logits = intensity_outputs[-1] if self.intensity_escape else None
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
            pooled_atom_indices, pooled_atom_component_indices = self._candidate_atom_dropout(
                candidates.atom_indices,
                candidates.atom_component_indices,
                num_fragments,
            )
            fragment_embeddings, complement_embeddings = pool_component_and_complement_embeddings(
                atom_embeddings,
                pooled_atom_indices,
                pooled_atom_component_indices,
                fragment_batch_indices,
                mol_pyg.batch,
                num_components=num_fragments,
                num_molecules=batch_size,
            )
            boundary_embeddings = (
                self._boundary_embeddings(
                    mol_pyg,
                    boundary_atom_embeddings,
                    chemical_bond_embeddings,
                    candidates.boundary_bond_indices,
                    candidates.boundary_component_indices,
                    candidates.atom_indices,
                    candidates.atom_component_indices,
                    num_fragments,
                )
                if self.candidate_boundary_block
                else fragment_embeddings.new_zeros(fragment_embeddings.shape)
            )
            # The TRUE atom membership; see the unique-fragment branch above.
            candidate_inputs = CandidateFeatureInputs(
                mol_pyg=mol_pyg,
                atom_indices=candidates.atom_indices,
                atom_component_indices=candidates.atom_component_indices,
                boundary_bond_indices=candidates.boundary_bond_indices,
                boundary_component_indices=candidates.boundary_component_indices,
                component_batch_indices=fragment_batch_indices,
                cut_counts=fragment_cut_counts,
                composition_element_columns=self.composition_element_columns,
                num_components=num_fragments,
                batch_size=batch_size,
                dtype=fragment_embeddings.dtype,
            )
            fragment_candidate_features = build_candidate_features(
                self.candidate_feature_specs,
                candidate_inputs,
            )
            fragment_sizes = candidate_inputs.component_sizes
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
                self._candidate_boundary_dropout(boundary_embeddings),
                charge_log_probabilities + component_mechanism_log_probabilities,
                original_component_mask_log_probabilities,
                fragment_batch_indices,
                candidates.component_mask_indices,
                batch_size=batch_size,
                num_masks=num_masks,
                context=context[fragment_batch_indices],
                return_support_logits=(self.intensity_head.ion_h_support_head is not None),
                component_neutral_masses=(
                    candidate_inputs.component_neutral_masses
                    if self.intensity_mass_pair_bias
                    else None
                ),
                ce_features=spectrum_ce_features,
                candidate_features=fragment_candidate_features,
            )
            component_residuals, mask_residuals, ion_h_logits = intensity_outputs[:3]
            escape_logits = intensity_outputs[-1] if self.intensity_escape else None
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
                # `component_cut_counts` carries the whole mask's cut count on every one of
                # its components (`masster_core.pyx:851`), so for a mask that yields
                # three or more pieces this over-counts an end piece's own boundary. It is an
                # upper bound, so the valence bound stays conservative here: it can leave an
                # unreachable channel in place, never remove a reachable one. The
                # unique-fragment path above uses exact per-fragment boundary counts.
                fragment_cut_counts,
            )

        # 4. Remove fragments that cannot form an ion under the configured adduct.
        chemically_valid_fragments = valid_channels.flatten(1).any(dim=1)
        if return_annotation:
            # The true atom memberships of the fragment rows, re-indexed onto the fragments
            # that survive this step. Nothing downstream reads memberships, so they are
            # carried only for the export.
            if self.unique_fragment_mode:
                membership_atoms = fragment_atom_indices
                membership_fragments = atom_fragment_indices
            else:
                membership_atoms = candidates.atom_indices
                membership_fragments = candidates.atom_component_indices
            old_to_new_valid = chemically_valid_fragments.to(th.long).cumsum(0) - 1
            membership_keep = chemically_valid_fragments[membership_fragments]
            annotation_atom_indices = membership_atoms[membership_keep]
            annotation_atom_fragment_indices = old_to_new_valid[
                membership_fragments[membership_keep]
            ]
        fragment_batch_indices = fragment_batch_indices[chemically_valid_fragments]
        fragment_cut_counts = fragment_cut_counts[chemically_valid_fragments]
        ion_mzs = ion_mzs[chemically_valid_fragments]
        valid_channels = valid_channels[chemically_valid_fragments]
        ion_h_logits = ion_h_logits[chemically_valid_fragments]
        ion_h_support_logits = ion_h_support_logits[chemically_valid_fragments]
        if learned_ion_mode_proposal_logits is not None:
            learned_ion_mode_proposal_logits = learned_ion_mode_proposal_logits[
                chemically_valid_fragments
            ]
        if self.intensity_output == "independent":
            corrected_fragment_logits = corrected_fragment_logits[chemically_valid_fragments]
        # Normalize the final and proposal distributions together over valid fragments.
        fragment_scores = th.stack(
            (fragment_log_probabilities, base_fragment_log_probabilities), dim=1
        )[chemically_valid_fragments]
        score_batch_indices = 2 * fragment_batch_indices[:, None] + th.arange(
            2,
            device=fragment_batch_indices.device,
        )
        normalizers = _segment_logsumexp(
            fragment_scores.flatten(),
            score_batch_indices.flatten(),
            2 * batch_size,
        ).reshape(batch_size, 2)
        fragment_log_probabilities, base_fragment_log_probabilities = (
            fragment_scores - normalizers[fragment_batch_indices]
        ).unbind(dim=1)
        num_fragments = int(fragment_batch_indices.numel())

        # 5. Separate full chemical support from selected spectral output.
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
        retained_ion_h_probability = bond_logits.new_zeros(batch_size).index_add(
            0,
            fragment_batch_indices,
            retained_intensity_mass.to(bond_logits),
        ) / fragment_counts.clamp_min(1)
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
        retained_joint_intensity_fraction = (
            retained_joint_intensity
            / total_joint_intensity.clamp_min(th.finfo(total_joint_intensity.dtype).tiny)
        )
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
        retained_ion_mode_probability = bond_logits.new_zeros(batch_size).index_add(
            0,
            fragment_batch_indices,
            retained_mode_mass.to(bond_logits),
        ) / fragment_counts.clamp_min(1)
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
            "candidate_topology_assembly_batch_ms": candidates.timings.topology_assembly_ms,
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
            # Proposal probability of each fragment the intensity head scores, aligned with
            # ion_h_fragment_batch_idxs. The proposal-loss tensors cannot stand in for it:
            # marginal selection gives those the whole enumerated pool.
            "ion_h_fragment_proposal_logprobs": base_fragment_log_probabilities,
            "valid_ion_h_count": valid_ion_h_count,
            "retained_ion_h_count": retained_ion_h_count,
            "retained_ion_h_probability": retained_ion_h_probability,
            "retained_joint_channel_intensity_fraction": retained_joint_intensity_fraction,
            "ion_h_cutoff_pruned_fraction": 1.0
            - retained_ion_h_count / valid_ion_h_count.clamp_min(1),
            "valid_ion_mode_count": valid_ion_mode_count,
            "retained_ion_mode_count": retained_ion_mode_count,
            "retained_ion_mode_probability": retained_ion_mode_probability,
            "ion_mode_cutoff_pruned_fraction": (
                1.0 - retained_ion_mode_count / valid_ion_mode_count.clamp_min(1)
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
            if escape_logits is not None:
                # sigmoid(x) is the escape's share of (candidates + escape), so the mass it
                # needs is that share's odds times the candidate mass -- and the odds of a
                # sigmoid are just exp(x). Going through sigmoid and back would break under
                # reduced precision: fp16 and bf16 round any guard of the form 1 - 1e-4 to
                # exactly 1.0, so the share saturates and the odds become inf. Clamping the
                # logit instead bounds the odds at ~8.1e3, which fp16 holds comfortably, and
                # caps the escape share at 0.9999. At the head's initial bias the share is
                # 0.1, near the measured unreachable fraction.
                escape_odds = escape_logits.to(pred_intensities.dtype).clamp(max=9.0).exp()
                candidate_mass = pred_intensities.new_zeros(batch_size).index_add(
                    0, pred_batch_idxs, pred_intensities
                )
                output["pred_escape_intensities"] = candidate_mass * escape_odds
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
        output.update(pool_proposal_outputs)
        if return_annotation:
            mode_channel_indices = (
                th.arange(valid_channels.shape[1], device=valid_channels.device)
                .reshape(1, -1, 1)
                .expand_as(valid_channels)
            )
            h_channel_shifts = (
                th.arange(
                    -self.max_h_transfer, self.max_h_transfer + 1, device=valid_channels.device
                )
                .reshape(1, 1, -1)
                .expand_as(valid_channels)
            )
            output.update(
                {
                    "pred_pathway_fragment_idxs": fragment_channel_indices.flatten()[
                        selected_flat_valid
                    ],
                    "pred_pathway_ion_mode_idxs": mode_channel_indices.flatten()[
                        selected_flat_valid
                    ],
                    "pred_pathway_h_shifts": h_channel_shifts.flatten()[selected_flat_valid],
                    "fragment_atom_indices": annotation_atom_indices,
                    "fragment_atom_fragment_idxs": annotation_atom_fragment_indices,
                }
            )
        return output
