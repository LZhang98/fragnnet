"""MASShatter: one-pass bond-set fragmentation spectrum prediction.

Unlike the DAG-based FraGNNet path, MASShatter never enumerates fragments ahead of
time. A single GNN pass over the intact precursor produces one break logit per
chemical bond; those independent Bernoulli logits define a distribution over all
``2 ** num_bonds`` cut masks, from which only the top ``max_masks`` are materialized.
Each retained mask is cut into connected components, and every component is expanded
into the ion channels it could plausibly produce (charge retention vs charge
migration, plus +/- hydrogen transfers). The resulting sparse
(m/z, log-probability, batch index) triples are what ``SpectrumPL`` consumes.

The probability of an observed pathway therefore factorizes as::

    P(mask) * P(component keeps the charge | mask) * P(ion mode, H shift | component)

with the mass of the un-enumerated masks reported separately as
``pred_oos_logprobs`` (out-of-support).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch as th
import torch.nn as nn
from torch_geometric.data import Batch

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterCandidateBatch,
    MassShatterStaticGraph,
    build_mass_shatter_candidates,
)
from fragnnet.model.bond_fragment_heads import (
    AtomChargeAffinityHead,
    BondFocusedBreakHead,
    CandidateIntensityTransformerHead,
    EndpointBondBreakHead,
    _segment_logsumexp,
    component_charge_log_probabilities,
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
    if not bool((counts > 0).all()):
        raise ValueError("every spectrum must have at least one CE value")
    return sums / counts


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
        if not bool((counts > 0).all()):
            raise ValueError("every spectrum must have at least one CE value")
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
            if nce is None:
                raise ValueError("NCE values are required by the configured context")
            if self.nce_embedding is None:
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
            if ace is None:
                raise ValueError("ACE values are required by the configured context")
            if self.ace_embedding is None:
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
    2. Rank whole cut masks by their Bernoulli probability, keep the top
       ``max_masks``, and cut each into connected components.
    3. Score which component keeps the charge, then which ion mode / hydrogen-transfer
       channel that component emits; a transformer over a molecule's candidate set
       supplies residual corrections so components compete with one another.
    4. Flatten every chemically valid channel into the sparse prediction contract.

    Attributes:
        max_masks: Number of cut masks materialized per precursor.
        min_mask_probability: Masks below this probability are dropped.
        explore_masks: Number of random exploration masks added during training.
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
        precursor_gnn_type: str,
        precursor_normalization: str,
        precursor_dropout: float,
        context_dim: int,
        metadata_embedding_dim: int,
        head_hidden_dim: int,
        head_dropout: float,
        bond_num_layers: int,
        intensity_transformer_dim: int,
        intensity_transformer_num_layers: int,
        intensity_transformer_num_heads: int,
        intensity_transformer_feedforward_dim: int,
        intensity_transformer_num_inducing_points: int,
        intensity_transformer_dropout: float,
        use_mask_residuals: bool,
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
        min_mask_probability: float,
        explore_masks: int,
        exploration_mode: str,
        exploration_temperature: float,
        exploration_cut_count_weights: Sequence[float],
        include_single_bond_cuts: bool,
        max_h_transfer: int,
        include_cmf: bool,
        charge_pooling: str,
    ) -> None:
        """Initialize the MASShatter model.

        Args:
            node_feature_dim: Width of the precursor atom features.
            edge_feature_dim: Width of the precursor bond features.
            precursor_hidden_dim: Hidden width of the precursor GNN; also the width of
                the pooled fragment/complement/boundary embeddings.
            precursor_num_layers: Number of message-passing layers.
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
            intensity_transformer_dim: Model width of the candidate transformer.
            intensity_transformer_num_layers: Depth of the candidate transformer.
            intensity_transformer_num_heads: Attention heads per layer.
            intensity_transformer_feedforward_dim: Feed-forward width per layer.
            intensity_transformer_num_inducing_points: Latent tokens used to keep
                attention linear in the number of candidates.
            intensity_transformer_dropout: Dropout inside the candidate transformer.
            use_mask_residuals: Whether transformer residuals may redistribute retained
                probability between selected masks.
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
            min_mask_probability: Probability floor below which a ranked mask is
                dropped; its mass falls into the out-of-support bucket.
            explore_masks: Random masks injected during training only.
            exploration_mode: ``bernoulli_replace`` swaps exploration masks in for the
                weakest ranked masks; ``bounded_additive`` appends them.
            exploration_temperature: Temperature applied to bond logits when sampling
                exploration masks.
            exploration_cut_count_weights: Sampling weights indexed by cut count; the
                0- and 1-cut entries must be zero because those masks are either the
                intact precursor or already covered by the ranked enumeration.
            include_single_bond_cuts: Enumerate single-bond cuts in the ranked masks.
            max_h_transfer: Largest hydrogen transfer per fragment, in either
                direction.
            include_cmf: Emit the charge-migration ion channel alongside CRF.
            charge_pooling: ``logsumexp`` treats every atom as a potential charge site
                (favoring large components); ``logmeanexp`` removes that size prior.

        Raises:
            ValueError: If any hyperparameter is out of range or an adduct in
                ``precursor_types`` has no tabulated mass difference.
        """
        super().__init__()
        if max_masks <= 0 or max_h_transfer < 0:
            raise ValueError("max_masks must be positive and max_h_transfer nonnegative")
        if not math.isfinite(min_mask_probability) or not 0.0 <= min_mask_probability <= 1.0:
            raise ValueError(
                "min_mask_probability must be finite and in [0, 1], "
                f"got {min_mask_probability}"
            )
        if explore_masks < 0:
            raise ValueError(f"explore_masks must be nonnegative, got {explore_masks}")
        if exploration_mode not in {"bernoulli_replace", "bounded_additive"}:
            raise ValueError(f"unknown exploration mode: {exploration_mode}")
        if exploration_mode == "bernoulli_replace" and explore_masks >= max_masks:
            raise ValueError(
                "bernoulli_replace requires explore_masks smaller than max_masks, "
                f"got {explore_masks} and {max_masks}"
            )
        if not math.isfinite(exploration_temperature) or exploration_temperature <= 0.0:
            raise ValueError(
                f"exploration_temperature must be positive, got {exploration_temperature}"
            )
        cut_count_weights = tuple(float(weight) for weight in exploration_cut_count_weights)
        if (
            len(cut_count_weights) < 3
            or any(not math.isfinite(weight) or weight < 0.0 for weight in cut_count_weights)
            or cut_count_weights[0] != 0.0
            or cut_count_weights[1] != 0.0
            or sum(cut_count_weights[2:]) <= 0.0
        ):
            raise ValueError(
                "exploration_cut_count_weights must contain nonnegative weights indexed by "
                "cut count, with zero weights for 0/1 cuts and positive mass above 1"
            )
        if bond_num_layers < 0:
            raise ValueError("bond_num_layers must be nonnegative")
        if charge_pooling not in {"logsumexp", "logmeanexp"}:
            raise ValueError(f"unknown charge pooling mode: {charge_pooling}")

        # Sorting fixes the adduct index order that ``spec_prec_type`` refers to and
        # that the registered chemistry buffers are laid out along.
        self.precursor_types = tuple(sorted(precursor_types))
        if not self.precursor_types:
            raise ValueError("precursor_types must not be empty")
        missing_precursors = set(self.precursor_types) - PREC_TYPE_TO_MASS_DIFF.keys()
        if missing_precursors:
            raise ValueError(f"unsupported precursor types: {sorted(missing_precursors)}")
        self.max_masks = max_masks
        self.min_mask_probability = min_mask_probability
        self.explore_masks = explore_masks
        self.exploration_mode = exploration_mode
        self.exploration_temperature = exploration_temperature
        self.exploration_cut_count_weights = cut_count_weights
        self.include_single_bond_cuts = include_single_bond_cuts
        self.max_h_transfer = max_h_transfer
        self.include_cmf = include_cmf
        self.charge_pooling = charge_pooling
        self.use_mask_residuals = use_mask_residuals
        # Every component fans out into num_ion_modes * num_h_shifts candidate ions:
        # charge retention (CRF) and optionally charge migration (CMF), crossed with
        # hydrogen transfers from -max_h_transfer to +max_h_transfer.
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
        )
        self.context_encoder = MassShatterContextEncoder(
            num_precursor_types=len(self.precursor_types),
            num_instrument_types=len(instrument_types),
            num_fragmentation_modes=len(fragmentation_modes),
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
            )
        self.charge_head = AtomChargeAffinityHead(
            precursor_hidden_dim,
            context_dim,
            head_hidden_dim,
            head_dropout,
        )
        self.intensity_head = CandidateIntensityTransformerHead(
            fragment_embedding_dim=precursor_hidden_dim,
            context_dim=context_dim,
            model_dim=intensity_transformer_dim,
            num_layers=intensity_transformer_num_layers,
            num_heads=intensity_transformer_num_heads,
            feedforward_dim=intensity_transformer_feedforward_dim,
            num_inducing_points=intensity_transformer_num_inducing_points,
            num_ion_modes=self.num_ion_modes,
            num_h_shifts=self.num_h_shifts,
            dropout=intensity_transformer_dropout,
        )
        self._setup_chemistry_buffers()

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
        # Not every adduct has a meaningful charge-migration counterpart (a CMF row
        # equal to its own CRF row would only duplicate m/z values).
        self.register_buffer(
            "cmf_precursor_valid",
            th.tensor(
                [p in PREC_TYPE_TO_CMF_MASS_DIFF for p in self.precursor_types],
                dtype=th.bool,
            ),
        )

    def _component_masses_and_valid_channels(
        self,
        mol_pyg: Batch,
        candidates: MassShatterCandidateBatch,
        precursor_type_indices: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        """Translate component atom sets into monoisotopic ion channels.

        Args:
            mol_pyg: Batched precursor graphs carrying the per-atom chemistry fields.
            candidates: Sparse component pathways selected for this batch.
            precursor_type_indices: Adduct index per spectrum, shape ``(batch_size,)``.

        Returns:
            Tuple of ``(ion_mzs, valid_channels)``, both shaped
            ``(num_components, num_ion_modes, num_h_shifts)``. ``valid_channels`` is
            False wherever a channel would need more atoms of some element than the
            component actually contains (or a CMF channel the adduct does not define).
        """
        num_components = candidates.num_components
        atom_indices = candidates.atom_indices
        component_indices = candidates.atom_component_indices
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
        component_masses = atom_masses.new_zeros(num_components).index_add(
            0,
            component_indices,
            per_atom_neutral_mass[atom_indices],
        )

        component_counts = th.zeros(
            (num_components, self.required_atomic_numbers.numel()),
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
            component_counts[:, column].index_add_(0, component_indices, values)

        component_precursor_types = precursor_type_indices[candidates.component_batch_indices]
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

    @staticmethod
    def _oos_log_probabilities(
        candidates: MassShatterCandidateBatch,
        batch_size: int,
    ) -> th.Tensor:
        """Return the log probability mass left in the un-enumerated masks.

        Only ``max_masks`` of the ``2 ** num_bonds`` masks are materialized, so the
        selected probabilities do not sum to one. The remainder, ``log(1 - retained)``,
        is handed to the loss as the out-of-support bucket. Clamping keeps the log
        finite when the retained mass rounds to exactly one.

        Args:
            candidates: Sparse component pathways selected for this batch.
            batch_size: Number of spectra in the batch.

        Returns:
            Out-of-support log probabilities of shape ``(batch_size,)``.
        """
        selected_probabilities = candidates.mask_log_probabilities.exp()
        retained_mass = selected_probabilities.new_zeros(batch_size).index_add(
            0,
            candidates.mask_batch_indices,
            selected_probabilities,
        )
        eps = th.finfo(retained_mass.dtype).eps
        return th.log1p(-retained_mass.clamp(max=1.0 - eps))

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
            - "pred_mzs"/"pred_logprobs"/"pred_batch_idxs": one entry per valid
              (component, ion mode, H shift) pathway, shape ``(num_pathways,)``.
            - "pred_oos_logprobs": un-enumerated mask mass, shape ``(batch_size,)``.
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

        # 1. Encode the intact precursor once and score every chemical bond.
        atom_embeddings = self.precursor_encoder(
            mol_pyg.x,
            mol_pyg.batch,
            mol_pyg.edge_index,
            mol_pyg.edge_attr,
        )
        # ``chemical_bond_index`` is the undirected single-entry-per-bond view of the
        # graph (``edge_index`` is directed), so either endpoint identifies the molecule.
        bond_batch_indices = mol_pyg.batch[mol_pyg.chemical_bond_index[0]]
        bond_logits = self.bond_break_head(
            atom_embeddings,
            mol_pyg.chemical_bond_index,
            mol_pyg.chemical_bond_attr,
            context=context,
            bond_batch_indices=bond_batch_indices,
        )

        # 2. Select global masks and construct sparse component pathways.
        # Mask identities are picked from detached logits on CPU; the returned log
        # probabilities are recomputed from the live logits, so gradients still reach
        # every cut/intact bond decision. Exploration masks are training-only.
        candidates = build_mass_shatter_candidates(
            bond_logits,
            mol_pyg.chemical_bond_index,
            mol_pyg.batch,
            mol_pyg.ptr,
            max_masks=self.max_masks,
            min_mask_probability=self.min_mask_probability,
            explore_masks=self.explore_masks if self.training else 0,
            exploration_mode=self.exploration_mode,
            exploration_temperature=self.exploration_temperature,
            exploration_cut_count_weights=self.exploration_cut_count_weights,
            include_single_bond_cuts=self.include_single_bond_cuts,
            static_graph=static_graph,
        )
        # Describe every component three ways: the atoms it kept, the atoms it lost
        # (its complement within the precursor), and the bonds that were cut to make it.
        num_components = candidates.num_components
        fragment_embeddings, complement_embeddings = pool_component_and_complement_embeddings(
            atom_embeddings,
            candidates.atom_indices,
            candidates.atom_component_indices,
            candidates.component_batch_indices,
            mol_pyg.batch,
            num_components=num_components,
            num_molecules=batch_size,
        )
        # Averaging the endpoints keeps the bond embedding invariant to endpoint order.
        chemical_bond_embeddings = (
            atom_embeddings[mol_pyg.chemical_bond_index[0]]
            + atom_embeddings[mol_pyg.chemical_bond_index[1]]
        ) / 2.0
        boundary_embeddings = pool_boundary_bond_embeddings(
            chemical_bond_embeddings,
            candidates.boundary_bond_indices,
            candidates.boundary_component_indices,
            num_components=num_components,
        )

        # 3. Score charge retention and conditional ion/H-transfer outcomes.
        atom_charge_logits = self.charge_head(
            atom_embeddings,
            context=context,
            atom_batch_indices=mol_pyg.batch,
        )
        # Pool atom charge affinities into a per-component score, normalized across the
        # sibling components of the same mask: exactly one of them carries the charge.
        charge_log_probabilities = component_charge_log_probabilities(
            atom_charge_logits,
            candidates.atom_indices,
            candidates.atom_component_indices,
            candidates.component_mask_indices,
            pooling=self.charge_pooling,
        )
        num_masks = candidates.mask_log_probabilities.shape[0]
        component_sizes = th.bincount(
            candidates.atom_component_indices,
            minlength=num_components,
        )
        original_component_mask_log_probabilities = candidates.mask_log_probabilities[
            candidates.component_mask_indices
        ]
        # The transformer sees a molecule's whole candidate set at once, so it can move
        # probability between competing masks and between siblings inside a mask. It
        # returns additive corrections in log space, never absolute probabilities.
        component_residuals, mask_residuals, ion_h_logits = self.intensity_head(
            fragment_embeddings,
            complement_embeddings,
            boundary_embeddings,
            candidates.component_cut_counts,
            component_sizes,
            charge_log_probabilities,
            original_component_mask_log_probabilities,
            candidates.component_batch_indices,
            candidates.component_mask_indices,
            batch_size=batch_size,
            num_masks=num_masks,
            context=context[candidates.component_batch_indices],
        )
        if not self.use_mask_residuals:
            mask_residuals = th.zeros_like(mask_residuals)

        # Reweight the masks: renormalize the corrected logits within each spectrum, then
        # rescale to the retained mass the Bernoulli enumeration originally held. The
        # residuals therefore redistribute probability among selected masks without
        # changing how much mass is out of support (see ``_oos_log_probabilities``).
        mask_logits = candidates.mask_log_probabilities + mask_residuals
        original_retained_log_probabilities = _segment_logsumexp(
            candidates.mask_log_probabilities,
            candidates.mask_batch_indices,
            batch_size,
        )
        mask_log_normalizers = _segment_logsumexp(
            mask_logits,
            candidates.mask_batch_indices,
            batch_size,
        )
        mask_log_probabilities = (
            mask_logits
            - mask_log_normalizers[candidates.mask_batch_indices]
            + original_retained_log_probabilities[candidates.mask_batch_indices]
        )
        component_mask_log_probabilities = mask_log_probabilities[
            candidates.component_mask_indices
        ]
        # Same treatment one level down: corrected charge scores are renormalized over
        # the sibling components of each mask, which stay a proper distribution.
        component_charge_logits = charge_log_probabilities + component_residuals
        component_log_normalizers = _segment_logsumexp(
            component_charge_logits,
            candidates.component_mask_indices,
            num_masks,
        )
        charge_log_probabilities = component_charge_logits - component_log_normalizers[
            candidates.component_mask_indices
        ]
        ion_mzs, valid_channels = self._component_masses_and_valid_channels(
            mol_pyg,
            candidates,
            precursor_type_indices,
        )
        # Chemically impossible channels are masked out before the softmax, so each
        # component spends all of its probability on channels it can actually produce.
        ion_h_log_probabilities = masked_ion_hydrogen_log_probabilities(
            ion_h_logits,
            valid_channels,
        )

        # 4. Flatten valid pathways into SpectrumPL's sparse prediction contract.
        # Chain rule in log space: P(mask) * P(charge | mask) * P(ion mode, H | component).
        component_log_probabilities = (
            component_mask_log_probabilities + charge_log_probabilities
        )
        pathway_log_probabilities = (
            component_log_probabilities[:, None, None] + ion_h_log_probabilities
        )
        # Drop the masked-out channels; what survives is one flat peak list per batch.
        flat_valid = valid_channels.flatten()
        pred_mzs = ion_mzs.flatten()[flat_valid]
        pred_logprobs = pathway_log_probabilities.flatten()[flat_valid]
        pred_batch_idxs = (
            candidates.component_batch_indices[:, None, None]
            .expand_as(valid_channels)
            .flatten()[flat_valid]
        )
        candidate_cut_counts = (
            candidates.component_cut_counts[:, None, None]
            .expand_as(valid_channels)
            .flatten()[flat_valid]
        )
        pred_oos_logprobs = self._oos_log_probabilities(candidates, batch_size)
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
        mean_mask_cut_count = bond_logits.new_zeros(batch_size).index_add(
            0,
            candidates.mask_batch_indices,
            candidates.mask_cut_counts.to(bond_logits),
        ) / selected_mask_count_float
        mask_cut_fractions = {}
        cut_count_buckets = {
            "0": candidates.mask_cut_counts == 0,
            "1": candidates.mask_cut_counts == 1,
            "2": candidates.mask_cut_counts == 2,
            "3": candidates.mask_cut_counts == 3,
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
        return {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
            "pred_oos_logprobs": pred_oos_logprobs,
            "candidate_mzs_by_cut": pred_mzs,
            "candidate_batch_idxs_by_cut": pred_batch_idxs,
            "candidate_cut_counts": candidate_cut_counts,
            "bond_break_logits": bond_logits,
            "atom_charge_logits": atom_charge_logits,
            "retained_mask_probability": 1.0 - pred_oos_logprobs.exp(),
            "selected_mask_count": selected_mask_count,
            "exploration_mask_count": candidates.exploration_mask_count,
            "exploration_mean_cut_count": candidates.exploration_mean_cut_count,
            "component_pathway_count": component_pathway_count,
            "mean_mask_cut_count": mean_mask_cut_count,
            "proposal_single_cut_fraction": candidates.proposal_single_cut_fraction,
            "unique_component_count": candidates.unique_component_count,
            "components_per_mask": component_pathway_count_float / selected_mask_count_float,
            "unique_components_per_mask": (
                candidates.unique_component_count / selected_mask_count_float
            ),
            **mask_cut_fractions,
            **candidate_timing_tensors,
        }
