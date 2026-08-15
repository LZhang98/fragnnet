"""MASShatter: one-pass bond-set fragmentation spectrum prediction."""

from __future__ import annotations

from collections.abc import Sequence

import torch as th
import torch.nn as nn
from torch_geometric.data import Batch

from fragnnet.frag.mass_shatter_candidates import (
    MassShatterCandidateBatch,
    build_mass_shatter_candidates,
)
from fragnnet.model.bond_fragment_heads import (
    AtomChargeAffinityHead,
    EndpointBondBreakHead,
    JointIonHydrogenHead,
    component_charge_log_probabilities,
    masked_ion_hydrogen_log_probabilities,
    pool_boundary_bond_embeddings,
    pool_complement_embeddings,
    pool_component_embeddings,
)
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
    """Encode acquisition metadata once per spectrum."""

    def __init__(
        self,
        *,
        num_precursor_types: int,
        num_instrument_types: int,
        num_fragmentation_modes: int,
        use_nce: bool,
        use_ace: bool,
        metadata_embedding_dim: int,
        context_dim: int,
        nce_mean: float,
        nce_std: float,
        ace_mean: float,
        ace_std: float,
    ) -> None:
        super().__init__()
        if num_precursor_types <= 0:
            raise ValueError("num_precursor_types must be positive")
        if metadata_embedding_dim <= 0 or context_dim <= 0:
            raise ValueError("context dimensions must be positive")
        if nce_std <= 0.0 or ace_std <= 0.0:
            raise ValueError("CE standard deviations must be positive")

        self.use_nce = use_nce
        self.use_ace = use_ace
        self.nce_mean = nce_mean
        self.nce_std = nce_std
        self.ace_mean = ace_mean
        self.ace_std = ace_std
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
        input_dim = metadata_embedding_dim
        input_dim += int(use_nce) + int(use_ace)
        input_dim += metadata_embedding_dim * int(self.instrument_embedding is not None)
        input_dim += metadata_embedding_dim * int(self.fragmentation_mode_embedding is not None)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

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
        """Return one conditioning vector per spectrum."""
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
            mean_nce = _mean_values_by_batch(nce, nce_batch_indices, batch_size)
            features.append(((mean_nce - self.nce_mean) / self.nce_std).unsqueeze(1))
        if self.use_ace:
            if ace is None:
                raise ValueError("ACE values are required by the configured context")
            mean_ace = _mean_values_by_batch(ace, ace_batch_indices, batch_size)
            features.append(((mean_ace - self.ace_mean) / self.ace_std).unsqueeze(1))
        return self.projector(th.cat(features, dim=1))


class MassShatterModel(nn.Module):
    """Predict sparse MS/MS spectra from globally ranked precursor bond masks."""

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
        precursor_types: Sequence[str],
        instrument_types: Sequence[str],
        fragmentation_modes: Sequence[str],
        use_nce: bool,
        use_ace: bool,
        nce_mean: float,
        nce_std: float,
        ace_mean: float,
        ace_std: float,
        max_masks: int,
        include_single_bond_cuts: bool,
        max_h_transfer: int,
        include_cmf: bool,
        charge_pooling: str,
    ) -> None:
        super().__init__()
        if max_masks <= 0 or max_h_transfer < 0:
            raise ValueError("max_masks must be positive and max_h_transfer nonnegative")
        if charge_pooling not in {"logsumexp", "logmeanexp"}:
            raise ValueError(f"unknown charge pooling mode: {charge_pooling}")

        self.precursor_types = tuple(sorted(precursor_types))
        if not self.precursor_types:
            raise ValueError("precursor_types must not be empty")
        missing_precursors = set(self.precursor_types) - PREC_TYPE_TO_MASS_DIFF.keys()
        if missing_precursors:
            raise ValueError(f"unsupported precursor types: {sorted(missing_precursors)}")
        self.max_masks = max_masks
        self.include_single_bond_cuts = include_single_bond_cuts
        self.max_h_transfer = max_h_transfer
        self.include_cmf = include_cmf
        self.charge_pooling = charge_pooling
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
            metadata_embedding_dim=metadata_embedding_dim,
            context_dim=context_dim,
            nce_mean=nce_mean,
            nce_std=nce_std,
            ace_mean=ace_mean,
            ace_std=ace_std,
        )
        self.bond_break_head = EndpointBondBreakHead(
            precursor_hidden_dim,
            edge_feature_dim,
            context_dim,
            head_hidden_dim,
            head_dropout,
        )
        self.charge_head = AtomChargeAffinityHead(
            precursor_hidden_dim,
            context_dim,
            head_hidden_dim,
            head_dropout,
        )
        self.ion_hydrogen_head = JointIonHydrogenHead(
            precursor_hidden_dim,
            context_dim,
            head_hidden_dim,
            self.num_ion_modes,
            self.num_h_shifts,
            head_dropout,
        )
        self._setup_chemistry_buffers()

    def _setup_chemistry_buffers(self) -> None:
        """Precompute per-adduct mass and validity deltas."""

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
        self.hydrogen_count_column = required_atomic_numbers.index(1)
        atomic_number_to_column = {
            atomic_number: column for column, atomic_number in enumerate(required_atomic_numbers)
        }

        def formula_deltas(parsed: list[dict[str, int]]) -> th.Tensor:
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
        """Translate component atom sets into monoisotopic ion channels."""
        num_components = candidates.num_components
        atom_indices = candidates.atom_indices
        component_indices = candidates.atom_component_indices
        atom_masses = mol_pyg.atom_monoisotopic_masses
        atom_hydrogens = mol_pyg.atom_hydrogen_counts
        atom_charges = mol_pyg.atom_formal_charges
        atom_numbers = mol_pyg.atom_atomic_numbers

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
    ) -> dict[str, th.Tensor]:
        """Generate a sparse monoisotopic spectrum for every precursor."""
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
        bond_batch_indices = mol_pyg.batch[mol_pyg.chemical_bond_index[0]]
        bond_logits = self.bond_break_head(
            atom_embeddings,
            mol_pyg.chemical_bond_index,
            mol_pyg.chemical_bond_attr,
            context=context,
            bond_batch_indices=bond_batch_indices,
        )

        # 2. Select global masks and construct sparse component pathways.
        candidates = build_mass_shatter_candidates(
            bond_logits,
            mol_pyg.chemical_bond_index,
            mol_pyg.batch,
            mol_pyg.ptr,
            max_masks=self.max_masks,
            include_single_bond_cuts=self.include_single_bond_cuts,
        )
        num_components = candidates.num_components
        fragment_embeddings = pool_component_embeddings(
            atom_embeddings,
            candidates.atom_indices,
            candidates.atom_component_indices,
            num_components=num_components,
        )
        complement_embeddings = pool_complement_embeddings(
            atom_embeddings,
            candidates.complement_atom_indices,
            candidates.complement_component_indices,
            num_components=num_components,
        )
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
        charge_log_probabilities = component_charge_log_probabilities(
            atom_charge_logits,
            candidates.atom_indices,
            candidates.atom_component_indices,
            candidates.component_mask_indices,
            pooling=self.charge_pooling,
        )
        ion_h_logits = self.ion_hydrogen_head(
            fragment_embeddings,
            complement_embeddings,
            boundary_embeddings,
            candidates.component_cut_counts,
            context=context[candidates.component_batch_indices],
        )
        ion_mzs, valid_channels = self._component_masses_and_valid_channels(
            mol_pyg,
            candidates,
            precursor_type_indices,
        )
        ion_h_log_probabilities = masked_ion_hydrogen_log_probabilities(
            ion_h_logits,
            valid_channels,
        )

        # 4. Flatten valid pathways into SpectrumPL's sparse prediction contract.
        component_log_probabilities = (
            candidates.mask_log_probabilities[candidates.component_mask_indices]
            + charge_log_probabilities
        )
        pathway_log_probabilities = (
            component_log_probabilities[:, None, None] + ion_h_log_probabilities
        )
        flat_valid = valid_channels.flatten()
        pred_mzs = ion_mzs.flatten()[flat_valid]
        pred_logprobs = pathway_log_probabilities.flatten()[flat_valid]
        pred_batch_idxs = (
            candidates.component_batch_indices[:, None, None]
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
        return {
            "pred_mzs": pred_mzs,
            "pred_logprobs": pred_logprobs,
            "pred_batch_idxs": pred_batch_idxs,
            "pred_oos_logprobs": pred_oos_logprobs,
            "bond_break_logits": bond_logits,
            "atom_charge_logits": atom_charge_logits,
            "retained_mask_probability": 1.0 - pred_oos_logprobs.exp(),
            "selected_mask_count": selected_mask_count,
            "component_pathway_count": component_pathway_count,
        }
