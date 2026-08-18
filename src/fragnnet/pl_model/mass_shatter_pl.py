"""PyTorch Lightning training wrapper for MASShatter."""

from __future__ import annotations

from typing import Any

import torch as th

from fragnnet.model import MassShatterModel
from fragnnet.model.loss import ce_switch_pair_bce, get_sparse_cross_entropy_fn
from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.utils.feat_utils import get_mol_feats_sizes


class MassShatterPL(SpectrumPL):
    """Train MASShatter with SpectrumPL and FraGNNet's CE-pair objective."""

    hparams: Any

    def _setup_model(self) -> None:
        if self.hparams.compile:
            raise ValueError(
                "MASShatter candidate selection is CPU/Python code; set compile: False"
            )
        if not self.hparams.spec_params["prec_type"]:
            raise ValueError("MASShatter requires spec_params.prec_type: True")
        node_feature_dim, edge_feature_dim = get_mol_feats_sizes(
            self.hparams.mol_params["pyg_node_feats"],
            self.hparams.mol_params["pyg_edge_feats"],
            self.hparams.mol_params["pyg_pe_embed_k"],
        )
        instrument_types = (
            self.hparams.spec_params["inst_types"] if self.hparams.spec_params["inst_type"] else []
        )
        fragmentation_modes = (
            self.hparams.spec_params["frag_modes"] if self.hparams.spec_params["frag_mode"] else []
        )
        self.model = MassShatterModel(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            precursor_hidden_dim=self.hparams.mol_hidden_size,
            precursor_num_layers=self.hparams.mol_num_layers,
            precursor_gnn_type=self.hparams.mol_gnn_type,
            precursor_normalization=self.hparams.mol_normalization,
            precursor_dropout=self.hparams.mol_dropout,
            context_dim=self.hparams.mass_shatter_context_dim,
            metadata_embedding_dim=self.hparams.mass_shatter_metadata_embedding_dim,
            head_hidden_dim=self.hparams.mass_shatter_head_hidden_dim,
            head_dropout=self.hparams.mass_shatter_head_dropout,
            bond_num_layers=self.hparams.mass_shatter_bond_num_layers,
            intensity_transformer_dim=(
                self.hparams.mass_shatter_intensity_transformer_dim
            ),
            intensity_transformer_num_layers=(
                self.hparams.mass_shatter_intensity_transformer_num_layers
            ),
            intensity_transformer_num_heads=(
                self.hparams.mass_shatter_intensity_transformer_num_heads
            ),
            intensity_transformer_feedforward_dim=(
                self.hparams.mass_shatter_intensity_transformer_feedforward_dim
            ),
            intensity_transformer_num_inducing_points=(
                self.hparams.mass_shatter_intensity_transformer_num_inducing_points
            ),
            intensity_transformer_dropout=(
                self.hparams.mass_shatter_intensity_transformer_dropout
            ),
            precursor_types=self.hparams.spec_params["prec_types"],
            instrument_types=instrument_types,
            fragmentation_modes=fragmentation_modes,
            use_nce=self.hparams.spec_params["nce"],
            use_ace=self.hparams.spec_params["ace"],
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            ace_mean=self.hparams.ace_mean,
            ace_std=self.hparams.ace_std,
            max_masks=self.hparams.mass_shatter_max_masks,
            explore_masks=self.hparams.mass_shatter_explore_masks,
            exploration_mode=self.hparams.mass_shatter_exploration_mode,
            exploration_temperature=(
                self.hparams.mass_shatter_exploration_temperature
            ),
            exploration_cut_count_weights=(
                self.hparams.mass_shatter_exploration_cut_count_weights
            ),
            topology_cache_size=self.hparams.mass_shatter_topology_cache_size,
            include_single_bond_cuts=(self.hparams.mass_shatter_include_single_bond_cuts),
            max_h_transfer=self.hparams.mass_shatter_max_h_transfer,
            include_cmf=self.hparams.mass_shatter_include_cmf,
            charge_pooling=self.hparams.mass_shatter_charge_pooling,
        )
        self._check_ce_params()

    def _setup_loss_fn(self) -> None:
        """Use FraGNNet's sparse CE and adjacent-CE switch supervision."""
        if self.hparams.loss_type != "cross_entropy":
            raise ValueError("the first MASShatter training path supports loss_type: cross_entropy")
        if not self.hparams.sparse_cosine_similarity or not self.hparams.sum_ints:
            raise ValueError("MASShatter requires sparse_cosine_similarity and sum_ints")
        if self.hparams.ce_pair_min_endpoint_reliability is not None:
            raise ValueError("ce_pair_min_endpoint_reliability is not implemented")
        self.binned_loss = False
        sparse_ce_fn = get_sparse_cross_entropy_fn(
            dist=self.hparams.output_distribution,
            vectorized=self.hparams.loss_vectorized,
            tolerance=self.tolerance,
            relative=self.relative,
            tolerance_min_mz=self.tolerance_min_mz,
            oos_tolerance_multiple=self.hparams.oos_tolerance_multiple,
            gaussian_renormalize=self.hparams.gaussian_renormalize,
            pm_tolerance_multiple=self.hparams.pm_tolerance_multiple,
            loss_batch_size=self.hparams.loss_batch_size,
        )

        instrument_types = sorted(self.hparams.spec_params["inst_types"])
        precursor_types = sorted(self.hparams.spec_params["prec_types"])
        eligible_instrument_indices = th.empty(0, dtype=th.long)
        eligible_precursor_indices = th.empty(0, dtype=th.long)
        if self.hparams.ce_pair_loss_weight != 0.0:
            eligible_instrument_indices = th.tensor(
                [
                    instrument_types.index(instrument_type)
                    for instrument_type in self.hparams.ce_pair_eligible_inst_types
                ],
                dtype=th.long,
            )
            eligible_precursor_indices = th.tensor(
                [
                    precursor_types.index(precursor_type)
                    for precursor_type in self.hparams.ce_pair_eligible_prec_types
                ],
                dtype=th.long,
            )

        def loss_fn(
            true_mzs: th.Tensor,
            true_logprobs: th.Tensor,
            true_batch_idxs: th.Tensor,
            pred_mzs: th.Tensor,
            pred_logprobs: th.Tensor,
            pred_batch_idxs: th.Tensor,
            pred_oos_logprobs: th.Tensor,
            retained_mask_probability: th.Tensor,
            selected_mask_count: th.Tensor,
            exploration_mask_count: th.Tensor,
            exploration_mean_cut_count: th.Tensor,
            component_pathway_count: th.Tensor,
            topology_cache_hit_rate: th.Tensor,
            candidate_input_transfer_batch_ms: th.Tensor,
            candidate_mask_selection_batch_ms: th.Tensor,
            candidate_topology_batch_ms: th.Tensor,
            candidate_assembly_batch_ms: th.Tensor,
            candidate_output_transfer_batch_ms: th.Tensor,
            candidate_total_batch_ms: th.Tensor,
            candidate_total_molecule_ms: th.Tensor,
            **kwargs,
        ) -> dict[str, th.Tensor]:
            tolerance_per_sample = kwargs.get("tol_per_sample")
            minimum_mz_per_sample = kwargs.get("min_mz_per_sample")
            ios_ce, oos_ce, true_oos_logprobs, _ = sparse_ce_fn(
                true_mzs,
                true_logprobs,
                true_batch_idxs,
                pred_mzs,
                pred_logprobs,
                pred_batch_idxs,
                pred_oos_logprobs,
                tol_per_sample=tolerance_per_sample,
                min_mz_per_sample=minimum_mz_per_sample,
            )
            primary_loss = ios_ce + oos_ce if self.hparams.oos_loss else ios_ce
            loss = self.hparams.primary_loss_weight * primary_loss
            loss_values = {
                "ios_ce": ios_ce,
                "oos_ce": oos_ce,
                "spec_ce": ios_ce + oos_ce,
                "primary_loss": primary_loss,
                "candidate_wrecall": 1.0 - true_oos_logprobs.exp(),
                "retained_mask_probability": retained_mask_probability.detach(),
                "selected_mask_count": selected_mask_count.detach().to(pred_logprobs),
                "exploration_mask_count": exploration_mask_count.detach().to(pred_logprobs),
                "exploration_mean_cut_count": (
                    exploration_mean_cut_count.detach().to(pred_logprobs)
                ),
                "component_pathway_count": component_pathway_count.detach().to(pred_logprobs),
                "topology_cache_hit_rate": topology_cache_hit_rate.detach(),
                "candidate_input_transfer_batch_ms": (candidate_input_transfer_batch_ms.detach()),
                "candidate_mask_selection_batch_ms": (candidate_mask_selection_batch_ms.detach()),
                "candidate_topology_batch_ms": candidate_topology_batch_ms.detach(),
                "candidate_assembly_batch_ms": candidate_assembly_batch_ms.detach(),
                "candidate_output_transfer_batch_ms": (candidate_output_transfer_batch_ms.detach()),
                "candidate_total_batch_ms": candidate_total_batch_ms.detach(),
                "candidate_total_molecule_ms": candidate_total_molecule_ms.detach(),
            }

            if self.hparams.ce_pair_loss_weight != 0.0:
                batch_size = int(kwargs["batch_size"])
                pair_loss = pred_logprobs.new_zeros(batch_size)
                eligible_endpoints = pred_logprobs.new_zeros(batch_size)
                if kwargs["ce_pair_enabled"]:
                    instrument_indices = kwargs["spec_inst_type"].reshape(-1)
                    precursor_indices = kwargs["spec_prec_type"].reshape(-1)
                    eligible_samples = th.isin(
                        instrument_indices,
                        eligible_instrument_indices.to(instrument_indices.device),
                    ) & th.isin(
                        precursor_indices,
                        eligible_precursor_indices.to(precursor_indices.device),
                    )
                    pair_loss, eligible_endpoints = ce_switch_pair_bce(
                        true_mzs=kwargs["ce_pair_true_mzs"],
                        true_ints=kwargs["ce_pair_true_ints"],
                        true_batch_idxs=kwargs["ce_pair_true_batch_idxs"],
                        pred_mzs=kwargs["ce_pair_pred_mzs"],
                        pred_logprobs=kwargs["ce_pair_pred_logprobs"],
                        pred_batch_idxs=kwargs["ce_pair_pred_batch_idxs"],
                        group_ids=kwargs["group_id"].reshape(-1),
                        ce_values=kwargs["ce_pair_ce_values"],
                        eligible_samples=eligible_samples,
                        batch_size=batch_size,
                        min_endpoint_confidence=(self.hparams.ce_pair_min_endpoint_confidence),
                        min_contender_relative_intensity=(self.hparams.target_rel_ints_thresh),
                        tol_per_sample=tolerance_per_sample,
                        min_mz_per_sample=minimum_mz_per_sample,
                    )
                loss = loss + self.hparams.ce_pair_loss_weight * pair_loss
                loss_values["ce_pair_loss"] = pair_loss
                loss_values["ce_pair_eligible_endpoints"] = eligible_endpoints
            loss_values["loss"] = loss
            return loss_values

        self.loss_fn = loss_fn
        loss_names = [
            "loss",
            "primary_loss",
            "ios_ce",
            "oos_ce",
            "spec_ce",
            "candidate_wrecall",
            "retained_mask_probability",
            "selected_mask_count",
            "exploration_mask_count",
            "exploration_mean_cut_count",
            "component_pathway_count",
            "topology_cache_hit_rate",
            "candidate_input_transfer_batch_ms",
            "candidate_mask_selection_batch_ms",
            "candidate_topology_batch_ms",
            "candidate_assembly_batch_ms",
            "candidate_output_transfer_batch_ms",
            "candidate_total_batch_ms",
            "candidate_total_molecule_ms",
        ]
        if self.hparams.ce_pair_loss_weight != 0.0:
            loss_names.extend(["ce_pair_loss", "ce_pair_eligible_endpoints"])
        self.loss_names = loss_names
        self.metric_names.update(loss_names)

    def forward(self, **batch_kwargs) -> dict[str, th.Tensor]:
        """Select the MASShatter inputs from SpectrumPL's collated batch."""
        return self.model(
            mol_pyg=batch_kwargs["mol_pyg"],
            spec_prec_type=batch_kwargs["spec_prec_type"],
            spec_inst_type=batch_kwargs.get("spec_inst_type"),
            spec_frag_mode=batch_kwargs.get("spec_frag_mode"),
            spec_nce=batch_kwargs.get("spec_nce"),
            spec_nce_batch_idxs=batch_kwargs.get("spec_nce_batch_idxs"),
            spec_ace=batch_kwargs.get("spec_ace"),
            spec_ace_batch_idxs=batch_kwargs.get("spec_ace_batch_idxs"),
        )
