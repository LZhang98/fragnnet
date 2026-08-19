"""PyTorch Lightning training wrapper for MASShatter.

MASShatter emits the same sparse ``(pred_mzs, pred_logprobs, pred_batch_idxs,
pred_oos_logprobs)`` contract as FraGNNet, so all of SpectrumPL's preprocessing,
metric, and reduction machinery is reused unchanged. This module only supplies the
three model-specific pieces:

- :meth:`MassShatterPL._setup_model` maps flat hparams onto ``MassShatterModel``.
- :meth:`MassShatterPL._setup_loss_fn` wraps the sparse cross entropy and the
  adjacent-CE switch objective, and forwards MASShatter's candidate-generation
  diagnostics into the logged metrics.
- :meth:`MassShatterPL.transfer_batch_to_device` keeps a CPU copy of the graph
  indices, because mask selection runs in Python on the host.
"""

from __future__ import annotations

from typing import Any

import torch as th

from fragnnet.frag.mass_shatter_candidates import MassShatterStaticGraph
from fragnnet.model import MassShatterModel
from fragnnet.model.loss import ce_switch_pair_bce, get_sparse_cross_entropy_fn
from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.spec_utils import calculate_match_mzs

_CUT_COUNT_BUCKETS = {
    "0": lambda cut_counts: cut_counts == 0,
    "1": lambda cut_counts: cut_counts == 1,
    "2": lambda cut_counts: cut_counts == 2,
    "3": lambda cut_counts: cut_counts == 3,
    "4plus": lambda cut_counts: cut_counts >= 4,
}


def _candidate_coverage_by_cut_count(
    *,
    true_mzs: th.Tensor,
    true_ints: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_batch_idxs: th.Tensor,
    candidate_cut_counts: th.Tensor,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
) -> dict[str, th.Tensor]:
    """Measure true-peak coverage using candidates from one cut-count bucket.

    A true peak may be covered by multiple buckets, so the resulting recall values
    describe each bucket's standalone support and are not additive.
    """
    batch_size = int(true_batch_idxs.max().item()) + 1
    metrics = {
        f"candidate_{metric}_cut_{bucket}": true_ints.new_zeros(batch_size)
        for bucket in _CUT_COUNT_BUCKETS
        for metric in ("recall", "wrecall")
    }
    for batch_index in range(batch_size):
        true_mask = true_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_true_ints = true_ints[true_mask]
        if batch_true_mzs.numel() == 0:
            raise ValueError(f"spectrum {batch_index} has no true peaks")
        true_intensity = batch_true_ints.sum()
        for bucket, select_bucket in _CUT_COUNT_BUCKETS.items():
            candidate_mask = (candidate_batch_idxs == batch_index) & select_bucket(
                candidate_cut_counts
            )
            if not candidate_mask.any():
                continue
            match_mask = calculate_match_mzs(
                batch_true_mzs,
                candidate_mzs[candidate_mask],
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
            true_match_mask = match_mask.any(dim=1)
            metrics[f"candidate_recall_cut_{bucket}"][batch_index] = true_match_mask.to(
                batch_true_ints
            ).mean()
            metrics[f"candidate_wrecall_cut_{bucket}"][batch_index] = (
                batch_true_ints * true_match_mask
            ).sum() / true_intensity
    return metrics


class MassShatterPL(SpectrumPL):
    """Train MASShatter with SpectrumPL and FraGNNet's CE-pair objective.

    Attributes:
        model: The wrapped :class:`MassShatterModel`.
        loss_fn: Closure built in :meth:`_setup_loss_fn`; called with the batch merged
            with the model's output dictionary, returning per-spectrum vectors that
            SpectrumPL reduces.
        loss_names: Keys of ``loss_fn``'s output that get logged every step.
    """

    hparams: Any

    def _setup_model(self) -> None:
        """Instantiate ``MassShatterModel`` from the flat hparam namespace.

        Raises:
            ValueError: If ``compile`` is on, or the spectrum config does not provide
                the precursor type MASShatter needs to compute adduct-dependent m/z.
        """
        # Candidate selection runs Python/CPU code inside forward, which dynamo cannot
        # trace; compiling would graph-break every batch.
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
        # An empty vocabulary tells the context encoder to skip that embedding entirely,
        # so a disabled metadata field costs no parameters.
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
            intensity_transformer_dim=(self.hparams.mass_shatter_intensity_transformer_dim),
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
            intensity_transformer_dropout=(self.hparams.mass_shatter_intensity_transformer_dropout),
            use_mask_residuals=self.hparams.mass_shatter_use_mask_residuals,
            precursor_types=self.hparams.spec_params["prec_types"],
            instrument_types=instrument_types,
            fragmentation_modes=fragmentation_modes,
            int_embedder=self.hparams.int_embedder,
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_size=self.hparams.ce_insert_size,
            use_nce=self.hparams.spec_params["nce"],
            use_ace=self.hparams.spec_params["ace"],
            nce_max=self.hparams.nce_max,
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            ace_max=self.hparams.ace_max,
            ace_mean=self.hparams.ace_mean,
            ace_std=self.hparams.ace_std,
            max_masks=self.hparams.mass_shatter_max_masks,
            unique_fragment_mode=self.hparams.mass_shatter_unique_fragment_mode,
            max_enumerated_masks=(self.hparams.mass_shatter_max_enumerated_masks),
            min_mask_probability=self.hparams.mass_shatter_min_mask_probability,
            explore_masks=self.hparams.mass_shatter_explore_masks,
            exploration_mode=self.hparams.mass_shatter_exploration_mode,
            exploration_temperature=(self.hparams.mass_shatter_exploration_temperature),
            exploration_cut_count_weights=(self.hparams.mass_shatter_exploration_cut_count_weights),
            include_single_bond_cuts=(self.hparams.mass_shatter_include_single_bond_cuts),
            max_h_transfer=self.hparams.mass_shatter_max_h_transfer,
            include_cmf=self.hparams.mass_shatter_include_cmf,
            charge_pooling=self.hparams.mass_shatter_charge_pooling,
        )
        self._check_ce_params()

    def _setup_loss_fn(self) -> None:
        """Use FraGNNet's sparse CE and adjacent-CE switch supervision.

        Builds ``self.loss_fn`` as a closure over the configured sparse cross entropy
        and, when ``ce_pair_loss_weight`` is nonzero, the CE-switch BCE term. Also
        registers the metric names so SpectrumPL logs MASShatter's candidate-generation
        diagnostics alongside the losses.

        Raises:
            ValueError: If the configured loss/spectrum options are outside the subset
                this training path supports.
        """
        if self.hparams.loss_type != "cross_entropy":
            raise ValueError("the first MASShatter training path supports loss_type: cross_entropy")
        if not self.hparams.sparse_cosine_similarity or not self.hparams.sum_ints:
            raise ValueError("MASShatter requires sparse_cosine_similarity and sum_ints")
        if self.hparams.ce_pair_min_endpoint_reliability is not None:
            raise ValueError("ce_pair_min_endpoint_reliability is not implemented")
        # Predictions are exact monoisotopic m/z values, so targets stay unbinned and
        # are matched against predictions by instrument tolerance instead.
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

        # The CE-pair term only applies to configured instrument/adduct combinations.
        # Translate those names into the sorted label indices the batch carries (the
        # same sort order the model uses), once at setup rather than per batch.
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

        # SpectrumPL calls this with the training dict merged with the model's output
        # dict, so every key MassShatterModel returns has to be named here or absorbed
        # by **kwargs. The diagnostics are named explicitly because they are re-emitted
        # as metrics below.
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
            mean_mask_cut_count: th.Tensor,
            proposal_single_cut_fraction: th.Tensor,
            unique_component_count: th.Tensor,
            components_per_mask: th.Tensor,
            unique_components_per_mask: th.Tensor,
            productive_mask_fraction: th.Tensor,
            duplicate_component_pathway_fraction: th.Tensor,
            mask_cut_fraction_0: th.Tensor,
            mask_cut_fraction_1: th.Tensor,
            mask_cut_fraction_2: th.Tensor,
            mask_cut_fraction_3: th.Tensor,
            mask_cut_fraction_4plus: th.Tensor,
            candidate_input_transfer_batch_ms: th.Tensor,
            candidate_mask_selection_batch_ms: th.Tensor,
            candidate_topology_batch_ms: th.Tensor,
            candidate_assembly_batch_ms: th.Tensor,
            candidate_output_transfer_batch_ms: th.Tensor,
            candidate_total_batch_ms: th.Tensor,
            candidate_total_molecule_ms: th.Tensor,
            **kwargs,
        ) -> dict[str, th.Tensor]:
            """Return per-spectrum loss terms and metrics, each of shape ``(batch_size,)``.

            Every value stays a per-spectrum vector; SpectrumPL owns the weighted batch
            reduction, so reducing here would double-count spectra from merged groups.
            """
            tolerance_per_sample = kwargs.get("tol_per_sample")
            minimum_mz_per_sample = kwargs.get("min_mz_per_sample")
            # Peaks are matched by instrument tolerance. ``ios_ce`` scores the matched
            # true peaks against the predicted pathways; ``oos_ce`` charges the true
            # intensity that no predicted m/z covers against the model's own declared
            # out-of-support mass, and ``true_oos_logprobs`` is that unmatched true mass.
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
            # Everything past "primary_loss" is monitoring only, so it is detached:
            # candidate_wrecall is how much true intensity the enumerated masks could
            # explain at all (the ceiling on this batch), retained_mask_probability is
            # how much of the Bernoulli mask distribution survived truncation, and the
            # counts/timings track the cost of candidate generation.
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
                "mean_mask_cut_count": mean_mask_cut_count.detach().to(pred_logprobs),
                "proposal_single_cut_fraction": (
                    proposal_single_cut_fraction.detach().to(pred_logprobs)
                ),
                "unique_component_count": unique_component_count.detach().to(pred_logprobs),
                "components_per_mask": components_per_mask.detach().to(pred_logprobs),
                "unique_components_per_mask": (
                    unique_components_per_mask.detach().to(pred_logprobs)
                ),
                "productive_mask_fraction": (productive_mask_fraction.detach().to(pred_logprobs)),
                "duplicate_component_pathway_fraction": (
                    duplicate_component_pathway_fraction.detach().to(pred_logprobs)
                ),
                "mask_cut_fraction_0": mask_cut_fraction_0.detach().to(pred_logprobs),
                "mask_cut_fraction_1": mask_cut_fraction_1.detach().to(pred_logprobs),
                "mask_cut_fraction_2": mask_cut_fraction_2.detach().to(pred_logprobs),
                "mask_cut_fraction_3": mask_cut_fraction_3.detach().to(pred_logprobs),
                "mask_cut_fraction_4plus": (mask_cut_fraction_4plus.detach().to(pred_logprobs)),
                "candidate_input_transfer_batch_ms": (candidate_input_transfer_batch_ms.detach()),
                "candidate_mask_selection_batch_ms": (candidate_mask_selection_batch_ms.detach()),
                "candidate_topology_batch_ms": candidate_topology_batch_ms.detach(),
                "candidate_assembly_batch_ms": candidate_assembly_batch_ms.detach(),
                "candidate_output_transfer_batch_ms": (candidate_output_transfer_batch_ms.detach()),
                "candidate_total_batch_ms": candidate_total_batch_ms.detach(),
                "candidate_total_molecule_ms": candidate_total_molecule_ms.detach(),
            }

            if self.hparams.ce_pair_loss_weight != 0.0:
                # The term is train-only (SpectrumPL sets ``ce_pair_enabled``), but the
                # keys must exist on every split so the logged metrics stay aligned;
                # hence the zero vectors when it is skipped.
                batch_size = int(kwargs["batch_size"])
                pair_loss = pred_logprobs.new_zeros(batch_size)
                eligible_endpoints = pred_logprobs.new_zeros(batch_size)
                if kwargs["ce_pair_enabled"]:
                    # Only spectra from a configured instrument *and* adduct take part;
                    # elsewhere the observed winner switch is too noisy to supervise.
                    instrument_indices = kwargs["spec_inst_type"].reshape(-1)
                    precursor_indices = kwargs["spec_prec_type"].reshape(-1)
                    eligible_samples = th.isin(
                        instrument_indices,
                        eligible_instrument_indices.to(instrument_indices.device),
                    ) & th.isin(
                        precursor_indices,
                        eligible_precursor_indices.to(precursor_indices.device),
                    )
                    # Supervises which of two adjacent-CE spectra ranks the observed
                    # winner higher. It reads the ``ce_pair_*`` peak lists, which
                    # SpectrumPL passes through without the filtering and intensity
                    # transform applied to the tensors the primary CE consumes.
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
        # Every name here must be a key of loss_values above: SpectrumPL reduces and logs
        # exactly these, so a missing key fails at the first logging step.
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
            "mean_mask_cut_count",
            "proposal_single_cut_fraction",
            "unique_component_count",
            "components_per_mask",
            "unique_components_per_mask",
            "productive_mask_fraction",
            "duplicate_component_pathway_fraction",
            "mask_cut_fraction_0",
            "mask_cut_fraction_1",
            "mask_cut_fraction_2",
            "mask_cut_fraction_3",
            "mask_cut_fraction_4plus",
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

    def _setup_metric_fns(self) -> None:
        """Add validation candidate coverage broken down by mask cut count."""
        super()._setup_metric_fns()
        coverage_metric_names = {
            f"candidate_{metric}_cut_{bucket}"
            for bucket in _CUT_COUNT_BUCKETS
            for metric in ("recall", "wrecall")
        }
        for metric_name in coverage_metric_names:
            self.metric_min_max[metric_name] = None
        self.auxiliary_metric_names.update(coverage_metric_names)
        self.metric_names.update(coverage_metric_names)
        self.extra_metric_args.extend(
            [
                "candidate_mzs_by_cut",
                "candidate_batch_idxs_by_cut",
                "candidate_cut_counts",
            ]
        )
        base_metric_fn = self.metric_fn

        def metric_fn(**kwargs: Any) -> dict[str, th.Tensor]:
            metrics = base_metric_fn(**kwargs)
            metrics.update(
                _candidate_coverage_by_cut_count(
                    true_mzs=kwargs["true_mzs"],
                    true_ints=kwargs["true_ints"],
                    true_batch_idxs=kwargs["true_batch_idxs"],
                    candidate_mzs=kwargs["candidate_mzs_by_cut"],
                    candidate_batch_idxs=kwargs["candidate_batch_idxs_by_cut"],
                    candidate_cut_counts=kwargs["candidate_cut_counts"],
                    tolerance=self.tolerance,
                    relative=self.relative,
                    tolerance_min_mz=self.tolerance_min_mz,
                )
            )
            return metrics

        self.metric_fn = metric_fn

    def forward(self, **batch_kwargs) -> dict[str, th.Tensor]:
        """Select the MASShatter inputs from SpectrumPL's collated batch.

        The collated batch carries fields for every model in the repo, so only the ones
        MASShatter uses are forwarded. Optional metadata is read with ``get`` because
        the dataset omits those keys entirely when the corresponding spec_param is off,
        and the model then also has no embedding expecting them.

        Args:
            **batch_kwargs: The collated batch produced by SpectrumPL's dataloader.

        Returns:
            The model's sparse prediction dictionary, forwarded unchanged.
        """
        return self.model(
            mol_pyg=batch_kwargs["mol_pyg"],
            spec_prec_type=batch_kwargs["spec_prec_type"],
            spec_inst_type=batch_kwargs.get("spec_inst_type"),
            spec_frag_mode=batch_kwargs.get("spec_frag_mode"),
            spec_nce=batch_kwargs.get("spec_nce"),
            spec_nce_batch_idxs=batch_kwargs.get("spec_nce_batch_idxs"),
            spec_ace=batch_kwargs.get("spec_ace"),
            spec_ace_batch_idxs=batch_kwargs.get("spec_ace_batch_idxs"),
            static_graph=batch_kwargs.get("mass_shatter_static_graph"),
        )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: th.device,
        dataloader_idx: int,
    ) -> Any:
        """Keep static candidate-generation graph indices on CPU.

        Mask selection and connected-component search run in Python on the host, so the
        graph topology is needed on CPU every step. Snapshotting it here — before the
        base implementation moves the batch to the accelerator — means only the bond
        logits have to come back per batch, instead of the whole index structure.

        Args:
            batch: The collated batch, still on CPU.
            device: Target accelerator device.
            dataloader_idx: Index of the dataloader that produced the batch.

        Returns:
            The transferred batch, with the CPU-resident graph indices attached under
            ``mass_shatter_static_graph``.
        """
        mol_pyg = batch["mol_pyg"]
        static_graph = MassShatterStaticGraph(
            chemical_bond_index=mol_pyg.chemical_bond_index,
            atom_batch_indices=mol_pyg.batch,
            atom_ptr=mol_pyg.ptr,
        )
        transferred = super().transfer_batch_to_device(batch, device, dataloader_idx)
        transferred["mass_shatter_static_graph"] = static_graph
        return transferred
