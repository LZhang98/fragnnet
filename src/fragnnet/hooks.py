"""Per-model hooks: the model-specific steps that core code runs through the registry.

Each model type names a hooks class in ``registry.MODEL_REGISTRY``; core calls
``registry.get_model_hooks(model_type)`` and the method it needs. This base class holds the
defaults, which the core models (NEIMS, precursor, GNN, CLIP) and MassFormer use
unchanged (docs/plans/repo_reorg_plan.md, "Plugin interface").

A hooks module imports nothing model-specific at module level: config and checkpoint migration
must work without loading a model or its compiled extensions. Methods import what they need.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ModelHooks:
    """Default hooks: every method is a no-op."""

    def migrate_template_config(self, config_d: dict[str, Any]) -> None:
        """Backfill keys that templates saved before they existed lack.

        ``runner.load_config`` calls this on the template dict after resolving the model type and
        normalizing legacy key spellings, and before validating the custom config against it.

        Args:
            config_d: Template config, modified in place.
        """

    def migrate_checkpoint_hparams(self, hparams: dict[str, Any]) -> None:
        """Backfill keys that checkpoints saved before they existed lack.

        Called only from ``pl_utils.get_pl_hparams``, after the shared migrations. Fresh configs
        must declare these keys through the template, so constructors never call it.

        Args:
            hparams: Copy of the checkpoint hyperparameters, modified in place.
        """

    def uses_ce_context_sampler(self, config_d: dict[str, Any]) -> bool:
        """Whether training batches come from a CE-context sampler (one window per batch).

        Core's CE-window and CE-pair checks, and the train-sampler choice, follow this.

        Args:
            config_d: Resolved config.
        """
        return config_d["batch_sampler_type"] == "ce_context"

    def validate_config(self, config_d: dict[str, Any]) -> None:
        """Check this model's own config keys.

        ``runner.validate_sampler_config`` calls it after the shared checks that come first and
        after every other model's ``reject_foreign_config``.

        Args:
            config_d: Resolved config.

        Raises:
            ValueError: If a setting is invalid.
        """

    def reject_foreign_config(self, config_d: dict[str, Any]) -> None:
        """Reject a config for another model type that switches on this model's settings.

        Every model's config carries every key, so another model's keys are inert unless set.

        Args:
            config_d: Resolved config of another model type.

        Raises:
            ValueError: If the config enables one of this model's settings.
        """

    def apply_inference_config(
        self, config_d: dict[str, Any], *, disable_preproc: bool, output_formula_str: bool
    ) -> None:
        """Adjust the model's own config blocks for inference.

        ``script_utils.apply_inference_config`` calls it after its shared edits.

        Args:
            config_d: Config, modified in place.
            disable_preproc: Whether the caller turned dataset preprocessing off.
            output_formula_str: Whether formula strings are exported.
        """

    def filter_dataset(self, dataset: Any, config: dict[str, Any]) -> None:
        """Drop spectra the model cannot use.

        ``SpecMolDataset.__init__`` calls it after building the frames and before preprocessing,
        so every way of building a dataset applies it.

        Args:
            dataset: The dataset, modified in place.
            config: The constructor's keyword arguments (the resolved config).
        """

    def build_batch_sampler(
        self, ds: Any, config_d: dict[str, Any], sampler: Any, generator: Any
    ) -> Any | None:
        """Return the model's own batch sampler for ``ds``, or None for standard batching.

        ``runner.init_dataloader`` calls it after choosing ``sampler``. A returned batch sampler
        replaces the DataLoader's ``batch_size``, ``sampler`` and ``drop_last``.

        Args:
            ds: The dataset; ``ds.split`` is "train", "val", "test" or "predict_only".
            config_d: Resolved config.
            sampler: The index sampler core chose: None on the train split when
                ``uses_ce_context_sampler``, ``SequentialSampler`` on the other splits.
            generator: The generator the samplers share; SpectrumPL reseeds it each epoch.
        """
        return None

    def dataloader_options(self, config_d: dict[str, Any]) -> dict[str, Any]:
        """Extra DataLoader keyword arguments.

        Args:
            config_d: Resolved config.
        """
        return {}

    def trainer_options(self, config_d: dict[str, Any]) -> dict[str, Any]:
        """Trainer keyword arguments that depend on how the model loads data.

        Args:
            config_d: Resolved config.
        """
        if config_d["automatic_optimization"] and config_d["num_workers"] > 0:
            # little hack to prevent memory explosion
            # https://discuss.pytorch.org/t/how-to-share-data-among-dataloader-processes-to-save-memory/108772
            # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
            return {"reload_dataloaders_every_n_epochs": 1}
        return {}

    def advances_sampler_each_epoch(self, hparams: Any) -> bool:
        """Whether the training batches change from epoch to epoch.

        If so, ``SpectrumPL._seed_dataloader`` calls the sampler's ``set_epoch`` and refreshes
        Lightning's cached batch counts; otherwise it only recomputes batches.

        Args:
            hparams: The module's hyperparameters (attribute access).
        """
        return hparams.bond_size_curriculum or hparams.batch_sampler_type == "ce_context"

    def log_resolved_settings(
        self, model: Any, config_d: dict[str, Any], wandb_run: Any | None
    ) -> None:
        """Log settings the model resolved at construction.

        ``runner.run_training`` calls this after building the model and merging any saved W&B
        config, before creating callbacks.

        Args:
            model: The constructed Lightning module.
            config_d: Resolved config.
            wandb_run: The active W&B run, or None when W&B is disabled.
        """


class ExportHooks:
    """Methods the inference exporters (``fragnnet.utils.script_utils``) call on the model.

    The defaults describe a model with peak outputs only. FraGNNet and MASSter, whose exports
    carry formula/node annotation, mix this class into their Lightning modules and override what
    they need. The exporters call these on the module they hold and fall back to a default
    instance for any other object. Test stubs skip ``__init__``, so nothing here reads
    ``hparams``.
    """

    # Whether the outputs carry the formula, node and joint annotation axes.
    exports_annotation_axes = False
    # Whether the export can name the SMARTS rules behind each node.
    supports_rule_provenance = False

    def export_index_vars(
        self, nb_iso: bool
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Index variables (batch, node, formula, nb-node) the exporter rebatches.

        Args:
            nb_iso: Whether the isomorphic-neighbourhood outputs are exported.
        """
        assert not nb_iso
        batch = ["pred_batch_idxs", "true_batch_idxs"]
        if not self.exports_annotation_axes:
            return batch, [], [], []
        return (
            batch + ["pred_formula_batch_idxs", "pred_node_batch_idxs", "pred_joint_batch_idxs"],
            ["pred_node_node_idxs", "pred_joint_node_idxs"],
            ["pred_formula_formula_idxs", "pred_joint_formula_idxs"],
            [],
        )

    def nb_index_vars(self) -> tuple[set[str], set[str], set[str]]:
        """Batch, formula and nb-node index variables of the nb_iso outputs; none by default."""
        return set(), set(), set()

    def inference_batch_options(self, exporter: str, need_annotation: bool) -> dict[str, Any]:
        """Batch keys to set before ``inference_step``.

        Args:
            exporter: "pkl" or "tabular" (the tabular and HDF5 exporters).
            need_annotation: Whether the export reads the formula or node annotation.
        """
        return {}

    def postprocess_export_peaks(
        self,
        batch_input: dict[str, Any],
        batch_result: dict[str, Any],
        peaks: tuple[Any, Any, Any],
        *,
        need_annotation: bool,
        return_cc_hash: bool,
        return_rule_provenance: bool,
    ) -> tuple[tuple[Any, Any, Any], Any]:
        """Adjust the predicted peaks before export and build any annotation state.

        Args:
            batch_input: The batch, modified in place where the model adds keys.
            batch_result: The ``inference_step`` output, modified in place likewise.
            peaks: Predicted (m/z, intensity, batch index) tensors.
            need_annotation: Whether the export reads the formula or node annotation.
            return_cc_hash: Whether node cc hashes are exported.
            return_rule_provenance: Whether node rule names are exported.

        Returns:
            The peaks to export and an opaque annotation state for the methods below (None).
        """
        return peaks, None

    def export_formula_row_deltas(
        self, batch_input: dict[str, Any], formula_cumsizes: Any, annotation: Any
    ) -> Any:
        """Precursor delta per formula row, for ``pred_formula_adducts``; None leaves NaN."""
        return None

    def export_node_ccs(self, frag_pyg: Any, b_idx: int, node_idxs: Any, annotation: Any) -> Any:
        """One spectrum's node cc bitmasks for ``pred_node_ccs``; NaN when unavailable."""
        return np.nan

    def export_node_rules(self, b_idx: int, node_idxs: Any, annotation: Any) -> Any:
        """One spectrum's node SMARTS rule names for ``pred_node_rules``; NaN when unavailable."""
        return np.nan
