"""FraGNNet's hooks, registered for ``model_type: frag_gnn`` and ``fragnnet`` (see
``fragnnet.hooks``).

Config validation must work without building the model or loading its compiled extensions, so
this module imports core only at module level.
"""

from __future__ import annotations

from typing import Any

from fragnnet.hooks import ModelHooks


class FraGNNetHooks(ModelHooks):
    """FraGNNet's fragment-DAG batch samplers and curriculum."""

    def validate_config(self, config_d: dict[str, Any]) -> None:
        num_enabled_batch_samplers = sum(
            int(config_d[key])
            for key in (
                "dynamic_batch_sampler",
                "dual_group_dynamic_batch_sampler",
            )
        )
        if num_enabled_batch_samplers > 1:
            raise ValueError(
                "masster_batch_sampler, dynamic_batch_sampler, and "
                "dual_group_dynamic_batch_sampler are mutually exclusive"
            )

        if config_d["dynamic_batch_sampler"]:
            if config_d["automatic_optimization"]:
                raise ValueError(
                    f"Dynamic batch sampler cannot be used with automatic optimization "
                    f"(got automatic_optimization={config_d['automatic_optimization']})"
                )
            if config_d["dynamic_batch_sampler_mode"] not in ["frag_node", "frag_edge"]:
                raise ValueError(
                    "Dynamic batch sampler only supports frag_node or frag_edge mode "
                    f"(got '{config_d['dynamic_batch_sampler_mode']}')"
                )
            if config_d["dynamic_batch_sampler_max"] is None:
                raise ValueError(
                    "Dynamic batch sampler requires dynamic_batch_sampler_max to be set"
                )

        if config_d["batch_sampler_type"] == "ce_context":
            if not config_d["dynamic_batch_sampler"]:
                raise ValueError(
                    "batch_sampler_type='ce_context' requires dynamic_batch_sampler=True"
                )
            if config_d["dynamic_batch_sampler_mode"] != "frag_edge":
                raise ValueError(
                    "batch_sampler_type='ce_context' requires "
                    "dynamic_batch_sampler_mode='frag_edge' for fragment-DAG models"
                )
            if config_d["dag_size_curriculum"]:
                raise ValueError(
                    "batch_sampler_type='ce_context' does not support DAG-size curriculum"
                )

        if config_d["dag_size_curriculum"]:
            if not config_d["dynamic_batch_sampler"]:
                raise ValueError("DAG-size curriculum requires dynamic_batch_sampler=True")
            from fragnnet.models.fragnnet.fragnnet_data_sampler import validate_dag_size_curriculum

            validate_dag_size_curriculum(
                config_d["dag_size_curriculum_start_fraction"],
                config_d["dag_size_curriculum_epochs"],
            )
            if (
                config_d["max_epochs"] > 0
                and config_d["dag_size_curriculum_epochs"] >= config_d["max_epochs"]
            ):
                raise ValueError("dag_size_curriculum_epochs must be less than max_epochs")

        if config_d["dual_group_dynamic_batch_sampler"]:
            if config_d["automatic_optimization"]:
                raise ValueError(
                    f"Dual-group dynamic batch sampler cannot be used with automatic optimization "
                    f"(got automatic_optimization={config_d['automatic_optimization']})"
                )
            dual_group_sampler_mode = config_d.get("dual_group_dynamic_batch_sampler_mode")
            if dual_group_sampler_mode not in ["frag_node", "frag_edge"]:
                raise ValueError(
                    f"Dual-group dynamic batch sampler only supports frag_node or frag_edge mode "
                    f"(got '{dual_group_sampler_mode}')"
                )
            dual_group_sampler_max = config_d.get("dual_group_dynamic_batch_sampler_max")
            if dual_group_sampler_max is None:
                raise ValueError(
                    "Dual-group dynamic batch sampler requires "
                    "dual_group_dynamic_batch_sampler_max to be set"
                )

        if (
            "frag_params" in config_d
            and config_d["frag_params"]["include_cmf"]
            and "cmf_h_formulae_idx" not in config_d["frag_params"]["pyg_node_feats"]
        ):
            raise ValueError(
                "include_cmf is True but 'cmf_h_formulae_idx' is missing from pyg_node_feats"
            )

    def reject_foreign_config(self, config_d: dict[str, Any]) -> None:
        if config_d["dynamic_batch_sampler"]:
            raise ValueError("Dynamic batch sampler can only be used with frag_gnn model")
        if config_d["batch_sampler_type"] == "ce_context":
            raise ValueError("batch_sampler_type='ce_context' requires dynamic_batch_sampler=True")
        if config_d["dag_size_curriculum"]:
            raise ValueError("DAG-size curriculum requires dynamic_batch_sampler=True")
        if config_d["dual_group_dynamic_batch_sampler"]:
            raise ValueError(
                "Dual-group dynamic batch sampler can only be used with frag_gnn model"
            )

    def apply_inference_config(
        self, config_d: dict[str, Any], *, disable_preproc: bool, output_formula_str: bool
    ) -> None:
        if "frag_params" in config_d:
            # The DAG dataset emits frag_formula_str for the formula export.
            config_d["frag_params"]["formula_str"] = True
            # Disable preprocessing to save memory, whether or not disable_preproc is set.
            config_d["frag_params"]["preload"] = False
            config_d["frag_params"]["preprocess"] = False

    def builds_own_batch_sampler(self, config_d: dict[str, Any]) -> bool:
        return bool(
            config_d["dynamic_batch_sampler"] or config_d["dual_group_dynamic_batch_sampler"]
        )

    def build_batch_sampler(
        self, ds: Any, config_d: dict[str, Any], sampler: Any, generator: Any
    ) -> Any | None:
        from fragnnet.models.fragnnet.fragnnet_data_sampler import (
            CEContextDynamicBatchSampler,
            DualGroupDynamicBatchSampler,
            SpecMolFragDynamicBatchSampler,
        )

        split = ds.split
        if config_d["dynamic_batch_sampler"]:
            if split == "train":
                return_batch_at = config_d["train_batch_size"]
                if not config_d["automatic_optimization"]:
                    return_batch_at *= config_d["accumulate_grad_batches"]
            else:
                return_batch_at = config_d["eval_batch_size"]
            if config_d["batch_sampler_type"] == "ce_context":
                return CEContextDynamicBatchSampler(
                    ds,
                    max_num=config_d["dynamic_batch_sampler_max"],
                    ce_key_by_inst=config_d["ce_context_ce_key_by_inst"],
                    limited_by=config_d["dynamic_batch_sampler_mode"],
                    spectra_per_context=config_d["ce_context_spectra_per_context"],
                    return_batch_at=return_batch_at,
                    seed=config_d["seed"],
                    skip_too_big=split == "train",
                )
            if split == "train":
                return SpecMolFragDynamicBatchSampler(
                    ds,
                    max_num=config_d["dynamic_batch_sampler_max"],
                    limited_by=config_d["dynamic_batch_sampler_mode"],
                    skip_too_big=True,
                    return_batch_at=return_batch_at,
                    sampler=sampler,
                    dag_size_curriculum=config_d["dag_size_curriculum"],
                    dag_size_curriculum_start_fraction=config_d[
                        "dag_size_curriculum_start_fraction"
                    ],
                    dag_size_curriculum_epochs=config_d["dag_size_curriculum_epochs"],
                )
            return SpecMolFragDynamicBatchSampler(
                ds,
                max_num=config_d["dynamic_batch_sampler_max"],
                limited_by=config_d["dynamic_batch_sampler_mode"],
                skip_too_big=False,
                return_batch_at=return_batch_at,
                sampler=sampler,
            )
        if config_d["dual_group_dynamic_batch_sampler"]:
            if split == "train":
                return_batch_at = config_d["train_batch_size"] * config_d["accumulate_grad_batches"]
                return DualGroupDynamicBatchSampler(
                    ds,
                    max_num=config_d["dual_group_dynamic_batch_sampler_max"],
                    limited_by=config_d["dual_group_dynamic_batch_sampler_mode"],
                    skip_too_big=True,
                    return_batch_at=return_batch_at,
                    sample_k1=config_d["dual_group_sampler_k1"],
                    sample_k2=config_d["dual_group_sampler_k2"],
                    sample_k1_per_key=config_d["dual_group_sampler_k1_per_key"],
                    sample_k2_per_key=config_d["dual_group_sampler_k2_per_key"],
                    generator=generator,
                )
            return_batch_at = config_d["eval_batch_size"]
            return SpecMolFragDynamicBatchSampler(
                ds,
                max_num=config_d["dual_group_dynamic_batch_sampler_max"],
                limited_by=config_d["dual_group_dynamic_batch_sampler_mode"],
                skip_too_big=False,
                return_batch_at=return_batch_at,
                sampler=sampler,
            )
        return None

    def trainer_options(self, config_d: dict[str, Any]) -> dict[str, Any]:
        if config_d["automatic_optimization"]:
            return super().trainer_options(config_d)
        if config_d["dynamic_batch_sampler"] or config_d["dual_group_dynamic_batch_sampler"]:
            # with out this progress will not track dataset length change
            # this will call train_dataloader and val_dataloader before every epoch
            # not sure why this fixed our problem but it works
            # Use per-epoch batch indices for validation checks. DAG curriculum training also
            # refreshes Lightning's cached train/validation batch counts after sampler updates.
            return {"reload_dataloaders_every_n_epochs": 1, "check_val_every_n_epoch": 1}
        return {}

    def advances_sampler_each_epoch(self, hparams: Any) -> bool:
        return (
            hparams.dag_size_curriculum
            or hparams.bond_size_curriculum
            or hparams.batch_sampler_type == "ce_context"
        )
