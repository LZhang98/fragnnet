"""FraGNNet's hooks, registered for ``model_type: frag_gnn`` and ``fragnnet`` (see
``fragnnet.hooks``).

Config validation must work without building the model or loading its compiled extensions, so
this module imports core only at module level.
"""

from __future__ import annotations

from typing import Any

from fragnnet.dataset.data_sampler import validate_dag_size_curriculum
from fragnnet.hooks import ModelHooks


class FraGNNetHooks(ModelHooks):
    """FraGNNet's fragment-DAG batch samplers and curriculum: validation."""

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
