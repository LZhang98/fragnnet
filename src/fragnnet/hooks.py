"""Per-model hooks: the model-specific steps that core code runs through the registry.

Each model type names a hooks class in ``registry.MODEL_REGISTRY``; core calls
``registry.get_model_hooks(model_type)`` and the method it needs. This base class holds the
defaults, which the core models (NEIMS, ChemLM, precursor, GNN, MCES, CLIP) and the baselines use
unchanged (docs/plans/repo_reorg_plan.md, "Plugin interface").

A hooks module imports nothing model-specific at module level: config and checkpoint migration
must work without loading a model or its compiled extensions. Methods import what they need.
"""

from __future__ import annotations

from typing import Any


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
