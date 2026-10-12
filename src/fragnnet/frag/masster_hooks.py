"""MASSter's hooks, registered for ``model_type: masster`` (see ``fragnnet.hooks``).

Config and checkpoint migration must work without building the model or loading its compiled
extensions, so this module imports core only.
"""

from __future__ import annotations

import logging
from typing import Any

from fragnnet.hooks import ModelHooks
from fragnnet.utils.pl_utils import migrate_masster_proposal_method, migrate_rt_ccs_spec_params

# MASSter's sampler keys. Every model's config carries them; they mean something only for MASSter.
MASSTER_SAMPLER_KEYS = (
    "masster_batch_sampler",
    "masster_share_proposal_across_ev",
    "masster_unique_fragment_mode",
)

# Checkpoints saved before the RT/CCS heads existed: heads off, with their original defaults.
_LEGACY_RT_CCS_DEFAULTS = {
    "masster_predict_rt_ccs": False,
    "masster_predict_rt": False,
    "masster_predict_ccs": False,
    "masster_rt_loss_weight": 0.1,
    "masster_ccs_loss_weight": 0.1,
    "masster_rt_scale": 60.0,
    "masster_ccs_scale": 100.0,
}


def _backfill_legacy_rt_ccs(hparams: dict) -> None:
    for key, value in _LEGACY_RT_CCS_DEFAULTS.items():
        hparams.setdefault(key, value)


def migrate_masster_rt_ccs_hparams(hparams: dict) -> None:
    """Explicitly migrate pre-RT/CCS checkpoints to the disabled legacy behavior."""
    migrate_rt_ccs_spec_params(hparams)
    if hparams.get("model_type") == "masster":
        _backfill_legacy_rt_ccs(hparams)


class MassterHooks(ModelHooks):
    """MASSter's template and checkpoint migrations and resolved-settings logging."""

    def migrate_template_config(self, config_d: dict[str, Any]) -> None:
        # Compatibility boundary for templates saved before candidate backends existed.
        # Add the schema key before custom config validation and CLI overrides.
        if "masster_candidate_backend" not in config_d:
            config_d["masster_candidate_backend"] = "cpu"
        if "masster_proposal_method" not in config_d:
            config_d["masster_proposal_method"] = "cut_masks"
        if "masster_fast_gnn_beam_width" not in config_d:
            config_d["masster_fast_gnn_beam_width"] = 256
        if "masster_fast_gnn_max_boundary_cuts" not in config_d:
            config_d["masster_fast_gnn_max_boundary_cuts"] = 3

    def migrate_checkpoint_hparams(self, hparams: dict[str, Any]) -> None:
        _backfill_legacy_rt_ccs(hparams)
        # Explicit checkpoint migration: pre-context-input MASSter weights use the
        # structural-only GNN width. Fresh configs must declare this key via the template.
        if "masster_context_in_gnn" not in hparams:
            hparams["masster_context_in_gnn"] = False
        # Explicit checkpoint migration: pre-attention-topology MASSter weights carry the
        # induced latent set. Fresh configs must declare this key via the template.
        if "masster_intensity_transformer_attention" not in hparams:
            hparams["masster_intensity_transformer_attention"] = "induced"
        # Backend choice changes construction only; old checkpoints retain CPU semantics.
        if "masster_candidate_backend" not in hparams:
            hparams["masster_candidate_backend"] = "cpu"
        migrate_masster_proposal_method(hparams)
        if "masster_proposal_method" not in hparams:
            hparams["masster_proposal_method"] = "cut_masks"
        if "masster_fast_gnn_beam_width" not in hparams:
            hparams["masster_fast_gnn_beam_width"] = 256
        if "masster_fast_gnn_max_boundary_cuts" not in hparams:
            hparams["masster_fast_gnn_max_boundary_cuts"] = 3
        # Checkpoints from before marginal fragment selection kept the whole enumerated pool.
        if "masster_fragment_selection" not in hparams:
            hparams["masster_fragment_selection"] = "mask_order"
        if "masster_selected_fragment_target" not in hparams:
            hparams["masster_selected_fragment_target"] = None

    def uses_ce_context_sampler(self, config_d: dict[str, Any]) -> bool:
        return (
            bool(config_d["masster_batch_sampler"])
            or config_d["batch_sampler_type"] == "ce_context"
        )

    def validate_config(self, config_d: dict[str, Any]) -> None:
        # Hard reads: a MASSter config must declare its sampler keys.
        use_masster_batch_sampler = config_d["masster_batch_sampler"]
        share_proposal_across_ev = config_d["masster_share_proposal_across_ev"]
        use_unique_fragments = config_d["masster_unique_fragment_mode"]
        if not isinstance(use_masster_batch_sampler, bool):
            raise ValueError("masster_batch_sampler must be boolean")
        if not isinstance(share_proposal_across_ev, bool):
            raise ValueError("masster_share_proposal_across_ev must be boolean")
        if use_masster_batch_sampler:
            if not config_d["automatic_optimization"]:
                raise ValueError("masster_batch_sampler requires automatic_optimization=True")
            masster_batch_sampler_max = config_d["masster_batch_sampler_max"]
            if (
                isinstance(masster_batch_sampler_max, bool)
                or not isinstance(masster_batch_sampler_max, int)
                or masster_batch_sampler_max <= 0
            ):
                raise ValueError("masster_batch_sampler_max must be a positive integer")
        if share_proposal_across_ev and not use_masster_batch_sampler:
            raise ValueError("masster_share_proposal_across_ev requires masster_batch_sampler=True")
        if config_d["cross_ev_fragment_loss_weight"] > 0:
            if not share_proposal_across_ev:
                raise ValueError(
                    "cross_ev_fragment_loss_weight > 0 requires "
                    "masster_share_proposal_across_ev=True"
                )
            if not use_unique_fragments:
                raise ValueError(
                    "cross_ev_fragment_loss_weight > 0 requires masster_unique_fragment_mode=True"
                )

    def reject_foreign_config(self, config_d: dict[str, Any]) -> None:
        # Soft reads: another model's (historical) config may omit MASSter's keys.
        enabled_elsewhere = [key for key in MASSTER_SAMPLER_KEYS if config_d.get(key)]
        if enabled_elsewhere:
            raise ValueError(
                f"{enabled_elsewhere} require model_type='masster' "
                f"(got '{config_d['model_type']}')"
            )
        if config_d["cross_ev_fragment_loss_weight"] > 0:
            raise ValueError(
                "cross_ev_fragment_loss_weight > 0 requires masster_share_proposal_across_ev=True"
            )

    def log_resolved_settings(
        self, model: Any, config_d: dict[str, Any], wandb_run: Any | None
    ) -> None:
        resolved_proposal = config_d["masster_proposal_method"]
        resolved_backend = (
            "torch_device"
            if resolved_proposal == "connected_fragment_beam"
            else model.model.candidate_backend
        )
        logging.info("Resolved MASSter candidate backend: %s", resolved_backend)
        if wandb_run is not None:
            wandb_run.summary["masster_candidate_backend_resolved"] = resolved_backend
            wandb_run.summary["masster_proposal_method_resolved"] = resolved_proposal
