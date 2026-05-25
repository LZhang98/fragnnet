from typing import Any

import lightning as L
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.model import MolEncoder, SpectrumEncoder
from fragnnet.model.loss import clip_loss
from fragnnet.utils.misc_utils import scatter_reduce
from fragnnet.utils.nn_utils import build_lr_scheduler, get_optimizer_class


class SpectrumMolClipPL(L.LightningModule):
    """CLIP-style spectrum-molecule contrastive pretraining module."""

    hparams: Any

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.clip_temperature = float(self.hparams.get("clip_temperature", 0.07))
        self.clip_embed_dim = int(self.hparams.get("clip_embed_dim", 256))
        self.clip_intensity_transform = self.hparams.get("clip_intensity_transform", "sqrt")
        self._setup_model()
        self._val_mol_embeds: list[th.Tensor] = []
        self._val_spec_embeds: list[th.Tensor] = []

    def _setup_model(self) -> None:
        spec_encoder_params = self.hparams.get("spectrum_encoder_params", {})
        spec_input_dim = int(spec_encoder_params.get("input_dim", 2))
        spec_hidden_dim = int(spec_encoder_params.get("hidden_dim", self.hparams.mlp_hidden_size))
        spec_num_heads = int(spec_encoder_params.get("num_heads", 4))
        if spec_hidden_dim % spec_num_heads != 0:
            raise ValueError(
                "spectrum_encoder_params.hidden_dim must be divisible by "
                f"num_heads ({spec_hidden_dim} vs {spec_num_heads})"
            )

        # --- Spectrum-side conditioning embedders ---
        # These mirror the mol encoder's metadata conditioning so both modalities
        # see the same experimental context (CE, instrument, frag mode).
        # Sizes default to 0 (disabled) for backward compatibility.
        spec_ce_embed_size = int(spec_encoder_params.get("spec_ce_embed_size", 0))
        spec_inst_embed_size = int(spec_encoder_params.get("spec_inst_embed_size", 0))
        spec_frag_mode_embed_size = int(spec_encoder_params.get("spec_frag_mode_embed_size", 0))
        self._spec_ce_embed_size = spec_ce_embed_size
        self._spec_inst_embed_size = spec_inst_embed_size
        self._spec_frag_mode_embed_size = spec_frag_mode_embed_size

        if spec_ce_embed_size > 0:
            self.spec_ce_linear = nn.Linear(1, spec_ce_embed_size)
        if spec_inst_embed_size > 0:
            num_inst = len(self.hparams.spec_params["inst_types"])
            self.spec_inst_embed = nn.Embedding(num_inst + 1, spec_inst_embed_size)
        if spec_frag_mode_embed_size > 0:
            num_frag = len(self.hparams.spec_params["frag_modes"])
            self.spec_frag_mode_embed = nn.Embedding(num_frag + 1, spec_frag_mode_embed_size)

        spec_cond_dim = spec_ce_embed_size + spec_inst_embed_size + spec_frag_mode_embed_size
        augmented_input_dim = spec_input_dim + spec_cond_dim

        # Keep the molecule encoder under `model.*` so CLIP checkpoints can be
        # loaded by FraGNNetPL's existing finetune path. FraGNNetPL strips the
        # top-level `model.` prefix and then matches shared component names such
        # as `mol_embedder` and `mol_pool`.
        self.model = MolEncoder(
            mol_node_feats=self.hparams.mol_params["pyg_node_feats"],
            mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
            mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
            mol_hidden_size=self.hparams.mol_hidden_size,
            mol_num_layers=self.hparams.mol_num_layers,
            mol_gnn_type=self.hparams.mol_gnn_type,
            mol_normalization=self.hparams.mol_normalization,
            mol_dropout=self.hparams.mol_dropout,
            mol_pool_type=self.hparams.mol_pool_type,
            output_dim=self.clip_embed_dim,
            mlp_hidden_size=self.hparams.mlp_hidden_size,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            int_embedder=self.hparams.int_embedder,
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            nce_mean=self.hparams.nce_mean,
            nce_std=self.hparams.nce_std,
            nce_max=self.hparams.nce_max,
            prec_insert_location=self.hparams.prec_insert_location,
            prec_insert_size=self.hparams.prec_insert_size,
            prec_types=self.hparams.spec_params["prec_types"],
            inst_insert_location=self.hparams.inst_insert_location,
            inst_insert_size=self.hparams.inst_insert_size,
            inst_types=self.hparams.spec_params["inst_types"],
            frag_mode_insert_location=self.hparams.frag_mode_insert_location,
            frag_mode_insert_size=self.hparams.frag_mode_insert_size,
            frag_modes=self.hparams.spec_params["frag_modes"],
            mol_num_heads=self.hparams.get("mol_num_heads", 4),
            use_nce=self.hparams.get("use_nce", self.hparams.spec_params.get("nce", False)),
            use_ace=self.hparams.get("use_ace", self.hparams.spec_params.get("ace", False)),
            ace_max=self.hparams.get("ace_max", 100.0),
            ace_mean=self.hparams.get("ace_mean", 20.0),
            ace_std=self.hparams.get("ace_std", 15.0),
        )

        self.spec_encoder = SpectrumEncoder(
            spectrum_encoder_type=spec_encoder_params.get("encoder_type", "set_transformer"),
            input_dim=augmented_input_dim,
            hidden_dim=spec_hidden_dim,
            output_dim=self.clip_embed_dim,
            num_heads=spec_num_heads,
            num_inds=int(spec_encoder_params.get("num_inds", 32)),
            num_blocks=int(spec_encoder_params.get("num_layers", 2)),
            dropout=float(spec_encoder_params.get("dropout", self.hparams.mlp_dropout)),
            normalization=spec_encoder_params.get("normalization", "layer"),
        )

        if self.hparams.compile:
            self.model = th.compile(self.model, dynamic=True)
            self.spec_encoder = th.compile(self.spec_encoder, dynamic=True)

    def _build_peak_features(self, batch: dict[str, Any]) -> th.Tensor:
        mzs = batch["spec_mzs"].float()
        ints = batch["spec_ints"].float().clamp_min(0.0)
        batch_idxs = batch["spec_batch_idxs"].long()
        batch_size = int(batch["batch_size"].item())
        mz_max = float(self.hparams.mz_max)

        if self.clip_intensity_transform == "sqrt":
            ints = th.sqrt(ints)
        elif self.clip_intensity_transform == "log1p":
            ints = th.log1p(ints)
        elif self.clip_intensity_transform == "none":
            pass
        else:
            raise ValueError(f"Unknown clip_intensity_transform={self.clip_intensity_transform!r}")

        max_ints = scatter_reduce(
            ints,
            batch_idxs,
            reduce="amax",
            dim_size=batch_size,
            default=0.0,
            include_self=True,
        )
        ints = ints / max_ints[batch_idxs].clamp_min(1e-12)
        mzs_norm = mzs / mz_max

        # Precursor-relative features: broadcast spectrum-level prec_mz to each peak.
        # neutral_loss_norm encodes how far below the precursor each peak sits (signed,
        # so peaks above precursor get a negative value — rare but informative).
        # prec_norm gives the absolute scale of the molecule, shared across all peaks.
        prec_mz_per_spec = batch["spec_prec_mz"].float().reshape(-1)  # (batch_size,)
        prec_mz_per_peak = prec_mz_per_spec[batch_idxs]  # (total_peaks,)
        prec_norm = prec_mz_per_peak / mz_max
        neutral_loss_norm = (prec_mz_per_peak - mzs) / mz_max

        return th.stack([mzs_norm, ints, neutral_loss_norm, prec_norm], dim=-1)

    def _build_conditioned_peak_features(self, batch: dict[str, Any]) -> th.Tensor:
        """Build per-peak feature tensor with optional CE/inst/frag-mode conditioning.

        Each conditioning vector is spectrum-level (one per spectrum) and is broadcast
        to every peak in that spectrum, then concatenated with (mz, intensity).

        Args:
            batch: Batch dictionary from the dataloader.

        Returns:
            Peak features of shape (total_peaks, 2 + cond_dim).
        """
        peak_x = self._build_peak_features(batch)  # (total_peaks, 2)
        batch_idxs = batch["spec_batch_idxs"].long()
        batch_size = int(batch["batch_size"].item())

        parts = [peak_x]

        if self._spec_ce_embed_size > 0:
            use_nce = self.hparams.get("use_nce", self.hparams.spec_params.get("nce", False))
            ce = batch.get("spec_nce") if use_nce else None
            ce = ce if ce is not None else batch.get("spec_ce")
            ce_batch_idxs = batch.get("spec_nce_batch_idxs") if use_nce else None
            ce_batch_idxs = (
                ce_batch_idxs if ce_batch_idxs is not None else batch.get("spec_ce_batch_idxs")
            )
            nce_mean = float(self.hparams.nce_mean)
            nce_std = float(self.hparams.nce_std)
            ce_f = ce.float()
            missing = ce_f < 0
            ce_norm = ((ce_f - nce_mean) / nce_std).masked_fill(missing, 0.0).unsqueeze(-1)
            ce_embed = self.spec_ce_linear(ce_norm)  # (n_ce_vals, spec_ce_embed_size)
            ce_embed = ce_embed.masked_fill(missing.unsqueeze(-1), 0.0)
            # Reduce multiple CE values per spectrum to one via mean
            idx_2d = ce_batch_idxs.unsqueeze(1).expand_as(ce_embed)
            ce_per_spec = scatter_reduce(
                ce_embed, idx_2d, reduce="mean", dim_size=batch_size, include_self=False
            )
            parts.append(ce_per_spec[batch_idxs])

        if self._spec_inst_embed_size > 0:
            inst_type = batch["spec_inst_type"]
            parts.append(self.spec_inst_embed(inst_type)[batch_idxs])

        if self._spec_frag_mode_embed_size > 0:
            frag_mode = batch["spec_frag_mode"]
            parts.append(self.spec_frag_mode_embed(frag_mode)[batch_idxs])

        return th.cat(parts, dim=-1) if len(parts) > 1 else parts[0]

    def forward(self, batch: dict[str, Any]) -> tuple[th.Tensor, th.Tensor]:
        ce = batch.get("spec_ce")
        ce_batch_idxs = batch.get("spec_ce_batch_idxs")
        if self.hparams.get("use_nce", self.hparams.spec_params.get("nce", False)):
            ce = batch.get("spec_nce", ce)
            ce_batch_idxs = batch.get("spec_nce_batch_idxs", ce_batch_idxs)

        mol_embed = self.model(
            batch["mol_pyg"],
            ce=ce,
            ce_batch_idxs=ce_batch_idxs,
            ace=batch.get("spec_ace"),
            ace_batch_idxs=batch.get("spec_ace_batch_idxs"),
            prec_type=batch.get("spec_prec_type"),
            inst_type=batch.get("spec_inst_type"),
            frag_mode=batch.get("spec_frag_mode"),
        )

        peak_x = self._build_conditioned_peak_features(batch)
        spec_embed = self.spec_encoder(peak_x, batch_idx=batch["spec_batch_idxs"])

        return F.normalize(mol_embed, dim=-1), F.normalize(spec_embed, dim=-1)

    @staticmethod
    def _retrieval_metrics(
        query_embed: th.Tensor, target_embed: th.Tensor, prefix: str, top_k: int = 5
    ) -> dict[str, th.Tensor]:
        sim = query_embed @ target_embed.T
        batch_size = sim.shape[0]
        labels = th.arange(batch_size, device=sim.device)
        order = th.argsort(sim, dim=1, descending=True)
        ranks = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        k = min(top_k, batch_size)
        return {
            f"{prefix}_top1": (ranks == 1).float().mean(),
            f"{prefix}_top{k}": (ranks <= k).float().mean(),
            f"{prefix}_mrr": (1.0 / ranks.float()).mean(),
        }

    def _common_step(self, batch: dict[str, Any], split: str) -> th.Tensor:
        mol_embed, spec_embed = self.forward(batch)
        loss = clip_loss(spec_embed, mol_embed, temperature=self.clip_temperature)
        batch_size = int(batch["batch_size"].item())
        self.log(
            f"{split}_loss",
            loss,
            on_step=(split == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        with th.no_grad():
            metrics = {}
            metrics.update(self._retrieval_metrics(spec_embed, mol_embed, "s2m"))
            metrics.update(self._retrieval_metrics(mol_embed, spec_embed, "m2s"))
            for name, value in metrics.items():
                self.log(
                    f"{split}_{name}",
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=(split != "train" and name in {"s2m_top1", "s2m_mrr"}),
                    batch_size=batch_size,
                )
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> th.Tensor:
        return self._common_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> th.Tensor:
        with th.no_grad():
            mol_embed, spec_embed = self.forward(batch)
        loss = clip_loss(spec_embed, mol_embed, temperature=self.clip_temperature)
        batch_size = int(batch["batch_size"].item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._val_mol_embeds.append(mol_embed.detach())
        self._val_spec_embeds.append(spec_embed.detach())
        return loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_mol_embeds:
            return
        mol_embed = th.cat(self._val_mol_embeds, dim=0)
        spec_embed = th.cat(self._val_spec_embeds, dim=0)
        self._val_mol_embeds.clear()
        self._val_spec_embeds.clear()
        metrics = {}
        metrics.update(self._retrieval_metrics(spec_embed, mol_embed, "s2m"))
        metrics.update(self._retrieval_metrics(mol_embed, spec_embed, "m2s"))
        for name, value in metrics.items():
            self.log(
                f"val_{name}",
                value,
                prog_bar=(name in {"s2m_top1", "s2m_mrr"}),
            )

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> th.Tensor:
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        optimizer_cls = get_optimizer_class(self.hparams.optimizer)
        optimizer = optimizer_cls(
            self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
        ret = {"optimizer": optimizer}
        if self.hparams.lr_schedule:
            scheduler = build_lr_scheduler(
                optimizer=optimizer,
                decay_rate=self.hparams.lr_decay_rate,
                warmup_steps=self.hparams.lr_warmup_steps,
                decay_steps=self.hparams.lr_decay_steps,
                schedule_type=self.hparams.get("lr_schedule_type", "exponential"),
                total_steps=self.hparams.get("lr_total_steps"),
                min_lr_ratio=self.hparams.get("lr_min_ratio", 0.0),
            )
            ret["lr_scheduler"] = {
                "scheduler": scheduler,
                "frequency": 1,
                "interval": "step",
            }
        return ret
