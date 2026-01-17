from typing import Any

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from fragnnet.model import GNNModel, SpectrumEncoder
from fragnnet.model.loss import clip_loss
from fragnnet.pl_model import SpectrumPL

# Try to import Lightning, supporting both lightning.pytorch and pytorch_lightning for compatibility
try:
    import lightning.pytorch as pl
except ModuleNotFoundError:
    pass


class SpectrumMolClipPL(SpectrumPL):
    """
    PyTorch Lightning module for a clip loss model on ms/ms spectra and molecular graphs.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.save_hyperparameters()

        # setup loss function
        self.loss_fn = self.clip_loss_fn

        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay
        self.auxiliary_scores = self.hparams.get("auxiliary_scores", [])

        # Define CLIP loss
        self.loss_fn = nn.CosineEmbeddingLoss()

    def _setup_model(self) -> None:
        """Initialize molecule and spectrum encoders used for CLIP-style training."""
        # TODO: initialize model based on params. finalize params
        self.mol_encoder = GNNModel(
            mol_node_feats=self.hparams.mol_params["pyg_node_feats"],  # mol feats
            mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
            mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
            mol_hidden_size=self.hparams.mol_hidden_size,
            mol_num_layers=self.hparams.mol_num_layers,
            mol_gnn_type=self.hparams.mol_gnn_type,
            mol_normalization=self.hparams.mol_normalization,
            mol_dropout=self.hparams.mol_dropout,
            mol_pool_type=self.hparams.mol_pool_type,
            mlp_hidden_size=self.hparams.mlp_hidden_size,  # FFN
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            mz_max=self.hparams.mz_max,
            mz_bin_res=self.hparams.mz_bin_res,
            ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
            ff_bidirectional=self.hparams.ff_bidirectional,
            ff_output_map_size=self.hparams.ff_output_map_size,
            ff_output_activation=self.hparams.ff_output_activation,
            int_embedder=self.hparams.int_embedder,  # cross entropy
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            ce_mean=self.hparams.ce_mean,
            ce_std=self.hparams.ce_std,
            ce_max=self.hparams.ce_max,
            prec_insert_location=self.hparams.prec_insert_location,  # precursor
            prec_insert_size=self.hparams.prec_insert_size,
            prec_types=self.hparams.spec_params["prec_types"],
            inst_insert_location=self.hparams.inst_insert_location,  # instrument
            inst_insert_size=self.hparams.inst_insert_size,
            inst_types=self.hparams.spec_params["inst_types"],
            log_min=self.hparams.log_min,
        )

        self.spec_encoder = SpectrumEncoder(
            spectrum_encoder_type=self.hparams.spec_params.get("encoder_type", "set_transformer"),
            input_dim=self.hparams.spec_params["input_dim"],
            hidden_dim=self.hparams.spec_params["mlp_hidden_dim"],
            output_dim=self.hparams.spec_params["embed_dim"],
            num_heads=self.hparams.spec_params.get("num_heads", 4),
            num_inds=self.hparams.spec_params.get("num_inds", 32),
            num_blocks=self.hparams.spec_params["num_layers"],
            dropout=self.hparams.spec_params["dropout"],
            normalization=self.hparams.spec_params.get("normalization", "batch"),
        )

        # compile
        if self.hparams.compile:
            # th_dynamo.reset()
            # self.dynamo_prof = th_dynamo.utils.CompileProfiler()
            self.mol_encoder = th.compile(self.mol_encoder, backend=self.dynamo_prof, dynamic=True)
            self.spec_encoder = th.compile(
                self.spec_encoder, backend=self.dynamo_prof, dynamic=True
            )

    def forward(self, batch: dict[str, Any]) -> tuple[th.Tensor, th.Tensor]:
        """Forward pass through both encoders."""
        # Extract mol and spectrum data from batch
        mol_pyg = batch["mol_pyg"]
        spec_ints = batch["spec_ints"]
        spec_batch_idxs = batch["spec_batch_idxs"]

        # Get CE, Prec, Inst if available
        ce = batch.get("spec_ce")
        prec_type = batch.get("spec_prec_type")
        inst_type = batch.get("spec_inst_type")

        # Encode molecule
        mol_embed = self.mol_encoder(
            mol_pyg,
            ce=ce,
            ce_batch_idxs=spec_batch_idxs if ce is not None else None,
            prec_type=prec_type,
            inst_type=inst_type,
        )

        # Encode spectrum (assuming spectrum is represented as mz/int pairs)
        # You may need to preprocess this into the right format for your encoder
        spec_embed = self.spec_encoder(
            x=spec_ints.unsqueeze(-1),  # or combine mzs and ints
            batch_idx=spec_batch_idxs,
            ce=ce,
            ce_batch_idxs=spec_batch_idxs if ce is not None else None,
            prec_type=prec_type,
            inst_type=inst_type,
        )

        return mol_embed, spec_embed

    def _common_step(
        self, batch: dict[str, Any], split: str = "train"
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Common logic for train/val/test/inference steps."""
        # Forward pass to get embeddings
        mol_embed, spec_embed = self.forward(batch)

        # Normalize embeddings (important for contrastive learning)
        mol_embed = F.normalize(mol_embed, dim=-1)
        spec_embed = F.normalize(spec_embed, dim=-1)

        # Compute CLIP loss
        loss = clip_loss(mol_embed, spec_embed)

        # Log metrics
        self.log(
            f"{split}_loss",
            loss,
            on_step=(split == "train"),
            on_epoch=True,
            prog_bar=True,
            batch_size=batch["batch_size"],
        )

        return mol_embed, spec_embed, loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> th.Tensor:
        """Training step."""
        _, _, loss = self._common_step(batch, split="train")
        return loss

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> th.Tensor:
        """Validation step."""
        _, _, loss = self._common_step(batch, split="val")
        return loss

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> th.Tensor:
        """Test step."""
        _, _, loss = self._common_step(batch, split="test")
        return loss
