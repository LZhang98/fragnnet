
import torch as th
import torch.nn as nn
from fragnnet.model import SiameseGNNModel

# Try to import Lightning, supporting both lightning.pytorch and pytorch_lightning for compatibility
try:
	import lightning.pytorch as pl
except ModuleNotFoundError:
	import pytorch_lightning as pl
class MCESPL(pl.LightningModule):
    """
    PyTorch Lightning module for MCES regression using SiameseGNNModel and MCESDataset.
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SiameseGNNModel(
            mol_node_feats=self.hparams.mol_params["pyg_node_feats"],	# mol feats
            mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
            mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
            mol_hidden_size=self.hparams.mol_hidden_size,
            mol_num_layers=self.hparams.mol_num_layers,
            mol_gnn_type=self.hparams.mol_gnn_type,
            mol_normalization=self.hparams.mol_normalization,
            mol_dropout=self.hparams.mol_dropout,
            mol_pool_type=self.hparams.mol_pool_type,
            mlp_hidden_size=self.hparams.mlp_hidden_size,
            mlp_dropout=self.hparams.mlp_dropout,
            mlp_num_layers=self.hparams.mlp_num_layers,
            mlp_use_residuals=self.hparams.mlp_use_residuals,
            #mz_max=0,  # not used
            #mz_bin_res=0,  # not used
            #ff_prec_mz_offset=0,  # not used
            #ff_bidirectional=False,  # not used
            #ff_output_map_size=0,  # not used
            #ff_output_activation="none",  # not used
			int_embedder=self.hparams.int_embedder, # not used
			#int_embedder_tight=self.hparams.int_embedder_tight, # not used
            ce_insert_type=self.hparams.ce_insert_type,
            ce_insert_location=self.hparams.ce_insert_location,
            ce_insert_merge=self.hparams.ce_insert_merge,
            ce_insert_size=self.hparams.ce_insert_size,
            ce_max=self.hparams.ce_max,
            ce_mean=self.hparams.ce_mean,
            ce_std=self.hparams.ce_std,
            prec_insert_location=self.hparams.prec_insert_location,
            prec_insert_size=self.hparams.prec_insert_size,
            prec_types=self.hparams.spec_params["prec_types"],
            inst_insert_location=self.hparams.inst_insert_location,
            inst_insert_size=self.hparams.inst_insert_size,
            inst_types=self.hparams.spec_params["inst_types"],
            #log_min=0.0
        )
        self.loss_fn = nn.MSELoss()
        self.lr = self.hparams.lr
        self.weight_decay = self.hparams.weight_decay

    def forward(self, mol_pyg_i, mol_pyg_j, **kwargs):
        # Forward pass through SiameseGNNModel
        return self.model(mol_pyg_i, mol_pyg_j, **kwargs)

    def training_step(self, batch, batch_idx):
        # batch: dict with keys 'mol_pyg_i', 'mol_pyg_j', 'mces'
        pred = self.forward(batch['mol_pyg_i'], batch['mol_pyg_j'])
        target = batch['mces'].float().to(pred.device)
        loss = self.loss_fn(pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred = self.forward(batch['mol_pyg_i'], batch['mol_pyg_j'])
        target = batch['mces'].float().to(pred.device)
        loss = self.loss_fn(pred, target)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        pred = self.forward(batch['mol_pyg_i'], batch['mol_pyg_j'])
        target = batch['mces'].float().to(pred.device)
        loss = self.loss_fn(pred, target)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = th.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        return optimizer
