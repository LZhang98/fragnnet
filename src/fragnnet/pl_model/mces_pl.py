
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
        mol_node_feats,
        mol_edge_feats,
        mol_pe_embed_k,
        mol_hidden_size,
        mol_num_layers,
        mol_gnn_type,
        mol_normalization,
        mol_dropout,
        mol_pool_type,
        mlp_hidden_size,
        mlp_dropout,
        mlp_num_layers,
        mlp_use_residuals,
        int_embedder,
        ce_insert_type,
        ce_insert_location,
        ce_insert_merge,
        ce_insert_size,
        ce_max,
        ce_mean,
        ce_std,
        prec_insert_location,
        prec_insert_size,
        prec_types,
        inst_insert_location,
        inst_insert_size,
        inst_types,
        lr=1e-3,
        weight_decay=0.0,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = SiameseGNNModel(
            mol_node_feats=mol_node_feats,
            mol_edge_feats=mol_edge_feats,
            mol_pe_embed_k=mol_pe_embed_k,
            mol_hidden_size=mol_hidden_size,
            mol_num_layers=mol_num_layers,
            mol_gnn_type=mol_gnn_type,
            mol_normalization=mol_normalization,
            mol_dropout=mol_dropout,
            mol_pool_type=mol_pool_type,
            mlp_hidden_size=mlp_hidden_size,
            mlp_dropout=mlp_dropout,
            mlp_num_layers=mlp_num_layers,
            mlp_use_residuals=mlp_use_residuals,
            mz_max=0,  # not used
            mz_bin_res=0,  # not used
            ff_prec_mz_offset=0,  # not used
            ff_bidirectional=False,  # not used
            ff_output_map_size=0,  # not used
            ff_output_activation="none",  # not used
            int_embedder=int_embedder,
            ce_insert_type=ce_insert_type,
            ce_insert_location=ce_insert_location,
            ce_insert_merge=ce_insert_merge,
            ce_insert_size=ce_insert_size,
            ce_max=ce_max,
            ce_mean=ce_mean,
            ce_std=ce_std,
            prec_insert_location=prec_insert_location,
            prec_insert_size=prec_insert_size,
            prec_types=prec_types,
            inst_insert_location=inst_insert_location,
            inst_insert_size=inst_insert_size,
            inst_types=inst_types,
            log_min=0.0
        )
        self.loss_fn = nn.MSELoss()
        self.lr = lr
        self.weight_decay = weight_decay

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
