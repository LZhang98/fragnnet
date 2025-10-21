
import torch as th
import torch.nn as nn
import torch.nn.functional as F
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

		# setup loss function
		if self.hparams.loss_type == "mse":
			self.loss_fn = nn.MSELoss()
		elif self.hparams.loss_type == "mae":
			self.loss_fn = nn.L1Loss()
		else:
			raise ValueError(f"Not Supported loss_type {self.hparams.loss_type}")
   
		self.lr = self.hparams.lr
		self.weight_decay = self.hparams.weight_decay
		self.auxiliary_scores = self.hparams.get("auxiliary_scores", [])

	def normalize_mces(self, mces_value):
		"""Normalize MCES value (if needed for inference)."""
		if self.hparams.get("mces_normalization") == "minmax":
			return (mces_value - self.hparams.mces_min) / (self.hparams.mces_max - self.hparams.mces_min + 1e-8)
		elif self.hparams.get("mces_normalization") == "zscore":
			return (mces_value - self.hparams.mces_mean) / (self.hparams.mces_std + 1e-8)
		return mces_value

	def denormalize_mces(self, normalized_value):
		"""Denormalize MCES predictions back to original scale."""
		if self.hparams.get("mces_normalization") == "minmax":
			return normalized_value * (self.hparams.mces_max - self.hparams.mces_min + 1e-8) + self.hparams.mces_min
		elif self.hparams.get("mces_normalization") == "zscore":
			return normalized_value * (self.hparams.mces_std + 1e-8) + self.hparams.mces_mean
		return normalized_value

	def _log_score(self, pred, target, score_type, split_name, original_scale=False):
		"""Helper to compute and log scores in original scale."""
		if original_scale:
			score_name = f"{split_name}_{score_type}_original"
			pred_denorm = self.denormalize_mces(pred.squeeze())
			target_denorm = self.denormalize_mces(target)
		else:
			score_name = f"{split_name}_{score_type}"
			pred_denorm = pred.squeeze()
			target_denorm = target
   
		with th.no_grad():
			if score_type == "mae":
				score = F.l1_loss(pred_denorm, target_denorm)
			elif score_type == "mse":
				score = F.mse_loss(pred_denorm, target_denorm)
			else:
				raise ValueError(f"Unsupported score_type {score_type} for logging original scale.")
		
		self.log(score_name, score, on_step=False, on_epoch=True, prog_bar=False)

	def forward(self, mol_pyg_1, mol_pyg_2, **kwargs):
		# Forward pass through SiameseGNNModel
		return self.model(mol_pyg_1, mol_pyg_2, **kwargs)

	def training_step(self, batch, batch_idx):
		# batch: dict with keys 'mol_1_mol_pyg', 'mol_2_mol_pyg', 'mces'
		pred = self.forward(batch['mol_1_mol_pyg'], batch['mol_2_mol_pyg'])
		target = batch['mces'].float().to(pred.device)
		loss = self.loss_fn(pred, target)  # Loss on normalized scale
		
		# Log loss on normalized scale
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
		
		# Log auxiliary scores in both normalized and original scales
		for score_type in self.auxiliary_scores:
			self._log_score(pred, target, score_type, "train", original_scale=False)
			self._log_score(pred, target, score_type, "train", original_scale=True)

		return loss

	def validation_step(self, batch, batch_idx):
		pred = self.forward(batch['mol_1_mol_pyg'], batch['mol_2_mol_pyg'])
		target = batch['mces'].float().to(pred.device)
		loss = self.loss_fn(pred, target)  # Loss on normalized scale
		
		# Log loss on normalized scale
		self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		
		# Log MAE in both normalized and original scales
		for score_type in self.auxiliary_scores:
			self._log_score(pred, target, score_type, "val", original_scale=False)
			self._log_score(pred, target, score_type, "val", original_scale=True)
		return loss

	def test_step(self, batch, batch_idx):
		pred = self.forward(batch['mol_1_mol_pyg'], batch['mol_2_mol_pyg'])
		target = batch['mces'].float().to(pred.device)
		loss = self.loss_fn(pred, target)  # Loss on normalized scale
		
		# Log loss on normalized scale
		self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
		
		# Log MAE in both normalized and original scales
		for score_type in self.auxiliary_scores:
			self._log_score(pred, target, score_type, "test", original_scale=False)
			self._log_score(pred, target, score_type, "test", original_scale=True)

		return loss

	def configure_optimizers(self):
		
		if self.hparams.optimizer == "adam":
			optimizer_cls = th.optim.Adam
		elif self.hparams.optimizer == "adamw":
			optimizer_cls = th.optim.AdamW
		elif self.hparams.optimizer == "sgd":
			optimizer_cls = th.optim.SGD
		else:
			raise ValueError(f"Unknown optimizer {self.optimizer}")
		
		optimizer = optimizer_cls(
			self.parameters(),
			lr=self.lr,
			weight_decay=self.weight_decay
		)
		return optimizer