
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from fragnnet.model import SiameseGNNModel
from fragnnet.utils.data_utils import min_max_denormalize, zscore_denormalize
from fragnnet.utils.nn_utils import build_lr_scheduler, get_optimizer_class
import torch._dynamo as th_dynamo

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
		self._setup_model()

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

	def _setup_model(self):
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
			combination_mode=self.hparams.mces_embed_combine,
			#log_min=0.0
		)

		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)
   
	def _log_score(self, pred, target, score_type, split_name, original_scale=False):
		"""Helper to compute and log scores (mean and std) in normalized or original scale."""
		if original_scale is False:
			score_prefix = f"{split_name}_normalized_{score_type}"
			pred_denorm = pred.squeeze()
			target_denorm = target
		else:
			pred_denorm = self.denormalize_mces(pred.squeeze())
			target_denorm = self.denormalize_mces(target)
			score_prefix = f"{split_name}_original_{score_type}"
			
		if score_type == "mae":
			# Compute per-sample absolute errors
			errors = th.abs(pred_denorm - target_denorm)
		elif score_type == "mse":
			# Compute per-sample squared errors
			errors = (pred_denorm - target_denorm) ** 2
		else:
			raise ValueError(f"Unsupported score_type {score_type} for logging.")
		
		# Log mean (MAE or MSE)
		score_mean = errors.mean()
		self.log(score_prefix, score_mean, on_step=False, on_epoch=True, prog_bar=False)
		
		# Log standard deviation
		score_std = errors.std()
		self.log(f"{score_prefix}_std", score_std, on_step=False, on_epoch=True, prog_bar=False)


	def forward(self, mol_pyg_1, mol_pyg_2, **kwargs):
		# Forward pass through SiameseGNNModel
		return self.model(mol_pyg_1, mol_pyg_2, **kwargs)

	def training_step(self, batch, batch_idx):
		# batch: dict with keys 'mol_1_mol_pyg', 'mol_2_mol_pyg', 'mces'
		# Note: 'mces' is already normalized by the dataset
		pred = self.forward(batch['mol_1_mol_pyg'], batch['mol_2_mol_pyg'])
		target = batch['mces'].float().to(pred.device)
		loss = self.loss_fn(pred, target)  # Loss on normalized scale
		
		# Log loss on normalized scale
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
		
		# Log auxiliary scores in both normalized and original scales
		for score_type in self.auxiliary_scores:
			self._log_score(pred, target, score_type, "train", original_scale=False)
			if self.hparams.mces_normalization not in ["none", None]:
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
			if self.hparams.mces_normalization not in ["none", None]:
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
			if self.hparams.mces_normalization not in ["none", None]:
				self._log_score(pred, target, score_type, "test", original_scale=True)

		return loss

	def predict_step(self, batch, batch_idx):
		"""Prediction step that returns denormalized MCES values."""
		pred = self.forward(batch['mol_1_mol_pyg'], batch['mol_2_mol_pyg'])
		# Denormalize predictions to return in original scale
		pred_original = self.denormalize_mces(pred.squeeze())
		return pred_original

	def denormalize_mces(self, normalized_mces):
		"""Denormalize MCES values based on the normalization type."""
		if self.hparams.mces_normalization == "zscore":
			denorm_mces = zscore_denormalize(normalized_mces, self.hparams.mces_mean, self.hparams.mces_std)
		elif self.hparams.mces_normalization == "minmax":
			denorm_mces = min_max_denormalize(normalized_mces, self.hparams.mces_min, self.hparams.mces_max)
		elif self.hparams.mces_normalization == "none" or self.hparams.mces_normalization is None:
			denorm_mces = normalized_mces
		else:
			raise ValueError(f"Unsupported mces_normalization {self.hparams.mces_normalization} for denormalization.")
		return denorm_mces

	def configure_optimizers(self):
		"""
		Configure optimizer and learning rate scheduler for Lightning.
		"""
		optimizer_cls = get_optimizer_class(self.hparams.optimizer)
		optimizer = optimizer_cls(
			self.parameters(), 
			lr=self.hparams.lr, 
			weight_decay=self.hparams.weight_decay
		)
		ret = {
			"optimizer": optimizer,
		}
		if self.hparams.lr_schedule:
			scheduler = build_lr_scheduler(
				optimizer=optimizer, 
				decay_rate=self.hparams.lr_decay_rate, 
				warmup_steps=self.hparams.lr_warmup_steps,
				decay_steps=self.hparams.lr_decay_steps,
			)
			ret["lr_scheduler"] = {
				"scheduler": scheduler,
				"frequency": 1,
				"interval": "step",
			}
		return ret