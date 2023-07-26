import torch as th
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from torch.profiler import profile, record_function, ProfilerActivity
import torch._dynamo as th_dynamo

from frag_gnn.nn.nn_utils import build_lr_scheduler, JointPredictor
from frag_gnn.nn.loss import get_sparse_cross_entropy_fn, get_sparse_entropy_fn
from frag_gnn.utils.spec_utils import batched_filter_func, batched_bin_func, calculate_cosine_similarity, get_ints_transform_func, get_ints_untransform_func, batched_l1normalize, calculate_weighted_recall
from frag_gnn.utils.misc_utils import safelog, flatten_lol, th_temp_seed, scatter_reduce
from frag_gnn.frag.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.spec_utils import MZ_MAX, MZ_BIN_RES, INTS_THRESH
from frag_gnn.utils.plot_utils import plot_spectra_sparse


def batchwise_max(xs, batch_idxs):
	""" for debugging """

	batch_size = th.max(batch_idxs)+1
	maxs = th.zeros([batch_size],device=xs.device,dtype=xs.dtype)
	for b in range(batch_size):
		maxs[b] = th.max(xs[batch_idxs==b])
	return maxs

def batchwise_lse(xs, batch_idxs):
	""" for debugging """

	batch_size = th.max(batch_idxs)+1
	lses = th.zeros([batch_size],device=xs.device,dtype=xs.dtype)
	for b in range(batch_size):
		lses[b] = th.logsumexp(xs[batch_idxs==b],0)
	return lses

def to_cpu(data_d,non_blocking=True,detach=True):

	for k in data_d.keys():
		if isinstance(data_d[k],th.Tensor):
			data = data_d[k]
			if detach:
				data = data.detach()
			data = data.to("cpu",non_blocking=non_blocking)
			data_d[k] = data
	return data_d

class FragGNN(pl.LightningModule):

	def __init__(
		self,
		num_depth: int,
		num_hs: int,
		num_elements: int,
		int_embedder: str,
		mol_pyg_node_feats: list[str],
		mol_pyg_edge_feats: list[str],
		mol_pyg_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_dropout: float,
		mol_normalization: str,
		mol_pool_type: str,
		frag_pyg_node_feats: list[str],
		frag_pyg_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_dropout: float,
		frag_normalization: str,
		frag_pool_type: str,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		lr: float,
		lr_schedule: bool,
		weight_decay: float,
		lr_decay_rate: float,
		lr_warmup_steps: int,
		lr_decay_steps: int,
		optimizer: str,
		output_distribution: str,
		loss_vectorized: bool,
		loss_tolerance_ppm: float,
		loss_tolerance_abs: float,
		auxiliary_scores: list[str],
		ints_thresh: float,
		mz_max: float,
		mz_bin_res: float,
		bin_sum_ints: bool,
		ints_transform: str,
		formula_entropy_weight: float,
		node_entropy_weight: float,
		automatic_optimization: bool,
		num_log_train_images: int,
		num_log_val_images: int,
		**kwargs
	):

		super().__init__()
		self.save_hyperparameters()
		self.model = JointPredictor(
			num_depth=num_depth,
			num_hs=num_hs,
			num_elements=num_elements,
			int_embedder=int_embedder,
			mol_node_feats=mol_pyg_node_feats,
			mol_edge_feats=mol_pyg_edge_feats,
			mol_pe_embed_k=mol_pyg_pe_embed_k,
			mol_hidden_size=mol_hidden_size,
			mol_num_layers=mol_num_layers,
			mol_gnn_type=mol_gnn_type,
			mol_dropout=mol_dropout,
			mol_normalization=mol_normalization,
			mol_pool_type=mol_pool_type,
			frag_node_feats=frag_pyg_node_feats,
			frag_edge_feats=frag_pyg_edge_feats,
			frag_hidden_size=frag_hidden_size,
			frag_num_layers=frag_num_layers,
			frag_gnn_type=frag_gnn_type,
			frag_dropout=frag_dropout,
			frag_normalization=frag_normalization,
			frag_pool_type=frag_pool_type,
			mlp_hidden_size=mlp_hidden_size,
			mlp_dropout=mlp_dropout,
			mlp_num_layers=mlp_num_layers,
			mlp_use_residuals=mlp_use_residuals
		)
		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model.compile_submodules(backend=self.dynamo_prof,dynamic=True)
		# set tolerance
		if self.hparams.loss_tolerance_ppm is not None:
			self.tolerance = self.hparams.loss_tolerance_ppm
			self.relative = True
		else:
			assert self.hparams.loss_tolerance_abs is not None
			self.tolerance = self.hparams.loss_tolerance_abs
			self.relative = False
		# setup functions
		self._setup_loss_fn()
		def _filter_func(mzs, ints, batch_idxs):
			return batched_filter_func(mzs,ints,batch_idxs,ints_thresh,mz_max)
		self.filter_func = _filter_func
		def _bin_func(mzs, ints, batch_idxs):
			return batched_bin_func(mzs,ints,batch_idxs,mz_bin_res,bin_sum_ints)
		self.bin_func = _bin_func
		self.ints_transform_func = get_ints_transform_func(ints_transform)
		self.ints_untransform_func = get_ints_untransform_func(ints_transform)
		self.ints_normalize_func = batched_l1normalize
		self.auxiliary_score_fns = {}
		if "cos_sim" in auxiliary_scores:
			def _cos_sim(
				true_mzs,
				true_ints,
				true_batch_idxs,
				pred_mzs,
				pred_ints,
				pred_batch_idxs,
				**kwargs):
				# untransform
				true_ints = self.ints_untransform_func(true_ints)
				# true_ints = self.ints_normalize_func(true_ints,true_batch_idxs)
				pred_ints = self.ints_untransform_func(pred_ints)
				# pred_ints = self.ints_normalize_func(pred_ints,pred_batch_idxs)
				# calculate
				return calculate_cosine_similarity(
					true_mzs,
					true_ints,
					true_batch_idxs,
					pred_mzs,
					pred_ints,
					pred_batch_idxs,
					**kwargs
				)
			self.auxiliary_score_fns["cos_sim"] = _cos_sim
		if "wrecall" in auxiliary_scores:
			def _wrecall(
				true_mzs,
				true_ints,
				true_batch_idxs,
				pred_mzs,
				pred_ints,
				pred_batch_idxs,
				**kwargs):
				# untransform
				true_ints = self.ints_untransform_func(true_ints)
				true_ints = self.ints_normalize_func(true_ints,true_batch_idxs)
				pred_ints = self.ints_untransform_func(pred_ints)
				pred_ints = self.ints_normalize_func(pred_ints,pred_batch_idxs)
				# calculate
				return calculate_weighted_recall(
					true_mzs,
					true_ints,
					true_batch_idxs,
					pred_mzs,
					pred_ints,
					pred_batch_idxs,
					tolerance=self.tolerance,
					relative=self.relative,
					**kwargs
				)
			self.auxiliary_score_fns["wrecall"] = _wrecall
		self.automatic_optimization = automatic_optimization
		self.train_results = None
		self.train_counter = 0
		self.val_results = None
		self.val_counter = 0

	def _setup_loss_fn(
		self):

		sparse_ce = get_sparse_cross_entropy_fn(
			self.hparams.output_distribution,
			self.hparams.loss_vectorized,
			self.tolerance,
			self.relative
		)
		sparse_entropy = get_sparse_entropy_fn()
		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			pred_formula_logprobs,
			pred_formula_batch_idxs,
			pred_node_logprobs,
			pred_node_batch_idxs,
			pred_node_formula_entropy):
			spec_ce = sparse_ce(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs
			)
			loss = spec_ce
			formula_e = sparse_entropy(
				pred_formula_logprobs,
				pred_formula_batch_idxs
			)
			loss = loss + self.hparams.formula_entropy_weight * formula_e
			node_e = sparse_entropy(
				pred_node_logprobs,
				pred_node_batch_idxs
			)
			loss = loss + self.hparams.node_entropy_weight * node_e
			node_formula_e = scatter_reduce(
				th.exp(pred_node_logprobs) * pred_node_formula_entropy,
				pred_node_batch_idxs,
				"sum",
				0,
				th.max(pred_node_batch_idxs)+1
			)
			loss = loss + self.hparams.node_formula_entropy_weight * node_formula_e
			loss_d = {
				"loss": loss,
				"spec_ce": spec_ce,
				"formula_e": formula_e,
				"node_e": node_e,
				"node_formula_e": node_formula_e
			}
			return loss_d
		self.loss_fn = _loss_fn

	def preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs):

		# filter
		spec_mzs, spec_ints, spec_batch_idxs = self.filter_func(
			spec_mzs,
			spec_ints,
			spec_batch_idxs
		)
		# transform
		spec_ints = self.ints_transform_func(spec_ints)
		# renormalize
		spec_ints = self.ints_normalize_func(
			spec_ints,
			spec_batch_idxs
		)
		# log
		spec_ints = safelog(spec_ints)
		return spec_mzs, spec_ints, spec_batch_idxs

	def predict_step(self,**batch_kwargs):
		# TODO: might need to change this
		return self.forward(**batch_kwargs)

	def forward(self,**batch_kwargs):

		# get predictions
		pred = self.model.forward(**batch_kwargs)
		return pred

	def _common_step(self, batch, split="train"):

		# preprocess spec
		batch_size = batch["batch_size"]
		unique_id = batch["spec_unique_id"]
		smiles = batch["mol_smiles"]
		spec_mzs = batch["spec_mzs"]
		spec_ints = batch["spec_ints"]
		spec_batch_idxs = batch["spec_batch_idxs"]
		# print("pre_spec_max",batchwise_max(spec_ints,spec_batch_idxs))
		spec_mzs, spec_ints, spec_batch_idxs = self.preproc_spec(
			spec_mzs,
			spec_ints,
			spec_batch_idxs
		)

		pred = self.forward(**batch)

		both = {
			"true_mzs": spec_mzs,
			"true_logprobs": spec_ints,
			"true_batch_idxs": spec_batch_idxs,
			**pred
		}

		loss_d = self.loss_fn(**both)
		loss = loss_d["loss"]
		mean_loss = th.mean(loss,dim=0)
		# log losses and entropies
		with th.no_grad():
			for k,v in loss_d.items():
				mean_v = th.mean(v,dim=0)
				self.log(
					f"{split}_{k}_batch",
					mean_v.item(),
					batch_size=batch_size,
					on_epoch=True
				)
		results = {
			"unique_id": unique_id,
			"smiles": smiles,
			"loss": loss,
			"mean_loss": mean_loss,
			**both
		}
		assert spec_mzs.shape[0] > 0, spec_mzs.shape[0]
		with th.no_grad():
			for metric_name,metric_fn in self.auxiliary_score_fns.items():
				metric = metric_fn(
					spec_mzs,
					th.exp(spec_ints),
					spec_batch_idxs,
					pred["pred_mzs"],
					th.exp(pred["pred_logprobs"]),
					pred["pred_batch_idxs"]
				)
				mean_metric = th.mean(metric,dim=0)
				self.log(
					f"{split}_{metric_name}_batch",
					mean_metric,
					batch_size=batch_size,
					on_epoch=True
				)
				results[metric_name] = metric
		return results

	def training_step(self, batch, batch_idx):

		if self.hparams.automatic_optimization:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results.pop("mean_loss")
			self._update_results(batch_results,"train")
			return mean_loss
		else:
			raise NotImplementedError("Manual optimization not implemented")

	def validation_step(self, batch, batch_idx):

		batch_results = self._common_step(batch, split="val")
		mean_loss = batch_results.pop("mean_loss")
		self._update_results(batch_results,"val")
		return mean_loss

	def test_step(self, batch, batch_idx):

		batch_results = self._common_step(batch, split="test")
		mean_loss = batch_results.pop("mean_loss")
		# self._update_results(batch_results,"test")
		return mean_loss

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

	def _get_wandb_logger(self):

		wandb_logger = None
		for logger in self.loggers:
			if isinstance(logger, pl.loggers.WandbLogger):
				wandb_logger = logger		
		return wandb_logger

	def _update_results(self, batch_results, split):

		if split == "train":
			num_log_images = self.hparams.num_log_train_images
			results_attr = "train_results"
			counter_attr = "train_counter"
		elif split == "val":
			num_log_images = self.hparams.num_log_val_images
			results_attr = "val_results"
			counter_attr = "val_counter"
		else:
			raise NotImplementedError(split)
		# filter keys (filtering first to save time/memory)
		keys = [
			"unique_ids",
			"smiles",
			"loss",
			"true_mzs",
			"true_logprobs",
			"true_unique_ids",
			"pred_mzs",
			"pred_logprobs",
			"pred_unique_ids"
		]
		keys.extend(list(self.auxiliary_score_fns.keys()))
		unique_ids = batch_results.pop("unique_id")
		true_batch_idxs = batch_results.pop("true_batch_idxs")
		pred_batch_idxs = batch_results.pop("pred_batch_idxs")
		batch_results = {k:v for k,v in batch_results.items() if k in keys}
		# update unique_ids
		true_unique_ids = unique_ids[true_batch_idxs]
		pred_unique_ids = unique_ids[pred_batch_idxs]
		batch_results["unique_ids"] = unique_ids
		batch_results["true_unique_ids"] = true_unique_ids
		batch_results["pred_unique_ids"] = pred_unique_ids
		# transfer to cpu
		batch_results = to_cpu(batch_results)
		# setup results dict
		if getattr(self,results_attr) is None:
			setattr(self,results_attr,{k:list() for k in keys})
		else:
			assert set(keys) == set(getattr(self,results_attr).keys())
		results_dict = getattr(self,results_attr)
		# add to results dict
		for k,v in batch_results.items():
			results_dict[k].append(v)
		# increment counter
		setattr(self,counter_attr,getattr(self,counter_attr)+unique_ids.shape[0])

	def _consolidate_results(self, split):
		
		results = getattr(self,f"{split}_results")
		if results is None:
			return
		keys = results.keys()
		for k in keys:
			if isinstance(results[k][0],th.Tensor):
				results[k] = th.cat(results[k],dim=0)
			else:
				assert isinstance(results[k][0],list)
				results[k] = flatten_lol(results[k])

	def _log_images(self, split):

		if split == "train":
			results = self.train_results
			num_log_images = self.hparams.num_log_train_images
			counter = self.train_counter
		elif split == "val":
			results = self.val_results
			num_log_images = self.hparams.num_log_val_images
			counter = self.val_counter
		else:
			raise NotImplementedError(split)
		num_log_images = min(num_log_images,counter)
		# log all the metrics
		for metric_name in ["loss","cos_sim","wrecall"]:
			self.log(
				f"{split}_{metric_name}_epoch",
				results[metric_name].mean()
			)
		# if num_log_images == 0:
		# 	return
		wandb_logger = self._get_wandb_logger()
		if wandb_logger is None:
			return
		# randomly sample unique_ids
		unique_ids = th.unique(results["unique_ids"],sorted=True)
		if num_log_images == unique_ids.shape[0]:
			sample_idxs = th.arange(num_log_images)
			sample_unique_ids = unique_ids
		else:
			with th_temp_seed(420):
				sample_idxs = th.randperm(unique_ids.shape[0])[:num_log_images]
				sample_unique_ids = unique_ids[sample_idxs]
		# plot images
		for i in range(num_log_images):
			unique_id = sample_unique_ids[i].item()
			unique_idx = th.nonzero(results["unique_ids"] == unique_id,as_tuple=False).item()
			true_mask = results["true_unique_ids"] == unique_id
			pred_mask = results["pred_unique_ids"] == unique_id
			smiles = results["smiles"][unique_idx]
			loss = results["loss"][unique_idx].item()
			if "cos_sim" in results:
				cos_sim = results["cos_sim"][unique_idx].item()
			else:
				cos_sim = np.nan
			if "wrecall" in results:
				wrecall = results["wrecall"][unique_idx].item()
			else:
				wrecall = np.nan
			true_mzs = results["true_mzs"][true_mask]
			true_ints = th.exp(results["true_logprobs"][true_mask])
			pred_mzs = results["pred_mzs"][pred_mask]
			pred_ints = th.exp(results["pred_logprobs"][pred_mask])
			# untransform
			true_ints = self.ints_untransform_func(true_ints)
			true_ints = self.ints_normalize_func(
				true_ints,
				th.zeros_like(true_ints,dtype=th.long)
			)
			pred_ints = self.ints_untransform_func(pred_ints)
			pred_ints = self.ints_normalize_func(
				pred_ints,
				th.zeros_like(pred_ints,dtype=th.long)
			)
			# cast to numpy
			true_mzs = true_mzs.numpy()
			true_ints = true_ints.numpy()
			pred_mzs = pred_mzs.numpy()
			pred_ints = pred_ints.numpy()
			assert np.isclose(np.sum(true_ints),1.0), np.sum(true_ints)
			assert np.isclose(np.sum(pred_ints),1.0), np.sum(pred_ints)
			# plot
			data = plot_spectra_sparse(
				true_mzs,
				true_ints,
				pred_mzs,
				pred_ints,
				smiles,
				return_data=True
			)
			wandb_logger.log_image(
				key=f"{split}_example_{i}",
				caption=[f"unique_id = {unique_id}, epoch = {self.current_epoch:03d}, loss = {loss:.3f}, cos_sim = {cos_sim:.3f}, wrecall = {wrecall:.3f}"],
				images=[data]
			)

	def on_train_epoch_end(self):

		# consolidate results
		self._consolidate_results("train")
		# log images
		self._log_images("train")
		# reset
		self.train_results = None
		self.train_counter = 0

	def on_validation_epoch_end(self):
		
		# consolidate results
		self._consolidate_results("val")
		# log images
		self._log_images("val")
		# reset
		self.val_results = None
		self.val_counter = 0

def get_default_model_params():
	
	raise NotImplementedError

import pdb