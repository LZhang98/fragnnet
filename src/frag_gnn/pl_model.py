import torch as th
import torch.nn.functional as F
import numpy as np

# make cc happy
try:
	import lightning.pytorch as pl
except ModuleNotFoundError:
	import pytorch_lightning as pl
	
from torch.profiler import profile, record_function, ProfilerActivity
import torch._dynamo as th_dynamo

from frag_gnn.utils.nn_utils import build_lr_scheduler
from frag_gnn.model import FragGNNModel, NeimsModel
from frag_gnn.loss import get_sparse_cross_entropy_fn, sparse_entropy_fn, sparse_conditional_entropy_fn, get_edge_loss_fn
from frag_gnn.utils.spec_utils import batched_filter_func, batched_bin_func, calculate_cosine_similarity, \
	calculate_sparse_cosine_similarity, calculate_optimal_cosine_similarity, calculate_optimal_sparse_cosine_similarity, \
	get_ints_transform_func, get_ints_untransform_func, batched_l1_normalize, calculate_recall, calculate_weighted_recall, \
	binned_cosine_similarity
from frag_gnn.utils.misc_utils import safelog, flatten_lol, th_temp_seed, scatter_reduce, scatter_logsumexp, to_cpu
from frag_gnn.utils.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.spec_utils import MZ_MAX, MZ_BIN_RES, INTS_THRESH
from frag_gnn.utils.plot_utils import plot_spectra_sparse


class SpectrumPL(pl.LightningModule):

	def __init__(self,**kwargs):

		super().__init__()
		self.save_hyperparameters()
		# setup functions
		self.metric_names = set()
		self._setup_model()
		self._setup_tolerance()
		self._setup_loss_fn()
		self._setup_spec_fns()
		self._setup_metric_fns()
		self.train_results = None
		self.train_counter = 0
		self.train_mean_metrics = []
		self.val_results = None
		self.val_counter = 0
		self.val_mean_metrics = []

	def _setup_model(self):
		raise NotImplementedError

	def _setup_tolerance(self):

		# set tolerance
		if self.hparams.loss_tolerance_rel is not None:
			self.tolerance = self.hparams.loss_tolerance_rel
			self.relative = True
		else:
			assert self.hparams.loss_tolerance_abs is not None
			self.tolerance = self.hparams.loss_tolerance_abs
			self.relative = False

	def _setup_loss_fn(self):
		raise NotImplementedError

	def _setup_spec_fns(self):

		def _filter_func(mzs, ints, batch_idxs):
			return batched_filter_func(mzs,ints,batch_idxs,self.hparams.ints_thresh,self.hparams.mz_max)
		self.filter_func = _filter_func
		def _bin_func(mzs, ints, batch_idxs):
			return batched_bin_func(mzs,ints,batch_idxs,self.hparams.mz_max,self.hparams.mz_bin_res,self.hparams.sum_ints)
		self.bin_func = _bin_func
		self.ints_transform_func = get_ints_transform_func(self.hparams.ints_transform)
		self.ints_untransform_func = get_ints_untransform_func(self.hparams.ints_transform)
		self.ints_normalize_func = batched_l1_normalize

	def _setup_metric_fns(self):

		# setup metrics
		self.metric_min_max = {"loss": "min"}
		self.metric_fns = {}
		if "cos_sim" in self.hparams.auxiliary_scores:
			if self.hparams.sparse_cosine_similarity:
				cosine_similarity_fn = calculate_sparse_cosine_similarity
			else:
				cosine_similarity_fn = calculate_cosine_similarity
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
				pred_ints = self.ints_untransform_func(pred_ints)
				# calculate
				cos_sim = cosine_similarity_fn(
					true_mzs,
					true_ints,
					true_batch_idxs,
					pred_mzs,
					pred_ints,
					pred_batch_idxs,
					mz_bin_res=self.hparams.mz_bin_res,
					mz_max=self.hparams.mz_max,
					sum_ints=self.hparams.sum_ints
				)
				return cos_sim
			self.metric_fns["cos_sim"] = _cos_sim
			self.metric_min_max["cos_sim"] = "max"
		if "recall" in self.hparams.auxiliary_scores:
			def _recall(
				true_mzs,
				true_ints,
				true_batch_idxs,
				pred_mzs,
				pred_ints,
				pred_batch_idxs,
				**kwargs):
				return calculate_recall(
					true_mzs,
					true_batch_idxs,
					pred_mzs,
					pred_batch_idxs,
					tolerance=self.tolerance,
					relative=self.relative
				)
			self.metric_fns["recall"] = _recall
			self.metric_min_max["recall"] = "max"
		if "wrecall" in self.hparams.auxiliary_scores:
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
					relative=self.relative
				)
			self.metric_fns["wrecall"] = _wrecall
			self.metric_min_max["wrecall"] = "max"
		if "opt_cos_sim" in self.hparams.auxiliary_scores:
			if self.hparams.sparse_cosine_similarity:
				optimal_cosine_similarity_fn = calculate_optimal_sparse_cosine_similarity
			else:
				optimal_cosine_similarity_fn = calculate_optimal_cosine_similarity
			def _opt_cos_sim(
				true_mzs,
				true_ints,
				true_batch_idxs,
				pred_mzs,
				pred_ints,
				pred_batch_idxs,
				**kwargs):
				# untransform
				true_ints = self.ints_untransform_func(true_ints)
				pred_ints = self.ints_untransform_func(pred_ints)
				# calculate
				opt_cos_sim = optimal_cosine_similarity_fn(
					true_mzs,
					true_ints,
					true_batch_idxs,
					pred_mzs,
					pred_ints,
					pred_batch_idxs,
					mz_bin_res=self.hparams.mz_bin_res,
					mz_max=self.hparams.mz_max,
					sum_ints=self.hparams.sum_ints
				)
				return opt_cos_sim
			self.metric_fns["opt_cos_sim"] = _opt_cos_sim
			self.metric_min_max["opt_cos_sim"] = None
		if "true_spec_e" in self.hparams.auxiliary_scores:
			def _true_spec_e(
				true_mzs,
				true_ints,
				true_batch_idxs,
				pred_mzs,
				pred_ints,
				pred_batch_idxs,
				**kwargs):
				# untransform, normalize, log
				true_ints = self.ints_untransform_func(true_ints)
				true_ints = self.ints_normalize_func(true_ints,true_batch_idxs)
				true_log_probs = th.log(true_ints)
				true_spec_e = sparse_entropy_fn(true_log_probs,true_batch_idxs)[0]
				assert not th.isnan(true_spec_e).any(), true_spec_e
				return true_spec_e
			self.metric_fns["true_spec_e"] = _true_spec_e
			self.metric_min_max["true_spec_e"] = None
		if "true_spec_ne" in self.hparams.auxiliary_scores:
			def _true_spec_ne(
				true_mzs,
				true_ints,
				true_batch_idxs,
				pred_mzs,
				pred_ints,
				pred_batch_idxs,
				**kwargs):
				# untransform, normalize, log
				true_ints = self.ints_untransform_func(true_ints)
				true_ints = self.ints_normalize_func(true_ints,true_batch_idxs)
				true_log_probs = th.log(true_ints)
				true_spec_ne = sparse_entropy_fn(true_log_probs,true_batch_idxs)[1]
				assert not th.isnan(true_spec_ne).any(), true_spec_ne
				return true_spec_ne
			self.metric_fns["true_spec_ne"] = _true_spec_ne
			self.metric_min_max["true_spec_ne"] = None
		self.metric_names.update(list(self.metric_min_max.keys()))

	def _preproc_spec(self,spec_mzs,spec_ints,spec_batch_idxs):

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
		true_mzs = batch["spec_mzs"]
		true_ints = batch["spec_ints"]
		true_batch_idxs = batch["spec_batch_idxs"]
		# print("pre_spec_max",batchwise_max(spec_ints,spec_batch_idxs))
		true_mzs, true_logprobs, true_batch_idxs = self._preproc_spec(
			true_mzs,
			true_ints,
			true_batch_idxs
		)
		pred = self.forward(**batch)
		both = {
			"true_mzs": true_mzs,
			"true_logprobs": true_logprobs,
			"true_batch_idxs": true_batch_idxs,
			**pred
		}
		loss_d = self.loss_fn(**both)
		loss = loss_d["loss"]
		mean_loss = th.mean(loss,dim=0)
		total_loss = th.sum(loss,dim=0)
		# just log loss
		self.log(
			f"{split}_batch_loss",
			mean_loss,
			batch_size=batch_size,
			on_epoch=True
		)
		results = {
			"unique_id": unique_id,
			"smiles": smiles,
			"mean_loss": mean_loss,
			"total_loss": total_loss,
			**both,
			**loss_d
		}
		assert true_mzs.shape[0] > 0, true_mzs.shape[0]
		with th.no_grad():
			for metric_name,metric_fn in self.metric_fns.items():
				metric = metric_fn(
					true_mzs,
					th.exp(true_logprobs),
					true_batch_idxs,
					pred["pred_mzs"],
					th.exp(pred["pred_logprobs"]),
					pred["pred_batch_idxs"]
				)
				results[metric_name] = metric
		for metric_name in self.metric_names:
			assert metric_name in results, metric_name
		return results

	def training_step(self, batch, batch_idx):
		""" training loop

		Args:
			batch (_type_): _description_
			batch_idx (_type_): _description_

		Raises:
			NotImplementedError: _description_

		Returns:
			_type_: _description_
		"""

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
			results_attr = "train_results"
			counter_attr = "train_counter"
		elif split == "val":
			results_attr = "val_results"
			counter_attr = "val_counter"
		else:
			raise NotImplementedError(split)
		# filter keys (filtering first to save time/memory)
		keys = [
			"unique_ids",
			"smiles",
			"true_mzs",
			"true_logprobs",
			"true_unique_ids",
			"pred_mzs",
			"pred_logprobs",
			"pred_unique_ids"
		]
		keys.extend(list(self.metric_names))
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
		# check all metrics
		assert all([metric_name in batch_results for metric_name in self.metric_names])
		# transfer to cpu
		batch_results = to_cpu(batch_results,detach=True)
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
		# log all the metrics
		mean_metrics = {k:v.mean() for k,v in results.items() if k in self.metric_names}
		for k,v in mean_metrics.items():
			self.log(
				f"{split}_{k}_epoch",
				v
			)
		# update epoch stats
		mean_metrics_list = getattr(self,f"{split}_mean_metrics")
		mean_metrics_list.append(mean_metrics)
		# log best metric
		for k in ["loss","cos_sim"]:
			if k not in self.metric_names:
				continue
			metric_epochs = [item[k] for item in mean_metrics_list]
			if self.metric_min_max[k] == "min":
				metric_best = th.min(th.tensor(metric_epochs))
			elif self.metric_min_max[k] == "max":
				metric_best = th.max(th.tensor(metric_epochs))
			else:
				assert self.metric_min_max[k] is None, self.metric_min_max[k]
				continue
			self.log(
				f"{split}_{k}_best",
				metric_best
			)

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
		
		if not self.trainer.sanity_checking:
			# consolidate results
			self._consolidate_results("val")
			# log images
			self._log_images("val")
		# reset
		self.val_results = None
		self.val_counter = 0

	def _print_grad_norm(self,prefix=None,total_num_params=5):

		if prefix is None:
			prefix = "no_prefix"
		print(f">> {prefix}, {self._num_need_update}, {self._num_acc_gradient}")
		opt = self.optimizers()
		num_params = 0
		for pg in opt.param_groups:
			for p in pg['params']:
				if p.grad is not None:
					print(th.norm(p.grad))
					num_params += 1
				if num_params > total_num_params:
					break
			if num_params > total_num_params:
				break
	
	def on_after_backward(self):
		
		if self.hparams.check_gradient_norm:
			self._print_grad_norm(prefix="after_backward")

	def on_before_optimizer_step(self, optimizer):

		if self.hparams.check_gradient_norm:
			self._print_grad_norm(prefix="before_optimizer_step")

	def on_after_optimizer_step(self, optimizer):

		self._num_need_update = 0

class FragGNNPL(SpectrumPL):

	def _setup_model(self):

		# frag GNN
		self.model = FragGNNModel(
			num_depth=self.hparams.num_depth,
			num_hs=self.hparams.num_hs,
			num_elements=self.hparams.num_elements,
			int_embedder=self.hparams.int_embedder,
			int_embedder_tight=self.hparams.int_embedder_tight,
			mol_node_feats=self.hparams.mol_params["pyg_node_feats"],
			mol_edge_feats=self.hparams.mol_params["pyg_edge_feats"],
			mol_pe_embed_k=self.hparams.mol_params["pyg_pe_embed_k"],
			mol_hidden_size=self.hparams.mol_hidden_size,
			mol_num_layers=self.hparams.mol_num_layers,
			mol_gnn_type=self.hparams.mol_gnn_type,
			mol_dropout=self.hparams.mol_dropout,
			mol_normalization=self.hparams.mol_normalization,
			mol_pool_type=self.hparams.mol_pool_type,
			frag_node_feats=self.hparams.frag_params["pyg_node_feats"],
			frag_edge_feats=self.hparams.frag_params["pyg_edge_feats"],
			frag_hidden_size=self.hparams.frag_hidden_size,
			frag_num_layers=self.hparams.frag_num_layers,
			frag_gnn_type=self.hparams.frag_gnn_type,
			frag_dropout=self.hparams.frag_dropout,
			frag_normalization=self.hparams.frag_normalization,
			frag_pool_type=self.hparams.frag_pool_type,
			mlp_output_format=self.hparams.mlp_output_format,
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_normalization=self.hparams.mlp_normalization,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			skip_edge_loss=self.hparams.skip_edge_loss,
			mask_null_formula=self.hparams.mask_null_formula,
			predict_oos=self.hparams.predict_oos,
			cc_interstage_type=self.hparams.cc_interstage_type
		)
		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model.compile_submodules(backend=self.dynamo_prof,dynamic=True)

		# for sampler
		self._smf = self.hparams.custom_sampler == "SpecMolFragDynamicBatchSampler"
		self._num_need_update = 0

		# this used in manul update
		self._num_acc_gradient = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
		self.automatic_optimization = self.hparams.automatic_optimization

	def _setup_loss_fn(self):

		# cross entropy
		sparse_ce_fn = get_sparse_cross_entropy_fn(
			self.hparams.output_distribution,
			self.hparams.loss_vectorized,
			self.tolerance,
			self.relative,
			self.hparams.oos_tolerance_multiple
		)
		# edge loss
		edge_h_range_loss_fn = get_edge_loss_fn(self.hparams.edge_h_range_loss_fn,4.0)
		edge_h_transfer_loss_fn = get_edge_loss_fn(self.hparams.edge_h_transfer_loss_fn,4.0) #,1.0)

		if self.hparams.sparse_cosine_similarity:
			cosine_similarity_fn = calculate_sparse_cosine_similarity
		else:
			cosine_similarity_fn = calculate_cosine_similarity
		def cosine_distance_fn(
			true_mzs,
			true_ints,
			true_batch_idxs,
			pred_mzs,
			pred_ints,
			pred_batch_idxs):
			# calculate
			return 1.-cosine_similarity_fn(
				true_mzs,
				th.exp(true_ints),
				true_batch_idxs,
				pred_mzs,
				th.exp(pred_ints),
				pred_batch_idxs,
				mz_bin_res=self.hparams.mz_bin_res,
				mz_max=self.hparams.mz_max,
				sum_ints=self.hparams.sum_ints
			)

		def edge_loss(
			pred_edge_logprobs,
			pred_edge_h_diffs,
			pred_edge_h_range_masks,
			pred_edge_h_logprobs,
			pred_edge_batch_idxs):
			edge_h_range_costs = edge_h_range_loss_fn(pred_edge_h_diffs * pred_edge_h_range_masks.float())
			edge_h_range_costs = edge_h_range_costs.reshape(edge_h_range_costs.shape[0],-1)
			edge_h_transfer_costs = edge_h_transfer_loss_fn(pred_edge_h_diffs)
			edge_h_transfer_costs = edge_h_transfer_costs.reshape(edge_h_transfer_costs.shape[0],-1)
			edge_h_logprobs = pred_edge_h_logprobs.reshape(pred_edge_h_logprobs.shape[0],-1)
			edge_h_range_avg_cost = th.logsumexp(edge_h_logprobs + safelog(edge_h_range_costs),dim=1)
			edge_h_transfer_avg_cost = th.logsumexp(edge_h_logprobs + safelog(edge_h_transfer_costs),dim=1)
			h_range_loss = scatter_logsumexp(
				pred_edge_logprobs + edge_h_range_avg_cost,
				pred_edge_batch_idxs,
				dim=0
			)
			h_transfer_loss = scatter_logsumexp(
				pred_edge_logprobs + edge_h_transfer_avg_cost,
				pred_edge_batch_idxs,
				dim=0
			)
			return h_range_loss, h_transfer_loss
		
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
			pred_node_formula_logprobs,
			pred_null_formula_logprob,
			pred_edge_logprobs,
			pred_edge_h_diffs,
			pred_edge_h_range_masks,
			pred_edge_h_logprobs,
			pred_edge_batch_idxs,
			pred_oos_logprobs):
			spec_ce, true_oos_logprobs = sparse_ce_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				pred_oos_logprobs
			)
			spec_cd = cosine_distance_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs
			)
			if self.hparams.loss_type == "cross_entropy":
				loss = spec_ce
			else:
				assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
				loss = spec_cd
			loss = self.hparams.primary_loss_weight * loss
			spec_e, spec_ne = sparse_entropy_fn(
				pred_logprobs,
				pred_batch_idxs
			)
			formula_e, formula_ne = sparse_entropy_fn(
				pred_formula_logprobs,
				pred_formula_batch_idxs
			)
			loss = loss + self.hparams.formula_entropy_weight * formula_e \
				+ self.hparams.formula_normalized_entropy_weight * formula_ne
			node_e, node_ne = sparse_entropy_fn(
				pred_node_logprobs,
				pred_node_batch_idxs
			)
			loss = loss + self.hparams.node_entropy_weight * node_e \
				+ self.hparams.node_normalized_entropy_weight * node_ne
			
			node_formula_e, node_formula_ne = sparse_conditional_entropy_fn(
				pred_node_logprobs,
				pred_node_formula_logprobs,
				pred_node_batch_idxs
			)
			loss = loss + self.hparams.node_formula_entropy_weight * node_formula_e \
				+ self.hparams.node_formula_normalized_entropy_weight * node_formula_ne
			null_formula_prob = th.exp(pred_null_formula_logprob)
			oos_prob = th.exp(pred_oos_logprobs)
			true_oos_prob = th.exp(true_oos_logprobs)
			loss_d = {
				"loss": loss,
				"spec_cd": spec_cd,
				"spec_ce": spec_ce,
				"spec_e": spec_e,
				"spec_ne": spec_ne,
				"formula_e": formula_e,
				"formula_ne": formula_ne,
				"node_e": node_e,
				"node_ne": node_ne,
				"node_formula_e": node_formula_e,
				"node_formula_ne": node_formula_ne,
				"null_formula_prob": null_formula_prob,
				"oos_prob": oos_prob,
				"true_oos_prob": true_oos_prob
			}
			if not self.hparams.skip_edge_loss:
				r_el, t_el = edge_loss(
					pred_edge_logprobs,
					pred_edge_h_diffs,
					pred_edge_h_range_masks,
					pred_edge_h_logprobs,
					pred_edge_batch_idxs
				)
				loss = loss + self.hparams.edge_h_range_loss_weight * r_el \
					+ self.hparams.edge_h_transfer_loss_weight * t_el
				edge_e, edge_ne = sparse_entropy_fn(
					pred_edge_logprobs,
					pred_edge_batch_idxs
				)
				loss = loss + self.hparams.edge_entropy_weight * edge_e \
					+ self.hparams.edge_normalized_entropy_weight * edge_ne
				loss_d["loss"] = loss
				loss_d["edge_h_range_loss"] = r_el
				loss_d["edge_h_transfer_loss"] = t_el
				loss_d["edge_e"] = edge_e
				loss_d["edge_ne"] = edge_ne
			return loss_d
		self.loss_fn = _loss_fn
		# flag losses for tracking
		loss_names = [
			"loss",
			"spec_cd",
			"spec_ce",
			"spec_e",
			"spec_ne",
			"formula_e",
			"formula_ne",
			"node_e",
			"node_ne",
			"node_formula_e",
			"node_formula_ne",
			"null_formula_prob",
			"oos_prob",
			"true_oos_prob"
		]
		if not self.hparams.skip_edge_loss:
			loss_names.extend(
				[
					"edge_h_range_loss",
					"edge_h_transfer_loss",
					"edge_e",
					"edge_ne"
				]
			)
		self.metric_names.update(loss_names)

	def _manual_opt(self):
		""" run manual opt
		"""
		opt = self.optimizers()
		# scale gradients
		if self._smf:
			for pg in opt.param_groups:
				for p in pg['params']:
					if p.grad is not None:
						p.grad.data.mul_(self._num_acc_gradient / self._num_need_update)					
		self.on_before_optimizer_step(opt)
		# clip gradients
		self.clip_gradients(opt, gradient_clip_val=self.hparams.gradient_clip_val, gradient_clip_algorithm=self.hparams.gradient_clip_algorithm)
		# call opt
		opt.step()
		self.on_after_optimizer_step(opt)
		# clean gradient
		opt.zero_grad()

	def training_step(self, batch, batch_idx):
		""" training loop for FragGNNPL

		Args:
			batch (_type_): _description_
			batch_idx (_type_): _description_

		Raises:
			NotImplementedError: _description_

		Returns:
			_type_: _description_
		"""

		if self.hparams.automatic_optimization:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results.pop("mean_loss")
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self._num_need_update += self.hparams.train_batch_size
			self._update_results(batch_results,"train")
			return mean_loss
		
		elif self._smf:
			if batch['batch_size'].item() == 0:
				return th.tensor(0.0)		
			batch_results = self._common_step(batch, split="train")
			batch_size = batch['batch_size'].item()
			total_loss = batch_results.pop("total_loss")
			mean_loss = total_loss / self._num_acc_gradient
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self.manual_backward(mean_loss)
			# accumulate gradients of N samples
			self._num_need_update += batch_size
			if self._num_need_update >= self._num_acc_gradient:
				self._manual_opt()
			self._update_results(batch_results,"train")
			return mean_loss
		
		else:
			batch_results = self._common_step(batch, split="train")
			mean_loss = batch_results.pop("mean_loss") / self.hparams.accumulate_grad_batches
			if self.hparams.debug_zero_loss:
				mean_loss = 0. * mean_loss
				self.print(mean_loss)
			self.manual_backward(mean_loss)
			# self.backward(mean_loss)
			# accumulate gradients of N samples
			self._num_need_update += self.hparams.train_batch_size
			if (batch_idx + 1) % self.hparams.accumulate_grad_batches == 0:
				self._manual_opt()
			self._update_results(batch_results,"train")
			return mean_loss

	def on_train_epoch_end(self):

		if not self.hparams.automatic_optimization and self._smf and self._num_need_update > 0:
			# this is used for manul opt when lr_schedulers is not available from torch lightning
			# TODO: add drop_last check?
			self._manual_opt()
		super().on_train_epoch_end()

	def optimizer_step(self, *args, **kwargs):
		super().optimizer_step(*args, **kwargs)
		opt = self.optimizers()
		self.on_after_optimizer_step(opt)

class NeimsPL(SpectrumPL):

	def _setup_model(self):

		self.model = NeimsModel(
			mol_fingerprint_morgan=self.hparams.mol_params["fingerprint_morgan"],
			mol_fingerprint_rdkit=self.hparams.mol_params["fingerprint_rdkit"],
			mol_fingerprint_maccs=self.hparams.mol_params["fingerprint_maccs"],
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
		)
		# compile
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)

	def _setup_loss_fn(self):

		# define loss
		assert self.hparams.loss_type in ["cosine_distance","cosine_distance_dense"], self.hparams.loss_type
		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			pred_specs,
			**kwargs):
			if self.hparams.loss_type == "cosine_distance":
				if self.hparams.sparse_cosine_similarity:
					cosine_similarity_fn = calculate_sparse_cosine_similarity
				else:
					cosine_similarity_fn = calculate_cosine_similarity
				spec_cd = 1.-cosine_similarity_fn(
					true_mzs,
					th.exp(true_logprobs),
					true_batch_idxs,
					pred_mzs,
					th.exp(pred_logprobs),
					pred_batch_idxs,
					mz_bin_res=self.hparams.mz_bin_res,
					mz_max=self.hparams.mz_max,
					sum_ints=self.hparams.sum_ints
				)
			else:
				# _pred_specs = self.bin_func(pred_mzs,th.exp(pred_logprobs),pred_batch_idxs)
				true_specs = self.bin_func(true_mzs,th.exp(true_logprobs),true_batch_idxs)
				spec_cd = 1.-binned_cosine_similarity(true_specs,pred_specs)
			loss = spec_cd
			loss_d = {
				"loss": loss,
				"spec_cd": spec_cd
			}
			return loss_d
		self.loss_fn = _loss_fn
		# flag metrics for tracking
		loss_names = [
			"loss",
			"spec_cd"
		]
		self.metric_names.update(loss_names)


def test_fraggnn():

	from frag_gnn.runner import load_config
	from frag_gnn.dataset import SpecMolFragDataset
	from frag_gnn.utils.misc_utils import print_shapes
	from frag_gnn.utils.spec_utils import bin_func, batched_bin_func, binned_cosine_similarity
	import torch.nn.functional as F
	from frag_gnn.utils.misc_utils import to_device

	template_fp = "config/template.yml"
	custom_fps = [
		"config/debug/debug_d3.yml",
	]
	
	for custom_fp in custom_fps:

		print(custom_fp)

		config_d = load_config(template_fp, custom_fp)
		device = "cuda:0" if config_d["accelerator"] == "gpu" else "cpu"

		ds = SpecMolFragDataset(split="train", **config_d)
		dl = th.utils.data.DataLoader(
			ds,
			batch_size=8,
			shuffle=False,
			num_workers=0,
			collate_fn=ds.get_collate_fn(),
		)
		dl_iter = iter(dl)
		batch = next(dl_iter)
		print("> batch")
		print_shapes(batch)
		batch = to_device(batch,device)

		pl_model = FragGNNPL(**config_d)
		pl_model.train()
		pl_model.to(device)

		outputs = pl_model.forward(**batch)
		print("> train outputs")
		print_shapes(outputs)
		print(th.exp(outputs["pred_logprobs"]).sum())

		results = pl_model._common_step(batch, split="train")
		print_shapes(results)

		mean_loss = results.pop("mean_loss")
		print("> loss")
		print(mean_loss)
		mean_loss.backward()


if __name__ == "__main__":

	from frag_gnn.utils.misc_utils import th_temp_seed

	with th_temp_seed(420):
		test_fraggnn()
	# with th_temp_seed(420):
	# 	test_neims()