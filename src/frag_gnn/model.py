import torch as th
import numpy as np
import lightning.pytorch as pl
from torch.profiler import profile, record_function, ProfilerActivity
import torch._dynamo as th_dynamo

from frag_gnn.nn.nn_utils import build_lr_scheduler, JointPredictor
from frag_gnn.nn.loss import get_sparse_cross_entropy_fn, get_sparse_entropy_fn
from frag_gnn.utils.spec_utils import batched_filter_func, batched_bin_func, calculate_cosine_similarity, get_ints_transform_func, batched_l1normalize, calculate_weighted_recall
from frag_gnn.utils.misc_utils import safelog
from frag_gnn.frag.frag_utils import NUM_ELEMENTS
from frag_gnn.utils.spec_utils import MZ_MAX, MZ_BIN_RES, INTS_THRESH


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
		mol_pool_type: str,
		frag_pyg_node_feats: list[str],
		frag_pyg_edge_feats: list[str],
		frag_hidden_size: int,
		frag_num_layers: int,
		frag_gnn_type: str,
		frag_dropout: float,
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
			mol_pool_type=mol_pool_type,
			frag_node_feats=frag_pyg_node_feats,
			frag_edge_feats=frag_pyg_edge_feats,
			frag_hidden_size=frag_hidden_size,
			frag_num_layers=frag_num_layers,
			frag_gnn_type=frag_gnn_type,
			frag_dropout=frag_dropout,
			frag_pool_type=frag_pool_type,
			mlp_hidden_size=mlp_hidden_size,
			mlp_dropout=mlp_dropout,
			mlp_num_layers=mlp_num_layers,
			mlp_use_residuals=mlp_use_residuals
		)
		if self.hparams.compile:
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model.compile_submodules(backend=self.dynamo_prof,dynamic=True)

		self._setup_loss_fn()
		def _filter_func(mzs, ints, batch_idxs):
			return batched_filter_func(mzs,ints,batch_idxs,ints_thresh,mz_max)
		self.filter_func = _filter_func
		def _bin_func(mzs, ints, batch_idxs):
			return batched_bin_func(mzs,ints,batch_idxs,mz_bin_res,bin_sum_ints)
		self.bin_func = _bin_func
		self.ints_transform_func = get_ints_transform_func(ints_transform)
		self.normalize_func = batched_l1normalize
		# set tolerance
		if self.hparams.loss_tolerance_ppm is not None:
			self.tolerance = self.hparams.loss_tolerance_ppm
			self.relative = True
		else:
			assert self.hparams.loss_tolerance_abs is not None
			self.tolerance = self.hparams.loss_tolerance_abs
			self.relative = False
		self.auxiliary_score_fns = {}
		if "cos_sim" in auxiliary_scores:
			def _cos_sim(*args,**kwargs):
				return th.mean(calculate_cosine_similarity(*args,**kwargs),dim=0)
			self.auxiliary_score_fns["cos_sim"] = _cos_sim
		if "wrecall" in auxiliary_scores:
			def _wrecall(*args,**kwargs):
				return th.mean(
					calculate_weighted_recall(
						*args,
						tolerance=self.tolerance,
						relative=self.relative,
						**kwargs
					), dim=0
				)
			self.auxiliary_score_fns["wrecall"] = _wrecall
		self.automatic_optimization = automatic_optimization

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
			pred_node_batch_idxs):
			spec_ce = sparse_ce(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs
			)
			loss = spec_ce
			# if self.hparams.formula_entropy_weight != 0.:
			formula_e = sparse_entropy(
				pred_formula_logprobs,
				pred_formula_batch_idxs
			)
			loss = loss + self.hparams.formula_entropy_weight * formula_e
			# if self.hparams.node_entropy_weight != 0.:
			node_e = sparse_entropy(
				pred_node_logprobs,
				pred_node_batch_idxs
			)
			loss = loss + self.hparams.node_entropy_weight * node_e
			return loss
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
		spec_ints = self.normalize_func(
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

	def _common_step(self, batch, name="train"):

		# preprocess spec
		spec_mzs = batch["spec_mzs"]
		spec_ints = batch["spec_ints"]
		spec_batch_idxs = batch["spec_batch_idxs"]
		batch_size = batch["batch_size"]
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
		# with profile(
		# 		activities=[ProfilerActivity.CUDA],
		# 		profile_memory=True
		# 	) as prof:
		# 	with record_function("loss"):
		# 		loss = self.loss_fn(**both)
		# print(prof.key_averages().table(sort_by="cuda_time_total"))
		loss = self.loss_fn(**both)
		self.log(
			f"{name}_loss",
			loss.item(),
			batch_size=batch_size,
			on_epoch=True
		)
		assert spec_mzs.shape[0] > 0, spec_mzs.shape[0]
		for fn_name,fn in self.auxiliary_score_fns.items():
			score = fn(
				spec_mzs,
				th.exp(spec_ints),
				spec_batch_idxs,
				pred["pred_mzs"],
				th.exp(pred["pred_logprobs"]),
				pred["pred_batch_idxs"]
			)
			self.log(
				f"{name}_{fn_name}",
				score,
				batch_size=batch_size,
				on_epoch=True
			)
		return loss

	def training_step(self, batch, batch_idx):

		if self.hparams.automatic_optimization:
			return self._common_step(batch, name="train")
		else:
			raise NotImplementedError("Manual optimization not implemented")
			# opt = self.optimizers()
			# opt.zero_grad()
			# loss = self._common_step(batch, name="train")
			# self.manual_backward(loss)
			# opt.step()
			# return loss

	def validation_step(self, batch, batch_idx):

		return self._common_step(batch, name="val")

	def test_step(self, batch, batch_idx):

		return self._common_step(batch, name="test")

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


def get_default_model_params():

	default_model_params = {
		"num_depth": 5, # max_depth + 1
		"num_hs": 4,
		"num_elements": NUM_ELEMENTS,
		"int_embedder": "abs-sines",
		"mol_node_feats": [
			"a_onehot",
			"a_degree",
			"a_hybrid",
			"a_formal",
			"a_radical",
			"a_ring",
			"a_mass",
			"a_chiral",
		],
		"mol_edge_feats": ["b_degree"],
		"mol_pe_embed_k": 0,
		"mol_hidden_size": 256,
		"mol_num_layers": 4,
		"mol_gnn_type": "GINE",
		"mol_dropout": 0.0,
		"frag_node_feats": ["cc","base_formula","depth"],
		"frag_edge_feats": ["cc","base_formula"],
		"frag_hidden_size": 256,
		"frag_num_layers": 4,
		"frag_gnn_type": "GINE",
		"frag_dropout": 0.0,
		"mlp_hidden_size": 256,
		"mlp_dropout": 0.0,
		"mlp_num_layers": 2,
		"mlp_use_residuals": True,
		"lr": 1e-4,
		"lr_schedule": False,
		"weight_decay": 0.0,
		"lr_decay_rate": 0.95,
		"warmup": 1000,
		"optimizer": "adamw",
		"output_distribution": "gaussian",
		"loss_vectorized": False,
		"loss_tolerance_ppm": 10,
		"loss_tolerance_abs": None, #0.001,
		"auxiliary_scores": ["cos_sim"],
		"ints_thresh": INTS_THRESH,
		"mz_max": MZ_MAX,
		"mz_bin_res": MZ_BIN_RES,
		"bin_sum_ints": True,
		"ints_transform": "log10t3" # none
	}

	return default_model_params
