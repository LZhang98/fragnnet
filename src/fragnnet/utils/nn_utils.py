import copy
import math
import torch as th
import torch.nn as nn
import torch_geometric as pyg
import copy
from fragnnet.model.nn_blocks import ZerosAggregation


def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_lr_scheduler(
	optimizer, decay_rate: float, decay_steps: int = 5000, warmup_steps: int = 1000
):
	"""build_lr_scheduler.

	Args:
		optimizer:
		decay_rate (float): decay_rate
		decay_steps (int): decay_steps
		warmup_steps (int): warmup_steps
	"""

	def lr_lambda(step):
		if step >= warmup_steps:
			# Adjust
			step = step - warmup_steps
			rate = decay_rate ** (step // decay_steps)
		else:
			rate = 1 - math.exp(-step / warmup_steps)
		return rate

	scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
	return scheduler

def build_pool_module(pool_type: str, node_dim: int, num_step_set2set:int=3):
	"""medthos build pooling layer for given pool_type and dim
		node dim is required for attention layer. 
		if pool type is not one of 'sum','mean','max','attention', return ZerosAggregation
	Args:
		pool_type (str): pooling type in ['sum','mean','max','attention']
		node_dim (int): node_dim, used for attention

	Returns:
		_type_: _description_
	"""

	if pool_type == "sum":
		pool_module = pyg.nn.SumAggregation()
	elif pool_type == "mean":
		pool_module = pyg.nn.MeanAggregation()
	elif pool_type == "max":
		pool_module = pyg.nn.MaxAggregation()
	elif pool_type == "attention":
		pool_module = pyg.nn.AttentionalAggregation(
			gate_nn = nn.Sequential(
				nn.Linear(node_dim, node_dim),
				nn.ReLU(),
				nn.Linear(node_dim, 1)
			)
		)
	elif pool_type == "set2set":
		"With Set2Set for graph level prediction  (Order Matters: Sequence to sequence for sets) https://arxiv.org/abs/1511.06391"
		pool_module = nn.Sequential(
			pyg.nn.Set2Set(node_dim,processing_steps=num_step_set2set),
			nn.Linear(2 * node_dim, node_dim),
		)
	elif pool_type == "softmax":
		pool_module = pyg.nn.SoftmaxAggregation(learn=True)
	elif pool_type == "powermean":
		pool_module = pyg.nn.PowerMeanAggregation(learn=True)
	elif pool_type == "mean_std_softmax":
		raise NotImplementedError
		# TODO: this need hook up in the way that size of input need change respect this
		#pool_module = pyg.nn.MultiAggregation(['mean', 'std',  pyg.nn.SoftmaxAggregation(learn=True)])
	else:
		assert pool_type == "none", pool_type
		# just return 0s
		pool_module = ZerosAggregation()
	return pool_module

def get_norm(normalization,hidden_size):

	# norm_fn may or may not depend on the batch index
	if normalization == "batch":
		norm_mod = nn.BatchNorm1d(hidden_size)
		norm_fn = lambda x, b, n: n(x)
	elif normalization == "layer":
		norm_mod = nn.LayerNorm(hidden_size)
		norm_fn = lambda x, b, n: n(x)
	elif normalization == "graph":
		norm_mod = pyg.nn.GraphNorm(hidden_size)
		norm_fn = lambda x, b, n: n(x,b)
	else:
		assert normalization == "none", normalization
		norm_mod = nn.Identity()
		norm_fn = lambda x, b, n: n(x)	
	return norm_mod, norm_fn

def mask_prediction_by_mass(raw_prediction, prec_mass_idx, prec_mass_offset):
	# adapted from NEIMS
	# raw_prediction is [B,D], prec_mass_idx is [B]

	max_idx = raw_prediction.shape[1]
	assert th.all(prec_mass_idx < max_idx)
	idx = th.arange(max_idx, device=prec_mass_idx.device)
	mask = (
		idx.unsqueeze(0) <= (
			prec_mass_idx.unsqueeze(1) +
			prec_mass_offset)).float()
	return mask * raw_prediction


def reverse_prediction(raw_prediction, prec_mass_idx, prec_mass_offset):
	# adapted from NEIMS
	# raw_prediction is [B,D], prec_mass_idx is [B]

	batch_size = raw_prediction.shape[0]
	max_idx = raw_prediction.shape[1]
	assert th.all(prec_mass_idx < max_idx)
	rev_prediction = th.flip(raw_prediction, dims=(1,))
	# convention is to shift right, so we express as negative to go left
	offset_idx = th.minimum(
		max_idx * th.ones_like(prec_mass_idx),
		prec_mass_idx + prec_mass_offset + 1)
	shifts = - (max_idx - offset_idx)
	gather_idx = th.arange(
		max_idx,
		device=raw_prediction.device).unsqueeze(0).expand(
		batch_size,
		max_idx)
	gather_idx = (gather_idx - shifts.unsqueeze(1)) % max_idx
	offset_rev_prediction = th.gather(rev_prediction, 1, gather_idx)
	# you could mask_prediction_by_mass here but it's unnecessary
	return offset_rev_prediction


def nan_forward_hook(self, input, output):
	if isinstance(output, tuple):
		outputs = list(output)
	elif isinstance(output, dict):
		outputs = list(output.values())
	else:
		outputs = [output]
	for i, val in enumerate(outputs):
		nan_mask = th.isnan(val)
		if nan_mask.any():
			print(">> In", self.__class__.__name__)
			raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def nan_backward_hook(self, grad_input, grad_output):
	for i, val in enumerate(grad_input):
		if val is None:
			continue
		nan_mask = th.isnan(val)
		if nan_mask.any():
			print(">> In", self.__class__.__name__)
			raise RuntimeError(f"Found NAN in grad_input {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])
	for i, val in enumerate(grad_output):
		if val is None:
			continue
		nan_mask = th.isnan(val)
		if nan_mask.any():
			print(">> In", self.__class__.__name__)
			raise RuntimeError(f"Found NAN in grad_output {i} at indices: ", nan_mask.nonzero(), "where:", val[nan_mask.nonzero()[:, 0].unique(sorted=True)])


def decompile_jit_ckpt(ckpt):
	"""
	convert compiled ckpt to normal version and return a patched checkpoint
	Note this is a in place change
	"""
 
	# inspired by
	# https://github.com/pytorch/pytorch/issues/101107#issuecomment-1801128683
	#print(f"> src: {path}")

	if ckpt['hyper_parameters']['compile'] == False:
		return ckpt

	in_state_dict = ckpt["state_dict"]
	pairings = [
		(src_key,  src_key.replace('._orig_mod',''))
		for src_key in in_state_dict.keys()
	]
	#if all(src_key == dest_key for src_key, dest_key in pairings):
	#	return  # Do not write checkpoint if no need to repair!
	out_state_dict = {}
	for src_key, dest_key in pairings:
		if src_key != dest_key:
			print(f"{src_key}  ==>  {dest_key}")
		out_state_dict[dest_key] = in_state_dict[src_key]

	ckpt["state_dict"] = out_state_dict
	ckpt['hyper_parameters']['compile'] = False
	return ckpt

def is_ckpt_compiled(ckpt) -> bool:
	state_dict = ckpt["state_dict"]
	for key in state_dict.keys():
		if '_orig_mod' in key:
			return True
	return False

def get_pl_hparams(ckpt) -> dict:
	"""return a copied version of hyper_parameters

	Args:
		ckpt (_type_): _description_

	Returns:
		dict: _description_
	"""
	hyper_parameters = copy.deepcopy(ckpt["hyper_parameters"])
	return hyper_parameters