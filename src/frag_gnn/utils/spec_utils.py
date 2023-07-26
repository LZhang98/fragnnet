import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import  AllChem
import torch as th
import torch.nn.functional as F

from frag_gnn.utils.formula_utils import ELECTRON_MASS, PREC_TYPE_TO_MASS_DIFF
from frag_gnn.utils.misc_utils import scatter_logsoftmax, scatter_l1normalize, scatter_reduce


# TODO: actually fill these out
PREC_TYPES = [
	"[M+H]+",
	"[M-H]-"
]
PREC_TYPE_TO_IDX = {prec_type: idx for idx,prec_type in enumerate(PREC_TYPES)}
NCE_MEAN = 100.
NCE_STD = 50.
MZ_MAX = 1500.0
MZ_BIN_RES = 0.01
INTS_THRESH = 0.0

def prec_type_idx(prec_type):
	idx = PREC_TYPE_TO_IDX[prec_type]
	return idx

def prec_type_mass_diff(prec_type):
	mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
	return mass_diff

def transform_nce(nce):
	return (nce-NCE_MEAN)/NCE_STD	

def filter_func(mzs,ints,ints_thresh,mz_max):

	thresh_mask = ints > ints_thresh
	max_mask = mzs < mz_max
	both_mask = thresh_mask & max_mask
	mzs = mzs[both_mask]
	ints = ints[both_mask]
	return mzs, ints

def bin_func(mzs,ints,mz_max,mz_bin_res,return_index,sum_ints):
	"""
	Note: if return_index is True, returns the (possibly non-unique) bin index for each mz
	Note: intensities may not be normalized due to peak merging
	"""

	assert th.max(mzs) < mz_max, (th.max(mzs),mz_max)
	# bin
	bins = th.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res,device=mzs.device,dtype=mzs.dtype)
	bin_idx = th.searchsorted(bins,mzs,side="right")
	if return_index:
		return bin_idx
	else:
		bin_spec = scatter_reduce(
			src=ints,
			index=bin_idx,
			reduce="sum" if sum_ints else "max",
			dim=0,
			dim_size=bins.shape[0]
		)
		if th.all(bin_spec == 0.):
			print("> warning: bin_spec is all zeros!")
			bin_spec[-1] = 1.
		return bin_spec

def batch_func(mzs_list,ints_list):

	assert len(mzs_list) == len(ints_list)
	batch_size = len(mzs_list)
	batch_idxs = th.arange(batch_size)
	spec_sizes = th.tensor([mzs.shape[0] for mzs in mzs_list])
	batch_idxs = th.repeat_interleave(batch_idxs,spec_sizes)
	mzs = th.cat(mzs_list,dim=0)
	ints = th.cat(ints_list,dim=0)
	return mzs, ints, batch_idxs


def batched_filter_func(mzs,ints,batch_idxs,ints_thresh,mz_max):

	thresh_mask = ints > ints_thresh
	max_mask = mzs < mz_max
	both_mask = thresh_mask & max_mask
	mzs = mzs[both_mask]
	ints = ints[both_mask]
	batch_idxs = batch_idxs[both_mask]
	return mzs, ints, batch_idxs


def batched_bin_func(mzs,ints,batch_idxs,mz_max,mz_bin_res,sum_ints):

	if mzs.shape[0] == 0:
		import pdb; pdb.set_trace()
	assert th.max(mzs) < mz_max, (th.max(mzs),mz_max)
	batch_size = th.max(batch_idxs)+1
	bins = th.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res,device=mzs.device,dtype=mzs.dtype)
	num_bins = bins.shape[0]
	bin_idx = th.searchsorted(bins,mzs,side="right")
	bin_offsets = (th.arange(batch_size,device=mzs.device)*num_bins)[batch_idxs]
	bin_idx = bin_idx + bin_offsets
	bin_spec = scatter_reduce(
		src=ints,
		index=bin_idx,
		reduce="sum" if sum_ints else "max",
		dim=0,
		dim_size=num_bins*batch_size
	)
	bin_spec = bin_spec.reshape(batch_size,num_bins)
	if th.any(th.all(bin_spec==0.,dim=1)):
		print("> warning: bin_spec is all zeros!")
		mask = th.zeros_like(bin_spec,dtype=th.bool)
		mask[:,-1] = 1.
		mask = mask*th.all(bin_spec==0.,dim=1,keepdim=True)
		bin_spec[mask] = 1.
	return bin_spec


def merge_sparse_specs(*peakses,renormalize=False):
	"""
	this will result in peaks that are really close in mass (<5ppm)
	they are probably the same peak, but our model can handle this type of ambiguity
	for now, let's keep them unmerged
	TODO: this will result in inconsistent comparison with ICEBERG/SCARF
	"""

	merged_peaks = {}
	# total_intensity = 0.
	for peaks in peakses:
		for mz, intensity in peaks:
			if mz in merged_peaks:
				# TODO: sum or max?
				merged_peaks[mz] += intensity
			else:
				merged_peaks[mz] = intensity
	merged_peaks = sorted(list(merged_peaks.items()), key=lambda x: x[0])
	if renormalize:
		total_intensity = sum([intensity for mz, intensity in merged_peaks])
		merged_peaks = [(mz, intensity/total_intensity) for mz, intensity in merged_peaks]
	return merged_peaks


def calculate_spectrum_entropy(log_ints,batch_idxs):

	k = th.max(batch_idxs)+1
	log_norm_ints = scatter_logsoftmax(log_ints,batch_idxs,dim=0)
	entropy = -scatter_reduce(
		src=th.exp(log_norm_ints)*log_norm_ints,
		index=batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=k
	)
	return entropy


def calculate_topk_weighted_recall(true_mzs,true_logits,true_batch_idxs,pred_mzs,pred_logits,pred_batch_idxs,tolerance=1e-5,k=10):
	"""
	topk_weighted_recall: weighted fraction of top-k true peaks that are matched by at least one top-k true peak
	topk_frac: fraction of true intensity that is explained by at least one top-k true peaks
	topk_frac is an upper bound on topk_weighted_recall
	Note: this function could be parallelized with vmap
	"""

	batch_size = th.max(true_batch_idxs)+1
	topk_weighted_recall = th.zeros([batch_size],device=true_mzs.device)
	topk_frac = th.zeros([batch_size],device=true_mzs.device)
	true_ints = th.exp(scatter_logsoftmax(true_logits,true_batch_idxs,dim=0))
	pred_ints = th.exp(scatter_logsoftmax(pred_logits,pred_batch_idxs,dim=0))
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		if b_true_mask.sum() < k:
			b_true_argtopk = th.arange(b_true_mask.sum(),device=b_true_mask.device)
		else:
			b_true_argtopk = th.topk(true_ints[b_true_mask],k,dim=0,largest=True)[1]
		if b_pred_mask.sum() < k:
			b_pred_argtopk = th.arange(b_pred_mask.sum(),device=b_pred_mask.device)
		else:
			b_pred_argtopk = th.topk(pred_ints[b_pred_mask],k,dim=0,largest=True)[1]
		b_true_topk_mzs = true_mzs[b_true_mask][b_true_argtopk]
		b_pred_topk_mzs = pred_mzs[b_pred_mask][b_pred_argtopk]
		b_true_topk_match = calculate_match_mzs(
			b_true_topk_mzs,
			b_pred_topk_mzs,
			tolerance=tolerance,
			relative=True
		)
		b_true_topk_match = th.any(b_true_topk_match,dim=1)
		b_true_ints_topk = true_ints[b_true_mask][b_true_argtopk]
		b_true_ints_topk_sum = th.sum(b_true_ints_topk)
		b_true_ints_topk_match_sum = th.sum(b_true_ints_topk[b_true_topk_match])
		b_topk_weighted_recall = b_true_ints_topk_match_sum / b_true_ints_topk_sum
		topk_weighted_recall[b] = topk_weighted_recall[b] + b_topk_weighted_recall
		b_true_ints_sum = th.sum(true_ints[b_true_mask])
		b_topk_frac = b_true_ints_topk_sum / b_true_ints_sum
		topk_frac[b] = topk_frac[b] + b_topk_frac
	return topk_weighted_recall, topk_frac


def calculate_recalls(true_mzs,true_logits,true_batch_idxs,pred_mzs,pred_logits,pred_batch_idxs,tolerance=1e-5):
	"""
	Note: this function could be parallelized with vmap
	"""

	batch_size = th.max(true_batch_idxs)+1
	recall = th.zeros([batch_size],device=true_mzs.device)
	weighted_recall = th.zeros([batch_size],device=true_mzs.device)
	true_ints = th.exp(scatter_logsoftmax(true_logits,true_batch_idxs,dim=0))
	pred_ints = th.exp(scatter_logsoftmax(pred_logits,pred_batch_idxs,dim=0))
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
		b_true_match = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=True
		)
		b_true_match = th.any(b_true_match,dim=1)
		b_recall = th.sum(b_true_match)/b_true_mzs.shape[0]
		recall[b] = recall[b] + b_recall
		b_weighted_recall = th.sum(b_true_ints[b_true_match])/th.sum(b_true_ints)
		weighted_recall[b] = weighted_recall[b] + b_weighted_recall
	return recall, weighted_recall


def calculate_match_mzs(
		true_mzs,
		pred_mzs,
		tolerance=1e-5,
		relative=True,
		divisor_mz_floor=200
	):
	"""
	works with numpy arrays or torch tensors
	inspired by the function in ms-pred
	NOT BATCHED
	"""

	if isinstance(true_mzs,th.Tensor):
		assert isinstance(pred_mzs,th.Tensor)
		abs_func = th.abs
		copy_func = th.clone
	else:
		assert isinstance(true_mzs,np.ndarray)
		assert isinstance(pred_mzs,np.ndarray)
		abs_func = np.abs
		copy_func = np.copy
	diff_mzs = abs_func(true_mzs.reshape(-1,1)-pred_mzs.reshape(1,-1))
	min_diff_mzs = tolerance
	if relative:
		divisor_mzs = copy_func(true_mzs)
		divisor_mzs[divisor_mzs<divisor_mz_floor] = divisor_mz_floor
		diff_mzs = (diff_mzs / divisor_mzs.reshape(-1,1))
	match_mzs = diff_mzs < min_diff_mzs
	return match_mzs


def calculate_cosine_similarity(
		true_mzs,
		true_ints,
		true_batch_idxs,
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_bin_res=0.01,
		mz_max=1500.,
		sum_ints=True
	):

	# true_mzs, true_ints, true_batch_idxs = batched_filter_func(true_mzs,true_ints,true_batch_idxs,ints_thresh,mz_max)
	true_spec = batched_bin_func(
		true_mzs,
		true_ints,
		true_batch_idxs,
		mz_max,
		mz_bin_res,
		sum_ints
	)
	# pred_mzs, pred_ints, pred_batch_idxs = batched_filter_func(pred_mzs,pred_ints,pred_batch_idxs,ints_thresh,mz_max)
	pred_spec = batched_bin_func(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max,
		mz_bin_res,
		sum_ints
	)
	cos_sims = binned_cosine_similarity(true_spec,pred_spec)
	#print(f"cos_sims {cos_sims}")
	return cos_sims


def calculate_weighted_recall(
		true_mzs,
		true_ints,
		true_batch_idxs,
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		tolerance=1e-5,
		relative=True
	):

	batch_size = th.max(true_batch_idxs)+1
	weighted_recall = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_true_match = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative
		)
		b_true_match = th.any(b_true_match,dim=1)
		b_weighted_recall = th.sum(b_true_ints[b_true_match])/th.sum(b_true_ints)
		weighted_recall[b] = b_weighted_recall
	return weighted_recall


def binned_cosine_similarity(true_spec,pred_spec):

	true_spec = F.normalize(true_spec,p=2,dim=1)
	pred_spec = F.normalize(pred_spec,p=2,dim=1)
	cos_sims = (true_spec.unsqueeze(1) @ pred_spec.unsqueeze(2)).squeeze(2).squeeze(1)
	return cos_sims


def get_ints_transform_func(ints_transform):

	_func = None
	if ints_transform == "log10":
		_func = lambda ints: th.log10(ints+1.)
	elif ints_transform == "log10t3":
		_func = lambda ints: th.log10(ints/3.+1.)
	elif ints_transform == "loge":
		_func = lambda ints: th.log(ints+1.)
	elif ints_transform == "sqrt":
		_func = lambda ints: th.sqrt(ints)
	elif ints_transform == "none":
		_func = lambda ints: ints
	else:
		raise ValueError("Invalid ints_transform: {}".format(ints_transform))
	return _func


def get_ints_untransform_func(ints_transform):

	_func = None
	if ints_transform == "log10":
		_func = lambda ints: 10**ints-1.
	elif ints_transform == "log10t3":
		_func = lambda ints: (10**ints-1.)*3.
	elif ints_transform == "loge":
		_func = lambda ints: th.exp(ints)-1.
	elif ints_transform == "sqrt":
		_func = lambda ints: ints**2
	elif ints_transform == "none":
		_func = lambda ints: ints
	else:
		raise ValueError("Invalid ints_transform: {}".format(ints_transform))
	return _func


def batched_l1normalize(ints, batch_idxs):

	ints = scatter_l1normalize(
		ints,
		batch_idxs,
		dim=0
	)
	return ints
