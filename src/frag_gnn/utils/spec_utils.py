import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import torch as th
import torch.nn.functional as F
import scipy
import scipy.optimize

from frag_gnn.utils.formula_utils import ELECTRON_MASS, PREC_TYPE_TO_MASS_DIFF
from frag_gnn.utils.misc_utils import scatter_logsoftmax, scatter_l1normalize, scatter_l2normalize, scatter_reduce, EPS, th_setdiff1d, safelog
from typing import List, Tuple, Union

# TODO: actually fill these out
PREC_TYPES = [
	"[M+H]+",
	"[M-H]-"
]
PREC_TYPE_TO_IDX = {prec_type: idx for idx,prec_type in enumerate(PREC_TYPES)}
# rough approximations from NIST20
NCE_MEAN = 60.
NCE_STD = 40.
NCE_MAX = 200.
MZ_MAX = 1500.0
MZ_BIN_RES = 0.01
INTS_THRESH = 0.0


def prec_type_idx(prec_type):
	idx = PREC_TYPE_TO_IDX[prec_type]
	return idx

def prec_type_mass_diff(prec_type):
	mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
	return mass_diff

def get_ints_transform_func(ints_transform):
	""" method to get int transferom func

	Args:
		ints_transform (_type_): _description_

	Raises:
		ValueError: _description_

	Returns:
		_type_: _description_
	"""

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
	"""

	Args:
		ints_transform (_type_): _description_

	Raises:
		ValueError: _description_

	Returns:
		_type_: _description_
	"""
	if ints_transform == "log10":
		max_ints = float(np.log10(1000. + 1.))
		def _untransform_fn(x): return 10**x - 1.
	elif ints_transform == "log10t3":
		max_ints = float(np.log10(1000. + 1.) / 3.)
		def _untransform_fn(x): return 10**(3 * x) - 1.
	elif ints_transform == "loge":
		max_ints = float(np.log(1000. + 1.))
		def _untransform_fn(x): return th.exp(x) - 1.
	elif ints_transform == "sqrt":
		max_ints = float(np.sqrt(1000.))
		def _untransform_fn(x): return x**2
	elif ints_transform == "none":
		max_ints = 1000.
		def _untransform_fn(x): return x
	else:
		raise ValueError("invalid transform")
	def _func(ints):
		ints = ints / (th.max(ints) + EPS) * max_ints
		ints = _untransform_fn(ints)
		ints = th.clamp(ints, min=0.)
		assert not th.isnan(ints).any()
		return ints
	return _func

def transform_nce(nce):
	return (nce-NCE_MEAN)/NCE_STD	

def transform_nce_to_ace(nce:float, mw:float, charge_factor:int = 1) -> float:
	"""get ace from given nce and mw
	   Absolute energy (eV) = (settling NCE) x (Isolation center) / (500 m/z) x (charge factor)
	Args:
		nce (float): normalized collision energy
		mw (float): Isolation center mw, most time it is the precursor ion mass

	Returns:
		float: absolute energy
	"""
	return nce  * mw  / 500 *  charge_factor 

def transform_ace(ace:float) -> float:
	""" ev most likely between 0 to 100, return normalized absolute collision energy for ml

	Args:
		ace (float): _description_

	Raises:
		NotImplementedError: _description_
	Returns:
		float: normalized absolute collision energy for ml
	"""
	# TODO fix this
	raise NotImplementedError
	#return ace/100 	

def filter_func(mzs: Union[th.Tensor,np.ndarray], 
				ints:Union[th.Tensor,np.ndarray], 
				ints_thresh:float, mz_max:float) -> Tuple[Union[th.Tensor,np.ndarray], Union[th.Tensor,np.ndarray]]:
	"""filter spectrum by intesnity value and max mz

	Args:
		mzs (Union[th.Tensor,np.ndarray]): m/z s
		ints (Union[th.Tensor,np.ndarray]): intesnities 
		ints_thresh (float): intesnity thresh hold
		mz_max (float): max mz, if mz_max <= 0, mz_max filter will be ignored

	Returns:
		Tuple[Union[th.Tensor,np.ndarray], Union[th.Tensor,np.ndarray]]: mzs, ints
	"""
	thresh_mask = ints > ints_thresh
	
	if mz_max > 0:
		max_mask = mzs < mz_max
		both_mask = thresh_mask & max_mask
	else:
		both_mask = thresh_mask

	mzs = mzs[both_mask]
	ints = ints[both_mask]
	return mzs, ints

def bin_func(mzs:th.Tensor,ints:th.Tensor,mz_max:float,mz_bin_res:float,return_index:bool,sum_ints:bool):
	"""
	return binned spectra
	Note: if return_index is True, returns the (possibly non-unique) bin index for each mz
	Note: intensities may not be normalized due to peak merging
	Args:
		mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.  
		ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.  
		mz_max (float): max mz value allowed
		mz_bin_res (float): bin size
		return_index (bool): if return_index is True, returns the (possibly non-unique) bin index for each mz
		sum_ints (bool): flag for sum intensities within the bin, else take max
	Returns:
		_type_: _description_
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
			reduce="sum" if sum_ints else "amax",
			dim=0,
			dim_size=bins.shape[0]
		)
		if th.all(bin_spec == 0.):
			print("> warning: bin_spec is all zeros!")
			bin_spec[-1] = 1.
		return bin_spec

def batch_func(mzs_list:Union[th.Tensor,np.ndarray],ints_list:Union[th.Tensor,np.ndarray]) -> Tuple[th.Tensor,th.Tensor,th.Tensor]:
	""" create a batch for given list of peak list, batch size is the size of peak list

	Args:
		mzs_list (th.Tensor): list of m/z list
		ints_list (th.Tensor): list of intensity list

	Returns:
		Tuple[th.Tensor,th.Tensor,th.Tensor]: mz tensoer, intensity tensor, batch_idxs tensor
	"""

	assert len(mzs_list) == len(ints_list)
	batch_size = len(mzs_list)
	batch_idxs = th.arange(batch_size)
	spec_sizes = th.tensor([mzs.shape[0] for mzs in mzs_list])
	batch_idxs = th.repeat_interleave(batch_idxs,spec_sizes)
	mzs = th.cat(mzs_list,dim=0)
	ints = th.cat(ints_list,dim=0)
	return mzs, ints, batch_idxs


def batched_filter_func(mzs:th.Tensor,ints:th.Tensor,batch_idxs:th.Tensor,ints_thresh:float,mz_max: float) -> Tuple[th.Tensor,th.Tensor,th.Tensor]:
	"""bacthed filter func for filter spectra

	Args:
		mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.  
		ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.  
		batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor. 
		ints_thresh (float): intesnity thresh hold
		mz_max (float): max m/z, if mz_max <= 0, mz_max filter will be ignored

	Returns:
		Tuple[th.Tensor,th.Tensor,th.Tensor]: mz tensoer, intensity tensor, batch_idxs tensor
	"""

	thresh_mask = ints > ints_thresh
	if mz_max > 0:
		max_mask = mzs < mz_max
		both_mask = thresh_mask & max_mask
	else:
		both_mask = thresh_mask
	mzs = mzs[both_mask]
	ints = ints[both_mask]
	batch_idxs = batch_idxs[both_mask]
	return mzs, ints, batch_idxs


def batched_bin_func(
		mzs:th.Tensor,
		ints:th.Tensor,
		batch_idxs:th.Tensor,
		mz_max:float,
		mz_bin_res:float,
		sum_ints:bool,
		sparse:bool=False,
		remove_prec_peaks:bool=False,
		prec_mzs:th.Tensor=None) -> th.Tensor:
	"""method to get binned spectra for batch

	Args:
		mzs (th.Tensor): 1d flat tensor of m/zs; multiple m/z lists are concatenated.  
		ints (th.Tensor): 1d flat tensor of intensities; multiple intensities lists are concatenated.  
		batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor. 
		mz_max (float): max mz value allowed
		mz_bin_res (float): bin size
		sum_ints (bool): flag for sum intensities within the bin, else take max
		sparse (bool, optional): flag to use sparse  method. Defaults to False.

	Returns:
		_type_: binned spectra
	"""

	if mzs.shape[0] == 0:
		import pdb; pdb.set_trace()
	assert th.max(mzs) < mz_max, (th.max(mzs),mz_max)
	batch_size = th.max(batch_idxs)+1
	bins = th.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res,device=mzs.device,dtype=mzs.dtype)
	num_bins = bins.shape[0]
	bin_idxs = th.searchsorted(bins,mzs,side="right")
	bin_offsets = (th.arange(batch_size,device=mzs.device)*num_bins)[batch_idxs]
	bin_idxs = bin_idxs + bin_offsets
	if remove_prec_peaks:
		assert prec_mzs is not None
		assert th.max(prec_mzs) <  mz_max, (th.max(prec_mzs),mz_max)
		prec_mz_bin_idxs = th.searchsorted(bins,prec_mzs,side="right")
		prec_mz_bin_offsets = th.arange(batch_size,device=mzs.device)*num_bins
		prec_mz_bin_idxs = prec_mz_bin_idxs + prec_mz_bin_offsets
		prec_ints_mask = th.isin(bin_idxs,prec_mz_bin_idxs)
		ints = ints*(1-prec_ints_mask.float())
	if sparse:
		un_bin_idxs, un_bin_idxs_rev = th.unique(bin_idxs,return_inverse=True)
		new_bin_idxs = th.arange(un_bin_idxs.shape[0],device=un_bin_idxs.device)
		un_bin_ints = scatter_reduce(
			src=ints,
			index=new_bin_idxs[un_bin_idxs_rev],
			reduce="sum" if sum_ints else "amax",
			dim=0,
			dim_size=new_bin_idxs.shape[0]
		)
		un_bin_batch_idxs = un_bin_idxs // num_bins
		return un_bin_idxs, un_bin_ints, un_bin_batch_idxs
	else:
		bin_spec = scatter_reduce(
			src=ints,
			index=bin_idxs,
			reduce="sum" if sum_ints else "amax",
			dim=0,
			dim_size=num_bins*batch_size
		)
		bin_spec = bin_spec.reshape(batch_size,num_bins)
		if th.any(th.all(bin_spec==0.,dim=1)):
			print("> warning: bin_spec is all zeros!")
			mask = th.zeros_like(bin_spec,dtype=th.bool)
			mask[:,0] = 1.
			mask = mask*th.all(bin_spec==0.,dim=1,keepdim=True)
			bin_spec = bin_spec + mask.float()
		return bin_spec


def merge_sparse_specs(*peakses,renormalize:bool=False,sum_ints:bool=True) -> th.Tensor:
	"""
	this will result in peaks that are really close in mass (<5ppm)
	they are probably the same peak, but our model can handle this type of ambiguity
	for now, let's keep them unmerged

	Args:
		renormalize (bool, optional): flag to get renormalize spectra. Defaults to False.
		sum_ints (bool, optional): flag for sum intensities within the bin, else take max

	Returns:
		_type_: _description_
	"""
	merged_peaks = {}
	# total_intensity = 0.
	for peaks in peakses:
		for mz, intensity in peaks:
			if mz in merged_peaks:
				if sum_ints:
					merged_peaks[mz] += intensity
				else:
					merged_peaks[mz] = max(merged_peaks[mz],intensity)
			else:
				merged_peaks[mz] = intensity
	merged_peaks = sorted(list(merged_peaks.items()), key=lambda x: x[0])
	if renormalize:
		total_intensity = sum([intensity for mz, intensity in merged_peaks])
		merged_peaks = [(mz, intensity/total_intensity) for mz, intensity in merged_peaks]
	return merged_peaks


def calculate_spectrum_entropy(log_ints:th.Tensor,batch_idxs:th.Tensor) -> th.Tensor:
	""" method to compute entropy.
		NOTE: this is NOT same spectra entropy in this https://www.nature.com/articles/s41592-023-02012-9 a
		nd https://www.nature.com/articles/s41592-021-01331-z
	Args:
		log_ints (th.Tensor): intensity in log scale
		batch_idxs (th.Tensor): batch idx

	Returns:
		th.Tensor: spectrum entropy
	"""

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
	Note: this function is not used, and need fix
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

def calculate_match_mzs(
		true_mzs:th.Tensor,
		pred_mzs:th.Tensor,
		tolerance:float=1e-5,
		relative:bool=True,
		divisor_mz_floor:float=200
	) -> th.Tensor:
	"""
	Method to match two spectra based on m/z, return a N x M matrix, where N is number of mz in pred_mzs, M is number of mz in pred_mzs
	Each cellm i,j means if pred_mzs[i] matches true_mzs[j]
	works with numpy arrays or torch tensors inspired by the function in ms-pred
	NOT BATCHED
	Args:
		true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.  
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.  
		tolerance (float, optional): Tolerance; Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): Flag to use a relative measure. Defaults to True.
		divisor_mz_floor (float, optional): Divisor floor used in relative measure. Defaults to 200 Da.

	Returns:
		th.Tensor: return a N x M matrix of True or False, dim 0 for predicted, dim 1 for ground truth
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

def match_mz_for_batch_idx(true_mzs:th.Tensor,
						  true_batch_idxs:th.Tensor,
						  pred_mzs:th.Tensor, 
						  pred_batch_idxs:th.Tensor, 
						  tolerance:float, 
						  relative:bool, 
						  batch_idx:int) -> th.Tensor:
	""" 
	Auxiliary function to match m/z values for a given batch index.

    Args:
        true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.  
        true_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
        pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.  
        pred_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
        tolerance (float, optional): Tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
        relative (bool, optional): Flag to use a relative measure. Defaults to True.
        batch_idx (int): Batch index.

    Returns:
        th.Tensor: 2d binary matrix
	"""

	b_true_mask = (true_batch_idxs==batch_idx)
	b_pred_mask = (pred_batch_idxs==batch_idx)
	b_true_mzs = true_mzs[b_true_mask]
	b_pred_mzs = pred_mzs[b_pred_mask]
	b_true_match = calculate_match_mzs(
		b_true_mzs,
		b_pred_mzs,
		tolerance=tolerance,
		relative=relative
	)
	return b_true_match

def calculate_recall(
		true_mzs:th.Tensor,
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_batch_idxs:th.Tensor,
		tolerance:float=1e-5,
		relative:bool=True,
	) -> th.Tensor:
	"""  
	Calculate m/z recall between two m/z lists.

	Note: it is  caller's repsonse to filter ultra low peaks

	Args:
		true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.  
        true_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.  
        pred_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.Tensor: recalls
	"""
	batch_size = th.max(true_batch_idxs)+1
	recall = th.zeros([batch_size],device=true_mzs.device,dtype=th.float)

	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
						  true_batch_idxs,
						  pred_mzs, 
						  pred_batch_idxs, 
						  tolerance, 
						  relative, 
						  b)
		b_true_match_mzs = th.any(b_true_match,dim=1)
		b_relevant = true_mzs[(true_batch_idxs==b)].shape[0]
		b_recall = th.sum(b_true_match_mzs.float())/b_relevant
		recall[b] = b_recall
	return recall

def calculate_weighted_recall(
		true_mzs:th.Tensor,
		true_ints:th.Tensor,
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_ints:th.Tensor,
		pred_batch_idxs:th.Tensor,
		tolerance:float=1e-5,
		relative:bool=True
	) -> th.Tensor:
	""" calculate  m/z recall weighted by ground truth ints between two spectra
		Note: it is  caller's repsonse to filter ultra low peaks
	Args:
		true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.  
        true_ints (th.Tensor): ground truth intensities
		true_ints (th.Tensor): 1d flat tensor of true intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.  
        pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.Tensor: weighted recalls
	"""

	batch_size = th.max(true_batch_idxs)+1
	weighted_recall = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
					true_batch_idxs,
					pred_mzs, 
					pred_batch_idxs, 
					tolerance, 
					relative, 
					b)
		b_true_match_mzs = th.any(b_true_match,dim=1)
		b_relevant_int = true_ints[(true_batch_idxs==b)]		
		b_weighted_recall = th.sum(b_relevant_int[b_true_match_mzs])/th.sum(b_relevant_int)
		weighted_recall[b] = b_weighted_recall
	return weighted_recall

def calculate_precision(
	true_mzs:th.Tensor, 
	true_batch_idxs:th.Tensor,
	pred_mzs:th.Tensor,
	pred_batch_idxs:th.Tensor,
	tolerance:float=1e-5,
	relative:bool=True,
)	-> th.Tensor:
	""" calculate m/z precision between two m/z lists
		Note: it is  caller's repsonse to filter ultra low peaks
	Args:
		true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.  
        true_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.  
        pred_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.Tensor: precisions
	"""
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	precision = th.zeros([batch_size],device=true_mzs.device,dtype=true_mzs.dtype)

	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
						  true_batch_idxs,
						  pred_mzs, 
						  pred_batch_idxs, 
						  tolerance, 
						  relative, 
						  b)
		b_pred_match_mzs = th.any(b_true_match,dim=0)
		b_retrieved  = pred_mzs[(pred_batch_idxs==b)].shape[0]
		b_precision = th.sum(b_pred_match_mzs.float())/b_retrieved
		precision[b] = b_precision

	return precision

def calculate_weighted_precision(
		true_mzs:th.Tensor,
		true_ints:th.Tensor,
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_ints:th.Tensor,
		pred_batch_idxs:th.Tensor,
		tolerance:float=1e-5,
		relative:bool=True
	):
	""" calculate  m/z precision weighted by ground truth ints between two spectra
		Note: it is  caller's repsonse to filter ultra low peaks
	Args:
		true_mzs (th.Tensor): 1d flat tensor of true m/zs; multiple m/z lists are concatenated.  
        true_ints (th.Tensor): ground truth intensities
		true_ints (th.Tensor): 1d flat tensor of true intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs; multiple m/z lists are concatenated.  
        pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of batch indices; each cell, batch_idxs[i] indicates the batch index for mzs[i] and ints[i], should be the same size as mz tensor.  
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.e.
	Returns:
		th.Tensor: recalls
	"""
	batch_size = th.max(true_batch_idxs)+1
	weighted_precision = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
					true_batch_idxs,
					pred_mzs, 
					pred_batch_idxs, 
					tolerance, 
					relative, 
					b)
		
		b_pred_match_mzs = th.any(b_true_match,dim=0)
		b_retrieved_int = pred_ints[(pred_batch_idxs==b)]
		b_weighted_precision = th.sum(b_retrieved_int[b_pred_match_mzs])/th.sum(b_retrieved_int)
		weighted_precision[b] = b_weighted_precision
	return weighted_precision

def calculate_dice(
		true_mzs:th.Tensor, 
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_batch_idxs:th.Tensor,
		tolerance:float=1e-5,
		relative:bool=True,
	):
	""" calculate m/z dice between two m/z lists
		Note: since we can have mulitple matches, this is usinge total_macthed from both m/z list / total_m/z
		Note: it is caller's repsonse to filter ultra low peaks
	Args:
		true_mzs (th.Tensor): ground truth m/z s
		true_batch_idxs (th.Tensor): batch idxs for ground truth
		pred_mzs (th.Tensor): predicted m/z s
		pred_batch_idxs (th.Tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.Tensor: dices
	"""
	batch_size = th.max(true_batch_idxs)+1
	dice = th.zeros([batch_size],device=true_mzs.device,dtype=true_mzs.dtype)
	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
					true_batch_idxs,
					pred_mzs, 
					pred_batch_idxs, 
					tolerance, 
					relative, 
					b)
		
		b_pred_match_mzs = th.any(b_true_match,dim=0)
		b_true_match_mzs = th.any(b_true_match,dim=1)
		b_total_matched = th.sum(b_pred_match_mzs.float()) + th.sum(b_true_match_mzs.float())
		b_retrieved  = pred_mzs[(pred_batch_idxs==b)].shape[0]
		b_relevant = true_mzs[(true_batch_idxs==b)].shape[0]
		b_dice = b_total_matched /(b_retrieved + b_relevant)
		dice[b] = b_dice
	return dice

def calculate_ndcg_hungarian(
	true_mzs:th.Tensor, 
	true_ints:th.Tensor,
	true_batch_idxs:th.Tensor,
	pred_mzs:th.Tensor,
	pred_ints:th.Tensor,
	pred_batch_idxs:th.Tensor,
	tolerance:float=1e-5,
	relative:bool=True,
	sum_ints:bool=True,
	union:bool=False,
	optimistic:bool=True
	) -> th.Tensor:
	"""calculate m/z Normalized Discounted Cumulative Gain between two peak list
		WIP
	Args:
		true_mzs (th.Tensor): 1d flat tensor of ground truth m/zs, each row is an m/z array 
		true_ints (th.Tensor): 1d flat tensor of ground truth intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of ground truth batch indices, each cell, true_batch_idxs[i] indicates batch index for true_mzs[i] and true_ints[i], should be the same size as mz tensor
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs, each row is an m/z array 
		pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of predicted batch indices, each cell, pred_batch_idxs[i] indicates batch index for pred_mzs[i] and pred_ints[i], should be the same size as mz tensor
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative match else use abs tolerance measured in Da. Defaults to True.
		sum_ints (bool, optional): flag to get summed intesnities in aggregate, else get max. Defaults to True.

	Returns:
		th.Tensor:  Normalized Discounted Cumulative Gain
	"""
	# round peaks
	true_mzs, true_ints, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_ints,
		true_batch_idxs,
		sum_ints=sum_ints
	)
	pred_mzs, pred_ints, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		sum_ints=sum_ints
	)

	# l1 normalize
	true_ints = scatter_l1normalize(
		true_ints,
		true_batch_idxs,
		dim=0
	)
	pred_ints = scatter_l1normalize(
		pred_ints,
		pred_batch_idxs,
		dim=0
	)

	# calculate
	batch_size = th.max(true_batch_idxs)+1
	th_device = true_ints.device
	ndcg = th.zeros([batch_size],device=th_device,dtype=true_ints.dtype)
	for batch_idx in range(batch_size):

		b_true_ints = true_ints[(true_batch_idxs==batch_idx)]
		b_pred_ints = pred_ints[(pred_batch_idxs==batch_idx)]
		b_match_mzs = match_mz_for_batch_idx(
			true_mzs,
			true_batch_idxs,
			pred_mzs, 
			pred_batch_idxs, 
			tolerance, 
			relative, 
			batch_idx
		)
		
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)

		b_score = b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs] * \
			(b_true_ints[b_true_match_mzs].unsqueeze(1) * b_pred_ints[b_pred_match_mzs].unsqueeze(0))
		b_true_match_idxs, b_pred_match_idxs = scipy_linear_sum_assignment(b_score, maximize=True)

		b_pred_match_ints = b_pred_ints[b_pred_match_mzs][b_pred_match_idxs]
		b_true_match_ints = b_true_ints[b_true_match_mzs][b_true_match_idxs]

		if union:

			b_pred_unmatch_idxs = th_setdiff1d(th.arange(b_pred_match_mzs.sum(),device=th_device),b_pred_match_idxs)
			b_pred_unmatch_ints = th.cat([b_pred_ints[~b_pred_match_mzs],b_pred_ints[b_pred_match_mzs][b_pred_unmatch_idxs]],dim=0)
			b_true_unmatch_idxs = th_setdiff1d(th.arange(b_true_match_mzs.sum(),device=th_device),b_true_match_idxs)
			b_true_unmatch_ints = th.cat([b_true_ints[~b_true_match_mzs],b_true_ints[b_true_match_mzs][b_true_unmatch_idxs]],dim=0)

			b_pred_all_ints = th.cat(
				[
					b_pred_match_ints,
					b_pred_unmatch_ints,
					# heuristic: rank the unmatched true peaks by their intensity
					-(th.argsort(b_true_unmatch_ints,descending=optimistic)+1).type(b_true_unmatch_ints.dtype)
				],
				dim=0
			)

			b_true_all_ints = th.cat(
				[
					b_true_match_ints,
					# score of zero for the unmatched predicted peaks
					th.zeros_like(b_pred_unmatch_ints),
					b_true_unmatch_ints
				],
				dim=0
			)

			b_pred_ranking = th.argsort(b_pred_all_ints,descending=True)
			b_true_ranking = th.argsort(b_true_all_ints,descending=True)

			b_denom = th.log2(2 + th.arange(b_true_all_ints.shape[0], dtype=b_true_all_ints.dtype, device=th_device))
			b_dcg = th.sum(b_true_all_ints[b_pred_ranking] / b_denom)
			b_idcg = th.sum(b_true_all_ints[b_true_ranking] / b_denom)
			b_ndcg = b_dcg/b_idcg

		else: # intersection

			if b_true_match_idxs.shape[0] == 0:

				b_ndcg = 0.

			else:

				b_pred_ranking = th.argsort(b_pred_match_ints,descending=True)
				b_true_ranking = th.argsort(b_true_match_ints,descending=True)

				b_denom = th.log2(2 + th.arange(b_true_match_ints.shape[0], dtype=b_true_match_ints.dtype, device=th_device))
				b_dcg = th.sum(b_true_match_ints[b_pred_ranking] / b_denom)
				b_idcg = th.sum(b_true_match_ints[b_true_ranking] / b_denom)
				b_ndcg = b_dcg/b_idcg

		ndcg[batch_idx] = b_ndcg

	return ndcg

# cosine similarity code ... all of them
def calculate_cosine_similarity(
		true_mzs:th.Tensor,
		true_ints:th.Tensor,
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_ints:th.Tensor,
		pred_batch_idxs:th.Tensor,
		mz_max:float=1500.,
		mz_bin_res:float=0.01,
		sum_ints:bool=True,
		remove_prec_peaks:bool=False,
		true_prec_mzs:th.Tensor=None,
	) -> th.Tensor:
	"""calculate cosine similarity using binned spectra

	Args:
		true_mzs (th.Tensor): 1d flat tensor of ground truth m/zs, each row is an m/z array 
		true_ints (th.Tensor): 1d flat tensor of ground truth intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of ground truth batch indices, each cell, true_batch_idxs[i] indicates batch index for true_mzs[i] and true_ints[i], should be the same size as mz tensor
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs, each row is an m/z array 
		pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of predicted batch indices, each cell, pred_batch_idxs[i] indicates batch index for pred_mzs[i] and pred_ints[i], should be the same size as mz tensor
		mz_max (float, optional): max mz. Defaults to 1500.
		mz_bin_res (float, optional): bin size measured in Da. Defaults to 0.01.
		sum_ints (bool, optional): flag to get sum intensities in the same bin else use max. Defaults to True.

	Returns:
		th.Tensor: cosine similarity using binned spectra
	"""
	true_spec = batched_bin_func(
		true_mzs,
		true_ints,
		true_batch_idxs,
		mz_max,
		mz_bin_res,
		sum_ints,
		sparse=False,
		remove_prec_peaks=remove_prec_peaks,
		prec_mzs=true_prec_mzs
	)
	pred_spec = batched_bin_func(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max,
		mz_bin_res,
		sum_ints,
		sparse=False,
		remove_prec_peaks=remove_prec_peaks,
		prec_mzs=true_prec_mzs
	)
	cos_sims = binned_cosine_similarity(true_spec,pred_spec)
	return cos_sims

def calculate_optimal_cosine_similarity(
		true_mzs:th.Tensor,
		true_ints:th.Tensor,
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_ints:th.Tensor,
		pred_batch_idxs:th.Tensor,
		mz_max=1500.,
		mz_bin_res=0.01,
		sum_ints=True
	) -> th.Tensor:
	""" 
	Calculate the optimal cosine similarity using binned spectra, 
	Assuming all intensities are correctly predicted for m/z s appeared both in predicted and true spectrum,
	Using predicted intensities if m/z s are not in true spectrum.
	
	Args:
		true_mzs (th.Tensor): 1d flat tensor of ground truth m/zs, each row is an m/z array 
		true_ints (th.Tensor): 1d flat tensor of ground truth intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of ground truth batch indices, each cell, true_batch_idxs[i] indicates batch index for true_mzs[i] and true_ints[i], should be the same size as mz tensor
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs, each row is an m/z array 
		pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of predicted batch indices, each cell, pred_batch_idxs[i] indicates batch index for pred_mzs[i] and pred_ints[i], should be the same size as mz tensor
		mz_max (float, optional): max mz. Defaults to 1500.
		mz_bin_res (float, optional): bin size measured in Da. Defaults to 0.01.
		sum_ints (bool, optional): flag to get sum intensities in the same bin else use max. Defaults to True.

	Returns:
		th.Tensor: optimal cosine similarity
	"""

	true_spec = batched_bin_func(
		true_mzs,
		true_ints,
		true_batch_idxs,
		mz_max,
		mz_bin_res,
		sum_ints
	)
	pred_spec = batched_bin_func(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max,
		mz_bin_res,
		sum_ints
	)
	pred_mask = pred_spec > 0.
	pred_spec = true_spec * pred_mask.float() + pred_spec * (~pred_mask).float()
	opt_cos_sims = binned_cosine_similarity(true_spec,pred_spec)
	return opt_cos_sims

def calculate_sparse_cosine_similarity(
		true_mzs:th.Tensor,
		true_ints:th.Tensor,
		true_batch_idxs:th.Tensor,
		pred_mzs:th.Tensor,
		pred_ints:th.Tensor,
		pred_batch_idxs:th.Tensor,
		mz_max:float=1500.,
		mz_bin_res:float=0.01,
		sum_ints:bool=True,
		remove_prec_peaks:bool=False,
		true_prec_mzs:th.Tensor=None,
	) -> th.Tensor:
	"""
	Calculate the cosine similarity using binned spectra and sparse methods,

	Args:
		true_mzs (th.Tensor): _description_
		true_ints (th.Tensor): _description_
		true_batch_idxs (th.Tensor): _description_
		pred_mzs (th.Tensor): _description_
		pred_ints (th.Tensor): _description_
		pred_batch_idxs (th.Tensor): _description_
		mz_max (float, optional): _description_. Defaults to 1500..
		mz_bin_res (float, optional): _description_. Defaults to 0.01.
		sum_ints (bool, optional): _description_. Defaults to True.

	Returns:
		th.Tensor: cosine similarity
	"""

	# sparse bin
	true_bin_idxs, true_bin_ints, true_bin_batch_idxs = batched_bin_func(
		true_mzs,
		true_ints,
		true_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		sum_ints=sum_ints,
		sparse=True,
		remove_prec_peaks=remove_prec_peaks,
		prec_mzs=true_prec_mzs
	)
	pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs = batched_bin_func(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		sum_ints=sum_ints,
		sparse=True,
		remove_prec_peaks=remove_prec_peaks,
		prec_mzs=true_prec_mzs
	)
	cos_sims = cos_sim_helper(
		true_bin_idxs,
		true_bin_ints,
		true_bin_batch_idxs,
		pred_bin_idxs,
		pred_bin_ints,
		pred_bin_batch_idxs
	)
	return cos_sims


def calculate_optimal_sparse_cosine_similarity(
		true_mzs,
		true_ints,
		true_batch_idxs,
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max=1500.,
		mz_bin_res=0.01,
		sum_ints=True
	) -> th.Tensor:
	"""	
	Calculate the optimal cosine similarity using binned spectra and sparse methods, 
	Assuming all intensities are correctly predicted for m/z s appeared both in predicted and true spectrum,
	Using predicted intensities if m/z s are not in true spectrum.

	Args:
		true_mzs (th.Tensor): 1d flat tensor of ground truth m/zs, each row is an m/z array 
		true_ints (th.Tensor): 1d flat tensor of ground truth intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of ground truth batch indices, each cell, true_batch_idxs[i] indicates batch index for true_mzs[i] and true_ints[i], should be the same size as mz tensor
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs, each row is an m/z array 
		pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of predicted batch indices, each cell, pred_batch_idxs[i] indicates batch index for pred_mzs[i] and pred_ints[i], should be the same size as mz tensor
		mz_max (_type_, optional): _description_. Defaults to 1500Da.
		mz_bin_res (float, optional): _description_. Defaults to 0.01.
		sum_ints (bool, optional): _description_. Defaults to True.

	Returns:
		th.Tensor: optimal cosine similarity
	"""
	# sparse bin
	true_bin_idxs, true_bin_ints, true_bin_batch_idxs = batched_bin_func(
		true_mzs,
		true_ints,
		true_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		sum_ints=sum_ints,
		sparse=True
	)
	pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs = batched_bin_func(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		sum_ints=sum_ints,
		sparse=True
	)
	pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
	true_mask = th.isin(true_bin_idxs, pred_bin_idxs[pred_mask])
	pred_bin_ints[~pred_mask] = 0.
	pred_bin_ints[pred_mask] = true_bin_ints[true_mask]
	opt_cos_sims = cos_sim_helper(
		true_bin_idxs,
		true_bin_ints,
		true_bin_batch_idxs,
		pred_bin_idxs,
		pred_bin_ints,
		pred_bin_batch_idxs
	)
	return opt_cos_sims


def calculate_optimal_unbinned_cosine_similarity(
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
	opt_cos_sim = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
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
		b_opt_cos_sim = th.sum(F.normalize(b_true_ints,p=2,dim=0)[b_true_match]**2,dim=0)
		opt_cos_sim[b] = b_opt_cos_sim
	# assert th.all(opt_cos_sim>=0.) and th.all(opt_cos_sim<=1.), opt_cos_sim
	return opt_cos_sim


def binned_cosine_similarity(true_spec,pred_spec):

	true_spec = F.normalize(true_spec,p=2,dim=1)
	pred_spec = F.normalize(pred_spec,p=2,dim=1)
	cos_sims = (true_spec.unsqueeze(1) @ pred_spec.unsqueeze(2)).squeeze(2).squeeze(1)
	return cos_sims


def cos_sim_helper(true_bin_idxs, true_bin_ints, true_bin_batch_idxs, pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs):

	# l2 normalize
	true_bin_ints = scatter_l2normalize(
		true_bin_ints,
		true_bin_batch_idxs,
		dim=0
	)
	pred_bin_ints = scatter_l2normalize(
		pred_bin_ints,
		pred_bin_batch_idxs,
		dim=0
	)
	# dot product
	pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
	true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
	both_bin_ints = pred_bin_ints[pred_mask] * true_bin_ints[true_mask]
	assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
	cos_sims = scatter_reduce(
		src=both_bin_ints,
		index=pred_bin_batch_idxs[pred_mask],
		reduce="sum",
		dim=0,
		dim_size=th.max(true_bin_batch_idxs)+1
	)
	return cos_sims

def batched_l1_normalize(ints, batch_idxs):

	ints = scatter_l1normalize(
		ints,
		batch_idxs,
		dim=0
	)
	return ints

def batched_mf100_normalize(ints, batch_idxs):

	max_ints = scatter_reduce(
		ints,
		batch_idxs,
		reduce="max",
		dim=0,
		dim_size=th.max(batch_idxs)+1
	)
	ints = (ints / max_ints[batch_idxs]) * 1000.
	return ints

def calculate_sparse_cosine_hungarian(
	true_mzs:th.Tensor, 
	true_ints:th.Tensor,
	true_batch_idxs:th.Tensor,
	pred_mzs:th.Tensor,
	pred_ints:th.Tensor,
	pred_batch_idxs:th.Tensor,
	tolerance:float=1e-5,
	relative:bool=True,
	sum_ints:bool=True,
	remove_prec_peaks:bool=False,
	true_prec_mzs:th.Tensor=None,
	) -> th.Tensor:
	""" Calculate cosine similarity using the Hungarian algorithm as the matching method.

	Args:
		true_mzs (th.Tensor): 1d flat tensor of ground truth m/zs, each row is an m/z array 
		true_ints (th.Tensor): 1d flat tensor of ground truth intensities, each row is an intensities array, should be the same size as mz tensor
		true_batch_idxs (th.Tensor): 1d flat tensor of ground truth batch indices, each cell, true_batch_idxs[i] indicates batch index for true_mzs[i] and true_ints[i], should be the same size as mz tensor
		pred_mzs (th.Tensor): 1d flat tensor of predicted m/zs, each row is an m/z array 
		pred_ints (th.Tensor): 1d flat tensor of predicted intensities, each row is an intensities array, should be the same size as mz tensor
		pred_batch_idxs (th.Tensor): 1d flat tensor of predicted batch indices, each cell, pred_batch_idxs[i] indicates batch index for pred_mzs[i] and pred_ints[i], should be the same size as mz tensor
		tolerance (float):
		relative (bool, optional): flag to use relative match else use abs tolerance measured in Da. Defaults to True.
		sum_ints (bool, optional): flag to get summed intesnities in aggregate, else get max. Defaults to True.

	Returns:
		th.Tensor: cosine similarity tensor, one per pair 
	"""
	# round peaks
	true_mzs, true_ints, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_ints,
		true_batch_idxs,
		sum_ints=sum_ints
	)
	pred_mzs, pred_ints, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		sum_ints=sum_ints
	)
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	cos_hun = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for batch_idx in range(batch_size):
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_mask = (pred_batch_idxs==batch_idx)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
		if remove_prec_peaks:
			# remove precursor peaks
			assert true_prec_mzs is not None
			b_true_prec_mz = true_prec_mzs[batch_idx:batch_idx+1]
			b_true_prec_mask = calculate_match_mzs(
				b_true_mzs,
				b_true_prec_mz,
				tolerance=tolerance,
				relative=relative
			).squeeze(1)
			b_true_ints = b_true_ints*(1-b_true_prec_mask.float())
			b_pred_prec_mask = calculate_match_mzs(
				b_pred_mzs,
				b_true_prec_mz,
				tolerance=tolerance,
				relative=relative
			).squeeze(1)
			b_pred_ints = b_pred_ints*(1-b_pred_prec_mask.float())
		# l2 normalize
		b_true_ints = F.normalize(b_true_ints,p=2,dim=0)
		b_pred_ints = F.normalize(b_pred_ints,p=2,dim=0)
		b_match_mzs = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative
		)
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)
		b_score = b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs] * \
			(b_true_ints[b_true_match_mzs].unsqueeze(1) * b_pred_ints[b_pred_match_mzs].unsqueeze(0))
		b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
		b_cos_hun = th.dot(
			b_true_ints[b_true_match_mzs][b_true_idxs],
			b_pred_ints[b_pred_match_mzs][b_pred_idxs],
		)
		cos_hun[batch_idx] = b_cos_hun
	return cos_hun

def round_aggregate_peaks(mzs:th.Tensor, ints:th.Tensor, batch_idxs:th.Tensor,decimals:int=4,sum_ints:bool=True) -> Tuple[th.Tensor,th.Tensor,th.Tensor]:
	""" methds to round and aggreate peaks to give decimals points

	Args:
		mzs (th.Tensor): 1d flat tensor of m/zs, each row is an m/z array 
		ints (th.Tensor): 1d flat tensor of intensities, each row is an intensities array, should be the same size as mz tensor
		batch_idxs (th.Tensor): 1d flat tensor of batch indices, each cell, batch_idxs[i] indicates batch index for mzs[i] and ints[i], should be the same size as mz tensor
		decimals (int, optional): Decimals. Defaults to 4.
		sum_ints (bool, optional): Flag to sum intensities if True. Defaults to True.
	Returns:
		Tuple[th.Tensor,th.Tensor,th.Tensor]: round_mzs, round_ints, round_batch_idxs
	"""

	batch_size = th.max(batch_idxs)+1
	round_mzs, round_ints, round_batch_idxs = [], [], []
	for b in range(batch_size):
		b_mask = (batch_idxs==b)
		b_round_mzs = th.round(mzs[b_mask], decimals=decimals)
		b_ints = ints[b_mask]
		b_round_mzs_un, b_round_mzs_inv = th.unique(b_round_mzs, return_inverse=True)
		b_round_ints = scatter_reduce(
			src=b_ints,
			index=b_round_mzs_inv,
			reduce="sum" if sum_ints else "amax",
			dim=0,
			dim_size=b_round_mzs_un.shape[0]
		)
		round_mzs.append(b_round_mzs_un)
		round_ints.append(b_round_ints)
		round_batch_idxs.append(th.full_like(b_round_mzs_un,b,dtype=batch_idxs.dtype))
	round_mzs = th.cat(round_mzs,dim=0)
	round_ints = th.cat(round_ints,dim=0)
	round_batch_idxs = th.cat(round_batch_idxs,dim=0)
	return round_mzs, round_ints, round_batch_idxs

def scipy_linear_sum_assignment(matrix, maximize=False):

	device = matrix.device
	matrix = matrix.cpu().numpy()
	x_idx, y_idx = scipy.optimize.linear_sum_assignment(matrix, maximize=maximize)
	x_idx = th.as_tensor(x_idx,dtype=th.long,device=device)
	y_idx = th.as_tensor(y_idx,dtype=th.long,device=device)
	return x_idx, y_idx


def calculate_union_distribution(
	true_mzs:th.Tensor, 
	true_ints:th.Tensor,
	true_batch_idxs:th.Tensor,
	pred_mzs:th.Tensor,
	pred_ints:th.Tensor,
	pred_batch_idxs:th.Tensor,
	tolerance:float=1e-5,
	relative:bool=True,
	sum_ints:bool=True,
	) -> th.Tensor:

	# round peaks
	true_mzs, true_ints, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_ints,
		true_batch_idxs,
		sum_ints=sum_ints
	)
	pred_mzs, pred_ints, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		sum_ints=sum_ints
	)
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	union_mzs, union_ints, union_batch_idxs = [], [], []
	for batch_idx in range(batch_size):
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_mask = (pred_batch_idxs==batch_idx)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
		# l1 normalize
		b_true_ints = F.normalize(b_true_ints,p=1,dim=0)
		b_pred_ints = F.normalize(b_pred_ints,p=1,dim=0)
		b_match_mzs = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative
		)
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)
		b_score = b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs] * \
			b_true_ints[b_true_match_mzs].unsqueeze(1) + b_pred_ints[b_pred_match_mzs].unsqueeze(0)
		b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
		# combine the matches with the ones that didn't match
		# for the matches, take the average mz
		b_union_mzs = th.cat(
			[
				b_true_mzs[~b_true_match_mzs],
				b_pred_mzs[~b_pred_match_mzs],
				0.5*(b_true_mzs[b_true_match_mzs][b_true_idxs]+b_pred_mzs[b_pred_match_mzs][b_pred_idxs])
			],dim=0
		)
		b_union_ints = th.cat(
			[
				0.5*b_true_ints[~b_true_match_mzs],
				0.5*b_pred_ints[~b_pred_match_mzs],
				0.5*b_score[b_true_idxs,b_pred_idxs]
			],dim=0
		)
		b_union_as = th.argsort(b_union_mzs)
		b_union_mzs = b_union_mzs[b_union_as]
		b_union_ints = F.normalize(b_union_ints[b_union_as],p=1,dim=0)
		b_union_batch_idxs = th.full_like(b_union_mzs,batch_idx,dtype=true_batch_idxs.dtype)
		union_mzs.append(b_union_mzs)
		union_ints.append(b_union_ints)
		union_batch_idxs.append(b_union_batch_idxs)
	return union_mzs, union_ints, union_batch_idxs

def calculate_sparse_jensen_hungarian(
	true_mzs:th.Tensor, 
	true_ints:th.Tensor,
	true_batch_idxs:th.Tensor,
	pred_mzs:th.Tensor,
	pred_ints:th.Tensor,
	pred_batch_idxs:th.Tensor,
	tolerance:float=1e-5,
	relative:bool=True,
	sum_ints:bool=True,
	remove_prec_peaks:bool=False,
	true_prec_mzs:th.Tensor=None,
	) -> th.Tensor:

	# round peaks
	true_mzs, true_ints, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_ints,
		true_batch_idxs,
		sum_ints=sum_ints
	)
	pred_mzs, pred_ints, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		sum_ints=sum_ints
	)
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	jensen_hun = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for batch_idx in range(batch_size):
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_mask = (pred_batch_idxs==batch_idx)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
		# l1 normalize
		b_true_ints = F.normalize(b_true_ints,p=1,dim=0)
		b_pred_ints = F.normalize(b_pred_ints,p=1,dim=0)
		b_match_mzs = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative
		)
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)
		b_score = b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs] * \
			b_true_ints[b_true_match_mzs].unsqueeze(1) + b_pred_ints[b_pred_match_mzs].unsqueeze(0)
		b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
		b_union_ints = th.cat(
			[
				0.5*b_true_ints[~b_true_match_mzs],
				0.5*b_pred_ints[~b_pred_match_mzs],
				0.5*b_score[b_true_idxs,b_pred_idxs]
			],dim=0
		)
		b_union_ints = F.normalize(b_union_ints,p=1,dim=0)
		b_true_range = (0,(~b_true_match_mzs).long().sum())
		b_pred_range = (b_true_range[1],b_true_range[1]+(~b_pred_match_mzs).long().sum())
		b_kl1_true_probs = th.cat(
			[
				b_true_ints[~b_true_match_mzs],
				b_true_ints[b_true_match_mzs][b_true_idxs]
			],dim=0
		)
		b_kl1_union_probs = th.cat(
			[
				b_union_ints[b_true_range[0]:b_true_range[1]],
				b_union_ints[b_pred_range[1]:]
			],dim=0
		)
		b_kl1 = th.sum(b_kl1_true_probs*(safelog(b_kl1_true_probs)-safelog(b_kl1_union_probs)), dim=0)
		b_kl2_pred_probs = th.cat(
			[
				b_pred_ints[~b_pred_match_mzs],
				b_pred_ints[b_pred_match_mzs][b_pred_idxs]
			],dim=0
		)
		b_kl2_union_probs = th.cat(
			[
				b_union_ints[b_pred_range[0]:b_pred_range[1]],
				b_union_ints[b_pred_range[1]:]
			],dim=0
		)
		b_kl2 = th.sum(b_kl2_pred_probs*(safelog(b_kl2_pred_probs)-safelog(b_kl2_union_probs)), dim=0)
		b_jensen = 0.5*(b_kl1+b_kl2)
		jensen_hun[batch_idx] = b_jensen
	return jensen_hun

### helpers

def opt_cos_sim_helper(
	true_bin_idxs,
	true_bin_ints,
	true_bin_batch_idxs,
	pred_bin_idxs,
	pred_bin_ints,
	pred_bin_batch_idxs
):

	pred_opt_mask = th.isin(pred_bin_idxs, true_bin_idxs)
	true_opt_mask = th.isin(true_bin_idxs, pred_bin_idxs[pred_opt_mask])
	pred_opt_bin_ints = pred_bin_ints.clone()
	pred_opt_bin_ints[~pred_opt_mask] = 0.
	pred_opt_bin_ints[pred_opt_mask] = true_bin_ints[true_opt_mask]
	opt_cos_sim = cos_sim_helper(
		true_bin_idxs,
		true_bin_ints,
		true_bin_batch_idxs,
		pred_bin_idxs,
		pred_opt_bin_ints,
		pred_bin_batch_idxs
	)
	return opt_cos_sim

def cos_hun_helper(
	b_true_ints,
	b_pred_ints,
	b_match_mask,
	b_true_match_mask,
	b_pred_match_mask,
	remove_prec_peak,
	b_true_prec_mask,
	b_pred_prec_mask):

	if remove_prec_peak:
		import pdb; pdb.set_trace()
		b_true_ints = b_true_ints*(1-b_true_prec_mask.float())
		b_pred_ints = b_pred_ints*(1-b_pred_prec_mask.float())
	b_true_ints = F.normalize(b_true_ints,p=2,dim=0)
	b_pred_ints = F.normalize(b_pred_ints,p=2,dim=0)
	b_score = b_match_mask[b_true_match_mask][:,b_pred_match_mask] * \
		(b_true_ints[b_true_match_mask].unsqueeze(1) * b_pred_ints[b_pred_match_mask].unsqueeze(0))
	b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
	b_cos_hun = th.dot(
		b_true_ints[b_true_match_mask][b_true_idxs],
		b_pred_ints[b_pred_match_mask][b_pred_idxs],
	)
	return b_cos_hun

def ndcg_helper(
	b_true_ints,
	b_pred_ints,
	b_match_mask,
	b_true_match_mask,
	b_pred_match_mask,
	optimistic,
	union):

	th_device = b_true_ints.device
	b_score = b_match_mask[b_true_match_mask][:,b_pred_match_mask] * \
			(b_true_ints[b_true_match_mask].unsqueeze(1) * b_pred_ints[b_pred_match_mask].unsqueeze(0))
	b_true_match_idxs, b_pred_match_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
	b_pred_match_ints = b_pred_ints[b_pred_match_mask][b_pred_match_idxs]
	b_true_match_ints = b_true_ints[b_true_match_mask][b_true_match_idxs]
	if union:
		b_pred_unmatch_idxs = th_setdiff1d(th.arange(b_pred_match_mask.sum(),device=th_device),b_pred_match_idxs)
		b_pred_unmatch_ints = th.cat([b_pred_ints[~b_pred_match_mask],b_pred_ints[b_pred_match_mask][b_pred_unmatch_idxs]],dim=0)
		b_true_unmatch_idxs = th_setdiff1d(th.arange(b_true_match_mask.sum(),device=th_device),b_true_match_idxs)
		b_true_unmatch_ints = th.cat([b_true_ints[~b_true_match_mask],b_true_ints[b_true_match_mask][b_true_unmatch_idxs]],dim=0)
		b_pred_all_ints = th.cat(
			[
				b_pred_match_ints,
				b_pred_unmatch_ints,
				# heuristic: rank the unmatched true peaks by their intensity
				-(th.argsort(b_true_unmatch_ints,descending=optimistic)+1).type(b_true_unmatch_ints.dtype)
			],
			dim=0
		)
		b_true_all_ints = th.cat(
			[
				b_true_match_ints,
				# score of zero for the unmatched predicted peaks
				th.zeros_like(b_pred_unmatch_ints),
				b_true_unmatch_ints
			],
			dim=0
		)
		b_pred_ranking = th.argsort(b_pred_all_ints,descending=True)
		b_true_ranking = th.argsort(b_true_all_ints,descending=True)
		b_denom = th.log2(2 + th.arange(b_true_all_ints.shape[0], dtype=b_true_all_ints.dtype, device=th_device))
		b_dcg = th.sum(b_true_all_ints[b_pred_ranking] / b_denom)
		b_idcg = th.sum(b_true_all_ints[b_true_ranking] / b_denom)
		b_ndcg = b_dcg/b_idcg
	else: # intersection
		if b_true_match_idxs.shape[0] == 0:
			b_ndcg = 0.
		else:
			b_pred_ranking = th.argsort(b_pred_match_ints,descending=True)
			b_true_ranking = th.argsort(b_true_match_ints,descending=True)
			b_denom = th.log2(2 + th.arange(b_true_match_ints.shape[0], dtype=b_true_match_ints.dtype, device=th_device))
			b_dcg = th.sum(b_true_match_ints[b_pred_ranking] / b_denom)
			b_idcg = th.sum(b_true_match_ints[b_true_ranking] / b_denom)
			b_ndcg = b_dcg/b_idcg
	return b_ndcg

def jss_hun_helper(
	b_true_ints,
	b_pred_ints,
	b_match_mask,
	b_true_match_mask,
	b_pred_match_mask,
	remove_prec_peak,
	b_true_prec_mask,
	b_pred_prec_mask):

	if remove_prec_peak:
		b_true_ints = b_true_ints*(1-b_true_prec_mask.float())
		b_pred_ints = b_pred_ints*(1-b_pred_prec_mask.float())
	b_true_ints = F.normalize(b_true_ints,p=1,dim=0)
	b_pred_ints = F.normalize(b_pred_ints,p=1,dim=0)
	if th.all(b_pred_ints) == 0.:
		# heuristic to prevent nan
		b_pred_ints = th.ones_like(b_pred_ints) / b_pred_ints.shape[0]
	b_score = b_match_mask[b_true_match_mask][:,b_pred_match_mask] * \
		(b_true_ints[b_true_match_mask].unsqueeze(1) + b_pred_ints[b_pred_match_mask].unsqueeze(0))
	b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
	b_union_ints = th.cat(
		[
			0.5*b_true_ints[~b_true_match_mask],
			0.5*b_pred_ints[~b_pred_match_mask],
			0.5*b_score[b_true_idxs,b_pred_idxs]
		],dim=0
	)
	b_union_ints = F.normalize(b_union_ints,p=1,dim=0)
	b_true_range = (0,(~b_true_match_mask).long().sum())
	b_pred_range = (b_true_range[1],b_true_range[1]+(~b_pred_match_mask).long().sum())
	b_kl1_true_probs = th.cat(
		[
			b_true_ints[~b_true_match_mask],
			b_true_ints[b_true_match_mask][b_true_idxs]
		],dim=0
	)
	b_kl1_union_probs = th.cat(
		[
			b_union_ints[b_true_range[0]:b_true_range[1]],
			b_union_ints[b_pred_range[1]:]
		],dim=0
	)
	b_kl1 = th.sum(b_kl1_true_probs*(safelog(b_kl1_true_probs)-safelog(b_kl1_union_probs)), dim=0)
	b_kl2_pred_probs = th.cat(
		[
			b_pred_ints[~b_pred_match_mask],
			b_pred_ints[b_pred_match_mask][b_pred_idxs]
		],dim=0
	)
	b_kl2_union_probs = th.cat(
		[
			b_union_ints[b_pred_range[0]:b_pred_range[1]],
			b_union_ints[b_pred_range[1]:]
		],dim=0
	)
	b_kl2 = th.sum(b_kl2_pred_probs*(safelog(b_kl2_pred_probs)-safelog(b_kl2_union_probs)), dim=0)
	b_jss_hun = 1.-0.5*(b_kl1+b_kl2)
	return b_jss_hun

def jss_helper(
	true_bin_idxs,
	true_bin_ints,
	true_bin_batch_idxs,
	pred_bin_idxs,
	pred_bin_ints,
	pred_bin_batch_idxs):
	
	batch_size = th.max(true_bin_batch_idxs)+1
	# l1 normalize
	true_bin_ints = scatter_l1normalize(
		true_bin_ints,
		true_bin_batch_idxs,
		dim=0
	)
	pred_bin_ints = scatter_l1normalize(
		pred_bin_ints,
		pred_bin_batch_idxs,
		dim=0
	)
	# union distribution
	union_bin_idxs, union_bin_idxs_rev = th.unique(th.cat([true_bin_idxs,pred_bin_idxs],dim=0),return_inverse=True)
	union_bin_ints = scatter_reduce(
		src=0.5*th.cat([true_bin_ints,pred_bin_ints],dim=0),
		index=union_bin_idxs_rev,
		reduce="sum",
		dim=0,
		dim_size=union_bin_idxs.shape[0]
	)
	# union_bin_batch_idxs = scatter_reduce(
	# 	src=th.cat([true_bin_batch_idxs,pred_bin_batch_idxs],dim=0),
	# 	index=union_bin_idxs_rev,
	# 	reduce="amax",
	# 	dim=0,
	# 	dim_size=union_bin_idxs.shape[0]
	# )
	# union_bin_ints = scatter_l1normalize(
	# 	union_bin_ints,
	# 	union_bin_batch_idxs,
	# 	dim=0
	# )
	# kl1
	kl1_union_bin_ints = union_bin_ints[union_bin_idxs_rev[:true_bin_idxs.shape[0]]]
	kl1 = scatter_reduce(
		true_bin_ints * (safelog(true_bin_ints) - safelog(kl1_union_bin_ints)),
		true_bin_batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size
	)
	# kl2
	kl2_union_bin_ints = union_bin_ints[union_bin_idxs_rev[true_bin_idxs.shape[0]:]]
	kl2 = scatter_reduce(
		pred_bin_ints * (safelog(pred_bin_ints) - safelog(kl2_union_bin_ints)),
		pred_bin_batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size
	)
	# jss
	jss = 1.-0.5*(kl1+kl2)
	return jss
