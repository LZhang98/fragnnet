import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
import torch as th
import torch.nn.functional as F
import scipy
import scipy.optimize

from frag_gnn.utils.formula_utils import ELECTRON_MASS, PREC_TYPE_TO_MASS_DIFF
from frag_gnn.utils.misc_utils import scatter_logsoftmax, scatter_l1normalize, scatter_l2normalize, scatter_reduce, EPS


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
		ace (_type_): _description_

	Raises:
		NotImplementedError: _description_
	Returns:
		float: normalized absolute collision energy for ml
	"""
	# TODO fix this
	raise NotImplementedError
	#return ace/100 	

def filter_func(mzs:List[float],ints:List[float],ints_thresh:float,mz_max:float):

	thresh_mask = ints > ints_thresh
	max_mask = mzs < mz_max
	both_mask = thresh_mask & max_mask
	mzs = mzs[both_mask]
	ints = ints[both_mask]
	return mzs, ints

def bin_func(mzs:th.tensor,ints:th.tensor,mz_max:float,mz_bin_res:float,return_index:bool,sum_ints:bool):
	"""
	Note: if return_index is True, returns the (possibly non-unique) bin index for each mz
	Note: intensities may not be normalized due to peak merging
	Args:
		mzs (th.tensor): tensor of mzs
		ints (th.tensor): tensor of intensity
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


def batched_bin_func(mzs:th.tensor,ints:th.tensor,batch_idxs:th.tensor,mz_max:float,mz_bin_res:float,sum_ints:bool,sparse:bool=False) -> th.tensor:
	"""method to get binned spectra for batch

	Args:
		mzs (th.tensor): tensor of mzs
		ints (th.tensor): tensor of intensity
		batch_idxs (th.tensor): tensor of batch idxs
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
	bin_idx = th.searchsorted(bins,mzs,side="right")
	bin_offsets = (th.arange(batch_size,device=mzs.device)*num_bins)[batch_idxs]
	bin_idx = bin_idx + bin_offsets
	if sparse:
		un_bin_idxs, un_bin_idxs_rev = th.unique(bin_idx,return_inverse=True)
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
			index=bin_idx,
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


def merge_sparse_specs(*peakses,renormalize:bool=False,sum_ints:bool=True) -> th.tensor:
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


def calculate_spectrum_entropy(log_ints:th.tensor,batch_idxs:th.tensor) -> th.tensor:
	""" method to compute entropy.
		NOTE: this is NOT same spectra entropy in this https://www.nature.com/articles/s41592-023-02012-9 a
		nd https://www.nature.com/articles/s41592-021-01331-z
	Args:
		log_ints (th.tensor): intensity in log scale
		batch_idxs (th.tensor): batch idx

	Returns:
		th.tensor: spectrum entropy
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
		true_mzs:th.tensor,
		pred_mzs:th.tensor,
		tolerance:float=1e-5,
		relative:bool=True,
		divisor_mz_floor:float=200
	) -> th.tensor:
	"""
	Method to match two spectra based on m/z
	works with numpy arrays or torch tensors  inspired by the function in ms-pred
	NOT BATCHED
	Args:
		true_mzs (th.tensor): ground truth m/z s
		pred_mzs (th.tensor): predicted m/z s
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
		divisor_mz_floor (float, optional): divisor floor used in relative. Defaults to 200 Da.

	Returns:
		th.tensor: macthed m/zs tensor, dim 0 for predicted, dim 1 for ground truth
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

def match_mz_for_batch_idx(true_mzs:th.tensor,
						  true_batch_idxs:th.tensor,
						  pred_mzs:th.tensor, 
						  pred_batch_idxs:th.tensor, 
						  tolerance:float, 
						  relative:bool, 
						  batch_idx:int):
	""" aux function to mathc mzs
	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_batch_idxs (th.tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
		batch_idx (int): batch idx
	Returns:
		_type_: _description_
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
		true_mzs:th.tensor,
		true_batch_idxs:th.tensor,
		pred_mzs:th.tensor,
		pred_batch_idxs:th.tensor,
		tolerance:float=1e-5,
		relative:bool=True
	) -> th.tensor:
	"""calculate m/z recall between two spectra

	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_batch_idxs (th.tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.tensor: recalls
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
		true_mzs,
		true_ints,
		true_batch_idxs,
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		tolerance=1e-5,
		relative=True
	) -> th.tensor:
	"""calculate  m/z recall weighted by ground truth ints between two spectra

	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_ints (th.tensor): ground truth intensities
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_ints (th.tensor): predicted intensities
		pred_batch_idxs (th.tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.tensor: weighted recalls
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
	true_mzs:th.tensor, 
	true_ints:th.tensor,
	true_batch_idxs:th.tensor,
	pred_mzs:th.tensor,
	pred_ints:th.tensor,
	pred_batch_idxs:th.tensor,
	tolerance:float=1e-5,
	relative:bool=True,
)	-> th.tensor:
	"""calculate m/z precision between two spectra

	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_batch_idxs (th.tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.tensor: recalls
	"""
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	precision = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)


	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
						  true_batch_idxs,
						  pred_mzs, 
						  pred_batch_idxs, 
						  tolerance, 
						  relative, 
						  b)
		b_pred_match_mzs = th.any(b_true_match,dim=0)
		b_retrieved  = pred_mzs[(true_batch_idxs==b)].shape[0]
		b_precision = th.sum(b_pred_match_mzs.float())/b_retrieved
		precision[b] = b_precision

	return precision

def calculate_weighted_precision(
		true_mzs,
		true_ints,
		true_batch_idxs,
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		tolerance=1e-5,
		relative=True
	):
	"""calculate  m/z precision weighted by ground truth ints between two spectra

	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_batch_idxs (th.tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.tensor: recalls
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
		b_retrieved_int = pred_ints[(true_batch_idxs==b)]
		b_weighted_precision = th.sum(b_retrieved_int[b_pred_match_mzs])/th.sum(b_retrieved_int)
		weighted_precision[b] = b_weighted_precision
	return weighted_precision

def calculate_dice(
		true_mzs:th.tensor, 
		true_ints:th.tensor,
		true_batch_idxs:th.tensor,
		pred_mzs:th.tensor,
		pred_ints:th.tensor,
		pred_batch_idxs:th.tensor,
		tolerance:float=1e-5,
		relative:bool=True,
	):
	"""calculate m/z dice between two spectra

	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_ints (th.tensor): ground truth intensities
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_ints (th.tensor): predicted intensities
		pred_batch_idxs (th.tensor): batch idxs for prediction
		tolerance (float, optional): tolerance, Da if not relative, else ratios (NOT PPM). Defaults to 1e-5.
		relative (bool, optional): flag to use relative measure. Defaults to True.
	Returns:
		th.tensor: dices
	"""
	batch_size = th.max(true_batch_idxs)+1
	dice = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for b in range(batch_size):
		b_true_match = match_mz_for_batch_idx(true_mzs,
					true_batch_idxs,
					pred_mzs, 
					pred_batch_idxs, 
					tolerance, 
					relative, 
					b)
		
		b_pred_match_mzs = th.any(b_true_match,dim=0)
		b_retrieved  = pred_mzs[(true_batch_idxs==b)].shape[0]
		b_relevant = true_ints[(true_batch_idxs==b)].shape[0]	
		b_dice = th.sum(b_pred_match_mzs.float())/(b_retrieved + b_relevant) * 2.0
		dice[b] = b_dice
	return dice

# cosine similarity code ... all of them
def calculate_cosine_similarity(
		true_mzs:th.tensor,
		true_ints:th.tensor,
		true_batch_idxs:th.tensor,
		pred_mzs:th.tensor,
		pred_ints:th.tensor,
		pred_batch_idxs:th.tensor,
		mz_bin_res:float=0.01,
		mz_max:float=1500.,
		sum_ints:bool=True
	):
	"""calculate cosine similarity using binned spectra

	Args:
		true_mzs (th.tensor): ground truth m/z s
		true_ints (th.tensor): ground truth intensities
		true_batch_idxs (th.tensor): batch idxs for ground truth
		pred_mzs (th.tensor): predicted m/z s
		pred_ints (th.tensor): predicted intensities
		pred_batch_idxs (th.tensor): batch idxs for prediction
		mz_bin_res (float, optional): bin size measured in Da. Defaults to 0.01.
		mz_max (float, optional): max mz. Defaults to 1500..
		sum_ints (bool, optional): flag to get sum intensities in the same bin else use max. Defaults to True.

	Returns:
		_type_: _description_
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
	cos_sims = binned_cosine_similarity(true_spec,pred_spec)
	return cos_sims

def calculate_optimal_cosine_similarity(
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
		true_mzs,
		true_ints,
		true_batch_idxs,
		pred_mzs,
		pred_ints,
		pred_batch_idxs,
		mz_max=1500.,
		mz_bin_res=0.01,
		sum_ints=True
	):

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
	cos_sims = sparse_binned_cosine_similarity(
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
	):

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
	opt_cos_sims = sparse_binned_cosine_similarity(
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


def sparse_binned_cosine_similarity(true_bin_idxs, true_bin_ints, true_bin_batch_idxs, pred_bin_idxs, pred_bin_ints, pred_bin_batch_idxs):

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
	true_mzs, 
	true_ints, 
	true_batch_idxs, 
	pred_mzs, 
	pred_ints, 
	pred_batch_idxs,
	tolerance=1e-5,
	relative=True,
	sum_ints=True
):

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
	# l2 normalize
	true_ints = scatter_l2normalize(
		true_ints,
		true_batch_idxs,
		dim=0
	)
	pred_ints = scatter_l2normalize(
		pred_ints,
		pred_batch_idxs,
		dim=0
	)
	# calculate
	batch_size = th.max(true_batch_idxs)+1
	cos_hun = th.zeros([batch_size],device=true_ints.device,dtype=true_ints.dtype)
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
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
		cos_hun[b] = b_cos_hun
	return cos_hun

def round_aggregate_peaks(mzs:th.tensor,ints:th.tensor,batch_idxs:th.tensor,decimals:int=4,sum_ints:bool=True):
	""" methds to round and aggreate peaks to give decimals points

	Args:
		mzs (th.tensor): list of mzs
		ints (th.tensor): list of intensities
		batch_idxs (th.tensor): batch_idxs
		decimals (int, optional): decimals. Defaults to 4.
		sum_ints (bool, optional): flag sum intensities if True. Defaults to True.

	Returns:
		_type_: _description_
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