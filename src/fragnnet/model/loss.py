"""
Land of loss functions
"""

import numpy as np
import math
import torch as th
import torch.nn.functional as F
from typing import Callable

from fragnnet.utils.misc_utils import scatter_reduce, EPS, safelog, LOG_ZERO, scatter_logsumexp, scatter_logl2normalize, scatter_logsoftmax, TOLERANCE_MIN_MZ
from fragnnet.utils.spec_utils import calculate_match_mzs, scipy_linear_sum_assignment, round_aggregate_peaks, batched_bin_func, jss_helper

# LOG_TRUNC_FACTOR = {
# 	i: float(th.log(th.erf(th.tensor(i)/np.sqrt(2)))) for i in range(0,6)
# }

def mog_log_prob(
		samples: th.Tensor,
		means: th.Tensor,
		variances: th.Tensor,
		log_weights: th.Tensor,
		log_trunc_factors: th.Tensor) -> th.Tensor:
	""" Mixture of Gaussian
		We are modeling each peaks are mixture of guassian for given set of fragments
	Args:
		samples (_type_): b_true_mz,
		means (_type_): b_pred_mzs
		variances (_type_): b_variances
		log_weights (_type_): b_pred_logprobs
		log_trunc_factors (_type_): b_log_trunc_factors

	Returns:
		_type_: _description_
	"""
	
	samples = samples.reshape(-1,1)
	means = means.reshape(1,-1)
	variances = variances.reshape(1,-1)
	log_weights = log_weights.reshape(1,-1)
	log_trunc_factors = log_trunc_factors.reshape(1,-1)
	# [B,K]
	normal_log_probs = -0.5*(samples-means)**2/variances - 0.5*th.log(2*th.pi*variances)
	normal_log_probs = normal_log_probs - log_trunc_factors
	# [B]
	log_probs = th.logsumexp(normal_log_probs + log_weights, dim=1)
	return log_probs

def mog_ce_fn(
		b_true_mzs: th.Tensor,
		b_pred_mzs: th.Tensor,
		b_pred_logprobs: th.Tensor,
		relative: bool,
		tolerance: float,
		tolerance_min_mz: float,
		tolerance_multiple: float,
		gaussian_renormalize: bool,
		**kwargs) -> th.Tensor:

	if relative:
		b_stds = th.clamp(b_pred_mzs,min=tolerance_min_mz)*tolerance
	else:
		b_stds = tolerance*th.ones_like(b_pred_mzs)
	b_vars = b_stds**2
	if gaussian_renormalize:
		assert tolerance_multiple > 0, tolerance_multiple
		# log_trunc_factor = LOG_TRUNC_FACTOR.get(tolerance_multiple,1.)
		# b_log_trunc_factors = log_trunc_factor*th.ones_like(b_pred_mzs)
		sigma = th.tensor(float(np.sqrt(2)), device=b_pred_mzs.device)
		trunc_factor = th.erf(th.tensor(tolerance_multiple,device=b_pred_mzs.device)/ sigma)
		b_log_trunc_factors = th.log(trunc_factor*th.ones_like(b_pred_mzs))
	else:
		b_log_trunc_factors = th.zeros_like(b_pred_mzs)
	b_logprobs = mog_log_prob(
		b_true_mzs,
		b_pred_mzs,
		b_vars,
		b_pred_logprobs,
		b_log_trunc_factors
	)
	
	return b_logprobs

def pm_ce_fn(
		b_pred_mzs: th.Tensor,
		b_true_mzs: th.Tensor,
		b_pred_logprobs: th.Tensor,
		relative: bool,
		tolerance: float,
		tolerance_multiple: float,
		tolerance_min_mz: float,
		**kwargs) -> th.Tensor:

	"""
	Compute per-true-peak log-probability via peak-marginal aggregation.

	For each true peak in `b_true_mzs`, this function aggregates the
	log-probability mass assigned by the predicted peaks `b_pred_mzs`
	within an m/z tolerance window. Concretely, it performs a
	log-sum-exp over the predicted log-probabilities `b_pred_logprobs`
	that match each true peak, yielding a length-`T` tensor (one value
	per true peak).

	Args:
		b_pred_mzs: [P] Predicted m/z values for the current spectrum.
		b_true_mzs: [T] Ground-truth m/z values for the current spectrum.
		b_pred_logprobs: [P] Predicted log-probabilities (intensities) per predicted peak.
		relative: If True, the tolerance is relative to m/z; otherwise, absolute.
		tolerance: Base tolerance value (absolute or the factor used for relative).
		tolerance_multiple: Multiplicative factor applied to `tolerance` when matching.
		tolerance_min_mz: Minimum m/z used when applying relative tolerance.

	Returns:
		[T] Log-probabilities per true peak obtained by aggregating matching
		predicted peak log-probabilities via log-sum-exp.
	"""

	# Compute boolean match mask [T, P] indicating which predicted peaks
	# fall within the m/z tolerance window of each true peak.
	b_match_mask = calculate_match_mzs(
		b_true_mzs,
		b_pred_mzs,
		tolerance=tolerance*tolerance_multiple,
		relative=relative,
		tolerance_min_mz=tolerance_min_mz
	)
	# Aggregate matched predicted log-probs for each true peak via log-sum-exp.
	# Unmatched (false mask) entries are suppressed by adding a large negative
	# value (`LOG_ZERO`) so they do not contribute to the sum.
	b_logprobs = th.logsumexp(
		b_pred_logprobs.reshape(1,-1)*b_match_mask.to(b_pred_logprobs.dtype) + LOG_ZERO(b_pred_logprobs.dtype)*(~b_match_mask).to(b_pred_logprobs.dtype),
		dim=1)
	return b_logprobs

def sparse_cross_entropy_seq(
		true_mzs: th.Tensor,
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		pred_oos_logprobs: th.Tensor,
		ce_fn: Callable,
		tolerance: float,
		relative: bool,
		tolerance_min_mz: float,
		tolerance_multiple: int,
		gaussian_renormalize: bool) -> th.Tensor:
	"""compute sparse cross entropy in squence

	Args:
		true_mzs (_type_): groundtruth mzs
		true_logprobs (_type_): groundtruth log prob of each mz aka intensities
		true_batch_idxs (_type_): spec idx signed to each groundtruth
		pred_mzs (_type_): predicted mzs
		pred_logprobs (_type_): predicted log prob of each mz aka intensities
		pred_batch_idxs (_type_): pec idx signed to each prediction
		ce_fn (_type_, optional): _description_. Defaults to mog_log_prob.
		tolerance (_type_, optional): _description_. Defaults to 1e-3.
		relative (bool, optional): _description_. Defaults to False.
		oos_tolerance_multiple (int, optional): _description_.
		gaussian_renormalize (bool, optional): _description_.
		pm_tolerance_multiple (int, optional): _description_.
		
	Returns:
		_type_: _description_
	"""

	# print(pred_logprobs.shape[0]*true_logprobs.shape[0]*4 / 1e6)
	mask_value = LOG_ZERO(true_logprobs.dtype)
	batch_size = th.max(true_batch_idxs)+1
	# ces = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
	ios_ces = th.zeros([batch_size], dtype=true_logprobs.dtype,device=true_logprobs.device)
	oos_ces = th.zeros([batch_size], dtype=true_logprobs.dtype,device=true_logprobs.device)
	true_oos_logprobs = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
	true_oos_entropies = th.zeros_like(true_oos_logprobs)
	for batch_idx in th.arange(batch_size):
		# get groundtruth belongs to this batch idx
		b_true_mask = true_batch_idxs==batch_idx
		# get predictions belongs to this batch idx
		b_pred_mask = pred_batch_idxs==batch_idx
		# get mzs and logprobs
		b_true_logprobs = true_logprobs[b_true_mask]
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_logprobs = pred_logprobs[b_pred_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		# get matches
		b_match_mask = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance_multiple*tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		if tolerance_multiple == -1:
			b_true_match_mask = th.ones_like(b_true_mzs,dtype=th.bool)
		else:
			assert tolerance_multiple > 0, tolerance_multiple	
			b_true_match_mask = th.any(b_match_mask,dim=1)
		if not th.any(b_true_match_mask):
			print("> Warning: Everything is OOS!")
		# calculate ios logprobs
		b_ios_logprobs = ce_fn(
			b_true_mzs=b_true_mzs,
			b_pred_mzs=b_pred_mzs,
			b_pred_logprobs=b_pred_logprobs,
			relative=relative,
			tolerance=tolerance,
			tolerance_min_mz=tolerance_min_mz,
			tolerance_multiple=tolerance_multiple,
			gaussian_renormalize=gaussian_renormalize)
		b_ios_ce = th.sum(b_true_match_mask.to(b_true_logprobs.dtype) * -th.exp(b_true_logprobs) * b_ios_logprobs)
		# calculate oos logprobs
		b_true_oos_logprobs = th.logsumexp( mask_value*b_true_match_mask.to(b_true_logprobs.dtype) + b_true_logprobs*(~b_true_match_mask).to(b_true_logprobs.dtype), dim=0)
		b_true_oos_entropy = - th.sum( th.exp(b_true_logprobs[~b_true_match_mask]-b_true_oos_logprobs) * (b_true_logprobs[~b_true_match_mask]-b_true_oos_logprobs), dim=0)
		b_pred_oos_logprobs = pred_oos_logprobs[batch_idx]
		b_oos_ce = -th.exp(b_true_oos_logprobs) * b_pred_oos_logprobs
		# combine
		ios_ces[batch_idx] = b_ios_ce
		oos_ces[batch_idx] = b_oos_ce
		true_oos_logprobs[batch_idx] = b_true_oos_logprobs
		true_oos_entropies[batch_idx] = b_true_oos_entropy
	return ios_ces, oos_ces, true_oos_logprobs, true_oos_entropies

def sparse_cross_entropy_vec(
		true_mzs: th.Tensor,
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		pred_oos_logprobs: th.Tensor,
		tolerance: float,
		relative: bool,
		tolerance_min_mz: float,
		oos_tolerance_multiple: int,
		gaussian_renormalize: bool,
		loss_batch_size: int) -> th.Tensor:
	"""
	Vectorized computation of sparse cross-entropy loss for batched spectra.

	Args:
		true_mzs: Ground truth m/z values.
		true_logprobs: Ground truth log-probabilities (intensities).
		true_batch_idxs: Batch indices for ground truth peaks.
		pred_mzs: Predicted m/z values.
		pred_logprobs: Predicted log-probabilities (intensities).
		pred_batch_idxs: Batch indices for predicted peaks.
		pred_oos_logprobs: Predicted log-probabilities for out-of-sample (OOS) peaks.
		tolerance: m/z matching tolerance.
		relative: Whether tolerance is relative to m/z.
		tolerance_min_mz: Minimum m/z for relative tolerance.
		oos_tolerance_multiple: Multiplier for OOS tolerance.
		gaussian_renormalize: Whether to renormalize Gaussian.
		loss_batch_size: Batch size for loss computation.

	Returns:
		ios_ce: In-sample cross-entropy per batch.
		oos_ce: Out-of-sample cross-entropy per batch.
		0., 1.: Placeholders for compatibility.
	"""

	# Set up types and constants
	float_dtype = true_logprobs.dtype
	int_type = true_batch_idxs.dtype
	device = true_logprobs.device
	mask_value = LOG_ZERO(float_dtype)
	batch_size = th.max(true_batch_idxs)+1

	# Calculate number of loss batches for memory efficiency
	loss_num_batches = (batch_size // loss_batch_size) + int(batch_size % loss_batch_size > 0)

	# Compute cumulative sum of counts for batch slicing
	true_batch_cumsum = th.cumsum(F.pad(th.unique(true_batch_idxs,return_counts=True)[1],(1,0),"constant",0),dim=0)
	pred_batch_cumsum = th.cumsum(F.pad(th.unique(pred_batch_idxs,return_counts=True)[1],(1,0),"constant",0),dim=0)

	# Initialize output tensors for in-sample and out-of-sample cross-entropy
	ios_ce = th.zeros([batch_size],dtype=float_dtype,device=device)
	oos_ce = th.zeros([batch_size],dtype=float_dtype,device=device)

	# Loop over loss batches for memory efficiency
	for bl in range(loss_num_batches):
		# Compute batch slice indices
		bl_lower = bl*loss_batch_size
		bl_upper = min((bl+1)*loss_batch_size,batch_size)
		bl_true_lower = true_batch_cumsum[bl_lower]
		bl_true_upper = true_batch_cumsum[bl_upper]
		bl_pred_lower = pred_batch_cumsum[bl_lower]
		bl_pred_upper = pred_batch_cumsum[bl_upper]
		bl_batch_size = bl_upper-bl_lower

		# Slice batch data and adjust indices to local batch
		bl_true_batch_idxs = true_batch_idxs[bl_true_lower:bl_true_upper] - bl_lower
		bl_pred_batch_idxs = pred_batch_idxs[bl_pred_lower:bl_pred_upper] - bl_lower
		bl_true_logprobs = true_logprobs[bl_true_lower:bl_true_upper]
		bl_pred_logprobs = pred_logprobs[bl_pred_lower:bl_pred_upper]
		bl_true_mzs = true_mzs[bl_true_lower:bl_true_upper]
		bl_pred_mzs = pred_mzs[bl_pred_lower:bl_pred_upper]
		bl_pred_oos_logprobs = pred_oos_logprobs[bl_lower:bl_upper]

		# Create mask for matching batch indices
		bl_batch_mask = bl_true_batch_idxs.reshape(-1,1) == bl_pred_batch_idxs.reshape(1,-1)
		# Create mask for matching m/z values within tolerance
		bl_match_mask = calculate_match_mzs(
			bl_true_mzs,
			bl_pred_mzs,
			tolerance=oos_tolerance_multiple*tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		# Combine masks: only consider matches within same batch and within m/z tolerance
		bl_both_mask = bl_batch_mask & bl_match_mask
		del bl_batch_mask, bl_match_mask

		# Mask for true peaks that have any matching prediction
		bl_true_both_mask = th.any(bl_both_mask,dim=1)

		# Compute Gaussian stds and log truncation factors for predictions
		if relative:
			bl_stds = th.clamp(bl_pred_mzs,min=tolerance_min_mz)*tolerance
		else:
			bl_stds = tolerance*th.ones_like(bl_pred_mzs)
		bl_vars = bl_stds**2
		if gaussian_renormalize:
			assert oos_tolerance_multiple > 0, oos_tolerance_multiple
			sigma = th.tensor(float(np.sqrt(2)), device=device)
			bl_trunc_factor = th.erf(th.tensor(oos_tolerance_multiple, device=device) / sigma)
			bl_log_trunc_factors = th.log(bl_trunc_factor*th.ones_like(bl_pred_mzs))
		else:
			bl_log_trunc_factors = th.zeros_like(bl_pred_mzs)
		
		# Compute log-probabilities for in-sample peaks (vectorized Gaussian mixture)
		bl_ios_log_probs = -0.5*(bl_true_mzs.reshape(-1,1)-bl_pred_mzs.reshape(1,-1))**2/bl_vars.reshape(1,-1) - 0.5*th.log(2*th.pi*bl_vars.reshape(1,-1))
		bl_ios_log_probs = bl_ios_log_probs + bl_pred_logprobs.reshape(1,-1) - bl_log_trunc_factors.reshape(1,-1)
		# Mask out non-matching pairs with mask_value (large negative)
		bl_ios_log_probs = bl_ios_log_probs + (~bl_both_mask).to(float_dtype)*mask_value
		# Log-sum-exp over predictions for each true peak
		bl_ios_log_probs = th.logsumexp(bl_ios_log_probs,dim=1)

		# Compute in-sample cross-entropy for this batch
		if not th.any(bl_true_both_mask):
			# If no matches, fill with LOG_ZERO
			bl_ios_ce = th.full([bl_batch_size],LOG_ZERO(float_dtype),dtype=float_dtype,device=device)
		else:
			# Sum over matched true peaks, reduce by batch index
			bl_ios_ce = scatter_reduce(
				(-th.exp(bl_true_logprobs)*bl_ios_log_probs)[bl_true_both_mask],
				bl_true_batch_idxs[bl_true_both_mask],
				"sum",
				dim=0,
				dim_size=bl_batch_size,
				default=0.)
			ios_ce[bl_lower:bl_upper] = bl_ios_ce

		# Compute out-of-sample logprobs for true peaks with no match
		if not th.all(bl_true_both_mask):
			bl_true_oos_logprobs = scatter_logsumexp(
				bl_true_logprobs[~bl_true_both_mask],
				bl_true_batch_idxs[~bl_true_both_mask],
				dim_size=bl_batch_size
			)
		else:
			# If all matched, fill with LOG_ZERO
			bl_true_oos_logprobs = th.full([bl_batch_size],LOG_ZERO(float_dtype),dtype=float_dtype,device=device)
		# Compute out-of-sample cross-entropy
		bl_oos_ce = -th.exp(bl_true_oos_logprobs)*bl_pred_oos_logprobs
		oos_ce[bl_lower:bl_upper] = bl_oos_ce
		
	return ios_ce, oos_ce, 0., 1.

def get_sparse_cross_entropy_fn(
		dist: str,
		vectorized: bool,
		tolerance: float,
		relative: bool,
		tolerance_min_mz: float,
		oos_tolerance_multiple: float,
		gaussian_renormalize: bool,
		pm_tolerance_multiple: float,
		loss_batch_size: int) -> Callable:

	if dist == "gaussian":
		ce_fn = mog_ce_fn
	elif dist == "peak_marginal":
		ce_fn = pm_ce_fn
	else:
		raise NotImplementedError

	if vectorized:
		assert dist == "gaussian", dist
		sce_fn = lambda *args: sparse_cross_entropy_vec(
			*args,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz,
			oos_tolerance_multiple=oos_tolerance_multiple,
			gaussian_renormalize=gaussian_renormalize,
			loss_batch_size=loss_batch_size
		)
	else:
		if dist == "gaussian":
			tolerance_multiple = oos_tolerance_multiple
		else:
			tolerance_multiple = pm_tolerance_multiple
		sce_fn = lambda *args: sparse_cross_entropy_seq(
			*args,
			ce_fn=ce_fn,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz,
			tolerance_multiple=tolerance_multiple,
			gaussian_renormalize=gaussian_renormalize,
		)
	return sce_fn

def sparse_entropy_fn(
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		oos_logprobs: th.Tensor=None,
		renormalize: bool=False,
		support_size_delta: float=0.) -> th.Tensor:

	if renormalize:
		logpartition = scatter_logsumexp(
			logprobs,
			batch_idxs
		)
		if oos_logprobs is not None:
			logpartition = th.logsumexp(th.stack([logpartition,oos_logprobs],dim=1),dim=1)
			oos_logprobs = oos_logprobs - logpartition
		logprobs = logprobs - th.gather(
			input=logpartition,
			index=batch_idxs,
			dim=0
		)
	logprobs = th.clamp(logprobs,max=0.)
	probs = th.exp(logprobs)	
	plogp = probs*logprobs
	batch_size = th.max(batch_idxs)+1
	entropy = -scatter_reduce(
		src=plogp,
		index=batch_idxs,
		reduce="sum",
		dim_size=batch_size,
	)
	assert th.min(entropy) >= -0.001, th.min(entropy)
	support_size = scatter_reduce(
		src=th.ones_like(batch_idxs,dtype=logprobs.dtype),
		index=batch_idxs,
		reduce="sum",
		dim_size=batch_size,
	)
	support_size += support_size_delta
	if oos_logprobs is not None:
		assert oos_logprobs.shape[0] == batch_size, (oos_logprobs.shape,batch_size)
		entropy = entropy - th.exp(oos_logprobs)*oos_logprobs
		support_size = support_size + 1.
	max_entropy = safelog(support_size)
	# rescale entropy
	max_entropy = th.clamp(max_entropy,min=EPS)
	entropy = th.clamp(entropy,min=th.zeros_like(max_entropy),max=max_entropy)
	norm_entropy = entropy/max_entropy
	assert not th.any(norm_entropy < 0.) and not th.any(norm_entropy > 1.)
	return entropy, norm_entropy

def sparse_conditional_entropy_fn(
		marginal_logprobs: th.Tensor,
		marginal_batch_idxs: th.Tensor,
		conditional_logprobs: th.Tensor,
		conditional_idxs: th.Tensor,
		conditional_batch_idxs: th.Tensor,
		conditional_support_size_delta: float=0.) -> th.Tensor:

	batch_size = th.max(conditional_batch_idxs)+1
	marginal_support_size = marginal_logprobs.shape[0]
	marginal_logprobs = th.clamp(marginal_logprobs,max=0.)
	conditional_logprobs = th.clamp(conditional_logprobs,max=0.)
	marginal_probs = th.exp(marginal_logprobs)
	conditional_plogp = th.exp(conditional_logprobs)*conditional_logprobs
	conditional_entropy = -scatter_reduce(
		conditional_plogp,
		index=conditional_idxs,
		reduce="sum",
		dim_size=marginal_support_size
	)
	entropy = scatter_reduce(
		marginal_probs * conditional_entropy,
		index=marginal_batch_idxs,
		reduce="sum",
		dim_size=batch_size
	)
	assert th.min(entropy) >= -0.001, th.min(entropy)
	conditional_support_size = scatter_reduce(
		th.ones_like(conditional_logprobs,dtype=marginal_logprobs.dtype),
		index=conditional_idxs,
		reduce="sum",
		dim_size=marginal_support_size
	)
	conditional_support_size += conditional_support_size_delta
	max_entropy = scatter_reduce(
		marginal_probs * th.clamp(safelog(conditional_support_size),min=0.),
		index=marginal_batch_idxs,
		reduce="sum",
		dim_size=batch_size
	)
	if th.any(max_entropy < 0.):
		import pdb; pdb.set_trace()
	# rescale entropy
	max_entropy = th.clamp(max_entropy,min=EPS)
	entropy = th.clamp(entropy,min=th.zeros_like(max_entropy),max=max_entropy)
	norm_entropy = entropy/max_entropy
	assert not th.any(norm_entropy < 0.) and not th.any(norm_entropy > 1.)
	return entropy, norm_entropy

def get_edge_loss_fn(edge_loss_fn_type: str, constant: float) -> Callable:

	if edge_loss_fn_type == "quadratic":
		edge_loss_fn = lambda x: constant*x**2
	elif edge_loss_fn_type == "linear":
		edge_loss_fn = lambda x: constant*th.abs(x)
	elif edge_loss_fn_type == "exponential":
		edge_loss_fn = lambda x: constant*th.exp(th.abs(x))
	else:
		raise NotImplementedError
	return edge_loss_fn

def sparse_cosine_distance(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		mz_max: float=1500.,
		mz_bin_res: float=0.01,
		log_distance: bool=False) -> th.Tensor:

	# sparse bin
	true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	return sparse_cosine_distance_binned(
		true_bin_idxs,
		true_bin_logprobs,
		true_bin_batch_idxs,
		pred_bin_idxs,
		pred_bin_logprobs,
		pred_bin_batch_idxs,
		log_distance=log_distance
	)

def sparse_cosine_distance_binned(
	true_bin_idxs: th.Tensor,
	true_bin_logprobs: th.Tensor,
	true_bin_batch_idxs: th.Tensor,
	pred_bin_idxs: th.Tensor,
	pred_bin_logprobs: th.Tensor,
	pred_bin_batch_idxs: th.Tensor,
	log_distance: bool=False) -> th.Tensor:

	# l2 normalize
	true_bin_logprobs = scatter_logl2normalize(
		true_bin_logprobs,
		true_bin_batch_idxs
	)
	pred_bin_logprobs = scatter_logl2normalize(
		pred_bin_logprobs,
		pred_bin_batch_idxs
	)
	# dot product
	pred_mask = th.isin(pred_bin_idxs, true_bin_idxs)
	true_mask = th.isin(true_bin_idxs, pred_bin_idxs)
	batch_size = th.max(true_bin_batch_idxs)+1
	if th.any(pred_mask):
		both_bin_logprobs = pred_bin_logprobs[pred_mask] + true_bin_logprobs[true_mask]
		assert th.all(pred_bin_batch_idxs[pred_mask] == true_bin_batch_idxs[true_mask])
		log_cos_sim = scatter_logsumexp(
			both_bin_logprobs,
			pred_bin_batch_idxs[pred_mask],
			dim_size=batch_size
		)
	else:
		# cosine similarities are all zero
		log_cos_sim = th.full(
			size=(batch_size,),
			fill_value=LOG_ZERO(pred_bin_logprobs.dtype),
			dtype=pred_bin_logprobs.dtype,
			device=pred_bin_logprobs.device
		)
		# involve pred_logprobs to keep gradient
		log_cos_sim = log_cos_sim + 0.*th.mean(pred_bin_logprobs,dim=0)
	if log_distance:
		cos_dist = th.log1p(-th.exp(log_cos_sim))
	else:
		cos_dist = 1.-th.exp(log_cos_sim)
	return cos_dist

def sparse_cosine_distance_hungarian(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		log_distance: bool=False) -> th.Tensor:
	"""
	Compute the sparse cosine distance between spectra using the Hungarian algorithm for optimal matching.

	Args:
		true_mzs: Ground truth m/z values.
		true_logprobs: Ground truth log-probabilities (intensities).
		true_batch_idxs: Batch indices for ground truth peaks.
		pred_mzs: Predicted m/z values.
		pred_logprobs: Predicted log-probabilities (intensities).
		pred_batch_idxs: Batch indices for predicted peaks.
		tolerance: m/z matching tolerance.
		relative: Whether tolerance is relative to m/z.
		tolerance_min_mz: Minimum m/z for relative tolerance.
		log_distance: If True, return log(1-cosine); else return 1-cosine.

	Returns:
		Tensor of cosine distances per batch.
	"""

	# Aggregate and round peaks for both true and predicted spectra
	true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		agg="lse"
	)
	pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		agg="lse"
	)
	# Get number of batches
	batch_size = th.max(true_batch_idxs)+1
	cos_dist_hun = th.zeros([batch_size],device=true_logprobs.device,dtype=true_logprobs.dtype)
	for batch_idx in range(batch_size):
		# Select peaks for current batch
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_mask = (pred_batch_idxs==batch_idx)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_logprobs = true_logprobs[b_true_mask]
		b_pred_logprobs = pred_logprobs[b_pred_mask]
		# L2 normalize log-probabilities for cosine similarity
		b_true_logprobs = b_true_logprobs - 0.5*th.logsumexp(2*b_true_logprobs,dim=0)
		b_pred_logprobs = b_pred_logprobs - 0.5*th.logsumexp(2*b_pred_logprobs,dim=0)
		# Compute m/z matching mask within tolerance
		b_match_mzs = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		# Identify which true and predicted peaks have matches
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)
		# Compute score matrix for matched peaks (outer sum of logprobs, exponentiated)
		b_score = th.exp(
			b_true_logprobs[b_true_match_mzs].detach().unsqueeze(1) + \
			b_pred_logprobs[b_pred_match_mzs].detach().unsqueeze(0)
		)
		# Mask out unmatched pairs with LOG_ZERO
		b_score[~b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs]] = LOG_ZERO(b_score.dtype)
		# Hungarian algorithm for optimal matching (maximizing score)
		b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score, maximize=True)
		# Compute log-cosine similarity for matched pairs
		b_log_cos_hun = th.logsumexp(
			b_true_logprobs[b_true_match_mzs][b_true_idxs] + \
			b_pred_logprobs[b_pred_match_mzs][b_pred_idxs],
			dim=0
		)
		# Convert to cosine distance
		if log_distance:
			b_cos_dist_hun = th.log1p(-th.exp(b_log_cos_hun))
		else:
			b_cos_dist_hun = 1.-th.exp(b_log_cos_hun)
		cos_dist_hun[batch_idx] = b_cos_dist_hun
	return cos_dist_hun

def sparse_jensen_shannon_divergence_hungarian(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		log_min: float=EPS) -> th.Tensor:
	"""
	Compute the sparse Jensen-Shannon divergence between spectra using the Hungarian algorithm for optimal matching.

	Args:
		true_mzs: Ground truth m/z values.
		true_logprobs: Ground truth log-probabilities (intensities).
		true_batch_idxs: Batch indices for ground truth peaks.
		pred_mzs: Predicted m/z values.
		pred_logprobs: Predicted log-probabilities (intensities).
		pred_batch_idxs: Batch indices for predicted peaks.
		tolerance: m/z matching tolerance.
		relative: Whether tolerance is relative to m/z.
		tolerance_min_mz: Minimum m/z for relative tolerance.
		log_min: Minimum log value to avoid log(0).

	Returns:
		Tensor of Jensen-Shannon divergences per batch (normalized to [0,1]).
	"""

	# Aggregate and round peaks for both true and predicted spectra
	true_mzs, true_logprobs, true_batch_idxs = round_aggregate_peaks(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		agg="lse"
	)
	pred_mzs, pred_logprobs, pred_batch_idxs = round_aggregate_peaks(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		agg="lse"
	)
	# Get number of batches
	batch_size = th.max(true_batch_idxs)+1
	jsd_hun = th.zeros([batch_size],device=true_logprobs.device,dtype=true_logprobs.dtype)
	
	for batch_idx in range(batch_size):
		# Select peaks for current batch
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_mask = (pred_batch_idxs==batch_idx)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_logprobs = true_logprobs[b_true_mask]
		b_pred_logprobs = pred_logprobs[b_pred_mask]
		
		# Normalize probabilities (sum to 1)
		b_true_probs = th.exp(b_true_logprobs)
		b_true_sum = th.sum(b_true_probs)
		if b_true_sum > 0:
			b_true_probs = b_true_probs / b_true_sum
		
		b_pred_probs = th.exp(b_pred_logprobs)
		b_pred_sum = th.sum(b_pred_probs)
		if b_pred_sum > 0:
			b_pred_probs = b_pred_probs / b_pred_sum

		# Compute m/z matching mask within tolerance
		b_match_mzs = calculate_match_mzs(
			b_true_mzs,
			b_pred_mzs,
			tolerance=tolerance,
			relative=relative,
			tolerance_min_mz=tolerance_min_mz
		)
		
		# Identify which true and predicted peaks have matches
		b_true_match_mzs = th.any(b_match_mzs,dim=1)
		b_pred_match_mzs = th.any(b_match_mzs,dim=0)
		
		# Filter to only matched peaks to reduce matrix size
		b_true_probs_m = b_true_probs[b_true_match_mzs]
		b_pred_probs_m = b_pred_probs[b_pred_match_mzs]
		
		# Compute score matrix (Gain)
		# Gain = 0.5 * [ (p+q) log(p+q) - p log p - q log q ]
		p_expand = b_true_probs_m.unsqueeze(1)
		q_expand = b_pred_probs_m.unsqueeze(0)
		sum_pq = p_expand + q_expand
		
		term1 = sum_pq * th.log(sum_pq + EPS)
		term2 = p_expand * th.log(p_expand + EPS)
		term3 = q_expand * th.log(q_expand + EPS)
		
		b_score = 0.5 * (term1 - term2 - term3)
		
		# Mask invalid matches
		valid_matches = b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs]
		# Set invalid matches to a large negative number
		b_score[~valid_matches] = -1e9
		
		# Hungarian algorithm
		if b_score.numel() == 0:
			# No matched peaks => assignment matrix is empty.
			# Keep `total_gain` connected to the computation graph so downstream
			# loss still has `requires_grad=True` (manual backward would otherwise crash).
			if b_pred_probs.numel() > 0:
				total_gain = b_pred_probs.sum() * 0.0
			elif b_true_probs.numel() > 0:
				total_gain = b_true_probs.sum() * 0.0
			else:
				total_gain = th.tensor(0.0, device=b_score.device, dtype=b_score.dtype)
		else:
			# We detach b_score for the assignment step because the Hungarian algorithm 
			# is non-differentiable (calculates indices). 
			# Gradients will flow through the values selected from the original b_score below.
			b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(b_score.detach(), maximize=True)
			
			# Filter invalid matches (where score is -1e9)
			matched_gains = b_score[b_true_idxs, b_pred_idxs]
			valid_mask = matched_gains > -1e8
			if th.any(valid_mask):
				total_gain = th.sum(matched_gains[valid_mask])
			else:
				# All matches invalid; return a graph-connected zero.
				total_gain = b_score.sum() * 0.0
			
		# Normalized JSD = 1 - Total Gain / log 2
		LOG2 = th.log(th.tensor(2.0, device=total_gain.device, dtype=total_gain.dtype))
		b_jsd = 1. - total_gain / LOG2
		b_jsd = th.clamp(b_jsd, min=0., max=1.)
		jsd_hun[batch_idx] = b_jsd
		
	return jsd_hun

def sparse_jensen_shannon_divergence(
		true_mzs: th.Tensor, 
		true_logprobs: th.Tensor,
		true_batch_idxs: th.Tensor,
		pred_mzs: th.Tensor,
		pred_logprobs: th.Tensor,
		pred_batch_idxs: th.Tensor,
		mz_max: float=1500.,
		mz_bin_res: float=0.01,
		log_min: float=EPS) -> th.Tensor:
	"""
	Compute the sparse Jensen-Shannon divergence between two sets of spectra.

	Args:
		true_mzs: Ground truth m/z values.
		true_logprobs: Ground truth log-probabilities (intensities).
		true_batch_idxs: Batch indices for ground truth peaks.
		pred_mzs: Predicted m/z values.
		pred_logprobs: Predicted log-probabilities (intensities).
		pred_batch_idxs: Batch indices for predicted peaks.
		mz_max: Maximum m/z value for binning.
		mz_bin_res: Resolution of m/z bins.
		log_min: Minimum log value to avoid log(0).

	Returns:
		Jensen-Shannon divergence (distance) between the binned spectra, per batch.
	"""

	# Bin the ground truth peaks into sparse bins using batched_bin_func
	true_bin_idxs, true_bin_logprobs, true_bin_batch_idxs = batched_bin_func(
		true_mzs,
		true_logprobs,
		true_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",      # Use log-sum-exp aggregation for intensities
		sparse=True     # Use sparse representation for efficiency
	)
	# Bin the predicted peaks into sparse bins
	pred_bin_idxs, pred_bin_logprobs, pred_bin_batch_idxs = batched_bin_func(
		pred_mzs,
		pred_logprobs,
		pred_batch_idxs,
		mz_max=mz_max,
		mz_bin_res=mz_bin_res,
		agg="lse",
		sparse=True
	)
	# Calculate Jensen-Shannon similarity using the helper function
	jss = jss_helper(
		true_bin_idxs,
		true_bin_logprobs.exp(),   # Convert log-probabilities to probabilities
		true_bin_batch_idxs,
		pred_bin_idxs,
		pred_bin_logprobs.exp(),
		pred_bin_batch_idxs,
		log_min=log_min
	)
	# Return 1 - similarity to get the divergence (distance)
	return 1.-jss

def get_pairwise_cossim(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		mz_max: float,
		mz_bin_res: float,
		chunk_size: int = 32) -> th.Tensor:
	"""
	Compute pairwise cosine similarity matrix for a batch of spectra.

	Args:
		mzs: m/z values.
		logprobs: log-probabilities (intensities).
		batch_idxs: Batch indices.
		batch_size: Number of spectra in the batch.
		mz_max: Maximum m/z value.
		mz_bin_res: m/z bin resolution.
		chunk_size: Batch size for chunked computation to save memory.

	Returns:
		Pairwise similarity matrix of shape [batch_size, batch_size].
	"""
	
	if mzs.numel() == 0:
		return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

	# Binning
	bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
		mzs, logprobs, batch_idxs, 
		mz_max=mz_max, 
		mz_bin_res=mz_bin_res, 
		agg="lse", 
		sparse=True
	)
 
	# Normalize (L2 on probabilities)
	norm_logprobs = scatter_logl2normalize(bin_logprobs, bin_batch_idxs)
	norm_probs = norm_logprobs.exp()
	
	# Construct sparse tensor then dense
	# We need to recover local bin indices from global bin indices returned by batched_bin_func
	# batched_bin_func uses global_idx = batch_idx * num_bins + local_idx
	# So local_idx = global_idx % num_bins
	# We must ensure num_bins matches exactly what batched_bin_func used.
	num_bins = th.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res,device=mzs.device,dtype=mzs.dtype).shape[0]
	
	indices = th.stack([bin_batch_idxs, bin_idxs % num_bins])
	
	V = th.sparse_coo_tensor(indices, norm_probs, (batch_size, num_bins), device=mzs.device)
	V_dense = V.to_dense()
	
	if chunk_size is None or chunk_size >= batch_size:
		# Compute pairwise similarity
		S = th.mm(V_dense, V_dense.t())
	else:
		S = th.zeros((batch_size, batch_size), device=mzs.device, dtype=V_dense.dtype)
		for i in range(0, batch_size, chunk_size):
			i_end = min(i + chunk_size, batch_size)
			V_i = V_dense[i:i_end]
			for j in range(0, batch_size, chunk_size):
				j_end = min(j + chunk_size, batch_size)
				V_j = V_dense[j:j_end]
				S[i:i_end, j:j_end] = th.mm(V_i, V_j.t())

	return S

def get_pairwise_jss_sim(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		mz_max: float,
		mz_bin_res: float,
		chunk_size: int = 32) -> th.Tensor:
	"""
	Compute pairwise Jensen-Shannon similarity matrix for a batch of spectra.

	Args:
		mzs: m/z values.
		logprobs: log-probabilities (intensities).
		batch_idxs: Batch indices.
		batch_size: Number of spectra in the batch.
		mz_max: Maximum m/z value.
		mz_bin_res: m/z bin resolution.
		chunk_size: Batch size for chunked computation to save memory.

	Returns:
		Pairwise similarity matrix of shape [batch_size, batch_size].
	"""
	
	if mzs.numel() == 0:
		return th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)

	# Binning
	bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
		mzs, logprobs, batch_idxs, 
		mz_max=mz_max, 
		mz_bin_res=mz_bin_res, 
		agg="lse", 
		sparse=True
	)
 
	# Normalize (L1 on probabilities)
	# log_softmax to get log probabilities
	log_probs = scatter_logsoftmax(bin_logprobs, bin_batch_idxs)
	probs = log_probs.exp()
	
	# Construct sparse tensor then dense
	num_bins = th.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res,device=mzs.device,dtype=mzs.dtype).shape[0]
	
	indices = th.stack([bin_batch_idxs, bin_idxs % num_bins])
	
	V = th.sparse_coo_tensor(indices, probs, (batch_size, num_bins), device=mzs.device)
	V_dense = V.to_dense()  # (B, D)
	
	# Compute JSS
	# JSD(P, Q) = H(M) - 0.5 H(P) - 0.5 H(Q)
	# M = 0.5 (P + Q)
	P = V_dense
	
	# H(P)
	H_P = -th.sum(P * th.log(P + EPS), dim=1)  # (B,)
	
	# H(M)
	H_M = th.zeros((batch_size, batch_size), device=mzs.device, dtype=P.dtype)
	if chunk_size is None or chunk_size >= batch_size:
		for i in range(batch_size):
			M_i = 0.5 * (P[i].unsqueeze(0) + P)
			H_M[i] = -th.sum(M_i * th.log(M_i + EPS), dim=1)
	else:
		for i in range(0, batch_size, chunk_size):
			i_end = min(i + chunk_size, batch_size)
			P_i = P[i:i_end]  # (chunk_i, D)
			for j in range(0, batch_size, chunk_size):
				j_end = min(j + chunk_size, batch_size)
				P_j = P[j:j_end]  # (chunk_j, D)
				M_ij = 0.5 * (P_i.unsqueeze(1) + P_j.unsqueeze(0))
				H_M[i:i_end, j:j_end] = -th.sum(M_ij * th.log(M_ij + EPS), dim=2)
	
	JSD = H_M - 0.5 * H_P.unsqueeze(1) - 0.5 * H_P.unsqueeze(0)
	JSS = 1.0 - JSD / math.log(2.0)
	
	return JSS

def clip_loss(spectrum_embeddings, mol_embeddings, temperature=0.07):
	"""
	Compute the CLIP contrastive loss between spectrum and molecule embeddings.
	
	Args:
		spectrum_embeddings: torch.Tensor of shape [batch_size, embedding_dim]
		mol_embeddings: torch.Tensor of shape [batch_size, embedding_dim]
		temperature: scalar, temsperature parameter for scaling logits

	Returns:
		loss: scalar, the symmetric contrastive loss
	"""

	# Normalize embeddings to unit length
	spectrum_embeddings = spectrum_embeddings / spectrum_embeddings.norm(dim=1, keepdim=True)
	mol_embeddings = mol_embeddings / mol_embeddings.norm(dim=1, keepdim=True)

	# Compute cosine similarity matrix (logits)
	# this is equivalent to computing the cosine similarity for each pair. 
	logits_per_specturm = spectrum_embeddings @ mol_embeddings.t() / temperature  # [batch_size, batch_size]
	logits_per_mol = logits_per_specturm.t()  # [batch_size, batch_size]
	
	# Target labels are the indices of the diagonal
	batch_size = spectrum_embeddings.size(0)
	labels = th.arange(batch_size).to(spectrum_embeddings.device)
	
	# Compute cross-entropy loss in both directions
	loss_i2t = F.cross_entropy(logits_per_specturm, labels)
	loss_t2i = F.cross_entropy(logits_per_mol, labels)
	loss = (loss_i2t + loss_t2i) / 2
	return loss

def get_pairwise_binned_cross_entropy(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		mz_max: float,
		mz_bin_res: float,
		chunk_size: int = 32) -> th.Tensor:
	"""
	Compute pairwise cross-entropy matrix for a batch of spectra.
	Cross-Entropy(P, Q) = - sum(P * log(Q))

	Args:
		mzs: m/z values.
		logprobs: log-probabilities (intensities).
		batch_idxs: Batch indices.
		batch_size: Number of spectra in the batch.
		mz_max: Maximum m/z value.
		mz_bin_res: m/z bin resolution.
		chunk_size: Batch size for chunked computation to save memory.

	Returns:
		Pairwise cross-entropy matrix of shape [batch_size, batch_size].
	"""
	
	if mzs.numel() == 0:
		return th.zeros((batch_size, batch_size), device=mzs.device)

	# Binning
	bin_idxs, bin_logprobs, bin_batch_idxs = batched_bin_func(
		mzs, logprobs, batch_idxs, 
		mz_max=mz_max, 
		mz_bin_res=mz_bin_res, 
		agg="lse", 
		sparse=True
	)
 
	# Normalize to get probabilities (L1 normalization)
	# log_softmax to get log probabilities properly normalized
	log_probs = scatter_logsoftmax(bin_logprobs, bin_batch_idxs)
	probs = log_probs.exp()
	
	# Construct sparse tensor then dense
	num_bins = th.arange(mz_bin_res, mz_max+mz_bin_res, step=mz_bin_res, device=mzs.device, dtype=mzs.dtype).shape[0]
	
	indices = th.stack([bin_batch_idxs, bin_idxs % num_bins])
	
	# Create sparse tensor for Probs
	P = th.sparse_coo_tensor(indices, probs, (batch_size, num_bins), device=mzs.device)
	
	# We also need LogProbs for the Q part: log(Q)
	# We use a small epsilon for stability when Q is 0, effective log(Q+eps)
	# But since we use sparse representation, 0s are implicit.
	# For CE = - sum P_i * log Q_i
	# If Q_i is 0 (missing peak), log Q_i is -inf (or small number). 
	# P_i * log Q_i will be large if P_i > 0.
	# This simple matrix multiplication approach usually assumes 0 * anything = 0, 
	# but here we specifically want P_i * log(0) to be penalized.
	#
	# Strategy:
	# 1. Convert P to dense.
	# 2. Convert Q's logprobs to dense, filling missing values with a very small log-prob (e.g. log(epsilon)).
	#    However, batched_bin_func returns only present bins.
	#    If we just densify P, we get 0s for missing bins. 
	#    P_dense corresponds to the probabilities.
	#    Calculated CE(i, j) = - sum_k P_ik * log(Q_jk)
	
	if chunk_size is None or chunk_size >= batch_size:
		P_dense = P.to_dense() # (B, D)
		
		# Q_dense needs to be log probabilities.
		# Note: P.to_dense() fills missing values with 0. 
		# For Log Probabilities, missing values should be LOG_ZERO or log(eps).
		# We can reconstruct Q_log_dense from the same indices but using log_probs values.
		Q_log_sparse = th.sparse_coo_tensor(indices, log_probs, (batch_size, num_bins), device=mzs.device)
		Q_log_dense = Q_log_sparse.to_dense()
		
		# Now, the zeros in Q_log_dense are actually "missing" values which should be LOG_ZERO.
		# But wait, sparse_coo_tensor.to_dense() fills with 0. 0 in log-space is prob=1, which is WRONG.
		# We need to initialize with a very small value (LOG_ZERO) and then fill in the values.
		# Or, simpler: convert, then mask where P (probs) was 0? No, we need where Q was 0.
		
		# Better approach for Q:
		# 1. Start with full matrix of LOG_ZERO/log(eps).
		# 2. Add the sparse values.
		# However, `to_dense()` doesn't support "fill value".
		
		# Workaround: 
		# Q_log_dense = Q_log_sparse.to_dense() gives 0s where there are no entries.
		# We know matching entries are in indices.
		# Alternatively: P_dense is easy (0 probs is fine).
		# For Q, we want `log(Q + eps)`.
		# So we can just take P_dense (which is Q_dense probs), add eps, and log.
		# Q_log_dense = th.log(P_dense + EPS)
		
		Q_log_dense = th.log(P_dense + EPS)
		
		# CE = - P @ Q_log.T
		CE = - th.mm(P_dense, Q_log_dense.t())
		
	else:
		CE = th.zeros((batch_size, batch_size), device=mzs.device, dtype=mzs.dtype)
		
		# Pre-computation can be expensive if we densify everything.
		# But we can chunk.
		
		for i in range(0, batch_size, chunk_size):
			i_end = min(i + chunk_size, batch_size)
			i_idxs = th.arange(i, i_end, device=mzs.device)
			P_i = th.index_select(P, 0, i_idxs).to_dense() # (chunk_i, D)
			
			for j in range(0, batch_size, chunk_size):
				j_end = min(j + chunk_size, batch_size)
				j_idxs = th.arange(j, j_end, device=mzs.device)
				P_j = th.index_select(P, 0, j_idxs).to_dense() # (chunk_j, D)
				
				# Q_j_log
				Q_j_log = th.log(P_j + EPS)
				
				# CE[i, j] = - sum P_i * log Q_j
				CE[i:i_end, j:j_end] = - th.mm(P_i, Q_j_log.t())

	return CE

def get_pairwise_sparse_jss_sim_hun(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ) -> th.Tensor:
	"""
	Compute pairwise Jensen-Shannon Similarity matrix for a batch of spectra using Hungarian matching.
	Sim = 1 - JSD.
	
	Args:
		mzs: m/z values.
		logprobs: log-probabilities (intensities).
		batch_idxs: Batch indices.
		batch_size: Number of spectra.
		tolerance: m/z matching tolerance.
		relative: Whether tolerance is relative.
		tolerance_min_mz: Minimum m/z for relative tolerance.
		
	Returns:
		Matrix of shape [batch_size, batch_size] containing pairwise similarities.
	"""
	
	# Pre-aggregate peaks to save work in loop
	mzs, logprobs, batch_idxs = round_aggregate_peaks(
		mzs,
		logprobs,
		batch_idxs,
		agg="lse"
	)
	
	sim_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)
	
	# Separate spectra
	spectra_mzs = []
	spectra_logprobs = []
	for i in range(batch_size):
		mask = (batch_idxs == i)
		spectra_mzs.append(mzs[mask])
		spectra_logprobs.append(logprobs[mask])
		
	for i in range(batch_size):
		# Self similarity is always 1.
		sim_matrix[i, i] = 1.0
		
		for j in range(i + 1, batch_size):
			jsd = _compute_pair_jss_hun(
				spectra_mzs[i], spectra_logprobs[i],
				spectra_mzs[j], spectra_logprobs[j],
				tolerance, relative, tolerance_min_mz
			)
			sim = 1.0 - jsd
			sim_matrix[i, j] = sim
			sim_matrix[j, i] = sim
			
	return sim_matrix


def _compute_pair_ce_hun(
	b_true_mzs: th.Tensor,
	b_true_logprobs: th.Tensor,
	b_pred_mzs: th.Tensor,
	b_pred_logprobs: th.Tensor,
	tolerance: float,
	relative: bool,
	tolerance_min_mz: float
) -> th.Tensor:
	"""
	Helper to compute Cross Entropy for a single pair of spectra using Hungarian matching.
	CE(P, Q) = - sum P_i log Q_matching_i
	If no match, log Q is effectively log(epsilon).
	Returns scalar tensor.
	"""
	# Normalize probabilities (sum to 1)
	b_true_probs = th.exp(b_true_logprobs)
	b_true_sum = th.sum(b_true_probs)
	if b_true_sum > 0:
		b_true_probs = b_true_probs / b_true_sum
	
	b_pred_probs = th.exp(b_pred_logprobs)
	b_pred_sum = th.sum(b_pred_probs)
	if b_pred_sum > 0:
		b_pred_probs = b_pred_probs / b_pred_sum
	
	# If P is empty (sum=0), CE = 0 (convention, 0 * log(Q))
	if b_true_sum == 0:
		return th.tensor(0.0, device=b_true_mzs.device, dtype=b_true_logprobs.dtype)

	# Q needs to be in log space for CE calculation.
	# But we need log(Q) where Q matches P.
	# Base contribution (if P_i has no match in Q): P_i * log(epsilon)
	# Gain of matching P_i to Q_j: P_i * log(Q_j) - P_i * log(epsilon) = P_i * (log(Q_j) - log(epsilon))
	# We want to MAXIMIZE sum(P_i * log(Q_matching_i)).
	
	log_eps = th.log(th.tensor(EPS, device=b_true_mzs.device, dtype=b_true_logprobs.dtype))
	
	# If Q is empty (sum=0), all matches are missing.
	# CE = - sum P_i * log(eps) = - 1.0 * log(eps) = -log(eps) (since P is normalized)
	if b_pred_sum == 0:
		return -log_eps

	# Compute m/z matching mask within tolerance
	b_match_mzs = calculate_match_mzs(
		b_true_mzs,
		b_pred_mzs,
		tolerance=tolerance,
		relative=relative,
		tolerance_min_mz=tolerance_min_mz
	)
	
	# Identify which true and predicted peaks have matches
	b_true_match_mzs = th.any(b_match_mzs,dim=1)
	b_pred_match_mzs = th.any(b_match_mzs,dim=0)
	
	# Filter to only matched peaks
	b_true_probs_m = b_true_probs[b_true_match_mzs]
	b_pred_probs_m = b_pred_probs[b_pred_match_mzs]
	
	# Compute Score (Gain) Matrix
	# Gain(i, j) = P_i * (log(Q_j) - log(eps))
	# Note: Q_j are probabilities.
	# We use log(Q_j + eps) to avoid log(0) if Q_j is very small, though here Q is normalized props.
	q_log_m = th.log(b_pred_probs_m + EPS)
	
	# Outer product logic: P_i * (log Q_j - log_eps)
	# P_i is (N,). log_Q_j is (M,).
	# Gain is (N, M).
	gain_matrix = b_true_probs_m.unsqueeze(1) * (q_log_m.unsqueeze(0) - log_eps)
	
	# Mask invalid matches
	valid_matches = b_match_mzs[b_true_match_mzs][:,b_pred_match_mzs]
	
	# Set invalid matches to a large negative number
	# Gain can be negative if log(Q) < log(eps), but unlikely given eps is small.
	# But to prevent selection, use -infinity.
	gain_matrix[~valid_matches] = -1e9
	
	# Hungarian algorithm
	if gain_matrix.numel() == 0:
		total_gain = th.tensor(0.0, device=gain_matrix.device, dtype=gain_matrix.dtype)
	else:
		b_true_idxs, b_pred_idxs = scipy_linear_sum_assignment(gain_matrix.detach(), maximize=True)
		
		matched_gains = gain_matrix[b_true_idxs, b_pred_idxs]
		valid_mask = matched_gains > -1e8
		if th.any(valid_mask):
			total_gain = th.sum(matched_gains[valid_mask])
		else:
			total_gain = th.tensor(0.0, device=gain_matrix.device, dtype=gain_matrix.dtype)
			
	# Total Score (sum P log Q) = Total Gain + Base Score
	# Base Score = sum_i P_i * log(eps) = 1.0 * log(eps) = log(eps)
	total_score = total_gain + log_eps
	
	# CE = - Total Score
	ce = -total_score
	return ce


def get_pairwise_cross_entropy_hun(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		chunk_size: int=32) -> th.Tensor:
	"""
	Compute pairwise Cross Entropy matrix for a batch of spectra using Hungarian matching.
	CE(P, Q) = - sum P_i log Q_matching_i.

	Args:
		mzs: m/z values.
		logprobs: log-probabilities (intensities).
		batch_idxs: Batch indices.
		batch_size: Number of spectra.
		tolerance: m/z matching tolerance.
		relative: Whether tolerance is relative.
		tolerance_min_mz: Minimum m/z for relative tolerance.
		chunk_size: Ignored in this iterative implementation, but kept for API compatibility.
		
	Returns:
		Matrix of shape [batch_size, batch_size] containing pairwise cross entropies.
	"""
	
	# Pre-aggregate peaks to save work in loop
	mzs, logprobs, batch_idxs = round_aggregate_peaks(
		mzs,
		logprobs,
		batch_idxs,
		agg="lse"
	)
	
	ce_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)
	
	# Separate spectra
	spectra_mzs = []
	spectra_logprobs = []
	for i in range(batch_size):
		mask = (batch_idxs == i)
		spectra_mzs.append(mzs[mask])
		spectra_logprobs.append(logprobs[mask])
		
	# Fill matrix
	# CE is NOT symmetric. Loop over all pairs.
	# Iterate over all i, j.
	for i in range(batch_size):
		for j in range(batch_size):
			ce_matrix[i, j] = _compute_pair_ce_hun(
				spectra_mzs[i], spectra_logprobs[i],
				spectra_mzs[j], spectra_logprobs[j],
				tolerance, relative, tolerance_min_mz
			)
			# Optimization: could skip self-check if needed, but good for validation.
			
	return ce_matrix

def get_pairwise_cross_entropy_vec(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		dist: str="gaussian",
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		chunk_size: int=32) -> th.Tensor:
	"""
	Compute pairwise Cross Entropy matrix for a batch of spectra using vectorized "soft" matching.
	CE(P, Q) = - sum P_i log Q_soft(m_i)
	
	Args:
		mzs: m/z values.
		logprobs: log-probabilities (intensities).
		batch_idxs: Batch indices.
		batch_size: Number of spectra.
		dist: "gaussian" or "peak_marginal".
		tolerance: m/z matching tolerance.
		relative: Whether tolerance is relative.
		tolerance_min_mz: Minimum m/z for relative tolerance.
		chunk_size: Processing chunk size.
		
	Returns:
		Matrix of shape [batch_size, batch_size] containing pairwise cross entropies.
	"""
	
	# Pre-aggregate peaks
	mzs, logprobs, batch_idxs = round_aggregate_peaks(
		mzs,
		logprobs,
		batch_idxs,
		agg="lse"
	)
	
	if dist == "gaussian":
		ce_fn = mog_ce_fn
		tolerance_multiple = 3.0 # Standard Gaussian truncation
		gaussian_renormalize = True
	elif dist == "peak_marginal":
		ce_fn = pm_ce_fn
		tolerance_multiple = 1.0
		gaussian_renormalize = False
	else:
		raise ValueError(f"Unknown dist: {dist}")
		
	ce_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)
	
	spectra_mzs = []
	spectra_logprobs = []
	for i in range(batch_size):
		mask = (batch_idxs == i)
		spectra_mzs.append(mzs[mask])
		spectra_logprobs.append(logprobs[mask])

	# Normalize probabilities
	spectra_probs = []
	for i in range(batch_size):
		p = th.exp(spectra_logprobs[i])
		p_sum = p.sum()
		if p_sum > 0:
			spectra_probs.append(p / p_sum)
		else:
			spectra_probs.append(p)

	# Iterative computation
	for i in range(batch_size):
		p_mzs = spectra_mzs[i]
		p_probs = spectra_probs[i]
		
		# Skip empty spectra
		if p_probs.numel() == 0:
			continue
			
		for j in range(batch_size):
			q_mzs = spectra_mzs[j]
			q_logprobs = spectra_logprobs[j]
			
			if q_logprobs.numel() == 0:
				# Q is empty. CE = - sum P log(eps) = - log(eps)
				ce_matrix[i, j] = -th.log(th.tensor(EPS, device=mzs.device, dtype=logprobs.dtype))
				continue
				
			# Compute Q_soft(m_p) -> logprobs of P_mzs under Q distribution
			# ce_fn(b_true_mzs=P, b_pred_mzs=Q, b_pred_logprobs=Q ...)
			# Note: we pass unnormalized q_logprobs because ce_fn usually expects raw weights,
			# but strictly speaking for proper probability density, Q should also be normalized.
			# Let's normalize Q logprobs for consistency with probabilistic interpretation.
			q_probs_unnorm = th.exp(q_logprobs)
			q_sum = q_probs_unnorm.sum()
			if q_sum > 0:
				current_q_logprobs = q_logprobs - th.log(q_sum)
			else:
				current_q_logprobs = q_logprobs

			q_soft_logprobs = ce_fn(
				b_true_mzs=p_mzs,
				b_pred_mzs=q_mzs,
				b_pred_logprobs=current_q_logprobs,
				relative=relative,
				tolerance=tolerance,
				tolerance_min_mz=tolerance_min_mz,
				tolerance_multiple=tolerance_multiple,
				gaussian_renormalize=gaussian_renormalize
			)
			
			# Q_soft_logprobs contains log probability of observing p_mzs given distribution Q.
			# However, ce_fn (esp. mog_ce_fn) calculates a density (if renormalized) or just a weighted sum.
			# If it's density, values can be > 0 (log > 0).
			# For discrete cross-entropy, P are probabilities summing to 1.
			# CE = - sum P_k * log(Likelihood_k)
			
			ce = -th.sum(p_probs * q_soft_logprobs)
			ce_matrix[i, j] = ce
			
	return ce_matrix

def get_pairwise_jss_vec(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		dist: str="gaussian",
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		chunk_size: int=32) -> th.Tensor:
	"""
	Compute pairwise Jensen-Shannon Similarity matrix using vectorized soft matching.
	JSS = 1 - JSD / log(2)
	JSD(P, Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
	M = 0.5 * (P + Q)
	
	Args: same as get_pairwise_cross_entropy_vec
	"""
	
	# Pre-aggregate peaks
	mzs, logprobs, batch_idxs = round_aggregate_peaks(
		mzs,
		logprobs,
		batch_idxs,
		agg="lse"
	)
	
	if dist == "gaussian":
		ce_fn = mog_ce_fn
		tolerance_multiple = 3.0
		gaussian_renormalize = True
	elif dist == "peak_marginal":
		ce_fn = pm_ce_fn
		tolerance_multiple = 1.0
		gaussian_renormalize = False
	else:
		raise ValueError(f"Unknown dist: {dist}")
	
	sim_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)
	
	spectra_mzs = []
	spectra_logprobs = []
	spectra_probs = []
	
	for i in range(batch_size):
		mask = (batch_idxs == i)
		s_mzs = mzs[mask]
		s_logprobs = logprobs[mask]
		spectra_mzs.append(s_mzs)
		spectra_logprobs.append(s_logprobs)
		
		# Normalize
		p = th.exp(s_logprobs)
		p_sum = p.sum()
		if p_sum > 0:
			spectra_probs.append(p / p_sum)
		else:
			spectra_probs.append(p)
			
	LOG2 = th.log(th.tensor(2.0, device=mzs.device, dtype=logprobs.dtype))
	
	for i in range(batch_size):
		# Self-similarity
		sim_matrix[i, i] = 1.0
		
		p_mzs = spectra_mzs[i]
		p_probs = spectra_probs[i]
		
		for j in range(i + 1, batch_size):
			q_mzs = spectra_mzs[j]
			q_probs = spectra_probs[j]
			
			if p_probs.numel() == 0 or q_probs.numel() == 0:
				# If one is empty, JSS=0 (max distance)
				sim_matrix[i, j] = 0.0
				sim_matrix[j, i] = 0.0
				continue
				
			# Prepare normalized logprobs for ce_fn
			p_logprobs_norm = th.log(p_probs + 1e-30) # Avoid log(0) issues if any
			q_logprobs_norm = th.log(q_probs + 1e-30)
			
			# We need M = 0.5(P+Q).
			# But P and Q are continuous distributions defined by (mzs, probs).
			# Evaluating KL(P || M) requires evaluating P and M at locations of P.
			# P(x) is discrete spikes at p_mzs.
			# M(x) = 0.5 P(x) + 0.5 Q(x).
			# At location x \in p_mzs:
			# P(x) = p_probs
			# Q(x) = q_soft(x) (estimated via ce_fn)
			
			# Q at P locations
			q_at_p_log = ce_fn(
				b_true_mzs=p_mzs,
				b_pred_mzs=q_mzs,
				b_pred_logprobs=q_logprobs_norm,
				relative=relative,
				tolerance=tolerance,
				tolerance_min_mz=tolerance_min_mz,
				tolerance_multiple=tolerance_multiple,
				gaussian_renormalize=gaussian_renormalize
			)
			q_at_p = th.exp(q_at_p_log)
			
			# P at Q locations
			p_at_q_log = ce_fn(
				b_true_mzs=q_mzs,
				b_pred_mzs=p_mzs,
				b_pred_logprobs=p_logprobs_norm,
				relative=relative,
				tolerance=tolerance,
				tolerance_min_mz=tolerance_min_mz,
				tolerance_multiple=tolerance_multiple,
				gaussian_renormalize=gaussian_renormalize
			)
			p_at_q = th.exp(p_at_q_log)
			
			# M at P locations = 0.5 * P + 0.5 * Q_at_P
			# Note: P here is probability mass. Q_at_P is density or mass depending on dist.
			# If dist=gaussian, Q_at_P is density. P is mass.
			# This is a known issue with discrete-continuous mixture.
			# However, for similarity metric, we treat them as compatible.
			# Generally peak_marginal is better for this as it is mass-like (box kernel).
			m_at_p = 0.5 * p_probs + 0.5 * q_at_p
			
			# M at Q locations
			m_at_q = 0.5 * p_at_q + 0.5 * q_probs
			
			# KL(P || M) = sum P * log(P / M_at_P)
			kl_p_m = th.sum(p_probs * (th.log(p_probs + EPS) - th.log(m_at_p + EPS)))
			
			# KL(Q || M) = sum Q * log(Q / M_at_q)
			kl_q_m = th.sum(q_probs * (th.log(q_probs + EPS) - th.log(m_at_q + EPS)))
			
			jsd = 0.5 * kl_p_m + 0.5 * kl_q_m
			
			sim = 1.0 - jsd / LOG2
			
			# Clamp for numerical stability
			sim = th.clamp(sim, 0.0, 1.0)
			
			sim_matrix[i, j] = sim
			sim_matrix[j, i] = sim
			
	return sim_matrix

def get_pairwise_cossim_vec(
		mzs: th.Tensor,
		logprobs: th.Tensor,
		batch_idxs: th.Tensor,
		batch_size: int,
		dist: str="gaussian",
		tolerance: float=1e-5,
		relative: bool=True,
		tolerance_min_mz: float=TOLERANCE_MIN_MZ,
		chunk_size: int=32) -> th.Tensor:
	"""
	Compute pairwise Cosine Similarity matrix using vectorized soft matching.
	Sim(P, Q) = 0.5 * ( P . Q_soft + Q . P_soft )
	Inputs are assumed to be unnormalized logprobs, will be L2 normalized internally.
	"""
	
	# Pre-aggregate peaks
	mzs, logprobs, batch_idxs = round_aggregate_peaks(
		mzs,
		logprobs,
		batch_idxs,
		agg="lse"
	)
	
	if dist == "gaussian":
		ce_fn = mog_ce_fn
		tolerance_multiple = 3.0
		gaussian_renormalize = True
	elif dist == "peak_marginal":
		ce_fn = pm_ce_fn
		tolerance_multiple = 1.0
		gaussian_renormalize = False
	else:
		raise ValueError(f"Unknown dist: {dist}")
		
	sim_matrix = th.zeros((batch_size, batch_size), device=mzs.device, dtype=logprobs.dtype)
	
	spectra_mzs = []
	spectra_l2probs = [] # L2 normalized probability-like vectors
	
	for i in range(batch_size):
		mask = (batch_idxs == i)
		s_mzs = mzs[mask]
		s_logprobs = logprobs[mask]
		spectra_mzs.append(s_mzs)
		
		# L2 Normalize: x / ||x||_2.
		# Operations in log space for stability?
		# log(x / ||x||) = log(x) - log(||x||)
		# ||x|| = sqrt(sum x^2) = sqrt(sum exp(2*logx))
		# log_norm = 0.5 * logsumexp(2 * logx)
		log_norm = 0.5 * th.logsumexp(2 * s_logprobs, dim=0)
		s_logprobs_l2 = s_logprobs - log_norm
		spectra_l2probs.append(th.exp(s_logprobs_l2))

	for i in range(batch_size):
		sim_matrix[i, i] = 1.0
		
		p_mzs = spectra_mzs[i]
		p_vec = spectra_l2probs[i]
		p_logvec = th.log(p_vec + 1e-30) # For passing to ce_fn
		
		for j in range(i + 1, batch_size):
			q_mzs = spectra_mzs[j]
			q_vec = spectra_l2probs[j]
			q_logvec = th.log(q_vec + 1e-30)
			
			if p_vec.numel() == 0 or q_vec.numel() == 0:
				sim_matrix[i, j] = 0.0
				sim_matrix[j, i] = 0.0
				continue
				
			# Q at P locations (soft projection of Q onto P's basis)
			# Result is weighted sum of Q's vec at P's locations.
			# But ce_fn returns log-prob.
			# Here we interpret q_logvec as "weights" in the MOG.
			# ce_fn calculates: logsumexp( -dist + log_weight )
			# Exponentiated: sum( weight * kernel )
			# This is exactly the convolution / soft projection.
			
			q_at_p_log = ce_fn(
				b_true_mzs=p_mzs,
				b_pred_mzs=q_mzs,
				b_pred_logprobs=q_logvec,
				relative=relative,
				tolerance=tolerance,
				tolerance_min_mz=tolerance_min_mz,
				tolerance_multiple=tolerance_multiple,
				gaussian_renormalize=gaussian_renormalize
			)
			q_at_p = th.exp(q_at_p_log)
			
			# P at Q locations
			p_at_q_log = ce_fn(
				b_true_mzs=q_mzs,
				b_pred_mzs=p_mzs,
				b_pred_logprobs=p_logvec,
				relative=relative,
				tolerance=tolerance,
				tolerance_min_mz=tolerance_min_mz,
				tolerance_multiple=tolerance_multiple,
				gaussian_renormalize=gaussian_renormalize
			)
			p_at_q = th.exp(p_at_q_log)
			
			# Dot product
			# P . Q_soft = sum(p_vec * q_at_p)
			dot_pq = th.sum(p_vec * q_at_p)
			
			# Q . P_soft = sum(q_vec * p_at_q)
			dot_qp = th.sum(q_vec * p_at_q)
			
			# Symmetric sim
			sim = 0.5 * (dot_pq + dot_qp)
			
			# Clamp
			sim = th.clamp(sim, 0.0, 1.0)
			
			sim_matrix[i, j] = sim
			sim_matrix[j, i] = sim
			
	return sim_matrix