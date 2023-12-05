import numpy as np
import torch as th
import torch.nn.functional as F
import math

from frag_gnn.utils.misc_utils import scatter_reduce, EPS, safelog, scatter_logsumexp, LOG_ZERO
from frag_gnn.utils.spec_utils import calculate_match_mzs


def mog_log_prob(samples,means,variances,log_weights):
	""" Mixture of Gaussian
		We are modeling each peaks are mixture of guassian for given set of fragments
	Args:
		samples (_type_): b_true_mz,
		means (_type_): b_pred_mzs
		variances (_type_): b_variances
		log_weights (_type_): b_pred_logprobs

	Returns:
		_type_: _description_
	"""
	
	samples = samples.reshape(-1,1)
	means = means.reshape(1,-1)
	variances = variances.reshape(1,-1)
	log_weights = log_weights.reshape(1,-1)
	# [B,K]
	normal_log_probs = -0.5*(samples-means)**2/variances - 0.5*th.log(2*th.pi*variances)
	# [B]
	log_probs = th.logsumexp(normal_log_probs + log_weights, dim=1)
	return log_probs


def momu_log_prob(samples,means,variances,log_weights,tolerances=1e-3):
	"""
	Mixture of mollified unfiform distributions (normal approximation to uniform)
	https://stats.stackexchange.com/questions/572557/how-to-sample-and-compute-the-likelihood-from-a-mollified-uniform-distribution
	https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.cdf
	
	This is pretty slow (due to erf) and does not seem to work very well.
	"""

	samples = samples.reshape(-1,1)
	means = means.reshape(1,-1)
	inv_variances = variances.reshape(1,-1).reciprocal()
	log_weights = log_weights.reshape(1,-1)
	# [B,K]
	v1 = (means+tolerances-samples)*inv_variances
	v2 = (means-tolerances-samples)*inv_variances
	mol_normal_log_probs = th.log(
		0.5*(1.+th.erf((v1-means)+inv_variances/math.sqrt(2))) \
		-0.5*(1.+th.erf((v2-means)+inv_variances/math.sqrt(2))) \
		+1e-9
	)
	# [B]
	log_probs = th.logsumexp(mol_normal_log_probs + log_weights, dim=1)
	return log_probs


def sparse_cross_entropy_vec(true_mzs,true_logprobs,true_batch_idxs,pred_mzs,pred_logprobs,pred_batch_idxs,log_prob_fn=mog_log_prob,tolerance=1e-3,relative=False):
	"""
	this is just for th.compile
	"""

	mask_value = LOG_ZERO
	batch_size = th.max(true_batch_idxs)+1
	def vec_ce(true_batch_idx,true_logprob,true_mz):
		b_pred_mask = (pred_batch_idxs==true_batch_idx).float()
		b_pred_logprobs = pred_logprobs*b_pred_mask + mask_value*(1.-b_pred_mask)
		b_pred_mzs = pred_mzs
		if relative:
			b_variances = (th.clamp(b_pred_mzs,min=200.)*tolerance)**2
		else:
			b_variances = (tolerance**2)*th.ones_like(b_pred_mzs)
		b_true_logprob = true_logprob
		b_true_mz = true_mz
		# note: this is a sum over one dimension (just to convert to scalar)
		b_ce = th.sum(-th.exp(b_true_logprob)*log_prob_fn(b_true_mz,b_pred_mzs,b_variances,b_pred_logprobs))
		return b_ce
	ce = th.vmap(vec_ce)(true_batch_idxs,true_logprobs,true_mzs)
	ce = scatter_reduce(
		src=ce,
		index=true_batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size
	)
	return ce


def sparse_cross_entropy_seq(true_mzs,true_logprobs,true_batch_idxs,pred_mzs,pred_logprobs,pred_batch_idxs,pred_oos_logprobs,log_prob_fn=mog_log_prob,tolerance=1e-3,relative=False,oos_tolerance_multiple=1):
	"""compute sparse cross entropy in squence

	Args:
		true_mzs (_type_): groundtruth mzs
		true_logprobs (_type_): groundtruth log prob of each mz aka intesnites
		true_batch_idxs (_type_): spec idx signed to each groundtruth
		pred_mzs (_type_): predicted mzs
		pred_logprobs (_type_): predicted log prob of each mz aka intesnites
		pred_batch_idxs (_type_): pec idx signed to each prediction
		log_prob_fn (_type_, optional): _description_. Defaults to mog_log_prob.
		tolerance (_type_, optional): _description_. Defaults to 1e-3.
		relative (bool, optional): _description_. Defaults to False.

	Returns:
		_type_: _description_
	"""

	mask_value = LOG_ZERO
	batch_size = th.max(true_batch_idxs)+1
	ces = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
	true_oos_logprobs = th.zeros([batch_size],dtype=true_logprobs.dtype,device=true_logprobs.device)
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
		if oos_tolerance_multiple == -1:
			b_true_match_mask = th.ones_like(b_true_mzs,dtype=th.bool)
		else:
			assert oos_tolerance_multiple > 0, oos_tolerance_multiple
			b_match_mask = calculate_match_mzs(
				b_true_mzs,
				b_pred_mzs,
				tolerance=oos_tolerance_multiple*tolerance,
				relative=relative
			)
			b_true_match_mask = th.any(b_match_mask,dim=1)
		if not th.any(b_true_match_mask):
			print("> Warning: Everything is OOS!")
		# batch discrete Cross Entropy Loss 
		# discrete probability distributions of p(ground truth) * log(q(predicted))
		if relative:
			b_variances = (th.clamp(b_pred_mzs,min=200.)*tolerance)**2
		else:
			b_variances = (tolerance**2)*th.ones_like(b_pred_mzs)
		b_is_ce = th.sum(b_true_match_mask.float() * -th.exp(b_true_logprobs) * log_prob_fn(b_true_mzs,b_pred_mzs,b_variances,b_pred_logprobs))
		# calculate oos logprobs
		b_true_oos_logprobs = th.logsumexp(mask_value*b_true_match_mask.float() + b_true_logprobs*(~b_true_match_mask).float(), dim=0)
		b_pred_oos_logprobs = pred_oos_logprobs[batch_idx]
		b_oos_ce = -th.exp(b_true_oos_logprobs) * b_pred_oos_logprobs
		# combine
		b_ce = b_is_ce + b_oos_ce
		ces[batch_idx] = b_ce
		true_oos_logprobs[batch_idx] = b_true_oos_logprobs
	return ces, true_oos_logprobs


def get_sparse_cross_entropy_fn(dist,vectorized,tolerance,relative,oos_tolerance_multiple):

	if dist == "gaussian":
		log_prob_fn = mog_log_prob
	elif dist == "mol_uniform":
		log_prob_fn = momu_log_prob
	else:
		raise NotImplementedError()

	if vectorized:
		raise NotImplementedError
		# sce_fn = lambda *args: sparse_cross_entropy_vec(
		# 	*args,
		# 	log_prob_fn=log_prob_fn,
		# 	tolerance=tolerance,
		# 	relative=relative
		# )
	else:
		sce_fn = lambda *args: sparse_cross_entropy_seq(
			*args,
			log_prob_fn=log_prob_fn,
			tolerance=tolerance,
			relative=relative,
			oos_tolerance_multiple=oos_tolerance_multiple
		)
		
	return sce_fn


def sparse_entropy_fn(logprobs,batch_idxs,oos_logprobs=None,renormalize=False,support_size_delta=0.):

	if renormalize:
		logpartition = scatter_logsumexp(
			logprobs,
			batch_idxs,
			dim=0
		)
		if oos_logprobs is not None:
			logpartition = th.logsumexp(th.stack([logpartition,oos_logprobs],dim=1),dim=1)
			oos_logprobs = oos_logprobs - logpartition
		logprobs = logprobs - th.gather(
			input=logpartition,
			index=batch_idxs,
			dim=0
		)
	probs = th.exp(logprobs)	
	plogp = probs*logprobs
	batch_size = th.max(batch_idxs)+1
	entropy = -scatter_reduce(
		src=plogp,
		index=batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size,
	)
	support_size = scatter_reduce(
		src=th.ones_like(batch_idxs,dtype=logprobs.dtype),
		index=batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size,
	)
	support_size += support_size_delta
	if oos_logprobs is not None:
		assert oos_logprobs.shape[0] == batch_size, (oos_logprobs.shape,batch_size)
		entropy = entropy - th.exp(oos_logprobs)*oos_logprobs
		support_size = support_size + 1.
	max_entropy = safelog(support_size)
	norm_entropy = entropy/th.clamp(max_entropy,min=EPS)
	return entropy, norm_entropy


def sparse_conditional_entropy_fn(
	marginal_logprobs,
	marginal_batch_idxs,
	conditional_logprobs,
	conditional_idxs,
	conditional_batch_idxs,
	conditional_support_size_delta=0.
):

	batch_size = th.max(conditional_batch_idxs)+1
	marginal_support_size = marginal_logprobs.shape[0]
	marginal_probs = th.exp(marginal_logprobs)
	conditional_plogp = th.exp(conditional_logprobs)*conditional_logprobs
	conditional_entropy = -scatter_reduce(
		conditional_plogp,
		index=conditional_idxs,
		reduce="sum",
		dim=0,
		dim_size=marginal_support_size
	)
	entropy = scatter_reduce(
		marginal_probs * conditional_entropy,
		index=marginal_batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size
	)
	conditional_support_size = scatter_reduce(
		th.ones_like(conditional_logprobs,dtype=marginal_logprobs.dtype),
		index=conditional_idxs,
		reduce="sum",
		dim=0,
		dim_size=marginal_support_size
	)
	conditional_support_size += conditional_support_size_delta
	max_entropy = scatter_reduce(
		marginal_probs * th.clamp(safelog(conditional_support_size),min=0.),
		index=marginal_batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=batch_size
	)
	if th.any(max_entropy < 0.):
		import pdb; pdb.set_trace()
	norm_entropy = entropy/th.clamp(max_entropy,min=EPS)
	return entropy, norm_entropy

def get_edge_loss_fn(edge_loss_fn_type,constant):

	if edge_loss_fn_type == "quadratic":
		edge_loss_fn = lambda x: constant*x**2
	elif edge_loss_fn_type == "linear":
		edge_loss_fn = lambda x: constant*th.abs(x)
	elif edge_loss_fn_type == "exponential":
		edge_loss_fn = lambda x: constant*th.exp(th.abs(x))
	else:
		raise NotImplementedError
	return edge_loss_fn


if __name__ == "__main__":

	th.manual_seed(420)

	# Create a mixture of 1D Gaussians with 2 components
	means = th.tensor([0.0,1.0])
	variances = th.tensor([[1.0,0.5]])
	log_weights = th.log(th.tensor([0.6,0.4]))

	samples = 3*th.rand(10)
	print(samples)

	log_probs = mog_log_prob(samples,means,variances,log_weights)
	print(th.exp(log_probs))
