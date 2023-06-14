import torch as th
import torch.nn.functional as F
import math

from frag_gnn.utils.misc_utils import scatter_reduce


def mog_log_prob(samples,means,variances,log_weights):
	
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
	this is just for torch.compile
	"""

	mask_value = -1e9
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
	ce = th.mean(ce)
	return ce


def sparse_cross_entropy_seq(true_mzs,true_logprobs,true_batch_idxs,pred_mzs,pred_logprobs,pred_batch_idxs,log_prob_fn=mog_log_prob,tolerance=1e-3,relative=False):
	""" 
	this is just for torch.compile
	"""

	ce = 0.
	mask_value = -1e9
	batch_size = th.max(true_batch_idxs)+1
	for batch_idx in range(batch_size):
		b_pred_mask = (pred_batch_idxs==batch_idx).float()
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_logprobs = pred_logprobs*b_pred_mask + mask_value*(1-b_pred_mask)
		b_pred_mzs = pred_mzs
		if relative:
			b_variances = (th.clamp(b_pred_mzs,min=200.)*tolerance)**2
		else:
			b_variances = (tolerance**2)*th.ones_like(b_pred_mzs)
		b_true_logprobs = true_logprobs[b_true_mask]
		b_true_mzs = true_mzs[b_true_mask]
		b_ce = th.sum(-th.exp(b_true_logprobs)*log_prob_fn(b_true_mzs,b_pred_mzs,b_variances,b_pred_logprobs))
		ce = ce + b_ce
	ce = ce / batch_size
	return ce


def get_sparse_cross_entropy_fn(dist,vectorized,tolerance,relative):

	if dist == "gaussian":
		log_prob_fn = mog_log_prob	
	elif dist == "mol_uniform":
		log_prob_fn = momu_log_prob
	else:
		raise NotImplementedError()
	
	if vectorized:
		sce_fn = lambda *args: sparse_cross_entropy_vec(
			*args,
			log_prob_fn=log_prob_fn,
			tolerance=tolerance,
			relative=relative
		)
	else:
		sce_fn = lambda *args: sparse_cross_entropy_seq(
			*args,
			log_prob_fn=log_prob_fn,
			tolerance=tolerance,
			relative=relative
		)

	return sce_fn


def sparse_entropy_fn(logprobs,batch_idxs):

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
	return th.mean(entropy)


def get_sparse_entropy_fn():

	return sparse_entropy_fn


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
