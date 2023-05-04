import torch as th
import torch.nn.functional as F
import torch_scatter as th_s


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

def sparse_cross_entropy(true_mzs,true_logprobs,true_batch_idxs,pred_mzs,pred_logprobs,pred_batch_idxs,batch_size,variance=1e-3,vectorized=False):
	""" this is slow, but fully vectorizing takes a lot of memory """

	assert len(true_mzs) == len(true_logprobs) == len(true_batch_idxs)
	assert len(pred_mzs) == len(pred_logprobs) == len(pred_batch_idxs)

	def seq_ce(batch_idx,mask_value=-1e9):
		b_pred_mask = (pred_batch_idxs==batch_idx).float()
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_logprobs = pred_logprobs*b_pred_mask + (1-b_pred_mask)*mask_value
		b_pred_mzs = pred_mzs
		b_variances = variance*th.ones_like(b_pred_mzs)
		b_true_logprobs = true_logprobs[b_true_mask]
		b_true_mzs = true_mzs[b_true_mask]
		b_ce = th.sum(-th.exp(b_true_logprobs)*mog_log_prob(b_true_mzs,b_pred_mzs,b_variances,b_pred_logprobs))
		return b_ce

	def par_ce(true_batch_idx,true_logprob,true_mz,mask_value=-1e9):
		b_pred_mask = (pred_batch_idxs==true_batch_idx).float()
		b_pred_logprobs = pred_logprobs*b_pred_mask + mask_value*(1.-b_pred_mask)
		b_pred_mzs = pred_mzs
		b_variances = variance*th.ones_like(b_pred_mzs)
		b_true_logprob = true_logprob
		b_true_mz = true_mz
		# note: this is a sum over one dimension (just to convert to scalar)
		b_ce = th.sum(-th.exp(b_true_logprob)*mog_log_prob(b_true_mz,b_pred_mzs,b_variances,b_pred_logprobs))
		return b_ce

	if vectorized:
		ce = th.vmap(par_ce)(true_batch_idxs,true_logprobs,true_mzs)
		ce = th_s.scatter_sum(
			ce,
			true_batch_idxs,
			dim=0,
		)
		ce = th.mean(ce)
	else:
		ce = 0.
		for b in range(batch_size):
			ce = ce + seq_ce(b)
		ce = ce / batch_size
	return ce

def sparse_cross_entropy_vec(true_mzs,true_logprobs,true_batch_idxs,pred_mzs,pred_logprobs,pred_batch_idxs,batch_size,variance):

	def vec_ce(true_batch_idx,true_logprob,true_mz,mask_value=-1e9):
		b_pred_mask = (pred_batch_idxs==true_batch_idx).float()
		b_pred_logprobs = pred_logprobs*b_pred_mask + mask_value*(1.-b_pred_mask)
		b_pred_mzs = pred_mzs
		b_variances = variance*th.ones_like(b_pred_mzs)
		b_true_logprob = true_logprob
		b_true_mz = true_mz
		# note: this is a sum over one dimension (just to convert to scalar)
		b_ce = th.sum(-th.exp(b_true_logprob)*mog_log_prob(b_true_mz,b_pred_mzs,b_variances,b_pred_logprobs))
		return b_ce
	
	ce = th.vmap(vec_ce)(true_batch_idxs,true_logprobs,true_mzs)
	ce = th_s.scatter_sum(
		ce,
		true_batch_idxs,
		dim=0,
	)
	ce = th.mean(ce)
	return ce

def sparse_cross_entropy_seq(true_mzs,true_logprobs,true_batch_idxs,pred_mzs,pred_logprobs,pred_batch_idxs,batch_size,variance):
	""" 
	this is just for torch.compile
	"""

	ce = 0.
	mask_value = -1e9
	for batch_idx in range(batch_size):
		b_pred_mask = (pred_batch_idxs==batch_idx).float()
		b_true_mask = (true_batch_idxs==batch_idx)
		b_pred_logprobs = pred_logprobs*b_pred_mask + (1-b_pred_mask)*mask_value
		b_pred_mzs = pred_mzs
		b_variances = variance*th.ones_like(b_pred_mzs)
		b_true_logprobs = true_logprobs[b_true_mask]
		b_true_mzs = true_mzs[b_true_mask]
		b_ce = th.sum(-th.exp(b_true_logprobs)*mog_log_prob(b_true_mzs,b_pred_mzs,b_variances,b_pred_logprobs))
		# b_mog_log_prob = -0.5*(b_true_mzs.reshape(-1,1)-b_pred_mzs.reshape(1,-1))**2/b_variances.reshape(1,-1) \
		# 	- 0.5*th.log(2*th.pi*b_variances.reshape(1,-1))
		# b_mog_log_prob = th.logsumexp(b_mog_log_prob + b_pred_logprobs.reshape(1,-1),dim=1)
		# b_ce = th.sum(-th.exp(b_true_logprobs)*b_mog_log_prob)
		ce = ce + b_ce
	ce = ce / batch_size
	return ce

def get_sparse_cross_entropy_fn(vectorized):

	if vectorized:
		return sparse_cross_entropy_vec
	else:
		return sparse_cross_entropy_seq

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
