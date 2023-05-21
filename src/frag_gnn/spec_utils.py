import numpy as np
import torch as th

from frag_gnn.utils.misc_utils import scatter_normalize, scatter_reduce


def bin_func(mzs,ints,mz_max,mz_bin_res,ints_thresh,return_index):

	# import pdb; pdb.set_trace()
	mzs = np.array(mzs,dtype=np.float32)
	bins = np.arange(mz_bin_res,mz_max+mz_bin_res,step=mz_bin_res).astype(np.float32)
	bin_idx = np.searchsorted(bins,mzs,side="right")
	if return_index:
		return bin_idx.tolist()
	else:
		ints = np.array(ints,dtype=np.float32)
		bin_spec = np.zeros([len(bins)],dtype=np.float32)
		for i in range(len(mzs)):
			if bin_idx[i] < len(bin_spec) and ints[i] >= ints_thresh:
				bin_spec[bin_idx[i]] = max(bin_spec[bin_idx[i]],ints[i])
		if np.all(bin_spec == 0.):
			print("> warning: bin_spec is all zeros!")
			bin_spec[-1] = 1.
		return bin_spec

def merge_sparse_specs(*peakses):
	"""
	this will result in peaks that are really close in mass (<5ppm)
	they are probably the same peak, but our model can handle this type of ambiguity
	for now, let's keep them unmerged
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
			# total_intensity += intensity
	# for mz in merged_peaks.keys():
	# 	merged_peaks[mz] /= total_intensity
	merged_peaks = sorted(list(merged_peaks.items()), key=lambda x: x[0])
	return merged_peaks

def calculate_spectrum_entropy(log_ints,batch_idxs):

	k = th.max(batch_idxs)+1
	log_norm_ints = scatter_normalize(log_ints,batch_idxs,dim=0)
	entropy = -scatter_reduce(
		src=th.exp(log_norm_ints)*log_norm_ints,
		index=batch_idxs,
		reduce="sum",
		dim=0,
		dim_size=k
	)
	return entropy

def calculate_topk_weighted_recall(true_mzs,true_logits,true_batch_idxs,pred_mzs,pred_logits,pred_batch_idxs,tolerance=5e-3,k=10):
	"""
	topk_weighted_recall: weighted fraction of top-k true peaks that are matched by at least one top-k true peak
	topk_frac: fraction of true intensity that is explained by at least one top-k true peaks
	topk_frac is an upper bound on topk_weighted_recall
	Note: this function could be parallelized with vmap
	"""

	batch_size = th.max(true_batch_idxs)+1
	topk_weighted_recall = th.zeros([batch_size],device=true_mzs.device)
	topk_frac = th.zeros([batch_size],device=true_mzs.device)
	true_ints = th.exp(scatter_normalize(true_logits,true_batch_idxs,dim=0))
	pred_ints = th.exp(scatter_normalize(pred_logits,pred_batch_idxs,dim=0))
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
		b_diff_topk_mzs = th.abs(b_true_topk_mzs.reshape(-1,1)-b_pred_topk_mzs.reshape(1,-1))
		b_true_topk_match = th.any(b_diff_topk_mzs<tolerance,dim=1)
		b_true_ints_topk = true_ints[b_true_mask][b_true_argtopk]
		b_true_ints_topk_sum = th.sum(b_true_ints_topk)
		b_true_ints_topk_match_sum = th.sum(b_true_ints_topk[b_true_topk_match])
		b_topk_weighted_recall = b_true_ints_topk_match_sum / b_true_ints_topk_sum
		topk_weighted_recall[b] = topk_weighted_recall[b] + b_topk_weighted_recall
		b_true_ints_sum = th.sum(true_ints[b_true_mask])
		b_topk_frac = b_true_ints_topk_sum / b_true_ints_sum
		topk_frac[b] = topk_frac[b] + b_topk_frac
	return topk_weighted_recall, topk_frac

def calculate_recalls(true_mzs,true_logits,true_batch_idxs,pred_mzs,pred_logits,pred_batch_idxs,tolerance=5e-3):
	"""
	Note: this function could be parallelized with vmap
	"""

	batch_size = th.max(true_batch_idxs)+1
	recall = th.zeros([batch_size],device=true_mzs.device)
	weighted_recall = th.zeros([batch_size],device=true_mzs.device)
	true_ints = th.exp(scatter_normalize(true_logits,true_batch_idxs,dim=0))
	pred_ints = th.exp(scatter_normalize(pred_logits,pred_batch_idxs,dim=0))
	for b in range(batch_size):
		b_true_mask = (true_batch_idxs==b)
		b_pred_mask = (pred_batch_idxs==b)
		b_true_mzs = true_mzs[b_true_mask]
		b_pred_mzs = pred_mzs[b_pred_mask]
		b_true_ints = true_ints[b_true_mask]
		b_pred_ints = pred_ints[b_pred_mask]
		b_diff_mzs = th.abs(b_true_mzs.reshape(-1,1)-b_pred_mzs.reshape(1,-1))
		b_true_match = th.any(b_diff_mzs<tolerance,dim=1)
		b_recall = th.sum(b_true_match)/b_true_mzs.shape[0]
		recall[b] = recall[b] + b_recall
		b_weighted_recall = th.sum(b_true_ints[b_true_match])/th.sum(b_true_ints)
		weighted_recall[b] = weighted_recall[b] + b_weighted_recall
	return recall, weighted_recall
