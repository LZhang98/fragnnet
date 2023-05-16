import numpy as np


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

	merged_peaks = {}
	for peaks in peakses:
		for mz, int in peaks:
			if mz in merged_peaks:
				merged_peaks[mz] += int
			else:
				merged_peaks = int
	merged_peaks = sorted(list(merged_peaks.items()), key=lambda x: x[0])
	return merged_peaks
