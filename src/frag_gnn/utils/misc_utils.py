import time
from functools import wraps
from distutils.util import strtobool
from contextlib import contextmanager
import numpy as np
import torch as th
import os
import joblib
import tqdm
import pandas as pd
import threading
import torch_geometric as pyg

TQDM_DISABLE = False
PPM = 1/1000000
EPS = 1e-9

def timeit(func):
	# adapted from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
	@wraps(func)
	def timeit_wrapper(*args, **kwargs):
		start_time = time.perf_counter()
		result = func(*args, **kwargs)
		end_time = time.perf_counter()
		total_time = end_time - start_time
		print(f'Function {func.__name__} Took {total_time:.4f} seconds')
		return result
	return timeit_wrapper

def booltype(x):
	return bool(strtobool(x))

def none_or_nan(thing):
	if thing is None:
		return True
	elif isinstance(thing,float) and np.isnan(thing):
		return True
	elif pd.isnull(thing):
		return True
	else:
		return False

@contextmanager
def np_temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)

@contextmanager
def th_temp_seed(seed):
	state = th.get_rng_state()
	th.manual_seed(seed)
	try:
		yield
	finally:
		th.set_rng_state(state)

def flatten_lol(lol):
	return [item for sublist in lol for item in sublist]

def wandb_symlink(run_dir,wandb_symlink_dp,job_id):
	symlink_dst = os.path.join(wandb_symlink_dp,str(job_id))
	symlink_src = os.path.split(os.path.abspath(run_dir))[0]
	if os.path.islink(symlink_dst):
		os.unlink(symlink_dst)
	os.symlink(symlink_src,symlink_dst)

def list_str2float(str_list):
	return [float(str_item) for str_item in str_list]

# https://stackoverflow.com/a/58936697/6937913
@contextmanager
def tqdm_joblib(tqdm_object):
	"""Context manager to patch joblib to report into tqdm progress bar given as argument"""
	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)

		def __call__(self, *args, **kwargs):
			tqdm_object.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		tqdm_object.close()

# some utils for function timeout
# adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call

class TimeoutException(Exception): pass

# @contextmanager
# def time_limit(seconds):
# 	if seconds is None:
# 		yield
# 	def signal_handler(signum, frame):
# 		raise TimeoutException(f"Timed out! ({seconds} seconds)")
# 	signal.signal(signal.SIGALRM, signal_handler)
# 	signal.alarm(seconds)
# 	try:
# 		yield
# 	finally:
# 		signal.alarm(0)

def timeout_func(func, args=None, kwargs=None, timeout=30, default=None):
	"""This function will spawn a thread and run the given function
	using the args, kwargs and return the given default value if the
	timeout is exceeded.
	http://stackoverflow.com/questions/492519/timeout-on-a-python-function-call
	"""
	class InterruptableThread(threading.Thread):
		def __init__(self):
			threading.Thread.__init__(self)
			self.result = default
			self.exc_info = (None, None, None)
		def run(self):
			try:
				self.result = func(*(args or ()), **(kwargs or {}))
			except Exception as err:
				self.exc_info = sys.exc_info()
		def suicide(self):
			raise TimeoutException(
				"{0} timeout (taking more than {1} sec)".format(func.__name__, timeout)
			)
	it = InterruptableThread()
	it.start()
	it.join(timeout)
	if it.exc_info[0] is not None:
		a, b, c = it.exc_info
		raise Exception(a, b, c)  # communicate that to caller
	if it.is_alive():
		it.suicide()
		raise RuntimeError
	else:
		return it.result

def my_tqdm(*args,**kwargs):
	return tqdm.tqdm(*args,**kwargs,disable=TQDM_DISABLE)

def get_tensor_memory_usage(tensor):
	return tensor.nelement()*tensor.element_size()

def get_pyg_memory_usage(pyg_graph):

	return pyg.profile.get_data_size(pyg_graph)

def scatter_masked_softmax(logits,mask,subset_idxs,mask_logprob=np.log(1e-4),dim=-1,log=True):
	# calculate appropriate mask value
	with th.no_grad():
		c = scatter_masked_logsumexp(logits,mask,subset_idxs,dim=dim)
		lm = th.gather(
			input=c,
			index=subset_idxs,
			dim=dim
		)
		mask_value = mask_logprob + lm
	# apply mask
	masked_logits = mask*logits + (1-mask)*mask_value
	# normalize
	masked_logits = scatter_logsoftmax(masked_logits,subset_idxs,dim)
	if not log:
		# exponentiate
		return th.exp(masked_logits)
	else:
		return masked_logits

def scatter_masked_logsumexp(logits,mask,subset_idxs,mask_value=-1e9,dim=-1):

	# apply mask
	masked_logits = mask*logits + (1-mask)*mask_value
	# normalize
	masked_logsumexp = scatter_logsumexp(masked_logits,subset_idxs,dim=dim)
	return masked_logsumexp

def scatter_logsumexp(logits,subset_idxs,dim=-1,eps=EPS):

	k = th.max(subset_idxs)+1
	sm = scatter_reduce(
		src=logits,
		index=subset_idxs,
		reduce="amax",
		dim=dim,
		dim_size=k
	)
	lm = th.gather(
		input=sm,
		index=subset_idxs,
		dim=dim
	)
	logits = logits - lm
	se = scatter_reduce(
		src=th.exp(logits),
		index=subset_idxs,
		reduce="sum",
		dim=dim,
		dim_size=k
	)
	return sm + th.log(se + eps)

def scatter_logsoftmax(logits,subset_idxs,dim=-1):
	# calculate normalizing constant
	c = scatter_logsumexp(logits,subset_idxs,dim=dim)
	# apply normalizing constant
	logits = logits - c[subset_idxs]
	return logits

def scatter_l1normalize(vals,subset_idxs,dim=-1):
	# calculate normalizing constant
	c = scatter_reduce(
		src=vals,
		index=subset_idxs,
		reduce="sum",
		dim=dim,
		dim_size=th.max(subset_idxs)+1
	)
	# apply normalizing constant
	vals = vals/c[subset_idxs]
	return vals

def scatter_reduce(src,index,reduce,dim,dim_size):

	result_shape = src.shape[:dim] + (dim_size,) + src.shape[dim+1:]
	results = th.zeros(result_shape,dtype=src.dtype,device=src.device)
	results.scatter_reduce_(
		dim=dim,
		index=index,
		src=src,
		reduce=reduce
	)
	return results

def safelog(x,eps=EPS):
	return th.log(x+eps)

