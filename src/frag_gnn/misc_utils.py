import time
from functools import wraps
from distutils.util import strtobool
from contextlib import contextmanager
import numpy as np
import torch as th
import wandb
import os
import joblib
import tqdm
import pandas as pd
import signal


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

@contextmanager
def time_limit(seconds):
	def signal_handler(signum, frame):
		raise TimeoutException(f"Timed out! ({seconds} seconds)")
	signal.signal(signal.SIGALRM, signal_handler)
	signal.alarm(seconds)
	try:
		yield
	finally:
		signal.alarm(0)

