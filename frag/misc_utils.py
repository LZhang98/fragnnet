import time
from functools import wraps
from distutils.util import strtobool
from contextlib import contextmanager
import numpy as np
import torch as th


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