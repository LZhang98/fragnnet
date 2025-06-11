import torch as th
import numpy as np
import logging
import inspect

# make cc happy
try:
	import lightning.pytorch as pl
	from lightning.utilities.seed import _collect_rng_states, _set_rng_states
except ModuleNotFoundError:
	import pytorch_lightning as pl
	from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states
	
import torch._dynamo as th_dynamo

from fragnnet.utils.nn_utils import build_lr_scheduler
from fragnnet.model import FragGNNModel, NeimsModel, PrecursorModel, GNNModel
from fragnnet.model.loss import get_sparse_cross_entropy_fn, sparse_entropy_fn, sparse_conditional_entropy_fn, get_edge_loss_fn, sparse_cosine_distance_hungarian, sparse_cosine_distance_binned
from fragnnet.utils.spec_utils import *
from fragnnet.utils.misc_utils import safelog, flatten_lol, th_temp_seed, scatter_logsumexp, to_cpu, scatter_l1normalize, LOG_ZERO
from fragnnet.utils.plot_utils import plot_spectra_sparse
from fragnnet.pl_model import SpectrumPL

class BinnedPL(SpectrumPL):

	def _setup_loss_fn(self):
		
		# binned cosine distance
		def cos_dist_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs):
			return sparse_cosine_distance_binned(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs,
				log_distance=(self.hparams.loss_type == "log_cosine_distance")
			)
		
		assert self.hparams.loss_type == "cosine_distance", self.hparams.loss_type
		assert self.hparams.sparse_cosine_similarity
		self.binned_loss = True

		def _loss_fn(
			true_mzs,
			true_logprobs,
			true_batch_idxs,
			pred_mzs,
			pred_logprobs,
			pred_batch_idxs,
			**kwargs):
			
			spec_cd = cos_dist_fn(
				true_mzs,
				true_logprobs,
				true_batch_idxs,
				pred_mzs,
				pred_logprobs,
				pred_batch_idxs
			)
			loss = spec_cd
			loss_d = {
				"loss": loss,
				"spec_cd": spec_cd
			}
			return loss_d

		self.loss_fn = _loss_fn
		# flag metrics for tracking
		loss_names = [
			"loss",
			"spec_cd"
		]
		self.metric_names.update(loss_names)