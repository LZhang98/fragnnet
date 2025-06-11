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
from fragnnet.pl_model import BinnedPL

class NeimsPL(BinnedPL):

	def _setup_model(self):

		self.model = NeimsModel(
			mol_fingerprint_morgan=self.hparams.mol_params["fingerprint_morgan"],
			mol_fingerprint_rdkit=self.hparams.mol_params["fingerprint_rdkit"],
			mol_fingerprint_maccs=self.hparams.mol_params["fingerprint_maccs"],
			mlp_hidden_size=self.hparams.mlp_hidden_size,
			mlp_dropout=self.hparams.mlp_dropout,
			mlp_num_layers=self.hparams.mlp_num_layers,
			mlp_use_residuals=self.hparams.mlp_use_residuals,
			mz_max=self.hparams.mz_max,
			mz_bin_res=self.hparams.mz_bin_res,
			ff_prec_mz_offset=self.hparams.ff_prec_mz_offset,
			ff_bidirectional=self.hparams.ff_bidirectional,
			ff_output_map_size=self.hparams.ff_output_map_size,
			ff_output_activation=self.hparams.ff_output_activation,
			int_embedder=self.hparams.int_embedder,
			ce_insert_type=self.hparams.ce_insert_type,
			ce_insert_location=self.hparams.ce_insert_location,
			ce_insert_merge=self.hparams.ce_insert_merge,
			ce_insert_size=self.hparams.ce_insert_size,
   			ce_mean=self.hparams.ce_mean,
			ce_std=self.hparams.ce_std,
			ce_max=self.hparams.ce_max,
			prec_insert_location=self.hparams.prec_insert_location,
			prec_insert_size=self.hparams.prec_insert_size,
			prec_types=self.hparams.spec_params["prec_types"],
			inst_insert_location=self.hparams.inst_insert_location,
			inst_insert_size=self.hparams.inst_insert_size,
			inst_types=self.hparams.spec_params["inst_types"],
			log_min=self.hparams.log_min
		)
		
		self._check_ce_params()

		# compile
		if self.hparams.compile:
			th_dynamo.reset()
			self.dynamo_prof = th_dynamo.utils.CompileProfiler()
			self.model = th.compile(self.model,backend=self.dynamo_prof,dynamic=True)