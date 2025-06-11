import torch as th
import torch.nn as nn
import numpy as np
from pprint import pprint
from pyteomics.mass import Composition

from fragnnet.utils.frag_utils import get_node_feats, get_edge_feats, th_long_to_mask
from fragnnet.utils.misc_utils import scatter_logsumexp, scatter_logsoftmax, scatter_reduce, scatter_masked_softmax, scatter_masked_logsumexp
from fragnnet.utils.feat_utils import get_mol_feats_sizes, get_mol_fp_size
from fragnnet.utils.nn_utils import *
from fragnnet.model.form_embedder import get_embedder
from fragnnet.utils.misc_utils import check_pyg_compile, LOG_ZERO, check_pyg_full_compile
from fragnnet.utils.spec_utils import transform_ce, batched_bin_func
from fragnnet.utils.data_utils import combine_formulae
from fragnnet.utils.formula_utils import PREC_TYPE_TO_FORMULA_DIFF

class CEModel:
	""" class for handling collision engery embedding
	"""
	def _ce_init(
		self,
		int_embedder,
		ce_insert_location: str,
		ce_insert_type: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
		ce_mean: float,
  		ce_std: float,
    	ce_max: float,):

		# ce stuff
		assert ce_insert_type in ["id","lin","embed","bin"]
		assert ce_insert_location in ["none","mol","frag","mlp"]
		self.ce_insert_type = ce_insert_type
		self.ce_insert_location = ce_insert_location
		self.ce_insert_merge = ce_insert_merge
		self.ce_insert_size = ce_insert_size
		self.int_embedder = int_embedder
		self.ce_max = ce_max
		self.ce_mean = ce_mean
		self.ce_std = ce_std
		self._ce_location_check()
		self._setup_ce()

	def _ce_location_check(self):

		raise NotImplementedError

	def _setup_ce(self):

		# embedding type
		if self.ce_insert_type == "id":
			def ce_transform(ce):
				ce = transform_ce(ce, self.ce_mean, self.ce_std)
				ce = ce.reshape(-1,1)
				ce = th.repeat_interleave(ce, self.ce_insert_size, dim=1)
				return ce
			ce_embedder = nn.Identity()
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "lin":
			def ce_transform(ce):
				ce = transform_ce(ce, self.ce_mean, self.ce_std)
				ce = ce.reshape(-1,1)
				return ce
			ce_embedder = nn.Linear(1,self.ce_insert_size)
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "embed":
			def ce_transform(ce):
				ce = th.clamp(ce, min=0, max=int(self.ce_max)-1)
				ce = th.round(ce, decimals=0).long()
				ce = ce.reshape(-1,1)
				return ce
			embedder = get_embedder(self.int_embedder, max_count_int=int(self.ce_max))
			ce_embedder = nn.Sequential(
				embedder,
				nn.Linear(embedder.num_dim,self.ce_insert_size)
			)	
			ce_input_dim = self.ce_insert_size
		elif self.ce_insert_type == "bin":
			def ce_transform(ce):
				ce = th.clamp(ce, min=0, max=int(self.ce_max)-10)
				ce = th.round(ce, decimals=-1).long() // 10
				ce = F.one_hot(ce, num_classes=int(self.ce_max)//10).float()
				return ce
			ce_embedder = nn.Linear(int(self.ce_max)//10,self.ce_insert_size)
			ce_input_dim = self.ce_insert_size
		# location
		if self.ce_insert_location == "mol":
			ce_mol_input_dim = ce_input_dim
			ce_mlp_input_dim = 0
		elif self.ce_insert_location == "mlp":
			ce_mol_input_dim = 0
			ce_mlp_input_dim = ce_input_dim
		else:
			assert self.ce_insert_location == "none"
			ce_mol_input_dim = 0
			ce_mlp_input_dim = 0
		self.ce_transform = ce_transform
		self.ce_embedder = ce_embedder
		self.ce_mol_input_dim = ce_mol_input_dim
		self.ce_mlp_input_dim = ce_mlp_input_dim

	def embed_ce(self, ce, ce_batch_idxs, batch_size):

		if self.ce_insert_location != "none":
			ce_embed = self.ce_transform(ce)
			ce_embed = self.ce_embedder(ce_embed)
			# possibly merge the embeddings
			if self.ce_insert_merge:
				ce_embed = scatter_reduce(
					src=ce_embed,
					index=ce_batch_idxs.unsqueeze(1).expand_as(ce_embed),
					reduce="mean",
					dim_size=batch_size,
					include_self=False
				)
		else:
			ce_embed = None
		return ce_embed

class PrecModel:

	def _prec_init(
		self,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_num_types: int):

		self.prec_insert_location = prec_insert_location
		self.prec_embedder = nn.Embedding(prec_num_types+1, prec_insert_size)	
		prec_dim = prec_insert_size

		self._prec_location_check()

		if self.prec_insert_location == "mol":
			prec_mol_input_dim = prec_dim
			prec_mlp_input_dim = 0
		elif self.prec_insert_location == "mlp":
			prec_mol_input_dim = 0
			prec_mlp_input_dim = prec_dim
		else:
			assert self.prec_insert_location == "none"
			prec_mol_input_dim = 0
			prec_mlp_input_dim = 0
		self.prec_mol_input_dim = prec_mol_input_dim
		self.prec_mlp_input_dim = prec_mlp_input_dim

	def _prec_location_check(self):

		raise NotImplementedError

	def embed_prec(self, prec_type):

		if self.prec_insert_location != "none":
			prec_embed = self.prec_embedder(prec_type)
		else:
			prec_embed = None
		return prec_embed

class InstModel:

	def _inst_init(
		self,
		inst_insert_location: str,
		inst_insert_size: int,
		inst_num_types: int):

		self.inst_insert_location = inst_insert_location
		self.inst_embedder = nn.Embedding(inst_num_types+1, inst_insert_size)	
		inst_dim = inst_insert_size

		self._inst_location_check()

		if self.inst_insert_location == "mol":
			inst_mol_input_dim = inst_dim
			inst_mlp_input_dim = 0
		elif self.inst_insert_location == "mlp":
			inst_mol_input_dim = 0
			inst_mlp_input_dim = inst_dim
		else:
			assert self.inst_insert_location == "none"
			inst_mol_input_dim = 0
			inst_mlp_input_dim = 0
		self.inst_mol_input_dim = inst_mol_input_dim
		self.inst_mlp_input_dim = inst_mlp_input_dim

	def _inst_location_check(self):

		raise NotImplementedError

	def embed_inst(self, inst_type):

		if self.inst_insert_location != "none":
			inst_embed = self.inst_embedder(inst_type)
		else:
			inst_embed = None
		return inst_embed
