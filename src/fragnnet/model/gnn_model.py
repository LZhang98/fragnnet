import torch as th
import torch.nn as nn

from fragnnet.utils.feat_utils import get_mol_feats_sizes
from fragnnet.utils.nn_utils import *
from fragnnet.model.model_base import CEModel, PrecModel, InstModel

class GNNModel(nn.Module, CEModel, PrecModel, InstModel):

	def __init__(
		self,
		mol_node_feats: list[str],
		mol_edge_feats: list[str],
		mol_pe_embed_k: int,
		mol_hidden_size: int,
		mol_num_layers: int,
		mol_gnn_type: str,
		mol_normalization: str,
		mol_dropout: float,
		mol_pool_type: str,
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ff_output_map_size: int,
		ff_output_activation: str,
		int_embedder: str,
		ce_insert_type: str,
		ce_insert_location: str,
		ce_insert_merge: bool,
		ce_insert_size: int,
		ce_max: float,
		ce_mean: float,
		ce_std: float,
		prec_insert_location: str,
		prec_insert_size: int,
		prec_types: list[str],
		inst_insert_location: str,
		inst_insert_size: int,
		inst_types: list[str],
		log_min: float
	):
		# nn.Module init
		super().__init__()
		# collision energy
		self._ce_init(
			int_embedder=int_embedder,
			ce_insert_location=ce_insert_location,
			ce_insert_type=ce_insert_type,
			ce_insert_merge=ce_insert_merge,
			ce_insert_size=ce_insert_size,
			ce_max=ce_max,
			ce_mean=ce_mean,
			ce_std=ce_std
		)
		# precursor
		self._prec_init(
			prec_insert_location=prec_insert_location,
			prec_insert_size=prec_insert_size,
			prec_num_types=len(prec_types)
		)
		# instrument
		self._inst_init(
			inst_insert_location=inst_insert_location,
			inst_insert_size=inst_insert_size,
			inst_num_types=len(inst_types))

		# calculate node/edge feats sizes
		self.mol_node_feats = mol_node_feats
		self.mol_edge_feats = mol_edge_feats
		self.mol_pe_embed_k = mol_pe_embed_k
		self._compute_mol_feats_sizes()

		# setup mol gnn
		self.mol_node_feats_size += self.ce_mol_input_dim + self.prec_mol_input_dim + self.inst_mol_input_dim
		mol_kwargs = {
			"node_feats_size": self.mol_node_feats_size,
			"edge_feats_size": self.mol_edge_feats_size,
			"hidden_size": mol_hidden_size,
			"num_layers": mol_num_layers,
			"gnn_type": mol_gnn_type,
			"dropout": mol_dropout,
			"normalization": mol_normalization,
		}
		# Mol GNN
		self.mol_embedder = GNN(**mol_kwargs)
		self.mol_pool_type = mol_pool_type
		self.mol_pool = build_pool_module(mol_pool_type,mol_hidden_size)

		# MLP input = GNN output
		self.mlp_input_dim = mol_hidden_size
		# metadata
		self.mlp_input_dim += self.ce_mlp_input_dim + self.prec_mlp_input_dim + self.inst_mlp_input_dim

		self.ffn = SpecFFN(
			input_size=self.mlp_input_dim,
			hidden_size=mlp_hidden_size,
			mz_max=mz_max,
			mz_bin_res=mz_bin_res,
			num_layers=mlp_num_layers,
			dropout=mlp_dropout,
			use_residuals=mlp_use_residuals,
			bidirectional=ff_bidirectional,
			prec_mz_offset=ff_prec_mz_offset,
			output_map_size=ff_output_map_size,
			output_activation=ff_output_activation,
			log_min=log_min
		)

	def forward(
		self, 
		mol_pyg: pyg.data.Data,
		spec_prec_mz: th.Tensor,
		spec_nce: th.Tensor = None,
		spec_nce_batch_idxs: th.Tensor = None,
		spec_prec_type: th.Tensor = None,
		spec_inst_type: th.Tensor = None,
		**kwargs
	):
		# mol features
		# mol_x: mol level node feature matrix
		# mol_edge_index: mol graph connectivity in COO format with shape [2, num_edges]
		# edge_attr: mol graph edge feature matrix with shape [num_edges, num_edge_features]
		# batch: sample idx repsect to current batch
		mol_x, mol_edge_index, mol_edge_attr, mol_batch = mol_pyg.x, mol_pyg.edge_index, mol_pyg.edge_attr, mol_pyg.batch

		# int_dtype = mol_edge_index.dtype
		batch_size = mol_batch[-1]+1

		# metadata embedders
		# get ce value
		ce = spec_nce
		ce_batch_idxs = spec_nce_batch_idxs
		ce_embed = self.embed_ce(ce, ce_batch_idxs, batch_size)
		# get prec value
		prec_embed = self.embed_prec(spec_prec_type)
		# get inst value
		inst_embed = self.embed_inst(spec_inst_type)		

		# metadata embeddings at the node feature level
		if self.ce_insert_location == "mol":
			mol_ce_embed = th.repeat_interleave(ce_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_ce_embed],dim=1)
		if self.prec_insert_location == "mol":
			mol_prec_embed = th.repeat_interleave(prec_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_prec_embed],dim=1)
		if self.inst_insert_location == "mol":
			mol_inst_embed = th.repeat_interleave(inst_embed,th.unique(mol_batch,return_counts=True)[1],dim=0)
			mol_x = th.cat([mol_x,mol_inst_embed],dim=1)
		
		# get per-atom embeddings
		mol_embed_gnn = self.mol_embedder(
			mol_x,
			mol_batch,
			mol_edge_index,
			mol_edge_attr
		)
		mol_embed_gnn_pool = self.mol_pool(mol_embed_gnn,mol_batch)
		ffn_input = mol_embed_gnn_pool

		if self.ce_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,ce_embed],dim=1)
		if self.prec_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,prec_embed],dim=1)
		if self.inst_insert_location == "mlp":
			ffn_input = th.cat([ffn_input,inst_embed],dim=1)

		# apply ffn
		pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(ffn_input,spec_prec_mz)
		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"pred_specs": pred_specs
		}
		return out_d
		
	def _compute_mol_feats_sizes(self):
		""" method compute mol feature size
			these features don't rely on any model parameters
		"""
		self.mol_node_feats_size, self.mol_edge_feats_size = get_mol_feats_sizes(
			self.mol_node_feats, 
			self.mol_edge_feats, 
			self.mol_pe_embed_k
		)

	def _ce_location_check(self):

		assert self.ce_insert_location in ["mlp","mol","none"], f"ce_insert_location={self.ce_insert_location} not supported"

	def _prec_location_check(self):
		
		assert self.prec_insert_location in ["mlp","mol","none"], f"prec_insert_location={self.prec_insert_location} not supported"

	def _inst_location_check(self):

		assert self.inst_insert_location in ["mlp","mol","none"], f"prec_insert_location={self.inst_insert_location} not supported"