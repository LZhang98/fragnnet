import torch as th
import torch.nn as nn

from frag_gnn.massformer.nn_utils import GFv2Embedder
from frag_gnn.utils.nn_utils import *
from frag_gnn.utils.misc_utils import scatter_reduce
from frag_gnn.model import get_ce_mlp_stuff


class MassFormerModel(nn.Module):

	def __init__(
		self, 
		mlp_hidden_size: int,
		mlp_dropout: float,
		mlp_num_layers: int,
		mlp_use_residuals: bool,
		mz_max: int,
		mz_bin_res: float,
		ff_prec_mz_offset: int,
		ff_bidirectional: bool,
		ce_insert_mlp: bool,
		ce_insert_mlp_lin: bool,
		m_ce_insert_mlp_embed: bool,
		m_ce_insert_mlp_bin: bool,
		int_embedder: str,    
	):
		
		super().__init__()

		self.gf = GFv2Embedder()
		self.mlp_input_dim = self.gf.get_embed_dim()

		# ce stuff
		self.ce_insert_mlp = ce_insert_mlp
		self.ce_insert_mlp_lin = ce_insert_mlp_lin
		self.m_ce_insert_mlp_embed = m_ce_insert_mlp_embed
		self.m_ce_insert_mlp_bin = m_ce_insert_mlp_bin
		ce_mlp_stuff = get_ce_mlp_stuff(
			ce_insert_mlp=self.ce_insert_mlp,
			ce_insert_mlp_lin=self.ce_insert_mlp_lin,
			m_ce_insert_mlp_embed=self.m_ce_insert_mlp_embed,
			m_ce_insert_mlp_bin=self.m_ce_insert_mlp_bin,
			int_embedder=int_embedder,
			ce_dim=12)
		if self.ce_insert_mlp or self.ce_insert_mlp_lin:
			self.ce_transform = ce_mlp_stuff[0]
			self.ce_embedder = ce_mlp_stuff[1]
			self.ce_mlp_input_dim = ce_mlp_stuff[2]
			self.mlp_input_dim += self.ce_mlp_input_dim
		if self.m_ce_insert_mlp_embed or self.m_ce_insert_mlp_bin:
			self.m_ce_transform = ce_mlp_stuff[0]
			self.m_ce_embedder = ce_mlp_stuff[1]
			self.m_ce_mlp_input_dim = ce_mlp_stuff[2]
			self.mlp_input_dim += self.m_ce_mlp_input_dim

		self.ffn = SpecFFN(
			input_size=self.mlp_input_dim,
			hidden_size=mlp_hidden_size,
			mz_max=mz_max,
			mz_bin_res=mz_bin_res,
			num_layers=mlp_num_layers,
			dropout=mlp_dropout,
			prec_mz_offset=ff_prec_mz_offset,
			bidirectional=ff_bidirectional,
			use_residuals=mlp_use_residuals
		)

	def get_split_params(self):

		nopt_params, pt_params = self.gf.get_split_params()
		pt_params.extend(list(self.ffn.parameters()))
		return nopt_params, pt_params

	def forward(
		self,
		mol_mf_idx: th.Tensor,
		mol_mf_attn_bias: th.Tensor,
		mol_mf_attn_edge_type: th.Tensor,
		mol_mf_spatial_pos: th.Tensor,
		mol_mf_in_degree: th.Tensor,
		mol_mf_out_degree: th.Tensor,
		mol_mf_x: th.Tensor,
		mol_mf_edge_input: th.Tensor,
		mol_mf_y: th.Tensor,
		spec_prec_mz: th.Tensor,
		spec_nce: th.Tensor = None,
		spec_m_nce: th.Tensor = None,
		spec_m_nce_batch_idxs: th.Tensor = None,
		mf_perturb: th.Tensor = None,
		**kwargs):

		gf_d = {
			"idx": mol_mf_idx,
			"attn_bias": mol_mf_attn_bias,
			"attn_edge_type": mol_mf_attn_edge_type,
			"spatial_pos": mol_mf_spatial_pos,
			"in_degree": mol_mf_in_degree,
			"out_degree": mol_mf_out_degree,
			"x": mol_mf_x,
			"edge_input": mol_mf_edge_input,
			"y": mol_mf_y,

		}
		fh = self.gf(gf_d,perturb=mf_perturb)
		batch_size = fh.shape[0]
		# get ce
		ce = spec_nce
		m_ce = spec_m_nce
		m_ce_batch_idxs = spec_m_nce_batch_idxs
		# add CE information
		if self.ce_insert_mlp or self.ce_insert_mlp_lin:
			assert ce is not None
			ce = ce.reshape(-1,1)
			mlp_ce_embed = self.ce_embedder(self.ce_transform(ce))
			fh = th.cat([fh,mlp_ce_embed],dim=1)
		if self.m_ce_insert_mlp_embed or self.m_ce_insert_mlp_bin:
			assert m_ce is not None
			assert m_ce_batch_idxs is not None
			mlp_m_ce_embed = self.m_ce_transform(m_ce)
			mlp_m_ce_embed = self.m_ce_embedder(mlp_m_ce_embed)
			mlp_m_ce_embed = scatter_reduce(
				src=mlp_m_ce_embed,
				index=m_ce_batch_idxs.unsqueeze(1).expand_as(mlp_m_ce_embed),
				reduce="mean",
				dim_size=batch_size,
				include_self=False
			)
			# print(fh.shape, mlp_m_ce_embed.shape)
			fh = th.cat([fh,mlp_m_ce_embed],dim=1)
		# apply ffn
		pred_mzs, pred_logprobs, pred_batch_idxs, pred_specs = self.ffn(fh,spec_prec_mz)
		out_d = {
			"pred_mzs": pred_mzs,
			"pred_logprobs": pred_logprobs,
			"pred_batch_idxs": pred_batch_idxs,
			"pred_specs": pred_specs
		}
		return out_d
